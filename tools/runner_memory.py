import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
import math
from tools import builder_memory as builder
from utils import misc, dist_utils
from utils.common_utils import write_bin_float32, read_bin_float32
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
from utils.metrics_memory import Metrics
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from tqdm import tqdm
import numpy as np
from datasets.shapenet_synset_dict import shapenet_config
from datasets.PCN_synset_dict import config as pcn_config
from utils.graph_construction import get_radius_graph
# from thop import profile
# import torchprofile


def _model_memory_size(config):
    """Return memory size without requiring category metadata."""
    memory_size = config.model.get("memory_size", None)
    if memory_size is None:
        memory_size = config.model.get("class_num", None)
    if memory_size is None:
        raise KeyError("model.memory_size must be specified")
    return int(memory_size)


def _evaluation_class_count(config):
    """Dataset metadata used by metrics, never by model training."""
    if config.model.get("label_free_training", False):
        return 0
    class_count = config.model.get("evaluation_class_count", None)
    if class_count is None:
        class_count = config.model.get("class_num", None)
    if class_count is None:
        raise KeyError(
            "model.evaluation_class_count must be specified for metrics"
        )
    return int(class_count)


def _metric_names(config):
    """Hide classification accuracy when no classification task exists."""
    names = Metrics.names()
    if config.model.get("label_free_training", False):
        names = [name for name in names if name != "ClsAcc"]
    return names


def _filter_metric_values(config, values):
    """Keep metric values aligned with ``_metric_names``."""
    if not config.model.get("label_free_training", False):
        return values
    return [
        value
        for name, value in zip(Metrics.names(), values)
        if name != "ClsAcc"
    ]


def _metric_summary(config, values):
    """Build checkpoint metrics without inventing a label-free ClsAcc."""
    return Metrics(
        config.consider_metric,
        dict(zip(_metric_names(config), values)),
    )


def _taxonomy_values(taxonomy_ids):
    if isinstance(taxonomy_ids, torch.Tensor):
        return taxonomy_ids.detach().cpu().reshape(-1).tolist()
    if isinstance(taxonomy_ids, np.ndarray):
        return taxonomy_ids.reshape(-1).tolist()
    if isinstance(taxonomy_ids, (list, tuple)):
        values = []
        for taxonomy_id in taxonomy_ids:
            if isinstance(taxonomy_id, torch.Tensor):
                values.extend(taxonomy_id.detach().cpu().reshape(-1).tolist())
            elif isinstance(taxonomy_id, np.ndarray):
                values.extend(taxonomy_id.reshape(-1).tolist())
            else:
                values.append(taxonomy_id)
        return values
    return [taxonomy_ids]

def _projected_remap_key(data_path):
    return "shapenet34_remap" if "34" in os.path.basename(data_path) else "remap"

def _projected_taxonomy_indices(taxonomy_ids, data_path):
    remap_key = _projected_remap_key(data_path)
    indices = []
    for taxonomy_id in _taxonomy_values(taxonomy_ids):
        if isinstance(taxonomy_id, bytes):
            taxonomy_id = taxonomy_id.decode()
        if isinstance(taxonomy_id, (str, np.str_)):
            indices.append(int(shapenet_config[remap_key][str(taxonomy_id)]))
        else:
            indices.append(int(taxonomy_id))
    return indices

def _projected_taxonomy_tensor(taxonomy_ids, data_path):
    return torch.tensor(_projected_taxonomy_indices(taxonomy_ids, data_path), dtype=torch.long).cuda()

def _projected_taxonomy_synsets(taxonomy_ids, data_path):
    remap_key = _projected_remap_key(data_path)
    reid2id = {v: k for k, v in shapenet_config[remap_key].items()}
    return [reid2id[str(i)] for i in _projected_taxonomy_indices(taxonomy_ids, data_path)]


def _attach_training_taxonomy(input_dict, taxonomy_ids, config):
    """Keep category annotations outside explicitly label-free models."""
    if not config.model.get("label_free_training", False):
        input_dict["taxonomy_ids"] = taxonomy_ids


def _sanitize_label_free_model_input(input_dict, config):
    """Keep labels and external banks outside a label-free model boundary."""
    if config.model.get("label_free_training", False):
        for key in ("taxonomy_ids", "keys", "values", "classifier"):
            input_dict.pop(key, None)
    return input_dict


def _learned_memory_key_values(base_model):
    """Read the model-owned memory without gathering it through DP output."""
    model = base_model.module if hasattr(base_model, "module") else base_model
    memory_vector = model.memory_vector.detach()
    key_dim = int(model.memory_key_dim)
    return memory_vector[:, :key_dim], memory_vector[:, key_dim:]


def _coarse_points_bnc(points):
    """Normalize a coarse point tensor to the Chamfer [B, N, 3] layout."""
    if points.ndim != 3:
        raise ValueError(f"Expected a 3-D point tensor, got {points.shape}")
    if points.shape[-1] == 3:
        return points
    if points.shape[1] == 3:
        return points.transpose(2, 1).contiguous()
    raise ValueError(f"Cannot infer point layout from {points.shape}")


def _optional_memory_loss(model, method_name, input_dict, strict=False):
    """Call an optional memory loss without masking strict-run failures."""
    method = getattr(model, method_name, None)
    if method is None:
        return None
    if strict:
        return method(input_dict)
    try:
        return method(input_dict)
    except Exception:
        return None

def run_net_memory(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    # build dataset
    (train_sampler, train_dataloader), (_, test_dataloader) = builder.dataset_builder(args, config.dataset.train), \
                                                            builder.dataset_builder(args, config.dataset.val)
    # build model
    base_model = builder.model_builder(config.model)
    if args.use_gpu:
        base_model.to(args.local_rank)

    # from IPython import embed; embed()
    
    # parameter setting
    start_epoch = 0
    best_metrics = None
    metrics = None

    # resume ckpts
    if args.resume:
        start_epoch, best_metrics = builder.resume_model(base_model, args, logger = logger)
        best_metrics = Metrics(config.consider_metric, best_metrics)
    elif args.start_ckpts is not None:
        builder.load_model(base_model, args.start_ckpts, logger = logger)

    # print model info
    print_log('Trainable_parameters:', logger = logger)
    print_log('=' * 25, logger = logger)
    for name, param in base_model.named_parameters():
        if param.requires_grad:
            print_log(name, logger=logger)
    print_log('=' * 25, logger = logger)
    
    print_log('Untrainable_parameters:', logger = logger)
    print_log('=' * 25, logger = logger)
    for name, param in base_model.named_parameters():
        if not param.requires_grad:
            print_log(name, logger=logger)
    print_log('=' * 25, logger = logger)

    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[args.local_rank % torch.cuda.device_count()], find_unused_parameters=args.find_unused_parameters)
        print_log('Using Distributed Data parallel ...' , logger = logger)
    else:
        print_log('Using Data parallel ...' , logger = logger)
        base_model = nn.DataParallel(base_model).cuda()
    label_free_training = bool(
        config.model.get("label_free_training", False)
    )
    # optimizer & scheduler
    optimizer = builder.build_optimizer(base_model, config)
    
    # Criterion
    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()

    if args.resume:
        builder.resume_optimizer(optimizer, args, logger = logger)
    scheduler = builder.build_scheduler(base_model, optimizer, config, last_epoch=start_epoch-1)
    if args.resume:
        builder.resume_scheduler(scheduler, args, logger=logger)

    # if config.model.classifier_model == "pointnet_cls":
    #     classifier = pointnet_cls.get_model(config.model.class_num, normal_channel=config.model.use_normals)
    #     classifier_loss = pointnet_cls.get_loss(gamma=config.model.focal_loss_gamma)
    # elif config.model.classifier_model == "pointnet_msg_cls":
    #     classifier = pointnet_msg_cls.get_model(config.model.class_num, normal_channel=config.model.use_normals)
    #     classifier_loss = pointnet_msg_cls.get_loss(gamma=config.model.focal_loss_gamma)
    # if config.model.get("train_classifier", False) is False:
    #     classifier.requires_grad_(False)
    #     classifier.eval()
    # classifier.load_state_dict(torch.load(config.model.classifier_model_path)['model_state_dict'])
    # classifier.cuda()
    # print_log('Use pretrained model from %s' % config.model.classifier_model_path)

    # Label-free models keep the memory bank in their checkpoint parameters.
    if label_free_training:
        keys, values = _learned_memory_key_values(base_model)
    else:
        memory_size = _model_memory_size(config)
        if args.keys is not None:
            keys = read_bin_float32(args.keys).reshape(memory_size, -1)
            keys = torch.from_numpy(keys).cuda()
        else:
            keys = torch.rand((memory_size, config.model.encoder_config.embed_dim), dtype=torch.float32).cuda()

        if args.values is not None:
            values = read_bin_float32(args.values).reshape(memory_size, -1)
            values = torch.from_numpy(values).cuda()
        else:
            values = torch.rand((memory_size, config.model.decoder_config.embed_dim), dtype=torch.float32).cuda()

    # trainval
    # training
    base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()

        
        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['SparseLoss', 'DenseLoss', "SumLoss", "SimpLoss", "TaskLoss",
                               "LinkLoss", "DenoisedLoss", "RankingLoss", "ConsistentLoss", "TrainClassifierLoss",
                               "OrthogonalLoss", "CompactnessLoss", "SeperationLoss", "MaxInterLoss", "MinIntraLoss",
                               "FeatureClusterLoss", "MemoryFeatureLoss"])

        num_iter = 0

        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)
        for idx, (taxonomy_ids, model_ids, adj, data) in enumerate(tqdm(train_dataloader, desc="Processing", unit="batch")):
            data_time.update(time.time() - batch_start_time)
            # npoints = config.dataset.train._base_.N_POINTS
            dataset_name = config.dataset.train._base_.NAME
            npoints = config.dataset.train._base_.N_POINTS
            if  'PCN' in dataset_name or dataset_name == 'Completion3D' or 'ProjectShapeNet' in dataset_name:
                partial_points = data[0].cuda()
                gt_points = data[1].cuda()
                bs = partial_points.shape[0]
                adj = adj.cuda()
                if not label_free_training:
                    taxonomy_ids = np.squeeze(taxonomy_ids)
                    taxonomy_ids = torch.from_numpy(taxonomy_ids).cuda()
                
                input_dict = {
                    "partial_points": partial_points,
                    "gt_points": gt_points,
                    "adj": adj,
                    "keys": keys,
                    "values": values,
                    "junction_index": torch.arange(224).reshape(-1, 224).cuda().repeat(bs, 1),
                }
                _attach_training_taxonomy(input_dict, taxonomy_ids, config)
                
                if config.dataset.train._base_.CARS:
                    if idx == 0:
                        print_log('padding while KITTI training', logger=logger)
                    # partial_points, gt_points = misc.random_scale(partial_points, gt_points) # specially for KITTI finetune
                    partial_points = misc.random_dropping(partial_points, epoch) # specially for KITTI finetune
                    input_dict["partial_points"] = partial_points

            elif 'V2XSeqSPD' in dataset_name:
                partial_points = data[0].cuda()
                gt_points = data[1].cuda()
                # print(gt_points.shape)
                # print(label.shape)
                # print(label)
                # gt_points = data.cuda()
                # partial_points, _ = misc.seprate_point_cloud(gt_points, npoints, [int(npoints * 1/4) , int(npoints * 3/4)], fixed_points = None)
                # partial_points = partial_points.cuda()
            elif dataset_name == 'ShapeNet' or dataset_name == 'ShapeNetGNN':
                gt = data.cuda()
                partial, _ = misc.seprate_point_cloud(gt, npoints, [int(npoints * 1/4) , int(npoints * 3/4)], fixed_points = None)
                partial = partial.cuda()
                bs = partial.shape[0]
                if config.model.get("do_adj", False):
                    adj = get_radius_graph(partial, r=0.2)
                adj = adj.cuda()
                if not label_free_training:
                    taxonomy_ids = np.squeeze(taxonomy_ids)
                    taxonomy_ids = torch.from_numpy(taxonomy_ids).cuda()
                junction_index = torch.arange(384).reshape(-1, 384).cuda().repeat(bs, 1)
                
                input_dict = {
                    "partial_points": partial,
                    "gt_points": gt,
                    "adj": adj,
                    "keys": keys,
                    "values": values,
                    "junction_index": junction_index,
                }
                _attach_training_taxonomy(input_dict, taxonomy_ids, config)
            
            elif 'Projected_ShapeNet' in dataset_name:
                partial_points = data[0].cuda()
                gt_points = data[1].cuda()
                adj = adj.cuda()
                if not label_free_training:
                    taxonomy_ids = _projected_taxonomy_tensor(
                        taxonomy_ids,
                        config.dataset.train._base_.DATA_PATH,
                    )
                bs = partial_points.shape[0]
                junction_index = torch.arange(384).reshape(-1, 384).cuda().repeat(bs, 1)
                
                input_dict = {
                    "partial_points": partial_points,
                    "gt_points": gt_points,
                    "adj": adj,
                    "keys": keys,
                    "values": values,
                    "junction_index": junction_index,
                }
                _attach_training_taxonomy(input_dict, taxonomy_ids, config)
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            if label_free_training:
                # The learnable bank is a model parameter, not per-sample data.
                input_dict.pop("keys", None)
                input_dict.pop("values", None)

            num_iter += 1
            
            # input_dict["classifier"] = classifier
            input_dict = base_model(
                _sanitize_label_free_model_input(input_dict, config)
            )  # input bnc,output b n c , input -1~1
            if label_free_training:
                keys, values = _learned_memory_key_values(base_model)
            else:
                keys = input_dict["keys"].detach()
                values = input_dict["values"].detach()
            
            if 'SPAC' in type(base_model.module).__name__:
                sparse_loss, dense_loss = base_model.module.get_loss(input_dict)
                sparse_loss = sparse_loss * config.model.get("sparse_loss_param", 1)
                dense_loss = dense_loss * config.model.get("dense_loss_param", 1)
                loss_denoised = torch.tensor(0).cuda()
            elif 'FSCSVD' in type(base_model.module).__name__:
                _loss, losses_list = base_model.module.get_loss(input_dict)
                # Native FSC optimizes all three reconstruction stages
                # (coarse + fine1 + fine2).  Keep fine1 in the main training
                # objective instead of silently dropping it in memory runs.
                sparse_loss = losses_list[0] + losses_list[1]
                dense_loss = losses_list[2]
                sparse_loss = sparse_loss * config.model.get("sparse_loss_param", 1)
                dense_loss = dense_loss * config.model.get("dense_loss_param", 1)
                loss_denoised = torch.tensor(0).cuda()
            elif 'SPAC' not in type(base_model.module).__name__:
                loss_denoised, sparse_loss, dense_loss = base_model.module.get_rebuild_loss(input_dict)
                sparse_loss = sparse_loss * config.model.get("sparse_loss_param", 1)
                dense_loss = dense_loss * config.model.get("dense_loss_param", 1)
            else:
                loss_denoised = torch.tensor(0).cuda()
                sparse_loss = torch.tensor(0).cuda()
                dense_loss = torch.tensor(0).cuda()
            
            strict_label_free_adamemory = (
                label_free_training
                and hasattr(base_model.module, "get_adamemory_losses")
            )
            zero_loss = input_dict["partial_points"].new_zeros(())
            if strict_label_free_adamemory:
                linkprediction_loss = zero_loss
                query_ranking_loss = zero_loss
            else:
                try:
                    linkprediction_loss = base_model.module.get_link_prediction_loss(input_dict)
                except Exception:
                    linkprediction_loss = zero_loss
                try:
                    query_ranking_loss = base_model.module.get_ranking_loss(input_dict)
                except Exception:
                    query_ranking_loss = zero_loss
            if label_free_training:
                class_constraint = input_dict["partial_points"].new_zeros(())
                class_loss = input_dict["partial_points"].new_zeros(())
            else:
                try:
                    class_constraint = base_model.module.get_class_constraint(input_dict)
                    class_constraint = config.model.get("class_constraint_param", 1) * class_constraint
                except:
                    class_constraint = torch.tensor(0).cuda()
                try:
                    class_loss = base_model.module.get_class_loss(input_dict)
                    class_loss = config.model.train_classifier_loss_param * class_loss
                except:
                    class_loss = torch.tensor(0).cuda()
            if strict_label_free_adamemory:
                orthogonal_constriant_loss = zero_loss
            else:
                try:
                    orthogonal_constriant_loss = base_model.module.get_orthogonal_constriant(input_dict)
                    orthogonal_constriant_loss = config.model.get("orthogonal_constriant_loss_param", 1) * orthogonal_constriant_loss
                except Exception:
                    orthogonal_constriant_loss = zero_loss

            if strict_label_free_adamemory:
                adamemory_losses = base_model.module.get_adamemory_losses(
                    input_dict
                )
                cluster_weight = float(
                    config.model.get("cluster_loss_weight", 1.0)
                )
                scatter_weight = float(
                    config.model.get("scatter_loss_weight", 1.0)
                )
                key_sim_loss = (
                    0.5
                    * cluster_weight
                    * adamemory_losses["key_cluster"]
                )
                value_sim_loss = (
                    0.5
                    * cluster_weight
                    * adamemory_losses["value_cluster"]
                )
                compactness_loss = key_sim_loss + value_sim_loss
                sim_seperation_loss = (
                    scatter_weight * adamemory_losses["scatter"]
                )
            else:
                compactness_losses = _optional_memory_loss(
                    base_model.module,
                    "get_compactness_loss",
                    input_dict,
                    strict=config.model.get("strict_memory_loss", False),
                )
                if compactness_losses is not None:
                    key_sim_loss, value_sim_loss = compactness_losses
                    key_sim_loss = config.model.get("key_sim_loss_param", 1) * key_sim_loss
                    value_sim_loss = config.model.get("value_sim_loss_param", 1) * value_sim_loss
                    compactness_loss = key_sim_loss + value_sim_loss
                else:
                    key_sim_loss = zero_loss
                    value_sim_loss = zero_loss
                    compactness_loss = zero_loss
                sim_seperation_loss = _optional_memory_loss(
                    base_model.module,
                    "get_sim_seperation_loss",
                    input_dict,
                    strict=config.model.get("strict_memory_loss", False),
                )
                if sim_seperation_loss is not None:
                    sim_seperation_loss = config.model.get("get_sim_seperation_loss_param", 1) * sim_seperation_loss
                else:
                    sim_seperation_loss = zero_loss

            if strict_label_free_adamemory:
                max_inter_loss = zero_loss
                min_intra_loss = zero_loss
                simplification_loss = zero_loss
            else:
                try:
                    max_inter_loss, min_intra_loss = base_model.module.get_MaxInterMinIntra_loss(input_dict)
                    max_inter_loss = config.model.get("get_MaxInter_loss_param", 1) * max_inter_loss
                    min_intra_loss = config.model.get("get_MinIntra_loss_param", 1) * min_intra_loss
                except Exception:
                    max_inter_loss = zero_loss
                    min_intra_loss = zero_loss
                try:
                    simplification_loss = base_model.module.get_simplification_loss(input_dict)
                    simplification_loss = config.model.alpha * simplification_loss
                except Exception:
                    simplification_loss = zero_loss
            
            # task_loss是使用经过预训练的、冻结的分类器得到的分类损失。
            # 这里就是将task loss设为0，try-except语句是因为后续的例如tensrboard操作会用到task loss
            if label_free_training:
                task_loss = input_dict["partial_points"].new_zeros(())
            else:
                try:
                    # task_loss = classifier_loss(pred, taxonomy_ids.long(), trans_feat)
                    task_loss = config.model.task_loss_param * task_loss
                except:
                    task_loss = torch.tensor(0).cuda()

            # 把所有loss的数量级都对齐到dense loss的数量级
            if (
                not strict_label_free_adamemory
                and config.model.get("align_loss", False)
            ):
                target_magnitude = math.floor(math.log10(dense_loss))
                sparse_loss = sparse_loss * 10**(target_magnitude - math.floor(math.log10(sparse_loss)))
                task_loss = task_loss * 10**(target_magnitude - math.floor(math.log10(task_loss)))
                loss_denoised = loss_denoised * 10**(target_magnitude - math.floor(math.log10(loss_denoised)))
                # linkprediction_loss = linkprediction_loss * 10**(target_magnitude - math.floor(math.log10(linkprediction_loss)))
                # simplification_loss = simplification_loss * 10**(target_magnitude - math.floor(math.log10(simplification_loss)))

            _loss = task_loss + simplification_loss + linkprediction_loss + sparse_loss + \
                dense_loss + loss_denoised + query_ranking_loss + class_constraint + class_loss + orthogonal_constriant_loss + \
                compactness_loss + sim_seperation_loss + max_inter_loss + min_intra_loss
         
            _loss.backward()
            
            if config.model.get("tensorboard_write_grad", False):
                n_itr = epoch * n_batches + idx
                if train_writer is not None and idx % 100 == 0:
                    for name, param in base_model.named_parameters():
                        try:
                            train_writer.add_histogram(name, param.clone().cpu().data.numpy(), n_itr)
                            train_writer.add_histogram(name + "/grad", param.grad.clone().cpu().data.numpy(), n_itr)
                        except AttributeError:
                            print("{} has no grad.".format(name))

            # n_itr = epoch * n_batches + idx
            # if train_writer is not None and idx % 100 == 0:
            #     for name, param in base_model.named_parameters():
            #         if name == "module.base_model.keys":
            #             train_writer.add_histogram(name, param.clone().cpu().data.numpy(), n_itr)
            #             train_writer.add_histogram(name + "/grad", param.grad.clone().cpu().data.numpy(), n_itr)
            #         if name == "module.base_model.values":
            #             train_writer.add_histogram(name, param.clone().cpu().data.numpy(), n_itr)
            #             train_writer.add_histogram(name + "/grad", param.grad.clone().cpu().data.numpy(), n_itr)

            # for name, param in base_model.named_parameters():
            #     if param.grad is None:
            #         print(name)

            # forward
            if num_iter == config.step_per_update:
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), getattr(config, 'grad_norm_clip', 10), norm_type=2)
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()
                if label_free_training:
                    keys, values = _learned_memory_key_values(base_model)

            if args.distributed:
                sparse_loss = dist_utils.reduce_tensor(sparse_loss, args)
                dense_loss = dist_utils.reduce_tensor(dense_loss, args)
                _loss = dist_utils.reduce_tensor(_loss, args)
                task_loss = dist_utils.reduce_tensor(task_loss, args)
                simplification_loss = dist_utils.reduce_tensor(simplification_loss, args)
                linkprediction_loss = dist_utils.reduce_tensor(linkprediction_loss, args)
                loss_denoised = dist_utils.reduce_tensor(loss_denoised, args)
                query_ranking_loss = dist_utils.reduce_tensor(query_ranking_loss, args)
                class_constraint = dist_utils.reduce_tensor(class_constraint, args)
                class_loss = dist_utils.reduce_tensor(class_loss, args)
                orthogonal_constriant_loss = dist_utils.reduce_tensor(orthogonal_constriant_loss, args)
                compactness_loss = dist_utils.reduce_tensor(compactness_loss, args)
                sim_seperation_loss = dist_utils.reduce_tensor(sim_seperation_loss, args)
                max_inter_loss = dist_utils.reduce_tensor(max_inter_loss, args)
                min_intra_loss = dist_utils.reduce_tensor(min_intra_loss, args)
                key_sim_loss = dist_utils.reduce_tensor(key_sim_loss, args)
                value_sim_loss = dist_utils.reduce_tensor(value_sim_loss, args)
                losses.update([sparse_loss.item() * 1000, dense_loss.item() * 1000, _loss.item(),
                               simplification_loss.item(), task_loss.item(), linkprediction_loss.item(),
                               loss_denoised.item(), query_ranking_loss.item(), class_constraint.item(),
                               class_loss.item(), orthogonal_constriant_loss.item(), compactness_loss.item(),
                               sim_seperation_loss.item(), max_inter_loss.item(), min_intra_loss.item(),
                               key_sim_loss.item(), value_sim_loss.item()])
            else:
                losses.update([sparse_loss.item() * 1000, dense_loss.item() * 1000, _loss.item(),
                               simplification_loss.item(), task_loss.item(), linkprediction_loss.item(),
                               loss_denoised.item(), query_ranking_loss.item(), class_constraint.item(),
                               class_loss.item(), orthogonal_constriant_loss.item(), compactness_loss.item(),
                               sim_seperation_loss.item(), max_inter_loss.item(), min_intra_loss.item(),
                               key_sim_loss.item(), value_sim_loss.item()])


            if args.distributed:
                torch.cuda.synchronize()

            n_itr = epoch * n_batches + idx
            # if train_writer is not None:
            #     train_writer.add_scalar('Loss/Batch/Sparse', sparse_loss.item() * 1000, n_itr)
            #     train_writer.add_scalar('Loss/Batch/Dense', dense_loss.item() * 1000, n_itr)
            #     train_writer.add_scalar('Loss/Batch/SumLoss', _loss.item(), n_itr)
            #     train_writer.add_scalar('Loss/Batch/SampleNetLoss', simplification_loss.item(), n_itr)
            #     train_writer.add_scalar('Loss/Batch/TaskLoss', task_loss.item(), n_itr)
            #     train_writer.add_scalar('Loss/Batch/LinkPredLoss', linkprediction_loss.item(), n_itr)
            #     train_writer.add_scalar('Loss/Batch/DenoisedLoss', loss_denoised.item(), n_itr)
            #     train_writer.add_scalar('Loss/Batch/RankingLoss', query_ranking_loss.item(), n_itr)
            #     train_writer.add_scalar('Loss/Batch/ConsistentLoss', class_constraint.item(), n_itr)
            #     train_writer.add_scalar('Loss/Batch/TrainClassifierLoss', class_loss.item(), n_itr)
            #     train_writer.add_scalar('Loss/Batch/OrthogonalLoss', orthogonal_constriant_loss.item(), n_itr)
            #     train_writer.add_scalar('Loss/Batch/CompactnessLoss', compactness_loss.item(), n_itr)
            #     train_writer.add_scalar('Loss/Batch/CosineSimilarSeperationLoss', sim_seperation_loss.item(), n_itr)
            #     train_writer.add_scalar('Loss/Batch/MaxInterLoss', max_inter_loss.item(), n_itr)
            #     train_writer.add_scalar('Loss/Batch/MinIntraLoss', min_intra_loss.item(), n_itr)

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 100 == 0:
                print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s lr = %.6f' %
                            (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                            ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr']), logger = logger)
                if label_free_training:
                    print_log(
                        'MemoryLosses: FeatureCluster = %.6f, MemoryFeature = %.6f, '
                        'MemorySeparation = %.6f, CompactnessTotal = %.6f' % (
                            key_sim_loss.item(), value_sim_loss.item(),
                            sim_seperation_loss.item(), compactness_loss.item()
                        ),
                        logger=logger,
                    )

            if config.scheduler.type == 'GradualWarmup':
                if n_itr < config.scheduler.kwargs_2.total_epoch:
                    scheduler.step()

        if isinstance(scheduler, list):
            for item in scheduler:
                misc.step_scheduler(item, epoch + 1)
        else:
            misc.step_scheduler(scheduler, epoch + 1)
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Sparse*1000', losses.avg(0), epoch)
            train_writer.add_scalar('Loss/Epoch/Dense*1000', losses.avg(1), epoch)
            train_writer.add_scalar('Loss/Epoch/Sparse', losses.avg(0) / 1000, epoch)
            train_writer.add_scalar('Loss/Epoch/Dense', losses.avg(1) / 1000, epoch)
            train_writer.add_scalar('Loss/Epoch/SumLoss', losses.avg(2), epoch)
            train_writer.add_scalar('Loss/Epoch/SimplificationLoss', losses.avg(3), epoch)
            train_writer.add_scalar('Loss/Epoch/TaskLoss', losses.avg(4), epoch)
            train_writer.add_scalar('Loss/Epoch/LinkPredLoss', losses.avg(5), epoch)
            train_writer.add_scalar('Loss/Epoch/DenoisedLoss', losses.avg(6), epoch)
            train_writer.add_scalar('Loss/Epoch/RankingLoss', losses.avg(7), epoch)
            train_writer.add_scalar('Loss/Epoch/ConsistentLoss', losses.avg(8), epoch)
            train_writer.add_scalar('Loss/Epoch/TrainClassifierLoss', losses.avg(9), epoch)
            train_writer.add_scalar('Loss/Epoch/OrthogonalLoss', losses.avg(10), epoch)
            train_writer.add_scalar('Loss/Epoch/CompactnessLoss', losses.avg(11), epoch)
            train_writer.add_scalar('Loss/Epoch/CosineSimilarSeperationLoss', losses.avg(12), epoch)
            train_writer.add_scalar('Loss/Epoch/MaxInterLoss', losses.avg(13), epoch)
            train_writer.add_scalar('Loss/Epoch/MinIntraLoss', losses.avg(14), epoch)
            train_writer.add_scalar('Loss/Epoch/FeatureClusterLoss', losses.avg(15), epoch)
            train_writer.add_scalar('Loss/Epoch/MemoryFeatureLoss', losses.avg(16), epoch)

        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s' %
            (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()]), logger = logger)
        if label_free_training:
            print_log(
                '[Training] MemoryLossesAvg: FeatureCluster = %.6f, '
                'MemoryFeature = %.6f, MemorySeparation = %.6f, '
                'CompactnessTotal = %.6f' % (
                    losses.avg(15), losses.avg(16), losses.avg(12),
                    losses.avg(11),
                ),
                logger=logger,
            )

        # if (epoch == config.max_epoch or epoch % args.val_freq == 0) and (epoch >= config.max_epoch / 4 or epoch >= 20):
        # Determine whether to run validation this epoch.
        # early_stop_epoch: if set in config, forces validation + stop at that epoch.
        early_stop_epoch = int(config.get("early_stop_epoch", 0))
        _run_val_this_epoch = (
            (config.max_epoch - epoch) <= 20
            or epoch % args.val_freq == 0
            or (early_stop_epoch > 0 and epoch == early_stop_epoch)
        )
        if _run_val_this_epoch:
            # Validate the current model
            metrics = validate(base_model, test_dataloader, epoch, ChamferDisL1, ChamferDisL2, val_writer, args, config, None, None,
                               keys, values, logger=logger)

            # Save ckeckpoints
            if  metrics.better_than(best_metrics):
                best_metrics = metrics
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, scheduler=scheduler, logger = logger)
                if not label_free_training:
                    write_bin_float32(keys.cpu().numpy(), "{}/{}.bin".format(args.experiment_path, "keys-best"))
                    write_bin_float32(values.cpu().numpy(), "{}/{}.bin".format(args.experiment_path, "values-best"))
        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, scheduler=scheduler, logger = logger)
        if not label_free_training:
            write_bin_float32(keys.cpu().numpy(), "{}/{}.bin".format(args.experiment_path, "keys-last"))
            write_bin_float32(values.cpu().numpy(), "{}/{}.bin".format(args.experiment_path, "values-last"))
        if (
            epoch % args.val_freq == 0
            or (config.max_epoch - epoch) < 20
            or (early_stop_epoch > 0 and epoch == early_stop_epoch)
        ):
            builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args, scheduler=scheduler, logger = logger)
            if not label_free_training:
                write_bin_float32(keys.cpu().numpy(), "{}/{}.bin".format(args.experiment_path, f'keys-epoch-{epoch:03d}'))
                write_bin_float32(values.cpu().numpy(), "{}/{}.bin".format(args.experiment_path, f'values-epoch-{epoch:03d}'))

        # Early-stop: if early_stop_epoch is configured and we've reached it,
        # terminate training after the above validation and checkpoint saving.
        if early_stop_epoch > 0 and epoch >= early_stop_epoch:
            print_log(
                f'[EarlyStop] early_stop_epoch={early_stop_epoch} reached at epoch {epoch}. '
                f'Stopping training early (max_epoch={config.max_epoch}).',
                logger=logger,
            )
            break

    if train_writer is not None and val_writer is not None:
        train_writer.close()
        val_writer.close()

def validate(base_model, test_dataloader, epoch, ChamferDisL1, ChamferDisL2, val_writer, args, config, classifier, classifier_loss,
             keys, values, logger = None):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode

    test_losses = AverageMeter(['SparseLossL1', 'SparseLossL2', 'DenseLossL1', 'DenseLossL2',
                                ])
    test_metrics = AverageMeter(_metric_names(config))
    category_metrics = dict()
    eval_records = []
    n_samples = len(test_dataloader) # bs is 1

    interval = max(n_samples // 10, 1)

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, adj, data) in enumerate(test_dataloader):
            taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0]
            model_id = model_ids[0]

            npoints = config.dataset.val._base_.N_POINTS
            dataset_name = config.dataset.val._base_.NAME
            if 'PCN' in dataset_name or dataset_name == 'Completion3D' or 'ProjectShapeNet' in dataset_name:
                partial_points = data[0].cuda()
                gt_points = data[1].cuda()
                adj = adj.cuda()
                input_dict = {
                    "partial_points": partial_points,
                    "gt_points": gt_points,
                    "adj": adj,
                    "taxonomy_ids": torch.tensor(taxonomy_ids).cuda(),
                    "keys": keys,
                    "values": values,
                    "junction_index": torch.arange(224).reshape(-1, 224).cuda(),
                }
                taxonomy_ids_remap = np.squeeze(taxonomy_ids)
                taxonomy_ids_remap = torch.from_numpy(taxonomy_ids_remap).unsqueeze(0).cuda()
                # 把taxonomy_ids重新映射回原来的编号
                reid2id = {v:k for k, v in pcn_config["remap"].items()}
                taxonomy_ids = [reid2id[str(i)] for i in taxonomy_ids]
            elif 'V2XSeqSPD' in dataset_name:
                partial_points = data[0].cuda()
                gt_points = data[1].cuda()
                gt_seg = data[2].cuda()
                # gt_points = data.cuda()
                # partial_points, _ = misc.seprate_point_cloud(gt_points, npoints, [int(npoints * 1/4) , int(npoints * 3/4)], fixed_points = None)
                # partial_points = partial_points.cuda()
            elif dataset_name == 'ShapeNet' or dataset_name == 'ShapeNetGNN':
                gt_points = data.cuda()
                partial_points, _ = misc.seprate_point_cloud(gt_points, npoints, [int(npoints * 1/4) , int(npoints * 3/4)], fixed_points = None)
                partial_points = partial_points.cuda()
                if config.model.get("do_adj", False):
                    adj = get_radius_graph(partial_points, r=0.2)
                adj = adj.cuda()
                junction_index = torch.arange(384).reshape(-1, 384).cuda()

                
                input_dict = {
                    "partial_points": partial_points,
                    "gt_points": gt_points,
                    "adj": adj,
                    "taxonomy_ids": taxonomy_ids,
                    "keys": keys,
                    "values": values,
                    "junction_index": junction_index,
                }
                taxonomy_ids_remap = np.squeeze(taxonomy_ids)
                taxonomy_ids_remap = torch.from_numpy(taxonomy_ids_remap).unsqueeze(0).cuda()
                # 把taxonomy_ids重新映射回原来的编号
                remap_key = "shapenet34_remap" if "34" in os.path.basename(config.dataset.val._base_.DATA_PATH) else "remap"
                reid2id = {v:k for k, v in shapenet_config[remap_key].items()}
                taxonomy_ids = [reid2id[str(i)] for i in taxonomy_ids]
            elif 'Projected_ShapeNet' in dataset_name:
                partial_points = data[0].cuda()
                gt_points = data[1].cuda()
                adj = adj.cuda()
                taxonomy_ids_remap = _projected_taxonomy_tensor(taxonomy_ids, config.dataset.val._base_.DATA_PATH)
                taxonomy_ids_tensor = taxonomy_ids_remap
                taxonomy_ids = _projected_taxonomy_synsets(taxonomy_ids, config.dataset.val._base_.DATA_PATH)
                junction_index = torch.arange(384).reshape(-1, 384).cuda()
                input_dict = {
                    "partial_points": partial_points,
                    "gt_points": gt_points,
                    "adj": adj,
                    "taxonomy_ids": taxonomy_ids_tensor,
                    "keys": keys,
                    "values": values,
                    "junction_index": junction_index,
                }
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')
            
            # eval_classifier = classifier
            # eval_classifier.requires_grad_(False)
            # eval_classifier.eval()
            # input_dict["classifier"] = eval_classifier
            input_dict = base_model(
                _sanitize_label_free_model_input(input_dict, config)
            )  # input bnc,output b n c , input -1~1
            simp_pc = input_dict["coarse_points"]
            dense_points = input_dict["rebuild_points"]
            simp_pc = _coarse_points_bnc(simp_pc)
            
            # try的部分是利用同时训练的classifier进行评估，except部分是利用预训练的冻结的classifier评估
            # try:
            #     pred_choice = input_dict["pred"].data.max(1)[1]
            #     pred = input_dict["pred"]
            #     trans_feat = input_dict["trans_feat"]
            # except:
            #     pred, trans_feat, _ = classifier(simp_pc) # input bxcxn
            #     pred_choice = pred.data.max(1)[1]
            #     simp_pc = simp_pc.transpose(2,1)  # transpose to bnc for simplification loss
            
            # sparse_loss, dense_loss = base_model.module.get_rebuild_loss(input_dict)
            # linkprediction_loss = base_model.module.get_link_prediction_loss(input_dict)
            # simplification_loss = base_model.module.get_simplification_loss(input_dict)
            # simplification_loss = config.model.alpha * simplification_loss
            # try:
            #     query_ranking_loss = base_model.module.get_ranking_loss(input_dict)
            # except:
            #     query_ranking_loss = torch.tensor(0).cuda()

            # try:
            #     # task_loss = classifier_loss(pred, taxonomy_ids_remap.long(), trans_feat)
            #     task_loss = config.model.task_loss_param * task_loss
            # except:
            #     task_loss = torch.tensor(0).cuda()
            # # 把所有loss的数量级都对齐到dense loss的数量级
            # if config.model.get("align_loss", False):
            #     target_magnitude = math.floor(math.log10(dense_loss))
            #     sparse_loss = sparse_loss * 10**(target_magnitude - math.floor(math.log10(sparse_loss)))
            #     task_loss = task_loss * 10**(target_magnitude - math.floor(math.log10(task_loss)))
            #     # linkprediction_loss = linkprediction_loss * 10**(target_magnitude - math.floor(math.log10(linkprediction_loss)))
            #     # simplification_loss = simplification_loss * 10**(target_magnitude - math.floor(math.log10(simplification_loss)))

            # _loss = task_loss + simplification_loss + linkprediction_loss + sparse_loss + dense_loss + query_ranking_loss
            
            # ret = base_model(partial_points)
            # coarse_points = ret[0]
            # dense_points = ret[1]

            sparse_loss_l1 =  ChamferDisL1(simp_pc, gt_points)
            sparse_loss_l2 =  ChamferDisL2(simp_pc, gt_points)
            dense_loss_l1 =  ChamferDisL1(dense_points, gt_points)
            dense_loss_l2 =  ChamferDisL2(dense_points, gt_points)

            loss_values = [
                sparse_loss_l1.item() * 1000,
                sparse_loss_l2.item() * 1000,
                dense_loss_l1.item() * 1000,
                dense_loss_l2.item() * 1000,
            ]
            test_losses.update(loss_values)

            # dense_points_all = dist_utils.gather_tensor(dense_points, args)
            # gt_all = dist_utils.gather_tensor(gt_points, args)

            # _metrics = Metrics.get(dense_points_all, gt_all)
            _metrics = Metrics.get(dense_points, gt_points,
                                   taxonomy_ids_remap, taxonomy_ids_remap,
                                   num_class=_evaluation_class_count(config),
                                   include_cls=not config.model.get("label_free_training", False))
            _metrics = _filter_metric_values(config, _metrics)
            _metrics = [_metric.item() for _metric in _metrics]

            for _taxonomy_id in taxonomy_ids:
                _taxonomy_id = str(_taxonomy_id)
                if _taxonomy_id not in category_metrics:
                    category_metrics[_taxonomy_id] = AverageMeter(
                        _metric_names(config)
                    )
                category_metrics[_taxonomy_id].update(_metrics)
                eval_records.append({
                    'sample_id': f'{_taxonomy_id}:{model_id}',
                    'taxonomy_id': _taxonomy_id,
                    'losses': loss_values,
                    'metrics': _metrics,
                })


            # if val_writer is not None and idx % 200 == 0:
            #     input_pc = partial_points.squeeze().detach().cpu().numpy()
            #     input_pc = misc.get_ptcloud_img(input_pc)
            #     val_writer.add_image('Model%02d/Input'% idx , input_pc, epoch, dataformats='HWC')

            #     sparse = simp_pc.squeeze().cpu().numpy()
            #     sparse_img = misc.get_ptcloud_img(sparse)
            #     val_writer.add_image('Model%02d/Sparse' % idx, sparse_img, epoch, dataformats='HWC')

            #     dense = dense_points.squeeze().cpu().numpy()
            #     dense_img = misc.get_ptcloud_img(dense)
            #     val_writer.add_image('Model%02d/Dense' % idx, dense_img, epoch, dataformats='HWC')
                
            #     gt_ptcloud = gt_points.squeeze().cpu().numpy()
            #     gt_ptcloud_img = misc.get_ptcloud_img(gt_ptcloud)
            #     val_writer.add_image('Model%02d/DenseGT' % idx, gt_ptcloud_img, epoch, dataformats='HWC')
        
            if (idx+1) % interval == 0:
                print_log('Test[%d/%d] Taxonomy = %s Sample = %s Losses = %s Metrics = %s' %
                            (idx + 1, n_samples, taxonomy_id, model_id, ['%.4f' % l for l in test_losses.val()], 
                            ['%.4f' % m for m in _metrics]), logger=logger)
        if args.distributed:
            eval_records = dist_utils.gather_eval_records(eval_records, args)

        test_losses.reset()
        test_metrics.reset()
        category_metrics = dict()
        for record in eval_records:
            test_losses.update(record['losses'])
            taxonomy_id = record['taxonomy_id']
            if taxonomy_id not in category_metrics:
                category_metrics[taxonomy_id] = AverageMeter(
                    _metric_names(config)
                )
            category_metrics[taxonomy_id].update(record['metrics'])

        for _,v in category_metrics.items():
            test_metrics.update(v.avg())
        print_log('[Validation] EPOCH: %d  Metrics = %s' % (epoch, ['%.4f' % m for m in test_metrics.avg()]), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()
     
    # Print testing results
    shapenet_dict = json.load(open(config.dataset.val._base_.TEST_JSON, 'r'))
    print_log('============================ TEST RESULTS ============================',logger=logger)
    msg = ''
    msg += 'Taxonomy\t'
    msg += '#Sample\t'
    for metric in test_metrics.items:
        msg += metric + '\t'
    msg += '#ModelName\t'
    print_log(msg, logger=logger)


    for taxonomy_id in category_metrics:
        msg = ''
        msg += (taxonomy_id + '\t')
        msg += (str(category_metrics[taxonomy_id].count(0)) + '\t')
        for value in category_metrics[taxonomy_id].avg():
            msg += '%.3f \t' % value
        msg += shapenet_dict[taxonomy_id] + '\t'
        print_log(msg, logger=logger)

    msg = ''
    msg += 'Overall\t\t'
    for value in test_metrics.avg():
        msg += '%.3f \t' % value
    print_log(msg, logger=logger)

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Loss/Epoch/Sparse*1000', test_losses.avg(0), epoch)
        val_writer.add_scalar('Loss/Epoch/Dense*1000', test_losses.avg(2), epoch)
        val_writer.add_scalar('Loss/Epoch/Sparse', test_losses.avg(0) / 1000, epoch)
        val_writer.add_scalar('Loss/Epoch/Dense', test_losses.avg(2) / 1000, epoch)
        for i, metric in enumerate(test_metrics.items):
            val_writer.add_scalar('Metric/%s' % metric, test_metrics.avg(i), epoch)

    return _metric_summary(config, test_metrics.avg())


crop_ratio = {
    'easy': 1/4,
    'median' :1/2,
    'hard':3/4
}

def test_net_memory(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger = logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)
 
    base_model = builder.model_builder(config.model)
    # load checkpoints
    builder.load_model(base_model, args.ckpts, logger = logger)
    if args.use_gpu:
        base_model.to(args.local_rank)

    #  DDP    
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[args.local_rank % torch.cuda.device_count()], find_unused_parameters=args.find_unused_parameters)
        print_log('Using Distributed Data parallel ...' , logger = logger)
    else:
        # print_log('Using Data parallel ...' , logger = logger)
        # base_model = nn.DataParallel(base_model).cuda()
        print_log('Do Not Use Data parallel ...' , logger = logger)

    # Criterion
    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()

    test(base_model, test_dataloader, ChamferDisL1, ChamferDisL2, args, config, logger=logger)

def test(base_model, test_dataloader, ChamferDisL1, ChamferDisL2, args, config, logger = None):

    base_model.eval()  # set model to eval mode
    # if config.model.classifier_model == "pointnet_cls":
    #     classifier = pointnet_cls.get_model(config.model.class_num, normal_channel=config.model.use_normals)
    #     classifier_loss = pointnet_cls.get_loss(gamma=config.model.focal_loss_gamma)
    # elif config.model.classifier_model == "pointnet_msg_cls":
    #     classifier = pointnet_msg_cls.get_model(config.model.class_num, normal_channel=config.model.use_normals)
    #     classifier_loss = pointnet_msg_cls.get_loss(gamma=config.model.focal_loss_gamma)
    # classifier.requires_grad_(False)
    # classifier.eval()
    # classifier.load_state_dict(torch.load(config.model.classifier_model_path)['model_state_dict'])
    # classifier.cuda()
    # print_log('Use pretrained model from %s' % config.model.classifier_model_path)

    # New label-free models own their learnable memory in the checkpoint.
    if config.model.get("label_free_training", False):
        keys, values = _learned_memory_key_values(base_model)
    else:
        assert args.keys is not None
        assert args.values is not None
        if args.keys == "1" or args.values == "1":
            keys = torch.tensor(0).cuda()  # 用于补位的
            values = torch.tensor(0).cuda()  # 用于补位的
        else:
            memory_size = _model_memory_size(config)
            keys = read_bin_float32(args.keys).reshape(memory_size, -1)
            keys = torch.from_numpy(keys).cuda()
            values = read_bin_float32(args.values).reshape(memory_size, -1)
            values = torch.from_numpy(values).cuda()

    test_losses = AverageMeter(['SparseLossL1', 'SparseLossL2', 'DenseLossL1', 'DenseLossL2',
                                ])
    test_metrics = AverageMeter(_metric_names(config))
    category_metrics = dict()
    eval_records = []
    n_samples = len(test_dataloader) # bs is 1
    shapenet_dict = json.load(open(config.dataset.val._base_.TEST_JSON, 'r'))

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, adj, data) in tqdm(enumerate(test_dataloader)):
            # taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0]
            taxonomy_id = taxonomy_ids
            model_id = model_ids

            npoints = config.dataset.test._base_.N_POINTS
            dataset_name = config.dataset.test._base_.NAME
            
            # 获取dataset的data_root字段
            data_root = test_dataloader.dataset.data_root if hasattr(test_dataloader.dataset, 'data_root') else None
            
            if  'V2XSeqSPD' in dataset_name or 'ProjectShapeNet' in dataset_name:
                partial_points = data[0].cuda()
                gt_points = data[1].cuda()

                ret = base_model(partial_points)
                coarse_points = ret[0]
                dense_points = ret[1]
                
                # # 进行模型输出的保存
                # print("Saving model output...")
                # save_dir = "./{}/vis".format(args.experiment_path)
                # os.makedirs(save_dir, exist_ok=True)
                # # coarse_points.cpu().numpy().astype(np.float32).tofile("{}/{}_coarse.bin".format(save_dir, model_id))
                # dense_points.cpu().numpy().astype(np.float32).tofile("{}/{}_dense.bin".format(save_dir, model_id))

                sparse_loss_l1 =  ChamferDisL1(coarse_points, gt_points)
                sparse_loss_l2 =  ChamferDisL2(coarse_points, gt_points)
                dense_loss_l1 =  ChamferDisL1(dense_points, gt_points)
                dense_loss_l2 =  ChamferDisL2(dense_points, gt_points)

                loss_values = [
                    sparse_loss_l1.item() * 1000,
                    sparse_loss_l2.item() * 1000,
                    dense_loss_l1.item() * 1000,
                    dense_loss_l2.item() * 1000,
                ]
                test_losses.update(loss_values)

                # _metrics = Metrics.get(dense_points ,gt_points)
                # test_metrics.update(_metrics)
                _metrics = Metrics.get(
                    dense_points,
                    gt_points,
                    require_emd=False,
                    include_cls=False,
                )
                _metrics = _filter_metric_values(config, _metrics)
                _metrics = [_metric.item() for _metric in _metrics]
                
                for _taxonomy_id in taxonomy_ids:
                    _taxonomy_id = str(_taxonomy_id)
                    if _taxonomy_id not in category_metrics:
                        category_metrics[_taxonomy_id] = AverageMeter(
                            _metric_names(config)
                        )
                    category_metrics[_taxonomy_id].update(_metrics)
                    eval_records.append({
                        'sample_id': f'{_taxonomy_id}:{model_id}',
                        'taxonomy_id': _taxonomy_id,
                        'losses': loss_values,
                        'metrics': _metrics,
                    })
            
            elif dataset_name == 'ShapeNet' or dataset_name == 'ShapeNetGNN':
                gt_points = data.cuda()
                choice = [torch.Tensor([1,1,1]),torch.Tensor([1,1,-1]),torch.Tensor([1,-1,1]),torch.Tensor([-1,1,1]),
                            torch.Tensor([-1,-1,1]),torch.Tensor([-1,1,-1]), torch.Tensor([1,-1,-1]),torch.Tensor([-1,-1,-1])]
                num_crop = int(npoints * crop_ratio[args.mode])
                taxonomy_ids_remap = np.squeeze(taxonomy_ids)
                taxonomy_ids_input = taxonomy_ids
                taxonomy_ids_remap = torch.from_numpy(taxonomy_ids_remap).unsqueeze(0).cuda()
                # 把taxonomy_ids重新映射回原来的编号
                if "34" in data_root.split("/")[-1]:
                    reid2id = {v:k for k, v in shapenet_config["shapenet34_remap"].items()}
                else:
                    reid2id = {v:k for k, v in shapenet_config["remap"].items()}
                taxonomy_ids = [reid2id[str(i)] for i in taxonomy_ids]
                for view_idx, item in enumerate(choice):
                    partial_points, _ = misc.seprate_point_cloud(gt_points, npoints, num_crop, fixed_points = item)
                    partial_points = partial_points.cuda()
                    partial_points = misc.fps(partial_points, 2048)
                    adj = get_radius_graph(partial_points, r=0.2)
                    adj = adj.cuda()
                    input_dict = {
                        "partial_points": partial_points,
                        "gt_points": gt_points,
                        "adj": adj,
                        "taxonomy_ids": taxonomy_ids_input,
                        "model_id": model_id,
                        "keys": keys,
                        "values": values,
                        "junction_index": torch.arange(384).reshape(-1, 384).cuda(),
                    }

                    input_dict = base_model(
                        _sanitize_label_free_model_input(input_dict, config)
                    )  # input bnc,output b n c , input -1~1
                    simp_pc = input_dict["coarse_points"]
                    dense_points = input_dict["rebuild_points"]
                    simp_pc = simp_pc.transpose(2,1)

                    # pred_choice = input_dict["pred"].data.max(1)[1]
                    simp_pc = simp_pc.transpose(2,1)  # transpose to bnc for simplification loss

                    sparse_loss_l1 =  ChamferDisL1(simp_pc, gt_points)
                    sparse_loss_l2 =  ChamferDisL2(simp_pc, gt_points)
                    dense_loss_l1 =  ChamferDisL1(dense_points, gt_points)
                    dense_loss_l2 =  ChamferDisL2(dense_points, gt_points)

                    loss_values = [
                        sparse_loss_l1.item() * 1000,
                        sparse_loss_l2.item() * 1000,
                        dense_loss_l1.item() * 1000,
                        dense_loss_l2.item() * 1000,
                    ]
                    test_losses.update(loss_values)

                    _metrics = Metrics.get(dense_points, gt_points,
                                        taxonomy_ids_remap, taxonomy_ids_remap,
                                        num_class=_evaluation_class_count(config),
                                        include_cls=not config.model.get("label_free_training", False))
                    _metrics = _filter_metric_values(config, _metrics)
                    _metrics = [_metric.item() for _metric in _metrics]

                    for _taxonomy_id in taxonomy_ids:
                        _taxonomy_id = str(_taxonomy_id)
                        if _taxonomy_id not in category_metrics:
                            category_metrics[_taxonomy_id] = AverageMeter(
                                _metric_names(config)
                            )
                        category_metrics[_taxonomy_id].update(_metrics)
                        eval_records.append({
                            'sample_id': f'{_taxonomy_id}:{model_id}:{view_idx}',
                            'taxonomy_id': _taxonomy_id,
                            'losses': loss_values,
                            'metrics': _metrics,
                        })

            elif 'Projected_ShapeNet' in dataset_name:
                partial_points = data[0].cuda()
                gt_points = data[1].cuda()
                adj = adj.cuda()
                taxonomy_ids_remap = _projected_taxonomy_tensor(taxonomy_ids, data_root)
                taxonomy_ids_tensor = taxonomy_ids_remap
                taxonomy_ids = _projected_taxonomy_synsets(taxonomy_ids, data_root)
                input_dict = {
                    "partial_points": partial_points,
                    "gt_points": gt_points,
                    "adj": adj,
                    "taxonomy_ids": taxonomy_ids_tensor,
                    "model_id": model_id,
                    "keys": keys,
                    "values": values,
                    "junction_index": torch.arange(384).reshape(-1, 384).cuda(),
                }
                
                # 用于可视化筛选sample
                # file_list = set(os.listdir(r"/home/csstuer/yyx/GS-Net-yyx/vis_save/Ada_memory_intersection/gt"))
                # if "{}_{}.bin".format(taxonomy_ids[0], model_ids[0]) not in file_list:
                #     continue
                
                # ==== 计算模型复杂度 ======
                # flops, params = profile(base_model, inputs=(input_dict))
                # # flops = torchprofile.profile_macs(base_model, input_dict)
                # print(f"FLOPS: {flops / 1e9:.2f} GFLOPS")  # 转换为GFLOPS
                # print(f"Parameters: {params / 1e6:.2f} M")
                # ========================
                
                input_dict = base_model(
                    _sanitize_label_free_model_input(input_dict, config)
                )  # input bnc,output b n c , input -1~1
                # sampled_coarse = input_dict["sampled_coarse"]
                simp_pc = input_dict["coarse_points"]
                dense_points = input_dict["rebuild_points"]
                simp_pc = simp_pc.transpose(2,1)
                # sampled_coarse = sampled_coarse.transpose(2,1)
                
                
                
                # pred, trans_feat, _ = classifier(simp_pc) # input bxcxn
                # pred_choice = pred.data.max(1)[1]
                simp_pc = simp_pc.transpose(2,1)  # transpose to bnc for simplification loss
                
                # sparse_loss, dense_loss = base_model.module.get_rebuild_loss(input_dict)
                # linkprediction_loss = base_model.module.get_link_prediction_loss(input_dict)
                # simplification_loss = base_model.module.get_simplification_loss(input_dict)
                # simplification_loss = config.model.alpha * simplification_loss

                # try:
                #     # task_loss = classifier_loss(pred, taxonomy_ids_remap.long(), trans_feat)
                #     task_loss = config.model.task_loss_param * task_loss
                # except:
                #     task_loss = torch.tensor(0).cuda()
                # _loss = task_loss + simplification_loss + linkprediction_loss + sparse_loss + dense_loss
                
                # 进行模型输出的保存
                # if shapenet_dict[taxonomy_ids[0]] != "airplane":
                #     continue
                # print("Saving model output...")
                # input_point_save_dir = "./{}/vis/input_point_cloud".format(args.experiment_path)
                # dense_point_save_dir = "./{}/vis/dense_point_cloud".format(args.experiment_path)
                # coarse_point_save_dir = "./{}/vis/coarse_point_cloud".format(args.experiment_path)
                # sampled_point_save_dir = "./{}/vis/sampled_point_cloud".format(args.experiment_path)
                # input_fps_point_save_dir = "./{}/vis/fps_point_cloud".format(args.experiment_path)
                # gt_save_dir = "./{}/vis/gt".format(args.experiment_path)
                # os.makedirs(input_point_save_dir, exist_ok=True)
                # os.makedirs(dense_point_save_dir, exist_ok=True)
                # os.makedirs(coarse_point_save_dir, exist_ok=True)
                # os.makedirs(sampled_point_save_dir, exist_ok=True)
                # os.makedirs(input_fps_point_save_dir, exist_ok=True)
                # os.makedirs(gt_save_dir, exist_ok=True)
                # # 保存点云
                # input_dict["partial_points"].cpu().numpy().astype(np.float32).tofile("{}/{}_{}_input.bin".format(input_point_save_dir,
                #                                                                              shapenet_dict[taxonomy_ids[0]],
                #                                                                              model_id))
                # sampled_coarse.cpu().numpy().astype(np.float32).tofile("{}/{}_{}_sampled.bin".format(sampled_point_save_dir,
                #                                                                              shapenet_dict[taxonomy_ids[0]],
                #                                                                              model_id))
                # simp_pc.cpu().numpy().astype(np.float32).tofile("{}/{}_{}_coarse.bin".format(coarse_point_save_dir,
                #                                                                              shapenet_dict[taxonomy_ids[0]],
                #                                                                              model_id))
                # dense_points.cpu().numpy().astype(np.float32).tofile("{}/{}_{}_dense.bin".format(dense_point_save_dir,
                #                                                                                  shapenet_dict[taxonomy_ids[0]],
                #                                                                                  model_id))
                # input_dict["input_fps"].cpu().numpy().astype(np.float32).tofile("{}/{}_{}_fps.bin".format(input_fps_point_save_dir,
                #                                                                                  shapenet_dict[taxonomy_ids[0]],
                #                                                                                  model_id))
                # gt_points.cpu().numpy().astype(np.float32).tofile("{}/{}_{}_gt.bin".format(gt_save_dir,
                #                                                                            shapenet_dict[taxonomy_ids[0]],
                #                                                                            model_id))
                
                sparse_loss_l1 =  ChamferDisL1(simp_pc, gt_points)
                sparse_loss_l2 =  ChamferDisL2(simp_pc, gt_points)
                dense_loss_l1 =  ChamferDisL1(dense_points, gt_points)
                dense_loss_l2 =  ChamferDisL2(dense_points, gt_points)

                loss_values = [
                    sparse_loss_l1.item() * 1000,
                    sparse_loss_l2.item() * 1000,
                    dense_loss_l1.item() * 1000,
                    dense_loss_l2.item() * 1000,
                ]
                test_losses.update(loss_values)

                _metrics = Metrics.get(dense_points, gt_points,
                                    taxonomy_ids_remap, taxonomy_ids_remap,
                                    num_class=_evaluation_class_count(config),
                                    include_cls=not config.model.get("label_free_training", False))
                _metrics = _filter_metric_values(config, _metrics)
                _metrics = [_metric.item() for _metric in _metrics]

                # 可视化保存数据
                # if _metrics[0] > 85:
                #     from utils.common_utils import mkdir_p
                #     save_path = r"/home/csstuer/yyx/GS-Net-yyx/vis_save/PCN/"
                #     mkdir_p("{}/partial".format(save_path))
                #     mkdir_p("{}/pred".format(save_path))
                #     mkdir_p("{}/gt".format(save_path))
                #     write_bin_float32(input_dict["partial_points"].cpu().numpy(), "{}/partial/{}_{}.bin".format(save_path, taxonomy_ids[0], model_ids[0]))
                #     write_bin_float32(input_dict["rebuild_points"].cpu().numpy(), "{}/pred/{}_{}.bin".format(save_path, taxonomy_ids[0], model_ids[0]))
                #     write_bin_float32(input_dict["gt_points"].cpu().numpy(), "{}/gt/{}_{}.bin".format(save_path, taxonomy_ids[0], model_ids[0]))

                # from utils.common_utils import mkdir_p
                # path = r"/home/csstuer/yyx/GS-Net-yyx/vis_save/pseudo_SnowFlakeNet/pred"
                # mkdir_p(path)
                # write_bin_float32(input_dict["rebuild_points"].cpu().numpy(), "{}/{}_{}.bin".format(path, taxonomy_ids[0], model_ids[0]))
                
                

                for _taxonomy_id in taxonomy_ids:
                    _taxonomy_id = str(_taxonomy_id)
                    if _taxonomy_id not in category_metrics:
                        category_metrics[_taxonomy_id] = AverageMeter(
                            _metric_names(config)
                        )
                    category_metrics[_taxonomy_id].update(_metrics)
                    eval_records.append({
                        'sample_id': f'{_taxonomy_id}:{model_id}',
                        'taxonomy_id': _taxonomy_id,
                        'losses': loss_values,
                        'metrics': _metrics,
                    })

            elif 'PCN' in dataset_name:
                gt_points = data[1].cuda()
                taxonomy_ids_remap = np.squeeze(taxonomy_ids)
                taxonomy_ids_input = taxonomy_ids
                taxonomy_ids_remap = torch.from_numpy(taxonomy_ids_remap).unsqueeze(0).cuda()
                # 把taxonomy_ids重新映射回原来的编号
                reid2id = {v:k for k, v in pcn_config["remap"].items()}
                taxonomy_ids = [reid2id[str(i)] for i in taxonomy_ids]
                if config.model.get("shapenet_to_pcn_eval", False):
                    if args.mode not in crop_ratio:
                        raise ValueError(
                            "ShapeNet-to-PCN evaluation requires --mode to be "
                            "one of: easy, median, hard"
                        )
                    choice = [
                        torch.Tensor([1, 1, 1]),
                        torch.Tensor([1, 1, -1]),
                        torch.Tensor([1, -1, 1]),
                        torch.Tensor([-1, 1, 1]),
                        torch.Tensor([-1, -1, 1]),
                        torch.Tensor([-1, 1, -1]),
                        torch.Tensor([1, -1, -1]),
                        torch.Tensor([-1, -1, -1]),
                    ]
                    num_crop = int(npoints * crop_ratio[args.mode])
                    partial_views = []
                    for view_idx, item in enumerate(choice):
                        partial_points, _ = misc.seprate_point_cloud(
                            gt_points,
                            npoints,
                            num_crop,
                            fixed_points=item,
                        )
                        partial_points = misc.fps(partial_points, 2048)
                        partial_views.append(
                            (view_idx, partial_points, None)
                        )
                else:
                    partial_views = [(0, data[0].cuda(), adj.cuda())]

                for view_idx, partial_points, partial_adj in partial_views:
                    if partial_adj is None:
                        partial_adj = get_radius_graph(
                            partial_points, r=0.2
                        ).cuda()
                    input_dict = {
                        "partial_points": partial_points,
                        "gt_points": gt_points,
                        "adj": partial_adj,
                        "taxonomy_ids": taxonomy_ids_input,
                        "model_id": model_id,
                        "keys": keys,
                        "values": values,
                        "junction_index": torch.arange(224).reshape(-1, 224).cuda(),
                    }

                    input_dict = base_model(
                        _sanitize_label_free_model_input(input_dict, config)
                    )
                    coarse_points = _coarse_points_bnc(
                        input_dict["coarse_points"]
                    )
                    dense_points = input_dict["rebuild_points"]

                    sparse_loss_l1 = ChamferDisL1(coarse_points, gt_points)
                    sparse_loss_l2 = ChamferDisL2(coarse_points, gt_points)
                    dense_loss_l1 = ChamferDisL1(dense_points, gt_points)
                    dense_loss_l2 = ChamferDisL2(dense_points, gt_points)

                    loss_values = [
                        sparse_loss_l1.item() * 1000,
                        sparse_loss_l2.item() * 1000,
                        dense_loss_l1.item() * 1000,
                        dense_loss_l2.item() * 1000,
                    ]
                    test_losses.update(loss_values)

                    _metrics = Metrics.get(
                        dense_points,
                        gt_points,
                        taxonomy_ids_remap,
                        taxonomy_ids_remap,
                        num_class=_evaluation_class_count(config),
                        include_cls=not config.model.get("label_free_training", False),
                    )
                    _metrics = _filter_metric_values(config, _metrics)
                    _metrics = [_metric.item() for _metric in _metrics]

                    for _taxonomy_id in taxonomy_ids:
                        _taxonomy_id = str(_taxonomy_id)
                        if _taxonomy_id not in category_metrics:
                            category_metrics[_taxonomy_id] = AverageMeter(
                                _metric_names(config)
                            )
                        category_metrics[_taxonomy_id].update(_metrics)
                        eval_records.append({
                            'sample_id': (
                                f'{_taxonomy_id}:{model_id}:{view_idx}'
                            ),
                            'taxonomy_id': _taxonomy_id,
                            'losses': loss_values,
                            'metrics': _metrics,
                        })
            elif dataset_name == 'KITTI':
                partial_points = data[0].cuda()
                adj = adj.cuda()
                input_dict = {
                    "partial_points": partial_points,
                    # "gt_points": gt_points,
                    "adj": adj,
                    "taxonomy_ids": taxonomy_ids,
                    "model_id": model_id,
                    "keys": keys,
                    "values": values,
                    "junction_index": torch.arange(224).reshape(-1, 224).cuda(),
                }
                input_dict = base_model(
                    _sanitize_label_free_model_input(input_dict, config)
                )
                dense_points = input_dict["rebuild_points"]
                target_path = os.path.join(args.experiment_path, 'vis_result')
                if not os.path.exists(target_path):
                    os.mkdir(target_path)
                misc.visualize_KITTI(
                    os.path.join(target_path, f'{model_id}_{idx:03d}'),
                    [partial_points[0].cpu(), dense_points[0].cpu()]
                )
                continue
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            if (idx+1) % 200 == 0:
                print_log('Test[%d/%d] Taxonomy = %s Sample = %s Losses = %s Metrics = %s' %
                            (idx + 1, n_samples, taxonomy_id, model_id, ['%.4f' % l for l in test_losses.val()], 
                            ['%.4f' % m for m in _metrics]), logger=logger)
        if dataset_name == 'KITTI':
            return

        if args.distributed:
            eval_records = dist_utils.gather_eval_records(eval_records, args)

        test_losses.reset()
        test_metrics.reset()
        category_metrics = dict()
        for record in eval_records:
            test_losses.update(record['losses'])
            taxonomy_id = record['taxonomy_id']
            if taxonomy_id not in category_metrics:
                category_metrics[taxonomy_id] = AverageMeter(
                    _metric_names(config)
                )
            category_metrics[taxonomy_id].update(record['metrics'])

        for _,v in category_metrics.items():
            test_metrics.update(v.avg())
        print_log('[TEST] Metrics = %s' % (['%.4f' % m for m in test_metrics.avg()]), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

    # Print testing results
    shapenet_dict = json.load(open(config.dataset.test._base_.TEST_JSON, 'r'))
    print_log('============================ TEST RESULTS ============================',logger=logger)
    msg = ''
    msg += 'Taxonomy\t'
    msg += '#Sample\t'
    for metric in test_metrics.items:
        msg += metric + '\t'
    msg += '#ModelName\t'
    print_log(msg, logger=logger)


    for taxonomy_id in category_metrics:
        msg = ''
        msg += (taxonomy_id + '\t')
        msg += (str(category_metrics[taxonomy_id].count(0)) + '\t')
        for value in category_metrics[taxonomy_id].avg():
            msg += '%.3f \t' % value
        msg += shapenet_dict[taxonomy_id] + '\t'
        print_log(msg, logger=logger)

    msg = ''
    msg += 'Overall \t\t'
    for value in test_metrics.avg():
        msg += '%.5f \t' % value
    print_log(msg, logger=logger)
    return
