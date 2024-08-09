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

def run_net_memory(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    # Build dataset
    (train_sampler, train_dataloader), (_, test_dataloader) = builder.dataset_builder(args, config.dataset.train), \
                                                            builder.dataset_builder(args, config.dataset.val)
    # Build model
    base_model = builder.model_builder(config.model)
    if args.use_gpu:
        base_model.to(args.local_rank)

    # Parameter setting
    start_epoch = 0
    best_metrics = None
    metrics = None

    # Resume checkpoints
    if args.resume:
        start_epoch, best_metrics = builder.resume_model(base_model, args, logger=logger)
        best_metrics = Metrics(config.consider_metric, best_metrics)
    elif args.start_ckpts is not None:
        builder.load_model(base_model, args.start_ckpts, logger=logger)

    # Print model info
    print_log('Trainable parameters:', logger=logger)
    print_log('=' * 25, logger=logger)
    for name, param in base_model.named_parameters():
        if param.requires_grad:
            print_log(name, logger=logger)
    print_log('=' * 25, logger=logger)
    
    print_log('Untrainable parameters:', logger=logger)
    print_log('=' * 25, logger=logger)
    for name, param in base_model.named_parameters():
        if not param.requires_grad:
            print_log(name, logger=logger)
    print_log('=' * 25, logger=logger)

    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger=logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[args.local_rank % torch.cuda.device_count()], find_unused_parameters=True)
        print_log('Using Distributed Data parallel ...', logger=logger)
    else:
        print_log('Using Data parallel ...', logger=logger)
        base_model = nn.DataParallel(base_model).cuda()
    
    # Optimizer & scheduler
    optimizer = builder.build_optimizer(base_model, config)
    
    # Criterion
    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()

    if args.resume:
        builder.resume_optimizer(optimizer, args, logger=logger)
    scheduler = builder.build_scheduler(base_model, optimizer, config, last_epoch=start_epoch-1)

    # Load memory bank data or initialize
    memory_size = config.model.get("memory_size", config.model.class_num)
    if args.keys is not None:
        keys = read_bin_float32(args.keys).reshape(memory_size, -1)
        keys = torch.from_numpy(keys).cuda()
    else:
        keys = torch.rand((memory_size, config.model.encoder_config.embed_dim), dtype=torch.float32).cuda()

    if args.values is not None:
        values = read_bin_float32(args.values).reshape(memory_size, -1)
        values = torch.from_numpy(keys).cuda()
    else:
        values = torch.rand((memory_size, config.model.decoder_config.embed_dim), dtype=torch.float32).cuda()

    # Train and validate
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
                               "OrthogonalLoss", "CompactnessLoss", "SeperationLoss", "MaxInterLoss", "MinIntraLoss"])

        num_iter = 0
        n_batches = len(train_dataloader)
        for idx, (taxonomy_ids, model_ids, adj, data) in enumerate(tqdm(train_dataloader, desc="Processing", unit="batch")):
            data_time.update(time.time() - batch_start_time)
            dataset_name = config.dataset.train._base_.NAME
            npoints = config.dataset.train._base_.N_POINTS
            if 'PCN' in dataset_name or dataset_name == 'Completion3D' or 'ProjectShapeNet' in dataset_name:
                partial_points = data[0].cuda()
                gt_points = data[1].cuda()
                adj = adj.cuda()
                taxonomy_ids = np.squeeze(taxonomy_ids)
                taxonomy_ids = torch.from_numpy(taxonomy_ids).cuda()
                
                input_dict = {
                    "partial_points": partial_points,
                    "gt_points": gt_points,
                    "adj": adj,
                    "taxonomy_ids": taxonomy_ids,
                    "keys": keys,
                    "values": values,
                }
                
                if config.dataset.train._base_.CARS:
                    if idx == 0:
                        print_log('padding while KITTI training', logger=logger)
                    partial_points = misc.random_dropping(partial_points, epoch)

            elif 'V2XSeqSPD' in dataset_name:
                partial_points = data[0].cuda()
                gt_points = data[1].cuda()
            elif dataset_name == 'ShapeNet' or dataset_name == 'ShapeNetGNN':
                gt = data.cuda()
                partial, _ = misc.seprate_point_cloud(gt, npoints, [int(npoints * 1/4), int(npoints * 3/4)], fixed_points=None)
                partial = partial.cuda()
                if config.model.get("do_adj", False):
                    adj = get_radius_graph(partial, r=0.2)
                adj = adj.cuda()
                taxonomy_ids = np.squeeze(taxonomy_ids)
                taxonomy_ids = torch.from_numpy(taxonomy_ids).cuda()
                
                input_dict = {
                    "partial_points": partial,
                    "gt_points": gt,
                    "adj": adj,
                    "taxonomy_ids": taxonomy_ids,
                    "keys": keys,
                    "values": values,
                }
            
            elif 'Projected_ShapeNet' in dataset_name:
                partial_points = data[0].cuda()
                gt_points = data[1].cuda()
                adj = adj.cuda()
                taxonomy_ids = np.squeeze(taxonomy_ids)
                taxonomy_ids = torch.from_numpy(taxonomy_ids).cuda()
                
                input_dict = {
                    "partial_points": partial_points,
                    "gt_points": gt_points,
                    "adj": adj,
                    "taxonomy_ids": taxonomy_ids,
                    "keys": keys,
                    "values": values,
                }
            else:
                raise NotImplementedError(f'Train phase does not support {dataset_name}')

            num_iter += 1
            
            input_dict = base_model(input_dict)
            keys = input_dict["keys"].detach()
            values = input_dict["values"].detach()
            simp_pc = input_dict["sampled_coarse"]
            simp_pc = simp_pc.transpose(2,1)
            
            simp_pc = simp_pc.transpose(2,1)

            loss_denoised, sparse_loss, dense_loss = base_model.module.get_rebuild_loss(input_dict)
            sparse_loss = sparse_loss * config.model.get("sparse_loss_param", 1)
            dense_loss = dense_loss * config.model.get("dense_loss_param", 1)
            try:
                linkprediction_loss = base_model.module.get_link_prediction_loss(input_dict)
            except:
                linkprediction_loss = torch.tensor(0).cuda()
            try:
                query_ranking_loss = base_model.module.get_ranking_loss(input_dict)
            except:
                query_ranking_loss = torch.tensor(0).cuda()
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
            try:
                orthogonal_constriant_loss = base_model.module.get_orthogonal_constriant(input_dict)
                orthogonal_constriant_loss = config.model.get("orthogonal_constriant_loss_param", 1) * orthogonal_constriant_loss
            except:
                orthogonal_constriant_loss = torch.tensor(0).cuda()
            try:
                key_sim_loss, value_sim_loss = base_model.module.get_compactness_loss(input_dict)
                key_sim_loss = config.model.get("key_sim_loss_param", 1) * key_sim_loss
                value_sim_loss = config.model.get("value_sim_loss_param", 1) * value_sim_loss
                compactness_loss = key_sim_loss + value_sim_loss
            except:
                compactness_loss = torch.tensor(0).cuda()
            try:
                sim_seperation_loss = base_model.module.get_sim_seperation_loss(input_dict)
                sim_seperation_loss = config.model.get("get_sim_seperation_loss_param", 1) * sim_seperation_loss
            except:
                sim_seperation_loss = torch.tensor(0).cuda()
            try:
                max_inter_loss, min_intra_loss = base_model.module.get_MaxInterMinIntra_loss(input_dict)
                max_inter_loss = config.model.get("get_MaxInter_loss_param", 1) * max_inter_loss
                min_intra_loss = config.model.get("get_MinIntra_loss_param", 1) * min_intra_loss
            except:
                max_inter_loss = torch.tensor(0).cuda()
                min_intra_loss = torch.tensor(0).cuda()
            try:
                simplification_loss = base_model.module.get_simplification_loss(input_dict)
                simplification_loss = config.model.alpha * simplification_loss
            except:
                simplification_loss = torch.tensor(0).cuda()
            
            try:
                task_loss = config.model.task_loss_param * task_loss
            except:
                task_loss = torch.tensor(0).cuda()

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

            if num_iter == config.step_per_update:
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), getattr(config, 'grad_norm_clip', 10), norm_type=2)
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            if args.distributed:
                sparse_loss = dist_utils.reduce_tensor(sparse_loss, args)
                dense_loss = dist_utils.reduce_tensor(dense_loss, args)
                _loss = dist_utils.reduce_tensor(_loss, args)
                task_loss = dist_utils.reduce_tensor(task_loss, args)
                simplification_loss = dist_utils.reduce_tensor(simplification_loss, args)
                linkprediction_loss = dist_utils.reduce_tensor(linkprediction_loss, args)
                loss_denoised = dist_utils.reduce_tensor(loss_denoised, args)
                loss_denoised = dist_utils.reduce_tensor(query_ranking_loss, args)
                class_constraint = dist_utils.reduce_tensor(class_constraint, args)
                class_loss = dist_utils.reduce_tensor(class_loss, args)
                orthogonal_constriant_loss = dist_utils.reduce_tensor(orthogonal_constriant_loss, args)
                compactness_loss = dist_utils.reduce_tensor(compactness_loss, args)
                sim_seperation_loss = dist_utils.reduce_tensor(sim_seperation_loss, args)
                max_inter_loss = dist_utils.reduce_tensor(max_inter_loss, args)
                min_intra_loss = dist_utils.reduce_tensor(min_intra_loss, args)
                losses.update([sparse_loss.item() * 1000, dense_loss.item() * 1000, _loss.item(),
                               simplification_loss.item(), task_loss.item(), linkprediction_loss.item(),
                               loss_denoised.item(), query_ranking_loss.item(), class_constraint.item(),
                               class_loss.item(), orthogonal_constriant_loss.item(), compactness_loss.item(),
                               sim_seperation_loss.item(), max_inter_loss.item(), min_intra_loss.item()])
            else:
                losses.update([sparse_loss.item() * 1000, dense_loss.item() * 1000, _loss.item(),
                               simplification_loss.item(), task_loss.item(), linkprediction_loss.item(),
                               loss_denoised.item(), query_ranking_loss.item(), class_constraint.item(),
                               class_loss.item(), orthogonal_constriant_loss.item(), compactness_loss.item(),
                               sim_seperation_loss.item(), max_inter_loss.item(), min_intra_loss.item()])

            if args.distributed:
                torch.cuda.synchronize()

            n_itr = epoch * n_batches + idx
            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Sparse', sparse_loss.item() * 1000, n_itr)
                train_writer.add_scalar('Loss/Batch/Dense', dense_loss.item() * 1000, n_itr)
                train_writer.add_scalar('Loss/Batch/SumLoss', _loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/SampleNetLoss', simplification_loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/TaskLoss', task_loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/LinkPredLoss', linkprediction_loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/DenoisedLoss', loss_denoised.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/RankingLoss', query_ranking_loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/ConsistentLoss', class_constraint.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/TrainClassifierLoss', class_loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/OrthogonalLoss', orthogonal_constriant_loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/CompactnessLoss', compactness_loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/CosineSimilarSeperationLoss', sim_seperation_loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/MaxInterLoss', max_inter_loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/MinIntraLoss', min_intra_loss.item(), n_itr)

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 100 == 0:
                print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s lr = %.6f' %
                            (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                            ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr']), logger=logger)

            if config.scheduler.type == 'GradualWarmup':
                if n_itr < config.scheduler.kwargs_2.total_epoch:
                    scheduler.step()

        if isinstance(scheduler, list):
            for item in scheduler:
                item.step()
        else:
            scheduler.step()
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

        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s' %
            (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()]), logger=logger)

        if (config.max_epoch - epoch) <= 20 or epoch % args.val_freq == 0:
            metrics = validate(base_model, test_dataloader, epoch, ChamferDisL1, ChamferDisL2, val_writer, args, config, None, None,
                               keys, values, logger=logger)

            if metrics.better_than(best_metrics):
                best_metrics = metrics
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger=logger)
                write_bin_float32(keys.cpu().numpy(), "{}/{}.bin".format(args.experiment_path, "keys-best"))
                write_bin_float32(values.cpu().numpy(), "{}/{}.bin".format(args.experiment_path, "values-best"))
        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger=logger)
        write_bin_float32(keys.cpu().numpy(), "{}/{}.bin".format(args.experiment_path, "keys-last"))
        write_bin_float32(values.cpu().numpy(), "{}/{}.bin".format(args.experiment_path, "values-last"))
        if epoch % args.val_freq == 0:
            builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args, logger=logger)
            write_bin_float32(keys.cpu().numpy(), "{}/{}.bin".format(args.experiment_path, f'keys-epoch-{epoch:03d}'))
            write_bin_float32(values.cpu().numpy(), "{}/{}.bin".format(args.experiment_path, f'values-epoch-{epoch:03d}'))
        if (config.max_epoch - epoch) < 20:
            builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args, logger=logger)
            write_bin_float32(keys.cpu().numpy(), "{}/{}.bin".format(args.experiment_path, f'keys-epoch-{epoch:03d}'))
            write_bin_float32(values.cpu().numpy(), "{}/{}.bin".format(args.experiment_path, f'values-epoch-{epoch:03d}'))
    if train_writer is not None and val_writer is not None:
        train_writer.close()
        val_writer.close()

def validate(base_model, test_dataloader, epoch, ChamferDisL1, ChamferDisL2, val_writer, args, config, classifier, classifier_loss,
             keys, values, logger=None):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger=logger)
    base_model.eval()

    test_losses = AverageMeter(['SparseLossL1', 'SparseLossL2', 'DenseLossL1', 'DenseLossL2'])
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()
    n_samples = len(test_dataloader)

    interval = n_samples // 10

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
                    "taxonomy_ids": taxonomy_ids,
                    "keys": keys,
                    "values": values,
                }
                taxonomy_ids_remap = np.squeeze(taxonomy_ids)
                taxonomy_ids_remap = torch.from_numpy(taxonomy_ids_remap).unsqueeze(0).cuda()
                reid2id = {v:k for k, v in pcn_config["remap"].items()}
                taxonomy_ids = [reid2id[str(i)] for i in taxonomy_ids]
            elif 'V2XSeqSPD' in dataset_name:
                partial_points = data[0].cuda()
                gt_points = data[1].cuda()
                gt_seg = data[2].cuda()
            elif dataset_name == 'ShapeNet' or dataset_name == 'ShapeNetGNN':
                gt_points = data.cuda()
                partial_points, _ = misc.seprate_point_cloud(gt_points, npoints, [int(npoints * 1/4), int(npoints * 3/4)], fixed_points=None)
                partial_points = partial_points.cuda()
                if config.model.get("do_adj", False):
                    adj = get_radius_graph(partial_points, r=0.2)
                adj = adj.cuda()
                
                input_dict = {
                    "partial_points": partial_points,
                    "gt_points": gt_points,
                    "adj": adj,
                    "taxonomy_ids": taxonomy_ids,
                    "keys": keys,
                    "values": values,
                }
                taxonomy_ids_remap = np.squeeze(taxonomy_ids)
                taxonomy_ids_remap = torch.from_numpy(taxonomy_ids_remap).unsqueeze(0).cuda()
                if "34" in config.dataset.val._base_.DATA_PATH.split("/"):
                    reid2id = {v:k for k, v in shapenet_config["shapenet34_remap"].items()}
                else:
                    reid2id = {v:k for k, v in shapenet_config["remap"].items()}
                taxonomy_ids = [reid2id[str(i)] for i in taxonomy_ids]
            elif 'Projected_ShapeNet' in dataset_name:
                partial_points = data[0].cuda()
                gt_points = data[1].cuda()
                adj = adj.cuda()
                input_dict = {
                    "partial_points": partial_points,
                    "gt_points": gt_points,
                    "adj": adj,
                    "taxonomy_ids": taxonomy_ids,
                    "keys": keys,
                    "values": values,
                }
                taxonomy_ids_remap = np.squeeze(taxonomy_ids)
                taxonomy_ids_remap = torch.from_numpy(taxonomy_ids_remap).unsqueeze(0).cuda()
                if "34" in config.dataset.val._base_.DATA_PATH.split("/"):
                    reid2id = {v:k for k, v in shapenet_config["shapenet34_remap"].items()}
                else:
                    reid2id = {v:k for k, v in shapenet_config["remap"].items()}
                taxonomy_ids = [reid2id[str(i)] for i in taxonomy_ids]
            else:
                raise NotImplementedError(f'Train phase does not support {dataset_name}')
            
            input_dict = base_model(input_dict)
            simp_pc = input_dict["sampled_coarse"]
            dense_points = input_dict["rebuild_points"]
            simp_pc = simp_pc.transpose(2,1)
            
            sparse_loss_l1 = ChamferDisL1(simp_pc, gt_points)
            sparse_loss_l2 = ChamferDisL2(simp_pc, gt_points)
            dense_loss_l1 = ChamferDisL1(dense_points, gt_points)
            dense_loss_l2 = ChamferDisL2(dense_points, gt_points)

            if args.distributed:
                sparse_loss_l1 = dist_utils.reduce_tensor(sparse_loss_l1, args)
                sparse_loss_l2 = dist_utils.reduce_tensor(sparse_loss_l2, args)
                dense_loss_l1 = dist_utils.reduce_tensor(dense_loss_l1, args)
                dense_loss_l2 = dist_utils.reduce_tensor(dense_loss_l2, args)

            test_losses.update([sparse_loss_l1.item() * 1000,
                                sparse_loss_l2.item() * 1000,
                                dense_loss_l1.item() * 1000,
                                dense_loss_l2.item() * 1000])

            _metrics = Metrics.get(dense_points, gt_points, taxonomy_ids_remap, taxonomy_ids_remap, num_class=config.model.class_num)
            if args.distributed:
                _metrics = [dist_utils.reduce_tensor(_metric, args).item() for _metric in _metrics]
            else:
                _metrics = [_metric.item() for _metric in _metrics]

            for _taxonomy_id in taxonomy_ids:
                if _taxonomy_id not in category_metrics:
                    category_metrics[_taxonomy_id] = AverageMeter(Metrics.names())
                category_metrics[_taxonomy_id].update(_metrics)

            if (idx+1) % interval == 0:
                print_log('Test[%d/%d] Taxonomy = %s Sample = %s Losses = %s Metrics = %s' %
                            (idx + 1, n_samples, taxonomy_id, model_id, ['%.4f' % l for l in test_losses.val()], 
                            ['%.4f' % m for m in _metrics]), logger=logger)
        for _, v in category_metrics.items():
            test_metrics.update(v.avg())
        print_log('[Validation] EPOCH: %d  Metrics = %s' % (epoch, ['%.4f' % m for m in test_metrics.avg()]), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()
     
    # Print testing results
    shapenet_dict = json.load(open(config.dataset.val._base_.TEST_JSON, 'r'))
    print_log('============================ TEST RESULTS ============================', logger=logger)
    msg = 'Taxonomy\t#Sample\t'
    for metric in test_metrics.items:
        msg += metric + '\t'
    msg += '#ModelName\t'
    print_log(msg, logger=logger)

    for taxonomy_id in category_metrics:
        msg = (taxonomy_id + '\t')
        msg += (str(category_metrics[taxonomy_id].count(0)) + '\t')
        for value in category_metrics[taxonomy_id].avg():
            msg += '%.3f \t' % value
        msg += shapenet_dict[taxonomy_id] + '\t'
        print_log(msg, logger=logger)

    msg = 'Overall\t\t'
    for value in test_metrics.avg():
        msg += '%.3f \t' % value
    print_log(msg, logger=logger)

    if val_writer is not None:
        val_writer.add_scalar('Loss/Epoch/Sparse*1000', test_losses.avg(0), epoch)
        val_writer.add_scalar('Loss/Epoch/Dense*1000', test_losses.avg(2), epoch)
        val_writer.add_scalar('Loss/Epoch/Sparse', test_losses.avg(0) / 1000, epoch)
        val_writer.add_scalar('Loss/Epoch/Dense', test_losses.avg(2) / 1000, epoch)
        for i, metric in enumerate(test_metrics.items):
            val_writer.add_scalar('Metric/%s' % metric, test_metrics.avg(i), epoch)

    return Metrics(config.consider_metric, test_metrics.avg())

crop_ratio = {
    'easy': 1/4,
    'median': 1/2,
    'hard': 3/4
}

def test_net_memory(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ...', logger=logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)
 
    base_model = builder.model_builder(config.model)
    builder.load_model(base_model, args.ckpts, logger=logger)
    if args.use_gpu:
        base_model.to(args.local_rank)

    # DDP    
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger=logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[args.local_rank % torch.cuda.device_count()], find_unused_parameters=True)
        print_log('Using Distributed Data parallel ...', logger=logger)
    else:
        print_log('Do Not Use Data parallel ...', logger=logger)

    # Criterion
    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()

    test(base_model, test_dataloader, ChamferDisL1, ChamferDisL2, args, config, logger=logger)

def test(base_model, test_dataloader, ChamferDisL1, ChamferDisL2, args, config, logger=None):

    base_model.eval()

    assert args.keys is not None
    assert args.values is not None
    if args.keys == "1" or args.values == "1":
        keys = torch.tensor(0).cuda()
        values = torch.tensor(0).cuda()
    else:
        memory_size = config.model.get("memory_size", config.model.class_num)
        keys = read_bin_float32(args.keys).reshape(memory_size, -1)
        keys = torch.from_numpy(keys).cuda()
        values = read_bin_float32(args.values).reshape(memory_size, -1)
        values = torch.from_numpy(values).cuda()

    test_losses = AverageMeter(['SparseLossL1', 'SparseLossL2', 'DenseLossL1', 'DenseLossL2'])
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()
    n_samples = len(test_dataloader)
    shapenet_dict = json.load(open(config.dataset.val._base_.TEST_JSON, 'r'))

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, adj, data) in tqdm(enumerate(test_dataloader)):
            taxonomy_id = taxonomy_ids
            model_id = model_ids

            npoints = config.dataset.test._base_.N_POINTS
            dataset_name = config.dataset.test._base_.NAME
            if 'ProjectShapeNet' in dataset_name:
                partial_points = data[0].cuda()
                gt_points = data[1].cuda()

                ret = base_model(partial_points)
                coarse_points = ret[0]
                dense_points = ret[1]

                sparse_loss_l1 = ChamferDisL1(coarse_points, gt_points)
                sparse_loss_l2 = ChamferDisL2(coarse_points, gt_points)
                dense_loss_l1 = ChamferDisL1(dense_points, gt_points)
                dense_loss_l2 = ChamferDisL2(dense_points, gt_points)

                if args.distributed:
                    sparse_loss_l1 = dist_utils.reduce_tensor(sparse_loss_l1, args)
                    sparse_loss_l2 = dist_utils.reduce_tensor(sparse_loss_l2, args)
                    dense_loss_l1 = dist_utils.reduce_tensor(dense_loss_l1, args)
                    dense_loss_l2 = dist_utils.reduce_tensor(dense_loss_l2, args)

                test_losses.update([sparse_loss_l1.item() * 1000, sparse_loss_l2.item() * 1000, dense_loss_l1.item() * 1000, dense_loss_l2.item() * 1000])
                _metrics = Metrics.get(dense_points, gt_points, require_emd=False)
                
                if taxonomy_id not in category_metrics:
                    category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
                category_metrics[taxonomy_id].update(_metrics)
            
            elif dataset_name == 'ShapeNet' or dataset_name == 'ShapeNetGNN':
                gt_points = data.cuda()
                choice = [torch.Tensor([1,1,1]),torch.Tensor([1,1,-1]),torch.Tensor([1,-1,1]),torch.Tensor([-1,1,1]),
                            torch.Tensor([-1,-1,1]),torch.Tensor([-1,1,-1]), torch.Tensor([1,-1,-1]),torch.Tensor([-1,-1,-1])]
                num_crop = int(npoints * crop_ratio[args.mode])
                taxonomy_ids_remap = np.squeeze(taxonomy_ids)
                taxonomy_ids_input = taxonomy_ids
                taxonomy_ids_remap = torch.from_numpy(taxonomy_ids_remap).unsqueeze(0).cuda()
                reid2id = {v:k for k, v in shapenet_config["remap"].items()}
                taxonomy_ids = [reid2id[str(i)] for i in taxonomy_ids]
                for item in choice:           
                    partial_points, _ = misc.seprate_point_cloud(gt_points, npoints, num_crop, fixed_points=item)
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
                    }

                    input_dict = base_model(input_dict)
                    simp_pc = input_dict["sampled_coarse"]
                    dense_points = input_dict["rebuild_points"]
                    simp_pc = simp_pc.transpose(2,1)

                    sparse_loss_l1 = ChamferDisL1(simp_pc, gt_points)
                    sparse_loss_l2 = ChamferDisL2(simp_pc, gt_points)
                    dense_loss_l1 = ChamferDisL1(dense_points, gt_points)
                    dense_loss_l2 = ChamferDisL2(dense_points, gt_points)

                    if args.distributed:
                        sparse_loss_l1 = dist_utils.reduce_tensor(sparse_loss_l1, args)
                        sparse_loss_l2 = dist_utils.reduce_tensor(sparse_loss_l2, args)
                        dense_loss_l1 = dist_utils.reduce_tensor(dense_loss_l1, args)
                        dense_loss_l2 = dist_utils.reduce_tensor(dense_loss_l2, args)
                    
                    test_losses.update([sparse_loss_l1.item() * 1000,
                                        sparse_loss_l2.item() * 1000,
                                        dense_loss_l1.item() * 1000,
                                        dense_loss_l2.item() * 1000])

                    _metrics = Metrics.get(dense_points, gt_points, taxonomy_ids_remap, taxonomy_ids_remap, num_class=config.model.class_num)
                    if args.distributed:
                        _metrics = [dist_utils.reduce_tensor(_metric, args).item() for _metric in _metrics]
                    else:
                        _metrics = [_metric.item() for _metric in _metrics]

                    for _taxonomy_id in taxonomy_ids:
                        if _taxonomy_id not in category_metrics:
                            category_metrics[_taxonomy_id] = AverageMeter(Metrics.names())
                        category_metrics[_taxonomy_id].update(_metrics)

            elif 'Projected_ShapeNet' in dataset_name:
                partial_points = data[0].cuda()
                gt_points = data[1].cuda()
                adj = adj.cuda()
                input_dict = {
                    "partial_points": partial_points,
                    "gt_points": gt_points,
                    "adj": adj,
                    "taxonomy_ids": taxonomy_ids,
                    "model_id": model_id,
                    "keys": keys,
                    "values": values,
                }

                taxonomy_ids_remap = np.squeeze(taxonomy_ids)
                taxonomy_ids_remap = torch.from_numpy(taxonomy_ids_remap).unsqueeze(0).cuda()
                reid2id = {v:k for k, v in shapenet_config["remap"].items()}
                taxonomy_ids = [reid2id[str(i)] for i in taxonomy_ids]
                
                input_dict = base_model(input_dict)
                sampled_coarse = input_dict["sampled_coarse"]
                simp_pc = input_dict["coarse_points"]
                dense_points = input_dict["rebuild_points"]
                simp_pc = simp_pc.transpose(2,1)

                sparse_loss_l1 = ChamferDisL1(simp_pc, gt_points)
                sparse_loss_l2 = ChamferDisL2(simp_pc, gt_points)
                dense_loss_l1 = ChamferDisL1(dense_points, gt_points)
                dense_loss_l2 = ChamferDisL2(dense_points, gt_points)

                if args.distributed:
                    sparse_loss_l1 = dist_utils.reduce_tensor(sparse_loss_l1, args)
                    sparse_loss_l2 = dist_utils.reduce_tensor(sparse_loss_l2, args)
                    dense_loss_l1 = dist_utils.reduce_tensor(dense_loss_l1, args)
                    dense_loss_l2 = dist_utils.reduce_tensor(dense_loss_l2, args)
                
                test_losses.update([sparse_loss_l1.item() * 1000,
                                    sparse_loss_l2.item() * 1000,
                                    dense_loss_l1.item() * 1000,
                                    dense_loss_l2.item() * 1000])

                _metrics = Metrics.get(dense_points, gt_points, taxonomy_ids_remap, taxonomy_ids_remap, num_class=config.model.class_num)
                if args.distributed:
                    _metrics = [dist_utils.reduce_tensor(_metric, args).item() for _metric in _metrics]
                else:
                    _metrics = [_metric.item() for _metric in _metrics]

                for _taxonomy_id in taxonomy_ids:
                    if _taxonomy_id not in category_metrics:
                        category_metrics[_taxonomy_id] = AverageMeter(Metrics.names())
                    category_metrics[_taxonomy_id].update(_metrics)

            elif 'PCN' in dataset_name:
                gt_points = data[1].cuda()
                choice = [torch.Tensor([1,1,1]),torch.Tensor([1,1,-1]),torch.Tensor([1,-1,1]),torch.Tensor([-1,1,1]),
                            torch.Tensor([-1,-1,1]),torch.Tensor([-1,1,-1]), torch.Tensor([1,-1,-1]),torch.Tensor([-1,-1,-1])]
                num_crop = int(npoints * crop_ratio[args.mode])
                taxonomy_ids_remap = np.squeeze(taxonomy_ids)
                taxonomy_ids_input = taxonomy_ids
                taxonomy_ids_remap = torch.from_numpy(taxonomy_ids_remap).unsqueeze(0).cuda()
                reid2id = {v:k for k, v in pcn_config["remap"].items()}
                taxonomy_ids = [reid2id[str(i)] for i in taxonomy_ids]
                for item in choice:           
                    partial_points, _ = misc.seprate_point_cloud(gt_points, npoints, num_crop, fixed_points=item)
                    partial_points = misc.fps(partial_points, 2048)
                    adj = adj.cuda()
                    input_dict = {
                        "partial_points": partial_points,
                        "gt_points": gt_points,
                        "adj": adj,
                        "taxonomy_ids": taxonomy_ids_input,
                        "model_id": model_id,
                        "keys": keys,
                        "values": values,
                    }
                    
                    input_dict = base_model(input_dict)
                    coarse_points = input_dict["coarse_points"]
                    dense_points = input_dict["rebuild_points"]

                    sparse_loss_l1 = ChamferDisL1(coarse_points, gt_points)
                    sparse_loss_l2 = ChamferDisL2(coarse_points, gt_points)
                    dense_loss_l1 = ChamferDisL1(dense_points, gt_points)
                    dense_loss_l2 = ChamferDisL2(dense_points, gt_points)

                    test_losses.update([sparse_loss_l1.item() * 1000, sparse_loss_l2.item() * 1000, dense_loss_l1.item() * 1000, dense_loss_l2.item() * 1000])

                    _metrics = Metrics.get(dense_points, gt_points, taxonomy_ids_remap, taxonomy_ids_remap, num_class=config.model.class_num)

                    for _taxonomy_id in taxonomy_ids:
                        if _taxonomy_id not in category_metrics:
                            category_metrics[_taxonomy_id] = AverageMeter(Metrics.names())
                        category_metrics[_taxonomy_id].update(_metrics)
            elif dataset_name == 'KITTI':
                partial_points = data.cuda()
                ret = base_model(partial_points)
                dense_points = ret[-1]
                target_path = os.path.join(args.experiment_path, 'vis_result')
                if not os.path.exists(target_path):
                    os.mkdir(target_path)
                misc.visualize_KITTI(
                    os.path.join(target_path, f'{model_id}_{idx:03d}'),
                    [partial_points[0].cpu(), dense_points[0].cpu()]
                )
                continue
            else:
                raise NotImplementedError(f'Train phase does not support {dataset_name}')

            if (idx+1) % 200 == 0:
                print_log('Test[%d/%d] Taxonomy = %s Sample = %s Losses = %s Metrics = %s' %
                            (idx + 1, n_samples, taxonomy_id, model_id, ['%.4f' % l for l in test_losses.val()], 
                            ['%.4f' % m for m in _metrics]), logger=logger)
        if dataset_name == 'KITTI':
            return
        for _, v in category_metrics.items():
            test_metrics.update(v.avg())
        print_log('[TEST] Metrics = %s' % (['%.4f' % m for m in test_metrics.avg()]), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

    shapenet_dict = json.load(open(config.dataset.test._base_.TEST_JSON, 'r'))
    print_log('============================ TEST RESULTS ============================', logger=logger)
    msg = 'Taxonomy\t#Sample\t'
    for metric in test_metrics.items:
        msg += metric + '\t'
    msg += '#ModelName\t'
    print_log(msg, logger=logger)

    for taxonomy_id in category_metrics:
        msg = (taxonomy_id + '\t')
        msg += (str(category_metrics[taxonomy_id].count(0)) + '\t')
        for value in category_metrics[taxonomy_id].avg():
            msg += '%.3f \t' % value
        msg += shapenet_dict[taxonomy_id] + '\t'
        print_log(msg, logger=logger)

    msg = 'Overall \t\t'
    for value in test_metrics.avg():
        msg += '%.5f \t' % value
    print_log(msg, logger=logger)
    return 
