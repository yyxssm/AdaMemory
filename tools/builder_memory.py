import os, sys
# online package
import torch
# optimizer
import torch.optim as optim
from timm.scheduler import CosineLRScheduler
# dataloader
from datasets import build_dataset_from_cfg
from models import build_model_from_cfg
# utils
from utils.logger import *
from utils.misc import *

from torch.nn.utils.rnn import pad_sequence

def collate_fn_projected_shapenet(batch):
    taxonomy_ids = [item[0] for item in batch]
    model_ids = [item[1] for item in batch]
    # 处理两种数据格式：
    # 1. (taxonomy_id, model_id, (partial, gt)) - Projected_ShapeNet格式
    # 2. (taxonomy_id, model_id, partial, gt, [adj]) - GNN格式
    if len(batch) > 0 and isinstance(batch[0][2], tuple):
        # Projected_ShapeNet格式
        partial_data = [item[2][0].clone().detach() for item in batch]
        gt_data = [item[2][1].clone().detach() for item in batch]
        adjs = [torch.tensor((0,1)) for item in batch]
    else:
        # GNN格式
        partial_data = [item[2].clone().detach() for item in batch]
        gt_data = [item[3].clone().detach() for item in batch]
        try:
            adjs = [item[4].clone().detach() for item in batch]
        except IndexError:
            adjs = [torch.tensor((0,1)) for item in batch]

    # # 使用 pad_sequence 对 partial_data 和 gt_data 进行填充
    # taxonomy_ids = np.squeeze(taxonomy_ids)
    partial_data = pad_sequence(partial_data, batch_first=True)
    gt_data = pad_sequence(gt_data, batch_first=True)
    adjs = pad_sequence(adjs, batch_first=True)

    return taxonomy_ids, model_ids, adjs, (partial_data, gt_data)


def collate_fn_pcn(batch):
    taxonomy_ids = [item[0] for item in batch]
    model_ids = [item[1] for item in batch]
    partial_data = [item[2].clone().detach() for item in batch]
    gt_data = [item[3].clone().detach() for item in batch]
    # 如果没有构建gnn，那么就随便搞一个东西来补位
    try:
        adjs = [item[4].clone().detach() for item in batch]
    except IndexError:
        adjs = [torch.tensor((0,1)) for item in batch]

    # # 使用 pad_sequence 对 partial_data 和 gt_data 进行填充
    # taxonomy_ids = np.squeeze(taxonomy_ids)
    partial_data = pad_sequence(partial_data, batch_first=True)
    gt_data = pad_sequence(gt_data, batch_first=True)
    adjs = pad_sequence(adjs, batch_first=True)

    return taxonomy_ids, model_ids, adjs, (partial_data, gt_data)


def collate_fn_kitti(batch):
    taxonomy_ids = [item[0] for item in batch]
    model_ids = [item[1] for item in batch]
    partial_data = [item[2].clone().detach() for item in batch]
    # gt_data = [item[3].clone().detach() for item in batch]
    gt_data = None
    # 如果没有构建gnn，那么就随便搞一个东西来补位
    try:
        adjs = [item[4].clone().detach() for item in batch]
    except IndexError:
        adjs = [torch.tensor((0,1)) for item in batch]

    # # 使用 pad_sequence 对 partial_data 和 gt_data 进行填充
    # taxonomy_ids = np.squeeze(taxonomy_ids)
    partial_data = pad_sequence(partial_data, batch_first=True)
    # gt_data = pad_sequence(gt_data, batch_first=True)
    adjs = pad_sequence(adjs, batch_first=True)

    return taxonomy_ids, model_ids, adjs, (partial_data, gt_data)


def collate_fn_shapenet(batch):
    taxonomy_ids = [item[0] for item in batch]
    model_ids = [item[1] for item in batch]
    # 对于shapenet55类别来说，partial_data是需要进行fps采样的，所以采样代码在runner里，所以这里只有gt
    gt_data = [item[2].clone().detach() for item in batch]
    # shapenet数据集是没有adj的，因为对于partial points的adj，需要先FPS采样之后才能得到，所以放到runner里面了
    adjs = [torch.tensor((0,1)) for item in batch]

    # # 使用 pad_sequence 对 partial_data 和 gt_data 进行填充
    # taxonomy_ids = np.squeeze(taxonomy_ids)
    gt_data = pad_sequence(gt_data, batch_first=True)
    adjs = pad_sequence(adjs, batch_first=True)

    return taxonomy_ids, model_ids, adjs, gt_data


def dataset_builder(args, config):
    dataset = build_dataset_from_cfg(config._base_, config.others)
    shuffle = config.others.subset == 'train'
    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle = shuffle)
        if dataset.__class__.__name__ == "ShapeNet":
            dataloader = torch.utils.data.DataLoader(dataset, batch_size = config.others.bs if shuffle else 1,
                                    **dataloader_worker_kwargs(args.num_workers, persistent=shuffle),
                                    drop_last = config.others.subset == 'train',
                                    worker_init_fn = worker_init_fn,
                                    sampler = sampler,
                                    collate_fn=collate_fn_shapenet)
        elif "PCN" in dataset.__class__.__name__:
            dataloader = torch.utils.data.DataLoader(dataset, batch_size = config.others.bs if shuffle else 1,
                                                **dataloader_worker_kwargs(args.num_workers, persistent=shuffle),
                                                drop_last = config.others.subset == 'train',
                                                worker_init_fn = worker_init_fn,
                                                sampler = sampler,
                                                collate_fn=collate_fn_pcn)
        elif "KITTI" in dataset.__class__.__name__:
            dataloader = torch.utils.data.DataLoader(dataset, batch_size = config.others.bs if shuffle else 1,
                                                **dataloader_worker_kwargs(args.num_workers, persistent=shuffle),
                                                drop_last = config.others.subset == 'train',
                                                worker_init_fn = worker_init_fn,
                                                sampler = sampler,
                                                collate_fn=collate_fn_kitti)
        else:
            dataloader = torch.utils.data.DataLoader(dataset, batch_size = config.others.bs if shuffle else 1,
                                                **dataloader_worker_kwargs(args.num_workers, persistent=shuffle),
                                                drop_last = config.others.subset == 'train',
                                                worker_init_fn = worker_init_fn,
                                                sampler = sampler,
                                                collate_fn=collate_fn_projected_shapenet)
    else:
        sampler = None
        if dataset.__class__.__name__ == "ShapeNet":
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.others.bs if shuffle else 1,
                                                    shuffle = shuffle, 
                                                    drop_last = config.others.subset == 'train',
                                                    **dataloader_worker_kwargs(args.num_workers, persistent=shuffle),
                                                    worker_init_fn=worker_init_fn,
                                                    collate_fn=collate_fn_shapenet)
        elif "PCN" in dataset.__class__.__name__:
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.others.bs if shuffle else 1,
                                                    shuffle = shuffle, 
                                                    drop_last = config.others.subset == 'train',
                                                    **dataloader_worker_kwargs(args.num_workers, persistent=shuffle),
                                                    worker_init_fn=worker_init_fn,
                                                    collate_fn=collate_fn_pcn)
        elif "KITTI" in dataset.__class__.__name__:
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.others.bs if shuffle else 1,
                                                    shuffle = shuffle, 
                                                    drop_last = config.others.subset == 'train',
                                                    **dataloader_worker_kwargs(args.num_workers, persistent=shuffle),
                                                    worker_init_fn=worker_init_fn,
                                                    collate_fn=collate_fn_kitti)
        else:
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.others.bs if shuffle else 1,
                                                    shuffle = shuffle, 
                                                    drop_last = config.others.subset == 'train',
                                                    **dataloader_worker_kwargs(args.num_workers, persistent=shuffle),
                                                    worker_init_fn=worker_init_fn,
                                                    collate_fn=collate_fn_projected_shapenet)
    return sampler, dataloader

def model_builder(config):
    model = build_model_from_cfg(config)
    return model

def build_optimizer(base_model, config):
    opti_config = config.optimizer
    if opti_config.type == 'AdamW':
        def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
            decay = []
            no_decay = []
            for name, param in model.module.named_parameters():
                if not param.requires_grad:
                    continue  # frozen weights
                if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
                    no_decay.append(param)
                else:
                    decay.append(param)
            return [
                {'params': no_decay, 'weight_decay': 0.},
                {'params': decay, 'weight_decay': weight_decay}]
        param_groups = add_weight_decay(base_model, weight_decay=opti_config.kwargs.weight_decay)
        optimizer = optim.AdamW(param_groups, **opti_config.kwargs)
    elif opti_config.type == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, base_model.parameters()), **opti_config.kwargs)
    elif opti_config.type == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, base_model.parameters()), **opti_config.kwargs)
    else:
        raise NotImplementedError()

    return optimizer

def build_scheduler(base_model, optimizer, config, last_epoch=-1):
    sche_config = config.scheduler
    if sche_config.type == 'LambdaLR':
        scheduler = build_lambda_sche(optimizer, sche_config.kwargs, last_epoch=last_epoch)  # misc.py
    elif sche_config.type == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, last_epoch=last_epoch, **sche_config.kwargs)
    elif sche_config.type == 'GradualWarmup':
        scheduler_steplr = torch.optim.lr_scheduler.StepLR(optimizer, last_epoch=last_epoch, **sche_config.kwargs_1)
        scheduler = GradualWarmupScheduler(optimizer, after_scheduler=scheduler_steplr, **sche_config.kwargs_2)
    elif sche_config.type == 'MultiStepLR_GradualWarmup':
        scheduler_steplr = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=config.scheduler.LR_DECAY_STEP, gamma=config.scheduler.GAMMA)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=config.scheduler.WARMUP_STEPS,
                                          after_scheduler=scheduler_steplr)
    elif sche_config.type == 'CosLR':
        scheduler = CosineLRScheduler(optimizer,
                t_initial=sche_config.kwargs.t_max,
                lr_min=sche_config.kwargs.min_lr,
                warmup_t=sche_config.kwargs.initial_epochs,
                t_in_epochs=True)
    else:
        raise NotImplementedError()
    
    if config.get('bnmscheduler') is not None:
        bnsche_config = config.bnmscheduler
        if bnsche_config.type == 'Lambda':
            bnscheduler = build_lambda_bnsche(base_model, bnsche_config.kwargs)  # misc.py
        scheduler = [scheduler, bnscheduler]
    
    return scheduler

def resume_model(base_model, args, logger = None):
    ckpt_path = os.path.join(args.experiment_path, 'ckpt-last.pth')
    if not os.path.exists(ckpt_path):
        print_log(f'[RESUME INFO] no checkpoint file from path {ckpt_path}...', logger = logger)
        return 0, 0
    print_log(f'[RESUME INFO] Loading model weights from {ckpt_path}...', logger = logger )

    # load state dict
    map_location = {'cuda:%d' % 0: 'cuda:%d' % args.local_rank}
    state_dict = torch.load(ckpt_path, map_location=map_location)
    # parameter resume of base model
    # if args.local_rank == 0:
    base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}
    base_model.load_state_dict(base_ckpt)

    # parameter
    start_epoch = state_dict['epoch'] + 1
    best_metrics = state_dict['best_metrics']
    if not isinstance(best_metrics, dict):
        best_metrics = best_metrics.state_dict()
    # print(best_metrics)

    print_log(f'[RESUME INFO] resume ckpts @ {start_epoch - 1} epoch( best_metrics = {str(best_metrics):s})', logger = logger)
    return start_epoch, best_metrics

def resume_optimizer(optimizer, args, logger = None):
    ckpt_path = os.path.join(args.experiment_path, 'ckpt-last.pth')
    if not os.path.exists(ckpt_path):
        print_log(f'[RESUME INFO] no checkpoint file from path {ckpt_path}...', logger = logger)
        return 0, 0, 0
    print_log(f'[RESUME INFO] Loading optimizer from {ckpt_path}...', logger = logger )
    # load state dict
    state_dict = torch.load(ckpt_path, map_location='cpu')
    # optimizer
    optimizer.load_state_dict(state_dict['optimizer'])

def resume_scheduler(scheduler, args, logger = None):
    ckpt_path = os.path.join(args.experiment_path, 'ckpt-last.pth')
    if not os.path.exists(ckpt_path):
        return False
    state_dict = torch.load(ckpt_path, map_location='cpu')
    lr_scheduler = scheduler[0] if isinstance(scheduler, (list, tuple)) else scheduler
    lr_scheduler.optimizer.load_state_dict(state_dict['optimizer'])
    scheduler_state = state_dict.get('scheduler')
    if scheduler_state is None:
        print_log('[RESUME INFO] No scheduler state in checkpoint; using epoch-based scheduler fallback.', logger=logger)
        return False
    load_scheduler_state_dict(scheduler, scheduler_state)
    print_log(f'[RESUME INFO] Loading scheduler from {ckpt_path}...', logger=logger)
    return True

def save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, prefix, args, scheduler=None, logger = None):
    if args.local_rank == 0:
        torch.save({
                    'base_model' : base_model.module.state_dict() if args.distributed else base_model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'scheduler' : scheduler_state_dict(scheduler),
                    'epoch' : epoch,
                    'metrics' : metrics.state_dict() if metrics is not None else dict(),
                    'best_metrics' : best_metrics.state_dict() if best_metrics is not None else dict(),
                    }, os.path.join(args.experiment_path, prefix + '.pth'))
        print_log(f"Save checkpoint at {os.path.join(args.experiment_path, prefix + '.pth')}", logger = logger)

def load_model(base_model, ckpt_path, logger = None):
    if not os.path.exists(ckpt_path):
        raise NotImplementedError('no checkpoint file from path %s...' % ckpt_path)
    print_log(f'Loading weights from {ckpt_path}...', logger = logger )

    # load state dict
    state_dict = torch.load(ckpt_path, map_location='cpu')
    # parameter resume of base model
    if state_dict.get('model') is not None:
        base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['model'].items()}
    elif state_dict.get('base_model') is not None:
        base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}
    else:
        raise RuntimeError('mismatch of ckpt weight')

    model_name = type(base_model).__name__
    if model_name == 'FSCSVDAdaMemoryLabelFreeTokenQueryV1':
        checkpoint_memory = base_ckpt.get('memory_vector')
        target_memory = base_model.state_dict()['memory_vector']
        if (
            checkpoint_memory is not None
            and tuple(checkpoint_memory.shape) != tuple(target_memory.shape)
        ):
            checkpoint_shape = tuple(checkpoint_memory.shape)
            target_shape = tuple(target_memory.shape)
            if (
                len(checkpoint_shape) == 2
                and len(target_shape) == 2
                and checkpoint_shape[1] == 1280
                and target_shape[1] == 512
            ):
                mismatch_reason = (
                    'The former 1024-D key checkpoints ([M, 1280]) cannot '
                    'be resumed by the 256-D key model ([M, 512]). '
                )
            else:
                mismatch_reason = (
                    'Checkpoint and config must use the same memory_size '
                    'and key/value dimensions. '
                )
            raise RuntimeError(
                'The FSC AdaMemory checkpoint memory shape '
                f'{checkpoint_shape} does not match the configured v1_0 '
                f'shape {target_shape}. {mismatch_reason}'
                'Load an original FSCSVD checkpoint as the pretrained '
                'backbone, or start a new v1_0 AdaMemory run.'
            )

        legacy_memory_keys = {
            'Encoder.memory_vector',
            'Encoder.gating_alpha',
            'Encoder.gating_beta',
        }
        detected_legacy_keys = sorted(
            legacy_memory_keys.intersection(base_ckpt)
        )
        if detected_legacy_keys:
            raise RuntimeError(
                'The checkpoint contains the obsolete FSC AdaMemory '
                'parameters under Encoder and is incompatible with the '
                'label-free v1_0 plugin. Load an original FSCSVD checkpoint '
                'as the pretrained backbone, or use a checkpoint trained '
                'with the current FSCSVDAdaMemoryLabelFreeTokenQueryV1. '
                'Detected keys: '
                + ', '.join(detected_legacy_keys)
            )

        incompatible = base_model.load_state_dict(base_ckpt, strict=True)
        if incompatible.missing_keys:
            print_log(
                '[LOAD INFO] Initialize FSC v1_0 plugin parameters not present '
                'in the checkpoint: ' + ', '.join(incompatible.missing_keys),
                logger=logger,
            )
        if incompatible.unexpected_keys:
            print_log(
                '[LOAD INFO] Ignore unexpected checkpoint parameters: '
                + ', '.join(incompatible.unexpected_keys),
                logger=logger,
            )
    else:
        base_model.load_state_dict(base_ckpt, strict=True)

    epoch = -1
    if state_dict.get('epoch') is not None:
        epoch = state_dict['epoch']
    if state_dict.get('metrics') is not None:
        metrics = state_dict['metrics']
        if not isinstance(metrics, dict):
            metrics = metrics.state_dict()
    else:
        metrics = 'No Metrics'
    print_log(f'ckpts @ {epoch} epoch( performance = {str(metrics):s})', logger = logger)
    return
