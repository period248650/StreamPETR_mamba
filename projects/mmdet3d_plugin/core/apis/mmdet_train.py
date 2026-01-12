# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
# ---------------------------------------------
#  Modified by Shihao Wang
# ---------------------------------------------
import random
import warnings

import numpy as np
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (HOOKS, DistSamplerSeedHook, EpochBasedRunner,
                         Fp16OptimizerHook, OptimizerHook, build_optimizer,
                         build_runner, get_dist_info)
from mmcv.utils import build_from_cfg

# ============================================================
# 猴子补丁：修复 mmcv 与新版 PyTorch 的兼容性问题
# 问题：mmcv 的 scatter 函数传递整数 device_ids，但新版 PyTorch 期望 torch.device
# ============================================================
def _patch_mmcv_scatter():
    """修复 mmcv scatter 函数与 PyTorch 的兼容性问题"""
    try:
        from mmcv.parallel import scatter_gather
        
        _original_scatter = scatter_gather.scatter
        
        def _patched_scatter(inputs, target_gpus, dim=0):
            """将整数 GPU IDs 转换为 torch.device 对象"""
            if target_gpus and isinstance(target_gpus[0], int):
                target_gpus = [torch.device('cuda', gpu) for gpu in target_gpus]
            return _original_scatter(inputs, target_gpus, dim)
        
        scatter_gather.scatter = _patched_scatter
        
        # 同时修复 scatter_kwargs
        _original_scatter_kwargs = scatter_gather.scatter_kwargs
        
        def _patched_scatter_kwargs(inputs, kwargs, target_gpus, dim=0):
            """将整数 GPU IDs 转换为 torch.device 对象"""
            if target_gpus and isinstance(target_gpus[0], int):
                target_gpus = [torch.device('cuda', gpu) for gpu in target_gpus]
            return _original_scatter_kwargs(inputs, kwargs, target_gpus, dim)
        
        scatter_gather.scatter_kwargs = _patched_scatter_kwargs
        
    except Exception as e:
        warnings.warn(f"Failed to patch mmcv scatter: {e}")

def _patch_pytorch_parallel():
    """修复 PyTorch nn.parallel._functions 与整数设备 ID 的兼容性问题"""
    try:
        from torch.nn.parallel import _functions as parallel_functions
        
        _original_get_stream = parallel_functions._get_stream
        
        def _patched_get_stream(device):
            """确保 device 是 torch.device 对象"""
            if isinstance(device, int):
                device = torch.device('cuda', device)
            return _original_get_stream(device)
        
        parallel_functions._get_stream = _patched_get_stream
        
    except Exception as e:
        warnings.warn(f"Failed to patch PyTorch parallel functions: {e}")

# 应用补丁
_patch_mmcv_scatter()
_patch_pytorch_parallel()
# ============================================================

from mmdet.core import EvalHook

from mmdet.datasets import (build_dataset,
                            replace_ImageToTensor)
from mmdet.utils import get_root_logger
import time
import os.path as osp
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from projects.mmdet3d_plugin.core.evaluation.eval_hooks import CustomDistEvalHook
from projects.mmdet3d_plugin.datasets import custom_build_dataset
def custom_train_detector(model,
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   timestamp=None,
                   eval_model=None,
                   meta=None):
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
   
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    #assert len(dataset)==1s
    if 'imgs_per_gpu' in cfg.data:
        logger.warning('"imgs_per_gpu" is deprecated in MMDet V2.0. '
                       'Please use "samples_per_gpu" instead')
        if 'samples_per_gpu' in cfg.data:
            logger.warning(
                f'Got "imgs_per_gpu"={cfg.data.imgs_per_gpu} and '
                f'"samples_per_gpu"={cfg.data.samples_per_gpu}, "imgs_per_gpu"'
                f'={cfg.data.imgs_per_gpu} is used in this experiments')
        else:
            logger.warning(
                'Automatically set "samples_per_gpu"="imgs_per_gpu"='
                f'{cfg.data.imgs_per_gpu} in this experiments')
        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu

    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            shuffler_sampler=cfg.data.shuffler_sampler,  # dict(type='DistributedGroupSampler'),
            nonshuffler_sampler=cfg.data.nonshuffler_sampler,  # dict(type='DistributedSampler'),
            runner_type=cfg.runner,
        ) for ds in dataset
    ]

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        current_device = torch.cuda.current_device()
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[current_device],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
        if eval_model is not None:
            eval_model = MMDistributedDataParallel(
                eval_model.cuda(),
                device_ids=[current_device],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
    else:
        # 修复：确保 gpu_ids 是正确的设备索引
        if hasattr(cfg, 'gpu_ids') and len(cfg.gpu_ids) > 0:
            gpu_id = cfg.gpu_ids[0] if isinstance(cfg.gpu_ids[0], int) else 0
        else:
            gpu_id = 0
        # 修复：使用 torch.device 对象而不是整数，避免 PyTorch 兼容性问题
        device = torch.device('cuda', gpu_id)
        model = MMDataParallel(
            model.to(device), device_ids=[gpu_id])
        if eval_model is not None:
            eval_model = MMDataParallel(
                eval_model.to(device), device_ids=[gpu_id])


    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    if 'runner' not in cfg:
        cfg.runner = {
            'type': 'EpochBasedRunner',
            'max_epochs': cfg.total_epochs
        }
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)
    else:
        if 'total_epochs' in cfg:
            assert cfg.total_epochs == cfg.runner.max_epochs
    if eval_model is not None:
        runner = build_runner(
            cfg.runner,
            default_args=dict(
                model=model,
                eval_model=eval_model,
                optimizer=optimizer,
                work_dir=cfg.work_dir,
                logger=logger,
                meta=meta))
    else:
        runner = build_runner(
            cfg.runner,
            default_args=dict(
                model=model,
                optimizer=optimizer,
                work_dir=cfg.work_dir,
                logger=logger,
                meta=meta))

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))
    
    # register profiler hook
    #trace_config = dict(type='tb_trace', dir_name='work_dir')
    #profiler_config = dict(on_trace_ready=trace_config)
    #runner.register_profiler_hook(profiler_config)
    
    if distributed:
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        # Support batch_size > 1 in validation
        val_samples_per_gpu = cfg.data.val.pop('samples_per_gpu', 1)
        if val_samples_per_gpu > 1:
            assert False
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(
                cfg.data.val.pipeline)
        val_dataset = custom_build_dataset(cfg.data.val, dict(test_mode=True))

        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=val_samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False,
            shuffler_sampler=cfg.data.shuffler_sampler,  # dict(type='DistributedGroupSampler'),
            nonshuffler_sampler=cfg.data.nonshuffler_sampler,  # dict(type='DistributedSampler'),
        )
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_cfg['jsonfile_prefix'] = osp.join('val', cfg.work_dir, time.ctime().replace(' ','_').replace(':','_'))
        eval_hook = CustomDistEvalHook if distributed else EvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    # user-defined hooks
    if cfg.get('custom_hooks', None):
        custom_hooks = cfg.custom_hooks
        assert isinstance(custom_hooks, list), \
            f'custom_hooks expect list type, but got {type(custom_hooks)}'
        for hook_cfg in cfg.custom_hooks:
            assert isinstance(hook_cfg, dict), \
                'Each item in custom_hooks expects dict type, but got ' \
                f'{type(hook_cfg)}'
            hook_cfg = hook_cfg.copy()
            priority = hook_cfg.pop('priority', 'NORMAL')
            hook = build_from_cfg(hook_cfg, HOOKS)
            runner.register_hook(hook, priority=priority)

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow)

