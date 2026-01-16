# ------------------------------------------------------------------------
# Copyright (c) 2024 ISSM-StreamPETR. All Rights Reserved.
# ------------------------------------------------------------------------
"""
配置文件：具有密集特征聚合的交替扫描 ISSM-StreamPETR
(Dense Alternating ISSM-StreamPETR)

核心改进：
1. 单向扫描：每层只执行一次 SSM，计算开销降低 50%
2. 层间交替：奇数层用模式 A，偶数层用模式 B
3. 密集连接：F_L = F_{L-1} + F_{L-2}（类似 DenseNet）
4. 全局感知：通过跨层融合实现隐式双向信息流

相比原始 ISSM-StreamPETR 的优势：
- 计算效率：FLOPs 减少约 40-50%
- 信息流动：更好的长距离依赖建模
- 拓扑鲁棒性：不同层看到不同的扫描顺序
"""

_base_ = [
    '../../../mmdetection3d/configs/_base_/datasets/nus-3d.py',
    '../../../mmdetection3d/configs/_base_/default_runtime.py'
]

plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

# Model 设置
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)


num_views = 6

# 训练配置
num_gpus = 8
batch_size = 2
num_iters_per_epoch = 28130 // (num_gpus * batch_size)
num_epochs = 60
queue_length = 1
num_frame_losses = 1
collect_keys = ['lidar2img', 'intrinsics', 'extrinsics', 'timestamp', 'img_timestamp', 'ego_pose', 'ego_pose_inv']


issm_num_heads = 8
issm_chunk_size = 256

# 模型配置
model = dict(
    type='ISSMStreamPETR',  # 使用改进的 ISSM 版本
    num_frame_head_grads=num_frame_losses,
    num_frame_backbone_grads=num_frame_losses,
    num_frame_losses=num_frame_losses,
    use_grid_mask=True,
    
    # === 图像骨干网络（与原始 StreamPETR 保持一致）===
    img_backbone=dict(
        init_cfg=dict(
            type='Pretrained', 
            checkpoint="ckpts/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth",
            prefix='backbone.'),       
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        with_cp=False,  # 【v7 修复】禁用 checkpoint，避免与 DDP 冲突
        style='pytorch'),
    
    # === 图像颈部网络（与原始 StreamPETR 保持一致）===
    img_neck=dict(
        type='CPFPN',
        in_channels=[1024, 2048],
        out_channels=256,
        num_outs=2),
    
    # === 2D 检测头（与原始 StreamPETR 保持一致）===
    img_roi_head=dict(
        type='FocalHead',
        num_classes=10,
        in_channels=256,
        loss_cls2d=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=2.0),  # 【恢复原版】
        loss_centerness=dict(type='GaussianFocalLoss', reduction='mean', loss_weight=1.0),  # 【恢复原版】
        loss_bbox2d=dict(type='L1Loss', loss_weight=5.0),  # 【恢复原版】
        loss_iou2d=dict(type='GIoULoss', loss_weight=2.0),  # 【恢复原版】
        loss_centers2d=dict(type='L1Loss', loss_weight=10.0),  # 【恢复原版】
        train_cfg=dict(
            assigner2d=dict(
                type='HungarianAssigner2D',
                cls_cost=dict(type='FocalLossCost', weight=2.),
                reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0),
                centers2d_cost=dict(type='BBox3DL1Cost', weight=10.0)))),
    
    # === ISSM-StreamPETR 检测头（改进版）===
    pts_bbox_head=dict(
        type='ISSMStreamPETRHead',  # 使用改进的 ISSM 版本
        num_classes=10,
        in_channels=256,
        num_query=192,  # 基础 query 数量（优化后）
        num_propagated=64,  # 时序传播 query 数量
        
        # StreamPETR 设置
        with_ego_pos=True,
        match_with_velo=False,
        LID=True,
        with_position=True,
        position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        code_weights=[2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        
        # === ISSM 核心参数 ===
        use_issm=True,
        issm_num_heads=issm_num_heads,
        
        # ISSM 解码器配置
        issm_decoder=dict(
            type='DenseAlternatingISSMDecoder',
            num_layers=6,  # 解码器层数
            d_model=256,
            num_heads=issm_num_heads,
            chunk_size=issm_chunk_size,
            dropout=0.1,
        ),
        
        # Bbox 编码器（与原始 StreamPETR 保持一致）
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=10),
        
        # 损失函数（与原始 StreamPETR 保持一致）
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0),
    ),
    
    # === 训练和测试配置（与原始 StreamPETR 保持一致）===
    train_cfg=dict(
        pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_size_factor=4,
            assigner=dict(
                type='HungarianAssigner3D',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                iou_cost=dict(type='IoUCost', weight=0.0),
                pc_range=point_cloud_range))),
    
    test_cfg=dict(pts=dict(max_per_img=300))
)

# === 数据集配置 ===
dataset_type = 'CustomNuScenesDataset'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

# 数据增强配置
ida_aug_conf = {
    "resize_lim": (0.48, 0.58),
    "final_dim": (256, 768),
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (0.0, 0.0),
    "H": 900,
    "W": 1600,
    "rand_flip": True,
}

# 用于 StreamPETR 的收集键
# collect_keys = ['lidar2img', 'intrinsics', 'extrinsics', 'timestamp', 'img_timestamp', 'ego_pose', 'ego_pose_inv']

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_bbox=True,
        with_label=True, with_bbox_depth=True),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='ResizeCropFlipRotImage', data_aug_conf=ida_aug_conf, training=True),
    dict(type='GlobalRotScaleTransImage',
        rot_range=[-0.3925, 0.3925],
        translation_std=[0, 0, 0],
        scale_ratio_range=[0.95, 1.05],
        reverse_angle=True,
        training=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='PETRFormatBundle3D', class_names=class_names, collect_keys=collect_keys + ['prev_exists']),
    dict(type='Collect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'gt_bboxes', 'gt_labels', 'centers2d', 'depths', 'prev_exists'] + collect_keys,
        meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'scene_token', 'gt_bboxes_3d', 'gt_labels_3d'))
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='ResizeCropFlipRotImage', data_aug_conf=ida_aug_conf, training=False),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='PETRFormatBundle3D',
                collect_keys=collect_keys,
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['img'] + collect_keys,
                meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'scene_token'))
        ])
]

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes2d_temporal_infos_train.pkl',
        num_frame_losses=num_frame_losses,
        seq_split_num=2,  # streaming video training
        seq_mode=True,  # streaming video training
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        collect_keys=collect_keys + ['img', 'prev_exists', 'img_metas'],
        queue_length=queue_length,
        test_mode=False,
        use_valid_flag=True,
        filter_empty_gt=False,
        box_type_3d='LiDAR'),
    val=dict(type=dataset_type, pipeline=test_pipeline, collect_keys=collect_keys + ['img', 'img_metas'], queue_length=queue_length, ann_file=data_root + 'nuscenes2d_temporal_infos_val.pkl', classes=class_names, modality=input_modality),
    test=dict(type=dataset_type, pipeline=test_pipeline, collect_keys=collect_keys + ['img', 'img_metas'], queue_length=queue_length, ann_file=data_root + 'nuscenes2d_temporal_infos_val.pkl', classes=class_names, modality=input_modality),
    shuffler_sampler=dict(type='InfiniteGroupEachSampleInBatchSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

# === 优化器配置 ===
optimizer = dict(
    type='AdamW',
    lr=4e-4,  # bs 8: 2e-4 || bs 16: 4e-4（与原版一致）
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),  # 预训练backbone用小学习率
            # ISSM 使用正常学习率，让模型能够学习
        }),
    weight_decay=0.01)

optimizer_config = dict(type='Fp16OptimizerHook', loss_scale='dynamic', grad_clip=dict(max_norm=35, norm_type=2))  # 【恢复】梯度裁剪阈值

# === 学习率调度（与原始 StreamPETR 保持一致）===
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)

# === 评估配置 ===
evaluation = dict(interval=num_iters_per_epoch*num_epochs, pipeline=test_pipeline)

# === 运行时设置（与原始 StreamPETR 保持一致）===
# 注意：ISSM fallback 实现需要 find_unused_parameters=True
# 因为分块近似算法可能导致某些参数的梯度路径不同
find_unused_parameters = True
checkpoint_config = dict(interval=num_iters_per_epoch, max_keep_ckpts=3)
runner = dict(type='IterBasedRunner', max_iters=num_epochs * num_iters_per_epoch)
load_from = None
resume_from = None
# workflow = [('train', 1)]

# 日志配置
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

# 实验说明
custom_hooks = [dict(type='SetEpochInfoHook')]




