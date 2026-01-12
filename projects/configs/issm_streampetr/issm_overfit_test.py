# ------------------------------------------------------------------------
# Copyright (c) 2024 ISSM-StreamPETR. All Rights Reserved.
# ------------------------------------------------------------------------
"""
过拟合测试配置

目的：验证 ISSM-StreamPETR 代码是否有 bug
- 使用验证集作为训练集和测试集
- 关闭所有数据增强
- 使用较大学习率
- 只使用少量样本
- 如果模型能过拟合（loss 趋近 0，mAP 趋近 100%），说明代码正确

使用方法：
1. 单 GPU 测试（推荐，方便调试）:
   python tools/train.py projects/configs/issm_streampetr/issm_overfit_test.py --gpu-ids 0

2. 多 GPU 测试:
   ./tools/dist_train.sh projects/configs/issm_streampetr/issm_overfit_test.py 2
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

# ============================================================================
# 过拟合测试关键设置
# ============================================================================
num_gpus = 1  # 单 GPU 测试，方便调试
batch_size = 1  # 小 batch size
num_samples = 50  # 只使用 50 个样本（约 10 个场景）
num_iters_per_epoch = num_samples // batch_size  # 每 epoch 迭代次数
num_epochs = 200  # 足够多的 epoch 让模型过拟合
eval_interval = num_iters_per_epoch * 10  # 每 10 个 epoch 评估一次

queue_length = 1
num_frame_losses = 1
collect_keys = ['lidar2img', 'intrinsics', 'extrinsics', 'timestamp', 'img_timestamp', 'ego_pose', 'ego_pose_inv']

# ISSM 参数
issm_num_heads = 8
issm_chunk_size = 256

# 模型配置（与正式配置相同）
model = dict(
    type='ISSMStreamPETR',
    num_frame_head_grads=num_frame_losses,
    num_frame_backbone_grads=num_frame_losses,
    num_frame_losses=num_frame_losses,
    use_grid_mask=False,  # 【关闭】Grid Mask 数据增强
    
    # === 图像骨干网络 ===
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
        with_cp=False,
        style='pytorch'),
    
    # === 图像颈部网络 ===
    img_neck=dict(
        type='CPFPN',
        in_channels=[1024, 2048],
        out_channels=256,
        num_outs=2),
    
    # === 2D 检测头 ===
    img_roi_head=dict(
        type='FocalHead',
        num_classes=10,
        in_channels=256,
        loss_cls2d=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=2.0),
        loss_centerness=dict(type='GaussianFocalLoss', reduction='mean', loss_weight=1.0),
        loss_bbox2d=dict(type='L1Loss', loss_weight=5.0),
        loss_iou2d=dict(type='GIoULoss', loss_weight=2.0),
        loss_centers2d=dict(type='L1Loss', loss_weight=10.0),
        train_cfg=dict(
            assigner2d=dict(
                type='HungarianAssigner2D',
                cls_cost=dict(type='FocalLossCost', weight=2.),
                reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0),
                centers2d_cost=dict(type='BBox3DL1Cost', weight=10.0)))),
    
    # === ISSM-StreamPETR 检测头 ===
    pts_bbox_head=dict(
        type='ISSMStreamPETRHead',
        num_classes=10,
        in_channels=256,
        num_query=192,
        num_propagated=64,
        
        # StreamPETR 设置
        with_ego_pos=True,
        match_with_velo=False,
        LID=True,
        with_position=True,
        position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        code_weights=[2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        
        # ISSM 核心参数
        use_issm=True,
        issm_num_heads=issm_num_heads,
        
        # ISSM 解码器配置
        issm_decoder=dict(
            type='DenseAlternatingISSMDecoder',
            num_layers=6,
            d_model=256,
            num_heads=issm_num_heads,
            chunk_size=issm_chunk_size,
            dropout=0.0,  # 【关闭】Dropout，帮助过拟合
        ),
        
        # Bbox 编码器
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=10),
        
        # 损失函数
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0),
    ),
    
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

# 【关键】关闭所有数据增强
ida_aug_conf = {
    "resize_lim": (0.53, 0.53),  # 固定缩放比例，不随机
    "final_dim": (256, 768),
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (0.0, 0.0),  # 无旋转
    "H": 900,
    "W": 1600,
    "rand_flip": False,  # 【关闭】随机翻转
}

# 【关键】简化训练 pipeline，关闭数据增强
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_bbox=True,
        with_label=True, with_bbox_depth=True),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='ResizeCropFlipRotImage', data_aug_conf=ida_aug_conf, training=False),  # 【关键】training=False
    # 【移除】GlobalRotScaleTransImage（3D 数据增强）
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

# 【关键】使用小型过拟合数据集
# 首先运行: python tools/create_overfit_subset.py --num-samples 50
# 这会创建 data/nuscenes/nuscenes2d_temporal_infos_overfit.pkl
data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes2d_temporal_infos_overfit.pkl',  # 【使用小型过拟合数据集】
        num_frame_losses=num_frame_losses,
        seq_split_num=1,  # 不分割序列
        seq_mode=True,
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        collect_keys=collect_keys + ['img', 'prev_exists', 'img_metas'],
        queue_length=queue_length,
        test_mode=False,
        use_valid_flag=True,
        filter_empty_gt=False,
        box_type_3d='LiDAR'),
    val=dict(
        type=dataset_type, 
        pipeline=test_pipeline, 
        collect_keys=collect_keys + ['img', 'img_metas'], 
        queue_length=queue_length, 
        ann_file=data_root + 'nuscenes2d_temporal_infos_overfit.pkl',  # 【同一个小型数据集】
        classes=class_names, 
        modality=input_modality),
    test=dict(
        type=dataset_type, 
        pipeline=test_pipeline, 
        collect_keys=collect_keys + ['img', 'img_metas'], 
        queue_length=queue_length, 
        ann_file=data_root + 'nuscenes2d_temporal_infos_overfit.pkl',  # 【同一个小型数据集】
        classes=class_names, 
        modality=input_modality),
    shuffler_sampler=dict(type='InfiniteGroupEachSampleInBatchSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

# === 优化器配置 ===
# 【关键】使用较大学习率帮助过拟合
optimizer = dict(
    type='AdamW',
    lr=2e-4,  # 较大的学习率
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.0)  # 【关闭】权重衰减，帮助过拟合

optimizer_config = dict(
    type='Fp16OptimizerHook', 
    loss_scale='dynamic', 
    grad_clip=dict(max_norm=35, norm_type=2)
)

# === 学习率调度 ===
# 【简单的学习率调度】
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=0.001,
    step=[num_iters_per_epoch * 100, num_iters_per_epoch * 150],  # 在 100 和 150 epoch 时降低学习率
    gamma=0.1
)

# === 评估配置 ===
# 每 10 个 epoch 评估一次
evaluation = dict(interval=eval_interval, pipeline=test_pipeline)

# === 运行时设置 ===
find_unused_parameters = True
checkpoint_config = dict(interval=num_iters_per_epoch * 20, max_keep_ckpts=5)  # 每 20 epoch 保存一次
runner = dict(type='IterBasedRunner', max_iters=num_epochs * num_iters_per_epoch)
load_from = None
resume_from = None

# 日志配置 - 【关键】更频繁地打印日志，方便观察 loss 变化
log_config = dict(
    interval=10,  # 每 10 次迭代打印一次
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

custom_hooks = [dict(type='SetEpochInfoHook')]

# ============================================================================
# 过拟合实验预期结果
# ============================================================================
# 
# 如果代码正确：
#   - loss_cls 应该从 ~1.0 降到 < 0.1
#   - loss_bbox 应该从 ~1.0 降到 < 0.01
#   - 训练集上的 mAP 应该逐渐上升到 > 50%（理想情况接近 100%）
#
# 如果代码有 bug：
#   - loss 可能不下降或下降极慢
#   - loss 可能震荡不稳定
#   - mAP 停留在很低的水平（< 5%）
#
# 调试建议：
#   1. 首先观察前 100 次迭代的 loss 变化趋势
#   2. 如果 loss_bbox 不下降，检查 anchor 更新逻辑
#   3. 如果 loss_cls 不下降，检查分类头
#   4. 使用 TensorBoard 可视化 loss 曲线
# ============================================================================
