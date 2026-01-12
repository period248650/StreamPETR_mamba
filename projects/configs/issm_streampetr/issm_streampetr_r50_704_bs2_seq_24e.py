# ------------------------------------------------------------------------
# ISSM-StreamPETR Configuration Example
# ------------------------------------------------------------------------

_base_ = [
    '../../../configs/_base_/default_runtime.py'
]

# 模型配置
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

# 点云范围
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], 
    std=[57.375, 57.120, 58.395], 
    to_rgb=False
)

# 输入配置
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False
)

# 模型
model = dict(
    type='Petr3D',
    use_grid_mask=True,
    
    # 图像编码器
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        pretrained='torchvision://resnet50'
    ),
    
    img_neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=4,
        relu_before_extra_convs=True
    ),
    
    # ISSM-StreamPETR 检测头（核心改进）
    pts_bbox_head=dict(
        type='ISSMStreamPETRHead',
        num_classes=10,  # nuScenes 10 类
        in_channels=256,
        embed_dims=256,
        num_query=900,
        num_reg_fcs=2,
        
        # Memory 配置（简化后只需 num_propagated）
        num_propagated=256,
        
        # ISSM 特定配置
        use_issm=True,
        issm_num_heads=8,          # 多头数量
        
        # Transformer 配置（如果不用 ISSM）
        transformer=dict(
            type='PETRTransformer',
            decoder=dict(
                type='PETRTransformerDecoder',
                return_intermediate=True,
                num_layers=6,
                transformerlayers=dict(
                    type='PETRTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1
                        )
                    ],
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')
                )
            )
        ),
        
        # 位置编码
        positional_encoding=dict(
            type='SinePositionalEncoding3D',
            num_feats=128,
            normalize=True
        ),
        
        # 损失函数
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0
        ),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0),
        
        # BBox 编码器
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=10
        ),
        
        # 其他配置
        with_ego_pos=True,
        match_with_velo=True,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
        match_costs=[2.0, 0.25, 0.0]
    ),
    
    # 训练和测试配置
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
                pc_range=point_cloud_range
            )
        )
    ),
    
    test_cfg=dict(
        pts=dict(
            pc_range=point_cloud_range[:2],
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=4,
            voxel_size=voxel_size[:2],
            nms_type='rotate',
            pre_max_size=1000,
            post_max_size=83,
            nms_thr=0.2
        )
    )
)

# 数据配置
dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
              'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
              'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']),
    dict(type='Collect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'])
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='MultiScaleFlipAug3D',
         img_scale=(1600, 900),
         pts_scale_ratio=1,
         flip=False,
         transforms=[
             dict(type='DefaultFormatBundle3D', class_names=['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
              'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'], with_label=False),
             dict(type='Collect3D', keys=['img'])
         ])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_temporal_train.pkl',
        pipeline=train_pipeline,
        classes=['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
              'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'],
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        box_type_3d='LiDAR'
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_temporal_val.pkl',
        pipeline=test_pipeline,
        classes=['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
              'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'],
        modality=input_modality,
        test_mode=True
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_temporal_val.pkl',
        pipeline=test_pipeline,
        classes=['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
              'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'],
        modality=input_modality,
        test_mode=True
    ),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

# 优化器
optimizer = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01
)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# 学习率调度
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3
)

# 运行配置
total_epochs = 24
evaluation = dict(interval=24, pipeline=test_pipeline)
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ]
)

checkpoint_config = dict(interval=1)
