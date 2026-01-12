# ------------------------------------------------------------------------
# Copyright (c) 2024 ISSM-StreamPETR. All Rights Reserved.
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.runner import force_fp32
from mmdet.core import (build_assigner, build_sampler, multi_apply, reduce_mean)
from mmdet.models import HEADS, build_loss
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models.utils import NormedLinear
from mmdet3d.core.bbox.coders import build_bbox_coder
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox

from projects.mmdet3d_plugin.models.utils.positional_encoding import pos2posemb3d, pos2posemb1d, nerf_positional_encoding
from projects.mmdet3d_plugin.models.utils.misc import MLN, topk_gather, transform_reference_points, memory_refresh, SELayer_Linear
from projects.mmdet3d_plugin.models.utils.petr_issm import DenseAlternatingISSMDecoder


@HEADS.register_module()
class ISSMStreamPETRHead(AnchorFreeHead): 
    def __init__(
        self,
        num_classes,
        in_channels=256,
        stride=16,
        embed_dims=256,
        num_query=100,
        num_reg_fcs=2,
        num_propagated=64,  # 传播到下一帧的 query 数量
        with_ego_pos=True,
        match_with_velo=True,
        match_costs=None,
        transformer=None,
        sync_cls_avg_factor=False,
        code_weights=None,
        bbox_coder=None,
        loss_cls=dict(
            type='CrossEntropyLoss',
            bg_cls_weight=0.1,
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0),
        train_cfg=dict(
            assigner=dict(
                type='HungarianAssigner3D',
                cls_cost=dict(type='ClassificationCost', weight=1.),
                reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0)),),
        test_cfg=dict(max_per_img=100),
        depth_step=0.8,
        depth_num=64,
        LID=False,
        depth_start=1,
        position_range=[-65, -65, -8.0, 65, 65, 8.0],
        normedlinear=False,
        # ISSM 特定参数
        use_issm=True,
        issm_num_heads=8,
        init_cfg=None,
        **kwargs
    ):
        # 必须在 super().__init__() 之前设置的属性
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]

        self.code_weights = self.code_weights[:self.code_size]

        if match_costs is not None:
            self.match_costs = match_costs
        else:
            self.match_costs = self.code_weights

        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor
        class_weight = loss_cls.get('class_weight', None)
        if class_weight is not None and (self.__class__ is ISSMStreamPETRHead):
            assert isinstance(class_weight, float), 'Expected ' \
                'class_weight to have type float. Found ' \
                f'{type(class_weight)}.'
            bg_cls_weight = loss_cls.get('bg_cls_weight', class_weight)
            assert isinstance(bg_cls_weight, float), 'Expected ' \
                'bg_cls_weight to have type float. Found ' \
                f'{type(bg_cls_weight)}.'
            class_weight = torch.ones(num_classes + 1) * class_weight
            class_weight[num_classes] = bg_cls_weight
            loss_cls.update({'class_weight': class_weight})
            if 'bg_cls_weight' in loss_cls:
                loss_cls.pop('bg_cls_weight')
            self.bg_cls_weight = bg_cls_weight

        if train_cfg:
            assert 'assigner' in train_cfg, 'assigner should be provided ' \
                'when train_cfg is set.'
            assigner = train_cfg['assigner']
            self.assigner = build_assigner(assigner)
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

        self.num_query = num_query
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_propagated = num_propagated
        self.with_ego_pos = with_ego_pos
        self.match_with_velo = match_with_velo
        self.num_reg_fcs = num_reg_fcs
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.embed_dims = embed_dims
        self.depth_step = depth_step
        self.depth_num = depth_num
        self.position_dim = depth_num * 3
        self.LID = LID
        self.depth_start = depth_start
        self.stride = stride

        self.act_cfg = transformer.get('act_cfg', 
                                       dict(type='ReLU', inplace=True)) if transformer else dict(type='ReLU', inplace=True)
        self.num_pred = 6
        self.normedlinear = normedlinear
        
        # ISSM 参数
        self.use_issm = use_issm
        self.issm_num_heads = issm_num_heads
        
        super(ISSMStreamPETRHead, self).__init__(num_classes, in_channels, init_cfg=kwargs.get('init_cfg', None))

        # 在 super().__init__() 之后继续初始化
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_iou = build_loss(loss_iou)

        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights), requires_grad=False)
        self.match_costs = nn.Parameter(torch.tensor(
            self.match_costs), requires_grad=False)
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = nn.Parameter(torch.tensor(
            self.bbox_coder.pc_range), requires_grad=False)
        self.position_range = nn.Parameter(torch.tensor(
            position_range), requires_grad=False)
        
        # 深度坐标编码
        if self.LID:
            index = torch.arange(start=0, end=self.depth_num, step=1).float()
            index_1 = index + 1
            bin_size = (self.position_range[3] - self.depth_start) / (self.depth_num * (1 + self.depth_num))
            coords_d = self.depth_start + bin_size * index * index_1
        else:
            index = torch.arange(start=0, end=self.depth_num, step=1).float()
            bin_size = (self.position_range[3] - self.depth_start) / self.depth_num
            coords_d = self.depth_start + bin_size * index

        self.coords_d = nn.Parameter(coords_d, requires_grad=False)

        self._init_layers()
        self.reset_memory()

    def _init_layers(self):
        """初始化网络层"""
        
        # === 分类和回归头 ===
        self.num_pred = 6 if self.use_issm else 6  # decoder 层数
        
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        if self.normedlinear:
            cls_branch.append(NormedLinear(self.embed_dims, self.cls_out_channels))
        else:
            cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        self.cls_branches = nn.ModuleList([fc_cls for _ in range(self.num_pred)])
        self.reg_branches = nn.ModuleList([reg_branch for _ in range(self.num_pred)])

        # === 位置编码 ===
        self.position_encoder = nn.Sequential(
            nn.Linear(self.position_dim, self.embed_dims * 4),
            nn.ReLU(),
            nn.Linear(self.embed_dims * 4, self.embed_dims),
        )

        self.memory_embed = nn.Sequential(
            nn.Linear(self.in_channels, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )
        
        self.featurized_pe = SELayer_Linear(self.embed_dims)

        # === Query 初始化 ===
        self.reference_points = nn.Embedding(self.num_query, 3)
        if self.num_propagated > 0:
            self.pseudo_reference_points = nn.Embedding(self.num_propagated, 3)

        self.query_embedding = nn.Sequential(
            nn.Linear(self.embed_dims * 3 // 2, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )

        self.spatial_alignment = MLN(8)
        self.time_embedding = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims)
        )

        # === Ego Pose 编码 ===
        if self.with_ego_pos:
            self.ego_pose_pe = MLN(180)
            self.ego_pose_memory = MLN(180)

        # === ISSM 解码器（核心改进）===
        if self.use_issm:
            # 【修复】在 __init__ 中立即初始化 issm_decoder，避免 DDP 未使用参数问题
            self.issm_decoder = DenseAlternatingISSMDecoder(
                num_layers=self.num_pred,
                d_model=self.embed_dims,
                num_heads=self.issm_num_heads,
            )

    def init_weights(self):
        """初始化权重"""
        nn.init.uniform_(self.reference_points.weight.data, 0, 1)
        if self.num_propagated > 0:
            nn.init.uniform_(self.pseudo_reference_points.weight.data, 0, 1)
            self.pseudo_reference_points.weight.requires_grad = False
        # 注意：issm_decoder 已在 __init__ 中初始化，不再重置

    def reset_memory(self):
        """重置 Memory Queue"""
        self.memory_embedding = None
        self.memory_reference_point = None
        self.memory_timestamp = None
        self.memory_egopose = None
        self.memory_velo = None

    def pre_update_memory(self, data):
        """更新记忆队列（训练前/测试时）
        
        简化版：只保留 num_propagated 个 query 用于传播
        """
        # 获取 batch_size 和 device
        if 'prev_exists' in data:
            x = data['prev_exists']
            B = x.size(0)
            device = x.device
        else:
            # 测试模式：从 img_feats 获取 batch_size
            img_feats = data.get('img_feats')
            if img_feats is not None:
                B = img_feats.size(0)
                device = img_feats.device
            else:
                raise ValueError("Either 'prev_exists' or 'img_feats' must be provided")
            # 测试时默认为新场景的第一帧 (prev_exists = 0)
            x = torch.zeros(B, 1, device=device)
        
        # 简化：memory 大小 = num_propagated
        N_prop = self.num_propagated
        
        # 初始化 memory（仅 num_propagated 大小）
        if self.memory_embedding is None:
            self.memory_embedding = x.new_zeros(B, N_prop, self.embed_dims)
            self.memory_reference_point = x.new_zeros(B, N_prop, 3)
            self.memory_timestamp = x.new_zeros(B, N_prop, 1)
            self.memory_egopose = x.new_zeros(B, N_prop, 4, 4)
            self.memory_velo = x.new_zeros(B, N_prop, 2)
        else:
            # 更新 memory（坐标变换、时间戳更新）
            if 'timestamp' in data and 'ego_pose_inv' in data:
                self.memory_timestamp += data['timestamp'].unsqueeze(-1).unsqueeze(-1)
                self.memory_egopose = data['ego_pose_inv'].unsqueeze(1) @ self.memory_egopose
                self.memory_reference_point = transform_reference_points(self.memory_reference_point, data['ego_pose_inv'], reverse=False)
            # 场景切换时清空 memory
            self.memory_timestamp = memory_refresh(self.memory_timestamp, x)
            self.memory_reference_point = memory_refresh(self.memory_reference_point, x)
            self.memory_embedding = memory_refresh(self.memory_embedding, x)
            self.memory_egopose = memory_refresh(self.memory_egopose, x)
            self.memory_velo = memory_refresh(self.memory_velo, x)
        
        # 首帧填充 pseudo_reference_points（非可学习）
        if self.num_propagated > 0:
            pseudo_reference_points = self.pseudo_reference_points.weight * (self.pc_range[3:6] - self.pc_range[0:3]) + self.pc_range[0:3]
            self.memory_reference_point = self.memory_reference_point + (1 - x).view(B, 1, 1) * pseudo_reference_points
            self.memory_egopose = self.memory_egopose + (1 - x).view(B, 1, 1, 1) * torch.eye(4, device=device)

    @force_fp32(apply_to=('mlvl_feats'))
    def forward(self, memory_center, img_metas, **data):
        """
        前向传播
        
        Args:
            memory_center: 特征中心点坐标 [B*N, H, W, 2]
            img_metas: 图像元信息
            **data: 其他数据（img_feats, ego_pose, intrinsics 等）
        
        Returns:
            outs: 包含 all_cls_scores 和 all_bbox_preds 的字典
            
        ISSM 工作流程：
        1. 使用完整特征序列（N*H*W），不做特征采样
        2. 每层进行随机重排，打破固定视图顺序
        3. SSM 扫描处理重排后的序列
        4. 还原原始顺序后进行密集聚合
        """
        # 更新 Memory Bank
        self.pre_update_memory(data)
        
        # 提取特征
        x = data['img_feats']
        B, N, C, H, W = x.shape
        num_tokens = N * H * W
        
        # === 1. 生成位置编码和几何信息（与 StreamPETR 一致）===
        # ISSM 使用完整特征序列，在每层进行随机重排
        pos_embed, cone, img_coords_3d_for_issm = self.position_embeding(data, memory_center, img_metas)
        # pos_embed: [B, L, D] 位置编码（用于特征增强）
        # cone: [B, L, 8] 空间对齐信息（Focal PETR）
        # img_coords_3d_for_issm: [B, L, 3] 原始 3D 坐标（用于 ISSM 距离计算）
        
        # === 2. 特征展平（保持原始视图顺序，ISSM 每层会随机重排）===
        memory = x.permute(0, 1, 3, 4, 2).reshape(B, num_tokens, C)
        
        img_feats = self.memory_embed(memory)
        
        # === 3. 空间对齐（Focal PETR 机制）===
        img_feats = self.spatial_alignment(img_feats, cone)
        
        # === 4. 特征化位置编码（与 StreamPETR 一致）===
        pos_embed = self.featurized_pe(pos_embed, img_feats)
        
        # 注意：img_coords_3d_for_issm 已在步骤 1 中获取，无需重复计算
        
        # === 5. 初始化 Query（包含时序对齐）===
        queries, query_pos, reference_points, rec_ego_pose = self._init_queries(B, img_metas, **data)
        
        # 归一化 reference_points 用于距离计算（确保坐标系对齐）
        reference_points = (reference_points - self.pc_range[:3]) / (self.pc_range[3:6] - self.pc_range[0:3])  # [0, 1]
        
        # === 6. ISSM 逐层解码 ===
        if self.use_issm:
            # 逐层细化：Head 接管循环控制
            all_cls_scores = []
            all_bbox_preds = []
            all_output_queries = []
            all_output_reference_points = []
            
            curr_query = queries  # [B, N_q, D], N_q = 192 + 64 = 256
            curr_query_pos = query_pos  # [B, N_q, D]
            
            init_reference_points = reference_points.clone()  # [B, N_q, 3]
            
            # 特征缓存（逐层更新）
            curr_feat = img_feats  # 当前层特征
            

            # 遍历每一层 ISSM Layer
            for layer_idx in range(self.num_pred):
                # === Step 1: 准备输入特征（直接使用上一层输出）===
                feat_input = curr_feat
                
                # === Step 2: 保持原始视图顺序（不打乱）===
                # 多视图图像有空间连续性，不同于点云的随机性
                # 双向扫描已在 ISSM 层内实现，无需层间/视图顺序变化
                # 直接使用原始顺序：[front, front_right, front_left, back, back_left, back_right]
                
                # === Step 3: 运行单层 ISSM（直接使用原始顺序）===
                query_out, feat_output = self.issm_decoder.layers[layer_idx](
                    queries=curr_query,  # [B, N_q, D], N_q = 256
                    anchors=init_reference_points,  # Query 的 3D 锚点
                    features_perm=feat_input,  # 直接使用，不重排
                    pos_embed_perm=pos_embed,
                    key_pos_3d=img_coords_3d_for_issm,  # Key 的 3D 坐标
                    query_pos=curr_query_pos  # Query 位置编码
                )
                
                # === Step 4: 预测分类和回归 ===
                cls_scores = self.cls_branches[layer_idx](query_out)  # [B, N_q, num_cls]
                
                # 回归预测基于初始 reference_points
                reference = inverse_sigmoid(init_reference_points.clone())
                tmp = self.reg_branches[layer_idx](query_out)
                tmp[..., 0:3] += reference[..., 0:3]
                tmp[..., 0:3] = tmp[..., 0:3].sigmoid()
                bbox_preds = tmp  # [B, N_q, code_size]
                
                all_cls_scores.append(cls_scores)
                all_bbox_preds.append(bbox_preds)
                all_output_queries.append(query_out)
                all_output_reference_points.append(init_reference_points)
                
                # === Step 5: 更新状态 ===
                curr_query = query_out
                curr_feat = feat_output  # 逐层更新特征
        else:
            raise NotImplementedError("Original Transformer not implemented")
        
        # === 7. Stack 结果 ===
        all_cls_scores = torch.stack(all_cls_scores)  # [num_layers, B, N_q, num_cls]
        all_bbox_preds = torch.stack(all_bbox_preds)  # [num_layers, B, N_q, code_size]
        
        # 反归一化 bbox 坐标到真实空间（米）
        all_bbox_preds[..., 0:3] = (
            all_bbox_preds[..., 0:3] * (self.pc_range[3:6] - self.pc_range[0:3]) + 
            self.pc_range[0:3]
        )
        
        # === 8. 更新 Memory ===
        if self.training:
            # 使用最后一层的输出更新 memory
            final_queries = all_output_queries[-1]  # [B, N_q, D]
            final_reference_points = all_output_reference_points[-1]  # [B, N_q, 3] 归一化坐标
            
            # 反归一化 reference_points 用于 memory 存储
            final_reference_points_denorm = (
                final_reference_points * (self.pc_range[3:6] - self.pc_range[0:3]) + 
                self.pc_range[0:3]
            )
            
            # 使用 temporal_alignment 返回的 rec_ego_pose
            self._post_update_memory(
                final_queries, final_reference_points_denorm, 
                all_cls_scores[-1], rec_ego_pose, all_bbox_preds, 
                all_output_queries, **data
            )
        
        # === 9. 返回结果 ===
        outs = {
            'all_cls_scores': all_cls_scores,
            'all_bbox_preds': all_bbox_preds,
        }
        
        return outs

    def position_embeding(self, data, memory_centers, img_metas):
        """
        生成每个像素的 3D 坐标和位置编码（使用完整特征序列，无 topk 采样）
        
        Args:
            data: 包含 intrinsics, lidar2img 等的字典
            memory_centers: [BN, H, W, 2] 特征中心点坐标（归一化）
            img_metas: 图像元信息
        
        Returns:
            coords_position_embeding: [B, L, D] 位置编码（用于特征增强）
            cone: [B, L, 8] 用于 spatial alignment 的几何信息（内参 + 深度坐标）
            img_coords_3d: [B, L, 3] 原始 3D 坐标（归一化，用于 ISSM 距离计算）
        """
        eps = 1e-5
        BN, H, W, _ = memory_centers.shape
        B = data['intrinsics'].size(0)
        N = BN // B  # num_views
        
        # === 1. 处理相机内参 ===
        intrinsic = torch.stack([
            data['intrinsics'][..., 0, 0], 
            data['intrinsics'][..., 1, 1]
        ], dim=-1)  # [B, N, 2] 焦距 (fx, fy)
        intrinsic = torch.abs(intrinsic) / 1e3  # 归一化
        intrinsic = intrinsic.repeat(1, H*W, 1).view(B, -1, 2)
        LEN = intrinsic.size(1)  # N * H * W
        
        # === 2. 像素坐标转换 ===
        # 【修复】处理测试模式下 img_metas 的嵌套结构
        img_meta = img_metas[0]
        if isinstance(img_meta, list):
            img_meta = img_meta[0]
        pad_h, pad_w, _ = img_meta['pad_shape'][0]
        memory_centers = memory_centers.clone()
        memory_centers[..., 0] = memory_centers[..., 0] * pad_w
        memory_centers[..., 1] = memory_centers[..., 1] * pad_h
        
        # === 3. 深度采样（使用完整序列）===
        D = self.coords_d.shape[0]  # depth_num
        memory_centers = memory_centers.detach().view(B, LEN, 1, 2)
        centers = memory_centers.repeat(1, 1, D, 1)  # [B, LEN, D, 2]
        
        coords_d = self.coords_d.view(1, 1, D, 1).repeat(B, LEN, 1, 1)
        coords = torch.cat([centers, coords_d], dim=-1)  # [B, L, D, 3] (u, v, d)
        
        # === 4. 转齐次坐标 ===
        coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)  # [B, L, D, 4]
        coords[..., :2] = coords[..., :2] * torch.maximum(
            coords[..., 2:3], 
            torch.ones_like(coords[..., 2:3]) * eps
        )  # (u*d, v*d, d, 1)
        
        coords = coords.unsqueeze(-1)  # [B, L, D, 4, 1]
        
        # === 5. 反投影到 3D 空间 ===
        img2lidars = data['lidar2img'].inverse()  # [B*N, 4, 4]
        img2lidars = img2lidars.view(BN, 1, 1, 4, 4).repeat(1, H*W, D, 1, 1)
        img2lidars = img2lidars.view(B, LEN, D, 4, 4)
        
        coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3]  # [B, L, D, 3]
        
        # === 6. 归一化到 [0, 1] ===
        # 【修复】位置编码使用 position_range，距离计算使用 pc_range
        # 保存原始 coords3d 用于后续生成 img_coords_3d
        coords3d_raw = coords3d.clone()
        
        # 位置编码归一化（使用 position_range，与 StreamPETR 一致）
        coords3d[..., 0:3] = (
            (coords3d[..., 0:3] - self.position_range[0:3]) / 
            (self.position_range[3:6] - self.position_range[0:3])
        )
        
        # === 7. Reshape 保留所有深度信息（与 StreamPETR 一致）===
        coords3d = coords3d.reshape(B, LEN, D * 3)  # [B, L, D*3]
        
        # === 8. 生成位置编码 ===
        pos_embed = inverse_sigmoid(coords3d)
        coords_position_embeding = self.position_encoder(pos_embed)  # [B, L, D]
        
        # === 9. 构造 cone 用于 spatial alignment（Focal PETR）===
        # cone 包含：内参 + 最远深度坐标 + 中间深度坐标
        cone = torch.cat([intrinsic, coords3d[..., -3:], coords3d[..., -90:-87]], dim=-1)  # [B, L, 8]
        
        # === 10. 提取最后一个深度的 3D 坐标用于 ISSM 距离计算 ===
        # 【关键修复】使用 pc_range 归一化，与 Query anchors 坐标系对齐！
        # coords3d_raw 是未归一化的原始 3D 坐标，取最后一个深度
        coords3d_raw_last = coords3d_raw[..., -1, :]  # [B, L, 3] 最后一个深度的原始 xyz
        img_coords_3d = (
            (coords3d_raw_last - self.pc_range[0:3]) / 
            (self.pc_range[3:6] - self.pc_range[0:3])
        )  # [B, L, 3] 使用 pc_range 归一化，与 anchors 对齐
        
        return coords_position_embeding, cone, img_coords_3d

    def temporal_alignment(self, query_pos, tgt, reference_points):
        """
        时序对齐模块（简化版：只传播 num_propagated 个 query）
        
        功能：
        1. 处理历史 Memory 的位置编码
        2. 添加 Ego Pose 编码（自车运动）
        3. 添加时间戳编码（区分不同时刻）
        4. 将 propagated queries 拼接到当前 queries
        
        简化说明：
        - 不再使用 temp_memory（剩余的 memory 不参与 ISSM 交互）
        - 只保留 num_propagated 个 query 用于跨帧传播
        
        Args:
            query_pos: [B, N_q, D] 当前帧 query 位置编码
            tgt: [B, N_q, D] 当前帧 query 内容
            reference_points: [B, N_q, 3] 当前帧 reference points（归一化）
        
        Returns:
            tgt: [B, N_q + N_prop, D] 拼接后的 query 内容
            query_pos: [B, N_q + N_prop, D] 拼接后的位置编码
            reference_points: [B, N_q + N_prop, 3] 拼接后的 reference points
            rec_ego_pose: [B, N_q + N_prop, 4, 4] ego pose 矩阵
        """
        B = query_pos.size(0)

        # === 1. 处理历史 Memory（只取 num_propagated 个）===
        prop_reference_point = (self.memory_reference_point - self.pc_range[:3]) / (self.pc_range[3:6] - self.pc_range[0:3])
        prop_pos = self.query_embedding(pos2posemb3d(prop_reference_point))
        prop_memory = self.memory_embedding
        rec_ego_pose = torch.eye(4, device=query_pos.device).unsqueeze(0).unsqueeze(0).repeat(B, query_pos.size(1), 1, 1)
        
        # === 2. Ego Pose 编码（自车运动）===
        if self.with_ego_pos:
            # 当前帧的 ego motion（相对于自己为0）
            rec_ego_motion = torch.cat([torch.zeros_like(reference_points[...,:3]), rec_ego_pose[..., :3, :].flatten(-2)], dim=-1)
            rec_ego_motion = nerf_positional_encoding(rec_ego_motion)
            tgt = self.ego_pose_memory(tgt, rec_ego_motion)
            query_pos = self.ego_pose_pe(query_pos, rec_ego_motion)
            
            # 历史帧的 ego motion（包含速度、时间戳、相对位姿）
            memory_ego_motion = torch.cat([self.memory_velo, self.memory_timestamp, self.memory_egopose[..., :3, :].flatten(-2)], dim=-1).float()
            memory_ego_motion = nerf_positional_encoding(memory_ego_motion)
            prop_pos = self.ego_pose_pe(prop_pos, memory_ego_motion)
            prop_memory = self.ego_pose_memory(prop_memory, memory_ego_motion)

        # === 3. 时间戳编码 ===
        query_pos += self.time_embedding(pos2posemb1d(torch.zeros_like(reference_points[...,:1])))  # 当前帧 t=0
        prop_pos += self.time_embedding(pos2posemb1d(self.memory_timestamp).float())  # 历史帧 t<0

        # === 4. Query 传播（拼接 propagated queries）===
        if self.num_propagated > 0:
            tgt = torch.cat([tgt, prop_memory], dim=1)
            query_pos = torch.cat([query_pos, prop_pos], dim=1)
            reference_points = torch.cat([reference_points, prop_reference_point], dim=1)
            rec_ego_pose = torch.eye(4, device=query_pos.device).unsqueeze(0).unsqueeze(0).repeat(B, query_pos.shape[1], 1, 1)
            
        return tgt, query_pos, reference_points, rec_ego_pose
    
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """
        加载 checkpoint 时的兼容性处理
        
        注意：
        - ISSM 使用新的解码器，无需转换 Transformer 相关参数
        - 但保留此方法以便未来扩展和从 StreamPETR 预训练权重初始化
        """
        # NOTE: 使用 `AnchorFreeHead` 而非 `TransformerHead`
        # 因为 `AnchorFreeHead._load_from_state_dict` 不应该在这里调用
        # 调用默认的 `Module._load_from_state_dict` 就足够了

        # ISSM 特定处理：跳过不兼容的 Transformer 参数
        # 如果从 StreamPETR checkpoint 加载，忽略 transformer 相关的 key
        version = local_metadata.get('version', None)
        if (version is None or version < 2) and self.__class__ is ISSMStreamPETRHead:
            # ISSM 没有 transformer，移除相关参数以避免加载错误
            state_dict_keys = list(state_dict.keys())
            for k in state_dict_keys:
                # 跳过 Transformer 相关参数（ISSM 使用 ISSM decoder）
                if 'transformer.' in k:
                    print(f"[ISSM] Skipping incompatible key from StreamPETR checkpoint: {k}")
                    del state_dict[k]
                    unexpected_keys.append(prefix + k)

        super(AnchorFreeHead,
              self)._load_from_state_dict(state_dict, prefix, local_metadata,
                                          strict, missing_keys,
                                          unexpected_keys, error_msgs)

    def _init_queries(self, B, img_metas, **data):
        """
        初始化 Query 和 Anchors
        
        参考 StreamPETR 的以下部分：
        1. streampetr_head.py forward() 第 598-604 行：reference_points 初始化和位置编码
        2. streampetr_head.py temporal_alignment() 第 420-449 行：完整的时序对齐（现已实现）
        
        说明：
        - 完整保留了 StreamPETR 的 temporal_alignment 功能
        - 包含 Ego Pose 编码、时间戳编码、Query 传播
        
        【v5 更新】：额外返回 query_pos 用于 StreamPETR 风格位置编码
        
        Args:
            B: batch size
            img_metas: 图像元信息
            **data: 其他数据
        
        Returns:
            queries: [B, N_q + N_prop, D] Query 特征（包含 propagated queries）
            query_pos: [B, N_q + N_prop, D] Query 位置编码（新增！用于 ISSM 位置交互）
            reference_points: [B, N_q + N_prop, 3] Query 的 3D reference points
            rec_ego_pose: [B, N_q + N_prop, 4, 4] Ego pose 矩阵（用于 memory 更新）
        """
        device = self.reference_points.weight.device
        
        # === 1. 获取可学习的 reference points ===
        # 对应 streampetr_head.py line 598: reference_points = self.reference_points.weight
        reference_points = self.reference_points.weight  # [num_query, 3] in [0,1]
        
        # === 2. 生成 Query 位置编码 ===
        # 对应 streampetr_head.py line 600-601
        query_pos = self.query_embedding(pos2posemb3d(reference_points))  # [num_query, D]
        # 添加 batch 维度
        query_pos = query_pos.unsqueeze(0).expand(B, -1, -1)  # [B, N_q, D]
        tgt = torch.zeros_like(query_pos)  # [B, N_q, D]
        
        # 给 reference_points 添加 batch 维度（用于 temporal_alignment）
        reference_points = reference_points.unsqueeze(0).expand(B, -1, -1)  # [B, num_query, 3]
        
        # === 3. 时序对齐（简化版：只传播 num_propagated 个 query）===
        # 包含 Ego Pose 编码、时间戳编码、Query 传播
        tgt, query_pos, reference_points, rec_ego_pose = self.temporal_alignment(
            query_pos, tgt, reference_points
        )
        
        # === 4. 构造返回值 ===
        queries = tgt  # [B, N_q + N_prop, D] 纯 Query 内容特征
        
        # reference_points: [B, N_q + N_prop, 3]  对应的 reference points（归一化）
        
        # 反归一化 reference_points 到实际空间
        reference_points_result = reference_points * (self.pc_range[3:6] - self.pc_range[0:3]) + self.pc_range[0:3]  # [B, N_q + N_prop, 3]
        
        return queries, query_pos, reference_points_result, rec_ego_pose

    def _post_update_memory(self, queries, anchors, cls_scores, rec_ego_pose, 
                           all_bbox_preds, outs_dec, **data):
        """
        训练后更新 Memory Queue（简化版）
        
        选择高置信度的 Query 写入 Memory，用于下一帧
        简化：只保留 num_propagated 个 query，无 DN
        """
        N_prop = self.num_propagated
        
        # 获取最后一层的预测结果
        rec_reference_points = all_bbox_preds[..., :3][-1]
        rec_velo = all_bbox_preds[..., -2:][-1]
        rec_memory = outs_dec[-1]
        rec_score = cls_scores.sigmoid().topk(1, dim=-1).values[..., 0:1]
        rec_timestamp = torch.zeros_like(rec_score, dtype=torch.float64)
        
        # === 选择 top num_propagated 个 query ===
        _, topk_indexes = torch.topk(rec_score, N_prop, dim=1)
        rec_timestamp = topk_gather(rec_timestamp, topk_indexes)
        rec_reference_points = topk_gather(rec_reference_points, topk_indexes).detach()
        rec_memory = topk_gather(rec_memory, topk_indexes).detach()
        rec_ego_pose = topk_gather(rec_ego_pose, topk_indexes)
        rec_velo = topk_gather(rec_velo, topk_indexes).detach()
        
        # === 直接替换 Memory（大小固定为 num_propagated）===
        self.memory_embedding = rec_memory
        self.memory_timestamp = rec_timestamp
        self.memory_egopose = rec_ego_pose
        self.memory_reference_point = rec_reference_points
        self.memory_velo = rec_velo
        
        # 变换到当前帧坐标系
        self.memory_reference_point = transform_reference_points(
            self.memory_reference_point, data['ego_pose'], reverse=False
        )
        self.memory_timestamp -= data['timestamp'].unsqueeze(-1).unsqueeze(-1)
        self.memory_egopose = data['ego_pose'].unsqueeze(1) @ self.memory_egopose

    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_bboxes_ignore=None):
        """Compute regression and classification targets for one image."""
        num_bboxes = bbox_pred.size(0)
        
        # assigner and sampler
        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                                gt_labels, gt_bboxes_ignore, 
                                                self.match_costs, self.match_with_velo)
        sampling_result = self.sampler.sample(assign_result, bbox_pred, gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes, ),
                                    self.num_classes,
                                    dtype=torch.long)
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        code_size = gt_bboxes.size(1)
        bbox_targets = torch.zeros_like(bbox_pred)[..., :code_size]
        bbox_weights = torch.zeros_like(bbox_pred)
        
        if sampling_result.num_gts > 0:
            bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
            bbox_weights[pos_inds] = 1.0
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        
        return (labels, label_weights, bbox_targets, bbox_weights, 
                pos_inds, neg_inds)

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """Compute regression and classification targets for a batch image."""
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
             self._get_target_single, cls_scores_list, bbox_preds_list,
             gt_labels_list, gt_bboxes_list, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """Loss function for outputs from a single decoder layer."""
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list, 
                                           gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
                bbox_preds[isnotnan, :10], normalized_bbox_targets[isnotnan, :10], 
                bbox_weights[isnotnan, :10], avg_factor=num_total_pos)

        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)
        return loss_cls, loss_bbox

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             preds_dicts,
             gt_bboxes_ignore=None,
             img_metas=None):
        """Loss function.
        
        Args:
            gt_bboxes_list: Ground truth bboxes.
            gt_labels_list: Ground truth labels.
            preds_dicts: Predictions from the head.
            gt_bboxes_ignore: Ignored bboxes.
            img_metas: Image meta info (optional, for compatibility).
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device
        gt_bboxes_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_list]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        losses_cls, losses_bbox = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list, 
            all_gt_bboxes_ignore_list)

        loss_dict = dict()
        
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1],
                                           losses_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1

        return loss_dict

    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions."""
        preds_dicts = self.bbox_coder.decode(preds_dicts)
        num_samples = len(preds_dicts)

        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            bboxes = img_metas[i]['box_type_3d'](bboxes, bboxes.size(-1))
            scores = preds['scores']
            labels = preds['labels']
            ret_list.append([bboxes, scores, labels])
        return ret_list
