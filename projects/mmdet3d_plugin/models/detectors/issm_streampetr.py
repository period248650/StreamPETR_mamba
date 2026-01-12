# ------------------------------------------------------------------------
# Copyright (c) 2024 ISSM-StreamPETR. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Petr3D (https://github.com/megvii-research/PETR)
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
import torch
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from projects.mmdet3d_plugin.models.utils.misc import locations


@DETECTORS.register_module()
class ISSMStreamPETR(MVXTwoStageDetector):
    """ISSM-StreamPETR: StreamPETR with Internal State Space Model.
    
    Incorporates ISSM (Internal State Space Model) for efficient temporal
    modeling in 3D object detection from multi-view camera images.
    
    Key Features:
    - ISSM-based transformer decoder for efficient sequence modeling
    - Support for dense alternating scanning patterns
    - Memory-efficient temporal feature propagation
    - Multi-scale feature extraction from images
    """

    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 num_frame_head_grads=2,
                 num_frame_backbone_grads=2,
                 num_frame_losses=2,
                 stride=16,
                 position_level=0,
                 aux_2d_only=True,
                 single_test=False,
                 pretrained=None):
        """Initialize ISSMStreamPETR.
        
        Args:
            use_grid_mask (bool): Whether to use grid mask augmentation.
            num_frame_head_grads (int): Number of frames for head gradient computation.
            num_frame_backbone_grads (int): Number of frames for backbone gradient computation.
            num_frame_losses (int): Number of frames for loss computation.
            stride (int): Feature stride.
            position_level (int): Which level of features to use for position encoding.
            aux_2d_only (bool): Whether to use 2D auxiliary losses only.
            single_test (bool): Whether to test on single frame.
            Other args: Same as MVXTwoStageDetector.
        """
        super(ISSMStreamPETR, self).__init__(
            pts_voxel_layer, pts_voxel_encoder,
            pts_middle_encoder, pts_fusion_layer,
            img_backbone, pts_backbone, img_neck, pts_neck,
            pts_bbox_head, img_roi_head, img_rpn_head,
            train_cfg, test_cfg, pretrained)
        
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.prev_scene_token = None
        self.num_frame_head_grads = num_frame_head_grads
        self.num_frame_backbone_grads = num_frame_backbone_grads
        self.num_frame_losses = num_frame_losses
        self.single_test = single_test
        self.stride = stride
        self.position_level = position_level
        self.aux_2d_only = aux_2d_only
        self.test_flag = False

    def prepare_location(self, img_metas, **data):
        """Prepare location/memory_center for the detection head.
        
        Args:
            img_metas: Image meta information.
            **data: Data dict containing img_feats.
            
        Returns:
            Tensor: Location tensor of shape [B*N, H, W, 2].
        """
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        bs, n = data['img_feats'].shape[:2]
        x = data['img_feats'].flatten(0, 1)
        location = locations(x, self.stride, pad_h, pad_w)[None].repeat(bs*n, 1, 1, 1)
        return location

    def forward_roi_head(self, location, **data):
        """Forward function for FocalHead (2D auxiliary head).
        
        注意：ISSM 不使用 topk_indexes 进行特征采样（ISSM 有自己的扫描机制），
        但 FocalHead 仍需要前向传播以：
        1. 训练时：计算 2D 辅助损失，帮助特征学习
        2. 测试时：不需要调用（aux_2d_only=True）
        
        Args:
            location: Pixel locations.
            **data: Data dict including img_feats.
            
        Returns:
            dict: Contains prediction outputs for 2D auxiliary loss.
        """
        if (self.aux_2d_only and not self.training) or not self.with_img_roi_head:
            return {}
        else:
            outs_roi = self.img_roi_head(location, **data)
            return outs_roi
    
    def extract_img_feat(self, img, len_queue=1, training_mode=False):
        """Extract features of images.
        
        Args:
            img (Tensor): Images of shape (B, N, C, H, W) or (B*N, C, H, W).
            len_queue (int): Length of the temporal queue.
            training_mode (bool): Whether in training mode.
            
        Returns:
            Tensor: Extracted image features of shape (B, len_queue, N, C, H, W)
                    or (B, N, C, H, W) depending on training_mode.
        """
        # 【修复】处理测试模式下 img 可能是 list 的情况
        if isinstance(img, list):
            img = img[0]
        
        if img is None:
            return None
        
        B = img.size(0)

        if img is not None:
            # Handle different input dimensions
            if img.dim() == 6:
                img = img.flatten(1, 2)
            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            
            # Apply grid mask augmentation
            if self.use_grid_mask:
                img = self.grid_mask(img)

            # Extract features using backbone
            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        
        # Apply neck if available
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        # Reshape features for temporal modeling
        BN, C, H, W = img_feats[self.position_level].size()
        if self.training or training_mode:
            img_feats_reshaped = img_feats[self.position_level].view(
                B, len_queue, int(BN/B / len_queue), C, H, W)
        else:
            img_feats_reshaped = img_feats[self.position_level].view(
                B, int(BN/B/len_queue), C, H, W)

        return img_feats_reshaped

    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_feat(self, img, T, training_mode=False):
        """Extract features from images.
        
        Args:
            img (Tensor): Input images.
            T (int): Number of temporal frames.
            training_mode (bool): Whether in training mode.
            
        Returns:
            Tensor: Extracted features.
        """
        img_feats = self.extract_img_feat(img, T, training_mode=training_mode)
        return img_feats

    def obtain_history_memory(self,
                              gt_bboxes_3d=None,
                              gt_labels_3d=None,
                              gt_bboxes=None,
                              gt_labels=None,
                              img_metas=None,
                              centers2d=None,
                              depths=None,
                              gt_bboxes_ignore=None,
                              **data):
        """Process temporal frames and accumulate losses.
        
        Args:
            gt_bboxes_3d: Ground truth 3D boxes for each frame.
            gt_labels_3d: Ground truth 3D labels for each frame.
            gt_bboxes: Ground truth 2D boxes for each frame.
            gt_labels: Ground truth 2D labels for each frame.
            img_metas: Image meta info for each frame.
            centers2d: 2D centers for each frame.
            depths: Depths for each frame.
            gt_bboxes_ignore: Ignored boxes.
            **data: Other data including img_feats.
            
        Returns:
            dict: Accumulated losses.
        """
        losses = dict()
        T = data['img'].size(1)
        num_nograd_frames = T - self.num_frame_head_grads
        num_grad_losses = T - self.num_frame_losses
        
        for i in range(T):
            requires_grad = False
            return_losses = False
            data_t = dict()
            
            # Extract data for current frame
            for key in data:
                data_t[key] = data[key][:, i]
            
            data_t['img_feats'] = data_t['img_feats']
            
            if i >= num_nograd_frames:
                requires_grad = True
            if i >= num_grad_losses:
                return_losses = True
            
            # Forward for current frame
            # Note: After transpose in forward(), img_metas[i] is a tuple
            # We need to convert it back to a list for the head
            img_metas_i = list(img_metas[i]) if isinstance(img_metas[i], tuple) else img_metas[i]
            
            loss = self.forward_pts_train(
                gt_bboxes_3d[i], gt_labels_3d[i], gt_bboxes[i],
                gt_labels[i], img_metas_i, centers2d[i], depths[i],
                requires_grad=requires_grad, return_losses=return_losses, **data_t)
            
            if loss is not None:
                for key, value in loss.items():
                    losses['frame_' + str(i) + "_" + key] = value
        
        return losses

    def forward_pts_train(self,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          gt_bboxes,
                          gt_labels,
                          img_metas,
                          centers2d,
                          depths,
                          requires_grad=True,
                          return_losses=False,
                          **data):
        """Forward function for point cloud branch.
        
        Args:
            gt_bboxes_3d: Ground truth 3D boxes.
            gt_labels_3d: Ground truth 3D labels.
            gt_bboxes: Ground truth 2D boxes.
            gt_labels: Ground truth 2D labels.
            img_metas: Image meta information.
            centers2d: 2D centers of 3D boxes.
            depths: Depths of 3D boxes.
            requires_grad (bool): Whether to compute gradients.
            return_losses (bool): Whether to return losses.
            **data: Other data including img_feats.
            
        Returns:
            dict or None: Losses if return_losses=True, else None.
        """
        # Prepare location (memory_center) for the head
        location = self.prepare_location(img_metas, **data)
        
        # === FocalHead 前向传播（用于 2D 辅助损失）===
        # 注意：ISSM 不使用 topk_indexes，但 FocalHead 仍需前向传播以计算 2D 辅助损失
        outs_roi = self.forward_roi_head(location, **data)
        
        if not requires_grad:
            self.eval()
            with torch.no_grad():
                outs = self.pts_bbox_head(location, img_metas, **data)
            self.train()
        else:
            outs = self.pts_bbox_head(location, img_metas, **data)

        if return_losses:
            loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
            losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
            
            # === 2D 辅助损失 (FocalHead) ===
            if self.with_img_roi_head and outs_roi:
                # 计算 2D 辅助损失
                aux_losses = self.img_roi_head.loss(
                    gt_bboxes,  # gt_bboxes2d_list
                    gt_labels,  # gt_labels2d_list
                    centers2d,  # centers2d
                    depths,     # depths
                    outs_roi,   # preds_dicts (来自 forward_roi_head)
                    img_metas,  # img_metas
                    gt_bboxes_ignore=None
                )
                
                # 添加 2D 辅助损失
                for key, value in aux_losses.items():
                    losses['aux_2d_' + key] = value
            
            return losses
        else:
            return None

    @force_fp32(apply_to=('img'))
    def forward(self, return_loss=True, **data):
        """Calls either forward_train or forward_test depending on mode.
        
        Args:
            return_loss (bool): Whether to return loss.
            **data: Input data including img, img_metas, gt_bboxes_3d, etc.
            
        Returns:
            dict or list: Training losses or detection results.
        """
        if return_loss:
            # Reorganize data: transpose list of dicts to dict of lists
            for key in ['gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes', 'gt_labels', 'centers2d', 'depths', 'img_metas']:
                if key in data:
                    data[key] = list(zip(*data[key]))
            return self.forward_train(**data)
        else:
            return self.forward_test(**data)

    def forward_train(self,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      gt_bboxes_ignore=None,
                      depths=None,
                      centers2d=None,
                      **data):
        """Forward training function.
        
        Args:
            img_metas (list[dict]): Meta information of each sample.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sample.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                for 2D boxes.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ignored boxes.
            depths (list[torch.Tensor], optional): Depths of 3D boxes.
            centers2d (list[torch.Tensor], optional): 2D centers of 3D boxes.
            **data: Other keyword arguments including 'img'.
            
        Returns:
            dict: Losses of each branch.
        """
        if self.test_flag:  # for interval evaluation
            self.pts_bbox_head.reset_memory()
            self.test_flag = False

        T = data['img'].size(1)

        prev_img = data['img'][:, :-self.num_frame_backbone_grads]
        rec_img = data['img'][:, -self.num_frame_backbone_grads:]
        rec_img_feats = self.extract_feat(rec_img, self.num_frame_backbone_grads)

        if T - self.num_frame_backbone_grads > 0:
            self.eval()
            with torch.no_grad():
                prev_img_feats = self.extract_feat(prev_img, T - self.num_frame_backbone_grads, True)
            self.train()
            data['img_feats'] = torch.cat([prev_img_feats, rec_img_feats], dim=1)
        else:
            data['img_feats'] = rec_img_feats

        losses = self.obtain_history_memory(
            gt_bboxes_3d, gt_labels_3d, gt_bboxes,
            gt_labels, img_metas, centers2d, depths, gt_bboxes_ignore, **data)

        return losses

    def forward_test(self, img_metas, rescale=False, **data):
        """Forward testing function.
        
        与原始 petr3d.py 保持一致的测试数据处理逻辑。
        
        Args:
            img_metas (list[dict]): Meta information of images.
            rescale (bool): Whether to rescale results.
            **data: Other arguments including img, intrinsics, lidar2img, 
                    timestamp, ego_pose, ego_pose_inv, etc.
            
        Returns:
            list[dict]: Detection results.
        """
        self.test_flag = True
        
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        # 【关键修复】与 petr3d.py 完全一致的数据处理
        # 不跳过任何字段！timestamp, ego_pose, ego_pose_inv 对时序对齐至关重要
        for key in data:
            if key != 'img':
                # 处理嵌套结构: [[tensor]] -> tensor with batch dim
                data[key] = data[key][0][0].unsqueeze(0)
            else:
                # img 只需要去掉一层嵌套
                data[key] = data[key][0]

        return self.simple_test(img_metas[0], **data)

    def simple_test_pts(self, img_metas, **data):
        """Simple test function for point cloud branch.
        
        与原始 petr3d.py 保持一致的测试逻辑。
        
        Args:
            img_metas (list[dict]): Meta information.
            **data: Data dict including img_feats, intrinsics, lidar2img, 
                    timestamp, ego_pose, ego_pose_inv, etc.
            
        Returns:
            list[dict]: Detection results.
        """
        # 准备 location (memory_center)
        location = self.prepare_location(img_metas, **data)
        
        if img_metas[0]['scene_token'] != self.prev_scene_token:
            self.prev_scene_token = img_metas[0]['scene_token']
            data['prev_exists'] = data['img'].new_zeros(1)
            self.pts_bbox_head.reset_memory()
        else:
            data['prev_exists'] = data['img'].new_ones(1)
        
        # 调用 head 进行推理（ISSM 不需要 topk_indexes）
        outs = self.pts_bbox_head(location, img_metas, **data)
        
        bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def simple_test(self, img_metas, img=None, prev_bev=None, rescale=False, **data):
        """Simple test function without augmentation.
        
        Args:
            img_metas (list[dict]): Meta information.
            img (Tensor, optional): Images.
            prev_bev (Tensor, optional): Previous BEV features.
            rescale (bool): Whether to rescale boxes.
            **data: Other data including intrinsics, lidar2img, etc.
            
        Returns:
            list[dict]: Detection results.
        """
        # 处理 img：从 data 中获取或使用传入的参数
        if img is None and 'img' in data:
            img = data['img']
        
        # 【关键】确保 data['img'] 存在，供后续 simple_test_pts 使用
        data['img'] = img
        
        # 提取图像特征
        data['img_feats'] = self.extract_img_feat(img, 1)
        
        bbox_list = [dict() for _ in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(img_metas, **data)
        
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        
        return bbox_list

    def set_epoch(self, epoch):
        """Set epoch for the model.
        
        This method is called by SetEpochInfoHook to update the current epoch.
        Can be used to adjust model behavior based on training progress.
        
        Args:
            epoch (int): Current epoch number.
        """
        self.epoch = epoch
        # Pass epoch to head if it has set_epoch method
        if hasattr(self.pts_bbox_head, 'set_epoch'):
            self.pts_bbox_head.set_epoch(epoch)
