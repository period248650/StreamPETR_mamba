#!/usr/bin/env python
"""
调试脚本：检查模型预测输出的坐标范围和分类分数
用于排查 mAP = 0 的问题

使用方法：
cd /mnt/c/Users/17203/Desktop/Research/StreamPETR_mamba_new
PYTHONPATH="projects:$PYTHONPATH" python debug_predictions.py \
    projects/configs/issm_streampetr/issm_streampetr_dense_alternating.py \
    ckpts/test/iter_38676.pth
"""

import argparse
import torch
import numpy as np
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
import os
import importlib


def parse_args():
    parser = argparse.ArgumentParser(description='Debug model predictions')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--num-samples', type=int, default=5, help='number of samples to check')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 加载配置
    cfg = Config.fromfile(args.config)
    
    # 导入插件
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(f"Loading plugin from: {_module_path}")
                plg_lib = importlib.import_module(_module_path)
    
    # 构建模型
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    
    # 加载权重
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    
    model.cuda()
    model.eval()
    
    # 构建数据集
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=1,
        dist=False,
        shuffle=False
    )
    
    print("\n" + "="*80)
    print("开始检查模型预测输出...")
    print("="*80)
    
    # 检查 bbox_coder 的配置
    bbox_coder_cfg = cfg.model.pts_bbox_head.bbox_coder
    print(f"\n[配置信息]")
    print(f"  post_center_range: {bbox_coder_cfg.post_center_range}")
    print(f"  pc_range: {bbox_coder_cfg.pc_range}")
    print(f"  max_num: {bbox_coder_cfg.max_num}")
    print(f"  score_threshold: {bbox_coder_cfg.get('score_threshold', None)}")
    
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            if i >= args.num_samples:
                break
                
            print(f"\n{'='*80}")
            print(f"样本 {i + 1}/{args.num_samples}")
            print("="*80)
            
            # 处理 DataContainer 格式的数据
            def unwrap_data(val):
                """递归解包 DataContainer"""
                if hasattr(val, 'data'):
                    return unwrap_data(val.data)
                elif isinstance(val, list) and len(val) > 0:
                    # 检查是否需要进一步解包
                    if hasattr(val[0], 'data'):
                        return [unwrap_data(v) for v in val]
                    return val
                return val
            
            # 执行前向传播
            # 先获取 pts_bbox_head 的输出
            model.pts_bbox_head.reset_memory()
            
            # 模拟 forward_test 的过程
            # 处理 img
            img = data['img']
            img = unwrap_data(img)
            if isinstance(img, list):
                img = img[0]
            if isinstance(img, list):
                img = img[0]
            img = img.cuda()
            
            # 处理 img_metas
            img_metas = data['img_metas']
            img_metas = unwrap_data(img_metas)
            if isinstance(img_metas, list):
                img_metas = img_metas[0]
            if isinstance(img_metas, list):
                img_metas = img_metas  # 保持 list 格式
            
            print(f"  img shape: {img.shape}")
            print(f"  img_metas type: {type(img_metas)}")
            
            # 提取图像特征
            img_feats = model.extract_img_feat(img, 1)
            
            # 准备 location
            B, N, C, H, W = img_feats.shape
            dtype = img_feats.dtype
            device = img_feats.device
            coords_h = torch.arange(H, device=device).float() * model.pts_bbox_head.feat_stride / model.pts_bbox_head.inp_img_h
            coords_w = torch.arange(W, device=device).float() * model.pts_bbox_head.feat_stride / model.pts_bbox_head.inp_img_w
            coords = torch.stack(torch.meshgrid([coords_w, coords_h])).permute(1, 2, 0)  # [H, W, 2]
            coords = coords.unsqueeze(0).repeat(B*N, 1, 1, 1)  # [B*N, H, W, 2]
            
            # 准备数据
            test_data = {
                'img_feats': img_feats,
            }
            
            # 从 data 中提取其他必需字段
            for key in ['intrinsics', 'extrinsics', 'lidar2img', 'ego_pose', 'ego_pose_inv', 'timestamp', 'img_timestamp']:
                if key in data:
                    val = data[key]
                    val = unwrap_data(val)
                    if isinstance(val, list):
                        val = val[0]
                    if isinstance(val, list):
                        val = val[0]
                    if isinstance(val, torch.Tensor):
                        test_data[key] = val.cuda()
                    else:
                        test_data[key] = val
            
            # 设置 prev_exists
            test_data['prev_exists'] = torch.ones(1, device=device)
            
            # 调用 pts_bbox_head
            try:
                outs = model.pts_bbox_head(coords, img_metas, None, **test_data)
                
                # 检查输出
                all_cls_scores = outs['all_cls_scores']  # [num_layers, B, N_q, num_cls]
                all_bbox_preds = outs['all_bbox_preds']  # [num_layers, B, N_q, code_size]
                
                print(f"\n[Head 输出形状]")
                print(f"  all_cls_scores: {all_cls_scores.shape}")
                print(f"  all_bbox_preds: {all_bbox_preds.shape}")
                
                # 取最后一层的输出
                cls_scores = all_cls_scores[-1]  # [B, N_q, num_cls]
                bbox_preds = all_bbox_preds[-1]  # [B, N_q, code_size]
                
                # 检查分类分数
                cls_probs = cls_scores.sigmoid()
                max_probs = cls_probs.max(dim=-1)[0]  # [B, N_q]
                
                print(f"\n[分类分数统计]")
                print(f"  最大概率: {max_probs.max().item():.6f}")
                print(f"  平均概率: {max_probs.mean().item():.6f}")
                print(f"  最小概率: {max_probs.min().item():.6f}")
                print(f"  概率 > 0.1 的数量: {(max_probs > 0.1).sum().item()}")
                print(f"  概率 > 0.05 的数量: {(max_probs > 0.05).sum().item()}")
                print(f"  概率 > 0.01 的数量: {(max_probs > 0.01).sum().item()}")
                
                # 检查 bbox 坐标
                print(f"\n[Bbox 坐标统计]")
                print(f"  cx (x中心): min={bbox_preds[..., 0].min().item():.2f}, max={bbox_preds[..., 0].max().item():.2f}, mean={bbox_preds[..., 0].mean().item():.2f}")
                print(f"  cy (y中心): min={bbox_preds[..., 1].min().item():.2f}, max={bbox_preds[..., 1].max().item():.2f}, mean={bbox_preds[..., 1].mean().item():.2f}")
                print(f"  cz (z中心): min={bbox_preds[..., 2].min().item():.2f}, max={bbox_preds[..., 2].max().item():.2f}, mean={bbox_preds[..., 2].mean().item():.2f}")
                
                # 检查 w, l, h (log 空间)
                print(f"\n[Bbox 尺寸统计 (log 空间)]")
                print(f"  log(w): min={bbox_preds[..., 3].min().item():.2f}, max={bbox_preds[..., 3].max().item():.2f}")
                print(f"  log(l): min={bbox_preds[..., 4].min().item():.2f}, max={bbox_preds[..., 4].max().item():.2f}")
                print(f"  log(h): min={bbox_preds[..., 5].min().item():.2f}, max={bbox_preds[..., 5].max().item():.2f}")
                
                # 转换为真实尺寸
                w_real = bbox_preds[..., 3].exp()
                l_real = bbox_preds[..., 4].exp()
                h_real = bbox_preds[..., 5].exp()
                print(f"\n[Bbox 尺寸统计 (真实空间, 米)]")
                print(f"  w: min={w_real.min().item():.2f}, max={w_real.max().item():.2f}")
                print(f"  l: min={l_real.min().item():.2f}, max={l_real.max().item():.2f}")
                print(f"  h: min={h_real.min().item():.2f}, max={h_real.max().item():.2f}")
                
                # 检查 post_center_range 过滤
                post_range = torch.tensor(bbox_coder_cfg.post_center_range, device=device)
                centers = bbox_preds[0, :, :3]  # [N_q, 3]
                in_range = (centers >= post_range[:3]).all(dim=-1) & (centers <= post_range[3:]).all(dim=-1)
                
                print(f"\n[Post Center Range 过滤]")
                print(f"  在范围内的预测数量: {in_range.sum().item()} / {centers.shape[0]}")
                
                # 检查是否有 NaN 或 Inf
                print(f"\n[数值检查]")
                print(f"  cls_scores 有 NaN: {torch.isnan(cls_scores).any().item()}")
                print(f"  cls_scores 有 Inf: {torch.isinf(cls_scores).any().item()}")
                print(f"  bbox_preds 有 NaN: {torch.isnan(bbox_preds).any().item()}")
                print(f"  bbox_preds 有 Inf: {torch.isinf(bbox_preds).any().item()}")
                
                # 模拟 decode 过程
                print(f"\n[Decode 模拟]")
                cls_probs_flat = cls_probs[0].view(-1)  # [N_q * num_cls]
                scores, indices = cls_probs_flat.topk(300)
                labels = indices % 10  # num_classes
                bbox_idx = torch.div(indices, 10, rounding_mode='floor')
                selected_bboxes = bbox_preds[0, bbox_idx, :]
                
                print(f"  Top-300 分数: min={scores.min().item():.6f}, max={scores.max().item():.6f}")
                
                # 检查选中的 bbox 是否在范围内
                selected_centers = selected_bboxes[:, :3]
                selected_in_range = (selected_centers >= post_range[:3]).all(dim=-1) & (selected_centers <= post_range[3:]).all(dim=-1)
                print(f"  Top-300 中在范围内的数量: {selected_in_range.sum().item()}")
                
                # 最终保留的数量
                final_count = (selected_in_range).sum().item()
                print(f"  最终保留的预测数量: {final_count}")
                
            except Exception as e:
                print(f"  错误: {e}")
                import traceback
                traceback.print_exc()
    
    print("\n" + "="*80)
    print("调试完成")
    print("="*80)
    
    # 总结问题
    print("\n[可能的问题原因]")
    print("1. 如果分类分数太低（最大 < 0.1），说明模型没有学会分类")
    print("2. 如果 bbox 坐标超出 post_center_range，说明位置预测有问题")
    print("3. 如果有 NaN/Inf，说明数值计算有问题")
    print("4. 如果 Top-300 中在范围内的数量为 0，说明所有预测都被过滤掉了")


if __name__ == '__main__':
    main()
