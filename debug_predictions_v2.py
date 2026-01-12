#!/usr/bin/env python
"""
调试脚本：检查模型预测输出的坐标范围和分类分数
用于排查 mAP = 0 的问题

使用方法：
cd /mnt/c/Users/17203/Desktop/Research/StreamPETR_mamba_new
PYTHONPATH="projects:$PYTHONPATH" python debug_predictions_v2.py \
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
from mmdet3d.apis import single_gpu_test
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
    
    # Hook 到 pts_bbox_head 的 get_bboxes 方法来检查输入
    original_get_bboxes = model.pts_bbox_head.get_bboxes
    
    debug_info = {}
    
    def hooked_get_bboxes(preds_dicts, img_metas, rescale=False):
        """Hook 版本的 get_bboxes，用于调试"""
        # 保存原始输入用于分析
        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']
        
        debug_info['all_cls_scores'] = all_cls_scores.detach().cpu()
        debug_info['all_bbox_preds'] = all_bbox_preds.detach().cpu()
        
        # 调用原始方法
        return original_get_bboxes(preds_dicts, img_metas, rescale)
    
    model.pts_bbox_head.get_bboxes = hooked_get_bboxes
    
    from mmcv.parallel import MMDataParallel
    model_wrapped = MMDataParallel(model, device_ids=[0])
    
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            if i >= args.num_samples:
                break
                
            print(f"\n{'='*80}")
            print(f"样本 {i + 1}/{args.num_samples}")
            print("="*80)
            
            # 重置 memory
            model.pts_bbox_head.reset_memory()
            
            # 使用标准的前向传播
            try:
                result = model_wrapped(return_loss=False, rescale=True, **data)
                
                # 分析 debug_info
                if 'all_cls_scores' in debug_info:
                    all_cls_scores = debug_info['all_cls_scores']  # [num_layers, B, N_q, num_cls]
                    all_bbox_preds = debug_info['all_bbox_preds']  # [num_layers, B, N_q, code_size]
                    
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
                    post_range = torch.tensor(bbox_coder_cfg.post_center_range)
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
                    
                    # 检查检测结果
                    print(f"\n[检测结果]")
                    if result and len(result) > 0:
                        pts_bbox = result[0].get('pts_bbox', {})
                        boxes_3d = pts_bbox.get('boxes_3d', None)
                        scores_3d = pts_bbox.get('scores_3d', None)
                        labels_3d = pts_bbox.get('labels_3d', None)
                        
                        if boxes_3d is not None:
                            print(f"  检测到的框数量: {len(boxes_3d)}")
                            if len(boxes_3d) > 0:
                                print(f"  框的分数范围: {scores_3d.min().item():.4f} - {scores_3d.max().item():.4f}")
                        else:
                            print(f"  检测到的框数量: 0")
                    else:
                        print(f"  结果为空")
                
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
