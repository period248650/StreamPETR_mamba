#!/usr/bin/env python
"""
调试脚本 v3：检查模型预测输出
直接 hook nms_free_coder 来捕获预测信息

使用方法：
cd /mnt/c/Users/17203/Desktop/Research/StreamPETR_mamba_new
PYTHONPATH="projects:$PYTHONPATH" python debug_predictions_v3.py \
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

# 全局变量用于存储调试信息
DEBUG_INFO = {
    'cls_scores_list': [],
    'bbox_preds_list': [],
    'final_boxes_list': [],
    'final_scores_list': [],
    'mask_info_list': [],
}


def parse_args():
    parser = argparse.ArgumentParser(description='Debug model predictions')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--num-samples', type=int, default=5, help='number of samples to check')
    return parser.parse_args()


def patch_nms_free_coder():
    """Patch NMSFreeCoder 来捕获调试信息"""
    from projects.mmdet3d_plugin.core.bbox.coders.nms_free_coder import NMSFreeCoder
    
    original_decode_single = NMSFreeCoder.decode_single
    
    def patched_decode_single(self, cls_scores, bbox_preds):
        """Patched version to capture debug info"""
        max_num = self.max_num

        cls_scores_sigmoid = cls_scores.sigmoid()
        scores, indexs = cls_scores_sigmoid.view(-1).topk(max_num)
        labels = indexs % self.num_classes
        bbox_index = torch.div(indexs, self.num_classes, rounding_mode='floor')
        bbox_preds_selected = bbox_preds[bbox_index]

        from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox
        final_box_preds = denormalize_bbox(bbox_preds_selected, self.pc_range)   
        final_scores = scores 
        final_preds = labels 

        # 保存调试信息
        DEBUG_INFO['cls_scores_list'].append({
            'raw_max': cls_scores.max().item(),
            'raw_min': cls_scores.min().item(),
            'sigmoid_max': cls_scores_sigmoid.max().item(),
            'sigmoid_mean': cls_scores_sigmoid.mean().item(),
            'topk_scores_max': scores.max().item(),
            'topk_scores_min': scores.min().item(),
        })
        
        DEBUG_INFO['bbox_preds_list'].append({
            'cx_range': (bbox_preds[:, 0].min().item(), bbox_preds[:, 0].max().item()),
            'cy_range': (bbox_preds[:, 1].min().item(), bbox_preds[:, 1].max().item()),
            'cz_range': (bbox_preds[:, 2].min().item(), bbox_preds[:, 2].max().item()),
            'selected_cx_range': (bbox_preds_selected[:, 0].min().item(), bbox_preds_selected[:, 0].max().item()),
            'selected_cy_range': (bbox_preds_selected[:, 1].min().item(), bbox_preds_selected[:, 1].max().item()),
            'selected_cz_range': (bbox_preds_selected[:, 2].min().item(), bbox_preds_selected[:, 2].max().item()),
        })

        # use score threshold
        if self.score_threshold is not None:
            thresh_mask = final_scores >= self.score_threshold
        
        if self.post_center_range is not None:
            self.post_center_range = torch.tensor(self.post_center_range, device=scores.device)
            
            mask = (final_box_preds[..., :3] >=
                    self.post_center_range[:3]).all(1)
            mask &= (final_box_preds[..., :3] <=
                     self.post_center_range[3:]).all(1)

            # 保存 mask 信息
            DEBUG_INFO['mask_info_list'].append({
                'total_before_filter': len(final_box_preds),
                'in_range_count': mask.sum().item(),
                'final_box_cx_range': (final_box_preds[:, 0].min().item(), final_box_preds[:, 0].max().item()),
                'final_box_cy_range': (final_box_preds[:, 1].min().item(), final_box_preds[:, 1].max().item()),
                'final_box_cz_range': (final_box_preds[:, 2].min().item(), final_box_preds[:, 2].max().item()),
            })

            if self.score_threshold:
                mask &= thresh_mask

            boxes3d = final_box_preds[mask]
            scores = final_scores[mask]
            labels = final_preds[mask]
            
            DEBUG_INFO['final_boxes_list'].append(len(boxes3d))
            DEBUG_INFO['final_scores_list'].append(scores.cpu().numpy() if len(scores) > 0 else [])
            
            predictions_dict = {
                'bboxes': boxes3d,
                'scores': scores,
                'labels': labels
            }

        else:
            raise NotImplementedError(
                'Need to reorganize output as a batch, only '
                'support post_center_range is not None for now!')
        return predictions_dict
    
    NMSFreeCoder.decode_single = patched_decode_single
    print("[Patch] NMSFreeCoder.decode_single has been patched for debugging")


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
    
    # 应用 patch
    patch_nms_free_coder()
    
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
    
    # 单卡调试：不使用 MMDataParallel，手动 scatter 数据
    from mmcv.parallel import scatter
    
    results = []
    
    for i, data in enumerate(data_loader):
        if i >= args.num_samples:
            break
            
        print(f"\n{'='*80}")
        print(f"样本 {i + 1}/{args.num_samples}")
        print("="*80)
        
        with torch.no_grad():
            # 手动 scatter 数据到 GPU
            data_gpu = scatter(data, [torch.cuda.current_device()])[0]
            result = model(return_loss=False, rescale=True, **data_gpu)
        
        results.extend(result)
        
        # 打印当前样本的调试信息
        if len(DEBUG_INFO['cls_scores_list']) > i:
            cls_info = DEBUG_INFO['cls_scores_list'][i]
            bbox_info = DEBUG_INFO['bbox_preds_list'][i]
            mask_info = DEBUG_INFO['mask_info_list'][i] if i < len(DEBUG_INFO['mask_info_list']) else None
            
            print(f"\n[分类分数统计]")
            print(f"  原始 logits: max={cls_info['raw_max']:.4f}, min={cls_info['raw_min']:.4f}")
            print(f"  Sigmoid 后: max={cls_info['sigmoid_max']:.6f}, mean={cls_info['sigmoid_mean']:.6f}")
            print(f"  Top-300 分数: max={cls_info['topk_scores_max']:.6f}, min={cls_info['topk_scores_min']:.6f}")
            
            print(f"\n[Bbox 坐标统计 (Head 输出, 真实坐标)]")
            print(f"  所有预测 cx: {bbox_info['cx_range'][0]:.2f} ~ {bbox_info['cx_range'][1]:.2f}")
            print(f"  所有预测 cy: {bbox_info['cy_range'][0]:.2f} ~ {bbox_info['cy_range'][1]:.2f}")
            print(f"  所有预测 cz: {bbox_info['cz_range'][0]:.2f} ~ {bbox_info['cz_range'][1]:.2f}")
            print(f"  Top-300 cx: {bbox_info['selected_cx_range'][0]:.2f} ~ {bbox_info['selected_cx_range'][1]:.2f}")
            print(f"  Top-300 cy: {bbox_info['selected_cy_range'][0]:.2f} ~ {bbox_info['selected_cy_range'][1]:.2f}")
            print(f"  Top-300 cz: {bbox_info['selected_cz_range'][0]:.2f} ~ {bbox_info['selected_cz_range'][1]:.2f}")
            
            if mask_info:
                print(f"\n[Post Center Range 过滤]")
                print(f"  过滤前数量: {mask_info['total_before_filter']}")
                print(f"  在范围内数量: {mask_info['in_range_count']}")
                print(f"  denormalize 后 cx: {mask_info['final_box_cx_range'][0]:.2f} ~ {mask_info['final_box_cx_range'][1]:.2f}")
                print(f"  denormalize 后 cy: {mask_info['final_box_cy_range'][0]:.2f} ~ {mask_info['final_box_cy_range'][1]:.2f}")
                print(f"  denormalize 后 cz: {mask_info['final_box_cz_range'][0]:.2f} ~ {mask_info['final_box_cz_range'][1]:.2f}")
            
            print(f"\n[最终检测结果]")
            if i < len(DEBUG_INFO['final_boxes_list']):
                print(f"  保留的框数量: {DEBUG_INFO['final_boxes_list'][i]}")
    
    print("\n" + "="*80)
    print("调试完成")
    print("="*80)
    
    # 总结
    print("\n[问题诊断]")
    if len(DEBUG_INFO['cls_scores_list']) > 0:
        avg_sigmoid_max = np.mean([x['sigmoid_max'] for x in DEBUG_INFO['cls_scores_list']])
        avg_in_range = np.mean([x['in_range_count'] for x in DEBUG_INFO['mask_info_list']]) if DEBUG_INFO['mask_info_list'] else 0
        avg_final = np.mean(DEBUG_INFO['final_boxes_list']) if DEBUG_INFO['final_boxes_list'] else 0
        
        print(f"  平均最大分类概率: {avg_sigmoid_max:.6f}")
        print(f"  平均在范围内的框数: {avg_in_range:.1f}")
        print(f"  平均最终保留的框数: {avg_final:.1f}")
        
        if avg_sigmoid_max < 0.1:
            print("\n  ⚠️ 问题: 分类概率太低！模型可能没有正确学习分类任务。")
        if avg_in_range == 0:
            print("\n  ⚠️ 问题: 所有预测都被 post_center_range 过滤掉了！检查坐标预测。")
        if avg_final == 0 and avg_in_range > 0:
            print("\n  ⚠️ 问题: 有框在范围内但最终数量为0，可能是 score_threshold 过滤。")


if __name__ == '__main__':
    main()
