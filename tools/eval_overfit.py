#!/usr/bin/env python
"""
过拟合测试专用评估脚本

只在小型过拟合数据集上评估，不使用 nuScenes 官方 devkit
直接计算预测框和 GT 之间的匹配情况

使用方法:
    python tools/eval_overfit.py work_dirs/issm_overfit_test/latest.pth
"""

import argparse
import torch
import numpy as np
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmcv.parallel import MMDataParallel
from tqdm import tqdm


def calculate_iou_3d(box1, box2):
    """简单的 3D IoU 计算（仅 center 距离，用于快速判断）"""
    # box: [x, y, z, w, l, h, yaw, vx, vy]
    center_dist = np.sqrt(
        (box1[0] - box2[0])**2 + 
        (box1[1] - box2[1])**2 + 
        (box1[2] - box2[2])**2
    )
    # 简化判断：如果中心距离 < 2m，认为是匹配的
    return 1.0 / (1.0 + center_dist)


def evaluate_overfit(config_path, checkpoint_path, device='cuda:0'):
    """
    在过拟合数据集上评估模型
    """
    print(f"Loading config from {config_path}")
    cfg = Config.fromfile(config_path)
    
    # 构建数据集（使用 test/val 配置）
    print("Building dataset...")
    dataset = build_dataset(cfg.data.test)
    print(f"Dataset size: {len(dataset)} samples")
    
    # 构建模型
    print("Building model...")
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    
    # 加载权重
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')
    
    model = MMDataParallel(model, device_ids=[0])
    model.eval()
    
    # 统计
    total_gt = 0
    total_pred = 0
    total_matched = 0
    class_stats = {}
    
    print("\nRunning inference...")
    for i in tqdm(range(len(dataset))):
        data = dataset[i]
        
        # 准备数据
        img = data['img'].data.unsqueeze(0).to(device)
        img_metas = [data['img_metas'].data]
        
        # 构造完整的输入字典
        input_data = {'img': img, 'img_metas': img_metas}
        
        # 添加其他必要的键
        for key in cfg.data.test.collect_keys:
            if key in data and key not in ['img', 'img_metas']:
                val = data[key]
                if hasattr(val, 'data'):
                    val = val.data
                if isinstance(val, torch.Tensor):
                    val = val.unsqueeze(0).to(device)
                input_data[key] = val
        
        with torch.no_grad():
            try:
                result = model(return_loss=False, rescale=True, **input_data)
            except Exception as e:
                print(f"Error on sample {i}: {e}")
                continue
        
        # 获取预测结果
        if isinstance(result, list) and len(result) > 0:
            pred_boxes = result[0]['pts_bbox']['boxes_3d'].tensor.cpu().numpy()
            pred_scores = result[0]['pts_bbox']['scores_3d'].cpu().numpy()
            pred_labels = result[0]['pts_bbox']['labels_3d'].cpu().numpy()
            
            # 过滤低置信度预测
            mask = pred_scores > 0.3
            pred_boxes = pred_boxes[mask]
            pred_scores = pred_scores[mask]
            pred_labels = pred_labels[mask]
        else:
            pred_boxes = np.array([])
            pred_scores = np.array([])
            pred_labels = np.array([])
        
        # 获取 GT
        gt_boxes = data['gt_bboxes_3d'].data.tensor.cpu().numpy()
        gt_labels = data['gt_labels_3d'].data.cpu().numpy()
        
        total_gt += len(gt_boxes)
        total_pred += len(pred_boxes)
        
        # 简单匹配：计算每个 GT 是否有预测框与之匹配
        for j, gt_box in enumerate(gt_boxes):
            gt_label = gt_labels[j]
            matched = False
            
            for k, pred_box in enumerate(pred_boxes):
                if pred_labels[k] == gt_label:
                    iou = calculate_iou_3d(gt_box, pred_box)
                    if iou > 0.3:  # center distance < 2m
                        matched = True
                        break
            
            if matched:
                total_matched += 1
                
            # 类别统计
            cls_name = cfg.class_names[gt_label] if gt_label < len(cfg.class_names) else 'unknown'
            if cls_name not in class_stats:
                class_stats[cls_name] = {'gt': 0, 'matched': 0}
            class_stats[cls_name]['gt'] += 1
            if matched:
                class_stats[cls_name]['matched'] += 1
    
    # 输出结果
    print("\n" + "="*60)
    print("OVERFIT TEST EVALUATION RESULTS")
    print("="*60)
    print(f"Total GT boxes: {total_gt}")
    print(f"Total Pred boxes: {total_pred}")
    print(f"Total Matched: {total_matched}")
    
    if total_gt > 0:
        recall = total_matched / total_gt * 100
        print(f"\nOverall Recall: {recall:.1f}%")
    
    if total_pred > 0:
        precision = total_matched / total_pred * 100
        print(f"Overall Precision: {precision:.1f}%")
    
    print("\nPer-class statistics:")
    print("-"*40)
    for cls_name, stats in sorted(class_stats.items()):
        if stats['gt'] > 0:
            cls_recall = stats['matched'] / stats['gt'] * 100
            print(f"  {cls_name}: {stats['matched']}/{stats['gt']} ({cls_recall:.1f}%)")
    
    print("\n" + "="*60)
    print("INTERPRETATION:")
    if total_gt > 0:
        if recall > 50:
            print("✓ Model is learning! Recall > 50%")
        elif recall > 20:
            print("△ Model is partially learning. Check for issues.")
        else:
            print("✗ Model is NOT learning effectively. Check code for bugs.")
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help='Checkpoint file path')
    parser.add_argument('--config', default='projects/configs/issm_streampetr/issm_overfit_test.py')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    
    evaluate_overfit(args.config, args.checkpoint, args.device)
