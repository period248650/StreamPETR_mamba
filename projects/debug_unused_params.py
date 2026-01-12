#!/usr/bin/env python
"""
调试脚本：精确定位 DDP 未使用参数

使用方法：
1. 在训练前运行此脚本，获取参数索引到名称的映射
2. 对照 DDP 报错中的参数索引，找到具体模块

命令行：
python projects/debug_unused_params.py configs/issm_streampetr/issm_streampetr_dense_alternating.py
"""

import sys
import argparse
import torch
import torch.nn as nn

# 添加项目路径
sys.path.insert(0, 'projects/')

def parse_args():
    parser = argparse.ArgumentParser(description='Debug unused parameters')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--target-indices', nargs='+', type=int, 
                        default=list(range(234, 250)),
                        help='Target parameter indices to debug')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 延迟导入，确保路径正确
    try:
        from mmcv import Config
        from mmdet3d.models import build_detector
    except ImportError:
        print("请确保已安装 mmcv 和 mmdet3d")
        return
    
    print("=" * 80)
    print("DDP 未使用参数调试器")
    print("=" * 80)
    print(f"配置文件: {args.config}")
    print(f"目标参数索引: {args.target_indices}")
    print()
    
    # 加载配置
    cfg = Config.fromfile(args.config)
    
    # 构建模型
    model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    
    # 列出所有参数
    print("=" * 80)
    print("参数索引到模块名称映射（目标区域）")
    print("=" * 80)
    
    target_params = []
    for i, (name, param) in enumerate(model.named_parameters()):
        if i in args.target_indices:
            target_params.append((i, name, param.shape, param.requires_grad))
            print(f"[{i:4d}] {name}")
            print(f"       shape={list(param.shape)}, requires_grad={param.requires_grad}")
    
    print()
    print("=" * 80)
    print("诊断结果")
    print("=" * 80)
    
    # 分析模块来源
    modules_found = set()
    for idx, name, shape, req_grad in target_params:
        parts = name.split('.')
        if len(parts) >= 2:
            module_prefix = '.'.join(parts[:2])
            modules_found.add(module_prefix)
    
    print(f"受影响的顶层模块: {modules_found}")
    print()
    
    # 给出修复建议
    print("修复建议：")
    for module in modules_found:
        if 'img_roi_head' in module:
            print(f"  - {module}: 2D 辅助检测头。检查是否所有 iteration 都使用 2D loss")
        elif 'pts_bbox_head' in module:
            print(f"  - {module}: 3D 检测头。检查 ISSM decoder 参数是否全部参与 loss")
        elif 'img_backbone' in module:
            print(f"  - {module}: 图像骨干网络。检查是否有冻结层配置问题")
        elif 'img_neck' in module:
            print(f"  - {module}: 颈部网络。检查输出是否全部使用")
        else:
            print(f"  - {module}: 请检查该模块是否有条件分支导致部分参数未使用")
    
    print()
    print("=" * 80)
    print("完整参数列表（可选查看）")
    print("=" * 80)
    total_params = sum(1 for _ in model.parameters())
    print(f"模型总参数数: {total_params}")
    print()
    print("运行以下命令查看完整参数列表:")
    print("  python projects/debug_unused_params.py <config> --target-indices $(seq 0 300)")

if __name__ == '__main__':
    main()
