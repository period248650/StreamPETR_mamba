#!/usr/bin/env python
"""
简化版调试脚本：直接分析 ISSM Decoder 参数
无需完整的 mmdet3d 依赖

使用方法：
cd /path/to/StreamPETR_mamba_new/projects
python debug_param_simple.py
"""

import sys
import os

# 添加必要路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 80)
print("ISSM Decoder 参数分析 - 简化版")
print("=" * 80)

try:
    import torch
    import torch.nn as nn
    print(f"PyTorch 版本: {torch.__version__}")
except ImportError:
    print("错误: 无法导入 torch")
    sys.exit(1)

# 目标参数索引（DDP 报错中的索引）
TARGET_INDICES = list(range(234, 250))
print(f"目标参数索引: {TARGET_INDICES}")
print()

# ============================================================================
# 直接导入 petr_issm 模块（避免完整的 mmdet3d_plugin 导入链）
# ============================================================================

try:
    # 尝试直接导入
    from mmdet3d_plugin.models.utils.petr_issm import (
        DenseAlternatingISSMDecoder,
        _SingleDirectionISSMLayer
    )
    print("✓ 成功导入 DenseAlternatingISSMDecoder")
except ImportError as e:
    print(f"导入失败: {e}")
    print("\n尝试手动分析参数结构...")
    
    # 手动计算参数数量
    print("\n" + "=" * 80)
    print("手动参数分析（基于代码结构）")
    print("=" * 80)
    
    d_model = 256
    d_state = 64
    d_conv = 4
    expand = 2
    num_heads = 8
    d_dist = 16
    num_layers = 6
    
    d_inner = expand * d_model  # 512
    
    print(f"\n配置参数:")
    print(f"  d_model = {d_model}")
    print(f"  d_state = {d_state}")
    print(f"  d_inner = {d_inner}")
    print(f"  num_heads = {num_heads}")
    print(f"  num_layers = {num_layers}")
    
    print(f"\n每层 _SingleDirectionISSMLayer 的参数:")
    
    # 计算每层参数
    params_per_layer = []
    
    # key_proj: Linear(d_model, d_inner*2 + 2 + num_heads)
    params_per_layer.append(("key_proj.weight", d_model * (d_inner*2 + 2 + num_heads)))
    
    # key_conv: Conv1d(d_inner + 2, d_inner + 2, kernel=d_conv, groups=d_inner+2)
    params_per_layer.append(("key_conv.weight", (d_inner + 2) * d_conv))
    params_per_layer.append(("key_conv.bias", d_inner + 2))
    
    # query_proj: Linear(d_model, d_inner)
    params_per_layer.append(("query_proj.weight", d_model * d_inner))
    
    # dist_encoder: Sequential(Linear, SiLU, Linear)
    params_per_layer.append(("dist_encoder.0.weight", d_model * (d_dist * 2)))
    params_per_layer.append(("dist_encoder.0.bias", d_dist * 2))
    params_per_layer.append(("dist_encoder.2.weight", (d_dist * 2) * d_dist))
    params_per_layer.append(("dist_encoder.2.bias", d_dist))
    
    # bc_proj: Linear(d_dist, 2, bias=False)
    params_per_layer.append(("bc_proj.weight", d_dist * 2))
    
    # dt_proj: Linear(d_dist, num_heads, bias=False)
    params_per_layer.append(("dt_proj.weight", d_dist * num_heads))
    
    # dt_bias: Parameter(num_heads)
    params_per_layer.append(("dt_bias", num_heads))
    
    # A_log: Parameter(num_heads)
    params_per_layer.append(("A_log", num_heads))
    
    # D: Parameter(num_heads)
    params_per_layer.append(("D", num_heads))
    
    # key_norm: LayerNorm(d_inner)
    params_per_layer.append(("key_norm.weight", d_inner))
    params_per_layer.append(("key_norm.bias", d_inner))
    
    # out_key_proj: Linear(d_inner, d_model)
    params_per_layer.append(("out_key_proj.weight", d_inner * d_model))
    
    # query_norm: LayerNorm(d_inner)
    params_per_layer.append(("query_norm.weight", d_inner))
    params_per_layer.append(("query_norm.bias", d_inner))
    
    # out_query_proj: Linear(d_inner, d_model)
    params_per_layer.append(("out_query_proj.weight", d_inner * d_model))
    
    # query_to_state: Linear(d_model, d_state)
    params_per_layer.append(("query_to_state.weight", d_model * d_state))
    
    print(f"\n  每层参数数量: {len(params_per_layer)}")
    for i, (name, size) in enumerate(params_per_layer):
        print(f"    [{i:2d}] {name}: {size}")
    
    total_per_layer = len(params_per_layer)
    total_decoder = total_per_layer * num_layers
    
    print(f"\n  6 层 ISSM Decoder 总参数数: {total_decoder}")
    
    # 计算目标索引对应的层和参数
    print(f"\n" + "=" * 80)
    print("目标索引分析 (234-249)")
    print("=" * 80)
    
    for idx in TARGET_INDICES:
        if idx < total_decoder:
            layer_idx = idx // total_per_layer
            param_idx = idx % total_per_layer
            param_name = params_per_layer[param_idx][0]
            print(f"  索引 {idx} -> layers.{layer_idx}.{param_name}")
        else:
            extra_idx = idx - total_decoder
            print(f"  索引 {idx} -> decoder 之后的参数 (可能是 box_refinement_layers 或其他模块)")
    
    sys.exit(0)

# ============================================================================
# 如果导入成功，进行完整分析
# ============================================================================

print("\n" + "=" * 80)
print("创建 DenseAlternatingISSMDecoder 实例")
print("=" * 80)

decoder = DenseAlternatingISSMDecoder(
    num_layers=6,
    d_model=256,
    d_state=64,
    d_conv=4,
    expand=2,
    num_views=6,
    feat_h=24,
    feat_w=44,
    num_heads=8,
    chunk_size=256,
    dropout=0.1,
    box_refinement=False,  # 与 head 中的设置一致
    fusion_type='add',
)

print("\n>>> ISSM Decoder 参数列表：")
print("-" * 100)

all_params = list(decoder.named_parameters())
total_params = len(all_params)

print(f"总参数数: {total_params}")
print()

# 只显示目标区域附近的参数
start_idx = max(0, min(TARGET_INDICES) - 10)
end_idx = min(total_params, max(TARGET_INDICES) + 10)

print(f"显示索引 {start_idx} - {end_idx} 的参数：")
print("-" * 100)

for i, (name, param) in enumerate(all_params):
    if start_idx <= i <= end_idx:
        marker = " *** 目标 ***" if i in TARGET_INDICES else ""
        print(f"[{i:4d}] {name:60s} shape={str(list(param.shape)):20s}{marker}")

# 额外检查：box_refinement_layers
print("\n" + "=" * 80)
print("模块状态检查")
print("=" * 80)
print(f"box_refinement: {decoder.box_refinement}")
print(f"box_refinement_layers: {decoder.box_refinement_layers}")
print(f"reorder: {decoder.reorder}")
print(f"fusion_layers: {decoder.fusion_layers}")
print(f"fusion_gates: {decoder.fusion_gates}")

print("\n" + "=" * 80)
print("诊断结论")
print("=" * 80)

if total_params < max(TARGET_INDICES):
    print(f"⚠ 目标索引 {TARGET_INDICES} 超出 ISSM Decoder 的参数范围 (0-{total_params-1})")
    print("  这意味着未使用的参数来自 ISSM Decoder 之外的模块（如 FocalHead）")
else:
    print("  目标索引在 ISSM Decoder 范围内，检查上面标记为 '*** 目标 ***' 的参数")
