"""
调试脚本：精确定位参数索引 234-249 对应的模块

根据 DDP 报错信息：
Parameter indices which did not receive grad for rank 0: 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249
"""

import torch
import torch.nn as nn
import sys
sys.path.insert(0, '/mnt/c/Users/17203/Desktop/Research/StreamPETR_mamba_new/projects')

# 检查问题参数的索引范围
target_indices = list(range(234, 250))

print("=" * 80)
print("调试：定位未获得梯度的参数 (索引 234-249)")
print("=" * 80)

# 创建 ISSM Decoder 实例
from mmdet3d_plugin.models.utils.petr_issm import DenseAlternatingISSMDecoder

print("\n>>> 创建 DenseAlternatingISSMDecoder ...")
decoder = DenseAlternatingISSMDecoder(
    num_layers=6,
    d_model=256,
    d_state=64,
    num_views=6,
    feat_h=24,
    feat_w=44,
    num_heads=8,
    box_refinement=False,  # 与 head 中的设置一致
    fusion_type='add',
)

print("\n>>> Decoder 参数列表：")
print("-" * 100)
for i, (name, param) in enumerate(decoder.named_parameters()):
    marker = " *** 可能未使用 ***" if i in target_indices else ""
    print(f"[{i:3d}] {name:60s} shape={str(list(param.shape)):20s} grad={param.requires_grad}{marker}")

decoder_param_count = sum(1 for _ in decoder.parameters())
print(f"\n>>> Decoder 总参数数: {decoder_param_count}")

# 分析 box_refinement_layers
print("\n" + "=" * 80)
print("检查 box_refinement_layers 状态")
print("=" * 80)
print(f"box_refinement 启用: {decoder.box_refinement}")
print(f"box_refinement_layers: {decoder.box_refinement_layers}")

# 使用 box_refinement=True 再次检查
print("\n>>> 创建带 box_refinement=True 的 Decoder ...")
decoder_with_box = DenseAlternatingISSMDecoder(
    num_layers=6,
    d_model=256,
    d_state=64,
    num_views=6,
    feat_h=24,
    feat_w=44,
    num_heads=8,
    box_refinement=True,  # 启用
    fusion_type='add',
)

print("\n>>> 带 Box Refinement 的参数列表 (仅显示可能问题区域)：")
print("-" * 100)
for i, (name, param) in enumerate(decoder_with_box.named_parameters()):
    if i >= 220:  # 只显示后半部分
        marker = " *** 目标区域 ***" if i in target_indices else ""
        print(f"[{i:3d}] {name:60s} shape={str(list(param.shape)):20s}{marker}")

box_param_count = sum(1 for _ in decoder_with_box.parameters())
print(f"\n>>> 带 Box Refinement 的总参数数: {box_param_count}")
print(f">>> Box Refinement 增加的参数数: {box_param_count - decoder_param_count}")

