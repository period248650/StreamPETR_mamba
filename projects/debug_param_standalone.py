#!/usr/bin/env python
"""
独立调试脚本：分析 DDP 未使用参数
完全不依赖 mmdet3d_plugin 的导入

使用方法：
python debug_param_standalone.py
"""

import torch
import torch.nn as nn

print("=" * 80)
print("DDP 未使用参数分析 - 独立版本")
print("=" * 80)
print(f"PyTorch 版本: {torch.__version__}")

# ============================================================================
# 基于代码分析的参数索引映射
# ============================================================================

# 配置参数（与 issm_streampetr_dense_alternating.py 一致）
d_model = 256
d_state = 64
d_conv = 4
expand = 2
num_heads = 8
d_dist = 16
num_layers = 6
d_inner = expand * d_model  # 512

print(f"\n配置:")
print(f"  d_model={d_model}, d_state={d_state}, d_inner={d_inner}")
print(f"  num_heads={num_heads}, num_layers={num_layers}")

# ============================================================================
# 构建完整的参数索引映射
# ============================================================================

# ISSMStreamPETRHead 的参数结构（按照 PyTorch 的遍历顺序）
# 注意：这是基于代码分析的估算

print("\n" + "=" * 80)
print("ISSMStreamPETRHead 参数结构分析")
print("=" * 80)

param_list = []
idx = 0

# 1. cls_branches (6 层)
print("\n[cls_branches] 分类头")
for layer_idx in range(6):
    # 每个 cls_branch: Linear -> LN -> ReLU -> Linear -> LN -> ReLU -> Linear(out)
    # num_reg_fcs=2, 所以有 2 个 (Linear + LN + ReLU) + 1 个 Linear(out)
    param_list.append(f"cls_branches.{layer_idx}.0.weight (Linear)")
    param_list.append(f"cls_branches.{layer_idx}.0.bias")
    param_list.append(f"cls_branches.{layer_idx}.1.weight (LN)")
    param_list.append(f"cls_branches.{layer_idx}.1.bias (LN)")
    param_list.append(f"cls_branches.{layer_idx}.3.weight (Linear)")
    param_list.append(f"cls_branches.{layer_idx}.3.bias")
    param_list.append(f"cls_branches.{layer_idx}.4.weight (LN)")
    param_list.append(f"cls_branches.{layer_idx}.4.bias (LN)")
    param_list.append(f"cls_branches.{layer_idx}.6.weight (Linear out)")
    param_list.append(f"cls_branches.{layer_idx}.6.bias")

print(f"  参数数: {len(param_list)} (每层 10 个, 共 6 层)")

cls_end = len(param_list)

# 2. reg_branches (6 层)
print("\n[reg_branches] 回归头")
reg_start = len(param_list)
for layer_idx in range(6):
    # 每个 reg_branch: 2 * (Linear + ReLU) + Linear(out)
    param_list.append(f"reg_branches.{layer_idx}.0.weight (Linear)")
    param_list.append(f"reg_branches.{layer_idx}.0.bias")
    param_list.append(f"reg_branches.{layer_idx}.2.weight (Linear)")
    param_list.append(f"reg_branches.{layer_idx}.2.bias")
    param_list.append(f"reg_branches.{layer_idx}.4.weight (Linear out)")
    param_list.append(f"reg_branches.{layer_idx}.4.bias")

print(f"  参数数: {len(param_list) - reg_start} (每层 6 个, 共 6 层)")

reg_end = len(param_list)

# 3. position_encoder
print("\n[position_encoder] 位置编码器")
pos_start = len(param_list)
param_list.append("position_encoder.0.weight")
param_list.append("position_encoder.0.bias")
param_list.append("position_encoder.2.weight")
param_list.append("position_encoder.2.bias")
print(f"  参数数: {len(param_list) - pos_start}")

# 4. memory_embed
print("\n[memory_embed] Memory 编码器")
mem_start = len(param_list)
param_list.append("memory_embed.0.weight")
param_list.append("memory_embed.0.bias")
param_list.append("memory_embed.2.weight")
param_list.append("memory_embed.2.bias")
print(f"  参数数: {len(param_list) - mem_start}")

# 5. featurized_pe (SELayer_Linear)
print("\n[featurized_pe] SELayer")
se_start = len(param_list)
param_list.append("featurized_pe.fc1.weight")
param_list.append("featurized_pe.fc1.bias")
param_list.append("featurized_pe.fc2.weight")
param_list.append("featurized_pe.fc2.bias")
print(f"  参数数: {len(param_list) - se_start}")

# 6. reference_points (Embedding)
print("\n[reference_points] Query Embedding")
param_list.append("reference_points.weight")

# 7. pseudo_reference_points (Embedding) - requires_grad=False，不计入
print("\n[pseudo_reference_points] Pseudo Embedding (requires_grad=False, 不计入)")

# 8. query_embedding
print("\n[query_embedding] Query Embedding MLP")
qe_start = len(param_list)
param_list.append("query_embedding.0.weight")
param_list.append("query_embedding.0.bias")
param_list.append("query_embedding.2.weight")
param_list.append("query_embedding.2.bias")
print(f"  参数数: {len(param_list) - qe_start}")

# 9. spatial_alignment (MLN)
print("\n[spatial_alignment] MLN")
sa_start = len(param_list)
param_list.append("spatial_alignment.fc1.weight")
param_list.append("spatial_alignment.fc1.bias")
param_list.append("spatial_alignment.fc2.weight")
param_list.append("spatial_alignment.fc2.bias")
param_list.append("spatial_alignment.fc3.weight")
param_list.append("spatial_alignment.fc3.bias")
print(f"  参数数: {len(param_list) - sa_start}")

# 10. time_embedding
print("\n[time_embedding] 时间编码")
te_start = len(param_list)
param_list.append("time_embedding.0.weight")
param_list.append("time_embedding.0.bias")
param_list.append("time_embedding.1.weight (LN)")
param_list.append("time_embedding.1.bias (LN)")
print(f"  参数数: {len(param_list) - te_start}")

# 11. ego_pose_pe (MLN)
print("\n[ego_pose_pe] Ego Pose PE (MLN)")
ep_start = len(param_list)
param_list.append("ego_pose_pe.fc1.weight")
param_list.append("ego_pose_pe.fc1.bias")
param_list.append("ego_pose_pe.fc2.weight")
param_list.append("ego_pose_pe.fc2.bias")
param_list.append("ego_pose_pe.fc3.weight")
param_list.append("ego_pose_pe.fc3.bias")
print(f"  参数数: {len(param_list) - ep_start}")

# 12. ego_pose_memory (MLN)
print("\n[ego_pose_memory] Ego Pose Memory (MLN)")
em_start = len(param_list)
param_list.append("ego_pose_memory.fc1.weight")
param_list.append("ego_pose_memory.fc1.bias")
param_list.append("ego_pose_memory.fc2.weight")
param_list.append("ego_pose_memory.fc2.bias")
param_list.append("ego_pose_memory.fc3.weight")
param_list.append("ego_pose_memory.fc3.bias")
print(f"  参数数: {len(param_list) - em_start}")

head_params_before_issm = len(param_list)
print(f"\n>>> Head 参数 (ISSM Decoder 之前): {head_params_before_issm}")

# 13. issm_decoder (DenseAlternatingISSMDecoder)
print("\n[issm_decoder] ISSM Decoder")
issm_start = len(param_list)

# 每层 _SingleDirectionISSMLayer 的参数
params_per_issm_layer = [
    "key_proj.weight",
    "key_conv.weight",
    "key_conv.bias",
    "query_proj.weight",
    "dist_encoder.0.weight",
    "dist_encoder.0.bias",
    "dist_encoder.2.weight",
    "dist_encoder.2.bias",
    "bc_proj.weight",
    "dt_proj.weight",
    "dt_bias",
    "A_log",
    "D",
    "key_norm.weight",
    "key_norm.bias",
    "out_key_proj.weight",
    "query_norm.weight",
    "query_norm.bias",
    "out_query_proj.weight",
    "query_to_state.weight",
]

for layer_idx in range(6):
    for param_name in params_per_issm_layer:
        param_list.append(f"issm_decoder.layers.{layer_idx}.{param_name}")

print(f"  每层参数数: {len(params_per_issm_layer)}")
print(f"  6 层总参数数: {len(param_list) - issm_start}")

total_head_params = len(param_list)
print(f"\n>>> ISSMStreamPETRHead 总参数数 (估算): {total_head_params}")

# ============================================================================
# 分析目标索引
# ============================================================================

TARGET_INDICES = list(range(234, 250))

print("\n" + "=" * 80)
print(f"目标索引分析: {TARGET_INDICES}")
print("=" * 80)

for idx in TARGET_INDICES:
    if idx < len(param_list):
        print(f"  [{idx:4d}] {param_list[idx]}")
    else:
        print(f"  [{idx:4d}] 超出 Head 范围 - 可能在 img_backbone/img_neck/img_roi_head 中")

# ============================================================================
# 结论
# ============================================================================

print("\n" + "=" * 80)
print("诊断结论")
print("=" * 80)

# 检查目标索引落在哪个模块
if max(TARGET_INDICES) < head_params_before_issm:
    print("⚠ 目标索引在 ISSMStreamPETRHead 的基础模块中（cls_branches, position_encoder 等）")
elif min(TARGET_INDICES) >= issm_start and max(TARGET_INDICES) < total_head_params:
    layer_idx = (min(TARGET_INDICES) - issm_start) // len(params_per_issm_layer)
    print(f"⚠ 目标索引在 ISSM Decoder 的第 {layer_idx} 层")
    print(f"  可能原因: ISSM 层的某些参数未获得梯度（如 A_log, D, dt_bias）")
elif max(TARGET_INDICES) >= total_head_params:
    print("⚠ 目标索引超出 Head 范围")
    print("  可能原因: img_roi_head (FocalHead) 未被调用，其参数无梯度")
    print("  推荐修复: 在 forward_pts_train 中添加 2D 辅助损失计算")

print("\n" + "=" * 80)
print("重要提醒")
print("=" * 80)
print("""
上述分析是基于代码结构的估算，实际参数索引可能略有不同。

为了精确定位，请在训练代码中添加以下调试代码：

```python
# 在 train_detector.py 或 runner 中，模型构建后添加：
for i, (name, p) in enumerate(model.named_parameters()):
    if i in range(234, 250):
        print(f"[{i}] {name} requires_grad={p.requires_grad}")
```

或者设置环境变量获取更多信息：
```bash
export TORCH_DISTRIBUTED_DEBUG=DETAIL
```
""")
