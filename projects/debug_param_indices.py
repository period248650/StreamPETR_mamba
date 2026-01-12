"""
调试脚本：列出 ISSM-StreamPETR Head 的所有参数及其索引

用于诊断 DDP 未使用参数问题
"""

import torch
import torch.nn as nn
import sys
sys.path.insert(0, '/mnt/c/Users/17203/Desktop/Research/StreamPETR_mamba_new/projects')

from mmdet3d_plugin.models.dense_heads.issm_streampetr_head import ISSMStreamPETRHead
from mmdet3d_plugin.models.utils.petr_issm import DenseAlternatingISSMDecoder

def list_all_parameters():
    """列出 Head 和 ISSM Decoder 的所有参数"""
    
    # 创建一个简单的 Head 实例
    # 需要提供必要的配置
    head_cfg = dict(
        type='ISSMStreamPETRHead',
        num_classes=10,
        in_channels=256,
        num_query=300,
        num_pred=6,
        memory_len=512,
        topk_proposals=128,
        num_propagated=128,
        with_ego_pos=True,
        use_issm=True,
        issm_cfg=dict(
            d_model=256,
            d_state=64,
            d_conv=4,
            expand=2,
            num_heads=8,
            d_dist=16,
        ),
        code_weights=[2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.25, 0.25],
        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        LID=True,
        with_position=True,
        position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        scalar=10,
        noise_scale=1.0,
        noise_trans=0.0,
        dn_weight=1.0,
        split=0.75,
    )
    
    # 仅列出 ISSM Decoder 的参数
    print("=" * 80)
    print("ISSM Decoder 参数列表")
    print("=" * 80)
    
    decoder = DenseAlternatingISSMDecoder(
        num_layers=6,
        d_model=256,
        d_state=64,
        d_conv=4,
        expand=2,
        num_heads=8,
        d_dist=16,
    )
    
    param_idx = 0
    for name, param in decoder.named_parameters():
        print(f"[{param_idx:3d}] {name:50s} shape={list(param.shape):20s} requires_grad={param.requires_grad}")
        param_idx += 1
    
    print(f"\n总参数数: {param_idx}")
    
    # 识别可能未使用的参数类型
    print("\n" + "=" * 80)
    print("参数分类分析")
    print("=" * 80)
    
    for layer_idx in range(6):
        layer = decoder.layers[layer_idx]
        print(f"\n--- Layer {layer_idx} ---")
        
        # dist_encoder 参数
        for i, sub_layer in enumerate(layer.dist_encoder):
            if hasattr(sub_layer, 'weight'):
                print(f"  dist_encoder.{i}.weight: {list(sub_layer.weight.shape)}")
            if hasattr(sub_layer, 'bias') and sub_layer.bias is not None:
                print(f"  dist_encoder.{i}.bias: {list(sub_layer.bias.shape)}")
        
        # SSM 相关参数
        print(f"  bc_proj.weight: {list(layer.bc_proj.weight.shape)}")
        print(f"  dt_proj.weight: {list(layer.dt_proj.weight.shape)}")
        print(f"  A_log: {list(layer.A_log.shape)}")
        print(f"  D: {list(layer.D.shape)}")
        print(f"  dt_bias: {list(layer.dt_bias.shape)}")
    
    # 如果 decoder 中有 reorder, fusion_layers, fusion_gates
    if hasattr(decoder, 'reorder') and decoder.reorder is not None:
        print("\n--- Reorder 参数 ---")
        for name, param in decoder.reorder.named_parameters():
            print(f"  {name}: {list(param.shape)}")
    else:
        print("\n--- Reorder: None ---")
    
    if hasattr(decoder, 'fusion_layers') and decoder.fusion_layers is not None:
        print("\n--- Fusion Layers 参数 ---")
        for name, param in decoder.fusion_layers.named_parameters():
            print(f"  {name}: {list(param.shape)}")
    else:
        print("\n--- Fusion Layers: None ---")

if __name__ == "__main__":
    list_all_parameters()
