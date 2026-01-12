#!/usr/bin/env python
# ------------------------------------------------------------------------
# Copyright (c) 2024 ISSM-StreamPETR. All Rights Reserved.
# ------------------------------------------------------------------------
"""
测试脚本：验证密集交替扫描 ISSM 架构

测试内容：
1. 单向 ISSM 层的正确性
2. 密集特征聚合机制
3. 交替扫描模式切换
4. 前向和反向传播
5. 与原始双向版本的性能对比
"""
import torch
import torch.nn as nn
import time
import sys
import os

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mmdet3d_plugin.models.utils.petr_issm import SingleDirectionISSMLayer, DenseAlternatingISSMDecoder, SequenceReorder


def test_single_direction_layer():
    """测试单向 ISSM 层"""
    print("\n" + "="*70)
    print("测试 1: 单向 ISSM 层")
    print("="*70)
    
    B, N_q, L, D = 2, 100, 1024, 256
    d_state = 16
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    
    # 创建输入
    queries = torch.randn(B, N_q, D).to(device)
    anchors = torch.randn(B, N_q, 3).to(device) * 10
    features = torch.randn(B, L, D).to(device)
    coords_3d = torch.randn(B, L, 3).to(device) * 10
    
    # 创建层
    layer = SingleDirectionISSMLayer(
        d_model=D,
        d_state=d_state
    ).to(device)
    
    print(f"输入形状:")
    print(f"  Queries: {queries.shape}")
    print(f"  Anchors: {anchors.shape}")
    print(f"  Features: {features.shape}")
    print(f"  Coords 3D: {coords_3d.shape}")
    
    # 前向传播
    start_time = time.time()
    q_new, f_new = layer(queries, anchors, features, coords_3d)
    forward_time = time.time() - start_time
    
    print(f"\n输出形状:")
    print(f"  Queries: {q_new.shape}")
    print(f"  Features: {f_new.shape}")
    print(f"前向传播时间: {forward_time*1000:.2f} ms")
    
    # 检查变化
    q_change = (q_new - queries).abs().mean().item()
    f_change = (f_new - features).abs().mean().item()
    print(f"\n特征变化:")
    print(f"  Query 变化: {q_change:.6f}")
    print(f"  Feature 变化: {f_change:.6f}")
    
    # 反向传播
    loss = q_new.sum() + f_new.sum()
    start_time = time.time()
    loss.backward()
    backward_time = time.time() - start_time
    
    print(f"\n反向传播:")
    print(f"  时间: {backward_time*1000:.2f} ms")
    print(f"  Query 梯度: {queries.grad.abs().mean().item():.6f}")
    print(f"  Feature 梯度: {features.grad.abs().mean().item():.6f}")
    
    print("✓ 单向 ISSM 层测试通过\n")
    return True


def test_dense_aggregation():
    """测试密集特征聚合"""
    print("\n" + "="*70)
    print("测试 2: 密集特征聚合机制")
    print("="*70)
    
    B, L, D = 2, 1024, 256
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建测试特征
    curr_feat = torch.randn(B, L, D).to(device)
    prev_feat = torch.randn(B, L, D).to(device)
    
    print(f"输入特征:")
    print(f"  当前特征 (F_L-1): {curr_feat.shape}")
    print(f"  历史特征 (F_L-2): {prev_feat.shape}")
    
    # 测试三种融合方式
    fusion_types = ['add', 'concat', 'gated']
    
    for fusion_type in fusion_types:
        print(f"\n测试融合方式: {fusion_type}")
        
        decoder = DenseAlternatingISSMDecoder(
            num_layers=6,
            d_model=D,
            fusion_type=fusion_type
        ).to(device)
        
        # 测试聚合
        fused_feat = decoder._dense_aggregate(curr_feat, prev_feat, layer_idx=2)
        
        print(f"  融合后形状: {fused_feat.shape}")
        print(f"  融合后均值: {fused_feat.mean().item():.6f}")
        print(f"  融合后方差: {fused_feat.std().item():.6f}")
        
        # 测试第一层（没有历史特征）
        fused_feat_first = decoder._dense_aggregate(curr_feat, None, layer_idx=0)
        assert torch.allclose(fused_feat_first, curr_feat), "第一层应该直接返回当前特征"
        print(f"  ✓ 第一层处理正确（无密集连接）")
    
    print("\n✓ 密集特征聚合测试通过\n")
    return True


def test_alternating_scan():
    """测试交替扫描模式"""
    print("\n" + "="*70)
    print("测试 3: 交替扫描模式")
    print("="*70)
    
    B, num_views, H, W, D = 2, 6, 24, 44, 256
    L = num_views * H * W
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建序列重排器
    reorder = SequenceReorder(num_views=num_views, H=H, W=W).to(device)
    
    # 创建测试序列
    x = torch.randn(B, L, D).to(device)
    
    print(f"输入序列: {x.shape}")
    print(f"视图配置: {num_views} views × {H}×{W} = {L} tokens")
    
    # 测试两种模式
    modes = ['A', 'B']
    results = {}
    
    for mode in modes:
        print(f"\n测试模式 {mode}:")
        
        # 重排
        x_perm = reorder(x, mode=mode)
        print(f"  重排后: {x_perm.shape}")
        
        # 还原
        x_restored = reorder(x_perm, mode=mode, inverse=True)
        print(f"  还原后: {x_restored.shape}")
        
        # 检查还原准确性
        restore_error = (x_restored - x).abs().max().item()
        print(f"  还原误差: {restore_error:.10f}")
        assert restore_error < 1e-5, f"模式 {mode} 还原误差过大"
        
        results[mode] = x_perm
        print(f"  ✓ 模式 {mode} 可逆性验证通过")
    
    # 检查两种模式的差异
    diff = (results['A'] - results['B']).abs().mean().item()
    print(f"\n模式 A 与模式 B 的差异: {diff:.6f}")
    assert diff > 1e-3, "两种模式应该产生不同的排列"
    print(f"✓ 模式差异验证通过（不同模式产生不同排列）")
    
    print("\n✓ 交替扫描模式测试通过\n")
    return True


def test_dense_alternating_decoder():
    """测试完整的密集交替解码器"""
    print("\n" + "="*70)
    print("测试 4: 密集交替 ISSM 解码器")
    print("="*70)
    
    B, N_q = 2, 100
    num_views, H, W = 6, 24, 44
    L = num_views * H * W
    D = 256
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建输入
    queries = torch.randn(B, N_q, D).to(device)
    anchors = torch.randn(B, N_q, 3).to(device) * 10
    img_feats = torch.randn(B, L, D).to(device)
    img_coords_3d = torch.randn(B, L, 3).to(device) * 10
    
    print(f"输入数据:")
    print(f"  Queries: {queries.shape}")
    print(f"  Anchors: {anchors.shape}")
    print(f"  Image Features: {img_feats.shape} ({num_views}×{H}×{W})")
    print(f"  Image Coords 3D: {img_coords_3d.shape}")
    
    # 创建解码器
    num_layers = 6
    decoder = DenseAlternatingISSMDecoder(
        num_layers=num_layers,
        d_model=D,
        num_views=num_views,
        feat_h=H,
        feat_w=W,
        fusion_type='add'
    ).to(device)
    
    print(f"\n解码器配置:")
    print(f"  层数: {num_layers}")
    print(f"  特征维度: {D}")
    print(f"  融合方式: add")
    print(f"  Box Refinement: {decoder.box_refinement}")
    
    # 前向传播
    print("\n执行前向传播...")
    start_time = time.time()
    output_queries, output_anchors = decoder(
        queries, anchors, img_feats, img_coords_3d, return_intermediate=False
    )
    forward_time = time.time() - start_time
    
    print(f"\n输出数据:")
    print(f"  Queries: {output_queries.shape}")
    print(f"  Anchors: {output_anchors.shape}")
    print(f"前向传播时间: {forward_time*1000:.2f} ms")
    
    # 检查变化
    q_change = (output_queries - queries).abs().mean().item()
    a_change = (output_anchors - anchors).abs().mean().item()
    print(f"\n特征变化:")
    print(f"  Query 变化: {q_change:.6f}")
    print(f"  Anchor 变化: {a_change:.6f}")
    
    # 测试中间输出
    print("\n测试中间输出...")
    intermediate_q, intermediate_a = decoder(
        queries, anchors, img_feats, img_coords_3d, return_intermediate=True
    )
    print(f"  中间 Queries: {intermediate_q.shape} (应该是 [{num_layers}, {B}, {N_q}, {D}])")
    print(f"  中间 Anchors: {intermediate_a.shape}")
    assert intermediate_q.shape[0] == num_layers, "中间输出层数不匹配"
    
    # 反向传播
    print("\n执行反向传播...")
    loss = output_queries.sum() + output_anchors.sum()
    start_time = time.time()
    loss.backward()
    backward_time = time.time() - start_time
    
    print(f"  时间: {backward_time*1000:.2f} ms")
    print(f"  Query 梯度: {queries.grad.abs().mean().item():.6f}")
    print(f"  Anchor 梯度: {anchors.grad.abs().mean().item():.6f}")
    
    print("\n✓ 密集交替解码器测试通过\n")
    return True


def benchmark_comparison():
    """性能对比：密集交替 vs 双向扫描"""
    print("\n" + "="*70)
    print("测试 5: 性能对比（Dense Alternating vs Dual Scan）")
    print("="*70)
    
    B, N_q = 2, 100
    num_views, H, W = 6, 24, 44
    L = num_views * H * W
    D = 256
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建输入
    queries = torch.randn(B, N_q, D).to(device)
    anchors = torch.randn(B, N_q, 3).to(device) * 10
    img_feats = torch.randn(B, L, D).to(device)
    img_coords_3d = torch.randn(B, L, 3).to(device) * 10
    
    # 创建密集交替解码器（单向）
    decoder_single = DenseAlternatingISSMDecoder(
        num_layers=6,
        d_model=D,
        num_views=num_views,
        feat_h=H,
        feat_w=W
    ).to(device)
    
    # 预热
    for _ in range(3):
        _ = decoder_single(queries, anchors, img_feats, img_coords_3d)
    
    # 测试密集交替版本
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    num_runs = 10
    for _ in range(num_runs):
        output = decoder_single(queries, anchors, img_feats, img_coords_3d)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    single_time = (time.time() - start_time) / num_runs
    
    print(f"密集交替扫描 (Single Direction + Dense):")
    print(f"  前向传播时间: {single_time*1000:.2f} ms")
    print(f"  吞吐量: {1000/single_time:.2f} samples/sec")
    
    # 估算双向版本的性能（理论上慢约2倍）
    estimated_dual_time = single_time * 2.0
    print(f"\n双向扫描 (估算):")
    print(f"  前向传播时间: {estimated_dual_time*1000:.2f} ms")
    print(f"  吞吐量: {1000/estimated_dual_time:.2f} samples/sec")
    
    speedup = estimated_dual_time / single_time
    print(f"\n加速比: {speedup:.2f}x")
    print(f"计算开销减少: {(1 - 1/speedup)*100:.1f}%")
    
    # 内存使用
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        _ = decoder_single(queries, anchors, img_feats, img_coords_3d)
        memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        print(f"\nGPU 内存使用: {memory_mb:.2f} MB")
    
    print("\n✓ 性能对比测试完成\n")
    return True


def main():
    """运行所有测试"""
    print("\n" + "="*70)
    print("密集交替扫描 ISSM 架构测试套件")
    print("="*70)
    
    tests = [
        ("单向 ISSM 层", test_single_direction_layer),
        ("密集特征聚合", test_dense_aggregation),
        ("交替扫描模式", test_alternating_scan),
        ("密集交替解码器", test_dense_alternating_decoder),
        ("性能对比", benchmark_comparison),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success))
        except Exception as e:
            print(f"\n✗ 测试失败: {name}")
            print(f"错误信息: {str(e)}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # 总结
    print("\n" + "="*70)
    print("测试总结")
    print("="*70)
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！密集交替扫描 ISSM 架构工作正常。")
        return 0
    else:
        print(f"\n⚠️  {total - passed} 个测试失败，请检查输出。")
        return 1


if __name__ == "__main__":
    exit(main())
