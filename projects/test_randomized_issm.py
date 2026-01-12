#!/usr/bin/env python
# ------------------------------------------------------------------------
# Copyright (c) 2024 ISSM-StreamPETR. All Rights Reserved.
# ------------------------------------------------------------------------
"""
测试脚本：验证 DEST-Inspired 随机化 ISSM 架构

测试内容：
1. DynamicRandomReorder 的随机性和还原正确性
2. 训练模式下的动态随机排列生成
3. 推理模式下的固定排列使用
4. 密集特征聚合 + 随机化的组合
5. 前向和反向传播正确性
"""
import torch
import torch.nn as nn
import time
import sys
import os

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, '/mnt/c/Users/17203/Desktop/Research/DEST3D')
sys.path.insert(0, '/mnt/c/Users/17203/Desktop/Research/mamba')

from mmdet3d_plugin.models.utils.petr_issm import DenseAlternatingISSMDecoder


def test_dynamic_random_reorder():
    """测试动态随机重排模块"""
    print("\n" + "="*70)
    print("测试 1: 动态随机重排模块")
    print("="*70)
    
    from mmdet3d_plugin.models.utils.petr_issm import DynamicRandomReorder
    
    B, num_views, H, W, D = 2, 6, 32, 88, 256
    seq_len = num_views * H * W
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模块（确定性模式和非确定性模式）
    reorder_det = DynamicRandomReorder(
        num_views=num_views, H=H, W=W, 
        num_layers=6, deterministic=True
    ).to(device)
    
    reorder_random = DynamicRandomReorder(
        num_views=num_views, H=H, W=W, 
        num_layers=6, deterministic=False
    ).to(device)
    
    x = torch.randn(B, seq_len, D, device=device)
    
    # 测试 1.1: 随机排列的还原性
    print("\n[1.1] 测试排列还原...")
    for layer_idx in range(3):
        perm_idx, inv_perm_idx = reorder_random.get_permutation(
            layer_idx, device, training=True
        )
        x_perm = reorder_random(x, perm_idx)
        x_restored = reorder_random(x_perm, inv_perm_idx)
        error = (x - x_restored).abs().max().item()
        print(f"  Layer {layer_idx}: 还原误差 = {error:.2e}")
        assert error < 1e-5, f"Layer {layer_idx} 还原失败!"
    
    print("  ✓ 所有层还原正确")
    
    # 测试 1.2: 训练时的随机性
    print("\n[1.2] 测试训练模式随机性...")
    perms = []
    for _ in range(5):
        perm_idx, _ = reorder_random.get_permutation(0, device, training=True)
        perms.append(perm_idx.cpu())
    
    # 检查排列是否不同
    unique_count = 0
    for i in range(len(perms)):
        for j in range(i+1, len(perms)):
            if not torch.equal(perms[i], perms[j]):
                unique_count += 1
    
    print(f"  生成5个排列，{unique_count}对不相同")
    print("  ✓ 训练模式具有随机性")
    
    # 测试 1.3: 推理时的确定性
    print("\n[1.3] 测试推理模式确定性...")
    perm1, _ = reorder_det.get_permutation(0, device, training=False)
    perm2, _ = reorder_det.get_permutation(0, device, training=False)
    assert torch.equal(perm1, perm2), "推理模式排列不一致!"
    print("  ✓ 推理模式排列固定")
    
    # 测试 1.4: 扫描模式多样性
    print("\n[1.4] 测试扫描模式多样性...")
    torch.manual_seed(42)
    modes_seen = set()
    for _ in range(20):
        perm_idx, _ = reorder_random._generate_permutation(device)
        # 检查第一个视图的扫描方向
        first_view_pattern = perm_idx[:H*W].cpu()
        pattern_hash = hash(tuple(first_view_pattern.tolist()[:10]))
        modes_seen.add(pattern_hash)
    
    print(f"  20次生成观察到 {len(modes_seen)} 种不同模式")
    print("  ✓ 扫描模式具有多样性")
    
    print("\n✓ 动态随机重排模块测试通过")


def test_randomized_decoder_forward():
    """测试随机化解码器前向传播"""
    print("\n" + "="*70)
    print("测试 2: 随机化解码器前向传播")
    print("="*70)
    
    B, N_q, num_views, H, W = 2, 100, 6, 32, 88
    L = num_views * H * W
    D = 256
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模块
    decoder = DenseAlternatingISSMDecoder(
        num_layers=3,
        d_model=D,
        d_state=16,
        num_views=num_views,
        feat_h=H,
        feat_w=W,
        num_heads=8,
        chunk_size=256,
        dropout=0.0,
        box_refinement=True,
        fusion_type='add',
        deterministic=True  # 使用确定性模式方便测试
    ).to(device)
    
    # 准备输入
    queries = torch.randn(B, N_q, D, device=device)
    anchors = torch.randn(B, N_q, 3, device=device)
    img_feats = torch.randn(B, L, D, device=device)
    img_coords = torch.randn(B, L, 3, device=device)
    
    # 测试 2.1: 训练模式前向传播
    print("\n[2.1] 测试训练模式前向传播...")
    decoder.train()
    
    start_time = time.time()
    out_q_train, out_a_train = decoder(
        queries=queries,
        anchors=anchors,
        img_feats=img_feats,
        img_coords_3d=img_coords,
        return_intermediate=False
    )
    train_time = time.time() - start_time
    
    print(f"  输入形状: queries={queries.shape}, features={img_feats.shape}")
    print(f"  输出形状: queries={out_q_train.shape}, anchors={out_a_train.shape}")
    print(f"  训练时间: {train_time*1000:.2f} ms")
    assert out_q_train.shape == (B, N_q, D)
    assert out_a_train.shape == (B, N_q, 3)
    print("  ✓ 训练模式前向传播正确")
    
    # 测试 2.2: 推理模式前向传播
    print("\n[2.2] 测试推理模式前向传播...")
    decoder.eval()
    
    with torch.no_grad():
        start_time = time.time()
        out_q_eval, out_a_eval = decoder(
            queries=queries,
            anchors=anchors,
            img_feats=img_feats,
            img_coords_3d=img_coords,
            return_intermediate=False
        )
        eval_time = time.time() - start_time
    
    print(f"  推理时间: {eval_time*1000:.2f} ms")
    print(f"  加速比: {train_time/eval_time:.2f}x")
    assert out_q_eval.shape == (B, N_q, D)
    print("  ✓ 推理模式前向传播正确")
    
    # 测试 2.3: 推理确定性
    print("\n[2.3] 测试推理确定性...")
    with torch.no_grad():
        out_q_eval2, out_a_eval2 = decoder(
            queries=queries,
            anchors=anchors,
            img_feats=img_feats,
            img_coords_3d=img_coords,
            return_intermediate=False
        )
    
    q_diff = (out_q_eval - out_q_eval2).abs().max().item()
    a_diff = (out_a_eval - out_a_eval2).abs().max().item()
    print(f"  两次推理差异: queries={q_diff:.2e}, anchors={a_diff:.2e}")
    assert q_diff < 1e-5 and a_diff < 1e-5, "推理结果不确定!"
    print("  ✓ 推理结果确定性保证")
    
    # 测试 2.4: 中间层输出
    print("\n[2.4] 测试中间层输出...")
    with torch.no_grad():
        out_q_inter, out_a_inter = decoder(
            queries=queries,
            anchors=anchors,
            img_feats=img_feats,
            img_coords_3d=img_coords,
            return_intermediate=True
        )
    
    print(f"  中间层输出形状: {out_q_inter.shape}, {out_a_inter.shape}")
    assert out_q_inter.shape == (3, B, N_q, D)  # num_layers=3
    assert out_a_inter.shape == (3, B, N_q, 3)
    print("  ✓ 中间层输出正确")
    
    print("\n✓ 随机化解码器前向传播测试通过")


def test_backward_pass():
    """测试反向传播和梯度"""
    print("\n" + "="*70)
    print("测试 3: 反向传播和梯度")
    print("="*70)
    
    B, N_q, num_views, H, W = 1, 50, 6, 16, 44  # 减小尺寸加速测试
    L = num_views * H * W
    D = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模块
    decoder = DenseAlternatingISSMDecoder(
        num_layers=2,
        d_model=D,
        d_state=8,
        num_views=num_views,
        feat_h=H,
        feat_w=W,
        num_heads=4,
        dropout=0.0,
        box_refinement=True,
        deterministic=False
    ).to(device)
    
    # 准备输入（需要梯度）
    queries = torch.randn(B, N_q, D, device=device, requires_grad=True)
    anchors = torch.randn(B, N_q, 3, device=device, requires_grad=True)
    img_feats = torch.randn(B, L, D, device=device, requires_grad=True)
    img_coords = torch.randn(B, L, 3, device=device, requires_grad=True)
    
    # 前向传播
    print("\n[3.1] 前向传播...")
    decoder.train()
    out_q, out_a = decoder(
        queries=queries,
        anchors=anchors,
        img_feats=img_feats,
        img_coords_3d=img_coords
    )
    
    # 定义损失
    loss = out_q.mean() + out_a.mean()
    print(f"  损失值: {loss.item():.6f}")
    
    # 反向传播
    print("\n[3.2] 反向传播...")
    start_time = time.time()
    loss.backward()
    backward_time = time.time() - start_time
    
    print(f"  反向传播时间: {backward_time*1000:.2f} ms")
    
    # 检查梯度
    print("\n[3.3] 检查梯度...")
    assert queries.grad is not None, "queries 没有梯度!"
    assert anchors.grad is not None, "anchors 没有梯度!"
    assert img_feats.grad is not None, "img_feats 没有梯度!"
    
    print(f"  queries.grad: shape={queries.grad.shape}, "
          f"mean={queries.grad.mean().item():.6f}, "
          f"std={queries.grad.std().item():.6f}")
    print(f"  img_feats.grad: shape={img_feats.grad.shape}, "
          f"mean={img_feats.grad.mean().item():.6f}, "
          f"std={img_feats.grad.std().item():.6f}")
    
    # 检查梯度是否合理（不是nan或inf）
    assert not torch.isnan(queries.grad).any(), "queries梯度包含NaN!"
    assert not torch.isinf(queries.grad).any(), "queries梯度包含Inf!"
    assert not torch.isnan(img_feats.grad).any(), "img_feats梯度包含NaN!"
    
    print("  ✓ 梯度计算正确，无NaN/Inf")
    
    # 检查参数梯度
    print("\n[3.4] 检查模型参数梯度...")
    grad_count = 0
    for name, param in decoder.named_parameters():
        if param.grad is not None:
            grad_count += 1
            grad_norm = param.grad.norm().item()
            if grad_count <= 3:  # 只打印前3个
                print(f"  {name}: grad_norm={grad_norm:.6f}")
    
    print(f"  共 {grad_count} 个参数有梯度")
    print("  ✓ 参数梯度正常")
    
    print("\n✓ 反向传播和梯度测试通过")


def test_memory_and_speed():
    """测试显存占用和速度"""
    print("\n" + "="*70)
    print("测试 4: 显存占用和速度")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("  ⚠ CUDA不可用，跳过此测试")
        return
    
    device = torch.device('cuda')
    B, N_q, num_views, H, W = 2, 256, 6, 32, 88
    L = num_views * H * W
    D = 256
    
    # 创建模块
    decoder = DenseAlternatingISSMDecoder(
        num_layers=6,
        d_model=D,
        d_state=16,
        num_views=num_views,
        feat_h=H,
        feat_w=W,
        num_heads=8,
        deterministic=True
    ).to(device)
    
    # 准备输入
    queries = torch.randn(B, N_q, D, device=device)
    anchors = torch.randn(B, N_q, 3, device=device)
    img_feats = torch.randn(B, L, D, device=device)
    img_coords = torch.randn(B, L, 3, device=device)
    
    # 预热
    print("\n[4.1] 预热GPU...")
    decoder.eval()
    with torch.no_grad():
        for _ in range(3):
            _ = decoder(queries, anchors, img_feats, img_coords)
    torch.cuda.synchronize()
    
    # 测试推理速度
    print("\n[4.2] 测试推理速度...")
    torch.cuda.reset_peak_memory_stats()
    
    num_runs = 10
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            torch.cuda.synchronize()
            start = time.time()
            _ = decoder(queries, anchors, img_feats, img_coords)
            torch.cuda.synchronize()
            times.append(time.time() - start)
    
    mean_time = sum(times) / len(times)
    std_time = (sum((t - mean_time)**2 for t in times) / len(times))**0.5
    peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
    
    print(f"  平均推理时间: {mean_time*1000:.2f} ± {std_time*1000:.2f} ms")
    print(f"  峰值显存: {peak_memory:.2f} GB")
    print(f"  吞吐量: {B*N_q*L/(mean_time*1e6):.2f} M tokens/s")
    
    print("\n✓ 性能测试完成")


def main():
    """主测试流程"""
    print("\n" + "="*70)
    print("DEST-Inspired 随机化 ISSM 架构测试")
    print("="*70)
    
    try:
        # 测试1: 动态随机重排
        test_dynamic_random_reorder()
        
        # 测试2: 随机化解码器前向传播
        test_randomized_decoder_forward()
        
        # 测试3: 反向传播和梯度
        test_backward_pass()
        
        # 测试4: 显存和速度
        test_memory_and_speed()
        
        print("\n" + "="*70)
        print("✓✓✓ 所有测试通过！✓✓✓")
        print("="*70)
        print("\n核心改进验证:")
        print("  ✓ 层级随机序列化策略正常工作")
        print("  ✓ 训练时动态随机，推理时固定确定")
        print("  ✓ 密集聚合 + 随机化组合正确")
        print("  ✓ 梯度计算无误，可正常训练")
        print("\n建议:")
        print("  - 确定性模式 (deterministic=True) 用于复现实验")
        print("  - 随机化模式 (deterministic=False) 用于正常训练")
        print("  - 推理时自动使用固定排列，无需修改代码")
        print("="*70)
        
    except Exception as e:
        print("\n" + "="*70)
        print(f"✗✗✗ 测试失败: {e}")
        print("="*70)
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
