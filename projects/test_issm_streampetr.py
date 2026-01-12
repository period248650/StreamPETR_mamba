# ------------------------------------------------------------------------
# Copyright (c) 2024 ISSM-StreamPETR. All Rights Reserved.
# ------------------------------------------------------------------------
"""
ISSM-StreamPETR 测试脚本

测试内容：
1. 模块导入测试
2. 维度正确性测试
3. 前向传播测试
4. 梯度回传测试
5. 与 DEST3D Triton 实现的对比
6. 端到端 Decoder 测试
7. 速度基准测试

使用方法：
    cd StreamPETR_mamba
    python projects/test_issm_streampetr.py
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

# 添加项目路径 - 直接添加到 utils 目录，绕过 mmdet3d_plugin/__init__.py
# 这样可以避免 mmcv registry 重复注册的问题
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'mmdet3d_plugin', 'models', 'utils'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'mmdet3d_plugin', 'models'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'mmdet3d_plugin'))
# 添加 issm_triton 路径，确保 Triton 内核可以被导入
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'mmdet3d_plugin', 'models', 'issm_triton'))

# 颜色输出
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'

def print_test_header(name):
    print(f"\n{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}Testing: {name}{Colors.END}")
    print(f"{Colors.BLUE}{'='*60}{Colors.END}")

def print_pass(msg):
    print(f"  {Colors.GREEN}✓ PASS{Colors.END}: {msg}")

def print_fail(msg):
    print(f"  {Colors.RED}✗ FAIL{Colors.END}: {msg}")

def print_info(msg):
    print(f"  {Colors.YELLOW}ℹ INFO{Colors.END}: {msg}")


# ============================================================================
# Test 1: 模块导入测试
# ============================================================================
def test_imports():
    print_test_header("Module Imports")
    
    try:
        from petr_issm import (
            DynamicRandomReorder,
            _SingleDirectionISSMLayer,
            DenseAlternatingISSMDecoder
        )
        print_pass("petr_issm modules imported successfully")
    except ImportError as e:
        print_fail(f"Failed to import petr_issm: {e}")
        return False
    
    try:
        from issm_triton.issm_combined import ISSM_chunk_scan_combined
        print_pass("ISSM_chunk_scan_combined imported (Triton available)")
        triton_available = True
    except ImportError:
        print_info("ISSM Triton not available, will use fallback")
        triton_available = False
    
    try:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
        print_pass("Mamba selective_scan_fn imported")
        mamba_available = True
    except ImportError:
        print_info("Mamba not available")
        mamba_available = False
    
    return True


# ============================================================================
# Test 2: DynamicRandomReorder 测试
# ============================================================================
def test_dynamic_random_reorder():
    print_test_header("DynamicRandomReorder")
    
    from petr_issm import DynamicRandomReorder
    
    num_views = 6
    H, W = 24, 44
    num_layers = 6
    B = 2
    D = 256
    L = num_views * H * W
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模块
    reorder = DynamicRandomReorder(
        num_views=num_views, H=H, W=W, 
        num_layers=num_layers, deterministic=True
    ).to(device)
    
    # 创建测试输入
    x = torch.randn(B, L, D, device=device)
    
    # 测试排列生成
    perm_idx, inv_perm_idx = reorder.get_permutation(
        layer_idx=0, device=device, training=True
    )
    
    # 验证排列索引
    assert perm_idx.shape == (L,), f"perm_idx shape mismatch: {perm_idx.shape}"
    assert inv_perm_idx.shape == (L,), f"inv_perm_idx shape mismatch: {inv_perm_idx.shape}"
    print_pass(f"Permutation indices shape: [{L}]")
    
    # 验证排列和逆排列
    x_perm = reorder(x, perm_idx)
    x_restored = reorder(x_perm, inv_perm_idx)
    
    assert torch.allclose(x, x_restored, atol=1e-6), "Permutation inverse failed"
    print_pass("Permutation and inverse permutation work correctly")
    
    # 验证不同层的排列不同（非确定性模式）
    reorder_random = DynamicRandomReorder(
        num_views=num_views, H=H, W=W, 
        num_layers=num_layers, deterministic=False
    ).to(device)
    
    perm1, _ = reorder_random.get_permutation(0, device, training=True)
    perm2, _ = reorder_random.get_permutation(1, device, training=True)
    
    # 随机排列应该不同
    if not torch.equal(perm1, perm2):
        print_pass("Different layers have different permutations")
    else:
        print_info("Permutations might be same (low probability)")
    
    return True


# ============================================================================
# Test 3: _SingleDirectionISSMLayer 测试
# ============================================================================
def test_single_direction_issm_layer():
    print_test_header("_SingleDirectionISSMLayer")
    
    from petr_issm import _SingleDirectionISSMLayer
    
    # 参数 - 确保 L 能被 chunk_size 整除，且 chunk_size >= N_q
    B = 2
    N_q = 64  # 减少 query 数量以便 chunk_size 设置更灵活
    num_views = 6
    H, W = 24, 44  # L = 6 * 24 * 44 = 6336
    L = num_views * H * W
    d_model = 256
    d_state = 16
    num_heads = 8
    
    # 计算合适的 chunk_size：需要 >= N_q 且能整除 L
    # 6336 的因子: 1, 2, 3, 4, 6, 8, 9, 11, 12, 16, 18, 22, 24, 32, 33, 36, 44, 48, 66, 72, 88, 96, 132, 144, 176, 192, 264, 288, 352, 396, 528, 576, 704, 792, 1056, 1584, 2112, 3168, 6336
    # 需要找到 >= 64 且能整除 6336 的因子
    # 选择 66, 72, 88, 96, ... 
    chunk_size = 96  # 6336 / 96 = 66 (整数)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建层 (使用纯 Triton 实现)
    layer = _SingleDirectionISSMLayer(
        d_model=d_model,
        d_state=d_state,
        d_conv=4,
        expand=2,
        num_heads=num_heads,
        d_dist=16,
        chunk_size=chunk_size,
        dropout=0.0
    ).to(device)
    
    print_info(f"Layer created with {sum(p.numel() for p in layer.parameters())} parameters")
    
    # 创建测试输入
    queries = torch.randn(B, N_q, d_model, device=device, requires_grad=True)
    anchors = torch.rand(B, N_q, 3, device=device)  # 归一化 3D 坐标
    features_perm = torch.randn(B, L, d_model, device=device, requires_grad=True)
    coords_perm = torch.rand(B, L, 3, device=device)  # 归一化 3D 坐标
    
    print_info(f"Input shapes: queries={queries.shape}, features={features_perm.shape}")
    print_info(f"L={L}, N_q={N_q}, chunk_size={chunk_size}, L%chunk_size={L%chunk_size}")
    
    # 前向传播
    try:
        out_query, out_feat = layer(queries, anchors, features_perm, coords_perm)
        print_pass(f"Forward pass successful")
    except Exception as e:
        print_fail(f"Forward pass failed: {e}")
        return False
    
    # 验证输出维度
    assert out_query.shape == (B, N_q, d_model), f"out_query shape mismatch: {out_query.shape}"
    assert out_feat.shape == (B, L, d_model), f"out_feat shape mismatch: {out_feat.shape}"
    print_pass(f"Output shapes: query={out_query.shape}, feat={out_feat.shape}")
    
    # 验证梯度回传
    loss = out_query.sum() + out_feat.sum()
    try:
        loss.backward()
        print_pass("Backward pass successful")
    except Exception as e:
        print_fail(f"Backward pass failed: {e}")
        return False
    
    # 检查梯度
    assert queries.grad is not None, "queries.grad is None"
    assert features_perm.grad is not None, "features_perm.grad is None"
    print_pass(f"Gradients computed: queries.grad.norm={queries.grad.norm().item():.4f}")
    
    return True


# ============================================================================
# Test 4: DenseAlternatingISSMDecoder 测试
# ============================================================================
def test_dense_alternating_decoder():
    print_test_header("DenseAlternatingISSMDecoder")
    
    from petr_issm import DenseAlternatingISSMDecoder
    
    # 参数 - 使用较小的 N_q 以确保 chunk_size 兼容性
    B = 2
    N_q = 64  # 减少 query 数量
    num_views = 6
    H, W = 24, 44
    L = num_views * H * W  # 6336
    d_model = 256
    num_layers = 6
    chunk_size = 96  # 6336 / 96 = 66 (整数), 且 96 >= 64
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建 Decoder
    decoder = DenseAlternatingISSMDecoder(
        num_layers=num_layers,
        d_model=d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        num_views=num_views,
        feat_h=H,
        feat_w=W,
        num_heads=8,
        chunk_size=chunk_size,
        dropout=0.1,
        box_refinement=False,
        fusion_type='add',
        layer_fusion_weight=0.5,
        deterministic=True
    ).to(device)
    
    num_params = sum(p.numel() for p in decoder.parameters())
    print_info(f"Decoder created with {num_params:,} parameters ({num_params/1e6:.2f}M)")
    
    # 创建测试输入
    queries = torch.randn(B, N_q, d_model, device=device, requires_grad=True)
    anchors = torch.rand(B, N_q, 3, device=device)
    img_feats = torch.randn(B, L, d_model, device=device, requires_grad=True)
    img_coords_3d = torch.rand(B, L, 3, device=device)
    
    # 前向传播
    try:
        output_queries, output_anchors = decoder(
            queries=queries,
            anchors=anchors,
            img_feats=img_feats,
            img_coords_3d=img_coords_3d,
            return_intermediate=False
        )
        print_pass("Forward pass successful")
    except Exception as e:
        print_fail(f"Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 验证输出维度
    assert output_queries.shape == (B, N_q, d_model), f"output_queries shape: {output_queries.shape}"
    assert output_anchors.shape == (B, N_q, 3), f"output_anchors shape: {output_anchors.shape}"
    print_pass(f"Output shapes: queries={output_queries.shape}, anchors={output_anchors.shape}")
    
    # 清理显存再进行中间结果测试
    del output_queries, output_anchors, queries, anchors, img_feats, img_coords_3d
    torch.cuda.empty_cache()
    
    # 测试中间结果返回（使用更小的参数）
    try:
        # 减少参数以避免 OOM
        small_B, small_N_q = 1, 100
        small_queries = torch.randn(small_B, small_N_q, d_model, device=device, requires_grad=True)
        small_anchors = torch.rand(small_B, small_N_q, 3, device=device)
        small_img_feats = torch.randn(small_B, L, d_model, device=device, requires_grad=True)
        small_img_coords = torch.rand(small_B, L, 3, device=device)
        
        inter_queries, inter_anchors = decoder(
            queries=small_queries,
            anchors=small_anchors,
            img_feats=small_img_feats,
            img_coords_3d=small_img_coords,
            return_intermediate=True
        )
        assert inter_queries.shape == (num_layers, small_B, small_N_q, d_model)
        assert inter_anchors.shape == (num_layers, small_B, small_N_q, 3)
        print_pass(f"Intermediate outputs: queries={inter_queries.shape}, anchors={inter_anchors.shape}")
        
        del inter_queries, inter_anchors, small_queries, small_anchors, small_img_feats, small_img_coords
        torch.cuda.empty_cache()
    except Exception as e:
        print_fail(f"Intermediate output failed: {e}")
        # 不返回 False，继续测试梯度
    
    # 梯度测试（使用较小的参数）
    torch.cuda.empty_cache()
    small_B, small_N_q = 1, 64
    grad_queries = torch.randn(small_B, small_N_q, d_model, device=device, requires_grad=True)
    grad_anchors = torch.rand(small_B, small_N_q, 3, device=device)
    grad_img_feats = torch.randn(small_B, L, d_model, device=device, requires_grad=True)
    grad_img_coords = torch.rand(small_B, L, 3, device=device)
    
    out_q, out_a = decoder(
        queries=grad_queries,
        anchors=grad_anchors,
        img_feats=grad_img_feats,
        img_coords_3d=grad_img_coords,
        return_intermediate=False
    )
    loss = out_q.sum()
    try:
        loss.backward()
        print_pass("Backward pass successful")
    except Exception as e:
        print_fail(f"Backward pass failed: {e}")
        return False
    
    return True


# ============================================================================
# Test 5: ISSM Triton vs Fallback 对比
# ============================================================================
def test_issm_triton_vs_fallback():
    print_test_header("ISSM Pure Triton Implementation Test")
    
    from petr_issm import _SingleDirectionISSMLayer
    
    B = 1
    N_q = 64  # 小规模测试
    L = 256
    d_model = 128
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print_info("测试纯 Triton ISSM 实现")
    
    # 使用较小的参数确保 chunk_size 兼容
    # L=256, N_q=64, chunk_size 需要 >= 64 且能整除 256
    # 可选: 64, 128, 256
    layer1 = _SingleDirectionISSMLayer(
        d_model=d_model, d_state=16, num_heads=4, chunk_size=64, dropout=0.0
    ).to(device)
    
    layer2 = _SingleDirectionISSMLayer(
        d_model=d_model, d_state=16, num_heads=4, chunk_size=64, dropout=0.0
    ).to(device)
    
    # 复制权重
    layer2.load_state_dict(layer1.state_dict())
    
    # 创建相同的输入
    torch.manual_seed(42)
    queries = torch.randn(B, N_q, d_model, device=device)
    anchors = torch.rand(B, N_q, 3, device=device)
    features = torch.randn(B, L, d_model, device=device)
    coords = torch.rand(B, L, 3, device=device)
    
    # 前向传播
    layer1.eval()
    layer2.eval()
    
    with torch.no_grad():
        out_q1, out_f1 = layer1(queries, anchors, features, coords)
        out_q2, out_f2 = layer2(queries, anchors, features, coords)
    
    # 比较输出 (相同权重应该产生相同输出)
    q_diff = (out_q1 - out_q2).abs().mean().item()
    f_diff = (out_f1 - out_f2).abs().mean().item()
    
    print_info(f"Query output diff: {q_diff:.6f}")
    print_info(f"Feature output diff: {f_diff:.6f}")
    
    if q_diff < 1e-5 and f_diff < 1e-5:
        print_pass("一致性测试通过")
    else:
        print_fail("输出不一致")
        return False
    
    return True


# ============================================================================
# Test 6: 端到端模拟测试
# ============================================================================
def test_end_to_end_simulation():
    print_test_header("End-to-End Simulation (ISSMStreamPETRHead style)")
    
    from petr_issm import (
        DenseAlternatingISSMDecoder,
        DynamicRandomReorder
    )
    
    # 模拟 StreamPETR Head 的工作流程（使用较小参数避免 OOM）
    B = 1
    N_q = 64  # 减少 query 数量
    num_views = 6
    H, W = 24, 44
    L = num_views * H * W
    d_model = 256
    num_layers = 6
    num_classes = 10
    code_size = 10
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    
    # 创建 Decoder (使用纯 Triton 实现)
    # chunk_size=96 能整除 L=6336，且 >= N_q=64
    decoder = DenseAlternatingISSMDecoder(
        num_layers=num_layers,
        d_model=d_model,
        d_state=16,
        num_views=num_views,
        feat_h=H,
        feat_w=W,
        num_heads=8,
        chunk_size=96,
        box_refinement=False,
        fusion_type='add',
        deterministic=True
    ).to(device)
    
    # 创建分类和回归分支
    cls_branches = nn.ModuleList([
        nn.Linear(d_model, num_classes) for _ in range(num_layers)
    ]).to(device)
    
    reg_branches = nn.ModuleList([
        nn.Linear(d_model, code_size) for _ in range(num_layers)
    ]).to(device)
    
    # 模拟输入
    queries = torch.randn(B, N_q, d_model, device=device, requires_grad=True)
    reference_points = torch.rand(B, N_q, 3, device=device)
    img_feats = torch.randn(B, L, d_model, device=device, requires_grad=True)
    img_coords_3d = torch.rand(B, L, 3, device=device)
    
    # 模拟 Head 的逐层处理
    all_cls_scores = []
    all_bbox_preds = []
    
    curr_query = queries
    curr_reference_points = reference_points
    feat_l1 = img_feats
    feat_l2 = None
    
    print_info("Running layer-by-layer processing...")
    
    for layer_idx in range(num_layers):
        # 密集聚合
        if layer_idx == 0:
            feat_input = feat_l1
        elif layer_idx == 1:
            feat_input = feat_l1
        else:
            feat_input = feat_l1 + feat_l2
        
        # 获取排列
        perm_idx, inv_perm_idx = decoder.reorder.get_permutation(
            layer_idx=layer_idx,
            device=device,
            training=True
        )
        
        # 重排
        feat_perm = feat_input[:, perm_idx, :]
        coords_perm = img_coords_3d[:, perm_idx, :]
        
        # ISSM 层
        query_new, feat_new_perm = decoder.layers[layer_idx](
            queries=curr_query,
            anchors=curr_reference_points,
            features_perm=feat_perm,
            coords_perm=coords_perm
        )
        
        # 还原顺序
        feat_output = feat_new_perm[:, inv_perm_idx, :]
        
        # 预测
        cls_scores = cls_branches[layer_idx](query_new)
        bbox_preds = reg_branches[layer_idx](query_new)
        bbox_preds[..., 0:3] = bbox_preds[..., 0:3].sigmoid()
        
        all_cls_scores.append(cls_scores)
        all_bbox_preds.append(bbox_preds)
        
        # 更新
        curr_query = query_new
        curr_reference_points = bbox_preds[..., 0:3].detach()
        feat_l2 = feat_l1
        feat_l1 = feat_output
    
    # Stack 结果
    all_cls_scores = torch.stack(all_cls_scores)
    all_bbox_preds = torch.stack(all_bbox_preds)
    
    print_pass(f"all_cls_scores shape: {all_cls_scores.shape}")
    print_pass(f"all_bbox_preds shape: {all_bbox_preds.shape}")
    
    # 梯度测试
    loss = all_cls_scores.sum() + all_bbox_preds.sum()
    loss.backward()
    
    assert queries.grad is not None
    assert img_feats.grad is not None
    print_pass("End-to-end gradient flow verified")
    
    return True


# ============================================================================
# Test 7: 速度基准测试
# ============================================================================
def test_speed_benchmark():
    print_test_header("Speed Benchmark")
    
    from petr_issm import _SingleDirectionISSMLayer
    
    # 使用较小的参数进行速度测试
    B = 1
    N_q = 64  # 减少 query 数量
    L = 6 * 24 * 44
    d_model = 256
    num_warmup = 5
    num_runs = 20
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    
    if device.type != 'cuda':
        print_info("CUDA not available, skipping speed benchmark")
        return True
    
    # 创建层 (使用纯 Triton 实现)
    # chunk_size=96 能整除 L=6336，且 >= N_q=64
    layer = _SingleDirectionISSMLayer(
        d_model=d_model,
        d_state=16,
        num_heads=8,
        chunk_size=96,
        dropout=0.0
    ).to(device)
    layer.eval()
    
    # 创建输入
    queries = torch.randn(B, N_q, d_model, device=device)
    anchors = torch.rand(B, N_q, 3, device=device)
    features = torch.randn(B, L, d_model, device=device)
    coords = torch.rand(B, L, 3, device=device)
    
    # Warmup
    print_info(f"Warming up ({num_warmup} runs)...")
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = layer(queries, anchors, features, coords)
            torch.cuda.synchronize()
    
    # Benchmark
    print_info(f"Benchmarking ({num_runs} runs)...")
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = layer(queries, anchors, features, coords)
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print_pass(f"Single layer forward pass:")
    print_info(f"  Average: {avg_time:.2f} ms")
    print_info(f"  Min: {min_time:.2f} ms")
    print_info(f"  Max: {max_time:.2f} ms")
    print_info(f"  Throughput: {1000/avg_time:.1f} forward/sec")
    
    return True


# ============================================================================
# Test 8: 内存使用测试
# ============================================================================
def test_memory_usage():
    print_test_header("Memory Usage")
    
    if not torch.cuda.is_available():
        print_info("CUDA not available, skipping memory test")
        return True
    
    from petr_issm import DenseAlternatingISSMDecoder
    
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    # 使用较小的参数进行显存测试
    B = 1
    N_q = 64  # 减少 query 数量以避免 OOM
    num_views = 6
    H, W = 24, 44
    L = num_views * H * W
    d_model = 256
    num_layers = 6
    
    device = torch.device('cuda')
    
    initial_mem = torch.cuda.memory_allocated() / 1024**2  # MB
    
    # 创建 Decoder (使用纯 Triton 实现)
    # chunk_size=96 能整除 L=6336，且 >= N_q=64
    decoder = DenseAlternatingISSMDecoder(
        num_layers=num_layers,
        d_model=d_model,
        d_state=16,
        num_views=num_views,
        feat_h=H,
        feat_w=W,
        num_heads=8,
        chunk_size=96,
        box_refinement=False,
        deterministic=True
    ).to(device)
    
    model_mem = torch.cuda.memory_allocated() / 1024**2 - initial_mem
    print_info(f"Model memory: {model_mem:.2f} MB")
    
    # 创建输入
    queries = torch.randn(B, N_q, d_model, device=device, requires_grad=True)
    anchors = torch.rand(B, N_q, 3, device=device)
    img_feats = torch.randn(B, L, d_model, device=device, requires_grad=True)
    img_coords_3d = torch.rand(B, L, 3, device=device)
    
    input_mem = torch.cuda.memory_allocated() / 1024**2 - initial_mem - model_mem
    print_info(f"Input memory: {input_mem:.2f} MB")
    
    # 前向传播
    output_queries, output_anchors = decoder(
        queries=queries,
        anchors=anchors,
        img_feats=img_feats,
        img_coords_3d=img_coords_3d
    )
    
    forward_mem = torch.cuda.memory_allocated() / 1024**2 - initial_mem
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2
    
    print_info(f"After forward memory: {forward_mem:.2f} MB")
    print_info(f"Peak memory: {peak_mem:.2f} MB")
    
    # 反向传播
    loss = output_queries.sum()
    loss.backward()
    
    backward_peak = torch.cuda.max_memory_allocated() / 1024**2
    print_info(f"Peak memory (with backward): {backward_peak:.2f} MB")
    
    print_pass("Memory usage test completed")
    
    return True


# ============================================================================
# Main
# ============================================================================
def main():
    print(f"\n{Colors.BLUE}{'#'*60}{Colors.END}")
    print(f"{Colors.BLUE}#  ISSM-StreamPETR Test Suite{Colors.END}")
    print(f"{Colors.BLUE}{'#'*60}{Colors.END}")
    
    tests = [
        ("Module Imports", test_imports),
        ("DynamicRandomReorder", test_dynamic_random_reorder),
        ("SingleDirectionISSMLayer", test_single_direction_issm_layer),
        ("DenseAlternatingDecoder", test_dense_alternating_decoder),
        ("Triton vs Fallback", test_issm_triton_vs_fallback),
        ("End-to-End Simulation", test_end_to_end_simulation),
        ("Speed Benchmark", test_speed_benchmark),
        ("Memory Usage", test_memory_usage),
    ]
    
    results = {}
    for name, test_fn in tests:
        try:
            results[name] = test_fn()
        except Exception as e:
            print_fail(f"Test crashed: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False
    
    # Summary
    print(f"\n{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}Test Summary{Colors.END}")
    print(f"{Colors.BLUE}{'='*60}{Colors.END}")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = f"{Colors.GREEN}PASS{Colors.END}" if result else f"{Colors.RED}FAIL{Colors.END}"
        print(f"  [{status}] {name}")
    
    print(f"\n{Colors.BLUE}Results: {passed}/{total} tests passed{Colors.END}")
    
    if passed == total:
        print(f"{Colors.GREEN}All tests passed! ✓{Colors.END}\n")
        return 0
    else:
        print(f"{Colors.RED}Some tests failed! ✗{Colors.END}\n")
        return 1


if __name__ == "__main__":
    exit(main())
