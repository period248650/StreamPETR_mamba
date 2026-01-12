# ------------------------------------------------------------------------
# Test script for Pure Triton ISSM Implementation
# 测试纯 Triton ISSM 实现
# ------------------------------------------------------------------------
"""
验证 StreamPETR_mamba 中的纯 Triton ISSM 实现

运行方式:
    cd /mnt/c/Users/17203/Desktop/Research/StreamPETR_mamba
    python projects/test_pure_triton_issm.py
"""

import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))

# Test 1: Import issm_triton module
print("\n" + "="*60)
print("Test 1: Import issm_triton module")
print("="*60)

try:
    from projects.mmdet3d_plugin.models.issm_triton import ISSM_chunk_scan_combined
    print("[PASS] Successfully imported ISSM_chunk_scan_combined from issm_triton")
except ImportError as e:
    print(f"[FAIL] Import error: {e}")
    print("This may be due to missing triton package")

try:
    from projects.mmdet3d_plugin.models.issm_triton.issm_combined import ISSMChunkScanCombinedFn
    print("[PASS] Successfully imported ISSMChunkScanCombinedFn")
except ImportError as e:
    print(f"[FAIL] Import error: {e}")

# Test 2: Import petr_issm module (without mamba_ssm dependency)
print("\n" + "="*60)
print("Test 2: Import petr_issm module (Pure Triton)")
print("="*60)

try:
    from projects.mmdet3d_plugin.models.utils.petr_issm import (
        DynamicRandomReorder,
        DenseAlternatingISSMDecoder,
        ISSM_TRITON_AVAILABLE
    )
    print("[PASS] Successfully imported petr_issm components")
    print(f"       ISSM_TRITON_AVAILABLE = {ISSM_TRITON_AVAILABLE}")
except ImportError as e:
    print(f"[FAIL] Import error: {e}")
    sys.exit(1)

# Test 3: Create DynamicRandomReorder module
print("\n" + "="*60)
print("Test 3: Create DynamicRandomReorder module")
print("="*60)

try:
    reorder = DynamicRandomReorder(
        num_views=6,
        H=32,
        W=88,
        num_layers=6,
        deterministic=True
    )
    print("[PASS] Created DynamicRandomReorder")
    
    # Test permutation generation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    perm, inv_perm = reorder.get_permutation(layer_idx=0, device=device, training=True)
    print(f"       Permutation shape: {perm.shape}")
    print(f"       Inverse permutation shape: {inv_perm.shape}")
    
    # Verify inverse permutation
    identity = perm[inv_perm]
    expected = torch.arange(len(perm), device=device)
    if torch.all(identity == expected):
        print("[PASS] Inverse permutation correctly restores order")
    else:
        print("[FAIL] Inverse permutation is incorrect")
except Exception as e:
    print(f"[FAIL] Error: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Create DenseAlternatingISSMDecoder (requires Triton)
print("\n" + "="*60)
print("Test 4: Create DenseAlternatingISSMDecoder")
print("="*60)

if not ISSM_TRITON_AVAILABLE:
    print("[SKIP] ISSM_TRITON_AVAILABLE is False, cannot create decoder")
else:
    try:
        decoder = DenseAlternatingISSMDecoder(
            num_layers=2,
            d_model=256,
            d_state=16,
            d_conv=4,
            expand=2,
            num_views=6,
            feat_h=32,
            feat_w=88,
            num_heads=8,
            chunk_size=256,
            dropout=0.1,
            box_refinement=True,
            fusion_type='add',
            deterministic=True
        )
        print("[PASS] Created DenseAlternatingISSMDecoder (Pure Triton)")
        
        # Count parameters
        num_params = sum(p.numel() for p in decoder.parameters())
        print(f"       Total parameters: {num_params:,}")
        
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()

# Test 5: Forward pass test (GPU required for Triton)
print("\n" + "="*60)
print("Test 5: Forward pass test")
print("="*60)

if torch.cuda.is_available():
    try:
        device = torch.device('cuda')
        decoder = decoder.to(device)
        
        # Create dummy inputs
        batch_size = 2
        num_queries = 100
        seq_len = 6 * 32 * 88  # num_views * H * W
        d_model = 256
        
        queries = torch.randn(batch_size, num_queries, d_model, device=device)
        anchors = torch.randn(batch_size, num_queries, 3, device=device)
        img_feats = torch.randn(batch_size, seq_len, d_model, device=device)
        img_coords_3d = torch.randn(batch_size, seq_len, 3, device=device)
        
        print(f"       Input queries shape: {queries.shape}")
        print(f"       Input img_feats shape: {img_feats.shape}")
        
        # Run forward pass
        decoder.eval()
        with torch.no_grad():
            output_queries, output_anchors = decoder(
                queries=queries,
                anchors=anchors,
                img_feats=img_feats,
                img_coords_3d=img_coords_3d,
                return_intermediate=False
            )
        
        print(f"       Output queries shape: {output_queries.shape}")
        print(f"       Output anchors shape: {output_anchors.shape}")
        print("[PASS] Forward pass completed successfully")
        
    except Exception as e:
        print(f"[FAIL] Forward pass error: {e}")
        import traceback
        traceback.print_exc()
else:
    print("[SKIP] GPU not available, skipping forward pass test")

# Test 6: Triton ISSM kernel direct test
print("\n" + "="*60)
print("Test 6: Direct Triton ISSM kernel test")
print("="*60)

if torch.cuda.is_available() and ISSM_TRITON_AVAILABLE:
    try:
        from projects.mmdet3d_plugin.models.issm_triton import ISSM_chunk_scan_combined
        
        device = torch.device('cuda')
        batch = 2
        seqlen = 256
        nheads = 8
        headdim = 64
        dstate = 16
        chunk_size = 64
        
        # Create inputs
        x = torch.randn(batch, seqlen, nheads, headdim, device=device, dtype=torch.float32)
        dt = torch.randn(batch, seqlen, nheads, dstate, device=device, dtype=torch.float32).abs() + 0.01
        A = -torch.rand(nheads, dstate, device=device, dtype=torch.float32).abs() - 0.1
        B = torch.randn(batch, seqlen, 1, dstate, device=device, dtype=torch.float32)
        C = torch.randn(batch, seqlen, 1, dstate, device=device, dtype=torch.float32)
        D = torch.randn(nheads, headdim, device=device, dtype=torch.float32)
        initial_states = torch.randn(batch, nheads, headdim, dstate, device=device, dtype=torch.float32)
        
        print(f"       x shape: {x.shape}")
        print(f"       dt shape: {dt.shape}")
        print(f"       A shape: {A.shape}")
        print(f"       B shape: {B.shape}")
        print(f"       C shape: {C.shape}")
        
        # Run ISSM
        out, final_states = ISSM_chunk_scan_combined(
            x, dt, A, B, C,
            chunk_size=chunk_size,
            D=D,
            initial_states=initial_states,
            return_final_states=True
        )
        
        print(f"       Output shape: {out.shape}")
        print(f"       Final states shape: {final_states.shape}")
        print("[PASS] Triton ISSM kernel executed successfully")
        
    except Exception as e:
        print(f"[FAIL] Triton ISSM error: {e}")
        import traceback
        traceback.print_exc()
else:
    if not torch.cuda.is_available():
        print("[SKIP] GPU not available")
    else:
        print("[SKIP] ISSM_TRITON_AVAILABLE is False")

# Summary
print("\n" + "="*60)
print("Test Summary")
print("="*60)
print("Pure Triton ISSM Implementation Tests Completed")
print(f"ISSM_TRITON_AVAILABLE: {ISSM_TRITON_AVAILABLE}")
print("No mamba_ssm dependency required!")
