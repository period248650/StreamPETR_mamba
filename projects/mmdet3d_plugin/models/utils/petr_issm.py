# ------------------------------------------------------------------------
# Copyright (c) 2024 ISSM-StreamPETR. All Rights Reserved.
# ------------------------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from mmcv.cnn import xavier_init
from mmcv.runner.base_module import BaseModule


ISSM_TRITON_AVAILABLE = False
RMSNormGated = None
try:
    # 方式1: 相对导入（正常训练时使用）
    from ..issm_triton.issm_combined import ISSM_chunk_scan_combined
    from ..issm_triton.layernorm_gated import RMSNorm as RMSNormGated
    ISSM_TRITON_AVAILABLE = True
except ImportError:
    try:
        # 方式2: 绝对导入（独立测试时使用）
        from issm_triton.issm_combined import ISSM_chunk_scan_combined
        from issm_triton.layernorm_gated import RMSNorm as RMSNormGated
        ISSM_TRITON_AVAILABLE = True
    except ImportError:
        try:
            # 方式3: 直接从 models 目录导入
            from models.issm_triton.issm_combined import ISSM_chunk_scan_combined
            from models.issm_triton.layernorm_gated import RMSNorm as RMSNormGated
            ISSM_TRITON_AVAILABLE = True
        except ImportError:
            print("[WARNING] issm_triton not available. Please check the import path.")



# ============================================================================
# Part 1: Single Direction ISSM Layer (DEST3D-Faithful Implementation)
# ============================================================================

class _SingleDirectionISSMLayer(nn.Module):
    
    def __init__(
        self,
        d_model=256,
        d_conv=4,
        expand=2,
        num_heads=8,
        d_dist=16,  # 距离编码的中间维度
        chunk_size=256,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        A_init_range=(1, 16),
        conv_bias=True,
        bias=False,
        dropout=0.0,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        self._factory_kwargs = factory_kwargs
        self._bias = bias
        self._conv_bias = conv_bias
        self._dropout = dropout
        
        # 初始化基础参数
        self._init_basic_params(d_model, d_conv, expand, num_heads, d_dist, chunk_size)
        # 初始化投影层
        self._init_projections()
        # 初始化时间步参数
        self._init_dt_params(dt_min, dt_max, dt_init_floor)
        # 初始化状态转移参数
        self._init_state_params(A_init_range)
        # 初始化输出层
        self._init_output_layers()
        # 初始化距离权重参数
        self._init_dist_weight_params()
        
        self.register_buffer('_dummy_buffer', torch.zeros(1))
    
    def _compute_distance_features(self, anchors, key_pos_3d, B_batch, L, N_q, device, dtype):
        if key_pos_3d is not None and anchors is not None:
            # 计算 3D 相对位置: delta = key_pos - query_anchor
            # anchors: [B, N_q, 3] -> [B, 1, N_q, 3]
            # key_pos_3d: [B, L, 3] -> [B, L, 1, 3]
            query_pos_expanded = anchors.unsqueeze(1)      # [B, 1, N_q, 3]
            key_pos_expanded = key_pos_3d.unsqueeze(2)     # [B, L, 1, 3]
            
            # 相对位置
            delta = key_pos_expanded - query_pos_expanded  # [B, L, N_q, 3]
            
            # 【与 DEST3D 一致】对数缩放 + 归一化（防止距离过大导致梯度问题）
            # sign(delta) * log2(|delta| * scale + 1) / log2(max_val)
            log_scale = 512.0
            max_log_val = 8.0
            delta_encoded = torch.sign(delta) * torch.log2(torch.abs(delta) * log_scale + 1.0) / math.log2(max_log_val)
            delta_encoded = delta_encoded / 4.0  # 归一化到 [-1, 1] 范围
            
            # 通过 MLP 编码距离
            intermediate_feat = self.dist_mlp(delta_encoded)  # [B, L, N_q, d_dist]
        else:
            # 【防御性代码】正常调用时不会进入此分支
            intermediate_feat = torch.zeros(B_batch, L, N_q, self.d_dist, device=device, dtype=dtype)
        
        return intermediate_feat
    
    def _init_basic_params(self, d_model, d_conv, expand, num_heads, d_dist, chunk_size):
        """Initialize basic parameters - 初始化基础参数"""
        self.d_model = d_model
        self.d_conv = d_conv
        self.expand = expand
        self.num_heads = num_heads
        self.d_dist = d_dist
        self.chunk_size = chunk_size
        
        # 扩展内部维度
        self.d_inner = int(self.expand * self.d_model)
        self.headdim = self.d_inner // self.num_heads
        assert self.d_inner % self.num_heads == 0, "d_inner must be divisible by num_heads"
    
    def _init_projections(self):
        factory_kwargs = self._factory_kwargs
        
        # Key (特征+位置融合后) 投影：生成 z, x, bc_bias, dt_bias
        d_in_key_proj = self.d_inner * 2 + 2 + self.num_heads
        self.key_proj = nn.Linear(self.d_model, d_in_key_proj, bias=self._bias, **factory_kwargs)
        
        # 局部卷积 (类似 DEST3D 的 PointLiteConv)
        d_key_conv = self.d_inner + 2
        self.key_conv = nn.Conv1d(
            in_channels=d_key_conv,
            out_channels=d_key_conv,
            bias=self._conv_bias,
            kernel_size=self.d_conv,
            groups=d_key_conv,
            padding=self.d_conv - 1,
            **factory_kwargs,
        )
        
        # Query 投影：将 Query 映射到 initial_states
        self.query_proj = nn.Linear(self.d_model, self.d_inner, bias=False, **factory_kwargs)
        
        # 距离编码 MLP（3D 几何距离 -> 中间特征）
        dist_hidden = 64  # 隐藏层维度
        self.dist_mlp = nn.Sequential(
            nn.Linear(3, dist_hidden, bias=True, **factory_kwargs),
            nn.ReLU(inplace=True),
            nn.Linear(dist_hidden, self.d_dist, bias=False, **factory_kwargs)
        )
        
        # 从中间特征生成 B, C, dt
        # bc_proj: [d_dist] -> [2] (B 和 C 各一个标量)
        self.bc_proj = nn.Linear(self.d_dist, 2, bias=False, **factory_kwargs)
        # dt_proj: [d_dist] -> [num_heads]
        self.dt_proj = nn.Linear(self.d_dist, self.num_heads, bias=False, **factory_kwargs)
    
    def _init_dt_params(self, dt_min, dt_max, dt_init_floor):
        """Initialize time step parameters - 初始化时间步参数"""
        factory_kwargs = self._factory_kwargs
        
        dt = torch.exp(
            torch.rand(self.num_heads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True
    
    def _init_state_params(self, A_init_range):
        """Initialize state transition parameters - 初始化状态转移参数"""
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        
        # SSM 参数 A
        A = torch.empty(self.num_heads, dtype=torch.float32).uniform_(*A_init_range)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        
        # D "skip" 参数
        self.D = nn.Parameter(torch.ones(self.num_heads))
        self.D._no_weight_decay = True
    
    def _init_output_layers(self):
        factory_kwargs = self._factory_kwargs
        
        # Key 输出层（与 DEST3D 一致：使用 RMSNormGated）
        assert RMSNormGated is not None, "RMSNormGated is required but not available"
        self.key_norm = RMSNormGated(self.d_inner, eps=1e-5, norm_before_gate=False)
        self.out_key_proj = nn.Linear(self.d_inner, self.d_model, bias=self._bias, **factory_kwargs)
        
        # Query 输出层（从 last_states 恢复）
        self.query_norm = nn.LayerNorm(self.d_inner, **factory_kwargs)
        self.out_query_proj = nn.Linear(self.d_inner, self.d_model, bias=False, **factory_kwargs)
        
        # Query FFN
        self.query_ffn = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 4, **factory_kwargs),
            nn.GELU(),
            nn.Dropout(self._dropout),
            nn.Linear(self.d_model * 4, self.d_model, **factory_kwargs),
            nn.Dropout(self._dropout),
        )
        self.query_ffn_norm = nn.LayerNorm(self.d_model, **factory_kwargs)
        
        # Dropout 和激活函数
        self.dropout = nn.Dropout(self._dropout)
        self.act = nn.SiLU()
    
    def _init_dist_weight_params(self):
        """初始化距离编码相关参数"""
        # 初始化 dist_mlp（与 DEST3D 的 cpb_mlp 一致）
        for m in self.dist_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # 初始化 bc_proj 和 dt_proj
        nn.init.xavier_uniform_(self.bc_proj.weight)
        nn.init.xavier_uniform_(self.dt_proj.weight)
        
    def forward(self, queries, anchors, features_perm, pos_embed_perm, key_pos_3d=None, query_pos=None):
        """
        前向传播 - 使用 3D 几何距离生成 B、C、dt（DEST3D 风格）
        
        Args:
            queries: [B, N_q, d_model] Query 特征
            anchors: [B, N_q, 3] Query 的 3D 锚点位置
            features_perm: [B, L, d_model] 已重排的特征序列（Key）
            pos_embed_perm: [B, L, d_model] 已重排的位置编码（Key 的位置）
            key_pos_3d: [B, L, 3] Key 的 3D 位置
            query_pos: [B, N_q, d_model] Query 的位置编码
            
        Returns:
            q_new: [B, N_q, d_model] 更新后的 Query
            f_new: [B, L, d_model] 更新后的 Feature
        """
        B_batch, N_q, _ = queries.shape
        L = features_perm.shape[1]
        
        # 动态设置 d_state = N_q（Query 数量）
        d_state = N_q
        
        # 调试信息（首次调用时打印）
        if not hasattr(self, '_debug_printed'):
            self._debug_printed = True
            print(f"[ISSM] d_state={d_state}, N_q={N_q}, L={L}, d_dist={self.d_dist}")
        
        # ========================================
        # 隐式位置编码融合
        # ========================================
        if query_pos is None:
            query_pos = torch.zeros_like(queries)
        
        # 【隐式融合】特征 + 位置编码 直接相加（StreamPETR 风格）
        query_with_pos = queries + query_pos           # [B, N_q, d_model]
        key_with_pos = features_perm + pos_embed_perm  # [B, L, d_model]
        
        # === 1. Key (融合特征) 投影 ===
        key_proj_out = self.key_proj(key_with_pos)
        z, xbc, dt_bias = torch.split(
            key_proj_out, 
            [self.d_inner, self.d_inner + 2, self.num_heads], 
            dim=-1
        )
        
        # === 2. 局部卷积 ===
        xbc = rearrange(xbc, 'b l d -> b d l')
        xbc = self.key_conv(xbc)[..., :L]
        xbc = self.act(xbc)
        xbc = rearrange(xbc, 'b d l -> b l d')
        
        x, b_bias, c_bias = torch.split(xbc, [self.d_inner, 1, 1], dim=-1)
        b_bias = b_bias.squeeze(-1)  # [B, L]
        c_bias = c_bias.squeeze(-1)  # [B, L]
        
        # === 3. Query (融合特征) 作为 initial_states ===
        initial_states = self.query_proj(query_with_pos)  # [B, N_q, d_inner]
        initial_states = rearrange(initial_states, "b n (h hd) -> b h hd n", h=self.num_heads)
        # initial_states: [B, nheads, headdim, N_q=d_state]
        
        # === 4. 【核心】使用 3D 距离生成 B, C, dt ===
        intermediate_feat = self._compute_distance_features(
            anchors, key_pos_3d, B_batch, L, N_q, queries.device, queries.dtype
        )
        
        # 从中间特征生成 B, C 参数（每个 Query-Key 对有独立的值）
        bc = self.bc_proj(intermediate_feat)  # [B, L, N_q, 2]
        b_base, c_base = torch.split(bc, [1, 1], dim=-1)  # [B, L, N_q, 1]
        b_base = b_base.squeeze(-1)  # [B, L, N_q]
        c_base = c_base.squeeze(-1)  # [B, L, N_q]
        
        # B, C: 加上 bias
        B_ssm = (b_base + b_bias.unsqueeze(-1)).unsqueeze(2)  # [B, L, 1, N_q]
        C_ssm = (c_base + c_bias.unsqueeze(-1)).unsqueeze(2)  # [B, L, 1, N_q]
        
        # dt 参数：从中间特征生成（每个 Query-Key 对有独立的值）
        dt_base = self.dt_proj(intermediate_feat)  # [B, L, N_q, num_heads]
        dt_base = dt_base.permute(0, 1, 3, 2)  # [B, L, num_heads, N_q]
        
        # dt = softplus(dt_base + dt_bias + dt_bias_param)
        dt = F.softplus(
            dt_base + 
            dt_bias.unsqueeze(-1) + 
            self.dt_bias.view(1, 1, -1, 1)
        )  # [B, L, nheads, N_q=d_state]
        
        # === 5. A 参数 ===
        A = -torch.exp(self.A_log.float())
        A = repeat(A, "h -> h d", d=d_state)  # [nheads, d_state]
        
        # === 6. 运行 SSM ===
        x_shaped = rearrange(x, "b l (h hd) -> b l h hd", h=self.num_heads)
        
        y, last_states = self._run_issm_triton(
            x_shaped, dt, A, B_ssm, C_ssm, initial_states, d_state
        )
        
        # === 7. Key 输出处理（与 DEST3D 一致）===
        y = rearrange(y, "b l h hd -> b l (h hd)")
        D_expanded = self.D.unsqueeze(-1).expand(-1, self.headdim).reshape(-1)
        y = y + x * D_expanded.unsqueeze(0).unsqueeze(0)
        # 【与 DEST3D 一致】使用 RMSNormGated：norm(y, z) = norm(y) * silu(z)
        y = self.key_norm(y, z)
        out_feat = self.out_key_proj(y)
        out_feat = features_perm + self.dropout(out_feat)  # 残差连接
        
        # === 8. 【核心】Query 从 last_states 更新 ===
        # last_states: [B, nheads, headdim, N_q]
        last_states_reshaped = rearrange(last_states, "b h hd n -> b n (h hd)")  # [B, N_q, d_inner]
        last_states_normed = self.query_norm(last_states_reshaped)
        query_update = self.out_query_proj(last_states_normed)  # [B, N_q, d_model]
        
        # Query 残差连接
        out_query = queries + self.dropout(query_update)
        
        # === 9. Query FFN ===
        query_ffn_normed = self.query_ffn_norm(out_query)
        out_query = out_query + self.query_ffn(query_ffn_normed)
        
        # === 10. DDP 兼容性：确保所有参数有梯度 ===
        ssm_param_sum = self.A_log.sum() + self.D.sum() + self.dt_bias.sum()
        param_reg = ssm_param_sum * 1e-8
        out_feat = out_feat + param_reg
        out_query = out_query + param_reg
        
        return out_query, out_feat
    
    def _run_issm_triton(self, x, dt, A, B, C, initial_states, d_state):
        """运行层内双向 ISSM 扫描（DEST3D 风格）
        
        双向扫描确保 Query 能同时看到序列两端的信息：
        - 正向扫描：Query 获取序列末尾附近的强信号
        - 反向扫描：Query 获取序列开头附近的强信号
        - 取平均：Query 获得全局视野
        """
        if not ISSM_TRITON_AVAILABLE:
            raise RuntimeError(
                "ISSM Triton implementation not available. "
                "Please ensure triton is installed: pip install triton>=2.1.0"
            )
        
        B_batch, L, nheads, headdim = x.shape

        if not hasattr(self, '_triton_compile_printed'):
            self._triton_compile_printed = True
            print(f"[ISSM Debug] Calling Triton ISSM kernel with BIDIRECTIONAL scan...")
            print(f"[ISSM Debug] x: {x.shape}, dt: {dt.shape}, B: {B.shape}, C: {C.shape}")
            print(f"[ISSM Debug] A: {A.shape}, initial_states: {initial_states.shape}")
            print(f"[ISSM Debug] chunk_size: {self.chunk_size}, d_state: {d_state}")
        
        # === 正向扫描 ===
        y_fwd, last_states_fwd = ISSM_chunk_scan_combined(
            x, dt, A, B, C,
            chunk_size=self.chunk_size,
            D=self.D,
            z=None,
            initial_states=initial_states,
            dt_softplus=False,
            return_final_states=True
        )
        
        # === 反向扫描 ===
        # 翻转所有与序列位置相关的张量
        x_back = torch.flip(x, dims=[1])
        dt_back = torch.flip(dt, dims=[1])
        B_back = torch.flip(B, dims=[1])
        C_back = torch.flip(C, dims=[1])
        
        y_back, last_states_back = ISSM_chunk_scan_combined(
            x_back, dt_back, A, B_back, C_back,
            chunk_size=self.chunk_size,
            D=self.D,
            z=None,
            initial_states=initial_states,
            dt_softplus=False,
            return_final_states=True
        )
        
        # 反向输出翻转回原顺序
        y_back = torch.flip(y_back, dims=[1])
        
        # === 双向融合：取平均（与 DEST3D 一致）===
        y = (y_fwd + y_back) / 2
        last_states = (last_states_fwd + last_states_back) / 2
        
        return y, last_states
    
# ============================================================================
# Part 2: ISSM Decoder Layer Container
# ============================================================================

class DenseAlternatingISSMDecoder(BaseModule):
    """
    ISSM 解码器层容器
    
    说明：
    - 这是一个"层容器"，主要用于管理 ISSM 层和相关配置
    - forward() 循环由 issm_streampetr_head.py 控制，以实现更灵活的逻辑
    - 包括：确定性视图重排、隐式双向扫描、加权密集聚合等
    - B、C、dt 使用 3D 几何距离生成（DEST3D 风格）
    
    Args:
        num_layers (int): 解码器层数
        d_model (int): 特征维度
        d_conv (int): 因果卷积核大小
        expand (int): 内部维度扩展因子
        num_heads (int): 并行头数
        chunk_size (int): Triton 块大小
        dropout (float): Dropout 比例
        layer_fusion_weight (float): 密集聚合权重，F_L = w * F_{L-1} + (1-w) * F_{L-2}
        d_dist (int): 距离特征的中间维度
    """
    
    def __init__(
        self,
        num_layers=6,
        d_model=256,
        d_conv=4,
        expand=2,
        num_heads=8,
        chunk_size=256,
        dropout=0.1,
        layer_fusion_weight=0.8,  # 密集聚合权重：当前层 vs 前一层
        d_dist=16,                # 距离特征的中间维度
        device=None,
        dtype=None,
        init_cfg=None,
        **kwargs  # 兼容旧配置文件中的多余参数
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(init_cfg=init_cfg)
        
        assert num_layers >= 2, "至少需要2层才能实现密集连接"
        assert 0.0 <= layer_fusion_weight <= 1.0, "layer_fusion_weight 必须在 [0, 1] 范围内"
        
        # 检查 Triton ISSM 是否可用
        if not ISSM_TRITON_AVAILABLE:
            raise RuntimeError(
                "Pure Triton ISSM implementation is required but not available. "
                "Please ensure triton is installed: pip install triton>=2.1.0"
            )
        
        # 保存配置（供 head 使用）
        self.num_layers = num_layers
        self.d_model = d_model
        self.layer_fusion_weight = layer_fusion_weight
        self.d_dist = d_dist
        
        # 打印配置信息
        print(f"[DenseAlternatingISSMDecoder] num_layers={num_layers}, "
              f"layer_fusion_weight={layer_fusion_weight}")
        
        # === ISSM 层堆叠 ===
        self.layers = nn.ModuleList([
            _SingleDirectionISSMLayer(
                d_model=d_model,
                d_conv=d_conv,
                expand=expand,
                num_heads=num_heads,
                d_dist=d_dist,
                chunk_size=chunk_size,
                dropout=dropout,
                **factory_kwargs
            )
            for _ in range(num_layers)
        ])
        
    def init_weights(self):
        """初始化权重"""
        for layer in self.layers:
            if hasattr(layer, '_init_dist_weight_params'):
                layer._init_dist_weight_params()