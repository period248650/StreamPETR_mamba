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
# GatedFFN: DEST3D 风格的门控前馈网络
# ============================================================================

class GatedFFN(nn.Module):
    """DEST3D 风格的门控 FFN (类似 RGBlock，但使用 Linear 而非 Conv1d)"""
    def __init__(self, in_features, hidden_features=None, out_features=None, 
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        # fc1 输出 2 倍隐藏维度，用于门控分割
        self.fc1 = nn.Linear(in_features, hidden_features * 2)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        # 门控机制：分成两路，一路激活后与另一路相乘
        x, v = self.fc1(x).chunk(2, dim=-1)
        x = self.act(x) * v  # 门控
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


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
        d_dist=16,  
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
        
        self._init_basic_params(d_model, d_conv, expand, num_heads, d_dist, chunk_size)
        self._init_projections()
        self._init_dt_params(dt_min, dt_max, dt_init_floor)
        self._init_state_params(A_init_range)
        self._init_output_layers()
        self._init_dist_weight_params()
        
        self.register_buffer('_dummy_buffer', torch.zeros(1))
    
    def _compute_distance_features(self, anchors, key_pos_3d, B_batch, L, N_q, device, dtype):
        if key_pos_3d is not None and anchors is not None:
            query_pos_expanded = anchors.unsqueeze(1)      # [B, 1, N_q, 3]
            key_pos_expanded = key_pos_3d.unsqueeze(2)     # [B, L, 1, 3]
            
            delta = query_pos_expanded - key_pos_expanded  # [B, L, N_q, 3]
            

            log_scale = 20.0
            delta_encoded = torch.sign(delta) * torch.log2(torch.abs(delta) * log_scale + 1.0) / math.log2(8.0)
            delta_encoded = delta_encoded / 4.0
            
            intermediate_feat = self.dist_mlp(delta_encoded)  # [B, L, N_q, d_dist]
        
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
        

        self.bc_proj = nn.Linear(self.d_dist, 2, bias=False, **factory_kwargs)
        self.dt_proj = nn.Linear(self.d_dist, self.num_heads, bias=False, **factory_kwargs)
    
    def _init_dt_params(self, dt_min, dt_max, dt_init_floor):
        """Initialize time step parameters - 初始化时间步参数"""
        factory_kwargs = self._factory_kwargs
        
        dt = torch.exp(
            torch.rand(self.num_heads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True
    
    def _init_state_params(self, A_init_range):
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        
        A = torch.empty(self.num_heads, dtype=torch.float32).uniform_(*A_init_range)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        

        self.D = nn.Parameter(torch.ones(self.num_heads))
        self.D._no_weight_decay = True
    
    def _init_output_layers(self):
        factory_kwargs = self._factory_kwargs
        
        assert RMSNormGated is not None, "RMSNormGated is required but not available"
        self.key_norm = RMSNormGated(self.d_inner, eps=1e-5, norm_before_gate=False)
        self.out_key_proj = nn.Linear(self.d_inner, self.d_model, bias=self._bias, **factory_kwargs)
        
        self.query_norm = nn.LayerNorm(self.d_inner, **factory_kwargs)
        self.out_query_proj = nn.Linear(self.d_inner, self.d_model, bias=False, **factory_kwargs)
        
        # DEST3D 风格的门控 FFN
        self.query_ffn = GatedFFN(
            in_features=self.d_model,
            hidden_features=self.d_model * 4,
            out_features=self.d_model,
            act_layer=nn.GELU,
            drop=self._dropout
        )
        self.query_ffn_norm = nn.LayerNorm(self.d_model, **factory_kwargs)
        
        # Dropout 和激活函数
        self.dropout = nn.Dropout(self._dropout)
        self.act = nn.SiLU()
    
    def _init_dist_weight_params(self):
        """初始化距离编码相关参数"""
        for m in self.dist_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        nn.init.xavier_uniform_(self.bc_proj.weight)
        nn.init.xavier_uniform_(self.dt_proj.weight)
        
    def forward(self, queries, anchors, features_perm, pos_embed_perm, key_pos_3d=None, query_pos=None):
        """
        前向传播 - 使用 3D 几何距离生成 B、C、dt
        
        Args:
            queries: [B, N_q, d_model] Query 特征
            anchors: [B, N_q, 3] Query 的 3D 锚点位置
            features_perm: [B, L, d_model] 图像特征序列（Key）
            pos_embed_perm: [B, L, d_model] 位置编码
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
        
        if query_pos is None:
            query_pos = torch.zeros_like(queries)
        
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
        y = self.key_norm(y, z)
        out_feat = self.out_key_proj(y)
        out_feat = features_perm + self.dropout(out_feat)  # 残差连接
        
        # === 8. Query 从 last_states 更新 ===
        last_states_reshaped = rearrange(last_states, "b h hd n -> b n (h hd)")  # [B, N_q, d_inner]
        last_states_normed = self.query_norm(last_states_reshaped)
        query_update = self.out_query_proj(last_states_normed)  # [B, N_q, d_model]
        
        # Query 残差连接
        out_query = queries + self.dropout(query_update)
        
        # === 9. Query FFN ===
        query_ffn_normed = self.query_ffn_norm(out_query)
        out_query = out_query + self.query_ffn(query_ffn_normed)
        
        
        return out_query, out_feat
    
    def _run_issm_triton(self, x, dt, A, B, C, initial_states, d_state):
        
        B_batch, L, nheads, headdim = x.shape
        
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
    Args:
        num_layers (int): 解码器层数
        d_model (int): 特征维度
        d_conv (int): 因果卷积核大小
        expand (int): 内部维度扩展因子
        num_heads (int): 并行头数
        chunk_size (int): Triton 块大小
        dropout (float): Dropout 比例
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
        d_dist=16,                
        device=None,
        dtype=None,
        init_cfg=None,
        **kwargs  
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(init_cfg=init_cfg)
        
        # 保存配置（供 head 使用）
        self.num_layers = num_layers
        self.d_model = d_model
        self.d_dist = d_dist
        
        
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