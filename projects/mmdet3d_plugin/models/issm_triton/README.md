# ISSM Triton Implementation for StreamPETR

## 概述

本目录包含 ISSM (Interactive State Space Model) 的纯 Triton 实现，移植自 DEST3D (ICLR 2025)。

**重要变更**：本实现已完全移除对 `mamba_ssm` 包的依赖，使用纯 Triton kernel 实现所有 SSM 操作。

## 文件结构

```
issm_triton/
├── __init__.py              # 模块导出
├── issm_combined.py         # ISSM 主接口 (ISSM_chunk_scan_combined)
├── issm_chunk_scan.py       # Chunk scan 前向/反向 Triton kernels
├── issm_chunk_state.py      # Chunk state 前向/反向 Triton kernels
├── issm_state_passing.py    # State passing 前向/反向 Triton kernels
├── issm_cabt_bmm.py         # CABT BMM 操作 Triton kernels
├── layernorm_gated.py       # 门控 RMS Norm Triton kernel
└── softplus.py              # Softplus 激活 Triton kernel
```

## 主要 API

### `ISSM_chunk_scan_combined`

主要的 ISSM 扫描函数，支持自动微分。

```python
from projects.mmdet3d_plugin.models.issm_triton import ISSM_chunk_scan_combined

out, final_states = ISSM_chunk_scan_combined(
    x,                    # (batch, seqlen, nheads, headdim) - 输入特征
    dt,                   # (batch, seqlen, nheads, dstate) - 时间步长
    A,                    # (nheads, dstate) - 状态转移矩阵
    B,                    # (batch, seqlen, ngroups, dstate) - 输入投影
    C,                    # (batch, seqlen, ngroups, dstate) - 输出投影
    chunk_size=256,       # Triton chunk 大小
    D=None,               # (nheads, headdim) - Skip 连接
    z=None,               # (batch, seqlen, nheads, headdim) - 门控
    dt_bias=None,         # (nheads,) - dt 偏置
    initial_states=None,  # (batch, nheads, headdim, dstate) - 初始状态
    seq_idx=None,         # (batch, seqlen) - 序列索引
    dt_softplus=False,    # 是否对 dt 应用 softplus
    dt_limit=(0.0, inf),  # dt 裁剪范围
    return_final_states=True  # 是否返回最终状态
)
```

## 与 petr_issm.py 的集成

`petr_issm.py` 中的 `_SingleDirectionISSMLayer` 和 `DenseAlternatingISSMDecoder` 已更新为使用此纯 Triton 实现：

```python
# petr_issm.py 中的导入
from ..issm_triton.issm_combined import ISSM_chunk_scan_combined
ISSM_TRITON_AVAILABLE = True
```

## 依赖

- **PyTorch** >= 2.0
- **Triton** >= 2.1.0
- **einops**

**注意**：不再需要 `mamba_ssm` 包！

## 测试

运行测试脚本验证实现：

```bash
cd /path/to/StreamPETR_mamba
python projects/test_pure_triton_issm.py
```

## 性能

Triton kernels 针对以下方面进行了优化：
- 自动调优 (autotune) 最优 block 配置
- 高效的 chunk-wise 并行处理
- 最小化 GPU 内存访问

## 参考

- **DEST3D** (ICLR 2025): Interactive State Space Model for 3D Detection
- **Mamba** (NeurIPS 2023): Selective State Space Models
- **Triton**: OpenAI Triton Language

## License

Apache 2.0 License (继承自 DEST3D)
