# ISSM-StreamPETR 环境配置完整指南

**版本**: v1.0  
**最后更新**: 2025-12-17  
**适用于**: 改进后的 StreamPETR_mamba (DFA-ISSM)

---

## 📋 系统要求

### 硬件要求
- **GPU**: NVIDIA GPU，显存 ≥ 16GB（推荐 RTX 3090/4090 或 A100）
- **内存**: ≥ 32GB RAM
- **存储**: ≥ 500GB 可用空间（用于数据集）

### 软件要求
- **操作系统**: Linux (Ubuntu 18.04/20.04/22.04)
- **Python**: 3.8 - 3.10
- **CUDA**: 11.1 - 11.8（推荐 11.3）
- **GCC**: 7.5 - 9.x

---

## 🚀 快速安装（推荐）

### 方案 A: 完整安装脚本

创建并运行以下脚本：

```bash
#!/bin/bash
# save as: install_issm_streampetr.sh

set -e  # 遇到错误立即退出

# ========================================
# 1. 创建 Conda 环境
# ========================================
echo "Step 1: Creating conda environment..."
conda create -n issm_streampetr python=3.8 -y
source $(conda info --base)/etc/profile.d/conda.sh
conda activate issm_streampetr

# ========================================
# 2. 安装 PyTorch (CUDA 11.3)
# ========================================
echo "Step 2: Installing PyTorch..."
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 \
    --extra-index-url https://download.pytorch.org/whl/cu113

# 验证 PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# ========================================
# 3. 安装 MMDetection3D 及其依赖
# ========================================
echo "Step 3: Installing MMDetection3D..."
pip install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12.1/index.html
pip install mmdet==2.28.2
pip install mmsegmentation==0.30.0

# 克隆并安装 mmdetection3d
cd /tmp
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v1.0.0rc6
pip install -e .
cd -

# ========================================
# 4. 安装 Triton（纯 Triton ISSM 实现）
# ========================================
echo "Step 4: Installing Triton..."
pip install triton>=2.1.0

# 注意：不再需要 mamba_ssm！
# ISSM 使用纯 Triton 实现，位于 projects/mmdet3d_plugin/models/issm_triton/

# ========================================
# 5. 安装其他依赖
# ========================================
echo "Step 5: Installing additional dependencies..."
pip install \
    einops \
    flash-attn==0.2.8 \
    timm==0.9.2 \
    numpy==1.23.5 \
    opencv-python \
    pillow \
    matplotlib \
    tensorboard \
    scipy \
    scikit-learn \
    pyyaml \
    termcolor

# ========================================
# 6. 安装 PointNet2（如果需要）
# ========================================
echo "Step 6: Installing PointNet2 (optional)..."
cd /mnt/c/Users/17203/Desktop/Research/DEST3D/pointnet2
python setup.py install --user
cd -

# ========================================
# 7. 验证安装
# ========================================
echo "Step 7: Verifying installation..."
cd /mnt/c/Users/17203/Desktop/Research/StreamPETR_mamba
python -c "
import torch
import mmcv
import mmdet
import mmdet3d
print('✅ PyTorch:', torch.__version__)
print('✅ CUDA Available:', torch.cuda.is_available())
print('✅ MMCV:', mmcv.__version__)
print('✅ MMDet:', mmdet.__version__)
print('✅ MMDet3D:', mmdet3d.__version__)

try:
    from projects.mmdet3d_plugin.models.issm_triton import ISSM_chunk_scan_combined
    print('✅ Pure Triton ISSM installed')
except ImportError:
    print('⚠️  Pure Triton ISSM not available')

try:
    import triton
    print('✅ Triton:', triton.__version__)
except ImportError:
    print('⚠️  Triton not available')
"

echo "========================================="
echo "✅ Installation complete!"
echo "========================================="
echo "Next steps:"
echo "1. Prepare NuScenes dataset (see docs/data_preparation.md)"
echo "2. Download pretrained weights"
echo "3. Run: bash tools/dist_train.sh <config> <num_gpus>"
```

### 使用方法：
```bash
chmod +x install_issm_streampetr.sh
bash install_issm_streampetr.sh
```

---

## 🔧 分步安装（手动）

### Step 1: 创建 Conda 环境

```bash
conda create -n issm_streampetr python=3.8 -y
conda activate issm_streampetr
```

### Step 2: 安装 PyTorch

**选项 A: CUDA 11.3（推荐）**
```bash
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 \
    --extra-index-url https://download.pytorch.org/whl/cu113
```

**选项 B: CUDA 11.1**
```bash
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 \
    -f https://download.pytorch.org/whl/torch_stable.html
```

**选项 C: CUDA 11.8（如果使用 RTX 40 系列）**
```bash
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118
```

**验证安装**:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### Step 3: 安装 MMCV 和 MMDetection3D

```bash
# 安装 MMCV（根据您的 PyTorch 和 CUDA 版本选择）
# CUDA 11.3, PyTorch 1.12.1
pip install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12.1/index.html

# 安装 MMDetection 和 MMSegmentation
pip install mmdet==2.28.2
pip install mmsegmentation==0.30.0

# 克隆并安装 MMDetection3D
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v1.0.0rc6
pip install -v -e .
cd ..
```

### Step 4: 安装 Mamba-SSM

```bash
cd /mnt/c/Users/17203/Desktop/Research/mamba
pip install -e .
```

**或从 PyPI 安装**:
```bash
pip install mamba-ssm
```

**注意**: Mamba-SSM 需要编译 CUDA 内核，这可能需要几分钟。

### Step 5: 安装 Triton（用于 ISSM 加速）

```bash
# Triton 2.x (for CUDA 11.x)
pip install triton==2.0.0

# 或 Triton 3.x (for newer CUDA)
pip install triton==3.0.0
```

**验证**:
```bash
python -c "import triton; print(triton.__version__)"
```

### Step 6: 安装其他依赖

```bash
pip install einops
pip install timm==0.9.2
pip install numpy==1.23.5
pip install opencv-python
pip install pillow
pip install matplotlib
pip install tensorboard
pip install scipy
pip install scikit-learn
pip install pyyaml
pip install termcolor
```

### Step 7: 安装 Flash Attention（可选，但强烈推荐）

**对于 CUDA 11.3 + PyTorch 1.12**:
```bash
pip install flash-attn==0.2.8
```

**对于 CUDA 11.8 + PyTorch 2.0**:
```bash
pip install flash-attn==2.3.0
```

**注意**: 
- Flash Attention 需要较新的 GPU（Ampere 架构及以上，如 RTX 3090, A100）
- 如果编译失败，可以跳过此步骤（代码会自动使用标准 Attention）

### Step 8: 安装 PointNet2（如果需要）

```bash
cd /mnt/c/Users/17203/Desktop/Research/DEST3D/pointnet2
python setup.py install --user
cd -
```

---

## ✅ 验证安装

创建验证脚本 `verify_env.py`:

```python
#!/usr/bin/env python
import sys

def check_installation():
    errors = []
    
    # 1. PyTorch
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        if not torch.cuda.is_available():
            errors.append("❌ CUDA not available in PyTorch")
        else:
            print(f"✅ CUDA Available: {torch.version.cuda}")
            print(f"✅ GPU Count: {torch.cuda.device_count()}")
    except ImportError:
        errors.append("❌ PyTorch not installed")
    
    # 2. MMCV
    try:
        import mmcv
        print(f"✅ MMCV: {mmcv.__version__}")
    except ImportError:
        errors.append("❌ MMCV not installed")
    
    # 3. MMDetection
    try:
        import mmdet
        print(f"✅ MMDet: {mmdet.__version__}")
    except ImportError:
        errors.append("❌ MMDetection not installed")
    
    # 4. MMDetection3D
    try:
        import mmdet3d
        print(f"✅ MMDet3D: {mmdet3d.__version__}")
    except ImportError:
        errors.append("❌ MMDetection3D not installed")
    
    # 5. Mamba-SSM
    try:
        from mamba_ssm import Mamba
        print("✅ Mamba-SSM installed")
    except ImportError:
        errors.append("⚠️  Mamba-SSM not available (fallback will be used)")
    
    # 6. Triton
    try:
        import triton
        print(f"✅ Triton: {triton.__version__}")
    except ImportError:
        errors.append("⚠️  Triton not available (fallback will be used)")
    
    # 7. Flash Attention
    try:
        import flash_attn
        print("✅ Flash Attention installed")
    except ImportError:
        print("ℹ️  Flash Attention not available (optional)")
    
    # 8. Einops
    try:
        import einops
        print("✅ Einops installed")
    except ImportError:
        errors.append("❌ Einops not installed")
    
    # 9. 测试 ISSM 模块导入
    print("\n" + "="*50)
    print("Testing ISSM modules...")
    sys.path.insert(0, '/mnt/c/Users/17203/Desktop/Research/StreamPETR_mamba/projects')
    try:
        from mmdet3d_plugin.models.utils.single_direction_issm_layer import SingleDirectionISSMLayer
        print("✅ SingleDirectionISSMLayer imported successfully")
    except Exception as e:
        errors.append(f"❌ Failed to import ISSM layer: {e}")
    
    try:
        from mmdet3d_plugin.models.utils.issm_transformer import DenseAlternatingISSMDecoder
        print("✅ DenseAlternatingISSMDecoder imported successfully")
    except Exception as e:
        errors.append(f"❌ Failed to import ISSM decoder: {e}")
    
    # Summary
    print("\n" + "="*50)
    if errors:
        print("❌ Installation Issues Found:")
        for err in errors:
            print(f"  {err}")
        return 1
    else:
        print("✅ All checks passed! Environment is ready.")
        return 0

if __name__ == "__main__":
    sys.exit(check_installation())
```

运行验证：
```bash
python verify_env.py
```

---

## 🔍 常见问题排查

### 问题 1: CUDA 版本不匹配
```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

**解决方案**: 确保 PyTorch CUDA 版本与系统 CUDA 版本匹配：
```bash
nvcc --version  # 查看系统 CUDA 版本
python -c "import torch; print(torch.version.cuda)"  # 查看 PyTorch CUDA 版本
```

### 问题 2: MMCV 编译错误
```
ERROR: Failed building wheel for mmcv-full
```

**解决方案**: 使用预编译版本：
```bash
pip install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12.1/index.html
```

### 问题 3: Mamba-SSM 编译失败
```
error: command 'gcc' failed with exit status 1
```

**解决方案**: 安装编译工具：
```bash
sudo apt-get update
sudo apt-get install build-essential
```

### 问题 4: Triton 无法导入
```
ImportError: cannot import name 'triton' from 'triton'
```

**解决方案**: 重新安装 Triton：
```bash
pip uninstall triton -y
pip install triton==2.0.0 --no-cache-dir
```

### 问题 5: Flash Attention 编译失败
```
ninja: build stopped: subcommand failed
```

**解决方案**: Flash Attention 是可选的，可以跳过：
```bash
# 在代码中会自动回退到标准 Attention
# 或者尝试安装特定版本
pip install flash-attn==0.2.8 --no-build-isolation
```

### 问题 6: 显存不足
```
RuntimeError: CUDA out of memory
```

**解决方案**:
1. 减小 batch size（修改配置文件中的 `samples_per_gpu`）
2. 使用梯度累积
3. 启用 Flash Attention（节省显存）

---

## 🎯 推荐配置组合

### 配置 A: 高性能（推荐）
```
- GPU: RTX 3090 / 4090 / A100
- CUDA: 11.3
- PyTorch: 1.12.1
- Flash Attention: 0.2.8
- Triton: 2.0.0
```

### 配置 B: 兼容性
```
- GPU: RTX 2080 Ti / V100
- CUDA: 11.1
- PyTorch: 1.9.0
- Flash Attention: 跳过
- Triton: 2.0.0
```

### 配置 C: 最新硬件
```
- GPU: RTX 4090 / H100
- CUDA: 11.8
- PyTorch: 2.0.1
- Flash Attention: 2.3.0
- Triton: 3.0.0
```

---

## 📦 完整依赖列表

将以下内容保存为 `requirements_issm.txt`:

```txt
# Core
torch==1.12.1+cu113
torchvision==0.13.1+cu113
torchaudio==0.12.1

# MMDetection3D stack
mmcv-full==1.6.0
mmdet==2.28.2
mmsegmentation==0.30.0
mmdet3d==1.0.0rc6

# SSM and Acceleration
mamba-ssm>=1.0.0
triton==2.0.0
flash-attn==0.2.8  # optional

# Utilities
einops>=0.6.0
timm==0.9.2
numpy==1.23.5
opencv-python>=4.5.0
pillow>=9.0.0
matplotlib>=3.5.0
tensorboard>=2.10.0
scipy>=1.9.0
scikit-learn>=1.1.0
pyyaml>=6.0
termcolor>=2.0.0

# Optional
gpustat
wandb  # for experiment tracking
```

安装：
```bash
pip install -r requirements_issm.txt
```

---

## 🚀 下一步

环境配置完成后：

1. **准备数据集**: 参考 [docs/data_preparation.md](docs/data_preparation.md)
2. **下载预训练权重**: 参考主 README
3. **运行测试**:
   ```bash
   cd /mnt/c/Users/17203/Desktop/Research/StreamPETR_mamba
   python projects/test_issm_components.py
   python projects/test_dense_alternating_issm.py
   ```
4. **开始训练**:
   ```bash
   bash tools/dist_train.sh projects/configs/issm_streampetr/issm_streampetr_r50.py 8
   ```

---

## 📞 获取帮助

如果遇到问题：
1. 检查 CUDA 和 PyTorch 版本是否匹配
2. 查看 [常见问题排查](#常见问题排查) 部分
3. 运行 `verify_env.py` 诊断环境

**祝您训练顺利！** 🎉
