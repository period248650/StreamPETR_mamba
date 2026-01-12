#!/usr/bin/env bash
# 单 GPU 测试脚本（避免 DDP 兼容性问题）

CONFIG=$1
CHECKPOINT=$2

if [ -z "$CONFIG" ] || [ -z "$CHECKPOINT" ]; then
    echo "Usage: bash tools/single_gpu_test.sh CONFIG CHECKPOINT [OPTIONS]"
    echo "Example: bash tools/single_gpu_test.sh projects/configs/issm_streampetr/issm_streampetr_dense_alternating.py ckpts/test/latest.pth --eval bbox"
    exit 1
fi

# 移除前两个参数，剩余的作为额外选项
shift 2

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=0 \
python $(dirname "$0")/test.py \
    $CONFIG \
    $CHECKPOINT \
    --launcher none \
    "$@"
