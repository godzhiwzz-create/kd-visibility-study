#!/bin/bash
# KD可见度研究 - 云端实验启动脚本

# 设置环境
export PATH=/root/miniconda3/bin:$PATH
export PYTHONPATH=/root/autodl-tmp/kd_visibility_claude:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

PYTHON=/root/miniconda3/bin/python

# 参数
KD_BRANCH=${1:-"student_only"}
VISIBILITY=${2:-"light"}
EPOCHS=${3:-100}

echo "=========================================="
echo "启动实验: KD分支=$KD_BRANCH, 可见度=$VISIBILITY"
echo "=========================================="
echo "Python版本: $($PYTHON --version)"
echo "CUDA可用: $($PYTHON -c 'import torch; print(torch.cuda.is_available())')"

# 安装依赖（首次运行）
echo "检查依赖..."
$PYTHON -c "import ultralytics" 2>/dev/null || pip install -q ultralytics
$PYTHON -c "import tensorboard" 2>/dev/null || pip install -q tensorboard
$PYTHON -c "import scipy" 2>/dev/null || pip install -q scipy

# 运行训练
cd /root/autodl-tmp/kd_visibility_claude
echo "开始训练..."
$PYTHON scripts/train_yolo.py \
    --kd-branch $KD_BRANCH \
    --visibility $VISIBILITY \
    --epochs $EPOCHS

echo "=========================================="
echo "实验完成!"
echo "=========================================="
