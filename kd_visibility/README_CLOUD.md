# KD可见度研究 - 云端训练指南

## 环境配置

云端服务器已配置：
- GPU: NVIDIA GeForce RTX 5090 (32GB)
- CUDA: 13.0
- 数据集路径: `/root/autodl-tmp/shared_datasets/low_visibility_kd/cityscapes_yolo`

## 快速开始

### 1. 单实验运行

```bash
cd /root/autodl-tmp/kd_visibility_claude
bash run_experiment.sh <kd_branch> <visibility> <epochs>
```

参数：
- `kd_branch`: student_only, logit_only, feature_only, attention_only, localization_only
- `visibility`: light, moderate, heavy
- `epochs`: 训练轮数（默认100）

示例：
```bash
# 运行 logit-only + light 实验
bash run_experiment.sh logit_only light 100

# 运行 student-only baseline + heavy
bash run_experiment.sh student_only heavy 100
```

### 2. 批量运行5×3实验矩阵

```bash
cd /root/autodl-tmp/kd_visibility_claude
python scripts/run_matrix.py --epochs 100 --output-root outputs
```

### 3. 后台运行（推荐）

使用 tmux/screen 保持会话：
```bash
tmux new -s kd_experiment
bash run_experiment.sh logit_only light 100
# Ctrl+B, D 分离会话
tmux attach -t kd_experiment  # 重新连接
```

或使用 nohup：
```bash
nohup bash run_experiment.sh logit_only light 100 > log.txt 2>&1 &
tail -f log.txt
```

## 实验矩阵

| KD Branch | Light (β=0.005) | Moderate (β=0.01) | Heavy (β=0.02) |
|-----------|-----------------|-------------------|----------------|
| student_only | ✅ | ✅ | ✅ |
| logit_only | ✅ | ✅ | ✅ |
| feature_only | ✅ | ✅ | ✅ |
| attention_only | ✅ | ✅ | ✅ |
| localization_only | ✅ | ✅ | ✅ |

## 输出目录

```
outputs/
└── kd_visibility_yolo/
    ├── student_only/
    │   ├── light/
    │   ├── moderate/
    │   └── heavy/
    ├── logit_only/
    │   └── ...
    └── ...
```

每个实验输出：
- `best.pt`: 最佳模型权重
- `training_history.json`: 训练历史
- `config.json`: 实验配置
- `logs/`: TensorBoard日志

## 监控训练

```bash
# TensorBoard
tensorboard --logdir outputs/kd_visibility_yolo --port 6006

# 查看GPU使用
nvidia-smi -l 1

# 查看实验进度
watch -n 10 'ls outputs/kd_visibility_yolo/*/*/training_history.json 2>/dev/null | wc -l'
```

## 安装依赖

```bash
pip install ultralytics tensorboard scipy
```

## 数据说明

- **clear**: 清晰图像 (`cityscapes_yolo/clear/images/`)
- **foggy**: 雾天图像 (`cityscapes_yolo/foggy_all/images/`)
  - `*_beta_0.005.png`: light 可见度
  - `*_beta_0.010.png`: moderate 可见度
  - `*_beta_0.020.png`: heavy 可见度
