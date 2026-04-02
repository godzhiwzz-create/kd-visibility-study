# KD Visibility Mechanism Study: 实验补强计划（一区标准）

> **目标**: 解决 Correlation≠Causation、Statistical Fragility、Generalization Gap，形成机制→方法的完整闭环
>
> **硬件配置**: 25核CPU / 90GB内存 / RTX 5090 32GB

---

## 硬件资源分配策略

```
总资源: 25核CPU + 90GB内存 + 32GB显存

分配方案:
├── DataLoader: 16核 (留9核给训练)
├── 内存缓存: 60GB (留30GB给系统)
├── 显存: 30GB (留2GB余量)
└── 并行实验: 2-3个 (如有多卡)
```

---

## 优化后的超参数配置

### 1. DataLoader优化（CPU密集型）

| 参数 | 低配(8核/16G) | **本机(25核/90G)** | 提升 |
|------|--------------|-------------------|------|
| **num_workers** | 8 | **16** | 2x |
| **pin_memory** | False | **True** | 加速H2D传输 |
| **persistent_workers** | False | **True** | 减少进程创建开销 |
| **prefetch_factor** | 2 | **4** | 预加载更多batch |
| **batch_size** | 16 | **32** | 2x |

```python
# 优化后的DataLoader
train_loader = DataLoader(
    dataset,
    batch_size=32,           # 显存允许的最大值
    num_workers=16,          # 25核中分配16核给数据加载
    pin_memory=True,         # 加速CPU->GPU传输
    persistent_workers=True, # 保持worker进程存活
    prefetch_factor=4,       # 每个worker预加载4个batch
)
```

### 2. 内存缓存策略（90GB优势）

```python
# 方案A: 全内存缓存（如果数据集<60GB）
# CityFog约20-30GB，可以完全载入内存

class MemoryCachedDataset:
    """90GB内存允许全量缓存"""
    def __init__(self, data_root, cache_in_memory=True):
        self.data_root = data_root
        self.cache = {}

        if cache_in_memory:
            self._preload_to_memory()

    def _preload_to_memory(self):
        """预加载全部数据到内存（25核并行）"""
        from multiprocessing import Pool

        with Pool(processes=25) as pool:  # 全核并行加载
            self.cache = pool.map(self._load_single, self.image_paths)
        print(f"已缓存 {len(self.cache)} 张图像到内存")

# 方案B: 磁盘缓存 + 大页内存
# 使用Linux HugePages优化
```

### 3. 多实验并行（25核支持）

```python
# 即使单卡，也可以用CPU并行跑多个实验的预处理/评估

from concurrent.futures import ProcessPoolExecutor
import subprocess

experiments = [
    (0.0, 'student_only'),
    (0.1, 'student_only'),
    # ... 18个实验
]

def run_experiment(args):
    occ, method = args
    cmd = f"python train_causal.py --occlusion {occ} --method {method}"
    return subprocess.run(cmd, shell=True)

# 3个实验并行跑（每个用8核，共24核）
with ProcessPoolExecutor(max_workers=3) as executor:
    executor.map(run_experiment, experiments)
```

---

## Part 3: Causal Validation（优化版）

### 3.1 实验设计

| 变量 | 控制方式 | 取值 |
|------|----------|------|
| Visibility (雾) | 固定为0 | 无雾，β=0 |
| Occlusion (遮挡) | 人为控制 | 0, 0.1, 0.2, 0.3, 0.4, 0.5 |
| KD Branch | 3种 | student-only, logit KD, localization KD |

### 3.2 样本量与时间估算

```
配置: 25核 / 90GB / 5090 32GB / batch=32

单实验:
- 训练时间: ~6-8小时 (原18小时)
- 峰值显存: ~26GB
- 峰值内存: ~40GB

18个实验:
- 串行: 5-6天 (原13天)
- 3路并行: 2天
```

### 3.3 优化后的训练脚本

```python
# train_causal_optimized.py
import torch
import multiprocessing as mp

# 根据硬件自动优化配置
AUTO_CONFIG = {
    'num_workers': min(16, mp.cpu_count() - 9),  # 25核留9核给训练
    'batch_size': 32 if torch.cuda.get_device_properties(0).total_memory > 30e9 else 16,
    'pin_memory': True,
    'persistent_workers': True,
    'prefetch_factor': 4,
    'cache_in_memory': True,  # 90GB允许内存缓存
}

def get_optimal_config():
    """根据硬件返回最优配置"""
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    cpu_count = mp.cpu_count()

    config = {
        # DataLoader
        'num_workers': min(16, cpu_count - 4),
        'pin_memory': True,
        'persistent_workers': True,
        'prefetch_factor': 4,

        # 训练
        'batch_size': 32 if gpu_mem >= 30 else 16,
        'imgsz': 640,
        'epochs': 150,

        # 优化器
        'lr0': 0.01,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,

        # 内存优化
        'cache_images': True,  # 预加载到内存
        'half': True,  # FP16
    }

    return config
```

### 3.4 批量启动脚本（支持并行）

```bash
#!/bin/bash
# run_causal_optimized.sh
# 针对 25核/90GB/32GB 优化

OCCLUSION_RATIOS=(0.0 0.1 0.2 0.3 0.4 0.5)
METHODS=("student_only" "logit_only" "localization_only")

# 检测硬件
CPU_COUNT=$(nproc)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)

echo "硬件配置: ${CPU_COUNT}核 / ${GPU_MEM}MB显存"

# 根据硬件选择batch size
if [ "$GPU_MEM" -gt 30000 ]; then
    BATCH_SIZE=32
    WORKERS=16
    CACHE="--cache ram"  # 内存缓存
else
    BATCH_SIZE=16
    WORKERS=8
    CACHE=""
fi

echo "优化配置: batch=$BATCH_SIZE, workers=$WORKERS"

# 运行18个实验
total=0
for occ in "${OCCLUSION_RATIOS[@]}"; do
    for method in "${METHODS[@]}"; do
        total=$((total + 1))
        echo "[$total/18] occlusion=$occ, method=$method"

        python causal_experiment/train_causal.py \
            --occlusion_ratio $occ \
            --kd_branch $method \
            --epochs 150 \
            --batch_size $BATCH_SIZE \
            --workers $WORKERS \
            --pin_memory \
            --persistent_workers \
            --prefetch_factor 4 \
            $CACHE \
            --half  # FP16加速
    done
done
```

---

## Part 4: Occlusion-aware KD

### 4.1 超参数

| 参数 | 取值 | 说明 |
|------|------|------|
| **temperature** | 4.0 | KD温度系数 |
| **kd_weight** | 1.0 | KD损失权重 |
| **occlusion_aware** | True | 启用遮挡感知 |

### 4.2 内存优化实现

```python
# occlusion_aware_kd_optimized.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class OcclusionAwareKD(nn.Module):
    """
    遮挡感知知识蒸馏 - 内存优化版
    """
    def __init__(self, base_temperature=4.0, occlusion_aware=True):
        super().__init__()
        self.temperature = base_temperature
        self.occlusion_aware = occlusion_aware

    @torch.cuda.amp.autocast()  # 自动混合精度
    def forward(self, student_logits, teacher_logits,
                transmission_map=None, targets=None):
        """
        使用FP16减少显存占用
        """
        # FP16计算
        student_probs = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)

        kl_div = F.kl_div(student_probs, teacher_probs, reduction='none')

        if self.occlusion_aware and transmission_map is not None:
            w = self.compute_visibility_weight(transmission_map, teacher_logits)
            weighted_kl = (kl_div * w).sum() / (w.sum() + 1e-8)
        else:
            weighted_kl = kl_div.mean()

        return weighted_kl * (self.temperature ** 2)
```

---

## Part 5: Generalization（Faster R-CNN）

### 5.1 优化配置

| 参数 | 低配 | **本机优化** | 理由 |
|------|------|-------------|------|
| batch_size | 4 | **12** | 32GB显存支持 |
| num_workers | 4 | **12** | 25核支持 |
| pin_memory | False | **True** | 加速传输 |

### 5.2 训练脚本

```python
# faster_rcnn_optimized.py

def get_faster_rcnn_config():
    return {
        'epochs': 12,
        'batch_size': 12,  # 32GB显存支持
        'lr': 0.005,
        'momentum': 0.9,
        'weight_decay': 0.0005,
        'num_workers': 12,  # 25核分配
        'pin_memory': True,
    }
```

---

## Part 6: Statistical Robustness

### 6.1 并行化策略

```python
# 90GB内存支持3路并行（每路30GB）

EXPERIMENTS = []
for beta in [0.003, 0.005, 0.008, 0.01, 0.015, 0.02]:
    for seed in [42, 43, 44]:
        for branch in ["student", "logit", "localization"]:
            EXPERIMENTS.append((beta, seed, branch))

# 3路并行（利用25核）
from multiprocessing import Pool

def run_exp(args):
    beta, seed, branch = args
    cmd = f"python train.py --beta {beta} --seed {seed} --branch {branch}"
    subprocess.run(cmd, shell=True)

with Pool(3) as p:  # 3路并行
    p.map(run_exp, EXPERIMENTS)

# Wall time: 22天 → 7天
```

---

## 预估资源（优化后）

| Phase | 实验数 | 原GPU时间 | **优化后** | 加速比 |
|-------|--------|-----------|-----------|--------|
| Phase 1 (Causal) | 18 | ~13天 | **~5天** | 2.6x |
| Phase 2 (Stat) | 90 | ~66天 | **~22天** | 3x |
| Phase 3 (Method) | ~5 | ~4天 | **~1.5天** | 2.7x |
| Phase 4 (General) | 6 | ~8天 | **~3天** | 2.7x |
| **总计** | **119** | **~91天** | **~31天** | **2.9x** |

**建议**: Phase 1+3 现在只需 **~6.5天**（原5天）

---

## 系统级优化

### 1. Linux内核优化（90GB内存）

```bash
# /etc/sysctl.conf
vm.swappiness=10          # 减少swap
vm.dirty_ratio=40         # 增加dirty page
vm.dirty_background_ratio=10
kernel.numa_balancing=1   # NUMA优化

# 大页内存（HugePages）
echo 2048 > /proc/sys/vm/nr_hugepages  # 4GB大页内存
```

### 2. PyTorch优化环境变量

```bash
# .bashrc 或启动脚本
export OMP_NUM_THREADS=16           # OpenMP线程
export MKL_NUM_THREADS=16           # MKL线程
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  # 减少显存碎片
export CUDA_LAUNCH_BLOCKING=0       # 异步启动
```

### 3. 数据预加载脚本

```bash
#!/bin/bash
# preload_data.sh
# 预加载数据集到内存缓存

echo "预加载CityFog数据集到内存..."
find /root/kd_visibility/data/CityFog -name "*.png" -o -name "*.jpg" | \
    xargs -P 25 -I {} cat {} > /dev/null  # 25核并行读取
echo "预加载完成"
```

---

## 快速启动命令

```bash
# SSH登录
ssh -p 39655 -i ~/.ssh/id_ed25519_autodl root@connect.bjb2.seetacloud.com

cd /root/kd_visibility

# 1. 系统优化
sudo sysctl -w vm.swappiness=10
export OMP_NUM_THREADS=16

# 2. 预加载数据
bash scripts/preload_data.sh

# 3. 运行Phase 1（优化版）
bash run_causal_optimized.sh

# 或3路并行（如果有数据并行需求）
python run_parallel.py --workers 3
```

---

*文档版本: 2.0 (5090+25核+90GB优化版)*
*最后更新: 2026-04-01*
