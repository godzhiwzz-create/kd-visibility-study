# 实验补强计划 - 超参数配置

> 本文档详细记录所有实验的超参数设置，确保可复现性

---

## 1. 基础训练超参数（所有实验共用）

| 参数 | 取值 | 说明 |
|------|------|------|
| **epochs** | 150 | 训练轮数（Causal/Statistical） |
| **batch_size** | 16 | YOLOv8 batch size |
| **imgsz** | 640 | 输入图像尺寸 |
| **lr0** | 0.01 | 初始学习率 |
| **lrf** | 0.01 | 最终学习率比例 (lr0 * lrf) |
| **optimizer** | SGD | 优化器 |
| **momentum** | 0.937 | SGD动量 |
| **weight_decay** | 0.0005 | 权重衰减 |
| **seed** | 42 | 随机种子（默认） |

### 学习率调度

YOLOv8 默认使用 `cos_lr`（余弦退火）：
```python
lr_schedule = cosine_annealing(epoch, epochs, lr0, lrf * lr0)
```

---

## 2. Phase 1: Causal Validation

### 2.1 实验配置

| 参数 | 取值 |
|------|------|
| occlusion_ratio | [0.0, 0.1, 0.2, 0.3, 0.4, 0.5] |
| kd_branch | ["student_only", "logit_only", "localization_only"] |
| beta (雾度) | 0.0（固定，无雾） |
| total_experiments | 6 × 3 = 18 |

### 2.2 训练命令

```bash
python causal_experiment/train_causal.py \
    --occlusion_ratio 0.2 \
    --kd_branch localization_only \
    --epochs 150 \
    --batch_size 16 \
    --lr0 0.01 \
    --lrf 0.01 \
    --seed 42
```

---

## 3. Phase 2: Statistical Robustness

### 3.1 实验配置

| 参数 | 取值 |
|------|------|
| beta | [0.003, 0.005, 0.008, 0.01, 0.015, 0.02] |
| seeds | [42, 43, 44] |
| kd_branch | ["student_only", "logit_only", "localization_only", "feature_only", "attention_only"] |
| total_experiments | 6 × 3 × 5 = 90 |

### 3.2 训练命令

```bash
# 示例：β=0.01, seed=43, localization KD
python scripts/train_yolo_kd.py \
    --beta 0.01 \
    --kd_branch localization_only \
    --epochs 150 \
    --batch_size 16 \
    --seed 43
```

---

## 4. Phase 3: Occlusion-aware KD

### 4.1 OcclusionAwareKD 超参数

| 参数 | 取值 | 说明 |
|------|------|------|
| **temperature** | 4.0 | KD温度系数 |
| **kd_weight** | 1.0 | KD损失权重 (λ_KD) |
| **occlusion_aware** | True | 是否启用遮挡感知 |

### 4.2 损失函数

```python
L_total = L_det + λ_KD * L_OAKD

其中：
- L_det: 检测损失（分类 + 定位）
- L_OAKD: 遮挡感知KD损失
- λ_KD = 1.0
```

### 4.3 权重计算公式

```python
W(x) = t(x) · (1 - H(P_T(x)) / H_max)

其中：
- t(x): transmission map（雾的物理模型）
- H(P_T(x)): 教师预测熵
- H_max = log(num_classes)
```

### 4.4 对比实验配置

| 方法 | 配置 |
|------|------|
| Student-only | baseline，无KD |
| Vanilla KD | temperature=4.0, occlusion_aware=False |
| OAKD (Ours) | temperature=4.0, occlusion_aware=True |

---

## 5. Phase 4: Faster R-CNN Generalization

### 5.1 架构差异

| 参数 | YOLOv8 | Faster R-CNN |
|------|--------|--------------|
| batch_size | 16 | 4（内存限制） |
| epochs | 150 | 12（收敛更快） |
| lr | 0.01 | 0.005 |
| momentum | 0.937 | 0.9 |
| weight_decay | 0.0005 | 0.0005 |
| optimizer | SGD | SGD |

### 5.2 训练命令

```bash
python faster_rcnn_kd/train_faster_rcnn.py \
    --kd_branch localization_only \
    --epochs 12 \
    --batch_size 4 \
    --lr 0.005 \
    --momentum 0.9 \
    --weight_decay 0.0005 \
    --kd_weight 1.0 \
    --temperature 4.0
```

---

## 6. 模型架构配置

### 6.1 YOLOv8 教师-学生配置

| 角色 | 模型 | 参数量 | 预训练 |
|------|------|--------|--------|
| Teacher | YOLOv8s | ~11M | COCO预训练 |
| Student | YOLOv8n | ~3.2M | COCO预训练 |

### 6.2 Faster R-CNN 配置

| 角色 | 模型 | 参数量 | 预训练 |
|------|------|--------|--------|
| Teacher | Faster R-CNN R50-FPN | ~41M | ImageNet预训练 |
| Student | Faster R-CNN R50-FPN | ~41M | 从头训练 |

---

## 7. 数据增强配置

### 7.1 标准YOLO增强（所有实验）

```yaml
hsv_h: 0.015    # HSV色调增强
hsv_s: 0.7      # HSV饱和度增强
hsv_v: 0.4      # HSV亮度增强
degrees: 0.0    # 旋转角度
translate: 0.1  # 平移比例
scale: 0.5      # 缩放比例
shear: 0.0      # 剪切角度
flipud: 0.0     # 上下翻转概率
fliplr: 0.5     # 左右翻转概率
mosaic: 1.0     # Mosaic增强概率
mixup: 0.0      # MixUp增强概率
copy_paste: 0.0 # Copy-paste增强概率
```

### 7.2 Occlusion增强（仅Phase 1）

```python
occlusion_ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
occ_color = 128  # 灰色填充
occ_min_bbox_size = 100  # 最小bbox面积阈值
```

---

## 8. 评估配置

| 参数 | 取值 | 说明 |
|------|------|------|
| conf_threshold | 0.001 | 置信度阈值（评估时） |
| iou_threshold | 0.6 | NMS IoU阈值 |
| max_det | 300 | 最大检测数 |
| metrics | mAP@50, mAP@50:95 | 评估指标 |

---

## 9. 关键设计决策

### 9.1 为什么 temperature=4.0？

- Hinton原始KD论文推荐：3-10
- 检测任务通常用4.0（平衡分类和定位）
- 过高：软标签过于平滑，信息丢失
- 过低：接近硬标签，KD效果减弱

### 9.2 为什么 kd_weight=1.0？

- 与检测损失同量级
- 可根据实验调整（0.5-2.0范围）
- Phase 3可尝试动态权重

### 9.3 为什么 epochs=150？

- YOLOv8默认配置
- Causal实验需要充分收敛以观察趋势
- 与原始实验保持一致，确保可比性

---

## 10. 完整配置YAML示例

```yaml
# causal_experiment_config.yaml

# 实验配置
experiment:
  name: "causal_validation"
  phase: 1
  occlusion_ratios: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
  kd_branches: ["student_only", "logit_only", "localization_only"]
  beta: 0.0  # 无雾

# 训练超参数
training:
  epochs: 150
  batch_size: 16
  imgsz: 640
  lr0: 0.01
  lrf: 0.01
  optimizer: SGD
  momentum: 0.937
  weight_decay: 0.0005
  seed: 42

# KD超参数
kd:
  temperature: 4.0
  kd_weight: 1.0
  occlusion_aware: false  # Phase 1不使用

# 模型配置
model:
  teacher: "yolov8s.pt"
  student: "yolov8n.pt"
  num_classes: 8

# 数据配置
data:
  path: "data/CityFog"
  train: "images/train"
  val: "images/val"
  test: "images/test"

# 评估配置
evaluation:
  conf_threshold: 0.001
  iou_threshold: 0.6
  metrics: ["mAP50", "mAP50:95"]
```

---

## 11. 超参数调优建议

### 11.1 如果Phase 1结果不理想

| 问题 | 解决方案 |
|------|----------|
| 趋势不明显 | 增加epochs到200，或调整occlusion_ratios范围 |
| KD gain太小 | 调整kd_weight到2.0，或temperature到2.0/8.0 |
| 过拟合 | 增加weight_decay到0.001，或添加dropout |

### 11.2 Phase 3 OAKD调优

```python
# 可调参数网格
temperatures = [2.0, 4.0, 8.0]
kd_weights = [0.5, 1.0, 2.0]
occlusion_aware_modes = [True, False]

# 共 3 × 3 × 2 = 18个组合需测试
```

---

*文档版本: 1.0*
*最后更新: 2026-04-01*
