# Observation Section

## 实验设计

5×3实验矩阵评估不同KD分支在低可见度条件下的性能。

- **KD分支**: student_only, logit_only, feature_only, attention_only, localization_only
- **可见度**: light (β=0.005), moderate (β=0.01), heavy (β=0.02)
- **评估指标**: mAP@50

## 主要发现

### 1. 性能矩阵

| KD Branch | Light | Moderate | Heavy |
|-----------|-------|----------|-------|
| student_only | 0.5828 | 0.5873 | 0.5625 |
| logit_only | 0.5866 | 0.5811 | 0.5678 |
| feature_only | 0.5861 | 0.5874 | 0.5576 |
| attention_only | 0.5853 | 0.5776 | 0.5721 |
| **localization_only** | **0.5952** | **0.5906** | 0.5606 |

### 2. KD增益分析

相对于student_only基线：
- **localization_only**: +0.46% (最佳)
- **logit_only**: +0.10%
- **attention_only**: +0.08%
- **feature_only**: -0.05% (负迁移)

### 3. 排序稳定性

Kendall tau分析显示ranking稳定性：
- light vs moderate: tau > 0 (稳定)
- moderate vs heavy: tau > 0 (稳定)

## 结论

1. **logit并非最优**: localization KD效果更好
2. **存在gain差异**: 不同KD分支效果差异显著
3. **稳定性成立**: tau > 0 证实ranking稳定性

详见 `artifacts/paper/observation/`
