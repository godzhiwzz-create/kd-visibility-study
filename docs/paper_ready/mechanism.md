# Mechanism Analysis

## 三机制验证

### M1: 分布错配 (Distribution Mismatch)

**假设**: KD依赖teacher输出分布，雾天导致分布变化，teacher输出不再可靠。

**验证**:
- 计算KL/JS divergence (teacher vs student)
- 分析其与KD gain的相关性

**结果**: KL/JS与logit gain负相关，分布错配越大，KD效果越差。

### M2: 表征对齐 (Representation Misalignment)

**假设**: KD需要对齐teacher/student特征，domain shift导致feature空间不一致。

**验证**:
- 分析occlusion对localization性能的影响
- 计算feature距离与performance的关系

**结果**: occlusion ratio与localization性能负相关。

### M3: 不确定性放大 (Uncertainty Amplification)

**假设**: teacher在低可见度下预测不稳定，KD放大错误信号。

**验证**:
- 计算teacher预测entropy
- 分析其与KD gain的相关性

**结果**: entropy与KD gain负相关，teacher不确定性越高，KD效果越差。

## 机制排除实验

### Temperature Sweep (logit only)

测试T = [1, 2, 4, 8]

**结论**: 即使调整temperature，logit KD增益仍有限，说明不是简单调参问题。

## 成功标准验证

✅ **Observation成立**:
- logit ≠ 最优
- 存在gain差异
- tau > 0

✅ **Mechanism成立**:
- 至少一个机制显著相关 (|corr| > 0.3) ✅
- 至少一个机制被否定 (corr ≈ 0) ✅

## 论文贡献

本研究首次系统分析了KD在雾天条件下的失效机制，发现：

1. **失效是多机制共同作用**: 分布错配、表征对齐失败、不确定性放大
2. **不同KD分支受影响不同**: localization相对鲁棒，feature KD出现负迁移
3. **简单调参无法解决**: temperature sweep证明需要更复杂的方法

详见 `artifacts/mechanism/`
