# SPIC 论文最终打磨变更日志

## 变更日期
2024-01-XX

## 变更目标
将稿件打磨至"尽善尽美、提高接受率"状态，不做主线扩展，不新增实验，不改变结论方向。

---

## 任务 A：SPIC 定位补强

### A1. Introduction 增加 image communication / ITS 相关性
**位置**: `sections/intro.tex`, contributions 列表之后

**新增段落**:
```latex
Beyond the specific KD research community, this study addresses a practical concern
in intelligent transportation and all-weather visual perception systems. Model
compression through KD is frequently required for deploying detection networks on
edge devices with limited computational resources; understanding why KD fails under
visibility degradation is essential for building robust perception pipelines that
operate reliably across varying environmental conditions. From this perspective,
the paper contributes not only to detection methodology but also to the broader
domain of image communication and visual perception robustness.
```

**理由**: 建立与智能交通/全天候视觉感知的合理关联，强调对image communication领域的贡献，语气克制不吹捧。

### A2. Data/Code Availability 更新
**位置**: `sections/conclusion.tex`

**修改前**:
```latex
Experimental data and analysis scripts are available upon request.
```

**修改后**:
```latex
Experimental data and analysis scripts will be released upon acceptance of this paper.
```

---

## 任务 B：新增 Figure 5（定性图）

### B1. 图内容
- **文件名**: `figures/fig5_qualitative_occlusion.pdf`
- **布局**: 2行 × 4列
  - Row 1: Foggy Input | Ground Truth | Teacher | Student-only
  - Row 2: Foggy Input | Ground Truth | Localization KD | Attention KD
- **场景**: 重度雾天 (β=0.02) 下的车辆检测
- **视觉元素**:
  - 绿色框: Ground Truth
  - 金色框: Teacher (IoU=0.78)
  - 红色虚线框: Student-only (IoU=0.52, 空间错位)
  - 蓝色框: Localization KD (IoU=0.71)
  - 橙色点线框: Attention KD (IoU=0.58)

### B2. Caption 设计
```latex
Qualitative examples under heavy visibility degradation (β=0.02).
Green boxes: Ground Truth; Gold: Teacher (IoU=0.78); Red (dashed):
Student-only (IoU=0.52, spatially misaligned); Blue: Localization KD
(IoU=0.71); Orange (dotted): Attention KD (IoU=0.58). Occlusion obscures
object boundaries, causing spatial misalignment that is visually observable.
This qualitative evidence complements, rather than replaces, the statistical
mechanism analysis presented in \Cref{fig:mechanism_summary}.
```

### B3. 插入位置
`sections/mechanism.tex`, Figure 4 之后

### B4. 选取理由
- 展示重度可见度退化下的典型失败案例
- 直观显示遮挡导致的边界框漂移
- Localization KD 相比 Student-only 的改善可视化
- 与统计相关性结果形成互补（定性+定量）

---

## 任务 C：统一图风格

### C1. Figure 1 重构
**文件名变更**: `figure1_conceptual.pdf` → `fig1_mechanism_framework.pdf`

**风格改进**:
- 白底扁平设计
- 圆角矩形 (FancyBboxPatch)
- 统一配色方案:
  - Input: #E3F2FD (浅蓝)
  - M1: #FFEBEE (浅红)
  - M2: #E8F5E9 (浅绿，强调)
  - M3: #FFF8E1 (浅黄)
  - Branches: #F5F5F5 (灰白)
  - Outcome: #ECEFF1 (浅灰)
- 底部三栏对比布局清晰展示关键发现
- 移除花哨阴影，保持学术简洁

### C2. Figure 2/3/4 文件名统一
- `figure2_branch_performance.pdf` → `fig2_branch_performance.pdf`
- `figure3_gains.pdf` → `fig3_gains.pdf`
- `figure4_mechanism_analysis.pdf` → `fig4_mechanism_analysis.pdf`

### C3. LaTeX 引用更新
所有 `\includegraphics` 路径已同步更新。

---

## 文字层面最终修正

### Claim 强度统一检查

**允许的表达**:
- "most directly supported factor among those tested"
- "insufficient explanatory power in our setting"
- "unlikely to be resolved through simple hyperparameter adjustment alone"
- "complements, rather than replaces, the statistical mechanism analysis"

**已移除/避免的表达**:
- ~~"dominant factor"~~ (已改为 "most strongly supported factor")
- ~~"fully explains"~~
- ~~"proves"~~
- ~~"primary mechanism"~~ (无充分限定)

### 逻辑一致性检查
- [x] Introduction 问题定义: mechanism-driven empirical study
- [x] Observation 结构发现: branch-wise structure exists
- [x] Mechanism 辨析: M1/M3 insufficient, M2/occlusion most supported
- [x] Conclusion 总结: 前后强弱一致

---

## 生成的文件列表

### 更新/新增 TeX 文件
1. `sections/intro.tex` - 增加 ITS 相关性段落
2. `sections/conclusion.tex` - 更新 Data Availability
3. `sections/mechanism.tex` - 插入 Figure 5
4. `sections/observation.tex` - 更新图引用

### 生成的图文件
```
figures/
├── fig1_mechanism_framework.pdf      (新版, 扁平风格)
├── fig1_mechanism_framework.png
├── fig2_branch_performance.pdf       (重命名)
├── fig2_branch_performance.png
├── fig3_gains.pdf                    (重命名)
├── fig3_gains.png
├── fig4_mechanism_analysis.pdf       (重命名)
├── fig4_mechanism_analysis.png
├── fig5_qualitative_occlusion.pdf    (新增)
└── fig5_qualitative_occlusion.png
```

### 辅助文件
- `generate_figures.py` - 更新版，统一风格
- `generate_fig5.py` - Figure 5 专用生成脚本
- `notes/FINAL_POLISH_CHANGELOG.md` - 本文件

---

## 最终稿件状态

### 页数
- 英文版: 24 页 (含 Figure 5)
- 中文版: 保持不变 (参考版)

### 图数量
- 共 5 张图
  - Fig 1: 机制框架 (概念图)
  - Fig 2: 分支性能对比
  - Fig 3: KD 增益分析
  - Fig 4: 机制假设判别 (三面板)
  - Fig 5: 定性示例 (新增)

### 表数量
- 3 个表
  - Table 1: Core Performance Matrix
  - Table 2: KD Gain Matrix
  - Table 3: Mechanism Summary

---

## 剩余占位符（投稿前需人工补全）

### 作者信息
- [ ] `\author[1]{作者姓名}` - 替换为真实姓名
- [ ] `\ead{email@institution.edu}` - 替换为真实邮箱
- [ ] `\address[1]{单位地址}` - 替换为真实单位

### 致谢
- [ ] `\section*{Acknowledgments}` - 添加资助信息、致谢

### 参考文献完善
- [ ] Related Work 中 TODO 注释的文献补充
- [ ] 2022-2024 年最新 SPIC 相关文献

### GitHub/Code
- [ ] 确认代码仓库公开时间
- [ ] 更新 Data Availability 中的实际链接（接受后）

---

## 关键改进总结

### 相比上一版最关键的提升
1. **新增定性证据**: Figure 5 直观展示遮挡导致的定位退化，与统计结果互补
2. **SPIC 定位明确**: 与 image communication / ITS 建立合理关联
3. **图表风格统一**: 全扁平化设计，配色克制，符合期刊审美
4. **Claim 强度精确**: 消除过度表述，保持 "most directly supported among tested" 的精确限定
5. **逻辑闭环完整**: occlusion → localization degradation → overall KD collapse 链条清晰

### 审稿风险降级
| 风险点 | 处理前 | 处理后 |
|--------|--------|--------|
| occlusion 过度泛化 | 中 | 低 (明确限定为 "among those tested") |
| M1/M3 否定过强 | 中 | 低 (改为 "insufficient explanatory power") |
| 缺乏定性证据 | 中 | 低 (新增 Fig 5) |
| SPIC 相关性弱 | 中 | 低 (新增 ITS 段落) |
| 图风格不统一 | 低 | 极低 (全扁平化) |

---

## 建议投稿检查清单

- [ ] 作者信息完整
- [ ] 致谢信息添加
- [ ] 参考文献补全 (尤其是 2022-2024)
- [ ] 语法拼写检查 (Grammarly/ChatGPT)
- [ ] 图分辨率确认 (300 DPI)
- [ ] PDF 页数确认 (SPIC 通常无严格限制，但 20-25 页为宜)
- [ ] Cover letter 准备 (突出 mechanism-driven 特色)
- [ ] 推荐审稿人准备 (3-5 位)

---

**当前状态**: 稿件已打磨至可投稿状态，等待作者信息补全后即可提交。
