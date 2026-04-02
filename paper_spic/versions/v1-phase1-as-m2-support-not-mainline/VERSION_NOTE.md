# v1-phase1-as-m2-support-not-mainline

## 核心定位
- 将后续 Phase 1 结果写成对 M2（occlusion / spatial information loss）的补充性控制验证。
- 不让 Phase 1 接管论文主线，主叙事仍然是 Observation -> Mechanism。

## 英文版改进
- 只做最小改动，优先保证英文主稿节奏与排版稳定。
- 在 abstract、intro、mechanism、discussion、conclusion 中补入一致口径：Phase 1 strengthens the M2 interpretation, but does not replace the main evidence chain.
- 避免把补强实验写成新的中心贡献。

## 中文版改进
- 将摘要、问题设定、观察、机制和讨论统一改为“Phase 1 支持 M2，但不接管主线”。
- 删除或弱化过重措辞，例如“遮挡悖论”“强正则化效应”。

## 留存文件
- main.pdf：英文版已编译 PDF
- main_cn.pdf：中文版已编译 PDF
- main.tex / main_cn.tex：主文件快照
- sections/：英文关键章节快照
- sections_cn/：中文关键章节快照

## 编译结果
- main.pdf：27 pages
- main_cn.pdf：17 pages
- 英文版仍有轻微 overfull hbox 警告，但没有结构性编译错误，版面可用。
- 中文版仍有若干 natbib 未定义引用警告，但不影响 PDF 产出。
