# DADG 实施任务清单（Claude ↔ Codex 共用）

> **权限规则**：只有 Claude（本地 plan 持有者）可以修改本文件的任务结构/范围。
> Codex 接手时**严格按此清单顺序执行**，只更新 `Status` 列，不新增/删除/改写任务。
> 遇到偏离清单的情况（bug、需要调整方法、发现新需求），停下来写到 `## Codex 反馈` 段落里，等 Claude 下次上线审阅。

参考：
- 完整 Plan: `~/.claude/plans/zesty-hopping-rocket.md`
- 方法方向: `~/.claude/projects/-Users-godzhi-code---------kd-visibility/memory/project_sota_dadg.md`
- 坍缩诊断: `~/.claude/projects/-Users-godzhi-code---------kd-visibility/memory/project_gate_collapse.md`

---

## 任务列表

| # | 任务 | Status | Owner | 备注 |
|---|------|--------|-------|------|
| 1 | 创建 gate/ 目录骨架（models/ losses/ training/ configs/ experiments/） | **completed** | Claude | 本地已建 |
| 2 | 实现 `gate/models/dadg.py`（轻量 CNN + softmax head） | **completed** | Claude | `gate/models/dadg.py`，~0.7M 参数 |
| 3 | 实现 `gate/losses/divergence.py`（feature cos-dist / attn KL / loc IoU gap） | **completed** | Claude | `gate/losses/divergence.py` |
| 4 | 实现 `gate/losses/gate_loss.py`（`KL(G ‖ softmax(-d/τ))` + entropy floor） | **completed** | Claude | `gate/losses/gate_loss.py` |
| 5 | 写 `gate/configs/dadg.yaml` 默认超参 | **completed** | Claude | τ=1.0, floor=0.05, gate_lr=1e-3 |
| 6 | 实现 `gate/training/train_dadg.py` 训练脚本骨架 | **completed** | Claude | 含 `# INTEGRATE:` 标记的服务端对接点 |
| 7 | 本地 `python3 -m py_compile` 语法检查 | **completed** | Claude | 4 个文件全部通过 |
| 8 | 服务端 dryrun（1 epoch, fraction=0.05, 1 seed） | pending | - | **要 GPU**；需先完成 INTEGRATE |
| 9 | 坍缩对照实验（无 stop_gradient） | pending | - | 论文 ablation 必需 |
| 10 | 主实验（5 seeds × Cityscapes-Foggy, 20 epochs） | pending | - | guardian/queue 模式 |
| 11 | 外部评估（beta_0.005/0.01/0.02 + RTTS + Foggy Driving） | pending | - | 对比全部 baselines |
| 12 | Ablation 套件（6 组：监督信号/τ/floor/辅助输入/容量/去除 sg） | pending | - | Task #9 已覆盖"去除 sg" |

### 服务端接手指南（Codex / 未来会话必读）

**Task #8 之前必须完成的集成工作**（两个函数，在 `gate/training/train_dadg.py` 顶部 `# INTEGRATE:` 注释里）：

1. `build_teacher_student(cfg)` — 接入服务端 `core.kd` 模块：
   - 加载冻结的 YOLOv8l teacher
   - 构建 YOLOv8n student + dataloader
   - 返回 `(teacher, student, loader)`

2. `compute_kd_features(teacher, student, batch)` — 前向 + 返回字典，包含：
   - `student_feat`, `teacher_feat`（neck 特征，(B,C,H,W)）
   - `student_attn`, `teacher_attn`（(B,H,W) 或 (B,1,H,W)）
   - `student_boxes`, `teacher_boxes`（(B,N,4) matched）
   - `feature_kd_loss`, `attention_kd_loss`, `localization_kd_loss`, `detection_loss`（标量）
   - `input_image`（(B,3,H,W)）

   服务端已有的 branch-level 蒸馏 loss 函数（在之前 `3way_gate_fix_experiments` 用过的）可以直接复用，把聚合前的单 branch 分量返回出来即可。

3. 之后即可按 config 跑：
   ```bash
   cd /root/kd_visibility
   python3 gate/training/train_dadg.py --config gate/configs/dadg.yaml \
     --override experiment.name=dadg_dryrun data.fraction=0.05 student.epochs=1
   ```
   验证 `gate_trajectory.csv` 显示 `w_feat` 不压到 0（任务 #8 的成功条件）。

---

## 关键硬约束（执行时必须遵守）

1. **Gate 的梯度只来自 L_gate**，KD loss 到 gate 的路径必须 `stop_gradient`。违反此项即复现坍缩（任务 #9 是故意违反作为对照，任务 #10 不得违反）。
2. **推理时不依赖任何外部参数**（无 β 输入）。Gate 接收雾图，输出权重。
3. 不要引入新 student 架构、不要引入 β 估计模块、不要扩展其他机制轴（低照度/运动模糊等留为后续工作）。

## 未决技术选择（Codex 实现时若需决定，填默认值并标注）

- Divergence metric 具体归一化方式（建议：各 metric 先 per-batch min-max，再 stack 成 (B,3)）
- 两套 optimizer 的学习率（gate=1e-3, student 走 YOLOv8 default）
- Gate backbone：5 层 conv（C=32,64,128,128,256）+ GAP + 2 层 MLP(256→128→3)

## Codex 反馈（遇到问题写在这里）

（空）

---

## 进度同步协议

每完成一项任务，Codex 必须：
1. 把对应行 `Status` 改为 `completed` 或 `blocked`
2. 如果 blocked，写原因到"Codex 反馈"段
3. 关键产物的相对路径写进"备注"列
4. 提交一次 git commit，commit message 格式：`DADG task #N: <subject> — <status>`
