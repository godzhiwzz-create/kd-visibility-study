# Codex 接手文档

你（Codex）正在接替 Claude 推进 **跨天气 KD 动态门控 SOTA 方法（DADG）** 的实现。
按本文档从上到下执行；**不要**擅自改变方法方向、任务顺序或红线约束。

---

## 第一步：依次读完以下文件（按顺序，这是必读）

| # | 文件 | 为什么读 |
|---|------|---------|
| 1 | `~/.claude/plans/zesty-hopping-rocket.md` | 完整的 SOTA 方法 plan（已获用户批准），含方法、实验、ablation 设计 |
| 2 | `~/.claude/projects/-Users-godzhi-code---------kd-visibility/memory/MEMORY.md` | 记忆索引，看所有相关记忆条目 |
| 3 | `~/.claude/projects/.../memory/project_sota_dadg.md` | SOTA 论文定位、硬约束、未决技术选择 |
| 4 | `~/.claude/projects/.../memory/project_gate_collapse.md` | 为什么之前的 gate 全部失败；新设计的红线 |
| 5 | `~/.claude/projects/.../memory/project_dadg_progress.md` | 当前代码进度、阻塞点、服务端接手指南 |
| 6 | `~/.claude/projects/.../memory/feedback_research_scope.md` | 用户对研究范围的偏好，哪些方向已被排除 |
| 7 | `~/.claude/projects/.../memory/project_grant.md` | 科研资助标注要求（结题硬性） |
| 8 | `~/.claude/projects/.../memory/reference_server.md` | 服务端 SSH 和目录结构 |
| 9 | `kd_visibility/gate/TODO_DADG.md` | **执行清单，你的工作就是推进这个清单** |
| 10 | `kd_visibility/gate/models/dadg.py` | Gate 网络代码（已完成） |
| 11 | `kd_visibility/gate/losses/divergence.py` | 散度代码（已完成） |
| 12 | `kd_visibility/gate/losses/gate_loss.py` | Gate loss 代码（已完成） |
| 13 | `kd_visibility/gate/training/train_dadg.py` | **训练骨架，含 `# INTEGRATE:` 注释标注的接入点——你的第一个实现任务** |
| 14 | `kd_visibility/gate/configs/dadg.yaml` | 默认超参 |

读完这些你就有完整上下文。不要跳读。

---

## 第二步：确认当前进度

运行：
```bash
cat kd_visibility/gate/TODO_DADG.md
```

目前状态：
- Task #1–#7（本地代码骨架、语法检查）全部 **completed**（Claude 已做）
- Task #8（服务端 dryrun）**pending**，是你的下一个任务
- Task #9–#12 依赖 #8 通过

---

## 第三步：Task #8 的具体执行步骤

### 3.1 SSH 到服务端，检查 GPU 可用

```bash
ssh -p 39655 root@connect.bjb2.seetacloud.com
nvidia-smi   # 必须有卡才能继续，否则停下来告诉用户
```

### 3.2 同步代码

```bash
cd /root/kd_visibility
git pull origin master
# 本地 Claude 还未 push；如本地仓库有未提交内容，先在本地 commit+push 再到服务端 pull
```

### 3.3 实现 `train_dadg.py` 里的两个 INTEGRATE 函数

打开 `gate/training/train_dadg.py`，找到带 `# INTEGRATE:` 注释的两个 `NotImplementedError`：

1. **`build_teacher_student(cfg)`** — 返回 `(teacher, student, dataloader)`
   - Teacher：加载 `cfg["teacher"]["weights"]` 的 YOLOv8l，`eval()` + 冻结所有参数
   - Student：构建 `cfg["student"]["weights"]` 的 YOLOv8n
   - Dataloader：用服务端 `core.kd.data` 或现有的 YOLOv8 dataloader 构建，从 `cfg["data"]`
   - **参考**：服务端 `/root/kd_visibility/gate/experiments/3way_gate_fix_experiments/v1_exp1_stop_gradient_seed42/args.yaml` 里已有的 YOLOv8 训练参数

2. **`compute_kd_features(teacher, student, batch)`** — 返回字典，必须包含：
   - `student_feat`, `teacher_feat` — `(B, C, H, W)` backbone / neck 特征
   - `student_attn`, `teacher_attn` — `(B, H, W)` 或 `(B, 1, H, W)` 空间注意力
   - `student_boxes`, `teacher_boxes` — `(B, N, 4)` 匹配后的 bbox
   - `feature_kd_loss`, `attention_kd_loss`, `localization_kd_loss` — **标量**，每个 branch 单独的蒸馏 loss
   - `detection_loss` — 标量，标准 YOLOv8 检测 loss
   - `input_image` — `(B, 3, H, W)`，原图（gate 的输入）
   - **参考**：服务端之前 `3way_gate_fix_experiments` 使用的蒸馏流程里已有 feature/attention/localization 三分支 loss——把它们在聚合前的分量分别返回即可。找服务端 `core/kd/` 或 `low_visibility_kd/` 目录下的蒸馏实现。

### 3.4 运行 dryrun

```bash
cd /root/kd_visibility
python3 gate/training/train_dadg.py \
    --config gate/configs/dadg.yaml \
    --override experiment.name=dadg_dryrun data.fraction=0.05 student.epochs=1
```

### 3.5 Task #8 成功条件（必须全部满足）

1. 脚本无异常退出
2. `gate/experiments/dadg_main/dadg_dryrun/gate_trajectory.csv` 产生
3. CSV 里的 `w_feat` 列在训练过程中 **不会低于 entropy_floor（0.05）** —— 证明不坍缩
4. `gate/total_loss` 数值整体呈下降趋势

---

## 🔴 永不违反的红线

1. **Gate 的梯度只来自 `L_gate`，永远不能来自 KD loss 或 detection loss。**
   代码里 `gate_w_for_kd = gate(img).detach()` 这一行是防坍缩的核心，**不要改**。
   唯一例外是 Task #9（坍缩对照实验），设置 `cfg.dadg.stop_gradient_from_kd=false` 触发。

2. **推理时不允许输入 β、退化标签或任何额外参数。** Gate 只吃雾图。

3. **不引入新架构**（YOLOv9 / RT-DETR / DETR），不引入 β 估计模块，不扩展到低照度/运动模糊等新机制。那些是 out-of-scope。

4. **不改动 `TODO_DADG.md` 的任务结构**。你只能更新 Status 列。遇到任何需要增减任务、改变方法、调整范围的情况，**停下来，写到 `TODO_DADG.md` 的"Codex 反馈"段**，等 Claude 下次上线裁决。

5. **遇到 GPU 不可用、依赖缺失、数据集路径不对等环境问题，不要自行替代或 mock**。在"Codex 反馈"里写明阻塞原因，停下来。

---

## 每完成一个任务必须做的事

1. 把 `TODO_DADG.md` 里对应行的 Status 改为 `completed`（或 `blocked`）
2. 关键产物路径填到"备注"列
3. `git add <相关文件>` → `git commit -m "DADG task #N: <subject> — completed"`
4. 如果修改了服务端代码但本地没有，在 commit message 里注明 `(server-side only)`
5. 阻塞时不要猜——停下来在 `TODO_DADG.md` 的"Codex 反馈"段写明问题

---

## 出现以下任何情况，立即停下来等 Claude

- 用户明确要求停止
- 方法方向有疑问（比如"这样改会不会更好"——**不要擅自改**，写反馈）
- 实验结果异常（loss 爆炸、显存不够、精度远低于预期）
- 需要引入新依赖或修改 `core/`（共享层，改动要慎重）
- `TODO_DADG.md` 里没列的工作

---

## 快速检索

- 方法详情 → `~/.claude/plans/zesty-hopping-rocket.md`
- 坍缩原因 → `project_gate_collapse.md`
- 当前进度 → `project_dadg_progress.md` + `TODO_DADG.md`
- 服务端信息 → `reference_server.md`
- 代码骨架 → `kd_visibility/gate/{models,losses,training,configs}/`
