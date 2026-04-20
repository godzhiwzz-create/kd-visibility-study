# Paper-Used Experiments

This directory contains the result files that are directly used by the IEEE
Access submission. These files are summaries or verification artifacts, not the
full remote run cache.

## Directory Map

| Directory | Paper role | Main files |
|-----------|------------|------------|
| `01_observation_main_matrix/` | Main branch-wise observation matrix used for the structured phenomenon | `figure2_branch_performance.csv`, `figure3_gain_relative.csv` |
| `03_phase2_dense_trend_validation/` | Dense trend checkpoints used to reinforce M2 | `phase2_key_checkpoints.md` |
| `04_representative_true_kd_yolo/` | True teacher-student validation in YOLO | `true_kd_results.csv` |
| `05_cross_architecture_faster_rcnn/` | Faster R-CNN boundary check | `faster_rcnn_results.csv`, `raw/*.json` |
| `06_mechanism_dense_checks/` | M1/M3 diagnostics and M2 occlusion evidence | `m13_dense_metrics.csv`, `m13_dense_summary.json`, `m2_occlusion_loc.json` |

## Interpretation Rule

The paper treats these files as a single evidence chain. Do not read any one
table as a stand-alone method benchmark. The intended reading is:

1. The main matrix establishes branch-structured KD degradation.
2. Mechanism tests compare whether M1, M2, or M3 explains the same branch-level
   constraints.
3. Dense checks and validation runs test whether the supported M2 reading is
   stable enough to guide a bounded design implication.
