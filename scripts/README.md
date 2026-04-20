# Experiment Scripts

This directory contains runnable scripts grouped by their role in the paper.

## Layout

| Directory | Role |
|-----------|------|
| `yolo_dense_mechanism_checks/` | dense YOLO checks for M1 distribution mismatch and M3 uncertainty diagnostics |
| `faster_rcnn_boundary_check/` | Faster R-CNN true KD, M2 branch matrix, and M2-spatial-aware validation |

The scripts are remote-driver code. They expect the remote workspace root
`/root/kd_visibility`, dataset caches under `/root/autodl-tmp`, and output
folders that are not tracked in Git.

The paper-facing outputs from these runs are committed under `experiments/`.
