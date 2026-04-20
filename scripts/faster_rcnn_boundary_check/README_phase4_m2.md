# Faster R-CNN Phase 4 M2 Cross-Architecture Plan

This directory contains the execution chain for a focused cross-architecture M2 check on Faster R-CNN.

## Goal
- Test whether branch-dependent KD behavior changes systematically as occlusion increases under a fixed degraded setting.
- Keep the scope narrow: this is a `3 occlusion ratios x 3 branches` matrix at `degradation_level = 0.5`.

## Matrix
- Occlusion ratios: `0.0`, `0.2`, `0.4`
- Branches:
  - `student_only`
  - `logit_only`
  - `localization_only`
- Seed: `42`

## Files
- `phase4_m2_tasks.json`: task manifest
- `run_faster_rcnn_m2_queue.py`: serial queue runner
- `faster_rcnn_m2_guard.py`: auto-restart guard
- `summarize_phase4_m2.py`: writes `phase4_m2_results.csv` after completion

## Server-side launch
```bash
nohup /root/miniconda3/bin/python3 /root/kd_visibility/scripts/faster_rcnn_boundary_check/faster_rcnn_m2_guard.py \
  > /root/kd_visibility/logs/faster_rcnn_phase4_m2_guard.out 2>&1 &
```

## Outputs
- Run directory: `/root/kd_visibility/outputs_faster_rcnn_true_phase4_m2/`
- State file: `/root/kd_visibility/scripts/faster_rcnn_boundary_check/phase4_m2_state.json`
- Summary CSV: `/root/kd_visibility/scripts/faster_rcnn_boundary_check/phase4_m2_results.csv`
