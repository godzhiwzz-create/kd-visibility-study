# Faster R-CNN Boundary Check

This script group supports the paper's cross-architecture boundary check. It
tests whether the M2 spatial-reliability interpretation remains useful when the
idea is mapped from YOLO to a proposal-based detector.

## Files

- `train_faster_rcnn_true_kd.py`: training and evaluation entry point.
- `run_faster_rcnn_true_fix_queue.py`: corrected true-KD queue used after the
  label-path fix.
- `run_faster_rcnn_m2_queue.py`: branch matrix queue for the M2 boundary check.
- `run_faster_rcnn_occaware_queue.py`: M2-spatial-aware validation queue.
- `summarize_phase4_m2.py`: writes a compact CSV summary after the M2 queue.
- `*_guard.py`: remote watchdogs for long-running queues.

## Paper Output

The paper-facing summary is stored in:

- `experiments/05_cross_architecture_faster_rcnn/faster_rcnn_results.csv`

The submitted representative result reports M2-spatial-aware KD as best in the
Faster R-CNN boundary setting, while also noting that plain branch ranking is
architecture dependent.
