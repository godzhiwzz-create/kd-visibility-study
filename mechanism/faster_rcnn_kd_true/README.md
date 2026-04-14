# Faster R-CNN True KD Validation

This directory contains the active cross-architecture validation code.

## Status
- Original phase4 runs were invalid because labels were not loaded correctly.
- The dataset label-path bug has been fixed in `train_faster_rcnn_true_kd.py`.
- A remote guarded rerun is used to regenerate valid results.

## Main files
- `train_faster_rcnn_true_kd.py`: actual training and evaluation script
- `run_faster_rcnn_true_fix_queue.py`: fix rerun queue
- `faster_rcnn_true_fix_guard.py`: remote guard for the fix rerun
