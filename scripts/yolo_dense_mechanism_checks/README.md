# YOLO Dense Mechanism Checks

This script group supports the paper's M1 and M3 mechanism diagnostics.

## Files

- `m13_dense_tasks.json`: task manifest for the dense degradation levels.
- `train_m13_dense.py`: YOLO training entry point with degradation and
  occlusion rendering helpers.
- `run_m13_dense_queue.py`: serial remote queue runner.
- `m13_dense_guard.py`: guard process that waits for the Faster R-CNN M2 queue
  before launching dense checks.
- `analyze_m13_dense.py`: computes KL/JS divergence, teacher entropy, branch
  gains, and Pearson summaries.

## Paper Outputs

The paper-facing outputs are stored in:

- `experiments/06_mechanism_dense_checks/m13_dense_metrics.csv`
- `experiments/06_mechanism_dense_checks/m13_dense_summary.json`
