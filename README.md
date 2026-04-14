# KD Visibility Study

Public workspace for research on knowledge distillation under low-visibility
and cross-weather conditions. The project follows a three-layer architecture.

## Structure

```
kd_visibility/
├── core/           # Shared utilities (degradation, data loading, evaluation)
├── mechanism/      # Mechanism study — how KD degrades under low visibility
│   ├── m13_dense_check/        # Dense degradation-level sweep (YOLOv8)
│   └── faster_rcnn_kd_true/    # Cross-architecture validation (Faster R-CNN)
├── gate/           # SOTA cross-weather KD method (active development)
└── paper_spic/     # Manuscript availability note (under journal review)
```

## Layer Summary

| Layer | Status | Description |
|-------|--------|-------------|
| `core/` | Stable | Degradation simulation, Cityscapes data utilities, eval helpers |
| `mechanism/` | Complete | Phase 4 experiments; findings submitted to SPIC |
| `gate/` | Active | Designing a robust cross-weather KD training paradigm |

## Notes

The manuscript source and PDF are not tracked here during journal review.
Validation scripts assume the remote dataset and output paths documented in
their local README files. Legacy code and temporary outputs are omitted.
