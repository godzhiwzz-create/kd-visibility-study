# KD Visibility Study

Public companion workspace for the IEEE Access submission:

**Branch-Wise Mechanism Analysis of Knowledge Distillation for Object Detection
Under Visibility Degradation**

This repository is organized around the paper's evidence chain. It keeps the
release-safe experiment summaries and analysis scripts, while excluding local
datasets, model checkpoints, manuscript source/PDF files, reference-paper PDFs,
and temporary run outputs.

## Repository Layout

```text
kd_visibility/
├── paper/          # Manuscript availability note only
├── experiments/    # Paper-cited result tables and verification summaries
├── scripts/        # Remote-run scripts for dense YOLO and Faster R-CNN checks
├── docs/           # Paper-to-repository guide and reproduction notes
└── requirements.txt
```

## Paper Evidence Chain

| Stage | Paper role | Repository location |
|-------|------------|---------------------|
| S1 observation | `5 x 3` branch-wise KD matrix on Foggy Cityscapes | `experiments/01_observation_main_matrix/` |
| S2 mechanism test | M1 distribution mismatch, M2 occlusion-related spatial information loss, M3 uncertainty amplification | `experiments/06_mechanism_dense_checks/` |
| S3 reinforcement | dense degradation checkpoints used to check M2 stability | `experiments/03_phase2_dense_trend_validation/` |
| S4 design consequence | representative true teacher-student validation | `experiments/04_representative_true_kd_yolo/` |
| S5 boundary check | Faster R-CNN cross-architecture validation | `experiments/05_cross_architecture_faster_rcnn/` |

## Key Submitted Findings

- The main matrix shows branch-wise, non-uniform KD behavior under visibility
  degradation. Localization-only transfer is strongest under light and moderate
  fog, while heavy degradation changes the branch ordering.
- Logit KD is measurable but not dominant in the observation stage.
- Among the tested mechanisms, M2 receives the strongest direct support:
  occlusion ratio and localization performance have a strong negative relation
  (`r = -0.989`, `p = 0.0015`).
- M1 and M3 are plausible background factors but are insufficient as dominant
  explanations in the submitted setting.
- Representative validation gives the best reported YOLO result to the
  occlusion-aware KD variant (`0.5495` mAP@50, `0.3462` mAP@50--95).
- The Faster R-CNN boundary check supports the mechanism perspective but keeps
  the claim bounded: branch ranking is architecture dependent.

## What Is Not Included

- Foggy Cityscapes / Cityscapes images and labels.
- Teacher/student checkpoints and runtime output directories.
- IEEE Access manuscript source, compiled PDF, cover letter, author photos, and
  submission archive.
- Draft notes, duplicate manuscript folders, and reference-paper PDFs.

See `docs/paper_summary.md` for the paper-to-repository reading guide and
`docs/reproduction_notes.md` for runtime assumptions.
