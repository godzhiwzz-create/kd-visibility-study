# Cross-Architecture Faster R-CNN Validation

This folder stores the Faster R-CNN cross-architecture validation results synchronized from the remote server.

## Representative results
- Student-only (occ=0.2, degradation level=0.5): mAP@50 = 0.6258, mAP@50-95 = 0.3755
- Logit KD (occ=0.2, degradation level=0.5): mAP@50 = 0.6356, mAP@50-95 = 0.3853
- Localization-only (occ=0.2, degradation level=0.5): mAP@50 = 0.6299, mAP@50-95 = 0.3691
- M2-spatial-aware KD (occ=0.2, degradation level=0.5): mAP@50 = 0.6483, mAP@50-95 = 0.3901

## Interpretation
- Plain branch differences remain visible under Faster R-CNN, with logit KD outperforming the plain localization branch.
- A detector-matched M2-spatial-aware variant performs best at the same representative degraded setting.
- The result is therefore most useful as cautious evidence that mechanism-guided design may transfer when mapped to the detector's supervision structure, rather than as a claim of universal branch ordering.
