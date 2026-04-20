# Representative True KD Validation

This folder stores the final representative YOLO true teacher-student validation used in the paper.

## Results used in the manuscript
- Student-only: `mAP@50 = 0.5406`, `mAP@50-95 = 0.3374`
- Logit KD: `mAP@50 = 0.5468`, `mAP@50-95 = 0.3398`
- Occlusion-aware KD: `mAP@50 = 0.5495`, `mAP@50-95 = 0.3462`

Interpretation in the paper:
- Plain logit KD recovers a small gain over student-only.
- A mechanism-guided occlusion-aware variant yields the best result at the representative degraded setting.
- This result is used as validation evidence, not as a stand-alone method claim.
