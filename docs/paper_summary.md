# Paper Summary

The submitted paper studies why knowledge distillation behaves differently
across supervision branches when visibility degrades.

## Research Question

The central question is not whether KD can improve detection under fog. The
paper asks which type of teacher supervision becomes unreliable first and which
failure mechanism best explains the branch transitions.

## Experimental Logic

The analysis starts with a `5 x 3` matrix on Foggy Cityscapes:

- branches: student-only, logit, feature, attention, localization
- visibility levels: light, moderate, heavy
- primary metric: mAP@50

The observation matrix gives three constraints:

- logit transfer is not the dominant branch
- localization transfer remains comparatively robust before heavy degradation
- heavy degradation changes branch ordering

The paper then tests three candidate mechanisms:

- M1: distribution mismatch, measured by KL/JS divergence
- M2: occlusion-related spatial information loss, measured by the
  occlusion-localization relation
- M3: uncertainty amplification, measured by teacher entropy

## Main Finding

Among the tested mechanisms, M2 receives the strongest direct support. The
occlusion-localization relation is strongly negative (`r = -0.989`,
`p = 0.0015`), while M1 and M3 remain weakly related to their target branch
behaviors in the submitted setting.

## Validation Scope

The paper keeps the claim bounded. It does not argue that occlusion is the only
cause of degraded KD behavior, or that localization KD is universally best.
Instead, it argues that KD under visibility degradation is better understood as
a supervision-recoverability problem: teacher signals help only when the visual
evidence needed by that signal remains recoverable from the student's degraded
input.
