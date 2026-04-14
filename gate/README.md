# Gate Layer: SOTA Cross-Weather KD Method

This layer contains the active research toward a state-of-the-art
cross-weather knowledge distillation method for object detection.

## Research Direction

Under construction. Goal: design and implement a KD training paradigm that
generalizes robustly across weather conditions (fog, rain, haze, low-light),
building on findings from the mechanism layer.

## Relationship to Mechanism Layer

The mechanism layer (`../mechanism/`) established how KD branches degrade
under low-visibility conditions. The gate layer will use those findings to
design an adaptive KD strategy that remains effective across weather domains.

## Shared Utilities

Import from `core/` for data processing and evaluation:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core.degradation import apply_global_degradation, apply_bbox_occlusion
from core.data_utils import level_to_source, CLASS_NAMES
from core.eval_utils import set_seed
```
