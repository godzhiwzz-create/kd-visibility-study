#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

WORK_ROOT = Path('/root/kd_visibility')
OUTPUT_DIR = WORK_ROOT / 'outputs_faster_rcnn_true_phase4_m2'
SUMMARY_CSV = WORK_ROOT / 'faster_rcnn_kd_true' / 'phase4_m2_results.csv'


def main() -> None:
    rows = []
    for results_path in sorted(OUTPUT_DIR.glob('*/results.json')):
        data = json.loads(results_path.read_text())
        rows.append(
            {
                'setting': results_path.parent.name,
                'occlusion_ratio': data.get('occlusion_ratio'),
                'degradation_level': data.get('degradation_level'),
                'kd_branch': data.get('kd_branch'),
                'mAP50': round(float(data.get('mAP50', 0.0)), 4),
                'mAP50_95': round(float(data.get('mAP50_95', 0.0)), 4),
                'best_epoch': data.get('best_epoch'),
                'kd_weight': data.get('kd_weight'),
                'temperature': data.get('temperature'),
            }
        )
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_CSV.open('w', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'setting',
                'occlusion_ratio',
                'degradation_level',
                'kd_branch',
                'mAP50',
                'mAP50_95',
                'best_epoch',
                'kd_weight',
                'temperature',
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f'wrote {SUMMARY_CSV}')


if __name__ == '__main__':
    main()
