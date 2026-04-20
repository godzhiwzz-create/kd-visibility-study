#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from scipy import stats
from torchvision import transforms
from ultralytics import YOLO

WORK_ROOT = Path('/root/kd_visibility')
TASK_ROOT = WORK_ROOT / 'scripts/yolo_dense_mechanism_checks'
OUT_ROOT = WORK_ROOT / 'outputs_m13_dense_runs'
EFFECTIVE_ROOT = WORK_ROOT / 'outputs_causal_effective_runs'
CACHE_ROOT = Path('/root/autodl-tmp/kd_visibility_effective_cache/datasets')
TEACHER = Path('/root/autodl-tmp/visibility_ablation/low_visibility_kd/experiments/v1_teacher_clear_yolov8l/weights/best.pt')
DEGS = [0.0, 0.3, 0.5, 0.7, 1.0]
SEED = 42
OCC = 0.0
TR = transforms.Compose([transforms.Resize((640, 640)), transforms.ToTensor()])


def exp_name(branch: str, deg: float) -> str:
    return f'occ_{OCC}_{branch}_deg_{float(deg)}_seed_{SEED}'


def load_map50(result_path: Path) -> float:
    data = json.loads(result_path.read_text())
    if 'metrics' in data and 'map50' in data['metrics']:
        return float(data['metrics']['map50'])
    if 'mAP50' in data:
        return float(data['mAP50'])
    raise KeyError(f'mAP50 not found in {result_path}')


def load_weights(result_path: Path) -> Path:
    data = json.loads(result_path.read_text())
    if 'best_weights' in data:
        return Path(data['best_weights'])
    if 'weights_path' in data:
        return Path(data['weights_path'])
    raise KeyError(f'weights path not found in {result_path}')


def val_images_for_deg(deg: float) -> list[Path]:
    data_yaml = CACHE_ROOT / f'deg_{deg}_occ_{OCC}_seed_{SEED}' / 'data.yaml'
    data = yaml.safe_load(data_yaml.read_text())
    root = Path(data['path'])
    return sorted((root / 'val' / 'images').glob('*.png'))


def cls_probs(model: YOLO, img_path: Path) -> torch.Tensor:
    img = Image.open(img_path).convert('RGB')
    x = TR(img).unsqueeze(0)
    device = next(model.model.parameters()).device
    x = x.to(device)
    with torch.no_grad():
        out = model.model(x)
    pred = out[0] if isinstance(out, (tuple, list)) else out
    cls_logits = pred[:, 4:, :]
    probs = torch.softmax(cls_logits, dim=1)
    return probs.squeeze(0).permute(1, 0).cpu()  # [anchors, classes]


def mean_kl_js(student_model: YOLO, teacher_model: YOLO, images: list[Path]) -> tuple[float, float, float]:
    kls, jss, ents = [], [], []
    eps = 1e-8
    sample = images[:50]
    for img in sample:
        tp = cls_probs(teacher_model, img)
        sp = cls_probs(student_model, img)
        m = 0.5 * (tp + sp)
        kl = (tp * (tp.add(eps).log() - sp.add(eps).log())).sum(dim=1).mean().item()
        js = 0.5 * (tp * (tp.add(eps).log() - m.add(eps).log())).sum(dim=1).mean().item() + \
             0.5 * (sp * (sp.add(eps).log() - m.add(eps).log())).sum(dim=1).mean().item()
        ent = (-(tp * tp.add(eps).log()).sum(dim=1)).mean().item()
        kls.append(kl)
        jss.append(js)
        ents.append(ent)
    return float(np.mean(kls)), float(np.mean(jss)), float(np.mean(ents))


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    teacher = YOLO(str(TEACHER))
    teacher.model.to(device).eval()

    rows = []
    student_map = {}
    logit_map = {}
    attn_map = {}

    for deg in DEGS:
        student_map[deg] = load_map50(OUT_ROOT / exp_name('student_only', deg) / 'results.json')
        logit_map[deg] = load_map50(EFFECTIVE_ROOT / exp_name('logit_only', deg) / 'results.json')
        attn_map[deg] = load_map50(OUT_ROOT / exp_name('attention_only', deg) / 'results.json')

    for deg in DEGS:
        images = val_images_for_deg(deg)
        logit_result = EFFECTIVE_ROOT / exp_name('logit_only', deg) / 'results.json'
        attn_result = OUT_ROOT / exp_name('attention_only', deg) / 'results.json'
        logit_model = YOLO(str(load_weights(logit_result)))
        attn_model = YOLO(str(load_weights(attn_result)))
        logit_model.model.to(device).eval()
        attn_model.model.to(device).eval()
        kl, js, teacher_entropy = mean_kl_js(logit_model, teacher, images)
        rows.append({
            'degradation_level': deg,
            'student_map50': student_map[deg],
            'logit_map50': logit_map[deg],
            'attention_map50': attn_map[deg],
            'logit_gain': logit_map[deg] - student_map[deg],
            'attention_gain': attn_map[deg] - student_map[deg],
            'kl_divergence': kl,
            'js_divergence': js,
            'teacher_entropy': teacher_entropy,
        })

    out_csv = TASK_ROOT / 'm13_dense_metrics.csv'
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    m1_x = [r['kl_divergence'] for r in rows]
    m1_y = [r['logit_gain'] * 100 for r in rows]
    m3_x = [r['teacher_entropy'] for r in rows]
    m3_y = [r['attention_gain'] * 100 for r in rows]
    summary = {
        'm1': {'r': float(stats.pearsonr(m1_x, m1_y)[0]), 'p': float(stats.pearsonr(m1_x, m1_y)[1]), 'x': m1_x, 'y': m1_y},
        'm3': {'r': float(stats.pearsonr(m3_x, m3_y)[0]), 'p': float(stats.pearsonr(m3_x, m3_y)[1]), 'x': m3_x, 'y': m3_y},
    }
    (TASK_ROOT / 'm13_dense_summary.json').write_text(json.dumps(summary, indent=2) + '\n')

if __name__ == '__main__':
    main()
