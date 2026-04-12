#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
import yaml
from PIL import Image, ImageFile
from ultralytics import YOLO

WORK_ROOT = Path('/root/kd_visibility')
DATA_ROOT = Path('/root/autodl-tmp/shared_datasets/low_visibility_kd/cityscapes_yolo')
CLEAR_ROOT = DATA_ROOT / 'clear'
FOGGY_ROOT = DATA_ROOT / 'foggy_all'
CACHE_ROOT = Path('/root/autodl-tmp/kd_visibility_effective_cache/datasets')
DEFAULT_PREP_WORKERS = min(24, max(1, (os.cpu_count() or 8) - 1))
CLASS_NAMES = {
    0: 'person', 1: 'rider', 2: 'car', 3: 'truck',
    4: 'bus', 5: 'train', 6: 'motorcycle', 7: 'bicycle'
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stable_int(*parts: object) -> int:
    text = '::'.join(map(str, parts)).encode('utf-8')
    return int(hashlib.md5(text).hexdigest()[:8], 16)


def level_to_source(degradation_level: float) -> tuple[str, float | None, float, float]:
    level = float(degradation_level)
    if level <= 0.0:
        return 'clear', None, 1.0, 0.0
    if level <= 0.3:
        return 'foggy', 0.005, 1.0, 0.0
    if level <= 0.5:
        return 'foggy', 0.01, 1.0, 0.0
    if level <= 0.7:
        return 'foggy', 0.02, 0.95, 8.0
    return 'foggy', 0.02, 0.88, 18.0


def branch_hypers(kd_branch: str) -> dict[str, float]:
    hyp = {
        'lr0': 0.005,
        'lrf': 0.001,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'box': 7.5,
        'cls': 0.8,
        'dfl': 1.5,
        'hsv_h': 0.02,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 3.0,
        'translate': 0.15,
        'scale': 0.5,
        'fliplr': 0.5,
        'mosaic': 0.5,
        'mixup': 0.1,
        'erasing': 0.1,
    }
    if kd_branch == 'logit_only':
        hyp.update({'lr0': 0.004, 'box': 6.5, 'cls': 1.0, 'mixup': 0.2, 'erasing': 0.15})
    elif kd_branch == 'localization_only':
        hyp.update({'lr0': 0.004, 'box': 8.5, 'dfl': 2.0, 'mixup': 0.05, 'erasing': 0.05})
    elif kd_branch == 'attention_only':
        hyp.update({'lr0': 0.003, 'box': 6.5, 'cls': 0.9, 'mixup': 0.15, 'erasing': 0.12, 'translate': 0.18})
    elif kd_branch == 'feature_only':
        hyp.update({'lr0': 0.003, 'box': 6.0, 'cls': 0.9, 'mixup': 0.12, 'erasing': 0.10})
    elif kd_branch != 'student_only':
        raise ValueError(f'Unsupported kd_branch: {kd_branch}')
    return hyp


def find_image_and_label(split: str, stem: str, degradation_level: float) -> tuple[Path, Path]:
    source_type, beta, _, _ = level_to_source(degradation_level)
    if source_type == 'clear':
        img = CLEAR_ROOT / 'images' / split / f'{stem}_leftImg8bit.png'
        lbl = CLEAR_ROOT / 'labels' / split / f'{stem}_leftImg8bit.txt'
    else:
        beta_str = f'{beta:.3f}'.rstrip('0').rstrip('.')
        img = FOGGY_ROOT / 'images' / split / f'{stem}_leftImg8bit_foggy_beta_{beta_str}.png'
        lbl = FOGGY_ROOT / 'labels' / split / f'{stem}_leftImg8bit_foggy_beta_{beta_str}.txt'
        if not lbl.exists():
            lbl = CLEAR_ROOT / 'labels' / split / f'{stem}_leftImg8bit.txt'
    return img, lbl


def load_yolo_labels(label_path: Path, width: int, height: int) -> list[tuple[int, int, int, int, int]]:
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls_id = int(float(parts[0]))
        xc, yc, bw, bh = map(float, parts[1:])
        x1 = max(0, int((xc - bw / 2) * width))
        y1 = max(0, int((yc - bh / 2) * height))
        x2 = min(width, int((xc + bw / 2) * width))
        y2 = min(height, int((yc + bh / 2) * height))
        if x2 > x1 and y2 > y1:
            boxes.append((cls_id, x1, y1, x2, y2))
    return boxes


def apply_global_degradation(image: np.ndarray, contrast: float, haze_bias: float) -> np.ndarray:
    out = image.astype(np.float32) * contrast + haze_bias
    return np.clip(out, 0, 255).astype(np.uint8)


def read_image_resilient(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is not None:
        return image
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    with Image.open(path) as pil_img:
        pil_img.load()
        rgb = pil_img.convert('RGB')
    return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)


def apply_bbox_occlusion(image: np.ndarray, boxes: Iterable[tuple[int, int, int, int, int]], occlusion_ratio: float, rng: random.Random) -> np.ndarray:
    if occlusion_ratio <= 0:
        return image
    out = image.copy()
    for _, x1, y1, x2, y2 in boxes:
        bw, bh = x2 - x1, y2 - y1
        if bw * bh < 100:
            continue
        occ_w = max(1, int(bw * np.sqrt(occlusion_ratio)))
        occ_h = max(1, int(bh * np.sqrt(occlusion_ratio)))
        max_x = max(0, bw - occ_w)
        max_y = max(0, bh - occ_h)
        ox = x1 + (rng.randint(0, max_x) if max_x > 0 else 0)
        oy = y1 + (rng.randint(0, max_y) if max_y > 0 else 0)
        fill = int(rng.uniform(96, 160))
        out[oy:min(y2, oy + occ_h), ox:min(x2, ox + occ_w)] = fill
    return out


def ensure_symlink(target: Path, link_path: Path) -> None:
    if link_path.is_symlink() and link_path.resolve() == target.resolve():
        return
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    os.symlink(target, link_path)


def prepare_sample(job: dict) -> int:
    cv2.setNumThreads(1)
    split = job['split']
    stem = job['stem']
    degradation_level = job['degradation_level']
    occlusion_ratio = job['occlusion_ratio']
    seed = job['seed']
    contrast = job['contrast']
    haze_bias = job['haze_bias']
    src_img = Path(job['src_img'])
    src_lbl = Path(job['src_lbl'])
    dst_img = Path(job['dst_img'])
    dst_lbl = Path(job['dst_lbl'])
    needs_render = bool(job['needs_render'])

    if dst_img.exists() and dst_lbl.exists():
        return 1

    dst_img.parent.mkdir(parents=True, exist_ok=True)
    dst_lbl.parent.mkdir(parents=True, exist_ok=True)

    if needs_render:
        image = read_image_resilient(src_img)
        image = apply_global_degradation(image, contrast, haze_bias)
        boxes = load_yolo_labels(src_lbl, image.shape[1], image.shape[0])
        rng = random.Random(stable_int(split, stem, degradation_level, occlusion_ratio, seed))
        image = apply_bbox_occlusion(image, boxes, occlusion_ratio, rng)
        ok = cv2.imwrite(str(dst_img), image)
        if not ok:
            raise RuntimeError(f'Failed to write image: {dst_img}')
    else:
        ensure_symlink(src_img, dst_img)

    if not dst_lbl.exists():
        shutil.copy2(src_lbl, dst_lbl)
    return 1


def build_dataset(degradation_level: float, occlusion_ratio: float, seed: int, prep_workers: int) -> Path:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    cache_key = f'deg_{degradation_level}_occ_{occlusion_ratio}_seed_{seed}'
    ds_root = CACHE_ROOT / cache_key
    for split in ['train', 'val']:
        (ds_root / split / 'images').mkdir(parents=True, exist_ok=True)
        (ds_root / split / 'labels').mkdir(parents=True, exist_ok=True)

    split_dirs = {'train': CLEAR_ROOT / 'images' / 'train', 'val': CLEAR_ROOT / 'images' / 'val'}
    _, _, contrast, haze_bias = level_to_source(degradation_level)
    needs_render = occlusion_ratio > 0 or contrast != 1.0 or haze_bias != 0.0

    jobs = []
    for split, ref_dir in split_dirs.items():
        for ref_img in sorted(ref_dir.glob('*_leftImg8bit.png')):
            stem = ref_img.name.replace('_leftImg8bit.png', '')
            src_img, src_lbl = find_image_and_label(split, stem, degradation_level)
            if not src_img.exists() or not src_lbl.exists():
                continue
            jobs.append({
                'split': split,
                'stem': stem,
                'degradation_level': degradation_level,
                'occlusion_ratio': occlusion_ratio,
                'seed': seed,
                'contrast': contrast,
                'haze_bias': haze_bias,
                'src_img': str(src_img),
                'src_lbl': str(src_lbl),
                'dst_img': str(ds_root / split / 'images' / src_img.name),
                'dst_lbl': str(ds_root / split / 'labels' / (src_img.stem + '.txt')),
                'needs_render': needs_render,
            })

    if prep_workers <= 1 or len(jobs) < 32:
        for job in jobs:
            prepare_sample(job)
    else:
        with ProcessPoolExecutor(max_workers=prep_workers) as ex:
            for _ in ex.map(prepare_sample, jobs, chunksize=16):
                pass

    data_yaml = {
        'path': str(ds_root),
        'train': 'train/images',
        'val': 'val/images',
        'names': CLASS_NAMES,
        'nc': len(CLASS_NAMES),
    }
    yaml_path = ds_root / 'data.yaml'
    yaml_path.write_text(yaml.safe_dump(data_yaml, sort_keys=False))
    return yaml_path


def evaluate_model(model_path: Path, data_yaml: Path, device: str) -> dict[str, float]:
    model = YOLO(str(model_path))
    results = model.val(data=str(data_yaml), split='val', device=device, verbose=False, plots=False)
    return {
        'map50': float(results.box.map50),
        'map75': float(results.box.map75),
        'map': float(results.box.map),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--occlusion_ratio', type=float, required=True)
    parser.add_argument('--beta', type=float, required=True)
    parser.add_argument('--kd_branch', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--prep_workers', type=int, default=DEFAULT_PREP_WORKERS)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='outputs_m13_dense_runs')
    args = parser.parse_args()

    set_seed(args.seed)
    device = '0' if torch.cuda.is_available() else 'cpu'
    data_yaml = build_dataset(args.beta, args.occlusion_ratio, args.seed, args.prep_workers)

    exp_name = f"occ_{args.occlusion_ratio}_{args.kd_branch}_deg_{float(args.beta)}_seed_{args.seed}"
    exp_root = WORK_ROOT / args.output_dir / exp_name
    exp_root.mkdir(parents=True, exist_ok=True)

    hyp = branch_hypers(args.kd_branch)
    train_args = {
        'data': str(data_yaml),
        'epochs': args.epochs,
        'imgsz': args.imgsz,
        'batch': args.batch_size,
        'device': device,
        'project': str(exp_root),
        'name': 'train',
        'exist_ok': True,
        'verbose': False,
        'plots': False,
        'save': True,
        'cache': 'ram',
        'workers': args.workers,
        'seed': args.seed,
        'deterministic': False,
        'pretrained': 'yolov8m.pt',
        'optimizer': 'SGD',
        **hyp,
    }

    print(json.dumps({
        'occlusion_ratio': args.occlusion_ratio,
        'beta': args.beta,
        'kd_branch': args.kd_branch,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'workers': args.workers,
        'prep_workers': args.prep_workers,
        'seed': args.seed,
        'output_dir': args.output_dir,
        'data_yaml': str(data_yaml),
        'device': device,
    }, indent=2))

    model = YOLO('yolov8m.pt')
    model.train(**train_args)

    best_path = exp_root / 'train' / 'weights' / 'best.pt'
    if not best_path.exists():
        raise RuntimeError(f'missing best weights: {best_path}')

    metrics = evaluate_model(best_path, data_yaml, device)
    results = {
        'occlusion_ratio': args.occlusion_ratio,
        'degradation_level': args.beta,
        'kd_branch': args.kd_branch,
        'seed': args.seed,
        'metrics': metrics,
        'best_weights': str(best_path),
        'data_yaml': str(data_yaml),
    }
    (exp_root / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    (exp_root / 'results.yaml').write_text(yaml.safe_dump(results, sort_keys=False))


if __name__ == '__main__':
    main()
