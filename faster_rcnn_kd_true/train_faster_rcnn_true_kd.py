#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.roi_heads import fastrcnn_loss
from torchvision.transforms import functional as TF

sys.path.insert(0, '/root/kd_visibility/causal_experiment')
from train_causal_effective import build_dataset, set_seed  # noqa: E402

WORK_ROOT = Path('/root/kd_visibility')
CLASS_NAMES = {
    1: 'person', 2: 'rider', 3: 'car', 4: 'truck',
    5: 'bus', 6: 'train', 7: 'motorcycle', 8: 'bicycle'
}


def collate_fn(batch):
    return tuple(zip(*batch))


class YoloDetectionDataset(Dataset):
    def __init__(self, data_yaml: str | Path, split: str, max_samples: int | None = None):
        cfg = yaml.safe_load(Path(data_yaml).read_text())
        root = Path(cfg['path'])
        image_rel = cfg[split]
        self.image_dir = (root / image_rel).resolve()
        # Effective datasets are laid out as split/images and split/labels.
        # Keep a fallback for older layouts to avoid silently dropping all boxes.
        primary_label_dir = self.image_dir.parent / 'labels'
        fallback_label_dir = self.image_dir.parent.parent / 'labels' / self.image_dir.name
        self.label_dir = primary_label_dir if primary_label_dir.exists() else fallback_label_dir
        self.image_paths = sorted([*self.image_dir.glob('*.png'), *self.image_dir.glob('*.jpg'), *self.image_dir.glob('*.jpeg')])
        if max_samples is not None:
            self.image_paths = self.image_paths[:max_samples]
        self.image_ids = {str(path): idx + 1 for idx, path in enumerate(self.image_paths)}

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        width, height = image.size
        image_tensor = TF.to_tensor(image)

        label_path = self.label_dir / f'{image_path.stem}.txt'
        boxes: list[list[float]] = []
        labels: list[int] = []
        areas: list[float] = []
        if label_path.exists():
            for raw in label_path.read_text().splitlines():
                parts = raw.strip().split()
                if len(parts) != 5:
                    continue
                cls_id = int(float(parts[0])) + 1
                xc, yc, bw, bh = map(float, parts[1:])
                x1 = max(0.0, (xc - bw / 2.0) * width)
                y1 = max(0.0, (yc - bh / 2.0) * height)
                x2 = min(float(width), (xc + bw / 2.0) * width)
                y2 = min(float(height), (yc + bh / 2.0) * height)
                if x2 <= x1 or y2 <= y1:
                    continue
                boxes.append([x1, y1, x2, y2])
                labels.append(cls_id)
                areas.append((x2 - x1) * (y2 - y1))

        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            'image_id': torch.tensor([self.image_ids[str(image_path)]], dtype=torch.int64),
            'area': torch.tensor(areas, dtype=torch.float32) if areas else torch.zeros((0,), dtype=torch.float32),
            'iscrowd': torch.zeros((len(labels),), dtype=torch.int64),
            'orig_size': torch.tensor([height, width], dtype=torch.int64),
            'size': torch.tensor([height, width], dtype=torch.int64),
            'file_name': str(image_path),
        }
        return image_tensor, target


def create_detector(num_classes: int) -> nn.Module:
    model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=ResNet50_Weights.IMAGENET1K_V2)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


class TrueFasterRCNNKD(nn.Module):
    def __init__(
        self,
        num_classes: int,
        kd_branch: str,
        kd_weight: float,
        temperature: float,
        teacher_path: str | None = None,
    ):
        super().__init__()
        self.kd_branch = kd_branch
        self.kd_weight = float(kd_weight)
        self.temperature = float(temperature)
        self.num_classes = int(num_classes)
        self.kd_degradation_level = 0.0
        self.kd_occlusion_ratio = 0.0
        self.student = create_detector(num_classes)
        self.teacher = None
        if kd_branch != 'student_only':
            if not teacher_path:
                raise ValueError('teacher_path is required for KD branches')
            teacher_path_obj = Path(teacher_path)
            if not teacher_path_obj.exists():
                raise FileNotFoundError(f'teacher checkpoint not found: {teacher_path}')
            self.teacher = create_detector(num_classes)
            payload = torch.load(teacher_path_obj, map_location='cpu')
            state_dict = payload.get('model_state_dict', payload)
            self.teacher.load_state_dict(state_dict)
            self.teacher.eval()
            for param in self.teacher.parameters():
                param.requires_grad = False

    @staticmethod
    def _clone_targets(targets: list[dict[str, Any]]) -> list[dict[str, Any]]:
        cloned = []
        for target in targets:
            item = {}
            for key, value in target.items():
                if torch.is_tensor(value):
                    item[key] = value.clone()
                else:
                    item[key] = value
            cloned.append(item)
        return cloned

    @staticmethod
    def _ensure_ordered(features: Any) -> OrderedDict[str, torch.Tensor]:
        if isinstance(features, torch.Tensor):
            return OrderedDict([('0', features)])
        return features

    def _student_forward_train(self, images, targets):
        original_sizes = [tuple(img.shape[-2:]) for img in images]
        images_list, targets = self.student.transform(images, targets)
        features = self._ensure_ordered(self.student.backbone(images_list.tensors))
        proposals, proposal_losses = self.student.rpn(images_list, features, targets)
        proposals, matched_idxs, labels, regression_targets = self.student.roi_heads.select_training_samples(proposals, targets)
        box_features = self.student.roi_heads.box_roi_pool(features, proposals, images_list.image_sizes)
        box_features = self.student.roi_heads.box_head(box_features)
        class_logits, box_regression = self.student.roi_heads.box_predictor(box_features)
        loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
        losses = {
            'loss_classifier': loss_classifier,
            'loss_box_reg': loss_box_reg,
            **proposal_losses,
        }
        return losses, None, {
            'images_list': images_list,
            'features': features,
            'proposals': proposals,
            'labels': labels,
            'class_logits': class_logits,
            'box_regression': box_regression,
        }

    def _teacher_logits(self, images, targets, proposals):
        if self.teacher is None:
            return None, None
        teacher_targets = self._clone_targets(targets)
        images_list, _ = self.teacher.transform(images, teacher_targets)
        features = self._ensure_ordered(self.teacher.backbone(images_list.tensors))
        box_features = self.teacher.roi_heads.box_roi_pool(features, proposals, images_list.image_sizes)
        box_features = self.teacher.roi_heads.box_head(box_features)
        class_logits, box_regression = self.teacher.roi_heads.box_predictor(box_features)
        return class_logits, box_regression

    def _logit_kd(self, student_logits, teacher_logits):
        if teacher_logits is None:
            return student_logits.sum() * 0.0
        t = self.temperature
        s_logits = torch.nan_to_num(student_logits, nan=0.0, posinf=20.0, neginf=-20.0).clamp_(-20.0, 20.0)
        t_logits = torch.nan_to_num(teacher_logits, nan=0.0, posinf=20.0, neginf=-20.0).clamp_(-20.0, 20.0)
        student_log_probs = F.log_softmax(s_logits / t, dim=1)
        teacher_probs = F.softmax(t_logits / t, dim=1)
        teacher_probs = torch.nan_to_num(teacher_probs, nan=0.0, posinf=1.0, neginf=0.0)
        teacher_probs = teacher_probs.clamp_min(1e-8)
        teacher_probs = teacher_probs / teacher_probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
        kd = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (t ** 2)
        if not torch.isfinite(kd):
            return student_logits.sum() * 0.0
        return kd

    def _occlusion_aware_kd(self, student_logits, teacher_logits):
        if teacher_logits is None:
            return student_logits.sum() * 0.0
        t = self.temperature
        s_logits = torch.nan_to_num(student_logits, nan=0.0, posinf=20.0, neginf=-20.0).clamp_(-20.0, 20.0)
        t_logits = torch.nan_to_num(teacher_logits, nan=0.0, posinf=20.0, neginf=-20.0).clamp_(-20.0, 20.0)
        student_log_probs = F.log_softmax(s_logits / t, dim=1)
        teacher_probs = F.softmax(t_logits / t, dim=1)
        teacher_probs = torch.nan_to_num(teacher_probs, nan=0.0, posinf=1.0, neginf=0.0)
        teacher_probs = teacher_probs.clamp_min(1e-8)
        teacher_probs = teacher_probs / teacher_probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
        per_roi_kl = F.kl_div(student_log_probs, teacher_probs, reduction='none').sum(dim=1) * (t ** 2)

        entropy = -(teacher_probs * teacher_probs.clamp_min(1e-8).log()).sum(dim=1)
        max_entropy = math.log(max(teacher_probs.shape[1], 2))
        conf_weight = 1.0 - (entropy / max_entropy).clamp(0.0, 1.0)
        visibility_scalar = max(0.1, 1.0 - 0.5 * self.kd_degradation_level - 0.8 * self.kd_occlusion_ratio)
        weights = conf_weight * visibility_scalar
        kd = (per_roi_kl * weights).sum() / weights.sum().clamp_min(1e-6)
        if not torch.isfinite(kd):
            return student_logits.sum() * 0.0
        return kd

    def _m2_spatial_aware_kd(self, student_logits, teacher_logits, student_box_regression, teacher_box_regression, labels):
        if teacher_logits is None or teacher_box_regression is None:
            return student_logits.sum() * 0.0

        labels_cat = torch.cat(labels, dim=0)
        positive = torch.where(labels_cat > 0)[0]
        if positive.numel() == 0:
            return student_logits.sum() * 0.0

        t = self.temperature
        s_logits = torch.nan_to_num(student_logits, nan=0.0, posinf=20.0, neginf=-20.0).clamp_(-20.0, 20.0)
        t_logits = torch.nan_to_num(teacher_logits, nan=0.0, posinf=20.0, neginf=-20.0).clamp_(-20.0, 20.0)

        s_pos = s_logits[positive]
        t_pos = t_logits[positive]
        teacher_probs = F.softmax(t_pos / t, dim=1)
        teacher_probs = torch.nan_to_num(teacher_probs, nan=0.0, posinf=1.0, neginf=0.0)
        teacher_probs = teacher_probs.clamp_min(1e-8)
        teacher_probs = teacher_probs / teacher_probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
        student_log_probs = F.log_softmax(s_pos / t, dim=1)
        per_roi_kl = F.kl_div(student_log_probs, teacher_probs, reduction='none').sum(dim=1) * (t ** 2)

        max_entropy = math.log(max(teacher_probs.shape[1], 2))
        entropy = -(teacher_probs * teacher_probs.clamp_min(1e-8).log()).sum(dim=1)
        confidence = teacher_probs[:, 1:].max(dim=1).values if teacher_probs.shape[1] > 1 else teacher_probs[:, 0]
        reliability = confidence * (1.0 - (entropy / max_entropy).clamp(0.0, 1.0))

        visibility_scalar = max(0.15, 1.0 - 0.5 * self.kd_degradation_level - 0.8 * self.kd_occlusion_ratio)
        cls_scale = visibility_scalar
        loc_scale = 1.0 + 1.5 * (1.0 - visibility_scalar)

        student_boxes = student_box_regression.reshape(-1, self.num_classes, 4)
        teacher_boxes = teacher_box_regression.reshape(-1, self.num_classes, 4)
        chosen_labels = labels_cat[positive]
        s_boxes = student_boxes[positive, chosen_labels]
        t_boxes = teacher_boxes[positive, chosen_labels]
        per_roi_loc = F.smooth_l1_loss(s_boxes, t_boxes, reduction='none').mean(dim=1)

        cls_weights = reliability * cls_scale
        loc_weights = reliability * loc_scale
        cls_term = (per_roi_kl * cls_weights).sum() / cls_weights.sum().clamp_min(1e-6)
        loc_term = (per_roi_loc * loc_weights).sum() / loc_weights.sum().clamp_min(1e-6)
        kd = 0.25 * cls_term + 0.75 * loc_term
        if not torch.isfinite(kd):
            return student_logits.sum() * 0.0
        return kd

    def _localization_kd(self, student_box_regression, teacher_box_regression, labels):
        labels_cat = torch.cat(labels, dim=0)
        positive = torch.where(labels_cat > 0)[0]
        if positive.numel() == 0:
            return student_box_regression.sum() * 0.0
        student_boxes = student_box_regression.reshape(-1, self.num_classes, 4)
        teacher_boxes = teacher_box_regression.reshape(-1, self.num_classes, 4)
        chosen_labels = labels_cat[positive]
        s = student_boxes[positive, chosen_labels]
        t = teacher_boxes[positive, chosen_labels]
        return F.smooth_l1_loss(s, t)

    def forward(self, images, targets=None):
        if self.training:
            losses, _, aux = self._student_forward_train(images, targets)
            if self.kd_branch != 'student_only':
                with torch.no_grad():
                    teacher_logits, teacher_box_regression = self._teacher_logits(images, targets, aux['proposals'])
                if self.kd_branch == 'logit_only':
                    kd_loss = self._logit_kd(aux['class_logits'], teacher_logits)
                elif self.kd_branch == 'occlusion_aware':
                    kd_loss = self._occlusion_aware_kd(aux['class_logits'], teacher_logits)
                elif self.kd_branch == 'm2_spatial_aware':
                    kd_loss = self._m2_spatial_aware_kd(
                        aux['class_logits'],
                        teacher_logits,
                        aux['box_regression'],
                        teacher_box_regression,
                        aux['labels'],
                    )
                elif self.kd_branch == 'localization_only':
                    kd_loss = self._localization_kd(aux['box_regression'], teacher_box_regression, aux['labels'])
                else:
                    raise ValueError(f'Unsupported kd_branch: {self.kd_branch}')
                if not torch.isfinite(kd_loss):
                    kd_loss = aux['class_logits'].sum() * 0.0
                losses['loss_kd'] = kd_loss * self.kd_weight
            return losses
        return self.student(images)


def maybe_subset(dataset: Dataset, max_samples: int | None, seed: int) -> Dataset:
    if max_samples is None or len(dataset) <= max_samples:
        return dataset
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    return Subset(dataset, indices[:max_samples])


def coco_metrics(model: TrueFasterRCNNKD, data_loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    images_info = []
    annotations = []
    predictions = []
    ann_id = 1
    seen_images: set[int] = set()
    with torch.no_grad():
        for images, targets in data_loader:
            batch_images = [img.to(device) for img in images]
            outputs = model(batch_images)
            for image_tensor, target, output in zip(images, targets, outputs):
                image_id = int(target['image_id'].item())
                height = int(target['orig_size'][0].item())
                width = int(target['orig_size'][1].item())
                if image_id not in seen_images:
                    images_info.append({'id': image_id, 'width': width, 'height': height, 'file_name': Path(target['file_name']).name})
                    seen_images.add(image_id)
                gt_boxes = target['boxes'].cpu().tolist()
                gt_labels = target['labels'].cpu().tolist()
                for box, label in zip(gt_boxes, gt_labels):
                    x1, y1, x2, y2 = box
                    annotations.append({
                        'id': ann_id,
                        'image_id': image_id,
                        'category_id': int(label),
                        'bbox': [x1, y1, x2 - x1, y2 - y1],
                        'area': max(0.0, (x2 - x1) * (y2 - y1)),
                        'iscrowd': 0,
                    })
                    ann_id += 1
                out_boxes = output['boxes'].detach().cpu().tolist()
                out_labels = output['labels'].detach().cpu().tolist()
                out_scores = output['scores'].detach().cpu().tolist()
                for box, label, score in zip(out_boxes, out_labels, out_scores):
                    x1, y1, x2, y2 = box
                    predictions.append({
                        'image_id': image_id,
                        'category_id': int(label),
                        'bbox': [x1, y1, x2 - x1, y2 - y1],
                        'score': float(score),
                    })
    if not images_info:
        return {'mAP50': 0.0, 'mAP50_95': 0.0}
    coco_gt = COCO()
    coco_gt.dataset = {
        'images': images_info,
        'annotations': annotations,
        'categories': [{'id': idx, 'name': name} for idx, name in CLASS_NAMES.items()],
    }
    coco_gt.createIndex()
    if not predictions:
        return {'mAP50': 0.0, 'mAP50_95': 0.0}
    coco_dt = coco_gt.loadRes(predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.params.imgIds = [img['id'] for img in images_info]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    map_5095 = max(0.0, float(coco_eval.stats[0]))
    map_50 = max(0.0, float(coco_eval.stats[1]))
    return {
        'mAP50_95': map_5095,
        'mAP50': map_50,
    }


def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    totals: dict[str, float] = {}
    steps = 0
    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        batch_targets = []
        for target in targets:
            item = {}
            for key, value in target.items():
                item[key] = value.to(device) if torch.is_tensor(value) else value
            batch_targets.append(item)
        loss_dict = model(images, batch_targets)
        total_loss = sum(loss_dict.values())
        if not torch.isfinite(total_loss):
            loss_kd = loss_dict.get('loss_kd')
            base_losses = [value for key, value in loss_dict.items() if key != 'loss_kd']
            base_total = sum(base_losses) if base_losses else None
            if loss_kd is not None and base_total is not None and torch.isfinite(base_total):
                loss_dict['loss_kd'] = loss_kd.detach() * 0.0
                total_loss = sum(loss_dict.values())
                if torch.isfinite(total_loss):
                    optimizer.zero_grad(set_to_none=True)
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=5.0)
                    optimizer.step()
                    steps += 1
                    for key, value in loss_dict.items():
                        totals[key] = totals.get(key, 0.0) + float(value.detach().item())
                    totals['loss_total'] = totals.get('loss_total', 0.0) + float(total_loss.detach().item())
                    continue
            raise RuntimeError(f'Non-finite loss: {total_loss.item()}')
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=5.0)
        optimizer.step()
        steps += 1
        for key, value in loss_dict.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().item())
        totals['loss_total'] = totals.get('loss_total', 0.0) + float(total_loss.detach().item())
    return {key: value / max(steps, 1) for key, value in totals.items()}


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--occlusion_ratio', type=float, required=True)
    parser.add_argument('--beta', type=float, required=True)
    parser.add_argument('--kd_branch', choices=['student_only', 'logit_only', 'localization_only', 'occlusion_aware', 'm2_spatial_aware'], required=True)
    parser.add_argument('--teacher_path', default='')
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--prep_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--step_size', type=int, default=8)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--kd_weight', type=float, default=1.0)
    parser.add_argument('--temperature', type=float, default=2.0)
    parser.add_argument('--output_dir', default='outputs_faster_rcnn_true_kd')
    parser.add_argument('--max_train_samples', type=int, default=None)
    parser.add_argument('--max_val_samples', type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    degradation_level = float(args.beta)
    exp_name = f'occ_{args.occlusion_ratio}_{args.kd_branch}_deg_{degradation_level}_seed_{args.seed}'
    output_root = (WORK_ROOT / args.output_dir / exp_name).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    data_yaml = build_dataset(
        degradation_level=degradation_level,
        occlusion_ratio=args.occlusion_ratio,
        seed=args.seed,
        prep_workers=max(1, int(args.prep_workers)),
    )
    train_dataset = YoloDetectionDataset(data_yaml, 'train', max_samples=args.max_train_samples)
    val_dataset = YoloDetectionDataset(data_yaml, 'val', max_samples=args.max_val_samples)
    train_dataset = maybe_subset(train_dataset, args.max_train_samples, args.seed)
    val_dataset = maybe_subset(val_dataset, args.max_val_samples, args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=max(1, min(args.workers, 4)), collate_fn=collate_fn, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TrueFasterRCNNKD(
        num_classes=len(CLASS_NAMES) + 1,
        kd_branch=args.kd_branch,
        kd_weight=args.kd_weight,
        temperature=args.temperature,
        teacher_path=args.teacher_path or None,
    ).to(device)
    model.kd_degradation_level = degradation_level
    model.kd_occlusion_ratio = float(args.occlusion_ratio)
    optimizer = optim.SGD([p for p in model.parameters() if p.requires_grad], lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    history = []
    best_map50 = -1.0
    best_epoch = 0
    best_model_path = output_root / 'best_model_student.pth'
    last_model_path = output_root / 'last_model_student.pth'

    config = vars(args) | {'data_yaml': str(data_yaml), 'device': str(device)}
    (output_root / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    print(json.dumps(config, indent=2, ensure_ascii=False))

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, optimizer, train_loader, device)
        eval_metrics = coco_metrics(model, val_loader, device)
        scheduler.step()
        row = {'epoch': epoch, **train_metrics, **eval_metrics, 'lr': float(optimizer.param_groups[0]['lr'])}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

        torch.save({'epoch': epoch, 'model_state_dict': model.student.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, last_model_path)
        if eval_metrics['mAP50'] > best_map50:
            best_map50 = eval_metrics['mAP50']
            best_epoch = epoch
            torch.save({'epoch': epoch, 'model_state_dict': model.student.state_dict()}, best_model_path)

    (output_root / 'history.json').write_text(json.dumps(history, indent=2, ensure_ascii=False) + '\n')
    best_row = max(history, key=lambda item: item['mAP50']) if history else {'epoch': 0, 'mAP50': 0.0, 'mAP50_95': 0.0}
    result = {
        'occlusion_ratio': args.occlusion_ratio,
        'degradation_level': degradation_level,
        'kd_branch': args.kd_branch,
        'epochs': args.epochs,
        'seed': args.seed,
        'batch_size': args.batch_size,
        'workers': args.workers,
        'prep_workers': args.prep_workers,
        'teacher_path': args.teacher_path,
        'kd_weight': args.kd_weight,
        'temperature': args.temperature,
        'data_yaml': str(data_yaml),
        'run_dir': str(output_root),
        'best_model_path': str(best_model_path),
        'last_model_path': str(last_model_path),
        'best_epoch': int(best_epoch),
        'mAP50': float(best_row['mAP50']),
        'mAP50_95': float(best_row['mAP50_95']),
    }
    save_json(output_root / 'results.json', result)
    (output_root / 'results.yaml').write_text(yaml.safe_dump(result, sort_keys=False, allow_unicode=True))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
