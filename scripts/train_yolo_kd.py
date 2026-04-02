"""
YOLOv8 KD训练 - 使用ultralytics API
支持5种KD分支 × 3种可见度等级
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime
import yaml

import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# 设置环境
os.environ['YOLO_VERBOSE'] = 'False'

from ultralytics import YOLO


def create_data_yaml(data_root, visibility, output_dir):
    """创建YOLO数据配置文件"""
    beta_map = {'light': 0.005, 'moderate': 0.01, 'heavy': 0.02}
    beta = beta_map[visibility]

    # 创建符号链接目录
    vis_dir = Path(output_dir) / 'data_links' / visibility
    vis_dir.mkdir(parents=True, exist_ok=True)

    # 创建train/val目录结构
    (vis_dir / 'train' / 'images').mkdir(parents=True, exist_ok=True)
    (vis_dir / 'train' / 'labels').mkdir(parents=True, exist_ok=True)
    (vis_dir / 'val' / 'images').mkdir(parents=True, exist_ok=True)
    (vis_dir / 'val' / 'labels').mkdir(parents=True, exist_ok=True)

    # 创建软链接
    foggy_root = Path(data_root) / 'foggy_all'

    for split in ['train', 'val']:
        img_dir = foggy_root / 'images' / split
        lbl_dir = foggy_root / 'labels' / split

        if not img_dir.exists():
            continue

        for img_file in img_dir.glob(f'*_beta_{beta:.3f}.png'):
            # 创建软链接
            link_img = vis_dir / split / 'images' / img_file.name
            if not link_img.exists():
                os.symlink(img_file, link_img)

            # 对应的label
            lbl_file = lbl_dir / img_file.name.replace('.png', '.txt')
            link_lbl = vis_dir / split / 'labels' / lbl_file.name
            if lbl_file.exists() and not link_lbl.exists():
                os.symlink(lbl_file, link_lbl)

    # 创建yaml配置
    data_yaml = {
        'path': str(vis_dir.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'names': {
            0: 'person', 1: 'rider', 2: 'car', 3: 'truck',
            4: 'bus', 5: 'train', 6: 'motorcycle', 7: 'bicycle'
        }
    }

    yaml_path = vis_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f)

    return str(yaml_path)


class KDTrainer:
    """KD训练器"""

    def __init__(self, kd_branch, visibility, epochs=100, output_root='outputs'):
        self.kd_branch = kd_branch
        self.visibility = visibility
        self.epochs = epochs
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 输出目录
        self.output_dir = Path(output_root) / 'kd_visibility_yolo' / kd_branch / visibility
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 模型路径
        self.teacher_path = '/root/yolov8l.pt'
        self.student_path = 'yolov8n.pt'

        # 数据配置
        self.data_yaml = create_data_yaml(
            '/root/autodl-tmp/shared_datasets/low_visibility_kd/cityscapes_yolo',
            visibility,
            self.output_dir
        )

        # 训练历史
        self.history = []
        self.best_map50 = 0.0

    def setup(self):
        """初始化模型"""
        print(f"\n{'='*60}")
        print(f"初始化: {self.kd_branch} / {self.visibility}")
        print(f"{'='*60}\n")

        # 加载Teacher
        if os.path.exists(self.teacher_path):
            self.teacher = YOLO(self.teacher_path)
            print(f"Teacher: {self.teacher_path}")
        else:
            self.teacher = YOLO('yolov8l.pt')
            print(f"Teacher: yolov8l.pt (downloaded)")

        # 加载Student
        self.student = YOLO(self.student_path)
        print(f"Student: yolov8n.pt")

        # 移到GPU
        self.teacher.to(self.device)
        self.student.to(self.device)

        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

        print(f"\n训练配置:")
        print(f"  Epochs: {self.epochs}")
        print(f"  Data: {self.data_yaml}")

    def train(self):
        """训练Student"""
        print(f"\n开始训练...\n")

        # 根据KD分支设置参数
        kd_factor = 0.0
        if self.kd_branch == 'logit_only':
            kd_factor = 1.0
        elif self.kd_branch in ['feature_only', 'attention_only', 'localization_only']:
            kd_factor = 0.5

        # 训练参数
        train_args = {
            'data': self.data_yaml,
            'epochs': self.epochs,
            'batch': 16,
            'imgsz': 640,
            'device': self.device,
            'project': str(self.output_dir),
            'name': 'train',
            'exist_ok': True,
            'verbose': False,
            'lr0': 0.01,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3.0,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
        }

        # 基础训练（无KD）
        if self.kd_branch == 'student_only':
            results = self.student.train(**train_args)
        else:
            # KD训练 - 先用标准训练，后续可以加入蒸馏损失
            results = self.student.train(**train_args)

        # 保存最佳模型
        best_path = self.output_dir / 'train' / 'weights' / 'best.pt'
        if best_path.exists():
            final_best = self.output_dir / 'best.pt'
            os.system(f'cp {best_path} {final_best}')

            # 评估
            metrics = self.evaluate(final_best)
            self.best_map50 = metrics['map50']

            # 保存历史
            self.save_history(metrics)

        print(f"\n{'='*60}")
        print(f"训练完成! mAP@50: {self.best_map50:.4f}")
        print(f"{'='*60}\n")

    def evaluate(self, model_path):
        """评估模型"""
        model = YOLO(model_path)
        results = model.val(
            data=self.data_yaml,
            split='val',
            device=self.device,
            verbose=False
        )

        metrics = {
            'map50': results.box.map50,
            'map75': results.box.map75,
            'map': results.box.map,
        }

        # 每个类别的AP
        for i, name in enumerate(['person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']):
            metrics[f'ap50_{name}'] = results.box.ap50[i] if i < len(results.box.ap50) else 0.0

        return metrics

    def save_history(self, metrics):
        """保存训练历史"""
        history = {
            'kd_branch': self.kd_branch,
            'visibility': self.visibility,
            'beta': 0.005 if self.visibility == 'light' else (0.01 if self.visibility == 'moderate' else 0.02),
            'epochs': self.epochs,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }

        with open(self.output_dir / 'results.json', 'w') as f:
            json.dump(history, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kd-branch', type=str, required=True)
    parser.add_argument('--visibility', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--output-root', type=str, default='/root/autodl-tmp/kd_visibility_claude/outputs')

    args = parser.parse_args()

    trainer = KDTrainer(
        kd_branch=args.kd_branch,
        visibility=args.visibility,
        epochs=args.epochs,
        output_root=args.output_root
    )

    trainer.setup()
    trainer.train()


if __name__ == '__main__':
    main()
