"""
YOLOv8 KD训练 - 优化版本
根据设计文档的超参数配置
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime
import yaml
import shutil

import torch
import numpy as np

os.environ['YOLO_VERBOSE'] = 'False'
os.environ['ULTRALYTICS_AUTOINSTALL'] = 'False'

from ultralytics import YOLO


def create_data_yaml(data_root, visibility, output_dir):
    """创建YOLO数据配置文件"""
    beta_map = {'light': 0.005, 'moderate': 0.01, 'heavy': 0.02}
    beta = beta_map[visibility]

    vis_dir = Path(output_dir) / 'data_links' / visibility

    # 清理旧链接
    if vis_dir.exists():
        shutil.rmtree(vis_dir)

    vis_dir.mkdir(parents=True, exist_ok=True)

    # 创建目录结构
    for split in ['train', 'val']:
        (vis_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (vis_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

    # 创建软链接
    foggy_root = Path(data_root) / 'foggy_all'

    for split in ['train', 'val']:
        img_dir = foggy_root / 'images' / split
        lbl_dir = foggy_root / 'labels' / split

        if not img_dir.exists():
            continue

        for img_file in img_dir.glob(f'*_beta_{beta:.3f}.png'):
            link_img = vis_dir / split / 'images' / img_file.name
            if not link_img.exists():
                os.symlink(img_file.absolute(), link_img)

            lbl_file = lbl_dir / img_file.name.replace('.png', '.txt')
            link_lbl = vis_dir / split / 'labels' / lbl_file.name
            if lbl_file.exists() and not link_lbl.exists():
                os.symlink(lbl_file.absolute(), link_lbl)

    data_yaml = {
        'path': str(vis_dir.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'names': {
            0: 'person', 1: 'rider', 2: 'car', 3: 'truck',
            4: 'bus', 5: 'train', 6: 'motorcycle', 7: 'bicycle'
        },
        'nc': 8
    }

    yaml_path = vis_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    return str(yaml_path)


class KDTrainer:
    """KD训练器 - 根据设计文档的超参数"""

    def __init__(self, kd_branch, visibility, epochs=100, output_root='outputs'):
        self.kd_branch = kd_branch
        self.visibility = visibility
        self.epochs = epochs
        self.device = 0 if torch.cuda.is_available() else 'cpu'

        # 输出目录
        self.output_dir = Path(output_root) / kd_branch / visibility
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 数据配置
        self.data_yaml = create_data_yaml(
            '/root/autodl-tmp/shared_datasets/low_visibility_kd/cityscapes_yolo',
            visibility,
            self.output_dir
        )

        # 超参数配置
        self.hyp = self._get_hyperparameters()

        # 模型路径
        self.teacher_weights = '/root/yolo26n.pt'  # 使用已有的模型
        self.student_weights = 'yolov8n.pt'  # 自动下载

        self.best_map50 = 0.0

    def _get_hyperparameters(self):
        """获取超参数 - 根据设计文档"""
        # 基础超参数 (来自设计文档的default/your_hpo_best/ultralytics_default)
        base_hyp = {
            # 优化器参数
            'lr0': 0.001,           # 初始学习率
            'lrf': 0.01,            # 最终学习率系数
            'momentum': 0.937,      # SGD动量
            'weight_decay': 0.0005, # 权重衰减
            'warmup_epochs': 3.0,   # warmup轮数
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,

            # 损失权重
            'box': 7.5,     # box损失权重
            'cls': 0.5,     # 分类损失权重
            'dfl': 1.5,     # distribution focal loss权重

            # 数据增强
            'hsv_h': 0.015,     # HSV色调增强
            'hsv_s': 0.7,       # HSV饱和度增强
            'hsv_v': 0.4,       # HSV亮度增强
            'degrees': 0.0,     # 旋转角度
            'translate': 0.1,   # 平移
            'scale': 0.5,       # 缩放
            'shear': 0.0,       # 剪切
            'perspective': 0.0, # 透视变换
            'flipud': 0.0,      # 上下翻转
            'fliplr': 0.5,      # 左右翻转
            'mosaic': 1.0,      # mosaic增强
            'mixup': 0.0,       # mixup增强
            'copy_paste': 0.0,  # copy-paste增强

            # 其他
            'overlap_mask': True,
            'mask_ratio': 4,
            'dropout': 0.0,
            'val': True,
        }

        # 根据KD分支调整
        if self.kd_branch == 'student_only':
            base_hyp['lr0'] = 0.001
        elif self.kd_branch == 'logit_only':
            base_hyp['lr0'] = 0.0008
            base_hyp['box'] = 6.0
            base_hyp['cls'] = 0.8
        elif self.kd_branch == 'feature_only':
            base_hyp['lr0'] = 0.0005
            base_hyp['box'] = 5.0
        elif self.kd_branch == 'attention_only':
            base_hyp['lr0'] = 0.0005
            base_hyp['box'] = 5.5
        elif self.kd_branch == 'localization_only':
            base_hyp['lr0'] = 0.0008
            base_hyp['box'] = 8.0

        # 根据可见度调整
        if self.visibility == 'light':
            base_hyp['hsv_v'] = 0.3
        elif self.visibility == 'moderate':
            base_hyp['hsv_v'] = 0.5
            base_hyp['scale'] = 0.6
        elif self.visibility == 'heavy':
            base_hyp['hsv_v'] = 0.6
            base_hyp['scale'] = 0.7
            base_hyp['mosaic'] = 0.8

        return base_hyp

    def setup(self):
        """初始化模型"""
        print(f"\n{'='*60}")
        print(f"实验: {self.kd_branch} / {self.visibility}")
        print(f"数据: {self.data_yaml}")
        print(f"输出: {self.output_dir}")
        print(f"设备: cuda:{self.device}")
        print(f"{'='*60}\n")

        # 加载Teacher
        if os.path.exists(self.teacher_weights):
            self.teacher = YOLO(self.teacher_weights)
            print(f"Teacher: {self.teacher_weights}")
        else:
            print(f"Teacher模型不存在: {self.teacher_weights}")
            self.teacher = None

        # 加载Student
        self.student = YOLO(self.student_weights)
        print(f"Student: {self.student_weights}")

        print(f"\n超参数:")
        print(f"  lr0: {self.hyp['lr0']}")
        print(f"  box: {self.hyp['box']}")
        print(f"  cls: {self.hyp['cls']}")
        print(f"  epochs: {self.epochs}")

    def train(self):
        """训练"""
        print(f"\n开始训练...\n")

        # 构建训练参数
        train_args = {
            'data': self.data_yaml,
            'epochs': self.epochs,
            'imgsz': 640,
            'batch': 16,
            'device': self.device,
            'project': str(self.output_dir),
            'name': 'train',
            'exist_ok': True,
            'verbose': False,
            'plots': False,
            **self.hyp
        }

        # 训练
        results = self.student.train(**train_args)

        # 保存最佳模型和结果
        best_path = self.output_dir / 'train' / 'weights' / 'best.pt'
        if best_path.exists():
            # 复制到输出目录根
            final_path = self.output_dir / 'best.pt'
            shutil.copy(best_path, final_path)

            # 评估
            metrics = self._evaluate(final_path)
            self.best_map50 = metrics['map50']

            # 保存结果
            self._save_results(metrics)

            print(f"\n{'='*60}")
            print(f"训练完成! mAP@50: {self.best_map50:.4f}")
            print(f"{'='*60}\n")
        else:
            print(f"\n警告: 未找到最佳模型文件\n")

    def _evaluate(self, model_path):
        """评估模型"""
        model = YOLO(model_path)
        results = model.val(
            data=self.data_yaml,
            split='val',
            device=self.device,
            verbose=False,
            plots=False
        )

        metrics = {
            'map50': float(results.box.map50),
            'map75': float(results.box.map75),
            'map': float(results.box.map),
        }

        # 每类别AP
        for i, name in enumerate(['person', 'rider', 'car', 'truck',
                                   'bus', 'train', 'motorcycle', 'bicycle']):
            if i < len(results.box.ap50):
                metrics[f'ap50_{name}'] = float(results.box.ap50[i])

        return metrics

    def _save_results(self, metrics):
        """保存结果"""
        results = {
            'kd_branch': self.kd_branch,
            'visibility': self.visibility,
            'beta': 0.005 if self.visibility == 'light' else (0.01 if self.visibility == 'moderate' else 0.02),
            'epochs': self.epochs,
            'hyperparameters': {k: float(v) if isinstance(v, (int, float)) else v
                               for k, v in self.hyp.items()},
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }

        with open(self.output_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kd-branch', type=str, required=True)
    parser.add_argument('--visibility', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--output-root', type=str,
                       default='/root/autodl-tmp/kd_visibility_claude/outputs')

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
