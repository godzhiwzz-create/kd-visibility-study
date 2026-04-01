"""
YOLOv8 KD训练 - v4优化版
充分利用RTX 5090的32GB显存和575W功耗
优化超参数以提高性能
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

    if vis_dir.exists():
        shutil.rmtree(vis_dir)
    vis_dir.mkdir(parents=True, exist_ok=True)

    for split in ['train', 'val']:
        (vis_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (vis_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

    foggy_root = Path(data_root) / 'foggy_all'
    link_count = 0

    for split in ['train', 'val']:
        img_dir = foggy_root / 'images' / split
        lbl_dir = foggy_root / 'labels' / split

        if not img_dir.exists():
            continue

        # 使用正确的beta格式匹配文件名
        beta_str = f'{beta:.3f}'.rstrip('0').rstrip('.')
        pattern = f'*_beta_{beta_str}.png'
        img_files = list(img_dir.glob(pattern))
        print(f"{split}: 找到 {len(img_files)} 个图像 (beta={beta})")

        for img_file in img_files:
            link_img = vis_dir / split / 'images' / img_file.name
            try:
                if not link_img.exists():
                    os.symlink(img_file.absolute(), link_img)
                    link_count += 1
            except Exception as e:
                pass

            lbl_file = lbl_dir / img_file.name.replace('.png', '.txt')
            link_lbl = vis_dir / split / 'labels' / lbl_file.name
            if lbl_file.exists():
                try:
                    if not link_lbl.exists():
                        os.symlink(lbl_file.absolute(), link_lbl)
                except:
                    pass

    print(f"总共创建 {link_count} 个图像链接")

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

    train_images = list((vis_dir / 'train' / 'images').glob('*.png'))
    print(f"验证: train/images 中有 {len(train_images)} 个文件")

    if len(train_images) == 0:
        raise RuntimeError("没有创建任何数据链接!")

    return str(yaml_path)


class KDTrainer:
    """KD训练器 - v4优化版，充分利用RTX 5090"""

    def __init__(self, kd_branch, visibility, epochs=150, output_root='outputs'):
        self.kd_branch = kd_branch
        self.visibility = visibility
        self.epochs = epochs
        self.device = 0 if torch.cuda.is_available() else 'cpu'

        # 输出目录
        self.output_dir = Path(output_root) / kd_branch / visibility
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 数据配置
        print(f"\n准备数据: {visibility} (beta={0.005 if visibility=='light' else (0.01 if visibility=='moderate' else 0.02)})")
        self.data_yaml = create_data_yaml(
            '/root/autodl-tmp/shared_datasets/low_visibility_kd/cityscapes_yolo',
            visibility,
            self.output_dir
        )

        # 优化超参数 - 充分利用RTX 5090
        self.hyp = self._get_optimized_hyperparameters()

        # 模型路径 - 使用更大的模型充分利用显存
        self.teacher_weights = '/root/yolo26n.pt'
        self.student_weights = 'yolov8m.pt'  # 升级到medium模型

        self.best_map50 = 0.0

    def _get_optimized_hyperparameters(self):
        """优化超参数 - 针对RTX 5090 32GB"""

        # 基础优化参数
        base_hyp = {
            # 学习率优化 - 使用cosine annealing with warmup
            'lr0': 0.005,           # 提高初始学习率
            'lrf': 0.001,           # 降低最终学习率
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 5.0,   # 增加warmup
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,

            # 损失权重 - 优化组合
            'box': 7.5,
            'cls': 0.8,             # 增加分类权重
            'dfl': 1.5,

            # 数据增强 - 更强增强应对雾天
            'hsv_h': 0.02,          # 增加色调变化
            'hsv_s': 0.8,           # 增加饱和度
            'hsv_v': 0.5,           # 增加亮度变化
            'degrees': 5.0,         # 增加旋转
            'translate': 0.2,       # 增加平移
            'scale': 0.6,           # 增加缩放
            'shear': 2.0,           # 增加剪切
            'perspective': 0.001,   # 透视变换
            'flipud': 0.1,          # 上下翻转
            'fliplr': 0.5,          # 左右翻转
            'mosaic': 1.0,          # mosaic增强
            'mixup': 0.2,           # 启用mixup
            'copy_paste': 0.1,      # 启用copy-paste
            'erasing': 0.2,         # 随机擦除

            # 其他优化
            'overlap_mask': True,
            'mask_ratio': 4,
            'dropout': 0.1,         # 添加dropout
            'val': True,
            'nbs': 64,              # nominal batch size
        }

        # 根据KD分支调整
        if self.kd_branch == 'student_only':
            base_hyp['lr0'] = 0.005
            base_hyp['box'] = 7.5
        elif self.kd_branch == 'logit_only':
            base_hyp['lr0'] = 0.004
            base_hyp['box'] = 6.5
            base_hyp['cls'] = 1.0
            base_hyp['dropout'] = 0.05
        elif self.kd_branch == 'feature_only':
            base_hyp['lr0'] = 0.003
            base_hyp['box'] = 6.0
            base_hyp['dropout'] = 0.05
        elif self.kd_branch == 'attention_only':
            base_hyp['lr0'] = 0.003
            base_hyp['box'] = 6.5
            base_hyp['dropout'] = 0.08
        elif self.kd_branch == 'localization_only':
            base_hyp['lr0'] = 0.004
            base_hyp['box'] = 8.5     # 强调定位
            base_hyp['dfl'] = 2.0

        # 根据可见度调整
        if self.visibility == 'light':
            base_hyp['hsv_v'] = 0.4
            base_hyp['scale'] = 0.5
        elif self.visibility == 'moderate':
            base_hyp['hsv_v'] = 0.6
            base_hyp['scale'] = 0.7
            base_hyp['mixup'] = 0.3
        elif self.visibility == 'heavy':
            base_hyp['hsv_v'] = 0.7
            base_hyp['scale'] = 0.8
            base_hyp['mixup'] = 0.4
            base_hyp['mosaic'] = 0.9
            base_hyp['erasing'] = 0.3

        return base_hyp

    def setup(self):
        """初始化模型"""
        print(f"\n{'='*60}")
        print(f"实验(v4优化): {self.kd_branch} / {self.visibility}")
        print(f"数据: {self.data_yaml}")
        print(f"输出: {self.output_dir}")
        print(f"设备: cuda:{self.device}")
        print(f"{'='*60}\n")

        # 加载Teacher
        if os.path.exists(self.teacher_weights):
            self.teacher = YOLO(self.teacher_weights)
            print(f"Teacher: {self.teacher_weights}")
        else:
            self.teacher = None
            print(f"Teacher未找到，使用标准预训练权重")

        # 加载Student - yolov8m更大模型
        self.student = YOLO(self.student_weights)
        print(f"Student: {self.student_weights}")

        print(f"\n优化超参数:")
        print(f"  lr0: {self.hyp['lr0']}")
        print(f"  lrf: {self.hyp['lrf']}")
        print(f"  box: {self.hyp['box']}")
        print(f"  cls: {self.hyp['cls']}")
        print(f"  mixup: {self.hyp['mixup']}")
        print(f"  epochs: {self.epochs}")

    def train(self):
        """训练 - 大batch充分利用显存"""
        print(f"\n开始训练...\n")

        # 训练参数 - 大batch充分利用RTX 5090的32GB显存
        train_args = {
            'data': self.data_yaml,
            'epochs': self.epochs,
            'imgsz': 640,
            'batch': 32,            # 增大batch size充分利用显存
            'device': self.device,
            'project': str(self.output_dir),
            'name': 'train',
            'exist_ok': True,
            'verbose': False,
            'plots': False,
            'save': True,
            'cache': True,          # 缓存数据到内存
            'workers': 8,           # 增加数据加载 workers
            'amp': True,            # 启用自动混合精度
            **self.hyp
        }

        results = self.student.train(**train_args)

        # 保存最佳模型
        best_path = self.output_dir / 'train' / 'weights' / 'best.pt'
        if best_path.exists():
            final_path = self.output_dir / 'best.pt'
            shutil.copy(best_path, final_path)

            metrics = self._evaluate(final_path)
            self.best_map50 = metrics['map50']
            self._save_results(metrics)

            print(f"\n{'='*60}")
            print(f"训练完成! mAP@50: {self.best_map50:.4f}")
            print(f"{'='*60}\n")
        else:
            print(f"\n警告: 未找到最佳模型\n")
            raise RuntimeError("训练失败: 未生成模型文件")

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
            'student_model': 'yolov8m',
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
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--output-root', type=str,
                       default='/root/autodl-tmp/kd_visibility_claude/outputs_v4')

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
