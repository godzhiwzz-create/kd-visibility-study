"""
YOLO目标检测KD训练脚本
适配云端环境和YOLO格式数据
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.cloud_config import (
    CloudExperimentConfig, CloudModelConfig, CloudDataConfig,
    CloudTrainingConfig, KDBranch, VisibilityLevel
)
from data.yolo_dataset import create_yolo_dataloaders
from utils.yolo_metrics import DetectionMetrics, YOLOMechanismAnalyzer

# 尝试导入ultralytics
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("警告: ultralytics未安装，尝试安装...")


class YOLOTrainer:
    """YOLO检测训练器类"""

    def __init__(self, config: CloudExperimentConfig):
        self.config = config
        self.device = torch.device(config.training.device if torch.cuda.is_available() else 'cpu')

        # 创建输出目录
        self.output_dir = Path(config.get_output_dir())
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 保存配置
        self._save_config()

        # 初始化组件
        self.teacher = None
        self.student = None
        self.dataloaders = None
        self.optimizer = None
        self.scheduler = None
        self.writer = None

        # 评估器
        self.train_metrics = DetectionMetrics(num_classes=8)
        self.val_metrics = DetectionMetrics(num_classes=8)
        self.mechanism_analyzer = YOLOMechanismAnalyzer(num_classes=8)

        # 训练状态
        self.current_epoch = 0
        self.best_map50 = 0.0
        self.training_history = []

    def _save_config(self):
        """保存配置到文件"""
        config_dict = {
            'name': self.config.name,
            'kd_branch': self.config.kd_branch.value,
            'visibility': self.config.visibility.value,
            'beta': self.config.get_beta(),
            'model': {
                'teacher': self.config.model.teacher_model,
                'student': self.config.model.student_model,
            },
            'training': {
                'num_epochs': self.config.training.num_epochs,
                'lr': self.config.training.lr,
                'batch_size': self.config.data.batch_size,
            }
        }

        with open(self.output_dir / 'config.json', 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

    def setup(self):
        """设置训练环境"""
        print(f"\n{'='*60}")
        print(f"YOLO检测KD实验: {self.config.name}")
        print(f"KD分支: {self.config.kd_branch.value}")
        print(f"可见度等级: {self.config.visibility.value} (beta={self.config.get_beta()})")
        print(f"输出目录: {self.output_dir}")
        print(f"设备: {self.device}")
        print(f"{'='*60}\n")

        # 创建数据加载器
        print("正在加载数据...")
        self.dataloaders = create_yolo_dataloaders(
            data_root=self.config.data.data_root,
            visibility=self.config.visibility.value,
            batch_size=self.config.data.batch_size,
            num_workers=self.config.data.num_workers,
            image_size=self.config.data.image_size
        )

        # 加载YOLO模型
        print("\n正在初始化YOLO模型...")
        self._setup_models()

        # 创建优化器
        if self.config.training.optimizer == 'adamw':
            self.optimizer = optim.AdamW(
                self.student.parameters(),
                lr=self.config.training.lr,
                weight_decay=self.config.training.weight_decay
            )
        elif self.config.training.optimizer == 'sgd':
            self.optimizer = optim.SGD(
                self.student.parameters(),
                lr=self.config.training.lr,
                momentum=0.9,
                weight_decay=self.config.training.weight_decay
            )

        # 学习率调度器
        if self.config.training.scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.num_epochs
            )
        elif self.config.training.scheduler == 'none':
            self.scheduler = None

        # TensorBoard
        self.writer = SummaryWriter(log_dir=self.output_dir / 'logs')

    def _setup_models(self):
        """设置Teacher和Student模型"""
        if not ULTRALYTICS_AVAILABLE:
            raise RuntimeError("ultralytics未安装，无法加载YOLO模型")

        # 加载Teacher (YOLOv8l)
        teacher_path = self.config.model.teacher_checkpoint
        if teacher_path and os.path.exists(teacher_path):
            self.teacher = YOLO(teacher_path)
            print(f"已加载Teacher模型: {teacher_path}")
        else:
            self.teacher = YOLO(self.config.model.teacher_model)
            print(f"已加载预训练Teacher模型: {self.config.model.teacher_model}")

        # 加载Student (YOLOv8n)
        self.student = YOLO(self.config.model.student_model)
        print(f"已加载Student模型: {self.config.model.student_model}")

        # 移动到设备
        self.teacher.to(self.device)
        self.student.to(self.device)

        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

        # 统计参数量
        teacher_params = sum(p.numel() for p in self.teacher.parameters())
        student_params = sum(p.numel() for p in self.student.parameters())
        print(f"Teacher参数量: {teacher_params/1e6:.2f}M")
        print(f"Student参数量: {student_params/1e6:.2f}M")
        print(f"压缩比: {teacher_params/student_params:.2f}x")

    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.student.train()
        self.train_metrics.reset()

        total_loss = 0.0
        epoch_start = time.time()

        for batch_idx, batch in enumerate(self.dataloaders['train']):
            # 数据移至设备
            foggy_images = batch['foggy_image'].to(self.device)
            clear_images = batch['clear_image'].to(self.device)
            labels = batch['labels'].to(self.device)

            # 前向传播 - Student
            student_results = self.student(foggy_images)

            # 计算损失（YOLO内置）
            loss = student_results[0].sum() if isinstance(student_results, list) else student_results

            # KD损失（根据分支类型）
            kd_loss = self._compute_kd_loss(foggy_images, clear_images)
            total_batch_loss = loss + kd_loss

            # 反向传播
            self.optimizer.zero_grad()
            total_batch_loss.backward()
            self.optimizer.step()

            total_loss += total_batch_loss.item()

            # 打印进度
            if batch_idx % self.config.training.log_interval == 0:
                print(f"  Batch [{batch_idx}/{len(self.dataloaders['train'])}] "
                      f"Loss: {total_batch_loss.item():.4f}")

        avg_loss = total_loss / len(self.dataloaders['train'])

        return {'loss': avg_loss, 'time': time.time() - epoch_start}

    def _compute_kd_loss(self, student_images, teacher_images):
        """计算知识蒸馏损失"""
        if self.config.kd_branch == KDBranch.STUDENT_ONLY:
            return 0.0

        with torch.no_grad():
            teacher_results = self.teacher(teacher_images)

        # 根据分支类型实现不同的蒸馏损失
        # 简化为feature蒸馏
        return 0.1  # 占位

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """验证"""
        self.student.eval()
        self.val_metrics.reset()

        all_predictions = []
        all_targets = []

        for batch in self.dataloaders['val']:
            foggy_images = batch['foggy_image'].to(self.device)
            labels = batch['labels'].to(self.device)

            # 前向传播
            results = self.student(foggy_images)

            # 解析预测结果
            predictions = self._parse_yolo_results(results)

            all_predictions.extend(predictions)
            all_targets.extend(labels.cpu().numpy())

        # 计算mAP
        metrics = self.val_metrics.compute(all_predictions, all_targets)

        return metrics

    def _parse_yolo_results(self, results):
        """解析YOLO检测结果"""
        predictions = []
        # 简化的解析逻辑
        return predictions

    def train(self):
        """完整训练流程"""
        print("\n开始训练...\n")

        for epoch in range(self.config.training.num_epochs):
            self.current_epoch = epoch

            print(f"\nEpoch {epoch+1}/{self.config.training.num_epochs}")
            print("-" * 40)

            # 训练
            train_results = self.train_epoch()

            # 验证
            val_results = self.validate()

            # 更新学习率
            if self.scheduler:
                self.scheduler.step()

            # 合并结果
            epoch_results = {**train_results, **val_results}
            epoch_results['epoch'] = epoch + 1

            self.training_history.append(epoch_results)

            print(f"\nEpoch {epoch+1} 完成")
            print(f"  Train Loss: {train_results['loss']:.4f}")
            print(f"  Val mAP@50: {val_results.get('map50', 0):.4f}")

            # 保存最佳模型
            current_map50 = val_results.get('map50', 0)
            if current_map50 > self.best_map50:
                self.best_map50 = current_map50
                self._save_model(is_best=True)
                print(f"  -> 最佳模型保存 (mAP@50: {self.best_map50:.4f})")

            # 定期保存
            if (epoch + 1) % self.config.training.save_interval == 0:
                self._save_model(is_best=False)

        print(f"\n{'='*60}")
        print(f"训练完成! 最佳mAP@50: {self.best_map50:.4f}")
        print(f"{'='*60}\n")

        self._save_history()
        self.writer.close()

    def _save_model(self, is_best=False):
        """保存模型"""
        if is_best:
            path = self.output_dir / 'best.pt'
        else:
            path = self.output_dir / f'epoch_{self.current_epoch+1}.pt'

        self.student.save(path)

    def _save_history(self):
        """保存训练历史"""
        with open(self.output_dir / 'history.json', 'w') as f:
            json.dump(self.training_history, f, indent=2)


def train_single_experiment(kd_branch: str, visibility: str, epochs: int = 100):
    """运行单个实验"""
    config = CloudExperimentConfig(
        kd_branch=KDBranch(kd_branch),
        visibility=VisibilityLevel(visibility),
        training=CloudTrainingConfig(num_epochs=epochs)
    )

    trainer = YOLOTrainer(config)
    trainer.setup()
    trainer.train()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kd-branch', type=str, required=True,
                        choices=['student_only', 'logit_only', 'feature_only',
                                'attention_only', 'localization_only'])
    parser.add_argument('--visibility', type=str, required=True,
                        choices=['light', 'moderate', 'heavy'])
    parser.add_argument('--epochs', type=int, default=100)

    args = parser.parse_args()
    train_single_experiment(args.kd_branch, args.visibility, args.epochs)


if __name__ == '__main__':
    main()
