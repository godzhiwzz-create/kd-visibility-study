"""
知识蒸馏训练脚本
支持5种KD分支 × 3种可见度等级的完整实验矩阵
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

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.experiment_config import (
    ExperimentConfig, ModelConfig, DataConfig,
    KDLossConfig, TrainingConfig, KDBranch, VisibilityLevel
)
from data.cityscapes_dataset import create_dataloaders
from models.segmentation import TeacherStudentPair
from losses.kd_losses import CombinedKDLoss
from utils.metrics import SegmentationMetrics, MechanismAnalyzer


class Trainer:
    """训练器类"""

    def __init__(self, config: ExperimentConfig):
        """
        初始化训练器

        Args:
            config: 实验配置
        """
        self.config = config
        self.device = torch.device(config.training.device if torch.cuda.is_available() else 'cpu')

        # 创建输出目录
        self.output_dir = Path(config.get_output_dir())
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 保存配置
        self._save_config()

        # 初始化组件
        self.model = None
        self.dataloaders = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.writer = None

        # 评估器
        self.train_metrics = SegmentationMetrics(num_classes=19)
        self.val_metrics = SegmentationMetrics(num_classes=19)
        self.mechanism_analyzer = MechanismAnalyzer(num_classes=19)

        # 训练状态
        self.current_epoch = 0
        self.best_val_miou = 0.0
        self.training_history = []

    def _save_config(self):
        """保存配置到文件"""
        config_dict = {
            'name': self.config.name,
            'kd_branch': self.config.kd_branch.value,
            'visibility': self.config.visibility.value,
            'beta': self.config.get_beta(),
            'model': {
                'teacher_backbone': self.config.model.teacher_backbone,
                'student_backbone': self.config.model.student_backbone,
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
        print(f"实验: {self.config.name}")
        print(f"KD分支: {self.config.kd_branch.value}")
        print(f"可见度等级: {self.config.visibility.value} (beta={self.config.get_beta()})")
        print(f"输出目录: {self.output_dir}")
        print(f"设备: {self.device}")
        print(f"{'='*60}\n")

        # 创建数据加载器
        print("正在加载数据...")
        self.dataloaders = create_dataloaders(
            data_config=self.config.data,
            batch_size=self.config.data.batch_size,
            num_workers=self.config.data.num_workers
        )
        print(f"训练样本数: {len(self.dataloaders['train'].dataset)}")
        print(f"验证样本数: {len(self.dataloaders['val'].dataset)}")

        # 创建模型
        print("\n正在初始化模型...")
        teacher_config = {
            'backbone': self.config.model.teacher_backbone,
            'pretrained': self.config.model.teacher_pretrained,
            'checkpoint': self.config.model.teacher_checkpoint
        }
        student_config = {
            'backbone': self.config.model.student_backbone,
            'pretrained': self.config.model.student_pretrained
        }

        self.model = TeacherStudentPair(
            teacher_config=teacher_config,
            student_config=student_config,
            num_classes=19
        ).to(self.device)

        # 统计参数量
        teacher_params = sum(p.numel() for p in self.model.teacher.parameters())
        student_params = sum(p.numel() for p in self.model.student.parameters())
        print(f"Teacher参数量: {teacher_params/1e6:.2f}M")
        print(f"Student参数量: {student_params/1e6:.2f}M")
        print(f"压缩比: {teacher_params/student_params:.2f}x")

        # 创建损失函数
        kd_config = {
            'temperature': self.config.kd_loss.temperature,
            'logit_weight': self.config.kd_loss.logit_weight,
            'feature_weight': self.config.kd_loss.feature_weight,
            'attention_weight': self.config.kd_loss.attention_weight,
            'localization_weight': self.config.kd_loss.localization_weight,
            'feature_layers': self.config.kd_loss.feature_layers
        }

        self.criterion = CombinedKDLoss(
            kd_branch=self.config.kd_branch.value,
            kd_config=kd_config,
            num_classes=19
        ).to(self.device)

        # 创建优化器
        if self.config.training.optimizer == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.student.parameters(),
                lr=self.config.training.lr,
                weight_decay=self.config.training.weight_decay
            )
        elif self.config.training.optimizer == 'sgd':
            self.optimizer = optim.SGD(
                self.model.student.parameters(),
                lr=self.config.training.lr,
                momentum=0.9,
                weight_decay=self.config.training.weight_decay
            )
        else:
            raise ValueError(f"未知的优化器: {self.config.training.optimizer}")

        # 创建学习率调度器
        if self.config.training.scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.num_epochs
            )
        elif self.config.training.scheduler == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.training.num_epochs // 3,
                gamma=0.1
            )
        elif self.config.training.scheduler == 'none':
            self.scheduler = None
        else:
            raise ValueError(f"未知的调度器: {self.config.training.scheduler}")

        # 创建TensorBoard写入器
        self.writer = SummaryWriter(log_dir=self.output_dir / 'logs')

    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        self.train_metrics.reset()

        total_loss = 0.0
        loss_components = {}

        # 机制分析指标累计
        mechanism_metrics_sum = {}

        for batch_idx, batch in enumerate(self.dataloaders['train']):
            # 数据移至设备
            clear_images = batch['clear_image'].to(self.device)
            foggy_images = batch['foggy_image'].to(self.device)
            labels = batch['label'].to(self.device)

            # 前向传播
            self.optimizer.zero_grad()

            with torch.no_grad():
                teacher_output = self.model.get_teacher_output(clear_images)
            student_output = self.model.get_student_output(foggy_images)

            # 计算损失
            loss, loss_dict = self.criterion(
                student_output,
                teacher_output,
                labels
            )

            # 反向传播
            loss.backward()
            self.optimizer.step()

            # 累计统计
            total_loss += loss.item()
            for key, value in loss_dict.items():
                loss_components[key] = loss_components.get(key, 0.0) + value

            # 更新指标
            self.train_metrics.update(student_output['logits'], labels)

            # 机制分析（每隔一定间隔）
            if batch_idx % 50 == 0:
                with torch.no_grad():
                    metrics = self.mechanism_analyzer.analyze_batch(
                        student_output,
                        teacher_output
                    )
                    for key, value in metrics.items():
                        mechanism_metrics_sum[key] = mechanism_metrics_sum.get(key, 0.0) + value

            # 打印进度
            if batch_idx % self.config.training.log_interval == 0:
                print(f"  Batch [{batch_idx}/{len(self.dataloaders['train'])}] "
                      f"Loss: {loss.item():.4f}")

        # 计算平均
        num_batches = len(self.dataloaders['train'])
        avg_loss = total_loss / num_batches
        for key in loss_components:
            loss_components[key] /= num_batches

        # 获取训练指标
        train_metrics = self.train_metrics.compute()

        results = {
            'loss': avg_loss,
            **{f'loss_{k}': v for k, v in loss_components.items()},
            **{f'train_{k}': v for k, v in train_metrics.items()},
            **{f'mech_{k}': v / max(1, num_batches // 50)
               for k, v in mechanism_metrics_sum.items()}
        }

        return results

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """验证"""
        self.model.eval()
        self.val_metrics.reset()

        total_loss = 0.0
        mechanism_metrics_sum = {}

        for batch in self.dataloaders['val']:
            clear_images = batch['clear_image'].to(self.device)
            foggy_images = batch['foggy_image'].to(self.device)
            labels = batch['label'].to(self.device)

            # 前向传播
            teacher_output = self.model.get_teacher_output(clear_images)
            student_output = self.model.get_student_output(foggy_images)

            # 计算损失
            loss, loss_dict = self.criterion(
                student_output,
                teacher_output,
                labels
            )

            total_loss += loss.item()

            # 更新指标
            self.val_metrics.update(student_output['logits'], labels)

            # 机制分析
            metrics = self.mechanism_analyzer.analyze_batch(
                student_output,
                teacher_output
            )
            for key, value in metrics.items():
                mechanism_metrics_sum[key] = mechanism_metrics_sum.get(key, 0.0) + value

        # 计算平均
        num_batches = len(self.dataloaders['val'])
        avg_loss = total_loss / num_batches

        # 获取验证指标
        val_metrics = self.val_metrics.compute()

        results = {
            'val_loss': avg_loss,
            **{f'val_{k}': v for k, v in val_metrics.items()},
            **{f'val_mech_{k}': v / num_batches
               for k, v in mechanism_metrics_sum.items()}
        }

        return results

    def train(self):
        """完整训练流程"""
        print("\n开始训练...\n")

        for epoch in range(self.config.training.num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            print(f"\nEpoch {epoch+1}/{self.config.training.num_epochs}")
            print("-" * 40)

            # 训练
            train_results = self.train_epoch()

            # 验证
            val_results = self.validate()

            # 更新学习率
            if self.scheduler is not None:
                self.scheduler.step()

            # 合并结果
            epoch_results = {**train_results, **val_results}
            epoch_results['epoch'] = epoch + 1
            epoch_results['lr'] = self.optimizer.param_groups[0]['lr']

            # 保存历史
            self.training_history.append(epoch_results)

            # 打印结果
            epoch_time = time.time() - epoch_start_time
            print(f"\nEpoch {epoch+1} 完成 ({epoch_time:.1f}s)")
            print(f"  Train Loss: {train_results['loss']:.4f}, "
                  f"mIoU: {train_results['train_mean_iou']:.4f}")
            print(f"  Val Loss: {val_results['val_loss']:.4f}, "
                  f"mIoU: {val_results['val_mean_iou']:.4f}")

            # 记录到TensorBoard
            for key, value in epoch_results.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(key, value, epoch)

            # 保存最佳模型
            if val_results['val_mean_iou'] > self.best_val_miou:
                self.best_val_miou = val_results['val_mean_iou']
                self.save_checkpoint(is_best=True)
                print(f"  -> 最佳模型保存 (mIoU: {self.best_val_miou:.4f})")

            # 定期保存
            if (epoch + 1) % self.config.training.save_interval == 0:
                self.save_checkpoint(is_best=False)

        # 训练结束
        print(f"\n{'='*60}")
        print(f"训练完成!")
        print(f"最佳验证mIoU: {self.best_val_miou:.4f}")
        print(f"{'='*60}\n")

        # 保存最终历史
        self.save_history()
        self.writer.close()

    def save_checkpoint(self, is_best: bool = False):
        """保存模型检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.student.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_miou': self.best_val_miou,
            'config': {
                'kd_branch': self.config.kd_branch.value,
                'visibility': self.config.visibility.value,
            }
        }

        if is_best:
            path = self.output_dir / 'best_model.pth'
        else:
            path = self.output_dir / f'checkpoint_epoch_{self.current_epoch+1}.pth'

        torch.save(checkpoint, path)

    def save_history(self):
        """保存训练历史"""
        history_path = self.output_dir / 'training_history.json'

        # 转换为可JSON序列化的格式
        serializable_history = []
        for record in self.training_history:
            clean_record = {}
            for k, v in record.items():
                if isinstance(v, (int, float)):
                    clean_record[k] = v
                elif isinstance(v, list):
                    clean_record[k] = [float(x) if isinstance(x, np.floating) else x for x in v]
            serializable_history.append(clean_record)

        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_history, f, indent=2, ensure_ascii=False)


def train_single_experiment(
    kd_branch: str,
    visibility: str,
    num_epochs: int = 50,
    output_root: str = "outputs"
):
    """
    运行单个实验

    Args:
        kd_branch: KD分支类型
        visibility: 可见度等级
        num_epochs: 训练轮数
        output_root: 输出根目录
    """
    # 创建配置
    config = ExperimentConfig(
        name="kd_visibility_study",
        kd_branch=KDBranch(kd_branch),
        visibility=VisibilityLevel(visibility),
        training=TrainingConfig(
            num_epochs=num_epochs,
            output_root=output_root
        )
    )

    # 创建训练器并训练
    trainer = Trainer(config)
    trainer.setup()
    trainer.train()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='KD可见度研究训练脚本')
    parser.add_argument('--kd-branch', type=str, required=True,
                        choices=['student_only', 'logit_only', 'feature_only',
                                'attention_only', 'localization_only'],
                        help='KD分支类型')
    parser.add_argument('--visibility', type=str, required=True,
                        choices=['light', 'moderate', 'heavy'],
                        help='可见度等级')
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--output-root', type=str, default='outputs',
                        help='输出根目录')

    args = parser.parse_args()

    train_single_experiment(
        kd_branch=args.kd_branch,
        visibility=args.visibility,
        num_epochs=args.epochs,
        output_root=args.output_root
    )


if __name__ == '__main__':
    main()
