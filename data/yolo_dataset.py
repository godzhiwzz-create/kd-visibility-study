"""
YOLO格式雾天目标检测数据集加载器
适配云端 /shared_datasets/low_visibility_kd/cityscapes_yolo/
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from typing import Tuple, Dict, List, Optional
import random


class YOLOFoggyDataset(Dataset):
    """
    YOLO格式Cityscapes雾天目标检测数据集

    支持三种可见度等级：
    - light: beta=0.005
    - moderate: beta=0.01
    - heavy: beta=0.02
    """

    # Cityscapes类别定义（YOLO格式）
    NUM_CLASSES = 8
    CLASS_NAMES = ['person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']

    # beta值映射
    BETA_MAP = {
        'light': 0.005,
        'moderate': 0.01,
        'heavy': 0.02
    }

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        visibility: str = "moderate",
        image_size: Tuple[int, int] = (640, 640),
        transform: Optional[transforms.Compose] = None,
        max_objects: int = 100
    ):
        """
        初始化数据集

        Args:
            data_root: 数据根目录 (如 /shared_datasets/low_visibility_kd/cityscapes_yolo)
            split: 数据集划分 (train/val)
            visibility: 可见度等级 (light/moderate/heavy)
            image_size: 图像尺寸 (H, W)
            max_objects: 每张图最大目标数（用于填充）
        """
        self.data_root = data_root
        self.split = split
        self.beta = self.BETA_MAP[visibility]
        self.image_size = image_size
        self.max_objects = max_objects

        # 路径设置
        self.clear_image_dir = os.path.join(data_root, 'clear', 'images', split)
        self.foggy_image_dir = os.path.join(data_root, 'foggy_all', 'images', split)
        self.label_dir = os.path.join(data_root, 'foggy_all', 'labels', split)

        # 获取样本列表
        self.samples = self._load_samples()

        # 图像变换
        self.transform = transform or self._get_default_transform()

    def _load_samples(self) -> List[Dict]:
        """加载样本列表"""
        samples = []

        if not os.path.exists(self.foggy_image_dir):
            return samples

        # 获取所有清晰图像
        clear_images = set()
        if os.path.exists(self.clear_image_dir):
            clear_images = {f.replace('_leftImg8bit.png', '')
                          for f in os.listdir(self.clear_image_dir)
                          if f.endswith('.png')}

        # 获取当前beta值的雾天图像
        for filename in os.listdir(self.foggy_image_dir):
            if f'_foggy_beta_{self.beta:.3f}.png' in filename:
                basename = filename.replace(f'_leftImg8bit_foggy_beta_{self.beta:.3f}.png', '')

                foggy_path = os.path.join(self.foggy_image_dir, filename)
                label_path = os.path.join(self.label_dir, filename.replace('.png', '.txt'))

                # 对应的清晰图像
                clear_filename = f'{basename}_leftImg8bit.png'
                clear_path = os.path.join(self.clear_image_dir, clear_filename)

                if os.path.exists(clear_path) and os.path.exists(label_path):
                    samples.append({
                        'basename': basename,
                        'clear_path': clear_path,
                        'foggy_path': foggy_path,
                        'label_path': label_path
                    })

        return samples

    def _get_default_transform(self) -> transforms.Compose:
        """默认图像变换"""
        return transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _load_labels(self, label_path: str) -> torch.Tensor:
        """
        加载YOLO格式标签

        YOLO格式: <class_id> <x_center> <y_center> <width> <height>
        所有值都是相对于图像尺寸的归一化值

        Returns:
            [max_objects, 5] 张量，格式为 [class_id, x, y, w, h]
            不足max_objects的部分用-1填充
        """
        labels = torch.full((self.max_objects, 5), -1.0)

        if not os.path.exists(label_path):
            return labels

        with open(label_path, 'r') as f:
            lines = f.readlines()

        num_objects = min(len(lines), self.max_objects)

        for i, line in enumerate(lines[:num_objects]):
            parts = line.strip().split()
            if len(parts) == 5:
                labels[i, 0] = float(parts[0])  # class_id
                labels[i, 1] = float(parts[1])  # x_center
                labels[i, 2] = float(parts[2])  # y_center
                labels[i, 3] = float(parts[3])  # width
                labels[i, 4] = float(parts[4])  # height

        return labels

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        """获取单个样本"""
        sample = self.samples[idx]

        # 加载图像
        clear_image = Image.open(sample['clear_path']).convert('RGB')
        foggy_image = Image.open(sample['foggy_path']).convert('RGB')

        # 应用变换
        clear_tensor = self.transform(clear_image)
        foggy_tensor = self.transform(foggy_image)

        # 加载标签
        labels = self._load_labels(sample['label_path'])

        return {
            'clear_image': clear_tensor,
            'foggy_image': foggy_tensor,
            'labels': labels,
            'metadata': {
                'basename': sample['basename'],
                'beta': self.beta,
                'num_objects': (labels[:, 0] >= 0).sum().item()
            }
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """
    自定义collate函数处理变长标签
    """
    clear_images = torch.stack([b['clear_image'] for b in batch])
    foggy_images = torch.stack([b['foggy_image'] for b in batch])
    labels = torch.stack([b['labels'] for b in batch])

    metadata = [b['metadata'] for b in batch]

    return {
        'clear_image': clear_images,
        'foggy_image': foggy_images,
        'labels': labels,
        'metadata': metadata
    }


def create_yolo_dataloaders(
    data_root: str,
    visibility: str = "moderate",
    batch_size: int = 8,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (640, 640)
) -> Dict[str, DataLoader]:
    """
    创建YOLO格式数据加载器

    Args:
        data_root: 数据根目录
        visibility: 可见度等级
        batch_size: 批次大小
        num_workers: 数据加载线程数
        image_size: 图像尺寸

    Returns:
        包含train/val DataLoader的字典
    """
    dataloaders = {}

    for split in ['train', 'val']:
        dataset = YOLOFoggyDataset(
            data_root=data_root,
            split=split,
            visibility=visibility,
            image_size=image_size
        )

        print(f"{split}数据集: {len(dataset)} 样本")

        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == 'train'),
            collate_fn=collate_fn
        )

    return dataloaders
