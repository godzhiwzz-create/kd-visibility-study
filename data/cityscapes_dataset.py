"""
Cityscapes雾天数据集加载器
支持可见度等级选择和不同退化程度
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
from typing import Tuple, Dict, List, Optional


class CityscapesFoggyDataset(Dataset):
    """
    Cityscapes雾天数据集

    支持三种可见度等级：
    - light: beta=0.005
    - moderate: beta=0.01
    - heavy: beta=0.02
    """

    # Cityscapes类别定义
    NUM_CLASSES = 19
    IGNORE_INDEX = 255

    # 类别名称映射
    CLASS_NAMES = [
        'road', 'sidewalk', 'building', 'wall', 'fence',
        'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain',
        'sky', 'person', 'rider', 'car', 'truck',
        'bus', 'train', 'motorcycle', 'bicycle'
    ]

    def __init__(
        self,
        clear_root: str,
        foggy_root: str,
        annotation_root: str,
        filename_list: str,
        beta: float = 0.01,
        split: str = "train",
        image_size: Tuple[int, int] = (512, 1024),
        transform: Optional[transforms.Compose] = None,
        target_transform: Optional[transforms.Compose] = None
    ):
        """
        初始化数据集

        Args:
            clear_root: 清晰图像根目录
            foggy_root: 雾天图像根目录
            annotation_root: 标注根目录
            filename_list: 文件名列表路径
            beta: 雾浓度参数 (0.005, 0.01, 0.02)
            split: 数据集划分 (train/val/test)
            image_size: 图像尺寸 (H, W)
            transform: 图像变换
            target_transform: 标注变换
        """
        self.clear_root = clear_root
        self.foggy_root = foggy_root
        self.annotation_root = annotation_root
        self.beta = beta
        self.split = split
        self.image_size = image_size

        # 加载文件名列表
        self.samples = self._load_filenames(filename_list)

        # 图像变换
        self.transform = transform or self._get_default_transform()
        self.target_transform = target_transform or self._get_default_target_transform()

        # 过滤当前split的样本
        self.samples = [s for s in self.samples if s['split'] == split]

    def _load_filenames(self, filename_list: str) -> List[Dict]:
        """加载文件名列表"""
        samples = []

        with open(filename_list, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # 解析路径: train/aachen/aachen_000004_000019
                parts = line.split('/')
                if len(parts) >= 3:
                    split, city, basename = parts[0], parts[1], parts[2]

                    samples.append({
                        'split': split,
                        'city': city,
                        'basename': basename
                    })

        return samples

    def _get_foggy_path(self, sample: Dict) -> str:
        """获取雾天图像路径"""
        # Foggy Cityscapes命名规则: {basename}_leftImg8bit_foggy_beta_{beta}.png
        foggy_filename = f"{sample['basename']}_leftImg8bit_foggy_beta_{self.beta:.3f}.png"
        return os.path.join(
            self.foggy_root,
            self.split,
            sample['city'],
            foggy_filename
        )

    def _get_clear_path(self, sample: Dict) -> str:
        """获取清晰图像路径"""
        clear_filename = f"{sample['basename']}_leftImg8bit.png"
        return os.path.join(
            self.clear_root,
            self.split,
            sample['city'],
            clear_filename
        )

    def _get_label_path(self, sample: Dict) -> str:
        """获取标注路径"""
        # 使用gtFine标注
        label_filename = f"{sample['basename']}_gtFine_labelIds.png"
        return os.path.join(
            self.annotation_root,
            self.split,
            sample['city'],
            label_filename
        )

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

    def _get_default_target_transform(self) -> transforms.Compose:
        """默认标注变换"""
        return transforms.Compose([
            transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.Lambda(lambda x: torch.from_numpy(np.array(x)).long())
        ])

    def _encode_target(self, target: np.ndarray) -> torch.Tensor:
        """
        将Cityscapes原始标注编码为训练用的类别标签
        将原始labelIds映射到训练用的trainIds
        """
        # Cityscapes ID映射 (labelId -> trainId)
        id_to_trainid = {
            7: 0,   # road
            8: 1,   # sidewalk
            11: 2,  # building
            12: 3,  # wall
            13: 4,  # fence
            17: 5,  # pole
            19: 6,  # traffic_light
            20: 7,  # traffic_sign
            21: 8,  # vegetation
            22: 9,  # terrain
            23: 10, # sky
            24: 11, # person
            25: 12, # rider
            26: 13, # car
            27: 14, # truck
            28: 15, # bus
            31: 16, # train
            32: 17, # motorcycle
            33: 18, # bicycle
        }

        # 将未定义的类别标记为255 (ignore)
        target_copy = np.ones_like(target) * 255
        for label_id, train_id in id_to_trainid.items():
            target_copy[target == label_id] = train_id

        return torch.from_numpy(target_copy).long()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        """获取单个样本"""
        sample = self.samples[idx]

        # 加载雾天图像
        foggy_path = self._get_foggy_path(sample)
        foggy_image = Image.open(foggy_path).convert('RGB')

        # 加载清晰图像（用于teacher）
        clear_path = self._get_clear_path(sample)
        clear_image = Image.open(clear_path).convert('RGB')

        # 加载标注
        label_path = self._get_label_path(sample)
        label = Image.open(label_path)

        # 应用变换
        foggy_tensor = self.transform(foggy_image)
        clear_tensor = self.transform(clear_image)
        label_tensor = self._encode_target(np.array(label))

        return {
            'foggy_image': foggy_tensor,
            'clear_image': clear_tensor,
            'label': label_tensor,
            'metadata': {
                'basename': sample['basename'],
                'city': sample['city'],
                'beta': self.beta
            }
        }


def create_dataloaders(
    data_config,
    batch_size: int = 4,
    num_workers: int = 4
) -> Dict[str, DataLoader]:
    """
    创建数据加载器

    Args:
        data_config: 数据配置对象
        batch_size: 批次大小
        num_workers: 数据加载线程数

    Returns:
        包含train/val/test DataLoader的字典
    """
    dataloaders = {}

    for split in ['train', 'val', 'test']:
        dataset = CityscapesFoggyDataset(
            clear_root=data_config.clear_root,
            foggy_root=data_config.foggy_root,
            annotation_root=data_config.annotation_root,
            filename_list=data_config.filename_list,
            beta=data_config.get_beta(),
            split=split,
            image_size=data_config.image_size
        )

        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == 'train')
        )

    return dataloaders
