"""
云端服务器配置
适配 AutoDL 云端环境
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import os


class KDBranch(Enum):
    """KD分支类型"""
    STUDENT_ONLY = "student_only"
    LOGIT_ONLY = "logit_only"
    FEATURE_ONLY = "feature_only"
    ATTENTION_ONLY = "attention_only"
    LOCALIZATION_ONLY = "localization_only"


class VisibilityLevel(Enum):
    """可见度等级"""
    LIGHT = "light"
    MODERATE = "moderate"
    HEAVY = "heavy"


@dataclass
class CloudDataConfig:
    """云端数据配置"""
    # 数据路径
    data_root: str = "/root/autodl-tmp/shared_datasets/low_visibility_kd/cityscapes_yolo"

    # 可见度配置（雾浓度）
    visibility_levels: Dict[str, float] = field(default_factory=lambda: {
        "light": 0.005,
        "moderate": 0.01,
        "heavy": 0.02
    })

    # 数据增强
    image_size: tuple = (640, 640)
    batch_size: int = 16
    num_workers: int = 4


@dataclass
class CloudModelConfig:
    """云端模型配置"""
    # Teacher模型 (YOLOv8l)
    teacher_model: str = "yolov8l"
    teacher_pretrained: bool = True
    teacher_checkpoint: Optional[str] = "/root/yolov8l.pt"

    # Student模型 (YOLOv8n)
    student_model: str = "yolov8n"
    student_pretrained: bool = True


@dataclass
class CloudTrainingConfig:
    """云端训练配置"""
    # 基础训练参数
    num_epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 5e-4

    # 优化器
    optimizer: str = "adamw"
    scheduler: str = "cosine"

    # 设备
    device: str = "cuda"

    # 日志
    log_interval: int = 10
    save_interval: int = 10

    # 输出目录
    output_root: str = "/root/autodl-tmp/kd_visibility_claude/outputs"


@dataclass
class CloudExperimentConfig:
    """云端完整实验配置"""
    name: str = "kd_visibility_yolo"

    # 实验类型
    kd_branch: KDBranch = KDBranch.LOGIT_ONLY
    visibility: VisibilityLevel = VisibilityLevel.LIGHT

    # 子配置
    model: CloudModelConfig = field(default_factory=CloudModelConfig)
    data: CloudDataConfig = field(default_factory=CloudDataConfig)
    training: CloudTrainingConfig = field(default_factory=CloudTrainingConfig)

    def get_output_dir(self) -> str:
        """获取输出目录"""
        import os
        return os.path.join(
            self.training.output_root,
            self.name,
            self.kd_branch.value,
            self.visibility.value
        )

    def get_beta(self) -> float:
        """获取当前可见度对应的雾浓度参数"""
        return self.data.visibility_levels[self.visibility.value]


def create_cloud_experiment_matrix() -> List[CloudExperimentConfig]:
    """
    创建云端5×3实验矩阵配置
    """
    configs = []

    for branch in KDBranch:
        for vis in VisibilityLevel:
            config = CloudExperimentConfig(
                name="kd_visibility_yolo",
                kd_branch=branch,
                visibility=vis
            )
            configs.append(config)

    return configs
