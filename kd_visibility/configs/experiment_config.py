"""
知识蒸馏在可见度退化下的失效机制研究 - 配置文件
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class KDBranch(Enum):
    """KD分支类型"""
    STUDENT_ONLY = "student_only"      # 基线，无蒸馏
    LOGIT_ONLY = "logit_only"          # 仅logit蒸馏
    FEATURE_ONLY = "feature_only"      # 仅feature蒸馏
    ATTENTION_ONLY = "attention_only"  # 仅attention蒸馏
    LOCALIZATION_ONLY = "localization_only"  # 仅localization蒸馏


class VisibilityLevel(Enum):
    """可见度等级"""
    LIGHT = "light"        # 轻度退化
    MODERATE = "moderate"  # 中度退化
    HEAVY = "heavy"        # 重度退化


@dataclass
class ModelConfig:
    """模型配置"""
    # Teacher模型
    teacher_backbone: str = "resnet50"
    teacher_pretrained: bool = True
    teacher_checkpoint: Optional[str] = None  # 预训练权重路径

    # Student模型
    student_backbone: str = "resnet18"
    student_pretrained: bool = True


@dataclass
class DataConfig:
    """数据配置"""
    # 数据路径
    clear_root: str = "数据集/cityscapes/clear"
    foggy_root: str = "数据集/cityscapes/foggy"
    annotation_root: str = "数据集/cityscapes/annotations/gtFine"
    filename_list: str = "数据集/cityscapes/foggy_trainval_refined_filenames.txt"

    # 可见度配置（雾浓度）
    visibility_levels: Dict[str, float] = field(default_factory=lambda: {
        "light": 0.005,      # beta参数，越小雾越轻
        "moderate": 0.01,
        "heavy": 0.02
    })

    # 数据增强
    image_size: tuple = (512, 1024)
    batch_size: int = 4
    num_workers: int = 4


@dataclass
class KDLossConfig:
    """KD损失配置"""
    # Logit蒸馏
    logit_weight: float = 1.0
    temperature: float = 4.0

    # Feature蒸馏
    feature_weight: float = 1.0
    feature_layers: List[str] = field(default_factory=lambda: ["layer1", "layer2", "layer3", "layer4"])

    # Attention蒸馏
    attention_weight: float = 1.0

    # Localization蒸馏（用于目标检测）
    localization_weight: float = 1.0


@dataclass
class TrainingConfig:
    """训练配置"""
    # 基础训练参数
    num_epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 1e-4

    # 优化器
    optimizer: str = "adamw"
    scheduler: str = "cosine"

    # 设备
    device: str = "cuda"

    # 日志
    log_interval: int = 10
    save_interval: int = 10

    # 输出目录
    output_root: str = "outputs"


@dataclass
class ExperimentConfig:
    """完整实验配置"""
    name: str = "kd_visibility_experiment"

    # 实验类型
    kd_branch: KDBranch = KDBranch.LOGIT_ONLY
    visibility: VisibilityLevel = VisibilityLevel.LIGHT

    # 子配置
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    kd_loss: KDLossConfig = field(default_factory=KDLossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

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


def create_experiment_matrix() -> List[ExperimentConfig]:
    """
    创建5×3实验矩阵配置

    Returns:
        15个实验配置的列表
    """
    configs = []

    for branch in KDBranch:
        for vis in VisibilityLevel:
            config = ExperimentConfig(
                name="kd_visibility_study",
                kd_branch=branch,
                visibility=vis
            )
            configs.append(config)

    return configs
