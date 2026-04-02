"""
语义分割模型定义
支持不同的backbone和分割头
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import List, Dict, Tuple, Optional


class SegmentationBackbone(nn.Module):
    """
    分割网络Backbone
    支持ResNet18/50/101等
    """

    def __init__(
        self,
        arch: str = "resnet50",
        pretrained: bool = True,
        return_layers: List[str] = None
    ):
        """
        初始化Backbone

        Args:
            arch: 网络架构 (resnet18/resnet50/resnet101)
            pretrained: 是否使用预训练权重
            return_layers: 要返回特征图的层名列表
        """
        super().__init__()
        self.return_layers = return_layers or ["layer4"]

        # 加载预训练模型
        if arch == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = models.resnet18(weights=weights)
        elif arch == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = models.resnet50(weights=weights)
        elif arch == "resnet101":
            weights = models.ResNet101_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = models.resnet101(weights=weights)
        else:
            raise ValueError(f"不支持的架构: {arch}")

        # 提取各层
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # 特征通道数
        self.channels = {
            "layer1": 64 if arch == "resnet18" else 256,
            "layer2": 128 if arch == "resnet18" else 512,
            "layer3": 256 if arch == "resnet18" else 1024,
            "layer4": 512 if arch == "resnet18" else 2048,
        }

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播，返回指定层的特征"""
        features = {}

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        if "layer1" in self.return_layers:
            features["layer1"] = x

        x = self.layer2(x)
        if "layer2" in self.return_layers:
            features["layer2"] = x

        x = self.layer3(x)
        if "layer3" in self.return_layers:
            features["layer3"] = x

        x = self.layer4(x)
        if "layer4" in self.return_layers:
            features["layer4"] = x

        return features


class FPNHead(nn.Module):
    """
    Feature Pyramid Network分割头
    """

    def __init__(
        self,
        in_channels: List[int],
        num_classes: int,
        feature_dim: int = 256
    ):
        """
        初始化FPN头

        Args:
            in_channels: 输入特征通道数列表
            num_classes: 输出类别数
            feature_dim: 中间特征维度
        """
        super().__init__()

        #  lateral connections
        self.lateral_convs = nn.ModuleList()
        for in_ch in in_channels:
            self.lateral_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, feature_dim, 1),
                    nn.BatchNorm2d(feature_dim),
                    nn.ReLU(inplace=True)
                )
            )

        # 输出卷积
        self.output_conv = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(feature_dim, num_classes, 1)
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """前向传播"""
        # 自顶向下路径
        laterals = []
        for i, feat in enumerate(features):
            lateral = self.lateral_convs[i](feat)
            laterals.append(lateral)

        # 上采样并融合
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i],
                size=laterals[i - 1].shape[2:],
                mode='bilinear',
                align_corners=False
            )

        # 使用最高分辨率特征
        output = self.output_conv(laterals[0])
        return output


class SimpleSegmentationHead(nn.Module):
    """
    简单分割头：仅使用最后一层特征
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        feature_dim: int = 256
    ):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, feature_dim, 3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),

            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),

            nn.Conv2d(feature_dim, num_classes, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


class SegmentationModel(nn.Module):
    """
    完整的语义分割模型
    支持Teacher和Student配置
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        num_classes: int = 19,
        pretrained: bool = True,
        use_fpn: bool = False,
        return_features: bool = False,
        feature_layers: Optional[List[str]] = None
    ):
        """
        初始化分割模型

        Args:
            backbone: backbone架构
            num_classes: 类别数
            pretrained: 是否使用预训练权重
            use_fpn: 是否使用FPN分割头
            return_features: 是否返回中间特征（用于KD）
            feature_layers: 要返回的特征层
        """
        super().__init__()

        self.return_features = return_features
        self.feature_layers = feature_layers or ["layer4"]

        # Backbone
        self.backbone = SegmentationBackbone(
            arch=backbone,
            pretrained=pretrained,
            return_layers=self.feature_layers if return_features else ["layer4"]
        )

        # 分割头
        if use_fpn and len(self.feature_layers) > 1:
            in_channels = [self.backbone.channels[l] for l in self.feature_layers]
            self.seg_head = FPNHead(in_channels, num_classes)
        else:
            in_ch = self.backbone.channels["layer4"]
            self.seg_head = SimpleSegmentationHead(in_ch, num_classes)

        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Returns:
            包含以下键的字典:
            - 'logits': 分割logits [B, num_classes, H, W]
            - 'features': 中间特征字典（如果return_features=True）
        """
        # 获取特征
        features = self.backbone(x)

        # 分割预测
        if isinstance(self.seg_head, FPNHead):
            feature_list = [features[l] for l in self.feature_layers]
            logits = self.seg_head(feature_list)
        else:
            logits = self.seg_head(features["layer4"])

        # 上采样到原图尺寸
        logits = F.interpolate(
            logits,
            size=x.shape[2:],
            mode='bilinear',
            align_corners=True
        )

        output = {'logits': logits}

        if self.return_features:
            output['features'] = features

        return output


class TeacherStudentPair(nn.Module):
    """
    Teacher-Student模型对
    用于知识蒸馏训练
    """

    def __init__(
        self,
        teacher_config: Dict,
        student_config: Dict,
        num_classes: int = 19
    ):
        """
        初始化Teacher-Student对

        Args:
            teacher_config: Teacher模型配置
            student_config: Student模型配置
            num_classes: 类别数
        """
        super().__init__()

        # Teacher模型（清晰图像，预训练）
        self.teacher = SegmentationModel(
            backbone=teacher_config['backbone'],
            num_classes=num_classes,
            pretrained=teacher_config.get('pretrained', True),
            return_features=True,
            feature_layers=['layer1', 'layer2', 'layer3', 'layer4']
        )

        # 加载Teacher预训练权重
        if 'checkpoint' in teacher_config and teacher_config['checkpoint']:
            self._load_teacher_checkpoint(teacher_config['checkpoint'])

        # Student模型（雾天图像）
        self.student = SegmentationModel(
            backbone=student_config['backbone'],
            num_classes=num_classes,
            pretrained=student_config.get('pretrained', True),
            return_features=True,
            feature_layers=['layer1', 'layer2', 'layer3', 'layer4']
        )

        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

    def _load_teacher_checkpoint(self, checkpoint_path: str):
        """加载Teacher预训练权重"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        self.teacher.load_state_dict(state_dict, strict=False)
        print(f"已加载Teacher权重: {checkpoint_path}")

    @torch.no_grad()
    def get_teacher_output(self, clear_images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """获取Teacher输出（清晰图像）"""
        self.teacher.eval()
        return self.teacher(clear_images)

    def get_student_output(self, foggy_images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """获取Student输出（雾天图像）"""
        return self.student(foggy_images)

    def forward(
        self,
        clear_images: torch.Tensor,
        foggy_images: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Returns:
            包含teacher和student输出的字典
        """
        with torch.no_grad():
            teacher_output = self.get_teacher_output(clear_images)

        student_output = self.get_student_output(foggy_images)

        return {
            'teacher_logits': teacher_output['logits'],
            'teacher_features': teacher_output.get('features', {}),
            'student_logits': student_output['logits'],
            'student_features': student_output.get('features', {})
        }
