"""
知识蒸馏损失函数
支持5种KD分支：logit, feature, attention, localization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math


class LogitDistillationLoss(nn.Module):
    """
    Logit蒸馏损失（软标签蒸馏）
    使用KL散度匹配teacher和student的输出分布
    """

    def __init__(self, temperature: float = 4.0, weight: float = 1.0):
        """
        初始化

        Args:
            temperature: 温度参数，软化概率分布
            weight: 损失权重
        """
        super().__init__()
        self.temperature = temperature
        self.weight = weight
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        计算logit蒸馏损失

        Args:
            student_logits: [B, C, H, W] student输出
            teacher_logits: [B, C, H, W] teacher输出

        Returns:
            蒸馏损失
        """
        # 调整形状为 [B*H*W, C]
        B, C, H, W = student_logits.shape
        s_logits = student_logits.permute(0, 2, 3, 1).reshape(-1, C)
        t_logits = teacher_logits.permute(0, 2, 3, 1).reshape(-1, C)

        # 应用温度
        s_probs = F.log_softmax(s_logits / self.temperature, dim=1)
        t_probs = F.softmax(t_logits / self.temperature, dim=1)

        # KL散度
        loss = self.kl_div(s_probs, t_probs) * (self.temperature ** 2)

        return loss * self.weight

    def compute_kl_divergence(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor
    ) -> float:
        """
        计算KL散度值（用于分析，不参与训练）

        Returns:
            KL散度标量值
        """
        with torch.no_grad():
            B, C, H, W = student_logits.shape
            s_logits = student_logits.permute(0, 2, 3, 1).reshape(-1, C)
            t_logits = teacher_logits.permute(0, 2, 3, 1).reshape(-1, C)

            s_probs = F.log_softmax(s_logits, dim=1)
            t_probs = F.softmax(t_logits, dim=1)

            kl = F.kl_div(s_probs, t_probs, reduction='batchmean')

        return kl.item()


class FeatureDistillationLoss(nn.Module):
    """
    Feature蒸馏损失
    匹配teacher和student的中间层特征
    """

    def __init__(
        self,
        layers: List[str] = None,
        weight: float = 1.0,
        loss_type: str = "l2"  # l2, cosine
    ):
        """
        初始化

        Args:
            layers: 要蒸馏的特征层列表
            weight: 损失权重
            loss_type: 损失类型 (l2, cosine)
        """
        super().__init__()
        self.layers = layers or ["layer3", "layer4"]
        self.weight = weight
        self.loss_type = loss_type

        # 用于对齐维度的1x1卷积
        self.adaptation_convs = nn.ModuleDict()

    def _get_adaptation_conv(
        self,
        layer_name: str,
        s_channels: int,
        t_channels: int
    ) -> nn.Module:
        """获取或创建维度适配卷积"""
        key = f"{layer_name}_{s_channels}_{t_channels}"
        if key not in self.adaptation_convs:
            self.adaptation_convs[key] = nn.Conv2d(
                s_channels, t_channels, 1, bias=False
            )
        return self.adaptation_convs[key]

    def forward(
        self,
        student_features: Dict[str, torch.Tensor],
        teacher_features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        计算feature蒸馏损失

        Args:
            student_features: Student特征字典
            teacher_features: Teacher特征字典

        Returns:
            蒸馏损失
        """
        total_loss = 0.0

        for layer in self.layers:
            if layer not in student_features or layer not in teacher_features:
                continue

            s_feat = student_features[layer]
            t_feat = teacher_features[layer]

            # 维度对齐
            if s_feat.shape[1] != t_feat.shape[1]:
                adapt_conv = self._get_adaptation_conv(
                    layer, s_feat.shape[1], t_feat.shape[1]
                )
                s_feat = adapt_conv(s_feat)

            # 空间尺寸对齐
            if s_feat.shape[2:] != t_feat.shape[2:]:
                s_feat = F.interpolate(
                    s_feat,
                    size=t_feat.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )

            # 计算损失
            if self.loss_type == "l2":
                loss = F.mse_loss(s_feat, t_feat.detach())
            elif self.loss_type == "cosine":
                s_feat = s_feat.flatten(2)
                t_feat = t_feat.flatten(2)
                loss = 1 - F.cosine_similarity(
                    s_feat, t_feat.detach(), dim=1
                ).mean()
            else:
                raise ValueError(f"未知的损失类型: {self.loss_type}")

            total_loss += loss

        return total_loss * self.weight / len(self.layers)

    def compute_feature_distance(
        self,
        student_features: Dict[str, torch.Tensor],
        teacher_features: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        计算特征距离（用于分析）

        Returns:
            每层特征距离的字典
        """
        distances = {}

        with torch.no_grad():
            for layer in self.layers:
                if layer not in student_features or layer not in teacher_features:
                    continue

                s_feat = student_features[layer]
                t_feat = teacher_features[layer]

                # L2距离
                l2_dist = F.mse_loss(s_feat, t_feat, reduction='mean').item()

                # 余弦相似度
                s_feat_flat = s_feat.flatten(1)
                t_feat_flat = t_feat.flatten(1)
                cos_sim = F.cosine_similarity(
                    s_feat_flat, t_feat_flat, dim=1
                ).mean().item()

                distances[layer] = {
                    'l2': l2_dist,
                    'cosine': cos_sim
                }

        return distances


class AttentionDistillationLoss(nn.Module):
    """
    Attention蒸馏损失
    基于特征图计算的注意力图进行蒸馏
    """

    def __init__(
        self,
        layers: List[str] = None,
        weight: float = 1.0,
        attention_type: str = "sum"  # sum, max
    ):
        """
        初始化

        Args:
            layers: 要蒸馏的特征层
            weight: 损失权重
            attention_type: 注意力计算方式
        """
        super().__init__()
        self.layers = layers or ["layer3", "layer4"]
        self.weight = weight
        self.attention_type = attention_type

    def _compute_attention(
        self,
        features: torch.Tensor
    ) -> torch.Tensor:
        """从特征计算注意力图"""
        if self.attention_type == "sum":
            # 通道维度求和
            attention = torch.sum(torch.abs(features), dim=1, keepdim=True)
        elif self.attention_type == "max":
            # 通道维度最大值
            attention = torch.max(torch.abs(features), dim=1, keepdim=True)[0]
        else:
            raise ValueError(f"未知的注意力类型: {self.attention_type}")

        # 归一化
        B, _, H, W = attention.shape
        attention = attention.view(B, -1)
        attention = attention / (attention.sum(dim=1, keepdim=True) + 1e-8)
        attention = attention.view(B, 1, H, W)

        return attention

    def forward(
        self,
        student_features: Dict[str, torch.Tensor],
        teacher_features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        计算attention蒸馏损失

        Args:
            student_features: Student特征字典
            teacher_features: Teacher特征字典

        Returns:
            蒸馏损失
        """
        total_loss = 0.0

        for layer in self.layers:
            if layer not in student_features or layer not in teacher_features:
                continue

            s_feat = student_features[layer]
            t_feat = teacher_features[layer]

            # 计算注意力图
            s_att = self._compute_attention(s_feat)
            t_att = self._compute_attention(t_feat)

            # 空间尺寸对齐
            if s_att.shape[2:] != t_att.shape[2:]:
                s_att = F.interpolate(
                    s_att,
                    size=t_att.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )

            # 使用KL散度
            s_att = s_att.flatten(1) + 1e-8
            t_att = t_att.flatten(1) + 1e-8

            loss = F.kl_div(
                s_att.log(),
                t_att.detach(),
                reduction='batchmean'
            )

            total_loss += loss

        return total_loss * self.weight / len(self.layers)


class SegmentationLoss(nn.Module):
    """
    语义分割基础损失（硬标签）
    使用交叉熵损失
    """

    def __init__(self, ignore_index: int = 255, weight: float = 1.0):
        super().__init__()
        self.weight = weight
        self.ce = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            reduction='mean'
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        计算分割损失

        Args:
            logits: [B, C, H, W] 预测logits
            targets: [B, H, W] 目标标签

        Returns:
            交叉熵损失
        """
        return self.ce(logits, targets) * self.weight


class CombinedKDLoss(nn.Module):
    """
    组合KD损失
    根据KD分支类型选择激活不同的损失组件
    """

    def __init__(
        self,
        kd_branch: str,
        kd_config: Dict,
        num_classes: int = 19,
        ignore_index: int = 255
    ):
        """
        初始化组合损失

        Args:
            kd_branch: KD分支类型
                - student_only: 仅硬标签
                - logit_only: 硬标签 + logit蒸馏
                - feature_only: 硬标签 + feature蒸馏
                - attention_only: 硬标签 + attention蒸馏
                - localization_only: 硬标签 + localization蒸馏
            kd_config: KD损失配置
            num_classes: 类别数
            ignore_index: 忽略标签
        """
        super().__init__()

        self.kd_branch = kd_branch

        # 基础分割损失（所有分支都有）
        self.seg_loss = SegmentationLoss(
            ignore_index=ignore_index,
            weight=1.0
        )

        # 根据分支初始化不同的蒸馏损失
        self.logit_loss = None
        self.feature_loss = None
        self.attention_loss = None

        if kd_branch == "logit_only" or kd_branch == "logit":
            self.logit_loss = LogitDistillationLoss(
                temperature=kd_config.get('temperature', 4.0),
                weight=kd_config.get('logit_weight', 1.0)
            )

        elif kd_branch == "feature_only" or kd_branch == "feature":
            self.feature_loss = FeatureDistillationLoss(
                layers=kd_config.get('feature_layers', ['layer3', 'layer4']),
                weight=kd_config.get('feature_weight', 1.0),
                loss_type='l2'
            )

        elif kd_branch == "attention_only" or kd_branch == "attention":
            self.attention_loss = AttentionDistillationLoss(
                layers=kd_config.get('attention_layers', ['layer3', 'layer4']),
                weight=kd_config.get('attention_weight', 1.0)
            )

        elif kd_branch == "localization_only" or kd_branch == "localization":
            # 对于语义分割，localization可以解释为边界关注
            # 使用边界检测的feature蒸馏
            self.feature_loss = FeatureDistillationLoss(
                layers=['layer4'],  # 高层特征包含定位信息
                weight=kd_config.get('localization_weight', 1.0),
                loss_type='l2'
            )

        elif kd_branch == "student_only" or kd_branch == "baseline":
            # 基线，无蒸馏
            pass

        else:
            raise ValueError(f"未知的KD分支: {kd_branch}")

    def forward(
        self,
        student_output: Dict[str, torch.Tensor],
        teacher_output: Dict[str, torch.Tensor],
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算总损失

        Args:
            student_output: Student模型输出
            teacher_output: Teacher模型输出
            targets: 真实标签

        Returns:
            total_loss: 总损失
            loss_dict: 各损失分量的字典
        """
        loss_dict = {}

        # 基础分割损失
        seg_loss = self.seg_loss(student_output['logits'], targets)
        loss_dict['seg'] = seg_loss.item()

        total_loss = seg_loss

        # 根据分支添加蒸馏损失
        if self.logit_loss is not None:
            logit_loss = self.logit_loss(
                student_output['logits'],
                teacher_output['logits']
            )
            total_loss = total_loss + logit_loss
            loss_dict['kd_logit'] = logit_loss.item()

        if self.feature_loss is not None:
            feat_loss = self.feature_loss(
                student_output.get('features', {}),
                teacher_output.get('features', {})
            )
            total_loss = total_loss + feat_loss
            loss_dict['kd_feature'] = feat_loss.item()

        if self.attention_loss is not None:
            att_loss = self.attention_loss(
                student_output.get('features', {}),
                teacher_output.get('features', {})
            )
            total_loss = total_loss + att_loss
            loss_dict['kd_attention'] = att_loss.item()

        loss_dict['total'] = total_loss.item()

        return total_loss, loss_dict

    @torch.no_grad()
    def compute_analysis_metrics(
        self,
        student_output: Dict[str, torch.Tensor],
        teacher_output: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        计算分析用的指标（不用于训练）

        Returns:
            指标字典
        """
        metrics = {}

        # KL散度
        if 'logits' in student_output and 'logits' in teacher_output:
            logit_loss = LogitDistillationLoss(temperature=1.0)
            metrics['kl_div'] = logit_loss.compute_kl_divergence(
                student_output['logits'],
                teacher_output['logits']
            )

        # JS散度
            B, C, H, W = student_output['logits'].shape
            s_logits = student_output['logits'].permute(0, 2, 3, 1).reshape(-1, C)
            t_logits = teacher_output['logits'].permute(0, 2, 3, 1).reshape(-1, C)

            s_probs = F.softmax(s_logits, dim=1)
            t_probs = F.softmax(t_logits, dim=1)
            m_probs = 0.5 * (s_probs + t_probs)

            js_div = 0.5 * (
                F.kl_div(m_probs.log(), s_probs, reduction='batchmean') +
                F.kl_div(m_probs.log(), t_probs, reduction='batchmean')
            )
            metrics['js_div'] = js_div.item()

        # 特征距离
        if self.feature_loss is not None:
            feat_dists = self.feature_loss.compute_feature_distance(
                student_output.get('features', {}),
                teacher_output.get('features', {})
            )
            for layer, dists in feat_dists.items():
                for metric, value in dists.items():
                    metrics[f'feat_{layer}_{metric}'] = value

        return metrics
