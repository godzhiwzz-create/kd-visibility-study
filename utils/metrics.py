"""
语义分割评估指标
支持mIoU、像素精度等，以及机制分析指标
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
from scipy.stats import entropy as scipy_entropy


class SegmentationMetrics:
    """
    语义分割评估指标计算器
    """

    def __init__(self, num_classes: int = 19, ignore_index: int = 255):
        """
        初始化

        Args:
            num_classes: 类别数
            ignore_index: 忽略标签
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        """重置累计统计"""
        self.confusion_matrix = np.zeros(
            (self.num_classes, self.num_classes),
            dtype=np.int64
        )
        self.total_samples = 0

    def _compute_confusion_matrix(
        self,
        pred: np.ndarray,
        target: np.ndarray
    ) -> np.ndarray:
        """
        计算混淆矩阵

        Args:
            pred: [H, W] 预测标签
            target: [H, W] 真实标签

        Returns:
            [num_classes, num_classes] 混淆矩阵
        """
        mask = (target >= 0) & (target < self.num_classes)
        label = self.num_classes * target[mask].astype(np.int64) + pred[mask]
        count = np.bincount(
            label,
            minlength=self.num_classes ** 2
        )
        confusion_matrix = count.reshape(
            (self.num_classes, self.num_classes)
        )
        return confusion_matrix

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ):
        """
        更新统计

        Args:
            pred: [B, H, W] 或 [B, C, H, W] 预测结果
            target: [B, H, W] 真实标签
        """
        if pred.dim() == 4:
            # [B, C, H, W] -> [B, H, W]
            pred = pred.argmax(dim=1)

        pred = pred.cpu().numpy()
        target = target.cpu().numpy()

        for i in range(pred.shape[0]):
            # 忽略无效标签
            valid_mask = target[i] != self.ignore_index
            pred_i = pred[i][valid_mask]
            target_i = target[i][valid_mask]

            self.confusion_matrix += self._compute_confusion_matrix(pred_i, target_i)
            self.total_samples += 1

    def compute(self) -> Dict[str, float]:
        """
        计算所有指标

        Returns:
            指标字典
        """
        confusion_matrix = self.confusion_matrix

        # 像素精度
        pixel_acc = np.diag(confusion_matrix).sum() / (
            confusion_matrix.sum() + 1e-10
        )

        # 类别平均精度
        class_acc = np.diag(confusion_matrix) / (
            confusion_matrix.sum(axis=1) + 1e-10
        )
        mean_acc = np.nanmean(class_acc)

        # IoU
        intersection = np.diag(confusion_matrix)
        union = (
            confusion_matrix.sum(axis=1) +
            confusion_matrix.sum(axis=0) -
            intersection
        )
        iou = intersection / (union + 1e-10)
        mean_iou = np.nanmean(iou)

        # 频率加权IoU
        freq = confusion_matrix.sum(axis=1) / confusion_matrix.sum()
        freq_weighted_iou = (freq * iou).sum()

        return {
            'pixel_acc': float(pixel_acc),
            'mean_acc': float(mean_acc),
            'mean_iou': float(mean_iou),
            'freq_weighted_iou': float(freq_weighted_iou),
            'iou_per_class': iou.tolist()
        }


class UncertaintyMetrics:
    """
    不确定性分析指标
    用于机制M3：不确定性放大
    """

    @staticmethod
    def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
        """
        计算预测熵

        Args:
            logits: [B, C, H, W] 模型输出

        Returns:
            [B, H, W] 每个像素的熵
        """
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)

        # 熵 = -sum(p * log(p))
        entropy = -(probs * log_probs).sum(dim=1)

        return entropy

    @staticmethod
    def compute_confidence(logits: torch.Tensor) -> torch.Tensor:
        """
        计算预测置信度（最大概率）

        Args:
            logits: [B, C, H, W] 模型输出

        Returns:
            [B, H, W] 每个像素的置信度
        """
        probs = F.softmax(logits, dim=1)
        confidence = probs.max(dim=1)[0]
        return confidence

    @staticmethod
    def compute_predictive_entropy(
        logits: torch.Tensor,
        reduction: str = 'mean'
    ) -> float:
        """
        计算整体预测熵

        Args:
            logits: [B, C, H, W] 模型输出
            reduction: 降维方式 ('mean', 'sum', 'pixel')

        Returns:
            熵值
        """
        entropy = UncertaintyMetrics.compute_entropy(logits)

        if reduction == 'mean':
            return entropy.mean().item()
        elif reduction == 'sum':
            return entropy.sum().item()
        elif reduction == 'pixel':
            return entropy.mean(dim=0).cpu().numpy()
        else:
            raise ValueError(f"未知的reduction: {reduction}")

    @staticmethod
    def compute_expected_calibration_error(
        logits: torch.Tensor,
        targets: torch.Tensor,
        n_bins: int = 10
    ) -> float:
        """
        计算期望校准误差 (ECE)

        Args:
            logits: [B, C, H, W] 模型输出
            targets: [B, H, W] 真实标签
            n_bins: 分箱数

        Returns:
            ECE值
        """
        probs = F.softmax(logits, dim=1)
        confidences, predictions = probs.max(dim=1)

        # 过滤忽略标签
        valid_mask = targets != 255
        confidences = confidences[valid_mask].cpu().numpy()
        predictions = predictions[valid_mask].cpu().numpy()
        targets = targets[valid_mask].cpu().numpy()

        accuracies = (predictions == targets).astype(float)

        # 分箱
        bins = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bins[:-1]
        bin_uppers = bins[1:]

        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return float(ece)


class DistributionMetrics:
    """
    分布分析指标
    用于机制M1：分布错配
    """

    @staticmethod
    def compute_kl_divergence(
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        temperature: float = 1.0
    ) -> float:
        """
        计算KL散度

        Args:
            student_logits: [B, C, H, W]
            teacher_logits: [B, C, H, W]
            temperature: 温度参数

        Returns:
            KL散度
        """
        B, C, H, W = student_logits.shape

        # 调整形状
        s_logits = student_logits.permute(0, 2, 3, 1).reshape(-1, C)
        t_logits = teacher_logits.permute(0, 2, 3, 1).reshape(-1, C)

        # 计算概率
        s_log_probs = F.log_softmax(s_logits / temperature, dim=1)
        t_probs = F.softmax(t_logits / temperature, dim=1)

        # KL散度
        kl = F.kl_div(s_log_probs, t_probs, reduction='batchmean')

        return kl.item() * (temperature ** 2)

    @staticmethod
    def compute_js_divergence(
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor
    ) -> float:
        """
        计算JS散度

        Args:
            student_logits: [B, C, H, W]
            teacher_logits: [B, C, H, W]

        Returns:
            JS散度
        """
        B, C, H, W = student_logits.shape

        # 调整形状
        s_logits = student_logits.permute(0, 2, 3, 1).reshape(-1, C)
        t_logits = teacher_logits.permute(0, 2, 3, 1).reshape(-1, C)

        # 计算概率
        s_probs = F.softmax(s_logits, dim=1)
        t_probs = F.softmax(t_logits, dim=1)

        # M = 0.5 * (P + Q)
        m_probs = 0.5 * (s_probs + t_probs)

        # JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
        kl_pm = F.kl_div(m_probs.log(), s_probs, reduction='batchmean')
        kl_qm = F.kl_div(m_probs.log(), t_probs, reduction='batchmean')

        js = 0.5 * (kl_pm + kl_qm)

        return js.item()

    @staticmethod
    def compute_total_variation(
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor
    ) -> float:
        """
        计算总变差距离

        Args:
            student_logits: [B, C, H, W]
            teacher_logits: [B, C, H, W]

        Returns:
            总变差距离
        """
        B, C, H, W = student_logits.shape

        s_probs = F.softmax(student_logits, dim=1)
        t_probs = F.softmax(teacher_logits, dim=1)

        # TV = 0.5 * sum(|P - Q|)
        tv = 0.5 * torch.abs(s_probs - t_probs).sum(dim=1).mean()

        return tv.item()


class RepresentationMetrics:
    """
    表征分析指标
    用于机制M2：表征对齐失败
    """

    @staticmethod
    def compute_feature_l2_distance(
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
        normalize: bool = True
    ) -> float:
        """
        计算特征L2距离

        Args:
            student_features: [B, C, H, W]
            teacher_features: [B, C, H, W]
            normalize: 是否归一化

        Returns:
            L2距离
        """
        if normalize:
            s_feat = F.normalize(student_features.flatten(1), dim=1)
            t_feat = F.normalize(teacher_features.flatten(1), dim=1)
        else:
            s_feat = student_features.flatten(1)
            t_feat = teacher_features.flatten(1)

        l2_dist = torch.norm(s_feat - t_feat, p=2, dim=1).mean()
        return l2_dist.item()

    @staticmethod
    def compute_cosine_similarity(
        student_features: torch.Tensor,
        teacher_features: torch.Tensor
    ) -> float:
        """
        计算特征余弦相似度

        Args:
            student_features: [B, C, H, W]
            teacher_features: [B, C, H, W]

        Returns:
            余弦相似度
        """
        s_feat = student_features.flatten(1)
        t_feat = teacher_features.flatten(1)

        cos_sim = F.cosine_similarity(s_feat, t_feat, dim=1).mean()
        return cos_sim.item()

    @staticmethod
    def compute_cka(
        student_features: torch.Tensor,
        teacher_features: torch.Tensor
    ) -> float:
        """
        计算Centered Kernel Alignment (CKA)
        用于衡量特征表示的相似性

        Args:
            student_features: [B, C, H, W]
            teacher_features: [B, C, H, W]

        Returns:
            CKA值 [0, 1]
        """
        # 展平特征
        X = student_features.flatten(1).cpu().numpy()  # [B, D_s]
        Y = teacher_features.flatten(1).cpu().numpy()  # [B, D_t]

        # 中心化
        X = X - X.mean(axis=0, keepdims=True)
        Y = Y - Y.mean(axis=0, keepdims=True)

        # 计算核矩阵
        K = X @ X.T
        L = Y @ Y.T

        # 计算CKA
        hsic = np.trace(K @ L)
        norm_k = np.trace(K @ K)
        norm_l = np.trace(L @ L)

        cka = hsic / (np.sqrt(norm_k * norm_l) + 1e-10)

        return float(cka)


class MechanismAnalyzer:
    """
    机制分析器
    整合所有机制相关指标
    """

    def __init__(self, num_classes: int = 19):
        self.num_classes = num_classes
        self.dist_metrics = DistributionMetrics()
        self.repr_metrics = RepresentationMetrics()
        self.unc_metrics = UncertaintyMetrics()

    def analyze_batch(
        self,
        student_output: Dict[str, torch.Tensor],
        teacher_output: Dict[str, torch.Tensor],
        features_to_analyze: List[str] = None
    ) -> Dict[str, float]:
        """
        分析一个batch的机制指标

        Args:
            student_output: Student输出
            teacher_output: Teacher输出
            features_to_analyze: 要分析的特征层

        Returns:
            指标字典
        """
        features_to_analyze = features_to_analyze or ['layer3', 'layer4']
        metrics = {}

        # M1: 分布错配指标
        if 'logits' in student_output and 'logits' in teacher_output:
            s_logits = student_output['logits']
            t_logits = teacher_output['logits']

            metrics['kl_div'] = self.dist_metrics.compute_kl_divergence(s_logits, t_logits)
            metrics['js_div'] = self.dist_metrics.compute_js_divergence(s_logits, t_logits)
            metrics['total_variation'] = self.dist_metrics.compute_total_variation(s_logits, t_logits)

        # M2: 表征对齐指标
        s_features = student_output.get('features', {})
        t_features = teacher_output.get('features', {})

        for layer in features_to_analyze:
            if layer in s_features and layer in t_features:
                s_feat = s_features[layer]
                t_feat = t_features[layer]

                metrics[f'{layer}_l2_dist'] = self.repr_metrics.compute_feature_l2_distance(s_feat, t_feat)
                metrics[f'{layer}_cos_sim'] = self.repr_metrics.compute_cosine_similarity(s_feat, t_feat)

                # CKA计算较慢，可以采样计算
                if s_feat.shape[0] <= 8:  # 只在batch较小时计算
                    metrics[f'{layer}_cka'] = self.repr_metrics.compute_cka(s_feat, t_feat)

        # M3: 不确定性指标
        if 'logits' in teacher_output:
            t_logits = teacher_output['logits']
            metrics['teacher_entropy'] = self.unc_metrics.compute_predictive_entropy(t_logits)
            metrics['teacher_confidence'] = self.unc_metrics.compute_confidence(t_logits).mean().item()

        if 'logits' in student_output:
            s_logits = student_output['logits']
            metrics['student_entropy'] = self.unc_metrics.compute_predictive_entropy(s_logits)
            metrics['student_confidence'] = self.unc_metrics.compute_confidence(s_logits).mean().item()

        return metrics
