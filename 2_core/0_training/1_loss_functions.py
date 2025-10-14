"""
自定义损失函数集合

提供各种训练场景下的损失函数实现。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """
    Focal Loss实现，用于解决类别不平衡问题
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        """
        初始化Focal Loss
        
        Args:
            alpha: 平衡因子
            gamma: 调节因子
            reduction: 归约方式
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            inputs: 预测logits
            targets: 真实标签
            
        Returns:
            计算的损失
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingLoss(nn.Module):
    """
    标签平滑损失函数
    """
    
    def __init__(self, smoothing: float = 0.1, reduction: str = 'mean'):
        """
        初始化标签平滑损失
        
        Args:
            smoothing: 平滑系数
            reduction: 归约方式
        """
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            inputs: 预测logits
            targets: 真实标签
            
        Returns:
            计算的损失
        """
        log_prob = F.log_softmax(inputs, dim=-1)
        nll_loss = F.nll_loss(log_prob, targets, reduction='none')
        
        smooth_loss = -log_prob.mean(dim=-1)
        loss = (1 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class ContrastiveLoss(nn.Module):
    """
    对比学习损失函数
    """
    
    def __init__(self, temperature: float = 0.07, reduction: str = 'mean'):
        """
        初始化对比损失
        
        Args:
            temperature: 温度参数
            reduction: 归约方式
        """
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.reduction = reduction
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            features: 特征向量
            labels: 标签
            
        Returns:
            对比损失
        """
        device = features.device
        batch_size = features.shape[0]
        
        # 归一化特征
        features = F.normalize(features, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # 创建标签mask
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # 移除对角线
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # 计算损失
        exp_logits = torch.exp(similarity_matrix) * logits_mask
        log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True))
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = -mean_log_prob_pos
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class RankingLoss(nn.Module):
    """
    排序损失函数，用于偏好学习
    """
    
    def __init__(self, margin: float = 1.0, reduction: str = 'mean'):
        """
        初始化排序损失
        
        Args:
            margin: 边界值
            reduction: 归约方式
        """
        super(RankingLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction
    
    def forward(self, positive_scores: torch.Tensor, negative_scores: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            positive_scores: 正样本得分
            negative_scores: 负样本得分
            
        Returns:
            排序损失
        """
        loss = F.relu(self.margin - positive_scores + negative_scores)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class KLDivergenceLoss(nn.Module):
    """
    KL散度损失，用于知识蒸馏
    """
    
    def __init__(self, temperature: float = 3.0, alpha: float = 0.5, reduction: str = 'mean'):
        """
        初始化KL散度损失
        
        Args:
            temperature: 蒸馏温度
            alpha: 平衡因子
            reduction: 归约方式
        """
        super(KLDivergenceLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.reduction = reduction
        self.kl_div = nn.KLDivLoss(reduction='none')
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, 
                targets: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            student_logits: 学生模型logits
            teacher_logits: 教师模型logits
            targets: 真实标签
            
        Returns:
            蒸馏损失
        """
        # 软标签损失
        student_log_prob = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_prob = F.softmax(teacher_logits / self.temperature, dim=-1)
        kl_loss = self.kl_div(student_log_prob, teacher_prob).sum(dim=-1)
        kl_loss = kl_loss * (self.temperature ** 2)
        
        # 硬标签损失
        ce_loss = self.ce_loss(student_logits, targets)
        
        # 组合损失
        total_loss = self.alpha * kl_loss + (1 - self.alpha) * ce_loss
        
        if self.reduction == 'mean':
            return total_loss.mean()
        elif self.reduction == 'sum':
            return total_loss.sum()
        else:
            return total_loss


class AdaptiveLoss(nn.Module):
    """
    自适应损失函数，根据训练阶段调整损失权重
    """
    
    def __init__(self, base_loss: nn.Module, warmup_steps: int = 1000):
        """
        初始化自适应损失
        
        Args:
            base_loss: 基础损失函数
            warmup_steps: 预热步数
        """
        super(AdaptiveLoss, self).__init__()
        self.base_loss = base_loss
        self.warmup_steps = warmup_steps
        self.step_count = 0
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        前向传播
        
        Args:
            *args: 传递给基础损失的参数
            **kwargs: 传递给基础损失的关键字参数
            
        Returns:
            自适应损失
        """
        loss = self.base_loss(*args, **kwargs)
        
        # 计算权重
        if self.step_count < self.warmup_steps:
            weight = self.step_count / self.warmup_steps
        else:
            weight = 1.0
        
        self.step_count += 1
        return loss * weight
    
    def reset_step_count(self):
        """重置步数计数器"""
        self.step_count = 0


class MultiTaskLoss(nn.Module):
    """
    多任务损失函数
    """
    
    def __init__(self, task_weights: Dict[str, float], loss_functions: Dict[str, nn.Module]):
        """
        初始化多任务损失
        
        Args:
            task_weights: 任务权重字典
            loss_functions: 损失函数字典
        """
        super(MultiTaskLoss, self).__init__()
        self.task_weights = task_weights
        self.loss_functions = nn.ModuleDict(loss_functions)
        
        # 验证任务名称一致性
        assert set(task_weights.keys()) == set(loss_functions.keys()), \
            "任务权重和损失函数的任务名称必须一致"
    
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向传播
        
        Args:
            predictions: 预测结果字典
            targets: 目标标签字典
            
        Returns:
            总损失和各任务损失
        """
        total_loss = 0.0
        task_losses = {}
        
        for task_name, weight in self.task_weights.items():
            if task_name in predictions and task_name in targets:
                loss_fn = self.loss_functions[task_name]
                task_loss = loss_fn(predictions[task_name], targets[task_name])
                weighted_loss = weight * task_loss
                
                total_loss += weighted_loss
                task_losses[task_name] = task_loss.detach()
        
        return total_loss, task_losses


class SequentialLoss(nn.Module):
    """
    序列生成任务的损失函数
    """
    
    def __init__(self, ignore_index: int = -100, label_smoothing: float = 0.0):
        """
        初始化序列损失
        
        Args:
            ignore_index: 忽略的索引
            label_smoothing: 标签平滑系数
        """
        super(SequentialLoss, self).__init__()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        
        if label_smoothing > 0:
            self.loss_fn = LabelSmoothingLoss(smoothing=label_smoothing)
        else:
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            logits: 模型输出logits
            labels: 真实标签
            
        Returns:
            序列损失
        """
        # 重塑张量以适应损失函数
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # 展平
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)
        
        # 过滤忽略的标签
        if self.ignore_index != -100:
            valid_mask = flat_labels != self.ignore_index
            if valid_mask.sum() == 0:
                return torch.tensor(0.0, device=logits.device, requires_grad=True)
            
            flat_logits = flat_logits[valid_mask]
            flat_labels = flat_labels[valid_mask]
        
        return self.loss_fn(flat_logits, flat_labels)


# 损失函数工厂
class LossFactory:
    """损失函数工厂类"""
    
    @staticmethod
    def create_loss(loss_type: str, **kwargs) -> nn.Module:
        """
        创建损失函数
        
        Args:
            loss_type: 损失类型
            **kwargs: 损失函数参数
            
        Returns:
            损失函数实例
        """
        loss_types = {
            'cross_entropy': nn.CrossEntropyLoss,
            'mse': nn.MSELoss,
            'l1': nn.L1Loss,
            'focal': FocalLoss,
            'label_smoothing': LabelSmoothingLoss,
            'contrastive': ContrastiveLoss,
            'ranking': RankingLoss,
            'kl_divergence': KLDivergenceLoss,
            'sequential': SequentialLoss
        }
        
        if loss_type not in loss_types:
            raise ValueError(f"不支持的损失类型: {loss_type}")
        
        return loss_types[loss_type](**kwargs)
    
    @staticmethod
    def create_multi_task_loss(task_config: Dict[str, Dict[str, Any]]) -> MultiTaskLoss:
        """
        创建多任务损失
        
        Args:
            task_config: 任务配置字典
            
        Returns:
            多任务损失实例
        """
        task_weights = {}
        loss_functions = {}
        
        for task_name, config in task_config.items():
            task_weights[task_name] = config.get('weight', 1.0)
            loss_type = config.get('loss_type', 'cross_entropy')
            loss_params = config.get('loss_params', {})
            
            loss_functions[task_name] = LossFactory.create_loss(loss_type, **loss_params)
        
        return MultiTaskLoss(task_weights, loss_functions)


# 便捷函数
def get_loss_function(config: Dict[str, Any]) -> nn.Module:
    """
    根据配置获取损失函数
    
    Args:
        config: 损失函数配置
        
    Returns:
        损失函数实例
    """
    loss_type = config.get('type', 'cross_entropy')
    loss_params = config.get('params', {})
    
    return LossFactory.create_loss(loss_type, **loss_params)