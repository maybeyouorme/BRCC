"""
ARPL / RPL 损失函数 (从 ARPL-master 适配到 1D 信号域)

包含:
  - Dist:     距离度量模块 (L2 / Dot Product)，管理可学习类别中心
  - ARPLoss:  对抗倒易点损失 (Adversarial Reciprocal Point Loss)
              用 dot 和 L2 距离之差构造 logits，加上边际排序损失
  - RPLoss:   倒易点损失 (Reciprocal Point Loss)，仅用 L2 距离
  - Softmax:  标准 Softmax 交叉熵损失（基线对比用）

参考: Chen et al., "Adversarial Reciprocal Points Learning for Open Set Recognition"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================
# Dist: 距离度量模块
# =========================================================================

class Dist(nn.Module):
    """
    距离度量模块。维护一组可学习的类别中心 (reciprocal points)，
    支持 L2 和 Dot Product 两种度量方式。

    Args:
        num_classes: 类别数
        feat_dim:    特征维度
        num_centers: 每个类别的 center 数 (RPLoss 可 >1, ARPLoss 固定为 1)
    """
    def __init__(self, num_classes=10, feat_dim=128, num_centers=1):
        super(Dist, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.num_centers = num_centers

        # 初始化 centers (与 ARPL-master 保持一致: 0.1 * randn)
        self.centers = nn.Parameter(
            0.1 * torch.randn(num_classes * num_centers, self.feat_dim)
        )

    def forward(self, features, center=None, metric='l2'):
        """
        Args:
            features: 特征 [B, feat_dim]
            center:   可选，外部指定的中心（默认用 self.centers）
            metric:   'l2' 或 'dot'
        Returns:
            dist: 距离 [B, num_classes] (若 num_centers>1 则取均值)
        """
        if center is None:
            center = self.centers

        if metric == 'l2':
            # ||a-b||^2 = a^2 + b^2 - 2ab^T
            f_2 = torch.sum(features ** 2, dim=1, keepdim=True)          # [B, 1]
            c_2 = torch.sum(center ** 2, dim=1, keepdim=True)            # [P, 1]
            dist = (f_2
                    - 2 * torch.matmul(features, center.t())
                    + c_2.t())                                           # [B, P]
            dist = dist / float(features.shape[1])                       # 特征维度归一化
        else:
            # Dot Product 相似度
            dist = features.matmul(center.t())                           # [B, P]

        # 若每个类别有多个中心，取均值
        dist = torch.reshape(dist, [-1, self.num_classes, self.num_centers])
        dist = torch.mean(dist, dim=2)                                   # [B, num_classes]
        return dist


# =========================================================================
# ARPLoss: 对抗倒易点损失
# =========================================================================

class ARPLoss(nn.Module):
    """
    对抗倒易点损失 (Adapted from ARPL-master).

    核心思路:
      - 用 Dot 和 L2 距离的差异构造 logits
      - 交叉熵损失 + 边际排序损失 (使已知类样本接近其中心)

    Args:
        num_classes: 类别数
        feat_dim:    特征维度
        weight_pl:   原型损失权重 (margin loss 系数)
        temp:        温度参数
    """
    def __init__(self, num_classes=10, feat_dim=128, weight_pl=0.1, temp=1.0):
        super(ARPLoss, self).__init__()
        self.weight_pl = weight_pl
        self.temp = temp
        self.Dist = Dist(num_classes=num_classes, feat_dim=feat_dim, num_centers=1)
        self.points = self.Dist.centers
        self.radius = nn.Parameter(torch.Tensor(1))
        self.radius.data.fill_(0)
        self.margin_loss = nn.MarginRankingLoss(margin=1.0)

    def forward(self, x, y, labels=None):
        """
        Args:
            x: 特征 (embedding)     [B, feat_dim]
            y: logits (模型输出, ARPLoss 内部会重新计算，此参数仅用于接口兼容)
            labels: 标签 [B] (训练时提供，测试时为 None)
        Returns:
            logits: 重新计算的距离 logits [B, num_classes]
            loss:   标量损失 (测试时返回 0)
        """
        dist_dot_p = self.Dist(x, center=self.points, metric='dot')
        dist_l2_p = self.Dist(x, center=self.points)                     # L2
        logits = dist_l2_p - dist_dot_p                                   # [B, num_classes]

        if labels is None:
            return logits, 0

        # 1) 交叉熵损失 (在距离 logits 上)
        loss = F.cross_entropy(logits / self.temp, labels)

        # 2) 边际排序损失: 迫使已知类样本与对应中心的距离小于 radius
        center_batch = self.points[labels, :]
        _dis_known = (x - center_batch).pow(2).mean(1)                   # [B]
        target = torch.ones(_dis_known.size()).to(x.device)
        loss_r = self.margin_loss(self.radius, _dis_known, target)

        loss = loss + self.weight_pl * loss_r

        # 3) 联合训练：同时用 CE 监督模型自身的分类头 (logits_from_model)
        #    这样模型自身的 classifier 也能得到训练，评估时可用其 logits 做 OOD 检测
        if y is not None:
            loss = loss + F.cross_entropy(y / self.temp, labels)

        return logits, loss

    def fake_loss(self, x):
        """
        用于生成对抗样本的 fake loss (GAN 混淆训练用)。
        此处保留接口，当前版本未启用 CS (Confusing Sample) 训练。
        """
        logits = self.Dist(x, center=self.points)
        prob = F.softmax(logits, dim=1)
        loss = (prob * torch.log(prob)).sum(1).mean().exp()
        return loss


# =========================================================================
# RPLoss: 倒易点损失 (简化版，无对抗项)
# =========================================================================

class RPLoss(nn.Module):
    """
    倒易点损失 (Adapted from ARPL-master).

    仅使用 L2 距离构造 logits，加上 MSE 约束使已知类样本靠近其中心。

    Args:
        num_classes: 类别数
        feat_dim:    特征维度
        weight_pl:   原型损失权重
        temp:        温度参数
        num_centers: 每个类别的中心数 (可 >1)
    """
    def __init__(self, num_classes=10, feat_dim=128, weight_pl=0.1,
                 temp=1.0, num_centers=1):
        super(RPLoss, self).__init__()
        self.weight_pl = weight_pl
        self.temp = temp
        self.Dist = Dist(num_classes=num_classes, feat_dim=feat_dim,
                         num_centers=num_centers)
        self.radius = nn.Parameter(torch.Tensor(1))
        self.radius.data.fill_(0)

    def forward(self, x, y, labels=None):
        """
        Args:
            x: 特征 (embedding)  [B, feat_dim]
            y: logits (未使用)
            labels: 标签 [B]
        Returns:
            logits: softmax 概率 [B, num_classes]
            loss:   标量损失
        """
        dist = self.Dist(x)
        logits = F.softmax(dist, dim=1)

        if labels is None:
            return logits, 0

        loss = F.cross_entropy(dist / self.temp, labels)

        center_batch = self.Dist.centers[labels, :]
        _dis = (x - center_batch).pow(2).mean(1)
        loss_r = F.mse_loss(_dis, self.radius.expand_as(_dis))

        loss = loss + self.weight_pl * loss_r

        return logits, loss


# =========================================================================
# Softmax: 标准交叉熵损失 (基线对比用)
# =========================================================================

class Softmax(nn.Module):
    """
    标准 Softmax 交叉熵损失。
    模型输出的 logits 直接用于 softmax 和交叉熵计算。
    """
    def __init__(self, num_classes=None, feat_dim=None, weight_pl=None, temp=1.0):
        super(Softmax, self).__init__()
        self.temp = temp

    def forward(self, x, y, labels=None):
        """
        Args:
            x: 特征 (未使用, 接口兼容)
            y: logits [B, num_classes]
            labels: 标签 [B]
        Returns:
            logits: softmax 概率 [B, num_classes]
            loss:   标量损失
        """
        logits = F.softmax(y, dim=1)
        if labels is None:
            return logits, 0
        loss = F.cross_entropy(y / self.temp, labels)
        return logits, loss
