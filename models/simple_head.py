"""
实验三：简单 1×1 卷积分割头（无金字塔池化），作为消融实验对照。
"""

import torch
import torch.nn as nn


class SimpleHead(nn.Module):
    """
    仅一个 1×1 卷积，直接将 backbone 特征映射到类别 logits。

    输入: [B, in_channels, H, W]
    输出: [B, num_classes, H, W]
    """

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
