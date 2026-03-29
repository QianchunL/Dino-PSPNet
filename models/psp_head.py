import torch
import torch.nn as nn
import torch.nn.functional as F


class PyramidPoolingModule(nn.Module):
    """
    金字塔池化模块 (PPM)，来自 PSPNet 论文 (Zhao et al., 2017)。
    对输入特征图做 4 个尺度的自适应平均池化，降维后上采样拼接回原特征图。
    """

    def __init__(self, in_channels: int, pool_sizes: tuple = (1, 2, 3, 6)):
        super().__init__()
        reduced_channels = in_channels // len(pool_sizes)

        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(pool_size),
                nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduced_channels),
                nn.ReLU(inplace=True),
            )
            for pool_size in pool_sizes
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2], x.shape[3]
        pooled = [x]
        for stage in self.stages:
            out = stage(x)
            out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)
            pooled.append(out)
        return torch.cat(pooled, dim=1)


class PSPHead(nn.Module):
    """
    PSPNet 分割头：PPM + 卷积输出逐像素类别预测。

    输入: [B, in_channels, H, W]
    输出: [B, num_classes, H, W]（与输入同分辨率，最终由训练脚本上采样到原图）
    """

    def __init__(self, in_channels: int, num_classes: int, pool_sizes: tuple = (1, 2, 3, 6), dropout: float = 0.1):
        super().__init__()
        self.ppm = PyramidPoolingModule(in_channels, pool_sizes)

        # PPM 拼接后的通道数 = in_channels + len(pool_sizes) * (in_channels // len(pool_sizes))
        # 由于整除可能有余数，用实际计算值
        reduced = in_channels // len(pool_sizes)
        fused_channels = in_channels + reduced * len(pool_sizes)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(fused_channels, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(512, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ppm(x)
        x = self.bottleneck(x)
        return x
