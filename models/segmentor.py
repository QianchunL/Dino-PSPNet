"""
Segmentor：Backbone + Head 的组合模型。

ResNet101PSPNet:
  - 主分割头：layer4 → PSPHead
  - 辅助分割头：layer3 → AuxHead（deep supervision，训练时 loss 权重 0.4）
  - 训练模式 forward 返回 (main_logits, aux_logits)，均已上采样到原图尺寸
  - 评估模式 forward 只返回 main_logits

Segmentor（DINOv3 实验二/三用）:
  - 单头，forward 始终返回 main_logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import ResNet101Backbone, DINOv3Backbone
from .psp_head import PSPHead
from .simple_head import SimpleHead


class AuxHead(nn.Module):
    """辅助分割头：3×3 conv + BN + ReLU + Dropout + 1×1 classifier。"""

    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResNet101PSPNet(nn.Module):
    """
    实验一：空洞 ResNet101 + PSPHead + AuxHead。

    训练时: forward(x) → (main_logits, aux_logits)，两者都已上采样到原图分辨率。
    评估时: forward(x) → main_logits。
    """

    def __init__(self, num_classes: int = 21, pretrained: bool = True, frozen_backbone: bool = True):
        super().__init__()
        self.backbone = ResNet101Backbone(pretrained=pretrained, frozen=frozen_backbone)
        self.main_head = PSPHead(
            in_channels=ResNet101Backbone.out_channels,
            num_classes=num_classes,
        )
        self.aux_head = AuxHead(
            in_channels=ResNet101Backbone.aux_channels,
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor):
        input_size = (x.shape[2], x.shape[3])
        aux_feat, main_feat = self.backbone(x)

        main_logits = self.main_head(main_feat)
        main_logits = F.interpolate(main_logits, size=input_size, mode="bilinear", align_corners=False)

        if self.training:
            aux_logits = self.aux_head(aux_feat)
            aux_logits = F.interpolate(aux_logits, size=input_size, mode="bilinear", align_corners=False)
            return main_logits, aux_logits

        return main_logits


class Segmentor(nn.Module):
    """DINOv3 实验（实验二/三）用的单头模型，forward 始终返回 main_logits。"""

    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = (x.shape[2], x.shape[3])
        feat   = self.backbone(x)
        logits = self.head(feat)
        if logits.shape[2:] != input_size:
            logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
        return logits


# ──────────────────────────────────────────────
# 便捷工厂函数
# ──────────────────────────────────────────────

def build_resnet_pspnet(num_classes: int = 21, pretrained: bool = True, frozen_backbone: bool = True) -> ResNet101PSPNet:
    """实验一：空洞 ResNet101 + PSPHead（含 AuxHead）"""
    return ResNet101PSPNet(num_classes=num_classes, pretrained=pretrained, frozen_backbone=frozen_backbone)


def build_dinov3_pspnet(num_classes: int = 21, frozen: bool = True) -> Segmentor:
    """实验二：DINOv3 + PSPHead"""
    backbone = DINOv3Backbone(frozen=frozen)
    head = PSPHead(in_channels=backbone.out_channels, num_classes=num_classes)
    return Segmentor(backbone, head)


def build_dinov3_simple(num_classes: int = 21, frozen: bool = True) -> Segmentor:
    """实验三：DINOv3 + SimpleHead（消融实验）"""
    backbone = DINOv3Backbone(frozen=frozen)
    head = SimpleHead(in_channels=backbone.out_channels, num_classes=num_classes)
    return Segmentor(backbone, head)
