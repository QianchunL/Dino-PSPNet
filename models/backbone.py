"""
Backbone 封装：ResNet101（实验一）和 DINOv3（实验二/三）。

ResNet101Backbone 按 PSPNet 论文做空洞卷积改造：
  - layer3: stride 1 + dilation 2  → 输出 stride 累计 8
  - layer4: stride 1 + dilation 4  → 输出 stride 累计 8
forward 返回 (aux_feat, main_feat)，分别对应 layer3/layer4 输出，
供 auxiliary loss 分支和主分割头使用。

DINOv3Backbone forward 返回单个 2D 特征图 [B, C, H, W]。
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models


def _set_layer_dilation(layer: nn.Sequential, dilation: int) -> None:
    """
    将 ResNet layer（由多个 Bottleneck block 组成）中的 3×3 conv 替换为空洞卷积，
    同时去掉第一个 block 的下采样 stride，保持空间分辨率不变。
    """
    for block in layer:
        # conv2 是 Bottleneck 中唯一的 3×3 卷积
        block.conv2.dilation = (dilation, dilation)
        block.conv2.padding  = (dilation, dilation)
        block.conv2.stride   = (1, 1)
        # 第一个 block 的 downsample 分支（1×1 conv + BN）也有 stride=2，需置 1
        if block.downsample is not None:
            block.downsample[0].stride = (1, 1)


class ResNet101Backbone(nn.Module):
    """
    实验一 backbone：ImageNet 预训练的 ResNet101，空洞卷积版本。

    输出 stride = 8（原始 ResNet 为 32）：
      stem + layer1 + layer2: stride 8（正常）
      layer3: stride 1 + dilation 2  → 仍是 stride 8
      layer4: stride 1 + dilation 4  → 仍是 stride 8

    forward 返回:
      aux_feat  [B, 1024, H/8, W/8]  —— layer3 输出，用于 auxiliary loss
      main_feat [B, 2048, H/8, W/8]  —— layer4 输出，送入 PSP 主分割头
    """

    aux_channels  = 1024
    out_channels  = 2048

    def __init__(self, pretrained: bool = True, frozen: bool = True):
        super().__init__()
        base = tv_models.resnet101(
            weights=tv_models.ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
        )

        self.stem   = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1  # stride 累计 4,  channels 256
        self.layer2 = base.layer2  # stride 累计 8,  channels 512
        self.layer3 = base.layer3  # 改为 dilation=2，stride 累计 8,  channels 1024
        self.layer4 = base.layer4  # 改为 dilation=4，stride 累计 8,  channels 2048

        # 按论文修改：layer3 dilation=2，layer4 dilation=4
        _set_layer_dilation(self.layer3, dilation=2)
        _set_layer_dilation(self.layer4, dilation=4)

        if frozen:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        aux  = self.layer3(x)   # [B, 1024, H/8, W/8]
        main = self.layer4(aux) # [B, 2048, H/8, W/8]
        return aux, main


class DINOv3Backbone(nn.Module):
    """
    实验二/三 backbone：DINOv3 ViT-S/16（facebook/dinov3-vits16-pretrain-lvd1689m）。
    输出 patch token 重排为 2D 特征图 [B, 384, H/16, W/16]。
    默认冻结所有参数。
    """

    def __init__(self, model_id: str = "facebook/dinov3-vits16-pretrain-lvd1689m", frozen: bool = True):
        super().__init__()
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(model_id)
        self.out_channels = self.model.config.hidden_size  # 384 for ViT-S
        self.patch_size   = self.model.config.patch_size   # 16

        if frozen:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h_patches = H // self.patch_size
        w_patches = W // self.patch_size

        outputs = self.model(pixel_values=x, output_hidden_states=False)
        # last_hidden_state: [B, 1 + num_registers + N, hidden_size]
        seq = outputs.last_hidden_state

        # DINOv3 序列: [CLS, reg1, reg2, reg3, reg4, patch_1, ..., patch_N]
        num_prefix   = 1 + 4  # 1 CLS + 4 register tokens
        patch_tokens = seq[:, num_prefix:, :]  # [B, N, C]

        feat = patch_tokens.permute(0, 2, 1)                        # [B, C, N]
        feat = feat.reshape(B, self.out_channels, h_patches, w_patches)
        return feat
