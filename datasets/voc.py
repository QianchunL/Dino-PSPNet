"""
Pascal VOC 2012 语义分割数据集加载。

训练集支持两种 split：
  "train"    — 官方训练集，1,464 张
  "trainaug" — SBD 增强版，10,582 张（论文使用）
               需要额外的 trainaug.txt 和 SegmentationClassAug/ 目录

数据增强（训练集，按 PSPNet 论文）：
  1. 随机缩放（0.5–2.0）
  2. 随机水平翻转
  3. 随机旋转（-10°~+10°）
  4. 随机高斯模糊
  5. 随机裁剪到 crop_size × crop_size

label=255 的边界像素在损失计算时忽略。
"""

from pathlib import Path
import random

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision.transforms as T


VOC_MEAN = (0.485, 0.456, 0.406)
VOC_STD  = (0.229, 0.224, 0.225)

# 用均值像素填充图像边缘（pad 时用）
_PAD_FILL = (int(VOC_MEAN[0] * 255), int(VOC_MEAN[1] * 255), int(VOC_MEAN[2] * 255))


class VOCSegmentation(Dataset):
    """
    Pascal VOC 2012 语义分割数据集。

    目录结构（torchvision 标准布局 + SBD 增强）：
        root/
          VOCdevkit/
            VOC2012/
              JPEGImages/
              SegmentationClass/          # 原始 mask（val 使用）
              SegmentationClassAug/       # SBD 增强 mask（trainaug 使用）
              ImageSets/Segmentation/
                train.txt
                val.txt
                trainaug.txt              # SBD 增强训练集列表
    """

    def __init__(
        self,
        root: str,
        split: str = "trainaug",    # "train" | "trainaug" | "val"
        crop_size: int = 512,
        scale_range: tuple = (0.5, 2.0),
        augment: bool = True,
    ):
        super().__init__()
        self.root      = Path(root) / "VOCdevkit" / "VOC2012"
        self.split     = split
        self.crop_size = crop_size
        self.scale_range = scale_range
        self.augment   = augment

        # trainaug 使用 SBD 增强 mask 目录
        self.mask_dir = (
            self.root / "SegmentationClassAug"
            if split == "trainaug"
            else self.root / "SegmentationClass"
        )

        list_file = self.root / "ImageSets" / "Segmentation" / f"{split}.txt"
        with open(list_file) as f:
            self.ids = [line.strip() for line in f if line.strip()]

        self.normalize = T.Normalize(mean=VOC_MEAN, std=VOC_STD)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        name  = self.ids[idx]
        image = Image.open(self.root / "JPEGImages" / f"{name}.jpg").convert("RGB")
        mask  = Image.open(self.mask_dir / f"{name}.png")

        if self.augment:
            image, mask = self._augment(image, mask)
        else:
            image, mask = self._val_transform(image, mask)

        image = self.normalize(TF.to_tensor(image))
        mask  = torch.from_numpy(np.array(mask, dtype=np.int64))
        return image, mask

    # ── 训练增强 ──────────────────────────────────────────────────────────

    def _augment(self, image: Image.Image, mask: Image.Image):
        # 1. 随机缩放（0.5–2.0）
        scale    = random.uniform(*self.scale_range)
        w, h     = image.size
        new_w, new_h = int(w * scale), int(h * scale)
        image = TF.resize(image, (new_h, new_w), interpolation=TF.InterpolationMode.BILINEAR)
        mask  = TF.resize(mask,  (new_h, new_w), interpolation=TF.InterpolationMode.NEAREST)

        # 2. 随机水平翻转
        if random.random() < 0.5:
            image = TF.hflip(image)
            mask  = TF.hflip(mask)

        # 3. 随机旋转（-10°~+10°），mask 用 255 填充边缘
        angle = random.uniform(-10, 10)
        image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=_PAD_FILL)
        mask  = TF.rotate(mask,  angle, interpolation=TF.InterpolationMode.NEAREST,  fill=255)

        # 4. 随机高斯模糊（仅作用于图像，50% 概率）
        if random.random() < 0.5:
            radius = random.uniform(0.1, 2.0)
            image = image.filter(ImageFilter.GaussianBlur(radius=radius))

        # 5. 随机裁剪到 crop_size × crop_size（不足时用均值/255 填充）
        image, mask = self._random_crop(image, mask)
        return image, mask

    def _random_crop(self, image: Image.Image, mask: Image.Image):
        w, h  = image.size
        pad_h = max(self.crop_size - h, 0)
        pad_w = max(self.crop_size - w, 0)
        if pad_h > 0 or pad_w > 0:
            image = TF.pad(image, (0, 0, pad_w, pad_h), fill=_PAD_FILL)
            mask  = TF.pad(mask,  (0, 0, pad_w, pad_h), fill=255)
        w, h = image.size
        i = random.randint(0, h - self.crop_size)
        j = random.randint(0, w - self.crop_size)
        image = TF.crop(image, i, j, self.crop_size, self.crop_size)
        mask  = TF.crop(mask,  i, j, self.crop_size, self.crop_size)
        return image, mask

    # ── 验证集变换 ────────────────────────────────────────────────────────

    def _val_transform(self, image: Image.Image, mask: Image.Image):
        """
        验证集：resize 到 crop_size 以兼容 DataLoader 批处理。
        evaluate.py 的多尺度评估直接在原图上操作，不经过此函数。
        """
        image = TF.resize(image, (self.crop_size, self.crop_size), interpolation=TF.InterpolationMode.BILINEAR)
        mask  = TF.resize(mask,  (self.crop_size, self.crop_size), interpolation=TF.InterpolationMode.NEAREST)
        return image, mask
