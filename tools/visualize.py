"""
可视化脚本：分割结果对比图 + PCA 特征可视化。

用法：
    # 分割结果对比（输入 / GT / 预测）
    python visualize.py seg --checkpoint checkpoints/exp1_resnet101_pspnet/best.pth \
                            --backbone resnet101 --head psp --num_images 8

    # PCA 特征可视化（对比 backbone 特征质量）
    python visualize.py pca --backbone resnet101 --num_images 4
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from PIL import Image
from torch.utils.data import DataLoader

from datasets.voc import VOCSegmentation, VOC_MEAN, VOC_STD
from models.segmentor import build_resnet_pspnet, build_dinov3_pspnet, build_dinov3_simple
from models.backbone import ResNet101Backbone, DINOv3Backbone

# VOC 调色板（21 类）
VOC_COLORMAP = np.array([
    [0,0,0],[128,0,0],[0,128,0],[128,128,0],[0,0,128],
    [128,0,128],[0,128,128],[128,128,128],[64,0,0],[192,0,0],
    [64,128,0],[192,128,0],[64,0,128],[192,0,128],[64,128,128],
    [192,128,128],[0,64,0],[128,64,0],[0,192,0],[128,192,0],
    [0,64,128],
], dtype=np.uint8)

VOC_CLASSES = [
    "background","aeroplane","bicycle","bird","boat","bottle","bus","car",
    "cat","chair","cow","diningtable","dog","horse","motorbike","person",
    "pottedplant","sheep","sofa","train","tvmonitor",
]


def label_to_rgb(label: np.ndarray) -> np.ndarray:
    label = label.copy()
    label[label == 255] = 0
    return VOC_COLORMAP[label]


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    mean = np.array(VOC_MEAN)
    std  = np.array(VOC_STD)
    img  = tensor.permute(1, 2, 0).cpu().numpy()
    img  = img * std + mean
    return np.clip(img, 0, 1)


# ── 分割对比图 ──────────────────────────────────────────────────────────

def vis_seg(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.backbone == "resnet101":
        model = build_resnet_pspnet(num_classes=21, pretrained=False)
    elif args.backbone == "dinov3" and args.head == "psp":
        model = build_dinov3_pspnet(num_classes=21)
    else:
        model = build_dinov3_simple(num_classes=21)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()

    ds     = VOCSegmentation(root=args.data_root, split="val", crop_size=args.crop_size, augment=False)
    loader = DataLoader(ds, batch_size=args.num_images, shuffle=False)
    images, masks = next(iter(loader))

    with torch.no_grad():
        preds = model(images.to(device)).argmax(dim=1).cpu().numpy()

    n = images.shape[0]
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i in range(n):
        axes[i, 0].imshow(denormalize(images[i]))
        axes[i, 0].set_title("Image")
        axes[i, 1].imshow(label_to_rgb(masks[i].numpy()))
        axes[i, 1].set_title("Ground Truth")
        axes[i, 2].imshow(label_to_rgb(preds[i]))
        axes[i, 2].set_title("Prediction")
        for ax in axes[i]:
            ax.axis("off")

    plt.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[saved] {out}")


# ── PCA 特征可视化 ──────────────────────────────────────────────────────

def vis_pca(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.backbone == "resnet101":
        backbone = ResNet101Backbone(pretrained=True, frozen=True).to(device).eval()
    else:
        backbone = DINOv3Backbone(frozen=True).to(device).eval()

    ds     = VOCSegmentation(root=args.data_root, split="val", crop_size=args.crop_size, augment=False)
    loader = DataLoader(ds, batch_size=args.num_images, shuffle=False)
    images, _ = next(iter(loader))

    with torch.no_grad():
        feat = backbone(images.to(device))  # [B, C, H, W]

    B, C, H, W = feat.shape
    feat_np = feat.permute(0, 2, 3, 1).reshape(-1, C).cpu().numpy()

    pca = PCA(n_components=3)
    pca_feat = pca.fit_transform(feat_np)                     # [B*H*W, 3]
    pca_feat = pca_feat.reshape(B, H, W, 3)

    # 归一化到 [0, 1]
    pca_feat -= pca_feat.min()
    pca_feat /= (pca_feat.max() + 1e-8)

    fig, axes = plt.subplots(args.num_images, 2, figsize=(8, 4 * args.num_images))
    if args.num_images == 1:
        axes = axes[np.newaxis, :]
    for i in range(args.num_images):
        axes[i, 0].imshow(denormalize(images[i]))
        axes[i, 0].set_title("Image")
        axes[i, 1].imshow(pca_feat[i])
        axes[i, 1].set_title(f"PCA features ({args.backbone})")
        for ax in axes[i]:
            ax.axis("off")

    plt.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[saved] {out}")


# ── CLI ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode")

    seg = sub.add_parser("seg")
    seg.add_argument("--checkpoint", required=True)
    seg.add_argument("--backbone",   default="resnet101", choices=["resnet101", "dinov3"])
    seg.add_argument("--head",       default="psp",      choices=["psp", "simple"])
    seg.add_argument("--data_root",  default="./data")
    seg.add_argument("--crop_size",  type=int, default=512)
    seg.add_argument("--num_images", type=int, default=8)
    seg.add_argument("--output",     default="./vis/seg_results.png")

    pca = sub.add_parser("pca")
    pca.add_argument("--backbone",   default="resnet101", choices=["resnet101", "dinov3"])
    pca.add_argument("--data_root",  default="./data")
    pca.add_argument("--crop_size",  type=int, default=512)
    pca.add_argument("--num_images", type=int, default=4)
    pca.add_argument("--output",     default="./vis/pca_features.png")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "seg":
        vis_seg(args)
    elif args.mode == "pca":
        vis_pca(args)
    else:
        print("Usage: python visualize.py {seg|pca} [options]")
