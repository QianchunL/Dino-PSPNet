"""
评估脚本：多尺度 + 水平翻转推理，计算验证集 mIoU（逐类别详细输出）。

论文评估协议：
  - 测试尺度: {0.5, 0.75, 1.0, 1.25, 1.5, 1.75}（相对于原图）
  - 每个尺度额外做水平翻转，共 12 次前向
  - 将各尺度/翻转的 softmax 概率双线性上采样回原图尺寸后平均
  - argmax 得到最终预测

用法：
    python evaluate.py --checkpoint checkpoints/exp1_resnet101_pspnet/best.pth \\
                       --backbone resnet101 --head psp

    # 单尺度（更快）：
    python evaluate.py --checkpoint ... --scales 1.0 --no_flip
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import torchvision.transforms as T

from datasets.voc import VOCSegmentation, VOC_MEAN, VOC_STD
from models.segmentor import build_resnet_pspnet, build_dinov3_pspnet, build_dinov3_simple

VOC_CLASSES = [
    "background",  "aeroplane", "bicycle",  "bird",       "boat",
    "bottle",      "bus",       "car",       "cat",        "chair",
    "cow",         "diningtable", "dog",     "horse",      "motorbike",
    "person",      "pottedplant", "sheep",   "sofa",       "train",
    "tvmonitor",
]

# ── 原图尺寸数据集（batch_size=1，不 resize） ─────────────────────────────

class VOCOriginalSize(Dataset):
    """验证集，保留原始分辨率（evaluate.py 专用，batch_size=1）。"""

    def __init__(self, root: str):
        voc_root = Path(root) / "VOCdevkit" / "VOC2012"
        list_file = voc_root / "ImageSets" / "Segmentation" / "val.txt"
        with open(list_file) as f:
            self.ids = [l.strip() for l in f if l.strip()]
        self.img_dir  = voc_root / "JPEGImages"
        self.mask_dir = voc_root / "SegmentationClass"
        self.normalize = T.Normalize(mean=VOC_MEAN, std=VOC_STD)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name  = self.ids[idx]
        image = Image.open(self.img_dir  / f"{name}.jpg").convert("RGB")
        mask  = Image.open(self.mask_dir / f"{name}.png")
        # 不 resize，保留原始尺寸
        image_t = self.normalize(TF.to_tensor(image))   # [3, H, W]
        mask_np = np.array(mask, dtype=np.int64)
        return image_t, mask_np, name


# ── 单张图多尺度推理 ──────────────────────────────────────────────────────

@torch.no_grad()
def predict_multiscale(
    model: torch.nn.Module,
    image: torch.Tensor,           # [3, H, W]，已归一化
    scales: list,
    flip: bool,
    num_classes: int,
    device: torch.device,
) -> np.ndarray:
    """
    对一张图做多尺度 + 翻转推理，返回 argmax 预测标签 [H, W]。
    """
    _, H, W = image.shape
    probs_sum = torch.zeros(num_classes, H, W, device=device)

    for scale in scales:
        # resize 到 scale 倍
        new_h, new_w = int(H * scale), int(W * scale)
        img_scaled = F.interpolate(
            image.unsqueeze(0).to(device),
            size=(new_h, new_w), mode="bilinear", align_corners=False,
        )

        for do_flip in ([False, True] if flip else [False]):
            inp = img_scaled.flip(-1) if do_flip else img_scaled

            logits = model(inp)  # [1, C, h, w]

            # softmax → 上采样回原图尺寸
            prob = F.softmax(logits, dim=1)
            prob = F.interpolate(prob, size=(H, W), mode="bilinear", align_corners=False)

            if do_flip:
                prob = prob.flip(-1)

            probs_sum += prob.squeeze(0)

    pred = probs_sum.argmax(dim=0).cpu().numpy()  # [H, W]
    return pred


# ── 混淆矩阵工具 ──────────────────────────────────────────────────────────

def update_confusion(conf: np.ndarray, pred: np.ndarray, gt: np.ndarray, ignore_index: int = 255):
    mask  = gt != ignore_index
    pred  = pred[mask]
    gt    = gt[mask]
    n     = conf.shape[0]
    np.add.at(conf, (gt, pred), 1)


def compute_miou(conf: np.ndarray):
    iou_list = []
    for c in range(conf.shape[0]):
        tp    = conf[c, c]
        denom = conf[c, :].sum() + conf[:, c].sum() - tp
        if denom > 0:
            iou_list.append(tp / denom)
    return iou_list, float(np.mean(iou_list)) if iou_list else 0.0


# ── 主评估函数 ────────────────────────────────────────────────────────────

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模型
    if args.backbone == "resnet101":
        model = build_resnet_pspnet(num_classes=21, pretrained=False)
    elif args.backbone == "dinov3" and args.head == "psp":
        model = build_dinov3_pspnet(num_classes=21, frozen=True)
    else:
        model = build_dinov3_simple(num_classes=21, frozen=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()

    scales = args.scales
    flip   = not args.no_flip
    print(f"[eval] scales={scales}  flip={flip}")

    # 数据（原图尺寸，batch_size=1）
    ds = VOCOriginalSize(root=args.data_root)

    num_classes = 21
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)

    for i, (image, mask_np, name) in enumerate(ds):
        pred = predict_multiscale(model, image, scales, flip, num_classes, device)
        update_confusion(conf, pred, mask_np)
        if (i + 1) % 100 == 0:
            _, cur_miou = compute_miou(conf)
            print(f"  [{i+1}/{len(ds)}] running mIoU={cur_miou:.4f}")

    iou_list, miou = compute_miou(conf)

    # 逐类别输出
    print(f"\n{'Class':<16} {'IoU':>8}")
    print("-" * 26)
    for c, iou in enumerate(iou_list):
        print(f"{VOC_CLASSES[c]:<16} {iou:.4f}")
    print("-" * 26)
    print(f"{'mIoU':<16} {miou:.4f}")


# ── CLI ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--backbone",    default="resnet101", choices=["resnet101", "dinov3"])
    p.add_argument("--head",        default="psp",       choices=["psp", "simple"])
    p.add_argument("--data_root",   default="./data")
    p.add_argument("--scales",      type=float, nargs="+",
                   default=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
                   help="多尺度测试的缩放比例列表")
    p.add_argument("--no_flip",     action="store_true",
                   help="禁用水平翻转增强（加速评估）")
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
