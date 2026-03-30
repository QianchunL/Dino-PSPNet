"""
训练脚本（按 PSPNet 论文设置）。

优化器: SGD + momentum=0.9 + weight_decay=1e-4
学习率: poly decay（按 iteration），base_lr=0.01
损失:   CrossEntropyLoss(main) + 0.4 * CrossEntropyLoss(aux)  [仅 resnet101]
数据:   trainaug（10,582 张）

用法示例：
    # 实验一：ResNet101 + PSPNet
    python train.py --backbone resnet101 --head psp

    # 实验二：DINOv3 + PSPNet
    python train.py --backbone dinov3 --head psp --experiment exp2_dinov3_pspnet

    # 实验三：DINOv3 + 简单头（消融）
    python train.py --backbone dinov3 --head simple --experiment exp3_dinov3_simple

    # batch_size 减半时，按线性缩放规则减半 lr：
    python train.py --backbone resnet101 --batch_size 8 --lr 0.005
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets.voc import VOCSegmentation
from models.segmentor import (
    build_resnet_pspnet, build_dinov3_pspnet, build_dinov3_simple, ResNet101PSPNet
)


# ── Poly LR（按 iteration） ───────────────────────────────────────────────

def poly_lr(base_lr: float, cur_iter: int, max_iter: int, power: float = 0.9) -> float:
    return base_lr * (1.0 - cur_iter / max_iter) ** power


def set_lr(optimizer, base_lr: float) -> None:
    """按 poly 衰减更新 lr；各组保持初始时设定的 lr_mult 比例。"""
    for pg in optimizer.param_groups:
        pg["lr"] = base_lr * pg.get("lr_mult", 1.0)


# ── mIoU（训练时快速估算，用混淆矩阵） ───────────────────────────────────

def batch_confusion(preds: np.ndarray, labels: np.ndarray, num_classes: int, ignore_index: int = 255):
    mask = labels != ignore_index
    preds  = preds[mask]
    labels = labels[mask]
    conf = np.bincount(num_classes * labels + preds, minlength=num_classes ** 2)
    return conf.reshape(num_classes, num_classes)


def miou_from_confusion(conf: np.ndarray) -> float:
    iou = []
    for c in range(conf.shape[0]):
        denom = conf[c, :].sum() + conf[:, c].sum() - conf[c, c]
        if denom > 0:
            iou.append(conf[c, c] / denom)
    return float(np.mean(iou)) if iou else 0.0


# ── 主训练循环 ────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # ── 数据集 ──
    train_ds = VOCSegmentation(
        root=args.data_root,
        split=args.split_train,
        crop_size=args.crop_size,
        scale_range=tuple(args.scale_range),
        augment=True,
    )
    val_ds = VOCSegmentation(
        root=args.data_root,
        split="val",
        crop_size=args.crop_size,
        augment=False,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    print(f"[data] train={len(train_ds)}, val={len(val_ds)}")

    # ── 模型 ──
    if args.backbone == "resnet101":
        model = build_resnet_pspnet(
            num_classes=args.num_classes,
            pretrained=True,
            frozen_backbone=args.frozen_backbone,
        )
    elif args.backbone == "dinov3" and args.head == "psp":
        model = build_dinov3_pspnet(num_classes=args.num_classes, frozen=args.frozen_backbone)
    elif args.backbone == "dinov3" and args.head == "simple":
        model = build_dinov3_simple(num_classes=args.num_classes, frozen=args.frozen_backbone)
    else:
        raise ValueError(f"Unknown backbone={args.backbone}, head={args.head}")
    model = model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[model] trainable={trainable:,} / total={total:,} params")

    # ── 优化器：SGD + momentum，论文设置 ──
    # 全量微调时 backbone 用 lr×backbone_lr_mult（论文 0.1），head 用 lr×1.0
    if not args.frozen_backbone and hasattr(model, "backbone"):
        backbone_ids = {id(p) for p in model.backbone.parameters()}
        param_groups = [
            {"params": [p for p in model.parameters() if id(p) not in backbone_ids],
             "lr": args.lr, "lr_mult": 1.0},
            {"params": list(model.backbone.parameters()),
             "lr": args.lr * args.backbone_lr_mult, "lr_mult": args.backbone_lr_mult},
        ]
    else:
        param_groups = [
            {"params": list(filter(lambda p: p.requires_grad, model.parameters())),
             "lr": args.lr, "lr_mult": 1.0},
        ]
    optimizer = torch.optim.SGD(
        param_groups,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_index)

    # ── 输出目录 ──
    save_dir = Path(args.save_dir) / args.experiment
    save_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(Path(args.log_dir) / args.experiment))

    use_aux   = isinstance(model, ResNet101PSPNet)
    iters_per_epoch = len(train_loader)
    max_iters = args.max_iters
    epochs    = math.ceil(max_iters / iters_per_epoch)
    cur_iter  = 0
    best_miou = 0.0
    print(f"[train] max_iters={max_iters}  iters/epoch={iters_per_epoch}  => epochs={epochs}")

    # ── 训练循环 ──
    for epoch in range(1, epochs + 1):
        model.train()
        # 冻结 backbone 时保持其 BN 在 eval 模式，防止 running stats 被 VOC 数据污染
        if args.frozen_backbone and hasattr(model, "backbone"):
            model.backbone.eval()
        total_loss = total_main = total_aux = 0.0
        t0 = time.time()
        num_classes = args.num_classes
        conf = np.zeros((num_classes, num_classes), dtype=np.int64)

        for images, masks in train_loader:
            # Poly LR：按 iteration 更新
            lr = poly_lr(args.lr, cur_iter, max_iters)
            set_lr(optimizer, lr)
            cur_iter += 1

            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()

            if use_aux:
                main_logits, aux_logits = model(images)
                loss_main = criterion(main_logits, masks)
                loss_aux  = criterion(aux_logits,  masks)
                loss = loss_main + args.aux_weight * loss_aux
                total_main += loss_main.item()
                total_aux  += loss_aux.item()
            else:
                main_logits = model(images)
                loss = criterion(main_logits, masks)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # 训练集 mIoU 估算（每 batch 累积混淆矩阵）
            preds = main_logits.detach().argmax(dim=1).cpu().numpy()
            conf += batch_confusion(preds, masks.cpu().numpy(), num_classes, args.ignore_index)

        n = len(train_loader)
        train_miou = miou_from_confusion(conf)
        elapsed    = time.time() - t0

        # ── 验证（resnet101 切 eval 模式，关闭 aux） ──
        model.eval()
        val_conf = np.zeros((num_classes, num_classes), dtype=np.int64)
        val_loss_total = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                logits = model(images)  # eval 模式只返回 main_logits
                val_loss_total += criterion(logits, masks).item()
                preds  = logits.argmax(dim=1).cpu().numpy()
                val_conf += batch_confusion(preds, masks.cpu().numpy(), num_classes, args.ignore_index)
        val_miou = miou_from_confusion(val_conf)
        val_loss_avg = val_loss_total / len(val_loader)

        # ── 日志 ──
        writer.add_scalar("train/loss",    total_loss / n, epoch)
        writer.add_scalar("train/mIoU",    train_miou,     epoch)
        writer.add_scalar("train/lr",      lr,             epoch)
        writer.add_scalar("val/loss",      val_loss_avg,   epoch)
        writer.add_scalar("val/mIoU",      val_miou,       epoch)
        if use_aux:
            writer.add_scalar("train/loss_main", total_main / n, epoch)
            writer.add_scalar("train/loss_aux",  total_aux  / n, epoch)

        log = (f"[{epoch:03d}/{epochs}] "
               f"loss={total_loss/n:.4f}  val_loss={val_loss_avg:.4f}  "
               f"train_mIoU={train_miou:.4f}  val_mIoU={val_miou:.4f}  "
               f"lr={lr:.2e}  {elapsed:.0f}s")
        if use_aux:
            log += f"  (main={total_main/n:.4f} aux={total_aux/n:.4f})"
        print(log)

        # ── 保存 checkpoint ──
        if epoch % args.save_every == 0:
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict()},
                save_dir / f"epoch{epoch:03d}.pth",
            )

        if val_miou > best_miou:
            best_miou = val_miou
            torch.save({"epoch": epoch, "model": model.state_dict(), "miou": best_miou}, save_dir / "best.pth")
            print(f"  [*] new best mIoU={best_miou:.4f}")

    writer.close()
    print(f"\n[done] best val mIoU = {best_miou:.4f}")


# ── 命令行参数 ────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    # 模型
    p.add_argument("--backbone",        default="resnet101",              choices=["resnet101", "dinov3"])
    p.add_argument("--head",            default="psp",                    choices=["psp", "simple"])
    p.add_argument("--num_classes",     type=int,   default=21)
    p.add_argument("--frozen_backbone", action=argparse.BooleanOptionalAction, default=True,
                   help="冻结 backbone（默认）；--no_frozen_backbone 解冻微调")
    # 数据
    p.add_argument("--data_root",       default="./data")
    p.add_argument("--split_train",     default="trainaug",               help="train | trainaug")
    p.add_argument("--crop_size",       type=int,   default=512)
    p.add_argument("--scale_range",     type=float, nargs=2,              default=[0.5, 2.0])
    p.add_argument("--num_workers",     type=int,   default=4)
    # 训练（论文默认值）
    p.add_argument("--max_iters",       type=int,   default=30000,
                   help="总迭代数（论文 VOC=30K）；epochs 由 max_iters / iters_per_epoch 自动推算")
    p.add_argument("--batch_size",      type=int,   default=16)
    p.add_argument("--lr",              type=float, default=0.01,
                   help="head base lr（batch_size=16 基准）；其他 batch_size 建议按线性缩放")
    p.add_argument("--backbone_lr_mult", type=float, default=0.1,
                   help="全量微调时 backbone lr = lr × backbone_lr_mult（论文设置 0.1）")
    p.add_argument("--weight_decay",    type=float, default=1e-4)
    p.add_argument("--aux_weight",      type=float, default=0.4,         help="auxiliary loss 权重（仅 resnet101）")
    p.add_argument("--ignore_index",    type=int,   default=255)
    # 输出
    p.add_argument("--experiment",      default="exp1_resnet101_pspnet")
    p.add_argument("--save_dir",        default="./checkpoints")
    p.add_argument("--log_dir",         default="./runs")
    p.add_argument("--save_every",      type=int,   default=5)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
