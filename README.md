# DINOv3 + PSPNet 语义分割

以 DINOv3（2025，Meta AI，自监督 ViT）为 backbone，结合 PSPNet 金字塔池化模块，在 Pascal VOC 2012 上验证自监督视觉特征的分割能力。

---

## 实验设计

### 实验逻辑链

```
实验一 vs 实验二 → DINOv3 自监督特征 vs ResNet101 监督特征，谁更适合分割？
实验二 vs 实验三 → 金字塔池化对 ViT 特征是否还有附加价值？
```

| 实验 | Backbone | 分割头 | 训练方式 | 目的 |
|------|----------|--------|----------|------|
| 实验一（冻结） | ResNet101（ImageNet 预训练） | PSPHead | linear probing | CNN baseline |
| 实验一（微调） | ResNet101（ImageNet 预训练） | PSPHead | full fine-tuning | 对标论文 |
| 实验二 | DINOv3 ViT-S/16（自监督） | PSPHead | linear probing | 核心实验 |
| 实验三 | DINOv3 ViT-S/16（自监督） | 1×1 conv | linear probing | 消融：PPM 是否必要 |

---

## 实验结果

> 评估协议：多尺度（0.5, 0.75, 1.0, 1.25, 1.5, 1.75）+ 水平翻转，VOC 2012 **val set**

### 汇总

| 实验 | Backbone | 训练方式 | mIoU (val) |
|------|----------|----------|-----------|
| 实验一（冻结） | ResNet101 | linear probing | 0.7021 |
| 实验一（微调） | ResNet101 | full fine-tuning | 0.7873 |
| **实验二** | **DINOv3 ViT-S/16** | **linear probing** | **0.8239** |
| 实验三 | DINOv3 ViT-S/16 | linear probing（无PPM） | 0.8127 |
| 论文 PSPNet | ResNet101 | full fine-tuning | 0.826（test set） |

### 跨实验分析

| 对比 | 差值 | 结论 |
|------|------|------|
| 实验一冻结 → 实验二 | +12.2 pts | DINOv3 自监督特征远优于 ResNet101 监督特征 |
| 实验一微调 → 实验二 | +3.7 pts | DINOv3 冻结特征超过 ResNet101 全量微调 |
| 实验三 → 实验二 | +1.1 pts | PPM 在 ViT 时代仍有效，但边际价值大幅降低 |
| 实验三 → 实验一微调 | +2.5 pts | 仅 1×1 卷积的 DINOv3 仍优于全量微调的 ResNet101 |

### 逐类别结果

<details>
<summary>实验一（冻结）mIoU = 0.7021</summary>

| Class | IoU | Class | IoU |
|-------|-----|-------|-----|
| background | 0.9187 | cow | 0.7994 |
| aeroplane | 0.7962 | diningtable | 0.5016 |
| bicycle | 0.3576 | dog | 0.8267 |
| bird | 0.8015 | horse | 0.7471 |
| boat | 0.6170 | motorbike | 0.7346 |
| bottle | 0.7202 | person | 0.7871 |
| bus | 0.8935 | pottedplant | 0.5015 |
| car | 0.8080 | sheep | 0.8198 |
| cat | 0.8770 | sofa | 0.3888 |
| chair | 0.3226 | train | 0.8223 |
| | | tvmonitor | 0.7032 |
</details>

<details>
<summary>实验一（微调）mIoU = 0.7873</summary>

| Class | IoU | Class | IoU |
|-------|-----|-------|-----|
| background | 0.9455 | cow | 0.9138 |
| aeroplane | 0.9035 | diningtable | 0.5737 |
| bicycle | 0.4224 | dog | 0.9235 |
| bird | 0.9025 | horse | 0.8995 |
| boat | 0.6967 | motorbike | 0.8393 |
| bottle | 0.8159 | person | 0.8713 |
| bus | 0.9456 | pottedplant | 0.5967 |
| car | 0.8740 | sheep | 0.8999 |
| cat | 0.9516 | sofa | 0.5083 |
| chair | 0.3949 | train | 0.8773 |
| | | tvmonitor | 0.7775 |
</details>

<details>
<summary>实验二（DINOv3 + PSP）mIoU = 0.8239</summary>

| Class | IoU | Class | IoU |
|-------|-----|-------|-----|
| background | 0.9543 | cow | 0.8979 |
| aeroplane | 0.8960 | diningtable | 0.6996 |
| bicycle | 0.4343 | dog | 0.9238 |
| bird | 0.9196 | horse | 0.8823 |
| boat | 0.7912 | motorbike | 0.8952 |
| bottle | 0.8508 | person | 0.8966 |
| bus | 0.9493 | pottedplant | 0.7426 |
| car | 0.9076 | sheep | 0.8878 |
| cat | 0.9431 | sofa | 0.6307 |
| chair | 0.4911 | train | 0.9274 |
| | | tvmonitor | 0.7808 |
</details>

<details>
<summary>实验三（DINOv3 + 1×1 conv）mIoU = 0.8127</summary>

| Class | IoU | Class | IoU |
|-------|-----|-------|-----|
| background | 0.9522 | cow | 0.9066 |
| aeroplane | 0.8911 | diningtable | 0.6365 |
| bicycle | 0.4341 | dog | 0.9139 |
| bird | 0.8846 | horse | 0.8841 |
| boat | 0.7672 | motorbike | 0.8895 |
| bottle | 0.8402 | person | 0.8997 |
| bus | 0.9323 | pottedplant | 0.7098 |
| car | 0.9052 | sheep | 0.8888 |
| cat | 0.9371 | sofa | 0.6432 |
| chair | 0.5043 | train | 0.9157 |
| | | tvmonitor | 0.7301 |
</details>

---

## 模型结构

### 实验一：ResNet101 + PSPNet（含空洞卷积 + deep supervision）

```
输入图像           [B, 3, 512, 512]
    │
    ▼ stem + layer1 + layer2          [B, 512, 64, 64]   stride 累计 8
    ▼ layer3  ← dilation=2            [B, 1024, 64, 64]
    │   └──→ AuxHead → aux_logits     [B, 21, 512, 512]  (训练时 loss×0.4)
    ▼ layer4  ← dilation=4            [B, 2048, 64, 64]
    │
    ▼ PyramidPoolingModule
    │   AdaptiveAvgPool(1/2/3/6) → Conv1×1(2048→512) → upsample
    │   cat(原始 + 4分支) →           [B, 4096, 64, 64]
    ▼ bottleneck Conv3×3 + Conv1×1 →  [B, 21, 64, 64]
    ▼ 双线性上采样 →                   [B, 21, 512, 512]
```

空洞卷积将输出步长从 32 降到 8，特征图从 16×16 提升到 64×64。Deep supervision 对 layer3 额外施加分割损失（权重 0.4），推理时丢弃。

### 实验二：DINOv3 + PSPHead

```
输入图像                [B, 3, 512, 512]
    ▼ DINOv3 ViT-S/16（冻结）
    │   patch_size=16 → 32×32 patches
    │   序列: [CLS(1), registers(4), patches(1024)]
    │   取 patches → reshape →         [B, 384, 32, 32]
    ▼ PyramidPoolingModule（in=384，每分支降到 96）
    │   cat →                          [B, 768, 32, 32]
    ▼ bottleneck Conv3×3 + Conv1×1 →   [B, 21, 32, 32]
    ▼ 双线性上采样 →                    [B, 21, 512, 512]
```

### 实验三：DINOv3 + SimpleHead（消融）

```
输入图像                [B, 3, 512, 512]
    ▼ DINOv3 ViT-S/16（冻结） →        [B, 384, 32, 32]
    ▼ Conv1×1(384→21) →                [B, 21, 32, 32]
    ▼ 双线性上采样 →                    [B, 21, 512, 512]
```

---

## 环境安装

```bash
pip install torch torchvision
pip install transformers>=4.56.0
pip install tensorboard tqdm
```

---

## 数据准备

### Pascal VOC 2012

解压到 `./data/`：
```
data/VOCdevkit/VOC2012/
    JPEGImages/
    SegmentationClass/
    ImageSets/Segmentation/{train,val,trainval}.txt
```

### SBD 增强训练集（论文使用，10,582 张）

下载 `SegmentationClassAug/` 和 `trainaug.txt`，放到：
```
VOCdevkit/VOC2012/SegmentationClassAug/
VOCdevkit/VOC2012/ImageSets/Segmentation/trainaug.txt
```

---

## 运行命令

### 训练

```bash
# 实验一：ResNet101 + PSPNet（冻结 backbone）
python tools/train.py --config configs/default.yaml

# 实验一变体：全量微调（对标论文）
python tools/train.py --config configs/exp1_finetune.yaml

# 实验二：DINOv3 + PSPNet
python tools/train.py --config configs/exp2_dinov3_psp.yaml

# 实验三：DINOv3 + 简单头（消融）
python tools/train.py --config configs/exp3_dinov3_simple.yaml
```

> batch size 变化时只需同步调整 `--lr`（线性缩放），epochs 由 `max_iters / iters_per_epoch` 自动推算。

| batch_size | lr | iters/epoch | epochs（≈30K） |
|---|---|---|---|
| 16 | 0.01 | 661 | 46 |
| 32 | 0.02 | 330 | 91 |
| 64 | 0.04 | 165 | 182 |

### 评估

```bash
# 多尺度 + 翻转（论文协议）
python tools/evaluate.py \
  --checkpoint checkpoints/exp2_dinov3_psp/best.pth \
  --backbone dinov3 --head psp

# 单尺度快速评估
python tools/evaluate.py \
  --checkpoint checkpoints/exp2_dinov3_psp/best.pth \
  --backbone dinov3 --head psp --scales 1.0 --no_flip
```

---

## 项目结构

```
├── configs/
│   ├── default.yaml            实验一（冻结）
│   ├── exp1_finetune.yaml      实验一（微调）
│   ├── exp2_dinov3_psp.yaml    实验二
│   └── exp3_dinov3_simple.yaml 实验三
├── models/
│   ├── backbone.py             ResNet101Backbone / DINOv3Backbone
│   ├── psp_head.py             PyramidPoolingModule + PSPHead
│   ├── simple_head.py          SimpleHead（1×1 conv）
│   └── segmentor.py            ResNet101PSPNet / Segmentor + 工厂函数
├── datasets/
│   └── voc.py                  VOC 2012 数据加载（支持 trainaug）
└── tools/
    ├── train.py                训练脚本（--max_iters 驱动）
    ├── evaluate.py             多尺度评估
    └── visualize.py            分割结果 + PCA 特征可视化
```

---

## 参考

- PSPNet: [arxiv 1612.01105](https://arxiv.org/abs/1612.01105)
- DINOv3: [arxiv 2508.10104](https://arxiv.org/abs/2508.10104) · [HuggingFace](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m)
- DINOv2: [arxiv 2304.07193](https://arxiv.org/abs/2304.07193)（linear probing 评估协议参考）
- Pascal VOC: [Ultralytics Docs](https://docs.ultralytics.com/datasets/detect/voc)
