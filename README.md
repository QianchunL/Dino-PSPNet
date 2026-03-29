# DINOv3 + PSPNet 语义分割

以 DINOv3 为 backbone，结合 PSPNet 金字塔池化模块，在 Pascal VOC 2012 上验证自监督视觉特征的分割能力。

---

## 实验设计

| 实验 | Backbone | 分割头 | 目的 |
|------|----------|--------|------|
| 实验一 | ResNet101（ImageNet 预训练，冻结） | 金字塔池化 PSPHead | CNN baseline |
| 实验二 | DINOv3 ViT-S/16（自监督，冻结） | 金字塔池化 PSPHead | 核心实验 |
| 实验三 | DINOv3 ViT-S/16（自监督，冻结） | 1×1 卷积 SimpleHead | 消融：PPM 对 ViT 是否必要 |

**实验逻辑链**
- 实验一 vs 实验二：ResNet101 监督特征 vs DINOv3 自监督特征，谁更适合分割？
- 实验二 vs 实验三：金字塔池化模块对 ViT 特征是否还有附加价值？

---

## 模型结构与数据维度

### 实验一：ResNet101 PSPNet（含空洞卷积 + deep supervision）

```
输入图像           [B, 3, 512, 512]
    │
    ▼ stem (conv1+bn+relu+maxpool)
    │                              [B, 64, 128, 128]
    ▼ layer1                       [B, 256, 128, 128]   stride 累计 4
    ▼ layer2                       [B, 512, 64, 64]     stride 累计 8
    ▼ layer3  ← dilation=2         [B, 1024, 64, 64]    stride 累计 8（不再下采样）
    │   │
    │   └──→ AuxHead ──→ aux_logits [B, 21, 64, 64]
    │           Conv3×3(1024→256) + BN + ReLU + Dropout
    │           Conv1×1(256→21)
    │           ↑ 仅训练时计算，loss 权重 0.4；推理时丢弃
    │
    ▼ layer4  ← dilation=4         [B, 2048, 64, 64]    stride 累计 8
    │
    ▼ PyramidPoolingModule
    │   ├── AdaptiveAvgPool(1×1) → Conv1×1(2048→512) → BN+ReLU → upsample → [B, 512, 64, 64]
    │   ├── AdaptiveAvgPool(2×2) → Conv1×1(2048→512) → BN+ReLU → upsample → [B, 512, 64, 64]
    │   ├── AdaptiveAvgPool(3×3) → Conv1×1(2048→512) → BN+ReLU → upsample → [B, 512, 64, 64]
    │   └── AdaptiveAvgPool(6×6) → Conv1×1(2048→512) → BN+ReLU → upsample → [B, 512, 64, 64]
    │   cat(原始特征 + 4 个分支) → [B, 2048+4×512, 64, 64] = [B, 4096, 64, 64]
    │
    ▼ bottleneck
    │   Conv3×3(4096→512) + BN + ReLU + Dropout(0.1)
    │   Conv1×1(512→21)
    │                              [B, 21, 64, 64]
    │
    ▼ 双线性上采样回输入分辨率      [B, 21, 512, 512]
```

**空洞卷积（Dilated Convolution）说明**

原始 ResNet 的 layer3 和 layer4 各自将 stride 设为 2，导致最终输出步长为 32（特征图是输入的 1/32）。PSPNet 将这两层的 3×3 卷积改为空洞卷积：

- layer3：stride 1 + dilation 2（感受野不变，但不缩小分辨率）
- layer4：stride 1 + dilation 4

使输出步长保持在 8，特征图分辨率为输入的 1/8（512 输入 → 64×64 特征图），保留更多空间细节。

**Deep Supervision（深度监督）说明**

辅助头接在 layer3 输出处，在训练时额外施加一个分割损失：

```
total_loss = main_loss + 0.4 × aux_loss
```

目的是给 backbone 中间层提供更强的梯度信号，缓解深层网络梯度消失问题。推理时辅助头完全丢弃，不增加计算开销。

---

### 实验二：DINOv3 + PSPHead

```
输入图像                [B, 3, 512, 512]
    │
    ▼ DINOv3 ViT-S/16（冻结）
    │   patch_size=16 → 512/16 = 32×32 = 1024 个 patch
    │   序列: [CLS(1), registers(4), patches(1024)]  共 1029 个 token
    │   取 patches 部分 → reshape
    │                       [B, 384, 32, 32]
    │
    ▼ PyramidPoolingModule（同实验一，in_channels=384）
    │   每个分支降维到 384//4=96
    │   cat → [B, 384+4×96, 32, 32] = [B, 768, 32, 32]
    │
    ▼ bottleneck
    │   Conv3×3(768→512) + BN + ReLU + Dropout
    │   Conv1×1(512→21)  → [B, 21, 32, 32]
    │
    ▼ 双线性上采样                  [B, 21, 512, 512]
```

---

### 实验三：DINOv3 + SimpleHead（消融）

```
输入图像                [B, 3, 512, 512]
    │
    ▼ DINOv3 ViT-S/16（冻结）  [B, 384, 32, 32]
    │
    ▼ Conv1×1(384→21)          [B, 21, 32, 32]
    │
    ▼ 双线性上采样              [B, 21, 512, 512]
```

---

## 训练配置（按 PSPNet 论文）

| 项目 | 论文设置 | 说明 |
|------|----------|------|
| 优化器 | SGD | momentum=0.9，weight_decay=1e-4 |
| 学习率 | 0.01 | base lr，按 iteration poly decay |
| LR 调度 | Poly | `lr = 0.01 × (1 − iter/max_iter)^0.9`，**按 iteration** 更新（非 epoch） |
| Batch size | 16 | 显存不足可减半，lr 同比例减半（线性缩放规则） |
| 训练轮数 | 50 epochs | — |
| 损失函数 | CrossEntropyLoss | ignore_index=255（VOC 边界像素） |
| 辅助损失权重 | 0.4 | 仅实验一（ResNet101）使用 |
| 训练集 | trainaug（10,582 张） | SBD 增强版，见数据准备 |

**数据增强（训练集）**

| 操作 | 参数 |
|------|------|
| 随机缩放 | 0.5–2.0 |
| 随机水平翻转 | 50% 概率 |
| 随机旋转 | ±10°，边缘用均值色/255 填充 |
| 随机高斯模糊 | 50% 概率，radius 0.1–2.0 |
| 随机裁剪 | 512×512，不足时用均值色填充 |
| 归一化 | mean=(0.485,0.456,0.406)，std=(0.229,0.224,0.225) |

**评估协议（多尺度）**

| 项目 | 设置 |
|------|------|
| 测试尺度 | {0.5, 0.75, 1.0, 1.25, 1.5, 1.75} |
| 水平翻转 | 每个尺度额外做一次翻转，共 12 次前向 |
| 融合方式 | softmax 概率上采样至原图尺寸后平均，再 argmax |
| 指标 | mIoU（混淆矩阵，忽略 label=255） |

---

## 项目结构

```
├── models/
│   ├── backbone.py      ResNet101Backbone（空洞卷积）/ DINOv3Backbone
│   ├── psp_head.py      PyramidPoolingModule + PSPHead
│   ├── simple_head.py   简单 1×1 卷积头（实验三）
│   └── segmentor.py     ResNet101PSPNet（含 AuxHead）/ Segmentor + 工厂函数
├── datasets/
│   └── voc.py           VOC 2012 数据加载（支持 trainaug）
├── configs/
│   └── default.yaml     训练超参数
├── train.py             训练脚本
├── evaluate.py          多尺度评估（mIoU 逐类别）
└── visualize.py         分割对比图 + PCA 特征可视化
```

---

## 环境安装

```bash
pip install torch torchvision
pip install transformers>=4.56.0
pip install tensorboard scikit-learn
```

---

## 数据准备

### Pascal VOC 2012（必须）

下载官方数据集，解压到 `./data/`：

```
data/
  VOCdevkit/
    VOC2012/
      JPEGImages/
      SegmentationClass/
      ImageSets/Segmentation/{train,val,trainval}.txt
```

### SBD 增强训练集（推荐，论文使用）

SBD（Semantic Boundaries Dataset）将训练集扩充到 10,582 张。**需要单独下载**，有两种方式：

**方式一（推荐）：下载打包好的 SegmentationClassAug**

在 GitHub 搜索 `VOC2012 SegmentationClassAug`，下载现成打包的压缩包，解压到：
```
VOCdevkit/VOC2012/SegmentationClassAug/   ← 增强 mask（PNG 格式）
VOCdevkit/VOC2012/ImageSets/Segmentation/trainaug.txt  ← 图片列表
```

**方式二：torchvision 自动下载**

```bash
python -c "
import torchvision.datasets as D
D.SBDataset('./data', image_set='train_noval', download=True)
"
```

> 如果暂时不想处理 SBD，训练时加 `--split_train train` 回退到官方 1,464 张训练集，效果会略差但流程完全一致。

---

## 训练

```bash
# 实验一：ResNet101 + PSPNet（baseline）
python train.py --backbone resnet101 --head psp \
                --experiment exp1_resnet101_pspnet

# 显存不足时减半 batch_size，同步线性缩放 lr
python train.py --backbone resnet101 --batch_size 8 --lr 0.005

# 实验二：DINOv3 + PSPNet（核心实验）
python train.py --backbone dinov3 --head psp \
                --experiment exp2_dinov3_pspnet

# 实验三：DINOv3 + 简单头（消融）
python train.py --backbone dinov3 --head simple \
                --experiment exp3_dinov3_simple

# 使用官方训练集（不用 SBD）
python train.py --backbone resnet101 --split_train train
```

TensorBoard 日志：
```bash
tensorboard --logdir ./runs
```

---

## 评估

```bash
# 多尺度评估（论文设置，较慢）
python evaluate.py --checkpoint checkpoints/exp1_resnet101_pspnet/best.pth \
                   --backbone resnet101 --head psp

# 单尺度快速评估
python evaluate.py --checkpoint ... --scales 1.0 --no_flip
```

输出逐类别 IoU 及 mIoU。

---

## 可视化

```bash
# 分割结果对比（输入 / GT / 预测）
python visualize.py seg --checkpoint checkpoints/exp1_resnet101_pspnet/best.pth \
                        --backbone resnet101 --head psp --num_images 8

# PCA 特征图（对比不同 backbone 的特征质量）
python visualize.py pca --backbone resnet101 --num_images 4
python visualize.py pca --backbone dinov3    --num_images 4
```

---

## 参考

- PSPNet: [arxiv 1612.01105](https://arxiv.org/abs/1612.01105)
- DINOv3: [arxiv 2508.10104](https://arxiv.org/abs/2508.10104) | [HuggingFace](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m)
- Pascal VOC: [Ultralytics Docs](https://docs.ultralytics.com/datasets/detect/voc)
- SBD: [Berkeley EECS](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/)
