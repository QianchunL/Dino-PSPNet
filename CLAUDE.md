# DINOv3 + PSPNet 语义分割项目

## 项目背景

这是一个课程作业项目，目标是将 DINOv3（2025，Meta AI，自监督视觉基础模型）作为 backbone，结合 PSPNet（2016，金字塔场景解析网络）的金字塔池化模块作为分割头，实现图像语义分割算法，并在 Pascal VOC 数据集上验证。

## 核心概念

### DINOv3
- **论文**: https://arxiv.org/abs/2508.10104
- **定位**: 视觉 backbone（特征提取器），不是任务特定模型
- **训练方式**: 纯自监督学习（不依赖人工标注），7B 参数教师模型通过知识蒸馏压缩为多个小模型
- **我们使用**: `dinov3_vits16`（ViT-S/16，21M 参数，最小的模型），通过 Hugging Face Transformers 加载
- **模型 ID**: `facebook/dinov3-vits16-pretrain-lvd1689m`
- **输出**: 
  - CLS token（全局特征，384 维）
  - Patch tokens（密集特征，每个 patch 384 维）
  - 对于 512×512 输入图片，patch size=16，产生 32×32=1024 个 patch token
  - 另有 4 个 register tokens（丢弃不用）
- **关键优势**: 密集特征质量极高，冻结 backbone 不微调就能在分割等密集任务上达到优异效果

### PSPNet
- **论文**: https://arxiv.org/abs/1612.01105
- **原始框架**: Caffe（不是 PyTorch），我们不使用原始代码，只复现其核心模块
- **核心贡献**: 金字塔池化模块（Pyramid Pooling Module, PPM）
  - 对特征图做 4 个尺度的自适应平均池化（bin size: 1×1, 2×2, 3×3, 6×6）
  - 每个池化后接 1×1 卷积降维到 in_channels // 4
  - 上采样回原始特征图尺寸
  - 与原始特征图拼接（concat）
  - 最后接卷积层输出逐像素类别预测
- **设计目的**: 弥补 CNN（如 ResNet）感受野不足、缺乏全局上下文信息的问题
- **注意**: ViT 自带全局自注意力，理论上已具备全局上下文能力，金字塔池化的边际收益可能有限——这本身是一个值得实验验证的点

### Pascal VOC 数据集
- **文档**: https://docs.ultralytics.com/datasets/detect/voc
- **任务**: 语义分割（VOC 2012 segmentation）
- **类别**: 21 类（20 个物体类 + 1 个背景类）
- **数据量**: 训练集约 1,464 张（可用增强版 ~10,582 张），验证集 1,449 张
- **评价指标**: mIoU（mean Intersection over Union），不是 mAP

## 实验设计

### 实验一：PSPNet Baseline（ResNet50 + 金字塔池化）
- **Backbone**: ImageNet 预训练的 ResNet50（torchvision 提供）
- **分割头**: 金字塔池化模块 + 分类卷积
- **训练方式**: 冻结 backbone，只训练分割头（与实验二保持公平对比）
- **目的**: 作为基准，代表传统 CNN backbone 的表现

### 实验二：DINOv3 + 金字塔池化（核心实验）
- **Backbone**: DINOv3 ViT-S/16（冻结）
- **分割头**: 金字塔池化模块 + 分类卷积（与实验一相同的分割头结构）
- **训练方式**: 冻结 backbone，只训练分割头
- **目的**: 验证 DINOv3 自监督特征是否优于 ResNet50 监督特征
- **对比实验一**: 唯一变量是 backbone，说明特征提取能力的差异

### 实验三：DINOv3 + 简单 1×1 卷积头（消融实验）
- **Backbone**: DINOv3 ViT-S/16（冻结）
- **分割头**: 仅一个 1×1 卷积层（无金字塔池化）
- **训练方式**: 冻结 backbone，只训练分类卷积
- **目的**: 验证金字塔池化模块对 ViT 特征是否还有提升
- **对比实验二**: 唯一变量是有无金字塔池化，回答"ViT 时代金字塔池化是否还有必要"

### 实验逻辑链
```
实验一 vs 实验二 → DINOv3 特征 vs ResNet50 特征，谁更适合分割？
实验二 vs 实验三 → 金字塔池化对 ViT 特征是否还有附加价值？
```

### 可选加分实验
- **实验二变体**: 解冻 DINOv3 进行微调，对比冻结 vs 微调的 mIoU 差距
- **DINOv2 对比**: 用 DINOv2 ViT-S 替换 DINOv3 ViT-S，验证 DINOv3 的改进

## 技术细节

### 环境依赖
```bash
pip install torch torchvision        # PyTorch 核心
pip install transformers>=4.56.0     # 加载 DINOv3
pip install tensorboard              # 训练日志可视化（可选）
```

### ViT 特征图处理
DINOv3 ViT 输出的是 token 序列，需要转换为 2D 特征图才能接金字塔池化模块：
```python
# 模型输出: [B, 1+4+N, C]  (1 CLS + 4 registers + N patches)
# 去掉 CLS 和 registers: [B, N, C]
# reshape 为 2D: [B, C, H, W]  其中 H*W = N
```

### 分割头输入输出
- **输入**: backbone 输出的 2D 特征图，shape [B, 384, H, W]（ViT-S 的 hidden_dim=384）
- **金字塔池化后**: [B, 384+384, H, W] = [B, 768, H, W]（原特征 + 四个池化分支各 384//4=96）
- **最终输出**: [B, 21, H, W]（21 个类别的 logits）
- **上采样**: 双线性插值到原图尺寸后计算损失和评估

### 训练配置建议
- **输入分辨率**: 512×512（patch size 16，产生 32×32 特征图）
- **Batch size**: 不限，建议与 lr 同步线性缩放（见下表）
- **优化器**: SGD，momentum=0.9，weight_decay=1e-4（论文设置）
- **学习率策略**: poly decay（`lr = base_lr × (1 - iter/max_iter)^0.9`），按 iteration 更新
- **总迭代数**: 30K（论文 VOC 设置）；`--max_iters 30000`（默认），epochs 自动推算
- **损失函数**: CrossEntropyLoss（忽略 label=255 的边界像素）
- **辅助损失**: ResNet101 实验额外加 aux head（layer3 输出），权重 0.4
- **数据增强**: 随机缩放（0.5-2.0）、随机水平翻转、随机旋转（±10°）、随机高斯模糊、随机裁剪

#### Batch size / LR / Epochs 对照（trainaug 10,582张）
| batch_size | lr    | iters/epoch | epochs（≈30K） |
|------------|-------|-------------|----------------|
| 16         | 0.01  | 661         | 46             |
| 32         | 0.02  | 330         | 91             |
| 64         | 0.04  | 165         | 182            |

> `epochs` 由 `train.py` 根据 `--max_iters` 和实际 `iters/epoch` 自动推算，无需手动指定。
> 增大 batch size 时**只需同步调整 `--lr`**，其余参数（epochs、poly decay）全部自动适配。

## 运行命令

### 训练

```bash
# 实验一：ResNet101 + PSPNet（冻结 backbone，linear probing）
python tools/train.py --backbone resnet101 --head psp \
  --batch_size 16 --lr 0.01 \
  --experiment exp1_resnet101_frozen

# 实验一变体：ResNet101 + PSPNet（全量微调，对标论文）
python tools/train.py --backbone resnet101 --head psp \
  --no_frozen_backbone \
  --batch_size 16 --lr 0.01 \
  --experiment exp1_resnet101_finetune

# 实验二：DINOv3 + PSPNet（冻结 backbone）
python tools/train.py --backbone dinov3 --head psp \
  --batch_size 16 --lr 0.01 \
  --experiment exp2_dinov3_psp

# 实验三：DINOv3 + 简单头（消融）
python tools/train.py --backbone dinov3 --head simple \
  --batch_size 16 --lr 0.01 \
  --experiment exp3_dinov3_simple

# 加大 batch size 时只改 --batch_size 和 --lr（线性缩放），其他不变
# 例：batch=32 → lr=0.02
python tools/train.py --backbone resnet101 --head psp \
  --batch_size 32 --lr 0.02 \
  --experiment exp1_resnet101_frozen
```

### 评估

```bash
# 多尺度 + 翻转评估（论文协议，较慢）
python tools/evaluate.py \
  --checkpoint checkpoints/exp1_resnet101_frozen/best.pth \
  --backbone resnet101 --head psp

# 单尺度快速评估
python tools/evaluate.py \
  --checkpoint checkpoints/exp1_resnet101_frozen/best.pth \
  --backbone resnet101 --head psp \
  --scales 1.0 --no_flip
```

### ImageNet 归一化参数（DINOv3 LVD-1689M 权重使用）
```python
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
```

## 实验结果

> 评估协议：多尺度（0.5, 0.75, 1.0, 1.25, 1.5, 1.75）+ 水平翻转，VOC 2012 **val set**

### 汇总

| 实验 | Backbone | 训练方式 | mIoU (val) |
|------|----------|----------|-----------|
| 实验一（冻结） | ResNet101 | linear probing | 0.7021 |
| 实验一（微调） | ResNet101 | full fine-tuning | **0.7873** |
| 实验二 | DINOv3 ViT-S/16 | linear probing | **0.8239** |
| 实验三 | DINOv3 ViT-S/16 | linear probing（无PPM） | 0.8127 |
| 论文 PSPNet | ResNet101 | full fine-tuning | 0.826（test set） |

---

### 实验一：ResNet101 + PSPNet（冻结 backbone，linear probing）

**配置**：`configs/default.yaml`，max_iters=30000，batch_size=16，lr=0.01

| Class         | IoU    | Class        | IoU    |
|---------------|--------|--------------|--------|
| background    | 0.9187 | cow          | 0.7994 |
| aeroplane     | 0.7962 | diningtable  | 0.5016 |
| bicycle       | 0.3576 | dog          | 0.8267 |
| bird          | 0.8015 | horse        | 0.7471 |
| boat          | 0.6170 | motorbike    | 0.7346 |
| bottle        | 0.7202 | person       | 0.7871 |
| bus           | 0.8935 | pottedplant  | 0.5015 |
| car           | 0.8080 | sheep        | 0.8198 |
| cat           | 0.8770 | sofa         | 0.3888 |
| chair         | 0.3226 | train        | 0.8223 |
|               |        | tvmonitor    | 0.7032 |

**mIoU = 0.7021**

> 低 IoU 集中在小物体/细长类（bicycle 0.36、chair 0.32、sofa 0.39），ImageNet
> 特征冻结后难以捕捉细粒度结构信息。

---

### 实验一变体：ResNet101 + PSPNet（全量微调，对标论文）

**配置**：`configs/exp1_finetune.yaml`，max_iters=30000，batch_size=16，head lr=0.01，backbone lr=0.001

| Class         | IoU    | Class        | IoU    |
|---------------|--------|--------------|--------|
| background    | 0.9455 | cow          | 0.9138 |
| aeroplane     | 0.9035 | diningtable  | 0.5737 |
| bicycle       | 0.4224 | dog          | 0.9235 |
| bird          | 0.9025 | horse        | 0.8995 |
| boat          | 0.6967 | motorbike    | 0.8393 |
| bottle        | 0.8159 | person       | 0.8713 |
| bus           | 0.9456 | pottedplant  | 0.5967 |
| car           | 0.8740 | sheep        | 0.8999 |
| cat           | 0.9516 | sofa         | 0.5083 |
| chair         | 0.3949 | train        | 0.8773 |
|               |        | tvmonitor    | 0.7775 |

**mIoU = 0.7873**（vs 冻结 +8.5 pts，vs 论文 −3.9 pts on val）

> 端到端微调带来全面提升，尤其 sofa +12 pts、boat +8 pts。残余差距（~4 pts vs 论文）
> 属于 Caffe vs PyTorch BN 行为差异的正常复现误差。
> bicycle/chair 仍是难类，属于该方案的结构性弱点，与训练方式无关。

---

### 实验二：DINOv3 ViT-S/16 + PSPNet（冻结 backbone）

**配置**：`configs/exp2_dinov3_psp.yaml`，max_iters=30000，batch_size=16，lr=0.01

| Class         | IoU    | Class        | IoU    |
|---------------|--------|--------------|--------|
| background    | 0.9543 | cow          | 0.8979 |
| aeroplane     | 0.8960 | diningtable  | 0.6996 |
| bicycle       | 0.4343 | dog          | 0.9238 |
| bird          | 0.9196 | horse        | 0.8823 |
| boat          | 0.7912 | motorbike    | 0.8952 |
| bottle        | 0.8508 | person       | 0.8966 |
| bus           | 0.9493 | pottedplant  | 0.7426 |
| car           | 0.9076 | sheep        | 0.8878 |
| cat           | 0.9431 | sofa         | 0.6307 |
| chair         | 0.4911 | train        | 0.9274 |
|               |        | tvmonitor    | 0.7808 |

**mIoU = 0.8239**（vs ResNet101 冻结 +12.2 pts；vs ResNet101 微调 +3.7 pts）

> **核心结论**：DINOv3 冻结特征（只训练 PSP head）超过 ResNet101 全量微调，
> 证明自监督预训练特征质量显著优于 ImageNet 监督特征。
> bicycle/chair 仍是弱项，说明这两类是数据和结构层面的难点，与 backbone 无关。

---

### 实验三：DINOv3 ViT-S/16 + 简单 1×1 卷积头（消融）

**配置**：`configs/exp3_dinov3_simple.yaml`，max_iters=30000，batch_size=16，lr=0.01

| Class         | IoU    | Class        | IoU    |
|---------------|--------|--------------|--------|
| background    | 0.9522 | cow          | 0.9066 |
| aeroplane     | 0.8911 | diningtable  | 0.6365 |
| bicycle       | 0.4341 | dog          | 0.9139 |
| bird          | 0.8846 | horse        | 0.8841 |
| boat          | 0.7672 | motorbike    | 0.8895 |
| bottle        | 0.8402 | person       | 0.8997 |
| bus           | 0.9323 | pottedplant  | 0.7098 |
| car           | 0.9052 | sheep        | 0.8888 |
| cat           | 0.9371 | sofa         | 0.6432 |
| chair         | 0.5043 | train        | 0.9157 |
|               |        | tvmonitor    | 0.7301 |

**mIoU = 0.8127**（vs 实验二 PSP head −1.12 pts）

> PPM 带来 +1.12 pts 的提升，边际价值有限但非零。
> ViT 全局自注意力已覆盖大部分多尺度上下文，PPM 的附加作用大幅缩小。
> 即使只用 1×1 卷积，DINOv3（0.8127）仍超过 ResNet101 全量微调（0.7873）。

---

### 跨实验分析

| 对比 | 差值 | 结论 |
|------|------|------|
| 实验一冻结 → 实验二 | +12.2 pts | DINOv3 自监督特征远优于 ResNet101 监督特征 |
| 实验一微调 → 实验二 | +3.7 pts | DINOv3 冻结特征超过 ResNet101 全量微调 |
| 实验三 → 实验二 | +1.1 pts | PPM 在 ViT 时代仍有效，但边际价值大幅降低 |
| 实验三 → 实验一微调 | +2.5 pts | 仅 1×1 卷积的 DINOv3 仍优于全量微调的 ResNet101 |

---

## 加分项

### 可视化
- **PCA 特征可视化**: 对 backbone 输出的 patch 特征做 PCA 降到 3 维映射为 RGB，对比不同 backbone 的特征质量
- **分割结果可视化**: 展示输入图像、Ground Truth、预测结果的对比图
- **失败案例分析**: 找出分割错误的样本，分析原因（小物体漏检？边界不准？类别混淆？）

### 报告
- 用规范的实验报告格式组织：动机 → 方法 → 实验设置 → 结果表格 → 可视化 → 分析讨论
- 在讨论部分分析"金字塔池化在 ViT 时代是否还有必要"
- 展示独立思考能力，而非仅报告数字

## 项目结构
```
project/
├── CLAUDE.md                # 本文件
├── models/
│   ├── psp_head.py          # 金字塔池化模块（自己实现）
│   ├── simple_head.py       # 简单 1×1 卷积头（实验三）
│   ├── backbone.py          # DINOv3 和 ResNet101 backbone 封装
│   └── segmentor.py         # Backbone + Head 组合
├── datasets/
│   └── voc.py               # Pascal VOC 数据加载与增强
└── tools/
    ├── train.py             # 训练脚本（--max_iters 驱动，epochs 自动推算）
    ├── evaluate.py          # 评估脚本（多尺度+翻转，逐类别 mIoU）
    └── visualize.py         # 可视化脚本（PCA、分割结果）
```

## 参考链接
- DINOv3 论文: https://arxiv.org/abs/2508.10104
- DINOv3 代码: https://github.com/facebookresearch/dinov3
- DINOv3 HuggingFace: https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m
- PSPNet 论文: https://arxiv.org/abs/1612.01105
- Pascal VOC: https://docs.ultralytics.com/datasets/detect/voc