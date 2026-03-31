# DINOv3 + PSPNet — Claude Code 上下文

项目文档见 README.md，包含实验设计、结果、运行命令和项目结构。

## 背景

课程作业：将 DINOv3（2025，Meta AI，自监督 ViT，`facebook/dinov3-vits16-pretrain-lvd1689m`）作为 backbone，结合 PSPNet 金字塔池化模块，在 Pascal VOC 2012 上做语义分割。评价指标：mIoU（不是 mAP）。

## 关键技术细节

- DINOv3 输出序列：`[CLS(1), registers(4), patches(N)]`，只取 patch tokens reshape 为 2D 特征图
- ResNet101 backbone 做空洞卷积改造（layer3 dilation=2，layer4 dilation=4），输出步长 8
- 冻结 backbone 训练时需保持 BN 在 eval 模式（代码已处理）
- 损失：CrossEntropyLoss + 0.4×aux_loss（仅 ResNet101）；ignore_index=255
- LR：poly decay 按 iteration 更新；`--max_iters 30000`，epochs 自动推算
- 全量微调时 backbone lr = head lr × 0.1（两个 param group）

## 已完成实验结果（VOC 2012 val，多尺度评估）

| 实验 | mIoU |
|------|------|
| ResNet101 冻结 + PSP | 0.7021 |
| ResNet101 微调 + PSP | 0.7873 |
| DINOv3 冻结 + PSP | **0.8239** |
| DINOv3 冻结 + 1×1 conv | 0.8127 |
