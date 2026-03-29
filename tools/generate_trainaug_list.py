"""
生成 trainaug.txt：从 SegmentationClassAug/ 目录的文件名构建增强训练集列表，
并剔除 val.txt 中的验证集图片，避免 train/val 污染。

用法：
    python tools/generate_trainaug_list.py --data_root ./data

生成文件：
    data/VOCdevkit/VOC2012/ImageSets/Segmentation/trainaug.txt
"""

import argparse
from pathlib import Path


def main(args):
    voc_root = Path(args.data_root) / "VOCdevkit" / "VOC2012"
    aug_dir  = voc_root / "SegmentationClassAug"
    seg_dir  = voc_root / "ImageSets" / "Segmentation"

    if not aug_dir.exists():
        raise FileNotFoundError(f"找不到 {aug_dir}，请先解压 SegmentationClassAug/")

    # 读取验证集 ID（需要剔除）
    val_ids = set()
    with open(seg_dir / "val.txt") as f:
        for line in f:
            val_ids.add(line.strip())

    # 从 SegmentationClassAug/ 中的 PNG 文件名提取 ID
    aug_ids = sorted(p.stem for p in aug_dir.glob("*.png"))

    # 剔除验证集
    trainaug_ids = [i for i in aug_ids if i not in val_ids]

    out_file = seg_dir / "trainaug.txt"
    with open(out_file, "w") as f:
        f.write("\n".join(trainaug_ids) + "\n")

    print(f"[done] trainaug.txt 生成完毕")
    print(f"  SegmentationClassAug/ 共 {len(aug_ids)} 张")
    print(f"  剔除 val 集 {len(val_ids)} 张（实际重叠 {len(aug_ids) - len(trainaug_ids)} 张）")
    print(f"  最终 trainaug: {len(trainaug_ids)} 张")
    print(f"  输出: {out_file}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="./data")
    main(p.parse_args())
