"""usage:
python tools/onnx/export_ref_points_onnx.py \
    --cfg /mnt/volumes/ad-perception-al-sh01/liqianchun338/bevperception-dev-diffdata-lcfusion-0310-gs/projects/configs/nusc/mv2dfusion-centerpoint-rtdetr_combined_0305_lc_onemodel_0402_lddebug_fullalign.py \
    --ckpt /mnt/volumes/ad-perception-al-sh01/lhm/onemodel/bevperception/work_dirs/mv2dfusion-centerpoint-rtdetr_combined_0305_lc_onemodel_0312/iter_195200.pth \
    --outpath onnx_models/reference_points_sampling.onnx
"""
import argparse
import importlib
import os
import sys

import torch
from mmcv import Config
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet.models import build_detector
from mmdet3d.models import build_model

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from export_onnx_plugin import DEVICE, export_onnx_model, load_model_and_config


class ReferencePointsSamplingWrapper(torch.nn.Module):
    """Keep a 3-input ONNX interface for reference point sampling."""

    def __init__(self, ref_points_sampling, bev_h, bev_w, num_points_in_pillar):
        super().__init__()
        self.ref_points_sampling = ref_points_sampling
        self.register_buffer("bev_h", torch.tensor(bev_h, dtype=torch.int64))
        self.register_buffer("bev_w", torch.tensor(bev_w, dtype=torch.int64))
        self.register_buffer("num_points_in_pillar", torch.tensor(num_points_in_pillar, dtype=torch.int64))

    def forward(self, pc_range, lidar2img, img_shape):
        return self.ref_points_sampling(
            self.bev_h,
            self.bev_w,
            pc_range,
            self.num_points_in_pillar,
            lidar2img,
            img_shape,
        )


def _resolve_path(path_or_rel, workspace_root):
    if os.path.isabs(path_or_rel):
        return path_or_rel
    return os.path.join(workspace_root, path_or_rel)


def _get_cfg_value(cfg_obj, key, default=None):
    if hasattr(cfg_obj, key):
        return getattr(cfg_obj, key)
    if isinstance(cfg_obj, dict):
        return cfg_obj.get(key, default)
    return default


def _get_encoder_info(model, cfg, encoder_name):
    if not hasattr(model, encoder_name):
        raise RuntimeError(f"Model has no encoder named '{encoder_name}'")

    bev_encoder = getattr(model, encoder_name)
    encoder = getattr(bev_encoder, "encoder", None)
    if encoder is None or not hasattr(encoder, "ref_points_sampling"):
        raise RuntimeError(f"model.{encoder_name}.encoder.ref_points_sampling not found")

    cfg_encoder = _get_cfg_value(cfg.model, encoder_name)
    if cfg_encoder is None:
        raise RuntimeError(f"cfg.model.{encoder_name} not found")

    bev_h = _get_cfg_value(cfg_encoder, "bev_h")
    bev_w = _get_cfg_value(cfg_encoder, "bev_w")
    pc_range = _get_cfg_value(cfg_encoder, "pc_range")

    if bev_h is None and encoder_name == "lane_bev_encoder":
        bev_h = cfg.get("lane_bev_h", None)
    if bev_w is None and encoder_name == "lane_bev_encoder":
        bev_w = cfg.get("lane_bev_w", None)
    if pc_range is None and hasattr(encoder, "pc_range"):
        pc_range = encoder.pc_range

    if bev_h is None or bev_w is None or pc_range is None:
        raise RuntimeError(
            f"Failed to resolve bev_h/bev_w/pc_range for '{encoder_name}': "
            f"bev_h={bev_h}, bev_w={bev_w}, pc_range={pc_range}"
        )

    return encoder, bev_h, bev_w, pc_range


def _resolve_ref_points_components(model, cfg, encoder_name=None):
    candidate_names = []
    if encoder_name is not None:
        candidate_names.append(encoder_name)
    else:
        for name in ["bev_encoder", "lane_bev_encoder"]:
            if hasattr(model, name):
                candidate_names.append(name)

    valid_names = []
    last_error = None
    for name in candidate_names:
        try:
            encoder, bev_h, bev_w, pc_range = _get_encoder_info(model, cfg, name)
            valid_names.append((name, encoder, bev_h, bev_w, pc_range))
        except RuntimeError as exc:
            last_error = exc

    if encoder_name is not None:
        if not valid_names:
            raise last_error or RuntimeError(f"Cannot resolve encoder '{encoder_name}'")
        name, encoder, bev_h, bev_w, pc_range = valid_names[0]
        return encoder, bev_h, bev_w, pc_range, name

    if len(valid_names) == 1:
        name, encoder, bev_h, bev_w, pc_range = valid_names[0]
        return encoder, bev_h, bev_w, pc_range, name

    if len(valid_names) > 1:
        available = ", ".join(name for name, *_ in valid_names)
        raise RuntimeError(
            "Multiple ref_points_sampling encoders found. "
            f"Please specify --encoder-name explicitly: {available}"
        )

    raise last_error or RuntimeError(
        "Cannot find ref_points_sampling encoder path. "
        "Expected model.bev_encoder.encoder.ref_points_sampling or "
        "model.lane_bev_encoder.encoder.ref_points_sampling."
    )


def export_reference_points_only(args):
    current_file_path = os.path.abspath(__file__)
    workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
    config_file = _resolve_path(args.cfg, workspace_root)
    checkpoint_file = _resolve_path(args.ckpt, workspace_root)

    model, cfg, _ = load_model_and_config(config_file, checkpoint_file)

    encoder, bev_h, bev_w, pc_range, encoder_name = _resolve_ref_points_components(
        model, cfg, encoder_name=args.encoder_name
    )
    encoder.eval()
    ref_points_sampling = encoder.ref_points_sampling.eval()
    num_points_in_pillar = encoder.num_points_in_pillar

    print(f"Using encoder path: model.{encoder_name}.encoder.ref_points_sampling")

    bs = 1
    device = DEVICE
    dtype = torch.float32

    if hasattr(cfg, "camera_names"):
        num_cam = len(cfg.camera_names)
    else:
        cfg_encoder = _get_cfg_value(cfg.model, encoder_name)
        num_cam = _get_cfg_value(cfg_encoder, "num_cams", 7)

    lidar2img = torch.randn(bs, num_cam, 4, 4, device=device, dtype=dtype)

    if hasattr(cfg, "input_shape"):
        img_h, img_w = cfg.input_shape
    else:
        img_h, img_w = 576.0, 1024.0
    img_shape = torch.tensor([[img_h, img_w]], device=device, dtype=dtype)

    pc_range_tensor = torch.tensor(pc_range, device=device, dtype=dtype)

    wrapper = ReferencePointsSamplingWrapper(
        ref_points_sampling=ref_points_sampling,
        bev_h=bev_h,
        bev_w=bev_w,
        num_points_in_pillar=num_points_in_pillar,
    ).to(device)
    wrapper.eval()

    if not os.path.exists(args.outpath):
        os.makedirs(args.outpath)
    onnx_path = os.path.join(args.outpath, args.onnx_name)

    export_onnx_model(
        model=wrapper,
        export_inputs=(
            pc_range_tensor,
            lidar2img,
            img_shape,
        ),
        onnx_path=onnx_path,
        input_names=["pc_range", "lidar2img", "img_shape"],
        output_names=["ref_3d", "ref_2d", "reference_points_cam", "bev_mask"],
        simplify_model=True,
        verbose=True,
        print_weights_info=False,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Export reference points sampling ONNX only")
    parser.add_argument(
        "--cfg",
        type=str,
        default="/mnt/volumes/ad-perception-al-sh01/lhm/onemodel/bevperception/projects/configs/nusc/mv2dfusion-centerpoint-rtdetr_combined_0305_lc_onemodel_0312.py",
        help="config file path",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/mnt/volumes/ad-perception-al-sh01/lhm/onemodel/bevperception/work_dirs/mv2dfusion-centerpoint-rtdetr_combined_0305_lc_onemodel_0312/iter_195200.pth",
        help="checkpoint file path",
    )
    parser.add_argument("--outpath", type=str, required=True, help="output directory")
    parser.add_argument(
        "--onnx_name",
        type=str,
        default="reference_points_sampling.onnx",
        help="output onnx filename",
    )
    parser.add_argument(
        "--encoder-name",
        type=str,
        default=None,
        choices=["bev_encoder", "lane_bev_encoder"],
        help="explicit encoder branch to export",
    )
    return parser.parse_args()


if __name__ == "__main__":
    export_reference_points_only(parse_args())
