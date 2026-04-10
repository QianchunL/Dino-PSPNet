#!/usr/bin/env python
"""
Export LD ONNX that consumes shared camera features (from camera_backbone.onnx).

输入：
  - feat_s32: [B, V, 256, H32, W32]（HybridEncoder stride=32 输出；对应配置 lane_position_level=2）
  - lidar2img: [B, V, 4, 4]

图内包含：
  - lane_feat_proj（1x1 conv + BN + ReLU），把 256 -> lane_dim(默认128)
  - lane_bev_encoder（BEVPerceptionTransformer / BEVFormerEncoder）
  - lane/curb/stopline/arrow heads

运行命令（工作目录：onemodel/bevperception）::

  export CFG=projects/configs/nusc/mv2dfusion-centerpoint-rtdetr_combined_0305_lc_onemodel_0312.py
  export CKPT=work_dirs/mv2dfusion-centerpoint-rtdetr_combined_0305_lc_onemodel_0312/iter_195200.pth
  export OUT=work_dirs/mv2dfusion-centerpoint-rtdetr_combined_0305_lc_onemodel_0312/onnx

  python tools/onnx/export_ld_from_imgfeats_onnx.py \
        --config /mnt/volumes/ad-perception-al-sh01/liqianchun338/bevperception-dev-diffdata-lcfusion-0310-gs/projects/configs/nusc/mv2dfusion-centerpoint-rtdetr_combined_0305_lc_onemodel_0402_lddebug_fullalign.py \
        --checkpoint /mnt/volumes/ad-perception-al-sh01/lhm/onemodel/bevperception/work_dirs/mv2dfusion-centerpoint-rtdetr_combined_0305_lc_onemodel_0312/iter_195200.pth \
        --output onnx_models/ld_from_imgfeats.onnx
"""

import argparse
import math
import os
import sys
import numpy as np

import torch
import torch.nn as nn

from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from export_onnx_plugin import register_custom_symbolic_for_export

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


# ---- MSDA patch (export as custom mmdeploy op) ----
def _patch_msda_for_ld():
    register_custom_symbolic_for_export()


def _get_pad_size(cfg):
    final_h, final_w = None, None
    for src in [cfg, cfg.get("data", {})]:
        if hasattr(src, "ida_aug_conf"):
            final_h, final_w = src.ida_aug_conf["final_dim"]
            break
    if final_h is None:
        final_h, final_w = 576, 1024
    size_div = 32
    h_pad = int(math.ceil(final_h / size_div) * size_div)
    w_pad = int(math.ceil(final_w / size_div) * size_div)
    return h_pad, w_pad


class LDFromImgFeatsForONNX(nn.Module):
    def __init__(self, model, cfg):
        super().__init__()
        self.lane_feat_proj = model.lane_feat_proj
        self.lane_bev_encoder = model.lane_bev_encoder
        self.lane_head = model.lane_head
        self.curb_head = model.curb_head
        self.stopline_head = model.stopline_head
        self.arrow_head = model.arrow_head
        self.curb_bev_neck = getattr(model, "curb_bev_neck", None)
        self.stopline_bev_neck = getattr(model, "stopline_bev_neck", None)
        self.arrow_bev_neck = getattr(model, "arrow_bev_neck", None)

        self.num_views = cfg.model.lane_bev_encoder.get("num_cams", 7)
        self.pad_h, self.pad_w = _get_pad_size(cfg)
        self.bev_h = cfg.get("lane_bev_h", 75)
        self.bev_w = cfg.get("lane_bev_w", 225)

    def _build_img_metas(self, lidar2img_bv):
        # IMPORTANT: keep lidar2img as Tensor to avoid baking it as a constant in ONNX.
        # BEVFormerEncoder.point_sampling supports tensor lidar2img via the non-list branch.
        # Use shape [1, V, 4, 4] so that torch.cat over batch gives (B, V, 4, 4).
        lidar2img_tensor = lidar2img_bv.unsqueeze(0).to(dtype=torch.float32)
        meta = dict(
            lidar2img=lidar2img_tensor,
            img_shape=[(self.pad_h, self.pad_w, 3)] * self.num_views,
            can_bus=[0.0] * 18,
        )
        return [meta]

    def _apply_bev_neck(self, bev_feats, neck):
        if neck is None:
            return bev_feats
        bs = bev_feats.shape[0]
        x = bev_feats.view(bs, self.bev_h, self.bev_w, -1).permute(0, 3, 1, 2).contiguous()
        x = neck(x)
        x = x.permute(0, 2, 3, 1).contiguous().view(bs, self.bev_h * self.bev_w, -1)
        return x

    def forward(self, feat_s32, lidar2img):
        # feat_s32: [B,V,256,H,W] -> project to lane_dim
        B, V, C, H, W = feat_s32.shape
        x = feat_s32.reshape(B * V, C, H, W)
        if self.lane_feat_proj is not None:
            x = self.lane_feat_proj(x)
        # lane_img_feats expects list of [B,V,C_lane,H,W]
        lane_img_feat = x.view(B, V, x.shape[1], x.shape[2], x.shape[3])
        img_metas = self._build_img_metas(lidar2img[0])
        bev_feats, _ = self.lane_bev_encoder([lane_img_feat], None, prev_bev=None, img_metas=img_metas)

        lane_out = self.lane_head(bev_feats, mlvl_feats=[lane_img_feat], img_metas=img_metas)
        curb_out = self.curb_head(self._apply_bev_neck(bev_feats, self.curb_bev_neck), mlvl_feats=[lane_img_feat], img_metas=img_metas)
        stopline_out = self.stopline_head(self._apply_bev_neck(bev_feats, self.stopline_bev_neck), mlvl_feats=[lane_img_feat], img_metas=img_metas)
        arrow_out = self.arrow_head(self._apply_bev_neck(bev_feats, self.arrow_bev_neck), mlvl_feats=[lane_img_feat], img_metas=img_metas)

        return (
            lane_out["all_cls_scores"], lane_out["all_pts_preds"],
            curb_out["all_cls_scores"], curb_out["all_pts_preds"],
            stopline_out["all_cls_scores"], stopline_out["all_pts_preds"],
            arrow_out["all_cls_scores"], arrow_out["all_bbox_preds"],
        )


def build_model_from_config(config_path, checkpoint_path):
    cfg = Config.fromfile(config_path)
    if hasattr(cfg, "plugin") and cfg.plugin:
        import importlib
        plugin_dirs = cfg.plugin_dir if isinstance(cfg.plugin_dir, list) else [cfg.plugin_dir]
        for plugin_dir in plugin_dirs:
            module_parts = os.path.dirname(plugin_dir).split("/")
            module_path = module_parts[0]
            for m in module_parts[1:]:
                module_path = module_path + "." + m
            importlib.import_module(module_path)
    _patch_msda_for_ld()
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, checkpoint_path, map_location="cpu")
    model.eval()
    return model, cfg


def create_dummy_inputs(cfg, device):
    B = 1
    V = cfg.model.lane_bev_encoder.get("num_cams", 7)
    H_pad, W_pad = _get_pad_size(cfg)
    C = 256
    h32, w32 = math.ceil(H_pad / 32), math.ceil(W_pad / 32)
    return dict(
        feat_s32=torch.randn(B, V, C, h32, w32, device=device),
        lidar2img=torch.eye(4, device=device).view(1, 1, 4, 4).repeat(B, V, 1, 1),
    )


def main():
    ap = argparse.ArgumentParser("Export LD-from-imgfeats ONNX")
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--opset", type=int, default=16)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    model, cfg = build_model_from_config(args.config, args.checkpoint)
    wrapper = LDFromImgFeatsForONNX(model, cfg).to(args.device).eval()
    dummy = create_dummy_inputs(cfg, args.device)
    input_names = list(dummy.keys())
    input_tuple = tuple(dummy[k] for k in input_names)

    output_names = [
        "lane_all_cls_scores", "lane_all_pts_preds",
        "curb_all_cls_scores", "curb_all_pts_preds",
        "stopline_all_cls_scores", "stopline_all_pts_preds",
        "arrow_all_cls_scores", "arrow_all_bbox_preds",
    ]
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            input_tuple,
            args.output,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={},
            opset_version=args.opset,
            do_constant_folding=True,
            verbose=False,
        )
    print(f"Exported: {args.output}")


if __name__ == "__main__":
    main()

