#!/usr/bin/env python
"""
Export RT-DETR query generation part to ONNX (consumes HybridEncoder features).

This exporter is meant to be used with:
  - `tools/onnx/export_camera_backbone_onnx.py`  -> outputs feat_s8/s16/s32
  - (optional) `tools/onnx/export_od_head.py`    -> consumes RT-DETR outputs + LiDAR tokens/queries + memory

This ONNX contains:
  - RTDETRQueryHead "encoder output" branch (anchors + TopK per cam)
  - depth_head + center2lidar projection
  - spatial_alignment on image features (for fusion head)

Inputs (static, default B=1, V=7):
  - feat_s8  : [B, V, 256, H8,  W8]
  - feat_s16 : [B, V, 256, H16, W16]
  - feat_s32 : [B, V, 256, H32, W32]
  - intrinsics      : [B, V, 4, 4]
  - extrinsics      : [B, V, 4, 4]
  - extrinsics_inv  : [B, V, 4, 4]

Outputs:
  - feat_flatten_img : [V, L_img, 256]     (aligned)
  - spatial_flatten_img : [num_levels, 2]
  - lidar2img        : [B, V, 4, 4]
  - dyn_query        : [B, global_topk, prob_bin, 4]
  - query_feats      : [B, global_topk, 256]

Notes:
  - This exporter keeps shapes static (no dynamic_axes). Change resolution/cams => re-export.
  - This implementation reuses helper functions from `export_rtdetr_image_branch_onnx.py`.

Usage:
python tools/onnx/export_rtdetr_from_imgfeats_onnx.py \
  --config /mnt/volumes/ad-perception-al-sh01/liqianchun338/bevperception/projects/configs/nusc/mv2dfusion-centerpoint-rtdetr_hellodata_0320_02voxel_128dim_debug.py \
  --checkpoint /mnt/volumes/ad-perception-al-sh01/lhm/onemodel/bevperception/work_dirs/mv2dfusion-centerpoint-rtdetr_combined_0305_lc_onemodel_0312/iter_195200.pth \
  --output onnx_models/rtdetr_from_imgfeats_0312.onnx \
  --topk-per-cam 100 \
  --global-topk 200
"""

import argparse
import importlib.util
import inspect
import math
import os
import sys

import torch
import torch.nn as nn

from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet.models import DETECTORS, build_detector


def _load_module_from_path(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {module_name} from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


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

    attn_cfgs = (
        cfg.model.get("fusion_bbox_head", {})
        .get("transformer", {})
        .get("decoder", {})
        .get("transformerlayers", {})
        .get("attn_cfgs", [])
    )
    for attn_cfg in attn_cfgs:
        if attn_cfg.get("type") != "MixedCrossAttention":
            continue
        for key in ("img_embed_dims", "pts_embed_dims"):
            if key in attn_cfg:
                print(
                    f"[build_model_from_config] drop unsupported attention cfg key "
                    f"for MixedCrossAttention: {key}"
                )
                attn_cfg.pop(key)

    detector_type = cfg.model["type"]
    detector_cls = DETECTORS.get(detector_type)
    if detector_cls is not None:
        detector_sig = inspect.signature(detector_cls.__init__)
        unsupported_keys = [
            key for key in list(cfg.model.keys())
            if key != "type" and key not in detector_sig.parameters
        ]
        for key in unsupported_keys:
            print(
                f"[build_model_from_config] drop unsupported model cfg key "
                f"for {detector_type}: {key}"
            )
            cfg.model.pop(key)
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, checkpoint_path, map_location="cpu", strict=False)
    model.eval()
    return model, cfg


def _resolve_output_path(output_path: str) -> str:
    if output_path.endswith(".onnx"):
        return output_path
    if os.path.isdir(output_path) or os.path.splitext(output_path)[1] == "":
        return os.path.join(output_path, "rtdetr_from_imgfeats.onnx")
    return output_path


class RTDETRFromImgFeatsForONNX(nn.Module):
    def __init__(self, model, cfg, helpers_mod, topk_per_cam=100, global_topk=200):
        super().__init__()
        qh = model.rtdetr_query_head
        self.input_proj = qh.input_proj
        self.enc_output = qh.enc_output
        self.enc_score_head = qh.enc_score_head
        self.enc_bbox_head = qh.enc_bbox_head
        self.depth_head = qh.depth_head
        self.register_buffer("depth_bins", qh.depth_bins.clone())
        self.hidden_dim = qh.hidden_dim
        self.num_classes = qh.num_classes
        self.prob_bin = qh.prob_bin

        self.spatial_alignment = model.fusion_bbox_head.spatial_alignment
        self.register_buffer("pc_range", model.fusion_bbox_head.pc_range.clone())

        head_cfg = cfg.model.get("fusion_bbox_head", {})
        self.img_feat_start_level = int(head_cfg.get("img_feat_start_level", 0))
        decoder_layer_cfg = head_cfg.get("transformer", {}).get("decoder", {}).get("transformerlayers", {})
        cross_attn_cfg = decoder_layer_cfg.get("attn_cfgs", [{}])[-1]
        self.num_img_feat_levels = int(cross_attn_cfg.get("num_levels", 2))

        self.topk_per_cam = int(topk_per_cam)
        self.global_topk = int(global_topk)

        # Reuse center2lidar helper only; keep local anchor generation to avoid inf constants in ONNX.
        self._center2lidar = helpers_mod.RTDETRImageBranchForONNX._center2lidar

        # Cache pad shape as python floats for center_2d conversion
        H_pad, W_pad = _get_pad_size(cfg)
        self._H_pad = float(H_pad)
        self._W_pad = float(W_pad)

    @staticmethod
    def _generate_anchors_safe(spatial_shapes, device, grid_size=0.05, eps=1e-2):
        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(
                torch.arange(h, dtype=torch.float32, device=device),
                torch.arange(w, dtype=torch.float32, device=device),
                indexing="ij",
            )
            grid_xy = torch.stack([grid_x, grid_y], dim=-1)
            valid_wh = torch.tensor([w, h], dtype=torch.float32, device=device)
            grid_xy = (grid_xy + 0.5) / valid_wh
            wh = torch.ones_like(grid_xy) * grid_size * (2.0**lvl)
            anchor = torch.cat([grid_xy, wh], dim=-1).reshape(-1, 4)
            anchors.append(anchor)
        anchors = torch.cat(anchors, dim=0).unsqueeze(0)
        valid_mask = ((anchors > eps) & (anchors < 1 - eps)).all(-1, keepdim=True)
        safe_anchors = anchors.clamp(min=eps, max=1 - eps)
        anchors = torch.log(safe_anchors / (1 - safe_anchors))
        anchors = torch.where(valid_mask, anchors, torch.full_like(anchors, 1e4))
        return anchors, valid_mask

    def forward(self, feat_s8, feat_s16, feat_s32, intrinsics, extrinsics, extrinsics_inv):
        # feat_s*: [B,V,256,H,W], B assumed 1 for exporter
        B, V, C, H8, W8 = feat_s8.shape
        device = feat_s8.device
        K = self.topk_per_cam

        # encoder feats as [V,256,H,W]
        enc_feat_s8 = feat_s8.reshape(B * V, C, H8, W8)
        enc_feat_s16 = feat_s16.reshape(B * V, C, feat_s16.shape[-2], feat_s16.shape[-1])
        enc_feat_s32 = feat_s32.reshape(B * V, C, feat_s32.shape[-2], feat_s32.shape[-1])
        encoder_feats = [enc_feat_s8, enc_feat_s16, enc_feat_s32]

        # input_proj + flatten
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(encoder_feats)]
        spatial_shapes_enc = []
        feat_list_enc = []
        for feat in proj_feats:
            _, _, h, w = feat.shape
            spatial_shapes_enc.append((h, w))
            feat_list_enc.append(feat.flatten(2).permute(0, 2, 1))
        memory = torch.cat(feat_list_enc, dim=1)  # [V, L, C]

        anchors, valid_mask = self._generate_anchors_safe(spatial_shapes_enc, device)
        valid_mask_float = valid_mask.to(memory.dtype)
        output_memory = self.enc_output["norm"](self.enc_output["proj"](memory * valid_mask_float))

        enc_cls = self.enc_score_head(output_memory)             # [V, L, num_classes]
        enc_bbox = self.enc_bbox_head(output_memory) + anchors   # [V, L, 4]

        # TopK per cam
        per_anchor_scores = enc_cls.max(-1).values
        per_anchor_scores = per_anchor_scores.masked_fill(~valid_mask.squeeze(-1), -1e8)
        topk_scores, topk_ind = torch.topk(per_anchor_scores, K, dim=1)  # [V,K]
        topk_ind_expand = topk_ind.unsqueeze(-1)
        query_feats = output_memory.gather(dim=1, index=topk_ind_expand.expand(-1, -1, self.hidden_dim))  # [V,K,C]
        ref_boxes = torch.sigmoid(enc_bbox.gather(dim=1, index=topk_ind_expand.expand(-1, -1, 4)))  # [V,K,4]

        # Depth estimation
        intrinsics0 = intrinsics.squeeze(0)        # [V,4,4]
        extrinsics0 = extrinsics.squeeze(0)
        extrinsics_inv0 = extrinsics_inv.squeeze(0)
        intr_per_q = intrinsics0[:, None].expand(-1, K, -1, -1).reshape(V * K, 4, 4)
        ext_inv_per_q = extrinsics_inv0[:, None].expand(-1, K, -1, -1).reshape(V * K, 4, 4)
        intr_feat = intr_per_q.reshape(V * K, 16) * 0.01
        depth_input = torch.cat([query_feats.detach().reshape(V * K, -1), intr_feat], dim=-1)
        depth_logits = self.depth_head(depth_input)
        depth_logits = torch.nan_to_num(depth_logits, nan=0.0, posinf=1e4, neginf=-1e4)
        depth_prob = torch.softmax(depth_logits.float(), dim=-1)  # [V*K, prob_bin]

        # Project depth bins to lidar
        ref_cx = ref_boxes[..., 0]
        ref_cy = ref_boxes[..., 1]
        center_2d = torch.stack([ref_cx.reshape(-1) * self._W_pad, ref_cy.reshape(-1) * self._H_pad], dim=-1)  # [V*K,2]
        depth_bin_vals = self.depth_bins.unsqueeze(0).expand(V * K, -1)
        center_sample = self._center2lidar(self, center_2d, depth_bin_vals, intr_per_q, ext_inv_per_q)  # [V*K,D,3]

        dyn_query = torch.cat([center_sample, depth_prob.unsqueeze(-1)], dim=-1)  # [V*K,D,4]
        query_feats_flat = query_feats.detach().reshape(V * K, -1)        # [V*K,C]

        # Global TopK
        all_scores = topk_scores.reshape(-1)
        _, global_indices = all_scores.topk(self.global_topk, dim=0, largest=True, sorted=True)
        dyn_query = dyn_query[global_indices]
        query_feats_out = query_feats_flat[global_indices]

        # Spatial alignment (use selected levels)
        intrinsics_scaled = intrinsics0 / 1e3
        ext_3x4 = extrinsics0[..., :3, :]
        mln_input = torch.cat([
            intrinsics_scaled[..., 0, 0:1],
            intrinsics_scaled[..., 1, 1:2],
            ext_3x4.flatten(-2),
        ], dim=-1).unsqueeze(1)  # [V,1,14]

        start_lvl = self.img_feat_start_level
        end_lvl = start_lvl + self.num_img_feat_levels
        feat_list = []
        spatial_shapes = []
        for i in range(start_lvl, end_lvl):
            feat = encoder_feats[i]
            V_i, C_i, H_i, W_i = feat.shape
            feat_list.append(feat.reshape(V_i, C_i, -1).transpose(1, 2))
            spatial_shapes.append([H_i, W_i])
        feat_flatten_img = torch.cat(feat_list, dim=1)
        feat_flatten_img = self.spatial_alignment(feat_flatten_img, mln_input).float()
        spatial_flatten_img = torch.tensor(spatial_shapes, dtype=torch.long, device=device)

        lidar2img = torch.bmm(intrinsics0, extrinsics0).unsqueeze(0)  # [B,V,4,4]
        return (
            feat_flatten_img,
            spatial_flatten_img,
            lidar2img,
            dyn_query.unsqueeze(0),
            query_feats_out.unsqueeze(0),
        )


def create_dummy_inputs(cfg, device):
    B = 1
    V = cfg.model.fusion_bbox_head.transformer.decoder.transformerlayers.attn_cfgs[1].get("num_cams", 7)
    H_pad, W_pad = _get_pad_size(cfg)
    C = 256
    h8, w8 = math.ceil(H_pad / 8), math.ceil(W_pad / 8)
    h16, w16 = math.ceil(H_pad / 16), math.ceil(W_pad / 16)
    h32, w32 = math.ceil(H_pad / 32), math.ceil(W_pad / 32)
    return dict(
        feat_s8=torch.randn(B, V, C, h8, w8, device=device),
        feat_s16=torch.randn(B, V, C, h16, w16, device=device),
        feat_s32=torch.randn(B, V, C, h32, w32, device=device),
        intrinsics=torch.eye(4, device=device).view(1, 1, 4, 4).repeat(B, V, 1, 1),
        extrinsics=torch.eye(4, device=device).view(1, 1, 4, 4).repeat(B, V, 1, 1),
        extrinsics_inv=torch.eye(4, device=device).view(1, 1, 4, 4).repeat(B, V, 1, 1),
    )


def main():
    ap = argparse.ArgumentParser("Export RTDETR-from-imgfeats ONNX")
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--opset", type=int, default=16)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--topk-per-cam", type=int, default=100)
    ap.add_argument("--global-topk", type=int, default=200)
    args = ap.parse_args()

    # Ensure project root in path for plugin imports
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, project_root)

    helpers = _load_module_from_path(
        "export_rtdetr_image_branch_onnx",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "export_rtdetr_image_branch_onnx.py"),
    )

    model, cfg = build_model_from_config(args.config, args.checkpoint)
    wrapper = RTDETRFromImgFeatsForONNX(model, cfg, helpers, args.topk_per_cam, args.global_topk).to(args.device).eval()

    dummy = create_dummy_inputs(cfg, args.device)
    input_names = list(dummy.keys())
    input_tuple = tuple(dummy[k] for k in input_names)
    output_names = ["feat_flatten_img", "spatial_flatten_img", "lidar2img", "dyn_query", "query_feats"]

    output_path = _resolve_output_path(args.output)
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            input_tuple,
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={},
            opset_version=args.opset,
            do_constant_folding=True,
            verbose=False,
        )
    print(f"Exported: {output_path}")


if __name__ == "__main__":
    main()

