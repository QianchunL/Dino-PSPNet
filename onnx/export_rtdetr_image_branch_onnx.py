#!/usr/bin/env python
"""Export RT-DETR image branch (PResNet + HybridEncoder + RTDETRQueryHead) to ONNX.

Pipeline:
  img [B, N_cam, 3, H, W] → PResNet → HybridEncoder → RTDETRQueryHead
    → TopK per view (100) → depth estimation → center2lidar
    → Global TopK (200) → spatial_alignment → outputs for fusion

Outputs (same interface as export_image_branch_onnx.py):
  - feat_flatten_img  [N_cam, L_img, C]
  - spatial_flatten_img [num_levels, 2]
  - lidar2img  [1, N_cam, 4, 4]
  - dyn_query  [1, global_topk, prob_bin, 4]
  - query_feats [1, global_topk, C]

Usage:
  python tools/export_rtdetr_image_branch_onnx.py \\
      --config /mnt/volumes/ad-perception-al-sh01/cm/mv2d/projects/configs/nusc/mv2dfusion-centerpoint-rtdetr_hellodata.py \\
      --checkpoint /mnt/volumes/ad-perception-al-sh01/cm/mv2d/work_dirs/mv2dfusion-centerpoint-rtdetr_hellodata/iter_13104.pth \\
      --output work_dirs/mv2dfusion_rtdetr/rtdetr_image_branch.onnx
"""

import argparse
import os
import sys
import math
import numpy as np

import cv2
import mmcv
import torch
import torch.nn as nn
import torch.nn.functional as F

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector

# Import FrozenBatchNorm2d so we can detect and convert it
from projects.mmdet3d_plugin.models.backbones.presnet import FrozenBatchNorm2d


def convert_frozen_bn_to_bn(module):
    """Recursively convert FrozenBatchNorm2d → nn.BatchNorm2d (eval mode).

    FrozenBatchNorm2d uses manual math (x * scale + bias) which ONNX exports
    as separate Mul + Add ops. Standard nn.BatchNorm2d in eval mode exports as
    a proper BatchNormalization ONNX op, which onnxsim can then fuse into the
    preceding Conv layer — resulting in a cleaner graph (Conv with bias, no
    extra Mul/Add).
    """
    module_output = module
    if isinstance(module, FrozenBatchNorm2d):
        n = module.num_features
        bn = nn.BatchNorm2d(n, eps=module.eps)
        bn.weight.data.copy_(module.weight.data)
        bn.bias.data.copy_(module.bias.data)
        bn.running_mean.copy_(module.running_mean)
        bn.running_var.copy_(module.running_var)
        bn.num_batches_tracked.zero_()
        bn.eval()
        # Freeze parameters (no grad)
        for p in bn.parameters():
            p.requires_grad = False
        module_output = bn
    for name, child in module.named_children():
        new_child = convert_frozen_bn_to_bn(child)
        if new_child is not child:
            module_output.add_module(name, new_child)
    return module_output


# =====================================================================
# ONNX-friendly RT-DETR Image Branch Wrapper
# =====================================================================
class RTDETRImageBranchForONNX(nn.Module):
    """ONNX-friendly wrapper for PResNet + HybridEncoder + RTDETRQueryHead.

    Inputs:
        img:             [B, N_cam, 3, H, W]  (B=1)
        intrinsics:      [B, N_cam, 4, 4]
        extrinsics:      [B, N_cam, 4, 4]
        extrinsics_inv:  [B, N_cam, 4, 4]

    Outputs:
        feat_flatten_img:    [N_cam, L_img, C]
        spatial_flatten_img: [num_levels, 2]
        lidar2img:           [1, N_cam, 4, 4]
        dyn_query:           [1, global_topk, prob_bin, 4]
        query_feats:         [1, global_topk, C]
    """

    def __init__(self, model, topk_per_cam=100, global_topk=200,
                 img_feat_start_level=0, num_img_feat_levels=2):
        super().__init__()

        # ---- Backbone + Encoder ----
        self.backbone = model.img_backbone
        self.hybrid_encoder = model.hybrid_encoder

        # ---- RTDETRQueryHead components ----
        qh = model.rtdetr_query_head
        self.input_proj = qh.input_proj
        self.enc_output = qh.enc_output
        self.enc_score_head = qh.enc_score_head
        self.enc_bbox_head = qh.enc_bbox_head
        self.depth_head = qh.depth_head
        self.register_buffer('depth_bins', qh.depth_bins.clone())
        self.hidden_dim = qh.hidden_dim
        self.num_classes = qh.num_classes
        self.prob_bin = qh.prob_bin
        self.feat_strides = list(qh.feat_strides)

        # ---- Spatial alignment from fusion head ----
        self.spatial_alignment = model.fusion_bbox_head.spatial_alignment

        # ---- pc_range ----
        self.register_buffer('pc_range', model.fusion_bbox_head.pc_range.clone())

        # ---- Parameters ----
        self.topk_per_cam = topk_per_cam
        self.global_topk = global_topk
        self.img_feat_start_level = img_feat_start_level
        self.num_img_feat_levels = num_img_feat_levels

    # =================================================================
    # Anchor generation (from RTDETRQueryHead)
    # =================================================================
    def _generate_anchors(self, spatial_shapes, device, grid_size=0.05, eps=1e-2):
        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(
                torch.arange(h, dtype=torch.float32, device=device),
                torch.arange(w, dtype=torch.float32, device=device),
                indexing='ij')
            grid_xy = torch.stack([grid_x, grid_y], dim=-1)
            valid_WH = torch.tensor([w, h], dtype=torch.float32, device=device)
            grid_xy = (grid_xy + 0.5) / valid_WH
            wh = torch.ones_like(grid_xy) * grid_size * (2.0 ** lvl)
            anchor = torch.cat([grid_xy, wh], dim=-1).reshape(-1, 4)
            anchors.append(anchor)
        anchors = torch.cat(anchors, dim=0).unsqueeze(0)
        valid_mask = ((anchors > eps) & (anchors < 1 - eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        anchors = torch.where(valid_mask, anchors, torch.tensor(float('inf'), device=device))
        return anchors, valid_mask

    # =================================================================
    # center2lidar (analytical, no torch.inverse)
    # =================================================================
    @staticmethod
    def _intrinsic_inverse(K):
        """Analytical inverse of 4x4 intrinsic matrix."""
        N = K.shape[0]
        K_inv = torch.zeros_like(K)
        fx = K[:, 0, 0]
        fy = K[:, 1, 1]
        cx = K[:, 0, 2]
        cy = K[:, 1, 2]
        s = K[:, 0, 1]
        K_inv[:, 0, 0] = 1.0 / fx.clamp(min=1e-6)
        K_inv[:, 1, 1] = 1.0 / fy.clamp(min=1e-6)
        K_inv[:, 0, 1] = -s / (fx * fy).clamp(min=1e-6)
        K_inv[:, 0, 2] = (s * cy - cx * fy) / (fx * fy).clamp(min=1e-6)
        K_inv[:, 1, 2] = -cy / fy.clamp(min=1e-6)
        K_inv[:, 2, 2] = 1.0
        K_inv[:, 3, 3] = 1.0
        return K_inv

    # def _center2lidar(self, center_2d, depths, intrinsics, extrinsics_inv):
    #     """Project 2D centers + depths to lidar coordinates.

    #     Args:
    #         center_2d:  [N, 2] pixel (u, v)
    #         depths:     [N, D] depth values per bin
    #         intrinsics: [N, 4, 4]
    #         extrinsics_inv: [N, 4, 4]

    #     Returns:
    #         pts_lidar: [N, D, 3]
    #     """
    #     N, D = depths.shape
    #     K_inv = self._intrinsic_inverse(intrinsics)
    #     # img2lidar = extrinsics_inv @ K_inv
    #     img2lidar = torch.bmm(extrinsics_inv, K_inv)  # [N, 4, 4]

    #     u = center_2d[:, 0:1].unsqueeze(1).expand(-1, D, -1)  # [N, D, 1]
    #     v = center_2d[:, 1:2].unsqueeze(1).expand(-1, D, -1)  # [N, D, 1]
    #     d = depths.unsqueeze(-1)  # [N, D, 1]

    #     pts_hom = torch.cat([u * d, v * d, d, torch.ones_like(d)], dim=-1)  # [N, D, 4]
    #     pts_lidar = torch.matmul(pts_hom, img2lidar.transpose(-1, -2))  # [N, D, 4]
    #     return pts_lidar[..., :3]

    def _center2lidar(self, center_2d, depths, intrinsics, extrinsics_inv):
        N, D = depths.shape

        # 提取内参
        fx = intrinsics[:, 0, 0].clamp(min=1e-6)  # [N]
        fy = intrinsics[:, 1, 1].clamp(min=1e-6)
        cx = intrinsics[:, 0, 2]
        cy = intrinsics[:, 1, 2]

        u = center_2d[:, 0:1].unsqueeze(1).expand(-1, D, -1)  # [N,D,1]
        v = center_2d[:, 1:2].unsqueeze(1).expand(-1, D, -1)
        d = depths.unsqueeze(-1)  # [N,D,1]

        # Step 1: 逐元素运算 (全 FP32, 不经过 TF32)
        # 先除以 fx/fy 将值域缩小, 再乘以 d
        X_c = (u - cx[:, None, None]) / fx[:, None, None] * d  # max ≈ 30
        Y_c = (v - cy[:, None, None]) / fy[:, None, None] * d  # max ≈ 17
        Z_c = d                                                  # max = 90

        pts_cam = torch.cat([X_c, Y_c, Z_c, torch.ones_like(d)], dim=-1)  # [N,D,4], max=90

        # Step 2: camera → lidar (小值 MatMul, TF32 安全)
        pts_lidar = torch.matmul(pts_cam, extrinsics_inv.transpose(-1, -2))  # [N,D,4]
        return pts_lidar[..., :3]

    # =================================================================
    # Forward
    # =================================================================
    def forward(self, img, intrinsics, extrinsics, extrinsics_inv):
        img = img.squeeze(0)                    # [N_cam, 3, H, W]
        intrinsics = intrinsics.squeeze(0)      # [N_cam, 4, 4]
        extrinsics = extrinsics.squeeze(0)      # [N_cam, 4, 4]
        extrinsics_inv = extrinsics_inv.squeeze(0)  # [N_cam, 4, 4]

        N_cam = img.shape[0]
        K = self.topk_per_cam
        device = img.device

        # ===== 1. Backbone =====
        backbone_feats = self.backbone(img)
        if isinstance(backbone_feats, dict):
            backbone_feats = list(backbone_feats.values())

        # ===== 2. HybridEncoder =====
        encoder_feats = self.hybrid_encoder(backbone_feats)
        # encoder_feats: list of [N_cam, 256, H_i, W_i], 3 levels

        # ===== 3. RTDETRQueryHead: input projection + flatten =====
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(encoder_feats)]

        spatial_shapes_enc = []
        feat_list_enc = []
        for feat in proj_feats:
            _, _, h, w = feat.shape
            spatial_shapes_enc.append((h, w))
            feat_list_enc.append(feat.flatten(2).permute(0, 2, 1))
        memory = torch.cat(feat_list_enc, dim=1)  # [N_cam, L, C]

        # ===== 4. Encoder output + score/bbox =====
        anchors, valid_mask = self._generate_anchors(spatial_shapes_enc, device)
        valid_mask_float = valid_mask.to(memory.dtype)
        output_memory = self.enc_output['norm'](
            self.enc_output['proj'](memory * valid_mask_float))

        enc_cls = self.enc_score_head(output_memory)      # [N_cam, L, num_classes]
        enc_bbox = self.enc_bbox_head(output_memory) + anchors  # [N_cam, L, 4]

        # ===== 5. TopK per view =====
        topk_scores, topk_ind = torch.topk(
            enc_cls.max(-1).values, K, dim=1)             # [N_cam, K]

        topk_ind_expand = topk_ind.unsqueeze(-1)
        query_feats = output_memory.gather(
            dim=1, index=topk_ind_expand.expand(-1, -1, self.hidden_dim))  # [N_cam, K, C]
        ref_boxes = F.sigmoid(enc_bbox.gather(
            dim=1, index=topk_ind_expand.expand(-1, -1, 4)))  # [N_cam, K, 4]

        # ===== 6. Depth estimation =====
        # intrinsics feature
        intr_per_q = intrinsics[:, None].expand(-1, K, -1, -1).reshape(
            N_cam * K, 4, 4)
        ext_inv_per_q = extrinsics_inv[:, None].expand(-1, K, -1, -1).reshape(
            N_cam * K, 4, 4)

        intr_feat = intr_per_q.reshape(N_cam * K, 16) * 0.01
        depth_input = torch.cat([
            query_feats.detach().reshape(N_cam * K, -1), intr_feat], dim=-1)
        depth_logits = self.depth_head(depth_input)       # [N_cam*K, prob_bin]
        depth_logits = torch.nan_to_num(depth_logits, nan=0.0, posinf=1e4, neginf=-1e4)
        depth_prob = F.softmax(depth_logits.float(), dim=-1)  # [N_cam*K, prob_bin]

        # ===== 7. Project depth bins to lidar =====
        # Get pixel coordinates from normalized ref_boxes
        # pad_shape: (576, 1024) after resize + pad to divisor 32
        H_pad, W_pad = 576.0, 1024.0
        ref_cx = ref_boxes[..., 0]  # [N_cam, K]
        ref_cy = ref_boxes[..., 1]  # [N_cam, K]
        center_2d = torch.stack([
            ref_cx.reshape(-1) * W_pad,
            ref_cy.reshape(-1) * H_pad
        ], dim=-1)  # [N_cam*K, 2]

        depth_bin_vals = self.depth_bins.unsqueeze(0).expand(N_cam * K, -1)
        center_sample = self._center2lidar(
            center_2d, depth_bin_vals, intr_per_q, ext_inv_per_q)  # [N_cam*K, prob_bin, 3]

        # ===== 8. Build dyn_query =====
        dyn_query = torch.cat([
            center_sample, depth_prob.unsqueeze(-1)], dim=-1)  # [N_cam*K, prob_bin, 4]

        query_feats_flat = query_feats.detach().reshape(N_cam * K, -1)  # [N_cam*K, C]

        # ===== 9. Global TopK =====
        all_scores = topk_scores.reshape(-1)              # [N_cam*K]
        _, global_indices = all_scores.topk(
            self.global_topk, dim=0, largest=True, sorted=True)
        dyn_query = dyn_query[global_indices]              # [global_topk, prob_bin, 4]
        query_feats_out = query_feats_flat[global_indices] # [global_topk, C]

        # ===== 10. Spatial alignment on encoder features =====
        intrinsics_scaled = intrinsics / 1e3
        ext_3x4 = extrinsics[..., :3, :]
        mln_input = torch.cat([
            intrinsics_scaled[..., 0, 0:1],
            intrinsics_scaled[..., 1, 1:2],
            ext_3x4.flatten(-2),
        ], dim=-1).unsqueeze(1)                            # [N_cam, 1, 14]

        start_lvl = self.img_feat_start_level
        end_lvl = start_lvl + self.num_img_feat_levels
        feat_list = []
        spatial_shapes = []
        for i in range(start_lvl, end_lvl):
            feat = encoder_feats[i]
            N_c, C_f, H_f, W_f = feat.shape
            feat_list.append(feat.reshape(N_c, C_f, -1).transpose(1, 2))
            spatial_shapes.append([H_f, W_f])
        feat_flatten_img = torch.cat(feat_list, dim=1)
        feat_flatten_img = self.spatial_alignment(feat_flatten_img, mln_input)
        feat_flatten_img = feat_flatten_img.float()

        spatial_flatten_img = torch.tensor(
            spatial_shapes, dtype=torch.long, device=device)

        # ===== 11. lidar2img =====
        lidar2img = torch.bmm(intrinsics, extrinsics).unsqueeze(0)

        return (
            feat_flatten_img,                # [N_cam, L_img, C]
            spatial_flatten_img,             # [num_levels, 2]
            lidar2img,                       # [1, N_cam, 4, 4]
            dyn_query.unsqueeze(0),          # [1, global_topk, prob_bin, 4]
            query_feats_out.unsqueeze(0),    # [1, global_topk, C]
        )


# =====================================================================
# Image preprocessing (aligned with cfg: ResizeCropFlipRotImage + Normalize + Pad)
# =====================================================================
def _parse_img_preprocess_from_cfg(cfg):
    """Read ida_aug_conf / img_norm_cfg / PadMultiViewImage from mmcv Config."""
    ida = getattr(cfg, 'ida_aug_conf', None)
    if ida is None:
        ida = dict(final_dim=(576, 1024), direct_resize=True)
    norm = getattr(cfg, 'img_norm_cfg', None)
    if norm is None:
        norm = dict(mean=[0.0, 0.0, 0.0], std=[255.0, 255.0, 255.0], to_rgb=True)
    size_divisor = 32
    test_pipe = cfg.data.get('test', {})
    if isinstance(test_pipe, dict):
        pipeline = test_pipe.get('pipeline', [])
    else:
        pipeline = []
    for step in pipeline:
        if isinstance(step, dict) and step.get('type') == 'PadMultiViewImage':
            size_divisor = int(step.get('size_divisor', 32))
            break
    return dict(ida_aug_conf=ida, img_norm_cfg=norm, size_divisor=size_divisor)


def _preprocess_image_like_pipeline(img_bgr, pp):
    """BGR uint8/float HWC → float32 CHW, and 3×3 ida_mat (same as ResizeCropFlipRotImage).

    Order matches ``test_pipeline``: geom → NormalizeMultiviewImage → PadMultiViewImage.
    """
    ida_conf = pp['ida_aug_conf']
    fH, fW = ida_conf['final_dim']
    img_h, img_w = img_bgr.shape[:2]
    direct_resize = bool(ida_conf.get('direct_resize', False))

    if direct_resize:
        resize_w = fW / max(int(img_w), 1)
        resize_h = fH / max(int(img_h), 1)
        img = cv2.resize(img_bgr, (int(fW), int(fH)), interpolation=cv2.INTER_LINEAR)
        ida_mat = np.eye(3, dtype=np.float32)
        ida_mat[0, 0] = resize_w
        ida_mat[1, 1] = resize_h
    else:
        base_resize = max(fH / max(img_h, 1), fW / max(img_w, 1))
        resize_dims = (int(img_w * base_resize), int(img_h * base_resize))
        newW, newH = resize_dims
        crop_w = int(max(0, newW - fW) / 2)
        crop_h = int(max(0, newH - fH) / 2)
        img = cv2.resize(img_bgr, resize_dims, interpolation=cv2.INTER_LINEAR)
        l, u, r, b = crop_w, crop_h, crop_w + fW, crop_h + fH
        img = img[u:b, l:r]
        ida_mat = np.eye(3, dtype=np.float32)
        ida_mat[0, 0] = base_resize
        ida_mat[1, 1] = base_resize
        ida_mat[0, 2] = -crop_w
        ida_mat[1, 2] = -crop_h

    norm_cfg = pp['img_norm_cfg']
    img = mmcv.imnormalize(
        img.astype(np.float32),
        np.array(norm_cfg['mean'], dtype=np.float32),
        np.array(norm_cfg['std'], dtype=np.float32),
        to_rgb=norm_cfg.get('to_rgb', True),
    )
    img = mmcv.impad_to_multiple(img, pp['size_divisor'], pad_val=0)
    img_chw = img.transpose(2, 0, 1).astype(np.float32)
    return img_chw, ida_mat


# =====================================================================
# Build model
# =====================================================================
def build_model_from_config(config_path, checkpoint_path):
    cfg = Config.fromfile(config_path)
    if hasattr(cfg, 'plugin') and cfg.plugin:
        import importlib
        if hasattr(cfg, 'plugin_dir'):
            plugin_dirs = cfg.plugin_dir if isinstance(cfg.plugin_dir, list) else [cfg.plugin_dir]
            for plugin_dir in plugin_dirs:
                _module_dir = os.path.dirname(plugin_dir).split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(f'Loading plugin: {_module_path}')
                importlib.import_module(_module_path)

    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    model.eval()
    if os.path.isfile(checkpoint_path):
        print(f'Loading checkpoint: {checkpoint_path}')
        load_checkpoint(model, checkpoint_path, map_location='cpu')
    else:
        print(f'[WARNING] Checkpoint not found: {checkpoint_path}')
    return model, cfg


# =====================================================================
# Main
# =====================================================================
def load_real_inputs_from_pkl(pkl_path, sample_idx=0, device='cpu', infos=None,
                               quiet=False, cfg=None):
    """从 pkl 文件中读取真实数据，构建模型输入。

    几何与归一化对齐 ``cfg.data.test`` 中的
    ``ResizeCropFlipRotImage`` + ``NormalizeMultiviewImage`` + ``PadMultiViewImage``
    （由 ``cfg.ida_aug_conf`` / ``cfg.img_norm_cfg`` 驱动）。

    Args:
        infos: 若已加载 ``data['infos']``，传入可避免每帧重复读 pkl。
        quiet: 为 True 时减少打印（多帧循环用）。
        cfg: mmcv Config；**应传入** ``build_model_from_config`` 返回的 cfg，以与训练/评测一致。
        sample_idx: 支持负数下标（如 -1 表示最后一帧），但注意与 CLI 中
            ``--sample-idx -1`` 表示「全帧导出」的语义由 main 区分。

    Returns:
        inputs: dict of tensors (img, intrinsics, extrinsics, extrinsics_inv)
        token: 该帧的 sample token（用于保存子目录名）
    """
    import pickle

    CAM_ORDER = [
        'camera_backward', 'camera_forward_far', 'camera_forward_wide',
        'camera_pano_leftfront', 'camera_pano_rightfront',
        'camera_pano_leftrear', 'camera_pano_rightrear',
    ]

    if cfg is not None:
        pp = _parse_img_preprocess_from_cfg(cfg)
    else:
        pp = dict(
            ida_aug_conf=dict(final_dim=(576, 1024), direct_resize=True),
            img_norm_cfg=dict(
                mean=[0.0, 0.0, 0.0], std=[255.0, 255.0, 255.0], to_rgb=True),
            size_divisor=32,
        )
        if not quiet:
            print('[WARN] cfg=None: using defaults direct_resize=True, '
                  'final_dim=(576,1024). Pass cfg for exact pipeline match.')

    def build_intrinsic_4x4(cam_K_3x3, ida_mat_3x3):
        K = np.eye(4, dtype=np.float64)
        K[:3, :3] = cam_K_3x3
        ida_4x4 = np.eye(4, dtype=np.float64)
        ida_4x4[:3, :3] = ida_mat_3x3.astype(np.float64)
        return (ida_4x4 @ K).astype(np.float32)

    def build_extrinsic_4x4(s2l_rot, s2l_trans):
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = np.array(s2l_rot)
        T[:3, 3] = np.array(s2l_trans)
        return np.linalg.inv(T).astype(np.float32)

    fH, fW = pp['ida_aug_conf']['final_dim']
    if not quiet:
        dr = pp['ida_aug_conf'].get('direct_resize', False)
        print(f'[preprocess] final_dim=({fH},{fW}), direct_resize={dr}, '
              f'size_divisor={pp["size_divisor"]}')

    if infos is None:
        if not quiet:
            print(f'Loading pkl: {pkl_path}')
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        infos = data['infos']
    n = len(infos)
    si = sample_idx if sample_idx >= 0 else n + sample_idx
    if si < 0 or si >= n:
        raise IndexError(f'sample_idx={sample_idx} -> resolved {si}, num_infos={n}')
    sample = infos[si]
    if not quiet:
        print(f'Sample {sample_idx} -> [{si}]: token={sample["token"]}, '
              f'scene={sample["scene_token"]}')

    imgs_list, intrinsics_list, extrinsics_list, extrinsics_inv_list = [], [], [], []
    for cam_name in CAM_ORDER:
        cam = sample['cams'][cam_name]
        img_path = cam['data_path']
        if not os.path.isfile(img_path):
            if not quiet:
                print(f'  [WARN] {cam_name}: image not found: {img_path}')
            h_pad = int(np.ceil(fH / pp['size_divisor']) * pp['size_divisor'])
            w_pad = int(np.ceil(fW / pp['size_divisor']) * pp['size_divisor'])
            img_np = np.zeros((3, h_pad, w_pad), dtype=np.float32)
            ida_mat = np.eye(3, dtype=np.float32)
        else:
            img_bgr = mmcv.imread(img_path)
            if img_bgr is None:
                if not quiet:
                    print(f'  [WARN] {cam_name}: mmcv.imread failed: {img_path}')
                h_pad = int(np.ceil(fH / pp['size_divisor']) * pp['size_divisor'])
                w_pad = int(np.ceil(fW / pp['size_divisor']) * pp['size_divisor'])
                img_np = np.zeros((3, h_pad, w_pad), dtype=np.float32)
                ida_mat = np.eye(3, dtype=np.float32)
            else:
                img_np, ida_mat = _preprocess_image_like_pipeline(img_bgr, pp)
            if not quiet:
                print(f'  {cam_name}: {img_np.shape}')
        imgs_list.append(img_np)

        cam_K = np.array(cam['cam_intrinsic'], dtype=np.float64)
        intrinsics_list.append(build_intrinsic_4x4(cam_K, ida_mat))

        extrinsic = build_extrinsic_4x4(cam['sensor2lidar_rotation'],
                                         cam['sensor2lidar_translation'])
        extrinsics_list.append(extrinsic)
        extrinsics_inv_list.append(np.linalg.inv(extrinsic).astype(np.float32))

    inputs = dict(
        img=torch.from_numpy(np.stack(imgs_list)[np.newaxis]).float().to(device),
        intrinsics=torch.from_numpy(np.stack(intrinsics_list)[np.newaxis]).float().to(device),
        extrinsics=torch.from_numpy(np.stack(extrinsics_list)[np.newaxis]).float().to(device),
        extrinsics_inv=torch.from_numpy(np.stack(extrinsics_inv_list)[np.newaxis]).float().to(device),
    )
    if not quiet:
        print(f'Real inputs loaded:')
        for k, v in inputs.items():
            print(f'  {k}: {v.shape}')
    return inputs, sample['token']


def _sanitize_token_for_dir(token):
    """将 sample token 转为可用作目录名的字符串。"""
    s = str(token).strip()
    for c in ('/', '\\', '\0', ':', '*', '?', '"', '<', '>', '|'):
        s = s.replace(c, '_')
    return s or 'unknown_token'


def _subdir_name_for_token(token, frame_idx, dup_counts):
    """目录名优先为 token；重复时追加 _1, _2 …"""
    base = _sanitize_token_for_dir(token)
    if base not in dup_counts:
        dup_counts[base] = 0
        return base
    dup_counts[base] += 1
    return f'{base}_{dup_counts[base]}'


def save_tensors_as_bin(tensors_dict, save_dir):
    """将 tensor dict 中的每个 tensor 保存为 .bin 文件 (raw float32/bool)。"""
    os.makedirs(save_dir, exist_ok=True)
    shapes_lines = []
    for name, tensor in tensors_dict.items():
        arr = tensor.detach().cpu().numpy()
        if arr.dtype == np.bool_:
            # bool 转 int32 方便 C++ 读取
            arr = arr.astype(np.int32)
        else:
            arr = arr.astype(np.float32)
        path = os.path.join(save_dir, f'{name}.bin')
        arr.tofile(path)
        size_kb = os.path.getsize(path) / 1024
        shapes_lines.append(f'{name}: shape={list(arr.shape)}, dtype={arr.dtype}')
        print(f'  Saved: {path} ({size_kb:.1f} KB) shape={list(arr.shape)}')

    # 保存 shapes 描述文件
    info_path = os.path.join(save_dir, 'shapes.txt')
    with open(info_path, 'w') as f:
        f.write('\n'.join(shapes_lines) + '\n')
    print(f'  Saved: {info_path}')


def run_all_frames_save_io(wrapper, pkl_path, save_io_dir, device='cpu', cfg=None):
    """
    对 pkl 中的所有帧逐帧运行 image branch wrapper，保存每帧的输入/输出。

    图像分支无时序状态，每帧独立推理即可。

    Args:
        wrapper: RTDETRImageBranchForONNX 实例
        pkl_path: pkl 文件路径
        save_io_dir: 输出根目录，每帧保存到 <token>/inputs/ 和 <token>/outputs/
        device: 运行设备
        cfg: mmcv Config，传给 ``load_real_inputs_from_pkl`` 以与 ``test_pipeline`` 一致
    """
    import json
    import pickle

    print(f'\n{"="*60}')
    print(f'[all-frames IO] pkl: {pkl_path}')
    print(f'[all-frames IO] output dir: {save_io_dir}')
    print(f'{"="*60}')

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    infos = data['infos']
    num_frames = len(infos)
    print(f'[all-frames IO] Total frames: {num_frames}')

    output_names = ['feat_flatten_img', 'spatial_flatten_img', 'lidar2img',
                    'dyn_query', 'query_feats']

    wrapper = wrapper.to(device).eval()
    os.makedirs(save_io_dir, exist_ok=True)
    dup_counts = {}
    manifest = []

    for frame_idx in range(num_frames):
        token = infos[frame_idx].get('token', f'frame_{frame_idx}')
        sub_name = _subdir_name_for_token(token, frame_idx, dup_counts)
        # 1. Load real inputs for this frame (infos 预加载，避免每帧重复读 pkl)
        inputs, _ = load_real_inputs_from_pkl(
            pkl_path, frame_idx, device, infos=infos, quiet=True, cfg=cfg)

        # 2. Forward
        with torch.no_grad():
            outputs = wrapper(**inputs)

        # 3. Save inputs/outputs under token-named folder
        frame_dir = os.path.join(save_io_dir, sub_name)
        save_tensors_as_bin(inputs, os.path.join(frame_dir, 'inputs'))
        output_dict = {name: out for name, out in zip(output_names, outputs)}
        save_tensors_as_bin(output_dict, os.path.join(frame_dir, 'outputs'))

        manifest.append({
            'frame': frame_idx,
            'subdir': sub_name,
            'token': token,
            'scene_token': infos[frame_idx].get('scene_token'),
        })

        if frame_idx % 10 == 0 or frame_idx == num_frames - 1:
            print(f'  Frame {frame_idx + 1}/{num_frames} ({sub_name}): done')

    man_path = os.path.join(save_io_dir, 'manifest.json')
    with open(man_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f'\n[all-frames IO] All {num_frames} frames saved to {save_io_dir}')
    print(f'  Per-frame: {save_io_dir}/<token>/inputs|outputs/')
    print(f'  manifest: {man_path}')


def main():
    parser = argparse.ArgumentParser(description='Export RT-DETR image branch to ONNX')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output', type=str, default='rtdetr_image_branch.onnx')
    parser.add_argument('--opset', type=int, default=16)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--topk-per-cam', type=int, default=100)
    parser.add_argument('--global-topk', type=int, default=200)
    parser.add_argument('--num-cams', type=int, default=7)
    parser.add_argument('--img-h', type=int, default=576)
    parser.add_argument('--img-w', type=int, default=1024)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--ann-file', type=str, default=None,
                        help='pkl 文件路径，使用真实数据替代 dummy inputs')
    parser.add_argument('--sample-idx', type=int, default=0,
                        help='pkl 中的 sample 索引 (单帧; 负数=倒数; 与 --save-io 联用时 -1 表示导出所有帧)')
    parser.add_argument('--save-io', type=str, default=None,
                        help='保存模型输入/输出为 .bin 文件的目录')
    args = parser.parse_args()

    # 1. Build model
    model, cfg = build_model_from_config(args.config, args.checkpoint)

    # 2. Determine FPN levels for spatial_alignment
    head_cfg = cfg.model.get('fusion_bbox_head', {})
    img_feat_start_level = head_cfg.get('img_feat_start_level', 0)
    decoder_layer_cfg = head_cfg.get('transformer', {}).get('decoder', {}).get('transformerlayers', {})
    cross_attn_cfg = decoder_layer_cfg.get('attn_cfgs', [{}])[-1]
    num_img_feat_levels = cross_attn_cfg.get('num_levels', 2)

    # 3. Create wrapper
    wrapper = RTDETRImageBranchForONNX(
        model,
        topk_per_cam=args.topk_per_cam,
        global_topk=args.global_topk,
        img_feat_start_level=img_feat_start_level,
        num_img_feat_levels=num_img_feat_levels,
    )

    # 3.1 Convert FrozenBatchNorm2d → nn.BatchNorm2d for clean ONNX export
    #     (avoids extra Mul+Add nodes; enables Conv+BN fusion in onnxsim)
    wrapper = convert_frozen_bn_to_bn(wrapper)
    wrapper.eval()

    n_params = sum(p.numel() for p in wrapper.parameters())
    print(f'\nCreated wrapper: {type(wrapper).__name__}')
    print(f'  Parameters: {n_params / 1e6:.2f} M')
    print(f'  topk_per_cam: {args.topk_per_cam}')
    print(f'  global_topk: {args.global_topk}')
    print(f'  prob_bin: {wrapper.prob_bin}')
    print(f'  img_feat_start_level: {img_feat_start_level}')
    print(f'  num_img_feat_levels: {num_img_feat_levels}')

    # ================================================================
    # 全帧 IO 保存模式: --ann-file + --save-io + --sample-idx -1
    # 逐帧运行 image branch，保存每帧输入/输出
    # ================================================================
    if args.ann_file and args.save_io and args.sample_idx == -1:
        run_device = 'cuda' if torch.cuda.is_available() else args.device
        run_all_frames_save_io(wrapper, args.ann_file, args.save_io,
                               device=run_device, cfg=cfg)
        print('\nDone!')
        return

    # ================================================================
    # 原有模式: 单帧验证 / ONNX 导出
    # ================================================================
    # 4. Build inputs (real data or dummy)
    device = args.device
    if args.ann_file:
        # 使用真实数据（预处理与 cfg.data.test 一致）
        dummy, _sample_token = load_real_inputs_from_pkl(
            args.ann_file, args.sample_idx, device, cfg=cfg)
    else:
        # Dummy inputs
        N_cam = args.num_cams
        H, W = args.img_h, args.img_w
        dummy = dict(
            img=torch.randn(1, N_cam, 3, H, W, device=device),
            intrinsics=torch.eye(4, device=device).reshape(1, 1, 4, 4).expand(1, N_cam, -1, -1).clone(),
            extrinsics=torch.eye(4, device=device).reshape(1, 1, 4, 4).expand(1, N_cam, -1, -1).clone(),
            extrinsics_inv=torch.eye(4, device=device).reshape(1, 1, 4, 4).expand(1, N_cam, -1, -1).clone(),
        )
        for i in range(N_cam):
            dummy['intrinsics'][0, i, 0, 0] = 800.0 + i * 10
            dummy['intrinsics'][0, i, 1, 1] = 800.0 + i * 10
            dummy['intrinsics'][0, i, 0, 2] = W / 2.0
            dummy['intrinsics'][0, i, 1, 2] = H / 2.0

    print(f'\nInputs:')
    for k, v in dummy.items():
        print(f'  {k}: {v.shape}')

    # 5. Verify forward
    print('\nVerifying forward pass...')
    wrapper = wrapper.to(device)
    inputs = {k: v.to(device) for k, v in dummy.items()}
    with torch.no_grad():
        outputs = wrapper(**inputs)
    output_names = ['feat_flatten_img', 'spatial_flatten_img', 'lidar2img', 'dyn_query', 'query_feats']
    for name, out in zip(output_names, outputs):
        print(f'  {name}: {out.shape} ({out.dtype})')
    print('Forward pass OK!')

    # 5.1. Save inputs/outputs as .bin
    if args.save_io:
        if args.ann_file:
            # 单帧真实数据: 保存到 <token>/ 子目录
            sub_name = _sanitize_token_for_dir(_sample_token)
            frame_dir = os.path.join(args.save_io, sub_name)
            print(f'\nSaving inputs to {frame_dir}/inputs/ (token={_sample_token})')
            save_tensors_as_bin(inputs, os.path.join(frame_dir, 'inputs'))
            print(f'\nSaving outputs to {frame_dir}/outputs/:')
            output_dict = {name: out for name, out in zip(output_names, outputs)}
            save_tensors_as_bin(output_dict, os.path.join(frame_dir, 'outputs'))
        else:
            print(f'\nSaving inputs to {args.save_io}/inputs/:')
            save_tensors_as_bin(inputs, os.path.join(args.save_io, 'inputs'))
            print(f'\nSaving outputs to {args.save_io}/outputs/:')
            output_dict = {name: out for name, out in zip(output_names, outputs)}
            save_tensors_as_bin(output_dict, os.path.join(args.save_io, 'outputs'))

    if args.verify_only:
        return

    # 6. Export ONNX
    print(f'\nExporting to ONNX: {args.output}')
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    input_names = list(dummy.keys())
    onnx_output_names = output_names
    input_tuple = tuple(inputs[k] for k in input_names)

    with torch.no_grad():
        torch.onnx.export(
            wrapper, input_tuple, args.output,
            input_names=input_names,
            output_names=onnx_output_names,
            opset_version=args.opset,
            do_constant_folding=True,
        )
    size_mb = os.path.getsize(args.output) / 1024 / 1024
    print(f'ONNX saved: {args.output} ({size_mb:.1f} MB)')

    # 7. Simplify
    try:
        import onnx
        from onnxsim import simplify as onnxsim_simplify
        print('Running onnx-simplifier...')
        model_onnx = onnx.load(args.output)
        model_sim, check = onnxsim_simplify(model_onnx)
        if check:
            sim_path = args.output.replace('.onnx', '_sim.onnx')
            onnx.save(model_sim, sim_path)
            print(f'Simplified: {sim_path} ({os.path.getsize(sim_path)/1024/1024:.1f} MB)')
    except ImportError:
        print('[INFO] onnxsim not installed, skipping.')
    except Exception as e:
        print(f'[WARNING] onnxsim failed: {e}')

    print('\nDone!')


if __name__ == '__main__':
    main()

