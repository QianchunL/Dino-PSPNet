#!/usr/bin/env python
"""
Export MV2DFusionHead to ONNX format.

该脚本只导出模型的 fusion_bbox_head (MV2DFusionHead) 部分为 ONNX。
导出的 ONNX 模型包含：
  - 图像特征空间对齐 (spatial_alignment)
  - 点云特征嵌入 (pts_embed / pts_query_embed)
  - 动态 query 编码 (dyn_q_embed / dyn_q_enc / dyn_q_pos / ...)
  - Transformer Decoder (self-attention + cross-attention)
  - 分类 / 回归预测头 (cls_branches / reg_branches)

  - 时序 memory bank 作为输入/输出 (外部维护 pre/post_update_memory 逻辑)
  - 输出 outs_dec_last 用于外部更新 memory bank

不包含：
  - 图像 backbone / FPN / FCOS (特征提取在 ONNX 外部完成)
  - 点云 backbone / CenterPoint (特征提取在 ONNX 外部完成)
  - pre_update_memory / post_update_memory (外部维护)
  - Denoising (仅训练时使用)
  - NMS 后处理 (ONNX 外部完成)

Usage:
    python tools/onnx/export_fusion_head_onnx.py \
        --config /mnt/volumes/ad-perception-al-sh01/liqianchun338/bevperception-dev-diffdata-lcfusion-0310-gs/projects/configs/nusc/mv2dfusion-centerpoint-rtdetr_hellodata_0320_02voxel_128dim_debug.py \
        --checkpoint /mnt/volumes/ad-perception-al-sh01/lhm/onemodel/bevperception/work_dirs/mv2dfusion-centerpoint-rtdetr_combined_0305_lc_onemodel_0312/iter_195200.pth \
        --output ./onnx_models/fusion_head.onnx \
        --no-nms
"""

import argparse
import copy
import json
import math
import os
import sys

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


# =====================================================================
# Custom ONNX op for MultiScaleDeformableAttention
# (exports as mmdeploy::MMCVMultiScaleDeformableAttention)
#
# Reference: export_onnx_plugin.py
# =====================================================================
import onnx as _onnx_module


class _MSDAForONNX(torch.autograd.Function):
    """MultiScaleDeformableAttnFunction wrapper that exports to
    mmdeploy::MMCVMultiScaleDeformableAttention custom ONNX op.

    forward() uses the original CUDA kernel (or pure-PyTorch fallback)
    so that verify_forward still gives correct numerics.
    symbolic() emits the custom op node for TensorRT deployment.
    """

    @staticmethod
    def forward(ctx, value, spatial_shapes, level_start_index,
                sampling_locations, attention_weights, im2col_step=64):
        # Pure PyTorch grid_sample implementation for forward pass.
        # value: [B, HW, G, D] — plugin native layout.
        B, num_value, num_groups, group_dims = value.shape
        _, n_q, _, num_levels, num_pts, _ = sampling_locations.shape
        sampling_grids = 2.0 * sampling_locations - 1.0
        attention_weights_r = attention_weights.reshape(
            B, n_q, num_groups, num_levels, num_pts)
        output = torch.zeros(
            B, n_q, num_groups, group_dims,
            device=value.device, dtype=value.dtype)
        for lvl in range(num_levels):
            H = int(spatial_shapes[lvl, 0].item())
            W = int(spatial_shapes[lvl, 1].item())
            start = int(level_start_index[lvl].item())
            val_l = value[:, start:start + H * W].permute(0, 2, 3, 1).reshape(
                B * num_groups, group_dims, H, W)
            grid_l = sampling_grids[:, :, :, lvl].permute(0, 2, 1, 3, 4).reshape(
                B * num_groups, n_q, num_pts, 2)
            sampled = F.grid_sample(
                val_l, grid_l, mode='bilinear', padding_mode='zeros',
                align_corners=False).reshape(B, num_groups, group_dims, n_q, num_pts)
            w = attention_weights_r[:, :, :, lvl].permute(0, 2, 1, 3).unsqueeze(2)
            output = output + (sampled * w).sum(-1).permute(0, 3, 1, 2)
        return output.reshape(B, n_q, num_groups * group_dims)

    @staticmethod
    def symbolic(g, value, spatial_shapes, level_start_index,
                 sampling_locations, attention_weights, im2col_step=64):
        """Emit mmdeploy::MMCVMultiScaleDeformableAttention ONNX node."""
        spatial_shapes_i32 = g.op(
            "Cast", spatial_shapes,
            to_i=_onnx_module.TensorProto.INT32)
        level_start_index_i32 = g.op(
            "Cast", level_start_index,
            to_i=_onnx_module.TensorProto.INT32)
        return g.op(
            "mmdeploy::MMCVMultiScaleDeformableAttention",
            value,
            spatial_shapes_i32,
            level_start_index_i32,
            sampling_locations,
            attention_weights,
            im2col_step_i=im2col_step,
        )


class _MSDAProxy:
    """Drop-in replacement for MultiScaleDeformableAttnFunction.
    Routes .apply() through _MSDAForONNX so that ONNX export emits
    the custom op while forward still computes correct results."""
    @staticmethod
    def apply(value, spatial_shapes, level_start_index,
              sampling_locations, attention_weights, im2col_step=64):
        return _MSDAForONNX.apply(
            value, spatial_shapes, level_start_index,
            sampling_locations, attention_weights, im2col_step)


# Monkey-patch BEFORE importing any model code
import mmcv.ops.multi_scale_deform_attn as _msda_mod
_msda_mod.MultiScaleDeformableAttnFunction = _MSDAProxy

# Now import project code (will use the patched version)
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.models.utils.transformer import inverse_sigmoid

from projects.mmdet3d_plugin.models.utils.positional_encoding import (
    pos2posemb3d, pos2posemb1d, nerf_positional_encoding
)


# =====================================================================
# ONNX-friendly decoder forward (avoids boolean mask indexing)
# =====================================================================
def onnx_decoder_forward(decoder, query, query_pos, reference_points,
                         dyn_q_coords, dyn_q_probs, dyn_q_mask,
                         dyn_q_pos_branch, dyn_q_pos_with_prob_branch,
                         dyn_q_prob_branch, **layer_kwargs):
    """
    ONNX-friendly version of MV2DFusionTransformerDecoder.forward.

    After cm/cp0224 merge, all tensors use [B, Q, C] convention (not [Q, B, C]).

    Applies the same feat_flatten_img / feat_flatten_pts down_proj + view as
    MV2DFusionTransformerDecoder.forward before iterating layers (see decoder
    hoist of down_proj to Identity inside MixedCrossAttention).

    Args:
        query:         [B, Q_total, C]
        query_pos:     [B, Q_total, C]
        reference_points: [B, Q_total, 3]
        dyn_q_coords:  [B, Q_total, prob_bin, 3]  full-size, zero-padded
        dyn_q_probs:   [B, Q_total, prob_bin]  full-size, uniform for non-dyn
        dyn_q_mask:    [B, Q_total] bool
    """
    dyn_q_logits = dyn_q_probs.clamp(min=1e-8).log()  # [B, Q, prob_bin]
    dyn_q_mask_float = dyn_q_mask.unsqueeze(-1).float()  # [B, Q, 1]

    # Mirror MV2DFusionTransformerDecoder.forward (mv2dfusion_transformer.py ~365–380):
    # decoder hoists down_proj_img/down_proj_pts to run ONCE before the layer loop, and
    # replaces each MixedCrossAttention.down_proj_* with Identity. Calling layers directly
    # without this leaves feat_flatten_img as [BN,HW,256] while the layer still does
    # view(..., num_groups, group_dims) with group_dims=16 → RuntimeError.
    feat_img = layer_kwargs.get('feat_flatten_img')
    if feat_img is not None and feat_img.dim() == 3 and hasattr(
            decoder, 'down_proj_img') and hasattr(decoder, 'num_groups_img'):
        x = decoder.down_proj_img(feat_img)
        bn_cams, num_value_img, _ = x.shape
        layer_kwargs['feat_flatten_img'] = x.view(
            bn_cams, num_value_img,
            decoder.num_groups_img, decoder.group_dims_img).contiguous()
    feat_pts = layer_kwargs.get('feat_flatten_pts')
    if feat_pts is not None and feat_pts.dim() == 3 and hasattr(
            decoder, 'down_proj_pts') and hasattr(decoder, 'num_groups_pts'):
        x = decoder.down_proj_pts(feat_pts)
        bs_pts, num_value_pts, _ = x.shape
        layer_kwargs['feat_flatten_pts'] = x.view(
            bs_pts, num_value_pts,
            decoder.num_groups_pts, decoder.group_dims_pts).contiguous()

    intermediate = []
    intermediate_reference_points = [reference_points]
    intermediate_dyn_q_logits = []

    for i, layer in enumerate(decoder.layers):
        # layer expects [B, Q, C] convention
        query = layer(query, query_pos=query_pos,
                      prev_ref_point=reference_points, **layer_kwargs)
        if decoder.post_norm is not None:
            interm_q = decoder.post_norm(query)
        else:
            interm_q = query

        # Apply prob_branch to ALL queries (dense), then mask
        # query is already [B, Q, C]
        all_logits_res = dyn_q_prob_branch[i](query)  # [B, Q, prob_bin]

        # Update logits only for dyn queries
        dyn_q_logits = dyn_q_logits + all_logits_res * dyn_q_mask_float
        dyn_q_probs_cur = dyn_q_logits.softmax(-1)  # [B, Q, prob_bin]

        # Update reference_points: weighted sum of dyn_q_coords
        dyn_q_ref = torch.matmul(
            dyn_q_probs_cur.unsqueeze(-2), dyn_q_coords).squeeze(-2)
        new_reference_points = torch.where(
            dyn_q_mask.unsqueeze(-1), dyn_q_ref, reference_points)
        reference_points = new_reference_points

        # Update query_pos (all in [B, Q, C])
        dyn_q_pos = dyn_q_pos_branch(dyn_q_coords.flatten(-2, -1))  # [B,Q,C]
        dyn_q_pos = dyn_q_pos_with_prob_branch(dyn_q_pos, dyn_q_probs_cur)
        query_pos = torch.where(
            dyn_q_mask.unsqueeze(-1), dyn_q_pos, query_pos)

        intermediate.append(interm_q)
        intermediate_reference_points.append(new_reference_points)
        intermediate_dyn_q_logits.append(dyn_q_logits)

    return (torch.stack(intermediate),
            torch.stack(intermediate_reference_points),
            torch.stack(intermediate_dyn_q_logits))


# =====================================================================
# ONNX Wrapper
# =====================================================================
class MV2DFusionHeadForONNX(nn.Module):
    """
    ONNX-friendly wrapper for MV2DFusionHead.

    将 MV2DFusionHead 的推理逻辑封装为纯 Tensor 输入/输出的 Module，
    适用于 torch.onnx.export。

    输入 (18 个):
      ---- 来自 2D ONNX ----
      - feat_flatten_img: [N, L_img, C]   — 已 spatial_alignment
      - spatial_flatten_img: [4, 2]       — FPN levels 1-4 空间形状 (H, W)
      - lidar2img: [B, N, 4, 4]
      - dyn_query: [B, N_img_q, prob_bin, 4]
      - dyn_query_feats: [B, N_img_q, C]
      ---- 来自 3D ONNX ----
      - pts_feat: [B, L_pts, C_pts]
      - pts_query_center: [B, N_pts_q, 3]
      - pts_query_feat: [B, N_pts_q, C_pts]
      ---- 上一帧输出的 memory bank (首帧传入全零) ----
      - memory_embedding: [B, memory_len, C]
      - memory_reference_point: [B, memory_len, 3]
      - memory_timestamp: [B, memory_len, 1]
      - memory_velo: [B, memory_len, 2]
      - memory_egopose: [B, memory_len, 4, 4]
      - memory_query_mask: [B, memory_len, 1]  (bool)
      ---- 当前帧信息 ----
      - ego_pose: [B, 4, 4]       当前帧 ego pose (ego→world)
      - ego_pose_inv: [B, 4, 4]   当前帧 ego pose 的逆 (world→ego)
      - timestamp: [B]             当前帧时间戳
      - prev_exists: [B]           是否有前一帧 (float 0/1)
      - prev_timestamp: [B]       上一帧时间戳 (用于 velocity propagation)

    输出 (9 个):
      ---- 检测结果 ----
      - all_cls_scores: [num_layers, B, Q_total, num_classes]
      - all_bbox_preds: [num_layers, B, Q_total, code_size]
      - all_is_dynamic_preds: [num_layers, B, Q_total, 1]  动静态预测
      ---- 更新后的 memory bank (直接作为下一帧 memory_* 输入) ----
      - out_memory_embedding: [B, memory_len, C]
      - out_memory_reference_point: [B, memory_len, 3]
      - out_memory_timestamp: [B, memory_len, 1]
      - out_memory_velo: [B, memory_len, 2]
      - out_memory_egopose: [B, memory_len, 4, 4]
      - out_memory_query_mask: [B, memory_len, 1]

    自包含的时序循环:
      帧N 输出的 memory → 帧N+1 直接作为 memory 输入 → ...
      首帧只需传入全零 memory + prev_exists=0

    包含:
    - pre_update_memory: ego_pose_inv 变换 → memory_refresh → 首帧 propagated 初始化
    - Transformer Decoder (self-attention + cross-attention)
    - 分类 / 回归预测 (cls_branches / reg_branches)
    - post_update_memory: topk 选择 → concat → ego_pose 变换 → 截断

    简化项：
    1. 无 Denoising (仅训练时使用)
    2. 无 NMS 后处理 (ONNX 外部完成)
    3. img_metas 替换为 pad_shape_hw tensor (buffer)
    4. dyn_query (变长 list) 替换为固定大小 padded tensor + mask
    5. 避免 boolean mask indexing → 使用 torch.where 密集操作
    6. 避免 repeat_interleave → 使用 unsqueeze + expand + reshape
    7. MultiScaleDeformableAttnFunction → mmdeploy::MMCVMultiScaleDeformableAttention 自定义 ONNX op
    8. spatial_alignment 已移至 2D ONNX, fusion ONNX 直接接收处理后的特征
    9. memory bank 自包含: pre_update_memory + post_update_memory 均内置

    外部只需:
    - 首帧: 传入全零 memory + prev_exists=0
    - 后续帧: 将上一帧的 out_memory_* 直接作为本帧的 memory_* 输入
    """

    def __init__(self, head, pad_shape_hw=None, fpn_spatial_shapes=None,
                 no_nms=False):
        """
        Args:
            head: MV2DFusionHead 实例
            pad_shape_hw: (H_pad, W_pad) e.g. (576, 1024)
            fpn_spatial_shapes: list of (H, W) per FPN level, e.g. [(72,128),(36,64)]
            no_nms: 如果 True, 跳过 NMS 和 post_update_memory,
                    输出 outs_dec_last + tgt_query_mask + pre_memory_*
        """
        super().__init__()
        self.no_nms = no_nms
        # ===== 从原始 head 复制所有子模块和参数 =====
        self.embed_dims = head.embed_dims
        self.num_pred = head.num_pred
        self.code_size = head.code_size
        self.num_propagated = head.num_propagated
        self.memory_len = head.memory_len
        self.with_ego_pos = head.with_ego_pos
        self.num_classes = head.num_classes
        self.cls_out_channels = head.cls_out_channels
        self.prob_bin = head.prob_bin
        self.topk_proposals = head.topk_proposals

        # 常量参数
        self.pc_range = head.pc_range  # nn.Parameter, [6]

        # BEV NMS 参数 (与 MV2DFusionHead.forward 对齐)
        self.post_bev_nms_thr = getattr(head, 'post_bev_nms_thr', 0.2)
        self.post_bev_nms_score = getattr(head, 'post_bev_nms_score', 0.0)
        self.post_bev_nms_ops = getattr(head, 'post_bev_nms_ops', [])
        self.propagated_score_bonus = getattr(head, 'propagated_score_bonus', 0.0)
        self.num_query_total = getattr(head, 'num_query', 200)  # pts+img queries

        # Transformer decoder
        self.transformer = head.transformer

        # 分类 / 回归 / 动静态分支
        self.cls_branches = head.cls_branches
        self.reg_branches = head.reg_branches
        self.is_dynamic_branches = head.is_dynamic_branches

        # Query 相关
        self.reference_points = head.reference_points
        self.query_embedding = head.query_embedding
        self.time_embedding = head.time_embedding

        # Dynamic query 相关
        self.dyn_q_embed = head.dyn_q_embed
        self.dyn_q_enc = head.dyn_q_enc
        self.dyn_q_pos = head.dyn_q_pos
        self.dyn_q_pos_with_prob = head.dyn_q_pos_with_prob
        self.dyn_q_prob_branch = head.dyn_q_prob_branch

        # Point cloud 相关
        self.pts_embed = head.pts_embed
        self.pts_query_embed = head.pts_query_embed
        self.pts_q_embed = head.pts_q_embed

        # Ego pose 相关
        if self.with_ego_pos:
            self.ego_pose_pe = head.ego_pose_pe
            self.ego_pose_memory = head.ego_pose_memory

        # Pseudo reference points (for propagated queries)
        if hasattr(head, 'pseudo_reference_points') and head.num_propagated > 0:
            self.pseudo_reference_points = head.pseudo_reference_points
            # 预计算世界坐标系下的 pseudo reference points (用于 pre_update_memory)
            with torch.no_grad():
                pseudo_ref_world = head.pseudo_reference_points.weight.data * (
                    head.pc_range.data[3:6] - head.pc_range.data[0:3]
                ) + head.pc_range.data[0:3]
            self.register_buffer('pseudo_ref_world', pseudo_ref_world)  # [num_propagated, 3]

        # ===== 注册常量 buffer =====
        # 图像 pad 尺寸 (用于 img_metas_proxy)
        if pad_shape_hw is not None:
            self.register_buffer(
                'pad_shape_hw',
                torch.tensor(pad_shape_hw, dtype=torch.float32))

        # FPN 空间形状 (静态常量，不作为 ONNX 输入)
        if fpn_spatial_shapes is not None:
            spatial_t = torch.tensor(fpn_spatial_shapes, dtype=torch.long)
            self.register_buffer('spatial_flatten_img', spatial_t)
            # 预计算 level_start_index_img
            level_start = torch.cat([
                spatial_t.new_zeros((1,)),
                spatial_t.prod(1).cumsum(0)[:-1]
            ])
            self.register_buffer('level_start_index_img', level_start)

    def forward(self,
                # ----- 来自 2D ONNX -----
                feat_flatten_img,        # [N, L_img, C]  已 spatial_alignment
                lidar2img,               # [B, N, 4, 4]
                # ----- 来自 3D ONNX -----
                pts_feat,                # [B, L_pts, C_pts]
                pts_query_center,        # [B, N_pts_q, 3]
                pts_query_feat,          # [B, N_pts_q, C_pts]
                # ----- 来自 2D ONNX (query) -----
                dyn_query,               # [B, N_img_q, prob_bin, 4]  (xyz + prob)
                dyn_query_feats,         # [B, N_img_q, C]
                # ----- 上一帧输出的 memory bank (首帧传入全零) -----
                memory_embedding,        # [B, memory_len, C]
                memory_reference_point,  # [B, memory_len, 3]
                memory_timestamp,        # [B, memory_len, 1]
                memory_velo,             # [B, memory_len, 2]
                memory_egopose,          # [B, memory_len, 4, 4]
                memory_query_mask,       # [B, memory_len, 1]  (bool)
                # ----- 当前帧信息 -----
                ego_pose=None,           # [B, 4, 4]  ego→world (no_nms 模式不需要)
                ego_pose_inv=None,       # [B, 4, 4]  world→ego
                timestamp=None,          # [B]
                prev_exists=None,        # [B]  float 0/1
                prev_timestamp=None,     # [B]  上一帧时间戳 (用于 velocity propagation)
                ):
        """
        Forward pass for ONNX export (自包含时序: pre + decoder + post).

        内部自动完成:
          - pre_update_memory: ego_pose_inv 变换 → memory_refresh → propagated 初始化
          - level_start_index_img 从 spatial_flatten_img 计算
          - dyn_query → dyn_query_coords, dyn_query_probs, dyn_reference_points, mask
          - temporal_alignment: ego_pose_pe / ego_pose_memory / time_embedding
          - Transformer Decoder
          - post_update_memory: topk 选择 → concat → ego_pose 变换 → 截断

        Returns:
            all_cls_scores: [num_layers, B, Q_total, num_classes]
            all_bbox_preds: [num_layers, B, Q_total, code_size]
            all_is_dynamic_preds: [num_layers, B, Q_total, 1]
            out_memory_embedding: [B, memory_len, C]
            out_memory_reference_point: [B, memory_len, 3]
            out_memory_timestamp: [B, memory_len, 1]
            out_memory_velo: [B, memory_len, 2]
            out_memory_egopose: [B, memory_len, 4, 4]
            out_memory_query_mask: [B, memory_len, 1]
        """
        B = pts_feat.shape[0]
        device = pts_feat.device

        # ===== 0a. pre_update_memory: 坐标变换 =====
        # 将上一帧 post_update_memory 输出的 memory 变换到当前帧坐标系
        # memory_timestamp += timestamp  (累加时间偏移)
        M = memory_embedding.shape[1]
        memory_timestamp = memory_timestamp + \
            timestamp.unsqueeze(-1).unsqueeze(-1)           # [B, M, 1]

        # memory_egopose = ego_pose_inv @ memory_egopose
        # ONNX-friendly: 显式 expand + reshape + bmm 替代 4D broadcast @
        ego_inv_exp = ego_pose_inv.unsqueeze(1).expand(B, M, 4, 4).reshape(B * M, 4, 4)
        memory_egopose = torch.bmm(
            ego_inv_exp, memory_egopose.reshape(B * M, 4, 4)
        ).reshape(B, M, 4, 4)                               # [B, M, 4, 4]

        # memory_reference_point = ego_pose_inv @ homogeneous(ref)
        ref_homo = torch.cat([
            memory_reference_point,
            torch.ones_like(memory_reference_point[..., :1])
        ], dim=-1)                                          # [B, M, 4]
        # ONNX-friendly: expand + bmm
        memory_reference_point = torch.bmm(
            ego_inv_exp, ref_homo.reshape(B * M, 4, 1)
        ).reshape(B, M, 4)[..., :3]                         # [B, M, 3]

        # ===== 0a-2. Velocity-based temporal propagation =====
        # 与 Python 训练代码 pre_update_memory 一致:
        #   vel_dt = (timestamp - prev_timestamp).clamp(min=0)
        #   rot = memory_egopose[..., :2, :2]
        #   rotated_velo = rot @ memory_velo
        #   memory_reference_point[..., :2] += rotated_velo * vel_dt
        #
        # memory_velo 存储在 ego_origin 坐标系;
        # memory_egopose 已被变换到当前帧坐标系 (经过 ego_pose_inv @ 后)
        # 用 memory_egopose[:2,:2] 旋转速度到当前帧坐标系
        #
        # ONNX-friendly: 用 prev_exists 做 gate (首帧 prev_exists=0 → vel_dt=0)
        # 避免 if 分支, 保证 ONNX 静态图
        vel_dt = (timestamp - prev_timestamp).clamp(min=0)  # [B]
        vel_dt = vel_dt * prev_exists                        # 首帧 → 0
        # rot: [B, M, 2, 2]
        rot = memory_egopose[..., :2, :2]
        # rotated_velo: [B, M, 2]
        rotated_velo = torch.matmul(
            rot, memory_velo.unsqueeze(-1)).squeeze(-1)     # [B, M, 2]
        # vel_offset: [B, M, 2]
        vel_offset = rotated_velo * vel_dt.view(B, 1, 1)
        memory_reference_point = torch.cat([
            memory_reference_point[..., :2] + vel_offset,
            memory_reference_point[..., 2:3]
        ], dim=-1)                                          # [B, M, 3]

        # ===== 0b. memory_refresh: 新场景清零 =====
        # prev_exists: [B], float 0 or 1;  0 = 新场景 → 清零
        px = prev_exists.view(B, 1, 1)                     # [B, 1, 1]
        memory_embedding = memory_embedding * px
        memory_reference_point = memory_reference_point * px
        memory_timestamp = memory_timestamp * px
        memory_velo = memory_velo * px
        memory_egopose = memory_egopose * prev_exists.view(B, 1, 1, 1)
        # bool mask: 清零用 AND
        memory_query_mask = memory_query_mask & (
            prev_exists.view(B, 1, 1) > 0.5)

        # ===== 0c. 首帧: 初始化 propagated slots =====
        if self.num_propagated > 0:
            is_first = (1.0 - prev_exists).view(B, 1, 1)   # [B, 1, 1] 首帧=1
            # reference_point[:, :P] += (1-x) * pseudo_ref_world
            prop_ref = memory_reference_point[:, :self.num_propagated] + \
                is_first * self.pseudo_ref_world              # broadcast [P, 3]
            rest_ref = memory_reference_point[:, self.num_propagated:]
            memory_reference_point = torch.cat([prop_ref, rest_ref], dim=1)

            # egopose[:, :P] += (1-x) * eye(4)
            eye4 = torch.eye(4, device=device, dtype=memory_egopose.dtype)
            prop_ego = memory_egopose[:, :self.num_propagated] + \
                is_first.unsqueeze(-1) * eye4                 # [B, P, 4, 4]
            rest_ego = memory_egopose[:, self.num_propagated:]
            memory_egopose = torch.cat([prop_ego, rest_ego], dim=1)

            # query_mask[:, :P] |= is_first
            is_first_bool = (prev_exists.view(B, 1, 1) < 0.5)  # [B, 1, 1]
            prop_mask = memory_query_mask[:, :self.num_propagated] | is_first_bool
            rest_mask = memory_query_mask[:, self.num_propagated:]
            memory_query_mask = torch.cat([prop_mask, rest_mask], dim=1)

        # ===== 0d. spatial_flatten_img / level_start_index_img 已注册为 buffer =====
        spatial_flatten_img = self.spatial_flatten_img
        level_start_index_img = self.level_start_index_img
        pad_shape_hw = self.pad_shape_hw

        # ===== 0e. 从 dyn_query 计算 5 个变量 =====
        dyn_query_coords_raw = dyn_query[..., :3]       # [B, N_img_q, prob_bin, 3]
        dyn_query_probs = dyn_query[..., 3]              # [B, N_img_q, prob_bin]

        # 归一化坐标到 pc_range [0, 1]
        dyn_query_coords = (dyn_query_coords_raw - self.pc_range[:3]) / (
            self.pc_range[3:6] - self.pc_range[:3])

        # 加权求和得到 reference points
        dyn_reference_points = torch.matmul(
            dyn_query_probs.unsqueeze(-2), dyn_query_coords
        ).squeeze(-2)

        # mask: topk 输出均有效
        num_img_q = dyn_query.shape[1]
        dyn_query_mask = torch.ones(
            B, num_img_q, dtype=torch.bool, device=device)

        # ===== 1. feat_flatten_img 已由 2D ONNX 处理完毕, 直接使用 =====

        # ===== 2. 处理点云特征 =====
        feat_flatten_pts = self.pts_embed(pts_feat)
        # pos_flatten_pts: 在 MixedCrossAttention.forward 中未实际使用
        # 传入 None 即可; 如果 transformer 接口要求, 传零张量
        pos_flatten_pts = torch.zeros(
            B, feat_flatten_pts.shape[1], 2,
            device=device, dtype=feat_flatten_pts.dtype)

        # ===== 3. 准备 query =====
        num_query_pts = pts_query_center.shape[1]
        num_query_img = dyn_reference_points.shape[1]
        num_query = num_query_pts + num_query_img

        # 点云 query 参考点 (归一化到 [0, 1])
        pts_ref = pts_query_center.clone()
        pts_ref[..., 0:3] = (pts_ref[..., 0:3] - self.pc_range[0:3]) / (
                self.pc_range[3:6] - self.pc_range[0:3])

        # 拼接 query mask: [pts (全 True), img (dyn_query_mask)]
        query_mask = torch.cat([
            torch.ones(B, num_query_pts, dtype=torch.bool, device=device),
            dyn_query_mask
        ], dim=1)  # [B, num_query]

        # 拼接参考点: [pts, img]
        reference_points = torch.cat([pts_ref, dyn_reference_points], dim=1)

        # ===== 4. prepare_for_dn (推理模式: 无 denoising) =====
        pad_size = 0

        # ===== 5. 构建 attention mask (避免 repeat_interleave) =====
        tgt_size = num_query + self.num_propagated
        src_size = num_query + self.memory_len

        attn_mask = torch.zeros(
            (B, tgt_size, src_size), dtype=torch.bool, device=device)

        # 使用输入的 memory_query_mask (而非硬编码零)
        mem_mask_2d = memory_query_mask[:, :, 0]  # [B, memory_len]

        if self.num_propagated > 0:
            prop_mask = mem_mask_2d[:, :self.num_propagated]  # [B, num_propagated]
            tgt_query_mask = torch.cat([query_mask, prop_mask], dim=1)
        else:
            tgt_query_mask = query_mask

        src_query_mask = torch.cat([query_mask, mem_mask_2d], dim=1)
        attn_mask[:, :, pad_size:] = ~src_query_mask[:, None]

        # ONNX-friendly repeat for multi-head: unsqueeze + expand + reshape
        num_heads = self.transformer.decoder.layers[0].attentions[0].num_heads
        # [B, tgt, src] → [B, 1, tgt, src] → [B, H, tgt, src] → [B*H, tgt, src]
        attn_mask = attn_mask.unsqueeze(1).expand(
            -1, num_heads, -1, -1).reshape(B * num_heads, tgt_size, src_size)

        # ===== 6. 构建 query content (tgt) =====
        tgt_img = self.dyn_q_embed.weight.repeat(B, num_query_img, 1)
        tgt_pts = self.pts_q_embed.weight.repeat(B, num_query_pts, 1)
        tgt = torch.cat([tgt_pts, tgt_img], dim=1)

        pad_query_feats = torch.zeros(
            B, num_query, self.embed_dims, device=device, dtype=tgt.dtype)
        pts_query_feat_enc = self.pts_query_embed(pts_query_feat)
        pad_query_feats[:, :num_query_pts] = pts_query_feat_enc
        pad_query_feats[:, num_query_pts:] = dyn_query_feats
        tgt = self.dyn_q_enc(tgt, pad_query_feats)

        # ===== 7. 构建 query positional encoding =====
        query_pos = self.query_embedding(pos2posemb3d(reference_points))

        # ===== 8. 时序对齐 (使用输入的 memory bank) =====
        # 归一化 memory_reference_point to [0, 1]
        temp_reference_point = (
            memory_reference_point - self.pc_range[:3]
        ) / (self.pc_range[3:6] - self.pc_range[0:3])
        temp_pos = self.query_embedding(pos2posemb3d(temp_reference_point))
        temp_memory = memory_embedding

        rec_ego_pose = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0)\
            .repeat(B, num_query, 1, 1)

        if self.with_ego_pos:
            rec_ego_motion = torch.cat(
                [torch.zeros_like(reference_points[..., :3]),
                 rec_ego_pose[..., :3, :].flatten(-2)], dim=-1)
            rec_ego_motion = nerf_positional_encoding(rec_ego_motion)
            tgt = self.ego_pose_memory(tgt, rec_ego_motion)
            query_pos = self.ego_pose_pe(query_pos, rec_ego_motion)

            # 使用输入的 memory_velo, memory_timestamp, memory_egopose
            memory_ego_motion = torch.cat(
                [memory_velo, memory_timestamp,
                 memory_egopose[..., :3, :].flatten(-2)], dim=-1).float()
            memory_ego_motion = nerf_positional_encoding(memory_ego_motion)
            temp_pos = self.ego_pose_pe(temp_pos, memory_ego_motion)
            temp_memory = self.ego_pose_memory(temp_memory, memory_ego_motion)

        query_pos = query_pos + self.time_embedding(
            pos2posemb1d(torch.zeros_like(reference_points[..., :1])))
        temp_pos = temp_pos + self.time_embedding(
            pos2posemb1d(memory_timestamp).float())

        # 添加 propagated queries (前 num_propagated 条 memory)
        if self.num_propagated > 0:
            tgt = torch.cat(
                [tgt, temp_memory[:, :self.num_propagated]], dim=1)
            query_pos = torch.cat(
                [query_pos, temp_pos[:, :self.num_propagated]], dim=1)
            reference_points = torch.cat(
                [reference_points,
                 temp_reference_point[:, :self.num_propagated]], dim=1)
            temp_memory = temp_memory[:, self.num_propagated:]
            temp_pos = temp_pos[:, self.num_propagated:]

        # ===== 9. 编码动态 query 的位置分布 =====
        query_pos_det = self.dyn_q_pos(dyn_query_coords.flatten(-2, -1))
        query_pos_det = self.dyn_q_pos_with_prob(
            query_pos_det, dyn_query_probs)
        query_pos[:, num_query_pts:num_query] = query_pos_det

        # ===== 10. 构建 dyn_q 相关的 full-size 张量 (dense, ONNX-friendly) =====
        # dyn_q_mask_full: [B, Q_total]  True only for valid image queries
        dyn_q_mask_full = torch.zeros(
            B, tgt.shape[1], dtype=torch.bool, device=device)
        dyn_q_mask_full[:, num_query_pts:num_query] = True
        dyn_q_mask_full[:, pad_size:] = (
            dyn_q_mask_full[:, pad_size:] & tgt_query_mask)

        # dyn_q_coords_full: [B, Q_total, prob_bin, 3]  zero-padded
        Q_total = tgt.shape[1]
        dyn_q_coords_full = torch.zeros(
            B, Q_total, self.prob_bin, 3, device=device, dtype=tgt.dtype)
        dyn_q_coords_full[:, num_query_pts:num_query] = dyn_query_coords

        # dyn_q_probs_full: [B, Q_total, prob_bin]  uniform for non-dyn
        uniform_val = 1.0 / self.prob_bin
        dyn_q_probs_full = torch.full(
            (B, Q_total, self.prob_bin), uniform_val,
            device=device, dtype=tgt.dtype)
        dyn_q_probs_full[:, num_query_pts:num_query] = dyn_query_probs

        # ===== 11. Transformer Decoder (ONNX-friendly) =====
        img_metas_proxy = [{'pad_shape': [
            (int(pad_shape_hw[0].item()), int(pad_shape_hw[1].item()))]}]

        # NOTE: After cm/cp0224 merge, decoder layer uses [B, Q, C] convention
        # (not the old [Q, B, C]), so we pass tensors directly without transpose.

        # Use ONNX-friendly decoder forward
        outs_dec, reference_points_out, dyn_q_logits = onnx_decoder_forward(
            self.transformer.decoder,
            # decoder-level args (handled by onnx_decoder_forward)
            query=tgt,                  # [B, Q, C]
            query_pos=query_pos,        # [B, Q, C]
            reference_points=reference_points,
            dyn_q_coords=dyn_q_coords_full,
            dyn_q_probs=dyn_q_probs_full,
            dyn_q_mask=dyn_q_mask_full,
            dyn_q_pos_branch=self.dyn_q_pos,
            dyn_q_pos_with_prob_branch=self.dyn_q_pos_with_prob,
            dyn_q_prob_branch=self.dyn_q_prob_branch,
            # These go to layer._forward via **layer_kwargs
            temp_memory=temp_memory,    # [B, M, C]
            temp_pos=temp_pos,          # [B, M, C]
            feat_flatten_img=feat_flatten_img,
            spatial_flatten_img=spatial_flatten_img,
            level_start_index_img=level_start_index_img,
            pc_range=self.pc_range,
            img_metas=img_metas_proxy,
            lidar2img=lidar2img,
            feat_flatten_pts=feat_flatten_pts,
            pos_flatten_pts=pos_flatten_pts,
            attn_masks=[attn_mask, None],
            query_key_padding_mask=None,
            key_padding_mask=None,
        )

        # outs_dec: [num_layers, B, Q, C] (already in [B, Q, C] convention)
        outs_dec = outs_dec

        # ===== 12. 分类 + 回归预测 =====
        outs_dec = torch.nan_to_num(outs_dec)
        outputs_classes = []
        outputs_coords = []
        outputs_is_dynamic = []
        for lvl in range(outs_dec.shape[0]):
            reference = inverse_sigmoid(reference_points_out[lvl].clone())
            assert reference.shape[-1] == 3
            outputs_class = self.cls_branches[lvl](outs_dec[lvl])
            tmp = self.reg_branches[lvl](outs_dec[lvl])
            tmp[..., 0:3] += reference[..., 0:3]
            tmp[..., 0:3] = tmp[..., 0:3].sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(tmp)
            outputs_is_dynamic.append(self.is_dynamic_branches[lvl](outs_dec[lvl]))

        all_cls_scores = torch.stack(outputs_classes)
        all_bbox_preds = torch.stack(outputs_coords)
        all_is_dynamic_preds = torch.stack(outputs_is_dynamic)  # [num_pred, B, Q_total, 1]
        all_bbox_preds[..., 0:3] = (
            all_bbox_preds[..., 0:3] * (self.pc_range[3:6] - self.pc_range[0:3])
            + self.pc_range[0:3])

        # ===== 12b. mask out invalid queries (ONNX-friendly) =====
        # 与原始 MV2DFusionHead.forward 对齐: 无效 query logit 设为 -40
        # 使用 torch.where 替代 boolean mask indexing, 保证 ONNX 兼容
        invalid_mask = (~tgt_query_mask).unsqueeze(0).unsqueeze(-1)  # [1, B, Q_total, 1]
        all_cls_scores = torch.where(
            invalid_mask,
            torch.tensor(-40.0, device=device, dtype=all_cls_scores.dtype),
            all_cls_scores)
        all_bbox_preds = torch.where(
            invalid_mask,
            torch.zeros(1, device=device, dtype=all_bbox_preds.dtype),
            all_bbox_preds)
        all_is_dynamic_preds = torch.where(
            invalid_mask,
            torch.tensor(-40.0, device=device, dtype=all_is_dynamic_preds.dtype),
            all_is_dynamic_preds)

        # ===== 12c-alt. no_nms 模式: 直接输出, 跳过 NMS 和 post_update =====
        if self.no_nms:
            outs_dec_last = outs_dec[-1]  # [B, Q_total, C]
            return (all_cls_scores, all_bbox_preds, all_is_dynamic_preds,
                    outs_dec_last, tgt_query_mask,
                    memory_embedding, memory_reference_point,
                    memory_timestamp, memory_velo,
                    memory_egopose, memory_query_mask)

        # ===== 12c. BEV NMS (ONNX-friendly) =====
        # 与 MV2DFusionHead.forward line 858-888 对齐
        # 解决 Track ID 不稳定的根因: 去除同一物体的新/旧 query 重复检测
        if len(self.post_bev_nms_ops) > 0:
            from torchvision.ops import nms as torchvision_nms

            last_preds = all_bbox_preds[-1]  # [B, Q_total, code_size]

            # denormalize_bbox inline: log-space → actual size, sin/cos → angle
            bev_cx = last_preds[..., 0]
            bev_cy = last_preds[..., 1]
            bev_w = last_preds[..., 3].clamp(max=10).exp()
            bev_l = last_preds[..., 4].clamp(max=10).exp()

            # score: sigmoid → max over classes
            score_bev = all_cls_scores[-1].sigmoid().max(-1).values  # [B, Q_total]
            score_bev = torch.where(
                tgt_query_mask, score_bev, torch.zeros_like(score_bev))

            # propagated_score_bonus: 给传播 query 加分 (优先保留)
            if self.propagated_score_bonus > 0 and self.num_propagated > 0:
                bonus = torch.zeros_like(score_bev)
                prop_start = num_query
                prop_end = num_query + self.num_propagated
                bonus[:, prop_start:prop_end] = self.propagated_score_bonus
                # 只给有效的 propagated query 加分
                prop_valid = tgt_query_mask[:, prop_start:prop_end].float()
                bonus[:, prop_start:prop_end] = bonus[:, prop_start:prop_end] * prop_valid
                score_bev = score_bev + bonus

            # xywhr → xyxy (轴对齐近似, 用于标准 NMS)
            half_w = bev_w / 2
            half_l = bev_l / 2
            boxes_x1 = bev_cx - half_w
            boxes_y1 = bev_cy - half_l
            boxes_x2 = bev_cx + half_w
            boxes_y2 = bev_cy + half_l
            boxes_xyxy = torch.stack(
                [boxes_x1, boxes_y1, boxes_x2, boxes_y2], dim=-1)  # [B, Q, 4]

            # 逐 batch 做 NMS (部署时 B=1)
            nms_keep_mask = torch.zeros_like(tgt_query_mask)
            for b in range(B):
                keep = torchvision_nms(
                    boxes_xyxy[b], score_bev[b],
                    iou_threshold=self.post_bev_nms_thr)
                nms_keep_mask[b, keep] = True

            # score_thr 过滤
            score_thr_mask = all_cls_scores[-1].sigmoid().max(-1).values > \
                self.post_bev_nms_score
            nms_keep_mask = nms_keep_mask & score_thr_mask

            # 更新 tgt_query_mask
            tgt_query_mask = tgt_query_mask & nms_keep_mask

            # 被 NMS 抑制的 query → logit = -40
            suppressed = (~tgt_query_mask).unsqueeze(0).unsqueeze(-1)
            all_cls_scores = torch.where(
                suppressed,
                torch.tensor(-40.0, device=device, dtype=all_cls_scores.dtype),
                all_cls_scores)
            all_bbox_preds = torch.where(
                suppressed,
                torch.zeros(1, device=device, dtype=all_bbox_preds.dtype),
                all_bbox_preds)
            all_is_dynamic_preds = torch.where(
                suppressed,
                torch.tensor(-40.0, device=device, dtype=all_is_dynamic_preds.dtype),
                all_is_dynamic_preds)

        # ===== 13. post_update_memory (ONNX-friendly) =====
        # 从最后一层获取用于更新 memory 的信息
        rec_reference_points = all_bbox_preds[-1, ..., :3]   # [B, Q_total, 3]
        rec_velo = all_bbox_preds[-1, ..., -2:]               # [B, Q_total, 2]
        rec_memory = outs_dec[-1]                              # [B, Q_total, C]

        # 取每个 query 的最大类别分数
        rec_score = all_cls_scores[-1].sigmoid()               # [B, Q_total, num_cls]
        rec_score = rec_score.max(dim=-1, keepdim=True).values # [B, Q_total, 1]

        # ONNX-friendly: 用 torch.where 替代 rec_score[~query_mask] = 0
        rec_score = torch.where(
            tgt_query_mask.unsqueeze(-1),
            rec_score,
            torch.zeros_like(rec_score))

        # topk 选择 topk_proposals 个最高分 query
        topk_score, topk_idx = torch.topk(
            rec_score, self.topk_proposals, dim=1)  # [B, K, 1]

        # gather 被选中的 query 信息
        K = self.topk_proposals
        rec_reference_points = torch.gather(
            rec_reference_points, 1,
            topk_idx.expand(-1, -1, 3))                        # [B, K, 3]
        rec_velo = torch.gather(
            rec_velo, 1,
            topk_idx.expand(-1, -1, 2))                        # [B, K, 2]
        rec_memory = torch.gather(
            rec_memory, 1,
            topk_idx.expand(-1, -1, self.embed_dims))          # [B, K, C]

        # rec_ego_pose: 当前帧的所有 query 都是 identity
        rec_ego_pose_sel = torch.eye(
            4, device=device, dtype=rec_memory.dtype
        ).unsqueeze(0).unsqueeze(0).expand(B, K, -1, -1)      # [B, K, 4, 4]

        # rec_timestamp: 当前帧 = 0
        rec_timestamp = torch.zeros(
            B, K, 1, device=device, dtype=rec_memory.dtype)

        # gather query_mask for selected queries
        rec_query_mask = torch.gather(
            tgt_query_mask.unsqueeze(-1).to(
                dtype=memory_query_mask.dtype), 1,
            topk_idx)                                           # [B, K, 1]

        # 拼接: [新 topk, 旧 memory] → 截断到 memory_len
        out_memory_embedding = torch.cat(
            [rec_memory, memory_embedding], dim=1
        )[:, :self.memory_len]                                  # [B, memory_len, C]

        out_memory_reference_point = torch.cat(
            [rec_reference_points, memory_reference_point], dim=1
        )[:, :self.memory_len]                                  # [B, memory_len, 3]

        out_memory_timestamp = torch.cat(
            [rec_timestamp, memory_timestamp], dim=1
        )[:, :self.memory_len]                                  # [B, memory_len, 1]

        out_memory_velo = torch.cat(
            [rec_velo, memory_velo], dim=1
        )[:, :self.memory_len]                                  # [B, memory_len, 2]

        out_memory_egopose = torch.cat(
            [rec_ego_pose_sel, memory_egopose], dim=1
        )[:, :self.memory_len]                                  # [B, memory_len, 4, 4]

        out_memory_query_mask = torch.cat(
            [rec_query_mask, memory_query_mask], dim=1
        )[:, :self.memory_len]                                  # [B, memory_len, 1]

        # 用 ego_pose 变换 reference_points (齐次坐标变换)
        # ONNX-friendly: 显式 expand + bmm
        M_out = self.memory_len
        ref_homo = torch.cat([
            out_memory_reference_point,
            torch.ones_like(out_memory_reference_point[..., :1])
        ], dim=-1)                                              # [B, memory_len, 4]
        ego_pose_exp = ego_pose.unsqueeze(1).expand(B, M_out, 4, 4).reshape(B * M_out, 4, 4)
        out_memory_reference_point = torch.bmm(
            ego_pose_exp, ref_homo.reshape(B * M_out, 4, 1)
        ).reshape(B, M_out, 4)[..., :3]                        # [B, memory_len, 3]

        # 调整 timestamp: 减去当前帧时间戳
        out_memory_timestamp = out_memory_timestamp - \
            timestamp.unsqueeze(-1).unsqueeze(-1)               # [B, memory_len, 1]

        # 累积 ego_pose
        # ONNX-friendly: 显式 expand + bmm
        out_memory_egopose = torch.bmm(
            ego_pose_exp, out_memory_egopose.reshape(B * M_out, 4, 4)
        ).reshape(B, M_out, 4, 4)                              # [B, memory_len, 4, 4]

        return (all_cls_scores, all_bbox_preds, all_is_dynamic_preds,
                out_memory_embedding, out_memory_reference_point,
                out_memory_timestamp, out_memory_velo,
                out_memory_egopose, out_memory_query_mask)


def build_model_from_config(config_path, checkpoint_path):
    """从配置文件构建完整模型并加载权重。"""
    cfg = Config.fromfile(config_path)

    # 设置 plugin
    if hasattr(cfg, 'plugin') and cfg.plugin:
        import importlib
        if hasattr(cfg, 'plugin_dir'):
            plugin_dirs = cfg.plugin_dir if isinstance(
                cfg.plugin_dir, list) else [cfg.plugin_dir]
            for plugin_dir in plugin_dirs:
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(f'Loading plugin: {_module_path}')
                plg_lib = importlib.import_module(_module_path)

    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    model.eval()

    # 加载 checkpoint
    print(f'Loading checkpoint: {checkpoint_path}')
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')
    print('Checkpoint loaded successfully.')

    return model, cfg


def _get_pad_size(cfg):
    """从配置文件中提取 padded 图像尺寸 (H_pad, W_pad)。"""
    # 从 ida_aug_conf 中获取 final_dim
    final_h, final_w = None, None
    for src in [cfg, cfg.get('data', {})]:
        if hasattr(src, 'ida_aug_conf'):
            final_h, final_w = src.ida_aug_conf['final_dim']
            break
    if final_h is None:
        # 尝试从 train_pipeline / test_pipeline 里找 ResizeCropFlipRotImage
        for pipe_key in ['train_pipeline', 'test_pipeline']:
            pipeline = cfg.get(pipe_key, [])
            for step in pipeline:
                if isinstance(step, dict):
                    aug_conf = step.get('data_aug_conf', None)
                    if aug_conf and 'final_dim' in aug_conf:
                        final_h, final_w = aug_conf['final_dim']
                        break
            if final_h is not None:
                break
    if final_h is None:
        final_h, final_w = 640, 1600
        print(f'  [WARN] Cannot find final_dim in config, '
              f'using default: ({final_h}, {final_w})')

    # PadMultiViewImage size_divisor (默认 32)
    size_div = 32
    H_pad = int(math.ceil(final_h / size_div) * size_div)
    W_pad = int(math.ceil(final_w / size_div) * size_div)
    return H_pad, W_pad


def _get_fpn_spatial_shapes(cfg):
    """从配置中计算图像特征空间形状 (H, W)。
    
    支持两种路径:
      1. FPN (img_neck) 路径: 从 backbone out_indices + neck config 推算 strides
      2. HybridEncoder (RT-DETR) 路径: 直接从 hybrid_encoder.feat_strides 读取,
         并且只取前 2 级 (stride 8, 16), 因为 extract_img_feat 做了
         img_feats_det = img_feats[:2]
    """
    H_pad, W_pad = _get_pad_size(cfg)
    import math as _math

    # 从 config 中读取 FPN level 范围
    head_cfg = cfg.model.get('fusion_bbox_head', {})
    img_feat_start_level = head_cfg.get('img_feat_start_level', 1)
    decoder_layer_cfg = head_cfg.get('transformer', {}).get('decoder', {}).get('transformerlayers', {})
    cross_attn_cfg = decoder_layer_cfg.get('attn_cfgs', [{}])[-1]
    num_levels = cross_attn_cfg.get('num_levels', 4)

    # ---- RT-DETR HybridEncoder 路径 ----
    he_cfg = cfg.model.get('hybrid_encoder', None)
    if he_cfg is not None:
        # HybridEncoder outputs 3 levels (feat_strides=[8,16,32]),
        # but extract_img_feat does: img_feats_det = img_feats[:2]
        # So fusion head only sees the first 2 levels (stride 8, 16).
        all_strides = list(he_cfg.get('feat_strides', [8, 16, 32]))
        # img_feats_det = img_feats[:2] means only first 2 levels
        det_strides = all_strides[:2]
        selected_strides = det_strides[img_feat_start_level:
                                       img_feat_start_level + num_levels]

        shapes = [(_math.ceil(H_pad / s), _math.ceil(W_pad / s))
                  for s in selected_strides]

        print(f'  HybridEncoder path:')
        print(f'    all feat_strides: {all_strides}')
        print(f'    img_feats_det strides ([:2]): {det_strides}')
        print(f'    img_feat_start_level={img_feat_start_level}, num_levels={num_levels}')
        print(f'    selected strides: {selected_strides}')
        print(f'    spatial shapes: {shapes}')

        return shapes, (H_pad, W_pad)

    # ---- FPN (img_neck) 路径 ----
    neck_cfg = cfg.model.get('img_neck', {})
    neck_start_level = neck_cfg.get('start_level', 0)
    backbone_cfg = cfg.model.get('img_backbone', {})
    num_stages = backbone_cfg.get('num_stages', 4)

    # ResNet stage strides: stage 0=4, stage 1=8, stage 2=16, stage 3=32
    stage_strides = [4, 8, 16, 32][:num_stages]
    out_indices = list(backbone_cfg.get('out_indices', tuple(range(num_stages))))
    fpn_base_strides = [stage_strides[i] for i in out_indices]
    # FPN may add extra levels (add_extra_convs) with stride doubling
    num_outs = neck_cfg.get('num_outs', len(fpn_base_strides))
    while len(fpn_base_strides) < num_outs:
        fpn_base_strides.append(fpn_base_strides[-1] * 2)

    selected_strides = fpn_base_strides[img_feat_start_level:
                                        img_feat_start_level + num_levels]

    shapes = [(_math.ceil(H_pad / s), _math.ceil(W_pad / s))
              for s in selected_strides]

    print(f'  FPN path:')
    print(f'    img_feat_start_level={img_feat_start_level}, num_levels={num_levels}')
    print(f'    FPN all strides: {fpn_base_strides}')
    print(f'    selected strides: {selected_strides}')
    print(f'    spatial shapes: {shapes}')

    return shapes, (H_pad, W_pad)


def create_dummy_inputs(cfg, device='cpu', no_nms=False):
    """
    根据配置文件创建 dummy 输入张量。
    自动从 cfg 中读取图像尺寸、BEV 尺寸、num_cams 等参数。

    Args:
        no_nms: 如果 True, 不包含 ego_pose (post_update_memory 在外部完成)
    """
    B = 1

    # ===== 从配置中提取参数 =====
    head_cfg = cfg.model.fusion_bbox_head
    C = head_cfg.get('in_channels', 256)
    C_pts = head_cfg.get('pts_in_channels', 512)
    prob_bin = head_cfg.get('prob_bin', 25)

    # 从 cross attention 配置中读取 num_cams / bev_h / bev_w
    decoder_layer_cfg = head_cfg.transformer.decoder.transformerlayers
    cross_attn_cfg = decoder_layer_cfg.attn_cfgs[1]
    N = cross_attn_cfg.get('num_cams', 7)
    bev_h = cross_attn_cfg.get('bev_h', 200)
    bev_w = cross_attn_cfg.get('bev_w', 126)

    # FPN 空间形状
    fpn_shapes, (H_pad, W_pad) = _get_fpn_spatial_shapes(cfg)
    print(f'  Image pad size: H_pad={H_pad}, W_pad={W_pad}')
    print(f'  num_cams={N}, bev_h={bev_h}, bev_w={bev_w}')
    print(f'  C={C}, C_pts={C_pts}, prob_bin={prob_bin}')
    print(f'  FPN spatial shapes (levels 1-4): {fpn_shapes}')

    # ----- feat_flatten_img (已经过 spatial_alignment, 由 2D ONNX 输出) -----
    # 计算 L_img = sum(H_i * W_i) for FPN levels
    L_img = sum(h * w for h, w in fpn_shapes)
    feat_flatten_img = torch.randn(N, L_img, C, device=device)

    # NOTE: spatial_flatten_img 已作为 buffer 注册在 wrapper 中，不再作为 ONNX 输入

    # ----- lidar2img (由 2D ONNX 输出) -----
    lidar2img = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0)\
        .repeat(B, N, 1, 1)

    # ----- 点云特征 -----
    L_pts = bev_h * bev_w
    pts_feat = torch.randn(B, L_pts, C_pts, device=device)

    # ----- 点云 query -----
    N_pts_q = 100   # CenterPoint test_cfg.all_task_max_num
    pts_query_center = torch.randn(B, N_pts_q, 3, device=device) * 20
    pts_query_feat = torch.randn(B, N_pts_q, C_pts, device=device)

    # ----- 动态 query (图像) -----
    # dyn_query: [B, N_img_q, prob_bin, 4]  最后一维 = [x, y, z, depth_prob]
    # dyn_query_feats: [B, N_img_q, C]
    # 这两个输入直接对接 image_branch ONNX 的输出
    # N_img_q = max_img_query_per_sample (2D global_topk), 与 2D ONNX 输出一致
    N_img_q = cfg.model.get('max_img_query_per_sample', 200)

    dyn_query_coords_raw = torch.rand(B, N_img_q, prob_bin, 3, device=device) * 50 - 25
    dyn_query_probs_raw = torch.softmax(
        torch.randn(B, N_img_q, prob_bin, device=device), dim=-1)
    dyn_query = torch.cat(
        [dyn_query_coords_raw, dyn_query_probs_raw.unsqueeze(-1)],
        dim=-1)  # [B, N_img_q, prob_bin, 4]
    dyn_query_feats = torch.randn(B, N_img_q, C, device=device)

    # ----- 时序 memory bank -----
    memory_len = head_cfg.get('memory_len', 1536)
    num_propagated = head_cfg.get('num_propagated', 256)
    print(f'  memory_len={memory_len}, num_propagated={num_propagated}')

    # 首帧初始化: 全零 memory, 但 propagated slots 设置 pseudo_reference_points
    memory_embedding = torch.zeros(B, memory_len, C, device=device)
    memory_reference_point = torch.zeros(B, memory_len, 3, device=device)
    memory_timestamp = torch.zeros(B, memory_len, 1, device=device)
    memory_velo = torch.zeros(B, memory_len, 2, device=device)
    memory_egopose = torch.zeros(B, memory_len, 4, 4, device=device)
    memory_query_mask = torch.zeros(
        B, memory_len, 1, dtype=torch.bool, device=device)

    # 首帧: propagated slots 需要初始化 (模拟 pre_update_memory 对首帧的处理)
    if num_propagated > 0:
        memory_egopose[:, :num_propagated] = torch.eye(
            4, device=device).unsqueeze(0).unsqueeze(0)
        memory_query_mask[:, :num_propagated] = True

    # ----- 当前帧信息 -----
    ego_pose_inv = torch.eye(4, device=device).unsqueeze(0).repeat(B, 1, 1)
    timestamp = torch.zeros(B, device=device)
    prev_exists = torch.zeros(B, device=device)  # 0 = 首帧
    prev_timestamp = torch.zeros(B, device=device)  # 上一帧时间戳

    inputs = dict(
        feat_flatten_img=feat_flatten_img,
        lidar2img=lidar2img,
        pts_feat=pts_feat,
        pts_query_center=pts_query_center,
        pts_query_feat=pts_query_feat,
        dyn_query=dyn_query,
        dyn_query_feats=dyn_query_feats,
        memory_embedding=memory_embedding,
        memory_reference_point=memory_reference_point,
        memory_timestamp=memory_timestamp,
        memory_velo=memory_velo,
        memory_egopose=memory_egopose,
        memory_query_mask=memory_query_mask,
    )

    # ego_pose: 完整模式需要 (post_update_memory), no-nms 传 dummy
    # 注意: 必须始终包含 ego_pose 以保持 forward() 的参数位置对齐,
    # 否则 torch.onnx.export 按位置传参会导致后续参数 (ego_pose_inv,
    # timestamp, prev_exists, prev_timestamp) 全部错位
    if no_nms:
        ego_pose = torch.zeros(B, 4, 4, device=device)  # no-nms 不使用, 占位
    else:
        ego_pose = torch.eye(4, device=device).unsqueeze(0).repeat(B, 1, 1)
    inputs['ego_pose'] = ego_pose

    inputs['ego_pose_inv'] = ego_pose_inv
    inputs['timestamp'] = timestamp
    inputs['prev_exists'] = prev_exists
    inputs['prev_timestamp'] = prev_timestamp

    return inputs


def verify_forward(wrapper, dummy_inputs, device='cpu'):
    """验证 wrapper 的 forward 能正常运行。"""
    print('\n' + '=' * 60)
    print('Verifying forward pass...')
    print('=' * 60)

    wrapper = wrapper.to(device)
    inputs = {k: v.to(device) for k, v in dummy_inputs.items()}

    with torch.no_grad():
        outputs = wrapper(**inputs)

    if wrapper.no_nms:
        (cls_scores, bbox_preds, is_dynamic_preds,
         outs_dec_last, tgt_query_mask,
         pre_mem_emb, pre_mem_ref, pre_mem_ts, pre_mem_vel,
         pre_mem_ego, pre_mem_mask) = outputs
        print(f'  [no-nms mode]')
        print(f'  all_cls_scores shape:        {cls_scores.shape}')
        print(f'  all_bbox_preds shape:        {bbox_preds.shape}')
        print(f'  all_is_dynamic_preds shape:  {is_dynamic_preds.shape}')
        print(f'  outs_dec_last shape:         {outs_dec_last.shape}')
        print(f'  tgt_query_mask shape:        {tgt_query_mask.shape}')
        print(f'  cls_scores range: '
              f'[{cls_scores.min().item():.4f}, {cls_scores.max().item():.4f}]')
        print(f'  --- pre-update memory bank ---')
        print(f'  pre_memory_embedding:        {pre_mem_emb.shape}')
        print(f'  pre_memory_reference_point:  {pre_mem_ref.shape}')
        print(f'  pre_memory_timestamp:        {pre_mem_ts.shape}')
        print(f'  pre_memory_velo:             {pre_mem_vel.shape}')
        print(f'  pre_memory_egopose:          {pre_mem_ego.shape}')
        print(f'  pre_memory_query_mask:       {pre_mem_mask.shape}')
    else:
        (cls_scores, bbox_preds, is_dynamic_preds,
         out_mem_emb, out_mem_ref, out_mem_ts, out_mem_vel,
         out_mem_ego, out_mem_mask) = outputs
        print(f'  [full mode with NMS + post_update]')
        print(f'  all_cls_scores shape:        {cls_scores.shape}')
        print(f'  all_bbox_preds shape:        {bbox_preds.shape}')
        print(f'  all_is_dynamic_preds shape:  {is_dynamic_preds.shape}')
        print(f'  cls_scores range: '
              f'[{cls_scores.min().item():.4f}, {cls_scores.max().item():.4f}]')
        print(f'  bbox_preds range:  '
              f'[{bbox_preds.min().item():.4f}, {bbox_preds.max().item():.4f}]')
        print(f'  --- updated memory bank ---')
        print(f'  out_memory_embedding:        {out_mem_emb.shape}')
        print(f'  out_memory_reference_point:  {out_mem_ref.shape}')
        print(f'  out_memory_timestamp:        {out_mem_ts.shape}')
        print(f'  out_memory_velo:             {out_mem_vel.shape}')
        print(f'  out_memory_egopose:          {out_mem_ego.shape}')
        print(f'  out_memory_query_mask:       {out_mem_mask.shape}')

    print('Forward pass OK!\n')
    return outputs


def export_to_onnx(wrapper, dummy_inputs, output_path,
                   opset_version=16, device='cpu'):
    """导出 ONNX 模型。"""
    print('\n' + '=' * 60)
    print(f'Exporting to ONNX: {output_path}')
    print(f'  opset_version: {opset_version}')
    print(f'  no_nms: {wrapper.no_nms}')
    print('=' * 60)

    wrapper = wrapper.to(device).eval()
    inputs = {k: v.to(device) for k, v in dummy_inputs.items()}

    input_names = list(inputs.keys())
    input_tuple = tuple(inputs[k] for k in input_names)

    if wrapper.no_nms:
        output_names = [
            'all_cls_scores', 'all_bbox_preds', 'all_is_dynamic_preds',
            'outs_dec_last', 'tgt_query_mask',
            'pre_memory_embedding', 'pre_memory_reference_point',
            'pre_memory_timestamp', 'pre_memory_velo',
            'pre_memory_egopose', 'pre_memory_query_mask',
        ]
    else:
        output_names = [
            'all_cls_scores', 'all_bbox_preds', 'all_is_dynamic_preds',
            'out_memory_embedding', 'out_memory_reference_point',
            'out_memory_timestamp', 'out_memory_velo',
            'out_memory_egopose', 'out_memory_query_mask',
        ]

    # NOTE: 不设置 dynamic_axes，让所有维度保持静态。
    # 原因: feat_flatten_img / mln_input 的 dim 0 = B*N (=7)，
    #        而其他输入的 dim 0 = B (=1)。
    #        如果都标记为同一个动态轴 'batch_or_BN'，trtexec 在没有
    #        --minShapes/--optShapes/--maxShapes 时会将所有动态维度
    #        自动设为 1，导致 feat_flatten_img dim0=1 但 lidar2img
    #        的 N=7 维度不变，内部 reshape 元素数不匹配而报错:
    #        "Reshaping [7,x,8,52] to [1,x,8,4,13]"
    # 部署时 B=1, N=7 均为固定值，无需动态轴。
    dynamic_axes = {}

    os.makedirs(
        os.path.dirname(output_path)
        if os.path.dirname(output_path) else '.', exist_ok=True)

    try:
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                input_tuple,
                output_path,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=opset_version,
                do_constant_folding=True,
                verbose=False,
            )
        print(f'\nONNX export successful: {output_path}')
        print(f'  File size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB')

        # ---- onnx-simplifier ----
        try:
            import onnx
            from onnxsim import simplify as onnxsim_simplify
            print('\nRunning onnx-simplifier ...')
            model_onnx = onnx.load(output_path)
            model_sim, check = onnxsim_simplify(model_onnx)
            if check:
                sim_path = output_path.replace('.onnx', '_sim.onnx')
                onnx.save(model_sim, sim_path)
                print(f'  Simplified model saved: {sim_path}')
                print(f'  File size: '
                      f'{os.path.getsize(sim_path) / 1024 / 1024:.1f} MB')
            else:
                print('  [WARNING] onnx-simplifier check failed, '
                      'using original model.')
        except ImportError:
            print('\n[INFO] onnxsim not installed, skipping simplification. '
                  'Install with: pip install onnxsim')
        except Exception as e_sim:
            print(f'\n[WARNING] onnx-simplifier failed: {e_sim}')
            print('  Using original (unsimplified) model.')

    except Exception as e:
        print(f'\n[WARNING] torch.onnx.export failed: {e}')
        print('\nFalling back to torch.jit.trace + save ...')
        traced_path = output_path.replace('.onnx', '.pt')
        try:
            with torch.no_grad():
                traced = torch.jit.trace(wrapper, input_tuple)
            traced.save(traced_path)
            print(f'Traced model saved: {traced_path}')
        except Exception as e2:
            print(f'torch.jit.trace also failed: {e2}')
            print('\nSaving as plain state_dict instead...')
            sd_path = output_path.replace('.onnx', '_state_dict.pth')
            torch.save(wrapper.state_dict(), sd_path)
            print(f'State dict saved: {sd_path}')


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
            arr = arr.astype(np.int32)
        elif arr.dtype in (np.int32, np.int64):
            pass
        else:
            arr = arr.astype(np.float32)
        path = os.path.join(save_dir, f'{name}.bin')
        arr.tofile(path)
        size_kb = os.path.getsize(path) / 1024
        shapes_lines.append(f'{name}: shape={list(arr.shape)}, dtype={arr.dtype}')
    # 保存 shapes 描述文件
    info_path = os.path.join(save_dir, 'shapes.txt')
    with open(info_path, 'w') as f:
        f.write('\n'.join(shapes_lines) + '\n')


def run_all_frames_save_io(model, cfg, ann_file, save_io_dir, device='cuda'):
    """
    对 pkl 中的所有帧运行完整模型，保存 fusion head 每帧的输入输出。

    流程:
      1. 构建测试数据集 + dataloader
      2. 启用 fusion head 的 IO saving 机制 (_save_io_dir)；每帧子目录名为
         pkl 中 ``token``（重复 token 时加 ``_1``、``_2`` …），并写入 ``manifest.json``
      3. 逐帧运行 model.forward_test → fusion head 自动:
         - 保存输入 (pts_feat, dyn_query, memory_*, ego_pose, lidar2img, ...)
         - pre_update_memory (时序 memory 变换)
         - Transformer Decoder + 分类/回归
         - 保存输出 (all_cls_scores, all_bbox_preds, outs_dec_last, tgt_query_mask)
         - 保存 pre_memory_* (pre_update_memory 之后的 memory 状态)
         - post_update_memory (更新 memory bank 供下一帧使用)
      4. pseudo_reference_points 在首帧由 fusion head 自动保存

    Args:
        model: 完整的 MV2DFusion 模型 (已加载权重)
        cfg: mmcv Config 对象
        ann_file: pkl 文件路径 (包含所有帧的信息)
        save_io_dir: 输出目录
        device: 运行设备 ('cuda' or 'cpu')
    """
    from mmdet3d.datasets import build_dataset
    from projects.mmdet3d_plugin.datasets.builder import build_dataloader
    from mmcv.parallel import MMDataParallel
    import pickle

    with open(ann_file, 'rb') as f:
        pkl_data = pickle.load(f)
    data_infos = pkl_data.get('infos', [])
    dup_counts = {}
    full_subdirs = []
    for fi, info in enumerate(data_infos):
        token = info.get('token', f'frame_{fi}')
        full_subdirs.append(_subdir_name_for_token(token, fi, dup_counts))

    # 1. 构建测试数据集
    test_cfg = copy.deepcopy(cfg.data.test)
    test_cfg['ann_file'] = ann_file
    test_cfg['test_mode'] = True
    dataset = build_dataset(test_cfg)
    num_frames = len(dataset)

    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,  # 0 for debugging / deterministic order
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.get('nonshuffler_sampler',
                                         dict(type='DistributedSampler')),
    )

    print(f'\n{"="*60}')
    print(f'[all-frames IO] Dataset: {ann_file}')
    print(f'[all-frames IO] Total frames: {num_frames}')
    print(f'[all-frames IO] Output dir: {save_io_dir}')
    print(f'{"="*60}')

    _save_io_frame_dirs = full_subdirs[:num_frames]
    while len(_save_io_frame_dirs) < num_frames:
        fi = len(_save_io_frame_dirs)
        _save_io_frame_dirs.append(f'frame_{fi:04d}')
    if len(data_infos) != num_frames:
        print(f'[WARNING] pkl infos count ({len(data_infos)}) != dataset length '
              f'({num_frames}); per-frame dirs aligned to dataset order.')

    # 2. 启用 fusion head IO saving
    head = model.fusion_bbox_head
    head._save_io_dir = save_io_dir
    head._save_io_frame_dirs = _save_io_frame_dirs
    head._save_io_frame_idx = 0

    # 确保 memory 被 reset (首帧自动处理)
    head.memory_embedding = None

    os.makedirs(save_io_dir, exist_ok=True)

    # 3. 保存 dataset metadata (ego_pose, timestamp, token)
    for fi, info in enumerate(data_infos):
        sub = full_subdirs[fi] if fi < len(full_subdirs) else f'frame_{fi:04d}'
        fd = os.path.join(save_io_dir, sub)
        os.makedirs(fd, exist_ok=True)
        # ego2global as ego_pose [4,4]
        from pyquaternion import Quaternion as _Quat
        e2g_r = _Quat(info['ego2global_rotation']).rotation_matrix
        e2g_t = np.array(info['ego2global_translation'])
        ego_pose = np.eye(4, dtype=np.float32)
        ego_pose[:3, :3] = e2g_r
        ego_pose[:3, 3] = e2g_t
        ego_pose.tofile(os.path.join(fd, 'ego_pose_global.bin'))
        # timestamp
        np.array([info['timestamp']], dtype=np.float64).tofile(
            os.path.join(fd, 'timestamp_global.bin'))
        # token
        with open(os.path.join(fd, 'token.txt'), 'w') as tf:
            tf.write(str(info.get('token', '')))
    print(f'[all-frames IO] Saved metadata for {len(data_infos)} frames')

    # 4. Wrap model and run all frames
    if device == 'cuda' or device.startswith('cuda:'):
        model = model.cuda()
        model = MMDataParallel(model, device_ids=[0])
    model.eval()

    for frame_idx, data_dict in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data_dict)

        if frame_idx % 10 == 0 or frame_idx == num_frames - 1:
            sub = _save_io_frame_dirs[frame_idx]
            print(f'  Frame {frame_idx + 1}/{num_frames} ({sub}): done')

    manifest = []
    for fi in range(num_frames):
        token = ''
        scene_token = None
        if fi < len(data_infos):
            token = data_infos[fi].get('token', '')
            scene_token = data_infos[fi].get('scene_token')
        manifest.append({
            'frame': fi,
            'subdir': _save_io_frame_dirs[fi],
            'token': token,
            'scene_token': scene_token,
        })
    man_path = os.path.join(save_io_dir, 'manifest.json')
    with open(man_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f'\n[all-frames IO] All {num_frames} frames saved to {save_io_dir}')
    print(f'  Per-frame directories: {save_io_dir}/<token>/ (see {man_path})')
    print(f'  Each frame contains:')
    print(f'    input_*.bin  — fusion head 输入 (ONNX wrapper 接口)')
    print(f'    pre_memory_*.bin — pre_update_memory 之后的 memory 状态')
    print(f'    all_cls_scores.bin, all_bbox_preds.bin — 分类/回归输出')
    print(f'    outs_dec_last.bin — decoder 最后一层输出')
    print(f'    tgt_query_mask.bin, obj_idxes.bin — query mask + tracking IDs')
    if hasattr(head if not hasattr(model, 'module') else model.module.fusion_bbox_head,
               'pseudo_reference_points'):
        print(f'  Global files:')
        print(f'    pseudo_reference_points_normalized.bin — [num_propagated, 3]')
        print(f'    pseudo_reference_points_world.bin — [num_propagated, 3]')


def main():
    parser = argparse.ArgumentParser(
        description='Export MV2DFusionHead to ONNX')
    parser.add_argument(
        '--config', type=str,
        default='projects/configs/nusc/'
                'mv2dfusion-centerpoint-fcos_1600_gridmask-ep24_'
                'nusc_base_epoch_hellodata_0227_hw.py',
        help='Config file path')
    parser.add_argument(
        '--checkpoint', type=str,
        default='/mnt/volumes/ad-perception-al-sh01/cm/mv2d/work_dirs/'
                'mv2dfusion-centerpoint-fcos_1600_gridmask-ep24_'
                'nusc_base_epoch_hellodata_0227_hw/'
                'iter_10101.pth',
        help='Checkpoint file path')
    parser.add_argument(
        '--output', type=str, default='fusion_head.onnx',
        help='Output ONNX file path')
    parser.add_argument(
        '--opset', type=int, default=16,
        help='ONNX opset version (>=16 recommended for grid_sample)')
    parser.add_argument(
        '--device', type=str, default='cpu',
        help='Device for export (cpu recommended to avoid device mismatch)')
    parser.add_argument(
        '--verify-only', action='store_true',
        help='Only verify forward pass, do not export')
    parser.add_argument(
        '--no-nms', action='store_true',
        help='Export without NMS and post_update_memory '
             '(outputs outs_dec_last + tgt_query_mask + pre_memory_*)')
    # ---- 精度对齐: 真实数据 IO 保存 ----
    parser.add_argument(
        '--ann-file', type=str, default=None,
        help='pkl 文件路径 (包含所有帧信息)，使用真实数据替代 dummy inputs')
    parser.add_argument(
        '--sample-idx', type=int, default=0,
        help='pkl 中的 sample 索引 (仅单帧模式使用)')
    parser.add_argument(
        '--save-io', type=str, default=None,
        help='保存每帧输入/输出为 .bin 文件的目录')
    parser.add_argument(
        '--io-dir-2d', type=str, default=None,
        help='2D ONNX 输出目录 (单帧模式: 加载 feat_flatten_img, dyn_query 等)')
    parser.add_argument(
        '--io-dir-pts', type=str, default=None,
        help='3D ONNX 输出目录 (单帧模式: 加载 pts_feat, pts_query_center 等)')
    args = parser.parse_args()

    # 1. 构建完整模型并加载权重
    model, cfg = build_model_from_config(args.config, args.checkpoint)

    # 1.5 Also patch the transformer module's local reference
    #     (in case it was bound before our early patch)
    import projects.mmdet3d_plugin.models.utils.mv2dfusion_transformer as _tfm_mod
    _tfm_mod.MultiScaleDeformableAttnFunction = _MSDAProxy

    # ================================================================
    # 全帧 IO 保存模式: --ann-file + --save-io
    # 使用完整模型推理所有帧，fusion head 内部自动保存每帧 IO
    # 包含时序 memory 的自然更新 (pre/post_update_memory)
    # ================================================================
    if args.ann_file and args.save_io:
        run_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        run_all_frames_save_io(
            model, cfg, args.ann_file, args.save_io, device=run_device)
        print('\nDone!')
        return

    # ================================================================
    # 原有模式: ONNX 导出 / 单帧验证
    # ================================================================
    # 1.6 Force model to target device to avoid mixed cuda/cpu tensors
    model = model.to(args.device)

    # 2. 提取 fusion_bbox_head
    head = model.fusion_bbox_head
    head.eval()
    print(f'\nExtracted fusion_bbox_head: {type(head).__name__}')
    print(f'  embed_dims: {head.embed_dims}')
    print(f'  num_pred (decoder layers): {head.num_pred}')
    print(f'  num_classes: {head.num_classes}')
    print(f'  code_size: {head.code_size}')
    print(f'  prob_bin: {head.prob_bin}')
    print(f'  pc_range: {head.pc_range.data.tolist()}')

    # 3. 创建 ONNX wrapper (pad 尺寸 + FPN 空间形状 固化为 buffer)
    fpn_shapes, pad_hw = _get_fpn_spatial_shapes(cfg)
    wrapper = MV2DFusionHeadForONNX(
        head,
        pad_shape_hw=pad_hw,
        fpn_spatial_shapes=fpn_shapes,
        no_nms=args.no_nms,
    )
    if args.no_nms:
        print(f'  [no-nms mode] Skipping NMS and post_update_memory')
    # Ensure wrapper and ALL sub-modules/buffers are on the same device
    wrapper = wrapper.to(args.device).eval()
    print(f'\nCreated ONNX wrapper: {type(wrapper).__name__}')

    # Verify all parameters/buffers are on the target device
    device_set = set()
    for name, p in wrapper.named_parameters():
        device_set.add(str(p.device))
    for name, b in wrapper.named_buffers():
        device_set.add(str(b.device))
    print(f'  Devices in wrapper: {device_set}')
    if len(device_set) > 1:
        print(f'  [WARNING] Mixed devices detected! Forcing all to {args.device}')
        wrapper = wrapper.to(args.device)

    n_params = sum(p.numel() for p in wrapper.parameters())
    n_params_trainable = sum(
        p.numel() for p in wrapper.parameters() if p.requires_grad)
    print(f'  Total parameters: {n_params / 1e6:.2f} M')
    print(f'  Trainable parameters: {n_params_trainable / 1e6:.2f} M')

    # 4. 创建输入 (真实数据 or dummy)
    if args.io_dir_2d and args.io_dir_pts and args.ann_file:
        # 单帧真实数据模式: 从预计算的 2D/3D IO 目录加载
        dummy_inputs = load_real_inputs_from_dirs(
            args.ann_file, args.sample_idx,
            args.io_dir_2d, args.io_dir_pts,
            cfg, device=args.device, no_nms=args.no_nms)
    else:
        # Dummy 输入模式
        dummy_inputs = create_dummy_inputs(
            cfg, device=args.device, no_nms=args.no_nms)

    # 4.5 pseudo_reference_points 初始化已移入 forward() 的 pre_update_memory
    #     首帧 prev_exists=0 时自动完成, 无需外部初始化

    print(f'\nInputs created:')
    for k, v in dummy_inputs.items():
        print(f'  {k}: {v.shape} ({v.dtype})')

    # 5. 验证 forward
    outputs = verify_forward(wrapper, dummy_inputs, device=args.device)

    # 5.1. 保存单帧 IO (如果指定了 --save-io 但没有 --ann-file 的全帧模式)
    if args.save_io and not args.ann_file:
        frame_dir = os.path.join(args.save_io, 'frame_0000')
        print(f'\nSaving single-frame IO to {frame_dir}:')
        print(f'  Saving inputs to {frame_dir}/inputs/:')
        save_tensors_as_bin(dummy_inputs, os.path.join(frame_dir, 'inputs'))
        print(f'  Saving outputs to {frame_dir}/outputs/:')
        if wrapper.no_nms:
            (cls_scores, bbox_preds, is_dynamic_preds,
             outs_dec_last, tgt_query_mask,
             pre_mem_emb, pre_mem_ref, pre_mem_ts, pre_mem_vel,
             pre_mem_ego, pre_mem_mask) = outputs
            output_dict = {
                'all_cls_scores': cls_scores,
                'all_bbox_preds': bbox_preds,
                'all_is_dynamic_preds': is_dynamic_preds,
                'outs_dec_last': outs_dec_last,
                'tgt_query_mask': tgt_query_mask,
                'pre_memory_embedding': pre_mem_emb,
                'pre_memory_reference_point': pre_mem_ref,
                'pre_memory_timestamp': pre_mem_ts,
                'pre_memory_velo': pre_mem_vel,
                'pre_memory_egopose': pre_mem_ego,
                'pre_memory_query_mask': pre_mem_mask,
            }
        else:
            (cls_scores, bbox_preds, is_dynamic_preds,
             out_mem_emb, out_mem_ref, out_mem_ts, out_mem_vel,
             out_mem_ego, out_mem_mask) = outputs
            output_dict = {
                'all_cls_scores': cls_scores,
                'all_bbox_preds': bbox_preds,
                'all_is_dynamic_preds': is_dynamic_preds,
                'out_memory_embedding': out_mem_emb,
                'out_memory_reference_point': out_mem_ref,
                'out_memory_timestamp': out_mem_ts,
                'out_memory_velo': out_mem_vel,
                'out_memory_egopose': out_mem_ego,
                'out_memory_query_mask': out_mem_mask,
            }
        save_tensors_as_bin(output_dict, os.path.join(frame_dir, 'outputs'))

        # 保存 pseudo_reference_points
        if hasattr(wrapper, 'pseudo_reference_points') and wrapper.num_propagated > 0:
            pseudo_ref = wrapper.pseudo_reference_points.weight.data.cpu().numpy().astype(np.float32)
            pseudo_ref_world = wrapper.pseudo_ref_world.cpu().numpy().astype(np.float32)
            pseudo_ref.tofile(os.path.join(args.save_io, 'pseudo_reference_points_normalized.bin'))
            pseudo_ref_world.tofile(os.path.join(args.save_io, 'pseudo_reference_points_world.bin'))
            with open(os.path.join(args.save_io, 'pseudo_reference_points_shapes.txt'), 'w') as f:
                f.write(f'pseudo_reference_points_normalized: shape={list(pseudo_ref.shape)}, dtype=float32\n')
                f.write(f'pseudo_reference_points_world: shape={list(pseudo_ref_world.shape)}, dtype=float32\n')
            print(f'  pseudo_reference_points saved: {list(pseudo_ref.shape)}')

    # 6. 导出 ONNX
    if not args.verify_only:
        export_to_onnx(
            wrapper, dummy_inputs, args.output,
            opset_version=args.opset, device=args.device)

    print('\nDone!')


def load_real_inputs_from_dirs(ann_file, sample_idx, io_dir_2d, io_dir_pts,
                                cfg, device='cpu', no_nms=False):
    """
    从预计算的 2D/3D IO 目录加载真实输入 (单帧模式)。

    Args:
        ann_file: pkl 文件路径
        sample_idx: 帧索引
        io_dir_2d: 2D ONNX 输出目录 (含 feat_flatten_img.bin, dyn_query.bin 等)
        io_dir_pts: 3D ONNX 输出目录 (含 pts_feat.bin, pts_query_center.bin 等)
        cfg: mmcv Config
        device: 目标设备
        no_nms: 是否跳过 NMS
    """
    import pickle
    from pyquaternion import Quaternion

    B = 1
    head_cfg = cfg.model.fusion_bbox_head
    C = head_cfg.get('in_channels', 256)
    memory_len = head_cfg.get('memory_len', 1536)
    num_propagated = head_cfg.get('num_propagated', 256)

    # 加载 2D ONNX 输出
    io_2d_out = os.path.join(io_dir_2d, 'outputs')
    feat_flatten_img = torch.from_numpy(
        np.fromfile(os.path.join(io_2d_out, 'feat_flatten_img.bin'), dtype=np.float32)
    )
    # 推断 shape: [N, L_img, C]
    fpn_shapes, _ = _get_fpn_spatial_shapes(cfg)
    L_img = sum(h * w for h, w in fpn_shapes)
    decoder_layer_cfg = head_cfg.transformer.decoder.transformerlayers
    cross_attn_cfg = decoder_layer_cfg.attn_cfgs[1]
    N = cross_attn_cfg.get('num_cams', 7)
    feat_flatten_img = feat_flatten_img.reshape(N, L_img, C)

    lidar2img = torch.from_numpy(
        np.fromfile(os.path.join(io_2d_out, 'lidar2img.bin'), dtype=np.float32)
    ).reshape(B, N, 4, 4)

    dyn_query = torch.from_numpy(
        np.fromfile(os.path.join(io_2d_out, 'dyn_query.bin'), dtype=np.float32)
    )
    prob_bin = head_cfg.get('prob_bin', 25)
    N_img_q = cfg.model.get('max_img_query_per_sample', 200)
    dyn_query = dyn_query.reshape(B, N_img_q, prob_bin, 4)

    query_feats = torch.from_numpy(
        np.fromfile(os.path.join(io_2d_out, 'query_feats.bin'), dtype=np.float32)
    ).reshape(B, N_img_q, C)

    # 加载 3D ONNX 输出
    io_pts_out = os.path.join(io_dir_pts, 'outputs')
    # pts_feat (lidar_feat): [B, L_pts, C_pts]
    bev_h = cross_attn_cfg.get('bev_h', 200)
    bev_w = cross_attn_cfg.get('bev_w', 126)
    C_pts = head_cfg.get('pts_in_channels', 512)
    pts_feat = torch.from_numpy(
        np.fromfile(os.path.join(io_pts_out, 'lidar_feat.bin'), dtype=np.float32)
    ).reshape(B, bev_h * bev_w, C_pts)

    # pts query
    N_pts_q = 100  # CenterPoint test_cfg.all_task_max_num
    pts_query_center_file = os.path.join(io_pts_out, 'query_pos.bin')
    pts_query_feat_file = os.path.join(io_pts_out, 'query_feat.bin')
    pts_query_center = torch.from_numpy(
        np.fromfile(pts_query_center_file, dtype=np.float32)
    ).reshape(B, N_pts_q, 3)
    pts_query_feat = torch.from_numpy(
        np.fromfile(pts_query_feat_file, dtype=np.float32)
    ).reshape(B, N_pts_q, C_pts)

    # 加载 pkl 中的帧信息 (ego_pose, timestamp)
    with open(ann_file, 'rb') as f:
        data = pickle.load(f)
    sample = data['infos'][sample_idx]

    # ego_pose
    e2g_r = Quaternion(sample['ego2global_rotation']).rotation_matrix
    e2g_t = np.array(sample['ego2global_translation'])
    ego_pose_np = np.eye(4, dtype=np.float32)
    ego_pose_np[:3, :3] = e2g_r
    ego_pose_np[:3, 3] = e2g_t
    ego_pose_inv_np = np.linalg.inv(ego_pose_np).astype(np.float32)

    ego_pose_inv = torch.from_numpy(ego_pose_inv_np).unsqueeze(0)
    timestamp = torch.tensor([sample['timestamp'] / 1e6], dtype=torch.float32)
    prev_exists = torch.zeros(B, dtype=torch.float32)  # 首帧

    # memory bank (首帧初始化)
    memory_embedding = torch.zeros(B, memory_len, C)
    memory_reference_point = torch.zeros(B, memory_len, 3)
    memory_timestamp = torch.zeros(B, memory_len, 1)
    memory_velo = torch.zeros(B, memory_len, 2)
    memory_egopose = torch.zeros(B, memory_len, 4, 4)
    memory_query_mask = torch.zeros(B, memory_len, 1, dtype=torch.bool)

    if num_propagated > 0:
        memory_egopose[:, :num_propagated] = torch.eye(4).unsqueeze(0).unsqueeze(0)
        memory_query_mask[:, :num_propagated] = True

    inputs = dict(
        feat_flatten_img=feat_flatten_img,
        lidar2img=lidar2img,
        pts_feat=pts_feat,
        pts_query_center=pts_query_center,
        pts_query_feat=pts_query_feat,
        dyn_query=dyn_query,
        dyn_query_feats=query_feats,
        memory_embedding=memory_embedding,
        memory_reference_point=memory_reference_point,
        memory_timestamp=memory_timestamp,
        memory_velo=memory_velo,
        memory_egopose=memory_egopose,
        memory_query_mask=memory_query_mask,
    )

    # ego_pose 必须始终包含以保持 forward() 参数位置对齐
    if no_nms:
        inputs['ego_pose'] = torch.zeros(B, 4, 4)
    else:
        inputs['ego_pose'] = torch.from_numpy(ego_pose_np).unsqueeze(0)
    inputs['ego_pose_inv'] = ego_pose_inv
    inputs['timestamp'] = timestamp
    inputs['prev_exists'] = prev_exists
    inputs['prev_timestamp'] = torch.zeros(B, dtype=torch.float32)  # 首帧无前帧

    # Move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    print(f'\nReal inputs loaded from:')
    print(f'  2D: {io_dir_2d}')
    print(f'  3D: {io_dir_pts}')
    print(f'  pkl: {ann_file} (sample {sample_idx})')
    for k, v in inputs.items():
        print(f'  {k}: {v.shape} ({v.dtype})')

    return inputs


if __name__ == '__main__':
    main()
