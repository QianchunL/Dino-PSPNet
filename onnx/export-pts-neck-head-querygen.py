#!/usr/bin/env python
"""Export SECOND backbone + pts_neck + pts_bbox_head + pts_query_generator as ONE combined ONNX.

Default constants match mv2dfusion-centerpoint-rtdetr_combined_0305 / hellodata 0.2m voxel configs.
For mv2dfusion-centerpoint-rtdetr_hellodata_0320_02voxel_128dim.py (same BEV geometry), pass:

  python spconv2onnx/export-pts-neck-head-querygen.py \\
    --config projects/configs/nusc/mv2dfusion-centerpoint-rtdetr_hellodata_0320_02voxel_128dim.py \\
    --ckpt /path/to/latest.pth

Combined model:
  Inputs:
    - bev_feat  [B, 640, 128, 200]   SparseEncoder dense BEV output (0.2 voxel, out_size_factor=4)
  Outputs:
    - lidar_feat      [1, H*W, 512]
    - query_feat_out  [1, N_q, 512]
    - query_pos       [1, N_q, 3]

Usage:
export CFG=/mnt/volumes/ad-perception-al-sh01/liqianchun338/bevperception/projects/configs/nusc/mv2dfusion-centerpoint-rtdetr_hellodata_0320_02voxel_128dim_debug.py
export CKPT=/mnt/volumes/ad-perception-al-sh01/lhm/onemodel/bevperception/work_dirs/mv2dfusion-centerpoint-rtdetr_combined_0305_lc_onemodel_0312/iter_195200.pth
export OUT=/mnt/volumes/ad-perception-al-sh01/liqianchun338/bevperception-dev-diffdata-lcfusion-0310-gs/onnx_models
python tools/onnx/export-pts-neck-head-querygen.py \
  --config /mnt/volumes/ad-perception-al-sh01/liqianchun338/bevperception/projects/configs/nusc/mv2dfusion-centerpoint-rtdetr_hellodata_0320_02voxel_128dim_debug.py \
  --ckpt /mnt/volumes/ad-perception-al-sh01/lhm/onemodel/bevperception/work_dirs/mv2dfusion-centerpoint-rtdetr_combined_0305_lc_onemodel_0312/iter_195200.pth \
  --output-dir onnx_models \
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import math
import argparse
import collections
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# Default config constants (from rtdetr_combined_0305.py)
# Overridden by --config if provided
# ============================================================
_DEFAULT_CFG = dict(
    VOXEL_SIZE=[0.2, 0.2, 0.2],
    POINT_CLOUD_RANGE=[-60, -51.2, -3.0, 100, 51.2, 5.0],
    SPARSE_SHAPE=[41, 512, 800],
    OUT_SIZE_FACTOR=4,
    VIRTUAL_VOXEL_SIZE=[0.8, 0.8],
    MAX_NUM=500,
    POST_MAX_SIZE=100,
    FINAL_TOP_K=100,
    SCORE_THRESHOLD=0.1,
    POST_CENTER_RANGE=[-65.0, -55.0, -10.0, 105.0, 55.0, 10.0],
    MAXPOOL_KERNEL_SIZE=3,
    TASK_CONFIG=[(2, 0), (3, 2), (3, 5), (1, 8), (1, 9)],
    # SECOND backbone
    SECOND_IN_CHANNELS=640,
    SECOND_OUT_CHANNELS=[128, 256],
    SECOND_LAYER_NUMS=[5, 5],
    SECOND_LAYER_STRIDES=[1, 2],
    # SECONDFPN neck
    NECK_IN_CHANNELS=[128, 256],
    NECK_OUT_CHANNELS=[256, 256],
    NECK_UPSAMPLE_STRIDES=[1, 2],
    # Head
    HEAD_IN_CHANNELS=512,
    # QueryGen
    QUERYGEN_IN_CHANNELS=512,
    QUERYGEN_HIDDEN_CHANNEL=128,
    # 与 MV2DFusion.pts_yaw_transform 对齐；见 mv2dfusion.py _swap_bbox_outs_dim_wl / _transform_proposals_yaw
    PTS_YAW_TRANSFORM=False,
)


def load_config_from_file(config_path):
    """从 config 文件读取所有 pts-neck-head-querygen 相关参数。"""
    from mmcv import Config
    cfg = Config.fromfile(config_path)

    voxel_size = list(cfg.voxel_size)
    point_cloud_range = list(cfg.point_cloud_range)
    sparse_shape = list(cfg.sparse_shape)

    pts_bb_cfg = cfg.model.pts_backbone
    bb_cfg = pts_bb_cfg.pts_backbone  # SECOND
    neck_cfg = pts_bb_cfg.pts_neck  # SECONDFPN
    head_cfg = pts_bb_cfg.pts_bbox_head  # CenterHeadMaxPool

    # out_size_factor from bbox_coder
    out_size_factor = head_cfg.bbox_coder.out_size_factor
    virtual_voxel_size = [voxel_size[0] * out_size_factor,
                          voxel_size[1] * out_size_factor]

    # task_config: [(num_class, label_offset), ...]
    tasks = head_cfg.tasks
    task_config = []
    label_offset = 0
    for t in tasks:
        nc = t['num_class']
        task_config.append((nc, label_offset))
        label_offset += nc

    # query generator
    qg_cfg = cfg.model.get('pts_query_generator', {})
    qg_in_ch = qg_cfg.get('in_channels', sum(neck_cfg.out_channels))
    qg_hidden_ch = qg_cfg.get('hidden_channel', 128)
    if qg_cfg.get('virtual_voxel_size') is not None:
        virtual_voxel_size = list(qg_cfg['virtual_voxel_size'])

    # CenterHead top-K after heatmap maxpool: use CenterPoint pts_backbone.test_cfg.pts
    # (not model.test_cfg.pts, which is fusion NMS and may differ, e.g. 83 vs 100).
    pts_test = pts_bb_cfg.get('test_cfg', {}).get('pts', {})
    post_max_size = int(pts_test.get('post_max_size', 100))

    params = dict(
        VOXEL_SIZE=voxel_size,
        POINT_CLOUD_RANGE=point_cloud_range,
        SPARSE_SHAPE=sparse_shape,
        OUT_SIZE_FACTOR=out_size_factor,
        VIRTUAL_VOXEL_SIZE=virtual_voxel_size,
        MAX_NUM=head_cfg.bbox_coder.get('max_num', 500),
        POST_MAX_SIZE=post_max_size,
        FINAL_TOP_K=post_max_size,
        SCORE_THRESHOLD=head_cfg.bbox_coder.get('score_threshold', 0.1),
        POST_CENTER_RANGE=list(head_cfg.bbox_coder.get('post_center_range',
                                                        [-65, -55, -10, 105, 55, 10])),
        MAXPOOL_KERNEL_SIZE=3,
        TASK_CONFIG=task_config,
        SECOND_IN_CHANNELS=bb_cfg.in_channels,
        SECOND_OUT_CHANNELS=list(bb_cfg.out_channels),
        SECOND_LAYER_NUMS=list(bb_cfg.layer_nums),
        SECOND_LAYER_STRIDES=list(bb_cfg.layer_strides),
        NECK_IN_CHANNELS=list(neck_cfg.in_channels),
        NECK_OUT_CHANNELS=list(neck_cfg.out_channels),
        NECK_UPSAMPLE_STRIDES=list(neck_cfg.upsample_strides),
        HEAD_IN_CHANNELS=head_cfg.in_channels if isinstance(head_cfg.in_channels, int)
                         else sum(neck_cfg.out_channels),
        QUERYGEN_IN_CHANNELS=qg_in_ch,
        QUERYGEN_HIDDEN_CHANNEL=qg_hidden_ch,
        PTS_YAW_TRANSFORM=bool(cfg.model.get('pts_yaw_transform', False)),
    )

    print(f"Config loaded from: {config_path}")
    print(f"  voxel_size={voxel_size}, sparse_shape={sparse_shape}, "
          f"out_size_factor={out_size_factor}")
    print(f"  SECOND: in_ch={bb_cfg.in_channels}, out_ch={list(bb_cfg.out_channels)}")
    print(f"  BEV: H={sparse_shape[1]//out_size_factor}, W={sparse_shape[2]//out_size_factor}")
    print(f"  tasks: {task_config}")
    print(f"  pts_yaw_transform={params['PTS_YAW_TRANSFORM']}")
    return params


def convert_syncbn_to_bn(module):
    for name, child in module.named_children():
        if isinstance(child, nn.SyncBatchNorm):
            bn = nn.BatchNorm2d(child.num_features, eps=child.eps,
                                momentum=child.momentum, affine=child.affine,
                                track_running_stats=child.track_running_stats)
            if child.affine:
                bn.weight.data.copy_(child.weight.data)
                bn.bias.data.copy_(child.bias.data)
            if child.track_running_stats:
                bn.running_mean.copy_(child.running_mean)
                bn.running_var.copy_(child.running_var)
                bn.num_batches_tracked.copy_(child.num_batches_tracked)
            setattr(module, name, bn)
        else:
            convert_syncbn_to_bn(child)
    return module


def load_module_weights(model, ckpt_path, prefix, strict=False):
    device = 'cpu'
    for p in model.parameters():
        device = p.device
        break
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get('state_dict', ckpt)
    new_state = collections.OrderedDict()
    for key, val in state_dict.items():
        if key.startswith(prefix):
            new_state[key[len(prefix):]] = val
    if len(new_state) == 0:
        raise RuntimeError(f"No keys with prefix '{prefix}'")
    missing, unexpected = model.load_state_dict(new_state, strict=strict)
    print(f"  Loaded {len(new_state) - len(unexpected)} weights (prefix='{prefix}')")
    if missing:
        print(f"    Missing ({len(missing)}): {missing[:5]}")
    if unexpected:
        print(f"    Unexpected ({len(unexpected)}): {unexpected[:5]}")
    return model


class CombinedNeckHeadQueryGenWrapper(nn.Module):
    HEAD_KEYS = ['heatmap', 'reg', 'height', 'dim', 'rot', 'vel']

    def __init__(self, second_block0, second_block1, neck, head, query_gen,
                 voxel_size, point_cloud_range, out_size_factor,
                 max_num, post_max_size, final_top_k, maxpool_kernel_size,
                 task_config, norm_bbox=True, pts_yaw_transform=False):
        super().__init__()
        self.second_block0 = second_block0
        self.second_block1 = second_block1
        self.neck = neck
        self.head = head
        self.pre_bev_embed = query_gen.pre_bev_embed
        self.query_embed = query_gen.query_embed
        self.query_pred_embed = query_gen.query_pred_embed
        self.register_buffer('voxel_size', torch.tensor(voxel_size[:2], dtype=torch.float32))
        self.register_buffer('pc_range', torch.tensor(point_cloud_range, dtype=torch.float32))
        self.out_size_factor = out_size_factor
        self.max_num = max_num
        self.post_max_size = post_max_size
        self.final_top_k = final_top_k
        self.maxpool_kernel_size = maxpool_kernel_size
        self.norm_bbox = norm_bbox
        self.task_num_classes = [t[0] for t in task_config]
        self.task_label_offsets = [t[1] for t in task_config]
        self.pts_yaw_transform = pts_yaw_transform

    @staticmethod
    def _neg_pi_half_yaw(yaw):
        """Match MV2DFusion._neg_pi_half_yaw: CenterPoint yaw → MV2D convention."""
        yaw = -yaw - (math.pi / 2)
        return torch.atan2(torch.sin(yaw), torch.cos(yaw))

    @staticmethod
    def _gather_feat(feats, inds):
        dim = feats.size(2)
        inds = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), dim)
        return feats.gather(1, inds)

    def _topk(self, scores, K):
        batch, cat, height, width = scores.size()
        hw = height * width
        topk_scores, topk_inds = torch.topk(scores.reshape(batch, cat, -1), K)
        topk_inds = topk_inds % hw
        topk_ys = torch.div(topk_inds, width, rounding_mode='floor').float()
        topk_xs = (topk_inds % width).float()
        topk_score, topk_ind = torch.topk(topk_scores.reshape(batch, -1), K)
        topk_clses = torch.div(topk_ind, K, rounding_mode='floor')
        topk_inds = self._gather_feat(topk_inds.reshape(batch, -1, 1), topk_ind).reshape(batch, K)
        topk_ys = self._gather_feat(topk_ys.reshape(batch, -1, 1), topk_ind).reshape(batch, K)
        topk_xs = self._gather_feat(topk_xs.reshape(batch, -1, 1), topk_ind).reshape(batch, K)
        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

    def _transpose_and_gather_feat(self, feat, ind):
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.reshape(feat.size(0), -1, feat.size(3))
        return self._gather_feat(feat, ind)

    def decode_single_task(self, heatmap, reg, height, dim, rot, vel,
                           num_classes, label_offset):
        K = self.post_max_size
        heatmap = heatmap.sigmoid()
        pad = (self.maxpool_kernel_size - 1) // 2
        hmax = F.max_pool2d(heatmap, kernel_size=self.maxpool_kernel_size, stride=1, padding=pad)
        heatmap = heatmap * (hmax == heatmap).float()
        scores, inds, clses, ys, xs = self._topk(heatmap, K=K)
        reg_g = self._transpose_and_gather_feat(reg, inds)
        hei_g = self._transpose_and_gather_feat(height, inds)
        dim_g = self._transpose_and_gather_feat(dim, inds)
        rot_sin = self._transpose_and_gather_feat(rot[:, 0:1], inds)
        rot_cos = self._transpose_and_gather_feat(rot[:, 1:2], inds)
        vel_g = self._transpose_and_gather_feat(vel, inds)
        # WLH → LWH on dim branch (same as MV2DFusion._swap_bbox_outs_dim_wl when pts_yaw_transform)
        if self.pts_yaw_transform:
            dim_g = dim_g[:, :, [1, 0, 2]]
        xs = (xs.unsqueeze(-1) + reg_g[:, :, 0:1]) * self.out_size_factor * self.voxel_size[0] + self.pc_range[0]
        ys = (ys.unsqueeze(-1) + reg_g[:, :, 1:2]) * self.out_size_factor * self.voxel_size[1] + self.pc_range[1]
        yaw = torch.atan2(rot_sin, rot_cos)
        if self.norm_bbox:
            dim_g = torch.exp(dim_g)
        boxes = torch.cat([xs, ys, hei_g, dim_g, yaw, vel_g], dim=-1)
        labels = clses.float() + label_offset
        return boxes, scores, labels

    def grid_sample_bev(self, bev_feat, xy):
        x_norm = (xy[:, :, 0] - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0]) * 2 - 1
        y_norm = (xy[:, :, 1] - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1]) * 2 - 1
        grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(1)
        sampled = F.grid_sample(bev_feat, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        return sampled.squeeze(2).permute(0, 2, 1)

    @staticmethod
    def pos2embed(pos, num_pos_feats=128, temperature=10000):
        scale = 2 * math.pi
        pos = pos * scale
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
        dim_t = 2 * (dim_t // 2) / num_pos_feats + 1
        pos_x = pos[..., None] / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
        return pos_x.flatten(-2)

    def forward(self, bev_feat):
        x0 = self.second_block0(bev_feat)
        x1 = self.second_block1(x0)
        neck_feat = self.neck([x0, x1])
        neck_feat = neck_feat[0]
        ret_dicts = self.head.forward_single(neck_feat)

        all_boxes, all_scores, all_labels = [], [], []
        for task_id, d in enumerate(ret_dicts):
            boxes, scores, labels = self.decode_single_task(
                d['heatmap'], d['reg'], d['height'], d['dim'],
                d['rot'], d['vel'],
                num_classes=self.task_num_classes[task_id],
                label_offset=self.task_label_offsets[task_id])
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        all_boxes = torch.cat(all_boxes, dim=1)
        all_scores = torch.cat(all_scores, dim=1)
        all_labels = torch.cat(all_labels, dim=1)

        _, topk_idx = torch.topk(all_scores, self.final_top_k, dim=1)
        all_boxes = torch.gather(all_boxes, 1, topk_idx.unsqueeze(-1).expand(-1, -1, all_boxes.size(-1)))
        all_scores = torch.gather(all_scores, 1, topk_idx)

        # CenterPoint → MV2D yaw on 9-D box tensor index 6 (same as MV2DFusion._transform_proposals_yaw)
        if self.pts_yaw_transform:
            all_boxes[:, :, 6] = self._neg_pi_half_yaw(all_boxes[:, :, 6])

        query_xyz = all_boxes[:, :, :3]
        query_pred = torch.cat([all_boxes[:, :, 3:9], all_scores.unsqueeze(-1)], dim=-1)
        query_feat = self.grid_sample_bev(neck_feat, query_xyz[:, :, :2])

        bev_feat_flat = neck_feat[0].flatten(1).T
        bev_feat_out = self.pre_bev_embed(bev_feat_flat) + bev_feat_flat
        qf = query_feat[0]
        qf_out = self.query_embed(qf) + qf
        pred_pos_enc = self.pos2embed(query_pred[0], 32, temperature=20)
        qf_out = qf_out + self.query_pred_embed(pred_pos_enc)

        return (bev_feat_out.unsqueeze(0), qf_out.unsqueeze(0), query_xyz[:1])


def build_pts_backbone(C):
    from mmdet3d.models.backbones.second import SECOND
    return SECOND(in_channels=C['SECOND_IN_CHANNELS'],
                  out_channels=C['SECOND_OUT_CHANNELS'],
                  layer_nums=C['SECOND_LAYER_NUMS'],
                  layer_strides=C['SECOND_LAYER_STRIDES'],
                  norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                  conv_cfg=dict(type='Conv2d', bias=False))


def build_pts_neck(C):
    from mmdet3d.models.necks.second_fpn import SECONDFPN
    return SECONDFPN(in_channels=C['NECK_IN_CHANNELS'],
                     out_channels=C['NECK_OUT_CHANNELS'],
                     upsample_strides=C['NECK_UPSAMPLE_STRIDES'],
                     norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                     upsample_cfg=dict(type='deconv', bias=False),
                     conv_cfg=dict(type='Conv2d', bias=False),
                     use_conv_for_no_stride=True)


def build_pts_bbox_head(C):
    import projects.mmdet3d_plugin  # noqa: F401
    from mmdet3d.models.builder import build_head
    # Reconstruct tasks from TASK_CONFIG
    class_names_all = ['car', 'van', 'truck', 'other_vehicle', 'bus',
                       'Cyclist_has', 'Cyclist_non', 'tricycle',
                       'pedestrian', 'trafficcone']
    tasks = []
    for num_cls, offset in C['TASK_CONFIG']:
        names = class_names_all[offset:offset + num_cls]
        tasks.append(dict(num_class=num_cls, class_names=names))
    return build_head(dict(
        type='CenterHeadMaxPool', in_channels=C['HEAD_IN_CHANNELS'],
        tasks=tasks,
        common_heads=dict(reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(type='CenterPointBBoxCoder',
                        post_center_range=C['POST_CENTER_RANGE'],
                        max_num=C['MAX_NUM'],
                        score_threshold=C['SCORE_THRESHOLD'],
                        out_size_factor=C['OUT_SIZE_FACTOR'],
                        voxel_size=C['VOXEL_SIZE'][:2],
                        code_size=9,
                        pc_range=C['POINT_CLOUD_RANGE'][:2]),
        separate_head=dict(type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True))


def build_pts_query_generator(C):
    import projects.mmdet3d_plugin  # noqa: F401
    from projects.mmdet3d_plugin.models.builder import build_query_generator
    return build_query_generator(dict(
        type='PointCloudQueryGenerator',
        in_channels=C['QUERYGEN_IN_CHANNELS'],
        hidden_channel=C['QUERYGEN_HIDDEN_CHANNEL'],
        pts_use_cat=False, dataset='nuscenes',
        virtual_voxel_size=C['VIRTUAL_VOXEL_SIZE'],
        point_cloud_range=C['POINT_CLOUD_RANGE'],
        head_pc_range=C['POINT_CLOUD_RANGE'],
        input_is_bev=True))


def export_onnx(model, dummy_inputs, output_path, input_names, output_names, opset_version=16):
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    torch.onnx.export(model, dummy_inputs, output_path, input_names=input_names,
                      output_names=output_names, opset_version=opset_version,
                      do_constant_folding=True)
    import onnx
    onnx.checker.check_model(onnx.load(output_path))
    print(f"  Saved: {output_path} ({os.path.getsize(output_path) / 1024 / 1024:.1f} MB)")


def _sanitize_token_for_dir(token):
    s = str(token).strip()
    for c in ('/', '\\', '\0', ':', '*', '?', '"', '<', '>', '|'):
        s = s.replace(c, '_')
    return s or 'unknown_token'


def _subdir_name_for_token(token, frame_idx, dup_counts):
    """目录名优先为 token；无 token 用 frame_{idx}；重复则 token_1, token_2 …"""
    if token is None or (isinstance(token, str) and not token.strip()):
        base = f'frame_{frame_idx:06d}'
    else:
        base = _sanitize_token_for_dir(token)
    if base not in dup_counts:
        dup_counts[base] = 0
        return base
    dup_counts[base] += 1
    return f'{base}_{dup_counts[base]}'


def save_tensors_as_bin(tensors_dict, save_dir):
    """Save tensor dict as .bin files."""
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
        shapes_lines.append(f'{name}: shape={list(arr.shape)}, dtype={arr.dtype}')
    info_path = os.path.join(save_dir, 'shapes.txt')
    with open(info_path, 'w') as f:
        f.write('\n'.join(shapes_lines) + '\n')


def run_all_frames_save_io(wrapper, ckpt_path, ann_file, save_io_dir,
                           config_path=None, workers_per_gpu=4, max_frames=None):
    """
    对 pkl 中所有帧运行完整模型，通过 hook 捕获 dense_bev，
    再送入 wrapper 推理，保存每帧的输入/输出。

    流程:
      full model forward → hook 捕获 pts_backbone.pts_backbone 的输入 (dense_bev)
      dense_bev → CombinedNeckHeadQueryGenWrapper → lidar_feat, query_feat, query_pos

    注意: 每帧会跑完整 MV2DFusion（含 7 路相机 load + PIL resize），在 NFS 上较慢；
    提高 ``workers_per_gpu`` 可并行解码图像，通常比 0 更快。
    """
    import copy
    import pickle
    from mmcv import Config
    from mmcv.runner import load_checkpoint
    from mmdet.models import build_detector
    from mmdet3d.datasets import build_dataset
    from projects.mmdet3d_plugin.datasets.builder import build_dataloader
    from mmcv.parallel import MMDataParallel

    # 1. Build full model from config (must match checkpoint architecture)
    if config_path is None:
        config_path = ('projects/configs/nusc/'
                       'mv2dfusion-centerpoint-rtdetr_combined_0305.py')
    print(f'\n[all-frames IO] Loading config: {config_path}')
    cfg = Config.fromfile(config_path)

    # Load plugin
    if hasattr(cfg, 'plugin') and cfg.plugin:
        import importlib
        if hasattr(cfg, 'plugin_dir'):
            plugin_dirs = cfg.plugin_dir if isinstance(cfg.plugin_dir, list) else [cfg.plugin_dir]
            for plugin_dir in plugin_dirs:
                _module_dir = os.path.dirname(plugin_dir).split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                importlib.import_module(_module_path)

    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    model.eval()
    print(f'[all-frames IO] Loading checkpoint: {ckpt_path}')
    load_checkpoint(model, ckpt_path, map_location='cpu', strict=False)

    # 2. Build dataset
    test_cfg = copy.deepcopy(cfg.data.test)
    test_cfg['ann_file'] = ann_file
    test_cfg['test_mode'] = True
    dataset = build_dataset(test_cfg)
    num_frames = len(dataset)

    data_loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=workers_per_gpu,
        dist=False, shuffle=False,
        nonshuffler_sampler=cfg.data.get('nonshuffler_sampler',
                                         dict(type='DistributedSampler')),
    )

    n_export = num_frames if max_frames is None else min(num_frames, max_frames)
    print(f'[all-frames IO] Total frames: {num_frames}, will export: {n_export}')
    print(f'[all-frames IO] DataLoader workers: {workers_per_gpu}')
    print(f'[all-frames IO] Output dir: {save_io_dir}')

    # 3. Hook: 捕获 SECOND backbone 的输入 = SparseEncoder 的 dense_bev 输出
    #    model.pts_backbone (CenterPoint) → pts_backbone (SECOND)
    #    SECOND 的输入 x 就是 dense_bev [1, 640, H, W]
    captured = {}

    def bev_pre_hook(module, args):
        # SECOND.forward(x) → x is dense_bev
        captured['dense_bev'] = args[0].detach().clone()

    second_backbone = model.pts_backbone.pts_backbone  # SECOND module
    hook_handle = second_backbone.register_forward_pre_hook(bev_pre_hook)

    # 4. Wrap model and run
    model = model.cuda()
    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    device = next(wrapper.parameters()).device
    wrapper = wrapper.cuda().eval()

    output_names = ['lidar_feat', 'query_feat_out', 'query_pos']
    os.makedirs(save_io_dir, exist_ok=True)

    data_infos = getattr(dataset, 'data_infos', None)
    dup_counts = {}
    manifest = []

    for frame_idx, data_dict in enumerate(data_loader):
        if max_frames is not None and frame_idx >= max_frames:
            break
        tok = None
        if data_infos is not None and frame_idx < len(data_infos):
            tok = data_infos[frame_idx].get('token')
        sub_name = _subdir_name_for_token(tok, frame_idx, dup_counts)

        with torch.no_grad():
            # 运行完整模型 → hook 自动捕获 dense_bev
            result = model(return_loss=False, rescale=True, **data_dict)

        dense_bev = captured['dense_bev']  # [1, 640, H, W]

        # 通过 wrapper 推理 (与 ONNX 部署一致)
        with torch.no_grad():
            outputs = wrapper(dense_bev.cuda())

        # 保存输入/输出（目录名 = token，与 image / sparse_encoder 导出一致）
        frame_dir = os.path.join(save_io_dir, sub_name)
        save_tensors_as_bin({'bev_feat': dense_bev},
                            os.path.join(frame_dir, 'inputs'))
        output_dict = {name: out for name, out in zip(output_names, outputs)}
        save_tensors_as_bin(output_dict, os.path.join(frame_dir, 'outputs'))

        manifest.append({
            'frame': frame_idx,
            'subdir': sub_name,
            'token': tok,
        })

        if frame_idx % 10 == 0 or frame_idx == n_export - 1:
            print(f'  Frame {frame_idx + 1}/{n_export} ({sub_name}): done')

    hook_handle.remove()
    man_path = os.path.join(save_io_dir, 'manifest.json')
    with open(man_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f'\n[all-frames IO] Saved {len(manifest)} frames under {save_io_dir}/<token>/')
    print(f'  manifest: {man_path}')
    print(f'  Each frame: inputs/bev_feat.bin, outputs/{{lidar_feat,query_feat_out,query_pos}}.bin')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None,
                        help="Config file path (读取所有参数，替代硬编码默认值)")
    parser.add_argument("--ckpt", type=str,
                        default="/mnt/volumes/ad-perception-al-sh01/cm/mv2d/work_dirs/"
                                "mv2dfusion-centerpoint-rtdetr_combined_0305/"
                                "iter_1302.pth")
    parser.add_argument("--output-dir", type=str, default="work_dirs/mv2dfusion-centerpoint-rtdetr_combined_0305/")
    parser.add_argument("--prefix", type=str, default="mv2dfusion_rtdetr_combined_0305")
    parser.add_argument("--opset", type=int, default=16)
    # ---- 精度对齐: IO 保存 ----
    parser.add_argument("--bev-input", type=str, default=None,
                        help="单帧 dense_bev .bin 文件路径 (替代 dummy 随机输入)")
    parser.add_argument("--save-io", type=str, default=None,
                        help="保存输入/输出为 .bin 文件的目录")
    parser.add_argument("--ann-file", type=str, default=None,
                        help="pkl 文件路径，--sample-idx -1 时保存所有帧 IO")
    parser.add_argument("--sample-idx", type=int, default=-1,
                        help="pkl 中 sample 索引 (-1 = 所有帧)")
    parser.add_argument(
        "--io-workers",
        type=int,
        default=4,
        help="全帧导出时 DataLoader 的 workers_per_gpu（图像解码并行；0=主进程串行，NFS 上可试 4~8）",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="全帧导出时最多处理前 N 帧（调试）",
    )
    args = parser.parse_args()

    # ---- 从 config 读取参数，或使用默认值 ----
    if args.config:
        C = load_config_from_file(args.config)
    else:
        C = dict(_DEFAULT_CFG)

    BEV_H = C['SPARSE_SHAPE'][1] // C['OUT_SIZE_FACTOR']
    BEV_W = C['SPARSE_SHAPE'][2] // C['OUT_SIZE_FACTOR']
    BEV_C = C['SECOND_IN_CHANNELS']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}, BEV: {BEV_C}x{BEV_H}x{BEV_W}, Queries: {C['FINAL_TOP_K']}")

    print("\n[0/3] pts_backbone (SECOND)")
    backbone = build_pts_backbone(C)
    backbone = load_module_weights(backbone, args.ckpt, prefix="pts_backbone.pts_backbone.")
    backbone = convert_syncbn_to_bn(backbone).eval()

    print("\n[1/3] pts_neck (SECONDFPN)")
    neck = build_pts_neck(C)
    neck = load_module_weights(neck, args.ckpt, prefix="pts_backbone.pts_neck.")
    neck = convert_syncbn_to_bn(neck).eval()

    print("\n[2/3] pts_bbox_head (CenterHeadMaxPool)")
    head = build_pts_bbox_head(C)
    head = load_module_weights(head, args.ckpt, prefix="pts_backbone.pts_bbox_head.").eval()

    print("\n[3/3] pts_query_generator")
    gen = build_pts_query_generator(C)
    gen = load_module_weights(gen, args.ckpt, prefix="pts_query_generator.").eval()

    wrapper = CombinedNeckHeadQueryGenWrapper(
        backbone.blocks[0], backbone.blocks[1], neck, head, gen,
        C['VOXEL_SIZE'], C['POINT_CLOUD_RANGE'], C['OUT_SIZE_FACTOR'],
        C['MAX_NUM'], C['POST_MAX_SIZE'], C['FINAL_TOP_K'],
        C['MAXPOOL_KERNEL_SIZE'], C['TASK_CONFIG'],
        pts_yaw_transform=C.get('PTS_YAW_TRANSFORM', False)).to(device).eval()

    # ================================================================
    # 全帧 IO 保存模式: --ann-file + --save-io
    # 运行完整模型获取每帧 dense_bev，再通过 wrapper 推理并保存
    # ================================================================
    if args.ann_file and args.save_io and args.sample_idx == -1:
        run_all_frames_save_io(
            wrapper, args.ckpt, args.ann_file, args.save_io,
            config_path=args.config,
            workers_per_gpu=args.io_workers,
            max_frames=args.max_frames)
        print("\nDone!")
        return

    # ================================================================
    # 原有模式: 单帧验证 / ONNX 导出
    # ================================================================
    # 构建输入
    if args.bev_input:
        print(f"\nLoading BEV from: {args.bev_input}")
        bev_np = np.fromfile(args.bev_input, dtype=np.float32)
        bev_feat = torch.from_numpy(bev_np).reshape(1, BEV_C, BEV_H, BEV_W).to(device)
        print(f"  bev_feat: {list(bev_feat.shape)}")
    else:
        bev_feat = torch.randn(1, BEV_C, BEV_H, BEV_W, device=device)

    print("\nForward test...")
    with torch.no_grad():
        lf, qf, qp = wrapper(bev_feat)
    print(f"  lidar_feat: {list(lf.shape)}, query_feat: {list(qf.shape)}, query_pos: {list(qp.shape)}")

    # 保存单帧 IO
    if args.save_io:
        output_names = ['lidar_feat', 'query_feat_out', 'query_pos']
        if args.ann_file and args.sample_idx >= 0:
            import pickle as _pkl
            with open(args.ann_file, 'rb') as _f:
                _infos = _pkl.load(_f)['infos']
            _si = args.sample_idx
            _tok = _infos[_si].get('token')
            frame_dir = os.path.join(args.save_io, _sanitize_token_for_dir(_tok))
        else:
            frame_dir = args.save_io
        print(f"\nSaving IO to {frame_dir}/:")
        save_tensors_as_bin({'bev_feat': bev_feat},
                            os.path.join(frame_dir, 'inputs'))
        save_tensors_as_bin({
            'lidar_feat': lf, 'query_feat_out': qf, 'query_pos': qp
        }, os.path.join(frame_dir, 'outputs'))

    # 导出 ONNX
    output_path = os.path.join(args.output_dir, f"{args.prefix}.second_neck_head_querygen.onnx")
    export_onnx(wrapper, (bev_feat,), output_path,
                input_names=['bev_feat'], output_names=['lidar_feat', 'query_feat_out', 'query_pos'],
                opset_version=args.opset)
    print(f"\nDone! Input: [1,{BEV_C},{BEV_H},{BEV_W}] -> lidar_feat, query_feat, query_pos")


if __name__ == "__main__":
    main()
