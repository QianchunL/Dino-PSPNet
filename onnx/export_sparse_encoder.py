# Export SparseEncoder (pts_middle_encoder) from MV2DFusion to ONNX
#
# Reference: export-scn-hr-selfmodel-with-corners-clip.py
#
# Config: mv2dfusion-centerpoint-rtdetr_combined_0305.py
# Model: pts_backbone -> pts_middle_encoder (SparseEncoder, basicblock)
#
# Output:
#   *.scn.onnx — SparseEncoder (sparse conv, custom exptool tracing)
#     input:  voxels [N, 5]  (sparse voxel features, fp16)
#     output: dense BEV [B, 640, 128, 200]
#
# Usage:
#   cd mv2d/spconv2onnx
#   python export-mv2dfusion-sparse-encoder.py \
#       --ckpt ...pth --config ...py --ann-file ...pkl --sample-idx -1 \
#       --save-onnx out.scn.onnx --save-io ./dump --save-io-all-frames \
#       --verify-onnx --no-normalize-intensity
"""
PYTHONPATH=/mnt/volumes/ad-perception-al-sh01/zhouli978/gitlab/mv2d/spconv2onnx/ python tools/onnx/export_sparse_encoder.py \
  --config /mnt/volumes/ad-perception-al-sh01/liqianchun338/bevperception/projects/configs/nusc/mv2dfusion-centerpoint-rtdetr_hellodata_0320_02voxel_128dim_debug.py \
  --ckpt /mnt/volumes/ad-perception-al-sh01/lhm/onemodel/bevperception/work_dirs/mv2dfusion-centerpoint-rtdetr_combined_0305_lc_onemodel_0312/iter_195200.pth \
  --save-onnx onnx_models/sparse_encoder_0312.scn.onnx \
  --ann-file /mnt/volumes/ad-perception-al-sh01/zhouli978/gitlab/mv2d/dataset/20260309/hr_infos_1_sweeps_train_1230_total_all_truck_merge.pkl \
  --save-io debug/sparse_encoder_io_all_0312 \
  --save-io-all-frames
"""
import sys; sys.path.insert(0, ".")

import os
import torch
import torch.nn as nn
import pickle
import argparse
import collections
import json
import numpy as np
import types

# Mock missing modules so that exptool / funcs can be imported
# (det3d and tools.sparseconv_quantization are not available in this env)
def _create_mock_modules():
    """Create mock modules to satisfy import dependencies of exptool/funcs."""
    # Mock det3d.models.backbones.scn
    det3d = types.ModuleType("det3d")
    det3d_models = types.ModuleType("det3d.models")
    det3d_backbones = types.ModuleType("det3d.models.backbones")
    det3d_scn = types.ModuleType("det3d.models.backbones.scn")
    det3d_scn.SpMiddleResNetFHD = type("SpMiddleResNetFHD", (), {})
    det3d_scn.SparseBasicBlock = type("SparseBasicBlock", (), {})
    det3d.models = det3d_models
    det3d_models.backbones = det3d_backbones
    det3d_backbones.scn = det3d_scn
    for m in [det3d, det3d_models, det3d_backbones, det3d_scn]:
        sys.modules[m.__name__] = m

    # Mock tools.sparseconv_quantization
    tools = types.ModuleType("tools")
    tools_sq = types.ModuleType("tools.sparseconv_quantization")
    tools_sq.QuantAdd = type("QuantAdd", (), {"forward": lambda *a, **k: None})
    tools_sq.SparseConvolutionQunat = type("SparseConvolutionQunat", (), {"forward": lambda *a, **k: None})
    tools_sq.initialize = lambda: None
    tools_sq.disable_quantization = lambda m: type("_", (), {"apply": lambda s: None})()
    tools_sq.quant_sparseconv_module = lambda m: None
    tools_sq.quant_add_module = lambda m: None
    tools.sparseconv_quantization = tools_sq
    sys.modules["tools"] = tools
    sys.modules["tools.sparseconv_quantization"] = tools_sq

_create_mock_modules()

import cumm.tensorview as tv
from spconv.pytorch import SparseSequential
from spconv.pytorch import conv as spconv_conv

import exptool


# ============================================================
# BN fusion utilities (adapted from funcs.py for mmdet3d SparseEncoder)
# ============================================================

def fuse_bn_weights(conv_w_OKI, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b):
    """Fuse BatchNorm weights into sparse convolution weights."""
    NDim = conv_w_OKI.ndim - 2
    permute = [0, NDim + 1] + [i + 1 for i in range(NDim)]
    conv_w_OIK = conv_w_OKI.permute(*permute)
    if conv_b is None:
        conv_b = torch.zeros_like(bn_rm)
    if bn_w is None:
        bn_w = torch.ones_like(bn_rm)
    if bn_b is None:
        bn_b = torch.zeros_like(bn_rm)
    bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)
    conv_w_OIK = conv_w_OIK * (bn_w * bn_var_rsqrt).reshape([-1] + [1] * (len(conv_w_OIK.shape) - 1))
    conv_b = (conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b
    permute = [0, ] + [i + 2 for i in range(NDim)] + [1, ]
    conv_w_OKI = conv_w_OIK.permute(*permute).contiguous()
    return torch.nn.Parameter(conv_w_OKI), torch.nn.Parameter(conv_b)


def fuse_bn(conv_layer, bn_layer):
    """Fuse BN into sparse conv."""
    assert not (conv_layer.training or bn_layer.training), "Fusion only for eval!"
    conv_layer.weight, conv_layer.bias = fuse_bn_weights(
        conv_layer.weight, conv_layer.bias,
        bn_layer.running_mean, bn_layer.running_var,
        bn_layer.eps, bn_layer.weight, bn_layer.bias
    )


def fuse_sparse_conv_module(seq):
    """
    Fuse a SparseSequential(SparseConv, BN, ReLU) into a single SparseConv with act_type=ReLU.
    Returns the fused SparseConv.
    """
    # seq is SparseSequential: [0]=conv, [1]=BN, [2]=ReLU
    c = seq[0]
    b = seq[1]
    fuse_bn(c, b)
    c.act_type = tv.gemm.Activation.ReLU
    return c


def new_sparse_basic_block_forward(block):
    """Create a new forward function for SparseBasicBlock after BN fusion.
    
    NOTE: conv1 already has act_type=ReLU fused, so we do NOT apply explicit
    ReLU after conv1 (matching the reference funcs.py behavior with is_fuse_relu=True).
    Only the post-residual ReLU is kept as an explicit operation for tracing.
    """
    def forward(x):
        identity = x.features
        out = block.conv1(x)
        # conv1.act_type=ReLU handles ReLU internally, no explicit ReLU needed here
        out = block.conv2(out)
        if block.downsample is not None:
            identity = block.downsample(x)
        out = out.replace_feature(out.features + identity)
        out = out.replace_feature(block.relu(out.features))
        return out
    return forward


def fuse_sparse_basic_block(block):
    """
    Fuse BN+ReLU in a mmdet3d SparseBasicBlock.
    - conv1: fuse bn1, set act_type=ReLU
    - conv2: fuse bn2 (no ReLU, that's the post-residual ReLU)
    
    IMPORTANT: mmdet3d's SparseBasicBlock uses nn.ReLU(inplace=True).
    In-place ReLU breaks exptool tracing because the output tensor has the
    same id() as the input, causing register_tensor() to overwrite the input's
    tensor ID. This creates disconnected nodes in the ONNX graph.
    We replace it with nn.ReLU(inplace=False) to fix this.
    """
    fuse_bn(block.conv1, block.bn1)
    block.conv1.act_type = tv.gemm.Activation.ReLU
    fuse_bn(block.conv2, block.bn2)
    # Remove bn1, bn2 (no longer needed)
    delattr(block, "bn1")
    delattr(block, "bn2")
    # CRITICAL: Replace inplace ReLU with non-inplace to fix ONNX tracing
    block.relu = torch.nn.ReLU(inplace=False)
    # Replace forward
    block.forward = new_sparse_basic_block_forward(block)


def layer_fusion_bn_relu(model):
    """
    Fuse all BN+ReLU in SparseEncoder model.
    Handles: conv_input, encoder_layers (SparseBasicBlock + downsample convs), conv_out
    """
    from mmdet3d.ops.sparse_block import SparseBasicBlock

    # 1. Fuse conv_input: SparseSequential(SubMConv3d, BN1d, ReLU)
    model.conv_input = fuse_sparse_conv_module(model.conv_input)

    # 2. Fuse conv_out: SparseSequential(SparseConv3d, BN1d, ReLU)
    model.conv_out = fuse_sparse_conv_module(model.conv_out)

    # 3. Fuse encoder_layers
    for stage_name, stage_module in model.encoder_layers.named_children():
        for block_name, block in stage_module.named_children():
            if isinstance(block, SparseBasicBlock):
                fuse_sparse_basic_block(block)
            elif isinstance(block, SparseSequential):
                # Downsample conv: SparseSequential(SparseConv3d, BN1d, ReLU)
                fused = fuse_sparse_conv_module(block)
                # Replace the SparseSequential with single fused conv
                setattr(stage_module, block_name, fused)

    return model


# ============================================================
# Weight loading
# ============================================================

def load_sparse_encoder_weights(model, ckpt_path):
    """Load pts_middle_encoder weights from MV2DFusion checkpoint.
    
    Handles spconv weight format conversion:
    - spconv 1.x: (I, O, K, K, K)  or (K, K, K, I, O)
    - spconv 2.x: (O, K, K, K, I)
    """
    device = next(model.parameters()).device
    ckpt = torch.load(ckpt_path, map_location=device)

    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt

    prefix = "pts_backbone.pts_middle_encoder."
    new_state = collections.OrderedDict()
    for key, val in state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]
            new_state[new_key] = val

    if len(new_state) == 0:
        raise RuntimeError(f"No keys found with prefix '{prefix}' in checkpoint. "
                           f"Available prefixes: {set(k.split('.')[0] for k in state_dict.keys())}")

    # Manually load weights with spconv format conversion
    model_state = model.state_dict()
    loaded = 0
    converted = 0
    skipped = []
    for key, model_param in model_state.items():
        if key not in new_state:
            skipped.append(key)
            continue
        ckpt_param = new_state[key]
        if ckpt_param.dim() == 5 and model_param.dim() == 5 and ckpt_param.shape != model_param.shape:
            # Try permutation: (I, O, K, K, K) -> (O, K, K, K, I)
            converted_param = ckpt_param.permute(1, 2, 3, 4, 0).contiguous()
            if converted_param.shape == model_param.shape:
                model_param.copy_(converted_param)
                loaded += 1
                converted += 1
            else:
                # Try permutation: (K, K, K, I, O) -> (O, K, K, K, I)
                converted_param = ckpt_param.permute(4, 0, 1, 2, 3).contiguous()
                if converted_param.shape == model_param.shape:
                    model_param.copy_(converted_param)
                    loaded += 1
                    converted += 1
                else:
                    print(f"  ⚠️  Cannot convert {key}: ckpt {ckpt_param.shape} -> model {model_param.shape}")
                    skipped.append(key)
        elif ckpt_param.shape == model_param.shape:
            model_param.copy_(ckpt_param)
            loaded += 1
        else:
            print(f"  ⚠️  Shape mismatch {key}: ckpt {ckpt_param.shape} vs model {model_param.shape}")
            skipped.append(key)

    print(f"✓ Loaded {loaded}/{len(model_state)} weights for SparseEncoder ({converted} spconv weights converted)")
    if skipped:
        print(f"  Skipped keys ({len(skipped)}): {skipped[:5]}...")

    return model


# ============================================================
# Point cloud preprocessing (match PyTorch test_pipeline + voxelize)
# ============================================================
# Align with:
#   - PointsRangeFilter: mmdet3d BasePoints.in_range_3d (strict > min, strict < max)
#   - NormalizePoints: intensity channel /= 255 (see tools/infer_centerpoint_from_bin.py)
#   - AV2LoadPointsFromFile._load_points: .bin 4 or 5 floats per point


def _load_bin_points_av2_style(pts_path):
    """Load .bin as AV2LoadPointsFromFile: 5 cols preferred, else 4; else truncate to 4n."""
    points = np.fromfile(str(pts_path), dtype=np.float32)
    if len(points) == 0:
        raise ValueError(f"Empty bin: {pts_path}")
    if len(points) < 4:
        raise ValueError(f"Bin too small: {pts_path}")
    if len(points) % 5 == 0:
        points = points.reshape(-1, 5)
        lidar = points[:, :4].astype(np.float32)
    elif len(points) % 4 == 0:
        lidar = points.reshape(-1, 4).astype(np.float32)
    else:
        n = len(points) // 4
        lidar = points[: n * 4].reshape(-1, 4).astype(np.float32)
    return lidar


def _points_in_range_3d_mmdet3d(points_xyz, point_cloud_range):
    """Same as mmdet3d.core.points.BasePoints.in_range_3d (SECOND-style open bounds)."""
    pc = np.asarray(point_cloud_range, dtype=np.float32)
    x, y, z = points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2]
    return ((x > pc[0]) & (y > pc[1]) & (z > pc[2]) &
            (x < pc[3]) & (y < pc[4]) & (z < pc[5]))


def _normalize_points_intensity(points_np):
    """Match test_pipeline NormalizePoints: divide intensity (4th channel) by 255."""
    out = points_np.copy()
    out[:, 3] = out[:, 3] / 255.0
    return out


# ============================================================
# Main
# ============================================================

def _resolve_sample_idx(sample_idx, num_infos):
    """Support negative index (e.g. -1 = last frame)."""
    if sample_idx < 0:
        return num_infos + sample_idx
    return sample_idx


def load_real_voxels_from_pkl(pkl_path, sample_idx=0, voxel_size=None, point_cloud_range=None,
                              sparse_shape=None, max_num_points=20, max_voxels=200000,
                              normalize_intensity=True, infos=None, voxelizer=None,
                              quiet=False):
    """从 pkl 文件中加载点云并做体素化, 模拟 HardSimpleVFE 输出。

    预处理顺序与 config 中 test_pipeline 一致:
      PointsRangeFilter (in_range_3d) → NormalizePoints (intensity/255) → Voxelization

    HardSimpleVFE: 对每个 voxel 内的点取均值 → [N_voxels, num_features]

    Args:
        normalize_intensity: 与 MV2DFusion hellodata config 中 ``NormalizePoints`` 一致；
            若部署侧不做 /255，可传 False 做对比。
        infos: 若已 ``pickle.load`` 过 ``data['infos']``，传入可避免重复读盘。
        voxelizer: 复用 ``mmcv.ops.Voxelization``（多帧导出时显著加速）。
        quiet: 为 True 时少打印日志（多帧循环用）。

    Returns:
        voxels: [N_voxels, 4] half tensor (CUDA)
        coors:  [N_voxels, 4] int  tensor (CUDA)  (batch_idx, z, y, x)
    """
    from mmcv.ops import Voxelization

    if voxel_size is None:
        voxel_size = [0.2, 0.2, 0.2]
    if point_cloud_range is None:
        point_cloud_range = [-60, -51.2, -3.0, 100, 51.2, 5.0]

    if infos is None:
        if not quiet:
            print(f"Loading pkl: {pkl_path}")
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        infos = data["infos"]

    n = len(infos)
    si = _resolve_sample_idx(sample_idx, n)
    if si < 0 or si >= n:
        raise IndexError(f"sample_idx resolved to {si}, num_infos={n}")

    sample = infos[si]
    lidar_path = sample["lidar_path"]
    if not quiet:
        print(f"  Sample {sample_idx} -> [{si}]: token={sample['token']}, lidar={lidar_path}")

    # 加载点云 (.pcd or .bin) — .bin 与 AV2LoadPointsFromFile 一致
    if lidar_path.endswith(".pcd"):
        # PCD 格式 (HelloData)
        import struct
        with open(lidar_path, "rb") as f:
            header = {}
            while True:
                line = f.readline().decode("ascii", errors="ignore").strip()
                if line.startswith("DATA"):
                    header["data"] = line.split()[-1]
                    break
                if " " in line:
                    key, val = line.split(" ", 1)
                    header[key.lower()] = val
            num_points = int(header.get("points", header.get("width", "0")))
            fields = header.get("fields", "x y z intensity").split()
            sizes = [int(s) for s in header.get("size", "4 4 4 4").split()]
            types = header.get("type", "F F F F").split()
            point_size = sum(sizes)
            raw = f.read(num_points * point_size)
        # 解析为 numpy
        points_list = []
        for i in range(num_points):
            offset = i * point_size
            vals = []
            field_offset = 0
            for j, (sz, tp) in enumerate(zip(sizes, types)):
                if tp == "F" and sz == 4:
                    vals.append(struct.unpack_from("<f", raw, offset + field_offset)[0])
                elif tp == "U" and sz == 1:
                    vals.append(struct.unpack_from("<B", raw, offset + field_offset)[0])
                elif tp == "U" and sz == 4:
                    vals.append(struct.unpack_from("<I", raw, offset + field_offset)[0])
                else:
                    vals.append(0.0)
                field_offset += sz
            points_list.append(vals[:4])  # x, y, z, intensity
        points_np = np.array(points_list, dtype=np.float32)
    else:
        points_np = _load_bin_points_av2_style(lidar_path)

    if not quiet:
        print(f"  Loaded {len(points_np)} points, shape={points_np.shape}")

    # PointsRangeFilter: 与 mmdet3d BasePoints.in_range_3d 一致
    mask = _points_in_range_3d_mmdet3d(points_np[:, :3], point_cloud_range)
    points_np = points_np[mask]
    if not quiet:
        print(f"  After PointsRangeFilter (in_range_3d): {len(points_np)} points")

    # NormalizePoints: intensity /= 255
    if normalize_intensity:
        points_np = _normalize_points_intensity(points_np)
        if not quiet:
            print(f"  After NormalizePoints (intensity/255): range [{points_np[:, 3].min():.4f}, {points_np[:, 3].max():.4f}]")

    # 体素化 (使用 mmdet3d Voxelization)
    if voxelizer is None:
        voxelizer = Voxelization(
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            max_num_points=max_num_points,
            max_voxels=max_voxels,
            deterministic=False,
        )
    points_tensor = torch.from_numpy(points_np).cuda()
    voxels_out, coors_out, num_points_per_voxel = voxelizer(points_tensor)
    # voxels_out: [N, max_num_points, 4]
    # num_points_per_voxel: [N]

    # HardSimpleVFE: 对有效点取均值
    # mask invalid points (padded zeros)
    voxel_features = voxels_out.sum(dim=1) / num_points_per_voxel.unsqueeze(-1).float().clamp(min=1)
    # [N, 4]

    # 添加 batch index
    batch_coors = torch.cat([
        torch.zeros(coors_out.shape[0], 1, dtype=torch.int32, device=coors_out.device),
        coors_out.int()
    ], dim=1)  # [N, 4] (batch_idx, z, y, x)

    if not quiet:
        print(f"  Voxels: {voxel_features.shape}, Coors: {batch_coors.shape}")
    return voxel_features.half(), batch_coors.int()


def save_bin_files(tensors_dict, save_dir):
    """保存 tensor dict 为 .bin 文件。"""
    os.makedirs(save_dir, exist_ok=True)
    for name, tensor in tensors_dict.items():
        if isinstance(tensor, torch.Tensor):
            arr = tensor.detach().cpu().numpy()
        else:
            arr = np.array(tensor)
        path = os.path.join(save_dir, f"{name}.bin")
        arr.tofile(path)
        size_kb = os.path.getsize(path) / 1024
        print(f"  {path} ({size_kb:.1f} KB) shape={list(arr.shape)} dtype={arr.dtype}")

    # shapes.txt
    info_path = os.path.join(save_dir, "shapes.txt")
    with open(info_path, "w") as f:
        for name, tensor in tensors_dict.items():
            if isinstance(tensor, torch.Tensor):
                f.write(f"{name}: shape={list(tensor.shape)}, dtype={tensor.dtype}\n")
            else:
                arr = np.array(tensor)
                f.write(f"{name}: shape={list(arr.shape)}, dtype={arr.dtype}\n")
    print(f"  {info_path}")


def _sanitize_token_for_dir(token):
    """将 sample token 转为可用作目录名的字符串（与 export_rtdetr_image_branch_onnx 一致）。"""
    s = str(token).strip()
    for c in ('/', '\\', '\0', ':', '*', '?', '"', '<', '>', '|'):
        s = s.replace(c, '_')
    return s or 'unknown_token'


def _subdir_name_for_token(token, frame_idx, dup_counts):
    """目录名优先为 token；无 token 时用 frame_{idx:06d}；重复时 token_1, token_2 …"""
    if token is None or (isinstance(token, str) and not token.strip()):
        base = f"frame_{frame_idx:06d}"
    else:
        base = _sanitize_token_for_dir(token)
    if base not in dup_counts:
        dup_counts[base] = 0
        return base
    dup_counts[base] += 1
    return f"{base}_{dup_counts[base]}"


def load_sparse_encoder_config(config_path):
    """从 mmcv Config 中读取 SparseEncoder 相关参数。

    读取路径: model.pts_backbone.pts_middle_encoder.*
    以及顶层的 voxel_size, point_cloud_range, sparse_shape
    """
    # 添加项目根目录到 path，确保 plugin 能被加载
    project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from mmcv import Config
    cfg = Config.fromfile(config_path)

    # 顶层参数
    voxel_size = list(cfg.voxel_size)
    point_cloud_range = list(cfg.point_cloud_range)
    sparse_shape = list(cfg.sparse_shape)

    # SparseEncoder 参数
    me_cfg = cfg.model.pts_backbone.pts_middle_encoder
    in_channels = me_cfg.get('in_channels', 4)
    output_channels = me_cfg.get('output_channels', 128)
    encoder_channels = tuple(tuple(x) for x in me_cfg.encoder_channels)
    # encoder_paddings 可能含 list, 需递归转 tuple
    def _to_tuple(x):
        if isinstance(x, (list, tuple)):
            return tuple(_to_tuple(i) for i in x)
        return x
    encoder_paddings = _to_tuple(me_cfg.encoder_paddings)
    order = tuple(me_cfg.get('order', ('conv', 'norm', 'act')))
    block_type = me_cfg.get('block_type', 'basicblock')

    # max_num_points for voxelization
    voxel_layer_cfg = cfg.model.pts_backbone.get('pts_voxel_layer', {})
    max_num_points = voxel_layer_cfg.get('max_num_points', 20)

    params = dict(
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        sparse_shape=sparse_shape,
        in_channels=in_channels,
        output_channels=output_channels,
        encoder_channels=encoder_channels,
        encoder_paddings=encoder_paddings,
        order=order,
        block_type=block_type,
        max_num_points=max_num_points,
    )
    print(f"Config loaded from: {config_path}")
    for k, v in params.items():
        print(f"  {k}: {v}")
    return params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export SparseEncoder (pts_middle_encoder) to ONNX")
    parser.add_argument("--config", type=str, default=None,
                        help="Config file path (读取 sparse_shape, voxel_size 等参数)")
    parser.add_argument("--in-channel", type=int, default=4,
                        help="SparseEncoder input channels (仅在无 --config 时使用)")
    parser.add_argument("--ckpt", type=str,
                        default="/mnt/volumes/ad-perception-al-sh01/cm/mv2d/work_dirs/mv2dfusion-centerpoint-rtdetr_combined_0305/iter_1302.pth",
                        help="MV2DFusion checkpoint path")
    parser.add_argument("--input", type=str, default=None,
                        help="Input pickle data (voxels/coors format), random if not provided")
    parser.add_argument("--ann-file", type=str, default=None,
                        help="Data info pkl path, load real point cloud and voxelize")
    parser.add_argument("--sample-idx", type=int, default=0,
                        help="Sample index in ann-file pkl")
    parser.add_argument("--save-onnx", type=str,
                        default="../work_dirs/mv2dfusion-centerpoint-rtdetr_combined_0305/mv2dfusion_rtdetr_combined_0305.scn.onnx",
                        help="Output ONNX path")
    parser.add_argument("--save-tensor", type=str, default=None,
                        help="Save input/output tensor for C++ verification (exptool format)")
    parser.add_argument("--save-io", type=str, default=None,
                        help="Save model inputs/outputs as .bin files to this directory")
    parser.add_argument(
        "--save-io-all-frames",
        action="store_true",
        help="With --ann-file and --save-io: per frame under <token>/ (same as image branch; "
             "requires --save-io)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="With --save-io-all-frames: only process first N frames (debug)",
    )
    parser.add_argument("--verify-onnx", action="store_true",
                        help="Verify ONNX output matches PyTorch output")
    parser.add_argument("--no-normalize-intensity", action="store_true",
                        help="Skip intensity/255 (only for debug; inference uses NormalizePoints)")
    args = parser.parse_args()

    if args.save_io_all_frames and not args.save_io:
        parser.error("--save-io-all-frames requires --save-io")
    if args.save_io_all_frames and not args.ann_file:
        parser.error("--save-io-all-frames requires --ann-file")

    # ---- 从 config 读取参数，或使用默认值 ----
    if args.config:
        cfg_params = load_sparse_encoder_config(args.config)
        sparse_shape = cfg_params['sparse_shape']
        voxel_size = cfg_params['voxel_size']
        point_cloud_range = cfg_params['point_cloud_range']
        in_channels = cfg_params['in_channels']
        output_channels = cfg_params['output_channels']
        encoder_channels = cfg_params['encoder_channels']
        encoder_paddings = cfg_params['encoder_paddings']
        order = cfg_params['order']
        block_type = cfg_params['block_type']
    else:
        # 默认值 (mv2dfusion-centerpoint-rtdetr_combined_0305.py)
        sparse_shape = [41, 512, 800]
        voxel_size = [0.2, 0.2, 0.2]
        point_cloud_range = [-60, -51.2, -3.0, 100, 51.2, 5.0]
        in_channels = args.in_channel
        output_channels = 128
        encoder_channels = ((16, 16, 32), (32, 32, 64), (64, 64, 64))
        encoder_paddings = ((0, 0, 1), (0, 0, 1), (0, 0, (0, 1, 1)))
        order = ('conv', 'norm', 'act')
        block_type = 'basicblock'
        cfg_params = {'max_num_points': 20}

    # ---- Build SparseEncoder ----
    from mmdet3d.models.middle_encoders.sparse_encoder import SparseEncoder

    model = SparseEncoder(
        in_channels=in_channels,
        sparse_shape=sparse_shape,
        output_channels=output_channels,
        order=order,
        encoder_channels=encoder_channels,
        encoder_paddings=encoder_paddings,
        block_type=block_type,
    ).cuda().eval().half()

    print("Original model:")
    print(model)

    # ---- Load weights ----
    print(f"\n🔥 Loading weights from {args.ckpt}")
    model = load_sparse_encoder_weights(model, args.ckpt)

    # ---- Fuse BN + ReLU ----
    print("\n🔥 Fusing BN + ReLU into sparse convolutions")
    model = layer_fusion_bn_relu(model)

    print("\nFused model:")
    print(model)

    # ---- Prepare input ----
    torch_output = None
    if args.ann_file:
        max_num_pts = cfg_params['max_num_points']
        norm = not args.no_normalize_intensity

        if args.save_io_all_frames:
            from mmcv.ops import Voxelization

            print(f"\n📦 --save-io-all-frames: {args.ann_file}")
            with open(args.ann_file, "rb") as f:
                _data = pickle.load(f)
            infos = _data["infos"]
            n_total = len(infos)
            n_loop = min(n_total, args.max_frames) if args.max_frames else n_total
            print(f"  frames in pkl: {n_total}, exporting: {n_loop}")

            voxelizer = Voxelization(
                voxel_size=voxel_size,
                point_cloud_range=point_cloud_range,
                max_num_points=max_num_pts,
                max_voxels=200000,
                deterministic=False,
            )
            os.makedirs(args.save_io, exist_ok=True)
            manifest = []
            _dup_subdirs = {}
            for i in range(n_loop):
                sub_name = _subdir_name_for_token(infos[i].get("token"), i, _dup_subdirs)
                try:
                    v_i, c_i = load_real_voxels_from_pkl(
                        args.ann_file,
                        sample_idx=i,
                        voxel_size=voxel_size,
                        point_cloud_range=point_cloud_range,
                        sparse_shape=sparse_shape,
                        max_num_points=max_num_pts,
                        normalize_intensity=norm,
                        infos=infos,
                        voxelizer=voxelizer,
                        quiet=True,
                    )
                    if v_i.shape[0] == 0:
                        manifest.append({
                            "frame": i,
                            "subdir": sub_name,
                            "ok": False,
                            "error": "zero voxels",
                            "token": infos[i].get("token"),
                        })
                    else:
                        with torch.no_grad():
                            out_i = model(v_i, c_i, 1)
                        sub = os.path.join(args.save_io, sub_name)
                        save_bin_files(
                            {"voxels": v_i, "coors": c_i},
                            os.path.join(sub, "inputs"),
                        )
                        save_bin_files(
                            {"dense_bev": out_i},
                            os.path.join(sub, "outputs"),
                        )
                        manifest.append({
                            "frame": i,
                            "subdir": sub_name,
                            "ok": True,
                            "token": infos[i].get("token"),
                            "lidar_path": infos[i].get("lidar_path"),
                            "timestamp": infos[i].get("timestamp"),
                            "num_voxels": int(v_i.shape[0]),
                        })
                except Exception as e:
                    manifest.append({
                        "frame": i,
                        "subdir": sub_name,
                        "ok": False,
                        "error": repr(e),
                        "token": infos[i].get("token") if i < len(infos) else None,
                    })
                if (i + 1) % 50 == 0 or i == n_loop - 1:
                    print(f"  ... {i + 1}/{n_loop} frames")

            man_path = os.path.join(args.save_io, "manifest.json")
            with open(man_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
            print(f"\n✓ Per-frame IO: {args.save_io}/<token>/inputs|outputs/")
            print(f"  manifest: {man_path}")

            # One tensor for ONNX trace / verify: --sample-idx (e.g. -1 = last)
            voxels, coors = load_real_voxels_from_pkl(
                args.ann_file,
                args.sample_idx,
                voxel_size=voxel_size,
                point_cloud_range=point_cloud_range,
                sparse_shape=sparse_shape,
                max_num_points=max_num_pts,
                normalize_intensity=norm,
                infos=infos,
                voxelizer=voxelizer,
                quiet=False,
            )
            batch_size = 1
            spatial_shape = sparse_shape
        else:
            voxels, coors = load_real_voxels_from_pkl(
                args.ann_file,
                args.sample_idx,
                voxel_size=voxel_size,
                point_cloud_range=point_cloud_range,
                sparse_shape=sparse_shape,
                max_num_points=max_num_pts,
                normalize_intensity=norm,
            )
            batch_size = 1
            spatial_shape = sparse_shape
    elif args.input:
        with open(args.input, "rb") as f:
            voxels, coors, spatial_shape, batch_size = pickle.load(f)
            voxels = torch.tensor(voxels).half().cuda()
            coors = torch.tensor(coors).int().cuda()
    else:
        voxels = torch.zeros(1, in_channels).half().cuda()
        coors = torch.zeros(1, 4).int().cuda()
        batch_size = 1
        spatial_shape = sparse_shape

    print(f"\nInput (ONNX trace): voxels={voxels.shape}, coors={coors.shape}")

    # ---- Save single-frame inputs as .bin ----
    if args.save_io and not args.save_io_all_frames:
        print(f"\n📦 Saving inputs to {args.save_io}/inputs/:")
        save_bin_files({
            "voxels": voxels,
            "coors": coors,
        }, os.path.join(args.save_io, "inputs"))

    # ---- PyTorch forward (single-frame save_io or verify) ----
    need_forward = args.verify_onnx or (args.save_io and not args.save_io_all_frames)
    if need_forward:
        print("\n🔥 Running PyTorch forward...")
        with torch.no_grad():
            torch_output = model(voxels, coors, batch_size)
        print(f"  PyTorch output: {torch_output.shape}, dtype={torch_output.dtype}")
        print(f"  range: [{torch_output.min().item():.6f}, {torch_output.max().item():.6f}]")

        if args.save_io and not args.save_io_all_frames:
            print(f"\n📦 Saving outputs to {args.save_io}/outputs/:")
            save_bin_files({
                "dense_bev": torch_output,
            }, os.path.join(args.save_io, "outputs"))

    # ---- Wrap model to match exptool interface ----
    class SparseEncoderWrapper(torch.nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder

        def forward(self, voxels, coors, batch_size, spatial_shape):
            out = self.encoder(voxels, coors, batch_size)
            return [out]

    wrapper = SparseEncoderWrapper(model).cuda().eval().half()

    # ---- Export SparseEncoder ONNX ----
    print("\n" + "=" * 60)
    print("Exporting SparseEncoder ONNX (sparse conv, exptool tracing)")
    print("=" * 60)
    exptool.export_onnx(wrapper, voxels, coors, batch_size, spatial_shape, args.save_onnx, args.save_tensor)

    # ---- Verify ONNX vs PyTorch ----
    if args.verify_onnx:
        print("\n" + "=" * 60)
        print("Verifying ONNX vs PyTorch...")
        print("=" * 60)
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(args.save_onnx,
                                        providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            print(f"  ONNX inputs:")
            for inp in sess.get_inputs():
                print(f"    {inp.name}: {inp.shape} ({inp.type})")
            print(f"  ONNX outputs:")
            for out in sess.get_outputs():
                print(f"    {out.name}: {out.shape} ({out.type})")

            onnx_inputs = {
                sess.get_inputs()[0].name: voxels.cpu().numpy().astype(np.float16),
            }
            ort_outputs = sess.run(None, onnx_inputs)
            onnx_out = ort_outputs[0]

            # 对比
            pt_out = torch_output.float().cpu().numpy()
            onnx_out_f = onnx_out.astype(np.float32)
            diff = np.abs(pt_out - onnx_out_f)
            print(f"\n  Comparison:")
            print(f"    PyTorch output shape: {pt_out.shape}")
            print(f"    ONNX output shape:    {onnx_out_f.shape}")
            print(f"    Max Abs Error:   {diff.max():.6e}")
            print(f"    Mean Abs Error:  {diff.mean():.6e}")
            print(f"    PyTorch range:   [{pt_out.min():.6f}, {pt_out.max():.6f}]")
            print(f"    ONNX range:      [{onnx_out_f.min():.6f}, {onnx_out_f.max():.6f}]")

            THRESHOLD = 0.1  # fp16 精度较低
            if diff.max() < THRESHOLD:
                print(f"    ✅ PASS (max error < {THRESHOLD})")
            else:
                print(f"    ⚠️  WARN (max error >= {THRESHOLD})")
        except ImportError:
            print("  [SKIP] onnxruntime not installed")
        except Exception as e:
            print(f"  [ERROR] Verification failed: {e}")

    # ---- Summary ----
    bev_h = sparse_shape[1] // 4   # 512 // 4 = 128
    bev_w = sparse_shape[2] // 4   # 800 // 4 = 200
    print("\n" + "=" * 60)
    print("Export Summary")
    print("=" * 60)
    print(f"  SparseEncoder: {args.save_onnx}")
    print(f"    voxels [N, {in_channels}] → dense BEV [B, 640, {bev_h}, {bev_w}]")
    if args.save_io:
        if args.save_io_all_frames:
            print(f"  Per-frame IO: {args.save_io}/<token>/ + manifest.json")
        else:
            print(f"  IO saved to: {args.save_io}/")
    print(f"\n  Deployment pipeline:")
    print(f"    1. scn.onnx:                      voxels → dense BEV [B, 640, {bev_h}, {bev_w}]")
    print(f"    2. second_neck_head_querygen.onnx: BEV → SECOND → neck → head → querygen")
