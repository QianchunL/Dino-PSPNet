import argparse
import inspect
import os

import numpy as np
import onnx
import onnx.numpy_helper
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet.models import DETECTORS as MMDET_DETECTORS

# from mmdeploy.apis import extract_model, get_predefined_partition_cfg, torch2onnx
# from mmdeploy.core import patch_model
from mmdet3d.models import build_model
from onnxsim import simplify

# from projects.mmdet3d_plugin.bevformer.modules.util import (
#     multi_scale_deformable_attn_pytorch,
# )
from projects.mmdet3d_plugin.datasets.builder import build_dataloader  # noqa

# 全局设备配置
DEVICE = "cuda"

# ============================================================================
# Function 包装器：将 multi_scale_deformable_attn_pytorch 包装为 Function
# ============================================================================


class MultiScaleDeformableAttnPytorchFunction(torch.autograd.Function):
    """包装 multi_scale_deformable_attn_pytorch 为 Function，以便 ONNX 追踪和替换"""

    @staticmethod
    def forward(
        ctx, value, value_spatial_shapes, level_start_index, sampling_locations, attention_weights, im2col_step
    ):
        """Forward 方法：调用原始的 multi_scale_deformable_attn_pytorch 函数"""

        # 获取真正的原始函数
        import importlib

        util_module = importlib.import_module("projects.mmdet3d_plugin.bevformer.modules.util")
        if hasattr(util_module, "_original_multi_scale_deformable_attn_pytorch"):
            orig_func = util_module._original_multi_scale_deformable_attn_pytorch
        else:
            orig_func = util_module.multi_scale_deformable_attn_pytorch
            util_module._original_multi_scale_deformable_attn_pytorch = orig_func
        return orig_func(
            value, value_spatial_shapes, level_start_index, sampling_locations, attention_weights, im2col_step
        )

    @staticmethod
    def symbolic(g, value, value_spatial_shapes, level_start_index, sampling_locations, attention_weights, im2col_step):
        """ONNX symbolic 方法：替换为 mmdeploy 算子"""
        print("[自定义 SYMBOLIC] MultiScaleDeformableAttnPytorchFunction 被调用")

        # 将 value_spatial_shapes 和 level_start_index 转换为 int32
        value_spatial_shapes_int32 = g.op("Cast", value_spatial_shapes, to_i=onnx.TensorProto.INT32)
        level_start_index_int32 = g.op("Cast", level_start_index, to_i=onnx.TensorProto.INT32)

        return g.op(
            "mmdeploy::MMCVMultiScaleDeformableAttention",
            value,
            value_spatial_shapes_int32,
            level_start_index_int32,
            sampling_locations,
            attention_weights,
            im2col_step_i=im2col_step,
        )


def register_custom_symbolic_for_export():
    """不依赖 mmdeploy，直接注册 symbolic"""
    from projects.mmdet3d_plugin.bevformer.modules.multi_scale_deformable_attn_function import (
        MultiScaleDeformableAttnFunction_fp16,
        MultiScaleDeformableAttnFunction_fp32,
    )

    def ms_deform_attn_symbolic(
        g,
        value,
        value_spatial_shapes,
        level_start_index,
        sampling_locations,
        attention_weights,
        im2col_step=64,
    ):
        """自定义 symbolic 函数 - 替换为 mmdeploy 算子"""
        print("[自定义 SYMBOLIC] MultiScaleDeformableAttnFunction 被调用")
        try:
            print(f"  输入形状: value={value.type().sizes()}, spatial_shapes={value_spatial_shapes.type().sizes()}")
        except Exception as e:
            print(f"Error: {e}")
            pass

        # 将 value_spatial_shapes 和 level_start_index 转换为 int32
        value_spatial_shapes_int32 = g.op("Cast", value_spatial_shapes, to_i=onnx.TensorProto.INT32)
        level_start_index_int32 = g.op("Cast", level_start_index, to_i=onnx.TensorProto.INT32)

        return g.op(
            "mmdeploy::MMCVMultiScaleDeformableAttention",
            value,
            value_spatial_shapes_int32,
            level_start_index_int32,
            sampling_locations,
            attention_weights,
            im2col_step_i=im2col_step,
        )

    # 直接给 Function 类添加 symbolic 方法
    if not hasattr(MultiScaleDeformableAttnFunction_fp32, "symbolic"):
        MultiScaleDeformableAttnFunction_fp32.symbolic = staticmethod(ms_deform_attn_symbolic)
        print("✓ 已为 MultiScaleDeformableAttnFunction_fp32 添加 symbolic")
    else:
        print("⚠ MultiScaleDeformableAttnFunction_fp32 已有 symbolic，将被覆盖")
        MultiScaleDeformableAttnFunction_fp32.symbolic = staticmethod(ms_deform_attn_symbolic)

    if not hasattr(MultiScaleDeformableAttnFunction_fp16, "symbolic"):
        MultiScaleDeformableAttnFunction_fp16.symbolic = staticmethod(ms_deform_attn_symbolic)
        print("✓ 已为 MultiScaleDeformableAttnFunction_fp16 添加 symbolic")
    else:
        print("⚠ MultiScaleDeformableAttnFunction_fp16 已有 symbolic，将被覆盖")
        MultiScaleDeformableAttnFunction_fp16.symbolic = staticmethod(ms_deform_attn_symbolic)


def setup_export_environment(level_start_index):
    """设置导出环境：替换 multi_scale_deformable_attn_pytorch 为 Function 包装器"""
    import importlib
    import sys

    # 使用 importlib 避免命名冲突
    util_module = importlib.import_module("projects.mmdet3d_plugin.bevformer.modules.util")

    # 保存原始函数
    if not hasattr(util_module, "_original_multi_scale_deformable_attn_pytorch"):
        util_module._original_multi_scale_deformable_attn_pytorch = util_module.multi_scale_deformable_attn_pytorch

    # 创建替换函数
    def replacement_func(
        value, value_spatial_shapes, level_start_index, sampling_locations, attention_weights, im2col_step
    ):
        """替换函数：调用 Function 包装器"""
        return MultiScaleDeformableAttnPytorchFunction.apply(
            value, value_spatial_shapes, level_start_index, sampling_locations, attention_weights, im2col_step
        )

    # 替换模块中的函数
    util_module.multi_scale_deformable_attn_pytorch = replacement_func

    # 替换当前模块中的导入
    current_module = sys.modules[__name__]
    if hasattr(current_module, "multi_scale_deformable_attn_pytorch"):
        current_module.multi_scale_deformable_attn_pytorch = replacement_func

    # 替换其他模块中的引用
    # 先获取所有模块名称的列表，避免在迭代时字典大小改变
    original_func = util_module._original_multi_scale_deformable_attn_pytorch
    module_names = list(sys.modules.keys())
    for module_name in module_names:
        module = sys.modules.get(module_name)
        if module is None:
            continue
        if hasattr(module, "multi_scale_deformable_attn_pytorch"):
            if module.multi_scale_deformable_attn_pytorch is original_func:
                module.multi_scale_deformable_attn_pytorch = replacement_func

    print("✓ 已替换 multi_scale_deformable_attn_pytorch 为 Function 包装器")


# 在文件加载时调用（注册 symbolic）
register_custom_symbolic_for_export()

# ============================================================================
# ONNX 导出包装器类
# ============================================================================


class Stage2ModelWrapper(torch.nn.Module):
    """用于 ONNX 导出 Stage2 模型（将位置参数转换为关键字参数）
    因为 MapTRv2.forward() 只接受关键字参数，不能直接传入位置参数
    """

    def __init__(
        self,
        model,
        attr_names=None,
        has_curb=False,
        has_stopline=False,
        has_arrow=False,
        has_segment_style=False,
        use_marking_map=False,
        use_height_map=False,
    ):
        super().__init__()
        self.model = model
        self.attr_names = attr_names or []
        self.has_curb = has_curb
        self.has_stopline = has_stopline
        self.has_arrow = has_arrow
        self.has_segment_style = has_segment_style
        self.use_marking_map = use_marking_map
        self.use_height_map = use_height_map
        self.line_query_groups = getattr(getattr(model, "pts_bbox_head", None), "query_groups_one2one", None)

    def _get_line_group(self, group_name):
        if self.line_query_groups is None:
            return None
        for group in self.line_query_groups:
            if group["name"] == group_name:
                return group
        return None

    def _slice_group_tensor(self, tensor, group_name, class_only=False):
        if tensor is None:
            return None
        group = self._get_line_group(group_name)
        if group is None:
            return None
        group_tensor = tensor[:, group["start"] : group["end"]]
        if class_only:
            class_ids = list(group.get("class_ids", []))
            if len(class_ids) == 0:
                return None
            return group_tensor[..., class_ids]
        return group_tensor

    def _collect_unified_line_outputs(self, compile_outs):
        outputs = []
        outputs.append(self._slice_group_tensor(compile_outs.get("all_cls_scores"), "divider", class_only=True))
        outputs.append(self._slice_group_tensor(compile_outs.get("all_pts_preds"), "divider"))
        if "cls_type" in self.attr_names:
            outputs.append(self._slice_group_tensor(compile_outs.get("all_cls_type_outputs"), "divider"))
        if self.has_stopline:
            outputs.append(self._slice_group_tensor(compile_outs.get("all_cls_scores"), "stopline", class_only=True))
            outputs.append(self._slice_group_tensor(compile_outs.get("all_pts_preds"), "stopline"))
        if self.has_arrow:
            outputs.append(compile_outs.get("arrow_all_cls_scores"))
            outputs.append(compile_outs.get("arrow_all_bbox_preds"))
        if self.has_curb:
            outputs.append(self._slice_group_tensor(compile_outs.get("all_cls_scores"), "boundary", class_only=True))
            outputs.append(self._slice_group_tensor(compile_outs.get("all_pts_preds"), "boundary"))
        for attr_name in self.attr_names:
            if attr_name == "cls_type":
                continue
            outputs.append(self._slice_group_tensor(compile_outs.get(f"all_{attr_name}_outputs"), "divider"))
        if self.has_segment_style:
            outputs.append(self._slice_group_tensor(compile_outs.get("all_segment_style_outputs"), "divider"))
        return tuple(outputs)

    def _collect_outputs(self, compile_outs):
        use_unified_line_outputs = (
            self.line_query_groups is not None
            and "stopline_all_cls_scores" not in compile_outs
            and "curb_all_cls_scores" not in compile_outs
        )
        if use_unified_line_outputs:
            return self._collect_unified_line_outputs(compile_outs)

        outputs = []
        # maptr outputs (always present)
        outputs.append(compile_outs.get("all_cls_scores"))
        outputs.append(compile_outs.get("all_pts_preds"))
        # keep cls_type in the legacy third position
        if "cls_type" in self.attr_names:
            outputs.append(compile_outs.get("all_cls_type_outputs"))
        # stopline outputs (optional)
        if self.has_stopline:
            outputs.append(compile_outs.get("stopline_all_cls_scores"))
            outputs.append(compile_outs.get("stopline_all_pts_preds"))
        # arrow outputs (optional)
        if self.has_arrow:
            outputs.append(compile_outs.get("arrow_all_cls_scores"))
            outputs.append(compile_outs.get("arrow_all_bbox_preds"))
        # curb outputs (optional)
        if self.has_curb:
            outputs.append(compile_outs.get("curb_all_cls_scores"))
            outputs.append(compile_outs.get("curb_all_pts_preds"))
        # keep remaining attributes at the end, e.g. color
        for attr_name in self.attr_names:
            if attr_name == "cls_type":
                continue
            attr_key = f"all_{attr_name}_outputs"
            outputs.append(compile_outs.get(attr_key))
        # keep new segment_style logits at the end to avoid shifting legacy outputs
        if self.has_segment_style:
            outputs.append(compile_outs.get("all_segment_style_outputs"))
        return tuple(outputs)

    def _build_dummy_metas(self, img, marking_map=None, height_map=None):
        batch_size = img.shape[0]
        dummy_metas = [[{} for _ in range(batch_size)]]
        for b in range(batch_size):
            if self.use_marking_map and marking_map is not None:
                dummy_metas[0][b]["marking_map"] = marking_map[b]
            if self.use_height_map and height_map is not None:
                dummy_metas[0][b]["height_map"] = height_map[b]
        return dummy_metas

    def forward(self, img, ref_2d, reference_points_cam, bev_mask, marking_map=None, height_map=None):
        """Forward 方法，支持 lane_color_lc 的 LC 先验输入。"""
        dummy_metas = self._build_dummy_metas(img, marking_map=marking_map, height_map=height_map)
        result = self.model.forward_dummy(
            img=img,
            dummy_metas=dummy_metas,
            ref_2d=ref_2d,
            reference_points_cam=reference_points_cam,
            bev_mask=bev_mask,
        )
        # 在 compile_mode 下，返回 (bev_embed, compile_outs)
        # compile_outs 是字典，需要提取为 tensor
        if isinstance(result, tuple) and len(result) == 2:
            _, compile_outs = result
            if isinstance(compile_outs, dict):
                return self._collect_outputs(compile_outs)
        if isinstance(result, dict):
            return self._collect_outputs(result)
        return result


class BEVEncoderWrapper(torch.nn.Module):
    """用于 ONNX 导出 BEVFormerEncoder"""

    def __init__(self, encoder):
        super().__init__()
        self.bev_encoder = encoder

    def forward(self, mlvl_feats, ref_2d, reference_points_cam, bev_mask):
        return self.bev_encoder(
            mlvl_feats,
            None,
            None,
            ref_2d=ref_2d,
            reference_points_cam=reference_points_cam,
            bev_mask=bev_mask,
            ref_points_sampling_compile=True,
        )


class SelfAttentionWrapper(torch.nn.Module):
    """用于 ONNX 导出 TemporalSelfAttention"""

    def __init__(self, self_attn, spatial_shapes, level_start_index):
        super().__init__()
        self.self_attn = self_attn
        self.spatial_shapes = spatial_shapes
        self.level_start_index = level_start_index

    def forward(self, query, bev_pos, ref_2d):
        output = self.self_attn(
            query,
            None,
            None,
            None,
            query_pos=bev_pos,
            key_pos=bev_pos,
            attn_mask=None,
            key_padding_mask=None,
            reference_points=ref_2d,
            spatial_shapes=self.spatial_shapes,
            level_start_index=self.level_start_index,
        )
        return output


class CrossAttentionWrapper(torch.nn.Module):
    """用于 ONNX 导出 SpatialCrossAttention"""

    def __init__(self, cross_attn, spatial_shapes, level_start_index):
        super().__init__()
        self.cross_attn = cross_attn
        self.spatial_shapes = spatial_shapes.reshape([1, 2])
        self.level_start_index = level_start_index

    def forward(self, query, key, value, reference_points, reference_points_cam, bev_mask):
        print("spatial_shapes: ", self.spatial_shapes)

        output = self.cross_attn(
            query,
            key,
            value,
            None,
            query_pos=None,
            key_pos=None,
            reference_points=reference_points,
            reference_points_cam=reference_points_cam,
            mask=None,
            attn_mask=None,
            key_padding_mask=None,
            bev_mask=bev_mask,
            spatial_shapes=self.spatial_shapes,
            level_start_index=self.level_start_index,
        )
        return output


# ============================================================================
# 辅助函数
# ============================================================================


def convert_mmcv_attention_int64_to_int32(onnx_model):
    """仅针对 mmdeploy::MMCVMultiScaleDeformableAttention 节点转换 int64 到 int32

    Args:
        onnx_model: ONNX 模型对象

    Returns:
        onnx_model: 转换后的 ONNX 模型对象
    """
    # ONNX 数据类型常量
    INT64 = 7
    INT32 = 6
    OP_TYPE = "MMCVMultiScaleDeformableAttention"

    # 创建名称到 initializer 的映射
    initializer_map = {init.name: init for init in onnx_model.graph.initializer}

    # 创建名称到 value_info 的映射
    value_info_map = {vi.name: vi for vi in onnx_model.graph.value_info}

    # 创建名称到 input 的映射
    input_map = {inp.name: inp for inp in onnx_model.graph.input}

    # 查找所有 mmdeploy::MMCVMultiScaleDeformableAttention 节点
    attention_nodes = [node for node in onnx_model.graph.node if node.op_type == OP_TYPE]

    if not attention_nodes:
        print(f"  未找到 {OP_TYPE} 节点")
        return onnx_model

    print(f"  找到 {len(attention_nodes)} 个 {OP_TYPE} 节点，开始转换 int64 到 int32...")

    # 处理每个 attention 节点
    for node in attention_nodes:
        print(f"  处理节点: {node.name}")

        # 处理节点的输入
        for input_name in node.input:
            if not input_name:  # 跳过空输入
                continue

            # 检查是否是 initializer
            if input_name in initializer_map:
                initializer = initializer_map[input_name]
                if initializer.data_type == INT64:
                    print(f"    转换输入 initializer {input_name} 从 int64 到 int32")
                    # 将 int64 数组转换为 int32
                    int64_array = onnx.numpy_helper.to_array(initializer)
                    int32_array = int64_array.astype(np.int32)
                    # 更新初始值
                    initializer.CopyFrom(onnx.numpy_helper.from_array(int32_array, name=initializer.name))
                    initializer.data_type = INT32

            # 检查是否是 graph input
            elif input_name in input_map:
                input_tensor = input_map[input_name]
                if input_tensor.type.tensor_type.elem_type == INT64:
                    print(f"    转换输入 {input_name} 从 int64 到 int32")
                    input_tensor.type.tensor_type.elem_type = INT32

            # 检查是否是 value_info（中间值）
            elif input_name in value_info_map:
                value_info = value_info_map[input_name]
                if value_info.type.tensor_type.elem_type == INT64:
                    print(f"    转换中间值 {input_name} 从 int64 到 int32")
                    value_info.type.tensor_type.elem_type = INT32

        # 处理节点的属性（attribute）
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.TENSOR:
                # 如果是 tensor 属性且是 int64，转换为 int32
                if attr.t.data_type == INT64:
                    print(f"    转换节点 {node.name} 的属性 {attr.name} 从 int64 到 int32")
                    int64_array = onnx.numpy_helper.to_array(attr.t)
                    int32_array = int64_array.astype(np.int32)
                    attr.t.CopyFrom(onnx.numpy_helper.from_array(int32_array))
                    attr.t.data_type = INT32

    print(f"  ✓ 完成 {OP_TYPE} 节点的 int64 到 int32 转换")
    return onnx_model


def load_model_and_config(config_path, checkpoint_path, compile_stage1=False, compile_stage2=False):
    """加载模型和配置的辅助函数

    Args:
        config_path: 配置文件路径
        checkpoint_path: 权重文件路径
        compile_stage1: 是否编译 stage1
        compile_stage2: 是否编译 stage2

    Returns:
        model: 加载好的模型
        cfg: 配置对象
    """
    # 1. 加载配置
    cfg = Config.fromfile(config_path)
    cfg.model.train_cfg = None
    detector_cls = MMDET_DETECTORS.get(cfg.model.type)
    detector_sig = inspect.signature(detector_cls.__init__) if detector_cls is not None else None

    def _supports_kwarg(name):
        return detector_sig is not None and name in detector_sig.parameters

    if _supports_kwarg("compile_stage1_mode"):
        cfg.model.compile_stage1_mode = compile_stage1
    else:
        cfg.model.pop("compile_stage1_mode", None)

    if _supports_kwarg("compile_stage2_mode"):
        cfg.model.compile_stage2_mode = compile_stage2
    else:
        cfg.model.pop("compile_stage2_mode", None)

    if _supports_kwarg("maptr_visualizer"):
        cfg.model.maptr_visualizer = None
    else:
        cfg.model.pop("maptr_visualizer", None)

    # 支持 OD / LD 两种 encoder 路径；reference point 导出会在后续显式选择实际分支。
    if hasattr(cfg.model, "bev_encoder") and hasattr(cfg.model.bev_encoder, "encoder"):
        cfg.model.bev_encoder.encoder.ref_points_sampling_compile = True
    if hasattr(cfg.model, "lane_bev_encoder") and hasattr(cfg.model.lane_bev_encoder, "encoder"):
        cfg.model.lane_bev_encoder.encoder.ref_points_sampling_compile = True

    # 2. 构建模型
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))

    # 3. 处理 FP16
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)

    # 4. 加载权重
    model = model.to(DEVICE)
    _ = load_checkpoint(model, checkpoint_path, map_location=DEVICE)
    model.eval()

    first_weight = next(model.parameters())
    print(f"模型权重设备: {first_weight.device}")

    # 5. 构建Input
    input_data = {}
    num_cam = len(cfg.camera_names) if hasattr(cfg, "camera_names") else 7
    img = torch.randn(1, num_cam, 3, 576, 1024, device=DEVICE, dtype=torch.float32)  # 主输入图像
    encoder_cfg = None
    if hasattr(cfg.model, "bev_encoder"):
        encoder_cfg = cfg.model.bev_encoder
    elif hasattr(cfg.model, "lane_bev_encoder"):
        encoder_cfg = cfg.model.lane_bev_encoder
    else:
        raise AttributeError("cfg.model has neither 'bev_encoder' nor 'lane_bev_encoder'")

    bev_h = encoder_cfg.bev_h
    bev_w = encoder_cfg.bev_w
    num_query = bev_h * bev_w  # 16875 = 75 * 225
    num_points_in_pillar = encoder_cfg.encoder.num_points_in_pillar

    ref_2d = torch.randn(1, num_query, 1, 2, device=DEVICE, dtype=torch.float32)
    reference_points_cam = torch.randn(
        num_cam, 1, num_query, num_points_in_pillar, 2, device=DEVICE, dtype=torch.float32
    )
    bev_mask = torch.randn(num_cam, 1, num_query, num_points_in_pillar, device=DEVICE, dtype=torch.float32) > 0.5
    mlvl_feats = [torch.randn(1, num_cam, 128, 18, 32, device=DEVICE, dtype=torch.float32)]
    query = torch.randn(1, num_query, 128, device=DEVICE, dtype=torch.float32)
    bev_pos = torch.randn(1, num_query, 128, device=DEVICE, dtype=torch.float32)
    spatial_shapes_self_attn = torch.tensor([[bev_h, bev_w]], device=DEVICE, dtype=torch.int32)
    spatial_shapes_cross_attn = torch.tensor([[18, 32]], device=DEVICE, dtype=torch.int32)
    level_start_index = torch.tensor([0], device=DEVICE, dtype=torch.int32)
    cross_key = torch.randn(num_cam, 576, 1, 128, device=DEVICE, dtype=torch.float32)
    cross_value = torch.randn(num_cam, 576, 1, 128, device=DEVICE, dtype=torch.float32)
    ref_3d = torch.randn(1, 2, num_query, 3, device=DEVICE, dtype=torch.float32)

    input_data["img"] = img
    input_data["ref_2d"] = ref_2d
    input_data["reference_points_cam"] = reference_points_cam
    input_data["bev_mask"] = bev_mask
    input_data["mlvl_feats"] = mlvl_feats
    input_data["query"] = query
    input_data["bev_pos"] = bev_pos
    input_data["spatial_shapes_self_attn"] = spatial_shapes_self_attn
    input_data["spatial_shapes_cross_attn"] = spatial_shapes_cross_attn
    input_data["level_start_index"] = level_start_index
    input_data["cross_key"] = cross_key
    input_data["cross_value"] = cross_value
    input_data["ref_3d"] = ref_3d

    return model, cfg, input_data


def print_output_shapes(model, export_inputs, output_names):
    """导出前打印每个输出 tensor 的维度，便于核对顺序。"""
    with torch.no_grad():
        outputs = model(*export_inputs)

    if torch.is_tensor(outputs):
        outputs = (outputs,)
    elif isinstance(outputs, list):
        outputs = tuple(outputs)
    elif not isinstance(outputs, tuple):
        outputs = (outputs,)

    print("========== Output Shapes ==========")
    print(f"actual outputs: {len(outputs)}, named outputs: {len(output_names)}")
    if len(outputs) != len(output_names):
        print("WARNING: output count does not match output_names count")

    for idx, output in enumerate(outputs):
        name = output_names[idx] if idx < len(output_names) else f"<unnamed_output_{idx}>"
        if torch.is_tensor(output):
            print(f"[{idx}] {name}: shape={tuple(output.shape)}, dtype={output.dtype}")
        else:
            print(f"[{idx}] {name}: type={type(output)}")
    print("===================================")


def export_onnx_model(
    model,
    export_inputs,
    onnx_path,
    input_names,
    output_names,
    simplify_model=True,
    verbose=False,
    print_weights_info=False,
):
    """通用的 ONNX 导出和简化函数

    Args:
        model: 要导出的模型（或包装器）
        export_inputs: 导出时的输入数据（tuple）
        onnx_path: ONNX 文件保存路径
        input_names: 输入节点名称列表
        output_names: 输出节点名称列表
        simplify_model: 是否简化模型，默认 True
        verbose: 是否打印导出细节，默认 False
        print_weights_info: 是否打印权重信息，默认 False

    Returns:
        final_onnx_path: 最终保存的 `_s32.onnx` 文件路径
    """
    # 创建输出目录
    outdir = os.path.dirname(onnx_path)
    if outdir and not os.path.exists(outdir):
        os.makedirs(outdir)

    print_output_shapes(model, export_inputs, output_names)

    final_onnx_path = onnx_path.replace(".onnx", "_s32.onnx")
    temp_onnx_path = onnx_path.replace(".onnx", "_tmp.onnx")
    stale_paths = [
        onnx_path,
        temp_onnx_path,
        onnx_path.replace(".onnx", "_sim.onnx"),
        onnx_path.replace(".onnx", "_sim_s32.onnx"),
    ]

    # 导出 ONNX 到临时文件，避免在目录中保留中间产物
    with torch.no_grad():
        torch.onnx.export(
            model,
            export_inputs,
            temp_onnx_path,
            input_names=input_names,
            output_names=output_names,
            export_params=True,
            keep_initializers_as_inputs=False,
            do_constant_folding=True,
            opset_version=17,
            verbose=False,
            # operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
        )

    # 加载临时 ONNX 模型
    onnx_orig = onnx.load(temp_onnx_path)

    # 转换 mmdeploy::MMCVMultiScaleDeformableAttention 节点的 int64 到 int32
    print("🔄 转换 mmdeploy::MMCVMultiScaleDeformableAttention 节点的 int64 类型为 int32...")
    onnx_orig = convert_mmcv_attention_int64_to_int32(onnx_orig)
    if simplify_model:
        onnx_simp, check = simplify(onnx_orig, skip_constant_folding=True)
        assert check, "Simplified ONNX model could not be validated"

        # 简化后再次转换（因为简化可能会引入新的 int64）
        print("🔄 转换简化后的 mmdeploy::MMCVMultiScaleDeformableAttention 节点的 int64 类型为 int32...")
        onnx_simp = convert_mmcv_attention_int64_to_int32(onnx_simp)
        final_model = onnx_simp
    else:
        final_model = onnx_orig

    onnx.save(final_model, final_onnx_path)
    print(f"🚀 Export completed. ONNX saved as {final_onnx_path} 🤗")

    for stale_path in stale_paths:
        if stale_path == final_onnx_path:
            continue
        if os.path.exists(stale_path):
            os.remove(stale_path)

    # 可选：打印权重信息
    if print_weights_info:
        for initializer in final_model.graph.initializer:
            if initializer.data_type == 1:  # FLOAT (FP32)
                print(f"FP32权重: {initializer.name}, shape: {initializer.dims}")
            elif initializer.data_type == 10:  # FLOAT16
                print(f"FP16权重: {initializer.name}, shape: {initializer.dims}")
            elif initializer.data_type == 6:  # INT32
                print(f"INT32权重: {initializer.name}, shape: {initializer.dims}")
            elif initializer.data_type == 7:  # INT64
                print(f"INT64权重: {initializer.name}, shape: {initializer.dims}")
            else:
                print(f"其他类型权重: {initializer.name}, shape: {initializer.dims}")

    return final_onnx_path


# ============================================================================
# 导出函数
# ============================================================================


def export_maptr_to_onnx(args):
    # 1. 设置路径
    current_file_path = os.path.abspath(__file__)
    config_file = os.path.join(os.path.dirname(os.path.dirname(current_file_path)), args.cfg)
    checkpoint_file = args.ckpt

    # 2. 加载模型和配置
    model, cfg, _ = load_model_and_config(config_file, checkpoint_file, args.stage1, args.stage2)

    input_data = {}
    # 3. 导出ONNX模型
    if not os.path.exists(args.outpath):
        os.makedirs(args.outpath)
    if args.stage1:
        img = torch.randn(1, 7, 3, 576, 1024, device=DEVICE, dtype=torch.float32)  # 主输入图像
        onnx_path = os.path.join(args.outpath, "maptr_stage1_model.onnx")
        output_names = [
            "maptr_stage1_feats",
        ]
        input_data["img"] = img
        export_model = model
        input_names = list(input_data.keys())
    elif args.stage2:
        img = torch.randn(1, 7, 128, 18, 32, device=DEVICE, dtype=torch.float32)  # 主输入图像
        onnx_path = os.path.join(args.outpath, "maptr_stage2_model.onnx")

        # Check if model has stopline_head
        has_stopline = hasattr(model, "pts_bbox_stopline_head") and model.pts_bbox_stopline_head is not None
        # Check if model has curb_head
        has_curb = hasattr(model, "pts_bbox_curb_head") and model.pts_bbox_curb_head is not None
        # Check if model has attribute config
        has_attr = (
            hasattr(model, "pts_bbox_head")
            and model.pts_bbox_head is not None
            and hasattr(model.pts_bbox_head, "attributes_config")
            and model.pts_bbox_head.attributes_config is not None
            and len(model.pts_bbox_head.attributes_config) > 0
        )
        attr_names = list(model.pts_bbox_head.attributes_config.keys()) if has_attr else []
        # Check if model has arrow_head
        has_arrow = hasattr(model, "pts_bbox_arrow_head") and model.pts_bbox_arrow_head is not None
        unified_line_groups = getattr(getattr(model, "pts_bbox_head", None), "query_groups_one2one", None)
        unified_line_group_names = {group["name"] for group in unified_line_groups} if unified_line_groups else set()
        has_stopline = has_stopline or ("stopline" in unified_line_group_names)
        has_curb = has_curb or ("boundary" in unified_line_group_names)
        # Check if model has segment-style head
        has_segment_style = (
            hasattr(model, "pts_bbox_head")
            and model.pts_bbox_head is not None
            and getattr(model.pts_bbox_head, "segment_style_enabled", False)
        )

        # Build output names dynamically
        output_names = ["maptr_all_cls_scores", "maptr_all_pts_preds"]
        if "cls_type" in attr_names:
            output_names.append("maptr_all_cls_type_preds")
        if has_stopline:
            output_names.extend(["stopline_all_cls_scores", "stopline_all_pts_preds"])
        if has_arrow:
            output_names.extend(["arrow_all_cls_scores", "arrow_all_bbox_preds"])
        if has_curb:
            output_names.extend(["curb_all_cls_scores", "curb_all_pts_preds"])
        for attr_name in attr_names:
            if attr_name == "cls_type":
                continue
            output_names.append(f"maptr_all_{attr_name}_preds")
        if has_segment_style:
            output_names.append("maptr_all_segment_style_preds")

        print(f"Model has stopline_head: {has_stopline}")
        print(f"Model has curb_head: {has_curb}")
        print(f"Model has attributes: {has_attr}")
        print(f"Attribute names: {attr_names}")
        print(f"Model has arrow_head: {has_arrow}")
        print(f"Model has segment_style_head: {has_segment_style}")
        print(f"Output names: {output_names}")

        input_data["maptr_stage1_feats"] = img

        # 准备 reference points sampling 的 4 个输出作为输入
        # 从配置中获取参数以生成正确维度的 tensor
        bev_h = cfg.model.bev_encoder.bev_h
        bev_w = cfg.model.bev_encoder.bev_w
        num_query = bev_h * bev_w  # 16875 = 75 * 225
        num_points_in_pillar = cfg.model.bev_encoder.encoder.num_points_in_pillar
        num_cam = len(cfg.camera_names) if hasattr(cfg, "camera_names") else 7

        # 根据用户提供的维度创建 tensor
        # ref_2d: [1, 16875, 1, 2]
        ref_2d = torch.randn(1, num_query, 1, 2, device=DEVICE, dtype=torch.float32)
        # reference_points_cam: [7, 1, 16875, 4, 2]
        reference_points_cam = torch.randn(
            num_cam, 1, num_query, num_points_in_pillar, 2, device=DEVICE, dtype=torch.float32
        )
        # bev_mask: [7, 1, 16875, 4] (boolean)
        bev_mask = torch.randn(num_cam, 1, num_query, num_points_in_pillar, device=DEVICE, dtype=torch.float32) > 0.5

        input_data["ref_2d"] = ref_2d
        input_data["reference_points_cam"] = reference_points_cam
        input_data["bev_mask"] = bev_mask
        print("bev_mask shape: ", bev_mask.shape)

        use_marking_map = getattr(model, "divider_marking_channels", 0) > 0
        use_height_map = max(
            getattr(model, "curb_height_channels", 0),
            getattr(model, "divider_height_channels", 0),
        ) > 0
        print(f"Model uses marking_map prior: {use_marking_map}")
        print(f"Model uses height_map prior: {use_height_map}")

        if use_marking_map:
            divider_h = getattr(model, "divider_neck_bev_h", bev_h)
            divider_w = getattr(model, "divider_neck_bev_w", bev_w)
            marking_channels = getattr(model, "divider_marking_channels", 0)
            marking_map = torch.randn(1, marking_channels, divider_h, divider_w, device=DEVICE, dtype=torch.float32)
            input_data["marking_map"] = marking_map
            print("marking_map shape: ", marking_map.shape)

        if use_height_map:
            height_h = getattr(model, "curb_neck_bev_h", getattr(model, "divider_neck_bev_h", bev_h))
            height_w = getattr(model, "curb_neck_bev_w", getattr(model, "divider_neck_bev_w", bev_w))
            height_channels = max(
                getattr(model, "curb_height_channels", 0),
                getattr(model, "divider_height_channels", 0),
            )
            height_map = torch.randn(1, height_channels, height_h, height_w, device=DEVICE, dtype=torch.float32)
            input_data["height_map"] = height_map
            print("height_map shape: ", height_map.shape)

        setup_export_environment(1)

        wrapper = Stage2ModelWrapper(
            model,
            attr_names,
            has_curb,
            has_stopline,
            has_arrow,
            has_segment_style,
            use_marking_map=use_marking_map,
            use_height_map=use_height_map,
        )
        wrapper.eval()
        export_model = wrapper

        # 定义输入名称，顺序必须与 export_inputs 完全一致
        input_names = [
            "maptr_stage1_feats",  # img
            "ref_2d",  # ref_2d
            "reference_points_cam",  # reference_points_cam
            "bev_mask",  # bev_mask
        ]
        if use_marking_map:
            input_names.append("marking_map")
        if use_height_map:
            input_names.append("height_map")

    # stage2 模式下，传入所有输入包括 reference points tensor
    if args.stage2:
        export_inputs = [img, ref_2d, reference_points_cam, bev_mask]
        if "marking_map" in input_data:
            export_inputs.append(input_data["marking_map"])
        if "height_map" in input_data:
            export_inputs.append(input_data["height_map"])
        export_inputs = tuple(export_inputs)
    else:
        export_inputs = (img,)

    print(f"Export inputs count: {len(export_inputs)}")
    print(f"Input names count: {len(input_names)}")
    print(f"Input names: {input_names}")

    # 导出 ONNX 模型
    export_onnx_model(
        model=export_model,
        export_inputs=export_inputs,
        onnx_path=onnx_path,
        input_names=input_names,
        output_names=output_names,
        simplify_model=True,
        verbose=True,
        print_weights_info=True,
    )


def export_bevencoder(model, cfg, input_data, args):
    # 1. 获取 encoder
    encoder = model.bev_encoder
    encoder.eval()

    # 4. 创建包装器
    print("encoder: ", encoder)
    wrapper = BEVEncoderWrapper(encoder)
    print("wrapper: ", wrapper)
    wrapper.eval()

    mlvl_feats = input_data["mlvl_feats"]
    # ref_3d = input_data["ref_3d"]
    # ref_3d = None
    ref_2d = input_data["ref_2d"]
    reference_points_cam = input_data["reference_points_cam"]
    bev_mask = input_data["bev_mask"]

    # 5. 导出
    onnx_path = os.path.join(args.outpath, "bevencoder.onnx")

    export_onnx_model(
        model=wrapper,
        export_inputs=(mlvl_feats, ref_2d, reference_points_cam, bev_mask),
        onnx_path=onnx_path,
        input_names=["mlvl_feats", "ref_2d", "reference_points_cam", "bev_mask"],
        output_names=["bev_feats"],
    )


def export_self_attention(model, cfg, input_data, args):
    """导出 TemporalSelfAttention 模块

    Args:
        model: 加载好的模型
        cfg: 配置对象
        args: 命令行参数
    """
    # 1. 获取 self-attention 模块（从第一层获取）
    encoder = model.bev_encoder.encoder
    first_layer = encoder.layers[0]
    self_attn = first_layer.attentions[0]  # TemporalSelfAttention
    self_attn.eval()

    query = input_data["query"]
    bev_pos = input_data["bev_pos"]
    ref_2d = input_data["ref_2d"]
    spatial_shapes = input_data["spatial_shapes_self_attn"]
    level_start_index = input_data["level_start_index"]
    # 4. 创建包装器
    wrapper = SelfAttentionWrapper(self_attn, spatial_shapes, level_start_index)
    wrapper.eval()

    # 5. 导出
    onnx_path = os.path.join(args.outpath, "self_attention.onnx")

    export_onnx_model(
        model=wrapper,
        export_inputs=(query, bev_pos, ref_2d),
        onnx_path=onnx_path,
        input_names=["query", "bev_pos", "ref_2d"],
        output_names=["out_query"],
    )


def export_cross_attention(model, cfg, input_data, args):
    """导出 SpatialCrossAttention 模块

    Args:
        model: 加载好的模型
        cfg: 配置对象
        args: 命令行参数
    """
    # 1. 获取 cross-attention 模块（从第一层获取）
    encoder = model.bev_encoder.encoder
    first_layer = encoder.layers[0]
    cross_attn = first_layer.attentions[1]  # SpatialCrossAttention or SpatialSplitCrossAttention
    cross_attn.eval()
    query = input_data["query"]
    key = input_data["cross_key"]
    value = input_data["cross_value"]
    reference_points = input_data["ref_3d"]
    reference_points_cam = input_data["reference_points_cam"]
    spatial_shapes = input_data["spatial_shapes_cross_attn"]
    level_start_index = input_data["level_start_index"]
    bev_mask = input_data["bev_mask"]

    # 2. 设置导出环境：替换 multi_scale_deformable_attn_pytorch 为 Function 包装器
    print("\n" + "=" * 80)
    print("设置导出环境：替换 multi_scale_deformable_attn_pytorch 为 Function 包装器")
    print("=" * 80)
    setup_export_environment(level_start_index)

    # 3. 创建包装器
    wrapper = CrossAttentionWrapper(cross_attn, spatial_shapes, level_start_index)
    wrapper.eval()

    # 4. 导出
    onnx_path = os.path.join(args.outpath, "cross_attention_custom.onnx")

    print("\n" + "=" * 80)
    print("开始导出 ONNX 模型（使用 Function 包装器）")
    print("=" * 80)
    export_onnx_model(
        model=wrapper,
        export_inputs=(query, key, value, reference_points, reference_points_cam, bev_mask),
        onnx_path=onnx_path,
        input_names=["query", "key", "value", "reference_points", "reference_points_cam", "bev_mask"],
        output_names=["out_query"],
    )


def export_reference_points_sampling(args):
    """导出 reference points sampling 模块

    Args:
        args: 命令行参数
    """

    # 1. 设置路径
    current_file_path = os.path.abspath(__file__)
    config_file = os.path.join(os.path.dirname(os.path.dirname(current_file_path)), args.cfg)
    checkpoint_file = args.ckpt

    # 2. 加载模型和配置
    model, cfg, _ = load_model_and_config(config_file, checkpoint_file)

    # 3. 获取 encoder 和 reference points sampling 模块
    encoder = model.bev_encoder.encoder
    encoder.eval()
    ref_points_sampling = encoder.ref_points_sampling.eval()

    # 4. 从配置中获取参数
    bev_h = cfg.model.bev_encoder.bev_h
    bev_w = cfg.model.bev_encoder.bev_w
    pc_range = cfg.model.bev_encoder.pc_range
    num_points_in_pillar = encoder.num_points_in_pillar

    # 5. 准备输入
    bs = 1
    device = DEVICE
    dtype = torch.float32

    # lidar2img: (B, N, 4, 4)
    num_cam = len(cfg.camera_names) if hasattr(cfg, "camera_names") else 7
    lidar2img = torch.randn(bs, num_cam, 4, 4, device=device, dtype=dtype)

    # img_shape: (B, 2) [height, width]
    if hasattr(cfg, "input_shape"):
        img_h, img_w = cfg.input_shape
    else:
        img_h, img_w = 576.0, 1024.0
    img_shape = torch.tensor([[img_h, img_w]], device=device, dtype=dtype)

    # pc_range: (6,)
    pc_range_tensor = torch.tensor(pc_range, device=device, dtype=dtype)

    # 6. 导出
    onnx_path = os.path.join(args.outpath, "reference_points_sampling.onnx")

    export_onnx_model(
        model=ref_points_sampling,
        export_inputs=(
            torch.tensor(bev_h, device=device),
            torch.tensor(bev_w, device=device),
            pc_range_tensor,
            torch.tensor(num_points_in_pillar, device=device),
            lidar2img,
            img_shape,
        ),
        onnx_path=onnx_path,
        input_names=["bev_h", "bev_w", "pc_range", "num_points_in_pillar", "lidar2img", "img_shape"],
        output_names=["ref_3d", "ref_2d", "reference_points_cam", "bev_mask"],
    )


def parse_args():
    parser = argparse.ArgumentParser(description="MapTR onnx export...")
    parser.add_argument(
        "--cfg",
        type=str,
        help="test config file path",
        default="projects/configs/road_static_repvgga2_7v_04m_4head_78w_lane_color.py",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        help="checkpoint file",
        default="work_dirs/road_static_repvgga2_7v_04m_4head_78w_lane_color/epoch_4.pth",
    )
    parser.add_argument("--outpath", help="output path")
    parser.add_argument("--stage1", action="store_true", help="stage1 compile")
    parser.add_argument("--stage2", action="store_true", help="stage2 compile")
    parser.add_argument("--ref_points", action="store_true", help="export reference points sampling only")
    parser.add_argument("--bevencoder", action="store_true", help="export bevencoder module")
    parser.add_argument("--self_attn", action="store_true", help="export self-attention module")
    parser.add_argument("--cross_attn", action="store_true", help="export cross-attention module")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # 根据命令行参数选择导出模式
    if args.ref_points:
        # 导出 reference points sampling
        export_reference_points_sampling(args)
    elif args.bevencoder or args.self_attn or args.cross_attn:
        # 导出 bevencoder 或 attention 模块
        # 1. 设置路径
        current_file_path = os.path.abspath(__file__)
        config_file = os.path.join(os.path.dirname(os.path.dirname(current_file_path)), args.cfg)
        checkpoint_file = args.ckpt

        # 2. 加载模型和配置
        model, cfg, input_data = load_model_and_config(config_file, checkpoint_file)

        # 3. 根据参数调用相应的导出函数
        if args.bevencoder:
            export_bevencoder(model, cfg, input_data, args)

        if args.self_attn:
            export_self_attention(model, cfg, input_data, args)

        if args.cross_attn:
            export_cross_attention(model, cfg, input_data, args)
    else:
        # 默认导出完整模型（stage1 或 stage2）
        export_maptr_to_onnx(args)
