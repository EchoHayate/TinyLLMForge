from __future__ import annotations

from dataclasses import dataclass
import gc
import hashlib
import math
import re
from types import MappingProxyType
from typing import Mapping
import weakref


HIDDEN_SIZE = 5120
GLOBAL_KEY_HEADS = 16
GLOBAL_VALUE_HEADS = 48
HEAD_DIM = 128
CONV_KERNEL_WIDTH = 4


def _shape(tensor: object) -> tuple[int, ...]:
    try:
        return tuple(int(width) for width in tensor.shape)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("candidate tensor shape is invalid") from error


def _require_tensor(
    tensor: object,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: object | None = None,
) -> object:
    if tensor is None or _shape(tensor) != shape:
        raise ValueError(f"{name} shape must equal {shape}")
    floating = getattr(tensor, "is_floating_point", None)
    if not callable(floating) or not floating():
        raise ValueError(f"{name} must use a floating point dtype")
    if dtype is not None and tensor.dtype != dtype:
        raise ValueError(f"{name} dtype is invalid")
    return tensor


def _require_unquantized(module: object, name: str) -> None:
    if getattr(module, "quant_method", None) is not None:
        raise ValueError(f"{name} must be unquantized")


def _tensor_digest(tensor: object) -> str:
    digest = hashlib.sha256()
    digest.update(str(_shape(tensor)).encode("utf-8"))
    digest.update(str(getattr(tensor, "dtype", None)).encode("utf-8"))
    try:
        import torch

        payload = (
            tensor.detach()
            .contiguous()
            .view(torch.uint8)
            .cpu()
            .numpy()
            .tobytes()
        )
    except (AttributeError, RuntimeError, TypeError, ValueError):
        payload = str(getattr(tensor, "label", repr(tensor))).encode(
            "utf-8"
        )
    digest.update(payload)
    return digest.hexdigest()


def build_parameter_identity_record(
    *,
    candidate_slice_digests: Mapping[str, str],
    checkpoint_full_digests: Mapping[str, str],
    reconstructed_full_digests: Mapping[str, str],
) -> dict:
    candidate = dict(candidate_slice_digests)
    checkpoint = dict(checkpoint_full_digests)
    reconstructed = dict(reconstructed_full_digests)
    for name, values in (
        ("candidate slice", candidate),
        ("checkpoint full", checkpoint),
        ("reconstructed full", reconstructed),
    ):
        if not values or any(
            not isinstance(key, str)
            or not key
            or not isinstance(value, str)
            or not re.fullmatch(r"[0-9a-f]{64}", value)
            for key, value in values.items()
        ):
            raise ValueError(f"{name} parameter digests are invalid")
    if (
        checkpoint.keys() != reconstructed.keys()
        or checkpoint != reconstructed
    ):
        raise ValueError(
            "candidate slices do not reconstruct checkpoint"
        )
    return {
        "parameter_digests": candidate,
        "checkpoint_full_parameter_digests": checkpoint,
        "reconstructed_full_parameter_digests": reconstructed,
        "checkpoint_reconstruction_match": True,
    }


def _slice_rows(tensor: object, start: int, length: int) -> object:
    return tensor.narrow(0, start, length).contiguous()


def _slice_columns(tensor: object, start: int, length: int) -> object:
    return tensor.narrow(1, start, length).contiguous()


def _select_state_parameter(
    tensor: object,
    *,
    name: str,
    full_shape: tuple[int, ...],
    half_shape: tuple[int, ...],
    start: int,
    length: int,
) -> object:
    shape = _shape(tensor)
    if shape == half_shape:
        return tensor
    if shape == full_shape:
        return _slice_rows(tensor, start, length)
    raise ValueError(
        f"{name} shape must equal {full_shape} or {half_shape}"
    )


@dataclass(frozen=True)
class LogicalTP2LinearAttentionView:
    logical_parallel_size: int
    logical_rank: int
    key_head_range: tuple[int, int]
    value_head_range: tuple[int, int]
    output_input_range: tuple[int, int]
    pair_group: object
    qkv_weight: object
    qkv_weight_segments: tuple[object, object, object]
    z_weight: object
    z_weight_half: object
    b_weight: object
    b_weight_half: object
    a_weight: object
    a_weight_half: object
    ab_weight_half: object
    conv_weight: object
    A_log: object
    dt_bias: object
    norm_weight: object
    norm_eps: float
    output_accumulation_weight: object
    tensor_digests: Mapping[str, str]


def build_logical_tp2_linear_attention_view(
    layer: object,
    logical_rank: int,
    pair_group: object,
) -> LogicalTP2LinearAttentionView:
    if type(layer).__name__ != "Qwen35LinearAttentionShell":
        raise ValueError("layer must be a Qwen35LinearAttentionShell")
    if (
        isinstance(logical_rank, bool)
        or not isinstance(logical_rank, int)
        or logical_rank not in (0, 1)
    ):
        raise ValueError("logical_rank must be zero or one")
    pinned = (
        getattr(layer, "local_key_heads", None),
        getattr(layer, "local_value_heads", None),
        getattr(layer, "key_head_dim", None),
        getattr(layer, "value_head_dim", None),
    )
    if pinned != (4, 12, HEAD_DIM, HEAD_DIM):
        raise ValueError(
            "layer does not match pinned Qwen3.8 TP4 dimensions"
        )
    norm_eps = getattr(layer, "norm_eps", None)
    if (
        isinstance(norm_eps, bool)
        or not isinstance(norm_eps, (int, float))
        or not math.isfinite(norm_eps)
        or norm_eps <= 0
    ):
        raise ValueError("layer norm_eps is invalid")

    projection_names = (
        "in_proj_qkv",
        "in_proj_z",
        "in_proj_b",
        "in_proj_a",
        "out_proj",
    )
    projections = {}
    for name in projection_names:
        module = getattr(layer, name, None)
        if module is None:
            raise ValueError(f"{name} is missing")
        _require_unquantized(module, name)
        projections[name] = module

    qkv_full = _require_tensor(
        getattr(projections["in_proj_qkv"], "weight", None),
        name="in_proj_qkv.weight",
        shape=(10240, HIDDEN_SIZE),
    )
    z_full = _require_tensor(
        getattr(projections["in_proj_z"], "weight", None),
        name="in_proj_z.weight",
        shape=(6144, HIDDEN_SIZE),
    )
    b_full = _require_tensor(
        getattr(projections["in_proj_b"], "weight", None),
        name="in_proj_b.weight",
        shape=(48, HIDDEN_SIZE),
    )
    a_full = _require_tensor(
        getattr(projections["in_proj_a"], "weight", None),
        name="in_proj_a.weight",
        shape=(48, HIDDEN_SIZE),
    )
    out_full = _require_tensor(
        getattr(projections["out_proj"], "prefill_weight", None),
        name="out_proj.prefill_weight",
        shape=(HIDDEN_SIZE, 6144),
    )
    accumulation = _require_tensor(
        getattr(projections["out_proj"], "accumulation_weight", None),
        name="out_proj.accumulation_weight",
        shape=(HIDDEN_SIZE, 1536),
    )
    if str(accumulation.dtype) != "torch.float32":
        raise ValueError("out_proj accumulation must use FP32")
    if getattr(projections["out_proj"], "tp_size", None) != 4:
        raise ValueError("out_proj must retain the TP4 baseline")

    conv_source = getattr(layer, "logical_tp2_conv_weight", None)
    _require_tensor(
        conv_source,
        name="logical_tp2_conv_weight",
        shape=_shape(conv_source),
    )
    A_log_source = getattr(layer, "logical_tp2_A_log", None)
    _require_tensor(
        A_log_source,
        name="logical_tp2_A_log",
        shape=_shape(A_log_source),
    )
    dt_bias_source = getattr(layer, "logical_tp2_dt_bias", None)
    _require_tensor(
        dt_bias_source,
        name="logical_tp2_dt_bias",
        shape=_shape(dt_bias_source),
    )
    norm_weight = _require_tensor(
        getattr(layer, "norm_weight", None),
        name="norm_weight",
        shape=(HEAD_DIM,),
    )

    import torch

    key_start = logical_rank * 8
    value_start = logical_rank * 24
    key_width_start = logical_rank * 1024
    value_width_start = logical_rank * 3072
    a_weight_half = a_full.narrow(0, value_start, 24)
    b_weight_half = b_full.narrow(0, value_start, 24)
    qkv_weight_segments = (
        qkv_full.narrow(0, key_width_start, 1024),
        qkv_full.narrow(0, 2048 + key_width_start, 1024),
        qkv_full.narrow(0, 4096 + value_width_start, 3072),
    )
    selected = {
        "qkv_weight": qkv_full,
        "qkv_weight_segments": qkv_weight_segments,
        "z_weight": z_full,
        "z_weight_half": z_full.narrow(
            0,
            value_width_start,
            3072,
        ),
        "b_weight": b_full,
        "b_weight_half": b_weight_half,
        "a_weight": a_full,
        "a_weight_half": a_weight_half,
        "ab_weight_half": torch.cat((
            a_weight_half,
            b_weight_half,
        ), dim=0).contiguous(),
        "conv_weight": _select_state_parameter(
            conv_source,
            name="logical_tp2_conv_weight",
            full_shape=(10240, CONV_KERNEL_WIDTH),
            half_shape=(5120, CONV_KERNEL_WIDTH),
            start=key_width_start,
            length=1024,
        ),
        "A_log": _select_state_parameter(
            A_log_source,
            name="logical_tp2_A_log",
            full_shape=(GLOBAL_VALUE_HEADS,),
            half_shape=(24,),
            start=value_start,
            length=24,
        ),
        "dt_bias": _select_state_parameter(
            dt_bias_source,
            name="logical_tp2_dt_bias",
            full_shape=(GLOBAL_VALUE_HEADS,),
            half_shape=(24,),
            start=value_start,
            length=24,
        ),
        "norm_weight": norm_weight,
        "output_accumulation_weight": _slice_columns(
            out_full,
            value_width_start,
            3072,
        ).to(dtype=torch.float32).contiguous(),
    }
    if _shape(conv_source) == (10240, CONV_KERNEL_WIDTH):
        selected["conv_weight"] = torch.cat((
            _slice_rows(conv_source, key_width_start, 1024),
            _slice_rows(conv_source, 2048 + key_width_start, 1024),
            _slice_rows(
                conv_source,
                4096 + value_width_start,
                3072,
            ),
        ), dim=0).contiguous()
    digest_tensors = {
        name: tensor
        for name, tensor in selected.items()
        if name != "qkv_weight_segments"
    }
    digest_tensors.update({
        f"qkv_weight_segment_{index}": tensor
        for index, tensor in enumerate(qkv_weight_segments)
    })
    return LogicalTP2LinearAttentionView(
        logical_parallel_size=2,
        logical_rank=logical_rank,
        key_head_range=(key_start, key_start + 8),
        value_head_range=(value_start, value_start + 24),
        output_input_range=(
            value_width_start,
            value_width_start + 3072,
        ),
        pair_group=pair_group,
        norm_eps=float(norm_eps),
        tensor_digests=MappingProxyType({
            name: _tensor_digest(tensor)
            for name, tensor in sorted(digest_tensors.items())
        }),
        **selected,
    )


def build_layer_parameter_identity(
    layer: object,
    view: LogicalTP2LinearAttentionView,
) -> dict:
    import torch

    checkpoint_full = {
        "qkv_weight": layer.in_proj_qkv.weight,
        "z_weight": layer.in_proj_z.weight,
        "b_weight": layer.in_proj_b.weight,
        "a_weight": layer.in_proj_a.weight,
        "norm_weight": layer.norm_weight,
        "output_accumulation_weight": (
            layer.out_proj.prefill_weight
            .to(dtype=torch.float32)
            .contiguous()
        ),
    }
    out_full = checkpoint_full["output_accumulation_weight"]
    reconstructed_full = {
        **checkpoint_full,
        "output_accumulation_weight": torch.cat((
            _slice_columns(out_full, 0, 3072),
            _slice_columns(out_full, 3072, 3072),
        ), dim=1).contiguous(),
    }
    return build_parameter_identity_record(
        candidate_slice_digests=dict(view.tensor_digests),
        checkpoint_full_digests={
            name: _tensor_digest(tensor)
            for name, tensor in checkpoint_full.items()
        },
        reconstructed_full_digests={
            name: _tensor_digest(tensor)
            for name, tensor in reconstructed_full.items()
        },
    )


def merge_parameter_identity_records(
    *records: Mapping[str, object],
) -> dict:
    candidate = {}
    checkpoint = {}
    reconstructed = {}
    for record in records:
        if record.get("checkpoint_reconstruction_match") is not True:
            raise ValueError(
                "parameter reconstruction proof is incomplete"
            )
        candidate.update(record["parameter_digests"])
        checkpoint.update(
            record["checkpoint_full_parameter_digests"]
        )
        reconstructed.update(
            record["reconstructed_full_parameter_digests"]
        )
    return build_parameter_identity_record(
        candidate_slice_digests=candidate,
        checkpoint_full_digests=checkpoint,
        reconstructed_full_digests=reconstructed,
    )


class CandidateSetupLifecycle:
    def __init__(self) -> None:
        self._warmup_started = False
        self._views = []

    @property
    def views(self) -> tuple[LogicalTP2LinearAttentionView, ...]:
        return tuple(self._views)

    def register_candidate_view(
        self,
        view: LogicalTP2LinearAttentionView,
    ) -> None:
        if self._warmup_started:
            raise RuntimeError(
                "candidate setup must finish before warmup"
            )
        if type(view) is not LogicalTP2LinearAttentionView:
            raise ValueError("candidate view has the wrong type")
        self._views.append(view)

    def mark_warmup_started(self) -> None:
        self._warmup_started = True


def release_global_tp4_decode_accumulation(layer: object) -> dict:
    weight = getattr(layer.out_proj, "accumulation_weight", None)
    if weight is None:
        raise RuntimeError(
            "global TP4 decode accumulation weight is missing"
        )
    released_bytes = int(weight.numel()) * int(weight.element_size())
    reference = weakref.ref(weight)
    layer.out_proj.accumulation_weight = None
    del weight
    gc.collect()
    if reference() is not None:
        raise RuntimeError(
            "global TP4 decode accumulation weight remained live"
        )
    return {
        "released_bytes": released_bytes,
        "released": True,
    }
