#!/usr/bin/env python3
"""Run the Qwen3.8 topology-local TP2 linear-attention microgate.

The production model remains unchanged.  This worker builds immutable TP2
views from checkpoint-backed layer-0 tensors before warmup, executes the
baseline and candidate mixer arms, and emits append-only per-rank evidence.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
import traceback
from types import MappingProxyType
from typing import Callable, Mapping

from tinyvllm.engine.topology_local_tp2_island import (
    TopologyLocalTP2PairMap,
    TopologyLocalTP2StateIdentity,
    assemble_logical_state_half,
    validate_state_publication,
)


WORKER_SCHEMA = "qwen38.topology-local-tp2-island-worker.v1"
ACTIVE_TOKEN_GROUPS = (1, 4, 8)
WARMUP_PAIR_COUNT = 2
MEASURED_PAIR_COUNT = 15
WORLD_SIZE = 4
HIDDEN_SIZE = 5120
GLOBAL_KEY_HEADS = 16
GLOBAL_VALUE_HEADS = 48
HEAD_DIM = 128
CONV_KERNEL_WIDTH = 4
LINEAR_ATTENTION_LAYERS = 48
STATE_CAPACITY = 8
PROJECTED_STEADY_INCREMENT_CEILING_BYTES = 1920 * 1024 * 1024


def checkpoint_state_tensor_slices(
    logical_rank: int,
) -> dict[str, tuple[tuple[int, int], ...]]:
    if logical_rank not in (0, 1):
        raise ValueError("logical_rank must be zero or one")
    key_start = logical_rank * 1024
    value_start = logical_rank * 3072
    head_start = logical_rank * 24
    return {
        "conv1d.weight": (
            (key_start, 1024),
            (2048 + key_start, 1024),
            (4096 + value_start, 3072),
        ),
        "A_log": ((head_start, 24),),
        "dt_bias": ((head_start, 24),),
    }


def project_persistent_reservation() -> dict[str, int]:
    output_increment_per_layer = HIDDEN_SIZE * 1536 * 4
    parameter_increment_per_layer = (
        5120 * CONV_KERNEL_WIDTH * 2
        + 12 * 4
        + 12 * 2
    )
    capacity_eight_state_increment = int(295.5 * 1024 * 1024)
    reservation = (
        (LINEAR_ATTENTION_LAYERS - 1) * output_increment_per_layer
        + capacity_eight_state_increment
        + (LINEAR_ATTENTION_LAYERS - 1)
        * parameter_increment_per_layer
    )
    projected = (
        LINEAR_ATTENTION_LAYERS * output_increment_per_layer
        + capacity_eight_state_increment
        + LINEAR_ATTENTION_LAYERS * parameter_increment_per_layer
    )
    return {
        "unmeasured_output_projection_bytes": (
            (LINEAR_ATTENTION_LAYERS - 1)
            * output_increment_per_layer
        ),
        "capacity_eight_state_increment_bytes": (
            capacity_eight_state_increment
        ),
        "unmeasured_parameter_increment_bytes": (
            (LINEAR_ATTENTION_LAYERS - 1)
            * parameter_increment_per_layer
        ),
        "reservation_bytes": reservation,
        "projected_integrated_increment_bytes": projected,
    }


def locate_layer_zero_linear_attention(model: object) -> object:
    layer_stack = getattr(model, "layer_stack", None)
    layers = getattr(layer_stack, "layers", None)
    if layers is None or len(layers) < 1:
        raise ValueError("model layer stack is missing layer zero")
    layer = layers[0]
    if getattr(layer, "block_type", None) != "linear_attention":
        raise ValueError("checkpoint layer zero must be linear attention")
    mixer = getattr(layer, "linear_attention", None)
    if mixer is None:
        raise ValueError("linear attention mixer is missing")
    return mixer


def load_logical_tp2_state_parameters(
    model_root: Path,
    *,
    logical_rank: int,
    device: object,
    safe_open_factory=None,
) -> dict[str, object]:
    if logical_rank not in (0, 1):
        raise ValueError("logical_rank must be zero or one")
    root = Path(model_root).resolve()
    index_path = root / "model.safetensors.index.json"
    if not index_path.is_file() or index_path.is_symlink():
        raise ValueError("checkpoint index is missing or unsafe")
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise ValueError("checkpoint index is invalid") from error
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("checkpoint weight_map is invalid")
    prefix = "model.language_model.layers.0.linear_attn."
    source_names = {
        "conv1d.weight": prefix + "conv1d.weight",
        "A_log": prefix + "A_log",
        "dt_bias": prefix + "dt_bias",
    }
    if any(name not in weight_map for name in source_names.values()):
        raise ValueError("checkpoint is missing layer-zero state parameters")
    if safe_open_factory is None:
        from safetensors import safe_open as safe_open_factory

    loaded = {}
    by_shard = {}
    for short_name, source_name in source_names.items():
        by_shard.setdefault(weight_map[source_name], []).append(
            (short_name, source_name)
        )
    for shard_name, requested in sorted(by_shard.items()):
        if (
            not isinstance(shard_name, str)
            or not shard_name
            or Path(shard_name).is_absolute()
        ):
            raise ValueError("checkpoint shard path is invalid")
        shard_path = (root / shard_name).resolve()
        if root not in shard_path.parents:
            raise ValueError("checkpoint shard escapes model root")
        with safe_open_factory(
            shard_path,
            framework="pt",
            device="cpu",
        ) as handle:
            available = set(handle.keys())
            for short_name, source_name in requested:
                if source_name not in available:
                    raise ValueError(
                        f"checkpoint shard is missing {source_name}"
                    )
                loaded[short_name] = handle.get_tensor(source_name)

    conv = loaded["conv1d.weight"]
    if _shape(conv) == (10240, 1, CONV_KERNEL_WIDTH):
        conv = conv.squeeze(1)
    if _shape(conv) != (10240, CONV_KERNEL_WIDTH):
        raise ValueError("checkpoint conv1d.weight shape is invalid")
    if _shape(loaded["A_log"]) != (GLOBAL_VALUE_HEADS,):
        raise ValueError("checkpoint A_log shape is invalid")
    if _shape(loaded["dt_bias"]) != (GLOBAL_VALUE_HEADS,):
        raise ValueError("checkpoint dt_bias shape is invalid")

    import torch

    all_slices = {
        rank: checkpoint_state_tensor_slices(rank)
        for rank in (0, 1)
    }
    logical_conv = {
        rank: torch.cat(tuple(
            _slice_rows(conv, start, length)
            for start, length in all_slices[rank]["conv1d.weight"]
        ), dim=0).contiguous()
        for rank in (0, 1)
    }
    reconstructed_conv = torch.cat((
        _slice_rows(logical_conv[0], 0, 1024),
        _slice_rows(logical_conv[1], 0, 1024),
        _slice_rows(logical_conv[0], 1024, 1024),
        _slice_rows(logical_conv[1], 1024, 1024),
        _slice_rows(logical_conv[0], 2048, 3072),
        _slice_rows(logical_conv[1], 2048, 3072),
    ), dim=0).contiguous()
    logical_A_log = {
        rank: _slice_rows(
            loaded["A_log"],
            *all_slices[rank]["A_log"][0],
        )
        for rank in (0, 1)
    }
    logical_dt_bias = {
        rank: _slice_rows(
            loaded["dt_bias"],
            *all_slices[rank]["dt_bias"][0],
        )
        for rank in (0, 1)
    }
    state_parameter_identity = build_parameter_identity_record(
        candidate_slice_digests={
            "conv_weight": _tensor_digest(logical_conv[logical_rank]),
            "A_log": _tensor_digest(logical_A_log[logical_rank]),
            "dt_bias": _tensor_digest(logical_dt_bias[logical_rank]),
        },
        checkpoint_full_digests={
            "conv_weight": _tensor_digest(conv),
            "A_log": _tensor_digest(loaded["A_log"]),
            "dt_bias": _tensor_digest(loaded["dt_bias"]),
        },
        reconstructed_full_digests={
            "conv_weight": _tensor_digest(reconstructed_conv),
            "A_log": _tensor_digest(torch.cat((
                logical_A_log[0],
                logical_A_log[1],
            ), dim=0).contiguous()),
            "dt_bias": _tensor_digest(torch.cat((
                logical_dt_bias[0],
                logical_dt_bias[1],
            ), dim=0).contiguous()),
        },
    )
    slices = all_slices[logical_rank]
    conv_segments = tuple(
        _slice_rows(conv, start, length)
        for start, length in slices["conv1d.weight"]
    )
    head_start, head_count = slices["A_log"][0]
    dt_start, dt_count = slices["dt_bias"][0]
    return {
        "conv_weight": torch.cat(
            conv_segments,
            dim=0,
        ).to(device=device).contiguous(),
        "A_log": _slice_rows(
            loaded["A_log"],
            head_start,
            head_count,
        ).to(device=device, dtype=torch.float32).contiguous(),
        "dt_bias": _slice_rows(
            loaded["dt_bias"],
            dt_start,
            dt_count,
        ).to(device=device).contiguous(),
        "parameter_identity": state_parameter_identity,
    }


def _positive_integer(value: object, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")
    return value


def build_case_matrix() -> tuple[dict, ...]:
    rows = []
    for active_tokens in ACTIVE_TOKEN_GROUPS:
        for phase, count in (
            ("warmup", WARMUP_PAIR_COUNT),
            ("measured", MEASURED_PAIR_COUNT),
        ):
            for repetition in range(count):
                rows.append({
                    "active_tokens": active_tokens,
                    "phase": phase,
                    "repetition": repetition,
                    "arm_order": (
                        ("baseline", "candidate")
                        if repetition % 2 == 0
                        else ("candidate", "baseline")
                    ),
                    "seed": (
                        2026090800
                        + active_tokens * 100
                        + repetition
                        + (0 if phase == "warmup" else 10)
                    ),
                })
    return tuple(rows)


def validate_case_matrix(cases) -> tuple[dict, ...]:
    try:
        normalized = tuple(dict(row) for row in cases)
    except (TypeError, ValueError) as error:
        raise ValueError("case matrix must match the frozen matrix") from error
    if normalized != build_case_matrix():
        raise ValueError("case matrix must match the frozen matrix")
    return normalized


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
    if checkpoint.keys() != reconstructed.keys() or checkpoint != reconstructed:
        raise ValueError("candidate slices do not reconstruct checkpoint")
    return {
        "parameter_digests": candidate,
        "checkpoint_full_parameter_digests": checkpoint,
        "reconstructed_full_parameter_digests": reconstructed,
        "checkpoint_reconstruction_match": True,
    }


def _tensor_nbytes(tensor: object) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


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
    z_weight: object
    b_weight: object
    a_weight: object
    conv_weight: object
    A_log: object
    dt_bias: object
    norm_weight: object
    norm_eps: float
    output_accumulation_weight: object
    tensor_digests: Mapping[str, str]


def build_logical_tp2_layer_view(
    layer: object,
    logical_rank: int,
    pair_group: object,
) -> LogicalTP2LinearAttentionView:
    if type(layer).__name__ != "Qwen35LinearAttentionShell":
        raise ValueError("layer must be a Qwen35LinearAttentionShell")
    if logical_rank not in (0, 1):
        raise ValueError("logical_rank must be zero or one")
    pinned = (
        getattr(layer, "local_key_heads", None),
        getattr(layer, "local_value_heads", None),
        getattr(layer, "key_head_dim", None),
        getattr(layer, "value_head_dim", None),
    )
    if pinned != (4, 12, HEAD_DIM, HEAD_DIM):
        raise ValueError("layer does not match pinned Qwen3.8 TP4 dimensions")
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
    selected = {
        "qkv_weight": qkv_full,
        "z_weight": z_full,
        "b_weight": b_full,
        "a_weight": a_full,
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
            _slice_rows(conv_source, 4096 + value_width_start, 3072),
        ), dim=0).contiguous()
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
            for name, tensor in sorted(selected.items())
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


def merge_parameter_identity_records(*records: Mapping[str, object]) -> dict:
    candidate = {}
    checkpoint = {}
    reconstructed = {}
    for record in records:
        if record.get("checkpoint_reconstruction_match") is not True:
            raise ValueError("parameter reconstruction proof is incomplete")
        candidate.update(record["parameter_digests"])
        checkpoint.update(record["checkpoint_full_parameter_digests"])
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
            raise RuntimeError("candidate setup must finish before warmup")
        if type(view) is not LogicalTP2LinearAttentionView:
            raise ValueError("candidate view has the wrong type")
        self._views.append(view)

    def mark_warmup_started(self) -> None:
        self._warmup_started = True


class CandidateStateLifecycle:

    def __init__(self) -> None:
        self.published_identity = None

    def publish(
        self,
        *,
        request_id: int,
        generation: int,
        slot_id: int,
        layer_index: int,
    ) -> TopologyLocalTP2StateIdentity:
        identity = TopologyLocalTP2StateIdentity(
            request_id=request_id,
            generation=generation,
            slot_id=slot_id,
            layer_index=layer_index,
        )
        self.published_identity = identity
        return identity

    def require(
        self,
        *,
        request_id: int,
        generation: int,
        slot_id: int,
        layer_index: int,
    ) -> TopologyLocalTP2StateIdentity:
        candidate = TopologyLocalTP2StateIdentity(
            request_id=request_id,
            generation=generation,
            slot_id=slot_id,
            layer_index=layer_index,
        )
        if self.published_identity is None:
            raise RuntimeError("candidate state is not published")
        validate_state_publication(self.published_identity, candidate)
        return candidate

    def unpublish(self) -> None:
        self.published_identity = None


def build_lifecycle_record(
    *,
    state_identity_match: bool,
    stale_generation_rejected: bool,
    different_request_rejected: bool,
    publish_after_success: bool,
    baseline_state_unchanged: bool,
    temporary_state_retired: bool,
    fallback_count: int,
) -> dict:
    proofs = {
        "state_identity_match": state_identity_match,
        "stale_generation_rejected": stale_generation_rejected,
        "different_request_rejected": different_request_rejected,
        "publish_after_success": publish_after_success,
        "baseline_state_unchanged": baseline_state_unchanged,
        "temporary_state_retired": temporary_state_retired,
    }
    if (
        any(type(value) is not bool for value in proofs.values())
        or not all(proofs.values())
        or type(fallback_count) is not int
        or fallback_count != 0
    ):
        raise ValueError("lifecycle proof is incomplete")
    return {**proofs, "fallback_count": fallback_count}


class OwnedWorkerResources:

    def __init__(
        self,
        *,
        process_groups: list,
        tensor_reservations: list,
        destroy_group: Callable[[object], None],
        release_tensor: Callable[[object], None],
        candidate_state_lifecycle: CandidateStateLifecycle,
    ) -> None:
        self._process_groups = list(process_groups)
        self._tensor_reservations = list(tensor_reservations)
        self._destroy_group = destroy_group
        self._release_tensor = release_tensor
        self._state_lifecycle = candidate_state_lifecycle
        self._receipt = None

    def register_tensor(self, tensor: object) -> object:
        if self._receipt is not None:
            raise RuntimeError("worker resources are already closed")
        self._tensor_reservations.append(tensor)
        return tensor

    def close(self, *, failed: bool) -> dict:
        if self._receipt is not None:
            return dict(self._receipt)
        state_was_published = (
            self._state_lifecycle.published_identity is not None
        )
        self._state_lifecycle.unpublish()
        released = 0
        while self._tensor_reservations:
            self._release_tensor(self._tensor_reservations.pop())
            released += 1
        destroyed = 0
        while self._process_groups:
            self._destroy_group(self._process_groups.pop())
            destroyed += 1
        self._receipt = {
            "process_groups_destroyed": destroyed,
            "tensor_reservations_released": released,
            "candidate_state_unpublished": (
                state_was_published
                and self._state_lifecycle.published_identity is None
            ),
            "failed": bool(failed),
        }
        return dict(self._receipt)


def _release_owned_tensor(tensor: object) -> None:
    storage = getattr(tensor, "untyped_storage", None)
    if callable(storage):
        storage().resize_(0)


def _pair_reduce(
    output: object,
    pair_group: object,
    *,
    distributed=None,
) -> object:
    if pair_group is None:
        raise ValueError("pair_group must be explicit")
    if distributed is None:
        import torch.distributed as distributed
    distributed.all_reduce(output, group=pair_group)
    return output


def _apply_candidate_gated_rmsnorm(
    core: object,
    gate: object,
    norm_weight: object,
    *,
    token_count: int,
    logical_parallel_size: int,
    eps: float,
    gated_rmsnorm: Callable,
) -> object:
    if token_count == 1:
        padded_core = core.repeat(logical_parallel_size, 1)
        padded_gate = gate.repeat(logical_parallel_size, 1)
        return gated_rmsnorm(
            padded_core,
            padded_gate,
            norm_weight,
            eps=eps,
        )[:core.shape[0]]
    return gated_rmsnorm(core, gate, norm_weight, eps=eps)


def _run_gated_delta_and_norm(
    convolved,
    projected_z,
    projected_a,
    projected_b,
    recurrent_state,
    view,
):
    from tinyvllm.layers.gated_delta import (
        qwen35_gated_delta_chunk,
        qwen35_gated_delta_recurrent,
        qwen35_gated_rmsnorm,
    )

    token_count = convolved.shape[0]
    key_width = 8 * HEAD_DIM
    value_width = 24 * HEAD_DIM
    query, key, value = convolved.split(
        (key_width, key_width, value_width),
        dim=-1,
    )
    query = query.reshape(
        token_count,
        8,
        HEAD_DIM,
    ).repeat_interleave(3, dim=1)
    key = key.reshape(
        token_count,
        8,
        HEAD_DIM,
    ).repeat_interleave(3, dim=1)
    value = value.reshape(token_count, 24, HEAD_DIM)
    delta_rule = (
        qwen35_gated_delta_recurrent
        if token_count == 1
        else qwen35_gated_delta_chunk
    )
    core, next_recurrent = delta_rule(
        query,
        key,
        value,
        projected_a,
        projected_b,
        view.A_log,
        view.dt_bias,
        recurrent_state,
    )
    norm_core = core.reshape(-1, HEAD_DIM)
    norm_gate = projected_z.reshape(-1, HEAD_DIM)
    gated = _apply_candidate_gated_rmsnorm(
        norm_core,
        norm_gate,
        view.norm_weight,
        token_count=token_count,
        logical_parallel_size=view.logical_parallel_size,
        eps=view.norm_eps,
        gated_rmsnorm=qwen35_gated_rmsnorm,
    )
    return (
        gated.reshape(token_count, value_width),
        next_recurrent,
    )


def run_candidate_mixer(
    hidden,
    convolution_state,
    recurrent_state,
    *,
    view: LogicalTP2LinearAttentionView,
    _event_trace: dict | None = None,
):
    import torch
    import torch.nn.functional as F

    from tinyvllm.layers.gated_delta import (
        qwen35_causal_depthwise_conv,
    )

    if _event_trace is not None:
        _event_trace["projection_start"].record()
    key_start = view.key_head_range[0] * HEAD_DIM
    value_start = view.value_head_range[0] * HEAD_DIM
    qkv_full = F.linear(hidden, view.qkv_weight)
    qkv = torch.cat((
        qkv_full.narrow(-1, key_start, 1024),
        qkv_full.narrow(-1, 2048 + key_start, 1024),
        qkv_full.narrow(-1, 4096 + value_start, 3072),
    ), dim=-1)
    z = F.linear(hidden, view.z_weight).narrow(
        -1,
        value_start,
        3072,
    )
    b = F.linear(hidden, view.b_weight).narrow(
        -1,
        view.value_head_range[0],
        24,
    )
    a = F.linear(hidden, view.a_weight).narrow(
        -1,
        view.value_head_range[0],
        24,
    )
    if _event_trace is not None:
        _event_trace["projection_end"].record()
    convolved, next_convolution = qwen35_causal_depthwise_conv(
        qkv,
        convolution_state,
        view.conv_weight,
    )
    gated, next_recurrent = _run_gated_delta_and_norm(
        convolved,
        z,
        a,
        b,
        recurrent_state,
        view,
    )
    if _event_trace is not None:
        _event_trace["core_end"].record()
    local = F.linear(
        gated.float(),
        view.output_accumulation_weight,
    )
    if _event_trace is not None:
        _event_trace["output_gemm_end"].record()
    _pair_reduce(local, view.pair_group)
    if _event_trace is not None:
        _event_trace["collective_end"].record()
    return local.to(hidden.dtype), next_convolution, next_recurrent


def _run_candidate_diagnostic(
    hidden,
    convolution_state,
    recurrent_state,
    *,
    view: LogicalTP2LinearAttentionView,
) -> dict[str, int]:
    import torch

    events = {
        name: torch.cuda.Event(enable_timing=True)
        for name in (
            "projection_start",
            "projection_end",
            "core_end",
            "output_gemm_end",
            "collective_end",
        )
    }
    run_candidate_mixer(
        hidden,
        convolution_state,
        recurrent_state,
        view=view,
        _event_trace=events,
    )
    events["collective_end"].synchronize()

    def elapsed(left: str, right: str) -> int:
        return int(events[left].elapsed_time(events[right]) * 1_000_000)

    return {
        "input_projection_cuda_ns": elapsed(
            "projection_start",
            "projection_end",
        ),
        "linear_attention_core_cuda_ns": elapsed(
            "projection_end",
            "core_end",
        ),
        "output_projection_gemm_cuda_ns": elapsed(
            "core_end",
            "output_gemm_end",
        ),
        "pair_collective_cuda_ns": elapsed(
            "output_gemm_end",
            "collective_end",
        ),
    }


def build_state_migration_record(
    *,
    latency_ns: int,
    source_bytes: int,
    transferred_bytes: int,
    retained_bytes: int,
    temporary_peak_allocated_bytes: int,
    steady_allocated_bytes: int,
    source_digest: str,
    candidate_digest: str,
    temporary_allocated_bytes_after_release: int,
    temporary_released_before_timing: bool | None = None,
) -> dict:
    integers = {
        "latency_ns": latency_ns,
        "source_bytes": source_bytes,
        "transferred_bytes": transferred_bytes,
        "retained_bytes": retained_bytes,
        "temporary_peak_allocated_bytes": temporary_peak_allocated_bytes,
        "steady_allocated_bytes": steady_allocated_bytes,
        "temporary_allocated_bytes_after_release": (
            temporary_allocated_bytes_after_release
        ),
    }
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        for value in integers.values()
    ):
        raise ValueError("migration counters must be non-negative integers")
    if temporary_allocated_bytes_after_release != 0:
        raise RuntimeError(
            "migration temporary allocation survived steady timing: "
            f"{temporary_allocated_bytes_after_release} bytes"
        )
    for name, digest in (
        ("source_digest", source_digest),
        ("candidate_digest", candidate_digest),
    ):
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    if (
        temporary_released_before_timing is not None
        and temporary_released_before_timing is not True
    ):
        raise RuntimeError(
            "migration temporary allocation survived steady timing"
        )
    return {
        **integers,
        "source_digest": source_digest,
        "candidate_digest": candidate_digest,
        "temporary_released_before_timing": True,
    }


def validate_complete_rank_rows(rows: list[dict]) -> tuple[dict, ...]:
    if not isinstance(rows, (list, tuple)):
        raise ValueError("rank rows must be a sequence")
    ranks = [row.get("rank") for row in rows if isinstance(row, dict)]
    if sorted(ranks) != [0, 1, 2, 3]:
        raise RuntimeError("rank completion must cover ranks 0..3")
    return tuple(dict(row) for row in rows)


def resolve_attempt_output(
    attempt_root: Path,
    relative_name: str,
) -> Path:
    root = Path(attempt_root).resolve()
    if (
        not isinstance(relative_name, str)
        or not relative_name
        or Path(relative_name).is_absolute()
    ):
        raise ValueError("attempt output path is invalid")
    candidate = (root / relative_name).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError("attempt output path is outside attempt root")
    return candidate


def _relative_errors(actual, expected) -> tuple[float, float]:
    difference = (actual.float() - expected.float()).abs()
    denominator = expected.float().abs().clamp_min(1e-12)
    return (
        float(difference.max().item()),
        float((difference / denominator).max().item()),
    )


def _timed_arm(call: Callable[[], tuple]) -> dict:
    import torch

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    submitted_at = time.perf_counter_ns()
    start.record()
    output, convolution, recurrent = call()
    end.record()
    return {
        "output": output,
        "convolution": convolution,
        "recurrent": recurrent,
        "start": start,
        "end": end,
        "host_submission_ns": time.perf_counter_ns() - submitted_at,
    }


def run_mixer_pair(
    *,
    attempt: str,
    case: dict,
    baseline_layer,
    candidate_view: LogicalTP2LinearAttentionView,
    baseline_states: tuple,
    candidate_states: tuple,
    downstream_weight,
) -> dict:
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F

    if not isinstance(attempt, str) or not attempt:
        raise ValueError("attempt must be non-empty")
    if case.get("arm_order") not in (
        ("baseline", "candidate"),
        ("candidate", "baseline"),
    ):
        raise ValueError("case arm_order is invalid")
    hidden = case.get("hidden")
    if hidden is None:
        raise ValueError("case hidden tensor is missing")
    baseline_hidden = hidden.clone()
    candidate_hidden = hidden.clone()
    baseline_convolution = baseline_states[0].clone()
    baseline_recurrent = baseline_states[1].clone()
    candidate_convolution = candidate_states[0].clone()
    candidate_recurrent = candidate_states[1].clone()
    results = {}
    for arm in case["arm_order"]:
        if arm == "baseline":
            results[arm] = _timed_arm(lambda: baseline_layer(
                baseline_hidden,
                baseline_convolution,
                baseline_recurrent,
            ))
        else:
            results[arm] = _timed_arm(lambda: run_candidate_mixer(
                candidate_hidden,
                candidate_convolution,
                candidate_recurrent,
                view=candidate_view,
            ))
    torch.cuda.current_stream().synchronize()
    baseline = results["baseline"]
    candidate = results["candidate"]
    component_diagnostics = _run_candidate_diagnostic(
        hidden.clone(),
        candidate_states[0].clone(),
        candidate_states[1].clone(),
        view=candidate_view,
    )

    pair_groups = case.get("pair_groups", ((0, 1), (2, 3)))
    pair_map = TopologyLocalTP2PairMap(tuple(
        tuple(group) for group in pair_groups
    ))
    rank = (
        int(dist.get_rank())
        if dist.is_available() and dist.is_initialized()
        else int(case.get("rank", 0))
    )
    if dist.is_available() and dist.is_initialized():
        gathered_outputs = [
            torch.empty_like(candidate["output"])
            for _ in range(WORLD_SIZE)
        ]
        gathered_baseline_convolution = [
            torch.empty_like(baseline["convolution"])
            for _ in range(WORLD_SIZE)
        ]
        gathered_baseline_recurrent = [
            torch.empty_like(baseline["recurrent"])
            for _ in range(WORLD_SIZE)
        ]
        gathered_candidate_convolution = [
            torch.empty_like(candidate["convolution"])
            for _ in range(WORLD_SIZE)
        ]
        gathered_candidate_recurrent = [
            torch.empty_like(candidate["recurrent"])
            for _ in range(WORLD_SIZE)
        ]
        for local, gathered in (
            (candidate["output"], gathered_outputs),
            (baseline["convolution"], gathered_baseline_convolution),
            (baseline["recurrent"], gathered_baseline_recurrent),
            (candidate["convolution"], gathered_candidate_convolution),
            (candidate["recurrent"], gathered_candidate_recurrent),
        ):
            dist.all_gather(gathered, local)
    else:
        gathered_outputs = [candidate["output"]] * WORLD_SIZE
        gathered_baseline_convolution = [
            baseline["convolution"]
        ] * WORLD_SIZE
        gathered_baseline_recurrent = [baseline["recurrent"]] * WORLD_SIZE
        gathered_candidate_convolution = [
            candidate["convolution"]
        ] * WORLD_SIZE
        gathered_candidate_recurrent = [
            candidate["recurrent"]
        ] * WORLD_SIZE

    first_pair, second_pair = pair_map.pair_groups
    pair_output_abs, pair_output_rel = _relative_errors(
        gathered_outputs[first_pair[0]],
        gathered_outputs[second_pair[0]],
    )
    pair_convolution_errors = []
    pair_recurrent_errors = []
    for logical_rank in (0, 1):
        left_rank = first_pair[logical_rank]
        right_rank = second_pair[logical_rank]
        pair_convolution_errors.append(_relative_errors(
            gathered_candidate_convolution[left_rank],
            gathered_candidate_convolution[right_rank],
        ))
        pair_recurrent_errors.append(_relative_errors(
            gathered_candidate_recurrent[left_rank],
            gathered_candidate_recurrent[right_rank],
        ))
    baseline_full_convolution = torch.cat(
        gathered_baseline_convolution,
        dim=0,
    )
    baseline_full_recurrent = torch.cat(
        gathered_baseline_recurrent,
        dim=0,
    )
    candidate_full_convolution = torch.cat((
        gathered_candidate_convolution[first_pair[0]],
        gathered_candidate_convolution[first_pair[1]],
    ), dim=0)
    candidate_full_recurrent = torch.cat((
        gathered_candidate_recurrent[first_pair[0]],
        gathered_candidate_recurrent[first_pair[1]],
    ), dim=0)
    output_abs, output_rel = _relative_errors(
        candidate["output"],
        baseline["output"],
    )
    convolution_abs, convolution_rel = _relative_errors(
        candidate_full_convolution,
        baseline_full_convolution,
    )
    recurrent_abs, recurrent_rel = _relative_errors(
        candidate_full_recurrent,
        baseline_full_recurrent,
    )
    baseline_logits = F.linear(
        baseline["output"].float(),
        downstream_weight.float(),
    )
    candidate_logits = F.linear(
        candidate["output"].float(),
        downstream_weight.float(),
    )
    return {
        "schema": WORKER_SCHEMA,
        "attempt": attempt,
        "active_tokens": case["active_tokens"],
        "phase": case["phase"],
        "repetition": case["repetition"],
        "arm_order": list(case["arm_order"]),
        "rank": rank,
        "pair_id": pair_map.identity(rank).pair_id,
        "logical_rank": candidate_view.logical_rank,
        "baseline_cuda_ns": int(
            baseline["start"].elapsed_time(baseline["end"]) * 1_000_000
        ),
        "candidate_cuda_ns": int(
            candidate["start"].elapsed_time(candidate["end"]) * 1_000_000
        ),
        "baseline_host_submission_ns": baseline["host_submission_ns"],
        "candidate_host_submission_ns": candidate["host_submission_ns"],
        "candidate_component_diagnostics": component_diagnostics,
        "output_max_abs_error": output_abs,
        "output_max_rel_error": output_rel,
        "convolution_max_abs_error": convolution_abs,
        "convolution_max_rel_error": convolution_rel,
        "recurrent_max_abs_error": recurrent_abs,
        "recurrent_max_rel_error": recurrent_rel,
        "pair_replica_output_max_abs_error": pair_output_abs,
        "pair_replica_output_max_rel_error": pair_output_rel,
        "pair_replica_convolution_max_abs_error": max(
            row[0] for row in pair_convolution_errors
        ),
        "pair_replica_convolution_max_rel_error": max(
            row[1] for row in pair_convolution_errors
        ),
        "pair_replica_recurrent_max_abs_error": max(
            row[0] for row in pair_recurrent_errors
        ),
        "pair_replica_recurrent_max_rel_error": max(
            row[1] for row in pair_recurrent_errors
        ),
        "output_within_tolerance": bool(torch.allclose(
            candidate["output"],
            baseline["output"],
            atol=2e-2,
            rtol=2e-3,
        )),
        "convolution_within_tolerance": bool(torch.allclose(
            candidate_full_convolution,
            baseline_full_convolution,
            atol=2e-2,
            rtol=2e-3,
        )),
        "recurrent_within_tolerance": bool(torch.allclose(
            candidate_full_recurrent,
            baseline_full_recurrent,
            atol=2e-2,
            rtol=2e-3,
        )),
        "pair_replicas_within_tolerance": (
            pair_output_abs <= 2e-4
            and pair_output_rel <= 2e-4
            and all(
                absolute <= 2e-4 and relative <= 2e-4
                for absolute, relative in (
                    *pair_convolution_errors,
                    *pair_recurrent_errors,
                )
            )
        ),
        "greedy_argmax_equal": bool(torch.equal(
            candidate_logits.argmax(dim=-1),
            baseline_logits.argmax(dim=-1),
        )),
        "finite": all(bool(torch.isfinite(tensor).all().item()) for tensor in (
            candidate["output"],
            candidate["convolution"],
            candidate["recurrent"],
        )),
        "parameter_digests": dict(candidate_view.tensor_digests),
    }


def _atomic_write_json(path: Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".partial",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(
            payload,
            handle,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_write_jsonl(path: Path, rows: tuple[dict, ...]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".partial",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ))
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def build_failure_record(
    error: BaseException,
    *,
    stage: str,
) -> dict[str, str]:
    if not isinstance(stage, str) or not stage:
        raise ValueError("failure stage must be a non-empty string")
    return {
        "type": type(error).__name__,
        "message": str(error),
        "stage": stage,
        "traceback": "".join(traceback.format_exception(
            type(error),
            error,
            error.__traceback__,
        )),
    }


def _load_checkpoint_model(model_root: Path, rank: int):
    import torch

    from tinyvllm.config import Config
    from tinyvllm.engine.model_runner import (
        _load_qwen35_model_runner_model,
        _resolve_qwen38_text_profile,
    )
    from tinyvllm.models.qwen38_text_adopter import (
        adopt_qwen38_text_config,
        read_qwen38_source_identity,
    )

    config = Config(
        model=str(Path(model_root).resolve()),
        tensor_parallel_size=WORLD_SIZE,
        max_num_seqs=STATE_CAPACITY,
        enforce_eager=True,
        kv_offload_mvp0=False,
    )
    profile = _resolve_qwen38_text_profile(
        config.hf_config,
        model_dir=config.model,
        adopt_qwen38_text=adopt_qwen38_text_config,
        read_source_identity=read_qwen38_source_identity,
    )
    if profile is None:
        raise ValueError("checkpoint is not the pinned Qwen3.8 topology")
    default_dtype = torch.get_default_dtype()
    default_device = (
        torch.get_default_device()
        if hasattr(torch, "get_default_device")
        else "cpu"
    )
    try:
        torch.set_default_dtype(torch.bfloat16)
        torch.set_default_device("cuda")
        return _load_qwen35_model_runner_model(
            config,
            rank,
            qwen38_text_profile=profile,
        )
    finally:
        torch.set_default_dtype(default_dtype)
        torch.set_default_device(default_device)


def case_seeds(seed: int, rank: int) -> dict[str, int]:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank not in range(4):
        raise ValueError("rank must be in [0, 3]")
    return {
        "hidden": seed,
        "state": seed + rank * 100_000,
    }


def _make_case_inputs(
    *,
    active_tokens: int,
    seed: int,
    rank: int,
    device: object,
):
    import torch

    seeds = case_seeds(seed, rank)
    hidden_generator = torch.Generator(device=device)
    hidden_generator.manual_seed(seeds["hidden"])
    state_generator = torch.Generator(device=device)
    state_generator.manual_seed(seeds["state"])
    hidden = torch.randn(
        (active_tokens, HIDDEN_SIZE),
        generator=hidden_generator,
        dtype=torch.bfloat16,
        device=device,
    )
    convolution = torch.randn(
        (2560, CONV_KERNEL_WIDTH),
        generator=state_generator,
        dtype=torch.bfloat16,
        device=device,
    )
    recurrent = torch.randn(
        (12, HEAD_DIM, HEAD_DIM),
        generator=state_generator,
        dtype=torch.float32,
        device=device,
    )
    hidden.mul_(0.02)
    convolution.mul_(0.02)
    recurrent.mul_(0.02)
    return hidden, convolution, recurrent


def _migrate_state_once(
    *,
    local_convolution,
    local_recurrent,
    logical_rank: int,
    distributed,
    device: object,
) -> tuple[tuple, dict]:
    import gc
    import torch

    before_allocated = int(torch.cuda.memory_allocated(device))
    torch.cuda.reset_peak_memory_stats(device)
    convolution_quarters = [
        torch.empty_like(local_convolution)
        for _ in range(WORLD_SIZE)
    ]
    recurrent_quarters = [
        torch.empty_like(local_recurrent)
        for _ in range(WORLD_SIZE)
    ]
    started = torch.cuda.Event(enable_timing=True)
    completed = torch.cuda.Event(enable_timing=True)
    started.record()
    distributed.all_gather(
        convolution_quarters,
        local_convolution,
    )
    distributed.all_gather(
        recurrent_quarters,
        local_recurrent,
    )
    candidate_convolution = assemble_logical_state_half(
        tuple(convolution_quarters),
        logical_rank,
    )
    candidate_recurrent = assemble_logical_state_half(
        tuple(recurrent_quarters),
        logical_rank,
    )
    completed.record()
    completed.synchronize()
    latency_ns = int(started.elapsed_time(completed) * 1_000_000)
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    source_digest = hashlib.sha256(
        "".join(
            _tensor_digest(tensor)
            for tensor in (
                *convolution_quarters,
                *recurrent_quarters,
            )
        ).encode("ascii")
    ).hexdigest()
    candidate_digest = hashlib.sha256(
        (
            _tensor_digest(candidate_convolution)
            + _tensor_digest(candidate_recurrent)
        ).encode("ascii")
    ).hexdigest()
    source_bytes = (
        _tensor_nbytes(local_convolution)
        + _tensor_nbytes(local_recurrent)
    )
    retained_bytes = (
        _tensor_nbytes(candidate_convolution)
        + _tensor_nbytes(candidate_recurrent)
    )
    del convolution_quarters
    del recurrent_quarters
    gc.collect()
    after_release = int(torch.cuda.memory_allocated(device))
    temporary_after_release = max(
        0,
        after_release - before_allocated - retained_bytes,
    )
    record = build_state_migration_record(
        latency_ns=latency_ns,
        source_bytes=source_bytes,
        transferred_bytes=source_bytes * (WORLD_SIZE - 1),
        retained_bytes=retained_bytes,
        temporary_peak_allocated_bytes=max(
            0,
            peak_allocated - before_allocated,
        ),
        steady_allocated_bytes=max(0, after_release - before_allocated),
        source_digest=source_digest,
        candidate_digest=candidate_digest,
        temporary_allocated_bytes_after_release=temporary_after_release,
    )
    return (candidate_convolution, candidate_recurrent), record


def _runtime_capability(rank: int, device: object) -> dict:
    import socket
    import torch
    import torch.distributed as dist

    properties = torch.cuda.get_device_properties(device)
    return {
        "rank": rank,
        "hostname": socket.gethostname(),
        "device_name": properties.name,
        "device_uuid": str(getattr(properties, "uuid", "")),
        "physical_memory_bytes": int(properties.total_memory),
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda),
        "nccl_available": bool(dist.is_nccl_available()),
    }


def run_worker_campaign(
    *,
    attempt: str,
    source_revision: str,
    model_root: Path,
    pair_groups: tuple[tuple[int, int], tuple[int, int]],
    output_root: Path,
    cases: tuple[dict, ...],
) -> dict:
    """Execute a rank-local campaign.

    CUDA/checkpoint imports stay inside this function so the frozen contracts
    remain unit-testable on a CPU-only control host.
    """
    import torch
    import torch.distributed as dist

    if len(source_revision) != 40:
        raise ValueError("source_revision must be a full Git SHA")
    pair_map = TopologyLocalTP2PairMap(pair_groups)
    cases = validate_case_matrix(cases)
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != WORLD_SIZE or rank not in range(WORLD_SIZE):
        raise ValueError("worker requires exactly four valid ranks")
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl")
    created_groups = [
        dist.new_group(ranks=list(group))
        for group in pair_groups
    ]
    identity = pair_map.identity(rank)
    pair_group = created_groups[identity.pair_id]
    state_lifecycle = CandidateStateLifecycle()
    resources = OwnedWorkerResources(
        process_groups=[None, pair_group],
        tensor_reservations=[],
        destroy_group=lambda group: (
            dist.destroy_process_group()
            if group is None
            else dist.destroy_process_group(group)
        ),
        release_tensor=_release_owned_tensor,
        candidate_state_lifecycle=state_lifecycle,
    )
    failed = True
    local_rows = []
    migration_rows = []
    memory_row = {}
    capability = _runtime_capability(rank, device)
    failure = None
    lifecycle_proofs = None
    stage = "checkpoint_model_load"
    try:
        load_start_allocated = int(torch.cuda.memory_allocated(device))
        model, owner, partition_identity = _load_checkpoint_model(
            model_root,
            rank,
        )
        stage = "layer_zero_lookup"
        mixer = locate_layer_zero_linear_attention(model)
        stage = "state_parameter_load"
        state_parameters = load_logical_tp2_state_parameters(
            model_root,
            logical_rank=identity.logical_rank,
            device=device,
        )
        state_parameter_identity = state_parameters.pop(
            "parameter_identity"
        )
        mixer.logical_tp2_conv_weight = state_parameters["conv_weight"]
        mixer.logical_tp2_A_log = state_parameters["A_log"]
        mixer.logical_tp2_dt_bias = state_parameters["dt_bias"]
        setup = CandidateSetupLifecycle()
        stage = "candidate_view_build"
        view = build_logical_tp2_layer_view(
            mixer,
            identity.logical_rank,
            pair_group,
        )
        stage = "parameter_identity"
        parameter_identity = merge_parameter_identity_records(
            build_layer_parameter_identity(mixer, view),
            state_parameter_identity,
        )
        setup.register_candidate_view(view)
        del mixer.logical_tp2_conv_weight
        del mixer.logical_tp2_A_log
        del mixer.logical_tp2_dt_bias
        del state_parameters
        for tensor in (
            view.conv_weight,
            view.A_log,
            view.dt_bias,
            view.output_accumulation_weight,
        ):
            resources.register_tensor(tensor)

        stage = "persistent_reservation"
        reservation_contract = project_persistent_reservation()
        reservation = resources.register_tensor(torch.empty(
            reservation_contract["reservation_bytes"],
            dtype=torch.uint8,
            device=device,
        ))
        downstream_generator = torch.Generator(device=device)
        downstream_generator.manual_seed(2026090899)
        downstream_weight = resources.register_tensor(torch.randn(
            (256, HIDDEN_SIZE),
            generator=downstream_generator,
            dtype=torch.bfloat16,
            device=device,
        ))

        migration_state = None
        baseline_state = None
        stage = "state_migration"
        for phase, count in (
            ("warmup", WARMUP_PAIR_COUNT),
            ("measured", MEASURED_PAIR_COUNT),
        ):
            for repetition in range(count):
                _, local_convolution, local_recurrent = _make_case_inputs(
                    active_tokens=1,
                    seed=2026090900 + repetition,
                    rank=rank,
                    device=device,
                )
                migration_state, migration = _migrate_state_once(
                    local_convolution=local_convolution,
                    local_recurrent=local_recurrent,
                    logical_rank=identity.logical_rank,
                    distributed=dist,
                    device=device,
                )
                baseline_state = (local_convolution, local_recurrent)
                if phase == "measured":
                    migration_rows.append({
                        "schema": WORKER_SCHEMA,
                        "attempt": attempt,
                        "source_revision": source_revision,
                        "phase": phase,
                        "repetition": repetition,
                        "rank": rank,
                        "pair_id": identity.pair_id,
                        "logical_rank": identity.logical_rank,
                        **migration,
                    })

        if migration_state is None or baseline_state is None:
            raise RuntimeError("state migration matrix produced no state")
        baseline_state_digest_before = tuple(
            _tensor_digest(tensor) for tensor in baseline_state
        )
        torch.cuda.reset_peak_memory_stats(device)
        setup.mark_warmup_started()
        last_case = None
        stage = "paired_measurement"
        for frozen_case in cases:
            last_case = frozen_case
            hidden, _, _ = _make_case_inputs(
                active_tokens=frozen_case["active_tokens"],
                seed=frozen_case["seed"],
                rank=rank,
                device=device,
            )
            case = {
                **frozen_case,
                "hidden": hidden,
                "rank": rank,
                "pair_groups": pair_groups,
            }
            row = run_mixer_pair(
                attempt=attempt,
                case=case,
                baseline_layer=mixer,
                candidate_view=view,
                baseline_states=baseline_state,
                candidate_states=migration_state,
                downstream_weight=downstream_weight,
            )
            correctness_fields = (
                "output_within_tolerance",
                "convolution_within_tolerance",
                "recurrent_within_tolerance",
                "pair_replicas_within_tolerance",
                "greedy_argmax_equal",
                "finite",
            )
            if not all(row[field] is True for field in correctness_fields):
                raise RuntimeError(
                    "mixer correctness or lifecycle gate failed"
                )
            state_lifecycle.publish(
                request_id=frozen_case["active_tokens"],
                generation=(
                    frozen_case["active_tokens"] * 100
                    + frozen_case["repetition"]
                ),
                slot_id=0,
                layer_index=0,
            )
            if frozen_case["phase"] == "measured":
                local_rows.append({
                    **row,
                    **parameter_identity,
                    "source_revision": source_revision,
                    "physical_device_uuid": capability["device_uuid"],
                })
        if last_case is None:
            raise RuntimeError("worker campaign produced no cases")
        request_id = last_case["active_tokens"]
        generation = (
            last_case["active_tokens"] * 100
            + last_case["repetition"]
        )
        exact_identity = state_lifecycle.require(
            request_id=request_id,
            generation=generation,
            slot_id=0,
            layer_index=0,
        )
        stale_generation_rejected = False
        different_request_rejected = False
        try:
            state_lifecycle.require(
                request_id=request_id,
                generation=generation + 1,
                slot_id=0,
                layer_index=0,
            )
        except RuntimeError:
            stale_generation_rejected = True
        try:
            state_lifecycle.require(
                request_id=request_id + 1,
                generation=generation,
                slot_id=0,
                layer_index=0,
            )
        except RuntimeError:
            different_request_rejected = True
        lifecycle_proofs = {
            "state_identity_match": (
                exact_identity == state_lifecycle.published_identity
            ),
            "stale_generation_rejected": stale_generation_rejected,
            "different_request_rejected": different_request_rejected,
            "publish_after_success": True,
            "baseline_state_unchanged": (
                tuple(_tensor_digest(tensor) for tensor in baseline_state)
                == baseline_state_digest_before
            ),
        }
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        memory_row = {
            "schema": WORKER_SCHEMA,
            "attempt": attempt,
            "source_revision": source_revision,
            "rank": rank,
            "load_start_allocated_bytes": load_start_allocated,
            "steady_allocated_bytes": int(
                torch.cuda.memory_allocated(device)
            ),
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "physical_memory_bytes": capability[
                "physical_memory_bytes"
            ],
            "peak_allocated_ratio": (
                peak_allocated / capability["physical_memory_bytes"]
            ),
            **reservation_contract,
        }
        stage = "lifecycle_finalize"
        failed = False
    except BaseException as error:
        failure = build_failure_record(error, stage=stage)
    finally:
        cleanup = resources.close(failed=failed)
    _atomic_write_jsonl(
        resolve_attempt_output(
            output_root,
            f"measurement_rows.rank-{rank}.jsonl",
        ),
        tuple(local_rows),
    )
    _atomic_write_jsonl(
        resolve_attempt_output(
            output_root,
            f"migration_rows.rank-{rank}.jsonl",
        ),
        tuple(migration_rows),
    )
    _atomic_write_json(
        resolve_attempt_output(output_root, f"memory.rank-{rank}.json"),
        memory_row,
    )
    _atomic_write_json(
        resolve_attempt_output(
            output_root,
            f"capability.rank-{rank}.json",
        ),
        capability,
    )
    cleanup = {
        **cleanup,
        "rank": rank,
        "classification": "CLEAN" if not failed else "FAILED",
        "failure": failure,
    }
    lifecycle_row = {
        "schema": WORKER_SCHEMA,
        "attempt": attempt,
        "source_revision": source_revision,
        "rank": rank,
    }
    if lifecycle_proofs is not None and not failed:
        lifecycle_row.update(build_lifecycle_record(
            **lifecycle_proofs,
            temporary_state_retired=(
                cleanup["candidate_state_unpublished"] is True
            ),
            fallback_count=0,
        ))
    else:
        lifecycle_row.update({
            "state_identity_match": False,
            "stale_generation_rejected": False,
            "different_request_rejected": False,
            "publish_after_success": False,
            "baseline_state_unchanged": False,
            "temporary_state_retired": (
                cleanup["candidate_state_unpublished"] is True
            ),
            "fallback_count": 0,
        })
    _atomic_write_json(
        resolve_attempt_output(output_root, f"cleanup.rank-{rank}.json"),
        cleanup,
    )
    _atomic_write_json(
        resolve_attempt_output(output_root, f"lifecycle.rank-{rank}.json"),
        lifecycle_row,
    )
    result = {
        "schema": WORKER_SCHEMA,
        "attempt": attempt,
        "source_revision": source_revision,
        "rank": rank,
        "rows": local_rows,
        "migration_rows": migration_rows,
        "memory": memory_row,
        "capability": capability,
        "cleanup": cleanup,
        "lifecycle": lifecycle_row,
    }
    if failure is not None:
        raise RuntimeError(
            f"rank {rank} campaign failed: {failure['type']}: "
            f"{failure['message']}"
        )
    return result


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--pair-groups", required=True)
    return parser


def main(argv=None) -> int:
    args = build_argument_parser().parse_args(argv)
    pair_groups = tuple(
        tuple(int(rank) for rank in group.split(","))
        for group in args.pair_groups.split(";")
    )
    run_worker_campaign(
        attempt=args.attempt,
        source_revision=args.source_revision,
        model_root=args.model_root,
        pair_groups=pair_groups,
        output_root=args.output_root,
        cases=build_case_matrix(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
