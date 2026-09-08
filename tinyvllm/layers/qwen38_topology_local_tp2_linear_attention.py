from __future__ import annotations

from dataclasses import dataclass
import gc
import hashlib
import math
import re
from types import MappingProxyType
from typing import Mapping
import weakref

import torch


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


def candidate_gated_delta_chunk_size(token_count: int) -> int:
    if (
        isinstance(token_count, bool)
        or not isinstance(token_count, int)
        or token_count < 2
    ):
        raise ValueError(
            "multi-token chunk size requires token_count >= 2"
        )
    return token_count if token_count <= 8 else 64


@dataclass(frozen=True)
class _LogicalTP2Projection:
    convolved: object
    gate: object
    projected_a: object
    projected_b: object
    next_convolution: object


class Qwen38TopologyLocalTP2LinearAttention(torch.nn.Module):

    def __init__(
        self,
        *,
        baseline: object,
        candidate_view: LogicalTP2LinearAttentionView,
        pair_reduce,
    ):
        super().__init__()
        if not callable(baseline):
            raise ValueError("baseline must be callable")
        if not callable(pair_reduce):
            raise ValueError("pair_reduce must be callable")
        if getattr(candidate_view, "pair_group", None) is None:
            raise ValueError("candidate_view pair_group must be explicit")
        self.baseline = baseline
        self.candidate_view = candidate_view
        self._pair_reduce = pair_reduce
        self._phase = "tp4_prefill"
        self._telemetry = {
            "tp4_prefill_calls": 0,
            "tp2_decode_calls": 0,
            "recurrent_token_one_calls": 0,
            "short_chunk_calls": 0,
            "chunk_64_calls": 0,
            "pair_local_all_reduce_calls": 0,
            "global_tp4_decode_all_reduce_calls": 0,
            "phase_transition_count": 0,
        }
        self._correctness_trace_enabled = False
        self._last_output_digest = None

    @property
    def phase(self) -> str:
        return self._phase

    def activate_tp2_decode(self) -> None:
        if self._phase != "tp4_prefill":
            raise RuntimeError("TP2 decode phase is already active")
        self._phase = "tp2_decode"
        self._telemetry["phase_transition_count"] += 1

    def activate_tp4_prefill(self) -> None:
        if self._phase != "tp2_decode":
            raise RuntimeError("TP2 decode phase is not active")
        self._phase = "tp4_prefill"

    def _project_logical_tp2(
        self,
        hidden_states,
        convolution_state,
    ) -> _LogicalTP2Projection:
        import torch.nn.functional as F

        from tinyvllm.layers.gated_delta import (
            qwen35_causal_depthwise_conv,
        )

        view = self.candidate_view
        qkv_full = F.linear(hidden_states, view.qkv_weight)
        key_width_start = view.key_head_range[0] * HEAD_DIM
        value_width_start = view.value_head_range[0] * HEAD_DIM
        qkv = torch.cat((
            qkv_full.narrow(-1, key_width_start, 1024),
            qkv_full.narrow(-1, 2048 + key_width_start, 1024),
            qkv_full.narrow(-1, 4096 + value_width_start, 3072),
        ), dim=-1)
        gate = F.linear(hidden_states, view.z_weight).narrow(
            -1,
            value_width_start,
            3072,
        )
        projected_a, projected_b = F.linear(
            hidden_states,
            view.ab_weight_half,
        ).split((24, 24), dim=-1)
        convolved, next_convolution = qwen35_causal_depthwise_conv(
            qkv,
            convolution_state,
            view.conv_weight,
        )
        return _LogicalTP2Projection(
            convolved=convolved,
            gate=gate,
            projected_a=projected_a,
            projected_b=projected_b,
            next_convolution=next_convolution,
        )

    def _run_delta(
        self,
        projected: _LogicalTP2Projection,
        recurrent_state,
        *,
        token_count: int,
    ) -> tuple[object, object]:
        from tinyvllm.layers.gated_delta import (
            qwen35_gated_delta_chunk,
            qwen35_gated_delta_recurrent,
            qwen35_gated_rmsnorm,
        )

        view = self.candidate_view
        key_width = 8 * HEAD_DIM
        value_width = 24 * HEAD_DIM
        query, key, value = projected.convolved.split(
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
        if token_count == 1:
            core, next_recurrent = qwen35_gated_delta_recurrent(
                query,
                key,
                value,
                projected.projected_a,
                projected.projected_b,
                view.A_log,
                view.dt_bias,
                recurrent_state,
            )
        else:
            core, next_recurrent = qwen35_gated_delta_chunk(
                query,
                key,
                value,
                projected.projected_a,
                projected.projected_b,
                view.A_log,
                view.dt_bias,
                recurrent_state,
                chunk_size=candidate_gated_delta_chunk_size(
                    token_count
                ),
            )
        norm_core = core.reshape(-1, HEAD_DIM)
        norm_gate = projected.gate.reshape(-1, HEAD_DIM)
        if token_count == 1:
            norm_core = norm_core.repeat(
                view.logical_parallel_size,
                1,
            )
            norm_gate = norm_gate.repeat(
                view.logical_parallel_size,
                1,
            )
            gated = qwen35_gated_rmsnorm(
                norm_core,
                norm_gate,
                view.norm_weight,
                eps=view.norm_eps,
            )[:token_count]
        else:
            gated = qwen35_gated_rmsnorm(
                norm_core,
                norm_gate,
                view.norm_weight,
                eps=view.norm_eps,
            )
        return (
            gated.reshape(token_count, value_width),
            next_recurrent,
        )

    def _output_projection(self, core, gate):
        del gate
        import torch.nn.functional as F

        return F.linear(
            core.float(),
            self.candidate_view.output_accumulation_weight,
        )

    def forward(
        self,
        hidden_states,
        convolution_state,
        recurrent_state,
    ):
        if self._phase == "tp4_prefill":
            self._telemetry["tp4_prefill_calls"] += 1
            return self.baseline(
                hidden_states,
                convolution_state,
                recurrent_state,
            )
        if self._phase != "tp2_decode":
            raise RuntimeError(
                f"unsupported linear-attention phase: {self._phase}"
            )
        try:
            token_count = int(hidden_states.shape[0])
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError(
                "hidden_states must expose a valid token count"
            ) from error
        if token_count <= 0:
            raise ValueError("hidden_states token count must be positive")

        projected = self._project_logical_tp2(
            hidden_states,
            convolution_state,
        )
        core, next_recurrent = self._run_delta(
            projected,
            recurrent_state,
            token_count=token_count,
        )
        local = self._output_projection(core, projected.gate)
        self._pair_reduce(local, self.candidate_view.pair_group)
        output = local.to(dtype=hidden_states.dtype)
        if self._correctness_trace_enabled:
            self._last_output_digest = output_digest(output)

        self._telemetry["tp2_decode_calls"] += 1
        self._telemetry["pair_local_all_reduce_calls"] += 1
        if token_count == 1:
            self._telemetry["recurrent_token_one_calls"] += 1
        elif token_count <= 8:
            self._telemetry["short_chunk_calls"] += 1
        else:
            self._telemetry["chunk_64_calls"] += 1
        return output, projected.next_convolution, next_recurrent

    def enable_correctness_trace(self, enabled: bool) -> None:
        self._correctness_trace_enabled = bool(enabled)
        self._last_output_digest = None

    def correctness_output_digest(self) -> str:
        if (
            not self._correctness_trace_enabled
            or self._last_output_digest is None
        ):
            raise RuntimeError(
                "candidate correctness output is unavailable"
            )
        return self._last_output_digest

    def telemetry_snapshot(self) -> dict:
        return {
            "schema_version":
                "qwen38.topology-local-tp2-linear-attention.v1",
            "phase": self._phase,
            **self._telemetry,
        }


def output_digest(output: torch.Tensor) -> str:
    if output.dtype is not torch.bfloat16:
        raise ValueError("candidate output digest requires BF16")
    return hashlib.sha256(
        output.detach()
        .contiguous()
        .view(torch.uint8)
        .cpu()
        .numpy()
        .tobytes()
    ).hexdigest()


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
