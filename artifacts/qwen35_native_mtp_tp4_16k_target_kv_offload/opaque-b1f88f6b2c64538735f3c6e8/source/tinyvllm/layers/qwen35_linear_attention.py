from dataclasses import dataclass
import math

import torch
from torch import nn

from tinyvllm.layers.gated_delta import (
    qwen35_causal_depthwise_conv,
    qwen35_causal_depthwise_conv_prefix_trace,
    qwen35_gated_delta_chunk,
    qwen35_gated_delta_prefix_trace,
    qwen35_gated_delta_recurrent,
    qwen35_gated_rmsnorm,
)
from tinyvllm.utils.context import get_context


@dataclass(frozen=True)
class Qwen35LinearStateTrace:
    convolution: torch.Tensor
    recurrent: torch.Tensor


class Qwen35LinearAttentionShell(nn.Module):

    def __init__(
        self,
        *,
        local_key_heads: int,
        local_value_heads: int,
        key_head_dim: int,
        value_head_dim: int,
        norm_eps: float,
        in_proj_qkv: nn.Module,
        in_proj_z: nn.Module,
        in_proj_b: nn.Module,
        in_proj_a: nn.Module,
        out_proj: nn.Module,
        conv_weight: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        norm_weight: torch.Tensor,
    ):
        super().__init__()
        for name, value in (
            ("local_key_heads", local_key_heads),
            ("local_value_heads", local_value_heads),
            ("key_head_dim", key_head_dim),
            ("value_head_dim", value_head_dim),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer")
        if local_value_heads % local_key_heads != 0:
            raise ValueError(
                "local_value_heads must be divisible by local_key_heads"
            )
        if (
            isinstance(norm_eps, bool)
            or not isinstance(norm_eps, (int, float))
            or not math.isfinite(norm_eps)
            or norm_eps <= 0
        ):
            raise ValueError("norm_eps must be a positive finite number")

        self.local_key_heads = local_key_heads
        self.local_value_heads = local_value_heads
        self.key_head_dim = key_head_dim
        self.value_head_dim = value_head_dim
        self.norm_eps = float(norm_eps)
        self.in_proj_qkv = in_proj_qkv
        self.in_proj_z = in_proj_z
        self.in_proj_b = in_proj_b
        self.in_proj_a = in_proj_a
        self.out_proj = out_proj
        self.register_buffer("conv_weight", conv_weight)
        self.register_buffer("A_log", A_log)
        self.register_buffer("dt_bias", dt_bias)
        self.register_buffer("norm_weight", norm_weight)

        key_width = local_key_heads * key_head_dim
        value_width = local_value_heads * value_head_dim
        conv_width = 2 * key_width + value_width
        if conv_weight.ndim != 2 or conv_weight.shape[0] != conv_width:
            raise ValueError(
                "conv_weight must have [conv_width, history_width] shape"
            )
        if conv_weight.shape[1] <= 0:
            raise ValueError("conv_weight history width must be positive")
        if A_log.shape != (local_value_heads,):
            raise ValueError("A_log must match local_value_heads")
        if dt_bias.shape != (local_value_heads,):
            raise ValueError("dt_bias must match local_value_heads")
        if norm_weight.shape != (value_head_dim,):
            raise ValueError("norm_weight must match value_head_dim")
        for name, tensor in (
            ("conv_weight", conv_weight),
            ("A_log", A_log),
            ("dt_bias", dt_bias),
            ("norm_weight", norm_weight),
        ):
            if not tensor.is_floating_point():
                raise ValueError(f"{name} must use a floating point dtype")
        if conv_weight.dtype != dt_bias.dtype:
            raise ValueError(
                "linear-attention compute parameter dtype must match"
            )
        if not (
            conv_weight.device
            == A_log.device
            == dt_bias.device
            == norm_weight.device
        ):
            raise ValueError("linear-attention parameter device must match")

    @staticmethod
    def _validate_projection(
        output: torch.Tensor,
        *,
        name: str,
        token_count: int,
        feature_count: int,
        reference: torch.Tensor,
    ) -> None:
        if not isinstance(output, torch.Tensor):
            raise ValueError(f"{name} output must be a tensor")
        if output.ndim != 2:
            raise ValueError(f"{name} output must be rank two")
        if not output.is_floating_point():
            raise ValueError(
                f"{name} output must use a floating point dtype"
            )
        if output.shape[0] != token_count:
            raise ValueError(
                f"{name} token count must equal {token_count}"
            )
        if output.shape[1] != feature_count:
            raise ValueError(
                f"{name} feature dimension must equal {feature_count}"
            )
        if output.dtype != reference.dtype:
            raise ValueError(f"{name} dtype must match hidden_states")
        if output.device != reference.device:
            raise ValueError(f"{name} device must match hidden_states")

    def _project_context_rows(
        self,
        projection: nn.Module,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        context = get_context()
        forward = projection
        prefill_forward = getattr(
            projection,
            "forward_prefill",
            None,
        )
        if context.is_prefill and callable(prefill_forward):
            forward = prefill_forward
        if not context.is_prefill:
            return forward(hidden_states)
        cu_seqlens_q = getattr(context, "cu_seqlens_q", None)
        cu_seqlens_k = getattr(context, "cu_seqlens_k", None)
        if (
            isinstance(cu_seqlens_q, torch.Tensor)
            and isinstance(cu_seqlens_k, torch.Tensor)
            and cu_seqlens_q.numel() == cu_seqlens_k.numel()
            and cu_seqlens_q.numel() > 1
        ):
            q_offsets = cu_seqlens_q.tolist()
            k_offsets = cu_seqlens_k.tolist()
            outputs = []
            for index in range(len(q_offsets) - 1):
                q_start = int(q_offsets[index])
                q_end = int(q_offsets[index + 1])
                q_length = q_end - q_start
                k_length = int(k_offsets[index + 1]) - int(k_offsets[index])
                segment = hidden_states[q_start:q_end]
                if k_length > q_length:
                    segment = torch.cat(
                        (
                            segment.new_zeros(
                                k_length - q_length,
                                segment.shape[1],
                            ),
                            segment,
                        ),
                        dim=0,
                    )
                outputs.append(forward(segment)[-q_length:])
            return torch.cat(outputs, dim=0)
        target_rows = int(getattr(context, "max_seqlen_k", 0))
        if target_rows <= hidden_states.shape[0]:
            return forward(hidden_states)
        padded = torch.cat(
            (
                hidden_states.new_zeros(
                    target_rows - hidden_states.shape[0],
                    hidden_states.shape[1],
                ),
                hidden_states,
            ),
            dim=0,
        )
        return forward(padded)[-hidden_states.shape[0]:]

    def _apply_decode_gated_rmsnorm(
        self,
        core: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        tensor_parallel_size = int(
            getattr(self.in_proj_qkv, "tp_size", 1)
        )
        if tensor_parallel_size == 1:
            return qwen35_gated_rmsnorm(
                core,
                gate,
                self.norm_weight,
                eps=self.norm_eps,
            )
        global_core = core.repeat(tensor_parallel_size, 1)
        global_gate = gate.repeat(tensor_parallel_size, 1)
        global_output = qwen35_gated_rmsnorm(
            global_core,
            global_gate,
            self.norm_weight,
            eps=self.norm_eps,
        )
        return global_output[:core.shape[0]]

    def _project_z(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self._project_context_rows(
            self.in_proj_z,
            hidden_states,
        )

    @staticmethod
    def _delta_context_rows(token_count: int) -> int:
        context = get_context()
        if not context.is_prefill:
            return token_count
        cu_seqlens_q = getattr(context, "cu_seqlens_q", None)
        cu_seqlens_k = getattr(context, "cu_seqlens_k", None)
        if (
            isinstance(cu_seqlens_q, torch.Tensor)
            and isinstance(cu_seqlens_k, torch.Tensor)
            and cu_seqlens_q.numel() == 2
            and cu_seqlens_k.numel() == 2
        ):
            q_length = int(cu_seqlens_q[1]) - int(cu_seqlens_q[0])
            k_length = int(cu_seqlens_k[1]) - int(cu_seqlens_k[0])
            if q_length == token_count:
                return max(token_count, k_length)
        return max(
            token_count,
            int(getattr(context, "max_seqlen_k", 0)),
        )

    def _forward(
        self,
        hidden_states: torch.Tensor,
        convolution_state: torch.Tensor,
        recurrent_state: torch.Tensor,
        *,
        capture_state_trace: bool,
    ):
        if not isinstance(hidden_states, torch.Tensor):
            raise ValueError("hidden_states must be a tensor")
        if hidden_states.ndim != 2:
            raise ValueError("hidden_states must be rank two")
        if not hidden_states.is_floating_point():
            raise ValueError(
                "hidden_states must use a floating point dtype"
            )

        key_width = self.local_key_heads * self.key_head_dim
        value_width = self.local_value_heads * self.value_head_dim
        conv_width = 2 * key_width + value_width
        if convolution_state.shape != (
            conv_width,
            self.conv_weight.shape[1],
        ):
            raise ValueError(
                "convolution_state shape must match conv width and history"
            )
        if recurrent_state.shape != (
            self.local_value_heads,
            self.value_head_dim,
            self.key_head_dim,
        ):
            raise ValueError(
                "recurrent_state shape must match physical orientation"
            )
        for name, state in (
            ("convolution_state", convolution_state),
            ("recurrent_state", recurrent_state),
        ):
            if not state.is_floating_point():
                raise ValueError(f"{name} must use a floating point dtype")
            if state.device != hidden_states.device:
                raise ValueError(f"{name} device must match hidden_states")
        if convolution_state.dtype != hidden_states.dtype:
            raise ValueError(
                "convolution_state dtype must match hidden_states"
            )
        for name, parameter in (
            ("conv_weight", self.conv_weight),
            ("A_log", self.A_log),
            ("dt_bias", self.dt_bias),
            ("norm_weight", self.norm_weight),
        ):
            if not parameter.is_floating_point():
                raise ValueError(
                    f"{name} must use a floating point dtype"
                )
            if parameter.device != hidden_states.device:
                raise ValueError(
                    f"{name} device must match hidden_states"
                )
        for name, parameter in (
            ("conv_weight", self.conv_weight),
            ("dt_bias", self.dt_bias),
        ):
            if parameter.dtype != hidden_states.dtype:
                raise ValueError(
                    f"{name} dtype must match hidden_states"
                )

        token_count = hidden_states.shape[0]
        projected_qkv = self._project_context_rows(
            self.in_proj_qkv,
            hidden_states,
        )
        projected_z = self._project_z(hidden_states)
        projected_b = self._project_context_rows(
            self.in_proj_b,
            hidden_states,
        )
        projected_a = self._project_context_rows(
            self.in_proj_a,
            hidden_states,
        )
        for output, name, width in (
            (projected_qkv, "in_proj_qkv", conv_width),
            (projected_z, "in_proj_z", value_width),
            (projected_b, "in_proj_b", self.local_value_heads),
            (projected_a, "in_proj_a", self.local_value_heads),
        ):
            self._validate_projection(
                output,
                name=name,
                token_count=token_count,
                feature_count=width,
                reference=hidden_states,
            )

        convolution_trace = None
        if capture_state_trace:
            convolved, convolution_trace = (
                qwen35_causal_depthwise_conv_prefix_trace(
                    projected_qkv,
                    convolution_state,
                    self.conv_weight,
                )
            )
            candidate_convolution_state = convolution_trace[-1]
        else:
            convolved, candidate_convolution_state = (
                qwen35_causal_depthwise_conv(
                    projected_qkv,
                    convolution_state,
                    self.conv_weight,
                )
            )
        self._validate_projection(
            convolved,
            name="causal convolution",
            token_count=token_count,
            feature_count=conv_width,
            reference=hidden_states,
        )
        if candidate_convolution_state.shape != convolution_state.shape:
            raise ValueError(
                "candidate convolution state shape must remain unchanged"
            )
        if (
            candidate_convolution_state.dtype != convolution_state.dtype
            or candidate_convolution_state.device != convolution_state.device
        ):
            raise ValueError(
                "candidate convolution state dtype and device must remain "
                "unchanged"
            )

        query, key, value = convolved.split(
            (key_width, key_width, value_width),
            dim=-1,
        )
        query = query.reshape(
            token_count,
            self.local_key_heads,
            self.key_head_dim,
        )
        key = key.reshape(
            token_count,
            self.local_key_heads,
            self.key_head_dim,
        )
        value = value.reshape(
            token_count,
            self.local_value_heads,
            self.value_head_dim,
        )
        repeat = self.local_value_heads // self.local_key_heads
        if repeat > 1:
            query = query.repeat_interleave(repeat, dim=1)
            key = key.repeat_interleave(repeat, dim=1)

        delta_rows = self._delta_context_rows(token_count)
        if delta_rows > token_count:
            prefix_rows = delta_rows - token_count
            query = torch.cat((
                query.new_zeros(
                    prefix_rows,
                    query.shape[1],
                    query.shape[2],
                ),
                query,
            ))
            key = torch.cat((
                key.new_zeros(
                    prefix_rows,
                    key.shape[1],
                    key.shape[2],
                ),
                key,
            ))
            value = torch.cat((
                value.new_zeros(
                    prefix_rows,
                    value.shape[1],
                    value.shape[2],
                ),
                value,
            ))
            projected_a = torch.cat((
                projected_a.new_full(
                    (prefix_rows, projected_a.shape[1]),
                    float("-inf"),
                ),
                projected_a,
            ))
            projected_b = torch.cat((
                projected_b.new_full(
                    (prefix_rows, projected_b.shape[1]),
                    float("-inf"),
                ),
                projected_b,
            ))
        recurrent_trace = None
        if capture_state_trace:
            core, recurrent_trace = qwen35_gated_delta_prefix_trace(
                query,
                key,
                value,
                projected_a,
                projected_b,
                self.A_log,
                self.dt_bias,
                recurrent_state,
            )
            candidate_recurrent_state = recurrent_trace[-1]
        else:
            delta_rule = (
                qwen35_gated_delta_recurrent
                if token_count == 1
                else qwen35_gated_delta_chunk
            )
            core, candidate_recurrent_state = delta_rule(
                query,
                key,
                value,
                projected_a,
                projected_b,
                self.A_log,
                self.dt_bias,
                recurrent_state,
            )
        if delta_rows > token_count:
            core = core[-token_count:]
            if recurrent_trace is not None:
                recurrent_trace = recurrent_trace[-token_count:]
        expected_core_shape = (
            token_count,
            self.local_value_heads,
            self.value_head_dim,
        )
        if core.shape != expected_core_shape:
            raise ValueError(
                "gated-delta output shape must match value heads"
            )
        if core.dtype != hidden_states.dtype or core.device != hidden_states.device:
            raise ValueError(
                "gated-delta output dtype and device must match hidden_states"
            )
        if candidate_recurrent_state.shape != recurrent_state.shape:
            raise ValueError(
                "candidate recurrent state shape must remain unchanged"
            )
        if (
            candidate_recurrent_state.dtype != recurrent_state.dtype
            or candidate_recurrent_state.device != recurrent_state.device
        ):
            raise ValueError(
                "candidate recurrent state dtype and device must remain "
                "unchanged"
            )
        norm_core = core.reshape(-1, self.value_head_dim)
        norm_gate = projected_z.reshape(-1, self.value_head_dim)
        if token_count == 1:
            gated = self._apply_decode_gated_rmsnorm(
                norm_core,
                norm_gate,
            )
        else:
            gated = qwen35_gated_rmsnorm(
                norm_core,
                norm_gate,
                self.norm_weight,
                eps=self.norm_eps,
            )
        gated = gated.reshape(token_count, value_width)
        output = self._project_context_rows(
            self.out_proj,
            gated,
        )
        if not isinstance(output, torch.Tensor):
            raise ValueError("out_proj output must be a tensor")
        if output.ndim != 2:
            raise ValueError("out_proj output must be rank two")
        if not output.is_floating_point():
            raise ValueError(
                "out_proj output must use a floating point dtype"
            )
        if output.shape[0] != token_count:
            raise ValueError(
                f"out_proj token count must equal {token_count}"
            )
        if output.dtype != hidden_states.dtype:
            raise ValueError("out_proj dtype must match hidden_states")
        if output.device != hidden_states.device:
            raise ValueError("out_proj device must match hidden_states")
        trace = (
            None
            if not capture_state_trace
            else Qwen35LinearStateTrace(
                convolution=convolution_trace,
                recurrent=recurrent_trace,
            )
        )
        return (
            output,
            candidate_convolution_state,
            candidate_recurrent_state,
            trace,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        convolution_state: torch.Tensor,
        recurrent_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output, convolution, recurrent, _ = self._forward(
            hidden_states,
            convolution_state,
            recurrent_state,
            capture_state_trace=False,
        )
        return output, convolution, recurrent

    def forward_with_state_trace(
        self,
        hidden_states: torch.Tensor,
        convolution_state: torch.Tensor,
        recurrent_state: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Qwen35LinearStateTrace,
    ]:
        output, convolution, recurrent, trace = self._forward(
            hidden_states,
            convolution_state,
            recurrent_state,
            capture_state_trace=True,
        )
        return output, convolution, recurrent, trace
