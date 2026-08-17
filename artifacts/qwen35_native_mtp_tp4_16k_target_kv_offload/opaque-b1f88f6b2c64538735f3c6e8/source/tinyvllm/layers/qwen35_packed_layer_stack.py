from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import nn

from tinyvllm.engine.hybrid_state import HybridStateLease
from tinyvllm.engine.qwen35_state_transaction import (
    Qwen35CrossLayerStateTransaction,
)
from tinyvllm.layers.qwen35_decoder_layer import Qwen35DecoderLayerShell
from tinyvllm.layers.qwen35_packed_stateful_decoder_layer import (
    Qwen35PackedStatefulLinearDecoderLayer,
)
from tinyvllm.utils.context import get_context, temporary_context


@dataclass(frozen=True)
class Qwen35PreparedLayerStack:
    hidden_states: torch.Tensor
    final_candidates: tuple[
        tuple[torch.Tensor, torch.Tensor],
        ...,
    ]
    prefix_candidates: tuple[
        tuple[
            tuple[
                tuple[torch.Tensor, torch.Tensor],
                ...,
            ],
            ...,
        ],
        ...,
    ] | None


class Qwen35PackedHeterogeneousLayerStack(nn.Module):

    def __init__(
        self,
        layers: tuple[Qwen35DecoderLayerShell, ...],
        state_transaction: Qwen35CrossLayerStateTransaction,
    ):
        super().__init__()
        if not isinstance(layers, tuple) or not layers:
            raise ValueError("layers must be a non-empty tuple")
        if any(
            not isinstance(layer, Qwen35DecoderLayerShell)
            for layer in layers
        ):
            raise ValueError(
                "layers must contain only Qwen35DecoderLayerShell values"
            )
        if not isinstance(
            state_transaction,
            Qwen35CrossLayerStateTransaction,
        ):
            raise ValueError(
                "state_transaction must be a "
                "Qwen35CrossLayerStateTransaction"
            )
        linear_indices = tuple(
            layer_index
            for layer_index, layer in enumerate(layers)
            if layer.block_type == "linear_attention"
        )
        adapter_indices = tuple(
            adapter.layer_index
            for adapter in state_transaction.adapters
        )
        if adapter_indices != linear_indices:
            raise ValueError(
                "transaction adapter indices must match linear layer indices"
            )
        self.layers = nn.ModuleList(layers)
        self.state_transaction = state_transaction
        self.linear_indices = linear_indices

    @staticmethod
    def _position_segment(
        position_ids: torch.Tensor,
        start: int,
        end: int,
    ) -> torch.Tensor:
        return (
            position_ids[start:end]
            if position_ids.ndim == 1
            else position_ids[:, start:end]
        )

    @staticmethod
    @contextmanager
    def _request_local_context(
        token_counts: tuple[int, ...],
        request_index: int,
        start: int,
        end: int,
    ):
        context = get_context()
        token_count = end - start
        request_local_prefill = (
            context.is_prefill
            and isinstance(context.cu_seqlens_q, torch.Tensor)
            and isinstance(context.cu_seqlens_k, torch.Tensor)
            and context.cu_seqlens_q.numel() == len(token_counts) + 1
            and context.cu_seqlens_k.numel() == len(token_counts) + 1
        )
        if request_local_prefill:
            q_start = int(context.cu_seqlens_q[request_index])
            q_end = int(context.cu_seqlens_q[request_index + 1])
            if q_start != start or q_end != end:
                raise ValueError(
                    "token_counts must match prefill query boundaries"
                )
            k_start = int(context.cu_seqlens_k[request_index])
            k_end = int(context.cu_seqlens_k[request_index + 1])
            k_length = k_end - k_start
            reference_lens = context.prefill_attention_reference_lens
            with temporary_context(
                cu_seqlens_q=context.cu_seqlens_q.new_tensor(
                    (0, token_count)
                ),
                cu_seqlens_k=context.cu_seqlens_k.new_tensor(
                    (0, k_length)
                ),
                max_seqlen_q=token_count,
                max_seqlen_k=k_length,
                slot_mapping=(
                    None
                    if context.slot_mapping is None
                    else context.slot_mapping[start:end]
                ),
                block_tables=(
                    None
                    if context.block_tables is None
                    else context.block_tables[
                        request_index:request_index + 1
                    ]
                ),
                prefill_attention_reference_lens=(
                    None
                    if reference_lens is None
                    else (reference_lens[request_index],)
                ),
            ) as request_context:
                yield request_context
            return

        request_local_spec_verify = (
            context.mode == "spec_verify"
            and isinstance(context.context_lens, torch.Tensor)
            and context.context_lens.numel() == len(token_counts)
            and context.spec_verify_query_lens == token_counts
        )
        if request_local_spec_verify:
            with temporary_context(
                slot_mapping=(
                    None
                    if context.slot_mapping is None
                    else context.slot_mapping[start:end]
                ),
                context_lens=context.context_lens[
                    request_index:request_index + 1
                ],
                block_tables=(
                    None
                    if context.block_tables is None
                    else context.block_tables[
                        request_index:request_index + 1
                    ]
                ),
                spec_verify_query_lens=(token_count,),
                kv_offload_logical_block_tables=(
                    None
                    if context.kv_offload_logical_block_tables
                    is None
                    else [
                        context.kv_offload_logical_block_tables[
                            request_index
                        ]
                    ]
                ),
                kv_offload_context_lens=(
                    None
                    if context.kv_offload_context_lens is None
                    else [
                        context.kv_offload_context_lens[
                            request_index
                        ]
                    ]
                ),
            ) as request_context:
                yield request_context
            return

        request_local_decode = (
            not context.is_prefill
            and isinstance(context.context_lens, torch.Tensor)
            and context.context_lens.numel() == len(token_counts)
            and all(count == 1 for count in token_counts)
        )
        if request_local_decode:
            with temporary_context(
                slot_mapping=(
                    None
                    if context.slot_mapping is None
                    else context.slot_mapping[
                        request_index:request_index + 1
                    ]
                ),
                context_lens=context.context_lens[
                    request_index:request_index + 1
                ],
                block_tables=(
                    None
                    if context.block_tables is None
                    else context.block_tables[
                        request_index:request_index + 1
                    ]
                ),
                kv_offload_logical_block_tables=(
                    None
                    if context.kv_offload_logical_block_tables
                    is None
                    else [
                        context.kv_offload_logical_block_tables[
                            request_index
                        ]
                    ]
                ),
                kv_offload_context_lens=(
                    None
                    if context.kv_offload_context_lens is None
                    else [
                        context.kv_offload_context_lens[
                            request_index
                        ]
                    ]
                ),
            ) as request_context:
                yield request_context
            return

        yield context

    def _run_full_layer(
        self,
        layer: Qwen35DecoderLayerShell,
        token_counts: tuple[int, ...],
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        outputs = []
        offset = 0
        for request_index, token_count in enumerate(token_counts):
            end = offset + token_count
            with self._request_local_context(
                token_counts,
                request_index,
                offset,
                end,
            ):
                outputs.append(layer(
                    self._position_segment(position_ids, offset, end),
                    hidden_states[offset:end],
                ))
            offset = end
        return torch.cat(outputs)

    def _run_linear_layer(
        self,
        layer: Qwen35DecoderLayerShell,
        adapter,
        token_counts: tuple[int, ...],
        hidden_states: torch.Tensor,
        convolution_states: torch.Tensor,
        recurrent_states: torch.Tensor,
        *,
        capture_prefix_states: bool,
    ):
        outputs = []
        candidate_convolution = []
        candidate_recurrent = []
        request_traces = []
        offset = 0
        for request_index, token_count in enumerate(token_counts):
            end = offset + token_count
            segment = hidden_states[offset:end]
            with self._request_local_context(
                token_counts,
                request_index,
                offset,
                end,
            ):
                first_residual = segment
                normalized = layer.input_layernorm(segment)
                layer._validate_component_output(
                    normalized,
                    segment,
                    "input_layernorm",
                )
                if capture_prefix_states:
                    trace_forward = getattr(
                        layer.linear_attention,
                        "forward_with_state_trace",
                        None,
                    )
                    if not callable(trace_forward):
                        raise ValueError(
                            "linear_attention must expose "
                            "forward_with_state_trace"
                        )
                    mixer_output = trace_forward(
                        normalized,
                        convolution_states[request_index],
                        recurrent_states[request_index],
                    )
                else:
                    mixer_output = layer.linear_attention(
                        normalized,
                        convolution_states[request_index],
                        recurrent_states[request_index],
                    )
                if (
                    not isinstance(mixer_output, tuple)
                    or len(mixer_output)
                    != (4 if capture_prefix_states else 3)
                ):
                    raise ValueError(
                        "linear_attention output has invalid trace arity"
                    )
                mixed, next_convolution, next_recurrent = (
                    mixer_output[:3]
                )
                layer._validate_component_output(
                    mixed,
                    segment,
                    "linear_attention",
                )
                adapter._validate_candidate(
                    next_convolution,
                    convolution_states[request_index],
                    "convolution_state",
                )
                adapter._validate_candidate(
                    next_recurrent,
                    recurrent_states[request_index],
                    "recurrent_state",
                )
                segment = first_residual + mixed
                second_residual = segment
                normalized = layer.post_attention_layernorm(segment)
                layer._validate_component_output(
                    normalized,
                    segment,
                    "post_attention_layernorm",
                )
                mlp_output = layer.mlp(normalized)
                layer._validate_component_output(
                    mlp_output,
                    segment,
                    "mlp",
                )
                outputs.append(second_residual + mlp_output)
                if capture_prefix_states:
                    trace = mixer_output[3]
                    if (
                        not isinstance(trace.convolution, torch.Tensor)
                        or trace.convolution.shape[0] != token_count
                        or not isinstance(trace.recurrent, torch.Tensor)
                        or trace.recurrent.shape[0] != token_count
                    ):
                        raise ValueError(
                            "linear attention prefix trace inventory "
                            "must match token count"
                        )
                    prefix_pairs = []
                    for prefix_index in range(token_count):
                        adapter._validate_candidate(
                            trace.convolution[prefix_index],
                            convolution_states[request_index],
                            "convolution_state",
                        )
                        adapter._validate_candidate(
                            trace.recurrent[prefix_index],
                            recurrent_states[request_index],
                            "recurrent_state",
                        )
                        prefix_pairs.append((
                            trace.convolution[prefix_index],
                            trace.recurrent[prefix_index],
                        ))
                    request_traces.append(tuple(prefix_pairs))
            candidate_convolution.append(next_convolution)
            candidate_recurrent.append(next_recurrent)
            offset = end
        return (
            torch.cat(outputs),
            (
                torch.stack(candidate_convolution),
                torch.stack(candidate_recurrent),
            ),
            (
                tuple(request_traces)
                if capture_prefix_states
                else None
            ),
        )

    def prepare_transactional(
        self,
        leases: tuple[HybridStateLease, ...],
        token_counts: tuple[int, ...],
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        *,
        initial_candidates=None,
        capture_prefix_states: bool = False,
    ) -> Qwen35PreparedLayerStack:
        Qwen35PackedStatefulLinearDecoderLayer._validate_inputs(
            leases,
            token_counts,
            position_ids,
            hidden_states,
        )
        if not isinstance(capture_prefix_states, bool):
            raise ValueError(
                "capture_prefix_states must be a boolean"
            )
        slot_ids = tuple(
            adapter._validate_lease_batch(leases)
            for adapter in self.state_transaction.adapters
        )
        reference_slot_ids = slot_ids[0]
        if any(
            value != reference_slot_ids
            for value in slot_ids[1:]
        ):
            raise RuntimeError(
                "adapters resolved inconsistent slot ids"
            )
        if initial_candidates is None:
            gathered = self.state_transaction.gather(leases)
        else:
            self.state_transaction._validate_candidates(
                self.state_transaction.adapters,
                reference_slot_ids,
                initial_candidates,
            )
            gathered = initial_candidates
        candidates = []
        prefix_candidates = (
            tuple(
                [list() for _ in range(token_count)]
                for token_count in token_counts
            )
            if capture_prefix_states
            else None
        )
        linear_offset = 0
        for layer in self.layers:
            if layer.block_type == "full_attention":
                hidden_states = self._run_full_layer(
                    layer,
                    token_counts,
                    position_ids,
                    hidden_states,
                )
                continue
            adapter = self.state_transaction.adapters[linear_offset]
            convolution_states, recurrent_states = gathered[linear_offset]
            (
                hidden_states,
                layer_candidates,
                layer_prefixes,
            ) = self._run_linear_layer(
                layer,
                adapter,
                token_counts,
                hidden_states,
                convolution_states,
                recurrent_states,
                capture_prefix_states=capture_prefix_states,
            )
            candidates.append(layer_candidates)
            if capture_prefix_states:
                for sequence_index, sequence_prefixes in enumerate(
                    layer_prefixes
                ):
                    for prefix_index, candidate_pair in enumerate(
                        sequence_prefixes
                    ):
                        prefix_candidates[
                            sequence_index
                        ][prefix_index].append(candidate_pair)
            linear_offset += 1

        normalized_prefix_candidates = (
            None
            if prefix_candidates is None
            else tuple(
                tuple(
                    tuple(prefix)
                    for prefix in sequence_prefixes
                )
                for sequence_prefixes in prefix_candidates
            )
        )
        return Qwen35PreparedLayerStack(
            hidden_states=hidden_states,
            final_candidates=tuple(candidates),
            prefix_candidates=normalized_prefix_candidates,
        )

    def prepare(
        self,
        leases: tuple[HybridStateLease, ...],
        token_counts: tuple[int, ...],
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        tuple[tuple[torch.Tensor, torch.Tensor], ...],
    ]:
        prepared = self.prepare_transactional(
            leases,
            token_counts,
            position_ids,
            hidden_states,
        )
        return prepared.hidden_states, prepared.final_candidates

    def commit(
        self,
        leases: tuple[HybridStateLease, ...],
        candidates: tuple[
            tuple[torch.Tensor, torch.Tensor],
            ...,
        ],
    ) -> None:
        self.state_transaction.commit(leases, candidates)

    def forward(
        self,
        leases: tuple[HybridStateLease, ...],
        token_counts: tuple[int, ...],
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states, candidates = self.prepare(
            leases,
            token_counts,
            position_ids,
            hidden_states,
        )
        self.commit(leases, candidates)
        return hidden_states
