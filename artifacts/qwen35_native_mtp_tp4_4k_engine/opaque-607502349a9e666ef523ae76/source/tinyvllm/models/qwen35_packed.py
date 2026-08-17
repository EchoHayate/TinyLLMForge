from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from tinyvllm.engine.hybrid_state import HybridStateLease
from tinyvllm.layers.qwen35_packed_layer_stack import (
    Qwen35PackedHeterogeneousLayerStack,
)


@dataclass
class Qwen35PreparedModelStep:
    leases: tuple[HybridStateLease, ...]
    token_counts: tuple[int, ...]
    normalized: torch.Tensor
    logits: torch.Tensor | None
    final_candidates: tuple[
        tuple[torch.Tensor, torch.Tensor],
        ...,
    ]
    prefix_candidates: object | None = None
    state: str = "prepared"


class Qwen35PackedForCausalLM(nn.Module):

    def __init__(
        self,
        embed_tokens: nn.Module,
        layer_stack: Qwen35PackedHeterogeneousLayerStack,
        final_norm: nn.Module,
        lm_head: nn.Module,
    ):
        super().__init__()
        if not isinstance(embed_tokens, nn.Module):
            raise ValueError("embed_tokens must be a module")
        if type(layer_stack) is not Qwen35PackedHeterogeneousLayerStack:
            raise ValueError(
                "layer_stack must be an exact packed Qwen3.5 layer stack"
            )
        if not isinstance(final_norm, nn.Module):
            raise ValueError("final_norm must be a module")
        if not isinstance(lm_head, nn.Module):
            raise ValueError("lm_head must be a module")
        self.embed_tokens = embed_tokens
        self.layer_stack = layer_stack
        self.final_norm = final_norm
        self.lm_head = lm_head

    @staticmethod
    def _validate_public_inputs(
        leases: tuple[HybridStateLease, ...],
        token_counts: tuple[int, ...],
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
    ) -> None:
        if not isinstance(leases, tuple) or not leases:
            raise ValueError("leases must be a non-empty tuple")
        if any(type(lease) is not HybridStateLease for lease in leases):
            raise ValueError(
                "leases must contain only HybridStateLease values"
            )
        if not isinstance(token_counts, tuple) or not token_counts:
            raise ValueError("token_counts must be a non-empty tuple")
        if len(token_counts) != len(leases):
            raise ValueError(
                "leases and token_counts batch size must match"
            )
        if any(
            isinstance(token_count, bool)
            or not isinstance(token_count, int)
            or token_count <= 0
            for token_count in token_counts
        ):
            raise ValueError(
                "token_counts must contain positive integers"
            )
        if not isinstance(input_ids, torch.Tensor):
            raise ValueError("input_ids must be a tensor")
        if input_ids.ndim != 1:
            raise ValueError("input_ids must be rank one")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("input_ids must use an integer dtype")
        if sum(token_counts) != input_ids.shape[0]:
            raise ValueError(
                "token_counts sum must match input_ids token count"
            )
        if input_embeds is None:
            return
        if not isinstance(input_embeds, torch.Tensor):
            raise ValueError("input_embeds must be a tensor")
        if input_embeds.ndim != 2:
            raise ValueError("input_embeds must be rank two")
        if not input_embeds.is_floating_point():
            raise ValueError(
                "input_embeds must use a floating point dtype"
            )
        if input_embeds.shape[0] != input_ids.shape[0]:
            raise ValueError(
                "input_embeds token count must match input_ids"
            )

    @staticmethod
    def _validate_hidden_output(
        output,
        *,
        token_count: int,
        name: str,
        reference: torch.Tensor | None = None,
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
                f"{name} output token count must remain unchanged"
            )
        if reference is None:
            return
        if output.shape[1] != reference.shape[1]:
            raise ValueError(
                f"{name} output hidden width must remain unchanged"
            )
        if output.dtype != reference.dtype:
            raise ValueError(
                f"{name} output dtype must remain unchanged"
            )
        if output.device != reference.device:
            raise ValueError(
                f"{name} output device must remain unchanged"
            )

    @staticmethod
    def _distributed_output_role() -> tuple[int, int] | None:
        if not torch.distributed.is_available():
            return None
        if not torch.distributed.is_initialized():
            return None
        world_size = int(torch.distributed.get_world_size())
        rank = int(torch.distributed.get_rank())
        if world_size <= 0 or rank < 0 or rank >= world_size:
            raise ValueError("distributed output role is invalid")
        return rank, world_size

    @classmethod
    def _validate_logits(
        cls,
        logits,
        hidden_states: torch.Tensor,
    ) -> None:
        distributed_role = cls._distributed_output_role()
        if (
            distributed_role is not None
            and distributed_role[1] > 1
            and distributed_role[0] != 0
        ):
            if logits is not None:
                raise ValueError(
                    "non-root lm_head output must be None"
                )
            return
        if logits is None and distributed_role is not None:
            raise ValueError(
                "rank zero lm_head output must be a tensor"
            )
        if not isinstance(logits, torch.Tensor):
            raise ValueError("lm_head output must be a tensor")
        if logits.ndim != 2:
            raise ValueError("lm_head output must be rank two")
        if not logits.is_floating_point():
            raise ValueError(
                "lm_head output must use a floating point dtype"
            )
        if (
            logits.shape[0] <= 0
            or logits.shape[0] > hidden_states.shape[0]
        ):
            raise ValueError(
                "lm_head output logit row count must be positive "
                "and no larger than hidden token count"
            )
        if logits.shape[1] <= 0:
            raise ValueError(
                "lm_head output vocabulary width must be positive"
            )
        if logits.device != hidden_states.device:
            raise ValueError(
                "lm_head output device must match hidden states"
            )

    def prepare_step(
        self,
        leases: tuple[HybridStateLease, ...],
        token_counts: tuple[int, ...],
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
        *,
        initial_candidates=None,
        capture_prefix_states: bool = False,
    ) -> Qwen35PreparedModelStep:
        self._validate_public_inputs(
            leases,
            token_counts,
            input_ids,
            input_embeds,
        )
        hidden_states = (
            self.embed_tokens(input_ids)
            if input_embeds is None
            else input_embeds
        )
        self._validate_hidden_output(
            hidden_states,
            token_count=input_ids.shape[0],
            name="embed_tokens",
        )
        prepared_stack = self.layer_stack.prepare_transactional(
            leases,
            token_counts,
            position_ids,
            hidden_states,
            initial_candidates=initial_candidates,
            capture_prefix_states=capture_prefix_states,
        )
        hidden_states = prepared_stack.hidden_states
        normalized = self.final_norm(hidden_states)
        self._validate_hidden_output(
            normalized,
            token_count=input_ids.shape[0],
            name="final_norm",
            reference=hidden_states,
        )
        logits = self.lm_head(normalized)
        self._validate_logits(logits, normalized)
        return Qwen35PreparedModelStep(
            leases=leases,
            token_counts=token_counts,
            normalized=normalized,
            logits=logits,
            final_candidates=prepared_stack.final_candidates,
            prefix_candidates=prepared_stack.prefix_candidates,
        )

    def commit_prepared_step(
        self,
        leases: tuple[HybridStateLease, ...],
        prepared: Qwen35PreparedModelStep,
    ) -> None:
        if type(prepared) is not Qwen35PreparedModelStep:
            raise ValueError(
                "prepared must be Qwen35PreparedModelStep"
            )
        if prepared.state != "prepared":
            raise RuntimeError(
                "prepared model step is not active: "
                f"{prepared.state}"
            )
        if leases != prepared.leases:
            raise ValueError(
                "prepared model step lease identity mismatch"
            )
        self.layer_stack.commit(
            leases,
            prepared.final_candidates,
        )
        prepared.state = "committed"

    def run_step(
        self,
        leases: tuple[HybridStateLease, ...],
        token_counts: tuple[int, ...],
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        prepared = self.prepare_step(
            leases,
            token_counts,
            input_ids,
            position_ids,
            input_embeds=input_embeds,
        )
        self.commit_prepared_step(leases, prepared)
        return prepared.normalized, prepared.logits
