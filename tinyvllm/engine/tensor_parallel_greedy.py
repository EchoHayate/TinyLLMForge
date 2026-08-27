from __future__ import annotations

import sys

import torch
import torch.distributed as dist


def profile_collective(operation, tensor, call, **metadata):
    profiler_module = sys.modules.get(
        "tinyvllm.engine.decode_internal_profiler"
    )
    helper = getattr(
        profiler_module,
        "profile_collective",
        None,
    )
    if helper is None:
        return call(tensor)
    return helper(
        operation,
        tensor,
        call,
        **metadata,
    )


def _positive_integer(value, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_topology(rank, world_size) -> tuple[int, int]:
    world_size = _positive_integer(world_size, "world_size")
    if (
        isinstance(rank, bool)
        or not isinstance(rank, int)
        or rank < 0
        or rank >= world_size
    ):
        raise ValueError("rank must be in [0, world_size)")
    return rank, world_size


def _validate_root_logits(
    logits,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if not isinstance(logits, torch.Tensor):
        raise ValueError("root logits must be a tensor")
    if logits.ndim != 2:
        raise ValueError("root logits must be rank two")
    if logits.shape[0] != batch_size:
        raise ValueError("root logits row count must match batch_size")
    if not logits.is_floating_point():
        raise ValueError("root logits must use a floating dtype")
    if logits.device != device:
        raise ValueError("root logits device must match selector device")
    return logits


def _validate_token_ids(
    token_ids,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if not isinstance(token_ids, torch.Tensor):
        raise ValueError("selected token IDs must be a tensor")
    if token_ids.dtype != torch.int64:
        raise ValueError("selected token IDs must use torch.int64")
    if token_ids.shape != (batch_size,):
        raise ValueError(
            "selected token IDs must have exact shape [batch_size]"
        )
    if token_ids.device != device:
        raise ValueError(
            "selected token IDs device must match selector device"
        )
    if not token_ids.is_contiguous():
        raise ValueError("selected token IDs must be contiguous")
    if bool(torch.any(token_ids < 0).item()):
        raise ValueError("selected token IDs must be nonnegative")
    return token_ids


def select_tensor_parallel_greedy_tokens(
    logits: torch.Tensor | None,
    *,
    rank: int,
    world_size: int,
    batch_size: int,
    device: torch.device,
    broadcast=None,
) -> torch.Tensor:
    rank, world_size = _validate_topology(rank, world_size)
    batch_size = _positive_integer(batch_size, "batch_size")
    device = torch.device(device)
    if rank == 0:
        logits = _validate_root_logits(
            logits,
            batch_size=batch_size,
            device=device,
        )
        token_ids = logits.argmax(dim=-1).to(
            device=device,
            dtype=torch.int64,
        ).contiguous()
    else:
        if logits is not None:
            raise ValueError("non-root logits must be None")
        token_ids = torch.empty(
            batch_size,
            dtype=torch.int64,
            device=device,
        )
    if world_size > 1:
        operation = dist.broadcast if broadcast is None else broadcast
        if not callable(operation):
            raise ValueError("broadcast must be callable")
        profile_collective(
            "greedy_token_broadcast",
            token_ids,
            lambda tensor: operation(tensor, src=0),
            site_role="greedy_token_broadcast",
            collective_kind="broadcast",
            process_group="tensor_parallel",
            execution_phase="decode",
            async_mode=False,
            source_rank=0,
        )
    return _validate_token_ids(
        token_ids,
        batch_size=batch_size,
        device=device,
    )
