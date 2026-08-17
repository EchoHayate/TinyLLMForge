from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import torch
from safetensors import safe_open

from tinyvllm.engine.qwen35_hybrid_model_owner import (
    Qwen35HybridModelOwner,
    build_qwen35_hybrid_model_owner,
)
from tinyvllm.engine.qwen35_hybrid_prefix_runtime_identity import (
    validate_qwen35_model_fingerprint,
)
from tinyvllm.models.qwen35_checkpoint_binding import (
    Qwen35CheckpointBindingPlan,
    Qwen35CheckpointTensorBinding,
)
from tinyvllm.models.qwen35_checkpoint_tiles import (
    Qwen35CheckpointTile,
    Qwen35CheckpointTilePlan,
    build_qwen35_checkpoint_tile_plan,
)
from tinyvllm.models.qwen35_checkpoint_tile_policy import (
    Qwen35CheckpointTileBudgetDecision,
    select_qwen35_checkpoint_tile_budget,
)
from tinyvllm.models.qwen35_packed import Qwen35PackedForCausalLM


@dataclass(frozen=True)
class Qwen35TiledCheckpointLoadStats:
    assigned_bindings: int
    source_tensors: int
    shard_count: int
    tile_count: int
    destination_bytes: int
    materialized_bytes: int
    peak_tile_bytes: int


@dataclass(frozen=True)
class Qwen35TiledLoadedCheckpointCandidate:
    owner: Qwen35HybridModelOwner
    binding_plan: Qwen35CheckpointBindingPlan
    tile_plan: Qwen35CheckpointTilePlan
    stats: Qwen35TiledCheckpointLoadStats
    model_fingerprint: str


@dataclass(frozen=True)
class Qwen35PolicyTiledLoadedCheckpointCandidate:
    loaded: Qwen35TiledLoadedCheckpointCandidate
    decision: Qwen35CheckpointTileBudgetDecision


def _checkpoint_directory(value) -> Path:
    try:
        path = Path(value)
    except TypeError as error:
        raise ValueError(
            "checkpoint_dir must be an existing directory"
        ) from error
    if not path.is_dir():
        raise ValueError(
            "checkpoint_dir must be an existing directory"
        )
    return path.resolve()


def _safe_shard_path(checkpoint_dir: Path, shard_name: str) -> Path:
    if not isinstance(shard_name, str) or not shard_name:
        raise ValueError("checkpoint shard must be a non-empty string")
    relative = PurePosixPath(shard_name)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or "\\" in shard_name
        or shard_name != str(relative)
        or not shard_name.endswith(".safetensors")
    ):
        raise ValueError("checkpoint shard must be a safe relative path")
    path = (checkpoint_dir / Path(*relative.parts)).resolve()
    if checkpoint_dir not in path.parents:
        raise ValueError("checkpoint shard must remain in checkpoint_dir")
    if not path.is_file():
        raise ValueError(f"missing checkpoint shard: {shard_name}")
    return path


def _registered_tensor_ids(
    model: Qwen35PackedForCausalLM,
) -> set[int]:
    return {
        id(tensor)
        for _, tensor in (
            list(model.named_parameters(remove_duplicate=False))
            + list(model.named_buffers(remove_duplicate=False))
        )
    }


def _validate_candidate(
    candidate,
) -> tuple[
    Qwen35PackedForCausalLM,
    Qwen35CheckpointBindingPlan,
]:
    if not isinstance(candidate, tuple) or len(candidate) != 2:
        raise ValueError(
            "candidate_factory must return an exact two-item tuple"
        )
    model, binding_plan = candidate
    if type(model) is not Qwen35PackedForCausalLM:
        raise ValueError(
            "candidate model must be an exact Qwen35PackedForCausalLM"
        )
    if type(binding_plan) is not Qwen35CheckpointBindingPlan:
        raise ValueError(
            "candidate binding plan must be an exact "
            "Qwen35CheckpointBindingPlan"
        )
    registered_ids = _registered_tensor_ids(model)
    for binding in binding_plan.bindings:
        if type(binding) is not Qwen35CheckpointTensorBinding:
            raise ValueError(
                "binding plan entries must be exact "
                "Qwen35CheckpointTensorBinding values"
            )
        destination = binding.destination
        if (
            not isinstance(destination, torch.Tensor)
            or destination.device.type != "cpu"
        ):
            raise ValueError(
                "candidate destinations must be CPU non-meta tensors"
            )
        if id(destination) not in registered_ids:
            raise ValueError(
                "candidate destination must be registered by model"
            )
    return model, binding_plan


def _copy_qwen35_checkpoint_tile(
    tile: Qwen35CheckpointTile,
    tensor: torch.Tensor,
) -> None:
    if type(tile) is not Qwen35CheckpointTile:
        raise ValueError("tile must be an exact Qwen35CheckpointTile")
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(
            f"materialized tile must be a tensor: {tile.source_name}"
        )
    if tensor.device.type != "cpu":
        raise ValueError(
            f"materialized tile must be CPU: {tile.source_name}"
        )
    if tensor.dtype != tile.dtype:
        raise ValueError(
            f"materialized tile dtype mismatch: {tile.source_name}"
        )
    if tuple(tensor.shape) != tile.tile_shape:
        raise ValueError(
            f"materialized tile shape mismatch: {tile.source_name}"
        )
    if tensor.numel() * tensor.element_size() != tile.byte_count:
        raise ValueError(
            f"materialized tile byte mismatch: {tile.source_name}"
        )
    destination = tile.destination[tile.destination_slices]
    if tuple(destination.shape) != tile.destination_shape:
        raise ValueError(
            f"destination tile shape mismatch: {tile.target}"
        )
    allows_runtime_cast = (
        tile.target.endswith("linear_attention.norm_weight")
        and tile.dtype == torch.float32
        and destination.dtype == torch.bfloat16
    )
    if destination.dtype != tile.dtype and not allows_runtime_cast:
        raise ValueError(
            f"destination tile dtype mismatch: {tile.target}"
        )
    with torch.no_grad():
        destination.copy_(tensor)


def _load_qwen35_candidate_with_tile_plan(
    model: Qwen35PackedForCausalLM,
    binding_plan: Qwen35CheckpointBindingPlan,
    tile_plan: Qwen35CheckpointTilePlan,
    checkpoint_dir: str | Path,
    model_fingerprint: str,
) -> Qwen35TiledLoadedCheckpointCandidate:
    if type(tile_plan) is not Qwen35CheckpointTilePlan:
        raise ValueError(
            "tile_plan must be an exact Qwen35CheckpointTilePlan"
        )
    if (
        tile_plan.tensor_parallel_size
        != binding_plan.tensor_parallel_size
        or tile_plan.tensor_parallel_rank
        != binding_plan.tensor_parallel_rank
        or tile_plan.binding_count != len(binding_plan.bindings)
    ):
        raise ValueError("tile_plan must match binding_plan")
    for tile in tile_plan.tiles:
        if (
            tile.binding_index < 0
            or tile.binding_index >= len(binding_plan.bindings)
            or tile.destination
            is not binding_plan.bindings[tile.binding_index].destination
        ):
            raise ValueError("tile_plan must match binding_plan")
    directory = _checkpoint_directory(checkpoint_dir)

    by_shard = {}
    shard_paths = {}
    source_shapes = {}
    for tile in tile_plan.tiles:
        by_shard.setdefault(tile.shard, {}).setdefault(
            tile.source_name,
            [],
        ).append(tile)
        existing = source_shapes.setdefault(
            tile.source_name,
            tile.source_tensor_shape,
        )
        if existing != tile.source_tensor_shape:
            raise ValueError(
                "conflicting tiled checkpoint source shape: "
                f"{tile.source_name}"
            )
    for shard_name in by_shard:
        shard_paths[shard_name] = _safe_shard_path(
            directory,
            shard_name,
        )

    materialized_bytes = 0
    tile_count = 0
    loaded_sources = 0
    for shard_name in sorted(by_shard):
        requested = by_shard[shard_name]
        with safe_open(
            shard_paths[shard_name],
            framework="pt",
            device="cpu",
        ) as handle:
            available = set(handle.keys())
            for source_name in sorted(requested):
                if source_name not in available:
                    raise ValueError(
                        "missing requested source in shard: "
                        f"{source_name}"
                    )
                slice_view = handle.get_slice(source_name)
                if tuple(slice_view.get_shape()) != source_shapes[
                    source_name
                ]:
                    raise ValueError(
                        "safetensors source shape mismatch: "
                        f"{source_name}"
                    )
                for tile in requested[source_name]:
                    tensor = slice_view[tile.source_slices]
                    _copy_qwen35_checkpoint_tile(tile, tensor)
                    materialized_bytes += tile.byte_count
                    tile_count += 1
                    del tensor
                del slice_view
                loaded_sources += 1

    if loaded_sources != tile_plan.source_count:
        raise ValueError(
            "tiled checkpoint source coverage is incomplete"
        )
    if tile_count != len(tile_plan.tiles):
        raise ValueError("tiled checkpoint tile coverage is incomplete")
    owner = build_qwen35_hybrid_model_owner(model)
    return Qwen35TiledLoadedCheckpointCandidate(
        owner=owner,
        binding_plan=binding_plan,
        tile_plan=tile_plan,
        stats=Qwen35TiledCheckpointLoadStats(
            assigned_bindings=tile_plan.binding_count,
            source_tensors=loaded_sources,
            shard_count=len(by_shard),
            tile_count=tile_count,
            destination_bytes=tile_plan.destination_bytes,
            materialized_bytes=materialized_bytes,
            peak_tile_bytes=tile_plan.peak_tile_bytes,
        ),
        model_fingerprint=model_fingerprint,
    )


def load_qwen35_fresh_checkpoint_candidate_tiled(
    candidate_factory: Callable[
        [],
        tuple[
            Qwen35PackedForCausalLM,
            Qwen35CheckpointBindingPlan,
        ],
    ],
    checkpoint_dir: str | Path,
    *,
    max_tile_bytes: int,
    model_fingerprint: str,
) -> Qwen35TiledLoadedCheckpointCandidate:
    if not callable(candidate_factory):
        raise ValueError("candidate_factory must be callable")
    model_fingerprint = validate_qwen35_model_fingerprint(
        model_fingerprint
    )
    model, binding_plan = _validate_candidate(candidate_factory())
    tile_plan = build_qwen35_checkpoint_tile_plan(
        binding_plan,
        max_tile_bytes=max_tile_bytes,
    )
    return _load_qwen35_candidate_with_tile_plan(
        model,
        binding_plan,
        tile_plan,
        checkpoint_dir,
        model_fingerprint,
    )


def load_qwen35_fresh_checkpoint_candidate_with_tile_policy(
    candidate_factory: Callable[
        [],
        tuple[
            Qwen35PackedForCausalLM,
            Qwen35CheckpointBindingPlan,
        ],
    ],
    checkpoint_dir: str | Path,
    *,
    max_tile_bytes: int,
    max_tile_count: int,
    model_fingerprint: str,
) -> Qwen35PolicyTiledLoadedCheckpointCandidate:
    if not callable(candidate_factory):
        raise ValueError("candidate_factory must be callable")
    model_fingerprint = validate_qwen35_model_fingerprint(
        model_fingerprint
    )
    model, binding_plan = _validate_candidate(candidate_factory())
    decision = select_qwen35_checkpoint_tile_budget(
        binding_plan,
        max_tile_bytes=max_tile_bytes,
        max_tile_count=max_tile_count,
    )
    loaded = _load_qwen35_candidate_with_tile_plan(
        model,
        binding_plan,
        decision.tile_plan,
        checkpoint_dir,
        model_fingerprint,
    )
    return Qwen35PolicyTiledLoadedCheckpointCandidate(
        loaded=loaded,
        decision=decision,
    )
