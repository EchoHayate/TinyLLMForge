from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType

import torch
from safetensors import safe_open

from tinyvllm.models.qwen35_checkpoint_assignment import (
    Qwen35CheckpointAssignmentResult,
    assign_qwen35_checkpoint_tensors,
)
from tinyvllm.models.qwen35_checkpoint_binding import (
    Qwen35CheckpointBindingPlan,
    Qwen35CheckpointTensorBinding,
)


@dataclass(frozen=True)
class Qwen35CheckpointMaterialization:
    source_tensors: Mapping[str, torch.Tensor]
    source_count: int
    shard_count: int
    materialized_bytes: int


@dataclass(frozen=True)
class Qwen35CheckpointLoadResult:
    materialization: Qwen35CheckpointMaterialization
    assignment: Qwen35CheckpointAssignmentResult


_DTYPE_BYTES = {
    "BF16": 2,
    "F32": 4,
}
_TORCH_DTYPES = {
    "BF16": torch.bfloat16,
    "F32": torch.float32,
}


def _positive_budget(value) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError(
            "max_materialized_bytes must be a positive integer"
        )
    return value


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


def _source_contracts(
    binding_plan: Qwen35CheckpointBindingPlan,
) -> tuple[
    dict[str, Qwen35CheckpointTensorBinding],
    int,
]:
    contracts = {}
    required_bytes = 0
    for binding in binding_plan.bindings:
        if type(binding) is not Qwen35CheckpointTensorBinding:
            raise ValueError(
                "binding plan entries must be exact "
                "Qwen35CheckpointTensorBinding values"
            )
        source = binding.load.weight.source
        metadata = binding.load.metadata
        existing = contracts.get(source.name)
        if existing is not None:
            if (
                existing.load.weight.source.shard != source.shard
                or existing.load.metadata != metadata
            ):
                raise ValueError(
                    f"conflicting checkpoint source contract: {source.name}"
                )
            continue
        byte_width = _DTYPE_BYTES.get(metadata.dtype)
        if byte_width is None:
            raise ValueError(
                f"unsupported checkpoint reader dtype: {metadata.dtype}"
            )
        element_count = 1
        for dimension in metadata.shape:
            element_count *= dimension
        required_bytes += element_count * byte_width
        contracts[source.name] = binding
    return contracts, required_bytes


def materialize_qwen35_checkpoint_sources(
    binding_plan: Qwen35CheckpointBindingPlan,
    checkpoint_dir: str | Path,
    *,
    max_materialized_bytes: int,
) -> Qwen35CheckpointMaterialization:
    if type(binding_plan) is not Qwen35CheckpointBindingPlan:
        raise ValueError(
            "binding_plan must be an exact Qwen35CheckpointBindingPlan"
        )
    budget = _positive_budget(max_materialized_bytes)
    contracts, required_bytes = _source_contracts(binding_plan)
    if required_bytes > budget:
        raise ValueError(
            f"required checkpoint bytes {required_bytes} "
            "exceeds max_materialized_bytes"
        )
    directory = _checkpoint_directory(checkpoint_dir)

    by_shard = {}
    shard_paths = {}
    for source_name, binding in contracts.items():
        shard_name = binding.load.weight.source.shard
        by_shard.setdefault(shard_name, []).append(source_name)
    for shard_name in by_shard:
        shard_paths[shard_name] = _safe_shard_path(
            directory,
            shard_name,
        )

    materialized = {}
    for shard_name in sorted(by_shard):
        requested = tuple(sorted(by_shard[shard_name]))
        with safe_open(
            shard_paths[shard_name],
            framework="pt",
            device="cpu",
        ) as handle:
            available = set(handle.keys())
            for source_name in requested:
                if source_name not in available:
                    raise ValueError(
                        f"missing requested source in shard: {source_name}"
                    )
                tensor = handle.get_tensor(source_name)
                binding = contracts[source_name]
                metadata = binding.load.metadata
                if not isinstance(tensor, torch.Tensor):
                    raise ValueError(
                        f"materialized source must be a tensor: {source_name}"
                    )
                if tensor.device.type != "cpu":
                    raise ValueError(
                        f"materialized source must be CPU: {source_name}"
                    )
                if tensor.dtype != _TORCH_DTYPES[metadata.dtype]:
                    raise ValueError(
                        f"materialized source dtype mismatch: {source_name}"
                    )
                if tuple(tensor.shape) != metadata.shape:
                    raise ValueError(
                        f"materialized source shape mismatch: {source_name}"
                    )
                if (
                    tensor.numel() * tensor.element_size()
                    != (
                        metadata.data_offsets[1]
                        - metadata.data_offsets[0]
                    )
                ):
                    raise ValueError(
                        f"materialized source byte mismatch: {source_name}"
                    )
                materialized[source_name] = tensor

    if set(materialized) != set(contracts):
        raise ValueError("checkpoint materialization coverage is incomplete")
    return Qwen35CheckpointMaterialization(
        source_tensors=MappingProxyType(materialized),
        source_count=len(materialized),
        shard_count=len(by_shard),
        materialized_bytes=required_bytes,
    )


def load_and_assign_qwen35_checkpoint(
    binding_plan: Qwen35CheckpointBindingPlan,
    checkpoint_dir: str | Path,
    *,
    max_materialized_bytes: int,
) -> Qwen35CheckpointLoadResult:
    materialization = materialize_qwen35_checkpoint_sources(
        binding_plan,
        checkpoint_dir,
        max_materialized_bytes=max_materialized_bytes,
    )
    assignment = assign_qwen35_checkpoint_tensors(
        binding_plan,
        materialization.source_tensors,
    )
    return Qwen35CheckpointLoadResult(
        materialization=materialization,
        assignment=assignment,
    )
