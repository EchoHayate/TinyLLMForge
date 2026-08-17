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
from tinyvllm.models.qwen35_checkpoint_assignment import (
    _assign_qwen35_checkpoint_source_bindings,
)
from tinyvllm.models.qwen35_checkpoint_binding import (
    Qwen35CheckpointBindingPlan,
    Qwen35CheckpointTensorBinding,
)
from tinyvllm.models.qwen35_packed import Qwen35PackedForCausalLM


@dataclass(frozen=True)
class Qwen35StreamedCheckpointLoadStats:
    assigned_bindings: int
    source_tensors: int
    shard_count: int
    loaded_bytes: int
    peak_source_bytes: int


@dataclass(frozen=True)
class Qwen35LoadedCheckpointCandidate:
    owner: Qwen35HybridModelOwner
    binding_plan: Qwen35CheckpointBindingPlan
    stats: Qwen35StreamedCheckpointLoadStats
    model_fingerprint: str


@dataclass(frozen=True)
class _SourceContract:
    bindings: tuple[Qwen35CheckpointTensorBinding, ...]
    shard: str
    dtype: torch.dtype
    shape: tuple[int, ...]
    byte_count: int


def move_qwen35_loaded_checkpoint_candidate_to_device(
    candidate,
    device,
) -> Qwen35LoadedCheckpointCandidate:
    if type(candidate) is not Qwen35LoadedCheckpointCandidate:
        raise ValueError(
            "candidate must be an exact Qwen35LoadedCheckpointCandidate"
        )
    target_device = torch.device(device)
    if target_device.type not in ("cuda", "meta"):
        raise ValueError("candidate target device must be CUDA")
    if target_device.type == "cuda" and target_device.index is None:
        target_device = torch.device(
            "cuda",
            torch.cuda.current_device(),
        )
    owner = candidate.owner
    model = owner.model
    pool = owner.pool
    if pool.device.type != "cpu":
        raise ValueError("candidate pool must start on CPU")
    if pool._bindings:
        raise ValueError("candidate pool must be unbound")

    migrated_tensors = {
        key: tensor.to(device=target_device)
        for key, tensor in pool._tensors.items()
    }
    model.to(device=target_device)
    if any(
        tensor.device != target_device
        for tensor in (
            list(model.parameters())
            + list(model.buffers())
        )
    ):
        raise RuntimeError("candidate model migration is incomplete")
    pool._tensors = migrated_tensors
    pool.device = target_device
    for adapter in owner.state_transaction.adapters:
        if adapter.pool is not pool:
            raise RuntimeError(
                "candidate adapter lost state pool identity"
            )
        adapter.convolution = migrated_tensors[
            (adapter.layer_index, "linear_convolution")
        ]
        adapter.recurrent = migrated_tensors[
            (adapter.layer_index, "linear_recurrent")
        ]
    if owner.runtime_bridge.pool is not pool:
        raise RuntimeError(
            "candidate runtime bridge lost state pool identity"
        )
    return candidate


_DTYPES = {
    "BF16": torch.bfloat16,
    "F32": torch.float32,
}
_DTYPE_BYTES = {
    "BF16": 2,
    "F32": 4,
}


def _positive_budget(value) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError(
            "max_tensor_bytes must be a positive integer"
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


def _source_contracts(
    binding_plan: Qwen35CheckpointBindingPlan,
    budget: int,
) -> tuple[dict[str, _SourceContract], int, int]:
    grouped = {}
    for binding in binding_plan.bindings:
        source_name = binding.load.weight.source.name
        grouped.setdefault(source_name, []).append(binding)

    contracts = {}
    total_bytes = 0
    peak_bytes = 0
    for source_name, source_bindings in grouped.items():
        first = source_bindings[0]
        shard = first.load.weight.source.shard
        metadata = first.load.metadata
        for binding in source_bindings[1:]:
            if (
                binding.load.weight.source.shard != shard
                or binding.load.metadata != metadata
            ):
                raise ValueError(
                    "conflicting checkpoint source contract: "
                    f"{source_name}"
                )
        dtype = _DTYPES.get(metadata.dtype)
        byte_width = _DTYPE_BYTES.get(metadata.dtype)
        if dtype is None or byte_width is None:
            raise ValueError(
                f"unsupported checkpoint streaming dtype: {metadata.dtype}"
            )
        element_count = 1
        for dimension in metadata.shape:
            element_count *= dimension
        byte_count = element_count * byte_width
        if (
            metadata.data_offsets[1]
            - metadata.data_offsets[0]
            != byte_count
        ):
            raise ValueError(
                f"checkpoint metadata byte mismatch: {source_name}"
            )
        if byte_count > budget:
            raise ValueError(
                f"checkpoint source bytes {byte_count} "
                "exceeds max_tensor_bytes"
            )
        contracts[source_name] = _SourceContract(
            bindings=tuple(source_bindings),
            shard=shard,
            dtype=dtype,
            shape=metadata.shape,
            byte_count=byte_count,
        )
        total_bytes += byte_count
        peak_bytes = max(peak_bytes, byte_count)
    return contracts, total_bytes, peak_bytes


def _validate_materialized_source(
    source_name: str,
    tensor,
    contract: _SourceContract,
) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(
            f"materialized source must be a tensor: {source_name}"
        )
    if tensor.device.type != "cpu":
        raise ValueError(
            f"materialized source must be CPU: {source_name}"
        )
    if tensor.dtype != contract.dtype:
        raise ValueError(
            f"materialized source dtype mismatch: {source_name}"
        )
    if tuple(tensor.shape) != contract.shape:
        raise ValueError(
            f"materialized source shape mismatch: {source_name}"
        )
    if tensor.numel() * tensor.element_size() != contract.byte_count:
        raise ValueError(
            f"materialized source byte mismatch: {source_name}"
        )
    return tensor


def load_qwen35_fresh_checkpoint_candidate(
    candidate_factory: Callable[
        [],
        tuple[
            Qwen35PackedForCausalLM,
            Qwen35CheckpointBindingPlan,
        ],
    ],
    checkpoint_dir: str | Path,
    *,
    max_tensor_bytes: int,
    model_fingerprint: str,
) -> Qwen35LoadedCheckpointCandidate:
    if not callable(candidate_factory):
        raise ValueError("candidate_factory must be callable")
    budget = _positive_budget(max_tensor_bytes)
    model_fingerprint = validate_qwen35_model_fingerprint(
        model_fingerprint
    )
    model, binding_plan = _validate_candidate(candidate_factory())
    contracts, total_bytes, peak_bytes = _source_contracts(
        binding_plan,
        budget,
    )
    directory = _checkpoint_directory(checkpoint_dir)

    by_shard = {}
    shard_paths = {}
    for source_name, contract in contracts.items():
        by_shard.setdefault(contract.shard, []).append(source_name)
    for shard_name in by_shard:
        shard_paths[shard_name] = _safe_shard_path(
            directory,
            shard_name,
        )

    assigned_bindings = 0
    loaded_sources = 0
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
                        "missing requested source in shard: "
                        f"{source_name}"
                    )
                contract = contracts[source_name]
                source = _validate_materialized_source(
                    source_name,
                    handle.get_tensor(source_name),
                    contract,
                )
                assigned_bindings += (
                    _assign_qwen35_checkpoint_source_bindings(
                        contract.bindings,
                        source,
                        tensor_parallel_size=(
                            binding_plan.tensor_parallel_size
                        ),
                        tensor_parallel_rank=(
                            binding_plan.tensor_parallel_rank
                        ),
                    )
                )
                loaded_sources += 1
                del source

    if loaded_sources != len(contracts):
        raise ValueError("streamed checkpoint source coverage is incomplete")
    owner = build_qwen35_hybrid_model_owner(model)
    return Qwen35LoadedCheckpointCandidate(
        owner=owner,
        binding_plan=binding_plan,
        stats=Qwen35StreamedCheckpointLoadStats(
            assigned_bindings=assigned_bindings,
            source_tensors=loaded_sources,
            shard_count=len(by_shard),
            loaded_bytes=total_bytes,
            peak_source_bytes=peak_bytes,
        ),
        model_fingerprint=model_fingerprint,
    )
