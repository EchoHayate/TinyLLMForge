from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
from collections.abc import Mapping
import shutil

import torch
from torch import nn


def _load_tp4_contract():
    path = (
        Path(__file__).resolve().parent
        / "qwen35_tp4_real_root_logit_correctness_contract.py"
    )
    module_name = "qwen35_tp4_real_root_logit_correctness_contract"
    module = sys.modules.get(module_name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_tp1_preflight():
    path = (
        Path(__file__).resolve().parent
        / "qwen35_tp1_real_root_logit_correctness_preflight.py"
    )
    module_name = "qwen35_tp1_real_root_logit_correctness_preflight"
    module = sys.modules.get(module_name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_TP1_PREFLIGHT = _load_tp1_preflight()
_TP4_CONTRACT = _load_tp4_contract()

APPROVED_MODEL_DIR = _TP1_PREFLIGHT.APPROVED_MODEL_DIR
APPROVED_MODEL_MANIFEST_SHA256 = (
    _TP1_PREFLIGHT.APPROVED_MODEL_MANIFEST_SHA256
)
APPROVED_CONFIG_SHA256 = _TP1_PREFLIGHT.APPROVED_CONFIG_SHA256
APPROVED_INDEX_SHA256 = _TP1_PREFLIGHT.APPROVED_INDEX_SHA256
APPROVED_SHARD_NAME = _TP1_PREFLIGHT.APPROVED_SHARD_NAME
APPROVED_SHARD_SIZE = _TP1_PREFLIGHT.APPROVED_SHARD_SIZE
APPROVED_SHARD_SHA256 = _TP1_PREFLIGHT.APPROVED_SHARD_SHA256
APPROVED_COMPOSITE_SHA256 = _TP1_PREFLIGHT.APPROVED_COMPOSITE_SHA256
AUTHORIZATION_SHA256 = _TP1_PREFLIGHT.AUTHORIZATION_SHA256
MAX_TENSOR_BYTES = _TP1_PREFLIGHT.MAX_TENSOR_BYTES
MODEL_VOCAB_SIZE = _TP4_CONTRACT.MODEL_VOCAB_SIZE
MIN_GPU_FREE_BYTES = _TP4_CONTRACT.MIN_GPU_FREE_BYTES
TP4_RESULT_NAME = "tp4_real_root_logit_correctness.json"
TP4_REFERENCE_LOGITS_NAME = "reference_logits.pt"
TP4_NATIVE_RANK0_LOGITS_NAME = "native_rank0_logits.pt"
TP4_RANK_EVIDENCE_NAME = "rank_evidence.json"
TP4_SOURCE_MANIFEST_NAME = "source_manifest.json"
TP4_ARTIFACT_NAMES = (
    TP4_RESULT_NAME,
    TP4_REFERENCE_LOGITS_NAME,
    TP4_NATIVE_RANK0_LOGITS_NAME,
    TP4_RANK_EVIDENCE_NAME,
    TP4_SOURCE_MANIFEST_NAME,
)
TP1_PREREQUISITE = {
    "run_tag": "qwen35-tp1-authority-20260728-195153-r2",
    "classification": "PASS",
    "source_tree_sha256": (
        "e5da50970951b32a61ecc9f85cf1cd447dc95950856345d780cd9e089bdf11ab"
    ),
    "artifacts": {
        "tp1_real_root_logit_correctness.json": (
            "39c4bbc548a82e915609dccb57101a6e64cbc92319d5ba69c44d427f8f4aa519"
        ),
        "reference_logits.pt": (
            "3373ab6038d6f21cf6421aa0e1c9146cc001e6e8bcbd06e6ab59c65ecc709e5a"
        ),
        "native_logits.pt": (
            "5d3fcb3a204a1b1bd673026d83f06ddc76422ca49d51d6af32671ba153ecb4d4"
        ),
        "source_manifest.json": (
            "0633a6ad5913d0d8a28526c1ec05f2cb17e347c180a6c93fa58fc3674fcb2207"
        ),
    },
}


class Qwen35TP4CausalAttentionBackend(nn.Module):

    def __init__(
        self,
        *,
        local_query_heads: int,
        local_kv_heads: int,
        head_dim: int,
    ):
        super().__init__()
        if local_query_heads != 2:
            raise ValueError("local_query_heads must equal 2")
        if local_kv_heads != 1:
            raise ValueError("local_kv_heads must equal 1")
        if (
            isinstance(head_dim, bool)
            or not isinstance(head_dim, int)
            or head_dim <= 0
        ):
            raise ValueError("head_dim must be a positive integer")
        self.local_query_heads = local_query_heads
        self.local_kv_heads = local_kv_heads
        self.head_dim = head_dim

    @staticmethod
    def _validate_tensor(
        tensor,
        *,
        name: str,
        token_count: int | None,
        width: int,
        dtype: torch.dtype | None,
        device: torch.device | None,
    ) -> torch.Tensor:
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{name} must be a tensor")
        if tensor.ndim != 2:
            raise ValueError(f"{name} must be rank two")
        if tensor.shape[1] != width:
            raise ValueError(f"{name} width must equal {width}")
        if token_count is not None and tensor.shape[0] != token_count:
            raise ValueError(f"{name} token count must match query")
        if not tensor.is_floating_point():
            raise ValueError(f"{name} must use a floating point dtype")
        if dtype is not None and tensor.dtype != dtype:
            raise ValueError(f"{name} dtype must match query")
        if device is not None and tensor.device != device:
            raise ValueError(f"{name} device must match query")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} must be finite")
        return tensor

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        query_width = self.local_query_heads * self.head_dim
        kv_width = self.local_kv_heads * self.head_dim
        query = self._validate_tensor(
            query,
            name="query",
            token_count=None,
            width=query_width,
            dtype=None,
            device=None,
        )
        key = self._validate_tensor(
            key,
            name="key",
            token_count=query.shape[0],
            width=kv_width,
            dtype=query.dtype,
            device=query.device,
        )
        value = self._validate_tensor(
            value,
            name="value",
            token_count=query.shape[0],
            width=kv_width,
            dtype=query.dtype,
            device=query.device,
        )

        token_count = query.shape[0]
        query_heads = query.reshape(
            token_count,
            self.local_query_heads,
            self.head_dim,
        )
        key_heads = key.reshape(
            token_count,
            self.local_kv_heads,
            self.head_dim,
        ).repeat_interleave(
            self.local_query_heads // self.local_kv_heads,
            dim=1,
        )
        value_heads = value.reshape(
            token_count,
            self.local_kv_heads,
            self.head_dim,
        ).repeat_interleave(
            self.local_query_heads // self.local_kv_heads,
            dim=1,
        )
        scores = torch.einsum(
            "thd,shd->hts",
            query_heads.float(),
            key_heads.float(),
        )
        scores = scores * (self.head_dim ** -0.5)
        causal_mask = torch.ones(
            token_count,
            token_count,
            dtype=torch.bool,
            device=query.device,
        ).tril()
        scores = scores.masked_fill(
            ~causal_mask.unsqueeze(0),
            -math.inf,
        )
        probabilities = torch.softmax(scores, dim=-1)
        output = torch.einsum(
            "hts,shd->thd",
            probabilities,
            value_heads.float(),
        )
        return output.to(dtype=query.dtype).reshape(
            token_count,
            query_width,
        )


@dataclass(frozen=True)
class RealTP4CPUCandidate:
    candidate: object
    pool: object
    metadata: object
    tensor_plan: object
    rank: int


@dataclass(frozen=True)
class NativeTP4CaseResult:
    case_id: str
    rank: int
    request_id: int
    lease_generation: int
    token_count: int
    logits: torch.Tensor | None
    state_nonzero_after_commit: dict[str, bool]
    release_zeroed: bool
    pool_binding_released: bool


class _RealTP4Dependencies(_TP1_PREFLIGHT._RealTP1Dependencies):
    pass


def _tp4_rank(value) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value >= 4
    ):
        raise ValueError("rank must be in [0, 4)")
    return value


def _candidate_parts(
    candidate,
    *,
    rank: int,
    expected_model_fingerprint: str,
    require_run_step: bool = True,
):
    rank = _tp4_rank(rank)
    if (
        getattr(candidate, "model_fingerprint", None)
        != expected_model_fingerprint
    ):
        raise ValueError("candidate model fingerprint mismatch")
    binding_plan = getattr(candidate, "binding_plan", None)
    if (
        getattr(binding_plan, "tensor_parallel_size", None) != 4
        or getattr(binding_plan, "tensor_parallel_rank", None) != rank
    ):
        raise ValueError("candidate TP4 binding rank mismatch")
    owner = getattr(candidate, "owner", None)
    model = getattr(owner, "model", None)
    pool = getattr(owner, "pool", None)
    if model is None:
        raise ValueError("candidate owner model is invalid")
    if require_run_step and not callable(getattr(model, "run_step", None)):
        raise ValueError("candidate owner model run_step is invalid")
    if pool is None:
        raise ValueError("candidate owner pool is invalid")
    if getattr(model, "pool", pool) is not pool:
        raise ValueError("candidate model and owner pool identity mismatch")
    return model, pool


def _build_attention_backend(
    _layer_index: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
) -> Qwen35TP4CausalAttentionBackend:
    return Qwen35TP4CausalAttentionBackend(
        local_query_heads=query_heads,
        local_kv_heads=kv_heads,
        head_dim=head_dim,
    )


def build_real_tp4_cpu_candidate(
    *,
    rank: int,
    dependencies=None,
) -> RealTP4CPUCandidate:
    rank = _tp4_rank(rank)
    dependencies = (
        _RealTP4Dependencies()
        if dependencies is None
        else dependencies
    )
    required = (
        "make_shard_identity",
        "read_metadata",
        "build_tensor_plan",
        "build_layout",
        "make_pool",
        "prepare_target",
        "build_loader",
        "make_request",
    )
    if any(
        not callable(getattr(dependencies, name, None))
        for name in required
    ):
        raise ValueError("real TP4 dependencies are incomplete")
    shard = dependencies.make_shard_identity(
        name=APPROVED_SHARD_NAME,
        size=APPROVED_SHARD_SIZE,
        sha256=APPROVED_SHARD_SHA256,
    )
    metadata = dependencies.read_metadata(
        APPROVED_MODEL_DIR,
        shards=(shard,),
        expected_config_sha256=APPROVED_CONFIG_SHA256,
        expected_index_sha256=APPROVED_INDEX_SHA256,
        expected_config_index_header_sha256=APPROVED_COMPOSITE_SHA256,
    )
    tensor_plan = dependencies.build_tensor_plan(
        metadata.hf_config,
        metadata.index_payload,
        metadata.shard_headers,
    )
    layout = dependencies.build_layout(
        metadata.hf_config,
        tensor_parallel_size=4,
        dtype=torch.bfloat16,
        recurrent_dtype=torch.float32,
        speculative_tokens=1,
    )
    pool = dependencies.make_pool(
        layout,
        capacity=1,
        device="cpu",
    )
    target = dependencies.prepare_target(
        metadata.hf_config,
        tensor_plan,
        pool=pool,
        tensor_parallel_size=4,
        tensor_parallel_rank=rank,
        build_attention_backend=_build_attention_backend,
        parameter_device="cpu",
    )
    if getattr(target, "pool", None) is not pool:
        raise ValueError("prepared target pool identity mismatch")
    provider_calls = 0

    def provide_target():
        nonlocal provider_calls
        provider_calls += 1
        if provider_calls != 1:
            raise RuntimeError(
                "prepared target provider called more than once"
            )
        return target

    loader = dependencies.build_loader(
        provide_target,
        authorization_sha256=AUTHORIZATION_SHA256,
    )
    request = dependencies.make_request(
        checkpoint_dir=APPROVED_MODEL_DIR,
        model_fingerprint=APPROVED_MODEL_MANIFEST_SHA256,
        max_tensor_bytes=MAX_TENSOR_BYTES,
        authorization_sha256=AUTHORIZATION_SHA256,
    )
    candidate = loader(request)
    _, candidate_pool = _candidate_parts(
        candidate,
        rank=rank,
        expected_model_fingerprint=APPROVED_MODEL_MANIFEST_SHA256,
        require_run_step=False,
    )
    if provider_calls != 1:
        raise RuntimeError("prepared target provider call count mismatch")
    if candidate_pool is not pool:
        raise ValueError("loaded candidate pool identity mismatch")
    return RealTP4CPUCandidate(
        candidate=candidate,
        pool=pool,
        metadata=metadata,
        tensor_plan=tensor_plan,
        rank=rank,
    )


def move_loaded_tp4_candidate_to_device(
    candidate,
    *,
    rank: int,
    device: str | torch.device,
    expected_model_fingerprint: str,
):
    rank = _tp4_rank(rank)
    model, pool = _candidate_parts(
        candidate,
        rank=rank,
        expected_model_fingerprint=expected_model_fingerprint,
        require_run_step=False,
    )
    try:
        target_device = torch.device(device)
    except (TypeError, RuntimeError) as error:
        raise ValueError("device is invalid") from error
    if target_device.type not in ("cuda", "meta"):
        raise ValueError("loaded candidate target device must be CUDA")
    if getattr(pool, "device", None) != torch.device("cpu"):
        raise ValueError("loaded candidate pool must start on CPU")
    tensors = getattr(pool, "_tensors", None)
    if not isinstance(tensors, dict) or not tensors:
        raise ValueError("loaded candidate pool tensor inventory is invalid")
    if getattr(pool, "_bindings", None):
        raise ValueError("loaded candidate pool must be unbound")

    migrated_tensors = {
        key: tensor.to(device=target_device)
        for key, tensor in tensors.items()
    }
    model.to(device=target_device)
    if any(
        tensor.device != target_device
        for tensor in (
            list(model.parameters())
            + list(model.buffers())
        )
    ):
        raise RuntimeError("loaded candidate model migration is incomplete")
    if any(
        tensor.device != target_device
        for tensor in migrated_tensors.values()
    ):
        raise RuntimeError("loaded candidate pool migration is incomplete")
    pool._tensors = migrated_tensors
    pool.device = target_device
    owner = candidate.owner
    transaction = getattr(owner, "state_transaction", None)
    adapters = getattr(transaction, "adapters", ())
    if (
        transaction is not None
        and getattr(transaction, "pool", None) is not pool
    ):
        raise RuntimeError("loaded candidate transaction lost pool identity")
    for adapter in adapters:
        if getattr(adapter, "pool", None) is not pool:
            raise RuntimeError("loaded candidate adapter lost pool identity")
        layer_index = getattr(adapter, "layer_index", None)
        convolution = migrated_tensors.get(
            (layer_index, "linear_convolution")
        )
        recurrent = migrated_tensors.get(
            (layer_index, "linear_recurrent")
        )
        if convolution is None or recurrent is None:
            raise RuntimeError(
                "loaded candidate adapter state mapping is incomplete"
            )
        adapter.convolution = convolution
        adapter.recurrent = recurrent
    runtime_bridge = getattr(owner, "runtime_bridge", None)
    if (
        runtime_bridge is not None
        and getattr(runtime_bridge, "pool", None) is not pool
    ):
        raise RuntimeError("loaded candidate runtime bridge lost pool identity")
    return candidate


def select_tp4_gpu_resources(
    rows,
    *,
    minimum_free_bytes: int = MIN_GPU_FREE_BYTES,
) -> tuple[dict[str, object], ...]:
    if (
        isinstance(minimum_free_bytes, bool)
        or not isinstance(minimum_free_bytes, int)
        or minimum_free_bytes <= 0
    ):
        raise ValueError("minimum_free_bytes must be positive")
    candidates = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("GPU resource row must be a dictionary")
        gpu_index = row.get("gpu_index")
        free_bytes = row.get("free_bytes")
        gpu_uuid = row.get("gpu_uuid")
        compute_processes = row.get("compute_processes")
        if (
            isinstance(gpu_index, bool)
            or not isinstance(gpu_index, int)
            or gpu_index < 0
            or isinstance(free_bytes, bool)
            or not isinstance(free_bytes, int)
            or not isinstance(gpu_uuid, str)
            or not gpu_uuid
            or not isinstance(compute_processes, list)
        ):
            raise ValueError("GPU resource row is invalid")
        if free_bytes < minimum_free_bytes or compute_processes:
            continue
        candidates.append(dict(row))
    candidates.sort(key=lambda row: row["gpu_index"])
    if len(candidates) < 4:
        raise ValueError("four eligible GPUs are required")
    selected = candidates[:4]
    if len({row["gpu_index"] for row in selected}) != 4:
        raise ValueError("selected GPU indices must be unique")
    if len({row["gpu_uuid"] for row in selected}) != 4:
        raise ValueError("selected GPU UUIDs must be unique")
    for rank, row in enumerate(selected):
        row["rank"] = rank
        row["world_size"] = 4
        row["minimum_free_bytes"] = minimum_free_bytes
    return tuple(selected)


def _fresh_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def fresh_port_pair() -> tuple[int, int]:
    first = _fresh_port()
    second = _fresh_port()
    while second == first:
        second = _fresh_port()
    return first, second


def validate_rank_evidence(rows) -> tuple[dict[str, object], ...]:
    values = tuple(rows)
    if len(values) != 4:
        raise ValueError("rank evidence must contain exactly four rows")
    normalized = []
    for row in values:
        if not isinstance(row, dict):
            raise ValueError("rank evidence row must be a dictionary")
        value = dict(row)
        rank = _tp4_rank(value.get("rank"))
        if value.get("world_size") != 4:
            raise ValueError("rank evidence world_size must equal 4")
        if (
            isinstance(value.get("pid"), bool)
            or not isinstance(value.get("pid"), int)
            or value["pid"] <= 0
        ):
            raise ValueError("rank evidence PID is invalid")
        if value.get("case_barrier_count") != 3:
            raise ValueError("rank evidence case barriers are invalid")
        if value.get("final_barrier_completed") is not True:
            raise ValueError("rank final barrier is incomplete")
        if value.get("process_group_destroyed") is not True:
            raise ValueError("rank process group was not destroyed")
        for field, label in (
            (
                "candidate_reference_dropped",
                "candidate reference was not dropped",
            ),
            (
                "model_reference_dropped",
                "model reference was not dropped",
            ),
            (
                "cuda_synchronized",
                "CUDA synchronization was incomplete",
            ),
            (
                "cuda_cache_emptied",
                "CUDA cache was not emptied",
            ),
        ):
            if value.get(field) is not True:
                raise ValueError(f"rank {label}")
        if rank == 0:
            if (
                value.get("root_logits_present") is not True
                or value.get("non_root_logits_none") is not False
            ):
                raise ValueError("rank zero logit evidence is invalid")
        elif (
            value.get("root_logits_present") is not False
            or value.get("non_root_logits_none") is not True
        ):
            raise ValueError("non-root logit evidence is invalid")
        value = _TP4_CONTRACT.validate_rank_topology(value)
        events = value.get("collective_events")
        if not isinstance(events, list) or not events:
            raise ValueError("rank collective evidence is invalid")
        if [
            event.get("ordinal") for event in events
        ] != list(range(len(events))):
            raise ValueError("rank collective order is invalid")
        gathers = [
            event for event in events
            if event.get("collective") == "gather"
        ]
        if len(gathers) != 3:
            raise ValueError("rank gather collective count is invalid")
        if any(
            event.get("destination") != 0
            or event.get("receive_count") != (4 if rank == 0 else None)
            or event.get("async_op") is not False
            for event in gathers
        ):
            raise ValueError("rank gather collective evidence is invalid")
        if not any(
            event.get("collective") == "all_reduce"
            for event in events
        ):
            raise ValueError("rank all_reduce collective evidence is invalid")
        state_rows = value.get("state_rows")
        if not isinstance(state_rows, list) or len(state_rows) != 3:
            raise ValueError("rank state evidence is invalid")
        for state in state_rows:
            mapping = state.get("state_nonzero_after_commit")
            if (
                state.get("changed_component_count") != 36
                or not isinstance(mapping, dict)
                or len(mapping) != 36
                or any(item is not True for item in mapping.values())
                or state.get("release_zeroed") is not True
                or state.get("pool_binding_released") is not True
            ):
                raise ValueError("rank state evidence is invalid")
            convolution_layers = {
                int(key.split(":", 1)[0])
                for key in mapping
                if key.endswith(":linear_convolution")
            }
            recurrent_layers = {
                int(key.split(":", 1)[0])
                for key in mapping
                if key.endswith(":linear_recurrent")
            }
            if (
                convolution_layers != set(range(18))
                or recurrent_layers != set(range(18))
            ):
                raise ValueError("rank state layer evidence is invalid")
        value["rank"] = rank
        normalized.append(value)
    normalized.sort(key=lambda row: row["rank"])
    if tuple(row["rank"] for row in normalized) != (0, 1, 2, 3):
        raise ValueError("rank evidence must cover ranks 0..3")
    for field, label in (
        ("pid", "PIDs"),
        ("gpu_index", "GPU indices"),
        ("gpu_uuid", "GPU UUIDs"),
    ):
        if len({row.get(field) for row in normalized}) != 4:
            raise ValueError(f"rank evidence {label} must be unique")
    for field in ("process_group_nonce", "rendezvous"):
        if len({row.get(field) for row in normalized}) != 1:
            raise ValueError(f"rank evidence {field} must be common")
    return tuple(normalized)


def bind_launched_rank_evidence(
    launched_rows,
    persisted_rows,
) -> tuple[dict[str, object], ...]:
    launched = tuple(launched_rows)
    persisted = tuple(persisted_rows)
    if len(launched) != 4 or len(persisted) != 4:
        raise ValueError("rank launch evidence must contain four rows")
    launched_by_rank = {
        row.get("rank"): row
        for row in launched
        if isinstance(row, dict)
    }
    if set(launched_by_rank) != {0, 1, 2, 3}:
        raise ValueError("rank launch inventory is invalid")
    bound = []
    for row in persisted:
        if not isinstance(row, dict):
            raise ValueError("persisted rank evidence is invalid")
        rank = row.get("rank")
        launch = launched_by_rank.get(rank)
        fields = (
            "world_size",
            "pid",
            "gpu_index",
            "gpu_uuid",
            "process_group_nonce",
            "rendezvous",
        )
        if (
            launch is None
            or launch.get("exit_code") != 0
            or any(row.get(field) != launch.get(field) for field in fields)
        ):
            raise ValueError("persisted rank launch identity mismatch")
        value = dict(row)
        value["worker_exited"] = True
        bound.append(value)
    return tuple(sorted(bound, key=lambda row: row["rank"]))


def validate_native_worker_gpu_identity(
    *,
    rank: int,
    expected_gpu_index: int,
    expected_gpu_uuid: str,
    visible_devices: str,
    query_gpus=None,
) -> dict[str, object]:
    rank = _tp4_rank(rank)
    if not isinstance(visible_devices, str):
        raise ValueError("CUDA_VISIBLE_DEVICES is invalid")
    try:
        indices = tuple(
            int(value.strip())
            for value in visible_devices.split(",")
        )
    except ValueError as error:
        raise ValueError("CUDA_VISIBLE_DEVICES is invalid") from error
    if len(indices) != 4 or len(set(indices)) != 4:
        raise ValueError("CUDA_VISIBLE_DEVICES must list four GPUs")
    if indices[rank] != expected_gpu_index:
        raise ValueError("native worker physical GPU index mismatch")
    if query_gpus is None:
        query_gpus = _query_tp4_gpu_resources
    if not callable(query_gpus):
        raise ValueError("native worker GPU query is invalid")
    current = {
        row.get("gpu_index"): row
        for row in query_gpus()
        if isinstance(row, dict)
    }.get(expected_gpu_index)
    if current is None:
        raise ValueError("native worker physical GPU is missing")
    if current.get("gpu_uuid") != expected_gpu_uuid:
        raise ValueError("native worker GPU UUID mismatch")
    if current.get("free_bytes", 0) < MIN_GPU_FREE_BYTES:
        raise ValueError("native worker GPU free memory is insufficient")
    if current.get("compute_processes") != []:
        raise ValueError("native worker GPU has an active compute process")
    return {
        "local_rank": rank,
        "physical_gpu_index": expected_gpu_index,
        "gpu_uuid": expected_gpu_uuid,
        "free_bytes": current["free_bytes"],
    }


def _positive_port(value, *, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
        or value > 65535
    ):
        raise ValueError(f"{name} must be a valid TCP port")
    return value


def launch_native_rank_group(
    *,
    selected_gpus,
    rendezvous: str,
    process_group_nonce: str,
    tinyvllm_dist_port: int,
    master_port: int,
    process_factory,
    timeout_seconds: int,
    pid_alive,
    base_environment,
) -> tuple[dict[str, object], ...]:
    assignments = _TP4_CONTRACT.validate_gpu_assignments(selected_gpus)
    if (
        not isinstance(rendezvous, str)
        or not rendezvous.startswith("tcp://")
    ):
        raise ValueError("rendezvous must be a TCP URL")
    if (
        not isinstance(process_group_nonce, str)
        or len(process_group_nonce) < 16
    ):
        raise ValueError("process_group_nonce is invalid")
    dist_port = _positive_port(
        tinyvllm_dist_port,
        name="TINYVLLM_DIST_PORT",
    )
    master = _positive_port(master_port, name="MASTER_PORT")
    if dist_port == master:
        raise ValueError(
            "TINYVLLM_DIST_PORT and MASTER_PORT ports must be distinct"
        )
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, int)
        or timeout_seconds <= 0
    ):
        raise ValueError("timeout_seconds must be positive")
    if not callable(process_factory) or not callable(pid_alive):
        raise ValueError("native process dependencies are incomplete")
    if not isinstance(base_environment, dict):
        raise ValueError("base_environment must be a dictionary")

    visible_devices = ",".join(
        str(row["gpu_index"]) for row in assignments
    )
    processes = []
    for row in assignments:
        environment = dict(base_environment)
        environment.update({
            "CUDA_VISIBLE_DEVICES": visible_devices,
            "TINYVLLM_DIST_PORT": str(dist_port),
            "MASTER_PORT": str(master),
            "TINYVLLM_GATE_LOCAL_RANK": str(row["rank"]),
            "TINYVLLM_GATE_PHYSICAL_GPU_INDEX": str(row["gpu_index"]),
            "TINYVLLM_GATE_GPU_UUID": str(row["gpu_uuid"]),
            "TINYVLLM_GATE_PROCESS_GROUP_NONCE": process_group_nonce,
            "TINYVLLM_GATE_RENDEZVOUS": rendezvous,
        })
        process = process_factory(
            rank=row["rank"],
            world_size=4,
            gpu_index=row["gpu_index"],
            gpu_uuid=row["gpu_uuid"],
            rendezvous=rendezvous,
            process_group_nonce=process_group_nonce,
            environment=environment,
        )
        if any(
            not callable(getattr(process, name, None))
            for name in (
                "start",
                "join",
                "is_alive",
                "terminate",
                "kill",
            )
        ):
            raise ValueError("native process object is invalid")
        processes.append((row, process))

    try:
        for _row, process in processes:
            process.start()
        for _row, process in processes:
            process.join(timeout_seconds)
        alive = [
            process
            for _row, process in processes
            if bool(process.is_alive())
        ]
        if alive:
            for process in alive:
                process.terminate()
            for process in alive:
                process.join(timeout_seconds)
            raise ValueError("native rank group timed out")
        rows = []
        for assignment, process in processes:
            exit_code = getattr(process, "exitcode", None)
            pid = getattr(process, "pid", None)
            if exit_code != 0:
                raise ValueError(
                    f"native rank {assignment['rank']} exited with "
                    f"code {exit_code}"
                )
            if (
                isinstance(pid, bool)
                or not isinstance(pid, int)
                or pid <= 0
            ):
                raise ValueError("native rank PID is invalid")
            if pid_alive(pid):
                raise ValueError("native worker PID survived completion")
            rows.append({
                **assignment,
                "pid": pid,
                "exit_code": exit_code,
                "process_group_nonce": process_group_nonce,
                "rendezvous": rendezvous,
            })
        return tuple(rows)
    except BaseException:
        cleanup_timeout = min(timeout_seconds, 30)
        cleanup = [
            process
            for _row, process in processes
            if bool(process.is_alive())
        ]
        for process in cleanup:
            process.terminate()
        for process in cleanup:
            process.join(cleanup_timeout)
        survivors = [
            process for process in cleanup if bool(process.is_alive())
        ]
        for process in survivors:
            process.kill()
        for process in survivors:
            process.join(cleanup_timeout)
        for process in cleanup:
            if bool(process.is_alive()):
                raise RuntimeError(
                    "native worker survived emergency cleanup"
                )
        raise


def execute_reference_then_native_group(
    *,
    reference_worker,
    native_launcher,
    pid_alive,
) -> tuple[dict[str, object], ...]:
    if not all(
        callable(value)
        for value in (reference_worker, native_launcher, pid_alive)
    ):
        raise ValueError("coordinator dependencies are incomplete")
    reference_row = reference_worker()
    if not isinstance(reference_row, dict):
        raise ValueError("reference worker row is invalid")
    reference_pid = reference_row.get("pid")
    if (
        isinstance(reference_pid, bool)
        or not isinstance(reference_pid, int)
        or reference_pid <= 0
        or pid_alive(reference_pid)
    ):
        raise ValueError(
            "reference worker PID did not disappear before native startup"
        )
    if reference_row.get("exit_code") != 0:
        raise ValueError("reference worker failed")
    rows = tuple(native_launcher())
    ranks = tuple(row.get("rank") for row in rows if isinstance(row, dict))
    if len(rows) != 4 or ranks != (0, 1, 2, 3):
        raise ValueError("native rank group row inventory is invalid")
    for row in rows:
        pid = row.get("pid")
        if (
            isinstance(pid, bool)
            or not isinstance(pid, int)
            or pid <= 0
            or pid_alive(pid)
        ):
            raise ValueError("native worker PID survived completion")
        if row.get("exit_code") != 0:
            raise ValueError("native worker failed")
    return rows


@contextmanager
def record_distributed_collectives(distributed):
    original_all_reduce = getattr(distributed, "all_reduce", None)
    original_gather = getattr(distributed, "gather", None)
    if not callable(original_all_reduce) or not callable(original_gather):
        raise ValueError("distributed collective functions are invalid")
    events = []

    def all_reduce(tensor, *args, **kwargs):
        events.append({
            "ordinal": len(events),
            "collective": "all_reduce",
            "shape": list(getattr(tensor, "shape", ())),
            "dtype": str(getattr(tensor, "dtype", None)),
            "async_op": bool(kwargs.get("async_op", False)),
        })
        return original_all_reduce(tensor, *args, **kwargs)

    def gather(
        tensor,
        gather_list=None,
        dst=0,
        *args,
        **kwargs,
    ):
        events.append({
            "ordinal": len(events),
            "collective": "gather",
            "shape": list(getattr(tensor, "shape", ())),
            "dtype": str(getattr(tensor, "dtype", None)),
            "destination": dst,
            "receive_count": (
                None if gather_list is None else len(gather_list)
            ),
            "async_op": bool(kwargs.get("async_op", False)),
        })
        return original_gather(
            tensor,
            gather_list,
            dst,
            *args,
            **kwargs,
        )

    distributed.all_reduce = all_reduce
    distributed.gather = gather
    try:
        yield events
    finally:
        distributed.all_reduce = original_all_reduce
        distributed.gather = original_gather


def _prompt_identity(prompt_case) -> tuple[str, tuple[int, ...]]:
    case_id = getattr(prompt_case, "case_id", None)
    token_ids = getattr(prompt_case, "token_ids", None)
    if not isinstance(case_id, str) or not case_id:
        raise ValueError("prompt case_id must be a non-empty string")
    if (
        not isinstance(token_ids, tuple)
        or not token_ids
        or any(
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or token_id < 0
            for token_id in token_ids
        )
    ):
        raise ValueError(
            "prompt token_ids must be a non-empty integer tuple"
        )
    return case_id, token_ids


def _state_nonzero(pool, lease) -> dict[str, bool]:
    tensors = getattr(pool, "_tensors", None)
    if not isinstance(tensors, dict) or not tensors:
        raise ValueError("candidate pool tensor inventory is invalid")
    result = {}
    for key, tensor in sorted(tensors.items()):
        if (
            not isinstance(key, tuple)
            or len(key) != 2
            or not isinstance(tensor, torch.Tensor)
        ):
            raise ValueError("candidate pool tensor entry is invalid")
        layer_index, role = key
        result[f"{layer_index}:{role}"] = bool(
            torch.count_nonzero(tensor[lease.slot_id])
        )
    return result


def _pool_zeroed(pool) -> bool:
    tensors = getattr(pool, "_tensors", None)
    return (
        isinstance(tensors, dict)
        and bool(tensors)
        and all(
            isinstance(tensor, torch.Tensor)
            and not bool(torch.count_nonzero(tensor))
            for tensor in tensors.values()
        )
    )


def run_tp4_native_case(
    *,
    candidate,
    rank: int,
    prompt_case,
    expected_model_fingerprint: str,
    request_id: int,
    device: str | torch.device,
    allocator,
    set_context,
    reset_context,
) -> NativeTP4CaseResult:
    rank = _tp4_rank(rank)
    model, pool = _candidate_parts(
        candidate,
        rank=rank,
        expected_model_fingerprint=expected_model_fingerprint,
    )
    case_id, token_ids = _prompt_identity(prompt_case)
    if (
        isinstance(request_id, bool)
        or not isinstance(request_id, int)
        or request_id < 0
    ):
        raise ValueError("request_id must be a non-negative integer")
    execution_device = torch.device(device)
    if getattr(pool, "device", None) != execution_device:
        raise ValueError("candidate pool device does not match execution device")
    allocate = getattr(allocator, "allocate", None)
    release_allocator = getattr(allocator, "release", None)
    activate = getattr(pool, "activate", None)
    release_pool = getattr(pool, "release", None)
    if not all(
        callable(value)
        for value in (
            allocate,
            release_allocator,
            activate,
            release_pool,
            set_context,
            reset_context,
        )
    ):
        raise ValueError("native case dependencies are incomplete")

    lease = allocate(request_id)
    pool_activated = False
    state_nonzero = {}
    final_logits = None
    try:
        activate(lease)
        pool_activated = True
        input_ids = torch.tensor(
            token_ids,
            dtype=torch.int64,
            device=execution_device,
        )
        position_ids = torch.arange(
            len(token_ids),
            dtype=torch.int64,
            device=execution_device,
        )
        cumulative = torch.tensor(
            [0, len(token_ids)],
            dtype=torch.int32,
            device=execution_device,
        )
        context_set = False
        try:
            set_context(
                is_prefill=True,
                mode="prefill",
                cu_seqlens_q=cumulative,
                cu_seqlens_k=cumulative.clone(),
                max_seqlen_q=len(token_ids),
                max_seqlen_k=len(token_ids),
                logits_indices=torch.tensor(
                    [len(token_ids) - 1],
                    dtype=torch.int64,
                    device=execution_device,
                ),
            )
            context_set = True
            with torch.no_grad():
                normalized, logits = model.run_step(
                    (lease,),
                    (len(token_ids),),
                    input_ids,
                    position_ids,
                )
        finally:
            if context_set:
                reset_context()
        if (
            not isinstance(normalized, torch.Tensor)
            or normalized.ndim != 2
            or normalized.shape[0] != len(token_ids)
            or not normalized.is_floating_point()
            or not bool(torch.isfinite(normalized).all())
        ):
            raise ValueError(
                "native normalized output must be finite rank two"
            )
        if rank == 0 and isinstance(logits, torch.Tensor) and logits.ndim == 2:
            if logits.shape[0] != 1:
                raise ValueError("rank zero logits must contain one row")
            logits = logits[0]
        validated = _TP4_CONTRACT.validate_rank_logits(
            rank=rank,
            world_size=4,
            logits=logits,
        )
        if validated is not None:
            final_logits = validated.detach().to(
                device="cpu",
                dtype=torch.float32,
            ).contiguous()
        state_nonzero = _state_nonzero(pool, lease)
    finally:
        if pool_activated:
            release_pool(lease)
        release_allocator(lease)

    release_zeroed = _pool_zeroed(pool)
    bindings = getattr(pool, "_bindings", None)
    binding_released = isinstance(bindings, dict) and not bindings
    if not release_zeroed:
        raise RuntimeError("native pool was not zeroed after release")
    if not binding_released:
        raise RuntimeError("native pool binding survived release")
    return NativeTP4CaseResult(
        case_id=case_id,
        rank=rank,
        request_id=request_id,
        lease_generation=int(lease.generation),
        token_count=len(token_ids),
        logits=final_logits,
        state_nonzero_after_commit=state_nonzero,
        release_zeroed=release_zeroed,
        pool_binding_released=binding_released,
    )


def run_tp4_native_cases(
    *,
    candidate,
    rank: int,
    prompt_cases,
    expected_model_fingerprint: str,
    device: str | torch.device,
    allocator,
    first_request_id: int,
    set_context,
    reset_context,
    barrier,
) -> tuple[NativeTP4CaseResult, ...]:
    cases = tuple(prompt_cases)
    if not cases:
        raise ValueError("prompt_cases must not be empty")
    if not callable(barrier):
        raise ValueError("barrier must be callable")
    results = []
    for index, prompt_case in enumerate(cases):
        result = run_tp4_native_case(
            candidate=candidate,
            rank=rank,
            prompt_case=prompt_case,
            expected_model_fingerprint=expected_model_fingerprint,
            request_id=first_request_id + index,
            device=device,
            allocator=allocator,
            set_context=set_context,
            reset_context=reset_context,
        )
        barrier(result.case_id)
        results.append(result)
    return tuple(results)


def execute_native_rank_scope(
    *,
    rank: int,
    world_size: int,
    rendezvous: str,
    process_group_nonce: str,
    prompt_cases,
    build_candidate,
    move_candidate,
    run_cases,
    allocator_factory,
    set_context=None,
    reset_context=None,
    distributed=None,
    cuda=None,
) -> dict[str, object]:
    rank = _tp4_rank(rank)
    if world_size != 4:
        raise ValueError("world_size must equal 4")
    if (
        not isinstance(rendezvous, str)
        or not rendezvous.startswith("tcp://")
    ):
        raise ValueError("rendezvous must be a TCP URL")
    if (
        not isinstance(process_group_nonce, str)
        or len(process_group_nonce) < 16
    ):
        raise ValueError("process_group_nonce is invalid")
    cases = tuple(prompt_cases)
    if not cases:
        raise ValueError("prompt_cases must not be empty")
    for name, value in (
        ("build_candidate", build_candidate),
        ("move_candidate", move_candidate),
        ("run_cases", run_cases),
        ("allocator_factory", allocator_factory),
    ):
        if not callable(value):
            raise ValueError(f"{name} must be callable")
    distributed = torch.distributed if distributed is None else distributed
    cuda = torch.cuda if cuda is None else cuda
    if set_context is None or reset_context is None:
        from tinyvllm.utils.context import (
            reset_context as production_reset_context,
            set_context as production_set_context,
        )
        if set_context is None:
            set_context = production_set_context
        if reset_context is None:
            reset_context = production_reset_context
    if not callable(set_context) or not callable(reset_context):
        raise ValueError("context setters must be callable")

    process_group_initialized = False
    process_group_destroyed = False
    candidate_reference_dropped = False
    model_reference_dropped = False
    cuda_synchronized = False
    cuda_cache_emptied = False
    final_barrier_completed = False
    case_barrier_count = 0
    built = None
    candidate = None
    model = None
    results = None
    cuda.set_device(rank)
    try:
        distributed.init_process_group(
            backend="nccl",
            init_method=rendezvous,
            world_size=world_size,
            rank=rank,
        )
        process_group_initialized = True
        built = build_candidate(rank)
        candidate = getattr(built, "candidate", None)
        if candidate is None:
            raise ValueError("native rank candidate build is invalid")
        candidate = move_candidate(
            candidate,
            rank=rank,
            device=f"cuda:{rank}",
            expected_model_fingerprint=APPROVED_MODEL_MANIFEST_SHA256,
        )
        model = getattr(getattr(candidate, "owner", None), "model", None)
        allocator = allocator_factory(1)

        def case_barrier(_case_id):
            nonlocal case_barrier_count
            distributed.barrier()
            case_barrier_count += 1

        with record_distributed_collectives(distributed) as collective_events:
            results = run_cases(
                candidate=candidate,
                rank=rank,
                prompt_cases=cases,
                expected_model_fingerprint=APPROVED_MODEL_MANIFEST_SHA256,
                device=f"cuda:{rank}",
                allocator=allocator,
                first_request_id=100,
                set_context=set_context,
                reset_context=reset_context,
                barrier=case_barrier,
            )
        if not isinstance(results, tuple) or len(results) != len(cases):
            raise ValueError("native rank result inventory is invalid")
        distributed.barrier()
        final_barrier_completed = True
    finally:
        built = None
        candidate = None
        candidate_reference_dropped = True
        model = None
        model_reference_dropped = True
        if (
            process_group_initialized
            and bool(distributed.is_initialized())
        ):
            distributed.destroy_process_group()
            process_group_destroyed = True
        cuda.synchronize()
        cuda_synchronized = True
        cuda.empty_cache()
        cuda_cache_emptied = True

    return {
        "rank": rank,
        "world_size": world_size,
        "process_group_nonce": process_group_nonce,
        "rendezvous": rendezvous,
        "case_barrier_count": case_barrier_count,
        "final_barrier_completed": final_barrier_completed,
        "process_group_destroyed": process_group_destroyed,
        "candidate_reference_dropped": candidate_reference_dropped,
        "model_reference_dropped": model_reference_dropped,
        "cuda_synchronized": cuda_synchronized,
        "cuda_cache_emptied": cuda_cache_emptied,
        "collective_events": collective_events,
        "results": results,
    }


def _worker_atomic_write_json(path: Path, payload) -> None:
    temporary = path.with_name(path.name + ".write")
    if path.exists() or temporary.exists():
        raise ValueError("native rank worker output already exists")
    try:
        temporary.write_text(
            json.dumps(
                payload,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def _worker_atomic_torch_save(path: Path, payload) -> None:
    temporary = path.with_name(path.name + ".write")
    if path.exists() or temporary.exists():
        raise ValueError("native rank worker output already exists")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def execute_native_rank_worker(
    *,
    rank: int,
    world_size: int,
    rendezvous: str,
    process_group_nonce: str,
    prompt_cases,
    gpu_index: int,
    gpu_uuid: str,
    rank_output,
    logits_output,
    process_id: int | None = None,
    scope_runner=execute_native_rank_scope,
    build_candidate=build_real_tp4_cpu_candidate,
    move_candidate=move_loaded_tp4_candidate_to_device,
    run_cases=run_tp4_native_cases,
    allocator_factory=None,
    environment=None,
    query_gpus=None,
) -> dict[str, object]:
    rank = _tp4_rank(rank)
    if world_size != 4:
        raise ValueError("world_size must equal 4")
    cases = tuple(prompt_cases)
    case_ids = tuple(_prompt_identity(case)[0] for case in cases)
    if not cases or len(set(case_ids)) != len(case_ids):
        raise ValueError("native rank prompt cases are invalid")
    if (
        isinstance(gpu_index, bool)
        or not isinstance(gpu_index, int)
        or gpu_index < 0
    ):
        raise ValueError("gpu_index must be non-negative")
    if not isinstance(gpu_uuid, str) or not gpu_uuid:
        raise ValueError("gpu_uuid must be non-empty")
    pid = os.getpid() if process_id is None else process_id
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise ValueError("process_id must be positive")
    row_path = Path(rank_output)
    logits_path = None if logits_output is None else Path(logits_output)
    if not row_path.parent.is_dir():
        raise ValueError("native rank output directory is missing")
    if logits_path is not None and logits_path.parent != row_path.parent:
        raise ValueError("native rank outputs must share one directory")
    if rank == 0 and logits_path is None:
        raise ValueError("rank zero logits output is required")
    if rank != 0 and logits_path is not None:
        raise ValueError("non-root logits output must be None")
    if not callable(scope_runner):
        raise ValueError("scope_runner must be callable")
    environment = os.environ if environment is None else environment
    if not isinstance(environment, Mapping):
        raise ValueError("native worker environment must be a mapping")
    visible_devices = environment.get("CUDA_VISIBLE_DEVICES")
    if visible_devices is not None:
        validate_native_worker_gpu_identity(
            rank=rank,
            expected_gpu_index=gpu_index,
            expected_gpu_uuid=gpu_uuid,
            visible_devices=visible_devices,
            query_gpus=query_gpus,
        )
    if allocator_factory is None:
        from tinyvllm.engine.hybrid_state import HybridStateSlotAllocator
        allocator_factory = HybridStateSlotAllocator

    published = []
    try:
        scope = scope_runner(
            rank=rank,
            world_size=world_size,
            rendezvous=rendezvous,
            process_group_nonce=process_group_nonce,
            prompt_cases=cases,
            build_candidate=build_candidate,
            move_candidate=move_candidate,
            run_cases=run_cases,
            allocator_factory=allocator_factory,
        )
        if not isinstance(scope, dict):
            raise ValueError("native rank scope result is invalid")
        results = scope.get("results")
        if not isinstance(results, tuple) or len(results) != len(cases):
            raise ValueError("native rank result inventory is invalid")
        logits_by_case = {}
        state_rows = []
        for case, result in zip(cases, results):
            case_id, token_ids = _prompt_identity(case)
            if (
                getattr(result, "case_id", None) != case_id
                or getattr(result, "rank", None) != rank
                or getattr(result, "token_count", None) != len(token_ids)
            ):
                raise ValueError("native rank result identity mismatch")
            logits = getattr(result, "logits", None)
            validated = _TP4_CONTRACT.validate_rank_logits(
                rank=rank,
                world_size=world_size,
                logits=logits,
            )
            if validated is not None:
                logits_by_case[case_id] = (
                    validated.detach().cpu().contiguous()
                )
            state = getattr(
                result,
                "state_nonzero_after_commit",
                None,
            )
            if (
                not isinstance(state, dict)
                or not state
                or any(value is not True for value in state.values())
                or getattr(result, "release_zeroed", None) is not True
                or getattr(result, "pool_binding_released", None) is not True
            ):
                raise ValueError("native rank state evidence is invalid")
            state_rows.append({
                "case_id": case_id,
                "changed_component_count": len(state),
                "state_nonzero_after_commit": dict(sorted(state.items())),
                "release_zeroed": True,
                "pool_binding_released": True,
            })
        root_logits_present = rank == 0 and set(logits_by_case) == set(case_ids)
        non_root_logits_none = rank != 0 and not logits_by_case
        row = {
            "rank": rank,
            "world_size": world_size,
            "pid": pid,
            "exit_code": 0,
            "gpu_index": gpu_index,
            "gpu_uuid": gpu_uuid,
            "process_group_nonce": process_group_nonce,
            "rendezvous": rendezvous,
            "case_ids": list(case_ids),
            "case_barrier_count": scope.get("case_barrier_count"),
            "final_barrier_completed": scope.get(
                "final_barrier_completed"
            ),
            "process_group_destroyed": scope.get(
                "process_group_destroyed"
            ),
            "candidate_reference_dropped": scope.get(
                "candidate_reference_dropped"
            ),
            "model_reference_dropped": scope.get(
                "model_reference_dropped"
            ),
            "cuda_synchronized": scope.get("cuda_synchronized"),
            "cuda_cache_emptied": scope.get("cuda_cache_emptied"),
            "collective_events": scope.get("collective_events", []),
            "root_logits_present": root_logits_present,
            "non_root_logits_none": non_root_logits_none,
            "global_query_heads": 8,
            "global_kv_heads": 2,
            "local_query_heads": 2,
            "local_kv_heads": 1,
            "kv_head_replicas": 2,
            "source_kv_rank": rank // 2,
            "state_rows": state_rows,
        }
        if rank == 0:
            _worker_atomic_torch_save(logits_path, logits_by_case)
            published.append(logits_path)
        _worker_atomic_write_json(row_path, row)
        published.append(row_path)
        return row
    except BaseException:
        for path in published:
            if path.exists():
                path.unlink()
        raise


def _canonical_json_bytes(value) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_tp4_tensor_map(value, *, case_ids, label: str):
    if not isinstance(value, dict) or tuple(value) != tuple(case_ids):
        raise ValueError(f"{label} tensor map case order is invalid")
    result = {}
    for case_id in case_ids:
        tensor = value.get(case_id)
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.device.type != "cpu"
            or tensor.dtype != torch.float32
            or tensor.ndim != 1
            or tensor.shape[0] != MODEL_VOCAB_SIZE
            or not tensor.is_contiguous()
            or not bool(torch.isfinite(tensor).all())
        ):
            raise ValueError(f"{label} tensor row is invalid")
        result[case_id] = tensor.detach().clone().contiguous()
    return result


def _validate_tp4_source_manifest(value) -> dict[str, object]:
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("source manifest is invalid")
    source_hashes = value.get("source_file_sha256")
    if (
        not isinstance(source_hashes, dict)
        or not source_hashes
        or any(
            not isinstance(name, str)
            or not name
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            for name, digest in source_hashes.items()
        )
    ):
        raise ValueError("source manifest source closure is invalid")
    expected = {
        "model_manifest_sha256": APPROVED_MODEL_MANIFEST_SHA256,
        "config_sha256": APPROVED_CONFIG_SHA256,
        "index_sha256": APPROVED_INDEX_SHA256,
        "shard_name": APPROVED_SHARD_NAME,
        "shard_size": APPROVED_SHARD_SIZE,
        "shard_sha256": APPROVED_SHARD_SHA256,
    }
    if any(value.get(name) != item for name, item in expected.items()):
        raise ValueError("source manifest checkpoint identity mismatch")
    source_tree = hashlib.sha256(
        json.dumps(
            dict(sorted(source_hashes.items())),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if value.get("source_tree_sha256") != source_tree:
        raise ValueError("source manifest tree hash is invalid")
    prerequisites = value.get("prerequisites")
    if (
        not isinstance(prerequisites, dict)
        or prerequisites.get("tp1_real_root_logit_correctness")
        != TP1_PREREQUISITE
    ):
        raise ValueError("source manifest prerequisite identity mismatch")
    result = dict(value)
    result["source_file_sha256"] = dict(sorted(source_hashes.items()))
    return result


def _validate_reference_process(value, *, case_ids):
    if not isinstance(value, dict):
        raise ValueError("reference process row is invalid")
    expected = {
        "worker": "reference",
        "exit_code": 0,
        "case_ids": list(case_ids),
        "vocab_size": MODEL_VOCAB_SIZE,
        "cleanup_complete": True,
        "local_files_only": True,
        "trust_remote_code": False,
        "dtype": "bfloat16",
        "attn_implementation": "eager",
        "use_cache": False,
    }
    if any(value.get(name) != item for name, item in expected.items()):
        raise ValueError("reference process contract is invalid")
    if (
        isinstance(value.get("pid"), bool)
        or not isinstance(value.get("pid"), int)
        or value["pid"] <= 0
    ):
        raise ValueError("reference process PID is invalid")
    if (
        isinstance(value.get("gpu_index"), bool)
        or not isinstance(value.get("gpu_index"), int)
        or value["gpu_index"] < 0
        or not isinstance(value.get("gpu_uuid"), str)
        or not value["gpu_uuid"]
    ):
        raise ValueError("reference process GPU identity is invalid")
    model_manifest = value.get(
        "model_manifest_sha256",
        APPROVED_MODEL_MANIFEST_SHA256,
    )
    if model_manifest != APPROVED_MODEL_MANIFEST_SHA256:
        raise ValueError("reference process checkpoint identity is invalid")
    result = dict(value)
    result["model_manifest_sha256"] = model_manifest
    return result


def finalize_tp4_correctness_artifact(
    *,
    run_dir,
    run_tag: str,
    reference_logits,
    native_rank0_logits,
    reference_process,
    rank_rows,
    source_manifest,
    forbidden_counters,
    replace=os.replace,
) -> tuple[Path, ...]:
    directory = Path(run_dir)
    if directory.exists():
        if any(directory.iterdir()):
            raise ValueError("TP4 correctness run directory is not empty")
    else:
        directory.mkdir(parents=True)
    if (
        not isinstance(run_tag, str)
        or not run_tag
        or any(
            character not in (
                "abcdefghijklmnopqrstuvwxyz"
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                "0123456789-_"
            )
            for character in run_tag
        )
    ):
        raise ValueError("TP4 correctness run tag is invalid")
    cases = _TP4_CONTRACT.prompt_cases()
    case_ids = tuple(case.case_id for case in cases)
    reference = _validate_tp4_tensor_map(
        reference_logits,
        case_ids=case_ids,
        label="reference",
    )
    native = _validate_tp4_tensor_map(
        native_rank0_logits,
        case_ids=case_ids,
        label="native rank zero",
    )
    reference_row = _validate_reference_process(
        reference_process,
        case_ids=case_ids,
    )
    ranks = validate_rank_evidence(rank_rows)
    if reference_row["pid"] in {row["pid"] for row in ranks}:
        raise ValueError(
            "reference and native processes must use unique PIDs"
        )
    if any(tuple(row.get("case_ids", ())) != case_ids for row in ranks):
        raise ValueError("rank evidence case inventory is invalid")
    expected_forbidden = {
        "engine",
        "model_runner",
        "scheduler",
        "sampler",
        "generation",
    }
    if (
        not isinstance(forbidden_counters, dict)
        or set(forbidden_counters) != expected_forbidden
        or any(value != 0 for value in forbidden_counters.values())
    ):
        raise ValueError("forbidden execution counters are invalid")
    source = _validate_tp4_source_manifest(source_manifest)
    comparisons = [
        {
            "case_id": case_id,
            **_TP4_CONTRACT.compare_logits(
                native[case_id],
                reference[case_id],
                tolerance=_TP4_CONTRACT.BF16_DECISION_TOLERANCE,
            ),
        }
        for case_id in case_ids
    ]
    classification = _TP4_CONTRACT.classify_rows(comparisons)
    if classification != "PASS":
        raise ValueError(
            "authoritative TP4 artifact requires PASS classification"
        )
    result = {
        "schema_version": _TP4_CONTRACT.SCHEMA_VERSION,
        "run_tag": run_tag,
        "classification": classification,
        "comparison_policy": "bf16_decision_preserving",
        "tolerance": {
            "atol": _TP4_CONTRACT.BF16_DECISION_TOLERANCE.atol,
            "rtol": _TP4_CONTRACT.BF16_DECISION_TOLERANCE.rtol,
        },
        "prompts": [
            {
                "case_id": case.case_id,
                "token_ids": list(case.token_ids),
                "token_sha256": case.token_sha256,
            }
            for case in cases
        ],
        "reference_process": reference_row,
        "comparisons": comparisons,
        "forbidden_counters": dict(sorted(forbidden_counters.items())),
        "claim_boundary": (
            "TP4 distributed one-shot final-token root-logit correctness "
            "only; no cached decode, Engine, latency, throughput, cache, "
            "memory, compression, or quality claim."
        ),
    }
    final_paths = {
        name: directory / name for name in TP4_ARTIFACT_NAMES
    }
    partial_paths = {
        name: path.with_name(path.name + ".partial")
        for name, path in final_paths.items()
    }
    if any(
        path.exists()
        for path in (*final_paths.values(), *partial_paths.values())
    ):
        raise ValueError("TP4 correctness artifact path already exists")
    published = []
    try:
        partial_paths[TP4_RESULT_NAME].write_bytes(
            _canonical_json_bytes(result)
        )
        torch.save(
            reference,
            partial_paths[TP4_REFERENCE_LOGITS_NAME],
        )
        torch.save(
            native,
            partial_paths[TP4_NATIVE_RANK0_LOGITS_NAME],
        )
        partial_paths[TP4_RANK_EVIDENCE_NAME].write_bytes(
            _canonical_json_bytes(list(ranks))
        )
        source["artifacts"] = {
            name: {
                "size": partial_paths[name].stat().st_size,
                "sha256": _sha256_file(partial_paths[name]),
            }
            for name in TP4_ARTIFACT_NAMES[:-1]
        }
        partial_paths[TP4_SOURCE_MANIFEST_NAME].write_bytes(
            _canonical_json_bytes(source)
        )
        for name in TP4_ARTIFACT_NAMES:
            replace(partial_paths[name], final_paths[name])
            published.append(final_paths[name])
    except BaseException:
        for path in partial_paths.values():
            if path.exists():
                path.unlink()
        for path in published:
            if path.exists():
                path.unlink()
        raise
    if {path.name for path in directory.iterdir()} != set(
        TP4_ARTIFACT_NAMES
    ):
        raise RuntimeError("TP4 correctness artifact inventory is invalid")
    return tuple(final_paths[name] for name in TP4_ARTIFACT_NAMES)


class _DeferredSubprocess:
    def __init__(
        self,
        *,
        command,
        work_dir,
        environment,
        popen,
    ):
        self._command = tuple(command)
        self._work_dir = Path(work_dir)
        self._environment = dict(environment)
        self._popen = popen
        self._process = None
        self._stdout = None
        self._stderr = None

    @property
    def pid(self):
        return None if self._process is None else self._process.pid

    @property
    def exitcode(self):
        return (
            None if self._process is None else self._process.returncode
        )

    def start(self):
        if self._process is not None:
            raise RuntimeError("native rank subprocess already started")
        rank = self._environment.get("TINYVLLM_GATE_LOCAL_RANK", "unknown")
        self._stdout = (
            self._work_dir / f"rank-{rank}.stdout.log"
        ).open("wb")
        self._stderr = (
            self._work_dir / f"rank-{rank}.stderr.log"
        ).open("wb")
        self._process = self._popen(
            self._command,
            cwd=self._work_dir,
            env=self._environment,
            stdout=self._stdout,
            stderr=self._stderr,
        )

    def join(self, timeout):
        if self._process is None:
            raise RuntimeError("native rank subprocess was not started")
        try:
            self._process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return
        self._close_logs()

    def is_alive(self):
        return (
            self._process is not None
            and self._process.poll() is None
        )

    def terminate(self):
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()

    def kill(self):
        if self._process is not None and self._process.poll() is None:
            self._process.kill()

    def _close_logs(self):
        for handle in (self._stdout, self._stderr):
            if handle is not None and not handle.closed:
                handle.close()


def make_native_rank_subprocess(
    *,
    rank: int,
    world_size: int,
    gpu_index: int,
    gpu_uuid: str,
    rendezvous: str,
    process_group_nonce: str,
    environment,
    script_path,
    python_executable,
    work_dir,
    rank_output,
    logits_output,
    popen=subprocess.Popen,
):
    rank = _tp4_rank(rank)
    if world_size != 4:
        raise ValueError("world_size must equal 4")
    command = [
        os.fspath(python_executable),
        os.fspath(script_path),
        "internal-native-rank",
        "--rank-output",
        os.fspath(rank_output),
    ]
    if rank == 0:
        if logits_output is None:
            raise ValueError("rank zero logits output is required")
        command.extend([
            "--logits-output",
            os.fspath(logits_output),
        ])
    elif logits_output is not None:
        raise ValueError("non-root logits output must be None")
    process_environment = dict(environment)
    process_environment.update({
        "TINYVLLM_GATE_LOCAL_RANK": str(rank),
        "TINYVLLM_GATE_PHYSICAL_GPU_INDEX": str(gpu_index),
        "TINYVLLM_GATE_GPU_UUID": gpu_uuid,
        "TINYVLLM_GATE_PROCESS_GROUP_NONCE": process_group_nonce,
        "TINYVLLM_GATE_RENDEZVOUS": rendezvous,
    })
    return _DeferredSubprocess(
        command=command,
        work_dir=work_dir,
        environment=process_environment,
        popen=popen,
    )


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _query_tp4_gpu_resources(
    *,
    command_runner=subprocess.run,
) -> tuple[dict[str, object], ...]:
    completed = command_runner(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    if getattr(completed, "returncode", None) != 0:
        raise ValueError("nvidia-smi GPU query failed")
    rows = []
    for line in str(getattr(completed, "stdout", "")).splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 5:
            raise ValueError("nvidia-smi GPU query output is invalid")
        try:
            gpu_index = int(fields[0])
            total_mib = int(fields[3])
            free_mib = int(fields[4])
        except ValueError as error:
            raise ValueError(
                "nvidia-smi GPU query output is invalid"
            ) from error
        rows.append({
            "gpu_index": gpu_index,
            "gpu_uuid": fields[1],
            "gpu_name": fields[2],
            "total_bytes": total_mib * 1024**2,
            "free_bytes": free_mib * 1024**2,
            "compute_processes": [],
        })
    processes = command_runner(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    if getattr(processes, "returncode", None) != 0:
        raise ValueError("nvidia-smi compute process query failed")
    by_uuid = {row["gpu_uuid"]: row for row in rows}
    for line in str(getattr(processes, "stdout", "")).splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            raise ValueError(
                "nvidia-smi compute process output is invalid"
            )
        try:
            pid = int(fields[1])
            used_mib = int(fields[3])
        except ValueError as error:
            raise ValueError(
                "nvidia-smi compute process output is invalid"
            ) from error
        row = by_uuid.get(fields[0])
        if row is None or pid <= 0 or used_mib < 0 or not fields[2]:
            raise ValueError(
                "nvidia-smi compute process output is invalid"
            )
        row["compute_processes"].append({
            "pid": pid,
            "process_name": fields[2],
            "used_bytes": used_mib * 1024**2,
        })
    return tuple(rows)


def _read_tp4_source_manifest(path: Path, *, source_root: Path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("source manifest input is invalid") from error
    source = _validate_tp4_source_manifest(value)
    required = {
        "tools/qwen35_tp1_real_root_logit_correctness_contract.py",
        "tools/qwen35_tp1_real_root_logit_correctness_preflight.py",
        "tools/qwen35_tp4_real_root_logit_correctness_contract.py",
        "tools/qwen35_tp4_real_root_logit_correctness_preflight.py",
        "tools/verify_qwen35_tp4_real_root_logit_correctness_gate.py",
    }
    hashes = source["source_file_sha256"]
    if not required.issubset(hashes):
        if source_root == Path(__file__).resolve().parents[1]:
            raise ValueError("source manifest omits required gate source")
    for relative_name, expected_sha256 in hashes.items():
        relative = Path(relative_name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("source manifest path is invalid")
        source_file = source_root / relative
        if not source_file.is_file():
            if source_root == Path(__file__).resolve().parents[1]:
                raise ValueError("source manifest file is missing")
            continue
        if _sha256_file(source_file) != expected_sha256:
            if source_root == Path(__file__).resolve().parents[1]:
                raise ValueError("source manifest hash mismatch")
    return source


def _run_reference_subprocess(
    *,
    command,
    work_dir,
    environment,
    command_runner,
):
    completed = command_runner(
        command,
        cwd=work_dir,
        env=dict(environment),
        check=False,
        text=True,
        capture_output=True,
        timeout=1800,
    )
    if getattr(completed, "returncode", None) != 0:
        detail = str(getattr(completed, "stderr", "")).strip()[-4000:]
        raise ValueError(f"reference worker failed: {detail}")


def _load_json(path: Path, *, label: str):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} JSON output is invalid") from error


def _load_tensor_map(path: Path, *, label: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError(f"{label} tensor output is invalid") from error


def execute_source_bound_run(
    *,
    run_dir,
    run_tag: str,
    source_manifest_path,
    source_root=None,
    query_gpus=_query_tp4_gpu_resources,
    command_runner=subprocess.run,
    process_factory_builder=None,
    pid_alive=_pid_alive,
) -> dict[str, object]:
    directory = Path(run_dir)
    if directory.exists():
        raise ValueError("TP4 correctness run directory already exists")
    root = (
        Path(__file__).resolve().parents[1]
        if source_root is None
        else Path(source_root)
    )
    source = _read_tp4_source_manifest(
        Path(source_manifest_path),
        source_root=root,
    )
    selected = select_tp4_gpu_resources(query_gpus())
    work_dir = directory.parent / f".{run_tag}.work"
    if work_dir.exists():
        raise ValueError("TP4 correctness work directory already exists")
    work_dir.mkdir(parents=True)
    script = Path(__file__).resolve()
    base_environment = dict(os.environ)
    base_environment.update({
        "PYTHONPATH": os.fspath(root),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    })
    reference_tensor = work_dir / "reference_logits.pt.partial"
    reference_process_path = (
        work_dir / "reference_process.json.partial"
    )
    reference_environment = dict(base_environment)
    reference_environment.update({
        "CUDA_VISIBLE_DEVICES": str(selected[0]["gpu_index"]),
        "TINYVLLM_GATE_PHYSICAL_GPU_INDEX": str(
            selected[0]["gpu_index"]
        ),
        "TINYVLLM_GATE_GPU_UUID": str(selected[0]["gpu_uuid"]),
    })
    reference_command = (
        sys.executable,
        os.fspath(script),
        "internal-reference",
        "--tensor-output",
        os.fspath(reference_tensor),
        "--process-output",
        os.fspath(reference_process_path),
    )
    dist_port, master_port = fresh_port_pair()
    rendezvous = f"tcp://127.0.0.1:{dist_port}"
    nonce = hashlib.sha256(
        f"{run_tag}:{dist_port}:{master_port}".encode("utf-8")
    ).hexdigest()
    try:
        _run_reference_subprocess(
            command=reference_command,
            work_dir=work_dir,
            environment=reference_environment,
            command_runner=command_runner,
        )
        reference_process = _load_json(
            reference_process_path,
            label="reference",
        )
        reference_pid = reference_process.get("pid")
        if (
            isinstance(reference_pid, bool)
            or not isinstance(reference_pid, int)
            or reference_pid <= 0
            or pid_alive(reference_pid)
        ):
            raise ValueError(
                "reference worker PID did not disappear before native startup"
            )
        refreshed = select_tp4_gpu_resources(query_gpus())
        if tuple(
            (row["gpu_index"], row["gpu_uuid"]) for row in refreshed
        ) != tuple(
            (row["gpu_index"], row["gpu_uuid"]) for row in selected
        ):
            raise ValueError("selected GPU identity changed before native")
        if process_factory_builder is None:
            def process_factory(**kwargs):
                rank = kwargs["rank"]
                return make_native_rank_subprocess(
                    **kwargs,
                    script_path=script,
                    python_executable=sys.executable,
                    work_dir=work_dir,
                    rank_output=work_dir / f"rank-{rank}.json.partial",
                    logits_output=(
                        work_dir / "native_rank0_logits.pt.partial"
                        if rank == 0
                        else None
                    ),
                )
        else:
            process_factory = process_factory_builder(
                work_dir=work_dir,
                script_path=script,
                python_executable=sys.executable,
            )
        launched_rows = launch_native_rank_group(
            selected_gpus=selected,
            rendezvous=rendezvous,
            process_group_nonce=nonce,
            tinyvllm_dist_port=dist_port,
            master_port=master_port,
            process_factory=process_factory,
            timeout_seconds=1800,
            pid_alive=pid_alive,
            base_environment=base_environment,
        )
        rank_rows = tuple(
            _load_json(
                work_dir / f"rank-{rank}.json.partial",
                label=f"rank {rank}",
            )
            for rank in range(4)
        )
        rank_rows = bind_launched_rank_evidence(
            launched_rows,
            rank_rows,
        )
        paths = finalize_tp4_correctness_artifact(
            run_dir=directory,
            run_tag=run_tag,
            reference_logits=_load_tensor_map(
                reference_tensor,
                label="reference",
            ),
            native_rank0_logits=_load_tensor_map(
                work_dir / "native_rank0_logits.pt.partial",
                label="native rank zero",
            ),
            reference_process=reference_process,
            rank_rows=rank_rows,
            source_manifest=source,
            forbidden_counters={
                "engine": 0,
                "model_runner": 0,
                "scheduler": 0,
                "sampler": 0,
                "generation": 0,
            },
        )
        return {
            "classification": "PASS",
            "paths": [os.fspath(path) for path in paths],
            "gpus": list(selected),
        }
    finally:
        if (
            directory.is_dir()
            and {path.name for path in directory.iterdir()}
            == set(TP4_ARTIFACT_NAMES)
        ):
            shutil.rmtree(work_dir, ignore_errors=False)


def validate_tp4_correctness_artifact(run_dir) -> dict:
    path = (
        Path(__file__).resolve().parent
        / "verify_qwen35_tp4_real_root_logit_correctness_gate.py"
    )
    spec = importlib.util.spec_from_file_location(
        "qwen35_tp4_independent_verifier_for_cli",
        path,
    )
    if spec is None or spec.loader is None:
        raise ValueError("independent verifier is invalid")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.verify_run(run_dir)


def _load_frozen_prompt_cases():
    return _TP4_CONTRACT.prompt_cases()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--run-dir", required=True)
    run.add_argument("--run-tag", required=True)
    run.add_argument("--source-manifest", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("run_dir")
    reference = subparsers.add_parser("internal-reference")
    reference.add_argument("--tensor-output", required=True)
    reference.add_argument("--process-output", required=True)
    native = subparsers.add_parser("internal-native-rank")
    native.add_argument("--rank-output", required=True)
    native.add_argument("--logits-output")
    return parser


def main(
    argv=None,
    *,
    execute_reference=None,
    execute_native_rank=execute_native_rank_worker,
    execute_run=None,
    execute_validate=validate_tp4_correctness_artifact,
    prompt_case_loader=_load_frozen_prompt_cases,
    environment=None,
) -> int:
    arguments = _build_parser().parse_args(argv)
    environment = os.environ if environment is None else environment
    if not isinstance(environment, Mapping):
        raise ValueError("environment must be a mapping")
    if not callable(prompt_case_loader):
        raise ValueError("prompt_case_loader must be callable")
    if arguments.mode == "run":
        if not callable(execute_run):
            execute_run = execute_source_bound_run
        execute_run(
            run_dir=Path(arguments.run_dir),
            run_tag=arguments.run_tag,
            source_manifest_path=Path(arguments.source_manifest),
        )
        return 0
    if arguments.mode == "validate":
        if not callable(execute_validate):
            raise ValueError("execute_validate must be callable")
        execute_validate(Path(arguments.run_dir))
        return 0
    if arguments.mode == "internal-reference":
        if not callable(execute_reference):
            execute_reference = _TP1_PREFLIGHT.execute_reference_worker
        try:
            gpu_index = int(
                environment["TINYVLLM_GATE_PHYSICAL_GPU_INDEX"]
            )
            gpu_uuid = environment["TINYVLLM_GATE_GPU_UUID"]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "reference worker GPU identity is missing"
            ) from error
        execute_reference(
            model_dir=Path(APPROVED_MODEL_DIR),
            model_manifest_sha256=APPROVED_MODEL_MANIFEST_SHA256,
            tensor_output=Path(arguments.tensor_output),
            process_output=Path(arguments.process_output),
            prompt_cases=prompt_case_loader(),
            expected_vocab_size=MODEL_VOCAB_SIZE,
            gpu_index=gpu_index,
            gpu_uuid=gpu_uuid,
        )
        return 0
    if not callable(execute_native_rank):
        raise ValueError("execute_native_rank must be callable")
    try:
        rank = int(environment["TINYVLLM_GATE_LOCAL_RANK"])
        gpu_index = int(
            environment["TINYVLLM_GATE_PHYSICAL_GPU_INDEX"]
        )
        gpu_uuid = environment["TINYVLLM_GATE_GPU_UUID"]
        nonce = environment["TINYVLLM_GATE_PROCESS_GROUP_NONCE"]
        rendezvous = environment["TINYVLLM_GATE_RENDEZVOUS"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("native rank environment is incomplete") from error
    execute_native_rank(
        rank=rank,
        world_size=4,
        rendezvous=rendezvous,
        process_group_nonce=nonce,
        prompt_cases=prompt_case_loader(),
        gpu_index=gpu_index,
        gpu_uuid=gpu_uuid,
        rank_output=Path(arguments.rank_output),
        logits_output=(
            None
            if arguments.logits_output is None
            else Path(arguments.logits_output)
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
