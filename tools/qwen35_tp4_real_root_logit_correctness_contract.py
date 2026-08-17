from __future__ import annotations

from collections.abc import Iterable, Mapping
import importlib.util
from pathlib import Path
import sys

import torch


def _load_tp1_contract():
    path = (
        Path(__file__).resolve().parent
        / "qwen35_tp1_real_root_logit_correctness_contract.py"
    )
    module_name = "qwen35_tp1_real_root_logit_correctness_contract"
    module = sys.modules.get(module_name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_TP1 = _load_tp1_contract()

SCHEMA_VERSION = "qwen35.tp4-real-root-logit-correctness.v1"
WORLD_SIZE = 4
MODEL_VOCAB_SIZE = 248320
GLOBAL_QUERY_HEADS = 8
GLOBAL_KV_HEADS = 2
LOCAL_QUERY_HEADS = 2
LOCAL_KV_HEADS = 1
KV_HEAD_REPLICAS = 2
MIN_GPU_FREE_BYTES = 24 * 1024**3

BF16_DECISION_TOLERANCE = _TP1.BF16_DECISION_TOLERANCE
prompt_cases = _TP1.prompt_cases
compare_logits = _TP1.compare_logits
classify_rows = _TP1.classify_rows


def _integer(value, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _rank(rank, world_size) -> tuple[int, int]:
    rank_value = _integer(rank, name="rank")
    world_size_value = _integer(world_size, name="world size")
    if world_size_value != WORLD_SIZE:
        raise ValueError(f"world size must equal {WORLD_SIZE}")
    if rank_value < 0 or rank_value >= world_size_value:
        raise ValueError("rank must be in [0, world size)")
    return rank_value, world_size_value


def validate_rank_logits(*, rank: int, world_size: int, logits):
    rank_value, _ = _rank(rank, world_size)
    if rank_value != 0:
        if logits is not None:
            raise ValueError("non-root rank logits must be None")
        return None
    if not isinstance(logits, torch.Tensor):
        raise ValueError("rank zero logits must be a tensor")
    if logits.ndim != 1:
        raise ValueError("rank zero logits must be rank one")
    if logits.shape[0] != MODEL_VOCAB_SIZE:
        raise ValueError(
            f"rank zero vocabulary width must equal {MODEL_VOCAB_SIZE}"
        )
    if logits.dtype != torch.float32:
        raise ValueError("rank zero logits must use float32")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("rank zero logits must be finite")
    return logits


def validate_rank_topology(row: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(row, Mapping):
        raise ValueError("rank topology must be a mapping")
    result = dict(row)
    rank_value, _ = _rank(
        result.get("rank"),
        result.get("world_size"),
    )
    expected = {
        "global_query_heads": GLOBAL_QUERY_HEADS,
        "global_kv_heads": GLOBAL_KV_HEADS,
        "local_query_heads": LOCAL_QUERY_HEADS,
        "local_kv_heads": LOCAL_KV_HEADS,
        "kv_head_replicas": KV_HEAD_REPLICAS,
        "source_kv_rank": rank_value // KV_HEAD_REPLICAS,
    }
    for name, expected_value in expected.items():
        value = _integer(result.get(name), name=name)
        if value != expected_value:
            raise ValueError(
                f"{name} must equal {expected_value}"
            )
        result[name] = value
    result["rank"] = rank_value
    result["world_size"] = WORLD_SIZE
    return result


def validate_gpu_assignments(
    rows: Iterable[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    values = tuple(rows)
    if len(values) != WORLD_SIZE:
        raise ValueError("GPU assignments must contain exactly four rows")
    normalized = []
    for row in values:
        if not isinstance(row, Mapping):
            raise ValueError("GPU assignment must be a mapping")
        value = dict(row)
        rank_value, _ = _rank(
            value.get("rank"),
            value.get("world_size"),
        )
        gpu_index = _integer(value.get("gpu_index"), name="gpu_index")
        if gpu_index < 0:
            raise ValueError("gpu_index must be non-negative")
        gpu_uuid = value.get("gpu_uuid")
        if not isinstance(gpu_uuid, str) or not gpu_uuid.startswith("GPU-"):
            raise ValueError("gpu_uuid must be a GPU UUID")
        value["rank"] = rank_value
        value["world_size"] = WORLD_SIZE
        value["gpu_index"] = gpu_index
        value["gpu_uuid"] = gpu_uuid
        normalized.append(value)
    normalized.sort(key=lambda row: row["rank"])
    if tuple(row["rank"] for row in normalized) != tuple(range(WORLD_SIZE)):
        raise ValueError("GPU assignment ranks must be exactly 0..3")
    if len({row["gpu_index"] for row in normalized}) != WORLD_SIZE:
        raise ValueError("GPU indices must be unique")
    if len({row["gpu_uuid"] for row in normalized}) != WORLD_SIZE:
        raise ValueError("GPU UUIDs must be unique")
    return tuple(normalized)
