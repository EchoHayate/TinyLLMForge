from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_frozen_gate():
    path = (
        Path(__file__).resolve().parent
        / "qwen35_generic_speculative_tp4_gate.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_qwen35_generic_speculative_tp4_frozen_gate",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_frozen_gate = _load_frozen_gate()

_frozen_gate.SCHEMA_VERSION = (
    "qwen35.generic-speculative-tp4-16k-"
    "transactional-correctness.v1"
)
_frozen_gate.CLASSIFICATION = (
    "SECOND_MODEL_TP4_16K_ESTABLISHED"
)
_frozen_gate.CLAIM_SCOPE = "second_model_tp4_16k_only"
_frozen_gate.LIMITATIONS = (
    "phase1_not_promotable",
    "context_32k_not_established",
    "performance_not_established",
    "learned_drafter_not_established",
    "kv_quantization_not_established",
)
_frozen_gate.CONTEXT_TOKENS = 16384
_frozen_gate.MAX_MODEL_LEN = 33024
_frozen_gate.MAX_NUM_BATCHED_TOKENS = 132096
_frozen_gate.MAX_NUM_PREFILL_TOKENS_PER_STEP = 1024
_frozen_gate.KV_OFFLOAD_GPU_BLOCKS = 68
_frozen_gate.KV_OFFLOAD_LOGICAL_BLOCKS = 640
_frozen_gate.KV_OFFLOAD_BLOCKWISE_BLOCKS = 8
_frozen_gate.DEFAULT_SOURCE_FILES = tuple(
    dict.fromkeys(
        _frozen_gate.DEFAULT_SOURCE_FILES
        + (
            "tools/qwen35_generic_speculative_tp4_16k_gate.py",
            "tools/qwen35_generic_speculative_tp4_16k_worker.py",
            "tools/verify_qwen35_generic_speculative_tp4_16k_gate.py",
        )
    )
)

for _name, _value in vars(_frozen_gate).items():
    if not _name.startswith("__"):
        globals()[_name] = _value


_frozen_validate_result = _frozen_gate.validate_result


def validate_result(value: object) -> dict:
    normalized = _frozen_validate_result(value)
    movement = normalized["cells"]["ngram:b4"][
        "kv_rank_deltas"
    ]
    if sum(row["h2d_copies"] for row in movement) <= 0:
        raise ValueError(
            "16K batch-4 candidate requires real H2D copies"
        )
    if sum(row["h2d_bytes"] for row in movement) <= 0:
        raise ValueError(
            "16K batch-4 candidate requires real H2D bytes"
        )
    return normalized


_frozen_gate.validate_result = validate_result


if __name__ == "__main__":
    raise SystemExit(main())
