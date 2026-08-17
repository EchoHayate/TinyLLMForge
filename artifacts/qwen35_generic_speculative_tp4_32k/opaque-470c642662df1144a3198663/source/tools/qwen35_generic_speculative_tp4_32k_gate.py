from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_frozen_gate():
    path = (
        Path(__file__).resolve().parent
        / "qwen35_generic_speculative_tp4_gate.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_qwen35_generic_speculative_tp4_32k_frozen_gate",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_frozen_gate = _load_frozen_gate()

_frozen_gate.SCHEMA_VERSION = (
    "qwen35.generic-speculative-tp4-32k-"
    "transactional-correctness.v1"
)
_frozen_gate.CLASSIFICATION = (
    "SECOND_MODEL_TP4_32K_ESTABLISHED"
)
_frozen_gate.CLAIM_SCOPE = "second_model_tp4_32k_only"
_frozen_gate.LIMITATIONS = (
    "phase1_not_promotable",
    "performance_not_established",
    "learned_drafter_not_established",
    "kv_quantization_not_established",
)
_frozen_gate.CONTEXT_TOKENS = 32768
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
            "tools/qwen35_generic_speculative_tp4_32k_gate.py",
            "tools/qwen35_generic_speculative_tp4_32k_worker.py",
            "tools/verify_qwen35_generic_speculative_tp4_32k_gate.py",
        )
    )
)

for _name, _value in vars(_frozen_gate).items():
    if not _name.startswith("__"):
        globals()[_name] = _value


_frozen_validate_result = _frozen_gate.validate_result


def validate_result(value: object) -> dict:
    normalized = _frozen_validate_result(value)
    for batch_size in BATCH_SIZES:
        movement = normalized["cells"][
            f"ngram:b{batch_size}"
        ]["kv_rank_deltas"]
        if sum(row["h2d_copies"] for row in movement) <= 0:
            raise ValueError(
                f"32K batch-{batch_size} candidate "
                "requires real H2D copies"
            )
        if sum(row["h2d_bytes"] for row in movement) <= 0:
            raise ValueError(
                f"32K batch-{batch_size} candidate "
                "requires real H2D bytes"
            )
    return normalized


_frozen_gate.validate_result = validate_result


_run_campaign_impl = _frozen_gate.run_campaign


def _load_32k_verifier():
    path = (
        Path(__file__).resolve().parent
        / "verify_qwen35_generic_speculative_tp4_32k_gate.py"
    )
    spec = importlib.util.spec_from_file_location(
        "verify_qwen35_generic_speculative_tp4_32k_gate",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.verify_run


def run_campaign(
    *,
    model_path: str,
    gpu_indices: tuple[int, ...],
    output_dir: Path,
    dist_port_base: int,
    master_port_base: int,
    repo_root: Path | None = None,
    worker_script: Path | None = None,
    python_executable: str = sys.executable,
    source_files: tuple[str, ...] | None = None,
    verifier=None,
) -> dict:
    if worker_script is None:
        worker_script = (
            Path(__file__).resolve().parent
            / "qwen35_generic_speculative_tp4_32k_worker.py"
        )
    if source_files is None:
        source_files = DEFAULT_SOURCE_FILES
    if verifier is None:
        verifier = _load_32k_verifier()
    return _run_campaign_impl(
        model_path=model_path,
        gpu_indices=gpu_indices,
        output_dir=output_dir,
        dist_port_base=dist_port_base,
        master_port_base=master_port_base,
        repo_root=repo_root,
        worker_script=worker_script,
        python_executable=python_executable,
        source_files=source_files,
        verifier=verifier,
    )


_frozen_gate.run_campaign = run_campaign


if __name__ == "__main__":
    raise SystemExit(main())
