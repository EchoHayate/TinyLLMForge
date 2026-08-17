from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_TOOLS = Path(__file__).resolve().parent
_frozen_gate = _load_module(
    "_qwen35_native_mtp_tp4_32k_frozen_gate",
    _TOOLS
    / "qwen35_native_mtp_tp4_16k_target_kv_offload_gate.py",
)

_frozen_gate.SCHEMA_VERSION = (
    "qwen35.native-mtp-tp4-32k-target-kv-offload.v1"
)
_frozen_gate.CLASSIFICATION = (
    "QWEN35_NATIVE_MTP_TP4_32K_TARGET_KV_OFFLOAD_ESTABLISHED"
)
_frozen_gate.PROMPT_TOKENS = 32768
_frozen_gate.REQUIRED_LIMITATIONS = (
    "phase1_not_promotable",
    "proposal_kv_offload_not_established",
    "tp1_32k_not_established",
    "performance_not_established",
    "kv_quantization_not_established",
    "second_learned_structure_not_established",
)
_frozen_gate._rank_gate.PROMPT_TOKENS = (
    _frozen_gate.PROMPT_TOKENS
)
_frozen_gate._WORKER = (
    _TOOLS
    / "qwen35_native_mtp_tp4_32k_target_kv_offload_worker.py"
)
_frozen_gate._VERIFIER = (
    _TOOLS
    / "verify_qwen35_native_mtp_tp4_32k_target_kv_offload_gate.py"
)
_frozen_gate.DEFAULT_SOURCE_FILES = tuple(sorted(
    set(_frozen_gate.DEFAULT_SOURCE_FILES)
    | {
        (
            "tools/qwen35_native_mtp_tp4_32k_"
            "target_kv_offload_gate.py"
        ),
        (
            "tools/qwen35_native_mtp_tp4_32k_"
            "target_kv_offload_worker.py"
        ),
        (
            "tools/verify_qwen35_native_mtp_tp4_32k_"
            "target_kv_offload_gate.py"
        ),
    }
))

for _name, _value in vars(_frozen_gate).items():
    if (
        not _name.startswith("__")
        and _name != "_frozen_gate"
    ):
        globals()[_name] = _value


_frozen_validate_result = _frozen_gate.validate_result


def validate_result(value: object) -> dict:
    required = (
        ("h2d_copies", "real H2D copies"),
        ("h2d_bytes", "real H2D bytes"),
        ("d2h_copies", "real D2H copies"),
        ("d2h_bytes", "real D2H bytes"),
    )
    try:
        normalized = _frozen_validate_result(value)
    except ValueError as exc:
        inherited_messages = {
            (
                "native batch-4 requires real target-KV "
                f"{label.removeprefix('real ')}"
            ): label
            for _, label in required
        }
        label = inherited_messages.get(str(exc))
        if label is None:
            raise
        raise ValueError(
            "32K batch-4 native cell "
            f"requires {label}"
        ) from exc
    for batch_size in BATCH_SIZES:
        movement = normalized["cells"][
            f"native_mtp:b{batch_size}"
        ]["kv_rank_deltas"]
        for field, label in required:
            if sum(row[field] for row in movement) <= 0:
                raise ValueError(
                    f"32K batch-{batch_size} native cell "
                    f"requires {label}"
                )
    return normalized


_frozen_gate.validate_result = validate_result


if __name__ == "__main__":
    sys.exit(main())
