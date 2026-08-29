from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping, Optional, Sequence

from tools.cross_engine_k8_contract import HARD_STOP_BYTES
from tools.cross_engine_k8_workload import (
    REQUIRED_ARMS,
    aggregate_case_rows,
    classify_comparison,
    expected_case_identities,
)


VERIFICATION_SCHEMA_VERSION = "cross-engine-k8.verification.v1"
_PRODUCER_FILES = (
    "controller_manifest.json",
    "environment_manifest.json",
    "workload_manifest.json",
    "case_rows.jsonl",
    "correctness_rows.jsonl",
    "comparison.json",
    "summary.json",
    "gate.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"{path.name}:{line_number} is not an object"
                )
            rows.append(value)
    return rows


def _read_manifest(path: Path) -> dict[str, str]:
    entries = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        parts = line.split("  ", 1)
        if (
            len(parts) != 2
            or len(parts[0]) != 64
            or parts[1] in entries
        ):
            raise ValueError(f"invalid manifest line {line_number}")
        entries[parts[1]] = parts[0]
    return entries


def _ratio(numerator, denominator) -> float:
    numerator = float(numerator)
    denominator = float(denominator)
    if denominator <= 0:
        raise ValueError("comparison denominator must be positive")
    return numerator / denominator


def _comparison_from_aggregates(
    aggregates: Mapping,
    *,
    strongest_vllm_arm: str,
    evidence: Mapping,
) -> dict:
    tiny = aggregates["tinyllmforge_exact_k8"]
    vllm = aggregates[strongest_vllm_arm]
    aggregate = {
        "median_tpot_ratio": _ratio(
            tiny["aggregate"]["median_tpot_ns"],
            vllm["aggregate"]["median_tpot_ns"],
        ),
        "throughput_ratio": _ratio(
            tiny["aggregate"]["output_tokens_per_second"],
            vllm["aggregate"]["output_tokens_per_second"],
        ),
        "ttft_ratio": _ratio(
            tiny["aggregate"]["ttft_ns"],
            vllm["aggregate"]["ttft_ns"],
        ),
        "e2e_ratio": _ratio(
            tiny["aggregate"]["e2e_ns"],
            vllm["aggregate"]["e2e_ns"],
        ),
        "p95_tpot_ratio": _ratio(
            tiny["aggregate"]["p95_tpot_ns"],
            vllm["aggregate"]["p95_tpot_ns"],
        ),
        "p99_tpot_ratio": _ratio(
            tiny["aggregate"]["p99_tpot_ns"],
            vllm["aggregate"]["p99_tpot_ns"],
        ),
        "peak_gpu_memory_ratio": _ratio(
            tiny["aggregate"]["peak_gpu_memory_bytes"],
            vllm["aggregate"]["peak_gpu_memory_bytes"],
        ),
        "peak_rss_ratio": _ratio(
            tiny["aggregate"]["peak_rss_bytes"],
            vllm["aggregate"]["peak_rss_bytes"],
        ),
    }
    contexts = {
        context: {
            "median_tpot_ratio": _ratio(
                values["median_tpot_ns"],
                vllm["contexts"][context]["median_tpot_ns"],
            )
        }
        for context, values in tiny["contexts"].items()
    }
    return {
        **dict(evidence),
        "aggregate": aggregate,
        "contexts": contexts,
    }


def _correctness_valid(rows: Sequence[Mapping], eligible_arms) -> bool:
    by_context = {}
    seen = set()
    for row in rows:
        identity = (
            row.get("repetition"),
            row.get("context"),
            row.get("arm"),
        )
        if identity in seen:
            return False
        seen.add(identity)
        if row.get("matches_reference") is not True:
            return False
        tokens = row.get("token_ids")
        if not isinstance(tokens, list) or len(tokens) != 128:
            return False
        context = row.get("context")
        reference = by_context.setdefault(context, tokens)
        if tokens != reference:
            return False
    return bool(rows) and {
        row.get("arm") for row in rows
    } == set(eligible_arms)


def verify_bundle(
    bundle_root: Path,
    *,
    expected_source: str,
) -> dict:
    root = Path(bundle_root)
    reasons = []
    manifest_path = root / "manifest.sha256"
    try:
        manifest = _read_manifest(manifest_path)
    except (OSError, ValueError):
        manifest = {}
        reasons.append("MANIFEST_INVALID")
    if set(manifest) != set(_PRODUCER_FILES):
        reasons.append("MANIFEST_FILE_SET_MISMATCH")
    for name in _PRODUCER_FILES:
        path = root / name
        if not path.is_file():
            reasons.append(f"MISSING_FILE:{name}")
            continue
        if manifest.get(name) != _sha256(path):
            reasons.append("MANIFEST_DIGEST_MISMATCH")
            break
    try:
        controller = _read_json(root / "controller_manifest.json")
        environment = _read_json(root / "environment_manifest.json")
        workload = _read_json(root / "workload_manifest.json")
        producer_gate = _read_json(root / "gate.json")
        case_rows = _read_jsonl(root / "case_rows.jsonl")
        correctness_rows = _read_jsonl(root / "correctness_rows.jsonl")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return {
            "schema_version": VERIFICATION_SCHEMA_VERSION,
            "valid": False,
            "reasons": reasons + [f"PARSE_ERROR:{type(error).__name__}"],
            "recomputed_classification": "INCOMPLETE",
            "gate_reasons": ["parse_failure"],
            "producer_agrees": False,
        }
    if (
        controller.get("source_revision") != expected_source
        or environment.get("source_revision") != expected_source
    ):
        reasons.append("SOURCE_REVISION_MISMATCH")
    if (
        environment.get("model_inventory_sha256")
        != workload.get("model_inventory_sha256")
    ):
        reasons.append("MODEL_INVENTORY_MISMATCH")
    eligible_arms = tuple(controller.get("eligible_arms", ()))
    if set(REQUIRED_ARMS) - set(eligible_arms):
        reasons.append("REQUIRED_ARM_MISSING")
    expected = set()
    try:
        expected = set(expected_case_identities(workload, eligible_arms))
    except ValueError:
        reasons.append("WORKLOAD_INVALID")
    actual = {
        (
            row.get("repetition"),
            row.get("context"),
            row.get("arm"),
        )
        for row in case_rows
    }
    complete = (
        bool(expected)
        and actual == expected
        and len(case_rows) == len(expected)
    )
    if not complete:
        reasons.append("CASE_MATRIX_INCOMPLETE")
    correctness_valid = _correctness_valid(
        correctness_rows,
        eligible_arms,
    )
    if not correctness_valid:
        reasons.append("CORRECTNESS_INVALID")
    try:
        aggregates = aggregate_case_rows(case_rows)
        vllm_arms = [
            arm
            for arm in eligible_arms
            if arm.startswith("vllm_") and arm in aggregates
        ]
        strongest_vllm_arm = min(
            vllm_arms,
            key=lambda arm: aggregates[arm]["aggregate"][
                "median_tpot_ns"
            ],
        )
        comparison = _comparison_from_aggregates(
            aggregates,
            strongest_vllm_arm=strongest_vllm_arm,
            evidence={
                "complete": complete,
                "correctness_valid": correctness_valid,
                "storage_valid": (
                    controller.get("storage_valid") is True
                    and controller.get("remote_allocated_bytes", 0)
                    < min(
                        controller.get(
                            "remote_hard_limit_bytes",
                            HARD_STOP_BYTES,
                        ),
                        HARD_STOP_BYTES,
                    )
                ),
                "terminal_receipts_valid": (
                    controller.get("terminal_receipts_valid") is True
                ),
                "verifiers_agree": True,
            },
        )
        gate = classify_comparison(comparison)
    except (KeyError, TypeError, ValueError):
        strongest_vllm_arm = None
        comparison = {}
        gate = {
            "classification": "INCOMPLETE",
            "reasons": ["metric_recomputation_failed"],
        }
        reasons.append("METRIC_RECOMPUTATION_FAILED")
    producer_agrees = (
        producer_gate.get("classification") == gate["classification"]
    )
    if not producer_agrees:
        reasons.append("PRODUCER_CLASSIFICATION_MISMATCH")
    return {
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "valid": not reasons and producer_agrees,
        "reasons": list(dict.fromkeys(reasons)),
        "recomputed_classification": gate["classification"],
        "gate_reasons": gate.get("reasons", []),
        "producer_classification": producer_gate.get("classification"),
        "producer_agrees": producer_agrees,
        "strongest_vllm_arm": strongest_vllm_arm,
        "comparison": comparison,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--expected-source", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = verify_bundle(
        args.bundle,
        expected_source=args.expected_source,
    )
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.write_text(payload, encoding="utf-8")
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
