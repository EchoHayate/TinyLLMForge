"""Reproducible evidence gate for the native multi-token verifier."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from pathlib import Path


POLICIES = (
    "baseline",
    "legacy_rematerialize",
    "native",
    "oracle",
)

REQUIRED_ARTIFACTS = (
    "manifest.json",
    "capability.json",
    "case_rows.json",
    "event_rows.json",
    "summary.json",
    "report.md",
)

THRESHOLDS = {
    "k1_max_regression_fraction": 0.01,
    "min_continuation_steps": 16,
    "native_k_gt_1_must_beat_legacy": True,
    "target_forward_reduction_must_equal_removed_replay": True,
}

DTYPE_TOLERANCES = {
    "torch.float16": {
        "logits_rtol": 2e-3,
        "logits_atol": 2e-3,
        "kv_rtol": 2e-3,
        "kv_atol": 2e-3,
    },
    "torch.bfloat16": {
        "logits_rtol": 8e-3,
        "logits_atol": 8e-3,
        "kv_rtol": 8e-3,
        "kv_atol": 8e-3,
    },
}

CLAIM_BOUNDARIES = (
    "profiler-owned only",
    "strict greedy decoding only",
    "single sequence only",
    "linear drafts only",
    "eager execution only",
    "FP16/BF16 KV only",
    "no production batch throughput claim",
    "no ragged or tree verification claim",
    "no non-greedy equivalence claim",
    "no CUDA graph support claim",
    "no KV offload support claim",
    "no quantized KV support claim",
    "no memory reduction claim",
    "no production GO claim",
)

_PROMPT = (
    "Repeat the sequence alpha beta gamma while preserving exact spacing: "
    "alpha beta gamma alpha beta gamma."
)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(value) -> str:
    return sha256_text(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _case(
    case_id: str,
    *,
    draft_len: int,
    acceptance_case: str,
    history_len: int,
    block_case: str,
    eos_case: bool = False,
    output_budget_case: bool = False,
) -> dict:
    return {
        "case_id": case_id,
        "prompt": _PROMPT,
        "prompt_sha256": sha256_text(_PROMPT),
        "history_len": history_len,
        "draft_len": draft_len,
        "acceptance_case": acceptance_case,
        "eos_case": eos_case,
        "output_budget_case": output_budget_case,
        "block_case": block_case,
        "block_size": 256,
        "continuation_steps": 16,
        "draft_construction": acceptance_case,
    }


CASE_MATRIX = (
    _case(
        "k1-full-current",
        draft_len=1,
        acceptance_case="full",
        history_len=52,
        block_case="current_block",
    ),
    _case(
        "k4-zero-current",
        draft_len=4,
        acceptance_case="zero",
        history_len=52,
        block_case="current_block",
    ),
    _case(
        "k4-one-current",
        draft_len=4,
        acceptance_case="one",
        history_len=52,
        block_case="current_block",
    ),
    _case(
        "k4-partial-boundary",
        draft_len=4,
        acceptance_case="partial",
        history_len=255,
        block_case="one_new_block",
    ),
    _case(
        "k4-full-boundary",
        draft_len=4,
        acceptance_case="full",
        history_len=255,
        block_case="one_new_block",
    ),
    _case(
        "k8-full-boundary",
        draft_len=8,
        acceptance_case="full",
        history_len=255,
        block_case="one_new_block",
    ),
    _case(
        "k8-eos-boundary",
        draft_len=8,
        acceptance_case="partial",
        history_len=255,
        block_case="one_new_block",
        eos_case=True,
    ),
    _case(
        "k8-budget-boundary",
        draft_len=8,
        acceptance_case="full",
        history_len=255,
        block_case="one_new_block",
        output_budget_case=True,
    ),
    _case(
        "k16-full-multiblock",
        draft_len=16,
        acceptance_case="full",
        history_len=511,
        block_case="multiple_new_blocks",
    ),
)


def build_manifest(
    source_commit: str,
    source_dirty: bool,
    model_path: str,
    model_identifier: str,
    host: str,
    python_bin: str,
    torch_version: str,
    cuda_version: str,
    flash_attn_version: str,
    gpu_name: str,
    bf16_supported: bool,
    run_tag: str,
) -> dict:
    return {
        "schema_version": 1,
        "run_tag": str(run_tag),
        "created_unix_s": time.time(),
        "source_commit": str(source_commit),
        "source_dirty": bool(source_dirty),
        "model_path": str(model_path),
        "model_identifier": str(model_identifier),
        "host": str(host),
        "python_bin": str(python_bin),
        "torch_version": str(torch_version),
        "cuda_version": str(cuda_version),
        "flash_attn_version": str(flash_attn_version),
        "gpu_name": str(gpu_name),
        "bf16_supported": bool(bf16_supported),
        "dtype_tolerances": DTYPE_TOLERANCES,
        "thresholds": THRESHOLDS,
        "case_matrix": list(CASE_MATRIX),
        "case_matrix_sha256": sha256_json(CASE_MATRIX),
        "policies": list(POLICIES),
        "required_artifacts": list(REQUIRED_ARTIFACTS),
        "claim_boundaries": list(CLAIM_BOUNDARIES),
        "classification_on_success": "READY_FOR_PERFORMANCE_GATE",
        "process_port_pairs": [],
        "prompt_history_hashes": {
            case["case_id"]: {
                "prompt_sha256": case["prompt_sha256"],
                "history_contract_sha256": sha256_json({
                    "history_len": case["history_len"],
                    "draft_len": case["draft_len"],
                    "acceptance_case": case["acceptance_case"],
                }),
            }
            for case in CASE_MATRIX
        },
        "artifact_hashes": {},
    }


def _row_key(row: dict) -> tuple[str, str]:
    return str(row.get("case_id")), str(row.get("policy"))


def _expected_row_keys() -> set[tuple[str, str]]:
    return {
        (case["case_id"], policy)
        for case in CASE_MATRIX
        for policy in POLICIES
    }


def _required_comparison_fields() -> tuple[str, ...]:
    return (
        "status",
        "target_token_match",
        "accepted_prefix_match",
        "metadata_match",
        "continuation_token_match",
        "continuation_steps",
        "finite",
        "max_logit_abs_error",
        "max_kv_abs_error",
        "logits_within_tolerance",
        "kv_within_tolerance",
    )


def _median(values) -> float:
    values = [float(value) for value in values]
    if not values:
        raise ValueError("median requires evidence")
    return float(statistics.median(values))


def _incomplete(reasons: list[str], **extra) -> dict:
    return {
        "classification": "INCOMPLETE",
        "reasons": reasons,
        "exactness_pass": False,
        "replay_elimination_pass": False,
        "performance_direction_pass": False,
        "memory_is_diagnostic_only": True,
        **extra,
    }


def _no_go(reasons: list[str], **extra) -> dict:
    return {
        "classification": "NO_GO",
        "reasons": reasons,
        "exactness_pass": not any(
            "mismatch" in reason
            or "exactness" in reason
            or "non-finite" in reason
            for reason in reasons
        ),
        "replay_elimination_pass": not any(
            term in reason for reason in reasons
            for term in ("replay", "copy", "rematerialization")
        ),
        "performance_direction_pass": False,
        "memory_is_diagnostic_only": True,
        **extra,
    }


def classify_gate(
    manifest: dict,
    capability: dict,
    case_rows: list[dict],
    event_rows: list[dict],
) -> dict:
    structural = []
    if manifest.get("source_dirty") is not False:
        structural.append("canonical source_dirty must be false")
    if manifest.get("source_commit") in (None, ""):
        structural.append("source commit is missing")

    observed_keys = [_row_key(row) for row in case_rows]
    expected_keys = _expected_row_keys()
    if len(observed_keys) != len(set(observed_keys)):
        structural.append("duplicate policy/case rows")
    missing_keys = sorted(expected_keys - set(observed_keys))
    extra_keys = sorted(set(observed_keys) - expected_keys)
    if missing_keys:
        structural.append(f"missing policy/case rows: {missing_keys}")
    if extra_keys:
        structural.append(f"unexpected policy/case rows: {extra_keys}")
    for row in case_rows:
        process = row.get("process")
        if not isinstance(process, dict):
            structural.append(f"{_row_key(row)} missing process evidence")
            continue
        if process.get("returncode") != 0:
            structural.append(f"{_row_key(row)} process failed")
        for field in ("tinyvllm_dist_port", "master_port"):
            if not isinstance(process.get(field), int):
                structural.append(
                    f"{_row_key(row)} missing dynamic {field}"
                )
        if row.get("source_commit") != manifest.get("source_commit"):
            structural.append(f"{_row_key(row)} source commit mismatch")
        if row.get("source_dirty") is not False:
            structural.append(f"{_row_key(row)} source_dirty evidence")

    native_events = [
        event for event in event_rows
        if event.get("policy") == "native"
    ]
    native_event_ids = [
        str(event.get("case_id")) for event in native_events
    ]
    expected_case_ids = {case["case_id"] for case in CASE_MATRIX}
    if len(native_event_ids) != len(set(native_event_ids)):
        structural.append("duplicate native event rows")
    if set(native_event_ids) != expected_case_ids:
        structural.append("missing or unexpected native event rows")
    for event in native_events:
        for field in (
            "accepted_kv_rematerialization",
            "accepted_kv_copy_calls",
            "accepted_kv_replay_calls",
            "target_forward_count",
            "legacy_decode_replay_calls",
            "legacy_total_target_forwards",
            "native_total_target_forwards",
            "verifier_commit_ms",
            "legacy_verifier_commit_ms",
        ):
            if field not in event:
                structural.append(
                    f"{event.get('case_id')} native event missing {field}"
                )
        rematerialization = event.get(
            "accepted_kv_rematerialization",
            {},
        )
        for field in (
            "decode_calls",
            "rematerialized_tokens",
            "elapsed_ms",
        ):
            if field not in rematerialization:
                structural.append(
                    f"{event.get('case_id')} rematerialization missing {field}"
                )
    if structural:
        return _incomplete(structural)

    if capability.get("status") != "PASS":
        return _incomplete(["FlashAttention capability is unavailable"])
    capability_rows = capability.get("rows")
    if not isinstance(capability_rows, list):
        return _incomplete(["capability rows are missing"])
    required_dtypes = {"torch.float16"}
    if manifest.get("bf16_supported"):
        required_dtypes.add("torch.bfloat16")
    required_capability = {
        (dtype, query_len, block_case)
        for dtype in required_dtypes
        for query_len in (1, 3, 7, 15)
        for block_case in ("one_block", "cross_block")
    }
    observed_capability = {
        (
            row.get("dtype"),
            row.get("query_len"),
            row.get("block_case"),
        )
        for row in capability_rows
        if all(
            row.get(field) is True
            for field in (
                "gqa",
                "output_match",
                "kv_match",
                "future_row_masked",
                "finite",
            )
        )
    }
    if not required_capability <= observed_capability:
        return _incomplete(["capability matrix is incomplete or divergent"])

    by_key = {_row_key(row): row for row in case_rows}
    semantic_reasons = []
    for case in CASE_MATRIX:
        case_id = case["case_id"]
        rows = {
            policy: by_key[(case_id, policy)]
            for policy in POLICIES
        }
        for policy, row in rows.items():
            if row.get("status") == "INCOMPLETE":
                return _incomplete(
                    [f"{case_id}/{policy} row is INCOMPLETE"]
                )
            if row.get("status") != "PASS":
                semantic_reasons.append(
                    f"{case_id}/{policy} semantic failure"
                )
        comparison = rows["oracle"].get("comparison")
        if not isinstance(comparison, dict):
            return _incomplete([f"{case_id} comparison is missing"])
        missing_comparison = [
            field for field in _required_comparison_fields()
            if field not in comparison
        ]
        if missing_comparison:
            return _incomplete([
                f"{case_id} comparison missing {missing_comparison}"
            ])
        if comparison["status"] == "INCOMPLETE":
            return _incomplete(
                [f"{case_id} comparison is INCOMPLETE"]
            )
        if comparison["status"] != "PASS":
            semantic_reasons.extend(
                comparison.get("reasons")
                or [f"{case_id} native/oracle exactness failure"]
            )
        for field in (
            "target_token_match",
            "accepted_prefix_match",
            "metadata_match",
            "continuation_token_match",
            "finite",
            "logits_within_tolerance",
            "kv_within_tolerance",
        ):
            if comparison.get(field) is not True:
                semantic_reasons.append(
                    f"{case_id} comparison {field} mismatch"
                )
        if comparison["continuation_steps"] < THRESHOLDS[
            "min_continuation_steps"
        ]:
            semantic_reasons.append(
                f"{case_id} continuation exactness below 16"
            )

        baseline_hash = rows["baseline"].get("output_token_sha256")
        continuation_hash = rows["baseline"].get(
            "continuation_token_sha256"
        )
        if not baseline_hash or not continuation_hash:
            return _incomplete(
                [f"{case_id} baseline token hashes are missing"]
            )
        for policy in ("legacy_rematerialize", "native", "oracle"):
            if rows[policy].get("output_token_sha256") != baseline_hash:
                semantic_reasons.append(
                    f"{case_id}/{policy} output token mismatch"
                )
            if (
                rows[policy].get("continuation_token_sha256")
                != continuation_hash
            ):
                semantic_reasons.append(
                    f"{case_id}/{policy} continuation token mismatch"
                )
        if (
            rows["native"].get("accepted_tokens")
            != rows["oracle"].get("accepted_tokens")
        ):
            semantic_reasons.append(
                f"{case_id} accepted prefix mismatch"
            )
        if (
            rows["native"].get("sequence_token_sha256")
            != rows["oracle"].get("sequence_token_sha256")
            or rows["native"].get("block_table_after")
            != rows["oracle"].get("block_table_after")
        ):
            semantic_reasons.append(
                f"{case_id} committed lifecycle mismatch"
            )
    if semantic_reasons:
        return _no_go(sorted(set(semantic_reasons)))

    replay_reasons = []
    for event in native_events:
        case_id = event["case_id"]
        rematerialization = event["accepted_kv_rematerialization"]
        if (
            rematerialization["decode_calls"] != 0
            or rematerialization["rematerialized_tokens"]
            or float(rematerialization["elapsed_ms"]) != 0.0
        ):
            replay_reasons.append(
                f"{case_id} accepted KV rematerialization remains"
            )
        if event["accepted_kv_copy_calls"] != 0:
            replay_reasons.append(f"{case_id} accepted KV copy remains")
        if event["accepted_kv_replay_calls"] != 0:
            replay_reasons.append(f"{case_id} accepted KV replay remains")
        removed = (
            int(event["legacy_total_target_forwards"])
            - int(event["native_total_target_forwards"])
        )
        if removed != int(event["legacy_decode_replay_calls"]):
            replay_reasons.append(
                f"{case_id} target forward reduction does not equal "
                "removed replay calls"
            )
        if (
            int(event["accepted_count"]) == 0
            and event.get("zero_accept_included_in_throughput") is not True
        ):
            replay_reasons.append(
                f"{case_id} zero-accept event excluded from throughput"
            )
    if replay_reasons:
        return _no_go(replay_reasons)

    performance_missing = []
    for row in case_rows:
        for field in (
            "elapsed_s",
            "output_tokens",
            "output_tokens_per_s",
            "max_allocated_gpu_memory_bytes",
        ):
            value = row.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(
                float(value)
            ):
                performance_missing.append(
                    f"{_row_key(row)} missing performance {field}"
                )
    for event in native_events:
        for field in (
            "verifier_commit_ms",
            "legacy_verifier_commit_ms",
        ):
            value = event.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(
                float(value)
            ):
                performance_missing.append(
                    f"{event['case_id']} missing {field}"
                )
    if performance_missing:
        return _incomplete(performance_missing)

    performance_reasons = []
    k1_ratios = []
    for case in CASE_MATRIX:
        if case["draft_len"] != 1:
            continue
        baseline = by_key[(case["case_id"], "baseline")]
        native = by_key[(case["case_id"], "native")]
        k1_ratios.append(
            float(native["elapsed_s"]) / float(baseline["elapsed_s"])
        )
    k1_regression = max(k1_ratios) - 1.0
    if k1_regression > THRESHOLDS["k1_max_regression_fraction"]:
        performance_reasons.append(
            f"K=1 native regression {k1_regression:.6f} exceeds 1%"
        )

    accepted_k_gt_1 = [
        event for event in native_events
        if int(event["draft_len"]) > 1
        and int(event["accepted_count"]) > 0
    ]
    if not accepted_k_gt_1:
        return _incomplete(
            ["no accepted K>1 performance evidence"]
        )
    native_median = _median(
        event["verifier_commit_ms"]
        for event in accepted_k_gt_1
    )
    legacy_median = _median(
        event["legacy_verifier_commit_ms"]
        for event in accepted_k_gt_1
    )
    if native_median >= legacy_median:
        performance_reasons.append(
            "accepted K>1 native verifier-plus-commit median "
            "does not beat legacy"
        )
    if performance_reasons:
        return _no_go(
            performance_reasons,
            exactness_pass=True,
            replay_elimination_pass=True,
            k1_regression_fraction=k1_regression,
            native_k_gt_1_median_ms=native_median,
            legacy_k_gt_1_median_ms=legacy_median,
        )

    return {
        "classification": "READY_FOR_PERFORMANCE_GATE",
        "reasons": [],
        "exactness_pass": True,
        "replay_elimination_pass": True,
        "performance_direction_pass": True,
        "memory_is_diagnostic_only": True,
        "observed_case_rows": len(case_rows),
        "observed_native_events": len(native_events),
        "k1_regression_fraction": k1_regression,
        "native_k_gt_1_median_ms": native_median,
        "legacy_k_gt_1_median_ms": legacy_median,
        "max_allocated_gpu_memory_bytes": max(
            int(row["max_allocated_gpu_memory_bytes"])
            for row in case_rows
        ),
    }


def render_report(
    manifest: dict,
    capability: dict,
    case_rows: list[dict],
    event_rows: list[dict],
    summary: dict,
) -> str:
    lines = [
        "# Native Multi-Token Verifier Gate",
        "",
        "## Environment",
        "",
        "| Field | Value |",
        "| --- | --- |",
    ]
    for field in (
        "run_tag",
        "source_commit",
        "source_dirty",
        "model_identifier",
        "model_path",
        "host",
        "python_bin",
        "torch_version",
        "cuda_version",
        "flash_attn_version",
        "gpu_name",
    ):
        lines.append(f"| {field} | {manifest.get(field)} |")
    lines.extend([
        "",
        "## Capability",
        "",
        f"- Status: `{capability.get('status')}`",
        f"- Rows: `{len(capability.get('rows', []))}`",
        "- Required query lengths: `Q in {1,3,7,15}`",
        "",
        "## Exactness",
        "",
        "| Case | Oracle | Continuation | Max Logit Error | Max KV Error |",
        "| --- | --- | ---: | ---: | ---: |",
    ])
    oracle_rows = {
        row["case_id"]: row
        for row in case_rows
        if row.get("policy") == "oracle"
    }
    for case in CASE_MATRIX:
        comparison = oracle_rows.get(
            case["case_id"],
            {},
        ).get("comparison", {})
        lines.append(
            "| {case} | {status} | {steps} | {logit} | {kv} |".format(
                case=case["case_id"],
                status=comparison.get("status", "missing"),
                steps=comparison.get("continuation_steps", "missing"),
                logit=comparison.get("max_logit_abs_error", "missing"),
                kv=comparison.get("max_kv_abs_error", "missing"),
            )
        )
    lines.extend([
        "",
        "## Performance Direction",
        "",
        f"- K=1 regression fraction: `{summary.get('k1_regression_fraction')}`",
        "- Native accepted K>1 verifier-plus-commit median ms: "
        f"`{summary.get('native_k_gt_1_median_ms')}`",
        "- Legacy accepted K>1 verifier-plus-commit median ms: "
        f"`{summary.get('legacy_k_gt_1_median_ms')}`",
        "- Zero-accept events are included in end-to-end throughput.",
        "- Maximum allocated GPU memory is diagnostic only: "
        f"`{summary.get('max_allocated_gpu_memory_bytes')}` bytes.",
        "",
        "## Classification",
        "",
        f"Classification: `{summary['classification']}`",
        "",
        "Reasons:",
    ])
    if summary.get("reasons"):
        lines.extend(f"- {reason}" for reason in summary["reasons"])
    else:
        lines.append("- none")
    lines.extend([
        "",
        "## Non-Claims",
        "",
    ])
    lines.extend(f"- {boundary}" for boundary in CLAIM_BOUNDARIES)
    return "\n".join(lines) + "\n"


def _validate_manifest(manifest: dict) -> None:
    expected = {
        "dtype_tolerances": DTYPE_TOLERANCES,
        "thresholds": THRESHOLDS,
        "case_matrix": list(CASE_MATRIX),
        "case_matrix_sha256": sha256_json(CASE_MATRIX),
        "policies": list(POLICIES),
        "required_artifacts": list(REQUIRED_ARTIFACTS),
        "claim_boundaries": list(CLAIM_BOUNDARIES),
        "classification_on_success": "READY_FOR_PERFORMANCE_GATE",
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(f"manifest {field} drift")
    if manifest.get("source_dirty") is not False:
        raise ValueError("manifest source_dirty must be false")


def verify_artifacts(out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    for name in REQUIRED_ARTIFACTS:
        if not (out_dir / name).is_file():
            raise ValueError(f"missing artifact: {name}")
    manifest = json.loads((out_dir / "manifest.json").read_text())
    _validate_manifest(manifest)
    artifact_hashes = manifest.get("artifact_hashes")
    if not isinstance(artifact_hashes, dict):
        raise ValueError("manifest artifact_hashes are missing")
    for name in REQUIRED_ARTIFACTS:
        if name == "manifest.json":
            continue
        expected_hash = artifact_hashes.get(name)
        if not expected_hash:
            raise ValueError(f"missing SHA-256 for {name}")
        actual_hash = sha256_file(out_dir / name)
        if actual_hash != expected_hash:
            raise ValueError(f"SHA-256 mismatch for {name}")

    capability = json.loads((out_dir / "capability.json").read_text())
    case_rows = json.loads((out_dir / "case_rows.json").read_text())
    event_rows = json.loads((out_dir / "event_rows.json").read_text())
    recorded_summary = json.loads((out_dir / "summary.json").read_text())
    computed_summary = classify_gate(
        manifest,
        capability,
        case_rows,
        event_rows,
    )
    if recorded_summary != computed_summary:
        raise ValueError("summary.json differs from recomputed gate")
    expected_report = render_report(
        manifest,
        capability,
        case_rows,
        event_rows,
        computed_summary,
    )
    if (out_dir / "report.md").read_text() != expected_report:
        raise ValueError("report classification or contents differ")
    return computed_summary


def _parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--out-dir", required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--out-dir", required=True)
    report_parser = subparsers.add_parser("render-report")
    report_parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def main():
    args = _parse_args()
    out_dir = Path(args.out_dir)
    if args.command == "verify":
        result = verify_artifacts(out_dir)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.command == "render-report":
        manifest = json.loads((out_dir / "manifest.json").read_text())
        capability = json.loads((out_dir / "capability.json").read_text())
        case_rows = json.loads((out_dir / "case_rows.json").read_text())
        event_rows = json.loads((out_dir / "event_rows.json").read_text())
        summary = classify_gate(
            manifest,
            capability,
            case_rows,
            event_rows,
        )
        print(
            render_report(
                manifest,
                capability,
                case_rows,
                event_rows,
                summary,
            ),
            end="",
        )
        return
    raise SystemExit(
        "run is implemented by the isolated remote runner in Task 8"
    )


if __name__ == "__main__":
    main()
