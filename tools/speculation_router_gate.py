"""Source-auditable profitability gate for routed speculation."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path


OWNED_SOURCE_ROOTS = (
    "tinyvllm",
    "tools/draft_model_schema.py",
    "tools/profile_ngram_commit.py",
    "tools/source_audit.py",
    "tools/speculation_router_gate.py",
    "tools/test_speculation_router.py",
    "tools/test_speculation_router_gate.py",
    "tools/native_verifier_oracle.py",
    "tools/test_native_verifier_oracle.py",
    "tools/run_speculation_router_gate_remote.sh",
)

CONTROLLED_POLICIES = (
    "baseline",
    "legacy_rematerialize",
    "always_native",
    "routed_native",
    "oracle",
)

CONTROLLED_THRESHOLDS = {
    "profitable_region_max_elapsed_ratio": 0.95,
    "max_required_lifecycle_elapsed_ratio": 1.05,
    "min_continuation_steps": 16,
}


def _case(
    case_id: str,
    *,
    draft_len: int,
    acceptance_case: str,
    expected_accepted_count: int,
    history_len: int,
    block_case: str,
    eos_case: bool = False,
    output_budget_case: bool = False,
) -> dict:
    return {
        "case_id": case_id,
        "draft_len": draft_len,
        "acceptance_case": acceptance_case,
        "expected_accepted_count": expected_accepted_count,
        "history_len": history_len,
        "block_case": block_case,
        "eos_case": eos_case,
        "output_budget_case": output_budget_case,
        "continuation_steps": 16,
        "draft_construction": "controlled_target_derived",
    }


CONTROLLED_CASE_MATRIX = (
    _case(
        "k1-route-fallback",
        draft_len=1,
        acceptance_case="full",
        expected_accepted_count=1,
        history_len=52,
        block_case="current_block",
    ),
    _case(
        "k2-zero-current",
        draft_len=2,
        acceptance_case="zero",
        expected_accepted_count=0,
        history_len=52,
        block_case="current_block",
    ),
    _case(
        "k2-one-current",
        draft_len=2,
        acceptance_case="one",
        expected_accepted_count=1,
        history_len=52,
        block_case="current_block",
    ),
    _case(
        "k2-full-current",
        draft_len=2,
        acceptance_case="full",
        expected_accepted_count=2,
        history_len=52,
        block_case="current_block",
    ),
    _case(
        "k4-zero-current",
        draft_len=4,
        acceptance_case="zero",
        expected_accepted_count=0,
        history_len=52,
        block_case="current_block",
    ),
    _case(
        "k4-one-current",
        draft_len=4,
        acceptance_case="one",
        expected_accepted_count=1,
        history_len=52,
        block_case="current_block",
    ),
    _case(
        "k4-partial-boundary",
        draft_len=4,
        acceptance_case="partial",
        expected_accepted_count=2,
        history_len=255,
        block_case="one_new_block",
    ),
    _case(
        "k4-full-boundary",
        draft_len=4,
        acceptance_case="full",
        expected_accepted_count=4,
        history_len=255,
        block_case="one_new_block",
    ),
    _case(
        "k8-zero-current",
        draft_len=8,
        acceptance_case="zero",
        expected_accepted_count=0,
        history_len=52,
        block_case="current_block",
    ),
    _case(
        "k8-one-current",
        draft_len=8,
        acceptance_case="one",
        expected_accepted_count=1,
        history_len=52,
        block_case="current_block",
    ),
    _case(
        "k8-partial-boundary",
        draft_len=8,
        acceptance_case="partial",
        expected_accepted_count=4,
        history_len=255,
        block_case="one_new_block",
    ),
    _case(
        "k8-full-boundary",
        draft_len=8,
        acceptance_case="full",
        expected_accepted_count=8,
        history_len=255,
        block_case="one_new_block",
    ),
    _case(
        "k8-eos-boundary",
        draft_len=8,
        acceptance_case="partial",
        expected_accepted_count=2,
        history_len=255,
        block_case="real_eos_history",
        eos_case=True,
    ),
    _case(
        "k8-budget-boundary",
        draft_len=8,
        acceptance_case="full",
        expected_accepted_count=3,
        history_len=255,
        block_case="one_new_block",
        output_budget_case=True,
    ),
    _case(
        "k16-zero-multiblock",
        draft_len=16,
        acceptance_case="zero",
        expected_accepted_count=0,
        history_len=511,
        block_case="multi_block_context",
    ),
    _case(
        "k16-one-multiblock",
        draft_len=16,
        acceptance_case="one",
        expected_accepted_count=1,
        history_len=511,
        block_case="multi_block_context",
    ),
    _case(
        "k16-partial-multiblock",
        draft_len=16,
        acceptance_case="partial",
        expected_accepted_count=8,
        history_len=511,
        block_case="multi_block_context",
    ),
    _case(
        "k16-full-multiblock",
        draft_len=16,
        acceptance_case="full",
        expected_accepted_count=16,
        history_len=511,
        block_case="multi_block_context",
    ),
)


def build_controlled_manifest(
    *,
    source_evidence: dict,
    source_preflight: dict,
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
        "stage": "controlled",
        "run_tag": run_tag,
        "created_unix_s": time.time(),
        "source_commit": source_evidence["base_commit"],
        "source_dirty": source_evidence["dirty"],
        "source_tree_sha256": source_evidence["tree_sha256"],
        "source_evidence": source_evidence,
        "source_preflight": source_preflight,
        "model_path": model_path,
        "model_identifier": model_identifier,
        "host": host,
        "python_bin": python_bin,
        "torch_version": torch_version,
        "cuda_version": cuda_version,
        "flash_attn_version": flash_attn_version,
        "gpu_name": gpu_name,
        "bf16_supported": bool(bf16_supported),
        "thresholds": CONTROLLED_THRESHOLDS,
        "case_matrix": list(CONTROLLED_CASE_MATRIX),
        "policies": list(CONTROLLED_POLICIES),
        "classification_on_success": (
            "READY_FOR_REAL_DRAFTER_GATE"
        ),
        "process_port_pairs": [],
    }


def _incomplete(reasons: list[str]) -> dict:
    return {
        "classification": "INCOMPLETE",
        "reasons": sorted(set(reasons)),
        "exactness_pass": False,
        "replay_elimination_pass": False,
        "router_isolation_pass": False,
        "performance_direction_pass": False,
    }


def _no_go(
    reasons: list[str],
    *,
    exactness_pass: bool,
    replay_elimination_pass: bool,
    router_isolation_pass: bool,
    **extra,
) -> dict:
    return {
        "classification": "NO_GO",
        "reasons": sorted(set(reasons)),
        "exactness_pass": exactness_pass,
        "replay_elimination_pass": replay_elimination_pass,
        "router_isolation_pass": router_isolation_pass,
        "performance_direction_pass": False,
        **extra,
    }


def _row_key(row: dict) -> tuple[str, str]:
    return str(row.get("case_id")), str(row.get("policy"))


def _expected_row_keys() -> set[tuple[str, str]]:
    return {
        (case["case_id"], policy)
        for case in CONTROLLED_CASE_MATRIX
        for policy in CONTROLLED_POLICIES
    }


def _capability_complete(
    manifest: dict,
    capability: dict,
) -> bool:
    if capability.get("status") != "PASS":
        return False
    required_dtypes = {"torch.float16"}
    if manifest.get("bf16_supported"):
        required_dtypes.add("torch.bfloat16")
    required = {
        (dtype, query_len, block_case)
        for dtype in required_dtypes
        for query_len in (1, 3, 7, 15)
        for block_case in ("one_block", "cross_block")
    }
    observed = {
        (
            row.get("dtype"),
            row.get("query_len"),
            row.get("block_case"),
        )
        for row in capability.get("rows", [])
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
    return required <= observed


def classify_controlled_gate(
    manifest: dict,
    capability: dict,
    case_rows: list[dict],
    event_rows: list[dict],
    router_rows: list[dict],
) -> dict:
    structural = []
    if manifest.get("stage") != "controlled":
        structural.append("manifest stage mismatch")
    if manifest.get("thresholds") != CONTROLLED_THRESHOLDS:
        structural.append("manifest thresholds drift")
    if manifest.get("case_matrix") != list(
        CONTROLLED_CASE_MATRIX
    ):
        structural.append("manifest case matrix drift")
    if manifest.get("policies") != list(CONTROLLED_POLICIES):
        structural.append("manifest policies drift")
    evidence = manifest.get("source_evidence", {})
    if (
        manifest.get("source_tree_sha256")
        != evidence.get("tree_sha256")
    ):
        structural.append("manifest source tree mismatch")
    if (
        manifest.get("source_preflight", {}).get(
            "source_tree_sha256"
        )
        != manifest.get("source_tree_sha256")
    ):
        structural.append("source preflight tree mismatch")

    observed_keys = [_row_key(row) for row in case_rows]
    expected_keys = _expected_row_keys()
    if len(observed_keys) != len(set(observed_keys)):
        structural.append("duplicate policy/case rows")
    if set(observed_keys) != expected_keys:
        structural.append("missing or unexpected policy/case rows")
    for row in case_rows:
        process = row.get("process")
        if not isinstance(process, dict):
            structural.append(
                f"{_row_key(row)} missing process evidence"
            )
            continue
        if process.get("returncode") != 0:
            structural.append(f"{_row_key(row)} process failed")
        for field in (
            "tinyvllm_dist_port",
            "master_port",
        ):
            if not isinstance(process.get(field), int):
                structural.append(
                    f"{_row_key(row)} missing dynamic {field}"
                )
        if (
            row.get("source_tree_sha256")
            != manifest.get("source_tree_sha256")
        ):
            structural.append(
                f"{_row_key(row)} source tree mismatch"
            )
        for field in (
            "elapsed_s",
            "output_tokens",
            "output_tokens_per_s",
            "target_forward_count",
        ):
            value = row.get(field)
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                structural.append(
                    f"{_row_key(row)} invalid {field}"
                )

    port_pairs = [
        (
            row.get("tinyvllm_dist_port"),
            row.get("master_port"),
        )
        for row in manifest.get("process_port_pairs", [])
    ]
    if len(port_pairs) != len(set(port_pairs)):
        structural.append("duplicate dynamic port pairs")

    expected_router_cases = {
        case["case_id"] for case in CONTROLLED_CASE_MATRIX
    }
    router_keys = [
        (row.get("case_id"), row.get("policy"))
        for row in router_rows
    ]
    if len(router_keys) != len(set(router_keys)):
        structural.append("duplicate router rows")
    if {
        case_id for case_id, policy in router_keys
        if policy == "routed_native"
    } != expected_router_cases:
        structural.append("missing or unexpected router rows")

    native_event_keys = [
        (event.get("case_id"), event.get("policy"))
        for event in event_rows
    ]
    expected_event_keys = {
        (case["case_id"], policy)
        for case in CONTROLLED_CASE_MATRIX
        for policy in ("always_native", "routed_native")
        if not (
            policy == "routed_native"
            and case["draft_len"] <= 1
        )
    }
    if len(native_event_keys) != len(set(native_event_keys)):
        structural.append("duplicate native event rows")
    if set(native_event_keys) != expected_event_keys:
        structural.append("missing or unexpected native events")
    if structural:
        return _incomplete(structural)

    if not _capability_complete(manifest, capability):
        return _incomplete([
            "capability matrix is incomplete or divergent",
        ])

    by_key = {_row_key(row): row for row in case_rows}
    semantic = []
    for case in CONTROLLED_CASE_MATRIX:
        case_id = case["case_id"]
        baseline = by_key[(case_id, "baseline")]
        output_hash = baseline.get("output_token_sha256")
        continuation_hash = baseline.get(
            "continuation_token_sha256"
        )
        if not output_hash or not continuation_hash:
            return _incomplete([
                f"{case_id} baseline token hashes are missing",
            ])
        for policy in CONTROLLED_POLICIES:
            row = by_key[(case_id, policy)]
            if row.get("status") == "INCOMPLETE":
                return _incomplete([
                    f"{case_id}/{policy} row is INCOMPLETE",
                ])
            if row.get("status") != "PASS":
                semantic.append(
                    f"{case_id}/{policy} semantic failure"
                )
            if row.get("output_token_sha256") != output_hash:
                semantic.append(
                    f"{case_id}/{policy} output token mismatch"
                )
            if (
                row.get("continuation_token_sha256")
                != continuation_hash
            ):
                semantic.append(
                    f"{case_id}/{policy} continuation mismatch"
                )
        comparison = by_key[(case_id, "oracle")].get(
            "comparison"
        )
        if not isinstance(comparison, dict):
            return _incomplete([
                f"{case_id} comparison is missing",
            ])
        if comparison.get("status") == "INCOMPLETE":
            return _incomplete([
                f"{case_id} comparison is INCOMPLETE",
            ])
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
                semantic.append(
                    f"{case_id} comparison {field} mismatch"
                )
        if (
            int(comparison.get("continuation_steps", -1))
            < CONTROLLED_THRESHOLDS["min_continuation_steps"]
        ):
            semantic.append(
                f"{case_id} continuation below minimum"
            )
    if semantic:
        return _no_go(
            semantic,
            exactness_pass=False,
            replay_elimination_pass=False,
            router_isolation_pass=False,
        )

    replay = []
    for event in event_rows:
        rematerialization = event.get(
            "accepted_kv_rematerialization",
            {},
        )
        if (
            rematerialization.get("decode_calls") != 0
            or rematerialization.get("rematerialized_tokens")
            or float(rematerialization.get("elapsed_ms", math.nan))
            != 0.0
        ):
            replay.append(
                f"{event['case_id']}/{event['policy']} "
                "accepted KV rematerialization remains"
            )
        if event.get("accepted_kv_copy_calls") != 0:
            replay.append(
                f"{event['case_id']}/{event['policy']} "
                "accepted KV copy remains"
            )
        if event.get("accepted_kv_replay_calls") != 0:
            replay.append(
                f"{event['case_id']}/{event['policy']} "
                "accepted KV replay remains"
            )
    if replay:
        return _no_go(
            replay,
            exactness_pass=True,
            replay_elimination_pass=False,
            router_isolation_pass=False,
        )

    router_failures = []
    for row in router_rows:
        if int(row["draft_len"]) <= 1:
            if row.get("route") != "baseline_short_draft":
                router_failures.append(
                    f"{row['case_id']} short draft route mismatch"
                )
            for field in (
                "speculative_reservation_attempted",
                "spec_verify_prepare_calls",
                "spec_verify_forward_calls",
                "target_forward_count",
            ):
                expected = False if field == (
                    "speculative_reservation_attempted"
                ) else 0
                if row.get(field) != expected:
                    router_failures.append(
                        f"{row['case_id']} short route mutated {field}"
                    )
        elif row.get("route") != "native_multi_token":
            router_failures.append(
                f"{row['case_id']} multi-token route mismatch"
            )
    if router_failures:
        return _no_go(
            router_failures,
            exactness_pass=True,
            replay_elimination_pass=True,
            router_isolation_pass=False,
        )

    lifecycle_reasons = []
    profitable_ratios = []
    per_case_ratios = {}
    for case in CONTROLLED_CASE_MATRIX:
        case_id = case["case_id"]
        routed = by_key[(case_id, "routed_native")]
        baseline = by_key[(case_id, "baseline")]
        ratio = (
            float(routed["elapsed_s"])
            / float(baseline["elapsed_s"])
        )
        per_case_ratios[case_id] = ratio
        if ratio > CONTROLLED_THRESHOLDS[
            "max_required_lifecycle_elapsed_ratio"
        ]:
            lifecycle_reasons.append(
                f"required_lifecycle_regression:{case_id}"
            )
        if (
            int(case["draft_len"]) >= 2
            and int(case["expected_accepted_count"]) >= 2
        ):
            profitable_ratios.append(ratio)
    if lifecycle_reasons:
        return _no_go(
            lifecycle_reasons,
            exactness_pass=True,
            replay_elimination_pass=True,
            router_isolation_pass=True,
            per_case_elapsed_ratios=per_case_ratios,
        )
    best_ratio = min(profitable_ratios)
    if best_ratio >= CONTROLLED_THRESHOLDS[
        "profitable_region_max_elapsed_ratio"
    ]:
        return _no_go(
            ["no_profitable_k_ge_2_region"],
            exactness_pass=True,
            replay_elimination_pass=True,
            router_isolation_pass=True,
            best_profitable_region_elapsed_ratio=best_ratio,
            per_case_elapsed_ratios=per_case_ratios,
        )

    return {
        "classification": "READY_FOR_REAL_DRAFTER_GATE",
        "reasons": [],
        "exactness_pass": True,
        "replay_elimination_pass": True,
        "router_isolation_pass": True,
        "performance_direction_pass": True,
        "observed_case_rows": len(case_rows),
        "observed_native_events": len(event_rows),
        "observed_router_rows": len(router_rows),
        "best_profitable_region_elapsed_ratio": best_ratio,
        "median_profitable_region_elapsed_ratio": (
            statistics.median(profitable_ratios)
        ),
        "per_case_elapsed_ratios": per_case_ratios,
    }


def _parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.command == "verify":
        raise SystemExit(
            "artifact verification is not implemented yet"
        )


if __name__ == "__main__":
    main()
