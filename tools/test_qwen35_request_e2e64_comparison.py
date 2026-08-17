import json
from pathlib import Path

import pytest

import qwen35_request_e2e64_comparison as comparison


POLICIES = ("recompute", "exact_restore")
BASELINE_SOURCE = (
    "a26c543e79a9d4927fd0451d4a287363a677568a1daefe65a2a234a22f5997aa"
)
CANDIDATE_SOURCE = (
    "6f881fae7010cc5f048100b147a72fbf27ffba0f77bc34e2e2e68388a98a2837"
)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_attempt(
    root: Path,
    *,
    source_sha256: str,
    makespan_by_policy: dict[str, int],
    tokens: tuple[int, ...] = tuple(range(64)),
    row_count: int = 4,
    decode_step_count: int = 63,
) -> None:
    _write_json(
        root / "attempt_receipt.json",
        {
            "classification": "DOWNLOADED",
            "source_tree_sha256": source_sha256,
            "cleanup": {"classification": "CLEAN"},
        },
    )
    for repetition in range(5):
        for policy in POLICIES:
            case_id = (
                f"w2_long_reuse__measured__r{repetition}__{policy}"
            )
            case_root = (
                root / "download" / "output" / "cases" / case_id
            )
            makespan_ns = (
                makespan_by_policy[policy] + repetition * 1000
            )
            _write_json(
                case_root / "profile.json",
                {
                    "schema_version": (
                        "qwen35.tp4-w2-restore-profile-case.v1"
                    ),
                    "case_id": case_id,
                    "workload": "w2_long_reuse",
                    "phase": "measured",
                    "policy": policy,
                    "repetition": repetition,
                    "generated_tokens": 64,
                    "canonical_generated_tokens": 64,
                    "variant": "canonical_output",
                    "events": [],
                },
            )
            rows = []
            for request_index in range(row_count):
                request_id = f"request-{request_index}"
                rows.append({
                    "case_id": case_id,
                    "row_id": f"{case_id}__{request_id}",
                    "request_id": request_id,
                    "source_tree_sha256": source_sha256,
                    "workload": "w2_long_reuse",
                    "phase": "measured",
                    "policy": policy,
                    "repetition": repetition,
                    "generated_tokens": 64,
                    "output_token_ids": list(tokens),
                    "decode_step_ns": [
                        1_000_000 + request_index
                    ] * decode_step_count,
                    "ttft_ns": (
                        makespan_ns // 4 + request_index * 100
                    ),
                    "e2e_ns": (
                        makespan_ns
                        - (row_count - request_index - 1) * 100
                    ),
                })
            case_root.mkdir(parents=True, exist_ok=True)
            (case_root / "case_rows.jsonl").write_text(
                "".join(
                    json.dumps(row, sort_keys=True) + "\n"
                    for row in rows
                ),
                encoding="utf-8",
            )


def test_classifies_repeated_request_speedup_as_e2e_performance_pass(
    tmp_path,
):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_attempt(
        baseline,
        source_sha256=BASELINE_SOURCE,
        makespan_by_policy={
            "recompute": 1_000_000_000,
            "exact_restore": 900_000_000,
        },
    )
    _write_attempt(
        candidate,
        source_sha256=CANDIDATE_SOURCE,
        makespan_by_policy={
            "recompute": 880_000_000,
            "exact_restore": 792_000_000,
        },
    )

    result = comparison.compare_attempts(baseline, candidate)

    assert result["classification"] == "E2E_PERFORMANCE_PASS"
    assert result["output_parity"] is True
    for policy in POLICIES:
        policy_result = result["comparison"]["by_policy"][policy]
        assert policy_result["makespan_improvement_fraction"] > 0.1
        assert (
            policy_result[
                "request_throughput_improvement_fraction"
            ]
            > 0.1
        )
        pooled = policy_result["pooled_request_metrics"]
        assert pooled["e2e_ns"]["improvement_fraction"] > 0.1
        assert pooled["decode_ns"]["improvement_fraction"] == 0
        assert (
            policy_result[
                "output_token_throughput_improvement_fraction"
            ]
            > 0.1
        )


def test_rejects_output_token_mismatch(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_attempt(
        baseline,
        source_sha256=BASELINE_SOURCE,
        makespan_by_policy={
            "recompute": 1_000_000_000,
            "exact_restore": 900_000_000,
        },
    )
    _write_attempt(
        candidate,
        source_sha256=CANDIDATE_SOURCE,
        makespan_by_policy={
            "recompute": 800_000_000,
            "exact_restore": 700_000_000,
        },
        tokens=tuple(range(63)) + (999,),
    )

    result = comparison.compare_attempts(baseline, candidate)

    assert result["classification"] == "NO_GO"
    assert result["output_parity"] is False
    assert "token parity" in " ".join(result["reasons"])


def test_classifies_one_policy_improvement_as_mixed(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_attempt(
        baseline,
        source_sha256=BASELINE_SOURCE,
        makespan_by_policy={
            "recompute": 1_000_000_000,
            "exact_restore": 900_000_000,
        },
    )
    _write_attempt(
        candidate,
        source_sha256=CANDIDATE_SOURCE,
        makespan_by_policy={
            "recompute": 890_000_000,
            "exact_restore": 890_000_000,
        },
    )

    result = comparison.compare_attempts(baseline, candidate)

    assert result["classification"] == "MIXED"


def test_classifies_small_changes_as_no_material_change(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_attempt(
        baseline,
        source_sha256=BASELINE_SOURCE,
        makespan_by_policy={
            "recompute": 1_000_000_000,
            "exact_restore": 900_000_000,
        },
    )
    _write_attempt(
        candidate,
        source_sha256=CANDIDATE_SOURCE,
        makespan_by_policy={
            "recompute": 980_000_000,
            "exact_restore": 918_000_000,
        },
    )

    result = comparison.compare_attempts(baseline, candidate)

    assert result["classification"] == "NO_MATERIAL_E2E_CHANGE"


def test_classifies_material_slowdown_as_e2e_regression(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_attempt(
        baseline,
        source_sha256=BASELINE_SOURCE,
        makespan_by_policy={
            "recompute": 1_000_000_000,
            "exact_restore": 900_000_000,
        },
    )
    _write_attempt(
        candidate,
        source_sha256=CANDIDATE_SOURCE,
        makespan_by_policy={
            "recompute": 1_100_000_000,
            "exact_restore": 990_000_000,
        },
    )

    result = comparison.compare_attempts(baseline, candidate)

    assert result["classification"] == "E2E_REGRESSION"


@pytest.mark.parametrize(
    ("row_count", "decode_step_count", "reason"),
    (
        (3, 63, "four request rows"),
        (4, 62, "63 decode steps"),
    ),
)
def test_rejects_incomplete_request_shape(
    tmp_path,
    row_count,
    decode_step_count,
    reason,
):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_attempt(
        baseline,
        source_sha256=BASELINE_SOURCE,
        makespan_by_policy={
            "recompute": 1_000_000_000,
            "exact_restore": 900_000_000,
        },
    )
    _write_attempt(
        candidate,
        source_sha256=CANDIDATE_SOURCE,
        makespan_by_policy={
            "recompute": 900_000_000,
            "exact_restore": 800_000_000,
        },
        row_count=row_count,
        decode_step_count=decode_step_count,
    )

    result = comparison.compare_attempts(baseline, candidate)

    assert result["classification"] == "NO_GO"
    assert reason in " ".join(result["reasons"])
