from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools/qwen35_tp4_w2_restore_profile.py"


def _load():
    spec = importlib.util.spec_from_file_location(
        "qwen35_tp4_w2_restore_profile",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


profile = _load()


def _write_case(
    root,
    policy,
    repetition,
    *,
    generated_tokens=64,
    makespan_ns=None,
):
    case_id = (
        f"w2_long_reuse__measured__r{repetition}__{policy}"
    )
    case_dir = root / case_id
    case_dir.mkdir()
    reused = 3840 if policy == "exact_restore" else 0
    prefill = 64 if policy == "exact_restore" else 3904
    rows = []
    for request_index in range(4):
        rows.append({
            "request_id": f"request-{request_index}",
            "e2e_ns": (
                makespan_ns
                if makespan_ns is not None
                else 5_000 + repetition * 10
            ),
            "ttft_ns": 600 + repetition,
            "decode_step_ns": [
                100 for _ in range(generated_tokens - 1)
            ],
            "executed_prefill_tokens": prefill,
            "reused_kv_tokens": reused,
            "generated_tokens": generated_tokens,
            "output_token_ids": list(range(generated_tokens)),
        })
    (case_dir / "case_rows.jsonl").write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    events = []
    if policy == "exact_restore":
        events.extend([
            {
                "name": "release_flush",
                "request_id": None,
                "duration_ns": 7,
                "status": "ok",
            },
            {
                "name": "restore_total",
                "request_id": 0,
                "duration_ns": 11,
                "status": "ok",
            },
        ])
        for request_index in range(4):
            request_id = request_index + 1
            events.extend([
                {
                    "name": "release_flush",
                    "request_id": None,
                    "duration_ns": 10,
                    "status": "ok",
                },
                {
                    "name": "restore_prepare",
                    "request_id": request_id,
                    "duration_ns": 20,
                    "status": "ok",
                },
                {
                    "name": "restore_validate",
                    "request_id": request_id,
                    "duration_ns": 30,
                    "status": "ok",
                },
                {
                    "name": "restore_commit",
                    "request_id": request_id,
                    "duration_ns": 40,
                    "status": "ok",
                },
                {
                    "name": "restore_total",
                    "request_id": request_id,
                    "duration_ns": 100,
                    "status": "ok",
                },
            ])
    (case_dir / "profile.json").write_text(
        json.dumps({
            "schema_version": (
                "qwen35.tp4-w2-restore-profile-case.v1"
            ),
            "case_id": case_id,
            "policy": policy,
            "workload": "w2_long_reuse",
            "phase": "measured",
            "repetition": repetition,
            "variant": (
                "canonical_output"
                if generated_tokens == 64
                else "short_output"
            ),
            "canonical_generated_tokens": 64,
            "generated_tokens": generated_tokens,
            "events": events,
            "requests": [],
        }, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_aggregate_requires_five_paired_repetitions():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for repetition in range(4):
            for policy in ("recompute", "exact_restore"):
                _write_case(root, policy, repetition)

        try:
            profile.aggregate_profile(root)
        except ValueError as error:
            assert "five paired repetitions" in str(error)
        else:
            raise AssertionError("incomplete profile was accepted")


def test_aggregate_reports_restore_decode_and_reuse_boundaries():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for repetition in range(5):
            for policy in ("recompute", "exact_restore"):
                _write_case(root, policy, repetition)

        result = profile.aggregate_profile(root)

    assert result["schema_version"] == (
        "qwen35.tp4-w2-restore-profile.v1"
    )
    assert len(result["paired_repetitions"]) == 5
    exact = result["summary"]["exact_restore"]
    assert exact["median_executed_prefill_tokens"] == 256
    assert exact["median_reused_kv_tokens"] == 15360
    assert exact["median_restore_total_ns"] == 400
    assert exact["median_release_flush_ns"] == 40
    assert exact["median_restore_prepare_ns"] == 80
    assert exact["median_restore_validate_ns"] == 120
    assert exact["median_restore_commit_ns"] == 160
    assert result["evidence_boundary"] == (
        "restore_prepare includes rank-local restore work plus "
        "acknowledgement transport and waiting"
    )
    comparison = result["comparison"]
    assert comparison["prefill_token_reduction_fraction"] == (
        1.0 - 256 / 15616
    )
    assert comparison["median_restore_total_per_request_ns"] == 100
    assert comparison["median_restore_prepare_share"] == 0.2
    assert result["generated_tokens"] == 64


def test_aggregate_validates_eight_token_short_output_cases():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for repetition in range(5):
            for policy in ("recompute", "exact_restore"):
                _write_case(
                    root,
                    policy,
                    repetition,
                    generated_tokens=8,
                )

        result = profile.aggregate_profile(
            root,
            generated_tokens=8,
        )

    assert result["generated_tokens"] == 8
    assert all(
        row[policy]["generated_tokens"] == 8
        for row in result["paired_repetitions"]
        for policy in ("recompute", "exact_restore")
    )


def test_aggregate_rejects_generated_token_mismatch():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for repetition in range(5):
            for policy in ("recompute", "exact_restore"):
                _write_case(
                    root,
                    policy,
                    repetition,
                    generated_tokens=(
                        7
                        if repetition == 3
                        and policy == "exact_restore"
                        else 8
                    ),
                )

        try:
            profile.aggregate_profile(root, generated_tokens=8)
        except ValueError as error:
            assert "generated tokens" in str(error)
        else:
            raise AssertionError(
                "mismatched generated-token case was accepted"
            )


def test_aggregate_reports_ratio_of_medians_and_direction():
    recompute_makespans = (50, 100, 100, 500, 500)
    exact_makespans = (50, 50, 200, 200, 200)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for repetition in range(5):
            _write_case(
                root,
                "recompute",
                repetition,
                generated_tokens=8,
                makespan_ns=recompute_makespans[repetition],
            )
            _write_case(
                root,
                "exact_restore",
                repetition,
                generated_tokens=8,
                makespan_ns=exact_makespans[repetition],
            )

        result = profile.aggregate_profile(
            root,
            generated_tokens=8,
        )

    comparison = result["comparison"]
    assert comparison["median_makespan_speedup"] > 1.0
    assert (
        comparison["ratio_of_median_makespans_speedup"]
        < 1.0
    )
    assert (
        comparison["makespan_speedup_direction_agreement"]
        is False
    )
    assert comparison["makespan_classification"] == "inconclusive"
