#!/usr/bin/env python3
"""Dependency-light tests for TP4 decode replay evidence assembly."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile

from assemble_tp4_decode_replay import (
    PRODUCER_ARTIFACTS,
    REQUIRED_INPUTS,
    assemble_bundle,
)
import test_tp4_decode_replay_contract as contract_fixture


contract = contract_fixture.contract
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
RUN_TAG = "20260831-qwen38-tp4-decode-replay-r1"


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _source_identity() -> dict:
    return {
        "schema_version": "tinyllmforge.tp4-decode-replay-source.v1",
        "run_tag": RUN_TAG,
        "source_revision": "a" * 40,
        "source_tree_sha256": "b" * 64,
        "model_repository": "Qwen/Qwen3.8-27B",
        "model_revision": MODEL_REVISION,
    }


def _launch_admission() -> dict:
    return {
        "schema_version": "tinyllmforge.tp4-decode-replay-admission.v1",
        "run_tag": RUN_TAG,
        "strict_clean": True,
        "world_size": 4,
        "selected_gpus": [
            {
                "rank": rank,
                "index": rank,
                "uuid": f"GPU-{rank:04d}",
                "memory_used_mib": 128,
                "utilization_percent": 1,
                "compute_process_count": 0,
            }
            for rank in contract.RANKS
        ],
    }


def _cleanup() -> dict:
    return {
        "schema_version": "tinyllmforge.tp4-decode-replay-cleanup.v1",
        "run_tag": RUN_TAG,
        "classification": "CLEAN",
        "owned_children_remaining": [],
        "exact_tag_scans": [[], [], []],
        "rank_rows": [
            {
                "rank": rank,
                "exit_code": 0,
                "process_group_destroyed": True,
            }
            for rank in contract.RANKS
        ],
    }


def _request_rows(correctness_rows: list[dict]) -> list[dict]:
    by_pair = {row["pair_id"]: row for row in correctness_rows}
    rows = []
    for case in contract.build_case_matrix():
        outputs = by_pair[case["pair_id"]][
            f"{case['arm']}_outputs"
        ]
        for request_index, output in enumerate(outputs):
            rows.append({
                "row_id": (
                    f"{case['case_id']}:measured:request-{request_index}"
                ),
                "case_id": case["case_id"],
                "pair_id": case["pair_id"],
                "workload": case["workload"],
                "repetition": case["repetition"],
                "arm": case["arm"],
                "phase": "measured",
                "request_id": (
                    f"{case['case_id']}:request-{request_index}"
                ),
                "prompt_sha256": output["prompt_sha256"],
                "prompt_tokens": case["profile"]["prompt_tokens"],
                "generated_tokens": output["output_length"],
                "output_token_ids": copy.deepcopy(
                    output["output_token_ids"]
                ),
                "output_length": output["output_length"],
                "stop_reason": output["stop_reason"],
                "ttft_ns": 100_000_000,
                "tpot_ns": 95_000_000,
                "e2e_ns": 1_000_000_000,
                "admitted_ns": 1_000_000_000,
                "completed_ns": 2_000_000_000,
            })
    return rows


def _write_raw_attempt(root: Path) -> dict:
    evidence = contract_fixture._evidence()
    for correctness in evidence["correctness_rows"]:
        output_tokens = contract.WORKLOADS[
            correctness["workload"]
        ]["output_tokens"]
        for field in ("eager_outputs", "graph_outputs"):
            for output in correctness[field]:
                output["output_token_ids"] = [7] * output_tokens
                output["output_length"] = output_tokens
    source = _source_identity()
    _write_json(root / "source_manifest.json", source)
    (root / "source.patch").write_text(
        "diff --git a/frozen b/frozen\n",
        encoding="utf-8",
    )
    _write_json(root / "environment.json", {
        "schema_version": "tinyllmforge.tp4-decode-replay-environment.v1",
        "run_tag": RUN_TAG,
        "python": "3.11.9",
        "torch": "2.6.0",
        "cuda": "12.4",
    })
    _write_json(root / "gpu_inventory.json", _launch_admission())
    _write_json(root / "workload_profile.json", {
        "schema_version": "tinyllmforge.tp4-decode-replay-workload.v1",
        "run_tag": RUN_TAG,
        "model_repository": source["model_repository"],
        "model_revision": source["model_revision"],
        "dtype": "bfloat16",
        "tensor_parallel_size": 4,
        "temperature": 0.0,
        "measured_repetitions": contract.MEASURED_REPETITIONS,
        "workloads": contract.WORKLOADS,
        "cases": list(contract.build_case_matrix()),
    })
    _write_json(root / "process_receipts.json", {
        "schema_version": "tinyllmforge.tp4-decode-replay-processes.v1",
        "run_tag": RUN_TAG,
        "case_rows": [
            {
                "case_id": case["case_id"],
                "exit_code": 0,
                "timed_out": False,
                "dist_port": 20_000 + index,
                "started_ns": 1_000_000 + index * 10,
                "finished_ns": 1_000_005 + index * 10,
            }
            for index, case in enumerate(contract.build_case_matrix())
        ],
    })
    _write_jsonl(root / "rank_environment.jsonl", [
        {
            "row_id": f"environment:rank-{rank}",
            "run_tag": RUN_TAG,
            "rank": rank,
            "world_size": 4,
            "cuda_visible_device": str(rank),
        }
        for rank in contract.RANKS
    ])
    file_rows = {
        "rank_dispatch_events.jsonl": evidence[
            "rank_dispatch_rows"
        ],
        "rank_collective_events.jsonl": evidence[
            "rank_collective_rows"
        ],
        "rank_lifecycle_rows.jsonl": evidence[
            "rank_lifecycle_rows"
        ],
        "request_rows.jsonl": _request_rows(
            evidence["correctness_rows"]
        ),
        "performance_rows.jsonl": evidence["performance_rows"],
        "memory_rows.jsonl": evidence["memory_rows"],
        "correctness_rows.jsonl": evidence["correctness_rows"],
        "capture_cost_rows.jsonl": evidence["capture_cost_rows"],
    }
    for name, rows in file_rows.items():
        _write_jsonl(root / name, rows)
    return evidence


def _assemble(raw_root: Path, output_root: Path) -> dict:
    return assemble_bundle(
        raw_root=raw_root,
        output_root=output_root,
        source_identity=_source_identity(),
        launch_admission=_launch_admission(),
        cleanup=_cleanup(),
    )


def _expect_value_error(action, message: str) -> None:
    try:
        action()
    except ValueError as exc:
        assert message in str(exc), str(exc)
    else:
        raise AssertionError(f"expected ValueError containing {message!r}")


def test_assembler_writes_complete_manifested_go_bundle():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        raw = root / "raw"
        bundle = root / "final_bundle"
        raw.mkdir()
        _write_raw_attempt(raw)
        result = _assemble(raw, bundle)
        assert result["classification"] == "GO_STAGE1_JUSTIFIED"
        assert {path.name for path in bundle.iterdir()} == set(
            PRODUCER_ARTIFACTS
        )
        manifest = json.loads(
            (bundle / "manifest.json").read_text(encoding="utf-8")
        )
        assert set(manifest["artifacts"]) == (
            set(PRODUCER_ARTIFACTS) - {"manifest.json"}
        )
        for name, expected in manifest["artifacts"].items():
            assert hashlib.sha256(
                (bundle / name).read_bytes()
            ).hexdigest() == expected


def test_each_required_input_is_fail_closed_when_missing():
    for missing in REQUIRED_INPUTS:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            _write_raw_attempt(raw)
            (raw / missing).unlink()
            _expect_value_error(
                lambda: _assemble(raw, root / "bundle"),
                "required input",
            )


def test_each_required_input_is_fail_closed_when_truncated():
    for truncated in REQUIRED_INPUTS:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            _write_raw_attempt(raw)
            (raw / truncated).write_bytes(b"")
            _expect_value_error(
                lambda: _assemble(raw, root / "bundle"),
                "empty or truncated",
            )


def test_jsonl_requires_terminal_newline():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        raw = root / "raw"
        raw.mkdir()
        _write_raw_attempt(raw)
        path = raw / "performance_rows.jsonl"
        path.write_bytes(path.read_bytes().rstrip(b"\n"))
        _expect_value_error(
            lambda: _assemble(raw, root / "bundle"),
            "terminal newline",
        )


def test_no_tpot_benefit_still_assembles_serializable_no_go():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        raw = root / "raw"
        bundle = root / "bundle"
        raw.mkdir()
        _write_raw_attempt(raw)
        performance_path = raw / "performance_rows.jsonl"
        rows = [
            json.loads(line)
            for line in performance_path.read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        for row in rows:
            if row["arm"] == "graph":
                row["median_tpot_ms"] = 110.0
        _write_jsonl(performance_path, rows)
        result = _assemble(raw, bundle)
        summary = json.loads(
            (bundle / "summary.json").read_text(encoding="utf-8")
        )
        assert result["classification"] == "NO_GO_PERFORMANCE"
        assert summary["capture_amortization_tokens"] is None


def test_process_receipts_require_one_fresh_dynamic_port_per_arm():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        raw = root / "raw"
        raw.mkdir()
        _write_raw_attempt(raw)
        receipts_path = raw / "process_receipts.json"
        receipts = json.loads(
            receipts_path.read_text(encoding="utf-8")
        )
        receipts["case_rows"][1]["dist_port"] = (
            receipts["case_rows"][0]["dist_port"]
        )
        _write_json(receipts_path, receipts)
        _expect_value_error(
            lambda: _assemble(raw, root / "bundle"),
            "process receipts",
        )


def main() -> None:
    tests = (
        test_assembler_writes_complete_manifested_go_bundle,
        test_each_required_input_is_fail_closed_when_missing,
        test_each_required_input_is_fail_closed_when_truncated,
        test_jsonl_requires_terminal_newline,
        test_no_tpot_benefit_still_assembles_serializable_no_go,
        test_process_receipts_require_one_fresh_dynamic_port_per_arm,
    )
    for test in tests:
        test()
    print(f"{len(tests)} passed")


if __name__ == "__main__":
    main()
