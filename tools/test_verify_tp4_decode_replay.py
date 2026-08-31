#!/usr/bin/env python3
"""Mutation tests for the independent TP4 decode replay verifier."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import tempfile

from assemble_tp4_decode_replay import MANIFEST_SCHEMA
import test_assemble_tp4_decode_replay as fixture
import verify_tp4_decode_replay as verifier_module
from verify_tp4_decode_replay import verify_bundle


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


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def _rewrite_manifest(root: Path) -> None:
    artifacts = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }
    _write_json(root / "manifest.json", {
        "schema_version": MANIFEST_SCHEMA,
        "artifacts": artifacts,
    })


def _bundle(root: Path) -> Path:
    raw = root / "raw"
    bundle = root / "final_bundle"
    raw.mkdir()
    fixture._write_raw_attempt(raw)
    fixture._assemble(raw, bundle)
    return bundle


def _mutate_json(root: Path, name: str, mutate) -> None:
    path = root / name
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    _write_json(path, payload)
    _rewrite_manifest(root)


def _mutate_jsonl(root: Path, name: str, mutate) -> None:
    path = root / name
    rows = _read_jsonl(path)
    mutate(rows)
    _write_jsonl(path, rows)
    _rewrite_manifest(root)


def _verify_mutation(mutation: str) -> dict:
    with tempfile.TemporaryDirectory() as directory:
        bundle = _bundle(Path(directory))
        if mutation == "manifest_sha":
            manifest = json.loads(
                (bundle / "manifest.json").read_text(encoding="utf-8")
            )
            manifest["artifacts"]["performance_rows.jsonl"] = "0" * 64
            _write_json(bundle / "manifest.json", manifest)
        elif mutation == "source_tree":
            _mutate_json(
                bundle,
                "source_identity.json",
                lambda row: row.__setitem__(
                    "source_tree_sha256",
                    "c" * 64,
                ),
            )
        elif mutation == "model_revision":
            _mutate_json(
                bundle,
                "source_identity.json",
                lambda row: row.__setitem__(
                    "model_revision",
                    "d" * 40,
                ),
            )
        elif mutation == "workload":
            _mutate_json(
                bundle,
                "workload_profile.json",
                lambda row: row["workloads"]["Q0"].__setitem__(
                    "prompt_tokens",
                    255,
                ),
            )
        elif mutation == "output_token":
            def mutate(rows):
                rows[0]["graph_outputs"][0][
                    "output_token_ids"
                ][0] += 1

            _mutate_jsonl(
                bundle,
                "correctness_rows.jsonl",
                mutate,
            )
            def mutate_request(rows):
                target = next(
                    row
                    for row in rows
                    if row["pair_id"] == "Q0__r0"
                    and row["arm"] == "graph"
                    and row["request_id"].endswith(":request-0")
                )
                target["output_token_ids"][0] += 1

            _mutate_jsonl(
                bundle,
                "request_rows.jsonl",
                mutate_request,
            )
        elif mutation == "dispatch":
            def mutate(rows):
                target = next(
                    row for row in rows
                    if row["arm"] == "graph" and row["rank"] == 3
                )
                target["dispatch"] = "eager"

            _mutate_jsonl(
                bundle,
                "rank_dispatch_events.jsonl",
                mutate,
            )
        elif mutation == "graph_identity":
            def mutate(rows):
                target = next(
                    row for row in rows
                    if row["arm"] == "graph" and row["rank"] == 3
                )
                target["graph_identity_sha256"] = "e" * 64

            _mutate_jsonl(
                bundle,
                "rank_dispatch_events.jsonl",
                mutate,
            )
        elif mutation == "collective":
            def mutate(rows):
                target = next(row for row in rows if row["rank"] == 3)
                target["collective_order_sha256"] = "f" * 64

            _mutate_jsonl(
                bundle,
                "rank_collective_events.jsonl",
                mutate,
            )
        elif mutation == "cleanup":
            def mutate(rows):
                rows[0]["process_group_destroyed"] = False

            _mutate_jsonl(
                bundle,
                "rank_lifecycle_rows.jsonl",
                mutate,
            )
        elif mutation == "coverage":
            def mutate(rows):
                for row in rows:
                    if row["arm"] == "graph":
                        row["dispatch"] = "eager"
                        row["graph_identity_sha256"] = None
                        row["cache_state"] = "observing"
                        row["fallback_reason"] = "cold_identity"
                        row["graph_replay_count"] = 0

            _mutate_jsonl(
                bundle,
                "rank_dispatch_events.jsonl",
                mutate,
            )
        elif mutation in {"throughput", "tpot", "ttft", "p99_e2e"}:
            field = {
                "throughput": "output_tokens_per_second",
                "tpot": "median_tpot_ms",
                "ttft": "ttft_ms",
                "p99_e2e": "p99_e2e_ms",
            }[mutation]

            def mutate(rows):
                for row in rows:
                    if row["arm"] == "graph":
                        row[field] = (
                            90.0
                            if mutation == "throughput"
                            else (
                                1100.0
                                if mutation == "p99_e2e"
                                else 110.0
                            )
                        )

            _mutate_jsonl(
                bundle,
                "performance_rows.jsonl",
                mutate,
            )
        elif mutation in {"allocated", "reserved"}:
            field = (
                "peak_allocated_bytes"
                if mutation == "allocated"
                else "peak_reserved_bytes"
            )

            def mutate(rows):
                for row in rows:
                    if row["arm"] == "graph":
                        row[field] += 1024 * 1024 * 1024

            _mutate_jsonl(bundle, "memory_rows.jsonl", mutate)
        elif mutation == "producer":
            _mutate_json(
                bundle,
                "producer_classification.json",
                lambda row: row.__setitem__(
                    "classification",
                    "NO_GO_PERFORMANCE",
                ),
            )
        elif mutation == "process_port":
            def mutate(rows):
                rows["case_rows"][1]["dist_port"] = (
                    rows["case_rows"][0]["dist_port"]
                )

            _mutate_json(
                bundle,
                "process_receipts.json",
                mutate,
            )
        else:
            raise AssertionError(f"unknown mutation: {mutation}")
        return verify_bundle(bundle)


def test_verifier_reconstructs_go_from_hash_bound_raw_rows():
    with tempfile.TemporaryDirectory() as directory:
        result = verify_bundle(_bundle(Path(directory)))
    assert result["classification"] == "GO_STAGE1_JUSTIFIED"
    assert result["failed_gates"] == []
    assert result["verified_hashes"] is True
    assert result["metrics"]["replay_coverage"] == 1.0


def test_verifier_does_not_import_the_producer_assembler():
    source = inspect.getsource(verifier_module)
    assert "import assemble_tp4_decode_replay" not in source
    assert "from assemble_tp4_decode_replay" not in source


def test_integrity_and_frozen_identity_mutations_are_incomplete():
    for mutation in (
        "manifest_sha",
        "source_tree",
        "model_revision",
        "workload",
        "producer",
        "process_port",
    ):
        result = _verify_mutation(mutation)
        assert result["classification"] == "INCOMPLETE", (
            mutation,
            result,
        )


def test_correctness_and_lifecycle_mutations_fail_closed():
    for mutation in (
        "output_token",
        "dispatch",
        "graph_identity",
        "collective",
        "cleanup",
    ):
        result = _verify_mutation(mutation)
        assert (
            result["classification"]
            == "NO_GO_CORRECTNESS_OR_LIFECYCLE"
        ), (mutation, result)


def test_replay_coverage_mutation_has_distinct_no_go():
    result = _verify_mutation("coverage")
    assert result["classification"] == "NO_GO_MECHANISM_NOT_EXERCISED"


def test_every_performance_and_cost_gate_is_reconstructed():
    for mutation in (
        "throughput",
        "tpot",
        "ttft",
        "p99_e2e",
        "allocated",
        "reserved",
    ):
        result = _verify_mutation(mutation)
        assert result["classification"] == "NO_GO_PERFORMANCE", (
            mutation,
            result,
        )


def main() -> None:
    tests = (
        test_verifier_reconstructs_go_from_hash_bound_raw_rows,
        test_verifier_does_not_import_the_producer_assembler,
        test_integrity_and_frozen_identity_mutations_are_incomplete,
        test_correctness_and_lifecycle_mutations_fail_closed,
        test_replay_coverage_mutation_has_distinct_no_go,
        test_every_performance_and_cost_gate_is_reconstructed,
    )
    for test in tests:
        test()
    print(f"{len(tests)} passed")


if __name__ == "__main__":
    main()
