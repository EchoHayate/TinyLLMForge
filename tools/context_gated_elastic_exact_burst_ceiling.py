#!/usr/bin/env python3
"""Classify the source-bound elastic K16 ceiling probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from tools import profile_context_gated_elastic_exact_burst as profile


SCHEMA_VERSION = "context-gated-elastic-exact-burst.ceiling.v1"
VERIFICATION_SCHEMA_VERSION = (
    "context-gated-elastic-exact-burst.ceiling-verification.v1"
)
POLICIES = profile.POLICIES
CONTEXT_LENGTHS = profile.CONTEXT_LENGTHS
SAMPLING_POINTS = profile.SAMPLING_POINTS
CEILING_REPETITIONS = 3
PERFORMANCE_ROW_COUNT = (
    CEILING_REPETITIONS * len(POLICIES) * len(CONTEXT_LENGTHS)
)
CORRECTNESS_ROW_COUNT = (
    len(POLICIES) * len(CONTEXT_LENGTHS) * len(SAMPLING_POINTS)
)
MINIMUM_MEDIAN_TPOT_IMPROVEMENT_PCT = 1.5
MAXIMUM_K16_HOST_VISIBLE_GAP_NS = 40_000_000

CEILING_GO = "CEILING_GO"
NO_GO_INSUFFICIENT_INCREMENTAL_BENEFIT = (
    "NO_GO_INSUFFICIENT_INCREMENTAL_BENEFIT"
)
NO_GO_BURST_GAP = "NO_GO_BURST_GAP"
NO_GO_CORRECTNESS = "NO_GO_CORRECTNESS"
NO_GO_EVIDENCE_INCOMPLETE = "NO_GO_EVIDENCE_INCOMPLETE"

SOURCE_FILES = tuple(dict.fromkeys((
    *profile.SOURCE_FILES,
    "tools/context_gated_elastic_exact_burst_ceiling.py",
    "tools/test_context_gated_elastic_exact_burst_ceiling.py",
    "tools/run_context_gated_elastic_exact_burst_remote.py",
    "tools/test_run_context_gated_elastic_exact_burst_remote.py",
)))


def _reject_constant(value):
    raise ValueError(f"non-finite JSON value: {value}")


def read_json(path: Path):
    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        parse_constant=_reject_constant,
    )


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line, parse_constant=_reject_constant)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_source_manifest(
    *,
    source_root: Path,
    run_tag: str,
    source_commit: str,
) -> dict:
    root = Path(source_root).resolve()
    if not isinstance(run_tag, str) or not run_tag:
        raise ValueError("run tag is invalid")
    if (
        not isinstance(source_commit, str)
        or len(source_commit) != 40
        or any(
            character not in "0123456789abcdef"
            for character in source_commit
        )
    ):
        raise ValueError("source commit is invalid")
    hashes = {}
    for relative in SOURCE_FILES:
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError("ceiling source file is missing")
        hashes[relative] = sha256_file(path)
    return {
        "schema_version": SCHEMA_VERSION,
        "run_tag": run_tag,
        "source_commit": source_commit,
        "source_sha256": hashes,
    }


def expected_performance_identities(
) -> tuple[tuple[int, int, str], ...]:
    return profile.performance_identities(
        repetitions=CEILING_REPETITIONS,
    )


def expected_correctness_identities(
) -> tuple[tuple[int, str, str], ...]:
    return profile.correctness_identities()


def _inventory_status(
    performance_rows: list[dict],
    correctness_rows: list[dict],
) -> tuple[bool, str | None]:
    performance = [
        (
            row.get("repetition"),
            row.get("context_length"),
            row.get("policy"),
        )
        for row in performance_rows
        if isinstance(row, dict)
    ]
    correctness = [
        (
            row.get("context_length"),
            row.get("policy"),
            row.get("sampling_point"),
        )
        for row in correctness_rows
        if isinstance(row, dict)
    ]
    if (
        len(performance) != len(performance_rows)
        or len(correctness) != len(correctness_rows)
    ):
        return False, "row payload is not an object"
    if (
        len(performance) != len(set(performance))
        or len(correctness) != len(set(correctness))
    ):
        return False, "duplicate row identity"
    if set(performance) != set(expected_performance_identities()):
        return False, "performance row inventory is incomplete"
    if set(correctness) != set(expected_correctness_identities()):
        return False, "correctness row inventory is incomplete"
    return True, None


def _improvement_pct(control: float, candidate: float) -> float:
    if (
        not math.isfinite(control)
        or not math.isfinite(candidate)
        or control <= 0.0
        or candidate < 0.0
    ):
        raise ValueError("TPOT inputs must be finite and valid")
    return (control - candidate) / control * 100.0


def _selected_k16(row: dict) -> bool:
    summary = row["exact_greedy_decode_burst_summary"]
    return (
        summary["k16_acceptances"] > 0
        and summary["authorized_width_histogram"].get("16", 0) > 0
        and summary["per_width_commits"].get("16", 0) > 0
    )


def classify(metrics: dict) -> str:
    if metrics.get("evidence_complete") is not True:
        return NO_GO_EVIDENCE_INCOMPLETE
    if (
        metrics.get("correctness_exact") is not True
        or metrics.get("runtime_inventory_exact") is not True
        or metrics.get("k16_selected_at_256") is not True
        or metrics.get("k16_selected_at_2048") is not True
        or metrics.get("k16_absent_at_4096") is not True
        or metrics.get("k16_absent_at_8192") is not True
    ):
        return NO_GO_CORRECTNESS
    gap = metrics.get(
        "maximum_selected_k16_host_visible_gap_ns",
        math.inf,
    )
    if (
        isinstance(gap, bool)
        or not isinstance(gap, (int, float))
        or not math.isfinite(float(gap))
        or float(gap) > MAXIMUM_K16_HOST_VISIBLE_GAP_NS
    ):
        return NO_GO_BURST_GAP
    improvements = metrics.get(
        "eligible_context_median_tpot_improvement_pct"
    )
    if (
        not isinstance(improvements, dict)
        or set(improvements) != {"256", "2048"}
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in improvements.values()
        )
    ):
        return NO_GO_EVIDENCE_INCOMPLETE
    if max(float(value) for value in improvements.values()) < (
        MINIMUM_MEDIAN_TPOT_IMPROVEMENT_PCT
    ):
        return NO_GO_INSUFFICIENT_INCREMENTAL_BENEFIT
    return CEILING_GO


def summarize_evidence(
    performance_rows: list[dict],
    correctness_rows: list[dict],
    *,
    run_dir: Path,
) -> dict:
    complete, reason = _inventory_status(
        performance_rows,
        correctness_rows,
    )
    if not complete:
        metrics = {
            "schema_version": SCHEMA_VERSION,
            "evidence_complete": False,
            "evidence_error": reason,
            "performance_row_count": len(performance_rows),
            "correctness_row_count": len(correctness_rows),
        }
        metrics["classification"] = classify(metrics)
        return metrics
    try:
        profile_summary = profile.summarize_rows(
            performance_rows,
            expected_repetitions=CEILING_REPETITIONS,
        )
        validated_correctness = profile.validate_correctness_rows(
            correctness_rows,
            run_dir=run_dir,
        )
        validated_performance = [
            profile.validate_case_row(row)
            for row in performance_rows
        ]
    except (KeyError, TypeError, ValueError) as error:
        metrics = {
            "schema_version": SCHEMA_VERSION,
            "evidence_complete": True,
            "evidence_error": str(error),
            "performance_row_count": len(performance_rows),
            "correctness_row_count": len(correctness_rows),
            "correctness_exact": False,
            "runtime_inventory_exact": False,
            "k16_selected_at_256": False,
            "k16_selected_at_2048": False,
            "k16_absent_at_4096": False,
            "k16_absent_at_8192": False,
        }
        metrics["classification"] = classify(metrics)
        return metrics

    by_context_policy: dict[tuple[int, str], list[dict]] = {}
    for row in validated_performance:
        by_context_policy.setdefault(
            (row["context_length"], row["policy"]),
            [],
        ).append(row)
    improvements = {}
    for context_length in (256, 2048):
        control = statistics.median(
            row["amortized_tpot_median_ns"]
            for row in by_context_policy[
                (context_length, "fixed_k8")
            ]
        )
        candidate = statistics.median(
            row["amortized_tpot_median_ns"]
            for row in by_context_policy[
                (context_length, "context_gated_elastic_k16")
            ]
        )
        improvements[str(context_length)] = _improvement_pct(
            control,
            candidate,
        )

    candidate_rows = [
        row
        for row in validated_performance
        if row["policy"] == "context_gated_elastic_k16"
    ]
    selected_rows = [
        row for row in candidate_rows if _selected_k16(row)
    ]
    correctness_exact = (
        profile_summary["all_outputs_exact"] is True
        and len(validated_correctness) == CORRECTNESS_ROW_COUNT
    )
    runtime_inventory_exact = all(
        (
            summary["target_model_forwards"]
            == summary["graph_replays"]
            == summary["committed_tokens"]
            and summary["intermediate_token_d2h_calls"] == 0
            and summary["final_token_d2h_calls"]
            == summary["commits"]
            and summary["final_token_d2h_bytes"]
            == summary["committed_tokens"] * 8
        )
        for summary in (
            row["exact_greedy_decode_burst_summary"]
            for row in validated_performance
        )
    )
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "run_tag": validated_performance[0]["run_tag"],
        "source_commit": validated_performance[0]["source_commit"],
        "evidence_complete": True,
        "evidence_error": None,
        "performance_row_count": len(validated_performance),
        "correctness_row_count": len(validated_correctness),
        "correctness_exact": correctness_exact,
        "runtime_inventory_exact": runtime_inventory_exact,
        "k16_selected_at_256": all(
            _selected_k16(row)
            for row in by_context_policy[
                (256, "context_gated_elastic_k16")
            ]
        ),
        "k16_selected_at_2048": all(
            _selected_k16(row)
            for row in by_context_policy[
                (2048, "context_gated_elastic_k16")
            ]
        ),
        "k16_absent_at_4096": all(
            not _selected_k16(row)
            for row in by_context_policy[
                (4096, "context_gated_elastic_k16")
            ]
        ),
        "k16_absent_at_8192": all(
            not _selected_k16(row)
            for row in by_context_policy[
                (8192, "context_gated_elastic_k16")
            ]
        ),
        "eligible_context_median_tpot_improvement_pct":
            improvements,
        "maximum_selected_k16_host_visible_gap_ns": max(
            (
                row["maximum_host_visible_burst_gap_ns"]
                for row in selected_rows
            ),
            default=0,
        ),
    }
    metrics["classification"] = classify(metrics)
    return metrics


def _manifest_authority(run_dir: Path) -> tuple[dict, dict]:
    workload = read_json(run_dir / "workload_manifest.json")
    source = read_json(run_dir / "source_manifest.json")
    if (
        workload.get("repetitions") != CEILING_REPETITIONS
        or workload.get("performance_row_count")
        != PERFORMANCE_ROW_COUNT
        or workload.get("correctness_row_count")
        != CORRECTNESS_ROW_COUNT
    ):
        raise ValueError("workload manifest inventory mismatch")
    if (
        workload.get("run_tag") != source.get("run_tag")
        or workload.get("source_commit")
        != source.get("source_commit")
    ):
        raise ValueError("manifest source authority mismatch")
    hashes = source.get("source_sha256")
    if (
        not isinstance(hashes, dict)
        or not hashes
        or any(
            not isinstance(path, str)
            or not path
            or not isinstance(digest, str)
            or len(digest) != 64
            for path, digest in hashes.items()
        )
    ):
        raise ValueError("source manifest hashes are invalid")
    patch = run_dir / "source.patch"
    if not patch.is_file() or patch.read_bytes() != b"":
        raise ValueError("source patch must be empty")
    return workload, source


def _ceiling_source_authority(
    run_dir: Path,
    *,
    workload: dict,
) -> dict:
    manifest = read_json(run_dir / "ceiling_source_manifest.json")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("run_tag") != workload.get("run_tag")
        or manifest.get("source_commit")
        != workload.get("source_commit")
    ):
        raise ValueError("ceiling source manifest authority mismatch")
    hashes = manifest.get("source_sha256")
    if (
        not isinstance(hashes, dict)
        or set(hashes) != set(SOURCE_FILES)
        or any(
            not isinstance(digest, str)
            or len(digest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in digest
            )
            for digest in hashes.values()
        )
    ):
        raise ValueError("ceiling source manifest hashes are invalid")
    return manifest


def _verify_source_files(source: dict, source_root: Path) -> None:
    root = Path(source_root).resolve()
    for relative, expected in source["source_sha256"].items():
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError("source manifest path is invalid")
        if sha256_file(path) != expected:
            raise ValueError("source manifest hash mismatch")


def verify_artifact_directory(
    run_dir: Path,
    *,
    source_root: Path | None = None,
) -> dict:
    root = Path(run_dir)
    workload, source = _manifest_authority(root)
    ceiling_source = _ceiling_source_authority(
        root,
        workload=workload,
    )
    if source_root is not None:
        _verify_source_files(source, source_root)
        _verify_source_files(ceiling_source, source_root)
    performance = read_jsonl(root / "performance_rows.jsonl")
    correctness = read_jsonl(root / "correctness_rows.jsonl")
    reconstructed = summarize_evidence(
        performance,
        correctness,
        run_dir=root,
    )
    recorded_summary = read_json(root / "ceiling_summary.json")
    if recorded_summary != reconstructed:
        raise ValueError("recorded ceiling summary mismatch")
    expected_gate = {
        "schema_version": SCHEMA_VERSION,
        "run_tag": workload["run_tag"],
        "source_commit": workload["source_commit"],
        "classification": reconstructed["classification"],
    }
    if read_json(root / "ceiling_gate.json") != expected_gate:
        raise ValueError("recorded ceiling gate mismatch")
    expected_receipt = {
        **expected_gate,
        "performance_row_count": PERFORMANCE_ROW_COUNT,
        "correctness_row_count": CORRECTNESS_ROW_COUNT,
    }
    if read_json(root / "producer_receipt.json") != expected_receipt:
        raise ValueError("producer receipt mismatch")
    return {
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "verified": True,
        "run_tag": workload["run_tag"],
        "source_commit": workload["source_commit"],
        "classification": reconstructed["classification"],
        "performance_row_count": len(performance),
        "correctness_row_count": len(correctness),
    }


def produce_artifacts(
    run_dir: Path,
    *,
    source_root: Path | None = None,
) -> dict:
    root = Path(run_dir)
    workload, source = _manifest_authority(root)
    if source_root is None:
        raise ValueError("source root is required for artifact production")
    _verify_source_files(source, source_root)
    ceiling_source = build_source_manifest(
        source_root=source_root,
        run_tag=workload["run_tag"],
        source_commit=workload["source_commit"],
    )
    write_json(
        root / "ceiling_source_manifest.json",
        ceiling_source,
    )
    _verify_source_files(ceiling_source, source_root)
    summary = summarize_evidence(
        read_jsonl(root / "performance_rows.jsonl"),
        read_jsonl(root / "correctness_rows.jsonl"),
        run_dir=root,
    )
    write_json(root / "ceiling_summary.json", summary)
    if summary["classification"] == NO_GO_EVIDENCE_INCOMPLETE:
        raise ValueError(
            "incomplete evidence cannot produce a terminal ceiling gate"
        )
    gate = {
        "schema_version": SCHEMA_VERSION,
        "run_tag": workload["run_tag"],
        "source_commit": workload["source_commit"],
        "classification": summary["classification"],
    }
    receipt = {
        **gate,
        "performance_row_count": PERFORMANCE_ROW_COUNT,
        "correctness_row_count": CORRECTNESS_ROW_COUNT,
    }
    write_json(root / "ceiling_gate.json", gate)
    write_json(root / "producer_receipt.json", receipt)
    return receipt


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.verify_only:
        result = verify_artifact_directory(
            args.run_dir,
            source_root=args.source_root,
        )
    else:
        result = produce_artifacts(
            args.run_dir,
            source_root=args.source_root,
        )
    if args.output is not None:
        write_json(args.output, result)
    else:
        print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
