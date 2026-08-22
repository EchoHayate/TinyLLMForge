#!/usr/bin/env python3
"""Producer-gate tests for graph-resident greedy-tail evidence."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory

try:
    import pytest
except ModuleNotFoundError:
    class _Raises:
        def __init__(self, expected, *, match=None):
            self.expected = expected
            self.match = match

        def __enter__(self):
            return self

        def __exit__(self, exception_type, exception, _traceback):
            if exception_type is None:
                raise AssertionError(
                    f"did not raise {self.expected!r}"
                )
            if not issubclass(exception_type, self.expected):
                return False
            if (
                self.match is not None
                and re.search(self.match, str(exception)) is None
            ):
                raise AssertionError(
                    f"{exception!r} does not match {self.match!r}"
                )
            return True

    class _PytestCompat:
        @staticmethod
        def raises(expected, *, match=None):
            return _Raises(expected, match=match)

    pytest = _PytestCompat()


REPO_ROOT = Path(__file__).resolve().parents[1]
if os.fspath(REPO_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPO_ROOT))

from tools.graph_resident_greedy_tail_gate import (
    classify,
    produce_gate,
)
from tools.profile_graph_resident_greedy_tail import (
    SOURCE_FILES,
    context_cases,
    policy_order,
    summarize_rows,
    write_float32_sidecar,
)


RUN_TAG = "20260822-qwen3-06b-graph-greedy-tail-fixture"
SOURCE_COMMIT = "a" * 40


def _greedy_summary(policy: str) -> dict:
    steps = {
        "legacy": 0,
        "host_greedy": 128,
        "graph_greedy": 1,
    }[policy]
    return {
        "eligible_steps": steps,
        "optimized_steps": steps,
        "avoided_temperature_h2d_bytes": steps * 4,
        "avoided_softmax_calls": steps,
        "avoided_gumbel_rng_calls": steps,
        "avoided_stochastic_divisions": steps * 2,
        "avoided_stochastic_argmax_calls": steps,
        "avoided_where_calls": steps,
        "fallback_counts": (
            {"disabled": 128}
            if policy == "legacy"
            else {}
        ),
    }


def _receipt() -> dict:
    return {
        "source_identity": {
            "data_ptr": 4096,
            "shape": [1, 1024],
            "stride": [1024, 1],
            "storage_offset": 0,
            "dtype": "torch.float16",
            "device": "cuda:0",
        },
        "graph_generation": 1,
        "rank": 0,
        "capture_duration_ns": 1_000_000,
        "allocated_delta_bytes": 400_000,
        "reserved_delta_bytes": 2_000_000,
        "retained_logits_bytes": 300_000,
        "retained_float32_bytes": 600_000,
        "retained_token_bytes": 8,
        "retained_static_bytes": 900_008,
    }


def _graph_summary(policy: str) -> dict:
    steps = 127 if policy == "graph_greedy" else 0
    return {
        "eligible_steps": steps,
        "captured_graphs": (
            1 if policy == "graph_greedy" else 0
        ),
        "replayed_steps": steps,
        "final_token_d2h_calls": steps,
        "avoided_external_compute_logits_calls": steps,
        "avoided_external_float32_conversions": steps,
        "avoided_external_argmax_calls": steps,
        "fallback_counts": {},
        "quarantine_reason": None,
        "capture_receipt": (
            _receipt()
            if policy == "graph_greedy"
            else None
        ),
    }


def make_fixture_rows(
    *,
    host_scale: float = 0.94,
    graph_scale: float = 0.88,
) -> list[dict]:
    rows = []
    shapes = (
        ("short", 256, 1_000_000),
        ("medium", 2048, 1_200_000),
        ("long", 8192, 1_400_000),
    )
    for repetition in range(5):
        for bucket, prompt_tokens, base_tpot in shapes:
            output_ids = [
                10 + ((index + repetition) % 100)
                for index in range(128)
            ]
            text_digest = hashlib.sha256(
                json.dumps(output_ids).encode("utf-8")
            ).hexdigest()
            for policy in (
                "legacy",
                "host_greedy",
                "graph_greedy",
            ):
                scale = {
                    "legacy": 1.0,
                    "host_greedy": host_scale,
                    "graph_greedy": graph_scale,
                }[policy]
                tpot = round(base_tpot * scale)
                e2e = 20_000_000 + tpot * 127
                graph_summary = _graph_summary(policy)
                receipt = graph_summary["capture_receipt"]
                rows.append({
                    "schema_version":
                        "graph-resident-greedy-tail.case.v1",
                    "run_tag": RUN_TAG,
                    "source_commit": SOURCE_COMMIT,
                    "policy": policy,
                    "repetition": repetition,
                    "context_bucket": bucket,
                    "prompt_tokens": prompt_tokens,
                    "generated_tokens": 128,
                    "output_token_ids": list(output_ids),
                    "output_text_sha256": text_digest,
                    "ttft_ns": 20_000_000,
                    "e2e_ns": e2e,
                    "tpot_samples_ns": [tpot] * 127,
                    "decode_host_ns":
                        [round(tpot * 0.85)] * 127,
                    "decode_cuda_ns":
                        [round(tpot * 0.70)] * 127,
                    "output_tokens_per_second": (
                        128 * 1_000_000_000 / e2e
                    ),
                    "cuda_peak_allocated_bytes":
                        1_000_000_000,
                    "cuda_peak_reserved_bytes":
                        1_200_000_000,
                    "graph_capture_duration_ns": (
                        receipt["capture_duration_ns"]
                        if receipt is not None
                        else 0
                    ),
                    "graph_allocated_delta_bytes": (
                        receipt["allocated_delta_bytes"]
                        if receipt is not None
                        else 0
                    ),
                    "graph_reserved_delta_bytes": (
                        receipt["reserved_delta_bytes"]
                        if receipt is not None
                        else 0
                    ),
                    "graph_retained_static_bytes": (
                        receipt["retained_static_bytes"]
                        if receipt is not None
                        else 0
                    ),
                    "greedy_fast_path_summary":
                        _greedy_summary(policy),
                    "graph_resident_greedy_tail_summary":
                        graph_summary,
                })
    return rows


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _write_correctness_rows(
    run_dir: Path,
    *,
    mutation=None,
) -> list[dict]:
    rows = []
    for bucket, prompt_tokens, generated_tokens in context_cases():
        output_ids = list(range(generated_tokens))
        text_digest = hashlib.sha256(
            json.dumps(output_ids).encode("utf-8")
        ).hexdigest()
        for policy in (
            "legacy",
            "host_greedy",
            "graph_greedy",
        ):
            for point in (
                "prefill-final",
                "decode-first",
                "decode-final",
            ):
                values = [1.0, 2.0, 5.0, 3.0]
                if mutation is not None:
                    values = mutation(
                        bucket,
                        policy,
                        point,
                        list(values),
                    )
                sidecar = write_float32_sidecar(
                    run_dir,
                    f"logits/{bucket}-{policy}-{point}.f32",
                    values,
                )
                rows.append({
                    "schema_version":
                        "graph-resident-greedy-tail.correctness.v1",
                    "run_tag": RUN_TAG,
                    "source_commit": SOURCE_COMMIT,
                    "policy": policy,
                    "context_bucket": bucket,
                    "prompt_tokens": prompt_tokens,
                    "generated_tokens": generated_tokens,
                    "sampling_point": point,
                    "output_token_ids": output_ids,
                    "output_text_sha256": text_digest,
                    "logits_path": sidecar["path"],
                    "logits_shape": [1, 4],
                    "logits_element_count":
                        sidecar["element_count"],
                    "logits_byte_length":
                        sidecar["byte_length"],
                    "logits_sha256": sidecar["sha256"],
                    "greedy_fast_path_summary":
                        _greedy_summary(policy),
                    "graph_resident_greedy_tail_summary":
                        _graph_summary(policy),
                })
    return rows


def _load_correctness_rows(run_dir: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (
            run_dir / "correctness_rows.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]


def write_fixture_bundle(
    root: Path,
    *,
    performance_rows: list[dict] | None = None,
    correctness_mutation=None,
) -> tuple[Path, Path]:
    performance_rows = (
        make_fixture_rows()
        if performance_rows is None
        else performance_rows
    )
    run_dir = root / "run"
    repo_root = root / "repo"
    run_dir.mkdir()
    for relative in SOURCE_FILES:
        path = repo_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"fixture source for {relative}\n",
            encoding="utf-8",
        )
    source_sha256 = {
        relative: hashlib.sha256(
            (repo_root / relative).read_bytes()
        ).hexdigest()
        for relative in SOURCE_FILES
    }
    correctness_rows = _write_correctness_rows(
        run_dir,
        mutation=correctness_mutation,
    )
    _write_json(
        run_dir / "source_manifest.json",
        {
            "schema_version":
                "graph-resident-greedy-tail.source.v1",
            "run_tag": RUN_TAG,
            "source_commit": SOURCE_COMMIT,
            "source_sha256": source_sha256,
        },
    )
    _write_json(
        run_dir / "workload_manifest.json",
        {
            "schema_version":
                "graph-resident-greedy-tail.workload.v1",
            "run_tag": RUN_TAG,
            "source_commit": SOURCE_COMMIT,
            "model": "/models/Qwen3-0.6B",
            "context_cases": [
                {
                    "context_bucket": bucket,
                    "prompt_tokens": prompt_tokens,
                    "generated_tokens": generated_tokens,
                }
                for bucket, prompt_tokens, generated_tokens
                in context_cases()
            ],
            "repetitions": 5,
            "warmup_repetitions": 2,
            "batch_size": 1,
            "temperature": 0.0,
            "ignore_eos": True,
            "gpu_memory_utilization": 0.5,
            "policy_flags": {
                "legacy": {
                    "zero_temperature_greedy_fast_path": False,
                    "graph_resident_greedy_tail": False,
                },
                "host_greedy": {
                    "zero_temperature_greedy_fast_path": True,
                    "graph_resident_greedy_tail": False,
                },
                "graph_greedy": {
                    "zero_temperature_greedy_fast_path": True,
                    "graph_resident_greedy_tail": True,
                },
            },
            "policy_order": {
                str(index): list(policy_order(index))
                for index in range(5)
            },
            "correctness_sampling_points": [
                "prefill-final",
                "decode-first",
                "decode-final",
            ],
        },
    )
    _write_jsonl(
        run_dir / "case_rows.jsonl",
        performance_rows,
    )
    _write_jsonl(
        run_dir / "correctness_rows.jsonl",
        correctness_rows,
    )
    summary = summarize_rows(
        performance_rows,
        expected_repetitions=5,
    )
    summary["correctness_row_count"] = 27
    _write_json(run_dir / "summary.json", summary)
    return run_dir, repo_root


def _classify_rows(
    run_dir: Path,
    rows: list[dict],
) -> str:
    return classify(
        rows,
        _load_correctness_rows(run_dir),
        run_dir=run_dir,
    )["classification"]


def test_classify_accepts_complete_go_fixture() -> None:
    with TemporaryDirectory() as temporary:
        run_dir, _repo = write_fixture_bundle(Path(temporary))
        result = classify(
            make_fixture_rows(),
            _load_correctness_rows(run_dir),
            run_dir=run_dir,
        )
        assert result["classification"] == (
            "GO_GRAPH_RESIDENT_GREEDY_TAIL"
        )
        assert result[
            "legacy_median_tpot_winning_bucket_count"
        ] == 3
        assert result["correctness"]["max_abs"] == 0.0
        assert result["correctness"]["argmax_equal"] is True
        assert result["graph_replay_complete"] is True
        assert result["cost"]["capture_duration_ns"]["max"] > 0
        assert result["cost"]["retained_static_bytes"]["max"] > 0


def test_classify_uses_fixed_failure_precedence() -> None:
    with TemporaryDirectory() as temporary:
        run_dir, _repo = write_fixture_bundle(
            Path(temporary),
            correctness_mutation=lambda bucket, policy, point, values: (
                [1.0, 2.0, 5.3, 3.0]
                if (
                    bucket == "short"
                    and policy == "graph_greedy"
                    and point == "prefill-final"
                )
                else values
            ),
        )
        assert _classify_rows(
            run_dir,
            make_fixture_rows(),
        ) == "NO_GO_CORRECTNESS"

    with TemporaryDirectory() as temporary:
        run_dir, _repo = write_fixture_bundle(Path(temporary))
        rows = make_fixture_rows()
        graph = next(
            row for row in rows
            if row["policy"] == "graph_greedy"
        )
        graph[
            "graph_resident_greedy_tail_summary"
        ]["replayed_steps"] = 126
        assert _classify_rows(
            run_dir,
            rows,
        ) == "NO_GO_GRAPH_REPLAY_INCOMPLETE"

    with TemporaryDirectory() as temporary:
        run_dir, _repo = write_fixture_bundle(Path(temporary))
        assert _classify_rows(
            run_dir,
            make_fixture_rows(
                host_scale=0.98,
                graph_scale=0.96,
            ),
        ) == "NO_GO_LEGACY_TPOT_MEDIAN"

    with TemporaryDirectory() as temporary:
        run_dir, _repo = write_fixture_bundle(Path(temporary))
        rows = make_fixture_rows()
        for row in rows:
            if row["policy"] == "graph_greedy":
                row["tpot_samples_ns"][-10:] = (
                    [row["tpot_samples_ns"][0] * 2] * 10
                )
        assert _classify_rows(
            run_dir,
            rows,
        ) == "NO_GO_LEGACY_TPOT_P95"

    with TemporaryDirectory() as temporary:
        run_dir, _repo = write_fixture_bundle(Path(temporary))
        assert _classify_rows(
            run_dir,
            make_fixture_rows(
                host_scale=0.91,
                graph_scale=0.90,
            ),
        ) == "NO_GO_HOST_GREEDY_INCREMENTAL"

    with TemporaryDirectory() as temporary:
        run_dir, _repo = write_fixture_bundle(Path(temporary))
        rows = make_fixture_rows()
        for row in rows:
            if row["policy"] == "graph_greedy":
                row["ttft_ns"] = 20_800_001
        assert _classify_rows(
            run_dir,
            rows,
        ) == "NO_GO_PROTECTED_REGRESSION"

    with TemporaryDirectory() as temporary:
        run_dir, _repo = write_fixture_bundle(Path(temporary))
        rows = make_fixture_rows()
        for row in rows:
            if row["policy"] == "graph_greedy":
                row["graph_capture_duration_ns"] = 0
                row[
                    "graph_resident_greedy_tail_summary"
                ]["capture_receipt"]["capture_duration_ns"] = 0
        assert _classify_rows(
            run_dir,
            rows,
        ) == "NO_GO_COST_INCOMPLETE"

    with TemporaryDirectory() as temporary:
        run_dir, _repo = write_fixture_bundle(Path(temporary))
        rows = make_fixture_rows()
        for row in rows:
            row["source_commit"] = "c" * 40
        assert _classify_rows(
            run_dir,
            rows,
        ) == "NO_GO_EVIDENCE_INCOMPLETE"


def test_producer_writes_complete_manifest() -> None:
    with TemporaryDirectory() as temporary:
        run_dir, repo_root = write_fixture_bundle(
            Path(temporary)
        )
        gate = produce_gate(run_dir, repo_root=repo_root)
        assert gate["classification"] == (
            "GO_GRAPH_RESIDENT_GREEDY_TAIL"
        )
        manifest = json.loads(
            (run_dir / "manifest.sha256").read_text(
                encoding="utf-8"
            )
        )
        assert len({
            name
            for name in manifest["artifacts"]
            if name.startswith("logits/")
        }) == 27
        assert {
            "case_rows.jsonl",
            "correctness_rows.jsonl",
            "source_manifest.json",
            "workload_manifest.json",
            "summary.json",
            "comparison.json",
            "gate.json",
        } <= set(manifest["artifacts"])


def test_producer_rejects_duplicate_stale_and_nonfinite_evidence() -> None:
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        run_dir, repo_root = write_fixture_bundle(root)
        rows = make_fixture_rows()
        rows[-1] = deepcopy(rows[0])
        _write_jsonl(run_dir / "case_rows.jsonl", rows)
        with pytest.raises(ValueError, match="duplicate case identity"):
            produce_gate(run_dir, repo_root=repo_root)

    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        run_dir, repo_root = write_fixture_bundle(root)
        source = json.loads(
            (run_dir / "source_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        source["source_sha256"][SOURCE_FILES[0]] = "0" * 64
        _write_json(run_dir / "source_manifest.json", source)
        with pytest.raises(ValueError, match="source digest mismatch"):
            produce_gate(run_dir, repo_root=repo_root)

    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        run_dir, repo_root = write_fixture_bundle(root)
        correctness = _load_correctness_rows(run_dir)
        correctness[0]["logits_sha256"] = "0" * 64
        _write_jsonl(
            run_dir / "correctness_rows.jsonl",
            correctness,
        )
        with pytest.raises(ValueError, match="SHA256 mismatch"):
            produce_gate(run_dir, repo_root=repo_root)

    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        run_dir, repo_root = write_fixture_bundle(root)
        rows = make_fixture_rows()
        rows[0]["ttft_ns"] = math.nan
        raw = "".join(
            json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for row in rows
        )
        (run_dir / "case_rows.jsonl").write_text(
            raw,
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="non-finite"):
            produce_gate(run_dir, repo_root=repo_root)


def main() -> None:
    test_classify_accepts_complete_go_fixture()
    test_classify_uses_fixed_failure_precedence()
    test_producer_writes_complete_manifest()
    test_producer_rejects_duplicate_stale_and_nonfinite_evidence()
    print("graph-resident greedy-tail gate tests passed")


if __name__ == "__main__":
    main()
