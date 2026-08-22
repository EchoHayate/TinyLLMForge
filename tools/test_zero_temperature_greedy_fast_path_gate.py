#!/usr/bin/env python3
"""Producer-gate tests for zero-temperature greedy fast-path evidence."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if os.fspath(REPO_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPO_ROOT))

from tools.profile_zero_temperature_greedy_fast_path import (
    SOURCE_FILES,
    context_cases,
    summarize_rows,
    write_float32_sidecar,
)
from tools.zero_temperature_greedy_fast_path_gate import (
    classify,
    produce_gate,
)


RUN_TAG = "20260822-qwen3-06b-greedy-fast-fixture"
SOURCE_COMMIT = "a" * 40


def _fast_path_summary(policy: str) -> dict:
    steps = 128 if policy == "on" else 0
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
            if policy == "off"
            else {}
        ),
    }


def make_fixture_rows(on_scale: float = 0.90) -> list[dict]:
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
            for policy in ("off", "on"):
                scale = 1.0 if policy == "off" else on_scale
                tpot = round(base_tpot * scale)
                e2e = 20_000_000 + tpot * 127
                rows.append({
                    "schema_version":
                        "zero-temperature-greedy-fast-path.case.v1",
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
                    "decode_host_ns": [round(tpot * 0.85)] * 127,
                    "decode_cuda_ns": [round(tpot * 0.70)] * 127,
                    "output_tokens_per_second": (
                        128 * 1_000_000_000 / e2e
                    ),
                    "cuda_peak_allocated_bytes": 1_000_000_000,
                    "cuda_peak_reserved_bytes": 1_200_000_000,
                    "greedy_fast_path_summary":
                        _fast_path_summary(policy),
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
        for policy in ("off", "on"):
            for point_index, point in enumerate(
                ("prefill-final", "decode-first", "decode-final")
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
                        "zero-temperature-greedy-fast-path.correctness.v1",
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
                    "logits_byte_length": sidecar["byte_length"],
                    "logits_sha256": sidecar["sha256"],
                    "greedy_fast_path_summary":
                        _fast_path_summary(policy),
                })
    return rows


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
                "zero-temperature-greedy-fast-path.source.v1",
            "run_tag": RUN_TAG,
            "source_commit": SOURCE_COMMIT,
            "source_sha256": source_sha256,
        },
    )
    _write_json(
        run_dir / "workload_manifest.json",
        {
            "schema_version":
                "zero-temperature-greedy-fast-path.workload.v1",
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
            "policy_order": {
                str(index): (
                    ["off", "on"]
                    if index % 2 == 0
                    else ["on", "off"]
                )
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
    summary = summarize_rows(performance_rows)
    summary["correctness_row_count"] = 18
    _write_json(run_dir / "summary.json", summary)
    return run_dir, repo_root


def test_classify_accepts_complete_go_fixture() -> None:
    with TemporaryDirectory() as temporary:
        run_dir, _repo = write_fixture_bundle(Path(temporary))
        result = classify(
            make_fixture_rows(),
            json.loads(
                "["
                + ",".join(
                    line
                    for line in (
                        run_dir / "correctness_rows.jsonl"
                    ).read_text(encoding="utf-8").splitlines()
                )
                + "]"
            ),
            run_dir=run_dir,
        )
        assert result["classification"] == (
            "GO_ZERO_TEMPERATURE_GREEDY_FAST_PATH"
        )
        assert result["median_tpot_winning_bucket_count"] == 3
        assert result["correctness"]["max_abs"] == 0.0
        assert result["correctness"]["argmax_equal"] is True


@pytest.mark.parametrize(
    ("mutation", "classification"),
    [
        (
            lambda bucket, policy, point, values: (
                [1.0, 2.0, 5.3, 3.0]
                if (
                    bucket == "short"
                    and policy == "on"
                    and point == "prefill-final"
                )
                else values
            ),
            "NO_GO_CORRECTNESS",
        ),
        (
            lambda bucket, policy, point, values: (
                [1.0, 2.0, 4.8, 3.2]
                if (
                    bucket == "short"
                    and policy == "on"
                    and point == "prefill-final"
                )
                else values
            ),
            "NO_GO_CORRECTNESS",
        ),
        (
            lambda bucket, policy, point, values: (
                [1.0, 2.0, 3.0, 5.0]
                if (
                    bucket == "short"
                    and policy == "on"
                    and point == "prefill-final"
                )
                else values
            ),
            "NO_GO_CORRECTNESS",
        ),
    ],
)
def test_classify_rejects_logit_failures(
    mutation,
    classification,
) -> None:
    with TemporaryDirectory() as temporary:
        run_dir, _repo = write_fixture_bundle(
            Path(temporary),
            correctness_mutation=mutation,
        )
        correctness_rows = [
            json.loads(line)
            for line in (
                run_dir / "correctness_rows.jsonl"
            ).read_text(encoding="utf-8").splitlines()
        ]
        assert classify(
            make_fixture_rows(),
            correctness_rows,
            run_dir=run_dir,
        )["classification"] == classification


def test_classify_rejects_output_and_optimized_path_failures() -> None:
    with TemporaryDirectory() as temporary:
        run_dir, _repo = write_fixture_bundle(Path(temporary))
        correctness_rows = [
            json.loads(line)
            for line in (
                run_dir / "correctness_rows.jsonl"
            ).read_text(encoding="utf-8").splitlines()
        ]
        rows = make_fixture_rows()
        rows[1]["output_token_ids"][-1] += 1
        assert classify(
            rows,
            correctness_rows,
            run_dir=run_dir,
        )["classification"] == "NO_GO_CORRECTNESS"

        rows = make_fixture_rows()
        rows[1][
            "greedy_fast_path_summary"
        ]["optimized_steps"] = 127
        assert classify(
            rows,
            correctness_rows,
            run_dir=run_dir,
        )["classification"] == (
            "NO_GO_OPTIMIZED_PATH_INCOMPLETE"
        )


def test_classify_rejects_performance_and_protected_failures() -> None:
    with TemporaryDirectory() as temporary:
        run_dir, _repo = write_fixture_bundle(Path(temporary))
        correctness_rows = [
            json.loads(line)
            for line in (
                run_dir / "correctness_rows.jsonl"
            ).read_text(encoding="utf-8").splitlines()
        ]
        assert classify(
            make_fixture_rows(on_scale=1.0),
            correctness_rows,
            run_dir=run_dir,
        )["classification"] == "NO_GO_TPOT_MEDIAN"

        rows = make_fixture_rows()
        for row in rows:
            if row["policy"] == "on":
                row["ttft_ns"] = 20_800_001
        assert classify(
            rows,
            correctness_rows,
            run_dir=run_dir,
        )["classification"] == (
            "NO_GO_PROTECTED_REGRESSION"
        )


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
        correctness = [
            json.loads(line)
            for line in (
                run_dir / "correctness_rows.jsonl"
            ).read_text(encoding="utf-8").splitlines()
        ]
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
    for mutation in (
        lambda bucket, policy, point, values: (
            [1.0, 2.0, 5.3, 3.0]
            if (
                bucket == "short"
                and policy == "on"
                and point == "prefill-final"
            )
            else values
        ),
        lambda bucket, policy, point, values: (
            [1.0, 2.0, 4.8, 3.2]
            if (
                bucket == "short"
                and policy == "on"
                and point == "prefill-final"
            )
            else values
        ),
        lambda bucket, policy, point, values: (
            [1.0, 2.0, 3.0, 5.0]
            if (
                bucket == "short"
                and policy == "on"
                and point == "prefill-final"
            )
            else values
        ),
    ):
        test_classify_rejects_logit_failures(
            mutation,
            "NO_GO_CORRECTNESS",
        )
    test_classify_rejects_output_and_optimized_path_failures()
    test_classify_rejects_performance_and_protected_failures()
    test_producer_rejects_duplicate_stale_and_nonfinite_evidence()
    print("zero-temperature greedy gate tests passed")


if __name__ == "__main__":
    main()
