from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import profile_exact_burst_octet_folded_graph as profile


RUN_TAG = "20260830-octet-folded-fixture"
SOURCE_COMMIT = "a" * 40
PATCH_SHA256 = "b" * 64


def _case_row(
    policy: str,
    *,
    context_length: int = 2048,
    repetition: int = 0,
) -> dict:
    one_token_launches = (
        profile.GENERATED_TOKENS - 1
        if policy == "one_token_graph"
        else (profile.GENERATED_TOKENS - 1) % 8
    )
    folded_launches = (
        0
        if policy == "one_token_graph"
        else (profile.GENERATED_TOKENS - 1) // 8
    )
    burst_commits = (
        (profile.GENERATED_TOKENS - 1 + 7) // 8
    )
    return {
        "schema_version": profile.CASE_SCHEMA_VERSION,
        "run_tag": RUN_TAG,
        "source_commit": SOURCE_COMMIT,
        "source_patch_sha256": PATCH_SHA256,
        "policy": policy,
        "repetition": repetition,
        "order_position": profile.policy_order(
            repetition,
            profile.CONTEXT_LENGTHS.index(context_length),
        ).index(policy),
        "context_length": context_length,
        "prompt_tokens": context_length,
        "generated_tokens": profile.GENERATED_TOKENS,
        "temperature": 0.0,
        "ignore_eos": True,
        "tensor_parallel_size": 1,
        "max_num_seqs": 1,
        "completion_only": True,
        "prompt_sha256": "c" * 64,
        "output_token_ids": list(range(profile.GENERATED_TOKENS)),
        "output_text_sha256": "d" * 64,
        "one_token_graph_identity_sha256": "e" * 64,
        "folded_graph_identity_sha256": (
            "f" * 64 if policy == "octet_folded_graph" else None
        ),
        "logical_forwards": 127,
        "logical_replays": 127,
        "one_token_cuda_graph_launches": one_token_launches,
        "folded_cuda_graph_launches": folded_launches,
        "token_d2h_calls": burst_commits,
        "token_d2h_bytes": 1_016,
        "capture_duration_ns": 2_000_000,
        "capture_allocated_delta_bytes": 1_000_000,
        "capture_reserved_delta_bytes": 2_000_000,
        "capture_retained_static_bytes": 3_000_000,
        "cuda_peak_allocated_bytes": 1_000_000_000,
        "cuda_peak_reserved_bytes": 1_100_000_000,
        "ttft_ns": 10_000_000,
        "e2e_ns": 140_000_000,
        "tpot_samples_ns": [1_000_000.0] * 127,
        "tpot_median_ns": 1_000_000.0,
        "tpot_p95_ns": 1_000_000.0,
        "tpot_p99_ns": 1_000_000.0,
        "output_tokens_per_second": 914.285714,
        "host_visible_burst_gaps_ns": (
            [4_000_000] * burst_commits
        ),
        "maximum_host_visible_burst_gap_ns": 4_000_000,
        "fallback_count": 0,
        "rollback_count": 0,
        "quarantine_reason": None,
    }


def test_frozen_workload_inventory_and_alternating_order() -> None:
    assert profile.POLICIES == (
        "one_token_graph",
        "octet_folded_graph",
    )
    assert profile.CONTEXT_LENGTHS == (256, 2048, 8192)
    assert profile.GENERATED_TOKENS == 128
    assert profile.REPETITIONS == 5
    assert profile.WARMUP_REPETITIONS == 2
    assert profile.SAMPLING_POINTS == (
        "prefill-final",
        "decode-first",
        "decode-middle",
        "decode-final",
    )
    assert len(
        profile.performance_identities(
            repetitions=profile.REPETITIONS
        )
    ) == 30
    assert len(profile.correctness_identities()) == 24
    for repetition in range(profile.REPETITIONS):
        for context_index in range(len(profile.CONTEXT_LENGTHS)):
            expected = (
                profile.POLICIES
                if (repetition + context_index) % 2 == 0
                else tuple(reversed(profile.POLICIES))
            )
            assert profile.policy_order(
                repetition,
                context_index,
            ) == expected


def test_manifest_binds_source_patch_inventory_and_execution() -> None:
    manifest = profile.build_workload_manifest(
        model="/models/Qwen3-0.6B",
        device="cuda:0",
        run_tag=RUN_TAG,
        source_commit=SOURCE_COMMIT,
        source_patch_sha256=PATCH_SHA256,
        gpu_memory_utilization=0.5,
        environment={"fixture": True},
    )
    assert manifest["contexts"] == [256, 2048, 8192]
    assert manifest["policies"] == list(profile.POLICIES)
    assert manifest["performance_row_count"] == 30
    assert manifest["correctness_row_count"] == 24
    assert manifest["execution_order"] == [
        list(profile.policy_order(0, index))
        for index in range(3)
    ]
    assert manifest["source_commit"] == SOURCE_COMMIT
    assert manifest["source_patch_sha256"] == PATCH_SHA256

    with pytest.raises(ValueError, match="frozen"):
        profile.build_workload_manifest(
            model="/models/Qwen3-0.6B",
            device="cuda:0",
            run_tag=RUN_TAG,
            source_commit=SOURCE_COMMIT,
            source_patch_sha256=PATCH_SHA256,
            gpu_memory_utilization=0.5,
            environment={"fixture": True},
            repetitions=profile.REPETITIONS - 1,
        )


def test_llm_arms_only_differ_by_folded_flag() -> None:
    calls = []

    def fake_llm(model, **kwargs):
        calls.append((model, kwargs))
        return SimpleNamespace()

    with patch.dict(
        sys.modules,
        {"tinyvllm": SimpleNamespace(LLM=fake_llm)},
    ):
        for policy in profile.POLICIES:
            profile._construct_llm(
                model="/models/Qwen3-0.6B",
                device="cuda:0",
                prompt_tokens=2048,
                generated_tokens=profile.GENERATED_TOKENS,
                gpu_memory_utilization=0.5,
                policy=policy,
            )

    control = calls[0][1]
    candidate = calls[1][1]
    differing = {
        key
        for key in set(control) | set(candidate)
        if control.get(key) != candidate.get(key)
    }
    assert differing == {
        "exact_greedy_decode_burst_octet_folded_graph"
    }
    assert control["exact_greedy_decode_burst"] is True
    assert control["exact_greedy_decode_burst_tokens"] == 8
    assert control["exact_greedy_decode_burst_elastic_k16"] is False


def test_case_row_requires_complete_exact_runtime_evidence() -> None:
    row = profile.validate_case_row(
        _case_row("octet_folded_graph")
    )
    assert row["logical_forwards"] == row["logical_replays"]
    assert row["folded_cuda_graph_launches"] == 15
    assert row["one_token_cuda_graph_launches"] == 7

    for field in (
        "source_patch_sha256",
        "one_token_graph_identity_sha256",
        "logical_forwards",
        "folded_cuda_graph_launches",
        "capture_reserved_delta_bytes",
        "cuda_peak_reserved_bytes",
        "tpot_p95_ns",
        "maximum_host_visible_burst_gap_ns",
    ):
        malformed = _case_row("octet_folded_graph")
        malformed.pop(field)
        with pytest.raises(ValueError, match="fields"):
            profile.validate_case_row(malformed)


def test_case_row_rejects_non_finite_and_counter_drift() -> None:
    malformed = _case_row("octet_folded_graph")
    malformed["tpot_median_ns"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        profile.validate_case_row(malformed)

    malformed = _case_row("octet_folded_graph")
    malformed["logical_replays"] = 126
    with pytest.raises(ValueError, match="logical"):
        profile.validate_case_row(malformed)

    malformed = _case_row("octet_folded_graph")
    malformed["folded_cuda_graph_launches"] = 14
    with pytest.raises(ValueError, match="physical"):
        profile.validate_case_row(malformed)

    malformed = _case_row("octet_folded_graph")
    malformed["token_d2h_bytes"] -= 8
    with pytest.raises(ValueError, match="D2H byte"):
        profile.validate_case_row(malformed)

    malformed = _case_row("octet_folded_graph")
    malformed["host_visible_burst_gaps_ns"].pop()
    with pytest.raises(ValueError, match="host-visible"):
        profile.validate_case_row(malformed)

    malformed = _case_row("octet_folded_graph")
    malformed["order_position"] = 1 - malformed["order_position"]
    with pytest.raises(ValueError, match="execution order"):
        profile.validate_case_row(malformed)


def _correctness_rows(run_dir: Path) -> list[dict]:
    rows = []
    for context_length in profile.CONTEXT_LENGTHS:
        for policy in profile.POLICIES:
            for point in profile.SAMPLING_POINTS:
                sidecar = profile.write_float32_sidecar(
                    run_dir,
                    (
                        f"logits/{context_length}-{policy}-"
                        f"{point}.f32"
                    ),
                    (0.25, 1.5, -0.5),
                )
                runtime = _case_row(
                    policy,
                    context_length=context_length,
                )
                rows.append({
                    "schema_version":
                        profile.CORRECTNESS_SCHEMA_VERSION,
                    "run_tag": RUN_TAG,
                    "source_commit": SOURCE_COMMIT,
                    "source_patch_sha256": PATCH_SHA256,
                    "policy": policy,
                    "context_length": context_length,
                    "generated_tokens": profile.GENERATED_TOKENS,
                    "sampling_point": point,
                    "prompt_sha256": "c" * 64,
                    "output_token_ids": list(
                        range(profile.GENERATED_TOKENS)
                    ),
                    "output_text_sha256": "d" * 64,
                    "argmax_token_id": 1,
                    "logits_path": sidecar["path"],
                    "logits_shape": [1, 3],
                    "logits_element_count": sidecar[
                        "element_count"
                    ],
                    "logits_byte_length": sidecar["byte_length"],
                    "logits_sha256": sidecar["sha256"],
                    "one_token_graph_identity_sha256": runtime[
                        "one_token_graph_identity_sha256"
                    ],
                    "folded_graph_identity_sha256": runtime[
                        "folded_graph_identity_sha256"
                    ],
                    "logical_forwards": runtime[
                        "logical_forwards"
                    ],
                    "logical_replays": runtime["logical_replays"],
                    "one_token_cuda_graph_launches": runtime[
                        "one_token_cuda_graph_launches"
                    ],
                    "folded_cuda_graph_launches": runtime[
                        "folded_cuda_graph_launches"
                    ],
                    "token_d2h_calls": runtime["token_d2h_calls"],
                    "token_d2h_bytes": runtime["token_d2h_bytes"],
                    "fallback_count": runtime["fallback_count"],
                    "rollback_count": runtime["rollback_count"],
                    "quarantine_reason": runtime[
                        "quarantine_reason"
                    ],
                    "correctness_trace": True,
                })
    return rows


def test_correctness_rows_bind_sidecars_source_and_exact_pairs(
    tmp_path: Path,
) -> None:
    rows = _correctness_rows(tmp_path)
    validated = profile.validate_correctness_rows(
        rows,
        run_dir=tmp_path,
    )
    assert len(validated) == len(profile.correctness_identities())

    source_drift = [dict(row) for row in rows]
    source_drift[-1]["source_patch_sha256"] = "9" * 64
    with pytest.raises(ValueError, match="source identity"):
        profile.validate_correctness_rows(
            source_drift,
            run_dir=tmp_path,
        )

    mismatched = [dict(row) for row in rows]
    candidate = next(
        row
        for row in mismatched
        if row["policy"] == "octet_folded_graph"
        and row["context_length"] == 2048
        and row["sampling_point"] == "decode-middle"
    )
    sidecar = profile.write_float32_sidecar(
        tmp_path,
        "logits/mismatched.f32",
        (2.0, 1.5, -0.5),
    )
    candidate.update({
        "argmax_token_id": 0,
        "logits_path": sidecar["path"],
        "logits_element_count": sidecar["element_count"],
        "logits_byte_length": sidecar["byte_length"],
        "logits_sha256": sidecar["sha256"],
    })
    with pytest.raises(ValueError, match="policy mismatch"):
        profile.validate_correctness_rows(
            mismatched,
            run_dir=tmp_path,
        )


def test_correctness_sampling_covers_all_frozen_points() -> None:
    assert profile._correctness_sampled_logit_ordinals(
        context_length=2048,
    ) == (0, 6, 7)
    assert profile._correctness_trace_for_step(
        context_length=2048,
        emitted_total=1,
    ) is True
    assert profile._correctness_trace_for_step(
        context_length=2048,
        emitted_total=9,
    ) is False


def test_main_writes_performance_correctness_and_ceiling_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output_dir = tmp_path / "run"

    def fake_run_case(**kwargs):
        return _case_row(
            kwargs["policy"],
            context_length=kwargs["context_length"],
            repetition=kwargs["repetition"],
        )

    def fake_correctness_probe(**kwargs):
        return [
            row
            for row in _correctness_rows(kwargs["run_dir"])
            if row["context_length"] == kwargs["context_length"]
            and row["policy"] == kwargs["policy"]
        ]

    monkeypatch.setattr(profile, "run_case", fake_run_case)
    monkeypatch.setattr(
        profile,
        "run_correctness_probe",
        fake_correctness_probe,
    )
    monkeypatch.setattr(
        profile,
        "runtime_environment_manifest",
        lambda: {"fixture": True},
    )

    assert profile.main([
        "--model",
        "/models/Qwen3-0.6B",
        "--output-dir",
        str(output_dir),
        "--source-commit",
        SOURCE_COMMIT,
        "--source-patch-sha256",
        PATCH_SHA256,
        "--run-tag",
        RUN_TAG,
    ]) == 0

    assert len(
        (output_dir / "performance_rows.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ) == 30
    assert len(
        (output_dir / "correctness_rows.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ) == 24
    ceiling = json.loads(
        (output_dir / "ceiling.json").read_text(encoding="utf-8")
    )
    assert ceiling["classification"] == "NO_GO_CEILING"
    assert ceiling["performance_row_count"] == 30
    assert ceiling["correctness_row_count"] == 24
