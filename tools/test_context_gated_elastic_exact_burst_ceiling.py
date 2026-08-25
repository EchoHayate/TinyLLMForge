from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from tools import context_gated_elastic_exact_burst_ceiling as ceiling
from tools.test_profile_context_gated_elastic_exact_burst import (
    RUN_TAG,
    SOURCE_COMMIT,
    _case_row,
    _summary,
)


def _valid_metrics() -> dict:
    return {
        "evidence_complete": True,
        "correctness_exact": True,
        "runtime_inventory_exact": True,
        "k16_selected_at_256": True,
        "k16_selected_at_2048": True,
        "k16_absent_at_4096": True,
        "k16_absent_at_8192": True,
        "eligible_context_median_tpot_improvement_pct": {
            "256": 1.5,
            "2048": 0.0,
        },
        "maximum_selected_k16_host_visible_gap_ns": 40_000_000,
    }


def test_ceiling_constants_and_exact_threshold_are_frozen() -> None:
    assert ceiling.POLICIES == (
        "fixed_k8",
        "context_gated_elastic_k16",
    )
    assert ceiling.CONTEXT_LENGTHS == (256, 2048, 4096, 8192)
    assert ceiling.CEILING_REPETITIONS == 3
    assert ceiling.PERFORMANCE_ROW_COUNT == 24
    assert ceiling.CORRECTNESS_ROW_COUNT == 32
    assert ceiling.MINIMUM_MEDIAN_TPOT_IMPROVEMENT_PCT == 1.5
    assert ceiling.MAXIMUM_K16_HOST_VISIBLE_GAP_NS == 40_000_000
    assert ceiling.classify(_valid_metrics()) == ceiling.CEILING_GO


@pytest.mark.parametrize(
    ("mutation", "expected"),
    (
        (
            {"evidence_complete": False},
            ceiling.NO_GO_EVIDENCE_INCOMPLETE,
        ),
        (
            {"correctness_exact": False},
            ceiling.NO_GO_CORRECTNESS,
        ),
        (
            {"runtime_inventory_exact": False},
            ceiling.NO_GO_CORRECTNESS,
        ),
        (
            {"k16_selected_at_2048": False},
            ceiling.NO_GO_CORRECTNESS,
        ),
        (
            {"k16_absent_at_4096": False},
            ceiling.NO_GO_CORRECTNESS,
        ),
        (
            {
                "maximum_selected_k16_host_visible_gap_ns":
                    40_000_001
            },
            ceiling.NO_GO_BURST_GAP,
        ),
        (
            {
                "eligible_context_median_tpot_improvement_pct": {
                    "256": 1.499999,
                    "2048": 1.499999,
                }
            },
            ceiling.NO_GO_INSUFFICIENT_INCREMENTAL_BENEFIT,
        ),
    ),
)
def test_each_terminal_ceiling_classification_is_independent(
    mutation: dict,
    expected: str,
) -> None:
    metrics = _valid_metrics()
    metrics.update(mutation)
    assert ceiling.classify(metrics) == expected


def test_incomplete_and_duplicate_inventory_are_not_performance_failures(
    monkeypatch,
    tmp_path: Path,
) -> None:
    performance = [
        {
            "repetition": repetition,
            "context_length": context,
            "policy": policy,
        }
        for repetition, context, policy
        in ceiling.expected_performance_identities()
    ]
    correctness = [
        {
            "context_length": context,
            "policy": policy,
            "sampling_point": point,
        }
        for context, policy, point
        in ceiling.expected_correctness_identities()
    ]
    monkeypatch.setattr(
        ceiling.profile,
        "summarize_rows",
        lambda _rows, **_kwargs: {
            "all_outputs_exact": True,
        },
    )
    monkeypatch.setattr(
        ceiling.profile,
        "validate_correctness_rows",
        lambda rows, **_kwargs: rows,
    )

    assert ceiling.summarize_evidence(
        performance[:-1],
        correctness,
        run_dir=tmp_path,
    )["classification"] == ceiling.NO_GO_EVIDENCE_INCOMPLETE

    duplicate = performance + [deepcopy(performance[0])]
    assert ceiling.summarize_evidence(
        duplicate,
        correctness,
        run_dir=tmp_path,
    )["classification"] == ceiling.NO_GO_EVIDENCE_INCOMPLETE


def test_raw_rows_reconstruct_a_complete_ceiling_go(tmp_path: Path) -> None:
    performance = []
    for repetition, context, policy in (
        ceiling.expected_performance_identities()
    ):
        row = _case_row(
            policy,
            context_length=context,
            repetition=repetition,
        )
        if (
            policy == "context_gated_elastic_k16"
            and context in (256, 2048)
        ):
            row["amortized_tpot_samples_ns"] = [980_000.0] * 127
            row["amortized_tpot_median_ns"] = 980_000.0
            row["amortized_tpot_p95_ns"] = 980_000.0
            row["amortized_tpot_p99_ns"] = 980_000.0
        performance.append(row)

    correctness = []
    for context, policy, point in (
        ceiling.expected_correctness_identities()
    ):
        summary = _summary(policy, context)
        summary["sampled_logit_d2h_calls"] = 3
        summary["capture_receipts"][0]["correctness_trace"] = True
        sidecar = ceiling.profile.write_float32_sidecar(
            tmp_path,
            f"logits/{context}-{policy}-{point}.f32",
            (1.0, 4.0, 3.0),
        )
        correctness.append({
            "schema_version": (
                "context-gated-elastic-exact-burst.correctness.v1"
            ),
            "run_tag": RUN_TAG,
            "source_commit": SOURCE_COMMIT,
            "policy": policy,
            "context_length": context,
            "generated_tokens": 128,
            "sampling_point": point,
            "prompt_sha256": "b" * 64,
            "output_token_ids": list(range(128)),
            "output_text_sha256": "e" * 64,
            "argmax_token_id": 1,
            "logits_path": sidecar["path"],
            "logits_shape": [1, 3],
            "logits_element_count": sidecar["element_count"],
            "logits_byte_length": sidecar["byte_length"],
            "logits_sha256": sidecar["sha256"],
            "correctness_trace": True,
            "exact_greedy_decode_burst_summary": summary,
        })

    result = ceiling.summarize_evidence(
        performance,
        correctness,
        run_dir=tmp_path,
    )

    assert result["classification"] == ceiling.CEILING_GO
    assert result["performance_row_count"] == 24
    assert result["correctness_row_count"] == 32
    assert (
        result["eligible_context_median_tpot_improvement_pct"]["256"]
        == pytest.approx(2.0)
    )
    assert result["maximum_selected_k16_host_visible_gap_ns"] == (
        4_000_000
    )


def test_ceiling_source_manifest_hashes_every_task_source(
    monkeypatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "tinyvllm").mkdir()
    (tmp_path / "tools").mkdir()
    (tmp_path / "tinyvllm" / "config.py").write_text(
        "CONFIG = True\n",
        encoding="utf-8",
    )
    (tmp_path / "tools" / "ceiling.py").write_text(
        "CEILING = True\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        ceiling,
        "SOURCE_FILES",
        ("tinyvllm/config.py", "tools/ceiling.py"),
    )

    manifest = ceiling.build_source_manifest(
        source_root=tmp_path,
        run_tag="source-r1",
        source_commit="a" * 40,
    )

    assert manifest["run_tag"] == "source-r1"
    assert manifest["source_commit"] == "a" * 40
    assert set(manifest["source_sha256"]) == {
        "tinyvllm/config.py",
        "tools/ceiling.py",
    }
    assert all(
        len(digest) == 64
        for digest in manifest["source_sha256"].values()
    )


def test_incomplete_evidence_never_writes_terminal_gate(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        ceiling,
        "_manifest_authority",
        lambda _root: (
            {
                "run_tag": "partial-r1",
                "source_commit": "a" * 40,
            },
            {"source_sha256": {"x": "b" * 64}},
        ),
    )
    monkeypatch.setattr(
        ceiling,
        "_verify_source_files",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        ceiling,
        "build_source_manifest",
        lambda **_kwargs: {
            "schema_version": ceiling.SCHEMA_VERSION,
            "run_tag": "partial-r1",
            "source_commit": "a" * 40,
            "source_sha256": {"x": "b" * 64},
        },
    )
    monkeypatch.setattr(ceiling, "read_jsonl", lambda _path: [])
    monkeypatch.setattr(
        ceiling,
        "summarize_evidence",
        lambda *_args, **_kwargs: {
            "schema_version": ceiling.SCHEMA_VERSION,
            "classification": ceiling.NO_GO_EVIDENCE_INCOMPLETE,
            "evidence_complete": False,
        },
    )

    with pytest.raises(ValueError, match="incomplete"):
        ceiling.produce_artifacts(
            tmp_path,
            source_root=tmp_path,
        )

    assert (tmp_path / "ceiling_summary.json").is_file()
    assert not (tmp_path / "ceiling_gate.json").exists()
    assert not (tmp_path / "producer_receipt.json").exists()


def test_artifact_verifier_reconstructs_raw_evidence_and_rejects_tamper(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        ceiling,
        "SOURCE_FILES",
        ("tools/context_gated_elastic_exact_burst_ceiling.py",),
    )
    performance = [
        {
            "repetition": repetition,
            "context_length": context,
            "policy": policy,
        }
        for repetition, context, policy
        in ceiling.expected_performance_identities()
    ]
    correctness = [
        {
            "context_length": context,
            "policy": policy,
            "sampling_point": point,
        }
        for context, policy, point
        in ceiling.expected_correctness_identities()
    ]
    metrics = _valid_metrics()
    metrics["classification"] = ceiling.CEILING_GO
    metrics["performance_row_count"] = ceiling.PERFORMANCE_ROW_COUNT
    metrics["correctness_row_count"] = ceiling.CORRECTNESS_ROW_COUNT
    monkeypatch.setattr(
        ceiling,
        "summarize_evidence",
        lambda *_args, **_kwargs: deepcopy(metrics),
    )
    (tmp_path / "performance_rows.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in performance),
        encoding="utf-8",
    )
    (tmp_path / "correctness_rows.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in correctness),
        encoding="utf-8",
    )
    (tmp_path / "workload_manifest.json").write_text(
        json.dumps({
            "run_tag": "ceiling-r1",
            "source_commit": "a" * 40,
            "repetitions": ceiling.CEILING_REPETITIONS,
            "performance_row_count": ceiling.PERFORMANCE_ROW_COUNT,
            "correctness_row_count": ceiling.CORRECTNESS_ROW_COUNT,
        }),
        encoding="utf-8",
    )
    (tmp_path / "source_manifest.json").write_text(
        json.dumps({
            "run_tag": "ceiling-r1",
            "source_commit": "a" * 40,
            "source_sha256": {"tinyvllm/config.py": "b" * 64},
        }),
        encoding="utf-8",
    )
    (tmp_path / "source.patch").write_bytes(b"")
    ceiling.write_json(
        tmp_path / "ceiling_source_manifest.json",
        {
            "schema_version": ceiling.SCHEMA_VERSION,
            "run_tag": "ceiling-r1",
            "source_commit": "a" * 40,
            "source_sha256": {
                "tools/context_gated_elastic_exact_burst_ceiling.py":
                    "c" * 64,
            },
        },
    )
    ceiling.write_json(tmp_path / "ceiling_summary.json", metrics)
    ceiling.write_json(
        tmp_path / "ceiling_gate.json",
        {
            "schema_version": ceiling.SCHEMA_VERSION,
            "classification": ceiling.CEILING_GO,
            "source_commit": "a" * 40,
            "run_tag": "ceiling-r1",
        },
    )
    ceiling.write_json(
        tmp_path / "producer_receipt.json",
        {
            "schema_version": ceiling.SCHEMA_VERSION,
            "classification": ceiling.CEILING_GO,
            "source_commit": "a" * 40,
            "run_tag": "ceiling-r1",
            "performance_row_count": ceiling.PERFORMANCE_ROW_COUNT,
            "correctness_row_count": ceiling.CORRECTNESS_ROW_COUNT,
        },
    )

    receipt = ceiling.verify_artifact_directory(tmp_path)
    assert receipt["verified"] is True
    assert receipt["classification"] == ceiling.CEILING_GO

    gate = json.loads((tmp_path / "ceiling_gate.json").read_text())
    gate["classification"] = ceiling.NO_GO_BURST_GAP
    ceiling.write_json(tmp_path / "ceiling_gate.json", gate)
    with pytest.raises(ValueError, match="gate"):
        ceiling.verify_artifact_directory(tmp_path)
