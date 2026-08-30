#!/usr/bin/env python3
"""Tests for the persistent-decode ceiling evidence producer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools import profile_lease_sealed_persistent_decode_ceiling as profile


def test_timing_inventory_is_five_repetitions_for_three_contexts():
    assert profile.build_timing_identities() == tuple(
        (repetition, context)
        for repetition in range(5)
        for context in (256, 2048, 8192)
    )


def test_structural_inventory_is_one_matched_case_per_context():
    assert profile.build_trace_identities() == (256, 2048, 8192)


def test_trace_label_has_complete_ordered_identity():
    assert profile.trace_label(
        attempt="run-a",
        workload="exact_greedy",
        repetition=0,
        context=2048,
        burst=7,
        logical_tokens=8,
    ) == (
        "persistent_decode_trace/"
        "attempt=run-a/workload=exact_greedy/repetition=0/"
        "context=2048/burst=7/logical_tokens=8"
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("attempt", ""),
        ("workload", "contains/slash"),
        ("repetition", -1),
        ("context", 0),
        ("burst", -1),
        ("logical_tokens", 0),
    ],
)
def test_trace_label_rejects_invalid_identity(field, value):
    arguments = {
        "attempt": "run-a",
        "workload": "exact_greedy",
        "repetition": 0,
        "context": 256,
        "burst": 0,
        "logical_tokens": 8,
    }
    arguments[field] = value

    with pytest.raises(ValueError):
        profile.trace_label(**arguments)


class _FakeLlm:
    def __init__(self):
        self.clear_count = 0
        self.exit_count = 0

    def clear_reusable_prefix_cache(self):
        self.clear_count += 1

    def reset_peak_memory_stats(self, *, timeout_s):
        assert timeout_s == 60.0

    def memory_snapshots(self, *, timeout_s):
        assert timeout_s == 60.0
        return [{"allocated_bytes": 100, "reserved_bytes": 200}]

    def exit(self):
        self.exit_count += 1


def _request_result(context):
    return {
        "output_token_ids": [context % 31] * 128,
        "output_text": f"output-{context}",
        "ttft_ns": 1_000_000,
        "e2e_ns": 255_000_000,
        "amortized_tpot_samples_ns": [2_000_000] * 127,
        "decode_host_ns": [],
        "decode_cuda_ns": [],
        "host_visible_burst_gaps_ns": [20_000] * 16,
    }


def _summary():
    return {
        "target_model_forwards": 127,
        "committed_tokens": 127,
        "fallback_counts": {},
        "failures": 0,
        "rollbacks": 0,
        "quarantine_reason": None,
        "capture_receipts": [],
    }


def test_timing_case_uses_uninstrumented_exact_path(monkeypatch):
    llm = _FakeLlm()
    calls = []
    monkeypatch.setattr(
        profile,
        "_construct_exact_llm",
        lambda **_kwargs: llm,
    )

    def fake_run_request(_llm, **kwargs):
        calls.append(kwargs)
        return _request_result(256)

    monkeypatch.setattr(profile, "_run_exact_request", fake_run_request)
    monkeypatch.setattr(
        profile,
        "_runner_summaries",
        lambda _llm: ({}, {}),
    )
    monkeypatch.setattr(
        profile,
        "_combined_summary",
        lambda _llm, _before: _summary(),
    )
    monkeypatch.setattr(
        profile,
        "_aggregate_memory",
        lambda _rows: {
            "cuda_peak_allocated_bytes": 100,
            "cuda_peak_reserved_bytes": 200,
        },
    )

    row = profile.run_timing_case(
        model="/model",
        run_tag="run-a",
        source_commit="a" * 40,
        source_tree_sha256="b" * 64,
        runtime_identity_sha256="c" * 64,
        workload_identity_sha256="d" * 64,
        repetition=3,
        prompt_tokens=256,
        generated_tokens=128,
        gpu_memory_utilization=0.5,
    )

    assert len(calls) == 3
    assert all(call["profile_label"] is None for call in calls)
    assert all(call["policy"] == "decode_burst_k8" for call in calls)
    assert calls[-1]["prompt"] == profile._make_prompt(256, offset=0)
    assert row["arm"] == "uninstrumented"
    assert row["tpot_median_ns"] == 2_000_000
    assert row["target_model_forwards"] == 127
    assert row["committed_tokens"] == 127
    assert llm.clear_count == 2
    assert llm.exit_count == 1


class _Range:
    def __init__(self, label, events):
        self.label = label
        self.events = events

    def __enter__(self):
        self.events.append(("enter", self.label))

    def __exit__(self, *_args):
        self.events.append(("exit", self.label))


class _FakeStructuralLlm:
    def __init__(self):
        self._step_index = 0
        self._finished = False
        self.last_step_observation = None
        self.tokenizer = type(
            "Tokenizer",
            (),
            {"decode": staticmethod(lambda tokens: f"decoded-{len(tokens)}")},
        )()
        self.model_runner = type(
            "Runner",
            (),
            {"exact_greedy_decode_burst_summary":
                staticmethod(_summary)},
        )()
        self.scheduler = type(
            "Scheduler",
            (),
            {"exact_greedy_decode_burst_summary":
                staticmethod(_summary)},
        )()
        self.output_ids = []

    def add_request(self, _prompt, _params):
        return None

    def is_finished(self):
        return self._finished

    def step(self, **kwargs):
        assert kwargs["completion_only"] is True
        emitted = 1 if self._step_index == 0 else min(
            8,
            128 - len(self.output_ids),
        )
        self.output_ids.extend(range(
            len(self.output_ids),
            len(self.output_ids) + emitted,
        ))
        self.last_step_observation = {
            "is_prefill": self._step_index == 0,
            "new_completion_tokens_by_seq": {1: [0] * emitted},
        }
        self._step_index += 1
        self._finished = len(self.output_ids) == 128
        outputs = (
            [(1, list(self.output_ids))]
            if self._finished
            else []
        )
        return outputs, emitted

    def exit(self):
        return None


def test_structural_request_ranges_only_decode_steps(monkeypatch):
    events = []
    clock = iter(range(0, 1_000_000, 1_000)).__next__
    llm = _FakeStructuralLlm()
    monkeypatch.setattr(
        profile,
        "_sampling_params",
        lambda _generated_tokens: object(),
    )
    monkeypatch.setattr(
        profile,
        "_cuda_synchronize",
        lambda: None,
    )

    result = profile._run_structural_request(
        llm,
        prompt=[1] * 256,
        generated_tokens=128,
        run_tag="run-a",
        context=256,
        range_factory=lambda label: _Range(label, events),
        clock_ns=clock,
    )

    entered = [
        label for event, label in events if event == "enter"
    ]
    assert len(entered) == 16
    assert all("logical_tokens=8" in label for label in entered[:-1])
    assert entered[-1].endswith("logical_tokens=7")
    assert result["burst_logical_tokens"] == [8] * 15 + [7]
    assert len(result["output_token_ids"]) == 128


def _write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_source_manifest_hashes_every_qualification_source(tmp_path):
    for relative in profile.SOURCE_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")

    manifest = profile.build_source_manifest(
        repo_root=tmp_path,
        source_commit="a" * 40,
        run_tag="run-a",
    )

    assert set(manifest["source_sha256"]) == set(profile.SOURCE_FILES)
    assert all(
        len(value) == 64
        for value in manifest["source_sha256"].values()
    )
    assert {
        "tools/verify_lease_sealed_persistent_decode_ceiling.py",
        "tools/test_verify_lease_sealed_persistent_decode_ceiling.py",
        "tools/run_lease_sealed_persistent_decode_ceiling_remote.py",
        "tools/test_run_lease_sealed_persistent_decode_ceiling_remote.py",
    }.issubset(manifest["source_sha256"])


def test_finalize_writes_raw_trace_inventory_and_manifest(
    monkeypatch,
    tmp_path,
):
    output = tmp_path / "output"
    output.mkdir()
    timing_path = output / "timing_rows.jsonl"
    structural_path = output / "structural_rows.jsonl"
    timing_path.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in profile.synthetic_timing_rows_for_test()
        ),
        encoding="utf-8",
    )
    structural_path.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in profile.synthetic_structural_rows_for_test()
        ),
        encoding="utf-8",
    )
    trace_paths = {}
    for context in profile.CONTEXT_LENGTHS:
        path = tmp_path / f"context-{context}.sqlite"
        path.write_bytes(f"sqlite-{context}".encode())
        trace_paths[context] = path

    monkeypatch.setattr(
        profile,
        "read_decode_trace",
        lambda _path: {"ranges": [{}], "kernels": [{}]},
    )
    monkeypatch.setattr(
        profile,
        "_trace_context_summary",
        lambda structural, _parsed: (
            {
                "context_length": structural["context_length"],
                "profiled_tpot_median_ns": 2_100_000,
                "profiled_tpot_p95_ns": 2_200_000,
                "output_token_ids": structural["output_token_ids"],
                "output_text_sha256": structural["output_text_sha256"],
                "transaction_count": 1,
                "logical_token_count": 127,
                "eligible_zero_cost_ns_per_token": 100_000,
                "candidate_cuda_duration_ns": 10_000_000,
                "total_kernel_duration_ns": 20_000_000,
                "classified_launch_ratio": 1.0,
                "classified_duration_ratio": 1.0,
                "segment_signatures": ["e" * 64],
                "target_model_forwards": 127,
                "committed_tokens": 127,
                "fallback_count": 0,
                "failure_count": 0,
                "rollback_count": 0,
                "quarantine_reason": None,
            },
            [],
            [{
                "context": structural["context_length"],
                "segment_id": 0,
            }],
        ),
    )
    monkeypatch.setattr(
        profile,
        "compute_ceiling",
        lambda _timing, _summary: {"classification": "NO_GO"},
    )
    for name in (
        "source_manifest.json",
        "runtime_manifest.json",
        "gpu_admission.json",
        "workload_manifest.json",
    ):
        _write_json(output / name, {"fixture": name})

    profile.finalize_evidence(
        timing_path=timing_path,
        structural_path=structural_path,
        trace_paths=trace_paths,
        output_dir=output,
    )

    inventory = json.loads(
        (output / "trace_inventory.json").read_text(encoding="utf-8")
    )
    assert "traces" not in inventory
    assert [row["context_length"] for row in inventory["raw_traces"]] == [
        256,
        2048,
        8192,
    ]
    for row in inventory["raw_traces"]:
        path = trace_paths[row["context_length"]]
        assert row["remote_path"] == str(path)
        assert row["byte_length"] == path.stat().st_size
        assert row["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()

    manifest = json.loads(
        (output / "manifest.json").read_text(encoding="utf-8")
    )
    assert {
        row["path"] for row in manifest["artifacts"]
    } == set(profile.MANIFEST_FILES)
    segments = [
        json.loads(line)
        for line in (output / "segment_rows.jsonl").read_text().splitlines()
    ]
    assert [row["segment_id"] for row in segments] == [0, 1, 2]


def test_finalize_rejects_structural_output_mismatch(tmp_path):
    timing = profile.synthetic_timing_rows_for_test()
    structural = profile.synthetic_structural_rows_for_test()
    structural[1]["output_token_ids"][-1] += 1
    timing_path = tmp_path / "timing.jsonl"
    structural_path = tmp_path / "structural.jsonl"
    timing_path.write_text(
        "".join(json.dumps(row) + "\n" for row in timing),
        encoding="utf-8",
    )
    structural_path.write_text(
        "".join(json.dumps(row) + "\n" for row in structural),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="output mismatch"):
        profile.finalize_evidence(
            timing_path=timing_path,
            structural_path=structural_path,
            trace_paths={},
            output_dir=tmp_path / "out",
        )


def test_summary_fields_preserve_normalized_failure_counters():
    assert profile._summary_fields({
        "target_model_forwards": 127,
        "committed_tokens": 127,
        "fallback_count": 2,
        "failure_count": 3,
        "rollback_count": 4,
        "quarantine_reason": "test",
    }) == {
        "target_model_forwards": 127,
        "committed_tokens": 127,
        "fallback_count": 2,
        "failure_count": 3,
        "rollback_count": 4,
        "quarantine_reason": "test",
    }
