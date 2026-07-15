"""Dependency-light tests for the SAM drafter canonical gate."""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from pathlib import Path

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_GATE_PATH = os.path.join(_THIS_DIR, "sam_drafter_gate.py")
_SPEC = importlib.util.spec_from_file_location(
    "sam_drafter_gate_under_test",
    _GATE_PATH,
)
gate = importlib.util.module_from_spec(_SPEC)
sys.modules["sam_drafter_gate_under_test"] = gate
_SPEC.loader.exec_module(gate)


def _sam_events(spec: dict, prompt_tokens: int, output_tokens: int) -> list[dict]:
    events = []
    if spec["policy"] == "sam_match_aware":
        proposals = [(0, 0, 0), (4, 4, 0), (8, 8, 8), (16, 16, 10)]
        proposals.extend((4, 4, 3 if index < 4 else 2) for index in range(11))
    else:
        proposals = [(16, 16, 14)] * 14
    for step, (selected_k, proposed, accepted) in enumerate(proposals):
        metadata = {
            "selected_k": selected_k,
            "match_length": 0 if selected_k == 0 else selected_k,
            "continuation_region": (
                "prompt" if step % 2 == 0 else "generated"
            ),
        }
        events.append({
            "run_key": spec["run_key"],
            "policy": spec["policy"],
            "prompt_name": spec["prompt_name"],
            "prompt_class": spec["prompt_class"],
            "repetition": spec["repetition"],
            "event_type": "proposal",
            "step": step,
            "candidate_seq_id": 1,
            "selected_k": selected_k,
            "proposed_tokens": proposed,
            "draft_metadata": metadata,
            "runtime_mutation": False,
            "profiler_owned": True,
        })
        if proposed == 0:
            events.append({
                **events[-1],
                "event_type": "bypass",
                "accepted_count": 0,
            })
        else:
            events.append({
                **events[-1],
                "event_type": "verify",
                "accepted_count": accepted,
                "wasted_draft_tokens": proposed - accepted,
            })
    events.append({
        "run_key": spec["run_key"],
        "policy": spec["policy"],
        "prompt_name": spec["prompt_name"],
        "prompt_class": spec["prompt_class"],
        "repetition": spec["repetition"],
        "event_type": "index_integrity",
        "step": len(proposals),
        "candidate_seq_id": 1,
        "index_token_count": prompt_tokens + output_tokens,
        "history_match": True,
        "runtime_mutation": False,
        "profiler_owned": True,
    })
    return events


def _synthetic_complete_gate_rows():
    manifest = gate.build_manifest(
        repetitions=7,
        base_seed=20260715,
        source_commit="synthetic",
        source_dirty=False,
        model_path="/models/Qwen3-0.6B",
        model_identifier="Qwen3-0.6B",
        host="synthetic",
        python_bin="python3",
    )
    tps = {
        "baseline": 100.0,
        "ngram_fixed_k4": 108.0,
        "ngram_adaptive": 106.0,
        "sam_fixed_k16": 111.0,
        "sam_match_aware": 112.0,
    }
    rows = []
    events = []
    for spec in manifest["run_specs"]:
        output_tokens = 100
        prompt_tokens = 32
        policy = spec["policy"]
        if policy == "baseline":
            verify_attempts = drafted = accepted = waste = zero = 0
            selected = {}
            bypass = 0
        elif policy == "ngram_fixed_k4":
            verify_attempts, drafted, accepted, waste, zero = 20, 100, 60, 40, 2
            selected = {"4": 20}
            bypass = 0
        elif policy == "ngram_adaptive":
            verify_attempts, drafted, accepted, waste, zero = 18, 90, 56, 34, 2
            selected = {"1": 3, "2": 7, "4": 8}
            bypass = 0
        else:
            run_events = _sam_events(spec, prompt_tokens, output_tokens)
            events.extend(run_events)
            proposals = [
                event for event in run_events
                if event["event_type"] == "proposal"
            ]
            verifies = [
                event for event in run_events
                if event["event_type"] == "verify"
            ]
            verify_attempts = len(verifies)
            drafted = sum(event["proposed_tokens"] for event in proposals)
            accepted = sum(event["accepted_count"] for event in verifies)
            waste = drafted - accepted
            zero = sum(event["accepted_count"] == 0 for event in verifies)
            selected = {
                str(level): sum(
                    event["selected_k"] == level for event in proposals
                )
                for level in (0, 4, 8, 16)
            }
            bypass = sum(
                event["event_type"] == "bypass" for event in run_events
            )
        rows.append({
            **spec,
            "source_commit": "synthetic",
            "source_dirty": False,
            "model_identifier": "Qwen3-0.6B",
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "output_token_ids": [11, 12, 13],
            "output_token_sha256": gate.sha256_json([11, 12, 13]),
            "elapsed_s": output_tokens / tps[policy],
            "output_tokens_per_s": tps[policy],
            "proposal_events": sum(selected.values()),
            "verify_attempts": verify_attempts,
            "no_draft_positions": bypass,
            "drafted_tokens": drafted,
            "accepted_tokens": accepted,
            "wasted_draft_tokens": waste,
            "zero_accept_events": zero,
            "zero_accept_verify_ms": float(zero),
            "selected_k_counts": selected,
            "sam_build_ms": 0.1 if policy.startswith("sam_") else 0.0,
            "sam_extension_ms": 0.2 if policy.startswith("sam_") else 0.0,
            "sam_lookup_ms": 0.3 if policy.startswith("sam_") else 0.0,
            "sam_state_count": 128 if policy.startswith("sam_") else 0,
            "sam_indexed_tokens": (
                prompt_tokens + output_tokens
                if policy.startswith("sam_") else 0
            ),
            "sam_bypass_count": bypass,
            "runtime_mutation": False,
            "profiler_owned": True,
            "profiler_gate_pass": True,
            "profiler_gate_fail_reasons": [],
            "process": {
                "returncode": 0,
                "tinyvllm_dist_port": 20000 + spec["global_order"] * 2,
                "master_port": 20001 + spec["global_order"] * 2,
            },
        })
    return manifest, rows, events


def test_prompt_bank_has_five_stable_classes():
    assert [item["name"] for item in gate.PROMPT_BANK] == [
        "natural_prose",
        "structured_code_like",
        "repeated_long_context",
        "transition_heavy",
        "prompt_copy_retrieval",
    ]
    assert {item["workload_class"] for item in gate.PROMPT_BANK} == {
        "natural",
        "structured",
        "high_repeat",
        "transition_heavy",
        "prompt_copy",
    }
    for prompt in gate.PROMPT_BANK:
        assert prompt["prompt_sha256"] == gate.sha256_text(prompt["prompt"])


def test_run_specs_are_175_unique_rows_for_canonical():
    specs = gate.build_run_specs(repetitions=7, base_seed=20260715)
    assert len(specs) == 175
    assert len({item["run_key"] for item in specs}) == 175
    assert {item["policy"] for item in specs} == {
        "baseline",
        "ngram_fixed_k4",
        "ngram_adaptive",
        "sam_fixed_k16",
        "sam_match_aware",
    }
    assert all(item["max_num_seqs"] == 1 for item in specs)


def test_required_upload_paths_cover_all_runtime_imports():
    assert gate.REQUIRED_UPLOAD_PATHS == (
        "tinyvllm",
        "tools/draft_model_schema.py",
        "tools/profile_ngram_commit.py",
        "tools/sam_drafter_gate.py",
    )


def test_profiler_commands_are_policy_specific():
    specs = gate.build_run_specs(repetitions=1, base_seed=7)
    prompt = gate.PROMPT_BANK[0]
    commands = {
        spec["policy"]: gate._profiler_command(
            spec,
            prompt,
            "python3",
            "/model",
            gate.Path("/tmp/process.json"),
        )
        for spec in specs
        if spec["prompt_name"] == prompt["name"]
    }
    assert "--draft-source" not in commands["baseline"]
    for policy in (
        "ngram_fixed_k4",
        "ngram_adaptive",
        "sam_fixed_k16",
        "sam_match_aware",
    ):
        assert "--allow-zero-accept" in commands[policy]
    assert commands["sam_match_aware"][
        commands["sam_match_aware"].index("--draft-source") + 1
    ] == "sam"


def test_complete_175_row_fixture_is_go():
    manifest, rows, events = _synthetic_complete_gate_rows()
    summary = gate.summarize_rows(manifest, rows, events)
    assert summary["decision"] == "GO"
    assert summary["observed_rows"] == 175
    assert summary["correctness_pass"] is True
    assert summary["trace_reconciliation_pass"] is True
    assert summary["policy_exercise_pass"] is True


def test_missing_or_failed_evidence_is_incomplete_not_no_go():
    manifest, rows, events = _synthetic_complete_gate_rows()
    assert gate.summarize_rows(manifest, rows[:-1], events)["decision"] == "INCOMPLETE"
    rows[-1]["process"]["returncode"] = 1
    rows[-1]["elapsed_s"] = None
    rows[-1]["output_tokens_per_s"] = None
    assert gate.summarize_rows(manifest, rows, events)["decision"] == "INCOMPLETE"


def test_output_mismatch_is_incomplete():
    manifest, rows, events = _synthetic_complete_gate_rows()
    candidate = next(
        row for row in rows if row["policy"] == "sam_match_aware"
    )
    candidate["output_token_ids"] = [999]
    assert gate.summarize_rows(manifest, rows, events)["decision"] == "INCOMPLETE"


def test_performance_failure_is_no_go_only_after_evidence_passes():
    manifest, rows, events = _synthetic_complete_gate_rows()
    for row in rows:
        if row["policy"] == "sam_match_aware":
            row["output_tokens_per_s"] = 105.0
            row["elapsed_s"] = row["output_tokens"] / 105.0
    summary = gate.summarize_rows(manifest, rows, events)
    assert summary["decision"] == "NO_GO"
    assert summary["correctness_pass"] is True


def test_speedup_is_median_of_paired_ratios():
    manifest, rows, events = _synthetic_complete_gate_rows()
    pairs = [
        row for row in rows
        if row["prompt_name"] == "natural_prose"
        and row["repetition"] in (0, 1)
        and row["policy"] in ("baseline", "sam_match_aware")
    ]
    for row in pairs:
        if row["repetition"] == 0 and row["policy"] == "baseline":
            row["output_tokens_per_s"] = 10.0
        elif row["repetition"] == 0:
            row["output_tokens_per_s"] = 12.0
        elif row["policy"] == "baseline":
            row["output_tokens_per_s"] = 100.0
        else:
            row["output_tokens_per_s"] = 100.0
        row["elapsed_s"] = row["output_tokens"] / row["output_tokens_per_s"]
    summary = gate.summarize_rows(manifest, rows, events)
    natural_pairs = summary["paired_speedups"]["sam_vs_baseline"][
        "natural_prose"
    ]
    assert any(abs(value - 0.20) < 1e-12 for value in natural_pairs)
    assert 0.0 in natural_pairs


def test_zero_positive_reference_for_required_reduction_is_incomplete():
    manifest, rows, events = _synthetic_complete_gate_rows()
    for row in rows:
        if row["policy"] == "ngram_fixed_k4":
            row["verify_attempts"] = 0
    assert gate.summarize_rows(manifest, rows, events)["decision"] == "INCOMPLETE"


def test_missing_each_required_policy_branch_is_incomplete():
    for field, value in (
        ("selected_k", 0),
        ("selected_k", 4),
        ("selected_k", 8),
        ("selected_k", 16),
        ("continuation_region", "prompt"),
        ("continuation_region", "generated"),
        ("accepted_count", 0),
    ):
        manifest, rows, events = _synthetic_complete_gate_rows()
        filtered = [
            event for event in events
            if event.get(field, event.get("draft_metadata", {}).get(field))
            != value
        ]
        assert gate.summarize_rows(manifest, rows, filtered)["decision"] == "INCOMPLETE"


def test_verify_artifacts_recomputes_summary_report_and_hashes():
    manifest, rows, events = _synthetic_complete_gate_rows()
    summary = gate.summarize_rows(manifest, rows, events)
    with tempfile.TemporaryDirectory() as temporary:
        out_dir = Path(temporary)
        gate._write_canonical_artifacts(
            out_dir,
            manifest,
            rows,
            events,
            summary,
        )
        assert gate.verify_artifacts(out_dir)["decision"] == "GO"
        (out_dir / "report.md").write_text("tampered\n")
        try:
            gate.verify_artifacts(out_dir)
        except ValueError as exc:
            assert "report.md" in str(exc)
        else:
            raise AssertionError("tampered report accepted")


def test_resume_rejects_each_compatibility_mismatch():
    manifest, rows, _ = _synthetic_complete_gate_rows()
    spec = manifest["run_specs"][0]
    row = rows[0]
    assert gate._row_is_resumable(manifest, spec, row) is True
    mutations = (
        ("source_commit", "other"),
        ("source_dirty", True),
        ("model_identifier", "other-model"),
        ("prompt_sha256", "bad-hash"),
        ("policy", "sam_match_aware"),
        ("repetition", 99),
    )
    for field, value in mutations:
        changed = {**row, field: value}
        assert gate._row_is_resumable(manifest, spec, changed) is False
    changed = {**row, "process": {**row["process"], "returncode": 1}}
    assert gate._row_is_resumable(manifest, spec, changed) is False
    changed = {**row, "profiler_gate_pass": False}
    assert gate._row_is_resumable(manifest, spec, changed) is False
    changed = {**row, "elapsed_s": float("nan")}
    assert gate._row_is_resumable(manifest, spec, changed) is False


def test_remote_runner_uses_exact_host_python_model_and_isolation():
    source = (
        Path(_REPO_ROOT)
        / "tools"
        / "run_sam_drafter_gate_remote.sh"
    ).read_text()
    assert "sitian@10.232.195.203" in source
    assert "/data00/home/sitian/sitian-workspace01/tllm/env/bin/python" in source
    assert (
        "/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B"
        in source
    )
    assert "TINYVLLM_DIST_PORT" not in source
    assert "MASTER_PORT" not in source
    assert "sam-drafter-gates" in source


def main():
    test_prompt_bank_has_five_stable_classes()
    test_run_specs_are_175_unique_rows_for_canonical()
    test_required_upload_paths_cover_all_runtime_imports()
    test_profiler_commands_are_policy_specific()
    test_complete_175_row_fixture_is_go()
    test_missing_or_failed_evidence_is_incomplete_not_no_go()
    test_output_mismatch_is_incomplete()
    test_performance_failure_is_no_go_only_after_evidence_passes()
    test_speedup_is_median_of_paired_ratios()
    test_zero_positive_reference_for_required_reduction_is_incomplete()
    test_missing_each_required_policy_branch_is_incomplete()
    test_verify_artifacts_recomputes_summary_report_and_hashes()
    test_resume_rejects_each_compatibility_mismatch()
    test_remote_runner_uses_exact_host_python_model_and_isolation()
    print("sam drafter gate tests passed")


if __name__ == "__main__":
    main()
