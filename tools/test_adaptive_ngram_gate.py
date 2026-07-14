"""Dependency-light tests for the adaptive n-gram canonical gate.

Run: python3 tools/test_adaptive_ngram_gate.py
"""

from __future__ import annotations

import importlib.util
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_GATE_PATH = os.path.join(_THIS_DIR, "adaptive_ngram_gate.py")
_SPEC = importlib.util.spec_from_file_location("adaptive_ngram_gate_under_test", _GATE_PATH)
gate = importlib.util.module_from_spec(_SPEC)
sys.modules["adaptive_ngram_gate_under_test"] = gate
_SPEC.loader.exec_module(gate)
is_retryable_port_collision = gate._is_retryable_port_collision


def _adaptive_events(run_key: str) -> list[dict]:
    return [
        {
            "run_key": run_key,
            "policy": "adaptive",
            "selected_k": 2,
            "proposed_tokens": 2,
            "accepted_count": 2,
            "timing_ms": {"verify_commit_total_ms": 2.0},
            "adaptive_transition": {
                "levels": [1, 2, 4],
                "proposal_event": 1,
                "proposed_tokens": 2,
                "accepted_tokens": 2,
                "event_acceptance": 1.0,
                "acceptance_ema_before": 0.5,
                "acceptance_ema_after": 0.75,
                "full_accept_streak_before": 0,
                "full_accept_streak_after": 1,
                "selected_k_before": 2,
                "selected_k_after": 2,
                "transition_reason": "full_accept_streak",
                "promoted": False,
                "demoted": False,
            },
        },
        {
            "run_key": run_key,
            "policy": "adaptive",
            "selected_k": 2,
            "proposed_tokens": 2,
            "accepted_count": 2,
            "timing_ms": {"verify_commit_total_ms": 2.0},
            "adaptive_transition": {
                "levels": [1, 2, 4],
                "proposal_event": 2,
                "proposed_tokens": 2,
                "accepted_tokens": 2,
                "event_acceptance": 1.0,
                "acceptance_ema_before": 0.75,
                "acceptance_ema_after": 0.875,
                "full_accept_streak_before": 1,
                "full_accept_streak_after": 0,
                "selected_k_before": 2,
                "selected_k_after": 4,
                "transition_reason": "promote",
                "promoted": True,
                "demoted": False,
            },
        },
    ]


def _synthetic_complete_gate_rows(
    repetitions: int = 7,
    baseline_tps: float = 100.0,
    fixed_tps: dict[int, float] | None = None,
    adaptive_tps: float = 108.0,
    adaptive_waste: int = 20,
    fixed_k4_waste: int = 40,
    adaptive_zero_ms: float = 8.0,
    fixed_k4_zero_ms: float = 12.0,
) -> tuple[dict, list[dict], list[dict]]:
    fixed_tps = fixed_tps or {1: 103.0, 2: 104.0, 4: 105.0}
    specs = gate.build_run_specs(repetitions=repetitions, base_seed=20260714)
    manifest = gate.build_manifest(
        repetitions=repetitions,
        base_seed=20260714,
        source_commit="synthetic",
        source_dirty=False,
        model_path="/models/Qwen3-0.6B",
        model_identifier="Qwen3-0.6B",
        host="synthetic-host",
        python_bin="python3",
    )
    rows = []
    events = []
    for spec in specs:
        policy = spec["policy"]
        if policy == "baseline":
            tps = baseline_tps
            waste = 0
            zero_ms = 0.0
        elif policy == "adaptive":
            tps = adaptive_tps
            waste = adaptive_waste
            zero_ms = adaptive_zero_ms
        else:
            fixed_k = int(policy[-1])
            tps = fixed_tps[fixed_k]
            waste = fixed_k4_waste if fixed_k == 4 else max(1, fixed_k * 5)
            zero_ms = fixed_k4_zero_ms if fixed_k == 4 else float(fixed_k)

        output_tokens = 100
        row = {
            **spec,
            "model_path": "/models/Qwen3-0.6B",
            "model_identifier": "Qwen3-0.6B",
            "source_commit": "synthetic",
            "source_dirty": False,
            "prompt_tokens": 32,
            "output_tokens": output_tokens,
            "output_token_ids": [11, 12, 13],
            "output_token_sha256": gate.sha256_json([11, 12, 13]),
            "elapsed_s": output_tokens / tps,
            "output_tokens_per_s": tps,
            "proposal_events": 2 if policy != "baseline" else 0,
            "no_draft_positions": 0,
            "drafted_tokens": 100 if policy != "baseline" else 0,
            "accepted_tokens": (100 - waste) if policy != "baseline" else 0,
            "wasted_draft_tokens": waste,
            "draft_waste_rate": waste / 100.0 if policy != "baseline" else 0.0,
            "zero_accept_events": 1 if zero_ms else 0,
            "zero_accept_event_rate": 0.5 if zero_ms else 0.0,
            "zero_accept_verify_ms": zero_ms,
            "verify_timing_ms": {"verify_commit_total_ms": 20.0 if policy != "baseline" else 0.0},
            "selected_k_counts": (
                {"1": 0, "2": 2, "4": 0}
                if policy == "adaptive"
                else {"1": 2 if policy == "fixed_k1" else 0,
                      "2": 2 if policy == "fixed_k2" else 0,
                      "4": 2 if policy == "fixed_k4" else 0}
            ),
            "autoregressive_steps_avoided": (100 - waste) if policy != "baseline" else 0,
            "profiler_gate_pass": True,
            "profiler_gate_fail_reasons": [],
            "process": {
                "returncode": 0,
                "command": ["python3", "tools/profile_ngram_commit.py"],
                "tinyvllm_dist_port": 20000 + spec["global_order"] * 2,
                "master_port": 20001 + spec["global_order"] * 2,
                "stdout_path": f"logs/{spec['run_key']}.stdout.log",
                "stderr_path": f"logs/{spec['run_key']}.stderr.log",
            },
        }
        rows.append(row)
        if policy == "adaptive":
            events.extend(_adaptive_events(spec["run_key"]))
    return manifest, rows, events


def test_prompt_bank_has_four_stable_single_sequence_classes():
    assert [item["name"] for item in gate.PROMPT_BANK] == [
        "natural_prose",
        "structured_mixed",
        "repeated_long_context",
        "transition_heavy",
    ]
    assert {item["workload_class"] for item in gate.PROMPT_BANK} == {
        "natural",
        "mixed",
        "high_repeat",
        "transition_heavy",
    }
    for item in gate.PROMPT_BANK:
        assert item["prompt"]
        assert item["max_output_len"] > 0
        assert item["prompt_sha256"] == gate.sha256_text(item["prompt"])


def test_build_run_specs_is_complete_unique_and_deterministic():
    first = gate.build_run_specs(repetitions=2, base_seed=20260714)
    second = gate.build_run_specs(repetitions=2, base_seed=20260714)

    assert first == second
    assert len(first) == 4 * 5 * 2
    keys = [item["run_key"] for item in first]
    assert len(keys) == len(set(keys))
    assert {item["policy"] for item in first} == {
        "baseline",
        "fixed_k1",
        "fixed_k2",
        "fixed_k4",
        "adaptive",
    }
    assert all(item["max_num_seqs"] == 1 for item in first)


def test_replay_adaptive_trajectory_detects_tampering():
    events = _adaptive_events("adaptive-run")

    assert gate.replay_adaptive_trajectory(events)["valid"] is True
    events[1]["adaptive_transition"]["selected_k_after"] = 1
    replay = gate.replay_adaptive_trajectory(events)

    assert replay["valid"] is False
    assert replay["fail_reasons"]


def test_summarize_rows_returns_go_for_committed_threshold_case():
    manifest, rows, events = _synthetic_complete_gate_rows()

    summary = gate.summarize_rows(manifest, rows, events)

    assert summary["decision"] == "GO"
    assert summary["correctness_pass"] is True
    assert summary["observed_rows"] == 140


def test_summarize_rows_near_best_requires_both_waste_reductions():
    manifest, rows, events = _synthetic_complete_gate_rows(
        fixed_tps={1: 104.0, 2: 106.0, 4: 107.0},
        adaptive_tps=106.5,
        adaptive_waste=31,
        fixed_k4_waste=40,
        adaptive_zero_ms=11.0,
        fixed_k4_zero_ms=12.0,
    )

    summary = gate.summarize_rows(manifest, rows, events)

    assert summary["decision"] == "NO_GO"
    assert "adaptive_vs_fixed_gate_failed" in summary["decision_reasons"]


def test_summarize_rows_marks_missing_or_failed_process_incomplete():
    manifest, rows, events = _synthetic_complete_gate_rows(repetitions=1)

    assert gate.summarize_rows(manifest, rows[:-1], events)["decision"] == "INCOMPLETE"
    rows[-1]["process"]["returncode"] = 1
    assert gate.summarize_rows(manifest, rows, events)["decision"] == "INCOMPLETE"


def test_summarize_rows_marks_missing_prompt_token_count_incomplete():
    manifest, rows, events = _synthetic_complete_gate_rows(repetitions=1)
    rows[0]["prompt_tokens"] = None

    summary = gate.summarize_rows(manifest, rows, events)

    assert summary["decision"] == "INCOMPLETE"
    assert any("prompt_tokens_invalid" in reason for reason in summary["structural_failures"])


def test_natural_prompt_regression_forces_no_go():
    manifest, rows, events = _synthetic_complete_gate_rows()
    for row in rows:
        if row["prompt_name"] == "natural_prose" and row["policy"] == "adaptive":
            row["elapsed_s"] = row["output_tokens"] / 94.0
            row["output_tokens_per_s"] = 94.0

    summary = gate.summarize_rows(manifest, rows, events)

    assert summary["decision"] == "NO_GO"
    assert "natural_or_transition_regression" in summary["decision_reasons"]


def test_required_upload_paths_cover_profiler_imports():
    assert gate.REQUIRED_UPLOAD_PATHS == (
        "tinyvllm",
        "tools/draft_model_schema.py",
        "tools/profile_ngram_commit.py",
        "tools/adaptive_ngram_gate.py",
    )


def test_port_collision_retry_classifier_is_narrow():
    assert is_retryable_port_collision(
        1,
        "RuntimeError: The server socket has failed to listen: EADDRINUSE",
    )
    assert not is_retryable_port_collision(0, "EADDRINUSE")
    assert not is_retryable_port_collision(1, "RuntimeError: CUDA out of memory")


def test_normalize_row_uses_profiler_prompt_tokens_and_candidate_metrics():
    manifest = gate.build_manifest(
        repetitions=1,
        base_seed=20260714,
        source_commit="synthetic",
        source_dirty=False,
        model_path="/models/Qwen3-0.6B",
        model_identifier="Qwen3-0.6B",
        host="synthetic-host",
        python_bin="python3",
    )
    spec = next(item for item in manifest["run_specs"] if item["policy"] == "adaptive")
    profiler_result = {
        "summary": {
            "elapsed_s": 2.0,
            "output_tokens": 4,
            "output_tokens_per_s": 2.0,
            "commit_attempts": 1,
            "no_draft_steps": 3,
            "drafted_tokens": 2,
            "accepted_count": 1,
            "wasted_draft_tokens": 1,
            "draft_waste_rate": 0.5,
            "zero_accept_events": 0,
            "zero_accept_event_rate": 0.0,
            "zero_accept_verify_ms": 0.0,
            "verify_timing_ms": {"verify_commit_total_ms": 1.5},
            "selected_k_counts": {"1": 0, "2": 1, "4": 0},
            "candidate_autoregressive_steps_avoided": 1,
            "gate_pass": True,
            "gate_fail_reasons": [],
        },
        "per_prompt": [{
            "prompt_tokens": 27,
            "output_tokens": 4,
            "token_ids": [1, 2, 3, 4],
        }],
        "verify_events": [],
    }

    row, events = gate._normalize_row(
        manifest,
        spec,
        profiler_result,
        {
            "returncode": 0,
            "command": [],
            "tinyvllm_dist_port": 20000,
            "master_port": 20001,
        },
    )

    assert row["prompt_tokens"] == 27
    assert row["output_token_ids"] == [1, 2, 3, 4]
    assert row["wasted_draft_tokens"] == 1
    assert row["profiler_gate_pass"] is True
    assert events == []


def test_profiler_command_forces_fixed_length_greedy_measurement():
    spec = gate.build_run_specs(repetitions=1, base_seed=20260714)[0]
    prompt = next(item for item in gate.PROMPT_BANK if item["name"] == spec["prompt_name"])

    command = gate._profiler_command(
        spec,
        prompt,
        python_bin="python3",
        model_path="/models/Qwen3-0.6B",
        process_json=gate.Path("/tmp/process.json"),
    )

    assert "--ignore-eos" in command
    assert command[command.index("--max-num-seqs") + 1] == "1"
    assert command[command.index("--temperature") + 1] == "0.0"


def main():
    test_prompt_bank_has_four_stable_single_sequence_classes()
    test_build_run_specs_is_complete_unique_and_deterministic()
    test_replay_adaptive_trajectory_detects_tampering()
    test_summarize_rows_returns_go_for_committed_threshold_case()
    test_summarize_rows_near_best_requires_both_waste_reductions()
    test_summarize_rows_marks_missing_or_failed_process_incomplete()
    test_summarize_rows_marks_missing_prompt_token_count_incomplete()
    test_natural_prompt_regression_forces_no_go()
    test_required_upload_paths_cover_profiler_imports()
    test_port_collision_retry_classifier_is_narrow()
    test_normalize_row_uses_profiler_prompt_tokens_and_candidate_metrics()
    test_profiler_command_forces_fixed_length_greedy_measurement()
    print("adaptive ngram gate tests passed")


if __name__ == "__main__":
    main()
