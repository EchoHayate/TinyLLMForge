"""Dependency-light tests for the adaptive n-gram canonical gate.

Run: python3 tools/test_adaptive_ngram_gate.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

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


def _run(command: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=True,
    )


def _source_repo() -> tuple[tempfile.TemporaryDirectory, Path]:
    temporary = tempfile.TemporaryDirectory()
    root = Path(temporary.name)
    (root / "tinyvllm").mkdir()
    (root / "tinyvllm" / "__init__.py").write_text(
        "VALUE = 1\n",
        encoding="utf-8",
    )
    (root / "tools").mkdir()
    for name in (
        "draft_model_schema.py",
        "profile_ngram_commit.py",
        "adaptive_ngram_gate.py",
        "test_ngram_speculative.py",
        "test_adaptive_ngram_gate.py",
        "run_adaptive_ngram_gate_remote.sh",
    ):
        (root / "tools" / name).write_text(
            f"# {name}\n",
            encoding="utf-8",
        )
    _run(["git", "init"], root)
    _run(["git", "config", "user.name", "Gate Test"], root)
    _run(["git", "config", "user.email", "gate@example.invalid"], root)
    _run(["git", "add", "."], root)
    _run(["git", "commit", "-m", "base"], root)
    return temporary, root


def _synthetic_source_evidence() -> dict:
    files = []
    return {
        "schema_version": 1,
        "base_commit": "1" * 40,
        "dirty": False,
        "patch_path": "source.patch",
        "patch_sha256": gate.sha256_bytes(b""),
        "patch_size_bytes": 0,
        "owned_roots": list(gate.OWNED_SOURCE_ROOTS),
        "files": files,
        "tree_sha256": gate.source_tree_sha256(files),
    }


def _source_preflight(evidence: dict) -> dict:
    return {
        "schema_version": 1,
        "source_tree_sha256": evidence["tree_sha256"],
        "source_verify": {
            "returncode": 0,
            "stdout_sha256": gate.sha256_text("source verified\n"),
            "stderr_sha256": gate.sha256_text(""),
        },
        "k1_test": {
            "command": [
                "python3",
                "tools/test_ngram_speculative.py",
            ],
            "returncode": 0,
            "stdout_sha256": gate.sha256_text(
                "ngram speculative tests passed\n",
            ),
            "stderr_sha256": gate.sha256_text(""),
        },
    }


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
    source_evidence: dict | None = None,
    source_preflight: dict | None = None,
) -> tuple[dict, list[dict], list[dict]]:
    fixed_tps = fixed_tps or {1: 103.0, 2: 104.0, 4: 105.0}
    source_evidence = source_evidence or _synthetic_source_evidence()
    source_preflight = source_preflight or _source_preflight(source_evidence)
    specs = gate.build_run_specs(repetitions=repetitions, base_seed=20260714)
    manifest = gate.build_manifest(
        repetitions=repetitions,
        base_seed=20260714,
        source_commit=source_evidence["base_commit"],
        source_dirty=source_evidence["dirty"],
        model_path="/models/Qwen3-0.6B",
        model_identifier="Qwen3-0.6B",
        host="synthetic-host",
        python_bin="python3",
        source_evidence=source_evidence,
        source_preflight=source_preflight,
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
            "source_commit": source_evidence["base_commit"],
            "source_dirty": source_evidence["dirty"],
            "source_tree_sha256": source_evidence["tree_sha256"],
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
    source_evidence = _synthetic_source_evidence()
    manifest = gate.build_manifest(
        repetitions=1,
        base_seed=20260714,
        source_commit=source_evidence["base_commit"],
        source_dirty=source_evidence["dirty"],
        model_path="/models/Qwen3-0.6B",
        model_identifier="Qwen3-0.6B",
        host="synthetic-host",
        python_bin="python3",
        source_evidence=source_evidence,
        source_preflight=_source_preflight(source_evidence),
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


def test_source_evidence_reconstructs_dirty_owned_files():
    temporary, root = _source_repo()
    try:
        (root / "tools" / "profile_ngram_commit.py").write_text(
            "# profile_ngram_commit.py\nFAST_K1 = True\n",
            encoding="utf-8",
        )
        out_dir = root / "snapshot"
        evidence = gate.build_source_evidence(root, out_dir)

        assert evidence["schema_version"] == 1
        assert evidence["dirty"] is True
        assert evidence["base_commit"] == _run(
            ["git", "rev-parse", "HEAD"],
            root,
        ).stdout.strip()
        assert evidence["patch_size_bytes"] > 0
        assert evidence["tree_sha256"] == gate.source_tree_sha256(
            evidence["files"],
        )

        reconstructed = root / "reconstructed"
        gate.reconstruct_source_snapshot(
            root,
            reconstructed,
            evidence,
            out_dir / "source.patch",
        )
        gate.validate_source_snapshot(
            reconstructed,
            evidence,
            out_dir / "source.patch",
        )
        assert (
            reconstructed / "tools" / "profile_ngram_commit.py"
        ).read_text(encoding="utf-8").endswith("FAST_K1 = True\n")
    finally:
        temporary.cleanup()


def test_source_evidence_clean_tree_uses_empty_patch():
    temporary, root = _source_repo()
    try:
        out_dir = root / "snapshot"
        evidence = gate.build_source_evidence(root, out_dir)

        assert evidence["dirty"] is False
        assert (out_dir / "source.patch").read_bytes() == b""
        assert evidence["patch_sha256"] == gate.sha256_bytes(b"")
    finally:
        temporary.cleanup()


def test_source_evidence_rejects_untracked_owned_file():
    temporary, root = _source_repo()
    try:
        (root / "tinyvllm" / "untracked.py").write_text(
            "unexpected = True\n",
            encoding="utf-8",
        )
        try:
            gate.build_source_evidence(root, root / "snapshot")
        except ValueError as exc:
            assert "untracked owned source" in str(exc)
        else:
            raise AssertionError("untracked owned source must fail")
    finally:
        temporary.cleanup()


def test_validate_source_snapshot_rejects_changed_missing_and_extra_files():
    temporary, root = _source_repo()
    try:
        out_dir = root / "snapshot"
        evidence = gate.build_source_evidence(root, out_dir)
        source_root = out_dir / "source"

        target = source_root / "tinyvllm" / "__init__.py"
        original = target.read_bytes()
        target.write_bytes(b"changed\n")
        try:
            gate.validate_source_snapshot(
                source_root,
                evidence,
                out_dir / "source.patch",
            )
        except ValueError as exc:
            assert "source file hash mismatch" in str(exc)
        else:
            raise AssertionError("changed source file must fail")
        target.write_bytes(original)

        target.unlink()
        try:
            gate.validate_source_snapshot(
                source_root,
                evidence,
                out_dir / "source.patch",
            )
        except ValueError as exc:
            assert "source path set mismatch" in str(exc)
        else:
            raise AssertionError("missing source path must fail")
        target.write_bytes(original)

        extra = source_root / "tinyvllm" / "extra.py"
        extra.write_text("extra = True\n", encoding="utf-8")
        try:
            gate.validate_source_snapshot(
                source_root,
                evidence,
                out_dir / "source.patch",
            )
        except ValueError as exc:
            assert "source path set mismatch" in str(exc)
        else:
            raise AssertionError("extra source path must fail")
    finally:
        temporary.cleanup()


def test_validate_source_snapshot_rejects_patch_and_tree_tampering():
    temporary, root = _source_repo()
    try:
        (root / "tools" / "profile_ngram_commit.py").write_text(
            "# profile_ngram_commit.py\nFAST_K1 = True\n",
            encoding="utf-8",
        )
        out_dir = root / "snapshot"
        evidence = gate.build_source_evidence(root, out_dir)
        patch = out_dir / "source.patch"
        patch_payload = patch.read_bytes()
        assert patch_payload
        patch.write_bytes(
            bytes([patch_payload[0] ^ 1]) + patch_payload[1:],
        )
        try:
            gate.validate_source_snapshot(
                out_dir / "source",
                evidence,
                patch,
            )
        except ValueError as exc:
            assert "patch hash mismatch" in str(exc)
        else:
            raise AssertionError("changed patch must fail")

        patch.write_bytes(patch_payload)
        changed = dict(evidence)
        changed["tree_sha256"] = "0" * 64
        try:
            gate.validate_source_snapshot(
                out_dir / "source",
                changed,
                patch,
            )
        except ValueError as exc:
            assert "source tree hash mismatch" in str(exc)
        else:
            raise AssertionError("changed tree hash must fail")
    finally:
        temporary.cleanup()


def test_manifest_embeds_source_identity_and_rows_copy_it():
    temporary, root = _source_repo()
    try:
        snapshot = root / "snapshot"
        evidence = gate.build_source_evidence(root, snapshot)
        preflight = _source_preflight(evidence)
        manifest = gate.build_manifest(
            repetitions=1,
            base_seed=20260714,
            source_commit=evidence["base_commit"],
            source_dirty=evidence["dirty"],
            model_path="/models/Qwen3-0.6B",
            model_identifier="Qwen3-0.6B",
            host="synthetic-host",
            python_bin="python3",
            source_evidence=evidence,
            source_preflight=preflight,
        )

        assert manifest["schema_version"] == 2
        assert manifest["source_tree_sha256"] == evidence["tree_sha256"]
        assert manifest["source_evidence"] == evidence
        assert manifest["source_preflight"] == preflight

        spec = manifest["run_specs"][0]
        row, _ = gate._normalize_row(
            manifest,
            spec,
            {"summary": {}, "per_prompt": []},
            {
                "returncode": 1,
                "command": [],
                "tinyvllm_dist_port": 20000,
                "master_port": 20001,
            },
        )
        assert row["source_tree_sha256"] == evidence["tree_sha256"]
    finally:
        temporary.cleanup()


def test_source_preflight_must_match_and_pass():
    temporary, root = _source_repo()
    try:
        evidence = gate.build_source_evidence(root, root / "snapshot")
        gate.validate_source_preflight(_source_preflight(evidence), evidence)

        for mutation, expected in (
            (
                lambda value: value["k1_test"].update(returncode=1),
                "remote K1 test failed",
            ),
            (
                lambda value: value.update(source_tree_sha256="0" * 64),
                "preflight source tree mismatch",
            ),
        ):
            value = json.loads(json.dumps(_source_preflight(evidence)))
            mutation(value)
            try:
                gate.validate_source_preflight(value, evidence)
            except ValueError as exc:
                assert expected in str(exc)
            else:
                raise AssertionError(expected)
    finally:
        temporary.cleanup()


def test_structural_failures_reject_row_source_identity_mismatch():
    manifest, rows, events = _synthetic_complete_gate_rows(repetitions=1)
    rows[0]["source_tree_sha256"] = "0" * 64

    summary = gate.summarize_rows(manifest, rows, events)

    assert summary["decision"] == "INCOMPLETE"
    assert any(
        "source_tree_sha256_mismatch" in item
        for item in summary["structural_failures"]
    )


def _complete_artifact_fixture():
    temporary, root = _source_repo()
    (root / "tools" / "profile_ngram_commit.py").write_text(
        "# profile_ngram_commit.py\nFAST_K1 = True\n",
        encoding="utf-8",
    )
    snapshot = root / "snapshot"
    evidence = gate.build_source_evidence(root, snapshot)
    preflight = _source_preflight(evidence)
    manifest, rows, events = _synthetic_complete_gate_rows(
        repetitions=1,
        source_evidence=evidence,
        source_preflight=preflight,
    )
    out_dir = root / "artifacts"
    out_dir.mkdir()
    shutil.copytree(snapshot / "source", out_dir / "source")
    shutil.copyfile(
        snapshot / "source_evidence.json",
        out_dir / "source_evidence.json",
    )
    shutil.copyfile(snapshot / "source.patch", out_dir / "source.patch")
    (out_dir / "source_preflight.json").write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = gate.summarize_rows(manifest, rows, events)
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "raw_rows.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "event_rows.json").write_text(
        json.dumps(events, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "report.md").write_text(
        gate.render_report(manifest, summary),
        encoding="utf-8",
    )
    return temporary, root, out_dir


def test_verify_artifacts_reconstructs_recorded_source():
    temporary, root, out_dir = _complete_artifact_fixture()
    try:
        summary = gate.verify_artifacts(out_dir, repo_root=root)
        assert summary["decision"] == "GO"
    finally:
        temporary.cleanup()


def test_verify_artifacts_rejects_source_patch_and_preflight_tampering():
    def fail_k1_preflight(path: Path) -> None:
        preflight = json.loads(path.read_text(encoding="utf-8"))
        preflight["k1_test"]["returncode"] = 1
        path.write_text(
            json.dumps(preflight, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest_path = path.parent / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["source_preflight"] = preflight
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    cases = (
        (
            "source/tools/profile_ngram_commit.py",
            lambda path: path.write_text(
                path.read_text(encoding="utf-8") + "tamper\n",
                encoding="utf-8",
            ),
            "source file hash mismatch",
        ),
        (
            "source.patch",
            lambda path: path.write_bytes(
                bytes([path.read_bytes()[0] ^ 1]) + path.read_bytes()[1:],
            ),
            "patch hash mismatch",
        ),
        (
            "source_preflight.json",
            fail_k1_preflight,
            "remote K1 test failed",
        ),
    )
    for relative_path, mutate, expected in cases:
        temporary, root, out_dir = _complete_artifact_fixture()
        try:
            mutate(out_dir / relative_path)
            try:
                gate.verify_artifacts(out_dir, repo_root=root)
            except ValueError as exc:
                assert expected in str(exc)
            else:
                raise AssertionError(expected)
        finally:
            temporary.cleanup()


def test_validate_materialized_source_artifacts_rejects_resume_tampering():
    temporary, root, out_dir = _complete_artifact_fixture()
    try:
        gate.validate_materialized_source_artifacts(out_dir)
        source_path = out_dir / "source" / "tools" / "profile_ngram_commit.py"
        source_path.write_text(
            source_path.read_text(encoding="utf-8") + "tamper\n",
            encoding="utf-8",
        )
        try:
            gate.validate_materialized_source_artifacts(out_dir)
        except ValueError as exc:
            assert "source file hash mismatch" in str(exc)
        else:
            raise AssertionError("tampered resumable source must fail")
    finally:
        temporary.cleanup()


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
    test_source_evidence_reconstructs_dirty_owned_files()
    test_source_evidence_clean_tree_uses_empty_patch()
    test_source_evidence_rejects_untracked_owned_file()
    test_validate_source_snapshot_rejects_changed_missing_and_extra_files()
    test_validate_source_snapshot_rejects_patch_and_tree_tampering()
    test_manifest_embeds_source_identity_and_rows_copy_it()
    test_source_preflight_must_match_and_pass()
    test_structural_failures_reject_row_source_identity_mismatch()
    test_verify_artifacts_reconstructs_recorded_source()
    test_verify_artifacts_rejects_source_patch_and_preflight_tampering()
    test_validate_materialized_source_artifacts_rejects_resume_tampering()
    print("adaptive ngram gate tests passed")


if __name__ == "__main__":
    main()
