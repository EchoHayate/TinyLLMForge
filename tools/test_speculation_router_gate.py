"""Dependency-light tests for the speculation profitability router gate."""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import tempfile
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
GATE_PATH = THIS_DIR / "speculation_router_gate.py"
SPEC = importlib.util.spec_from_file_location(
    "speculation_router_gate_under_test",
    os.fspath(GATE_PATH),
)
gate = importlib.util.module_from_spec(SPEC)
sys.modules["speculation_router_gate_under_test"] = gate
SPEC.loader.exec_module(gate)


def _capability():
    return {
        "status": "PASS",
        "rows": [
            {
                "dtype": dtype,
                "query_len": query_len,
                "block_case": block_case,
                "gqa": True,
                "output_match": True,
                "kv_match": True,
                "future_row_masked": True,
                "finite": True,
            }
            for dtype in ("torch.float16", "torch.bfloat16")
            for query_len in (1, 3, 7, 15)
            for block_case in ("one_block", "cross_block")
        ],
    }


def _complete_fixture():
    source_evidence = {
        "schema_version": 1,
        "base_commit": "1" * 40,
        "dirty": False,
        "patch_path": "source.patch",
        "patch_sha256": "2" * 64,
        "patch_size_bytes": 0,
        "owned_roots": list(gate.OWNED_SOURCE_ROOTS),
        "files": [],
        "tree_sha256": "3" * 64,
    }
    source_preflight = {
        "schema_version": 1,
        "source_tree_sha256": source_evidence["tree_sha256"],
    }
    manifest = gate.build_controlled_manifest(
        source_evidence=source_evidence,
        source_preflight=source_preflight,
        model_path="/models/Qwen3-0.6B",
        model_identifier="Qwen3-0.6B",
        host="synthetic-host",
        python_bin="python3",
        torch_version="2.4.1",
        cuda_version="12.1",
        flash_attn_version="2.6.3",
        gpu_name="Synthetic GPU",
        bf16_supported=True,
        run_tag="synthetic-controlled",
    )
    case_rows = []
    event_rows = []
    router_rows = []
    next_port = 10000
    for case in gate.CONTROLLED_CASE_MATRIX:
        baseline_elapsed = 1.0
        accepted_count = int(case["expected_accepted_count"])
        for policy in gate.CONTROLLED_POLICIES:
            elapsed_s = baseline_elapsed
            if (
                policy == "routed_native"
                and case["draft_len"] >= 2
                and accepted_count >= 2
            ):
                elapsed_s = 0.90
            elif policy == "always_native":
                elapsed_s = 0.95
            process = {
                "returncode": 0,
                "tinyvllm_dist_port": next_port,
                "master_port": next_port + 1,
            }
            next_port += 2
            manifest["process_port_pairs"].append({
                "case_id": case["case_id"],
                "policy": policy,
                "tinyvllm_dist_port": process[
                    "tinyvllm_dist_port"
                ],
                "master_port": process["master_port"],
            })
            row = {
                "case_id": case["case_id"],
                "policy": policy,
                "status": "PASS",
                "process": process,
                "source_tree_sha256": source_evidence[
                    "tree_sha256"
                ],
                "elapsed_s": elapsed_s,
                "baseline_elapsed_s": baseline_elapsed,
                "output_tokens": 16,
                "output_tokens_per_s": 16.0 / elapsed_s,
                "output_token_sha256": (
                    f"output-{case['case_id']}"
                ),
                "continuation_token_sha256": (
                    f"continuation-{case['case_id']}"
                ),
                "accepted_count": accepted_count,
                "target_forward_count": (
                    16 - accepted_count
                    if policy != "baseline"
                    else 16
                ),
            }
            if policy == "oracle":
                row["comparison"] = {
                    "status": "PASS",
                    "target_token_match": True,
                    "accepted_prefix_match": True,
                    "metadata_match": True,
                    "continuation_token_match": True,
                    "finite": True,
                    "logits_within_tolerance": True,
                    "kv_within_tolerance": True,
                    "continuation_steps": 16,
                }
            case_rows.append(row)

        for policy in ("always_native", "routed_native"):
            if (
                policy == "routed_native"
                and case["draft_len"] <= 1
            ):
                continue
            event_rows.append({
                "case_id": case["case_id"],
                "policy": policy,
                "draft_len": case["draft_len"],
                "accepted_count": accepted_count,
                "accepted_kv_rematerialization": {
                    "decode_calls": 0,
                    "rematerialized_tokens": [],
                    "elapsed_ms": 0.0,
                },
                "accepted_kv_copy_calls": 0,
                "accepted_kv_replay_calls": 0,
                "target_forward_count": 1,
                "verifier_commit_ms": 1.0,
            })

        route = (
            "baseline_short_draft"
            if case["draft_len"] <= 1
            else "native_multi_token"
        )
        router_rows.append({
            "case_id": case["case_id"],
            "policy": "routed_native",
            "route": route,
            "draft_len": case["draft_len"],
            "route_fallback_reason": None,
            "speculative_reservation_attempted": (
                route == "native_multi_token"
            ),
            "spec_verify_prepare_calls": (
                1 if route == "native_multi_token" else 0
            ),
            "spec_verify_forward_calls": (
                1 if route == "native_multi_token" else 0
            ),
            "target_forward_count": (
                1 if route == "native_multi_token" else 0
            ),
        })
    return (
        manifest,
        _capability(),
        case_rows,
        event_rows,
        router_rows,
    )


def _classify(fixture):
    return gate.classify_controlled_gate(*fixture)


def test_complete_controlled_evidence_is_ready():
    summary = _classify(_complete_fixture())
    assert (
        summary["classification"]
        == "READY_FOR_REAL_DRAFTER_GATE"
    )
    assert summary["exactness_pass"] is True
    assert summary["replay_elimination_pass"] is True
    assert summary["router_isolation_pass"] is True


def test_no_profitable_region_is_no_go():
    fixture = list(_complete_fixture())
    rows = fixture[2]
    for row in rows:
        if row["policy"] == "routed_native":
            row["elapsed_s"] = row["baseline_elapsed_s"]
            row["output_tokens_per_s"] = (
                row["output_tokens"] / row["elapsed_s"]
            )
    summary = _classify(tuple(fixture))
    assert summary["classification"] == "NO_GO"
    assert "no_profitable_k_ge_2_region" in summary["reasons"]


def test_short_route_mutation_is_no_go():
    fixture = list(_complete_fixture())
    short = next(
        row for row in fixture[4]
        if row["draft_len"] == 1
    )
    short["speculative_reservation_attempted"] = True
    summary = _classify(tuple(fixture))
    assert summary["classification"] == "NO_GO"
    assert summary["router_isolation_pass"] is False


def test_native_replay_is_no_go():
    fixture = list(_complete_fixture())
    fixture[3][0]["accepted_kv_replay_calls"] = 1
    summary = _classify(tuple(fixture))
    assert summary["classification"] == "NO_GO"
    assert summary["replay_elimination_pass"] is False


def test_semantic_mismatch_is_no_go():
    fixture = list(_complete_fixture())
    oracle = next(
        row for row in fixture[2]
        if row["policy"] == "oracle"
    )
    oracle["comparison"]["kv_within_tolerance"] = False
    summary = _classify(tuple(fixture))
    assert summary["classification"] == "NO_GO"
    assert summary["exactness_pass"] is False


def test_missing_row_is_incomplete():
    fixture = list(_complete_fixture())
    fixture[2].pop()
    summary = _classify(tuple(fixture))
    assert summary["classification"] == "INCOMPLETE"


def test_failed_process_is_incomplete():
    fixture = list(_complete_fixture())
    fixture[2][0]["process"]["returncode"] = 2
    summary = _classify(tuple(fixture))
    assert summary["classification"] == "INCOMPLETE"


def test_duplicate_port_pair_is_incomplete():
    fixture = list(_complete_fixture())
    fixture[0]["process_port_pairs"][1].update(
        fixture[0]["process_port_pairs"][0]
    )
    summary = _classify(tuple(fixture))
    assert summary["classification"] == "INCOMPLETE"


def test_nonfinite_performance_is_incomplete():
    fixture = list(_complete_fixture())
    fixture[2][0]["elapsed_s"] = math.nan
    summary = _classify(tuple(fixture))
    assert summary["classification"] == "INCOMPLETE"


def test_lifecycle_regression_is_no_go():
    fixture = list(_complete_fixture())
    row = next(
        row for row in fixture[2]
        if row["policy"] == "routed_native"
        and row["case_id"] == "k2-zero-current"
    )
    row["elapsed_s"] = 1.06
    summary = _classify(tuple(fixture))
    assert summary["classification"] == "NO_GO"
    assert any(
        reason.startswith("required_lifecycle_regression:")
        for reason in summary["reasons"]
    )


def _write_driver_inputs(out_dir: Path):
    source_evidence = {
        "schema_version": 1,
        "base_commit": "1" * 40,
        "dirty": False,
        "tree_sha256": "3" * 64,
    }
    source_preflight = {
        "schema_version": 1,
        "source_tree_sha256": source_evidence["tree_sha256"],
        "model_identifier": "Qwen3-0.6B",
        "torch": "2.4.1",
        "cuda": "12.1",
        "flash_attn": "2.6.3",
        "gpu": "Synthetic GPU",
        "bf16_supported": True,
    }
    evidence_path = out_dir / "source_evidence.input.json"
    patch_path = out_dir / "source.input.patch"
    preflight_path = out_dir / "source_preflight.input.json"
    evidence_path.write_text(json.dumps(source_evidence))
    patch_path.write_bytes(b"")
    preflight_path.write_text(json.dumps(source_preflight))
    (out_dir / "capability.json").write_text(
        json.dumps(_capability())
    )
    return evidence_path, patch_path, preflight_path


def _fake_driver_process(calls, failed_keys=()):
    failed_keys = set(failed_keys)

    def run_process(
        *,
        python_bin,
        model_path,
        policy,
        case,
        out_path,
        log_dir,
    ):
        del python_bin, model_path, log_dir
        key = (case["case_id"], policy)
        calls.append(key)
        port = 20000 + len(calls) * 2
        process = {
            "returncode": 2 if key in failed_keys else 0,
            "tinyvllm_dist_port": port,
            "master_port": port + 1,
        }
        if key in failed_keys:
            return None, process
        if policy == "probe":
            payload = {
                "case_id": case["case_id"],
                "policy": "probe",
                "target_tokens": list(
                    range(10, 10 + int(case["draft_len"]))
                ),
                "vocab_size": 1000,
                "eos_token_id": 999,
                "prompt_token_count": 8,
                "history_tokens": list(
                    range(int(case["history_len"]))
                ),
            }
        else:
            route = (
                "baseline_short_draft"
                if int(case["draft_len"]) <= 1
                else "native_multi_token"
            )
            event = {
                "route": route,
                "route_fallback_reason": None,
                "draft_len": int(case["draft_len"]),
                "accepted_count": int(
                    case["expected_accepted_count"]
                ),
                "accepted_kv_rematerialization": {
                    "decode_calls": 0,
                    "rematerialized_tokens": [],
                    "elapsed_ms": 0.0,
                },
                "accepted_kv_copy_calls": 0,
                "accepted_kv_replay_calls": 0,
                "target_forward_count": (
                    0 if route == "baseline_short_draft" else 1
                ),
                "speculative_reservation_attempted": (
                    route == "native_multi_token"
                ),
                "spec_verify_prepare_calls": (
                    0 if route == "baseline_short_draft" else 1
                ),
                "spec_verify_forward_calls": (
                    0 if route == "baseline_short_draft" else 1
                ),
                "timing_ms": {"verify_commit_total_ms": 1.0},
            }
            payload = {
                "case_id": case["case_id"],
                "policy": policy,
                "status": "PASS",
                "draft_construction": (
                    "controlled_target_derived"
                ),
                "target_tokens": list(case["draft_tokens"]),
                "accepted_tokens": list(case["draft_tokens"]),
                "sequence_tokens_after": [1, 2, 3],
                "block_table_after": [0],
                "continuation_tokens": list(range(16)),
                "dtype": "torch.float16",
                "finite": True,
                "logits": [[0.0, 1.0]],
                "kv": {"keys": [[1.0]], "values": [[2.0]]},
                "continuation_logits": [[[0.0, 1.0]]],
                "continuation_kv": [{
                    "keys": [[1.0]],
                    "values": [[2.0]],
                }],
                "elapsed_s": 1.0,
                "output_tokens": 16,
                "output_tokens_per_s": 16.0,
                "output_token_sha256": (
                    f"output-{case['case_id']}"
                ),
                "continuation_token_sha256": (
                    f"continuation-{case['case_id']}"
                ),
                "event": event if policy in (
                    "always_native",
                    "routed_native",
                    "legacy_rematerialize",
                ) else None,
                "route": (
                    route if policy == "routed_native" else None
                ),
                "route_fallback_reason": None,
                "router_event": (
                    event if policy == "routed_native" else None
                ),
            }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload))
        return payload, process

    return run_process


def test_controlled_driver_runs_deterministic_prefix_with_unique_ports():
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        inputs = _write_driver_inputs(out_dir)
        calls = []
        original = gate._case_process
        gate._case_process = _fake_driver_process(calls)
        try:
            result = gate.run_controlled_gate(
                out_dir=out_dir,
                python_bin="python3",
                model_path="/model",
                source_evidence_path=inputs[0],
                source_patch_path=inputs[1],
                source_preflight_path=inputs[2],
                host="synthetic-host",
                run_tag="driver-test",
                case_limit=2,
            )
        finally:
            gate._case_process = original

    expected = []
    for case in gate.CONTROLLED_CASE_MATRIX[:2]:
        expected.append((case["case_id"], "probe"))
        expected.extend(
            (case["case_id"], policy)
            for policy in gate.CONTROLLED_POLICIES
        )
    assert calls == expected
    pairs = [
        (
            row["process"]["tinyvllm_dist_port"],
            row["process"]["master_port"],
        )
        for row in result["case_rows"]
    ]
    assert len(pairs) == len(set(pairs))


def test_controlled_driver_resume_retries_only_failed_row():
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        inputs = _write_driver_inputs(out_dir)
        failed_key = (
            gate.CONTROLLED_CASE_MATRIX[0]["case_id"],
            "routed_native",
        )
        first_calls = []
        original = gate._case_process
        gate._case_process = _fake_driver_process(
            first_calls,
            failed_keys={failed_key},
        )
        try:
            gate.run_controlled_gate(
                out_dir=out_dir,
                python_bin="python3",
                model_path="/model",
                source_evidence_path=inputs[0],
                source_patch_path=inputs[1],
                source_preflight_path=inputs[2],
                host="synthetic-host",
                run_tag="driver-test",
                case_limit=1,
            )
            initial_manifest = json.loads(
                (out_dir / "manifest.json").read_text()
            )
            baseline_process = next(
                row for row in initial_manifest[
                    "process_port_pairs"
                ]
                if row["case_id"] == failed_key[0]
                and row["policy"] == "baseline"
            )
            stale_router = [{
                "case_id": failed_key[0],
                "policy": "routed_native",
                "route": "stale",
            }]
            (out_dir / "router_rows.json").write_text(
                json.dumps(stale_router)
            )
            resume_calls = []
            gate._case_process = _fake_driver_process(resume_calls)
            result = gate.run_controlled_gate(
                out_dir=out_dir,
                python_bin="python3",
                model_path="/model",
                source_evidence_path=inputs[0],
                source_patch_path=inputs[1],
                source_preflight_path=inputs[2],
                host="synthetic-host",
                run_tag="driver-test",
                resume=True,
                case_limit=1,
            )
        finally:
            gate._case_process = original

    assert resume_calls == [failed_key]
    resumed_pairs = result["manifest"]["process_port_pairs"]
    pair_keys = [
        (row["case_id"], row["policy"])
        for row in resumed_pairs
    ]
    assert len(pair_keys) == len(set(pair_keys))
    assert next(
        row for row in resumed_pairs
        if row["case_id"] == failed_key[0]
        and row["policy"] == "baseline"
    ) == baseline_process
    router = next(
        row for row in result["router_rows"]
        if row["case_id"] == failed_key[0]
    )
    assert router["route"] == "baseline_short_draft"


def test_controlled_materialization_rejects_non_target_derived_label():
    case = dict(gate.CONTROLLED_CASE_MATRIX[0])
    case["draft_construction"] = "ngram"
    try:
        gate._materialize_controlled_case(
            case,
            {
                "target_tokens": [10],
                "vocab_size": 1000,
                "eos_token_id": 999,
                "prompt_token_count": 8,
                "history_tokens": list(range(64)),
            },
            source_tree_sha256="3" * 64,
        )
    except ValueError as exc:
        assert "controlled_target_derived" in str(exc)
    else:
        raise AssertionError(
            "controlled gate must reject real-source construction"
        )


def test_short_route_exactness_uses_baseline_reference():
    rows = {
        ("k1-route-fallback", "baseline"): {"baseline": True},
        ("k1-route-fallback", "routed_native"): {"routed": True},
        ("k1-route-fallback", "oracle"): {"oracle": True},
    }
    assert gate._exactness_reference_row(
        gate.CONTROLLED_CASE_MATRIX[0],
        rows,
    ) == {"baseline": True}


def test_compact_row_promotes_scalars_without_numeric_arrays():
    payload = {
        "case_id": "case-1",
        "policy": "routed_native",
        "status": "PASS",
        "draft_construction": "controlled_target_derived",
        "target_tokens": [10, 20],
        "accepted_tokens": [10],
        "logits": [[0.0, 1.0]],
        "kv": {"keys": [[1.0]], "values": [[2.0]]},
        "continuation_logits": [[[0.0, 1.0]]],
        "continuation_kv": [{
            "keys": [[1.0]],
            "values": [[2.0]],
        }],
        "event": {
            "accepted_count": 1,
            "target_forward_count": 1,
            "normal_decode_forward_count": 0,
        },
    }
    row = gate._normalize_controlled_row(
        payload,
        {
            "returncode": 0,
            "tinyvllm_dist_port": 20000,
            "master_port": 20001,
        },
        case_id="case-1",
        policy="routed_native",
        source_tree_sha256="3" * 64,
    )

    assert row["accepted_count"] == 1
    assert row["target_forward_count"] == 1
    assert "logits" not in row
    assert "kv" not in row
    assert "continuation_logits" not in row
    assert "continuation_kv" not in row
    assert row["logits_sha256"]
    assert row["kv_sha256"]


def _real_source_fixture():
    prompt_bank = {
        "schema_version": 1,
        "prompts": [
            {
                "prompt_id": f"{bucket}-1",
                "bucket": bucket,
                "prompt": f"fixture {bucket}",
                "max_tokens": 32,
                "seed": 0,
                "dtype": "torch.float16",
            }
            for bucket in (
                "natural",
                "code",
                "repetitive",
                "transition_heavy",
                "low_match",
                "eos",
                "short_context",
                "long_context",
            )
        ],
    }
    prompt_hash = gate.canonical_prompt_bank_sha256(prompt_bank)
    draft_source = {
        "schema_version": 1,
        "source_name": "fixture-learned-drafter",
        "source_type": "learned_speculative_head",
        "implementation_paths": ["tools/profile_ngram_commit.py"],
        "source_tree_sha256": "a" * 64,
        "checkpoint_identifier": "fixture/checkpoint",
        "checkpoint_config_sha256": "b" * 64,
        "tokenizer_identifier": "Qwen3-0.6B",
        "vocab_size": 151936,
        "hyperparameters": {"max_draft_tokens": 8},
        "consumes_target_hidden_states": True,
        "requires_additional_model_forward": True,
        "target_derived": False,
        "debug_stub": False,
        "prompt_bank_sha256": prompt_hash,
    }
    manifest = {
        "stage": "real-source",
        "source_tree_sha256": "a" * 64,
        "draft_source_sha256": gate._sha256_json(draft_source),
        "prompt_bank_sha256": prompt_hash,
        "policies": list(gate.REAL_POLICIES),
        "thresholds": gate.REAL_THRESHOLDS,
        "repetitions": 3,
        "warmup_repetitions": 1,
    }
    case_rows = []
    event_rows = []
    router_rows = []
    port = 30000
    for prompt in prompt_bank["prompts"]:
        baseline_elapsed = 1.0
        for policy, elapsed_s in (
            ("baseline", baseline_elapsed),
            ("source_always_native", 0.97),
            ("source_routed_native", 0.90),
        ):
            case_rows.append({
                "prompt_id": prompt["prompt_id"],
                "bucket": prompt["bucket"],
                "policy": policy,
                "status": "PASS",
                "source_tree_sha256": "a" * 64,
                "draft_source_sha256": manifest[
                    "draft_source_sha256"
                ],
                "prompt_bank_sha256": prompt_hash,
                "hyperparameters_sha256": gate._sha256_json(
                    draft_source["hyperparameters"]
                ),
                "elapsed_s": elapsed_s,
                "output_tokens": 32,
                "output_tokens_per_s": 32.0 / elapsed_s,
                "output_token_sha256": (
                    f"output-{prompt['prompt_id']}"
                ),
                "process": {
                    "returncode": 0,
                    "tinyvllm_dist_port": port,
                    "master_port": port + 1,
                },
            })
            port += 2
        event_rows.append({
            "prompt_id": prompt["prompt_id"],
            "policy": "source_routed_native",
            "proposal_elapsed_ms": 1.0,
            "proposed_count": 4,
            "accepted_count": 3,
            "rejected_count": 1,
            "target_forward_count": 1,
            "baseline_target_forward_count": 4,
            "accepted_kv_rematerialization": {
                "decode_calls": 0,
                "rematerialized_tokens": [],
                "elapsed_ms": 0.0,
            },
            "accepted_kv_copy_calls": 0,
            "accepted_kv_replay_calls": 0,
        })
        router_rows.append({
            "prompt_id": prompt["prompt_id"],
            "policy": "source_routed_native",
            "route": (
                "baseline_incompatible"
                if prompt["bucket"] == "low_match"
                else "native_multi_token"
            ),
            "route_fallback_reason": (
                "fixture incompatibility"
                if prompt["bucket"] == "low_match"
                else None
            ),
        })
    return (
        manifest,
        draft_source,
        prompt_bank,
        case_rows,
        event_rows,
        router_rows,
    )


def test_real_source_manifest_rejects_non_real_sources():
    fixture = list(_real_source_fixture())
    draft_source = fixture[1]
    gate.validate_draft_source_manifest(draft_source)
    for field, value in (
        ("target_derived", True),
        ("debug_stub", True),
        ("source_type", "ngram"),
        ("checkpoint_identifier", ""),
    ):
        invalid = dict(draft_source)
        invalid[field] = value
        try:
            gate.validate_draft_source_manifest(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(field)


def test_complete_real_source_evidence_is_go():
    assert gate.classify_real_source_gate(
        *_real_source_fixture()
    )["classification"] == "GO"


def test_real_source_performance_and_route_failures_are_no_go():
    mutations = []

    def no_elapsed_gain(fixture):
        for row in fixture[3]:
            if row["policy"] == "source_routed_native":
                row["elapsed_s"] = 0.96
                row["output_tokens_per_s"] = 32.0 / 0.96

    mutations.append(no_elapsed_gain)

    def no_tokens_per_s_gain(fixture):
        for row in fixture[3]:
            if row["policy"] == "source_routed_native":
                row["output_tokens"] = 28
                row["output_tokens_per_s"] = (
                    28.0 / row["elapsed_s"]
                )

    mutations.append(no_tokens_per_s_gain)

    def natural_regression(fixture):
        row = next(
            row for row in fixture[3]
            if row["bucket"] == "natural"
            and row["policy"] == "source_routed_native"
        )
        row["elapsed_s"] = 1.01
        row["output_tokens_per_s"] = 32.0 / 1.01

    mutations.append(natural_regression)

    def transition_regression(fixture):
        row = next(
            row for row in fixture[3]
            if row["bucket"] == "transition_heavy"
            and row["policy"] == "source_routed_native"
        )
        row["elapsed_s"] = 1.01
        row["output_tokens_per_s"] = 32.0 / 1.01

    mutations.append(transition_regression)

    def individual_prompt_regression(fixture):
        row = next(
            row for row in fixture[3]
            if row["bucket"] == "repetitive"
            and row["policy"] == "source_routed_native"
        )
        row["elapsed_s"] = 1.11
        row["output_tokens_per_s"] = 32.0 / 1.11

    mutations.append(individual_prompt_regression)

    def routed_slower(fixture):
        row = next(
            row for row in fixture[3]
            if row["bucket"] == "code"
            and row["policy"] == "source_routed_native"
        )
        row["elapsed_s"] = 0.98
        row["output_tokens_per_s"] = 32.0 / 0.98

    mutations.append(routed_slower)

    def no_fallback(fixture):
        for row in fixture[5]:
            row["route"] = "native_multi_token"
            row["route_fallback_reason"] = None

    mutations.append(no_fallback)

    def no_forward_reduction(fixture):
        fixture[4][0]["target_forward_count"] = 4

    mutations.append(no_forward_reduction)

    def output_mismatch(fixture):
        row = next(
            row for row in fixture[3]
            if row["policy"] == "source_routed_native"
        )
        row["output_token_sha256"] = "mismatch"

    mutations.append(output_mismatch)

    for mutate in mutations:
        fixture = list(_real_source_fixture())
        mutate(fixture)
        assert gate.classify_real_source_gate(
            *fixture
        )["classification"] == "NO_GO", mutate.__name__


def test_real_source_missing_identity_or_process_is_incomplete():
    fixture = list(_real_source_fixture())
    fixture[1]["checkpoint_identifier"] = ""
    assert gate.classify_real_source_gate(
        *fixture
    )["classification"] == "INCOMPLETE"

    fixture = list(_real_source_fixture())
    fixture[3][1]["process"].update(
        fixture[3][0]["process"]
    )
    assert gate.classify_real_source_gate(
        *fixture
    )["classification"] == "INCOMPLETE"

    fixture = list(_real_source_fixture())
    fixture[3][0]["process"]["returncode"] = 2
    assert gate.classify_real_source_gate(
        *fixture
    )["classification"] == "INCOMPLETE"

    fixture = list(_real_source_fixture())
    fixture[3][0]["elapsed_s"] = math.nan
    assert gate.classify_real_source_gate(
        *fixture
    )["classification"] == "INCOMPLETE"

    fixture = list(_real_source_fixture())
    fixture[1]["hyperparameters"]["max_draft_tokens"] = 16
    assert gate.classify_real_source_gate(
        *fixture
    )["classification"] == "INCOMPLETE"


def _write_real_driver_inputs(out_dir: Path):
    fixture = _real_source_fixture()
    source_evidence = {
        "schema_version": 1,
        "base_commit": "1" * 40,
        "dirty": False,
        "tree_sha256": "a" * 64,
    }
    source_preflight = {
        "schema_version": 1,
        "source_tree_sha256": "a" * 64,
        "model_identifier": "Qwen3-0.6B",
        "torch": "2.4.1",
        "cuda": "12.1",
        "flash_attn": "2.6.3",
        "gpu": "Synthetic GPU",
        "bf16_supported": True,
    }
    paths = {
        "source_evidence": out_dir / "source_evidence.input.json",
        "source_patch": out_dir / "source.input.patch",
        "source_preflight": out_dir / "source_preflight.input.json",
        "draft_source": out_dir / "draft_source.input.json",
        "prompt_bank": out_dir / "prompt_bank.input.json",
    }
    paths["source_evidence"].write_text(json.dumps(source_evidence))
    paths["source_patch"].write_bytes(b"")
    paths["source_preflight"].write_text(json.dumps(source_preflight))
    paths["draft_source"].write_text(json.dumps(fixture[1]))
    paths["prompt_bank"].write_text(json.dumps(fixture[2]))
    return paths


def _fake_real_process(calls, failed_keys=()):
    failed_keys = set(failed_keys)

    def run_process(
        *,
        python_bin,
        model_path,
        policy,
        prompt,
        draft_source,
        repetitions,
        warmup_repetitions,
        out_path,
        log_dir,
    ):
        del (
            python_bin,
            model_path,
            draft_source,
            repetitions,
            warmup_repetitions,
            log_dir,
        )
        key = (prompt["prompt_id"], policy)
        calls.append(key)
        port = 40000 + len(calls) * 2
        process = {
            "returncode": 2 if key in failed_keys else 0,
            "tinyvllm_dist_port": port,
            "master_port": port + 1,
        }
        if key in failed_keys:
            return None, process
        elapsed_s = {
            "baseline": 1.0,
            "source_always_native": 0.97,
            "source_routed_native": 0.90,
        }[policy]
        event = None
        router_event = None
        if policy == "source_routed_native":
            route = (
                "baseline_incompatible"
                if prompt["bucket"] == "low_match"
                else "native_multi_token"
            )
            event = {
                "proposal_elapsed_ms": 1.0,
                "proposed_count": 4,
                "accepted_count": 3,
                "rejected_count": 1,
                "target_forward_count": 1,
                "baseline_target_forward_count": 4,
                "accepted_kv_rematerialization": {
                    "decode_calls": 0,
                    "rematerialized_tokens": [],
                    "elapsed_ms": 0.0,
                },
                "accepted_kv_copy_calls": 0,
                "accepted_kv_replay_calls": 0,
            }
            router_event = {
                "route": route,
                "route_fallback_reason": (
                    "fixture incompatibility"
                    if route != "native_multi_token"
                    else None
                ),
            }
        payload = {
            "prompt_id": prompt["prompt_id"],
            "bucket": prompt["bucket"],
            "policy": policy,
            "status": "PASS",
            "elapsed_s": elapsed_s,
            "output_tokens": 32,
            "output_tokens_per_s": 32.0 / elapsed_s,
            "output_token_sha256": (
                f"output-{prompt['prompt_id']}"
            ),
            "event": event,
            "router_event": router_event,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload))
        return payload, process

    return run_process


def test_real_driver_runs_three_policies_and_resumes_failed_only():
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        paths = _write_real_driver_inputs(out_dir)
        failed_key = ("natural-1", "source_routed_native")
        calls = []
        original = gate._real_policy_process
        gate._real_policy_process = _fake_real_process(
            calls,
            failed_keys={failed_key},
        )
        try:
            gate.run_real_source_gate(
                out_dir=out_dir,
                python_bin="python3",
                model_path="/model",
                source_evidence_path=paths["source_evidence"],
                source_patch_path=paths["source_patch"],
                source_preflight_path=paths["source_preflight"],
                draft_source_path=paths["draft_source"],
                prompt_bank_path=paths["prompt_bank"],
                host="synthetic-host",
                run_tag="real-driver-test",
                repetitions=3,
                warmup_repetitions=1,
                prompt_limit=2,
            )
            first_expected = [
                (prompt_id, policy)
                for prompt_id in ("natural-1", "code-1")
                for policy in gate.REAL_POLICIES
            ]
            assert calls == first_expected
            resume_calls = []
            gate._real_policy_process = _fake_real_process(
                resume_calls
            )
            result = gate.run_real_source_gate(
                out_dir=out_dir,
                python_bin="python3",
                model_path="/model",
                source_evidence_path=paths["source_evidence"],
                source_patch_path=paths["source_patch"],
                source_preflight_path=paths["source_preflight"],
                draft_source_path=paths["draft_source"],
                prompt_bank_path=paths["prompt_bank"],
                host="synthetic-host",
                run_tag="real-driver-test",
                repetitions=3,
                warmup_repetitions=1,
                resume=True,
                prompt_limit=2,
            )
        finally:
            gate._real_policy_process = original

    assert resume_calls == [failed_key]
    assert len(result["case_rows"]) == 6
    assert result["summary"]["classification"] == "INCOMPLETE"


def test_validate_real_input_checks_prompt_hash():
    fixture = list(_real_source_fixture())
    result = gate.validate_real_input(fixture[1], fixture[2])
    assert result == {
        "status": "PASS",
        "source_name": "fixture-learned-drafter",
        "prompt_bank_sha256": fixture[1]["prompt_bank_sha256"],
    }
    fixture[2]["prompts"][0]["prompt"] = "tampered"
    try:
        gate.validate_real_input(fixture[1], fixture[2])
    except ValueError as exc:
        assert "prompt bank" in str(exc)
    else:
        raise AssertionError("tampered prompt bank was accepted")


def test_real_process_command_marks_real_source_stage():
    fixture = _real_source_fixture()
    draft_source = dict(fixture[1])
    draft_source["runtime_adapter"] = "ngram"
    calls = []
    original = gate.subprocess.run

    class Completed:
        returncode = 2
        stdout = ""
        stderr = "fixture stop"

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return Completed()

    gate.subprocess.run = fake_run
    try:
        with tempfile.TemporaryDirectory() as tmp:
            payload, process = gate._real_policy_process(
                python_bin="python3",
                model_path="/model",
                policy="source_routed_native",
                prompt=fixture[2]["prompts"][0],
                draft_source=draft_source,
                repetitions=3,
                warmup_repetitions=1,
                out_path=Path(tmp) / "raw.json",
                log_dir=Path(tmp) / "logs",
            )
    finally:
        gate.subprocess.run = original

    assert payload is None
    assert process["returncode"] == 2
    command = calls[0][0]
    assert command.count("--prompt") == 3
    assert command[command.index("--gate-stage") + 1] == "real-source"
    assert (
        command[command.index("--draft-construction") + 1]
        == "real_source"
    )
    assert "--allow-incompatible-fallback" in command
    assert (
        command[command.index("--warmup-repetitions") + 1]
        == "1"
    )


def test_validate_real_input_cli_parses_both_manifests():
    fixture = _real_source_fixture()
    with tempfile.TemporaryDirectory() as tmp:
        draft_path = Path(tmp) / "draft_source.json"
        prompt_path = Path(tmp) / "prompt_bank.json"
        draft_path.write_text(json.dumps(fixture[1]))
        prompt_path.write_text(json.dumps(fixture[2]))
        original = sys.argv
        try:
            sys.argv = [
                "speculation_router_gate.py",
                "validate-real-input",
                "--draft-source",
                str(draft_path),
                "--prompt-bank",
                str(prompt_path),
            ]
            args = gate._parse_args()
        finally:
            sys.argv = original
    assert args.command == "validate-real-input"
    assert args.draft_source == draft_path
    assert args.prompt_bank == prompt_path


def main():
    test_complete_controlled_evidence_is_ready()
    test_no_profitable_region_is_no_go()
    test_short_route_mutation_is_no_go()
    test_native_replay_is_no_go()
    test_semantic_mismatch_is_no_go()
    test_missing_row_is_incomplete()
    test_failed_process_is_incomplete()
    test_duplicate_port_pair_is_incomplete()
    test_nonfinite_performance_is_incomplete()
    test_lifecycle_regression_is_no_go()
    test_controlled_driver_runs_deterministic_prefix_with_unique_ports()
    test_controlled_driver_resume_retries_only_failed_row()
    test_controlled_materialization_rejects_non_target_derived_label()
    test_short_route_exactness_uses_baseline_reference()
    test_compact_row_promotes_scalars_without_numeric_arrays()
    test_real_source_manifest_rejects_non_real_sources()
    test_complete_real_source_evidence_is_go()
    test_real_source_performance_and_route_failures_are_no_go()
    test_real_source_missing_identity_or_process_is_incomplete()
    test_real_driver_runs_three_policies_and_resumes_failed_only()
    test_validate_real_input_checks_prompt_hash()
    test_real_process_command_marks_real_source_stage()
    test_validate_real_input_cli_parses_both_manifests()
    print("speculation router gate tests passed")


if __name__ == "__main__":
    main()
