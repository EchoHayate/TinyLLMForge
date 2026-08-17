from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT_PATH = (
    ROOT
    / "tools/qwen35_constructed_engine_model_runner_ownership_preflight.py"
)
VERIFIER_PATH = (
    ROOT
    / "tools/verify_qwen35_constructed_engine_model_runner_ownership_gate.py"
)
ORACLE_PATH = (
    ROOT
    / "experiments/qwen35_hybrid_state/"
    "qwen35-tp4-real-candidate-replay-20260728-145713/"
    "tp4_real_candidate_provenance_oracle.json"
)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _ledger(preflight):
    rows = []
    ranked = {
        "dist.init_process_group",
        "torch.cuda.set_device",
        "Qwen3ForCausalLM",
        "ModelRunner.warmup_model",
        "ModelRunner.allocate_kv_cache",
        "dist.barrier",
    }
    for dependency, count in (
        preflight.EXPECTED_CONSTRUCTOR_CALL_COUNTS.items()
    ):
        for index in range(count):
            rows.append({
                "sequence": len(rows),
                "dependency": dependency,
                "rank": index if dependency in ranked else None,
                "arguments": {},
                "result_identity": f"fixture.{dependency}:{index}",
            })
    return rows


def _build_run(directory):
    preflight = _load(PREFLIGHT_PATH, "constructed_fixture_preflight")
    oracle = json.loads(ORACLE_PATH.read_text())
    source_hashes = dict(oracle["source_file_sha256"])
    for name in (
        set(preflight.EXPECTED_FILE_SHA256)
        | set(preflight.CONSTRUCTOR_RUNTIME_FILE_SHA256)
        | set(preflight.DIRECT_GATE_FILE_SHA256)
    ):
        source_hashes[name] = hashlib.sha256(
            (ROOT / name).read_bytes()
        ).hexdigest()
    for name, path in (
        (
            "tools/qwen35_constructed_engine_model_runner_ownership_preflight.py",
            PREFLIGHT_PATH,
        ),
        (
            "tools/verify_qwen35_constructed_engine_model_runner_ownership_gate.py",
            VERIFIER_PATH,
        ),
    ):
        source_hashes[name] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    source_hashes = dict(sorted(source_hashes.items()))
    payloads = []
    for rank, pristine in enumerate(oracle["producer_rows"]):
        payloads.append({
            "rank": rank,
            "binding_hash_count": 320,
            "binding_destination_sha256": pristine[
                "binding_destination_sha256"
            ],
            "phase_hash_count": 26,
            "phase_destination_sha256": pristine[
                "phase_destination_sha256"
            ],
            "aggregate_destination_sha256": pristine[
                "aggregate_destination_sha256"
            ],
            "alias_groups": pristine["alias_groups"],
            "loader_stats": pristine["loader_stats"],
            "anticipated_identity": {
                "model_fingerprint": pristine[
                    "model_manifest_sha256"
                ],
                "layout_fingerprint": pristine[
                    "layout_fingerprint"
                ],
                "dtype": pristine["dtype"],
            },
            "transfer_evidence": {
                "candidate_published": True,
                "candidate_bound_before_engine_dispatch": False,
            },
        })
    bound_rows = tuple(
        {
            "participant_id": rank,
            "operation": "bind_loaded_checkpoint_candidate",
            "status": "bound",
            "model_fingerprint": (
                preflight.EXPECTED_FILE_SHA256
                and oracle["model_manifest_sha256"]
            ),
            "layout_fingerprint": oracle["producer_rows"][rank][
                "layout_fingerprint"
            ],
            "dtype": "bfloat16",
            "detail": "",
        }
        for rank in range(4)
    )
    envelope = {
        "command_id": 0,
        "method_name": (
            "bind_published_qwen35_loaded_checkpoint_candidate"
        ),
        "args": [],
        "requires_ack": True,
    }
    acknowledgements = [
        {
            "command_id": 0,
            "rank": rank,
            "status": "ok",
            "result": bound_rows[rank],
            "error_type": "",
            "error_detail": "",
        }
        for rank in (1, 2, 3)
    ]
    constructor_ledger = _ledger(preflight)
    smoke = {
        "status": "PASS",
        "source_contract": {
            "files": dict(preflight.EXPECTED_FILE_SHA256),
            "methods": dict(preflight.EXPECTED_METHOD_SHA256),
            "constructor_signatures": dict(
                preflight.EXPECTED_CONSTRUCTOR_SIGNATURES
            ),
            "forbidden_execution_forms": {
                "object_new": False,
                "constructor_ast_compile": False,
                "subclass_construction": False,
                "class_replacement": False,
            },
        },
        "constructor_evidence": {
            "engine_constructor_count": 1,
            "runner_constructor_count": 4,
            "runner_constructor_ranks": [0, 1, 2, 3],
            "dependency_call_counts": dict(
                preflight.EXPECTED_CONSTRUCTOR_CALL_COUNTS
            ),
            "original_dependency_identities": {"dependency": 1},
            "restored_dependency_identities": {"dependency": 1},
            "restoration_complete": True,
        },
        "class_identity": {
            "engine_module": (
                "_qwen35_constructed_runtime_production_llm_engine"
            ),
            "engine_qualname": "LLMEngine",
            "engine_exact_class": True,
            "runner_module": (
                "_qwen35_constructed_runtime_production_model_runner"
            ),
            "runner_qualname": "ModelRunner",
            "runner_exact_class_by_rank": [True, True, True, True],
        },
        "constructor_ledger": constructor_ledger,
        "rank_payloads": payloads,
        "first_binding": {
            "rows": bound_rows,
            "configuration": (
                oracle["model_manifest_sha256"],
                oracle["producer_rows"][0]["layout_fingerprint"],
                "bfloat16",
                0.25,
            ),
            "command_envelope": envelope,
            "worker_acknowledgements": acknowledgements,
            "zero_payload_command": True,
            "exact_repeat_zero_dispatch": None,
        },
        "repeat_binding": {
            "rows": bound_rows,
            "configuration": (
                oracle["model_manifest_sha256"],
                oracle["producer_rows"][0]["layout_fingerprint"],
                "bfloat16",
                0.25,
            ),
            "command_envelope": envelope,
            "worker_acknowledgements": acknowledgements,
            "zero_payload_command": True,
            "exact_repeat_zero_dispatch": True,
        },
        "transport_restoration": {
            "module_name": "tinyvllm.engine.model_runner_command_ack",
            "restored": True,
            "envelope_class_identity": True,
        },
        "forbidden_counters": {
            name: 0 for name in preflight.FORBIDDEN_COUNTER_NAMES
        },
        "cuda_initialized_after": False,
    }
    cleanup = {
        "release_rank_order": [3, 2, 1, 0],
        "all_selected_destinations_zero_after_clear": True,
        "non_selected_tensors_unchanged": True,
        "tensor_identity_preserved": True,
        "pool_unchanged": True,
        "all_inert_resources_closed": True,
        "production_exit_call_count": 0,
        "collected_private_objects": {
            "engine": True,
            **{f"runner_{rank}": True for rank in range(4)},
            **{f"candidate_{rank}": True for rank in range(4)},
        },
        "all_private_objects_collected": True,
    }
    artifact = preflight.build_constructed_runtime_artifact(
        run_tag="qwen35-constructed-test-20260728-180200",
        smoke=smoke,
        cleanup=cleanup,
        memory={
            "process_before": {
                "vmrss_kib": 300_000,
                "vmhwm_kib": 300_000,
            },
            "process_ready": {
                "vmrss_kib": 4_000_000,
                "vmhwm_kib": 6_000_000,
            },
            "process_after_cleanup": {
                "vmrss_kib": 800_000,
                "vmhwm_kib": 6_000_000,
            },
            "host_before": {"mem_available_kib": 30_000_000},
            "host_ready": {"mem_available_kib": 22_000_000},
        },
        source_file_sha256=source_hashes,
        prerequisite_oracle_sha256=(
            preflight._sha256(ORACLE_PATH.read_bytes())
        ),
        observed_user="sitian",
        observed_hostname="n232-195-203",
    )
    run_dir = Path(directory) / artifact["run_tag"]
    preflight.finalize_constructed_runtime_artifact(
        run_dir=run_dir,
        artifact=artifact,
        remote_target="sitian@10.232.195.203",
        remote_python=(
            "/data00/home/sitian/sitian-workspace01/tllm/env/bin/python"
        ),
    )
    return run_dir


def test_generated_artifact_passes_independent_verification():
    verifier = _load(VERIFIER_PATH, "constructed_verifier")
    with tempfile.TemporaryDirectory() as directory:
        run_dir = _build_run(directory)
        result = verifier.verify_run(
            run_dir,
            source_root=ROOT,
            prerequisite_oracle=ORACLE_PATH,
        )
    assert result["status"] == "PASS"
    assert result["checks"] >= 200


def _tamper(mutator, *, extra_file=False):
    verifier = _load(
        VERIFIER_PATH,
        "constructed_verifier_tamper",
    )
    with tempfile.TemporaryDirectory() as directory:
        run_dir = _build_run(directory)
        result_path = (
            run_dir
            / "constructed_engine_model_runner_ownership.json"
        )
        manifest_path = run_dir / "source_manifest.json"
        record = json.loads(result_path.read_text())
        manifest = json.loads(manifest_path.read_text())
        mutator(record)
        result_path.write_text(
            json.dumps(record, sort_keys=True, separators=(",", ":"))
            + "\n"
        )
        manifest["result_sha256"] = hashlib.sha256(
            result_path.read_bytes()
        ).hexdigest()
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":"))
            + "\n"
        )
        if extra_file:
            (run_dir / "extra.txt").write_text("tamper")
        try:
            verifier.verify_run(
                run_dir,
                source_root=ROOT,
                prerequisite_oracle=ORACLE_PATH,
            )
        except verifier.VerificationError:
            return
    raise AssertionError("tampered constructed artifact was accepted")


def test_verifier_rejects_tamper_matrix():
    mutations = [
        lambda row: row["class_identity"].__setitem__(
            "engine_exact_class",
            False,
        ),
        lambda row: row["constructor_replacement_allowlist"].append(
            "extra.replacement"
        ),
        lambda row: row["constructor_evidence"].__setitem__(
            "engine_constructor_count",
            2,
        ),
        lambda row: row["constructor_ledger"][0].__setitem__(
            "sequence",
            9,
        ),
        lambda row: row["rank_payloads"][2][
            "binding_destination_sha256"
        ].__setitem__(0, "0" * 64),
        lambda row: row["first_binding"]["command_envelope"].__setitem__(
            "args",
            ["forbidden"],
        ),
        lambda row: row["first_binding"][
            "worker_acknowledgements"
        ].pop(),
        lambda row: row["first_binding"].__setitem__(
            "configuration",
            ["bad"],
        ),
        lambda row: row["repeat_binding"].__setitem__(
            "exact_repeat_zero_dispatch",
            False,
        ),
        lambda row: row["forbidden_counters"].__setitem__(
            "inference",
            1,
        ),
        lambda row: row["memory"].__setitem__(
            "process_total_vmhwm_increment_kib",
            1,
        ),
        lambda row: row["cleanup"][
            "collected_private_objects"
        ].__setitem__("engine", False),
    ]
    for mutation in mutations:
        _tamper(mutation)
    _tamper(lambda row: None, extra_file=True)


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "constructed ownership verifier tests "
        f"passed ({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
