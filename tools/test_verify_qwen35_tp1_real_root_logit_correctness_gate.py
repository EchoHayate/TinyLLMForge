from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

import torch


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT_PATH = (
    ROOT / "tools/qwen35_tp1_real_root_logit_correctness_preflight.py"
)
CONTRACT_PATH = (
    ROOT / "tools/qwen35_tp1_real_root_logit_correctness_contract.py"
)
VERIFIER_PATH = (
    ROOT / "tools/verify_qwen35_tp1_real_root_logit_correctness_gate.py"
)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, os.fspath(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


preflight = _load("qwen35_tp1_artifact_preflight_under_test", PREFLIGHT_PATH)
contract = _load("qwen35_tp1_artifact_contract_under_test", CONTRACT_PATH)
verifier = _load("qwen35_tp1_artifact_verifier_under_test", VERIFIER_PATH)


def _sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def _canonical(value):
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _source_manifest():
    source_hashes = {
        "tools/qwen35_tp1_real_root_logit_correctness_contract.py": (
            "1" * 64
        ),
        "tools/qwen35_tp1_real_root_logit_correctness_preflight.py": (
            "2" * 64
        ),
        "tools/verify_qwen35_tp1_real_root_logit_correctness_gate.py": (
            "3" * 64
        ),
    }
    return {
        "schema_version": 1,
        "source_file_sha256": source_hashes,
        "source_tree_sha256": _sha256_bytes(_canonical(source_hashes)),
        "model_manifest_sha256": (
            preflight.APPROVED_MODEL_MANIFEST_SHA256
        ),
        "config_sha256": preflight.APPROVED_CONFIG_SHA256,
        "index_sha256": preflight.APPROVED_INDEX_SHA256,
        "shard_name": preflight.APPROVED_SHARD_NAME,
        "shard_size": preflight.APPROVED_SHARD_SIZE,
        "shard_sha256": preflight.APPROVED_SHARD_SHA256,
    }


def _process_row(worker, pid):
    return {
        "worker": worker,
        "pid": pid,
        "exit_code": 0,
        "model_manifest_sha256": (
            preflight.APPROVED_MODEL_MANIFEST_SHA256
        ),
        "gpu_index": 0,
        "gpu_uuid": "GPU-approved",
        "free_bytes_before": 40 * 1024**3,
        "minimum_free_bytes": 24 * 1024**3,
        "case_ids": ["p17", "p65", "synthetic"],
        "vocab_size": 32,
        "cleanup_complete": True,
        "start_timestamp": f"{worker}-start",
        "finish_timestamp": f"{worker}-finish",
        "torch_version": "2.4.1",
        "vmrss_kib": 100,
        "vmhwm_kib": 200,
        "max_memory_allocated": 1024,
        "max_memory_reserved": 2048,
        **({
            "transformers_version": "5.8.1",
            "local_files_only": True,
            "trust_remote_code": False,
            "dtype": "bfloat16",
            "attn_implementation": "eager",
            "use_cache": False,
        } if worker == "reference" else {
            "tensor_parallel_size": 1,
            "tensor_parallel_rank": 0,
            "dtype": "bfloat16",
            "recurrent_dtype": "float32",
            "engine_constructed": False,
            "model_runner_constructed": False,
            "scheduler_constructed": False,
            "sampler_constructed": False,
        }),
    }


def _state_rows():
    return [{
        "case_id": case.case_id,
        "prepare_read_only": True,
        "linear_layer_count": 18,
        "changed_component_count": 36,
        "full_attention_state_component_count": 0,
        "commit_count": 1,
        "release_zeroed": True,
        "pool_binding_released": True,
    } for case in contract.prompt_cases()]


def _tensor_maps():
    reference = {}
    native = {}
    for index, case in enumerate(contract.prompt_cases()):
        row = torch.linspace(-1.0, 1.0, 32, dtype=torch.float32)
        row = row + index
        row[5 + index] = 10.0 + index
        reference[case.case_id] = row.contiguous()
        native[case.case_id] = row.clone().contiguous()
    return reference, native


def _build_run(run_dir):
    reference, native = _tensor_maps()
    return preflight.finalize_tp1_correctness_artifact(
        run_dir=run_dir,
        run_tag="qwen35-tp1-artifact-test",
        reference_logits=reference,
        native_logits=native,
        reference_process=_process_row("reference", 101),
        native_process=_process_row("native", 202),
        state_rows=_state_rows(),
        source_manifest=_source_manifest(),
        forbidden_counters={
            "engine": 0,
            "model_runner": 0,
            "scheduler": 0,
            "sampler": 0,
            "generation": 0,
        },
    )


def _expect_error(callable_, fragment):
    try:
        callable_()
    except (ValueError, verifier.VerificationError) as exc:
        assert fragment in str(exc), str(exc)
    else:
        raise AssertionError(f"expected error containing {fragment!r}")


def test_finalizer_publishes_exact_four_file_inventory_and_verifies():
    with tempfile.TemporaryDirectory() as temporary_directory:
        run_dir = Path(temporary_directory) / "run"
        paths = _build_run(run_dir)
        assert {path.name for path in paths} == {
            "tp1_real_root_logit_correctness.json",
            "reference_logits.pt",
            "native_logits.pt",
            "source_manifest.json",
        }
        assert {path.name for path in run_dir.iterdir()} == {
            "tp1_real_root_logit_correctness.json",
            "reference_logits.pt",
            "native_logits.pt",
            "source_manifest.json",
        }
        verified = verifier.verify_run(run_dir)
        assert verified["classification"] == "PASS"
        assert verified["case_ids"] == ["p17", "p65", "synthetic"]
        assert verified["checks"] >= 100


def test_finalizer_refuses_shared_pid_cleanup_memory_state_and_forbidden_drift():
    reference, native = _tensor_maps()
    base = {
        "run_tag": "qwen35-tp1-artifact-refusal",
        "reference_logits": reference,
        "native_logits": native,
        "reference_process": _process_row("reference", 101),
        "native_process": _process_row("native", 202),
        "state_rows": _state_rows(),
        "source_manifest": _source_manifest(),
        "forbidden_counters": {
            "engine": 0,
            "model_runner": 0,
            "scheduler": 0,
            "sampler": 0,
            "generation": 0,
        },
    }

    def reject(mutator, fragment):
        with tempfile.TemporaryDirectory() as temporary_directory:
            arguments = copy.deepcopy(base)
            mutator(arguments)
            _expect_error(
                lambda: preflight.finalize_tp1_correctness_artifact(
                    run_dir=Path(temporary_directory) / "run",
                    **arguments,
                ),
                fragment,
            )

    reject(
        lambda value: value["native_process"].update({"pid": 101}),
        "separate",
    )
    reject(
        lambda value: value["reference_process"].update({
            "cleanup_complete": False,
        }),
        "cleanup",
    )
    reject(
        lambda value: value["native_process"].update({
            "free_bytes_before": 24 * 1024**3 - 1,
        }),
        "24 GiB",
    )
    reject(
        lambda value: value["state_rows"][0].update({
            "changed_component_count": 35,
        }),
        "state",
    )
    reject(
        lambda value: value["forbidden_counters"].update({"engine": 1}),
        "forbidden",
    )


def test_verifier_rejects_inventory_tensor_metric_tolerance_and_identity_tamper():
    def reject(mutator, fragment):
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir = Path(temporary_directory) / "run"
            _build_run(run_dir)
            mutator(run_dir)
            _expect_error(lambda: verifier.verify_run(run_dir), fragment)

    reject(
        lambda run_dir: (run_dir / "extra.txt").write_text("extra"),
        "inventory",
    )
    reject(
        lambda run_dir: torch.save(
            {"p17": torch.zeros(32)},
            run_dir / "native_logits.pt",
        ),
        "artifact",
    )

    def mutate_result(run_dir, mutator):
        path = run_dir / "tp1_real_root_logit_correctness.json"
        payload = json.loads(path.read_text())
        mutator(payload)
        path.write_bytes(_canonical(payload) + b"\n")
        manifest_path = run_dir / "source_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["artifacts"][path.name]["sha256"] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        manifest["artifacts"][path.name]["size"] = path.stat().st_size
        manifest_path.write_bytes(_canonical(manifest) + b"\n")

    reject(
        lambda run_dir: mutate_result(
            run_dir,
            lambda value: value["comparisons"][0].update({
                "max_abs_diff": 123.0,
            }),
        ),
        "derived metric",
    )
    reject(
        lambda run_dir: mutate_result(
            run_dir,
            lambda value: value["tolerance"].update({"atol": 1.0}),
        ),
        "tolerance",
    )
    reject(
        lambda run_dir: mutate_result(
            run_dir,
            lambda value: value["processes"]["native"].update({"pid": 101}),
        ),
        "separate",
    )
    reject(
        lambda run_dir: mutate_result(
            run_dir,
            lambda value: value.update({"classification": "NO_GO_LOGIT"}),
        ),
        "classification",
    )


def test_verifier_source_is_independent_of_producer_transformers_and_cuda():
    source = VERIFIER_PATH.read_text(encoding="utf-8")
    for forbidden in (
        "tinyvllm",
        "qwen35_tp1_real_root_logit_correctness_preflight",
        "transformers",
        "torch.cuda",
    ):
        assert forbidden not in source


def test_main():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and name != "test_main"
    ]
    for test in tests:
        test()
    print(f"PASS: {len(tests)} tests")


if __name__ == "__main__":
    test_main()
