from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


def _load(name, filename):
    path = TOOLS / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


contract = _load(
    "qwen35_tp4_cached_continuation_contract_for_verifier_test",
    "qwen35_tp4_cached_continuation_correctness_contract.py",
)
verifier = _load(
    "verify_qwen35_tp4_cached_continuation_correctness_gate",
    "verify_qwen35_tp4_cached_continuation_correctness_gate.py",
)


MODEL_MANIFEST_SHA256 = (
    "3e650a908234771c3cf1ac4e20c4d38f"
    "e69982efedaf4a3e631ad0b14aad7dd0"
)
SOURCE_TREE_SHA256 = "c" * 64


def _write_json(path, value):
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows():
    rows = []
    for workload in contract.WORKLOADS:
        spec = contract.workload_payload(workload)["spec"]
        expected_hit = workload in contract.HIT_WORKLOADS
        for request_index in range(spec["continuations"]):
            output = list(range(spec["generated_tokens"]))
            rows.append({
                "workload": workload,
                "request_index": request_index,
                "outcome": "continuation",
                "restore_hit": expected_hit,
                "restore_reason": (
                    "exact_hit"
                    if expected_hit
                    else contract.W4_EXPECTED_REASONS[request_index]
                ),
                "prompt_tokens": (
                    spec["shared_prefix_tokens"]
                    + spec["suffix_tokens"]
                ),
                "reused_tokens": (
                    spec["shared_prefix_tokens"]
                    if expected_hit
                    else 0
                ),
                "executed_prefill_tokens": (
                    spec["suffix_tokens"]
                    if expected_hit
                    else (
                        spec["shared_prefix_tokens"]
                        + spec["suffix_tokens"]
                    )
                ),
                "output_token_ids": output,
                "reference_output_token_ids": list(output),
                "logits_max_abs_diff": 0.0,
                "logits_allclose": True,
                "cache_identity_match": True,
                "rank_inventory": [0, 1, 2, 3],
                "process_group_destroyed": True,
                "owned_children_remaining": [],
            })
    return rows


def _fixture(root):
    run_dir = Path(root)
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = _rows()
    reference = {
        f"{row['workload']}:{row['request_index']}": (
            row["reference_output_token_ids"]
        )
        for row in rows
    }
    restored = {
        f"{row['workload']}:{row['request_index']}": (
            row["output_token_ids"]
        )
        for row in rows
    }
    logits = [{
        "workload": row["workload"],
        "request_index": row["request_index"],
        "max_abs_diff": row["logits_max_abs_diff"],
        "allclose": row["logits_allclose"],
    } for row in rows]
    _write_json(
        run_dir / "cached_continuation_correctness.json",
        {
            "schema_version": contract.SCHEMA_VERSION,
            "classification": "PASS",
            "model_manifest_sha256": MODEL_MANIFEST_SHA256,
            "workload_manifest_sha256": (
                contract.WORKLOAD_MANIFEST_SHA256
            ),
            "rows": rows,
        },
    )
    _write_json(run_dir / "reference_outputs.json", reference)
    _write_json(run_dir / "restored_outputs.json", restored)
    _write_json(run_dir / "registered_logits.json", logits)
    _write_json(
        run_dir / "source_manifest.json",
        {
            "schema_version": contract.SCHEMA_VERSION,
            "source_tree_sha256": SOURCE_TREE_SHA256,
            "model_manifest_sha256": MODEL_MANIFEST_SHA256,
            "workload_manifest_sha256": (
                contract.WORKLOAD_MANIFEST_SHA256
            ),
            "files": {
                name: _sha256(run_dir / name)
                for name in contract.ARTIFACT_NAMES[:-1]
            },
        },
    )
    return run_dir


def _expect_invalid(mutator, fragment):
    with tempfile.TemporaryDirectory() as temporary:
        run_dir = _fixture(temporary)
        mutator(run_dir)
        try:
            verifier.verify_run(run_dir)
        except verifier.VerificationError as error:
            assert fragment in str(error), str(error)
        else:
            raise AssertionError(f"tamper accepted: {fragment}")


def test_complete_exact_five_fixture_passes():
    with tempfile.TemporaryDirectory() as temporary:
        result = verifier.verify_run(_fixture(temporary))

    assert result["classification"] == "PASS"
    assert result["checks"]["row_count"] == 19
    assert result["checks"]["restore_hits"] == 16
    assert result["checks"]["w4_misses"] == 3


def test_extra_missing_or_symlink_artifact_is_rejected():
    _expect_invalid(
        lambda run_dir: (run_dir / "extra.json").write_text(
            "{}\n",
            encoding="utf-8",
        ),
        "artifact inventory",
    )
    _expect_invalid(
        lambda run_dir: (
            run_dir / "registered_logits.json"
        ).unlink(),
        "artifact inventory",
    )

    def symlink(run_dir):
        target = run_dir / "registered_logits.real.json"
        (run_dir / "registered_logits.json").rename(target)
        (run_dir / "registered_logits.json").symlink_to(target.name)

    _expect_invalid(symlink, "regular file")


def test_resigned_output_or_restore_tamper_is_rejected():
    def mutate_output(run_dir):
        result_path = run_dir / "cached_continuation_correctness.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["rows"][0]["output_token_ids"][-1] = 999
        _write_json(result_path, result)
        restored_path = run_dir / "restored_outputs.json"
        restored = json.loads(restored_path.read_text(encoding="utf-8"))
        restored["w1_medium_reuse:0"][-1] = 999
        _write_json(restored_path, restored)
        source_path = run_dir / "source_manifest.json"
        source = json.loads(source_path.read_text(encoding="utf-8"))
        source["files"]["cached_continuation_correctness.json"] = (
            _sha256(result_path)
        )
        source["files"]["restored_outputs.json"] = _sha256(
            restored_path
        )
        _write_json(source_path, source)

    _expect_invalid(mutate_output, "classification")

    def mutate_restore(run_dir):
        result_path = run_dir / "cached_continuation_correctness.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["rows"][0]["restore_hit"] = False
        _write_json(result_path, result)
        source_path = run_dir / "source_manifest.json"
        source = json.loads(source_path.read_text(encoding="utf-8"))
        source["files"]["cached_continuation_correctness.json"] = (
            _sha256(result_path)
        )
        _write_json(source_path, source)

    _expect_invalid(mutate_restore, "classification")


def test_source_model_and_workload_identity_tamper_is_rejected():
    def mutate(run_dir):
        path = run_dir / "source_manifest.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["workload_manifest_sha256"] = "0" * 64
        _write_json(path, payload)

    _expect_invalid(mutate, "workload manifest")


def test_independent_verification_is_written_outside_exact_five():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        run_dir = _fixture(root / "run")
        output = root / "verification" / "independent_verification.json"

        result = verifier.verify_and_write(
            run_dir,
            output_path=output,
        )
        payload = json.loads(output.read_text(encoding="utf-8"))

        assert result == payload
        assert payload["classification"] == "PASS"
        assert payload["model_manifest_sha256"] == (
            MODEL_MANIFEST_SHA256
        )
        assert set(path.name for path in run_dir.iterdir()) == set(
            contract.ARTIFACT_NAMES
        )


def test_independent_verification_refuses_run_internal_or_overwrite():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        run_dir = _fixture(root / "run")
        for output in (
            run_dir / "independent_verification.json",
            root / "existing.json",
        ):
            if output.parent != run_dir:
                output.write_text("{}\n", encoding="utf-8")
            try:
                verifier.verify_and_write(
                    run_dir,
                    output_path=output,
                )
            except ValueError as error:
                assert (
                    "outside" in str(error)
                    or "already exists" in str(error)
                )
            else:
                raise AssertionError(
                    "unsafe verification output was accepted"
                )


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 TP4 cached-continuation verifier tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
