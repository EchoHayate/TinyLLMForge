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


executor = _load(
    "qwen35_tp4_engine_executor_for_reference_verifier_test",
    "qwen35_tp4_engine_correctness_executor.py",
)
reference = _load(
    "qwen35_tp4_engine_reference_tokens_for_verifier_test",
    "qwen35_tp4_engine_reference_tokens.py",
)
verifier = _load(
    "verify_qwen35_tp4_engine_reference_tokens",
    "verify_qwen35_tp4_engine_reference_tokens.py",
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _token_sha256(tokens):
    return hashlib.sha256(
        json.dumps(
            list(tokens),
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _write(path, payload):
    path.write_text(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )


def _authority(root):
    run = root / "run"
    run.mkdir()
    payloads = executor.build_scenario_payloads()
    rows = []
    for scenario in reference.REFERENCE_SCENARIOS:
        payload = payloads[scenario]
        prompt = (
            payload["source_prompt_token_ids"]
            if scenario == "publish_source"
            else payload["request_prompt_token_ids"]
        )
        generated = payload["generated_tokens"]
        rows.append({
            "scenario": scenario,
            "prompt_token_count": len(prompt),
            "prompt_token_ids_sha256": _token_sha256(prompt),
            "generated_tokens": generated,
            "output_token_ids": list(range(generated)),
        })
    result = {
        "schema_version": reference.SCHEMA_VERSION,
        "classification": "PASS",
        "reference_backend": reference.REFERENCE_BACKEND,
        "generation_policy": dict(reference.GENERATION_POLICY),
        "model_manifest_sha256": "a" * 64,
        "source_tree_sha256": "b" * 64,
        "workload_manifest_sha256": "c" * 64,
        "rows": rows,
    }
    _write(run / "reference_tokens.json", result)
    _write(
        run / "source_manifest.json",
        {
            "schema_version": reference.SCHEMA_VERSION,
            "model_manifest_sha256": "a" * 64,
            "source_tree_sha256": "b" * 64,
            "workload_manifest_sha256": "c" * 64,
            "files": {
                "reference_tokens.json": _sha256(
                    run / "reference_tokens.json"
                ),
            },
        },
    )
    return run


def test_verifier_recomputes_frozen_prompt_matrix_and_writes_outside_run():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        run = _authority(root)
        result = verifier.verify_run(run)
        assert result == {
            "schema_version": reference.SCHEMA_VERSION,
            "classification": "PASS",
            "model_manifest_sha256": "a" * 64,
            "source_tree_sha256": "b" * 64,
            "workload_manifest_sha256": "c" * 64,
            "reference_tokens_sha256": _sha256(
                run / "reference_tokens.json"
            ),
            "source_manifest_sha256": _sha256(
                run / "source_manifest.json"
            ),
            "scenario_count": 5,
        }
        output = root / "verified.json"
        assert verifier.verify_and_write(
            run,
            output_path=output,
        ) == result
        assert json.loads(output.read_text()) == result


def test_verifier_rejects_tamper_and_frozen_prompt_mismatch():
    with tempfile.TemporaryDirectory() as temporary:
        run = _authority(Path(temporary))
        payload = json.loads(
            (run / "reference_tokens.json").read_text()
        )
        payload["rows"][0]["prompt_token_ids_sha256"] = "f" * 64
        _write(run / "reference_tokens.json", payload)
        manifest = json.loads(
            (run / "source_manifest.json").read_text()
        )
        manifest["files"]["reference_tokens.json"] = _sha256(
            run / "reference_tokens.json"
        )
        _write(run / "source_manifest.json", manifest)
        try:
            verifier.verify_run(run)
        except verifier.VerificationError as error:
            assert "prompt" in str(error)
        else:
            raise AssertionError("frozen prompt mismatch was accepted")


def test_verifier_rejects_extra_file_and_in_run_output():
    with tempfile.TemporaryDirectory() as temporary:
        run = _authority(Path(temporary))
        (run / "extra.json").write_text("{}\n")
        try:
            verifier.verify_run(run)
        except verifier.VerificationError as error:
            assert "inventory" in str(error)
        else:
            raise AssertionError("extra reference artifact was accepted")

    with tempfile.TemporaryDirectory() as temporary:
        run = _authority(Path(temporary))
        try:
            verifier.verify_and_write(
                run,
                output_path=run / "verification.json",
            )
        except ValueError as error:
            assert "outside" in str(error)
        else:
            raise AssertionError("in-run verification output was accepted")


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 TP4 Engine reference verifier tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
