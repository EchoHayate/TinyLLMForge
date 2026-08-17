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
    "qwen35_tp4_engine_executor_for_reference_test",
    "qwen35_tp4_engine_correctness_executor.py",
)
reference = _load(
    "qwen35_tp4_engine_reference_tokens",
    "qwen35_tp4_engine_reference_tokens.py",
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


def _write_json(path, payload):
    path.write_text(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )


def _configuration():
    return executor.ExecutorConfiguration(
        model_dir="/models/qwen35",
        model_manifest_path="/authority/model_manifest.json",
        model_manifest_sha256="a" * 64,
        source_tree_sha256="b" * 64,
        workload_manifest_path="/authority/workload_manifest.json",
        workload_manifest_sha256="c" * 64,
        model_fingerprint="qwen35-m8-authority",
        gpu_indices=(0, 1, 2, 3),
        dist_port=31001,
        master_port=31002,
        max_cache_entries=8,
        max_cache_bytes=1 << 30,
        timeout_s=600.0,
    )


def _build_authority(root):
    authority = root / "reference-authority"
    authority.mkdir()
    rows = []
    payloads = executor.build_scenario_payloads()
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
    _write_json(
        authority / "reference_tokens.json",
        {
            "schema_version": reference.SCHEMA_VERSION,
            "classification": "PASS",
            "reference_backend": (
                reference.REFERENCE_BACKEND
            ),
            "generation_policy": dict(
                reference.GENERATION_POLICY
            ),
            "model_manifest_sha256": "a" * 64,
            "source_tree_sha256": "b" * 64,
            "workload_manifest_sha256": "c" * 64,
            "rows": rows,
        },
    )
    _write_json(
        authority / "source_manifest.json",
        {
            "schema_version": reference.SCHEMA_VERSION,
            "model_manifest_sha256": "a" * 64,
            "source_tree_sha256": "b" * 64,
            "workload_manifest_sha256": "c" * 64,
            "files": {
                "reference_tokens.json": _sha256(
                    authority / "reference_tokens.json"
                ),
            },
        },
    )
    verification = root / "independent_verification.json"
    _write_json(
        verification,
        {
            "schema_version": reference.SCHEMA_VERSION,
            "classification": "PASS",
            "model_manifest_sha256": "a" * 64,
            "source_tree_sha256": "b" * 64,
            "workload_manifest_sha256": "c" * 64,
            "reference_tokens_sha256": _sha256(
                authority / "reference_tokens.json"
            ),
            "source_manifest_sha256": _sha256(
                authority / "source_manifest.json"
            ),
            "scenario_count": len(reference.REFERENCE_SCENARIOS),
        },
    )
    return authority, verification


def test_provider_requires_exact_source_bound_verified_authority():
    with tempfile.TemporaryDirectory() as temporary:
        authority, verification = _build_authority(Path(temporary))
        provider = reference.build_reference_token_provider(
            authority_dir=authority,
            verification_path=verification,
            configuration=_configuration(),
        )
        payload = executor.build_scenario_payloads()["restore_w1"]
        assert provider(
            scenario="restore_w1",
            prompt_token_ids=payload["request_prompt_token_ids"],
            generated_tokens=64,
        ) == list(range(64))

        tampered = json.loads(
            (authority / "reference_tokens.json").read_text()
        )
        tampered["rows"][0]["output_token_ids"][0] = 999
        _write_json(authority / "reference_tokens.json", tampered)
        try:
            reference.build_reference_token_provider(
                authority_dir=authority,
                verification_path=verification,
                configuration=_configuration(),
            )
        except ValueError as error:
            assert "verification" in str(error) or "hash" in str(error)
        else:
            raise AssertionError("tampered reference authority was accepted")


def test_provider_rejects_prompt_or_generation_mismatch():
    with tempfile.TemporaryDirectory() as temporary:
        authority, verification = _build_authority(Path(temporary))
        provider = reference.build_reference_token_provider(
            authority_dir=authority,
            verification_path=verification,
            configuration=_configuration(),
        )
        payload = executor.build_scenario_payloads()["publish_source"]
        source_prompt = payload["source_prompt_token_ids"]
        assert provider(
            scenario="publish_source",
            prompt_token_ids=source_prompt,
            generated_tokens=1,
        ) == [0]
        prompt = list(source_prompt)
        prompt[0] += 1
        for changes, message in (
            ({"prompt_token_ids": prompt}, "prompt"),
            ({"generated_tokens": 2}, "generated"),
            ({"scenario": "construct_and_bind"}, "scenario"),
        ):
            values = {
                "scenario": "publish_source",
                "prompt_token_ids": source_prompt,
                "generated_tokens": 1,
                **changes,
            }
            try:
                provider(**values)
            except ValueError as error:
                assert message in str(error)
            else:
                raise AssertionError(
                    f"{message} mismatch was accepted"
                )


def test_authority_inventory_and_configuration_binding_are_strict():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        authority, verification = _build_authority(root)
        (authority / "extra.json").write_text("{}\n")
        try:
            reference.build_reference_token_provider(
                authority_dir=authority,
                verification_path=verification,
                configuration=_configuration(),
            )
        except ValueError as error:
            assert "inventory" in str(error)
        else:
            raise AssertionError("extra authority artifact was accepted")

    with tempfile.TemporaryDirectory() as temporary:
        authority, verification = _build_authority(Path(temporary))
        payload = _configuration().to_payload()
        payload.pop("world_size")
        payload["gpu_indices"] = tuple(payload["gpu_indices"])
        payload["source_tree_sha256"] = "d" * 64
        mismatched = executor.ExecutorConfiguration(**payload)
        try:
            reference.build_reference_token_provider(
                authority_dir=authority,
                verification_path=verification,
                configuration=mismatched,
            )
        except ValueError as error:
            assert "source tree" in str(error)
        else:
            raise AssertionError("source identity mismatch was accepted")


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 TP4 Engine reference token tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
