from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace


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
    "qwen35_tp4_hybrid_prefix_benchmark_v2_contract_for_adapter_test",
    "qwen35_tp4_hybrid_prefix_benchmark_v2_contract.py",
)
adapter = _load(
    "qwen35_tp4_hybrid_prefix_benchmark_v2_engine_adapter",
    "qwen35_tp4_hybrid_prefix_benchmark_v2_engine_adapter.py",
)


MATCHED_FIELDS = (
    "model_dir",
    "tokenizer_dir",
    "tensor_parallel_size",
    "sampling",
    "prompt",
    "concurrency",
    "kv_capacity_bytes",
    "repetitions",
    "source_tree_sha256",
    "gpu_indices",
)


def _case(profile):
    return next(
        case
        for case in contract.build_case_matrix()
        if (
            case.profile == profile
            and case.workload == "w1_medium_reuse"
            and case.phase == "measured"
            and case.repetition == 0
        )
    )


def _authorized():
    return {
        "model_dir": "/models/qwen35",
        "tokenizer_dir": "/models/qwen35",
        "source_tree_sha256": "a" * 64,
        "gpu_indices": [2, 4, 5, 6],
    }


class FakeEngine:

    def __init__(self):
        self.configure_calls = []

    def configure_qwen35_hybrid_prefix_publication_runtime(self, **kwargs):
        self.configure_calls.append(kwargs)
        return {"configured": True}


def test_profile_configurations_differ_only_by_hybrid_prefix_selection():
    configurations = {
        profile: adapter.build_profile_configuration(
            profile=profile,
            case=_case(profile),
            workload_payload=contract.workload_payload(
                "w1_medium_reuse"
            ),
            authorized=_authorized(),
        )
        for profile in contract.PROFILES
    }

    assert {
        profile: {
            field: configuration[field]
            for field in MATCHED_FIELDS
        }
        for profile, configuration in configurations.items()
    } == {
        profile: {
            field: configurations["recompute"][field]
            for field in MATCHED_FIELDS
        }
        for profile in contract.PROFILES
    }
    assert {
        profile: {
            "hybrid_prefix_enabled": configuration[
                "hybrid_prefix_enabled"
            ],
            "representation": configuration["representation"],
        }
        for profile, configuration in configurations.items()
    } == {
        "recompute": {
            "hybrid_prefix_enabled": False,
            "representation": None,
        },
        "exact_restore": {
            "hybrid_prefix_enabled": True,
            "representation": "exact_restore",
        },
        "recurrent_int8_per_row": {
            "hybrid_prefix_enabled": True,
            "representation": "recurrent_int8_per_row",
        },
    }
    assert all(
        set(configuration) == set(configurations["recompute"])
        for configuration in configurations.values()
    )
    allowed_differences = {
        "hybrid_prefix_enabled",
        "representation",
    }
    differing_fields = {
        field
        for field in configurations["recompute"]
        if len({
            _freeze(configuration[field])
            for configuration in configurations.values()
        }) > 1
    }
    assert differing_fields == allowed_differences
    for field in set(configurations["recompute"]) - allowed_differences:
        assert len({
            _freeze(configuration[field])
            for configuration in configurations.values()
        }) == 1, field


def _freeze(value):
    if isinstance(value, dict):
        return tuple(
            (key, _freeze(item))
            for key, item in sorted(value.items())
        )
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def test_int8_adapter_configures_publication_runtime_with_p2_representation():
    engine = FakeEngine()
    runtime = adapter.BenchmarkEngineAdapter(
        configuration={
            "profile": "recurrent_int8_per_row",
            "hybrid_prefix_enabled": True,
            "representation": "recurrent_int8_per_row",
        },
        authorized=_authorized(),
        engine_factory=lambda configuration, authorized: engine,
    )

    runtime.configure_hybrid_prefix_publication_runtime(
        model_fingerprint="model-fingerprint",
        max_entries=16,
        max_bytes=2 * 1024**3,
        timeout_s=120.0,
    )

    assert engine.configure_calls == [{
        "model_fingerprint": "model-fingerprint",
        "max_entries": 16,
        "max_bytes": 2 * 1024**3,
        "timeout_s": 120.0,
        "representation": "recurrent_int8_per_row",
    }]


def test_correctness_configuration_is_serial_for_every_profile():
    for profile in contract.PROFILES:
        case = SimpleNamespace(
            **{
                **_case(profile).__dict__,
                "phase": "correctness",
                "concurrency": contract.CORRECTNESS_CONCURRENCY,
            }
        )
        configuration = adapter.build_profile_configuration(
            profile=profile,
            case=case,
            workload_payload=contract.workload_payload(
                "w1_medium_reuse"
            ),
            authorized=_authorized(),
        )
        assert configuration["concurrency"] == 1


def test_w3_configuration_uses_only_frozen_batched_fanout_payload():
    case = next(
        case
        for case in contract.build_case_matrix()
        if (
            case.profile == "recurrent_int8_per_row"
            and case.workload == "w3_batched_fanout"
            and case.phase == "measured"
            and case.repetition == 0
        )
    )
    payload = contract.workload_payload("w3_batched_fanout")

    configuration = adapter.build_profile_configuration(
        profile=case.profile,
        case=case,
        workload_payload=payload,
        authorized=_authorized(),
    )

    assert configuration["prompt"] == payload
    assert configuration["prompt"]["spec"]["kind"] == "batched_reuse"
    assert len(configuration["prompt"]["continuations"]) == 8
    assert configuration["concurrency"] == 8


def test_profile_configuration_rejects_profile_case_mismatch():
    case = _case("recompute")

    try:
        adapter.build_profile_configuration(
            profile="exact_restore",
            case=case,
            workload_payload=contract.workload_payload(case.workload),
            authorized=_authorized(),
        )
    except ValueError as error:
        assert "profile" in str(error).lower()
    else:
        raise AssertionError("profile/case mismatch was accepted")


def test_profile_configuration_rejects_noncanonical_gpu_assignment():
    case = _case("recompute")
    authorized = _authorized()
    authorized["gpu_indices"] = [2, 4, 5, 7]

    try:
        adapter.build_profile_configuration(
            profile=case.profile,
            case=case,
            workload_payload=contract.workload_payload(case.workload),
            authorized=authorized,
        )
    except ValueError as error:
        assert "gpu" in str(error).lower()
    else:
        raise AssertionError("noncanonical GPU assignment was accepted")


def test_profile_configuration_rejects_malformed_source_sha256():
    case = _case("recompute")
    authorized = _authorized()
    authorized["source_tree_sha256"] = "not-a-sha256"

    try:
        adapter.build_profile_configuration(
            profile=case.profile,
            case=case,
            workload_payload=contract.workload_payload(case.workload),
            authorized=authorized,
        )
    except ValueError as error:
        assert "sha" in str(error).lower()
    else:
        raise AssertionError("malformed source SHA256 was accepted")


def test_profile_configuration_rejects_empty_model_or_tokenizer_path():
    case = _case("recompute")
    for field in ("model_dir", "tokenizer_dir"):
        authorized = _authorized()
        authorized[field] = ""

        try:
            adapter.build_profile_configuration(
                profile=case.profile,
                case=case,
                workload_payload=contract.workload_payload(case.workload),
                authorized=authorized,
            )
        except ValueError as error:
            assert field.replace("_", " ") in str(error).lower()
        else:
            raise AssertionError(f"empty {field} was accepted")


def test_profile_configuration_rejects_workload_payload_drift():
    case = _case("recompute")
    payload = copy.deepcopy(contract.workload_payload(case.workload))
    payload["continuations"][0]["suffix_token_ids"][0] += 1

    try:
        adapter.build_profile_configuration(
            profile=case.profile,
            case=case,
            workload_payload=payload,
            authorized=_authorized(),
        )
    except ValueError as error:
        assert "workload" in str(error).lower()
    else:
        raise AssertionError("workload payload drift was accepted")


def test_profile_configuration_uses_frozen_capacity_and_repetition_values():
    case = _case("recompute")

    configuration = adapter.build_profile_configuration(
        profile=case.profile,
        case=case,
        workload_payload=contract.workload_payload(case.workload),
        authorized=_authorized(),
    )

    assert configuration["kv_capacity_bytes"] == 64 * 256
    assert configuration["repetitions"] == 1


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 TP4 hybrid-prefix benchmark v2 Engine adapter tests "
        f"passed ({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
