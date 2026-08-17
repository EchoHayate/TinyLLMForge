from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys


def _load_contract():
    module_name = "qwen35_tp4_engine_correctness_contract"
    module = sys.modules.get(module_name)
    if module is not None:
        return module
    path = (
        Path(__file__).resolve().parent
        / "qwen35_tp4_engine_correctness_contract.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


contract = _load_contract()


def _load_benchmark_contract():
    module_name = "qwen35_tp4_hybrid_prefix_benchmark_contract"
    module = sys.modules.get(module_name)
    if module is not None:
        return module
    path = (
        Path(__file__).resolve().parent
        / "qwen35_tp4_hybrid_prefix_benchmark_contract.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


CONFIGURATION_FIELDS = (
    "model_dir",
    "model_manifest_path",
    "model_manifest_sha256",
    "source_tree_sha256",
    "workload_manifest_path",
    "workload_manifest_sha256",
    "model_fingerprint",
    "gpu_indices",
    "world_size",
    "dist_port",
    "master_port",
    "max_cache_entries",
    "max_cache_bytes",
    "timeout_s",
)

_SCENARIO_ACTIONS = {
    "construct_and_bind": (
        "construct_engine",
        "begin_observation",
        "configure_exact_restore",
        "verify_rank_bindings",
        "close_engine",
        "verify_cleanup",
    ),
    "publish_source": (
        "construct_engine",
        "configure_exact_restore",
        "verify_rank_bindings",
        "begin_observation",
        "submit_source_request",
        "run_to_completion",
        "verify_publication_commit",
        "snapshot_cache",
        "close_engine",
        "verify_cleanup",
    ),
    "restore_w1": (
        "construct_engine",
        "configure_exact_restore",
        "verify_rank_bindings",
        "seed_source_fixture",
        "begin_observation",
        "submit_cached_continuation",
        "run_to_completion",
        "drain_release_events",
        "snapshot_cache",
        "close_engine",
        "verify_cleanup",
    ),
    "miss_w4_token": (
        "construct_engine",
        "configure_exact_restore",
        "verify_rank_bindings",
        "seed_source_fixture",
        "begin_observation",
        "submit_token_mismatch",
        "run_to_completion",
        "snapshot_cache",
        "close_engine",
        "verify_cleanup",
    ),
    "miss_w4_stale": (
        "construct_engine",
        "configure_exact_restore",
        "verify_rank_bindings",
        "seed_source_fixture",
        "invalidate_block_generation",
        "begin_observation",
        "submit_cached_continuation",
        "run_to_completion",
        "snapshot_cache",
        "close_engine",
        "verify_cleanup",
    ),
    "miss_w4_clear": (
        "construct_engine",
        "configure_exact_restore",
        "verify_rank_bindings",
        "seed_source_fixture",
        "clear_reusable_cache",
        "begin_observation",
        "submit_cached_continuation",
        "run_to_completion",
        "snapshot_cache",
        "close_engine",
        "verify_cleanup",
    ),
}


def _require_absolute_path(value, label):
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError(f"{label} must be an absolute path")
    return value


def _require_sha256(value, label):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(
            character not in "0123456789abcdef"
            for character in value
        )
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _require_positive_integer(value, label):
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError(f"{label} must be a positive integer")
    return value


@dataclass(frozen=True)
class ExecutorConfiguration:
    model_dir: str
    model_manifest_path: str
    model_manifest_sha256: str
    source_tree_sha256: str
    workload_manifest_path: str
    workload_manifest_sha256: str
    model_fingerprint: str
    gpu_indices: tuple[int, ...]
    dist_port: int
    master_port: int
    max_cache_entries: int
    max_cache_bytes: int
    timeout_s: float

    def __post_init__(self):
        for name in (
            "model_dir",
            "model_manifest_path",
            "workload_manifest_path",
        ):
            _require_absolute_path(getattr(self, name), name)
        for name in (
            "model_manifest_sha256",
            "source_tree_sha256",
            "workload_manifest_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if (
            not isinstance(self.model_fingerprint, str)
            or not self.model_fingerprint
        ):
            raise ValueError("model_fingerprint must be a non-empty string")
        if (
            not isinstance(self.gpu_indices, tuple)
            or len(self.gpu_indices) != contract.WORLD_SIZE
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in self.gpu_indices
            )
        ):
            raise ValueError(
                "gpu_indices must contain exactly four non-negative integers"
            )
        if len(set(self.gpu_indices)) != contract.WORLD_SIZE:
            raise ValueError("gpu_indices must be unique")
        for name in (
            "dist_port",
            "master_port",
            "max_cache_entries",
            "max_cache_bytes",
        ):
            _require_positive_integer(getattr(self, name), name)
        if self.dist_port == self.master_port:
            raise ValueError("dist_port and master_port must be different")
        if self.dist_port > 65535 or self.master_port > 65535:
            raise ValueError("ports must not exceed 65535")
        if (
            isinstance(self.timeout_s, bool)
            or not isinstance(self.timeout_s, (int, float))
            or self.timeout_s <= 0
        ):
            raise ValueError("timeout_s must be positive")
        object.__setattr__(self, "timeout_s", float(self.timeout_s))

    def to_payload(self):
        payload = {
            name: getattr(self, name)
            for name in CONFIGURATION_FIELDS
            if name != "world_size"
        }
        payload["gpu_indices"] = list(self.gpu_indices)
        payload["world_size"] = contract.WORLD_SIZE
        return {
            name: payload[name]
            for name in CONFIGURATION_FIELDS
        }


def build_scenario_plans():
    if tuple(_SCENARIO_ACTIONS) != tuple(contract.SCENARIOS):
        raise RuntimeError("Engine correctness scenario plan mismatch")
    payloads = build_scenario_payloads()
    return {
        scenario: {
            "scenario": scenario,
            "isolation": "fresh_engine_process_group",
            "actions": tuple(_SCENARIO_ACTIONS[scenario]),
            "expected": dict(expected),
            "payload": payloads[scenario],
        }
        for scenario, expected in contract.SCENARIOS.items()
    }


def _request_prompt(payload, continuation_index):
    continuation = payload["continuations"][continuation_index]
    prefix = list(payload["shared_prefix_token_ids"])
    for index, token_id in continuation["prefix_overrides"]:
        prefix[index] = token_id
    return prefix + list(continuation["suffix_token_ids"])


def _scenario_payload(
    *,
    workload,
    payload,
    continuation_index,
    generated_tokens,
):
    return {
        "workload": workload,
        "source_prompt_token_ids": list(
            payload["shared_prefix_token_ids"]
        ),
        "request_prompt_token_ids": _request_prompt(
            payload,
            continuation_index,
        ),
        "generated_tokens": generated_tokens,
        "invalidation": dict(
            payload["continuations"][continuation_index][
                "invalidation"
            ]
        ),
    }


def build_scenario_payloads():
    benchmark = _load_benchmark_contract()
    w1 = benchmark.workload_payload("w1_medium_reuse")
    w4 = benchmark.workload_payload("w4_miss_invalidation")
    return {
        "construct_and_bind": {
            "workload": None,
            "source_prompt_token_ids": [],
            "request_prompt_token_ids": [],
            "generated_tokens": 0,
            "invalidation": {"kind": "none"},
        },
        "publish_source": {
            **_scenario_payload(
                workload="w1_medium_reuse",
                payload=w1,
                continuation_index=0,
                generated_tokens=1,
            ),
            "request_prompt_token_ids": (
                list(w1["shared_prefix_token_ids"])
                + list(w1["source_suffix_token_ids"])
            ),
        },
        "restore_w1": _scenario_payload(
            workload="w1_medium_reuse",
            payload=w1,
            continuation_index=0,
            generated_tokens=64,
        ),
        "miss_w4_token": _scenario_payload(
            workload="w4_miss_invalidation",
            payload=w4,
            continuation_index=0,
            generated_tokens=32,
        ),
        "miss_w4_stale": _scenario_payload(
            workload="w4_miss_invalidation",
            payload=w4,
            continuation_index=1,
            generated_tokens=32,
        ),
        "miss_w4_clear": _scenario_payload(
            workload="w4_miss_invalidation",
            payload=w4,
            continuation_index=2,
            generated_tokens=32,
        ),
    }


def _default_runtime_factory(configuration):
    raise RuntimeError(
        "real Qwen3.5 TP4 Engine runtime is not implemented"
    )


def _default_backend_factory(configuration, *, scenario, expected):
    raise RuntimeError(
        "real Qwen3.5 TP4 Engine backend is not implemented"
    )


class AuditedScenarioRuntime:

    def __init__(
        self,
        configuration,
        *,
        backend_factory=_default_backend_factory,
    ):
        if not isinstance(configuration, ExecutorConfiguration):
            raise TypeError(
                "configuration must be an ExecutorConfiguration"
            )
        if not callable(backend_factory):
            raise TypeError("backend_factory must be callable")
        self.configuration = configuration
        self.backend_factory = backend_factory
        self._closed = False

    def run_scenario(self, *, scenario, expected, plan):
        if self._closed:
            raise RuntimeError("audited scenario runtime is closed")
        if (
            not isinstance(plan, dict)
            or plan.get("scenario") != scenario
            or plan.get("expected") != expected
            or plan.get("isolation")
            != "fresh_engine_process_group"
            or plan.get("payload")
            != build_scenario_payloads().get(scenario)
        ):
            raise ValueError("Engine correctness scenario plan mismatch")
        actions = plan.get("actions")
        if (
            not isinstance(actions, tuple)
            or not actions
            or actions[0] != "construct_engine"
            or actions[-2:] != ("close_engine", "verify_cleanup")
        ):
            raise ValueError("Engine correctness action plan mismatch")
        session = self.backend_factory(
            self.configuration,
            scenario=scenario,
            expected=dict(expected),
        )
        try:
            row = {"scenario": scenario}
            for action in actions:
                evidence = session.execute_action(
                    action=action,
                    scenario=scenario,
                    expected=dict(expected),
                )
                if not isinstance(evidence, dict):
                    raise ValueError(
                        f"Engine correctness action evidence is invalid: "
                        f"{action}"
                    )
                duplicate = set(row).intersection(evidence)
                if duplicate:
                    raise ValueError(
                        "Engine correctness duplicate action evidence: "
                        + ",".join(sorted(duplicate))
                    )
                row.update(evidence)
            failures = contract._validate_row(row, scenario)
            if failures:
                raise ValueError(
                    "Engine correctness action classification failed: "
                    + "; ".join(failures)
                )
            return row
        finally:
            session.close()

    def close(self):
        self._closed = True


class EngineCorrectnessExecutor:

    def __init__(
        self,
        *,
        configuration,
        runtime_factory=_default_runtime_factory,
    ):
        if not isinstance(configuration, ExecutorConfiguration):
            raise TypeError(
                "configuration must be an ExecutorConfiguration"
            )
        if not callable(runtime_factory):
            raise TypeError("runtime_factory must be callable")
        self.configuration = configuration
        self.runtime_factory = runtime_factory
        self.plans = build_scenario_plans()
        self._scenario_index = 0
        self._runtime = None
        self._closed = False

    def run_scenario(self, *, scenario, expected):
        if self._closed:
            raise RuntimeError("Engine correctness executor is closed")
        scenario_names = tuple(self.plans)
        if self._scenario_index >= len(scenario_names):
            raise ValueError("Engine correctness scenario matrix is complete")
        required_scenario = scenario_names[self._scenario_index]
        if scenario != required_scenario:
            raise ValueError(
                "Engine correctness scenario order mismatch: "
                f"expected {required_scenario}, got {scenario}"
            )
        plan = self.plans[scenario]
        if expected != plan["expected"]:
            raise ValueError(
                f"Engine correctness expected counts mismatch: {scenario}"
            )
        if self._runtime is None:
            self._runtime = self.runtime_factory(self.configuration)
        row = self._runtime.run_scenario(
            scenario=scenario,
            expected=dict(expected),
            plan=plan,
        )
        self._scenario_index += 1
        return row

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._runtime is not None:
            self._runtime.close()


def build_executor_factory(
    configuration,
    *,
    runtime_factory=_default_runtime_factory,
):
    if not isinstance(configuration, ExecutorConfiguration):
        raise TypeError(
            "configuration must be an ExecutorConfiguration"
        )

    def factory():
        return EngineCorrectnessExecutor(
            configuration=configuration,
            runtime_factory=runtime_factory,
        )

    return factory
