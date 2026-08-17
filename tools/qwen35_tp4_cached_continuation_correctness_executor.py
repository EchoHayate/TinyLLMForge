from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load(name, filename):
    module = sys.modules.get(name)
    if module is not None:
        return module
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


contract = _load(
    "qwen35_tp4_cached_continuation_correctness_contract",
    "qwen35_tp4_cached_continuation_correctness_contract.py",
)
engine_executor = _load(
    "qwen35_tp4_engine_correctness_executor",
    "qwen35_tp4_engine_correctness_executor.py",
)


def _default_session_factory(
    configuration,
    *,
    workload,
    request_index,
    payload,
):
    raise RuntimeError(
        "real Qwen3.5 TP4 cached-continuation session is not implemented"
    )


def _configuration_from_payload(payload):
    if not isinstance(payload, dict):
        raise ValueError("cached executor configuration is invalid")
    values = dict(payload)
    if values.pop("world_size", None) != contract.WORLD_SIZE:
        raise ValueError("cached executor world size mismatch")
    if isinstance(values.get("gpu_indices"), list):
        values["gpu_indices"] = tuple(values["gpu_indices"])
    configuration = engine_executor.ExecutorConfiguration(**values)
    if (
        configuration.workload_manifest_sha256
        != contract.WORKLOAD_MANIFEST_SHA256
    ):
        raise ValueError(
            "cached executor workload manifest SHA mismatch"
        )
    return configuration


def _expected_order():
    return tuple(
        (workload, request_index)
        for workload in contract.WORKLOADS
        for request_index in range(
            contract.workload_payload(workload)["spec"][
                "continuations"
            ]
        )
    )


class CachedContinuationExecutor:

    def __init__(
        self,
        configuration,
        *,
        session_factory=_default_session_factory,
    ):
        if not isinstance(
            configuration,
            engine_executor.ExecutorConfiguration,
        ):
            to_payload = getattr(configuration, "to_payload", None)
            if not callable(to_payload):
                raise TypeError(
                    "configuration must expose canonical payload"
                )
            configuration = _configuration_from_payload(to_payload())
        if (
            configuration.workload_manifest_sha256
            != contract.WORKLOAD_MANIFEST_SHA256
        ):
            raise ValueError(
                "cached executor workload manifest SHA mismatch"
            )
        if not callable(session_factory):
            raise TypeError("session_factory must be callable")
        self.configuration = configuration
        self.session_factory = session_factory
        self.order = _expected_order()
        self.position = 0
        self.closed = False

    def run_continuation(
        self,
        *,
        workload,
        request_index,
        payload,
    ):
        if self.closed:
            raise RuntimeError("cached executor is closed")
        if payload != contract.workload_payload(workload):
            raise ValueError("cached executor workload payload mismatch")
        if (
            self.position >= len(self.order)
            or self.order[self.position] != (workload, request_index)
        ):
            raise ValueError("cached executor row order mismatch")
        session = self.session_factory(
            self.configuration,
            workload=workload,
            request_index=request_index,
            payload=payload,
        )
        try:
            row = session.run()
            validation_row = dict(row)
            if (
                getattr(session, "defers_cleanup", False)
                and validation_row.get("process_group_destroyed") is False
            ):
                validation_row["process_group_destroyed"] = True
            failures = contract._validate_row(
                validation_row,
                (workload, request_index),
            )
            if failures:
                raise ValueError(
                    "cached executor row classification failed: "
                    + "; ".join(failures)
                )
            self.position += 1
            return row
        finally:
            session.close()

    def close(self):
        if self.closed:
            return
        self.closed = True
        close = getattr(self.session_factory, "close", None)
        if callable(close):
            close()


def build_configured_executor_factory(
    configuration_payload,
    *,
    session_factory=_default_session_factory,
):
    configuration = _configuration_from_payload(
        configuration_payload
    )

    def factory():
        return CachedContinuationExecutor(
            configuration,
            session_factory=session_factory,
        )

    return factory
