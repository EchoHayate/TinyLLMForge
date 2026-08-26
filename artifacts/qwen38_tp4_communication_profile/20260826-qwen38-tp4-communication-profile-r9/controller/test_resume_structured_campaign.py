from __future__ import annotations

import importlib.util
from pathlib import Path
import socket

import pytest


MODULE_PATH = Path(__file__).with_name(
    "resume_structured_campaign.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "resume_structured_campaign",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def case_ids(workload):
    return {
        f"{workload}__warmup__r{repetition}"
        for repetition in range(2)
    } | {
        f"{workload}__measured__r{repetition}"
        for repetition in range(5)
    }


def test_resume_case_batches_preserve_completed_cases_individually():
    module = load_module()
    completed = case_ids("P0") | {"P1__warmup__r0"}

    cases = module.resume_case_batches(completed)

    assert len(cases) == 27
    assert module.case_id(cases[0]) == "P1__warmup__r1"
    assert module.case_id(cases[-1]) == "Q2__measured__r4"
    assert not completed & {
        module.case_id(case) for case in cases
    }


def test_resume_case_batches_accept_partial_workload():
    module = load_module()
    completed = case_ids("P0")
    completed.remove("P0__measured__r4")

    cases = module.resume_case_batches(completed)

    assert module.case_id(cases[0]) == "P0__measured__r4"


def test_resume_case_batches_reject_unknown_case():
    module = load_module()

    with pytest.raises(
        RuntimeError,
        match="unknown completed cases",
    ):
        module.resume_case_batches(
            case_ids("P0") | {"unexpected__measured__r0"}
        )


def test_engine_kwargs_cover_long_single_request_model_length():
    module = load_module()

    normalized = module.normalize_engine_kwargs({
        "max_model_len": 2176,
        "max_num_batched_tokens": 2048,
        "max_num_seqs": 8,
    })

    assert normalized == {
        "max_model_len": 2176,
        "max_num_batched_tokens": 2176,
        "max_num_seqs": 8,
    }


def test_resource_aggregation_starts_only_after_engine_ready(tmp_path):
    module = load_module()
    marker = tmp_path / "ready.json"
    measured = {"phase": "measured"}

    assert not module.ready_for_resource_aggregation(
        measured,
        marker,
    )
    marker.write_text("{}\n", encoding="utf-8")
    assert module.ready_for_resource_aggregation(
        measured,
        marker,
    )
    assert not module.ready_for_resource_aggregation(
        {"phase": "warmup"},
        marker,
    )


def test_gpu_wait_covers_long_external_occupancy_window():
    module = load_module()

    assert module.GPU_WAIT_TIMEOUT_S >= 12 * 60 * 60


def test_allocate_free_tcp_port_returns_bindable_port():
    module = load_module()

    port = module.allocate_free_tcp_port()

    assert isinstance(port, int)
    assert 1024 < port <= 65535
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", port))


def test_only_cleaned_resource_interference_is_retryable():
    module = load_module()
    retryable = {
        "returncode": -15,
        "violations": ["unrelated GPU process"],
        "process_group_destroyed": True,
        "owned_children_remaining": [],
    }

    assert module.case_attempt_is_retryable(retryable)
    assert not module.case_attempt_is_retryable({
        **retryable,
        "violations": [],
    })
    assert not module.case_attempt_is_retryable({
        **retryable,
        "process_group_destroyed": False,
    })
