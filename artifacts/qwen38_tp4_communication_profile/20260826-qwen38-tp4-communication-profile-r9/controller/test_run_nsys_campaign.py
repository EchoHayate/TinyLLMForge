from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
MODULE_PATH = HERE / "run_nsys_campaign.py"


def _load():
    assert MODULE_PATH.is_file(), "Nsight campaign controller is missing"
    spec = importlib.util.spec_from_file_location(
        "qwen38_run_nsys_campaign_under_test",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_nsys_cases_cover_all_measured_repetitions_in_frozen_order():
    module = _load()

    assert [
        module.case_id(case) for case in module.nsys_cases()
    ] == [
        f"{workload}__nsys_replay__r{repetition}"
        for workload in ("P0", "P1", "Q0", "Q1", "Q2")
        for repetition in range(5)
    ]


def test_representatives_choose_nearest_median_with_lowest_tie_break():
    module = _load()
    decode_times = {
        "P0": {
            0: 90,
            1: 110,
            2: 100,
            3: 120,
            4: 80,
        },
        "P1": {
            0: 10,
            1: 20,
            2: 20,
            3: 30,
            4: 40,
        },
    }

    assert module.select_representatives(decode_times) == {
        "P0": 2,
        "P1": 1,
    }


def test_nsys_command_exports_sqlite_below_attempt_and_wraps_one_case():
    module = _load()
    case = module.nsys_cases()[0]
    command = module.build_nsys_command(
        case,
        output_prefix=module.NSYS / "P0-r0",
        result_path=module.NSYS_CASES / "P0__nsys_replay__r0.json",
    )

    assert command[:2] == ["/usr/local/bin/nsys", "profile"]
    assert "--trace=cuda,nvtx,osrt" in command
    assert "--sample=none" in command
    assert "--cpuctxsw=process-tree" in command
    assert "--trace-fork-before-exec=true" in command
    assert "--wait=all" in command
    assert "--export=sqlite" in command
    assert "--force-overwrite=true" in command
    assert (
        f"--output={module.NSYS / 'P0-r0'}"
        in command
    )
    assert command[-5:] == [
        str(module.PYTHON),
        str(module.CONTROLLER / "run_nsys_campaign.py"),
        "--run-case=P0__nsys_replay__r0",
        (
            "--result-path="
            f"{module.NSYS_CASES / 'P0__nsys_replay__r0.json'}"
        ),
        "--timeout-s=1800",
    ]


def test_only_cleaned_resource_interference_is_retryable():
    module = _load()

    assert module.case_attempt_is_retryable({
        "violations": ["unrelated GPU process on GPU-a: [42]"],
        "process_group_destroyed": True,
        "owned_children_remaining": [],
    })
    assert not module.case_attempt_is_retryable({
        "violations": [],
        "process_group_destroyed": True,
        "owned_children_remaining": [],
    })
    assert not module.case_attempt_is_retryable({
        "violations": ["GPU identity drift at index 2"],
        "process_group_destroyed": False,
        "owned_children_remaining": [99],
    })


def test_reparented_gpu_worker_with_attempt_token_is_owned():
    module = _load()

    assert module.is_owned_process(
        42,
        process_group_owned={11, 12},
        ownership_token="attempt-token",
        token_reader=lambda pid: (
            "attempt-token" if pid == 42 else None
        ),
    )
    assert not module.is_owned_process(
        43,
        process_group_owned={11, 12},
        ownership_token="attempt-token",
        token_reader=lambda pid: "another-attempt",
    )


def test_retryable_interference_stops_after_attempt_budget():
    module = _load()
    retryable = {
        "violations": ["unrelated GPU process on GPU-a: [42]"],
        "process_group_destroyed": True,
        "owned_children_remaining": [],
    }

    assert module.attempt_disposition(
        retryable,
        attempt_index=0,
        max_attempts=3,
    ) == "retry"
    assert module.attempt_disposition(
        retryable,
        attempt_index=2,
        max_attempts=3,
    ) == "retry_exhausted"
    assert module.attempt_disposition(
        {
            "violations": [],
            "process_group_destroyed": True,
            "owned_children_remaining": [],
        },
        attempt_index=0,
        max_attempts=3,
    ) == "terminal"
