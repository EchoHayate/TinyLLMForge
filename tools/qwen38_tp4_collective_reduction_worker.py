#!/usr/bin/env python3
"""Run source-bound Qwen3.8 TP4 synchronous-collective qualification."""

from __future__ import annotations

import argparse
from itertools import count
import json
import os
from pathlib import Path
from statistics import median
import tempfile
import time

if __package__:
    from tools.qwen38_collective_reduction import select_event_budget
    from tools.qwen38_tp4_communication_profile_worker import (
        WORKLOADS,
        build_request_specs,
    )
else:
    from qwen38_collective_reduction import select_event_budget
    from qwen38_tp4_communication_profile_worker import (
        WORKLOADS,
        build_request_specs,
    )


WORKER_SCHEMA = "qwen38.tp4-collective-reduction-worker.v1"
RANKS = (0, 1, 2, 3)
CALIBRATION_WORKLOADS = ("P0", "P1", "Q1")
TERMINAL_WORKLOADS = ("P0", "P1", "Q0", "Q1", "Q2")
EVENT_BUDGETS = (0, 8, 16, 32)
EXPECTED_COLLECTIVE_COUNT = 66
COHORT_COUNT = 17


def _atomic_write_json(path: Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".partial",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(
            payload,
            handle,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _default_engine_factory(model_root, **kwargs):
    from tinyvllm.engine.llm_engine import LLMEngine

    return LLMEngine(str(model_root), **kwargs)


def _default_sampling_params_factory(**kwargs):
    from tinyvllm.sampling_params import SamplingParams

    return SamplingParams(**kwargs)


def _reset_sequence_ids() -> None:
    from tinyvllm.engine.sequence import Sequence

    Sequence.counter = count()


def _validate_cleanup(receipt):
    if (
        not isinstance(receipt, dict)
        or receipt.get("process_group_destroyed") is not True
        or receipt.get("rank_exit_codes") != [0, 0, 0, 0]
        or receipt.get("owned_children_remaining") != []
    ):
        raise RuntimeError("TP4 worker cleanup receipt is incomplete")
    return dict(receipt)


def _validate_memory(rows):
    rows = tuple(rows)
    if (
        len(rows) != 4
        or sorted(row.get("rank") for row in rows) != list(RANKS)
    ):
        raise RuntimeError("TP4 memory rank inventory mismatch")
    return [dict(row) for row in sorted(
        rows,
        key=lambda row: row["rank"],
    )]


def _validate_engine_ownership(engine) -> None:
    model_runner = getattr(engine, "model_runner", None)
    if (
        getattr(model_runner, "rank", None) != 0
        or getattr(model_runner, "world_size", None) != 4
    ):
        raise RuntimeError("TP4 engine ownership mismatch")


def collective_reduction_case_id(
    *,
    campaign_phase,
    workload,
    phase,
    repetition,
    budget,
):
    if campaign_phase not in {"calibration", "terminal"}:
        raise ValueError("unknown campaign phase")
    if workload not in WORKLOADS:
        raise ValueError("unknown workload")
    if phase not in {"warmup", "measured"}:
        raise ValueError("unknown measurement phase")
    if type(repetition) is not int or repetition < 0:
        raise ValueError("invalid repetition")
    if budget not in EVENT_BUDGETS:
        raise ValueError("unsupported event budget")
    return (
        f"{campaign_phase}__{workload}__budget{budget}__"
        f"{phase}__r{repetition}"
    )


def _case_rows(campaign_phase, workloads, budgets):
    rows = []
    for workload in workloads:
        family, prompt_tokens, output_tokens, concurrency = (
            WORKLOADS[workload]
        )
        for budget in budgets:
            for phase, repetitions in (
                ("warmup", range(2)),
                ("measured", range(5)),
            ):
                for repetition in repetitions:
                    rows.append({
                        "campaign_phase": campaign_phase,
                        "workload": workload,
                        "workload_family": family,
                        "phase": phase,
                        "repetition": repetition,
                        "prompt_tokens": prompt_tokens,
                        "output_tokens": output_tokens,
                        "concurrency": concurrency,
                        "budget": budget,
                    })
    return tuple(rows)


def build_collective_reduction_cases(*, selected_budget: int):
    if selected_budget not in EVENT_BUDGETS[1:]:
        raise ValueError(
            "selected_budget must be one of 8, 16, or 32"
        )
    return {
        "calibration": _case_rows(
            "calibration",
            CALIBRATION_WORKLOADS,
            EVENT_BUDGETS,
        ),
        "terminal": _case_rows(
            "terminal",
            TERMINAL_WORKLOADS,
            (selected_budget,),
        ),
    }


def _run_requests(
    engine,
    requests,
    *,
    output_tokens,
    timeout_s,
    sampling_params_factory,
    clock_ns,
):
    if hasattr(engine, "run_requests"):
        return engine.run_requests(requests, timeout_s=timeout_s)

    lifecycle = {}
    for seq_id, request in enumerate(requests):
        sampling = sampling_params_factory(
            temperature=0.0,
            max_tokens=output_tokens,
            ignore_eos=True,
        )
        admitted_ns = clock_ns()
        engine.add_request(request["prompt_token_ids"], sampling)
        lifecycle[seq_id] = {
            "request_id": request["request_id"],
            "admitted_ns": admitted_ns,
            "first_token_ns": None,
            "token_timestamps_ns": [],
            "output_token_ids": [],
            "complete": False,
        }
    while not engine.is_finished():
        outputs, _ = engine.step()
        observation = engine.last_step_observation
        if not isinstance(observation, dict):
            raise RuntimeError("step observation is missing")
        step_end_ns = observation.get("step_end_ns")
        token_deltas = observation.get("new_completion_tokens_by_seq")
        if (
            type(step_end_ns) is not int
            or not isinstance(token_deltas, dict)
        ):
            raise RuntimeError("step timing observation is invalid")
        for raw_seq_id, tokens in token_deltas.items():
            seq_id = int(raw_seq_id)
            row = lifecycle.get(seq_id)
            if row is None or not isinstance(tokens, list):
                raise RuntimeError("token delta observation is invalid")
            if tokens:
                if row["first_token_ns"] is None:
                    row["first_token_ns"] = step_end_ns
                row["token_timestamps_ns"].extend(
                    [step_end_ns] * len(tokens)
                )
                row["output_token_ids"].extend(int(token) for token in tokens)
        for raw_seq_id, output in outputs:
            row = lifecycle.get(int(raw_seq_id))
            if row is None or list(output) != row["output_token_ids"]:
                raise RuntimeError("terminal token output mismatch")
            row["complete"] = True

    request_rows = []
    for seq_id in range(len(requests)):
        row = lifecycle[seq_id]
        timestamps = row["token_timestamps_ns"]
        if (
            row["complete"] is not True
            or row["first_token_ns"] is None
            or len(row["output_token_ids"]) != output_tokens
            or len(timestamps) != output_tokens
        ):
            raise RuntimeError("request completion evidence is incomplete")
        intervals = [
            current - previous
            for previous, current in zip(timestamps, timestamps[1:])
        ]
        request_rows.append({
            "request_id": row["request_id"],
            "output_token_ids": list(row["output_token_ids"]),
            "ttft_ns": row["first_token_ns"] - row["admitted_ns"],
            "tpot_ns": (
                sum(intervals) / len(intervals) if intervals else 0.0
            ),
            "e2e_ns": timestamps[-1] - row["admitted_ns"],
        })
    return {
        "requests": request_rows,
        "decode_time_ns": max(
            1,
            int(max(
                row["e2e_ns"] - row["ttft_ns"]
                for row in request_rows
            )),
        ),
    }


def _run_arm(
    *,
    engine,
    arm,
    attempt,
    source_revision,
    workload,
    repetition,
    budget,
    timeout_s,
    sampling_params_factory,
    clock_ns,
    reset_sequence_ids,
):
    family, prompt_tokens, output_tokens, concurrency = WORKLOADS[
        workload
    ]
    engine.configure_decode_internal_profile(
        False,
        "disabled",
        timeout_s=float(timeout_s),
    )
    policy = (
        {"enabled": False}
        if arm == "control"
        else {
            "enabled": True,
            "sample_budget": budget,
            "cohort_count": COHORT_COUNT,
            "expected_collective_count": (
                EXPECTED_COLLECTIVE_COUNT
            ),
            "source_revision": source_revision,
            "attempt": attempt,
            "workload": workload,
            "repetition": repetition,
        }
    )
    configured = engine.configure_synchronous_collective_census(
        policy,
        timeout_s=float(timeout_s),
    )
    if configured.get("rank_inventory") != list(RANKS):
        raise RuntimeError("collective census setup rank mismatch")
    engine.clear_reusable_prefix_cache()
    reset_sequence_ids()
    engine.reset_peak_memory_stats(timeout_s=float(timeout_s))
    requests = build_request_specs(
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        concurrency=concurrency,
    )
    measurement = _run_requests(
        engine,
        requests,
        output_tokens=output_tokens,
        timeout_s=float(timeout_s),
        sampling_params_factory=sampling_params_factory,
        clock_ns=clock_ns,
    )
    census = engine.finalize_synchronous_collective_census(
        already_synchronized_rank=0,
        timeout_s=float(timeout_s),
    )
    if census.get("rank_inventory") != list(RANKS):
        raise RuntimeError("collective census final rank mismatch")
    return {
        "arm": arm,
        "policy": policy,
        "requests": measurement["requests"],
        "decode_time_ns": measurement["decode_time_ns"],
        "census": census,
        "memory": _validate_memory(
            engine.memory_snapshots(timeout_s=float(timeout_s))
        ),
    }


def run_collective_reduction_pair(
    *,
    engine,
    attempt: str,
    source_revision: str,
    campaign_phase: str,
    workload: str,
    phase: str,
    repetition: int,
    budget: int,
    timeout_s: float,
    sampling_params_factory=_default_sampling_params_factory,
    clock_ns=time.monotonic_ns,
    reset_sequence_ids=_reset_sequence_ids,
):
    _validate_engine_ownership(engine)
    if workload not in WORKLOADS:
        raise ValueError("unknown workload")
    if budget not in EVENT_BUDGETS:
        raise ValueError("unsupported event budget")
    arm_order = (
        ("control", "instrumented")
        if repetition % 2 == 0
        else ("instrumented", "control")
    )
    arms = [
        _run_arm(
            engine=engine,
            arm=arm,
            attempt=attempt,
            source_revision=source_revision,
            workload=workload,
            repetition=repetition,
            budget=budget,
            timeout_s=timeout_s,
            sampling_params_factory=sampling_params_factory,
            clock_ns=clock_ns,
            reset_sequence_ids=reset_sequence_ids,
        )
        for arm in arm_order
    ]
    by_arm = {row["arm"]: row for row in arms}
    control_outputs = [
        row["output_token_ids"]
        for row in by_arm["control"]["requests"]
    ]
    instrumented_outputs = [
        row["output_token_ids"]
        for row in by_arm["instrumented"]["requests"]
    ]
    if control_outputs != instrumented_outputs:
        raise RuntimeError("control/instrumented output mismatch")
    return {
        "schema_version": WORKER_SCHEMA,
        "classification": "PASS",
        "case_id": collective_reduction_case_id(
            campaign_phase=campaign_phase,
            workload=workload,
            phase=phase,
            repetition=repetition,
            budget=budget,
        ),
        "attempt": attempt,
        "source_revision": source_revision,
        "campaign_phase": campaign_phase,
        "workload": workload,
        "workload_family": WORKLOADS[workload][0],
        "prompt_tokens": WORKLOADS[workload][1],
        "output_tokens": WORKLOADS[workload][2],
        "concurrency": WORKLOADS[workload][3],
        "phase": phase,
        "repetition": repetition,
        "budget": budget,
        "arm_order": list(arm_order),
        "arms": arms,
    }


def run_collective_reduction_campaign(
    *,
    attempt,
    source_revision,
    cases,
    model_root,
    timeout_s,
    engine_factory=_default_engine_factory,
    sampling_params_factory=_default_sampling_params_factory,
    clock_ns=time.monotonic_ns,
    reset_sequence_ids=_reset_sequence_ids,
    case_sink=None,
):
    cases = tuple(dict(case) for case in cases)
    if not cases:
        raise ValueError("campaign cases must not be empty")
    if case_sink is not None and not callable(case_sink):
        raise ValueError("case_sink must be callable")
    engine = None
    cleanup = None
    try:
        max_prompt = max(WORKLOADS[row["workload"]][1] for row in cases)
        max_output = max(WORKLOADS[row["workload"]][2] for row in cases)
        max_concurrency = max(
            WORKLOADS[row["workload"]][3] for row in cases
        )
        engine = engine_factory(
            Path(model_root),
            tensor_parallel_size=4,
            enforce_eager=True,
            max_num_seqs=max(8, max_concurrency),
            max_model_len=max_prompt + max_output,
            max_num_batched_tokens=max_prompt * max_concurrency,
        )
        _validate_engine_ownership(engine)
        receipts = []
        for case in cases:
            result = run_collective_reduction_pair(
                engine=engine,
                attempt=attempt,
                source_revision=source_revision,
                timeout_s=timeout_s,
                sampling_params_factory=sampling_params_factory,
                clock_ns=clock_ns,
                reset_sequence_ids=reset_sequence_ids,
                **{
                    key: case[key]
                    for key in (
                        "campaign_phase",
                        "workload",
                        "phase",
                        "repetition",
                        "budget",
                    )
                },
            )
            if case_sink is not None:
                case_sink(result)
            receipts.append({
                "case_id": result["case_id"],
                "classification": result["classification"],
                "budget": result["budget"],
            })
    finally:
        if engine is not None:
            cleanup = _validate_cleanup(engine.exit())
    return {
        "schema_version": WORKER_SCHEMA,
        "classification": "PASS",
        "attempt": attempt,
        "source_revision": source_revision,
        "cases": receipts,
        "cleanup": cleanup,
    }


def select_event_budget_from_cases(cases):
    grouped = {budget: [] for budget in EVENT_BUDGETS}
    identities = {budget: set() for budget in EVENT_BUDGETS}
    for case in cases:
        if (
            not isinstance(case, dict)
            or case.get("campaign_phase") != "calibration"
            or case.get("phase") != "measured"
            or case.get("budget") not in EVENT_BUDGETS
        ):
            continue
        arms = case.get("arms")
        if (
            not isinstance(arms, list)
            or {row.get("arm") for row in arms} != {
                "control",
                "instrumented",
            }
        ):
            raise ValueError("calibration case arms are invalid")
        by_arm = {row["arm"]: row for row in arms}
        control = by_arm["control"].get("decode_time_ns")
        instrumented = by_arm["instrumented"].get("decode_time_ns")
        if (
            isinstance(control, bool)
            or not isinstance(control, (int, float))
            or control <= 0
            or isinstance(instrumented, bool)
            or not isinstance(instrumented, (int, float))
            or instrumented <= 0
        ):
            raise ValueError("calibration timing is invalid")
        budget = case["budget"]
        identity = (case.get("workload"), case.get("repetition"))
        if identity in identities[budget]:
            raise ValueError("calibration case identity is duplicated")
        identities[budget].add(identity)
        grouped[budget].append(instrumented / control - 1.0)
    expected_identities = {
        (workload, repetition)
        for workload in CALIBRATION_WORKLOADS
        for repetition in range(5)
    }
    if any(
        identities[budget] != expected_identities
        for budget in EVENT_BUDGETS
    ):
        raise ValueError("calibration case coverage is incomplete")
    return select_event_budget([
        {
            "budget": budget,
            "median_overhead_ratio": median(grouped[budget]),
            "maximum_overhead_ratio": max(grouped[budget]),
        }
        for budget in EVENT_BUDGETS
    ])


def run_full_collective_reduction_campaign(
    *,
    attempt,
    source_revision,
    model_root,
    timeout_s,
    engine_factory=_default_engine_factory,
    sampling_params_factory=_default_sampling_params_factory,
    clock_ns=time.monotonic_ns,
    reset_sequence_ids=_reset_sequence_ids,
    case_sink=None,
    phase_runner=run_collective_reduction_campaign,
    case_matrix_builder=build_collective_reduction_cases,
    budget_selector=select_event_budget_from_cases,
    pid_resolver=os.getpid,
):
    calibration_rows = []

    def calibration_sink(row):
        calibration_rows.append(dict(row))
        if case_sink is not None:
            case_sink(row)

    matrix = case_matrix_builder(selected_budget=16)
    calibration = phase_runner(
        attempt=attempt,
        source_revision=source_revision,
        cases=matrix["calibration"],
        model_root=model_root,
        timeout_s=timeout_s,
        engine_factory=engine_factory,
        sampling_params_factory=sampling_params_factory,
        clock_ns=clock_ns,
        reset_sequence_ids=reset_sequence_ids,
        case_sink=calibration_sink,
    )
    selected_budget = budget_selector(calibration_rows)
    phase_results = [calibration]
    if selected_budget is not None:
        terminal = phase_runner(
            attempt=attempt,
            source_revision=source_revision,
            cases=case_matrix_builder(
                selected_budget=selected_budget
            )["terminal"],
            model_root=model_root,
            timeout_s=timeout_s,
            engine_factory=engine_factory,
            sampling_params_factory=sampling_params_factory,
            clock_ns=clock_ns,
            reset_sequence_ids=reset_sequence_ids,
            case_sink=case_sink,
        )
        phase_results.append(terminal)
    if any(
        not isinstance(result, dict)
        or result.get("classification") != "PASS"
        or result.get("attempt") != attempt
        or result.get("source_revision") != source_revision
        for result in phase_results
    ):
        raise RuntimeError("collective reduction phase failed")
    owned_pid = pid_resolver()
    if type(owned_pid) is not int or owned_pid <= 0:
        raise RuntimeError("collective reduction worker PID is invalid")
    return {
        "schema_version": WORKER_SCHEMA,
        "classification": "PASS",
        "attempt": attempt,
        "source_revision": source_revision,
        "selected_budget": selected_budget,
        "owned_pids": [owned_pid],
        "cases": [
            dict(case)
            for result in phase_results
            for case in result["cases"]
        ],
        "phase_cleanups": [
            dict(result["cleanup"])
            for result in phase_results
        ],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selected-budget", type=int)
    parser.add_argument(
        "--phase",
        choices=("calibration", "terminal", "full"),
        required=True,
    )
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    args = parser.parse_args(argv)

    def write_case(case_result):
        _atomic_write_json(
            args.output_dir / f"{case_result['case_id']}.json",
            case_result,
        )

    if args.phase == "full":
        if args.selected_budget is not None:
            parser.error("--selected-budget is invalid with --phase full")
        result = run_full_collective_reduction_campaign(
            attempt=args.attempt,
            source_revision=args.source_revision,
            model_root=args.model_root,
            timeout_s=args.timeout_s,
            case_sink=write_case,
        )
    else:
        if args.selected_budget is None:
            parser.error(
                "--selected-budget is required for a single phase"
            )
        cases = build_collective_reduction_cases(
            selected_budget=args.selected_budget
        )[args.phase]
        result = run_collective_reduction_campaign(
            attempt=args.attempt,
            source_revision=args.source_revision,
            cases=cases,
            model_root=args.model_root,
            timeout_s=args.timeout_s,
            case_sink=write_case,
        )
    _atomic_write_json(args.output, result)
    print(json.dumps({
        "classification": result["classification"],
        "case_count": len(result["cases"]),
        "selected_budget": result.get("selected_budget"),
        "output": str(args.output),
        "output_dir": str(args.output_dir),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
