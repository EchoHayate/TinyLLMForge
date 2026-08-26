from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import secrets
import signal
import statistics
import subprocess
import time

from resume_structured_campaign import (
    ATTEMPT,
    CASES,
    CONTROLLER,
    MODEL,
    PYTHON,
    SELECTED,
    SOURCE,
    allocate_free_tcp_port,
    canonical_write,
    case_attempt_is_retryable,
    gpu_inventory,
    normalize_engine_kwargs,
    process_group_pids,
    wait_for_clean_entry,
)


NSYS = ATTEMPT / "nsys"
NSYS_CASES = ATTEMPT / "artifacts" / "nsys_replay" / "cases"
WORKLOADS = {
    "P0": ("causal", 256, 128, 1),
    "P1": ("causal", 2048, 128, 1),
    "Q0": ("online", 256, 128, 4),
    "Q1": ("online", 256, 128, 8),
    "Q2": ("online", 2048, 128, 4),
}
OWNERSHIP_ENV = "TINYLLMFORGE_NSYS_OWNERSHIP_TOKEN"
DEFAULT_MAX_ATTEMPTS = 3


def nsys_cases() -> tuple[dict, ...]:
    return tuple(
        {
            "workload": workload,
            "workload_family": family,
            "phase": "nsys_replay",
            "repetition": repetition,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "concurrency": concurrency,
        }
        for workload, (
            family,
            prompt_tokens,
            output_tokens,
            concurrency,
        ) in WORKLOADS.items()
        for repetition in range(5)
    )


def case_id(case: dict) -> str:
    return (
        f"{case['workload']}__{case['phase']}__"
        f"r{case['repetition']}"
    )


def parse_case_id(selected_case_id: str) -> dict:
    matches = [
        case for case in nsys_cases()
        if case_id(case) == selected_case_id
    ]
    if len(matches) != 1:
        raise RuntimeError("requested Nsight case identity is invalid")
    return matches[0]


def select_representatives(
    decode_times: dict[str, dict[int, int]],
) -> dict[str, int]:
    result = {}
    for workload, repetitions in decode_times.items():
        if set(repetitions) != set(range(5)):
            raise RuntimeError(
                f"measured inventory is incomplete for {workload}"
            )
        median = statistics.median(repetitions.values())
        result[workload] = min(
            repetitions,
            key=lambda repetition: (
                abs(repetitions[repetition] - median),
                repetition,
            ),
        )
    return result


def measured_decode_times() -> dict[str, dict[int, int]]:
    result = {}
    for workload in WORKLOADS:
        repetitions = {}
        for repetition in range(5):
            path = CASES / f"{workload}__measured__r{repetition}.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            value = payload.get("decode_time_ns")
            if (
                payload.get("classification") != "PASS"
                or isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise RuntimeError(
                    f"invalid measured case: {path.name}"
                )
            repetitions[repetition] = value
        result[workload] = repetitions
    return result


def build_nsys_command(
    case: dict,
    *,
    output_prefix: Path,
    result_path: Path,
    timeout_s: float = 1800.0,
) -> list[str]:
    selected_case_id = case_id(case)
    if output_prefix.parent != NSYS:
        raise RuntimeError("Nsight output prefix is outside nsys root")
    if result_path.parent != NSYS_CASES:
        raise RuntimeError("Nsight result path is outside case root")
    return [
        "/usr/local/bin/nsys",
        "profile",
        "--trace=cuda,nvtx,osrt",
        "--sample=none",
        "--cpuctxsw=process-tree",
        "--trace-fork-before-exec=true",
        "--wait=all",
        "--export=sqlite",
        "--force-overwrite=true",
        f"--output={output_prefix}",
        str(PYTHON),
        str(CONTROLLER / "run_nsys_campaign.py"),
        f"--run-case={selected_case_id}",
        f"--result-path={result_path}",
        f"--timeout-s={int(timeout_s)}",
    ]


def read_process_ownership_token(pid: int) -> str | None:
    try:
        entries = (
            Path(f"/proc/{pid}/environ")
            .read_bytes()
            .split(b"\0")
        )
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None
    prefix = f"{OWNERSHIP_ENV}=".encode()
    for entry in entries:
        if entry.startswith(prefix):
            return entry[len(prefix):].decode(errors="replace")
    return None


def is_owned_process(
    pid: int,
    *,
    process_group_owned: set[int],
    ownership_token: str,
    token_reader=read_process_ownership_token,
) -> bool:
    return (
        pid in process_group_owned
        or token_reader(pid) == ownership_token
    )


def token_owned_pids(ownership_token: str) -> list[int]:
    owned = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if read_process_ownership_token(pid) == ownership_token:
            owned.append(pid)
    return sorted(owned)


def signal_token_owned_processes(
    ownership_token: str,
    selected_signal: signal.Signals,
) -> None:
    for pid in reversed(token_owned_pids(ownership_token)):
        try:
            os.kill(pid, selected_signal)
        except ProcessLookupError:
            continue


def attempt_disposition(
    attempt: dict,
    *,
    attempt_index: int,
    max_attempts: int,
) -> str:
    if not case_attempt_is_retryable(attempt):
        return "terminal"
    if attempt_index + 1 < max_attempts:
        return "retry"
    return "retry_exhausted"


def run_case(
    selected_case_id: str,
    *,
    result_path: Path,
    timeout_s: float,
) -> int:
    from tools.qwen38_tp4_communication_profile_worker import (
        _default_engine_factory,
        _default_sampling_params_factory,
        _reset_sequence_ids,
        run_profile_case,
    )

    case = parse_case_id(selected_case_id)
    result_path = Path(result_path)
    expected_path = NSYS_CASES / f"{selected_case_id}.json"
    if result_path != expected_path:
        raise RuntimeError("Nsight result path does not match case")
    if result_path.exists():
        raise RuntimeError(
            f"refusing to overwrite Nsight case {result_path.name}"
        )
    marker_value = os.environ.get(
        "TINYLLMFORGE_CASE_READY_MARKER"
    )
    if not marker_value:
        raise RuntimeError("case-ready marker path is missing")
    marker_path = Path(marker_value)
    if marker_path.parent != CONTROLLER:
        raise RuntimeError("case-ready marker path is invalid")

    def engine_factory(model_root, **kwargs):
        engine = _default_engine_factory(
            model_root,
            **normalize_engine_kwargs(kwargs),
        )
        canonical_write(
            marker_path,
            {
                "case_id": selected_case_id,
                "engine_ready_at_unix_ns": time.time_ns(),
            },
        )
        return engine

    result = run_profile_case(
        attempt="20260826-qwen38-tp4-communication-profile-r9",
        model_root=MODEL,
        timeout_s=timeout_s,
        engine_factory=engine_factory,
        sampling_params_factory=_default_sampling_params_factory,
        clock_ns=time.monotonic_ns,
        reset_sequence_ids=_reset_sequence_ids,
        **case,
    )
    canonical_write(result_path, result)
    print(json.dumps({
        "classification": result["classification"],
        "case_id": selected_case_id,
        "output": str(result_path),
    }, sort_keys=True))
    return 0


def monitor_case(
    case: dict,
    *,
    attempt_index: int,
    timeout_s: float,
) -> dict:
    selected_case_id = case_id(case)
    result_path = NSYS_CASES / f"{selected_case_id}.json"
    output_prefix = (
        NSYS / f"{case['workload']}-r{case['repetition']}"
    )
    command = build_nsys_command(
        case,
        output_prefix=output_prefix,
        result_path=result_path,
        timeout_s=timeout_s,
    )
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(
        str(index) for index, _ in SELECTED
    )
    dist_port = allocate_free_tcp_port()
    environment["TINYVLLM_DIST_PORT"] = str(dist_port)
    ownership_token = (
        f"{selected_case_id}.attempt-{attempt_index}."
        f"{secrets.token_hex(16)}"
    )
    environment[OWNERSHIP_ENV] = ownership_token
    marker_path = (
        CONTROLLER
        / (
            f"nsys-{selected_case_id}."
            f"attempt-{attempt_index}.engine-ready.json"
        )
    )
    if marker_path.exists():
        marker_path.unlink()
    environment["TINYLLMFORGE_CASE_READY_MARKER"] = str(
        marker_path
    )
    stdout_path = (
        CONTROLLER
        / f"nsys-{selected_case_id}.attempt-{attempt_index}.stdout"
    )
    stderr_path = (
        CONTROLLER
        / f"nsys-{selected_case_id}.attempt-{attempt_index}.stderr"
    )
    samples_path = CONTROLLER / "nsys-resource-samples.raw.jsonl"
    violations = []
    with (
        stdout_path.open("w", encoding="utf-8") as stdout,
        stderr_path.open("w", encoding="utf-8") as stderr,
        samples_path.open("a", encoding="utf-8") as samples,
    ):
        process = subprocess.Popen(
            command,
            cwd=SOURCE,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            text=True,
            start_new_session=True,
        )
        pgid = os.getpgid(process.pid)
        while process.poll() is None:
            rows = gpu_inventory()
            by_uuid = {row["gpu_uuid"]: row for row in rows}
            process_group_owned = set(process_group_pids(pgid))
            owned = set(process_group_owned)
            sample = {
                "captured_at_unix_ns": time.time_ns(),
                "case": case,
                "engine_ready": marker_path.is_file(),
                "owned_pids": sorted(owned),
                "selected_gpus": [],
            }
            for index, uuid in SELECTED:
                row = by_uuid.get(uuid)
                if row is None or row["gpu_index"] != index:
                    violations.append(
                        f"GPU identity drift at index {index}"
                    )
                    continue
                foreign = [
                    item["pid"]
                    for item in row["compute_processes"]
                    if not is_owned_process(
                        item["pid"],
                        process_group_owned=process_group_owned,
                        ownership_token=ownership_token,
                    )
                ]
                owned.update(
                    item["pid"]
                    for item in row["compute_processes"]
                    if is_owned_process(
                        item["pid"],
                        process_group_owned=process_group_owned,
                        ownership_token=ownership_token,
                    )
                )
                if foreign:
                    violations.append(
                        f"unrelated GPU process on {uuid}: {foreign}"
                    )
                sample["selected_gpus"].append(row)
            sample["owned_pids"] = sorted(owned)
            samples.write(
                json.dumps(
                    sample,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )
            samples.flush()
            if violations:
                os.killpg(pgid, signal.SIGTERM)
                signal_token_owned_processes(
                    ownership_token,
                    signal.SIGTERM,
                )
                break
            time.sleep(1.0)
        try:
            returncode = process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(pgid, signal.SIGKILL)
            signal_token_owned_processes(
                ownership_token,
                signal.SIGKILL,
            )
            returncode = process.wait(timeout=60)
    time.sleep(1.0)
    remaining = sorted(set(
        process_group_pids(pgid)
        + token_owned_pids(ownership_token)
    ))
    sqlite_path = output_prefix.with_suffix(".sqlite")
    return {
        "case_id": selected_case_id,
        "attempt_index": attempt_index,
        "command": command,
        "dist_port": dist_port,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "ready_marker_path": str(marker_path),
        "result_path": str(result_path),
        "sqlite_path": str(sqlite_path),
        "pid": process.pid,
        "pgid": pgid,
        "ownership_token": ownership_token,
        "returncode": returncode,
        "violations": violations,
        "process_group_destroyed": not remaining,
        "owned_children_remaining": remaining,
        "result_exists": result_path.is_file(),
        "sqlite_exists": sqlite_path.is_file(),
    }


def validate_existing_case(
    case: dict,
    *,
    representatives: dict[str, int],
) -> dict | None:
    selected_case_id = case_id(case)
    result_path = NSYS_CASES / f"{selected_case_id}.json"
    sqlite_path = (
        NSYS / f"{case['workload']}-r{case['repetition']}.sqlite"
    )
    if not result_path.exists() and not sqlite_path.exists():
        return None
    if not result_path.is_file() or not sqlite_path.is_file():
        raise RuntimeError(
            f"incomplete existing Nsight case {selected_case_id}"
        )
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if (
        payload.get("classification") != "PASS"
        or payload.get("case_id") != selected_case_id
        or payload.get("phase") != "nsys_replay"
    ):
        raise RuntimeError(
            f"invalid existing Nsight case {selected_case_id}"
        )
    expected_representative = (
        representatives[case["workload"]] == case["repetition"]
    )
    payload["representative"] = expected_representative
    payload["sqlite_path"] = str(sqlite_path)
    canonical_write(result_path, payload)
    return payload


def complete_case(
    case: dict,
    *,
    representatives: dict[str, int],
) -> dict:
    payload = validate_existing_case(
        case,
        representatives=representatives,
    )
    if payload is None:
        raise RuntimeError(
            f"Nsight case did not produce artifacts: {case_id(case)}"
        )
    return payload


def build_overhead_controls(
    results: list[dict],
    measured: dict[str, dict[int, int]],
) -> list[dict]:
    controls = []
    selected_uuids = [uuid for _, uuid in SELECTED]
    for result in results:
        workload = result["workload"]
        repetition = result["repetition"]
        profiled_ns = result["decode_time_ns"]
        unprofiled_ns = measured[workload][repetition]
        controls.append({
            "workload": workload,
            "repetition": repetition,
            "source_tree_sha256": (
                "dfdf6e758cbaa52fa24d8fa99550a709a8bf8bf81f8bc6d3f53842ec9c1a0654"
            ),
            "model_revision": (
                "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
            ),
            "rank_inventory": [0, 1, 2, 3],
            "gpu_uuids": selected_uuids,
            "unprofiled_ns": unprofiled_ns,
            "profiled_ns": profiled_ns,
            "relative_overhead": (
                profiled_ns / unprofiled_ns - 1.0
            ),
        })
    return controls


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-case")
    parser.add_argument("--result-path", type=Path)
    parser.add_argument("--smoke-case")
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS,
    )
    args = parser.parse_args()
    if args.max_attempts <= 0:
        parser.error("--max-attempts must be positive")
    if args.run_case is not None:
        if args.result_path is None or args.smoke_case is not None:
            parser.error(
                "--run-case requires --result-path and excludes "
                "--smoke-case"
            )
        return run_case(
            args.run_case,
            result_path=args.result_path,
            timeout_s=args.timeout_s,
        )
    if args.result_path is not None:
        parser.error("--result-path requires --run-case")

    for path in (SOURCE, MODEL, PYTHON, CASES):
        if not path.exists():
            raise RuntimeError(f"required path is missing: {path}")
    if len(list(CASES.glob("*.json"))) != 35:
        raise RuntimeError(
            "structured campaign must complete before Nsight"
        )
    NSYS.mkdir(parents=True, exist_ok=True)
    NSYS_CASES.mkdir(parents=True, exist_ok=True)
    measured = measured_decode_times()
    representatives = select_representatives(measured)
    cases = list(nsys_cases())
    if args.smoke_case is not None:
        smoke_case = parse_case_id(args.smoke_case)
        cases = [smoke_case]
    selected = wait_for_clean_entry()
    canonical_write(
        CONTROLLER / (
            "nsys-smoke-admission.json"
            if args.smoke_case is not None
            else "nsys-admission.json"
        ),
        {
            "captured_at_unix_ns": time.time_ns(),
            "selected_gpus": selected,
            "case_ids": [case_id(case) for case in cases],
            "representatives": representatives,
        },
    )

    results = []
    batch_results = []
    recovered_interference = []
    terminal_failure = False
    for case in cases:
        existing = validate_existing_case(
            case,
            representatives=representatives,
        )
        if existing is not None:
            results.append(existing)
            continue
        attempt_index = 0
        while True:
            wait_for_clean_entry()
            attempt = monitor_case(
                case,
                attempt_index=attempt_index,
                timeout_s=args.timeout_s,
            )
            batch_results.append(attempt)
            disposition = attempt_disposition(
                attempt,
                attempt_index=attempt_index,
                max_attempts=args.max_attempts,
            )
            attempt["disposition"] = disposition
            if disposition == "retry":
                recovered_interference.append({
                    "case_id": attempt["case_id"],
                    "attempt_index": attempt["attempt_index"],
                    "violations": attempt["violations"],
                })
                attempt_index += 1
                continue
            if (
                attempt["returncode"] != 0
                or attempt["violations"]
                or not attempt["process_group_destroyed"]
                or attempt["owned_children_remaining"]
                or not attempt["result_exists"]
                or not attempt["sqlite_exists"]
            ):
                terminal_failure = True
            else:
                results.append(complete_case(
                    case,
                    representatives=representatives,
                ))
            break
        if terminal_failure:
            break

    expected_count = 1 if args.smoke_case is not None else 25
    success = not terminal_failure and len(results) == expected_count
    receipt = {
        "classification": "PASS" if success else "FAIL",
        "mode": "smoke" if args.smoke_case is not None else "full",
        "completed_case_count": len(results),
        "case_ids": [result["case_id"] for result in results],
        "representatives": representatives,
        "batch_results": batch_results,
        "overhead_controls": build_overhead_controls(
            results,
            measured,
        ),
        "recovered_interference": recovered_interference,
        "violations": [
            violation
            for result in batch_results
            if result.get("disposition") != "retry"
            for violation in result["violations"]
        ],
        "process_groups_destroyed": all(
            result["process_group_destroyed"]
            for result in batch_results
        ),
        "owned_children_remaining": [
            pid
            for result in batch_results
            for pid in result["owned_children_remaining"]
        ],
    }
    canonical_write(
        CONTROLLER / (
            "nsys-smoke-receipt.json"
            if args.smoke_case is not None
            else "nsys-receipt.json"
        ),
        receipt,
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
