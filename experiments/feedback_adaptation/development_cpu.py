import argparse
import copy
import hashlib
import json
import os
import pathlib
import platform
import re
import subprocess
import sys
import time
import uuid

from development_tasks import catalog


SOURCE_ROOT = pathlib.Path(__file__).resolve().parent
SOURCE_NAMES = ("development_programs.py", "development_tasks.py", "development_cpu.py")


def digest(value):
    return hashlib.sha256(value).hexdigest()


LOADED_SOURCE_SHA256 = {
    name: digest((SOURCE_ROOT / name).read_bytes()) for name in SOURCE_NAMES
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, minimum=0):
    require(type(value) is int and minimum <= value <= 2**63 - 1, "invalid integer")


def exact(value, fields):
    require(type(value) is dict and set(value) == set(fields), "invalid fields")


def _manifest(mode, task_id, initial, run_id):
    require(type(run_id) is str and re.fullmatch("[0-9a-f]{32}", run_id), "invalid run ID")
    require(
        (mode == "audit" and task_id is None and initial is None)
        or (mode == "probe" and type(task_id) is str
            and task_id in ("dev-001", "dev-002", "dev-003", "dev-004")
            and initial in ("correct", "defect")),
        "invalid fixed dispatch",
    )
    for name, expected in LOADED_SOURCE_SHA256.items():
        require(digest((SOURCE_ROOT / name).read_bytes()) == expected, "source changed after import")
    tasks = json.loads(canonical(catalog()))
    schedule = []
    for task in tasks:
        if mode == "probe" and task["task_id"] != task_id:
            continue
        for role in ("correct", "defect", "other"):
            if mode == "probe" and role != initial:
                continue
            for case in task["private"]["cases"]:
                if mode == "probe" and case["id"] not in {
                    row["id"] for row in task["check_inputs"]
                }:
                    continue
                schedule.append({
                    "task_id": task["task_id"], "program": role, "case_id": case["id"],
                    "input": case["input"],
                    "source_sha256": digest(task["sources"][role].encode()),
                })
    return {
        "kind": "FIXED_DEVELOPMENT_MANIFEST", "version": 1, "run_id": run_id,
        "mode": mode, "task_id": task_id, "initial": initial,
        "source_sha256": dict(LOADED_SOURCE_SHA256), "tasks": tasks, "schedule": schedule,
        "identity": {"hostname": platform.node(), "uid": os.getuid(),
                     "python": platform.python_version()},
        "budget": {"calls": len(schedule), "cpu_seconds": 2, "wall_seconds": 5,
                   "output_bytes_per_stream": 2 * 1024 * 1024},
    }


def freeze(mode, task_id=None, initial=None):
    return _manifest(mode, task_id, initial, uuid.uuid4().hex)


def compare(manifest, observation):
    require(type(manifest) is dict, "invalid manifest")
    expected = _manifest(
        manifest.get("mode"), manifest.get("task_id"), manifest.get("initial"),
        manifest.get("run_id"),
    )
    require(canonical(manifest) == canonical(expected), "manifest changed")
    exact(observation, (
        "kind", "run_id", "worker_sha256", "hostname", "uid", "python", "limits",
        "records", "elapsed_ns",
    ))
    require(observation["kind"] == "FIXED_DEVELOPMENT_OUTPUT", "wrong output kind")
    require(observation["run_id"] == manifest["run_id"], "wrong run")
    require(observation["worker_sha256"] == manifest["source_sha256"]["development_programs.py"],
            "wrong worker")
    integer(observation["uid"])
    for key, value in manifest["identity"].items():
        require(type(observation[key]) is type(value) and observation[key] == value,
                "wrong worker identity")
    require(canonical(observation["limits"]) == canonical({"cpu_seconds": 2, "core_bytes": 0}),
            "worker limits mismatch")
    integer(observation["elapsed_ns"])
    records = observation["records"]
    require(type(records) is list and len(records) == len(manifest["schedule"]),
            "incomplete observations")
    observed = {}
    total_function_ns = 0
    for record, scheduled in zip(records, manifest["schedule"]):
        exact(record, ("task_id", "program", "case_id", "input", "value", "elapsed_ns"))
        for key in ("task_id", "program", "case_id", "input"):
            require(canonical(record[key]) == canonical(scheduled[key]), "record binding mismatch")
        integer(record["value"], -(2**63))
        integer(record["elapsed_ns"])
        observed[(record["task_id"], record["program"], record["case_id"])] = record["value"]
        total_function_ns += record["elapsed_ns"]
    require(total_function_ns <= observation["elapsed_ns"], "inconsistent function time")
    result = {
        "kind": "FIXED_DEVELOPMENT_COMPARISON",
        "scheduled_calls": len(manifest["schedule"]), "observed_calls": len(records),
        "function_wall_ns": total_function_ns, "worker_wall_ns": observation["elapsed_ns"],
        "benchmark_eligible": False,
    }
    if manifest["mode"] == "probe":
        task = next(task for task in manifest["tasks"] if task["task_id"] == manifest["task_id"])
        initial = manifest["initial"]
        cases = {row["id"]: row for row in task["private"]["cases"]}
        result["tool_view"] = {
            "initial_source_sha256": digest(task["sources"][initial].encode()),
            "results": [
                {"id": row["id"], "input": row["input"], "expected": cases[row["id"]]["expected"],
                 "actual": observed[(task["task_id"], initial, row["id"])], "status": "ok"}
                for row in task["check_inputs"]
            ],
        }
        return copy.deepcopy(result)
    summaries = []
    for task in manifest["tasks"]:
        task_id = task["task_id"]
        private = task["private"]
        cases = private["cases"]
        passed = {
            role: {case["id"]: observed[(task_id, role, case["id"])] == case["expected"]
                   for case in cases}
            for role in ("correct", "defect", "other")
        }
        counts = {role: sum(results.values()) for role, results in passed.items()}
        score_counts = {role: sum(results[case_id] for case_id in private["score_ids"])
                        for role, results in passed.items()}
        witness = private["witness_id"]
        feedback_supported = passed["correct"][witness] and not passed["defect"][witness]
        probe_ids = [row["id"] for row in task["check_inputs"]]
        information = private["verification_information"]
        information_verified = (
            witness in probe_ids if information == "sufficient"
            else witness not in probe_ids and all(passed["defect"][case_id] for case_id in probe_ids)
        )
        admitted = (
            counts["correct"] == len(cases)
            and all(score_counts[role] < len(private["score_ids"]) for role in ("defect", "other"))
            and feedback_supported and information_verified
        )
        summaries.append({
            "task_id": task_id, "family_id": task["family_id"], "domain_cases": len(cases),
            "score_cases": len(private["score_ids"]), "pass_counts": counts,
            "score_pass_counts": score_counts, "feedback_supported": feedback_supported,
            "information_verified": information_verified, "admitted": admitted,
            "failure_witnesses": {
                role: [case for case in cases if not passed[role][case["id"]]]
                for role in ("correct", "defect", "other")
            },
        })
    result.update(tasks=summaries, admitted=all(task["admitted"] for task in summaries))
    return copy.deepcopy(result)


def _write(path, data):
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _write_json(path, value):
    _write(path, (canonical(value) + "\n").encode())


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON field")
        result[key] = value
    return result


def run(output_dir, mode, task_id=None, initial=None):
    started = time.perf_counter_ns()
    manifest = freeze(mode, task_id, initial)
    output = pathlib.Path(output_dir).resolve()
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    for name, expected in manifest["source_sha256"].items():
        data = (SOURCE_ROOT / name).read_bytes()
        require(digest(data) == expected, "source changed before snapshot")
        _write(output / name, data)
    command = [
        sys.executable, "-I", "-B", str(output / "development_programs.py"),
        mode, *([task_id, initial] if mode == "probe" else []),
        "--run-id", manifest["run_id"],
    ]
    _write_json(output / "manifest.json", manifest)
    _write_json(output / "invocation.json", {
        "argv": command, "timeout_seconds": 5, "cwd": str(output),
        "environment": {"PATH": os.defpath, "LANG": "C", "LC_ALL": "C"},
    })
    status = "completed"
    error = None
    returncode = None
    stdout = b""
    stderr = b""
    launch_started = time.perf_counter_ns()
    try:
        process = subprocess.run(
            command, stdin=subprocess.DEVNULL, capture_output=True, timeout=5, cwd=output,
            env={"PATH": os.defpath, "LANG": "C", "LC_ALL": "C"}, check=False,
        )
        stdout, stderr, returncode = process.stdout, process.stderr, process.returncode
        if returncode != 0:
            status = "nonzero_exit"
    except subprocess.TimeoutExpired as failure:
        stdout, stderr = failure.stdout or b"", failure.stderr or b""
        status, error = "timeout", str(failure)
    except OSError as failure:
        status, error = "launch_error", str(failure)
    launch_wall = time.perf_counter_ns() - launch_started
    _write(output / "stdout.json", stdout)
    _write(output / "stderr.txt", stderr)
    transport = {
        "returncode": returncode, "launch_wall_ns": launch_wall,
        "stdout_bytes": len(stdout), "stderr_bytes": len(stderr),
        "stdout_sha256": digest(stdout), "stderr_sha256": digest(stderr),
    }
    _write_json(output / "transport.json", transport)
    comparison = None
    if status == "completed":
        if max(len(stdout), len(stderr)) > manifest["budget"]["output_bytes_per_stream"]:
            status = "output_limit"
        else:
            try:
                require(not stderr, "unexpected stderr")
                comparison = compare(manifest, json.loads(stdout, object_pairs_hook=_unique_object))
            except (ValueError, TypeError, KeyError, UnicodeError) as failure:
                status, error = "invalid_output", str(failure)
    result = {
        "kind": "FIXED_DEVELOPMENT_DIAGNOSTIC", "origin": "controlled_local_invocation",
        "run_id": manifest["run_id"], "manifest_sha256": digest(canonical(manifest).encode()),
        "status": status, "error": error, "isolated": False, "benchmark_eligible": False,
        "scheduled_calls": len(manifest["schedule"]),
        "observed_calls": None if comparison is None else comparison["observed_calls"],
        "comparison": comparison, "cpu_time_ns": None, "peak_rss_bytes": None,
        "controller_wall_ns": time.perf_counter_ns() - started,
    }
    _write_json(output / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description="Local fixed development CPU diagnostic; no model.")
    parser.add_argument("mode", choices=("audit", "probe"))
    parser.add_argument("--task-id", choices=("dev-001", "dev-002", "dev-003", "dev-004"))
    parser.add_argument("--initial", choices=("correct", "defect"))
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    try:
        result = run(args.output_dir, args.mode, args.task_id, args.initial)
    except (ValueError, OSError) as failure:
        parser.exit(2, f"{failure}\n")
    print(canonical(result))
    if result["status"] != "completed" or result["comparison"].get("admitted") is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
