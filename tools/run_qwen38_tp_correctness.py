from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


PLAN_SCHEMA = "tinyllmforge.qwen38-tp-correctness-plan.v1"
RECEIPT_SCHEMA = "tinyllmforge.qwen38-tp-correctness-runner-receipt.v1"
APPROVED_REMOTE_ROOT = Path(
    "/data00/home/sitian/tinyllmforge-workspaces/"
    "command-timeline-20260818"
).resolve()
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMAND_ORDER = (
    "official_tp1",
    "tinyllmforge_tp1",
    "tinyllmforge_tp4",
    "assemble",
    "verify",
)


def _require_below(path, root, label, *, allow_equal=False) -> Path:
    resolved = Path(path).resolve()
    root = Path(root).resolve()
    if resolved == root:
        if allow_equal:
            return resolved
        raise ValueError(f"{label} must be below approved remote root")
    if not resolved.is_relative_to(root):
        raise ValueError(f"{label} must be below approved remote root")
    return resolved


def _require_file(path, label) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"{label} must be a regular file")
    return resolved


def _require_directory(path, label) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_dir() or resolved.is_symlink():
        raise ValueError(f"{label} must be a directory")
    return resolved


def _require_sha256(value, label) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _positive_integer(value, label) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _command(
    name,
    argv,
    *,
    environment,
    cwd,
    output_paths,
):
    return {
        "name": name,
        "argv": [str(value) for value in argv],
        "env": dict(environment),
        "cwd": str(cwd),
        "output_paths": [str(path) for path in output_paths],
    }


def build_correctness_plan(
    *,
    attempt_root,
    source_root,
    model_root,
    model_manifest_path,
    source_tree_sha256,
    model_manifest_sha256,
    python_executable,
    torchrun_executable,
    gpu_indices,
    rendezvous_ports,
    prompt_token_ids,
    generated_tokens,
    topk,
    timeout_s,
) -> dict:
    approved = APPROVED_REMOTE_ROOT.resolve()
    attempt = _require_below(
        attempt_root,
        approved,
        "attempt_root",
    )
    source = _require_directory(source_root, "source_root")
    model = _require_directory(model_root, "model_root")
    manifest = _require_file(
        model_manifest_path,
        "model_manifest_path",
    )
    for path, label in (
        (source, "source_root"),
        (model, "model_root"),
        (manifest, "model_manifest_path"),
        (Path(python_executable), "python_executable"),
        (Path(torchrun_executable), "torchrun_executable"),
    ):
        _require_below(path, approved, label, allow_equal=False)

    source_tree_sha256 = _require_sha256(
        source_tree_sha256,
        "source_tree_sha256",
    )
    model_manifest_sha256 = _require_sha256(
        model_manifest_sha256,
        "model_manifest_sha256",
    )
    gpu_indices = tuple(gpu_indices)
    if (
        len(gpu_indices) != 4
        or len(set(gpu_indices)) != 4
        or any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            for index in gpu_indices
        )
    ):
        raise ValueError("gpu_indices must contain four distinct indices")
    rendezvous_ports = tuple(rendezvous_ports)
    if (
        len(rendezvous_ports) != 2
        or len(set(rendezvous_ports)) != 2
        or any(
            isinstance(port, bool)
            or not isinstance(port, int)
            or not 1024 <= port <= 65535
            for port in rendezvous_ports
        )
    ):
        raise ValueError("rendezvous ports must be distinct valid ports")
    prompt_token_ids = tuple(prompt_token_ids)
    if (
        not prompt_token_ids
        or any(
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or token_id < 0
            for token_id in prompt_token_ids
        )
    ):
        raise ValueError(
            "prompt_token_ids must contain non-negative integers"
        )
    generated_tokens = _positive_integer(
        generated_tokens,
        "generated_tokens",
    )
    topk = _positive_integer(topk, "topk")
    if (
        isinstance(timeout_s, bool)
        or not isinstance(timeout_s, (int, float))
        or float(timeout_s) <= 0
    ):
        raise ValueError("timeout_s must be positive")

    python_executable = Path(python_executable).resolve()
    torchrun_executable = Path(torchrun_executable).resolve()
    runner_path = source / "tools" / "run_qwen38_tp_correctness.py"
    verifier_path = source / "tools" / "qwen38_tp_correctness.py"
    rows_dir = attempt / "rows"
    process_dir = attempt / "processes"
    model_manifest_output = attempt / "model_manifest.json"
    source_manifest_output = attempt / "source_manifest.json"
    correctness_manifest_output = attempt / "correctness_manifest.json"
    correctness_rows_output = attempt / "correctness_rows.jsonl"
    cleanup_receipt_output = attempt / "cleanup_receipt.json"
    runner_receipt_output = attempt / "runner_receipt.json"
    common = [
        f"--source-root={source}",
        f"--model-root={model}",
        f"--model-manifest={manifest}",
        f"--source-tree-sha256={source_tree_sha256}",
        f"--model-manifest-sha256={model_manifest_sha256}",
        "--text-only",
        "--greedy",
        "--temperature=0",
        f"--generated-tokens={generated_tokens}",
        "--prompt-token-ids="
        + json.dumps(list(prompt_token_ids), separators=(",", ":")),
        f"--topk={topk}",
        "--dtype=bfloat16",
        "--disable-profiler",
    ]
    official_output = rows_dir / "official_tp1.jsonl"
    tp1_output = rows_dir / "tinyllmforge_tp1.jsonl"
    tp4_output = rows_dir / "tinyllmforge_tp4.jsonl"
    official_process = process_dir / "official_tp1.json"
    tp1_process = process_dir / "tinyllmforge_tp1.json"
    tp4_process = process_dir / "tinyllmforge_tp4.json"

    commands = {
        "official_tp1": _command(
            "official_tp1",
            [
                python_executable,
                runner_path,
                "worker",
                "--mode=official_tp1",
                f"--output={official_output}",
                f"--process-output={official_process}",
                *common,
            ],
            environment={"CUDA_VISIBLE_DEVICES": str(gpu_indices[0])},
            cwd=source,
            output_paths=(official_output, official_process),
        ),
        "tinyllmforge_tp1": _command(
            "tinyllmforge_tp1",
            [
                torchrun_executable,
                "--nproc-per-node=1",
                f"--master-port={rendezvous_ports[0]}",
                runner_path,
                "worker",
                "--mode=tinyllmforge_tp1",
                f"--output={tp1_output}",
                f"--process-output={tp1_process}",
                *common,
            ],
            environment={"CUDA_VISIBLE_DEVICES": str(gpu_indices[0])},
            cwd=source,
            output_paths=(tp1_output, tp1_process),
        ),
        "tinyllmforge_tp4": _command(
            "tinyllmforge_tp4",
            [
                torchrun_executable,
                "--nproc-per-node=4",
                f"--master-port={rendezvous_ports[1]}",
                runner_path,
                "worker",
                "--mode=tinyllmforge_tp4",
                f"--output={tp4_output}",
                f"--process-output={tp4_process}",
                *common,
            ],
            environment={
                "CUDA_VISIBLE_DEVICES": ",".join(
                    str(index) for index in gpu_indices
                )
            },
            cwd=source,
            output_paths=(tp4_output, tp4_process),
        ),
        "assemble": _command(
            "assemble",
            [
                python_executable,
                runner_path,
                "assemble",
                f"--attempt-root={attempt}",
                f"--model-manifest={manifest}",
                f"--official-rows={official_output}",
                f"--tinyllmforge-tp1-rows={tp1_output}",
                f"--tinyllmforge-tp4-rows={tp4_output}",
                f"--official-process={official_process}",
                f"--tinyllmforge-tp1-process={tp1_process}",
                f"--tinyllmforge-tp4-process={tp4_process}",
                f"--source-tree-sha256={source_tree_sha256}",
                f"--model-manifest-sha256={model_manifest_sha256}",
                "--prompt-token-ids="
                + json.dumps(
                    list(prompt_token_ids),
                    separators=(",", ":"),
                ),
                f"--generated-tokens={generated_tokens}",
                f"--topk={topk}",
            ],
            environment={},
            cwd=source,
            output_paths=(
                model_manifest_output,
                source_manifest_output,
                correctness_manifest_output,
                correctness_rows_output,
                cleanup_receipt_output,
            ),
        ),
        "verify": _command(
            "verify",
            [
                python_executable,
                verifier_path,
                attempt,
            ],
            environment={},
            cwd=source,
            output_paths=(),
        ),
    }
    write_paths = [
        official_output,
        tp1_output,
        tp4_output,
        official_process,
        tp1_process,
        tp4_process,
        model_manifest_output,
        source_manifest_output,
        correctness_manifest_output,
        correctness_rows_output,
        cleanup_receipt_output,
        runner_receipt_output,
    ]
    return {
        "schema_version": PLAN_SCHEMA,
        "attempt_root": str(attempt),
        "source_root": str(source),
        "model_root": str(model),
        "model_manifest_path": str(manifest),
        "source_tree_sha256": source_tree_sha256,
        "model_manifest_sha256": model_manifest_sha256,
        "gpu_indices": list(gpu_indices),
        "rendezvous_ports": list(rendezvous_ports),
        "prompt_token_ids": list(prompt_token_ids),
        "generated_tokens": generated_tokens,
        "topk": topk,
        "timeout_s": float(timeout_s),
        "command_order": list(_COMMAND_ORDER),
        "commands": commands,
        "write_paths": [str(path) for path in write_paths],
    }


def _write_receipt(path, receipt):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            receipt,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _validate_execution_plan(plan) -> Path:
    if (
        not isinstance(plan, dict)
        or plan.get("schema_version") != PLAN_SCHEMA
        or plan.get("command_order") != list(_COMMAND_ORDER)
        or not isinstance(plan.get("commands"), dict)
        or set(plan["commands"]) != set(_COMMAND_ORDER)
    ):
        raise ValueError("correctness plan schema mismatch")
    attempt = _require_below(
        plan.get("attempt_root"),
        APPROVED_REMOTE_ROOT,
        "attempt_root",
    )
    source = _require_below(
        plan.get("source_root"),
        APPROVED_REMOTE_ROOT,
        "source_root",
    )
    expected_runner = source / "tools" / "run_qwen38_tp_correctness.py"
    expected_verifier = source / "tools" / "qwen38_tp_correctness.py"
    write_paths = plan.get("write_paths")
    if not isinstance(write_paths, list) or not write_paths:
        raise ValueError("write_paths inventory is invalid")
    normalized_write_paths = {
        _require_below(path, attempt, "attempt_root")
        for path in write_paths
    }
    expected_output_paths = set()
    for name in _COMMAND_ORDER:
        command = plan["commands"][name]
        if (
            not isinstance(command, dict)
            or command.get("name") != name
            or not isinstance(command.get("argv"), list)
            or not command["argv"]
            or any(
                not isinstance(argument, str) or not argument
                for argument in command["argv"]
            )
            or not isinstance(command.get("env"), dict)
            or not isinstance(command.get("output_paths"), list)
        ):
            raise ValueError(f"{name} command schema mismatch")
        executable = _require_below(
            command["argv"][0],
            APPROVED_REMOTE_ROOT,
            "command executable",
        )
        if name == "official_tp1":
            valid_entry = (
                executable.name == "python"
                and len(command["argv"]) >= 4
                and Path(command["argv"][1]).resolve()
                == expected_runner
                and command["argv"][2:4]
                == ["worker", "--mode=official_tp1"]
            )
        elif name in {"tinyllmforge_tp1", "tinyllmforge_tp4"}:
            mode = name
            valid_entry = (
                executable.name == "torchrun"
                and expected_runner in {
                    Path(argument).resolve()
                    for argument in command["argv"]
                    if argument.startswith("/")
                }
                and "worker" in command["argv"]
                and f"--mode={mode}" in command["argv"]
            )
        elif name == "assemble":
            valid_entry = (
                executable.name == "python"
                and len(command["argv"]) >= 3
                and Path(command["argv"][1]).resolve()
                == expected_runner
                and command["argv"][2] == "assemble"
            )
        else:
            valid_entry = (
                executable.name == "python"
                and len(command["argv"]) == 3
                and Path(command["argv"][1]).resolve()
                == expected_verifier
                and Path(command["argv"][2]).resolve() == attempt
            )
        if not valid_entry:
            raise ValueError(f"{name} command executable mismatch")
        for path in command["output_paths"]:
            expected_output_paths.add(
                _require_below(path, attempt, "attempt_root")
            )
        serialized = json.dumps(command, sort_keys=True)
        if any(
            forbidden in serialized
            for forbidden in (
                "pkill",
                "killall",
                "kinit",
                "krenew",
                "adaptive-ngram",
                "/private/tmp",
            )
        ):
            raise ValueError(f"{name} command contains a forbidden action")
    receipt_path = attempt / "runner_receipt.json"
    expected_output_paths.add(receipt_path)
    if normalized_write_paths != expected_output_paths:
        raise ValueError("write_paths inventory mismatch")
    return attempt


def _normalize_command_result(name, result):
    if not isinstance(result, dict):
        raise ValueError("command result must be a mapping")
    remaining = result.get("owned_children_remaining", [])
    if not isinstance(remaining, list):
        raise ValueError("owned_children_remaining must be a list")
    for field in ("pid", "pgid", "returncode"):
        value = result.get(field)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{field} must be an integer")
    for field in ("stdout", "stderr"):
        if not isinstance(result.get(field, ""), str):
            raise ValueError(f"{field} must be a string")
    return {
        "name": name,
        "pid": result["pid"],
        "pgid": result["pgid"],
        "returncode": result["returncode"],
        "process_group_destroyed": result.get(
            "process_group_destroyed"
        ),
        "owned_children_remaining": list(remaining),
        "stdout": result.get("stdout", ""),
        "stderr": result.get("stderr", ""),
    }


def execute_correctness_plan(
    plan,
    *,
    run_command,
    verify_bundle,
) -> dict:
    if not callable(run_command) or not callable(verify_bundle):
        raise ValueError("runner dependencies must be callable")
    attempt = _validate_execution_plan(plan)
    attempt.mkdir(parents=True, exist_ok=True)
    receipt_path = attempt / "runner_receipt.json"
    processes = []
    failed_stage = None
    failure_reason = None
    verification = None

    for name in _COMMAND_ORDER[:-1]:
        command = plan["commands"][name]
        try:
            result = run_command(
                command,
                timeout_s=plan["timeout_s"],
            )
            row = _normalize_command_result(name, result)
        except Exception as error:
            failed_stage = name
            failure_reason = (
                f"{type(error).__name__}: {error}"
            )
            break
        processes.append(row)
        if row["returncode"] != 0:
            failed_stage = name
            failure_reason = f"nonzero exit code: {row['returncode']}"
            break
        if row["process_group_destroyed"] is not True:
            failed_stage = name
            failure_reason = "process group cleanup was not confirmed"
            break
        if row["owned_children_remaining"]:
            failed_stage = name
            failure_reason = "owned children remain after stage"
            break

    if failed_stage is None:
        try:
            verification = verify_bundle(attempt)
        except Exception as error:
            failed_stage = "verify"
            failure_reason = (
                f"{type(error).__name__}: {error}"
            )
        else:
            if (
                not isinstance(verification, dict)
                or verification.get("classification") != "PASS"
            ):
                failed_stage = "verify"
                failure_reason = "correctness verification did not pass"

    owned_children_remaining = sorted({
        child
        for row in processes
        for child in row["owned_children_remaining"]
    })
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "classification": (
            "PASS" if failed_stage is None else "FAIL"
        ),
        "failed_stage": failed_stage,
        "failure_reason": failure_reason,
        "attempt_root": str(attempt),
        "source_tree_sha256": plan["source_tree_sha256"],
        "model_manifest_sha256": plan["model_manifest_sha256"],
        "processes": processes,
        "owned_children_remaining": owned_children_remaining,
        "verification": verification,
    }
    _write_receipt(receipt_path, receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=("worker", "assemble"),
        help=(
            "Execution stages are launched only by an audited controller; "
            "this module currently exposes their immutable plan contract."
        ),
    )
    parser.parse_known_args(argv)
    raise RuntimeError(
        "Qwen3.8 correctness stage execution requires the audited "
        "campaign controller"
    )


if __name__ == "__main__":
    raise SystemExit(main())
