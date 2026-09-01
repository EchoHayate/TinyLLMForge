#!/usr/bin/env python3
"""Monitor and orchestrate the TP4 decode replay qualification gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tempfile
import time

if __package__:
    from tools.run_qwen38_tp4_communication_profile import (
        DEFAULT_SSH_TARGET,
        MAX_GPU_MEMORY_USED_MIB,
        MAX_GPU_UTILIZATION_PERCENT,
        build_ssh_argv,
        query_local_kerberos,
        query_remote_gpu_inventory,
        run_remote_argv,
        select_strict_clean_gpus,
        validate_selected_gpu_processes,
        wait_for_strict_clean_gpus,
    )
    from tools.assemble_tp4_decode_replay import (
        REQUIRED_INPUTS,
        assemble_bundle,
    )
    from tools.verify_tp4_decode_replay import verify_bundle
    from tools import tp4_decode_replay_contract as contract
else:
    from run_qwen38_tp4_communication_profile import (
        DEFAULT_SSH_TARGET,
        MAX_GPU_MEMORY_USED_MIB,
        MAX_GPU_UTILIZATION_PERCENT,
        build_ssh_argv,
        query_local_kerberos,
        query_remote_gpu_inventory,
        run_remote_argv,
        select_strict_clean_gpus,
        validate_selected_gpu_processes,
        wait_for_strict_clean_gpus,
    )
    from assemble_tp4_decode_replay import (
        REQUIRED_INPUTS,
        assemble_bundle,
    )
    from verify_tp4_decode_replay import verify_bundle
    import tp4_decode_replay_contract as contract


REMOTE_ROOT = (
    "/data00/home/sitian/tinyllmforge-workspaces/"
    "command-timeline-20260818/"
    "tp4-collective-stable-decode-replay"
)
MODEL_REPOSITORY = "Qwen/Qwen3.8-27B"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
MODEL_ROOT = (
    "/data00/home/sitian/tinyllmforge-workspaces/"
    "command-timeline-20260818/models/Qwen3.8-27B/"
    f"snapshots/{MODEL_REVISION}"
)
DEFAULT_REMOTE_PYTHON = "/data00/home/sitian/tllm/env/bin/python"
DEFAULT_COMMAND_TIMEOUT_S = 21_600
DEFAULT_GPU_WAIT_TIMEOUT_S = 21_600
DEFAULT_GPU_POLL_INTERVAL_S = 15
DEFAULT_RETRY_COUNT = 3
KERBEROS_GUARD_MARGIN_S = 900
PLAN_SCHEMA = "tinyllmforge.tp4-decode-replay-plan.v1"
RUN_TAG_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
SHA1_PATTERN = re.compile(r"^[0-9a-f]{40}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _below(path: str, root: str) -> bool:
    candidate = PurePosixPath(path)
    approved = PurePosixPath(root)
    return candidate.is_absolute() and candidate.is_relative_to(approved)


def _validate_run_tag(run_tag: object) -> str:
    if (
        not isinstance(run_tag, str)
        or not RUN_TAG_PATTERN.fullmatch(run_tag)
        or ".." in run_tag
    ):
        raise ValueError("run tag is invalid")
    return run_tag


def _validate_source_identity(
    source_identity: object,
    *,
    run_tag: str,
) -> dict:
    if (
        not isinstance(source_identity, dict)
        or source_identity.get("schema_version")
        != "tinyllmforge.tp4-decode-replay-source.v1"
        or source_identity.get("run_tag") != run_tag
        or not SHA1_PATTERN.fullmatch(
            str(source_identity.get("source_revision", ""))
        )
        or not SHA256_PATTERN.fullmatch(
            str(source_identity.get("source_tree_sha256", ""))
        )
        or source_identity.get("model_repository") != MODEL_REPOSITORY
        or source_identity.get("model_revision") != MODEL_REVISION
    ):
        raise ValueError("source identity is invalid")
    return dict(source_identity)


def _normalize_selected_gpus(selected_gpus: object) -> list[dict]:
    try:
        selected = select_strict_clean_gpus(list(selected_gpus))
    except (TypeError, ValueError) as error:
        raise ValueError(str(error)) from error
    if len(selected) != 4:
        raise ValueError("four strict-clean GPUs are required")
    return [dict(row) for row in selected]


def build_plan(
    *,
    run_tag: str,
    source_identity: dict,
    selected_gpus: list[dict],
) -> dict:
    run_tag = _validate_run_tag(run_tag)
    source = _validate_source_identity(
        source_identity,
        run_tag=run_tag,
    )
    selected = _normalize_selected_gpus(selected_gpus)
    attempt_root = f"{REMOTE_ROOT}/{run_tag}"
    runtime_root = f"{attempt_root}/runtime"
    paths = {
        "attempt_root": attempt_root,
        "source_root": f"{attempt_root}/source",
        "raw_root": f"{attempt_root}/raw",
        "bundle_root": f"{attempt_root}/final_bundle",
        "controller_root": f"{attempt_root}/controller",
        "worker_stdout_path": (
            f"{attempt_root}/controller/worker.stdout"
        ),
        "worker_stderr_path": (
            f"{attempt_root}/controller/worker.stderr"
        ),
        "remote_verification_path": (
            f"{attempt_root}/controller/"
            "remote_independent_verification.json"
        ),
        "remote_post_verification_manifest_path": (
            f"{attempt_root}/controller/"
            "remote_post_verification_manifest.json"
        ),
    }
    environment = {
        "TMPDIR": f"{runtime_root}/tmp",
        "XDG_CACHE_HOME": f"{runtime_root}/cache/xdg",
        "HF_HOME": f"{runtime_root}/cache/huggingface",
        "TRANSFORMERS_CACHE": (
            f"{runtime_root}/cache/huggingface/transformers"
        ),
        "TORCH_EXTENSIONS_DIR": (
            f"{runtime_root}/cache/torch-extensions"
        ),
        "CUDA_CACHE_PATH": f"{runtime_root}/cache/cuda",
    }
    process_environment = {
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": (
            f"{paths['source_root']}:{paths['source_root']}/tools"
        ),
    }
    if (
        not all(_below(value, REMOTE_ROOT) for value in paths.values())
        or not all(
            _below(value, attempt_root)
            for value in environment.values()
        )
    ):
        raise ValueError("planned remote path escapes approved root")
    return {
        "schema_version": PLAN_SCHEMA,
        "run_tag": run_tag,
        "remote_root": REMOTE_ROOT,
        "world_size": 4,
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "model_root": MODEL_ROOT,
        "source_identity": source,
        "source_revision": source["source_revision"],
        "source_tree_sha256": source["source_tree_sha256"],
        "selected_gpus": selected,
        "selected_gpu_indices": [
            row["gpu_index"] for row in selected
        ],
        "selected_gpu_uuids": [
            row["gpu_uuid"] for row in selected
        ],
        "paths": paths,
        "environment": environment,
        "process_environment": process_environment,
    }


def _validate_plan(plan: object) -> dict:
    if (
        not isinstance(plan, dict)
        or plan.get("schema_version") != PLAN_SCHEMA
        or plan.get("remote_root") != REMOTE_ROOT
        or plan.get("world_size") != 4
        or plan.get("model_repository") != MODEL_REPOSITORY
        or plan.get("model_revision") != MODEL_REVISION
        or plan.get("model_root") != MODEL_ROOT
    ):
        raise ValueError("TP4 decode replay plan is invalid")
    run_tag = _validate_run_tag(plan.get("run_tag"))
    source = _validate_source_identity(
        plan.get("source_identity"),
        run_tag=run_tag,
    )
    if (
        plan.get("source_revision") != source["source_revision"]
        or plan.get("source_tree_sha256")
        != source["source_tree_sha256"]
    ):
        raise ValueError("plan source identity drift")
    selected = _normalize_selected_gpus(plan.get("selected_gpus", []))
    if (
        plan.get("selected_gpu_indices")
        != [row["gpu_index"] for row in selected]
        or plan.get("selected_gpu_uuids")
        != [row["gpu_uuid"] for row in selected]
    ):
        raise ValueError("plan GPU identity drift")
    attempt_root = plan.get("paths", {}).get("attempt_root")
    if (
        not isinstance(attempt_root, str)
        or attempt_root != f"{REMOTE_ROOT}/{run_tag}"
        or not all(
            _below(value, REMOTE_ROOT)
            for value in plan.get("paths", {}).values()
        )
        or not all(
            _below(value, attempt_root)
            for value in plan.get("environment", {}).values()
        )
        or plan.get("process_environment")
        != {
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": (
                f"{plan['paths']['source_root']}:"
                f"{plan['paths']['source_root']}/tools"
            ),
        }
    ):
        raise ValueError("plan remote path is invalid")
    return dict(plan)


def _validate_preflight(preflight: object, plan: dict) -> None:
    if (
        not isinstance(preflight, dict)
        or preflight.get("classification") != "PASS"
        or preflight.get("attempt_exists") is not False
        or preflight.get("remote_root") != plan["remote_root"]
    ):
        raise ValueError("SSH/storage preflight rejected the run")


def _validate_admission(admission: object, plan: dict) -> None:
    if (
        not isinstance(admission, dict)
        or admission.get("classification") != "READY"
        or not isinstance(admission.get("selected_gpus"), list)
    ):
        raise ValueError("strict-clean admission rejected the run")
    selected = _normalize_selected_gpus(admission["selected_gpus"])
    if (
        [row["gpu_index"] for row in selected]
        != plan["selected_gpu_indices"]
        or [row["gpu_uuid"] for row in selected]
        != plan["selected_gpu_uuids"]
    ):
        raise ValueError("strict-clean GPU identity drift")


def _validate_verification_receipt(receipt: object) -> dict:
    if (
        not isinstance(receipt, dict)
        or receipt.get("classification") not in contract.CLASSIFICATIONS
        or not isinstance(receipt.get("failed_gates"), list)
        or any(
            not isinstance(gate, str) or not gate
            for gate in receipt["failed_gates"]
        )
        or not isinstance(receipt.get("verified_hashes"), bool)
        or not isinstance(
            receipt.get("producer_classification_matches"),
            bool,
        )
        or not isinstance(receipt.get("summary_matches"), bool)
        or not isinstance(receipt.get("metrics"), dict)
    ):
        raise ValueError("verification receipt is incomplete")
    if (
        receipt["classification"] != "INCOMPLETE"
        and (
            receipt["verified_hashes"] is not True
            or receipt["producer_classification_matches"] is not True
            or receipt["summary_matches"] is not True
        )
    ):
        raise ValueError("verification receipt is inconsistent")
    return dict(receipt)


def _execute_attempt(
    *,
    plan: dict,
    adapter: object,
    source: dict | None = None,
    preflight: dict | None = None,
) -> dict:
    plan = _validate_plan(plan)
    operation_error = None
    launch = None
    assembled = None
    remote_verification = None
    local_verification = None
    post_manifest = None
    cleanup = None
    try:
        if source is None:
            source = adapter.freeze_source(plan)
        if source != plan["source_identity"]:
            raise ValueError("source drift detected after freeze")
        if preflight is None:
            preflight = adapter.ssh_storage_preflight(plan, source)
        _validate_preflight(preflight, plan)
        admission = adapter.strict_clean_admission(plan, preflight)
        _validate_admission(admission, plan)
        launch = adapter.launch(plan, admission)
        waited = adapter.wait(plan, launch)
        downloaded = adapter.download(plan, waited)
        assembled = adapter.assemble(plan, downloaded)
        remote_verification = _validate_verification_receipt(
            adapter.remote_verify(plan, assembled)
        )
        post_manifest = (
            adapter.write_remote_post_verification_manifest(
                plan,
                remote_verification,
            )
        )
        local_verification = _validate_verification_receipt(
            adapter.local_verify(plan, assembled)
        )
        classifications = {
            assembled.get("classification"),
            remote_verification.get("classification"),
            local_verification.get("classification"),
        }
        if len(classifications) != 1 or None in classifications:
            raise RuntimeError(
                "producer and verifier classifications disagree"
            )
    except Exception as error:
        operation_error = error
    try:
        cleanup = adapter.validate_cleanup(plan, launch)
    except Exception as cleanup_error:
        if operation_error is not None:
            raise operation_error from cleanup_error
        raise
    if (
        not isinstance(cleanup, dict)
        or cleanup.get("classification") != "CLEAN"
    ):
        cleanup_error = RuntimeError(
            "cleanup validation did not prove a clean exit"
        )
        if operation_error is not None:
            raise operation_error from cleanup_error
        raise cleanup_error
    if operation_error is not None:
        raise operation_error
    return {
        "classification": assembled["classification"],
        "plan": plan,
        "source_identity": source,
        "preflight": preflight,
        "producer": assembled,
        "remote_verification": remote_verification,
        "remote_post_verification_manifest": post_manifest,
        "local_verification": local_verification,
        "cleanup": cleanup,
    }


def run_attempt(*, plan: dict, adapter: object) -> dict:
    return _execute_attempt(plan=plan, adapter=adapter)


def monitor_and_run(
    *,
    run_tag: str,
    gpu_monitor: object,
    adapter: object,
) -> dict:
    if not callable(gpu_monitor):
        raise ValueError("GPU monitor is invalid")
    seed = {"run_tag": _validate_run_tag(run_tag), "remote_root": REMOTE_ROOT}
    source = adapter.freeze_source(seed)
    source = _validate_source_identity(source, run_tag=run_tag)
    preflight = adapter.ssh_storage_preflight(seed, source)
    if (
        not isinstance(preflight, dict)
        or preflight.get("classification") != "PASS"
        or preflight.get("attempt_exists") is not False
        or preflight.get("remote_root") != REMOTE_ROOT
    ):
        raise ValueError("SSH/storage preflight rejected the run")
    monitor = gpu_monitor()
    if (
        not isinstance(monitor, dict)
        or monitor.get("classification") != "READY"
        or not isinstance(monitor.get("selected_gpus"), list)
    ):
        raise RuntimeError("four strict-clean GPUs were not admitted")
    plan = build_plan(
        run_tag=run_tag,
        source_identity=source,
        selected_gpus=monitor["selected_gpus"],
    )
    return _execute_attempt(
        plan=plan,
        adapter=adapter,
        source=source,
        preflight=preflight,
    )


def _capture_source_identity(run_tag: str) -> dict:
    root = Path(__file__).resolve().parents[1]

    def git(*arguments):
        result = subprocess.run(
            ["git", "-C", str(root), *arguments],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr or "git command failed")
        return result.stdout

    revision = git("rev-parse", "HEAD").strip()
    dirty = git(
        "status",
        "--porcelain=v1",
        "--untracked-files=no",
        "--",
        "tinyvllm",
        "tools",
    )
    if dirty:
        raise ValueError("source archive scope has tracked changes")
    tree = git(
        "ls-tree",
        "-r",
        "--full-tree",
        revision,
        "tinyvllm",
        "tools",
    ).encode("utf-8")
    return {
        "schema_version": "tinyllmforge.tp4-decode-replay-source.v1",
        "run_tag": run_tag,
        "source_revision": revision,
        "source_tree_sha256": hashlib.sha256(tree).hexdigest(),
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
    }


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


def _remote_driver_source() -> str:
    return r"""
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time

source_root, model_root, raw_root, run_tag, admission_json = sys.argv[1:]
sys.path[:0] = [source_root, str(Path(source_root) / "tools")]
import torch
import tp4_decode_replay_contract as contract
import tp4_decode_replay_worker as worker

raw = Path(raw_root)
cases_root = raw / "cases"
cases_root.mkdir(parents=True, exist_ok=True)
admission = json.loads(admission_json)

def write_json(path, payload):
    worker._atomic_write_json(path, payload)

def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ))
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)

used_ports = set()
def free_port():
    for _ in range(128):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
            handle.bind(("127.0.0.1", 0))
            port = handle.getsockname()[1]
        if port not in used_ports:
            used_ports.add(port)
            return port
    raise RuntimeError("could not allocate a fresh dynamic port")

aggregated = {
    "rank_dispatch_events.jsonl": [],
    "rank_collective_events.jsonl": [],
    "rank_lifecycle_rows.jsonl": [],
    "request_rows.jsonl": [],
    "performance_rows.jsonl": [],
    "memory_rows.jsonl": [],
    "correctness_rows.jsonl": [],
    "capture_cost_rows.jsonl": [],
}
process_rows = []
matrix = list(contract.build_case_matrix())
pair_ids = []
for case in matrix:
    if case["pair_id"] not in pair_ids:
        pair_ids.append(case["pair_id"])
for pair_id in pair_ids:
    pair_cases = tuple(
        case for case in matrix if case["pair_id"] == pair_id
    )
    case_results = {}
    for case in sorted(
        pair_cases,
        key=lambda row: row["order_index"],
    ):
        port = free_port()
        case_id = case["case_id"]
        environment = os.environ.copy()
        environment["TINYVLLM_DIST_PORT"] = str(port)
        process_logs = raw / "process-logs"
        process_logs.mkdir(parents=True, exist_ok=True)
        started_ns = time.monotonic_ns()
        completed = subprocess.run(
            [
                sys.executable,
                str(Path(source_root) / "tools"
                    / "tp4_decode_replay_worker.py"),
                "--model-root",
                model_root,
                "--case-json",
                json.dumps(
                    case,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "--output-dir",
                str(cases_root),
            ],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        finished_ns = time.monotonic_ns()
        (process_logs / f"{case_id}.stdout").write_text(
            completed.stdout,
            encoding="utf-8",
        )
        (process_logs / f"{case_id}.stderr").write_text(
            completed.stderr,
            encoding="utf-8",
        )
        process_rows.append({
            "case_id": case_id,
            "exit_code": completed.returncode,
            "timed_out": False,
            "dist_port": port,
            "started_ns": started_ns,
            "finished_ns": finished_ns,
        })
        if completed.returncode != 0:
            raise RuntimeError(
                "isolated arm failed: "
                + case_id
                + ": "
                + completed.stderr[-12000:]
            )
        result_path = cases_root / f"{case_id}.json"
        if not result_path.is_file():
            raise RuntimeError(
                "isolated arm did not write its case result: "
                + case_id
            )
        result = json.loads(result_path.read_text(encoding="utf-8"))
        case_results[case["arm"]] = result
    pair = worker.assemble_pair_result(
        pair_cases=pair_cases,
        arm_results=[
            case_results[case["arm"]]
            for case in sorted(
                pair_cases,
                key=lambda row: row["order_index"],
            )
        ],
    )
    aggregated["correctness_rows.jsonl"].append(
        pair["correctness_row"]
    )
    for arm in pair["arm_results"]:
        aggregated["rank_dispatch_events.jsonl"].extend(
            arm["rank_dispatch_rows"]
        )
        aggregated["rank_collective_events.jsonl"].extend(
            arm["rank_collective_rows"]
        )
        aggregated["rank_lifecycle_rows.jsonl"].extend(
            arm["rank_lifecycle_rows"]
        )
        aggregated["request_rows.jsonl"].extend(arm["request_rows"])
        aggregated["performance_rows.jsonl"].extend(
            arm["performance_rows"]
        )
        aggregated["memory_rows.jsonl"].extend(arm["memory_rows"])
        aggregated["capture_cost_rows.jsonl"].extend(
            arm["capture_cost_rows"]
        )

write_json(raw / "environment.json", {
    "schema_version": "tinyllmforge.tp4-decode-replay-environment.v1",
    "run_tag": run_tag,
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
})
write_json(raw / "workload_profile.json", {
    "schema_version": "tinyllmforge.tp4-decode-replay-workload.v1",
    "run_tag": run_tag,
    "model_repository": "Qwen/Qwen3.8-27B",
    "model_revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    "dtype": "bfloat16",
    "tensor_parallel_size": 4,
    "temperature": 0.0,
    "measured_repetitions": contract.MEASURED_REPETITIONS,
    "workloads": contract.WORKLOADS,
    "cases": matrix,
})
write_json(raw / "process_receipts.json", {
    "schema_version": "tinyllmforge.tp4-decode-replay-processes.v1",
    "run_tag": run_tag,
    "case_rows": process_rows,
})
write_jsonl(raw / "rank_environment.jsonl", [
    {
        "row_id": "environment:rank-" + str(rank),
        "run_tag": run_tag,
        "rank": rank,
        "world_size": 4,
        "cuda_visible_device": str(
            admission["selected_gpus"][rank]["index"]
        ),
    }
    for rank in range(4)
])
for name, rows in aggregated.items():
    write_jsonl(raw / name, rows)
print(json.dumps({
    "classification": "WORKER_COMPLETE",
    "case_count": len(process_rows),
    "pair_count": len(pair_ids),
}, sort_keys=True))
"""


class ProductionAdapter:
    def __init__(
        self,
        *,
        run_tag: str,
        local_attempt_root: Path,
        ssh_target: str = DEFAULT_SSH_TARGET,
        remote_python: str = DEFAULT_REMOTE_PYTHON,
        control_path: str | None = None,
        command_timeout_s: int = DEFAULT_COMMAND_TIMEOUT_S,
        retry_count: int = DEFAULT_RETRY_COUNT,
        local_command_runner=subprocess.run,
        kerberos_query=query_local_kerberos,
    ):
        self.run_tag = _validate_run_tag(run_tag)
        self.local_attempt_root = Path(local_attempt_root).resolve()
        self.local_controller_root = (
            self.local_attempt_root / "controller"
        )
        self.local_raw_root = self.local_attempt_root / "raw"
        self.local_bundle_root = (
            self.local_attempt_root / "final_bundle"
        )
        self.ssh_target = ssh_target
        self.remote_python = remote_python
        self.control_path = control_path
        self.command_timeout_s = int(command_timeout_s)
        self.retry_count = int(retry_count)
        if not callable(local_command_runner):
            raise ValueError("local command runner is invalid")
        if not callable(kerberos_query):
            raise ValueError("Kerberos query is invalid")
        self.local_command_runner = local_command_runner
        self.kerberos_query = kerberos_query
        self._source = None
        self._admission = None
        self._cleanup = None
        self._process = None
        self._plan = None

    def _query_kerberos_window(self) -> dict:
        return self.kerberos_query(
            minimum_lifetime_seconds=(
                self.command_timeout_s + KERBEROS_GUARD_MARGIN_S
            ),
        )

    def _require_kerberos_window(self) -> dict:
        kerberos = self._query_kerberos_window()
        if kerberos.get("classification") not in {"READY", "PASS"}:
            raise RuntimeError("Kerberos TTL preflight failed")
        return kerberos

    def _remote(self, remote_argv, *, timeout_s=None):
        result = run_remote_argv(
            ssh_target=self.ssh_target,
            remote_argv=list(remote_argv),
            control_path=self.control_path,
            timeout_s=(
                self.command_timeout_s
                if timeout_s is None
                else int(timeout_s)
            ),
            retry_count=self.retry_count,
        )
        if result.returncode != 0:
            raise RuntimeError(
                result.stderr or "remote command failed"
            )
        return result

    def _upload_bytes(self, remote_path: str, payload: bytes) -> None:
        script = "\n".join([
            "import os,sys,tempfile",
            "path=sys.argv[1]",
            "payload=sys.stdin.buffer.read()",
            "directory=os.path.dirname(path)",
            "fd,temp=tempfile.mkstemp(prefix='.upload.',dir=directory)",
            "with os.fdopen(fd,'wb') as handle:",
            "  handle.write(payload)",
            "  handle.flush()",
            "  os.fsync(handle.fileno())",
            "os.replace(temp,path)",
        ])
        argv = build_ssh_argv(
            ssh_target=self.ssh_target,
            remote_argv=[
                "python3",
                "-c",
                script,
                remote_path,
            ],
            control_path=self.control_path,
        )
        result = None
        for attempt in range(self.retry_count):
            result = self.local_command_runner(
                argv,
                input=payload,
                capture_output=True,
                check=False,
                timeout=self.command_timeout_s,
            )
            if (
                result.returncode != 255
                or attempt + 1 == self.retry_count
            ):
                break
            time.sleep(1.0)
        assert result is not None
        if result.returncode != 0:
            raise RuntimeError(
                result.stderr.decode(errors="replace")
                or "remote upload failed"
            )

    def _stage_source(self, plan: dict) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        archive = subprocess.Popen(
            [
                "git",
                "-C",
                str(repo_root),
                "archive",
                "--format=tar",
                plan["source_revision"],
                "tinyvllm",
                "tools",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert archive.stdout is not None
        receiver = subprocess.run(
            build_ssh_argv(
                ssh_target=self.ssh_target,
                remote_argv=[
                    "tar",
                    "-xf",
                    "-",
                    "-C",
                    plan["paths"]["source_root"],
                ],
                control_path=self.control_path,
            ),
            stdin=archive.stdout,
            capture_output=True,
            check=False,
            timeout=self.command_timeout_s,
        )
        archive.stdout.close()
        archive_stderr = (
            archive.stderr.read() if archive.stderr is not None else b""
        )
        archive_returncode = archive.wait()
        if archive_returncode != 0 or receiver.returncode != 0:
            raise RuntimeError(
                archive_stderr.decode(errors="replace")
                or receiver.stderr.decode(errors="replace")
                or "source staging failed"
            )

    def _download_archive(
        self,
        *,
        remote_root: str,
        names: tuple[str, ...],
        local_root: Path,
    ) -> None:
        local_root.mkdir(parents=True, exist_ok=True)
        sender = subprocess.Popen(
            build_ssh_argv(
                ssh_target=self.ssh_target,
                remote_argv=[
                    "tar",
                    "-cf",
                    "-",
                    "-C",
                    remote_root,
                    *names,
                ],
                control_path=self.control_path,
            ),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert sender.stdout is not None
        receiver = subprocess.run(
            ["tar", "-xf", "-", "-C", str(local_root)],
            stdin=sender.stdout,
            capture_output=True,
            check=False,
            timeout=self.command_timeout_s,
        )
        sender.stdout.close()
        sender_stderr = (
            sender.stderr.read() if sender.stderr is not None else b""
        )
        sender_returncode = sender.wait()
        if sender_returncode != 0 or receiver.returncode != 0:
            raise RuntimeError(
                sender_stderr.decode(errors="replace")
                or receiver.stderr.decode(errors="replace")
                or "compact evidence download failed"
            )

    def _upload_bundle(self, plan: dict) -> None:
        archive = subprocess.Popen(
            [
                "tar",
                "-cf",
                "-",
                "-C",
                str(self.local_attempt_root),
                "final_bundle",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert archive.stdout is not None
        receiver = subprocess.run(
            build_ssh_argv(
                ssh_target=self.ssh_target,
                remote_argv=[
                    "tar",
                    "-xf",
                    "-",
                    "-C",
                    plan["paths"]["attempt_root"],
                ],
                control_path=self.control_path,
            ),
            stdin=archive.stdout,
            capture_output=True,
            check=False,
            timeout=self.command_timeout_s,
        )
        archive.stdout.close()
        archive_stderr = (
            archive.stderr.read() if archive.stderr is not None else b""
        )
        archive_returncode = archive.wait()
        if archive_returncode != 0 or receiver.returncode != 0:
            raise RuntimeError(
                archive_stderr.decode(errors="replace")
                or receiver.stderr.decode(errors="replace")
                or "final bundle upload failed"
            )

    def _scan_exact_tag(self, plan: dict) -> list[dict]:
        script = "\n".join([
            "import json,os,sys",
            "tag=sys.argv[1]",
            "attempt_root=sys.argv[2]",
            "tag_bytes=tag.encode()",
            "excluded={os.getpid(),os.getppid()}",
            "rows=[]",
            "for name in os.listdir('/proc'):",
            "  if not name.isdigit(): continue",
            "  pid=int(name)",
            "  if pid in excluded: continue",
            "  try:",
            "    data=open('/proc/'+name+'/cmdline','rb').read()",
            "  except OSError:",
            "    data=b''",
            "  try:",
            "    environment=open('/proc/'+name+'/environ','rb').read()",
            "  except OSError:",
            "    environment=b''",
            "  command=data.replace(b'\\0',b' ').decode(errors='replace')",
            (
                "  if attempt_root in command "
                "or tag_bytes in environment:"
            ),
            "    rows.append({",
            "      'pid':pid,",
            "      'command':command,",
            "      'matched_cmdline':attempt_root in command,",
            "      'matched_environment':tag_bytes in environment,",
            "    })",
            "print(json.dumps(sorted(rows,key=lambda row:row['pid'])))",
        ])
        result = self._remote([
            "python3",
            "-c",
            script,
            plan["run_tag"],
            plan["paths"]["attempt_root"],
        ])
        rows = json.loads(result.stdout)
        if not isinstance(rows, list):
            raise ValueError("remote exact-tag scan is invalid")
        return rows

    def freeze_source(self, plan: dict) -> dict:
        run_tag = _validate_run_tag(plan["run_tag"])
        source = _capture_source_identity(run_tag)
        if self._source is not None and source != self._source:
            raise ValueError("frozen source identity drift")
        self._source = source
        self.local_controller_root.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(
            self.local_controller_root / "source_identity.json",
            source,
        )
        return dict(source)

    def ssh_storage_preflight(
        self,
        plan: dict,
        source: dict,
    ) -> dict:
        del source
        kerberos = self._query_kerberos_window()
        if kerberos.get("classification") not in {"READY", "PASS"}:
            receipt = {
                "classification": "INCOMPLETE",
                "reason": "Kerberos TTL preflight failed",
                "attempt_exists": False,
                "remote_root": REMOTE_ROOT,
                "kerberos": kerberos,
                "model_root": MODEL_ROOT,
                "model_revision": MODEL_REVISION,
            }
            _atomic_write_json(
                self.local_controller_root
                / "ssh_storage_preflight.json",
                receipt,
            )
            return receipt
        attempt_root = (
            plan.get("paths", {}).get("attempt_root")
            or f"{REMOTE_ROOT}/{self.run_tag}"
        )
        script = "\n".join([
            "import json,os,sys",
            "base,root,attempt,model,revision=sys.argv[1:]",
            "config_path=os.path.join(model,'config.json')",
            "config={}",
            "if os.path.isfile(config_path):",
            "  with open(config_path,encoding='utf-8') as handle:",
            "    config=json.load(handle)",
            "text=config.get('text_config',{}) if isinstance(config,dict) else {}",
            "print(json.dumps({",
            "'base_ready':os.path.isdir(base) and os.access(base,os.R_OK|os.W_OK|os.X_OK),",
            "'remote_root_safe':not os.path.islink(root) and (not os.path.exists(root) or os.path.isdir(root)),",
            "'attempt_exists':os.path.lexists(attempt),",
            "'model_ready':os.path.isdir(model) and os.access(model,os.R_OK|os.X_OK),",
            "'model_revision_matches':os.path.basename(os.path.realpath(model))==revision,",
            "'text_profile':{",
            "'num_hidden_layers':text.get('num_hidden_layers'),",
            "'hidden_size':text.get('hidden_size'),",
            "'vocab_size':text.get('vocab_size'),",
            "'dtype':text.get('dtype'),",
            "},",
            "},sort_keys=True))",
        ])
        base = str(PurePosixPath(REMOTE_ROOT).parent)
        result = self._remote([
            "python3",
            "-c",
            script,
            base,
            REMOTE_ROOT,
            attempt_root,
            MODEL_ROOT,
            MODEL_REVISION,
        ])
        state = json.loads(result.stdout)
        expected_profile = {
            "num_hidden_layers": 64,
            "hidden_size": 5120,
            "vocab_size": 248320,
            "dtype": "bfloat16",
        }
        if (
            not isinstance(state, dict)
            or state.get("base_ready") is not True
            or state.get("remote_root_safe") is not True
            or state.get("attempt_exists") is not False
            or state.get("model_ready") is not True
            or state.get("model_revision_matches") is not True
            or state.get("text_profile") != expected_profile
        ):
            raise ValueError("remote storage or model preflight failed")
        receipt = {
            "classification": "PASS",
            "attempt_exists": False,
            "remote_root": REMOTE_ROOT,
            "kerberos": kerberos,
            "model_root": MODEL_ROOT,
            "model_revision": MODEL_REVISION,
            "text_profile": expected_profile,
        }
        _atomic_write_json(
            self.local_controller_root
            / "ssh_storage_preflight.json",
            receipt,
        )
        return receipt

    def strict_clean_admission(
        self,
        plan: dict,
        preflight: dict,
    ) -> dict:
        del preflight
        self._require_kerberos_window()
        observed = query_remote_gpu_inventory(
            ssh_target=self.ssh_target,
            control_path=self.control_path,
            timeout_s=self.command_timeout_s,
            retry_count=self.retry_count,
        )
        selected = validate_selected_gpu_processes(
            selected=tuple(plan["selected_gpus"]),
            observed=list(observed),
            owned_pids=frozenset(),
        )
        if any(
            row["memory_used_mib"] > MAX_GPU_MEMORY_USED_MIB
            or row["utilization_percent"]
            > MAX_GPU_UTILIZATION_PERCENT
            for row in selected
        ):
            raise RuntimeError(
                "planned GPU inventory is not strict-clean"
            )
        rows = [
            {
                "rank": rank,
                "index": row["gpu_index"],
                "uuid": row["gpu_uuid"],
                "memory_used_mib": row["memory_used_mib"],
                "utilization_percent": row["utilization_percent"],
                "compute_process_count": len(
                    row["compute_processes"]
                ),
            }
            for rank, row in enumerate(selected)
        ]
        self._admission = {
            "schema_version": (
                "tinyllmforge.tp4-decode-replay-admission.v1"
            ),
            "run_tag": plan["run_tag"],
            "strict_clean": True,
            "world_size": 4,
            "selected_gpus": rows,
        }
        receipt = {
            "classification": "READY",
            "selected_gpus": [dict(row) for row in selected],
        }
        _atomic_write_json(
            self.local_controller_root
            / "strict_clean_admission.json",
            self._admission,
        )
        return receipt

    def launch(self, plan: dict, admission: dict) -> dict:
        del admission
        if self._source is None or self._admission is None:
            raise RuntimeError("launch prerequisites are incomplete")
        self._plan = dict(plan)
        directories = [
            plan["paths"]["attempt_root"],
            plan["paths"]["source_root"],
            plan["paths"]["raw_root"],
            f"{plan['paths']['raw_root']}/cases",
            plan["paths"]["controller_root"],
            *plan["environment"].values(),
        ]
        script = "\n".join([
            "import os,sys",
            "paths=sys.argv[1:]",
            "for path in paths:",
            "  if os.path.lexists(path):",
            "    raise SystemExit('path already exists: '+path)",
            "for path in paths:",
            "  os.makedirs(path,exist_ok=False)",
        ])
        self._remote(["python3", "-c", script, *directories])
        self._stage_source(plan)
        self._upload_bytes(
            f"{plan['paths']['raw_root']}/source_manifest.json",
            json.dumps(
                self._source,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8") + b"\n",
        )
        patch = subprocess.run(
            [
                "git",
                "-C",
                str(Path(__file__).resolve().parents[1]),
                "diff",
                "--binary",
                plan["source_revision"],
                "--",
                "tinyvllm",
                "tools",
            ],
            capture_output=True,
            check=True,
        ).stdout
        if not patch:
            patch = (
                b"# no tracked diff relative to frozen source revision\n"
            )
        self._upload_bytes(
            f"{plan['paths']['raw_root']}/source.patch",
            patch,
        )
        admission_payload = json.dumps(
            self._admission,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        self._upload_bytes(
            f"{plan['paths']['raw_root']}/gpu_inventory.json",
            admission_payload.encode("utf-8") + b"\n",
        )
        environment = dict(plan["environment"])
        environment.update(plan["process_environment"])
        environment["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(index) for index in plan["selected_gpu_indices"]
        )
        wrapper = "\n".join([
            "import json,os,subprocess,sys",
            (
                "python,driver,stdout_path,stderr_path,"
                "environment_json,*arguments=sys.argv[1:]"
            ),
            "environment=os.environ.copy()",
            "environment.update(json.loads(environment_json))",
            "with open(stdout_path,'w',encoding='utf-8') as stdout_handle, open(stderr_path,'w',encoding='utf-8') as stderr_handle:",
            "  result=subprocess.run([python,'-c',driver,*arguments],env=environment,stdout=stdout_handle,stderr=stderr_handle,text=True,check=False)",
            "print(json.dumps({'returncode':result.returncode},sort_keys=True))",
            "raise SystemExit(result.returncode)",
        ])
        remote_argv = [
            "python3",
            "-c",
            wrapper,
            self.remote_python,
            _remote_driver_source(),
            plan["paths"]["worker_stdout_path"],
            plan["paths"]["worker_stderr_path"],
            json.dumps(environment, sort_keys=True),
            plan["paths"]["source_root"],
            MODEL_ROOT,
            plan["paths"]["raw_root"],
            plan["run_tag"],
            admission_payload,
        ]
        self._process = subprocess.Popen(
            build_ssh_argv(
                ssh_target=self.ssh_target,
                remote_argv=remote_argv,
                control_path=self.control_path,
            ),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return {
            "local_ssh_pid": self._process.pid,
            "owned_pids": [self._process.pid],
        }

    def wait(self, plan: dict, launch: dict) -> dict:
        del launch
        if self._process is None:
            raise RuntimeError("worker process was not launched")
        stdout, stderr = self._process.communicate(
            timeout=self.command_timeout_s
        )
        returncode = self._process.returncode
        scans = [self._scan_exact_tag(plan) for _ in range(3)]
        self._cleanup = {
            "schema_version": (
                "tinyllmforge.tp4-decode-replay-cleanup.v1"
            ),
            "run_tag": plan["run_tag"],
            "classification": (
                "CLEAN"
                if returncode == 0 and scans == [[], [], []]
                else "DIRTY"
            ),
            "owned_children_remaining": [
                row for scan in scans for row in scan
            ],
            "exact_tag_scans": scans,
            "rank_rows": [
                {
                    "rank": rank,
                    "exit_code": 0 if returncode == 0 else returncode,
                    "process_group_destroyed": returncode == 0,
                }
                for rank in contract.RANKS
            ],
        }
        _atomic_write_json(
            self.local_controller_root / "worker_wait.json",
            {
                "returncode": returncode,
                "stdout": stdout,
                "stderr": stderr,
            },
        )
        if returncode != 0:
            tail = self._remote([
                "python3",
                "-c",
                (
                    "import pathlib,sys;"
                    "p=pathlib.Path(sys.argv[1]);"
                    "print(p.read_text(errors='replace')[-12000:] "
                    "if p.is_file() else '')"
                ),
                plan["paths"]["worker_stderr_path"],
            ]).stdout
            raise RuntimeError(
                f"remote TP4 worker failed with {returncode}: {tail}"
            )
        return {"exit_code": 0, "stdout": stdout}

    def download(self, plan: dict, waited: dict) -> dict:
        if waited.get("exit_code") != 0 or self._cleanup is None:
            raise RuntimeError("worker completion evidence is missing")
        validation_script = "\n".join([
            "import json,os,sys",
            "root=sys.argv[1]",
            "names=sys.argv[2:]",
            "rows={}",
            "for name in names:",
            "  path=os.path.join(root,name)",
            "  payload=open(path,'rb').read()",
            "  if not payload:",
            "    raise SystemExit('empty artifact: '+name)",
            "  if name.endswith('.jsonl') and not payload.endswith(b'\\n'):",
            "    raise SystemExit('partial JSONL: '+name)",
            "  rows[name]=len(payload)",
            "print(json.dumps(rows,sort_keys=True))",
        ])
        self._remote([
            "python3",
            "-c",
            validation_script,
            plan["paths"]["raw_root"],
            *REQUIRED_INPUTS,
        ])
        self._download_archive(
            remote_root=plan["paths"]["raw_root"],
            names=tuple(REQUIRED_INPUTS),
            local_root=self.local_raw_root,
        )
        return {
            "downloaded": True,
            "raw_root": str(self.local_raw_root),
            "files": list(REQUIRED_INPUTS),
        }

    def assemble(self, plan: dict, download: dict) -> dict:
        if download.get("downloaded") is not True:
            raise RuntimeError("compact raw evidence was not downloaded")
        result = assemble_bundle(
            raw_root=self.local_raw_root,
            output_root=self.local_bundle_root,
            source_identity=self._source,
            launch_admission=self._admission,
            cleanup=self._cleanup,
        )
        self._upload_bundle(plan)
        return result

    def remote_verify(
        self,
        plan: dict,
        assembled: dict,
    ) -> dict:
        del assembled
        result = self._remote([
            self.remote_python,
            (
                f"{plan['paths']['source_root']}/tools/"
                "verify_tp4_decode_replay.py"
            ),
            "--bundle-root",
            plan["paths"]["bundle_root"],
        ])
        verification = json.loads(result.stdout)
        self._upload_bytes(
            plan["paths"]["remote_verification_path"],
            json.dumps(
                verification,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8") + b"\n",
        )
        _atomic_write_json(
            self.local_controller_root
            / "remote_independent_verification.json",
            verification,
        )
        return verification

    def write_remote_post_verification_manifest(
        self,
        plan: dict,
        remote_verification: dict,
    ) -> dict:
        del remote_verification
        script = "\n".join([
            "import hashlib,json,os,sys,tempfile",
            "verification,manifest,output=sys.argv[1:]",
            "def digest(path):",
            "  value=hashlib.sha256()",
            "  with open(path,'rb') as handle:",
            "    for chunk in iter(lambda:handle.read(1024*1024),b''):",
            "      value.update(chunk)",
            "  return value.hexdigest()",
            "payload={'schema_version':'tinyllmforge.tp4-decode-replay-post-verification.v1','artifacts':{'remote_independent_verification.json':digest(verification),'final_bundle/manifest.json':digest(manifest)}}",
            "directory=os.path.dirname(output)",
            "fd,temp=tempfile.mkstemp(prefix='.manifest.',dir=directory)",
            "with os.fdopen(fd,'w',encoding='utf-8') as handle:",
            "  json.dump(payload,handle,sort_keys=True,separators=(',',':'),allow_nan=False)",
            "  handle.write('\\n')",
            "  handle.flush()",
            "  os.fsync(handle.fileno())",
            "os.replace(temp,output)",
            "print(json.dumps(payload,sort_keys=True))",
        ])
        result = self._remote([
            "python3",
            "-c",
            script,
            plan["paths"]["remote_verification_path"],
            f"{plan['paths']['bundle_root']}/manifest.json",
            plan[
                "paths"
            ]["remote_post_verification_manifest_path"],
        ])
        payload = json.loads(result.stdout)
        _atomic_write_json(
            self.local_controller_root
            / "remote_post_verification_manifest.json",
            payload,
        )
        return payload

    def local_verify(
        self,
        plan: dict,
        assembled: dict,
    ) -> dict:
        del assembled
        if (
            self._source is None
            or plan.get("source_revision")
            != self._source.get("source_revision")
        ):
            raise RuntimeError("frozen source is unavailable")
        self.local_controller_root.mkdir(parents=True, exist_ok=True)
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(
            dir=self.local_controller_root,
            prefix=".frozen-source-",
        ) as directory:
            frozen_root = Path(directory)
            archive = self.local_command_runner(
                [
                    "git",
                    "-C",
                    str(repo_root),
                    "archive",
                    "--format=tar",
                    plan["source_revision"],
                    "tools/tp4_decode_replay_contract.py",
                    "tools/verify_tp4_decode_replay.py",
                ],
                capture_output=True,
                check=False,
            )
            if archive.returncode != 0:
                raise RuntimeError(
                    archive.stderr.decode(errors="replace")
                    or "local frozen-source archive failed"
                )
            extracted = self.local_command_runner(
                ["tar", "-xf", "-", "-C", str(frozen_root)],
                input=archive.stdout,
                capture_output=True,
                check=False,
            )
            if extracted.returncode != 0:
                raise RuntimeError(
                    extracted.stderr.decode(errors="replace")
                    or "local frozen-source extraction failed"
                )
            verified = self.local_command_runner(
                [
                    sys.executable,
                    str(
                        frozen_root
                        / "tools"
                        / "verify_tp4_decode_replay.py"
                    ),
                    "--bundle-root",
                    str(self.local_bundle_root),
                ],
                text=True,
                capture_output=True,
                check=False,
                timeout=self.command_timeout_s,
            )
            if verified.returncode != 0:
                raise RuntimeError(
                    verified.stderr
                    or "local frozen-source verifier failed"
                )
            result = json.loads(verified.stdout)
            if not isinstance(result, dict):
                raise ValueError(
                    "local frozen-source verifier output is invalid"
                )
        _atomic_write_json(
            self.local_controller_root
            / "local_frozen_source_verification.json",
            result,
        )
        return result

    def validate_cleanup(
        self,
        plan: dict,
        launch: dict | None,
    ) -> dict:
        del launch
        scans = [self._scan_exact_tag(plan) for _ in range(3)]
        if self._cleanup is None:
            self._cleanup = {
                "schema_version": (
                    "tinyllmforge.tp4-decode-replay-cleanup.v1"
                ),
                "run_tag": plan["run_tag"],
                "classification": (
                    "CLEAN" if scans == [[], [], []] else "DIRTY"
                ),
                "owned_children_remaining": [
                    row for scan in scans for row in scan
                ],
                "exact_tag_scans": scans,
            }
        elif scans != [[], [], []]:
            self._cleanup["classification"] = "DIRTY"
            self._cleanup["owned_children_remaining"] = [
                row for scan in scans for row in scan
            ]
            self._cleanup["exact_tag_scans"] = scans
        _atomic_write_json(
            self.local_controller_root / "cleanup.json",
            self._cleanup,
        )
        return dict(self._cleanup)


def _placeholder_gpus() -> list[dict]:
    return [
        {
            "gpu_index": index,
            "gpu_uuid": f"GPU-PLAN-ONLY-{index}",
            "memory_used_mib": 0,
            "utilization_percent": 0,
            "compute_processes": [],
        }
        for index in range(4)
    ]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_only = subparsers.add_parser("plan-only")
    plan_only.add_argument("--run-tag", required=True)
    plan_only.add_argument("--output", type=Path)
    monitor = subparsers.add_parser("monitor-and-run")
    monitor.add_argument("--run-tag", required=True)
    monitor.add_argument("--ssh-target", default=DEFAULT_SSH_TARGET)
    monitor.add_argument(
        "--remote-python",
        default=DEFAULT_REMOTE_PYTHON,
    )
    monitor.add_argument("--control-path")
    monitor.add_argument(
        "--command-timeout-s",
        type=int,
        default=DEFAULT_COMMAND_TIMEOUT_S,
    )
    monitor.add_argument(
        "--gpu-wait-timeout-s",
        type=int,
        default=DEFAULT_GPU_WAIT_TIMEOUT_S,
    )
    monitor.add_argument(
        "--gpu-poll-interval-s",
        type=int,
        default=DEFAULT_GPU_POLL_INTERVAL_S,
    )
    monitor.add_argument("--retry-count", type=int, default=DEFAULT_RETRY_COUNT)
    monitor.add_argument("--local-attempt-root", type=Path)
    return parser


def main(
    argv=None,
    *,
    source_identity_builder=_capture_source_identity,
    gpu_monitor=None,
    adapter_factory=None,
) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "plan-only":
        source = source_identity_builder(args.run_tag)
        plan = build_plan(
            run_tag=args.run_tag,
            source_identity=source,
            selected_gpus=_placeholder_gpus(),
        )
        payload = {"mode": "plan-only", **plan}
        if args.output is not None:
            _atomic_write_json(args.output, payload)
        print(json.dumps(payload, sort_keys=True))
        return 0
    local_attempt_root = (
        args.local_attempt_root
        if args.local_attempt_root is not None
        else (
            Path(__file__).resolve().parents[1]
            / "artifacts"
            / "tp4_decode_replay"
            / args.run_tag
        )
    )
    if adapter_factory is None:
        adapter_factory = lambda: ProductionAdapter(
            run_tag=args.run_tag,
            local_attempt_root=local_attempt_root,
            ssh_target=args.ssh_target,
            remote_python=args.remote_python,
            control_path=args.control_path,
            command_timeout_s=args.command_timeout_s,
            retry_count=args.retry_count,
        )
    adapter = adapter_factory()
    if gpu_monitor is None:
        def query_inventory():
            adapter._require_kerberos_window()
            return query_remote_gpu_inventory(
                ssh_target=args.ssh_target,
                control_path=args.control_path,
                timeout_s=args.command_timeout_s,
                retry_count=args.retry_count,
            )

        gpu_monitor = lambda: wait_for_strict_clean_gpus(
            query_inventory=query_inventory,
            timeout_s=args.gpu_wait_timeout_s,
            poll_interval_s=args.gpu_poll_interval_s,
        )
    if not callable(gpu_monitor) or not callable(adapter_factory):
        raise RuntimeError("production monitor adapter is unavailable")
    result = monitor_and_run(
        run_tag=args.run_tag,
        gpu_monitor=gpu_monitor,
        adapter=adapter,
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
