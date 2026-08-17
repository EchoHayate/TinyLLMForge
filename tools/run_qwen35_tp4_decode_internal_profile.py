#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import shlex
import subprocess
import sys
import tarfile
import tempfile
import time

import qwen35_tp4_decode_internal_profile as decode_profile
import qwen35_tp4_hybrid_prefix_benchmark_contract as contract
import run_qwen35_tp4_hybrid_prefix_benchmark_remote as base_runner


REMOTE_PYTHON = (
    "/data00/home/sitian/sitian-workspace01/tllm/"
    "env/bin/python"
)
WORKLOAD = "w2_long_reuse"
POLICIES = ("recompute", "exact_restore")
GPU_INDICES = (2, 4, 5, 6)
GENERATED_TOKENS = 8
NSYS_REQUESTED_TRACE = "cuda,nvtx,osrt,nccl"
NSYS_TRACE = "cuda,nvtx,osrt"
MIN_GPU_FREE_BYTES = 25 * 1024**3
MAX_GPU_UTILIZATION_PERCENT = 10
GUARD_POLL_INTERVAL_S = 30
GUARD_MAX_WAIT_S = 1800
SSH_TARGET = "sitian@10.232.195.203"
SSH_OPTIONS = (
    "-o",
    "BatchMode=yes",
    "-o",
    "ControlMaster=no",
    "-o",
    "ControlPath=none",
    "-o",
    "ConnectTimeout=20",
)
KRB5CCNAME = "FILE:/Users/bytedance/krb5cc_sitian"
REMOTE_ROOT = base_runner.REMOTE_ROOT
LOCAL_OUTPUT_ROOT = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "qwen35_hybrid_state"
)
DEFAULT_PREREQUISITES_TAR = (
    LOCAL_OUTPUT_ROOT
    / "qwen35-tp4-strict-p1-canonical-20260811-r607-attempt001"
    / "plan"
    / "correctness_prerequisites.tar"
)
DEFAULT_PREREQUISITES = (
    LOCAL_OUTPUT_ROOT
    / "qwen35-tp4-strict-p1-canonical-20260811-r607-attempt001"
    / "plan"
    / "correctness_prerequisites.json"
)
DEFAULT_REMOTE_MODEL_DIR = (
    "/data00/home/sitian/sitian-workspace01/tllm/"
    "qwen35-hybrid-state-runs/"
    "qwen35-2b-hybrid-acquire-20260723-222004/model"
)
DEFAULT_REMOTE_MODEL_MANIFEST = (
    "/data00/home/sitian/sitian-workspace01/tllm/"
    "qwen35-hybrid-state-runs/"
    "qwen35-2b-hybrid-acquire-20260723-222004/"
    "model_manifest.json"
)
DEFAULT_LOCAL_MODEL_MANIFEST = (
    LOCAL_OUTPUT_ROOT
    / "qwen35-tp4-strict-p1-canonical-20260811-r607-attempt001"
    / "plan"
    / "model_manifest.json"
)


@dataclass(frozen=True)
class WorkerAuthorization:
    prerequisites_sha256: str
    source_tree_sha256: str
    model_manifest_sha256: str
    workload_manifest_sha256: str
    gpu_indices: tuple[int, int, int, int]


@dataclass(frozen=True)
class WorkerRuntimeArtifacts:
    model_dir: str
    model_manifest_path: str
    correctness_prerequisites_path: str
    workload_manifest_path: str


@dataclass(frozen=True)
class ProfileCase:
    case_id: str
    workload: str
    policy: str
    phase: str
    repetition: int


@dataclass(frozen=True)
class WorkerCommand:
    case_id: str
    workload: str
    policy: str
    phase: str
    repetition: int
    output_dir: str
    cwd: str
    env: dict
    argv: list

    def __getitem__(self, key):
        return getattr(self, key)


def _case_id(phase, repetition, policy):
    return (
        f"{WORKLOAD}__{phase}__r{repetition}__{policy}"
    )


def _pair_order(repetition):
    if repetition % 2 == 0:
        return POLICIES
    return tuple(reversed(POLICIES))


def build_profile_cases():
    cases = []
    for phase, repetitions in (("warmup", 1), ("measured", 5)):
        for repetition in range(repetitions):
            for policy in _pair_order(repetition):
                cases.append(ProfileCase(
                    case_id=_case_id(
                        phase,
                        repetition,
                        policy,
                    ),
                    workload=WORKLOAD,
                    policy=policy,
                    phase=phase,
                    repetition=repetition,
                ))
    return tuple(cases)


def evaluate_shared_gpu_guard(rows):
    if not isinstance(rows, list):
        raise ValueError("GPU rows must be a list")
    by_index = {}
    reasons = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("GPU row is invalid")
        gpu_index = row.get("gpu_index")
        if gpu_index not in GPU_INDICES:
            continue
        if gpu_index in by_index:
            raise ValueError("GPU row is duplicate")
        gpu_uuid = row.get("gpu_uuid")
        free_bytes = row.get("free_bytes")
        utilization = row.get("utilization_percent")
        compute_processes = row.get("compute_processes")
        if (
            not isinstance(gpu_uuid, str)
            or not gpu_uuid.startswith("GPU-")
            or isinstance(free_bytes, bool)
            or not isinstance(free_bytes, int)
            or isinstance(utilization, bool)
            or not isinstance(utilization, int)
            or not isinstance(compute_processes, list)
        ):
            raise ValueError("GPU row is invalid")
        by_index[gpu_index] = row
        if free_bytes < MIN_GPU_FREE_BYTES:
            reasons.append(
                f"GPU {gpu_index} has less than 25 GiB free"
            )
        if utilization > MAX_GPU_UTILIZATION_PERCENT:
            reasons.append(
                f"GPU {gpu_index} utilization exceeds 10 percent"
            )
    missing = [
        index for index in GPU_INDICES if index not in by_index
    ]
    if missing:
        reasons.append("fixed GPUs 2,4,5,6 are incomplete")
    selected = [
        by_index[index]
        for index in GPU_INDICES
        if index in by_index
    ]
    return {
        "classification": (
            "READY" if not reasons else "BLOCKED_RESOURCES"
        ),
        "resource_policy": "shared-low-utilization",
        "exclusive": False,
        "minimum_gpu_free_bytes": MIN_GPU_FREE_BYTES,
        "maximum_gpu_utilization_percent": (
            MAX_GPU_UTILIZATION_PERCENT
        ),
        "selected_gpus": selected,
        "reasons": reasons,
    }


def _validate_authorization(authorization):
    if not isinstance(authorization, WorkerAuthorization):
        raise ValueError("worker authorization is required")
    for name in (
        "prerequisites_sha256",
        "source_tree_sha256",
        "model_manifest_sha256",
        "workload_manifest_sha256",
    ):
        value = getattr(authorization, name)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"{name} is invalid")
    if authorization.gpu_indices != GPU_INDICES:
        raise ValueError("GPU indices must be 2,4,5,6")
    return authorization


def _validate_runtime_artifacts(runtime_artifacts):
    if not isinstance(runtime_artifacts, WorkerRuntimeArtifacts):
        raise ValueError("worker runtime artifacts are required")
    for value in (
        runtime_artifacts.model_dir,
        runtime_artifacts.model_manifest_path,
        runtime_artifacts.correctness_prerequisites_path,
        runtime_artifacts.workload_manifest_path,
    ):
        path = PurePosixPath(value)
        if not path.is_absolute() or ".." in path.parts:
            raise ValueError("worker runtime artifact path is invalid")
    return runtime_artifacts


def build_structured_commands(
    *,
    remote_source,
    remote_cases,
    ports,
    authorization,
    runtime_artifacts,
):
    authorization = _validate_authorization(authorization)
    runtime_artifacts = _validate_runtime_artifacts(runtime_artifacts)
    cases = build_profile_cases()
    if len(ports) != len(cases):
        raise ValueError("worker port inventory mismatch")
    gpu_csv = ",".join(
        str(index) for index in authorization.gpu_indices
    )
    commands = []
    for case, port_pair in zip(cases, ports):
        if (
            not isinstance(port_pair, (tuple, list))
            or len(port_pair) != 2
            or port_pair[0] == port_pair[1]
        ):
            raise ValueError("worker port pair is invalid")
        dist_port, master_port = port_pair
        output_dir = f"{remote_cases}/{case.case_id}"
        argv = [
            REMOTE_PYTHON,
            "tools/qwen35_tp4_hybrid_prefix_benchmark_worker.py",
            "--policy",
            case.policy,
            "--workload",
            case.workload,
            "--phase",
            case.phase,
            "--repetition",
            str(case.repetition),
            "--output-dir",
            output_dir,
            "--prerequisites-sha256",
            authorization.prerequisites_sha256,
            "--source-tree-sha256",
            authorization.source_tree_sha256,
            "--model-manifest-sha256",
            authorization.model_manifest_sha256,
            "--model-dir",
            runtime_artifacts.model_dir,
            "--model-manifest",
            runtime_artifacts.model_manifest_path,
            "--correctness-prerequisites",
            runtime_artifacts.correctness_prerequisites_path,
            "--workload-manifest",
            runtime_artifacts.workload_manifest_path,
            "--workload-manifest-sha256",
            authorization.workload_manifest_sha256,
            "--profile",
            "--generated-tokens-override",
            str(GENERATED_TOKENS),
            "--decode-internal-profile",
        ]
        commands.append(WorkerCommand(
            case_id=case.case_id,
            workload=case.workload,
            policy=case.policy,
            phase=case.phase,
            repetition=case.repetition,
            output_dir=output_dir,
            cwd=remote_source,
            env={
                "CUDA_VISIBLE_DEVICES": gpu_csv,
                "TINYVLLM_DIST_PORT": str(dist_port),
                "MASTER_PORT": str(master_port),
                "PYTHONPATH": remote_source,
                "PYTHONDONTWRITEBYTECODE": "1",
            },
            argv=argv,
        ))
    return tuple(commands)


def _replace_option(argv, option, value):
    result = list(argv)
    index = result.index(option)
    result[index + 1] = str(value)
    return result


def build_nsys_command(
    *,
    structured_command,
    remote_nsys,
    report_prefix,
    repetition,
):
    if not isinstance(structured_command, WorkerCommand):
        raise ValueError("structured worker command is required")
    policy = structured_command.policy
    replay_case_id = _case_id(
        "nsys_replay",
        repetition,
        policy,
    )
    replay_output = str(
        PurePosixPath(structured_command.output_dir).parent
        / replay_case_id
    )
    worker_argv = _replace_option(
        structured_command.argv,
        "--phase",
        "nsys_replay",
    )
    worker_argv = _replace_option(
        worker_argv,
        "--repetition",
        repetition,
    )
    worker_argv = _replace_option(
        worker_argv,
        "--output-dir",
        replay_output,
    )
    return WorkerCommand(
        case_id=replay_case_id,
        workload=WORKLOAD,
        policy=policy,
        phase="nsys_replay",
        repetition=repetition,
        output_dir=replay_output,
        cwd=structured_command.cwd,
        env=dict(structured_command.env),
        argv=[
            remote_nsys,
            "profile",
            f"--trace={NSYS_TRACE}",
            "--force-overwrite=true",
            "--sample=none",
            "--output",
            report_prefix,
            *worker_argv,
        ],
    )


def orchestrate_attempt(
    *,
    entry_guard,
    stage,
    worker_guard,
    run_worker,
    aggregate,
    run_nsys,
    cleanup,
):
    entry = entry_guard()
    if entry.get("classification") != "READY":
        return {
            "classification": entry.get(
                "classification",
                "BLOCKED_RESOURCES",
            ),
            "entry_guard": entry,
            "preserve_attempt": True,
        }
    stage()
    structured_results = []
    result = None
    try:
        for case in build_profile_cases():
            guard = worker_guard(case)
            if guard.get("classification") != "READY":
                result = {
                    "classification": "BLOCKED_WORKER_ENTRY",
                    "worker_guard": guard,
                    "structured_workers": structured_results,
                    "preserve_attempt": True,
                }
                break
            worker_result = run_worker(case)
            structured_results.append(worker_result)
            if worker_result.get("returncode") != 0:
                result = {
                    "classification": "FAILED_STRUCTURED_WORKER",
                    "structured_workers": structured_results,
                    "preserve_attempt": True,
                }
                break
        if result is None:
            structured = aggregate()
            repetition = structured["representative_repetition"]
            nsys = run_nsys(repetition)
            classification = (
                "COMPLETE_WITHOUT_NSYS"
                if nsys.get("classification") == "NSYS_UNAVAILABLE"
                else "COMPLETE"
            )
            result = {
                "classification": classification,
                "structured": structured,
                "structured_workers": structured_results,
                "nsys": nsys,
                "preserve_attempt": True,
            }
    finally:
        cleanup_result = cleanup()
    result["cleanup"] = cleanup_result
    if cleanup_result.get("classification") != "CLEAN":
        result["classification"] = "FAILED_CLEANUP"
    return result


def _atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(
            payload,
            handle,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_prerequisites_bundle(archive_path, destination):
    archive_path = Path(archive_path)
    destination = Path(destination)
    if destination.exists():
        raise ValueError("prerequisite extraction destination exists")
    destination.mkdir(parents=True)
    with tarfile.open(archive_path, "r") as archive:
        members = archive.getmembers()
        if not members:
            raise ValueError("prerequisite archive is empty")
        for member in members:
            path = PurePosixPath(member.name)
            if (
                path.is_absolute()
                or ".." in path.parts
                or not member.isfile()
                or member.issym()
                or member.islnk()
            ):
                raise ValueError(
                    "prerequisite archive member is unsafe"
                )
        archive.extractall(destination, members=members)
    prerequisites = destination / "correctness_prerequisites.json"
    if not prerequisites.is_file():
        raise ValueError(
            "prerequisite archive lacks correctness prerequisites"
        )
    return prerequisites


def _execution_env():
    environment = dict(os.environ)
    environment["KRB5CCNAME"] = KRB5CCNAME
    return environment


def _ssh_argv(remote_argv):
    command = " ".join(
        shlex.quote(str(value)) for value in remote_argv
    )
    return [
        "ssh",
        *SSH_OPTIONS,
        SSH_TARGET,
        "bash",
        "-lc",
        shlex.quote(command),
    ]


def _shell_ssh_argv(script):
    if not isinstance(script, str) or not script:
        raise ValueError("remote shell script is required")
    return [
        "ssh",
        *SSH_OPTIONS,
        SSH_TARGET,
        "bash",
        "-lc",
        shlex.quote(script),
    ]


def _scp_argv(source, destination, *, recursive=False):
    return [
        "scp",
        *SSH_OPTIONS,
        *(["-r"] if recursive else []),
        str(source),
        str(destination),
    ]


def _run_subprocess(
    argv,
    *,
    timeout_s=120.0,
    stdout_path=None,
):
    attempts = 3 if argv[0] in {"ssh", "scp"} else 1
    result = None
    for attempt in range(attempts):
        if stdout_path is None:
            result = subprocess.run(
                argv,
                check=False,
                text=True,
                capture_output=True,
                timeout=timeout_s,
                env=_execution_env(),
            )
        else:
            output = Path(stdout_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("wb") as stdout_handle:
                result = subprocess.run(
                    argv,
                    check=False,
                    stdout=stdout_handle,
                    stderr=subprocess.PIPE,
                    timeout=timeout_s,
                    env=_execution_env(),
                )
        if result.returncode != 255 or attempt + 1 == attempts:
            return result
        time.sleep(1.0)
    raise AssertionError("subprocess retry loop is unreachable")


def query_remote_gpu_rows():
    script = "\n".join([
        "import csv,json,subprocess",
        "gpu=subprocess.run([",
        " 'nvidia-smi',",
        " '--query-gpu=index,uuid,memory.free,utilization.gpu',",
        " '--format=csv,noheader,nounits',",
        "],check=True,text=True,capture_output=True)",
        "process=subprocess.run([",
        " 'nvidia-smi',",
        " '--query-compute-apps=gpu_uuid,pid,process_name,used_memory',",
        " '--format=csv,noheader,nounits',",
        "],check=True,text=True,capture_output=True)",
        "by_uuid={}",
        "for row in csv.reader(gpu.stdout.splitlines()):",
        " if not row: continue",
        " index,uuid,free_mib,utilization=[value.strip() for value in row]",
        " by_uuid[uuid]={",
        "  'gpu_index':int(index),",
        "  'gpu_uuid':uuid,",
        "  'free_bytes':int(free_mib)*1024*1024,",
        "  'utilization_percent':int(utilization),",
        "  'compute_processes':[],",
        " }",
        "for row in csv.reader(process.stdout.splitlines()):",
        " if not row: continue",
        " uuid,pid,name,used_mib=[value.strip() for value in row]",
        " if uuid not in by_uuid: continue",
        " by_uuid[uuid]['compute_processes'].append({",
        "  'pid':int(pid),",
        "  'process_name':name,",
        "  'used_bytes':int(used_mib)*1024*1024,",
        " })",
        "print(json.dumps({'gpus':list(by_uuid.values())},sort_keys=True))",
    ])
    result = _run_subprocess(
        _ssh_argv([REMOTE_PYTHON, "-c", script]),
        timeout_s=30.0,
    )
    if result.returncode != 0:
        raise RuntimeError(
            result.stderr.strip()
            or "remote GPU query failed"
        )
    payload = json.loads(result.stdout)
    return payload["gpus"]


def _guard(attempt_dir, label):
    rows = query_remote_gpu_rows()
    receipt = {
        "label": label,
        "observed_at_ns": time.time_ns(),
        **evaluate_shared_gpu_guard(rows),
    }
    _atomic_write_json(
        Path(attempt_dir) / "guards" / f"{label}.json",
        receipt,
    )
    return receipt


def _wait_for_guard(
    attempt_dir,
    label,
    *,
    query_rows=query_remote_gpu_rows,
    sleep=time.sleep,
    poll_interval_s=GUARD_POLL_INTERVAL_S,
    max_wait_s=GUARD_MAX_WAIT_S,
):
    samples = []
    waited_s = 0
    while True:
        observed_at_ns = time.time_ns()
        try:
            rows = query_rows()
            evaluation = evaluate_shared_gpu_guard(rows)
        except RuntimeError as error:
            evaluation = {
                "classification": "GPU_QUERY_ERROR",
                "resource_policy": "shared-low-utilization",
                "exclusive": False,
                "minimum_gpu_free_bytes": MIN_GPU_FREE_BYTES,
                "maximum_gpu_utilization_percent": (
                    MAX_GPU_UTILIZATION_PERCENT
                ),
                "selected_gpus": [],
                "reasons": [str(error)],
            }
        samples.append({
            "observed_at_ns": observed_at_ns,
            **evaluation,
        })
        receipt = {
            "label": label,
            "observed_at_ns": observed_at_ns,
            **evaluation,
            "waited_for_resources": len(samples) > 1,
            "wait_elapsed_s": waited_s,
            "poll_interval_s": poll_interval_s,
            "max_wait_s": max_wait_s,
            "samples": samples,
        }
        _atomic_write_json(
            Path(attempt_dir) / "guards" / f"{label}.json",
            receipt,
        )
        if evaluation["classification"] == "READY":
            return receipt
        if waited_s + poll_interval_s > max_wait_s:
            receipt["reasons"] = [
                *receipt["reasons"],
                "resource monitor reached maximum wait",
            ]
            _atomic_write_json(
                Path(attempt_dir) / "guards" / f"{label}.json",
                receipt,
            )
            return receipt
        sleep(poll_interval_s)
        waited_s += poll_interval_s


def _run_remote_worker(command, *, log_path, timeout_s=3600.0):
    environment = " ".join(
        shlex.quote(f"{key}={value}")
        for key, value in sorted(command.env.items())
    )
    worker = " ".join(
        shlex.quote(str(value)) for value in command.argv
    )
    script = (
        f"cd {shlex.quote(command.cwd)} && "
        f"env {environment} {worker}"
    )
    result = _run_subprocess(
        _shell_ssh_argv(script),
        timeout_s=timeout_s,
    )
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    Path(log_path).write_text(
        result.stdout + result.stderr,
        encoding="utf-8",
    )
    return {
        "case_id": command.case_id,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "log_path": str(log_path),
    }


def _stage_attempt(
    *,
    run_tag,
    attempt_dir,
    source_bundle,
    prerequisites_tar,
):
    remote_run = f"{REMOTE_ROOT}/{run_tag}"
    remote_tar = (
        f"{REMOTE_ROOT}/{run_tag}-"
        f"{source_bundle['tar_sha256']}.tar"
    )
    upload = _run_subprocess(
        _scp_argv(
            source_bundle["tar_path"],
            f"{SSH_TARGET}:{remote_tar}",
        ),
        timeout_s=300.0,
    )
    if upload.returncode != 0:
        raise RuntimeError(
            upload.stderr.strip() or "source upload failed"
        )
    stage_command = base_runner.build_remote_stage_command(
        run_tag=run_tag,
        source_tree_sha256=source_bundle["source_tree_sha256"],
        tar_sha256=source_bundle["tar_sha256"],
    )
    stage = _run_subprocess(
        _ssh_argv(stage_command),
        timeout_s=300.0,
    )
    if stage.returncode != 0:
        raise RuntimeError(
            stage.stderr.strip() or "remote source stage failed"
        )
    remote_prerequisites_tar = f"{remote_run}/prerequisites.tar"
    prerequisite_upload = _run_subprocess(
        _scp_argv(
            prerequisites_tar,
            f"{SSH_TARGET}:{remote_prerequisites_tar}",
        ),
        timeout_s=300.0,
    )
    if prerequisite_upload.returncode != 0:
        raise RuntimeError(
            prerequisite_upload.stderr.strip()
            or "prerequisite upload failed"
        )
    extract_script = " && ".join([
        (
            "mkdir -p "
            f"{shlex.quote(remote_run + '/prerequisites')}"
        ),
        (
            f"tar -xf {shlex.quote(remote_prerequisites_tar)} "
            f"-C {shlex.quote(remote_run + '/prerequisites')}"
        ),
        "mkdir -p " + " ".join(
            shlex.quote(path)
            for path in (
                f"{remote_run}/output/cases",
                f"{remote_run}/logs",
                f"{remote_run}/nsys",
                f"{remote_run}/overhead",
            )
        ),
    ])
    extract = _run_subprocess(
        _shell_ssh_argv(extract_script),
        timeout_s=300.0,
    )
    if extract.returncode != 0:
        raise RuntimeError(
            extract.stderr.strip()
            or "prerequisite extraction failed"
        )
    receipt = {
        "remote_run": remote_run,
        "remote_source": f"{remote_run}/source",
        "remote_cases": f"{remote_run}/output/cases",
        "remote_prerequisites": (
            f"{remote_run}/prerequisites/"
            "correctness_prerequisites.json"
        ),
        "source_tree_sha256": source_bundle[
            "source_tree_sha256"
        ],
        "source_tar_sha256": source_bundle["tar_sha256"],
        "prerequisites_tar_sha256": _sha256(prerequisites_tar),
    }
    _atomic_write_json(
        Path(attempt_dir) / "stage_receipt.json",
        receipt,
    )
    return receipt


def _download_directory(remote_path, local_parent):
    local_parent = Path(local_parent)
    local_parent.mkdir(parents=True, exist_ok=True)
    result = _run_subprocess(
        _scp_argv(
            f"{SSH_TARGET}:{remote_path}",
            local_parent,
            recursive=True,
        ),
        timeout_s=600.0,
    )
    if result.returncode != 0:
        raise RuntimeError(
            result.stderr.strip() or "artifact download failed"
        )


def _aggregate_twice(cases_root, attempt_dir):
    summary = decode_profile.aggregate_decode_profiles(cases_root)
    first_path = Path(attempt_dir) / "decode_summary.json"
    second_path = Path(attempt_dir) / "decode_summary.regenerated.json"
    _atomic_write_json(first_path, summary)
    regenerated = decode_profile.aggregate_decode_profiles(cases_root)
    _atomic_write_json(second_path, regenerated)
    if first_path.read_bytes() != second_path.read_bytes():
        raise RuntimeError("decode summary is not reproducible")
    return {
        **summary,
        "summary_sha256": _sha256(first_path),
        "regenerated_summary_sha256": _sha256(second_path),
    }


def _with_output_dir(command, output_dir):
    argv = _replace_option(
        command.argv,
        "--output-dir",
        output_dir,
    )
    return WorkerCommand(
        case_id=command.case_id,
        workload=command.workload,
        policy=command.policy,
        phase=command.phase,
        repetition=command.repetition,
        output_dir=output_dir,
        cwd=command.cwd,
        env=dict(command.env),
        argv=argv,
    )


def _run_overhead_pair(
    *,
    representative_command,
    remote_run,
    attempt_dir,
):
    baseline = _with_output_dir(
        representative_command,
        f"{remote_run}/overhead/baseline",
    )
    baseline_argv = list(baseline.argv)
    baseline_argv.remove("--decode-internal-profile")
    baseline = WorkerCommand(
        **{
            **baseline.__dict__,
            "argv": baseline_argv,
        }
    )
    profiled = _with_output_dir(
        representative_command,
        f"{remote_run}/overhead/decode_internal",
    )
    rows = []
    for label, command in (
        ("baseline", baseline),
        ("decode_internal", profiled),
    ):
        guard = _wait_for_guard(
            attempt_dir,
            f"overhead-{label}",
        )
        if guard["classification"] != "READY":
            return {
                "classification": "BLOCKED_RESOURCES",
                "rows": rows,
            }
        start = time.monotonic_ns()
        result = _run_remote_worker(
            command,
            log_path=(
                Path(attempt_dir)
                / "logs"
                / f"overhead-{label}.log"
            ),
        )
        elapsed_ns = time.monotonic_ns() - start
        rows.append({
            "label": label,
            "elapsed_ns": elapsed_ns,
            "returncode": result["returncode"],
        })
        if result["returncode"] != 0:
            return {
                "classification": "FAILED",
                "rows": rows,
            }
    return {
        "classification": "COMPLETE",
        "rows": rows,
        "decode_internal_to_baseline_ratio": (
            rows[1]["elapsed_ns"] / rows[0]["elapsed_ns"]
        ),
        "boundary": (
            "fresh-process wall time includes model initialization; "
            "the ratio is a smoke estimate, not per-token profiler overhead"
        ),
    }


def _nsys_stats_report_available(*, returncode, output):
    normalized = output.lower()
    return (
        returncode == 0
        and "error:" not in normalized
        and "could not be found" not in normalized
    )


def _nsys_stats_reports():
    return (
        "cuda_gpu_kern_sum",
        "nvtx_pushpop_sum",
        "nvtx_kern_sum",
        "nccl_sum",
    )


def _run_nsys_pair(
    *,
    commands,
    repetition,
    remote_run,
    attempt_dir,
):
    nsys_check = _run_subprocess(
        _shell_ssh_argv(
            "test -x /usr/local/bin/nsys && "
            "/usr/local/bin/nsys --version"
        ),
        timeout_s=30.0,
    )
    if nsys_check.returncode != 0:
        return {
            "classification": "NSYS_UNAVAILABLE",
            "stderr": nsys_check.stderr,
        }
    rows = []
    for policy in POLICIES:
        structured = next(
            command
            for command in commands
            if (
                command.phase == "measured"
                and command.repetition == repetition
                and command.policy == policy
            )
        )
        replay = build_nsys_command(
            structured_command=structured,
            remote_nsys="/usr/local/bin/nsys",
            report_prefix=(
                f"{remote_run}/nsys/"
                f"r{repetition}-{policy}"
            ),
            repetition=repetition,
        )
        guard = _wait_for_guard(
            attempt_dir,
            f"nsys-r{repetition}-{policy}",
        )
        if guard["classification"] != "READY":
            return {
                "classification": "BLOCKED_RESOURCES",
                "rows": rows,
            }
        result = _run_remote_worker(
            replay,
            log_path=(
                Path(attempt_dir)
                / "logs"
                / f"nsys-r{repetition}-{policy}.log"
            ),
        )
        row = {
            "policy": policy,
            "case_id": replay.case_id,
            "returncode": result["returncode"],
            "reports": {},
        }
        if result["returncode"] != 0:
            rows.append(row)
            return {
                "classification": "FAILED_NSYS_REPLAY",
                "rows": rows,
            }
        report_path = (
            f"{remote_run}/nsys/r{repetition}-{policy}.nsys-rep"
        )
        for report in _nsys_stats_reports():
            stats = _run_subprocess(
                _ssh_argv([
                    "/usr/local/bin/nsys",
                    "stats",
                    "--report",
                    report,
                    "--format",
                    "csv",
                    report_path,
                ]),
                timeout_s=300.0,
            )
            stats_path = (
                Path(attempt_dir)
                / "nsys_stats"
                / f"r{repetition}-{policy}-{report}.csv"
            )
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            stats_output = stats.stdout + stats.stderr
            stats_path.write_text(stats_output, encoding="utf-8")
            row["reports"][report] = {
                "returncode": stats.returncode,
                "path": str(stats_path),
                "available": _nsys_stats_report_available(
                    returncode=stats.returncode,
                    output=stats_output,
                ),
            }
        rows.append(row)
    return {
        "classification": "COMPLETE",
        "version": nsys_check.stdout.strip(),
        "requested_trace": NSYS_REQUESTED_TRACE,
        "effective_trace": NSYS_TRACE,
        "nccl_trace_argument_supported": False,
        "nccl_evidence_boundary": (
            "Nsight Systems 2024.7.1 on the target rejects the nccl "
            "trace argument; NCCL evidence is limited to CUDA kernel "
            "names and any supported stats reports"
        ),
        "rows": rows,
    }


def _cleanup_attempt_processes(run_tag):
    script = "\n".join([
        "import json,os,signal,sys,time",
        "tag=sys.argv[1]",
        "def matches():",
        " rows=[]",
        " for entry in os.listdir('/proc'):",
        "  if not entry.isdigit(): continue",
        "  pid=int(entry)",
        "  if pid==os.getpid(): continue",
        "  try:",
        "   raw=open(f'/proc/{pid}/cmdline','rb').read()",
        "  except OSError:",
        "   continue",
        "  text=raw.replace(b'\\0',b' ').decode('utf-8','replace')",
        "  if tag in text and (",
        "   'qwen35_tp4_hybrid_prefix_benchmark_worker.py' in text",
        "   or '/usr/local/bin/nsys profile' in text",
        "  ): rows.append({'pid':pid,'cmdline':text})",
        " return rows",
        "before=matches()",
        "for row in before:",
        " try: os.kill(row['pid'],signal.SIGTERM)",
        " except ProcessLookupError: pass",
        "time.sleep(1)",
        "after=matches()",
        "print(json.dumps({'before':before,'after':after},sort_keys=True))",
    ])
    result = _run_subprocess(
        _ssh_argv([REMOTE_PYTHON, "-c", script, run_tag]),
        timeout_s=30.0,
    )
    if result.returncode != 0:
        return {
            "classification": "DIRTY",
            "error": result.stderr.strip(),
        }
    payload = json.loads(result.stdout)
    return {
        "classification": (
            "CLEAN" if not payload["after"] else "DIRTY"
        ),
        "remaining_attempt_scoped_pids": [
            row["pid"] for row in payload["after"]
        ],
        "matched_attempt_scoped_gpu_pids": [],
        "remaining_profiler_children": [],
        "terminated_attempt_scoped_pids": [
            row["pid"] for row in payload["before"]
        ],
    }


def run_real_attempt(
    *,
    run_tag,
    prerequisites_path=DEFAULT_PREREQUISITES,
    prerequisites_tar=DEFAULT_PREREQUISITES_TAR,
    local_model_manifest=DEFAULT_LOCAL_MODEL_MANIFEST,
    remote_model_dir=DEFAULT_REMOTE_MODEL_DIR,
    remote_model_manifest=DEFAULT_REMOTE_MODEL_MANIFEST,
    output_root=LOCAL_OUTPUT_ROOT,
):
    run_tag = base_runner.safe_run_tag(run_tag)
    attempt_dir = Path(output_root) / run_tag
    if attempt_dir.exists():
        raise ValueError("fresh run tag is required")
    attempt_dir.mkdir(parents=True)
    extracted_prerequisites = _extract_prerequisites_bundle(
        prerequisites_tar,
        attempt_dir / "inputs" / "prerequisites_bundle",
    )
    if Path(prerequisites_path).read_bytes() != (
        extracted_prerequisites.read_bytes()
    ):
        raise ValueError(
            "prerequisite JSON differs from prerequisite archive"
        )
    status = contract.validate_prerequisites(
        extracted_prerequisites
    )
    if not status.authorized:
        result = {
            "classification": "BLOCKED_CORRECTNESS",
            "reasons": list(status.reasons),
            "preserve_attempt": True,
        }
        _atomic_write_json(
            attempt_dir / "attempt_receipt.json",
            result,
        )
        return result
    model_manifest_sha256 = _sha256(local_model_manifest)
    if model_manifest_sha256 != contract.MODEL_MANIFEST_SHA256:
        raise ValueError("local model manifest SHA mismatch")
    entry_guard = _wait_for_guard(attempt_dir, "entry")
    if entry_guard["classification"] != "READY":
        result = {
            "classification": "BLOCKED_RESOURCES",
            "entry_guard": entry_guard,
            "preserve_attempt": True,
        }
        _atomic_write_json(
            attempt_dir / "attempt_receipt.json",
            result,
        )
        return result
    source_bundle = base_runner.prepare_benchmark_source_bundle(
        repo_root=Path(__file__).resolve().parents[1],
        output_dir=attempt_dir / "inputs",
    )
    stage = _stage_attempt(
        run_tag=run_tag,
        attempt_dir=attempt_dir,
        source_bundle=source_bundle,
        prerequisites_tar=Path(prerequisites_tar),
    )
    ports = base_runner.allocate_remote_port_pairs(
        12,
        command_runner=base_runner._load_tool_module(
            "decode_profile_subprocess_adapter",
            "qwen35_tp4_engine_remote_subprocess_adapter.py",
        ).run_command,
        execution_env={"KRB5CCNAME": KRB5CCNAME},
    )
    authorization = WorkerAuthorization(
        prerequisites_sha256=_sha256(extracted_prerequisites),
        source_tree_sha256=source_bundle["source_tree_sha256"],
        model_manifest_sha256=model_manifest_sha256,
        workload_manifest_sha256=(
            contract.canonical_json_file_sha256(
                contract.workload_manifest_payload()
            )
        ),
        gpu_indices=GPU_INDICES,
    )
    runtime_artifacts = WorkerRuntimeArtifacts(
        model_dir=remote_model_dir,
        model_manifest_path=remote_model_manifest,
        correctness_prerequisites_path=stage[
            "remote_prerequisites"
        ],
        workload_manifest_path=(
            f"{stage['remote_run']}/workload_manifest.json"
        ),
    )
    commands = build_structured_commands(
        remote_source=stage["remote_source"],
        remote_cases=stage["remote_cases"],
        ports=ports,
        authorization=authorization,
        runtime_artifacts=runtime_artifacts,
    )
    command_receipt = [
        {
            "case_id": command.case_id,
            "policy": command.policy,
            "phase": command.phase,
            "repetition": command.repetition,
            "output_dir": command.output_dir,
            "env": command.env,
            "argv": command.argv,
        }
        for command in commands
    ]
    _atomic_write_json(
        attempt_dir / "commands.json",
        command_receipt,
    )
    worker_results = []
    final = None
    try:
        for command in commands:
            guard = _wait_for_guard(
                attempt_dir,
                f"worker-{command.case_id}",
            )
            if guard["classification"] != "READY":
                final = {
                    "classification": "BLOCKED_WORKER_ENTRY",
                    "worker_results": worker_results,
                    "preserve_attempt": True,
                }
                break
            result = _run_remote_worker(
                command,
                log_path=(
                    attempt_dir
                    / "logs"
                    / f"{command.case_id}.log"
                ),
            )
            worker_results.append(result)
            _atomic_write_json(
                attempt_dir / "worker_results.json",
                worker_results,
            )
            if result["returncode"] != 0:
                final = {
                    "classification": "FAILED_STRUCTURED_WORKER",
                    "worker_results": worker_results,
                    "preserve_attempt": True,
                }
                break
        if final is None:
            _download_directory(
                stage["remote_cases"],
                attempt_dir / "download",
            )
            cases_root = (
                attempt_dir / "download" / "cases"
            )
            structured = _aggregate_twice(
                cases_root,
                attempt_dir,
            )
            representative = structured[
                "representative_repetition"
            ]
            representative_command = next(
                command
                for command in commands
                if (
                    command.phase == "measured"
                    and command.repetition == representative
                    and command.policy == "recompute"
                )
            )
            overhead = _run_overhead_pair(
                representative_command=representative_command,
                remote_run=stage["remote_run"],
                attempt_dir=attempt_dir,
            )
            _atomic_write_json(
                attempt_dir / "overhead.json",
                overhead,
            )
            nsys = _run_nsys_pair(
                commands=commands,
                repetition=representative,
                remote_run=stage["remote_run"],
                attempt_dir=attempt_dir,
            )
            _atomic_write_json(
                attempt_dir / "nsys_receipt.json",
                nsys,
            )
            _download_directory(
                f"{stage['remote_run']}/nsys",
                attempt_dir / "download",
            )
            classification = (
                "COMPLETE_WITHOUT_NSYS"
                if nsys["classification"] == "NSYS_UNAVAILABLE"
                else "COMPLETE"
            )
            if overhead["classification"] != "COMPLETE":
                classification = "INCOMPLETE_OVERHEAD"
            if nsys["classification"] not in {
                "COMPLETE",
                "NSYS_UNAVAILABLE",
            }:
                classification = "INCOMPLETE_NSYS"
            final = {
                "classification": classification,
                "run_tag": run_tag,
                "resource_policy": "shared-low-utilization",
                "exclusive": False,
                "worker_count": len(worker_results),
                "structured": structured,
                "overhead": overhead,
                "nsys": nsys,
                "source_tree_sha256": source_bundle[
                    "source_tree_sha256"
                ],
                "preserve_attempt": True,
            }
    finally:
        cleanup = _cleanup_attempt_processes(run_tag)
    final["cleanup"] = cleanup
    if cleanup["classification"] != "CLEAN":
        final["classification"] = "FAILED_CLEANUP"
    _atomic_write_json(
        attempt_dir / "attempt_receipt.json",
        final,
    )
    return final


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", required=True)
    parser.add_argument(
        "--prerequisites",
        type=Path,
        default=DEFAULT_PREREQUISITES,
    )
    parser.add_argument(
        "--prerequisites-tar",
        type=Path,
        default=DEFAULT_PREREQUISITES_TAR,
    )
    parser.add_argument(
        "--local-model-manifest",
        type=Path,
        default=DEFAULT_LOCAL_MODEL_MANIFEST,
    )
    parser.add_argument(
        "--remote-model-dir",
        default=DEFAULT_REMOTE_MODEL_DIR,
    )
    parser.add_argument(
        "--remote-model-manifest",
        default=DEFAULT_REMOTE_MODEL_MANIFEST,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=LOCAL_OUTPUT_ROOT,
    )
    args = parser.parse_args(argv)
    result = run_real_attempt(
        run_tag=args.run_tag,
        prerequisites_path=args.prerequisites,
        prerequisites_tar=args.prerequisites_tar,
        local_model_manifest=args.local_model_manifest,
        remote_model_dir=args.remote_model_dir,
        remote_model_manifest=args.remote_model_manifest,
        output_root=args.output_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["classification"] in {
        "COMPLETE",
        "COMPLETE_WITHOUT_NSYS",
    } else 2


if __name__ == "__main__":
    raise SystemExit(main())
