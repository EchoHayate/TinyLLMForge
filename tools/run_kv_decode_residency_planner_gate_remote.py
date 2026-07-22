#!/usr/bin/env python3
"""Run the source-bound KV decode residency planner gate on remote GPU 0."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shlex
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
from dataclasses import asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
SSH_TARGET = "sitian@10.232.195.203"
SSH_CONTROL_PATH = "/tmp/ssh-sitian-10.232.195.203"
REMOTE_PYTHON = (
    "/data00/home/sitian/sitian-workspace01/tllm/env/bin/python"
)
REMOTE_MODEL = (
    "/data00/home/sitian/sitian-workspace01/.ms_cache/"
    "Qwen/Qwen3-0___6B"
)
REMOTE_RUN_ROOT = (
    "/data00/home/sitian/sitian-workspace01/tllm/"
    "kv-decode-residency-planner-runs"
)
CUDA_VISIBLE_DEVICES = "0"
OUTPUT_ROOT = ROOT / "experiments" / "kv_offload"
LOCAL_VERIFIER_PYTHON = Path(
    "/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python"
)
BASELINE_COMMIT = "94056ba"
BASELINE_OVERRIDE = "tinyvllm/layers/attention.py"
MODES = (
    "preflight",
    "smoke",
    "canonical",
    "download-only",
    "verify-only",
)
OWNED_SOURCE_ROOTS = (
    "tinyvllm",
    "tools/draft_model_schema.py",
    "tools/profile_ngram_commit.py",
    "tools/kv_decode_residency_planner_contract.py",
    "tools/verify_kv_decode_residency_planner_gate.py",
    "tools/run_kv_decode_residency_planner_gate_remote.py",
    "tools/test_kv_decode_residency_planner_gate.py",
    "tools/test_blockwise_attention_planning.py",
    "tools/test_kv_offload.py",
    "tools/test_model_runner_spec_verify.py",
    "tools/smoke_blockwise_prefill_remote.sh",
)
IGNORED_DIRTY_PREFIXES = ("experiments/",)
SSH_OPTIONS = (
    "-o",
    "ControlMaster=auto",
    "-o",
    f"ControlPath={SSH_CONTROL_PATH}",
    "-o",
    "ControlPersist=600",
)


def _load_tool(name: str, filename: str):
    path = TOOLS / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load tool: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


contract = _load_tool(
    "kv_decode_residency_planner_contract_for_runner",
    "kv_decode_residency_planner_contract.py",
)


def _canonical_bytes(value) -> bytes:
    return contract.canonical_json_bytes(value)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_bytes(_canonical_bytes(value) + b"\n")
    partial.replace(path)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_bytes(
        b"".join(_canonical_bytes(row) + b"\n" for row in rows)
    )
    partial.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_run_tag(run_tag: str) -> str:
    if (
        not run_tag
        or any(
            character
            not in (
                "abcdefghijklmnopqrstuvwxyz"
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                "0123456789-_"
            )
            for character in run_tag
        )
    ):
        raise ValueError("run tag contains unsupported characters")
    return run_tag


def _default_run_tag(mode: str) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    entropy = hashlib.sha256(os.urandom(16)).hexdigest()[:8]
    return f"qwen3-06b-kv-residency-{mode}-{stamp}-{entropy}"


def _remote_run_dir(run_tag: str) -> str:
    return f"{REMOTE_RUN_ROOT}/{_safe_run_tag(run_tag)}"


def _ssh_argv(remote_argv: list[str]) -> list[str]:
    remote_command = " ".join(shlex.quote(value) for value in remote_argv)
    return [
        "ssh",
        *SSH_OPTIONS,
        SSH_TARGET,
        "bash",
        "-lc",
        shlex.quote(remote_command),
    ]


def _run_remote(
    remote_argv: list[str],
    *,
    input_bytes: bytes | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    result = subprocess.run(
        _ssh_argv(remote_argv),
        input=input_bytes,
        capture_output=True,
        check=False,
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            result.stderr.decode("utf-8", errors="replace").strip()
            or result.stdout.decode("utf-8", errors="replace").strip()
            or f"remote command exited {result.returncode}"
        )
    return result


def allocate_port_pair() -> tuple[int, int]:
    handles = []
    ports = []
    try:
        while len(ports) < 2:
            handle = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            handle.bind(("127.0.0.1", 0))
            handles.append(handle)
            port = int(handle.getsockname()[1])
            if port not in ports:
                ports.append(port)
    finally:
        for handle in handles:
            handle.close()
    return ports[0], ports[1]


def allocate_unique_port_pairs(
    count: int,
    *,
    allocator=allocate_port_pair,
) -> list[tuple[int, int]]:
    pairs = []
    used = set()
    for _ in range(int(count)):
        dist_port, master_port = allocator()
        if dist_port == master_port:
            raise ValueError("worker ports must be distinct")
        if dist_port in used or master_port in used:
            raise ValueError("duplicate port across worker pairs")
        used.update((dist_port, master_port))
        pairs.append((dist_port, master_port))
    return pairs


def is_retryable_port_collision(returncode: int, stderr: str) -> bool:
    return int(returncode) != 0 and "EADDRINUSE" in str(stderr)


def _workload_prompts(workload: str) -> tuple[list[str], int, int]:
    sentence = (
        "TinyLLMForge residency-aware decode planner gate uses repeated "
        "stable tokens and deterministic greedy decoding. "
    )
    if workload == "single_long_context":
        return [sentence * 64], 1, 4096
    if workload == "multi_prompt_thrash":
        return [
            f"Prompt {index}: " + sentence * 40
            for index in range(2)
        ], 2, 2048
    raise ValueError(f"unknown workload: {workload}")


def build_worker_command(
    *,
    case,
    remote_source: str,
    case_output_dir: str,
    source_sha256: str,
    dist_port: int,
    master_port: int,
) -> dict:
    if dist_port == master_port:
        raise ValueError("worker ports must be distinct")
    prompts, max_num_seqs, max_model_len = _workload_prompts(
        case.workload
    )
    argv = [
        REMOTE_PYTHON,
        "tools/profile_ngram_commit.py",
        "--mode",
        "baseline-only",
        "--model",
        REMOTE_MODEL,
    ]
    for prompt in prompts:
        argv.extend(["--prompt", prompt])
    argv.extend([
        "--max-output-len",
        "16",
        "--temperature",
        "0.0",
        "--gpu-memory-utilization",
        "0.7",
        "--max-num-prefill-tokens-per-step",
        "256",
        "--max-num-seqs",
        str(max_num_seqs),
        "--max-model-len",
        str(max_model_len),
        "--kv-offload-mvp0",
        "--kv-offload-logical-blocks",
        "8",
        "--kv-offload-blockwise-prefill",
        "--kv-offload-blockwise-decode",
        "--kv-offload-gpu-blocks",
        str(case.gpu_blocks),
        "--kv-offload-blockwise-blocks",
        str(case.blockwise_blocks),
        "--out-json",
        f"{case_output_dir}/profile.json",
    ])
    if case.phase == "correctness":
        argv.extend([
            "--record-decode-logits",
            "--decode-logits-out",
            f"{case_output_dir}/decode_logits.pt",
        ])
    return {
        "cwd": remote_source,
        "env": {
            "CUDA_VISIBLE_DEVICES": CUDA_VISIBLE_DEVICES,
            "TINYVLLM_DIST_PORT": str(dist_port),
            "MASTER_PORT": str(master_port),
            "PYTHONPATH": remote_source,
            "PYTHONDONTWRITEBYTECODE": "1",
            "TINYVLLM_SOURCE_SHA256": source_sha256,
        },
        "argv": argv,
    }


def _tracked_owned_files() -> tuple[str, ...]:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--", *OWNED_SOURCE_ROOTS],
        cwd=ROOT,
        capture_output=True,
        check=True,
    )
    files = tuple(
        path.decode("utf-8")
        for path in result.stdout.split(b"\0")
        if path
    )
    if not files:
        raise RuntimeError("owned source manifest is empty")
    return tuple(sorted(files))


def _assert_clean_tracked_tree() -> None:
    result = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=no"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    dirty = []
    for line in result.stdout.splitlines():
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        if not any(path.startswith(prefix) for prefix in IGNORED_DIRTY_PREFIXES):
            dirty.append(line)
    if dirty:
        raise RuntimeError(
            "tracked tree must be clean before remote gate: "
            + "; ".join(dirty)
        )


def _copy_manifest_files(
    destination: Path,
    files: tuple[str, ...],
) -> None:
    for relative_path in files:
        source = ROOT / relative_path
        target = destination / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())


def _tree_manifest(source: Path, files: tuple[str, ...]) -> dict:
    entries = [
        {
            "path": relative_path,
            "sha256": _sha256_file(source / relative_path),
            "size": (source / relative_path).stat().st_size,
        }
        for relative_path in files
    ]
    return {
        "files": entries,
        "tree_sha256": contract.canonical_json_sha256(entries),
    }


def _git_output(*args: str) -> bytes:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            result.stderr.decode("utf-8", errors="replace").strip()
        )
    return result.stdout


def create_policy_snapshots(run_dir: Path) -> dict:
    _assert_clean_tracked_tree()
    files = _tracked_owned_files()
    staging = run_dir / "staging"
    baseline = staging / "baseline"
    candidate = staging / "candidate"
    _copy_manifest_files(baseline, files)
    _copy_manifest_files(candidate, files)
    baseline_attention = _git_output(
        "show",
        f"{BASELINE_COMMIT}:{BASELINE_OVERRIDE}",
    )
    (baseline / BASELINE_OVERRIDE).write_bytes(baseline_attention)
    differing = [
        relative_path
        for relative_path in files
        if (baseline / relative_path).read_bytes()
        != (candidate / relative_path).read_bytes()
    ]
    if differing != [BASELINE_OVERRIDE]:
        raise RuntimeError(
            "policy snapshots must differ only in "
            f"{BASELINE_OVERRIDE}: {differing}"
        )
    head = _git_output("rev-parse", "HEAD").decode().strip()
    policies = {
        "baseline": {
            "origin_commit": BASELINE_COMMIT,
            "runtime_base_commit": head,
            **_tree_manifest(baseline, files),
        },
        "candidate": {
            "origin_commit": head,
            "runtime_base_commit": head,
            **_tree_manifest(candidate, files),
        },
    }
    manifest = {
        "owned_source_roots": list(OWNED_SOURCE_ROOTS),
        "owned_files": list(files),
        "only_policy_difference": BASELINE_OVERRIDE,
        "policies": policies,
    }
    _write_json(staging / "source_manifest.json", manifest)
    return {
        "root": staging,
        "baseline": baseline,
        "candidate": candidate,
        "manifest": manifest,
    }


def _stream_directory(local_dir: Path, remote_dir: str) -> None:
    _run_remote(["mkdir", "-p", remote_dir])
    producer = subprocess.Popen(
        ["tar", "-C", str(local_dir), "-czf", "-", "."],
        stdout=subprocess.PIPE,
    )
    if producer.stdout is None:
        raise RuntimeError("could not open tar stream")
    consumer = subprocess.run(
        _ssh_argv(["tar", "-xzf", "-", "-C", remote_dir]),
        stdin=producer.stdout,
        capture_output=True,
        check=False,
    )
    producer.stdout.close()
    producer_returncode = producer.wait()
    if producer_returncode != 0 or consumer.returncode != 0:
        raise RuntimeError(
            consumer.stderr.decode("utf-8", errors="replace").strip()
            or "source snapshot stream failed"
        )


def _download_remote_tree(remote_dir: str, local_dir: Path) -> None:
    if local_dir.exists():
        raise ValueError(f"local run directory exists: {local_dir}")
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    archive = local_dir.with_suffix(".tar.partial")
    with archive.open("wb") as handle:
        result = subprocess.run(
            _ssh_argv([
                "tar",
                "-C",
                str(Path(remote_dir).parent),
                "-cf",
                "-",
                Path(remote_dir).name,
            ]),
            stdout=handle,
            stderr=subprocess.PIPE,
            check=False,
        )
    if result.returncode != 0:
        archive.unlink(missing_ok=True)
        raise RuntimeError(
            result.stderr.decode("utf-8", errors="replace")
        )
    partial = local_dir.with_name(local_dir.name + ".partial")
    partial.mkdir()
    with tarfile.open(archive, "r:") as tar:
        for member in tar.getmembers():
            member_path = Path(member.name)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError("unsafe remote artifact path")
        tar.extractall(partial)
    extracted = partial / Path(remote_dir).name
    extracted.replace(local_dir)
    partial.rmdir()
    archive.unlink()


def _remote_allocate_port_pair() -> tuple[int, int]:
    script = (
        "import json,socket\n"
        "handles=[]\n"
        "ports=[]\n"
        "try:\n"
        "  while len(ports)<2:\n"
        "    s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)\n"
        "    s.bind(('127.0.0.1',0)); handles.append(s)\n"
        "    p=int(s.getsockname()[1])\n"
        "    if p not in ports: ports.append(p)\n"
        "  print(json.dumps(ports))\n"
        "finally:\n"
        "  [s.close() for s in handles]\n"
    )
    result = _run_remote([REMOTE_PYTHON, "-c", script])
    ports = json.loads(result.stdout.decode("utf-8"))
    return int(ports[0]), int(ports[1])


def _render_command(command: dict) -> str:
    env = [
        f"{key}={value}"
        for key, value in command["env"].items()
    ]
    argv = " ".join(shlex.quote(value) for value in command["argv"])
    return (
        f"cd {shlex.quote(command['cwd'])} && "
        + "exec env "
        + " ".join(shlex.quote(value) for value in env)
        + " "
        + argv
    )


def _run_remote_worker(
    case_output_dir: str,
    command: dict,
) -> subprocess.CompletedProcess:
    metadata = {
        "cwd": command["cwd"],
        "env": command["env"],
        "argv": command["argv"],
    }
    metadata_bytes = _canonical_bytes(metadata) + b"\n"
    _run_remote(["mkdir", "-p", case_output_dir])
    _run_remote(
        [
            "bash",
            "-lc",
            "cat > "
            + shlex.quote(f"{case_output_dir}/command.json"),
        ],
        input_bytes=metadata_bytes,
    )
    shell = (
        f"echo $$ > {shlex.quote(f'{case_output_dir}/worker_pid.txt')}; "
        + _render_command(command)
        + f" > {shlex.quote(f'{case_output_dir}/stdout.txt')}"
        + f" 2> {shlex.quote(f'{case_output_dir}/stderr.txt')}"
    )
    return _run_remote(["bash", "-lc", shell], check=False)


def _remote_read_bytes(path: str) -> bytes:
    result = _run_remote(["cat", path])
    return result.stdout


def _run_case_with_retries(
    *,
    case,
    remote_source: str,
    case_output_dir: str,
    source_sha256: str,
    used_ports: set[int],
) -> tuple[dict, dict]:
    for attempt in range(1, 4):
        dist_port, master_port = _remote_allocate_port_pair()
        if (
            dist_port == master_port
            or dist_port in used_ports
            or master_port in used_ports
        ):
            continue
        used_ports.update((dist_port, master_port))
        command = build_worker_command(
            case=case,
            remote_source=remote_source,
            case_output_dir=case_output_dir,
            source_sha256=source_sha256,
            dist_port=dist_port,
            master_port=master_port,
        )
        result = _run_remote_worker(case_output_dir, command)
        stderr = _remote_read_bytes(
            f"{case_output_dir}/stderr.txt"
        ).decode("utf-8", errors="replace")
        if result.returncode == 0:
            return command, {
                "tinyvllm_dist_port": dist_port,
                "master_port": master_port,
                "attempts": attempt,
                "stderr": stderr,
            }
        if not is_retryable_port_collision(
            result.returncode,
            stderr,
        ):
            raise RuntimeError(stderr or "remote worker failed")
    raise RuntimeError("remote worker exhausted EADDRINUSE retries")


def _execution_matrix(mode: str):
    matrix = list(contract.build_case_matrix())
    if mode == "smoke":
        selected_pairs = {
            (
                "single_long_context",
                2,
                1,
                "correctness",
                0,
            ),
            (
                "multi_prompt_thrash",
                2,
                1,
                "measured",
                0,
            ),
        }
        matrix = [
            case
            for case in matrix
            if (
                case.workload,
                case.gpu_blocks,
                case.blockwise_blocks,
                case.phase,
                case.repetition,
            )
            in selected_pairs
        ]
    grouped = {}
    for case in matrix:
        grouped.setdefault(case.pair_id, []).append(case)
    ordered = []
    for pair_id in dict.fromkeys(case.pair_id for case in matrix):
        pair = grouped[pair_id]
        baseline = next(case for case in pair if case.policy == "baseline")
        candidate = next(case for case in pair if case.policy == "candidate")
        ordered.extend(
            (baseline, candidate)
            if baseline.repetition % 2 == 0
            else (candidate, baseline)
        )
    return tuple(ordered)


def _prompt_sha256(case) -> str:
    prompts, _, _ = _workload_prompts(case.workload)
    return contract.canonical_json_sha256(prompts)


def build_run_manifest(
    *,
    mode: str,
    source_sha256_by_policy: dict,
) -> dict:
    return {
        "schema_version": contract.SCHEMA_VERSION,
        "complete": True,
        "mode": mode,
        "expected_case_ids": [
            case.case_id for case in contract.build_case_matrix()
        ],
        "source_sha256_by_policy": dict(source_sha256_by_policy),
        "prompt_sha256_by_workload": {
            workload: _prompt_sha256(
                contract.GateCase(
                    workload=workload,
                    policy="baseline",
                    gpu_blocks=2,
                    blockwise_blocks=1,
                    repetition=0,
                    phase="measured",
                    warmup=False,
                )
            )
            for workload in contract.WORKLOADS
        },
        "cuda_visible_devices": CUDA_VISIBLE_DEVICES,
        "model_path": REMOTE_MODEL,
        "python_path": REMOTE_PYTHON,
    }


def _profile_to_row(
    *,
    case,
    profile: dict,
    source_sha256: str,
    ports: dict,
    worker_pid: int,
) -> dict:
    per_prompt = sorted(
        profile["per_prompt"],
        key=lambda item: int(item["prompt_index"]),
    )
    decoded_token_ids = [
        int(token_id)
        for item in per_prompt
        for token_id in item["token_ids"]
    ]
    summary = profile["summary"]
    kv_offload = profile.get("kv_offload") or {}
    logits_path = None
    if case.phase == "correctness":
        logits_path = f"cases/{case.case_id}/decode_logits.pt"
    return {
        "row_id": f"{case.case_id}:worker",
        "case_id": case.case_id,
        "policy": case.policy,
        "workload": case.workload,
        "gpu_blocks": case.gpu_blocks,
        "blockwise_blocks": case.blockwise_blocks,
        "repetition": case.repetition,
        "phase": case.phase,
        "warmup": case.warmup,
        "source_sha256": source_sha256,
        "worker_pid": int(worker_pid),
        "tinyvllm_dist_port": ports["tinyvllm_dist_port"],
        "master_port": ports["master_port"],
        "cuda_visible_devices": CUDA_VISIBLE_DEVICES,
        "model_path": REMOTE_MODEL,
        "python_path": REMOTE_PYTHON,
        "prompt_sha256": _prompt_sha256(case),
        "decoded_token_ids": decoded_token_ids,
        "decode_logits_path": logits_path,
        "decode_logits_sha256": profile.get(
            "decode_logits_sha256"
        ),
        "decode_logits_shape": profile.get(
            "decode_logits_shape"
        ),
        "decode_step_ms": summary["decode_step_ms"],
        "peak_cuda_allocated_bytes": summary[
            "peak_cuda_allocated_bytes"
        ],
        "peak_cuda_reserved_bytes": summary[
            "peak_cuda_reserved_bytes"
        ],
        "peak_resident_blocks": summary["peak_resident_blocks"],
        "kv_offload": {
            field: int(kv_offload.get(field, 0))
            for field in contract.KV_COUNTER_FIELDS
        },
        "planner": {
            field: int(profile["planner"].get(field, 0))
            for field in contract.PLANNER_COUNTER_FIELDS
        },
        "complete": summary.get("gate_pass") is True,
    }


def _write_remote_json(path: str, value) -> None:
    _run_remote(
        ["bash", "-lc", "cat > " + shlex.quote(path)],
        input_bytes=_canonical_bytes(value) + b"\n",
    )


def _write_remote_jsonl(path: str, rows: list[dict]) -> None:
    _run_remote(
        ["bash", "-lc", "cat > " + shlex.quote(path)],
        input_bytes=b"".join(
            _canonical_bytes(row) + b"\n" for row in rows
        ),
    )


def build_raw_summary(rows: list[dict]) -> dict:
    return {
        "schema_version": contract.SCHEMA_VERSION,
        "case_count": len(rows),
        "complete_count": sum(
            row.get("complete") is True
            for row in rows
        ),
        "row_ids_sha256": contract.canonical_json_sha256(
            [row.get("row_id") for row in rows]
        ),
        "case_ids_sha256": contract.canonical_json_sha256(
            [row.get("case_id") for row in rows]
        ),
    }


def _remote_sha256(path: str) -> str:
    result = _run_remote(["sha256sum", path])
    return result.stdout.decode("utf-8").split()[0]


def _remote_environment() -> dict:
    script = (
        "import json,os,platform,torch\n"
        "print(json.dumps({"
        "'schema_version':1,"
        "'host':platform.node(),"
        f"'python_path':{REMOTE_PYTHON!r},"
        "'python_realpath':os.path.realpath(__import__('sys').executable),"
        "'python_version':platform.python_version(),"
        "'torch_version':torch.__version__,"
        "'cuda_version':str(torch.version.cuda),"
        "'gpu_name':torch.cuda.get_device_name(0),"
        "'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),"
        f"'model_path':{REMOTE_MODEL!r}"
        "},sort_keys=True))\n"
    )
    result = _run_remote([
        "env",
        f"CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES}",
        REMOTE_PYTHON,
        "-c",
        script,
    ])
    return json.loads(result.stdout.decode("utf-8"))


def _run_preflight(remote_root: str) -> None:
    candidate = f"{remote_root}/source/candidate"
    commands = (
        [REMOTE_PYTHON, "tools/test_model_runner_spec_verify.py"],
        [REMOTE_PYTHON, "tools/test_blockwise_attention_planning.py"],
        [REMOTE_PYTHON, "tools/test_kv_offload.py"],
        [REMOTE_PYTHON, "tools/test_kv_decode_residency_planner_gate.py"],
        [
            REMOTE_PYTHON,
            "-m",
            "py_compile",
            "tinyvllm/engine/model_runner.py",
            "tools/profile_ngram_commit.py",
            "tools/run_kv_decode_residency_planner_gate_remote.py",
        ],
    )
    for command in commands:
        _run_remote([
            "env",
            f"CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES}",
            f"PYTHONPATH={candidate}:{candidate}/tools",
            "PYTHONDONTWRITEBYTECODE=1",
            "bash",
            "-lc",
            "cd "
            + shlex.quote(candidate)
            + " && "
            + " ".join(shlex.quote(value) for value in command),
        ])


def _verify_local(run_dir: Path) -> dict:
    if not LOCAL_VERIFIER_PYTHON.is_file():
        raise RuntimeError(
            f"local verifier Python missing: {LOCAL_VERIFIER_PYTHON}"
        )
    result = subprocess.run(
        [
            os.fspath(LOCAL_VERIFIER_PYTHON),
            os.fspath(
                TOOLS / "verify_kv_decode_residency_planner_gate.py"
            ),
            "--run-dir",
            os.fspath(run_dir),
            "--write-report",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return json.loads(result.stdout)


def run_remote_gate(mode: str, run_tag: str) -> dict:
    local_staging = OUTPUT_ROOT / f".{run_tag}.staging"
    if local_staging.exists():
        raise ValueError(f"local staging exists: {local_staging}")
    local_staging.mkdir(parents=True)
    snapshots = create_policy_snapshots(local_staging)
    remote_root = _remote_run_dir(run_tag)
    _run_remote([
        "mkdir",
        "-p",
        f"{remote_root}/source/baseline",
        f"{remote_root}/source/candidate",
        f"{remote_root}/cases",
        f"{remote_root}/logs",
    ])
    _stream_directory(
        snapshots["baseline"],
        f"{remote_root}/source/baseline",
    )
    _stream_directory(
        snapshots["candidate"],
        f"{remote_root}/source/candidate",
    )
    environment = _remote_environment()
    _run_preflight(remote_root)
    if mode == "preflight":
        return {
            "classification": "PREFLIGHT_PASS",
            "remote_run_dir": remote_root,
            "source_manifest": snapshots["manifest"],
            "environment": environment,
        }

    rows = []
    used_ports = set()
    worker_logs = []
    commands = []
    for case in _execution_matrix(mode):
        source = f"{remote_root}/source/{case.policy}"
        output = f"{remote_root}/cases/{case.case_id}"
        source_sha256 = snapshots["manifest"]["policies"][
            case.policy
        ]["tree_sha256"]
        command, ports = _run_case_with_retries(
            case=case,
            remote_source=source,
            case_output_dir=output,
            source_sha256=source_sha256,
            used_ports=used_ports,
        )
        profile = json.loads(
            _remote_read_bytes(f"{output}/profile.json")
        )
        worker_pid = int(
            _remote_read_bytes(
                f"{output}/worker_pid.txt"
            ).decode().strip()
        )
        rows.append(_profile_to_row(
            case=case,
            profile=profile,
            source_sha256=source_sha256,
            ports=ports,
            worker_pid=worker_pid,
        ))
        commands.append({
            "case": asdict(case),
            "command": command,
            "ports": ports,
        })
        for name in ("stdout.txt", "stderr.txt", "command.json"):
            worker_logs.append(
                f"cases/{case.case_id}/{name}"
            )

    source_sha256_by_policy = {
        policy: snapshots["manifest"]["policies"][policy][
            "tree_sha256"
        ]
        for policy in contract.POLICIES
    }
    manifest = build_run_manifest(
        mode=mode,
        source_sha256_by_policy=source_sha256_by_policy,
    )
    if mode == "smoke":
        manifest["expected_case_ids"] = [
            case.case_id for case in _execution_matrix(mode)
        ]
    source_manifest = {
        "schema_version": contract.SCHEMA_VERSION,
        **snapshots["manifest"],
        "source_sha256_by_policy": {
            policy: snapshots["manifest"]["policies"][policy][
                "tree_sha256"
            ]
            for policy in contract.POLICIES
        },
    }
    _write_remote_json(f"{remote_root}/manifest.json", manifest)
    _write_remote_json(
        f"{remote_root}/environment.json",
        environment,
    )
    _write_remote_json(
        f"{remote_root}/source_manifest.json",
        source_manifest,
    )
    _write_remote_json(
        f"{remote_root}/commands.json",
        commands,
    )
    _write_remote_jsonl(
        f"{remote_root}/case_rows.jsonl",
        rows,
    )
    _write_remote_json(
        f"{remote_root}/summary.json",
        build_raw_summary(rows),
    )
    _write_remote_json(
        f"{remote_root}/worker_logs_manifest.json",
        {
            "schema_version": contract.SCHEMA_VERSION,
            "logs": [
                {
                    "path": path,
                    "sha256": _remote_sha256(
                        f"{remote_root}/{path}"
                    ),
                }
                for path in worker_logs
            ],
        },
    )

    local_dir = OUTPUT_ROOT / run_tag
    _download_remote_tree(remote_root, local_dir)
    if mode == "smoke":
        report = {
            "schema_version": contract.SCHEMA_VERSION,
            "classification": "NON_AUTHORITATIVE_SMOKE",
            "verified_case_count": len(rows),
            "run_dir": os.fspath(local_dir),
        }
    else:
        report = _verify_local(local_dir)
    local_staging_archive = local_staging.with_suffix(".source.tar")
    with tarfile.open(local_staging_archive, "w") as tar:
        tar.add(local_staging, arcname="source")
    return report


def _parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", nargs="?", choices=MODES, default="canonical")
    parser.add_argument("--run-tag")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args(argv)
    if args.mode in {"download-only", "verify-only"} and not args.run_tag:
        parser.error(f"{args.mode} requires --run-tag")
    return args


def main(argv: list[str] | None = None) -> int:
    global OUTPUT_ROOT
    args = _parse_args(argv)
    OUTPUT_ROOT = args.output_root
    run_tag = args.run_tag or _default_run_tag(args.mode)
    if args.mode == "verify-only":
        report = _verify_local(OUTPUT_ROOT / run_tag)
    elif args.mode == "download-only":
        local_dir = OUTPUT_ROOT / run_tag
        _download_remote_tree(_remote_run_dir(run_tag), local_dir)
        report = _verify_local(local_dir)
    else:
        report = run_remote_gate(args.mode, run_tag)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
