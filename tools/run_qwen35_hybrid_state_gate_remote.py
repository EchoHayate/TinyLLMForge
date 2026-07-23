"""Non-destructive remote runner for the Qwen3.5 hybrid-state gate."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import secrets
import shlex
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath


REMOTE_TARGET = "sitian@10.232.195.203"
SSH_CONTROL_PATH = "/tmp/ssh-sitian-10.232.195.203"
REMOTE_PYTHON = (
    "/data00/home/sitian/sitian-workspace01/tllm/env/bin/python"
)
REMOTE_RUN_ROOT = (
    "/data00/home/sitian/sitian-workspace01/tllm/"
    "qwen35-hybrid-state-runs"
)
MODEL_REPOSITORY = "Qwen/Qwen3.5-2B"
LOCAL_RUN_ROOT = Path("experiments/qwen35_hybrid_state")
VERIFIER_OUTPUT_FILES = {
    "independent_verification.json",
    "report.md",
    "local_verifier_process.json",
    "smoke_evidence.json",
    "stdout/local_verifier.log",
    "stderr/local_verifier.log",
}
MODES = (
    "preflight",
    "acquire",
    "smoke",
    "canonical",
    "download-only",
    "verify-only",
)
SMOKE_CASE_IDS = (
    "environment_preflight",
    "architecture_verification",
    "same_path_repeatability__cached_repeatability__p17__r0__c17",
    "same_path_repeatability__cached_repeatability__p17__r1__c17",
    "one_shot_vs_cached__one_shot_vs_cached__p17__r0__c17",
    "state_export_import__state_export_import__p17__r0__c17",
    "post_run_audit",
)
OWNED_SOURCE_FILES = (
    "tools/qwen35_hybrid_state_contract.py",
    "tools/qwen35_hybrid_state_probe.py",
    "tools/verify_qwen35_hybrid_state_gate.py",
    "tools/run_qwen35_hybrid_state_gate_remote.py",
    "tools/test_qwen35_hybrid_state_contract.py",
    "tools/test_qwen35_hybrid_state_probe.py",
    "tools/test_verify_qwen35_hybrid_state_gate.py",
    "tools/test_run_qwen35_hybrid_state_gate_remote.py",
)
MIB = 1024**2
GIB = 1024**3
ARTIFACT_ALLOWANCE_BYTES = 512 * MIB
SAFETY_RESERVE_BYTES = 2 * GIB
MAX_PORT_ATTEMPTS = 3
DOWNLOAD_CHUNK_BYTES = 4 * MIB
DOWNLOAD_ATTEMPTS = 3
RUN_TAG_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
REVISION_PATTERN = re.compile(r"^[0-9a-fA-F]{40}$")


def validate_run_tag(value):
    text = str(value)
    if not RUN_TAG_PATTERN.fullmatch(text):
        raise ValueError("run tag must match [A-Za-z0-9_-]+")
    return text


def validate_resolved_revision(value):
    text = str(value)
    if not REVISION_PATTERN.fullmatch(text):
        raise ValueError("resolved revision must be a 40-hex commit")
    return text.lower()


def validate_artifact_path(value):
    text = str(value)
    path = PurePosixPath(text)
    if (
        not text
        or path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
    ):
        raise ValueError("artifact path must be safe and relative")
    return path.as_posix()


def remote_run_dir(run_tag):
    return f"{REMOTE_RUN_ROOT}/{validate_run_tag(run_tag)}"


def local_run_dir(repo_root, run_tag):
    return Path(repo_root) / LOCAL_RUN_ROOT / validate_run_tag(run_tag)


def build_ssh_command(remote_arguments):
    remote_command = shlex.join([str(value) for value in remote_arguments])
    return [
        "ssh",
        "-S",
        SSH_CONTROL_PATH,
        "-o",
        "BatchMode=yes",
        REMOTE_TARGET,
        remote_command,
    ]


def _run(command, **kwargs):
    return subprocess.run(command, check=False, **kwargs)


def _require_success(result, context):
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"{context} failed: {detail}")
    return result


def require_clean_owned_source(repo_root, command_runner=_run):
    root = Path(repo_root)
    status = command_runner(
        ["git", "status", "--porcelain", "--", *OWNED_SOURCE_FILES],
        cwd=root,
        text=True,
        capture_output=True,
    )
    _require_success(status, "owned-source status")
    if status.stdout.strip():
        raise ValueError("owned source files must be clean")
    commit = command_runner(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
        capture_output=True,
    )
    _require_success(commit, "source commit resolution")
    resolved = commit.stdout.strip()
    if not resolved:
        return "HEAD"
    if not REVISION_PATTERN.fullmatch(resolved):
        raise ValueError("source commit must be a 40-hex commit")
    return resolved.lower()


def _sha256_path(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(MIB), b""):
            digest.update(block)
    return digest.hexdigest()


def owned_source_hashes(repo_root):
    root = Path(repo_root)
    hashes = {}
    for relative in OWNED_SOURCE_FILES:
        path = root / relative
        if not path.is_file():
            raise ValueError(f"missing owned source file: {relative}")
        hashes[relative] = _sha256_path(path)
    return hashes


def build_source_manifest(repo_root, commit, staged, command_runner=_run):
    branch_result = command_runner(
        ["git", "branch", "--show-current"],
        cwd=Path(repo_root),
        text=True,
        capture_output=True,
    )
    _require_success(branch_result, "source branch resolution")
    branch = branch_result.stdout.strip()
    if not branch:
        raise ValueError("source branch must not be detached")
    return {
        "schema_version": 1,
        "branch": branch,
        "commit": validate_resolved_revision(commit),
        "clean": True,
        "local_file_sha256": dict(staged["local_file_sha256"]),
        "remote_file_sha256": dict(staged["remote_file_sha256"]),
    }


def build_source_tar(repo_root):
    root = Path(repo_root)
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for relative in OWNED_SOURCE_FILES:
            path = root / relative
            if not path.is_file():
                raise ValueError(f"missing owned source file: {relative}")
            info = archive.gettarinfo(os.fspath(path), arcname=relative)
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            with path.open("rb") as source:
                archive.addfile(info, source)
    return stream.getvalue()


def stage_owned_source(repo_root, run_tag, command_runner=_run):
    destination = remote_run_dir(run_tag)
    source_dir = f"{destination}/source"
    payload = build_source_tar(repo_root)
    command = build_ssh_command([
        "bash",
        "-c",
        (
            f"test ! -e {shlex.quote(source_dir)} && "
            f"mkdir -p {shlex.quote(source_dir)} && "
            f"tar -xf - -C {shlex.quote(source_dir)}"
        ),
    ])
    result = command_runner(
        command,
        input=payload,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _require_success(result, "source staging")
    expected = owned_source_hashes(repo_root)
    script = (
        "import hashlib,json,pathlib\n"
        f"root=pathlib.Path({source_dir!r})\n"
        f"names={json.dumps(list(OWNED_SOURCE_FILES), separators=(',', ':'))}\n"
        "out={}\n"
        "for name in names:\n"
        " p=root/name\n"
        " if not p.is_file(): raise SystemExit('missing:'+name)\n"
        " out[name]=hashlib.sha256(p.read_bytes()).hexdigest()\n"
        "print(json.dumps(out,sort_keys=True,separators=(',',':')))\n"
    )
    verified = command_runner(
        build_ssh_command([REMOTE_PYTHON, "-c", script]),
        text=True,
        capture_output=True,
    )
    _require_success(verified, "remote source hashing")
    actual = json.loads(verified.stdout)
    if actual != expected:
        raise ValueError("remote staged source hashes do not match local")
    return {
        "remote_source_dir": source_dir,
        "local_file_sha256": expected,
        "remote_file_sha256": actual,
    }


def evaluate_disk_preflight(*, declared_model_file_bytes, free_bytes):
    declared = int(declared_model_file_bytes)
    free = int(free_bytes)
    if declared < 0 or free < 0:
        raise ValueError("disk byte values must be non-negative")
    required = (
        declared
        + declared
        + ARTIFACT_ALLOWANCE_BYTES
        + SAFETY_RESERVE_BYTES
    )
    can_acquire = free >= required
    return {
        "declared_model_file_bytes": declared,
        "free_bytes": free,
        "required_bytes": required,
        "can_acquire": can_acquire,
        "classification_detail": (
            None if can_acquire else "INCOMPLETE_RESOURCE_BLOCKED"
        ),
    }


def _allowed_model_file(name):
    path = PurePosixPath(str(name))
    if path.is_absolute() or ".." in path.parts:
        return False
    basename = path.name
    if basename.endswith(".safetensors"):
        return True
    if basename.endswith(".index.json"):
        return True
    if basename.endswith(".py"):
        return True
    exact = {
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
    }
    return basename in exact


def build_model_file_inventory(siblings):
    files = {}
    for sibling in siblings:
        if isinstance(sibling, dict):
            name = sibling.get("rfilename")
            size = sibling.get("size")
        else:
            name = getattr(sibling, "rfilename", None)
            size = getattr(sibling, "size", None)
        if not isinstance(name, str) or not _allowed_model_file(name):
            continue
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValueError(f"invalid model metadata size: {name}")
        files[name] = size
    if "config.json" not in files:
        raise ValueError("model metadata is missing config.json")
    if not any(name.endswith(".safetensors") for name in files):
        raise ValueError("model metadata is missing safetensors weights")
    allow_patterns = sorted(files)
    return {
        "allow_patterns": allow_patterns,
        "declared_model_file_bytes": sum(files.values()),
        "files": {
            name: {"size": files[name]}
            for name in allow_patterns
        },
    }


def build_snapshot_download_script(
    *,
    resolved_revision,
    remote_run_dir,
    allow_patterns,
):
    revision = validate_resolved_revision(resolved_revision)
    destination = PurePosixPath(str(remote_run_dir))
    if not destination.is_absolute() or ".." in destination.parts:
        raise ValueError("remote run directory must be an absolute safe path")
    patterns = [validate_artifact_path(name) for name in allow_patterns]
    encoded_patterns = json.dumps(patterns, separators=(",", ":"))
    return "\n".join([
        "from huggingface_hub import snapshot_download",
        "snapshot_download(",
        f'    repo_id="{MODEL_REPOSITORY}",',
        f'    revision="{revision}",',
        f'    local_dir="{destination.as_posix()}/model",',
        "    local_dir_use_symlinks=False,",
        f"    allow_patterns={encoded_patterns},",
        ")",
        "",
    ])


def hash_and_validate_model_files(model_dir, expected_files):
    root = Path(model_dir)
    expected = [validate_artifact_path(name) for name in expected_files]
    inventory = {}
    for name in expected:
        path = root.joinpath(*PurePosixPath(name).parts)
        if not path.is_file():
            raise ValueError(f"missing model file: {name}")
        inventory[name] = {
            "size": path.stat().st_size,
            "sha256": _sha256_path(path),
        }
    for name in expected:
        if not name.endswith(".index.json"):
            continue
        payload = json.loads(
            root.joinpath(*PurePosixPath(name).parts).read_text()
        )
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"invalid model index: {name}")
        for shard in set(weight_map.values()):
            if shard not in inventory:
                raise ValueError(f"missing indexed shard: {shard}")
    return inventory


def build_remote_model_hash_script(*, remote_model_dir, expected_files):
    root = PurePosixPath(str(remote_model_dir))
    if not root.is_absolute() or ".." in root.parts:
        raise ValueError("remote model directory must be absolute and safe")
    expected = sorted(
        validate_artifact_path(name) for name in expected_files
    )
    return "\n".join([
        "import hashlib,json,pathlib",
        f'root=pathlib.Path("{root.as_posix()}")',
        f"expected={json.dumps(expected, separators=(',', ':'))}",
        "actual=sorted(p.relative_to(root).as_posix()",
        "              for p in root.rglob('*')",
        "              if p.is_file() and not",
        "              p.relative_to(root).as_posix().startswith(",
        "               '.cache/huggingface/'))",
        "missing=sorted(set(expected)-set(actual))",
        "extra=sorted(set(actual)-set(expected))",
        "if missing: raise SystemExit('missing model file:'+missing[0])",
        "if extra: raise SystemExit('unexpected model file:'+extra[0])",
        "out={}",
        "for name in expected:",
        " path=root/name",
        " digest=hashlib.sha256()",
        " with path.open('rb') as stream:",
        "  for block in iter(lambda:stream.read(1024*1024),b''):",
        "   digest.update(block)",
        " out[name]={'size':path.stat().st_size,'sha256':digest.hexdigest()}",
        "for name in expected:",
        " if not name.endswith('.index.json'): continue",
        " payload=json.loads((root/name).read_text())",
        " weight_map=payload.get('weight_map')",
        " if not isinstance(weight_map,dict) or not weight_map:",
        "  raise SystemExit('invalid model index:'+name)",
        " for shard in set(weight_map.values()):",
        "  if shard not in out:",
        "   raise SystemExit('missing indexed shard:'+shard)",
        "print(json.dumps(out,sort_keys=True,separators=(',',':')))",
        "",
    ])


def build_model_manifest(*, resolved_revision, remote_model_dir, files):
    revision = validate_resolved_revision(resolved_revision)
    root = PurePosixPath(str(remote_model_dir))
    if not root.is_absolute() or ".." in root.parts:
        raise ValueError("remote model directory must be absolute and safe")
    validated = {}
    for name in sorted(files):
        safe_name = validate_artifact_path(name)
        entry = files[name]
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("size"), int)
            or isinstance(entry["size"], bool)
            or entry["size"] < 0
            or not re.fullmatch(
                r"[0-9a-f]{64}",
                str(entry.get("sha256", "")),
            )
        ):
            raise ValueError(f"invalid model file identity: {safe_name}")
        validated[safe_name] = {
            "size": entry["size"],
            "sha256": entry["sha256"],
        }
    if not validated:
        raise ValueError("model file inventory must not be empty")
    total_weight_bytes = sum(
        entry["size"]
        for name, entry in validated.items()
        if name.endswith(".safetensors")
    )
    return {
        "schema_version": 1,
        "repository": MODEL_REPOSITORY,
        "resolved_revision": revision,
        "local_path": root.as_posix(),
        "remote_model_dir": root.as_posix(),
        "files": validated,
        "total_weight_bytes": total_weight_bytes,
        "trust_remote_code": False,
    }


def enrich_model_manifest(model_manifest, architecture):
    required = (
        "config_class",
        "model_class",
        "tokenizer_class",
        "tokenizer_vocab_size",
        "parameter_dtypes",
    )
    enriched = dict(model_manifest)
    for field in required:
        if field not in architecture:
            raise ValueError(f"architecture is missing {field}")
        enriched[field] = architecture[field]
    enriched["requested_dtype"] = "auto"
    return enriched


def build_environment_manifest(runtime, worker_attempt):
    if runtime.get("host") != "10.232.195.203":
        raise ValueError("remote runtime host mismatch")
    if runtime.get("user") != "sitian":
        raise ValueError("remote runtime user mismatch")
    packages = runtime.get("packages")
    if not isinstance(packages, dict):
        raise ValueError("runtime packages are missing")
    tiny_port = worker_attempt.get("tinyvllm_dist_port")
    master_port = worker_attempt.get("master_port")
    if (
        not isinstance(tiny_port, int)
        or not isinstance(master_port, int)
        or tiny_port == master_port
    ):
        raise ValueError("worker attempt ports are invalid")
    return {
        "schema_version": 1,
        "host": "10.232.195.203",
        "user": "sitian",
        "gpu_name": runtime.get("gpu_name"),
        "gpu_uuid": runtime.get("gpu_uuid"),
        "driver_version": runtime.get("driver_version"),
        "cuda_runtime_version": runtime.get("cuda_runtime_version"),
        "python_executable": REMOTE_PYTHON,
        "python_version": runtime.get("python_version"),
        "torch_version": packages.get("torch"),
        "transformers_version": packages.get("transformers"),
        "optional_packages": {
            name: packages.get(name)
            for name in (
                "fla",
                "causal_conv1d",
                "triton",
                "flash_attn",
            )
        },
        "environment": {
            "CUDA_VISIBLE_DEVICES": "0",
            "TINYVLLM_DIST_PORT": str(tiny_port),
            "MASTER_PORT": str(master_port),
        },
        "gpu_processes_before": list(runtime.get("gpu_processes", [])),
        "gpu_processes_after": list(
            runtime.get("gpu_processes_after", [])
        ),
    }


def allocate_unique_port_pairs(count, allocator=None):
    requested = int(count)
    if requested < 0:
        raise ValueError("port pair count must be non-negative")
    source = allocator or (lambda: 20000 + secrets.randbelow(40000))
    used = set()
    pairs = []
    for _ in range(requested):
        pair = (source(), source())
        for value in pair:
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or not 1 <= value <= 65535
            ):
                raise ValueError("allocator must return a valid port")
            if value in used:
                raise ValueError("allocator must return globally unique ports")
            used.add(value)
        pairs.append(pair)
    return pairs


def is_retryable_port_collision(attempt, stderr):
    return (
        isinstance(attempt, int)
        and 1 <= attempt < MAX_PORT_ATTEMPTS
        and "EADDRINUSE" in str(stderr)
    )


def run_with_port_retries(name, *, launch, pair_allocator):
    attempts = []
    for attempt in range(1, MAX_PORT_ATTEMPTS + 1):
        tiny_port, master_port = pair_allocator()
        if tiny_port == master_port:
            raise ValueError("process port pair must be distinct")
        result = dict(launch(attempt, tiny_port, master_port))
        row = {
            "name": str(name),
            "attempt": attempt,
            "tinyvllm_dist_port": tiny_port,
            "master_port": master_port,
            "exit_code": int(result.get("exit_code", 1)),
            "stdout": str(result.get("stdout", "")),
            "stderr": str(result.get("stderr", "")),
        }
        attempts.append(row)
        if row["exit_code"] == 0:
            return {"success": True, "attempts": attempts}
        if not is_retryable_port_collision(attempt, row["stderr"]):
            break
    return {"success": False, "attempts": attempts}


def build_remote_execution_plan(
    *,
    mode,
    remote_source_dir,
    remote_model_dir,
    remote_artifact_dir,
    contract_sha256,
):
    if mode not in {"smoke", "canonical"}:
        raise ValueError("execution plan mode must be smoke or canonical")
    source_root = PurePosixPath(str(remote_source_dir))
    model_root = PurePosixPath(str(remote_model_dir))
    artifact_root = PurePosixPath(str(remote_artifact_dir))
    for name, path in (
        ("source", source_root),
        ("model", model_root),
        ("artifact", artifact_root),
    ):
        if not path.is_absolute() or ".." in path.parts:
            raise ValueError(f"remote {name} directory must be absolute")
    digest = str(contract_sha256)
    if not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
        raise ValueError("contract sha256 must be 64-hex")
    test_files = (
        ("contract", "tools/test_qwen35_hybrid_state_contract.py"),
        ("probe", "tools/test_qwen35_hybrid_state_probe.py"),
        ("verifier", "tools/test_verify_qwen35_hybrid_state_gate.py"),
        ("runner", "tools/test_run_qwen35_hybrid_state_gate_remote.py"),
    )
    source_tests = []
    for name, relative in test_files:
        source_tests.append({
            "name": name,
            "command": [
                REMOTE_PYTHON,
                (source_root / relative).as_posix(),
            ],
            "environment": {"CUDA_VISIBLE_DEVICES": ""},
        })
    if mode == "smoke":
        worker_command = [
            REMOTE_PYTHON,
            "-c",
            build_smoke_probe_script(
                remote_source_dir=source_root.as_posix(),
                remote_model_dir=model_root.as_posix(),
                remote_artifact_dir=artifact_root.as_posix(),
                contract_sha256=digest,
            ),
        ]
    else:
        worker_command = [
            REMOTE_PYTHON,
            (source_root / "tools/qwen35_hybrid_state_probe.py").as_posix(),
            "run-canonical",
            "--model-dir",
            model_root.as_posix(),
            "--run-dir",
            artifact_root.as_posix(),
            "--contract-sha256",
            digest.lower(),
        ]
    worker = {
        "name": mode,
        "command": worker_command,
        "environment": {"CUDA_VISIBLE_DEVICES": "0"},
    }
    return {"source_tests": source_tests, "worker": worker}


def build_smoke_probe_script(
    *,
    remote_source_dir,
    remote_model_dir,
    remote_artifact_dir,
    contract_sha256,
):
    source_root = PurePosixPath(str(remote_source_dir))
    model_root = PurePosixPath(str(remote_model_dir))
    artifact_root = PurePosixPath(str(remote_artifact_dir))
    digest = str(contract_sha256).lower()
    for path in (source_root, model_root, artifact_root):
        if not path.is_absolute() or ".." in path.parts:
            raise ValueError("smoke paths must be absolute and safe")
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("contract sha256 must be 64-hex")
    selected = json.dumps(list(SMOKE_CASE_IDS), separators=(",", ":"))
    arguments = json.dumps([
        "run-canonical",
        "--model-dir",
        model_root.as_posix(),
        "--run-dir",
        artifact_root.as_posix(),
        "--contract-sha256",
        digest,
    ], separators=(",", ":"))
    return "\n".join([
        "import importlib.util,pathlib,sys",
        f"root=pathlib.Path({source_root.as_posix()!r})",
        "probe_path=root/'tools/qwen35_hybrid_state_probe.py'",
        "probe_spec=importlib.util.spec_from_file_location(",
        " 'qwen35_hybrid_state_probe',probe_path)",
        "probe=importlib.util.module_from_spec(probe_spec)",
        "sys.modules[probe_spec.name]=probe",
        "probe_spec.loader.exec_module(probe)",
        "original_build_case_matrix=probe.contract.build_case_matrix",
        f"selected_case_ids=set({selected})",
        "def smoke_build_case_matrix():",
        " return tuple(case for case in original_build_case_matrix()",
        "              if case.case_id in selected_case_ids)",
        "probe.contract.build_case_matrix=smoke_build_case_matrix",
        f"raise SystemExit(probe.main({arguments}))",
        "",
    ])


def audit_smoke_case_rows(rows):
    observed = {}
    for row in rows:
        case_id = row.get("case_id")
        if case_id in observed or case_id not in SMOKE_CASE_IDS:
            raise ValueError("smoke case domain mismatch")
        if row.get("complete") is not True:
            raise ValueError(f"incomplete smoke case: {case_id}")
        if row.get("failure_kind") is not None:
            raise ValueError(f"incomplete smoke case: {case_id}")
        observed[case_id] = row
    if set(observed) != set(SMOKE_CASE_IDS):
        raise ValueError("smoke case domain mismatch")
    return {
        "classification": "SMOKE_PASS",
        "case_ids": list(SMOKE_CASE_IDS),
        "claim_boundary": (
            "Smoke compatibility prerequisite only; not canonical GO."
        ),
    }


def select_verified_model_snapshot(candidates, *, resolved_revision):
    revision = validate_resolved_revision(resolved_revision)
    matches = []
    for candidate in candidates:
        if candidate.get("repository") != MODEL_REPOSITORY:
            continue
        if candidate.get("resolved_revision") != revision:
            continue
        files = candidate.get("files")
        remote_model_dir = candidate.get("remote_model_dir")
        if not isinstance(files, dict) or not files:
            continue
        if not isinstance(remote_model_dir, str):
            continue
        path = PurePosixPath(remote_model_dir)
        if not path.is_absolute() or ".." in path.parts:
            continue
        valid_files = True
        for entry in files.values():
            if (
                not isinstance(entry, dict)
                or not isinstance(entry.get("size"), int)
                or entry["size"] < 0
                or not re.fullmatch(
                    r"[0-9a-f]{64}",
                    str(entry.get("sha256", "")),
                )
            ):
                valid_files = False
                break
        if valid_files:
            matches.append(candidate)
    if len(matches) != 1:
        raise ValueError("expected exactly one verified model snapshot")
    return dict(matches[0])


def verify_remote_model_snapshot(snapshot, command_runner=_run):
    expected = snapshot.get("files")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("model snapshot files are missing")
    script = build_remote_model_hash_script(
        remote_model_dir=snapshot["remote_model_dir"],
        expected_files=sorted(expected),
    )
    result = _remote_python(script, command_runner=command_runner)
    _require_success(result, "remote model hashing")
    actual = json.loads(result.stdout)
    if actual != expected:
        raise ValueError("model snapshot hash mismatch")
    return actual


def launch_remote_worker(
    *,
    name,
    command,
    base_environment,
    pair_allocator,
    command_runner=_run,
):
    worker_command = [str(value) for value in command]
    environment = {
        str(key): str(value)
        for key, value in dict(base_environment).items()
    }

    def launch(_attempt, tiny_port, master_port):
        attempt_environment = {
            **environment,
            "TINYVLLM_DIST_PORT": str(tiny_port),
            "MASTER_PORT": str(master_port),
        }
        assignments = [
            f"{key}={attempt_environment[key]}"
            for key in sorted(attempt_environment)
        ]
        result = command_runner(
            build_ssh_command(["env", *assignments, *worker_command]),
            text=True,
            capture_output=True,
        )
        return {
            "exit_code": result.returncode,
            "stdout": result.stdout or "",
            "stderr": result.stderr or "",
            "command": worker_command,
            "environment": attempt_environment,
        }

    result = run_with_port_retries(
        name,
        launch=launch,
        pair_allocator=pair_allocator,
    )
    for row in result["attempts"]:
        row["command"] = list(worker_command)
        row["environment"] = {
            **environment,
            "TINYVLLM_DIST_PORT": str(row["tinyvllm_dist_port"]),
            "MASTER_PORT": str(row["master_port"]),
        }
    return result


def iter_download_ranges(size, *, chunk_size=DOWNLOAD_CHUNK_BYTES):
    total = int(size)
    chunk = int(chunk_size)
    if total < 0 or chunk <= 0:
        raise ValueError("download sizes must be valid")
    offset = 0
    while offset < total:
        length = min(chunk, total - offset)
        yield offset, length
        offset += length


def mode_policy(mode):
    policies = {
        "preflight": {
            "uses_ssh": True,
            "stages_source": True,
            "launches_process": False,
            "downloads": True,
            "verifies": False,
        },
        "acquire": {
            "uses_ssh": True,
            "stages_source": True,
            "launches_process": False,
            "downloads": True,
            "verifies": False,
        },
        "smoke": {
            "uses_ssh": True,
            "stages_source": True,
            "launches_process": True,
            "downloads": True,
            "verifies": True,
        },
        "canonical": {
            "uses_ssh": True,
            "stages_source": True,
            "launches_process": True,
            "downloads": True,
            "verifies": True,
        },
        "download-only": {
            "uses_ssh": True,
            "stages_source": False,
            "launches_process": False,
            "downloads": True,
            "verifies": False,
        },
        "verify-only": {
            "uses_ssh": False,
            "stages_source": False,
            "launches_process": False,
            "downloads": False,
            "verifies": True,
        },
    }
    if mode not in policies:
        raise ValueError(f"unsupported runner mode: {mode}")
    return dict(policies[mode])


def _atomic_json(path, payload):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    temporary.write_text(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def _artifact_inventory(run_dir):
    root = Path(run_dir)
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        rows.append({
            "path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "sha256": _sha256_path(path),
        })
    return rows


def _classification_input_inventory(run_dir):
    return [
        row for row in _artifact_inventory(run_dir)
        if row["path"] not in VERIFIER_OUTPUT_FILES
    ]


def write_complete_manifest(
    run_dir,
    *,
    source_commit,
    model_revision,
):
    destination = Path(run_dir)
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "classification": None,
        "source_commit": source_commit,
        "model_repository": MODEL_REPOSITORY,
        "model_resolved_revision": model_revision,
        "failure_kind": None,
        "failure_detail": None,
        "artifacts": _classification_input_inventory(destination),
    }
    _atomic_json(destination / "manifest.json", manifest)
    return manifest


def write_incomplete_manifest(
    run_dir,
    *,
    source_commit,
    model_revision,
    failure_kind,
    failure_detail,
):
    destination = Path(run_dir)
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "classification": "INCOMPLETE",
        "source_commit": source_commit,
        "model_repository": MODEL_REPOSITORY,
        "model_resolved_revision": model_revision,
        "failure_kind": str(failure_kind),
        "failure_detail": str(failure_detail),
        "artifacts": _artifact_inventory(destination),
    }
    _atomic_json(destination / "manifest.json", manifest)
    return manifest


def preserve_failed_execution(
    run_dir,
    *,
    source_commit,
    model_revision,
    failure_kind,
    failure_detail,
    download,
):
    try:
        download()
    except Exception as exc:
        failure_detail = (
            f"{failure_detail}; artifact download failed: {exc}"
        )
    return write_incomplete_manifest(
        run_dir,
        source_commit=source_commit,
        model_revision=model_revision,
        failure_kind=failure_kind,
        failure_detail=failure_detail,
    )


def validate_smoke_binding(
    smoke,
    *,
    source_commit,
    source_file_sha256,
    model_resolved_revision,
    model_files,
):
    expected = {
        "classification": "SMOKE_PASS",
        "source_commit": source_commit,
        "source_file_sha256": source_file_sha256,
        "model_resolved_revision": model_resolved_revision,
        "model_files": model_files,
    }
    for field, value in expected.items():
        if smoke.get(field) != value:
            raise ValueError(f"smoke binding mismatch: {field}")
    return True


def run_remote_source_tests(plan, command_runner=_run):
    rows = []
    for step in plan.get("source_tests", []):
        environment = {
            str(key): str(value)
            for key, value in step["environment"].items()
        }
        assignments = [
            f"{key}={environment[key]}" for key in sorted(environment)
        ]
        command = [str(value) for value in step["command"]]
        result = command_runner(
            build_ssh_command(["env", *assignments, *command]),
            text=True,
            capture_output=True,
        )
        row = {
            "name": str(step["name"]),
            "command": command,
            "environment": environment,
            "exit_code": int(result.returncode),
            "stdout": result.stdout or "",
            "stderr": result.stderr or "",
        }
        rows.append(row)
        if row["exit_code"] != 0:
            break
    return rows


def discover_model_snapshots(repo_root, *, resolved_revision):
    revision = validate_resolved_revision(resolved_revision)
    root = Path(repo_root) / LOCAL_RUN_ROOT
    candidates = []
    if not root.is_dir():
        return candidates
    for path in sorted(root.glob("*/model_manifest.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if payload.get("resolved_revision") != revision:
            continue
        payload = dict(payload)
        payload["run_tag"] = path.parent.name
        candidates.append(payload)
    return candidates


def build_process_manifests(execution):
    attempts = [dict(row) for row in execution.get("attempts", [])]
    successful = [
        row for row in attempts if row.get("exit_code") == 0
    ]
    process_rows = []
    port_rows = []
    for attempt in successful:
        process_rows.append({
            "name": attempt["name"],
            "attempt": attempt["attempt"],
            "command": list(attempt["command"]),
            "stdout_path": (
                f"stdout/{attempt['name']}-attempt-{attempt['attempt']}.log"
            ),
            "stderr_path": (
                f"stderr/{attempt['name']}-attempt-{attempt['attempt']}.log"
            ),
            "exit_code": attempt["exit_code"],
            "tinyvllm_dist_port": attempt["tinyvllm_dist_port"],
            "master_port": attempt["master_port"],
        })
        port_rows.append({
            "process": attempt["name"],
            "attempt": attempt["attempt"],
            "tinyvllm_dist_port": attempt["tinyvllm_dist_port"],
            "master_port": attempt["master_port"],
        })
    return (
        {"processes": process_rows},
        {"pairs": port_rows},
        {"attempts": attempts},
    )


def _remote_python(script, command_runner=_run):
    return command_runner(
        build_ssh_command([REMOTE_PYTHON, "-c", script]),
        text=True,
        capture_output=True,
    )


def build_preflight_script(run_tag):
    destination = remote_run_dir(run_tag)
    return "\n".join([
        "import getpass,importlib.metadata,json,os,pathlib,platform,shutil,subprocess,sys",
        "from huggingface_hub import HfApi",
        f"repo={MODEL_REPOSITORY!r}",
        "info=HfApi().model_info(repo,files_metadata=True)",
        "siblings=[{'rfilename':x.rfilename,'size':x.size} for x in info.siblings]",
        f"root=pathlib.Path({destination!r})",
        "root.mkdir(parents=True,exist_ok=True)",
        "usage=shutil.disk_usage(root)",
        "def version(name):",
        " try:return importlib.metadata.version(name)",
        " except importlib.metadata.PackageNotFoundError:return None",
        "gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits'],text=True,capture_output=True)",
        "gpu_identity=subprocess.run(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv,noheader'],text=True,capture_output=True)",
        "gpu_fields=[x.strip() for x in (gpu_identity.stdout.splitlines() or [''])[0].split(',')]",
        "torch_cuda=None",
        "try:",
        " import torch",
        " torch_cuda=torch.version.cuda",
        "except Exception:",
        " pass",
        "cache_roots=[pathlib.Path(os.environ.get('HF_HOME',pathlib.Path.home()/'.cache/huggingface'))/'hub',pathlib.Path(os.environ.get('HUGGINGFACE_HUB_CACHE',pathlib.Path.home()/'.cache/huggingface/hub'))]",
        "cache_roots=list(dict.fromkeys(str(path) for path in cache_roots))",
        "candidate_snapshots=[]",
        "for cache_root in cache_roots:",
        " root_path=pathlib.Path(cache_root)",
        " if root_path.is_dir():",
        "  candidate_snapshots.extend(str(path) for path in sorted(root_path.glob('models--Qwen--Qwen3.5-2B/snapshots/*')) if path.is_dir())",
        "out={'resolved_revision':info.sha,'siblings':siblings,'free_bytes':usage.free,'packages':{name:version(name) for name in ['torch','transformers','huggingface_hub','fla','causal_conv1d','triton','flash_attn']},'gpu_processes':gpu.stdout.splitlines(),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'host':'10.232.195.203','observed_hostname':platform.node(),'user':getpass.getuser(),'python_version':sys.version.split()[0],'gpu_name':gpu_fields[0] if len(gpu_fields)>0 else None,'gpu_uuid':gpu_fields[1] if len(gpu_fields)>1 else None,'driver_version':gpu_fields[2] if len(gpu_fields)>2 else None,'cuda_runtime_version':torch_cuda,'checked_cache_roots':cache_roots,'candidate_snapshots':candidate_snapshots}",
        "print(json.dumps(out,sort_keys=True,separators=(',',':')))",
        "",
    ])


def run_remote_preflight(run_tag, command_runner=_run):
    result = _remote_python(
        build_preflight_script(run_tag),
        command_runner=command_runner,
    )
    _require_success(result, "remote preflight")
    payload = json.loads(result.stdout)
    revision = validate_resolved_revision(payload["resolved_revision"])
    inventory = build_model_file_inventory(payload["siblings"])
    disk = evaluate_disk_preflight(
        declared_model_file_bytes=inventory["declared_model_file_bytes"],
        free_bytes=payload["free_bytes"],
    )
    return {
        **payload,
        **inventory,
        "resolved_revision": revision,
        "disk_preflight": disk,
    }


def build_preflight_summary(payload):
    disk = payload.get("disk_preflight")
    if not isinstance(disk, dict):
        raise ValueError("preflight disk decision is missing")
    if disk.get("can_acquire") is True:
        status = "READY_TO_ACQUIRE"
    else:
        status = disk.get("classification_detail")
        if not isinstance(status, str) or not status.startswith(
            "INCOMPLETE_"
        ):
            raise ValueError("preflight status is not fail-closed")
    packages = payload.get("packages")
    if not isinstance(packages, dict):
        raise ValueError("preflight runtime packages are missing")
    return {
        "status": status,
        "resolved_revision": validate_resolved_revision(
            payload["resolved_revision"]
        ),
        "declared_model_file_bytes": int(
            payload["declared_model_file_bytes"]
        ),
        "free_bytes": int(disk["free_bytes"]),
        "required_acquisition_peak_bytes": int(disk["required_bytes"]),
        "runtime": {
            "python_executable": REMOTE_PYTHON,
            "packages": packages,
        },
        "gpu_processes": list(payload.get("gpu_processes", [])),
        "checked_cache_roots": list(
            payload.get("checked_cache_roots", [])
        ),
        "candidate_snapshots": list(
            payload.get("candidate_snapshots", [])
        ),
    }


def acquire_model(
    run_tag,
    resolved_revision,
    *,
    command_runner=_run,
):
    revision = validate_resolved_revision(resolved_revision)
    preflight = run_remote_preflight(run_tag, command_runner=command_runner)
    if preflight["resolved_revision"] != revision:
        raise ValueError("resolved revision drifted before acquisition")
    if not preflight["disk_preflight"]["can_acquire"]:
        return {
            "classification": "INCOMPLETE",
            "failure_kind": "INCOMPLETE_RESOURCE_BLOCKED",
            "preflight": preflight,
        }
    script = build_snapshot_download_script(
        resolved_revision=revision,
        remote_run_dir=remote_run_dir(run_tag),
        allow_patterns=preflight["allow_patterns"],
    )
    result = _remote_python(script, command_runner=command_runner)
    _require_success(result, "immutable model acquisition")
    model_dir = f"{remote_run_dir(run_tag)}/model"
    hash_result = _remote_python(
        build_remote_model_hash_script(
            remote_model_dir=model_dir,
            expected_files=preflight["allow_patterns"],
        ),
        command_runner=command_runner,
    )
    _require_success(hash_result, "acquired model hashing")
    files = json.loads(hash_result.stdout)
    declared = preflight["files"]
    for name, entry in files.items():
        if entry["size"] != declared[name]["size"]:
            raise ValueError(f"acquired model size mismatch: {name}")
    return {
        "classification": None,
        "resolved_revision": revision,
        "allow_patterns": preflight["allow_patterns"],
        "model_manifest": build_model_manifest(
            resolved_revision=revision,
            remote_model_dir=model_dir,
            files=files,
        ),
        "preflight": preflight,
    }


def run_acquisition_mode(
    *,
    run_dir,
    source_commit,
    resolved_revision,
    acquire,
):
    try:
        result = acquire()
    except Exception as exc:
        return write_incomplete_manifest(
            run_dir,
            source_commit=source_commit,
            model_revision=resolved_revision,
            failure_kind="INCOMPLETE_ACQUISITION_FAILURE",
            failure_detail=str(exc),
        )
    if result.get("classification") == "INCOMPLETE":
        result = {
            **result,
            "failure_detail": result.get(
                "failure_detail",
                "immutable model acquisition was blocked",
            ),
        }
        write_incomplete_manifest(
            run_dir,
            source_commit=source_commit,
            model_revision=resolved_revision,
            failure_kind=result["failure_kind"],
            failure_detail=result["failure_detail"],
        )
    return result


def _remote_file_listing(
    run_tag,
    *,
    remote_subdir="artifacts",
    command_runner=_run,
):
    relative_root = validate_artifact_path(remote_subdir)
    destination = f"{remote_run_dir(run_tag)}/{relative_root}"
    script = "\n".join([
        "import json,pathlib",
        f"root=pathlib.Path({destination!r})",
        "rows=[]",
        "for path in sorted(root.rglob('*')):",
        " if path.is_file(): rows.append({'path':path.relative_to(root).as_posix(),'size':path.stat().st_size})",
        "print(json.dumps(rows,separators=(',',':')))",
    ])
    result = _remote_python(script, command_runner=command_runner)
    _require_success(result, "remote artifact listing")
    rows = json.loads(result.stdout)
    for row in rows:
        validate_artifact_path(row["path"])
        if not isinstance(row["size"], int) or row["size"] < 0:
            raise ValueError("remote artifact has invalid size")
    return rows


def _read_remote_range(
    run_tag,
    relative_path,
    offset,
    length,
    command_runner,
    *,
    remote_subdir="artifacts",
):
    relative_root = validate_artifact_path(remote_subdir)
    destination = f"{remote_run_dir(run_tag)}/{relative_root}"
    full_path = f"{destination}/{validate_artifact_path(relative_path)}"
    script = "\n".join([
        "import pathlib,sys",
        f"path=pathlib.Path({full_path!r})",
        f"offset={int(offset)}",
        f"length={int(length)}",
        "with path.open('rb') as stream:",
        " stream.seek(offset)",
        " sys.stdout.buffer.write(stream.read(length))",
    ])
    return command_runner(
        build_ssh_command([REMOTE_PYTHON, "-c", script]),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def download_artifacts(
    run_tag,
    local_destination,
    command_runner=_run,
    *,
    remote_subdir="artifacts",
):
    destination = Path(local_destination)
    destination.mkdir(parents=True, exist_ok=True)
    rows = _remote_file_listing(
        run_tag,
        remote_subdir=remote_subdir,
        command_runner=command_runner,
    )
    for row in rows:
        relative = validate_artifact_path(row["path"])
        size = row["size"]
        path = destination.joinpath(*PurePosixPath(relative).parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".partial")
        with partial.open("wb") as output:
            for offset, length in iter_download_ranges(size):
                for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
                    result = _read_remote_range(
                        run_tag,
                        relative,
                        offset,
                        length,
                        command_runner,
                        remote_subdir=remote_subdir,
                    )
                    if result.returncode == 0 and len(result.stdout) == length:
                        output.write(result.stdout)
                        break
                    if attempt == DOWNLOAD_ATTEMPTS:
                        raise RuntimeError(
                            f"artifact block download failed: {relative}"
                        )
        if partial.stat().st_size != size:
            raise RuntimeError(f"artifact byte count mismatch: {relative}")
        os.replace(partial, path)
    return rows


def run_local_verifier(
    repo_root,
    run_dir,
    command_runner=_run,
    *,
    domain="canonical",
):
    if domain not in {"canonical", "smoke"}:
        raise ValueError("verifier domain must be canonical or smoke")
    root = Path(repo_root)
    stdout_dir = Path(run_dir) / "stdout"
    stderr_dir = Path(run_dir) / "stderr"
    stdout_dir.mkdir(parents=True, exist_ok=True)
    stderr_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        os.fspath(root / "tools/verify_qwen35_hybrid_state_gate.py"),
        "--run-dir",
        os.fspath(run_dir),
        "--write-report",
    ]
    if domain == "smoke":
        command.extend(["--domain", "smoke"])
    result = command_runner(
        command,
        cwd=root,
        text=True,
        capture_output=True,
    )
    (stdout_dir / "local_verifier.log").write_text(result.stdout or "")
    (stderr_dir / "local_verifier.log").write_text(result.stderr or "")
    record = {
        "command": command,
        "exit_code": result.returncode,
        "stdout_sha256": hashlib.sha256(
            (result.stdout or "").encode()
        ).hexdigest(),
        "stderr_sha256": hashlib.sha256(
            (result.stderr or "").encode()
        ).hexdigest(),
    }
    _atomic_json(Path(run_dir) / "local_verifier_process.json", record)
    return record


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_jsonl(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        rows.append(json.loads(line))
    return rows


def _write_process_logs(run_dir, execution):
    destination = Path(run_dir)
    (destination / "stdout").mkdir(parents=True, exist_ok=True)
    (destination / "stderr").mkdir(parents=True, exist_ok=True)
    for attempt in execution["attempts"]:
        stem = f"{attempt['name']}-attempt-{attempt['attempt']}.log"
        (destination / "stdout" / stem).write_text(
            attempt["stdout"],
            encoding="utf-8",
        )
        (destination / "stderr" / stem).write_text(
            attempt["stderr"],
            encoding="utf-8",
        )


def _fresh_pair_allocator():
    used = set()

    def allocate():
        pair = []
        while len(pair) < 2:
            value = 20000 + secrets.randbelow(40000)
            if value in used:
                continue
            used.add(value)
            pair.append(value)
        return tuple(pair)

    return allocate


def execute_remote_gate(
    *,
    mode,
    run_tag,
    resolved_revision,
    repo_root,
    source_commit,
    staged_source,
    smoke_run_tag=None,
    command_runner=_run,
):
    revision = validate_resolved_revision(resolved_revision)
    destination = local_run_dir(repo_root, run_tag)
    destination.mkdir(parents=True, exist_ok=True)
    source_manifest = build_source_manifest(
        repo_root,
        source_commit,
        staged_source,
        command_runner=command_runner,
    )
    candidates = discover_model_snapshots(
        repo_root,
        resolved_revision=revision,
    )
    model_snapshot = select_verified_model_snapshot(
        candidates,
        resolved_revision=revision,
    )
    verify_remote_model_snapshot(
        model_snapshot,
        command_runner=command_runner,
    )
    if mode == "canonical":
        smoke_path = local_run_dir(
            repo_root,
            validate_run_tag(smoke_run_tag),
        ) / "smoke_evidence.json"
        smoke = _read_json(smoke_path)
        validate_smoke_binding(
            smoke,
            source_commit=source_commit,
            source_file_sha256=source_manifest["local_file_sha256"],
            model_resolved_revision=revision,
            model_files=model_snapshot["files"],
        )
    runtime = run_remote_preflight(
        run_tag,
        command_runner=command_runner,
    )
    if runtime["resolved_revision"] != revision:
        raise ValueError("resolved revision drifted before worker launch")
    contract_sha256 = source_manifest["local_file_sha256"][
        "tools/qwen35_hybrid_state_contract.py"
    ]
    plan = build_remote_execution_plan(
        mode=mode,
        remote_source_dir=staged_source["remote_source_dir"],
        remote_model_dir=model_snapshot["remote_model_dir"],
        remote_artifact_dir=f"{remote_run_dir(run_tag)}/artifacts",
        contract_sha256=contract_sha256,
    )
    source_tests = run_remote_source_tests(
        plan,
        command_runner=command_runner,
    )
    _atomic_json(destination / "source_tests.json", {
        "tests": source_tests,
    })
    if (
        len(source_tests) != len(plan["source_tests"])
        or any(row["exit_code"] != 0 for row in source_tests)
    ):
        return preserve_failed_execution(
            destination,
            source_commit=source_commit,
            model_revision=revision,
            failure_kind="INCOMPLETE_SOURCE_TEST_FAILURE",
            failure_detail="remote source tests did not all pass",
            download=lambda: download_artifacts(
                run_tag,
                destination,
                command_runner=command_runner,
            ),
        )
    execution = launch_remote_worker(
        name=mode,
        command=plan["worker"]["command"],
        base_environment=plan["worker"]["environment"],
        pair_allocator=_fresh_pair_allocator(),
        command_runner=command_runner,
    )
    _write_process_logs(destination, execution)
    processes, ports, attempts = build_process_manifests(execution)
    _atomic_json(destination / "processes.json", processes)
    _atomic_json(destination / "ports.json", ports)
    _atomic_json(destination / "process_attempts.json", attempts)
    if not execution["success"]:
        return preserve_failed_execution(
            destination,
            source_commit=source_commit,
            model_revision=revision,
            failure_kind="INCOMPLETE_WORKER_FAILURE",
            failure_detail="remote worker did not exit cleanly",
            download=lambda: download_artifacts(
                run_tag,
                destination,
                command_runner=command_runner,
            ),
        )
    download_artifacts(
        run_tag,
        destination,
        command_runner=command_runner,
    )
    architecture = _read_json(destination / "architecture.json")
    model_manifest = enrich_model_manifest(
        model_snapshot,
        architecture,
    )
    successful_attempt = next(
        row for row in reversed(execution["attempts"])
        if row["exit_code"] == 0
    )
    environment = build_environment_manifest(runtime, successful_attempt)
    _atomic_json(destination / "source_manifest.json", source_manifest)
    _atomic_json(destination / "model_manifest.json", model_manifest)
    _atomic_json(destination / "environment.json", environment)
    if mode == "smoke":
        audit_smoke_case_rows(
            _read_jsonl(destination / "case_rows.jsonl")
        )
        write_complete_manifest(
            destination,
            source_commit=source_commit,
            model_revision=revision,
        )
        verifier = run_local_verifier(
            repo_root,
            destination,
            domain="smoke",
        )
        verification = _read_json(
            destination / "independent_verification.json"
        )
        if verification.get("classification") != "SMOKE_PASS":
            return {
                "classification": "INCOMPLETE",
                "verification": verification,
                "verifier_process": verifier,
            }
        smoke_evidence = {
            "classification": "SMOKE_PASS",
            "case_ids": list(SMOKE_CASE_IDS),
            "claim_boundary": (
                "Smoke compatibility prerequisite only; not canonical GO."
            ),
            "source_commit": source_commit,
            "source_file_sha256": source_manifest["local_file_sha256"],
            "model_resolved_revision": revision,
            "model_files": model_snapshot["files"],
            "independent_verification_sha256": _sha256_path(
                destination / "independent_verification.json"
            ),
        }
        _atomic_json(destination / "smoke_evidence.json", smoke_evidence)
        return smoke_evidence
    write_complete_manifest(
        destination,
        source_commit=source_commit,
        model_revision=revision,
    )
    verifier = run_local_verifier(repo_root, destination)
    verification = _read_json(
        destination / "independent_verification.json"
    )
    return {
        "classification": verification["classification"],
        "verification": verification,
        "verifier_process": verifier,
    }


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=MODES)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--resolved-revision")
    parser.add_argument("--smoke-run-tag")
    parser.add_argument("--repo-root", default=os.fspath(Path.cwd()))
    return parser.parse_args(argv)


def validate_mode_arguments(*, mode, resolved_revision, smoke_run_tag):
    if mode in {"acquire", "smoke", "canonical"}:
        if resolved_revision is None:
            raise ValueError(f"{mode} requires --resolved-revision")
        validate_resolved_revision(resolved_revision)
    if mode == "canonical":
        if smoke_run_tag is None:
            raise ValueError("canonical requires --smoke-run-tag")
        validate_run_tag(smoke_run_tag)
    elif smoke_run_tag is not None:
        raise ValueError("--smoke-run-tag is valid only for canonical")
    return True


def main(argv=None):
    arguments = parse_arguments(argv)
    run_tag = validate_run_tag(arguments.run_tag)
    validate_mode_arguments(
        mode=arguments.mode,
        resolved_revision=arguments.resolved_revision,
        smoke_run_tag=arguments.smoke_run_tag,
    )
    repo_root = Path(arguments.repo_root).resolve()
    destination = local_run_dir(repo_root, run_tag)
    policy = mode_policy(arguments.mode)
    if arguments.mode == "verify-only":
        record = run_local_verifier(repo_root, destination)
        print(json.dumps(record, sort_keys=True))
        return int(record["exit_code"] != 0)
    if arguments.mode == "download-only":
        rows = download_artifacts(run_tag, destination)
        print(json.dumps({"downloaded": len(rows)}, sort_keys=True))
        return 0
    commit = require_clean_owned_source(repo_root)
    staged = stage_owned_source(repo_root, run_tag)
    if arguments.mode == "preflight":
        result = run_remote_preflight(run_tag)
        result["source_commit"] = commit
        result["source_manifest"] = staged
        _atomic_json(destination / "preflight.json", result)
        summary = build_preflight_summary(result)
        _atomic_json(destination / "summary.json", summary)
        print(json.dumps(summary, sort_keys=True))
        return int(summary["status"].startswith("INCOMPLETE_"))
    if arguments.mode == "acquire":
        result = run_acquisition_mode(
            run_dir=destination,
            source_commit=commit,
            resolved_revision=arguments.resolved_revision,
            acquire=lambda: acquire_model(
                run_tag,
                arguments.resolved_revision,
            ),
        )
        result["source_commit"] = commit
        result["source_manifest"] = staged
        _atomic_json(destination / "acquisition.json", result)
        if result.get("model_manifest") is not None:
            _atomic_json(
                destination / "model_manifest.json",
                result["model_manifest"],
            )
        if result.get("classification") == "INCOMPLETE":
            write_incomplete_manifest(
                destination,
                source_commit=commit,
                model_revision=arguments.resolved_revision,
                failure_kind=result["failure_kind"],
                failure_detail=result["failure_detail"],
            )
        print(json.dumps(result, sort_keys=True))
        return int(result.get("classification") == "INCOMPLETE")
    if policy["launches_process"]:
        try:
            result = execute_remote_gate(
                mode=arguments.mode,
                run_tag=run_tag,
                resolved_revision=arguments.resolved_revision,
                repo_root=repo_root,
                source_commit=commit,
                staged_source=staged,
                smoke_run_tag=arguments.smoke_run_tag,
            )
        except Exception as exc:
            manifest = preserve_failed_execution(
                destination,
                source_commit=commit,
                model_revision=arguments.resolved_revision,
                failure_kind="INCOMPLETE_RUNNER_FAILURE",
                failure_detail=str(exc),
                download=lambda: download_artifacts(run_tag, destination),
            )
            print(json.dumps(manifest, sort_keys=True))
            return 1
        print(json.dumps(result, sort_keys=True))
        return int(
            result.get("classification")
            not in {"SMOKE_PASS", "GO", "NO_GO"}
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
