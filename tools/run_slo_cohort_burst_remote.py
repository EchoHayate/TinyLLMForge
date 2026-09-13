#!/usr/bin/env python3
"""Remote orchestration contract for SLO-aware cohort-burst gates."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
from typing import Mapping

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import run_staged_inference_benchmark_remote as base
from tools import profile_slo_cohort_burst_ceiling as profile
from tools import slo_cohort_burst_ceiling as ceiling


APPROVED_ROOT = base.APPROVED_ROOT
TASK_REMOTE_ROOT = APPROVED_ROOT + "/slo-cohort-burst"
REMOTE_HOST = base.REMOTE_HOST
REMOTE_PYTHON = base.REMOTE_PYTHON
MODEL_PATH = base.MODEL_PATHS["qwen3-0.6b"]
DEFAULT_KERBEROS_CACHE = base.KRB5CCNAME
MINIMUM_KERBEROS_LIFETIME_SECONDS = 1_800
REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_ARTIFACT_ROOT = (
    REPO_ROOT / "artifacts" / "slo_cohort_burst_ceiling"
)
SOURCE_FILES = (
    "tinyvllm",
    "tools/slo_cohort_burst_ceiling.py",
    "tools/profile_slo_cohort_burst_ceiling.py",
    "tools/run_slo_cohort_burst_remote.py",
)
REQUIRED_TERMINAL_FILES = frozenset({
    "raw_rows.jsonl",
    "cost_table.json",
    "ceiling_summary.json",
    "source_manifest.json",
    "remote_verify.json",
})
COMPACT_FILES = REQUIRED_TERMINAL_FILES | frozenset({
    "runner.log",
})
DOWNLOAD_RETRIES = 3

validate_kerberos = base.validate_kerberos
require_pushed_head = base.require_pushed_head


def build_remote_paths(run_tag: str) -> dict[str, str]:
    tag = base.validate_run_tag(run_tag)
    paths = {
        "staging": f"{TASK_REMOTE_ROOT}/staging/{tag}",
        "primary": f"{TASK_REMOTE_ROOT}/runs/{tag}",
        "controller": (
            f"{TASK_REMOTE_ROOT}/controller-verification/{tag}"
        ),
    }
    if any(
        not path.startswith(TASK_REMOTE_ROOT + "/")
        for path in paths.values()
    ):
        raise ValueError("remote path is outside approved task root")
    return paths


def distributed_port(run_tag: str) -> int:
    tag = base.validate_run_tag(run_tag)
    digest = hashlib.sha256(tag.encode("utf-8")).digest()
    return 20_000 + int.from_bytes(digest[:4], "big") % 30_000


def build_remote_runtime_prelude(
    *,
    source: str,
    gpu_index: int,
    dist_port: int,
) -> str:
    if (
        not isinstance(source, str)
        or not source.startswith(TASK_REMOTE_ROOT + "/staging/")
        or not source.endswith("/source")
    ):
        raise ValueError("remote source path is invalid")
    if (
        isinstance(gpu_index, bool)
        or not isinstance(gpu_index, int)
        or gpu_index < 0
    ):
        raise ValueError("GPU index is invalid")
    if (
        isinstance(dist_port, bool)
        or not isinstance(dist_port, int)
        or not 20_000 <= dist_port < 50_000
    ):
        raise ValueError("distributed port is invalid")
    runtime = source.rsplit("/", 1)[0] + "/runtime"
    directories = {
        "TMPDIR": runtime + "/tmp",
        "TMP": runtime + "/tmp",
        "TEMP": runtime + "/tmp",
        "PYTHONPYCACHEPREFIX": runtime + "/pycache",
        "XDG_CACHE_HOME": runtime + "/xdg",
        "HF_HOME": runtime + "/hf-home",
        "TORCH_EXTENSIONS_DIR": runtime + "/torch-extensions",
    }
    exports = {
        **directories,
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_VISIBLE_DEVICES": str(gpu_index),
        "TINYVLLM_DIST_PORT": str(dist_port),
        "MASTER_PORT": str(dist_port),
        "PYTHONPATH": source,
    }
    return (
        "umask 077; mkdir -p "
        + " ".join(
            shlex.quote(path)
            for path in sorted(set(directories.values()))
        )
        + "; "
        + " ".join(
            f"export {name}={shlex.quote(value)};"
            for name, value in exports.items()
        )
        + " "
    )


def validate_source_commit(requested: str, *, pushed_head: str) -> str:
    if (
        not isinstance(requested, str)
        or re.fullmatch(r"[0-9a-f]{40}", requested) is None
        or not isinstance(pushed_head, str)
        or re.fullmatch(r"[0-9a-f]{40}", pushed_head) is None
    ):
        raise ValueError("source commit is invalid")
    if requested != pushed_head:
        raise ValueError("source commit does not match pushed head")
    return requested


def strict_clean_a100s(rows: list[dict]) -> list[dict]:
    return [
        row
        for row in base.strict_clean_gpus(rows)
        if "A100" in row["name"]
    ]


def wait_for_clean_a100(
    *,
    timeout_seconds: int,
    poll_interval_seconds: int,
) -> tuple[list[dict], dict]:
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, int)
        or timeout_seconds <= 0
        or isinstance(poll_interval_seconds, bool)
        or not isinstance(poll_interval_seconds, int)
        or poll_interval_seconds <= 0
    ):
        raise ValueError("GPU polling policy is invalid")
    deadline = time.monotonic() + timeout_seconds
    while True:
        validate_kerberos(
            minimum_lifetime_seconds=(
                MINIMUM_KERBEROS_LIFETIME_SECONDS
            )
        )
        try:
            rows = base.query_remote_gpu_rows()
        except RuntimeError as error:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "remote GPU inventory remained unavailable"
                ) from error
            time.sleep(poll_interval_seconds)
            continue
        clean = strict_clean_a100s(rows)
        if clean:
            return rows, clean[0]
        if time.monotonic() >= deadline:
            raise TimeoutError("no strict-clean A100 became available")
        time.sleep(poll_interval_seconds)


def validate_selected_gpu_still_clean(selected: dict) -> dict:
    observed = {
        row["index"]: row
        for row in base.query_remote_gpu_rows()
    }.get(selected.get("index"))
    if (
        observed is None
        or observed.get("uuid") != selected.get("uuid")
        or strict_clean_a100s([observed]) != [observed]
    ):
        raise RuntimeError("selected A100 is no longer strict-clean")
    return observed


def committed_source_archive(
    repo_root: Path,
    source_commit: str,
) -> bytes:
    if re.fullmatch(r"[0-9a-f]{40}", source_commit or "") is None:
        raise ValueError("source commit is invalid")
    result = subprocess.run(
        [
            "git",
            "archive",
            "--format=tar",
            "--prefix=source/",
            source_commit,
            "--",
            *SOURCE_FILES,
        ],
        cwd=Path(repo_root),
        capture_output=True,
        check=False,
    )
    base._require_success(result, "build committed source archive")
    if not isinstance(result.stdout, bytes) or not result.stdout:
        raise ValueError("committed source archive is empty")
    with tarfile.open(fileobj=io.BytesIO(result.stdout), mode="r:") as bundle:
        members = bundle.getmembers()
    if not members:
        raise ValueError("committed source archive is empty")
    for member in members:
        path = PurePosixPath(member.name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or not path.parts
            or path.parts[0] != "source"
            or member.issym()
            or member.islnk()
        ):
            raise ValueError("committed source archive is unsafe")
    return result.stdout


def upload_source_archive(*, staging: str, archive: bytes) -> str:
    prefix = TASK_REMOTE_ROOT + "/staging/"
    if (
        not isinstance(staging, str)
        or not staging.startswith(prefix)
        or base.validate_run_tag(staging[len(prefix):])
        != staging[len(prefix):]
    ):
        raise ValueError("remote staging path is invalid")
    if not isinstance(archive, bytes) or not archive:
        raise ValueError("source archive is empty")
    script = "\n".join((
        "import pathlib,sys,tarfile",
        f"staging=pathlib.Path({staging!r})",
        "staging.parent.mkdir(parents=True,exist_ok=True)",
        "staging.mkdir(parents=False,exist_ok=False)",
        "archive_path=staging/'source.tar'",
        "archive_path.write_bytes(sys.stdin.buffer.read())",
        "with tarfile.open(archive_path,'r:') as bundle:",
        " members=bundle.getmembers()",
        " if not members:",
        "  raise ValueError('empty source archive')",
        " for member in members:",
        "  path=pathlib.PurePosixPath(member.name)",
        "  if (path.is_absolute() or '..' in path.parts",
        "      or not path.parts or path.parts[0]!='source'",
        "      or member.issym() or member.islnk()):",
        "   raise ValueError('unsafe source archive member')",
        " bundle.extractall(staging)",
    ))
    result = base._run_remote_with_input(
        "python3 -c " + shlex.quote(script),
        archive,
    )
    base._require_success(result, "upload committed source archive")
    return staging + "/source"


def _runtime_environment(
    *,
    source: str,
    primary: str,
    gpu_index: int,
    run_tag: str,
) -> str:
    prelude = build_remote_runtime_prelude(
        source=source,
        gpu_index=gpu_index,
        dist_port=distributed_port(run_tag),
    )
    return f"cd {shlex.quote(source)} && {prelude}"


def build_worker_plan(
    *,
    paths: dict[str, str],
    run_tag: str,
    source_commit: str,
    gpu: dict,
) -> dict[str, object]:
    if paths != build_remote_paths(run_tag):
        raise ValueError("remote path inventory is invalid")
    validate_source_commit(
        source_commit,
        pushed_head=source_commit,
    )
    clean = strict_clean_a100s([gpu])
    if clean != [gpu]:
        raise ValueError("selected GPU is not strict-clean A100")
    source = paths["staging"] + "/source"
    prefix = _runtime_environment(
        source=source,
        primary=paths["primary"],
        gpu_index=gpu["index"],
        run_tag=run_tag,
    )
    run_command = (
        prefix
        + shlex.quote(REMOTE_PYTHON)
        + " -m tools.profile_slo_cohort_burst_ceiling"
        + " --mode run"
        + " --model "
        + shlex.quote(MODEL_PATH)
        + " --run-tag "
        + shlex.quote(run_tag)
        + " --source-commit "
        + source_commit
        + " --output-dir "
        + shlex.quote(paths["primary"])
        + " > "
        + shlex.quote(
            paths["staging"] + "/runtime/runner.log"
        )
        + " 2>&1"
        + " && mv "
        + shlex.quote(
            paths["staging"] + "/runtime/runner.log"
        )
        + " "
        + shlex.quote(paths["primary"] + "/runner.log")
    )
    verify_command = (
        prefix
        + shlex.quote(REMOTE_PYTHON)
        + " -m tools.profile_slo_cohort_burst_ceiling"
        + " --mode verify"
        + " --artifact-dir "
        + shlex.quote(paths["primary"])
        + " --output "
        + shlex.quote(paths["controller"] + "/remote_verify.json")
    )
    seal_script = "\n".join((
        "import hashlib,json,os,pathlib",
        f"primary=pathlib.Path({paths['primary']!r})",
        f"controller=pathlib.Path({paths['controller']!r})",
        f"required={sorted(REQUIRED_TERMINAL_FILES)!r}",
        "primary_verify=json.loads(",
        " (primary/'remote_verify.json').read_text())",
        "controller_verify=json.loads(",
        " (controller/'remote_verify.json').read_text())",
        "if primary_verify != controller_verify:",
        " raise ValueError('remote verifier disagreement')",
        "hashes={}",
        "for name in required:",
        " path=primary/name",
        " if not path.is_file() or path.is_symlink():",
        "  raise ValueError('terminal artifact missing: '+name)",
        " hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()",
        "receipt={",
        " 'schema_version':'slo-cohort-burst.remote-resume.v1',",
        " 'status':'COMPLETE',",
        f" 'run_tag':{run_tag!r},",
        f" 'source_commit':{source_commit!r},",
        f" 'remote_paths':{paths!r},",
        " 'artifact_sha256':hashes,",
        "}",
        "controller.mkdir(parents=True,exist_ok=True)",
        "destination=controller/'resume.json'",
        "with destination.open('x',encoding='utf-8') as handle:",
        " json.dump(receipt,handle,sort_keys=True,separators=(',',':'))",
        " handle.write('\\n')",
        " handle.flush()",
        " os.fsync(handle.fileno())",
    ))
    seal_command = (
        shlex.quote(REMOTE_PYTHON)
        + " -c "
        + shlex.quote(seal_script)
    )
    return {
        "schema_version": "slo-cohort-burst.worker-plan.v1",
        "run_tag": run_tag,
        "source_commit": source_commit,
        "gpu": dict(gpu),
        "paths": dict(paths),
        "commands": [run_command, verify_command, seal_command],
    }


def run_worker_plan(plan: dict) -> dict[str, object]:
    commands = plan.get("commands")
    if not isinstance(commands, list) or not commands:
        raise ValueError("worker plan commands are invalid")
    exitcodes = []
    for index, command in enumerate(commands):
        if not isinstance(command, str) or not command:
            raise ValueError("worker command is invalid")
        result = base._run_remote(command)
        exitcodes.append(result.returncode)
        base._require_success(result, f"remote worker stage {index}")
    return {
        "status": "COMPLETE",
        "exitcodes": exitcodes,
    }


def is_compact_artifact(relative: str) -> bool:
    if not isinstance(relative, str) or not relative:
        raise ValueError("artifact path is invalid")
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("artifact path is invalid")
    return path.as_posix() in COMPACT_FILES


def validate_resume_receipt(
    receipt: object,
    *,
    run_tag: str,
    source_commit: str,
    paths: dict[str, str],
) -> dict[str, object]:
    tag = base.validate_run_tag(run_tag)
    validate_source_commit(
        source_commit,
        pushed_head=source_commit,
    )
    if not isinstance(receipt, dict):
        raise ValueError("resume receipt is invalid")
    if (
        receipt.get("schema_version")
        != "slo-cohort-burst.remote-resume.v1"
        or receipt.get("status") != "COMPLETE"
        or receipt.get("run_tag") != tag
        or receipt.get("source_commit") != source_commit
        or receipt.get("remote_paths") != paths
    ):
        raise ValueError("resume source identity is invalid")
    hashes = receipt.get("artifact_sha256")
    if (
        not isinstance(hashes, dict)
        or set(hashes) != set(REQUIRED_TERMINAL_FILES)
        or any(
            re.fullmatch(r"[0-9a-f]{64}", str(value)) is None
            for value in hashes.values()
        )
    ):
        raise ValueError("resume terminal hashes are invalid")
    return json.loads(json.dumps(receipt))


def probe_resume_receipt(
    *,
    paths: dict[str, str],
    run_tag: str,
    source_commit: str,
) -> dict[str, object] | None:
    if paths != build_remote_paths(run_tag):
        raise ValueError("remote path inventory is invalid")
    script = "\n".join((
        "import json,pathlib",
        f"paths={paths!r}",
        "present={name:pathlib.Path(path).exists()",
        " for name,path in paths.items()}",
        "if not any(present.values()):",
        " print(json.dumps({'state':'ABSENT'}))",
        "elif not all(present.values()):",
        " raise ValueError('partial immutable remote attempt exists')",
        "else:",
        " receipt_path=pathlib.Path(paths['controller'])/'resume.json'",
        " if not receipt_path.is_file():",
        "  raise ValueError('remote attempt is not resumable')",
        " receipt=json.loads(receipt_path.read_text())",
        " print(json.dumps({'state':'COMPLETE','receipt':receipt},",
        "  sort_keys=True,separators=(',',':')))",
    ))
    result = base._run_remote(
        "python3 -c " + shlex.quote(script)
    )
    base._require_success(result, "remote resume probe")
    try:
        payload = json.loads(result.stdout)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("remote resume probe is invalid") from error
    if payload == {"state": "ABSENT"}:
        return None
    if (
        not isinstance(payload, dict)
        or payload.get("state") != "COMPLETE"
    ):
        raise ValueError("remote resume probe is invalid")
    return validate_resume_receipt(
        payload.get("receipt"),
        run_tag=run_tag,
        source_commit=source_commit,
        paths=paths,
    )


def _download_inventory_record(
    *,
    remote_root: str,
    record: dict,
    target: Path,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with target.open("xb") as handle:
        for chunk in record["chunks"]:
            last_error = None
            for _attempt in range(DOWNLOAD_RETRIES):
                try:
                    payload = base.download_chunk(
                        remote_root + "/" + record["path"],
                        offset=chunk["offset"],
                        length=chunk["length"],
                        expected_sha256=chunk["sha256"],
                    )
                    break
                except (RuntimeError, ValueError) as error:
                    last_error = error
            else:
                raise RuntimeError(
                    "artifact chunk download failed: "
                    + record["path"]
                ) from last_error
            handle.write(payload)
            digest.update(payload)
        handle.flush()
        os.fsync(handle.fileno())
    if (
        target.stat().st_size != record["size_bytes"]
        or digest.hexdigest() != record["sha256"]
    ):
        raise ValueError("downloaded artifact digest mismatch")


def download_compact_bundle(
    *,
    remote_path: str,
    local_parent: Path,
) -> Path:
    prefix = TASK_REMOTE_ROOT + "/runs/"
    if (
        not isinstance(remote_path, str)
        or not remote_path.startswith(prefix)
    ):
        raise ValueError("remote artifact path is invalid")
    run_tag = remote_path[len(prefix):]
    base.validate_run_tag(run_tag)
    destination = Path(local_parent) / run_tag
    if destination.exists() or destination.is_symlink():
        raise ValueError("local artifact destination already exists")
    partial = destination.with_name(destination.name + ".partial")
    if partial.exists() or partial.is_symlink():
        raise ValueError("local partial destination already exists")
    inventory = [
        row
        for row in base.fetch_remote_inventory(remote_path)
        if is_compact_artifact(row["path"])
    ]
    names = {row["path"] for row in inventory}
    if not REQUIRED_TERMINAL_FILES.issubset(names):
        raise ValueError("remote terminal artifact inventory is incomplete")
    partial.mkdir(parents=True)
    try:
        for record in inventory:
            _download_inventory_record(
                remote_root=remote_path,
                record=record,
                target=partial / record["path"],
            )
        partial.replace(destination)
    except BaseException:
        if partial.is_dir() and not partial.is_symlink():
            shutil.rmtree(partial)
        raise
    return destination


def verify_local_bundle(path: Path) -> dict[str, object]:
    root = Path(path)
    rows = profile._load_jsonl(root / "raw_rows.jsonl")
    source_identity = profile._load_json(
        root / "source_manifest.json"
    )
    cost_table = profile._load_json(root / "cost_table.json")
    summary = profile._load_json(root / "ceiling_summary.json")
    remote_verification = profile._load_json(
        root / "remote_verify.json"
    )
    artifact = {
        "schema_version": ceiling.ARTIFACT_SCHEMA_VERSION,
        "source_identity": source_identity,
        "cost_rows": profile.build_optimistic_cost_rows(rows),
        "cost_table": cost_table,
        "ceiling_summary": summary,
    }
    local_verification = ceiling.verify_ceiling_artifact(artifact)
    if local_verification != remote_verification:
        raise ValueError("local and remote verifier disagree")
    return local_verification


def validate_download_against_resume(
    destination: Path,
    receipt: Mapping[str, object],
) -> None:
    hashes = receipt.get("artifact_sha256")
    if not isinstance(hashes, dict):
        raise ValueError("resume terminal hashes are invalid")
    for name in REQUIRED_TERMINAL_FILES:
        path = Path(destination) / name
        if (
            not path.is_file()
            or path.is_symlink()
            or hashlib.sha256(path.read_bytes()).hexdigest()
            != hashes.get(name)
        ):
            raise ValueError(
                f"downloaded terminal artifact mismatch: {name}"
            )


def _write_json(path: Path, payload: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.write_text(
        json.dumps(
            payload,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def write_local_controller_receipt(
    *,
    destination: Path,
    source_commit: str,
    paths: dict[str, str],
    resume_receipt: dict[str, object],
    worker: dict[str, object],
    verification: dict[str, object],
    gpu_inventory: list[dict] | None,
    selected_gpu: dict | None,
    resumed: bool,
) -> Path:
    controller = Path(destination) / "controller"
    controller.mkdir(exist_ok=False)
    receipt_path = controller / "receipt.json"
    _write_json(receipt_path, {
        "schema_version": "slo-cohort-burst.local-controller.v1",
        "status": "COMPLETE",
        "source_commit": source_commit,
        "remote_paths": paths,
        "resume_receipt": resume_receipt,
        "worker": worker,
        "verification": verification,
        "gpu_inventory": gpu_inventory,
        "selected_gpu": selected_gpu,
        "resumed": resumed,
    })
    return receipt_path


def run_controller(args) -> dict[str, object]:
    if args.stage != "ceiling":
        raise ValueError("unsupported remote stage")
    tag = base.validate_run_tag(args.tag)
    local_parent = Path(args.local_artifact_root)
    local_destination = local_parent / tag
    if (
        local_destination.exists()
        or local_destination.is_symlink()
    ):
        raise ValueError("local artifact destination already exists")
    pushed_head = require_pushed_head(REPO_ROOT)
    source_commit = validate_source_commit(
        pushed_head
        if args.source_commit is None
        else args.source_commit,
        pushed_head=pushed_head,
    )
    validate_kerberos(
        minimum_lifetime_seconds=(
            MINIMUM_KERBEROS_LIFETIME_SECONDS
        )
    )
    paths = build_remote_paths(tag)
    resume_receipt = probe_resume_receipt(
        paths=paths,
        run_tag=tag,
        source_commit=source_commit,
    )
    resumed = resume_receipt is not None
    gpu_inventory = None
    selected_gpu = None
    worker = {"status": "RESUMED"}
    if not resumed:
        archive = committed_source_archive(REPO_ROOT, source_commit)
        source = upload_source_archive(
            staging=paths["staging"],
            archive=archive,
        )
        gpu_inventory, selected_gpu = wait_for_clean_a100(
            timeout_seconds=args.gpu_timeout_seconds,
            poll_interval_seconds=args.poll_interval_seconds,
        )
        validate_kerberos(
            minimum_lifetime_seconds=(
                MINIMUM_KERBEROS_LIFETIME_SECONDS
            )
        )
        selected_gpu = validate_selected_gpu_still_clean(
            selected_gpu
        )
        plan = build_worker_plan(
            paths=paths,
            run_tag=tag,
            source_commit=source_commit,
            gpu=selected_gpu,
        )
        plan["source"] = source
        worker = run_worker_plan(plan)
        resume_receipt = probe_resume_receipt(
            paths=paths,
            run_tag=tag,
            source_commit=source_commit,
        )
        if resume_receipt is None:
            raise RuntimeError("remote worker did not seal its evidence")
    destination = download_compact_bundle(
        remote_path=paths["primary"],
        local_parent=local_parent,
    )
    validate_download_against_resume(destination, resume_receipt)
    verification = verify_local_bundle(destination)
    controller_receipt = write_local_controller_receipt(
        destination=destination,
        source_commit=source_commit,
        paths=paths,
        resume_receipt=resume_receipt,
        worker=worker,
        verification=verification,
        gpu_inventory=gpu_inventory,
        selected_gpu=selected_gpu,
        resumed=resumed,
    )
    return {
        "status": "COMPLETE",
        "run_tag": tag,
        "source_commit": source_commit,
        "remote_paths": paths,
        "local_destination": os.fspath(destination),
        "local_controller_receipt": os.fspath(controller_receipt),
        "resumed": resumed,
        "classification": verification["classification"],
        "verification": verification,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the SLO cohort-burst Stage-0 ceiling gate",
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=("ceiling",),
    )
    parser.add_argument("--tag", required=True)
    parser.add_argument("--source-commit")
    parser.add_argument(
        "--local-artifact-root",
        default=os.fspath(LOCAL_ARTIFACT_ROOT),
    )
    parser.add_argument(
        "--gpu-timeout-seconds",
        type=int,
        default=7_200,
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=int,
        default=15,
    )
    args = parser.parse_args(argv)
    base.validate_run_tag(args.tag)
    if (
        args.gpu_timeout_seconds <= 0
        or args.poll_interval_seconds <= 0
    ):
        parser.error("GPU polling values must be positive")
    return args


def main(argv=None) -> int:
    result = run_controller(parse_args(argv))
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
