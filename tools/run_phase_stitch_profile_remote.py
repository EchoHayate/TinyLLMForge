#!/usr/bin/env python3
"""Source-bound clean-A100 controller for the Phase-Stitch profile gate."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tarfile
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
if os.fspath(REPO_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPO_ROOT))

from tools import phase_stitch_profile_contract as contract
from tools import run_staged_inference_benchmark_remote as base
from tools import run_zero_temperature_greedy_fast_path_remote as download
from tools.phase_stitch_profile_verify import verify_bundle


REMOTE_HOST = "sitian@10.232.195.203"
REMOTE_PYTHON = "/data00/home/sitian/tllm/env/bin/python"
MODEL_PATH = "/data00/home/sitian/.ms_cache/Qwen/Qwen3-0___6B"
KRB5_CACHE = "FILE:/Users/bytedance/krb5cc_sitian"
APPROVED_ROOT = (
    "/data00/home/sitian/tinyllmforge-workspaces/"
    "command-timeline-20260818"
)
TASK_REMOTE_ROOT = APPROVED_ROOT + "/phase-stitch-profile"
LOCAL_ARTIFACT_ROOT = REPO_ROOT / "artifacts/phase_stitch_profile"
MINIMUM_KERBEROS_LIFETIME_SECONDS = 5_400
TOOL_SOURCE_FILES = (
    "tools/phase_stitch_profile_contract.py",
    "tools/phase_stitch_profile_worker.py",
    "tools/phase_stitch_profile_gate.py",
    "tools/phase_stitch_profile_verify.py",
    "tools/test_phase_stitch_profile.py",
    "tools/test_phase_stitch_profile_benchmark.py",
    "tools/test_llm_engine_exact_greedy_decode_burst.py",
)

validate_kerberos = base.validate_kerberos
require_pushed_head = base.require_pushed_head
require_remote_destinations_absent = base.require_remote_destinations_absent
require_success = base._require_success


def remote_paths(run_tag):
    tag = base.validate_run_tag(run_tag)
    paths = {
        "staging": f"{TASK_REMOTE_ROOT}/staging/{tag}",
        "primary": f"{TASK_REMOTE_ROOT}/runs/{tag}",
        "controller": (
            f"{TASK_REMOTE_ROOT}/controller-verification/{tag}"
        ),
    }
    for path in paths.values():
        if not path.startswith(TASK_REMOTE_ROOT + "/"):
            raise ValueError("remote path is outside approved task root")
    return paths


def ensure_local_destination_absent(local_root, run_tag):
    tag = base.validate_run_tag(run_tag)
    destination = Path(local_root) / tag
    if destination.exists() or destination.is_symlink():
        raise ValueError("local run tag already exists")
    return destination


def strict_clean_a100s(rows):
    return [
        row
        for row in base.strict_clean_gpus(rows)
        if "A100" in row["name"]
    ]


def wait_for_clean_a100(*, timeout_seconds, poll_interval_seconds):
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
        rows = base.query_remote_gpu_rows()
        clean = strict_clean_a100s(rows)
        if clean:
            return rows, clean[0]
        if time.monotonic() >= deadline:
            raise TimeoutError("no strict-clean A100 became available")
        time.sleep(poll_interval_seconds)


def validate_selected_gpu_still_clean(selected):
    latest = {
        row["index"]: row
        for row in base.query_remote_gpu_rows()
    }.get(selected.get("index"))
    if (
        latest is None
        or latest.get("uuid") != selected.get("uuid")
        or strict_clean_a100s([latest]) != [latest]
    ):
        raise RuntimeError("selected A100 is no longer strict-clean")
    return latest


def _tracked_source_files(repo_root):
    result = subprocess.run(
        [
            "git",
            "ls-tree",
            "-r",
            "--name-only",
            "HEAD",
            "--",
            "tinyvllm",
        ],
        cwd=Path(repo_root),
        text=True,
        capture_output=True,
        check=False,
    )
    require_success(result, "enumerate tracked runtime source")
    runtime_files = tuple(
        line
        for line in result.stdout.splitlines()
        if line and (Path(repo_root) / line).is_file()
    )
    inventory = tuple(sorted({
        *runtime_files,
        *TOOL_SOURCE_FILES,
    }))
    if not runtime_files:
        raise ValueError("tracked runtime source inventory is empty")
    for relative_path in inventory:
        if (
            (
                not relative_path.startswith("tinyvllm/")
                and relative_path not in TOOL_SOURCE_FILES
            )
        ):
            raise ValueError("source archive inventory is invalid")
        result = subprocess.run(
            ["git", "cat-file", "-e", f"HEAD:{relative_path}"],
            cwd=Path(repo_root),
            text=True,
            capture_output=True,
            check=False,
        )
        require_success(
            result,
            f"resolve committed source file {relative_path}",
        )
    return inventory


def _committed_file_bytes(repo_root, relative_path):
    result = subprocess.run(
        ["git", "show", f"HEAD:{relative_path}"],
        cwd=Path(repo_root),
        capture_output=True,
        check=False,
    )
    require_success(
        result,
        f"read committed source file {relative_path}",
    )
    return result.stdout


def build_source_archive(repo_root):
    root = Path(repo_root)
    inventory = _tracked_source_files(root)
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:") as archive:
        root_info = tarfile.TarInfo("source")
        root_info.type = tarfile.DIRTYPE
        root_info.mode = 0o700
        archive.addfile(root_info)
        for relative_path in inventory:
            content = _committed_file_bytes(root, relative_path)
            info = tarfile.TarInfo(f"source/{relative_path}")
            info.mode = 0o600
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    return payload.getvalue(), inventory


def _run_remote(command, *, text=True):
    return base._run_remote(command, text=text)


def _run_remote_with_input(command, payload):
    return base._run_remote_with_input(command, payload)


def upload_source_archive(*, staging, archive):
    if not staging.startswith(TASK_REMOTE_ROOT + "/staging/"):
        raise ValueError("remote staging path is invalid")
    script = "\n".join((
        "import pathlib,sys,tarfile",
        f"staging=pathlib.Path({staging!r})",
        "staging.parent.mkdir(parents=True,exist_ok=True)",
        "staging.mkdir(parents=False,exist_ok=False)",
        "archive_path=staging/'source.tar'",
        "with archive_path.open('xb') as handle:",
        " handle.write(sys.stdin.buffer.read())",
        "with tarfile.open(archive_path,'r:') as bundle:",
        " members=bundle.getmembers()",
        " for member in members:",
        "  path=pathlib.PurePosixPath(member.name)",
        "  if (path.is_absolute() or '..' in path.parts",
        "      or member.issym() or member.islnk()):",
        "   raise ValueError('unsafe source archive member')",
        " bundle.extractall(staging)",
    ))
    result = _run_remote_with_input(
        f"python3 -c {shlex.quote(script)}",
        archive,
    )
    require_success(result, "source archive upload")
    return staging + "/source"


def remote_runtime_prelude(*, source, gpu_index):
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


def run_remote_preflight(*, source, gpu_index):
    smoke = "; ".join((
        "from tools import phase_stitch_profile_contract as contract",
        "from tools import phase_stitch_profile_gate",
        "from tools import phase_stitch_profile_verify",
        "from tools import phase_stitch_profile_worker",
        "assert len(contract.build_case_matrix()) == 8",
        "assert len(contract.expected_case_ids()) == 8",
    ))
    command = (
        "set -eu; "
        f"cd {shlex.quote(source)}; "
        + remote_runtime_prelude(
            source=source,
            gpu_index=gpu_index,
        )
        + f"test -x {shlex.quote(REMOTE_PYTHON)}; "
        + f"test -d {shlex.quote(MODEL_PATH)}; "
        + f"{shlex.quote(REMOTE_PYTHON)} -m compileall -q "
        + "tinyvllm tools; "
        + f"{shlex.quote(REMOTE_PYTHON)} -c {shlex.quote(smoke)}"
    )
    require_success(
        _run_remote(command),
        "remote source-bound preflight",
    )


def _source_hashes(repo_root):
    result = {}
    for relative_path in contract.SOURCE_FILES:
        result[relative_path] = hashlib.sha256(
            _committed_file_bytes(repo_root, relative_path)
        ).hexdigest()
    return result


def create_remote_run(
    *,
    paths,
    run_tag,
    source_commit,
    source_hashes,
    gpu_inventory,
    selected_gpu,
):
    payload = {
        "run_manifest": {
            "schema_version": contract.RUN_SCHEMA_VERSION,
            "run_tag": run_tag,
            "source_base_commit": source_commit,
            "source_files": source_hashes,
            "model": MODEL_PATH,
            "python": REMOTE_PYTHON,
            "cuda_visible_devices": str(selected_gpu["index"]),
            "clean_gpu_admission": True,
            "gpu_inventory": [selected_gpu],
            "observed_gpu_inventory": gpu_inventory,
            "case_order": list(contract.expected_case_ids()),
            "contract_sha256": contract.contract_sha256(),
        },
        "cases": contract.build_case_matrix(),
    }
    script = "\n".join((
        "import json,pathlib,sys",
        "payload=json.load(sys.stdin)",
        f"primary=pathlib.Path({paths['primary']!r})",
        f"controller=pathlib.Path({paths['controller']!r})",
        "primary.parent.mkdir(parents=True,exist_ok=True)",
        "controller.parent.mkdir(parents=True,exist_ok=True)",
        "primary.mkdir(parents=False,exist_ok=False)",
        "controller.mkdir(parents=False,exist_ok=False)",
        "(primary/'cases').mkdir()",
        "(controller/'case-exitcodes').mkdir()",
        "(controller/'case-specs').mkdir()",
        "(primary/'run_manifest.json').write_text(",
        " json.dumps(payload['run_manifest'],sort_keys=True)+'\\n')",
        "for case in payload['cases']:",
        " path=controller/'case-specs'/(case['case_id']+'.json')",
        " path.write_text(",
        "  json.dumps(case,sort_keys=True)+'\\n')",
    ))
    result = base._run_remote_with_input(
        f"python3 -c {shlex.quote(script)}",
        json.dumps(payload, allow_nan=False).encode("utf-8"),
    )
    require_success(result, "create immutable remote run")


def remote_execution_commands(
    *,
    source,
    primary,
    controller,
    model,
    gpu_index,
):
    prelude = remote_runtime_prelude(
        source=source,
        gpu_index=gpu_index,
    )
    commands = []
    for case in contract.build_case_matrix():
        case_id = case["case_id"]
        worker = " ".join((
            shlex.quote(REMOTE_PYTHON),
            "-m tools.phase_stitch_profile_worker",
            "--spec",
            shlex.quote(
                f"{controller}/case-specs/{case_id}.json"
            ),
            "--model",
            shlex.quote(model),
            "--output-dir",
            shlex.quote(f"{primary}/cases/{case_id}"),
        ))
        receipt = shlex.quote(
            f"{controller}/case-exitcodes/{case_id}"
        )
        commands.append(
            "set +e; "
            f"cd {shlex.quote(source)}; "
            + prelude
            + worker
            + f"; code=$?; printf '%s\\n' \"$code\" > {receipt}; "
            + "exit \"$code\""
        )
    for module, receipt_name, stdout_name in (
        (
            "tools.phase_stitch_profile_gate",
            "producer_exitcode",
            "producer.json",
        ),
        (
            "tools.phase_stitch_profile_verify",
            "verifier_exitcode",
            "verifier.json",
        ),
    ):
        command = " ".join((
            shlex.quote(REMOTE_PYTHON),
            "-m",
            module,
            "--run-dir",
            shlex.quote(primary),
        ))
        commands.append(
            "set +e; "
            f"cd {shlex.quote(source)}; "
            + prelude
            + command
            + f" > {shlex.quote(controller + '/' + stdout_name)}; "
            + "code=$?; "
            + f"printf '%s\\n' \"$code\" > "
            + shlex.quote(controller + "/" + receipt_name)
            + '; exit "$code"'
        )
    return tuple(commands)


def run_remote_pipeline(
    *,
    source,
    primary,
    controller,
    model,
    gpu_index,
):
    commands = remote_execution_commands(
        source=source,
        primary=primary,
        controller=controller,
        model=model,
        gpu_index=gpu_index,
    )
    exitcodes = []
    for index, command in enumerate(commands):
        result = _run_remote(command)
        exitcodes.append(result.returncode)
        require_success(result, f"remote pipeline stage {index}")
    return {
        "case_exitcodes": exitcodes[:len(contract.build_case_matrix())],
        "producer_exitcode": exitcodes[-2],
        "verifier_exitcode": exitcodes[-1],
    }


def download_terminal_bundle(*, paths, local_destination):
    destination = Path(local_destination)
    download.download_remote_tree_preserving_partial(
        paths["primary"],
        destination,
    )
    download.download_remote_tree_preserving_partial(
        paths["controller"],
        destination / "controller",
    )
    for receipt in (
        "producer_exitcode",
        "verifier_exitcode",
    ):
        value = (
            destination / "controller" / receipt
        ).read_text(encoding="utf-8").strip()
        if value != "0":
            raise ValueError(f"{receipt} is not zero")
    verification = verify_bundle(destination)
    remote_verification = json.loads(
        (destination / "controller/verifier.json").read_text(
            encoding="utf-8"
        )
    )
    if verification != remote_verification:
        raise ValueError("remote and local verifier receipts disagree")
    return verification


def run_controller(args):
    if args.host != REMOTE_HOST:
        raise ValueError("remote host is not approved")
    if args.model != MODEL_PATH:
        raise ValueError("model path is not approved")
    if os.environ.get("KRB5CCNAME", KRB5_CACHE) != KRB5_CACHE:
        raise ValueError("Kerberos cache path is not approved")
    local_destination = ensure_local_destination_absent(
        Path(args.local_artifact_root),
        args.run_tag,
    )
    pushed_head = require_pushed_head(REPO_ROOT)
    source_commit = (
        pushed_head
        if args.source_commit is None
        else args.source_commit
    )
    if (
        re.fullmatch(r"[0-9a-f]{40}", source_commit) is None
        or source_commit != pushed_head
    ):
        raise ValueError("source commit does not match pushed head")
    validate_kerberos(
        minimum_lifetime_seconds=MINIMUM_KERBEROS_LIFETIME_SECONDS
    )
    paths = remote_paths(args.run_tag)
    require_remote_destinations_absent(paths)
    gpu_inventory, selected_gpu = wait_for_clean_a100(
        timeout_seconds=args.gpu_wait_timeout_seconds,
        poll_interval_seconds=args.gpu_poll_interval_seconds,
    )
    if require_pushed_head(REPO_ROOT) != source_commit:
        raise ValueError("pushed head changed while waiting for GPU")
    archive, _archive_inventory = build_source_archive(REPO_ROOT)
    source = upload_source_archive(
        staging=paths["staging"],
        archive=archive,
    )
    validate_kerberos(
        minimum_lifetime_seconds=MINIMUM_KERBEROS_LIFETIME_SECONDS
    )
    run_remote_preflight(
        source=source,
        gpu_index=selected_gpu["index"],
    )
    selected_gpu = validate_selected_gpu_still_clean(selected_gpu)
    create_remote_run(
        paths=paths,
        run_tag=args.run_tag,
        source_commit=source_commit,
        source_hashes=_source_hashes(REPO_ROOT),
        gpu_inventory=gpu_inventory,
        selected_gpu=selected_gpu,
    )
    receipts = run_remote_pipeline(
        source=source,
        primary=paths["primary"],
        controller=paths["controller"],
        model=args.model,
        gpu_index=selected_gpu["index"],
    )
    verification = download_terminal_bundle(
        paths=paths,
        local_destination=local_destination,
    )
    return {
        "status": "COMPLETE",
        "run_tag": args.run_tag,
        "source_commit": source_commit,
        "remote_paths": paths,
        "selected_gpu": selected_gpu,
        "local_destination": os.fspath(local_destination),
        **receipts,
        **verification,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=REMOTE_HOST)
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--source-commit", default=None)
    parser.add_argument(
        "--local-artifact-root",
        default=os.fspath(LOCAL_ARTIFACT_ROOT),
    )
    parser.add_argument(
        "--gpu-wait-timeout-seconds",
        type=int,
        default=28_800,
    )
    parser.add_argument(
        "--gpu-poll-interval-seconds",
        type=int,
        default=60,
    )
    return parser.parse_args(argv)


def main(argv=None):
    os.environ["KRB5CCNAME"] = KRB5_CACHE
    result = run_controller(parse_args(argv))
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
