#!/usr/bin/env python3
"""Run the persistent-decode ceiling qualification on a clean A100."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import tempfile
import time

from tools import profile_lease_sealed_persistent_decode_ceiling as profile
from tools import run_staged_inference_benchmark_remote as base
from tools.verify_lease_sealed_persistent_decode_ceiling import (
    verify_artifact_directory,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
REMOTE_HOST = "sitian@10.232.195.203"
REMOTE_PYTHON = "/data00/home/sitian/tllm/env/bin/python"
MODEL_PATH = "/data00/home/sitian/.ms_cache/Qwen/Qwen3-0___6B"
KRB5_CACHE = "FILE:/Users/bytedance/krb5cc_sitian"
APPROVED_ROOT = (
    "/data00/home/sitian/tinyllmforge-workspaces/"
    "command-timeline-20260818"
)
TASK_REMOTE_ROOT = APPROVED_ROOT + "/persistent-decode-ceiling"
LOCAL_ARTIFACT_ROOT = (
    REPO_ROOT / "artifacts" / "lease_sealed_persistent_decode"
)
MINIMUM_KERBEROS_LIFETIME_SECONDS = 5_400
NSYS_PATH = "/usr/local/bin/nsys"
CONTEXT_LENGTHS = (256, 2048, 8192)
GENERATED_TOKENS = 128
COMPACT_FILES = frozenset({
    "source_manifest.json",
    "runtime_manifest.json",
    "gpu_admission.json",
    "workload_manifest.json",
    "timing_rows.jsonl",
    "structural_rows.jsonl",
    "timing_summary.json",
    "trace_inventory.json",
    "kernel_rows.jsonl",
    "segment_rows.jsonl",
    "ceiling.json",
    "manifest.json",
})

validate_kerberos = base.validate_kerberos
require_pushed_head = base.require_pushed_head
require_remote_destinations_absent = base.require_remote_destinations_absent


def remote_paths(run_tag: str) -> dict[str, str]:
    tag = base.validate_run_tag(run_tag)
    paths = {
        "staging": f"{TASK_REMOTE_ROOT}/staging/{tag}",
        "primary": f"{TASK_REMOTE_ROOT}/runs/{tag}",
        "controller": (
            f"{TASK_REMOTE_ROOT}/controller-verification/{tag}"
        ),
    }
    if any(
        not path.startswith(APPROVED_ROOT + "/")
        for path in paths.values()
    ):
        raise ValueError("remote path escaped approved root")
    return paths


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
    if not isinstance(rows, list):
        raise ValueError("GPU inventory must be a list")
    clean = []
    indices = set()
    uuids = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("GPU row must be an object")
        index = row.get("index")
        uuid = row.get("uuid")
        name = row.get("name")
        memory = row.get("memory_used_mib")
        utilization = row.get("utilization_percent")
        processes = row.get("compute_processes")
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or not isinstance(uuid, str)
            or not uuid
            or not isinstance(name, str)
            or isinstance(memory, bool)
            or not isinstance(memory, int)
            or memory < 0
            or isinstance(utilization, bool)
            or not isinstance(utilization, int)
            or utilization < 0
            or not isinstance(processes, list)
        ):
            raise ValueError("GPU row is invalid")
        if index in indices or uuid in uuids:
            raise ValueError("GPU inventory contains duplicate identities")
        indices.add(index)
        uuids.add(uuid)
        if (
            "A100" in name
            and memory == 0
            and utilization == 0
            and not processes
        ):
            clean.append(row)
    return clean


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
            minimum_lifetime_seconds=MINIMUM_KERBEROS_LIFETIME_SECONDS
        )
        rows = base.query_remote_gpu_rows()
        clean = strict_clean_a100s(rows)
        if clean:
            return rows, clean[0]
        if time.monotonic() >= deadline:
            raise TimeoutError("no strict-clean A100 became available")
        time.sleep(poll_interval_seconds)


def validate_selected_gpu_still_clean(selected: dict) -> dict:
    observed = {
        row["index"]: row for row in base.query_remote_gpu_rows()
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
    result = subprocess.run(
        [
            "git",
            "archive",
            "--format=tar",
            "--prefix=source/",
            source_commit,
            "--",
            "tinyvllm",
            "tools",
            *(
                relative
                for relative in profile.SOURCE_FILES
                if relative.startswith("docs/")
            ),
        ],
        cwd=Path(repo_root),
        capture_output=True,
        check=False,
    )
    base._require_success(result, "build committed source archive")
    if not isinstance(result.stdout, bytes) or not result.stdout:
        raise ValueError("committed source archive is empty")
    return result.stdout


def _run_remote(command: str):
    return base._run_remote(command)


def _run_remote_with_input(command: str, payload: bytes):
    return base._run_remote_with_input(command, payload)


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
        "      or not path.parts or path.parts[0] != 'source'",
        "      or member.issym() or member.islnk()):",
        "   raise ValueError('unsafe source archive member')",
        " bundle.extractall(staging)",
    ))
    result = _run_remote_with_input(
        "python3 -c " + shlex.quote(script),
        archive,
    )
    base._require_success(result, "upload committed source archive")
    return staging + "/source"


def _runtime_environment(
    *,
    source_dir: str,
    output_dir: str,
    gpu_index: int,
) -> list[str]:
    runtime = output_dir + "/runtime"
    return [
        f"CUDA_VISIBLE_DEVICES={gpu_index}",
        f"PYTHONPATH={source_dir}",
        "PYTHONNOUSERSITE=1",
        "PYTHONDONTWRITEBYTECODE=1",
        f"TMPDIR={runtime}/scratch",
        f"TMP={runtime}/scratch",
        f"TEMP={runtime}/scratch",
        f"PYTHONPYCACHEPREFIX={runtime}/pycache",
        f"XDG_CACHE_HOME={runtime}/xdg-cache",
        f"HF_HOME={runtime}/hf-cache",
        f"TORCH_EXTENSIONS_DIR={runtime}/torch-extensions",
    ]


def _identity_arguments(output_dir: str) -> str:
    script = (
        "import json,pathlib;"
        f"p=pathlib.Path({output_dir!r});"
        "s=json.loads((p/'source_manifest.json').read_text());"
        "r=json.loads((p/'runtime_manifest.json').read_text());"
        "w=json.loads((p/'workload_manifest.json').read_text());"
        "import hashlib;"
        "h=lambda x:hashlib.sha256(json.dumps(x,sort_keys=True,"
        "separators=(',',':')).encode()).hexdigest();"
        "print(s['source_tree_sha256'],h(r),h(w))"
    )
    return (
        "read SOURCE_TREE_SHA RUNTIME_SHA WORKLOAD_SHA <<EOF\n"
        + f"$({shlex.quote(REMOTE_PYTHON)} -c {shlex.quote(script)})\n"
        + "EOF\n"
    )


def build_nsys_command(
    *,
    source_dir: str,
    output_dir: str,
    run_tag: str,
    source_commit: str,
    gpu_index: int,
    prompt_tokens: int,
) -> list[str]:
    paths = (source_dir, output_dir)
    if any(
        not isinstance(path, str)
        or not path.startswith(TASK_REMOTE_ROOT + "/")
        for path in paths
    ):
        raise ValueError("remote execution path is invalid")
    base.validate_run_tag(run_tag)
    validate_source_commit(source_commit, pushed_head=source_commit)
    if prompt_tokens not in CONTEXT_LENGTHS:
        raise ValueError("prompt token count is invalid")
    prefix = output_dir + f"/nsys/context-{prompt_tokens}"
    inner = (
        _identity_arguments(output_dir)
        + "exec "
        + " ".join((
            shlex.quote(REMOTE_PYTHON),
            "-m tools.profile_lease_sealed_persistent_decode_ceiling",
            "--mode structural",
            "--model",
            shlex.quote(MODEL_PATH),
            "--run-tag",
            shlex.quote(run_tag),
            "--source-commit",
            source_commit,
            "--source-tree-sha256 \"$SOURCE_TREE_SHA\"",
            "--runtime-identity-sha256 \"$RUNTIME_SHA\"",
            "--workload-identity-sha256 \"$WORKLOAD_SHA\"",
            "--prompt-tokens",
            str(prompt_tokens),
            "--generated-tokens",
            str(GENERATED_TOKENS),
            "--output",
            shlex.quote(output_dir + "/structural_rows.jsonl"),
        ))
    )
    environment = _runtime_environment(
        source_dir=source_dir,
        output_dir=output_dir,
        gpu_index=gpu_index,
    )
    return [
        NSYS_PATH,
        "profile",
        "--trace=cuda,nvtx,osrt",
        "--force-overwrite=false",
        "--env-var=" + ",".join(environment),
        "--output",
        prefix,
        "bash",
        "-lc",
        inner,
    ]


def _setup_command(
    *,
    source: str,
    primary: str,
    controller: str,
    run_tag: str,
    source_commit: str,
    gpu: dict,
) -> str:
    script = "\n".join((
        "import hashlib,json,pathlib,platform,subprocess",
        "import torch",
        "from tools import profile_lease_sealed_persistent_decode_ceiling as p",
        f"root=pathlib.Path({primary!r})",
        f"control=pathlib.Path({controller!r})",
        "root.mkdir(parents=True,exist_ok=False)",
        "control.mkdir(parents=True,exist_ok=False)",
        "(root/'nsys').mkdir()",
        "(root/'runtime'/'scratch').mkdir(parents=True)",
        f"source=p.build_source_manifest(repo_root=pathlib.Path({source!r}),"
        f"source_commit={source_commit!r},run_tag={run_tag!r})",
        "canonical=lambda x:json.dumps(x,sort_keys=True,"
        "separators=(',',':')).encode()",
        "source['source_tree_sha256']=hashlib.sha256("
        "canonical(source['source_sha256'])).hexdigest()",
        "runtime={",
        " 'schema_version':'lease-sealed-persistent-decode.runtime.v1',",
        " 'python':platform.python_version(),",
        " 'pytorch':str(torch.__version__),",
        " 'cuda':str(torch.version.cuda),",
        " 'gpu':str(torch.cuda.get_device_name(0)),",
        " 'nsight_systems':subprocess.run("
        f"[{NSYS_PATH!r},'--version'],text=True,capture_output=True,"
        "check=True).stdout.strip(),",
        f" 'model_path':{MODEL_PATH!r},",
        " 'checkpoint_inventory_sha256':hashlib.sha256("
        f"str(sorted((x.name,x.stat().st_size) for x in pathlib.Path({MODEL_PATH!r}).iterdir())).encode()).hexdigest(),",
        " 'feature_configuration':{'policy':'decode_burst_k8',"
        "'tensor_parallel_size':1},",
        "}",
        "workload={",
        " 'schema_version':'lease-sealed-persistent-decode.workload.v1',",
        " 'contexts':[256,2048,8192],",
        " 'generated_tokens':128,'repetitions':5,",
        " 'temperature':0.0,'ignore_eos':True,'max_num_seqs':1,",
        "}",
        f"gpu={json.dumps(gpu, sort_keys=True)!r}",
        "gpu=json.loads(gpu)",
        "admission={",
        " 'schema_version':'lease-sealed-persistent-decode.gpu-admission.v1',",
        " 'strict_clean':True,'gpu_index':gpu['index'],",
        " 'gpu_uuid':gpu['uuid'],'compute_process_count':0,",
        " 'memory_used_mib':0,'utilization_gpu_pct':0,",
        "}",
        "for name,payload in (('source_manifest.json',source),"
        "('runtime_manifest.json',runtime),"
        "('workload_manifest.json',workload),"
        "('gpu_admission.json',admission)):",
        " (root/name).write_text(json.dumps(payload,sort_keys=True)+'\\n')",
        "(control/'gpu_admission_second.json').write_text("
        "json.dumps(admission,sort_keys=True)+'\\n')",
    ))
    return (
        f"cd {shlex.quote(source)} && "
        + "env "
        + " ".join(
            shlex.quote(value)
            for value in _runtime_environment(
                source_dir=source,
                output_dir=primary,
                gpu_index=gpu["index"],
            )
        )
        + " "
        + shlex.quote(REMOTE_PYTHON)
        + " -c "
        + shlex.quote(script)
    )


def _timing_command(
    *,
    source: str,
    primary: str,
    run_tag: str,
    source_commit: str,
    gpu_index: int,
) -> str:
    producer = (
        _identity_arguments(primary)
        + "for repetition in 0 1 2 3 4; do "
        + "for context in 256 2048 8192; do "
        + " ".join((
            shlex.quote(REMOTE_PYTHON),
            "-m tools.profile_lease_sealed_persistent_decode_ceiling",
            "--mode timing",
            "--model",
            shlex.quote(MODEL_PATH),
            "--run-tag",
            shlex.quote(run_tag),
            "--source-commit",
            source_commit,
            "--source-tree-sha256 \"$SOURCE_TREE_SHA\"",
            "--runtime-identity-sha256 \"$RUNTIME_SHA\"",
            "--workload-identity-sha256 \"$WORKLOAD_SHA\"",
            "--repetition \"$repetition\"",
            "--prompt-tokens \"$context\"",
            "--generated-tokens",
            str(GENERATED_TOKENS),
            "--output",
            shlex.quote(primary + "/timing_rows.jsonl"),
        ))
        + " || exit $?; done; done"
    )
    return (
        f"cd {shlex.quote(source)} && env "
        + " ".join(
            shlex.quote(value)
            for value in _runtime_environment(
                source_dir=source,
                output_dir=primary,
                gpu_index=gpu_index,
            )
        )
        + " bash -lc "
        + shlex.quote(producer)
    )


def build_worker_plan(
    *,
    paths: dict[str, str],
    run_tag: str,
    source_commit: str,
    gpu: dict,
) -> dict:
    if set(paths) != {"staging", "primary", "controller"}:
        raise ValueError("remote path inventory is invalid")
    source = paths["staging"] + "/source"
    commands: list[str | list[str]] = [
        _setup_command(
            source=source,
            primary=paths["primary"],
            controller=paths["controller"],
            run_tag=run_tag,
            source_commit=source_commit,
            gpu=gpu,
        ),
        _timing_command(
            source=source,
            primary=paths["primary"],
            run_tag=run_tag,
            source_commit=source_commit,
            gpu_index=gpu["index"],
        ),
    ]
    for context in CONTEXT_LENGTHS:
        commands.append(build_nsys_command(
            source_dir=source,
            output_dir=paths["primary"],
            run_tag=run_tag,
            source_commit=source_commit,
            gpu_index=gpu["index"],
            prompt_tokens=context,
        ))
    for context in CONTEXT_LENGTHS:
        prefix = paths["primary"] + f"/nsys/context-{context}"
        commands.append(
            f"{NSYS_PATH} export --type=sqlite --force-overwrite=true "
            f"--output {shlex.quote(prefix + '.sqlite')} "
            f"{shlex.quote(prefix + '.nsys-rep')}"
        )
    finalize = (
        _identity_arguments(paths["primary"])
        + " ".join((
            shlex.quote(REMOTE_PYTHON),
            "-m tools.profile_lease_sealed_persistent_decode_ceiling",
            "--mode finalize",
            "--timing-path",
            shlex.quote(paths["primary"] + "/timing_rows.jsonl"),
            "--structural-path",
            shlex.quote(paths["primary"] + "/structural_rows.jsonl"),
            "--output-dir",
            shlex.quote(paths["primary"]),
            *(
                item
                for context in CONTEXT_LENGTHS
                for item in (
                    "--trace",
                    shlex.quote(
                        f"{context}="
                        f"{paths['primary']}/nsys/context-{context}.sqlite"
                    ),
                )
            ),
        ))
    )
    commands.append(
        f"cd {shlex.quote(source)} && "
        + "env "
        + " ".join(
            shlex.quote(value)
            for value in _runtime_environment(
                source_dir=source,
                output_dir=paths["primary"],
                gpu_index=gpu["index"],
            )
        )
        + " bash -lc "
        + shlex.quote(finalize)
    )
    verification_bundle = paths["controller"] + "/remote-verify-bundle"
    copy_script = "\n".join((
        "import json,pathlib,shutil",
        f"source=pathlib.Path({paths['primary']!r})",
        f"target=pathlib.Path({verification_bundle!r})",
        "target.mkdir(parents=False,exist_ok=False)",
        "manifest=json.loads((source/'manifest.json').read_text())",
        "for row in manifest['artifacts']:",
        " relative=pathlib.PurePosixPath(row['path'])",
        " destination=target.joinpath(*relative.parts)",
        " destination.parent.mkdir(parents=True,exist_ok=True)",
        " shutil.copyfile(source.joinpath(*relative.parts),destination)",
        "shutil.copyfile(source/'manifest.json',target/'manifest.json')",
    ))
    commands.append(
        f"cd {shlex.quote(source)} && "
        f"{shlex.quote(REMOTE_PYTHON)} -c {shlex.quote(copy_script)} && "
        f"{shlex.quote(REMOTE_PYTHON)} -m "
        "tools.verify_lease_sealed_persistent_decode_ceiling "
        f"{shlex.quote(verification_bundle)} > "
        f"{shlex.quote(paths['controller'] + '/remote_verification.json')}"
    )
    return {
        "schema_version":
            "lease-sealed-persistent-decode.worker-plan.v1",
        "run_tag": run_tag,
        "source_commit": source_commit,
        "gpu": gpu,
        "paths": paths,
        "commands": commands,
    }


def run_worker_plan(plan: dict) -> dict:
    commands = plan.get("commands")
    if not isinstance(commands, list) or not commands:
        raise ValueError("worker plan commands are invalid")
    exitcodes = []
    for index, command in enumerate(commands):
        remote_command = (
            shlex.join(command)
            if isinstance(command, list)
            else command
        )
        result = _run_remote(remote_command)
        exitcodes.append(result.returncode)
        base._require_success(result, f"remote worker stage {index}")
    return {"status": "COMPLETE", "exitcodes": exitcodes}


def is_compact_artifact(relative: str) -> bool:
    if not isinstance(relative, str) or not relative:
        raise ValueError("artifact path is invalid")
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("artifact path is invalid")
    return path.as_posix() in COMPACT_FILES


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
            payload = base.download_chunk(
                remote_root + "/" + record["path"],
                offset=chunk["offset"],
                length=chunk["length"],
                expected_sha256=chunk["sha256"],
            )
            handle.write(payload)
            digest.update(payload)
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
    partial.mkdir(parents=True)
    try:
        for record in inventory:
            target = partial / record["path"]
            _download_inventory_record(
                remote_root=remote_path,
                record=record,
                target=target,
            )
        partial.replace(destination)
    except Exception:
        raise
    return destination


def verify_local_bundle(path: Path) -> dict:
    return verify_artifact_directory(path)


def stream_and_verify_raw_traces(
    *,
    remote_path: str,
    compact_path: Path,
    temporary_parent: Path,
) -> dict:
    inventory = {
        row["path"]: row
        for row in base.fetch_remote_inventory(remote_path)
    }
    trace_inventory = json.loads(
        (Path(compact_path) / "trace_inventory.json").read_text(
            encoding="utf-8"
        )
    )
    raw_traces = trace_inventory.get("raw_traces")
    if not isinstance(raw_traces, list):
        raise ValueError("raw trace inventory is invalid")
    seen = set()
    with tempfile.TemporaryDirectory(
        prefix="persistent-decode-raw-",
        dir=Path(temporary_parent),
    ) as temporary:
        temporary_root = Path(temporary)
        for row in raw_traces:
            context = row.get("context_length")
            if context not in CONTEXT_LENGTHS or context in seen:
                raise ValueError("raw trace context inventory is invalid")
            seen.add(context)
            relative = f"nsys/context-{context}.sqlite"
            expected_remote = remote_path + "/" + relative
            record = inventory.get(relative)
            if (
                row.get("remote_path") != expected_remote
                or record is None
                or row.get("byte_length") != record.get("size_bytes")
                or row.get("sha256") != record.get("sha256")
            ):
                raise ValueError("raw trace inventory mismatch")
            target = temporary_root / f"context-{context}.sqlite"
            _download_inventory_record(
                remote_root=remote_path,
                record=record,
                target=target,
            )
            target.unlink()
    if seen != set(CONTEXT_LENGTHS):
        raise ValueError("raw trace inventory is incomplete")
    verification = dict(verify_local_bundle(Path(compact_path)))
    verification["raw_trace_count"] = len(seen)
    return verification


def verify_remote_bundle_with_streamed_traces(
    *,
    remote_path: str,
    temporary_parent: Path,
) -> dict:
    parent = Path(temporary_parent)
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="persistent-decode-verify-",
        dir=parent,
    ) as temporary:
        compact = download_compact_bundle(
            remote_path=remote_path,
            local_parent=Path(temporary),
        )
        return stream_and_verify_raw_traces(
            remote_path=remote_path,
            compact_path=compact,
            temporary_parent=Path(temporary),
        )


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_controller_receipts(
    *,
    destination: Path,
    plan: dict,
    gpu_inventory: list[dict],
    selected_gpu: dict,
    worker: dict,
    verification: dict,
) -> Path:
    root = Path(destination)
    if not root.is_dir():
        raise ValueError("local artifact destination is missing")
    missing = sorted(
        relative
        for relative in COMPACT_FILES
        if not (root / relative).is_file()
    )
    if missing:
        raise ValueError(f"compact artifact is missing: {missing[0]}")
    controller = root / "controller"
    controller.mkdir(exist_ok=False)
    _write_json(controller / "plan.json", plan)
    _write_json(
        controller / "launch_admission.json",
        {
            "schema_version":
                "lease-sealed-persistent-decode.launch-admission.v1",
            "gpu_inventory": gpu_inventory,
            "selected_gpu": selected_gpu,
            "worker": worker,
        },
    )
    _write_json(
        controller / "download_manifest.json",
        {
            "schema_version":
                "lease-sealed-persistent-decode.download-manifest.v1",
            "artifacts": [
                {
                    "path": relative,
                    "byte_length": (root / relative).stat().st_size,
                    "sha256": hashlib.sha256(
                        (root / relative).read_bytes()
                    ).hexdigest(),
                }
                for relative in sorted(COMPACT_FILES)
            ],
        },
    )
    _write_json(
        controller / "local-verification.json",
        verification,
    )
    return controller


def run_controller(args) -> dict:
    if args.host != REMOTE_HOST or args.model != MODEL_PATH:
        raise ValueError("remote target is not approved")
    local_destination = (
        Path(args.local_artifact_root) / base.validate_run_tag(args.run_tag)
    )
    if local_destination.exists() or local_destination.is_symlink():
        raise ValueError("local artifact destination already exists")
    pushed_head = require_pushed_head(REPO_ROOT)
    source_commit = validate_source_commit(
        pushed_head if args.source_commit is None else args.source_commit,
        pushed_head=pushed_head,
    )
    validate_kerberos(
        minimum_lifetime_seconds=(
            args.gpu_timeout_seconds
            + MINIMUM_KERBEROS_LIFETIME_SECONDS
        )
    )
    paths = remote_paths(args.run_tag)
    require_remote_destinations_absent(paths)
    gpu_rows, selected = wait_for_clean_a100(
        timeout_seconds=args.gpu_timeout_seconds,
        poll_interval_seconds=args.poll_interval_seconds,
    )
    archive = committed_source_archive(REPO_ROOT, source_commit)
    source = upload_source_archive(
        staging=paths["staging"],
        archive=archive,
    )
    validate_kerberos(
        minimum_lifetime_seconds=MINIMUM_KERBEROS_LIFETIME_SECONDS
    )
    selected = validate_selected_gpu_still_clean(selected)
    plan = build_worker_plan(
        paths=paths,
        run_tag=args.run_tag,
        source_commit=source_commit,
        gpu=selected,
    )
    plan["source"] = source
    receipt = run_worker_plan(plan)
    verification = verify_remote_bundle_with_streamed_traces(
        remote_path=paths["primary"],
        temporary_parent=Path(args.local_artifact_root),
    )
    destination = download_compact_bundle(
        remote_path=paths["primary"],
        local_parent=Path(args.local_artifact_root),
    )
    controller = write_controller_receipts(
        destination=destination,
        plan=plan,
        gpu_inventory=gpu_rows,
        selected_gpu=selected,
        worker=receipt,
        verification=verification,
    )
    return {
        "status": "COMPLETE",
        "run_tag": args.run_tag,
        "source_commit": source_commit,
        "gpu_inventory": gpu_rows,
        "selected_gpu": selected,
        "remote_paths": paths,
        "local_destination": os.fspath(destination),
        "local_controller": os.fspath(controller),
        "worker": receipt,
        "verification": verification,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=REMOTE_HOST)
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--run-tag", required=True)
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
        default=30,
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    os.environ["KRB5CCNAME"] = KRB5_CACHE
    result = run_controller(parse_args(argv))
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
