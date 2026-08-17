from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
REMOTE_TARGET = "sitian@10.232.195.203"
CONTROL_SOCKET = "/tmp/ssh-sitian-10.232.195.203"
KRB5CCNAME = "FILE:/Users/bytedance/krb5cc_sitian"
REMOTE_PYTHON = (
    "/data00/home/sitian/sitian-workspace01/tllm/env/bin/python"
)
REMOTE_REPO = (
    "/data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge"
)
REMOTE_RUN_ROOT = (
    "/data00/home/sitian/sitian-workspace01/tllm/"
    "spec-verify-cuda-graph-runs"
)
CHECKPOINT_CANDIDATES = (
    "/data00/home/sitian/sitian-workspace01/.ms_cache/"
    "Qwen/Qwen3-0.6B",
    "/data00/home/sitian/sitian-workspace01/.ms_cache/"
    "Qwen/Qwen3-0___6B",
)
SOURCE_TOOL_FILES = (
    "run_spec_verify_cuda_graph_gate_remote.py",
    "spec_verify_cuda_graph_smoke.py",
    "verify_spec_verify_cuda_graph_gate.py",
)
MVP_CONTEXT_LENGTH = 4096
MVP_BATCH_SIZES = (1, 4)
MVP_QUERY_LENGTHS = (1, 3)
MVP_PAGE_TABLE_WIDTHS = (1, 2)


def build_ssh_command(remote_arguments) -> list[str]:
    remote_command = shlex.join(
        [str(value) for value in remote_arguments]
    )
    return [
        "ssh",
        "-S",
        CONTROL_SOCKET,
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=20",
        REMOTE_TARGET,
        remote_command,
    ]


def build_scp_command(
    local_path: Path,
    remote_path: str,
) -> list[str]:
    return [
        "scp",
        "-o",
        f"ControlPath={CONTROL_SOCKET}",
        "-o",
        "BatchMode=yes",
        str(local_path),
        f"{REMOTE_TARGET}:{remote_path}",
    ]


def build_scp_download_command(
    remote_path: str,
    local_path: Path,
) -> list[str]:
    return [
        "scp",
        "-o",
        f"ControlPath={CONTROL_SOCKET}",
        "-o",
        "BatchMode=yes",
        f"{REMOTE_TARGET}:{remote_path}",
        str(local_path),
    ]


def _preflight_script() -> str:
    return "\n".join([
        "import json,pathlib,subprocess,sys,torch",
        f"remote_python={REMOTE_PYTHON!r}",
        f"remote_repo={REMOTE_REPO!r}",
        f"checkpoint_paths={CHECKPOINT_CANDIDATES!r}",
        "gpu_query=subprocess.run([",
        " 'nvidia-smi','--query-gpu=index,uuid',",
        " '--format=csv,noheader,nounits'],",
        " capture_output=True,text=True,check=False)",
        "process_query=subprocess.run([",
        " 'nvidia-smi','--query-compute-apps=gpu_uuid',",
        " '--format=csv,noheader,nounits'],",
        " capture_output=True,text=True,check=False)",
        "gpu_uuids={}",
        "if gpu_query.returncode == 0:",
        " for line in gpu_query.stdout.splitlines():",
        "  fields=[field.strip() for field in line.split(',',1)]",
        "  if len(fields)==2 and fields[0].isdigit():",
        "   gpu_uuids[int(fields[0])]=fields[1]",
        "active_uuids=set()",
        "if process_query.returncode == 0:",
        " active_uuids={line.strip() for line in "
        "process_query.stdout.splitlines() if line.strip()}",
        "devices=[]",
        "if torch.cuda.is_available():",
        " for index in range(torch.cuda.device_count()):",
        "  devices.append({",
        "   'index':index,",
        "   'name':torch.cuda.get_device_name(index),",
        "   'compute_capability':list("
        "torch.cuda.get_device_capability(index)),",
        "   'uuid':gpu_uuids.get(index),",
        "  })",
        "idle=[]",
        "if gpu_query.returncode == 0 and process_query.returncode == 0:",
        " idle=[row['index'] for row in devices "
        "if row.get('uuid') and row['uuid'] not in active_uuids]",
        "payload={",
        " 'python_exists':pathlib.Path(remote_python).is_file(),",
        " 'repo_exists':pathlib.Path(remote_repo).is_dir(),",
        " 'cuda_available':torch.cuda.is_available(),",
        " 'torch_version':torch.__version__,",
        " 'cuda_version':torch.version.cuda,",
        " 'device_count':torch.cuda.device_count(),",
        " 'devices':devices,",
        " 'idle_gpu_indices':idle,",
        " 'checkpoint_candidates':[",
        "  {'path':path,'exists':pathlib.Path(path).is_dir()}",
        "  for path in checkpoint_paths],",
        " 'nvidia_smi_gpu_query_ok':gpu_query.returncode == 0,",
        " 'nvidia_smi_process_query_ok':"
        "process_query.returncode == 0,",
        "}",
        "print(json.dumps(payload,sort_keys=True,separators=(',',':')))",
    ])


def build_remote_preflight_command() -> list[str]:
    return [REMOTE_PYTHON, "-c", _preflight_script()]


def classify_preflight_payload(payload) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("preflight payload must be a dictionary")
    if payload.get("python_exists") is not True:
        raise ValueError("remote python is unavailable")
    if payload.get("repo_exists") is not True:
        raise ValueError("remote repo is unavailable")
    if payload.get("cuda_available") is not True:
        raise ValueError("remote CUDA is unavailable")

    idle_gpu_indices = payload.get("idle_gpu_indices")
    if not isinstance(idle_gpu_indices, list) or not idle_gpu_indices:
        raise ValueError("no idle GPU is available")
    if any(
        not isinstance(index, int) or index < 0
        for index in idle_gpu_indices
    ):
        raise ValueError("idle GPU inventory is invalid")

    candidates = payload.get("checkpoint_candidates")
    if not isinstance(candidates, list):
        raise ValueError("checkpoint inventory is invalid")
    checkpoint = next(
        (
            candidate.get("path")
            for candidate in candidates
            if isinstance(candidate, dict)
            and candidate.get("exists") is True
            and isinstance(candidate.get("path"), str)
            and candidate["path"]
        ),
        None,
    )
    if checkpoint is None:
        raise ValueError("no checkpoint candidate is available")

    return {
        **payload,
        "status": "READY",
        "gpu_index": idle_gpu_indices[0],
        "checkpoint": checkpoint,
    }


def build_source_archive(
    repo_root: Path,
    archive_path: Path,
) -> Path:
    repo_root = Path(repo_root)
    archive_path = Path(archive_path)
    if archive_path.exists():
        raise ValueError("source archive already exists")
    tinyvllm_root = repo_root / "tinyvllm"
    if not tinyvllm_root.is_dir():
        raise ValueError("tinyvllm source tree is missing")
    tool_paths = [
        repo_root / "tools" / filename
        for filename in SOURCE_TOOL_FILES
    ]
    if any(not path.is_file() for path in tool_paths):
        raise ValueError("source tool file is missing")

    def filter_member(member):
        parts = Path(member.name).parts
        if (
            "__pycache__" in parts
            or member.name.endswith((".pyc", ".pyo"))
            or member.issym()
            or member.islnk()
        ):
            return None
        return member

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w") as archive:
        archive.add(
            tinyvllm_root,
            arcname="source/tinyvllm",
            recursive=True,
            filter=filter_member,
        )
        for tool_path in tool_paths:
            archive.add(
                tool_path,
                arcname=f"source/tools/{tool_path.name}",
                recursive=False,
                filter=filter_member,
            )
    return archive_path


def build_remote_smoke_command(
    *,
    remote_source_root: str,
    checkpoint: str,
    remote_artifact: str,
    context_length: int,
    batch_sizes: tuple[int, ...],
    query_lengths: tuple[int, ...],
    page_table_widths: tuple[int, ...],
    gpu_index: int,
    measure_performance: bool = False,
) -> list[str]:
    command = [
        "env",
        f"CUDA_VISIBLE_DEVICES={gpu_index}",
        f"PYTHONPATH={remote_source_root}",
        REMOTE_PYTHON,
        f"{remote_source_root}/tools/spec_verify_cuda_graph_smoke.py",
        "--model",
        checkpoint,
        "--output-json",
        remote_artifact,
    ]
    if measure_performance:
        command.append("--measure-performance")
    command.extend([
        "--context-length",
        str(context_length),
        "--batch-sizes",
        *(str(value) for value in batch_sizes),
        "--query-lengths",
        *(str(value) for value in query_lengths),
        "--page-table-widths",
        *(str(value) for value in page_table_widths),
    ])
    return command


def normalize_gate_configuration(
    *,
    context_length: int,
    batch_sizes: tuple[int, ...],
    query_lengths: tuple[int, ...],
    page_table_widths: tuple[int, ...],
) -> dict:
    configuration = {
        "context_length": int(context_length),
        "batch_sizes": tuple(batch_sizes),
        "query_lengths": tuple(query_lengths),
        "page_table_widths": tuple(page_table_widths),
    }
    expected = {
        "context_length": MVP_CONTEXT_LENGTH,
        "batch_sizes": MVP_BATCH_SIZES,
        "query_lengths": MVP_QUERY_LENGTHS,
        "page_table_widths": MVP_PAGE_TABLE_WIDTHS,
    }
    for field, expected_value in expected.items():
        if configuration[field] != expected_value:
            raise ValueError(
                f"{field} must equal {expected_value!r} "
                "for the MVP correctness gate"
            )
    return configuration


def _run(command, **kwargs):
    environment = dict(os.environ)
    environment["KRB5CCNAME"] = KRB5CCNAME
    return subprocess.run(
        command,
        check=False,
        env=environment,
        **kwargs,
    )


def _require_success(result, context: str):
    if result.returncode != 0:
        detail = result.stderr or result.stdout or ""
        if isinstance(detail, bytes):
            detail = detail.decode("utf-8", errors="replace")
        raise RuntimeError(
            f"{context} failed: {str(detail).strip()}"
        )
    return result


def query_preflight_payload() -> dict:
    result = _run(
        build_ssh_command(build_remote_preflight_command()),
        text=True,
        capture_output=True,
    )
    _require_success(result, "remote preflight")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise ValueError("remote preflight JSON is invalid") from error
    if not isinstance(payload, dict):
        raise ValueError("remote preflight payload is invalid")
    return payload


def build_preflight_record(payload: dict) -> dict:
    base = {
        "schema_version": 1,
        "payload": payload,
        "source_upload_started": False,
        "cuda_gate_started": False,
    }
    try:
        classified = classify_preflight_payload(payload)
    except ValueError as error:
        return {
            **base,
            "status": "BLOCKED",
            "error": str(error),
        }
    return {
        **base,
        "status": "READY",
        "gpu_index": classified["gpu_index"],
        "checkpoint": classified["checkpoint"],
    }


def _write_json_atomic(path: Path, payload: dict) -> None:
    path = Path(path)
    if path.exists():
        raise ValueError("output artifact already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    partial.replace(path)


def execute_preflight_authority(
    *,
    output_path: Path,
    payload_executor=query_preflight_payload,
) -> dict:
    record = build_preflight_record(payload_executor())
    _write_json_atomic(output_path, record)
    return record


def preflight_exit_code(record: dict) -> int:
    status = record.get("status") if isinstance(record, dict) else None
    if status == "READY":
        return 0
    if status == "BLOCKED":
        return 2
    raise ValueError("preflight status is invalid")


def execute_preflight() -> dict:
    return classify_preflight_payload(query_preflight_payload())


def _new_run_tag() -> str:
    return (
        "spec-verify-cuda-graph-"
        f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
    )


def upload_source(*, output_path: Path) -> dict:
    run_tag = _new_run_tag()
    remote_run_root = f"{REMOTE_RUN_ROOT}/{run_tag}"
    remote_archive = f"{remote_run_root}/source.tar"
    remote_source_root = f"{remote_run_root}/source"
    remote_artifact = f"{remote_run_root}/artifact.json"
    with tempfile.TemporaryDirectory(
        prefix="spec-verify-cuda-graph-"
    ) as temporary:
        archive_path = Path(temporary) / "source.tar"
        build_source_archive(ROOT, archive_path)
        mkdir_result = _run(
            build_ssh_command([
                "mkdir",
                "-p",
                remote_run_root,
            ]),
            text=True,
            capture_output=True,
        )
        _require_success(mkdir_result, "remote run directory creation")
        upload_result = _run(
            build_scp_command(archive_path, remote_archive),
            text=True,
            capture_output=True,
        )
        _require_success(upload_result, "source archive upload")
        extract_result = _run(
            build_ssh_command([
                "tar",
                "-xf",
                remote_archive,
                "-C",
                remote_run_root,
            ]),
            text=True,
            capture_output=True,
        )
        _require_success(extract_result, "source archive extraction")
    return {
        "run_tag": run_tag,
        "remote_run_root": remote_run_root,
        "remote_source_root": remote_source_root,
        "remote_artifact": remote_artifact,
        "local_output": str(output_path),
    }


def execute_remote_smoke(
    *,
    preflight: dict,
    upload: dict,
    context_length: int,
    batch_sizes: tuple[int, ...],
    query_lengths: tuple[int, ...],
    page_table_widths: tuple[int, ...],
    measure_performance: bool = False,
) -> None:
    command = build_remote_smoke_command(
        remote_source_root=upload["remote_source_root"],
        checkpoint=preflight["checkpoint"],
        remote_artifact=upload["remote_artifact"],
        context_length=context_length,
        batch_sizes=batch_sizes,
        query_lengths=query_lengths,
        page_table_widths=page_table_widths,
        gpu_index=preflight["gpu_index"],
        measure_performance=measure_performance,
    )
    result = _run(
        build_ssh_command(command),
        text=True,
        capture_output=True,
    )
    _require_success(result, "remote CUDA smoke")


def download_remote_artifact(
    *,
    remote_artifact: str,
    output_path: Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = _run(
        build_scp_download_command(
            remote_artifact,
            output_path,
        ),
        text=True,
        capture_output=True,
    )
    _require_success(result, "artifact download")


def verify_local_artifact(*, output_path: Path) -> dict:
    result = _run(
        [
            sys.executable,
            str(
                ROOT
                / "tools"
                / "verify_spec_verify_cuda_graph_gate.py"
            ),
            str(output_path),
            "--repo-root",
            str(ROOT),
        ],
        text=True,
        capture_output=True,
    )
    _require_success(result, "local artifact verification")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise ValueError("local verifier JSON is invalid") from error
    return {"status": "PASS", **payload}


def execute_gate(
    *,
    output_path: Path,
    context_length: int,
    batch_sizes: tuple[int, ...],
    query_lengths: tuple[int, ...],
    page_table_widths: tuple[int, ...],
    measure_performance: bool = False,
    preflight_executor=execute_preflight,
    upload_executor=upload_source,
    smoke_executor=execute_remote_smoke,
    download_executor=download_remote_artifact,
    verify_executor=verify_local_artifact,
) -> dict:
    output_path = Path(output_path)
    if output_path.exists():
        raise ValueError("output artifact already exists")
    configuration = normalize_gate_configuration(
        context_length=context_length,
        batch_sizes=batch_sizes,
        query_lengths=query_lengths,
        page_table_widths=page_table_widths,
    )
    preflight = preflight_executor()
    upload = upload_executor(output_path=output_path)
    smoke_executor(
        preflight=preflight,
        upload=upload,
        **configuration,
        measure_performance=measure_performance,
    )
    download_executor(
        remote_artifact=upload["remote_artifact"],
        output_path=output_path,
    )
    verification = verify_executor(output_path=output_path)
    return {
        **verification,
        "preflight": preflight,
        "remote_run_root": upload.get("remote_run_root"),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
    )
    parser.add_argument(
        "--measure-performance",
        action="store_true",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=MVP_CONTEXT_LENGTH,
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=list(MVP_BATCH_SIZES),
    )
    parser.add_argument(
        "--query-lengths",
        type=int,
        nargs="+",
        default=list(MVP_QUERY_LENGTHS),
    )
    parser.add_argument(
        "--page-table-widths",
        type=int,
        nargs="+",
        default=list(MVP_PAGE_TABLE_WIDTHS),
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.preflight_only:
        record = execute_preflight_authority(
            output_path=args.output_json,
        )
        print(json.dumps(record, indent=2, sort_keys=True))
        return preflight_exit_code(record)
    result = execute_gate(
        output_path=args.output_json,
        context_length=args.context_length,
        batch_sizes=tuple(args.batch_sizes),
        query_lengths=tuple(args.query_lengths),
        page_table_widths=tuple(args.page_table_widths),
        measure_performance=args.measure_performance,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
