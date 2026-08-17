from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tarfile
import tempfile


TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from autoregressive_draft_cuda_graph_contract import (
    canonical_json_bytes,
    canonical_json_sha256,
)
from verify_autoregressive_draft_cuda_graph_gate import (
    verify_gate_bundle,
)


REMOTE_TARGET = "sitian@10.232.195.203"
REMOTE_PYTHON = (
    "/data00/home/sitian/sitian-workspace01/tllm/env/bin/python"
)
SSH_CONTROL_PATH = "/tmp/ssh-sitian-10.232.195.203"
REMOTE_RUN_ROOT = (
    "/data00/home/sitian/sitian-workspace01/tllm/"
    "autoregressive-draft-cuda-graph-runs"
)
REMOTE_PACKAGE_ROOT = (
    "/dev/shm/sitian/tllm-qwen35-target-qwen3-draft-20260815/"
    "run_packages"
)
DEFAULT_TARGET_MODEL = (
    "/dev/shm/sitian/tllm-qwen35-target-qwen3-draft-20260815/"
    "target-qwen3-1.7b"
)
DEFAULT_DRAFT_MODEL = (
    "/dev/shm/sitian/tllm-qwen35-target-qwen3-draft-20260815/"
    "draft"
)
SOURCE_PATHS = (
    "tinyvllm/",
    "tools/autoregressive_draft_cuda_graph_contract.py",
    "tools/autoregressive_draft_cuda_graph_gate.py",
    "tools/autoregressive_draft_performance_gate.py",
    "tools/autoregressive_draft_performance_worker.py",
    "tools/autoregressive_draft_tp4_engine_gate.py",
    "tools/speculative_runtime_performance_gate.py",
    "tools/verify_autoregressive_draft_cuda_graph_gate.py",
    "tools/run_autoregressive_draft_cuda_graph_gate_remote.py",
)
MAX_IDLE_MEMORY_USED_MIB = 1024
MAX_IDLE_UTILIZATION_PERCENT = 5
RUN_TAG_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
GPU_PROCESS_MARKER = "__TINYLLM_GPU_PROCESSES__"


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_run_tag(value: object) -> str:
    text = str(value)
    if not RUN_TAG_PATTERN.fullmatch(text):
        raise ValueError("run tag must match [A-Za-z0-9_-]+")
    return text


def classify_gpu_preflight(rows) -> dict:
    if not isinstance(rows, list):
        raise ValueError("GPU preflight rows must be a list")
    idle_indices = []
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("GPU preflight row must be a mapping")
        index = row.get("index")
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index in seen
        ):
            raise ValueError("GPU preflight index is invalid")
        seen.add(index)
        memory = row.get("memory_used_mib")
        utilization = row.get("utilization_percent")
        processes = row.get("compute_process_count")
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in (memory, utilization, processes)
        ):
            raise ValueError("GPU preflight metrics are invalid")
        if (
            memory <= MAX_IDLE_MEMORY_USED_MIB
            and utilization <= MAX_IDLE_UTILIZATION_PERCENT
            and processes == 0
        ):
            idle_indices.append(index)
    idle_indices.sort()
    if len(idle_indices) < 4:
        return {
            "status": "INCONCLUSIVE_ENVIRONMENT",
            "gpu_indices": [],
            "reason": "fewer than four clean GPUs are available",
        }
    return {
        "status": "READY",
        "gpu_indices": idle_indices[:4],
    }


def build_ssh_command(remote_arguments) -> list[str]:
    return [
        "ssh",
        "-S",
        SSH_CONTROL_PATH,
        "-o",
        "BatchMode=yes",
        REMOTE_TARGET,
        shlex.join([str(value) for value in remote_arguments]),
    ]


def build_remote_gate_command(
    *,
    source_root: str,
    target_model: str,
    draft_model: str,
    output_path: str,
    gpu_indices,
    provenance_path: str | None = None,
    environment_path: str | None = None,
    pythonpath_extra: str | None = None,
) -> list[str]:
    indices = tuple(gpu_indices)
    if (
        len(indices) != 4
        or len(set(indices)) != 4
        or any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            for index in indices
        )
    ):
        raise ValueError("remote gate requires four GPU indices")
    pythonpath = source_root
    if pythonpath_extra:
        pythonpath = f"{pythonpath_extra}:{pythonpath}"
    command = [
        "env",
        "CUDA_VISIBLE_DEVICES="
        + ",".join(str(index) for index in indices),
        f"PYTHONPATH={pythonpath}",
        "TINYLLM_GATE_LIVE_ENVIRONMENT=1",
        REMOTE_PYTHON,
        f"{source_root}/tools/autoregressive_draft_cuda_graph_gate.py",
        "--target-model",
        target_model,
        "--draft-model",
        draft_model,
        "--out",
        output_path,
        "--tensor-parallel-size",
        "4",
        "--batch-size",
        "4",
        "--max-proposal-tokens",
        "4",
        "--prompt-tokens",
        "256",
        "--output-tokens",
        "16",
        "--warmup-pairs",
        "2",
        "--measured-pairs",
        "8",
    ]
    if provenance_path is not None:
        command.extend(["--provenance", provenance_path])
    if environment_path is not None:
        command.extend(["--environment", environment_path])
    return command


def source_file_hashes(repo_root: Path) -> dict[str, str]:
    root = Path(repo_root)
    hashes = {}
    for relative in SOURCE_PATHS:
        path = root / relative.rstrip("/")
        if not path.exists():
            raise ValueError(f"missing source path: {relative}")
        candidates = (
            [path]
            if path.is_file()
            else sorted(
                child
                for child in path.rglob("*")
                if child.is_file()
            )
        )
        for candidate in candidates:
            if (
                "__pycache__" in candidate.parts
                or candidate.suffix in (".pyc", ".pyo")
            ):
                continue
            name = candidate.relative_to(root).as_posix()
            hashes[name] = _sha256_path(candidate)
    if not hashes:
        raise ValueError("source archive inventory is empty")
    return dict(sorted(hashes.items()))


def build_source_archive(
    repo_root: Path,
    archive_path: Path,
) -> None:
    root = Path(repo_root)
    archive_path = Path(archive_path)
    with tarfile.open(archive_path, "w:") as archive:
        for relative in SOURCE_PATHS:
            path = root / relative.rstrip("/")
            if not path.exists():
                raise ValueError(
                    f"missing source path: {relative}"
                )
            archive.add(
                path,
                arcname=f"source/{relative.rstrip('/')}",
                recursive=True,
                filter=lambda info: (
                    None
                    if "__pycache__" in Path(info.name).parts
                    or info.name.endswith((".pyc", ".pyo"))
                    else info
                ),
            )


def build_source_bundle(
    *,
    repo_root: Path,
    bundle_root: Path,
    provenance: dict,
    environment: dict,
    command_runner=subprocess.run,
) -> dict:
    repo_root = Path(repo_root)
    bundle_root = Path(bundle_root)
    bundle_root.mkdir(parents=True, exist_ok=False)
    archive_path = bundle_root / "source.tar"
    build_source_archive(repo_root, archive_path)
    source_hashes = source_file_hashes(repo_root)
    source_tree_sha256 = canonical_json_sha256(source_hashes)
    commit = command_runner(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if commit.returncode != 0:
        raise RuntimeError("source commit resolution failed")
    source_commit = commit.stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("source commit must be 40 lowercase hex")
    patch = command_runner(
        ["git", "diff", "--binary", "HEAD", "--", *SOURCE_PATHS],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if patch.returncode != 0:
        raise RuntimeError("source patch generation failed")
    patch_path = bundle_root / "source.patch"
    patch_path.write_bytes(patch.stdout)
    complete_provenance = {
        **provenance,
        "source_commit": source_commit,
        "source_patch_sha256": _sha256_path(patch_path),
        "source_tree_sha256": source_tree_sha256,
    }
    (bundle_root / "provenance.json").write_bytes(
        canonical_json_bytes(complete_provenance)
    )
    (bundle_root / "environment.json").write_bytes(
        canonical_json_bytes(environment)
    )
    manifest_seed = {
        "schema_version": 1,
        "source_patch": "source.patch",
        "source_files": source_hashes,
    }
    (bundle_root / "source_manifest_seed.json").write_bytes(
        canonical_json_bytes(manifest_seed)
    )
    with tarfile.open(archive_path, "a:") as archive:
        for filename in (
            "source.patch",
            "provenance.json",
            "environment.json",
            "source_manifest_seed.json",
        ):
            archive.add(
                bundle_root / filename,
                arcname=filename,
            )
    return {
        "archive_path": archive_path,
        "provenance": complete_provenance,
        "environment": environment,
        "source_files": source_hashes,
    }


def _query_remote_gpu_rows(
    command_runner=subprocess.run,
) -> list[dict]:
    script = (
        "nvidia-smi "
        "--query-gpu=index,uuid,memory.used,utilization.gpu "
        "--format=csv,noheader,nounits && "
        f"printf '\\n{GPU_PROCESS_MARKER}\\n' && "
        "(nvidia-smi --query-compute-apps=gpu_uuid "
        "--format=csv,noheader,nounits 2>/dev/null || true)"
    )
    result = command_runner(
        build_ssh_command(["bash", "-c", script]),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"remote GPU preflight failed: {detail}")
    return parse_gpu_query_output(result.stdout)


def parse_gpu_query_output(output: object) -> list[dict]:
    if not isinstance(output, str):
        raise ValueError("GPU query output must be text")
    before, marker, after = output.partition(
        GPU_PROCESS_MARKER
    )
    if not marker:
        raise ValueError("GPU query process marker is missing")
    process_counts = {}
    for line in after.splitlines():
        gpu_uuid = line.strip()
        if gpu_uuid:
            process_counts[gpu_uuid] = (
                process_counts.get(gpu_uuid, 0) + 1
            )
    rows = []
    for line in before.splitlines():
        if not line.strip():
            continue
        fields = [value.strip() for value in line.split(",")]
        if len(fields) != 4:
            raise ValueError("GPU query row is invalid")
        index, gpu_uuid, memory, utilization = fields
        try:
            row = {
                "index": int(index),
                "uuid": gpu_uuid,
                "memory_used_mib": int(memory),
                "utilization_percent": int(utilization),
                "compute_process_count": process_counts.get(
                    gpu_uuid,
                    0,
                ),
            }
        except ValueError as error:
            raise ValueError(
                "GPU query metric is invalid"
            ) from error
        rows.append(row)
    if not rows:
        raise ValueError("GPU query returned no devices")
    return rows


def _remote_preflight(
    *,
    target_model: str,
    draft_model: str,
    command_runner=subprocess.run,
) -> dict:
    gpu_rows = _query_remote_gpu_rows(
        command_runner=command_runner
    )
    gpu_classification = classify_gpu_preflight(gpu_rows)
    if gpu_classification["status"] != "READY":
        return {
            **gpu_classification,
            "gpu_rows": gpu_rows,
            "reason": gpu_classification["reason"],
        }
    script = (
        "import hashlib,json,pathlib,platform,subprocess,torch\n"
        "def digest_tree(root, tokenizer_only=False):\n"
        " root=pathlib.Path(root)\n"
        " if not root.is_dir(): return None\n"
        " names=[]\n"
        " for path in sorted(p for p in root.rglob('*') if p.is_file()):\n"
        "  rel=path.relative_to(root).as_posix()\n"
        "  if tokenizer_only and not any(x in rel.lower() for x in "
        "('tokenizer','vocab','merges','special_tokens')): continue\n"
        "  h=hashlib.sha256()\n"
        "  with path.open('rb') as source:\n"
        "   for block in iter(lambda:source.read(4*1024*1024),b''):"
        " h.update(block)\n"
        "  names.append((rel,h.hexdigest()))\n"
        " return hashlib.sha256(json.dumps(names,separators=(',',':'))"
        ".encode()).hexdigest()\n"
        f"target={target_model!r}\n"
        f"draft={draft_model!r}\n"
        f"package_root={REMOTE_PACKAGE_ROOT!r}\n"
        "nccl=torch.cuda.nccl.version() if torch.cuda.is_available() "
        "else None\n"
        "print(json.dumps({'host':platform.node(),"
        "'python_version':platform.python_version(),"
        "'torch_version':torch.__version__,"
        "'cuda_version':str(torch.version.cuda),"
        "'nccl_version':str(nccl),"
        "'target_exists':pathlib.Path(target).is_dir(),"
        "'draft_exists':pathlib.Path(draft).is_dir(),"
        "'package_root_exists':pathlib.Path(package_root).is_dir(),"
        "'target_model_fingerprint':digest_tree(target),"
        "'draft_model_fingerprint':digest_tree(draft),"
        "'tokenizer_fingerprint':digest_tree(target,True),"
        "},sort_keys=True))\n"
    )
    result = command_runner(
        build_ssh_command([REMOTE_PYTHON, "-c", script]),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"remote preflight failed: {detail}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            "remote preflight returned invalid JSON"
        ) from error
    if not all(
        payload.get(name) is True
        for name in (
            "target_exists",
            "draft_exists",
            "package_root_exists",
        )
    ):
        return {
            **payload,
            "status": "INCONCLUSIVE_ENVIRONMENT",
            "gpu_indices": [],
            "reason": "remote model or package prerequisite is missing",
        }
    return {
        **payload,
        **gpu_classification,
        "gpu_rows": gpu_rows,
    }


def _run_checked(
    command,
    *,
    context: str,
    command_runner=subprocess.run,
    **kwargs,
):
    result = command_runner(command, check=False, **kwargs)
    if result.returncode != 0:
        stdout = getattr(result, "stdout", "") or ""
        stderr = getattr(result, "stderr", "") or ""
        raise RuntimeError(
            f"{context} failed: {(stderr or stdout).strip()}"
        )
    return result


def _extract_download(archive_bytes: bytes, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=False)
    with tempfile.NamedTemporaryFile() as temporary:
        temporary.write(archive_bytes)
        temporary.flush()
        with tarfile.open(temporary.name, "r:") as archive:
            for member in archive.getmembers():
                member_path = Path(member.name)
                if member_path.is_absolute() or ".." in member_path.parts:
                    raise ValueError(
                        "download archive contains unsafe path"
                    )
            archive.extractall(output_root, filter="data")


def execute_remote_gate(
    *,
    repo_root: Path,
    run_tag: str,
    local_run: Path,
    target_model: str,
    draft_model: str,
    command_runner=subprocess.run,
) -> dict:
    run_tag = validate_run_tag(run_tag)
    local_run = Path(local_run)
    local_run.mkdir(parents=True, exist_ok=False)
    preflight = _remote_preflight(
        target_model=target_model,
        draft_model=draft_model,
        command_runner=command_runner,
    )
    (local_run / "preflight.json").write_bytes(
        canonical_json_bytes(preflight)
    )
    if preflight["status"] != "READY":
        return {
            "classification": "INCONCLUSIVE_ENVIRONMENT",
            "preflight": preflight,
            "local_run": str(local_run),
        }
    indices = preflight["gpu_indices"]
    selected_rows = [
        row
        for index in indices
        for row in preflight["gpu_rows"]
        if row["index"] == index
    ]
    provenance = {
        "python_version": preflight["python_version"],
        "torch_version": preflight["torch_version"],
        "cuda_version": preflight["cuda_version"],
        "nccl_version": preflight["nccl_version"],
        "target_model_fingerprint": preflight[
            "target_model_fingerprint"
        ],
        "draft_model_fingerprint": preflight[
            "draft_model_fingerprint"
        ],
        "tokenizer_fingerprint": preflight[
            "tokenizer_fingerprint"
        ],
        "gpu_uuids": [row["uuid"] for row in selected_rows],
    }
    environment = {
        "host": preflight["host"],
        "interference_detected": False,
        "gpu_before": [
            {"rank": rank, **row}
            for rank, row in enumerate(selected_rows)
        ],
        "gpu_after": [
            {"rank": rank, **row}
            for rank, row in enumerate(selected_rows)
        ],
    }
    bundle_root = local_run / "bundle"
    bundle = build_source_bundle(
        repo_root=repo_root,
        bundle_root=bundle_root,
        provenance=provenance,
        environment=environment,
        command_runner=command_runner,
    )
    remote_run = f"{REMOTE_RUN_ROOT}/{run_tag}"
    create = build_ssh_command([
        "bash",
        "-c",
        (
            f"test ! -e {shlex.quote(remote_run)} && "
            f"mkdir -p {shlex.quote(remote_run)} && "
            f"tar -xf - -C {shlex.quote(remote_run)}"
        ),
    ])
    _run_checked(
        create,
        context="remote source staging",
        command_runner=command_runner,
        input=bundle["archive_path"].read_bytes(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    remote_source = f"{remote_run}/source"
    remote_result = f"{remote_run}/result.json"
    gate_command = build_remote_gate_command(
        source_root=remote_source,
        target_model=target_model,
        draft_model=draft_model,
        output_path=remote_result,
        gpu_indices=indices,
        provenance_path=f"{remote_run}/provenance.json",
        environment_path=f"{remote_run}/environment.json",
        pythonpath_extra=REMOTE_PACKAGE_ROOT,
    )
    _run_checked(
        build_ssh_command(gate_command),
        context="remote CUDA graph gate",
        command_runner=command_runner,
        text=True,
        capture_output=True,
    )
    finalize_script = (
        "import hashlib,json,pathlib\n"
        f"root=pathlib.Path({remote_run!r})\n"
        "payload=json.loads((root/'result.json').read_text())\n"
        "seed=json.loads((root/'source_manifest_seed.json').read_text())\n"
        "canonical=(json.dumps(payload,sort_keys=True,separators=(',',':'),"
        "allow_nan=False)+'\\n').encode()\n"
        "seed['payload_sha256']=hashlib.sha256(canonical).hexdigest()\n"
        "(root/'source_manifest.json').write_text(json.dumps(seed,"
        "sort_keys=True,separators=(',',':'))+'\\n')\n"
    )
    _run_checked(
        build_ssh_command([
            REMOTE_PYTHON,
            "-c",
            finalize_script,
        ]),
        context="remote source manifest finalization",
        command_runner=command_runner,
        text=True,
        capture_output=True,
    )
    remote_verify = [
        "env",
        f"PYTHONPATH={REMOTE_PACKAGE_ROOT}:{remote_source}",
        REMOTE_PYTHON,
        f"{remote_source}/tools/"
        "verify_autoregressive_draft_cuda_graph_gate.py",
        "--payload",
        remote_result,
        "--source-root",
        remote_source,
        "--source-patch",
        f"{remote_run}/source.patch",
        "--source-manifest",
        f"{remote_run}/source_manifest.json",
        "--receipt",
        f"{remote_run}/verify.remote.json",
    ]
    _run_checked(
        build_ssh_command(remote_verify),
        context="archived remote verifier",
        command_runner=command_runner,
        text=True,
        capture_output=True,
    )
    checksum_script = (
        f"cd {shlex.quote(remote_run)} && "
        "find . -type f ! -name manifest.sha256 -print0 | "
        "sort -z | xargs -0 sha256sum > manifest.sha256 && "
        "sha256sum -c manifest.sha256"
    )
    _run_checked(
        build_ssh_command(["bash", "-c", checksum_script]),
        context="remote checksum verification",
        command_runner=command_runner,
        text=True,
        capture_output=True,
    )
    download = _run_checked(
        build_ssh_command([
            "tar",
            "-cf",
            "-",
            "-C",
            remote_run,
            ".",
        ]),
        context="remote artifact download",
        command_runner=command_runner,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    downloaded = local_run / "remote"
    _extract_download(download.stdout, downloaded)
    local_receipt = verify_gate_bundle(
        payload_path=downloaded / "result.json",
        source_root=downloaded / "source",
        source_patch_path=downloaded / "source.patch",
        source_manifest_path=downloaded / "source_manifest.json",
    )
    (downloaded / "verify.local.json").write_bytes(
        canonical_json_bytes(local_receipt)
    )
    manifest_check = subprocess.run(
        ["shasum", "-a", "256", "-c", "manifest.sha256"],
        cwd=downloaded,
        text=True,
        capture_output=True,
        check=False,
    )
    if manifest_check.returncode != 0:
        raise RuntimeError(
            "local checksum verification failed: "
            + (manifest_check.stderr or manifest_check.stdout)
        )
    return {
        "classification": local_receipt["classification"],
        "preflight": preflight,
        "local_run": str(local_run),
        "receipt": local_receipt,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[1]),
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--out")
    parser.add_argument("--run-tag")
    parser.add_argument("--local-run")
    parser.add_argument(
        "--target-model",
        default=DEFAULT_TARGET_MODEL,
    )
    parser.add_argument(
        "--draft-model",
        default=DEFAULT_DRAFT_MODEL,
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.preflight_only:
        result = _remote_preflight(
            target_model=args.target_model,
            draft_model=args.draft_model,
        )
    else:
        if not args.run_tag or not args.local_run:
            raise ValueError(
                "--run-tag and --local-run are required"
            )
        result = execute_remote_gate(
            repo_root=Path(args.repo_root),
            run_tag=args.run_tag,
            local_run=Path(args.local_run),
            target_model=args.target_model,
            draft_model=args.draft_model,
        )
    encoded = (
        json.dumps(result, sort_keys=True, separators=(",", ":"))
        + "\n"
    )
    if args.out:
        Path(args.out).write_text(encoded, encoding="utf-8")
    else:
        sys.stdout.write(encoded)
    return 0


if __name__ == "__main__":
    sys.exit(main())
