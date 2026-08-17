from __future__ import annotations

import argparse
from collections.abc import Mapping
import ast
import getpass
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys
import tarfile
import tempfile
import types


SCHEMA_VERSION = (
    "qwen35.real-checkpoint-loader-construction-preflight.v1"
)
REMOTE_TARGET = "sitian@10.232.195.203"
REMOTE_PYTHON = (
    "/data00/home/sitian/sitian-workspace01/tllm/env/bin/python"
)
APPROVED_MODEL_DIR = (
    "/data00/home/sitian/sitian-workspace01/tllm/"
    "qwen35-hybrid-state-runs/"
    "qwen35-2b-hybrid-acquire-20260723-222004/model"
)
APPROVED_MODEL_MANIFEST_SHA256 = (
    "3e650a908234771c3cf1ac4e20c4d38fe69982efedaf4a3e631ad0b14aad7dd0"
)
APPROVED_CONFIG_SHA256 = (
    "ed1c1723241f23f7f4e23430759cbd7dcfb4103cbdfe052bfe7626b57c2615b4"
)
APPROVED_INDEX_SHA256 = (
    "aca8afed9da75b0f050b408d270766fd77627f1af401e240f61c3b47d0db02f9"
)
APPROVED_SHARD_NAME = (
    "model.safetensors-00001-of-00001.safetensors"
)
APPROVED_SHARD_SIZE = 4548221488
APPROVED_SHARD_SHA256 = (
    "aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1"
)
APPROVED_COMPOSITE_SHA256 = (
    "27da983f5ef3e38480d8b5d5976e5c434fc4b5d0c70d09511c35154beecd8db9"
)
AUTHORIZATION_SHA256 = (
    "10a39d6eb918cb5e8d1ccf52a723cdca4db4dffb9fd4ded62b1b766474d4fde4"
)
MAX_TOTAL_VMHWM_INCREMENT_KIB = 524288
MAX_CONSTRUCTION_VMHWM_INCREMENT_KIB = 262144
PRODUCTION_SOURCE_FILES = (
    "tinyvllm/engine/hybrid_state.py",
    "tinyvllm/engine/qwen35_hybrid_model_owner.py",
    "tinyvllm/engine/qwen35_hybrid_prefix_runtime_identity.py",
    "tinyvllm/engine/qwen35_hybrid_state.py",
    "tinyvllm/engine/qwen35_layer_state.py",
    "tinyvllm/engine/qwen35_state_transaction.py",
    "tinyvllm/layers/embed_head.py",
    "tinyvllm/layers/gated_delta.py",
    "tinyvllm/layers/linear.py",
    "tinyvllm/layers/quantization.py",
    "tinyvllm/layers/qwen35_decoder_layer.py",
    "tinyvllm/layers/qwen35_full_attention.py",
    "tinyvllm/layers/qwen35_linear_attention.py",
    "tinyvllm/layers/qwen35_packed_layer_stack.py",
    "tinyvllm/layers/qwen35_packed_stateful_decoder_layer.py",
    "tinyvllm/layers/qwen35_primitives.py",
    "tinyvllm/layers/qwen35_rotary_embedding.py",
    "tinyvllm/models/qwen35_checkpoint.py",
    "tinyvllm/models/qwen35_checkpoint_assignment.py",
    "tinyvllm/models/qwen35_checkpoint_binding.py",
    "tinyvllm/models/qwen35_checkpoint_candidate_factory.py",
    "tinyvllm/models/qwen35_checkpoint_candidate_loader.py",
    "tinyvllm/models/qwen35_checkpoint_loader_configuration.py",
    "tinyvllm/models/qwen35_checkpoint_metadata.py",
    "tinyvllm/models/qwen35_checkpoint_streaming.py",
    "tinyvllm/models/qwen35_checkpoint_worker.py",
    "tinyvllm/models/qwen35_components.py",
    "tinyvllm/models/qwen35_factory.py",
    "tinyvllm/models/qwen35_packed.py",
    "tinyvllm/speculative/verifier.py",
    "tinyvllm/utils/context.py",
    "tools/qwen35_real_checkpoint_load_worker.py",
)
SOURCE_FILES = (
    *PRODUCTION_SOURCE_FILES,
    "tools/qwen35_real_checkpoint_loader_construction_preflight.py",
)
PACKAGE_NAMES = (
    "tinyvllm",
    "tinyvllm.engine",
    "tinyvllm.layers",
    "tinyvllm.models",
    "tinyvllm.speculative",
    "tinyvllm.utils",
)
TP_ROWS = ((1, 0), (2, 0), (2, 1))
SSH_CONTROL_PATH = "/tmp/ssh-sitian-10.232.195.203"
REMOTE_RUN_ROOT = (
    "/data00/home/sitian/sitian-workspace01/tllm/"
    "qwen35-real-checkpoint-loader-construction-runs"
)
LOCAL_RUN_ROOT = Path("experiments/qwen35_hybrid_state")
RUN_TAG_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


def _sha256(value, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(
            character not in "0123456789abcdef"
            for character in value
        )
    ):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _non_negative_integer(value, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_integer(value, name: str) -> int:
    value = _non_negative_integer(value, name)
    if value == 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _source_hashes(source_root) -> dict[str, str]:
    root = Path(source_root)
    result = {}
    for relative in SOURCE_FILES:
        path = root / relative
        if not path.is_file():
            raise ValueError(
                f"missing loader construction source: {relative}"
            )
        result[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _source_tree_sha256(hashes: Mapping[str, str]) -> str:
    payload = json.dumps(
        dict(hashes),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_run_tag(value) -> str:
    text = str(value)
    if not RUN_TAG_PATTERN.fullmatch(text):
        raise ValueError("run tag must match [A-Za-z0-9_-]+")
    return text


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


def build_source_tar(source_root) -> bytes:
    root = Path(source_root)
    if discover_production_source_files(root) != PRODUCTION_SOURCE_FILES:
        raise ValueError("production source closure does not match frozen set")
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for relative in SOURCE_FILES:
            path = root / relative
            if not path.is_file():
                raise ValueError(
                    f"missing loader construction source: {relative}"
                )
            info = archive.gettarinfo(os.fspath(path), arcname=relative)
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            with path.open("rb") as source:
                archive.addfile(info, source)
    return stream.getvalue()


def _require_success(result, context: str):
    if result.returncode != 0:
        detail = result.stderr or result.stdout or ""
        if isinstance(detail, bytes):
            detail = detail.decode("utf-8", errors="replace")
        raise RuntimeError(f"{context} failed: {str(detail).strip()}")
    return result


def stage_source(
    source_root,
    run_tag,
    *,
    command_runner=subprocess.run,
):
    run_tag = validate_run_tag(run_tag)
    remote_run_dir = f"{REMOTE_RUN_ROOT}/{run_tag}"
    remote_source_dir = f"{remote_run_dir}/source"
    payload = build_source_tar(source_root)
    staged = command_runner(
        build_ssh_command([
            "bash",
            "-c",
            (
                f"test ! -e {shlex.quote(remote_run_dir)} && "
                f"mkdir -p {shlex.quote(remote_source_dir)} && "
                f"tar -xf - -C {shlex.quote(remote_source_dir)}"
            ),
        ]),
        input=payload,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _require_success(staged, "construction source staging")
    local_hashes = _source_hashes(source_root)
    script = "\n".join([
        "import hashlib,json,pathlib",
        f"root=pathlib.Path({remote_source_dir!r})",
        f"names={list(SOURCE_FILES)!r}",
        "result={}",
        "for name in names:",
        " path=root/name",
        " if not path.is_file(): raise SystemExit('missing source: '+name)",
        " result[name]=hashlib.sha256(path.read_bytes()).hexdigest()",
        "print(json.dumps(result,sort_keys=True,separators=(',',':')))",
    ])
    verified = command_runner(
        build_ssh_command([
            "env",
            "PYTHONDONTWRITEBYTECODE=1",
            REMOTE_PYTHON,
            "-B",
            "-c",
            script,
        ]),
        text=True,
        capture_output=True,
    )
    _require_success(verified, "construction remote source hashing")
    remote_hashes = json.loads(verified.stdout)
    if remote_hashes != local_hashes:
        raise ValueError(
            "construction remote source hashes do not match local"
        )
    return {
        "remote_source_dir": remote_source_dir,
        "local_file_sha256": local_hashes,
        "remote_file_sha256": remote_hashes,
        "source_tree_sha256": _source_tree_sha256(local_hashes),
    }


def discover_production_source_files(source_root) -> tuple[str, ...]:
    root = Path(source_root)
    queue = [
        "tools/qwen35_real_checkpoint_load_worker.py",
        "tinyvllm/models/qwen35_checkpoint_metadata.py",
    ]
    observed = set()
    while queue:
        relative = queue.pop(0)
        if relative in observed:
            continue
        observed.add(relative)
        tree = ast.parse(
            (root / relative).read_text(encoding="utf-8"),
            filename=relative,
        )
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            else:
                modules = []
            for module in modules:
                if not module.startswith("tinyvllm."):
                    continue
                candidate = module.replace(".", "/") + ".py"
                if (root / candidate).is_file():
                    queue.append(candidate)
    return tuple(sorted(observed))


def _read_proc_status() -> dict[str, int]:
    result = {}
    for line in Path("/proc/self/status").read_text(
        encoding="utf-8"
    ).splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        parts = value.strip().split()
        if key in ("VmRSS", "VmHWM") and parts:
            result[key] = int(parts[0])
    return result


def _install_namespace_packages(source_root) -> None:
    root = Path(source_root)
    for name in PACKAGE_NAMES:
        package = types.ModuleType(name)
        package.__path__ = [str(root / name.replace(".", "/"))]
        sys.modules[name] = package


def _load_source_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load source module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def validate_construction_preflight(record):
    if not isinstance(record, Mapping):
        raise ValueError("construction preflight must be a mapping")
    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("construction preflight schema is invalid")
    if record.get("status") != "PASS":
        raise ValueError("construction preflight status must be PASS")
    exact = {
        "remote_target": REMOTE_TARGET,
        "remote_python": REMOTE_PYTHON,
        "checkpoint_dir": APPROVED_MODEL_DIR,
        "model_manifest_sha256": APPROVED_MODEL_MANIFEST_SHA256,
        "config_sha256": APPROVED_CONFIG_SHA256,
        "index_sha256": APPROVED_INDEX_SHA256,
        "config_index_header_sha256": APPROVED_COMPOSITE_SHA256,
    }
    for name, expected in exact.items():
        if record.get(name) != expected:
            raise ValueError(f"construction preflight {name} is invalid")
    if record.get("observed_user") != "sitian":
        raise ValueError("construction preflight user is invalid")
    if not isinstance(record.get("observed_hostname"), str) or not record[
        "observed_hostname"
    ]:
        raise ValueError("construction preflight host is invalid")

    source_hashes = record.get("source_file_sha256")
    if (
        not isinstance(source_hashes, Mapping)
        or set(source_hashes) != set(SOURCE_FILES)
    ):
        raise ValueError("construction preflight source hashes are invalid")
    for name, digest in source_hashes.items():
        _sha256(digest, f"source SHA256 for {name}")
    if (
        _sha256(record.get("source_tree_sha256"), "source_tree_sha256")
        != _source_tree_sha256(source_hashes)
    ):
        raise ValueError("construction preflight source tree is invalid")

    if record.get("metadata_bytes_read") != 144024:
        raise ValueError("construction metadata_bytes_read is invalid")
    if record.get("payload_bytes_read") != 0:
        raise ValueError("construction payload_bytes_read must be zero")
    if record.get("payload_hashes_recomputed") is not False:
        raise ValueError("construction payload hashes were recomputed")
    if record.get("provider_events") != []:
        raise ValueError("construction provider events must be empty")
    for name in (
        "loader_call_count",
        "pool_create_count",
        "backend_create_count",
    ):
        if _non_negative_integer(record.get(name), name) != 0:
            raise ValueError(f"construction {name} must be zero")
    if (
        record.get("cuda_initialized_before") is not False
        or record.get("cuda_initialized_after") is not False
    ):
        raise ValueError("construction CUDA must remain uninitialized")

    vmrss_before = _positive_integer(
        record.get("vmrss_before_kib"),
        "vmrss_before_kib",
    )
    vmrss_after_torch = _positive_integer(
        record.get("vmrss_after_torch_kib"),
        "vmrss_after_torch_kib",
    )
    vmrss_after = _positive_integer(
        record.get("vmrss_after_kib"),
        "vmrss_after_kib",
    )
    vmhwm_before = _positive_integer(
        record.get("vmhwm_before_kib"),
        "vmhwm_before_kib",
    )
    vmhwm_after_torch = _positive_integer(
        record.get("vmhwm_after_torch_kib"),
        "vmhwm_after_torch_kib",
    )
    vmhwm_after = _positive_integer(
        record.get("vmhwm_after_kib"),
        "vmhwm_after_kib",
    )
    total_increment = _non_negative_integer(
        record.get("total_vmhwm_increment_kib"),
        "total_vmhwm_increment_kib",
    )
    construction_increment = _non_negative_integer(
        record.get("construction_vmhwm_increment_kib"),
        "construction_vmhwm_increment_kib",
    )
    if (
        vmrss_before > vmhwm_before
        or vmrss_after_torch > vmhwm_after_torch
        or vmrss_after > vmhwm_after
    ):
        raise ValueError("construction VmRSS exceeds VmHWM")
    if total_increment != max(0, vmhwm_after - vmhwm_before):
        raise ValueError("construction total VmHWM is inconsistent")
    if construction_increment != max(
        0,
        vmhwm_after - vmhwm_after_torch,
    ):
        raise ValueError("construction phase VmHWM is inconsistent")
    if total_increment > MAX_TOTAL_VMHWM_INCREMENT_KIB:
        raise ValueError("construction total VmHWM exceeds ceiling")
    if (
        construction_increment
        > MAX_CONSTRUCTION_VMHWM_INCREMENT_KIB
    ):
        raise ValueError("construction phase VmHWM exceeds ceiling")

    rows = record.get("rows")
    if not isinstance(rows, list) or [
        (row.get("tp_size"), row.get("tp_rank"))
        for row in rows
        if isinstance(row, Mapping)
    ] != list(TP_ROWS):
        raise ValueError("construction TP rows are invalid")
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("construction TP rows are invalid")
        if row.get("loader_type") != (
            "Qwen35ManifestBoundCheckpointCandidateLoader"
        ):
            raise ValueError("construction loader type is invalid")
        if row.get("configuration_type") != (
            "Qwen35RankCheckpointLoaderConfiguration"
        ):
            raise ValueError("construction configuration type is invalid")
        if row.get("manifest_dir") != APPROVED_MODEL_DIR:
            raise ValueError("construction manifest directory is invalid")
        if (
            row.get("plan_loads") != 320
            or row.get("plan_skips") != 312
            or row.get("plan_payload_bytes") != 4548144832
        ):
            raise ValueError("construction tensor plan is invalid")
    return record


def run_construction_worker(
    *,
    checkpoint_dir,
    source_root,
    observed_user,
    observed_hostname,
    status_reader=_read_proc_status,
):
    root = Path(source_root)
    before = status_reader()
    _install_namespace_packages(root)
    import torch

    cuda_before = torch.cuda.is_initialized()
    after_torch = status_reader()
    metadata_module = __import__(
        "tinyvllm.models.qwen35_checkpoint_metadata",
        fromlist=["*"],
    )
    worker = _load_source_module(
        "_qwen35_real_checkpoint_load_worker_for_construction_preflight",
        root / "tools/qwen35_real_checkpoint_load_worker.py",
    )
    shard = metadata_module.Qwen35CheckpointShardIdentity(
        name=APPROVED_SHARD_NAME,
        size=APPROVED_SHARD_SIZE,
        sha256=APPROVED_SHARD_SHA256,
    )
    metadata = metadata_module.read_qwen35_checkpoint_metadata(
        checkpoint_dir,
        shards=(shard,),
        expected_config_sha256=APPROVED_CONFIG_SHA256,
        expected_index_sha256=APPROVED_INDEX_SHA256,
        expected_config_index_header_sha256=APPROVED_COMPOSITE_SHA256,
    )
    provider_events = []

    def forbidden_pool():
        provider_events.append("pool")
        raise AssertionError("pool provider called during construction")

    def forbidden_backend(*_args, **_kwargs):
        provider_events.append("backend")
        raise AssertionError(
            "attention backend provider called during construction"
        )

    rows = []
    for tensor_parallel_size, tensor_parallel_rank in TP_ROWS:
        loader = (
            worker.build_qwen35_real_checkpoint_rank_loader_from_metadata(
                metadata,
                checkpoint_dir=str(Path(checkpoint_dir).resolve()),
                model_manifest_sha256=APPROVED_MODEL_MANIFEST_SHA256,
                tensor_parallel_size=tensor_parallel_size,
                tensor_parallel_rank=tensor_parallel_rank,
                create_pool=forbidden_pool,
                build_attention_backend=forbidden_backend,
                authorization_sha256=AUTHORIZATION_SHA256,
            )
        )
        configuration = loader.configuration
        rows.append({
            "tp_size": tensor_parallel_size,
            "tp_rank": tensor_parallel_rank,
            "loader_type": type(loader).__name__,
            "configuration_type": type(configuration).__name__,
            "manifest_dir": configuration.manifest.checkpoint_dir,
            "plan_loads": len(configuration.tensor_plan.loads),
            "plan_skips": len(configuration.tensor_plan.skips),
            "plan_payload_bytes": configuration.tensor_plan.payload_bytes,
        })
    after = status_reader()
    source_hashes = _source_hashes(root)
    vmhwm_before = before["VmHWM"]
    vmhwm_after = after["VmHWM"]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "remote_target": REMOTE_TARGET,
        "remote_python": REMOTE_PYTHON,
        "observed_user": observed_user,
        "observed_hostname": observed_hostname,
        "checkpoint_dir": str(Path(checkpoint_dir).resolve()),
        "model_manifest_sha256": APPROVED_MODEL_MANIFEST_SHA256,
        "config_sha256": metadata.config_sha256,
        "index_sha256": metadata.index_sha256,
        "config_index_header_sha256": (
            metadata.config_index_header_sha256
        ),
        "source_file_sha256": source_hashes,
        "source_tree_sha256": _source_tree_sha256(source_hashes),
        "metadata_bytes_read": metadata.metadata_bytes_read,
        "payload_bytes_read": metadata.payload_bytes_read,
        "payload_hashes_recomputed": False,
        "provider_events": provider_events,
        "loader_call_count": 0,
        "pool_create_count": provider_events.count("pool"),
        "backend_create_count": provider_events.count("backend"),
        "cuda_initialized_before": cuda_before,
        "cuda_initialized_after": torch.cuda.is_initialized(),
        "vmrss_before_kib": before["VmRSS"],
        "vmrss_after_torch_kib": after_torch["VmRSS"],
        "vmrss_after_kib": after["VmRSS"],
        "vmhwm_before_kib": vmhwm_before,
        "vmhwm_after_torch_kib": after_torch["VmHWM"],
        "vmhwm_after_kib": vmhwm_after,
        "total_vmhwm_increment_kib": max(
            0,
            vmhwm_after - vmhwm_before,
        ),
        "construction_vmhwm_increment_kib": max(
            0,
            vmhwm_after - after_torch["VmHWM"],
        ),
        "rows": rows,
    }


def _atomic_write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
    try:
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _source_manifest(run_tag, staged):
    return {
        "schema_version": SCHEMA_VERSION,
        "run_tag": validate_run_tag(run_tag),
        "remote_target": REMOTE_TARGET,
        "remote_source_dir": staged["remote_source_dir"],
        "source_tree_sha256": staged["source_tree_sha256"],
        "local_file_sha256": dict(staged["local_file_sha256"]),
        "remote_file_sha256": dict(staged["remote_file_sha256"]),
    }


def run_remote_construction_preflight(
    source_root,
    run_tag,
    *,
    staged,
    local_run_root=LOCAL_RUN_ROOT,
    command_runner=subprocess.run,
):
    run_tag = validate_run_tag(run_tag)
    destination = Path(local_run_root) / run_tag
    if destination.exists():
        raise ValueError(
            f"local construction preflight directory already exists: "
            f"{destination}"
        )
    remote_run_dir = f"{REMOTE_RUN_ROOT}/{run_tag}"
    remote_artifact = (
        f"{remote_run_dir}/loader_construction_preflight.json"
    )
    worker = (
        f"{staged['remote_source_dir']}/"
        "tools/qwen35_real_checkpoint_loader_construction_preflight.py"
    )
    completed = command_runner(
        build_ssh_command([
            "env",
            "CUDA_VISIBLE_DEVICES=",
            "PYTHONDONTWRITEBYTECODE=1",
            REMOTE_PYTHON,
            "-B",
            worker,
            "internal-worker",
            "--source-root",
            staged["remote_source_dir"],
            "--checkpoint-dir",
            APPROVED_MODEL_DIR,
            "--output",
            remote_artifact,
        ]),
        text=True,
        capture_output=True,
    )
    _require_success(completed, "remote construction preflight worker")
    record = json.loads(completed.stdout)
    validate_construction_preflight(record)
    if (
        record["source_file_sha256"]
        != staged["local_file_sha256"]
        or record["source_file_sha256"]
        != staged["remote_file_sha256"]
        or record["source_tree_sha256"]
        != staged["source_tree_sha256"]
    ):
        raise ValueError("construction preflight source binding mismatch")
    source_manifest = _source_manifest(run_tag, staged)
    round_trip_script = "\n".join([
        "import json,pathlib,sys",
        f"root=pathlib.Path({remote_run_dir!r})",
        "payload=json.load(sys.stdin)",
        "record=json.loads((root/'loader_construction_preflight.json').read_text())",
        "temporary=root/'.source_manifest.json.tmp'",
        "temporary.write_text(json.dumps(payload['source_manifest'],sort_keys=True,separators=(',',':'))+'\\n')",
        "temporary.replace(root/'source_manifest.json')",
        "result={'loader_construction_preflight':record,'source_manifest':json.loads((root/'source_manifest.json').read_text())}",
        "print(json.dumps(result,sort_keys=True,separators=(',',':')))",
    ])
    round_trip = command_runner(
        build_ssh_command([
            "env",
            "PYTHONDONTWRITEBYTECODE=1",
            REMOTE_PYTHON,
            "-B",
            "-c",
            round_trip_script,
        ]),
        input=json.dumps({
            "loader_construction_preflight": record,
            "source_manifest": source_manifest,
        }),
        text=True,
        capture_output=True,
    )
    _require_success(round_trip, "construction artifact round trip")
    returned = json.loads(round_trip.stdout)
    if returned != {
        "loader_construction_preflight": record,
        "source_manifest": source_manifest,
    }:
        raise ValueError("construction artifact round-trip mismatch")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{run_tag}.",
        suffix=".tmp",
        dir=destination.parent,
    ))
    try:
        _atomic_write_json(
            temporary / "loader_construction_preflight.json",
            record,
        )
        _atomic_write_json(
            temporary / "source_manifest.json",
            source_manifest,
        )
        temporary.replace(destination)
    finally:
        if temporary.exists():
            for child in temporary.iterdir():
                child.unlink()
            temporary.rmdir()
    return record


def execute_remote_construction_preflight(
    source_root,
    run_tag,
    *,
    local_run_root=LOCAL_RUN_ROOT,
    command_runner=subprocess.run,
):
    staged = stage_source(
        source_root,
        run_tag,
        command_runner=command_runner,
    )
    return run_remote_construction_preflight(
        source_root,
        run_tag,
        staged=staged,
        local_run_root=local_run_root,
        command_runner=command_runner,
    )


def _worker_main(arguments) -> int:
    if str(Path(arguments.checkpoint_dir).resolve()) != APPROVED_MODEL_DIR:
        raise ValueError("worker checkpoint_dir is not the approved model")
    record = run_construction_worker(
        checkpoint_dir=arguments.checkpoint_dir,
        source_root=arguments.source_root,
        observed_user=getpass.getuser(),
        observed_hostname=socket.gethostname(),
    )
    validate_construction_preflight(record)
    output = Path(arguments.output)
    if output.exists():
        raise ValueError("construction preflight output already exists")
    _atomic_write_json(output, record)
    print(json.dumps(record, sort_keys=True, separators=(",", ":")))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--run-tag", required=True)
    run_parser.add_argument(
        "--source-root",
        default=str(Path(__file__).resolve().parents[1]),
    )
    worker_parser = subparsers.add_parser("internal-worker")
    worker_parser.add_argument("--source-root", required=True)
    worker_parser.add_argument("--checkpoint-dir", required=True)
    worker_parser.add_argument("--output", required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--artifact", required=True)
    arguments = parser.parse_args(argv)
    if arguments.mode == "internal-worker":
        return _worker_main(arguments)
    if arguments.mode == "validate":
        record = json.loads(Path(arguments.artifact).read_text())
        validate_construction_preflight(record)
    else:
        record = execute_remote_construction_preflight(
            arguments.source_root,
            arguments.run_tag,
        )
    print(json.dumps(record, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
