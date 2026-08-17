from __future__ import annotations

import argparse
from collections.abc import Mapping
import getpass
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import shlex
import socket
import subprocess
import sys
import tarfile
import tempfile


SCHEMA_VERSION = "qwen35.real-checkpoint-metadata-preflight.v1"
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
SOURCE_FILES = (
    "tinyvllm/models/qwen35_checkpoint_metadata.py",
    "tinyvllm/models/qwen35_checkpoint.py",
    "tools/qwen35_real_checkpoint_metadata_preflight.py",
)
SSH_CONTROL_PATH = "/tmp/ssh-sitian-10.232.195.203"
REMOTE_RUN_ROOT = (
    "/data00/home/sitian/sitian-workspace01/tllm/"
    "qwen35-real-checkpoint-metadata-preflight-runs"
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


def _positive_integer(value, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")
    return value


def _non_negative_integer(value, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _source_hashes(source_root) -> dict[str, str]:
    root = Path(source_root)
    result = {}
    for relative in SOURCE_FILES:
        path = root / relative
        if not path.is_file():
            raise ValueError(f"missing metadata preflight source: {relative}")
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
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for relative in SOURCE_FILES:
            path = root / relative
            if not path.is_file():
                raise ValueError(
                    f"missing metadata preflight source: {relative}"
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
    _require_success(staged, "metadata preflight source staging")

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
        build_ssh_command([REMOTE_PYTHON, "-c", script]),
        text=True,
        capture_output=True,
    )
    _require_success(verified, "metadata preflight source hashing")
    remote_hashes = json.loads(verified.stdout)
    if remote_hashes != local_hashes:
        raise ValueError(
            "metadata preflight remote source hashes do not match local"
        )
    return {
        "remote_source_dir": remote_source_dir,
        "local_file_sha256": local_hashes,
        "remote_file_sha256": remote_hashes,
        "source_tree_sha256": _source_tree_sha256(local_hashes),
    }


def _load_source_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load source module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_checkpoint_dependencies(source_root):
    root = Path(source_root)
    metadata = _load_source_module(
        "_qwen35_checkpoint_metadata_preflight_metadata",
        root / "tinyvllm/models/qwen35_checkpoint_metadata.py",
    )
    checkpoint = _load_source_module(
        "_qwen35_checkpoint_metadata_preflight_checkpoint",
        root / "tinyvllm/models/qwen35_checkpoint.py",
    )
    return metadata, checkpoint


def validate_metadata_preflight(record):
    if not isinstance(record, Mapping):
        raise ValueError("metadata preflight must be a mapping")
    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("metadata preflight schema_version is invalid")
    if record.get("status") != "PASS":
        raise ValueError("metadata preflight status must be PASS")
    exact_values = {
        "remote_target": REMOTE_TARGET,
        "remote_python": REMOTE_PYTHON,
        "checkpoint_dir": APPROVED_MODEL_DIR,
        "model_manifest_sha256": APPROVED_MODEL_MANIFEST_SHA256,
        "config_sha256": APPROVED_CONFIG_SHA256,
        "index_sha256": APPROVED_INDEX_SHA256,
        "config_index_header_sha256": APPROVED_COMPOSITE_SHA256,
        "payload_identity_source": "retained_approved_manifest",
    }
    for name, expected in exact_values.items():
        if record.get(name) != expected:
            raise ValueError(f"metadata preflight {name} is invalid")
    if record.get("observed_user") != "sitian":
        raise ValueError("metadata preflight observed_user is invalid")
    if not isinstance(record.get("observed_hostname"), str) or not record[
        "observed_hostname"
    ]:
        raise ValueError("metadata preflight observed_hostname is invalid")

    shards = record.get("shards")
    expected_shards = [{
        "name": APPROVED_SHARD_NAME,
        "size": APPROVED_SHARD_SIZE,
        "sha256": APPROVED_SHARD_SHA256,
    }]
    if shards != expected_shards:
        raise ValueError("metadata preflight shard identity is invalid")

    source_hashes = record.get("source_file_sha256")
    if (
        not isinstance(source_hashes, Mapping)
        or set(source_hashes) != set(SOURCE_FILES)
    ):
        raise ValueError("metadata preflight source hashes are invalid")
    for name, digest in source_hashes.items():
        _sha256(digest, f"source SHA256 for {name}")
    source_tree = _sha256(
        record.get("source_tree_sha256"),
        "source_tree_sha256",
    )
    if source_tree != _source_tree_sha256(source_hashes):
        raise ValueError("metadata preflight source tree is invalid")

    _positive_integer(
        record.get("metadata_bytes_read"),
        "metadata_bytes_read",
    )
    if _non_negative_integer(
        record.get("payload_bytes_read"),
        "payload_bytes_read",
    ) != 0:
        raise ValueError("metadata preflight payload_bytes_read must be zero")
    if record.get("payload_hashes_recomputed") is not False:
        raise ValueError("metadata preflight payload hashes were recomputed")

    layer_count = _positive_integer(
        record.get("layer_count"),
        "layer_count",
    )
    linear_count = _positive_integer(
        record.get("linear_attention_layer_count"),
        "linear_attention_layer_count",
    )
    full_count = _positive_integer(
        record.get("full_attention_layer_count"),
        "full_attention_layer_count",
    )
    if (
        layer_count != 24
        or linear_count != 18
        or full_count != 6
        or linear_count + full_count != layer_count
    ):
        raise ValueError("metadata preflight layer counts are invalid")

    index_weight_count = _positive_integer(
        record.get("index_weight_count"),
        "index_weight_count",
    )
    header_tensor_count = _positive_integer(
        record.get("header_tensor_count"),
        "header_tensor_count",
    )
    load_count = _positive_integer(
        record.get("load_count"),
        "load_count",
    )
    skip_count = _positive_integer(
        record.get("skip_count"),
        "skip_count",
    )
    if (
        index_weight_count != 632
        or header_tensor_count != index_weight_count
        or load_count != 320
        or skip_count != 312
        or load_count + skip_count != index_weight_count
    ):
        raise ValueError("metadata preflight tensor counts are invalid")

    plan_payload_bytes = _positive_integer(
        record.get("plan_payload_bytes"),
        "plan_payload_bytes",
    )
    index_total_size = _positive_integer(
        record.get("index_total_size"),
        "index_total_size",
    )
    if (
        plan_payload_bytes != 4548144832
        or plan_payload_bytes != index_total_size
    ):
        raise ValueError("metadata preflight payload total is invalid")
    if record.get("shard_count") != 1:
        raise ValueError("metadata preflight shard_count is invalid")
    return record


def run_metadata_worker(
    *,
    checkpoint_dir,
    source_root,
    observed_user,
    observed_hostname,
):
    metadata_module, checkpoint_module = _load_checkpoint_dependencies(
        source_root
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
    tensor_plan = checkpoint_module.build_qwen35_checkpoint_tensor_plan(
        metadata.hf_config,
        metadata.index_payload,
        metadata.shard_headers,
    )
    config = getattr(
        metadata.hf_config,
        "text_config",
        metadata.hf_config,
    )
    layer_types = tuple(config.layer_types)
    weight_map = metadata.index_payload["weight_map"]
    header_tensor_count = sum(
        1
        for header in metadata.shard_headers.values()
        for name in header
        if name != "__metadata__"
    )
    source_hashes = _source_hashes(source_root)
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
        "shards": [{
            "name": shard.name,
            "size": shard.size,
            "sha256": shard.sha256,
        }],
        "source_file_sha256": source_hashes,
        "source_tree_sha256": _source_tree_sha256(source_hashes),
        "metadata_bytes_read": metadata.metadata_bytes_read,
        "payload_bytes_read": metadata.payload_bytes_read,
        "payload_hashes_recomputed": False,
        "payload_identity_source": "retained_approved_manifest",
        "layer_count": len(layer_types),
        "linear_attention_layer_count": layer_types.count(
            "linear_attention"
        ),
        "full_attention_layer_count": layer_types.count(
            "full_attention"
        ),
        "index_weight_count": len(weight_map),
        "header_tensor_count": header_tensor_count,
        "load_count": len(tensor_plan.loads),
        "skip_count": len(tensor_plan.skips),
        "plan_payload_bytes": tensor_plan.payload_bytes,
        "index_total_size": metadata.index_payload["metadata"][
            "total_size"
        ],
        "shard_count": len(metadata.shard_headers),
    }


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


def run_remote_metadata_preflight(
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
            f"local metadata preflight directory already exists: {destination}"
        )
    remote_run_dir = f"{REMOTE_RUN_ROOT}/{run_tag}"
    remote_artifact = f"{remote_run_dir}/metadata_preflight.json"
    worker = (
        f"{staged['remote_source_dir']}/"
        "tools/qwen35_real_checkpoint_metadata_preflight.py"
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
    _require_success(completed, "remote metadata preflight worker")
    record = json.loads(completed.stdout)
    validate_metadata_preflight(record)
    if (
        record["source_file_sha256"]
        != staged["local_file_sha256"]
        or record["source_file_sha256"]
        != staged["remote_file_sha256"]
        or record["source_tree_sha256"]
        != staged["source_tree_sha256"]
    ):
        raise ValueError("metadata preflight source binding mismatch")

    source_manifest = _source_manifest(run_tag, staged)
    round_trip_script = "\n".join([
        "import json,pathlib,sys",
        f"root=pathlib.Path({remote_run_dir!r})",
        "payload=json.load(sys.stdin)",
        "metadata=json.loads((root/'metadata_preflight.json').read_text())",
        "temporary=root/'.source_manifest.json.tmp'",
        "temporary.write_text(json.dumps(payload['source_manifest'],sort_keys=True,separators=(',',':'))+'\\n')",
        "temporary.replace(root/'source_manifest.json')",
        "result={'metadata_preflight':metadata,'source_manifest':json.loads((root/'source_manifest.json').read_text())}",
        "print(json.dumps(result,sort_keys=True,separators=(',',':')))",
    ])
    round_trip = command_runner(
        build_ssh_command([REMOTE_PYTHON, "-c", round_trip_script]),
        input=json.dumps({
            "metadata_preflight": record,
            "source_manifest": source_manifest,
        }),
        text=True,
        capture_output=True,
    )
    _require_success(round_trip, "metadata preflight artifact round trip")
    returned = json.loads(round_trip.stdout)
    if returned != {
        "metadata_preflight": record,
        "source_manifest": source_manifest,
    }:
        raise ValueError("metadata preflight artifact round-trip mismatch")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{run_tag}.",
        suffix=".tmp",
        dir=destination.parent,
    ))
    try:
        _atomic_write_json(
            temporary / "metadata_preflight.json",
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


def execute_remote_metadata_preflight(
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
    return run_remote_metadata_preflight(
        source_root,
        run_tag,
        staged=staged,
        local_run_root=local_run_root,
        command_runner=command_runner,
    )


def _worker_main(arguments) -> int:
    if str(Path(arguments.checkpoint_dir).resolve()) != APPROVED_MODEL_DIR:
        raise ValueError("worker checkpoint_dir is not the approved model")
    record = run_metadata_worker(
        checkpoint_dir=arguments.checkpoint_dir,
        source_root=arguments.source_root,
        observed_user=getpass.getuser(),
        observed_hostname=socket.gethostname(),
    )
    validate_metadata_preflight(record)
    output = Path(arguments.output)
    if output.exists():
        raise ValueError("metadata preflight output already exists")
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
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--artifact", required=True)
    worker_parser = subparsers.add_parser("internal-worker")
    worker_parser.add_argument("--source-root", required=True)
    worker_parser.add_argument("--checkpoint-dir", required=True)
    worker_parser.add_argument("--output", required=True)
    arguments = parser.parse_args(argv)

    if arguments.mode == "internal-worker":
        return _worker_main(arguments)
    if arguments.mode == "validate":
        record = json.loads(Path(arguments.artifact).read_text())
        validate_metadata_preflight(record)
        print(json.dumps(record, sort_keys=True, separators=(",", ":")))
        return 0
    record = execute_remote_metadata_preflight(
        arguments.source_root,
        arguments.run_tag,
    )
    print(json.dumps(record, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
