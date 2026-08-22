"""Independent artifact-only verifier for staged inference benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import tarfile
from tempfile import TemporaryDirectory


OWNED_SOURCE_ROOTS = (
    "tools/arrival_load_driver.py",
    "tools/profile_prefix_cache.py",
    "tools/source_audit.py",
    "tools/staged_inference_benchmark_contract.py",
    "tools/staged_inference_benchmark_gate.py",
    "tools/staged_inference_benchmark_worker.py",
    "tools/test_arrival_load_driver.py",
    "tools/test_profile_prefix_cache.py",
    "tools/test_staged_inference_benchmark_contract.py",
    "tools/test_staged_inference_benchmark_gate.py",
    "tools/test_staged_inference_benchmark_worker.py",
    "tinyvllm",
)
PREFIX_STATES = ("cold", "warm", "cache_cleared")
PREFIX_PROFILE_POLICY = {
    "mode": "full",
    "shared_prefix_tokens": "256,1024,2048",
    "batch_prefix_tokens": "1024,2048",
    "batch_size": 8,
    "suffix_tokens": 64,
    "repetitions": 7,
    "warmup_repetitions": 2,
    "max_model_len": 4096,
    "max_num_batched_tokens": 8192,
    "max_num_seqs": 8,
    "gpu_memory_utilization": 0.5,
    "enforce_eager": True,
}
PREFIX_ENGINE_LIMITS = {
    "max_model_len": 4096,
    "max_num_batched_tokens": 8192,
    "max_num_seqs": 8,
}
CHUNKED_ENGINE_CONFIG = {
    "max_model_len": 4352,
    "max_num_batched_tokens": 16384,
    "max_num_seqs": 512,
}
CHUNKED_POLICIES = {
    "OFF": {
        **CHUNKED_ENGINE_CONFIG,
        "max_num_prefill_tokens_per_step": 0,
        "chunked_prefill_decode_first": True,
        "chunked_prefill_max_consecutive_chunks": 0,
        "chunked_prefill_mixed_batch": False,
        "chunked_prefill_adaptive_mixed": False,
        "chunked_prefill_slo_mixed": False,
    },
    "FAIR_CHUNKED": {
        **CHUNKED_ENGINE_CONFIG,
        "max_num_prefill_tokens_per_step": 128,
        "chunked_prefill_decode_first": False,
        "chunked_prefill_max_consecutive_chunks": 2,
        "chunked_prefill_mixed_batch": False,
        "chunked_prefill_adaptive_mixed": False,
        "chunked_prefill_slo_mixed": False,
    },
}
SERVICE_CLASS_BUCKETS = (
    "short__short",
    "short__long",
    "medium__short",
    "medium__long",
    "long__short",
    "long__long",
)
REQUIRED_PRIMARY_ARTIFACTS = {
    "run_manifest.json",
    "resolved_config.json",
    "workload_manifest.jsonl",
    "request_timeline.jsonl",
    "scheduler_trace.jsonl",
    "cache_trace.jsonl",
    "memory_trace.jsonl",
    "case_rows.jsonl",
    "summary.json",
    "report.md",
    "primary_verification_receipt.json",
    "manifest.sha256",
    "source_snapshot.tar",
    "source.patch",
}
EXPECTED_CORRECTNESS_CASES = {
    "cpu_collision_and_lifecycle_preflight",
    "repeat_255",
    "repeat_256",
    "repeat_257",
    "repeat_512",
    "repeat_513",
    "same_batch_p_q_p_first",
    "same_batch_p_q_p_middle",
    "same_batch_p_q_p",
    "shared_prefix_different_suffix",
    "cache_cleared",
}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(Path(path).read_bytes())


def _canonical_sha256(value: object) -> str:
    return _sha256_bytes(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    )


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    destination = Path(path)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(destination)


def _atomic_write_json(path: Path, value: object) -> None:
    _atomic_write_bytes(path, _json_bytes(value))


def _load_json(path: Path) -> dict:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict]:
    artifact = Path(path)
    try:
        payload = artifact.read_bytes()
    except OSError as error:
        raise ValueError(f"missing JSONL artifact: {artifact}") from error
    if payload and not payload.endswith(b"\n"):
        raise ValueError(f"truncated JSONL artifact: {artifact}")
    rows = []
    for line_number, line in enumerate(payload.splitlines(), 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid JSONL row {line_number}: {artifact}"
            ) from error
        if not isinstance(value, dict):
            raise ValueError(
                f"JSONL row {line_number} must be an object: {artifact}"
            )
        rows.append(value)
    return rows


def _validate_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[0-9a-f]{64}", value) is None
    ):
        raise ValueError(f"invalid {label}")
    return value


def _finite(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be finite numeric")
    return float(value)


def _positive_int(value: object, label: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _ratio(numerator: object, denominator: object, label: str) -> float:
    top = _finite(numerator, f"{label} numerator")
    bottom = _finite(denominator, f"{label} denominator")
    if bottom <= 0.0:
        raise ValueError(f"{label} denominator must be positive")
    return top / bottom


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(_finite(value, "percentile value") for value in values)
    index = max(
        0,
        min(
            len(ordered) - 1,
            math.ceil(percentile * len(ordered)) - 1,
        ),
    )
    return ordered[index]


def _verify_artifact_hashes(run_dir: Path) -> None:
    root = Path(run_dir)
    recorded = _load_json(root / "artifact_hashes.json")
    actual = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("artifact bundle contains a symbolic link")
        if path.name.endswith(".tmp"):
            raise ValueError("finalized artifact bundle contains a temp file")
        if (
            not path.is_file()
            or path.name == "artifact_hashes.json"
        ):
            continue
        relative = path.relative_to(root).as_posix()
        actual[relative] = _sha256_path(path)
    if set(recorded) != set(actual):
        raise ValueError("artifact hash path set mismatch")
    for relative, digest in recorded.items():
        _validate_sha256(digest, f"artifact hash for {relative}")
        if actual[relative] != digest:
            raise ValueError(f"artifact hash mismatch: {relative}")
    missing = sorted(REQUIRED_PRIMARY_ARTIFACTS - set(recorded))
    if missing:
        raise ValueError(
            "required primary artifacts are missing: " + ", ".join(missing)
        )


def _safe_extract_source_snapshot(
    archive_path: Path,
    destination: Path,
    *,
    expected_files: dict[str, int],
) -> None:
    expected_paths = set(expected_files)
    expected_directories = {
        parent.as_posix()
        for relative in expected_paths
        for parent in Path(relative).parents
        if parent.as_posix() != "."
    }
    with tarfile.open(archive_path, "r:") as archive:
        members = archive.getmembers()
        names = []
        file_names = []
        for member in members:
            relative = Path(member.name)
            normalized = relative.as_posix()
            if (
                not member.name
                or normalized != member.name
                or relative.is_absolute()
                or ".." in relative.parts
                or member.issym()
                or member.islnk()
                or not (member.isdir() or member.isfile())
            ):
                raise ValueError("source snapshot contains an unsafe member")
            if not normalized or normalized in names:
                raise ValueError("source snapshot member names are invalid")
            if member.isdir():
                if normalized not in expected_directories:
                    raise ValueError("source snapshot directory is unexpected")
            elif (
                normalized not in expected_paths
                or member.size != expected_files[normalized]
            ):
                raise ValueError("source snapshot file identity is invalid")
            else:
                file_names.append(normalized)
            names.append(normalized)
        if set(file_names) != expected_paths or len(file_names) != len(
            expected_paths
        ):
            raise ValueError("source snapshot file set mismatch")
        for member in members:
            target = destination / member.name
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError("source snapshot file payload is missing")
            with target.open("xb") as handle:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)


def _verify_source(run_dir: Path, manifest: dict) -> None:
    evidence = manifest.get("source_evidence")
    if not isinstance(evidence, dict) or evidence.get("schema_version") != 1:
        raise ValueError("invalid source evidence")
    commit = evidence.get("base_commit")
    if (
        not isinstance(commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", commit) is None
        or evidence.get("local_head") != commit
        or evidence.get("tracking_head") != commit
        or manifest.get("source_commit") != commit
        or evidence.get("dirty") is not False
    ):
        raise ValueError("source commit identity mismatch")
    if evidence.get("owned_roots") != list(OWNED_SOURCE_ROOTS):
        raise ValueError("owned source roots mismatch")
    patch_path = Path(run_dir) / "source.patch"
    patch = patch_path.read_bytes()
    if (
        evidence.get("patch_path") != "source.patch"
        or evidence.get("patch_size_bytes") != len(patch)
        or evidence.get("patch_sha256") != _sha256_bytes(patch)
        or patch
    ):
        raise ValueError("source patch identity mismatch")
    records = evidence.get("files")
    if not isinstance(records, list):
        raise ValueError("source file records are missing")
    expected_paths = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("invalid source file record")
        relative = record.get("path")
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not any(
                relative == root or relative.startswith(root + "/")
                for root in OWNED_SOURCE_ROOTS
            )
        ):
            raise ValueError("invalid source file path")
        _nonnegative_int(record.get("size_bytes"), "source file size")
        _validate_sha256(record.get("sha256"), "source file hash")
        expected_paths.append(relative)
    if (
        expected_paths != sorted(expected_paths)
        or len(expected_paths) != len(set(expected_paths))
        or any(
            not any(
                path == root or path.startswith(root + "/")
                for path in expected_paths
            )
            for root in OWNED_SOURCE_ROOTS
        )
    ):
        raise ValueError("source path set mismatch")
    with TemporaryDirectory() as temporary:
        source_root = Path(temporary) / "source"
        source_root.mkdir()
        _safe_extract_source_snapshot(
            Path(run_dir) / "source_snapshot.tar",
            source_root,
            expected_files={
                record["path"]: record["size_bytes"]
                for record in records
            },
        )
        actual_paths = sorted(
            path.relative_to(source_root).as_posix()
            for path in source_root.rglob("*")
            if path.is_file()
        )
        if actual_paths != sorted(expected_paths):
            raise ValueError("source snapshot path set mismatch")
        actual_records = []
        for expected in records:
            path = source_root / expected["path"]
            payload = path.read_bytes()
            actual = {
                "path": expected["path"],
                "size_bytes": len(payload),
                "sha256": _sha256_bytes(payload),
            }
            if actual != expected:
                raise ValueError(
                    f"source file identity mismatch: {expected['path']}"
                )
            actual_records.append(actual)
    tree_sha256 = _canonical_sha256(actual_records)
    if (
        tree_sha256 != evidence.get("tree_sha256")
        or tree_sha256 != manifest.get("source_tree_sha256")
    ):
        raise ValueError("source tree identity mismatch")


def _build_prefix_workload(model_tier: str) -> list[dict]:
    rows = []
    for prefix_tokens in (256, 1024, 2048):
        for state in PREFIX_STATES:
            rows.append({
                "case_id": f"single-{prefix_tokens}__{state}",
                "gate": "prefix",
                "model_tier": model_tier,
                "shape": f"single-{prefix_tokens}",
                "state": state,
                "prefix_tokens": prefix_tokens,
                "suffix_tokens": 64,
                "batch_size": 1,
                "warmup_repetitions": 2,
                "measured_repetitions": 7,
                "enforce_eager": True,
            })
    for prefix_tokens in (1024, 2048):
        for state in PREFIX_STATES:
            rows.append({
                "case_id": f"batch8-{prefix_tokens}__{state}",
                "gate": "prefix",
                "model_tier": model_tier,
                "shape": f"batch8-{prefix_tokens}",
                "state": state,
                "prefix_tokens": prefix_tokens,
                "suffix_tokens": 64,
                "batch_size": 8,
                "warmup_repetitions": 2,
                "measured_repetitions": 7,
                "enforce_eager": True,
            })
    return rows


def _build_chunked_workload() -> list[dict]:
    rng = random.Random(20260821)
    measured_shapes = [64] * 58 + [512] * 24 + [4096] * 14
    rng.shuffle(measured_shapes)
    prompt_ordinals: dict[int, int] = {}
    rows = []
    arrival_offset_ns = 0
    for index in range(104):
        warmup = index < 8
        prompt_tokens = (
            (64, 512, 4096)[index % 3]
            if warmup
            else measured_shapes[index - 8]
        )
        ordinal = prompt_ordinals.get(prompt_tokens, 0)
        prompt_ordinals[prompt_tokens] = ordinal + 1
        output_tokens = 16 if ordinal % 2 == 0 else 64
        if index < 40:
            phase = "steady"
            arrival_offset_ns += 25_000_000
        elif index < 72:
            phase = "burst"
            arrival_offset_ns += 5_000_000
        else:
            phase = "long_injection"
            arrival_offset_ns += (
                40_000_000 if prompt_tokens == 4096 else 15_000_000
            )
        prompt_class = {64: "short", 512: "medium", 4096: "long"}[
            prompt_tokens
        ]
        rows.append({
            "schema_version": 2,
            "request_id": (
                f"{'warmup' if warmup else 'measured'}-{index:03d}"
            ),
            "warmup": warmup,
            "phase": phase,
            "arrival_offset_ns": arrival_offset_ns,
            "prompt_tokens": prompt_tokens,
            "requested_output_tokens": output_tokens,
            "service_time_bucket": (
                f"{prompt_class}__"
                f"{'short' if output_tokens == 16 else 'long'}"
            ),
            "sampling": {
                "temperature": 0.0,
                "ignore_eos": True,
                "max_tokens": output_tokens,
            },
            "starvation_deadline_ns": 30_000_000_000,
            "drain_timeout_ns": 180_000_000_000,
        })
    return rows


def _policy_identity(gate_name: str) -> dict:
    if gate_name == "prefix":
        return dict(PREFIX_PROFILE_POLICY)
    return {
        "policies": CHUNKED_POLICIES,
        "policy_order": {
            str(repetition): (
                ["OFF", "FAIR_CHUNKED"]
                if repetition % 2 == 0
                else ["FAIR_CHUNKED", "OFF"]
            )
            for repetition in range(5)
        },
    }


def _expected_cases(
    gate_name: str,
    model_tier: str,
    model_path: str,
) -> list[dict]:
    if gate_name == "prefix":
        return [{
            "case_id": f"prefix_full__{model_tier}",
            "gate": "prefix",
            "model_tier": model_tier,
            "profile_args": {
                "model": model_path,
                **PREFIX_PROFILE_POLICY,
            },
        }]
    workload_sha256 = _canonical_sha256(_build_chunked_workload())
    rows = []
    for repetition in range(5):
        order = (
            ("OFF", "FAIR_CHUNKED")
            if repetition % 2 == 0
            else ("FAIR_CHUNKED", "OFF")
        )
        for ordinal, policy in enumerate(order):
            rows.append({
                "case_id": f"{policy.lower()}__r{repetition}",
                "gate": "chunked",
                "model_tier": model_tier,
                "policy": policy,
                "repetition": repetition,
                "policy_order": ordinal,
                "engine_config": dict(CHUNKED_POLICIES[policy]),
                "workload_sha256": workload_sha256,
                "model": model_path,
                "drain_timeout_ns": 180_000_000_000,
            })
    return rows


def _verify_manifest_identity(run_dir: Path, manifest: dict) -> list[dict]:
    root = Path(run_dir)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("status") != "FINALIZED"
    ):
        raise ValueError("run manifest is not finalized schema v1")
    gate_name = manifest.get("gate")
    model_tier = manifest.get("model_tier")
    if gate_name not in {"prefix", "chunked"}:
        raise ValueError("unsupported manifest gate")
    if model_tier not in {"qwen3-0.6b", "qwen3-8b"}:
        raise ValueError("unsupported manifest model tier")
    expected_manifest_sha = (
        root / "manifest.sha256"
    ).read_text(encoding="utf-8")
    if expected_manifest_sha != _sha256_path(
        root / "run_manifest.json"
    ) + "\n":
        raise ValueError("manifest sha256 mismatch")
    _verify_source(root, manifest)
    environment = manifest.get("environment_evidence")
    if not isinstance(environment, dict):
        raise ValueError("environment evidence is missing")
    if environment.get("model_tier") != model_tier:
        raise ValueError("environment model tier mismatch")
    for field in (
        "python_version",
        "torch_version",
        "cuda_version",
        "checkpoint_identifier",
        "model_path",
    ):
        if not isinstance(environment.get(field), str) or not environment[field]:
            raise ValueError(f"invalid environment field: {field}")
    _validate_sha256(
        environment.get("model_config_sha256"),
        "model config hash",
    )
    expected_limits = (
        PREFIX_ENGINE_LIMITS
        if gate_name == "prefix"
        else CHUNKED_ENGINE_CONFIG
    )
    if environment.get("engine_limits") != expected_limits:
        raise ValueError("environment engine limits mismatch")
    inventory = environment.get("gpu_inventory")
    selected = environment.get("selected_gpu_indices")
    allowed_gpu_counts = (
        {1}
        if model_tier == "qwen3-0.6b"
        else {1, 4}
    )
    if (
        not isinstance(inventory, list)
        or len(inventory) not in allowed_gpu_counts
        or not isinstance(selected, list)
        or len(selected) != len(inventory)
    ):
        raise ValueError("invalid GPU inventory")
    indices = []
    uuids = []
    for row in inventory:
        if not isinstance(row, dict):
            raise ValueError("invalid GPU inventory row")
        indices.append(_nonnegative_int(row.get("index"), "GPU index"))
        uuid = row.get("uuid")
        name = row.get("name")
        if not isinstance(uuid, str) or not uuid:
            raise ValueError("invalid GPU UUID")
        if not isinstance(name, str) or not name:
            raise ValueError("invalid GPU name")
        uuids.append(uuid)
    if (
        selected != indices
        or len(indices) != len(set(indices))
        or len(uuids) != len(set(uuids))
    ):
        raise ValueError("selected GPU inventory mismatch")
    if manifest.get("environment_sha256") != _canonical_sha256(environment):
        raise ValueError("environment hash mismatch")
    workload = _load_jsonl(root / "workload_manifest.jsonl")
    expected_workload = (
        _build_prefix_workload(model_tier)
        if gate_name == "prefix"
        else _build_chunked_workload()
    )
    if (
        workload != expected_workload
        or manifest.get("workload_sha256")
        != _canonical_sha256(workload)
    ):
        raise ValueError("workload identity mismatch")
    policy = _policy_identity(gate_name)
    if manifest.get("policy_sha256") != _canonical_sha256(policy):
        raise ValueError("policy identity mismatch")
    cases = _expected_cases(
        gate_name,
        model_tier,
        environment["model_path"],
    )
    case_order = [case["case_id"] for case in cases]
    case_paths = {
        case_id: f"case_specs/{case_id}.json"
        for case_id in case_order
    }
    if (
        manifest.get("case_order") != case_order
        or manifest.get("case_specs") != case_paths
    ):
        raise ValueError("case matrix identity mismatch")
    loaded_cases = [
        _load_json(root / case_paths[case_id])
        for case_id in case_order
    ]
    if loaded_cases != cases:
        raise ValueError("case specification identity mismatch")
    resolved = _load_json(root / "resolved_config.json")
    if resolved != {
        "gate": gate_name,
        "model_tier": model_tier,
        "model_path": environment["model_path"],
        "policy": policy,
        "cases": cases,
        "environment": environment,
    }:
        raise ValueError("resolved configuration identity mismatch")
    return workload


def _verify_process_receipt(
    path: Path,
    case_id: str,
    used_ports: set[int],
) -> None:
    process = _load_json(path)
    if (
        process.get("case_id") != case_id
        or process.get("returncode") != 0
    ):
        raise ValueError(f"case process failed or mismatched: {case_id}")
    exitcode = path.with_name("exitcode").read_text(encoding="utf-8")
    if exitcode != "0\n":
        raise ValueError(f"case exitcode mismatch: {case_id}")
    for field in ("master_port", "distributed_port"):
        port = _positive_int(process.get(field), f"{case_id} {field}")
        if port < 1024 or port > 65535:
            raise ValueError("distributed port is outside the valid range")
        if port in used_ports:
            raise ValueError("distributed ports are not globally unique")
        used_ports.add(port)


def _validate_chunked_traces(
    case: dict,
    scheduler: list[dict],
    memory: list[dict],
) -> None:
    if not scheduler:
        raise ValueError("chunked scheduler trace is empty")
    scheduler_steps = [
        _nonnegative_int(row.get("step_index"), "scheduler step index")
        for row in scheduler
    ]
    memory_steps = [
        _nonnegative_int(row.get("step_index"), "memory step index")
        for row in memory
    ]
    expected_steps = list(range(len(scheduler)))
    if scheduler_steps != expected_steps:
        raise ValueError("chunked scheduler steps are not contiguous")
    if memory_steps != scheduler_steps:
        raise ValueError("chunked scheduler and memory traces are not aligned")
    for row in scheduler:
        if (
            "policy" in row
            and row.get("policy") != case["policy"]
        ):
            raise ValueError("chunked scheduler policy mismatch")


def _prefix_identity(row: dict) -> tuple[str, str, int]:
    if row.get("schema_version") != 2:
        raise ValueError("Prefix raw schema mismatch")
    shape = row.get("shape")
    state = row.get("state")
    repetition = row.get("repetition")
    if (
        shape not in {
            "single-256",
            "single-1024",
            "single-2048",
            "batch8-1024",
            "batch8-2048",
        }
        or state not in PREFIX_STATES
        or isinstance(repetition, bool)
        or not isinstance(repetition, int)
        or repetition not in range(7)
        or row.get("warmup") is not False
        or row.get("case_id")
        != f"{shape}__{state}__r{repetition}"
    ):
        raise ValueError("Prefix raw identity mismatch")
    return shape, state, repetition


def _prefix_state_summary(
    rows: list[dict],
    cache_by_identity: dict[tuple[str, str, int], dict],
    memory_by_identity: dict[tuple[str, str, int], dict],
) -> dict:
    elapsed = []
    cached = []
    query = []
    batches = []
    max_abs = []
    mean_abs = []
    argmax = []
    correct = []
    retained_blocks = []
    retained_bytes = []
    clear_ns = []
    cuda_reserved = []
    for row in rows:
        identity = _prefix_identity(row)
        cache = cache_by_identity[identity]
        memory = memory_by_identity[identity]
        for field in (
            "cached_prompt_tokens",
            "executed_query_tokens",
            "retained_reusable_blocks",
            "retained_logical_kv_bytes",
            "cache_clear_host_ns",
        ):
            if row.get(field) != cache.get(field):
                raise ValueError("Prefix performance/cache mismatch")
        if (
            row.get("retained_logical_kv_bytes")
            != memory.get("retained_logical_kv_bytes")
            or row.get("cuda_peak_reserved_bytes")
            != memory.get("cuda_peak_reserved_bytes")
        ):
            raise ValueError("Prefix performance/memory mismatch")
        logit = row.get("logit")
        if not isinstance(logit, dict):
            raise ValueError("Prefix logit evidence is missing")
        ttft_ns = _positive_int(row.get("ttft_ns"), "Prefix TTFT")
        elapsed.append(ttft_ns / 1_000_000.0)
        cached.append(
            _nonnegative_int(
                row.get("cached_prompt_tokens"),
                "Prefix cached tokens",
            )
        )
        query.append(
            _positive_int(
                row.get("executed_query_tokens"),
                "Prefix query tokens",
            )
        )
        batches.append(
            _positive_int(row.get("model_batches"), "Prefix model batches")
        )
        max_abs.append(_finite(logit.get("max_abs"), "Prefix logit max"))
        mean_abs.append(_finite(logit.get("mean_abs"), "Prefix logit mean"))
        argmax.append(logit.get("argmax_match") is True)
        correct.append(row.get("correct") is True)
        retained_blocks.append(
            _nonnegative_int(
                cache.get("retained_reusable_blocks"),
                "Prefix retained blocks",
            )
        )
        retained_bytes.append(
            _nonnegative_int(
                cache.get("retained_logical_kv_bytes"),
                "Prefix retained bytes",
            )
        )
        clear_ns.append(
            _nonnegative_int(
                cache.get("cache_clear_host_ns"),
                "Prefix cache-clear time",
            )
        )
        cuda_reserved.append(
            _nonnegative_int(
                memory.get("cuda_peak_reserved_bytes"),
                "Prefix CUDA reserved",
            )
        )
    return {
        "samples": len(rows),
        "median_elapsed_ms": statistics.median(elapsed),
        "p95_elapsed_ms": _percentile(elapsed, 0.95),
        "median_cached_prompt_tokens": statistics.median(cached),
        "median_executed_query_tokens": statistics.median(query),
        "median_model_batches": statistics.median(batches),
        "peak_cuda_reserved_bytes": max(cuda_reserved),
        "exact_outputs": all(correct),
        "logit_argmax_match": all(argmax),
        "logit_max_abs": max(max_abs),
        "logit_mean_abs": max(mean_abs),
        "_retained_blocks": max(retained_blocks),
        "_retained_bytes": max(retained_bytes),
        "_clear_ms": statistics.median(clear_ns) / 1_000_000.0,
    }


def _validate_prefix_logit_diff(
    value: object,
    label: str,
    *,
    require_match: bool,
) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{label} is missing")
    max_abs = _finite(value.get("max_abs"), f"{label} max_abs")
    mean_abs = _finite(value.get("mean_abs"), f"{label} mean_abs")
    reference_argmax = _nonnegative_int(
        value.get("reference_argmax"),
        f"{label} reference argmax",
    )
    candidate_argmax = _nonnegative_int(
        value.get("candidate_argmax"),
        f"{label} candidate argmax",
    )
    if max_abs < 0.0 or mean_abs < 0.0:
        raise ValueError(f"{label} differences must be nonnegative")
    if require_match and (
        value.get("argmax_match") is not True
        or value.get("within_tolerance") is not True
        or reference_argmax != candidate_argmax
        or max_abs > 0.25
        or mean_abs > 0.05
    ):
        raise ValueError(f"{label} does not prove output equivalence")


def _validate_prefix_correctness(rows: list[dict]) -> None:
    by_case = {row.get("case"): row for row in rows}
    if (
        len(by_case) != len(rows)
        or set(by_case) != EXPECTED_CORRECTNESS_CASES
    ):
        raise ValueError("Prefix correctness evidence is incomplete")
    preflight = by_case["cpu_collision_and_lifecycle_preflight"]
    command = preflight.get("command")
    if (
        preflight.get("state") != "preflight"
        or preflight.get("returncode") != 0
        or preflight.get("correct") is not True
        or not isinstance(command, list)
        or not command
        or any(not isinstance(part, str) or not part for part in command)
        or not isinstance(preflight.get("stdout"), str)
        or not isinstance(preflight.get("stderr"), str)
    ):
        raise ValueError("Prefix CPU preflight evidence is invalid")
    expected_rows = {}
    for prompt_tokens in (255, 256, 257, 512, 513):
        cached_tokens = ((prompt_tokens - 1) // 256) * 256
        expected_rows[f"repeat_{prompt_tokens}"] = {
            "state": "warm",
            "prompt_tokens": prompt_tokens,
            "cached_tokens": cached_tokens,
            "query_tokens": prompt_tokens - cached_tokens,
            "expected_reusable_tokens": cached_tokens,
        }
    for case in (
        "same_batch_p_q_p_first",
        "same_batch_p_q_p_middle",
        "same_batch_p_q_p",
    ):
        expected_rows[case] = {
            "state": "same_batch",
            "prompt_tokens": 513,
            "cached_tokens": 0,
            "query_tokens": 513,
        }
    expected_rows["shared_prefix_different_suffix"] = {
        "state": "warm",
        "prompt_tokens": 320,
        "cached_tokens": 256,
        "query_tokens": 64,
        "expected_reusable_tokens": 256,
    }
    expected_rows["cache_cleared"] = {
        "state": "cache_cleared",
        "prompt_tokens": 320,
        "cached_tokens": 0,
        "query_tokens": 320,
        "expected_reusable_tokens": 0,
    }
    for case, expected in expected_rows.items():
        row = by_case[case]
        if (
            any(row.get(field) != value for field, value in expected.items())
            or row.get("correct") is not True
            or not isinstance(row.get("decoded"), str)
        ):
            raise ValueError(f"Prefix correctness row is invalid: {case}")
        _nonnegative_int(row.get("token_id"), f"{case} token ID")
        _validate_prefix_logit_diff(
            row.get("logit_diff"),
            f"{case} logit diff",
            require_match=True,
        )
    distinct = by_case["same_batch_p_q_p"].get("batch_q_logit_diff")
    _validate_prefix_logit_diff(
        distinct,
        "same-batch P/Q logit diff",
        require_match=False,
    )
    if _finite(
        distinct.get("max_abs"),
        "same-batch P/Q max_abs",
    ) <= 0.0:
        raise ValueError("same-batch P/Q prompts are not distinct")


def _rebuild_prefix_bundle(merged: dict[str, list[dict]]) -> dict:
    correctness = merged["request_timeline.jsonl"]
    _validate_prefix_correctness(correctness)
    expected_identities = {
        (shape, state, repetition)
        for shape in (
            "single-256",
            "single-1024",
            "single-2048",
            "batch8-1024",
            "batch8-2048",
        )
        for state in PREFIX_STATES
        for repetition in range(7)
    }
    identities = {}
    for filename in (
        "scheduler_trace.jsonl",
        "cache_trace.jsonl",
        "memory_trace.jsonl",
    ):
        values = [_prefix_identity(row) for row in merged[filename]]
        if (
            len(values) != len(expected_identities)
            or set(values) != expected_identities
        ):
            raise ValueError(f"Prefix raw matrix mismatch: {filename}")
        identities[filename] = values
    if not (
        identities["scheduler_trace.jsonl"]
        == identities["cache_trace.jsonl"]
        == identities["memory_trace.jsonl"]
    ):
        raise ValueError("Prefix raw files are not aligned")
    performance = merged["scheduler_trace.jsonl"]
    cache_by_identity = {
        _prefix_identity(row): row
        for row in merged["cache_trace.jsonl"]
    }
    memory_by_identity = {
        _prefix_identity(row): row
        for row in merged["memory_trace.jsonl"]
    }
    families = {"single": {}, "batch": {}}
    for shape in (
        "single-256",
        "single-1024",
        "single-2048",
        "batch8-1024",
        "batch8-2048",
    ):
        family = "single" if shape.startswith("single-") else "batch"
        prefix_tokens = int(shape.rsplit("-", 1)[1])
        batch_size = 1 if family == "single" else 8
        shape_rows = [row for row in performance if row["shape"] == shape]
        if any(
            row.get("shared_prefix_tokens") != prefix_tokens
            or row.get("suffix_tokens") != 64
            or row.get("batch_size") != batch_size
            for row in shape_rows
        ):
            raise ValueError("Prefix shape evidence mismatch")
        states = {}
        for state_name in PREFIX_STATES:
            states[state_name] = _prefix_state_summary(
                [
                    row for row in shape_rows
                    if row["state"] == state_name
                ],
                cache_by_identity,
                memory_by_identity,
            )
        retained_blocks = max(
            state.pop("_retained_blocks") for state in states.values()
        )
        retained_bytes = max(
            state.pop("_retained_bytes") for state in states.values()
        )
        clear_ms = max(
            state.pop("_clear_ms") for state in states.values()
        )
        families[family][str(prefix_tokens)] = {
            "prefix_tokens": prefix_tokens,
            "suffix_tokens": 64,
            "batch_size": batch_size,
            "expected_reusable_tokens": prefix_tokens * batch_size,
            **states,
            "retained_reusable_blocks": retained_blocks,
            "retained_logical_kv_bytes": retained_bytes,
            "median_cache_clear_host_ms": clear_ms,
        }
    return {
        "artifact_complete": True,
        "single": families["single"],
        "batch": families["batch"],
    }


def _classify_prefix(raw: dict) -> dict:
    structural = []
    correctness = []
    performance = []
    shapes = []
    if raw.get("artifact_complete") is not True:
        structural.append("prefix artifacts are incomplete")
    for family, expected in (
        ("single", ("256", "1024", "2048")),
        ("batch", ("1024", "2048")),
    ):
        values = raw.get(family)
        if (
            not isinstance(values, dict)
            or tuple(sorted(values, key=int)) != expected
        ):
            structural.append(f"invalid prefix {family} shapes")
            continue
        shapes.extend(
            (f"{family}-{prefix}", values[prefix])
            for prefix in expected
        )
    single_improvements = []
    batch_improvements = []
    cuda_regressions = []
    protected_regressions = []
    retained_blocks = 0
    retained_bytes = 0
    for shape_name, shape in shapes:
        prefix_tokens = int(shape_name.rsplit("-", 1)[1])
        batch_size = 1 if shape_name.startswith("single-") else 8
        expected_reusable = prefix_tokens * batch_size
        expected_cold_query = (prefix_tokens + 64) * batch_size
        expected_warm_query = 64 * batch_size
        if (
            shape.get("prefix_tokens") != prefix_tokens
            or shape.get("suffix_tokens") != 64
            or shape.get("batch_size") != batch_size
            or shape.get("expected_reusable_tokens") != expected_reusable
        ):
            structural.append(f"{shape_name}: shape identity mismatch")
            continue
        retained_blocks = max(
            retained_blocks,
            int(shape.get("retained_reusable_blocks", 0)),
        )
        retained_bytes = max(
            retained_bytes,
            int(shape.get("retained_logical_kv_bytes", 0)),
        )
        states = {}
        for state_name in PREFIX_STATES:
            row = shape.get(state_name)
            if not isinstance(row, dict) or row.get("samples") != 7:
                structural.append(
                    f"{shape_name} {state_name}: missing seven samples"
                )
                continue
            states[state_name] = row
            if (
                row.get("exact_outputs") is not True
                or row.get("logit_argmax_match") is not True
                or _finite(
                    row.get("logit_max_abs"),
                    f"{shape_name} {state_name} logit_max_abs",
                ) > 0.25
                or _finite(
                    row.get("logit_mean_abs"),
                    f"{shape_name} {state_name} logit_mean_abs",
                ) > 0.05
            ):
                correctness.append(
                    f"{shape_name} {state_name}: output or logit mismatch"
                )
        if set(states) != set(PREFIX_STATES):
            continue
        cold = states["cold"]
        warm = states["warm"]
        cleared = states["cache_cleared"]
        if not (
            cold.get("median_cached_prompt_tokens") == 0
            and cold.get("median_executed_query_tokens")
            == expected_cold_query
            and warm.get("median_cached_prompt_tokens")
            == expected_reusable
            and warm.get("median_executed_query_tokens")
            == expected_warm_query
            and cleared.get("median_cached_prompt_tokens") == 0
            and cleared.get("median_executed_query_tokens")
            == expected_cold_query
        ):
            correctness.append(
                f"{shape_name}: cached or query token accounting mismatch"
            )
        cold_elapsed = _finite(
            cold.get("median_elapsed_ms"),
            f"{shape_name} cold elapsed",
        )
        improvement = 1.0 - _ratio(
            warm.get("median_elapsed_ms"),
            cold_elapsed,
            f"{shape_name} warm elapsed",
        )
        cleared_ratio = _ratio(
            cleared.get("median_elapsed_ms"),
            cold_elapsed,
            f"{shape_name} cache-cleared elapsed",
        )
        cuda_ratio = _ratio(
            warm.get("peak_cuda_reserved_bytes"),
            cold.get("peak_cuda_reserved_bytes"),
            f"{shape_name} CUDA reserved",
        )
        cuda_regressions.append(cuda_ratio - 1.0)
        protected_regressions.extend((
            cleared_ratio - 1.0,
            cuda_ratio - 1.0,
        ))
        if cleared_ratio > 1.05:
            performance.append(
                f"{shape_name}: cache-cleared regression exceeds 5%"
            )
        if cuda_ratio > 1.05:
            performance.append(
                f"{shape_name}: CUDA reserved regression exceeds 5%"
            )
        if shape_name in ("single-1024", "single-2048"):
            single_improvements.append(improvement)
            if improvement < 0.20:
                performance.append(
                    f"{shape_name}: warm median TTFT improvement below 20%"
                )
        if shape_name.startswith("batch-"):
            batch_improvements.append(improvement)
            if warm.get("median_model_batches") != 1:
                performance.append(
                    f"{shape_name}: warm consumers require multiple batches"
                )
            if improvement < 0.15:
                performance.append(
                    f"{shape_name}: warm batch elapsed improvement below 15%"
                )
    if structural or correctness:
        classification = "PREFIX_CACHE_INCOMPLETE_OR_INCORRECT"
    elif performance:
        classification = "PREFIX_CACHE_NO_GO"
    else:
        classification = "PREFIX_CACHE_GO"
    return {
        "classification": classification,
        "structural_failures": sorted(set(structural)),
        "correctness_failures": sorted(set(correctness)),
        "performance_failures": sorted(set(performance)),
        "benefit": {
            "minimum_primary_improvement_fraction": round(
                min(single_improvements, default=0.0),
                12,
            ),
            "minimum_batch_improvement_fraction": round(
                min(batch_improvements, default=0.0),
                12,
            ),
        },
        "cost": {
            "worst_protected_metric_regression_fraction": round(
                max(protected_regressions, default=0.0),
                12,
            ),
            "maximum_cuda_reserved_regression_fraction": round(
                max(cuda_regressions, default=0.0),
                12,
            ),
            "maximum_retained_reusable_blocks": retained_blocks,
            "maximum_retained_logical_kv_bytes": retained_bytes,
        },
    }


def _validate_chunked_timeline(
    timeline: list[dict],
    workload: list[dict],
) -> None:
    expected = {row["request_id"]: row for row in workload}
    observed = [row.get("request_id") for row in timeline]
    expected_order = [row["request_id"] for row in workload]
    if observed != expected_order:
        raise ValueError("chunked request identity mismatch")
    sequence_ids = []
    case_epochs = []
    for row in timeline:
        source = expected[row["request_id"]]
        if (
            row.get("warmup") is not source["warmup"]
            or row.get("phase") != source["phase"]
            or row.get("prompt_token_count") != source["prompt_tokens"]
            or row.get("requested_output_tokens")
            != source["requested_output_tokens"]
            or row.get("service_time_bucket")
            != source["service_time_bucket"]
            or row.get("starvation_deadline_ns")
            != source["starvation_deadline_ns"]
        ):
            raise ValueError("chunked request identity mismatch")
        scheduled = _nonnegative_int(
            row.get("scheduled_arrival_ns"),
            "scheduled arrival",
        )
        arrival_offset = _nonnegative_int(
            source.get("arrival_offset_ns"),
            "workload arrival offset",
        )
        case_epochs.append(
            _nonnegative_int(
                scheduled - arrival_offset,
                "chunked case epoch",
            )
        )
        sequence_ids.append(
            _nonnegative_int(row.get("seq_id"), "chunked sequence ID")
        )
        actual = _nonnegative_int(row.get("actual_arrival_ns"), "arrival")
        first_scheduled = _nonnegative_int(
            row.get("first_scheduled_ns"),
            "first schedule",
        )
        first_token = _nonnegative_int(
            row.get("first_token_ns"),
            "first token",
        )
        completion = _nonnegative_int(
            row.get("completion_ns"),
            "completion",
        )
        timestamps = row.get("token_timestamps_ns")
        tokens = row.get("output_token_ids")
        if (
            not isinstance(timestamps, list)
            or not isinstance(tokens, list)
            or len(tokens) != source["requested_output_tokens"]
            or len(timestamps) != len(tokens)
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in tokens
            )
            or timestamps != sorted(timestamps)
            or scheduled > actual
            or actual > first_scheduled
            or first_scheduled > first_token
            or timestamps[0] != first_token
            or timestamps[-1] != completion
            or row.get("finish_reason") != "length"
            or row.get("error") is not None
        ):
            raise ValueError("chunked lifecycle ordering mismatch")
    if len(set(sequence_ids)) != len(sequence_ids):
        raise ValueError("chunked sequence IDs are not unique")
    if len(set(case_epochs)) != 1:
        raise ValueError("chunked arrivals do not share one workload epoch")


def _chunked_case_metrics(
    case: dict,
    timeline: list[dict],
    workload: list[dict],
    memory: list[dict],
    case_result: dict,
) -> tuple[dict, dict[str, list[int]]]:
    _validate_chunked_timeline(timeline, workload)
    measured = [row for row in timeline if row["warmup"] is False]
    complete = list(measured)
    outputs = {
        row["request_id"]: list(row["output_token_ids"])
        for row in measured
    }
    ttft_short = [
        row["first_token_ns"] - row["scheduled_arrival_ns"]
        for row in complete
        if row["service_time_bucket"].startswith("short__")
    ]
    short_itl = []
    all_itl = []
    for row in complete:
        gaps = [
            right - left
            for left, right in zip(
                row["token_timestamps_ns"],
                row["token_timestamps_ns"][1:],
            )
        ]
        if any(gap < 0 for gap in gaps):
            raise ValueError("negative inter-token latency")
        all_itl.extend(gaps)
        if row["service_time_bucket"].startswith("short__"):
            short_itl.extend(gaps)
    completions = {
        bucket: [
            row["completion_ns"] - row["scheduled_arrival_ns"]
            for row in complete
            if row["service_time_bucket"] == bucket
        ]
        for bucket in SERVICE_CLASS_BUCKETS
    }
    long_completion = [
        row["completion_ns"] - row["scheduled_arrival_ns"]
        for row in complete
        if row["service_time_bucket"].startswith("long__")
    ]
    start = min(row["scheduled_arrival_ns"] for row in complete)
    finish = max(row["completion_ns"] for row in complete)
    duration_s = (finish - start) / 1_000_000_000
    if duration_s <= 0:
        raise ValueError("chunked measured duration must be positive")
    if not memory:
        raise ValueError("chunked memory trace is empty")
    peak_reserved = max(
        _nonnegative_int(
            row.get("cuda_peak_reserved_bytes"),
            "chunked CUDA reserved",
        )
        for row in memory
    )
    if (
        case_result.get("case_id") != case["case_id"]
        or case_result.get("status") != "PASS"
        or case_result.get("error_type") is not None
        or case_result.get("error") is not None
        or case_result.get("request_count") != len(workload)
        or case_result.get("completed_request_count") != len(workload)
        or case_result.get("step_count") != len(memory)
    ):
        raise ValueError("chunked case result mismatch")
    return {
        "case_id": case["case_id"],
        "policy": case["policy"],
        "repetition": case["repetition"],
        "short_p99_ttft_ns": _percentile(ttft_short, 0.99),
        "short_p99_itl_ns": _percentile(short_itl, 0.99),
        "maximum_decode_gap_ns": max(all_itl),
        "service_class_p95_completion_ns": {
            bucket: _percentile(values, 0.95)
            for bucket, values in completions.items()
        },
        "long_p95_completion_ns": _percentile(
            long_completion,
            0.95,
        ),
        "request_throughput_rps": len(complete) / duration_s,
        "output_token_throughput_tps": (
            sum(len(row["output_token_ids"]) for row in complete)
            / duration_s
        ),
        "peak_cuda_reserved_bytes": peak_reserved,
        "exact_outputs": True,
        "complete_lifecycle": len(complete) == 96,
        "dropped_requests": max(0, 96 - len(measured)),
        "rejected_requests": 0,
        "truncated_requests": 0,
        "unfinished_requests": max(0, 96 - len(complete)),
        "starved_requests": 0,
    }, outputs


def _classify_chunked(raw: dict) -> dict:
    structural = []
    correctness = []
    performance = []
    repetitions = raw.get("repetitions")
    if raw.get("artifact_complete") is not True:
        structural.append("chunked artifacts are incomplete")
    if (
        not isinstance(repetitions, list)
        or len(repetitions) != 5
        or {
            row.get("repetition")
            for row in repetitions
            if isinstance(row, dict)
        } != set(range(5))
    ):
        structural.append("chunked repetition matrix is invalid")
        repetitions = []
    improvements = []
    favorable = 0
    protected = []
    cuda_regressions = []
    for repetition in repetitions:
        repetition_id = repetition["repetition"]
        baseline = repetition.get("OFF")
        candidate = repetition.get("FAIR_CHUNKED")
        if not isinstance(baseline, dict) or not isinstance(candidate, dict):
            structural.append(f"r{repetition_id}: missing policy row")
            continue
        for policy, row in (
            ("OFF", baseline),
            ("FAIR_CHUNKED", candidate),
        ):
            for field in (
                "short_p99_ttft_ns",
                "short_p99_itl_ns",
                "maximum_decode_gap_ns",
                "long_p95_completion_ns",
                "request_throughput_rps",
                "output_token_throughput_tps",
                "peak_cuda_reserved_bytes",
            ):
                _finite(row.get(field), f"{policy} {field}")
            buckets = row.get("service_class_p95_completion_ns")
            if not isinstance(buckets, dict) or set(buckets) != set(
                SERVICE_CLASS_BUCKETS
            ):
                raise ValueError("invalid chunked service classes")
            if (
                row.get("exact_outputs") is not True
                or row.get("complete_lifecycle") is not True
                or any(
                    row.get(field) != 0
                    for field in (
                        "dropped_requests",
                        "rejected_requests",
                        "truncated_requests",
                        "unfinished_requests",
                        "starved_requests",
                    )
                )
            ):
                correctness.append(
                    f"r{repetition_id} {policy}: lifecycle or output failure"
                )
        ttft_ratio = _ratio(
            candidate["short_p99_ttft_ns"],
            baseline["short_p99_ttft_ns"],
            f"r{repetition_id} short p99 TTFT",
        )
        improvement = 1.0 - ttft_ratio
        improvements.append(improvement)
        if improvement >= 0.10:
            favorable += 1
        itl_ratio = _ratio(
            candidate["short_p99_itl_ns"],
            baseline["short_p99_itl_ns"],
            f"r{repetition_id} short p99 ITL",
        )
        gap_ratio = _ratio(
            candidate["maximum_decode_gap_ns"],
            baseline["maximum_decode_gap_ns"],
            f"r{repetition_id} maximum decode gap",
        )
        long_ratio = _ratio(
            candidate["long_p95_completion_ns"],
            baseline["long_p95_completion_ns"],
            f"r{repetition_id} long p95 completion",
        )
        request_ratio = _ratio(
            candidate["request_throughput_rps"],
            baseline["request_throughput_rps"],
            f"r{repetition_id} request throughput",
        )
        token_ratio = _ratio(
            candidate["output_token_throughput_tps"],
            baseline["output_token_throughput_tps"],
            f"r{repetition_id} token throughput",
        )
        cuda_ratio = _ratio(
            candidate["peak_cuda_reserved_bytes"],
            baseline["peak_cuda_reserved_bytes"],
            f"r{repetition_id} CUDA reserved",
        )
        bucket_ratios = {
            bucket: _ratio(
                candidate["service_class_p95_completion_ns"][bucket],
                baseline["service_class_p95_completion_ns"][bucket],
                f"r{repetition_id} {bucket} p95 completion",
            )
            for bucket in SERVICE_CLASS_BUCKETS
        }
        protected.extend((
            itl_ratio - 1.0,
            gap_ratio - 1.0,
            long_ratio - 1.0,
            1.0 - request_ratio,
            1.0 - token_ratio,
            cuda_ratio - 1.0,
            *(ratio - 1.0 for ratio in bucket_ratios.values()),
        ))
        cuda_regressions.append(cuda_ratio - 1.0)
        if itl_ratio > 1.05:
            performance.append(
                f"r{repetition_id}: short p99 ITL regression exceeds 5%"
            )
        if gap_ratio > 1.10:
            performance.append(
                f"r{repetition_id}: maximum decode gap regression exceeds 10%"
            )
        if long_ratio > 1.10:
            performance.append(
                f"r{repetition_id}: long p95 completion regression exceeds 10%"
            )
        for bucket, ratio in bucket_ratios.items():
            if ratio > 1.10:
                performance.append(
                    f"r{repetition_id} {bucket}: p95 completion regression exceeds 10%"
                )
        if request_ratio < 0.97:
            performance.append(
                f"r{repetition_id}: request throughput regression exceeds 3%"
            )
        if token_ratio < 0.97:
            performance.append(
                f"r{repetition_id}: token throughput regression exceeds 3%"
            )
        if cuda_ratio > 1.05:
            performance.append(
                f"r{repetition_id}: CUDA reserved regression exceeds 5%"
            )
    if repetitions and favorable < 4:
        performance.append(
            "short p99 TTFT benefit direction is absent in four of five repetitions"
        )
    if structural or correctness:
        classification = "FAIR_CHUNKED_INCOMPLETE"
    elif performance:
        classification = "FAIR_CHUNKED_NO_GO"
    else:
        classification = "FAIR_CHUNKED_GO"
    return {
        "classification": classification,
        "structural_failures": sorted(set(structural)),
        "correctness_failures": sorted(set(correctness)),
        "performance_failures": sorted(set(performance)),
        "benefit": {
            "short_p99_ttft_improvement_fraction": round(
                statistics.median(improvements) if improvements else 0.0,
                12,
            ),
            "favorable_repetitions": favorable,
        },
        "cost": {
            "worst_protected_metric_regression_fraction": round(
                max(protected, default=0.0),
                12,
            ),
            "maximum_cuda_reserved_regression_fraction": round(
                max(cuda_regressions, default=0.0),
                12,
            ),
        },
    }


def _verify_prefix(
    run_dir: Path,
    manifest: dict,
) -> dict:
    root = Path(run_dir)
    case_id = manifest["case_order"][0]
    case_dir = root / "cases" / case_id
    used_ports: set[int] = set()
    _verify_process_receipt(
        case_dir / "process.json",
        case_id,
        used_ports,
    )
    output = case_dir / "output"
    mapping = {
        "request_timeline.jsonl": "prefix_correctness_rows.jsonl",
        "scheduler_trace.jsonl": "prefix_performance_rows.jsonl",
        "cache_trace.jsonl": "prefix_cache_rows.jsonl",
        "memory_trace.jsonl": "prefix_memory_rows.jsonl",
    }
    merged = {}
    for destination, source in mapping.items():
        case_rows = _load_jsonl(output / source)
        primary_rows = _load_jsonl(root / destination)
        if primary_rows != case_rows:
            raise ValueError(f"Prefix merged raw mismatch: {destination}")
        merged[destination] = primary_rows
    bundle = _rebuild_prefix_bundle(merged)
    worker = _load_json(output / "summary.json")
    if worker.get("staged_contract_bundle") != bundle:
        raise ValueError("Prefix worker summary disagrees with raw evidence")
    case_rows = _load_jsonl(root / "case_rows.jsonl")
    if case_rows != merged["scheduler_trace.jsonl"]:
        raise ValueError("Prefix case rows mismatch")
    return _classify_prefix(bundle)


def _verify_chunked(
    run_dir: Path,
    manifest: dict,
    workload: list[dict],
) -> dict:
    root = Path(run_dir)
    merged = {
        "request_timeline.jsonl": [],
        "scheduler_trace.jsonl": [],
        "cache_trace.jsonl": [],
        "memory_trace.jsonl": [],
    }
    metrics_by_repetition: dict[int, dict[str, dict]] = {}
    outputs_by_repetition: dict[int, dict[str, dict[str, list[int]]]] = {}
    case_rows = []
    used_ports: set[int] = set()
    for case_id in manifest["case_order"]:
        case = _load_json(root / manifest["case_specs"][case_id])
        case_dir = root / "cases" / case_id
        _verify_process_receipt(
            case_dir / "process.json",
            case_id,
            used_ports,
        )
        output = case_dir / "output"
        timeline = _load_jsonl(output / "request_timeline.jsonl")
        scheduler = _load_jsonl(output / "scheduler_trace.jsonl")
        memory = _load_jsonl(output / "memory_trace.jsonl")
        _validate_chunked_traces(case, scheduler, memory)
        case_result = _load_json(output / "case_result.json")
        identity = {
            "case_id": case_id,
            "policy": case["policy"],
            "repetition": case["repetition"],
        }
        merged["request_timeline.jsonl"].extend(
            {**identity, **row} for row in timeline
        )
        merged["scheduler_trace.jsonl"].extend(
            {**identity, **row} for row in scheduler
        )
        merged["memory_trace.jsonl"].extend(
            {**identity, **row} for row in memory
        )
        metrics, outputs = _chunked_case_metrics(
            case,
            timeline,
            workload,
            memory,
            case_result,
        )
        case_rows.append(metrics)
        metrics_by_repetition.setdefault(case["repetition"], {})[
            case["policy"]
        ] = metrics
        outputs_by_repetition.setdefault(case["repetition"], {})[
            case["policy"]
        ] = outputs
    for filename, expected_rows in merged.items():
        if _load_jsonl(root / filename) != expected_rows:
            raise ValueError(f"chunked merged raw mismatch: {filename}")
    if _load_jsonl(root / "case_rows.jsonl") != case_rows:
        raise ValueError("chunked case rows mismatch")
    repetitions = []
    for repetition in range(5):
        policies = metrics_by_repetition.get(repetition, {})
        outputs = outputs_by_repetition.get(repetition, {})
        if set(policies) != {"OFF", "FAIR_CHUNKED"}:
            raise ValueError("chunked policy matrix mismatch")
        exact = outputs["OFF"] == outputs["FAIR_CHUNKED"]
        policies["OFF"]["exact_outputs"] = exact
        policies["FAIR_CHUNKED"]["exact_outputs"] = exact
        repetitions.append({
            "repetition": repetition,
            **policies,
        })
    return _classify_chunked({
        "artifact_complete": True,
        "repetitions": repetitions,
    })


def _render_report(manifest: dict, summary: dict) -> str:
    benefit = summary.get("benefit", {})
    cost = summary.get("cost", {})
    return "\n".join((
        f"# {manifest['run_tag']}",
        "",
        f"- Gate: `{manifest['gate']}`",
        f"- Model tier: `{manifest['model_tier']}`",
        f"- Classification: `{summary.get('classification')}`",
        "",
        "| Benefit | Cost |",
        "| --- | --- |",
        (
            f"| `{json.dumps(benefit, sort_keys=True)}` "
            f"| `{json.dumps(cost, sort_keys=True)}` |"
        ),
        "",
    ))


def _verify_primary_outputs(
    run_dir: Path,
    manifest: dict,
    summary: dict,
) -> None:
    root = Path(run_dir)
    if manifest.get("classification") != summary.get("classification"):
        raise ValueError("manifest classification mismatch")
    if (root / "summary.json").read_bytes() != _json_bytes(summary):
        raise ValueError("primary summary disagrees with independent rebuild")
    report = _render_report(manifest, summary).encode("utf-8")
    if (root / "report.md").read_bytes() != report:
        raise ValueError("primary report disagrees with independent rebuild")
    receipt = _load_json(root / "primary_verification_receipt.json")
    expected_receipt = {
        "schema_version": 1,
        "status": "PASS",
        "classification": summary["classification"],
        "case_count": len(manifest["case_order"]),
        "source_tree_sha256": manifest["source_tree_sha256"],
        "environment_sha256": manifest["environment_sha256"],
        "workload_sha256": manifest["workload_sha256"],
        "policy_sha256": manifest["policy_sha256"],
    }
    if receipt != expected_receipt:
        raise ValueError("primary verification receipt mismatch")


def verify_run(run_dir: Path, controller_dir: Path) -> dict:
    root = Path(run_dir)
    controller = Path(controller_dir)
    if controller.exists():
        raise ValueError(f"controller directory already exists: {controller}")
    _verify_artifact_hashes(root)
    manifest = _load_json(root / "run_manifest.json")
    workload = _verify_manifest_identity(root, manifest)
    if manifest["gate"] == "prefix":
        summary = _verify_prefix(root, manifest)
    else:
        summary = _verify_chunked(root, manifest, workload)
    _verify_primary_outputs(root, manifest, summary)
    controller.parent.mkdir(parents=True, exist_ok=True)
    controller.mkdir()
    summary_bytes = _json_bytes(summary)
    report_bytes = _render_report(manifest, summary).encode("utf-8")
    _atomic_write_bytes(controller / "summary.json", summary_bytes)
    _atomic_write_bytes(controller / "report.md", report_bytes)
    receipt = {
        "status": "PASS",
        "run_manifest_sha256": _sha256_path(
            root / "run_manifest.json"
        ),
        "primary_summary_sha256": _sha256_path(root / "summary.json"),
        "controller_summary_sha256": _sha256_path(
            controller / "summary.json"
        ),
        "classification": summary["classification"],
    }
    _atomic_write_json(controller / "verification_receipt.json", receipt)
    _atomic_write_bytes(controller / "verify.exitcode", b"0\n")
    return summary


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Independently verify a staged inference benchmark bundle",
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("controller_dir", type=Path)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    verify_run(args.run_dir, args.controller_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
