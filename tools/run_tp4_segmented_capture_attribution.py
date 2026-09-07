#!/usr/bin/env python3
"""Strict-clean controller for TP4 segmented-capture Phase A1."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile
import time

if __package__:
    from tools.run_qwen38_tp4_communication_profile import (
        DEFAULT_SSH_TARGET,
        build_ssh_argv,
        query_local_kerberos,
        query_remote_gpu_inventory,
        run_remote_argv,
        validate_selected_gpu_processes,
        wait_for_strict_clean_gpus,
    )
else:
    from run_qwen38_tp4_communication_profile import (
        DEFAULT_SSH_TARGET,
        build_ssh_argv,
        query_local_kerberos,
        query_remote_gpu_inventory,
        run_remote_argv,
        validate_selected_gpu_processes,
        wait_for_strict_clean_gpus,
    )


REMOTE_BASE = (
    "/data00/home/sitian/tinyllmforge-workspaces/"
    "command-timeline-20260818"
)
REMOTE_ROOT = f"{REMOTE_BASE}/tp4-segmented-capture-attribution"
MODEL_REPOSITORY = "Qwen/Qwen3.8-27B"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
MODEL_ROOT = (
    f"{REMOTE_BASE}/models/Qwen3.8-27B/snapshots/{MODEL_REVISION}"
)
SOURCE_SCHEMA = "tinyllmforge.tp4-segmented-attribution-source.v1"
PLAN_SCHEMA = "tinyllmforge.tp4-segmented-attribution-plan.v1"
ADMISSION_SCHEMA = (
    "tinyllmforge.tp4-segmented-attribution-admission.v1"
)
CLEANUP_SCHEMA = (
    "tinyllmforge.tp4-segmented-attribution-cleanup.v1"
)
VERIFICATION_SCHEMA = (
    "tinyllmforge.tp4-segmented-attribution-verification.v1"
)
MANIFEST_SCHEMA = (
    "tinyllmforge.tp4-segmented-attribution-manifest.v1"
)
POST_MANIFEST_SCHEMA = (
    "tinyllmforge.tp4-segmented-attribution-post-verification.v1"
)
RUN_TAG_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
WORLD_SIZE = 4
MAX_GPU_MEMORY_USED_MIB = 1_024
MAX_GPU_UTILIZATION_PERCENT = 5
MAX_ADDED_MEMORY_BYTES_PER_RANK = 512 * 1024 * 1024
MAX_SEGMENT_NS = 1_800_000_000
MAX_LIFECYCLE_NS = 4_500_000_000
SUCCESSFUL_TERMINAL_STATES = (
    "REPAIR_CANDIDATE",
    "PIVOT_TP4_COMMUNICATION_COMPUTE_FUSION",
)
ALLOWED_TERMINAL_STATES = (*SUCCESSFUL_TERMINAL_STATES, "INCOMPLETE")
DEFAULT_REMOTE_PYTHON = "/data00/home/sitian/tllm/env/bin/python"
DEFAULT_COMMAND_TIMEOUT_S = 21_600
DEFAULT_GPU_WAIT_TIMEOUT_S = 21_600
DEFAULT_GPU_POLL_INTERVAL_S = 15
DEFAULT_RETRY_COUNT = 3
KERBEROS_GUARD_MARGIN_S = 900
MIN_LOCAL_ARTIFACT_FREE_BYTES = 1 * 1024**3
MIN_REMOTE_ARTIFACT_FREE_BYTES = 8 * 1024**3

_BASE_RANGES = ((0, 16), (16, 32), (32, 48), (48, 64))
_BASE_CONTROLS = (
    {
        "control_id": "stitched_p4_repeat_0",
        "ranges": _BASE_RANGES,
        "kind": "stitched",
        "pool_mode": "shared",
        "formal_route_row": True,
    },
    {
        "control_id": "stitched_p4_repeat_1",
        "ranges": _BASE_RANGES,
        "kind": "stitched",
        "pool_mode": "shared",
        "formal_route_row": False,
    },
    *(
        {
            "control_id": f"isolated_{start}_{end}",
            "ranges": ((start, end),),
            "kind": "isolated",
            "pool_mode": "isolated",
            "formal_route_row": False,
        }
        for start, end in _BASE_RANGES
    ),
)
_POOL_CONTROLS = (
    {
        "control_id": "pool_fastest_shared",
        "ranges": (),
        "kind": "pool_control",
        "pool_mode": "shared",
        "formal_route_row": False,
    },
    {
        "control_id": "pool_fastest_isolated",
        "ranges": (),
        "kind": "pool_control",
        "pool_mode": "isolated",
        "formal_route_row": False,
    },
    {
        "control_id": "pool_slowest_shared",
        "ranges": (),
        "kind": "pool_control",
        "pool_mode": "shared",
        "formal_route_row": False,
    },
    {
        "control_id": "pool_slowest_isolated",
        "ranges": (),
        "kind": "pool_control",
        "pool_mode": "isolated",
        "formal_route_row": False,
    },
)
EXPECTED_CONTROL_IDS = tuple(
    row["control_id"] for row in (*_BASE_CONTROLS, *_POOL_CONTROLS)
)
PRODUCER_ARTIFACT_NAMES = (
    "source_identity.json",
    "plan.json",
    "admission.json",
    "phase_rows.jsonl",
    "scratch_rows.jsonl",
    "rank_results.json",
    "process_receipts.json",
    "cleanup_receipt.json",
    "diagnosis.json",
    "worker_summary.json",
)


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _validate_run_tag(run_tag: object) -> str:
    if (
        not isinstance(run_tag, str)
        or not RUN_TAG_PATTERN.fullmatch(run_tag)
        or ".." in run_tag
    ):
        raise ValueError("run tag is invalid")
    return run_tag


def _is_below(path: str, root: str) -> bool:
    candidate = PurePosixPath(path)
    approved = PurePosixPath(root)
    return candidate.is_absolute() and candidate.is_relative_to(approved)


def _matches_exact_tag_identity(
    command: bytes,
    environment: bytes,
    *,
    run_tag: str,
    attempt_root: str,
) -> bool:
    tag_entry = f"TINYLLMFORGE_RUN_TAG={run_tag}".encode("utf-8")
    attempt_bytes = attempt_root.encode("utf-8")
    return (
        any(
            argument == attempt_bytes
            or argument.startswith(attempt_bytes + b"/")
            for argument in command.split(b"\0")
        )
        or tag_entry in environment.split(b"\0")
    )


def _atomic_write_json(path: Path, payload: object) -> bytes:
    encoded = (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    return encoded


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _artifact_hashes(root: Path, names: tuple[str, ...]) -> dict:
    hashes = {}
    for name in names:
        path = root / name
        if not path.is_file():
            raise RuntimeError(f"producer artifact is missing: {name}")
        payload = path.read_bytes()
        if not payload:
            raise RuntimeError(f"producer artifact is empty: {name}")
        hashes[name] = _sha256_bytes(payload)
    return hashes


def write_pre_verification_manifest(bundle_root: Path) -> dict:
    bundle_root = Path(bundle_root)
    payload = {
        "schema_version": MANIFEST_SCHEMA,
        "artifacts": _artifact_hashes(
            bundle_root,
            PRODUCER_ARTIFACT_NAMES,
        ),
    }
    _atomic_write_json(bundle_root / "manifest.json", payload)
    return payload


def validate_pre_verification_manifest(bundle_root: Path) -> dict:
    bundle_root = Path(bundle_root)
    manifest_path = bundle_root / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("pre-verification manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = _artifact_hashes(bundle_root, PRODUCER_ARTIFACT_NAMES)
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != MANIFEST_SCHEMA
        or manifest.get("artifacts") != expected
    ):
        raise RuntimeError("pre-verification manifest hash mismatch")
    return manifest


def build_post_verification_manifest(
    *,
    bundle_root: Path,
    remote_verification: dict,
    local_verification: dict,
    cleanup_bytes: bytes,
    final_live_exact_tag_scan: list,
) -> dict:
    identity = validate_verifier_identity(
        remote_verification,
        local_verification,
    )
    pre_manifest = validate_pre_verification_manifest(bundle_root)
    pre_bytes = (Path(bundle_root) / "manifest.json").read_bytes()
    return {
        "schema_version": POST_MANIFEST_SCHEMA,
        "artifacts": {
            "manifest.json": _sha256_bytes(pre_bytes),
            "remote_independent_verification.json": _sha256_bytes(
                remote_verification["bytes"]
            ),
            "local_independent_verification.json": _sha256_bytes(
                local_verification["bytes"]
            ),
            "cleanup_receipt.json": _sha256_bytes(cleanup_bytes),
        },
        "pre_verification_manifest_sha256": _sha256_bytes(pre_bytes),
        "pre_verification_manifest": pre_manifest,
        "verifier_byte_identical": identity["byte_identical"],
        "verifier_sha256": identity["sha256"],
        "classification": identity["classification"],
        "final_live_exact_tag_scan_sha256": canonical_sha256(
            final_live_exact_tag_scan
        ),
    }


def validate_post_verification_manifest(
    manifest: object,
    *,
    bundle_root: Path,
    remote_verification: dict,
    local_verification: dict,
    cleanup_bytes: bytes,
    final_live_exact_tag_scan: list,
) -> dict:
    try:
        expected = build_post_verification_manifest(
            bundle_root=bundle_root,
            remote_verification=remote_verification,
            local_verification=local_verification,
            cleanup_bytes=cleanup_bytes,
            final_live_exact_tag_scan=final_live_exact_tag_scan,
        )
    except RuntimeError as error:
        raise RuntimeError(
            "post-verification manifest validation failed"
        ) from error
    if manifest != expected:
        raise RuntimeError("post-verification manifest mismatch")
    return expected


def _validate_source_identity(
    source_identity: object,
    *,
    run_tag: str,
) -> dict:
    if (
        not isinstance(source_identity, dict)
        or source_identity.get("schema_version") != SOURCE_SCHEMA
        or source_identity.get("phase") != "A1"
        or source_identity.get("run_tag") != run_tag
        or source_identity.get("model_repository") != MODEL_REPOSITORY
        or source_identity.get("model_revision") != MODEL_REVISION
        or not re.fullmatch(
            r"[0-9a-f]{40}",
            str(source_identity.get("source_revision", "")),
        )
        or any(
            not re.fullmatch(
                r"[0-9a-f]{64}",
                str(source_identity.get(name, "")),
            )
            for name in (
                "source_tree_sha256",
                "worker_sha256",
                "verifier_sha256",
            )
        )
    ):
        raise ValueError("source identity is invalid")
    return dict(source_identity)


def _capture_source_identity(run_tag: str) -> dict:
    root = Path(__file__).resolve().parents[1]

    def git(*arguments: str) -> bytes:
        result = subprocess.run(
            ["git", "-C", str(root), *arguments],
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                result.stderr.decode(errors="replace")
                or "git command failed"
            )
        return result.stdout

    revision = git("rev-parse", "HEAD").decode().strip()
    dirty = git(
        "status",
        "--porcelain=v1",
        "--untracked-files=no",
        "--",
        "tinyvllm",
        "tools",
    )
    if dirty:
        raise ValueError("source archive scope has tracked changes")
    tree = git(
        "ls-tree",
        "-r",
        "--full-tree",
        revision,
        "tinyvllm",
        "tools",
    )
    worker = root / "tools" / "tp4_segmented_capture_attribution_worker.py"
    verifier = (
        root / "tools" / "verify_tp4_segmented_capture_attribution.py"
    )
    return {
        "schema_version": SOURCE_SCHEMA,
        "phase": "A1",
        "run_tag": _validate_run_tag(run_tag),
        "source_revision": revision,
        "source_tree_sha256": _sha256_bytes(tree),
        "worker_sha256": _sha256_bytes(worker.read_bytes()),
        "verifier_sha256": _sha256_bytes(verifier.read_bytes()),
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
    }


def derive_diagnosis(
    *,
    phase_rows: list[dict],
    scratch_rows: list[dict],
) -> dict:
    if not isinstance(phase_rows, list) or not phase_rows:
        raise RuntimeError("phase rows are unavailable for diagnosis")
    if not isinstance(scratch_rows, list) or not scratch_rows:
        raise RuntimeError("scratch rows are unavailable for diagnosis")
    checkpoints = ("S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7")
    divergence_by_rank = {}
    for rank in range(WORLD_SIZE):
        rows = [row for row in scratch_rows if row.get("rank") == rank]
        first = next(
            (
                checkpoint
                for checkpoint in checkpoints
                if any(
                    row.get("checkpoint") == checkpoint
                    and (
                        row.get("key_diff", {}).get("equal_to_s0")
                        is not True
                        or row.get("value_diff", {}).get("equal_to_s0")
                        is not True
                    )
                    for row in rows
                )
            ),
            None,
        )
        divergence_by_rank[rank] = first
    divergences = set(divergence_by_rank.values())
    if len(divergences) != 1 or None in divergences:
        raise RuntimeError("scratch divergence is missing or disagrees")
    first_divergence = next(iter(divergences))
    formal = [
        row
        for row in phase_rows
        if row.get("control_id") == "stitched_p4_repeat_0"
    ]
    if not formal:
        raise RuntimeError("formal stitched rows are unavailable")
    phase_names = (
        "snapshot_and_prepare_ns",
        "graph_object_create_ns",
        "capture_context_enter_ns",
        "capture_body_ns",
        "capture_context_exit_and_instantiate_ns",
        "post_capture_synchronize_ns",
        "post_capture_restore_ns",
        "graph_reset_ns",
    )
    maxima = {
        name: max(int(row[name]) for row in formal)
        for name in phase_names
    }
    slow_phase = max(maxima, key=lambda name: (maxima[name], name))
    root_causes = {
        "S1": (
            "eager_scratch_write",
            "_AttributionCudaBackend.run_eager",
            "preserve and restore the eager scratch baseline",
        ),
        "S3": (
            "capture_scratch_write",
            "_AttributionCudaBackend.capture_attributed_segment",
            "isolate capture-time scratch writes from the baseline",
        ),
        "S5": (
            "replay_scratch_write",
            "_AttributionCudaBackend.replay",
            "isolate replay-time scratch writes from the baseline",
        ),
    }
    cause, symbol, statement = root_causes.get(
        first_divergence,
        (
            "scratch_lifecycle_divergence",
            "_AttributionCudaBackend.run_control",
            "repair the first localized scratch lifecycle transition",
        ),
    )
    return {
        "first_scratch_divergence": first_divergence,
        "slow_capture_phase": slow_phase,
        "root_cause_kind": cause,
        "source_path": (
            "tools/tp4_segmented_capture_attribution_worker.py"
        ),
        "source_symbol": symbol,
        "repair_statement": statement,
        "repair_count": 1,
        "projected_max_segment_ns": max(
            int(row["segment_total_ns"]) for row in formal
        ),
        "projected_lifecycle_ns": max(
            int(row["program_lifecycle_ns"]) for row in formal
        ),
        "projected_graph_count": 4,
    }


def _normalize_gpus(selected_gpus: object) -> list[dict]:
    if not isinstance(selected_gpus, list) or len(selected_gpus) != WORLD_SIZE:
        raise ValueError("strict_clean requires exactly four GPUs")
    normalized = []
    indices = set()
    uuids = set()
    for gpu in selected_gpus:
        if not isinstance(gpu, dict):
            raise ValueError("strict-clean GPU inventory is invalid")
        index = gpu.get("gpu_index")
        uuid = gpu.get("gpu_uuid")
        memory = gpu.get("memory_used_mib")
        utilization = gpu.get("utilization_percent")
        processes = gpu.get("compute_processes")
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or not isinstance(uuid, str)
            or not uuid
            or index in indices
            or uuid in uuids
            or isinstance(memory, bool)
            or not isinstance(memory, int)
            or memory < 0
            or memory > MAX_GPU_MEMORY_USED_MIB
            or isinstance(utilization, bool)
            or not isinstance(utilization, int)
            or utilization < 0
            or utilization > MAX_GPU_UTILIZATION_PERCENT
            or processes != []
        ):
            raise ValueError("strict-clean GPU inventory is invalid")
        indices.add(index)
        uuids.add(uuid)
        normalized.append(dict(gpu))
    return normalized


def build_plan(
    *,
    run_tag: str,
    source_identity: dict,
    selected_gpus: list[dict],
    admission_mode: str,
) -> dict:
    run_tag = _validate_run_tag(run_tag)
    if admission_mode != "strict_clean":
        raise ValueError("strict_clean admission is required")
    source = _validate_source_identity(
        source_identity,
        run_tag=run_tag,
    )
    selected = _normalize_gpus(selected_gpus)
    attempt_root = f"{REMOTE_ROOT}/{run_tag}"
    runtime_root = f"{attempt_root}/runtime"
    paths = {
        "attempt_root": attempt_root,
        "source_root": f"{attempt_root}/source",
        "raw_root": f"{attempt_root}/raw",
        "bundle_root": f"{attempt_root}/final_bundle",
        "controller_root": f"{attempt_root}/controller",
        "worker_stdout_path": f"{attempt_root}/controller/worker.stdout",
        "worker_stderr_path": f"{attempt_root}/controller/worker.stderr",
        "remote_verification_path": (
            f"{attempt_root}/controller/"
            "remote_independent_verification.json"
        ),
        "post_verification_manifest_path": (
            f"{attempt_root}/controller/"
            "post_verification_manifest.json"
        ),
    }
    environment = {
        "TMPDIR": f"{runtime_root}/tmp",
        "XDG_CACHE_HOME": f"{runtime_root}/cache/xdg",
        "HF_HOME": f"{runtime_root}/cache/huggingface",
        "TRANSFORMERS_CACHE": (
            f"{runtime_root}/cache/huggingface/transformers"
        ),
        "TORCH_EXTENSIONS_DIR": (
            f"{runtime_root}/cache/torch-extensions"
        ),
        "CUDA_CACHE_PATH": f"{runtime_root}/cache/cuda",
    }
    if (
        not all(_is_below(path, attempt_root) for path in paths.values())
        or not all(
            _is_below(path, attempt_root)
            for path in environment.values()
        )
    ):
        raise ValueError("remote path escapes the large mounted root")
    controls = [dict(row) for row in (*_BASE_CONTROLS, *_POOL_CONTROLS)]
    return {
        "schema_version": PLAN_SCHEMA,
        "phase": "A1",
        "run_tag": run_tag,
        "admission_mode": "strict_clean",
        "strict_clean": True,
        "source_identity": source,
        "source_revision": source["source_revision"],
        "source_tree_sha256": source["source_tree_sha256"],
        "worker_sha256": source["worker_sha256"],
        "verifier_sha256": source["verifier_sha256"],
        "plan_sha256": canonical_sha256(_BASE_CONTROLS),
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "model_root": MODEL_ROOT,
        "dtype": "bfloat16",
        "tensor_parallel_size": WORLD_SIZE,
        "batch_size": 8,
        "prompt_length": 256,
        "max_tokens": 2,
        "model_length": 384,
        "controls": controls,
        "max_segment_ns": MAX_SEGMENT_NS,
        "max_lifecycle_ns": MAX_LIFECYCLE_NS,
        "max_added_memory_bytes_per_rank": (
            MAX_ADDED_MEMORY_BYTES_PER_RANK
        ),
        "selected_gpus": selected,
        "selected_gpu_indices": [
            row["gpu_index"] for row in selected
        ],
        "paths": paths,
        "environment": environment,
        "process_environment": {
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": (
                f"{paths['source_root']}:"
                f"{paths['source_root']}/tools"
            ),
            "TINYLLMFORGE_RUN_TAG": run_tag,
        },
    }


def validate_verifier_identity(remote: object, local: object) -> dict:
    if not isinstance(remote, dict) or not isinstance(local, dict):
        raise RuntimeError("verifier output is invalid")
    remote_bytes = remote.get("bytes")
    local_bytes = local.get("bytes")
    remote_result = remote.get("result")
    local_result = local.get("result")
    if (
        not isinstance(remote_bytes, bytes)
        or not isinstance(local_bytes, bytes)
        or remote_bytes != local_bytes
    ):
        raise RuntimeError("verifier results are not byte-identical")
    if (
        not isinstance(remote_result, dict)
        or remote_result != local_result
        or remote_result.get("schema_version") != VERIFICATION_SCHEMA
        or remote_result.get("phase") != "A1"
    ):
        raise RuntimeError("verifier identity is invalid")
    classification = remote_result.get("classification")
    if not isinstance(classification, str) or not classification:
        raise RuntimeError("verifier identity is invalid")
    return {
        "byte_identical": True,
        "sha256": hashlib.sha256(remote_bytes).hexdigest(),
        "classification": classification,
        "source_revision": remote_result.get("source_revision"),
        "run_tag": remote_result.get("run_tag"),
    }


def _require_preflight(preflight: object) -> dict:
    if (
        not isinstance(preflight, dict)
        or preflight.get("classification") != "PASS"
        or preflight.get("attempt_exists") is not False
        or preflight.get("local_attempt_exists") is not False
        or preflight.get("remote_root") != REMOTE_ROOT
    ):
        raise RuntimeError("SSH/storage preflight rejected the run")
    return preflight


def _require_kerberos(receipt: object) -> dict:
    if (
        not isinstance(receipt, dict)
        or receipt.get("classification") not in {"READY", "PASS"}
    ):
        raise RuntimeError("Kerberos TTL preflight failed")
    return receipt


def _require_cleanup(receipt: object) -> dict:
    scans = (
        receipt.get("final_exact_tag_scans")
        if isinstance(receipt, dict)
        else None
    )
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema_version") != CLEANUP_SCHEMA
        or receipt.get("classification") != "CLEAN"
        or scans != [[], [], []]
    ):
        raise RuntimeError("exact-tag cleanup is not clean")
    return receipt


def _monitor_and_run(seed: dict, adapter: object) -> dict:
    if not isinstance(seed, dict):
        raise ValueError("run seed is invalid")
    run_tag = _validate_run_tag(seed.get("run_tag"))
    if seed.get("admission_mode") != "strict_clean":
        raise ValueError("strict_clean admission is required")

    source = _validate_source_identity(
        adapter.freeze_source(seed),
        run_tag=run_tag,
    )
    preflight = _require_preflight(
        adapter.ssh_storage_preflight(seed, source)
    )
    kerberos = _require_kerberos(
        adapter.kerberos_ttl_guard(seed, preflight)
    )
    admission = adapter.gpu_admission(seed, preflight)
    if (
        not isinstance(admission, dict)
        or admission.get("classification") != "READY"
    ):
        raise RuntimeError("four strict-clean GPUs were not admitted")
    plan = build_plan(
        run_tag=run_tag,
        source_identity=source,
        selected_gpus=admission.get("selected_gpus"),
        admission_mode="strict_clean",
    )

    launch = None
    operation_error = None
    cleanup = None
    try:
        launch = adapter.launch_once(plan, admission)
        waited = adapter.wait(plan, launch)
    except BaseException as error:
        operation_error = error
    if launch is not None:
        try:
            cleanup = _require_cleanup(
                adapter.owned_cleanup(plan, launch)
            )
        except BaseException as cleanup_error:
            if operation_error is not None:
                raise operation_error from cleanup_error
            raise
    if operation_error is not None:
        raise operation_error
    if cleanup is None:
        raise RuntimeError("launch cleanup evidence is missing")

    downloaded = adapter.download(plan, waited, cleanup)
    remote = adapter.remote_verify(plan, downloaded)
    local = adapter.local_verify(plan, downloaded)
    verifier_identity = adapter.validate_verifier_identity(
        plan,
        remote,
        local,
    )
    post_manifest = adapter.write_post_verification_manifest(
        plan,
        downloaded,
        remote,
        local,
        verifier_identity,
        cleanup,
    )
    final_scan = adapter.final_live_exact_tag_scan(plan)
    if final_scan != []:
        raise RuntimeError("final live exact-tag scan is not empty")
    classification = verifier_identity.get("classification")
    if classification not in SUCCESSFUL_TERMINAL_STATES:
        raise RuntimeError(f"Phase A1 terminal state: {classification}")
    return {
        "classification": classification,
        "plan": plan,
        "source_identity": source,
        "preflight": preflight,
        "kerberos": kerberos,
        "admission": admission,
        "launch": launch,
        "wait": waited,
        "cleanup": cleanup,
        "download": downloaded,
        "remote_verification": remote["result"],
        "local_verification": local["result"],
        "verifier_identity": verifier_identity,
        "post_verification_manifest": post_manifest,
        "final_live_exact_tag_scan": final_scan,
    }


def monitor_and_run(seed: dict, adapter: object) -> dict:
    try:
        return _monitor_and_run(seed, adapter)
    except BaseException as error:
        persist = getattr(adapter, "persist_incomplete", None)
        if callable(persist):
            try:
                persist(seed, error)
            except BaseException as persistence_error:
                raise error from persistence_error
        raise


class ProductionAdapter:
    """Hardened one-launch SSH transport for Phase A1."""

    def __init__(
        self,
        *,
        run_tag: str,
        local_attempt_root: Path,
        ssh_target: str = DEFAULT_SSH_TARGET,
        remote_python: str = DEFAULT_REMOTE_PYTHON,
        control_path: str | None = None,
        command_timeout_s: int = DEFAULT_COMMAND_TIMEOUT_S,
        gpu_wait_timeout_s: int = DEFAULT_GPU_WAIT_TIMEOUT_S,
        gpu_poll_interval_s: int = DEFAULT_GPU_POLL_INTERVAL_S,
        retry_count: int = DEFAULT_RETRY_COUNT,
        local_command_runner=subprocess.run,
        kerberos_query=query_local_kerberos,
    ):
        self.run_tag = _validate_run_tag(run_tag)
        self.local_attempt_root = Path(local_attempt_root).resolve()
        self.local_controller_root = self.local_attempt_root / "controller"
        self.local_raw_root = self.local_attempt_root / "raw"
        self.local_bundle_root = self.local_attempt_root / "final_bundle"
        self.ssh_target = ssh_target
        self.remote_python = remote_python
        self.control_path = control_path
        self.command_timeout_s = int(command_timeout_s)
        self.gpu_wait_timeout_s = int(gpu_wait_timeout_s)
        self.gpu_poll_interval_s = int(gpu_poll_interval_s)
        self.retry_count = int(retry_count)
        if not callable(local_command_runner):
            raise ValueError("local command runner is invalid")
        if not callable(kerberos_query):
            raise ValueError("Kerberos query is invalid")
        self.local_command_runner = local_command_runner
        self.kerberos_query = kerberos_query
        self._source = None
        self._admission = None
        self._cleanup = None
        self._process = None

    def _remote(self, remote_argv, *, timeout_s=None):
        result = run_remote_argv(
            ssh_target=self.ssh_target,
            remote_argv=list(remote_argv),
            control_path=self.control_path,
            timeout_s=(
                self.command_timeout_s
                if timeout_s is None
                else int(timeout_s)
            ),
            retry_count=self.retry_count,
            command_runner=self.local_command_runner,
        )
        if result.returncode != 0:
            raise RuntimeError(
                getattr(result, "stderr", "")
                or "remote command failed"
            )
        return result

    def _remote_once(self, remote_argv, *, timeout_s=None):
        result = self.local_command_runner(
            build_ssh_argv(
                ssh_target=self.ssh_target,
                remote_argv=list(remote_argv),
                control_path=self.control_path,
            ),
            text=True,
            capture_output=True,
            check=False,
            timeout=(
                self.command_timeout_s
                if timeout_s is None
                else int(timeout_s)
            ),
        )
        if result.returncode != 0:
            raise RuntimeError(
                getattr(result, "stderr", "")
                or "non-idempotent remote command failed"
            )
        return result

    def _upload_bytes(self, remote_path: str, payload: bytes) -> None:
        script = "\n".join([
            "import os,sys,tempfile",
            "path=sys.argv[1]",
            "payload=sys.stdin.buffer.read()",
            "directory=os.path.dirname(path)",
            "fd,temp=tempfile.mkstemp(prefix='.upload.',dir=directory)",
            "with os.fdopen(fd,'wb') as handle:",
            "  handle.write(payload)",
            "  handle.flush()",
            "  os.fsync(handle.fileno())",
            "os.replace(temp,path)",
        ])
        result = None
        for attempt in range(self.retry_count):
            result = self.local_command_runner(
                build_ssh_argv(
                    ssh_target=self.ssh_target,
                    remote_argv=[
                        "python3",
                        "-c",
                        script,
                        remote_path,
                    ],
                    control_path=self.control_path,
                ),
                input=payload,
                capture_output=True,
                check=False,
                timeout=self.command_timeout_s,
            )
            if (
                result.returncode != 255
                or attempt + 1 == self.retry_count
            ):
                break
            time.sleep(1.0)
        if result is None or result.returncode != 0:
            raise RuntimeError(
                (
                    b""
                    if result is None
                    else getattr(result, "stderr", b"")
                ).decode(errors="replace")
                or "remote upload failed"
            )

    def _stage_source(self, plan: dict) -> None:
        root = Path(__file__).resolve().parents[1]
        receiver = None
        archive_returncode = None
        archive_stderr = b""
        for attempt in range(self.retry_count):
            archive = subprocess.Popen(
                [
                    "git",
                    "-C",
                    str(root),
                    "archive",
                    "--format=tar",
                    plan["source_revision"],
                    "tinyvllm",
                    "tools",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            assert archive.stdout is not None
            receiver = self.local_command_runner(
                build_ssh_argv(
                    ssh_target=self.ssh_target,
                    remote_argv=[
                        "tar",
                        "-xf",
                        "-",
                        "-C",
                        plan["paths"]["source_root"],
                    ],
                    control_path=self.control_path,
                ),
                stdin=archive.stdout,
                capture_output=True,
                check=False,
                timeout=self.command_timeout_s,
            )
            archive.stdout.close()
            archive_stderr = (
                archive.stderr.read()
                if archive.stderr is not None
                else b""
            )
            archive_returncode = archive.wait()
            if (
                receiver.returncode != 255
                or attempt + 1 == self.retry_count
            ):
                break
            time.sleep(1.0)
        if (
            receiver is None
            or archive_returncode != 0
            or receiver.returncode != 0
        ):
            raise RuntimeError(
                archive_stderr.decode(errors="replace")
                or (
                    b""
                    if receiver is None
                    else getattr(receiver, "stderr", b"")
                ).decode(errors="replace")
                or "source staging failed"
            )

    def _download_archive(
        self,
        *,
        remote_root: str,
        names: tuple[str, ...],
        local_root: Path,
    ) -> None:
        local_root.mkdir(parents=True, exist_ok=True)
        sender = None
        receiver = None
        sender_stderr = b""
        sender_returncode = None
        for attempt in range(self.retry_count):
            sender = subprocess.Popen(
                build_ssh_argv(
                    ssh_target=self.ssh_target,
                    remote_argv=[
                        "tar",
                        "-cf",
                        "-",
                        "-C",
                        remote_root,
                        *names,
                    ],
                    control_path=self.control_path,
                ),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            assert sender.stdout is not None
            receiver = self.local_command_runner(
                ["tar", "-xf", "-", "-C", str(local_root)],
                stdin=sender.stdout,
                capture_output=True,
                check=False,
                timeout=self.command_timeout_s,
            )
            sender.stdout.close()
            sender_stderr = (
                sender.stderr.read()
                if sender.stderr is not None
                else b""
            )
            sender_returncode = sender.wait()
            if (
                sender_returncode != 255
                or attempt + 1 == self.retry_count
            ):
                break
            time.sleep(1.0)
        if (
            sender is None
            or receiver is None
            or sender_returncode != 0
            or receiver.returncode != 0
        ):
            raise RuntimeError(
                sender_stderr.decode(errors="replace")
                or (
                    b""
                    if receiver is None
                    else getattr(receiver, "stderr", b"")
                ).decode(errors="replace")
                or "bounded artifact download failed"
            )

    def _upload_bundle(self, plan: dict) -> None:
        receiver = None
        returncode = None
        stderr = b""
        for attempt in range(self.retry_count):
            archive = subprocess.Popen(
                [
                    "tar",
                    "-cf",
                    "-",
                    "-C",
                    str(self.local_attempt_root),
                    "final_bundle",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            assert archive.stdout is not None
            receiver = self.local_command_runner(
                build_ssh_argv(
                    ssh_target=self.ssh_target,
                    remote_argv=[
                        "tar",
                        "-xf",
                        "-",
                        "-C",
                        plan["paths"]["attempt_root"],
                    ],
                    control_path=self.control_path,
                ),
                stdin=archive.stdout,
                capture_output=True,
                check=False,
                timeout=self.command_timeout_s,
            )
            archive.stdout.close()
            stderr = (
                archive.stderr.read()
                if archive.stderr is not None
                else b""
            )
            returncode = archive.wait()
            if (
                receiver.returncode != 255
                or attempt + 1 == self.retry_count
            ):
                break
            time.sleep(1.0)
        if (
            receiver is None
            or returncode != 0
            or receiver.returncode != 0
        ):
            raise RuntimeError(
                stderr.decode(errors="replace")
                or (
                    b""
                    if receiver is None
                    else receiver.stderr
                ).decode(errors="replace")
                or "bundle upload failed"
            )

    def freeze_source(self, seed: dict) -> dict:
        if self._source is None and os.path.lexists(self.local_attempt_root):
            raise ValueError("local attempt root already exists")
        source = _capture_source_identity(seed["run_tag"])
        if self._source is not None and source != self._source:
            raise ValueError("frozen source identity drift")
        self._source = source
        self.local_controller_root.mkdir(
            parents=True,
            exist_ok=self.local_attempt_root.exists(),
        )
        _atomic_write_json(
            self.local_controller_root / "source_identity.json",
            source,
        )
        return dict(source)

    def persist_incomplete(
        self,
        seed: dict,
        error: BaseException,
    ) -> dict:
        receipt = {
            "schema_version": (
                "tinyllmforge.tp4-segmented-attribution-terminal.v1"
            ),
            "phase": "A1",
            "run_tag": _validate_run_tag(seed["run_tag"]),
            "classification": "INCOMPLETE",
            "launch_started": self._process is not None,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        _atomic_write_json(
            self.local_controller_root / "terminal_result.json",
            receipt,
        )
        return receipt

    def ssh_storage_preflight(
        self,
        seed: dict,
        source: dict,
    ) -> dict:
        del source
        attempt_root = f"{REMOTE_ROOT}/{seed['run_tag']}"
        script = "\n".join([
            "import json,os,sys",
            "base,root,attempt,model,revision,tag=sys.argv[1:]",
            "storage=os.statvfs(base)",
            "tag_entry=('TINYLLMFORGE_RUN_TAG='+tag).encode()",
            "attempt_bytes=attempt.encode()",
            "excluded={os.getpid(),os.getppid()}",
            "stale=[]",
            "for name in os.listdir('/proc'):",
            "  if not name.isdigit(): continue",
            "  pid=int(name)",
            "  if pid in excluded: continue",
            "  try: command=open('/proc/'+name+'/cmdline','rb').read()",
            "  except OSError: command=b''",
            "  try: environment=open('/proc/'+name+'/environ','rb').read()",
            "  except OSError: environment=b''",
            "  matched_cmdline=any(value==attempt_bytes or value.startswith(attempt_bytes+b'/') for value in command.split(b'\\0'))",
            "  matched_environment=tag_entry in environment.split(b'\\0')",
            "  if matched_cmdline or matched_environment:",
            "    stale.append({'pid':pid,'matched_cmdline':matched_cmdline,'matched_environment':matched_environment})",
            "config_path=os.path.join(model,'config.json')",
            "config={}",
            "if os.path.isfile(config_path):",
            "  with open(config_path,encoding='utf-8') as handle:",
            "    config=json.load(handle)",
            "text=config.get('text_config',{}) if isinstance(config,dict) else {}",
            "print(json.dumps({",
            "'base_ready':os.path.isdir(base) and os.access(base,os.R_OK|os.W_OK|os.X_OK),",
            "'remote_root_safe':not os.path.islink(root) and (not os.path.exists(root) or os.path.isdir(root)),",
            "'attempt_exists':os.path.lexists(attempt),",
            "'remote_free_bytes':storage.f_bavail*storage.f_frsize,",
            "'stale_exact_tag_processes':sorted(stale,key=lambda row:row['pid']),",
            "'model_ready':os.path.isdir(model) and os.access(model,os.R_OK|os.X_OK),",
            "'model_revision_matches':os.path.basename(os.path.realpath(model))==revision,",
            "'text_profile':{",
            "'num_hidden_layers':text.get('num_hidden_layers'),",
            "'hidden_size':text.get('hidden_size'),",
            "'vocab_size':text.get('vocab_size'),",
            "'dtype':text.get('dtype'),",
            "},",
            "},sort_keys=True))",
        ])
        result = self._remote([
            "python3",
            "-c",
            script,
            REMOTE_BASE,
            REMOTE_ROOT,
            attempt_root,
            MODEL_ROOT,
            MODEL_REVISION,
            seed["run_tag"],
        ])
        state = json.loads(result.stdout)
        expected_profile = {
            "num_hidden_layers": 64,
            "hidden_size": 5120,
            "vocab_size": 248320,
            "dtype": "bfloat16",
        }
        local_free_bytes = shutil.disk_usage(
            self.local_attempt_root
        ).free
        remote_free_bytes = state.get("remote_free_bytes")
        if (
            not isinstance(state, dict)
            or state.get("base_ready") is not True
            or state.get("remote_root_safe") is not True
            or state.get("attempt_exists") is not False
            or state.get("model_ready") is not True
            or state.get("model_revision_matches") is not True
            or state.get("text_profile") != expected_profile
        ):
            raise ValueError("remote storage or model preflight failed")
        if (
            isinstance(local_free_bytes, bool)
            or not isinstance(local_free_bytes, int)
            or local_free_bytes < MIN_LOCAL_ARTIFACT_FREE_BYTES
            or isinstance(remote_free_bytes, bool)
            or not isinstance(remote_free_bytes, int)
            or remote_free_bytes < MIN_REMOTE_ARTIFACT_FREE_BYTES
        ):
            raise ValueError("insufficient local or remote artifact space")
        stale_exact_tag_processes = state.get(
            "stale_exact_tag_processes"
        )
        if stale_exact_tag_processes != []:
            raise ValueError("stale exact-tag process exists")
        receipt = {
            "classification": "PASS",
            "attempt_exists": False,
            "local_attempt_exists": False,
            "remote_root": REMOTE_ROOT,
            "model_root": MODEL_ROOT,
            "model_revision": MODEL_REVISION,
            "text_profile": expected_profile,
            "local_free_bytes": local_free_bytes,
            "remote_free_bytes": remote_free_bytes,
            "minimum_local_artifact_free_bytes": (
                MIN_LOCAL_ARTIFACT_FREE_BYTES
            ),
            "minimum_remote_artifact_free_bytes": (
                MIN_REMOTE_ARTIFACT_FREE_BYTES
            ),
            "stale_exact_tag_processes": [],
        }
        _atomic_write_json(
            self.local_controller_root / "ssh_storage_preflight.json",
            receipt,
        )
        return receipt

    def kerberos_ttl_guard(self, seed: dict, preflight: dict) -> dict:
        del seed, preflight
        receipt = self.kerberos_query(
            minimum_lifetime_seconds=(
                self.command_timeout_s + KERBEROS_GUARD_MARGIN_S
            ),
        )
        if receipt.get("classification") not in {"READY", "PASS"}:
            receipt = {
                **receipt,
                "classification": "INCOMPLETE",
                "reason": "Kerberos TTL preflight failed",
            }
        _atomic_write_json(
            self.local_controller_root / "kerberos_ttl_guard.json",
            receipt,
        )
        return receipt

    def gpu_admission(self, seed: dict, preflight: dict) -> dict:
        del preflight

        def query_inventory():
            return query_remote_gpu_inventory(
                ssh_target=self.ssh_target,
                control_path=self.control_path,
                timeout_s=self.command_timeout_s,
                retry_count=self.retry_count,
                command_runner=self.local_command_runner,
            )

        monitor = wait_for_strict_clean_gpus(
            query_inventory=query_inventory,
            timeout_s=self.gpu_wait_timeout_s,
            poll_interval_s=self.gpu_poll_interval_s,
        )
        if monitor.get("classification") != "READY":
            return monitor
        selected = _normalize_gpus(monitor["selected_gpus"])
        self._admission = {
            "schema_version": ADMISSION_SCHEMA,
            "run_tag": seed["run_tag"],
            "admission_mode": "strict_clean",
            "strict_clean": True,
            "selected_gpus": [
                {
                    "rank": rank,
                    "index": row["gpu_index"],
                    "uuid": row["gpu_uuid"],
                    "memory_used_mib": row["memory_used_mib"],
                    "utilization_percent": row["utilization_percent"],
                    "compute_processes": [],
                }
                for rank, row in enumerate(selected)
            ],
        }
        _atomic_write_json(
            self.local_controller_root / "strict_clean_admission.json",
            self._admission,
        )
        return {
            "classification": "READY",
            "selected_gpus": selected,
            "foreign_processes": [],
            "samples": monitor.get("samples", []),
        }

    def launch_once(self, plan: dict, admission: dict) -> dict:
        if self._process is not None:
            raise RuntimeError("duplicate launch is forbidden")
        if self._source is None or self._admission is None:
            raise RuntimeError("launch prerequisites are incomplete")
        launch_kerberos = self.kerberos_query(
            minimum_lifetime_seconds=(
                self.command_timeout_s + KERBEROS_GUARD_MARGIN_S
            ),
        )
        _atomic_write_json(
            self.local_controller_root
            / "launch_kerberos_ttl_guard.json",
            launch_kerberos,
        )
        if launch_kerberos.get("classification") not in {
            "READY",
            "PASS",
        }:
            raise RuntimeError(
                "Kerberos TTL preflight failed at launch"
            )
        observed = query_remote_gpu_inventory(
            ssh_target=self.ssh_target,
            control_path=self.control_path,
            timeout_s=self.command_timeout_s,
            retry_count=self.retry_count,
            command_runner=self.local_command_runner,
        )
        validate_selected_gpu_processes(
            selected=tuple(admission["selected_gpus"]),
            observed=observed,
            owned_pids=frozenset(),
        )
        directories = [
            plan["paths"]["attempt_root"],
            plan["paths"]["source_root"],
            plan["paths"]["raw_root"],
            plan["paths"]["bundle_root"],
            plan["paths"]["controller_root"],
            *plan["environment"].values(),
        ]
        script = "\n".join([
            "import os,sys",
            "paths=sys.argv[1:]",
            "for path in paths:",
            "  if os.path.lexists(path):",
            "    raise SystemExit('path already exists: '+path)",
            "for path in paths:",
            "  os.makedirs(path,exist_ok=False)",
        ])
        self._remote_once(["python3", "-c", script, *directories])
        self._stage_source(plan)
        for name, payload in (
            ("source_identity.json", self._source),
            ("plan.json", plan),
            ("admission.json", self._admission),
        ):
            self._upload_bytes(
                f"{plan['paths']['raw_root']}/{name}",
                _canonical_bytes(payload),
            )
        environment = {
            **plan["environment"],
            **plan["process_environment"],
            "CUDA_VISIBLE_DEVICES": ",".join(
                str(index) for index in plan["selected_gpu_indices"]
            ),
        }
        wrapper = "\n".join([
            "import json,os,subprocess,sys",
            "python,stdout_path,stderr_path,env_json,*argv=sys.argv[1:]",
            "environment=os.environ.copy()",
            "environment.update(json.loads(env_json))",
            "with open(stdout_path,'w',encoding='utf-8') as out, open(stderr_path,'w',encoding='utf-8') as err:",
            "  result=subprocess.run([python,*argv],env=environment,stdout=out,stderr=err,text=True,check=False)",
            "raise SystemExit(result.returncode)",
        ])
        remote_argv = [
            "python3",
            "-c",
            wrapper,
            self.remote_python,
            plan["paths"]["worker_stdout_path"],
            plan["paths"]["worker_stderr_path"],
            json.dumps(environment, sort_keys=True),
            (
                f"{plan['paths']['source_root']}/tools/"
                "tp4_segmented_capture_attribution_worker.py"
            ),
            "--run-tag",
            plan["run_tag"],
            "--source-revision",
            plan["source_revision"],
            "--model-root",
            MODEL_ROOT,
            "--output-root",
            plan["paths"]["raw_root"],
            "--timeout-s",
            "900",
        ]
        self._process = subprocess.Popen(
            build_ssh_argv(
                ssh_target=self.ssh_target,
                remote_argv=remote_argv,
                control_path=self.control_path,
            ),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        receipt = {
            "launch_count": 1,
            "local_ssh_pid": self._process.pid,
            "owned_pids": [self._process.pid],
        }
        _atomic_write_json(
            self.local_controller_root / "launch.json",
            receipt,
        )
        return receipt

    def wait(self, plan: dict, launch: dict) -> dict:
        del launch
        if self._process is None:
            raise RuntimeError("worker process was not launched")
        stdout, stderr = self._process.communicate(
            timeout=self.command_timeout_s
        )
        returncode = self._process.returncode
        _atomic_write_json(
            self.local_controller_root / "worker_wait.json",
            {
                "returncode": returncode,
                "stdout": stdout,
                "stderr": stderr,
            },
        )
        if returncode != 0:
            tail = self._remote([
                "python3",
                "-c",
                (
                    "import pathlib,sys;"
                    "p=pathlib.Path(sys.argv[1]);"
                    "print(p.read_text(errors='replace')[-12000:] "
                    "if p.is_file() else '')"
                ),
                plan["paths"]["worker_stderr_path"],
            ]).stdout
            raise RuntimeError(
                f"remote Phase A1 worker failed with {returncode}: {tail}"
            )
        return {"exit_code": 0, "stdout": stdout}

    def _scan_exact_tag(self, plan: dict) -> list[dict]:
        script = "\n".join([
            "import json,os,sys",
            "tag,attempt_root=sys.argv[1:]",
            "tag_entry=('TINYLLMFORGE_RUN_TAG='+tag).encode()",
            "attempt_bytes=attempt_root.encode()",
            "excluded={os.getpid(),os.getppid()}",
            "rows=[]",
            "for name in os.listdir('/proc'):",
            "  if not name.isdigit(): continue",
            "  pid=int(name)",
            "  if pid in excluded: continue",
            "  try: command=open('/proc/'+name+'/cmdline','rb').read()",
            "  except OSError: command=b''",
            "  try: environment=open('/proc/'+name+'/environ','rb').read()",
            "  except OSError: environment=b''",
            "  matched_cmdline=any(value==attempt_bytes or value.startswith(attempt_bytes+b'/') for value in command.split(b'\\0'))",
            "  matched_environment=tag_entry in environment.split(b'\\0')",
            "  if matched_cmdline or matched_environment:",
            "    rows.append({'pid':pid,'matched_cmdline':matched_cmdline,'matched_environment':matched_environment})",
            "print(json.dumps(sorted(rows,key=lambda row:row['pid'])))",
        ])
        rows = json.loads(self._remote([
            "python3",
            "-c",
            script,
            plan["run_tag"],
            plan["paths"]["attempt_root"],
        ]).stdout)
        if not isinstance(rows, list):
            raise RuntimeError("exact-tag scan is invalid")
        return rows

    def _reap_exact_tag(
        self,
        plan: dict,
        observed_rows: list[dict],
    ) -> dict:
        requested = sorted({
            row["pid"]
            for row in observed_rows
            if (
                isinstance(row, dict)
                and type(row.get("pid")) is int
                and row["pid"] > 0
                and (
                    row.get("matched_cmdline") is True
                    or row.get("matched_environment") is True
                )
            )
        })
        if len(requested) != len(observed_rows):
            raise RuntimeError("exact-tag cleanup inventory is invalid")
        script = "\n".join([
            "import json,os,signal,sys,time",
            "tag,attempt_root,*raw=sys.argv[1:]",
            "tag_entry=('TINYLLMFORGE_RUN_TAG='+tag).encode()",
            "attempt_bytes=attempt_root.encode()",
            "excluded={os.getpid(),os.getppid()}",
            "requested=sorted({int(value) for value in raw})",
            "def matches(pid):",
            "  if pid in excluded: return False",
            "  try:",
            "    command=open(f'/proc/{pid}/cmdline','rb').read()",
            "    environment=open(f'/proc/{pid}/environ','rb').read()",
            "  except OSError: return False",
            "  return any(value==attempt_bytes or value.startswith(attempt_bytes+b'/') for value in command.split(b'\\0')) or tag_entry in environment.split(b'\\0')",
            "eligible=[pid for pid in requested if matches(pid)]",
            "for pid in eligible:",
            "  try: os.kill(pid,signal.SIGTERM)",
            "  except ProcessLookupError: pass",
            "deadline=time.monotonic()+10.0",
            "while time.monotonic()<deadline:",
            "  remaining=[pid for pid in eligible if matches(pid)]",
            "  if not remaining: break",
            "  time.sleep(0.1)",
            "remaining=[pid for pid in eligible if matches(pid)]",
            "for pid in remaining:",
            "  try: os.kill(pid,signal.SIGKILL)",
            "  except ProcessLookupError: pass",
            "time.sleep(0.2)",
            "survivors=[pid for pid in eligible if matches(pid)]",
            "print(json.dumps({'requested_pids':requested,'terminated_pids':eligible,'killed_pids':remaining,'remaining_pids':survivors},sort_keys=True))",
        ])
        receipt = json.loads(self._remote_once([
            "python3",
            "-c",
            script,
            plan["run_tag"],
            plan["paths"]["attempt_root"],
            *(str(pid) for pid in requested),
        ]).stdout)
        if (
            not isinstance(receipt, dict)
            or receipt.get("requested_pids") != requested
            or receipt.get("remaining_pids") != []
        ):
            raise RuntimeError("exact-tag owned cleanup failed")
        return receipt

    def owned_cleanup(self, plan: dict, launch: dict) -> dict:
        del launch
        initial = self._scan_exact_tag(plan)
        reap = self._reap_exact_tag(plan, initial) if initial else None
        self._finalize_local_process()
        final = [self._scan_exact_tag(plan) for _ in range(3)]
        returncode = (
            None if self._process is None else self._process.poll()
        )
        rank_rows = [
            {
                "rank": rank,
                "exit_code": returncode,
                "process_group_destroyed": returncode == 0,
            }
            for rank in range(WORLD_SIZE)
        ]
        clean = final == [[], [], []] and returncode == 0
        self._cleanup = {
            "schema_version": CLEANUP_SCHEMA,
            "run_tag": plan["run_tag"],
            "classification": "CLEAN" if clean else "DIRTY",
            "owned_children_remaining": [
                row for scan in final for row in scan
            ],
            "rank_rows": rank_rows,
            "exact_tag_scans": [initial, *final],
            "final_exact_tag_scans": final,
        }
        if reap is not None:
            self._cleanup["reap_receipt"] = reap
        _atomic_write_json(
            self.local_controller_root / "cleanup_receipt.json",
            self._cleanup,
        )
        return dict(self._cleanup)

    def _finalize_local_process(self) -> None:
        process = getattr(self, "_process", None)
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate(timeout=5)

    def download(
        self,
        plan: dict,
        waited: dict,
        cleanup: dict,
    ) -> dict:
        if waited.get("exit_code") != 0:
            raise RuntimeError("worker completion evidence is missing")
        raw_names = (
            "source_identity.json",
            "plan.json",
            "admission.json",
            "phase_rows.jsonl",
            "scratch_rows.jsonl",
            "rank_results.json",
            "process_receipts.json",
            "worker_summary.json",
        )
        self._download_archive(
            remote_root=plan["paths"]["raw_root"],
            names=raw_names,
            local_root=self.local_raw_root,
        )
        self.local_bundle_root.mkdir(parents=True, exist_ok=False)
        for name in raw_names:
            shutil.copy2(
                self.local_raw_root / name,
                self.local_bundle_root / name,
            )
        _atomic_write_json(
            self.local_bundle_root / "cleanup_receipt.json",
            cleanup,
        )
        phase_rows = [
            json.loads(line)
            for line in (
                self.local_bundle_root / "phase_rows.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        scratch_rows = [
            json.loads(line)
            for line in (
                self.local_bundle_root / "scratch_rows.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        _atomic_write_json(
            self.local_bundle_root / "diagnosis.json",
            derive_diagnosis(
                phase_rows=phase_rows,
                scratch_rows=scratch_rows,
            ),
        )
        manifest = write_pre_verification_manifest(
            self.local_bundle_root
        )
        self._upload_bundle(plan)
        return {
            "downloaded": True,
            "bundle_root": str(self.local_bundle_root),
            "manifest_sha256": _sha256_bytes(
                (self.local_bundle_root / "manifest.json").read_bytes()
            ),
            "manifest": manifest,
        }

    def remote_verify(self, plan: dict, downloaded: dict) -> dict:
        if downloaded.get("downloaded") is not True:
            raise RuntimeError("bundle was not downloaded")
        result = run_remote_argv(
            ssh_target=self.ssh_target,
            remote_argv=[
                self.remote_python,
                (
                    f"{plan['paths']['source_root']}/tools/"
                    "verify_tp4_segmented_capture_attribution.py"
                ),
                "--bundle-root",
                plan["paths"]["bundle_root"],
            ],
            control_path=self.control_path,
            timeout_s=self.command_timeout_s,
            retry_count=self.retry_count,
            command_runner=self.local_command_runner,
        )
        if result.returncode not in (0, 1):
            raise RuntimeError(
                result.stderr or "remote verifier failed"
            )
        payload = result.stdout.encode("utf-8")
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as error:
            raise RuntimeError("remote verifier output is invalid") from error
        (
            self.local_controller_root
            / "remote_independent_verification.json"
        ).write_bytes(payload)
        return {"result": parsed, "bytes": payload}

    def local_verify(self, plan: dict, downloaded: dict) -> dict:
        if downloaded.get("downloaded") is not True:
            raise RuntimeError("bundle was not downloaded")
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(
            dir=self.local_controller_root,
            prefix=".frozen-source-",
        ) as directory:
            frozen = Path(directory)
            archive = self.local_command_runner(
                [
                    "git",
                    "-C",
                    str(root),
                    "archive",
                    "--format=tar",
                    plan["source_revision"],
                    "tinyvllm/engine/segmented_capture_attribution.py",
                    "tools/tp4_segmented_capture_attribution_worker.py",
                    "tools/verify_tp4_segmented_capture_attribution.py",
                ],
                capture_output=True,
                check=False,
            )
            if archive.returncode != 0:
                raise RuntimeError("local frozen-source archive failed")
            extracted = self.local_command_runner(
                ["tar", "-xf", "-", "-C", str(frozen)],
                input=archive.stdout,
                capture_output=True,
                check=False,
            )
            if extracted.returncode != 0:
                raise RuntimeError("local frozen-source extraction failed")
            verified = self.local_command_runner(
                [
                    sys.executable,
                    str(
                        frozen
                        / "tools"
                        / "verify_tp4_segmented_capture_attribution.py"
                    ),
                    "--bundle-root",
                    str(self.local_bundle_root),
                ],
                capture_output=True,
                check=False,
                timeout=self.command_timeout_s,
            )
            if verified.returncode not in (0, 1):
                raise RuntimeError(
                    verified.stderr.decode(errors="replace")
                    or "local verifier failed"
                )
            payload = verified.stdout
            parsed = json.loads(payload)
        (self.local_controller_root / "local_independent_verification.json").write_bytes(
            payload
        )
        return {"result": parsed, "bytes": payload}

    def validate_verifier_identity(
        self,
        plan: dict,
        remote: dict,
        local: dict,
    ) -> dict:
        identity = validate_verifier_identity(remote, local)
        if (
            identity["source_revision"] != plan["source_revision"]
            or identity["run_tag"] != plan["run_tag"]
        ):
            raise RuntimeError("verifier source identity disagrees")
        return identity

    def write_post_verification_manifest(
        self,
        plan: dict,
        downloaded: dict,
        remote: dict,
        local: dict,
        verifier_identity: dict,
        cleanup: dict,
    ) -> dict:
        del downloaded, verifier_identity
        cleanup_bytes = _canonical_bytes(cleanup)
        manifest = build_post_verification_manifest(
            bundle_root=self.local_bundle_root,
            remote_verification=remote,
            local_verification=local,
            cleanup_bytes=cleanup_bytes,
            final_live_exact_tag_scan=[],
        )
        validate_post_verification_manifest(
            manifest,
            bundle_root=self.local_bundle_root,
            remote_verification=remote,
            local_verification=local,
            cleanup_bytes=cleanup_bytes,
            final_live_exact_tag_scan=[],
        )
        encoded = _atomic_write_json(
            self.local_controller_root
            / "post_verification_manifest.json",
            manifest,
        )
        self._upload_bytes(
            plan["paths"]["remote_verification_path"],
            remote["bytes"],
        )
        self._upload_bytes(
            (
                f"{plan['paths']['controller_root']}/"
                "local_independent_verification.json"
            ),
            local["bytes"],
        )
        self._upload_bytes(
            plan["paths"]["post_verification_manifest_path"],
            encoded,
        )
        return manifest

    def final_live_exact_tag_scan(self, plan: dict) -> list[dict]:
        rows = self._scan_exact_tag(plan)
        _atomic_write_json(
            self.local_controller_root / "final_live_exact_tag_scan.json",
            rows,
        )
        return rows


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _parse_args(argv=None):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", required=True)
    parser.add_argument(
        "--admission-mode",
        choices=("strict_clean",),
        default="strict_clean",
    )
    parser.add_argument("--ssh-target", default=DEFAULT_SSH_TARGET)
    parser.add_argument("--remote-python", default=DEFAULT_REMOTE_PYTHON)
    parser.add_argument("--control-path")
    parser.add_argument(
        "--command-timeout-s",
        type=int,
        default=DEFAULT_COMMAND_TIMEOUT_S,
    )
    parser.add_argument(
        "--gpu-wait-timeout-s",
        type=int,
        default=DEFAULT_GPU_WAIT_TIMEOUT_S,
    )
    parser.add_argument(
        "--gpu-poll-interval-s",
        type=int,
        default=DEFAULT_GPU_POLL_INTERVAL_S,
    )
    parser.add_argument(
        "--retry-count",
        type=int,
        default=DEFAULT_RETRY_COUNT,
    )
    parser.add_argument("--local-attempt-root", type=Path)
    return parser.parse_args(argv)


def main(argv=None, *, adapter_factory=None) -> int:
    args = _parse_args(argv)
    local_attempt_root = (
        args.local_attempt_root
        if args.local_attempt_root is not None
        else (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "qwen35_hybrid_state"
            / args.run_tag
        )
    )
    if adapter_factory is None:
        adapter_factory = ProductionAdapter
    adapter = adapter_factory(
        run_tag=args.run_tag,
        local_attempt_root=local_attempt_root,
        ssh_target=args.ssh_target,
        remote_python=args.remote_python,
        control_path=args.control_path,
        command_timeout_s=args.command_timeout_s,
        gpu_wait_timeout_s=args.gpu_wait_timeout_s,
        gpu_poll_interval_s=args.gpu_poll_interval_s,
        retry_count=args.retry_count,
    )
    result = monitor_and_run(
        {
            "run_tag": args.run_tag,
            "admission_mode": args.admission_mode,
        },
        adapter,
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
