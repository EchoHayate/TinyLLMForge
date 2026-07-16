"""Canonical single-sequence gate for adaptive n-gram speculation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import math
import os
import random
import re
import shutil
import socket
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
_NGRAM_PATH = _REPO_ROOT / "tinyvllm" / "speculative" / "ngram.py"
_NGRAM_SPEC = importlib.util.spec_from_file_location("adaptive_ngram_gate_policy", _NGRAM_PATH)
ngram = importlib.util.module_from_spec(_NGRAM_SPEC)
sys.modules["adaptive_ngram_gate_policy"] = ngram
_NGRAM_SPEC.loader.exec_module(ngram)

AdaptiveDraftState = ngram.AdaptiveDraftState
update_adaptive_draft_state = ngram.update_adaptive_draft_state

REQUIRED_UPLOAD_PATHS = (
    "tinyvllm",
    "tools/draft_model_schema.py",
    "tools/profile_ngram_commit.py",
    "tools/adaptive_ngram_gate.py",
)
OWNED_SOURCE_ROOTS = (
    "tinyvllm",
    "tools/draft_model_schema.py",
    "tools/profile_ngram_commit.py",
    "tools/adaptive_ngram_gate.py",
    "tools/test_ngram_speculative.py",
    "tools/test_adaptive_ngram_gate.py",
    "tools/run_adaptive_ngram_gate_remote.sh",
)
MAX_PORT_COLLISION_RETRIES = 3

POLICIES = {
    "baseline": {
        "mode": "baseline-only",
        "draft_policy": "fixed",
        "max_draft_tokens": None,
    },
    "fixed_k1": {
        "mode": "candidate-only",
        "draft_policy": "fixed",
        "max_draft_tokens": 1,
    },
    "fixed_k2": {
        "mode": "candidate-only",
        "draft_policy": "fixed",
        "max_draft_tokens": 2,
    },
    "fixed_k4": {
        "mode": "candidate-only",
        "draft_policy": "fixed",
        "max_draft_tokens": 4,
    },
    "adaptive": {
        "mode": "candidate-only",
        "draft_policy": "adaptive",
        "max_draft_tokens": 4,
    },
}

THRESHOLDS = {
    "adaptive_vs_baseline_min": 0.05,
    "adaptive_vs_best_fixed_min": 0.02,
    "adaptive_near_best_fixed_min": -0.01,
    "adaptive_waste_reduction_vs_k4_min": 0.20,
    "adaptive_zero_cost_reduction_vs_k4_min": 0.15,
    "natural_transition_ratio_min": 0.95,
}

PROMPT_BANK_BASE = (
    {
        "name": "natural_prose",
        "workload_class": "natural",
        "prompt": (
            "Explain why a small engineering team should separate correctness "
            "benchmarks from performance benchmarks. Use concrete examples and "
            "finish with a short recommendation."
        ),
        "max_output_len": 96,
    },
    {
        "name": "structured_mixed",
        "workload_class": "mixed",
        "prompt": (
            "Continue this deterministic checklist with eight more items, keeping "
            "the exact format:\n- check input\n- check output\n- record timing\n"
        ),
        "max_output_len": 96,
    },
    {
        "name": "repeated_long_context",
        "workload_class": "high_repeat",
        "prompt": (
            "alpha beta gamma delta epsilon alpha beta gamma delta epsilon " * 96
            + "\nContinue the sequence exactly:"
        ),
        "max_output_len": 128,
    },
    {
        "name": "transition_heavy",
        "workload_class": "transition_heavy",
        "prompt": (
            "A A A A B B B B C C C C. Now switch topics and explain in natural "
            "language why repeated patterns can stop abruptly, then emit: "
            "A A A A B B B B C C C C."
        ),
        "max_output_len": 112,
    },
)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256_text(payload)


def _git(
    repo_root: Path,
    *args: str,
    input_bytes: bytes | None = None,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        input=input_bytes,
        capture_output=True,
        check=False,
    )


def _checked_git(
    repo_root: Path,
    *args: str,
    input_bytes: bytes | None = None,
) -> bytes:
    result = _git(repo_root, *args, input_bytes=input_bytes)
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(
            f"git {' '.join(args)} failed with {result.returncode}: {stderr}"
        )
    return result.stdout


def _is_owned_source_path(relative_path: str) -> bool:
    normalized = Path(relative_path).as_posix().lstrip("./")
    return any(
        normalized == root or normalized.startswith(root + "/")
        for root in OWNED_SOURCE_ROOTS
    )


def _git_path_set(repo_root: Path, *args: str) -> set[str]:
    output = _checked_git(repo_root, *args)
    return {
        value.decode("utf-8", errors="surrogateescape")
        for value in output.split(b"\0")
        if value
    }


def expand_owned_source_paths(repo_root: Path) -> tuple[str, ...]:
    repo_root = repo_root.resolve()
    paths = []
    for owned_root in OWNED_SOURCE_ROOTS:
        root = repo_root / owned_root
        if not root.exists():
            raise ValueError(f"missing owned source path: {owned_root}")
        if root.is_symlink():
            raise ValueError(f"owned source path is a symlink: {owned_root}")
        candidates = root.rglob("*") if root.is_dir() else (root,)
        for candidate in candidates:
            if candidate.is_symlink():
                raise ValueError(
                    "owned source contains a symlink: "
                    f"{candidate.relative_to(repo_root).as_posix()}"
                )
            if candidate.is_file():
                paths.append(candidate.relative_to(repo_root).as_posix())
            elif candidate.exists() and not candidate.is_dir():
                raise ValueError(
                    "owned source contains a non-regular path: "
                    f"{candidate.relative_to(repo_root).as_posix()}"
                )
    if len(paths) != len(set(paths)):
        raise ValueError("owned source roots overlap")
    return tuple(sorted(paths))


def hash_source_tree(
    source_root: Path,
    relative_paths: tuple[str, ...],
) -> list[dict]:
    source_root = source_root.resolve()
    files = []
    for relative_path in sorted(relative_paths):
        path = source_root / relative_path
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"source path is not a regular file: {relative_path}")
        payload = path.read_bytes()
        files.append({
            "path": relative_path,
            "size_bytes": len(payload),
            "sha256": sha256_bytes(payload),
        })
    return files


def source_tree_sha256(files: list[dict]) -> str:
    canonical = json.dumps(
        files,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return sha256_bytes(canonical)


def _validate_sha256(value, label: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"invalid {label}")


def validate_source_snapshot(
    source_root: Path,
    evidence: dict,
    patch_path: Path,
) -> dict:
    if evidence.get("schema_version") != 1:
        raise ValueError("unsupported source evidence schema")
    base_commit = evidence.get("base_commit")
    if (
        not isinstance(base_commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", base_commit) is None
    ):
        raise ValueError("invalid source base commit")
    _validate_sha256(evidence.get("patch_sha256"), "patch sha256")
    _validate_sha256(evidence.get("tree_sha256"), "source tree sha256")
    if evidence.get("owned_roots") != list(OWNED_SOURCE_ROOTS):
        raise ValueError("owned source roots mismatch")

    patch_payload = patch_path.read_bytes()
    if len(patch_payload) != evidence.get("patch_size_bytes"):
        raise ValueError("patch size mismatch")
    if sha256_bytes(patch_payload) != evidence["patch_sha256"]:
        raise ValueError("patch hash mismatch")

    expected_files = evidence.get("files")
    if not isinstance(expected_files, list):
        raise ValueError("source evidence files must be a list")
    expected_paths = []
    for record in expected_files:
        if not isinstance(record, dict):
            raise ValueError("invalid source file record")
        relative_path = record.get("path")
        if (
            not isinstance(relative_path, str)
            or not relative_path
            or Path(relative_path).is_absolute()
            or ".." in Path(relative_path).parts
            or not _is_owned_source_path(relative_path)
        ):
            raise ValueError("invalid source file path")
        if not isinstance(record.get("size_bytes"), int) or record["size_bytes"] < 0:
            raise ValueError("invalid source file size")
        _validate_sha256(record.get("sha256"), "source file sha256")
        expected_paths.append(relative_path)
    if expected_paths != sorted(expected_paths) or len(expected_paths) != len(
        set(expected_paths)
    ):
        raise ValueError("source file records must be unique and sorted")

    source_root = source_root.resolve()
    actual_paths = []
    for path in source_root.rglob("*"):
        if path.is_symlink():
            raise ValueError("source snapshot contains a symlink")
        if path.is_file():
            actual_paths.append(path.relative_to(source_root).as_posix())
        elif path.exists() and not path.is_dir():
            raise ValueError("source snapshot contains a non-regular path")
    actual_paths.sort()
    if actual_paths != expected_paths:
        raise ValueError("source path set mismatch")

    actual_files = hash_source_tree(source_root, tuple(actual_paths))
    for expected, actual in zip(expected_files, actual_files):
        if (
            expected["size_bytes"] != actual["size_bytes"]
            or expected["sha256"] != actual["sha256"]
        ):
            raise ValueError(f"source file hash mismatch: {expected['path']}")
    actual_tree_sha256 = source_tree_sha256(actual_files)
    if actual_tree_sha256 != evidence["tree_sha256"]:
        raise ValueError("source tree hash mismatch")
    return {
        "valid": True,
        "source_tree_sha256": actual_tree_sha256,
        "file_count": len(actual_files),
    }


def reconstruct_source_snapshot(
    repo_root: Path,
    source_root: Path,
    evidence: dict,
    patch_path: Path,
) -> None:
    repo_root = repo_root.resolve()
    source_root = source_root.resolve()
    if source_root.exists():
        if any(source_root.iterdir()):
            raise ValueError("source reconstruction destination is not empty")
    else:
        source_root.mkdir(parents=True)

    archive = _checked_git(
        repo_root,
        "archive",
        "--format=tar",
        evidence["base_commit"],
        "--",
        *OWNED_SOURCE_ROOTS,
    )
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
        members = tar.getmembers()
        for member in members:
            member_path = Path(member.name)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError("git archive contains an unsafe path")
            if member.issym() or member.islnk():
                raise ValueError("git archive contains a link")
        tar.extractall(source_root, members=members)

    patch_payload = patch_path.read_bytes()
    if patch_payload:
        environment = os.environ.copy()
        environment["GIT_CEILING_DIRECTORIES"] = str(source_root.parent)
        result = subprocess.run(
            [
                "git",
                "apply",
                "--binary",
                "--whitespace=nowarn",
                "-",
            ],
            cwd=source_root,
            env=environment,
            input=patch_payload,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace").strip()
            raise ValueError(
                f"source patch reconstruction failed with "
                f"{result.returncode}: {stderr}"
            )
    validate_source_snapshot(source_root, evidence, patch_path)


def build_source_evidence(repo_root: Path, out_dir: Path) -> dict:
    repo_root = repo_root.resolve()
    out_dir = out_dir.resolve()
    base_commit = _checked_git(repo_root, "rev-parse", "HEAD").decode().strip()
    if re.fullmatch(r"[0-9a-f]{40}", base_commit) is None:
        raise ValueError("git HEAD did not resolve to a full commit")

    changed_paths = _git_path_set(
        repo_root,
        "diff",
        "--name-only",
        "-z",
        base_commit,
        "--",
    )
    outside_changes = sorted(
        path for path in changed_paths if not _is_owned_source_path(path)
    )
    if outside_changes:
        raise ValueError(
            "changed path outside owned source boundary: "
            + ", ".join(outside_changes)
        )
    untracked_paths = _git_path_set(
        repo_root,
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
    )
    untracked_owned = sorted(
        path for path in untracked_paths if _is_owned_source_path(path)
    )
    if untracked_owned:
        raise ValueError(
            "untracked owned source: " + ", ".join(untracked_owned)
        )
    untracked_outside = sorted(
        path for path in untracked_paths if not _is_owned_source_path(path)
    )
    if untracked_outside:
        raise ValueError(
            "untracked path outside owned source boundary: "
            + ", ".join(untracked_outside)
        )

    relative_paths = expand_owned_source_paths(repo_root)
    out_dir.mkdir(parents=True, exist_ok=False)
    staged_source = out_dir / "source"
    for relative_path in relative_paths:
        destination = staged_source / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo_root / relative_path, destination)

    patch_payload = _checked_git(
        repo_root,
        "diff",
        "--binary",
        "--no-ext-diff",
        base_commit,
        "--",
        *OWNED_SOURCE_ROOTS,
    )
    patch_path = out_dir / "source.patch"
    patch_path.write_bytes(patch_payload)
    files = hash_source_tree(staged_source, relative_paths)
    evidence = {
        "schema_version": 1,
        "base_commit": base_commit,
        "dirty": bool(patch_payload),
        "patch_path": "source.patch",
        "patch_sha256": sha256_bytes(patch_payload),
        "patch_size_bytes": len(patch_payload),
        "owned_roots": list(OWNED_SOURCE_ROOTS),
        "files": files,
        "tree_sha256": source_tree_sha256(files),
    }
    _atomic_write_json(out_dir / "source_evidence.json", evidence)
    validate_source_snapshot(staged_source, evidence, patch_path)
    with tempfile.TemporaryDirectory() as temporary:
        reconstruct_source_snapshot(
            repo_root,
            Path(temporary) / "source",
            evidence,
            patch_path,
        )
    return evidence


def validate_source_preflight(preflight: dict, evidence: dict) -> None:
    if not isinstance(preflight, dict) or preflight.get("schema_version") != 1:
        raise ValueError("unsupported source preflight schema")
    if preflight.get("source_tree_sha256") != evidence.get("tree_sha256"):
        raise ValueError("preflight source tree mismatch")
    for field, failure_message in (
        ("source_verify", "remote source verification failed"),
        ("k1_test", "remote K1 test failed"),
    ):
        record = preflight.get(field)
        if not isinstance(record, dict):
            raise ValueError(f"missing {field} preflight record")
        if record.get("returncode") != 0:
            raise ValueError(failure_message)
        _validate_sha256(record.get("stdout_sha256"), f"{field} stdout sha256")
        _validate_sha256(record.get("stderr_sha256"), f"{field} stderr sha256")
    command = preflight["k1_test"].get("command")
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(value, str) and value for value in command)
        or "tools/test_ngram_speculative.py" not in command
    ):
        raise ValueError("invalid remote K1 test command")


PROMPT_BANK = tuple(
    {**item, "prompt_sha256": sha256_text(item["prompt"])}
    for item in PROMPT_BANK_BASE
)


def _run_key(repetition: int, prompt_name: str, policy: str) -> str:
    return f"r{repetition:02d}__{prompt_name}__{policy}"


def build_run_specs(repetitions: int, base_seed: int) -> list[dict]:
    if repetitions <= 0:
        raise ValueError("repetitions must be > 0")
    specs = []
    global_order = 0
    for repetition in range(repetitions):
        batch = []
        for prompt in PROMPT_BANK:
            for policy_name, policy in POLICIES.items():
                batch.append({
                    "run_key": _run_key(repetition, prompt["name"], policy_name),
                    "repetition": repetition,
                    "seed": base_seed + repetition,
                    "prompt_name": prompt["name"],
                    "prompt_class": prompt["workload_class"],
                    "prompt_sha256": prompt["prompt_sha256"],
                    "max_output_len": prompt["max_output_len"],
                    "policy": policy_name,
                    "mode": policy["mode"],
                    "draft_policy": policy["draft_policy"],
                    "max_draft_tokens": policy["max_draft_tokens"],
                    "max_num_seqs": 1,
                })
        random.Random(base_seed + repetition).shuffle(batch)
        for run_order, spec in enumerate(batch):
            spec["run_order"] = run_order
            spec["global_order"] = global_order
            global_order += 1
            specs.append(spec)
    return specs


def build_manifest(
    repetitions: int,
    base_seed: int,
    source_commit: str,
    source_dirty: bool,
    model_path: str,
    model_identifier: str,
    host: str,
    python_bin: str,
    source_evidence: dict,
    source_preflight: dict,
    extra_environment: dict | None = None,
) -> dict:
    if source_commit != source_evidence.get("base_commit"):
        raise ValueError("source commit does not match source evidence")
    if bool(source_dirty) != source_evidence.get("dirty"):
        raise ValueError("source dirty flag does not match source evidence")
    validate_source_preflight(source_preflight, source_evidence)
    specs = build_run_specs(repetitions, base_seed)
    return {
        "schema_version": 2,
        "created_at_unix_s": time.time(),
        "source_commit": source_commit,
        "source_dirty": bool(source_dirty),
        "source_tree_sha256": source_evidence["tree_sha256"],
        "source_evidence": source_evidence,
        "source_preflight": source_preflight,
        "model_path": model_path,
        "model_identifier": model_identifier,
        "host": host,
        "python_bin": python_bin,
        "repetitions": repetitions,
        "base_seed": base_seed,
        "expected_rows": len(specs),
        "policies": POLICIES,
        "prompt_bank": list(PROMPT_BANK),
        "thresholds": THRESHOLDS,
        "run_specs": specs,
        "environment": extra_environment or {},
        "claim_scope": {
            "single_sequence": True,
            "greedy_only": True,
            "ragged_batched_verify": False,
            "production_batch_throughput": False,
            "memory_reduction": False,
        },
    }


def _atomic_write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _prompt_by_name(name: str) -> dict:
    for prompt in PROMPT_BANK:
        if prompt["name"] == name:
            return prompt
    raise KeyError(name)


def _reserve_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _allocate_distinct_ports(used_ports: set[int]) -> tuple[int, int]:
    while True:
        dist_port = _reserve_port()
        master_port = _reserve_port()
        if dist_port != master_port and dist_port not in used_ports and master_port not in used_ports:
            used_ports.update((dist_port, master_port))
            return dist_port, master_port


def _is_retryable_port_collision(returncode: int, output: str) -> bool:
    if returncode == 0:
        return False
    lowered = output.lower()
    return "eaddrinuse" in lowered or "address already in use" in lowered


def _model_identifier(model_path: str) -> str:
    config_path = Path(model_path) / "config.json"
    if config_path.exists():
        try:
            config = _load_json(config_path)
        except (OSError, json.JSONDecodeError):
            config = {}
        name = config.get("_name_or_path")
        if name:
            return str(name)
    return Path(model_path).name


def _profiler_command(
    spec: dict,
    prompt: dict,
    python_bin: str,
    model_path: str,
    process_json: Path,
) -> list[str]:
    command = [
        python_bin,
        str(_THIS_DIR / "profile_ngram_commit.py"),
        "--model", model_path,
        "--prompt", prompt["prompt"],
        "--max-output-len", str(prompt["max_output_len"]),
        "--ignore-eos",
        "--warmup-output-len", str(min(8, prompt["max_output_len"])),
        "--temperature", "0.0",
        "--ngram-size", "5",
        "--max-commit-events", "0",
        "--mode", spec["mode"],
        "--draft-policy", spec["draft_policy"],
        "--max-num-seqs", "1",
        "--max-model-len", "4096",
        "--gpu-memory-utilization", "0.7",
        "--out-json", str(process_json),
    ]
    if spec["policy"] != "baseline":
        command.extend((
            "--draft-source", "ngram",
            "--max-draft-tokens", str(spec["max_draft_tokens"]),
            "--allow-zero-accept",
        ))
    return command


def _normalize_row(
    manifest: dict,
    spec: dict,
    profiler_result: dict | None,
    process: dict,
) -> tuple[dict, list[dict]]:
    prompt = _prompt_by_name(spec["prompt_name"])
    summary = (profiler_result or {}).get("summary", {})
    per_prompt = (profiler_result or {}).get("per_prompt", [])
    prompt_result = per_prompt[0] if len(per_prompt) == 1 else {}
    output_token_ids = list(prompt_result.get("token_ids", []))
    row = {
        **spec,
        "model_path": manifest["model_path"],
        "model_identifier": manifest["model_identifier"],
        "source_commit": manifest["source_commit"],
        "source_dirty": manifest["source_dirty"],
        "source_tree_sha256": manifest["source_tree_sha256"],
        "prompt_tokens": prompt_result.get("prompt_tokens", summary.get("prompt_tokens")),
        "output_tokens": int(prompt_result.get("output_tokens", summary.get("output_tokens", 0)) or 0),
        "output_token_ids": output_token_ids,
        "output_token_sha256": sha256_json(output_token_ids),
        "elapsed_s": summary.get("elapsed_s"),
        "output_tokens_per_s": summary.get("output_tokens_per_s"),
        "proposal_events": int(summary.get("commit_attempts", 0) or 0),
        "no_draft_positions": int(summary.get("no_draft_steps", 0) or 0),
        "drafted_tokens": int(summary.get("drafted_tokens", 0) or 0),
        "accepted_tokens": int(summary.get("accepted_count", 0) or 0),
        "wasted_draft_tokens": int(summary.get("wasted_draft_tokens", 0) or 0),
        "draft_waste_rate": float(summary.get("draft_waste_rate", 0.0) or 0.0),
        "zero_accept_events": int(summary.get("zero_accept_events", 0) or 0),
        "zero_accept_event_rate": float(summary.get("zero_accept_event_rate", 0.0) or 0.0),
        "zero_accept_verify_ms": float(summary.get("zero_accept_verify_ms", 0.0) or 0.0),
        "verify_timing_ms": summary.get("verify_timing_ms", {}),
        "selected_k_counts": summary.get("selected_k_counts", {"1": 0, "2": 0, "4": 0}),
        "autoregressive_steps_avoided": int(
            summary.get("candidate_autoregressive_steps_avoided", 0) or 0
        ),
        "profiler_gate_pass": summary.get("gate_pass") is True,
        "profiler_gate_fail_reasons": list(summary.get("gate_fail_reasons", [])),
        "process": process,
    }
    events = []
    for event_index, event in enumerate((profiler_result or {}).get("verify_events", [])):
        events.append({
            "run_key": spec["run_key"],
            "policy": spec["policy"],
            "prompt_name": prompt["name"],
            "repetition": spec["repetition"],
            "event_index": event_index,
            **event,
        })
    return row, events


def run_gate(
    out_dir: Path,
    python_bin: str,
    model_path: str,
    repetitions: int,
    base_seed: int,
    host: str,
    resume: bool,
    source_root: Path,
    source_evidence_path: Path,
    source_patch_path: Path,
    source_preflight_path: Path,
    extra_environment: dict | None = None,
) -> dict:
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    source_root = source_root.resolve()
    source_evidence = _load_json(source_evidence_path)
    source_preflight = _load_json(source_preflight_path)
    validate_source_snapshot(
        source_root,
        source_evidence,
        source_patch_path,
    )
    validate_source_preflight(source_preflight, source_evidence)
    manifest_path = out_dir / "manifest.json"
    manifest = build_manifest(
        repetitions=repetitions,
        base_seed=base_seed,
        source_commit=source_evidence["base_commit"],
        source_dirty=source_evidence["dirty"],
        model_path=model_path,
        model_identifier=_model_identifier(model_path),
        host=host,
        python_bin=python_bin,
        source_evidence=source_evidence,
        source_preflight=source_preflight,
        extra_environment=extra_environment,
    )
    if manifest_path.exists() and resume:
        existing_manifest = _load_json(manifest_path)
        comparable_keys = (
            "source_commit",
            "source_dirty",
            "source_tree_sha256",
            "source_evidence",
            "source_preflight",
            "model_path",
            "repetitions",
            "base_seed",
            "expected_rows",
            "thresholds",
            "prompt_bank",
        )
        if any(existing_manifest.get(key) != manifest.get(key) for key in comparable_keys):
            raise ValueError("resume manifest does not match requested gate")
        manifest = existing_manifest
        validate_materialized_source_artifacts(out_dir, manifest=manifest)
    else:
        _atomic_write_json(manifest_path, manifest)
        materialize_source_artifacts(
            out_dir,
            source_root,
            source_evidence_path,
            source_patch_path,
            source_preflight_path,
        )

    raw_path = out_dir / "raw_rows.json"
    event_path = out_dir / "event_rows.json"
    raw_rows = _load_json(raw_path) if raw_path.exists() and resume else []
    event_rows = _load_json(event_path) if event_path.exists() and resume else []
    existing_keys = [row.get("run_key") for row in raw_rows]
    if len(existing_keys) != len(set(existing_keys)):
        raise ValueError("duplicate run keys in resumable raw rows")
    completed = set(existing_keys)
    used_ports = {
        int(port)
        for row in raw_rows
        for port in (
            row.get("process", {}).get("tinyvllm_dist_port"),
            row.get("process", {}).get("master_port"),
        )
        if port is not None
    }
    logs_dir = out_dir / "logs"
    process_dir = out_dir / "process_json"
    logs_dir.mkdir(exist_ok=True)
    process_dir.mkdir(exist_ok=True)

    for spec in manifest["run_specs"]:
        if spec["run_key"] in completed:
            continue
        prompt = _prompt_by_name(spec["prompt_name"])
        process_json = process_dir / f"{spec['run_key']}.json"
        stdout_path = logs_dir / f"{spec['run_key']}.stdout.log"
        stderr_path = logs_dir / f"{spec['run_key']}.stderr.log"
        command = _profiler_command(spec, prompt, python_bin, model_path, process_json)
        started = time.time()
        port_attempts = []
        for attempt_index in range(MAX_PORT_COLLISION_RETRIES + 1):
            dist_port, master_port = _allocate_distinct_ports(used_ports)
            environment = os.environ.copy()
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            environment["PYTHONPATH"] = str(_REPO_ROOT)
            environment["TINYVLLM_DIST_PORT"] = str(dist_port)
            environment["MASTER_PORT"] = str(master_port)
            process_json.unlink(missing_ok=True)
            completed_process = subprocess.run(
                command,
                cwd=_REPO_ROOT,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            port_attempts.append({
                "attempt": attempt_index + 1,
                "tinyvllm_dist_port": dist_port,
                "master_port": master_port,
                "returncode": completed_process.returncode,
                "port_collision": _is_retryable_port_collision(
                    completed_process.returncode,
                    completed_process.stdout + completed_process.stderr,
                ),
            })
            if not port_attempts[-1]["port_collision"]:
                break
        finished = time.time()
        stdout_path.write_text(completed_process.stdout, encoding="utf-8")
        stderr_path.write_text(completed_process.stderr, encoding="utf-8")
        profiler_result = None
        parse_error = None
        if process_json.exists():
            try:
                profiler_result = _load_json(process_json)
            except (OSError, json.JSONDecodeError) as exc:
                parse_error = str(exc)
        process = {
            "returncode": completed_process.returncode,
            "command": command,
            "tinyvllm_dist_port": dist_port,
            "master_port": master_port,
            "port_attempts": port_attempts,
            "stdout_path": str(stdout_path.relative_to(out_dir)),
            "stderr_path": str(stderr_path.relative_to(out_dir)),
            "process_json_path": str(process_json.relative_to(out_dir)),
            "started_at_unix_s": started,
            "finished_at_unix_s": finished,
            "parse_error": parse_error,
        }
        row, events = _normalize_row(manifest, spec, profiler_result, process)
        raw_rows.append(row)
        event_rows.extend(events)
        completed.add(spec["run_key"])
        _atomic_write_json(raw_path, raw_rows)
        _atomic_write_json(event_path, event_rows)

    summary = summarize_rows(manifest, raw_rows, event_rows)
    report = render_report(manifest, summary)
    _atomic_write_json(out_dir / "summary.json", summary)
    _atomic_write_text(out_dir / "report.md", report)
    return summary


def replay_adaptive_trajectory(events: list[dict]) -> dict:
    state = AdaptiveDraftState()
    failures = []
    for index, event in enumerate(events):
        transition = event.get("adaptive_transition")
        if not isinstance(transition, dict):
            failures.append(f"event[{index}] missing adaptive_transition")
            continue
        if int(event.get("selected_k", -1)) != state.selected_k:
            failures.append(
                f"event[{index}] selected_k={event.get('selected_k')} expected={state.selected_k}"
            )
        try:
            expected = update_adaptive_draft_state(
                state,
                proposed=int(event["proposed_tokens"]),
                accepted=int(event["accepted_count"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            failures.append(f"event[{index}] invalid counts: {exc}")
            continue
        if transition != expected:
            failures.append(f"event[{index}] transition mismatch")
    return {
        "valid": not failures,
        "fail_reasons": failures,
        "events": len(events),
        "final_state": {
            "selected_k": state.selected_k,
            "acceptance_ema": state.acceptance_ema,
            "full_accept_streak": state.full_accept_streak,
            "proposal_events": state.proposal_events,
        },
    }


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def _safe_reduction(new_value: float, reference: float) -> float:
    if reference == 0.0:
        return 1.0 if new_value == 0.0 else float("-inf")
    return 1.0 - new_value / reference


def _structural_failures(manifest: dict, raw_rows: list[dict]) -> list[str]:
    failures = []
    expected = int(manifest["expected_rows"])
    if len(raw_rows) != expected:
        failures.append(f"row_count={len(raw_rows)} expected={expected}")
    keys = [row.get("run_key") for row in raw_rows]
    if len(keys) != len(set(keys)):
        failures.append("duplicate_run_keys")
    expected_keys = {spec["run_key"] for spec in manifest["run_specs"]}
    if set(keys) != expected_keys:
        failures.append("run_key_set_mismatch")
    ports = []
    for row in raw_rows:
        process = row.get("process", {})
        for field in ("source_commit", "source_dirty", "source_tree_sha256"):
            if row.get(field) != manifest.get(field):
                failures.append(
                    f"{row.get('run_key')}:{field}_mismatch"
                )
        if process.get("returncode") != 0:
            failures.append(f"{row.get('run_key')}:process_returncode={process.get('returncode')}")
        if row.get("profiler_gate_pass") is not True:
            failures.append(f"{row.get('run_key')}:profiler_gate_pass=false")
        if row.get("max_num_seqs") != 1:
            failures.append(f"{row.get('run_key')}:max_num_seqs!=1")
        prompt_tokens = row.get("prompt_tokens")
        if not isinstance(prompt_tokens, int) or prompt_tokens <= 0:
            failures.append(f"{row.get('run_key')}:prompt_tokens_invalid")
        dist_port = process.get("tinyvllm_dist_port")
        master_port = process.get("master_port")
        if dist_port is None or master_port is None or dist_port == master_port:
            failures.append(f"{row.get('run_key')}:invalid_ports")
        else:
            ports.extend((dist_port, master_port))
        for field in ("elapsed_s", "output_tokens_per_s"):
            value = row.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                failures.append(f"{row.get('run_key')}:{field}_invalid")
    if len(ports) != len(set(ports)):
        failures.append("reused_ports")
    return failures


def summarize_rows(manifest: dict, raw_rows: list[dict], event_rows: list[dict]) -> dict:
    structural_failures = _structural_failures(manifest, raw_rows)
    summary = {
        "schema_version": 1,
        "decision": "INCOMPLETE",
        "decision_reasons": [],
        "expected_rows": int(manifest["expected_rows"]),
        "observed_rows": len(raw_rows),
        "structural_failures": structural_failures,
        "correctness_pass": False,
        "trajectory_replay_pass": False,
        "adaptive_exercise_pass": False,
    }
    if structural_failures:
        summary["decision_reasons"] = ["incomplete_artifacts_or_processes"]
        return summary

    rows_by_key = {row["run_key"]: row for row in raw_rows}
    correctness_failures = []
    for row in raw_rows:
        if row["policy"] == "baseline":
            continue
        baseline_key = _run_key(row["repetition"], row["prompt_name"], "baseline")
        baseline = rows_by_key.get(baseline_key)
        if baseline is None:
            correctness_failures.append(f"{row['run_key']}:missing_baseline")
            continue
        if row["output_token_ids"] != baseline["output_token_ids"]:
            correctness_failures.append(f"{row['run_key']}:output_mismatch")
        if row["output_tokens"] != baseline["output_tokens"]:
            correctness_failures.append(f"{row['run_key']}:output_length_mismatch")

    adaptive_events_by_run = {}
    for event in event_rows:
        if event.get("policy") == "adaptive":
            adaptive_events_by_run.setdefault(event["run_key"], []).append(event)
    trajectory_failures = []
    transition_reasons = set()
    selected_levels = set()
    for row in raw_rows:
        if row["policy"] != "adaptive":
            continue
        events = sorted(
            adaptive_events_by_run.get(row["run_key"], []),
            key=lambda item: item.get("event_index", 0),
        )
        replay = replay_adaptive_trajectory(events)
        if not replay["valid"]:
            trajectory_failures.extend(
                f"{row['run_key']}:{reason}" for reason in replay["fail_reasons"]
            )
        for event in events:
            selected_levels.add(int(event["selected_k"]))
            transition = event.get("adaptive_transition", {})
            if "selected_k_after" in transition:
                selected_levels.add(int(transition["selected_k_after"]))
            transition_reasons.add(transition.get("transition_reason"))

    repeat_capable = {"structured_mixed", "repeated_long_context", "transition_heavy"}
    proposal_coverage = {
        prompt_name: any(
            row["policy"] == "adaptive"
            and row["prompt_name"] == prompt_name
            and row["proposal_events"] > 0
            for row in raw_rows
        )
        for prompt_name in repeat_capable
    }
    adaptive_exercise_failures = []
    if not all(proposal_coverage.values()):
        adaptive_exercise_failures.append("missing_repeat_capable_proposal")
    if len(selected_levels) < 2:
        adaptive_exercise_failures.append("fewer_than_two_selected_levels")
    if not ({"promote", "weak_acceptance", "zero_accept"} & transition_reasons):
        adaptive_exercise_failures.append("missing_promotion_or_demotion")

    summary["correctness_failures"] = correctness_failures
    summary["trajectory_failures"] = trajectory_failures
    summary["adaptive_exercise_failures"] = adaptive_exercise_failures
    summary["correctness_pass"] = not correctness_failures
    summary["trajectory_replay_pass"] = not trajectory_failures
    summary["adaptive_exercise_pass"] = not adaptive_exercise_failures
    summary["adaptive_selected_levels"] = sorted(selected_levels)
    summary["adaptive_transition_reasons"] = sorted(
        reason for reason in transition_reasons if reason
    )
    if correctness_failures or trajectory_failures:
        summary["decision"] = "NO_GO"
        summary["decision_reasons"] = ["mandatory_correctness_failed"]
        return summary

    repetitions = int(manifest["repetitions"])
    aggregate_by_policy = {policy: [] for policy in POLICIES}
    waste_by_policy = {policy: [] for policy in POLICIES}
    zero_cost_by_policy = {policy: [] for policy in POLICIES}
    per_prompt_by_policy = {
        prompt["name"]: {policy: [] for policy in POLICIES}
        for prompt in PROMPT_BANK
    }
    for repetition in range(repetitions):
        for policy in POLICIES:
            policy_rows = [
                row for row in raw_rows
                if row["repetition"] == repetition and row["policy"] == policy
            ]
            total_tokens = sum(int(row["output_tokens"]) for row in policy_rows)
            total_elapsed = sum(float(row["elapsed_s"]) for row in policy_rows)
            aggregate_by_policy[policy].append(total_tokens / total_elapsed)
            waste_by_policy[policy].append(
                float(sum(int(row["wasted_draft_tokens"]) for row in policy_rows))
            )
            zero_cost_by_policy[policy].append(
                sum(float(row["zero_accept_verify_ms"]) for row in policy_rows)
            )
            for row in policy_rows:
                per_prompt_by_policy[row["prompt_name"]][policy].append(
                    float(row["output_tokens_per_s"])
                )

    aggregate_medians = {
        policy: _median(values)
        for policy, values in aggregate_by_policy.items()
    }
    per_prompt_medians = {
        prompt_name: {
            policy: _median(values)
            for policy, values in policy_values.items()
        }
        for prompt_name, policy_values in per_prompt_by_policy.items()
    }
    waste_medians = {
        policy: _median(values)
        for policy, values in waste_by_policy.items()
    }
    zero_cost_medians = {
        policy: _median(values)
        for policy, values in zero_cost_by_policy.items()
    }
    fixed_policies = ("fixed_k1", "fixed_k2", "fixed_k4")
    best_fixed_policy = max(fixed_policies, key=aggregate_medians.get)
    baseline_tps = aggregate_medians["baseline"]
    adaptive_tps = aggregate_medians["adaptive"]
    best_fixed_tps = aggregate_medians[best_fixed_policy]
    adaptive_vs_baseline = adaptive_tps / baseline_tps - 1.0
    adaptive_vs_best_fixed = adaptive_tps / best_fixed_tps - 1.0
    waste_reduction = _safe_reduction(
        waste_medians["adaptive"],
        waste_medians["fixed_k4"],
    )
    zero_cost_reduction = _safe_reduction(
        zero_cost_medians["adaptive"],
        zero_cost_medians["fixed_k4"],
    )
    protected_prompt_ratios = {
        prompt_name: (
            per_prompt_medians[prompt_name]["adaptive"]
            / per_prompt_medians[prompt_name]["baseline"]
        )
        for prompt_name in ("natural_prose", "transition_heavy")
    }

    summary.update({
        "aggregate_throughput_by_repetition": aggregate_by_policy,
        "aggregate_throughput_medians": aggregate_medians,
        "per_prompt_throughput_medians": per_prompt_medians,
        "wasted_draft_tokens_medians": waste_medians,
        "zero_accept_verify_ms_medians": zero_cost_medians,
        "best_fixed_policy": best_fixed_policy,
        "adaptive_vs_baseline": adaptive_vs_baseline,
        "adaptive_vs_best_fixed": adaptive_vs_best_fixed,
        "adaptive_waste_reduction_vs_k4": waste_reduction,
        "adaptive_zero_cost_reduction_vs_k4": zero_cost_reduction,
        "protected_prompt_ratios": protected_prompt_ratios,
        "thresholds": manifest["thresholds"],
    })

    reasons = []
    thresholds = manifest["thresholds"]
    if not summary["adaptive_exercise_pass"]:
        reasons.append("adaptive_exercise_failed")
    if adaptive_vs_baseline < thresholds["adaptive_vs_baseline_min"]:
        reasons.append("adaptive_vs_baseline_gate_failed")
    direct_fixed_win = (
        adaptive_vs_best_fixed >= thresholds["adaptive_vs_best_fixed_min"]
    )
    efficient_near_tie = (
        adaptive_vs_best_fixed >= thresholds["adaptive_near_best_fixed_min"]
        and waste_reduction >= thresholds["adaptive_waste_reduction_vs_k4_min"]
        and zero_cost_reduction >= thresholds["adaptive_zero_cost_reduction_vs_k4_min"]
    )
    if not (direct_fixed_win or efficient_near_tie):
        reasons.append("adaptive_vs_fixed_gate_failed")
    if any(
        ratio < thresholds["natural_transition_ratio_min"]
        for ratio in protected_prompt_ratios.values()
    ):
        reasons.append("natural_or_transition_regression")

    summary["decision"] = "NO_GO" if reasons else "GO"
    summary["decision_reasons"] = reasons or ["all_committed_gates_passed"]
    return summary


def render_report(manifest: dict, summary: dict) -> str:
    lines = [
        "# Adaptive N-Gram Speculation Gate",
        "",
        f"- Decision: **{summary['decision']}**",
        f"- Reasons: `{', '.join(summary.get('decision_reasons', []))}`",
        f"- Rows: `{summary.get('observed_rows', 0)}/{summary.get('expected_rows', 0)}`",
        f"- Source: `{manifest['source_commit']}` (dirty={manifest['source_dirty']})",
        f"- Model: `{manifest['model_identifier']}` at `{manifest['model_path']}`",
        f"- Host/Python: `{manifest['host']}` / `{manifest['python_bin']}`",
        "",
        "## Aggregate Throughput",
        "",
        "| Policy | Median tok/s |",
        "|---|---:|",
    ]
    for policy, value in summary.get("aggregate_throughput_medians", {}).items():
        lines.append(f"| {policy} | {value:.6f} |")
    lines.extend((
        "",
        "## Per-Prompt Throughput",
        "",
        "| Prompt | Baseline | Fixed K1 | Fixed K2 | Fixed K4 | Adaptive |",
        "|---|---:|---:|---:|---:|---:|",
    ))
    for prompt_name, values in summary.get("per_prompt_throughput_medians", {}).items():
        lines.append(
            f"| {prompt_name} | {values['baseline']:.6f} | "
            f"{values['fixed_k1']:.6f} | {values['fixed_k2']:.6f} | "
            f"{values['fixed_k4']:.6f} | {values['adaptive']:.6f} |"
        )
    lines.extend((
        "",
        "## Audits",
        "",
        f"- Correctness: `{summary.get('correctness_pass')}`",
        f"- Trajectory replay: `{summary.get('trajectory_replay_pass')}`",
        f"- Adaptive exercise: `{summary.get('adaptive_exercise_pass')}`",
        f"- Selected levels: `{summary.get('adaptive_selected_levels', [])}`",
        f"- Transition reasons: `{summary.get('adaptive_transition_reasons', [])}`",
        "",
        "## Fixed Thresholds",
        "",
        "```json",
        json.dumps(manifest["thresholds"], indent=2, sort_keys=True),
        "```",
        "",
        "## Claim Boundaries",
        "",
        "This decision covers only greedy single-sequence Qwen3-0.6B runs on the "
        "recorded host and prompt bank. It does not establish ragged batched "
        "verification correctness, production batch throughput, queueing-tail "
        "latency, memory-capacity reduction, or transfer to other models.",
        "",
        "A GO should be followed by a separate ragged batched target-verify and "
        "load-aware K=0..N design. A NO_GO should retain the correctness and "
        "measurement machinery while preferring the best measured fixed policy "
        "only in its validated regime or moving to a higher-quality draft source.",
        "",
    ))
    return "\n".join(lines)


def materialize_source_artifacts(
    out_dir: Path,
    source_root: Path,
    evidence_path: Path,
    patch_path: Path,
    preflight_path: Path,
) -> None:
    evidence = _load_json(evidence_path)
    preflight = _load_json(preflight_path)
    validate_source_snapshot(source_root, evidence, patch_path)
    validate_source_preflight(preflight, evidence)
    destination = out_dir / "source"
    if destination.exists():
        shutil.rmtree(destination)
    for record in evidence["files"]:
        relative_path = record["path"]
        target = destination / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_root / relative_path, target)
    shutil.copyfile(evidence_path, out_dir / "source_evidence.json")
    shutil.copyfile(patch_path, out_dir / "source.patch")
    shutil.copyfile(preflight_path, out_dir / "source_preflight.json")
    validate_source_snapshot(
        destination,
        evidence,
        out_dir / "source.patch",
    )


def validate_materialized_source_artifacts(
    out_dir: Path,
    manifest: dict | None = None,
) -> tuple[dict, dict]:
    out_dir = out_dir.resolve()
    evidence = _load_json(out_dir / "source_evidence.json")
    preflight = _load_json(out_dir / "source_preflight.json")
    if manifest is None and (out_dir / "manifest.json").is_file():
        manifest = _load_json(out_dir / "manifest.json")
    if manifest is not None:
        if manifest.get("source_evidence") != evidence:
            raise ValueError("manifest source evidence mismatch")
        if manifest.get("source_preflight") != preflight:
            raise ValueError("manifest source preflight mismatch")
        if manifest.get("source_tree_sha256") != evidence.get("tree_sha256"):
            raise ValueError("manifest source tree mismatch")
    validate_source_snapshot(
        out_dir / "source",
        evidence,
        out_dir / "source.patch",
    )
    validate_source_preflight(preflight, evidence)
    return evidence, preflight


def verify_artifacts(
    out_dir: Path,
    repo_root: Path = _REPO_ROOT,
) -> dict:
    out_dir = out_dir.resolve()
    manifest = _load_json(out_dir / "manifest.json")
    evidence, _ = validate_materialized_source_artifacts(
        out_dir,
        manifest=manifest,
    )
    with tempfile.TemporaryDirectory() as temporary:
        reconstruct_source_snapshot(
            repo_root,
            Path(temporary) / "source",
            evidence,
            out_dir / "source.patch",
        )
    raw_rows = _load_json(out_dir / "raw_rows.json")
    event_rows = _load_json(out_dir / "event_rows.json")
    expected_summary = summarize_rows(manifest, raw_rows, event_rows)
    actual_summary = _load_json(out_dir / "summary.json")
    if actual_summary != expected_summary:
        raise ValueError("summary.json does not match independent recomputation")
    expected_report = render_report(manifest, expected_summary)
    actual_report = (out_dir / "report.md").read_text(encoding="utf-8")
    if actual_report != expected_report:
        raise ValueError("report.md does not match independent regeneration")
    return expected_summary


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--out-dir", type=Path, required=True)
    run_parser.add_argument("--python-bin", required=True)
    run_parser.add_argument("--model-path", required=True)
    run_parser.add_argument("--repetitions", type=int, choices=[1, 7], required=True)
    run_parser.add_argument("--base-seed", type=int, default=20260714)
    run_parser.add_argument("--source-root", type=Path, required=True)
    run_parser.add_argument("--source-evidence", type=Path, required=True)
    run_parser.add_argument("--source-patch", type=Path, required=True)
    run_parser.add_argument("--source-preflight", type=Path, required=True)
    run_parser.add_argument("--host", required=True)
    run_parser.add_argument("--resume", action="store_true")

    snapshot_parser = subparsers.add_parser("snapshot-source")
    snapshot_parser.add_argument("--repo-root", type=Path, required=True)
    snapshot_parser.add_argument("--out-dir", type=Path, required=True)

    source_verify_parser = subparsers.add_parser("verify-source")
    source_verify_parser.add_argument("--source-root", type=Path, required=True)
    source_verify_parser.add_argument("--evidence", type=Path, required=True)
    source_verify_parser.add_argument("--patch", type=Path, required=True)

    summarize_parser = subparsers.add_parser("summarize")
    summarize_parser.add_argument("--out-dir", type=Path, required=True)

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "run":
        summary = run_gate(
            out_dir=args.out_dir,
            python_bin=args.python_bin,
            model_path=args.model_path,
            repetitions=args.repetitions,
            base_seed=args.base_seed,
            host=args.host,
            resume=args.resume,
            source_root=args.source_root,
            source_evidence_path=args.source_evidence,
            source_patch_path=args.source_patch,
            source_preflight_path=args.source_preflight,
            extra_environment={
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
        )
    elif args.command == "snapshot-source":
        summary = build_source_evidence(args.repo_root, args.out_dir)
    elif args.command == "verify-source":
        evidence = _load_json(args.evidence)
        summary = validate_source_snapshot(
            args.source_root,
            evidence,
            args.patch,
        )
    elif args.command == "summarize":
        manifest = _load_json(args.out_dir / "manifest.json")
        raw_rows = _load_json(args.out_dir / "raw_rows.json")
        event_rows = _load_json(args.out_dir / "event_rows.json")
        summary = summarize_rows(manifest, raw_rows, event_rows)
        _atomic_write_json(args.out_dir / "summary.json", summary)
        _atomic_write_text(args.out_dir / "report.md", render_report(manifest, summary))
    else:
        summary = verify_artifacts(args.out_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary.get("decision") == "INCOMPLETE":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
