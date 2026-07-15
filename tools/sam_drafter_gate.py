"""Canonical profiler-only gate for the prompt+dynamic SAM drafter."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import socket
import statistics
import subprocess
import time
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent

REQUIRED_UPLOAD_PATHS = (
    "tinyvllm",
    "tools/draft_model_schema.py",
    "tools/profile_ngram_commit.py",
    "tools/sam_drafter_gate.py",
)
MAX_PORT_COLLISION_RETRIES = 3

POLICIES = {
    "baseline": {
        "mode": "baseline-only",
        "draft_source": "ngram",
        "draft_policy": "fixed",
        "max_draft_tokens": None,
    },
    "ngram_fixed_k4": {
        "mode": "candidate-only",
        "draft_source": "ngram",
        "draft_policy": "fixed",
        "max_draft_tokens": 4,
    },
    "ngram_adaptive": {
        "mode": "candidate-only",
        "draft_source": "ngram",
        "draft_policy": "adaptive",
        "max_draft_tokens": 4,
    },
    "sam_fixed_k16": {
        "mode": "candidate-only",
        "draft_source": "sam",
        "draft_policy": "sam-fixed",
        "max_draft_tokens": 16,
    },
    "sam_match_aware": {
        "mode": "candidate-only",
        "draft_source": "sam",
        "draft_policy": "sam-match-aware",
        "max_draft_tokens": 16,
    },
}

THRESHOLDS = {
    "sam_vs_baseline_min": 0.10,
    "sam_vs_ngram_k4_min": 0.03,
    "sam_near_ngram_k4_min": -0.01,
    "verify_attempt_reduction_min": 0.25,
    "draft_waste_reduction_min": 0.25,
    "critical_prompt_speedup_min": -0.05,
}

PROMPT_BANK_BASE = (
    {
        "name": "natural_prose",
        "workload_class": "natural",
        "prompt": (
            "Explain why benchmark correctness must be established before "
            "performance tuning. Use two concrete engineering examples and "
            "finish with a concise recommendation."
        ),
        "max_output_len": 96,
    },
    {
        "name": "structured_code_like",
        "workload_class": "structured",
        "prompt": (
            "Continue this exact checklist format with twelve new lines:\n"
            "- validate input\n- run baseline\n- compare output\n- record timing\n"
        ),
        "max_output_len": 112,
    },
    {
        "name": "repeated_long_context",
        "workload_class": "high_repeat",
        "prompt": (
            "alpha beta gamma delta epsilon " * 128
            + "\nContinue the token pattern exactly:"
        ),
        "max_output_len": 128,
    },
    {
        "name": "transition_heavy",
        "workload_class": "transition_heavy",
        "prompt": (
            "A A A A B B B B C C C C. Now explain in natural language why "
            "a repeated pattern can stop abruptly, then emit the original "
            "A/B/C sequence one more time."
        ),
        "max_output_len": 112,
    },
    {
        "name": "prompt_copy_retrieval",
        "workload_class": "prompt_copy",
        "prompt": (
            "Reference block:\n"
            "BEGIN ALPHA\nid: 17\nstatus: verified\nowner: inference\nEND ALPHA\n"
            "Reference block:\n"
            "BEGIN BETA\nid: 29\nstatus: pending\nowner: runtime\nEND BETA\n"
            "Copy the complete ALPHA block exactly, then explain its status."
        ),
        "max_output_len": 112,
    },
)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
        order = list(POLICIES)
        random.Random(base_seed + repetition).shuffle(order)
        for prompt in PROMPT_BANK:
            for policy in order:
                config = POLICIES[policy]
                specs.append({
                    "run_key": _run_key(repetition, prompt["name"], policy),
                    "repetition": repetition,
                    "prompt_name": prompt["name"],
                    "prompt_class": prompt["workload_class"],
                    "prompt_sha256": prompt["prompt_sha256"],
                    "policy": policy,
                    "mode": config["mode"],
                    "draft_source": config["draft_source"],
                    "draft_policy": config["draft_policy"],
                    "max_draft_tokens": config["max_draft_tokens"],
                    "max_num_seqs": 1,
                    "seed": base_seed + repetition,
                    "global_order": global_order,
                })
                global_order += 1
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
    extra_environment: dict | None = None,
) -> dict:
    specs = build_run_specs(repetitions, base_seed)
    return {
        "schema_version": 1,
        "gate": "sam_drafter_canonical",
        "created_unix_s": time.time(),
        "source_commit": source_commit,
        "source_dirty": bool(source_dirty),
        "model_path": model_path,
        "model_identifier": model_identifier,
        "host": host,
        "python_bin": python_bin,
        "repetitions": repetitions,
        "base_seed": base_seed,
        "expected_rows": len(specs),
        "policies": POLICIES,
        "thresholds": THRESHOLDS,
        "prompt_bank": list(PROMPT_BANK),
        "run_specs": specs,
        "required_upload_paths": list(REQUIRED_UPLOAD_PATHS),
        "extra_environment": dict(extra_environment or {}),
        "claim_scope": {
            "single_sequence": True,
            "greedy_only": True,
            "profiler_owned": True,
            "ragged_batched_verify": False,
            "production_batch_throughput": False,
            "queue_tail_latency": False,
            "memory_reduction": False,
        },
    }


def _atomic_write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    os.replace(temporary, path)


def _atomic_write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload)
    os.replace(temporary, path)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load_json(path: Path):
    return json.loads(path.read_text())


def _prompt_by_name(name: str) -> dict:
    for prompt in PROMPT_BANK:
        if prompt["name"] == name:
            return prompt
    raise KeyError(name)


def _reserve_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


def _allocate_distinct_ports(used_ports: set[int]) -> tuple[int, int]:
    ports = []
    while len(ports) < 2:
        port = _reserve_port()
        if port not in used_ports and port not in ports:
            ports.append(port)
    used_ports.update(ports)
    return ports[0], ports[1]


def _is_retryable_port_collision(returncode: int, output: str) -> bool:
    if returncode == 0:
        return False
    normalized = output.lower()
    return "address already in use" in normalized or "eaddrinuse" in normalized


def _model_identifier(model_path: str) -> str:
    path = Path(model_path)
    return f"{path.name}:{sha256_text(str(path.resolve()))[:16]}"


def _profiler_command(
    spec: dict,
    prompt: dict,
    python_bin: str,
    model_path: str,
    process_json: Path,
) -> list[str]:
    command = [
        python_bin,
        "tools/profile_ngram_commit.py",
        "--model",
        model_path,
        "--prompt",
        prompt["prompt"],
        "--max-output-len",
        str(prompt["max_output_len"]),
        "--ignore-eos",
        "--warmup-output-len",
        str(min(8, prompt["max_output_len"])),
        "--temperature",
        "0.0",
        "--max-commit-events",
        "0",
        "--max-num-seqs",
        "1",
        "--max-model-len",
        "4096",
        "--gpu-memory-utilization",
        "0.7",
        "--mode",
        spec["mode"],
        "--out-json",
        str(process_json),
    ]
    if spec["policy"] != "baseline":
        command.extend([
            "--draft-source",
            spec["draft_source"],
            "--draft-policy",
            spec["draft_policy"],
            "--max-draft-tokens",
            str(spec["max_draft_tokens"]),
            "--allow-zero-accept",
        ])
        if spec["draft_source"] == "ngram":
            command.extend(["--ngram-size", "5"])
    return command


def _row_is_resumable(manifest: dict, spec: dict, row: dict) -> bool:
    elapsed_s = row.get("elapsed_s")
    throughput = row.get("output_tokens_per_s")
    return (
        row.get("source_commit") == manifest.get("source_commit")
        and row.get("source_dirty") == manifest.get("source_dirty")
        and row.get("model_identifier") == manifest.get("model_identifier")
        and row.get("prompt_sha256") == spec.get("prompt_sha256")
        and row.get("policy") == spec.get("policy")
        and row.get("repetition") == spec.get("repetition")
        and row.get("process", {}).get("returncode") == 0
        and row.get("profiler_gate_pass") is True
        and isinstance(elapsed_s, (int, float))
        and math.isfinite(elapsed_s)
        and elapsed_s > 0
        and isinstance(throughput, (int, float))
        and math.isfinite(throughput)
        and throughput > 0
    )


def _normalize_row(
    manifest: dict,
    spec: dict,
    profiler_result: dict | None,
    process: dict,
) -> tuple[dict, list[dict]]:
    result = profiler_result or {}
    summary = result.get("summary", {})
    per_prompt = result.get("per_prompt", [])
    prompt_result = per_prompt[0] if per_prompt else {}
    token_ids = prompt_result.get(
        "token_ids",
        result.get("token_ids", []),
    )
    row = {
        **spec,
        "source_commit": manifest["source_commit"],
        "source_dirty": manifest["source_dirty"],
        "model_identifier": manifest["model_identifier"],
        "prompt_tokens": prompt_result.get("prompt_tokens"),
        "output_tokens": summary.get(
            "output_tokens",
            prompt_result.get("output_tokens"),
        ),
        "output_token_ids": token_ids,
        "output_token_sha256": sha256_json(token_ids),
        "elapsed_s": summary.get("elapsed_s"),
        "output_tokens_per_s": summary.get("output_tokens_per_s"),
        "proposal_events": sum(
            1 for event in result.get("sam_events", [])
            if event.get("event_type") == "proposal"
        ) if spec["draft_source"] == "sam" else summary.get("commit_attempts", 0),
        "verify_attempts": summary.get("commit_attempts", 0),
        "no_draft_positions": summary.get("no_draft_steps", 0),
        "drafted_tokens": summary.get("drafted_tokens", 0),
        "accepted_tokens": summary.get("accepted_count", 0),
        "wasted_draft_tokens": summary.get("wasted_draft_tokens", 0),
        "zero_accept_events": summary.get("zero_accept_events", 0),
        "zero_accept_verify_ms": summary.get("zero_accept_verify_ms", 0.0),
        "selected_k_counts": summary.get("selected_k_counts", {}),
        "sam_build_ms": summary.get("sam_build_ms", 0.0),
        "sam_extension_ms": summary.get("sam_extension_ms", 0.0),
        "sam_lookup_ms": summary.get("sam_lookup_ms", 0.0),
        "sam_state_count": summary.get("sam_state_count", 0),
        "sam_indexed_tokens": summary.get("sam_indexed_tokens", 0),
        "sam_bypass_count": summary.get("sam_bypass_count", 0),
        "runtime_mutation": summary.get("runtime_mutation", False),
        "profiler_owned": summary.get("profiler_owned", True),
        "profiler_gate_pass": summary.get("gate_pass", False),
        "profiler_gate_fail_reasons": summary.get("gate_fail_reasons", []),
        "process": process,
    }
    events = []
    seen = set()
    for event in result.get("verify_events", []) + result.get("sam_events", []):
        event_type = event.get("event_type", "verify")
        stable_key = (
            f"{spec['run_key']}:{event.get('step', -1)}:"
            f"{event_type}:{event.get('candidate_seq_id', -1)}"
        )
        if stable_key in seen:
            continue
        seen.add(stable_key)
        events.append({
            **event,
            "event_key": stable_key,
            "event_index": len(events),
            "run_key": spec["run_key"],
            "policy": spec["policy"],
            "prompt_name": spec["prompt_name"],
            "prompt_class": spec["prompt_class"],
            "repetition": spec["repetition"],
        })
    return row, events


def run_gate(
    out_dir: Path,
    python_bin: str,
    model_path: str,
    repetitions: int,
    base_seed: int,
    source_commit: str,
    source_dirty: bool,
    host: str,
    resume: bool,
    extra_environment: dict | None = None,
) -> dict:
    out_dir = Path(out_dir)
    run_data_dir = out_dir.parent / f"{out_dir.name}.runs"
    logs_dir = run_data_dir / "logs"
    process_dir = run_data_dir / "process_json"
    logs_dir.mkdir(parents=True, exist_ok=True)
    process_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(
        repetitions,
        base_seed,
        source_commit,
        source_dirty,
        model_path,
        _model_identifier(model_path),
        host,
        python_bin,
        extra_environment,
    )
    _atomic_write_json(out_dir / "manifest.json", manifest)
    raw_path = out_dir / "raw_rows.json"
    event_path = out_dir / "event_rows.json"
    rows = _load_json(raw_path) if resume and raw_path.exists() else []
    events = _load_json(event_path) if resume and event_path.exists() else []
    rows_by_key = {row["run_key"]: row for row in rows}
    used_ports = set()
    for spec in manifest["run_specs"]:
        existing = rows_by_key.get(spec["run_key"])
        if existing and _row_is_resumable(manifest, spec, existing):
            continue
        rows = [row for row in rows if row["run_key"] != spec["run_key"]]
        events = [
            event for event in events if event["run_key"] != spec["run_key"]
        ]
        prompt = _prompt_by_name(spec["prompt_name"])
        process_json = process_dir / f"{spec['run_key']}.json"
        stdout_path = logs_dir / f"{spec['run_key']}.stdout.log"
        stderr_path = logs_dir / f"{spec['run_key']}.stderr.log"
        process = {}
        profiler_result = None
        for attempt in range(MAX_PORT_COLLISION_RETRIES + 1):
            dist_port, master_port = _allocate_distinct_ports(used_ports)
            env = os.environ.copy()
            env.update(extra_environment or {})
            env["TINYVLLM_DIST_PORT"] = str(dist_port)
            env["MASTER_PORT"] = str(master_port)
            command = _profiler_command(
                spec,
                prompt,
                python_bin,
                model_path,
                process_json,
            )
            completed = subprocess.run(
                command,
                cwd=_REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
            )
            stdout_path.write_text(completed.stdout)
            stderr_path.write_text(completed.stderr)
            process = {
                "returncode": completed.returncode,
                "command": command,
                "tinyvllm_dist_port": dist_port,
                "master_port": master_port,
                "attempt": attempt,
                "stdout_path": str(stdout_path.relative_to(out_dir.parent)),
                "stderr_path": str(stderr_path.relative_to(out_dir.parent)),
                "process_json": str(process_json.relative_to(out_dir.parent)),
            }
            if completed.returncode == 0 and process_json.exists():
                profiler_result = _load_json(process_json)
                break
            if not _is_retryable_port_collision(
                completed.returncode,
                completed.stdout + completed.stderr,
            ):
                break
        row, normalized_events = _normalize_row(
            manifest,
            spec,
            profiler_result,
            process,
        )
        rows.append(row)
        events.extend(normalized_events)
        _atomic_write_json(raw_path, rows)
        _atomic_write_json(event_path, events)
    summary = summarize_rows(manifest, rows, events)
    _write_canonical_artifacts(out_dir, manifest, rows, events, summary)
    return {
        "manifest": manifest,
        "raw_rows": rows,
        "event_rows": events,
        "summary": summary,
    }


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def reconcile_run_trace(row: dict, events: list[dict]) -> dict:
    if not row["policy"].startswith("sam_"):
        return {"valid": True, "fail_reasons": []}
    failures = []
    proposal_events = [
        event for event in events if event.get("event_type") == "proposal"
    ]
    verify_events = [
        event for event in events if event.get("event_type") == "verify"
    ]
    bypass_events = [
        event for event in events if event.get("event_type") == "bypass"
    ]
    integrity_events = [
        event for event in events
        if event.get("event_type") == "index_integrity"
    ]
    checks = {
        "verify_attempts": (
            row.get("verify_attempts"),
            len(verify_events),
        ),
        "drafted_tokens": (
            row.get("drafted_tokens"),
            sum(int(event.get("proposed_tokens", 0)) for event in proposal_events),
        ),
        "accepted_tokens": (
            row.get("accepted_tokens"),
            sum(int(event.get("accepted_count", 0)) for event in verify_events),
        ),
        "zero_accept_events": (
            row.get("zero_accept_events"),
            sum(int(event.get("accepted_count", 0)) == 0 for event in verify_events),
        ),
        "sam_bypass_count": (
            row.get("sam_bypass_count"),
            len(bypass_events),
        ),
    }
    for field, (observed, expected) in checks.items():
        if observed != expected:
            failures.append(f"{field}_trace_mismatch:{observed}!={expected}")
    if row.get("wasted_draft_tokens") != (
        int(row.get("drafted_tokens", 0)) - int(row.get("accepted_tokens", 0))
    ):
        failures.append("wasted_draft_tokens_trace_mismatch")
    selected_counts = {
        str(level): sum(
            int(event.get("selected_k", -1)) == level
            for event in proposal_events
        )
        for level in (0, 4, 8, 16)
    }
    normalized_counts = {
        str(level): int(row.get("selected_k_counts", {}).get(str(level), 0))
        for level in (0, 4, 8, 16)
    }
    if selected_counts != normalized_counts:
        failures.append("selected_k_counts_trace_mismatch")
    for proposal in proposal_events:
        matching = [
            event
            for event in (
                bypass_events
                if int(proposal.get("proposed_tokens", 0)) == 0
                else verify_events
            )
            if event.get("step") == proposal.get("step")
            and event.get("candidate_seq_id") == proposal.get("candidate_seq_id")
        ]
        if len(matching) != 1:
            failures.append(
                f"proposal_terminal_event_mismatch:step={proposal.get('step')}"
            )
    if not integrity_events:
        failures.append("missing_index_integrity_event")
    else:
        if any(event.get("history_match") is not True for event in integrity_events):
            failures.append("index_integrity_history_mismatch")
        if int(integrity_events[-1].get("index_token_count", -1)) != (
            int(row.get("prompt_tokens", -1))
            + int(row.get("output_tokens", -1))
        ):
            failures.append("index_integrity_token_count_mismatch")
    if any(event.get("runtime_mutation") is not False for event in events):
        failures.append("event_runtime_mutation_not_false")
    if any(event.get("profiler_owned") is not True for event in events):
        failures.append("event_profiler_owned_not_true")
    return {"valid": not failures, "fail_reasons": failures}


def _structural_failures(
    manifest: dict,
    raw_rows: list[dict],
    event_rows: list[dict],
) -> list[str]:
    failures = []
    expected_specs = {
        spec["run_key"]: spec for spec in manifest.get("run_specs", [])
    }
    rows_by_key = {}
    for row in raw_rows:
        key = row.get("run_key")
        if key in rows_by_key:
            failures.append(f"duplicate_row:{key}")
        rows_by_key[key] = row
    if set(rows_by_key) != set(expected_specs):
        failures.append("run_key_set_mismatch")
    if len(raw_rows) != manifest.get("expected_rows"):
        failures.append("row_count_mismatch")
    used_ports = set()
    for key, row in rows_by_key.items():
        spec = expected_specs.get(key)
        if spec is None:
            continue
        for field in (
            "source_commit",
            "source_dirty",
            "model_identifier",
        ):
            if row.get(field) != manifest.get(field):
                failures.append(f"{key}:{field}_mismatch")
        for field in ("prompt_sha256", "policy", "repetition"):
            if row.get(field) != spec.get(field):
                failures.append(f"{key}:{field}_mismatch")
        process = row.get("process", {})
        if process.get("returncode") != 0:
            failures.append(f"{key}:process_failed")
        for port_field in ("tinyvllm_dist_port", "master_port"):
            port = process.get(port_field)
            if not isinstance(port, int) or port <= 0:
                failures.append(f"{key}:{port_field}_invalid")
            elif port in used_ports:
                failures.append(f"{key}:{port_field}_reused")
            else:
                used_ports.add(port)
        if row.get("profiler_gate_pass") is not True:
            failures.append(f"{key}:profiler_gate_failed")
        prompt_tokens = row.get("prompt_tokens")
        if not isinstance(prompt_tokens, int) or prompt_tokens <= 0:
            failures.append(f"{key}:prompt_tokens_invalid")
        for field in ("elapsed_s", "output_tokens_per_s"):
            value = row.get(field)
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                failures.append(f"{key}:{field}_invalid")
        for field in (
            "output_tokens",
            "proposal_events",
            "verify_attempts",
            "no_draft_positions",
            "drafted_tokens",
            "accepted_tokens",
            "wasted_draft_tokens",
            "zero_accept_events",
            "sam_bypass_count",
        ):
            value = row.get(field)
            if not isinstance(value, int) or value < 0:
                failures.append(f"{key}:{field}_invalid")
        if row["policy"].startswith("sam_"):
            if row.get("runtime_mutation") is not False:
                failures.append(f"{key}:runtime_mutation_not_false")
            if row.get("profiler_owned") is not True:
                failures.append(f"{key}:profiler_owned_not_true")
    events_by_run = {}
    for event in event_rows:
        events_by_run.setdefault(event.get("run_key"), []).append(event)
    for key, row in rows_by_key.items():
        trace = reconcile_run_trace(row, events_by_run.get(key, []))
        failures.extend(f"{key}:{reason}" for reason in trace["fail_reasons"])
    return failures


def summarize_rows(
    manifest: dict,
    raw_rows: list[dict],
    event_rows: list[dict],
) -> dict:
    structural_failures = _structural_failures(
        manifest,
        raw_rows,
        event_rows,
    )
    rows_by_pair = {}
    for row in raw_rows:
        rows_by_pair.setdefault(
            (row.get("repetition"), row.get("prompt_name")),
            {},
        )[row.get("policy")] = row
    correctness_failures = []
    required_policies = set(POLICIES)
    for pair, policies in rows_by_pair.items():
        if set(policies) != required_policies:
            structural_failures.append(f"{pair}:paired_policy_set_mismatch")
            continue
        baseline_ids = policies["baseline"].get("output_token_ids")
        baseline_hash = sha256_json(baseline_ids)
        for policy, row in policies.items():
            if row.get("output_token_sha256") != sha256_json(
                row.get("output_token_ids")
            ):
                correctness_failures.append(
                    f"{pair}:{policy}:output_hash_mismatch"
                )
            if sha256_json(row.get("output_token_ids")) != baseline_hash:
                correctness_failures.append(
                    f"{pair}:{policy}:output_mismatch"
                )
    paired_speedups = {
        "sam_vs_baseline": {prompt["name"]: [] for prompt in PROMPT_BANK},
        "sam_vs_ngram_k4": {prompt["name"]: [] for prompt in PROMPT_BANK},
    }
    verify_reductions = []
    waste_reductions = []
    for (repetition, prompt_name), policies in rows_by_pair.items():
        if set(policies) != required_policies:
            continue
        sam = policies["sam_match_aware"]
        baseline = policies["baseline"]
        ngram = policies["ngram_fixed_k4"]
        throughputs = (
            sam.get("output_tokens_per_s"),
            baseline.get("output_tokens_per_s"),
            ngram.get("output_tokens_per_s"),
        )
        if any(
            not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
            for value in throughputs
        ):
            continue
        paired_speedups["sam_vs_baseline"][prompt_name].append(
            sam["output_tokens_per_s"] / baseline["output_tokens_per_s"] - 1.0
        )
        paired_speedups["sam_vs_ngram_k4"][prompt_name].append(
            sam["output_tokens_per_s"] / ngram["output_tokens_per_s"] - 1.0
        )
        if ngram["verify_attempts"] > 0:
            verify_reductions.append(
                1.0 - sam["verify_attempts"] / ngram["verify_attempts"]
            )
        if ngram["wasted_draft_tokens"] > 0:
            waste_reductions.append(
                1.0
                - sam["wasted_draft_tokens"] / ngram["wasted_draft_tokens"]
            )
    if not verify_reductions:
        structural_failures.append("missing_positive_verify_reference")
    if not waste_reductions:
        structural_failures.append("missing_positive_waste_reference")
    all_vs_baseline = [
        value
        for values in paired_speedups["sam_vs_baseline"].values()
        for value in values
    ]
    all_vs_ngram = [
        value
        for values in paired_speedups["sam_vs_ngram_k4"].values()
        for value in values
    ]
    sam_events = [
        event for event in event_rows
        if event.get("policy") == "sam_match_aware"
    ]
    proposal_events = [
        event for event in sam_events if event.get("event_type") == "proposal"
    ]
    verify_events = [
        event for event in sam_events if event.get("event_type") == "verify"
    ]
    selected_levels = {
        int(event.get("selected_k", -1)) for event in proposal_events
    }
    regions = {
        event.get("draft_metadata", {}).get("continuation_region")
        for event in proposal_events
    }
    exercise_failures = []
    for level in (0, 4, 8, 16):
        if level not in selected_levels:
            exercise_failures.append(f"missing_selected_k_{level}")
    for region in ("prompt", "generated"):
        if region not in regions:
            exercise_failures.append(f"missing_continuation_region_{region}")
    if not any(int(event.get("accepted_count", -1)) == 0 for event in verify_events):
        exercise_failures.append("missing_zero_accept_verify")
    if not any(
        int(event.get("proposed_tokens", 0)) > 1
        and int(event.get("accepted_count", -1))
        == int(event.get("proposed_tokens", 0))
        for event in verify_events
    ):
        exercise_failures.append("missing_full_multi_token_accept")
    trace_failures = [
        reason
        for reason in structural_failures
        if "trace_mismatch" in reason
        or "proposal_terminal_event_mismatch" in reason
        or "index_integrity" in reason
        or "missing_index_integrity" in reason
    ]
    evidence_failures = (
        structural_failures + correctness_failures + exercise_failures
    )
    sam_vs_baseline = _median(all_vs_baseline) if all_vs_baseline else None
    sam_vs_ngram = _median(all_vs_ngram) if all_vs_ngram else None
    verify_reduction = (
        _median(verify_reductions) if verify_reductions else None
    )
    waste_reduction = _median(waste_reductions) if waste_reductions else None
    critical_prompt_medians = {
        prompt["name"]: (
            _median(paired_speedups["sam_vs_baseline"][prompt["name"]])
            if paired_speedups["sam_vs_baseline"][prompt["name"]]
            else None
        )
        for prompt in PROMPT_BANK
    }
    performance_failures = []
    if not evidence_failures:
        if sam_vs_baseline < THRESHOLDS["sam_vs_baseline_min"]:
            performance_failures.append("sam_vs_baseline_gate_failed")
        direct_win = sam_vs_ngram >= THRESHOLDS["sam_vs_ngram_k4_min"]
        efficient_near_tie = (
            sam_vs_ngram >= THRESHOLDS["sam_near_ngram_k4_min"]
            and verify_reduction
            >= THRESHOLDS["verify_attempt_reduction_min"]
            and waste_reduction >= THRESHOLDS["draft_waste_reduction_min"]
        )
        if not (direct_win or efficient_near_tie):
            performance_failures.append("sam_vs_ngram_gate_failed")
        for prompt_name, speedup in critical_prompt_medians.items():
            if speedup < THRESHOLDS["critical_prompt_speedup_min"]:
                performance_failures.append(
                    f"critical_prompt_regression:{prompt_name}"
                )
    if evidence_failures:
        decision = "INCOMPLETE"
        decision_reasons = evidence_failures
    elif performance_failures:
        decision = "NO_GO"
        decision_reasons = performance_failures
    else:
        decision = "GO"
        decision_reasons = []
    throughput_by_policy = {
        policy: _median([
            row["output_tokens_per_s"]
            for row in raw_rows
            if row.get("policy") == policy
            and isinstance(row.get("output_tokens_per_s"), (int, float))
            and math.isfinite(row["output_tokens_per_s"])
        ])
        for policy in POLICIES
        if any(row.get("policy") == policy for row in raw_rows)
    }
    return {
        "decision": decision,
        "decision_reasons": decision_reasons,
        "expected_rows": manifest.get("expected_rows"),
        "observed_rows": len(raw_rows),
        "structural_failures": structural_failures,
        "correctness_failures": correctness_failures,
        "exercise_failures": exercise_failures,
        "correctness_pass": not correctness_failures,
        "trace_reconciliation_pass": not trace_failures,
        "policy_exercise_pass": not exercise_failures,
        "paired_speedups": paired_speedups,
        "median_sam_vs_baseline": sam_vs_baseline,
        "median_sam_vs_ngram_k4": sam_vs_ngram,
        "median_verify_attempt_reduction": verify_reduction,
        "median_draft_waste_reduction": waste_reduction,
        "critical_prompt_medians": critical_prompt_medians,
        "throughput_by_policy": throughput_by_policy,
        "sam_cpu_overhead_ms": {
            field: sum(float(row.get(field, 0.0)) for row in raw_rows)
            for field in ("sam_build_ms", "sam_extension_ms", "sam_lookup_ms")
        },
        "thresholds": dict(THRESHOLDS),
        "claim_scope": manifest.get("claim_scope", {}),
    }


def render_report(manifest: dict, summary: dict) -> str:
    reasons = summary.get("decision_reasons", [])
    lines = [
        "# SAM Drafter Canonical Gate",
        "",
        "## Decision",
        "",
        f"- Decision: `{summary['decision']}`",
        f"- Reasons: {', '.join(reasons) if reasons else 'none'}",
        "",
        "## Environment",
        "",
        f"- Host: `{manifest['host']}`",
        f"- Source commit: `{manifest['source_commit']}`",
        f"- Source dirty: `{manifest['source_dirty']}`",
        f"- Model: `{manifest['model_identifier']}`",
        "",
        "## Completeness",
        "",
        f"- Rows: `{summary['observed_rows']}/{summary['expected_rows']}`",
        f"- Correctness pass: `{summary['correctness_pass']}`",
        (
            "- Trace reconciliation pass: "
            f"`{summary['trace_reconciliation_pass']}`"
        ),
        f"- Policy exercise pass: `{summary['policy_exercise_pass']}`",
        "",
        "## Median Throughput",
        "",
    ]
    for policy, throughput in summary.get("throughput_by_policy", {}).items():
        lines.append(f"- `{policy}`: `{throughput:.6f}` tok/s")
    lines.extend([
        "",
        "## Paired Metrics",
        "",
        (
            "- SAM vs baseline: "
            f"`{summary['median_sam_vs_baseline']}`"
        ),
        (
            "- SAM vs n-gram K4: "
            f"`{summary['median_sam_vs_ngram_k4']}`"
        ),
        (
            "- Verify-attempt reduction: "
            f"`{summary['median_verify_attempt_reduction']}`"
        ),
        (
            "- Draft-waste reduction: "
            f"`{summary['median_draft_waste_reduction']}`"
        ),
        "",
        "## Critical Prompts",
        "",
    ])
    for prompt, speedup in summary.get("critical_prompt_medians", {}).items():
        lines.append(f"- `{prompt}`: `{speedup}`")
    lines.extend([
        "",
        "## Policy Exercise",
        "",
        f"- Failures: `{summary.get('exercise_failures', [])}`",
        "",
        "## SAM CPU Overhead",
        "",
    ])
    for field, value in summary.get("sam_cpu_overhead_ms", {}).items():
        lines.append(f"- `{field}`: `{value:.6f}` ms")
    lines.extend([
        "",
        "## Fixed Thresholds",
        "",
        f"```json\n{json.dumps(summary['thresholds'], indent=2, sort_keys=True)}\n```",
        "",
        "## Claim Boundaries",
        "",
        f"```json\n{json.dumps(summary['claim_scope'], indent=2, sort_keys=True)}\n```",
        "",
        "## Next Direction",
        "",
        (
            "- Proceed to broader but still profiler-owned validation."
            if summary["decision"] == "GO"
            else "- Stop performance promotion and inspect failed evidence or thresholds."
            if summary["decision"] == "NO_GO"
            else "- Repair incomplete evidence and rerun without changing thresholds."
        ),
        "",
    ])
    return "\n".join(lines)


def _canonical_json_bytes(payload) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _write_canonical_artifacts(
    out_dir: Path,
    manifest: dict,
    raw_rows: list[dict],
    event_rows: list[dict],
    summary: dict,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_bytes = _canonical_json_bytes(manifest)
    raw_bytes = _canonical_json_bytes(raw_rows)
    event_bytes = _canonical_json_bytes(event_rows)
    stored_summary = {
        **summary,
        "input_artifact_sha256": {
            "manifest.json": sha256_bytes(manifest_bytes),
            "raw_rows.json": sha256_bytes(raw_bytes),
            "event_rows.json": sha256_bytes(event_bytes),
        },
    }
    _atomic_write_json(out_dir / "manifest.json", manifest)
    _atomic_write_json(out_dir / "raw_rows.json", raw_rows)
    _atomic_write_json(out_dir / "event_rows.json", event_rows)
    _atomic_write_json(out_dir / "summary.json", stored_summary)
    _atomic_write_text(
        out_dir / "report.md",
        render_report(manifest, stored_summary),
    )


def verify_artifacts(out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    expected = {
        "manifest.json",
        "raw_rows.json",
        "event_rows.json",
        "summary.json",
        "report.md",
    }
    observed = {path.name for path in out_dir.iterdir() if path.is_file()}
    if observed != expected:
        raise ValueError(
            f"canonical file set mismatch: expected={sorted(expected)} "
            f"observed={sorted(observed)}"
        )
    manifest_path = out_dir / "manifest.json"
    raw_path = out_dir / "raw_rows.json"
    event_path = out_dir / "event_rows.json"
    stored_summary = _load_json(out_dir / "summary.json")
    expected_hashes = {
        "manifest.json": sha256_bytes(manifest_path.read_bytes()),
        "raw_rows.json": sha256_bytes(raw_path.read_bytes()),
        "event_rows.json": sha256_bytes(event_path.read_bytes()),
    }
    if stored_summary.get("input_artifact_sha256") != expected_hashes:
        raise ValueError("input artifact hash mismatch")
    manifest = _load_json(manifest_path)
    rows = _load_json(raw_path)
    events = _load_json(event_path)
    regenerated = summarize_rows(manifest, rows, events)
    regenerated["input_artifact_sha256"] = expected_hashes
    if regenerated != stored_summary:
        raise ValueError("summary.json regeneration mismatch")
    expected_report = render_report(manifest, regenerated)
    if (out_dir / "report.md").read_text() != expected_report:
        raise ValueError("report.md regeneration mismatch")
    return regenerated


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--out-dir", type=Path, required=True)
    run_parser.add_argument("--python-bin", required=True)
    run_parser.add_argument("--model-path", required=True)
    run_parser.add_argument("--repetitions", type=int, default=7)
    run_parser.add_argument("--base-seed", type=int, default=20260715)
    run_parser.add_argument("--source-commit", required=True)
    run_parser.add_argument("--source-dirty", action="store_true")
    run_parser.add_argument("--host", default=socket.gethostname())
    run_parser.add_argument("--resume", action="store_true")
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "verify":
        print(json.dumps(verify_artifacts(args.out_dir), indent=2))
        return
    if args.command != "run":
        raise ValueError("command must be run or verify")
    run_gate(
        args.out_dir,
        args.python_bin,
        args.model_path,
        args.repetitions,
        args.base_seed,
        args.source_commit,
        args.source_dirty,
        args.host,
        args.resume,
    )


if __name__ == "__main__":
    main()
