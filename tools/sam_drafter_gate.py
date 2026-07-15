"""Canonical profiler-only gate for the prompt+dynamic SAM drafter."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import socket
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
        if existing and (
            existing.get("source_commit") == source_commit
            and existing.get("source_dirty") == source_dirty
            and existing.get("model_identifier") == manifest["model_identifier"]
            and existing.get("prompt_sha256") == spec["prompt_sha256"]
            and existing.get("policy") == spec["policy"]
            and existing.get("repetition") == spec["repetition"]
            and existing.get("process", {}).get("returncode") == 0
            and existing.get("profiler_gate_pass") is True
        ):
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
    return {"manifest": manifest, "raw_rows": rows, "event_rows": events}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--python-bin", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--base-seed", type=int, default=20260715)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-dirty", action="store_true")
    parser.add_argument("--host", default=socket.gethostname())
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
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
