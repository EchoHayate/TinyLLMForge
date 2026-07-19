"""Production arrival-load gate contracts and offline orchestration helpers."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import socket
import statistics
import subprocess
import tarfile
import time
from collections import Counter
from pathlib import Path


SCHEMA_VERSION = 1
GENERATOR_VERSION = 1

OWNED_SOURCE_ROOTS = (
    "tinyvllm",
    "tools/source_audit.py",
    "tools/arrival_load_gate.py",
    "tools/arrival_load_driver.py",
    "tools/arrival_load_verify.py",
    "tools/test_arrival_load_gate.py",
    "tools/test_arrival_load_driver.py",
    "tools/test_arrival_load_verify.py",
    "tools/test_chunked_prefill.py",
    "tools/run_arrival_load_gate_remote.sh",
    "tools/test_run_arrival_load_gate_remote.py",
)

IGNORED_UNTRACKED_PREFIXES = (
    "experiments/adaptive_ngram/20260717-k1-sam-canonical",
    "experiments/adaptive_ngram/20260717-k1-sam-smoke-r2",
    "experiments/adaptive_ngram/20260717-k1-sam-smoke",
    "experiments/arrival_load",
    "experiments/speculation_router",
)

FINAL_ARTIFACT_FILES = (
    "run_manifest.json",
    "calibration_manifest.jsonl",
    "calibration_rows.jsonl",
    "workload_manifest.jsonl",
    "request_timeline.jsonl",
    "scheduler_trace.jsonl",
    "memory_trace.jsonl",
    "case_rows.jsonl",
    "summary.json",
    "report.md",
    "source_evidence.json",
    "source.patch",
    "source_snapshot.tar.gz",
    "artifact_hashes.json",
)

COMMON_ENGINE_CONFIG = {
    "max_num_batched_tokens": 16384,
    "max_num_seqs": 512,
    "max_num_prefill_tokens_per_step": 128,
    "enforce_eager": False,
}

POLICY_FIELDS = (
    "chunked_prefill_decode_first",
    "chunked_prefill_max_consecutive_chunks",
    "chunked_prefill_mixed_batch",
    "chunked_prefill_mixed_min_prompt_tokens",
)

POLICY_OVERRIDES = {
    "P0": {},
    "P1": {
        "chunked_prefill_decode_first": True,
        "chunked_prefill_max_consecutive_chunks": 0,
        "chunked_prefill_mixed_batch": False,
        "chunked_prefill_mixed_min_prompt_tokens": 0,
    },
    "P2": {
        "chunked_prefill_decode_first": False,
        "chunked_prefill_max_consecutive_chunks": 2,
        "chunked_prefill_mixed_batch": False,
        "chunked_prefill_mixed_min_prompt_tokens": 0,
    },
    "P3": {
        "chunked_prefill_decode_first": False,
        "chunked_prefill_max_consecutive_chunks": 0,
        "chunked_prefill_mixed_batch": True,
        "chunked_prefill_mixed_min_prompt_tokens": 0,
    },
}

PROMPT_CLASS_TARGET_TOKENS = {
    "short": 64,
    "medium": 512,
    "long": 1536,
}
OUTPUT_CLASS_TOKENS = {
    "short": 16,
    "long": 64,
}

CALIBRATION_INITIAL_RATE_RPS = 0.5
CALIBRATION_MAX_DOUBLINGS = 8
CALIBRATION_BISECTION_STEPS = 3
CALIBRATION_REQUESTS_PER_RATE = 24
CALIBRATION_DRAIN_TIMEOUT_NS = 120_000_000_000

CANONICAL_WARMUP_REQUESTS = 8
CANONICAL_MEASURED_REQUESTS = 64
FAIRNESS_REQUESTS_PER_BUCKET = 20
CANONICAL_DRAIN_TIMEOUT_NS = 120_000_000_000
STARVATION_DEADLINE_NS = 5_000_000_000
MEASURED_REPETITIONS = 3
POLICY_ORDER_BY_REPETITION = {
    0: ("P0", "P2", "P3"),
    1: ("P2", "P3", "P0"),
    2: ("P3", "P0", "P2"),
}
_REPO_ROOT = Path(__file__).resolve().parent.parent

CANONICAL_SCENARIOS = (
    "steady_moderate",
    "near_saturation",
    "overload",
    "burst",
    "long_prompt_pressure",
    "mixed_service_fairness",
)

ARRIVAL_SEEDS = {
    "steady_moderate": 601,
    "near_saturation": 901,
    "overload": 1201,
    "burst": 1701,
    "long_prompt_pressure": 1901,
    "mixed_service_fairness": 2301,
}

SCENARIO_RATE_MULTIPLIERS = {
    "steady_moderate": 0.6,
    "near_saturation": 0.9,
    "overload": 1.2,
    "burst": 0.9,
    "long_prompt_pressure": 0.9,
    "mixed_service_fairness": 0.9,
}


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def environment_identity_sha256(evidence: dict) -> str:
    if not isinstance(evidence, dict):
        raise ValueError("environment evidence must be an object")
    identity = {
        key: value
        for key, value in evidence.items()
        if key not in {"run_tag", "tinyvllm_file"}
    }
    return canonical_json_sha256(identity)


def nearest_rank(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("nearest_rank requires finite samples")
    if not math.isfinite(percentile) or not 0.0 < percentile <= 1.0:
        raise ValueError("percentile must be finite and in (0, 1]")
    normalized = [float(value) for value in values]
    if not all(math.isfinite(value) for value in normalized):
        raise ValueError("nearest_rank requires finite samples")
    normalized.sort()
    index = math.ceil(len(normalized) * percentile) - 1
    return normalized[index]


def _prompt_record(
    prompt_id: str,
    prompt_class: str,
    prompt: str,
    prompt_token_ids: list[int],
) -> dict:
    record = {
        "prompt_id": prompt_id,
        "prompt": prompt,
        "prompt_token_ids": [int(value) for value in prompt_token_ids],
        "prompt_token_count": len(prompt_token_ids),
        "prompt_class": prompt_class,
    }
    record["prompt_sha256"] = canonical_json_sha256({
        "prompt": record["prompt"],
        "prompt_token_ids": record["prompt_token_ids"],
    })
    return record


def build_prompt_bank(tokenizer, *, model_id: str) -> dict:
    prompts = []
    seed_text = (
        "TinyLLMForge deterministic arrival load prompt token "
        "scheduling fairness latency throughput memory evidence "
    )
    for prompt_class, target_tokens in PROMPT_CLASS_TARGET_TOKENS.items():
        repetitions = 1
        token_ids = []
        prompt = ""
        while len(token_ids) < target_tokens:
            prompt = (seed_text * repetitions).strip()
            token_ids = list(tokenizer.encode(prompt))
            repetitions *= 2
        token_ids = token_ids[:target_tokens]
        prompts.append(_prompt_record(
            f"{prompt_class}-0",
            prompt_class,
            prompt,
            token_ids,
        ))
    bank = {
        "schema_version": SCHEMA_VERSION,
        "model_id": str(model_id),
        "prompts": sorted(
            prompts,
            key=lambda record: record["prompt_id"],
        ),
    }
    bank["prompt_bank_sha256"] = canonical_json_sha256(bank)
    return bank


def validate_prompt_bank(prompt_bank: dict) -> None:
    if prompt_bank.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported prompt bank schema")
    prompts = prompt_bank.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("prompt bank requires prompts")
    prompt_ids = []
    for record in prompts:
        prompt_id = record.get("prompt_id")
        if not isinstance(prompt_id, str) or not prompt_id:
            raise ValueError("invalid prompt id")
        prompt_ids.append(prompt_id)
        token_ids = record.get("prompt_token_ids")
        if not isinstance(token_ids, list) or not token_ids:
            raise ValueError("invalid prompt token ids")
        if record.get("prompt_token_count") != len(token_ids):
            raise ValueError("prompt token count mismatch")
        expected_hash = canonical_json_sha256({
            "prompt": record.get("prompt"),
            "prompt_token_ids": token_ids,
        })
        if record.get("prompt_sha256") != expected_hash:
            raise ValueError(f"prompt hash mismatch: {prompt_id}")
    if prompt_ids != sorted(prompt_ids):
        raise ValueError("prompt records must be sorted")
    if len(prompt_ids) != len(set(prompt_ids)):
        raise ValueError("duplicate prompt id")
    without_hash = {
        key: value
        for key, value in prompt_bank.items()
        if key != "prompt_bank_sha256"
    }
    if (
        prompt_bank.get("prompt_bank_sha256")
        != canonical_json_sha256(without_hash)
    ):
        raise ValueError("prompt bank hash mismatch")


def resolve_policy_config(policy_name: str, defaults: dict) -> dict:
    if policy_name not in POLICY_OVERRIDES:
        raise ValueError(f"unknown policy: {policy_name}")
    missing = [field for field in POLICY_FIELDS if field not in defaults]
    if missing:
        raise ValueError(
            "missing policy defaults: " + ", ".join(missing)
        )
    return {
        **COMMON_ENGINE_CONFIG,
        **{field: defaults[field] for field in POLICY_FIELDS},
        **POLICY_OVERRIDES[policy_name],
    }


def policy_identity(resolved_config: dict) -> str:
    return canonical_json_sha256(resolved_config)


def deduplicate_policies(resolved: dict[str, dict]) -> dict:
    expected_names = ("P0", "P1", "P2", "P3")
    if tuple(resolved) != expected_names:
        raise ValueError("policies must be ordered P0, P1, P2, P3")
    identity_by_name = {
        name: policy_identity(resolved[name])
        for name in expected_names
    }
    names_by_identity: dict[str, list[str]] = {}
    for name in expected_names:
        names_by_identity.setdefault(
            identity_by_name[name],
            [],
        ).append(name)
    for names in names_by_identity.values():
        if len(names) > 1 and names != ["P0", "P1"]:
            raise ValueError(
                "unexpected policy identity collision: "
                + ", ".join(names)
            )
    canonical_policy_by_name = {}
    for name in expected_names:
        aliases = names_by_identity[identity_by_name[name]]
        canonical_policy_by_name[name] = aliases[0]
    return {
        "identity_by_name": identity_by_name,
        "canonical_policy_by_name": canonical_policy_by_name,
        "aliases_by_canonical_policy": {
            names[0]: names
            for names in names_by_identity.values()
        },
    }


def build_calibration_manifest(prompt_bank: dict) -> list[dict]:
    validate_prompt_bank(prompt_bank)
    prompt_hash = prompt_bank["prompt_bank_sha256"]
    rows = []
    for rate_index in range(CALIBRATION_MAX_DOUBLINGS + 1):
        requested_rate_rps = (
            CALIBRATION_INITIAL_RATE_RPS * (2 ** rate_index)
        )
        rows.append({
            "schema_version": SCHEMA_VERSION,
            "generator_version": GENERATOR_VERSION,
            "calibration_id": f"p0-rate-{rate_index:02d}",
            "policy": "P0",
            "requested_rate_rps": requested_rate_rps,
            "request_count": CALIBRATION_REQUESTS_PER_RATE,
            "seed": 4000 + rate_index,
            "prompt_bank_sha256": prompt_hash,
            "drain_timeout_ns": CALIBRATION_DRAIN_TIMEOUT_NS,
        })
    return rows


def reconstruct_calibration_backlog_samples(
    timeline_rows: list[dict],
    *,
    sample_count: int = 33,
) -> list[dict]:
    if (
        not isinstance(timeline_rows, list)
        or not timeline_rows
        or isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count < 2
    ):
        raise ValueError("invalid calibration backlog inputs")
    intervals = []
    for row in timeline_rows:
        if not isinstance(row, dict):
            raise ValueError("timeline row must be an object")
        scheduled_arrival_ns = int(_finite_number(
            row.get("scheduled_arrival_ns"),
            "scheduled_arrival_ns",
        ))
        completion_ns = int(_finite_number(
            row.get("completion_ns"),
            "completion_ns",
        ))
        if completion_ns < scheduled_arrival_ns:
            raise ValueError("completion precedes scheduled arrival")
        intervals.append((scheduled_arrival_ns, completion_ns))
    window_start_ns = min(start for start, _ in intervals)
    window_end_ns = max(start for start, _ in intervals)
    if window_end_ns <= window_start_ns:
        raise ValueError("offered arrival window must have duration")
    window_duration_ns = window_end_ns - window_start_ns
    samples = []
    for index in range(sample_count):
        sample_ns = (
            window_start_ns
            + (window_duration_ns * index) // (sample_count - 1)
        )
        samples.append({
            "relative_time_s": (
                sample_ns - window_start_ns
            ) / 1_000_000_000.0,
            "unfinished_count": sum(
                start <= sample_ns < completion
                for start, completion in intervals
            ),
        })
    return samples


def _ols_slope(
    samples: list[dict],
    offered_window_duration_s: float,
) -> float:
    if not isinstance(samples, list) or len(samples) < 2:
        raise ValueError("backlog_samples require at least two rows")
    window_duration_s = _finite_number(
        offered_window_duration_s,
        "offered_window_duration_s",
    )
    if window_duration_s <= 0.0:
        raise ValueError("offered window duration must be positive")
    points = []
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("backlog sample must be an object")
        relative_time_s = _finite_number(
            sample.get("relative_time_s"),
            "relative_time_s",
        )
        unfinished_count = _finite_number(
            sample.get("unfinished_count"),
            "unfinished_count",
        )
        points.append((relative_time_s, unfinished_count))
    points.sort()
    tail = [
        point for point in points
        if (
            (2.0 * window_duration_s) / 3.0
            <= point[0]
            <= window_duration_s
        )
    ]
    if len(tail) < 2:
        raise ValueError(
            "backlog samples do not cover final offered window third"
        )
    mean_x = statistics.fmean(point[0] for point in tail)
    mean_y = statistics.fmean(point[1] for point in tail)
    denominator = sum(
        (point[0] - mean_x) ** 2
        for point in tail
    )
    if denominator <= 0.0:
        raise ValueError("backlog sample times must vary")
    return sum(
        (point[0] - mean_x) * (point[1] - mean_y)
        for point in tail
    ) / denominator


def select_lambda_ref(calibration_rows: list[dict]) -> dict:
    if not isinstance(calibration_rows, list) or not calibration_rows:
        return {
            "status": "INCOMPLETE",
            "error_type": "no_stable_point",
            "error": "calibration rows are empty",
            "lambda_ref": None,
            "evaluated_rows": [],
        }
    evaluated_rows = []
    for row in calibration_rows:
        evaluated = dict(row) if isinstance(row, dict) else {}
        try:
            offered_rate = _finite_number(
                evaluated.get("offered_rate_rps"),
                "offered_rate_rps",
            )
            throughput = _finite_number(
                evaluated.get(
                    "completed_request_throughput_rps"
                ),
                "completed_request_throughput_rps",
            )
            if offered_rate <= 0.0 or throughput < 0.0:
                raise ValueError(
                    "calibration rates must be non-negative"
                )
            slope = _ols_slope(
                evaluated.get("backlog_samples"),
                evaluated.get("offered_window_duration_s"),
            )
            slope_threshold = max(
                0.01,
                0.02 * offered_rate,
            )
            structural_ok = (
                evaluated.get("complete_requests") is True
                and evaluated.get("exact_outputs") is True
                and evaluated.get("finite_metrics") is True
            )
            slope_within_threshold = (
                slope <= slope_threshold
                or math.isclose(
                    slope,
                    slope_threshold,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            )
            stable = structural_ok and slope_within_threshold
            evaluated.update({
                "offered_rate_rps": offered_rate,
                "completed_request_throughput_rps": throughput,
                "backlog_slope_rps": slope,
                "backlog_slope_threshold_rps": slope_threshold,
                "stable": stable,
                "stability_error": None,
            })
        except (TypeError, ValueError) as exc:
            evaluated.update({
                "stable": False,
                "stability_error": str(exc),
            })
        evaluated_rows.append(evaluated)
    evaluated_rows.sort(key=lambda row: (
        float(row.get("offered_rate_rps", math.inf))
        if isinstance(
            row.get("offered_rate_rps"),
            (int, float),
        )
        and not isinstance(row.get("offered_rate_rps"), bool)
        and math.isfinite(float(row["offered_rate_rps"]))
        else math.inf
    ))
    stable_rows = [
        row for row in evaluated_rows
        if row.get("stable") is True
    ]
    if not stable_rows:
        return {
            "status": "INCOMPLETE",
            "error_type": "no_stable_point",
            "error": "no structurally valid stable calibration rate",
            "lambda_ref": None,
            "evaluated_rows": evaluated_rows,
        }
    highest_stable_rate = max(
        row["offered_rate_rps"] for row in stable_rows
    )
    higher_unstable = [
        row for row in evaluated_rows
        if (
            isinstance(row.get("offered_rate_rps"), (int, float))
            and not isinstance(row.get("offered_rate_rps"), bool)
            and math.isfinite(float(row["offered_rate_rps"]))
            and row["offered_rate_rps"] > highest_stable_rate
            and row.get("stable") is not True
        )
    ]
    if not higher_unstable:
        return {
            "status": "INCOMPLETE",
            "error_type": "no_clear_ceiling",
            "error": "no higher unstable calibration rate",
            "lambda_ref": None,
            "evaluated_rows": evaluated_rows,
        }
    maximum_stable_throughput = max(
        row["completed_request_throughput_rps"]
        for row in stable_rows
    )
    eligible = [
        row for row in stable_rows
        if row["completed_request_throughput_rps"] >= (
            0.95 * maximum_stable_throughput
        )
    ]
    selected = max(
        eligible,
        key=lambda row: row["offered_rate_rps"],
    )
    ceiling = min(
        higher_unstable,
        key=lambda row: row["offered_rate_rps"],
    )
    return {
        "status": "PASS",
        "error_type": None,
        "error": None,
        "lambda_ref": selected["offered_rate_rps"],
        "maximum_stable_throughput_rps": (
            maximum_stable_throughput
        ),
        "ceiling_rate_rps": ceiling["offered_rate_rps"],
        "evaluated_rows": evaluated_rows,
    }


def build_case_matrix(run_manifest: dict) -> list[dict]:
    if not isinstance(run_manifest, dict):
        raise ValueError("run manifest must be an object")
    scenarios = run_manifest.get("required_scenarios")
    if scenarios != list(CANONICAL_SCENARIOS):
        raise ValueError(
            "required scenarios must match canonical order"
        )
    repetitions = run_manifest.get("measured_repetitions")
    if repetitions != MEASURED_REPETITIONS:
        raise ValueError(
            "measured repetitions must equal "
            f"{MEASURED_REPETITIONS}"
        )
    canonical_by_name = run_manifest.get(
        "canonical_policy_by_name"
    )
    if canonical_by_name != {
        "P0": "P0",
        "P1": "P0",
        "P2": "P2",
        "P3": "P3",
    }:
        raise ValueError("invalid canonical policy alias map")
    identities = run_manifest.get("policy_identity_by_name")
    if (
        not isinstance(identities, dict)
        or identities.get("P0") != identities.get("P1")
        or len({
            identities.get("P0"),
            identities.get("P2"),
            identities.get("P3"),
        }) != 3
    ):
        raise ValueError("invalid canonical policy identities")
    resolved = run_manifest.get(
        "resolved_policy_config_by_name"
    )
    if not isinstance(resolved, dict):
        raise ValueError("missing resolved policy configs")
    matrix = []
    for repetition in range(repetitions):
        policy_order = POLICY_ORDER_BY_REPETITION[repetition]
        for scenario in scenarios:
            for policy in policy_order:
                case_id = (
                    f"{scenario}__{policy}__r{repetition}"
                )
                matrix.append({
                    "case_id": case_id,
                    "run_tag": run_manifest.get("run_tag"),
                    "scenario": scenario,
                    "policy": policy,
                    "repetition": repetition,
                    "policy_order": list(policy_order).index(policy),
                    "resolved_config": dict(resolved[policy]),
                    "policy_identity": identities[policy],
                    "workload_sha256": run_manifest.get(
                        "workload_sha256"
                    ),
                    "source_tree_sha256": run_manifest.get(
                        "source_tree_sha256"
                    ),
                    "environment_sha256": run_manifest.get(
                        "environment_sha256"
                    ),
                    "drain_timeout_ns": run_manifest.get(
                        "drain_timeout_ns",
                        CANONICAL_DRAIN_TIMEOUT_NS,
                    ),
                })
    keys = {
        (
            row["policy"],
            row["scenario"],
            row["repetition"],
        )
        for row in matrix
    }
    if len(matrix) != 54 or len(keys) != len(matrix):
        raise ValueError("invalid canonical case matrix")
    return matrix


def allocate_port_pair() -> tuple[int, int]:
    sockets = []
    ports = []
    try:
        while len(ports) < 2:
            handle = socket.socket(
                socket.AF_INET,
                socket.SOCK_STREAM,
            )
            handle.bind(("127.0.0.1", 0))
            port = int(handle.getsockname()[1])
            sockets.append(handle)
            if port not in ports:
                ports.append(port)
    finally:
        for handle in sockets:
            handle.close()
    return ports[0], ports[1]


def _write_json(path: Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        "".join(
            json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    payload = Path(path).read_bytes()
    if payload and not payload.endswith(b"\n"):
        raise ValueError(f"JSONL missing final newline: {path}")
    return [
        json.loads(line)
        for line in payload.splitlines()
    ]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_local_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_deterministic_source_tar(
    source_dir: Path,
    archive_path: Path,
) -> None:
    temporary = archive_path.with_name(archive_path.name + ".tmp")
    with temporary.open("wb") as raw_handle:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw_handle,
            mtime=0,
        ) as gzip_handle:
            with tarfile.open(
                fileobj=gzip_handle,
                mode="w",
                format=tarfile.PAX_FORMAT,
            ) as archive:
                for path in sorted(source_dir.rglob("*")):
                    if not path.is_file():
                        continue
                    relative = path.relative_to(source_dir).as_posix()
                    info = archive.gettarinfo(
                        str(path),
                        arcname=f"source/{relative}",
                    )
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mtime = 0
                    info.mode = 0o644
                    with path.open("rb") as handle:
                        archive.addfile(info, handle)
    temporary.replace(archive_path)


def snapshot_source(
    repo_root: Path,
    out_dir: Path,
) -> dict:
    repo_root = Path(repo_root).resolve()
    out_dir = Path(out_dir).resolve()
    source_audit = _load_local_module(
        "arrival_load_source_audit",
        repo_root / "tools" / "source_audit.py",
    )
    evidence = source_audit.build_source_evidence(
        repo_root,
        out_dir,
        owned_roots=OWNED_SOURCE_ROOTS,
        ignored_untracked_prefixes=IGNORED_UNTRACKED_PREFIXES,
    )
    _write_deterministic_source_tar(
        out_dir / "source",
        out_dir / "source_snapshot.tar.gz",
    )
    return evidence


def _run_identity(run_manifest: dict) -> dict:
    fields = (
        "run_tag",
        "source_tree_sha256",
        "workload_sha256",
        "environment_sha256",
        "policy_identity_by_name",
        "canonical_policy_by_name",
        "resolved_policy_config_by_name",
    )
    return {
        field: run_manifest.get(field)
        for field in fields
    }


def _validate_stored_run_identity(
    run_dir: Path,
    run_manifest: dict,
) -> None:
    manifest_path = Path(run_dir) / "run_manifest.json"
    if not manifest_path.is_file():
        return
    stored = _read_json(manifest_path)
    if _run_identity(stored) != _run_identity(run_manifest):
        raise ValueError("resume identity mismatch")


def _validate_smoke_marker(run_manifest: dict) -> None:
    smoke = run_manifest.get("smoke_verification")
    if not isinstance(smoke, dict) or smoke.get("status") != "PASS":
        raise ValueError(
            "canonical requires a verified smoke marker"
        )
    for field in (
        "source_tree_sha256",
        "environment_sha256",
    ):
        if smoke.get(field) != run_manifest.get(field):
            raise ValueError(
                f"smoke {field} identity mismatch"
            )


def validate_run_evidence(
    run_manifest: dict,
    *,
    source_evidence_path: Path,
    environment_evidence_path: Path,
) -> None:
    source_evidence = _read_json(source_evidence_path)
    environment_evidence = _read_json(environment_evidence_path)
    if (
        run_manifest.get("source_tree_sha256")
        != source_evidence.get("tree_sha256")
    ):
        raise ValueError("source identity mismatch")
    if (
        run_manifest.get("environment_sha256")
        != environment_identity_sha256(environment_evidence)
    ):
        raise ValueError("environment identity mismatch")


def _case_is_complete(
    case_dir: Path,
    case_spec: dict,
    run_manifest: dict,
) -> bool:
    required = (
        "case_result.json",
        "process.json",
        "exitcode",
    )
    if not all((case_dir / name).is_file() for name in required):
        return False
    try:
        result = _read_json(case_dir / "case_result.json")
        process = _read_json(case_dir / "process.json")
        exitcode = int(
            (case_dir / "exitcode").read_text().strip()
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return (
        result.get("status") == "PASS"
        and exitcode == 0
        and process.get("case_id") == case_spec["case_id"]
        and process.get("identity") == _run_identity(run_manifest)
        and process.get("policy_identity")
        == case_spec["policy_identity"]
    )


def _is_port_collision(returncode: int, output: str) -> bool:
    return returncode != 0 and (
        "EADDRINUSE" in output
        or "address already in use" in output.lower()
    )


def _next_unused_port_pair(
    used_pairs: set[tuple[int, int]],
) -> tuple[int, int]:
    for _ in range(100):
        pair = allocate_port_pair()
        if pair[0] != pair[1] and pair not in used_pairs:
            used_pairs.add(pair)
            return pair
    raise RuntimeError("could not allocate a fresh port pair")


def _case_workload(
    workload_rows: list[dict],
    scenario: str,
) -> list[dict]:
    selected = [
        row for row in workload_rows
        if row.get("scenario") == scenario
    ]
    if not selected:
        raise ValueError(f"missing workload scenario: {scenario}")
    return selected


def _launch_case(
    *,
    run_dir: Path,
    python_bin: str,
    model_path: str,
    run_manifest: dict,
    case_spec: dict,
    workload_rows: list[dict],
    used_pairs: set[tuple[int, int]],
) -> dict:
    case_dir = Path(run_dir) / "processes" / case_spec["case_id"]
    case_dir.mkdir(parents=True, exist_ok=True)
    case_spec_path = case_dir / "case_spec.json"
    workload_path = case_dir / "workload_manifest.jsonl"
    _write_json(case_spec_path, case_spec)
    _write_jsonl(workload_path, workload_rows)
    command = [
        str(python_bin),
        str(_REPO_ROOT / "tools" / "arrival_load_driver.py"),
        "--case-spec",
        str(case_spec_path),
        "--workload-manifest",
        str(workload_path),
        "--model",
        str(model_path),
        "--output-dir",
        str(case_dir),
    ]
    attempts = []
    completed = None
    started_ns = time.time_ns()
    for attempt in range(2):
        dist_port, master_port = _next_unused_port_pair(
            used_pairs
        )
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["PYTHONPATH"] = str(_REPO_ROOT)
        environment["TINYVLLM_DIST_PORT"] = str(dist_port)
        environment["MASTER_PORT"] = str(master_port)
        completed = subprocess.run(
            command,
            cwd=_REPO_ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        attempts.append({
            "attempt": attempt + 1,
            "tinyvllm_dist_port": dist_port,
            "master_port": master_port,
            "returncode": int(completed.returncode),
        })
        combined = completed.stdout + "\n" + completed.stderr
        if not _is_port_collision(
            completed.returncode,
            combined,
        ):
            break
    finished_ns = time.time_ns()
    (case_dir / "stdout.log").write_text(
        completed.stdout,
        encoding="utf-8",
    )
    (case_dir / "stderr.log").write_text(
        completed.stderr,
        encoding="utf-8",
    )
    final_attempt = attempts[-1]
    process = {
        "case_id": case_spec["case_id"],
        "command": command,
        "pid": getattr(completed, "pid", None),
        "start_time_ns": started_ns,
        "end_time_ns": finished_ns,
        "returncode": int(completed.returncode),
        "tinyvllm_dist_port": final_attempt[
            "tinyvllm_dist_port"
        ],
        "master_port": final_attempt["master_port"],
        "attempts": attempts,
        "identity": _run_identity(run_manifest),
        "policy_identity": case_spec["policy_identity"],
        "source_tree_sha256": run_manifest.get(
            "source_tree_sha256"
        ),
        "workload_sha256": run_manifest.get(
            "workload_sha256"
        ),
        "environment_sha256": run_manifest.get(
            "environment_sha256"
        ),
    }
    _write_json(case_dir / "process.json", process)
    if not (case_dir / "exitcode").is_file():
        (case_dir / "exitcode").write_text(
            f"{completed.returncode}\n",
            encoding="utf-8",
        )
    return {
        "case_id": case_spec["case_id"],
        "status": (
            "PASS"
            if _case_is_complete(
                case_dir,
                case_spec,
                run_manifest,
            )
            else "INCOMPLETE"
        ),
        "process": process,
    }


def run_canonical(
    *,
    run_dir: Path,
    python_bin: str,
    model_path: str,
    run_manifest: dict,
    resume: bool = False,
) -> dict:
    run_dir = Path(run_dir)
    _validate_stored_run_identity(run_dir, run_manifest)
    _validate_smoke_marker(run_manifest)
    workload_path = run_dir / "workload_manifest.jsonl"
    if not workload_path.is_file():
        raise ValueError("missing frozen workload manifest")
    workload_rows = _read_jsonl(workload_path)
    matrix = build_case_matrix(run_manifest)
    process_root = run_dir / "processes"
    process_root.mkdir(parents=True, exist_ok=True)
    used_pairs = set()
    for process_path in process_root.glob("*/process.json"):
        try:
            process = _read_json(process_path)
            used_pairs.add((
                int(process["tinyvllm_dist_port"]),
                int(process["master_port"]),
            ))
        except (OSError, KeyError, TypeError, ValueError):
            continue
    rows = []
    for case_spec in matrix:
        case_dir = process_root / case_spec["case_id"]
        if case_dir.exists():
            if (
                resume
                and _case_is_complete(
                    case_dir,
                    case_spec,
                    run_manifest,
                )
            ):
                rows.append({
                    "case_id": case_spec["case_id"],
                    "status": "PASS",
                    "resumed": True,
                })
                continue
            replacement = process_root / (
                f"{case_spec['case_id']}.replaced."
                f"{time.time_ns()}"
            )
            shutil.move(case_dir, replacement)
        rows.append(_launch_case(
            run_dir=run_dir,
            python_bin=python_bin,
            model_path=model_path,
            run_manifest=run_manifest,
            case_spec=case_spec,
            workload_rows=_case_workload(
                workload_rows,
                case_spec["scenario"],
            ),
            used_pairs=used_pairs,
        ))
    status = (
        "PASS"
        if all(row["status"] == "PASS" for row in rows)
        else "INCOMPLETE"
    )
    return {
        "status": status,
        "case_count": len(rows),
        "case_results": rows,
    }


def run_calibration(
    *,
    run_dir: Path,
    python_bin: str,
    model_path: str,
    run_manifest: dict,
    resume: bool = False,
) -> dict:
    run_dir = Path(run_dir)
    _validate_stored_run_identity(run_dir, run_manifest)
    _validate_smoke_marker(run_manifest)
    case_rows_path = run_dir / "case_rows.jsonl"
    if (
        case_rows_path.is_file()
        and case_rows_path.read_bytes()
    ):
        raise ValueError(
            "calibration cannot change after canonical rows exist"
        )
    prompt_bank_path = run_dir / "prompt_bank.json"
    if not prompt_bank_path.is_file():
        raise ValueError("missing prompt bank")
    prompt_bank = _read_json(prompt_bank_path)
    validate_prompt_bank(prompt_bank)
    prompt = _prompt_by_class(prompt_bank)["short"]
    calibration_rows_path = run_dir / "calibration_rows.jsonl"
    existing_rows = (
        _read_jsonl(calibration_rows_path)
        if resume and calibration_rows_path.is_file()
        else []
    )
    rows_by_rate = {
        float(row["offered_rate_rps"]): row
        for row in existing_rows
        if isinstance(row.get("offered_rate_rps"), (int, float))
    }
    used_pairs = set()

    def calibration_workload(rate: float, index: int) -> list[dict]:
        offsets = _exponential_offsets(
            CALIBRATION_REQUESTS_PER_RATE,
            rate,
            4000 + index,
        )
        return [{
            "schema_version": SCHEMA_VERSION,
            "generator_version": GENERATOR_VERSION,
            "scenario": "calibration",
            "request_id": (
                f"calibration-{index:02d}-{request_index:04d}"
            ),
            "warmup": False,
            "arrival_offset_ns": offset,
            "requested_rate_rps": rate,
            "seed": 4000 + index,
            "prompt_id": prompt["prompt_id"],
            "prompt": prompt["prompt"],
            "prompt_sha256": prompt["prompt_sha256"],
            "prompt_token_ids": list(prompt["prompt_token_ids"]),
            "prompt_token_count": prompt["prompt_token_count"],
            "prompt_class": "short",
            "output_class": "short",
            "service_time_bucket": "short__short",
            "requested_output_tokens": OUTPUT_CLASS_TOKENS["short"],
            "sampling": {
                "temperature": 0.0,
                "ignore_eos": True,
                "max_tokens": OUTPUT_CLASS_TOKENS["short"],
            },
            "drain_timeout_ns": CALIBRATION_DRAIN_TIMEOUT_NS,
            "starvation_deadline_ns": STARVATION_DEADLINE_NS,
        } for request_index, offset in enumerate(offsets)]

    def aggregate_calibration_case(
        case_dir: Path,
        rate: float,
    ) -> dict:
        result = _read_json(case_dir / "case_result.json")
        timeline = _read_jsonl(
            case_dir / "request_timeline.jsonl"
        )
        complete = (
            result.get("status") == "PASS"
            and len(timeline) == CALIBRATION_REQUESTS_PER_RATE
            and all(
                row.get("completion_ns") is not None
                and row.get("error") is None
                and len(row.get("output_token_ids", []))
                == OUTPUT_CLASS_TOKENS["short"]
                for row in timeline
            )
        )
        timestamps = [
            int(row["completion_ns"])
            for row in timeline
            if row.get("completion_ns") is not None
        ]
        scheduled = [
            int(row["scheduled_arrival_ns"])
            for row in timeline
            if row.get("scheduled_arrival_ns") is not None
        ]
        duration_s = (
            (max(timestamps) - min(scheduled))
            / 1_000_000_000.0
            if timestamps and scheduled
            else float("nan")
        )
        throughput = (
            len(timeline) / duration_s
            if duration_s > 0.0
            else float("nan")
        )
        backlog_samples = reconstruct_calibration_backlog_samples(
            timeline
        )
        offered_window_duration_s = (
            (
                max(int(row["scheduled_arrival_ns"]) for row in timeline)
                - min(int(row["scheduled_arrival_ns"]) for row in timeline)
            )
            / 1_000_000_000.0
        )
        return {
            "case_id": result.get("case_id"),
            "offered_rate_rps": rate,
            "completed_request_throughput_rps": throughput,
            "complete_requests": complete,
            "exact_outputs": complete,
            "finite_metrics": (
                math.isfinite(throughput)
                and len(backlog_samples) >= 2
            ),
            "backlog_samples": backlog_samples,
            "offered_window_duration_s": offered_window_duration_s,
        }

    def execute_rate(rate: float, index: int) -> dict:
        if rate in rows_by_rate:
            return rows_by_rate[rate]
        case_id = f"calibration-p0-rate-{index:02d}"
        case_spec = {
            "case_id": case_id,
            "scenario": "calibration",
            "policy": "P0",
            "repetition": index,
            "requested_rate_rps": rate,
            "resolved_config": dict(
                run_manifest[
                    "resolved_policy_config_by_name"
                ]["P0"]
            ),
            "policy_identity": run_manifest[
                "policy_identity_by_name"
            ]["P0"],
            "workload_sha256": None,
            "source_tree_sha256": run_manifest.get(
                "source_tree_sha256"
            ),
            "environment_sha256": run_manifest.get(
                "environment_sha256"
            ),
            "drain_timeout_ns": CALIBRATION_DRAIN_TIMEOUT_NS,
        }
        workload = calibration_workload(rate, index)
        launched = _launch_case(
            run_dir=run_dir,
            python_bin=python_bin,
            model_path=model_path,
            run_manifest=run_manifest,
            case_spec=case_spec,
            workload_rows=workload,
            used_pairs=used_pairs,
        )
        row = aggregate_calibration_case(
            run_dir / "processes" / case_id,
            rate,
        )
        row["process_status"] = launched["status"]
        rows_by_rate[rate] = row
        _write_jsonl(
            calibration_rows_path,
            list(rows_by_rate.values()),
        )
        return row

    stable_rate = None
    unstable_rate = None
    rate_index = 0
    rate = CALIBRATION_INITIAL_RATE_RPS
    while rate_index <= CALIBRATION_MAX_DOUBLINGS:
        execute_rate(rate, rate_index)
        evaluated = select_lambda_ref(list(rows_by_rate.values()))
        stable_rates = [
            row["offered_rate_rps"]
            for row in evaluated["evaluated_rows"]
            if row.get("stable") is True
        ]
        unstable_rates = [
            row["offered_rate_rps"]
            for row in evaluated["evaluated_rows"]
            if (
                row.get("stable") is not True
                and isinstance(
                    row.get("offered_rate_rps"),
                    (int, float),
                )
            )
        ]
        stable_rate = max(stable_rates) if stable_rates else None
        higher_unstable = [
            value for value in unstable_rates
            if stable_rate is not None and value > stable_rate
        ]
        if higher_unstable:
            unstable_rate = min(higher_unstable)
            break
        rate_index += 1
        rate *= 2.0

    if stable_rate is not None and unstable_rate is not None:
        for bisection_index in range(CALIBRATION_BISECTION_STEPS):
            midpoint = (stable_rate + unstable_rate) / 2.0
            execute_rate(
                midpoint,
                CALIBRATION_MAX_DOUBLINGS
                + 1
                + bisection_index,
            )
            evaluated = select_lambda_ref(
                list(rows_by_rate.values())
            )
            midpoint_row = next(
                row for row in evaluated["evaluated_rows"]
                if math.isclose(
                    row["offered_rate_rps"],
                    midpoint,
                )
            )
            if midpoint_row.get("stable") is True:
                stable_rate = midpoint
            else:
                unstable_rate = midpoint

    ordered_rows = sorted(
        rows_by_rate.values(),
        key=lambda row: row["offered_rate_rps"],
    )
    _write_jsonl(calibration_rows_path, ordered_rows)
    selection = select_lambda_ref(ordered_rows)
    if selection["status"] != "PASS":
        return selection
    lambda_ref = selection["lambda_ref"]
    frozen_workload = build_canonical_workload(
        lambda_ref=lambda_ref,
        prompt_bank=prompt_bank,
    )
    _write_jsonl(
        run_dir / "workload_manifest.jsonl",
        frozen_workload,
    )
    workload_sha256 = canonical_json_sha256(frozen_workload)
    updated_manifest = dict(run_manifest)
    updated_manifest["workload_sha256"] = workload_sha256
    updated_manifest["calibration"] = {
        "status": "PASS",
        "lambda_ref_rps": lambda_ref,
        "stable_rate_rps": max(
            row["offered_rate_rps"]
            for row in selection["evaluated_rows"]
            if row.get("stable") is True
        ),
        "unstable_rate_rps": selection["ceiling_rate_rps"],
        "maximum_stable_throughput_rps": selection[
            "maximum_stable_throughput_rps"
        ],
    }
    _write_json(run_dir / "run_manifest.json", updated_manifest)
    return {
        **selection,
        "workload_sha256": workload_sha256,
    }


def _prompt_by_class(prompt_bank: dict) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = {}
    for prompt in prompt_bank["prompts"]:
        grouped.setdefault(prompt["prompt_class"], []).append(prompt)
    missing = set(PROMPT_CLASS_TARGET_TOKENS) - set(grouped)
    if missing:
        raise ValueError(
            "prompt bank missing classes: " + ", ".join(sorted(missing))
        )
    return {
        prompt_class: sorted(
            records,
            key=lambda record: record["prompt_id"],
        )[0]
        for prompt_class, records in grouped.items()
    }


def _exponential_offsets(
    count: int,
    requested_rate_rps: float,
    seed: int,
) -> list[int]:
    generator = random.Random(seed)
    elapsed_s = 0.0
    offsets = []
    for _ in range(count):
        elapsed_s += generator.expovariate(requested_rate_rps)
        offsets.append(round(elapsed_s * 1_000_000_000))
    return offsets


def _burst_offsets(count: int, seed: int) -> list[int]:
    if count != CANONICAL_MEASURED_REQUESTS:
        raise ValueError("burst workload requires 64 measured requests")
    generator = random.Random(seed)
    offsets = []
    for burst_index in range(4):
        burst_start_ns = burst_index * 2_250_000_000
        for _ in range(16):
            offsets.append(
                burst_start_ns
                + generator.randrange(0, 250_000_001)
            )
    return sorted(offsets)


def _balanced_classes(index: int) -> tuple[str, str]:
    buckets = (
        ("short", "short"),
        ("short", "long"),
        ("medium", "short"),
        ("medium", "long"),
        ("long", "short"),
        ("long", "long"),
    )
    return buckets[index % len(buckets)]


def _long_pressure_classes(index: int) -> tuple[str, str]:
    prompt_cycle = (
        "long",
        "long",
        "long",
        "long",
        "long",
        "long",
        "medium",
        "medium",
        "short",
        "short",
    )
    return prompt_cycle[index % len(prompt_cycle)], (
        "short" if index % 2 == 0 else "long"
    )


def _fairness_classes(index: int) -> tuple[str, str]:
    buckets = (
        ("short", "short"),
        ("short", "long"),
        ("medium", "short"),
        ("medium", "long"),
        ("long", "short"),
        ("long", "long"),
    )
    return buckets[index // FAIRNESS_REQUESTS_PER_BUCKET]


def _request_row(
    *,
    scenario: str,
    index: int,
    warmup: bool,
    arrival_offset_ns: int,
    requested_rate_rps: float,
    prompt_class: str,
    output_class: str,
    prompt: dict,
) -> dict:
    requested_output_tokens = OUTPUT_CLASS_TOKENS[output_class]
    phase = "warmup" if warmup else "measured"
    return {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "scenario": scenario,
        "request_id": f"{scenario}-{phase}-{index:04d}",
        "warmup": warmup,
        "arrival_offset_ns": int(arrival_offset_ns),
        "requested_rate_rps": float(requested_rate_rps),
        "seed": ARRIVAL_SEEDS[scenario],
        "prompt_id": prompt["prompt_id"],
        "prompt": prompt["prompt"],
        "prompt_sha256": prompt["prompt_sha256"],
        "prompt_token_ids": list(prompt["prompt_token_ids"]),
        "prompt_token_count": int(prompt["prompt_token_count"]),
        "prompt_class": prompt_class,
        "output_class": output_class,
        "service_time_bucket": (
            f"{prompt_class}__{output_class}"
        ),
        "requested_output_tokens": requested_output_tokens,
        "sampling": {
            "temperature": 0.0,
            "ignore_eos": True,
            "max_tokens": requested_output_tokens,
        },
        "drain_timeout_ns": CANONICAL_DRAIN_TIMEOUT_NS,
        "starvation_deadline_ns": STARVATION_DEADLINE_NS,
    }


def build_canonical_workload(
    *,
    lambda_ref: float,
    prompt_bank: dict,
) -> list[dict]:
    if not math.isfinite(lambda_ref) or lambda_ref <= 0.0:
        raise ValueError("lambda_ref must be finite and positive")
    validate_prompt_bank(prompt_bank)
    prompts = _prompt_by_class(prompt_bank)
    rows = []
    for scenario in CANONICAL_SCENARIOS:
        requested_rate_rps = (
            lambda_ref * SCENARIO_RATE_MULTIPLIERS[scenario]
        )
        measured_count = (
            FAIRNESS_REQUESTS_PER_BUCKET * 6
            if scenario == "mixed_service_fairness"
            else CANONICAL_MEASURED_REQUESTS
        )
        warmup_offsets = _exponential_offsets(
            CANONICAL_WARMUP_REQUESTS,
            requested_rate_rps,
            ARRIVAL_SEEDS[scenario] + 10_000,
        )
        warmup_end_ns = warmup_offsets[-1] if warmup_offsets else 0
        for index, arrival_offset_ns in enumerate(warmup_offsets):
            prompt_class, output_class = _balanced_classes(index)
            rows.append(_request_row(
                scenario=scenario,
                index=index,
                warmup=True,
                arrival_offset_ns=arrival_offset_ns,
                requested_rate_rps=requested_rate_rps,
                prompt_class=prompt_class,
                output_class=output_class,
                prompt=prompts[prompt_class],
            ))

        if scenario == "burst":
            relative_offsets = _burst_offsets(
                measured_count,
                ARRIVAL_SEEDS[scenario],
            )
        else:
            relative_offsets = _exponential_offsets(
                measured_count,
                requested_rate_rps,
                ARRIVAL_SEEDS[scenario],
            )
        measured_start_ns = warmup_end_ns + 1_000_000_000
        for index, relative_offset_ns in enumerate(relative_offsets):
            if scenario == "long_prompt_pressure":
                prompt_class, output_class = (
                    _long_pressure_classes(index)
                )
            elif scenario == "mixed_service_fairness":
                prompt_class, output_class = _fairness_classes(index)
            else:
                prompt_class, output_class = _balanced_classes(index)
            rows.append(_request_row(
                scenario=scenario,
                index=index,
                warmup=False,
                arrival_offset_ns=(
                    measured_start_ns + relative_offset_ns
                ),
                requested_rate_rps=requested_rate_rps,
                prompt_class=prompt_class,
                output_class=output_class,
                prompt=prompts[prompt_class],
            ))

    scenario_order = {
        name: index
        for index, name in enumerate(CANONICAL_SCENARIOS)
    }
    rows.sort(key=lambda row: (
        scenario_order[row["scenario"]],
        row["arrival_offset_ns"],
        row["request_id"],
    ))
    counts = Counter(row["request_id"] for row in rows)
    duplicates = [
        request_id
        for request_id, count in counts.items()
        if count != 1
    ]
    if duplicates:
        raise ValueError(
            "duplicate workload request ids: "
            + ", ".join(sorted(duplicates))
        )
    return rows


def _finite_number(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{label} must be finite")
    return normalized


def reconstruct_request_metrics(
    workload_rows: list[dict],
    timeline_rows: list[dict],
    scheduler_rows: list[dict],
) -> list[dict]:
    del scheduler_rows
    workload_by_id = {}
    for row in workload_rows:
        request_id = row.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("invalid workload request id")
        if request_id in workload_by_id:
            raise ValueError(f"duplicate workload request: {request_id}")
        workload_by_id[request_id] = row

    timeline_by_id = {}
    seq_ids = set()
    for row in timeline_rows:
        request_id = row.get("request_id")
        if request_id in timeline_by_id:
            raise ValueError(f"duplicate timeline request: {request_id}")
        if request_id not in workload_by_id:
            raise ValueError(f"unexpected timeline request: {request_id}")
        seq_id = row.get("seq_id")
        if not isinstance(seq_id, int) or seq_id < 0:
            raise ValueError(f"invalid seq_id for {request_id}")
        if seq_id in seq_ids:
            raise ValueError(f"duplicate sequence binding: {seq_id}")
        seq_ids.add(seq_id)
        timeline_by_id[request_id] = row

    if set(timeline_by_id) != set(workload_by_id):
        missing = sorted(set(workload_by_id) - set(timeline_by_id))
        raise ValueError(
            "missing timeline requests: " + ", ".join(missing)
        )

    metrics = []
    for request_id, workload in workload_by_id.items():
        timeline = timeline_by_id[request_id]
        timestamp_names = (
            "scheduled_arrival_ns",
            "actual_arrival_ns",
            "first_scheduled_ns",
            "first_token_ns",
            "completion_ns",
        )
        timestamps = {
            name: _finite_number(timeline.get(name), name)
            for name in timestamp_names
        }
        if not (
            timestamps["scheduled_arrival_ns"]
            <= timestamps["actual_arrival_ns"]
            <= timestamps["first_scheduled_ns"]
            <= timestamps["first_token_ns"]
            <= timestamps["completion_ns"]
        ):
            raise ValueError(
                f"invalid timestamp ordering for {request_id}"
            )
        token_timestamps = [
            _finite_number(value, "token timestamp")
            for value in timeline.get("token_timestamps_ns", [])
        ]
        output_token_ids = timeline.get("output_token_ids")
        if not isinstance(output_token_ids, list):
            raise ValueError(
                f"invalid output token ids for {request_id}"
            )
        if len(token_timestamps) != len(output_token_ids):
            raise ValueError(
                f"token timestamp count mismatch for {request_id}"
            )
        if (
            len(output_token_ids)
            != workload.get("requested_output_tokens")
        ):
            raise ValueError(
                f"output token count mismatch for {request_id}"
            )
        if not token_timestamps:
            raise ValueError(f"request has no output tokens: {request_id}")
        if token_timestamps[0] != timestamps["first_token_ns"]:
            raise ValueError(
                f"first token timestamp mismatch for {request_id}"
            )
        if token_timestamps[-1] > timestamps["completion_ns"]:
            raise ValueError(
                f"token after completion for {request_id}"
            )
        if any(
            current < previous
            for previous, current in zip(
                token_timestamps,
                token_timestamps[1:],
            )
        ):
            raise ValueError(
                f"non-monotonic token timestamps for {request_id}"
            )
        if timeline.get("error") is not None:
            raise ValueError(f"request error for {request_id}")
        if timeline.get("finish_reason") != "length":
            raise ValueError(
                f"unexpected finish reason for {request_id}"
            )
        itl_ns = [
            current - previous
            for previous, current in zip(
                token_timestamps,
                token_timestamps[1:],
            )
        ]
        metrics.append({
            **workload,
            "seq_id": timeline["seq_id"],
            "output_token_ids": list(output_token_ids),
            "finish_reason": timeline["finish_reason"],
            "scheduled_arrival_ns": timestamps[
                "scheduled_arrival_ns"
            ],
            "actual_arrival_ns": timestamps["actual_arrival_ns"],
            "first_scheduled_ns": timestamps["first_scheduled_ns"],
            "first_token_ns": timestamps["first_token_ns"],
            "completion_ns": timestamps["completion_ns"],
            "injection_lag_ns": (
                timestamps["actual_arrival_ns"]
                - timestamps["scheduled_arrival_ns"]
            ),
            "queue_delay_ns": (
                timestamps["first_scheduled_ns"]
                - timestamps["actual_arrival_ns"]
            ),
            "ttft_ns": (
                timestamps["first_token_ns"]
                - timestamps["scheduled_arrival_ns"]
            ),
            "e2e_ns": (
                timestamps["completion_ns"]
                - timestamps["scheduled_arrival_ns"]
            ),
            "itl_ns": itl_ns,
            "maximum_decode_gap_ns": (
                max(itl_ns) if itl_ns else None
            ),
        })
    return metrics


def _percentile_metrics(
    rows: list[dict],
    field: str,
) -> dict[str, float]:
    samples = [
        float(row[field])
        for row in rows
        if row.get(field) is not None
    ]
    if not samples:
        return {}
    return {
        f"p{percentile}_{field}": nearest_rank(
            samples,
            percentile / 100.0,
        )
        for percentile in (50, 95, 99)
    }


def _jain_index(values: list[float]) -> float:
    if not values or any(value < 0.0 for value in values):
        raise ValueError("invalid Jain index samples")
    denominator = len(values) * sum(value * value for value in values)
    if denominator == 0.0:
        return 0.0
    return (sum(values) ** 2) / denominator


def summarize_repetition(
    case: dict,
    request_metrics: list[dict],
    memory_rows: list[dict],
) -> dict:
    measured = [
        row for row in request_metrics
        if not row.get("warmup", False)
    ]
    if not measured:
        raise ValueError("repetition has no measured requests")
    start_ns = _finite_number(
        case.get("measurement_start_ns"),
        "measurement_start_ns",
    )
    end_ns = _finite_number(
        case.get("measurement_end_ns"),
        "measurement_end_ns",
    )
    if end_ns <= start_ns:
        raise ValueError("invalid measurement interval")
    duration_s = (end_ns - start_ns) / 1_000_000_000.0

    metrics = {
        "request_throughput_rps": len(measured) / duration_s,
        "output_token_throughput_tps": sum(
            len(row["output_token_ids"]) for row in measured
        ) / duration_s,
        "maximum_injection_lag_ns": max(
            row["injection_lag_ns"] for row in measured
        ),
        "maximum_decode_gap_ns": max(
            (
                row["maximum_decode_gap_ns"]
                for row in measured
                if row["maximum_decode_gap_ns"] is not None
            ),
            default=None,
        ),
    }
    for field in (
        "injection_lag_ns",
        "queue_delay_ns",
        "ttft_ns",
        "e2e_ns",
    ):
        metrics.update(_percentile_metrics(measured, field))
    itl_samples = [
        {"itl_ns": value}
        for row in measured
        for value in row["itl_ns"]
    ]
    metrics.update(_percentile_metrics(itl_samples, "itl_ns"))

    service_buckets = {}
    service_rates = []
    for bucket in case.get("required_service_buckets", []):
        bucket_rows = [
            row for row in measured
            if row["service_time_bucket"] == bucket
        ]
        if not bucket_rows:
            raise ValueError(f"missing service bucket: {bucket}")
        bucket_metrics = {
            "completed_requests": len(bucket_rows),
            "request_throughput_rps": len(bucket_rows) / duration_s,
            "worst_e2e_ns": max(row["e2e_ns"] for row in bucket_rows),
        }
        bucket_metrics.update(
            _percentile_metrics(bucket_rows, "e2e_ns")
        )
        service_buckets[bucket] = bucket_metrics
        service_rates.append(bucket_metrics["request_throughput_rps"])
    metrics["service_buckets"] = service_buckets
    metrics["jain_service_rate_index"] = _jain_index(service_rates)

    if not memory_rows:
        raise ValueError("repetition has no memory rows")
    for row in memory_rows:
        for field in (
            "cuda_allocated_bytes",
            "cuda_reserved_bytes",
            "used_kv_blocks",
            "kv_block_bytes",
        ):
            _finite_number(row.get(field), field)
    metrics["peak_cuda_allocated_bytes"] = int(max(
        row["cuda_allocated_bytes"] for row in memory_rows
    ))
    metrics["peak_cuda_reserved_bytes"] = int(max(
        row["cuda_reserved_bytes"] for row in memory_rows
    ))
    metrics["peak_used_kv_blocks"] = int(max(
        row["used_kv_blocks"] for row in memory_rows
    ))
    metrics["peak_kv_bytes"] = int(max(
        row["used_kv_blocks"] * row["kv_block_bytes"]
        for row in memory_rows
    ))

    return {
        "policy": case["policy"],
        "scenario": case["scenario"],
        "repetition": case["repetition"],
        "status": "PASS",
        "correctness": {
            "exact_outputs": True,
            "complete_requests": True,
            "no_starvation": True,
            "valid_lifecycle": True,
            "stable_p0_outputs": True,
        },
        "metrics": metrics,
    }


def aggregate_case_repetitions(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("cannot aggregate empty repetitions")
    if len({
        (row.get("policy"), row.get("scenario"))
        for row in rows
    }) != 1:
        raise ValueError("case aggregation requires one policy/scenario")
    repetition_ids = [row.get("repetition") for row in rows]
    if (
        any(not isinstance(value, int) for value in repetition_ids)
        or len(repetition_ids) != len(set(repetition_ids))
    ):
        raise ValueError("case repetitions must be unique integers")
    metric_names = (
        "request_throughput_rps",
        "output_token_throughput_tps",
        "p95_ttft_ns",
        "p95_itl_ns",
        "p99_ttft_ns",
        "p99_itl_ns",
        "p99_e2e_ns",
        "maximum_decode_gap_ns",
        "peak_cuda_reserved_bytes",
        "peak_kv_bytes",
    )
    medians = {}
    for metric_name in metric_names:
        values = [
            _finite_number(
                row["metrics"].get(metric_name),
                metric_name,
            )
            for row in rows
            if row["metrics"].get(metric_name) is not None
        ]
        if values:
            medians[metric_name] = statistics.median(values)
    worst_repetition = min(
        rows,
        key=lambda row: (
            _finite_number(
                row["metrics"].get("request_throughput_rps"),
                "request_throughput_rps",
            ),
            -_finite_number(
                row["metrics"].get("p95_ttft_ns"),
                "p95_ttft_ns",
            ),
            -_finite_number(
                row["metrics"].get("p95_itl_ns"),
                "p95_itl_ns",
            ),
        ),
    )
    return {
        "policy": rows[0]["policy"],
        "scenario": rows[0]["scenario"],
        "repetitions": len(rows),
        "median_metrics": medians,
        "worst_repetition": worst_repetition,
    }


def _ratio(candidate: dict, baseline: dict, metric: str) -> float:
    candidate_value = _finite_number(
        candidate["metrics"].get(metric),
        f"candidate {metric}",
    )
    baseline_value = _finite_number(
        baseline["metrics"].get(metric),
        f"baseline {metric}",
    )
    if baseline_value <= 0.0:
        raise ValueError(f"baseline {metric} must be positive")
    return candidate_value / baseline_value


def _candidate_classification(
    policy: str,
    paired_rows: list[tuple[dict, dict]],
) -> dict:
    ratios = {
        metric: [
            _ratio(candidate, baseline, metric)
            for baseline, candidate in paired_rows
        ]
        for metric in (
            "request_throughput_rps",
            "p95_ttft_ns",
            "p95_itl_ns",
            "p99_ttft_ns",
            "p99_itl_ns",
            "p99_e2e_ns",
            "maximum_decode_gap_ns",
            "peak_cuda_reserved_bytes",
            "peak_kv_bytes",
        )
    }
    bucket_ratios = []
    for baseline, candidate in paired_rows:
        baseline_buckets = baseline["metrics"]["service_buckets"]
        candidate_buckets = candidate["metrics"]["service_buckets"]
        if set(baseline_buckets) != set(candidate_buckets):
            raise ValueError("service bucket set mismatch")
        for bucket in sorted(baseline_buckets):
            baseline_bucket = _finite_number(
                baseline_buckets[bucket].get("p95_e2e_ns"),
                "baseline bucket p95",
            )
            candidate_bucket = _finite_number(
                candidate_buckets[bucket].get("p95_e2e_ns"),
                "candidate bucket p95",
            )
            if baseline_bucket <= 0.0:
                raise ValueError("baseline bucket p95 must be positive")
            bucket_ratios.append(candidate_bucket / baseline_bucket)

    median = {
        metric: statistics.median(values)
        for metric, values in ratios.items()
    }
    worst = {
        "request_throughput_rps": min(
            ratios["request_throughput_rps"]
        ),
        **{
            metric: max(values)
            for metric, values in ratios.items()
            if metric != "request_throughput_rps"
        },
    }
    guard_failures = []
    for metric in (
        "p99_ttft_ns",
        "p99_itl_ns",
        "p99_e2e_ns",
        "maximum_decode_gap_ns",
    ):
        if max(ratios[metric]) > 1.10:
            guard_failures.append(f"{metric} regression exceeds 10%")
    if bucket_ratios and max(bucket_ratios) > 1.10:
        guard_failures.append(
            "service bucket p95 E2E regression exceeds 10%"
        )

    median_paths = {
        "throughput": (
            median["request_throughput_rps"] >= 1.05
            and median["p95_ttft_ns"] <= 1.05
            and median["p95_itl_ns"] <= 1.05
        ),
        "latency": (
            (
                median["p95_ttft_ns"] <= 0.90
                and median["p95_itl_ns"] <= 1.05
            )
            or (
                median["p95_itl_ns"] <= 0.90
                and median["p95_ttft_ns"] <= 1.05
            )
        ) and median["request_throughput_rps"] >= 0.98,
        "memory": (
            min(
                median["peak_cuda_reserved_bytes"],
                median["peak_kv_bytes"],
            ) <= 0.95
            and median["request_throughput_rps"] >= 0.98
            and median["p95_ttft_ns"] <= 1.02
            and median["p95_itl_ns"] <= 1.02
        ),
    }
    worst_paths = {
        "throughput": (
            worst["request_throughput_rps"] >= 1.05
            and worst["p95_ttft_ns"] <= 1.05
            and worst["p95_itl_ns"] <= 1.05
        ),
        "latency": (
            (
                worst["p95_ttft_ns"] <= 0.90
                and worst["p95_itl_ns"] <= 1.05
            )
            or (
                worst["p95_itl_ns"] <= 0.90
                and worst["p95_ttft_ns"] <= 1.05
            )
        ) and worst["request_throughput_rps"] >= 0.98,
        "memory": (
            min(
                worst["peak_cuda_reserved_bytes"],
                worst["peak_kv_bytes"],
            ) <= 0.95
            and worst["request_throughput_rps"] >= 0.98
            and worst["p95_ttft_ns"] <= 1.02
            and worst["p95_itl_ns"] <= 1.02
        ),
    }
    benefit_path = next(
        (
            path
            for path in ("throughput", "latency", "memory")
            if median_paths[path] and worst_paths[path]
        ),
        None,
    )
    favorable_direction = (
        median["request_throughput_rps"] > 1.0
        or median["p95_ttft_ns"] < 1.0
        or median["p95_itl_ns"] < 1.0
        or median["peak_cuda_reserved_bytes"] < 1.0
        or median["peak_kv_bytes"] < 1.0
    )
    if guard_failures:
        classification = "NO_GO"
    elif benefit_path is not None:
        classification = "GO"
    elif favorable_direction:
        classification = "PROMISING_NOT_PROVEN"
    else:
        classification = "NO_GO"
    return {
        "policy": policy,
        "classification": classification,
        "benefit_path": benefit_path,
        "median_ratios": median,
        "worst_repetition_ratios": worst,
        "guard_failures": guard_failures,
    }


def classify_gate(
    run_manifest: dict,
    case_rows: list[dict],
) -> dict:
    structural_failures = []
    correctness_failures = []
    required_scenarios = run_manifest.get("required_scenarios")
    repetitions = run_manifest.get("measured_repetitions")
    canonical_by_name = run_manifest.get(
        "canonical_policy_by_name",
        {},
    )
    identities = run_manifest.get("policy_identity_by_name", {})
    if (
        not isinstance(required_scenarios, list)
        or not required_scenarios
        or not isinstance(repetitions, int)
        or repetitions < 3
    ):
        structural_failures.append("invalid required case matrix")
    if set(canonical_by_name) != {"P0", "P1", "P2", "P3"}:
        structural_failures.append("invalid policy alias map")
    if set(identities) != {"P0", "P1", "P2", "P3"}:
        structural_failures.append("invalid policy identity map")
    if (
        canonical_by_name.get("P1") == "P0"
        and identities.get("P1") != identities.get("P0")
    ):
        structural_failures.append("P1 alias identity mismatch")
    if identities.get("P2") in {
        identities.get("P0"),
        identities.get("P3"),
    }:
        structural_failures.append("unexpected P2 identity collision")
    if identities.get("P3") == identities.get("P0"):
        structural_failures.append("unexpected P3 identity collision")

    canonical_policies = []
    for name in ("P0", "P1", "P2", "P3"):
        canonical = canonical_by_name.get(name)
        if canonical == name and name not in canonical_policies:
            canonical_policies.append(name)
    expected_keys = {
        (policy, scenario, repetition)
        for policy in canonical_policies
        for scenario in required_scenarios or []
        for repetition in range(repetitions or 0)
    }
    observed_keys = []
    rows_by_key = {}
    for row in case_rows:
        key = (
            row.get("policy"),
            row.get("scenario"),
            row.get("repetition"),
        )
        observed_keys.append(key)
        rows_by_key[key] = row
        if row.get("status") != "PASS":
            structural_failures.append(
                f"incomplete case row: {key}"
            )
        metrics = row.get("metrics", {})
        for metric_name, metric_value in metrics.items():
            if (
                isinstance(metric_value, (int, float))
                and not isinstance(metric_value, bool)
                and not math.isfinite(float(metric_value))
            ):
                structural_failures.append(
                    f"non-finite metric {metric_name}: {key}"
                )
        correctness = row.get("correctness", {})
        for field in (
            "exact_outputs",
            "complete_requests",
            "no_starvation",
            "valid_lifecycle",
            "stable_p0_outputs",
        ):
            if correctness.get(field) is not True:
                correctness_failures.append(
                    f"{key} failed {field}"
                )
    if len(observed_keys) != len(set(observed_keys)):
        structural_failures.append("duplicate case rows")
    if set(observed_keys) != expected_keys:
        structural_failures.append("missing or unexpected case rows")

    if structural_failures:
        return {
            "classification": "INCOMPLETE",
            "structural_failures": sorted(set(structural_failures)),
            "correctness_failures": sorted(set(correctness_failures)),
            "candidate_results": {},
        }
    if correctness_failures:
        return {
            "classification": "NO_GO",
            "structural_failures": [],
            "correctness_failures": sorted(set(correctness_failures)),
            "candidate_results": {},
        }

    candidate_results = {}
    for policy in canonical_policies:
        if policy == "P0":
            continue
        paired_rows = []
        for scenario in required_scenarios:
            for repetition in range(repetitions):
                paired_rows.append((
                    rows_by_key[("P0", scenario, repetition)],
                    rows_by_key[(policy, scenario, repetition)],
                ))
        try:
            candidate_results[policy] = _candidate_classification(
                policy,
                paired_rows,
            )
        except ValueError as exc:
            structural_failures.append(f"{policy}: {exc}")
    if structural_failures:
        return {
            "classification": "INCOMPLETE",
            "structural_failures": sorted(set(structural_failures)),
            "correctness_failures": [],
            "candidate_results": candidate_results,
        }
    classifications = {
        result["classification"]
        for result in candidate_results.values()
    }
    if "GO" in classifications:
        classification = "GO"
    elif "PROMISING_NOT_PROVEN" in classifications:
        classification = "PROMISING_NOT_PROVEN"
    else:
        classification = "NO_GO"
    return {
        "classification": classification,
        "structural_failures": [],
        "correctness_failures": [],
        "candidate_results": candidate_results,
    }


def _case_metadata(case_spec: dict) -> dict:
    return {
        "case_id": case_spec["case_id"],
        "policy": case_spec["policy"],
        "scenario": case_spec["scenario"],
        "repetition": case_spec["repetition"],
    }


def _merged_case_rows(
    run_dir: Path,
    matrix: list[dict],
    filename: str,
) -> list[dict]:
    merged = []
    for case_spec in matrix:
        case_dir = run_dir / "processes" / case_spec["case_id"]
        rows = _read_jsonl(case_dir / filename)
        if filename == "request_timeline.jsonl":
            rows.sort(key=lambda row: (
                row.get("scheduled_arrival_ns", 0),
                row.get("request_id", ""),
            ))
        else:
            rows.sort(key=lambda row: (
                row.get("step_index", 0),
                row.get("step_end_ns", row.get("timestamp_ns", 0)),
            ))
        metadata = _case_metadata(case_spec)
        merged.extend({**row, **metadata} for row in rows)
    return merged


def _case_summary(
    run_dir: Path,
    case_spec: dict,
    workload_rows: list[dict],
) -> dict:
    case_dir = run_dir / "processes" / case_spec["case_id"]
    result = _read_json(case_dir / "case_result.json")
    process = _read_json(case_dir / "process.json")
    exitcode = int((case_dir / "exitcode").read_text().strip())
    if (
        result.get("status") != "PASS"
        or process.get("returncode") != 0
        or exitcode != 0
    ):
        return {
            **_case_metadata(case_spec),
            "status": "INCOMPLETE",
            "correctness": {
                "exact_outputs": False,
                "complete_requests": False,
                "no_starvation": False,
                "valid_lifecycle": False,
                "stable_p0_outputs": False,
            },
            "metrics": {},
        }
    selected_workload = _case_workload(
        workload_rows,
        case_spec["scenario"],
    )
    timeline_rows = _read_jsonl(
        case_dir / "request_timeline.jsonl"
    )
    scheduler_rows = _read_jsonl(
        case_dir / "scheduler_trace.jsonl"
    )
    memory_rows = _read_jsonl(
        case_dir / "memory_trace.jsonl"
    )
    request_metrics = reconstruct_request_metrics(
        selected_workload,
        timeline_rows,
        scheduler_rows,
    )
    measured = [
        row for row in request_metrics
        if not row.get("warmup", False)
    ]
    case = {
        **case_spec,
        "measurement_start_ns": min(
            row["scheduled_arrival_ns"] for row in measured
        ),
        "measurement_end_ns": max(
            row["completion_ns"] for row in measured
        ),
        "required_service_buckets": sorted({
            row["service_time_bucket"] for row in measured
        }),
    }
    summary = summarize_repetition(
        case,
        request_metrics,
        memory_rows,
    )
    return {
        "case_id": case_spec["case_id"],
        **summary,
    }


def _apply_output_correctness(
    case_rows: list[dict],
    timeline_rows: list[dict],
) -> None:
    outputs = {}
    for row in timeline_rows:
        outputs.setdefault(row["case_id"], {})[row["request_id"]] = (
            row.get("output_token_ids")
        )
    by_key = {
        (
            row["policy"],
            row["scenario"],
            row["repetition"],
        ): row
        for row in case_rows
    }
    for scenario in CANONICAL_SCENARIOS:
        baseline_by_repetition = []
        for repetition in range(MEASURED_REPETITIONS):
            baseline = by_key[("P0", scenario, repetition)]
            baseline_outputs = outputs.get(baseline["case_id"], {})
            baseline_by_repetition.append(baseline_outputs)
            for policy in ("P2", "P3"):
                candidate = by_key[(policy, scenario, repetition)]
                candidate_outputs = outputs.get(
                    candidate["case_id"],
                    {},
                )
                candidate["correctness"]["exact_outputs"] = (
                    candidate_outputs == baseline_outputs
                )
        stable = all(
            value == baseline_by_repetition[0]
            for value in baseline_by_repetition[1:]
        )
        for repetition in range(MEASURED_REPETITIONS):
            by_key[
                ("P0", scenario, repetition)
            ]["correctness"]["stable_p0_outputs"] = stable


def finalize_artifacts(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    manifest = _read_json(run_dir / "run_manifest.json")
    matrix = build_case_matrix(manifest)
    workload_rows = _read_jsonl(
        run_dir / "workload_manifest.jsonl"
    )
    timeline_rows = _merged_case_rows(
        run_dir,
        matrix,
        "request_timeline.jsonl",
    )
    scheduler_rows = _merged_case_rows(
        run_dir,
        matrix,
        "scheduler_trace.jsonl",
    )
    memory_rows = _merged_case_rows(
        run_dir,
        matrix,
        "memory_trace.jsonl",
    )
    _write_jsonl(
        run_dir / "request_timeline.jsonl",
        timeline_rows,
    )
    _write_jsonl(
        run_dir / "scheduler_trace.jsonl",
        scheduler_rows,
    )
    _write_jsonl(
        run_dir / "memory_trace.jsonl",
        memory_rows,
    )

    case_rows = [
        _case_summary(run_dir, case_spec, workload_rows)
        for case_spec in matrix
    ]
    _apply_output_correctness(case_rows, timeline_rows)
    _write_jsonl(run_dir / "case_rows.jsonl", case_rows)

    process_rows = []
    for case_spec in matrix:
        process = _read_json(
            run_dir
            / "processes"
            / case_spec["case_id"]
            / "process.json"
        )
        process_rows.append({
            "case_id": case_spec["case_id"],
            "tinyvllm_dist_port": int(
                process["tinyvllm_dist_port"]
            ),
            "master_port": int(process["master_port"]),
        })
    manifest["expected_case_ids"] = [
        row["case_id"] for row in matrix
    ]
    manifest["process_port_pairs"] = process_rows
    _write_json(run_dir / "run_manifest.json", manifest)

    summary = classify_gate(manifest, case_rows)
    _write_json(run_dir / "summary.json", summary)
    (run_dir / "report.md").write_text(
        render_report(manifest, summary),
        encoding="utf-8",
    )
    required = set(FINAL_ARTIFACT_FILES) - {
        "artifact_hashes.json"
    }
    missing = sorted(
        filename
        for filename in required
        if not (run_dir / filename).is_file()
    )
    if missing:
        raise ValueError(
            "missing final artifacts: " + ", ".join(missing)
        )
    hashes = {
        filename: sha256_file(run_dir / filename)
        for filename in sorted(required)
    }
    _write_json(run_dir / "artifact_hashes.json", hashes)
    return summary


def initialize_remote_run(
    *,
    run_dir: Path,
    model_path: str,
    run_tag: str,
    source_evidence_path: Path,
    environment_evidence_path: Path,
    smoke_verification: dict | None = None,
    tokenizer=None,
) -> dict:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    source_evidence = _read_json(source_evidence_path)
    environment_evidence = _read_json(environment_evidence_path)
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,
        )
    prompt_bank = build_prompt_bank(
        tokenizer,
        model_id=str(model_path),
    )
    defaults = {
        "chunked_prefill_decode_first": True,
        "chunked_prefill_max_consecutive_chunks": 0,
        "chunked_prefill_mixed_batch": False,
        "chunked_prefill_mixed_min_prompt_tokens": 0,
    }
    resolved = {
        name: resolve_policy_config(name, defaults)
        for name in ("P0", "P1", "P2", "P3")
    }
    aliases = deduplicate_policies(resolved)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_tag": str(run_tag),
        "model_path": str(model_path),
        "source_tree_sha256": source_evidence["tree_sha256"],
        "environment_sha256": environment_identity_sha256(
            environment_evidence
        ),
        "prompt_bank_sha256": prompt_bank["prompt_bank_sha256"],
        "workload_sha256": None,
        "required_scenarios": list(CANONICAL_SCENARIOS),
        "measured_repetitions": MEASURED_REPETITIONS,
        "policy_identity_by_name": aliases["identity_by_name"],
        "canonical_policy_by_name": aliases[
            "canonical_policy_by_name"
        ],
        "resolved_policy_config_by_name": resolved,
        "common_engine_config": dict(COMMON_ENGINE_CONFIG),
        "drain_timeout_ns": CANONICAL_DRAIN_TIMEOUT_NS,
    }
    if smoke_verification is not None:
        manifest["smoke_verification"] = dict(
            smoke_verification
        )
    _write_json(run_dir / "prompt_bank.json", prompt_bank)
    _write_jsonl(
        run_dir / "calibration_manifest.jsonl",
        build_calibration_manifest(prompt_bank),
    )
    _write_jsonl(run_dir / "calibration_rows.jsonl", [])
    _write_json(run_dir / "run_manifest.json", manifest)
    return manifest


def _smoke_workload(prompt_bank: dict) -> list[dict]:
    prompts = _prompt_by_class(prompt_bank)
    specifications = (
        ("decode-active", "short", "long", 0),
        ("chunked-long", "long", "short", 50_000_000),
        ("later-arrival", "medium", "short", 100_000_000),
    )
    rows = []
    for index, (
        name,
        prompt_class,
        output_class,
        arrival_offset_ns,
    ) in enumerate(specifications):
        prompt = prompts[prompt_class]
        output_tokens = (
            16 if output_class == "long"
            else 4
        )
        rows.append({
            "schema_version": SCHEMA_VERSION,
            "generator_version": GENERATOR_VERSION,
            "scenario": "lifecycle_smoke",
            "request_id": f"smoke-{index:02d}-{name}",
            "warmup": False,
            "arrival_offset_ns": arrival_offset_ns,
            "requested_rate_rps": 0.0,
            "seed": 0,
            "prompt_id": prompt["prompt_id"],
            "prompt": prompt["prompt"],
            "prompt_sha256": prompt["prompt_sha256"],
            "prompt_token_ids": list(prompt["prompt_token_ids"]),
            "prompt_token_count": prompt["prompt_token_count"],
            "prompt_class": prompt_class,
            "output_class": output_class,
            "service_time_bucket": (
                f"{prompt_class}__{output_class}"
            ),
            "requested_output_tokens": output_tokens,
            "sampling": {
                "temperature": 0.0,
                "ignore_eos": True,
                "max_tokens": output_tokens,
            },
            "drain_timeout_ns": CANONICAL_DRAIN_TIMEOUT_NS,
            "starvation_deadline_ns": STARVATION_DEADLINE_NS,
        })
    return rows


def _write_final_hashes(run_dir: Path) -> None:
    required = set(FINAL_ARTIFACT_FILES) - {
        "artifact_hashes.json"
    }
    missing = sorted(
        filename
        for filename in required
        if not (run_dir / filename).is_file()
    )
    if missing:
        raise ValueError(
            "missing final artifacts: " + ", ".join(missing)
        )
    _write_json(
        run_dir / "artifact_hashes.json",
        {
            filename: sha256_file(run_dir / filename)
            for filename in sorted(required)
        },
    )


def run_smoke(
    *,
    run_dir: Path,
    python_bin: str,
    model_path: str,
    run_tag: str,
    source_evidence_path: Path,
    environment_evidence_path: Path,
) -> dict:
    run_dir = Path(run_dir)
    manifest = initialize_remote_run(
        run_dir=run_dir,
        model_path=model_path,
        run_tag=run_tag,
        source_evidence_path=source_evidence_path,
        environment_evidence_path=environment_evidence_path,
    )
    prompt_bank = _read_json(run_dir / "prompt_bank.json")
    workload = _smoke_workload(prompt_bank)
    _write_jsonl(run_dir / "workload_manifest.jsonl", workload)
    manifest["run_type"] = "smoke"
    manifest["required_scenarios"] = ["lifecycle_smoke"]
    manifest["measured_repetitions"] = 1
    manifest["workload_sha256"] = canonical_json_sha256(workload)
    smoke_policies = ("P0", "P2")
    manifest["smoke_policies"] = list(smoke_policies)
    used_pairs = set()
    case_specs = []
    for policy_order, policy in enumerate(smoke_policies):
        case_specs.append({
            "case_id": f"lifecycle_smoke__{policy}__r0",
            "run_tag": run_tag,
            "scenario": "lifecycle_smoke",
            "policy": policy,
            "repetition": 0,
            "policy_order": policy_order,
            "resolved_config": dict(
                manifest["resolved_policy_config_by_name"][policy]
            ),
            "policy_identity": manifest[
                "policy_identity_by_name"
            ][policy],
            "workload_sha256": manifest["workload_sha256"],
            "source_tree_sha256": manifest[
                "source_tree_sha256"
            ],
            "environment_sha256": manifest[
                "environment_sha256"
            ],
            "drain_timeout_ns": CANONICAL_DRAIN_TIMEOUT_NS,
        })
    _write_json(run_dir / "run_manifest.json", manifest)
    launch_rows = [
        _launch_case(
            run_dir=run_dir,
            python_bin=python_bin,
            model_path=model_path,
            run_manifest=manifest,
            case_spec=case_spec,
            workload_rows=workload,
            used_pairs=used_pairs,
        )
        for case_spec in case_specs
    ]
    timeline = _merged_case_rows(
        run_dir,
        case_specs,
        "request_timeline.jsonl",
    )
    scheduler = _merged_case_rows(
        run_dir,
        case_specs,
        "scheduler_trace.jsonl",
    )
    memory = _merged_case_rows(
        run_dir,
        case_specs,
        "memory_trace.jsonl",
    )
    _write_jsonl(run_dir / "request_timeline.jsonl", timeline)
    _write_jsonl(run_dir / "scheduler_trace.jsonl", scheduler)
    _write_jsonl(run_dir / "memory_trace.jsonl", memory)
    case_rows = [
        _case_summary(run_dir, case_spec, workload)
        for case_spec in case_specs
    ]
    _apply_output_correctness_smoke(case_rows, timeline)
    _write_jsonl(run_dir / "case_rows.jsonl", case_rows)
    process_rows = []
    for case_spec in case_specs:
        process = _read_json(
            run_dir
            / "processes"
            / case_spec["case_id"]
            / "process.json"
        )
        process_rows.append({
            "case_id": case_spec["case_id"],
            "tinyvllm_dist_port": int(
                process["tinyvllm_dist_port"]
            ),
            "master_port": int(process["master_port"]),
        })
    manifest["expected_case_ids"] = [
        row["case_id"] for row in case_specs
    ]
    manifest["process_port_pairs"] = process_rows
    _write_json(run_dir / "run_manifest.json", manifest)
    exact_outputs = all(
        row["correctness"]["exact_outputs"]
        for row in case_rows
    )
    lifecycle_complete = (
        all(row["status"] == "PASS" for row in case_rows)
        and all(row["status"] == "PASS" for row in launch_rows)
        and bool(timeline)
        and bool(scheduler)
        and bool(memory)
    )
    summary = {
        "classification": "SMOKE_ONLY",
        "lifecycle_complete": lifecycle_complete,
        "exact_outputs": exact_outputs,
        "case_count": len(case_rows),
    }
    _write_json(run_dir / "summary.json", summary)
    (run_dir / "report.md").write_text(
        render_report(manifest, summary),
        encoding="utf-8",
    )
    _write_final_hashes(run_dir)
    return summary


def _apply_output_correctness_smoke(
    case_rows: list[dict],
    timeline_rows: list[dict],
) -> None:
    outputs = {}
    for row in timeline_rows:
        outputs.setdefault(row["policy"], {})[
            row["request_id"]
        ] = row.get("output_token_ids")
    exact = (
        outputs.get("P0")
        and outputs.get("P0") == outputs.get("P2")
    )
    for row in case_rows:
        row["correctness"]["exact_outputs"] = bool(exact)


def run_calibration_remote(
    *,
    run_dir: Path,
    python_bin: str,
    model_path: str,
    run_tag: str,
    source_evidence_path: Path,
    environment_evidence_path: Path,
    smoke_run_dir: Path,
) -> dict:
    smoke_manifest = _read_json(
        Path(smoke_run_dir) / "run_manifest.json"
    )
    smoke_summary = _read_json(
        Path(smoke_run_dir) / "summary.json"
    )
    if (
        smoke_summary.get("classification") != "SMOKE_ONLY"
        or smoke_summary.get("lifecycle_complete") is not True
        or smoke_summary.get("exact_outputs") is not True
    ):
        raise ValueError("calibration requires verified smoke")
    source_evidence = _read_json(source_evidence_path)
    environment_evidence = _read_json(environment_evidence_path)
    environment_sha256 = environment_identity_sha256(
        environment_evidence
    )
    if (
        smoke_manifest.get("source_tree_sha256")
        != source_evidence.get("tree_sha256")
        or smoke_manifest.get("environment_sha256")
        != environment_sha256
    ):
        raise ValueError("smoke identity mismatch")
    smoke_marker = {
        "status": "PASS",
        "run_tag": smoke_manifest["run_tag"],
        "source_tree_sha256": source_evidence["tree_sha256"],
        "environment_sha256": environment_sha256,
    }
    manifest = initialize_remote_run(
        run_dir=run_dir,
        model_path=model_path,
        run_tag=run_tag,
        source_evidence_path=source_evidence_path,
        environment_evidence_path=environment_evidence_path,
        smoke_verification=smoke_marker,
    )
    return run_calibration(
        run_dir=run_dir,
        python_bin=python_bin,
        model_path=model_path,
        run_manifest=manifest,
    )


def render_report(run_manifest: dict, summary: dict) -> str:
    del run_manifest
    return (
        "# Production Arrival-Load Gate\n\n"
        f"Classification: `{summary['classification']}`\n"
    )


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--python-bin", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--resume", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Production arrival-load gate orchestrator",
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )
    snapshot = subparsers.add_parser("snapshot-source")
    snapshot.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    snapshot.add_argument("--out-dir", type=Path, required=True)
    _add_run_arguments(subparsers.add_parser("run-calibration"))
    canonical = subparsers.add_parser("run-canonical")
    _add_run_arguments(canonical)
    canonical.add_argument(
        "--source-evidence",
        type=Path,
    )
    canonical.add_argument(
        "--environment-evidence",
        type=Path,
    )
    smoke = subparsers.add_parser("run-smoke")
    _add_run_arguments(smoke)
    smoke.add_argument("--run-tag", required=True)
    smoke.add_argument(
        "--source-evidence",
        type=Path,
        required=True,
    )
    smoke.add_argument(
        "--environment-evidence",
        type=Path,
        required=True,
    )
    remote_calibration = subparsers.add_parser(
        "run-calibration-remote"
    )
    _add_run_arguments(remote_calibration)
    remote_calibration.add_argument("--run-tag", required=True)
    remote_calibration.add_argument(
        "--source-evidence",
        type=Path,
        required=True,
    )
    remote_calibration.add_argument(
        "--environment-evidence",
        type=Path,
        required=True,
    )
    remote_calibration.add_argument(
        "--smoke-run-dir",
        type=Path,
        required=True,
    )
    freeze = subparsers.add_parser("freeze-workload")
    freeze.add_argument("--run-dir", type=Path, required=True)
    finalize = subparsers.add_parser("finalize-artifacts")
    finalize.add_argument("--run-dir", type=Path, required=True)
    verify = subparsers.add_parser("verify-harness")
    verify.add_argument("--run-dir", type=Path, required=True)
    return parser


def _freeze_existing_calibration(run_dir: Path) -> dict:
    manifest = _read_json(run_dir / "run_manifest.json")
    selection = select_lambda_ref(
        _read_jsonl(run_dir / "calibration_rows.jsonl")
    )
    if selection["status"] != "PASS":
        return selection
    prompt_bank = _read_json(run_dir / "prompt_bank.json")
    workload = build_canonical_workload(
        lambda_ref=selection["lambda_ref"],
        prompt_bank=prompt_bank,
    )
    _write_jsonl(run_dir / "workload_manifest.jsonl", workload)
    manifest["workload_sha256"] = canonical_json_sha256(workload)
    manifest["calibration"] = {
        "status": "PASS",
        "lambda_ref_rps": selection["lambda_ref"],
        "ceiling_rate_rps": selection["ceiling_rate_rps"],
        "maximum_stable_throughput_rps": selection[
            "maximum_stable_throughput_rps"
        ],
    }
    _write_json(run_dir / "run_manifest.json", manifest)
    return {
        **selection,
        "workload_sha256": manifest["workload_sha256"],
    }


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "snapshot-source":
        result = snapshot_source(args.repo_root, args.out_dir)
    elif args.command == "run-calibration":
        run_dir = Path(args.run_dir)
        result = run_calibration(
            run_dir=run_dir,
            python_bin=args.python_bin,
            model_path=args.model_path,
            run_manifest=_read_json(run_dir / "run_manifest.json"),
            resume=args.resume,
        )
    elif args.command == "run-smoke":
        result = run_smoke(
            run_dir=Path(args.run_dir),
            python_bin=args.python_bin,
            model_path=args.model_path,
            run_tag=args.run_tag,
            source_evidence_path=args.source_evidence,
            environment_evidence_path=args.environment_evidence,
        )
    elif args.command == "run-calibration-remote":
        result = run_calibration_remote(
            run_dir=Path(args.run_dir),
            python_bin=args.python_bin,
            model_path=args.model_path,
            run_tag=args.run_tag,
            source_evidence_path=args.source_evidence,
            environment_evidence_path=args.environment_evidence,
            smoke_run_dir=args.smoke_run_dir,
        )
    elif args.command == "freeze-workload":
        result = _freeze_existing_calibration(Path(args.run_dir))
    elif args.command == "run-canonical":
        run_dir = Path(args.run_dir)
        manifest = _read_json(run_dir / "run_manifest.json")
        if (
            args.source_evidence is None
            or args.environment_evidence is None
        ):
            raise ValueError(
                "run-canonical requires source and environment evidence"
            )
        validate_run_evidence(
            manifest,
            source_evidence_path=args.source_evidence,
            environment_evidence_path=args.environment_evidence,
        )
        result = run_canonical(
            run_dir=run_dir,
            python_bin=args.python_bin,
            model_path=args.model_path,
            run_manifest=manifest,
            resume=args.resume,
        )
    elif args.command == "finalize-artifacts":
        result = finalize_artifacts(Path(args.run_dir))
    else:
        verifier = _load_local_module(
            "arrival_load_independent_verify",
            _REPO_ROOT / "tools" / "arrival_load_verify.py",
        )
        result = verifier.verify_run(Path(args.run_dir))
    print(json.dumps(
        result,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
