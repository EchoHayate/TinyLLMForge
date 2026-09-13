#!/usr/bin/env python3
"""Remote orchestration contract for SLO-aware cohort-burst gates."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict
import gc
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import tarfile
import time
from typing import Mapping

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import run_staged_inference_benchmark_remote as base
from tools import profile_slo_cohort_burst_ceiling as profile
from tools import slo_cohort_burst_gate as qualification_gate
from tools import slo_cohort_burst_verify as qualification_verify


APPROVED_ROOT = base.APPROVED_ROOT
TASK_REMOTE_ROOT = APPROVED_ROOT + "/slo-cohort-burst"
REMOTE_HOST = base.REMOTE_HOST
REMOTE_PYTHON = base.REMOTE_PYTHON
MODEL_PATH = base.MODEL_PATHS["qwen3-0.6b"]
DEFAULT_KERBEROS_CACHE = base.KRB5CCNAME
MINIMUM_KERBEROS_LIFETIME_SECONDS = 1_800
REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_ARTIFACT_ROOT = (
    REPO_ROOT / "artifacts" / "slo_cohort_burst_ceiling"
)
QUALIFICATION_LOCAL_ARTIFACT_ROOT = (
    REPO_ROOT / "artifacts" / "slo_cohort_burst"
)
SOURCE_FILES = (
    "tinyvllm",
    "tools/slo_cohort_burst_ceiling.py",
    "tools/profile_slo_cohort_burst_ceiling.py",
    "tools/slo_cohort_burst_gate.py",
    "tools/slo_cohort_burst_verify.py",
    "tools/run_slo_cohort_burst_remote.py",
)
REQUIRED_TERMINAL_FILES = frozenset({
    "raw_rows.jsonl",
    "cost_table.json",
    "ceiling_summary.json",
    "source_manifest.json",
    "remote_verify.json",
})
COMPACT_FILES = REQUIRED_TERMINAL_FILES | frozenset({
    "runner.log",
})
QUALIFICATION_AUTHORITATIVE_FILES = frozenset({
    "source_manifest.json",
    "environment.json",
    "cost_profile_rows.jsonl",
    "arrival_traces.json",
    "cost_table.json",
    "decision_rows.jsonl",
    "execution_rows.jsonl",
    "request_rows.jsonl",
    "correctness_rows.jsonl",
    "summary.json",
    "manifest.json",
})
CORRECTNESS_TERMINAL_FILES = frozenset({
    "final_bundle/source_manifest.json",
    "final_bundle/environment.json",
    "final_bundle/cost_profile_rows.jsonl",
    "final_bundle/cost_table.json",
    "final_bundle/correctness_rows.jsonl",
    "final_bundle/manifest.json",
    "final_bundle/remote_verify.json",
})
CANONICAL_TERMINAL_FILES = frozenset({
    *(
        f"final_bundle/{name}"
        for name in QUALIFICATION_AUTHORITATIVE_FILES
    ),
    "final_bundle/remote_verify.json",
})
DOWNLOAD_RETRIES = 3
QUALIFICATION_WORKLOADS = ("decode_heavy", "mixed", "bursty_eos")
QUALIFICATION_LOADS = ("low", "medium", "high")
QUALIFICATION_REPETITIONS = 5
REQUESTS_PER_REPETITION = 26
LOAD_FRACTIONS = {
    "low": 0.40,
    "medium": 0.70,
    "high": 0.90,
}
PAIRED_ARM_ORDER = (
    ("baseline", "candidate"),
    ("candidate", "baseline"),
    ("baseline", "candidate"),
    ("candidate", "baseline"),
    ("baseline", "candidate"),
)
QUALIFICATION_SOURCE_PATHS = (
    "tinyvllm/config.py",
    "tinyvllm/engine/exact_greedy_cohort_burst.py",
    "tinyvllm/engine/llm_engine.py",
    "tinyvllm/engine/model_runner.py",
    "tinyvllm/engine/scheduler.py",
    "tinyvllm/engine/slo_cohort_burst.py",
    "tools/profile_slo_cohort_burst_ceiling.py",
    "tools/run_slo_cohort_burst_remote.py",
    "tools/slo_cohort_burst_ceiling.py",
    "tools/slo_cohort_burst_gate.py",
    "tools/slo_cohort_burst_verify.py",
)

validate_kerberos = base.validate_kerberos
require_pushed_head = base.require_pushed_head


def _canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _payload_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _load_json(path: Path) -> object:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(
                    f"JSONL row {line_number} must be an object"
                )
            rows.append(row)
    if not rows:
        raise ValueError("JSONL artifact is empty")
    return rows


def _source_digest_inventory(source_root: Path) -> dict[str, str]:
    root = Path(source_root)
    inventory = {}
    for relative in QUALIFICATION_SOURCE_PATHS:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"qualification source is missing: {relative}")
        inventory[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return inventory


def _lease_payload(lease) -> dict[str, object]:
    payload = asdict(lease)
    identity = payload.pop("identity_sha256")
    payload["schema_version"] = "exact-greedy-cohort-burst.lease.v1"
    if _payload_sha256(payload) != identity:
        raise ValueError("captured cohort lease identity mismatch")
    return payload


def _result_payload(result) -> dict[str, object]:
    return {
        "schema_version": "exact-greedy-cohort-burst.result-identity.v1",
        "lease_identity_sha256": result.lease_identity_sha256,
        "graph_identity_sha256": result.graph_identity_sha256,
        "graph_generation": result.graph_generation,
        "replay_count": result.replay_count,
        "rows": [asdict(row) for row in result.rows],
        "token_d2h_calls": result.token_d2h_calls,
        "sampled_logit_d2h_calls": result.sampled_logit_d2h_calls,
    }


def _publication_payload(publication) -> dict[str, object]:
    return {
        "ordered_sequence_ids": list(
            publication.ordered_sequence_ids
        ),
        "rows": [
            {
                "sequence_id": sequence_id,
                "commit_tokens": list(tokens),
            }
            for sequence_id, tokens in zip(
                publication.ordered_sequence_ids,
                publication.commit_tokens,
            )
        ],
    }


class _CohortEvidenceTap:
    def __init__(self, engine):
        self._engine = engine
        self._lease = None
        self._result = None
        self._publication = None
        self._original_prepare = (
            engine.scheduler.prepare_exact_greedy_cohort_burst
        )
        self._original_commit = (
            engine.scheduler.prepare_exact_greedy_cohort_burst_commit
        )
        self._original_call = engine.model_runner.call

        def capture_prepare(*args, **kwargs):
            lease = self._original_prepare(*args, **kwargs)
            if lease is not None:
                if self._lease is not None:
                    raise RuntimeError("overlapping captured cohort lease")
                self._lease = lease
            return lease

        def capture_call(method, *args, **kwargs):
            result = self._original_call(method, *args, **kwargs)
            if method == "run_exact_greedy_cohort_burst":
                self._result = result
            return result

        def capture_commit(*args, **kwargs):
            prepared = self._original_commit(*args, **kwargs)
            self._publication = (
                prepared.exact_cohort_burst_publication
            )
            return prepared

        engine.scheduler.prepare_exact_greedy_cohort_burst = (
            capture_prepare
        )
        engine.scheduler.prepare_exact_greedy_cohort_burst_commit = (
            capture_commit
        )
        engine.model_runner.call = capture_call

    def take(self, *, case: Mapping[str, object], observation: Mapping):
        execution = observation.get(
            "exact_greedy_cohort_burst_execution_telemetry"
        )
        decision = observation.get("slo_cohort_decision_telemetry")
        if decision is None and execution is None:
            return None
        if decision is None:
            raise RuntimeError("cohort decision evidence is missing")
        decision_row = {
            "schema_version": "slo-cohort-burst.decision-evidence.v1",
            "case": dict(case),
            "decision": dict(decision),
        }
        if execution is None:
            if any(
                value is not None
                for value in (
                    self._lease,
                    self._result,
                    self._publication,
                )
            ):
                raise RuntimeError(
                    "fallback decision retained partial execution state"
                )
            return decision_row, None
        if any(
            value is None
            for value in (
                self._lease,
                self._result,
                self._publication,
            )
        ):
            raise RuntimeError("cohort evidence capture is incomplete")
        lease = _lease_payload(self._lease)
        result = _result_payload(self._result)
        row = {
            "schema_version": "slo-cohort-burst.execution-evidence.v1",
            "case": dict(case),
            "lease": lease,
            "lease_identity_sha256": self._lease.identity_sha256,
            "result": result,
            "result_identity_sha256": _payload_sha256(result),
            "publication": _publication_payload(self._publication),
            "execution": dict(execution),
        }
        self._lease = None
        self._result = None
        self._publication = None
        return decision_row, row

    def close(self) -> None:
        self._engine.scheduler.prepare_exact_greedy_cohort_burst = (
            self._original_prepare
        )
        self._engine.scheduler.prepare_exact_greedy_cohort_burst_commit = (
            self._original_commit
        )
        self._engine.model_runner.call = self._original_call


def _qualification_prompt_tokens(
    *,
    source_commit: str,
    workload: str,
    ordinal: int,
    prompt_tokens: int,
) -> list[int]:
    seed = int(
        hashlib.sha256(
            f"{source_commit}:{workload}:{ordinal}".encode("utf-8")
        ).hexdigest()[:8],
        16,
    )
    return [
        100 + ((seed + token_index * 997) % 30_000)
        for token_index in range(prompt_tokens)
    ]


def _write_bytes_exclusive(path: Path, payload: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def write_qualification_bundle(
    *,
    output_dir: Path,
    stage: str,
    artifacts: Mapping[str, object],
) -> Path:
    if stage == "correctness":
        authoritative = set(
            qualification_verify.CORRECTNESS_AUTHORITATIVE_ARTIFACTS
        )
    elif stage == "canonical":
        authoritative = set(
            qualification_verify.AUTHORITATIVE_ARTIFACTS
        )
    else:
        raise ValueError("unsupported qualification stage")
    if set(artifacts) != authoritative:
        raise ValueError("qualification artifact inventory mismatch")
    root = Path(output_dir)
    if root.is_symlink() or (root.exists() and not root.is_dir()):
        raise ValueError("qualification output is invalid")
    root.mkdir(parents=True, exist_ok=True)
    final_bundle = root / "final_bundle"
    if final_bundle.exists() or final_bundle.is_symlink():
        raise ValueError("qualification final bundle already exists")
    final_bundle.mkdir()
    hashes = {}
    for relative in sorted(authoritative):
        payload = artifacts[relative]
        encoded = (
            b"".join(
                _canonical_json_bytes(row)
                for row in payload
            )
            if relative.endswith(".jsonl")
            else _canonical_json_bytes(payload)
        )
        _write_bytes_exclusive(final_bundle / relative, encoded)
        hashes[relative] = hashlib.sha256(encoded).hexdigest()
    _write_bytes_exclusive(
        final_bundle / "manifest.json",
        _canonical_json_bytes({
            "schema_version": qualification_verify.MANIFEST_SCHEMA_VERSION,
            "artifact_sha256": hashes,
        }),
    )
    return final_bundle


def build_frozen_arrival_traces(
    *,
    source_commit: str,
    saturation_rps_by_workload: Mapping[str, float],
) -> dict[str, object]:
    validate_source_commit(source_commit, pushed_head=source_commit)
    if set(saturation_rps_by_workload) != set(QUALIFICATION_WORKLOADS):
        raise ValueError("saturation workload inventory mismatch")
    saturation = {}
    for workload, value in saturation_rps_by_workload.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ValueError("saturation rate must be positive and finite")
        saturation[workload] = float(value)
    cases = []
    for workload in QUALIFICATION_WORKLOADS:
        for load in QUALIFICATION_LOADS:
            arrival_gap_ns = max(
                1,
                round(
                    1_000_000_000
                    / (
                        saturation[workload]
                        * LOAD_FRACTIONS[load]
                    )
                ),
            )
            for repetition in range(QUALIFICATION_REPETITIONS):
                requests = []
                for request_index in range(REQUESTS_PER_REPETITION):
                    ordinal = (
                        repetition * REQUESTS_PER_REPETITION
                        + request_index
                    )
                    if workload == "mixed":
                        prompt_tokens = (
                            256
                            if ordinal < 91
                            else 2048
                            if ordinal < 117
                            else 8192
                        )
                        maximum_output_tokens = (
                            64 if prompt_tokens == 256 else 128
                        )
                    else:
                        prompt_tokens = 256
                        maximum_output_tokens = (
                            128
                            if workload == "decode_heavy"
                            else 8 * (1 + ordinal % 16)
                        )
                    prompt = _qualification_prompt_tokens(
                        source_commit=source_commit,
                        workload=workload,
                        ordinal=ordinal,
                        prompt_tokens=prompt_tokens,
                    )
                    if workload == "bursty_eos":
                        arrival_offset_ns = (
                            request_index // 5
                        ) * arrival_gap_ns * 5
                    else:
                        arrival_offset_ns = (
                            request_index * arrival_gap_ns
                        )
                    requests.append({
                        "request_id": (
                            f"{workload}-{load}-r{repetition}"
                            f"-q{request_index}"
                        ),
                        "prompt_sha256": hashlib.sha256(
                            _canonical_json_bytes(prompt).rstrip(b"\n")
                        ).hexdigest(),
                        "arrival_offset_ns": arrival_offset_ns,
                        "prompt_tokens": prompt_tokens,
                        "maximum_output_tokens": maximum_output_tokens,
                        "ignore_eos": workload != "bursty_eos",
                    })
                cases.append({
                    "workload": workload,
                    "load": load,
                    "repetition": repetition,
                    "requests": requests,
                })
    return {
        "schema_version": "slo-cohort-burst.arrival-traces.v1",
        "minimum_requests_per_workload_load_arm": 128,
        "minimum_repetitions": QUALIFICATION_REPETITIONS,
        "arm_order_by_repetition": [
            list(order) for order in PAIRED_ARM_ORDER
        ],
        "cases": cases,
    }


def _qualification_engine_config(
    *,
    cost_table_path: Path,
    cohort_enabled: bool = True,
) -> dict[str, object]:
    return {
        "enforce_eager": False,
        "max_num_seqs": 8,
        "max_num_batched_tokens": 131_072,
        "max_model_len": 32_768,
        "max_num_prefill_tokens_per_step": 0,
        "autoregressive_draft_command_timeline": True,
        "autoregressive_draft_command_timeline_max_rows": 16_384,
        "exact_greedy_decode_burst": False,
        "exact_greedy_decode_burst_tokens": 8,
        "exact_greedy_cohort_burst": bool(cohort_enabled),
        "exact_greedy_cohort_burst_widths": (1, 2, 4, 8),
        "exact_greedy_cohort_burst_max_batch_size": 8,
        "exact_greedy_cohort_burst_target_itl_ns": 40_000_000,
        "exact_greedy_cohort_burst_target_ttft_ns": 1_000_000_000,
        "exact_greedy_cohort_burst_reserve_ns": 2_000_000,
        "exact_greedy_cohort_burst_cost_table_path": os.fspath(
            cost_table_path
        ),
    }


def build_correctness_matrix(*, run_case) -> list[dict]:
    if not callable(run_case):
        raise ValueError("correctness case runner must be callable")
    matrix = []
    for batch_size in (1, 2, 4, 8):
        for burst_width in (1, 2, 4, 8):
            baseline = run_case(
                batch_size=batch_size,
                burst_width=burst_width,
                arm="baseline",
            )
            candidate = run_case(
                batch_size=batch_size,
                burst_width=burst_width,
                arm="candidate",
            )
            baseline_rows = baseline.get("rows")
            candidate_rows = candidate.get("rows")
            if (
                not isinstance(baseline_rows, list)
                or not isinstance(candidate_rows, list)
                or len(baseline_rows) != batch_size
                or len(candidate_rows) != batch_size
            ):
                raise ValueError("correctness case row inventory mismatch")
            rows = []
            for row_index, (left, right) in enumerate(
                zip(baseline_rows, candidate_rows)
            ):
                rows.append({
                    "row_index": row_index,
                    "baseline_output_token_ids": list(
                        left["output_token_ids"]
                    ),
                    "candidate_output_token_ids": list(
                        right["output_token_ids"]
                    ),
                    "baseline_output_text_sha256": left[
                        "output_text_sha256"
                    ],
                    "candidate_output_text_sha256": right[
                        "output_text_sha256"
                    ],
                    "baseline_sampled_logits_sha256": left[
                        "sampled_logits_sha256"
                    ],
                    "candidate_sampled_logits_sha256": right[
                        "sampled_logits_sha256"
                    ],
                    "baseline_argmax_token_ids": list(
                        left["argmax_token_ids"]
                    ),
                    "candidate_argmax_token_ids": list(
                        right["argmax_token_ids"]
                    ),
                })
            matrix.append({
                "schema_version": (
                    "slo-cohort-burst.correctness-case.v1"
                ),
                "batch_size": batch_size,
                "burst_width": burst_width,
                "rows": rows,
                "duplicate_forwards": int(
                    candidate["duplicate_forwards"]
                ),
                "duplicate_commits": int(
                    candidate["duplicate_commits"]
                ),
                "unauthorized_kv_publications": int(
                    candidate["unauthorized_kv_publications"]
                ),
                "pending_leases_after_case": int(
                    candidate["pending_leases_after_case"]
                ),
            })
    return matrix


def run_canonical_matrix(
    *,
    arrival_traces: Mapping[str, object],
    run_case,
) -> dict[str, list[dict]]:
    if not callable(run_case):
        raise ValueError("canonical case runner must be callable")
    cases = arrival_traces.get("cases")
    arm_order = arrival_traces.get("arm_order_by_repetition")
    if not isinstance(cases, list) or not isinstance(arm_order, list):
        raise ValueError("canonical arrival trace is invalid")
    by_repetition = {}
    for case in cases:
        repetition = case.get("repetition")
        if (
            isinstance(repetition, bool)
            or not isinstance(repetition, int)
            or repetition < 0
        ):
            raise ValueError("canonical repetition is invalid")
        by_repetition.setdefault(repetition, []).append(case)
    rows = {
        "request_rows": [],
        "decision_rows": [],
        "execution_rows": [],
    }
    for repetition in sorted(by_repetition):
        if repetition >= len(arm_order):
            raise ValueError("canonical arm order is incomplete")
        order = arm_order[repetition]
        if (
            not isinstance(order, list)
            or sorted(order) != ["baseline", "candidate"]
        ):
            raise ValueError("canonical paired arm order is invalid")
        for arm in order:
            for trace_case in by_repetition[repetition]:
                result = run_case(
                    trace_case=trace_case,
                    arm=arm,
                )
                for name in rows:
                    produced = result.get(name)
                    if not isinstance(produced, list):
                        raise ValueError(
                            f"canonical {name} inventory is invalid"
                        )
                    rows[name].extend(produced)
    return rows


def _run_source_bound_calibration(
    *,
    model: str,
    source_commit: str,
    output_dir: Path,
) -> tuple[dict, list[dict], dict]:
    from tinyvllm.engine.llm_engine import LLMEngine
    from tinyvllm.sampling_params import SamplingParams

    cases = profile.build_frozen_case_inventory(source_commit)
    gpu_uuid, gpu_name = profile._gpu_identity()
    source_identity = {
        "source_commit": source_commit,
        "source_patch_sha256": profile._source_tree_sha256(),
        "model": "Qwen3-0.6B",
        "checkpoint_sha256": profile._checkpoint_sha256(Path(model)),
        "gpu_uuid": gpu_uuid,
        "gpu_name": gpu_name,
        "tensor_parallel_size": 1,
        "config_sha256": profile._config_sha256(cases),
    }
    calibration_dir = Path(output_dir) / "calibration"
    verification = profile.run_profile_inventory(
        model=model,
        cases=cases,
        output_dir=calibration_dir,
        source_identity=source_identity,
        engine_factory=LLMEngine,
        sampling_params_factory=SamplingParams,
        step_timer_factory=profile.CudaTargetStepTimer,
    )
    if (
        verification.get("verified") is not True
        or verification.get("classification") != "CONTINUE_RUNTIME"
    ):
        raise RuntimeError("source-bound Stage-0 calibration did not pass")
    raw_rows = _load_jsonl(calibration_dir / "raw_rows.jsonl")
    cost_table = _load_json(calibration_dir / "cost_table.json")
    bound_source_identity = cost_table.get("source_identity")
    if not isinstance(bound_source_identity, Mapping):
        raise ValueError("calibration source identity is missing")
    qualification_verify.verify_cost_table_against_profile_rows(
        cost_table,
        bound_source_identity,
        raw_rows,
    )
    return dict(bound_source_identity), raw_rows, cost_table


def _saturation_rates(cost_table: Mapping[str, object]) -> dict[str, float]:
    entries = cost_table.get("entries")
    if not isinstance(entries, Mapping):
        raise ValueError("cost table entries are missing")

    def token_rate(context_bucket: int) -> float:
        candidates = [
            entry
            for entry in entries.values()
            if (
                isinstance(entry, Mapping)
                and entry.get("batch_size") == 8
                and entry.get("burst_width") == 1
                and entry.get("context_bucket") >= context_bucket
            )
        ]
        if not candidates:
            raise ValueError("cost table lacks a conservative B8 entry")
        entry = min(
            candidates,
            key=lambda row: row["context_bucket"],
        )
        return 8_000_000_000.0 / float(entry["p99_ns"])

    short_rate = token_rate(384)
    mixed_rate = min(token_rate(384), token_rate(8_320))
    return {
        "decode_heavy": short_rate / 128.0,
        "mixed": mixed_rate / 128.0,
        "bursty_eos": short_rate / 68.0,
    }


def _set_cohort_arm(engine, *, enabled: bool, widths=(1, 2, 4, 8)):
    # Keep scheduler-side request lifecycle telemetry enabled for both
    # arms.  The ModelRunner config is the execution switch that makes
    # the baseline an ordinary K1 runtime.
    engine.scheduler.exact_greedy_cohort_burst = True
    engine.scheduler.exact_greedy_cohort_burst_widths = tuple(widths)
    engine.model_runner.config.exact_greedy_cohort_burst = bool(enabled)


def _graph_identity_by_batch(engine) -> dict[str, str]:
    identities = {}
    for batch_size in (1, 2, 4, 8):
        capability = (
            engine.model_runner.exact_greedy_cohort_burst_capability(
                batch_size=batch_size,
                block_table_width=(
                    engine.model_runner.config.max_model_len
                    + engine.model_runner.block_size
                    - 1
                )
                // engine.model_runner.block_size,
            )
        )
        identity = capability.get("graph_identity_sha256")
        if (
            capability.get("shape_supported") is not True
            or not isinstance(identity, str)
        ):
            raise RuntimeError(
                f"cohort graph unavailable for batch {batch_size}"
            )
        identities[str(batch_size)] = identity
    return identities


def _source_manifest(
    *,
    source_identity: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": "slo-cohort-burst.source-manifest.v1",
        "source_identity": dict(source_identity),
        "dirty": False,
        "source_sha256": _source_digest_inventory(REPO_ROOT),
    }


def _environment(
    *,
    source_identity: Mapping[str, object],
    graph_identity_sha256_by_batch: Mapping[str, str],
    eos_token_id: int,
) -> dict[str, object]:
    return {
        "schema_version": "slo-cohort-burst.environment.v1",
        "source_commit": source_identity["source_commit"],
        "model": source_identity["model"],
        "checkpoint_sha256": source_identity["checkpoint_sha256"],
        "gpu_uuid": source_identity["gpu_uuid"],
        "gpu_name": source_identity["gpu_name"],
        "tensor_parallel_size": 1,
        "temperature": 0.0,
        "completion_only": True,
        "eos_token_id": int(eos_token_id),
        "target_itl_ns": 40_000_000,
        "target_ttft_ns": 1_000_000_000,
        "reserve_ns": 2_000_000,
        "graph_identity_sha256_by_batch": dict(
            graph_identity_sha256_by_batch
        ),
    }


def _create_qualification_engine(
    *,
    model: str,
    cost_table_path: Path,
    cohort_enabled: bool = True,
):
    from tinyvllm.engine.llm_engine import LLMEngine

    return LLMEngine(
        model,
        **_qualification_engine_config(
            cost_table_path=cost_table_path,
            cohort_enabled=cohort_enabled,
        ),
    )


def _qualification_sampling_params_factory():
    from tinyvllm.sampling_params import SamplingParams

    return SamplingParams


def _capture_qualification_graphs(engine) -> None:
    for batch_size in (1, 2, 4, 8):
        graph = (
            engine.model_runner
            .capture_exact_greedy_cohort_burst_graph(
                batch_size,
                correctness_trace=True,
            )
        )
        if graph is None:
            raise RuntimeError(
                "correctness cohort graph unavailable for "
                f"batch {batch_size}"
            )


def _logits_rows(value, *, batch_size: int) -> list[list[float]]:
    if hasattr(value, "detach"):
        value = value.detach().to(
            dtype=__import__("torch").float32
        ).contiguous().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    if (
        not isinstance(value, (list, tuple))
        or len(value) != batch_size
    ):
        raise RuntimeError("correctness logits batch shape mismatch")
    rows = []
    for row in value:
        if not isinstance(row, (list, tuple)) or not row:
            raise RuntimeError("correctness logits row is empty")
        normalized = [float(item) for item in row]
        if any(not math.isfinite(item) for item in normalized):
            raise RuntimeError("correctness logits are not finite")
        rows.append(normalized)
    return rows


def _argmax(values: list[float]) -> int:
    return max(range(len(values)), key=values.__getitem__)


def _pending_cohort_inventory(engine) -> int:
    scheduler = engine.scheduler
    return int(
        getattr(
            scheduler,
            "_exact_greedy_cohort_burst_pending_lease",
            None,
        )
        is not None
    ) + int(
        getattr(
            scheduler,
            "_exact_greedy_cohort_burst_pending_transaction",
            None,
        )
        is not None
    )


def _run_correctness_case(
    *,
    engine,
    sampling_params_factory,
    source_commit: str,
    batch_size: int,
    burst_width: int,
    arm: str,
) -> dict[str, object]:
    if arm not in ("baseline", "candidate"):
        raise ValueError("correctness arm is invalid")
    if not engine.is_finished():
        raise RuntimeError("correctness engine is not drained")
    enabled = arm == "candidate"
    original_target_itl_ns = (
        engine.scheduler.exact_greedy_cohort_burst_target_itl_ns
    )
    original_target_ttft_ns = (
        engine.scheduler.exact_greedy_cohort_burst_target_ttft_ns
    )
    original_reserve_ns = (
        engine.scheduler.exact_greedy_cohort_burst_reserve_ns
    )
    _set_cohort_arm(
        engine,
        enabled=enabled,
        widths=(burst_width,),
    )
    if enabled and burst_width > 1:
        engine.scheduler.exact_greedy_cohort_burst_target_itl_ns = (
            1_000_000_000_000
        )
        engine.scheduler.exact_greedy_cohort_burst_target_ttft_ns = (
            1_000_000_000_000
        )
        engine.scheduler.exact_greedy_cohort_burst_reserve_ns = 0
    engine.enable_step_logits_authority_recording(
        True,
        timeout_s=60.0,
    )
    sequence_ids = []
    for row_index in range(batch_size):
        prompt = _qualification_prompt_tokens(
            source_commit=source_commit,
            workload=f"correctness-b{batch_size}-k{burst_width}",
            ordinal=row_index,
            prompt_tokens=256,
        )
        sequence_ids.append(engine.add_request(
            prompt,
            sampling_params_factory(
                temperature=0.0,
                max_tokens=burst_width + 1,
                ignore_eos=True,
            ),
        ))

    try:
        _outputs, _count = engine.step(completion_only=True)
        first_observation = dict(engine.last_step_observation or {})
        first_deltas = first_observation.get(
            "new_completion_tokens_by_seq"
        )
        if (
            not isinstance(first_deltas, Mapping)
            or any(
                len(first_deltas.get(sequence_id, ())) != 1
                for sequence_id in sequence_ids
            )
        ):
            raise RuntimeError(
                "correctness prefill did not emit one token per row"
            )

        tokens_by_sequence = {
            sequence_id: [] for sequence_id in sequence_ids
        }
        logits_by_sequence = {
            sequence_id: [] for sequence_id in sequence_ids
        }
        evidence_rows = []
        if enabled and burst_width > 1:
            engine.enable_step_logits_authority_recording(
                False,
                timeout_s=60.0,
            )
            tap = _CohortEvidenceTap(engine)
            try:
                _outputs, _count = engine.step(
                    completion_only=True,
                    exact_burst_correctness_trace=True,
                )
                observation = dict(engine.last_step_observation or {})
                captured = tap.take(
                    case={
                        "workload": "correctness",
                        "load": "correctness",
                        "repetition": 0,
                        "arm": arm,
                    },
                    observation=observation,
                )
            finally:
                tap.close()
            if captured is None:
                raise RuntimeError(
                    "correctness candidate did not execute cohort burst"
                )
            _decision_row, execution_row = captured
            evidence_rows.append(execution_row)
            result_rows = execution_row["result"]["rows"]
            if len(result_rows) != batch_size:
                raise RuntimeError(
                    "correctness result row inventory mismatch"
                )
            for sequence_id, result_row in zip(
                sequence_ids,
                result_rows,
            ):
                if result_row["sequence_id"] != sequence_id:
                    raise RuntimeError(
                        "correctness result sequence order mismatch"
                    )
                tokens_by_sequence[sequence_id].extend(
                    result_row["tokens"]
                )
                logits_by_sequence[sequence_id].extend(
                    _logits_rows(
                        result_row["sampled_logits"],
                        batch_size=burst_width,
                    )
                )
        else:
            for _ordinal in range(burst_width):
                _outputs, _count = engine.step(completion_only=True)
                observation = dict(engine.last_step_observation or {})
                deltas = observation.get(
                    "new_completion_tokens_by_seq"
                )
                if not isinstance(deltas, Mapping):
                    raise RuntimeError(
                        "correctness token deltas are unavailable"
                    )
                logits = _logits_rows(
                    engine.read_step_logits_authority(),
                    batch_size=batch_size,
                )
                for row_index, sequence_id in enumerate(sequence_ids):
                    delta = list(deltas.get(sequence_id, ()))
                    if len(delta) != 1:
                        raise RuntimeError(
                            "ordinary correctness step must emit one "
                            "token per row"
                        )
                    tokens_by_sequence[sequence_id].extend(delta)
                    logits_by_sequence[sequence_id].append(
                        logits[row_index]
                    )
        if not engine.is_finished():
            raise RuntimeError("correctness case did not drain")
    finally:
        engine.enable_step_logits_authority_recording(
            False,
            timeout_s=60.0,
        )
        engine.scheduler.exact_greedy_cohort_burst_target_itl_ns = (
            original_target_itl_ns
        )
        engine.scheduler.exact_greedy_cohort_burst_target_ttft_ns = (
            original_target_ttft_ns
        )
        engine.scheduler.exact_greedy_cohort_burst_reserve_ns = (
            original_reserve_ns
        )

    rows = []
    for sequence_id in sequence_ids:
        tokens = tokens_by_sequence[sequence_id]
        logits = logits_by_sequence[sequence_id]
        if len(tokens) != burst_width or len(logits) != burst_width:
            raise RuntimeError("correctness evidence length mismatch")
        rows.append({
            "output_token_ids": tokens,
            "output_text_sha256": hashlib.sha256(
                engine.tokenizer.decode(tokens).encode("utf-8")
            ).hexdigest(),
            "sampled_logits_sha256": _payload_sha256(logits),
            "argmax_token_ids": [_argmax(row) for row in logits],
        })

    duplicate_forwards = 0
    duplicate_commits = 0
    unauthorized_publications = 0
    if evidence_rows:
        evidence = evidence_rows[0]
        if evidence["result"]["replay_count"] != burst_width:
            duplicate_forwards = max(
                0,
                evidence["result"]["replay_count"] - burst_width,
            )
        publication = evidence["publication"]
        published_ids = publication["ordered_sequence_ids"]
        duplicate_commits = len(published_ids) - len(set(published_ids))
        for result_row, published_row in zip(
            evidence["result"]["rows"],
            publication["rows"],
        ):
            if (
                result_row["sequence_id"]
                != published_row["sequence_id"]
                or result_row["tokens"][
                    :len(published_row["commit_tokens"])
                ]
                != published_row["commit_tokens"]
            ):
                unauthorized_publications += 1
    return {
        "rows": rows,
        "duplicate_forwards": duplicate_forwards,
        "duplicate_commits": duplicate_commits,
        "unauthorized_kv_publications": unauthorized_publications,
        "pending_leases_after_case": _pending_cohort_inventory(engine),
    }


def _run_correctness_matrix_on_engine(
    *,
    engine,
    sampling_params_factory,
    source_commit: str,
) -> list[dict]:
    return build_correctness_matrix(
        run_case=lambda **case: _run_correctness_case(
            engine=engine,
            sampling_params_factory=sampling_params_factory,
            source_commit=source_commit,
            **case,
        )
    )


def _canonical_case_identity(
    trace_case: Mapping[str, object],
    arm: str,
) -> dict[str, object]:
    return {
        "workload": trace_case["workload"],
        "load": trace_case["load"],
        "repetition": trace_case["repetition"],
        "arm": arm,
    }


def _run_open_loop_case(
    *,
    engine,
    sampling_params_factory,
    source_commit: str,
    trace_case: Mapping[str, object],
    arm: str,
    clock_ns=time.monotonic_ns,
    sleep=time.sleep,
) -> dict[str, list[dict]]:
    if arm not in ("baseline", "candidate"):
        raise ValueError("canonical arm is invalid")
    if not engine.is_finished():
        raise RuntimeError("canonical engine is not drained")
    requests = trace_case.get("requests")
    if not isinstance(requests, list) or not requests:
        raise ValueError("canonical request inventory is empty")
    _set_cohort_arm(engine, enabled=arm == "candidate")
    try:
        torch = __import__("torch")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except (ImportError, AttributeError):
        pass
    tap = _CohortEvidenceTap(engine) if arm == "candidate" else None
    pending = deque(enumerate(requests))
    request_by_sequence = {}
    completed = {}
    decision_rows = []
    execution_rows = []
    peak_reserved_bytes = 0
    epoch_ns = clock_ns()
    deadline_ns = (
        epoch_ns
        + max(int(row["arrival_offset_ns"]) for row in requests)
        + 600_000_000_000
    )
    case = _canonical_case_identity(trace_case, arm)
    try:
        while pending or not engine.is_finished():
            now_ns = clock_ns()
            if now_ns >= deadline_ns:
                raise TimeoutError("canonical case exceeded drain timeout")
            while (
                pending
                and epoch_ns + int(
                    pending[0][1]["arrival_offset_ns"]
                ) <= now_ns
            ):
                request_index, request = pending.popleft()
                prompt = _qualification_prompt_tokens(
                    source_commit=source_commit,
                    workload=str(trace_case["workload"]),
                    ordinal=(
                        int(trace_case["repetition"])
                        * REQUESTS_PER_REPETITION
                        + request_index
                    ),
                    prompt_tokens=int(request["prompt_tokens"]),
                )
                prompt_sha = hashlib.sha256(
                    _canonical_json_bytes(prompt).rstrip(b"\n")
                ).hexdigest()
                if prompt_sha != request["prompt_sha256"]:
                    raise ValueError("canonical prompt identity mismatch")
                sequence_id = engine.add_request(
                    prompt,
                    sampling_params_factory(
                        temperature=0.0,
                        max_tokens=int(
                            request["maximum_output_tokens"]
                        ),
                        ignore_eos=bool(request["ignore_eos"]),
                    ),
                    arrival_ns=(
                        epoch_ns + int(request["arrival_offset_ns"])
                    ),
                )
                request_by_sequence[sequence_id] = request
                now_ns = clock_ns()

            if engine.is_finished():
                if pending:
                    wait_ns = max(
                        0,
                        epoch_ns
                        + int(pending[0][1]["arrival_offset_ns"])
                        - clock_ns(),
                    )
                    if wait_ns:
                        sleep(min(wait_ns / 1_000_000_000.0, 0.001))
                continue

            engine.step(completion_only=True)
            observation = dict(engine.last_step_observation or {})
            if observation.get("slo_cohort_telemetry_error") is not None:
                raise RuntimeError(
                    "canonical SLO telemetry failed: "
                    + str(observation["slo_cohort_telemetry_error"])
                )
            memory = observation.get("memory")
            if isinstance(memory, Mapping):
                peak_reserved_bytes = max(
                    peak_reserved_bytes,
                    int(memory.get("cuda_reserved_bytes", 0)),
                    int(memory.get("cuda_peak_reserved_bytes", 0)),
                )
            if tap is not None:
                captured = tap.take(case=case, observation=observation)
                if captured is not None:
                    decision_row, execution_row = captured
                    decision_rows.append(decision_row)
                    if execution_row is not None:
                        execution_rows.append(execution_row)
            terminal_rows = observation.get(
                "slo_cohort_request_telemetry"
            )
            if not isinstance(terminal_rows, list):
                raise RuntimeError(
                    "canonical request telemetry is unavailable"
                )
            for request_row in terminal_rows:
                sequence_id = request_row.get("sequence_id")
                frozen = request_by_sequence.get(sequence_id)
                if frozen is None or sequence_id in completed:
                    raise RuntimeError(
                        "canonical request lifecycle identity mismatch"
                    )
                normalized = dict(request_row)
                normalized["request_id"] = frozen["request_id"]
                completed[sequence_id] = {
                    "schema_version": (
                        "slo-cohort-burst.request-evidence.v1"
                    ),
                    "case": dict(case),
                    "prompt_sha256": frozen["prompt_sha256"],
                    "maximum_output_tokens": frozen[
                        "maximum_output_tokens"
                    ],
                    "ignore_eos": frozen["ignore_eos"],
                    "peak_cuda_reserved_bytes": 0,
                    "request": normalized,
                }
    finally:
        if tap is not None:
            tap.close()
    if set(completed) != set(request_by_sequence):
        raise RuntimeError("canonical request evidence is incomplete")
    for wrapper in completed.values():
        wrapper["peak_cuda_reserved_bytes"] = peak_reserved_bytes
    return {
        "request_rows": list(completed.values()),
        "decision_rows": decision_rows,
        "execution_rows": execution_rows,
    }


def _release_qualification_engine(engine):
    del engine
    return None


def _collect_qualification_engine_memory() -> None:
    gc.collect()
    try:
        torch = __import__("torch")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except (ImportError, AttributeError):
        pass


def _run_canonical_matrix_with_engine_factory(
    *,
    engine_factory,
    sampling_params_factory,
    source_commit: str,
    arrival_traces: Mapping[str, object],
    expected_graph_identities: Mapping[str, str],
) -> dict[str, list[dict]]:
    active_key = None
    active_engine = None

    def run_case(*, trace_case, arm):
        nonlocal active_key, active_engine
        key = (trace_case["repetition"], arm)
        if key != active_key:
            if active_engine is not None:
                active_engine = _release_qualification_engine(
                    active_engine
                )
                _collect_qualification_engine_memory()
            active_engine = engine_factory(arm=arm)
            active_key = key
            _set_cohort_arm(
                active_engine,
                enabled=arm == "candidate",
            )
            if (
                arm == "candidate"
                and _graph_identity_by_batch(active_engine)
                != dict(expected_graph_identities)
            ):
                raise RuntimeError(
                    "candidate graph identity changed across repetitions"
                )
        return _run_open_loop_case(
            engine=active_engine,
            sampling_params_factory=sampling_params_factory,
            source_commit=source_commit,
            trace_case=trace_case,
            arm=arm,
        )

    try:
        return run_canonical_matrix(
            arrival_traces=arrival_traces,
            run_case=run_case,
        )
    finally:
        if active_engine is not None:
            active_engine = _release_qualification_engine(active_engine)
            _collect_qualification_engine_memory()


def _relative_change(baseline: float, candidate: float) -> float:
    if baseline <= 0.0:
        if candidate == baseline:
            return 0.0
        raise ValueError("baseline metric must be positive")
    return (candidate - baseline) / baseline


def _build_canonical_summary(
    *,
    request_rows: list[dict],
    execution_rows: list[dict],
    correctness_rows: list[dict],
) -> dict[str, object]:
    grouped = {}
    for wrapper in request_rows:
        case = wrapper["case"]
        key = (
            case["workload"],
            case["load"],
            case["repetition"],
            case["arm"],
        )
        grouped.setdefault(key, []).append(wrapper)
    paired = {}
    for workload in QUALIFICATION_WORKLOADS:
        for load in QUALIFICATION_LOADS:
            for repetition in range(QUALIFICATION_REPETITIONS):
                identity = (workload, load, repetition)
                paired[identity] = {
                    arm: qualification_gate.summarize_request_rows([
                        wrapper["request"]
                        for wrapper in grouped[(*identity, arm)]
                    ])
                    for arm in ("baseline", "candidate")
                }
    improvements = {
        key: _relative_change(
            value["baseline"]["output_throughput_tps"],
            value["candidate"]["output_throughput_tps"],
        )
        for key, value in paired.items()
    }
    regressions = {
        metric: [
            max(
                0.0,
                _relative_change(
                    value["baseline"][metric],
                    value["candidate"][metric],
                ),
            )
            for value in paired.values()
        ]
        for metric in ("p99_itl_ns", "p99_ttft_ns", "p99_e2e_ns")
    }
    memory_regressions = [
        max(
            0.0,
            _relative_change(
                max(
                    row["peak_cuda_reserved_bytes"]
                    for row in grouped[(*key, "baseline")]
                ),
                max(
                    row["peak_cuda_reserved_bytes"]
                    for row in grouped[(*key, "candidate")]
                ),
            ),
        )
        for key in paired
    ]
    wasted = sum(
        row["execution"]["post_eos_wasted_forwards"]
        for row in execution_rows
    )
    total_slots = sum(
        row["execution"]["completed_replay_count"]
        * len(row["execution"]["generated_token_counts"])
        for row in execution_rows
    )
    correctness_passed = all(
        all(
            row["baseline_output_token_ids"]
            == row["candidate_output_token_ids"]
            and row["baseline_output_text_sha256"]
            == row["candidate_output_text_sha256"]
            and row["baseline_sampled_logits_sha256"]
            == row["candidate_sampled_logits_sha256"]
            and row["baseline_argmax_token_ids"]
            == row["candidate_argmax_token_ids"]
            == row["baseline_output_token_ids"]
            for row in case["rows"]
        )
        and all(
            case[field] == 0
            for field in (
                "duplicate_forwards",
                "duplicate_commits",
                "unauthorized_kv_publications",
                "pending_leases_after_case",
            )
        )
        for case in correctness_rows
    )
    lifecycle_closed = all(
        row["execution"]["pending_inventory"]
        == {"leases": 0, "transactions": 0}
        and row["execution"]["failure_reason"] is None
        and row["execution"]["rollback_reason"] is None
        and row["execution"]["quarantined"] is False
        for row in execution_rows
    )
    summary = {
        "schema_version": qualification_verify.SUMMARY_SCHEMA_VERSION,
        "evidence_complete": True,
        "source_exact": True,
        "verifier_agreement": True,
        "correctness_passed": correctness_passed,
        "lifecycle_closed": lifecycle_closed,
        "aggregate_throughput_improvement": statistics.mean(
            improvements.values()
        ),
        "medium_throughput_improvement": statistics.mean(
            value
            for key, value in improvements.items()
            if key[1] == "medium"
        ),
        "high_throughput_improvement": statistics.mean(
            value
            for key, value in improvements.items()
            if key[1] == "high"
        ),
        "worst_throughput_regression": max(
            0.0,
            -min(improvements.values()),
        ),
        "worst_p99_itl_regression": max(regressions["p99_itl_ns"]),
        "worst_p99_ttft_regression": max(regressions["p99_ttft_ns"]),
        "worst_p99_e2e_regression": max(regressions["p99_e2e_ns"]),
        "maximum_host_visible_gap_ns": int(max(
            value["candidate"]["maximum_host_visible_gap_ns"]
            for value in paired.values()
        )),
        "starved_requests": int(sum(
            value["candidate"]["starved_requests"]
            for value in paired.values()
        )),
        "post_eos_wasted_forward_fraction": (
            wasted / total_slots if total_slots else 0.0
        ),
        "peak_reserved_memory_regression": max(memory_regressions),
    }
    classification_input = {
        key: value
        for key, value in summary.items()
        if key != "schema_version"
    }
    summary["classification"] = (
        qualification_gate.classify_slo_cohort_burst(
            classification_input
        )
    )
    return summary


def stage_artifact_contract(stage: str) -> dict[str, object]:
    if stage == "ceiling":
        required = REQUIRED_TERMINAL_FILES
        compact = COMPACT_FILES
        local_root = LOCAL_ARTIFACT_ROOT
    elif stage == "correctness":
        required = CORRECTNESS_TERMINAL_FILES
        compact = required | {"runner.log"}
        local_root = QUALIFICATION_LOCAL_ARTIFACT_ROOT
    elif stage == "canonical":
        required = CANONICAL_TERMINAL_FILES
        compact = required | {"runner.log"}
        local_root = QUALIFICATION_LOCAL_ARTIFACT_ROOT
    else:
        raise ValueError("unsupported remote stage")
    return {
        "stage": stage,
        "required": frozenset(required),
        "compact": frozenset(compact),
        "local_root": Path(local_root),
    }


def build_remote_paths(run_tag: str) -> dict[str, str]:
    tag = base.validate_run_tag(run_tag)
    paths = {
        "staging": f"{TASK_REMOTE_ROOT}/staging/{tag}",
        "primary": f"{TASK_REMOTE_ROOT}/runs/{tag}",
        "controller": (
            f"{TASK_REMOTE_ROOT}/controller-verification/{tag}"
        ),
    }
    if any(
        not path.startswith(TASK_REMOTE_ROOT + "/")
        for path in paths.values()
    ):
        raise ValueError("remote path is outside approved task root")
    return paths


def validate_qualification_output_dir(
    value: object,
    *,
    run_tag: str,
) -> Path:
    expected = build_remote_paths(run_tag)["primary"]
    if not isinstance(value, (str, os.PathLike)):
        raise ValueError("qualification output path is invalid")
    candidate = os.fspath(value)
    if candidate != expected:
        raise ValueError(
            "qualification output must equal the immutable run path"
        )
    return Path(candidate)


def distributed_port(run_tag: str) -> int:
    tag = base.validate_run_tag(run_tag)
    digest = hashlib.sha256(tag.encode("utf-8")).digest()
    return 20_000 + int.from_bytes(digest[:4], "big") % 30_000


def build_remote_runtime_prelude(
    *,
    source: str,
    gpu_index: int,
    dist_port: int,
) -> str:
    if (
        not isinstance(source, str)
        or not source.startswith(TASK_REMOTE_ROOT + "/staging/")
        or not source.endswith("/source")
    ):
        raise ValueError("remote source path is invalid")
    if (
        isinstance(gpu_index, bool)
        or not isinstance(gpu_index, int)
        or gpu_index < 0
    ):
        raise ValueError("GPU index is invalid")
    if (
        isinstance(dist_port, bool)
        or not isinstance(dist_port, int)
        or not 20_000 <= dist_port < 50_000
    ):
        raise ValueError("distributed port is invalid")
    runtime = source.rsplit("/", 1)[0] + "/runtime"
    directories = {
        "TMPDIR": runtime + "/tmp",
        "TMP": runtime + "/tmp",
        "TEMP": runtime + "/tmp",
        "PYTHONPYCACHEPREFIX": runtime + "/pycache",
        "XDG_CACHE_HOME": runtime + "/xdg",
        "HF_HOME": runtime + "/hf-home",
        "TORCH_EXTENSIONS_DIR": runtime + "/torch-extensions",
    }
    exports = {
        **directories,
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_VISIBLE_DEVICES": str(gpu_index),
        "TINYVLLM_DIST_PORT": str(dist_port),
        "MASTER_PORT": str(dist_port),
        "PYTHONPATH": source,
    }
    return (
        "umask 077; mkdir -p "
        + " ".join(
            shlex.quote(path)
            for path in sorted(set(directories.values()))
        )
        + "; "
        + " ".join(
            f"export {name}={shlex.quote(value)};"
            for name, value in exports.items()
        )
        + " "
    )


def validate_source_commit(requested: str, *, pushed_head: str) -> str:
    if (
        not isinstance(requested, str)
        or re.fullmatch(r"[0-9a-f]{40}", requested) is None
        or not isinstance(pushed_head, str)
        or re.fullmatch(r"[0-9a-f]{40}", pushed_head) is None
    ):
        raise ValueError("source commit is invalid")
    if requested != pushed_head:
        raise ValueError("source commit does not match pushed head")
    return requested


def strict_clean_a100s(rows: list[dict]) -> list[dict]:
    return [
        row
        for row in base.strict_clean_gpus(rows)
        if "A100" in row["name"]
    ]


def wait_for_clean_a100(
    *,
    timeout_seconds: int,
    poll_interval_seconds: int,
) -> tuple[list[dict], dict]:
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, int)
        or timeout_seconds <= 0
        or isinstance(poll_interval_seconds, bool)
        or not isinstance(poll_interval_seconds, int)
        or poll_interval_seconds <= 0
    ):
        raise ValueError("GPU polling policy is invalid")
    deadline = time.monotonic() + timeout_seconds
    while True:
        validate_kerberos(
            minimum_lifetime_seconds=(
                MINIMUM_KERBEROS_LIFETIME_SECONDS
            )
        )
        try:
            rows = base.query_remote_gpu_rows()
        except RuntimeError as error:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "remote GPU inventory remained unavailable"
                ) from error
            time.sleep(poll_interval_seconds)
            continue
        clean = strict_clean_a100s(rows)
        if clean:
            return rows, clean[0]
        if time.monotonic() >= deadline:
            raise TimeoutError("no strict-clean A100 became available")
        time.sleep(poll_interval_seconds)


def validate_selected_gpu_still_clean(selected: dict) -> dict:
    observed = {
        row["index"]: row
        for row in base.query_remote_gpu_rows()
    }.get(selected.get("index"))
    if (
        observed is None
        or observed.get("uuid") != selected.get("uuid")
        or strict_clean_a100s([observed]) != [observed]
    ):
        raise RuntimeError("selected A100 is no longer strict-clean")
    return observed


def committed_source_archive(
    repo_root: Path,
    source_commit: str,
) -> bytes:
    if re.fullmatch(r"[0-9a-f]{40}", source_commit or "") is None:
        raise ValueError("source commit is invalid")
    result = subprocess.run(
        [
            "git",
            "archive",
            "--format=tar",
            "--prefix=source/",
            source_commit,
            "--",
            *SOURCE_FILES,
        ],
        cwd=Path(repo_root),
        capture_output=True,
        check=False,
    )
    base._require_success(result, "build committed source archive")
    if not isinstance(result.stdout, bytes) or not result.stdout:
        raise ValueError("committed source archive is empty")
    with tarfile.open(fileobj=io.BytesIO(result.stdout), mode="r:") as bundle:
        members = bundle.getmembers()
    if not members:
        raise ValueError("committed source archive is empty")
    for member in members:
        path = PurePosixPath(member.name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or not path.parts
            or path.parts[0] != "source"
            or member.issym()
            or member.islnk()
        ):
            raise ValueError("committed source archive is unsafe")
    return result.stdout


def upload_source_archive(*, staging: str, archive: bytes) -> str:
    prefix = TASK_REMOTE_ROOT + "/staging/"
    if (
        not isinstance(staging, str)
        or not staging.startswith(prefix)
        or base.validate_run_tag(staging[len(prefix):])
        != staging[len(prefix):]
    ):
        raise ValueError("remote staging path is invalid")
    if not isinstance(archive, bytes) or not archive:
        raise ValueError("source archive is empty")
    script = "\n".join((
        "import pathlib,sys,tarfile",
        f"staging=pathlib.Path({staging!r})",
        "staging.parent.mkdir(parents=True,exist_ok=True)",
        "staging.mkdir(parents=False,exist_ok=False)",
        "archive_path=staging/'source.tar'",
        "archive_path.write_bytes(sys.stdin.buffer.read())",
        "with tarfile.open(archive_path,'r:') as bundle:",
        " members=bundle.getmembers()",
        " if not members:",
        "  raise ValueError('empty source archive')",
        " for member in members:",
        "  path=pathlib.PurePosixPath(member.name)",
        "  if (path.is_absolute() or '..' in path.parts",
        "      or not path.parts or path.parts[0]!='source'",
        "      or member.issym() or member.islnk()):",
        "   raise ValueError('unsafe source archive member')",
        " bundle.extractall(staging)",
    ))
    result = base._run_remote_with_input(
        "python3 -c " + shlex.quote(script),
        archive,
    )
    base._require_success(result, "upload committed source archive")
    return staging + "/source"


def _runtime_environment(
    *,
    source: str,
    primary: str,
    gpu_index: int,
    run_tag: str,
) -> str:
    prelude = build_remote_runtime_prelude(
        source=source,
        gpu_index=gpu_index,
        dist_port=distributed_port(run_tag),
    )
    return f"cd {shlex.quote(source)} && {prelude}"


def build_worker_plan(
    *,
    paths: dict[str, str],
    run_tag: str,
    source_commit: str,
    gpu: dict,
    stage: str = "ceiling",
) -> dict[str, object]:
    if paths != build_remote_paths(run_tag):
        raise ValueError("remote path inventory is invalid")
    validate_source_commit(
        source_commit,
        pushed_head=source_commit,
    )
    clean = strict_clean_a100s([gpu])
    if clean != [gpu]:
        raise ValueError("selected GPU is not strict-clean A100")
    source = paths["staging"] + "/source"
    contract = stage_artifact_contract(stage)
    prefix = _runtime_environment(
        source=source,
        primary=paths["primary"],
        gpu_index=gpu["index"],
        run_tag=run_tag,
    )
    if stage == "ceiling":
        run_module = "tools.profile_slo_cohort_burst_ceiling"
        run_arguments = (
            " --mode run"
            + " --model "
            + shlex.quote(MODEL_PATH)
            + " --run-tag "
            + shlex.quote(run_tag)
            + " --source-commit "
            + source_commit
            + " --output-dir "
            + shlex.quote(paths["primary"])
        )
        verify_module = "tools.profile_slo_cohort_burst_ceiling"
        verify_arguments = (
            " --mode verify"
            + " --artifact-dir "
            + shlex.quote(paths["primary"])
            + " --output "
            + shlex.quote(
                paths["controller"] + "/remote_verify.json"
            )
        )
    else:
        run_module = "tools.run_slo_cohort_burst_remote"
        run_arguments = (
            " --worker-stage "
            + stage
            + " --model "
            + shlex.quote(MODEL_PATH)
            + " --tag "
            + shlex.quote(run_tag)
            + " --source-commit "
            + source_commit
            + " --output-dir "
            + shlex.quote(paths["primary"])
        )
        verify_module = "tools.slo_cohort_burst_verify"
        verify_arguments = (
            " "
            + shlex.quote(paths["primary"] + "/final_bundle")
            + " --stage "
            + stage
            + " --source-root "
            + shlex.quote(source)
            + " --output "
            + shlex.quote(
                paths["controller"] + "/remote_verify.json"
            )
        )
    runtime_log = paths["staging"] + "/runtime/runner.log"
    run_command = (
        prefix
        + shlex.quote(REMOTE_PYTHON)
        + " -m "
        + run_module
        + run_arguments
        + " > "
        + shlex.quote(runtime_log)
        + " 2>&1"
        + " && mv "
        + shlex.quote(runtime_log)
        + " "
        + shlex.quote(paths["primary"] + "/runner.log")
    )
    verify_command = (
        prefix
        + shlex.quote(REMOTE_PYTHON)
        + " -m "
        + verify_module
        + verify_arguments
    )
    required_files = sorted(contract["required"])
    seal_script = "\n".join((
        "import hashlib,json,os,pathlib",
        f"primary=pathlib.Path({paths['primary']!r})",
        f"controller=pathlib.Path({paths['controller']!r})",
        f"stage={stage!r}",
        f"required={required_files!r}",
        "controller_verify=json.loads(",
        " (controller/'remote_verify.json').read_text())",
        "if stage=='ceiling':",
        " primary_verify=json.loads(",
        "  (primary/'remote_verify.json').read_text())",
        " if primary_verify != controller_verify:",
        "  raise ValueError('remote verifier disagreement')",
        "else:",
        " target=primary/'final_bundle'/'remote_verify.json'",
        " with target.open('x',encoding='utf-8') as handle:",
        "  json.dump(controller_verify,handle,sort_keys=True,",
        "   separators=(',',':'),allow_nan=False)",
        "  handle.write('\\n')",
        "  handle.flush()",
        "  os.fsync(handle.fileno())",
        "hashes={}",
        "for name in required:",
        " path=primary/name",
        " if not path.is_file() or path.is_symlink():",
        "  raise ValueError('terminal artifact missing: '+name)",
        " hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()",
        "receipt={",
        " 'schema_version':'slo-cohort-burst.remote-resume.v1',",
        " 'status':'COMPLETE',",
        " 'stage':stage,",
        f" 'run_tag':{run_tag!r},",
        f" 'source_commit':{source_commit!r},",
        f" 'remote_paths':{paths!r},",
        " 'artifact_sha256':hashes,",
        "}",
        "controller.mkdir(parents=True,exist_ok=True)",
        "destination=controller/'resume.json'",
        "with destination.open('x',encoding='utf-8') as handle:",
        " json.dump(receipt,handle,sort_keys=True,separators=(',',':'))",
        " handle.write('\\n')",
        " handle.flush()",
        " os.fsync(handle.fileno())",
    ))
    seal_command = (
        shlex.quote(REMOTE_PYTHON)
        + " -c "
        + shlex.quote(seal_script)
    )
    return {
        "schema_version": "slo-cohort-burst.worker-plan.v1",
        "stage": stage,
        "run_tag": run_tag,
        "source_commit": source_commit,
        "gpu": dict(gpu),
        "paths": dict(paths),
        "commands": [run_command, verify_command, seal_command],
    }


def run_worker_plan(plan: dict) -> dict[str, object]:
    commands = plan.get("commands")
    if not isinstance(commands, list) or not commands:
        raise ValueError("worker plan commands are invalid")
    exitcodes = []
    for index, command in enumerate(commands):
        if not isinstance(command, str) or not command:
            raise ValueError("worker command is invalid")
        result = base._run_remote(command)
        exitcodes.append(result.returncode)
        base._require_success(result, f"remote worker stage {index}")
    return {
        "status": "COMPLETE",
        "exitcodes": exitcodes,
    }


def is_compact_artifact(
    relative: str,
    *,
    stage: str = "ceiling",
) -> bool:
    if not isinstance(relative, str) or not relative:
        raise ValueError("artifact path is invalid")
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("artifact path is invalid")
    return path.as_posix() in stage_artifact_contract(stage)["compact"]


def validate_resume_receipt(
    receipt: object,
    *,
    run_tag: str,
    source_commit: str,
    paths: dict[str, str],
    stage: str = "ceiling",
) -> dict[str, object]:
    stage_artifact_contract(stage)
    tag = base.validate_run_tag(run_tag)
    validate_source_commit(
        source_commit,
        pushed_head=source_commit,
    )
    if not isinstance(receipt, dict):
        raise ValueError("resume receipt is invalid")
    if (
        receipt.get("schema_version")
        != "slo-cohort-burst.remote-resume.v1"
        or receipt.get("status") != "COMPLETE"
        or receipt.get("stage") != stage
        or receipt.get("run_tag") != tag
        or receipt.get("source_commit") != source_commit
        or receipt.get("remote_paths") != paths
    ):
        raise ValueError("resume source identity is invalid")
    hashes = receipt.get("artifact_sha256")
    if (
        not isinstance(hashes, dict)
        or set(hashes)
        != set(stage_artifact_contract(stage)["required"])
        or any(
            re.fullmatch(r"[0-9a-f]{64}", str(value)) is None
            for value in hashes.values()
        )
    ):
        raise ValueError("resume terminal hashes are invalid")
    return json.loads(json.dumps(receipt))


def probe_resume_receipt(
    *,
    paths: dict[str, str],
    run_tag: str,
    source_commit: str,
    stage: str = "ceiling",
) -> dict[str, object] | None:
    if paths != build_remote_paths(run_tag):
        raise ValueError("remote path inventory is invalid")
    script = "\n".join((
        "import json,pathlib",
        f"paths={paths!r}",
        "present={name:pathlib.Path(path).exists()",
        " for name,path in paths.items()}",
        "if not any(present.values()):",
        " print(json.dumps({'state':'ABSENT'}))",
        "elif not all(present.values()):",
        " raise ValueError('partial immutable remote attempt exists')",
        "else:",
        " receipt_path=pathlib.Path(paths['controller'])/'resume.json'",
        " if not receipt_path.is_file():",
        "  raise ValueError('remote attempt is not resumable')",
        " receipt=json.loads(receipt_path.read_text())",
        " print(json.dumps({'state':'COMPLETE','receipt':receipt},",
        "  sort_keys=True,separators=(',',':')))",
    ))
    result = base._run_remote(
        "python3 -c " + shlex.quote(script)
    )
    base._require_success(result, "remote resume probe")
    try:
        payload = json.loads(result.stdout)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("remote resume probe is invalid") from error
    if payload == {"state": "ABSENT"}:
        return None
    if (
        not isinstance(payload, dict)
        or payload.get("state") != "COMPLETE"
    ):
        raise ValueError("remote resume probe is invalid")
    return validate_resume_receipt(
        payload.get("receipt"),
        run_tag=run_tag,
        source_commit=source_commit,
        paths=paths,
        stage=stage,
    )


def _download_inventory_record(
    *,
    remote_root: str,
    record: dict,
    target: Path,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with target.open("xb") as handle:
        for chunk in record["chunks"]:
            last_error = None
            for _attempt in range(DOWNLOAD_RETRIES):
                try:
                    payload = base.download_chunk(
                        remote_root + "/" + record["path"],
                        offset=chunk["offset"],
                        length=chunk["length"],
                        expected_sha256=chunk["sha256"],
                    )
                    break
                except (RuntimeError, ValueError) as error:
                    last_error = error
            else:
                raise RuntimeError(
                    "artifact chunk download failed: "
                    + record["path"]
                ) from last_error
            handle.write(payload)
            digest.update(payload)
        handle.flush()
        os.fsync(handle.fileno())
    if (
        target.stat().st_size != record["size_bytes"]
        or digest.hexdigest() != record["sha256"]
    ):
        raise ValueError("downloaded artifact digest mismatch")


def download_compact_bundle(
    *,
    remote_path: str,
    local_parent: Path,
    stage: str = "ceiling",
) -> Path:
    prefix = TASK_REMOTE_ROOT + "/runs/"
    if (
        not isinstance(remote_path, str)
        or not remote_path.startswith(prefix)
    ):
        raise ValueError("remote artifact path is invalid")
    run_tag = remote_path[len(prefix):]
    base.validate_run_tag(run_tag)
    destination = Path(local_parent) / run_tag
    if destination.exists() or destination.is_symlink():
        raise ValueError("local artifact destination already exists")
    partial = destination.with_name(destination.name + ".partial")
    if partial.exists() or partial.is_symlink():
        raise ValueError("local partial destination already exists")
    inventory = [
        row
        for row in base.fetch_remote_inventory(remote_path)
        if is_compact_artifact(row["path"], stage=stage)
    ]
    names = {row["path"] for row in inventory}
    required = stage_artifact_contract(stage)["required"]
    if not required.issubset(names):
        raise ValueError("remote terminal artifact inventory is incomplete")
    partial.mkdir(parents=True)
    try:
        for record in inventory:
            _download_inventory_record(
                remote_root=remote_path,
                record=record,
                target=partial / record["path"],
            )
        partial.replace(destination)
    except BaseException:
        if partial.is_dir() and not partial.is_symlink():
            shutil.rmtree(partial)
        raise
    return destination


def verify_local_bundle(
    path: Path,
    *,
    stage: str = "ceiling",
) -> dict[str, object]:
    if stage == "ceiling":
        return profile.verify_ceiling_bundle(Path(path))
    return qualification_verify.verify_artifact_directory(
        Path(path) / "final_bundle",
        source_root=REPO_ROOT,
        output=Path(path) / "final_bundle" / "local_verify.json",
        stage=stage,
    )


def validate_dual_verifier_agreement(
    destination: Path,
    local_verification: Mapping[str, object],
    *,
    stage: str,
) -> dict[str, object]:
    remote_path = (
        Path(destination) / "remote_verify.json"
        if stage == "ceiling"
        else Path(destination) / "final_bundle" / "remote_verify.json"
        if stage in ("correctness", "canonical")
        else None
    )
    if remote_path is None:
        raise ValueError("unsupported verifier stage")
    remote_verification = _load_json(remote_path)
    if (
        not isinstance(remote_verification, dict)
        or remote_verification != dict(local_verification)
    ):
        raise ValueError("remote/local verifier disagreement")
    return dict(remote_verification)


def validate_download_against_resume(
    destination: Path,
    receipt: Mapping[str, object],
    *,
    stage: str = "ceiling",
) -> None:
    hashes = receipt.get("artifact_sha256")
    if not isinstance(hashes, dict):
        raise ValueError("resume terminal hashes are invalid")
    for name in stage_artifact_contract(stage)["required"]:
        path = Path(destination) / name
        if (
            not path.is_file()
            or path.is_symlink()
            or hashlib.sha256(path.read_bytes()).hexdigest()
            != hashes.get(name)
        ):
            raise ValueError(
                f"downloaded terminal artifact mismatch: {name}"
            )


def _write_json(path: Path, payload: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.write_text(
        json.dumps(
            payload,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def write_local_controller_receipt(
    *,
    destination: Path,
    source_commit: str,
    paths: dict[str, str],
    resume_receipt: dict[str, object],
    worker: dict[str, object],
    verification: dict[str, object],
    remote_verification: dict[str, object],
    gpu_inventory: list[dict] | None,
    selected_gpu: dict | None,
    resumed: bool,
    stage: str = "ceiling",
) -> Path:
    controller = Path(destination) / "controller"
    controller.mkdir(exist_ok=False)
    receipt_path = controller / "receipt.json"
    _write_json(receipt_path, {
        "schema_version": "slo-cohort-burst.local-controller.v1",
        "status": "COMPLETE",
        "stage": stage,
        "source_commit": source_commit,
        "remote_paths": paths,
        "resume_receipt": resume_receipt,
        "worker": worker,
        "verification": verification,
        "remote_verification": remote_verification,
        "verifier_agreement": True,
        "gpu_inventory": gpu_inventory,
        "selected_gpu": selected_gpu,
        "resumed": resumed,
    })
    return receipt_path


def run_controller(args) -> dict[str, object]:
    stage = args.stage
    stage_artifact_contract(stage)
    tag = base.validate_run_tag(args.tag)
    local_parent = Path(args.local_artifact_root)
    local_destination = local_parent / tag
    if (
        local_destination.exists()
        or local_destination.is_symlink()
    ):
        raise ValueError("local artifact destination already exists")
    pushed_head = require_pushed_head(REPO_ROOT)
    source_commit = validate_source_commit(
        pushed_head
        if args.source_commit is None
        else args.source_commit,
        pushed_head=pushed_head,
    )
    validate_kerberos(
        minimum_lifetime_seconds=(
            MINIMUM_KERBEROS_LIFETIME_SECONDS
        )
    )
    paths = build_remote_paths(tag)
    resume_receipt = probe_resume_receipt(
        paths=paths,
        run_tag=tag,
        source_commit=source_commit,
        stage=stage,
    )
    resumed = resume_receipt is not None
    gpu_inventory = None
    selected_gpu = None
    worker = {"status": "RESUMED"}
    if not resumed:
        archive = committed_source_archive(REPO_ROOT, source_commit)
        source = upload_source_archive(
            staging=paths["staging"],
            archive=archive,
        )
        gpu_inventory, selected_gpu = wait_for_clean_a100(
            timeout_seconds=args.gpu_timeout_seconds,
            poll_interval_seconds=args.poll_interval_seconds,
        )
        validate_kerberos(
            minimum_lifetime_seconds=(
                MINIMUM_KERBEROS_LIFETIME_SECONDS
            )
        )
        selected_gpu = validate_selected_gpu_still_clean(
            selected_gpu
        )
        plan = build_worker_plan(
            paths=paths,
            run_tag=tag,
            source_commit=source_commit,
            gpu=selected_gpu,
            stage=stage,
        )
        plan["source"] = source
        worker = run_worker_plan(plan)
        resume_receipt = probe_resume_receipt(
            paths=paths,
            run_tag=tag,
            source_commit=source_commit,
            stage=stage,
        )
        if resume_receipt is None:
            raise RuntimeError("remote worker did not seal its evidence")
    download_arguments = {
        "remote_path": paths["primary"],
        "local_parent": local_parent,
    }
    if stage != "ceiling":
        download_arguments["stage"] = stage
    destination = download_compact_bundle(**download_arguments)
    validate_download_against_resume(
        destination,
        resume_receipt,
        stage=stage,
    )
    verification = (
        verify_local_bundle(destination)
        if stage == "ceiling"
        else verify_local_bundle(destination, stage=stage)
    )
    remote_verification = validate_dual_verifier_agreement(
        destination,
        verification,
        stage=stage,
    )
    controller_receipt = write_local_controller_receipt(
        destination=destination,
        source_commit=source_commit,
        paths=paths,
        resume_receipt=resume_receipt,
        worker=worker,
        verification=verification,
        remote_verification=remote_verification,
        gpu_inventory=gpu_inventory,
        selected_gpu=selected_gpu,
        resumed=resumed,
        stage=stage,
    )
    return {
        "status": "COMPLETE",
        "stage": stage,
        "run_tag": tag,
        "source_commit": source_commit,
        "remote_paths": paths,
        "local_destination": os.fspath(destination),
        "local_controller_receipt": os.fspath(controller_receipt),
        "resumed": resumed,
        "classification": verification["classification"],
        "verification": verification,
        "remote_verification": remote_verification,
        "verifier_agreement": True,
    }


def run_qualification_worker(args) -> dict[str, object]:
    stage = args.worker_stage
    if stage not in ("correctness", "canonical"):
        raise ValueError("qualification worker stage is invalid")
    source_commit = validate_source_commit(
        args.source_commit,
        pushed_head=args.source_commit,
    )
    output_dir = validate_qualification_output_dir(
        args.output_dir,
        run_tag=args.tag,
    )
    source_identity, cost_rows, cost_table = (
        _run_source_bound_calibration(
            model=args.model,
            source_commit=source_commit,
            output_dir=output_dir,
        )
    )
    cost_table_path = output_dir / "calibration" / "cost_table.json"
    engine = _create_qualification_engine(
        model=args.model,
        cost_table_path=cost_table_path,
        cohort_enabled=True,
    )
    try:
        sampling_params_factory = (
            _qualification_sampling_params_factory()
        )
        _capture_qualification_graphs(engine)
        graph_identities = _graph_identity_by_batch(engine)
        source_manifest = _source_manifest(
            source_identity=source_identity,
        )
        environment = _environment(
            source_identity=source_identity,
            graph_identity_sha256_by_batch=graph_identities,
            eos_token_id=engine.scheduler.eos,
        )
        correctness_rows = _run_correctness_matrix_on_engine(
            engine=engine,
            sampling_params_factory=sampling_params_factory,
            source_commit=source_commit,
        )
    finally:
        engine = _release_qualification_engine(engine)
        _collect_qualification_engine_memory()

    artifacts = {
        "source_manifest.json": source_manifest,
        "environment.json": environment,
        "cost_profile_rows.jsonl": cost_rows,
        "cost_table.json": cost_table,
        "correctness_rows.jsonl": correctness_rows,
    }
    result = {
        "status": "COMPLETE",
        "stage": stage,
        "run_tag": args.tag,
        "source_commit": source_commit,
        "correctness_case_count": len(correctness_rows),
    }
    if stage == "canonical":
        arrival_traces = build_frozen_arrival_traces(
            source_commit=source_commit,
            saturation_rps_by_workload=_saturation_rates(
                cost_table
            ),
        )

        def canonical_engine_factory(*, arm):
            return _create_qualification_engine(
                model=args.model,
                cost_table_path=cost_table_path,
                cohort_enabled=arm == "candidate",
            )

        canonical = _run_canonical_matrix_with_engine_factory(
            engine_factory=canonical_engine_factory,
            sampling_params_factory=sampling_params_factory,
            source_commit=source_commit,
            arrival_traces=arrival_traces,
            expected_graph_identities=graph_identities,
        )
        summary = _build_canonical_summary(
            request_rows=canonical["request_rows"],
            execution_rows=canonical["execution_rows"],
            correctness_rows=correctness_rows,
        )
        artifacts.update({
            "arrival_traces.json": arrival_traces,
            "decision_rows.jsonl": canonical["decision_rows"],
            "execution_rows.jsonl": canonical["execution_rows"],
            "request_rows.jsonl": canonical["request_rows"],
            "summary.json": summary,
        })
        result.update({
            "classification": summary["classification"],
            "request_row_count": len(canonical["request_rows"]),
            "decision_row_count": len(canonical["decision_rows"]),
            "execution_row_count": len(
                canonical["execution_rows"]
            ),
        })
    final_bundle = write_qualification_bundle(
        output_dir=output_dir,
        stage=stage,
        artifacts=artifacts,
    )
    result["final_bundle"] = os.fspath(final_bundle)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run a staged SLO cohort-burst qualification",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--stage",
        choices=("ceiling", "correctness", "canonical"),
    )
    mode.add_argument(
        "--worker-stage",
        choices=("correctness", "canonical"),
    )
    parser.add_argument("--tag", required=True)
    parser.add_argument("--source-commit")
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--local-artifact-root",
        default=None,
    )
    parser.add_argument(
        "--gpu-timeout-seconds",
        type=int,
        default=7_200,
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=int,
        default=15,
    )
    args = parser.parse_args(argv)
    base.validate_run_tag(args.tag)
    selected_stage = args.stage or args.worker_stage
    if args.local_artifact_root is None:
        args.local_artifact_root = os.fspath(
            stage_artifact_contract(selected_stage)["local_root"]
        )
    if args.worker_stage is not None:
        if args.source_commit is None or args.output_dir is None:
            parser.error(
                "worker stage requires --source-commit and --output-dir"
            )
        try:
            validate_qualification_output_dir(
                args.output_dir,
                run_tag=args.tag,
            )
        except ValueError as error:
            parser.error(str(error))
    if (
        args.gpu_timeout_seconds <= 0
        or args.poll_interval_seconds <= 0
    ):
        parser.error("GPU polling values must be positive")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    result = (
        run_qualification_worker(args)
        if args.worker_stage is not None
        else run_controller(args)
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
