from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping


SCHEMA_VERSION = "qwen35.tp4-hybrid-prefix-performance-cache.v1"
PREREQUISITE_SCHEMA_VERSION = (
    "qwen35.tp4-performance-prerequisites.v2"
)
PREREQUISITE_PROVENANCE_SCHEMA_VERSION = (
    "qwen35.tp4-performance-prerequisite-provenance.v1"
)
POLICIES = ("recompute", "exact_restore")
WORKLOADS = (
    "w0_short_control",
    "w1_medium_reuse",
    "w2_long_reuse",
    "w3_batched_fanout",
    "w4_miss_invalidation",
)
WARMUP_REPETITIONS = 1
CORRECTNESS_REPETITIONS = 1
MEASURED_REPETITIONS = 5
WORLD_SIZE = 4
MIN_GPU_FREE_BYTES = 24 * 1024**3
MAX_MODEL_LEN = 4096
TOKEN_ID_UPPER_BOUND = 32000
MODEL_MANIFEST_SHA256 = (
    "3e650a908234771c3cf1ac4e20c4d38f"
    "e69982efedaf4a3e631ad0b14aad7dd0"
)
TP4_ROOT_SOURCE_TREE_SHA256 = (
    "37135279047a569df8e0d26c6e396472"
    "02b27aca758ac8ac135bdab25612f20a"
)
TP4_ROOT_CORRECTNESS_SCHEMA_VERSION = (
    "qwen35.tp4-real-root-logit-correctness.v1"
)
CACHED_CONTINUATION_SCHEMA_VERSION = (
    "qwen35.tp4-cached-continuation-correctness.v1"
)
ENGINE_CORRECTNESS_SCHEMA_VERSION = (
    "qwen35.tp4-engine-model-runner-correctness.v1"
)
TP4_ROOT_CASE_IDS = ("p17", "p65", "synthetic")
ENGINE_CORRECTNESS_SCENARIOS = {
    "construct_and_bind": (0, 0, 0, 0, 0, 0, 0, 0),
    "publish_source": (1, 1, 1, 1, 0, 0, 1, 1),
    "restore_w1": (64, 64, 64, 0, 1, 0, 0, 1),
    "miss_w4_token": (33, 33, 32, 1, 0, 0, 0, 2),
    "miss_w4_stale": (33, 33, 32, 1, 0, 1, 0, 1),
    "miss_w4_clear": (33, 33, 32, 1, 0, 1, 0, 1),
}

THRESHOLDS = {
    "w1_ttft_max_ratio": 0.85,
    "w2_ttft_max_ratio": 0.75,
    "w3_throughput_min_ratio": 1.15,
    "per_repetition_ttft_max_ratio": 1.05,
    "decode_latency_max_ratio": 1.02,
    "initialization_max_ratio": 1.10,
    "control_e2e_max_ratio": 1.05,
    "peak_cuda_reserved_max_ratio": 1.10,
}

WORKLOAD_SPECS = {
    "w0_short_control": {
        "shared_prefix_tokens": 256,
        "suffix_tokens": 32,
        "continuations": 1,
        "generated_tokens": 32,
        "kind": "reuse",
    },
    "w1_medium_reuse": {
        "shared_prefix_tokens": 1024,
        "suffix_tokens": 64,
        "continuations": 4,
        "generated_tokens": 64,
        "kind": "reuse",
    },
    "w2_long_reuse": {
        "shared_prefix_tokens": 3840,
        "suffix_tokens": 64,
        "continuations": 4,
        "generated_tokens": 64,
        "kind": "reuse",
    },
    "w3_batched_fanout": {
        "shared_prefix_tokens": 2048,
        "suffix_tokens": 64,
        "continuations": 8,
        "generated_tokens": 32,
        "kind": "batched_reuse",
    },
    "w4_miss_invalidation": {
        "shared_prefix_tokens": 1024,
        "suffix_tokens": 64,
        "continuations": 3,
        "generated_tokens": 32,
        "kind": "miss_control",
    },
}


def _validate_workload_admission(workload, spec):
    source_seed_tokens = (
        spec["shared_prefix_tokens"]
        + spec["suffix_tokens"]
        + 1
    )
    if source_seed_tokens > MAX_MODEL_LEN:
        raise ValueError(
            f"{workload} source seed exceeds max_model_len: "
            f"{source_seed_tokens} > {MAX_MODEL_LEN}"
        )
    continuation_tokens = (
        spec["shared_prefix_tokens"]
        + spec["suffix_tokens"]
        + spec["generated_tokens"]
    )
    if continuation_tokens > MAX_MODEL_LEN:
        raise ValueError(
            f"{workload} continuation exceeds max_model_len: "
            f"{continuation_tokens} > {MAX_MODEL_LEN}"
        )


def _deterministic_token_ids(seed, count):
    state = seed
    token_ids = []
    for _ in range(count):
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        token_ids.append(1024 + state % (TOKEN_ID_UPPER_BOUND - 1024))
    return token_ids


def _build_workload_payload(workload):
    if workload not in WORKLOAD_SPECS:
        raise ValueError(f"unsupported workload: {workload}")
    spec = dict(WORKLOAD_SPECS[workload])
    _validate_workload_admission(workload, spec)
    workload_index = WORKLOADS.index(workload)
    token_seed = 2026072900 + workload_index
    shared_prefix = _deterministic_token_ids(
        token_seed,
        spec["shared_prefix_tokens"],
    )
    continuations = []
    for continuation_index in range(spec["continuations"]):
        continuations.append({
            "request_index": continuation_index,
            "suffix_token_ids": _deterministic_token_ids(
                token_seed + 100 + continuation_index,
                spec["suffix_tokens"],
            ),
            "prefix_overrides": [],
            "invalidation": {"kind": "none"},
        })
    if workload == "w4_miss_invalidation":
        mismatch_index = spec["shared_prefix_tokens"] // 2
        replacement = (
            shared_prefix[mismatch_index] + 1
        ) % TOKEN_ID_UPPER_BOUND
        if replacement < 1024:
            replacement += 1024
        continuations[0]["prefix_overrides"] = [[
            mismatch_index,
            replacement,
        ]]
        continuations[0]["invalidation"] = {
            "kind": "token_mismatch",
            "prefix_index": mismatch_index,
            "replacement_token_id": replacement,
        }
        continuations[1]["invalidation"] = {
            "kind": "stale_block_generation",
        }
        continuations[2]["invalidation"] = {
            "kind": "cache_clear",
        }
    return {
        "spec": spec,
        "token_seed": token_seed,
        "shared_prefix_token_ids": shared_prefix,
        "source_suffix_token_ids": _deterministic_token_ids(
            token_seed + 1,
            spec["suffix_tokens"],
        ),
        "continuations": continuations,
    }


def workload_payload(workload):
    return json.loads(canonical_json_bytes(
        _build_workload_payload(workload)
    ))


def workload_manifest_payload():
    return {
        "schema_version": SCHEMA_VERSION,
        "workloads": {
            workload: workload_payload(workload)
            for workload in WORKLOADS
        },
    }


def canonical_json_file_sha256(value):
    return hashlib.sha256(
        canonical_json_bytes(value) + b"\n"
    ).hexdigest()


EXCLUDED_CANDIDATES = (
    "int4_state",
    "token_sparse_state",
    "low_rank_state",
    "gist_layer_share",
)

TOP_LEVEL_ARTIFACTS = (
    "correctness_prerequisites.json",
    "workload_manifest.json",
    "source_manifest.json",
    "environment.json",
    "gpu_assignments.json",
    "commands.json",
    "case_rows.jsonl",
    "process_rows.jsonl",
    "logits_manifest.json",
    "worker_logs_manifest.json",
    "summary.json",
    "artifact_manifest.json",
    "independent_verification.json",
    "report.md",
)

ARTIFACT_MANIFEST_HASH_DOMAIN = (
    "correctness_prerequisites.json",
    "workload_manifest.json",
    "source_manifest.json",
    "environment.json",
    "gpu_assignments.json",
    "commands.json",
    "case_rows.jsonl",
    "process_rows.jsonl",
    "logits_manifest.json",
    "worker_logs_manifest.json",
    "summary.json",
)

NESTED_ARTIFACT_DIRECTORIES = (
    "prerequisites",
    "logits",
    "logs",
)

CASE_ROW_FIELDS = (
    "row_id",
    "case_id",
    "policy",
    "workload",
    "phase",
    "repetition",
    "request_id",
    "source_tree_sha256",
    "model_manifest_sha256",
    "workload_manifest_sha256",
    "correctness_prerequisites_sha256",
    "prompt_tokens",
    "reused_kv_tokens",
    "restored_hybrid_state",
    "executed_prefill_tokens",
    "generated_tokens",
    "ttft_ns",
    "e2e_ns",
    "decode_step_ns",
    "output_token_ids",
    "output_token_ids_sha256",
    "final_logits_path",
    "final_logits_sha256",
)

PROCESS_ROW_FIELDS = (
    "case_id",
    "policy",
    "workload",
    "phase",
    "repetition",
    "initialization_ns",
    "cuda_allocated_bytes",
    "cuda_reserved_bytes",
    "cuda_peak_allocated_bytes",
    "cuda_peak_reserved_bytes",
    "kv_capacity_bytes",
    "scheduler_visible_kv_blocks",
    "hybrid_cache_current_entries",
    "hybrid_cache_current_bytes",
    "hybrid_cache_current_logical_bytes",
    "hybrid_cache_deduplicated_bytes",
    "hybrid_cache_peak_entries",
    "hybrid_cache_peak_bytes",
    "hybrid_cache_hits",
    "hybrid_cache_misses",
    "hybrid_cache_evictions",
    "hybrid_cache_validation_failures",
    "hybrid_cache_failed_restores",
)

_PREREQUISITE_NAMES = (
    "tp4_root_logit",
    "cached_continuation",
    "engine_correctness",
)
_PREREQUISITE_ROW_FIELDS = {
    "run_tag",
    "source_tree_sha256",
    "artifact_path",
    "artifact_sha256",
    "independent_verification_path",
    "independent_verification_sha256",
    "provenance_path",
    "provenance_sha256",
    "classification",
}


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    workload: str
    policy: str
    phase: str
    repetition: int


@dataclass(frozen=True)
class PrerequisiteStatus:
    classification: str
    authorized: bool
    reasons: tuple[str, ...]


def canonical_json_bytes(value) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(value) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pair_order(repetition: int) -> tuple[str, str]:
    if (
        isinstance(repetition, bool)
        or not isinstance(repetition, int)
        or repetition < 0
    ):
        raise ValueError("repetition must be a non-negative integer")
    if repetition % 2 == 0:
        return POLICIES
    return tuple(reversed(POLICIES))


def _case_id(
    workload: str,
    policy: str,
    phase: str,
    repetition: int,
) -> str:
    return (
        f"{workload}__{phase}__r{repetition}__{policy}"
    )


def build_case_matrix() -> tuple[BenchmarkCase, ...]:
    cases = []
    phases = (
        ("warmup", WARMUP_REPETITIONS),
        ("correctness", CORRECTNESS_REPETITIONS),
        ("measured", MEASURED_REPETITIONS),
    )
    for workload in WORKLOADS:
        for phase, repetitions in phases:
            for repetition in range(repetitions):
                for policy in pair_order(repetition):
                    cases.append(BenchmarkCase(
                        case_id=_case_id(
                            workload,
                            policy,
                            phase,
                            repetition,
                        ),
                        workload=workload,
                        policy=policy,
                        phase=phase,
                        repetition=repetition,
                    ))
    return tuple(cases)


def _blocked(*reasons: str) -> PrerequisiteStatus:
    return PrerequisiteStatus(
        classification="BLOCKED_CORRECTNESS",
        authorized=False,
        reasons=tuple(reasons),
    )


def _safe_relative_file(root: Path, value, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} path is invalid")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} path is unsafe")
    path = root / relative
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} file is missing")
    return path


def _valid_sha256(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _nonnegative_integer(value) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int)
        and value >= 0
    )


def _validate_root_logit_documents(
    artifact,
    verification,
    source_tree_sha256,
):
    required = {
        "schema_version",
        "run_tag",
        "classification",
        "comparison_policy",
        "tolerance",
        "prompts",
        "reference_process",
        "comparisons",
        "forbidden_counters",
        "claim_boundary",
    }
    if (
        set(artifact) != required
        or artifact["schema_version"]
        != TP4_ROOT_CORRECTNESS_SCHEMA_VERSION
        or artifact["classification"] != "PASS"
        or artifact["comparison_policy"]
        != "registered_logits_strict_allclose"
        or artifact["tolerance"] != {"atol": 2e-5, "rtol": 0.0}
        or [row.get("case_id") for row in artifact["prompts"]]
        != list(TP4_ROOT_CASE_IDS)
        or artifact["reference_process"].get(
            "model_manifest_sha256"
        )
        != MODEL_MANIFEST_SHA256
        or artifact["forbidden_counters"] != {
            "engine": 0,
            "generation": 0,
            "model_runner": 0,
            "sampler": 0,
            "scheduler": 0,
        }
        or not isinstance(artifact["claim_boundary"], str)
        or "no cached decode" not in artifact["claim_boundary"]
    ):
        raise ValueError("tp4_root_logit artifact schema is invalid")
    comparisons = artifact["comparisons"]
    if (
        not isinstance(comparisons, list)
        or [row.get("case_id") for row in comparisons]
        != list(TP4_ROOT_CASE_IDS)
    ):
        raise ValueError("tp4_root_logit comparison schema is invalid")
    for row in comparisons:
        fields = {
            "native_winner_token_id",
            "native_runner_up_token_id",
            "native_winner_margin",
            "official_winner_token_id",
            "official_runner_up_token_id",
            "official_winner_margin",
            "native_topk_token_ids",
            "official_topk_token_ids",
        }
        if not fields.issubset(row):
            raise ValueError(
                "tp4_root_logit comparison schema is invalid"
            )
        if (
            row["native_winner_token_id"]
            != row["official_winner_token_id"]
            or row["official_winner_token_id"]
            not in row["native_topk_token_ids"]
            or row["native_winner_token_id"]
            not in row["official_topk_token_ids"]
            or (
                row["official_winner_margin"] > 0
                and row["native_winner_margin"] <= 0
            )
            or (
                row["official_winner_margin"] == 0
                and (
                    row["native_winner_margin"] != 0
                    or row["native_runner_up_token_id"]
                    != row["official_runner_up_token_id"]
                )
            )
        ):
            raise ValueError(
                "tp4_root_logit artifact does not prove PASS"
            )
    if (
        set(verification)
        != {"classification", "case_ids", "ranks", "checks"}
        or verification["classification"] != "PASS"
        or verification["case_ids"] != list(TP4_ROOT_CASE_IDS)
        or verification["ranks"] != [0, 1, 2, 3]
        or not _nonnegative_integer(verification["checks"])
        or verification["checks"] == 0
        or not _valid_sha256(source_tree_sha256)
    ):
        raise ValueError(
            "tp4_root_logit independent verification schema is invalid"
        )


def _cached_expected_keys():
    return tuple(
        (workload, request_index)
        for workload in WORKLOADS[1:]
        for request_index in range(
            WORKLOAD_SPECS[workload]["continuations"]
        )
    )


def _validate_cached_documents(
    artifact,
    verification,
    source_tree_sha256,
):
    required = {
        "schema_version",
        "classification",
        "model_manifest_sha256",
        "workload_manifest_sha256",
        "rows",
    }
    expected_workload_sha = canonical_json_file_sha256(
        workload_manifest_payload()
    )
    rows = artifact.get("rows")
    if (
        set(artifact) != required
        or artifact["schema_version"]
        != CACHED_CONTINUATION_SCHEMA_VERSION
        or artifact["classification"] != "PASS"
        or artifact["model_manifest_sha256"] != MODEL_MANIFEST_SHA256
        or artifact["workload_manifest_sha256"]
        != expected_workload_sha
        or not isinstance(rows, list)
        or tuple(
            (row.get("workload"), row.get("request_index"))
            for row in rows
        )
        != _cached_expected_keys()
    ):
        raise ValueError(
            "cached_continuation artifact schema is invalid"
        )
    restore_hits = 0
    w4_misses = 0
    for row in rows:
        workload = row["workload"]
        request_index = row["request_index"]
        spec = WORKLOAD_SPECS[workload]
        expected_hit = workload != "w4_miss_invalidation"
        expected_reason = (
            "exact_hit"
            if expected_hit
            else (
                "token_mismatch",
                "stale_block_generation",
                "cache_clear",
            )[request_index]
        )
        if (
            row.get("outcome") != "continuation"
            or row.get("restore_hit") is not expected_hit
            or row.get("restore_reason") != expected_reason
            or row.get("prompt_tokens")
            != spec["shared_prefix_tokens"] + spec["suffix_tokens"]
            or row.get("reused_tokens")
            != (spec["shared_prefix_tokens"] if expected_hit else 0)
            or row.get("executed_prefill_tokens")
            != (
                spec["suffix_tokens"]
                if expected_hit
                else spec["shared_prefix_tokens"] + spec["suffix_tokens"]
            )
            or row.get("output_token_ids")
            != row.get("reference_output_token_ids")
            or len(row.get("output_token_ids", ()))
            != spec["generated_tokens"]
            or row.get("logits_allclose") is not True
            or not isinstance(row.get("logits_max_abs_diff"), (int, float))
            or isinstance(row.get("logits_max_abs_diff"), bool)
            or not math.isfinite(row["logits_max_abs_diff"])
            or row["logits_max_abs_diff"] < 0
            or row["logits_max_abs_diff"] > 2e-5
            or row.get("cache_identity_match") is not True
            or row.get("rank_inventory") != [0, 1, 2, 3]
            or row.get("process_group_destroyed") is not True
            or row.get("owned_children_remaining") != []
        ):
            raise ValueError(
                "cached_continuation artifact does not prove PASS"
            )
        restore_hits += int(expected_hit)
        w4_misses += int(not expected_hit)
    checks = {
        "row_count": len(rows),
        "restore_hits": restore_hits,
        "w4_misses": w4_misses,
    }
    if (
        set(verification)
        != {
            "schema_version",
            "classification",
            "source_tree_sha256",
            "model_manifest_sha256",
            "workload_manifest_sha256",
            "checks",
        }
        or verification["schema_version"]
        != CACHED_CONTINUATION_SCHEMA_VERSION
        or verification["classification"] != "PASS"
        or verification["source_tree_sha256"] != source_tree_sha256
        or verification["model_manifest_sha256"]
        != MODEL_MANIFEST_SHA256
        or verification["workload_manifest_sha256"]
        != expected_workload_sha
        or verification["checks"] != checks
    ):
        raise ValueError(
            "cached_continuation independent verification schema is invalid"
        )


def _validate_engine_documents(
    artifact,
    verification,
    source_tree_sha256,
):
    required = {
        "schema_version",
        "classification",
        "model_manifest_sha256",
        "rows",
    }
    rows = artifact.get("rows")
    if (
        set(artifact) != required
        or artifact["schema_version"] != ENGINE_CORRECTNESS_SCHEMA_VERSION
        or artifact["classification"] != "PASS"
        or artifact["model_manifest_sha256"] != MODEL_MANIFEST_SHA256
        or not isinstance(rows, list)
        or [row.get("scenario") for row in rows]
        != list(ENGINE_CORRECTNESS_SCENARIOS)
    ):
        raise ValueError("engine_correctness artifact schema is invalid")
    restore_hits = 0
    restore_misses = 0
    for row in rows:
        expected = ENGINE_CORRECTNESS_SCENARIOS[row["scenario"]]
        values = (
            row.get("scheduler_steps"),
            row.get("model_runner_calls"),
            len(row.get("output_token_ids", ())),
            row.get("publication_commits"),
            row.get("restore_hits"),
            row.get("restore_misses"),
            row.get("release_events"),
            row.get("cache_entries_after"),
        )
        if (
            values != expected
            or row.get("engine_class")
            != "tinyvllm.engine.llm_engine.LLMEngine"
            or row.get("model_runner_class")
            != "tinyvllm.engine.model_runner.ModelRunner"
            or row.get("rank_inventory") != [0, 1, 2, 3]
            or row.get("ack_ranks") != [1, 2, 3]
            or row.get("output_token_ids")
            != row.get("reference_output_token_ids")
            or row.get("cache_identity_match") is not True
            or row.get("process_group_destroyed") is not True
            or row.get("rank_exit_codes") != [0, 0, 0, 0]
            or row.get("owned_children_remaining") != []
        ):
            raise ValueError(
                "engine_correctness artifact does not prove PASS"
            )
        restore_hits += row["restore_hits"]
        restore_misses += row["restore_misses"]
    checks = {
        "scenario_count": len(rows),
        "restore_hits": restore_hits,
        "restore_misses": restore_misses,
    }
    if (
        set(verification)
        != {
            "schema_version",
            "classification",
            "source_tree_sha256",
            "model_manifest_sha256",
            "checks",
        }
        or verification["schema_version"]
        != ENGINE_CORRECTNESS_SCHEMA_VERSION
        or verification["classification"] != "PASS"
        or verification["source_tree_sha256"] != source_tree_sha256
        or verification["model_manifest_sha256"]
        != MODEL_MANIFEST_SHA256
        or verification["checks"] != checks
    ):
        raise ValueError(
            "engine_correctness independent verification schema is invalid"
        )


def validate_authority_documents(
    name,
    artifact,
    verification,
    source_tree_sha256,
):
    if not isinstance(artifact, dict) or not isinstance(
        verification,
        dict,
    ):
        raise ValueError(f"{name} authority schema is invalid")
    if name == "tp4_root_logit":
        _validate_root_logit_documents(
            artifact,
            verification,
            source_tree_sha256,
        )
    elif name == "cached_continuation":
        _validate_cached_documents(
            artifact,
            verification,
            source_tree_sha256,
        )
    elif name == "engine_correctness":
        _validate_engine_documents(
            artifact,
            verification,
            source_tree_sha256,
        )
    else:
        raise ValueError("authority name is invalid")


def _validate_prerequisite_provenance(
    name,
    provenance,
    *,
    run_tag,
    source_tree_sha256,
):
    required = {
        "schema_version",
        "authority_name",
        "run_tag",
        "binding_kind",
        "source_tree_sha256",
        "model_manifest_sha256",
        "root_logit_receipt_gap",
        "plan_path",
        "plan_sha256",
        "authorization_path",
        "authorization_sha256",
        "receipt_path",
        "receipt_sha256",
    }
    if (
        not isinstance(provenance, dict)
        or set(provenance) != required
        or provenance["schema_version"]
        != PREREQUISITE_PROVENANCE_SCHEMA_VERSION
        or provenance["authority_name"] != name
        or provenance["run_tag"] != run_tag
        or provenance["source_tree_sha256"] != source_tree_sha256
        or provenance["model_manifest_sha256"]
        != MODEL_MANIFEST_SHA256
    ):
        raise ValueError(f"{name} provenance schema is invalid")
    if (
        provenance["binding_kind"] != "remote_execution_receipt"
        or provenance["root_logit_receipt_gap"] is not False
    ):
        raise ValueError(f"{name} receipt provenance is invalid")
    for path_field, sha_field in (
        ("plan_path", "plan_sha256"),
        ("authorization_path", "authorization_sha256"),
        ("receipt_path", "receipt_sha256"),
    ):
        relative = provenance[path_field]
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not _valid_sha256(provenance[sha_field])
        ):
            raise ValueError(f"{name} receipt provenance is invalid")


def validate_prerequisites(path) -> PrerequisiteStatus:
    prerequisite_path = Path(path)
    if not prerequisite_path.is_file():
        return _blocked("correctness prerequisite file is missing")
    try:
        payload = json.loads(
            prerequisite_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        return _blocked(f"correctness prerequisite file is invalid: {error}")
    if not isinstance(payload, dict):
        return _blocked("correctness prerequisite payload is invalid")
    expected_top_level = {
        "schema_version",
        "model_manifest_sha256",
        *_PREREQUISITE_NAMES,
    }
    if set(payload) != expected_top_level:
        return _blocked("correctness prerequisite schema is invalid")
    reasons = []
    if payload.get("schema_version") != PREREQUISITE_SCHEMA_VERSION:
        reasons.append("correctness prerequisite schema version mismatch")
    if payload.get("model_manifest_sha256") != MODEL_MANIFEST_SHA256:
        reasons.append("correctness prerequisite model manifest mismatch")
    root = prerequisite_path.parent
    for name in _PREREQUISITE_NAMES:
        row = payload.get(name)
        if not isinstance(row, dict) or set(row) != _PREREQUISITE_ROW_FIELDS:
            reasons.append(f"{name} prerequisite row is invalid")
            continue
        if row.get("classification") != "PASS":
            reasons.append(f"{name} classification is not PASS")
        source_tree = row.get("source_tree_sha256")
        if not _valid_sha256(source_tree):
            reasons.append(f"{name} source tree SHA is invalid")
        if (
            name == "tp4_root_logit"
            and source_tree != TP4_ROOT_SOURCE_TREE_SHA256
        ):
            reasons.append("root-logit source tree mismatch")
        artifact_document = None
        verification_document = None
        provenance_document = None
        for path_field, sha_field, label in (
            (
                "artifact_path",
                "artifact_sha256",
                f"{name} artifact",
            ),
            (
                "independent_verification_path",
                "independent_verification_sha256",
                f"{name} independent verification",
            ),
            (
                "provenance_path",
                "provenance_sha256",
                f"{name} provenance",
            ),
        ):
            expected_sha = row.get(sha_field)
            if not _valid_sha256(expected_sha):
                reasons.append(f"{label} SHA is invalid")
                continue
            try:
                artifact_path = _safe_relative_file(
                    root,
                    row.get(path_field),
                    label,
                )
            except ValueError as error:
                reasons.append(str(error))
                continue
            if sha256_file(artifact_path) != expected_sha:
                reasons.append(f"{label} SHA mismatch")
                continue
            try:
                artifact_payload = json.loads(
                    artifact_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                reasons.append(f"{label} payload is invalid")
                continue
            if path_field == "artifact_path":
                artifact_document = artifact_payload
            elif path_field == "independent_verification_path":
                verification_document = artifact_payload
            else:
                provenance_document = artifact_payload
        if (
            artifact_document is None
            or verification_document is None
            or provenance_document is None
        ):
            continue
        try:
            validate_authority_documents(
                name,
                artifact_document,
                verification_document,
                source_tree,
            )
            _validate_prerequisite_provenance(
                name,
                provenance_document,
                run_tag=row.get("run_tag"),
                source_tree_sha256=source_tree,
            )
            provenance_root = (
                prerequisite_path.parent / row["provenance_path"]
            ).parent
            for path_field, sha_field, label in (
                ("plan_path", "plan_sha256", "execution plan"),
                (
                    "authorization_path",
                    "authorization_sha256",
                    "consumed authorization",
                ),
                (
                    "receipt_path",
                    "receipt_sha256",
                    "execution receipt",
                ),
            ):
                evidence = _safe_relative_file(
                    provenance_root,
                    provenance_document[path_field],
                    f"{name} {label}",
                )
                if (
                    sha256_file(evidence)
                    != provenance_document[sha_field]
                ):
                    raise ValueError(f"{name} {label} SHA mismatch")
        except ValueError as error:
            reasons.append(str(error))
    if reasons:
        return _blocked(*reasons)
    return PrerequisiteStatus(
        classification="PASS",
        authorized=True,
        reasons=(),
    )


def _finite_ratio(value, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{label} must be a finite non-negative number")
    return float(value)


def classify_run(metrics: Mapping[str, object]) -> str:
    if not isinstance(metrics, Mapping):
        return "INVALID"
    if metrics.get("prerequisites_pass", True) is not True:
        return "BLOCKED_CORRECTNESS"
    eligible_gpu_count = metrics.get("eligible_gpu_count")
    if (
        isinstance(eligible_gpu_count, bool)
        or not isinstance(eligible_gpu_count, int)
        or eligible_gpu_count < WORLD_SIZE
    ):
        return "BLOCKED_RESOURCES"
    if (
        metrics.get("evidence_complete") is not True
        or metrics.get("measured_matrix_complete") is not True
        or metrics.get("correctness_pass") is not True
        or metrics.get("cache_accounting_valid") is not True
    ):
        return "INVALID"
    workloads = metrics.get("workloads")
    if not isinstance(workloads, Mapping) or set(workloads) != set(WORKLOADS):
        return "INVALID"
    try:
        w0 = workloads["w0_short_control"]
        w1 = workloads["w1_medium_reuse"]
        w2 = workloads["w2_long_reuse"]
        w3 = workloads["w3_batched_fanout"]
        w4 = workloads["w4_miss_invalidation"]
        if not all(
            isinstance(row, Mapping)
            for row in (w0, w1, w2, w3, w4)
        ):
            return "INVALID"
        no_go = (
            _finite_ratio(
                w1.get("median_ttft_ratio"),
                "W1 median TTFT ratio",
            ) > THRESHOLDS["w1_ttft_max_ratio"]
            or _finite_ratio(
                w2.get("median_ttft_ratio"),
                "W2 median TTFT ratio",
            ) > THRESHOLDS["w2_ttft_max_ratio"]
            or _finite_ratio(
                w3.get("throughput_ratio"),
                "W3 throughput ratio",
            ) < THRESHOLDS["w3_throughput_min_ratio"]
            or any(
                _finite_ratio(
                    workloads[name].get(
                        "max_repetition_ttft_ratio"
                    ),
                    f"{name} maximum repetition TTFT ratio",
                ) > THRESHOLDS["per_repetition_ttft_max_ratio"]
                for name in (
                    "w1_medium_reuse",
                    "w2_long_reuse",
                    "w3_batched_fanout",
                )
            )
            or any(
                _finite_ratio(
                    workloads[name].get("median_decode_ratio"),
                    f"{name} median decode ratio",
                ) > THRESHOLDS["decode_latency_max_ratio"]
                for name in WORKLOADS
            )
            or any(
                _finite_ratio(
                    workloads[name].get("median_e2e_ratio"),
                    f"{name} median end-to-end ratio",
                ) > THRESHOLDS["control_e2e_max_ratio"]
                for name in (
                    "w0_short_control",
                    "w4_miss_invalidation",
                )
            )
            or _finite_ratio(
                metrics.get("initialization_ratio"),
                "initialization ratio",
            ) > THRESHOLDS["initialization_max_ratio"]
            or _finite_ratio(
                metrics.get("peak_cuda_reserved_ratio"),
                "peak CUDA reserved ratio",
            ) > THRESHOLDS["peak_cuda_reserved_max_ratio"]
            or metrics.get("scheduler_visible_kv_capacity_equal")
            is not True
            or metrics.get("kv_capacity_bytes_equal") is not True
            or metrics.get("cache_within_limits") is not True
            or metrics.get("no_required_workload_evictions") is not True
        )
    except (KeyError, TypeError, ValueError):
        return "INVALID"
    return "NO_GO" if no_go else "GO"
