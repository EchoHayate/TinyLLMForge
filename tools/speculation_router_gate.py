"""Source-auditable profitability gate for routed speculation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import socket
import statistics
import subprocess
import time
from pathlib import Path


OWNED_SOURCE_ROOTS = (
    "tinyvllm",
    "tools/draft_model_schema.py",
    "tools/profile_ngram_commit.py",
    "tools/source_audit.py",
    "tools/speculation_router_gate.py",
    "tools/test_speculation_router.py",
    "tools/test_speculation_router_gate.py",
    "tools/native_verifier_oracle.py",
    "tools/test_native_verifier_oracle.py",
    "tools/run_speculation_router_gate_remote.sh",
)

CONTROLLED_POLICIES = (
    "baseline",
    "legacy_rematerialize",
    "always_native",
    "routed_native",
    "oracle",
)

CONTROLLED_THRESHOLDS = {
    "profitable_region_max_elapsed_ratio": 0.95,
    "max_required_lifecycle_elapsed_ratio": 1.05,
    "min_continuation_steps": 16,
}

REAL_POLICIES = (
    "baseline",
    "source_always_native",
    "source_routed_native",
)

REAL_THRESHOLDS = {
    "min_elapsed_improvement_fraction": 0.05,
    "min_tokens_per_s_improvement_fraction": 0.05,
    "max_natural_elapsed_ratio": 1.00,
    "max_transition_elapsed_ratio": 1.00,
    "max_individual_prompt_elapsed_ratio": 1.10,
}

_REAL_SOURCE_TYPES = (
    "learned_speculative_head",
    "external_draft_model",
)
_NEGATIVE_CONTROL_ADAPTERS = (
    "ngram",
    "sam",
    "dflash-toy",
    "dflash-toy-ngram-or-repeat",
)

_CONTROLLED_PROMPT = (
    "Repeat the sequence alpha beta gamma while preserving exact spacing: "
    "alpha beta gamma alpha beta gamma."
)
_CONTROLLED_EOS_PROMPT = (
    "<|im_start|>user\n"
    "Reply with exactly OK and then stop.<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def _case(
    case_id: str,
    *,
    draft_len: int,
    acceptance_case: str,
    expected_accepted_count: int,
    history_len: int,
    block_case: str,
    eos_case: bool = False,
    output_budget_case: bool = False,
) -> dict:
    return {
        "case_id": case_id,
        "draft_len": draft_len,
        "acceptance_case": acceptance_case,
        "expected_accepted_count": expected_accepted_count,
        "history_len": history_len,
        "block_case": block_case,
        "eos_case": eos_case,
        "output_budget_case": output_budget_case,
        "continuation_steps": 16,
        "draft_construction": "controlled_target_derived",
    }


CONTROLLED_CASE_MATRIX = (
    _case(
        "k1-route-fallback",
        draft_len=1,
        acceptance_case="full",
        expected_accepted_count=1,
        history_len=52,
        block_case="current_block",
    ),
    _case(
        "k2-zero-current",
        draft_len=2,
        acceptance_case="zero",
        expected_accepted_count=0,
        history_len=52,
        block_case="current_block",
    ),
    _case(
        "k2-one-current",
        draft_len=2,
        acceptance_case="one",
        expected_accepted_count=1,
        history_len=52,
        block_case="current_block",
    ),
    _case(
        "k2-full-current",
        draft_len=2,
        acceptance_case="full",
        expected_accepted_count=2,
        history_len=52,
        block_case="current_block",
    ),
    _case(
        "k4-zero-current",
        draft_len=4,
        acceptance_case="zero",
        expected_accepted_count=0,
        history_len=52,
        block_case="current_block",
    ),
    _case(
        "k4-one-current",
        draft_len=4,
        acceptance_case="one",
        expected_accepted_count=1,
        history_len=52,
        block_case="current_block",
    ),
    _case(
        "k4-partial-boundary",
        draft_len=4,
        acceptance_case="partial",
        expected_accepted_count=2,
        history_len=255,
        block_case="one_new_block",
    ),
    _case(
        "k4-full-boundary",
        draft_len=4,
        acceptance_case="full",
        expected_accepted_count=4,
        history_len=255,
        block_case="one_new_block",
    ),
    _case(
        "k8-zero-current",
        draft_len=8,
        acceptance_case="zero",
        expected_accepted_count=0,
        history_len=52,
        block_case="current_block",
    ),
    _case(
        "k8-one-current",
        draft_len=8,
        acceptance_case="one",
        expected_accepted_count=1,
        history_len=52,
        block_case="current_block",
    ),
    _case(
        "k8-partial-boundary",
        draft_len=8,
        acceptance_case="partial",
        expected_accepted_count=4,
        history_len=255,
        block_case="one_new_block",
    ),
    _case(
        "k8-full-boundary",
        draft_len=8,
        acceptance_case="full",
        expected_accepted_count=8,
        history_len=255,
        block_case="one_new_block",
    ),
    _case(
        "k8-eos-boundary",
        draft_len=8,
        acceptance_case="partial",
        expected_accepted_count=2,
        history_len=255,
        block_case="real_eos_history",
        eos_case=True,
    ),
    _case(
        "k8-budget-boundary",
        draft_len=8,
        acceptance_case="full",
        expected_accepted_count=3,
        history_len=255,
        block_case="one_new_block",
        output_budget_case=True,
    ),
    _case(
        "k16-zero-multiblock",
        draft_len=16,
        acceptance_case="zero",
        expected_accepted_count=0,
        history_len=511,
        block_case="multi_block_context",
    ),
    _case(
        "k16-one-multiblock",
        draft_len=16,
        acceptance_case="one",
        expected_accepted_count=1,
        history_len=511,
        block_case="multi_block_context",
    ),
    _case(
        "k16-partial-multiblock",
        draft_len=16,
        acceptance_case="partial",
        expected_accepted_count=8,
        history_len=511,
        block_case="multi_block_context",
    ),
    _case(
        "k16-full-multiblock",
        draft_len=16,
        acceptance_case="full",
        expected_accepted_count=16,
        history_len=511,
        block_case="multi_block_context",
    ),
)


def build_controlled_manifest(
    *,
    source_evidence: dict,
    source_preflight: dict,
    model_path: str,
    model_identifier: str,
    host: str,
    python_bin: str,
    torch_version: str,
    cuda_version: str,
    flash_attn_version: str,
    gpu_name: str,
    bf16_supported: bool,
    run_tag: str,
) -> dict:
    return {
        "schema_version": 1,
        "stage": "controlled",
        "run_tag": run_tag,
        "created_unix_s": time.time(),
        "source_commit": source_evidence["base_commit"],
        "source_dirty": source_evidence["dirty"],
        "source_tree_sha256": source_evidence["tree_sha256"],
        "source_evidence": source_evidence,
        "source_preflight": source_preflight,
        "model_path": model_path,
        "model_identifier": model_identifier,
        "host": host,
        "python_bin": python_bin,
        "torch_version": torch_version,
        "cuda_version": cuda_version,
        "flash_attn_version": flash_attn_version,
        "gpu_name": gpu_name,
        "bf16_supported": bool(bf16_supported),
        "thresholds": CONTROLLED_THRESHOLDS,
        "case_matrix": list(CONTROLLED_CASE_MATRIX),
        "policies": list(CONTROLLED_POLICIES),
        "classification_on_success": (
            "READY_FOR_REAL_DRAFTER_GATE"
        ),
        "process_port_pairs": [],
    }


def _allocate_port_pair() -> tuple[int, int]:
    sockets = []
    ports = []
    try:
        for _ in range(2):
            handle = socket.socket(
                socket.AF_INET,
                socket.SOCK_STREAM,
            )
            handle.bind(("127.0.0.1", 0))
            sockets.append(handle)
            ports.append(int(handle.getsockname()[1]))
    finally:
        for handle in sockets:
            handle.close()
    return ports[0], ports[1]


def _case_process(
    *,
    python_bin: str,
    model_path: str,
    policy: str,
    case: dict,
    out_path: Path,
    log_dir: Path,
) -> tuple[dict | None, dict]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    case_path = out_path.with_suffix(".case.json")
    case_path.write_text(
        json.dumps(case, indent=2, sort_keys=True) + "\n"
    )
    last_process = None
    for attempt in range(1, 4):
        dist_port, master_port = _allocate_port_pair()
        stdout_path = log_dir / (
            f"{case['case_id']}.{policy}.attempt{attempt}.stdout.log"
        )
        stderr_path = log_dir / (
            f"{case['case_id']}.{policy}.attempt{attempt}.stderr.log"
        )
        command = [
            str(python_bin),
            str(Path(__file__).with_name(
                "native_verifier_oracle.py"
            )),
            "run-case",
            "--policy",
            policy,
            "--case-json",
            str(case_path),
            "--out",
            str(out_path),
            "--model",
            str(model_path),
            "--continuation-steps",
            str(case["continuation_steps"]),
        ]
        environment = os.environ.copy()
        environment["TINYVLLM_DIST_PORT"] = str(dist_port)
        environment["MASTER_PORT"] = str(master_port)
        started = time.perf_counter()
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            env=environment,
            check=False,
        )
        stdout_path.write_text(completed.stdout)
        stderr_path.write_text(completed.stderr)
        last_process = {
            "returncode": int(completed.returncode),
            "command": command,
            "tinyvllm_dist_port": dist_port,
            "master_port": master_port,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "elapsed_s": time.perf_counter() - started,
            "attempt": attempt,
        }
        combined = completed.stdout + "\n" + completed.stderr
        retryable = completed.returncode != 0 and (
            "EADDRINUSE" in combined
            or "address already in use" in combined.lower()
        )
        if completed.returncode == 0:
            return json.loads(out_path.read_text()), last_process
        if not retryable:
            break
    return None, last_process or {
        "returncode": 1,
        "tinyvllm_dist_port": -1,
        "master_port": -1,
    }


def _materialize_controlled_case(
    case_spec: dict,
    probe: dict,
    *,
    source_tree_sha256: str,
) -> dict:
    from native_verifier_oracle import construct_draft_tokens

    if (
        case_spec.get("draft_construction")
        != "controlled_target_derived"
    ):
        raise ValueError(
            "controlled case requires "
            "draft_construction=controlled_target_derived"
        )
    targets = [int(token_id) for token_id in probe["target_tokens"]]
    history_len = int(case_spec["history_len"])
    draft_tokens = construct_draft_tokens(
        targets,
        acceptance_case=case_spec["acceptance_case"],
        vocab_size=int(probe["vocab_size"]),
    )
    prompt_token_count = int(probe["prompt_token_count"])
    if case_spec["eos_case"]:
        eos_token_id = int(probe["eos_token_id"])
        history_tokens = [
            int(token_id) for token_id in probe["history_tokens"]
        ]
        draft_len = int(case_spec["draft_len"])
        eos_indices = [
            index
            for index, token_id in enumerate(history_tokens)
            if (
                token_id == eos_token_id
                and index - draft_len + 1 >= prompt_token_count
            )
        ]
        if not eos_indices:
            raise ValueError(
                f"{case_spec['case_id']} probe history has no usable real EOS"
            )
        eos_index = eos_indices[-1]
        history_len = eos_index - draft_len + 1
        draft_tokens = history_tokens[
            history_len:history_len + draft_len
        ]
    completion_at_history = history_len - prompt_token_count
    max_tokens = (
        completion_at_history + 2
        if case_spec["output_budget_case"]
        else completion_at_history
        + len(draft_tokens)
        + int(case_spec["continuation_steps"])
        + 4
    )
    return {
        **case_spec,
        "prompt": (
            _CONTROLLED_EOS_PROMPT
            if case_spec["eos_case"]
            else _CONTROLLED_PROMPT
        ),
        "history_len": history_len,
        "draft_tokens": draft_tokens,
        "max_tokens": max_tokens,
        "ignore_eos": not bool(case_spec["eos_case"]),
        "source_tree_sha256": source_tree_sha256,
    }


_LARGE_EVIDENCE_FIELDS = (
    "logits",
    "kv",
    "continuation_logits",
    "continuation_kv",
    "physical_slots",
    "continuation_physical_slots",
    "event",
    "router_event",
)


def _sha256_json(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def canonical_prompt_bank_sha256(prompt_bank: dict) -> str:
    return _sha256_json(prompt_bank)


def _is_sha256(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_draft_source_manifest(draft_source: dict) -> None:
    required = (
        "schema_version",
        "source_name",
        "source_type",
        "implementation_paths",
        "source_tree_sha256",
        "checkpoint_identifier",
        "checkpoint_config_sha256",
        "tokenizer_identifier",
        "vocab_size",
        "hyperparameters",
        "consumes_target_hidden_states",
        "requires_additional_model_forward",
        "target_derived",
        "debug_stub",
        "prompt_bank_sha256",
    )
    missing = [
        field for field in required if field not in draft_source
    ]
    if missing:
        raise ValueError(
            "draft source manifest is missing: "
            + ", ".join(missing)
        )
    if draft_source["schema_version"] != 1:
        raise ValueError("unsupported draft source schema")
    if (
        not isinstance(draft_source["source_name"], str)
        or not draft_source["source_name"]
    ):
        raise ValueError("draft source name is missing")
    if draft_source["source_type"] not in _REAL_SOURCE_TYPES:
        raise ValueError("unsupported real draft source type")
    implementation_paths = draft_source["implementation_paths"]
    if (
        not isinstance(implementation_paths, list)
        or not implementation_paths
        or any(
            not isinstance(path, str) or not path
            for path in implementation_paths
        )
    ):
        raise ValueError("draft source implementation paths are invalid")
    for field in (
        "source_tree_sha256",
        "checkpoint_config_sha256",
        "prompt_bank_sha256",
    ):
        if not _is_sha256(draft_source[field]):
            raise ValueError(f"invalid draft source {field}")
    for field in (
        "checkpoint_identifier",
        "tokenizer_identifier",
    ):
        if (
            not isinstance(draft_source[field], str)
            or not draft_source[field]
        ):
            raise ValueError(f"draft source {field} is missing")
    if (
        not isinstance(draft_source["vocab_size"], int)
        or draft_source["vocab_size"] <= 1
    ):
        raise ValueError("draft source vocab_size is invalid")
    if not isinstance(draft_source["hyperparameters"], dict):
        raise ValueError("draft source hyperparameters are invalid")
    if draft_source["target_derived"] is not False:
        raise ValueError("real draft source must not be target-derived")
    if draft_source["debug_stub"] is not False:
        raise ValueError("real draft source must not be a debug stub")
    for field in (
        "consumes_target_hidden_states",
        "requires_additional_model_forward",
    ):
        if not isinstance(draft_source[field], bool):
            raise ValueError(f"draft source {field} must be boolean")
    if draft_source.get("runtime_adapter") in (
        _NEGATIVE_CONTROL_ADAPTERS
    ):
        raise ValueError(
            "existing prompt lookup/ngram/SAM sources are "
            "negative controls only"
        )


def validate_real_input(
    draft_source: dict,
    prompt_bank: dict,
) -> dict:
    validate_draft_source_manifest(draft_source)
    prompt_hash = canonical_prompt_bank_sha256(prompt_bank)
    if draft_source["prompt_bank_sha256"] != prompt_hash:
        raise ValueError("draft source prompt bank hash mismatch")
    return {
        "status": "PASS",
        "source_name": draft_source["source_name"],
        "prompt_bank_sha256": prompt_hash,
    }


def _real_row_key(row: dict) -> tuple[str, str]:
    return str(row.get("prompt_id")), str(row.get("policy"))


def classify_real_source_gate(
    manifest: dict,
    draft_source: dict,
    prompt_bank: dict,
    case_rows: list[dict],
    event_rows: list[dict],
    router_rows: list[dict],
) -> dict:
    structural = []
    try:
        validate_draft_source_manifest(draft_source)
    except ValueError as exc:
        structural.append(str(exc))
    prompt_hash = canonical_prompt_bank_sha256(prompt_bank)
    draft_source_hash = _sha256_json(draft_source)
    if manifest.get("stage") != "real-source":
        structural.append("manifest stage mismatch")
    if manifest.get("policies") != list(REAL_POLICIES):
        structural.append("manifest real policies drift")
    if manifest.get("thresholds") != REAL_THRESHOLDS:
        structural.append("manifest real thresholds drift")
    if manifest.get("draft_source_sha256") != draft_source_hash:
        structural.append("draft source identity mismatch")
    if manifest.get("prompt_bank_sha256") != prompt_hash:
        structural.append("prompt bank identity mismatch")
    if draft_source.get("prompt_bank_sha256") != prompt_hash:
        structural.append("draft source prompt bank mismatch")
    if (
        draft_source.get("source_tree_sha256")
        != manifest.get("source_tree_sha256")
    ):
        structural.append("draft source tree mismatch")
    prompts = prompt_bank.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        structural.append("prompt bank prompts are missing")
        prompts = []
    prompt_ids = [str(prompt.get("prompt_id")) for prompt in prompts]
    if (
        len(prompt_ids) != len(set(prompt_ids))
        or any(not prompt_id for prompt_id in prompt_ids)
    ):
        structural.append("prompt bank prompt ids are invalid")
    expected_keys = {
        (prompt_id, policy)
        for prompt_id in prompt_ids
        for policy in REAL_POLICIES
    }
    observed_keys = [_real_row_key(row) for row in case_rows]
    if (
        len(observed_keys) != len(set(observed_keys))
        or set(observed_keys) != expected_keys
    ):
        structural.append("missing or duplicate real-source rows")
    hyperparameters_sha256 = _sha256_json(
        draft_source.get("hyperparameters")
    )
    process_port_pairs = []
    for row in case_rows:
        key = _real_row_key(row)
        process = row.get("process")
        if (
            not isinstance(process, dict)
            or process.get("returncode") != 0
        ):
            structural.append(f"{key} process failed")
        elif any(
            not isinstance(process.get(field), int)
            for field in (
                "tinyvllm_dist_port",
                "master_port",
            )
        ):
            structural.append(f"{key} missing dynamic ports")
        else:
            process_port_pairs.append((
                process["tinyvllm_dist_port"],
                process["master_port"],
            ))
        for field, expected in (
            (
                "source_tree_sha256",
                manifest.get("source_tree_sha256"),
            ),
            ("draft_source_sha256", draft_source_hash),
            ("prompt_bank_sha256", prompt_hash),
            (
                "hyperparameters_sha256",
                hyperparameters_sha256,
            ),
        ):
            if row.get(field) != expected:
                structural.append(f"{key} {field} mismatch")
        for field in (
            "elapsed_s",
            "output_tokens",
            "output_tokens_per_s",
        ):
            value = row.get(field)
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0
            ):
                structural.append(f"{key} invalid {field}")
    if len(process_port_pairs) != len(set(process_port_pairs)):
        structural.append("duplicate real-source dynamic port pairs")
    expected_event_ids = set(prompt_ids)
    event_ids = [
        str(row.get("prompt_id")) for row in event_rows
        if row.get("policy") == "source_routed_native"
    ]
    if (
        len(event_ids) != len(set(event_ids))
        or set(event_ids) != expected_event_ids
    ):
        structural.append("missing or duplicate real-source events")
    router_ids = [
        str(row.get("prompt_id")) for row in router_rows
        if row.get("policy") == "source_routed_native"
    ]
    if (
        len(router_ids) != len(set(router_ids))
        or set(router_ids) != expected_event_ids
    ):
        structural.append("missing or duplicate real-source routes")
    if structural:
        return _incomplete(structural)

    by_key = {_real_row_key(row): row for row in case_rows}
    semantic = []
    for prompt_id in prompt_ids:
        baseline = by_key[(prompt_id, "baseline")]
        output_hash = baseline.get("output_token_sha256")
        if not output_hash:
            return _incomplete([
                f"{prompt_id} baseline output hash is missing",
            ])
        for policy in REAL_POLICIES:
            row = by_key[(prompt_id, policy)]
            if row.get("status") == "INCOMPLETE":
                return _incomplete([
                    f"{prompt_id}/{policy} row is INCOMPLETE",
                ])
            if row.get("status") != "PASS":
                semantic.append(
                    f"{prompt_id}/{policy} semantic failure"
                )
            if row.get("output_token_sha256") != output_hash:
                semantic.append(
                    f"{prompt_id}/{policy} output mismatch"
                )
    for event in event_rows:
        rematerialization = event.get(
            "accepted_kv_rematerialization",
            {},
        )
        if (
            rematerialization.get("decode_calls") != 0
            or rematerialization.get("rematerialized_tokens")
            or float(rematerialization.get("elapsed_ms", math.nan))
            != 0.0
            or event.get("accepted_kv_copy_calls") != 0
            or event.get("accepted_kv_replay_calls") != 0
        ):
            semantic.append(
                f"{event['prompt_id']} replay/copy/rematerialization"
            )
        if (
            int(event.get("baseline_target_forward_count", 0))
            - int(event.get("target_forward_count", 0))
            <= 0
        ):
            semantic.append(
                f"{event['prompt_id']} no target-forward reduction"
            )
    routes = {row.get("route") for row in router_rows}
    if "native_multi_token" not in routes:
        semantic.append("native route was not exercised")
    if not any(
        route is not None and route != "native_multi_token"
        for route in routes
    ):
        semantic.append("fallback route was not exercised")
    if semantic:
        return _no_go(
            semantic,
            exactness_pass=False,
            replay_elimination_pass=False,
            router_isolation_pass=False,
        )

    performance = []
    baseline_elapsed = 0.0
    routed_elapsed = 0.0
    baseline_tokens = 0.0
    routed_tokens = 0.0
    prompt_ratios = {}
    bucket_ratios: dict[str, list[float]] = {}
    for prompt in prompts:
        prompt_id = str(prompt["prompt_id"])
        bucket = str(prompt.get("bucket"))
        baseline = by_key[(prompt_id, "baseline")]
        routed = by_key[(prompt_id, "source_routed_native")]
        always = by_key[(prompt_id, "source_always_native")]
        ratio = (
            float(routed["elapsed_s"])
            / float(baseline["elapsed_s"])
        )
        prompt_ratios[prompt_id] = ratio
        bucket_ratios.setdefault(bucket, []).append(ratio)
        if ratio > REAL_THRESHOLDS[
            "max_individual_prompt_elapsed_ratio"
        ]:
            performance.append(
                f"{prompt_id} individual prompt regression"
            )
        if float(routed["elapsed_s"]) > float(always["elapsed_s"]):
            performance.append(
                f"{prompt_id} routed slower than always-native"
            )
        baseline_elapsed += float(baseline["elapsed_s"])
        routed_elapsed += float(routed["elapsed_s"])
        baseline_tokens += float(baseline["output_tokens"])
        routed_tokens += float(routed["output_tokens"])
    elapsed_improvement = 1.0 - (
        routed_elapsed / baseline_elapsed
    )
    baseline_tps = baseline_tokens / baseline_elapsed
    routed_tps = routed_tokens / routed_elapsed
    tps_improvement = routed_tps / baseline_tps - 1.0
    if elapsed_improvement < REAL_THRESHOLDS[
        "min_elapsed_improvement_fraction"
    ]:
        performance.append("aggregate elapsed gain below 5%")
    if tps_improvement < REAL_THRESHOLDS[
        "min_tokens_per_s_improvement_fraction"
    ]:
        performance.append("aggregate tokens/s gain below 5%")
    for bucket, threshold_key in (
        ("natural", "max_natural_elapsed_ratio"),
        (
            "transition_heavy",
            "max_transition_elapsed_ratio",
        ),
    ):
        ratios = bucket_ratios.get(bucket)
        if not ratios:
            return _incomplete([f"{bucket} bucket is missing"])
        if max(ratios) > REAL_THRESHOLDS[threshold_key]:
            performance.append(f"{bucket} bucket regression")
    if performance:
        return _no_go(
            performance,
            exactness_pass=True,
            replay_elimination_pass=True,
            router_isolation_pass=True,
            elapsed_improvement_fraction=elapsed_improvement,
            tokens_per_s_improvement_fraction=tps_improvement,
            per_prompt_elapsed_ratios=prompt_ratios,
        )
    return {
        "classification": "GO",
        "reasons": [],
        "exactness_pass": True,
        "replay_elimination_pass": True,
        "router_isolation_pass": True,
        "performance_direction_pass": True,
        "elapsed_improvement_fraction": elapsed_improvement,
        "tokens_per_s_improvement_fraction": tps_improvement,
        "per_prompt_elapsed_ratios": prompt_ratios,
    }


def _real_policy_process(
    *,
    python_bin: str,
    model_path: str,
    policy: str,
    prompt: dict,
    draft_source: dict,
    repetitions: int,
    warmup_repetitions: int,
    out_path: Path,
    log_dir: Path,
) -> tuple[dict | None, dict]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    dist_port, master_port = _allocate_port_pair()
    stdout_path = log_dir / (
        f"{prompt['prompt_id']}.{policy}.stdout.log"
    )
    stderr_path = log_dir / (
        f"{prompt['prompt_id']}.{policy}.stderr.log"
    )
    runtime_adapter = draft_source.get("runtime_adapter")
    if policy != "baseline" and not runtime_adapter:
        message = (
            "named real draft source has no registered runtime_adapter"
        )
        stdout_path.write_text("")
        stderr_path.write_text(message + "\n")
        return None, {
            "returncode": 2,
            "command": [],
            "tinyvllm_dist_port": dist_port,
            "master_port": master_port,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "elapsed_s": 0.0,
            "attempt": 1,
            "reason": message,
        }
    mode = "baseline-only" if policy == "baseline" else "candidate-only"
    routing = (
        "always-native"
        if policy == "source_always_native"
        else "fixed-profitability"
    )
    command = [
        str(python_bin),
        str(Path(__file__).with_name("profile_ngram_commit.py")),
        "--model",
        str(model_path),
        "--max-output-len",
        str(prompt["max_tokens"]),
        "--temperature",
        "0.0",
        "--mode",
        mode,
        "--max-num-seqs",
        "1",
        "--warmup-output-len",
        str(prompt["max_tokens"] if warmup_repetitions else 0),
        "--warmup-repetitions",
        str(warmup_repetitions),
        "--out-json",
        str(out_path),
    ]
    for _ in range(repetitions):
        command.extend([
            "--prompt",
            str(prompt["prompt"]),
        ])
    if policy != "baseline":
        command.extend([
            "--draft-source",
            str(runtime_adapter),
            "--draft-construction",
            "real_source",
            "--gate-stage",
            "real-source",
            "--speculation-routing",
            routing,
            "--max-draft-tokens",
            str(
                draft_source["hyperparameters"].get(
                    "max_draft_tokens",
                    8,
                )
            ),
        ])
        if policy == "source_routed_native":
            command.append("--allow-incompatible-fallback")
    environment = os.environ.copy()
    environment["TINYVLLM_DIST_PORT"] = str(dist_port)
    environment["MASTER_PORT"] = str(master_port)
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        env=environment,
        check=False,
    )
    stdout_path.write_text(completed.stdout)
    stderr_path.write_text(completed.stderr)
    process = {
        "returncode": int(completed.returncode),
        "command": command,
        "tinyvllm_dist_port": dist_port,
        "master_port": master_port,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "elapsed_s": time.perf_counter() - started,
        "attempt": 1,
    }
    if completed.returncode != 0 or not out_path.is_file():
        return None, process
    profile_payload = json.loads(out_path.read_text())
    summary = profile_payload["summary"]
    per_prompt = profile_payload.get("per_prompt", [])
    output_tokens = int(summary.get("output_tokens", 0))
    token_ids = (
        per_prompt[0].get("token_ids", [])
        if per_prompt
        else []
    )
    payload = {
        "prompt_id": prompt["prompt_id"],
        "bucket": prompt["bucket"],
        "policy": policy,
        "status": "PASS",
        "elapsed_s": float(summary["elapsed_s"]),
        "output_tokens": output_tokens,
        "output_tokens_per_s": float(
            summary["output_tokens_per_s"]
        ),
        "output_token_sha256": _sha256_json(token_ids),
        "event": (
            profile_payload.get("verify_events", [None])[0]
            if policy == "source_routed_native"
            and profile_payload.get("verify_events")
            else None
        ),
        "router_event": (
            profile_payload.get("router_events", [None])[0]
            if policy == "source_routed_native"
            and profile_payload.get("router_events")
            else None
        ),
    }
    _write_json(out_path, payload)
    return payload, process


def _build_real_manifest(
    *,
    source_evidence: dict,
    source_preflight: dict,
    draft_source: dict,
    prompt_bank: dict,
    model_path: str,
    host: str,
    python_bin: str,
    run_tag: str,
    repetitions: int,
    warmup_repetitions: int,
) -> dict:
    return {
        "schema_version": 1,
        "stage": "real-source",
        "run_tag": run_tag,
        "created_unix_s": time.time(),
        "source_tree_sha256": source_evidence["tree_sha256"],
        "source_evidence": source_evidence,
        "source_preflight": source_preflight,
        "draft_source_sha256": _sha256_json(draft_source),
        "prompt_bank_sha256": canonical_prompt_bank_sha256(
            prompt_bank
        ),
        "model_path": str(model_path),
        "host": host,
        "python_bin": python_bin,
        "policies": list(REAL_POLICIES),
        "thresholds": REAL_THRESHOLDS,
        "repetitions": int(repetitions),
        "warmup_repetitions": int(warmup_repetitions),
        "process_port_pairs": [],
    }


def _normalize_real_row(
    payload: dict | None,
    process: dict,
    *,
    prompt: dict,
    policy: str,
    manifest: dict,
    draft_source: dict,
) -> dict:
    identity = {
        "prompt_id": prompt["prompt_id"],
        "bucket": prompt["bucket"],
        "policy": policy,
        "source_tree_sha256": manifest["source_tree_sha256"],
        "draft_source_sha256": manifest["draft_source_sha256"],
        "prompt_bank_sha256": manifest["prompt_bank_sha256"],
        "hyperparameters_sha256": _sha256_json(
            draft_source["hyperparameters"]
        ),
        "process": process,
    }
    if payload is None:
        return {
            **identity,
            "status": "INCOMPLETE",
        }
    return {
        **{
            key: value
            for key, value in payload.items()
            if key not in ("event", "router_event")
        },
        **identity,
        "raw_payload_sha256": _sha256_json(payload),
    }


def run_real_source_gate(
    *,
    out_dir: Path,
    python_bin: str,
    model_path: str,
    source_evidence_path: Path,
    source_patch_path: Path,
    source_preflight_path: Path,
    draft_source_path: Path,
    prompt_bank_path: Path,
    host: str,
    run_tag: str,
    repetitions: int,
    warmup_repetitions: int,
    resume: bool = False,
    prompt_limit: int = 0,
) -> dict:
    if repetitions < 1 or warmup_repetitions < 0:
        raise ValueError("invalid real-source repetition counts")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw"
    log_dir = out_dir / "logs"
    raw_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    source_evidence = json.loads(
        Path(source_evidence_path).read_text()
    )
    source_preflight = json.loads(
        Path(source_preflight_path).read_text()
    )
    draft_source = json.loads(Path(draft_source_path).read_text())
    prompt_bank = json.loads(Path(prompt_bank_path).read_text())
    validate_real_input(draft_source, prompt_bank)
    if (
        source_evidence["tree_sha256"]
        != draft_source["source_tree_sha256"]
    ):
        raise ValueError("draft source tree identity mismatch")
    prompts = list(prompt_bank["prompts"])
    if prompt_limit:
        if prompt_limit < 1:
            raise ValueError("prompt_limit must be non-negative")
        prompts = prompts[:prompt_limit]
    manifest_path = out_dir / "manifest.json"
    if resume and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if any((
            manifest.get("run_tag") != run_tag,
            manifest.get("source_tree_sha256")
            != source_evidence["tree_sha256"],
            manifest.get("draft_source_sha256")
            != _sha256_json(draft_source),
            manifest.get("prompt_bank_sha256")
            != canonical_prompt_bank_sha256(prompt_bank),
        )):
            raise ValueError("real-source resume identity mismatch")
    else:
        manifest = _build_real_manifest(
            source_evidence=source_evidence,
            source_preflight=source_preflight,
            draft_source=draft_source,
            prompt_bank=prompt_bank,
            model_path=model_path,
            host=host,
            python_bin=python_bin,
            run_tag=run_tag,
            repetitions=repetitions,
            warmup_repetitions=warmup_repetitions,
        )
    rows_by_key = {
        _real_row_key(row): row
        for row in (
            _load_json_list(out_dir / "case_rows.json")
            if resume
            else []
        )
    }
    payloads = {}
    for prompt in prompts:
        for policy in REAL_POLICIES:
            key = (prompt["prompt_id"], policy)
            if rows_by_key.get(key, {}).get("status") == "PASS":
                raw_path = raw_dir / f"{key[0]}.{key[1]}.json"
                if raw_path.is_file():
                    payloads[key] = json.loads(raw_path.read_text())
                continue
            payload, process = _real_policy_process(
                python_bin=python_bin,
                model_path=model_path,
                policy=policy,
                prompt=prompt,
                draft_source=draft_source,
                repetitions=repetitions,
                warmup_repetitions=warmup_repetitions,
                out_path=raw_dir / f"{key[0]}.{key[1]}.json",
                log_dir=log_dir,
            )
            _record_process_pair(
                manifest,
                case_id=prompt["prompt_id"],
                policy=policy,
                process=process,
            )
            rows_by_key[key] = _normalize_real_row(
                payload,
                process,
                prompt=prompt,
                policy=policy,
                manifest=manifest,
                draft_source=draft_source,
            )
            if payload is not None:
                payloads[key] = payload
    selected_ids = {prompt["prompt_id"] for prompt in prompts}
    case_rows = sorted(
        (
            row for key, row in rows_by_key.items()
            if key[0] in selected_ids
        ),
        key=_real_row_key,
    )
    event_rows = []
    router_rows = []
    for row in case_rows:
        payload = payloads.get(_real_row_key(row), {})
        event = payload.get("event")
        if (
            row["policy"] == "source_routed_native"
            and isinstance(event, dict)
        ):
            event_rows.append({
                "prompt_id": row["prompt_id"],
                "policy": row["policy"],
                **event,
            })
        router = payload.get("router_event")
        if (
            row["policy"] == "source_routed_native"
            and isinstance(router, dict)
        ):
            router_rows.append({
                "prompt_id": row["prompt_id"],
                "policy": row["policy"],
                **router,
            })
    summary = classify_real_source_gate(
        manifest,
        draft_source,
        prompt_bank,
        case_rows,
        event_rows,
        router_rows,
    )
    for name, source in {
        "source_evidence.json": Path(source_evidence_path),
        "source.patch": Path(source_patch_path),
        "source_preflight.json": Path(source_preflight_path),
        "draft_source.json": Path(draft_source_path),
        "prompt_bank.json": Path(prompt_bank_path),
    }.items():
        target = out_dir / name
        if source.resolve() != target.resolve():
            shutil.copyfile(source, target)
    (out_dir / "prompt_bank.sha256").write_text(
        manifest["prompt_bank_sha256"] + "\n"
    )
    _write_json(manifest_path, manifest)
    _write_json(out_dir / "case_rows.json", case_rows)
    _write_json(out_dir / "event_rows.json", event_rows)
    _write_json(out_dir / "router_rows.json", router_rows)
    _write_json(out_dir / "summary.json", summary)
    return {
        "manifest": manifest,
        "case_rows": case_rows,
        "event_rows": event_rows,
        "router_rows": router_rows,
        "summary": summary,
    }


def _normalize_controlled_row(
    payload: dict | None,
    process: dict,
    *,
    case_id: str,
    policy: str,
    source_tree_sha256: str,
) -> dict:
    if payload is None:
        return {
            "case_id": case_id,
            "policy": policy,
            "status": "INCOMPLETE",
            "source_tree_sha256": source_tree_sha256,
            "process": process,
        }
    if (
        policy in ("always_native", "routed_native")
        and payload.get("draft_construction")
        != "controlled_target_derived"
    ):
        raise ValueError(
            "controlled runtime returned non-target-derived draft"
        )
    row = {
        key: value
        for key, value in payload.items()
        if key not in _LARGE_EVIDENCE_FIELDS
    }
    row.update({
        "case_id": case_id,
        "policy": policy,
        "source_tree_sha256": source_tree_sha256,
        "process": process,
        "raw_payload_sha256": _sha256_json(payload),
    })
    event = payload.get("event")
    if isinstance(event, dict):
        for field in (
            "accepted_count",
            "target_forward_count",
            "normal_decode_forward_count",
        ):
            if field in event:
                row[field] = int(event[field])
    for field in _LARGE_EVIDENCE_FIELDS:
        if field in payload and payload[field] is not None:
            row[f"{field}_sha256"] = _sha256_json(payload[field])
    return row


def _native_event_row(row: dict, payload: dict) -> dict | None:
    event = payload.get("event")
    if not isinstance(event, dict):
        return None
    return {
        "case_id": row["case_id"],
        "policy": row["policy"],
        "draft_len": int(event["draft_len"]),
        "accepted_count": int(event["accepted_count"]),
        "accepted_kv_rematerialization": event[
            "accepted_kv_rematerialization"
        ],
        "accepted_kv_copy_calls": int(
            event["accepted_kv_copy_calls"]
        ),
        "accepted_kv_replay_calls": int(
            event["accepted_kv_replay_calls"]
        ),
        "target_forward_count": int(
            event["target_forward_count"]
        ),
        "verifier_commit_ms": float(
            event["timing_ms"]["verify_commit_total_ms"]
        ),
    }


def _router_row(row: dict, payload: dict) -> dict | None:
    event = payload.get("router_event")
    if row["policy"] != "routed_native" or not isinstance(
        event,
        dict,
    ):
        return None
    return {
        "case_id": row["case_id"],
        "policy": "routed_native",
        "route": event.get("route"),
        "draft_len": int(event["draft_len"]),
        "route_fallback_reason": event.get(
            "route_fallback_reason"
        ),
        "speculative_reservation_attempted": bool(
            event.get("speculative_reservation_attempted", False)
        ),
        "spec_verify_prepare_calls": int(
            event.get("spec_verify_prepare_calls", 0)
        ),
        "spec_verify_forward_calls": int(
            event.get("spec_verify_forward_calls", 0)
        ),
        "target_forward_count": int(
            event.get("target_forward_count", 0)
        ),
    }


def _load_json_list(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    value = json.loads(path.read_text())
    if not isinstance(value, list):
        raise ValueError(f"{path.name} must contain a list")
    return value


def _exactness_reference_row(
    case_spec: dict,
    rows_by_key: dict[tuple[str, str], dict],
) -> dict:
    policy = (
        "baseline"
        if int(case_spec["draft_len"]) <= 1
        else "routed_native"
    )
    return rows_by_key[(case_spec["case_id"], policy)]


def _write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n"
    )


def _record_process_pair(
    manifest: dict,
    *,
    case_id: str,
    policy: str,
    process: dict,
) -> None:
    rows = [
        row
        for row in manifest.get("process_port_pairs", [])
        if (
            row.get("case_id"),
            row.get("policy"),
        ) != (case_id, policy)
    ]
    rows.append({
        "case_id": case_id,
        "policy": policy,
        "tinyvllm_dist_port": process[
            "tinyvllm_dist_port"
        ],
        "master_port": process["master_port"],
    })
    manifest["process_port_pairs"] = rows


def run_controlled_gate(
    *,
    out_dir: Path,
    python_bin: str,
    model_path: str,
    source_evidence_path: Path,
    source_patch_path: Path,
    source_preflight_path: Path,
    host: str,
    run_tag: str,
    resume: bool = False,
    case_limit: int = 0,
) -> dict:
    from native_verifier_oracle import (
        compare_native_and_oracle,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw"
    log_dir = out_dir / "logs"
    raw_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    source_evidence = json.loads(
        Path(source_evidence_path).read_text()
    )
    source_preflight = json.loads(
        Path(source_preflight_path).read_text()
    )
    source_tree_sha256 = source_evidence["tree_sha256"]
    capability = json.loads(
        (out_dir / "capability.json").read_text()
    )
    selected_cases = list(CONTROLLED_CASE_MATRIX)
    if case_limit:
        if case_limit < 1:
            raise ValueError("case_limit must be non-negative")
        selected_cases = selected_cases[:case_limit]

    existing_rows = {
        _row_key(row): row
        for row in (
            _load_json_list(out_dir / "case_rows.json")
            if resume
            else []
        )
    }
    manifest_path = out_dir / "manifest.json"
    if resume and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest.get("source_tree_sha256")
            != source_tree_sha256
            or manifest.get("model_path") != str(model_path)
            or manifest.get("run_tag") != run_tag
        ):
            raise ValueError("resume manifest identity mismatch")
    else:
        manifest = build_controlled_manifest(
            source_evidence=source_evidence,
            source_preflight=source_preflight,
            model_path=str(model_path),
            model_identifier=source_preflight[
                "model_identifier"
            ],
            host=host,
            python_bin=python_bin,
            torch_version=source_preflight["torch"],
            cuda_version=source_preflight["cuda"],
            flash_attn_version=source_preflight["flash_attn"],
            gpu_name=source_preflight["gpu"],
            bf16_supported=source_preflight["bf16_supported"],
            run_tag=run_tag,
        )

    rows_by_key = dict(existing_rows)
    payloads_by_key = {}
    for case_spec in selected_cases:
        needed_policies = [
            policy
            for policy in CONTROLLED_POLICIES
            if rows_by_key.get(
                (case_spec["case_id"], policy),
                {},
            ).get("status") != "PASS"
        ]
        case_path = raw_dir / (
            f"{case_spec['case_id']}.materialized.json"
        )
        if needed_policies:
            if resume and case_path.is_file():
                case = json.loads(case_path.read_text())
            else:
                probe_case = {
                    **case_spec,
                    "prompt": (
                        _CONTROLLED_EOS_PROMPT
                        if case_spec["eos_case"]
                        else _CONTROLLED_PROMPT
                    ),
                    "draft_tokens": [0] * int(
                        case_spec["draft_len"]
                    ),
                    "max_tokens": 2048,
                    "ignore_eos": True,
                }
                probe, probe_process = _case_process(
                    python_bin=python_bin,
                    model_path=model_path,
                    policy="probe",
                    case=probe_case,
                    out_path=raw_dir / (
                        f"{case_spec['case_id']}.probe.json"
                    ),
                    log_dir=log_dir,
                )
                _record_process_pair(
                    manifest,
                    case_id=case_spec["case_id"],
                    policy="probe",
                    process=probe_process,
                )
                if probe is None:
                    for policy in needed_policies:
                        rows_by_key[
                            (case_spec["case_id"], policy)
                        ] = _normalize_controlled_row(
                            None,
                            probe_process,
                            case_id=case_spec["case_id"],
                            policy=policy,
                            source_tree_sha256=source_tree_sha256,
                        )
                    continue
                case = _materialize_controlled_case(
                    case_spec,
                    probe,
                    source_tree_sha256=source_tree_sha256,
                )
                _write_json(case_path, case)
            for policy in needed_policies:
                payload, process = _case_process(
                    python_bin=python_bin,
                    model_path=model_path,
                    policy=policy,
                    case=case,
                    out_path=raw_dir / (
                        f"{case_spec['case_id']}.{policy}.json"
                    ),
                    log_dir=log_dir,
                )
                _record_process_pair(
                    manifest,
                    case_id=case_spec["case_id"],
                    policy=policy,
                    process=process,
                )
                key = (case_spec["case_id"], policy)
                rows_by_key[key] = _normalize_controlled_row(
                    payload,
                    process,
                    case_id=case_spec["case_id"],
                    policy=policy,
                    source_tree_sha256=source_tree_sha256,
                )
                if payload is not None:
                    payloads_by_key[key] = payload

    selected_ids = {
        case["case_id"] for case in selected_cases
    }
    case_rows = sorted(
        (
            row for key, row in rows_by_key.items()
            if key[0] in selected_ids
        ),
        key=_row_key,
    )
    for row in case_rows:
        key = _row_key(row)
        if key not in payloads_by_key:
            raw_path = raw_dir / f"{key[0]}.{key[1]}.json"
            if raw_path.is_file():
                payloads_by_key[key] = json.loads(
                    raw_path.read_text()
                )

    for case_spec in selected_cases:
        routed_key = (
            case_spec["case_id"],
            "routed_native",
        )
        oracle_key = (case_spec["case_id"], "oracle")
        if (
            _exactness_reference_row(
                case_spec,
                rows_by_key,
            ).get("status") == "PASS"
            and rows_by_key.get(oracle_key, {}).get("status")
            == "PASS"
        ):
            rows_by_key[oracle_key]["comparison"] = (
                compare_native_and_oracle(
                    payloads_by_key[_row_key(
                        _exactness_reference_row(
                            case_spec,
                            rows_by_key,
                        )
                    )],
                    payloads_by_key[oracle_key],
                )
            )

    event_rows = []
    router_rows = []
    for row in case_rows:
        payload = payloads_by_key.get(_row_key(row), {})
        if row["policy"] in ("always_native", "routed_native"):
            event = _native_event_row(row, payload)
            if event is not None and not (
                row["policy"] == "routed_native"
                and int(event["draft_len"]) <= 1
            ):
                event_rows.append(event)
        router = _router_row(row, payload)
        if router is not None:
            router_rows.append(router)

    case_rows = sorted(
        (
            row for key, row in rows_by_key.items()
            if key[0] in selected_ids
        ),
        key=_row_key,
    )
    summary = classify_controlled_gate(
        manifest,
        capability,
        case_rows,
        event_rows,
        router_rows,
    )
    source_targets = {
        "source_evidence.json": Path(source_evidence_path),
        "source.patch": Path(source_patch_path),
        "source_preflight.json": Path(source_preflight_path),
    }
    for name, source in source_targets.items():
        target = out_dir / name
        if source.resolve() != target.resolve():
            shutil.copyfile(source, target)
    _write_json(manifest_path, manifest)
    _write_json(out_dir / "case_rows.json", case_rows)
    _write_json(out_dir / "event_rows.json", event_rows)
    _write_json(out_dir / "router_rows.json", router_rows)
    _write_json(out_dir / "summary.json", summary)
    return {
        "manifest": manifest,
        "case_rows": case_rows,
        "event_rows": event_rows,
        "router_rows": router_rows,
        "summary": summary,
    }


def _incomplete(reasons: list[str]) -> dict:
    return {
        "classification": "INCOMPLETE",
        "reasons": sorted(set(reasons)),
        "exactness_pass": False,
        "replay_elimination_pass": False,
        "router_isolation_pass": False,
        "performance_direction_pass": False,
    }


def _no_go(
    reasons: list[str],
    *,
    exactness_pass: bool,
    replay_elimination_pass: bool,
    router_isolation_pass: bool,
    **extra,
) -> dict:
    return {
        "classification": "NO_GO",
        "reasons": sorted(set(reasons)),
        "exactness_pass": exactness_pass,
        "replay_elimination_pass": replay_elimination_pass,
        "router_isolation_pass": router_isolation_pass,
        "performance_direction_pass": False,
        **extra,
    }


def _row_key(row: dict) -> tuple[str, str]:
    return str(row.get("case_id")), str(row.get("policy"))


def _expected_row_keys() -> set[tuple[str, str]]:
    return {
        (case["case_id"], policy)
        for case in CONTROLLED_CASE_MATRIX
        for policy in CONTROLLED_POLICIES
    }


def _capability_complete(
    manifest: dict,
    capability: dict,
) -> bool:
    if capability.get("status") != "PASS":
        return False
    required_dtypes = {"torch.float16"}
    if manifest.get("bf16_supported"):
        required_dtypes.add("torch.bfloat16")
    required = {
        (dtype, query_len, block_case)
        for dtype in required_dtypes
        for query_len in (1, 3, 7, 15)
        for block_case in ("one_block", "cross_block")
    }
    observed = {
        (
            row.get("dtype"),
            row.get("query_len"),
            row.get("block_case"),
        )
        for row in capability.get("rows", [])
        if all(
            row.get(field) is True
            for field in (
                "gqa",
                "output_match",
                "kv_match",
                "future_row_masked",
                "finite",
            )
        )
    }
    return required <= observed


def classify_controlled_gate(
    manifest: dict,
    capability: dict,
    case_rows: list[dict],
    event_rows: list[dict],
    router_rows: list[dict],
) -> dict:
    structural = []
    if manifest.get("stage") != "controlled":
        structural.append("manifest stage mismatch")
    if manifest.get("thresholds") != CONTROLLED_THRESHOLDS:
        structural.append("manifest thresholds drift")
    if manifest.get("case_matrix") != list(
        CONTROLLED_CASE_MATRIX
    ):
        structural.append("manifest case matrix drift")
    if manifest.get("policies") != list(CONTROLLED_POLICIES):
        structural.append("manifest policies drift")
    evidence = manifest.get("source_evidence", {})
    if (
        manifest.get("source_tree_sha256")
        != evidence.get("tree_sha256")
    ):
        structural.append("manifest source tree mismatch")
    if (
        manifest.get("source_preflight", {}).get(
            "source_tree_sha256"
        )
        != manifest.get("source_tree_sha256")
    ):
        structural.append("source preflight tree mismatch")

    observed_keys = [_row_key(row) for row in case_rows]
    expected_keys = _expected_row_keys()
    if len(observed_keys) != len(set(observed_keys)):
        structural.append("duplicate policy/case rows")
    if set(observed_keys) != expected_keys:
        structural.append("missing or unexpected policy/case rows")
    for row in case_rows:
        process = row.get("process")
        if not isinstance(process, dict):
            structural.append(
                f"{_row_key(row)} missing process evidence"
            )
            continue
        if process.get("returncode") != 0:
            structural.append(f"{_row_key(row)} process failed")
        for field in (
            "tinyvllm_dist_port",
            "master_port",
        ):
            if not isinstance(process.get(field), int):
                structural.append(
                    f"{_row_key(row)} missing dynamic {field}"
                )
        if (
            row.get("source_tree_sha256")
            != manifest.get("source_tree_sha256")
        ):
            structural.append(
                f"{_row_key(row)} source tree mismatch"
            )
        for field in (
            "elapsed_s",
            "output_tokens",
            "output_tokens_per_s",
            "target_forward_count",
        ):
            value = row.get(field)
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                structural.append(
                    f"{_row_key(row)} invalid {field}"
                )

    port_pairs = [
        (
            row.get("tinyvllm_dist_port"),
            row.get("master_port"),
        )
        for row in manifest.get("process_port_pairs", [])
    ]
    if len(port_pairs) != len(set(port_pairs)):
        structural.append("duplicate dynamic port pairs")

    expected_router_cases = {
        case["case_id"] for case in CONTROLLED_CASE_MATRIX
    }
    router_keys = [
        (row.get("case_id"), row.get("policy"))
        for row in router_rows
    ]
    if len(router_keys) != len(set(router_keys)):
        structural.append("duplicate router rows")
    if {
        case_id for case_id, policy in router_keys
        if policy == "routed_native"
    } != expected_router_cases:
        structural.append("missing or unexpected router rows")

    native_event_keys = [
        (event.get("case_id"), event.get("policy"))
        for event in event_rows
    ]
    expected_event_keys = {
        (case["case_id"], policy)
        for case in CONTROLLED_CASE_MATRIX
        for policy in ("always_native", "routed_native")
        if not (
            policy == "routed_native"
            and case["draft_len"] <= 1
        )
    }
    if len(native_event_keys) != len(set(native_event_keys)):
        structural.append("duplicate native event rows")
    if set(native_event_keys) != expected_event_keys:
        structural.append("missing or unexpected native events")
    if structural:
        return _incomplete(structural)

    if not _capability_complete(manifest, capability):
        return _incomplete([
            "capability matrix is incomplete or divergent",
        ])

    by_key = {_row_key(row): row for row in case_rows}
    semantic = []
    for case in CONTROLLED_CASE_MATRIX:
        case_id = case["case_id"]
        baseline = by_key[(case_id, "baseline")]
        output_hash = baseline.get("output_token_sha256")
        continuation_hash = baseline.get(
            "continuation_token_sha256"
        )
        if not output_hash or not continuation_hash:
            return _incomplete([
                f"{case_id} baseline token hashes are missing",
            ])
        for policy in CONTROLLED_POLICIES:
            row = by_key[(case_id, policy)]
            if row.get("status") == "INCOMPLETE":
                return _incomplete([
                    f"{case_id}/{policy} row is INCOMPLETE",
                ])
            if row.get("status") != "PASS":
                semantic.append(
                    f"{case_id}/{policy} semantic failure"
                )
            if row.get("output_token_sha256") != output_hash:
                semantic.append(
                    f"{case_id}/{policy} output token mismatch"
                )
            if (
                row.get("continuation_token_sha256")
                != continuation_hash
            ):
                semantic.append(
                    f"{case_id}/{policy} continuation mismatch"
                )
        comparison = by_key[(case_id, "oracle")].get(
            "comparison"
        )
        if not isinstance(comparison, dict):
            return _incomplete([
                f"{case_id} comparison is missing",
            ])
        if comparison.get("status") == "INCOMPLETE":
            return _incomplete([
                f"{case_id} comparison is INCOMPLETE",
            ])
        for field in (
            "target_token_match",
            "accepted_prefix_match",
            "metadata_match",
            "continuation_token_match",
            "finite",
            "logits_within_tolerance",
            "kv_within_tolerance",
        ):
            if comparison.get(field) is not True:
                semantic.append(
                    f"{case_id} comparison {field} mismatch"
                )
        if (
            int(comparison.get("continuation_steps", -1))
            < CONTROLLED_THRESHOLDS["min_continuation_steps"]
        ):
            semantic.append(
                f"{case_id} continuation below minimum"
            )
    if semantic:
        return _no_go(
            semantic,
            exactness_pass=False,
            replay_elimination_pass=False,
            router_isolation_pass=False,
        )

    replay = []
    for event in event_rows:
        rematerialization = event.get(
            "accepted_kv_rematerialization",
            {},
        )
        if (
            rematerialization.get("decode_calls") != 0
            or rematerialization.get("rematerialized_tokens")
            or float(rematerialization.get("elapsed_ms", math.nan))
            != 0.0
        ):
            replay.append(
                f"{event['case_id']}/{event['policy']} "
                "accepted KV rematerialization remains"
            )
        if event.get("accepted_kv_copy_calls") != 0:
            replay.append(
                f"{event['case_id']}/{event['policy']} "
                "accepted KV copy remains"
            )
        if event.get("accepted_kv_replay_calls") != 0:
            replay.append(
                f"{event['case_id']}/{event['policy']} "
                "accepted KV replay remains"
            )
    if replay:
        return _no_go(
            replay,
            exactness_pass=True,
            replay_elimination_pass=False,
            router_isolation_pass=False,
        )

    router_failures = []
    for row in router_rows:
        if int(row["draft_len"]) <= 1:
            if row.get("route") != "baseline_short_draft":
                router_failures.append(
                    f"{row['case_id']} short draft route mismatch"
                )
            for field in (
                "speculative_reservation_attempted",
                "spec_verify_prepare_calls",
                "spec_verify_forward_calls",
                "target_forward_count",
            ):
                expected = False if field == (
                    "speculative_reservation_attempted"
                ) else 0
                if row.get(field) != expected:
                    router_failures.append(
                        f"{row['case_id']} short route mutated {field}"
                    )
        elif row.get("route") != "native_multi_token":
            router_failures.append(
                f"{row['case_id']} multi-token route mismatch"
            )
    if router_failures:
        return _no_go(
            router_failures,
            exactness_pass=True,
            replay_elimination_pass=True,
            router_isolation_pass=False,
        )

    lifecycle_reasons = []
    profitable_ratios = []
    per_case_ratios = {}
    for case in CONTROLLED_CASE_MATRIX:
        case_id = case["case_id"]
        routed = by_key[(case_id, "routed_native")]
        baseline = by_key[(case_id, "baseline")]
        ratio = (
            float(routed["elapsed_s"])
            / float(baseline["elapsed_s"])
        )
        per_case_ratios[case_id] = ratio
        if ratio > CONTROLLED_THRESHOLDS[
            "max_required_lifecycle_elapsed_ratio"
        ]:
            lifecycle_reasons.append(
                f"required_lifecycle_regression:{case_id}"
            )
        if (
            int(case["draft_len"]) >= 2
            and int(case["expected_accepted_count"]) >= 2
        ):
            profitable_ratios.append(ratio)
    if lifecycle_reasons:
        return _no_go(
            lifecycle_reasons,
            exactness_pass=True,
            replay_elimination_pass=True,
            router_isolation_pass=True,
            per_case_elapsed_ratios=per_case_ratios,
        )
    best_ratio = min(profitable_ratios)
    if best_ratio >= CONTROLLED_THRESHOLDS[
        "profitable_region_max_elapsed_ratio"
    ]:
        return _no_go(
            ["no_profitable_k_ge_2_region"],
            exactness_pass=True,
            replay_elimination_pass=True,
            router_isolation_pass=True,
            best_profitable_region_elapsed_ratio=best_ratio,
            per_case_elapsed_ratios=per_case_ratios,
        )

    return {
        "classification": "READY_FOR_REAL_DRAFTER_GATE",
        "reasons": [],
        "exactness_pass": True,
        "replay_elimination_pass": True,
        "router_isolation_pass": True,
        "performance_direction_pass": True,
        "observed_case_rows": len(case_rows),
        "observed_native_events": len(event_rows),
        "observed_router_rows": len(router_rows),
        "best_profitable_region_elapsed_ratio": best_ratio,
        "median_profitable_region_elapsed_ratio": (
            statistics.median(profitable_ratios)
        ),
        "per_case_elapsed_ratios": per_case_ratios,
    }


def _parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--out-dir", type=Path, required=True)
    real_input_parser = subparsers.add_parser(
        "validate-real-input"
    )
    real_input_parser.add_argument(
        "--draft-source",
        type=Path,
        required=True,
    )
    real_input_parser.add_argument(
        "--prompt-bank",
        type=Path,
        required=True,
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.command == "verify":
        raise SystemExit(
            "artifact verification is not implemented yet"
        )
    if args.command == "validate-real-input":
        result = validate_real_input(
            json.loads(args.draft_source.read_text()),
            json.loads(args.prompt_bank.read_text()),
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    raise AssertionError(args.command)


if __name__ == "__main__":
    main()
