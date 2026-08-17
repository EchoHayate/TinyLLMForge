from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load(name, filename):
    module = sys.modules.get(name)
    if module is not None:
        return module
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


contract = _load(
    "qwen35_tp4_cached_continuation_correctness_contract",
    "qwen35_tp4_cached_continuation_correctness_contract.py",
)
backend = _load(
    "qwen35_tp4_cached_continuation_backend_session",
    "qwen35_tp4_cached_continuation_backend_session.py",
)


def _run_request(
    engine,
    prompt,
    generated_tokens,
    *,
    timeout_s,
    record_logits=True,
):
    if record_logits:
        enabled = engine.enable_step_logits_authority_recording(
            True,
            timeout_s=timeout_s,
        )
        if enabled != {
            "enabled": True,
            "rank_inventory": [0, 1, 2, 3],
        }:
            raise ValueError(
                "probe logits recording enable mismatch"
            )
    engine.add_request(
        list(prompt),
        backend._sampling_params(generated_tokens),
    )
    output_token_ids = []
    executed_prefill_tokens = 0
    step_logits = []
    try:
        while not engine.is_finished():
            outputs, num_tokens = engine.step()
            if num_tokens > 0:
                executed_prefill_tokens += num_tokens
            observation = getattr(
                engine,
                "last_step_observation",
                None,
            )
            sampled = (
                isinstance(observation, dict)
                and observation.get("do_sample") is True
                and any(
                    token_ids
                    for token_ids in observation.get(
                        "new_completion_tokens_by_seq",
                        {},
                    ).values()
                )
            )
            if record_logits and sampled:
                step_logits.append(
                    engine.read_step_logits_authority()
                )
            for _, token_ids in outputs:
                if len(token_ids) >= generated_tokens:
                    output_token_ids = list(token_ids)
        if len(output_token_ids) != generated_tokens:
            raise ValueError("probe output token count mismatch")
        if record_logits and len(step_logits) != generated_tokens:
            raise ValueError("probe step logits count mismatch")
        return {
            "output_token_ids": output_token_ids,
            "executed_prefill_tokens": executed_prefill_tokens,
            "step_logits": step_logits,
        }
    finally:
        if record_logits:
            disabled = engine.enable_step_logits_authority_recording(
                False,
                timeout_s=timeout_s,
            )
            if disabled != {
                "enabled": False,
                "rank_inventory": [0, 1, 2, 3],
            }:
                raise ValueError(
                    "probe logits recording disable mismatch"
                )


def _snapshot(engine, *, timeout_s):
    rows = engine.qwen35_hybrid_prefix_authority_snapshots(
        timeout_s=timeout_s,
    )
    if (
        not isinstance(rows, tuple)
        or [row.get("rank") for row in rows] != [0, 1, 2, 3]
    ):
        raise ValueError("probe rank snapshot mismatch")
    reference = {
        name: rows[0][name]
        for name in rows[0]
        if name != "rank"
    }
    if any(
        {
            name: row[name]
            for name in row
            if name != "rank"
        }
        != reference
        for row in rows[1:]
    ):
        raise ValueError("probe rank snapshot parity mismatch")
    return dict(rows[0])


def _classify(recompute, restore):
    recompute_exact = recompute["comparison"]["allclose"] is True
    restore_exact = restore["comparison"]["allclose"] is True
    if recompute_exact and restore_exact:
        return "NO_DIVERGENCE"
    if recompute_exact and not restore_exact:
        return "RESTORE_ONLY_DIVERGENCE"
    if not recompute_exact and restore_exact:
        return "RECOMPUTE_ONLY_DIVERGENCE"
    return "ENGINE_WIDE_DIVERGENCE"


def run_probe(
    *,
    configuration,
    engine_factory,
    reference_executor_factory,
    logits_comparator=backend._default_logits_comparator,
    workload="w1_medium_reuse",
    request_index=0,
    generated_tokens=1,
):
    for value, label in (
        (engine_factory, "engine_factory"),
        (reference_executor_factory, "reference_executor_factory"),
        (logits_comparator, "logits_comparator"),
    ):
        if not callable(value):
            raise TypeError(f"{label} must be callable")
    payload = contract.workload_payload(workload)
    continuation_count = payload["spec"]["continuations"]
    if (
        isinstance(request_index, bool)
        or not isinstance(request_index, int)
        or request_index < 0
        or request_index >= continuation_count
    ):
        raise ValueError("probe request index mismatch")
    if (
        isinstance(generated_tokens, bool)
        or not isinstance(generated_tokens, int)
        or generated_tokens <= 0
    ):
        raise ValueError("probe generated tokens must be positive")

    prompt = backend._request_prompt(payload, request_index)
    reference_executor = reference_executor_factory()
    try:
        official = (
            reference_executor.generate_reference_with_step_logits(
                scenario="publish_source",
                prompt_token_ids=prompt,
                generated_tokens=generated_tokens,
                generation_policy={
                    "temperature": 0.0,
                    "ignore_eos": True,
                },
            )
        )
    finally:
        reference_executor.close()
    if (
        not isinstance(official, dict)
        or set(official) != {"output_token_ids", "step_logits"}
        or len(official["output_token_ids"]) != generated_tokens
        or len(official["step_logits"]) != generated_tokens
    ):
        raise ValueError("probe official reference evidence is invalid")

    engine = engine_factory(configuration)
    timeout_s = configuration.timeout_s
    engine.configure_qwen35_hybrid_prefix_publication_runtime(
        model_fingerprint=configuration.model_fingerprint,
        max_entries=configuration.max_cache_entries,
        max_bytes=configuration.max_cache_bytes,
        timeout_s=timeout_s,
    )
    cleanup_receipt = None
    try:
        engine.clear_qwen35_hybrid_prefix_caches(
            timeout_s=timeout_s,
        )
        recompute_before = _snapshot(engine, timeout_s=timeout_s)
        recompute = _run_request(
            engine,
            prompt,
            generated_tokens,
            timeout_s=timeout_s,
        )
        recompute_after = _snapshot(engine, timeout_s=timeout_s)
        recompute["restore_hits"] = (
            recompute_after["hits"] - recompute_before["hits"]
        )
        recompute["restore_misses"] = (
            recompute_after["misses"] - recompute_before["misses"]
        )
        recompute["comparison"] = logits_comparator(
            recompute.pop("step_logits"),
            official["step_logits"],
            atol=contract.REGISTERED_LOGITS_ATOL,
        )

        source_prompt = (
            list(payload["shared_prefix_token_ids"])
            + list(payload["source_suffix_token_ids"])
        )
        _run_request(
            engine,
            source_prompt,
            1,
            timeout_s=timeout_s,
            record_logits=False,
        )
        source_snapshot = _snapshot(engine, timeout_s=timeout_s)
        if (
            source_snapshot["current_entries"] != 1
            or source_snapshot["publication_commits"] < 1
            or not source_snapshot[
                "last_publication_block_identities"
            ]
        ):
            raise ValueError("probe source publication is incomplete")

        restore_before = _snapshot(engine, timeout_s=timeout_s)
        restore = _run_request(
            engine,
            prompt,
            generated_tokens,
            timeout_s=timeout_s,
        )
        restore_after = _snapshot(engine, timeout_s=timeout_s)
        restore["restore_hits"] = (
            restore_after["hits"] - restore_before["hits"]
        )
        restore["restore_misses"] = (
            restore_after["misses"] - restore_before["misses"]
        )
        restore["comparison"] = logits_comparator(
            restore.pop("step_logits"),
            official["step_logits"],
            atol=contract.REGISTERED_LOGITS_ATOL,
        )
        result = {
            "schema_version": (
                "qwen35.tp4-cached-first-divergence-probe.v1"
            ),
            "classification": _classify(recompute, restore),
            "workload": workload,
            "request_index": request_index,
            "generated_tokens": generated_tokens,
            "registered_logits_atol": (
                contract.REGISTERED_LOGITS_ATOL
            ),
            "registered_logits_rtol": 0.0,
            "official_output_token_ids": list(
                official["output_token_ids"]
            ),
            "recompute": recompute,
            "restore": restore,
            "claim_boundary": (
                "diagnostic comparison only; no cached-continuation "
                "correctness, Engine correctness, performance, cache, "
                "memory, compression, quality, or accuracy claim"
            ),
        }
    finally:
        cleanup_receipt = engine.exit()
    if (
        not isinstance(cleanup_receipt, dict)
        or cleanup_receipt.get("process_group_destroyed") is not True
        or cleanup_receipt.get("owned_children_remaining") != []
    ):
        raise ValueError("probe cleanup receipt is invalid")
    result["cleanup"] = cleanup_receipt
    return result
