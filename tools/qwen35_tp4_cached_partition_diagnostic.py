from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys


SCHEMA_VERSION = "qwen35.tp4-cached-partition-diagnostic.v1"
LOGITS_ATOL = 2e-5


def _load_module(module_name, filename):
    module = sys.modules.get(module_name)
    if module is not None:
        return module
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _default_engine_factory(configuration):
    backend = _load_module(
        "qwen35_tp4_engine_backend_session",
        "qwen35_tp4_engine_backend_session.py",
    )
    return backend._default_engine_factory(configuration)


def _default_reference_executor_factory(configuration):
    official = _load_module(
        "qwen35_tp4_engine_official_reference_executor",
        "qwen35_tp4_engine_official_reference_executor.py",
    )
    return official.build_official_reference_executor_factory(
        configuration
    )


def _default_logits_comparator(left, right, *, atol):
    backend = _load_module(
        "qwen35_tp4_cached_continuation_backend_session",
        "qwen35_tp4_cached_continuation_backend_session.py",
    )
    return backend._default_logits_comparator(
        left,
        right,
        atol=atol,
    )


def _sampling_params(max_tokens):
    backend = _load_module(
        "qwen35_tp4_cached_continuation_backend_session",
        "qwen35_tp4_cached_continuation_backend_session.py",
    )
    return backend._sampling_params(max_tokens)


def _authority_snapshot(engine, configuration):
    rows = engine.qwen35_hybrid_prefix_authority_snapshots(
        timeout_s=configuration.timeout_s,
    )
    world_size = len(configuration.gpu_indices)
    if (
        not isinstance(rows, tuple)
        or len(rows) != world_size
        or [row.get("rank") for row in rows]
        != list(range(world_size))
    ):
        raise ValueError("diagnostic authority rank inventory mismatch")
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
        } != reference
        for row in rows[1:]
    ):
        raise ValueError("diagnostic authority rank parity mismatch")
    return dict(rows[0])


def _counter_delta(before, after):
    return {
        name: after[name] - before[name]
        for name in ("hits", "misses", "publication_commits")
    }


def _run_request(
    engine,
    configuration,
    *,
    name,
    prompt_token_ids,
    generated_tokens,
):
    enabled = engine.enable_step_logits_authority_recording(
        True,
        timeout_s=configuration.timeout_s,
    )
    if enabled != {
        "enabled": True,
        "rank_inventory": list(
            range(len(configuration.gpu_indices))
        ),
    }:
        raise ValueError("diagnostic logits recording enable mismatch")
    engine.add_request(
        list(prompt_token_ids),
        _sampling_params(generated_tokens),
    )
    output_token_ids = []
    executed_prefill_tokens = 0
    prefill_chunks = []
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
            if isinstance(observation, dict):
                for scheduled in observation.get("scheduled", ()):
                    if (
                        isinstance(scheduled, dict)
                        and scheduled.get("is_decode") is False
                    ):
                        prefill_chunks.append([
                            int(scheduled["prefill_chunk_start"]),
                            int(scheduled["prefill_chunk_end"]),
                        ])
                sampled = (
                    observation.get("do_sample") is True
                    and any(
                        token_ids
                        for token_ids in observation.get(
                            "new_completion_tokens_by_seq",
                            {},
                        ).values()
                    )
                )
                if sampled:
                    step_logits.append(
                        engine.read_step_logits_authority()
                    )
            for _, token_ids in outputs:
                if len(token_ids) >= generated_tokens:
                    output_token_ids = list(token_ids)
        if len(output_token_ids) != generated_tokens:
            raise ValueError(
                f"{name} output token count mismatch"
            )
        if len(step_logits) != generated_tokens:
            raise ValueError(
                f"{name} step logits count mismatch"
            )
        return {
            "name": name,
            "executed_prefill_tokens": executed_prefill_tokens,
            "prefill_chunks": prefill_chunks,
            "output_token_ids": output_token_ids,
            "_step_logits": step_logits,
        }
    finally:
        disabled = engine.enable_step_logits_authority_recording(
            False,
            timeout_s=configuration.timeout_s,
        )
        if disabled != {
            "enabled": False,
            "rank_inventory": list(
                range(len(configuration.gpu_indices))
            ),
        }:
            raise ValueError(
                "diagnostic logits recording disable mismatch"
            )


def _comparison(
    logits_comparator,
    left,
    right,
):
    result = logits_comparator(
        left,
        right,
        atol=LOGITS_ATOL,
    )
    required = {
        "max_abs_diff",
        "per_step_max_abs_diff",
        "allclose",
        "first_mismatch_step",
    }
    if not isinstance(result, dict) or not required.issubset(result):
        raise ValueError("diagnostic logits comparison is incomplete")
    return {
        name: result[name]
        for name in (
            "max_abs_diff",
            "per_step_max_abs_diff",
            "allclose",
            "first_mismatch_step",
        )
    }


def _classify(comparisons):
    partition_equal = comparisons[
        "native_full_vs_native_partitioned_miss"
    ]["allclose"]
    restore_equal = comparisons[
        "native_partitioned_miss_vs_native_restored_hit"
    ]["allclose"]
    if not partition_equal and restore_equal:
        return "PARTITION_NON_EQUIVALENCE_RESTORE_EXACT"
    if partition_equal and not restore_equal:
        return "RESTORE_NON_EQUIVALENCE"
    if partition_equal and restore_equal:
        return "NATIVE_PATHS_EQUIVALENT"
    return "PARTITION_AND_RESTORE_NON_EQUIVALENCE"


def _cleanup_engine(engine, configuration):
    receipt = engine.exit()
    if (
        receipt.get("process_group_destroyed") is not True
        or receipt.get("rank_exit_codes")
        != [0] * len(configuration.gpu_indices)
        or receipt.get("owned_children_remaining") != []
    ):
        raise ValueError("diagnostic engine cleanup mismatch")
    return receipt


def run_full_phase(
    *,
    configuration,
    prompt_token_ids,
    generated_tokens,
    engine_factory=_default_engine_factory,
):
    engine = engine_factory(configuration)
    cleanup_receipt = None
    try:
        run = _run_request(
            engine,
            configuration,
            name="native_full",
            prompt_token_ids=prompt_token_ids,
            generated_tokens=generated_tokens,
        )
        cleanup_receipt = _cleanup_engine(engine, configuration)
        return {"run": run}
    finally:
        if cleanup_receipt is None:
            engine.exit()


def run_partition_phase(
    *,
    configuration,
    prompt_token_ids,
    generated_tokens,
    engine_factory=_default_engine_factory,
):
    engine = engine_factory(configuration)
    cleanup_receipt = None
    try:
        engine.configure_qwen35_hybrid_prefix_publication_runtime(
            model_fingerprint=configuration.model_fingerprint,
            max_entries=configuration.max_cache_entries,
            max_bytes=configuration.max_cache_bytes,
            timeout_s=configuration.timeout_s,
        )
        before_partitioned = _authority_snapshot(
            engine,
            configuration,
        )
        partitioned = _run_request(
            engine,
            configuration,
            name="native_partitioned_miss",
            prompt_token_ids=prompt_token_ids,
            generated_tokens=generated_tokens,
        )
        after_partitioned = _authority_snapshot(
            engine,
            configuration,
        )
        restored = _run_request(
            engine,
            configuration,
            name="native_restored_hit",
            prompt_token_ids=prompt_token_ids,
            generated_tokens=generated_tokens,
        )
        after_restored = _authority_snapshot(
            engine,
            configuration,
        )
        result = {
            "runs": [partitioned, restored],
            "cache_deltas": {
                "partitioned_miss": _counter_delta(
                    before_partitioned,
                    after_partitioned,
                ),
                "restored_hit": _counter_delta(
                    after_partitioned,
                    after_restored,
                ),
            },
        }
        cleanup_receipt = _cleanup_engine(engine, configuration)
        return result
    finally:
        if cleanup_receipt is None:
            engine.exit()


def merge_phase_artifacts(
    *,
    full_phase,
    partition_phase,
    prompt_tokens,
    generated_tokens,
    logits_comparator=_default_logits_comparator,
    official=None,
    official_reference=None,
):
    full = full_phase["run"]
    partitioned, restored = partition_phase["runs"]
    comparisons = {
        "native_full_vs_native_partitioned_miss": _comparison(
            logits_comparator,
            full["_step_logits"],
            partitioned["_step_logits"],
        ),
        "native_partitioned_miss_vs_native_restored_hit": _comparison(
            logits_comparator,
            partitioned["_step_logits"],
            restored["_step_logits"],
        ),
    }
    if official is not None:
        comparisons.update({
            "official_vs_native_full": _comparison(
                logits_comparator,
                official["step_logits"],
                full["_step_logits"],
            ),
            "official_vs_native_partitioned_miss": _comparison(
                logits_comparator,
                official["step_logits"],
                partitioned["_step_logits"],
            ),
            "official_vs_native_restored_hit": _comparison(
                logits_comparator,
                official["step_logits"],
                restored["_step_logits"],
            ),
        })
    runs = []
    for row in (full, partitioned, restored):
        runs.append({
            name: value
            for name, value in row.items()
            if name != "_step_logits"
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "classification": _classify(comparisons),
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "official_reference": (
            {"status": "not_requested"}
            if official_reference is None
            else official_reference
        ),
        "official_output_token_ids": (
            None
            if official is None
            else list(official["output_token_ids"])
        ),
        "runs": runs,
        "comparisons": comparisons,
        "cache_deltas": partition_phase["cache_deltas"],
    }


def run_diagnostic(
    *,
    configuration,
    prompt_token_ids,
    generated_tokens,
    engine_factory=_default_engine_factory,
    reference_executor_factory=None,
    logits_comparator=_default_logits_comparator,
):
    if reference_executor_factory is None:
        reference_executor_factory = (
            _default_reference_executor_factory(configuration)
        )
    reference_executor = reference_executor_factory()
    official = None
    official_reference = None
    try:
        try:
            official = (
                reference_executor.generate_reference_with_step_logits(
                    scenario="publish_source",
                    prompt_token_ids=list(prompt_token_ids),
                    generated_tokens=generated_tokens,
                    generation_policy={
                        "temperature": 0.0,
                        "ignore_eos": True,
                    },
                )
            )
        except Exception as error:
            official_reference = {
                "status": "unavailable",
                "error_type": type(error).__name__,
                "error_detail": str(error),
            }
        else:
            if (
                not isinstance(official, dict)
                or set(official)
                != {"output_token_ids", "step_logits"}
                or len(official["output_token_ids"])
                != generated_tokens
                or len(official["step_logits"])
                != generated_tokens
            ):
                raise ValueError(
                    "diagnostic official reference is invalid"
                )
            official_reference = {"status": "available"}
    finally:
        reference_executor.close()

    full_phase = run_full_phase(
        configuration=configuration,
        prompt_token_ids=prompt_token_ids,
        generated_tokens=generated_tokens,
        engine_factory=engine_factory,
    )
    partition_phase = run_partition_phase(
        configuration=configuration,
        prompt_token_ids=prompt_token_ids,
        generated_tokens=generated_tokens,
        engine_factory=engine_factory,
    )
    return merge_phase_artifacts(
        full_phase=full_phase,
        partition_phase=partition_phase,
        prompt_tokens=len(prompt_token_ids),
        generated_tokens=generated_tokens,
        logits_comparator=logits_comparator,
        official=official,
        official_reference=official_reference,
    )


def _load_configuration(path, source_inventory_path):
    driver = _load_module(
        "run_qwen35_tp4_engine_correctness_authority",
        "run_qwen35_tp4_engine_correctness_authority.py",
    )
    return driver.load_configuration(
        path,
        source_inventory_path=source_inventory_path,
    )


def _w1_prompt():
    contract = _load_module(
        "qwen35_tp4_cached_continuation_correctness_contract",
        "qwen35_tp4_cached_continuation_correctness_contract.py",
    )
    backend = _load_module(
        "qwen35_tp4_cached_continuation_backend_session",
        "qwen35_tp4_cached_continuation_backend_session.py",
    )
    payload = contract.workload_payload("w1_medium_reuse")
    return (
        backend._request_prompt(payload, 0),
        payload["spec"]["generated_tokens"],
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--source-inventory", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args(argv)
    output = Path(arguments.output)
    if output.exists():
        raise ValueError("diagnostic output already exists")
    prompt, generated_tokens = _w1_prompt()
    result = run_diagnostic(
        configuration=_load_configuration(
            arguments.configuration,
            arguments.source_inventory,
        ),
        prompt_token_ids=prompt,
        generated_tokens=generated_tokens,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(
        result,
        sort_keys=True,
        separators=(",", ":"),
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
