from __future__ import annotations

from contextlib import contextmanager
import importlib.util
from pathlib import Path


def _load_gate():
    path = (
        Path(__file__).resolve().parent
        / "qwen35_tp4_32k_h2d_slot_reuse_causal_diagnostic_gate.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_focused_h2d_slot_reuse_gate",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load focused H2D gate")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gate = _load_gate()
_MOVEMENT_INVENTORY_FIELDS = (
    "h2d_pair_inventory",
    "h2d_span_inventory",
    "d2h_pair_inventory",
    "d2h_span_inventory",
)


def _load_frozen_32k_worker():
    path = (
        Path(__file__).resolve().parent
        / "qwen35_native_mtp_tp4_32k_target_kv_offload_worker.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_focused_h2d_frozen_32k_worker",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load frozen 32K worker")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def capture_ordinary_target_forwards(engine):
    model_runner = engine.model_runner
    original_call = model_runner.call
    capture = {
        "ordinary_decode_target_forward_calls": 0,
    }

    def recorded_call(method_name, *args, **kwargs):
        result = original_call(method_name, *args, **kwargs)
        if method_name == "run":
            capture[
                "ordinary_decode_target_forward_calls"
            ] += 1
        return result

    model_runner.call = recorded_call
    try:
        yield capture
    finally:
        model_runner.call = original_call


def _cleanup_preserving_primary(primary, cleanup_calls):
    first_cleanup_error = None
    for label, callback in cleanup_calls:
        try:
            callback()
        except BaseException as error:
            if primary is not None:
                add_note = getattr(primary, "add_note", None)
                if callable(add_note):
                    add_note(
                        f"{label} cleanup failed: "
                        f"{type(error).__name__}: {error}"
                    )
            elif first_cleanup_error is None:
                first_cleanup_error = error
    if primary is None and first_cleanup_error is not None:
        raise first_cleanup_error


def _validated_baseline_observation(
    observation,
    *,
    before: int,
    after: int,
    prediction_state: dict,
) -> dict:
    if not isinstance(observation, dict):
        raise RuntimeError("engine step observation is unavailable")
    row = dict(observation)
    if after < before:
        raise RuntimeError("target forward count regressed")
    if after - before != 1:
        raise RuntimeError(
            "focused diagnostic requires one ordinary target forward"
        )
    if row.get("speculative_selected_seq_ids") not in ([], ()):
        raise RuntimeError(
            "focused diagnostic forbids speculative selection"
        )
    zero_fields = (
        "speculative_proposal_row_count",
        "speculative_first_target_callback_count",
        "speculative_fixed_q_group_count",
    )
    if any(row.get(field, 0) != 0 for field in zero_fields):
        raise RuntimeError("proposal callback is forbidden")
    empty_fields = (
        "speculative_output_token_counts",
        "speculative_accepted_draft_token_counts",
        "speculative_proposal_token_counts",
        "speculative_proposal_token_ids_by_seq",
        "speculative_accepted_draft_token_ids_by_seq",
    )
    if any(row.get(field, {}) not in ({},) for field in empty_fields):
        raise RuntimeError("speculative token evidence is forbidden")
    scheduled = row.get("scheduled")
    if not isinstance(scheduled, list) or not scheduled:
        raise RuntimeError("scheduled sequence evidence is unavailable")
    if row.get("batch_kind") == "mixed":
        raise RuntimeError("focused diagnostic forbids mixed batches")
    sequence_to_prompt = prediction_state[
        "sequence_to_prompt"
    ]
    prompt_tokens = prediction_state["prompt_tokens"]
    completion_tokens = prediction_state["completion_tokens"]
    for scheduled_row in scheduled:
        if not isinstance(scheduled_row, dict):
            raise RuntimeError("scheduled sequence row is invalid")
        sequence_id = int(scheduled_row["seq_id"])
        if sequence_id not in sequence_to_prompt:
            prompt_index = len(sequence_to_prompt)
            if prompt_index >= len(prompt_tokens):
                raise RuntimeError(
                    "scheduled sequence inventory exceeds prompts"
                )
            sequence_to_prompt[sequence_id] = prompt_index
            completion_tokens[sequence_id] = []
    raw_deltas = row.get("new_completion_tokens_by_seq")
    if not isinstance(raw_deltas, dict):
        raise RuntimeError("completion token deltas are unavailable")
    token_deltas = {
        int(sequence_id): [
            int(token_id) for token_id in token_ids
        ]
        for sequence_id, token_ids in raw_deltas.items()
    }
    scheduled_ids = {
        int(scheduled_row["seq_id"])
        for scheduled_row in scheduled
    }
    if not set(token_deltas).issubset(scheduled_ids):
        raise RuntimeError(
            "completion token delta references unscheduled sequence"
        )
    prediction_rows = []
    for scheduled_row in scheduled:
        sequence_id = int(scheduled_row["seq_id"])
        do_sample = bool(scheduled_row.get("do_sample", False))
        delta = token_deltas.get(sequence_id, [])
        if not do_sample:
            if delta:
                raise RuntimeError(
                    "non-sampling sequence produced completion tokens"
                )
            continue
        if len(delta) != 1:
            raise RuntimeError(
                "ordinary baseline must produce one token per row"
            )
        prompt_index = sequence_to_prompt[sequence_id]
        prior_completion = completion_tokens[sequence_id]
        input_tokens = (
            prior_completion
            if prior_completion
            else prompt_tokens[prompt_index]
        )
        if not input_tokens:
            raise RuntimeError("prediction input token is unavailable")
        position = (
            len(prompt_tokens[prompt_index])
            + len(prior_completion)
            - 1
        )
        prediction_rows.append({
            "sequence_id": sequence_id,
            "prompt_index": prompt_index,
            "prediction_index": len(prior_completion),
            "input_token_id": int(input_tokens[-1]),
            "position": position,
            "context_length": position + 1,
        })
    for sequence_id, delta in token_deltas.items():
        completion_tokens[sequence_id].extend(delta)
    row["execution_mode"] = "baseline"
    row["proposal_callback_count"] = 0
    row["shadow_target_forward_count"] = 0
    row["prediction_rows"] = prediction_rows
    row["authority_normal_decode_target_forward_calls"] = 1
    return row


def _ordinary_forward_count(capture) -> int:
    if capture is None:
        raise RuntimeError(
            "ordinary target-forward capture is required"
        )
    return int(
        capture.get(
            "ordinary_decode_target_forward_calls",
            0,
        )
    )


def _ordinary_attention_stage(engine) -> str:
    scheduler = getattr(engine, "scheduler", None)
    if scheduler is None:
        raise RuntimeError(
            "focused diagnostic scheduler state is unavailable"
        )
    if (
        bool(getattr(scheduler, "waiting", ()))
        or bool(getattr(scheduler, "prefilling", ()))
    ):
        return "prefill"
    if bool(getattr(scheduler, "running", ())):
        return "decode"
    raise RuntimeError(
        "focused diagnostic scheduler has no runnable stage"
    )


def run_generation_with_h2d_slot_reuse_diagnostic(
    *,
    engine,
    prompt_rows,
    sampling_params,
    synchronize,
    mode,
    batch_size,
    repetition,
    timing_epsilon_ms,
    target_forward_capture=None,
):
    gate.cell_key(mode, batch_size)
    if timing_epsilon_ms != gate.TIMING_EPSILON_MS:
        raise ValueError("timing epsilon is frozen at 0.2 ms")
    if len(prompt_rows) != batch_size:
        raise ValueError("prompt inventory does not match batch size")
    prediction_state = {
        "prompt_tokens": [
            [int(token_id) for token_id in row["token_ids"]]
            for row in prompt_rows
        ],
        "sequence_to_prompt": {},
        "completion_tokens": {},
    }
    if any(
        not token_ids
        for token_ids in prediction_state["prompt_tokens"]
    ):
        raise ValueError("focused diagnostic prompts must be nonempty")
    outputs_by_id = {}
    compact_logits = []
    observations = []
    slot_rows = None
    movement_inventory_rows = None
    target_forward_count = 0
    primary = None
    configured = False
    logits_enabled = False
    try:
        engine.configure_h2d_slot_reuse_diagnostic(
            mode,
            timeout_s=60.0,
        )
        configured = True
        engine.enable_step_logits_authority_recording(
            True,
            timeout_s=60.0,
        )
        logits_enabled = True
        for row in prompt_rows:
            engine.add_request(
                row["token_ids"],
                sampling_params,
            )
        engine_step = 0
        while not engine.is_finished():
            attention_stage = _ordinary_attention_stage(engine)
            engine.set_h2d_slot_reuse_diagnostic_context(
                engine_step,
                attention_stage,
                timeout_s=60.0,
            )
            before = _ordinary_forward_count(
                target_forward_capture
            )
            step_outputs, _ = engine.step()
            synchronize()
            after = _ordinary_forward_count(
                target_forward_capture
            )
            observation = _validated_baseline_observation(
                engine.last_step_observation,
                before=before,
                after=after,
                prediction_state=prediction_state,
            )
            target_forward_count += int(
                observation[
                    "authority_normal_decode_target_forward_calls"
                ]
            )
            prediction_indices = {
                int(row["prediction_index"])
                for row in observation["prediction_rows"]
            }
            if prediction_indices.intersection((0, 1)):
                if (
                    len(prediction_indices) != 1
                    or not prediction_indices.issubset({0, 1})
                ):
                    raise RuntimeError(
                        "focused prediction index inventory is invalid"
                    )
                prediction_index = next(iter(prediction_indices))
                compact_logits.extend(
                    gate.compact_prediction_logits(
                        engine.read_step_logits_authority(),
                        observation=observation,
                        prediction_index=prediction_index,
                        top_k=gate.TOP_K,
                    )
                )
            observations.append(observation)
            for sequence_id, token_ids in step_outputs:
                outputs_by_id[int(sequence_id)] = [
                    int(token_id) for token_id in token_ids
                ]
            engine_step += 1
        engine.flush_pending_hybrid_state_releases(
            timeout_s=60.0,
        )
        slot_rows = engine.drain_h2d_slot_reuse_diagnostic(
            timing_epsilon_ms=timing_epsilon_ms,
            expected_mode=mode,
            timeout_s=60.0,
        )
        movement_inventory_rows = engine.kv_offload_summaries(
            timeout_s=60.0,
        )
    except BaseException as error:
        primary = error
        raise
    finally:
        cleanup_calls = []
        if logits_enabled:
            cleanup_calls.append((
                "step-logit recording",
                lambda: engine.enable_step_logits_authority_recording(
                    False,
                    timeout_s=60.0,
                ),
            ))
        if configured:
            cleanup_calls.append((
                "H2D diagnostic disable",
                lambda: engine.configure_h2d_slot_reuse_diagnostic(
                    "off",
                    timeout_s=60.0,
                ),
            ))
        _cleanup_preserving_primary(primary, cleanup_calls)
    if len(compact_logits) != batch_size * 2:
        raise RuntimeError(
            "focused compact logit inventory is incomplete"
        )
    output_rows = []
    sequence_to_prompt = prediction_state["sequence_to_prompt"]
    for sequence_id, prompt_index in sorted(
        sequence_to_prompt.items(),
        key=lambda item: item[1],
    ):
        if sequence_id not in outputs_by_id:
            raise RuntimeError(
                "finished output is missing for scheduled sequence"
            )
        token_ids = outputs_by_id[sequence_id]
        if len(token_ids) != gate.MAX_OUTPUT_TOKENS:
            raise RuntimeError("engine output token count mismatch")
        output_rows.append({
            "prompt_index": prompt_index,
            "sequence_id": sequence_id,
            "token_count": len(token_ids),
            "token_ids": token_ids,
            "sha256": gate._json_sha256(token_ids),
        })
    if len(output_rows) != len(prompt_rows):
        raise RuntimeError(
            "engine output inventory does not match prompts"
        )
    return {
        "mode": mode,
        "batch_size": batch_size,
        "repetition": repetition,
        "output_rows": output_rows,
        "compact_logit_rows": compact_logits,
        "rank_slot_rows": list(slot_rows),
        "movement_inventory_rows": [
            {
                "rank": rank,
                **{
                    field: [
                        list(item)
                        for item in summary[field]
                    ]
                    for field in _MOVEMENT_INVENTORY_FIELDS
                },
            }
            for rank, summary in enumerate(
                movement_inventory_rows
            )
        ],
        "step_observations": observations,
        "target_forward_count": target_forward_count,
    }


def collect_runtime_metadata(
    *,
    torch_module,
    driver_version,
) -> dict:
    torch_version = str(
        getattr(torch_module, "__version__", "")
    )
    cuda_version = str(
        getattr(getattr(torch_module, "version", None), "cuda", "")
        or ""
    )
    driver_version = str(driver_version or "")
    device_names = [
        str(torch_module.cuda.get_device_name(index))
        for index in range(gate.WORLD_SIZE)
    ]
    if (
        not torch_version
        or not cuda_version
        or not driver_version
        or any(not name for name in device_names)
    ):
        raise RuntimeError(
            "required PyTorch/CUDA/driver metadata is unavailable"
        )
    return {
        "torch_version": torch_version,
        "torch_cuda_runtime_version": cuda_version,
        "nvidia_driver_version": driver_version,
        "cuda_device_names": device_names,
    }


def build_repetition_artifact(**fields) -> dict:
    row = dict(fields)
    row["schema"] = gate.SCHEMA
    row["policy"] = gate.POLICY
    digest_input = dict(row)
    row["cell_digest_sha256"] = gate._json_sha256(
        digest_input
    )
    return gate.validate_repetition(row)


def _movement_rows_with_inventories(
    movement_rows,
    inventory_rows,
) -> list[dict]:
    if (
        not isinstance(movement_rows, list)
        or not isinstance(inventory_rows, list)
        or len(movement_rows) != gate.WORLD_SIZE
        or len(inventory_rows) != gate.WORLD_SIZE
    ):
        raise RuntimeError(
            "focused movement rank inventory mismatch"
        )
    merged = []
    for rank, (movement, inventory) in enumerate(
        zip(movement_rows, inventory_rows)
    ):
        if (
            not isinstance(movement, dict)
            or not isinstance(inventory, dict)
            or movement.get("rank") != rank
            or inventory.get("rank") != rank
        ):
            raise RuntimeError(
                "focused movement rank row mismatch"
            )
        row = dict(movement)
        for field in _MOVEMENT_INVENTORY_FIELDS:
            values = inventory.get(field)
            if not isinstance(values, list):
                raise RuntimeError(
                    f"focused movement {field} is unavailable"
                )
            row[field] = [
                list(item) for item in values
            ]
        merged.append(row)
    return merged


def run_focused_repetition(
    *,
    model_path,
    gpu_indices,
    policy,
    mode,
    batch_size,
    repetition,
    dist_port,
    master_port,
    frozen_dependencies,
    torch_module,
    driver_version,
    repo_root,
    frozen_worker_loader=_load_frozen_32k_worker,
    source_digest_fn=gate.source_tree_sha256,
    runtime_metadata_collector=collect_runtime_metadata,
) -> dict:
    if policy != gate.POLICY:
        raise ValueError(
            "focused diagnostic policy must be baseline"
        )
    gate.cell_key(mode, batch_size)
    if not isinstance(frozen_dependencies, dict):
        raise TypeError("frozen dependencies must be a mapping")
    frozen_worker = frozen_worker_loader()
    generation_rows = []

    def focused_generation(
        *,
        engine,
        prompt_rows,
        sampling_params,
        synchronize,
        target_forward_capture=None,
    ):
        if target_forward_capture is not None:
            raise RuntimeError(
                "focused baseline worker owns target-forward capture"
            )
        with capture_ordinary_target_forwards(engine) as capture:
            result = run_generation_with_h2d_slot_reuse_diagnostic(
                engine=engine,
                prompt_rows=prompt_rows,
                sampling_params=sampling_params,
                synchronize=synchronize,
                mode=mode,
                batch_size=batch_size,
                repetition=repetition,
                timing_epsilon_ms=gate.TIMING_EPSILON_MS,
                target_forward_capture=capture,
            )
        generation_rows.append(result)
        return result["output_rows"], result["step_observations"]

    cell = frozen_worker.run_policy_cell(
        model_path=model_path,
        gpu_indices=tuple(gpu_indices),
        policy=gate.POLICY,
        batch_size=batch_size,
        dist_port=dist_port,
        master_port=master_port,
        run_generation_fn=focused_generation,
        **frozen_dependencies,
    )
    if len(generation_rows) != 1:
        raise RuntimeError(
            "frozen baseline cell generation inventory mismatch"
        )
    generation = generation_rows[0]
    if (
        cell.get("policy") != gate.POLICY
        or cell.get("batch_size") != batch_size
    ):
        raise RuntimeError(
            "frozen baseline cell identity mismatch"
        )
    model_identity = cell.get("model_identity")
    if not isinstance(model_identity, dict):
        raise RuntimeError(
            "frozen baseline model identity is unavailable"
        )
    checkpoint_sha256 = model_identity.get(
        "target_model_manifest_sha256"
    )
    if (
        not isinstance(checkpoint_sha256, str)
        or not checkpoint_sha256
    ):
        raise RuntimeError(
            "frozen baseline checkpoint digest is unavailable"
        )
    runtime_metadata = runtime_metadata_collector(
        torch_module=torch_module,
        driver_version=driver_version,
    )
    return build_repetition_artifact(
        mode=mode,
        batch_size=batch_size,
        repetition=repetition,
        world_size=gate.WORLD_SIZE,
        prompt_tokens=gate.PROMPT_TOKENS,
        max_output_tokens=gate.MAX_OUTPUT_TOKENS,
        max_proposal_tokens=gate.MAX_PROPOSAL_TOKENS,
        block_size=gate.BLOCK_SIZE,
        gpu_blocks=gate.GPU_BLOCKS,
        logical_blocks=gate.LOGICAL_BLOCKS,
        blockwise_blocks=gate.BLOCKWISE_BLOCKS,
        async_copy=True,
        batch_copy=True,
        writeback_on_evict=False,
        enforce_eager=True,
        **runtime_metadata,
        source_tree_sha256=source_digest_fn(repo_root),
        checkpoint_sha256=checkpoint_sha256,
        timing_epsilon_ms=gate.TIMING_EPSILON_MS,
        prompt_rows=list(cell["prompt_rows"]),
        output_rows=list(generation["output_rows"]),
        compact_logit_rows=list(
            generation["compact_logit_rows"]
        ),
        rank_slot_rows=list(generation["rank_slot_rows"]),
        step_observations=list(
            generation["step_observations"]
        ),
        target_forward_count=int(
            generation["target_forward_count"]
        ),
        kv_rank_deltas=_movement_rows_with_inventories(
            list(cell["kv_rank_deltas"]),
            list(generation["movement_inventory_rows"]),
        ),
        kv_capacity_rows=list(cell["kv_capacity_rows"]),
        cleanup=dict(cell["cleanup"]),
    )


def run_focused_campaign(
    *,
    repetitions,
    repetition_runner=run_focused_repetition,
    **common,
) -> dict:
    if (
        isinstance(repetitions, bool)
        or not isinstance(repetitions, int)
        or repetitions <= 0
    ):
        raise ValueError("repetitions must be a positive integer")
    if common.get("policy") != gate.POLICY:
        raise ValueError(
            "focused diagnostic policy must be baseline"
        )
    cells = {}
    for mode in gate.MODES:
        for batch_size in gate.BATCH_SIZES:
            key = gate.cell_key(mode, batch_size)
            cells[key] = [
                repetition_runner(
                    mode=mode,
                    batch_size=batch_size,
                    repetition=repetition,
                    **common,
                )
                for repetition in range(repetitions)
            ]
    return gate.validate_artifact({
        "schema": gate.SCHEMA,
        "cells": cells,
    })
