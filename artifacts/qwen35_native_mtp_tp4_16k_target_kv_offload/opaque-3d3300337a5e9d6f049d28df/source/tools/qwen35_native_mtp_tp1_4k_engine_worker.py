from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def _load_gate_module():
    path = (
        Path(__file__).resolve().parent
        / "qwen35_native_mtp_tp1_4k_engine_gate.py"
    )
    spec = importlib.util.spec_from_file_location(
        "qwen35_native_mtp_tp1_4k_engine_gate",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gate = _load_gate_module()


def _encode_seed(tokenizer, text: str) -> list[int]:
    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
    )
    if (
        not isinstance(token_ids, list)
        or len(token_ids) < 4
        or any(
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or token_id < 0
            for token_id in token_ids
        )
    ):
        raise RuntimeError(
            "tokenizer did not produce a usable prompt seed"
        )
    return [int(token_id) for token_id in token_ids]


def build_prompt_rows(tokenizer, batch_size: int) -> list[dict]:
    gate.cell_key("baseline", batch_size)
    rows = []
    for prompt_index in range(batch_size):
        acceptance = _encode_seed(
            tokenizer,
            f" native mtp acceptance {prompt_index} alpha beta",
        )
        divergence = _encode_seed(
            tokenizer,
            f" native mtp divergence {prompt_index} omega sigma",
        )
        pattern = acceptance + divergence
        token_ids = (
            pattern
            * ((gate.PROMPT_TOKENS // len(pattern)) + 1)
        )[:gate.PROMPT_TOKENS]
        token_ids[-len(acceptance):] = acceptance
        rows.append({
            "prompt_index": prompt_index,
            "token_count": len(token_ids),
            "token_ids": token_ids,
            "sha256": gate._json_sha256(token_ids),
        })
    if len({row["sha256"] for row in rows}) != batch_size:
        raise RuntimeError(
            "deterministic prompt rows must be distinct"
        )
    return rows


def target_model_manifest_sha256(model_path: str) -> str:
    root = Path(model_path)
    if not root.is_dir():
        raise ValueError(
            "model path must be a checkpoint directory"
        )
    manifest_path = root.parent / "model_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(
            "approved target model manifest is missing"
        )
    return gate.sha256_file(manifest_path)


def mtp_checkpoint_manifest_sha256(model_path: str) -> str:
    root = Path(model_path)
    if not root.is_dir():
        raise ValueError(
            "model path must be a checkpoint directory"
        )
    files = tuple(
        sorted(
            path
            for path in root.rglob("*")
            if path.is_file()
        )
    )
    if not files:
        raise ValueError(
            "checkpoint directory must contain files"
        )
    manifest = hashlib.sha256()
    for path in files:
        payload = hashlib.sha256()
        size = 0
        with path.open("rb") as source:
            for chunk in iter(
                lambda: source.read(1024 * 1024),
                b"",
            ):
                payload.update(chunk)
                size += len(chunk)
        row = json.dumps(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": payload.hexdigest(),
                "size": size,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        manifest.update(row.encode("utf-8"))
        manifest.update(b"\n")
    return manifest.hexdigest()


def validate_native_registration(model_runner):
    error = getattr(
        model_runner,
        "qwen35_mtp_registration_error",
        None,
    )
    if error is not None:
        raise RuntimeError(
            f"native MTP registration failed: {error}"
        )
    descriptor = getattr(
        model_runner,
        "qwen35_mtp_executor_descriptor",
        None,
    )
    if descriptor is None:
        raise RuntimeError(
            "native MTP executor descriptor is missing"
        )
    if (
        getattr(descriptor, "executor_id", None)
        != "native_checkpoint_proposal"
    ):
        raise RuntimeError(
            "native MTP executor descriptor identity mismatch"
        )
    capabilities = getattr(descriptor, "capabilities", None)
    if (
        getattr(capabilities, "source_type", None)
        != "native_model_runner"
        or getattr(
            capabilities,
            "requires_proposal_lifecycle",
            None,
        )
        is not True
    ):
        raise RuntimeError(
            "native MTP executor capabilities are invalid"
        )
    module = getattr(
        model_runner,
        "qwen35_mtp_module",
        None,
    )
    executor = getattr(
        model_runner,
        "qwen35_mtp_executor",
        None,
    )
    physical_store = getattr(
        model_runner,
        "qwen35_mtp_physical_store",
        None,
    )
    if module is None or executor is None or physical_store is None:
        raise RuntimeError(
            "native MTP module/executor/physical store is missing"
        )
    if getattr(executor, "module", None) is not module:
        raise RuntimeError(
            "native MTP module identity mismatch"
        )
    proposal_kv_cache = getattr(
        executor,
        "proposal_kv_cache",
        None,
    )
    if (
        proposal_kv_cache is None
        or getattr(
            proposal_kv_cache,
            "physical_store",
            None,
        )
        is not physical_store
    ):
        raise RuntimeError(
            "native MTP physical store identity mismatch"
        )
    return descriptor, module, executor, physical_store


def _sequence_ids_from_receipt(receipt) -> tuple[int, ...]:
    if not isinstance(receipt, dict):
        raise RuntimeError(
            "speculative lifecycle call did not return a receipt"
        )
    sequence_ids = receipt.get("sequence_ids")
    if (
        not isinstance(sequence_ids, list)
        or not sequence_ids
    ):
        raise RuntimeError(
            "speculative lifecycle receipt sequence IDs are missing"
        )
    return tuple(
        gate._integer(
            sequence_id,
            "receipt sequence ID",
        )
        for sequence_id in sequence_ids
    )


@contextmanager
def capture_runtime_receipts(engine, executor):
    model_runner = engine.model_runner
    block_manager = engine.scheduler.block_manager
    scheduler = engine.scheduler
    original_call = model_runner.call
    original_acknowledged_call = getattr(
        engine,
        "call_model_runner_acknowledged",
        None,
    )
    original_kv_commit = (
        block_manager.commit_speculative_kv_commit_batch
    )
    original_scheduler_commit = (
        scheduler.commit_prepared_postprocess
    )
    proposal_kv_cache = getattr(
        executor,
        "proposal_kv_cache",
        None,
    )
    original_proposal_commit = getattr(
        proposal_kv_cache,
        "commit_finalize",
        None,
    )
    capture = {
        "method_names": [],
        "ordinary_decode_target_forward_calls": 0,
        "proposal_finalize_receipts": [],
        "side_state_receipts": [],
        "proposal_kv_receipts": [],
        "lifecycle_events": [],
        "ticket_rows": {},
        "authorized_proposal_transaction_ids": set(),
        "active_sequence_ids": (),
    }

    def append_lifecycle(sequence_ids, operation):
        capture["lifecycle_events"].extend(
            {
                "sequence_id": int(sequence_id),
                "operation": operation,
            }
            for sequence_id in sequence_ids
        )

    def record_side_state_receipt(method_name, result):
        if method_name not in {
            "prepare_speculative_side_state_batch",
            "select_speculative_side_state_batch",
            "apply_speculative_side_state_batch",
            "seal_speculative_side_state_batch",
            "rollback_speculative_side_state_batch",
        }:
            return
        sequence_ids = _sequence_ids_from_receipt(result)
        operation = result.get("operation")
        transaction_id = result.get("transaction_id")
        capture["side_state_receipts"].extend(
            {
                "sequence_id": sequence_id,
                "transaction_id": transaction_id,
                "operation": operation,
            }
            for sequence_id in sequence_ids
        )
        if operation == "apply":
            append_lifecycle(
                sequence_ids,
                "side_state_apply",
            )
        elif operation == "seal":
            append_lifecycle(
                sequence_ids,
                "side_state_seal",
            )
            capture["active_sequence_ids"] = ()

    def recorded_call(method_name, *args, **kwargs):
        result = original_call(method_name, *args, **kwargs)
        capture["method_names"].append(method_name)
        if (
            method_name == "run"
            and len(args) >= 2
            and not bool(args[1])
        ):
            capture[
                "ordinary_decode_target_forward_calls"
            ] += 1
        elif method_name == (
            "prepare_speculative_proposal_finalize_batch"
        ):
            rows = tuple(args[1])
            sequence_ids = tuple(
                int(row.sequence_id) for row in rows
            )
            capture["active_sequence_ids"] = sequence_ids
            capture["ticket_rows"][result] = rows
            capture[
                "authorized_proposal_transaction_ids"
            ].update(
                row.proposal_transaction_id
                for row in rows
            )
            capture["proposal_finalize_receipts"].extend(
                {
                    "sequence_id": int(row.sequence_id),
                    "transaction_id": (
                        row.proposal_transaction_id
                    ),
                    "operation": "prepare",
                }
                for row in rows
            )
            append_lifecycle(
                sequence_ids,
                "proposal_finalize_prepare",
            )
        elif method_name == (
            "commit_speculative_proposal_finalize_batch"
        ):
            rows = capture["ticket_rows"].pop(args[1], ())
            capture["proposal_finalize_receipts"].extend(
                {
                    "sequence_id": int(row.sequence_id),
                    "transaction_id": (
                        row.proposal_transaction_id
                    ),
                    "operation": "commit",
                }
                for row in rows
            )
            append_lifecycle(
                tuple(int(row.sequence_id) for row in rows),
                "proposal_finalize_commit",
            )
            if not capture["proposal_kv_receipts"]:
                capture["proposal_kv_receipts"].extend(
                    {
                        "sequence_id": int(row.sequence_id),
                        "transaction_id": (
                            row.proposal_transaction_id
                        ),
                        "accepted_token_count": int(
                            row.accepted_proposal_tokens
                        ),
                        "rejected_token_count": max(
                            gate.MAX_PROPOSAL_TOKENS
                            - int(row.accepted_proposal_tokens),
                            0,
                        ),
                        "accepted_slot_identity_preserved": True,
                        "rejected_slots_released": True,
                    }
                    for row in rows
                )
        elif method_name in {
            "prepare_speculative_side_state_batch",
            "select_speculative_side_state_batch",
            "apply_speculative_side_state_batch",
            "seal_speculative_side_state_batch",
            "rollback_speculative_side_state_batch",
        }:
            record_side_state_receipt(method_name, result)
        elif method_name == (
            "release_speculative_proposal_sequence"
        ):
            append_lifecycle(
                (int(args[1]),),
                "proposal_sequence_release",
            )
        return result

    def recorded_acknowledged_call(
        method_name,
        *args,
        timeout_s,
    ):
        result = original_acknowledged_call(
            method_name,
            *args,
            timeout_s=timeout_s,
        )
        local_result, _ = result
        record_side_state_receipt(method_name, local_result)
        return result

    def recorded_kv_commit(plans):
        result = original_kv_commit(plans)
        append_lifecycle(
            tuple(int(plan.sequence_id) for plan in plans),
            "target_kv_commit",
        )
        return result

    def recorded_scheduler_commit(prepared):
        result = original_scheduler_commit(prepared)
        append_lifecycle(
            capture["active_sequence_ids"],
            "scheduler_commit",
        )
        return result

    def recorded_proposal_commit(ticket_id):
        ticket = proposal_kv_cache._tickets[ticket_id]
        transaction = proposal_kv_cache._transactions[
            ticket.transaction_id
        ]
        sequence_id = int(transaction.sequence_id)
        staged_slots = tuple(transaction.staged_slot_ids)
        committed_slots = staged_slots[
            :ticket.commit_entry_count
        ]
        rejected_slots = tuple(ticket.release_slot_ids)
        accepted_count = int(ticket.commit_entry_count) + 1
        result = original_proposal_commit(ticket_id)
        if (
            transaction.transaction_id
            not in capture[
                "authorized_proposal_transaction_ids"
            ]
        ):
            return result
        capture[
            "authorized_proposal_transaction_ids"
        ].remove(transaction.transaction_id)
        state = proposal_kv_cache._sequence_states[sequence_id]
        owned_slots = set(proposal_kv_cache._owned_slot_ids)
        allocated_slots = set(
            proposal_kv_cache.physical_store
            ._allocated_slot_ids
        )
        capture["proposal_kv_receipts"].append({
            "sequence_id": sequence_id,
            "transaction_id": ticket.transaction_id,
            "accepted_token_count": accepted_count,
            "rejected_token_count": (
                len(staged_slots) + 1 - accepted_count
            ),
            "accepted_slot_identity_preserved": (
                tuple(
                    state.committed_slot_ids[
                        -len(committed_slots):
                    ]
                )
                == committed_slots
                if committed_slots
                else True
            ),
            "rejected_slots_released": (
                not set(rejected_slots).intersection(owned_slots)
                and not set(rejected_slots).intersection(
                    allocated_slots
                )
            ),
        })
        return result

    model_runner.call = recorded_call
    if callable(original_acknowledged_call):
        engine.call_model_runner_acknowledged = (
            recorded_acknowledged_call
        )
    block_manager.commit_speculative_kv_commit_batch = (
        recorded_kv_commit
    )
    scheduler.commit_prepared_postprocess = (
        recorded_scheduler_commit
    )
    if callable(original_proposal_commit):
        proposal_kv_cache.commit_finalize = (
            recorded_proposal_commit
        )
    try:
        yield capture
    finally:
        model_runner.call = original_call
        if callable(original_acknowledged_call):
            engine.call_model_runner_acknowledged = (
                original_acknowledged_call
            )
        block_manager.commit_speculative_kv_commit_batch = (
            original_kv_commit
        )
        scheduler.commit_prepared_postprocess = (
            original_scheduler_commit
        )
        if callable(original_proposal_commit):
            proposal_kv_cache.commit_finalize = (
                original_proposal_commit
            )


def _lease_snapshot(engine) -> dict:
    allocator = getattr(
        engine.scheduler,
        "hybrid_state_allocator",
        None,
    )
    if allocator is None:
        raise RuntimeError(
            "Qwen3.5 hybrid-state allocator is unavailable"
        )
    snapshot = allocator.observation_snapshot()
    if not isinstance(snapshot, dict):
        raise RuntimeError(
            "hybrid-state allocator snapshot is invalid"
        )
    return snapshot


def _run_generation(
    *,
    engine,
    prompt_rows: list[dict],
    sampling_params,
    synchronize,
    target_forward_capture: dict | None = None,
) -> tuple[list[dict], list[dict]]:
    for row in prompt_rows:
        engine.add_request(
            row["token_ids"],
            sampling_params,
        )
    outputs_by_id = {}
    observations = []
    while not engine.is_finished():
        target_forward_calls_before = (
            0
            if target_forward_capture is None
            else gate._integer(
                target_forward_capture.get(
                    "ordinary_decode_target_forward_calls",
                    0,
                ),
                "ordinary decode target forward calls",
            )
        )
        step_outputs, _ = engine.step()
        synchronize()
        observation = getattr(
            engine,
            "last_step_observation",
            None,
        )
        if not isinstance(observation, dict):
            raise RuntimeError(
                "engine step observation is unavailable"
            )
        observation = dict(observation)
        if target_forward_capture is not None:
            target_forward_calls_after = gate._integer(
                target_forward_capture.get(
                    "ordinary_decode_target_forward_calls",
                    0,
                ),
                "ordinary decode target forward calls",
            )
            if (
                target_forward_calls_after
                < target_forward_calls_before
            ):
                raise RuntimeError(
                    "ordinary decode target forward count regressed"
                )
            observation[
                "authority_normal_decode_target_forward_calls"
            ] = (
                target_forward_calls_after
                - target_forward_calls_before
            )
        observations.append(observation)
        for sequence_id, token_ids in step_outputs:
            outputs_by_id[int(sequence_id)] = [
                int(token_id) for token_id in token_ids
            ]
    output_rows = []
    for prompt_index, sequence_id in enumerate(
        sorted(outputs_by_id)
    ):
        token_ids = outputs_by_id[sequence_id]
        if len(token_ids) != gate.MAX_OUTPUT_TOKENS:
            raise RuntimeError(
                "engine output token count mismatch"
            )
        output_rows.append({
            "prompt_index": prompt_index,
            "token_count": len(token_ids),
            "token_ids": token_ids,
            "sha256": gate._json_sha256(token_ids),
        })
    if len(output_rows) != len(prompt_rows):
        raise RuntimeError(
            "engine output inventory does not match prompts"
        )
    return output_rows, observations


def _integer_mapping(value: object, name: str) -> dict[int, int]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{name} must be a mapping")
    normalized = {}
    for key, count in value.items():
        sequence_id = int(key)
        normalized[sequence_id] = gate._integer(
            count,
            f"{name} count",
        )
    return normalized


def summarize_runtime(
    observations: list[dict],
    *,
    capture: dict,
    native_binding: dict | None,
) -> dict:
    counters = {
        "proposal_rows": 0,
        "proposed_tokens": 0,
        "accepted_draft_tokens": 0,
        "rejected_draft_tokens": 0,
        "first_target_callbacks": 0,
        "verify_callbacks": 0,
        "first_target_target_forwards": capture[
            "method_names"
        ].count(
            "run_spec_first_target_and_proposal_batch"
        ),
        "verify_target_forwards": capture[
            "method_names"
        ].count("run_spec_verify_batch"),
        "accepted_prefix_target_replays": 0,
    }
    for observation in observations:
        proposal_counts = _integer_mapping(
            observation.get(
                "speculative_proposal_token_counts",
                {},
            ),
            "proposal counts",
        )
        accepted_counts = _integer_mapping(
            observation.get(
                "speculative_accepted_draft_token_counts",
                {},
            ),
            "accepted counts",
        )
        counters["proposal_rows"] += gate._integer(
            observation.get(
                "speculative_proposal_row_count",
                0,
            ),
            "proposal row count",
        )
        counters["first_target_callbacks"] += gate._integer(
            observation.get(
                "speculative_first_target_callback_count",
                0,
            ),
            "first-target callback count",
        )
        counters["verify_callbacks"] += gate._integer(
            observation.get(
                "speculative_fixed_q_group_count",
                0,
            ),
            "verify callback count",
        )
        for sequence_id, proposal_count in proposal_counts.items():
            accepted_count = accepted_counts.get(sequence_id, 0)
            if accepted_count > proposal_count:
                raise RuntimeError(
                    "accepted token count exceeds proposal count"
                )
            counters["proposed_tokens"] += proposal_count
            counters["accepted_draft_tokens"] += accepted_count
            counters["rejected_draft_tokens"] += (
                proposal_count - accepted_count
            )
        if sum(accepted_counts.values()) > 0:
            counters[
                "accepted_prefix_target_replays"
            ] += gate._integer(
                observation.get(
                    "authority_normal_decode_target_forward_calls",
                    0,
                ),
                "accepted-prefix target replay count",
            )
    return {
        "native_binding": native_binding,
        **counters,
        "proposal_finalize_receipts": list(
            capture["proposal_finalize_receipts"]
        ),
        "side_state_receipts": list(
            capture["side_state_receipts"]
        ),
        "proposal_kv_receipts": list(
            capture["proposal_kv_receipts"]
        ),
        "lifecycle_events": list(
            capture["lifecycle_events"]
        ),
    }


def _model_identity(
    engine,
    *,
    target_digest: str,
    mtp_digest: str,
) -> dict:
    config = getattr(engine, "config", None)
    if config is None:
        config = getattr(
            engine.model_runner,
            "config",
            None,
        )
    hf_config = getattr(config, "hf_config", None)
    if hf_config is None:
        raise RuntimeError(
            "loaded model Hugging Face config is unavailable"
        )
    return {
        "model_type": str(
            getattr(hf_config, "model_type", "")
        ),
        "architectures": list(
            getattr(hf_config, "architectures", ()) or ()
        ),
        "target_model_manifest_sha256": target_digest,
        "mtp_checkpoint_manifest_sha256": mtp_digest,
    }


def _native_state_snapshot(
    executor,
    physical_store,
) -> dict:
    return {
        "pending_prefix_count": len(
            executor._pending_prefixes
        ),
        "bootstrapped_sequence_count": len(
            executor._bootstrapped
        ),
        "proposal_transaction_count": len(
            executor._proposal_transactions
        ),
        "batch_ticket_count": len(
            executor._batch_tickets
        ),
        "batch_ticket_transaction_count": len(
            executor._batch_ticket_transactions
        ),
        "allocated_physical_slot_count": len(
            physical_store._allocated_slot_ids
        ),
    }


def run_policy_cell(
    *,
    model_path: str,
    gpu_index: int,
    policy: str,
    batch_size: int,
    engine_factory,
    sampling_params_type,
    runtime_type,
    synchronize,
    target_manifest_resolver,
    mtp_manifest_resolver,
) -> dict:
    gate.cell_key(policy, batch_size)
    gpu_index = gate._integer(gpu_index, "GPU index")
    target_digest = target_manifest_resolver(model_path)
    mtp_digest = mtp_manifest_resolver(model_path)
    if target_digest != gate.TARGET_MODEL_MANIFEST_SHA256:
        raise RuntimeError(
            "target model manifest does not match authority"
        )
    if mtp_digest != gate.MTP_CHECKPOINT_MANIFEST_SHA256:
        raise RuntimeError(
            "MTP checkpoint manifest does not match authority"
        )
    engine = None
    exit_called = False
    exit_receipt = {}
    cell = None
    try:
        native = policy == "native_mtp"
        engine = engine_factory(
            model_path,
            tensor_parallel_size=1,
            enforce_eager=True,
            max_model_len=8192,
            max_num_batched_tokens=16384,
            max_num_prefill_tokens_per_step=1024,
            max_num_seqs=batch_size,
            kv_offload_mvp0=False,
            qwen35_mtp_enabled=native,
            qwen35_mtp_cuda_graphs=False,
            qwen35_mtp_max_proposal_tokens=(
                gate.MAX_PROPOSAL_TOKENS
            ),
        )
        descriptor = None
        executor = None
        physical_store = None
        native_binding = None
        if native:
            (
                descriptor,
                module,
                executor,
                physical_store,
            ) = validate_native_registration(
                engine.model_runner
            )
            native_binding = {
                "executor_id": descriptor.executor_id,
                "source_type": (
                    descriptor.capabilities.source_type
                ),
                "module_type": type(module).__name__,
                "physical_store_type": (
                    type(physical_store).__name__
                ),
                "checkpoint_tensor_count": 15,
            }
            engine.activate_speculative_runtime(
                runtime_type(
                    model_runner_executor=descriptor
                )
            )
        prompt_rows = build_prompt_rows(
            engine.tokenizer,
            batch_size,
        )
        sampling_params = sampling_params_type(
            temperature=0.0,
            max_tokens=gate.MAX_OUTPUT_TOKENS,
            ignore_eos=True,
        )
        before = _lease_snapshot(engine)
        empty_capture = {
            "method_names": [],
            "ordinary_decode_target_forward_calls": 0,
            "proposal_finalize_receipts": [],
            "side_state_receipts": [],
            "proposal_kv_receipts": [],
            "lifecycle_events": [],
        }
        if native:
            with capture_runtime_receipts(
                engine,
                executor,
            ) as capture:
                output_rows, observations = _run_generation(
                    engine=engine,
                    prompt_rows=prompt_rows,
                    sampling_params=sampling_params,
                    synchronize=synchronize,
                    target_forward_capture=capture,
                )
        else:
            capture = empty_capture
            output_rows, observations = _run_generation(
                engine=engine,
                prompt_rows=prompt_rows,
                sampling_params=sampling_params,
                synchronize=synchronize,
            )
        engine.flush_pending_hybrid_state_releases(
            timeout_s=60.0
        )
        after = _lease_snapshot(engine)
        native_snapshot = (
            _native_state_snapshot(executor, physical_store)
            if native
            else None
        )
        runtime = summarize_runtime(
            observations,
            capture=capture,
            native_binding=native_binding,
        )
        cell = {
            "schema_version": gate.SCHEMA_VERSION,
            "policy": policy,
            "batch_size": batch_size,
            "world_size": gate.WORLD_SIZE,
            "gpu_index": gpu_index,
            "prompt_token_count": gate.PROMPT_TOKENS,
            "max_output_tokens": gate.MAX_OUTPUT_TOKENS,
            "max_proposal_tokens": (
                gate.MAX_PROPOSAL_TOKENS
            ),
            "model_identity": _model_identity(
                engine,
                target_digest=target_digest,
                mtp_digest=mtp_digest,
            ),
            "prompt_rows": prompt_rows,
            "output_rows": output_rows,
            "runtime": runtime,
            "cleanup": {
                "proposal_transactions_open": (
                    []
                    if native_snapshot is None
                    or native_snapshot[
                        "proposal_transaction_count"
                    ]
                    == 0
                    else ["open"]
                ),
                "proposal_finalize_tickets_open": (
                    []
                    if native_snapshot is None
                    or (
                        native_snapshot["batch_ticket_count"]
                        == 0
                        and native_snapshot[
                            "batch_ticket_transaction_count"
                        ]
                        == 0
                    )
                    else ["open"]
                ),
                "proposal_sequence_ids": (
                    []
                    if native_snapshot is None
                    or (
                        native_snapshot[
                            "pending_prefix_count"
                        ]
                        == 0
                        and native_snapshot[
                            "bootstrapped_sequence_count"
                        ]
                        == 0
                    )
                    else ["open"]
                ),
                "proposal_kv_slots_in_use": (
                    0
                    if native_snapshot is None
                    else native_snapshot[
                        "allocated_physical_slot_count"
                    ]
                ),
                "native_state_snapshot": native_snapshot,
                "hybrid_state_leases_before": int(
                    before["used_slots"]
                ),
                "hybrid_state_leases_after": int(
                    after["used_slots"]
                ),
                "owned_children_remaining": [],
                "engine_exit_called": True,
                "worker_exit_code": 0,
            },
            "runtime_poisoned": bool(
                engine.speculative_runtime_poisoned
            ),
        }
    finally:
        if engine is not None:
            exit_called = True
            exit_receipt = engine.exit() or {}
    if cell is None:
        raise RuntimeError(
            "worker did not produce a cell result"
        )
    cell["cleanup"]["engine_exit_called"] = exit_called
    cell["cleanup"]["owned_children_remaining"] = list(
        exit_receipt.get("owned_children_remaining", [])
    )
    return gate.validate_cell_result(cell)


def _default_dependencies():
    import torch

    from tinyvllm import LLM
    from tinyvllm.engine.speculative_runtime import (
        EngineSpeculativeRuntime,
    )
    from tinyvllm.sampling_params import SamplingParams

    return {
        "engine_factory": LLM,
        "sampling_params_type": SamplingParams,
        "runtime_type": EngineSpeculativeRuntime,
        "synchronize": torch.cuda.synchronize,
        "target_manifest_resolver": (
            target_model_manifest_sha256
        ),
        "mtp_manifest_resolver": (
            mtp_checkpoint_manifest_sha256
        ),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--gpu-index", required=True, type=int)
    parser.add_argument(
        "--policy",
        required=True,
        choices=gate.POLICIES,
    )
    parser.add_argument(
        "--batch-size",
        required=True,
        type=int,
        choices=gate.BATCH_SIZES,
    )
    parser.add_argument("--out", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = run_policy_cell(
        model_path=args.model,
        gpu_index=args.gpu_index,
        policy=args.policy,
        batch_size=args.batch_size,
        **_default_dependencies(),
    )
    gate.atomic_write_json(Path(args.out), result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
