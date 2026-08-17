from __future__ import annotations

import importlib
import functools
import time


HYBRID_PREFIX_TIMEOUT_S = 120.0


def _default_engine_factory(configuration, authorized):
    module = importlib.import_module("tinyvllm.engine.llm_engine")
    engine = configuration["engine"]
    return module.LLMEngine(
        authorized["model_dir"],
        **engine,
    )


def _default_sampling_params_factory(**kwargs):
    module = importlib.import_module("tinyvllm.sampling_params")
    return module.SamplingParams(**kwargs)


def _request_prompt(workload_spec, request_index):
    continuation = workload_spec["continuations"][request_index]
    prefix = list(workload_spec["shared_prefix_token_ids"])
    for index, token_id in continuation["prefix_overrides"]:
        prefix[index] = token_id
    return prefix + list(continuation["suffix_token_ids"])


def _source_prompt(workload_spec):
    return (
        list(workload_spec["shared_prefix_token_ids"])
        + list(workload_spec["source_suffix_token_ids"])
    )


def _tensor_rows(value, count):
    if callable(getattr(value, "detach", None)):
        value = value.detach().float().cpu().tolist()
    if not isinstance(value, list):
        raise ValueError("benchmark step logits are invalid")
    if count == 1 and value and not isinstance(value[0], list):
        value = [value]
    if (
        len(value) != count
        or any(not isinstance(row, list) for row in value)
    ):
        raise ValueError("benchmark step logits shape mismatch")
    return value


class BenchmarkEngineAdapter:

    def __init__(
        self,
        configuration,
        authorized,
        *,
        engine_factory=_default_engine_factory,
        sampling_params_factory=_default_sampling_params_factory,
        clock_ns=time.monotonic_ns,
    ):
        if not isinstance(configuration, dict):
            raise ValueError("benchmark Engine configuration is invalid")
        if not isinstance(authorized, dict):
            raise ValueError("benchmark runtime authorization is invalid")
        for value, label in (
            (engine_factory, "engine_factory"),
            (sampling_params_factory, "sampling_params_factory"),
            (clock_ns, "clock_ns"),
        ):
            if not callable(value):
                raise TypeError(f"{label} must be callable")
        hybrid_prefix = configuration.get("hybrid_prefix")
        if not isinstance(hybrid_prefix, dict):
            raise ValueError(
                "benchmark hybrid prefix configuration is invalid"
            )
        representation = hybrid_prefix.get("representation")
        expected_representation = (
            "exact_full_fidelity"
            if configuration.get("policy") == "exact_restore"
            else "none"
        )
        if representation != expected_representation:
            raise ValueError(
                "benchmark schema-v1 representation is inconsistent "
                "with policy"
            )
        self.configuration = configuration
        self.authorized = authorized
        self.engine = engine_factory(configuration, authorized)
        self.sampling_params_factory = sampling_params_factory
        self.clock_ns = clock_ns
        profiling = configuration.get("profiling", {})
        if not isinstance(profiling, dict):
            raise ValueError(
                "benchmark profiling configuration is invalid"
            )
        self._profile_enabled = profiling.get("enabled", False)
        if not isinstance(self._profile_enabled, bool):
            raise ValueError(
                "benchmark profiling enabled flag is invalid"
            )
        self._profile_events = []
        self._profile_requests = []
        self._decode_internal_enabled = profiling.get(
            "decode_internal",
            False,
        )
        if not isinstance(self._decode_internal_enabled, bool):
            raise ValueError(
                "benchmark decode internal profiling flag is invalid"
            )
        self._decode_internal_snapshot = None
        if self._profile_enabled:
            self._install_profile_wrappers()
        self.timeout_s = HYBRID_PREFIX_TIMEOUT_S
        self.recurrent_calibration_capture = configuration.get(
            "recurrent_calibration_capture"
        )
        if self.recurrent_calibration_capture is not None:
            capture = self.recurrent_calibration_capture
            self.engine.configure_qwen35_recurrent_capture(
                capture_root=capture["capture_root"],
                model_manifest_sha256=(
                    capture["model_manifest_sha256"]
                ),
                source_tree_sha256=capture["source_tree_sha256"],
                workload_manifest_sha256=(
                    capture["workload_manifest_sha256"]
                ),
                world_size=capture["world_size"],
                workload_ids=capture["workload_ids"],
                timeout_s=self.timeout_s,
            )
        self.closed = False
        self.cleanup_receipt = None

    def _install_profile_wrappers(self):
        operations = {
            "flush_pending_hybrid_state_releases": (
                "release_flush",
                lambda args, kwargs: None,
            ),
            "acquire_qwen35_hybrid_prefix": (
                "restore_total",
                lambda args, kwargs: (
                    getattr(args[0], "seq_id", None)
                    if args else None
                ),
            ),
            "prepare_model_runner_hybrid_prefix_restore": (
                "restore_prepare",
                lambda args, kwargs: (
                    getattr(args[0], "request_id", None)
                    if args else None
                ),
            ),
            "validate_model_runner_hybrid_prefix_restore": (
                "restore_validate",
                lambda args, kwargs: (
                    getattr(args[0], "request_id", None)
                    if args else None
                ),
            ),
            "commit_model_runner_hybrid_prefix_restore": (
                "restore_commit",
                lambda args, kwargs: (
                    getattr(args[0], "request_id", None)
                    if args else None
                ),
            ),
            "rollback_model_runner_hybrid_prefix_restore": (
                "restore_rollback",
                lambda args, kwargs: (
                    getattr(args[0], "request_id", None)
                    if args else None
                ),
            ),
        }
        for method_name, (event_name, request_id_from_call) in (
            operations.items()
        ):
            method = getattr(self.engine, method_name, None)
            if not callable(method):
                continue

            @functools.wraps(method)
            def timed(*args, __method=method,
                      __event_name=event_name,
                      __request_id_from_call=request_id_from_call,
                      **kwargs):
                started_ns = self.clock_ns()
                status = "ok"
                try:
                    return __method(*args, **kwargs)
                except BaseException:
                    status = "error"
                    raise
                finally:
                    ended_ns = self.clock_ns()
                    self._profile_events.append({
                        "name": __event_name,
                        "request_id": __request_id_from_call(
                            args,
                            kwargs,
                        ),
                        "duration_ns": max(
                            0,
                            ended_ns - started_ns,
                        ),
                        "status": status,
                    })

            setattr(self.engine, method_name, timed)

    def profile_snapshot(self):
        return {
            "enabled": self._profile_enabled,
            "events": [
                dict(event) for event in self._profile_events
            ],
            "requests": [
                dict(request) for request in self._profile_requests
            ],
            "decode_internal": self._decode_internal_snapshot,
        }

    def configure_qwen35_hybrid_prefix_publication_runtime(
        self,
        *,
        model_fingerprint,
        max_entries,
        max_bytes,
        timeout_s,
        representation="exact_restore",
    ):
        self.timeout_s = timeout_s
        return (
            self.engine
            .configure_qwen35_hybrid_prefix_publication_runtime(
                model_fingerprint=model_fingerprint,
                max_entries=max_entries,
                max_bytes=max_bytes,
                timeout_s=timeout_s,
                representation=representation,
            )
        )

    def _sampling_params(self, generated_tokens):
        sampling = self.configuration["sampling"]
        return self.sampling_params_factory(
            temperature=sampling["temperature"],
            max_tokens=generated_tokens,
            ignore_eos=sampling["ignore_eos"],
        )

    def _admit(self, prompt, generated_tokens, request_id):
        before = {
            sequence.seq_id
            for sequence in self.engine.scheduler.waiting
        }
        admitted_ns = self.clock_ns()
        self.engine.add_request(
            list(prompt),
            self._sampling_params(generated_tokens),
        )
        appended = [
            sequence
            for sequence in self.engine.scheduler.waiting
            if sequence.seq_id not in before
        ]
        if len(appended) != 1:
            raise ValueError(
                "benchmark request admission is ambiguous"
            )
        if self.configuration["policy"] == "recompute":
            appended[0].hybrid_prefix_restore_attempted = True
            appended[0].hybrid_prefix_restore_hit = False
        return appended[0].seq_id, {
            "request_id": request_id,
            "admitted_ns": admitted_ns,
            "first_token_ns": None,
            "token_timestamps_ns": [],
            "output_token_ids": [],
            "executed_prefill_tokens": 0,
            "final_logits": None,
            "complete": False,
        }

    def _run_requests(
        self,
        requests,
        *,
        record_logits,
    ):
        lifecycle = {}
        for request in requests:
            seq_id, row = self._admit(
                request["prompt"],
                request["generated_tokens"],
                request["request_id"],
            )
            if seq_id in lifecycle:
                raise ValueError("benchmark sequence identity is duplicate")
            lifecycle[seq_id] = row
        if record_logits:
            enabled = (
                self.engine.enable_step_logits_authority_recording(
                    True,
                    timeout_s=self.timeout_s,
                )
            )
            if enabled != {
                "enabled": True,
                "rank_inventory": [0, 1, 2, 3],
            }:
                raise ValueError(
                    "benchmark logits recording enable mismatch"
                )
        previous_step_end_ns = max(
            row["admitted_ns"] for row in lifecycle.values()
        )
        try:
            while not self.engine.is_finished():
                outputs, _ = self.engine.step()
                observation = self.engine.last_step_observation
                if not isinstance(observation, dict):
                    raise ValueError(
                        "benchmark step observation is missing"
                    )
                step_end_ns = observation.get("step_end_ns")
                scheduled = observation.get("scheduled")
                token_deltas = observation.get(
                    "new_completion_tokens_by_seq"
                )
                if (
                    isinstance(step_end_ns, bool)
                    or not isinstance(step_end_ns, int)
                    or not isinstance(scheduled, list)
                    or not isinstance(token_deltas, dict)
                ):
                    raise ValueError(
                        "benchmark step observation is invalid"
                    )
                if step_end_ns < previous_step_end_ns:
                    raise ValueError(
                        "benchmark step timestamp is non-monotonic"
                    )
                previous_step_end_ns = step_end_ns
                sampled_ids = []
                for scheduled_row in scheduled:
                    if not isinstance(scheduled_row, dict):
                        raise ValueError(
                            "benchmark scheduled row is invalid"
                        )
                    seq_id = scheduled_row.get("seq_id")
                    row = lifecycle.get(seq_id)
                    if row is None:
                        raise ValueError(
                            "benchmark scheduled sequence is unknown"
                        )
                    if scheduled_row.get("is_decode") is False:
                        start = scheduled_row.get(
                            "prefill_chunk_start"
                        )
                        end = scheduled_row.get("prefill_chunk_end")
                        if (
                            isinstance(start, bool)
                            or not isinstance(start, int)
                            or isinstance(end, bool)
                            or not isinstance(end, int)
                            or end < start
                        ):
                            raise ValueError(
                                "benchmark prefill observation is invalid"
                            )
                        row["executed_prefill_tokens"] += end - start
                    if scheduled_row.get("do_sample") is True:
                        sampled_ids.append(seq_id)
                normalized_deltas = {}
                for raw_seq_id, delta in token_deltas.items():
                    try:
                        seq_id = int(raw_seq_id)
                    except (TypeError, ValueError) as error:
                        raise ValueError(
                            "benchmark token sequence is invalid"
                        ) from error
                    if (
                        seq_id not in lifecycle
                        or not isinstance(delta, list)
                        or any(
                            isinstance(token_id, bool)
                            or not isinstance(token_id, int)
                            or token_id < 0
                            for token_id in delta
                        )
                    ):
                        raise ValueError(
                            "benchmark token delta is invalid"
                        )
                    normalized_deltas[seq_id] = delta
                    row = lifecycle[seq_id]
                    if delta:
                        if row["first_token_ns"] is None:
                            row["first_token_ns"] = step_end_ns
                        row["token_timestamps_ns"].extend(
                            [step_end_ns] * len(delta)
                        )
                        row["output_token_ids"].extend(delta)
                if record_logits and sampled_ids:
                    logits = _tensor_rows(
                        self.engine.read_step_logits_authority(),
                        len(sampled_ids),
                    )
                    for seq_id, values in zip(sampled_ids, logits):
                        if normalized_deltas.get(seq_id):
                            lifecycle[seq_id]["final_logits"] = values
                for seq_id, output_token_ids in outputs:
                    row = lifecycle.get(seq_id)
                    if row is None:
                        raise ValueError(
                            "benchmark output sequence is unknown"
                        )
                    if list(output_token_ids) != row["output_token_ids"]:
                        raise ValueError(
                            "benchmark output and token delta mismatch"
                        )
                    row["complete"] = True
        finally:
            if record_logits:
                disabled = (
                    self.engine.enable_step_logits_authority_recording(
                        False,
                        timeout_s=self.timeout_s,
                    )
                )
                if disabled != {
                    "enabled": False,
                    "rank_inventory": [0, 1, 2, 3],
                }:
                    raise ValueError(
                        "benchmark logits recording disable mismatch"
                    )
        result = []
        for seq_id in lifecycle:
            row = lifecycle[seq_id]
            timestamps = row["token_timestamps_ns"]
            if (
                row["complete"] is not True
                or row["first_token_ns"] is None
                or not timestamps
            ):
                raise ValueError("benchmark request is incomplete")
            request_result = {
                "request_id": row["request_id"],
                "prompt_tokens": requests[len(result)][
                    "prompt_tokens"
                ],
                "reused_kv_tokens": (
                    requests[len(result)]["prompt_tokens"]
                    - row["executed_prefill_tokens"]
                ),
                "restored_hybrid_state": (
                    row["executed_prefill_tokens"]
                    < requests[len(result)]["prompt_tokens"]
                ),
                "executed_prefill_tokens": (
                    row["executed_prefill_tokens"]
                ),
                "generated_tokens": len(row["output_token_ids"]),
                "ttft_ns": (
                    row["first_token_ns"] - row["admitted_ns"]
                ),
                "e2e_ns": timestamps[-1] - row["admitted_ns"],
                "decode_step_ns": [
                    current - previous
                    for previous, current in zip(
                        timestamps,
                        timestamps[1:],
                    )
                ],
                "output_token_ids": row["output_token_ids"],
                "final_logits": (
                    row["final_logits"] if record_logits else None
                ),
            }
            result.append(request_result)
            if self._profile_enabled:
                self._profile_requests.append({
                    "request_id": request_result["request_id"],
                    "ttft_ns": request_result["ttft_ns"],
                    "decode_ns": max(
                        0,
                        request_result["e2e_ns"]
                        - request_result["ttft_ns"],
                    ),
                    "e2e_ns": request_result["e2e_ns"],
                    "executed_prefill_tokens": request_result[
                        "executed_prefill_tokens"
                    ],
                    "reused_kv_tokens": request_result[
                        "reused_kv_tokens"
                    ],
                })
        return result

    def _run_source(
        self,
        workload_spec,
        *,
        capture_workload_id=None,
    ):
        if capture_workload_id is not None:
            self.engine.arm_qwen35_recurrent_capture(
                capture_workload_id,
                timeout_s=self.timeout_s,
            )
        source = _source_prompt(workload_spec)
        self._run_requests(
            [{
                "request_id": "source",
                "prompt": source,
                "prompt_tokens": len(source),
                "generated_tokens": 1,
            }],
            record_logits=False,
        )
        if capture_workload_id is not None:
            self.engine.finish_qwen35_recurrent_capture_workload(
                capture_workload_id,
                timeout_s=self.timeout_s,
            )

    def _apply_w4_invalidation(self, request_index):
        if request_index == 0:
            return
        snapshots = (
            self.engine.qwen35_hybrid_prefix_authority_snapshots(
                timeout_s=self.timeout_s,
            )
        )
        if not snapshots:
            raise ValueError(
                "benchmark authority snapshot is missing"
            )
        if request_index == 1:
            block_identities = snapshots[0].get(
                "last_publication_block_identities"
            )
            if not block_identities:
                raise ValueError(
                    "benchmark stale block identities are missing"
                )
            self.engine.invalidate_qwen35_hybrid_prefix_blocks(
                block_identities,
                timeout_s=self.timeout_s,
            )
        elif request_index == 2:
            self.engine.clear_qwen35_hybrid_prefix_caches(
                timeout_s=self.timeout_s,
            )

    def run_benchmark_workload(
        self,
        *,
        workload,
        workload_spec,
        phase,
        repetition,
        policy,
    ):
        if self.closed:
            raise RuntimeError("benchmark Engine adapter is closed")
        if policy != self.configuration["policy"]:
            raise ValueError("benchmark policy configuration mismatch")
        spec = workload_spec["spec"]
        generated_tokens = spec["generated_tokens"]
        requests = []
        if workload == "w4_miss_invalidation":
            for request_index in range(spec["continuations"]):
                self._run_source(
                    workload_spec,
                    capture_workload_id=(
                        workload
                        if (
                            self.recurrent_calibration_capture
                            is not None
                            and request_index == 0
                        )
                        else None
                    ),
                )
                if policy == "exact_restore":
                    self._apply_w4_invalidation(request_index)
                prompt = _request_prompt(
                    workload_spec,
                    request_index,
                )
                requests.extend(self._run_requests(
                    [{
                        "request_id": f"request-{request_index}",
                        "prompt": prompt,
                        "prompt_tokens": len(prompt),
                        "generated_tokens": generated_tokens,
                    }],
                    record_logits=phase == "correctness",
                ))
        else:
            self._run_source(
                workload_spec,
                capture_workload_id=(
                    workload
                    if self.recurrent_calibration_capture is not None
                    else None
                ),
            )
            if workload == "w0_short_control" and policy == "exact_restore":
                self.engine.clear_qwen35_hybrid_prefix_caches(
                    timeout_s=self.timeout_s,
                )
            request_specs = []
            for request_index in range(spec["continuations"]):
                prompt = _request_prompt(
                    workload_spec,
                    request_index,
                )
                request_specs.append({
                    "request_id": f"request-{request_index}",
                    "prompt": prompt,
                    "prompt_tokens": len(prompt),
                    "generated_tokens": generated_tokens,
                })
            if self._decode_internal_enabled:
                case_id = (
                    f"{workload}__{phase}__r{repetition}__{policy}"
                )
                configured = (
                    self.engine.configure_decode_internal_profile(
                        True,
                        (
                            f"policy={policy}/"
                            f"case={case_id}"
                        ),
                        timeout_s=self.timeout_s,
                    )
                )
                if configured != {
                    "enabled": True,
                    "rank_inventory": [0, 1, 2, 3],
                }:
                    raise ValueError(
                        "decode internal profile configuration mismatch"
                    )
            try:
                if workload == "w3_batched_fanout":
                    if policy == "recompute":
                        self.engine.clear_reusable_prefix_cache()
                    requests = self._run_requests(
                        request_specs,
                        record_logits=phase == "correctness",
                    )
                else:
                    for request in request_specs:
                        if policy == "recompute":
                            self.engine.clear_reusable_prefix_cache()
                        requests.extend(self._run_requests(
                            [request],
                            record_logits=phase == "correctness",
                        ))
            finally:
                if self._decode_internal_enabled:
                    self._decode_internal_snapshot = (
                        self.engine.finalize_decode_internal_profile(
                            timeout_s=self.timeout_s,
                        )
                    )
        return {"requests": requests}

    def memory_snapshots(self, *, timeout_s):
        return self.engine.memory_snapshots(timeout_s=timeout_s)

    def capacity_snapshot(self):
        return self.engine.capacity_snapshot()

    def qwen35_hybrid_prefix_cache_snapshots(self, *, timeout_s):
        return self.engine.qwen35_hybrid_prefix_cache_snapshots(
            timeout_s=timeout_s
        )

    def close(self):
        if self.closed:
            return
        self.closed = True
        receipt = self.engine.exit()
        if (
            not isinstance(receipt, dict)
            or receipt.get("process_group_destroyed") is not True
            or receipt.get("rank_exit_codes") != [0, 0, 0, 0]
            or receipt.get("owned_children_remaining") != []
            or [
                row.get("rank")
                for row in receipt.get("rank_cleanup_receipts", [])
            ] != [0, 1, 2, 3]
            or any(
                row.get("process_group_destroyed") is not True
                for row in receipt["rank_cleanup_receipts"]
            )
        ):
            raise ValueError(
                "benchmark Engine cleanup receipt mismatch"
            )
        self.cleanup_receipt = receipt
