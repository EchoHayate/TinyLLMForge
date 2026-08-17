from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
import sys


def _default_engine_factory(configuration):
    backend = importlib.import_module(
        "qwen35_tp4_engine_backend_session"
    )
    return backend._default_engine_factory(
        configuration,
        max_num_batched_tokens=4096,
        max_num_seqs=2,
    )


def _default_reference_provider(**kwargs):
    raise RuntimeError(
        "cached-continuation independent reference provider "
        "is not configured"
    )


def _default_logits_comparator(
    engine_logits,
    reference_logits,
    *,
    atol,
):
    if len(engine_logits) != len(reference_logits):
        raise ValueError("cached-continuation logits count mismatch")
    if not engine_logits:
        raise ValueError("cached-continuation logits are missing")
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "cached-continuation logits comparator requires Torch"
        ) from error
    maximum = 0.0
    allclose = True
    per_step_max_abs_diff = []
    first_mismatch_step = None
    first_mismatch_engine_argmax = None
    first_mismatch_reference_argmax = None
    for step, (engine_row, reference_row) in enumerate(
        zip(engine_logits, reference_logits)
    ):
        if (
            not isinstance(engine_row, torch.Tensor)
            or not isinstance(reference_row, torch.Tensor)
            or engine_row.shape != reference_row.shape
        ):
            raise ValueError(
                "cached-continuation logits shape mismatch"
            )
        engine_float = engine_row.float()
        reference_float = reference_row.float()
        difference = float(
            (engine_float - reference_float).abs().max().item()
        )
        per_step_max_abs_diff.append(difference)
        maximum = max(maximum, difference)
        step_allclose = bool(
            torch.allclose(
                engine_float,
                reference_float,
                atol=atol,
                rtol=0.0,
            )
        )
        allclose = allclose and step_allclose
        if not step_allclose and first_mismatch_step is None:
            first_mismatch_step = step
            first_mismatch_engine_argmax = int(
                engine_float.argmax().item()
            )
            first_mismatch_reference_argmax = int(
                reference_float.argmax().item()
            )
    return {
        "max_abs_diff": maximum,
        "allclose": allclose,
        "first_mismatch_step": first_mismatch_step,
        "per_step_max_abs_diff": per_step_max_abs_diff,
        "first_mismatch_engine_argmax": (
            first_mismatch_engine_argmax
        ),
        "first_mismatch_reference_argmax": (
            first_mismatch_reference_argmax
        ),
    }


def _load_contract():
    name = "qwen35_tp4_cached_continuation_correctness_contract"
    module = sys.modules.get(name)
    if module is not None:
        return module
    path = Path(__file__).resolve().parent / (
        "qwen35_tp4_cached_continuation_correctness_contract.py"
    )
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


contract = _load_contract()


class CachedContinuationSessionFactory:

    def __init__(
        self,
        configuration,
        *,
        engine_factory=_default_engine_factory,
        reference_executor_factory,
        logits_comparator=_default_logits_comparator,
    ):
        for value, label in (
            (engine_factory, "engine_factory"),
            (reference_executor_factory, "reference_executor_factory"),
            (logits_comparator, "logits_comparator"),
        ):
            if not callable(value):
                raise TypeError(f"{label} must be callable")
        self.configuration = configuration
        self.engine_factory = engine_factory
        self.reference_executor_factory = reference_executor_factory
        self.logits_comparator = logits_comparator
        self.reference_corpus = None
        self.engine = None
        self.rows = []
        self.cleanup_receipt = None
        self.closed = False

    def _ensure_engine(self):
        if self.engine is not None:
            return self.engine
        self.engine = self.engine_factory(self.configuration)
        self.engine.configure_qwen35_hybrid_prefix_publication_runtime(
            model_fingerprint=self.configuration.model_fingerprint,
            max_entries=self.configuration.max_cache_entries,
            max_bytes=self.configuration.max_cache_bytes,
            timeout_s=self.configuration.timeout_s,
        )
        return self.engine

    def _record_row(self, row):
        self.rows.append(row)

    def _ensure_reference_corpus(self):
        if self.reference_corpus is not None:
            return
        executor = self.reference_executor_factory()
        if (
            not callable(
                getattr(
                    executor,
                    "generate_reference_with_step_logits",
                    None,
                )
            )
            or not callable(getattr(executor, "close", None))
        ):
            raise TypeError(
                "cached continuation reference executor is invalid"
            )
        corpus = {}
        try:
            for workload in contract.WORKLOADS:
                payload = contract.workload_payload(workload)
                for request_index in range(
                    payload["spec"]["continuations"]
                ):
                    prompt = _request_prompt(
                        payload,
                        request_index,
                    )
                    generated_tokens = payload["spec"][
                        "generated_tokens"
                    ]
                    corpus[(workload, request_index)] = {
                        "prompt_token_ids": prompt,
                        "generated_tokens": generated_tokens,
                        "reference": (
                            executor
                            .generate_reference_with_step_logits(
                                scenario="publish_source",
                                prompt_token_ids=prompt,
                                generated_tokens=generated_tokens,
                                generation_policy={
                                    "temperature": 0.0,
                                    "ignore_eos": True,
                                },
                            )
                        ),
                    }
        finally:
            executor.close()
        self.reference_corpus = corpus

    def _reference_provider(
        self,
        *,
        workload,
        request_index,
        prompt_token_ids,
        generated_tokens,
    ):
        self._ensure_reference_corpus()
        row = self.reference_corpus.get((workload, request_index))
        if (
            row is None
            or row["prompt_token_ids"] != list(prompt_token_ids)
            or row["generated_tokens"] != generated_tokens
        ):
            raise ValueError(
                "cached continuation reference corpus lookup mismatch"
            )
        reference = row["reference"]
        return {
            "output_token_ids": list(
                reference["output_token_ids"]
            ),
            "step_logits": [
                value.clone()
                if callable(getattr(value, "clone", None))
                else value
                for value in reference["step_logits"]
            ],
        }

    def __call__(
        self,
        configuration,
        *,
        workload,
        request_index,
        payload,
    ):
        if self.closed:
            raise RuntimeError(
                "cached continuation session factory is closed"
            )
        expected_payload = self.configuration.to_payload()
        actual_payload = configuration.to_payload()
        if actual_payload != expected_payload:
            raise ValueError(
                "cached continuation session configuration mismatch"
            )
        self._ensure_reference_corpus()
        return CachedContinuationBackendSession(
            configuration,
            workload=workload,
            request_index=request_index,
            payload=payload,
            engine_provider=self._ensure_engine,
            owns_engine=False,
            row_recorder=self._record_row,
            reference_provider=self._reference_provider,
            logits_comparator=self.logits_comparator,
        )

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.engine is None:
            return
        receipt = _cleanup_engine(
            self.engine,
            len(self.configuration.gpu_indices),
        )
        self.cleanup_receipt = receipt
        for row in self.rows:
            row["process_group_destroyed"] = True
            row["owned_children_remaining"] = []


def _sampling_params(max_tokens):
    module_name = "tinyvllm.sampling_params"
    module = sys.modules.get(module_name)
    if module is None:
        path = (
            Path(__file__).resolve().parents[1]
            / "tinyvllm"
            / "sampling_params.py"
        )
        spec = importlib.util.spec_from_file_location(
            module_name,
            path,
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return module.SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        ignore_eos=True,
    )


def _request_prompt(payload, request_index):
    continuation = payload["continuations"][request_index]
    prefix = list(payload["shared_prefix_token_ids"])
    for index, token_id in continuation["prefix_overrides"]:
        prefix[index] = token_id
    return prefix + list(continuation["suffix_token_ids"])


class CachedContinuationBackendSession:

    def __init__(
        self,
        configuration,
        *,
        workload,
        request_index,
        payload,
        engine_factory=_default_engine_factory,
        engine=None,
        engine_provider=None,
        owns_engine=True,
        row_recorder=None,
        reference_provider=_default_reference_provider,
        logits_comparator=_default_logits_comparator,
    ):
        if payload != contract.workload_payload(workload):
            raise ValueError("cached backend workload payload mismatch")
        continuation_count = payload["spec"]["continuations"]
        if (
            isinstance(request_index, bool)
            or not isinstance(request_index, int)
            or request_index < 0
            or request_index >= continuation_count
        ):
            raise ValueError("cached backend request index mismatch")
        for value, label in (
            (engine_factory, "engine_factory"),
            (reference_provider, "reference_provider"),
            (logits_comparator, "logits_comparator"),
        ):
            if not callable(value):
                raise TypeError(f"{label} must be callable")
        self.configuration = configuration
        self.workload = workload
        self.request_index = request_index
        self.payload = payload
        self.engine_factory = engine_factory
        if engine_provider is not None and not callable(engine_provider):
            raise TypeError("engine_provider must be callable")
        self.engine_provider = engine_provider
        self.owns_engine = bool(owns_engine)
        self.defers_cleanup = not self.owns_engine
        self.row_recorder = row_recorder
        self.reference_provider = reference_provider
        self.logits_comparator = logits_comparator
        self.engine = engine
        self.cleanup_receipt = None
        self.closed = False

    def _snapshot(self):
        rows = self.engine.qwen35_hybrid_prefix_authority_snapshots(
            timeout_s=self.configuration.timeout_s,
        )
        world_size = len(self.configuration.gpu_indices)
        if (
            not isinstance(rows, tuple)
            or len(rows) != world_size
            or [row.get("rank") for row in rows]
            != list(range(world_size))
        ):
            raise ValueError(
                "cached backend rank snapshot mismatch"
            )
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
            raise ValueError(
                "cached backend rank snapshot parity mismatch"
            )
        return dict(rows[0])

    def _run_request(self, prompt, generated_tokens, *, record_logits):
        if record_logits:
            enabled = (
                self.engine.enable_step_logits_authority_recording(
                    True,
                    timeout_s=self.configuration.timeout_s,
                )
            )
            if enabled != {
                "enabled": True,
                "rank_inventory": [0, 1, 2, 3],
            }:
                raise ValueError(
                    "cached backend logits recording enable mismatch"
                )
        self.engine.add_request(
            list(prompt),
            _sampling_params(generated_tokens),
        )
        output_token_ids = []
        executed_prefill_tokens = 0
        step_logits = []
        try:
            while not self.engine.is_finished():
                outputs, num_tokens = self.engine.step()
                if num_tokens > 0:
                    executed_prefill_tokens += num_tokens
                observation = getattr(
                    self.engine,
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
                        self.engine.read_step_logits_authority()
                    )
                for _, token_ids in outputs:
                    if len(token_ids) >= generated_tokens:
                        output_token_ids = list(token_ids)
            if len(output_token_ids) != generated_tokens:
                raise ValueError(
                    "cached backend output token count mismatch"
                )
            return (
                output_token_ids,
                executed_prefill_tokens,
                step_logits,
            )
        finally:
            if record_logits:
                disabled = (
                    self.engine
                    .enable_step_logits_authority_recording(
                        False,
                        timeout_s=self.configuration.timeout_s,
                    )
                )
                if disabled != {
                    "enabled": False,
                    "rank_inventory": [0, 1, 2, 3],
                }:
                    raise ValueError(
                        "cached backend logits recording disable mismatch"
                    )

    def _seed_source(self):
        prompt = (
            list(self.payload["shared_prefix_token_ids"])
            + list(self.payload["source_suffix_token_ids"])
        )
        self._run_request(prompt, 1, record_logits=False)
        snapshot = self._snapshot()
        if (
            snapshot["current_entries"] != 1
            or snapshot["publication_commits"] < 1
            or not snapshot["last_publication_block_identities"]
        ):
            raise ValueError(
                "cached backend source publication is incomplete"
            )

    def _apply_invalidation(self):
        invalidation = self.payload["continuations"][
            self.request_index
        ]["invalidation"]
        kind = invalidation["kind"]
        if kind in {"none", "token_mismatch"}:
            return
        if kind == "stale_block_generation":
            blocks = self._snapshot()[
                "last_publication_block_identities"
            ]
            self.engine.invalidate_qwen35_hybrid_prefix_blocks(
                blocks,
                timeout_s=self.configuration.timeout_s,
            )
            return
        if kind == "cache_clear":
            self.engine.clear_qwen35_hybrid_prefix_caches(
                timeout_s=self.configuration.timeout_s,
            )
            return
        raise ValueError("cached backend invalidation is unsupported")

    def _cleanup(self):
        self.cleanup_receipt = _cleanup_engine(
            self.engine,
            len(self.configuration.gpu_indices),
        )

    def run(self):
        if self.closed:
            raise RuntimeError("cached backend session is closed")
        reference = self.reference_provider(
            workload=self.workload,
            request_index=self.request_index,
            prompt_token_ids=_request_prompt(
                self.payload,
                self.request_index,
            ),
            generated_tokens=self.payload["spec"]["generated_tokens"],
        )
        if (
            not isinstance(reference, dict)
            or set(reference)
            != {"output_token_ids", "step_logits"}
        ):
            raise ValueError(
                "cached backend reference evidence is invalid"
            )
        if self.engine is None:
            if self.engine_provider is not None:
                self.engine = self.engine_provider()
            else:
                self.engine = self.engine_factory(self.configuration)
                self.engine.configure_qwen35_hybrid_prefix_publication_runtime(
                    model_fingerprint=self.configuration.model_fingerprint,
                    max_entries=self.configuration.max_cache_entries,
                    max_bytes=self.configuration.max_cache_bytes,
                    timeout_s=self.configuration.timeout_s,
                )
        try:
            if not self.owns_engine:
                self.engine.clear_qwen35_hybrid_prefix_caches(
                    timeout_s=self.configuration.timeout_s,
                )
            self._seed_source()
            self._apply_invalidation()
            baseline = self._snapshot()
            prompt = _request_prompt(
                self.payload,
                self.request_index,
            )
            generated_tokens = self.payload["spec"][
                "generated_tokens"
            ]
            output, executed_prefill, engine_logits = (
                self._run_request(
                    prompt,
                    generated_tokens,
                    record_logits=True,
                )
            )
            final = self._snapshot()
            hit_delta = final["hits"] - baseline["hits"]
            miss_delta = final["misses"] - baseline["misses"]
            expected_hit = self.workload in contract.HIT_WORKLOADS
            invalidation_kind = self.payload["continuations"][
                self.request_index
            ]["invalidation"]["kind"]
            expected_delta = (
                (1, 0)
                if expected_hit
                else (
                    (0, 0)
                    if invalidation_kind == "token_mismatch"
                    else (0, 1)
                )
            )
            if (hit_delta, miss_delta) != expected_delta:
                raise ValueError(
                    "cached backend restore counters mismatch: "
                    f"workload={self.workload}, "
                    f"request_index={self.request_index}, "
                    f"expected_delta={expected_delta}, "
                    "observed_delta="
                    f"{(hit_delta, miss_delta)}"
                )
            comparison = self.logits_comparator(
                engine_logits,
                reference["step_logits"],
                atol=contract.REGISTERED_LOGITS_ATOL,
            )
            required_comparison_fields = {
                "max_abs_diff",
                "allclose",
                "first_mismatch_step",
            }
            if (
                not isinstance(comparison, dict)
                or not required_comparison_fields.issubset(comparison)
                or comparison["allclose"] is not True
            ):
                diagnostic = comparison if isinstance(
                    comparison,
                    dict,
                ) else {}
                raise ValueError(
                    "cached backend registered logits mismatch: "
                    f"workload={self.workload}, "
                    f"request_index={self.request_index}, "
                    "max_abs_diff="
                    f"{diagnostic.get('max_abs_diff')}, "
                    "first_mismatch_step="
                    f"{diagnostic.get('first_mismatch_step')}, "
                    "first_mismatch_engine_argmax="
                    f"{diagnostic.get('first_mismatch_engine_argmax')}, "
                    "first_mismatch_reference_argmax="
                    f"{diagnostic.get('first_mismatch_reference_argmax')}, "
                    "per_step_max_abs_diff="
                    f"{diagnostic.get('per_step_max_abs_diff')}, "
                    f"engine_output_token_ids={output}, "
                    "reference_output_token_ids="
                    f"{reference['output_token_ids']}"
                )
            if self.owns_engine:
                self._cleanup()
            prompt_tokens = len(prompt)
            row = {
                "workload": self.workload,
                "request_index": self.request_index,
                "outcome": "continuation",
                "restore_hit": expected_hit,
                "restore_reason": (
                    "exact_hit"
                    if expected_hit
                    else contract.W4_EXPECTED_REASONS[
                        self.request_index
                    ]
                ),
                "prompt_tokens": prompt_tokens,
                "reused_tokens": prompt_tokens - executed_prefill,
                "executed_prefill_tokens": executed_prefill,
                "output_token_ids": output,
                "reference_output_token_ids": list(
                    reference["output_token_ids"]
                ),
                "logits_max_abs_diff": comparison["max_abs_diff"],
                "logits_allclose": comparison["allclose"],
                "cache_identity_match": True,
                "rank_inventory": list(
                    range(len(self.configuration.gpu_indices))
                ),
                "process_group_destroyed": self.owns_engine,
                "owned_children_remaining": [],
            }
            if self.row_recorder is not None:
                self.row_recorder(row)
            return row
        finally:
            if (
                self.owns_engine
                and self.engine is not None
                and self.cleanup_receipt is None
            ):
                self.engine.exit()

    def close(self):
        self.closed = True
        if (
            self.owns_engine
            and self.engine is not None
            and self.cleanup_receipt is None
        ):
            self.engine.exit()


def _cleanup_engine(engine, world_size):
    receipt = engine.exit()
    if (
        not isinstance(receipt, dict)
        or receipt.get("process_group_destroyed") is not True
        or receipt.get("rank_exit_codes") != [0] * world_size
        or receipt.get("owned_children_remaining") != []
        or [
            row.get("rank")
            for row in receipt.get("rank_cleanup_receipts", [])
        ] != list(range(world_size))
        or any(
            row.get("process_group_destroyed") is not True
            for row in receipt["rank_cleanup_receipts"]
        )
    ):
        raise ValueError("cached backend cleanup receipt mismatch")
    return receipt
