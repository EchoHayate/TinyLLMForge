from __future__ import annotations

import argparse
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, fields, is_dataclass
import json
from pathlib import Path
import pickle


SCHEMA_VERSION = 1
REQUIRED_Q_VALUES = (1, 2, 3, 4)
REQUIRED_BATCH_SIZES = (1, 4)
OWNERSHIP_GATE_GRAPH_MAX_RESERVED_BYTES = (
    3 * 1024 * 1024 * 1024
)
REQUIRED_REPORT_FIELDS = (
    "schema_version",
    "checkpoint_path",
    "checkpoint_manifest_sha256",
    "device_name",
    "torch_version",
    "cuda_version",
    "q_values",
    "batch_sizes",
    "loader_passed",
    "fused_model_runner_path_exercised",
    "target_forward_real",
    "target_logits_cuda",
    "target_hidden_cuda",
    "target_hidden_consumed_by_real_executor",
    "target_logits_not_passed_to_mtp_executor",
    "public_result_tensor_count",
    "public_result_pickle_roundtrip",
    "public_result_tensor_free",
    "executor_identity_preserved",
    "sequence_order_preserved",
    "graph_eager_first_target_tokens_equal",
    "graph_eager_proposal_tokens_equal",
    "graph_capture_count",
    "graph_replay_count",
    "cleanup_passed",
    "backend_failures",
    "status",
    "promotion_classification",
    "coverage",
    "limitations",
)

_REQUIRED_TRUE_FIELDS = (
    "loader_passed",
    "fused_model_runner_path_exercised",
    "target_forward_real",
    "target_logits_cuda",
    "target_hidden_cuda",
    "target_hidden_consumed_by_real_executor",
    "target_logits_not_passed_to_mtp_executor",
    "public_result_pickle_roundtrip",
    "public_result_tensor_free",
    "executor_identity_preserved",
    "sequence_order_preserved",
    "graph_eager_first_target_tokens_equal",
    "graph_eager_proposal_tokens_equal",
    "cleanup_passed",
)

_REQUIRED_FALSE_COVERAGE_FIELDS = (
    "tp4",
    "kv_offload",
    "long_context",
    "second_model",
    "performance",
)

_PLAIN_SCALAR_TYPES = (
    type(None),
    bool,
    int,
    float,
    str,
    bytes,
)


@dataclass
class _LoadedFusedCallObservation:
    forward_rows: list[dict] = field(default_factory=list)
    executor_rows: list[dict] = field(default_factory=list)
    executor_identity_preserved: bool = False
    model_identity_preserved: bool = False
    restored: bool = False


class _LoadedScenarioOwner:

    def __init__(
        self,
        runner,
        executor,
        *,
        block_start: int = 0,
        sequence_factory=None,
        sampling_params_factory=None,
        allocator_factory=None,
        reset_context=None,
        bootstrap_callback=None,
    ):
        if (
            isinstance(block_start, bool)
            or not isinstance(block_start, int)
            or block_start < 0
        ):
            raise ValueError(
                "block_start must be a non-negative integer"
            )
        if sequence_factory is None:
            from tinyvllm.engine.sequence import Sequence

            sequence_factory = Sequence
        if sampling_params_factory is None:
            from tinyvllm.sampling_params import SamplingParams

            sampling_params_factory = SamplingParams
        if allocator_factory is None:
            from tinyvllm.engine.hybrid_state import (
                HybridStateSlotAllocator,
            )

            allocator_factory = HybridStateSlotAllocator
        if reset_context is None:
            from tinyvllm.utils.context import reset_context

        for name, value in (
            ("sequence_factory", sequence_factory),
            ("sampling_params_factory", sampling_params_factory),
            ("allocator_factory", allocator_factory),
            ("reset_context", reset_context),
        ):
            if not callable(value):
                raise ValueError(f"{name} must be callable")
        if (
            bootstrap_callback is not None
            and not callable(bootstrap_callback)
        ):
            raise ValueError(
                "bootstrap_callback must be callable"
            )
        kv_cache = getattr(runner, "kv_cache", None)
        shape = getattr(kv_cache, "shape", None)
        if shape is None or len(shape) < 3:
            raise ValueError(
                "runner target KV cache must expose block capacity"
            )
        block_capacity = int(shape[2])
        owner = getattr(
            runner,
            "qwen35_hybrid_model_owner",
            None,
        )
        pool = getattr(owner, "pool", None)
        state_capacity = getattr(pool, "capacity", None)
        if (
            isinstance(state_capacity, bool)
            or not isinstance(state_capacity, int)
            or state_capacity <= 0
        ):
            raise ValueError(
                "runner hybrid state capacity is unavailable"
            )
        bridge = getattr(
            runner,
            "hybrid_state_runtime_bridge",
            None,
        )
        if bridge is None or getattr(bridge, "pool", None) is None:
            raise ValueError(
                "runner hybrid state runtime bridge is unavailable"
            )
        release_sequence = getattr(
            executor,
            "release_sequence",
            None,
        )
        if not callable(release_sequence):
            raise ValueError(
                "executor.release_sequence must be callable"
            )
        self.runner = runner
        self.executor = executor
        self.block_start = block_start
        self.block_capacity = block_capacity
        self.state_capacity = state_capacity
        self.sequence_factory = sequence_factory
        self.sampling_params_factory = sampling_params_factory
        self.allocator = allocator_factory(state_capacity)
        self.reset_context = reset_context
        self.bootstrap_callback = bootstrap_callback
        self.sequences = ()
        self.leases_by_sequence_id = {}
        self.reserved_block_ids = ()
        self._cleanup_result = None
        self._allocator_active_ids = set()

    def _zero_target_kv_block(self, block_id: int) -> None:
        self.runner.kv_cache[:, :, block_id].zero_()

    def _bootstrap_target_prefill(
        self,
        sequences: tuple,
    ) -> None:
        for sequence in sequences:
            sequence.num_computed_tokens = 0
            sequence.prefill_chunk_start = 0
            sequence.prefill_chunk_end = (
                sequence.num_prompt_tokens
            )
            sequence.prefill_chunk_final = True
        if self.bootstrap_callback is None:
            run_model_step = getattr(
                self.runner,
                "_run_model_step",
                None,
            )
            if not callable(run_model_step):
                raise ValueError(
                    "runner target prefill callback is unavailable"
                )
            sampled_tokens = run_model_step(
                list(sequences),
                is_prefill=True,
                do_sample=True,
            )
        else:
            sampled_tokens = self.bootstrap_callback(sequences)
        if not isinstance(sampled_tokens, (tuple, list)):
            raise ValueError(
                "target prefill must return sampled token rows"
            )
        if len(sampled_tokens) != len(sequences):
            raise ValueError(
                "target prefill sampled token row count mismatch"
            )
        for sequence, token_id in zip(
            sequences,
            sampled_tokens,
        ):
            if (
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or token_id < 0
            ):
                raise ValueError(
                    "target prefill sampled token must be "
                    "a non-negative integer"
                )
            append_token = getattr(
                sequence,
                "append_token",
                None,
            )
            if not callable(append_token):
                raise ValueError(
                    "loaded sequence append_token is unavailable"
                )
            sequence.num_computed_tokens = (
                sequence.num_prompt_tokens
            )
            append_token(token_id)

    def build(
        self,
        q: int,
        batch_size: int,
        *,
        sequence_id_base: int,
    ) -> tuple:
        if self.sequences:
            raise RuntimeError(
                "loaded scenario owner may build only once"
            )
        for name, value in (
            ("q", q),
            ("batch_size", batch_size),
            ("sequence_id_base", sequence_id_base),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(
                    f"{name} must be a positive integer"
                )
        if q not in REQUIRED_Q_VALUES:
            raise ValueError("q is outside the approved domain")
        if batch_size not in REQUIRED_BATCH_SIZES:
            raise ValueError(
                "batch_size is outside the approved domain"
            )
        if (
            self.block_start + batch_size
            > self.block_capacity
            or batch_size > self.state_capacity
        ):
            raise ValueError(
                "loaded scenario capacity is insufficient"
            )
        block_ids = tuple(
            range(
                self.block_start,
                self.block_start + batch_size,
            )
        )
        sequences = []
        try:
            for batch_index, block_id in enumerate(block_ids):
                sequence_id = sequence_id_base + batch_index
                sampling_params = self.sampling_params_factory(
                    temperature=0.0,
                    max_tokens=q + 1,
                    ignore_eos=True,
                )
                token_id = 128 + batch_index
                sequence = self.sequence_factory(
                    [token_id],
                    sampling_params,
                )
                sequence.seq_id = sequence_id
                sequence.sequence_epoch = 0
                sequence.block_table = [block_id]
                lease = self.allocator.allocate(sequence_id)
                sequence.hybrid_state_slot_id = lease.slot_id
                sequence.hybrid_state_generation = (
                    lease.generation
                )
                self._zero_target_kv_block(block_id)
                self.leases_by_sequence_id[sequence_id] = lease
                self._allocator_active_ids.add(sequence_id)
                sequences.append(sequence)
            self.reserved_block_ids = block_ids
            self.sequences = tuple(sequences)
            self._bootstrap_target_prefill(self.sequences)
            return self.sequences
        except BaseException:
            self.reserved_block_ids = block_ids
            self.sequences = tuple(sequences)
            self.cleanup()
            raise

    def _release_runtime_binding(self, lease) -> None:
        bridge = self.runner.hybrid_state_runtime_bridge
        pool = bridge.pool
        bindings = getattr(pool, "_bindings", None)
        expected = (lease.request_id, lease.generation)
        if isinstance(bindings, dict):
            current = bindings.get(lease.slot_id)
            if current is None:
                return
            if current != expected:
                raise RuntimeError(
                    "hybrid state runtime binding drift"
                )
            bridge.release((lease,))
            return
        try:
            pool.validate(lease)
        except RuntimeError as error:
            if str(error) == "not active":
                return
            raise
        bridge.release((lease,))

    def cleanup(self) -> dict:
        if self._cleanup_result is not None:
            return self._cleanup_result
        errors = []
        for sequence in reversed(self.sequences):
            sequence_id = int(sequence.seq_id)
            sequence_epoch = int(
                getattr(sequence, "sequence_epoch", 0)
            )
            try:
                self.executor.release_sequence(
                    sequence_id,
                    sequence_epoch=sequence_epoch,
                )
            except BaseException as error:
                errors.append(
                    f"{type(error).__name__}: {error}"
                )
            lease = self.leases_by_sequence_id.get(sequence_id)
            if lease is not None:
                try:
                    self._release_runtime_binding(lease)
                except BaseException as error:
                    errors.append(
                        f"{type(error).__name__}: {error}"
                    )
                try:
                    self.allocator.release(lease)
                    self._allocator_active_ids.discard(
                        sequence_id
                    )
                except BaseException as error:
                    errors.append(
                        f"{type(error).__name__}: {error}"
                    )
        zero_failures = 0
        for block_id in reversed(self.reserved_block_ids):
            try:
                self._zero_target_kv_block(block_id)
            except BaseException as error:
                zero_failures += 1
                errors.append(
                    f"{type(error).__name__}: {error}"
                )
        try:
            self.reset_context()
        except BaseException as error:
            errors.append(f"{type(error).__name__}: {error}")
        self._cleanup_result = {
            "cleanup_passed": not errors,
            "active_leases": len(self._allocator_active_ids),
            "nonzero_target_kv_rows": zero_failures,
            "errors": errors,
        }
        return self._cleanup_result


def _canonical_positive_integer_tuple(
    value: object,
    *,
    name: str,
) -> tuple[int, ...]:
    if not isinstance(value, (tuple, list)):
        raise ValueError(f"{name} must be a tuple or list")
    result = []
    for item in value:
        if (
            isinstance(item, bool)
            or not isinstance(item, int)
            or item <= 0
        ):
            raise ValueError(
                f"{name} must contain positive integers"
            )
        result.append(item)
    if not result:
        raise ValueError(f"{name} must not be empty")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return tuple(result)


def _validate_non_negative_integer(
    value: object,
    *,
    name: str,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _validate_coverage(coverage: object) -> None:
    if not isinstance(coverage, Mapping):
        raise ValueError("coverage must be a mapping")
    if coverage.get("tp1") is not True:
        raise ValueError("coverage tp1 must be true")
    for field in _REQUIRED_FALSE_COVERAGE_FIELDS:
        if coverage.get(field) is not False:
            raise ValueError(f"coverage {field} must be false")


def _is_torch_object(value: object) -> bool:
    module_name = type(value).__module__
    return module_name == "torch" or module_name.startswith("torch.")


def _tensor_metadata(
    value: object,
    *,
    name: str,
) -> dict:
    if not _is_torch_object(value):
        raise ValueError(f"{name} must be a torch tensor")
    is_cuda = getattr(value, "is_cuda", None)
    if is_cuda is not True:
        raise ValueError(f"{name} must be CUDA-resident")
    device = str(getattr(value, "device", ""))
    if not device.startswith("cuda"):
        raise ValueError(f"{name} device must be CUDA")
    shape = getattr(value, "shape", None)
    if shape is None:
        raise ValueError(f"{name} must expose shape")
    try:
        canonical_shape = tuple(int(item) for item in shape)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{name} shape must contain integers"
        ) from error
    if not canonical_shape:
        raise ValueError(f"{name} shape must not be empty")
    return {
        "device": device,
        "dtype": str(getattr(value, "dtype", "")),
        "shape": canonical_shape,
    }


@contextmanager
def _observe_loaded_fused_call(runner, executor):
    original_run_model = getattr(runner, "run_model", None)
    original_propose_batch = getattr(
        executor,
        "propose_batch",
        None,
    )
    if not callable(original_run_model):
        raise ValueError("runner.run_model must be callable")
    if not callable(original_propose_batch):
        raise ValueError(
            "executor.propose_batch must be callable"
        )
    observation = _LoadedFusedCallObservation()
    executor_identity = id(executor)
    model = getattr(runner, "model", None)
    model_identity = id(model)

    def observed_run_model(*args, **kwargs):
        if (
            kwargs.get("execution_mode") != "decode"
            or kwargs.get("return_hidden") is not True
        ):
            raise ValueError(
                "loaded ownership forward requires decode "
                "execution and return_hidden=True"
            )
        outputs = original_run_model(*args, **kwargs)
        if (
            not isinstance(outputs, tuple)
            or len(outputs) != 2
        ):
            raise ValueError(
                "loaded ownership forward must return "
                "(logits, hidden)"
            )
        logits, hidden = outputs
        logits_metadata = _tensor_metadata(
            logits,
            name="target logits",
        )
        hidden_metadata = _tensor_metadata(
            hidden,
            name="target hidden",
        )
        if (
            logits_metadata["device"]
            != hidden_metadata["device"]
        ):
            raise ValueError(
                "target hidden/logits device mismatch"
            )
        observation.forward_rows.append({
            "return_hidden": True,
            "execution_mode": "decode",
            "logits_device": logits_metadata["device"],
            "logits_dtype": logits_metadata["dtype"],
            "logits_shape": logits_metadata["shape"],
            "hidden_device": hidden_metadata["device"],
            "hidden_dtype": hidden_metadata["dtype"],
            "hidden_shape": hidden_metadata["shape"],
        })
        return outputs

    def observed_propose_batch(inputs):
        if not isinstance(inputs, tuple) or not inputs:
            raise ValueError(
                "observed proposal inputs must be a non-empty tuple"
            )
        if not observation.forward_rows:
            raise ValueError(
                "proposal executor ran before target forward"
            )
        forward_row = observation.forward_rows[-1]
        expected_device = forward_row["hidden_device"]
        expected_dtype = forward_row["hidden_dtype"]
        expected_width = forward_row["hidden_shape"][-1]
        for input_row in inputs:
            target_hidden = getattr(
                input_row,
                "target_hidden",
                None,
            )
            hidden_metadata = _tensor_metadata(
                target_hidden,
                name="proposal target hidden",
            )
            if hidden_metadata["device"] != expected_device:
                raise ValueError(
                    "proposal target hidden device mismatch"
                )
            if hidden_metadata["dtype"] != expected_dtype:
                raise ValueError(
                    "proposal target hidden dtype mismatch"
                )
            if hidden_metadata["shape"][-1] != expected_width:
                raise ValueError(
                    "proposal target hidden width mismatch"
                )
            target_logits = getattr(
                input_row,
                "target_logits",
                None,
            )
            if target_logits is not None:
                raise ValueError(
                    "target logits must not enter the MTP executor"
                )
            observation.executor_rows.append({
                "sequence_id": int(input_row.sequence_id),
                "hidden_device": hidden_metadata["device"],
                "hidden_dtype": hidden_metadata["dtype"],
                "hidden_shape": hidden_metadata["shape"],
                "target_logits_is_none": True,
            })
        return original_propose_batch(inputs)

    runner.run_model = observed_run_model
    executor.propose_batch = observed_propose_batch
    try:
        yield observation
    finally:
        runner.run_model = original_run_model
        executor.propose_batch = original_propose_batch
        observation.executor_identity_preserved = (
            id(executor) == executor_identity
        )
        observation.model_identity_preserved = (
            id(getattr(runner, "model", None))
            == model_identity
        )
        observation.restored = (
            runner.run_model is original_run_model
            and executor.propose_batch
            is original_propose_batch
        )


def _count_tensors(value: object) -> int:
    if _is_torch_object(value):
        return 1
    if isinstance(value, _PLAIN_SCALAR_TYPES):
        return 0
    if isinstance(value, Mapping):
        return sum(
            _count_tensors(key) + _count_tensors(item)
            for key, item in value.items()
        )
    if isinstance(value, (tuple, list, set, frozenset)):
        return sum(_count_tensors(item) for item in value)
    if is_dataclass(value) and not isinstance(value, type):
        return sum(
            _count_tensors(getattr(value, field.name))
            for field in fields(value)
        )
    return 0


def _validate_plain_public_value(
    value: object,
    *,
    name: str,
) -> None:
    if _is_torch_object(value):
        raise ValueError(f"{name} contains a torch tensor")
    if isinstance(value, _PLAIN_SCALAR_TYPES):
        return
    if callable(value):
        raise ValueError(f"{name} contains a callable")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _validate_plain_public_value(
                key,
                name=f"{name} mapping key",
            )
            _validate_plain_public_value(
                item,
                name=f"{name} mapping value",
            )
        return
    if isinstance(value, (tuple, list, set, frozenset)):
        for item in value:
            _validate_plain_public_value(
                item,
                name=f"{name} sequence item",
            )
        return
    if is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            _validate_plain_public_value(
                getattr(value, field.name),
                name=f"{name}.{field.name}",
            )
        return
    raise ValueError(
        f"{name} contains unsupported public result object: "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def _canonical_public_result(rows: object) -> tuple[dict, ...]:
    from tinyvllm.speculative.adapter import DraftProposal
    from tinyvllm.speculative.batch_runtime import (
        FirstTargetProposalResult,
    )

    if not isinstance(rows, tuple):
        raise ValueError("public result must be a tuple")
    canonical = []
    for row in rows:
        for name in ("target_hidden", "target_logits"):
            if hasattr(row, name):
                raise ValueError(
                    f"public result must not expose {name}"
                )
        if not isinstance(row, FirstTargetProposalResult):
            raise ValueError(
                "public result row must be "
                "FirstTargetProposalResult"
            )
        if (
            isinstance(row.sequence_id, bool)
            or not isinstance(row.sequence_id, int)
        ):
            raise ValueError(
                "public result sequence_id must be an integer"
            )
        if (
            isinstance(row.target_token, bool)
            or not isinstance(row.target_token, int)
        ):
            raise ValueError(
                "public result target_token must be an integer"
            )
        proposal = row.proposal
        if not isinstance(proposal, DraftProposal):
            raise ValueError(
                "public result proposal must be DraftProposal"
            )
        if proposal.sequence_id != row.sequence_id:
            raise ValueError(
                "public result proposal sequence ID mismatch"
            )
        if not isinstance(proposal.token_ids, tuple):
            raise ValueError(
                "public result proposal tokens must be a tuple"
            )
        if any(
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            for token_id in proposal.token_ids
        ):
            raise ValueError(
                "public result proposal tokens must be integers"
            )
        canonical.append({
            "sequence_id": row.sequence_id,
            "target_token": row.target_token,
            "proposal_token_ids": proposal.token_ids,
            "source_type": proposal.source_type,
        })
    return tuple(canonical)


def _validate_public_result(
    rows: object,
    sequence_ids: tuple[int, ...],
) -> dict:
    from tinyvllm.engine.speculative_proposal_executor import (
        assert_tensor_free,
    )

    sequence_ids = tuple(sequence_ids)
    canonical = _canonical_public_result(rows)
    observed_ids = tuple(
        row["sequence_id"]
        for row in canonical
    )
    if observed_ids != sequence_ids:
        raise ValueError("public result sequence order mismatch")
    _validate_plain_public_value(
        rows,
        name="public result",
    )
    assert_tensor_free(rows, name="public result")
    tensor_count = _count_tensors(rows)
    if tensor_count != 0:
        raise ValueError("public result contains a torch tensor")
    round_tripped = pickle.loads(
        pickle.dumps(
            rows,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    )
    _validate_plain_public_value(
        round_tripped,
        name="round-tripped public result",
    )
    round_trip_tensor_count = _count_tensors(round_tripped)
    if round_trip_tensor_count != 0:
        raise ValueError(
            "round-tripped public result contains a torch tensor"
        )
    if _canonical_public_result(round_tripped) != canonical:
        raise ValueError(
            "public result pickle round-trip changed payload"
        )
    return {
        "tensor_count": 0,
        "tensor_free": True,
        "pickle_roundtrip": True,
        "sequence_order_preserved": True,
        "canonical_rows": canonical,
    }


def _build_fused_ownership_probe(
    runner,
    descriptor,
    executor,
    *,
    scenario_owner_factory=_LoadedScenarioOwner,
    observer_factory=_observe_loaded_fused_call,
    public_result_validator=_validate_public_result,
):
    for name, value in (
        ("scenario_owner_factory", scenario_owner_factory),
        ("observer_factory", observer_factory),
        ("public_result_validator", public_result_validator),
    ):
        if not callable(value):
            raise ValueError(f"{name} must be callable")
    call = getattr(runner, "call", None)
    if not callable(call):
        raise ValueError("runner.call must be callable")
    graph_runner = getattr(executor, "graph_runner", None)
    if graph_runner is None:
        raise ValueError(
            "executor graph runner must be installed"
        )
    graph_summary = getattr(graph_runner, "summary", None)
    if not callable(graph_summary):
        raise ValueError(
            "executor graph runner must expose summary"
        )
    graph_min_observations = getattr(
        graph_runner,
        "min_observations",
        None,
    )
    if (
        isinstance(graph_min_observations, bool)
        or not isinstance(graph_min_observations, int)
        or graph_min_observations <= 0
    ):
        raise ValueError(
            "executor graph runner min_observations must be "
            "a positive integer"
        )
    next_sequence_id = 1_000_000
    next_block_start = 0

    def summary_counts():
        summary = graph_summary()
        if not isinstance(summary, Mapping):
            raise ValueError(
                "graph runner summary must be a mapping"
            )
        counts = {}
        for field in ("captures", "replays"):
            value = summary.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(
                    f"graph runner {field} must be "
                    "a non-negative integer"
                )
            counts[field] = value
        return counts

    def run_side(q, batch_size, *, graph_enabled):
        nonlocal next_sequence_id, next_block_start
        from tinyvllm.engine.speculative_proposal_executor import (
            ProposalFinalizeRow,
        )

        owner = scenario_owner_factory(
            runner,
            executor,
            block_start=next_block_start,
        )
        sequence_id_base = next_sequence_id
        next_sequence_id += batch_size
        next_block_start += batch_size
        result = None
        rows = None
        rollback_errors = []
        try:
            sequences = owner.build(
                q,
                batch_size,
                sequence_id_base=sequence_id_base,
            )
            sequence_ids = tuple(
                int(sequence.seq_id)
                for sequence in sequences
            )
            with observer_factory(
                runner,
                executor,
            ) as observation:
                rows = runner.call(
                    "run_spec_first_target_and_proposal_batch",
                    sequences,
                    descriptor,
                    (),
                )
            public = public_result_validator(
                rows,
                sequence_ids,
            )
            result = {
                "public": public,
                "observation": observation,
                "graph_enabled": graph_enabled,
            }
        finally:
            if rows is not None:
                try:
                    finalize_rows = tuple(
                        ProposalFinalizeRow(
                            sequence_id=int(row.sequence_id),
                            proposal_transaction_id=(
                                row.proposal
                                .proposal_transaction_id
                            ),
                            accepted_proposal_tokens=0,
                        )
                        for row in rows
                    )
                    ticket_id = (
                        executor.prepare_finalize_batch(
                            finalize_rows
                        )
                    )
                    executor.rollback_finalize_batch(ticket_id)
                except BaseException as error:
                    rollback_errors.append(
                        f"{type(error).__name__}: {error}"
                    )
            cleanup = dict(owner.cleanup())
            if rollback_errors:
                cleanup["cleanup_passed"] = False
                cleanup["errors"] = list(
                    cleanup.get("errors", ())
                ) + rollback_errors
        result["cleanup"] = cleanup
        return result

    def probe(q: int, batch_size: int) -> dict:
        if q not in REQUIRED_Q_VALUES:
            raise ValueError("q is outside the gate domain")
        if batch_size not in REQUIRED_BATCH_SIZES:
            raise ValueError(
                "batch_size is outside the gate domain"
            )
        summary_before = summary_counts()
        graph_warmup_sides = []
        if q > 1:
            for _observation_index in range(
                graph_min_observations
            ):
                graph_warmup_sides.append(run_side(
                    q,
                    batch_size,
                    graph_enabled=True,
                ))
        graph_side = run_side(
            q,
            batch_size,
            graph_enabled=True,
        )
        summary_after_graph = summary_counts()
        original_graph_runner = executor.graph_runner
        try:
            executor.graph_runner = None
            eager_side = run_side(
                q,
                batch_size,
                graph_enabled=False,
            )
        finally:
            executor.graph_runner = original_graph_runner
        summary_after_eager = summary_counts()
        if summary_after_eager != summary_after_graph:
            raise ValueError(
                "eager fused side changed graph counters"
            )
        graph_rows = graph_side["public"]["canonical_rows"]
        eager_rows = eager_side["public"]["canonical_rows"]
        graph_targets = tuple(
            row["target_token"]
            for row in graph_rows
        )
        eager_targets = tuple(
            row["target_token"]
            for row in eager_rows
        )
        graph_proposals = tuple(
            row["proposal_token_ids"]
            for row in graph_rows
        )
        eager_proposals = tuple(
            row["proposal_token_ids"]
            for row in eager_rows
        )
        observations = (
            graph_side["observation"],
            eager_side["observation"],
        )
        forward_rows = tuple(
            row
            for observation in observations
            for row in observation.forward_rows
        )
        executor_rows = tuple(
            row
            for observation in observations
            for row in observation.executor_rows
        )
        return {
            "q": q,
            "batch_size": batch_size,
            "capture_count": (
                summary_after_graph["captures"]
                - summary_before["captures"]
            ),
            "replay_count": (
                summary_after_graph["replays"]
                - summary_before["replays"]
            ),
            "first_target_tokens_equal": (
                graph_targets == eager_targets
            ),
            "proposal_tokens_equal": (
                graph_proposals == eager_proposals
            ),
            "public_result_tensor_count": sum(
                side["public"]["tensor_count"]
                for side in (graph_side, eager_side)
            ),
            "public_result_tensor_free": all(
                side["public"]["tensor_free"]
                for side in (graph_side, eager_side)
            ),
            "public_result_pickle_roundtrip": all(
                side["public"]["pickle_roundtrip"]
                for side in (graph_side, eager_side)
            ),
            "sequence_order_preserved": all(
                side["public"]["sequence_order_preserved"]
                for side in (graph_side, eager_side)
            ),
            "target_logits_cuda": bool(forward_rows) and all(
                row["logits_device"].startswith("cuda")
                for row in forward_rows
            ),
            "target_hidden_cuda": bool(forward_rows) and all(
                row["hidden_device"].startswith("cuda")
                for row in forward_rows
            ),
            "target_hidden_consumed_by_real_executor": (
                len(executor_rows) == 2 * batch_size
                and all(
                    row["hidden_device"].startswith("cuda")
                    for row in executor_rows
                )
            ),
            "target_logits_not_passed_to_mtp_executor": (
                bool(executor_rows)
                and all(
                    row["target_logits_is_none"] is True
                    for row in executor_rows
                )
            ),
            "executor_identity_preserved": all(
                observation.executor_identity_preserved
                for observation in observations
            ),
            "model_identity_preserved": all(
                observation.model_identity_preserved
                for observation in observations
            ),
            "observer_restored": all(
                observation.restored
                for observation in observations
            ),
            "cleanup_passed": all(
                side["cleanup"].get("cleanup_passed") is True
                for side in (
                    *graph_warmup_sides,
                    graph_side,
                    eager_side,
                )
            ),
            "cleanup_errors": tuple(
                error
                for side in (
                    *graph_warmup_sides,
                    graph_side,
                    eager_side,
                )
                for error in side["cleanup"].get(
                    "errors",
                    (),
                )
            ),
            "graph_first_target_tokens": graph_targets,
            "eager_first_target_tokens": eager_targets,
            "graph_proposal_tokens": graph_proposals,
            "eager_proposal_tokens": eager_proposals,
        }

    return probe


class RealLoadedModelRunnerOwnershipBackend:

    _FAILURE_DOMAINS = (
        "load",
        "ownership",
        "graph_eager",
        "cleanup",
    )

    def __init__(
        self,
        *,
        runtime_loader=None,
        runtime_metadata_loader=None,
        manifest_loader=None,
        probe_builder=None,
    ):
        if runtime_loader is None:
            runtime_loader = self._load_real_runtime
        if runtime_metadata_loader is None:
            runtime_metadata_loader = (
                self._load_runtime_metadata
            )
        if manifest_loader is None:
            manifest_loader = self._load_checkpoint_manifest
        if probe_builder is None:
            probe_builder = _build_fused_ownership_probe
        for name, value in (
            ("runtime_loader", runtime_loader),
            (
                "runtime_metadata_loader",
                runtime_metadata_loader,
            ),
            ("manifest_loader", manifest_loader),
            ("probe_builder", probe_builder),
        ):
            if not callable(value):
                raise ValueError(f"{name} must be callable")
        self._runtime_loader = runtime_loader
        self._runtime_metadata_loader = (
            runtime_metadata_loader
        )
        self._manifest_loader = manifest_loader
        self._probe_builder = probe_builder
        self._runner = None
        self._descriptor = None
        self._executor = None
        self._physical_store = None
        self._model = None
        self._probe = None
        self._blockers: dict[str, str] = {}

    @staticmethod
    def _load_checkpoint_manifest(checkpoint_path: str) -> str:
        from tools.qwen35_mtp_real_checkpoint_gate import (
            checkpoint_manifest_sha256,
        )

        return checkpoint_manifest_sha256(checkpoint_path)

    @staticmethod
    def _load_runtime_metadata() -> dict:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        return {
            "device_name": torch.cuda.get_device_name(
                torch.cuda.current_device()
            ),
            "torch_version": str(torch.__version__),
            "cuda_version": str(torch.version.cuda),
        }

    @staticmethod
    def _load_real_runtime(checkpoint_path: str):
        from tools.qwen35_mtp_real_checkpoint_gate import (
            RealQwen35MTPGateBackend,
        )

        return RealQwen35MTPGateBackend._load_real_runtime(
            checkpoint_path
        )

    def _record_blocker(
        self,
        domain: str,
        reason: object,
    ) -> None:
        if domain not in self._FAILURE_DOMAINS:
            raise ValueError(
                "unsupported ownership failure domain"
            )
        text = str(reason).strip()
        if not text:
            text = "unspecified ownership failure"
        self._blockers.setdefault(domain, text)

    @staticmethod
    def _runtime_value(runtime, name: str, default=None):
        if isinstance(runtime, Mapping):
            return runtime.get(name, default)
        return getattr(runtime, name, default)

    def _install_loaded_probe(self, runtime) -> None:
        runner = self._runtime_value(runtime, "runner")
        if runner is None:
            raise ValueError("loaded ModelRunner is unavailable")
        model = getattr(runner, "model", None)
        if model is None:
            raise ValueError("loaded target model is unavailable")
        run_model = getattr(runner, "run_model", None)
        if not callable(run_model):
            raise ValueError(
                "loaded target forward is unavailable"
            )
        owner = getattr(
            runner,
            "qwen35_hybrid_model_owner",
            None,
        )
        if owner is None or getattr(owner, "pool", None) is None:
            raise ValueError(
                "loaded hybrid model owner is unavailable"
            )
        descriptor = getattr(
            runner,
            "qwen35_mtp_executor_descriptor",
            None,
        )
        if descriptor is None:
            raise ValueError(
                "loaded MTP executor descriptor is unavailable"
            )
        executor = getattr(
            runner,
            "qwen35_mtp_executor",
            None,
        )
        if executor is None:
            raise ValueError(
                "loaded MTP executor is unavailable"
            )
        physical_store = getattr(
            runner,
            "qwen35_mtp_physical_store",
            None,
        )
        if physical_store is None:
            raise ValueError(
                "loaded MTP physical store is unavailable"
            )
        graph_runner = getattr(executor, "graph_runner", None)
        if graph_runner is None:
            raise ValueError(
                "loaded MTP graph runner is unavailable"
            )
        if not callable(getattr(graph_runner, "summary", None)):
            raise ValueError(
                "loaded MTP graph summary is unavailable"
            )
        max_reserved_bytes = getattr(
            graph_runner,
            "max_reserved_bytes",
            None,
        )
        if (
            isinstance(max_reserved_bytes, bool)
            or not isinstance(max_reserved_bytes, int)
            or max_reserved_bytes <= 0
        ):
            raise ValueError(
                "loaded MTP graph reserved-byte budget is "
                "unavailable"
            )
        graph_runner.max_reserved_bytes = max(
            max_reserved_bytes,
            OWNERSHIP_GATE_GRAPH_MAX_RESERVED_BYTES,
        )
        probe = self._probe_builder(
            runner,
            descriptor,
            executor,
        )
        if not callable(probe):
            raise ValueError(
                "loaded fused ownership probe is unavailable"
            )
        self._runner = runner
        self._descriptor = descriptor
        self._executor = executor
        self._physical_store = physical_store
        self._model = model
        self._probe = probe

    def _loaded_identities_preserved(self) -> bool:
        runner = self._runner
        return (
            runner is not None
            and getattr(runner, "model", None) is self._model
            and getattr(
                runner,
                "qwen35_mtp_executor_descriptor",
                None,
            )
            is self._descriptor
            and getattr(
                runner,
                "qwen35_mtp_executor",
                None,
            )
            is self._executor
            and getattr(
                runner,
                "qwen35_mtp_physical_store",
                None,
            )
            is self._physical_store
        )

    def load(self, checkpoint_path: str) -> dict:
        manifest_before = self._manifest_loader(
            checkpoint_path
        )
        try:
            metadata = self._runtime_metadata_loader()
            if not isinstance(metadata, Mapping):
                raise ValueError(
                    "runtime metadata loader must return a mapping"
                )
        except BaseException as error:
            metadata = {
                "device_name": None,
                "torch_version": None,
                "cuda_version": None,
            }
            self._record_blocker(
                "load",
                f"{type(error).__name__}: {error}",
            )
        try:
            runtime = self._runtime_loader(checkpoint_path)
            if (
                self._runtime_value(
                    runtime,
                    "loader_passed",
                    False,
                )
                is not True
            ):
                raise ValueError(
                    "real runtime loader did not pass"
                )
            self._install_loaded_probe(runtime)
        except BaseException as error:
            self._probe = None
            self._record_blocker(
                "load",
                f"{type(error).__name__}: {error}",
            )
        manifest_after = self._manifest_loader(
            checkpoint_path
        )
        if manifest_after != manifest_before:
            self._probe = None
            self._record_blocker(
                "load",
                "checkpoint manifest changed during runtime load",
            )
        loader_passed = (
            self._probe is not None
            and self._loaded_identities_preserved()
            and "load" not in self._blockers
        )
        return {
            "checkpoint_manifest_sha256": manifest_before,
            "device_name": metadata.get("device_name"),
            "torch_version": metadata.get("torch_version"),
            "cuda_version": metadata.get("cuda_version"),
            "loader_passed": loader_passed,
            "target_forward_real": loader_passed,
        }

    def compare_fused_graph_eager(
        self,
        q: int,
        batch_size: int,
    ) -> dict:
        if not callable(self._probe):
            raise RuntimeError(
                "loaded fused ownership probe is unavailable"
            )
        if not self._loaded_identities_preserved():
            self._record_blocker(
                "ownership",
                "loaded runtime identity changed before probe",
            )
            raise RuntimeError(
                "loaded runtime identity is not preserved"
            )
        try:
            result = self._probe(q, batch_size)
            if not isinstance(result, Mapping):
                raise ValueError(
                    "loaded fused ownership probe must return "
                    "a mapping"
                )
            result = dict(result)
        except BaseException as error:
            self._record_blocker(
                "ownership",
                f"{type(error).__name__}: {error}",
            )
            raise
        if not self._loaded_identities_preserved():
            self._record_blocker(
                "ownership",
                "loaded runtime identity changed after probe",
            )
        ownership_fields = (
            "target_logits_cuda",
            "target_hidden_cuda",
            "target_hidden_consumed_by_real_executor",
            "target_logits_not_passed_to_mtp_executor",
            "public_result_tensor_free",
            "public_result_pickle_roundtrip",
            "sequence_order_preserved",
            "executor_identity_preserved",
            "model_identity_preserved",
            "observer_restored",
        )
        if any(
            result.get(field) is False
            for field in ownership_fields
        ):
            self._record_blocker(
                "ownership",
                "loaded ownership observation failed",
            )
        graph_fields = (
            "first_target_tokens_equal",
            "proposal_tokens_equal",
        )
        if any(
            result.get(field) is False
            for field in graph_fields
        ):
            self._record_blocker(
                "graph_eager",
                "loaded graph/eager token parity failed",
            )
        if result.get("cleanup_passed") is False:
            self._record_blocker(
                "cleanup",
                "loaded ownership cleanup failed",
            )
        return result

    def failures(self) -> list[str]:
        return [
            f"{domain}: {self._blockers[domain]}"
            for domain in self._FAILURE_DOMAINS
            if domain in self._blockers
        ]


def _failure_case(q: int, batch_size: int, error) -> dict:
    return {
        "q": q,
        "batch_size": batch_size,
        "capture_count": 0,
        "replay_count": 0,
        "first_target_tokens_equal": False,
        "proposal_tokens_equal": False,
        "public_result_tensor_count": 1,
        "public_result_tensor_free": False,
        "public_result_pickle_roundtrip": False,
        "sequence_order_preserved": False,
        "target_logits_cuda": False,
        "target_hidden_cuda": False,
        "target_hidden_consumed_by_real_executor": False,
        "target_logits_not_passed_to_mtp_executor": False,
        "executor_identity_preserved": False,
        "model_identity_preserved": False,
        "observer_restored": False,
        "cleanup_passed": False,
        "error": f"{type(error).__name__}: {error}",
    }


def run_gate(checkpoint_path: str, *, backend=None) -> dict:
    if not isinstance(checkpoint_path, str) or not checkpoint_path:
        raise ValueError(
            "checkpoint_path must be a non-empty string"
        )
    if backend is None:
        backend = RealLoadedModelRunnerOwnershipBackend()
    load = getattr(backend, "load", None)
    compare = getattr(
        backend,
        "compare_fused_graph_eager",
        None,
    )
    if not callable(load) or not callable(compare):
        raise ValueError(
            "ownership backend must expose load and "
            "compare_fused_graph_eager"
        )
    backend_failures = []
    try:
        metadata = load(checkpoint_path)
        if not isinstance(metadata, Mapping):
            raise ValueError(
                "ownership backend load must return a mapping"
            )
    except BaseException as error:
        metadata = {
            "checkpoint_manifest_sha256": None,
            "device_name": None,
            "torch_version": None,
            "cuda_version": None,
            "loader_passed": False,
            "target_forward_real": False,
        }
        backend_failures.append(
            f"load: {type(error).__name__}: {error}"
        )
    cases = []
    for batch_size in REQUIRED_BATCH_SIZES:
        for q in REQUIRED_Q_VALUES:
            try:
                result = compare(q, batch_size)
                if not isinstance(result, Mapping):
                    raise ValueError(
                        "ownership comparison must return a mapping"
                    )
                case = dict(result)
                if (
                    case.get("q") != q
                    or case.get("batch_size") != batch_size
                ):
                    raise ValueError(
                        "ownership comparison identity mismatch"
                    )
            except BaseException as error:
                backend_failures.append(
                    "ownership "
                    f"Q{q}/B{batch_size}: "
                    f"{type(error).__name__}: {error}"
                )
                case = _failure_case(q, batch_size, error)
            cases.append(case)
    failures = getattr(backend, "failures", None)
    if callable(failures):
        for failure in failures():
            text = str(failure).strip()
            if text:
                backend_failures.append(text)

    def all_true(field: str) -> bool:
        return all(
            case.get(field) is True
            for case in cases
        )

    public_result_tensor_count = sum(
        int(case.get("public_result_tensor_count", 1))
        for case in cases
    )
    graph_capture_count = sum(
        int(case.get("capture_count", 0))
        for case in cases
    )
    graph_replay_count = sum(
        int(case.get("replay_count", 0))
        for case in cases
    )
    graph_family_counts_valid = all(
        (
            case.get("capture_count") == 0
            and case.get("replay_count") == 0
        )
        if case["q"] == 1
        else (
            isinstance(case.get("capture_count"), int)
            and case["capture_count"] > 0
            and isinstance(case.get("replay_count"), int)
            and case["replay_count"] > 0
        )
        for case in cases
    )
    fused_path_exercised = (
        len(cases)
        == len(REQUIRED_Q_VALUES) * len(REQUIRED_BATCH_SIZES)
        and all(
            case.get("q") in REQUIRED_Q_VALUES
            and case.get("batch_size")
            in REQUIRED_BATCH_SIZES
            and "error" not in case
            for case in cases
        )
    )
    critical = {
        "loader_passed": metadata.get("loader_passed") is True,
        "fused_model_runner_path_exercised": (
            fused_path_exercised
        ),
        "target_forward_real": (
            metadata.get("target_forward_real") is True
        ),
        "target_logits_cuda": all_true(
            "target_logits_cuda"
        ),
        "target_hidden_cuda": all_true(
            "target_hidden_cuda"
        ),
        "target_hidden_consumed_by_real_executor": all_true(
            "target_hidden_consumed_by_real_executor"
        ),
        "target_logits_not_passed_to_mtp_executor": all_true(
            "target_logits_not_passed_to_mtp_executor"
        ),
        "public_result_pickle_roundtrip": all_true(
            "public_result_pickle_roundtrip"
        ),
        "public_result_tensor_free": all_true(
            "public_result_tensor_free"
        ),
        "executor_identity_preserved": all_true(
            "executor_identity_preserved"
        ),
        "sequence_order_preserved": all_true(
            "sequence_order_preserved"
        ),
        "graph_eager_first_target_tokens_equal": all_true(
            "first_target_tokens_equal"
        ),
        "graph_eager_proposal_tokens_equal": all_true(
            "proposal_tokens_equal"
        ),
        "cleanup_passed": all_true("cleanup_passed"),
    }
    status = (
        "PASS"
        if (
            not backend_failures
            and graph_family_counts_valid
            and graph_capture_count == 6
            and graph_replay_count >= 6
            and public_result_tensor_count == 0
            and all(critical.values())
        )
        else "FAIL"
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_path": checkpoint_path,
        "checkpoint_manifest_sha256": metadata.get(
            "checkpoint_manifest_sha256"
        ),
        "device_name": metadata.get("device_name"),
        "torch_version": metadata.get("torch_version"),
        "cuda_version": metadata.get("cuda_version"),
        "q_values": list(REQUIRED_Q_VALUES),
        "batch_sizes": list(REQUIRED_BATCH_SIZES),
        **critical,
        "public_result_tensor_count": (
            public_result_tensor_count
        ),
        "graph_capture_count": graph_capture_count,
        "graph_replay_count": graph_replay_count,
        "backend_failures": backend_failures,
        "cases": cases,
        "status": status,
        "promotion_classification": "NOT_PROMOTABLE",
        "coverage": {
            "tp1": True,
            "tp4": False,
            "kv_offload": False,
            "long_context": False,
            "second_model": False,
            "performance": False,
        },
        "limitations": [
            "TP1 only",
            "KV offload disabled",
            "single Qwen3.5 architecture",
            "Q values limited to 1 through 4",
            "batch sizes limited to 1 and 4",
            "no long-context coverage",
            "no performance claim",
        ],
    }
    if status == "PASS":
        validate_ownership_gate_report(
            report,
            required_q_values=REQUIRED_Q_VALUES,
            required_batch_sizes=REQUIRED_BATCH_SIZES,
        )
    return report


def main(argv=None, *, backend=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    report = run_gate(args.checkpoint, backend=backend)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def validate_ownership_gate_report(
    report: object,
    *,
    required_q_values: tuple[int, ...],
    required_batch_sizes: tuple[int, ...],
) -> None:
    if not isinstance(report, Mapping):
        raise ValueError("ownership gate report must be a mapping")
    missing = [
        field
        for field in REQUIRED_REPORT_FIELDS
        if field not in report
    ]
    if missing:
        raise ValueError(
            "ownership gate report is missing required fields: "
            + ", ".join(missing)
        )
    if report["schema_version"] != SCHEMA_VERSION:
        raise ValueError("schema_version is unsupported")
    expected_q_values = _canonical_positive_integer_tuple(
        required_q_values,
        name="required_q_values",
    )
    expected_batch_sizes = _canonical_positive_integer_tuple(
        required_batch_sizes,
        name="required_batch_sizes",
    )
    if (
        _canonical_positive_integer_tuple(
            report["q_values"],
            name="q_values",
        )
        != expected_q_values
    ):
        raise ValueError("q_values do not cover the required domain")
    if (
        _canonical_positive_integer_tuple(
            report["batch_sizes"],
            name="batch_sizes",
        )
        != expected_batch_sizes
    ):
        raise ValueError(
            "batch_sizes do not cover the required domain"
        )
    for field in _REQUIRED_TRUE_FIELDS:
        if report[field] is not True:
            raise ValueError(f"{field} must be true")
    if report["public_result_tensor_count"] != 0:
        raise ValueError(
            "public_result_tensor_count must equal zero"
        )
    _validate_non_negative_integer(
        report["graph_capture_count"],
        name="graph_capture_count",
    )
    _validate_non_negative_integer(
        report["graph_replay_count"],
        name="graph_replay_count",
    )
    backend_failures = report["backend_failures"]
    if not isinstance(backend_failures, list):
        raise ValueError("backend_failures must be a list")
    if backend_failures:
        raise ValueError("backend_failures must be empty")
    if report["status"] != "PASS":
        raise ValueError("status must be PASS")
    if report["promotion_classification"] != "NOT_PROMOTABLE":
        raise ValueError(
            "promotion_classification must remain NOT_PROMOTABLE"
        )
    _validate_coverage(report["coverage"])
    limitations = report["limitations"]
    if (
        not isinstance(limitations, list)
        or not limitations
        or any(
            not isinstance(value, str) or not value.strip()
            for value in limitations
        )
    ):
        raise ValueError(
            "limitations must be a non-empty list of strings"
        )


if __name__ == "__main__":
    main()
