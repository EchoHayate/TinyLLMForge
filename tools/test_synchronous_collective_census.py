from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT / "tinyvllm/engine/synchronous_collective_census.py"
)


def _load():
    spec = importlib.util.spec_from_file_location(
        "synchronous_collective_census",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


census_module = _load()
CollectiveCensusPolicy = census_module.CollectiveCensusPolicy
SynchronousCollectiveCensus = census_module.SynchronousCollectiveCensus
activate_synchronous_collective_census = (
    census_module.activate_synchronous_collective_census
)
active_synchronous_collective_census = (
    census_module.active_synchronous_collective_census
)
census_layer = census_module.census_layer
observe_synchronous_collective = (
    census_module.observe_synchronous_collective
)
run_census_step = census_module.run_census_step


class FakeTensor:
    shape = (2, 4)
    dtype = "torch.bfloat16"


class FakeEvent:

    def __init__(self, elapsed_ms):
        self.elapsed_ms = elapsed_ms
        self.record_count = 0

    def record(self):
        self.record_count += 1

    def elapsed_time(self, other):
        return other.elapsed_ms - self.elapsed_ms


class FakeEvents:

    def __init__(self, elapsed_values=()):
        self.values = iter(elapsed_values)
        self.created = []

    def __call__(self):
        event = FakeEvent(next(self.values))
        self.created.append(event)
        return event


def _policy(
    *,
    sample_budget=0,
    expected_collective_count=130,
):
    return CollectiveCensusPolicy(
        sample_budget=sample_budget,
        cohort_count=17,
        expected_collective_count=expected_collective_count,
        source_revision="a" * 40,
        attempt="attempt-r1",
        workload="P0",
        repetition=2,
    )


def _census(
    *,
    sample_budget=0,
    expected_collective_count=130,
    elapsed_values=(),
):
    events = FakeEvents(elapsed_values)
    synchronizations = []
    census = SynchronousCollectiveCensus(
        rank=2,
        policy=_policy(
            sample_budget=sample_budget,
            expected_collective_count=expected_collective_count,
        ),
        event_factory=events,
        synchronize=lambda: synchronizations.append(True),
        stream_resolver=lambda: "cuda:2:stream:7",
    )
    return census, events, synchronizations


def _run_decode_step(census, call):
    return run_census_step(
        census,
        batch_kind="decode",
        is_decode=True,
        active_sequence_count=1,
        request_set_sha256="b" * 64,
        dispatch="eager",
        call=call,
    )


def _observe_embedding(calls):
    tensor = FakeTensor()
    result = observe_synchronous_collective(
        site_role="vocab_parallel_embedding",
        operation="vocab_parallel_embedding_all_reduce",
        tensor=tensor,
        call=lambda value: calls.append(value) or "done",
        collective_kind="all_reduce",
        process_group="tensor_parallel",
        execution_phase="decode_or_prefill",
        async_mode=False,
        source_rank=None,
        destination_rank=None,
    )
    return tensor, result


def test_policy_selects_a_stable_bounded_cohort():
    policy = _policy(sample_budget=8)

    first = policy.sampled_ordinals(
        decode_ordinal=5,
        collective_count=130,
    )
    second = policy.sampled_ordinals(
        decode_ordinal=5,
        collective_count=130,
    )

    assert first == second
    assert len(first) == 8
    assert len(set(first)) == 8
    assert min(first) >= 0
    assert max(first) < 130


@pytest.mark.parametrize("budget", [-1, 1, 33])
def test_policy_rejects_unsupported_event_budget(budget):
    with pytest.raises(ValueError, match="sample_budget"):
        _policy(sample_budget=budget)


def test_policy_rejects_invalid_identity_or_expected_count():
    with pytest.raises(ValueError, match="source_revision"):
        CollectiveCensusPolicy(
            sample_budget=0,
            cohort_count=17,
            expected_collective_count=130,
            source_revision="not-a-revision",
            attempt="attempt-r1",
            workload="P0",
            repetition=0,
        )
    with pytest.raises(ValueError, match="expected_collective_count"):
        _policy(expected_collective_count=0)


def test_count_only_observation_calls_collective_once_without_events():
    census, events, synchronizations = _census()
    calls = []

    with activate_synchronous_collective_census(census):
        tensor, result = _run_decode_step(
            census,
            lambda: _observe_embedding(calls),
        )

    snapshot = census.finalize()

    assert result == "done"
    assert calls == [tensor]
    assert events.created == []
    assert synchronizations == []
    assert len(snapshot["collectives"]) == 1
    row = snapshot["collectives"][0]
    assert row["site_id"] == "embedding.input"
    assert row["tensor_shape"] == [2, 4]
    assert row["tensor_dtype"] == "torch.bfloat16"
    assert row["tensor_bytes"] == 16
    assert row["event_sampled"] is False
    assert row["cuda_ns"] is None


def test_sampled_observation_records_exactly_two_events():
    census, events, synchronizations = _census(
        sample_budget=8,
        expected_collective_count=8,
        elapsed_values=(1.0, 1.25),
    )
    calls = []

    with activate_synchronous_collective_census(census):
        _run_decode_step(census, lambda: _observe_embedding(calls))

    snapshot = census.finalize()

    assert len(calls) == 1
    assert len(events.created) == 2
    assert [event.record_count for event in events.created] == [1, 1]
    assert synchronizations == [True]
    assert snapshot["collectives"][0]["event_sampled"] is True
    assert snapshot["collectives"][0]["cuda_ns"] == 250_000


def test_non_decode_step_records_no_collective():
    census, events, _ = _census(sample_budget=8)
    calls = []

    with activate_synchronous_collective_census(census):
        run_census_step(
            census,
            batch_kind="prefill",
            is_decode=False,
            active_sequence_count=1,
            request_set_sha256="b" * 64,
            dispatch="eager",
            call=lambda: _observe_embedding(calls),
        )

    snapshot = census.finalize()

    assert len(calls) == 1
    assert events.created == []
    assert snapshot["steps"] == []
    assert snapshot["collectives"] == []


def test_layer_context_builds_stable_row_parallel_site_id():
    census, _, _ = _census()
    calls = []

    def execute():
        with census_layer(3, "full_attention"):
            return observe_synchronous_collective(
                site_role="row_parallel_output",
                operation="row_parallel_all_reduce",
                tensor=FakeTensor(),
                call=lambda value: calls.append(value) or value,
                collective_kind="all_reduce",
                process_group="tensor_parallel",
                execution_phase="decode_or_prefill",
                async_mode=False,
                source_rank=None,
                destination_rank=None,
            )

    with activate_synchronous_collective_census(census):
        _run_decode_step(census, execute)

    row = census.finalize()["collectives"][0]
    assert row["site_id"] == "layer.003.attention.output"
    assert row["layer_index"] == 3
    assert row["layer_role"] == "full_attention"
    assert len(calls) == 1


def test_unknown_site_fails_before_collective_call():
    census, _, _ = _census()
    calls = []

    with activate_synchronous_collective_census(census):
        with pytest.raises(ValueError, match="site_role"):
            _run_decode_step(
                census,
                lambda: observe_synchronous_collective(
                    site_role="unknown",
                    operation="unknown",
                    tensor=FakeTensor(),
                    call=lambda value: calls.append(value),
                    collective_kind="all_reduce",
                    process_group="tensor_parallel",
                    execution_phase="decode",
                    async_mode=False,
                    source_rank=None,
                    destination_rank=None,
                ),
            )

    assert calls == []


def test_async_mode_is_rejected_before_collective_call():
    census, _, _ = _census()
    calls = []

    with activate_synchronous_collective_census(census):
        with pytest.raises(ValueError, match="async_mode"):
            _run_decode_step(
                census,
                lambda: observe_synchronous_collective(
                    site_role="vocab_parallel_embedding",
                    operation="vocab_parallel_embedding_all_reduce",
                    tensor=FakeTensor(),
                    call=lambda value: calls.append(value),
                    collective_kind="all_reduce",
                    process_group="tensor_parallel",
                    execution_phase="decode",
                    async_mode=True,
                    source_rank=None,
                    destination_rank=None,
                ),
            )

    assert calls == []


def test_operation_exception_is_not_retried_and_step_closes():
    census, _, _ = _census()
    calls = []
    error = RuntimeError("collective failed")

    def fail(value):
        calls.append(value)
        raise error

    with activate_synchronous_collective_census(census):
        with pytest.raises(RuntimeError) as exc_info:
            _run_decode_step(
                census,
                lambda: observe_synchronous_collective(
                    site_role="vocab_parallel_embedding",
                    operation="vocab_parallel_embedding_all_reduce",
                    tensor=FakeTensor(),
                    call=fail,
                    collective_kind="all_reduce",
                    process_group="tensor_parallel",
                    execution_phase="decode",
                    async_mode=False,
                    source_rank=None,
                    destination_rank=None,
                ),
            )

    assert exc_info.value is error
    assert len(calls) == 1
    snapshot = census.finalize()
    assert snapshot["steps"][0]["status"] == "failed"
    assert snapshot["collectives"][0]["status"] == "failed"


def test_finalize_rejects_open_step_or_layer():
    census, _, _ = _census()
    census.begin_step(
        batch_kind="decode",
        is_decode=True,
        active_sequence_count=1,
        request_set_sha256="b" * 64,
        dispatch="eager",
    )
    with pytest.raises(RuntimeError, match="open census scope"):
        census.finalize()
    census.end_step()

    with activate_synchronous_collective_census(census):
        with census_layer(0, "mlp"):
            with pytest.raises(RuntimeError, match="open census scope"):
                census.finalize()


def test_finalize_synchronizes_once_and_is_idempotent():
    census, _, synchronizations = _census(
        sample_budget=8,
        expected_collective_count=8,
        elapsed_values=(2.0, 2.5),
    )

    with activate_synchronous_collective_census(census):
        _run_decode_step(census, lambda: _observe_embedding([]))

    first = census.finalize()
    second = census.finalize()

    assert first == second
    assert synchronizations == [True]
    assert all("tensor" not in row for row in first["collectives"])


def test_activation_restores_previous_census():
    outer, _, _ = _census()
    inner, _, _ = _census()

    assert active_synchronous_collective_census() is None
    with activate_synchronous_collective_census(outer):
        assert active_synchronous_collective_census() is outer
        with activate_synchronous_collective_census(inner):
            assert active_synchronous_collective_census() is inner
        assert active_synchronous_collective_census() is outer
    assert active_synchronous_collective_census() is None
