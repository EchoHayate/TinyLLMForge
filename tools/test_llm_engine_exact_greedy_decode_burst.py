"""Dependency-light LLMEngine exact greedy decode burst tests."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
ENGINE_PATH = ROOT / "tinyvllm" / "engine" / "llm_engine.py"
BURST_PATH = (
    ROOT
    / "tinyvllm"
    / "engine"
    / "exact_greedy_decode_burst.py"
)


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


burst_module = _load_module(
    "llm_engine_exact_burst_contract_under_test",
    BURST_PATH,
)
ExactGreedyDecodeBurstFallback = (
    burst_module.ExactGreedyDecodeBurstFallback
)
ExactGreedyDecodeBurstResult = (
    burst_module.ExactGreedyDecodeBurstResult
)
build_exact_greedy_decode_burst_lease = (
    burst_module.build_exact_greedy_decode_burst_lease
)
validate_exact_greedy_decode_burst_result = (
    burst_module.validate_exact_greedy_decode_burst_result
)


def _partition(record, seqs, *, expected_schedule_generation):
    del record
    return SimpleNamespace(
        schedule_generation=expected_schedule_generation,
        selected_sequence_ids=(),
        suppressed_sequence_ids=tuple(
            sequence.seq_id for sequence in seqs
        ),
        selected_sequences=(),
        suppressed_sequences=tuple(seqs),
    )


def _load_step(partition_builder=_partition):
    tree = ast.parse(ENGINE_PATH.read_text(), filename=str(ENGINE_PATH))
    engine_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "LLMEngine"
    )
    method = next(
        node
        for node in engine_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "step"
    )
    function = ast.FunctionDef(
        name=method.name,
        args=method.args,
        body=method.body,
        decorator_list=[],
        returns=method.returns,
        type_comment=method.type_comment,
    )
    namespace = {
        "ExactGreedyDecodeBurstFallback": (
            ExactGreedyDecodeBurstFallback
        ),
        "validate_exact_greedy_decode_burst_result": (
            validate_exact_greedy_decode_burst_result
        ),
        "build_engine_speculative_partition": partition_builder,
    }
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[function], type_ignores=[])
            ),
            str(ENGINE_PATH),
            "exec",
        ),
        namespace,
    )
    return namespace["step"]


def _load_generate():
    tree = ast.parse(ENGINE_PATH.read_text(), filename=str(ENGINE_PATH))
    engine_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "LLMEngine"
    )
    method = next(
        node
        for node in engine_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "generate"
    )
    function = ast.FunctionDef(
        name=method.name,
        args=method.args,
        body=method.body,
        decorator_list=[],
        returns=method.returns,
        type_comment=method.type_comment,
    )
    namespace = {"perf_counter": lambda: 0.0}
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[function], type_ignores=[])
            ),
            str(ENGINE_PATH),
            "exec",
        ),
        namespace,
    )
    return namespace["generate"]


def _step_ast():
    tree = ast.parse(ENGINE_PATH.read_text(), filename=str(ENGINE_PATH))
    engine_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "LLMEngine"
    )
    return next(
        node
        for node in engine_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "step"
    )


class _Clock:
    def __init__(self, *values):
        self._values = iter(values)
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return next(self._values)


class _Sequence:
    seq_id = 7
    num_prompt_tokens = 2
    prefill_chunk_start = 0
    prefill_chunk_end = 0
    prefill_chunk_final = False
    step_is_decode = True
    step_do_sample = True
    sequence_epoch = 0
    temperature = 0.0
    ignore_eos = True
    max_tokens = 3

    def __init__(self):
        self.token_ids = [1, 2]
        self.block_table = [3]
        self.status = "running"

    def __len__(self):
        return len(self.token_ids)

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens :]

    @property
    def num_completion_tokens(self):
        return len(self.completion_token_ids)

    @property
    def last_token(self):
        return self.token_ids[-1]

    @property
    def is_finished(self):
        return self.status == "finished"


def _lease(*, width=3):
    return build_exact_greedy_decode_burst_lease(
        sequence_id=7,
        schedule_generation=11,
        graph_generation=13,
        requested_token_count=4,
        authorized_token_count=width,
        initial_completion_count=0,
        initial_sequence_length=2,
        block_table_identity=((3, 5),),
        write_block_id=3,
        write_block_generation=5,
        first_write_position=1,
        last_write_position=width,
        first_physical_slot=49,
        last_physical_slot=48 + width,
        remaining_output_tokens=3,
        completion_only=True,
    )


def _result(
    lease,
    *,
    tokens=(41, 42, 43),
    identity=None,
    sampled_logits=(),
):
    return ExactGreedyDecodeBurstResult(
        lease_identity_sha256=(
            lease.identity_sha256 if identity is None else identity
        ),
        tokens=tuple(tokens),
        replay_count=len(tokens),
        final_input_token=tokens[-1],
        final_position=lease.first_write_position + len(tokens),
        final_context_length=(
            lease.initial_sequence_length + len(tokens)
        ),
        final_physical_slot=(
            lease.first_physical_slot + len(tokens)
        ),
        graph_identity_sha256="a" * 64,
        token_d2h_calls=1,
        sampled_logit_d2h_calls=int(bool(sampled_logits)),
        sampled_logits=sampled_logits,
    )


class _Scheduler:
    last_policy_branch = "decode"
    last_speculative_selection = None
    schedule_generation = 11

    def __init__(
        self,
        sequence,
        *,
        lease,
        is_prefill=False,
        do_sample=True,
        batch_kind=None,
        previous_fallback_counts=None,
        no_lease_reason="waiting_present",
    ):
        self.sequence = sequence
        self.lease = lease
        self.is_prefill = is_prefill
        self.do_sample = do_sample
        self.batch_kind = batch_kind
        self.events = []
        self.release_count = 0
        self.pending_leases = int(lease is not None)
        self.fallback_reason = None
        self.fallback_counts = dict(
            previous_fallback_counts or {}
        )
        self.no_lease_reason = no_lease_reason
        self.host_visible_gap_ns = 0

    def observation_snapshot(self):
        return {"running_seq_ids": [self.sequence.seq_id]}

    def schedule(self, decision_now_ns):
        self.events.append(("schedule", decision_now_ns))
        if self.batch_kind is None:
            return (
                [self.sequence],
                self.is_prefill,
                self.do_sample,
            )
        return (
            [self.sequence],
            self.is_prefill,
            self.do_sample,
            self.batch_kind,
        )

    def prepare_exact_greedy_decode_burst(
        self,
        seqs,
        **kwargs,
    ):
        self.events.append(("lease", tuple(seqs), kwargs))
        if self.lease is None:
            self.fallback_reason = self.no_lease_reason
            self.fallback_counts[self.no_lease_reason] = (
                self.fallback_counts.get(self.no_lease_reason, 0)
                + 1
            )
        return self.lease

    def cancel_exact_greedy_decode_burst(self, lease, reason):
        assert lease is self.lease
        self.events.append(("cancel", lease, reason))
        self.pending_leases = 0
        self.fallback_reason = reason
        self.fallback_counts[reason] = (
            self.fallback_counts.get(reason, 0) + 1
        )

    def fail_exact_greedy_decode_burst(self, lease, *, terminal):
        assert lease is self.lease
        self.events.append(("fail", lease, terminal))
        if terminal:
            self.pending_leases = 0

    def prepare_exact_greedy_decode_burst_commit(
        self,
        seqs,
        lease,
        result,
        **kwargs,
    ):
        self.events.append(
            ("prepare_commit", tuple(seqs), lease, result, kwargs)
        )
        return SimpleNamespace(
            seqs=tuple(seqs),
            lease=lease,
            result=result,
            kwargs=kwargs,
        )

    def commit_prepared_postprocess(self, prepared):
        self.events.append(("commit", prepared))
        for token in prepared.result.tokens:
            self.sequence.token_ids.append(token)
        if (
            self.sequence.num_completion_tokens
            == self.sequence.max_tokens
        ):
            self.sequence.status = "finished"
            self.release_count += 1
        self.pending_leases = 0
        self.host_visible_gap_ns = prepared.kwargs[
            "host_visible_gap_ns"
        ]

    def drain_hybrid_state_release_events(self):
        return ()

    def postprocess(
        self,
        seqs,
        token_ids,
        is_prefill,
        do_sample,
        batch_kind,
        *,
        decision_now_ns,
        step_end_ns,
    ):
        self.events.append(
            (
                "ordinary_postprocess",
                tuple(seqs),
                tuple(token_ids),
                is_prefill,
                do_sample,
                batch_kind,
                decision_now_ns,
                step_end_ns,
            )
        )
        for token in token_ids:
            self.sequence.token_ids.append(token)

    def last_slo_observation(self):
        return {
            "decision_now_ns": 10,
            "step_end_ns": 30,
            "actual_step_duration_ns": 20,
        }

    def exact_greedy_decode_burst_summary(self):
        return {
            "pending_leases": self.pending_leases,
            "maximum_host_visible_gap_ns": (
                self.host_visible_gap_ns
            ),
            "fallback_counts": (
                dict(self.fallback_counts)
            ),
        }


class _ModelRunner:
    rank = 0
    world_size = 1

    def __init__(self, outcome, *, enabled=True):
        self.outcome = outcome
        self.config = SimpleNamespace(
            exact_greedy_decode_burst=enabled,
            exact_greedy_decode_burst_tokens=4,
        )
        self.events = []
        self.quarantine_reason = None

    def exact_greedy_decode_burst_capability(
        self,
        *,
        correctness_trace=False,
    ):
        self.events.append(("capability", correctness_trace))
        return {
            "available": True,
            "fallback_reason": None,
            "graph_identity_sha256": "a" * 64,
            "graph_generation": 13,
            "correctness_trace": correctness_trace,
        }

    def run_exact_greedy_decode_burst(self, seqs, lease):
        raise AssertionError(
            "exact burst execution must use ModelRunner.call"
        )

    def call(self, method_name, *args):
        if method_name == "run_exact_greedy_decode_burst":
            if len(args) == 2:
                seqs, lease = args
                correctness_trace = False
            else:
                seqs, lease, correctness_trace = args
            self.events.append(
                (
                    "burst",
                    tuple(seqs),
                    lease,
                    correctness_trace,
                )
            )
            if isinstance(self.outcome, BaseException):
                self.quarantine_reason = (
                    "replay_failure:" + type(self.outcome).__name__
                )
                raise self.outcome
            return self.outcome
        if method_name == "run":
            self.events.append(("ordinary", args))
            return [99]
        raise AssertionError(
            f"unexpected ModelRunner method {method_name}"
        )

    def exact_greedy_decode_burst_summary(self):
        return {
            "quarantine_reason": self.quarantine_reason,
        }

    def memory_snapshot(self):
        return {"cuda_allocated_bytes": 123}


def _engine(
    outcome,
    *,
    lease=None,
    enabled=True,
    is_prefill=False,
    do_sample=True,
    batch_kind=None,
    partition_builder=_partition,
    previous_fallback_counts=None,
    no_lease_reason="waiting_present",
):
    sequence = _Sequence()
    scheduler = _Scheduler(
        sequence,
        lease=lease,
        is_prefill=is_prefill,
        do_sample=do_sample,
        batch_kind=batch_kind,
        previous_fallback_counts=previous_fallback_counts,
        no_lease_reason=no_lease_reason,
    )
    model_runner = _ModelRunner(outcome, enabled=enabled)
    return (
        SimpleNamespace(
            _clock_ns=_Clock(10, 30),
            scheduler=scheduler,
            model_runner=model_runner,
            speculative_runtime=None,
            speculative_runtime_poisoned=False,
            speculative_runtime_poison_reason=None,
            last_batch_kind=None,
            last_scheduled_seqs=[],
            last_step_observation=None,
        ),
        sequence,
        scheduler,
        model_runner,
        _load_step(partition_builder),
    )


def test_eligible_decode_runs_one_burst_and_commits_ordered_delta_once():
    lease = _lease()
    result = _result(lease)
    engine, sequence, scheduler, model_runner, step = _engine(
        result,
        lease=lease,
    )

    outputs, num_tokens = step(engine, completion_only=True)

    assert outputs == [(7, [41, 42, 43])]
    assert num_tokens == -3
    assert [event[0] for event in scheduler.events].count("lease") == 1
    assert [event[0] for event in model_runner.events].count("burst") == 1
    assert not any(
        event[0] == "ordinary" for event in model_runner.events
    )
    assert [event[0] for event in scheduler.events].count("commit") == 1
    assert not any(
        event[0] == "ordinary_postprocess"
        for event in scheduler.events
    )
    assert sequence.completion_token_ids == [41, 42, 43]
    assert scheduler.release_count == 1
    assert engine.last_step_observation[
        "new_completion_tokens_by_seq"
    ] == {7: [41, 42, 43]}


def test_pre_replay_fallback_cancels_lease_and_runs_ordinary_once():
    lease = _lease()
    fallback = ExactGreedyDecodeBurstFallback(
        "static_state_bind_failure"
    )
    engine, sequence, scheduler, model_runner, step = _engine(
        fallback,
        lease=lease,
    )

    outputs, num_tokens = step(engine, completion_only=True)

    assert outputs == []
    assert num_tokens == -1
    assert sequence.completion_token_ids == [99]
    assert [event[0] for event in model_runner.events].count(
        "ordinary"
    ) == 1
    assert (
        "cancel",
        lease,
        "static_state_bind_failure",
    ) in scheduler.events
    assert [event[0] for event in scheduler.events].count(
        "ordinary_postprocess"
    ) == 1
    assert engine.last_step_observation[
        "exact_greedy_decode_burst_fallback_reason"
    ] == "static_state_bind_failure"
    assert engine.last_step_observation[
        "exact_greedy_decode_burst_accepted"
    ] is True
    assert engine.last_step_observation[
        "exact_greedy_decode_burst_width"
    ] == 3
    assert engine.last_step_observation[
        "exact_greedy_decode_burst_replay_count"
    ] == 0


def test_stale_result_identity_fails_before_scheduler_commit():
    lease = _lease()
    result = _result(lease, identity="b" * 64)
    engine, _sequence, scheduler, model_runner, step = _engine(
        result,
        lease=lease,
    )

    with pytest.raises(
        ValueError,
        match="burst result lease identity mismatch",
    ):
        step(engine, completion_only=True)

    assert not any(
        event[0] == "prepare_commit"
        for event in scheduler.events
    )
    assert [event[0] for event in scheduler.events].count("fail") == 1
    assert not any(
        event[0] == "ordinary" for event in model_runner.events
    )


def test_post_replay_exception_is_terminal_and_never_falls_back():
    lease = _lease()
    error = RuntimeError("replay exploded")
    engine, _sequence, scheduler, model_runner, step = _engine(
        error,
        lease=lease,
    )

    with pytest.raises(RuntimeError, match="replay exploded"):
        step(engine, completion_only=True)

    assert [event[0] for event in scheduler.events].count("fail") == 1
    assert not any(
        event[0] in {"ordinary", "cancel"}
        for event in model_runner.events + scheduler.events
    )


@pytest.mark.parametrize(
    (
        "enabled",
        "is_prefill",
        "do_sample",
        "batch_kind",
        "selected",
        "lease_available",
    ),
    (
        (False, False, True, None, False, True),
        (True, True, True, None, False, True),
        (True, False, False, None, False, True),
        (True, False, True, "mixed", False, True),
        (True, False, True, None, False, False),
    ),
)
def test_ineligible_or_contended_rows_preserve_ordinary_path(
    enabled,
    is_prefill,
    do_sample,
    batch_kind,
    selected,
    lease_available,
):
    lease = _lease() if lease_available else None

    def partition_builder(
        record,
        seqs,
        *,
        expected_schedule_generation,
    ):
        del record
        sequence = seqs[0]
        return SimpleNamespace(
            schedule_generation=expected_schedule_generation,
            selected_sequence_ids=(
                (sequence.seq_id,) if selected else ()
            ),
            suppressed_sequence_ids=(
                () if selected else (sequence.seq_id,)
            ),
            selected_sequences=((sequence,) if selected else ()),
            suppressed_sequences=(() if selected else tuple(seqs)),
        )

    engine, sequence, scheduler, model_runner, step = _engine(
        _result(lease or _lease()),
        lease=lease,
        enabled=enabled,
        is_prefill=is_prefill,
        do_sample=do_sample,
        batch_kind=batch_kind,
        partition_builder=partition_builder,
    )

    outputs, num_tokens = step(engine, completion_only=True)

    assert outputs == []
    assert num_tokens == (
        1 if batch_kind == "mixed" else (0 if is_prefill else -1)
    )
    assert sequence.completion_token_ids == [99]
    assert not any(
        event[0] == "burst" for event in model_runner.events
    )
    assert [event[0] for event in model_runner.events].count(
        "ordinary"
    ) == 1


def test_burst_dispatch_is_confined_to_non_speculative_branch():
    step = _step_ast()
    parent_by_node = {
        child: parent
        for parent in ast.walk(step)
        for child in ast.iter_child_nodes(parent)
    }
    burst_call = next(
        node
        for node in ast.walk(step)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "call"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "run_exact_greedy_decode_burst"
    )
    current = burst_call
    while current in parent_by_node:
        parent = parent_by_node[current]
        if (
            isinstance(parent, ast.If)
            and isinstance(parent.test, ast.Attribute)
            and parent.test.attr == "selected_sequences"
        ):
            assert any(
                current is child
                or current in tuple(ast.walk(child))
                for child in parent.orelse
            )
            break
        current = parent
    else:
        raise AssertionError(
            "exact burst dispatch is not guarded by the "
            "non-speculative branch"
        )


def test_direct_step_without_completion_only_authority_never_bursts():
    lease = _lease()
    engine, sequence, scheduler, model_runner, step = _engine(
        _result(lease),
        lease=lease,
    )

    step(engine)

    assert sequence.completion_token_ids == [99]
    assert not any(
        event[0] in {"capability", "burst"}
        for event in model_runner.events
    )
    assert not any(
        event[0] == "lease" for event in scheduler.events
    )


@pytest.mark.parametrize("step_error", (None, RuntimeError("step failed")))
def test_generate_scopes_completion_only_authority(step_error):
    generate = _load_generate()
    observations = []

    class Engine:
        tokenizer = SimpleNamespace(
            decode=lambda token_ids: str(token_ids)
        )

        def __init__(self):
            self.finished = False

        def add_request(self, prompt, sampling_params):
            pass

        def is_finished(self):
            return self.finished

        def step(self, *, completion_only=False):
            observations.append(completion_only)
            self.finished = True
            if step_error is not None:
                raise step_error
            return ([(0, [41])], -1)

    engine = Engine()
    if step_error is None:
        assert generate(
            engine,
            [[1, 2]],
            SimpleNamespace(),
            use_tqdm=False,
        ) == [{"text": "[41]", "token_ids": [41]}]
    else:
        with pytest.raises(RuntimeError, match="step failed"):
            generate(
                engine,
                [[1, 2]],
                SimpleNamespace(),
                use_tqdm=False,
            )

    assert observations == [True]
def test_observation_reports_this_step_scheduler_rejection():
    old_reason = "block_boundary_unsupported"
    current_reason = "waiting_present"
    engine, _sequence, _scheduler, _model_runner, step = _engine(
        _result(_lease()),
        lease=None,
        previous_fallback_counts={old_reason: 9},
        no_lease_reason=current_reason,
    )

    step(engine, completion_only=True)

    assert engine.last_step_observation[
        "exact_greedy_decode_burst_fallback_reason"
    ] == current_reason


def test_success_observation_exposes_exact_burst_identity_and_cost():
    lease = _lease()
    result = _result(lease)
    engine, _sequence, scheduler, _model_runner, step = _engine(
        result,
        lease=lease,
    )

    step(engine, completion_only=True)

    observation = engine.last_step_observation
    assert observation[
        "exact_greedy_decode_burst_attempted"
    ] is True
    assert observation[
        "exact_greedy_decode_burst_accepted"
    ] is True
    assert observation["exact_greedy_decode_burst_width"] == 3
    assert observation[
        "exact_greedy_decode_burst_lease_identity_sha256"
    ] == lease.identity_sha256
    assert observation[
        "exact_greedy_decode_burst_result_identity_sha256"
    ] == result.lease_identity_sha256
    assert observation[
        "exact_greedy_decode_burst_graph_identity_sha256"
    ] == result.graph_identity_sha256
    assert observation[
        "exact_greedy_decode_burst_replay_count"
    ] == 3
    assert observation[
        "exact_greedy_decode_burst_token_d2h_calls"
    ] == 1
    assert observation[
        "exact_greedy_decode_burst_sampled_logit_d2h_calls"
    ] == 0
    assert observation[
        "exact_greedy_decode_burst_host_visible_gap_ns"
    ] == 20
    assert observation[
        "exact_greedy_decode_burst_fallback_reason"
    ] is None
    assert observation[
        "exact_greedy_decode_burst_quarantine_reason"
    ] is None
    assert observation[
        "exact_greedy_decode_burst_pending_lease_count"
    ] == 0
    assert scheduler.host_visible_gap_ns == 20


def test_gate_only_k1_correctness_trace_is_explicitly_propagated():
    lease = _lease(width=1)
    sampled_logits = ((0, (1.0, 3.0, 2.0)),)
    result = _result(
        lease,
        tokens=(41,),
        sampled_logits=sampled_logits,
    )
    engine, sequence, scheduler, model_runner, step = _engine(
        result,
        lease=lease,
    )
    sequence.max_tokens = 1

    outputs, num_tokens = step(
        engine,
        completion_only=True,
        exact_burst_gate_width=1,
        exact_burst_correctness_trace=True,
    )

    assert outputs == [(7, [41])]
    assert num_tokens == -1
    lease_event = next(
        event for event in scheduler.events if event[0] == "lease"
    )
    assert lease_event[2]["configured_width"] == 1
    assert lease_event[2]["allow_single_token_gate"] is True
    assert ("capability", True) in model_runner.events
    assert any(
        event[0] == "burst" and event[3] is True
        for event in model_runner.events
    )
    commit_event = next(
        event
        for event in scheduler.events
        if event[0] == "prepare_commit"
    )
    assert commit_event[4]["correctness_trace"] is True
    assert commit_event[4]["gate_only_single_token"] is True
    observation = engine.last_step_observation
    assert observation[
        "exact_greedy_decode_burst_sampled_logits"
    ] == sampled_logits
    assert observation[
        "exact_greedy_decode_burst_correctness_trace"
    ] is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
