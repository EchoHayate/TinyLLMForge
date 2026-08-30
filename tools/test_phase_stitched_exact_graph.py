from dataclasses import replace
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import types

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    REPO_ROOT
    / "tinyvllm"
    / "engine"
    / "phase_stitched_exact_graph.py"
)
assert MODULE_PATH.is_file(), "phase-stitch contract module is missing"
SPEC = importlib.util.spec_from_file_location(
    "phase_stitched_exact_graph_under_test",
    MODULE_PATH,
)
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)

PhaseStitchPrefixResult = module.PhaseStitchPrefixResult
PhaseStitchSuffixResult = module.PhaseStitchSuffixResult
PhaseStitchTransaction = module.PhaseStitchTransaction
PhaseStitchMailboxBackend = module.PhaseStitchMailboxBackend
build_phase_stitch_lease = module.build_phase_stitch_lease
decide_phase_stitch_admission = module.decide_phase_stitch_admission
validate_phase_stitch_prefix = module.validate_phase_stitch_prefix
validate_phase_stitch_suffix = module.validate_phase_stitch_suffix

CONFIG_PATH = REPO_ROOT / "tinyvllm" / "config.py"


def _load_config_class():
    module_name = "phase_stitched_config_under_test"
    fake_transformers = types.ModuleType("transformers")

    class FakeAutoConfig:

        @staticmethod
        def from_pretrained(model):
            del model
            return types.SimpleNamespace(
                max_position_embeddings=4096,
                num_hidden_layers=4,
            )

    fake_transformers.AutoConfig = FakeAutoConfig
    original = sys.modules.get("transformers")
    sys.modules["transformers"] = fake_transformers
    try:
        config_module = types.ModuleType(module_name)
        config_module.__file__ = os.fspath(CONFIG_PATH)
        sys.modules[module_name] = config_module
        source = CONFIG_PATH.read_text(encoding="utf-8")
        exec(
            compile(
                "from __future__ import annotations\n" + source,
                os.fspath(CONFIG_PATH),
                "exec",
            ),
            config_module.__dict__,
        )
        return config_module.Config
    finally:
        if original is None:
            sys.modules.pop("transformers", None)
        else:
            sys.modules["transformers"] = original


def _lease(**overrides):
    values = {
        "sequence_id": 7,
        "sequence_generation": 3,
        "schedule_generation": 11,
        "prefill_graph_identity_sha256": "a" * 64,
        "prefill_graph_generation": 5,
        "decode_graph_identity_sha256": "b" * 64,
        "decode_graph_generation": 13,
        "prompt_token_count": 256,
        "final_prefill_first_position": 0,
        "final_prefill_last_position": 255,
        "initial_completion_count": 0,
        "remaining_output_tokens": 8,
        "decode_first_write_position": 256,
        "decode_last_write_position": 262,
        "decode_first_physical_slot": 1024,
        "decode_last_physical_slot": 1030,
        "block_table_identity": ((64, 9),),
        "completion_only": True,
        "source_identity_sha256": "c" * 64,
    }
    values.update(overrides)
    return build_phase_stitch_lease(**values)


def _admission(**overrides):
    values = {
        "enabled": True,
        "prefill_graph_available": True,
        "decode_graph_available": True,
        "prompt_token_count": 256,
        "prompt_token_allowlist": (256, 2048),
        "sequence_count": 1,
        "waiting_count": 0,
        "prefilling_count": 0,
        "do_sample": True,
        "temperatures": (0.0,),
        "ignore_eos": (True,),
        "completion_only": True,
        "remaining_output_tokens": 8,
        "decode_kv_capacity_tokens": 7,
        "tensor_parallel_size": 1,
        "rank": 0,
        "incompatible_modes": (),
        "pending_lease": False,
        "quarantined": False,
    }
    values.update(overrides)
    return decide_phase_stitch_admission(**values)


def test_phase_stitch_lease_binds_exact_k8_parent_transaction():
    lease = _lease()

    assert lease.parent_token_count == 8
    assert lease.authorized_decode_replay_count == 7
    assert lease.first_token_ordinal == 0
    assert lease.suffix_start_ordinal == 1
    assert len(lease.identity_sha256) == 64


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("sequence_generation", 4),
        ("schedule_generation", 12),
        ("prefill_graph_identity_sha256", "d" * 64),
        ("prefill_graph_generation", 6),
        ("decode_graph_identity_sha256", "e" * 64),
        ("decode_graph_generation", 14),
        ("block_table_identity", ((64, 10),)),
        ("remaining_output_tokens", 9),
        ("source_identity_sha256", "f" * 64),
    ),
)
def test_phase_stitch_lease_identity_binds_authoritative_fields(
    field,
    value,
):
    baseline = _lease()
    changed = _lease(**{field: value})

    assert changed.identity_sha256 != baseline.identity_sha256


def test_phase_stitch_lease_identity_binds_decode_physical_interval():
    baseline = _lease()
    changed = _lease(
        decode_first_physical_slot=2048,
        decode_last_physical_slot=2054,
    )

    assert changed.identity_sha256 != baseline.identity_sha256


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("sequence_id", True, "sequence_id"),
        (
            "prefill_graph_identity_sha256",
            "not-a-digest",
            "prefill_graph_identity_sha256",
        ),
        (
            "final_prefill_last_position",
            254,
            "prefill position interval",
        ),
        (
            "decode_last_write_position",
            263,
            "decode write interval",
        ),
        (
            "decode_last_physical_slot",
            1031,
            "decode physical interval",
        ),
        (
            "remaining_output_tokens",
            7,
            "remaining_output_tokens",
        ),
        ("completion_only", False, "completion-only"),
    ),
)
def test_phase_stitch_lease_rejects_invalid_authority(
    field,
    value,
    message,
):
    with pytest.raises(ValueError, match=message):
        _lease(**{field: value})


def test_phase_stitch_prefix_and_suffix_validate_parent_identity():
    lease = _lease()
    prefix = PhaseStitchPrefixResult(
        parent_lease_identity_sha256=lease.identity_sha256,
        token=101,
        token_ordinal=0,
        replay_count=0,
        d2h_calls=1,
        d2h_bytes=8,
    )
    suffix = PhaseStitchSuffixResult(
        parent_lease_identity_sha256=lease.identity_sha256,
        tokens=(102, 103, 104, 105, 106, 107, 108),
        first_token_ordinal=1,
        replay_count=7,
        d2h_calls=1,
        d2h_bytes=56,
    )

    assert validate_phase_stitch_prefix(lease, prefix) is prefix
    assert validate_phase_stitch_suffix(lease, suffix) is suffix

    with pytest.raises(ValueError, match="parent lease identity"):
        validate_phase_stitch_prefix(
            lease,
            replace(
                prefix,
                parent_lease_identity_sha256="d" * 64,
            ),
        )
    with pytest.raises(ValueError, match="parent lease identity"):
        validate_phase_stitch_suffix(
            lease,
            replace(
                suffix,
                parent_lease_identity_sha256="d" * 64,
            ),
        )


def test_phase_stitch_transaction_accepts_only_ordered_two_phase_commit():
    transaction = PhaseStitchTransaction(_lease())

    transaction.mark_replay_started()
    transaction.mark_prefix_ready()
    transaction.mark_prefix_committed()
    transaction.mark_suffix_ready(replay_count=7)
    transaction.mark_suffix_committed()
    transaction.close()

    assert transaction.state == "closed"
    assert transaction.last_authoritative_phase == "suffix_committed"
    assert transaction.completed_decode_replays == 7


def test_phase_stitch_transaction_rejects_duplicate_prefix_commit():
    transaction = PhaseStitchTransaction(_lease())
    transaction.mark_replay_started()
    transaction.mark_prefix_ready()
    transaction.mark_prefix_committed()

    with pytest.raises(ValueError, match="prefix commit"):
        transaction.mark_prefix_committed()


def test_phase_stitch_transaction_distinguishes_failure_visibility():
    before_prefix = PhaseStitchTransaction(_lease())
    before_prefix.mark_replay_started()
    before_prefix.fail("replay_failed")
    assert before_prefix.state == "failed_before_prefix"
    assert before_prefix.partial_visibility is False
    assert before_prefix.failure_reason == "replay_failed"

    after_prefix = PhaseStitchTransaction(_lease())
    after_prefix.mark_replay_started()
    after_prefix.mark_prefix_ready()
    after_prefix.mark_prefix_committed()
    after_prefix.fail("suffix_copy_failed")
    assert after_prefix.state == "failed_after_prefix"
    assert after_prefix.partial_visibility is True
    assert after_prefix.failure_reason == "suffix_copy_failed"


def test_phase_stitch_transaction_cancels_only_before_replay():
    transaction = PhaseStitchTransaction(_lease())
    transaction.cancel("unsupported_request")
    assert transaction.state == "cancelled"

    started = PhaseStitchTransaction(_lease())
    started.mark_replay_started()
    with pytest.raises(ValueError, match="cannot cancel"):
        started.cancel("too_late")


def test_phase_stitch_config_defaults_off_and_rejects_non_bool():
    Config = _load_config_class()
    with tempfile.TemporaryDirectory() as model:
        assert Config(
            model=model
        ).phase_stitched_exact_graph_runtime is False
        with pytest.raises(
            ValueError,
            match="phase_stitched_exact_graph_runtime must be a bool",
        ):
            Config(
                model=model,
                phase_stitched_exact_graph_runtime=1,
            )


def test_phase_stitch_admission_accepts_exact_supported_request():
    decision = _admission()

    assert decision.optimized is True
    assert decision.fallback_reason is None


@pytest.mark.parametrize(
    ("overrides", "reason"),
    (
        ({"enabled": False}, "disabled"),
        (
            {"prefill_graph_available": False},
            "prefill_graph_unavailable",
        ),
        (
            {"decode_graph_available": False},
            "decode_graph_unavailable",
        ),
        (
            {"prompt_token_count": 128},
            "prompt_shape_not_allowlisted",
        ),
        ({"sequence_count": 2}, "sequence_count_unsupported"),
        ({"waiting_count": 1}, "waiting_request_present"),
        (
            {"prefilling_count": 1},
            "prefilling_request_present",
        ),
        ({"do_sample": False}, "sampling_unsupported"),
        ({"temperatures": (0.1,)}, "temperature_nonzero"),
        ({"ignore_eos": (False,)}, "ignore_eos_required"),
        ({"completion_only": False}, "completion_only_required"),
        (
            {"remaining_output_tokens": 7},
            "output_budget_insufficient",
        ),
        (
            {"decode_kv_capacity_tokens": 6},
            "decode_kv_capacity_insufficient",
        ),
        (
            {"tensor_parallel_size": 2},
            "tensor_parallel_unsupported",
        ),
        ({"rank": 1}, "non_root_rank"),
        (
            {"incompatible_modes": ("kv_offload",)},
            "incompatible_mode:kv_offload",
        ),
        ({"pending_lease": True}, "lease_pending"),
        ({"quarantined": True}, "identity_quarantined"),
    ),
)
def test_phase_stitch_admission_matrix(overrides, reason):
    decision = _admission(**overrides)

    assert decision.optimized is False
    assert decision.fallback_reason == reason


class _Mailbox:
    def __init__(self, label, events):
        self.label = label
        self.events = events
        self.values = ()
        self.tolist_calls = 0

    def copy_(self, source, *, non_blocking):
        self.events.append(
            (
                "copy",
                self.label,
                tuple(source.values),
                non_blocking,
            )
        )
        self.values = tuple(source.values)
        return self

    def tolist(self):
        self.tolist_calls += 1
        self.events.append(("tolist", self.label))
        return list(self.values)


class _TokenSlice:
    def __init__(self, values):
        self.values = tuple(values)


class _MailboxEvent:
    def __init__(self, label, events):
        self.label = label
        self.events = events

    def record(self, stream):
        self.events.append(
            ("record", self.label, stream.label)
        )

    def synchronize(self):
        self.events.append(("synchronize", self.label))


class _MailboxStream:
    def __init__(self, label, events):
        self.label = label
        self.events = events

    def wait_event(self, event):
        self.events.append(
            ("wait_event", self.label, event.label)
        )


class _MailboxStreamContext:
    def __init__(self, stream, events):
        self.stream = stream
        self.events = events

    def __enter__(self):
        self.events.append(("enter_stream", self.stream.label))

    def __exit__(self, *_args):
        self.events.append(("exit_stream", self.stream.label))


def _mailbox_backend(events):
    compute_stream = _MailboxStream("compute", events)
    copy_stream = _MailboxStream("copy", events)
    first_token_mailbox = _Mailbox("first", events)
    suffix_mailbox = _Mailbox("suffix", events)
    backend = PhaseStitchMailboxBackend(
        copy_stream=copy_stream,
        first_token_mailbox=first_token_mailbox,
        suffix_mailbox=suffix_mailbox,
        event_factory=lambda label: _MailboxEvent(label, events),
        current_stream=lambda: compute_stream,
        stream_context=lambda stream: _MailboxStreamContext(
            stream,
            events,
        ),
        synchronize=lambda: events.append(("synchronize_all",)),
    )
    return backend, first_token_mailbox, suffix_mailbox


def test_phase_stitch_mailboxes_copy_one_then_seven_independently():
    events = []
    backend, first_mailbox, suffix_mailbox = _mailbox_backend(
        events
    )
    parent_identity = "a" * 64
    generation = backend.begin_transaction(parent_identity)

    prefix = backend.enqueue_first_token(
        parent_lease_identity_sha256=parent_identity,
        token_slice=_TokenSlice((101,)),
        mailbox_generation=generation,
    )
    suffix = backend.enqueue_suffix(
        parent_lease_identity_sha256=parent_identity,
        token_slice=_TokenSlice(
            (102, 103, 104, 105, 106, 107, 108)
        ),
        mailbox_generation=generation,
    )

    assert prefix.d2h_calls == 1
    assert prefix.d2h_bytes == 8
    assert suffix.d2h_calls == 1
    assert suffix.d2h_bytes == 56
    assert prefix.wait_token() == 101
    assert (
        "synchronize",
        "suffix_copy_done",
    ) not in events
    assert suffix.wait_tokens() == (
        102,
        103,
        104,
        105,
        106,
        107,
        108,
    )
    assert prefix.wait_token() == 101
    assert suffix.wait_tokens() == (
        102,
        103,
        104,
        105,
        106,
        107,
        108,
    )
    assert first_mailbox.tolist_calls == 1
    assert suffix_mailbox.tolist_calls == 1
    assert events[:10] == [
        ("record", "first_token_compute_done", "compute"),
        ("wait_event", "copy", "first_token_compute_done"),
        ("enter_stream", "copy"),
        ("copy", "first", (101,), True),
        ("record", "first_token_copy_done", "copy"),
        ("exit_stream", "copy"),
        ("record", "suffix_compute_done", "compute"),
        ("wait_event", "copy", "suffix_compute_done"),
        ("enter_stream", "copy"),
        (
            "copy",
            "suffix",
            (102, 103, 104, 105, 106, 107, 108),
            True,
        ),
    ]
    backend.release_transaction(generation)
    assert backend.active_generation is None


def test_phase_stitch_mailbox_rejects_duplicate_and_stale_enqueue():
    events = []
    backend, _, _ = _mailbox_backend(events)
    parent_identity = "a" * 64
    generation = backend.begin_transaction(parent_identity)
    backend.enqueue_first_token(
        parent_lease_identity_sha256=parent_identity,
        token_slice=_TokenSlice((101,)),
        mailbox_generation=generation,
    )

    with pytest.raises(ValueError, match="already enqueued"):
        backend.enqueue_first_token(
            parent_lease_identity_sha256=parent_identity,
            token_slice=_TokenSlice((102,)),
            mailbox_generation=generation,
        )
    with pytest.raises(ValueError, match="mailbox generation"):
        backend.enqueue_suffix(
            parent_lease_identity_sha256=parent_identity,
            token_slice=_TokenSlice(tuple(range(7))),
            mailbox_generation=generation + 1,
        )
    with pytest.raises(ValueError, match="parent lease identity"):
        backend.enqueue_suffix(
            parent_lease_identity_sha256="b" * 64,
            token_slice=_TokenSlice(tuple(range(7))),
            mailbox_generation=generation,
        )


def test_phase_stitch_mailbox_abort_synchronizes_before_release():
    events = []
    backend, _, _ = _mailbox_backend(events)
    generation = backend.begin_transaction("a" * 64)
    backend.enqueue_first_token(
        parent_lease_identity_sha256="a" * 64,
        token_slice=_TokenSlice((101,)),
        mailbox_generation=generation,
    )

    backend.abort_transaction(generation)

    assert events[-1] == ("synchronize_all",)
    assert backend.active_generation is None
    with pytest.raises(ValueError, match="mailbox generation"):
        backend.release_transaction(generation)


def _scheduler_phase_stitch_fixture():
    canonical_name = "tinyvllm.engine.phase_stitched_exact_graph"
    if canonical_name not in sys.modules:
        canonical_spec = importlib.util.spec_from_file_location(
            canonical_name,
            MODULE_PATH,
        )
        canonical = importlib.util.module_from_spec(canonical_spec)
        sys.modules[canonical_name] = canonical
        canonical_spec.loader.exec_module(canonical)
    from tools import test_scheduler_prepared_postprocess as fixture

    canonical = sys.modules[canonical_name]
    return fixture, canonical


def _prepare_scheduler_phase_stitch(scheduler, sequence):
    return scheduler.prepare_phase_stitch(
        (sequence,),
        schedule_generation=1,
        enabled=True,
        prefill_graph_available=True,
        decode_graph_available=True,
        prefill_graph_identity_sha256="a" * 64,
        prefill_graph_generation=5,
        decode_graph_identity_sha256="b" * 64,
        decode_graph_generation=13,
        prompt_token_allowlist=(16,),
        is_prefill=True,
        do_sample=True,
        batch_kind=None,
        completion_only=True,
        tensor_parallel_size=1,
        rank=0,
        incompatible_modes=(),
        source_identity_sha256="c" * 64,
    )


def test_scheduler_phase_stitch_preauthorizes_one_parent_lease():
    fixture, _ = _scheduler_phase_stitch_fixture()
    fixture.Sequence.block_size = 16
    scheduler = fixture.Scheduler(
        fixture._config(
            kvcache_block_size=16,
            phase_stitched_exact_graph_runtime=True,
        )
    )
    sequence = fixture._scheduled_prefill_sequence(
        scheduler,
        tuple(range(16)),
        chunk_end=16,
        final=True,
        do_sample=True,
        max_tokens=8,
    )
    sequence.ignore_eos = True
    scheduler.schedule_generation = 1
    original_completion = tuple(sequence.completion_token_ids)
    original_block_count = len(sequence.block_table)

    lease = _prepare_scheduler_phase_stitch(
        scheduler,
        sequence,
    )

    assert lease is not None
    assert lease.sequence_generation == 0
    assert lease.decode_first_write_position == 16
    assert lease.decode_last_write_position == 22
    assert (
        lease.decode_last_physical_slot
        - lease.decode_first_physical_slot
        + 1
        == 7
    )
    assert len(sequence.block_table) == original_block_count + 1
    assert lease.block_table_identity == (
        scheduler.block_manager.block_identities(
            tuple(sequence.block_table)
        )
    )
    assert tuple(sequence.completion_token_ids) == original_completion
    assert scheduler._phase_stitch_pending_lease == lease

    assert _prepare_scheduler_phase_stitch(
        scheduler,
        sequence,
    ) is None
    summary = scheduler.phase_stitch_summary()
    assert summary["attempts"] == 2
    assert summary["acceptances"] == 1
    assert summary["reserved_decode_write_positions"] == 7
    assert summary["fallback_counts"] == {"lease_pending": 1}


def test_scheduler_phase_stitch_commits_prefix_then_suffix():
    fixture, canonical = _scheduler_phase_stitch_fixture()
    fixture.Sequence.block_size = 16
    scheduler = fixture.Scheduler(
        fixture._config(
            kvcache_block_size=16,
            phase_stitched_exact_graph_runtime=True,
        )
    )
    sequence = fixture._scheduled_prefill_sequence(
        scheduler,
        tuple(range(16)),
        chunk_end=16,
        final=True,
        do_sample=True,
        max_tokens=8,
    )
    sequence.ignore_eos = True
    scheduler.schedule_generation = 1
    lease = _prepare_scheduler_phase_stitch(
        scheduler,
        sequence,
    )
    scheduler.mark_phase_stitch_replay_started(lease)

    prefix = canonical.PhaseStitchPrefixResult(
        parent_lease_identity_sha256=lease.identity_sha256,
        token=101,
        token_ordinal=0,
        replay_count=0,
        d2h_calls=1,
        d2h_bytes=8,
    )
    prepared_prefix = scheduler.prepare_phase_stitch_prefix_commit(
        (sequence,),
        lease,
        prefix,
    )
    scheduler.commit_prepared_postprocess(prepared_prefix)

    assert sequence.completion_token_ids == [101]
    assert sequence.num_computed_tokens == 16
    assert scheduler._phase_stitch_pending_lease == lease
    assert scheduler._phase_stitch_transaction.state == (
        "prefix_committed"
    )

    suffix = canonical.PhaseStitchSuffixResult(
        parent_lease_identity_sha256=lease.identity_sha256,
        tokens=(102, 103, 104, 105, 106, 107, 108),
        first_token_ordinal=1,
        replay_count=7,
        d2h_calls=1,
        d2h_bytes=56,
    )
    prepared_suffix = scheduler.prepare_phase_stitch_suffix_commit(
        (sequence,),
        lease,
        suffix,
    )
    scheduler.commit_prepared_postprocess(prepared_suffix)

    assert sequence.completion_token_ids == [
        101,
        102,
        103,
        104,
        105,
        106,
        107,
        108,
    ]
    assert scheduler._phase_stitch_pending_lease is None
    assert scheduler._phase_stitch_transaction is None
    summary = scheduler.phase_stitch_summary()
    assert summary["prefix_commits"] == 1
    assert summary["suffix_commits"] == 1
    assert summary["closed_transactions"] == 1
    assert summary["last_authoritative_phase"] == "suffix_committed"
    assert summary["last_completed_decode_replays"] == 7
    assert summary["sequence_generations"] == {
        str(sequence.seq_id): 1,
    }


def test_scheduler_phase_stitch_failure_quarantines_parent_identity():
    fixture, canonical = _scheduler_phase_stitch_fixture()
    fixture.Sequence.block_size = 16
    scheduler = fixture.Scheduler(
        fixture._config(
            kvcache_block_size=16,
            phase_stitched_exact_graph_runtime=True,
        )
    )
    sequence = fixture._scheduled_prefill_sequence(
        scheduler,
        tuple(range(16)),
        chunk_end=16,
        final=True,
        do_sample=True,
        max_tokens=8,
    )
    sequence.ignore_eos = True
    scheduler.schedule_generation = 1
    lease = _prepare_scheduler_phase_stitch(
        scheduler,
        sequence,
    )
    scheduler.mark_phase_stitch_replay_started(lease)
    bad_prefix = canonical.PhaseStitchPrefixResult(
        parent_lease_identity_sha256=lease.identity_sha256,
        token=101,
        token_ordinal=1,
        replay_count=0,
        d2h_calls=1,
        d2h_bytes=8,
    )

    with pytest.raises(ValueError, match="prefix token ordinal"):
        scheduler.prepare_phase_stitch_prefix_commit(
            (sequence,),
            lease,
            bad_prefix,
        )

    assert sequence.num_completion_tokens == 0
    assert scheduler._phase_stitch_pending_lease is None
    summary = scheduler.phase_stitch_summary()
    assert summary["failures_before_prefix"] == 1
    assert summary["quarantined_parent_identities"] == [
        lease.identity_sha256,
    ]
    assert summary["last_authoritative_phase"] == (
        "failed_before_prefix"
    )


def test_scheduler_phase_stitch_cancel_releases_only_reserved_blocks():
    fixture, _ = _scheduler_phase_stitch_fixture()
    fixture.Sequence.block_size = 16
    scheduler = fixture.Scheduler(
        fixture._config(
            kvcache_block_size=16,
            phase_stitched_exact_graph_runtime=True,
        )
    )
    sequence = fixture._scheduled_prefill_sequence(
        scheduler,
        tuple(range(16)),
        chunk_end=16,
        final=True,
        do_sample=True,
        max_tokens=8,
    )
    sequence.ignore_eos = True
    scheduler.schedule_generation = 1
    original_block_table = tuple(sequence.block_table)
    original_free_count = len(scheduler.block_manager.free_block_ids)
    lease = _prepare_scheduler_phase_stitch(
        scheduler,
        sequence,
    )

    scheduler.cancel_phase_stitch(lease, "engine_fallback")

    assert tuple(sequence.block_table) == original_block_table
    assert len(scheduler.block_manager.free_block_ids) == (
        original_free_count
    )
    assert scheduler._phase_stitch_pending_lease is None
    summary = scheduler.phase_stitch_summary()
    assert summary["cancellations"] == 1
    assert summary["last_authoritative_phase"] == "cancelled"
    assert summary["sequence_generations"] == {
        str(sequence.seq_id): 1,
    }


def test_scheduler_phase_stitch_failure_after_prefix_keeps_visibility():
    fixture, canonical = _scheduler_phase_stitch_fixture()
    fixture.Sequence.block_size = 16
    scheduler = fixture.Scheduler(
        fixture._config(
            kvcache_block_size=16,
            phase_stitched_exact_graph_runtime=True,
        )
    )
    sequence = fixture._scheduled_prefill_sequence(
        scheduler,
        tuple(range(16)),
        chunk_end=16,
        final=True,
        do_sample=True,
        max_tokens=8,
    )
    sequence.ignore_eos = True
    scheduler.schedule_generation = 1
    lease = _prepare_scheduler_phase_stitch(
        scheduler,
        sequence,
    )
    scheduler.mark_phase_stitch_replay_started(lease)
    prefix = canonical.PhaseStitchPrefixResult(
        parent_lease_identity_sha256=lease.identity_sha256,
        token=101,
        token_ordinal=0,
        replay_count=0,
        d2h_calls=1,
        d2h_bytes=8,
    )
    prepared = scheduler.prepare_phase_stitch_prefix_commit(
        (sequence,),
        lease,
        prefix,
    )
    scheduler.commit_prepared_postprocess(prepared)
    block_table_after_prefix = tuple(sequence.block_table)

    scheduler.fail_phase_stitch(lease, "suffix_copy_failed")

    assert sequence.completion_token_ids == [101]
    assert tuple(sequence.block_table) == block_table_after_prefix
    assert scheduler._phase_stitch_pending_lease is None
    summary = scheduler.phase_stitch_summary()
    assert summary["failures_after_prefix"] == 1
    assert summary["last_authoritative_phase"] == (
        "failed_after_prefix"
    )
    assert summary["last_failure_reason"] == "suffix_copy_failed"


class _RunnerTensor:
    _next_pointer = 1000

    def __init__(
        self,
        name,
        values,
        events,
        *,
        shape=None,
        dtype="torch.int64",
        device="cuda:0",
    ):
        self.name = name
        self.values = list(values)
        self.events = events
        self.shape = (
            tuple(shape)
            if shape is not None
            else (len(self.values),)
        )
        self.dtype = dtype
        self.device = device
        self._pointer = _RunnerTensor._next_pointer
        _RunnerTensor._next_pointer += 1

    def data_ptr(self):
        return self._pointer

    def stride(self):
        if len(self.shape) == 1:
            return (1,)
        if len(self.shape) == 2:
            return (self.shape[1], 1)
        return tuple(1 for _ in self.shape)

    def storage_offset(self):
        return 0

    def size(self):
        return self.shape

    def numel(self):
        result = 1
        for dimension in self.shape:
            result *= dimension
        return result

    def element_size(self):
        return 8 if "int64" in str(self.dtype) else 4

    def fill_(self, value):
        self.events.append(("fill", self.name, value))
        self.values = [value for _ in self.values]
        return self

    def zero_(self):
        return self.fill_(0)

    def copy_(self, source):
        source_values = list(source.values)
        self.events.append(
            ("copy", self.name, source.name)
        )
        self.values = source_values
        return self

    def __getitem__(self, index):
        if isinstance(index, slice):
            return _RunnerTensorView(self, index)
        return self.values[index]


class _RunnerTensorView:
    def __init__(self, parent, index):
        self.parent = parent
        self.index = index
        self.name = f"{parent.name}[{index.start}:{index.stop}]"

    @property
    def values(self):
        return self.parent.values[self.index]

    def copy_(self, source):
        self.parent.events.append(
            ("copy", self.name, source.name)
        )
        self.parent.values[self.index] = list(source.values)
        return self


class _RunnerHidden:
    def __init__(self, events):
        self.events = events

    def __getitem__(self, index):
        self.events.append(("select_hidden", index))
        return self


class _RunnerLogits:
    def __init__(self, events, token):
        self.events = events
        self.token = token

    def to(self, dtype):
        self.events.append(("float32_logits", dtype))
        return self

    def argmax(self, *, dim):
        self.events.append(("argmax", dim))
        return self.token


class _RunnerModel:
    def __init__(self, events, token):
        self.events = events
        self.token = token

    def compute_logits(self, hidden):
        del hidden
        self.events.append(("lm_head",))
        return _RunnerLogits(self.events, self.token)


class _RunnerReplay:
    def __init__(self, tensors, events, fail_at=None):
        self.tensors = tensors
        self.events = events
        self.fail_at = fail_at
        self.count = 0

    def replay(self):
        self.count += 1
        self.events.append(("decode_replay", self.count))
        if self.count == self.fail_at:
            raise RuntimeError("decode exploded")
        history_index = self.tensors["history_index"].values[0]
        token = 101 + self.count
        self.tensors["token_history"].values[history_index] = token
        self.tensors["input_token"].values[0] = token
        self.tensors["history_index"].values[0] += 1


class _RunnerDecodeGraph:
    def __init__(self, events, *, fail_at=None):
        self.events = events
        self.block_size = 16
        self.tensors = {
            "input_token": _RunnerTensor(
                "decode_input",
                [-1],
                events,
            ),
            "position": _RunnerTensor(
                "decode_position",
                [-1],
                events,
            ),
            "context_length": _RunnerTensor(
                "decode_context",
                [-1],
                events,
                dtype="torch.int32",
            ),
            "slot_mapping": _RunnerTensor(
                "decode_slot",
                [-1],
                events,
                dtype="torch.int32",
            ),
            "block_table": _RunnerTensor(
                "decode_block_table",
                [-1, -1, -1, -1],
                events,
                shape=(1, 4),
                dtype="torch.int32",
            ),
            "token_history": _RunnerTensor(
                "history",
                [-1] * 8,
                events,
            ),
            "history_index": _RunnerTensor(
                "history_index",
                [0],
                events,
            ),
        }
        self.tensor_identities = {
            name: {
                "data_ptr": tensor.data_ptr(),
                "shape": list(tensor.shape),
                "stride": list(tensor.stride()),
                "storage_offset": tensor.storage_offset(),
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
            }
            for name, tensor in sorted(self.tensors.items())
        }
        self.graph = _RunnerReplay(
            self.tensors,
            events,
            fail_at=fail_at,
        )

    def capability(self):
        return {
            "available": True,
            "graph_identity_sha256": "b" * 64,
            "graph_generation": 13,
            "rank": 0,
            "tensor_parallel_size": 1,
            "block_size": 16,
            "block_table_width": 4,
            "history_capacity": 8,
            "correctness_trace": False,
            "quarantine_reason": None,
        }

    @classmethod
    def _reset_static_state(
        cls,
        tensors,
        *,
        scratch_block_id,
        block_size,
    ):
        del cls, scratch_block_id, block_size
        tensors["input_token"].fill_(-1)
        tensors["position"].fill_(-1)
        tensors["context_length"].fill_(-1)
        tensors["slot_mapping"].fill_(-1)
        tensors["block_table"].fill_(-1)
        tensors["token_history"].fill_(-1)
        tensors["history_index"].zero_()


class _RunnerCompletion:
    def __init__(self, label, events):
        self.label = label
        self.events = events

    def synchronize(self):
        self.events.append(("synchronize", self.label))


class _RunnerMailboxValues:
    def __init__(self, label, values, events):
        self.label = label
        self.values = tuple(values)
        self.events = events

    def tolist(self):
        self.events.append(("tolist", self.label))
        return list(self.values)


class _RunnerMailboxBackend:
    def __init__(self, canonical, events):
        self.canonical = canonical
        self.events = events
        self.active_generation = None

    def begin_transaction(self, parent_identity):
        self.events.append(("begin_mailbox", parent_identity))
        self.active_generation = 1
        return 1

    def enqueue_first_token(
        self,
        *,
        parent_lease_identity_sha256,
        token_slice,
        mailbox_generation,
    ):
        assert mailbox_generation == self.active_generation
        self.events.append(("enqueue_first_token",))
        return self.canonical.PhaseStitchPrefixResult(
            parent_lease_identity_sha256=(
                parent_lease_identity_sha256
            ),
            token=None,
            token_ordinal=0,
            replay_count=0,
            d2h_calls=1,
            d2h_bytes=8,
            completion=_RunnerCompletion("prefix", self.events),
            mailbox=_RunnerMailboxValues(
                "prefix",
                token_slice.values,
                self.events,
            ),
        )

    def enqueue_suffix(
        self,
        *,
        parent_lease_identity_sha256,
        token_slice,
        mailbox_generation,
    ):
        assert mailbox_generation == self.active_generation
        self.events.append(("enqueue_suffix",))
        return self.canonical.PhaseStitchSuffixResult(
            parent_lease_identity_sha256=(
                parent_lease_identity_sha256
            ),
            tokens=None,
            first_token_ordinal=1,
            replay_count=7,
            d2h_calls=1,
            d2h_bytes=56,
            completion=_RunnerCompletion("suffix", self.events),
            mailbox=_RunnerMailboxValues(
                "suffix",
                token_slice.values,
                self.events,
            ),
        )

    def abort_transaction(self, generation):
        self.events.append(("abort_mailbox", generation))
        self.active_generation = None


def _model_runner_phase_stitch_fixture():
    canonical_name = "tinyvllm.engine.phase_stitched_exact_graph"
    if canonical_name not in sys.modules:
        canonical_spec = importlib.util.spec_from_file_location(
            canonical_name,
            MODULE_PATH,
        )
        canonical = importlib.util.module_from_spec(canonical_spec)
        sys.modules[canonical_name] = canonical
        canonical_spec.loader.exec_module(canonical)
    from tools import test_model_runner_spec_verify as fixture

    canonical = sys.modules[canonical_name]
    return fixture, canonical


def _model_runner_phase_stitch_case(*, fail_at=None):
    fixture, canonical = _model_runner_phase_stitch_fixture()
    events = []
    runner = object.__new__(fixture.ModelRunner)
    runner.config = types.SimpleNamespace(
        phase_stitched_exact_graph_runtime=True,
        tensor_parallel_size=1,
        cpu_offload=False,
        kv_offload_mvp0=False,
        kv_quant_bits=0,
        quest_top_k_blocks=-1,
        am_compact_blocks=0,
    )
    runner.enforce_eager = False
    runner.rank = 0
    runner.world_size = 1
    runner.block_size = 16
    runner._ordinary_graph_generation = 13
    runner._prefill_cuda_graph_step_id = 0
    prefill_entry = types.SimpleNamespace(
        identity=types.SimpleNamespace(token_count=16),
        identity_sha256="a" * 64,
        state="ready",
        tensors={"outputs": _RunnerHidden(events)},
    )
    runner.exact_prefill_cuda_graph_cache = types.SimpleNamespace(
        ready_entries={"a" * 64: prefill_entry},
    )
    runner.exact_greedy_decode_burst_graph = _RunnerDecodeGraph(
        events,
        fail_at=fail_at,
    )
    runner.phase_stitch_mailbox_backend = _RunnerMailboxBackend(
        canonical,
        events,
    )
    first_token = _RunnerTensor(
        "token0",
        [101],
        events,
    )
    runner.model = _RunnerModel(events, first_token)
    runner._exact_prefill_replay_tensor_shapes_match = (
        lambda **_kwargs: (
            events.append(("validate_prefill_tensors",))
            or True
        )
    )

    def replay_prefill(entry, **_kwargs):
        assert entry is prefill_entry
        events.append(("bind_prefill_live_tensors",))
        events.append(("prefill_replay",))
        return entry.tensors["outputs"]

    runner._replay_exact_prefill_graph = replay_prefill

    def prepare_block_tables(rows, name):
        events.append(("materialize_block_table", name))
        return _RunnerTensor(
            "live_block_table",
            list(rows[0]),
            events,
            shape=(1, len(rows[0])),
            dtype="torch.int32",
        )

    runner.prepare_block_tables_from_rows = prepare_block_tables
    sequence = types.SimpleNamespace(
        seq_id=7,
        block_table=[5, 6],
        num_prompt_tokens=16,
        num_completion_tokens=0,
    )
    lease = canonical.build_phase_stitch_lease(
        sequence_id=7,
        sequence_generation=0,
        schedule_generation=1,
        prefill_graph_identity_sha256="a" * 64,
        prefill_graph_generation=13,
        decode_graph_identity_sha256="b" * 64,
        decode_graph_generation=13,
        prompt_token_count=16,
        final_prefill_first_position=0,
        final_prefill_last_position=15,
        initial_completion_count=0,
        remaining_output_tokens=8,
        decode_first_write_position=16,
        decode_last_write_position=22,
        decode_first_physical_slot=6 * 16,
        decode_last_physical_slot=6 * 16 + 6,
        block_table_identity=((5, 2), (6, 4)),
        completion_only=True,
        source_identity_sha256=canonical.build_phase_stitch_source_identity(
            prefill_graph_identity_sha256="a" * 64,
            prefill_graph_generation=13,
            decode_graph_identity_sha256="b" * 64,
            decode_graph_generation=13,
        ),
    )
    input_ids = _RunnerTensor(
        "live_input_ids",
        list(range(16)),
        events,
    )
    positions = _RunnerTensor(
        "live_positions",
        list(range(16)),
        events,
    )
    context = types.SimpleNamespace(
        slot_mapping=_RunnerTensor(
            "live_slots",
            list(range(16)),
            events,
            dtype="torch.int32",
        ),
        cu_seqlens_q=_RunnerTensor(
            "live_cu_q",
            [0, 16],
            events,
            dtype="torch.int32",
        ),
        cu_seqlens_k=_RunnerTensor(
            "live_cu_k",
            [0, 16],
            events,
            dtype="torch.int32",
        ),
    )
    return (
        runner,
        canonical,
        events,
        sequence,
        lease,
        input_ids,
        positions,
        context,
    )


def test_model_runner_phase_stitch_composes_one_plus_seven_replays():
    (
        runner,
        _,
        events,
        sequence,
        lease,
        input_ids,
        positions,
        context,
    ) = _model_runner_phase_stitch_case()

    result = runner.run_phase_stitched_exact_graph(
        (sequence,),
        lease,
        input_ids=input_ids,
        positions=positions,
        context=context,
        current_block_table_identity=((5, 2), (6, 4)),
    )

    labels = [event[0] for event in events]
    assert labels.index("validate_prefill_tensors") < labels.index(
        "prefill_replay"
    )
    assert labels.index("bind_prefill_live_tensors") < labels.index(
        "prefill_replay"
    )
    assert labels.index("prefill_replay") < labels.index("lm_head")
    assert labels.index("lm_head") < labels.index("float32_logits")
    assert labels.index("float32_logits") < labels.index("argmax")
    history_copy = events.index(
        ("copy", "history[0:1]", "token0")
    )
    first_enqueue = events.index(("enqueue_first_token",))
    input_seed = events.index(
        ("copy", "decode_input", "token0")
    )
    assert history_copy < first_enqueue < input_seed
    replay_events = [
        event for event in events if event[0] == "decode_replay"
    ]
    assert replay_events == [
        ("decode_replay", ordinal) for ordinal in range(1, 8)
    ]
    assert events.index(("enqueue_suffix",)) > events.index(
        ("decode_replay", 7)
    )
    assert not any(
        event[0] in ("synchronize", "tolist")
        for event in events
    )
    assert result.prefill_forward_count == 1
    assert result.decode_replay_count == 7
    assert result.target_model_forward_count == 8
    assert result.prefix.wait_token() == 101
    assert result.suffix.wait_tokens() == (
        102,
        103,
        104,
        105,
        106,
        107,
        108,
    )
    summary = runner.phase_stitch_summary()
    assert summary["prefill_graph_replays"] == 1
    assert summary["decode_graph_replays"] == 7
    assert summary["target_model_forwards"] == 8


def test_model_runner_phase_stitch_rejects_identity_before_replay():
    (
        runner,
        _,
        events,
        sequence,
        lease,
        input_ids,
        positions,
        context,
    ) = _model_runner_phase_stitch_case()

    fallback = runner.run_phase_stitched_exact_graph(
        (sequence,),
        lease,
        input_ids=input_ids,
        positions=positions,
        context=context,
        current_block_table_identity=((5, 2), (6, 5)),
    )

    assert fallback.fallback_reason == "block_identity_drift"
    assert fallback.replay_count == 0
    assert not any(
        event[0] in ("prefill_replay", "decode_replay")
        for event in events
    )


def test_model_runner_phase_stitch_selects_bound_prefill_identity():
    (
        runner,
        _,
        _events,
        sequence,
        lease,
        input_ids,
        positions,
        context,
    ) = _model_runner_phase_stitch_case()
    bound_entry = next(
        iter(runner.exact_prefill_cuda_graph_cache.ready_entries.values())
    )
    decoy_entry = types.SimpleNamespace(
        identity=types.SimpleNamespace(token_count=16),
        identity_sha256="d" * 64,
        state="ready",
        tensors=bound_entry.tensors,
    )
    runner.exact_prefill_cuda_graph_cache.ready_entries = {
        decoy_entry.identity_sha256: decoy_entry,
        bound_entry.identity_sha256: bound_entry,
    }

    result = runner.run_phase_stitched_exact_graph(
        (sequence,),
        lease,
        input_ids=input_ids,
        positions=positions,
        context=context,
        current_block_table_identity=((5, 2), (6, 4)),
    )

    assert result.prefill_forward_count == 1
    assert result.decode_replay_count == 7


def test_model_runner_phase_stitch_mailbox_busy_falls_back_before_replay():
    (
        runner,
        _,
        events,
        sequence,
        lease,
        input_ids,
        positions,
        context,
    ) = _model_runner_phase_stitch_case()

    def reject_mailbox_ownership(_parent_identity):
        raise RuntimeError("mailbox busy")

    runner.phase_stitch_mailbox_backend.begin_transaction = (
        reject_mailbox_ownership
    )
    fallback = runner.run_phase_stitched_exact_graph(
        (sequence,),
        lease,
        input_ids=input_ids,
        positions=positions,
        context=context,
        current_block_table_identity=((5, 2), (6, 4)),
    )

    assert fallback.fallback_reason == "mailbox_acquire_failed"
    assert fallback.replay_count == 0
    assert not any(
        event[0] in ("prefill_replay", "decode_replay")
        for event in events
    )
    summary = runner.phase_stitch_summary()
    assert summary["failures"] == 0
    assert summary["quarantined_joint_identities"] == []


def test_model_runner_phase_stitch_quarantines_post_replay_failure():
    (
        runner,
        _,
        events,
        sequence,
        lease,
        input_ids,
        positions,
        context,
    ) = _model_runner_phase_stitch_case(fail_at=3)

    with pytest.raises(RuntimeError, match="decode exploded"):
        runner.run_phase_stitched_exact_graph(
            (sequence,),
            lease,
            input_ids=input_ids,
            positions=positions,
            context=context,
            current_block_table_identity=((5, 2), (6, 4)),
        )

    assert ("abort_mailbox", 1) in events
    summary = runner.phase_stitch_summary()
    assert summary["failures"] == 1
    assert summary["last_authoritative_phase"] == "decode_replay"
    assert summary["last_completed_decode_replays"] == 2
    assert summary["quarantined_joint_identities"] == [
        lease.source_identity_sha256,
    ]
