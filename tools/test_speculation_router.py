import importlib.util
import hashlib
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
if os.fspath(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, os.fspath(TOOLS_DIR))

ROUTER_PATH = REPO_ROOT / "tinyvllm" / "speculative" / "router.py"
SPEC = importlib.util.spec_from_file_location(
    "speculation_router_under_test",
    os.fspath(ROUTER_PATH),
)
router = importlib.util.module_from_spec(SPEC)
sys.modules["speculation_router_under_test"] = router
SPEC.loader.exec_module(router)

choose_speculation_route = router.choose_speculation_route
route_to_dict = router.route_to_dict

PROFILE_PATH = TOOLS_DIR / "profile_ngram_commit.py"
PROFILE_SPEC = importlib.util.spec_from_file_location(
    "speculation_profile_under_test",
    os.fspath(PROFILE_PATH),
)
profile = importlib.util.module_from_spec(PROFILE_SPEC)
sys.modules["speculation_profile_under_test"] = profile
PROFILE_SPEC.loader.exec_module(profile)


def test_finished_precedes_every_other_decision():
    route = choose_speculation_route(
        draft_len=8,
        finished=True,
        remaining_output_budget=16,
        native_compatible=True,
    )
    assert route.name == "baseline_finished"


def test_output_budget_precedes_short_draft():
    route = choose_speculation_route(
        draft_len=1,
        finished=False,
        remaining_output_budget=0,
        native_compatible=True,
    )
    assert route.name == "baseline_output_budget"


def test_zero_and_one_token_drafts_use_baseline():
    for draft_len in (0, 1):
        route = choose_speculation_route(
            draft_len=draft_len,
            finished=False,
            remaining_output_budget=16,
            native_compatible=True,
        )
        assert route.name == "baseline_short_draft"


def test_compatible_multi_token_draft_uses_native():
    route = choose_speculation_route(
        draft_len=4,
        finished=False,
        remaining_output_budget=16,
        native_compatible=True,
    )
    assert route.name == "native_multi_token"


def test_controlled_incompatibility_fails_closed():
    try:
        choose_speculation_route(
            draft_len=4,
            finished=False,
            remaining_output_budget=16,
            native_compatible=False,
            compatibility_reason="kv_offload_mvp0",
            allow_incompatible_fallback=False,
        )
    except ValueError as exc:
        assert "kv_offload_mvp0" in str(exc)
    else:
        raise AssertionError("expected fail-closed incompatibility")


def test_real_source_incompatibility_records_fallback():
    route = choose_speculation_route(
        draft_len=4,
        finished=False,
        remaining_output_budget=16,
        native_compatible=False,
        compatibility_reason="kv_offload_mvp0",
        allow_incompatible_fallback=True,
    )
    assert route_to_dict(route) == {
        "name": "baseline_incompatible",
        "draft_len": 4,
        "native_compatible": False,
        "fallback_reason": "kv_offload_mvp0",
    }


def test_negative_draft_length_is_rejected():
    try:
        choose_speculation_route(
            draft_len=-1,
            finished=False,
            remaining_output_budget=16,
            native_compatible=True,
        )
    except ValueError as exc:
        assert "draft_len" in str(exc)
    else:
        raise AssertionError("expected negative draft length rejection")


class FakeBlockManager:
    def __init__(self):
        self.reserve_calls = 0

    def reserve_append_blocks(self, seq, count):
        self.reserve_calls += 1
        raise AssertionError("baseline route must not reserve blocks")


class FakeRunner:
    def __init__(self, compatible=True):
        self.compatible = compatible
        self.validate_calls = 0

    def _validate_spec_verify_compatibility(self, **kwargs):
        self.validate_calls += 1
        if not self.compatible:
            raise RuntimeError("kv_offload_mvp0 is unsupported")


class FakeSeq:
    is_finished = False
    max_tokens = 32
    num_completion_tokens = 4


def _fake_llm(*, compatible=True):
    llm = type("LLM", (), {})()
    llm.model_runner = FakeRunner(compatible=compatible)
    llm.scheduler = type("Scheduler", (), {})()
    llm.scheduler.block_manager = FakeBlockManager()
    return llm


def test_short_draft_wrapper_performs_no_target_or_kv_work():
    llm = _fake_llm()
    event = profile.route_and_verify_draft(
        llm,
        FakeSeq(),
        [7],
        draft_source="fixture",
        allow_incompatible_fallback=False,
    )
    assert event["route"] == "baseline_short_draft"
    assert event["accepted_count"] == 0
    assert event["target_forward_count"] == 0
    assert event["speculative_reservation_attempted"] is False
    assert event["spec_verify_prepare_calls"] == 0
    assert event["spec_verify_forward_calls"] == 0
    assert event["accepted_kv_replay_calls"] == 0
    assert llm.model_runner.validate_calls == 0
    assert llm.scheduler.block_manager.reserve_calls == 0


def test_real_source_incompatibility_falls_back_before_mutation():
    llm = _fake_llm(compatible=False)
    event = profile.route_and_verify_draft(
        llm,
        FakeSeq(),
        [1, 2, 3, 4],
        draft_source="fixture",
        allow_incompatible_fallback=True,
    )
    assert event["route"] == "baseline_incompatible"
    assert "kv_offload_mvp0" in event["route_fallback_reason"]
    assert event["target_forward_count"] == 0
    assert llm.model_runner.validate_calls == 1
    assert llm.scheduler.block_manager.reserve_calls == 0


def test_controlled_incompatibility_raises_before_mutation():
    llm = _fake_llm(compatible=False)
    try:
        profile.route_and_verify_draft(
            llm,
            FakeSeq(),
            [1, 2, 3, 4],
            draft_source="fixture",
            allow_incompatible_fallback=False,
        )
    except ValueError as exc:
        assert "kv_offload_mvp0" in str(exc)
    else:
        raise AssertionError("expected controlled incompatibility failure")
    assert llm.scheduler.block_manager.reserve_calls == 0


def test_multi_token_wrapper_delegates_to_native_verifier():
    llm = _fake_llm()
    calls = []
    original = profile.verify_and_commit_block

    def fake_verify(llm_arg, seq_arg, draft_tokens, **kwargs):
        calls.append((llm_arg, seq_arg, list(draft_tokens), kwargs))
        return {
            "query_len": 3,
            "accepted_count": 4,
            "accepted_kv_replay_calls": 0,
        }

    profile.verify_and_commit_block = fake_verify
    try:
        event = profile.route_and_verify_draft(
            llm,
            FakeSeq(),
            [1, 2, 3, 4],
            draft_source="fixture",
            allow_incompatible_fallback=False,
        )
    finally:
        profile.verify_and_commit_block = original

    assert len(calls) == 1
    assert calls[0][2] == [1, 2, 3, 4]
    assert calls[0][3]["draft_source"] == "fixture"
    assert calls[0][3]["verifier_mode"] == "native"
    assert event["route"] == "native_multi_token"
    assert event["speculative_reservation_attempted"] is True
    assert event["spec_verify_prepare_calls"] == 1
    assert event["spec_verify_forward_calls"] == 1


def test_draft_verification_dispatches_fixed_profitability():
    calls = []
    original_route = profile.route_and_verify_draft

    def fake_route(llm, seq, draft_tokens, **kwargs):
        calls.append((list(draft_tokens), kwargs))
        return {"route": "baseline_short_draft", "accepted_count": 0}

    args = type("Args", (), {
        "speculation_routing": "fixed-profitability",
        "allow_incompatible_fallback": True,
        "simulate_kv_upload_mb": 0.0,
        "debug_target_hidden": False,
        "debug_hidden_to_draft_stub": False,
        "hidden_to_draft_adapter": "topk-stub",
        "debug_hidden_to_draft_top_k": 3,
        "verifier_mode": "legacy_rematerialize",
    })()
    draft = profile.DraftProposal(tokens=[9], source="fixture")
    profile.route_and_verify_draft = fake_route
    try:
        event = profile._run_draft_verification(
            object(),
            object(),
            draft,
            args,
        )
    finally:
        profile.route_and_verify_draft = original_route

    assert event["route"] == "baseline_short_draft"
    assert calls == [(
        [9],
        {
            "draft_source": "fixture",
            "allow_incompatible_fallback": True,
            "simulate_kv_upload_mb": 0.0,
            "debug_target_hidden": False,
            "debug_hidden_to_draft_stub": False,
            "hidden_to_draft_adapter": "topk-stub",
            "debug_hidden_to_draft_top_k": 3,
        },
    )]


def test_draft_verification_dispatches_always_native():
    calls = []
    original_verify = profile.verify_and_commit_block

    def fake_verify(llm, seq, draft_tokens, **kwargs):
        calls.append((list(draft_tokens), kwargs))
        return {"accepted_count": 0}

    args = type("Args", (), {
        "speculation_routing": "always-native",
        "allow_incompatible_fallback": False,
        "simulate_kv_upload_mb": 0.0,
        "debug_target_hidden": False,
        "debug_hidden_to_draft_stub": False,
        "hidden_to_draft_adapter": "topk-stub",
        "debug_hidden_to_draft_top_k": 3,
        "verifier_mode": "legacy_rematerialize",
    })()
    draft = profile.DraftProposal(tokens=[1, 2], source="fixture")
    profile.verify_and_commit_block = fake_verify
    try:
        profile._run_draft_verification(object(), object(), draft, args)
    finally:
        profile.verify_and_commit_block = original_verify
    assert calls[0][1]["verifier_mode"] == "native"


def test_route_summary_counts_routes_and_fallback_reasons():
    summary = profile.summarize_route_events([
        {
            "route": "baseline_short_draft",
            "route_fallback_reason": None,
        },
        {
            "route": "baseline_incompatible",
            "route_fallback_reason": "kv_offload_mvp0",
        },
        {
            "route": "native_multi_token",
            "route_fallback_reason": None,
        },
    ])
    assert summary["route_attempts"] == 3
    assert summary["route_counts"] == {
        "baseline_short_draft": 1,
        "baseline_finished": 0,
        "baseline_output_budget": 0,
        "baseline_incompatible": 1,
        "native_multi_token": 1,
    }
    assert summary["fallback_reason_counts"] == {
        "kv_offload_mvp0": 1,
    }


def _profile_args(**overrides):
    values = {
        "model": "model",
        "temperature": 0.0,
        "max_commit_events": 1,
        "warmup_output_len": 0,
        "simulate_kv_upload_mb": 0.0,
        "max_draft_tokens": 4,
        "draft_policy": "fixed",
        "draft_source": "ngram",
        "mode": "candidate-only",
        "max_num_seqs": 1,
        "verifier_mode": "legacy_rematerialize",
        "speculation_routing": "fixed-profitability",
        "allow_incompatible_fallback": False,
        "gate_stage": "controlled",
        "draft_construction": "controlled_target_derived",
        "enforce_eager": True,
        "kv_quant_bits": 0,
        "kv_offload_mvp0": False,
        "kv_offload_blockwise_decode": False,
        "kv_offload_blockwise_prefill": False,
        "quest_top_k_blocks": -1,
        "am_compact_blocks": 0,
        "kv_cartridge_blocks": 0,
    }
    values.update(overrides)
    return type("Args", (), values)()


def test_routing_profile_args_require_candidate_single_sequence_eager():
    profile.validate_profile_args(_profile_args())
    for name, value in (
        ("mode", "paired"),
        ("max_num_seqs", 2),
        ("enforce_eager", False),
    ):
        try:
            profile.validate_profile_args(_profile_args(**{name: value}))
        except ValueError as exc:
            assert "speculation routing" in str(exc)
        else:
            raise AssertionError(name)


def test_incompatible_fallback_requires_fixed_routing():
    try:
        profile.validate_profile_args(_profile_args(
            speculation_routing="always-native",
            allow_incompatible_fallback=True,
        ))
    except ValueError as exc:
        assert "allow-incompatible-fallback" in str(exc)
    else:
        raise AssertionError("always-native fallback was accepted")


def test_real_source_stage_rejects_controlled_target_derived():
    try:
        profile.validate_profile_args(_profile_args(
            gate_stage="real-source",
            draft_construction="controlled_target_derived",
        ))
    except ValueError as exc:
        assert "controlled_target_derived" in str(exc)
    else:
        raise AssertionError("controlled draft entered real-source gate")


def test_real_source_event_records_proposal_and_route_evidence():
    event = {
        "route": "native_multi_token",
        "route_fallback_reason": None,
        "accepted_count": 3,
        "target_forward_count": 1,
    }
    draft = profile.DraftProposal(
        tokens=[10, 20, 30, 40],
        source="fixture-learned-drafter",
        metadata={"temperature": 0.0, "checkpoint": "fixture"},
    )
    result = profile.attach_gate_source_evidence(
        event,
        draft,
        gate_stage="real-source",
        draft_construction="real_source",
        proposal_started_s=10.0,
        proposal_finished_s=10.002,
    )

    expected_hash = hashlib.sha256(
        json.dumps(
            draft.metadata,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert result["proposal_started_s"] == 10.0
    assert result["proposal_finished_s"] == 10.002
    assert abs(result["proposal_elapsed_ms"] - 2.0) < 1e-9
    assert result["proposed_tokens"] == [10, 20, 30, 40]
    assert result["proposed_count"] == 4
    assert result["accepted_count"] == 3
    assert result["rejected_count"] == 1
    assert result["source_metadata_sha256"] == expected_hash
    assert result["route"] == "native_multi_token"
    assert result["target_forward_count"] == 1


def main():
    test_finished_precedes_every_other_decision()
    test_output_budget_precedes_short_draft()
    test_zero_and_one_token_drafts_use_baseline()
    test_compatible_multi_token_draft_uses_native()
    test_controlled_incompatibility_fails_closed()
    test_real_source_incompatibility_records_fallback()
    test_negative_draft_length_is_rejected()
    test_short_draft_wrapper_performs_no_target_or_kv_work()
    test_real_source_incompatibility_falls_back_before_mutation()
    test_controlled_incompatibility_raises_before_mutation()
    test_multi_token_wrapper_delegates_to_native_verifier()
    test_draft_verification_dispatches_fixed_profitability()
    test_draft_verification_dispatches_always_native()
    test_route_summary_counts_routes_and_fallback_reasons()
    test_routing_profile_args_require_candidate_single_sequence_eager()
    test_incompatible_fallback_requires_fixed_routing()
    test_real_source_stage_rejects_controlled_target_derived()
    test_real_source_event_records_proposal_and_route_evidence()
    print("speculation router tests passed")


if __name__ == "__main__":
    main()
