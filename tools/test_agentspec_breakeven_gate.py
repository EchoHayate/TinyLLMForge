"""Dependency-light tests for the action speculation Stage 0 gate."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
GATE_PATH = THIS_DIR / "agentspec_breakeven_gate.py"
SPEC = importlib.util.spec_from_file_location(
    "agentspec_breakeven_gate_under_test",
    os.fspath(GATE_PATH),
)
gate = importlib.util.module_from_spec(SPEC)
sys.modules["agentspec_breakeven_gate_under_test"] = gate
SPEC.loader.exec_module(gate)

ACTION = gate.ACTION
COST_MODEL = gate.COST_MODEL
LATENT = gate.LATENT_ADAPTER
ROUTER = gate.ROUTER


def _inputs(**overrides):
    payload = {
        "actor_gpu_seconds": 0.08,
        "draft_gpu_tax": 0.10,
        "tool_seconds": 1.0,
        "baseline_utilization": 0.3,
        "match_probability": 0.55,
        "rollback_seconds": 0.0,
    }
    payload.update(overrides)
    return COST_MODEL.build_cost_inputs(**payload)


def _expect_value_error(callable_object, *args, **kwargs):
    try:
        callable_object(*args, **kwargs)
    except ValueError:
        return True
    raise AssertionError("expected ValueError")


# --- action identity and side-effect classification ---------------


def test_action_signature_is_argument_order_invariant():
    first = ACTION.build_action_signature(
        tool_name="search",
        arguments={"a": 1, "b": 2},
    )
    second = ACTION.build_action_signature(
        tool_name="search",
        arguments={"b": 2, "a": 1},
    )
    assert first.digest == second.digest
    assert first.matches(second)


def test_action_signature_separates_tool_and_arguments():
    left = ACTION.build_action_signature(
        tool_name="a",
        arguments={"x": "b"},
    )
    right = ACTION.build_action_signature(
        tool_name="ab",
        arguments={"x": ""},
    )
    assert left.digest != right.digest


def test_action_signature_rejects_non_string_keys():
    _expect_value_error(
        ACTION.build_action_signature,
        tool_name="search",
        arguments={1: 2},
    )


def test_action_signature_rejects_non_dict_arguments():
    _expect_value_error(
        ACTION.build_action_signature,
        tool_name="search",
        arguments=["x"],
    )


def test_action_signature_does_not_match_foreign_type():
    signature = ACTION.build_action_signature(
        tool_name="search",
        arguments={},
    )
    assert not signature.matches("other")


def test_unknown_side_effect_class_fails_closed():
    contract = ACTION.build_tool_contract(
        tool_name="mystery",
        side_effect_class="unknown",
        rollback_seconds=0.0,
    )
    assert not contract.speculation_eligible
    assert contract.ineligibility_reason is not None


def test_irreversible_tool_is_never_eligible():
    contract = ACTION.build_tool_contract(
        tool_name="send_payment",
        side_effect_class="irreversible",
        rollback_seconds=0.0,
    )
    assert not contract.speculation_eligible


def test_sandboxable_tool_requires_a_sandbox():
    without = ACTION.build_tool_contract(
        tool_name="write_file",
        side_effect_class="sandboxable",
        rollback_seconds=0.1,
        sandbox_available=False,
    )
    with_sandbox = ACTION.build_tool_contract(
        tool_name="write_file",
        side_effect_class="sandboxable",
        rollback_seconds=0.1,
        sandbox_available=True,
    )
    assert not without.speculation_eligible
    assert with_sandbox.speculation_eligible


def test_read_only_tool_is_eligible_and_has_no_reason():
    contract = ACTION.build_tool_contract(
        tool_name="search",
        side_effect_class="read_only",
        rollback_seconds=0.0,
    )
    assert contract.speculation_eligible
    assert contract.ineligibility_reason is None
    payload = ACTION.tool_contract_to_dict(contract)
    assert payload["speculation_eligible"] is True


def test_tool_contract_rejects_unknown_class_and_bad_rollback():
    _expect_value_error(
        ACTION.build_tool_contract,
        tool_name="x",
        side_effect_class="not_a_class",
        rollback_seconds=0.0,
    )
    _expect_value_error(
        ACTION.build_tool_contract,
        tool_name="x",
        side_effect_class="read_only",
        rollback_seconds=-1.0,
    )


# --- cost model ---------------------------------------------------


def test_cost_inputs_reject_out_of_domain_values():
    _expect_value_error(_inputs, baseline_utilization=1.0)
    _expect_value_error(_inputs, match_probability=1.5)
    _expect_value_error(_inputs, draft_gpu_tax=-0.1)
    _expect_value_error(_inputs, actor_gpu_seconds=0.0)
    _expect_value_error(_inputs, rollback_seconds=-0.5)
    _expect_value_error(_inputs, draft_ready_fraction=1.5)


def test_default_draft_ready_fraction_matches_closed_form():
    inputs = _inputs(draft_gpu_tax=0.25)
    expected = 0.25 / 1.25
    assert abs(
        inputs.effective_draft_ready_fraction - expected
    ) < 1e-12
    zero_tax = _inputs(draft_gpu_tax=0.0)
    assert zero_tax.effective_draft_ready_fraction == 0.0


def test_stability_bound_closed_form():
    assert COST_MODEL.stability_tax_bound(0.0) == float("inf")
    assert abs(COST_MODEL.stability_tax_bound(0.5) - 1.0) < 1e-12
    assert abs(
        COST_MODEL.stability_tax_bound(0.8) - 0.25
    ) < 1e-12


def test_baseline_latency_closed_form():
    inputs = _inputs(baseline_utilization=0.6, tool_seconds=1.0)
    expected = 0.08 / 0.4 + 1.0
    assert abs(
        COST_MODEL.baseline_latency(inputs) - expected
    ) < 1e-12


def test_unstable_operating_point_is_reported_not_raised():
    result = COST_MODEL.evaluate(
        _inputs(baseline_utilization=0.9, draft_gpu_tax=1.0)
    )
    assert result.verdict == "unstable_capacity"
    assert result.stable is False
    assert result.speedup is None
    assert result.speculative_latency_seconds is None
    assert result.speculative_utilization >= 1.0


def test_unit_tax_drafter_has_no_match_benefit():
    result = COST_MODEL.evaluate(
        _inputs(baseline_utilization=0.0, draft_gpu_tax=1.0)
    )
    assert result.verdict == "infeasible_no_match_benefit"
    assert result.minimum_match_probability is None


def test_minimum_match_probability_is_the_indifference_point():
    inputs = _inputs(baseline_utilization=0.6, rollback_seconds=0.2)
    result = COST_MODEL.evaluate(inputs)
    assert result.minimum_match_probability is not None
    at_boundary = COST_MODEL.evaluate(
        _inputs(
            baseline_utilization=0.6,
            rollback_seconds=0.2,
            match_probability=result.minimum_match_probability,
        )
    )
    assert abs(
        at_boundary.speculative_latency_seconds
        - at_boundary.baseline_latency_seconds
    ) < 1e-9


def test_above_and_below_minimum_match_probability_flip_verdict():
    inputs = _inputs(baseline_utilization=0.6, rollback_seconds=0.2)
    threshold = COST_MODEL.evaluate(
        inputs
    ).minimum_match_probability
    assert 0.0 < threshold < 1.0
    below = COST_MODEL.evaluate(
        _inputs(
            baseline_utilization=0.6,
            rollback_seconds=0.2,
            match_probability=max(0.0, threshold - 0.05),
        )
    )
    above = COST_MODEL.evaluate(
        _inputs(
            baseline_utilization=0.6,
            rollback_seconds=0.2,
            match_probability=min(1.0, threshold + 0.05),
        )
    )
    assert below.verdict == "net_negative"
    assert above.verdict == "net_positive"


def test_capacity_ratio_and_waste_track_the_draft_tax():
    result = COST_MODEL.evaluate(
        _inputs(draft_gpu_tax=0.25, match_probability=0.5)
    )
    assert abs(result.capacity_ratio - 1.0 / 1.25) < 1e-12
    assert abs(
        result.wasted_gpu_fraction - 0.5 * 0.25 / 1.25
    ) < 1e-12


def test_critical_utilization_is_a_real_boundary():
    inputs = _inputs(draft_gpu_tax=0.10, match_probability=0.55)
    result = COST_MODEL.evaluate(inputs)
    boundary = result.critical_utilization
    assert boundary is not None
    just_below = COST_MODEL.evaluate(
        _inputs(baseline_utilization=boundary - 1e-4)
    )
    just_above = COST_MODEL.evaluate(
        _inputs(baseline_utilization=boundary + 1e-3)
    )
    assert just_below.verdict == "net_positive"
    assert just_above.verdict != "net_positive"


def test_critical_draft_tax_is_a_real_boundary():
    result = COST_MODEL.evaluate(_inputs())
    boundary = result.critical_draft_tax
    assert boundary is not None
    just_below = COST_MODEL.evaluate(
        _inputs(draft_gpu_tax=boundary - 1e-4)
    )
    just_above = COST_MODEL.evaluate(
        _inputs(draft_gpu_tax=boundary + 1e-3)
    )
    assert just_below.verdict == "net_positive"
    assert just_above.verdict != "net_positive"


def test_hit_saving_is_capped_by_actor_sojourn():
    for tool_seconds in (0.2, 1.0, 5.0, 30.0):
        result = COST_MODEL.evaluate(
            _inputs(tool_seconds=tool_seconds)
        )
        sojourn = (
            result.baseline_latency_seconds - tool_seconds
        )
        saving = (
            result.baseline_latency_seconds
            - result.hit_latency_seconds
        )
        assert saving <= sojourn + 1e-12


def test_relative_speedup_decays_when_tool_dominates():
    near = COST_MODEL.evaluate(_inputs(tool_seconds=1.0))
    far = COST_MODEL.evaluate(_inputs(tool_seconds=5.0))
    assert far.speedup < near.speedup
    assert far.speedup > 1.0


def test_break_even_to_dict_is_json_serialisable():
    payload = COST_MODEL.break_even_to_dict(
        COST_MODEL.evaluate(_inputs(baseline_utilization=0.0))
    )
    assert payload["stability_tax_bound"] is None
    assert "effective_draft_ready_fraction" in payload["inputs"]
    json.dumps(payload)


def test_evaluate_rejects_foreign_input_objects():
    _expect_value_error(COST_MODEL.evaluate, {"tax": 1.0})


# --- latent drafter contract --------------------------------------


def _capabilities(**overrides):
    payload = {
        "source_type": "latent_code_action_drafter",
        "representation": "discrete_code",
        "requires_target_hidden": True,
        "requires_compressed_kv": False,
        "max_candidate_actions": 3,
        "max_horizon_actions": 1,
        "emits_match_confidence": True,
    }
    payload.update(overrides)
    return LATENT.build_capabilities(**payload)


def _candidate(name, confidence):
    signature = ACTION.build_action_signature(
        tool_name="search",
        arguments={"query": name},
    )
    return LATENT.LatentActionCandidate(signature, confidence)


def _proposal(candidates, source="latent_code_action_drafter"):
    return LATENT.LatentActionDraftProposal(
        1,
        0,
        tuple(candidates),
        source,
        0.008,
    )


def test_latent_representation_must_require_hidden_state():
    _expect_value_error(
        _capabilities,
        requires_target_hidden=False,
    )
    text = _capabilities(
        representation="text",
        requires_target_hidden=False,
    )
    assert text.representation == "text"


def test_capabilities_reject_unknown_representation():
    _expect_value_error(_capabilities, representation="telepathy")


def test_context_requires_declared_inputs():
    capabilities = _capabilities(requires_compressed_kv=True)
    missing_kv = LATENT.LatentActionDraftContext(
        1,
        0,
        (),
        (),
        target_hidden=object(),
    )
    _expect_value_error(
        LATENT.validate_context,
        capabilities,
        missing_kv,
    )
    complete = LATENT.LatentActionDraftContext(
        1,
        0,
        (),
        (),
        target_hidden=object(),
        compressed_kv_handle=object(),
    )
    assert (
        LATENT.validate_context(capabilities, complete) is complete
    )


def test_context_requires_hidden_state_when_declared():
    capabilities = _capabilities()
    context = LATENT.LatentActionDraftContext(1, 0, (), ())
    _expect_value_error(
        LATENT.validate_context,
        capabilities,
        context,
    )


def test_context_rejects_malformed_collections():
    capabilities = _capabilities(requires_target_hidden=False,
                                 representation="text")
    bad_digests = LATENT.LatentActionDraftContext(1, 0, [], ())
    _expect_value_error(
        LATENT.validate_context,
        capabilities,
        bad_digests,
    )
    bad_contracts = LATENT.LatentActionDraftContext(
        1,
        0,
        (),
        ("not_a_contract",),
    )
    _expect_value_error(
        LATENT.validate_context,
        capabilities,
        bad_contracts,
    )


def test_proposal_confidences_must_be_a_valid_distribution():
    capabilities = _capabilities()
    over_one = _proposal(
        [_candidate("a", 0.7), _candidate("b", 0.6)]
    )
    _expect_value_error(
        LATENT.validate_proposal,
        capabilities,
        over_one,
    )
    out_of_range = _proposal([_candidate("a", 1.2)])
    _expect_value_error(
        LATENT.validate_proposal,
        capabilities,
        out_of_range,
    )


def test_proposal_must_be_sorted_and_unique():
    capabilities = _capabilities()
    unsorted = _proposal(
        [_candidate("a", 0.2), _candidate("b", 0.5)]
    )
    _expect_value_error(
        LATENT.validate_proposal,
        capabilities,
        unsorted,
    )
    duplicate = _proposal(
        [_candidate("a", 0.4), _candidate("a", 0.3)]
    )
    _expect_value_error(
        LATENT.validate_proposal,
        capabilities,
        duplicate,
    )


def test_proposal_respects_declared_limits_and_source():
    capabilities = _capabilities(max_candidate_actions=1)
    too_many = _proposal(
        [_candidate("a", 0.5), _candidate("b", 0.2)]
    )
    _expect_value_error(
        LATENT.validate_proposal,
        capabilities,
        too_many,
    )
    wrong_source = _proposal(
        [_candidate("a", 0.5)],
        source="other_drafter",
    )
    _expect_value_error(
        LATENT.validate_proposal,
        capabilities,
        wrong_source,
    )


def test_uncalibrated_drafter_must_emit_one_certain_candidate():
    capabilities = _capabilities(emits_match_confidence=False)
    hedged = _proposal([_candidate("a", 0.6)])
    _expect_value_error(
        LATENT.validate_proposal,
        capabilities,
        hedged,
    )
    certain = _proposal([_candidate("a", 1.0)])
    assert (
        LATENT.validate_proposal(capabilities, certain) is certain
    )


def test_eligible_candidates_drops_unsafe_and_unknown_tools():
    contracts = (
        ACTION.build_tool_contract(
            tool_name="search",
            side_effect_class="read_only",
            rollback_seconds=0.0,
        ),
        ACTION.build_tool_contract(
            tool_name="send_payment",
            side_effect_class="irreversible",
            rollback_seconds=0.0,
        ),
    )
    payment = ACTION.build_action_signature(
        tool_name="send_payment",
        arguments={"amount": 1},
    )
    undeclared = ACTION.build_action_signature(
        tool_name="ghost_tool",
        arguments={},
    )
    candidates = (
        _candidate("a", 0.5),
        LATENT.LatentActionCandidate(payment, 0.3),
        LATENT.LatentActionCandidate(undeclared, 0.1),
    )
    kept = LATENT.eligible_candidates(
        _proposal(candidates),
        contracts,
    )
    assert len(kept) == 1
    assert kept[0].signature.tool_name == "search"


def test_aggregate_match_probability_widens_and_clamps():
    candidates = (
        _candidate("a", 0.5),
        _candidate("b", 0.3),
        _candidate("c", 0.3),
    )
    assert abs(
        LATENT.aggregate_match_probability(candidates, 1) - 0.5
    ) < 1e-12
    assert abs(
        LATENT.aggregate_match_probability(candidates, 2) - 0.8
    ) < 1e-12
    assert LATENT.aggregate_match_probability(candidates, 9) == 1.0
    _expect_value_error(
        LATENT.aggregate_match_probability,
        candidates,
        0,
    )


# --- router -------------------------------------------------------


def _route(**overrides):
    payload = {
        "proposal_available": True,
        "eligible_candidate_count": 1,
        "branch_count": 1,
        "aggregate_match_probability": 0.55,
        "break_even": COST_MODEL.evaluate(_inputs()),
    }
    payload.update(overrides)
    return ROUTER.choose_action_speculation_route(**payload)


def test_router_speculates_on_a_profitable_safe_point():
    route = _route()
    assert route.name == "speculative_commit_on_match"
    assert route.speculative_tool_calls == 1
    assert route.fallback_reason is None


def test_router_falls_back_without_a_proposal():
    assert (
        _route(proposal_available=False).name
        == "baseline_no_proposal"
    )
    assert _route(branch_count=0).name == "baseline_no_proposal"


def test_router_side_effect_guard_precedes_capacity_and_profit():
    unstable = COST_MODEL.evaluate(
        _inputs(baseline_utilization=0.9, draft_gpu_tax=1.0)
    )
    route = _route(
        eligible_candidate_count=0,
        break_even=unstable,
    )
    assert route.name == "baseline_side_effect_guard"


def test_router_capacity_guard_precedes_profitability():
    unstable = COST_MODEL.evaluate(
        _inputs(baseline_utilization=0.9, draft_gpu_tax=1.0)
    )
    route = _route(break_even=unstable)
    assert route.name == "baseline_capacity_guard"


def test_router_refuses_net_negative_points_by_default():
    negative = COST_MODEL.evaluate(
        _inputs(
            baseline_utilization=0.3,
            draft_gpu_tax=0.1,
            match_probability=0.1,
            rollback_seconds=2.0,
        )
    )
    assert negative.verdict == "net_negative"
    assert _route(break_even=negative).name == (
        "baseline_unprofitable"
    )
    forced = ROUTER.choose_action_speculation_route(
        proposal_available=True,
        eligible_candidate_count=1,
        branch_count=1,
        aggregate_match_probability=0.1,
        break_even=negative,
        allow_unprofitable=True,
    )
    assert forced.name == "speculative_commit_on_match"
    assert forced.fallback_reason is not None


def test_router_clamps_branch_count_to_eligible_candidates():
    route = _route(branch_count=4, eligible_candidate_count=2)
    assert route.branch_count == 2
    assert route.speculative_tool_calls == 2


def test_router_validates_arguments():
    _expect_value_error(_route, aggregate_match_probability=1.5)
    _expect_value_error(_route, branch_count=-1)
    _expect_value_error(_route, break_even={"verdict": "ok"})
    _expect_value_error(_route, proposal_available="yes")
    payload = ROUTER.route_to_dict(_route())
    assert payload["name"] == "speculative_commit_on_match"


# --- gate ---------------------------------------------------------


def test_gate_report_passes_and_is_deterministic():
    first = gate.build_report()
    second = gate.build_report()
    assert first["status"] == "PASS"
    assert first["payload_sha256"] == second["payload_sha256"]
    assert first["matrix_row_count"] == 270


def test_gate_invariants_are_all_pass():
    report = gate.build_report()
    failed = [
        item
        for item in report["invariants"]
        if item["status"] != "PASS"
    ]
    assert not failed
    assert len(report["invariants"]) >= 12


def test_gate_headline_ranks_cheap_drafters_higher():
    report = gate.build_report()
    per_drafter = report["headline"]["per_drafter"]
    code = per_drafter["discrete_code"]
    latent = per_drafter["continuous_latent"]
    text = per_drafter["text_small_model"]
    assert (
        code["critical_utilization"]
        > latent["critical_utilization"]
    )
    assert text["critical_utilization"] is None
    assert not text["profitable_utilizations"]


def test_gate_declares_its_claim_boundary():
    report = gate.build_report()
    assert "no measured speedup" in report["claim_boundary"]


def test_gate_cli_writes_a_stable_artifact():
    with tempfile.TemporaryDirectory() as directory:
        command = [
            sys.executable,
            os.fspath(GATE_PATH),
            "--output-dir",
            directory,
        ]
        completed = subprocess.run(
            command,
            cwd=os.fspath(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        target = (
            Path(directory) / "agentspec_breakeven_report.json"
        )
        payload = json.loads(target.read_text(encoding="utf-8"))
        assert payload["status"] == "PASS"
        assert payload["gate"] == "agentspec_breakeven"
        assert payload["schema_version"] == 1
