from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from tools.slo_cohort_burst_ceiling import build_frozen_cost_table


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    REPO_ROOT
    / "tinyvllm"
    / "engine"
    / "slo_cohort_burst.py"
)
SPEC = importlib.util.spec_from_file_location(
    "slo_cohort_burst_under_test",
    MODULE_PATH,
)
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)

ProtectedRequestSnapshot = module.ProtectedRequestSnapshot
RequestSLOState = module.RequestSLOState
SLOCohortBurstObservation = module.SLOCohortBurstObservation
SLOCohortCostTable = module.SLOCohortCostTable
select_slo_cohort_burst_width = module.select_slo_cohort_burst_width


SOURCE_IDENTITY = {
    "source_commit": "a" * 40,
    "source_patch_sha256": "b" * 64,
    "model": "Qwen3-0.6B",
    "checkpoint_sha256": "c" * 64,
    "gpu_uuid": "GPU-test",
    "gpu_name": "NVIDIA A100 80GB PCIe",
    "tensor_parallel_size": 1,
    "dtype": "torch.bfloat16",
    "config_sha256": "d" * 64,
}


def _state(
    sequence_id: int,
    *,
    arrival_ns: int = 0,
    first_token_visible_ns: int | None = 0,
    last_token_visible_ns: int | None = 0,
) -> RequestSLOState:
    return RequestSLOState(
        sequence_id=sequence_id,
        arrival_ns=arrival_ns,
        first_token_visible_ns=first_token_visible_ns,
        last_token_visible_ns=last_token_visible_ns,
        service_class="default",
    )


def _snapshot(
    sequence_id: int,
    *,
    category: str = "cohort",
    context_bucket: int = 2048,
    remaining_output_tokens: int = 8,
    writable_tokens: int = 8,
    state: RequestSLOState | None = None,
) -> ProtectedRequestSnapshot:
    return ProtectedRequestSnapshot(
        sequence_id=sequence_id,
        category=category,
        context_bucket=context_bucket,
        remaining_output_tokens=remaining_output_tokens,
        writable_tokens=writable_tokens,
        slo_state=state if state is not None else _state(sequence_id),
    )


def _table(
    *,
    batch_size: int = 2,
    costs: dict[int, int] | None = None,
    context_buckets: tuple[int, ...] = (2048,),
) -> SLOCohortCostTable:
    costs = costs or {1: 5_000_000, 2: 9_000_000, 4: 18_000_000, 8: 30_000_000}
    rows = []
    for context_bucket in context_buckets:
        for width, duration_ns in costs.items():
            rows.append({
                "schema_version": "slo-cohort-burst.cost-sample.v1",
                "sample_id": f"b{batch_size}-c{context_bucket}-k{width}",
                "batch_size": batch_size,
                "context_bucket": context_bucket,
                "burst_width": width,
                "duration_ns": duration_ns,
            })
    return SLOCohortCostTable.from_payload(
        build_frozen_cost_table(rows, SOURCE_IDENTITY)
    )


def _observation(
    *,
    decision_now_ns: int = 71_000_000,
    cohort: tuple[ProtectedRequestSnapshot, ...] | None = None,
    omitted_decode: tuple[ProtectedRequestSnapshot, ...] = (),
    waiting: tuple[ProtectedRequestSnapshot, ...] = (),
    incomplete_prefill: tuple[ProtectedRequestSnapshot, ...] = (),
    **overrides,
) -> SLOCohortBurstObservation:
    values = {
        "enabled": True,
        "decision_now_ns": decision_now_ns,
        "target_itl_ns": 100_000_000,
        "target_ttft_ns": 100_000_000,
        "reserve_ns": 10_000_000,
        "configured_widths": (1, 2, 4, 8),
        "cohort": cohort
        or (
            _snapshot(1),
            _snapshot(2),
        ),
        "omitted_runnable_decode": omitted_decode,
        "waiting": waiting,
        "incomplete_prefill": incomplete_prefill,
        "clock_valid": True,
        "all_greedy": True,
        "mixed_mode_unsupported": False,
        "graph_available": True,
        "graph_quarantined": False,
        "pending_lease": False,
        "cohort_shape_supported": True,
    }
    values.update(overrides)
    return SLOCohortBurstObservation(**values)


def test_policy_types_are_immutable() -> None:
    state = _state(1)
    snapshot = _snapshot(1, state=state)
    observation = _observation(cohort=(snapshot,))
    decision = select_slo_cohort_burst_width(
        observation,
        _table(batch_size=1),
    )
    for value, field, replacement in (
        (state, "arrival_ns", 1),
        (snapshot, "remaining_output_tokens", 1),
        (observation, "enabled", False),
        (decision, "selected_width", 1),
    ):
        with pytest.raises(FrozenInstanceError):
            setattr(value, field, replacement)


def test_selector_uses_minimum_slack_across_all_protected_requests() -> None:
    cohort = (
        _snapshot(1, state=_state(1, last_token_visible_ns=71_000_000)),
        _snapshot(2, state=_state(2, last_token_visible_ns=61_000_000)),
    )
    omitted = (
        _snapshot(
            3,
            category="omitted_decode",
            state=_state(
                3,
                first_token_visible_ns=0,
                last_token_visible_ns=0,
            ),
        ),
    )
    waiting = (
        _snapshot(
            4,
            category="waiting",
            state=_state(
                4,
                arrival_ns=21_000_000,
                first_token_visible_ns=None,
                last_token_visible_ns=None,
            ),
        ),
    )
    decision = select_slo_cohort_burst_width(
        _observation(
            cohort=cohort,
            omitted_decode=omitted,
            waiting=waiting,
        ),
        _table(),
    )
    assert decision.global_slack_ns == 19_000_000
    assert decision.selected_width == 4
    assert decision.reason == "selected"
    assert decision.protected_sequence_ids == (1, 2, 3, 4)


def test_unemitted_request_uses_ttft_slack() -> None:
    waiting = _snapshot(
        3,
        category="waiting",
        state=_state(
            3,
            arrival_ns=0,
            first_token_visible_ns=None,
            last_token_visible_ns=None,
        ),
    )
    decision = select_slo_cohort_burst_width(
        _observation(
            decision_now_ns=75_000_000,
            waiting=(waiting,),
        ),
        _table(),
    )
    assert decision.global_slack_ns == 15_000_000
    assert decision.selected_width == 2


def test_selector_uses_maximum_cost_across_cohort_context_buckets() -> None:
    cohort = (
        _snapshot(1, context_bucket=1024),
        _snapshot(2, context_bucket=2048),
    )
    rows = []
    for context_bucket, costs in (
        (1024, {1: 4_000_000, 2: 8_000_000, 4: 16_000_000, 8: 20_000_000}),
        (2048, {1: 5_000_000, 2: 9_000_000, 4: 18_000_000, 8: 35_000_000}),
    ):
        for width, duration_ns in costs.items():
            rows.append({
                "schema_version": "slo-cohort-burst.cost-sample.v1",
                "sample_id": f"b2-c{context_bucket}-k{width}",
                "batch_size": 2,
                "context_bucket": context_bucket,
                "burst_width": width,
                "duration_ns": duration_ns,
            })
    table = SLOCohortCostTable.from_payload(
        build_frozen_cost_table(rows, SOURCE_IDENTITY)
    )
    decision = select_slo_cohort_burst_width(
        _observation(cohort=cohort),
        table,
    )
    assert dict(decision.predicted_cost_ns_by_width)[8] == 35_000_000
    assert decision.selected_width == 4


@pytest.mark.parametrize(
    ("changes", "expected_reason"),
    (
        ({"enabled": False}, "disabled"),
        ({"clock_valid": False}, "clock_invalid"),
        (
            {
                "cohort": (
                    _snapshot(1),
                    replace(_snapshot(2), slo_state=None),
                ),
            },
            "missing_slo_state",
        ),
        ({"all_greedy": False}, "non_greedy_request"),
        ({"mixed_mode_unsupported": True}, "mixed_mode_unsupported"),
        ({"graph_available": False}, "graph_unavailable"),
        ({"graph_quarantined": True}, "graph_quarantined"),
        ({"pending_lease": True}, "pending_lease"),
        ({"cohort_shape_supported": False}, "cohort_shape_unsupported"),
        (
            {
                "cohort": (
                    _snapshot(1, remaining_output_tokens=1),
                    _snapshot(2),
                ),
            },
            "insufficient_output_budget",
        ),
        (
            {
                "cohort": (
                    _snapshot(1, writable_tokens=1),
                    _snapshot(2),
                ),
            },
            "kv_block_boundary",
        ),
    ),
)
def test_selector_reports_structural_fallback_reasons(
    changes: dict,
    expected_reason: str,
) -> None:
    decision = select_slo_cohort_burst_width(
        _observation(**changes),
        _table(),
    )
    assert decision.selected_width == 1
    assert decision.reason == expected_reason


def test_selector_fallback_precedence_is_stable() -> None:
    invalid_table = SLOCohortCostTable.invalid("test")
    decision = select_slo_cohort_burst_width(
        _observation(
            enabled=False,
            clock_valid=False,
            cohort=(replace(_snapshot(1), slo_state=None),),
            all_greedy=False,
            graph_available=False,
        ),
        invalid_table,
    )
    assert decision.selected_width == 1
    assert decision.reason == "disabled"


def test_invalid_timestamp_precedes_invalid_cost_table() -> None:
    future = _snapshot(
        1,
        state=_state(1, last_token_visible_ns=72_000_000),
    )
    decision = select_slo_cohort_burst_width(
        _observation(cohort=(future,)),
        SLOCohortCostTable.invalid("test"),
    )
    assert decision.reason == "clock_invalid"


def test_missing_cost_key_is_cost_table_invalid() -> None:
    decision = select_slo_cohort_burst_width(
        _observation(),
        _table(costs={1: 5_000_000, 2: 9_000_000}),
    )
    assert decision.selected_width == 1
    assert decision.reason == "cost_table_invalid"


def test_no_positive_slack_precedes_predicted_cost_rejection() -> None:
    decision = select_slo_cohort_burst_width(
        _observation(decision_now_ns=100_000_000),
        _table(costs={1: 1, 2: 1, 4: 1, 8: 1}),
    )
    assert decision.global_slack_ns == -10_000_000
    assert decision.reason == "no_slo_slack"


def test_predicted_cost_exceeds_slack_after_all_widths_are_checked() -> None:
    decision = select_slo_cohort_burst_width(
        _observation(),
        _table(costs={
            1: 5_000_000,
            2: 20_000_000,
            4: 30_000_000,
            8: 40_000_000,
        }),
    )
    assert decision.global_slack_ns == 19_000_000
    assert decision.selected_width == 1
    assert decision.reason == "predicted_cost_exceeds_slack"


def test_width_is_clipped_without_changing_cohort_membership() -> None:
    cohort = (
        _snapshot(7, remaining_output_tokens=4, writable_tokens=8),
        _snapshot(9, remaining_output_tokens=8, writable_tokens=4),
    )
    decision = select_slo_cohort_burst_width(
        _observation(cohort=cohort),
        _table(),
    )
    assert decision.selected_width == 4
    assert decision.protected_sequence_ids[:2] == (7, 9)


def test_cost_table_load_verifies_hash_and_schema(tmp_path) -> None:
    table = _table()
    path = tmp_path / "cost_table.json"
    path.write_text(
        json.dumps(table.to_payload(), sort_keys=True),
        encoding="utf-8",
    )
    loaded = SLOCohortCostTable.load(path)
    assert loaded.table_sha256 == table.table_sha256
    assert loaded.predicted_cost_ns(2, 2048, 8) == 30_000_000

    payload = table.to_payload()
    payload["entries"]["b2-c2048-k8"]["p99_ns"] += 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        SLOCohortCostTable.load(path)


def test_cost_table_rejects_invalid_source_identity() -> None:
    payload = _table().to_payload()
    payload["source_identity"] = {}
    with pytest.raises(ValueError, match="source identity"):
        SLOCohortCostTable.from_payload(payload)


@pytest.mark.parametrize(
    "state",
    (
        _state(1, arrival_ns=-1),
        _state(
            1,
            first_token_visible_ns=None,
            last_token_visible_ns=10,
        ),
        _state(
            1,
            arrival_ns=10,
            first_token_visible_ns=9,
            last_token_visible_ns=9,
        ),
        _state(
            1,
            first_token_visible_ns=11,
            last_token_visible_ns=10,
        ),
    ),
)
def test_request_slo_state_rejects_invalid_timestamp_order(
    state: RequestSLOState,
) -> None:
    with pytest.raises(ValueError):
        state.validate()
