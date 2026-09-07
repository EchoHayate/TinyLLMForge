from dataclasses import replace
import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "tinyvllm"
    / "engine"
    / "segmented_capture_attribution.py"
)
assert MODULE_PATH.is_file(), "segmented capture attribution contract is missing"
SPEC = importlib.util.spec_from_file_location(
    "segmented_capture_attribution_under_test",
    MODULE_PATH,
)
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)

CapturePhaseAccounting = module.CapturePhaseAccounting
AttributionDiagnosis = module.AttributionDiagnosis
ScratchCheckpointRecord = module.ScratchCheckpointRecord
ScratchDiffSummary = module.ScratchDiffSummary
ScratchTensorDigest = module.ScratchTensorDigest
aggregate_tp4_phase_rows = module.aggregate_tp4_phase_rows
canonical_sha256 = module.canonical_sha256
classify_phase_a1 = module.classify_phase_a1
validate_checkpoint_sequence = module.validate_checkpoint_sequence


def valid_phases() -> CapturePhaseAccounting:
    return CapturePhaseAccounting(
        snapshot_and_prepare_ns=10,
        graph_object_create_ns=11,
        capture_context_enter_ns=12,
        capture_body_ns=13,
        capture_context_exit_and_instantiate_ns=14,
        post_capture_synchronize_ns=15,
        post_capture_restore_ns=16,
        graph_reset_ns=17,
        segment_total_ns=100,
        program_lifecycle_ns=400,
    )


def test_phase_accounting_requires_every_non_negative_interval():
    assert valid_phases().measured_capture_ns == 65
    with pytest.raises(ValueError, match="non-negative"):
        replace(valid_phases(), capture_body_ns=-1)
    with pytest.raises(ValueError, match="segment_total"):
        replace(valid_phases(), segment_total_ns=64)
    with pytest.raises(ValueError, match="program_lifecycle"):
        replace(valid_phases(), program_lifecycle_ns=99)


def test_phase_accounting_rejects_boolean_durations():
    with pytest.raises(ValueError, match="non-negative"):
        replace(valid_phases(), graph_object_create_ns=True)


def digest(selector="key", sha256="a" * 64) -> ScratchTensorDigest:
    return ScratchTensorDigest(
        selector=selector,
        dtype="torch.bfloat16",
        shape=(64, 8, 1, 4, 128),
        byte_count=1_048_576,
        sha256=sha256,
    )


def exact_diff() -> ScratchDiffSummary:
    return ScratchDiffSummary(
        equal_to_s0=True,
        mismatching_element_count=0,
        first_mismatch=None,
        max_absolute_difference=0.0,
    )


def mismatch_diff() -> ScratchDiffSummary:
    return ScratchDiffSummary(
        equal_to_s0=False,
        mismatching_element_count=3,
        first_mismatch={
            "layer": 17,
            "scratch_slot_ordinal": 3,
            "head": 2,
            "element_offset": 11,
        },
        max_absolute_difference=0.5,
    )


def checkpoint(
    name,
    *,
    rank=0,
    segment_ordinal=None,
    key_diff=None,
    value_diff=None,
) -> ScratchCheckpointRecord:
    return ScratchCheckpointRecord(
        checkpoint=name,
        rank=rank,
        synchronized=True,
        keys=digest(),
        values=digest("value", "b" * 64),
        key_diff=exact_diff() if key_diff is None else key_diff,
        value_diff=exact_diff() if value_diff is None else value_diff,
        segment_ordinal=segment_ordinal,
    )


def test_scratch_checkpoint_sequence_is_exact_and_ordered():
    records = (
        checkpoint("S0"),
        checkpoint("S1"),
        checkpoint("S2"),
        checkpoint("S3", segment_ordinal=0),
        checkpoint("S3", segment_ordinal=1),
        checkpoint("S4"),
        checkpoint("S5"),
        checkpoint("S6"),
        checkpoint("S7"),
    )
    validate_checkpoint_sequence(records)

    with pytest.raises(ValueError, match="checkpoint order"):
        validate_checkpoint_sequence(records[:-1])
    with pytest.raises(ValueError, match="checkpoint order"):
        validate_checkpoint_sequence(records[:5] + records[6:])
    with pytest.raises(ValueError, match="segment ordinal"):
        validate_checkpoint_sequence(
            records[:3]
            + (checkpoint("S3", segment_ordinal=1),)
            + records[5:]
        )


def test_scratch_records_reject_unbounded_or_inconsistent_diffs():
    assert mismatch_diff().first_mismatch == {
        "layer": 17,
        "scratch_slot_ordinal": 3,
        "head": 2,
        "element_offset": 11,
    }
    with pytest.raises(ValueError, match="exact scratch diff"):
        replace(exact_diff(), mismatching_element_count=1)
    with pytest.raises(ValueError, match="first mismatch"):
        replace(mismatch_diff(), first_mismatch=None)
    with pytest.raises(ValueError, match="selector"):
        replace(digest(), selector="tensor")
    with pytest.raises(ValueError, match="sha256"):
        replace(digest(), sha256="A" * 64)
    with pytest.raises(ValueError, match="S3"):
        checkpoint("S3")
    with pytest.raises(ValueError, match="non-S3"):
        checkpoint("S4", segment_ordinal=0)


def phase_row(rank, body, total):
    phases = valid_phases()
    return {
        "rank": rank,
        "control_id": "isolated_0_16",
        "segment_ordinal": 0,
        "start_layer": 0,
        "end_layer": 16,
        **{
            name: getattr(phases, name)
            for name in module.CAPTURE_PHASE_NAMES
        },
        "capture_body_ns": body,
        "segment_total_ns": total,
        "program_lifecycle_ns": total + 100,
    }


def test_tp4_aggregation_uses_maximum_not_average():
    rows = [
        phase_row(0, 100, 200),
        phase_row(1, 110, 210),
        phase_row(2, 120, 220),
        phase_row(3, 900, 1_000),
    ]
    aggregate = aggregate_tp4_phase_rows(rows)
    assert aggregate["capture_body_ns"] == 900
    assert aggregate["segment_total_ns"] == 1_000
    assert aggregate["ranks"] == [0, 1, 2, 3]


def test_tp4_aggregation_rejects_missing_or_disagreeing_ranks():
    rows = [
        phase_row(0, 100, 200),
        phase_row(1, 110, 210),
        phase_row(2, 120, 220),
    ]
    with pytest.raises(ValueError, match="ranks"):
        aggregate_tp4_phase_rows(rows)

    rows.append(
        {
            **phase_row(3, 130, 230),
            "control_id": "isolated_16_32",
        }
    )
    with pytest.raises(ValueError, match="identity"):
        aggregate_tp4_phase_rows(rows)


def diagnosis(**changes) -> AttributionDiagnosis:
    values = {
        "first_scratch_divergence": "S4",
        "slow_capture_phase": "capture_context_exit_and_instantiate_ns",
        "root_cause_kind": "capture_restore_order",
        "source_path": "tools/tp4_segmented_capture_attribution_worker.py",
        "source_symbol": "_AttributionCudaBackend.capture_segment",
        "repair_statement": "restore scratch after capture synchronization",
        "repair_count": 1,
        "projected_max_segment_ns": 1_700_000_000,
        "projected_lifecycle_ns": 4_400_000_000,
        "projected_graph_count": 4,
    }
    values.update(changes)
    return AttributionDiagnosis(**values)


def valid_evidence(**changes):
    evidence = {
        "complete": True,
        "source_bound": True,
        "rank_agreement": True,
        "verifier_agreement": True,
        "cleanup": "CLEAN",
        "restore_round_trip_exact": True,
        "timing_evidence_eligible": True,
        "memory_gate_pass": True,
        "diagnosis": diagnosis(),
    }
    evidence.update(changes)
    return evidence


def test_phase_a1_classifier_selects_one_bounded_repair_candidate():
    result = classify_phase_a1(valid_evidence())
    assert result["classification"] == "REPAIR_CANDIDATE"
    assert result["failed_gates"] == []
    assert result["first_scratch_divergence"] == "S4"
    assert result["slow_capture_phase"] == (
        "capture_context_exit_and_instantiate_ns"
    )
    assert result["diagnosis_sha256"] == canonical_sha256(
        diagnosis().as_dict()
    )


@pytest.mark.parametrize(
    ("changes", "failed_gate"),
    [
        ({"complete": False}, "bundle_incomplete"),
        ({"source_bound": False}, "source_unbound"),
        ({"rank_agreement": False}, "rank_disagreement"),
        ({"verifier_agreement": False}, "verifier_disagreement"),
        ({"cleanup": "DIRTY"}, "cleanup_not_clean"),
    ],
)
def test_phase_a1_classifier_fails_incomplete_evidence_closed(
    changes,
    failed_gate,
):
    result = classify_phase_a1(valid_evidence(**changes))
    assert result["classification"] == "INCOMPLETE"
    assert failed_gate in result["failed_gates"]


@pytest.mark.parametrize(
    ("changes", "failed_gate"),
    [
        (
            {"restore_round_trip_exact": False},
            "scratch_restore_primitive",
        ),
        (
            {"timing_evidence_eligible": False},
            "timing_evidence_ineligible",
        ),
        ({"memory_gate_pass": False}, "memory_gate"),
        (
            {
                "diagnosis": diagnosis(
                    projected_max_segment_ns=1_800_000_001
                )
            },
            "projected_segment_ceiling",
        ),
        (
            {
                "diagnosis": diagnosis(
                    projected_lifecycle_ns=4_500_000_001
                )
            },
            "projected_lifecycle_ceiling",
        ),
        (
            {"diagnosis": diagnosis(projected_graph_count=5)},
            "projected_graph_count",
        ),
        (
            {"diagnosis": diagnosis(repair_count=2)},
            "single_repair_required",
        ),
    ],
)
def test_phase_a1_classifier_pivots_on_terminal_technical_gates(
    changes,
    failed_gate,
):
    result = classify_phase_a1(valid_evidence(**changes))
    assert (
        result["classification"]
        == "PIVOT_TP4_COMMUNICATION_COMPUTE_FUSION"
    )
    assert failed_gate in result["failed_gates"]


def test_phase_a1_classifier_rejects_phase_a2_go_state():
    with pytest.raises(ValueError, match="invalid for Phase A1"):
        classify_phase_a1(
            valid_evidence(requested_classification="GO_SEGMENTED_REPAIR")
        )


def test_phase_a1_classifier_handles_missing_diagnosis_as_incomplete():
    result = classify_phase_a1({})
    assert result["classification"] == "INCOMPLETE"
    assert "bundle_incomplete" in result["failed_gates"]
    assert "diagnosis_missing" in result["failed_gates"]
    assert result["first_scratch_divergence"] is None
    assert result["slow_capture_phase"] is None
    assert result["diagnosis_sha256"] is None


def test_diagnosis_requires_a_repository_relative_source_path():
    with pytest.raises(ValueError, match="source path"):
        diagnosis(source_path="/tmp/repair.py")
    with pytest.raises(ValueError, match="source path"):
        diagnosis(source_path="../repair.py")
