from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import PurePosixPath
import re


CAPTURE_PHASE_NAMES = (
    "snapshot_and_prepare_ns",
    "graph_object_create_ns",
    "capture_context_enter_ns",
    "capture_body_ns",
    "capture_context_exit_and_instantiate_ns",
    "post_capture_synchronize_ns",
    "post_capture_restore_ns",
    "graph_reset_ns",
)
SCRATCH_CHECKPOINTS = (
    "S0",
    "S1",
    "S2",
    "S3",
    "S4",
    "S5",
    "S6",
    "S7",
)
CONTROL_IDS = (
    "stitched_p4_repeat_0",
    "stitched_p4_repeat_1",
    "isolated_0_16",
    "isolated_16_32",
    "isolated_32_48",
    "isolated_48_64",
    "pool_fastest_shared",
    "pool_fastest_isolated",
    "pool_slowest_shared",
    "pool_slowest_isolated",
)


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class CapturePhaseAccounting:
    snapshot_and_prepare_ns: int
    graph_object_create_ns: int
    capture_context_enter_ns: int
    capture_body_ns: int
    capture_context_exit_and_instantiate_ns: int
    post_capture_synchronize_ns: int
    post_capture_restore_ns: int
    graph_reset_ns: int
    segment_total_ns: int
    program_lifecycle_ns: int

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in asdict(self).values()
        ):
            raise ValueError(
                "phase durations must be non-negative integers"
            )
        if self.segment_total_ns < self.measured_capture_ns:
            raise ValueError(
                "segment_total_ns is below measured capture"
            )
        if self.program_lifecycle_ns < self.segment_total_ns:
            raise ValueError(
                "program_lifecycle_ns is below segment_total_ns"
            )

    @property
    def measured_capture_ns(self) -> int:
        return sum(
            getattr(self, name)
            for name in CAPTURE_PHASE_NAMES[1:6]
        )


@dataclass(frozen=True)
class ScratchTensorDigest:
    selector: str
    dtype: str
    shape: tuple[int, ...]
    byte_count: int
    sha256: str

    def __post_init__(self) -> None:
        if self.selector not in {"key", "value"}:
            raise ValueError("scratch selector must be key or value")
        if not isinstance(self.dtype, str) or not self.dtype:
            raise ValueError("scratch dtype must be non-empty")
        if (
            not isinstance(self.shape, tuple)
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in self.shape
            )
        ):
            raise ValueError("scratch shape is invalid")
        if (
            isinstance(self.byte_count, bool)
            or not isinstance(self.byte_count, int)
            or self.byte_count < 0
        ):
            raise ValueError("scratch byte_count is invalid")
        if not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ValueError("scratch sha256 is invalid")


@dataclass(frozen=True)
class ScratchDiffSummary:
    equal_to_s0: bool
    mismatching_element_count: int
    first_mismatch: dict[str, int] | None
    max_absolute_difference: float

    def __post_init__(self) -> None:
        if not isinstance(self.equal_to_s0, bool):
            raise ValueError("equal_to_s0 must be a bool")
        if (
            isinstance(self.mismatching_element_count, bool)
            or not isinstance(self.mismatching_element_count, int)
            or self.mismatching_element_count < 0
        ):
            raise ValueError("scratch mismatch count is invalid")
        if (
            isinstance(self.max_absolute_difference, bool)
            or not isinstance(self.max_absolute_difference, (int, float))
            or not math.isfinite(self.max_absolute_difference)
            or self.max_absolute_difference < 0
        ):
            raise ValueError("scratch maximum difference is invalid")
        if self.equal_to_s0:
            if (
                self.mismatching_element_count != 0
                or self.first_mismatch is not None
                or self.max_absolute_difference != 0
            ):
                raise ValueError("exact scratch diff is inconsistent")
            return
        required = {
            "layer",
            "scratch_slot_ordinal",
            "head",
            "element_offset",
        }
        if (
            self.mismatching_element_count == 0
            or not isinstance(self.first_mismatch, dict)
            or set(self.first_mismatch) != required
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in self.first_mismatch.values()
            )
        ):
            raise ValueError("first mismatch location is invalid")


@dataclass(frozen=True)
class ScratchCheckpointRecord:
    checkpoint: str
    rank: int
    synchronized: bool
    keys: ScratchTensorDigest
    values: ScratchTensorDigest
    key_diff: ScratchDiffSummary
    value_diff: ScratchDiffSummary
    segment_ordinal: int | None = None

    def __post_init__(self) -> None:
        if self.checkpoint not in SCRATCH_CHECKPOINTS:
            raise ValueError("scratch checkpoint is invalid")
        if (
            isinstance(self.rank, bool)
            or not isinstance(self.rank, int)
            or self.rank < 0
        ):
            raise ValueError("scratch checkpoint rank is invalid")
        if self.synchronized is not True:
            raise ValueError("scratch checkpoint must be synchronized")
        if (
            not isinstance(self.keys, ScratchTensorDigest)
            or self.keys.selector != "key"
            or not isinstance(self.values, ScratchTensorDigest)
            or self.values.selector != "value"
            or not isinstance(self.key_diff, ScratchDiffSummary)
            or not isinstance(self.value_diff, ScratchDiffSummary)
        ):
            raise ValueError("scratch checkpoint payload is invalid")
        if self.checkpoint == "S3":
            if (
                isinstance(self.segment_ordinal, bool)
                or not isinstance(self.segment_ordinal, int)
                or self.segment_ordinal < 0
            ):
                raise ValueError("S3 requires a segment ordinal")
        elif self.segment_ordinal is not None:
            raise ValueError("non-S3 checkpoint forbids segment ordinal")


def validate_checkpoint_sequence(
    records: tuple[ScratchCheckpointRecord, ...],
) -> None:
    if (
        not isinstance(records, tuple)
        or len(records) < len(SCRATCH_CHECKPOINTS)
        or any(
            not isinstance(record, ScratchCheckpointRecord)
            for record in records
        )
    ):
        raise ValueError("scratch checkpoint order is invalid")
    ranks = {record.rank for record in records}
    if len(ranks) != 1:
        raise ValueError("scratch checkpoint ranks disagree")
    if tuple(record.checkpoint for record in records[:3]) != (
        "S0",
        "S1",
        "S2",
    ):
        raise ValueError("scratch checkpoint order is invalid")
    suffix_start = len(records) - 4
    if tuple(record.checkpoint for record in records[suffix_start:]) != (
        "S4",
        "S5",
        "S6",
        "S7",
    ):
        raise ValueError("scratch checkpoint order is invalid")
    segment_records = records[3:suffix_start]
    if (
        not segment_records
        or any(record.checkpoint != "S3" for record in segment_records)
    ):
        raise ValueError("scratch checkpoint order is invalid")
    ordinals = tuple(
        record.segment_ordinal for record in segment_records
    )
    if ordinals != tuple(range(len(segment_records))):
        raise ValueError("scratch segment ordinal is invalid")


def aggregate_tp4_phase_rows(rows: list[dict]) -> dict:
    if not isinstance(rows, list) or len(rows) != 4:
        raise ValueError("phase rows must contain TP4 ranks")
    ranks = sorted(row.get("rank") for row in rows)
    if ranks != [0, 1, 2, 3]:
        raise ValueError("phase rows must contain TP4 ranks")
    identity_names = (
        "control_id",
        "segment_ordinal",
        "start_layer",
        "end_layer",
    )
    identities = {
        tuple(row.get(name) for name in identity_names)
        for row in rows
    }
    if len(identities) != 1:
        raise ValueError("phase row identity disagrees across ranks")
    duration_names = CAPTURE_PHASE_NAMES + (
        "segment_total_ns",
        "program_lifecycle_ns",
    )
    for row in rows:
        CapturePhaseAccounting(
            **{name: row.get(name) for name in duration_names}
        )
    aggregate = {
        name: max(row[name] for row in rows)
        for name in duration_names
    }
    aggregate.update(
        dict(zip(identity_names, next(iter(identities)), strict=True))
    )
    aggregate["ranks"] = ranks
    return aggregate


@dataclass(frozen=True)
class AttributionDiagnosis:
    first_scratch_divergence: str
    slow_capture_phase: str
    root_cause_kind: str
    source_path: str
    source_symbol: str
    repair_statement: str
    repair_count: int
    projected_max_segment_ns: int
    projected_lifecycle_ns: int
    projected_graph_count: int

    def __post_init__(self) -> None:
        string_names = (
            "first_scratch_divergence",
            "slow_capture_phase",
            "root_cause_kind",
            "source_path",
            "source_symbol",
            "repair_statement",
        )
        if any(
            not isinstance(getattr(self, name), str)
            or not getattr(self, name)
            for name in string_names
        ):
            raise ValueError("attribution diagnosis strings are invalid")
        source_path = PurePosixPath(self.source_path)
        if source_path.is_absolute() or ".." in source_path.parts:
            raise ValueError(
                "attribution diagnosis source path must be relative"
            )
        integer_names = (
            "repair_count",
            "projected_max_segment_ns",
            "projected_lifecycle_ns",
            "projected_graph_count",
        )
        if any(
            isinstance(getattr(self, name), bool)
            or not isinstance(getattr(self, name), int)
            or getattr(self, name) < 0
            for name in integer_names
        ):
            raise ValueError("attribution diagnosis integers are invalid")
        if self.first_scratch_divergence not in SCRATCH_CHECKPOINTS:
            raise ValueError("first scratch divergence is invalid")
        if self.slow_capture_phase not in CAPTURE_PHASE_NAMES:
            raise ValueError("slow capture phase is invalid")

    def as_dict(self) -> dict:
        return asdict(self)


def _coerce_diagnosis(value: object) -> AttributionDiagnosis:
    if isinstance(value, AttributionDiagnosis):
        return value
    if isinstance(value, dict):
        return AttributionDiagnosis(**value)
    raise ValueError("attribution diagnosis is missing")


def classify_phase_a1(evidence: dict) -> dict:
    if not isinstance(evidence, dict):
        raise ValueError("Phase A1 evidence must be a mapping")
    if evidence.get("requested_classification") == "GO_SEGMENTED_REPAIR":
        raise ValueError("GO_SEGMENTED_REPAIR is invalid for Phase A1")

    incomplete_checks = (
        ("complete", True, "bundle_incomplete"),
        ("source_bound", True, "source_unbound"),
        ("rank_agreement", True, "rank_disagreement"),
        ("verifier_agreement", True, "verifier_disagreement"),
        ("cleanup", "CLEAN", "cleanup_not_clean"),
    )
    incomplete = [
        reason
        for name, expected, reason in incomplete_checks
        if evidence.get(name) != expected
    ]
    raw_diagnosis = evidence.get("diagnosis")
    diagnosis = (
        None
        if raw_diagnosis is None
        else _coerce_diagnosis(raw_diagnosis)
    )
    if diagnosis is None:
        incomplete.append("diagnosis_missing")
    if incomplete:
        classification = "INCOMPLETE"
        failed_gates = sorted(incomplete)
    else:
        technical = []
        if evidence.get("restore_round_trip_exact") is not True:
            technical.append("scratch_restore_primitive")
        if evidence.get("timing_evidence_eligible") is not True:
            technical.append("timing_evidence_ineligible")
        if evidence.get("memory_gate_pass") is not True:
            technical.append("memory_gate")
        if diagnosis.repair_count != 1:
            technical.append("single_repair_required")
        if diagnosis.projected_max_segment_ns > 1_800_000_000:
            technical.append("projected_segment_ceiling")
        if diagnosis.projected_lifecycle_ns > 4_500_000_000:
            technical.append("projected_lifecycle_ceiling")
        if diagnosis.projected_graph_count > 4:
            technical.append("projected_graph_count")
        if technical:
            classification = (
                "PIVOT_TP4_COMMUNICATION_COMPUTE_FUSION"
            )
            failed_gates = sorted(technical)
        else:
            classification = "REPAIR_CANDIDATE"
            failed_gates = []

    return {
        "schema_version": (
            "tinyllmforge.tp4-segmented-attribution-decision.v1"
        ),
        "phase": "A1",
        "classification": classification,
        "failed_gates": failed_gates,
        "first_scratch_divergence": (
            None
            if diagnosis is None
            else diagnosis.first_scratch_divergence
        ),
        "slow_capture_phase": (
            None if diagnosis is None else diagnosis.slow_capture_phase
        ),
        "diagnosis_sha256": (
            None
            if diagnosis is None
            else canonical_sha256(diagnosis.as_dict())
        ),
    }
