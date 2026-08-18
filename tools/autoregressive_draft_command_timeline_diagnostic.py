from __future__ import annotations

import copy
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
from pathlib import PurePosixPath
import statistics


SCHEMA_VERSION = 1
BLOCK_SCHEDULE = (
    ("eager", "graph"),
    ("graph", "eager"),
    ("graph", "eager"),
    ("eager", "graph"),
)
MEASURED_RUNS_PER_EPOCH = 5
MEASURED_RUNS_TOTAL = 40
BATCH_SIZE = 4
MAX_PROPOSAL_TOKENS = 4
PROMPT_TOKENS = 256
OUTPUT_TOKENS = 16
TEMPERATURE = 0.0
ROBUST_DISPERSION_LIMIT = 0.10
HALF_DRIFT_LIMIT = 0.15
ABSOLUTE_CONSERVATION_NS = 2_000_000
RELATIVE_CONSERVATION_LIMIT = 0.01
BOUNDARY_EXPLANATION_THRESHOLD = 0.60
BOUNDARY_BLOCK_COUNT = 3
UNEXPLAINED_E2E_LIMIT = 0.10
BOUNDARY_EXPLANATION_RATIO = Fraction(
    str(BOUNDARY_EXPLANATION_THRESHOLD)
)
UNEXPLAINED_E2E_RATIO = Fraction(str(UNEXPLAINED_E2E_LIMIT))
MAX_COLLECTION_ITEMS = 100_000
MAX_STRING_BYTES = 16_384
MAX_NESTING_DEPTH = 24
MAX_INTEGER = (1 << 63) - 1
BOUNDARY_NAMES = (
    "worker_queue_debt",
    "worker_cuda_execution",
    "ack_wait",
    "scheduler_postprocess",
)
ENGINE_STEP_PHASES = (
    "scheduler_schedule",
    "partition_and_step_setup",
    "ordinary_or_first_target_dispatch",
    "speculative_prepare",
    "scheduler_prepare_postprocess",
    "proposal_kv_prepare_commit",
    "proposal_lifecycle_finalize_prepare",
    "scheduler_commit_postprocess",
    "proposal_lifecycle_finalize_commit",
    "side_state_seal",
    "residency_precommit_or_seal",
    "ordinary_scheduler_postprocess",
)
SCHEDULER_POSTPROCESS_PHASES = frozenset({
    "scheduler_schedule",
    "partition_and_step_setup",
    "scheduler_prepare_postprocess",
    "scheduler_commit_postprocess",
    "ordinary_scheduler_postprocess",
})
EXACT_CONFIGURATION = {
    "tensor_parallel_size": 4,
    "batch_size": BATCH_SIZE,
    "max_proposal_tokens": MAX_PROPOSAL_TOKENS,
    "prompt_tokens": PROMPT_TOKENS,
    "output_tokens": OUTPUT_TOKENS,
    "temperature": TEMPERATURE,
    "proposal_kv_allocator": "direct",
    "proposal_kv_offload": False,
    "measured_runs_per_epoch": MEASURED_RUNS_PER_EPOCH,
    "measured_runs_total": MEASURED_RUNS_TOTAL,
}
TOP_LEVEL_KEYS = (
    "schema_version",
    "schedule",
    "configuration",
    "provenance",
    "raw_input_files",
    "source_files",
    "epochs",
    "blocks",
    "admission",
    "effects",
    "classification",
    "localized_boundary",
    "stable_but_unlocalized",
    "runtime_optimization_authorized",
    "performance_improvement_established",
    "phase_1_complete",
    "promotion_ready",
)


def _validate_bounded_json(
    value: object,
    *,
    name: str,
    depth: int = 0,
) -> None:
    if depth > MAX_NESTING_DEPTH:
        raise ValueError(f"{name} exceeds maximum nesting depth")
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, int):
        if abs(value) > MAX_INTEGER:
            raise ValueError(f"{name} contains an oversized integer")
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} contains a non-finite number")
        return
    if isinstance(value, str):
        if len(value.encode("utf-8")) > MAX_STRING_BYTES:
            raise ValueError(f"{name} contains an oversized string")
        return
    if isinstance(value, list):
        if len(value) > MAX_COLLECTION_ITEMS:
            raise ValueError(f"{name} contains an oversized list")
        for item in value:
            _validate_bounded_json(
                item,
                name=name,
                depth=depth + 1,
            )
        return
    if isinstance(value, dict):
        if len(value) > MAX_COLLECTION_ITEMS:
            raise ValueError(f"{name} contains an oversized mapping")
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{name} contains an invalid key")
            _validate_bounded_json(
                key,
                name=name,
                depth=depth + 1,
            )
            _validate_bounded_json(
                item,
                name=name,
                depth=depth + 1,
            )
        return
    raise ValueError(f"{name} contains a non-JSON value")


def canonical_json_bytes(value: object) -> bytes:
    _validate_bounded_json(value, name="canonical JSON")
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _compact_json_sha256(value: object) -> str:
    _validate_bounded_json(value, name="digest input")
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _mapping(value: object, name: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _list(value: object, name: str) -> list:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    if len(value) > MAX_COLLECTION_ITEMS:
        raise ValueError(f"{name} is oversized")
    return value


def _strict_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _integer(
    value: object,
    name: str,
    *,
    minimum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if abs(value) > MAX_INTEGER:
        raise ValueError(f"{name} exceeds the bounded integer range")
    return value


def _number(
    value: object,
    name: str,
    *,
    minimum: float | None = None,
    positive: bool = False,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{name} must be a finite number")
    normalized = float(value)
    if positive and normalized <= 0.0:
        raise ValueError(f"{name} must be positive")
    if minimum is not None and normalized < minimum:
        raise ValueError(f"{name} is below its minimum")
    return normalized


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be non-empty text")
    if len(value.encode("utf-8")) > MAX_STRING_BYTES:
        raise ValueError(f"{name} is oversized")
    return value


def _sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _source_commit(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) not in (40, 64)
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(
            "source commit must be lowercase hexadecimal"
        )
    return value


def _safe_relative_path(value: object, name: str) -> str:
    text = _text(value, name)
    path = PurePosixPath(text)
    if path.is_absolute() or "." in path.parts or ".." in path.parts:
        raise ValueError(f"{name} must be a safe relative path")
    return path.as_posix()


def _duration(start: object, finish: object, name: str) -> int:
    normalized_start = _integer(start, f"{name} start", minimum=0)
    normalized_finish = _integer(finish, f"{name} finish", minimum=0)
    if normalized_finish < normalized_start:
        raise ValueError(f"{name} duration is negative")
    return normalized_finish - normalized_start


def _fraction(
    value: object,
    name: str,
    *,
    minimum: int | None = None,
) -> Fraction:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an exact rational")
    if isinstance(value, int):
        normalized = Fraction(_integer(value, name), 1)
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        normalized = Fraction(str(value))
    elif isinstance(value, dict):
        if tuple(value) != ("numerator", "denominator"):
            raise ValueError(
                f"{name} rational representation is not canonical"
            )
        numerator = _integer(value["numerator"], f"{name} numerator")
        denominator = _integer(
            value["denominator"],
            f"{name} denominator",
            minimum=1,
        )
        normalized = Fraction(numerator, denominator)
        if (
            normalized.numerator != numerator
            or normalized.denominator != denominator
        ):
            raise ValueError(
                f"{name} rational representation is not reduced"
            )
    else:
        raise ValueError(f"{name} must be an exact rational")
    if minimum is not None and normalized < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return normalized


def _canonical_fraction(value: Fraction) -> int | dict:
    if value.denominator == 1:
        return _integer(value.numerator, "exact integer")
    return {
        "numerator": _integer(
            value.numerator,
            "exact rational numerator",
        ),
        "denominator": _integer(
            value.denominator,
            "exact rational denominator",
            minimum=1,
        ),
    }


def _median(values: list[int], name: str) -> int | dict:
    if not values:
        raise ValueError(f"{name} must not be empty")
    normalized = sorted(
        _fraction(value, name, minimum=0) for value in values
    )
    return _canonical_fraction(statistics.median(normalized))


def _sign(value: int | float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


@dataclass(frozen=True)
class EpochIdentity:
    block_index: int
    order: str
    label: str
    position: str
    epoch_index: int

    def __post_init__(self) -> None:
        _integer(self.block_index, "block index", minimum=0)
        _text(self.order, "epoch order")
        _text(self.label, "epoch label")
        _text(self.position, "epoch position")
        _integer(self.epoch_index, "epoch index", minimum=0)

    @property
    def key(self) -> str:
        return (
            f"b{self.block_index}-{self.label}-{self.position}"
        )

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "block_index": self.block_index,
            "order": self.order,
            "label": self.label,
            "position": self.position,
            "epoch_index": self.epoch_index,
        }


def expected_epoch_identities() -> tuple[EpochIdentity, ...]:
    identities = []
    epoch_index = 0
    for block_index, labels in enumerate(BLOCK_SCHEDULE):
        order = "_".join(labels)
        for position, label in zip(("first", "second"), labels):
            identities.append(EpochIdentity(
                block_index=block_index,
                order=order,
                label=label,
                position=position,
                epoch_index=epoch_index,
            ))
            epoch_index += 1
    return tuple(identities)


def _require_expected_identity(identity: EpochIdentity) -> None:
    if not isinstance(identity, EpochIdentity):
        raise ValueError("epoch identity must be EpochIdentity")
    expected = expected_epoch_identities()
    if (
        identity.epoch_index >= len(expected)
        or expected[identity.epoch_index] != identity
    ):
        raise ValueError("epoch identity is not in the fixed schedule")


def _normalize_prompt_rows(worker: dict) -> tuple[list[dict], str]:
    rows = _list(worker.get("prompt_rows"), "prompt rows")
    if len(rows) != BATCH_SIZE:
        raise ValueError("prompt row count must equal batch size four")
    normalized = []
    token_matrix = []
    for index, raw in enumerate(rows):
        row = _mapping(raw, f"prompt row {index}")
        if _integer(
            row.get("prompt_index"),
            "prompt index",
            minimum=0,
        ) != index:
            raise ValueError("prompt order is invalid")
        token_ids = _list(row.get("token_ids"), "prompt token IDs")
        if len(token_ids) != PROMPT_TOKENS:
            raise ValueError(
                "prompt token length must be exactly 256"
            )
        normalized_tokens = [
            _integer(token, "prompt token", minimum=0)
            for token in token_ids
        ]
        if _integer(
            row.get("token_count"),
            "prompt token count",
            minimum=0,
        ) != PROMPT_TOKENS:
            raise ValueError("prompt token count mismatch")
        digest = _sha256(row.get("sha256"), "prompt row digest")
        expected_digest = _compact_json_sha256(normalized_tokens)
        if digest != expected_digest:
            raise ValueError("prompt row digest mismatch")
        normalized.append({
            "prompt_index": index,
            "token_ids": normalized_tokens,
            "token_count": PROMPT_TOKENS,
            "sha256": digest,
        })
        token_matrix.append(normalized_tokens)
    prompt_sha256 = _sha256(
        worker.get("prompt_sha256"),
        "prompt digest",
    )
    if prompt_sha256 != _compact_json_sha256(token_matrix):
        raise ValueError("prompt digest mismatch")
    return normalized, prompt_sha256


def _normalize_rank_rows(
    value: object,
    *,
    name: str,
) -> list[dict]:
    rows = _list(value, name)
    if len(rows) != 4:
        raise ValueError(f"{name} must contain four ranks")
    normalized = []
    for expected_rank, raw in enumerate(rows):
        row = _mapping(raw, f"{name} rank row")
        rank = _integer(row.get("rank"), f"{name} rank", minimum=0)
        if rank != expected_rank:
            raise ValueError(
                f"{name} rank inventory must be exactly 0..3"
            )
        normalized.append(copy.deepcopy(row))
    return normalized


def _normalize_graph_evidence(
    correctness: dict,
    *,
    graph: bool,
    warmup: bool,
    previous_replays: list[int] | None,
    expected_resources: list[dict] | None,
    expected_identities: list[dict] | None,
) -> tuple[list[int], list[dict], list[dict], list[dict]]:
    counters = _normalize_rank_rows(
        correctness.get("rank_graph_counters"),
        name="rank graph counters",
    )
    resources = _normalize_rank_rows(
        correctness.get("rank_graph_resources"),
        name="rank graph resources",
    )
    identities = _normalize_rank_rows(
        correctness.get("rank_graph_identities"),
        name="rank graph identities",
    )
    replay_counts = []
    normalized_counters = []
    normalized_resources = []
    normalized_identities = []
    for rank in range(4):
        counter = counters[rank]
        normalized_counter = {
            "rank": rank,
            "capture_attempts": _integer(
                counter.get("capture_attempts"),
                "graph capture attempts",
                minimum=0,
            ),
            "captures": _integer(
                counter.get("captures"),
                "graph captures",
                minimum=0,
            ),
            "replays": _integer(
                counter.get("replays"),
                "graph replays",
                minimum=0,
            ),
            "quarantines": _integer(
                counter.get("quarantines"),
                "graph quarantines",
                minimum=0,
            ),
            "fallback_pre_replay": _integer(
                counter.get("fallback_pre_replay"),
                "graph pre-replay fallback",
                minimum=0,
            ),
        }
        resource = resources[rank]
        normalized_resource = {
            "rank": rank,
            "ready_entry_count": _integer(
                resource.get("ready_entry_count"),
                "graph ready entries",
                minimum=0,
            ),
            "static_bytes": _integer(
                resource.get("static_bytes"),
                "graph static bytes",
                minimum=0,
            ),
            "reserved_bytes": _integer(
                resource.get("reserved_bytes"),
                "graph reserved bytes",
                minimum=0,
            ),
            "total_capture_ns": _integer(
                resource.get("total_capture_ns"),
                "graph capture time",
                minimum=0,
            ),
        }
        identity_row = identities[rank]
        graph_sha = identity_row.get("sha256")
        normalized_identity = {
            "rank": rank,
            "sha256": (
                _sha256(graph_sha, "graph identity")
                if graph
                else graph_sha
            ),
        }
        if graph:
            if (
                normalized_counter["capture_attempts"] != 1
                or normalized_counter["captures"] != 1
            ):
                raise ValueError(
                    "graph capture identity changed"
                )
            if (
                normalized_counter["quarantines"] != 0
                or normalized_counter["fallback_pre_replay"] != 0
            ):
                raise ValueError(
                    "graph replay failure counter is nonzero"
                )
            if (
                normalized_resource["ready_entry_count"] != 1
                or normalized_resource["static_bytes"] <= 0
                or normalized_resource["reserved_bytes"] <= 0
                or normalized_resource["total_capture_ns"] <= 0
            ):
                raise ValueError("graph resource identity is invalid")
            if previous_replays is not None and (
                normalized_counter["replays"]
                <= previous_replays[rank]
            ):
                raise ValueError(
                    "graph replay count did not increase"
                )
            if (
                expected_resources is not None
                and normalized_resource != expected_resources[rank]
            ):
                raise ValueError(
                    "graph resource identity drifted"
                )
            if (
                expected_identities is not None
                and normalized_identity != expected_identities[rank]
            ):
                raise ValueError("graph identity drifted")
            if warmup and normalized_counter["replays"] <= 0:
                raise ValueError("graph warmup replay is missing")
        else:
            if any(
                normalized_counter[name] != 0
                for name in (
                    "capture_attempts",
                    "captures",
                    "replays",
                    "quarantines",
                    "fallback_pre_replay",
                )
            ):
                raise ValueError(
                    "eager mode must not report graph capture or replay"
                )
            if any(
                normalized_resource[name] != 0
                for name in (
                    "ready_entry_count",
                    "static_bytes",
                    "reserved_bytes",
                    "total_capture_ns",
                )
            ):
                raise ValueError(
                    "eager mode must not report a ready graph entry"
                )
            if graph_sha is not None:
                raise ValueError(
                    "eager mode must not report a graph identity"
                )
        replay_counts.append(normalized_counter["replays"])
        normalized_counters.append(normalized_counter)
        normalized_resources.append(normalized_resource)
        normalized_identities.append(normalized_identity)
    return (
        replay_counts,
        normalized_counters,
        normalized_resources,
        normalized_identities,
    )


def _normalize_nested_token_rows(
    value: object,
    *,
    name: str,
    maximum_width: int | None = None,
) -> list[list[list[int]]]:
    calls = _list(value, name)
    if not calls:
        raise ValueError(f"{name} must not be empty")
    normalized_calls = []
    for call_index, raw_call in enumerate(calls):
        call = _list(raw_call, f"{name} call {call_index}")
        if len(call) > BATCH_SIZE:
            raise ValueError(f"{name} exceeds batch size B4")
        normalized_rows = []
        for row_index, raw_row in enumerate(call):
            row = _list(
                raw_row,
                f"{name} call {call_index} row {row_index}",
            )
            if not row:
                raise ValueError(f"{name} contains an empty row")
            if maximum_width is not None and len(row) > maximum_width:
                raise ValueError(
                    f"{name} exceeds exact Q{maximum_width}"
                )
            normalized_rows.append([
                _integer(token, f"{name} token", minimum=0)
                for token in row
            ])
        normalized_calls.append(normalized_rows)
    return normalized_calls


def _normalize_parity(
    run: dict,
    *,
    mode: str,
    warmup: bool,
    previous_replays: list[int] | None,
    expected_resources: list[dict] | None,
    expected_graph_identities: list[dict] | None,
) -> tuple[dict, list[int], list[dict], list[dict]]:
    correctness = _mapping(
        run.get("correctness"),
        "run correctness",
    )
    outputs = _list(run.get("outputs"), "run outputs")
    if len(outputs) != BATCH_SIZE:
        raise ValueError("output row count must equal batch size four")
    normalized_outputs = []
    for raw_row in outputs:
        row = _list(raw_row, "output token row")
        if len(row) != OUTPUT_TOKENS:
            raise ValueError(
                "output token length must be exactly 16"
            )
        normalized_outputs.append([
            _integer(token, "output token", minimum=0)
            for token in row
        ])
    target_rows = _list(
        correctness.get("target_token_rows"),
        "target token rows",
    )
    normalized_targets = []
    if len(target_rows) != BATCH_SIZE:
        raise ValueError("target token rows must have exact B4 shape")
    for raw_row in target_rows:
        row = _list(raw_row, "target token row")
        if len(row) != OUTPUT_TOKENS:
            raise ValueError(
                "target token rows must have exact output16 shape"
            )
        normalized_targets.append([
            _integer(token, "target token", minimum=0)
            for token in row
        ])
    if normalized_targets != normalized_outputs:
        raise ValueError("target token parity failed")

    proposal_rows = _normalize_nested_token_rows(
        correctness.get("proposal_token_rows"),
        name="proposal token rows",
        maximum_width=MAX_PROPOSAL_TOKENS,
    )
    raw_lengths = _list(
        correctness.get("proposal_row_lengths"),
        "proposal row lengths",
    )
    if len(raw_lengths) != len(proposal_rows):
        raise ValueError("proposal row length inventory mismatch")
    normalized_lengths = []
    for call_index, raw_call_lengths in enumerate(raw_lengths):
        call_lengths = _list(
            raw_call_lengths,
            "proposal call row lengths",
        )
        if len(call_lengths) != len(proposal_rows[call_index]):
            raise ValueError("proposal row length inventory mismatch")
        normalized_call_lengths = [
            _integer(length, "proposal logical row length", minimum=1)
            for length in call_lengths
        ]
        if normalized_call_lengths != [
            len(row) for row in proposal_rows[call_index]
        ]:
            raise ValueError(
                "proposal row contains padding beyond its logical length"
            )
        normalized_lengths.append(normalized_call_lengths)

    prefix_calls = _list(
        correctness.get("accepted_prefix_counts"),
        "accepted prefix counts",
    )
    accepted_rows = _normalize_nested_token_rows(
        correctness.get("accepted_token_rows"),
        name="accepted token rows",
        maximum_width=MAX_PROPOSAL_TOKENS,
    )
    if (
        len(prefix_calls) != len(proposal_rows)
        or len(accepted_rows) != len(proposal_rows)
    ):
        raise ValueError("accepted token call inventory mismatch")
    normalized_prefixes = []
    for call_index, raw_prefixes in enumerate(prefix_calls):
        prefixes = _list(raw_prefixes, "accepted prefix call")
        if (
            len(prefixes) != len(proposal_rows[call_index])
            or len(accepted_rows[call_index])
            != len(proposal_rows[call_index])
        ):
            raise ValueError("accepted prefix row inventory mismatch")
        normalized_call = []
        for row_index, raw_prefix in enumerate(prefixes):
            prefix = _integer(
                raw_prefix,
                "accepted prefix count",
                minimum=0,
            )
            proposal = proposal_rows[call_index][row_index]
            accepted = accepted_rows[call_index][row_index]
            if prefix > len(proposal):
                raise ValueError(
                    "accepted prefix exceeds proposal row"
                )
            if accepted != proposal[:prefix]:
                raise ValueError(
                        "accepted token parity does not match "
                        "accepted prefixes"
                )
            normalized_call.append(prefix)
        normalized_prefixes.append(normalized_call)

    proposed_count = sum(
        len(row) for call in proposal_rows for row in call
    )
    accepted_count = sum(
        prefix for call in normalized_prefixes for prefix in call
    )
    acceptance = _mapping(
        correctness.get("acceptance"),
        "acceptance",
    )
    normalized_acceptance = {
        "proposed_tokens": _integer(
            acceptance.get("proposed_tokens"),
            "acceptance proposed tokens",
            minimum=0,
        ),
        "accepted_tokens": _integer(
            acceptance.get("accepted_tokens"),
            "acceptance accepted tokens",
            minimum=0,
        ),
        "rate": _number(
            acceptance.get("rate"),
            "acceptance rate",
            minimum=0.0,
        ),
    }
    if (
        normalized_acceptance["proposed_tokens"] != proposed_count
        or normalized_acceptance["accepted_tokens"] != accepted_count
        or normalized_acceptance["rate"] > 1.0
    ):
        raise ValueError("acceptance counts are inconsistent")
    expected_rate = (
        accepted_count / proposed_count if proposed_count else 0.0
    )
    if not math.isclose(
        normalized_acceptance["rate"],
        expected_rate,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError("acceptance rate is inconsistent")

    active_transactions = _integer(
        correctness.get("active_transaction_count"),
        "active transaction count",
        minimum=0,
    )
    if active_transactions != 0:
        raise ValueError(
            "active transaction count must be zero after the batch"
        )
    transaction_digest = _sha256(
        correctness.get("transaction_digest"),
        "transaction digest",
    )
    (
        replay_counts,
        graph_counters,
        resources,
        graph_identities,
    ) = _normalize_graph_evidence(
        correctness,
        graph=mode == "graph",
        warmup=warmup,
        previous_replays=previous_replays,
        expected_resources=expected_resources,
        expected_identities=expected_graph_identities,
    )
    return (
        {
            "target_token_rows": normalized_targets,
            "proposal_token_rows": proposal_rows,
            "proposal_row_lengths": normalized_lengths,
            "accepted_prefix_counts": normalized_prefixes,
            "accepted_token_rows": accepted_rows,
            "transaction_digest": transaction_digest,
            "active_transaction_count": active_transactions,
            "acceptance": normalized_acceptance,
            "rank_graph_counters": graph_counters,
            "rank_graph_resources": resources,
            "rank_graph_identities": graph_identities,
        },
        replay_counts,
        resources,
        graph_identities,
    )


def _parity_semantic_view(parity: dict) -> dict:
    return {
        key: parity[key]
        for key in (
            "target_token_rows",
            "proposal_token_rows",
            "proposal_row_lengths",
            "accepted_prefix_counts",
            "accepted_token_rows",
            "transaction_digest",
            "active_transaction_count",
            "acceptance",
        )
    }


def _normalize_timing(run: dict) -> dict:
    timing = _mapping(run.get("timing"), "run timing")
    if _integer(
        timing.get("request_count"),
        "timing request count",
        minimum=0,
    ) != BATCH_SIZE:
        raise ValueError("timing request count mismatch")
    if _integer(
        timing.get("total_output_tokens"),
        "timing total output tokens",
        minimum=0,
    ) != BATCH_SIZE * OUTPUT_TOKENS:
        raise ValueError("timing total output tokens mismatch")
    batch_elapsed_ns = _integer(
        timing.get("batch_elapsed_ns"),
        "batch elapsed time",
        minimum=1,
    )
    per_request = _list(
        timing.get("per_request"),
        "per-request timing",
    )
    if len(per_request) != BATCH_SIZE:
        raise ValueError("per-request timing count mismatch")
    normalized = []
    for sequence_id, raw in enumerate(per_request):
        row = _mapping(raw, "per-request timing row")
        if _integer(
            row.get("sequence_id"),
            "timing sequence ID",
            minimum=0,
        ) != sequence_id:
            raise ValueError("request order mismatch")
        if _integer(
            row.get("output_tokens"),
            "request output tokens",
            minimum=0,
        ) != OUTPUT_TOKENS:
            raise ValueError("request output length mismatch")
        normalized.append({
            "sequence_id": sequence_id,
            "output_tokens": OUTPUT_TOKENS,
            "ttft_ns": _integer(
                row.get("ttft_ns"),
                "TTFT",
                minimum=1,
            ),
            "tpot_ns": _integer(
                row.get("tpot_ns"),
                "TPOT",
                minimum=1,
            ),
            "completion_latency_ns": _integer(
                row.get("completion_latency_ns"),
                "completion latency",
                minimum=1,
            ),
        })
    return {
        "request_count": BATCH_SIZE,
        "total_output_tokens": BATCH_SIZE * OUTPUT_TOKENS,
        "batch_elapsed_ns": batch_elapsed_ns,
        "per_request": normalized,
    }


def _normalize_campaign_interval(run: dict) -> dict:
    interval = _mapping(
        run.get("campaign_interval"),
        "campaign interval",
    )
    normalized = {
        "started_at_unix_ns": _integer(
            interval.get("started_at_unix_ns"),
            "campaign Unix start",
            minimum=0,
        ),
        "finished_at_unix_ns": _integer(
            interval.get("finished_at_unix_ns"),
            "campaign Unix finish",
            minimum=0,
        ),
        "started_at_monotonic_ns": _integer(
            interval.get("started_at_monotonic_ns"),
            "campaign monotonic start",
            minimum=0,
        ),
        "finished_at_monotonic_ns": _integer(
            interval.get("finished_at_monotonic_ns"),
            "campaign monotonic finish",
            minimum=0,
        ),
    }
    if (
        normalized["finished_at_unix_ns"]
        <= normalized["started_at_unix_ns"]
        or normalized["finished_at_monotonic_ns"]
        <= normalized["started_at_monotonic_ns"]
    ):
        raise ValueError("campaign interval is invalid")
    return normalized


def _validate_repeat_skeleton(
    run: object,
    *,
    expected_repeat: int,
) -> dict:
    normalized = _mapping(run, "run")
    if _integer(normalized.get("repeat"), "public repeat") != expected_repeat:
        raise ValueError("public repeat order is invalid")
    timeline_repeat = _integer(
        normalized.get("command_timeline_repeat_index"),
        "command timeline repeat identity",
        minimum=0,
    )
    return {
        "repeat": expected_repeat,
        "command_timeline_repeat_index": timeline_repeat,
        "campaign_interval": _normalize_campaign_interval(normalized),
        "outputs": copy.deepcopy(normalized.get("outputs")),
        "timing": _normalize_timing(normalized),
        "correctness": copy.deepcopy(normalized.get("correctness")),
        "runtime": copy.deepcopy(
            _mapping(normalized.get("runtime"), "run runtime")
        ),
        "telemetry": copy.deepcopy(
            _mapping(normalized.get("telemetry"), "run telemetry")
        ),
    }


def validate_epoch_worker(
    worker: object,
    identity: EpochIdentity,
) -> dict:
    _require_expected_identity(identity)
    raw = _mapping(worker, "epoch worker")
    if raw.get("policy") != "learned":
        raise ValueError("worker policy must be learned")
    if _integer(
        raw.get("tensor_parallel_size"),
        "tensor parallel size",
        minimum=1,
    ) != 4:
        raise ValueError("tensor parallel size must be four")
    if _integer(
        raw.get("batch_size"),
        "batch size",
        minimum=1,
    ) != BATCH_SIZE:
        raise ValueError("batch size must be four")
    if _integer(
        raw.get("max_proposal_tokens"),
        "proposal token limit",
        minimum=1,
    ) != MAX_PROPOSAL_TOKENS:
        raise ValueError("proposal token limit must be four")
    prompt_rows, prompt_sha256 = _normalize_prompt_rows(raw)
    if _integer(
        raw.get("requested_output_tokens"),
        "requested output tokens",
        minimum=1,
    ) != OUTPUT_TOKENS:
        raise ValueError("requested output length must be 16")
    request_order = [
        _integer(value, "request order identity", minimum=0)
        for value in _list(raw.get("request_order"), "request order")
    ]
    if request_order != list(range(BATCH_SIZE)):
        raise ValueError("request order must be exactly 0,1,2,3")
    temperature = _number(raw.get("temperature"), "temperature")
    if temperature != TEMPERATURE:
        raise ValueError("temperature must be exactly zero")
    if raw.get("proposal_kv_allocator") != "direct":
        raise ValueError("Proposal-KV allocator must be direct")
    if _strict_bool(
        raw.get("proposal_kv_offload"),
        "Proposal-KV offload",
    ):
        raise ValueError("Proposal-KV offload must be disabled")
    source_commit = _source_commit(raw.get("source_commit"))
    source_tree_sha256 = _sha256(
        raw.get("source_tree_sha256"),
        "source tree digest",
    )
    target_checkpoint = _text(
        raw.get("target_checkpoint_identifier"),
        "target checkpoint identifier",
    )
    draft_checkpoint = _text(
        raw.get("draft_checkpoint_identifier"),
        "draft checkpoint identifier",
    )
    tokenizer = _text(
        raw.get("tokenizer_identifier"),
        "tokenizer identifier",
    )
    gpu_uuids = [
        _text(value, "GPU UUID")
        for value in _list(raw.get("gpu_uuids"), "GPU UUIDs")
    ]
    if len(gpu_uuids) != 4 or len(set(gpu_uuids)) != 4:
        raise ValueError("GPU UUID inventory must contain four identities")
    mode = _text(raw.get("cuda_graph_mode"), "CUDA Graph mode")
    if mode != identity.label:
        raise ValueError(
            "CUDA Graph mode does not match epoch identity"
        )

    warmup_runs = _list(raw.get("warmup_runs"), "warmup runs")
    if len(warmup_runs) != 1:
        raise ValueError("worker requires exactly one warmup run")
    measured_runs = _list(raw.get("measured_runs"), "measured runs")
    if len(measured_runs) != MEASURED_RUNS_PER_EPOCH:
        raise ValueError(
            "worker requires exactly five measured repeats"
        )

    warmup = _validate_repeat_skeleton(
        warmup_runs[0],
        expected_repeat=-1,
    )
    (
        warmup_parity,
        previous_replays,
        graph_resources,
        graph_identities,
    ) = _normalize_parity(
        warmup,
        mode=mode,
        warmup=True,
        previous_replays=None,
        expected_resources=None,
        expected_graph_identities=None,
    )
    normalized_runs = []
    reference_parity = None
    for repeat_index, raw_run in enumerate(measured_runs):
        run = _validate_repeat_skeleton(
            raw_run,
            expected_repeat=repeat_index,
        )
        parity, previous_replays, _, _ = _normalize_parity(
            run,
            mode=mode,
            warmup=False,
            previous_replays=previous_replays,
            expected_resources=graph_resources,
            expected_graph_identities=graph_identities,
        )
        semantic_parity = _parity_semantic_view(parity)
        if reference_parity is None:
            reference_parity = semantic_parity
        elif semantic_parity != reference_parity:
            raise ValueError(
                "measured repeat parity or transaction evidence drifted"
            )
        if run["timing"]["per_request"] and (
            [row["sequence_id"] for row in run["timing"]["per_request"]]
            != request_order
        ):
            raise ValueError("request order mismatch")
        normalized_runs.append({
            **run,
            "outputs": copy.deepcopy(parity["target_token_rows"]),
            "correctness": parity,
        })
    return {
        "policy": "learned",
        "tensor_parallel_size": 4,
        "batch_size": BATCH_SIZE,
        "max_proposal_tokens": MAX_PROPOSAL_TOKENS,
        "prompt_rows": prompt_rows,
        "prompt_sha256": prompt_sha256,
        "requested_output_tokens": OUTPUT_TOKENS,
        "request_order": request_order,
        "temperature": TEMPERATURE,
        "proposal_kv_allocator": "direct",
        "proposal_kv_offload": False,
        "source_commit": source_commit,
        "source_tree_sha256": source_tree_sha256,
        "target_checkpoint_identifier": target_checkpoint,
        "draft_checkpoint_identifier": draft_checkpoint,
        "tokenizer_identifier": tokenizer,
        "gpu_uuids": gpu_uuids,
        "cuda_graph_mode": mode,
        "warmup_runs": [{
            **warmup,
            "outputs": copy.deepcopy(
                warmup_parity["target_token_rows"]
            ),
            "correctness": warmup_parity,
        }],
        "measured_runs": normalized_runs,
    }


def _clock_identity(value: object) -> dict:
    clock = _mapping(value, "clock metadata")
    return {
        "boot_id": _text(clock.get("boot_id"), "boot ID"),
        "implementation": _text(
            clock.get("implementation"),
            "clock implementation",
        ),
        "resolution_s": _number(
            clock.get("resolution_s"),
            "clock resolution",
            positive=True,
        ),
        "monotonic": _strict_bool(
            clock.get("monotonic"),
            "clock monotonic flag",
        ),
        "adjustable": _strict_bool(
            clock.get("adjustable"),
            "clock adjustable flag",
        ),
        "captured_at_unix_ns": _integer(
            clock.get("captured_at_unix_ns"),
            "clock capture Unix timestamp",
            minimum=0,
        ),
    }


def _command_identity(row: dict) -> dict:
    engine_step_id = row.get("engine_step_id")
    if engine_step_id is not None:
        engine_step_id = _integer(
            engine_step_id,
            "command engine step identity",
            minimum=0,
        )
    return {
        "command_id": _integer(
            row.get("command_id"),
            "command identity",
            minimum=0,
        ),
        "method_name": _text(
            row.get("method_name"),
            "command method",
        ),
        "requires_ack": _strict_bool(
            row.get("requires_ack"),
            "command requires_ack",
        ),
        "engine_step_id": engine_step_id,
        "repeat_index": _integer(
            row.get("repeat_index"),
            "command repeat identity",
            minimum=0,
        ),
        "request_set_sha256": _sha256(
            row.get("request_set_sha256"),
            "command request digest",
        ),
        "batch_kind": _text(
            row.get("batch_kind"),
            "command batch kind",
        ),
        "speculative_selected_sequence_ids_sha256": _sha256(
            row.get("speculative_selected_sequence_ids_sha256"),
            "command selected-sequence digest",
        ),
        "dispatch_started_monotonic_ns": _integer(
            row.get("dispatch_started_monotonic_ns"),
            "command dispatch start",
            minimum=0,
        ),
        "dispatch_published_monotonic_ns": _integer(
            row.get("dispatch_published_monotonic_ns"),
            "command dispatch publication",
            minimum=0,
        ),
    }


def _validate_timestamp_in_interval(
    value: object,
    *,
    name: str,
    interval: dict,
    optional: bool = False,
) -> int | None:
    if value is None and optional:
        return None
    normalized = _integer(value, name, minimum=0)
    if not (
        interval["started_at_monotonic_ns"]
        <= normalized
        <= interval["finished_at_monotonic_ns"]
    ):
        raise ValueError(f"{name} lies outside the campaign interval")
    return normalized


def _normalize_command_row(
    raw: object,
    *,
    rank: int,
    interval: dict,
    expected_repeat: int,
    prompt_sha256: str,
) -> dict:
    row = _mapping(raw, "command row")
    if _integer(row.get("rank"), "command rank", minimum=0) != rank:
        raise ValueError("command rank identity is invalid")
    identity = _command_identity(row)
    if identity["repeat_index"] != expected_repeat:
        raise ValueError("command repeat identity mismatch")
    if identity["request_set_sha256"] != prompt_sha256:
        raise ValueError("command request digest mismatch")
    for name in (
        "dispatch_started_monotonic_ns",
        "dispatch_published_monotonic_ns",
    ):
        _validate_timestamp_in_interval(
            identity[name],
            name=name,
            interval=interval,
        )
    if (
        identity["dispatch_published_monotonic_ns"]
        < identity["dispatch_started_monotonic_ns"]
    ):
        raise ValueError("command dispatch timestamps are invalid")
    status = row.get("status")
    if status != "ok":
        raise ValueError("command status must be ok")
    error_type = row.get("error_type")
    error_detail = row.get("error_detail")
    if error_type != "" or error_detail != "":
        raise ValueError("successful command error fields must be empty")
    timestamps = {}
    for name in (
        "event_woken_monotonic_ns",
        "envelope_read_monotonic_ns",
        "method_started_monotonic_ns",
        "method_finished_monotonic_ns",
        "local_method_started_monotonic_ns",
        "local_method_finished_monotonic_ns",
        "ack_send_started_monotonic_ns",
        "ack_send_finished_monotonic_ns",
        "ack_wait_started_monotonic_ns",
        "ack_wait_finished_monotonic_ns",
        "terminal_error_monotonic_ns",
    ):
        timestamps[name] = _validate_timestamp_in_interval(
            row.get(name),
            name=name,
            interval=interval,
            optional=True,
        )
    requires_ack = identity["requires_ack"]
    if rank == 0:
        if (
            timestamps["local_method_started_monotonic_ns"] is None
            or timestamps["local_method_finished_monotonic_ns"] is None
            or timestamps["method_started_monotonic_ns"] is not None
            or timestamps["method_finished_monotonic_ns"] is not None
            or timestamps["event_woken_monotonic_ns"] is not None
            or timestamps["envelope_read_monotonic_ns"] is not None
            or timestamps["ack_send_started_monotonic_ns"] is not None
            or timestamps["ack_send_finished_monotonic_ns"] is not None
        ):
            raise ValueError("rank-zero command timestamps are invalid")
        method_start = timestamps["local_method_started_monotonic_ns"]
        method_finish = timestamps["local_method_finished_monotonic_ns"]
        if requires_ack:
            if (
                timestamps["ack_wait_started_monotonic_ns"] is None
                or timestamps["ack_wait_finished_monotonic_ns"] is None
            ):
                raise ValueError(
                    "acknowledged command has missing ack wait"
                )
        elif (
            timestamps["ack_wait_started_monotonic_ns"] is not None
            or timestamps["ack_wait_finished_monotonic_ns"] is not None
        ):
            raise ValueError(
                "non-ack command must not have ack timestamps"
            )
    else:
        if (
            timestamps["event_woken_monotonic_ns"] is None
            or timestamps["envelope_read_monotonic_ns"] is None
            or timestamps["method_started_monotonic_ns"] is None
            or timestamps["method_finished_monotonic_ns"] is None
            or timestamps["local_method_started_monotonic_ns"] is not None
            or timestamps["local_method_finished_monotonic_ns"] is not None
            or timestamps["ack_wait_started_monotonic_ns"] is not None
            or timestamps["ack_wait_finished_monotonic_ns"] is not None
        ):
            raise ValueError("worker command timestamps are invalid")
        method_start = timestamps["method_started_monotonic_ns"]
        method_finish = timestamps["method_finished_monotonic_ns"]
        if requires_ack:
            if (
                timestamps["ack_send_started_monotonic_ns"] is None
                or timestamps["ack_send_finished_monotonic_ns"] is None
            ):
                raise ValueError(
                    "acknowledged worker command has missing ack send"
                )
        elif (
            timestamps["ack_send_started_monotonic_ns"] is not None
            or timestamps["ack_send_finished_monotonic_ns"] is not None
        ):
            raise ValueError(
                "non-ack command must not have ack timestamps"
            )
    if method_start < identity["dispatch_published_monotonic_ns"]:
        raise ValueError("worker queue duration is negative")
    method_wall_ns = _duration(
        method_start,
        method_finish,
        "worker method",
    )
    if rank > 0 and (
        timestamps["envelope_read_monotonic_ns"]
        < timestamps["event_woken_monotonic_ns"]
        or method_start < timestamps["envelope_read_monotonic_ns"]
    ):
        raise ValueError("worker receive timestamps are invalid")
    ack_wait_ns = None
    if rank == 0 and requires_ack:
        ack_wait_ns = _duration(
            timestamps["ack_wait_started_monotonic_ns"],
            timestamps["ack_wait_finished_monotonic_ns"],
            "ack wait",
        )
        if (
            timestamps["ack_wait_started_monotonic_ns"]
            < method_start
        ):
            raise ValueError("ack wait starts before local method")
    if rank > 0 and requires_ack:
        if timestamps["ack_send_started_monotonic_ns"] < method_finish:
            raise ValueError("ack send starts before method finish")
        _duration(
            timestamps["ack_send_started_monotonic_ns"],
            timestamps["ack_send_finished_monotonic_ns"],
            "ack send",
        )
    return {
        "rank": rank,
        **identity,
        **timestamps,
        "status": "ok",
        "error_type": "",
        "error_detail": "",
        "method_start_ns": method_start,
        "method_finish_ns": method_finish,
        "worker_method_wall_ns": method_wall_ns,
        "worker_queue_wait_ns": (
            method_start
            - identity["dispatch_published_monotonic_ns"]
        ),
        "ack_wait_ns": ack_wait_ns,
    }


def _normalize_command_snapshots(
    timeline: dict,
    *,
    interval: dict,
    repeat_identity: int,
    prompt_sha256: str,
) -> tuple[list[dict], dict[int, dict[int, dict]], dict]:
    snapshots = _normalize_rank_rows(
        timeline.get("rank_snapshots"),
        name="command rank snapshots",
    )
    clocks = []
    rows_by_rank = {}
    command_inventory = None
    flat_rows = []
    for rank, snapshot in enumerate(snapshots):
        if _integer(
            snapshot.get("schema_version"),
            "command rank snapshot schema version",
            minimum=0,
        ) != SCHEMA_VERSION:
            raise ValueError("command timeline schema version mismatch")
        if _strict_bool(
            snapshot.get("enabled"),
            "command timeline enabled",
        ) is not True:
            raise ValueError("command timeline must be enabled")
        if _integer(
            snapshot.get("dropped_rows"),
            "dropped command rows",
            minimum=0,
        ) != 0:
            raise ValueError("command timeline rows are missing")
        clock = _clock_identity(snapshot.get("clock"))
        if clock["monotonic"] is not True:
            raise ValueError("clock metadata is not monotonic")
        clocks.append(clock)
        raw_rows = _list(snapshot.get("rows"), "command rows")
        if not raw_rows:
            raise ValueError("command rows are missing")
        normalized_rows = [
            _normalize_command_row(
                raw_row,
                rank=rank,
                interval=interval,
                expected_repeat=repeat_identity,
                prompt_sha256=prompt_sha256,
            )
            for raw_row in raw_rows
        ]
        command_ids = [row["command_id"] for row in normalized_rows]
        if len(command_ids) != len(set(command_ids)):
            raise ValueError("duplicate command IDs are invalid")
        if command_ids != sorted(command_ids):
            raise ValueError("command ID order is invalid")
        if any(
            current != previous + 1
            for previous, current in zip(
                command_ids,
                command_ids[1:],
            )
        ):
            raise ValueError("missing command IDs are invalid")
        identities = [
            {
                key: row[key]
                for key in (
                    "command_id",
                    "method_name",
                    "requires_ack",
                    "engine_step_id",
                    "repeat_index",
                    "request_set_sha256",
                    "batch_kind",
                    "speculative_selected_sequence_ids_sha256",
                    "dispatch_started_monotonic_ns",
                    "dispatch_published_monotonic_ns",
                )
            }
            for row in normalized_rows
        ]
        if command_inventory is None:
            command_inventory = identities
        elif identities != command_inventory:
            raise ValueError(
                "command inventories contain missing or reordered IDs"
            )
        rows_by_rank[rank] = {
            row["command_id"]: row for row in normalized_rows
        }
        flat_rows.extend(normalized_rows)
    compatible_clock = {
        key: clocks[0][key]
        for key in (
            "boot_id",
            "implementation",
            "resolution_s",
            "monotonic",
            "adjustable",
        )
    }
    for clock in clocks[1:]:
        if clock["boot_id"] != compatible_clock["boot_id"]:
            raise ValueError("boot ID mismatch across ranks")
        if any(
            clock[key] != compatible_clock[key]
            for key in (
                "implementation",
                "resolution_s",
                "monotonic",
                "adjustable",
            )
        ):
            raise ValueError("clock metadata mismatch across ranks")
    return flat_rows, rows_by_rank, compatible_clock


def _normalize_cuda_snapshots(
    timeline: dict,
    *,
    interval: dict,
    repeat_identity: int,
    prompt_sha256: str,
    command_ids: set[int],
    engine_step_ids: set[int],
) -> tuple[list[dict], dict[int, list[dict]]]:
    snapshots = _normalize_rank_rows(
        timeline.get("cuda_rank_snapshots"),
        name="CUDA rank snapshots",
    )
    flat_steps = []
    by_rank = {}
    for rank, snapshot in enumerate(snapshots):
        if snapshot.get("enabled") is not True:
            raise ValueError("CUDA timeline must be enabled")
        if snapshot.get("finalization_status") != "complete":
            raise ValueError("CUDA timeline finalization is incomplete")
        if (
            _integer(
                snapshot.get("dropped_steps"),
                "dropped CUDA steps",
                minimum=0,
            )
            != 0
            or _integer(
                snapshot.get("dropped_collectives"),
                "dropped CUDA collectives",
                minimum=0,
            )
            != 0
        ):
            raise ValueError("CUDA timeline rows are missing")
        normalized_steps = []
        for raw in _list(snapshot.get("steps"), "CUDA steps"):
            row = _mapping(raw, "CUDA step row")
            if _integer(row.get("rank"), "CUDA rank", minimum=0) != rank:
                raise ValueError("CUDA rank identity is invalid")
            command_id = _integer(
                row.get("command_id"),
                "CUDA command identity",
                minimum=0,
            )
            engine_step_id = _integer(
                row.get("engine_step_id"),
                "CUDA engine step identity",
                minimum=0,
            )
            if command_id not in command_ids:
                raise ValueError("CUDA row references an unknown command")
            if engine_step_id not in engine_step_ids:
                raise ValueError(
                    "CUDA row references an unknown engine step"
                )
            if _integer(
                row.get("repeat_index"),
                "CUDA repeat identity",
                minimum=0,
            ) != repeat_identity:
                raise ValueError("CUDA repeat identity mismatch")
            if _sha256(
                row.get("request_set_sha256"),
                "CUDA request digest",
            ) != prompt_sha256:
                raise ValueError("CUDA request digest mismatch")
            selected_sha256 = _sha256(
                row.get(
                    "speculative_selected_sequence_ids_sha256"
                ),
                "CUDA selected-sequence digest",
            )
            batch_kind = _text(
                row.get("batch_kind"),
                "CUDA batch kind",
            )
            wall_ns = _integer(
                row.get("wall_ns"),
                "CUDA method wall time",
                minimum=0,
            )
            cuda_ns = _integer(
                row.get("cuda_ns"),
                "CUDA duration",
                minimum=0,
            )
            if cuda_ns > wall_ns:
                raise ValueError(
                    "CUDA duration exceeds method wall time"
                )
            non_cuda = _integer(
                row.get("non_cuda_upper_bound_ns"),
                "CUDA non-CUDA upper bound",
                minimum=0,
            )
            if non_cuda != wall_ns - cuda_ns:
                raise ValueError(
                    "CUDA non-CUDA upper bound mismatch"
                )
            normalized = copy.deepcopy(row)
            normalized.update({
                "rank": rank,
                "command_id": command_id,
                "engine_step_id": engine_step_id,
                "repeat_index": repeat_identity,
                "request_set_sha256": prompt_sha256,
                "batch_kind": batch_kind,
                "speculative_selected_sequence_ids_sha256": (
                    selected_sha256
                ),
                "wall_ns": wall_ns,
                "cuda_ns": cuda_ns,
                "non_cuda_upper_bound_ns": non_cuda,
            })
            normalized_steps.append(normalized)
        for raw in _list(
            snapshot.get("collectives"),
            "CUDA collectives",
        ):
            row = _mapping(raw, "CUDA collective row")
            if _integer(
                row.get("rank"),
                "CUDA collective rank",
                minimum=0,
            ) != rank:
                raise ValueError("CUDA collective rank is invalid")
            command_id = _integer(
                row.get("command_id"),
                "CUDA collective command identity",
                minimum=0,
            )
            engine_step_id = _integer(
                row.get("engine_step_id"),
                "CUDA collective engine step identity",
                minimum=0,
            )
            if command_id not in command_ids:
                raise ValueError(
                    "CUDA collective references an unknown command"
                )
            if engine_step_id not in engine_step_ids:
                raise ValueError(
                    "CUDA collective references an unknown engine step"
                )
            wall_ns = _integer(
                row.get("wall_ns"),
                "CUDA collective wall time",
                minimum=0,
            )
            cuda_ns = _integer(
                row.get("cuda_ns"),
                "CUDA collective duration",
                minimum=0,
            )
            if cuda_ns > wall_ns:
                raise ValueError(
                    "CUDA collective duration exceeds wall time"
                )
        by_rank[rank] = normalized_steps
        flat_steps.extend(normalized_steps)
    return flat_steps, by_rank


def _normalize_engine_steps(
    timeline: dict,
    *,
    interval: dict,
    repeat_identity: int,
    prompt_sha256: str,
    command_ids: set[int],
) -> list[dict]:
    if _integer(
        timeline.get("engine_dropped_steps"),
        "dropped engine steps",
        minimum=0,
    ) != 0:
        raise ValueError("engine step rows are missing")
    raw_steps = _list(timeline.get("engine_steps"), "engine steps")
    if not raw_steps:
        raise ValueError("engine step rows are missing")
    normalized_steps = []
    seen_ids = set()
    for raw in raw_steps:
        row = _mapping(raw, "engine step row")
        engine_step_id = _integer(
            row.get("engine_step_id"),
            "engine step identity",
            minimum=0,
        )
        if engine_step_id in seen_ids:
            raise ValueError("duplicate engine step identity")
        seen_ids.add(engine_step_id)
        if _integer(
            row.get("repeat_index"),
            "engine repeat identity",
            minimum=0,
        ) != repeat_identity:
            raise ValueError("engine repeat identity mismatch")
        if _sha256(
            row.get("request_set_sha256"),
            "engine request digest",
        ) != prompt_sha256:
            raise ValueError("engine request digest mismatch")
        referenced_commands = [
            _integer(
                command_id,
                "engine command reference",
                minimum=0,
            )
            for command_id in _list(
                row.get("command_ids"),
                "engine command references",
            )
        ]
        if not referenced_commands:
            raise ValueError("engine command references are missing")
        if any(
            command_id not in command_ids
            for command_id in referenced_commands
        ):
            raise ValueError("engine row references an unknown command")
        started = _validate_timestamp_in_interval(
            row.get("started_monotonic_ns"),
            name="engine step start",
            interval=interval,
        )
        finished = _validate_timestamp_in_interval(
            row.get("finished_monotonic_ns"),
            name="engine step finish",
            interval=interval,
        )
        step_wall_ns = _duration(
            started,
            finished,
            "engine step",
        )
        if _integer(
            row.get("step_wall_ns"),
            "engine step wall time",
            minimum=0,
        ) != step_wall_ns:
            raise ValueError("engine step wall time mismatch")
        phases = _mapping(row.get("phases"), "engine step phases")
        if tuple(phases) != ENGINE_STEP_PHASES:
            raise ValueError("engine step phase inventory is invalid")
        normalized_phases = {}
        executed_intervals = []
        for phase_name in ENGINE_STEP_PHASES:
            phase = _mapping(
                phases[phase_name],
                f"engine phase {phase_name}",
            )
            executed = _strict_bool(
                phase.get("executed"),
                f"engine phase {phase_name} executed",
            )
            duration_ns = _integer(
                phase.get("duration_ns"),
                f"engine phase {phase_name} duration",
                minimum=0,
            )
            if not executed:
                if (
                    phase.get("started_monotonic_ns") is not None
                    or phase.get("finished_monotonic_ns") is not None
                    or duration_ns != 0
                ):
                    raise ValueError(
                        f"engine phase {phase_name} skipped row is invalid"
                    )
                normalized_phases[phase_name] = {
                    "executed": False,
                    "started_monotonic_ns": None,
                    "finished_monotonic_ns": None,
                    "duration_ns": 0,
                }
                continue
            phase_started = _validate_timestamp_in_interval(
                phase.get("started_monotonic_ns"),
                name=f"engine phase {phase_name} start",
                interval=interval,
            )
            phase_finished = _validate_timestamp_in_interval(
                phase.get("finished_monotonic_ns"),
                name=f"engine phase {phase_name} finish",
                interval=interval,
            )
            if phase_started < started or phase_finished > finished:
                raise ValueError(
                    f"engine phase {phase_name} lies outside its step"
                )
            if _duration(
                phase_started,
                phase_finished,
                f"engine phase {phase_name}",
            ) != duration_ns:
                raise ValueError(
                    f"engine phase {phase_name} duration mismatch"
                )
            executed_intervals.append(
                (phase_started, phase_finished, phase_name)
            )
            normalized_phases[phase_name] = {
                "executed": True,
                "started_monotonic_ns": phase_started,
                "finished_monotonic_ns": phase_finished,
                "duration_ns": duration_ns,
            }
        for previous, current in zip(
            sorted(executed_intervals),
            sorted(executed_intervals)[1:],
        ):
            if current[0] < previous[1]:
                raise ValueError("engine phase intervals overlap")
        status = _text(row.get("status"), "engine status")
        if (
            status != "ok"
            or row.get("error_type", "") != ""
            or row.get("detail", "") != ""
        ):
            raise ValueError(
                "engine status must be ok with empty error detail"
            )
        normalized_steps.append({
            "engine_step_id": engine_step_id,
            "repeat_index": repeat_identity,
            "request_set_sha256": prompt_sha256,
            "batch_kind": _text(
                row.get("batch_kind"),
                "engine batch kind",
            ),
            "speculative_selected_sequence_ids_sha256": _sha256(
                row.get(
                    "speculative_selected_sequence_ids_sha256"
                ),
                "engine selected-sequence digest",
            ),
            "command_ids": referenced_commands,
            "started_monotonic_ns": started,
            "finished_monotonic_ns": finished,
            "step_wall_ns": step_wall_ns,
            "phases": normalized_phases,
            "status": "ok",
            "error_type": "",
            "detail": "",
        })
    return normalized_steps


def _normalize_telemetry(
    run: dict,
    *,
    interval: dict,
    repeat_identity: int,
    gpu_uuids: list[str],
) -> dict:
    telemetry = _mapping(run.get("telemetry"), "repeat telemetry")
    gpu_rows = _list(telemetry.get("gpu_rows"), "GPU telemetry rows")
    host_rows = _list(telemetry.get("host_rows"), "host telemetry rows")
    if not gpu_rows or not host_rows:
        raise ValueError("telemetry coverage is incomplete")
    normalized_gpu = []
    seen_gpu_samples = set()
    for raw in gpu_rows:
        row = _mapping(raw, "GPU telemetry row")
        if _integer(
            row.get("repeat_index"),
            "GPU telemetry repeat identity",
            minimum=0,
        ) != repeat_identity:
            raise ValueError("GPU telemetry repeat identity mismatch")
        monotonic_ns = _validate_timestamp_in_interval(
            row.get("sampled_at_monotonic_ns"),
            name="GPU telemetry monotonic timestamp",
            interval=interval,
        )
        unix_ns = _integer(
            row.get("sampled_at_unix_ns"),
            "GPU telemetry Unix timestamp",
            minimum=0,
        )
        if not (
            interval["started_at_unix_ns"]
            <= unix_ns
            <= interval["finished_at_unix_ns"]
        ):
            raise ValueError(
                "GPU telemetry row lies outside the campaign interval"
            )
        gpu_uuid = _text(row.get("gpu_uuid"), "GPU telemetry UUID")
        if gpu_uuid not in gpu_uuids:
            raise ValueError("GPU telemetry UUID is unknown")
        identity = (gpu_uuid, monotonic_ns, unix_ns)
        if identity in seen_gpu_samples:
            raise ValueError("duplicate GPU telemetry row")
        seen_gpu_samples.add(identity)
        normalized_gpu.append({
            "repeat_index": repeat_identity,
            "sampled_at_unix_ns": unix_ns,
            "sampled_at_monotonic_ns": monotonic_ns,
            "gpu_uuid": gpu_uuid,
        })
    if set(row["gpu_uuid"] for row in normalized_gpu) != set(gpu_uuids):
        raise ValueError("GPU telemetry identity coverage is incomplete")
    normalized_host = []
    previous_host = None
    for raw in host_rows:
        row = _mapping(raw, "host telemetry row")
        if _integer(
            row.get("repeat_index"),
            "host telemetry repeat identity",
            minimum=0,
        ) != repeat_identity:
            raise ValueError("host telemetry repeat identity mismatch")
        monotonic_ns = _validate_timestamp_in_interval(
            row.get("sampled_at_monotonic_ns"),
            name="host telemetry monotonic timestamp",
            interval=interval,
        )
        unix_ns = _integer(
            row.get("sampled_at_unix_ns"),
            "host telemetry Unix timestamp",
            minimum=0,
        )
        if not (
            interval["started_at_unix_ns"]
            <= unix_ns
            <= interval["finished_at_unix_ns"]
        ):
            raise ValueError(
                "host telemetry row lies outside the campaign interval"
            )
        current = (monotonic_ns, unix_ns)
        if previous_host is not None and current <= previous_host:
            raise ValueError("host telemetry rows are reordered")
        previous_host = current
        normalized_host.append({
            "repeat_index": repeat_identity,
            "sampled_at_unix_ns": unix_ns,
            "sampled_at_monotonic_ns": monotonic_ns,
        })
    return {
        "gpu_rows": sorted(
            normalized_gpu,
            key=lambda row: (
                row["sampled_at_monotonic_ns"],
                row["gpu_uuid"],
            ),
        ),
        "host_rows": normalized_host,
    }


def _union_duration(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    ordered = sorted(intervals)
    merged = [list(ordered[0])]
    for start, finish in ordered[1:]:
        previous = merged[-1]
        if start <= previous[1]:
            previous[1] = max(previous[1], finish)
        else:
            merged.append([start, finish])
    return sum(finish - start for start, finish in merged)


def compute_sync_debt(repeat: object) -> dict:
    normalized = copy.deepcopy(_mapping(repeat, "joined repeat"))
    rows_by_rank = normalized.get("_rows_by_rank")
    cuda_by_rank = normalized.get("_cuda_by_rank")
    engine_steps = normalized.get("engine_steps")
    if (
        not isinstance(rows_by_rank, dict)
        or not isinstance(cuda_by_rank, dict)
        or not isinstance(engine_steps, list)
    ):
        raise ValueError("joined repeat lacks normalized timeline rows")
    queue_by_rank = {}
    debt_rows = []
    for rank in range(4):
        command_rows = [
            rows_by_rank[rank][command_id]
            for command_id in sorted(rows_by_rank[rank])
        ]
        previous_finish = None
        intervals = []
        for row in command_rows:
            queue_wait = row["worker_queue_wait_ns"]
            overlap = (
                0
                if previous_finish is None
                else max(
                    0,
                    previous_finish
                    - row["dispatch_published_monotonic_ns"],
                )
            )
            if overlap > queue_wait:
                raise ValueError(
                    "prior-command overlap exceeds containing queue interval"
                )
            row["queued_behind_prior_command_ns"] = overlap
            row["worker_ready_delay_ns"] = queue_wait - overlap
            if overlap:
                overlap_start = row[
                    "dispatch_published_monotonic_ns"
                ]
                overlap_finish = overlap_start + overlap
                if (
                    overlap_start < row[
                        "dispatch_published_monotonic_ns"
                    ]
                    or overlap_finish > row["method_start_ns"]
                ):
                    raise ValueError(
                        "debt overlap exceeds its containing intervals"
                    )
                intervals.append((overlap_start, overlap_finish))
                debt_rows.append({
                    "rank": rank,
                    "producing_command_id": row["command_id"] - 1,
                    "consuming_command_id": row["command_id"],
                    "overlap_ns": overlap,
                })
            previous_finish = row["method_finish_ns"]
        queue_by_rank[rank] = _union_duration(intervals)
    worker_queue_debt = max(queue_by_rank.values(), default=0)

    cuda_totals = {
        rank: sum(row["cuda_ns"] for row in cuda_by_rank[rank])
        for rank in range(4)
    }
    critical_rank = max(
        range(4),
        key=lambda rank: (cuda_totals[rank], rank),
    )
    worker_cuda_execution = cuda_totals[critical_rank]
    ack_intervals = []
    for row in rows_by_rank[0].values():
        if row["ack_wait_ns"] is not None:
            ack_intervals.append((
                row["ack_wait_started_monotonic_ns"],
                row["ack_wait_finished_monotonic_ns"],
            ))
    ack_wait = _union_duration(ack_intervals)
    scheduler_postprocess = sum(
        phase["duration_ns"]
        for step in engine_steps
        for name, phase in step["phases"].items()
        if name in SCHEDULER_POSTPROCESS_PHASES
    )
    components = {
        "worker_queue_debt": worker_queue_debt,
        "worker_cuda_execution": worker_cuda_execution,
        "ack_wait": ack_wait,
        "scheduler_postprocess": scheduler_postprocess,
    }
    step_wall_ns = sum(step["step_wall_ns"] for step in engine_steps)
    attributed_ns = sum(components.values())
    residual_ns = step_wall_ns - attributed_ns
    tolerance_ns = max(
        ABSOLUTE_CONSERVATION_NS,
        math.ceil(step_wall_ns * RELATIVE_CONSERVATION_LIMIT),
    )
    conservation_passed = (
        residual_ns >= 0
        and residual_ns <= tolerance_ns
    )
    if residual_ns < 0:
        raise ValueError(
            "timeline conservation is over-attributed"
        )
    if not conservation_passed:
        raise ValueError(
            "timeline conservation residual exceeds tolerance"
        )
    normalized.pop("_rows_by_rank", None)
    normalized.pop("_cuda_by_rank", None)
    normalized.update({
        "critical_rank": critical_rank,
        "queue_debt_by_rank_ns": {
            str(rank): queue_by_rank[rank] for rank in range(4)
        },
        "debt_rows": debt_rows,
        "components_ns": components,
        "conservation": {
            "step_wall_ns": step_wall_ns,
            "attributed_ns": attributed_ns,
            "residual_ns": residual_ns,
            "tolerance_ns": tolerance_ns,
            "passed": conservation_passed,
        },
    })
    return normalized


def join_repeat_timeline(
    worker: object,
    repeat_index: int,
) -> dict:
    normalized_worker = _mapping(worker, "epoch worker")
    repeat_index = _integer(
        repeat_index,
        "repeat index",
        minimum=0,
    )
    measured_runs = _list(
        normalized_worker.get("measured_runs"),
        "measured runs",
    )
    matching = [
        run
        for run in measured_runs
        if (
            isinstance(run, dict)
            and run.get("repeat") == repeat_index
        )
    ]
    if len(matching) != 1:
        raise ValueError("repeat inventory is missing or duplicated")
    run = matching[0]
    interval = _normalize_campaign_interval(run)
    repeat_identity = _integer(
        run.get("command_timeline_repeat_index"),
        "command timeline repeat identity",
        minimum=0,
    )
    prompt_sha256 = _sha256(
        normalized_worker.get("prompt_sha256"),
        "worker prompt digest",
    )
    timeline = _mapping(
        _mapping(run.get("runtime"), "run runtime").get(
            "command_timeline"
        ),
        "command timeline",
    )
    if _integer(
        timeline.get("schema_version"),
        "command timeline schema version",
        minimum=0,
    ) != SCHEMA_VERSION:
        raise ValueError("command timeline schema version mismatch")
    (
        command_rows,
        rows_by_rank,
        clock,
    ) = _normalize_command_snapshots(
        timeline,
        interval=interval,
        repeat_identity=repeat_identity,
        prompt_sha256=prompt_sha256,
    )
    command_ids = {
        row["command_id"] for row in command_rows
    }
    engine_steps = _normalize_engine_steps(
        timeline,
        interval=interval,
        repeat_identity=repeat_identity,
        prompt_sha256=prompt_sha256,
        command_ids=command_ids,
    )
    engine_step_ids = {
        row["engine_step_id"] for row in engine_steps
    }
    cuda_steps, cuda_by_rank = _normalize_cuda_snapshots(
        timeline,
        interval=interval,
        repeat_identity=repeat_identity,
        prompt_sha256=prompt_sha256,
        command_ids=command_ids,
        engine_step_ids=engine_step_ids,
    )
    engine_by_id = {
        row["engine_step_id"]: row for row in engine_steps
    }
    for row in cuda_steps:
        command = rows_by_rank[row["rank"]][row["command_id"]]
        if row["engine_step_id"] != command["engine_step_id"]:
            raise ValueError("CUDA command and engine identities mismatch")
        if row["wall_ns"] > command["worker_method_wall_ns"]:
            raise ValueError(
                "CUDA method wall time exceeds containing command"
            )
        engine = engine_by_id[row["engine_step_id"]]
        if (
            row[
                "speculative_selected_sequence_ids_sha256"
            ]
            != command[
                "speculative_selected_sequence_ids_sha256"
            ]
            or row[
                "speculative_selected_sequence_ids_sha256"
            ]
            != engine[
                "speculative_selected_sequence_ids_sha256"
            ]
        ):
            raise ValueError(
                "CUDA selected-sequence identity mismatch"
            )
        if (
            row["batch_kind"] != command["batch_kind"]
            or row["batch_kind"] != engine["batch_kind"]
        ):
            raise ValueError("CUDA batch kind identity mismatch")
    for rank_rows in rows_by_rank.values():
        for row in rank_rows.values():
            if row["engine_step_id"] is None:
                continue
            if row["engine_step_id"] not in engine_by_id:
                raise ValueError(
                    "command references an unknown engine step"
                )
            engine = engine_by_id[row["engine_step_id"]]
            if (
                row[
                    "speculative_selected_sequence_ids_sha256"
                ]
                != engine[
                    "speculative_selected_sequence_ids_sha256"
                ]
            ):
                raise ValueError(
                    "selected-sequence digest mismatch"
                )
    for command_id, rank_zero_row in rows_by_rank[0].items():
        if not rank_zero_row["requires_ack"]:
            continue
        worker_ack_finishes = [
            rows_by_rank[rank][command_id][
                "ack_send_finished_monotonic_ns"
            ]
            for rank in range(1, 4)
        ]
        if rank_zero_row[
            "ack_wait_finished_monotonic_ns"
        ] < max(worker_ack_finishes):
            raise ValueError(
                "rank-zero ack wait finished before a worker ack send"
            )
    for engine in engine_steps:
        command_intervals = []
        for command_id in engine["command_ids"]:
            rank_zero_row = rows_by_rank[0][command_id]
            command_intervals.append((
                rank_zero_row["method_start_ns"],
                rank_zero_row["method_finish_ns"],
            ))
            if rank_zero_row["ack_wait_ns"] is not None:
                command_intervals.append((
                    rank_zero_row[
                        "ack_wait_started_monotonic_ns"
                    ],
                    rank_zero_row[
                        "ack_wait_finished_monotonic_ns"
                    ],
                ))
        scheduler_intervals = [
            (
                phase["started_monotonic_ns"],
                phase["finished_monotonic_ns"],
            )
            for name, phase in engine["phases"].items()
            if (
                name in SCHEDULER_POSTPROCESS_PHASES
                and phase["executed"]
            )
        ]
        for scheduler_interval in scheduler_intervals:
            for command_interval in command_intervals:
                if max(
                    scheduler_interval[0],
                    command_interval[0],
                ) < min(
                    scheduler_interval[1],
                    command_interval[1],
                ):
                    raise ValueError(
                        "scheduler and command intervals overlap"
                    )
    telemetry = _normalize_telemetry(
        run,
        interval=interval,
        repeat_identity=repeat_identity,
        gpu_uuids=[
            _text(value, "GPU UUID")
            for value in _list(
                normalized_worker.get("gpu_uuids"),
                "GPU UUIDs",
            )
        ],
    )
    timing = _normalize_timing(run)
    joined = {
        "repeat": repeat_index,
        "repeat_identity": repeat_identity,
        "campaign_interval": interval,
        "clock": clock,
        "command_rows": command_rows,
        "cuda_steps": cuda_steps,
        "engine_steps": engine_steps,
        "telemetry": telemetry,
        "timing": timing,
        "_rows_by_rank": rows_by_rank,
        "_cuda_by_rank": cuda_by_rank,
    }
    return compute_sync_debt(joined)


def stationarity_for_values(values: object) -> dict:
    normalized = [
        _fraction(value, "stationarity value", minimum=0)
        for value in _list(values, "stationarity values")
    ]
    if len(normalized) != MEASURED_RUNS_PER_EPOCH:
        raise ValueError("stationarity requires exactly five values")
    median = statistics.median(normalized)
    deviations = [abs(value - median) for value in normalized]
    mad = statistics.median(deviations)
    robust_dispersion = (
        Fraction(0, 1) if median == 0 and mad == 0
        else math.inf if median == 0
        else mad / median
    )
    first_half_median = statistics.median(normalized[0:2])
    second_half_median = statistics.median(normalized[3:5])
    half_delta = abs(second_half_median - first_half_median)
    half_drift = (
        Fraction(0, 1) if median == 0 and half_delta == 0
        else math.inf if median == 0
        else half_delta / median
    )
    robust_passed = (
        robust_dispersion != math.inf
        and robust_dispersion <= Fraction(
            str(ROBUST_DISPERSION_LIMIT)
        )
    )
    drift_passed = (
        half_drift != math.inf
        and half_drift <= Fraction(str(HALF_DRIFT_LIMIT))
    )
    return {
        "values": [
            _canonical_fraction(value) for value in normalized
        ],
        "median": _canonical_fraction(median),
        "mad": _canonical_fraction(mad),
        "robust_dispersion": float(robust_dispersion),
        "robust_dispersion_limit": ROBUST_DISPERSION_LIMIT,
        "robust_dispersion_passed": robust_passed,
        "first_half_median": _canonical_fraction(
            first_half_median
        ),
        "second_half_median": _canonical_fraction(
            second_half_median
        ),
        "half_drift": float(half_drift),
        "half_drift_limit": HALF_DRIFT_LIMIT,
        "half_drift_passed": drift_passed,
        "passed": robust_passed and drift_passed,
    }


def build_epoch_admission(
    identity: EpochIdentity,
    raw_inputs: object,
) -> dict:
    _require_expected_identity(identity)
    raw = _mapping(raw_inputs, "epoch raw inputs")
    worker_value = raw.get("worker", raw)
    worker = validate_epoch_worker(worker_value, identity)
    repeats = [
        join_repeat_timeline(worker, repeat_index)
        for repeat_index in range(MEASURED_RUNS_PER_EPOCH)
    ]
    metric_values = {
        "e2e": [
            _median(
                [
                    row["completion_latency_ns"]
                    for row in repeat["timing"]["per_request"]
                ],
                "E2E timing",
            )
            for repeat in repeats
        ],
        "ttft": [
            _median(
                [
                    row["ttft_ns"]
                    for row in repeat["timing"]["per_request"]
                ],
                "TTFT timing",
            )
            for repeat in repeats
        ],
        "tpot": [
            _median(
                [
                    row["tpot_ns"]
                    for row in repeat["timing"]["per_request"]
                ],
                "TPOT timing",
            )
            for repeat in repeats
        ],
        "proposal_forward": [
            repeat["timing"].get(
                "proposal_forward_ns",
                repeat["components_ns"][
                    "worker_cuda_execution"
                ],
            )
            for repeat in repeats
        ],
        "worker_queue_debt": [
            repeat["components_ns"]["worker_queue_debt"]
            for repeat in repeats
        ],
        "queued_behind_prior_command": [
            repeat["components_ns"]["worker_queue_debt"]
            for repeat in repeats
        ],
        "worker_cuda_execution": [
            repeat["components_ns"]["worker_cuda_execution"]
            for repeat in repeats
        ],
        "ack_wait": [
            repeat["components_ns"]["ack_wait"]
            for repeat in repeats
        ],
        "scheduler_postprocess": [
            repeat["components_ns"]["scheduler_postprocess"]
            for repeat in repeats
        ],
        "conservation_residual": [
            repeat["conservation"]["residual_ns"]
            for repeat in repeats
        ],
    }
    stationarity = {
        metric: stationarity_for_values(values)
        for metric, values in metric_values.items()
    }
    timeline_passed = all(
        repeat["conservation"]["passed"] for repeat in repeats
    )
    stationarity_passed = all(
        row["passed"] for row in stationarity.values()
    )
    return {
        "identity": identity.to_dict(),
        "worker": worker,
        "repeats": repeats,
        "metrics": metric_values,
        "stationarity": stationarity,
        "identity_correctness_passed": True,
        "timeline_conservation_passed": timeline_passed,
        "stationarity_passed": stationarity_passed,
        "passed": timeline_passed and stationarity_passed,
    }


def _worker_identity_view(worker: dict) -> dict:
    return {
        "source_commit": worker["source_commit"],
        "source_tree_sha256": worker["source_tree_sha256"],
        "target_checkpoint_identifier": (
            worker["target_checkpoint_identifier"]
        ),
        "draft_checkpoint_identifier": (
            worker["draft_checkpoint_identifier"]
        ),
        "tokenizer_identifier": worker["tokenizer_identifier"],
        "gpu_uuids": worker["gpu_uuids"],
        "prompt_sha256": worker["prompt_sha256"],
        "prompt_rows": worker["prompt_rows"],
        "request_order": worker["request_order"],
        "requested_output_tokens": worker[
            "requested_output_tokens"
        ],
    }


def _validate_cross_epoch_identity(epochs: dict[str, dict]) -> None:
    expected_keys = [
        identity.key for identity in expected_epoch_identities()
    ]
    if list(epochs) != expected_keys:
        raise ValueError("epoch inventory or order is invalid")
    reference = _worker_identity_view(
        epochs[expected_keys[0]]["worker"]
    )
    messages = {
        "source_commit": "source commit mismatch",
        "source_tree_sha256": "source tree mismatch",
        "target_checkpoint_identifier": (
            "target checkpoint mismatch"
        ),
        "draft_checkpoint_identifier": (
            "draft checkpoint mismatch"
        ),
        "tokenizer_identifier": "tokenizer mismatch",
        "gpu_uuids": "GPU UUID mismatch",
        "prompt_sha256": "prompt digest mismatch",
        "prompt_rows": "prompt rows mismatch",
        "request_order": "request order mismatch",
        "requested_output_tokens": "output length mismatch",
    }
    for key in expected_keys[1:]:
        current = _worker_identity_view(epochs[key]["worker"])
        for field, expected_value in reference.items():
            if current[field] != expected_value:
                raise ValueError(messages[field])
    reference_parity = _parity_semantic_view(epochs[expected_keys[0]][
        "worker"
    ][
        "measured_runs"
    ][0]["correctness"])
    for key in expected_keys:
        for run in epochs[key]["worker"]["measured_runs"]:
            if _parity_semantic_view(
                run["correctness"]
            ) != reference_parity:
                raise ValueError(
                    "graph/eager token, acceptance, or transaction parity "
                    "mismatch"
                )
    graph_epochs = [
        epochs[identity.key]
        for identity in expected_epoch_identities()
        if identity.label == "graph"
    ]
    graph_reference = graph_epochs[0]["worker"]["warmup_runs"][0][
        "correctness"
    ]
    for epoch in graph_epochs[1:]:
        graph_current = epoch["worker"]["warmup_runs"][0][
            "correctness"
        ]
        if (
            graph_current["rank_graph_identities"]
            != graph_reference["rank_graph_identities"]
        ):
            raise ValueError(
                "graph identity mismatch across graph epochs"
            )
        if (
            graph_current["rank_graph_resources"]
            != graph_reference["rank_graph_resources"]
        ):
            raise ValueError(
                "graph resource mismatch across graph epochs"
            )


def summarize_boundary_effects(blocks: object) -> dict:
    raw_blocks = _list(blocks, "boundary effect blocks")
    if len(raw_blocks) != len(BLOCK_SCHEDULE):
        raise ValueError("boundary effects require four blocks")
    normalized_blocks = []
    median_e2e_fractions = []
    for block_index, raw in enumerate(raw_blocks):
        row = _mapping(raw, "boundary effect block")
        if _integer(
            row.get("block_index"),
            "effect block index",
            minimum=0,
        ) != block_index:
            raise ValueError("boundary effect block order is invalid")
        expected_order = "_".join(BLOCK_SCHEDULE[block_index])
        if row.get("order") != expected_order:
            raise ValueError("boundary effect block order mismatch")
        e2e_delta = _fraction(
            row.get("e2e_delta_ns"),
            "E2E paired delta",
        )
        components = _mapping(
            row.get("component_deltas_ns"),
            "component paired deltas",
        )
        if tuple(components) != BOUNDARY_NAMES:
            raise ValueError("boundary component inventory is invalid")
        normalized_components = {
            name: _fraction(
                components[name],
                f"{name} paired delta",
            )
            for name in BOUNDARY_NAMES
        }
        unexplained = _fraction(
            row.get("absolute_unexplained_ns"),
            "absolute unexplained E2E",
            minimum=0,
        )
        median_pair = row.get("median_e2e_pair_ns")
        if median_pair is None:
            median_e2e_fraction = _fraction(
                row.get("median_e2e_ns"),
                "median E2E",
                minimum=1,
            )
        else:
            pair = _list(median_pair, "median E2E pair")
            if len(pair) != 2:
                raise ValueError("median E2E pair must have two values")
            first = _fraction(
                pair[0],
                "first median E2E",
                minimum=1,
            )
            second = _fraction(
                pair[1],
                "second median E2E",
                minimum=1,
            )
            median_e2e_fraction = Fraction(first + second, 2)
        median_e2e_fractions.append(median_e2e_fraction)
        explanation = {}
        explanation_defined = {}
        for name in BOUNDARY_NAMES:
            component = normalized_components[name]
            if e2e_delta == 0 and component != 0:
                explanation[name] = None
                explanation_defined[name] = False
            elif e2e_delta == 0:
                explanation[name] = 0.0
                explanation_defined[name] = True
            else:
                explanation[name] = float(
                    abs(component) / abs(e2e_delta)
                )
                explanation_defined[name] = True
        normalized_blocks.append({
            "block_index": block_index,
            "order": expected_order,
            "e2e_delta_ns": _canonical_fraction(e2e_delta),
            "component_deltas_ns": {
                name: _canonical_fraction(value)
                for name, value in normalized_components.items()
            },
            "explanation_ratios": explanation,
            "explanation_ratio_defined": explanation_defined,
            "absolute_unexplained_ns": _canonical_fraction(
                unexplained
            ),
            "median_e2e_ns": _canonical_fraction(
                median_e2e_fraction
            ),
        })
    median_unexplained = statistics.median(
        _fraction(
            row["absolute_unexplained_ns"],
            "absolute unexplained E2E",
            minimum=0,
        )
        for row in normalized_blocks
    )
    median_e2e = statistics.median(median_e2e_fractions)
    unexplained_ratio = median_unexplained / median_e2e
    unexplained_ratio_passed = (
        median_unexplained.numerator
        * median_e2e.denominator
        * UNEXPLAINED_E2E_RATIO.denominator
        <= median_e2e.numerator
        * median_unexplained.denominator
        * UNEXPLAINED_E2E_RATIO.numerator
    )
    boundaries = {}
    for name in BOUNDARY_NAMES:
        component_values = [
            _fraction(
                row["component_deltas_ns"][name],
                f"{name} paired delta",
            )
            for row in normalized_blocks
        ]
        e2e_values = [
            _fraction(
                row["e2e_delta_ns"],
                "E2E paired delta",
            )
            for row in normalized_blocks
        ]
        qualifying = [
            row["block_index"]
            for row, component, e2e_delta in zip(
                normalized_blocks,
                component_values,
                e2e_values,
            )
            if (
                row["explanation_ratio_defined"][name]
                and abs(component)
                >= abs(e2e_delta) * BOUNDARY_EXPLANATION_RATIO
            )
        ]
        same_sign = [
            row["block_index"]
            for row, component, e2e_delta in zip(
                normalized_blocks,
                component_values,
                e2e_values,
            )
            if (
                _sign(component)
                == _sign(e2e_delta)
                != 0
            )
        ]
        label_signs = [_sign(value) for value in component_values]
        positive_label_count = label_signs.count(1)
        negative_label_count = label_signs.count(-1)
        label_common_sign = (
            1
            if positive_label_count >= BOUNDARY_BLOCK_COUNT
            else -1
            if negative_label_count >= BOUNDARY_BLOCK_COUNT
            else 0
        )
        position_values = [
            (
                component
                if row["order"] == "eager_graph"
                else -component
            )
            for row, component in zip(
                normalized_blocks,
                component_values,
            )
        ]
        aggregate_position = statistics.median(position_values)
        aggregate_position_sign = _sign(aggregate_position)
        position_same_direction_count = sum(
            _sign(value) == aggregate_position_sign != 0
            for value in position_values
        )
        position_balance_consistent = (
            aggregate_position_sign != 0
            and position_same_direction_count
            >= BOUNDARY_BLOCK_COUNT
        )
        order_group_checks = {}
        for order in ("eager_graph", "graph_eager"):
            group = [
                (row, position)
                for row, position in zip(
                    normalized_blocks,
                    position_values,
                )
                if row["order"] == order
            ]
            group_position = statistics.median(
                position for _row, position in group
            )
            direction_matches = (
                _sign(group_position)
                == aggregate_position_sign
                != 0
            )
            has_qualifying_block = any(
                row["block_index"] in qualifying
                and _sign(position) == aggregate_position_sign
                for row, position in group
            )
            order_group_checks[order] = {
                "aggregate_position_delta_ns": (
                    _canonical_fraction(group_position)
                ),
                "direction_matches": direction_matches,
                "has_qualifying_block": has_qualifying_block,
                "passed": (
                    direction_matches and has_qualifying_block
                ),
            }
        order_consistent = all(
            check["passed"] for check in order_group_checks.values()
        )
        sequence_interaction = (
            _fraction(
                order_group_checks["eager_graph"][
                    "aggregate_position_delta_ns"
                ],
                "eager-graph position effect",
            )
            - _fraction(
                order_group_checks["graph_eager"][
                    "aggregate_position_delta_ns"
                ],
                "graph-eager position effect",
            )
        )
        undefined = [
            row["block_index"]
            for row in normalized_blocks
            if not row["explanation_ratio_defined"][name]
        ]
        boundaries[name] = {
            "qualifying_block_indices": qualifying,
            "qualifying_block_count": len(qualifying),
            "same_sign_block_indices": same_sign,
            "same_sign_block_count": len(same_sign),
            "undefined_explanation_block_indices": undefined,
            "order_group_checks": order_group_checks,
            "aggregate_label_sign": label_common_sign,
            "aggregate_position_delta_ns": _canonical_fraction(
                aggregate_position
            ),
            "sequence_interaction_ns": _canonical_fraction(
                sequence_interaction
            ),
            "sequence_interaction_consistent": order_consistent,
            "position_balance_consistent": (
                position_balance_consistent
            ),
            "localized": (
                len(qualifying) >= BOUNDARY_BLOCK_COUNT
                and len(same_sign) >= BOUNDARY_BLOCK_COUNT
                and label_common_sign != 0
                and unexplained_ratio_passed
                and not undefined
                and position_balance_consistent
                and order_consistent
            ),
        }
    return {
        "blocks": normalized_blocks,
        "median_absolute_unexplained_ns": _canonical_fraction(
            median_unexplained
        ),
        "median_e2e_ns": _canonical_fraction(median_e2e),
        "median_unexplained_ratio": float(unexplained_ratio),
        "median_unexplained_ratio_passed": unexplained_ratio_passed,
        "boundaries": boundaries,
    }


def compute_paired_boundary_effects(
    epochs: object,
) -> dict:
    normalized_epochs = _mapping(epochs, "epochs")
    _validate_cross_epoch_identity(normalized_epochs)
    blocks = []
    for block_index, _labels in enumerate(BLOCK_SCHEDULE):
        identities = [
            identity
            for identity in expected_epoch_identities()
            if identity.block_index == block_index
        ]
        by_label = {
            identity.label: normalized_epochs[identity.key]
            for identity in identities
        }
        eager = by_label["eager"]
        graph = by_label["graph"]
        eager_e2e = statistics.median(
            _fraction(value, "eager E2E", minimum=0)
            for value in eager["metrics"]["e2e"]
        )
        graph_e2e = statistics.median(
            _fraction(value, "graph E2E", minimum=0)
            for value in graph["metrics"]["e2e"]
        )
        e2e_delta = graph_e2e - eager_e2e
        component_deltas = {}
        for name in BOUNDARY_NAMES:
            component_deltas[name] = (
                statistics.median(
                    _fraction(
                        value,
                        f"graph {name}",
                        minimum=0,
                    )
                    for value in graph["metrics"][name]
                )
                - statistics.median(
                    _fraction(
                        value,
                        f"eager {name}",
                        minimum=0,
                    )
                    for value in eager["metrics"][name]
                )
            )
        unexplained = abs(
            e2e_delta
            - sum(component_deltas.values(), Fraction(0, 1))
        )
        blocks.append({
            "block_index": block_index,
            "order": "_".join(BLOCK_SCHEDULE[block_index]),
            "e2e_delta_ns": _canonical_fraction(e2e_delta),
            "component_deltas_ns": {
                name: _canonical_fraction(value)
                for name, value in component_deltas.items()
            },
            "absolute_unexplained_ns": _canonical_fraction(
                unexplained
            ),
            "median_e2e_ns": _canonical_fraction(
                min(eager_e2e, graph_e2e)
            ),
            "median_e2e_pair_ns": [
                _canonical_fraction(eager_e2e),
                _canonical_fraction(graph_e2e),
            ],
        })
    return summarize_boundary_effects(blocks)


def classify_boundary(
    bundle_admission: object,
    effects: object,
) -> dict:
    admission = _mapping(bundle_admission, "bundle admission")
    normalized_effects = _mapping(effects, "boundary effects")
    identity_passed = _strict_bool(
        admission.get("identity_correctness_passed"),
        "identity/correctness admission",
    )
    timeline_passed = _strict_bool(
        admission.get("timeline_conservation_passed"),
        "timeline/conservation admission",
    )
    stationarity_passed = _strict_bool(
        admission.get("stationarity_passed"),
        "stationarity admission",
    )
    admission_passed = _strict_bool(
        admission.get("passed"),
        "bundle admission summary",
    )
    if admission_passed is not (
        identity_passed
        and timeline_passed
        and stationarity_passed
    ):
        raise ValueError("bundle admission summary is inconsistent")
    classification = "PAIRED_PROTOCOL_UNSTABLE"
    localized_boundary = None
    stable_but_unlocalized = False
    authorized = False
    if not identity_passed:
        classification = "INVALID_IDENTITY_OR_CORRECTNESS"
    elif not timeline_passed:
        classification = "TIMELINE_INCOMPLETE_OR_NONCONSERVING"
    elif not stationarity_passed:
        classification = "PAIRED_PROTOCOL_UNSTABLE"
    else:
        boundaries = _mapping(
            normalized_effects.get("boundaries"),
            "boundary summaries",
        )
        candidates = [
            name
            for name in BOUNDARY_NAMES
            if (
                isinstance(boundaries.get(name), dict)
                and boundaries[name].get("localized") is True
            )
        ]
        if len(candidates) == 1:
            localized_boundary = candidates[0]
            classification = "BOUNDARY_LOCALIZED"
            authorized = True
        else:
            stable_but_unlocalized = True
    return {
        "classification": classification,
        "localized_boundary": localized_boundary,
        "stable_but_unlocalized": stable_but_unlocalized,
        "runtime_optimization_authorized": authorized,
        "performance_improvement_established": False,
        "phase_1_complete": False,
        "promotion_ready": False,
    }


def _normalize_input_files(value: object) -> dict[str, dict]:
    rows = _mapping(value, "raw input files")
    if not rows:
        raise ValueError("raw input files must not be empty")
    normalized = {}
    for key in sorted(rows):
        name = _text(key, "raw input key")
        row = _mapping(rows[key], "raw input file row")
        normalized[name] = {
            "path": _safe_relative_path(
                row.get("path"),
                "raw input path",
            ),
            "sha256": _sha256(
                row.get("sha256"),
                "raw input digest",
            ),
        }
    return normalized


def _normalize_source_files(value: object) -> dict[str, str]:
    rows = _mapping(value, "source files")
    if not rows:
        raise ValueError("source files must not be empty")
    normalized = {}
    for raw_path in sorted(rows):
        path = _safe_relative_path(raw_path, "source path")
        normalized[path] = _sha256(
            rows[raw_path],
            f"source digest {path}",
        )
    return normalized


def _normalize_metadata(
    metadata: object,
) -> tuple[dict, dict]:
    row = _mapping(metadata, "artifact metadata")
    configuration = row.get(
        "configuration",
        EXACT_CONFIGURATION,
    )
    if configuration != EXACT_CONFIGURATION:
        raise ValueError("artifact configuration is not exact")
    provenance = row.get("provenance")
    if provenance is None:
        provenance = {
            key: copy.deepcopy(value)
            for key, value in row.items()
            if key != "configuration"
        }
    provenance = copy.deepcopy(
        _mapping(provenance, "artifact provenance")
    )
    if not provenance:
        raise ValueError("artifact provenance must not be empty")
    _validate_bounded_json(provenance, name="artifact provenance")
    return copy.deepcopy(EXACT_CONFIGURATION), provenance


def _build_blocks(epochs: dict[str, dict]) -> list[dict]:
    blocks = []
    for block_index, labels in enumerate(BLOCK_SCHEDULE):
        keys = [
            identity.key
            for identity in expected_epoch_identities()
            if identity.block_index == block_index
        ]
        blocks.append({
            "block_index": block_index,
            "order": "_".join(labels),
            "epoch_keys": keys,
            "passed": all(epochs[key]["passed"] for key in keys),
        })
    return blocks


def _build_admission(epochs: dict[str, dict]) -> dict:
    identity_passed = all(
        epoch["identity_correctness_passed"]
        for epoch in epochs.values()
    )
    timeline_passed = all(
        epoch["timeline_conservation_passed"]
        for epoch in epochs.values()
    )
    stationarity_passed = all(
        epoch["stationarity_passed"]
        for epoch in epochs.values()
    )
    return {
        "identity_correctness_passed": identity_passed,
        "timeline_conservation_passed": timeline_passed,
        "stationarity_passed": stationarity_passed,
        "measured_epoch_count": len(epochs),
        "measured_repeat_count_total": sum(
            len(epoch["repeats"]) for epoch in epochs.values()
        ),
        "passed": (
            identity_passed
            and timeline_passed
            and stationarity_passed
        ),
    }


def _assemble_artifact(
    *,
    configuration: dict,
    provenance: dict,
    raw_input_files: dict,
    source_files: dict,
    epochs: dict[str, dict],
) -> dict:
    blocks = _build_blocks(epochs)
    admission = _build_admission(epochs)
    effects = compute_paired_boundary_effects(epochs)
    classification = classify_boundary(admission, effects)
    return {
        "schema_version": SCHEMA_VERSION,
        "schedule": [
            {
                "block_index": index,
                "order": "_".join(labels),
                "labels": list(labels),
            }
            for index, labels in enumerate(BLOCK_SCHEDULE)
        ],
        "configuration": copy.deepcopy(configuration),
        "provenance": copy.deepcopy(provenance),
        "raw_input_files": copy.deepcopy(raw_input_files),
        "source_files": copy.deepcopy(source_files),
        "epochs": copy.deepcopy(epochs),
        "blocks": blocks,
        "admission": admission,
        "effects": effects,
        "classification": classification["classification"],
        "localized_boundary": classification["localized_boundary"],
        "stable_but_unlocalized": classification[
            "stable_but_unlocalized"
        ],
        "runtime_optimization_authorized": classification[
            "runtime_optimization_authorized"
        ],
        "performance_improvement_established": False,
        "phase_1_complete": False,
        "promotion_ready": False,
    }


def build_command_timeline_artifact(
    *,
    metadata: dict,
    epoch_raw_inputs: dict,
    input_files: dict,
    source_files: dict,
) -> dict:
    configuration, provenance = _normalize_metadata(metadata)
    expected = expected_epoch_identities()
    raw_epochs = _mapping(epoch_raw_inputs, "raw epoch inputs")
    if list(raw_epochs) != [identity.key for identity in expected]:
        raise ValueError("raw epoch inventory or order is invalid")
    epochs = {
        identity.key: build_epoch_admission(
            identity,
            raw_epochs[identity.key],
        )
        for identity in expected
    }
    _validate_cross_epoch_identity(epochs)
    artifact = _assemble_artifact(
        configuration=configuration,
        provenance=provenance,
        raw_input_files=_normalize_input_files(input_files),
        source_files=_normalize_source_files(source_files),
        epochs=epochs,
    )
    return validate_command_timeline_artifact(artifact)


def validate_command_timeline_artifact(
    artifact: object,
) -> dict:
    row = _mapping(artifact, "command timeline artifact")
    _validate_bounded_json(row, name="command timeline artifact")
    if tuple(row) != TOP_LEVEL_KEYS:
        raise ValueError(
            "artifact top-level keys are not canonical"
        )
    if row.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("artifact schema version is invalid")
    configuration, provenance = _normalize_metadata({
        "configuration": row.get("configuration"),
        "provenance": row.get("provenance"),
    })
    raw_inputs = _normalize_input_files(row.get("raw_input_files"))
    source_files = _normalize_source_files(row.get("source_files"))
    embedded_epochs = _mapping(row.get("epochs"), "artifact epochs")
    expected_identities = expected_epoch_identities()
    if list(embedded_epochs) != [
        identity.key for identity in expected_identities
    ]:
        raise ValueError("artifact epoch inventory or order is invalid")
    recomputed_epochs = {}
    for identity in expected_identities:
        embedded = _mapping(
            embedded_epochs[identity.key],
            "artifact epoch",
        )
        recomputed = build_epoch_admission(
            identity,
            {"worker": embedded.get("worker")},
        )
        if canonical_json_bytes(embedded) != canonical_json_bytes(
            recomputed
        ):
            raise ValueError(
                "artifact epoch recomputation mismatch"
            )
        recomputed_epochs[identity.key] = recomputed
    _validate_cross_epoch_identity(recomputed_epochs)
    expected = _assemble_artifact(
        configuration=configuration,
        provenance=provenance,
        raw_input_files=raw_inputs,
        source_files=source_files,
        epochs=recomputed_epochs,
    )
    if canonical_json_bytes(row) != canonical_json_bytes(expected):
        raise ValueError(
            "artifact derived-field recomputation mismatch"
        )
    if (
        row["runtime_optimization_authorized"]
        is not (
            row["classification"] == "BOUNDARY_LOCALIZED"
        )
    ):
        raise ValueError(
            "runtime optimization authorization is invalid"
        )
    if row["performance_improvement_established"] is not False:
        raise ValueError(
            "performance improvement must remain false"
        )
    return copy.deepcopy(expected)
