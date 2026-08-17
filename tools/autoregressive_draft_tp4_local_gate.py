from __future__ import annotations

import math


def _nested(row, *path):
    value = row
    for key in path:
        if not isinstance(value, dict) or key not in value:
            raise ValueError(
                "missing TP4 evidence field "
                + ".".join(path)
            )
        value = value[key]
    return value


def _identical(rows, path, label):
    values = tuple(_nested(row, *path) for row in rows)
    if values[0] in (None, "", (), {}):
        raise ValueError(f"{label} must be non-empty")
    if any(value != values[0] for value in values[1:]):
        raise ValueError(f"{label} mismatch across ranks")
    return values[0]


def _positive_count(value, label):
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError(f"{label} must be positive on every rank")
    return value


def _zero_count(value, label):
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value != 0
    ):
        raise ValueError(f"{label} must be zero after release")


def validate_autoregressive_draft_tp4_local_evidence(
    snapshots: tuple[dict, ...],
) -> dict:
    if not isinstance(snapshots, tuple) or len(snapshots) != 4:
        raise ValueError(
            "TP4 local evidence requires exactly four snapshots"
        )
    if any(not isinstance(row, dict) for row in snapshots):
        raise ValueError("TP4 snapshots must be dictionaries")
    try:
        ordered = tuple(
            sorted(snapshots, key=lambda row: row["rank"])
        )
    except (KeyError, TypeError) as error:
        raise ValueError("TP4 snapshot rank is invalid") from error
    if tuple(row.get("rank") for row in ordered) != (0, 1, 2, 3):
        raise ValueError(
            "TP4 local evidence requires exactly four "
            "distinct ranks 0..3"
        )
    if any(row.get("world_size") != 4 for row in ordered):
        raise ValueError("TP4 snapshot world_size must equal four")
    if any(row.get("registered") is not True for row in ordered):
        raise ValueError(
            "TP4 autoregressive draft must be registered "
            "on every rank"
        )
    if any(
        row.get("registration_error") is not None
        for row in ordered
    ):
        raise ValueError(
            "TP4 autoregressive draft registration error is present"
        )

    consensus = _identical(
        ordered,
        ("registration_consensus_sha256",),
        "registration consensus",
    )
    for owner in ("target", "draft"):
        _identical(
            ordered,
            (
                "checkpoint_identity",
                owner,
                "composite_sha256",
            ),
            f"{owner} checkpoint identity",
        )
        _identical(
            ordered,
            (
                "tokenizer_contract",
                owner,
                "composite_sha256",
            ),
            f"{owner} tokenizer identity",
        )
    _identical(
        ordered,
        ("executor_descriptor", "executor_id"),
        "executor ID",
    )
    _identical(
        ordered,
        ("executor_descriptor", "capabilities"),
        "logical capabilities",
    )
    _identical(
        ordered,
        ("executor", "backend_identity"),
        "backend identity",
    )
    logical_rows = _identical(
        ordered,
        ("executor", "logical_authority_rows"),
        "logical authority",
    )
    last_digest = _identical(
        ordered,
        ("executor", "last_logical_authority_sha256"),
        "logical authority digest",
    )
    if not isinstance(last_digest, str) or len(last_digest) != 64:
        raise ValueError(
            "logical authority digest must be SHA-256"
        )

    storage_ids = []
    total_bytes = 0
    timing_rows = []
    for row in ordered:
        executor = _nested(row, "executor")
        if executor.get("rank") != row["rank"]:
            raise ValueError("executor rank topology mismatch")
        if executor.get("world_size") != 4:
            raise ValueError("executor world_size topology mismatch")
        if executor.get(
            "logical_authority_digest_count"
        ) != len(logical_rows):
            raise ValueError(
                "logical authority digest count mismatch"
            )
        backend = _nested(executor, "backend")
        _positive_count(
            backend.get("local_prefill_forward_count"),
            "prefill forward count",
        )
        _positive_count(
            backend.get("local_decode_forward_count"),
            "decode forward count",
        )
        local_bytes = backend.get("local_proposal_kv_bytes")
        if (
            isinstance(local_bytes, bool)
            or not isinstance(local_bytes, int)
            or local_bytes <= 0
        ):
            raise ValueError(
                "local proposal KV bytes must be positive"
            )
        total_bytes += local_bytes
        storage_id = backend.get("proposal_kv_storage_id")
        if not isinstance(storage_id, str) or not storage_id:
            raise ValueError(
                "proposal KV storage identity must be non-empty"
            )
        storage_ids.append(storage_id)
        allocator = _nested(
            backend,
            "proposal_kv_cache",
            "entry_allocator",
        )
        if allocator.get("allocator_mode") != "direct":
            raise ValueError(
                "TP4 local gate allocator mode must be direct"
            )
        for name, label in (
            (
                "accepted_entry_copy_count",
                "accepted proposal KV copy",
            ),
            (
                "accepted_entry_replay_count",
                "accepted proposal KV replay",
            ),
            (
                "accepted_entry_rematerialization_count",
                "accepted proposal KV rematerialization",
            ),
        ):
            _zero_count(allocator.get(name), label)
        lifecycle = _nested(
            executor,
            "proposal_kv_lifecycle",
        )
        _zero_count(
            lifecycle.get("active_transaction_count"),
            "active transaction count",
        )
        _zero_count(
            lifecycle.get("prepared_ticket_count"),
            "active prepared ticket count",
        )
        _zero_count(
            _nested(
                lifecycle,
                "proposal_kv_cache",
                "owned_entry_count",
            ),
            "live proposal entry count",
        )
        _zero_count(
            _nested(
                backend,
                "physical_store",
                "allocated_slot_count",
            ),
            "live physical slot count",
        )
        timing = executor.get("timing_ms")
        if not isinstance(timing, dict) or not timing:
            raise ValueError("rank timing evidence is unavailable")
        timing_rows.append(timing)

    if len(set(storage_ids)) != 4:
        raise ValueError(
            "proposal KV storage identities must be distinct"
        )
    timing_names = tuple(sorted(timing_rows[0]))
    if any(tuple(sorted(row)) != timing_names for row in timing_rows):
        raise ValueError("rank timing keys mismatch")
    max_timing = {}
    for name in timing_names:
        values = tuple(row[name] for row in timing_rows)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value < 0
            for value in values
        ):
            raise ValueError("rank timing values must be finite")
        max_timing[name] = max(float(value) for value in values)

    return {
        "schema_version": 1,
        "rank_count": 4,
        "registration_consensus_sha256": consensus,
        "total_proposal_kv_bytes": total_bytes,
        "max_rank_timing_ms": max_timing,
        "classification": "NOT_PROMOTABLE",
        "promotion_boundary": {
            "real_checkpoint_tp4": "NOT_ESTABLISHED",
            "second_learned_structure": "NOT_ESTABLISHED",
            "contexts_4k_16k_32k": "NOT_ESTABLISHED",
            "performance": "NOT_ESTABLISHED",
            "real_kv_movement": "NOT_ESTABLISHED",
            "phase_1": "NOT_ACHIEVED",
        },
    }
