from dataclasses import dataclass, replace
from pathlib import Path
import sys
import types

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for package_name in (
    "tinyvllm",
    "tinyvllm.engine",
):
    package = types.ModuleType(package_name)
    package.__path__ = [
        str(ROOT / package_name.replace(".", "/"))
    ]
    sys.modules.setdefault(package_name, package)

from tinyvllm.engine.autoregressive_draft_tp import (
    AutoregressiveDraftRankRegistrationStatus,
    AutoregressiveDraftTensorParallelCoordinator,
)


def _status(
    rank,
    *,
    success=True,
    stage="ready",
    message=None,
):
    return AutoregressiveDraftRankRegistrationStatus(
        rank=rank,
        world_size=4,
        success=success,
        stage=stage,
        error_type=None if success else "RuntimeError",
        message=message,
        target_checkpoint_sha256="target",
        draft_checkpoint_sha256="draft",
        target_tokenizer_sha256="target-tokenizer",
        draft_tokenizer_sha256="draft-tokenizer",
        backend_identity="qwen3",
        executor_id="autoregressive-draft",
        capabilities_sha256="capabilities",
    )


@pytest.mark.parametrize("world_size", (2, 3, 5, 8))
def test_coordinator_rejects_unsupported_world_sizes(world_size):
    with pytest.raises(RuntimeError, match="TP1 or TP4"):
        AutoregressiveDraftTensorParallelCoordinator(
            rank=0,
            world_size=world_size,
            device="cpu",
        )


@pytest.mark.parametrize(
    ("rank", "world_size", "message"),
    (
        (-1, 1, "rank"),
        (1, 1, "rank"),
        (4, 4, "rank"),
        (True, 1, "rank"),
    ),
)
def test_coordinator_rejects_invalid_rank(rank, world_size, message):
    with pytest.raises(ValueError, match=message):
        AutoregressiveDraftTensorParallelCoordinator(
            rank=rank,
            world_size=world_size,
            device="cpu",
        )


def test_registration_collects_exactly_one_status_per_rank():
    statuses = tuple(_status(rank) for rank in range(4))
    coordinator = AutoregressiveDraftTensorParallelCoordinator(
        rank=2,
        world_size=4,
        device="cpu",
        gather_registration_status=lambda local: statuses,
    )

    assert (
        coordinator.collect_registration_status(statuses[2])
        == statuses
    )


def test_tp1_registration_returns_only_local_status():
    status = replace(_status(0), world_size=1)
    coordinator = AutoregressiveDraftTensorParallelCoordinator(
        rank=0,
        world_size=1,
        device="cpu",
        gather_registration_status=lambda _local: (
            pytest.fail("TP1 must not gather registration status")
        ),
    )

    assert coordinator.collect_registration_status(status) == (
        status,
    )


@pytest.mark.parametrize(
    ("statuses", "message"),
    (
        (
            (_status(0), _status(1), _status(2)),
            "exactly world_size",
        ),
        (
            (_status(0), _status(1), _status(1), _status(3)),
            "ranks 0..world_size-1",
        ),
        (
            tuple(
                replace(_status(rank), world_size=1)
                for rank in range(4)
            ),
            "world_size",
        ),
        (
            (_status(0), _status(1), _status(2), object()),
            "registration status",
        ),
    ),
)
def test_registration_rejects_malformed_status_sets(
    statuses,
    message,
):
    coordinator = AutoregressiveDraftTensorParallelCoordinator(
        rank=0,
        world_size=4,
        device="cpu",
        gather_registration_status=lambda local: statuses,
    )
    with pytest.raises(RuntimeError, match=message):
        coordinator.collect_registration_status(_status(0))


def test_registration_rejects_invalid_local_status_before_gather():
    coordinator = AutoregressiveDraftTensorParallelCoordinator(
        rank=0,
        world_size=4,
        device="cpu",
        gather_registration_status=lambda local: pytest.fail(
            "invalid local status must fail before gather"
        ),
    )

    with pytest.raises(ValueError, match="local registration status"):
        coordinator.collect_registration_status(_status(1))


def test_logical_digest_is_stable_for_sorted_dictionary_keys():
    gathered = []

    def gather(local):
        gathered.append(local.detach().clone())
        return tuple(
            local.detach().clone()
            for _ in range(4)
        )

    coordinator = AutoregressiveDraftTensorParallelCoordinator(
        rank=0,
        world_size=4,
        device="cpu",
        gather_digest=gather,
    )
    first = coordinator.assert_logical_authority(
        stage="proposal_preflight",
        rows={"b": [2, 3], "a": {"value": 1}},
    )
    second = coordinator.assert_logical_authority(
        stage="proposal_preflight",
        rows={"a": {"value": 1}, "b": [2, 3]},
    )

    assert first == second
    assert gathered[0].dtype == torch.uint8
    assert gathered[0].shape == (33,)
    assert gathered[0].is_contiguous()
    assert gathered[0][0].item() == 1


@dataclass(frozen=True)
class _LogicalRow:
    sequence_id: int
    proposal_tokens: tuple[int, ...]


def test_logical_encoder_accepts_dataclasses_recursively():
    coordinator = AutoregressiveDraftTensorParallelCoordinator(
        rank=0,
        world_size=1,
        device="cpu",
    )

    dataclass_digest = coordinator.assert_logical_authority(
        stage="materialized",
        rows=_LogicalRow(7, (4, 5, 6)),
    )
    dictionary_digest = coordinator.assert_logical_authority(
        stage="materialized",
        rows={
            "sequence_id": 7,
            "proposal_tokens": (4, 5, 6),
        },
    )

    assert dataclass_digest == dictionary_digest


@pytest.mark.parametrize(
    ("value", "message"),
    (
        (torch.tensor([1]), "tensor"),
        (float("nan"), "finite"),
        (float("inf"), "finite"),
        ({1, 2}, "set"),
        ({1: "value"}, "string keys"),
        (object(), "unsupported"),
    ),
)
def test_logical_encoder_rejects_noncanonical_values(
    value,
    message,
):
    coordinator = AutoregressiveDraftTensorParallelCoordinator(
        rank=0,
        world_size=1,
        device="cpu",
    )
    with pytest.raises((TypeError, ValueError), match=message):
        coordinator.assert_logical_authority(
            stage="invalid",
            rows=value,
        )


@pytest.mark.parametrize("stage", ("", None, 7))
def test_logical_authority_rejects_invalid_stage(stage):
    coordinator = AutoregressiveDraftTensorParallelCoordinator(
        rank=0,
        world_size=1,
        device="cpu",
    )
    with pytest.raises(ValueError, match="stage"):
        coordinator.assert_logical_authority(
            stage=stage,
            rows=(),
        )


def test_physical_ids_are_not_required_for_logical_agreement():
    coordinator = AutoregressiveDraftTensorParallelCoordinator(
        rank=0,
        world_size=1,
        device="cpu",
    )
    left = coordinator.assert_logical_authority(
        stage="materialized",
        rows={
            "sequence_id": 7,
            "proposal_token_ids": (4, 5, 6),
            "staged_entry_count": 2,
        },
    )
    right = coordinator.assert_logical_authority(
        stage="materialized",
        rows={
            "sequence_id": 7,
            "proposal_token_ids": (4, 5, 6),
            "staged_entry_count": 2,
        },
    )

    assert left == right


def test_unequal_success_digest_fails_with_stage_attribution():
    def gather(local):
        rows = [
            local.detach().clone()
            for _ in range(4)
        ]
        rows[3][-1] ^= 1
        return tuple(rows)

    coordinator = AutoregressiveDraftTensorParallelCoordinator(
        rank=1,
        world_size=4,
        device="cpu",
        gather_digest=gather,
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "proposal_materialized.*"
            "logical authority mismatch"
        ),
    ):
        coordinator.assert_logical_authority(
            stage="proposal_materialized",
            rows={"sequence_id": 7},
        )


def test_malformed_digest_gather_result_fails_closed():
    coordinator = AutoregressiveDraftTensorParallelCoordinator(
        rank=0,
        world_size=4,
        device="cpu",
        gather_digest=lambda local: (
            local,
            local,
            local,
        ),
    )

    with pytest.raises(RuntimeError, match="exactly world_size"):
        coordinator.assert_logical_authority(
            stage="proposal_preflight",
            rows={"sequence_id": 7},
        )


def test_peer_failure_forces_successful_rank_to_raise_common_error():
    def gather(local):
        rows = [
            local.detach().clone()
            for _ in range(4)
        ]
        rows[2][0] = 0
        rows[2][1:] = torch.arange(
            32,
            dtype=torch.uint8,
        )
        return tuple(rows)

    coordinator = AutoregressiveDraftTensorParallelCoordinator(
        rank=0,
        world_size=4,
        device="cpu",
        gather_digest=gather,
    )

    with pytest.raises(
        RuntimeError,
        match="bootstrap_prepare.*rank 2",
    ):
        coordinator.converge_stage(
            stage="bootstrap_prepare",
            rows={"sequence_id": 9},
            local_error=None,
        )


def test_local_tp1_failure_is_preserved_as_cause():
    coordinator = AutoregressiveDraftTensorParallelCoordinator(
        rank=0,
        world_size=1,
        device="cpu",
    )
    local_error = ValueError("local bootstrap failure")

    with pytest.raises(
        RuntimeError,
        match="bootstrap_prepare",
    ) as failure:
        coordinator.converge_stage(
            stage="bootstrap_prepare",
            rows={"sequence_id": 9},
            local_error=local_error,
        )

    assert failure.value.__cause__ is local_error
