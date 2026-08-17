from __future__ import annotations

from dataclasses import asdict, replace
import json
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
    "tinyvllm.speculative",
):
    package = types.ModuleType(package_name)
    package.__path__ = [
        str(ROOT / package_name.replace(".", "/"))
    ]
    sys.modules.setdefault(package_name, package)

from tinyvllm.engine.autoregressive_draft_registration import (
    AutoregressiveDraftRegistrationCandidate,
    CheckpointFingerprint,
    TokenizerContract,
    build_autoregressive_draft_registration_status,
    build_checkpoint_fingerprint,
    build_tokenizer_contract,
    validate_autoregressive_draft_registration_consensus,
    validate_tokenizer_compatibility,
)
from tinyvllm.engine.speculative_runtime import (
    ModelRunnerProposalExecutorDescriptor,
)
from tinyvllm.engine.speculative_proposal_executor import (
    assert_tensor_free,
)
from tinyvllm.speculative.adapter import DraftCapabilities


class _FakeBackend:
    backend_identity = "qwen3-autoregressive-draft"


class _FakeExecutor:
    capabilities = DraftCapabilities(
        source_type="independent_draft_model",
        supports_batch=True,
        requires_target_hidden=False,
        requires_target_logits=False,
        max_proposal_tokens=4,
        execution_domain="model_runner",
        requires_proposal_lifecycle=True,
        requires_full_token_history=False,
    )


def _checkpoint(label):
    return CheckpointFingerprint(
        model_path=f"/{label}",
        config_sha256="a" * 64,
        shard_sha256=(("model.safetensors", "b" * 64),),
        composite_sha256=(
            "c" * 63 + ("0" if label == "target" else "1")
        ),
    )


def _tokenizer(label):
    return TokenizerContract(
        model_path=f"/{label}",
        tokenizer_class="FakeTokenizer",
        normalization_sha256="d" * 64,
        ordered_token_to_id_sha256="e" * 64,
        vocab_size=8,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=2,
        stop_token_ids=(1,),
        artifact_sha256=(),
        composite_sha256=(
            "f" * 63 + (
                "0" if label == "target-tokenizer" else "1"
            )
        ),
    )


def _candidate():
    return AutoregressiveDraftRegistrationCandidate(
        target_checkpoint=_checkpoint("target"),
        draft_checkpoint=_checkpoint("draft"),
        target_tokenizer_contract=_tokenizer(
            "target-tokenizer"
        ),
        draft_tokenizer_contract=_tokenizer(
            "draft-tokenizer"
        ),
        model=object(),
        physical_store=object(),
        proposal_kv_cache=object(),
        backend=_FakeBackend(),
        executor=_FakeExecutor(),
        descriptor=ModelRunnerProposalExecutorDescriptor(
            executor_id="autoregressive-draft",
            capabilities=_FakeExecutor.capabilities,
        ),
    )


def _successful_status(rank):
    return build_autoregressive_draft_registration_status(
        rank=rank,
        world_size=4,
        stage="ready",
        candidate=_candidate(),
        error=None,
    )


class _FakeTokenizer:

    def __init__(
        self,
        *,
        vocab,
        init_kwargs=None,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=2,
    ):
        self._vocab = dict(vocab)
        self.init_kwargs = dict(init_kwargs or {})
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

    def get_vocab(self):
        return dict(self._vocab)


class _OtherFakeTokenizer(_FakeTokenizer):
    pass


def _write_checkpoint(
    path: Path,
    *,
    config=b'{"model_type":"qwen3"}',
    shards=None,
):
    path.mkdir()
    (path / "config.json").write_bytes(config)
    for name, payload in (
        shards
        if shards is not None
        else (
            ("model-00002-of-00002.safetensors", b"second"),
            ("model-00001-of-00002.safetensors", b"first"),
        )
    ):
        (path / name).write_bytes(payload)


def _tokenizer_contract(
    path: Path,
    *,
    vocab=None,
    tokenizer_class=_FakeTokenizer,
    init_kwargs=None,
    bos_token_id=0,
    eos_token_id=1,
    pad_token_id=2,
    stop_token_ids=(1,),
    artifacts=None,
):
    path.mkdir()
    for name, payload in (artifacts or {}).items():
        (path / name).write_bytes(payload)
    tokenizer = tokenizer_class(
        vocab=vocab or {"a": 0, "b": 1, "<pad>": 2},
        init_kwargs=init_kwargs,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )
    return build_tokenizer_contract(
        path,
        tokenizer,
        stop_token_ids=stop_token_ids,
    )


def test_checkpoint_fingerprint_covers_config_and_sorted_shards(
    tmp_path,
):
    model_path = tmp_path / "model"
    _write_checkpoint(model_path)

    fingerprint = build_checkpoint_fingerprint(model_path)

    assert isinstance(fingerprint, CheckpointFingerprint)
    assert fingerprint.model_path == str(model_path.resolve())
    assert tuple(
        name for name, _ in fingerprint.shard_sha256
    ) == (
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    )
    assert len(fingerprint.config_sha256) == 64
    assert len(fingerprint.composite_sha256) == 64


@pytest.mark.parametrize(
    ("changed_name", "changed_payload"),
    (
        ("config.json", b'{"model_type":"changed"}'),
        ("model-00001-of-00002.safetensors", b"changed"),
        ("model-00002-of-00002.safetensors", b"changed"),
    ),
)
def test_checkpoint_file_byte_change_changes_composite_hash(
    tmp_path,
    changed_name,
    changed_payload,
):
    first_path = tmp_path / "first"
    second_path = tmp_path / "second"
    _write_checkpoint(first_path)
    _write_checkpoint(second_path)
    (second_path / changed_name).write_bytes(changed_payload)

    first = build_checkpoint_fingerprint(first_path)
    second = build_checkpoint_fingerprint(second_path)

    assert first.composite_sha256 != second.composite_sha256


@pytest.mark.parametrize(
    ("create_config", "create_shard", "message"),
    (
        (False, True, "config.json"),
        (True, False, "safetensors"),
    ),
)
def test_incomplete_checkpoint_is_rejected(
    tmp_path,
    create_config,
    create_shard,
    message,
):
    model_path = tmp_path / "model"
    model_path.mkdir()
    if create_config:
        (model_path / "config.json").write_bytes(b"{}")
    if create_shard:
        (model_path / "model.safetensors").write_bytes(b"weights")

    with pytest.raises(ValueError, match=message):
        build_checkpoint_fingerprint(model_path)


def test_same_size_different_token_order_is_rejected(tmp_path):
    target = _tokenizer_contract(
        tmp_path / "target",
        vocab={"a": 0, "b": 1},
        eos_token_id=1,
        pad_token_id=None,
    )
    draft = _tokenizer_contract(
        tmp_path / "draft",
        vocab={"b": 0, "a": 1},
        eos_token_id=1,
        pad_token_id=None,
    )

    with pytest.raises(ValueError, match="ordered token-to-ID"):
        validate_tokenizer_compatibility(target, draft)


def test_exact_tokenizer_contract_passes(tmp_path):
    artifacts = {
        "tokenizer.json": b'{"version":"1.0"}',
        "merges.txt": b"a b\n",
    }
    target = _tokenizer_contract(
        tmp_path / "target",
        artifacts=artifacts,
        init_kwargs={"clean_up_tokenization_spaces": False},
    )
    draft = _tokenizer_contract(
        tmp_path / "draft",
        artifacts=artifacts,
        init_kwargs={"clean_up_tokenization_spaces": False},
    )

    validate_tokenizer_compatibility(target, draft)

    assert isinstance(target, TokenizerContract)
    assert target.vocab_size == 3
    assert tuple(name for name, _ in target.artifact_sha256) == (
        "merges.txt",
        "tokenizer.json",
    )


def test_tokenizer_contract_ignores_local_artifact_paths(tmp_path):
    artifacts = {
        "merges.txt": b"a b\n",
        "vocab.json": b'{"a":0,"b":1}',
    }
    target_path = tmp_path / "target"
    draft_path = tmp_path / "draft"
    target = _tokenizer_contract(
        target_path,
        artifacts=artifacts,
        init_kwargs={
            "name_or_path": str(target_path),
            "merges_file": str(target_path / "merges.txt"),
            "vocab_file": str(target_path / "vocab.json"),
            "clean_up_tokenization_spaces": False,
        },
    )
    draft = _tokenizer_contract(
        draft_path,
        artifacts=artifacts,
        init_kwargs={
            "name_or_path": str(draft_path),
            "merges_file": str(draft_path / "merges.txt"),
            "vocab_file": str(draft_path / "vocab.json"),
            "clean_up_tokenization_spaces": False,
        },
    )

    validate_tokenizer_compatibility(target, draft)

    assert (
        target.normalization_sha256
        == draft.normalization_sha256
    )


@pytest.mark.parametrize(
    ("field", "target_value", "draft_value"),
    (
        ("bos_token_id", 0, 4),
        ("eos_token_id", 1, (1, 4)),
        ("pad_token_id", 2, None),
        ("stop_token_ids", (1,), (1, 4)),
    ),
)
def test_special_token_mismatch_is_rejected(
    tmp_path,
    field,
    target_value,
    draft_value,
):
    target_kwargs = {}
    draft_kwargs = {}
    if field == "stop_token_ids":
        target_kwargs[field] = target_value
        draft_kwargs[field] = draft_value
    else:
        target_kwargs[field] = target_value
        draft_kwargs[field] = draft_value
    target = _tokenizer_contract(
        tmp_path / "target",
        **target_kwargs,
    )
    draft = _tokenizer_contract(
        tmp_path / "draft",
        **draft_kwargs,
    )

    with pytest.raises(ValueError, match=field):
        validate_tokenizer_compatibility(target, draft)


def test_tokenizer_class_mismatch_is_rejected(tmp_path):
    target = _tokenizer_contract(tmp_path / "target")
    draft = _tokenizer_contract(
        tmp_path / "draft",
        tokenizer_class=_OtherFakeTokenizer,
    )

    with pytest.raises(ValueError, match="tokenizer_class"):
        validate_tokenizer_compatibility(target, draft)


def test_normalization_configuration_mismatch_is_rejected(
    tmp_path,
):
    target = _tokenizer_contract(
        tmp_path / "target",
        init_kwargs={"normalizer": {"lowercase": False}},
    )
    draft = _tokenizer_contract(
        tmp_path / "draft",
        init_kwargs={"normalizer": {"lowercase": True}},
    )

    with pytest.raises(ValueError, match="normalization_sha256"):
        validate_tokenizer_compatibility(target, draft)


def test_common_tokenizer_artifact_mismatch_is_rejected(tmp_path):
    target = _tokenizer_contract(
        tmp_path / "target",
        artifacts={"tokenizer.json": b"target"},
    )
    draft = _tokenizer_contract(
        tmp_path / "draft",
        artifacts={"tokenizer.json": b"draft"},
    )

    with pytest.raises(
        ValueError,
        match=r"artifact_sha256.*tokenizer\.json",
    ):
        validate_tokenizer_compatibility(target, draft)


def test_missing_optional_artifact_does_not_replace_mapping_evidence(
    tmp_path,
):
    target = _tokenizer_contract(
        tmp_path / "target",
        artifacts={"tokenizer.json": b"target-only"},
    )
    draft = _tokenizer_contract(tmp_path / "draft")

    validate_tokenizer_compatibility(target, draft)

    assert target.artifact_sha256
    assert draft.artifact_sha256 == ()
    incompatible = _tokenizer_contract(
        tmp_path / "incompatible",
        vocab={"b": 0, "a": 1, "<pad>": 2},
    )
    with pytest.raises(ValueError, match="ordered token-to-ID"):
        validate_tokenizer_compatibility(target, incompatible)


def test_identity_snapshots_are_tensor_free_and_json_serializable(
    tmp_path,
):
    checkpoint_path = tmp_path / "checkpoint"
    tokenizer_path = tmp_path / "tokenizer"
    _write_checkpoint(checkpoint_path)
    checkpoint = build_checkpoint_fingerprint(checkpoint_path)
    tokenizer = _tokenizer_contract(tokenizer_path)
    snapshot = {
        "checkpoint": asdict(checkpoint),
        "tokenizer": asdict(tokenizer),
    }

    assert_tensor_free(snapshot, name="registration snapshot")
    json.dumps(snapshot)
    assert isinstance(snapshot["checkpoint"]["model_path"], str)
    assert isinstance(snapshot["tokenizer"]["model_path"], str)
    assert not any(
        isinstance(value, torch.Tensor)
        for row in snapshot.values()
        for value in row.values()
    )


def test_registration_candidate_preserves_private_objects():
    candidate = _candidate()

    assert candidate.descriptor.executor_id == (
        "autoregressive-draft"
    )
    assert candidate.model is not None
    assert candidate.physical_store is not None
    assert candidate.proposal_kv_cache is not None
    assert candidate.executor is not None


def test_registration_status_hashes_capabilities_deterministically():
    first = build_autoregressive_draft_registration_status(
        rank=0,
        world_size=4,
        stage="ready",
        candidate=_candidate(),
        error=None,
    )
    second = build_autoregressive_draft_registration_status(
        rank=3,
        world_size=4,
        stage="ready",
        candidate=_candidate(),
        error=None,
    )

    assert first.capabilities_sha256 == (
        second.capabilities_sha256
    )
    assert len(first.capabilities_sha256) == 64


def test_failed_registration_status_omits_partial_identities():
    error = RuntimeError("draft shard construction failed")

    status = build_autoregressive_draft_registration_status(
        rank=2,
        world_size=4,
        stage="build_backend",
        candidate=_candidate(),
        error=error,
    )

    assert status.rank == 2
    assert status.world_size == 4
    assert status.success is False
    assert status.stage == "build_backend"
    assert status.error_type == "RuntimeError"
    assert status.message == "draft shard construction failed"
    assert status.target_checkpoint_sha256 is None
    assert status.draft_checkpoint_sha256 is None
    assert status.target_tokenizer_sha256 is None
    assert status.draft_tokenizer_sha256 is None
    assert status.backend_identity is None
    assert status.executor_id is None
    assert status.capabilities_sha256 is None


def test_matching_four_rank_statuses_return_one_consensus_hash():
    statuses = tuple(
        _successful_status(rank) for rank in range(4)
    )

    consensus = (
        validate_autoregressive_draft_registration_consensus(
            statuses,
            world_size=4,
        )
    )

    assert len(consensus) == 64


@pytest.mark.parametrize(
    "field",
    (
        "target_checkpoint_sha256",
        "draft_checkpoint_sha256",
        "target_tokenizer_sha256",
        "draft_tokenizer_sha256",
        "backend_identity",
        "executor_id",
        "capabilities_sha256",
    ),
)
def test_one_rank_identity_mismatch_rejects_consensus(field):
    statuses = [
        _successful_status(rank) for rank in range(4)
    ]
    statuses[2] = replace(
        statuses[2],
        **{field: "mismatch"},
    )

    with pytest.raises(RuntimeError, match=field):
        validate_autoregressive_draft_registration_consensus(
            tuple(statuses),
            world_size=4,
        )


def test_failed_rank_status_names_rank_and_stage():
    statuses = [
        _successful_status(rank) for rank in range(4)
    ]
    statuses[2] = (
        build_autoregressive_draft_registration_status(
            rank=2,
            world_size=4,
            stage="load_weights",
            candidate=None,
            error=RuntimeError("shard load failed"),
        )
    )

    with pytest.raises(
        RuntimeError,
        match=r"rank 2.*load_weights",
    ):
        validate_autoregressive_draft_registration_consensus(
            tuple(statuses),
            world_size=4,
        )
