from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from tinyvllm.engine.autoregressive_draft_tp import (
    AutoregressiveDraftRankRegistrationStatus,
)


_TOKENIZER_ARTIFACT_NAMES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
)


@dataclass(frozen=True)
class CheckpointFingerprint:
    model_path: str
    config_sha256: str
    shard_sha256: tuple[tuple[str, str], ...]
    composite_sha256: str


@dataclass(frozen=True)
class TokenizerContract:
    model_path: str
    tokenizer_class: str
    normalization_sha256: str
    ordered_token_to_id_sha256: str
    vocab_size: int
    bos_token_id: int | None
    eos_token_id: int | tuple[int, ...] | None
    pad_token_id: int | None
    stop_token_ids: tuple[int, ...]
    artifact_sha256: tuple[tuple[str, str], ...]
    composite_sha256: str


@dataclass(frozen=True)
class AutoregressiveDraftRegistrationError:
    stage: str
    error_type: str
    message: str


@dataclass(frozen=True)
class AutoregressiveDraftRegistrationDependencies:
    build_checkpoint_fingerprint: object
    load_tokenizer: object
    build_tokenizer_contract: object
    validate_tokenizer_compatibility: object
    load_hf_config: object
    build_model: object
    load_weights: object
    move_model_to_target: object
    build_proposal_kv_allocator: object
    build_proposal_kv_cache: object
    build_backend: object
    build_graph_components: object
    build_executor: object
    build_descriptor: object


@dataclass(frozen=True)
class AutoregressiveDraftGraphComponents:
    scratch_cache: object
    scratch_owner: object
    backend: object
    runner: object


@dataclass(frozen=True)
class AutoregressiveDraftRegistrationCandidate:
    target_checkpoint: CheckpointFingerprint
    draft_checkpoint: CheckpointFingerprint
    target_tokenizer_contract: TokenizerContract
    draft_tokenizer_contract: TokenizerContract
    model: object
    physical_store: object
    proposal_kv_cache: object
    backend: object
    executor: object
    descriptor: object
    graph_components: (
        AutoregressiveDraftGraphComponents | None
    ) = None


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_value(value: Any):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(
                value.items(),
                key=lambda row: str(row[0]),
            )
        }
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        rows = [_canonical_value(item) for item in value]
        return sorted(rows, key=lambda item: repr(item))
    getstate = getattr(value, "__getstate__", None)
    if callable(getstate):
        state = getstate()
        if state is not None and state is not value:
            return {
                "__class__": (
                    f"{type(value).__module__}."
                    f"{type(value).__qualname__}"
                ),
                "state": _canonical_value(state),
            }
    return {
        "__class__": (
            f"{type(value).__module__}.{type(value).__qualname__}"
        ),
        "value": str(value),
    }


def _canonical_json(payload: Any) -> bytes:
    return json.dumps(
        _canonical_value(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _hash_payload(payload: Any) -> str:
    return _sha256_bytes(_canonical_json(payload))


def build_autoregressive_draft_registration_status(
    *,
    rank: int,
    world_size: int,
    stage: str,
    candidate: (
        AutoregressiveDraftRegistrationCandidate | None
    ),
    error: BaseException | None,
) -> AutoregressiveDraftRankRegistrationStatus:
    if error is not None:
        return AutoregressiveDraftRankRegistrationStatus(
            rank=rank,
            world_size=world_size,
            success=False,
            stage=stage,
            error_type=type(error).__name__,
            message=str(error),
            target_checkpoint_sha256=None,
            draft_checkpoint_sha256=None,
            target_tokenizer_sha256=None,
            draft_tokenizer_sha256=None,
            backend_identity=None,
            executor_id=None,
            capabilities_sha256=None,
        )
    if candidate is None:
        raise ValueError(
            "successful registration status requires candidate"
        )
    return AutoregressiveDraftRankRegistrationStatus(
        rank=rank,
        world_size=world_size,
        success=True,
        stage=stage,
        error_type=None,
        message=None,
        target_checkpoint_sha256=(
            candidate.target_checkpoint.composite_sha256
        ),
        draft_checkpoint_sha256=(
            candidate.draft_checkpoint.composite_sha256
        ),
        target_tokenizer_sha256=(
            candidate.target_tokenizer_contract.composite_sha256
        ),
        draft_tokenizer_sha256=(
            candidate.draft_tokenizer_contract.composite_sha256
        ),
        backend_identity=candidate.backend.backend_identity,
        executor_id=candidate.descriptor.executor_id,
        capabilities_sha256=_hash_payload(
            asdict(candidate.descriptor.capabilities)
        ),
    )


def validate_autoregressive_draft_registration_consensus(
    statuses: tuple[
        AutoregressiveDraftRankRegistrationStatus,
        ...,
    ],
    *,
    world_size: int,
) -> str:
    if (
        isinstance(world_size, bool)
        or not isinstance(world_size, int)
        or world_size <= 0
    ):
        raise ValueError("world_size must be a positive integer")
    if not isinstance(statuses, tuple):
        raise ValueError(
            "registration statuses must be a tuple"
        )
    if len(statuses) != world_size:
        raise RuntimeError(
            "registration status count must equal world_size"
        )
    if any(
        not isinstance(
            status,
            AutoregressiveDraftRankRegistrationStatus,
        )
        for status in statuses
    ):
        raise RuntimeError(
            "registration status row type is invalid"
        )
    ordered = tuple(sorted(statuses, key=lambda row: row.rank))
    if tuple(row.rank for row in ordered) != tuple(
        range(world_size)
    ):
        raise RuntimeError(
            "registration statuses must contain ranks "
            "0..world_size-1 exactly once"
        )
    for status in ordered:
        if status.world_size != world_size:
            raise RuntimeError(
                "registration status world_size mismatch"
            )
        if not status.success:
            raise RuntimeError(
                "autoregressive draft registration failed on "
                f"rank {status.rank} at stage {status.stage}: "
                f"{status.error_type}: {status.message}"
            )

    identity_fields = (
        "target_checkpoint_sha256",
        "draft_checkpoint_sha256",
        "target_tokenizer_sha256",
        "draft_tokenizer_sha256",
        "backend_identity",
        "executor_id",
        "capabilities_sha256",
    )
    reference = tuple(
        getattr(ordered[0], field)
        for field in identity_fields
    )
    for field, value in zip(identity_fields, reference):
        if not isinstance(value, str) or not value:
            raise RuntimeError(
                f"successful registration status missing {field}"
            )
    for status in ordered[1:]:
        for field, expected in zip(
            identity_fields,
            reference,
        ):
            if getattr(status, field) != expected:
                raise RuntimeError(
                    "autoregressive draft registration "
                    f"{field} mismatch across ranks"
                )
    return _hash_payload(reference)


def _resolved_directory(model_path) -> Path:
    path = Path(model_path).expanduser().resolve()
    if not path.is_dir():
        raise ValueError("model_path must be an existing directory")
    return path


def build_checkpoint_fingerprint(
    model_path,
) -> CheckpointFingerprint:
    path = _resolved_directory(model_path)
    config_path = path / "config.json"
    if not config_path.is_file():
        raise ValueError("checkpoint must contain config.json")
    shard_paths = tuple(sorted(path.glob("*.safetensors")))
    if not shard_paths:
        raise ValueError(
            "checkpoint must contain at least one .safetensors shard"
        )
    config_sha256 = _sha256_bytes(config_path.read_bytes())
    shard_sha256 = tuple(
        (shard_path.name, _sha256_bytes(shard_path.read_bytes()))
        for shard_path in shard_paths
    )
    composite_sha256 = _hash_payload({
        "config_sha256": config_sha256,
        "shard_sha256": shard_sha256,
    })
    return CheckpointFingerprint(
        model_path=str(path),
        config_sha256=config_sha256,
        shard_sha256=shard_sha256,
        composite_sha256=composite_sha256,
    )


def _optional_token_id(value, name: str) -> int | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(
            f"{name} must be a nonnegative integer or None"
        )
    return value


def _eos_token_id(
    value,
) -> int | tuple[int, ...] | None:
    if value is None or isinstance(value, int):
        return _optional_token_id(value, "eos_token_id")
    if not isinstance(value, (tuple, list)) or not value:
        raise ValueError(
            "eos_token_id must be an integer, non-empty tuple, or None"
        )
    normalized = tuple(
        _optional_token_id(item, "eos_token_id")
        for item in value
    )
    if len(set(normalized)) != len(normalized):
        raise ValueError("eos_token_id must not contain duplicates")
    return normalized


def _stop_token_ids(value) -> tuple[int, ...]:
    if not isinstance(value, (tuple, list, set, frozenset)):
        raise ValueError(
            "stop_token_ids must be a tuple, list, or set"
        )
    normalized = tuple(
        _optional_token_id(item, "stop_token_ids")
        for item in value
    )
    return tuple(sorted(set(normalized)))


def _ordered_vocabulary(tokenizer) -> tuple[tuple[int, str], ...]:
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if not callable(get_vocab):
        raise ValueError("tokenizer must expose callable get_vocab")
    vocabulary = get_vocab()
    if not isinstance(vocabulary, dict) or not vocabulary:
        raise ValueError("tokenizer vocabulary must be a non-empty dict")
    rows = []
    for token, token_id in vocabulary.items():
        if not isinstance(token, str):
            raise ValueError("tokenizer vocabulary tokens must be strings")
        if (
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or token_id < 0
        ):
            raise ValueError(
                "tokenizer vocabulary IDs must be nonnegative integers"
            )
        rows.append((token_id, token))
    if len({token_id for token_id, _ in rows}) != len(rows):
        raise ValueError("tokenizer vocabulary IDs must be unique")
    return tuple(sorted(rows))


def _tokenizer_normalization_configuration(tokenizer) -> dict:
    configuration = dict(
        getattr(tokenizer, "init_kwargs", {})
    )
    configuration.pop("name_or_path", None)
    for key, value in tuple(configuration.items()):
        if (
            key.endswith("_file")
            and isinstance(value, (str, Path))
        ):
            configuration[key] = Path(value).name
    return configuration


def build_tokenizer_contract(
    model_path,
    tokenizer,
    *,
    stop_token_ids=(),
) -> TokenizerContract:
    path = _resolved_directory(model_path)
    ordered_vocabulary = _ordered_vocabulary(tokenizer)
    tokenizer_class = (
        f"{type(tokenizer).__module__}."
        f"{type(tokenizer).__qualname__}"
    )
    normalization_sha256 = _hash_payload(
        _tokenizer_normalization_configuration(tokenizer)
    )
    ordered_token_to_id_sha256 = _hash_payload(
        ordered_vocabulary
    )
    bos_token_id = _optional_token_id(
        getattr(tokenizer, "bos_token_id", None),
        "bos_token_id",
    )
    eos_token_id = _eos_token_id(
        getattr(tokenizer, "eos_token_id", None)
    )
    pad_token_id = _optional_token_id(
        getattr(tokenizer, "pad_token_id", None),
        "pad_token_id",
    )
    normalized_stop_token_ids = _stop_token_ids(stop_token_ids)
    artifact_sha256 = tuple(
        (
            artifact_name,
            _sha256_bytes(artifact_path.read_bytes()),
        )
        for artifact_name in sorted(_TOKENIZER_ARTIFACT_NAMES)
        if (artifact_path := path / artifact_name).is_file()
    )
    payload = {
        "tokenizer_class": tokenizer_class,
        "normalization_sha256": normalization_sha256,
        "ordered_token_to_id_sha256": (
            ordered_token_to_id_sha256
        ),
        "vocab_size": len(ordered_vocabulary),
        "bos_token_id": bos_token_id,
        "eos_token_id": eos_token_id,
        "pad_token_id": pad_token_id,
        "stop_token_ids": normalized_stop_token_ids,
        "artifact_sha256": artifact_sha256,
    }
    return TokenizerContract(
        model_path=str(path),
        composite_sha256=_hash_payload(payload),
        **payload,
    )


def validate_tokenizer_compatibility(
    target: TokenizerContract,
    draft: TokenizerContract,
) -> None:
    if not isinstance(target, TokenizerContract):
        raise ValueError("target must be a TokenizerContract")
    if not isinstance(draft, TokenizerContract):
        raise ValueError("draft must be a TokenizerContract")
    fields = (
        "tokenizer_class",
        "normalization_sha256",
        "ordered_token_to_id_sha256",
        "vocab_size",
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "stop_token_ids",
    )
    for field in fields:
        if getattr(target, field) != getattr(draft, field):
            label = (
                "ordered token-to-ID"
                if field == "ordered_token_to_id_sha256"
                else field
            )
            raise ValueError(
                f"tokenizer contract mismatch: {label}"
            )
    target_artifacts = dict(target.artifact_sha256)
    draft_artifacts = dict(draft.artifact_sha256)
    for artifact_name in sorted(
        set(target_artifacts).intersection(draft_artifacts)
    ):
        if (
            target_artifacts[artifact_name]
            != draft_artifacts[artifact_name]
        ):
            raise ValueError(
                "tokenizer contract mismatch: "
                f"artifact_sha256 {artifact_name}"
            )


def _tokenizer_stop_token_ids(tokenizer) -> tuple[int, ...]:
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        return ()
    if isinstance(eos_token_id, int):
        return (eos_token_id,)
    return tuple(eos_token_id)


def build_autoregressive_draft_registration_dependencies(
) -> AutoregressiveDraftRegistrationDependencies:
    import torch
    from transformers import AutoConfig, AutoTokenizer

    from tinyvllm.engine.autoregressive_draft_executor import (
        AutoregressiveDraftProposalExecutor,
    )
    from tinyvllm.engine.autoregressive_draft_graph import (
        AutoregressiveDraftExactGraphRunner,
    )
    from tinyvllm.engine.proposal_kv_allocator import (
        DirectProposalKVAllocator,
    )
    from tinyvllm.engine.proposal_kv_cache import ProposalKVCache
    from tinyvllm.engine.qwen3_draft_backend import (
        Qwen3AutoregressiveDraftBackend,
    )
    from tinyvllm.engine.qwen3_draft_cuda_graph_backend import (
        Qwen3DraftCudaGraphBackend,
    )
    from tinyvllm.engine.qwen3_draft_graph_scratch import (
        Qwen3DraftGraphScratchOwner,
    )
    from tinyvllm.engine.qwen3_draft_proposal_kv import (
        build_qwen3_draft_proposal_kv_allocator,
    )
    from tinyvllm.engine.speculative_runtime import (
        ModelRunnerProposalExecutorDescriptor,
    )
    from tinyvllm.models.qwen3 import Qwen3ForCausalLM
    from tinyvllm.utils.loader import load_model

    def load_tokenizer(path):
        return AutoTokenizer.from_pretrained(path, use_fast=True)

    def tokenizer_contract(path, tokenizer):
        return build_tokenizer_contract(
            path,
            tokenizer,
            stop_token_ids=_tokenizer_stop_token_ids(tokenizer),
        )

    def move_model_to_target(model, target_model):
        try:
            target_parameter = next(target_model.parameters())
        except (AttributeError, StopIteration) as error:
            raise RuntimeError(
                "target model device and dtype are unavailable"
            ) from error
        model.to(
            device=target_parameter.device,
            dtype=target_parameter.dtype,
        )
        model.eval()
        return target_parameter.device, target_parameter.dtype

    def build_descriptor(executor):
        return ModelRunnerProposalExecutorDescriptor(
            executor_id="autoregressive-draft",
            capabilities=executor.capabilities,
        )

    def build_model(
        hf_config,
        *,
        tensor_parallel_rank,
        tensor_parallel_size,
    ):
        return Qwen3ForCausalLM(hf_config)

    def build_backend(
        *,
        model,
        proposal_kv_cache,
        backend_identity,
        model_fingerprint,
        tokenizer_fingerprint,
        tensor_parallel_rank,
        tensor_parallel_size,
    ):
        return Qwen3AutoregressiveDraftBackend(
            model=model,
            proposal_kv_cache=proposal_kv_cache,
            backend_identity=backend_identity,
            model_fingerprint=model_fingerprint,
            tokenizer_fingerprint=tokenizer_fingerprint,
            tensor_parallel_rank=tensor_parallel_rank,
            tensor_parallel_size=tensor_parallel_size,
        )

    def build_graph_components(
        *,
        config,
        backend,
        proposal_kv_cache,
        physical_store,
        device,
        dtype,
    ):
        device = torch.device(device)
        if device.type != "cuda":
            raise RuntimeError(
                "autoregressive draft CUDA graph requires a CUDA device"
            )
        device_index = (
            torch.cuda.current_device()
            if device.index is None
            else device.index
        )
        scratch_allocator = DirectProposalKVAllocator(
            physical_store
        )
        scratch_cache = ProposalKVCache(
            scratch_allocator
        )
        scratch_owner = Qwen3DraftGraphScratchOwner(
            live_cache=proposal_kv_cache,
            scratch_cache=scratch_cache,
        )
        block_table_width = int(
            physical_store.gpu_capacity
        )
        graph_backend = Qwen3DraftCudaGraphBackend(
            backend=backend,
            proposal_kv_cache=proposal_kv_cache,
            device=device,
            compute_dtype=dtype,
            block_table_width=block_table_width,
        )
        runner = AutoregressiveDraftExactGraphRunner(
            enabled=True,
            q_allowlist=(
                config
                .autoregressive_draft_cuda_graph_q_allowlist
            ),
            batch_allowlist=(
                config
                .autoregressive_draft_cuda_graph_batch_allowlist
            ),
            min_observations=(
                config
                .autoregressive_draft_cuda_graph_min_observations
            ),
            max_entries=(
                config
                .autoregressive_draft_cuda_graph_max_entries
            ),
            max_static_bytes=(
                config
                .autoregressive_draft_cuda_graph_max_static_bytes
            ),
            max_reserved_bytes=(
                config
                .autoregressive_draft_cuda_graph_max_reserved_bytes
            ),
            max_total_capture_ns=(
                config
                .autoregressive_draft_cuda_graph_max_total_capture_ns
            ),
            max_single_capture_ns=(
                config
                .autoregressive_draft_cuda_graph_max_single_capture_ns
            ),
            tensor_parallel_size=(
                backend.tensor_parallel_size
            ),
            tensor_parallel_rank=(
                backend.tensor_parallel_rank
            ),
            device_index=device_index,
            compute_dtype=str(dtype),
            backend_identity=backend.backend_identity,
            model_fingerprint=backend.model_fingerprint,
            tokenizer_fingerprint=(
                backend.tokenizer_fingerprint
            ),
            local_query_heads=backend.local_query_heads,
            local_kv_heads=physical_store.local_kv_heads,
            kv_block_table_width=block_table_width,
            proposal_kv_capacity=int(
                physical_store.gpu_capacity
            ),
            blockwise_offload=False,
            capture_backend=graph_backend,
            scratch_owner=scratch_owner,
        )
        return AutoregressiveDraftGraphComponents(
            scratch_cache=scratch_cache,
            scratch_owner=scratch_owner,
            backend=graph_backend,
            runner=runner,
        )

    return AutoregressiveDraftRegistrationDependencies(
        build_checkpoint_fingerprint=build_checkpoint_fingerprint,
        load_tokenizer=load_tokenizer,
        build_tokenizer_contract=tokenizer_contract,
        validate_tokenizer_compatibility=(
            validate_tokenizer_compatibility
        ),
        load_hf_config=AutoConfig.from_pretrained,
        build_model=build_model,
        load_weights=lambda model, path: load_model(model, path),
        move_model_to_target=move_model_to_target,
        build_proposal_kv_allocator=(
            build_qwen3_draft_proposal_kv_allocator
        ),
        build_proposal_kv_cache=ProposalKVCache,
        build_backend=build_backend,
        build_graph_components=build_graph_components,
        build_executor=(
            lambda **kwargs: AutoregressiveDraftProposalExecutor(
                **kwargs
            )
        ),
        build_descriptor=build_descriptor,
    )
