from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import os

from tinyvllm.engine.hybrid_state import HybridStateTensorPool
from tinyvllm.models.qwen35_checkpoint import (
    Qwen35CheckpointTensorPlan,
)
from tinyvllm.models.qwen35_checkpoint_candidate_factory import (
    prepare_qwen35_checkpoint_candidate_target,
)
from tinyvllm.models.qwen35_checkpoint_candidate_loader import (
    Qwen35AuthorizedCheckpointCandidateLoader,
    build_qwen35_authorized_checkpoint_candidate_loader,
)
from tinyvllm.models.qwen35_checkpoint_streaming import (
    Qwen35LoadedCheckpointCandidate,
)
from tinyvllm.models.qwen35_checkpoint_worker import (
    Qwen35CheckpointCandidateLoadRequest,
    validate_qwen35_checkpoint_candidate_load_request,
)


def _sha256(value, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(
            character not in "0123456789abcdef"
            for character in value
        )
    ):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _checkpoint_dir(value) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("checkpoint_dir must be a non-empty string")
    if "\x00" in value:
        raise ValueError("checkpoint_dir must not contain NUL")
    if len(value.encode("utf-8")) > 4096:
        raise ValueError(
            "checkpoint_dir must be at most 4096 UTF-8 bytes"
        )
    if not os.path.isabs(value):
        raise ValueError("checkpoint_dir must be absolute")
    if os.path.normpath(value) != value:
        raise ValueError("checkpoint_dir must be normalized")
    return value


def _tp_context(size, rank) -> tuple[int, int]:
    if (
        isinstance(size, bool)
        or not isinstance(size, int)
        or size <= 0
    ):
        raise ValueError(
            "tensor_parallel_size must be a positive integer"
        )
    if (
        isinstance(rank, bool)
        or not isinstance(rank, int)
        or rank < 0
        or rank >= size
    ):
        raise ValueError(
            "tensor_parallel_rank must be in "
            "[0, tensor_parallel_size)"
        )
    return size, rank


@dataclass(frozen=True)
class Qwen35CheckpointManifestIdentity:
    checkpoint_dir: str
    model_manifest_sha256: str
    config_sha256: str
    index_sha256: str
    config_index_header_sha256: str

    def __post_init__(self):
        _checkpoint_dir(self.checkpoint_dir)
        _sha256(
            self.model_manifest_sha256,
            "model_manifest_sha256",
        )
        _sha256(self.config_sha256, "config_sha256")
        _sha256(self.index_sha256, "index_sha256")
        _sha256(
            self.config_index_header_sha256,
            "config_index_header_sha256",
        )


@dataclass(frozen=True)
class Qwen35ManifestBoundCheckpointCandidateLoader:
    configuration: Qwen35RankCheckpointLoaderConfiguration
    authorized_loader: Qwen35AuthorizedCheckpointCandidateLoader

    def __call__(
        self,
        request: Qwen35CheckpointCandidateLoadRequest,
    ) -> Qwen35LoadedCheckpointCandidate:
        request = validate_qwen35_checkpoint_candidate_load_request(
            request
        )
        manifest = self.configuration.manifest
        if request.checkpoint_dir != manifest.checkpoint_dir:
            raise RuntimeError(
                "request checkpoint_dir does not match manifest"
            )
        if (
            request.model_fingerprint
            != manifest.model_manifest_sha256
        ):
            raise RuntimeError(
                "request model_fingerprint does not match manifest"
            )
        if (
            request.authorization_sha256
            != self.configuration.authorization_sha256
        ):
            raise RuntimeError(
                "request authorization does not match configuration"
            )
        return self.authorized_loader(request)


@dataclass(frozen=True)
class Qwen35RankCheckpointLoaderConfiguration:
    manifest: Qwen35CheckpointManifestIdentity
    hf_config: object
    tensor_plan: Qwen35CheckpointTensorPlan
    tensor_parallel_size: int
    tensor_parallel_rank: int
    create_pool: Callable[[], HybridStateTensorPool]
    build_attention_backend: Callable
    authorization_sha256: str

    def __post_init__(self):
        if type(self.manifest) is not Qwen35CheckpointManifestIdentity:
            raise ValueError(
                "manifest must be an exact "
                "Qwen35CheckpointManifestIdentity"
            )
        if type(self.tensor_plan) is not Qwen35CheckpointTensorPlan:
            raise ValueError(
                "tensor_plan must be an exact "
                "Qwen35CheckpointTensorPlan"
            )
        _tp_context(
            self.tensor_parallel_size,
            self.tensor_parallel_rank,
        )
        if not callable(self.create_pool):
            raise ValueError("create_pool must be callable")
        if not callable(self.build_attention_backend):
            raise ValueError(
                "build_attention_backend must be callable"
            )
        _sha256(
            self.authorization_sha256,
            "authorization_sha256",
        )

    def build_loader(
        self,
    ) -> Qwen35ManifestBoundCheckpointCandidateLoader:
        def prepare_target():
            pool = self.create_pool()
            if type(pool) is not HybridStateTensorPool:
                raise ValueError(
                    "create_pool must return an exact "
                    "HybridStateTensorPool"
                )
            return prepare_qwen35_checkpoint_candidate_target(
                self.hf_config,
                self.tensor_plan,
                pool=pool,
                tensor_parallel_size=self.tensor_parallel_size,
                tensor_parallel_rank=self.tensor_parallel_rank,
                build_attention_backend=self.build_attention_backend,
                parameter_device="cpu",
            )

        authorized_loader = (
            build_qwen35_authorized_checkpoint_candidate_loader(
                prepare_target,
                authorization_sha256=self.authorization_sha256,
            )
        )
        return Qwen35ManifestBoundCheckpointCandidateLoader(
            configuration=self,
            authorized_loader=authorized_loader,
        )
