from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from tinyvllm.models.qwen35_checkpoint_candidate_factory import (
    Qwen35PreparedCheckpointCandidateTarget,
)
from tinyvllm.models.qwen35_checkpoint_streaming import (
    Qwen35LoadedCheckpointCandidate,
    load_qwen35_fresh_checkpoint_candidate,
)
from tinyvllm.models.qwen35_checkpoint_worker import (
    Qwen35CheckpointCandidateLoadRequest,
    validate_qwen35_checkpoint_candidate_load_request,
)


def _authorization_sha256(value) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(
            character not in "0123456789abcdef"
            for character in value
        )
    ):
        raise ValueError(
            "authorization_sha256 must be a lowercase SHA256"
        )
    return value


@dataclass(frozen=True)
class Qwen35AuthorizedCheckpointCandidateLoader:
    prepare_target: Callable[
        [],
        Qwen35PreparedCheckpointCandidateTarget,
    ]
    authorization_sha256: str

    def __post_init__(self):
        if not callable(self.prepare_target):
            raise ValueError("prepare_target must be callable")
        _authorization_sha256(self.authorization_sha256)

    def __call__(
        self,
        request: Qwen35CheckpointCandidateLoadRequest,
    ) -> Qwen35LoadedCheckpointCandidate:
        request = validate_qwen35_checkpoint_candidate_load_request(
            request
        )
        if request.authorization_sha256 != self.authorization_sha256:
            raise RuntimeError(
                "checkpoint candidate loader authorization "
                "does not match request"
            )
        target = self.prepare_target()
        if type(target) is not Qwen35PreparedCheckpointCandidateTarget:
            raise ValueError(
                "prepare_target must return an exact "
                "Qwen35PreparedCheckpointCandidateTarget"
            )
        if target.assembly.parameter_device.type != "cpu":
            raise ValueError(
                "prepared checkpoint candidate target must be CPU"
            )
        return load_qwen35_fresh_checkpoint_candidate(
            target.take,
            request.checkpoint_dir,
            max_tensor_bytes=request.max_tensor_bytes,
            model_fingerprint=request.model_fingerprint,
        )


def build_qwen35_authorized_checkpoint_candidate_loader(
    prepare_target,
    *,
    authorization_sha256,
) -> Qwen35AuthorizedCheckpointCandidateLoader:
    return Qwen35AuthorizedCheckpointCandidateLoader(
        prepare_target=prepare_target,
        authorization_sha256=_authorization_sha256(
            authorization_sha256
        ),
    )
