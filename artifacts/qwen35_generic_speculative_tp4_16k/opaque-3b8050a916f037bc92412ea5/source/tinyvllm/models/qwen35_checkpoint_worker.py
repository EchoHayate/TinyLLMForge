from __future__ import annotations

from dataclasses import dataclass
import os


def _sha256(value, name):
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


def _checkpoint_dir(value):
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
    normalized = os.path.normpath(value)
    if normalized != value:
        raise ValueError("checkpoint_dir must be normalized")
    return value


def _max_tensor_bytes(value):
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError(
            "max_tensor_bytes must be a positive integer"
        )
    return value


@dataclass(frozen=True)
class Qwen35CheckpointCandidateLoadRequest:
    checkpoint_dir: str
    model_fingerprint: str
    max_tensor_bytes: int
    authorization_sha256: str

    def __post_init__(self):
        _checkpoint_dir(self.checkpoint_dir)
        _sha256(
            self.model_fingerprint,
            "model_fingerprint",
        )
        _max_tensor_bytes(self.max_tensor_bytes)
        _sha256(
            self.authorization_sha256,
            "authorization_sha256",
        )


def validate_qwen35_checkpoint_candidate_load_request(value):
    if type(value) is not Qwen35CheckpointCandidateLoadRequest:
        raise ValueError(
            "request must be an exact "
            "Qwen35CheckpointCandidateLoadRequest"
        )
    return value
