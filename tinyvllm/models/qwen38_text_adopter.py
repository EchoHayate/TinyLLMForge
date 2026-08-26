from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import re


QWEN38_REPOSITORY = "Qwen/Qwen3.8-27B"
QWEN38_ARCHITECTURE = "Qwen3_5ForConditionalGeneration"
QWEN38_TEXT_MODEL_TYPE = "qwen3_5_text"
QWEN38_NUM_HIDDEN_LAYERS = 64
QWEN38_HIDDEN_SIZE = 5120
QWEN38_INTERMEDIATE_SIZE = 17408
QWEN38_DTYPE = "bfloat16"
QWEN38_VOCAB_SIZE = 248320
QWEN38_MODEL_MANIFEST_SCHEMA = (
    "tinyllmforge.qwen38-model-manifest.v1"
)

_IMMUTABLE_REVISION = re.compile(r"^[0-9a-f]{40}$")
_MULTIMODAL_TOKEN_FIELDS = (
    "image_token_id",
    "video_token_id",
    "vision_start_token_id",
    "vision_end_token_id",
)


@dataclass(frozen=True)
class Qwen38TextRuntimeProfile:
    repository: str
    revision: str
    architecture: str
    text_model_type: str
    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int
    layer_types: tuple[str, ...]
    dtype: str
    vocab_size: int
    tie_word_embeddings: bool
    language_model_only: bool
    multimodal_token_ids: tuple[int, ...]


def _field(container, name):
    if isinstance(container, Mapping):
        if name not in container:
            raise ValueError(f"missing Qwen3.8 config field: {name}")
        return container[name]
    if not hasattr(container, name):
        raise ValueError(f"missing Qwen3.8 config field: {name}")
    return getattr(container, name)


def _positive_integer(container, name):
    value = _field(container, name)
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")
    return value


def _architecture(hf_config):
    architectures = _field(hf_config, "architectures")
    if (
        not isinstance(architectures, Sequence)
        or isinstance(architectures, (str, bytes))
        or tuple(architectures) != (QWEN38_ARCHITECTURE,)
    ):
        raise ValueError(
            "Qwen3.8 architecture must be "
            f"{QWEN38_ARCHITECTURE}"
        )
    return QWEN38_ARCHITECTURE


def _revision(hf_config):
    value = _field(hf_config, "_commit_hash")
    if (
        not isinstance(value, str)
        or _IMMUTABLE_REVISION.fullmatch(value) is None
    ):
        raise ValueError(
            "Qwen3.8 requires an immutable revision SHA"
        )
    return value


def _dtype(text_config):
    value = getattr(
        text_config,
        "dtype",
        getattr(text_config, "torch_dtype", None),
    )
    if not isinstance(value, str):
        value = str(value).removeprefix("torch.")
    if value != QWEN38_DTYPE:
        raise ValueError("Qwen3.8 text dtype must be bfloat16")
    return value


def _layer_types(text_config):
    value = _field(text_config, "layer_types")
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
    ):
        raise ValueError("Qwen3.8 layer_types must be a sequence")
    layer_types = tuple(value)
    expected = tuple(
        "full_attention" if (index + 1) % 4 == 0
        else "linear_attention"
        for index in range(QWEN38_NUM_HIDDEN_LAYERS)
    )
    if layer_types != expected:
        raise ValueError("Qwen3.8 layer cadence mismatch")
    return layer_types


def _multimodal_token_ids(hf_config, text_config):
    values = []
    for container in (hf_config, text_config):
        for name in _MULTIMODAL_TOKEN_FIELDS:
            value = getattr(container, name, None)
            if value is None:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(
                    f"Qwen3.8 {name} must be a non-negative integer"
                )
            values.append(value)
    return tuple(sorted(set(values)))


def adopt_qwen38_text_config(
    hf_config,
    *,
    repository=None,
    revision=None,
):
    if repository is None:
        repository = _field(hf_config, "_name_or_path")
    if repository != QWEN38_REPOSITORY:
        raise ValueError(
            "Qwen3.8 repository must be "
            f"{QWEN38_REPOSITORY}"
        )
    if revision is None:
        revision = _revision(hf_config)
    elif (
        not isinstance(revision, str)
        or _IMMUTABLE_REVISION.fullmatch(revision) is None
    ):
        raise ValueError(
            "Qwen3.8 requires an immutable revision SHA"
        )
    architecture = _architecture(hf_config)
    if _field(hf_config, "model_type") != "qwen3_5":
        raise ValueError("Qwen3.8 top-level model type must be qwen3_5")
    text_config = _field(hf_config, "text_config")
    text_model_type = _field(text_config, "model_type")
    if text_model_type != QWEN38_TEXT_MODEL_TYPE:
        raise ValueError(
            "Qwen3.8 text model type must be qwen3_5_text"
        )
    num_hidden_layers = _positive_integer(
        text_config,
        "num_hidden_layers",
    )
    if num_hidden_layers != QWEN38_NUM_HIDDEN_LAYERS:
        raise ValueError("Qwen3.8 num_hidden_layers must be 64")
    hidden_size = _positive_integer(text_config, "hidden_size")
    if hidden_size != QWEN38_HIDDEN_SIZE:
        raise ValueError("Qwen3.8 hidden_size must be 5120")
    intermediate_size = _positive_integer(
        text_config,
        "intermediate_size",
    )
    if intermediate_size != QWEN38_INTERMEDIATE_SIZE:
        raise ValueError(
            "Qwen3.8 intermediate_size must be 17408"
        )
    vocab_size = _positive_integer(text_config, "vocab_size")
    if vocab_size != QWEN38_VOCAB_SIZE:
        raise ValueError("Qwen3.8 vocab_size must be 248320")
    tie_word_embeddings = _field(
        text_config,
        "tie_word_embeddings",
    )
    if tie_word_embeddings is not False:
        raise ValueError(
            "Qwen3.8 tie_word_embeddings must be false"
        )
    language_model_only = _field(
        hf_config,
        "language_model_only",
    )
    if language_model_only is not False:
        raise ValueError(
            "Qwen3.8 language_model_only must be false"
        )
    return Qwen38TextRuntimeProfile(
        repository=repository,
        revision=revision,
        architecture=architecture,
        text_model_type=text_model_type,
        num_hidden_layers=num_hidden_layers,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        layer_types=_layer_types(text_config),
        dtype=_dtype(text_config),
        vocab_size=vocab_size,
        tie_word_embeddings=tie_word_embeddings,
        language_model_only=language_model_only,
        multimodal_token_ids=_multimodal_token_ids(
            hf_config,
            text_config,
        ),
    )


def read_qwen38_source_identity(model_dir):
    model_root = Path(model_dir).resolve()
    manifest_path = model_root.parent / "model_manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError(
            "Qwen3.8 model manifest must be an adjacent regular file"
        )
    try:
        def reject_duplicate_keys(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate JSON key: {key}")
                result[key] = value
            return result

        payload = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            "Qwen3.8 model manifest is not valid JSON"
        ) from error
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version")
        != QWEN38_MODEL_MANIFEST_SCHEMA
    ):
        raise ValueError("Qwen3.8 model manifest schema mismatch")
    manifest_root = payload.get("model_root")
    if (
        not isinstance(manifest_root, str)
        or manifest_root != str(model_root)
    ):
        raise ValueError("Qwen3.8 model root identity mismatch")
    repository = payload.get("repository")
    revision = payload.get("resolved_revision")
    if repository != QWEN38_REPOSITORY:
        raise ValueError("Qwen3.8 manifest repository mismatch")
    if (
        not isinstance(revision, str)
        or _IMMUTABLE_REVISION.fullmatch(revision) is None
    ):
        raise ValueError("Qwen3.8 manifest revision is not immutable")
    return {
        "repository": repository,
        "revision": revision,
    }


def is_qwen38_text_checkpoint(hf_config):
    try:
        adopt_qwen38_text_config(hf_config)
    except (TypeError, ValueError):
        return False
    return True


def reject_qwen38_multimodal_inputs(
    profile,
    *,
    input_ids,
    multimodal_inputs=None,
):
    if not isinstance(profile, Qwen38TextRuntimeProfile):
        raise TypeError("profile must be a Qwen38TextRuntimeProfile")
    if multimodal_inputs:
        raise ValueError(
            "Qwen3.8 first adopter is text-only; "
            "image/video inputs are unsupported"
        )
    forbidden = set(profile.multimodal_token_ids)
    if forbidden and any(token_id in forbidden for token_id in input_ids):
        raise ValueError(
            "Qwen3.8 first adopter is text-only; "
            "image/video tokens are unsupported"
        )


def validate_qwen38_sequence_batch(
    profile,
    seqs,
    *,
    is_prefill,
):
    if not isinstance(is_prefill, bool):
        raise TypeError("is_prefill must be a bool")
    for seq in seqs:
        multimodal_inputs = getattr(seq, "multimodal_inputs", None)
        input_ids = (
            getattr(seq, "prompt_token_ids")
            if is_prefill
            else (getattr(seq, "last_token"),)
        )
        reject_qwen38_multimodal_inputs(
            profile,
            input_ids=input_ids,
            multimodal_inputs=multimodal_inputs,
        )
