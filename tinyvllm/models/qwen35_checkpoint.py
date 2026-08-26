from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath


@dataclass(frozen=True)
class Qwen35CheckpointSource:
    name: str
    shard: str


@dataclass(frozen=True)
class Qwen35CheckpointLoadTarget:
    source: Qwen35CheckpointSource
    target: str
    packed_slot: str | int | None


@dataclass(frozen=True)
class Qwen35CheckpointSkip:
    source: Qwen35CheckpointSource
    scope: str


@dataclass(frozen=True)
class Qwen35CheckpointWeightPlan:
    loads: tuple[Qwen35CheckpointLoadTarget, ...]
    skips: tuple[Qwen35CheckpointSkip, ...]
    shards: tuple[str, ...]


@dataclass(frozen=True)
class Qwen35CheckpointTensorMetadata:
    dtype: str
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]


@dataclass(frozen=True)
class Qwen35CheckpointTensorLoad:
    weight: Qwen35CheckpointLoadTarget
    metadata: Qwen35CheckpointTensorMetadata
    transform: str


@dataclass(frozen=True)
class Qwen35CheckpointTensorPlan:
    loads: tuple[Qwen35CheckpointTensorLoad, ...]
    skips: tuple[Qwen35CheckpointSkip, ...]
    payload_bytes: int


def qwen35_mtp_skip_source_names(
    weight_plan: Qwen35CheckpointWeightPlan,
) -> tuple[str, ...]:
    if type(weight_plan) is not Qwen35CheckpointWeightPlan:
        raise ValueError(
            "weight_plan must be an exact "
            "Qwen35CheckpointWeightPlan"
        )
    names = []
    for skip in weight_plan.skips:
        if type(skip) is not Qwen35CheckpointSkip:
            raise ValueError(
                "weight plan skips must be "
                "Qwen35CheckpointSkip values"
            )
        if skip.scope == "mtp":
            names.append(skip.source.name)
    if len(set(names)) != len(names):
        raise ValueError(
            "MTP checkpoint skip sources must be unique"
        )
    return tuple(sorted(names))


_DTYPE_BYTES = {
    "BF16": 2,
    "F32": 4,
}
_CONFIG_DTYPES = {
    "bfloat16": "BF16",
    "float32": "F32",
}
_SHARED_LAYER_TARGETS = {
    "input_layernorm.weight": ("input_layernorm.weight", None),
    "post_attention_layernorm.weight": (
        "post_attention_layernorm.weight",
        None,
    ),
    "mlp.down_proj.weight": ("mlp.down_proj.weight", None),
    "mlp.gate_proj.weight": ("mlp.gate_up_proj.weight", 0),
    "mlp.up_proj.weight": ("mlp.gate_up_proj.weight", 1),
}
_LINEAR_LAYER_TARGETS = {
    "linear_attn.in_proj_qkv.weight": (
        "linear_attention.in_proj_qkv.weight",
        None,
    ),
    "linear_attn.in_proj_z.weight": (
        "linear_attention.in_proj_z.weight",
        None,
    ),
    "linear_attn.in_proj_b.weight": (
        "linear_attention.in_proj_b.weight",
        None,
    ),
    "linear_attn.in_proj_a.weight": (
        "linear_attention.in_proj_a.weight",
        None,
    ),
    "linear_attn.out_proj.weight": (
        "linear_attention.out_proj.weight",
        None,
    ),
    "linear_attn.conv1d.weight": (
        "linear_attention.conv_weight",
        None,
    ),
    "linear_attn.A_log": ("linear_attention.A_log", None),
    "linear_attn.dt_bias": ("linear_attention.dt_bias", None),
    "linear_attn.norm.weight": (
        "linear_attention.norm_weight",
        None,
    ),
}
_FULL_LAYER_TARGETS = {
    "self_attn.q_proj.weight": (
        "full_attention.q_projection.weight",
        None,
    ),
    "self_attn.k_proj.weight": (
        "full_attention.k_projection.weight",
        None,
    ),
    "self_attn.v_proj.weight": (
        "full_attention.v_projection.weight",
        None,
    ),
    "self_attn.o_proj.weight": (
        "full_attention.output_projection.weight",
        None,
    ),
    "self_attn.q_norm.weight": (
        "full_attention.q_norm.weight",
        None,
    ),
    "self_attn.k_norm.weight": (
        "full_attention.k_norm.weight",
        None,
    ),
}


def _positive_integer(config, field_name: str) -> int:
    if not hasattr(config, field_name):
        raise ValueError(f"missing Qwen3.5 config field: {field_name}")
    value = getattr(config, field_name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _config_dtype(config) -> str:
    if not hasattr(config, "dtype"):
        raise ValueError("missing Qwen3.5 config field: dtype")
    value = getattr(config, "dtype")
    if value not in _CONFIG_DTYPES:
        raise ValueError(f"unsupported Qwen3.5 config dtype: {value}")
    return _CONFIG_DTYPES[value]


def _layer_types(config, num_hidden_layers: int) -> tuple[str, ...]:
    if not hasattr(config, "layer_types"):
        raise ValueError("missing Qwen3.5 config field: layer_types")
    values = getattr(config, "layer_types")
    if not isinstance(values, (tuple, list)):
        raise ValueError("layer_types must be a tuple or list")
    if len(values) != num_hidden_layers:
        raise ValueError(
            "layer_types length must match num_hidden_layers"
        )
    normalized = tuple(values)
    for value in normalized:
        if value not in ("linear_attention", "full_attention"):
            raise ValueError(f"unsupported Qwen3.5 layer type: {value}")
    return normalized


def _tie_word_embeddings(config) -> bool:
    if not hasattr(config, "tie_word_embeddings"):
        raise ValueError(
            "missing Qwen3.5 config field: tie_word_embeddings"
        )
    value = getattr(config, "tie_word_embeddings")
    if type(value) is not bool:
        raise ValueError("tie_word_embeddings must be a bool")
    return value


def _validate_qwen38_text_profile(config, profile) -> None:
    if profile is None:
        return
    field_pairs = (
        ("text_model_type", "model_type"),
        ("num_hidden_layers", "num_hidden_layers"),
        ("hidden_size", "hidden_size"),
        ("intermediate_size", "intermediate_size"),
        ("vocab_size", "vocab_size"),
        ("dtype", "dtype"),
        ("tie_word_embeddings", "tie_word_embeddings"),
    )
    for profile_field, config_field in field_pairs:
        if not hasattr(profile, profile_field):
            raise ValueError(
                f"qwen38_text_profile missing {profile_field}"
            )
        if not hasattr(config, config_field):
            raise ValueError(
                f"missing Qwen3.5 config field: {config_field}"
            )
        if getattr(profile, profile_field) != getattr(
            config,
            config_field,
        ):
            raise ValueError(
                "qwen38_text_profile "
                f"{profile_field} does not match text_config"
            )
    if not hasattr(profile, "layer_types"):
        raise ValueError("qwen38_text_profile missing layer_types")
    if tuple(profile.layer_types) != tuple(
        getattr(config, "layer_types", ())
    ):
        raise ValueError(
            "qwen38_text_profile layer_types "
            "does not match text_config"
        )


def _expected_language_targets(
    layer_types: tuple[str, ...],
    tie_word_embeddings: bool,
) -> dict[str, tuple[str, str | int | None]]:
    expected = {
        "model.language_model.embed_tokens.weight": (
            "embed_tokens.weight",
            None,
        ),
        "model.language_model.norm.weight": (
            "final_norm.weight",
            None,
        ),
    }
    if not tie_word_embeddings:
        expected["lm_head.weight"] = ("lm_head.weight", None)
    for layer_index, layer_type in enumerate(layer_types):
        source_prefix = (
            f"model.language_model.layers.{layer_index}."
        )
        target_prefix = f"layers.{layer_index}."
        layer_targets = dict(_SHARED_LAYER_TARGETS)
        if layer_type == "linear_attention":
            layer_targets.update(_LINEAR_LAYER_TARGETS)
        else:
            layer_targets.update(_FULL_LAYER_TARGETS)
        for suffix, (target_suffix, packed_slot) in layer_targets.items():
            expected[source_prefix + suffix] = (
                target_prefix + target_suffix,
                packed_slot,
            )
    return expected


def _validate_shard_name(value) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("shard name must be a non-empty string")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or "\\" in value
        or value != str(path)
    ):
        raise ValueError("shard name must be a safe relative path")
    if not value.endswith(".safetensors"):
        raise ValueError("shard name must end in .safetensors")
    return value


def _language_tensor_contracts(
    config,
    layer_types: tuple[str, ...],
    tie_word_embeddings: bool,
) -> dict[str, tuple[str, tuple[int, ...], str]]:
    compute_dtype = _config_dtype(config)
    hidden_size = _positive_integer(config, "hidden_size")
    intermediate_size = _positive_integer(config, "intermediate_size")
    vocab_size = _positive_integer(config, "vocab_size")
    linear_num_key_heads = _positive_integer(
        config,
        "linear_num_key_heads",
    )
    linear_num_value_heads = _positive_integer(
        config,
        "linear_num_value_heads",
    )
    linear_key_head_dim = _positive_integer(
        config,
        "linear_key_head_dim",
    )
    linear_value_head_dim = _positive_integer(
        config,
        "linear_value_head_dim",
    )
    linear_conv_kernel_dim = _positive_integer(
        config,
        "linear_conv_kernel_dim",
    )
    num_attention_heads = _positive_integer(
        config,
        "num_attention_heads",
    )
    num_key_value_heads = _positive_integer(
        config,
        "num_key_value_heads",
    )
    head_dim = _positive_integer(config, "head_dim")
    key_width = linear_num_key_heads * linear_key_head_dim
    value_width = linear_num_value_heads * linear_value_head_dim
    conv_width = 2 * key_width + value_width
    query_gate_width = num_attention_heads * 2 * head_dim
    query_width = num_attention_heads * head_dim
    kv_width = num_key_value_heads * head_dim

    contracts = {
        "model.language_model.embed_tokens.weight": (
            compute_dtype,
            (vocab_size, hidden_size),
            "identity",
        ),
        "model.language_model.norm.weight": (
            compute_dtype,
            (hidden_size,),
            "identity",
        ),
    }
    if not tie_word_embeddings:
        contracts["lm_head.weight"] = (
            compute_dtype,
            (vocab_size, hidden_size),
            "identity",
        )
    shared = {
        "input_layernorm.weight": (
            compute_dtype,
            (hidden_size,),
            "identity",
        ),
        "post_attention_layernorm.weight": (
            compute_dtype,
            (hidden_size,),
            "identity",
        ),
        "mlp.gate_proj.weight": (
            compute_dtype,
            (intermediate_size, hidden_size),
            "identity",
        ),
        "mlp.up_proj.weight": (
            compute_dtype,
            (intermediate_size, hidden_size),
            "identity",
        ),
        "mlp.down_proj.weight": (
            compute_dtype,
            (hidden_size, intermediate_size),
            "identity",
        ),
    }
    linear = {
        "linear_attn.in_proj_qkv.weight": (
            compute_dtype,
            (conv_width, hidden_size),
            "identity",
        ),
        "linear_attn.in_proj_z.weight": (
            compute_dtype,
            (value_width, hidden_size),
            "identity",
        ),
        "linear_attn.in_proj_b.weight": (
            compute_dtype,
            (linear_num_value_heads, hidden_size),
            "identity",
        ),
        "linear_attn.in_proj_a.weight": (
            compute_dtype,
            (linear_num_value_heads, hidden_size),
            "identity",
        ),
        "linear_attn.out_proj.weight": (
            compute_dtype,
            (hidden_size, value_width),
            "identity",
        ),
        "linear_attn.conv1d.weight": (
            compute_dtype,
            (conv_width, 1, linear_conv_kernel_dim),
            "squeeze_conv_channel",
        ),
        "linear_attn.A_log": (
            "F32",
            (linear_num_value_heads,),
            "identity",
        ),
        "linear_attn.dt_bias": (
            compute_dtype,
            (linear_num_value_heads,),
            "identity",
        ),
        "linear_attn.norm.weight": (
            "F32",
            (linear_value_head_dim,),
            "identity",
        ),
    }
    full = {
        "self_attn.q_proj.weight": (
            compute_dtype,
            (query_gate_width, hidden_size),
            "identity",
        ),
        "self_attn.k_proj.weight": (
            compute_dtype,
            (kv_width, hidden_size),
            "identity",
        ),
        "self_attn.v_proj.weight": (
            compute_dtype,
            (kv_width, hidden_size),
            "identity",
        ),
        "self_attn.o_proj.weight": (
            compute_dtype,
            (hidden_size, query_width),
            "identity",
        ),
        "self_attn.q_norm.weight": (
            compute_dtype,
            (head_dim,),
            "identity",
        ),
        "self_attn.k_norm.weight": (
            compute_dtype,
            (head_dim,),
            "identity",
        ),
    }
    for layer_index, layer_type in enumerate(layer_types):
        prefix = f"model.language_model.layers.{layer_index}."
        layer_contracts = dict(shared)
        layer_contracts.update(
            linear if layer_type == "linear_attention" else full
        )
        for suffix, contract in layer_contracts.items():
            contracts[prefix + suffix] = contract
    return contracts


def _parse_tensor_metadata(
    source_name: str,
    value,
) -> Qwen35CheckpointTensorMetadata:
    if not isinstance(value, Mapping):
        raise ValueError(
            f"tensor metadata must be a mapping: {source_name}"
        )
    dtype = value.get("dtype")
    if dtype not in _DTYPE_BYTES:
        raise ValueError(
            f"unsupported tensor dtype for {source_name}: {dtype}"
        )
    shape_value = value.get("shape")
    if not isinstance(shape_value, (tuple, list)) or not shape_value:
        raise ValueError(
            f"tensor shape must contain positive integers: {source_name}"
        )
    shape = tuple(shape_value)
    if any(
        isinstance(dimension, bool)
        or not isinstance(dimension, int)
        or dimension <= 0
        for dimension in shape
    ):
        raise ValueError(
            f"tensor shape must contain positive integers: {source_name}"
        )
    offsets_value = value.get("data_offsets")
    if (
        not isinstance(offsets_value, (tuple, list))
        or len(offsets_value) != 2
        or any(
            isinstance(offset, bool) or not isinstance(offset, int)
            for offset in offsets_value
        )
    ):
        raise ValueError(
            f"data_offsets must contain two integers: {source_name}"
        )
    start, end = offsets_value
    if start < 0 or end <= start:
        raise ValueError(
            f"data_offsets must define a positive range: {source_name}"
        )
    element_count = 1
    for dimension in shape:
        element_count *= dimension
    if end - start != element_count * _DTYPE_BYTES[dtype]:
        raise ValueError(
            f"tensor byte count does not match metadata: {source_name}"
        )
    return Qwen35CheckpointTensorMetadata(
        dtype=dtype,
        shape=shape,
        data_offsets=(start, end),
    )


def build_qwen35_checkpoint_weight_plan(
    hf_config,
    index_payload: Mapping[str, object],
    *,
    qwen38_text_profile=None,
) -> Qwen35CheckpointWeightPlan:
    config = getattr(hf_config, "text_config", hf_config)
    _validate_qwen38_text_profile(config, qwen38_text_profile)
    num_hidden_layers = _positive_integer(config, "num_hidden_layers")
    layer_types = _layer_types(config, num_hidden_layers)
    tie_word_embeddings = _tie_word_embeddings(config)

    if not isinstance(index_payload, Mapping):
        raise ValueError("index payload must be a mapping")
    weight_map = index_payload.get("weight_map")
    if not isinstance(weight_map, Mapping) or not weight_map:
        raise ValueError("weight_map must be a non-empty mapping")

    language_sources = {}
    skip_sources = []
    shards = set()
    for source_name, shard_value in weight_map.items():
        if not isinstance(source_name, str) or not source_name:
            raise ValueError("source name must be a non-empty string")
        shard_name = _validate_shard_name(shard_value)
        source = Qwen35CheckpointSource(source_name, shard_name)
        shards.add(shard_name)
        if (
            source_name.startswith("model.language_model.")
            or source_name == "lm_head.weight"
        ):
            language_sources[source_name] = source
        elif source_name.startswith("model.visual."):
            skip_sources.append(Qwen35CheckpointSkip(source, "visual"))
        elif source_name.startswith("mtp."):
            skip_sources.append(Qwen35CheckpointSkip(source, "mtp"))
        else:
            raise ValueError(
                f"unsupported checkpoint scope: {source_name}"
            )

    expected = _expected_language_targets(
        layer_types,
        tie_word_embeddings,
    )
    observed_names = set(language_sources)
    expected_names = set(expected)
    missing = expected_names - observed_names
    unexpected = observed_names - expected_names
    if missing and unexpected:
        raise ValueError(
            "language-model weight set does not match config topology"
        )
    if missing:
        raise ValueError(
            "missing language-model weights: "
            + ", ".join(sorted(missing))
        )
    if unexpected:
        raise ValueError(
            "unexpected language-model weights: "
            + ", ".join(sorted(unexpected))
        )

    loads = []
    target_keys = set()
    for source_name in sorted(language_sources):
        target, packed_slot = expected[source_name]
        target_key = (target, packed_slot)
        if target_key in target_keys:
            raise ValueError(
                "duplicate logical target and packed slot: "
                f"{target} {packed_slot}"
            )
        target_keys.add(target_key)
        loads.append(Qwen35CheckpointLoadTarget(
            language_sources[source_name],
            target,
            packed_slot,
        ))

    skips = tuple(sorted(
        skip_sources,
        key=lambda entry: entry.source.name,
    ))
    if (
        qwen38_text_profile is not None
        and any(entry.scope != "visual" for entry in skips)
    ):
        raise ValueError(
            "Qwen3.8 checkpoint skip scope must be visual"
        )
    covered_names = {
        entry.source.name for entry in loads
    } | {
        entry.source.name for entry in skips
    }
    if covered_names != set(weight_map):
        raise ValueError("checkpoint source coverage is incomplete")
    return Qwen35CheckpointWeightPlan(
        loads=tuple(loads),
        skips=skips,
        shards=tuple(sorted(shards)),
    )


def build_qwen35_checkpoint_tensor_plan(
    hf_config,
    index_payload: Mapping[str, object],
    shard_headers: Mapping[str, Mapping[str, object]],
    *,
    qwen38_text_profile=None,
) -> Qwen35CheckpointTensorPlan:
    weight_plan = build_qwen35_checkpoint_weight_plan(
        hf_config,
        index_payload,
        qwen38_text_profile=qwen38_text_profile,
    )
    if not isinstance(shard_headers, Mapping):
        raise ValueError("shard_headers must be a mapping")
    if set(shard_headers) != set(weight_plan.shards):
        raise ValueError("shard header set must match weight plan")
    weight_map = index_payload["weight_map"]
    parsed_by_source = {}
    payload_bytes = 0
    for shard_name in weight_plan.shards:
        header = shard_headers[shard_name]
        if not isinstance(header, Mapping):
            raise ValueError(
                f"shard header must be a mapping: {shard_name}"
            )
        expected_sources = {
            source_name
            for source_name, source_shard in weight_map.items()
            if source_shard == shard_name
        }
        observed_sources = {
            source_name
            for source_name in header
            if source_name != "__metadata__"
        }
        if observed_sources != expected_sources:
            raise ValueError("header source set must match index")
        intervals = []
        for source_name in sorted(observed_sources):
            metadata = _parse_tensor_metadata(
                source_name,
                header[source_name],
            )
            parsed_by_source[source_name] = metadata
            intervals.append(metadata.data_offsets)
        intervals.sort()
        expected_start = 0
        for start, end in intervals:
            if start != expected_start:
                raise ValueError(
                    "tensor payload intervals must be contiguous"
                )
            expected_start = end
        payload_bytes += expected_start

    metadata_value = index_payload.get("metadata")
    if not isinstance(metadata_value, Mapping):
        raise ValueError("index metadata must be a mapping")
    total_size = metadata_value.get("total_size")
    if (
        isinstance(total_size, bool)
        or not isinstance(total_size, int)
        or total_size < 0
    ):
        raise ValueError(
            "index metadata total_size must be a non-negative integer"
        )
    if payload_bytes != total_size:
        raise ValueError(
            "payload byte total must match index metadata"
        )

    config = getattr(hf_config, "text_config", hf_config)
    num_hidden_layers = _positive_integer(config, "num_hidden_layers")
    layer_types = _layer_types(config, num_hidden_layers)
    tie_word_embeddings = _tie_word_embeddings(config)
    contracts = _language_tensor_contracts(
        config,
        layer_types,
        tie_word_embeddings,
    )
    loads = []
    for weight in weight_plan.loads:
        metadata = parsed_by_source[weight.source.name]
        expected_dtype, expected_shape, transform = contracts[
            weight.source.name
        ]
        if metadata.dtype != expected_dtype:
            raise ValueError(
                "tensor dtype does not match config: "
                f"{weight.source.name}"
            )
        if metadata.shape != expected_shape:
            raise ValueError(
                "tensor shape does not match config: "
                f"{weight.source.name}"
            )
        loads.append(Qwen35CheckpointTensorLoad(
            weight=weight,
            metadata=metadata,
            transform=transform,
        ))
    return Qwen35CheckpointTensorPlan(
        loads=tuple(loads),
        skips=weight_plan.skips,
        payload_bytes=payload_bytes,
    )
