from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from tinyvllm.models.qwen35_checkpoint import (
    _parse_tensor_metadata,
    _validate_shard_name,
)


@dataclass(frozen=True)
class Qwen35MTPCheckpointTensor:
    source_name: str
    shard: str
    dtype: str
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]
    destination_path: str
    packed_slot: str | int | None


@dataclass(frozen=True)
class Qwen35MTPCheckpointPlan:
    tensors: tuple[Qwen35MTPCheckpointTensor, ...]
    shards: tuple[str, ...]
    payload_bytes: int


_DESTINATIONS = {
    "mtp.fc.weight": ("fc.weight", None),
    "mtp.layers.0.input_layernorm.weight": (
        "layer.decoder_layer.input_layernorm.weight",
        None,
    ),
    "mtp.layers.0.self_attn.q_proj.weight": (
        "layer.decoder_layer.full_attention.q_projection.weight",
        None,
    ),
    "mtp.layers.0.self_attn.k_proj.weight": (
        "layer.decoder_layer.full_attention.k_projection.weight",
        None,
    ),
    "mtp.layers.0.self_attn.v_proj.weight": (
        "layer.decoder_layer.full_attention.v_projection.weight",
        None,
    ),
    "mtp.layers.0.self_attn.o_proj.weight": (
        "layer.decoder_layer.full_attention.output_projection.weight",
        None,
    ),
    "mtp.layers.0.self_attn.q_norm.weight": (
        "layer.decoder_layer.full_attention.q_norm.weight",
        None,
    ),
    "mtp.layers.0.self_attn.k_norm.weight": (
        "layer.decoder_layer.full_attention.k_norm.weight",
        None,
    ),
    "mtp.layers.0.post_attention_layernorm.weight": (
        "layer.decoder_layer.post_attention_layernorm.weight",
        None,
    ),
    "mtp.layers.0.mlp.gate_proj.weight": (
        "layer.decoder_layer.mlp.gate_up_proj.weight",
        0,
    ),
    "mtp.layers.0.mlp.up_proj.weight": (
        "layer.decoder_layer.mlp.gate_up_proj.weight",
        1,
    ),
    "mtp.layers.0.mlp.down_proj.weight": (
        "layer.decoder_layer.mlp.down_proj.weight",
        None,
    ),
    "mtp.norm.weight": ("norm.weight", None),
    "mtp.pre_fc_norm_embedding.weight": (
        "pre_fc_norm_embedding.weight",
        None,
    ),
    "mtp.pre_fc_norm_hidden.weight": (
        "pre_fc_norm_hidden.weight",
        None,
    ),
}


def _positive_integer(config, field_name: str) -> int:
    value = getattr(config, field_name, None)
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError(
            f"{field_name} must be a positive integer"
        )
    return value


def _mtp_contracts(config):
    hidden_size = _positive_integer(config, "hidden_size")
    intermediate_size = _positive_integer(
        config,
        "intermediate_size",
    )
    query_heads = _positive_integer(
        config,
        "num_attention_heads",
    )
    kv_heads = _positive_integer(
        config,
        "num_key_value_heads",
    )
    head_dim = _positive_integer(config, "head_dim")
    if getattr(config, "mtp_num_hidden_layers", None) != 1:
        raise ValueError(
            "mtp_num_hidden_layers must equal 1"
        )
    if (
        getattr(
            config,
            "mtp_use_dedicated_embeddings",
            None,
        )
        is not False
    ):
        raise ValueError(
            "MTP dedicated embeddings must be disabled"
        )
    if getattr(config, "tie_word_embeddings", None) is not True:
        raise ValueError(
            "tie_word_embeddings must be true"
        )
    if getattr(config, "dtype", None) != "bfloat16":
        raise ValueError(
            "Qwen3.5 MTP checkpoint must use bfloat16"
        )
    query_rows = query_heads * 2 * head_dim
    kv_rows = kv_heads * head_dim
    return {
        "mtp.fc.weight": ("BF16", (hidden_size, 2 * hidden_size)),
        "mtp.layers.0.input_layernorm.weight": (
            "BF16",
            (hidden_size,),
        ),
        "mtp.layers.0.self_attn.q_proj.weight": (
            "BF16",
            (query_rows, hidden_size),
        ),
        "mtp.layers.0.self_attn.k_proj.weight": (
            "BF16",
            (kv_rows, hidden_size),
        ),
        "mtp.layers.0.self_attn.v_proj.weight": (
            "BF16",
            (kv_rows, hidden_size),
        ),
        "mtp.layers.0.self_attn.o_proj.weight": (
            "BF16",
            (hidden_size, query_heads * head_dim),
        ),
        "mtp.layers.0.self_attn.q_norm.weight": (
            "BF16",
            (head_dim,),
        ),
        "mtp.layers.0.self_attn.k_norm.weight": (
            "BF16",
            (head_dim,),
        ),
        "mtp.layers.0.post_attention_layernorm.weight": (
            "BF16",
            (hidden_size,),
        ),
        "mtp.layers.0.mlp.gate_proj.weight": (
            "BF16",
            (intermediate_size, hidden_size),
        ),
        "mtp.layers.0.mlp.up_proj.weight": (
            "BF16",
            (intermediate_size, hidden_size),
        ),
        "mtp.layers.0.mlp.down_proj.weight": (
            "BF16",
            (hidden_size, intermediate_size),
        ),
        "mtp.norm.weight": ("BF16", (hidden_size,)),
        "mtp.pre_fc_norm_embedding.weight": (
            "BF16",
            (hidden_size,),
        ),
        "mtp.pre_fc_norm_hidden.weight": (
            "BF16",
            (hidden_size,),
        ),
    }


def build_qwen35_mtp_checkpoint_plan(
    hf_config,
    index_payload: Mapping[str, object],
    shard_headers: Mapping[str, Mapping[str, object]],
) -> Qwen35MTPCheckpointPlan:
    config = getattr(hf_config, "text_config", hf_config)
    contracts = _mtp_contracts(config)
    expected_sources = set(contracts)
    if not isinstance(index_payload, Mapping):
        raise ValueError(
            "index payload must be a mapping"
        )
    weight_map = index_payload.get("weight_map")
    if not isinstance(weight_map, Mapping):
        raise ValueError(
            "weight_map must be a mapping"
        )
    indexed_mtp_sources = {
        source_name
        for source_name in weight_map
        if (
            isinstance(source_name, str)
            and source_name.startswith("mtp.")
        )
    }
    missing = expected_sources - indexed_mtp_sources
    unexpected = indexed_mtp_sources - expected_sources
    if missing:
        raise ValueError(
            "missing MTP checkpoint sources: "
            + ", ".join(sorted(missing))
        )
    if unexpected:
        raise ValueError(
            "unexpected MTP checkpoint sources: "
            + ", ".join(sorted(unexpected))
        )
    if not isinstance(shard_headers, Mapping):
        raise ValueError(
            "shard_headers must be a mapping"
        )
    header_mtp_sources = {
        source_name
        for header in shard_headers.values()
        if isinstance(header, Mapping)
        for source_name in header
        if (
            isinstance(source_name, str)
            and source_name.startswith("mtp.")
        )
    }
    missing_headers = expected_sources - header_mtp_sources
    unexpected_headers = header_mtp_sources - expected_sources
    if missing_headers:
        raise ValueError(
            "missing MTP shard header sources: "
            + ", ".join(sorted(missing_headers))
        )
    if unexpected_headers:
        raise ValueError(
            "unexpected MTP shard header sources: "
            + ", ".join(sorted(unexpected_headers))
        )

    tensors = []
    shards = set()
    payload_bytes = 0
    for source_name in sorted(expected_sources):
        shard_name = _validate_shard_name(
            weight_map[source_name]
        )
        shards.add(shard_name)
        header = shard_headers.get(shard_name)
        if not isinstance(header, Mapping):
            raise ValueError(
                "MTP index shard is missing a header: "
                f"{shard_name}"
            )
        containing_shards = tuple(
            candidate_shard
            for candidate_shard, candidate_header
            in shard_headers.items()
            if (
                isinstance(candidate_header, Mapping)
                and source_name in candidate_header
            )
        )
        if containing_shards != (shard_name,):
            raise ValueError(
                "MTP source shard header does not match index: "
                f"{source_name}"
            )
        metadata = _parse_tensor_metadata(
            source_name,
            header[source_name],
        )
        expected_dtype, expected_shape = contracts[source_name]
        if metadata.dtype != expected_dtype:
            raise ValueError(
                "MTP tensor dtype does not match config: "
                f"{source_name}"
            )
        if metadata.shape != expected_shape:
            raise ValueError(
                "MTP tensor shape does not match config: "
                f"{source_name}"
            )
        destination_path, packed_slot = _DESTINATIONS[
            source_name
        ]
        tensors.append(
            Qwen35MTPCheckpointTensor(
                source_name=source_name,
                shard=shard_name,
                dtype=metadata.dtype,
                shape=metadata.shape,
                data_offsets=metadata.data_offsets,
                destination_path=destination_path,
                packed_slot=packed_slot,
            )
        )
        payload_bytes += (
            metadata.data_offsets[1]
            - metadata.data_offsets[0]
        )
    return Qwen35MTPCheckpointPlan(
        tensors=tuple(tensors),
        shards=tuple(sorted(shards)),
        payload_bytes=payload_bytes,
    )


def _default_weight_loader(destination, source) -> None:
    target = getattr(destination, "data", destination)
    copy = getattr(target, "copy_", None)
    if not callable(copy):
        raise ValueError(
            "MTP checkpoint destination must support copy_"
        )
    copy(source)


def _resolve_destination(module, path: str, packed_slot):
    if not isinstance(path, str) or not path:
        raise ValueError(
            "MTP checkpoint destination path must be non-empty"
        )
    parent = module
    for part in path.split("."):
        if not hasattr(parent, part):
            raise ValueError(
                "missing MTP checkpoint destination: "
                f"{path}"
            )
        parent = getattr(parent, part)
    loader = getattr(parent, "weight_loader", None)
    if callable(loader):
        return parent, loader
    if packed_slot is not None:
        raise ValueError(
            "MTP checkpoint destination must expose "
            f"weight_loader: {path}"
        )
    return parent, _default_weight_loader


def _source_dtype_name(source) -> str:
    value = str(getattr(source, "dtype", ""))
    if value in ("BF16", "bfloat16", "torch.bfloat16"):
        return "BF16"
    return value


def _snapshot_destination(destination):
    detach = getattr(destination, "detach", None)
    if not callable(detach):
        raise ValueError(
            "MTP checkpoint destination must support detach"
        )
    clone = getattr(detach(), "clone", None)
    if not callable(clone):
        raise ValueError(
            "MTP checkpoint destination must support clone"
        )
    return clone()


def _restore_destination(destination, snapshot) -> None:
    target = getattr(destination, "data", destination)
    copy = getattr(target, "copy_", None)
    if not callable(copy):
        raise ValueError(
            "MTP checkpoint destination must support copy_"
        )
    copy(snapshot)


def bind_qwen35_mtp_checkpoint(
    module,
    plan: Qwen35MTPCheckpointPlan,
    tensor_reader,
) -> tuple[str, ...]:
    if type(plan) is not Qwen35MTPCheckpointPlan:
        raise ValueError(
            "plan must be an exact Qwen35MTPCheckpointPlan"
        )
    if not callable(tensor_reader):
        raise ValueError(
            "tensor_reader must be callable"
        )
    resolved = []
    binding_keys = set()
    for tensor in plan.tensors:
        if type(tensor) is not Qwen35MTPCheckpointTensor:
            raise ValueError(
                "plan tensors must be "
                "Qwen35MTPCheckpointTensor values"
            )
        destination, loader = _resolve_destination(
            module,
            tensor.destination_path,
            tensor.packed_slot,
        )
        key = (id(destination), tensor.packed_slot)
        if key in binding_keys:
            raise ValueError(
                "duplicate MTP checkpoint destination binding"
            )
        binding_keys.add(key)
        resolved.append((tensor, destination, loader))

    sources = []
    for tensor, destination, loader in resolved:
        source = tensor_reader(tensor)
        if tuple(getattr(source, "shape", ())) != tensor.shape:
            raise ValueError(
                "MTP checkpoint source shape does not match plan: "
                f"{tensor.source_name}"
            )
        if _source_dtype_name(source) != tensor.dtype:
            raise ValueError(
                "MTP checkpoint source dtype does not match plan: "
                f"{tensor.source_name}"
            )
        sources.append(
            (tensor, destination, loader, source)
        )

    snapshots = {}
    for _, destination, _, _ in sources:
        identity = id(destination)
        if identity not in snapshots:
            snapshots[identity] = (
                destination,
                _snapshot_destination(destination),
            )
    try:
        for tensor, destination, loader, source in sources:
            if tensor.packed_slot is None:
                loader(destination, source)
            else:
                loader(
                    destination,
                    source,
                    tensor.packed_slot,
                )
    except Exception:
        for destination, snapshot in snapshots.values():
            _restore_destination(destination, snapshot)
        raise
    return tuple(sorted(
        tensor.source_name
        for tensor in plan.tensors
    ))


def read_qwen35_mtp_checkpoint_tensor(
    checkpoint_dir,
    tensor: Qwen35MTPCheckpointTensor,
):
    from safetensors import safe_open

    if type(tensor) is not Qwen35MTPCheckpointTensor:
        raise ValueError(
            "tensor must be an exact Qwen35MTPCheckpointTensor"
        )
    try:
        directory = Path(checkpoint_dir).resolve()
    except TypeError as error:
        raise ValueError(
            "checkpoint_dir must be an existing directory"
        ) from error
    if not directory.is_dir():
        raise ValueError(
            "checkpoint_dir must be an existing directory"
        )
    relative = PurePosixPath(tensor.shard)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or "\\" in tensor.shard
        or tensor.shard != str(relative)
        or not tensor.shard.endswith(".safetensors")
    ):
        raise ValueError(
            "MTP checkpoint shard must be a safe relative path"
        )
    shard_path = (
        directory / Path(*relative.parts)
    ).resolve()
    if directory not in shard_path.parents or not shard_path.is_file():
        raise ValueError(
            f"missing MTP checkpoint shard: {tensor.shard}"
        )
    with safe_open(
        shard_path,
        framework="pt",
        device="cpu",
    ) as handle:
        if tensor.source_name not in set(handle.keys()):
            raise ValueError(
                "missing MTP checkpoint source: "
                f"{tensor.source_name}"
            )
        return handle.get_tensor(tensor.source_name)
