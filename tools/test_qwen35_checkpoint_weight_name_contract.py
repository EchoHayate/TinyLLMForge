from __future__ import annotations

import builtins
from dataclasses import FrozenInstanceError
import importlib.util
from pathlib import Path
import sys
import types


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    module_name = "qwen35_checkpoint_for_test"
    path = ROOT / "tinyvllm/models/qwen35_checkpoint.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


checkpoint = _load_module()

Qwen35CheckpointLoadTarget = checkpoint.Qwen35CheckpointLoadTarget
Qwen35CheckpointSkip = checkpoint.Qwen35CheckpointSkip
Qwen35CheckpointSource = checkpoint.Qwen35CheckpointSource
Qwen35CheckpointWeightPlan = checkpoint.Qwen35CheckpointWeightPlan
Qwen35CheckpointTensorLoad = checkpoint.Qwen35CheckpointTensorLoad
Qwen35CheckpointTensorMetadata = checkpoint.Qwen35CheckpointTensorMetadata
Qwen35CheckpointTensorPlan = checkpoint.Qwen35CheckpointTensorPlan
build_qwen35_checkpoint_weight_plan = (
    checkpoint.build_qwen35_checkpoint_weight_plan
)
build_qwen35_checkpoint_tensor_plan = (
    checkpoint.build_qwen35_checkpoint_tensor_plan
)

SHARD = "model.safetensors-00001-of-00001.safetensors"
ROOT_SUFFIXES = (
    "embed_tokens.weight",
    "norm.weight",
)
SHARED_SUFFIXES = (
    "input_layernorm.weight",
    "mlp.down_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "post_attention_layernorm.weight",
)
LINEAR_SUFFIXES = (
    "linear_attn.A_log",
    "linear_attn.conv1d.weight",
    "linear_attn.dt_bias",
    "linear_attn.in_proj_a.weight",
    "linear_attn.in_proj_b.weight",
    "linear_attn.in_proj_qkv.weight",
    "linear_attn.in_proj_z.weight",
    "linear_attn.norm.weight",
    "linear_attn.out_proj.weight",
)
FULL_SUFFIXES = (
    "self_attn.k_norm.weight",
    "self_attn.k_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.q_proj.weight",
    "self_attn.v_proj.weight",
)
QWEN38_MTP_SOURCES = (
    "mtp.fc.weight",
    "mtp.layers.0.input_layernorm.weight",
    "mtp.layers.0.mlp.down_proj.weight",
    "mtp.layers.0.mlp.gate_proj.weight",
    "mtp.layers.0.mlp.up_proj.weight",
    "mtp.layers.0.post_attention_layernorm.weight",
    "mtp.layers.0.self_attn.k_norm.weight",
    "mtp.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight",
    "mtp.layers.0.self_attn.q_norm.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight",
    "mtp.norm.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
)


def _config(
    layer_types=("linear_attention", "full_attention"),
    *,
    num_hidden_layers=None,
    tie_word_embeddings=True,
    **overrides,
):
    if num_hidden_layers is None:
        num_hidden_layers = len(layer_types)
    values = dict(
        num_hidden_layers=num_hidden_layers,
        layer_types=layer_types,
        tie_word_embeddings=tie_word_embeddings,
        dtype="bfloat16",
        hidden_size=8,
        intermediate_size=12,
        vocab_size=32,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=3,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=5,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
    )
    values.update(overrides)
    text_config = types.SimpleNamespace(**values)
    return types.SimpleNamespace(text_config=text_config)


def _text_names(layer_types):
    names = {
        f"model.language_model.{suffix}"
        for suffix in ROOT_SUFFIXES
    }
    for layer_index, layer_type in enumerate(layer_types):
        prefix = f"model.language_model.layers.{layer_index}."
        names.update(prefix + suffix for suffix in SHARED_SUFFIXES)
        suffixes = (
            LINEAR_SUFFIXES
            if layer_type == "linear_attention"
            else FULL_SUFFIXES
        )
        names.update(prefix + suffix for suffix in suffixes)
    return names


def _index_for(
    layer_types=("linear_attention", "full_attention"),
    *,
    visual_count=0,
    mtp_count=0,
    tie_word_embeddings=True,
):
    names = _text_names(layer_types)
    if tie_word_embeddings is False:
        names.add("lm_head.weight")
    names.update(
        f"model.visual.synthetic.{index}.weight"
        for index in range(visual_count)
    )
    names.update(
        f"mtp.synthetic.{index}.weight"
        for index in range(mtp_count)
    )
    return {
        "metadata": {"total_size": 1},
        "weight_map": {
            name: SHARD
            for name in sorted(names)
        },
    }


def _qwen38_profile(config):
    text = config.text_config
    return types.SimpleNamespace(
        text_model_type=text.model_type,
        num_hidden_layers=text.num_hidden_layers,
        hidden_size=text.hidden_size,
        intermediate_size=text.intermediate_size,
        vocab_size=text.vocab_size,
        dtype=text.dtype,
        tie_word_embeddings=text.tie_word_embeddings,
        layer_types=tuple(text.layer_types),
        base_decode_auxiliary_skip_sources=QWEN38_MTP_SOURCES,
    )


def _qwen38_index(config):
    payload = _index_for(
        tuple(config.text_config.layer_types),
        tie_word_embeddings=config.text_config.tie_word_embeddings,
    )
    payload["weight_map"].update({
        source_name: SHARD
        for source_name in QWEN38_MTP_SOURCES
    })
    return payload


_DTYPE_BYTES = {
    "BF16": 2,
    "F32": 4,
}


def _expected_metadata(config, source_name):
    text = config.text_config
    hidden_size = text.hidden_size
    intermediate_size = text.intermediate_size
    query_width = (
        text.num_attention_heads * 2 * text.head_dim
    )
    kv_width = text.num_key_value_heads * text.head_dim
    key_width = (
        text.linear_num_key_heads * text.linear_key_head_dim
    )
    value_width = (
        text.linear_num_value_heads * text.linear_value_head_dim
    )
    conv_width = 2 * key_width + value_width
    compute_dtype = {
        "bfloat16": "BF16",
        "float32": "F32",
    }[text.dtype]
    if source_name == "lm_head.weight":
        return compute_dtype, (text.vocab_size, hidden_size)
    suffix = source_name.removeprefix("model.language_model.")
    if suffix == "embed_tokens.weight":
        return compute_dtype, (text.vocab_size, hidden_size)
    if suffix == "norm.weight":
        return compute_dtype, (hidden_size,)
    suffix = suffix.split(".", 2)[2]
    expected = {
        "input_layernorm.weight": (compute_dtype, (hidden_size,)),
        "post_attention_layernorm.weight": (
            compute_dtype,
            (hidden_size,),
        ),
        "mlp.gate_proj.weight": (
            compute_dtype,
            (intermediate_size, hidden_size),
        ),
        "mlp.up_proj.weight": (
            compute_dtype,
            (intermediate_size, hidden_size),
        ),
        "mlp.down_proj.weight": (
            compute_dtype,
            (hidden_size, intermediate_size),
        ),
        "linear_attn.in_proj_qkv.weight": (
            compute_dtype,
            (conv_width, hidden_size),
        ),
        "linear_attn.in_proj_z.weight": (
            compute_dtype,
            (value_width, hidden_size),
        ),
        "linear_attn.in_proj_b.weight": (
            compute_dtype,
            (text.linear_num_value_heads, hidden_size),
        ),
        "linear_attn.in_proj_a.weight": (
            compute_dtype,
            (text.linear_num_value_heads, hidden_size),
        ),
        "linear_attn.out_proj.weight": (
            compute_dtype,
            (hidden_size, value_width),
        ),
        "linear_attn.conv1d.weight": (
            compute_dtype,
            (
                conv_width,
                1,
                text.linear_conv_kernel_dim,
            ),
        ),
        "linear_attn.A_log": (
            "F32",
            (text.linear_num_value_heads,),
        ),
        "linear_attn.dt_bias": (
            compute_dtype,
            (text.linear_num_value_heads,),
        ),
        "linear_attn.norm.weight": (
            "F32",
            (text.linear_value_head_dim,),
        ),
        "self_attn.q_proj.weight": (
            compute_dtype,
            (query_width, hidden_size),
        ),
        "self_attn.k_proj.weight": (
            compute_dtype,
            (kv_width, hidden_size),
        ),
        "self_attn.v_proj.weight": (
            compute_dtype,
            (kv_width, hidden_size),
        ),
        "self_attn.o_proj.weight": (
            compute_dtype,
            (
                hidden_size,
                text.num_attention_heads * text.head_dim,
            ),
        ),
        "self_attn.q_norm.weight": (
            compute_dtype,
            (text.head_dim,),
        ),
        "self_attn.k_norm.weight": (
            compute_dtype,
            (text.head_dim,),
        ),
    }
    return expected[suffix]


def _header_for(config, payload):
    offset = 0
    header = {}
    for source_name in sorted(payload["weight_map"]):
        if (
            source_name.startswith("model.language_model.")
            or source_name == "lm_head.weight"
        ):
            dtype, shape = _expected_metadata(config, source_name)
        else:
            dtype, shape = "BF16", (1,)
        size = _DTYPE_BYTES[dtype]
        for dimension in shape:
            size *= dimension
        header[source_name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + size],
        }
        offset += size
    payload["metadata"]["total_size"] = offset
    return {SHARD: header}


def _tensor_plan(
    layer_types=("linear_attention", "full_attention"),
    *,
    visual_count=0,
    mtp_count=0,
):
    config = _config(layer_types)
    payload = _index_for(
        layer_types,
        visual_count=visual_count,
        mtp_count=mtp_count,
    )
    headers = _header_for(config, payload)
    return (
        config,
        payload,
        headers,
        build_qwen35_checkpoint_tensor_plan(
            config,
            payload,
            headers,
        ),
    )


def _loads_by_source(plan):
    return {
        entry.source.name: entry
        for entry in plan.loads
    }


def _expect_error(function, message):
    try:
        function()
    except ValueError as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(f"expected ValueError containing: {message}")


def test_exact_interleaved_mapping():
    payload = _index_for()
    plan = build_qwen35_checkpoint_weight_plan(_config(), payload)

    assert type(plan) is Qwen35CheckpointWeightPlan
    assert len(plan.loads) == 27
    assert plan.skips == ()
    assert plan.shards == (SHARD,)
    assert tuple(
        entry.source.name for entry in plan.loads
    ) == tuple(sorted(payload["weight_map"]))

    loads = _loads_by_source(plan)
    assert loads[
        "model.language_model.embed_tokens.weight"
    ].target == "embed_tokens.weight"
    assert loads[
        "model.language_model.norm.weight"
    ].target == "final_norm.weight"

    gate = loads[
        "model.language_model.layers.0.mlp.gate_proj.weight"
    ]
    up = loads[
        "model.language_model.layers.0.mlp.up_proj.weight"
    ]
    assert gate.target == "layers.0.mlp.gate_up_proj.weight"
    assert gate.packed_slot == 0
    assert up.target == gate.target
    assert up.packed_slot == 1

    assert loads[
        "model.language_model.layers.0.linear_attn.conv1d.weight"
    ].target == "layers.0.linear_attention.conv_weight"
    assert loads[
        "model.language_model.layers.0.linear_attn.norm.weight"
    ].target == "layers.0.linear_attention.norm_weight"
    assert loads[
        "model.language_model.layers.1.self_attn.q_proj.weight"
    ].target == "layers.1.full_attention.q_projection.weight"
    assert loads[
        "model.language_model.layers.1.self_attn.k_proj.weight"
    ].target == "layers.1.full_attention.k_projection.weight"
    assert loads[
        "model.language_model.layers.1.self_attn.v_proj.weight"
    ].target == "layers.1.full_attention.v_projection.weight"
    assert all(
        entry.packed_slot is None
        for entry in plan.loads
        if ".self_attn." in entry.source.name
    )


def test_explicit_skip_and_total_coverage():
    payload = _index_for(visual_count=2, mtp_count=2)
    plan = build_qwen35_checkpoint_weight_plan(_config(), payload)

    assert sum(entry.scope == "visual" for entry in plan.skips) == 2
    assert sum(entry.scope == "mtp" for entry in plan.skips) == 2
    assert len(plan.loads) + len(plan.skips) == len(
        payload["weight_map"]
    )
    observed = {
        entry.source.name for entry in plan.loads
    } | {
        entry.source.name for entry in plan.skips
    }
    assert observed == set(payload["weight_map"])
    assert tuple(
        entry.source.name for entry in plan.skips
    ) == tuple(sorted(
        name
        for name in payload["weight_map"]
        if not name.startswith("model.language_model.")
    ))


def test_qwen38_base_decode_accepts_only_declared_mtp_inventory():
    config = _config(
        tie_word_embeddings=False,
        model_type="qwen3_5_text",
    )
    payload = _qwen38_index(config)
    profile = _qwen38_profile(config)

    plan = build_qwen35_checkpoint_weight_plan(
        config,
        payload,
        qwen38_text_profile=profile,
    )

    assert tuple(
        entry.source.name
        for entry in plan.skips
        if entry.scope == "mtp"
    ) == QWEN38_MTP_SOURCES
    assert all(
        entry.scope in {"visual", "mtp"}
        for entry in plan.skips
    )

    missing = {
        "metadata": dict(payload["metadata"]),
        "weight_map": dict(payload["weight_map"]),
    }
    missing["weight_map"].pop(QWEN38_MTP_SOURCES[0])
    _expect_error(
        lambda: build_qwen35_checkpoint_weight_plan(
            config,
            missing,
            qwen38_text_profile=profile,
        ),
        "Qwen3.8 base-decode auxiliary checkpoint inventory mismatch",
    )

    extra = {
        "metadata": dict(payload["metadata"]),
        "weight_map": dict(payload["weight_map"]),
    }
    extra["weight_map"]["mtp.extra.weight"] = SHARD
    _expect_error(
        lambda: build_qwen35_checkpoint_weight_plan(
            config,
            extra,
            qwen38_text_profile=profile,
        ),
        "Qwen3.8 base-decode auxiliary checkpoint inventory mismatch",
    )


def test_untied_lm_head_mapping_and_tensor_contract():
    config = _config(tie_word_embeddings=False)
    payload = _index_for(tie_word_embeddings=False)
    headers = _header_for(config, payload)

    weight_plan = build_qwen35_checkpoint_weight_plan(
        config,
        payload,
    )
    tensor_plan = build_qwen35_checkpoint_tensor_plan(
        config,
        payload,
        headers,
    )

    weight = _loads_by_source(weight_plan)["lm_head.weight"]
    assert weight.target == "lm_head.weight"
    assert weight.packed_slot is None
    tensor = {
        entry.weight.source.name: entry
        for entry in tensor_plan.loads
    }["lm_head.weight"]
    assert tensor.metadata.dtype == "BF16"
    assert tensor.metadata.shape == (32, 8)
    assert tensor.transform == "identity"


def test_language_grammar_fails_closed():
    base = _index_for()

    missing = {
        "weight_map": dict(base["weight_map"]),
    }
    del missing["weight_map"][
        "model.language_model.layers.0.linear_attn.A_log"
    ]
    _expect_error(
        lambda: build_qwen35_checkpoint_weight_plan(_config(), missing),
        "missing language-model weights",
    )

    extra = {
        "weight_map": dict(base["weight_map"]),
    }
    extra["weight_map"][
        "model.language_model.layers.0.linear_attn.unknown"
    ] = SHARD
    _expect_error(
        lambda: build_qwen35_checkpoint_weight_plan(_config(), extra),
        "unexpected language-model weights",
    )

    wrong_full = {
        "weight_map": dict(base["weight_map"]),
    }
    del wrong_full["weight_map"][
        "model.language_model.layers.1.self_attn.q_proj.weight"
    ]
    wrong_full["weight_map"][
        "model.language_model.layers.1.linear_attn.in_proj_qkv.weight"
    ] = SHARD
    _expect_error(
        lambda: build_qwen35_checkpoint_weight_plan(
            _config(),
            wrong_full,
        ),
        "language-model weight set does not match config topology",
    )

    out_of_range = {
        "weight_map": dict(base["weight_map"]),
    }
    out_of_range["weight_map"][
        "model.language_model.layers.2.input_layernorm.weight"
    ] = SHARD
    _expect_error(
        lambda: build_qwen35_checkpoint_weight_plan(
            _config(),
            out_of_range,
        ),
        "unexpected language-model weights",
    )

    untied_missing_head = _index_for(tie_word_embeddings=False)
    del untied_missing_head["weight_map"]["lm_head.weight"]
    _expect_error(
        lambda: build_qwen35_checkpoint_weight_plan(
            _config(tie_word_embeddings=False),
            untied_missing_head,
        ),
        "missing language-model weights",
    )
    tied_extra_head = _index_for()
    tied_extra_head["weight_map"]["lm_head.weight"] = SHARD
    _expect_error(
        lambda: build_qwen35_checkpoint_weight_plan(
            _config(),
            tied_extra_head,
        ),
        "unexpected language-model weights",
    )
    _expect_error(
        lambda: build_qwen35_checkpoint_weight_plan(
            _config(tie_word_embeddings="false"),
            base,
        ),
        "tie_word_embeddings must be a bool",
    )
    _expect_error(
        lambda: build_qwen35_checkpoint_weight_plan(
            _config(("linear_attention", "unsupported")),
            base,
        ),
        "unsupported Qwen3.5 layer type",
    )
    _expect_error(
        lambda: build_qwen35_checkpoint_weight_plan(
            _config(num_hidden_layers=3),
            base,
        ),
        "layer_types length must match num_hidden_layers",
    )


def test_malformed_index_and_shards_fail_closed():
    config = _config()
    for payload in ({}, {"weight_map": {}}, {"weight_map": []}):
        _expect_error(
            lambda payload=payload: build_qwen35_checkpoint_weight_plan(
                config,
                payload,
            ),
            "weight_map must be a non-empty mapping",
        )

    cases = (
        ({"": SHARD}, "source name must be a non-empty string"),
        ({1: SHARD}, "source name must be a non-empty string"),
        ({"model.visual.x": ""}, "shard name must be a non-empty string"),
        ({"model.visual.x": 1}, "shard name must be a non-empty string"),
        ({"model.visual.x": "/tmp/a.safetensors"}, "safe relative path"),
        ({"model.visual.x": "../a.safetensors"}, "safe relative path"),
        ({"model.visual.x": "a.bin"}, "must end in .safetensors"),
        ({"unknown.weight": SHARD}, "unsupported checkpoint scope"),
    )
    for weight_map, message in cases:
        _expect_error(
            lambda weight_map=weight_map: (
                build_qwen35_checkpoint_weight_plan(
                    config,
                    {"weight_map": weight_map},
                )
            ),
            message,
        )


def test_official_qwen35_2b_topology_counts():
    layer_types = tuple(
        "full_attention"
        if (index + 1) % 4 == 0
        else "linear_attention"
        for index in range(24)
    )
    payload = _index_for(
        layer_types,
        visual_count=297,
        mtp_count=15,
    )
    plan = build_qwen35_checkpoint_weight_plan(
        _config(layer_types),
        payload,
    )

    assert len(plan.loads) == 320
    assert len(plan.skips) == 312
    assert sum(entry.scope == "visual" for entry in plan.skips) == 297
    assert sum(entry.scope == "mtp" for entry in plan.skips) == 15
    assert len(plan.loads) + len(plan.skips) == 632
    assert len({
        (entry.target, entry.packed_slot)
        for entry in plan.loads
    }) == 320


def test_read_only_and_frozen_contract():
    payload = _index_for(visual_count=1, mtp_count=1)
    before = {
        key: dict(value) if isinstance(value, dict) else value
        for key, value in payload.items()
    }
    original_open = builtins.open

    def forbidden_open(*args, **kwargs):
        raise AssertionError("planner must not open files")

    builtins.open = forbidden_open
    try:
        plan = build_qwen35_checkpoint_weight_plan(_config(), payload)
    finally:
        builtins.open = original_open

    assert payload == before
    frozen_values = (
        Qwen35CheckpointSource("x", SHARD),
        Qwen35CheckpointLoadTarget(
            Qwen35CheckpointSource("x", SHARD),
            "target",
            None,
        ),
        Qwen35CheckpointSkip(
            Qwen35CheckpointSource("x", SHARD),
            "visual",
        ),
        plan,
    )
    for value in frozen_values:
        try:
            value.test_mutation = True
        except (FrozenInstanceError, AttributeError):
            pass
        else:
            raise AssertionError(f"{type(value).__name__} is not frozen")


def test_tensor_metadata_and_transforms():
    config, payload, headers, plan = _tensor_plan(
        visual_count=2,
        mtp_count=2,
    )
    assert type(plan) is Qwen35CheckpointTensorPlan
    assert len(plan.loads) == 27
    assert len(plan.skips) == 4
    assert plan.payload_bytes == payload["metadata"]["total_size"]
    loads = {
        entry.weight.source.name: entry
        for entry in plan.loads
    }
    conv = loads[
        "model.language_model.layers.0.linear_attn.conv1d.weight"
    ]
    assert conv.metadata.dtype == "BF16"
    assert conv.metadata.shape == (20, 1, 5)
    assert conv.transform == "squeeze_conv_channel"
    assert loads[
        "model.language_model.layers.0.linear_attn.A_log"
    ].metadata.dtype == "F32"
    assert loads[
        "model.language_model.layers.0.linear_attn.norm.weight"
    ].metadata == Qwen35CheckpointTensorMetadata(
        "F32",
        (4,),
        tuple(headers[SHARD][
            "model.language_model.layers.0.linear_attn.norm.weight"
        ]["data_offsets"]),
    )
    assert all(
        entry.transform == "identity"
        for entry in plan.loads
        if entry is not conv
    )
    assert all(
        type(entry) is Qwen35CheckpointTensorLoad
        for entry in plan.loads
    )
    assert config.text_config.hidden_size == 8


def test_tensor_shape_and_dtype_fail_closed():
    cases = (
        (
            "model.language_model.embed_tokens.weight",
            "shape",
            [16, 16],
            "tensor shape does not match config",
        ),
        (
            "model.language_model.layers.0.mlp.gate_proj.weight",
            "shape",
            [8, 12],
            "tensor shape does not match config",
        ),
        (
            "model.language_model.layers.0.linear_attn.A_log",
            "dtype_shape",
            ("BF16", [4]),
            "tensor dtype does not match config",
        ),
        (
            "model.language_model.layers.0.linear_attn.norm.weight",
            "dtype_shape",
            ("BF16", [8]),
            "tensor dtype does not match config",
        ),
        (
            "model.language_model.layers.0.linear_attn.conv1d.weight",
            "shape",
            [20, 5, 1],
            "tensor shape does not match config",
        ),
        (
            "model.language_model.layers.1.self_attn.q_proj.weight",
            "shape",
            [8, 16],
            "tensor shape does not match config",
        ),
    )
    for source_name, field, value, message in cases:
        config = _config()
        payload = _index_for()
        headers = _header_for(config, payload)
        if field == "dtype_shape":
            dtype, shape = value
            headers[SHARD][source_name]["dtype"] = dtype
            headers[SHARD][source_name]["shape"] = shape
        else:
            headers[SHARD][source_name][field] = value
        _expect_error(
            lambda config=config, payload=payload, headers=headers: (
                build_qwen35_checkpoint_tensor_plan(
                    config,
                    payload,
                    headers,
                )
            ),
            message,
        )


def test_tensor_header_structure_fails_closed():
    config = _config()
    payload = _index_for()
    headers = _header_for(config, payload)

    _expect_error(
        lambda: build_qwen35_checkpoint_tensor_plan(
            config,
            payload,
            {},
        ),
        "shard header set must match weight plan",
    )
    _expect_error(
        lambda: build_qwen35_checkpoint_tensor_plan(
            config,
            payload,
            {**headers, "extra.safetensors": {}},
        ),
        "shard header set must match weight plan",
    )

    source = "model.language_model.embed_tokens.weight"
    malformed_cases = (
        (
            lambda value: value.pop(source),
            "header source set must match index",
        ),
        (
            lambda value: value.__setitem__(
                "model.visual.extra.weight",
                {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]},
            ),
            "header source set must match index",
        ),
        (
            lambda value: value.__setitem__(source, []),
            "tensor metadata must be a mapping",
        ),
        (
            lambda value: value[source].__setitem__("dtype", "F16"),
            "unsupported tensor dtype",
        ),
        (
            lambda value: value[source].__setitem__("shape", [0, 8]),
            "tensor shape must contain positive integers",
        ),
        (
            lambda value: value[source].__setitem__("shape", [True, 8]),
            "tensor shape must contain positive integers",
        ),
        (
            lambda value: value[source].__setitem__(
                "data_offsets",
                [1],
            ),
            "data_offsets must contain two integers",
        ),
        (
            lambda value: value[source].__setitem__(
                "data_offsets",
                [0, 1],
            ),
            "tensor byte count does not match metadata",
        ),
    )
    for mutate, message in malformed_cases:
        local_headers = {
            SHARD: {
                key: dict(value)
                for key, value in headers[SHARD].items()
            },
        }
        mutate(local_headers[SHARD])
        _expect_error(
            lambda local_headers=local_headers: (
                build_qwen35_checkpoint_tensor_plan(
                    config,
                    payload,
                    local_headers,
                )
            ),
            message,
        )

    ordered = sorted(headers[SHARD])
    overlap = {
        SHARD: {
            key: dict(value)
            for key, value in headers[SHARD].items()
        },
    }
    overlap_start, overlap_end = (
        overlap[SHARD][ordered[1]]["data_offsets"]
    )
    overlap_size = overlap_end - overlap_start
    overlap[SHARD][ordered[1]]["data_offsets"] = [
        overlap_start - 2,
        overlap_start - 2 + overlap_size,
    ]
    _expect_error(
        lambda: build_qwen35_checkpoint_tensor_plan(
            config,
            payload,
            overlap,
        ),
        "tensor payload intervals must be contiguous",
    )

    hole = {
        SHARD: {
            key: dict(value)
            for key, value in headers[SHARD].items()
        },
    }
    start, end = hole[SHARD][ordered[1]]["data_offsets"]
    hole[SHARD][ordered[1]]["data_offsets"] = [start + 2, end + 2]
    _expect_error(
        lambda: build_qwen35_checkpoint_tensor_plan(
            config,
            payload,
            hole,
        ),
        "tensor payload intervals must be contiguous",
    )

    bad_total = {
        "metadata": {
            "total_size": payload["metadata"]["total_size"] + 2,
        },
        "weight_map": dict(payload["weight_map"]),
    }
    _expect_error(
        lambda: build_qwen35_checkpoint_tensor_plan(
            config,
            bad_total,
            headers,
        ),
        "payload byte total must match index metadata",
    )


def test_tensor_plan_is_read_only_and_frozen():
    config = _config()
    payload = _index_for(visual_count=1, mtp_count=1)
    headers = _header_for(config, payload)
    payload_before = {
        "metadata": dict(payload["metadata"]),
        "weight_map": dict(payload["weight_map"]),
    }
    headers_before = {
        shard: {
            name: dict(metadata)
            for name, metadata in header.items()
        }
        for shard, header in headers.items()
    }
    original_open = builtins.open
    builtins.open = lambda *args, **kwargs: (
        _ for _ in ()
    ).throw(AssertionError("tensor planner must not open files"))
    try:
        plan = build_qwen35_checkpoint_tensor_plan(
            config,
            payload,
            headers,
        )
    finally:
        builtins.open = original_open
    assert payload == payload_before
    assert headers == headers_before
    for value in (
        plan.loads[0].metadata,
        plan.loads[0],
        plan,
    ):
        try:
            value.test_mutation = True
        except (FrozenInstanceError, AttributeError):
            pass
        else:
            raise AssertionError(f"{type(value).__name__} is not frozen")


def main():
    tests = (
        test_exact_interleaved_mapping,
        test_explicit_skip_and_total_coverage,
        test_language_grammar_fails_closed,
        test_malformed_index_and_shards_fail_closed,
        test_official_qwen35_2b_topology_counts,
        test_read_only_and_frozen_contract,
        test_tensor_metadata_and_transforms,
        test_tensor_shape_and_dtype_fail_closed,
        test_tensor_header_structure_fails_closed,
        test_tensor_plan_is_read_only_and_frozen,
    )
    for test in tests:
        test()
    print(
        "qwen35 checkpoint weight-name contract tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    main()
