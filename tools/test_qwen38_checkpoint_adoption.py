from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = ROOT / "tinyvllm/models/qwen35_checkpoint.py"

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


def _load_checkpoint():
    spec = importlib.util.spec_from_file_location(
        "qwen35_checkpoint_for_qwen38_test",
        CHECKPOINT_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _config():
    layer_types = tuple(
        "full_attention" if (index + 1) % 4 == 0
        else "linear_attention"
        for index in range(64)
    )
    return SimpleNamespace(text_config=SimpleNamespace(
        model_type="qwen3_5_text",
        num_hidden_layers=64,
        layer_types=layer_types,
        hidden_size=5120,
        intermediate_size=17408,
        vocab_size=248320,
        dtype="bfloat16",
        tie_word_embeddings=False,
    ))


def _profile():
    return SimpleNamespace(
        text_model_type="qwen3_5_text",
        num_hidden_layers=64,
        layer_types=_config().text_config.layer_types,
        hidden_size=5120,
        intermediate_size=17408,
        vocab_size=248320,
        dtype="bfloat16",
        tie_word_embeddings=False,
    )


def _index():
    names = {
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
        "lm_head.weight",
        "model.visual.patch_embed.proj.weight",
    }
    for index in range(64):
        prefix = f"model.language_model.layers.{index}."
        names.update(prefix + suffix for suffix in SHARED_SUFFIXES)
        suffixes = (
            FULL_SUFFIXES
            if (index + 1) % 4 == 0
            else LINEAR_SUFFIXES
        )
        names.update(prefix + suffix for suffix in suffixes)
    return {
        "metadata": {"total_size": 1},
        "weight_map": {
            name: "model-00001-of-00001.safetensors"
            for name in sorted(names)
        },
    }


def test_qwen38_untied_lm_head_is_loaded_and_visual_scope_is_skipped():
    checkpoint = _load_checkpoint()

    plan = checkpoint.build_qwen35_checkpoint_weight_plan(
        _config(),
        _index(),
        qwen38_text_profile=_profile(),
    )
    loads = {
        row.source.name: (row.target, row.packed_slot)
        for row in plan.loads
    }

    assert loads["lm_head.weight"] == ("lm_head.weight", None)
    assert loads["model.language_model.embed_tokens.weight"] == (
        "embed_tokens.weight",
        None,
    )
    assert any(
        row.source.name == "model.visual.patch_embed.proj.weight"
        and row.scope == "visual"
        for row in plan.skips
    )
    assert len(loads) == 851


def test_tied_qwen35_checkpoint_does_not_require_lm_head_source():
    checkpoint = _load_checkpoint()
    config = _config()
    config.text_config.tie_word_embeddings = True
    index = _index()
    del index["weight_map"]["lm_head.weight"]

    plan = checkpoint.build_qwen35_checkpoint_weight_plan(
        config,
        index,
    )

    assert all(
        row.source.name != "lm_head.weight"
        for row in plan.loads
    )


def test_tied_checkpoint_rejects_unexpected_lm_head_source():
    checkpoint = _load_checkpoint()
    config = _config()
    config.text_config.tie_word_embeddings = True

    try:
        checkpoint.build_qwen35_checkpoint_weight_plan(
            config,
            _index(),
        )
    except ValueError as error:
        assert "unexpected language-model weights" in str(error)
    else:
        raise AssertionError(
            "tied checkpoint accepted an independent lm_head.weight"
        )


def test_untied_checkpoint_requires_lm_head_source():
    checkpoint = _load_checkpoint()
    index = _index()
    del index["weight_map"]["lm_head.weight"]

    try:
        checkpoint.build_qwen35_checkpoint_weight_plan(
            _config(),
            index,
            qwen38_text_profile=_profile(),
        )
    except ValueError as error:
        assert "missing language-model weights" in str(error)
    else:
        raise AssertionError(
            "untied checkpoint accepted a missing lm_head.weight"
        )


def test_qwen38_profile_must_match_checkpoint_text_config():
    checkpoint = _load_checkpoint()
    profile = _profile()
    profile.hidden_size = 4096

    try:
        checkpoint.build_qwen35_checkpoint_weight_plan(
            _config(),
            _index(),
            qwen38_text_profile=profile,
        )
    except ValueError as error:
        assert "qwen38_text_profile hidden_size" in str(error)
    else:
        raise AssertionError(
            "checkpoint planner accepted profile/config drift"
        )


def test_qwen38_profile_allows_only_visual_checkpoint_skips():
    checkpoint = _load_checkpoint()
    index = _index()
    index["weight_map"]["mtp.synthetic.weight"] = (
        "model-00001-of-00001.safetensors"
    )

    try:
        checkpoint.build_qwen35_checkpoint_weight_plan(
            _config(),
            index,
            qwen38_text_profile=_profile(),
        )
    except ValueError as error:
        assert "Qwen3.8 checkpoint skip scope" in str(error)
    else:
        raise AssertionError(
            "Qwen3.8 checkpoint accepted a non-visual skip"
        )
