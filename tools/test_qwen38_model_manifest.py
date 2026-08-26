from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools/qwen38_model_manifest.py"
REPOSITORY = "Qwen/Qwen3.8-27B"
REVISION = "a" * 40


def _load():
    assert MODULE_PATH.is_file(), "Qwen3.8 model manifest is not implemented"
    spec = importlib.util.spec_from_file_location(
        "qwen38_model_manifest_under_test",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


manifest_module = _load()


def _write_fixture(root: Path):
    layer_types = [
        "full_attention" if (index + 1) % 4 == 0
        else "linear_attention"
        for index in range(64)
    ]
    config = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "language_model_only": False,
        "text_config": {
            "model_type": "qwen3_5_text",
            "num_hidden_layers": 64,
            "hidden_size": 5120,
            "intermediate_size": 17408,
            "dtype": "bfloat16",
            "vocab_size": 248320,
            "tie_word_embeddings": False,
            "layer_types": layer_types,
        },
    }
    (root / "config.json").write_text(
        json.dumps(config, sort_keys=True),
        encoding="utf-8",
    )
    (root / "tokenizer.json").write_text(
        '{"version":"1.0"}\n',
        encoding="utf-8",
    )
    (root / "tokenizer_config.json").write_text(
        '{"tokenizer_class":"Qwen2Tokenizer"}\n',
        encoding="utf-8",
    )
    (root / "model-00001-of-00002.safetensors").write_bytes(b"shard-1")
    (root / "model-00002-of-00002.safetensors").write_bytes(b"shard-2")
    index = {
        "metadata": {"total_size": 14},
        "weight_map": {
            "model.language_model.embed_tokens.weight": (
                "model-00001-of-00002.safetensors"
            ),
            "lm_head.weight": (
                "model-00001-of-00002.safetensors"
            ),
            "model.language_model.layers.0.input_layernorm.weight": (
                "model-00002-of-00002.safetensors"
            ),
        },
    }
    (root / "model.safetensors.index.json").write_text(
        json.dumps(index, sort_keys=True),
        encoding="utf-8",
    )


def test_builds_and_verifies_complete_immutable_manifest(tmp_path):
    _write_fixture(tmp_path)

    payload = manifest_module.build_model_manifest(
        tmp_path,
        repository=REPOSITORY,
        revision=REVISION,
    )
    path = tmp_path.parent / "model_manifest.json"
    path.write_bytes(manifest_module.canonical_bytes(payload))
    verified = manifest_module.verify_model_manifest(
        path,
        model_root=tmp_path,
    )

    assert verified == payload
    assert payload["schema_version"] == (
        "tinyllmforge.qwen38-model-manifest.v1"
    )
    assert payload["repository"] == REPOSITORY
    assert payload["resolved_revision"] == REVISION
    assert payload["config_sha256"] == payload["files"][
        "config.json"
    ]["sha256"]
    assert len(payload["text_config_sha256"]) == 64
    assert len(payload["tokenizer_inventory_sha256"]) == 64
    assert payload["checkpoint_index_sha256"] == payload["files"][
        "model.safetensors.index.json"
    ]["sha256"]
    assert payload["checkpoint_tensor_count"] == 3
    assert payload["tokenizer_files"] == [
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    assert payload["checkpoint_shards"] == [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]
    assert all(
        set(row) == {"sha256", "size"}
        for row in payload["files"].values()
    )


def test_manifest_is_canonical_and_deterministic(tmp_path):
    _write_fixture(tmp_path)

    first = manifest_module.build_model_manifest(
        tmp_path,
        repository=REPOSITORY,
        revision=REVISION,
    )
    second = manifest_module.build_model_manifest(
        tmp_path,
        repository=REPOSITORY,
        revision=REVISION,
    )

    assert manifest_module.canonical_bytes(first) == (
        manifest_module.canonical_bytes(second)
    )
    assert manifest_module.manifest_sha256(first) == (
        manifest_module.manifest_sha256(second)
    )


@pytest.mark.parametrize("revision", ("main", "", "g" * 40, "a" * 39))
def test_rejects_floating_or_invalid_revision(tmp_path, revision):
    _write_fixture(tmp_path)

    with pytest.raises(ValueError, match="immutable revision"):
        manifest_module.build_model_manifest(
            tmp_path,
            repository=REPOSITORY,
            revision=revision,
        )


def test_rejects_wrong_repository(tmp_path):
    _write_fixture(tmp_path)

    with pytest.raises(ValueError, match="repository"):
        manifest_module.build_model_manifest(
            tmp_path,
            repository="Qwen/another-model",
            revision=REVISION,
        )


def test_rejects_missing_tokenizer_inventory(tmp_path):
    _write_fixture(tmp_path)
    (tmp_path / "tokenizer.json").unlink()
    (tmp_path / "tokenizer_config.json").unlink()

    with pytest.raises(ValueError, match="tokenizer"):
        manifest_module.build_model_manifest(
            tmp_path,
            repository=REPOSITORY,
            revision=REVISION,
        )


def test_rejects_missing_or_extra_checkpoint_shards(tmp_path):
    _write_fixture(tmp_path)
    (tmp_path / "model-00002-of-00002.safetensors").unlink()
    with pytest.raises(ValueError, match="checkpoint shard"):
        manifest_module.build_model_manifest(
            tmp_path,
            repository=REPOSITORY,
            revision=REVISION,
        )

    _write_fixture(tmp_path)
    (tmp_path / "unlisted.safetensors").write_bytes(b"extra")
    with pytest.raises(ValueError, match="unlisted checkpoint shard"):
        manifest_module.build_model_manifest(
            tmp_path,
            repository=REPOSITORY,
            revision=REVISION,
        )


def test_rejects_symlink_and_tampered_file(tmp_path):
    _write_fixture(tmp_path)
    external = tmp_path.parent / "external-tokenizer.json"
    external.write_text("{}\n", encoding="utf-8")
    (tmp_path / "tokenizer.json").unlink()
    (tmp_path / "tokenizer.json").symlink_to(external)
    with pytest.raises(ValueError, match="symlink"):
        manifest_module.build_model_manifest(
            tmp_path,
            repository=REPOSITORY,
            revision=REVISION,
        )

    (tmp_path / "tokenizer.json").unlink()
    (tmp_path / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    payload = manifest_module.build_model_manifest(
        tmp_path,
        repository=REPOSITORY,
        revision=REVISION,
    )
    path = tmp_path.parent / "model_manifest.json"
    path.write_bytes(manifest_module.canonical_bytes(payload))
    (tmp_path / "tokenizer.json").write_text(
        "[]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        manifest_module.verify_model_manifest(
            path,
            model_root=tmp_path,
        )


def test_rejects_config_architecture_mismatch(tmp_path):
    _write_fixture(tmp_path)
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["architectures"] = ["SomeOtherArchitecture"]
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="config architecture"):
        manifest_module.build_model_manifest(
            tmp_path,
            repository=REPOSITORY,
            revision=REVISION,
        )


@pytest.mark.parametrize(
    "scope,field,value",
    (
        ("text_config", "vocab_size", 151936),
        ("text_config", "tie_word_embeddings", True),
        ("top_level", "language_model_only", True),
    ),
)
def test_rejects_output_head_config_drift(
    tmp_path,
    scope,
    field,
    value,
):
    _write_fixture(tmp_path)
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    target = (
        config["text_config"]
        if scope == "text_config"
        else config
    )
    target[field] = value
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match=field):
        manifest_module.build_model_manifest(
            tmp_path,
            repository=REPOSITORY,
            revision=REVISION,
        )


def test_rejects_missing_independent_lm_head(tmp_path):
    _write_fixture(tmp_path)
    index_path = tmp_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    del index["weight_map"]["lm_head.weight"]
    index_path.write_text(json.dumps(index), encoding="utf-8")

    with pytest.raises(ValueError, match="lm_head.weight"):
        manifest_module.build_model_manifest(
            tmp_path,
            repository=REPOSITORY,
            revision=REVISION,
        )


def test_rejects_duplicate_json_object_paths(tmp_path):
    _write_fixture(tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(
        '{"model_type":"qwen3_5","model_type":"qwen3_5"}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate JSON key"):
        manifest_module.build_model_manifest(
            tmp_path,
            repository=REPOSITORY,
            revision=REVISION,
        )


def test_verifier_rejects_absolute_or_duplicate_inventory_paths(tmp_path):
    _write_fixture(tmp_path)
    payload = manifest_module.build_model_manifest(
        tmp_path,
        repository=REPOSITORY,
        revision=REVISION,
    )
    path = tmp_path.parent / "model_manifest.json"

    absolute = dict(payload)
    absolute["files"] = dict(payload["files"])
    absolute["files"]["/absolute/config.json"] = absolute["files"].pop(
        "config.json"
    )
    path.write_bytes(manifest_module.canonical_bytes(absolute))
    with pytest.raises(ValueError, match="relative"):
        manifest_module.verify_model_manifest(path, model_root=tmp_path)

    duplicate = dict(payload)
    duplicate["checkpoint_shards"] = [
        payload["checkpoint_shards"][0],
        payload["checkpoint_shards"][0],
        *payload["checkpoint_shards"][1:],
    ]
    path.write_bytes(manifest_module.canonical_bytes(duplicate))
    with pytest.raises(ValueError, match="duplicate"):
        manifest_module.verify_model_manifest(path, model_root=tmp_path)


def test_verifier_rejects_manifest_model_root_drift_with_explicit_root(
    tmp_path,
):
    _write_fixture(tmp_path)
    payload = manifest_module.build_model_manifest(
        tmp_path,
        repository=REPOSITORY,
        revision=REVISION,
    )
    payload["model_root"] = str((tmp_path.parent / "other").resolve())
    path = tmp_path.parent / "model_manifest.json"
    path.write_bytes(manifest_module.canonical_bytes(payload))

    with pytest.raises(ValueError, match="model root"):
        manifest_module.verify_model_manifest(
            path,
            model_root=tmp_path,
        )
