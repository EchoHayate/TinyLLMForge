from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
MODEL_RUNNER_PATH = ROOT / "tinyvllm/engine/model_runner.py"


def _load_function(name):
    tree = ast.parse(
        MODEL_RUNNER_PATH.read_text(encoding="utf-8"),
        filename=str(MODEL_RUNNER_PATH),
    )
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    namespace = {}
    exec(
        compile(
            ast.Module(body=[function], type_ignores=[]),
            str(MODEL_RUNNER_PATH),
            "exec",
        ),
        namespace,
    )
    return namespace[name]


def test_resolves_qwen38_profile_only_for_official_architecture():
    resolve = _load_function("_resolve_qwen38_text_profile")
    calls = []
    profile = object()
    qwen38 = SimpleNamespace(
        architectures=["Qwen3_5ForConditionalGeneration"],
        text_config=SimpleNamespace(
            num_hidden_layers=64,
            hidden_size=5120,
            intermediate_size=17408,
        ),
    )

    def adopt(value, **identity):
        calls.append((value, identity))
        return profile

    assert resolve(
        qwen38,
        model_dir="/model",
        adopt_qwen38_text=adopt,
        read_source_identity=lambda _path: {
            "repository": "Qwen/Qwen3.8-27B",
            "revision": "a" * 40,
        },
    ) is profile
    assert calls == [(
        qwen38,
        {
            "repository": "Qwen/Qwen3.8-27B",
            "revision": "a" * 40,
        },
    )]
    assert resolve(
        SimpleNamespace(architectures=["Qwen3ForCausalLM"]),
        model_dir="/model",
        adopt_qwen38_text=lambda _value: calls.append("unexpected"),
        read_source_identity=lambda _path: calls.append(
            "unexpected identity read"
        ),
    ) is None
    assert len(calls) == 1


def test_same_architecture_with_other_topology_stays_qwen35():
    resolve = _load_function("_resolve_qwen38_text_profile")
    calls = []
    hf_config = SimpleNamespace(
        architectures=["Qwen3_5ForConditionalGeneration"],
        text_config=SimpleNamespace(
            num_hidden_layers=36,
            hidden_size=2048,
            intermediate_size=6144,
        ),
    )

    assert resolve(
        hf_config,
        model_dir="/model",
        adopt_qwen38_text=lambda value: calls.append(value),
        read_source_identity=lambda _path: calls.append(
            "unexpected identity read"
        ),
    ) is None
    assert calls == []


def test_model_runner_validates_before_distributed_and_checks_each_batch():
    source = MODEL_RUNNER_PATH.read_text(encoding="utf-8")
    init_body = source[
        source.index("class ModelRunner:"):
        source.index("    def bind_kv_block_identity_rows")
    ]
    run_body = source[
        source.index("    def run(self, seqs:"):
        source.index("    @torch.inference_mode()", source.index(
            "    def run(self, seqs:"
        ))
    ]

    assert init_body.index(
        "self.qwen38_text_profile ="
    ) < init_body.index("dist.init_process_group(")
    assert "validate_qwen38_sequence_batch(" in run_body
    assert run_body.index(
        "validate_qwen38_sequence_batch("
    ) < run_body.index("self.bind_kv_block_identity_rows(")
    compact_init_body = "".join(init_body.split())
    assert (
        "qwen38_text_profile=(self.qwen38_text_profile)"
        in compact_init_body
    )


def test_qwen38_manifest_is_accepted_by_shared_checkpoint_loader(tmp_path):
    identity = _load_function("_qwen35_checkpoint_manifest_identity")
    model_root = tmp_path / "model"
    model_root.mkdir()
    files = {
        "config.json": {"sha256": "1" * 64, "size": 1},
        "model.safetensors.index.json": {
            "sha256": "2" * 64,
            "size": 2,
        },
        "model-00001-of-00001.safetensors": {
            "sha256": "3" * 64,
            "size": 3,
        },
    }
    manifest = {
        "schema_version": "tinyllmforge.qwen38-model-manifest.v1",
        "repository": "Qwen/Qwen3.8-27B",
        "resolved_revision": "a" * 40,
        "model_root": str(model_root.resolve()),
        "checkpoint_shards": [
            "model-00001-of-00001.safetensors"
        ],
        "files": files,
    }
    manifest_path = tmp_path / "model_manifest.json"
    manifest_path.write_bytes(
        json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )

    result = identity(
        model_root,
        path_type=Path,
        json_module=json,
        hashlib_module=hashlib,
    )

    assert result["config_sha256"] == "1" * 64
    assert result["index_sha256"] == "2" * 64
    assert result["shards"] == ((
        "model-00001-of-00001.safetensors",
        files["model-00001-of-00001.safetensors"],
    ),)
