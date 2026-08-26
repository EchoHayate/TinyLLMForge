from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
MODEL_RUNNER_PATH = ROOT / "tinyvllm/engine/model_runner.py"


def _load_function(name, **namespace):
    tree = ast.parse(
        MODEL_RUNNER_PATH.read_text(encoding="utf-8"),
        filename=str(MODEL_RUNNER_PATH),
    )
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    exec(
        compile(
            ast.Module(body=[function], type_ignores=[]),
            str(MODEL_RUNNER_PATH),
            "exec",
        ),
        namespace,
    )
    return namespace[name]


def _load_model_runner_method(name, **namespace):
    tree = ast.parse(
        MODEL_RUNNER_PATH.read_text(encoding="utf-8"),
        filename=str(MODEL_RUNNER_PATH),
    )
    model_runner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ModelRunner"
    )
    method = next(
        node
        for node in model_runner.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    compiled = {}
    compiled.update(namespace)
    exec(
        compile(
            ast.Module(body=[method], type_ignores=[]),
            str(MODEL_RUNNER_PATH),
            "exec",
        ),
        compiled,
    )
    return compiled[name]


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


def test_qwen38_correctness_rank_identity_reports_loaded_partition_and_gpu():
    cuda = SimpleNamespace(
        current_device=lambda: 2,
        get_device_properties=lambda index: (
            SimpleNamespace(uuid=f"GPU-device-{index}")
        ),
    )
    identity = _load_model_runner_method(
        "qwen38_correctness_rank_identity",
        torch=SimpleNamespace(cuda=cuda),
    )
    runner = SimpleNamespace(
        rank=2,
        world_size=4,
        qwen38_correctness_weight_partition_identity={
            "expected_weight_shard_sha256": "1" * 64,
            "loaded_weight_shard_sha256": "1" * 64,
        },
    )

    assert identity(runner) == {
        "rank": 2,
        "gpu_index": 2,
        "gpu_uuid": "GPU-device-2",
        "expected_weight_shard_sha256": "1" * 64,
        "loaded_weight_shard_sha256": "1" * 64,
    }


def test_qwen38_correctness_rank_identity_normalizes_torch_cuuid():
    class TorchCudaUuid:

        def __str__(self):
            return "7dc22583-df04-6c76-4ba5-ea32c428c130"

    cuda = SimpleNamespace(
        current_device=lambda: 2,
        get_device_properties=lambda _index: (
            SimpleNamespace(uuid=TorchCudaUuid())
        ),
    )
    identity = _load_model_runner_method(
        "qwen38_correctness_rank_identity",
        torch=SimpleNamespace(cuda=cuda),
    )
    runner = SimpleNamespace(
        rank=2,
        world_size=4,
        qwen38_correctness_weight_partition_identity={
            "expected_weight_shard_sha256": "1" * 64,
            "loaded_weight_shard_sha256": "1" * 64,
        },
    )

    assert identity(runner)["gpu_uuid"] == (
        "GPU-7dc22583-df04-6c76-4ba5-ea32c428c130"
    )


def test_qwen38_correctness_rank_identity_fails_without_load_attestation():
    cuda = SimpleNamespace(
        current_device=lambda: 0,
        get_device_properties=lambda _index: (
            SimpleNamespace(uuid="GPU-device-0")
        ),
    )
    identity = _load_model_runner_method(
        "qwen38_correctness_rank_identity",
        torch=SimpleNamespace(cuda=cuda),
    )
    runner = SimpleNamespace(
        rank=0,
        world_size=1,
        qwen38_correctness_weight_partition_identity=None,
    )

    try:
        identity(runner)
    except RuntimeError as error:
        assert "load attestation" in str(error)
    else:
        raise AssertionError("missing rank-local load attestation was accepted")


def test_qwen38_partition_attestation_is_deterministic_and_rank_local():
    attest = _load_function(
        "_qwen38_checkpoint_partition_attestation",
        hashlib=hashlib,
        json=json,
    )
    source = SimpleNamespace(
        name="model.layers.0.weight",
        shard="model-00001-of-00018.safetensors",
    )
    metadata = SimpleNamespace(
        dtype="BF16",
        shape=(8, 4),
        data_offsets=(0, 64),
    )
    load = SimpleNamespace(
        weight=SimpleNamespace(
            source=source,
            target="layers.0.weight",
            packed_slot=None,
        ),
        metadata=metadata,
        transform="identity",
    )
    binding = SimpleNamespace(
        load=load,
        destination_name="layer_stack.layers.0.weight",
        destination_kind="parameter",
        loader_kind="row_parallel",
        local_shape=(8, 1),
        destination_slice=None,
        source_segments=None,
    )
    candidate = SimpleNamespace(
        binding_plan=SimpleNamespace(
            bindings=(binding,),
            tensor_parallel_size=4,
            tensor_parallel_rank=2,
        ),
        stats=SimpleNamespace(
            assigned_bindings=1,
            source_tensors=1,
            shard_count=1,
            loaded_bytes=64,
        ),
    )
    manifest_identity = {
        "manifest_sha256": "1" * 64,
        "composite_sha256": "2" * 64,
    }

    first = attest(
        manifest_identity,
        candidate,
        tensor_parallel_size=4,
        tensor_parallel_rank=2,
    )
    second = attest(
        manifest_identity,
        candidate,
        tensor_parallel_size=4,
        tensor_parallel_rank=2,
    )

    assert first == second
    assert first == {
        "expected_weight_shard_sha256": first[
            "expected_weight_shard_sha256"
        ],
        "loaded_weight_shard_sha256": first[
            "expected_weight_shard_sha256"
        ],
    }
    assert len(first["expected_weight_shard_sha256"]) == 64

    candidate.binding_plan.tensor_parallel_rank = 1
    other_rank = attest(
        manifest_identity,
        candidate,
        tensor_parallel_size=4,
        tensor_parallel_rank=1,
    )
    assert (
        other_rank["expected_weight_shard_sha256"]
        != first["expected_weight_shard_sha256"]
    )


def test_qwen38_partition_attestation_rejects_incomplete_strict_load():
    attest = _load_function(
        "_qwen38_checkpoint_partition_attestation",
        hashlib=hashlib,
        json=json,
    )
    source = SimpleNamespace(name="weight", shard="model.safetensors")
    binding = SimpleNamespace(
        load=SimpleNamespace(
            weight=SimpleNamespace(
                source=source,
                target="weight",
                packed_slot=None,
            ),
            metadata=SimpleNamespace(
                dtype="BF16",
                shape=(4, 4),
                data_offsets=(0, 32),
            ),
            transform="identity",
        ),
        destination_name="weight",
        destination_kind="parameter",
        loader_kind="replicated",
        local_shape=(4, 4),
        destination_slice=None,
        source_segments=None,
    )
    candidate = SimpleNamespace(
        binding_plan=SimpleNamespace(
            bindings=(binding,),
            tensor_parallel_size=1,
            tensor_parallel_rank=0,
        ),
        stats=SimpleNamespace(
            assigned_bindings=0,
            source_tensors=1,
            shard_count=1,
            loaded_bytes=32,
        ),
    )

    try:
        attest(
            {
                "manifest_sha256": "1" * 64,
                "composite_sha256": "2" * 64,
            },
            candidate,
            tensor_parallel_size=1,
            tensor_parallel_rank=0,
        )
    except RuntimeError as error:
        assert "coverage" in str(error)
    else:
        raise AssertionError("incomplete strict checkpoint load was attested")
