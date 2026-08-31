from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tools.quantized_draft_int4_microgate import DraftLinearShape
from tools.quantized_draft_int4_microgate_worker import (
    build_pair_schedule,
    extract_linear_shape_manifest,
    run_worker,
    validate_worker_arguments,
)
import tools.quantized_draft_int4_microgate_worker as worker


WORKER_PATH = Path(worker.__file__).resolve()


class _FakeLinear:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size


class _FakeNonLinear:
    pass


class _FakeModel:
    def named_modules(self):
        return (
            ("", self),
            ("model.layers.0.self_attn.q_proj", _FakeLinear(1024, 2048)),
            ("model.layers.0.self_attn.o_proj", _FakeLinear(2048, 1024)),
            ("model.layers.0.mlp.gate_proj", _FakeLinear(1024, 2048)),
            ("model.norm", _FakeNonLinear()),
        )


def _checkpoint(tmp_path):
    model = tmp_path / "Qwen3-0___6B"
    model.mkdir()
    (model / "config.json").write_text(
        '{"model_type":"qwen3"}',
        encoding="utf-8",
    )
    (model / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"fixture")
    return model


def _shapes():
    return (
        DraftLinearShape("m1_k1024_n2048_g128", 1024, 2048, 28, 128),
        DraftLinearShape("m1_k2048_n1024_g128", 2048, 1024, 14, 128),
    )


def _valid_worker_args(tmp_path, **overrides):
    approved = tmp_path / "command-timeline-20260818"
    output = approved / "quantized-draft-int4" / "fixture-r1" / "raw"
    values = {
        "model_path": _checkpoint(tmp_path),
        "output_dir": output,
        "approved_remote_root": approved,
        "device": "cuda:0",
        "seed": 20260831,
        "warmup_pairs": 2,
        "measured_pairs": 200,
        "group_size": 128,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_shape_extraction_coalesces_identical_linear_shapes(tmp_path):
    model_path = _checkpoint(tmp_path)

    manifest = extract_linear_shape_manifest(
        model_path=model_path,
        model_loader=lambda path: _FakeModel(),
    )

    names = sorted((
        "model.layers.0.mlp.gate_proj",
        "model.layers.0.self_attn.q_proj",
    ))
    expected_sha = hashlib.sha256(
        json.dumps(
            names,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert manifest["schema_version"] == 1
    assert manifest["checkpoint"]["path"] == str(model_path.resolve())
    assert manifest["shapes"] == [
        {
            "shape_id": "m1_k1024_n2048_g128",
            "input_features": 1024,
            "output_features": 2048,
            "execution_count": 2,
            "group_size": 128,
            "module_names_sha256": expected_sha,
        },
        {
            "shape_id": "m1_k2048_n1024_g128",
            "input_features": 2048,
            "output_features": 1024,
            "execution_count": 1,
            "group_size": 128,
            "module_names_sha256": hashlib.sha256(
                json.dumps(
                    ["model.layers.0.self_attn.o_proj"],
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        },
    ]


def test_shape_extraction_rejects_missing_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="checkpoint"):
        extract_linear_shape_manifest(
            model_path=tmp_path / "missing",
            model_loader=lambda path: _FakeModel(),
        )


def test_shape_extraction_rejects_empty_inventory(tmp_path):
    model_path = _checkpoint(tmp_path)

    with pytest.raises(ValueError, match="linear shape"):
        extract_linear_shape_manifest(
            model_path=model_path,
            model_loader=lambda path: type(
                "EmptyModel",
                (),
                {"named_modules": lambda self: (("", self),)},
            )(),
        )


def test_default_shape_loader_uses_tinyvllm_draft_builder(tmp_path):
    model_path = _checkpoint(tmp_path)
    events = []
    model = object()

    class _DeviceContext:
        def __enter__(self):
            events.append("enter_meta")

        def __exit__(self, error_type, error, traceback):
            events.append("exit_meta")

    fake_torch = SimpleNamespace(
        device=lambda name: (
            events.append(("device", name)) or _DeviceContext()
        ),
    )
    dependencies = SimpleNamespace(
        load_hf_config=lambda path: (
            events.append(("config", path))
            or SimpleNamespace(model_type="qwen3")
        ),
        build_model=lambda config, **kwargs: (
            events.append(("build", config.model_type, kwargs))
            or model
        ),
    )

    loaded = worker._load_tinyvllm_model(
        model_path,
        dependency_builder=lambda: dependencies,
        torch_module=fake_torch,
    )

    assert loaded is model
    assert ("config", str(model_path)) in events
    assert (
        "build",
        "qwen3",
        {"tensor_parallel_rank": 0, "tensor_parallel_size": 1},
    ) in events
    assert events.count("enter_meta") == 1
    assert events.count("exit_meta") == 1


def test_single_rank_process_group_is_owned_and_released(tmp_path):
    events = []

    class _Distributed:
        @staticmethod
        def is_initialized():
            return False

        @staticmethod
        def init_process_group(**kwargs):
            events.append(("init", kwargs))

        @staticmethod
        def destroy_process_group():
            events.append(("destroy",))

    rendezvous = tmp_path / "approved" / ".dist-rendezvous"
    fake_torch = SimpleNamespace(distributed=_Distributed())

    with worker._single_rank_process_group(
        torch_module=fake_torch,
        rendezvous_path=rendezvous,
    ):
        events.append(("body",))

    assert events[0][0] == "init"
    assert events[0][1]["backend"] == "nccl"
    assert events[0][1]["rank"] == 0
    assert events[0][1]["world_size"] == 1
    assert events[0][1]["init_method"].startswith("file://")
    assert events[1] == ("body",)
    assert events[2] == ("destroy",)


def test_schedule_is_position_balanced_and_complete():
    rows = build_pair_schedule(
        shapes=_shapes(),
        warmup_pairs=2,
        measured_pairs=200,
    )

    assert len(rows) == len(_shapes()) * 202
    for shape in _shapes():
        shape_rows = [
            row for row in rows if row["shape_id"] == shape.shape_id
        ]
        assert [row["pair_index"] for row in shape_rows[:2]] == [-2, -1]
        assert [
            row["pair_index"] for row in shape_rows[2:]
        ] == list(range(200))
        assert sum(
            row["arm_order"][0] == "bf16"
            for row in shape_rows
        ) == 101
        assert sum(
            row["arm_order"][0] == "fused_int4"
            for row in shape_rows
        ) == 101


def test_schedule_rejects_non_frozen_measured_pair_count():
    with pytest.raises(ValueError, match="200"):
        build_pair_schedule(
            shapes=_shapes(),
            warmup_pairs=2,
            measured_pairs=199,
        )


def test_worker_rejects_output_outside_approved_remote_root(tmp_path):
    args = _valid_worker_args(
        tmp_path,
        output_dir=tmp_path / "outside-approved-root",
    )

    with pytest.raises(ValueError, match="approved remote root"):
        validate_worker_arguments(args)


def test_worker_rejects_unfrozen_execution_contract(tmp_path):
    args = _valid_worker_args(tmp_path, measured_pairs=199)

    with pytest.raises(ValueError, match="measured_pairs"):
        validate_worker_arguments(args)


def test_worker_writes_cleanup_on_candidate_failure(
    tmp_path,
    monkeypatch,
):
    args = _valid_worker_args(tmp_path)
    monkeypatch.setattr(
        worker,
        "run_measured_candidate",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("candidate failed")
        ),
    )

    with pytest.raises(RuntimeError, match="candidate failed"):
        run_worker(args)

    cleanup = json.loads(
        (args.output_dir / "cleanup.json").read_text(encoding="utf-8")
    )
    assert cleanup["classification"] == "DIRTY"
    assert cleanup["error_type"] == "RuntimeError"
    assert cleanup["error_message"] == "candidate failed"


def test_worker_script_help_runs_outside_repository(tmp_path):
    completed = subprocess.run(
        [sys.executable, str(WORKER_PATH), "--help"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--approved-remote-root" in completed.stdout


def test_worker_loads_candidate_modules_from_staged_source_snapshot():
    source = WORKER_PATH.read_text(encoding="utf-8")

    assert "from tinyvllm.layers.fused_int4_linear import" not in source
    assert "from tinyvllm.layers.quantization import" not in source
    assert "_load_staged_module(" in source
