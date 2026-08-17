import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import types

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT / "tools/qwen35_recurrent_full_fidelity_capture_closer.py"
)

tinyvllm_package = types.ModuleType("tinyvllm")
tinyvllm_package.__path__ = [str(ROOT / "tinyvllm")]
engine_package = types.ModuleType("tinyvllm.engine")
engine_package.__path__ = [str(ROOT / "tinyvllm/engine")]
sys.modules.setdefault("tinyvllm", tinyvllm_package)
sys.modules.setdefault("tinyvllm.engine", engine_package)

from tinyvllm.engine.qwen35_recurrent_capture import (
    Qwen35RecurrentCaptureSession,
)
from tinyvllm.engine.qwen35_recurrent_capture_contract import (
    CAPTURE_IDENTITY_SCHEMA_VERSION,
    RANK_CAPTURE_MANIFEST_SCHEMA_VERSION,
    canonical_json_bytes,
    validate_run_identity,
)


def _load_closer():
    spec = importlib.util.spec_from_file_location(
        "qwen35_recurrent_full_fidelity_capture_closer_under_test",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


closer = _load_closer()


def capture_identity(
    *,
    world_size=1,
    workloads=("w0", "w1"),
    layers=(0, 2),
):
    return validate_run_identity({
        "schema_version": CAPTURE_IDENTITY_SCHEMA_VERSION,
        "model_manifest_sha256": "a" * 64,
        "source_tree_sha256": "b" * 64,
        "workload_manifest_sha256": "c" * 64,
        "world_size": world_size,
        "workload_ids": list(workloads),
        "linear_layer_indices": list(layers),
    })


def tensor_for(workload_index, layer_index):
    return (
        torch.arange(24, dtype=torch.float32)
        .reshape(2, 3, 4)
        .add(workload_index * 100 + layer_index)
    )


def build_rank_fixture(
    root,
    *,
    workloads=("w0", "w1"),
    layers=(0, 2),
):
    identity = capture_identity(workloads=workloads, layers=layers)
    records = {}
    for workload_index, workload_id in enumerate(workloads):
        session = Qwen35RecurrentCaptureSession(
            run_identity=identity,
            rank=0,
            staging_dir=root,
        )
        for layer_index in layers:
            record = session.capture_layer(
                workload_id=workload_id,
                layer_index=layer_index,
                tensor=tensor_for(workload_index, layer_index),
            )
            records[(workload_id, layer_index)] = record
        session.finish_workload(workload_id)
    return identity, root / "rank0", records


def receipt_path(rank_root, workload_id):
    return rank_root / "workloads" / f"{workload_id}.complete.json"


def read_receipt(rank_root, workload_id):
    return json.loads(receipt_path(rank_root, workload_id).read_text("utf-8"))


def write_receipt(rank_root, workload_id, receipt):
    receipt_path(rank_root, workload_id).write_bytes(
        canonical_json_bytes(receipt) + b"\n"
    )


def rewrite_tensor_and_receipt(
    rank_root,
    *,
    workload_id="w0",
    layer_index=0,
    tensor,
):
    receipt = read_receipt(rank_root, workload_id)
    row = next(
        row
        for row in receipt["tensors"]
        if row["layer_index"] == layer_index
    )
    tensor_path = rank_root.parent / row["relative_path"]
    torch.save(tensor, tensor_path)
    payload = tensor_path.read_bytes()
    row["sha256"] = hashlib.sha256(payload).hexdigest()
    write_receipt(rank_root, workload_id, receipt)


def close_fixture(rank_root, *, load_tensor=torch.load):
    return closer.close_rank_capture(
        staging_dir=rank_root,
        expected_workload_ids=("w0", "w1"),
        expected_linear_layer_indices=(0, 2),
        load_tensor=load_tensor,
    )


def test_close_rank_capture_publishes_canonical_manifest(tmp_path):
    identity, rank_root, _ = build_rank_fixture(tmp_path)

    manifest = close_fixture(rank_root)

    assert manifest.identity == identity
    assert manifest.rank == 0
    assert tuple(row.tensor_id for row in manifest.tensors) == (
        "rank0:w0:layer0:linear_recurrent",
        "rank0:w0:layer2:linear_recurrent",
        "rank0:w1:layer0:linear_recurrent",
        "rank0:w1:layer2:linear_recurrent",
    )
    manifest_path = rank_root / "rank_capture_manifest.json"
    assert manifest_path.is_file()
    assert json.loads(manifest_path.read_text("utf-8")) == {
        "schema_version": RANK_CAPTURE_MANIFEST_SCHEMA_VERSION,
        "identity": identity.payload(),
        "rank": 0,
        "tensors": [row.payload() for row in manifest.tensors],
    }
    assert not tuple(rank_root.rglob("*.tmp-*"))


def test_closer_hashes_and_loads_the_same_byte_snapshot(tmp_path):
    _, rank_root, _ = build_rank_fixture(tmp_path)
    observed_payloads = []

    def observe_load(source, *, map_location):
        assert isinstance(source, io.BytesIO)
        assert map_location == "cpu"
        payload = source.getvalue()
        observed_payloads.append(payload)
        return torch.load(
            io.BytesIO(payload),
            map_location="cpu",
            weights_only=True,
        )

    manifest = close_fixture(rank_root, load_tensor=observe_load)

    assert len(observed_payloads) == 4
    assert tuple(
        hashlib.sha256(payload).hexdigest()
        for payload in observed_payloads
    ) == tuple(row.sha256 for row in manifest.tensors)


@pytest.mark.parametrize("mode", ("missing", "extra"))
def test_missing_or_extra_workload_receipt_is_rejected(tmp_path, mode):
    _, rank_root, _ = build_rank_fixture(tmp_path)
    if mode == "missing":
        receipt_path(rank_root, "w1").unlink()
    else:
        (rank_root / "workloads" / "extra.complete.json").write_text(
            "{}\n",
            encoding="utf-8",
        )

    with pytest.raises(ValueError, match="receipt"):
        close_fixture(rank_root)
    assert not (rank_root / "rank_capture_manifest.json").exists()


@pytest.mark.parametrize(
    "mode",
    ("missing", "duplicate", "extra_tensor", "untracked"),
)
def test_invalid_tensor_inventory_is_rejected(tmp_path, mode):
    _, rank_root, records = build_rank_fixture(tmp_path)
    if mode == "missing":
        (rank_root.parent / records[("w0", 0)].relative_path).unlink()
    elif mode == "duplicate":
        receipt = read_receipt(rank_root, "w0")
        receipt["tensors"][1] = dict(receipt["tensors"][0])
        write_receipt(rank_root, "w0", receipt)
    elif mode == "extra_tensor":
        extra = rank_root / "tensors" / "w0" / "layer99.pt"
        torch.save(torch.ones(2, 3, 4), extra)
    else:
        (rank_root / "notes.txt").write_text("untracked", encoding="utf-8")

    with pytest.raises(ValueError):
        close_fixture(rank_root)
    assert not (rank_root / "rank_capture_manifest.json").exists()


def test_tampered_tensor_bytes_are_rejected(tmp_path):
    _, rank_root, records = build_rank_fixture(tmp_path)
    tensor_path = rank_root.parent / records[("w0", 0)].relative_path
    tensor_path.write_bytes(tensor_path.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="hash"):
        close_fixture(rank_root)


@pytest.mark.parametrize(
    ("tensor", "message"),
    (
        (torch.ones(2, 3, 4, dtype=torch.float16), "dtype"),
        (torch.ones(2, 3, dtype=torch.float32), "rank-3"),
    ),
)
def test_persisted_tensor_dtype_and_rank_are_authoritative(
    tmp_path,
    tensor,
    message,
):
    _, rank_root, _ = build_rank_fixture(tmp_path)
    rewrite_tensor_and_receipt(rank_root, tensor=tensor)

    with pytest.raises(ValueError, match=message):
        close_fixture(rank_root)


@pytest.mark.parametrize(
    "location",
    ("identity", "receipt_root", "workload_root", "tensor"),
)
def test_symlink_anywhere_below_rank_root_is_rejected(
    tmp_path,
    location,
):
    _, rank_root, records = build_rank_fixture(tmp_path)
    external = tmp_path / f"external-{location}"
    if location == "identity":
        original = rank_root / "capture_identity.json"
    elif location == "receipt_root":
        original = rank_root / "workloads"
    elif location == "workload_root":
        original = rank_root / "tensors" / "w0"
    else:
        original = rank_root.parent / records[("w0", 0)].relative_path
    original.rename(external)
    original.symlink_to(
        external,
        target_is_directory=external.is_dir(),
    )

    with pytest.raises(ValueError, match="symlink"):
        close_fixture(rank_root)


@pytest.mark.parametrize(
    ("workloads", "layers"),
    (
        (("w0",), (0, 2)),
        (("w0", "w1"), (0,)),
        (("w0", "w1", "w2"), (0, 2)),
        (("w0", "w1"), (0, 1, 2)),
    ),
)
def test_expected_inventory_must_match_capture_identity(
    tmp_path,
    workloads,
    layers,
):
    _, rank_root, _ = build_rank_fixture(tmp_path)

    with pytest.raises(ValueError, match="identity"):
        closer.close_rank_capture(
            staging_dir=rank_root,
            expected_workload_ids=workloads,
            expected_linear_layer_indices=layers,
        )


@pytest.mark.parametrize(
    ("workloads", "layers"),
    (
        ((False, "w1"), (0, 2)),
        (("w0", "w1"), (False, 2)),
    ),
)
def test_expected_inventory_rejects_bool_aliases(
    tmp_path,
    workloads,
    layers,
):
    _, rank_root, _ = build_rank_fixture(tmp_path)

    with pytest.raises(ValueError, match="expected"):
        closer.close_rank_capture(
            staging_dir=rank_root,
            expected_workload_ids=workloads,
            expected_linear_layer_indices=layers,
        )


def test_existing_rank_manifest_is_rejected(tmp_path):
    _, rank_root, _ = build_rank_fixture(tmp_path)
    manifest_path = rank_root / "rank_capture_manifest.json"
    manifest_path.write_text("pre-existing\n", encoding="utf-8")

    with pytest.raises(ValueError, match="manifest"):
        close_fixture(rank_root)
    assert manifest_path.read_text("utf-8") == "pre-existing\n"


def test_generated_temporary_path_is_not_evidence(tmp_path):
    _, rank_root, _ = build_rank_fixture(tmp_path)
    temporary = rank_root / f".rank_capture_manifest.json.tmp-{'a' * 32}"
    temporary.mkdir()

    with pytest.raises(ValueError, match="temporary"):
        close_fixture(rank_root)
    assert not (rank_root / "rank_capture_manifest.json").exists()


def test_cli_help_lists_required_arguments():
    result = subprocess.run(
        [
            sys.executable,
            str(MODULE_PATH),
            "--help",
        ],
        cwd=ROOT,
        env={
            **dict(),
            "PYTHONPATH": (
                "/tmp/tinyllmforge-pytest312-shim:"
                f"{ROOT}"
            ),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--capture-root" in result.stdout
    assert "--rank" in result.stdout
    assert "--expected-workload-id" in result.stdout
    assert "--expected-linear-layer-index" in result.stdout
