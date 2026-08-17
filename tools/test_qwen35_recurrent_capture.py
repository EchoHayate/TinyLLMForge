import gc
import importlib
import json
from pathlib import Path
import sys
import types
import weakref

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
tinyvllm_package = types.ModuleType("tinyvllm")
tinyvllm_package.__path__ = [str(ROOT / "tinyvllm")]
engine_package = types.ModuleType("tinyvllm.engine")
engine_package.__path__ = [str(ROOT / "tinyvllm/engine")]
sys.modules.setdefault("tinyvllm", tinyvllm_package)
sys.modules.setdefault("tinyvllm.engine", engine_package)

capture = importlib.import_module(
    "tinyvllm.engine.qwen35_recurrent_capture"
)
from tinyvllm.engine.qwen35_recurrent_capture_contract import (
    CAPTURE_IDENTITY_SCHEMA_VERSION,
    canonical_json_bytes,
    validate_run_identity,
)


def capture_identity(
    *,
    world_size=1,
    workloads=("w0",),
    layers=(0,),
    source_tree_sha256="b" * 64,
):
    return validate_run_identity({
        "schema_version": CAPTURE_IDENTITY_SCHEMA_VERSION,
        "model_manifest_sha256": "a" * 64,
        "source_tree_sha256": source_tree_sha256,
        "workload_manifest_sha256": "c" * 64,
        "world_size": world_size,
        "workload_ids": list(workloads),
        "linear_layer_indices": list(layers),
    })


def tensor_for(layer_index=0):
    return torch.arange(
        24,
        dtype=torch.float16,
    ).reshape(2, 3, 4).transpose(1, 2) + layer_index


def expected_tensor_path(root, workload_id="w0", layer_index=0):
    return (
        root
        / "rank0"
        / "tensors"
        / workload_id
        / f"layer{layer_index}.pt"
    )


def test_capture_persists_fp32_contiguous_cpu_tensor_and_metadata(
    tmp_path,
):
    identity = capture_identity(
        world_size=1,
        workloads=("w0",),
        layers=(0,),
    )
    session = capture.Qwen35RecurrentCaptureSession(
        run_identity=identity,
        rank=0,
        staging_dir=tmp_path,
    )
    source = tensor_for()
    record = session.capture_layer(
        workload_id="w0",
        layer_index=0,
        tensor=source,
    )
    persisted = torch.load(
        tmp_path / record.relative_path,
        weights_only=True,
    )
    assert persisted.dtype == torch.float32
    assert persisted.device.type == "cpu"
    assert persisted.is_contiguous()
    assert list(persisted.shape) == list(record.shape)
    assert record.logical_bytes == persisted.numel() * 4
    assert record.relative_path == "rank0/tensors/w0/layer0.pt"
    assert session.records == {record.tensor_id: record}


def test_capture_uses_required_conversion_order(tmp_path):
    calls = []

    class OrderedTensor:
        def __init__(self):
            self.value = tensor_for()

        def detach(self):
            calls.append("detach")
            return self

        def to(self, *, dtype=None, device=None):
            if dtype is not None:
                calls.append(("dtype", dtype))
                self.value = self.value.to(dtype=dtype)
                return self
            calls.append(("device", device))
            return self.value.to(device=device)

        def contiguous(self):
            calls.append("contiguous")
            self.value = self.value.contiguous()
            return self

    session = capture.Qwen35RecurrentCaptureSession(
        run_identity=capture_identity(),
        rank=0,
        staging_dir=tmp_path,
    )
    session.capture_layer(
        workload_id="w0",
        layer_index=0,
        tensor=OrderedTensor(),
    )
    assert calls == [
        "detach",
        ("dtype", torch.float32),
        "contiguous",
        ("device", "cpu"),
    ]


@pytest.mark.parametrize(
    "tensor",
    (
        torch.ones(2, 3, dtype=torch.float16),
        torch.ones(2, 0, 3, dtype=torch.float16),
    ),
)
def test_invalid_tensor_shape_is_rejected_before_serializer(
    tmp_path,
    monkeypatch,
    tensor,
):
    session = capture.Qwen35RecurrentCaptureSession(
        run_identity=capture_identity(),
        rank=0,
        staging_dir=tmp_path,
    )
    calls = []

    def unexpected_save(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("serializer must not be invoked")

    monkeypatch.setattr(capture, "save_tensor", unexpected_save)
    with pytest.raises(ValueError, match="shape"):
        session.capture_layer(
            workload_id="w0",
            layer_index=0,
            tensor=tensor,
        )
    assert calls == []
    assert not expected_tensor_path(tmp_path).exists()
    assert session.records == {}


@pytest.mark.parametrize(
    "workload_id",
    (
        "/tmp/escape",
        "../escape",
        "nested/escape",
        r"nested\escape",
        "./w0",
        "w0/",
        "w0/.",
        "w0\0alias",
    ),
)
def test_workload_id_cannot_escape_rank_tensor_root(
    tmp_path,
    workload_id,
):
    with pytest.raises(ValueError, match="workload_id"):
        capture.Qwen35RecurrentCaptureSession(
            run_identity=capture_identity(workloads=(workload_id,)),
            rank=0,
            staging_dir=tmp_path,
        )
    assert not (tmp_path / "rank0").exists()


def test_duplicate_tensor_id_is_rejected_before_serializer(
    tmp_path,
    monkeypatch,
):
    session = capture.Qwen35RecurrentCaptureSession(
        run_identity=capture_identity(),
        rank=0,
        staging_dir=tmp_path,
    )
    session.capture_layer(
        workload_id="w0",
        layer_index=0,
        tensor=tensor_for(),
    )
    calls = []

    def unexpected_save(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("serializer must not be invoked")

    monkeypatch.setattr(capture, "save_tensor", unexpected_save)
    with pytest.raises(ValueError, match="duplicate tensor"):
        session.capture_layer(
            workload_id="w0",
            layer_index=0,
            tensor=tensor_for(),
        )
    assert calls == []


def test_identical_identity_allows_next_untouched_workload(
    tmp_path,
):
    identity = capture_identity(
        workloads=("w0", "w1"),
        layers=(0,),
    )
    first = capture.Qwen35RecurrentCaptureSession(
        run_identity=identity,
        rank=0,
        staging_dir=tmp_path,
    )
    first_record = first.capture_layer(
        workload_id="w0",
        layer_index=0,
        tensor=tensor_for(),
    )
    first.finish_workload("w0")

    second = capture.Qwen35RecurrentCaptureSession(
        run_identity=identity,
        rank=0,
        staging_dir=tmp_path,
    )
    assert second.records == {first_record.tensor_id: first_record}
    second_record = second.capture_layer(
        workload_id="w1",
        layer_index=0,
        tensor=tensor_for(),
    )
    second.finish_workload("w1")
    assert set(second.records) == {
        first_record.tensor_id,
        second_record.tensor_id,
    }


def test_mismatched_or_nonidentical_identity_bytes_are_rejected(
    tmp_path,
):
    identity = capture_identity()
    capture.Qwen35RecurrentCaptureSession(
        run_identity=identity,
        rank=0,
        staging_dir=tmp_path,
    )
    with pytest.raises(ValueError, match="identity"):
        capture.Qwen35RecurrentCaptureSession(
            run_identity=capture_identity(
                source_tree_sha256="d" * 64,
            ),
            rank=0,
            staging_dir=tmp_path,
        )

    identity_path = tmp_path / "rank0" / "capture_identity.json"
    identity_path.write_text(
        json.dumps(identity.payload(), indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="identity"):
        capture.Qwen35RecurrentCaptureSession(
            run_identity=identity,
            rank=0,
            staging_dir=tmp_path,
        )


def test_completed_workload_cannot_be_reopened(tmp_path):
    identity = capture_identity()
    first = capture.Qwen35RecurrentCaptureSession(
        run_identity=identity,
        rank=0,
        staging_dir=tmp_path,
    )
    first.capture_layer(
        workload_id="w0",
        layer_index=0,
        tensor=tensor_for(),
    )
    first.finish_workload("w0")

    second = capture.Qwen35RecurrentCaptureSession(
        run_identity=identity,
        rank=0,
        staging_dir=tmp_path,
    )
    with pytest.raises(ValueError, match="completed workload"):
        second.capture_layer(
            workload_id="w0",
            layer_index=0,
            tensor=tensor_for(),
        )


def test_partial_workload_or_leftover_temporary_file_fails_closed(
    tmp_path,
):
    identity = capture_identity(workloads=("w0", "w1"))
    first = capture.Qwen35RecurrentCaptureSession(
        run_identity=identity,
        rank=0,
        staging_dir=tmp_path,
    )
    first.capture_layer(
        workload_id="w0",
        layer_index=0,
        tensor=tensor_for(),
    )
    with pytest.raises(ValueError, match="partial workload"):
        capture.Qwen35RecurrentCaptureSession(
            run_identity=identity,
            rank=0,
            staging_dir=tmp_path,
        )

    clean_root = tmp_path / "other"
    capture.Qwen35RecurrentCaptureSession(
        run_identity=identity,
        rank=0,
        staging_dir=clean_root,
    )
    leftover = capture._new_temporary_path(
        clean_root / "rank0" / "tensors" / "leftover.pt"
    )
    leftover.parent.mkdir(parents=True, exist_ok=True)
    leftover.write_bytes(b"incomplete")
    with pytest.raises(ValueError, match="temporary"):
        capture.Qwen35RecurrentCaptureSession(
            run_identity=identity,
            rank=0,
            staging_dir=clean_root,
        )


@pytest.mark.parametrize(
    "failure_point",
    ("serialization", "fsync", "rename"),
)
def test_failed_publication_leaves_no_final_tensor_or_record(
    tmp_path,
    monkeypatch,
    failure_point,
):
    session = capture.Qwen35RecurrentCaptureSession(
        run_identity=capture_identity(),
        rank=0,
        staging_dir=tmp_path,
    )

    if failure_point == "serialization":
        def fail_save(*args, **kwargs):
            raise OSError("serialization failed")

        monkeypatch.setattr(capture, "save_tensor", fail_save)
    elif failure_point == "fsync":
        def fail_fsync(path):
            raise OSError("fsync failed")

        monkeypatch.setattr(capture, "_fsync_regular_file", fail_fsync)
    else:
        original_publish = capture._publish_no_clobber

        def fail_tensor_publish(path, target):
            if path.name.startswith(".layer0.pt.tmp-"):
                raise OSError("rename failed")
            return original_publish(path, target)

        monkeypatch.setattr(
            capture,
            "_publish_no_clobber",
            fail_tensor_publish,
        )

    with pytest.raises(OSError, match="failed"):
        session.capture_layer(
            workload_id="w0",
            layer_index=0,
            tensor=tensor_for(),
        )
    assert not expected_tensor_path(tmp_path).exists()
    assert session.records == {}
    assert not tuple((tmp_path / "rank0").rglob("*.tmp-*"))


def test_symlinked_rank_root_and_preexisting_final_path_are_rejected(
    tmp_path,
):
    target = tmp_path / "actual-rank"
    target.mkdir()
    (tmp_path / "rank0").symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        capture.Qwen35RecurrentCaptureSession(
            run_identity=capture_identity(),
            rank=0,
            staging_dir=tmp_path,
        )

    clean_root = tmp_path / "clean"
    session = capture.Qwen35RecurrentCaptureSession(
        run_identity=capture_identity(),
        rank=0,
        staging_dir=clean_root,
    )
    final_path = expected_tensor_path(clean_root)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    final_path.write_bytes(b"pre-existing")
    with pytest.raises(ValueError, match="final path"):
        session.capture_layer(
            workload_id="w0",
            layer_index=0,
            tensor=tensor_for(),
        )


def test_symlinked_capture_root_and_empty_path_are_rejected(tmp_path):
    external = tmp_path / "external"
    external.mkdir()
    capture_root = tmp_path / "capture"
    capture_root.symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="capture root"):
        capture.Qwen35RecurrentCaptureSession(
            run_identity=capture_identity(),
            rank=0,
            staging_dir=capture_root,
        )
    assert tuple(external.iterdir()) == ()

    with pytest.raises(ValueError, match="capture root"):
        capture.Qwen35RecurrentCaptureSession(
            run_identity=capture_identity(),
            rank=0,
            staging_dir="",
        )

    with pytest.raises(ValueError, match="capture root"):
        capture.Qwen35RecurrentCaptureSession(
            run_identity=capture_identity(),
            rank=0,
            staging_dir="capture\0root",
        )

    traversal_root = tmp_path / "safe" / ".." / ".." / "outside"
    with pytest.raises(ValueError, match="capture root"):
        capture.Qwen35RecurrentCaptureSession(
            run_identity=capture_identity(),
            rank=0,
            staging_dir=traversal_root,
        )
    assert not (tmp_path.parent / "outside").exists()


def test_symlinked_tensor_root_is_rejected(tmp_path):
    rank_root = tmp_path / "rank0"
    rank_root.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    (rank_root / "tensors").symlink_to(
        external,
        target_is_directory=True,
    )
    with pytest.raises(ValueError, match="symlink"):
        capture.Qwen35RecurrentCaptureSession(
            run_identity=capture_identity(),
            rank=0,
            staging_dir=tmp_path,
        )
    assert tuple(external.iterdir()) == ()


def test_completed_workload_directory_symlink_is_rejected(tmp_path):
    identity = capture_identity()
    first = capture.Qwen35RecurrentCaptureSession(
        run_identity=identity,
        rank=0,
        staging_dir=tmp_path,
    )
    record = first.capture_layer(
        workload_id="w0",
        layer_index=0,
        tensor=tensor_for(),
    )
    first.finish_workload("w0")

    workload_root = tmp_path / "rank0" / "tensors" / "w0"
    external = tmp_path / "external-workload"
    workload_root.rename(external)
    workload_root.symlink_to(external, target_is_directory=True)
    assert (tmp_path / record.relative_path).is_file()

    with pytest.raises(ValueError, match="symlink"):
        capture.Qwen35RecurrentCaptureSession(
            run_identity=identity,
            rank=0,
            staging_dir=tmp_path,
        )


def test_tensor_publication_does_not_clobber_racing_final_path(
    tmp_path,
    monkeypatch,
):
    session = capture.Qwen35RecurrentCaptureSession(
        run_identity=capture_identity(),
        rank=0,
        staging_dir=tmp_path,
    )
    final_path = expected_tensor_path(tmp_path)
    original_fsync = capture._fsync_regular_file

    def create_racing_final(path):
        original_fsync(path)
        final_path.write_bytes(b"racing-writer")

    monkeypatch.setattr(
        capture,
        "_fsync_regular_file",
        create_racing_final,
    )
    with pytest.raises((FileExistsError, ValueError)):
        session.capture_layer(
            workload_id="w0",
            layer_index=0,
            tensor=tensor_for(),
        )
    assert final_path.read_bytes() == b"racing-writer"
    assert session.records == {}


def test_receipt_filename_is_bound_to_workload_id(tmp_path):
    identity = capture_identity()
    first = capture.Qwen35RecurrentCaptureSession(
        run_identity=identity,
        rank=0,
        staging_dir=tmp_path,
    )
    first.capture_layer(
        workload_id="w0",
        layer_index=0,
        tensor=tensor_for(),
    )
    first.finish_workload("w0")
    receipt_root = tmp_path / "rank0" / "workloads"
    (receipt_root / "w0.complete.json").rename(
        receipt_root / "alias.complete.json"
    )

    with pytest.raises(ValueError, match="receipt"):
        capture.Qwen35RecurrentCaptureSession(
            run_identity=identity,
            rank=0,
            staging_dir=tmp_path,
        )


def test_session_retains_only_metadata_records(
    tmp_path,
    monkeypatch,
):
    converted_references = []
    original_save = capture.save_tensor

    def observe_save(tensor, path):
        converted_references.append(weakref.ref(tensor))
        original_save(tensor, path)

    monkeypatch.setattr(capture, "save_tensor", observe_save)
    session = capture.Qwen35RecurrentCaptureSession(
        run_identity=capture_identity(),
        rank=0,
        staging_dir=tmp_path,
    )
    source = tensor_for()
    source_reference = weakref.ref(source)
    record = session.capture_layer(
        workload_id="w0",
        layer_index=0,
        tensor=source,
    )
    del source
    gc.collect()

    assert source_reference() is None
    assert len(converted_references) == 1
    assert converted_references[0]() is None
    assert session.records == {record.tensor_id: record}
    assert all(
        not isinstance(value, torch.Tensor)
        for value in session.__dict__.values()
    )


def test_identity_is_atomic_and_idempotent_only_for_exact_bytes(
    tmp_path,
    monkeypatch,
):
    identity = capture_identity()
    first = capture.Qwen35RecurrentCaptureSession(
        run_identity=identity,
        rank=0,
        staging_dir=tmp_path,
    )
    expected = canonical_json_bytes(identity.payload()) + b"\n"
    assert first.identity_path.read_bytes() == expected
    assert not tuple(first.rank_root.glob(".capture_identity.json.tmp-*"))

    replacements = []
    original_replace = Path.replace

    def observe_replace(path, target):
        replacements.append((path, target))
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", observe_replace)
    capture.Qwen35RecurrentCaptureSession(
        run_identity=identity,
        rank=0,
        staging_dir=tmp_path,
    )
    assert replacements == []


def test_workload_receipt_is_published_only_after_all_layers_exist(
    tmp_path,
):
    identity = capture_identity(layers=(0, 2))
    session = capture.Qwen35RecurrentCaptureSession(
        run_identity=identity,
        rank=0,
        staging_dir=tmp_path,
    )
    first = session.capture_layer(
        workload_id="w0",
        layer_index=0,
        tensor=tensor_for(),
    )
    receipt_path = tmp_path / "rank0" / "workloads" / "w0.complete.json"
    with pytest.raises(ValueError, match="missing layers"):
        session.finish_workload("w0")
    assert not receipt_path.exists()

    second = session.capture_layer(
        workload_id="w0",
        layer_index=2,
        tensor=tensor_for(2),
    )
    receipt = session.finish_workload("w0")
    assert receipt == {
        "rank": 0,
        "workload_id": "w0",
        "tensors": [first.payload(), second.payload()],
    }
    assert receipt_path.read_bytes() == (
        canonical_json_bytes(receipt) + b"\n"
    )


@pytest.mark.parametrize("tamper", ("remove", "replace"))
def test_workload_receipt_revalidates_tensor_evidence(
    tmp_path,
    tamper,
):
    session = capture.Qwen35RecurrentCaptureSession(
        run_identity=capture_identity(),
        rank=0,
        staging_dir=tmp_path,
    )
    record = session.capture_layer(
        workload_id="w0",
        layer_index=0,
        tensor=tensor_for(),
    )
    tensor_path = tmp_path / record.relative_path
    if tamper == "remove":
        tensor_path.unlink()
    else:
        tensor_path.write_bytes(b"replaced")

    receipt_path = tmp_path / "rank0" / "workloads" / "w0.complete.json"
    with pytest.raises(ValueError, match="evidence"):
        session.finish_workload("w0")
    assert not receipt_path.exists()
    assert "w0" not in session._completed_workloads


def test_completed_workload_name_with_tmp_text_can_be_reopened(tmp_path):
    identity = capture_identity(workloads=("job.tmp", "job.tmp-suffix"))
    first = capture.Qwen35RecurrentCaptureSession(
        run_identity=identity,
        rank=0,
        staging_dir=tmp_path,
    )
    first.capture_layer(
        workload_id="job.tmp",
        layer_index=0,
        tensor=tensor_for(),
    )
    first.finish_workload("job.tmp")
    second = capture.Qwen35RecurrentCaptureSession(
        run_identity=identity,
        rank=0,
        staging_dir=tmp_path,
    )
    second.capture_layer(
        workload_id="job.tmp-suffix",
        layer_index=0,
        tensor=tensor_for(),
    )
    second.finish_workload("job.tmp-suffix")

    reopened = capture.Qwen35RecurrentCaptureSession(
        run_identity=identity,
        rank=0,
        staging_dir=tmp_path,
    )
    assert reopened._completed_workloads == {"job.tmp", "job.tmp-suffix"}


def test_completed_workload_name_matching_temporary_pattern_can_be_reopened(
    tmp_path,
):
    workload_id = f".layer0.pt.tmp-{'a' * 32}"
    identity = capture_identity(workloads=(workload_id,))
    first = capture.Qwen35RecurrentCaptureSession(
        run_identity=identity,
        rank=0,
        staging_dir=tmp_path,
    )
    first.capture_layer(
        workload_id=workload_id,
        layer_index=0,
        tensor=tensor_for(),
    )
    first.finish_workload(workload_id)

    reopened = capture.Qwen35RecurrentCaptureSession(
        run_identity=identity,
        rank=0,
        staging_dir=tmp_path,
    )
    assert reopened._completed_workloads == {workload_id}


def test_generated_temporary_directory_fails_as_temporary(tmp_path):
    identity = capture_identity()
    session = capture.Qwen35RecurrentCaptureSession(
        run_identity=identity,
        rank=0,
        staging_dir=tmp_path,
    )
    temporary_directory = (
        session.tensor_root
        / f".layer0.pt.tmp-{'a' * 32}"
    )
    temporary_directory.mkdir()

    with pytest.raises(ValueError, match="temporary"):
        capture.Qwen35RecurrentCaptureSession(
            run_identity=identity,
            rank=0,
            staging_dir=tmp_path,
        )


@pytest.mark.parametrize(
    "name",
    (".user.tmp", ".notes.tmp-backup", ".layer0.pt.tmp-not-a-uuid"),
)
def test_unrelated_hidden_path_is_not_classified_as_generated_temporary(
    tmp_path,
    name,
):
    identity = capture_identity()
    session = capture.Qwen35RecurrentCaptureSession(
        run_identity=identity,
        rank=0,
        staging_dir=tmp_path,
    )
    (session.tensor_root / name).write_bytes(b"untracked")

    with pytest.raises(ValueError, match="untracked"):
        capture.Qwen35RecurrentCaptureSession(
            run_identity=identity,
            rank=0,
            staging_dir=tmp_path,
        )


def test_capture_recurrent_state_uses_the_session_contract(tmp_path):
    record = capture.capture_recurrent_state(
        run_identity=capture_identity(),
        rank=0,
        workload_id="w0",
        layer_index=0,
        tensor=tensor_for(),
        staging_dir=tmp_path,
    )
    assert record.tensor_id == "rank0:w0:layer0:linear_recurrent"
    assert expected_tensor_path(tmp_path).is_file()
