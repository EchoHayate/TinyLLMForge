import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import types
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
ASSEMBLER_PATH = (
    ROOT / "tools/qwen35_recurrent_full_fidelity_bundle_assembler.py"
)

tinyvllm_package = types.ModuleType("tinyvllm")
tinyvllm_package.__path__ = [str(ROOT / "tinyvllm")]
engine_package = types.ModuleType("tinyvllm.engine")
engine_package.__path__ = [str(ROOT / "tinyvllm/engine")]
sys.modules.setdefault("tinyvllm", tinyvllm_package)
sys.modules.setdefault("tinyvllm.engine", engine_package)


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


calibration_contract = _load_module(
    "qwen35_recurrent_int8_calibration_contract",
    ROOT / "tools/qwen35_recurrent_int8_calibration_contract.py",
)
closer = _load_module(
    "qwen35_recurrent_full_fidelity_capture_closer_for_assembler_tests",
    ROOT / "tools/qwen35_recurrent_full_fidelity_capture_closer.py",
)
calibration = _load_module(
    "qwen35_recurrent_int8_calibration_for_assembler_tests",
    ROOT / "tools/qwen35_recurrent_int8_calibration.py",
)
verifier = _load_module(
    "verify_qwen35_recurrent_int8_calibration_for_assembler_tests",
    ROOT / "tools/verify_qwen35_recurrent_int8_calibration.py",
)
assembler = _load_module(
    "qwen35_recurrent_full_fidelity_bundle_assembler_under_test",
    ASSEMBLER_PATH,
)

from tinyvllm.engine.qwen35_recurrent_capture import (
    Qwen35RecurrentCaptureSession,
)
from tinyvllm.engine.qwen35_recurrent_capture_contract import (
    CAPTURE_IDENTITY_SCHEMA_VERSION,
    canonical_json_bytes,
    validate_run_identity,
)


WORKLOADS = ("w0", "w1")
LAYERS = (0, 2)
MODEL_SHA256 = "a" * 64
SOURCE_SHA256 = "b" * 64
WORKLOAD_SHA256 = "c" * 64


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity(*, source_tree_sha256=SOURCE_SHA256):
    return validate_run_identity({
        "schema_version": CAPTURE_IDENTITY_SCHEMA_VERSION,
        "model_manifest_sha256": MODEL_SHA256,
        "source_tree_sha256": source_tree_sha256,
        "workload_manifest_sha256": WORKLOAD_SHA256,
        "world_size": 2,
        "workload_ids": list(WORKLOADS),
        "linear_layer_indices": list(LAYERS),
    })


def _tensor(rank, workload_index, layer_index):
    return (
        torch.arange(48, dtype=torch.float32)
        .reshape(2, 3, 8)
        .add(rank * 1000 + workload_index * 100 + layer_index)
    )


def _stage_rank(capture_root, rank, identity):
    for workload_index, workload_id in enumerate(WORKLOADS):
        session = Qwen35RecurrentCaptureSession(
            run_identity=identity,
            rank=rank,
            staging_dir=capture_root,
        )
        for layer_index in LAYERS:
            session.capture_layer(
                workload_id=workload_id,
                layer_index=layer_index,
                tensor=_tensor(rank, workload_index, layer_index),
            )
        session.finish_workload(workload_id)
    return capture_root / f"rank{rank}"


def _close_rank(capture_root, rank):
    return closer.close_rank_capture(
        staging_dir=capture_root / f"rank{rank}",
        expected_workload_ids=WORKLOADS,
        expected_linear_layer_indices=LAYERS,
    )


def _closed_capture(root, *, rank1_identity=None):
    capture_root = root / "capture"
    identity = _identity()
    _stage_rank(capture_root, 0, identity)
    _close_rank(capture_root, 0)
    _stage_rank(
        capture_root,
        1,
        identity if rank1_identity is None else rank1_identity,
    )
    _close_rank(capture_root, 1)
    return capture_root


def _assemble(capture_root, output_dir):
    return assembler.assemble_full_fidelity_bundle(
        capture_root=capture_root,
        output_dir=output_dir,
        model_manifest_sha256=MODEL_SHA256,
        source_tree_sha256=SOURCE_SHA256,
        workload_manifest_sha256=WORKLOAD_SHA256,
        world_size=2,
    )


def _manifest_path(capture_root, rank):
    return capture_root / f"rank{rank}" / "rank_capture_manifest.json"


def _read_manifest(capture_root, rank):
    return json.loads(_manifest_path(capture_root, rank).read_text("utf-8"))


def _write_manifest(capture_root, rank, manifest):
    _manifest_path(capture_root, rank).write_bytes(
        canonical_json_bytes(manifest) + b"\n"
    )


def test_assembler_publishes_existing_source_bundle_schema(tmp_path):
    capture_root = _closed_capture(tmp_path)
    output_dir = tmp_path / "bundle"

    result = _assemble(capture_root, output_dir)

    manifest = json.loads(
        (output_dir / "source_bundle_manifest.json").read_text("utf-8")
    )
    calibration_contract.validate_source_bundle_manifest(manifest)
    assert manifest["schema_version"] == (
        "qwen35.recurrent-full-fidelity-bundle.v1"
    )
    assert result["tensor_count"] == 8
    assert result["output_dir"] == str(output_dir)
    assert tuple(row["tensor_id"] for row in manifest["tensors"]) == tuple(
        f"rank{rank}:{workload}:layer{layer}:linear_recurrent"
        for rank in range(2)
        for workload in WORKLOADS
        for layer in LAYERS
    )
    for row in manifest["tensors"]:
        output_path = output_dir / row["relative_path"]
        capture_path = capture_root / (
            f"rank{row['rank']}/tensors/{row['workload_id']}/"
            f"layer{row['layer_index']}.pt"
        )
        assert output_path.read_bytes() == capture_path.read_bytes()
        assert row["sha256"] == _sha256(output_path)


def test_unclosed_rank_is_rejected(tmp_path):
    capture_root = tmp_path / "capture"
    identity = _identity()
    _stage_rank(capture_root, 0, identity)
    _close_rank(capture_root, 0)
    _stage_rank(capture_root, 1, identity)
    output_dir = tmp_path / "bundle"

    with pytest.raises(ValueError, match="closed|manifest"):
        _assemble(capture_root, output_dir)
    assert not output_dir.exists()


@pytest.mark.parametrize("mode", ("missing", "duplicate", "out_of_range"))
def test_rank_inventory_must_be_exact(tmp_path, mode):
    capture_root = _closed_capture(tmp_path)
    if mode == "missing":
        shutil.rmtree(capture_root / "rank1")
    elif mode == "duplicate":
        manifest = _read_manifest(capture_root, 1)
        manifest["rank"] = 0
        for row in manifest["tensors"]:
            row["rank"] = 0
            row["tensor_id"] = row["tensor_id"].replace("rank1:", "rank0:")
            row["relative_path"] = row["relative_path"].replace(
                "rank1/",
                "rank0/",
                1,
            )
        _write_manifest(capture_root, 1, manifest)
    else:
        shutil.copytree(capture_root / "rank1", capture_root / "rank2")

    output_dir = tmp_path / "bundle"
    with pytest.raises(ValueError, match="rank"):
        _assemble(capture_root, output_dir)
    assert not output_dir.exists()


def test_cross_rank_identity_mismatch_is_rejected(tmp_path):
    capture_root = _closed_capture(
        tmp_path,
        rank1_identity=_identity(source_tree_sha256="d" * 64),
    )
    output_dir = tmp_path / "bundle"

    with pytest.raises(ValueError, match="identity"):
        _assemble(capture_root, output_dir)
    assert not output_dir.exists()


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("model_manifest_sha256", "d" * 64),
        ("source_tree_sha256", "d" * 64),
        ("workload_manifest_sha256", "d" * 64),
    ),
)
def test_assembler_inputs_must_match_capture_identity(
    tmp_path,
    field,
    replacement,
):
    capture_root = _closed_capture(tmp_path)
    output_dir = tmp_path / "bundle"
    arguments = {
        "capture_root": capture_root,
        "output_dir": output_dir,
        "model_manifest_sha256": MODEL_SHA256,
        "source_tree_sha256": SOURCE_SHA256,
        "workload_manifest_sha256": WORKLOAD_SHA256,
        "world_size": 2,
    }
    arguments[field] = replacement

    with pytest.raises(ValueError, match="identity"):
        assembler.assemble_full_fidelity_bundle(**arguments)
    assert not output_dir.exists()


@pytest.mark.parametrize("mode", ("false_root", "absolute"))
def test_rank_payload_paths_cannot_rebind_the_capture_root(tmp_path, mode):
    capture_root = _closed_capture(tmp_path)
    manifest = _read_manifest(capture_root, 0)
    row = manifest["tensors"][0]
    if mode == "false_root":
        row["relative_path"] = "rank0/tensors/w1/layer0.pt"
    else:
        row["relative_path"] = str(
            capture_root / "rank0/tensors/w0/layer0.pt"
        )
    _write_manifest(capture_root, 0, manifest)
    output_dir = tmp_path / "bundle"

    with pytest.raises(ValueError, match="path|relative"):
        _assemble(capture_root, output_dir)
    assert not output_dir.exists()


def test_payload_tampering_after_rank_closure_is_rejected(tmp_path):
    capture_root = _closed_capture(tmp_path)
    payload = capture_root / "rank0/tensors/w0/layer0.pt"
    payload.write_bytes(payload.read_bytes() + b"tampered")
    output_dir = tmp_path / "bundle"

    with pytest.raises(ValueError, match="hash"):
        _assemble(capture_root, output_dir)
    assert not output_dir.exists()


def test_duplicate_tensor_ids_are_rejected(tmp_path):
    capture_root = _closed_capture(tmp_path)
    manifest = _read_manifest(capture_root, 0)
    manifest["tensors"][1] = dict(manifest["tensors"][0])
    _write_manifest(capture_root, 0, manifest)
    output_dir = tmp_path / "bundle"

    with pytest.raises(ValueError, match="tensor|inventory"):
        _assemble(capture_root, output_dir)
    assert not output_dir.exists()


def test_duplicate_output_relative_paths_are_rejected(tmp_path, monkeypatch):
    capture_root = _closed_capture(tmp_path)
    monkeypatch.setattr(
        assembler,
        "_source_relative_path",
        lambda record: Path("source/shared.pt"),
    )
    output_dir = tmp_path / "bundle"

    with pytest.raises(ValueError, match="output.*path|unique"):
        _assemble(capture_root, output_dir)
    assert not output_dir.exists()


@pytest.mark.parametrize(
    "workload_id",
    ("../escape", "nested/workload", "back\\slash", ".", ""),
)
def test_source_output_path_rejects_unsafe_workload_components(workload_id):
    record = SimpleNamespace(
        rank=0,
        workload_id=workload_id,
        layer_index=0,
    )

    with pytest.raises(ValueError, match="workload"):
        assembler._source_relative_path(record)


def test_extra_file_below_closed_rank_is_rejected(tmp_path):
    capture_root = _closed_capture(tmp_path)
    (capture_root / "rank0/extra.txt").write_text(
        "not evidence",
        encoding="utf-8",
    )
    output_dir = tmp_path / "bundle"

    with pytest.raises(ValueError, match="extra|inventory|untracked"):
        _assemble(capture_root, output_dir)
    assert not output_dir.exists()


def test_output_directory_cannot_be_inside_capture_root(tmp_path):
    capture_root = _closed_capture(tmp_path)
    output_dir = capture_root / "published-bundle"

    with pytest.raises(ValueError, match="output"):
        _assemble(capture_root, output_dir)
    assert not output_dir.exists()


@pytest.mark.parametrize("nonempty", (False, True))
def test_preexisting_output_directory_is_rejected(tmp_path, nonempty):
    capture_root = _closed_capture(tmp_path)
    output_dir = tmp_path / "bundle"
    output_dir.mkdir()
    if nonempty:
        (output_dir / "existing").write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="output"):
        _assemble(capture_root, output_dir)
    assert output_dir.is_dir()
    if nonempty:
        assert (output_dir / "existing").read_text("utf-8") == "keep"


def test_copy_failure_does_not_publish_final_output(tmp_path, monkeypatch):
    capture_root = _closed_capture(tmp_path)
    output_dir = tmp_path / "bundle"
    writes = 0
    original = assembler._write_payload_bytes

    def fail_second_write(path, payload):
        nonlocal writes
        writes += 1
        if writes == 2:
            raise OSError("synthetic copy failure")
        return original(path, payload)

    monkeypatch.setattr(assembler, "_write_payload_bytes", fail_second_write)

    with pytest.raises(OSError, match="synthetic copy failure"):
        _assemble(capture_root, output_dir)
    assert not output_dir.exists()
    assert not tuple(tmp_path.glob(".bundle.tmp-*"))


def test_atomic_publication_does_not_clobber_racing_output(
    tmp_path,
    monkeypatch,
):
    capture_root = _closed_capture(tmp_path)
    output_dir = tmp_path / "bundle"
    original_publish = assembler._publish_no_clobber

    def race_publish(temporary, final):
        final.mkdir()
        (final / "winner").write_text("racing writer", encoding="utf-8")
        return original_publish(temporary, final)

    monkeypatch.setattr(assembler, "_publish_no_clobber", race_publish)

    with pytest.raises(OSError):
        _assemble(capture_root, output_dir)
    assert (output_dir / "winner").read_text("utf-8") == "racing writer"
    assert not (output_dir / "source_bundle_manifest.json").exists()
    assert not tuple(tmp_path.glob(".bundle.tmp-*"))


def test_assembler_validates_and_writes_one_payload_at_a_time(
    tmp_path,
    monkeypatch,
):
    capture_root = _closed_capture(tmp_path)
    output_dir = tmp_path / "bundle"
    events = []
    original_write = assembler._write_payload_bytes

    def observe_load(source, *, map_location):
        events.append("load")
        return torch.load(
            source,
            map_location=map_location,
            weights_only=True,
        )

    def observe_write(path, payload):
        if Path(path).suffix == ".pt":
            events.append("write")
        return original_write(path, payload)

    monkeypatch.setattr(assembler, "_write_payload_bytes", observe_write)

    assembler.assemble_full_fidelity_bundle(
        capture_root=capture_root,
        output_dir=output_dir,
        model_manifest_sha256=MODEL_SHA256,
        source_tree_sha256=SOURCE_SHA256,
        workload_manifest_sha256=WORKLOAD_SHA256,
        world_size=2,
        load_tensor=observe_load,
    )

    assert events == ["load", "write"] * 8


def test_assembled_bundle_is_accepted_by_calibration_and_verifier(tmp_path):
    capture_root = _closed_capture(tmp_path)
    output_dir = tmp_path / "bundle"
    _assemble(capture_root, output_dir)
    manifest_path = output_dir / "source_bundle_manifest.json"
    thresholds_path = tmp_path / "thresholds.json"
    thresholds = {
        "schema_version": calibration_contract.THRESHOLD_SCHEMA_VERSION,
        "codec": calibration_contract.CODEC_ID,
        "pilot_source_bundle_sha256": _sha256(manifest_path),
        "max_abs_error": 100.0,
        "relative_l2_error": 100.0,
        "cosine_similarity": -1.0,
        "minimum_compression_ratio": 1.01,
    }
    thresholds_path.write_bytes(
        calibration_contract.canonical_json_bytes(thresholds) + b"\n"
    )
    calibration_dir = tmp_path / "calibration"

    calibration.run_calibration(
        output_dir,
        calibration_dir,
        thresholds_path=thresholds_path,
        load_tensor=lambda path: torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        ),
        save_tensor=torch.save,
    )
    verification = verifier.verify_calibration(calibration_dir)

    assert verification["classification"] in {"PASS", "NO_GO"}


def test_assembled_bundle_verifier_reports_real_no_go_classification(
    tmp_path,
):
    capture_root = _closed_capture(tmp_path)
    output_dir = tmp_path / "bundle"
    _assemble(capture_root, output_dir)
    manifest_path = output_dir / "source_bundle_manifest.json"
    thresholds_path = tmp_path / "thresholds.json"
    thresholds_path.write_bytes(
        calibration_contract.canonical_json_bytes({
            "schema_version": (
                calibration_contract.THRESHOLD_SCHEMA_VERSION
            ),
            "codec": calibration_contract.CODEC_ID,
            "pilot_source_bundle_sha256": _sha256(manifest_path),
            "max_abs_error": 100.0,
            "relative_l2_error": 100.0,
            "cosine_similarity": -1.0,
            "minimum_compression_ratio": 1_000_000.0,
        }) + b"\n"
    )
    calibration_dir = tmp_path / "calibration"
    calibration.run_calibration(
        output_dir,
        calibration_dir,
        thresholds_path=thresholds_path,
        load_tensor=lambda path: torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        ),
        save_tensor=torch.save,
    )

    verification = verifier.verify_calibration(calibration_dir)

    assert verification["classification"] == "NO_GO"
    assert verification["reasons"]


def test_cpu_full_capture_to_calibration_chain_is_closed_and_clean(
    tmp_path,
):
    capture_root = tmp_path / "capture"
    identity = _identity()
    base = torch.arange(
        24,
        dtype=torch.float32,
    ).reshape(2, 3, 4)

    for rank in range(2):
        for workload_index, workload_id in enumerate(WORKLOADS):
            session = Qwen35RecurrentCaptureSession(
                run_identity=identity,
                rank=rank,
                staging_dir=capture_root,
            )
            for layer_index in LAYERS:
                session.capture_layer(
                    workload_id=workload_id,
                    layer_index=layer_index,
                    tensor=base.add(
                        rank * 1000
                        + workload_index * 100
                        + layer_index
                    ),
                )
            session.finish_workload(workload_id)
        _close_rank(capture_root, rank)

    bundle_dir = tmp_path / "bundle"
    result = _assemble(capture_root, bundle_dir)
    manifest_path = bundle_dir / "source_bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    calibration_contract.validate_source_bundle_manifest(manifest)
    assert result["tensor_count"] == 8
    assert manifest["schema_version"] == (
        "qwen35.recurrent-full-fidelity-bundle.v1"
    )

    thresholds_path = tmp_path / "thresholds.json"
    thresholds_path.write_bytes(
        calibration_contract.canonical_json_bytes({
            "schema_version": (
                calibration_contract.THRESHOLD_SCHEMA_VERSION
            ),
            "codec": calibration_contract.CODEC_ID,
            "pilot_source_bundle_sha256": _sha256(manifest_path),
            "max_abs_error": 100.0,
            "relative_l2_error": 100.0,
            "cosine_similarity": -1.0,
            "minimum_compression_ratio": 1.01,
        }) + b"\n"
    )
    calibration_dir = tmp_path / "calibration"
    calibration.run_calibration(
        bundle_dir,
        calibration_dir,
        thresholds_path=thresholds_path,
        load_tensor=lambda path: torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        ),
        save_tensor=torch.save,
    )
    verification = verifier.verify_calibration(calibration_dir)
    assert verification["classification"] in {"PASS", "NO_GO"}

    expected_capture_files = {
        capture_root / f"rank{rank}/capture_identity.json"
        for rank in range(2)
    } | {
        capture_root / f"rank{rank}/rank_capture_manifest.json"
        for rank in range(2)
    } | {
        capture_root
        / f"rank{rank}/workloads/{workload_id}.complete.json"
        for rank in range(2)
        for workload_id in WORKLOADS
    } | {
        capture_root
        / (
            f"rank{rank}/tensors/{workload_id}/"
            f"layer{layer_index}.pt"
        )
        for rank in range(2)
        for workload_id in WORKLOADS
        for layer_index in LAYERS
    }
    actual_capture_files = {
        path
        for path in capture_root.rglob("*")
        if path.is_file()
    }
    assert actual_capture_files == expected_capture_files

    tracked_bundle_files = {
        bundle_dir / "source_bundle_manifest.json",
        *(
            bundle_dir / row["relative_path"]
            for row in manifest["tensors"]
        ),
    }
    actual_bundle_files = {
        path
        for path in bundle_dir.rglob("*")
        if path.is_file()
    }
    assert actual_bundle_files == tracked_bundle_files
    for root in (capture_root, bundle_dir, calibration_dir):
        for path in root.rglob("*"):
            assert not path.is_symlink()
            assert ".tmp-" not in path.name
            assert not path.name.endswith(".partial")
