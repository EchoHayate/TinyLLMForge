import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

import torch


ROOT = Path(__file__).resolve().parents[1]


def _load_module(module_name, relative_path):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


contract = _load_module(
    "qwen35_recurrent_int8_calibration_contract",
    "tools/qwen35_recurrent_int8_calibration_contract.py",
)
codec = _load_module(
    "tinyvllm.engine.qwen35_recurrent_int8_codec",
    "tinyvllm/engine/qwen35_recurrent_int8_codec.py",
)
producer = _load_module(
    "qwen35_recurrent_int8_calibration",
    "tools/qwen35_recurrent_int8_calibration.py",
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path, value):
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    )


def _read_rows(output):
    return [
        json.loads(line)
        for line in (
            output / "calibration_rows.jsonl"
        ).read_text().splitlines()
    ]


def _assert_partial_failure(
    output,
    *,
    error_type,
    completed_tensor_ids,
):
    failure = json.loads((output / "failure.json").read_text())
    assert failure["error_type"] == error_type
    assert failure["completed_tensor_ids"] == completed_tensor_ids
    assert not (output / "summary.json").exists()
    assert not (output / "artifact_manifest.json").exists()


def _refresh_threshold_binding(bundle, thresholds):
    payload = json.loads(thresholds.read_text())
    payload["pilot_source_bundle_sha256"] = _sha256(
        bundle / "source_bundle_manifest.json"
    )
    _write_json(thresholds, payload)


def _fixture(root):
    bundle = root / "bundle"
    tensors = []
    for rank in range(2):
        for layer_index in (0, 1):
            relative = Path(
                f"source/rank{rank}/w1/layer{layer_index}.pt"
            )
            path = bundle / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            tensor = torch.arange(
                2 * 3 * 8,
                dtype=torch.float32,
            ).reshape(2, 3, 8)
            tensor = tensor + rank * 100 + layer_index * 10 - 17
            tensor[0, 0].zero_()
            torch.save(tensor, path)
            tensors.append({
                "tensor_id": (
                    f"rank{rank}:w1:layer{layer_index}:linear_recurrent"
                ),
                "rank": rank,
                "workload_id": "w1",
                "layer_index": layer_index,
                "relative_path": relative.as_posix(),
                "sha256": _sha256(path),
                "shape": [2, 3, 8],
                "dtype": "float32",
                "logical_bytes": tensor.numel() * tensor.element_size(),
            })
    manifest = {
        "schema_version": contract.SOURCE_BUNDLE_SCHEMA_VERSION,
        "model_manifest_sha256": "1" * 64,
        "source_tree_sha256": "2" * 64,
        "workload_manifest_sha256": "3" * 64,
        "world_size": 2,
        "linear_layer_indices": [0, 1],
        "workload_ids": ["w1"],
        "tensors": tensors,
    }
    manifest_path = bundle / "source_bundle_manifest.json"
    _write_json(manifest_path, manifest)
    thresholds = {
        "schema_version": contract.THRESHOLD_SCHEMA_VERSION,
        "codec": contract.CODEC_ID,
        "pilot_source_bundle_sha256": _sha256(manifest_path),
        "max_abs_error": 2.0,
        "relative_l2_error": 1.0,
        "cosine_similarity": -1.0,
        "minimum_compression_ratio": 1.1,
    }
    thresholds_path = root / "thresholds.json"
    _write_json(thresholds_path, thresholds)
    return bundle, thresholds_path


class _Clock:
    def __init__(self):
        self.value = 100

    def __call__(self):
        self.value += 10
        return self.value


def test_producer_writes_complete_non_authoritative_artifacts():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        bundle, thresholds = _fixture(root)
        output = root / "output"
        before = {
            path.relative_to(bundle).as_posix(): _sha256(path)
            for path in bundle.rglob("*")
            if path.is_file()
        }
        result = producer.run_calibration(
            bundle,
            output,
            thresholds_path=thresholds,
            load_tensor=lambda path: torch.load(
                path,
                map_location="cpu",
                weights_only=True,
            ),
            save_tensor=torch.save,
            clock_ns=_Clock(),
        )
        assert result["classification"] in {"PASS", "NO_GO"}
        assert result["row_count"] == 4
        for name in (
            "source_bundle_manifest.json",
            "thresholds.json",
            "commands.json",
            "calibration_rows.jsonl",
            "summary.json",
            "artifact_manifest.json",
        ):
            assert (output / name).is_file(), name
        assert not (output / "independent_verification.json").exists()
        assert not (output / "report.md").exists()
        rows = _read_rows(output)
        assert len(rows) == 4
        assert all(contract.validate_calibration_row(row) == () for row in rows)
        expected_tensor_paths = {
            f"{kind}/rank{rank}/w1/layer{layer_index}.pt"
            for kind in ("source", "encoded_values", "scales", "decoded")
            for rank in range(2)
            for layer_index in (0, 1)
        }
        expected_manifest_paths = expected_tensor_paths | {
            "source_bundle_manifest.json",
            "thresholds.json",
            "commands.json",
            "calibration_rows.jsonl",
            "summary.json",
        }
        artifact_manifest = json.loads(
            (output / "artifact_manifest.json").read_text()
        )
        assert {
            artifact["path"]
            for artifact in artifact_manifest["artifacts"]
        } == expected_manifest_paths
        commands = json.loads((output / "commands.json").read_text())
        assert commands["codec"] == contract.CODEC_ID
        assert commands["started_at_utc"] <= commands["finished_at_utc"]
        assert commands["producer_only"] is True
        assert "independent" not in commands
        for row in rows:
            source = torch.load(
                output / row["source_path"],
                map_location="cpu",
                weights_only=True,
            )
            decoded = torch.load(
                output / row["decoded_path"],
                map_location="cpu",
                weights_only=True,
            )
            original = torch.load(
                bundle
                / next(
                    tensor["relative_path"]
                    for tensor in json.loads(
                        (
                            bundle / "source_bundle_manifest.json"
                        ).read_text()
                    )["tensors"]
                    if tensor["tensor_id"] == row["tensor_id"]
                ),
                map_location="cpu",
                weights_only=True,
            )
            assert torch.equal(source, original)
            assert decoded.dtype == torch.float32
            assert torch.isfinite(decoded).all().item()
        after = {
            path.relative_to(bundle).as_posix(): _sha256(path)
            for path in bundle.rglob("*")
            if path.is_file()
        }
        assert after == before


def test_preflight_failure_creates_no_output():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        bundle, thresholds = _fixture(root)
        manifest_path = bundle / "source_bundle_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["tensors"][0]["sha256"] = "f" * 64
        _write_json(manifest_path, manifest)
        _refresh_threshold_binding(bundle, thresholds)
        output = root / "output"
        try:
            producer.run_calibration(
                bundle,
                output,
                thresholds_path=thresholds,
                load_tensor=torch.load,
                save_tensor=torch.save,
                clock_ns=_Clock(),
            )
        except ValueError as error:
            assert "hash mismatch" in str(error)
        else:
            raise AssertionError("producer accepted a source hash mismatch")
        assert not output.exists()


def test_preflight_rejects_unknown_file_symlink_and_nonempty_output():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        bundle, thresholds = _fixture(root)
        (bundle / "unknown.bin").write_bytes(b"unknown")
        output = root / "unknown-output"
        try:
            producer.run_calibration(
                bundle,
                output,
                thresholds_path=thresholds,
                load_tensor=torch.load,
                save_tensor=torch.save,
            )
        except ValueError as error:
            assert "inventory mismatch" in str(error)
        else:
            raise AssertionError("producer accepted an unknown source file")
        assert not output.exists()

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        bundle, thresholds = _fixture(root)
        target = bundle / "source/rank0/w1/layer0.pt"
        target.unlink()
        target.symlink_to(bundle / "source/rank0/w1/layer1.pt")
        output = root / "symlink-output"
        try:
            producer.run_calibration(
                bundle,
                output,
                thresholds_path=thresholds,
                load_tensor=torch.load,
                save_tensor=torch.save,
            )
        except ValueError as error:
            assert "regular file" in str(error)
        else:
            raise AssertionError("producer accepted a source symlink")
        assert not output.exists()

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        bundle, thresholds = _fixture(root)
        output = root / "existing-output"
        output.mkdir()
        (output / "occupied").write_text("occupied")
        try:
            producer.run_calibration(
                bundle,
                output,
                thresholds_path=thresholds,
                load_tensor=torch.load,
                save_tensor=torch.save,
            )
        except ValueError as error:
            assert "absent or empty" in str(error)
        else:
            raise AssertionError("producer accepted non-empty output")
        assert not (output / "failure.json").exists()


def test_preflight_rejects_malformed_and_unbound_thresholds():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        bundle, thresholds = _fixture(root)
        payload = json.loads(thresholds.read_text())
        payload["extra"] = "forbidden"
        _write_json(thresholds, payload)
        output = root / "malformed-output"
        try:
            producer.run_calibration(
                bundle,
                output,
                thresholds_path=thresholds,
                load_tensor=torch.load,
                save_tensor=torch.save,
            )
        except ValueError as error:
            assert "fields mismatch" in str(error)
        else:
            raise AssertionError("producer accepted malformed thresholds")
        assert not output.exists()

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        bundle, thresholds = _fixture(root)
        payload = json.loads(thresholds.read_text())
        payload["pilot_source_bundle_sha256"] = "0" * 64
        _write_json(thresholds, payload)
        output = root / "unbound-output"
        try:
            producer.run_calibration(
                bundle,
                output,
                thresholds_path=thresholds,
                load_tensor=torch.load,
                save_tensor=torch.save,
            )
        except ValueError as error:
            assert "manifest binding" in str(error)
        else:
            raise AssertionError("producer accepted unbound thresholds")
        assert not output.exists()


def test_midrun_loader_encoder_save_and_clock_failures_are_not_success():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        bundle, thresholds = _fixture(root)
        output = root / "loader-output"
        calls = 0

        def load_second_fails(path):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise LookupError("injected loader failure")
            return torch.load(path, map_location="cpu", weights_only=True)

        try:
            producer.run_calibration(
                bundle,
                output,
                thresholds_path=thresholds,
                load_tensor=load_second_fails,
                save_tensor=torch.save,
                clock_ns=_Clock(),
            )
        except LookupError:
            pass
        else:
            raise AssertionError("producer swallowed loader failure")
        _assert_partial_failure(
            output,
            error_type="LookupError",
            completed_tensor_ids=["rank0:w1:layer0:linear_recurrent"],
        )

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        bundle, thresholds = _fixture(root)
        output = root / "encoder-output"
        original_encode = producer.encode_qwen35_recurrent_int8_per_row
        calls = 0

        def encode_second_fails(source):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise ArithmeticError("injected encoder failure")
            return original_encode(source)

        producer.encode_qwen35_recurrent_int8_per_row = encode_second_fails
        try:
            try:
                producer.run_calibration(
                    bundle,
                    output,
                    thresholds_path=thresholds,
                    load_tensor=lambda path: torch.load(
                        path,
                        map_location="cpu",
                        weights_only=True,
                    ),
                    save_tensor=torch.save,
                    clock_ns=_Clock(),
                )
            except ArithmeticError:
                pass
            else:
                raise AssertionError("producer swallowed encoder failure")
        finally:
            producer.encode_qwen35_recurrent_int8_per_row = original_encode
        _assert_partial_failure(
            output,
            error_type="ArithmeticError",
            completed_tensor_ids=["rank0:w1:layer0:linear_recurrent"],
        )

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        bundle, thresholds = _fixture(root)
        output = root / "save-output"
        calls = 0

        def save_after_values_fails(tensor, path):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("injected scale save failure")
            torch.save(tensor, path)

        try:
            producer.run_calibration(
                bundle,
                output,
                thresholds_path=thresholds,
                load_tensor=lambda path: torch.load(
                    path,
                    map_location="cpu",
                    weights_only=True,
                ),
                save_tensor=save_after_values_fails,
                clock_ns=_Clock(),
            )
        except OSError:
            pass
        else:
            raise AssertionError("producer swallowed save failure")
        _assert_partial_failure(
            output,
            error_type="OSError",
            completed_tensor_ids=[],
        )

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        bundle, thresholds = _fixture(root)
        output = root / "clock-output"
        timestamps = iter((20, 10))
        try:
            producer.run_calibration(
                bundle,
                output,
                thresholds_path=thresholds,
                load_tensor=lambda path: torch.load(
                    path,
                    map_location="cpu",
                    weights_only=True,
                ),
                save_tensor=torch.save,
                clock_ns=lambda: next(timestamps),
            )
        except ValueError as error:
            assert "non-decreasing" in str(error)
        else:
            raise AssertionError("producer accepted a decreasing clock")
        _assert_partial_failure(
            output,
            error_type="ValueError",
            completed_tensor_ids=[],
        )


def test_producer_is_cpu_only_and_rows_are_repeatable():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        bundle, thresholds = _fixture(root)
        original_lazy_init = torch.cuda._lazy_init

        def reject_cuda():
            raise AssertionError("producer initialized CUDA")

        torch.cuda._lazy_init = reject_cuda
        try:
            for name in ("first", "second"):
                producer.run_calibration(
                    bundle,
                    root / name,
                    thresholds_path=thresholds,
                    load_tensor=lambda path: torch.load(
                        path,
                        map_location="cpu",
                        weights_only=True,
                    ),
                    save_tensor=torch.save,
                    clock_ns=_Clock(),
                )
        finally:
            torch.cuda._lazy_init = original_lazy_init
        first_rows = _read_rows(root / "first")
        second_rows = _read_rows(root / "second")
        for rows in (first_rows, second_rows):
            for row in rows:
                row.pop("encode_ns")
                row.pop("decode_ns")
        assert first_rows == second_rows


def test_thresholds_are_canonicalized_in_output():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        bundle, thresholds = _fixture(root)
        payload = json.loads(thresholds.read_text())
        thresholds.write_text(json.dumps(payload, indent=4) + "\n")
        output = root / "output"
        producer.run_calibration(
            bundle,
            output,
            thresholds_path=thresholds,
            load_tensor=lambda path: torch.load(
                path,
                map_location="cpu",
                weights_only=True,
            ),
            save_tensor=torch.save,
            clock_ns=_Clock(),
        )
        assert (output / "thresholds.json").read_bytes() == (
            contract.canonical_json_bytes(payload) + b"\n"
        )


def test_commands_redact_runtime_arguments():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        bundle, thresholds = _fixture(root)
        output = root / "output"
        original_argv = sys.argv
        sys.argv = [
            "/Users/bytedance/private/producer.py",
            "--token",
            "secret-value",
            "--ssh",
            "sitian@10.232.195.203",
        ]
        try:
            producer.run_calibration(
                bundle,
                output,
                thresholds_path=thresholds,
                load_tensor=lambda path: torch.load(
                    path,
                    map_location="cpu",
                    weights_only=True,
                ),
                save_tensor=torch.save,
                clock_ns=_Clock(),
            )
        finally:
            sys.argv = original_argv
        commands_text = (output / "commands.json").read_text()
        commands = json.loads(commands_text)
        assert commands["argv"] == [
            "producer.py",
            "<redacted>",
            "<redacted>",
            "<redacted>",
            "<redacted>",
        ]
        for forbidden in (
            "/Users/bytedance",
            "secret-value",
            "sitian@",
            "10.232.195.203",
        ):
            assert forbidden not in commands_text


if __name__ == "__main__":
    test_producer_writes_complete_non_authoritative_artifacts()
    test_preflight_failure_creates_no_output()
    test_preflight_rejects_unknown_file_symlink_and_nonempty_output()
    test_preflight_rejects_malformed_and_unbound_thresholds()
    test_midrun_loader_encoder_save_and_clock_failures_are_not_success()
    test_producer_is_cpu_only_and_rows_are_repeatable()
    test_thresholds_are_canonicalized_in_output()
    test_commands_redact_runtime_arguments()
    print(
        "qwen35 recurrent int8 calibration producer tests passed "
        "(8 tests)"
    )
