import hashlib
import importlib.util
import json
import math
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
verifier = _load_module(
    "verify_qwen35_recurrent_int8_calibration",
    "tools/verify_qwen35_recurrent_int8_calibration.py",
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path, value):
    path.write_bytes(contract.canonical_json_bytes(value) + b"\n")


def _write_jsonl(path, rows):
    path.write_bytes(
        b"".join(
            contract.canonical_json_bytes(row) + b"\n"
            for row in rows
        )
    )


class _Clock:
    def __init__(self):
        self.value = 100

    def __call__(self):
        self.value += 10
        return self.value


def _fixture(root, *, threshold_overrides=None):
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
    if threshold_overrides is not None:
        thresholds.update(threshold_overrides)
    thresholds_path = root / "thresholds.json"
    _write_json(thresholds_path, thresholds)
    run_dir = root / "run"
    producer.run_calibration(
        bundle,
        run_dir,
        thresholds_path=thresholds_path,
        load_tensor=lambda path: torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        ),
        save_tensor=torch.save,
        clock_ns=_Clock(),
    )
    return run_dir


def _rows(run_dir):
    return [
        json.loads(line)
        for line in (
            run_dir / "calibration_rows.jsonl"
        ).read_text().splitlines()
    ]


def _refresh_artifact(run_dir, relative_path):
    manifest_path = run_dir / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    matches = [
        artifact
        for artifact in manifest["artifacts"]
        if artifact["path"] == relative_path
    ]
    assert len(matches) == 1, relative_path
    path = run_dir / relative_path
    matches[0]["size"] = path.stat().st_size
    matches[0]["sha256"] = _sha256(path)
    _write_json(manifest_path, manifest)


def _rewrite_rows(run_dir, rows):
    _write_jsonl(run_dir / "calibration_rows.jsonl", rows)
    _refresh_artifact(run_dir, "calibration_rows.jsonl")


def _assert_invalid(mutator):
    with tempfile.TemporaryDirectory() as temporary:
        run_dir = _fixture(Path(temporary))
        mutator(run_dir)
        try:
            verifier.verify_calibration(run_dir)
        except ValueError:
            pass
        else:
            raise AssertionError("verifier accepted tampered artifacts")
        assert not (run_dir / "independent_verification.json").exists()
        assert not (run_dir / "report.md").exists()


def test_valid_artifacts_are_independently_verified_once():
    with tempfile.TemporaryDirectory() as temporary:
        run_dir = _fixture(Path(temporary))
        producer_summary = json.loads(
            (run_dir / "summary.json").read_text()
        )
        result = verifier.verify_calibration(run_dir)
        assert result["classification"] == producer_summary["classification"]
        assert result["classification"] in {"PASS", "NO_GO"}
        assert result["independent"] is True
        assert result["tensor_count"] == 4
        assert result["logical_bytes"] == 4 * 2 * 3 * 8 * 4
        assert result["payload_bytes"] == 4 * 2 * 3 * 8
        assert result["scale_bytes"] == 4 * 2 * 3 * 4
        assert result["encoded_bytes"] == (
            result["payload_bytes"] + result["scale_bytes"]
        )
        assert math.isclose(
            result["compression_ratio"],
            result["logical_bytes"] / result["encoded_bytes"],
        )
        verification_path = run_dir / "independent_verification.json"
        report_path = run_dir / "report.md"
        assert json.loads(verification_path.read_text()) == result
        report = report_path.read_text()
        for expected in (
            "Independent Classification",
            "Producer-observed Timing",
            "model manifest",
            "source tree",
            "workload manifest",
            "No runtime integration",
            "GPU-memory",
            "speed",
            "quality authority",
        ):
            assert expected in report
        try:
            verifier.verify_calibration(run_dir)
        except ValueError as error:
            assert "single-use" in str(error)
        else:
            raise AssertionError("public verifier was not single-use")


def test_inventory_manifest_and_symlink_tampering_are_rejected():
    _assert_invalid(
        lambda run_dir: (run_dir / "extra.txt").write_text("extra")
    )
    _assert_invalid(
        lambda run_dir: (
            run_dir / "encoded_values/extra.pt"
        ).write_bytes(b"extra")
    )
    _assert_invalid(
        lambda run_dir: (run_dir / "unknown-empty-directory").mkdir()
    )

    def remove_tensor(run_dir):
        (run_dir / _rows(run_dir)[0]["decoded_path"]).unlink()

    _assert_invalid(remove_tensor)

    def replace_with_symlink(run_dir):
        path = run_dir / _rows(run_dir)[0]["decoded_path"]
        path.unlink()
        path.symlink_to(run_dir / _rows(run_dir)[1]["decoded_path"])

    _assert_invalid(replace_with_symlink)

    def break_manifest_hash(run_dir):
        path = run_dir / _rows(run_dir)[0]["decoded_path"]
        path.write_bytes(path.read_bytes() + b"tamper")

    _assert_invalid(break_manifest_hash)


def test_row_order_identity_accounting_metric_and_flag_tampering_is_rejected():
    def reordered(run_dir):
        rows = _rows(run_dir)
        rows[0], rows[1] = rows[1], rows[0]
        _rewrite_rows(run_dir, rows)

    _assert_invalid(reordered)

    def duplicate(run_dir):
        rows = _rows(run_dir)
        rows[1] = dict(rows[0])
        _rewrite_rows(run_dir, rows)

    _assert_invalid(duplicate)

    mutations = (
        ("codec", "changed-codec"),
        ("logical_bytes", 1),
        ("payload_bytes", 1),
        ("scale_bytes", 1),
        ("encoded_bytes", 1),
        ("compression_ratio", 99.0),
        ("zero_row_count", 99),
        ("saturation_count", 99),
        ("max_abs_error", 99.0),
        ("mean_abs_error", 99.0),
        ("rmse", 99.0),
        ("relative_l2_error", 99.0),
        ("cosine_similarity", 0.0),
        ("encode_ns", -1),
        ("decode_ns", -1.5),
        ("finite_source", False),
        ("finite_scales", False),
        ("finite_decoded", False),
    )
    for field, value in mutations:
        def mutate(run_dir, field=field, value=value):
            rows = _rows(run_dir)
            rows[0][field] = value
            _rewrite_rows(run_dir, rows)

        _assert_invalid(mutate)

    def nonfinite_timing(run_dir):
        rows = _rows(run_dir)
        rows[0]["decode_ns"] = float("inf")
        (run_dir / "calibration_rows.jsonl").write_text(
            "".join(
                json.dumps(
                    row,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=True,
                )
                + "\n"
                for row in rows
            )
        )
        _refresh_artifact(run_dir, "calibration_rows.jsonl")

    _assert_invalid(nonfinite_timing)


def test_tensor_value_dtype_shape_and_hash_tampering_is_rejected():
    def mutate_source(run_dir):
        rows = _rows(run_dir)
        path = run_dir / rows[0]["source_path"]
        tensor = torch.load(path, weights_only=True)
        tensor[0, 0, 0] += 1
        torch.save(tensor, path)
        rows[0]["source_sha256"] = _sha256(path)
        _rewrite_rows(run_dir, rows)
        _refresh_artifact(run_dir, rows[0]["source_path"])

    _assert_invalid(mutate_source)

    def mutate_values(run_dir, value=-128):
        rows = _rows(run_dir)
        path = run_dir / rows[0]["encoded_values_path"]
        tensor = torch.load(path, weights_only=True)
        tensor[0, 0, 0] = value
        torch.save(tensor, path)
        rows[0]["encoded_values_sha256"] = _sha256(path)
        _rewrite_rows(run_dir, rows)
        _refresh_artifact(run_dir, rows[0]["encoded_values_path"])

    _assert_invalid(mutate_values)
    _assert_invalid(
        lambda run_dir: mutate_values(run_dir, value=1)
    )

    def mutate_scale(run_dir, value=0.0):
        rows = _rows(run_dir)
        path = run_dir / rows[0]["scales_path"]
        tensor = torch.load(path, weights_only=True)
        tensor[0, 0] = value
        torch.save(tensor, path)
        rows[0]["scales_sha256"] = _sha256(path)
        _rewrite_rows(run_dir, rows)
        _refresh_artifact(run_dir, rows[0]["scales_path"])

    for value in (0.0, -1.0, float("nan"), float("inf")):
        _assert_invalid(
            lambda run_dir, value=value: mutate_scale(run_dir, value)
        )
    _assert_invalid(
        lambda run_dir: mutate_scale(run_dir, value=2.0)
    )

    def mutate_decoded(run_dir):
        rows = _rows(run_dir)
        path = run_dir / rows[0]["decoded_path"]
        tensor = torch.load(path, weights_only=True)
        tensor[0, 0, 0] += 1
        torch.save(tensor, path)
        rows[0]["decoded_sha256"] = _sha256(path)
        _rewrite_rows(run_dir, rows)
        _refresh_artifact(run_dir, rows[0]["decoded_path"])

    _assert_invalid(mutate_decoded)

    def mutate_decoded_dtype(run_dir):
        rows = _rows(run_dir)
        path = run_dir / rows[0]["decoded_path"]
        tensor = torch.load(path, weights_only=True).to(torch.float64)
        torch.save(tensor, path)
        rows[0]["decoded_sha256"] = _sha256(path)
        rows[0]["decoded_dtype"] = "float64"
        _rewrite_rows(run_dir, rows)
        _refresh_artifact(run_dir, rows[0]["decoded_path"])

    _assert_invalid(mutate_decoded_dtype)

    def mutate_source_dtype(run_dir):
        rows = _rows(run_dir)
        path = run_dir / rows[0]["source_path"]
        tensor = torch.load(path, weights_only=True).to(torch.float64)
        torch.save(tensor, path)
        rows[0]["source_sha256"] = _sha256(path)
        rows[0]["source_dtype"] = "float64"
        _rewrite_rows(run_dir, rows)
        _refresh_artifact(run_dir, rows[0]["source_path"])

    _assert_invalid(mutate_source_dtype)

    def mutate_shape(run_dir):
        rows = _rows(run_dir)
        path = run_dir / rows[0]["decoded_path"]
        tensor = torch.load(path, weights_only=True).reshape(3, 2, 8)
        torch.save(tensor, path)
        rows[0]["decoded_sha256"] = _sha256(path)
        rows[0]["decoded_shape"] = [3, 2, 8]
        _rewrite_rows(run_dir, rows)
        _refresh_artifact(run_dir, rows[0]["decoded_path"])

    _assert_invalid(mutate_shape)

    def remove_expected_row(run_dir):
        rows = _rows(run_dir)
        _rewrite_rows(run_dir, rows[:-1])

    _assert_invalid(remove_expected_row)


def test_coordinated_noncanonical_encoding_is_rejected():
    def coordinated_mutation(run_dir):
        rows = _rows(run_dir)
        row = rows[0]
        source_path = run_dir / row["source_path"]
        values_path = run_dir / row["encoded_values_path"]
        scales_path = run_dir / row["scales_path"]
        decoded_path = run_dir / row["decoded_path"]
        source = torch.load(source_path, weights_only=True)
        values = torch.load(values_path, weights_only=True)
        scales = torch.load(scales_path, weights_only=True)
        original = int(values[1, 0, 0].item())
        replacement = original + 1 if original < 127 else original - 1
        values[1, 0, 0] = replacement
        decoded = (
            values.to(dtype=torch.float32) * scales.unsqueeze(-1)
        ).contiguous()
        torch.save(values, values_path)
        torch.save(decoded, decoded_path)
        row["encoded_values_sha256"] = _sha256(values_path)
        row["decoded_sha256"] = _sha256(decoded_path)
        row["saturation_count"] = int(
            torch.logical_or(values == -127, values == 127).sum().item()
        )
        metrics = codec.qwen35_recurrent_int8_error_metrics(
            source,
            decoded,
        )
        for field in (
            "max_abs_error",
            "mean_abs_error",
            "rmse",
            "relative_l2_error",
            "cosine_similarity",
        ):
            row[field] = metrics[field]
        _rewrite_rows(run_dir, rows)
        _refresh_artifact(run_dir, row["encoded_values_path"])
        _refresh_artifact(run_dir, row["decoded_path"])

    _assert_invalid(coordinated_mutation)


def test_thresholds_override_untrusted_producer_summary():
    with tempfile.TemporaryDirectory() as temporary:
        run_dir = _fixture(
            Path(temporary),
            threshold_overrides={"max_abs_error": 1e-15},
        )
        original_result = json.loads(
            (run_dir / "summary.json").read_text()
        )
        assert original_result["classification"] == "NO_GO"
        summary = dict(original_result)
        summary["classification"] = "PASS"
        summary["reasons"] = []
        _write_json(run_dir / "summary.json", summary)
        _refresh_artifact(run_dir, "summary.json")
        result = verifier.verify_calibration(run_dir)
        assert result["classification"] == "NO_GO"
        assert any(
            "max_abs_error exceeds threshold" in reason
            for reason in result["reasons"]
        )


def test_post_production_threshold_replacement_is_rejected():
    with tempfile.TemporaryDirectory() as temporary:
        run_dir = _fixture(Path(temporary))
        thresholds = json.loads((run_dir / "thresholds.json").read_text())
        thresholds["max_abs_error"] = 1e-15
        _write_json(run_dir / "thresholds.json", thresholds)
        _refresh_artifact(run_dir, "thresholds.json")
        try:
            verifier.verify_calibration(run_dir)
        except ValueError as error:
            assert "threshold binding" in str(error)
        else:
            raise AssertionError("verifier accepted replaced thresholds")


if __name__ == "__main__":
    test_valid_artifacts_are_independently_verified_once()
    test_inventory_manifest_and_symlink_tampering_are_rejected()
    test_row_order_identity_accounting_metric_and_flag_tampering_is_rejected()
    test_tensor_value_dtype_shape_and_hash_tampering_is_rejected()
    test_coordinated_noncanonical_encoding_is_rejected()
    test_thresholds_override_untrusted_producer_summary()
    test_post_production_threshold_replacement_is_rejected()
    print(
        "qwen35 recurrent int8 calibration verifier tests passed "
        "(7 tests)"
    )
