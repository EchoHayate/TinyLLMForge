import importlib.util
from copy import deepcopy
from pathlib import Path
import sys


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


def test_exact_constants_and_expected_tensor_ids():
    assert contract.SCHEMA_VERSION == (
        "qwen35.recurrent-int8-calibration.v1"
    )
    assert contract.SOURCE_BUNDLE_SCHEMA_VERSION == (
        "qwen35.recurrent-full-fidelity-bundle.v1"
    )
    assert contract.CODEC_ID == (
        "qwen35_recurrent_symmetric_int8_per_row_v1"
    )
    assert contract.TOP_LEVEL_ARTIFACTS == (
        "source_bundle_manifest.json",
        "thresholds.json",
        "commands.json",
        "calibration_rows.jsonl",
        "summary.json",
        "artifact_manifest.json",
        "independent_verification.json",
        "report.md",
    )
    tensor_ids = contract.build_expected_tensor_ids(
        world_size=2,
        workload_ids=("w1",),
        linear_layer_indices=(0, 2),
    )
    assert tensor_ids == (
        "rank0:w1:layer0:linear_recurrent",
        "rank0:w1:layer2:linear_recurrent",
        "rank1:w1:layer0:linear_recurrent",
        "rank1:w1:layer2:linear_recurrent",
    )


def _hash(character="a"):
    return character * 64


def _manifest():
    tensors = []
    for rank in range(2):
        for layer_index in (0, 2):
            tensor_id = (
                f"rank{rank}:w1:layer{layer_index}:linear_recurrent"
            )
            tensors.append({
                "tensor_id": tensor_id,
                "rank": rank,
                "workload_id": "w1",
                "layer_index": layer_index,
                "relative_path": (
                    f"source/rank{rank}/w1/layer{layer_index}.pt"
                ),
                "sha256": _hash(chr(ord("a") + rank + layer_index)),
                "shape": [2, 3, 8],
                "dtype": "float32",
                "logical_bytes": 192,
            })
    return {
        "schema_version": contract.SOURCE_BUNDLE_SCHEMA_VERSION,
        "model_manifest_sha256": _hash("1"),
        "source_tree_sha256": _hash("2"),
        "workload_manifest_sha256": _hash("3"),
        "world_size": 2,
        "linear_layer_indices": [0, 2],
        "workload_ids": ["w1"],
        "tensors": tensors,
    }


def _threshold_payload():
    return {
        "schema_version": contract.THRESHOLD_SCHEMA_VERSION,
        "codec": contract.CODEC_ID,
        "pilot_source_bundle_sha256": _hash("4"),
        "max_abs_error": 0.1,
        "relative_l2_error": 0.02,
        "cosine_similarity": 0.999,
        "minimum_compression_ratio": 2.0,
    }


def _row(tensor_id="rank0:w1:layer0:linear_recurrent"):
    return {
        "tensor_id": tensor_id,
        "rank": int(tensor_id.split(":")[0].removeprefix("rank")),
        "workload_id": tensor_id.split(":")[1],
        "layer_index": int(
            tensor_id.split(":")[2].removeprefix("layer")
        ),
        "source_path": "source/rank0/w1/layer0.pt",
        "source_sha256": _hash("5"),
        "source_shape": [2, 3, 8],
        "source_dtype": "float32",
        "codec": contract.CODEC_ID,
        "encoded_values_path": "encoded_values/rank0/w1/layer0.pt",
        "encoded_values_sha256": _hash("6"),
        "encoded_values_shape": [2, 3, 8],
        "encoded_values_dtype": "int8",
        "scales_path": "scales/rank0/w1/layer0.pt",
        "scales_sha256": _hash("7"),
        "scales_shape": [2, 3],
        "scales_dtype": "float32",
        "decoded_path": "decoded/rank0/w1/layer0.pt",
        "decoded_sha256": _hash("8"),
        "decoded_shape": [2, 3, 8],
        "decoded_dtype": "float32",
        "logical_bytes": 192,
        "payload_bytes": 48,
        "scale_bytes": 24,
        "encoded_bytes": 72,
        "compression_ratio": 192 / 72,
        "zero_row_count": 1,
        "saturation_count": 5,
        "max_abs_error": 0.05,
        "mean_abs_error": 0.01,
        "rmse": 0.02,
        "relative_l2_error": 0.01,
        "cosine_similarity": 0.9999,
        "encode_ns": 100,
        "decode_ns": 80,
        "finite_source": True,
        "finite_scales": True,
        "finite_decoded": True,
    }


def test_source_bundle_manifest_and_thresholds_validate():
    assert contract.validate_source_bundle_manifest(_manifest()) == (
        "rank0:w1:layer0:linear_recurrent",
        "rank0:w1:layer2:linear_recurrent",
        "rank1:w1:layer0:linear_recurrent",
        "rank1:w1:layer2:linear_recurrent",
    )
    thresholds = contract.validate_thresholds(_threshold_payload())
    assert thresholds.max_abs_error == 0.1
    assert thresholds.minimum_compression_ratio == 2.0


def test_calibration_row_validates_exact_accounting():
    assert contract.validate_calibration_row(_row()) == ()


def test_unknown_fields_and_accounting_tamper_are_rejected():
    manifest = _manifest()
    manifest["unknown"] = True
    try:
        contract.validate_source_bundle_manifest(manifest)
    except ValueError as error:
        assert "fields" in str(error)
    else:
        raise AssertionError("manifest accepted an unknown field")

    row = _row()
    row["encoded_bytes"] += 1
    reasons = contract.validate_calibration_row(row)
    assert "encoded byte accounting mismatch" in reasons


def _rows_for_expected(expected):
    rows = []
    for tensor_id in expected:
        rank_text, workload_id, layer_text, _ = tensor_id.split(":")
        rank = int(rank_text.removeprefix("rank"))
        layer_index = int(layer_text.removeprefix("layer"))
        row = _row(tensor_id)
        for field_name, prefix in (
            ("source_path", "source"),
            ("encoded_values_path", "encoded_values"),
            ("scales_path", "scales"),
            ("decoded_path", "decoded"),
        ):
            row[field_name] = (
                f"{prefix}/rank{rank}/{workload_id}/"
                f"layer{layer_index}.pt"
            )
        rows.append(row)
    return tuple(rows)


def test_classifier_distinguishes_pass_no_go_and_invalid():
    expected = contract.build_expected_tensor_ids(
        world_size=2,
        workload_ids=("w1",),
        linear_layer_indices=(0, 2),
    )
    thresholds = contract.CalibrationThresholds(
        max_abs_error=0.1,
        relative_l2_error=0.02,
        cosine_similarity=0.999,
        minimum_compression_ratio=2.0,
    )
    rows = _rows_for_expected(expected)
    assert contract.classify_calibration(
        rows,
        expected_tensor_ids=expected,
        thresholds=thresholds,
    ) == ("PASS", ())

    no_go_rows = list(deepcopy(rows))
    no_go_rows[0]["relative_l2_error"] = 0.03
    classification, reasons = contract.classify_calibration(
        tuple(no_go_rows),
        expected_tensor_ids=expected,
        thresholds=thresholds,
    )
    assert classification == "NO_GO"
    assert any("relative_l2_error" in reason for reason in reasons)

    classification, reasons = contract.classify_calibration(
        rows[:-1],
        expected_tensor_ids=expected,
        thresholds=thresholds,
    )
    assert classification == "INVALID"
    assert any("tensor identity set" in reason for reason in reasons)


def test_classifier_rejects_false_finite_flags_as_invalid():
    expected = ("rank0:w1:layer0:linear_recurrent",)
    rows = list(_rows_for_expected(expected))
    rows[0]["finite_decoded"] = False
    classification, reasons = contract.classify_calibration(
        tuple(rows),
        expected_tensor_ids=expected,
        thresholds=contract.CalibrationThresholds(
            max_abs_error=0.1,
            relative_l2_error=0.02,
            cosine_similarity=0.999,
            minimum_compression_ratio=2.0,
        ),
    )
    assert classification == "INVALID"
    assert "finite_decoded must be true" in reasons


if __name__ == "__main__":
    test_exact_constants_and_expected_tensor_ids()
    test_source_bundle_manifest_and_thresholds_validate()
    test_calibration_row_validates_exact_accounting()
    test_unknown_fields_and_accounting_tamper_are_rejected()
    test_classifier_distinguishes_pass_no_go_and_invalid()
    test_classifier_rejects_false_finite_flags_as_invalid()
    print(
        "qwen35 recurrent int8 calibration contract tests passed "
        "(6 tests)"
    )
