import importlib.util
from dataclasses import replace
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]


def _load_module(module_name, relative_path):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


codec_module = _load_module(
    "tinyvllm.engine.qwen35_recurrent_int8_codec",
    "tinyvllm/engine/qwen35_recurrent_int8_codec.py",
)

QWEN35_RECURRENT_INT8_CODEC = (
    codec_module.QWEN35_RECURRENT_INT8_CODEC
)
Qwen35EncodedRecurrentInt8 = (
    codec_module.Qwen35EncodedRecurrentInt8
)
encode_qwen35_recurrent_int8_per_row = (
    codec_module.encode_qwen35_recurrent_int8_per_row
)
decode_qwen35_recurrent_int8_per_row = (
    codec_module.decode_qwen35_recurrent_int8_per_row
)
qwen35_recurrent_int8_error_metrics = (
    codec_module.qwen35_recurrent_int8_error_metrics
)


def test_encode_returns_exact_per_row_int8_metadata():
    source = torch.tensor(
        [[[
            -2.0, -1.0, 0.0, 2.0,
        ], [
            0.0, 0.0, 0.0, 0.0,
        ]]],
        dtype=torch.float32,
    )
    encoded = encode_qwen35_recurrent_int8_per_row(source)
    assert encoded.codec == QWEN35_RECURRENT_INT8_CODEC
    assert encoded.values.shape == source.shape
    assert encoded.values.dtype == torch.int8
    assert encoded.scales.shape == source.shape[:-1]
    assert encoded.scales.dtype == torch.float32
    assert encoded.source_shape == tuple(source.shape)
    assert encoded.source_dtype == torch.float32
    assert encoded.logical_bytes == source.numel() * 4
    assert encoded.payload_bytes == source.numel()
    assert encoded.scale_bytes == source.shape[0] * source.shape[1] * 4
    assert encoded.encoded_bytes == (
        encoded.payload_bytes + encoded.scale_bytes
    )
    assert encoded.values.min().item() >= -127
    assert encoded.values.max().item() <= 127
    assert encoded.scales[0, 1].item() == 1.0
    assert torch.count_nonzero(encoded.values[0, 1]).item() == 0


def test_decode_returns_private_finite_fp32_tensor():
    source = torch.arange(
        2 * 3 * 8,
        dtype=torch.float32,
    ).reshape(2, 3, 8) - 17.0
    encoded = encode_qwen35_recurrent_int8_per_row(source)
    decoded = decode_qwen35_recurrent_int8_per_row(encoded)
    assert decoded.shape == source.shape
    assert decoded.dtype == torch.float32
    assert decoded.device == source.device
    assert torch.isfinite(decoded).all().item()
    assert decoded.data_ptr() != source.data_ptr()
    assert decoded.data_ptr() != encoded.values.data_ptr()


def test_decode_can_target_an_explicit_device():
    source = torch.ones((1, 2, 4), dtype=torch.float32)
    encoded = encode_qwen35_recurrent_int8_per_row(source)
    decoded = decode_qwen35_recurrent_int8_per_row(
        encoded,
        device="cpu",
    )
    assert decoded.device.type == "cpu"


def test_decode_rejects_forbidden_negative_128():
    source = torch.ones((1, 2, 4), dtype=torch.float32)
    encoded = encode_qwen35_recurrent_int8_per_row(source)
    malformed_values = encoded.values.clone()
    malformed_values[0, 0, 0] = -128
    malformed = replace(encoded, values=malformed_values)
    try:
        decode_qwen35_recurrent_int8_per_row(malformed)
    except ValueError as error:
        assert "forbidden -128" in str(error)
    else:
        raise AssertionError("decoder accepted forbidden -128")


def test_error_metrics_are_recomputed_in_float64():
    source = torch.tensor(
        [[[0.0, 1.0, -2.0, 3.0]]],
        dtype=torch.float32,
    )
    decoded = source + torch.tensor(
        [[[0.0, 0.25, -0.5, 0.75]]],
        dtype=torch.float32,
    )
    metrics = qwen35_recurrent_int8_error_metrics(source, decoded)
    assert metrics["element_count"] == 4
    assert metrics["finite_source"] is True
    assert metrics["finite_decoded"] is True
    assert metrics["max_abs_error"] == 0.75
    assert metrics["mean_abs_error"] == 0.375
    assert metrics["rmse"] > 0
    assert metrics["relative_l2_error"] > 0
    assert -1.0 <= metrics["cosine_similarity"] <= 1.0


def test_error_metrics_define_both_zero_tensors_as_exact():
    source = torch.zeros((1, 2, 4), dtype=torch.float32)
    metrics = qwen35_recurrent_int8_error_metrics(source, source.clone())
    assert metrics["max_abs_error"] == 0.0
    assert metrics["mean_abs_error"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["relative_l2_error"] == 0.0
    assert metrics["cosine_similarity"] == 1.0


def test_error_metrics_reject_nonzero_decode_for_zero_source():
    source = torch.zeros((1, 2, 4), dtype=torch.float32)
    decoded = source.clone()
    decoded[0, 0, 0] = 1.0
    try:
        qwen35_recurrent_int8_error_metrics(source, decoded)
    except ValueError as error:
        assert "zero source norm" in str(error)
    else:
        raise AssertionError(
            "metrics accepted nonzero decode for zero source"
        )


def test_encode_isolated_from_later_source_mutation():
    source = torch.arange(
        2 * 3 * 4,
        dtype=torch.float32,
    ).reshape(2, 3, 4)
    encoded = encode_qwen35_recurrent_int8_per_row(source)
    values = encoded.values.clone()
    scales = encoded.scales.clone()
    source.add_(9000)
    torch.testing.assert_close(encoded.values, values)
    torch.testing.assert_close(encoded.scales, scales)


def test_decode_rejects_rank_two_encoded_shape():
    values = torch.zeros((2, 4), dtype=torch.int8)
    scales = torch.ones((2,), dtype=torch.float32)
    malformed = Qwen35EncodedRecurrentInt8(
        codec=QWEN35_RECURRENT_INT8_CODEC,
        values=values,
        scales=scales,
        source_shape=(2, 4),
        source_dtype=torch.float32,
        logical_bytes=32,
        payload_bytes=8,
        scale_bytes=8,
        encoded_bytes=16,
    )
    try:
        decode_qwen35_recurrent_int8_per_row(malformed)
    except ValueError as error:
        assert "rank three" in str(error)
    else:
        raise AssertionError("decoder accepted rank-two source shape")


def test_decode_rejects_noncontiguous_encoded_values():
    source = torch.arange(
        2 * 2 * 4,
        dtype=torch.float32,
    ).reshape(2, 2, 4)
    encoded = encode_qwen35_recurrent_int8_per_row(source)
    malformed = replace(
        encoded,
        values=encoded.values.transpose(0, 1),
    )
    assert malformed.values.shape == encoded.values.shape
    assert malformed.values.is_contiguous() is False
    try:
        decode_qwen35_recurrent_int8_per_row(malformed)
    except ValueError as error:
        assert "contiguous" in str(error)
    else:
        raise AssertionError("decoder accepted noncontiguous values")


def _expect_value_error(function, message):
    try:
        function()
    except ValueError as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(
            f"expected ValueError containing {message!r}"
        )


def test_encode_rejects_invalid_source_inputs():
    _expect_value_error(
        lambda: encode_qwen35_recurrent_int8_per_row("bad"),
        "must be a tensor",
    )
    _expect_value_error(
        lambda: encode_qwen35_recurrent_int8_per_row(
            torch.ones((2, 4), dtype=torch.float32)
        ),
        "rank three",
    )
    _expect_value_error(
        lambda: encode_qwen35_recurrent_int8_per_row(
            torch.ones((1, 2, 4), dtype=torch.bfloat16)
        ),
        "torch.float32",
    )
    invalid = torch.ones((1, 2, 4), dtype=torch.float32)
    invalid[0, 0, 0] = float("nan")
    _expect_value_error(
        lambda: encode_qwen35_recurrent_int8_per_row(invalid),
        "finite values",
    )


def test_decode_rejects_non_tensor_encoded_components():
    source = torch.ones((1, 2, 4), dtype=torch.float32)
    encoded = encode_qwen35_recurrent_int8_per_row(source)
    _expect_value_error(
        lambda: decode_qwen35_recurrent_int8_per_row(
            replace(encoded, values="bad")
        ),
        "values must be a tensor",
    )
    _expect_value_error(
        lambda: decode_qwen35_recurrent_int8_per_row(
            replace(encoded, scales="bad")
        ),
        "scales must be a tensor",
    )


def test_decode_rejects_scale_and_byte_accounting_tamper():
    source = torch.ones((1, 2, 4), dtype=torch.float32)
    encoded = encode_qwen35_recurrent_int8_per_row(source)
    zero_scales = encoded.scales.clone()
    zero_scales[0, 0] = 0
    _expect_value_error(
        lambda: decode_qwen35_recurrent_int8_per_row(
            replace(encoded, scales=zero_scales)
        ),
        "positive",
    )
    nan_scales = encoded.scales.clone()
    nan_scales[0, 0] = float("nan")
    _expect_value_error(
        lambda: decode_qwen35_recurrent_int8_per_row(
            replace(encoded, scales=nan_scales)
        ),
        "finite",
    )
    for field_name, message in (
        ("logical_bytes", "logical byte"),
        ("payload_bytes", "payload byte"),
        ("scale_bytes", "scale byte"),
        ("encoded_bytes", "total byte"),
    ):
        _expect_value_error(
            lambda field_name=field_name: (
                decode_qwen35_recurrent_int8_per_row(
                    replace(
                        encoded,
                        **{
                            field_name: (
                                getattr(encoded, field_name) + 1
                            )
                        },
                    )
                )
            ),
            message,
        )


if __name__ == "__main__":
    test_encode_returns_exact_per_row_int8_metadata()
    test_decode_returns_private_finite_fp32_tensor()
    test_decode_can_target_an_explicit_device()
    test_decode_rejects_forbidden_negative_128()
    test_error_metrics_are_recomputed_in_float64()
    test_error_metrics_define_both_zero_tensors_as_exact()
    test_error_metrics_reject_nonzero_decode_for_zero_source()
    test_encode_isolated_from_later_source_mutation()
    test_decode_rejects_rank_two_encoded_shape()
    test_decode_rejects_noncontiguous_encoded_values()
    test_encode_rejects_invalid_source_inputs()
    test_decode_rejects_non_tensor_encoded_components()
    test_decode_rejects_scale_and_byte_accounting_tamper()
    print("qwen35 recurrent int8 codec tests passed (13 tests)")
