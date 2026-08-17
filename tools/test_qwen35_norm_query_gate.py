import importlib.util
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "tinyvllm/layers/qwen35_primitives.py"
    spec = importlib.util.spec_from_file_location(
        "qwen35_primitives_test_target", path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


primitives = _load_module()
Qwen35OffsetRMSNorm = primitives.Qwen35OffsetRMSNorm
qwen35_apply_query_gate = primitives.qwen35_apply_query_gate


def _manual_offset_rmsnorm(
    tensor: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    tensor_fp32 = tensor.float()
    normalized = tensor_fp32 * torch.rsqrt(
        tensor_fp32.pow(2).mean(dim=-1, keepdim=True) + eps
    )
    return (normalized * (1.0 + weight.float())).to(tensor.dtype)


def test_zero_weight_matches_plain_rms_normalization() -> None:
    tensor = torch.tensor(
        [[[1.0, -2.0, 3.0], [0.5, 0.25, -0.75]]],
        dtype=torch.float32,
    )
    layer = Qwen35OffsetRMSNorm(3, eps=1e-6)
    original = tensor.clone()
    output = layer(tensor)
    expected = _manual_offset_rmsnorm(tensor, layer.weight, layer.eps)
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(tensor, original)
    torch.testing.assert_close(layer.weight, torch.zeros(3))


def test_nonzero_offsets_use_one_plus_weight() -> None:
    tensor = torch.tensor([[1.0, 2.0, -1.0]], dtype=torch.float32)
    layer = Qwen35OffsetRMSNorm(3)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([0.5, -0.25, -1.5]))
    output = layer(tensor)
    expected = _manual_offset_rmsnorm(tensor, layer.weight, layer.eps)
    torch.testing.assert_close(output, expected)

    normalized = _manual_offset_rmsnorm(
        tensor, torch.zeros_like(layer.weight), layer.eps
    )
    legacy_direct_weight = normalized * layer.weight
    assert not torch.allclose(output, legacy_direct_weight)


def test_bfloat16_preserves_dtype_with_fp32_oracle() -> None:
    tensor = torch.tensor(
        [[1.0, -3.0, 2.0], [0.25, 0.5, -0.125]],
        dtype=torch.bfloat16,
    )
    layer = Qwen35OffsetRMSNorm(3)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([0.2, -0.4, 0.1]))
    output = layer(tensor)
    expected = _manual_offset_rmsnorm(tensor, layer.weight, layer.eps)
    assert output.dtype == torch.bfloat16
    torch.testing.assert_close(output.float(), expected.float())


def _expect_value_error(function, message: str) -> None:
    try:
        function()
    except ValueError as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(f"expected ValueError containing {message!r}")


def test_offset_rmsnorm_rejects_invalid_contracts() -> None:
    for hidden_size in (True, 0, -1, 2.5):
        _expect_value_error(
            lambda hidden_size=hidden_size: Qwen35OffsetRMSNorm(hidden_size),
            "hidden_size",
        )
    for eps in (0.0, -1.0, float("inf"), float("nan")):
        _expect_value_error(
            lambda eps=eps: Qwen35OffsetRMSNorm(3, eps=eps),
            "eps",
        )
    layer = Qwen35OffsetRMSNorm(3)
    _expect_value_error(
        lambda: layer(torch.ones(2, 4)),
        "last dimension",
    )
    _expect_value_error(
        lambda: layer(torch.ones(2, 3, dtype=torch.int64)),
        "floating point",
    )


def test_query_gate_matches_sigmoid_oracle_and_preserves_inputs() -> None:
    attention_output = torch.tensor(
        [[1.0, -2.0, 3.0], [0.5, -0.25, 4.0]],
        dtype=torch.float32,
    )
    query_gate = torch.tensor(
        [[0.0, 20.0, -20.0], [-1.0, 1.0, 0.5]],
        dtype=torch.float32,
    )
    original_output = attention_output.clone()
    original_gate = query_gate.clone()
    actual = qwen35_apply_query_gate(attention_output, query_gate)
    expected = attention_output * torch.sigmoid(query_gate)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(attention_output, original_output)
    torch.testing.assert_close(query_gate, original_gate)
    torch.testing.assert_close(actual[0, 0], attention_output[0, 0] * 0.5)
    torch.testing.assert_close(
        actual[0, 1],
        attention_output[0, 1],
        rtol=0.0,
        atol=1e-6,
    )
    assert abs(actual[0, 2].item()) < 1e-7


def test_query_gate_preserves_attention_dtype() -> None:
    attention_output = torch.tensor(
        [[[0.58984375, -2.015625], [0.50390625, 3.03125]]],
        dtype=torch.bfloat16,
    )
    query_gate = torch.tensor(
        [[[-4.5625, -0.5078125], [1.015625, -1.53125]]],
        dtype=torch.bfloat16,
    )
    actual = qwen35_apply_query_gate(attention_output, query_gate)
    expected = attention_output * torch.sigmoid(query_gate)
    fp32_promoted = (
        attention_output.float() * torch.sigmoid(query_gate.float())
    ).to(torch.bfloat16)
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual.float(), expected.float())
    assert not torch.equal(expected, fp32_promoted)


def test_query_gate_rejects_broadcasting_and_invalid_dtypes() -> None:
    _expect_value_error(
        lambda: qwen35_apply_query_gate(
            torch.ones(2, 3), torch.ones(1, 3)
        ),
        "exactly match",
    )
    _expect_value_error(
        lambda: qwen35_apply_query_gate(
            torch.ones(2, 3, dtype=torch.int64), torch.ones(2, 3)
        ),
        "floating point",
    )
    _expect_value_error(
        lambda: qwen35_apply_query_gate(
            torch.ones(2, 3), torch.ones(2, 3, dtype=torch.int64)
        ),
        "floating point",
    )
    _expect_value_error(
        lambda: qwen35_apply_query_gate(
            torch.ones(2, 3, dtype=torch.float32),
            torch.ones(2, 3, dtype=torch.float64),
        ),
        "dtype",
    )


def main() -> None:
    test_zero_weight_matches_plain_rms_normalization()
    test_nonzero_offsets_use_one_plus_weight()
    test_bfloat16_preserves_dtype_with_fp32_oracle()
    test_offset_rmsnorm_rejects_invalid_contracts()
    test_query_gate_matches_sigmoid_oracle_and_preserves_inputs()
    test_query_gate_preserves_attention_dtype()
    test_query_gate_rejects_broadcasting_and_invalid_dtypes()
    print("qwen35 norm and query gate tests passed")


if __name__ == "__main__":
    main()
