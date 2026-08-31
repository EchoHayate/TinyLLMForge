from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import pytest

try:
    import torch
except ModuleNotFoundError:
    torch = ModuleType("torch")
    torch.Tensor = object
    torch.float16 = object()
    torch.bfloat16 = object()
    torch.float32 = object()
    torch.uint8 = object()
    sys.modules["torch"] = torch

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tinyvllm"
    / "layers"
    / "fused_int4_linear.py"
)
SPEC = importlib.util.spec_from_file_location(
    "tinyvllm_fused_int4_linear_contract",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)

FusedInt4Support = module.FusedInt4Support
fused_int4_linear = module.fused_int4_linear
fused_int4_support = module.fused_int4_support
warmup_fused_int4_linear = module.warmup_fused_int4_linear


@dataclass
class _FakeDevice:
    type: str
    index: int | None = 0


class _FakeTensor:
    def __init__(
        self,
        shape,
        dtype,
        device="cuda",
        *,
        contiguous=True,
    ):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = _FakeDevice(device, 0 if device == "cuda" else None)
        self.ndim = len(self.shape)
        self._contiguous = contiguous

    def is_contiguous(self):
        return self._contiguous


def _valid_fake_tensors():
    return (
        _FakeTensor((4, 1024), torch.bfloat16),
        _FakeTensor((2048, 512), torch.uint8),
        _FakeTensor((2048, 8), torch.float32),
    )


def test_module_import_does_not_eagerly_import_triton():
    assert "triton" not in module.__dict__
    assert "triton.language" not in module.__dict__


def test_support_accepts_aligned_cuda_contract():
    x, packed, scales = _valid_fake_tensors()

    result = fused_int4_support(
        x,
        packed,
        scales,
        group_size=128,
    )

    assert result == FusedInt4Support(True, None)


@pytest.mark.parametrize(
    ("mutation", "reason"),
    (
        ("cpu", "not_cuda"),
        ("device_mismatch", "not_cuda"),
        ("dtype", "unsupported_activation_dtype"),
        ("rank", "invalid_rank"),
        ("packed_shape", "packed_shape_mismatch"),
        ("scale_shape", "scale_shape_mismatch"),
        ("group", "unsupported_group_size"),
        ("noncontiguous", "noncontiguous"),
        ("m_alignment", "unsupported_alignment"),
        ("k_alignment", "unsupported_alignment"),
        ("n_alignment", "unsupported_alignment"),
    ),
)
def test_support_rejects_invalid_contract(mutation, reason):
    x, packed, scales = _valid_fake_tensors()
    group_size = 128
    if mutation == "cpu":
        x = _FakeTensor(x.shape, x.dtype, "cpu")
    elif mutation == "device_mismatch":
        packed = _FakeTensor(
            packed.shape,
            packed.dtype,
            "cuda",
        )
        packed.device = _FakeDevice("cuda", 1)
    elif mutation == "dtype":
        x = _FakeTensor(x.shape, torch.float32)
    elif mutation == "rank":
        x = _FakeTensor((1, 4, 1024), x.dtype)
    elif mutation == "packed_shape":
        packed = _FakeTensor((2048, 511), packed.dtype)
    elif mutation == "scale_shape":
        scales = _FakeTensor((2048, 7), scales.dtype)
    elif mutation == "group":
        group_size = 16
    elif mutation == "noncontiguous":
        scales = _FakeTensor(
            scales.shape,
            scales.dtype,
            contiguous=False,
        )
    elif mutation == "m_alignment":
        x = _FakeTensor((3, 1024), x.dtype)
    elif mutation == "k_alignment":
        x = _FakeTensor((4, 992), x.dtype)
        packed = _FakeTensor((2048, 496), packed.dtype)
        scales = _FakeTensor((2048, 31), scales.dtype)
        group_size = 32
    elif mutation == "n_alignment":
        packed = _FakeTensor((2047, 512), packed.dtype)
        scales = _FakeTensor((2047, 8), scales.dtype)
    else:
        raise AssertionError(mutation)

    result = fused_int4_support(
        x,
        packed,
        scales,
        group_size=group_size,
    )

    assert result == FusedInt4Support(False, reason)


def test_launch_passes_packed_weight_without_full_dequantization(monkeypatch):
    x, packed, scales = _valid_fake_tensors()
    output = _FakeTensor((4, 2048), torch.bfloat16)
    calls = []

    def fake_launch(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(module, "_launch_triton", fake_launch)

    result = fused_int4_linear(
        x,
        packed,
        scales,
        group_size=128,
        output=output,
    )

    assert result is output
    assert calls == [{
        "x": x,
        "packed_weight": packed,
        "scales": scales,
        "group_size": 128,
        "output": output,
    }]


@pytest.mark.parametrize(
    "output",
    (
        _FakeTensor((4, 2047), torch.bfloat16),
        _FakeTensor((4, 2048), torch.float16),
        _FakeTensor((4, 2048), torch.bfloat16, "cpu"),
        _FakeTensor(
            (4, 2048),
            torch.bfloat16,
            contiguous=False,
        ),
    ),
)
def test_launch_rejects_invalid_output(output):
    x, packed, scales = _valid_fake_tensors()

    with pytest.raises(ValueError, match="output"):
        fused_int4_linear(
            x,
            packed,
            scales,
            group_size=128,
            output=output,
        )


def test_launch_rejects_unsupported_inputs_without_fallback():
    x, packed, scales = _valid_fake_tensors()
    x = _FakeTensor((3, 1024), x.dtype)

    with pytest.raises(
        ValueError,
        match="unsupported_alignment",
    ):
        fused_int4_linear(
            x,
            packed,
            scales,
            group_size=128,
            output=_FakeTensor((3, 2048), torch.bfloat16),
        )


def test_warmup_compiles_every_shape_without_dequantization(monkeypatch):
    x1, packed1, scales1 = _valid_fake_tensors()
    x2 = _FakeTensor((1, 3072), torch.float16)
    packed2 = _FakeTensor((1024, 1536), torch.uint8)
    scales2 = _FakeTensor((1024, 48), torch.float32)
    out1 = _FakeTensor((4, 2048), torch.bfloat16)
    out2 = _FakeTensor((1, 1024), torch.float16)
    calls = []

    monkeypatch.setattr(
        module,
        "_launch_triton",
        lambda **kwargs: calls.append(kwargs),
    )

    warmup_fused_int4_linear((
        (x1, packed1, scales1, 128, out1),
        (x2, packed2, scales2, 64, out2),
    ))

    assert [call["packed_weight"] for call in calls] == [
        packed1,
        packed2,
    ]


def test_source_does_not_import_linear_dispatch_or_call_dequantize():
    source = module.__file__
    text = open(source, encoding="utf-8").read()

    assert "tinyvllm.layers.linear" not in text
    assert "dequantize_int4" not in text
    assert "sys.modules" not in text
