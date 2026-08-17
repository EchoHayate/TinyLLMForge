from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "tinyvllm/layers/gated_delta.py"
    spec = importlib.util.spec_from_file_location(
        "qwen35_gated_delta_prefix_trace_target",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gated_delta = _load_module()
qwen35_causal_depthwise_conv = (
    gated_delta.qwen35_causal_depthwise_conv
)
qwen35_causal_depthwise_conv_prefix_trace = (
    gated_delta.qwen35_causal_depthwise_conv_prefix_trace
)
qwen35_gated_delta_prefix_trace = (
    gated_delta.qwen35_gated_delta_prefix_trace
)
qwen35_gated_delta_recurrent = (
    gated_delta.qwen35_gated_delta_recurrent
)


def _recurrent_inputs(token_count, dtype=torch.float32):
    generator = torch.Generator().manual_seed(20260813 + token_count)
    query = torch.randn(
        token_count,
        2,
        3,
        generator=generator,
        dtype=dtype,
    )
    key = torch.randn(
        token_count,
        2,
        3,
        generator=generator,
        dtype=dtype,
    )
    value = torch.randn(
        token_count,
        2,
        4,
        generator=generator,
        dtype=dtype,
    )
    a = torch.randn(
        token_count,
        2,
        generator=generator,
        dtype=dtype,
    )
    b = torch.randn(
        token_count,
        2,
        generator=generator,
        dtype=dtype,
    )
    A_log = torch.randn(
        2,
        generator=generator,
        dtype=torch.float32,
    )
    dt_bias = torch.randn(
        2,
        generator=generator,
        dtype=dtype,
    )
    initial_state = torch.randn(
        2,
        4,
        3,
        generator=generator,
        dtype=dtype,
    )
    return (
        query,
        key,
        value,
        a,
        b,
        A_log,
        dt_bias,
        initial_state,
    )


@pytest.mark.parametrize("token_count", (2, 4))
def test_recurrent_prefix_trace_matches_one_token_oracle(token_count):
    inputs = _recurrent_inputs(token_count)
    initial = inputs[-1].clone()
    expected_outputs = []
    expected_states = []
    state = initial
    for index in range(token_count):
        output, state = qwen35_gated_delta_recurrent(
            inputs[0][index:index + 1],
            inputs[1][index:index + 1],
            inputs[2][index:index + 1],
            inputs[3][index:index + 1],
            inputs[4][index:index + 1],
            inputs[5],
            inputs[6],
            state,
        )
        expected_outputs.append(output)
        expected_states.append(state)

    output, states = qwen35_gated_delta_prefix_trace(*inputs)

    assert states.shape == (token_count, 2, 4, 3)
    torch.testing.assert_close(
        output,
        torch.cat(expected_outputs),
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        states,
        torch.stack(expected_states),
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(inputs[-1], initial)


def test_recurrent_prefix_trace_preserves_dtype_and_device():
    inputs = _recurrent_inputs(2, dtype=torch.bfloat16)

    output, states = qwen35_gated_delta_prefix_trace(*inputs)

    assert output.dtype == torch.bfloat16
    assert states.dtype == torch.bfloat16
    assert output.device == inputs[0].device
    assert states.device == inputs[-1].device


@pytest.mark.parametrize("token_count", (2, 4))
def test_convolution_prefix_trace_matches_one_token_oracle(
    token_count,
):
    generator = torch.Generator().manual_seed(
        20260820 + token_count
    )
    projected = torch.randn(
        token_count,
        3,
        generator=generator,
    )
    initial_state = torch.randn(
        3,
        4,
        generator=generator,
    )
    weight = torch.randn(
        3,
        4,
        generator=generator,
    )
    original_state = initial_state.clone()
    expected_outputs = []
    expected_states = []
    state = initial_state
    for index in range(token_count):
        output, state = qwen35_causal_depthwise_conv(
            projected[index:index + 1],
            state,
            weight,
        )
        expected_outputs.append(output)
        expected_states.append(state)

    output, states = qwen35_causal_depthwise_conv_prefix_trace(
        projected,
        initial_state,
        weight,
    )

    assert states.shape == (token_count, 3, 4)
    torch.testing.assert_close(
        output,
        torch.cat(expected_outputs),
    )
    torch.testing.assert_close(
        states,
        torch.stack(expected_states),
    )
    torch.testing.assert_close(initial_state, original_state)


def test_prefix_traces_reject_empty_input():
    recurrent_inputs = _recurrent_inputs(0)
    with pytest.raises(
        ValueError,
        match="at least one token",
    ):
        qwen35_gated_delta_prefix_trace(*recurrent_inputs)

    with pytest.raises(
        ValueError,
        match="at least one token",
    ):
        qwen35_causal_depthwise_conv_prefix_trace(
            torch.empty(0, 3),
            torch.zeros(3, 4),
            torch.ones(3, 4),
        )
