import importlib.util
import math
from pathlib import Path
import sys
import types

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]


def _load_module(module_name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


for package_name in ("tinyvllm", "tinyvllm.layers", "tinyvllm.utils"):
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / package_name.replace(".", "/"))]
    sys.modules[package_name] = package

context_module = _load_module(
    "tinyvllm.utils.context",
    "tinyvllm/utils/context.py",
)
context_state = context_module.get_context()

gated_delta = _load_module(
    "tinyvllm.layers.gated_delta",
    "tinyvllm/layers/gated_delta.py",
)
linear_attention = _load_module(
    "tinyvllm.layers.qwen35_linear_attention",
    "tinyvllm/layers/qwen35_linear_attention.py",
)
Qwen35LinearAttentionShell = linear_attention.Qwen35LinearAttentionShell


class _Linear(nn.Module):

    def __init__(
        self,
        name: str,
        events: list,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ):
        super().__init__()
        self.name = name
        self.events = events
        self.register_buffer("weight", weight)
        self.register_buffer("bias", bias)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        self.events.append(self.name)
        return (
            tensor @ self.weight.to(tensor.dtype)
            + self.bias.to(tensor.dtype)
        )


def _parameters(
    dtype: torch.dtype = torch.float32,
    stable_dtype=None,
) -> dict:
    generator = torch.Generator().manual_seed(41)
    shapes = {
        "qkv_weight": (4, 10),
        "qkv_bias": (10,),
        "z_weight": (4, 6),
        "z_bias": (6,),
        "b_weight": (4, 2),
        "b_bias": (2,),
        "a_weight": (4, 2),
        "a_bias": (2,),
        "out_weight": (6, 4),
        "out_bias": (4,),
        "conv_weight": (10, 3),
        "A_log": (2,),
        "dt_bias": (2,),
        "norm_weight": (3,),
    }
    values = {}
    for name, shape in shapes.items():
        value_dtype = (
            stable_dtype
            if stable_dtype is not None
            and name in ("A_log", "norm_weight")
            else dtype
        )
        values[name] = (
            torch.randn(*shape, generator=generator) / 3
        ).to(value_dtype)
    return values


def _manual_conv(
    projected: torch.Tensor,
    state: torch.Tensor,
    weight: torch.Tensor,
) -> tuple:
    if projected.shape[0] == 0:
        return projected.clone(), state.clone()
    combined = torch.cat(
        (state.unsqueeze(0), projected.T.unsqueeze(0)),
        dim=-1,
    ).to(weight.dtype)
    candidate = combined[:, :, -state.shape[1]:].squeeze(0)
    output = F.silu(F.conv1d(
        combined,
        weight.unsqueeze(1),
        padding=0,
        groups=weight.shape[0],
    )[:, :, -projected.shape[0]:])
    return (
        output.transpose(1, 2).squeeze(0).to(projected.dtype),
        candidate,
    )


def _manual_recurrent(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_v_k: torch.Tensor,
) -> tuple:
    key_dim = query.shape[-1]
    output_dtype = query.dtype
    query = query.float()
    query = query * torch.rsqrt(
        (query * query).sum(dim=-1, keepdim=True) + 1e-6
    )
    query = query / math.sqrt(key_dim)
    key = key.float()
    key = key * torch.rsqrt(
        (key * key).sum(dim=-1, keepdim=True) + 1e-6
    )
    state = state_v_k.float().transpose(-1, -2).clone()
    outputs = []
    for token_index in range(query.shape[0]):
        beta = torch.sigmoid(b[token_index].float())
        decay = torch.exp(
            -torch.exp(A_log.float())
            * F.softplus(a[token_index].float() + dt_bias.float())
        )
        state = state * decay[:, None, None]
        memory = torch.einsum("hk,hkv->hv", key[token_index], state)
        delta = (
            value[token_index].float() - memory
        ) * beta[:, None]
        state = state + torch.einsum(
            "hk,hv->hkv",
            key[token_index],
            delta,
        )
        outputs.append(
            torch.einsum("hk,hkv->hv", query[token_index], state)
        )
    return (
        torch.stack(outputs).to(output_dtype),
        state.transpose(-1, -2).to(state_v_k.dtype),
    )


def _manual_oracle(
    hidden_states: torch.Tensor,
    convolution_state: torch.Tensor,
    recurrent_state: torch.Tensor,
    *,
    stable_dtype=None,
) -> tuple:
    values = _parameters(hidden_states.dtype, stable_dtype)
    qkv = (
        hidden_states @ values["qkv_weight"]
        + values["qkv_bias"]
    )
    z = hidden_states @ values["z_weight"] + values["z_bias"]
    b = hidden_states @ values["b_weight"] + values["b_bias"]
    a = hidden_states @ values["a_weight"] + values["a_bias"]
    convolved, candidate_conv = _manual_conv(
        qkv,
        convolution_state,
        values["conv_weight"],
    )
    query, key, value = convolved.split((2, 2, 6), dim=-1)
    query = query.view(-1, 1, 2).repeat_interleave(2, dim=1)
    key = key.view(-1, 1, 2).repeat_interleave(2, dim=1)
    value = value.view(-1, 2, 3)
    delta_rule = (
        gated_delta.qwen35_gated_delta_recurrent
        if hidden_states.shape[0] == 1
        else gated_delta.qwen35_gated_delta_chunk
    )
    core, candidate_recurrent = delta_rule(
        query,
        key,
        value,
        a,
        b,
        values["A_log"],
        values["dt_bias"],
        recurrent_state,
    )
    core = core.reshape(-1, 3)
    z = z.reshape(-1, 3)
    normalized = core.float() * torch.rsqrt(
        core.float().pow(2).mean(dim=-1, keepdim=True) + 1e-6
    )
    gated = (
        normalized
        * values["norm_weight"].float()
        * F.silu(z.float())
    ).to(hidden_states.dtype)
    gated = gated.reshape(hidden_states.shape[0], 6)
    output = (
        gated @ values["out_weight"]
        + values["out_bias"]
    )
    return output, candidate_conv, candidate_recurrent


class _PrimitiveRecorder:

    def __init__(self, events: list):
        self.events = events
        self.original_conv = linear_attention.qwen35_causal_depthwise_conv
        self.original_recurrent = (
            linear_attention.qwen35_gated_delta_recurrent
        )
        self.original_chunk = linear_attention.qwen35_gated_delta_chunk
        self.original_norm = linear_attention.qwen35_gated_rmsnorm

    def __enter__(self):
        def conv(*args, **kwargs):
            self.events.append("causal_conv")
            return self.original_conv(*args, **kwargs)

        def recurrent(*args, **kwargs):
            self.events.append("gated_delta_recurrent")
            return self.original_recurrent(*args, **kwargs)

        def chunk(*args, **kwargs):
            self.events.append("gated_delta_chunk")
            return self.original_chunk(*args, **kwargs)

        def norm(*args, **kwargs):
            self.events.append("gated_rmsnorm")
            return self.original_norm(*args, **kwargs)

        linear_attention.qwen35_causal_depthwise_conv = conv
        linear_attention.qwen35_gated_delta_recurrent = recurrent
        linear_attention.qwen35_gated_delta_chunk = chunk
        linear_attention.qwen35_gated_rmsnorm = norm

    def __exit__(self, exc_type, exc_value, traceback):
        linear_attention.qwen35_causal_depthwise_conv = self.original_conv
        linear_attention.qwen35_gated_delta_recurrent = self.original_recurrent
        linear_attention.qwen35_gated_delta_chunk = self.original_chunk
        linear_attention.qwen35_gated_rmsnorm = self.original_norm


def _new_shell(
    events: list,
    dtype: torch.dtype = torch.float32,
    stable_dtype=None,
) -> Qwen35LinearAttentionShell:
    values = _parameters(dtype, stable_dtype)
    return Qwen35LinearAttentionShell(
        local_key_heads=1,
        local_value_heads=2,
        key_head_dim=2,
        value_head_dim=3,
        norm_eps=1e-6,
        in_proj_qkv=_Linear(
            "in_proj_qkv",
            events,
            values["qkv_weight"],
            values["qkv_bias"],
        ),
        in_proj_z=_Linear(
            "in_proj_z",
            events,
            values["z_weight"],
            values["z_bias"],
        ),
        in_proj_b=_Linear(
            "in_proj_b",
            events,
            values["b_weight"],
            values["b_bias"],
        ),
        in_proj_a=_Linear(
            "in_proj_a",
            events,
            values["a_weight"],
            values["a_bias"],
        ),
        out_proj=_Linear(
            "out_proj",
            events,
            values["out_weight"],
            values["out_bias"],
        ),
        conv_weight=values["conv_weight"],
        A_log=values["A_log"],
        dt_bias=values["dt_bias"],
        norm_weight=values["norm_weight"],
    )


def _fixture(dtype: torch.dtype = torch.float32) -> tuple:
    hidden = torch.tensor(
        [
            [1.0, -2.0, 0.5, 3.0],
            [-1.0, 0.25, 2.0, -0.5],
            [0.75, 1.25, -1.5, 0.5],
        ],
        dtype=dtype,
    )
    conv_state = (
        torch.arange(30, dtype=torch.float32).reshape(10, 3) / 20 - 0.5
    ).to(dtype)
    recurrent_state = (
        torch.arange(12, dtype=torch.float32).reshape(2, 3, 2) / 15
        - 0.25
    ).to(dtype)
    return hidden, conv_state, recurrent_state


def test_operation_order_numerical_oracle_and_state_nonmutation() -> None:
    events = []
    shell = _new_shell(events)
    hidden, conv_state, recurrent_state = _fixture()
    original_hidden = hidden.clone()
    original_conv = conv_state.clone()
    original_recurrent = recurrent_state.clone()
    with _PrimitiveRecorder(events):
        actual = shell(hidden, conv_state, recurrent_state)
    expected = _manual_oracle(hidden, conv_state, recurrent_state)
    assert events == [
        "in_proj_qkv",
        "in_proj_z",
        "in_proj_b",
        "in_proj_a",
        "causal_conv",
        "gated_delta_chunk",
        "gated_rmsnorm",
        "out_proj",
    ]
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor)
    torch.testing.assert_close(hidden, original_hidden)
    torch.testing.assert_close(conv_state, original_conv)
    torch.testing.assert_close(recurrent_state, original_recurrent)


def test_split_continuation_matches_one_shot() -> None:
    shell = _new_shell([])
    hidden, conv_state, recurrent_state = _fixture()
    full_output, full_conv, full_recurrent = shell(
        hidden,
        conv_state,
        recurrent_state,
    )
    first_output, first_conv, first_recurrent = shell(
        hidden[:1],
        conv_state,
        recurrent_state,
    )
    second_output, second_conv, second_recurrent = shell(
        hidden[1:],
        first_conv,
        first_recurrent,
    )
    torch.testing.assert_close(
        torch.cat((first_output, second_output)),
        full_output,
    )
    torch.testing.assert_close(second_conv, full_conv)
    torch.testing.assert_close(second_recurrent, full_recurrent)


def test_state_trace_matches_sequential_prefixes_and_fast_path() -> None:
    shell = _new_shell([])
    hidden, conv_state, recurrent_state = _fixture()
    original_conv = conv_state.clone()
    original_recurrent = recurrent_state.clone()
    expected_outputs = []
    expected_convolution = []
    expected_recurrent = []
    next_convolution = conv_state
    next_recurrent = recurrent_state
    for token_index in range(hidden.shape[0]):
        output, next_convolution, next_recurrent = shell(
            hidden[token_index:token_index + 1],
            next_convolution,
            next_recurrent,
        )
        expected_outputs.append(output)
        expected_convolution.append(next_convolution)
        expected_recurrent.append(next_recurrent)

    output, final_conv, final_recurrent, trace = (
        shell.forward_with_state_trace(
            hidden,
            conv_state,
            recurrent_state,
        )
    )
    fast_output, fast_conv, fast_recurrent = shell(
        hidden,
        conv_state,
        recurrent_state,
    )

    assert trace.convolution.shape == (
        hidden.shape[0],
        *conv_state.shape,
    )
    assert trace.recurrent.shape == (
        hidden.shape[0],
        *recurrent_state.shape,
    )
    torch.testing.assert_close(
        output,
        torch.cat(expected_outputs),
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        trace.convolution,
        torch.stack(expected_convolution),
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        trace.recurrent,
        torch.stack(expected_recurrent),
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(trace.convolution[-1], final_conv)
    torch.testing.assert_close(trace.recurrent[-1], final_recurrent)
    torch.testing.assert_close(fast_output, output, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(fast_conv, final_conv)
    torch.testing.assert_close(fast_recurrent, final_recurrent)
    torch.testing.assert_close(conv_state, original_conv)
    torch.testing.assert_close(recurrent_state, original_recurrent)


def test_cached_prefill_z_projection_uses_full_context_gemm_rows() -> None:
    events = []
    shell = _new_shell(events)
    hidden, conv_state, recurrent_state = _fixture()
    observed_shapes = []
    original_forward = shell.in_proj_z.forward

    def capture_shape(tensor):
        observed_shapes.append(tuple(tensor.shape))
        return original_forward(tensor)

    shell.in_proj_z.forward = capture_shape
    context_state.is_prefill = True
    context_state.max_seqlen_k = 5
    try:
        output, _, _ = shell(
            hidden[-2:],
            conv_state,
            recurrent_state,
        )
    finally:
        context_state.is_prefill = False
        context_state.max_seqlen_k = 0

    assert observed_shapes == [(5, 4)]
    assert output.shape == (2, 4)


def test_cached_prefill_all_projections_use_full_context_gemm_rows() -> None:
    shell = _new_shell([])
    hidden, conv_state, recurrent_state = _fixture()
    observed_shapes = {}

    for name in (
        "in_proj_qkv",
        "in_proj_z",
        "in_proj_b",
        "in_proj_a",
        "out_proj",
    ):
        projection = getattr(shell, name)
        original_forward = projection.forward

        def capture_shape(tensor, *, name=name, forward=original_forward):
            observed_shapes.setdefault(name, []).append(tuple(tensor.shape))
            return forward(tensor)

        projection.forward = capture_shape

    context_state.is_prefill = True
    context_state.max_seqlen_k = 5
    try:
        output, _, _ = shell(
            hidden[-2:],
            conv_state,
            recurrent_state,
        )
    finally:
        context_state.is_prefill = False
        context_state.max_seqlen_k = 0

    assert observed_shapes == {
        "in_proj_qkv": [(5, 4)],
        "in_proj_z": [(5, 4)],
        "in_proj_b": [(5, 4)],
        "in_proj_a": [(5, 4)],
        "out_proj": [(5, 6)],
    }
    assert output.shape == (2, 4)


def test_cached_prefill_uses_out_projection_prefill_path() -> None:
    shell = _new_shell([])
    hidden, conv_state, recurrent_state = _fixture()
    calls = []
    normal_forward = shell.out_proj.forward

    def prefill_forward(tensor):
        calls.append("prefill")
        return normal_forward(tensor)

    shell.out_proj.forward_prefill = prefill_forward
    context_state.is_prefill = True
    context_state.max_seqlen_k = 5
    try:
        shell(
            hidden[-2:],
            conv_state,
            recurrent_state,
        )
        context_state.is_prefill = False
        context_state.max_seqlen_k = 0
        shell(
            hidden[-2:],
            conv_state,
            recurrent_state,
        )
    finally:
        context_state.is_prefill = False
        context_state.max_seqlen_k = 0

    assert calls == ["prefill"]


def test_cached_prefill_delta_rule_uses_full_context_rows() -> None:
    shell = _new_shell([])
    hidden, conv_state, recurrent_state = _fixture()
    observed = {}
    original_chunk = linear_attention.qwen35_gated_delta_chunk

    def capture_chunk(
        query,
        key,
        value,
        a,
        b,
        A_log,
        dt_bias,
        recurrent_state,
        **kwargs,
    ):
        observed["query_shape"] = tuple(query.shape)
        observed["key_shape"] = tuple(key.shape)
        observed["value_shape"] = tuple(value.shape)
        observed["a_shape"] = tuple(a.shape)
        observed["b_shape"] = tuple(b.shape)
        observed["query_prefix"] = query[:-2].clone()
        observed["key_prefix"] = key[:-2].clone()
        observed["value_prefix"] = value[:-2].clone()
        observed["a_prefix"] = a[:-2].clone()
        observed["b_prefix"] = b[:-2].clone()
        return original_chunk(
            query,
            key,
            value,
            a,
            b,
            A_log,
            dt_bias,
            recurrent_state,
            **kwargs,
        )

    linear_attention.qwen35_gated_delta_chunk = capture_chunk
    context_state.is_prefill = True
    context_state.max_seqlen_k = 5
    try:
        output, _, _ = shell(
            hidden[-2:],
            conv_state,
            recurrent_state,
        )
    finally:
        context_state.is_prefill = False
        context_state.max_seqlen_k = 0
        linear_attention.qwen35_gated_delta_chunk = original_chunk

    assert observed["query_shape"] == (5, 2, 2)
    assert observed["key_shape"] == (5, 2, 2)
    assert observed["value_shape"] == (5, 2, 3)
    assert observed["a_shape"] == (5, 2)
    assert observed["b_shape"] == (5, 2)
    assert torch.count_nonzero(observed["query_prefix"]) == 0
    assert torch.count_nonzero(observed["key_prefix"]) == 0
    assert torch.count_nonzero(observed["value_prefix"]) == 0
    assert torch.isneginf(observed["a_prefix"]).all()
    assert torch.isneginf(observed["b_prefix"]).all()
    assert output.shape == (2, 4)


def test_bfloat16_preserves_dtype_and_matches_fp32_oracle() -> None:
    shell = _new_shell([], dtype=torch.bfloat16)
    hidden, conv_state, recurrent_state = _fixture(torch.bfloat16)
    actual = shell(hidden, conv_state, recurrent_state)
    expected = _manual_oracle(hidden, conv_state, recurrent_state)
    for actual_tensor, expected_tensor in zip(actual, expected):
        assert actual_tensor.dtype == torch.bfloat16
        torch.testing.assert_close(
            actual_tensor.float(),
            expected_tensor.float(),
            rtol=2e-2,
            atol=2e-2,
        )


def test_checkpoint_like_mixed_dtype_matches_fp32_oracle() -> None:
    shell = _new_shell(
        [],
        dtype=torch.bfloat16,
        stable_dtype=torch.float32,
    )
    hidden, conv_state, recurrent_state = _fixture(torch.bfloat16)
    original_hidden = hidden.clone()
    original_conv = conv_state.clone()
    original_recurrent = recurrent_state.clone()
    original_A_log = shell.A_log.clone()
    original_norm_weight = shell.norm_weight.clone()

    actual = shell(hidden, conv_state, recurrent_state)
    expected = _manual_oracle(
        hidden,
        conv_state,
        recurrent_state,
        stable_dtype=torch.float32,
    )

    assert shell.conv_weight.dtype == torch.bfloat16
    assert shell.dt_bias.dtype == torch.bfloat16
    assert shell.A_log.dtype == torch.float32
    assert shell.norm_weight.dtype == torch.float32
    for actual_tensor, expected_tensor in zip(actual, expected):
        assert actual_tensor.dtype == torch.bfloat16
        torch.testing.assert_close(
            actual_tensor.float(),
            expected_tensor.float(),
            rtol=2e-2,
            atol=2e-2,
        )
    torch.testing.assert_close(hidden, original_hidden)
    torch.testing.assert_close(conv_state, original_conv)
    torch.testing.assert_close(recurrent_state, original_recurrent)
    torch.testing.assert_close(shell.A_log, original_A_log)
    torch.testing.assert_close(shell.norm_weight, original_norm_weight)


def test_bfloat16_compute_with_fp32_recurrent_state_matches_oracle() -> None:
    shell = _new_shell(
        [],
        dtype=torch.bfloat16,
        stable_dtype=torch.float32,
    )
    hidden, conv_state, recurrent_state = _fixture(torch.bfloat16)
    recurrent_state = recurrent_state.float()
    actual = shell(hidden, conv_state, recurrent_state)
    expected = _manual_oracle(
        hidden,
        conv_state,
        recurrent_state,
        stable_dtype=torch.float32,
    )
    assert actual[0].dtype == torch.bfloat16
    assert actual[1].dtype == torch.bfloat16
    assert actual[2].dtype == torch.float32
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(
            actual_tensor.float(),
            expected_tensor.float(),
            rtol=2e-2,
            atol=2e-2,
        )


def test_fp32_recurrent_candidate_preserves_runtime_precision() -> None:
    shell = _new_shell(
        [],
        dtype=torch.bfloat16,
        stable_dtype=torch.float32,
    )
    hidden, conv_state, recurrent_state = _fixture(torch.bfloat16)
    recurrent_state = recurrent_state.float()
    observed = {}
    original_chunk = linear_attention.qwen35_gated_delta_chunk

    def capture_chunk(*args, **kwargs):
        core, candidate = original_chunk(*args, **kwargs)
        observed["candidate"] = candidate.clone()
        return core, candidate

    linear_attention.qwen35_gated_delta_chunk = capture_chunk
    try:
        _, _, actual_candidate = shell(
            hidden,
            conv_state,
            recurrent_state,
        )
    finally:
        linear_attention.qwen35_gated_delta_chunk = original_chunk

    assert actual_candidate.dtype == torch.float32
    torch.testing.assert_close(
        actual_candidate,
        observed["candidate"],
        rtol=0.0,
        atol=0.0,
    )


def test_checkpoint_like_mixed_dtype_split_matches_one_shot() -> None:
    shell = _new_shell(
        [],
        dtype=torch.bfloat16,
        stable_dtype=torch.float32,
    )
    hidden, conv_state, recurrent_state = _fixture(torch.bfloat16)
    full_output, full_conv, full_recurrent = shell(
        hidden,
        conv_state,
        recurrent_state,
    )
    first_output, first_conv, first_recurrent = shell(
        hidden[:1],
        conv_state,
        recurrent_state,
    )
    second_output, second_conv, second_recurrent = shell(
        hidden[1:],
        first_conv,
        first_recurrent,
    )
    for actual, expected in (
        (torch.cat((first_output, second_output)), full_output),
        (second_conv, full_conv),
        (second_recurrent, full_recurrent),
    ):
        torch.testing.assert_close(
            actual.float(),
            expected.float(),
            rtol=2e-2,
            atol=2e-2,
        )


def test_checkpoint_like_mixed_dtype_long_partition_matches_one_shot() -> None:
    shell = _new_shell(
        [],
        dtype=torch.bfloat16,
        stable_dtype=torch.float32,
    )
    hidden, conv_state, recurrent_state = _fixture(torch.bfloat16)
    hidden = hidden.repeat(272, 1)
    full_output, full_conv, full_recurrent = shell(
        hidden,
        conv_state,
        recurrent_state.float(),
    )
    first_output, first_conv, first_recurrent = shell(
        hidden[:1024],
        conv_state,
        recurrent_state.float(),
    )
    second_output, second_conv, second_recurrent = shell(
        hidden[1024:],
        first_conv,
        first_recurrent,
    )
    torch.testing.assert_close(
        torch.cat((first_output, second_output)).float(),
        full_output.float(),
        rtol=0.0,
        atol=2e-5,
    )
    torch.testing.assert_close(
        second_conv.float(),
        full_conv.float(),
        rtol=0.0,
        atol=2e-5,
    )
    torch.testing.assert_close(
        second_recurrent.float(),
        full_recurrent.float(),
        rtol=0.0,
        atol=2e-5,
    )


class _FailingOutput(nn.Module):

    def forward(self, tensor):
        raise RuntimeError("output projection failure")


def test_output_projection_failure_does_not_mutate_input_states() -> None:
    events = []
    shell = _new_shell(events)
    shell.out_proj = _FailingOutput()
    hidden, conv_state, recurrent_state = _fixture()
    original_conv = conv_state.clone()
    original_recurrent = recurrent_state.clone()
    try:
        shell(hidden, conv_state, recurrent_state)
    except RuntimeError as error:
        assert "output projection failure" in str(error)
    else:
        raise AssertionError("expected output projection failure")
    torch.testing.assert_close(conv_state, original_conv)
    torch.testing.assert_close(recurrent_state, original_recurrent)


class _ReturnModule(nn.Module):

    def __init__(self, output):
        super().__init__()
        self.output = output

    def forward(self, *args):
        return self.output


def _expect_value_error(function, message: str) -> None:
    try:
        function()
    except ValueError as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(f"expected ValueError containing {message!r}")


def test_constructor_and_input_state_contracts_fail_closed() -> None:
    values = _parameters()
    valid = dict(
        local_key_heads=1,
        local_value_heads=2,
        key_head_dim=2,
        value_head_dim=3,
        norm_eps=1e-6,
        in_proj_qkv=_ReturnModule(torch.ones(2, 10)),
        in_proj_z=_ReturnModule(torch.ones(2, 6)),
        in_proj_b=_ReturnModule(torch.ones(2, 2)),
        in_proj_a=_ReturnModule(torch.ones(2, 2)),
        out_proj=_ReturnModule(torch.ones(2, 4)),
        conv_weight=values["conv_weight"],
        A_log=values["A_log"],
        dt_bias=values["dt_bias"],
        norm_weight=values["norm_weight"],
    )
    for name in (
        "local_key_heads",
        "local_value_heads",
        "key_head_dim",
        "value_head_dim",
    ):
        for value in (True, 0, -1, 1.5):
            kwargs = dict(valid)
            kwargs[name] = value
            _expect_value_error(
                lambda kwargs=kwargs: Qwen35LinearAttentionShell(**kwargs),
                name,
            )
    kwargs = dict(valid)
    kwargs["local_key_heads"] = 2
    kwargs["local_value_heads"] = 3
    _expect_value_error(
        lambda: Qwen35LinearAttentionShell(**kwargs),
        "divisible",
    )
    for eps in (True, 0.0, -1.0, float("inf"), float("nan")):
        kwargs = dict(valid)
        kwargs["norm_eps"] = eps
        _expect_value_error(
            lambda kwargs=kwargs: Qwen35LinearAttentionShell(**kwargs),
            "norm_eps",
        )
    for name, value, message in (
        ("conv_weight", torch.ones(9, 3), "conv_weight"),
        ("A_log", torch.ones(1), "A_log"),
        ("dt_bias", torch.ones(1), "dt_bias"),
        ("norm_weight", torch.ones(2), "norm_weight"),
        (
            "norm_weight",
            torch.ones(3, dtype=torch.int64),
            "floating point",
        ),
        (
            "dt_bias",
            torch.ones(2, dtype=torch.float64),
            "compute parameter dtype",
        ),
        (
            "norm_weight",
            torch.ones(3, device="meta"),
            "parameter device",
        ),
    ):
        kwargs = dict(valid)
        kwargs[name] = value
        _expect_value_error(
            lambda kwargs=kwargs: Qwen35LinearAttentionShell(**kwargs),
            message,
        )

    shell = _new_shell([])
    hidden, conv_state, recurrent_state = _fixture()
    cases = (
        (
            torch.ones(2, 2, 2),
            conv_state,
            recurrent_state,
            "rank two",
        ),
        (
            torch.ones(2, 4, dtype=torch.int64),
            conv_state,
            recurrent_state,
            "floating point",
        ),
        (
            hidden,
            torch.ones(9, 3),
            recurrent_state,
            "convolution_state shape",
        ),
        (
            hidden,
            conv_state,
            torch.ones(2, 2, 3),
            "recurrent_state shape",
        ),
        (
            hidden,
            conv_state.to(torch.float64),
            recurrent_state,
            "convolution_state dtype",
        ),
        (
            hidden,
            conv_state,
            recurrent_state.to("meta"),
            "recurrent_state device",
        ),
    )
    for hidden_case, conv_case, recurrent_case, message in cases:
        _expect_value_error(
            lambda hidden_case=hidden_case, conv_case=conv_case,
            recurrent_case=recurrent_case: shell(
                hidden_case,
                conv_case,
                recurrent_case,
            ),
            message,
        )

    for name, value, message in (
        (
            "conv_weight",
            shell.conv_weight.to(torch.float64),
            "conv_weight dtype",
        ),
        (
            "dt_bias",
            shell.dt_bias.to(torch.float64),
            "dt_bias dtype",
        ),
        (
            "A_log",
            shell.A_log.to("meta"),
            "A_log device",
        ),
        (
            "norm_weight",
            shell.norm_weight.to("meta"),
            "norm_weight device",
        ),
    ):
        changed_shell = _new_shell([])
        setattr(changed_shell, name, value)
        _expect_value_error(
            lambda changed_shell=changed_shell: changed_shell(
                hidden,
                conv_state,
                recurrent_state,
            ),
            message,
        )


def test_projection_primitive_and_output_boundaries_fail_closed() -> None:
    hidden, conv_state, recurrent_state = _fixture()
    projection_cases = (
        ("in_proj_qkv", torch.ones(3, 9), "in_proj_qkv feature"),
        ("in_proj_z", torch.ones(2, 6), "in_proj_z token"),
        (
            "in_proj_b",
            torch.ones(3, 2, dtype=torch.float64),
            "in_proj_b dtype",
        ),
        (
            "in_proj_a",
            torch.ones(3, 2, device="meta"),
            "in_proj_a device",
        ),
        (
            "in_proj_qkv",
            torch.ones(3, 10, dtype=torch.int64),
            "floating point",
        ),
    )
    for name, output, message in projection_cases:
        shell = _new_shell([])
        setattr(shell, name, _ReturnModule(output))
        _expect_value_error(
            lambda shell=shell: shell(hidden, conv_state, recurrent_state),
            message,
        )

    shell = _new_shell([])
    original_conv = linear_attention.qwen35_causal_depthwise_conv
    try:
        linear_attention.qwen35_causal_depthwise_conv = (
            lambda *args, **kwargs: (
                torch.ones(3, 9),
                torch.ones_like(conv_state),
            )
        )
        _expect_value_error(
            lambda: shell(hidden, conv_state, recurrent_state),
            "causal convolution feature",
        )
    finally:
        linear_attention.qwen35_causal_depthwise_conv = original_conv

    shell = _new_shell([])
    original_recurrent = linear_attention.qwen35_gated_delta_recurrent
    original_chunk = linear_attention.qwen35_gated_delta_chunk
    try:
        invalid_delta = lambda *args, **kwargs: (
            torch.ones(3, 2, 2),
            torch.ones_like(recurrent_state),
        )
        linear_attention.qwen35_gated_delta_recurrent = invalid_delta
        linear_attention.qwen35_gated_delta_chunk = invalid_delta
        _expect_value_error(
            lambda: shell(hidden, conv_state, recurrent_state),
            "gated-delta output shape",
        )
    finally:
        linear_attention.qwen35_gated_delta_recurrent = original_recurrent
        linear_attention.qwen35_gated_delta_chunk = original_chunk

    for output, message in (
        (torch.ones(3, 2, 2), "out_proj output must be rank two"),
        (torch.ones(2, 4), "out_proj token"),
        (
            torch.ones(3, 4, dtype=torch.float64),
            "out_proj dtype",
        ),
        (
            torch.ones(3, 4, dtype=torch.int64),
            "floating point",
        ),
        (
            torch.ones(3, 4, device="meta"),
            "out_proj device",
        ),
        ("not a tensor", "out_proj output must be a tensor"),
    ):
        shell = _new_shell([])
        shell.out_proj = _ReturnModule(output)
        _expect_value_error(
            lambda shell=shell: shell(hidden, conv_state, recurrent_state),
            message,
        )


def main() -> None:
    test_operation_order_numerical_oracle_and_state_nonmutation()
    test_split_continuation_matches_one_shot()
    test_bfloat16_preserves_dtype_and_matches_fp32_oracle()
    test_checkpoint_like_mixed_dtype_matches_fp32_oracle()
    test_bfloat16_compute_with_fp32_recurrent_state_matches_oracle()
    test_fp32_recurrent_candidate_preserves_runtime_precision()
    test_checkpoint_like_mixed_dtype_split_matches_one_shot()
    test_output_projection_failure_does_not_mutate_input_states()
    test_constructor_and_input_state_contracts_fail_closed()
    test_projection_primitive_and_output_boundaries_fail_closed()
    print("qwen35 linear attention shell tests passed")


if __name__ == "__main__":
    main()
