import importlib.util
from pathlib import Path
import sys
import types
from unittest.mock import patch

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]


def _load_module(module_name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


for package_name in ("tinyvllm", "tinyvllm.layers"):
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / package_name.replace(".", "/"))]
    sys.modules[package_name] = package

_load_module(
    "tinyvllm.layers.qwen35_primitives",
    "tinyvllm/layers/qwen35_primitives.py",
)
full_attention = _load_module(
    "tinyvllm.layers.qwen35_full_attention",
    "tinyvllm/layers/qwen35_full_attention.py",
)
Qwen35FullAttentionShell = full_attention.Qwen35FullAttentionShell


class _LinearProjection(nn.Module):

    def __init__(
        self,
        name: str,
        events: list[str],
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


class _AffineNorm(nn.Module):

    def __init__(
        self,
        name: str,
        events: list[str],
        scale: torch.Tensor,
        bias: torch.Tensor,
    ):
        super().__init__()
        self.name = name
        self.events = events
        self.register_buffer("scale", scale)
        self.register_buffer("bias", bias)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        self.events.append(self.name)
        return (
            tensor * self.scale.to(tensor.dtype)
            + self.bias.to(tensor.dtype)
        )


class _ObservableRotary(nn.Module):

    def __init__(self, events: list[str], head_dim: int):
        super().__init__()
        self.events = events
        self.head_dim = head_dim

    def forward(
        self,
        position_ids: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.events.append("rotary")
        if position_ids.ndim == 2:
            positions = position_ids[0]
        else:
            positions = position_ids
        phase = positions.to(query.dtype).unsqueeze(-1)
        query_phase = phase.expand(-1, query.shape[1])
        key_phase = phase.expand(-1, key.shape[1])
        return query + query_phase, key - key_phase


class _ObservableAttention(nn.Module):

    def __init__(
        self,
        events: list[str],
        head_dim: int,
        query_heads: int,
        kv_heads: int,
    ):
        super().__init__()
        self.events = events
        self.head_dim = head_dim
        self.query_heads = query_heads
        self.kv_heads = kv_heads

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        self.events.append("attention_backend")
        tokens = query.shape[0]
        repeats = self.query_heads // self.kv_heads
        key_by_head = key.view(tokens, self.kv_heads, self.head_dim)
        value_by_head = value.view(tokens, self.kv_heads, self.head_dim)
        expanded_key = key_by_head.repeat_interleave(repeats, dim=1)
        expanded_value = value_by_head.repeat_interleave(repeats, dim=1)
        return (
            query.view(tokens, self.query_heads, self.head_dim)
            + 2 * expanded_key
            - expanded_value
        ).reshape(tokens, -1)


class _OutputProjection(nn.Module):

    def __init__(
        self,
        events: list[str],
        weight: torch.Tensor,
        bias: torch.Tensor,
    ):
        super().__init__()
        self.events = events
        self.register_buffer("weight", weight)
        self.register_buffer("bias", bias)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        self.events.append("output_projection")
        return (
            tensor @ self.weight.to(tensor.dtype)
            + self.bias.to(tensor.dtype)
        )


def _weights() -> dict[str, torch.Tensor]:
    return {
        "q_weight": torch.tensor(
            [
                [1.0, 0.0, -0.5, 0.25, 0.5, -1.0, 0.0, 0.75],
                [0.0, 1.0, 0.25, -0.5, -0.75, 0.5, 1.0, 0.0],
                [0.5, -0.25, 1.0, 0.0, 0.25, 0.0, -0.5, 1.0],
                [-0.5, 0.5, 0.0, 1.0, 1.0, 0.25, 0.5, -0.75],
            ]
        ),
        "q_bias": torch.tensor(
            [0.1, -0.2, 0.0, 0.5, -0.25, 0.75, 0.3, -0.4]
        ),
        "k_weight": torch.tensor(
            [
                [0.5, -0.25],
                [1.0, 0.5],
                [-0.5, 1.0],
                [0.25, -0.75],
            ]
        ),
        "k_bias": torch.tensor([0.2, -0.1]),
        "v_weight": torch.tensor(
            [
                [1.0, 0.5],
                [-0.5, 0.25],
                [0.75, -1.0],
                [0.0, 0.5],
            ]
        ),
        "v_bias": torch.tensor([-0.3, 0.4]),
        "q_scale": torch.tensor([[[1.5, -0.5]]]),
        "q_norm_bias": torch.tensor([[[0.25, 0.75]]]),
        "k_scale": torch.tensor([[[0.5, 2.0]]]),
        "k_norm_bias": torch.tensor([[[-0.5, 0.25]]]),
        "o_weight": torch.tensor(
            [
                [0.5, -1.0, 0.25],
                [1.0, 0.0, -0.5],
                [-0.25, 0.75, 1.0],
                [0.5, 0.25, -0.75],
            ]
        ),
        "o_bias": torch.tensor([0.1, -0.2, 0.3]),
    }


def _new_shell(
    events: list[str],
    dtype: torch.dtype = torch.float32,
) -> Qwen35FullAttentionShell:
    weights = {
        name: tensor.to(dtype)
        for name, tensor in _weights().items()
    }
    return Qwen35FullAttentionShell(
        head_dim=2,
        local_query_heads=2,
        local_kv_heads=1,
        q_projection=_LinearProjection(
            "q_projection",
            events,
            weights["q_weight"],
            weights["q_bias"],
        ),
        k_projection=_LinearProjection(
            "k_projection",
            events,
            weights["k_weight"],
            weights["k_bias"],
        ),
        v_projection=_LinearProjection(
            "v_projection",
            events,
            weights["v_weight"],
            weights["v_bias"],
        ),
        q_norm=_AffineNorm(
            "q_norm",
            events,
            weights["q_scale"],
            weights["q_norm_bias"],
        ),
        k_norm=_AffineNorm(
            "k_norm",
            events,
            weights["k_scale"],
            weights["k_norm_bias"],
        ),
        rotary=_ObservableRotary(events, head_dim=2),
        attention_backend=_ObservableAttention(
            events,
            head_dim=2,
            query_heads=2,
            kv_heads=1,
        ),
        output_projection=_OutputProjection(
            events,
            weights["o_weight"],
            weights["o_bias"],
        ),
    )


def _manual_oracle(
    position_ids: torch.Tensor,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    weights = _weights()
    dtype = hidden_states.dtype
    paired = (
        hidden_states
        @ weights["q_weight"].to(dtype)
        + weights["q_bias"].to(dtype)
    ).view(hidden_states.shape[0], 2, 4)
    query = paired[..., :2]
    gate = paired[..., 2:].reshape(hidden_states.shape[0], 4)
    key = (
        hidden_states
        @ weights["k_weight"].to(dtype)
        + weights["k_bias"].to(dtype)
    ).view(hidden_states.shape[0], 1, 2)
    value = (
        hidden_states
        @ weights["v_weight"].to(dtype)
        + weights["v_bias"].to(dtype)
    )
    query = (
        query * weights["q_scale"].to(dtype)
        + weights["q_norm_bias"].to(dtype)
    )
    key = (
        key * weights["k_scale"].to(dtype)
        + weights["k_norm_bias"].to(dtype)
    )
    positions = position_ids[0] if position_ids.ndim == 2 else position_ids
    phase = positions.to(dtype).unsqueeze(-1)
    rotated_query = query.reshape(hidden_states.shape[0], 4)
    rotated_query = rotated_query + phase.expand(-1, 4)
    rotated_key = (
        key.reshape(hidden_states.shape[0], 2)
        - phase.expand(-1, 2)
    )
    expanded_key = rotated_key.view(-1, 1, 2).repeat_interleave(2, dim=1)
    expanded_value = value.view(-1, 1, 2).repeat_interleave(2, dim=1)
    attention = (
        rotated_query.view(-1, 2, 2)
        + 2 * expanded_key
        - expanded_value
    ).reshape(hidden_states.shape[0], 4)
    gated = attention * torch.sigmoid(gate)
    return (
        gated @ weights["o_weight"].to(dtype)
        + weights["o_bias"].to(dtype)
    )


def test_operation_order_and_numerical_oracle() -> None:
    events = []
    shell = _new_shell(events)
    position_ids = torch.tensor(
        [[0, 2], [4, 6], [8, 10]],
        dtype=torch.int64,
    )
    hidden_states = torch.tensor(
        [[1.0, -2.0, 0.5, 3.0], [-1.0, 0.25, 2.0, -0.5]]
    )
    position_original = position_ids.clone()
    hidden_original = hidden_states.clone()
    actual = shell(position_ids, hidden_states)
    expected = _manual_oracle(position_ids, hidden_states)
    assert events == [
        "q_projection",
        "k_projection",
        "v_projection",
        "q_norm",
        "k_norm",
        "rotary",
        "attention_backend",
        "output_projection",
    ]
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(position_ids, position_original)
    torch.testing.assert_close(hidden_states, hidden_original)


def test_bfloat16_preserves_dtype_and_matches_native_math_oracle() -> None:
    events = []
    shell = _new_shell(events, dtype=torch.bfloat16)
    position_ids = torch.tensor([1, 3], dtype=torch.int64)
    hidden_states = torch.tensor(
        [[1.0, -2.0, 0.5, 3.0], [-1.0, 0.25, 2.0, -0.5]],
        dtype=torch.bfloat16,
    )
    actual = shell(position_ids, hidden_states)
    expected = _manual_oracle(position_ids, hidden_states)
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual.float(), expected.float())


def test_cached_prefill_transforms_use_full_context_rows() -> None:
    shell = _new_shell([])
    shell.attention_backend.forward = lambda query, key, value: query
    position_ids = torch.tensor([3, 4], dtype=torch.int64)
    hidden_states = torch.tensor(
        [[1.0, -2.0, 0.5, 3.0], [-1.0, 0.25, 2.0, -0.5]]
    )
    observed_shapes = {}

    for name in (
        "q_projection",
        "k_projection",
        "v_projection",
        "q_norm",
        "k_norm",
        "output_projection",
    ):
        transform = getattr(shell, name)
        original_forward = transform.forward

        def capture_shape(tensor, *, name=name, forward=original_forward):
            observed_shapes.setdefault(name, []).append(tuple(tensor.shape))
            return forward(tensor)

        transform.forward = capture_shape

    context = types.SimpleNamespace(
        is_prefill=True,
        max_seqlen_k=5,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
    )
    original_get_context = getattr(full_attention, "get_context", None)
    full_attention.get_context = lambda: context
    try:
        output = shell(position_ids, hidden_states)
    finally:
        if original_get_context is None:
            del full_attention.get_context
        else:
            full_attention.get_context = original_get_context

    assert observed_shapes == {
        "q_projection": [(5, 4)],
        "k_projection": [(5, 4)],
        "v_projection": [(5, 4)],
        "q_norm": [(5, 2, 2)],
        "k_norm": [(5, 1, 2)],
        "output_projection": [(5, 4)],
    }
    assert output.shape == (2, 3)


def test_cached_prefill_uses_output_projection_prefill_path() -> None:
    shell = _new_shell([])
    shell.attention_backend.forward = lambda query, key, value: query
    position_ids = torch.tensor([3, 4], dtype=torch.int64)
    hidden_states = torch.tensor(
        [[1.0, -2.0, 0.5, 3.0], [-1.0, 0.25, 2.0, -0.5]]
    )
    calls = []
    normal_forward = shell.output_projection.forward

    def prefill_forward(tensor):
        calls.append("prefill")
        return normal_forward(tensor)

    shell.output_projection.forward_prefill = prefill_forward
    context = types.SimpleNamespace(
        is_prefill=True,
        max_seqlen_k=5,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
    )
    original_get_context = getattr(full_attention, "get_context", None)
    full_attention.get_context = lambda: context
    try:
        shell(position_ids, hidden_states)
        context.is_prefill = False
        shell(position_ids, hidden_states)
    finally:
        if original_get_context is None:
            del full_attention.get_context
        else:
            full_attention.get_context = original_get_context

    assert calls == ["prefill"]


class _ReturnModule(nn.Module):

    def __init__(self, output):
        super().__init__()
        self.output = output

    def forward(self, *args):
        return self.output


class _IdentityModule(nn.Module):

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor


class _IdentityRotary(nn.Module):

    def forward(self, position_ids, query, key):
        return query, key


class _FirstArgument(nn.Module):

    def forward(self, query, key, value):
        return query


class _QuantizedCacheAttention(_FirstArgument):

    def __init__(self):
        super().__init__()
        self.k_cache = torch.zeros(1, 2, 1, 1, dtype=torch.int8)
        self.v_cache = torch.zeros_like(self.k_cache)
        self.kv_quant_bits = 8
        self.called = False

    def forward(self, query, key, value):
        self.called = True
        return super().forward(query, key, value)


class _RoutingObservableAttention(_FirstArgument):

    def __init__(self):
        super().__init__()
        self.k_cache = torch.zeros(1, 2, 1, 2)
        self.v_cache = torch.zeros_like(self.k_cache)
        self.kv_quant_bits = 0
        self.called = False

    def forward(self, query, key, value):
        self.called = True
        return super().forward(query, key, value)


class _AttentionResult(nn.Module):

    def __init__(self, output: torch.Tensor):
        super().__init__()
        self.output = output

    def forward(self, query, key, value):
        return self.output.to(device=query.device, dtype=query.dtype)


def _boundary_shell(**overrides) -> Qwen35FullAttentionShell:
    modules = {
        "q_projection": _ReturnModule(torch.ones(2, 8)),
        "k_projection": _ReturnModule(torch.ones(2, 2)),
        "v_projection": _ReturnModule(torch.ones(2, 2)),
        "q_norm": _IdentityModule(),
        "k_norm": _IdentityModule(),
        "rotary": _IdentityRotary(),
        "attention_backend": _FirstArgument(),
        "output_projection": _IdentityModule(),
    }
    modules.update(overrides)
    return Qwen35FullAttentionShell(
        head_dim=2,
        local_query_heads=2,
        local_kv_heads=1,
        **modules,
    )


def test_zero_and_nonzero_query_gates_apply_before_output_projection() -> None:
    attention = torch.tensor(
        [[2.0, -4.0, 6.0, -8.0], [1.0, 3.0, -5.0, -7.0]]
    )
    query = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
    )

    def paired_with_gate(gate: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (
                query[:, :2],
                gate[:, :2],
                query[:, 2:],
                gate[:, 2:],
            ),
            dim=-1,
        )

    zero_gate = torch.zeros_like(attention)
    zero_shell = _boundary_shell(
        q_projection=_ReturnModule(paired_with_gate(zero_gate)),
        attention_backend=_AttentionResult(attention),
    )
    zero_output = zero_shell(torch.tensor([0, 1]), torch.ones(2, 4))
    torch.testing.assert_close(zero_output, attention * 0.5)

    nonzero_gate = torch.tensor(
        [[20.0, -20.0, 1.0, -1.0], [-2.0, 2.0, 0.5, -0.5]]
    )
    nonzero_shell = _boundary_shell(
        q_projection=_ReturnModule(paired_with_gate(nonzero_gate)),
        attention_backend=_AttentionResult(attention),
    )
    nonzero_output = nonzero_shell(
        torch.tensor([0, 1]),
        torch.ones(2, 4),
    )
    expected = attention * torch.sigmoid(nonzero_gate)
    torch.testing.assert_close(nonzero_output, expected)


def test_cached_prefill_quantized_kv_uses_original_backend() -> None:
    backend = _QuantizedCacheAttention()
    shell = _boundary_shell(attention_backend=backend)
    context = full_attention.get_context()
    previous = vars(context).copy()
    try:
        context.is_prefill = True
        context.cu_seqlens_q = torch.tensor([0, 2], dtype=torch.int32)
        context.cu_seqlens_k = torch.tensor([0, 3], dtype=torch.int32)
        context.block_tables = torch.tensor([[0, 0]], dtype=torch.int32)
        context.slot_mapping = torch.tensor([0, 1], dtype=torch.int32)
        shell(torch.tensor([0, 1]), torch.ones(2, 4))
    finally:
        vars(context).clear()
        vars(context).update(previous)

    assert backend.called is True


def test_uncached_prefill_uses_dense_eager_attention() -> None:
    backend = _RoutingObservableAttention()
    shell = _boundary_shell(attention_backend=backend)
    context = full_attention.get_context()
    previous = vars(context).copy()
    eager_output = torch.tensor(
        [[2.0, 4.0, 6.0, 8.0], [1.0, 3.0, 5.0, 7.0]]
    )
    try:
        context.is_prefill = True
        context.cu_seqlens_q = torch.tensor([0, 2], dtype=torch.int32)
        context.cu_seqlens_k = torch.tensor([0, 2], dtype=torch.int32)
        context.block_tables = None
        with patch.object(
            full_attention,
            "qwen35_prefill_eager_attention",
            return_value=eager_output,
            create=True,
        ) as eager:
            actual = shell(torch.tensor([0, 1]), torch.ones(2, 4))
    finally:
        vars(context).clear()
        vars(context).update(previous)

    eager.assert_called_once()
    assert eager.call_args.kwargs["key_cache"] is backend.k_cache
    assert eager.call_args.kwargs["value_cache"] is backend.v_cache
    assert backend.called is False
    torch.testing.assert_close(
        actual,
        eager_output * torch.sigmoid(torch.ones_like(eager_output)),
    )


def test_cached_decode_uses_qwen35_bfloat16_eager_attention() -> None:
    backend = _RoutingObservableAttention()
    shell = _boundary_shell(attention_backend=backend)
    context = full_attention.get_context()
    previous = vars(context).copy()
    eager_output = torch.tensor(
        [[2.0, 4.0, 6.0, 8.0], [1.0, 3.0, 5.0, 7.0]]
    )
    try:
        context.is_prefill = False
        context.mode = "decode"
        context.block_tables = torch.tensor(
            [[0], [0]],
            dtype=torch.int32,
        )
        context.context_lens = torch.tensor(
            [1, 1],
            dtype=torch.int32,
        )
        context.slot_mapping = torch.tensor(
            [0, 1],
            dtype=torch.int32,
        )
        with patch.object(
            full_attention,
            "qwen35_cached_decode_eager_attention",
            return_value=eager_output,
            create=True,
        ) as eager:
            actual = shell(torch.tensor([0, 1]), torch.ones(2, 4))
    finally:
        vars(context).clear()
        vars(context).update(previous)

    eager.assert_called_once()
    assert eager.call_args.args[3] is backend.k_cache
    assert eager.call_args.args[4] is backend.v_cache
    assert backend.called is False
    torch.testing.assert_close(
        actual,
        eager_output * torch.sigmoid(torch.ones_like(eager_output)),
    )


def test_noncontiguous_projection_outputs_follow_shape_contract() -> None:
    query_gate = torch.arange(16, dtype=torch.float32).reshape(8, 2).t()
    key = torch.arange(4, dtype=torch.float32).reshape(2, 2).t()
    value = torch.arange(4, dtype=torch.float32).reshape(2, 2).t()
    assert not query_gate.is_contiguous()
    assert not key.is_contiguous()
    assert not value.is_contiguous()
    shell = _boundary_shell(
        q_projection=_ReturnModule(query_gate),
        k_projection=_ReturnModule(key),
        v_projection=_ReturnModule(value),
    )
    actual = shell(torch.tensor([0, 1]), torch.ones(2, 4))
    paired = query_gate.reshape(2, 2, 4)
    expected_query = paired[..., :2].reshape(2, 4)
    expected_gate = paired[..., 2:].reshape(2, 4)
    expected = expected_query * torch.sigmoid(expected_gate)
    torch.testing.assert_close(actual, expected)


def _expect_value_error(function, message: str) -> None:
    try:
        function()
    except ValueError as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(f"expected ValueError containing {message!r}")


def test_constructor_and_input_fail_closed() -> None:
    valid = dict(
        head_dim=2,
        local_query_heads=2,
        local_kv_heads=1,
        q_projection=_IdentityModule(),
        k_projection=_IdentityModule(),
        v_projection=_IdentityModule(),
        q_norm=_IdentityModule(),
        k_norm=_IdentityModule(),
        rotary=_IdentityRotary(),
        attention_backend=_FirstArgument(),
        output_projection=_IdentityModule(),
    )
    for name in ("head_dim", "local_query_heads", "local_kv_heads"):
        for value in (True, 0, -1, 2.5):
            kwargs = dict(valid)
            kwargs[name] = value
            _expect_value_error(
                lambda kwargs=kwargs: Qwen35FullAttentionShell(**kwargs),
                name,
            )
    shell = _boundary_shell()
    _expect_value_error(
        lambda: shell(
            torch.tensor([0, 1]),
            torch.ones(2, 2, 2),
        ),
        "rank two",
    )
    _expect_value_error(
        lambda: shell(
            torch.tensor([0, 1]),
            torch.ones(2, 4, dtype=torch.int64),
        ),
        "floating point",
    )


def test_projection_boundaries_fail_closed() -> None:
    hidden = torch.ones(2, 4)
    positions = torch.tensor([0, 1])
    cases = (
        (
            "q_projection",
            torch.ones(2, 2, 4),
            "q_projection output must be rank two",
        ),
        ("q_projection", torch.ones(3, 8), "q_projection token"),
        ("q_projection", torch.ones(2, 7), "q_projection feature"),
        (
            "q_projection",
            torch.ones(2, 8, dtype=torch.float64),
            "q_projection dtype",
        ),
        (
            "q_projection",
            torch.ones(2, 8, dtype=torch.int64),
            "q_projection output must use a floating point dtype",
        ),
        ("k_projection", torch.ones(2, 1, 2), "k_projection output must be rank two"),
        ("k_projection", torch.ones(3, 2), "k_projection token"),
        ("k_projection", torch.ones(2, 3), "k_projection feature"),
        (
            "k_projection",
            torch.ones(2, 2, dtype=torch.float64),
            "k_projection dtype",
        ),
        ("v_projection", torch.ones(1, 2), "v_projection token"),
        ("v_projection", torch.ones(2, 2, 1), "v_projection output must be rank two"),
        ("v_projection", torch.ones(2, 3), "v_projection feature"),
        (
            "v_projection",
            torch.ones(2, 2, dtype=torch.float64),
            "v_projection dtype",
        ),
    )
    for name, output, message in cases:
        shell = _boundary_shell(**{name: _ReturnModule(output)})
        _expect_value_error(
            lambda shell=shell: shell(positions, hidden),
            message,
        )
    for name, width in (
        ("q_projection", 8),
        ("k_projection", 2),
        ("v_projection", 2),
    ):
        shell = _boundary_shell(
            **{
                name: _ReturnModule(
                    torch.ones(2, width, device="meta")
                )
            }
        )
        _expect_value_error(
            lambda shell=shell: shell(positions, hidden),
            f"{name} device",
        )


def test_norm_rotary_attention_and_output_boundaries_fail_closed() -> None:
    hidden = torch.ones(2, 4)
    positions = torch.tensor([0, 1])
    cases = (
        (
            "q_norm",
            _ReturnModule(torch.ones(2, 2, 3)),
            "q_norm shape",
        ),
        (
            "q_norm",
            _ReturnModule(torch.ones(2, 2, 2, dtype=torch.float64)),
            "q_norm dtype",
        ),
        (
            "q_norm",
            _ReturnModule(torch.ones(2, 2, 2, device="meta")),
            "q_norm device",
        ),
        (
            "k_norm",
            _ReturnModule(torch.ones(2, 2, 2)),
            "k_norm shape",
        ),
        (
            "k_norm",
            _ReturnModule(torch.ones(2, 1, 2, dtype=torch.float64)),
            "k_norm dtype",
        ),
        (
            "k_norm",
            _ReturnModule(torch.ones(2, 1, 2, device="meta")),
            "k_norm device",
        ),
        (
            "rotary",
            _ReturnModule(torch.ones(2, 4)),
            "rotary must return",
        ),
        (
            "rotary",
            _ReturnModule((torch.ones(2, 3), torch.ones(2, 2))),
            "rotary query shape",
        ),
        (
            "rotary",
            _ReturnModule(
                (
                    torch.ones(2, 4),
                    torch.ones(2, 2, dtype=torch.float64),
                )
            ),
            "rotary key dtype",
        ),
        (
            "rotary",
            _ReturnModule(
                (
                    torch.ones(2, 4, dtype=torch.float64),
                    torch.ones(2, 2),
                )
            ),
            "rotary query dtype",
        ),
        (
            "rotary",
            _ReturnModule(
                (
                    torch.ones(2, 4, device="meta"),
                    torch.ones(2, 2),
                )
            ),
            "rotary query device",
        ),
        (
            "rotary",
            _ReturnModule(
                (
                    torch.ones(2, 4),
                    torch.ones(2, 2, device="meta"),
                )
            ),
            "rotary key device",
        ),
        (
            "attention_backend",
            _ReturnModule(torch.ones(2, 2, 2)),
            "attention_backend output must be rank two",
        ),
        (
            "attention_backend",
            _ReturnModule(torch.ones(1, 4)),
            "attention_backend token",
        ),
        (
            "attention_backend",
            _ReturnModule(torch.ones(2, 3)),
            "attention_backend feature",
        ),
        (
            "attention_backend",
            _ReturnModule(torch.ones(2, 4, dtype=torch.float64)),
            "attention_backend dtype",
        ),
        (
            "attention_backend",
            _ReturnModule(torch.ones(2, 4, device="meta")),
            "attention_backend device",
        ),
        (
            "output_projection",
            _ReturnModule(torch.ones(1, 3)),
            "output_projection token",
        ),
        (
            "output_projection",
            _ReturnModule(torch.ones(2, 1, 3)),
            "output_projection output must be rank two",
        ),
        (
            "output_projection",
            _ReturnModule(torch.ones(2, 3, dtype=torch.int64)),
            "output_projection output must use a floating point dtype",
        ),
        (
            "output_projection",
            _ReturnModule(torch.ones(2, 3, dtype=torch.float64)),
            "output_projection dtype",
        ),
        (
            "output_projection",
            _ReturnModule(torch.ones(2, 3, device="meta")),
            "output_projection device",
        ),
    )
    for name, module, message in cases:
        shell = _boundary_shell(**{name: module})
        _expect_value_error(
            lambda shell=shell: shell(positions, hidden),
            message,
        )


def main() -> None:
    test_operation_order_and_numerical_oracle()
    test_bfloat16_preserves_dtype_and_matches_native_math_oracle()
    test_zero_and_nonzero_query_gates_apply_before_output_projection()
    test_cached_prefill_quantized_kv_uses_original_backend()
    test_uncached_prefill_uses_dense_eager_attention()
    test_cached_decode_uses_qwen35_bfloat16_eager_attention()
    test_noncontiguous_projection_outputs_follow_shape_contract()
    test_constructor_and_input_fail_closed()
    test_projection_boundaries_fail_closed()
    test_norm_rotary_attention_and_output_boundaries_fail_closed()
    print("qwen35 full attention shell tests passed")


if __name__ == "__main__":
    main()
