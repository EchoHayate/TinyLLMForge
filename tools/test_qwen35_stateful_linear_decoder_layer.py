import importlib.util
from pathlib import Path
import sys
import types

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


for package_name in ("tinyvllm", "tinyvllm.engine", "tinyvllm.layers"):
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / package_name.replace(".", "/"))]
    sys.modules[package_name] = package

hybrid = _load_module(
    "tinyvllm.engine.hybrid_state",
    "tinyvllm/engine/hybrid_state.py",
)
adapter_module = _load_module(
    "tinyvllm.engine.qwen35_layer_state",
    "tinyvllm/engine/qwen35_layer_state.py",
)
decoder_module = _load_module(
    "tinyvllm.layers.qwen35_decoder_layer",
    "tinyvllm/layers/qwen35_decoder_layer.py",
)
stateful_module = _load_module(
    "tinyvllm.layers.qwen35_stateful_decoder_layer",
    "tinyvllm/layers/qwen35_stateful_decoder_layer.py",
)

HybridStateComponentSpec = hybrid.HybridStateComponentSpec
HybridStateLayout = hybrid.HybridStateLayout
HybridStateLease = hybrid.HybridStateLease
HybridStateTensorPool = hybrid.HybridStateTensorPool
Qwen35LayerStateAdapter = adapter_module.Qwen35LayerStateAdapter
Qwen35DecoderLayerShell = decoder_module.Qwen35DecoderLayerShell
Qwen35StatefulLinearDecoderLayer = (
    stateful_module.Qwen35StatefulLinearDecoderLayer
)


class _Affine(nn.Module):

    def __init__(
        self,
        name: str,
        events: list,
        scale: float,
        bias: float,
    ):
        super().__init__()
        self.name = name
        self.events = events
        self.scale = scale
        self.bias = bias

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        self.events.append(self.name)
        return tensor * self.scale + self.bias


class _StatefulMixer(nn.Module):

    def __init__(self, events: list):
        super().__init__()
        self.events = events

    def forward(
        self,
        hidden_states: torch.Tensor,
        convolution_state: torch.Tensor,
        recurrent_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.events.append("linear_attention")
        state_delta = hidden_states.sum().to(convolution_state.dtype)
        return (
            hidden_states * 0.5,
            convolution_state + state_delta,
            recurrent_state + state_delta * 0.25,
        )


class _ForbiddenFullAttention(nn.Module):

    def forward(self, position_ids, hidden_states):
        raise AssertionError("full attention must not be called")


class _IdentityFullAttention(nn.Module):

    def forward(self, position_ids, hidden_states):
        return hidden_states


class _Failing(nn.Module):

    def __init__(self, message: str, events: list, name: str):
        super().__init__()
        self.message = message
        self.events = events
        self.name = name

    def forward(self, *args):
        self.events.append(self.name)
        raise RuntimeError(self.message)


class _Return(nn.Module):

    def __init__(self, output, events: list, name: str):
        super().__init__()
        self.output = output
        self.events = events
        self.name = name

    def forward(self, *args):
        self.events.append(self.name)
        return self.output


def _fixture(
    dtype: torch.dtype = torch.float32,
) -> tuple:
    events = []
    layout = HybridStateLayout((
        HybridStateComponentSpec(
            3,
            "linear_convolution",
            (4, 3),
            dtype,
        ),
        HybridStateComponentSpec(
            3,
            "linear_recurrent",
            (2, 3, 2),
            dtype,
        ),
    ))
    pool = HybridStateTensorPool(layout, capacity=1, device="cpu")
    lease = HybridStateLease(0, 1, 17)
    pool.activate(lease)
    convolution = pool.component_tensor(3, "linear_convolution")
    recurrent = pool.component_tensor(3, "linear_recurrent")
    convolution[0].copy_(
        torch.arange(12, dtype=torch.float32).reshape(4, 3).to(dtype)
        / 10
    )
    recurrent[0].copy_(
        torch.arange(12, dtype=torch.float32).reshape(2, 3, 2).to(dtype)
        / 20
    )
    adapter = Qwen35LayerStateAdapter(pool, layer_index=3)
    decoder = Qwen35DecoderLayerShell(
        block_type="linear_attention",
        input_layernorm=_Affine(
            "input_layernorm",
            events,
            2.0,
            1.0,
        ),
        post_attention_layernorm=_Affine(
            "post_attention_layernorm",
            events,
            1.0,
            -0.25,
        ),
        mlp=_Affine("mlp", events, 0.2, 0.0),
        full_attention=_ForbiddenFullAttention(),
        linear_attention=_StatefulMixer(events),
    )
    wrapper = Qwen35StatefulLinearDecoderLayer(decoder, adapter)

    original_gather = adapter.gather
    original_commit = adapter.commit

    def gather(recorded_lease):
        events.append("gather")
        return original_gather(recorded_lease)

    def commit(recorded_lease, convolution_state, recurrent_state):
        events.append("commit")
        return original_commit(
            recorded_lease,
            convolution_state,
            recurrent_state,
        )

    adapter.gather = gather
    adapter.commit = commit
    return events, pool, lease, adapter, decoder, wrapper


def _hidden(dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.tensor(
        [
            [1.0, -2.0, 0.5, 3.0],
            [-1.0, 0.25, 2.0, -0.5],
        ],
        dtype=dtype,
    )


def _manual(
    hidden_states: torch.Tensor,
    convolution_state: torch.Tensor,
    recurrent_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    normalized = hidden_states * 2.0 + 1.0
    mixed = normalized * 0.5
    after_mixer = hidden_states + mixed
    post_normalized = after_mixer - 0.25
    output = after_mixer + post_normalized * 0.2
    state_delta = normalized.sum().to(convolution_state.dtype)
    return (
        output,
        convolution_state + state_delta,
        recurrent_state + state_delta * 0.25,
    )


def _pool_rows(pool: HybridStateTensorPool) -> tuple:
    return (
        pool.component_tensor(3, "linear_convolution")[0].clone(),
        pool.component_tensor(3, "linear_recurrent")[0].clone(),
    )


def _assert_pool_unchanged(
    pool: HybridStateTensorPool,
    original: tuple,
) -> None:
    current = _pool_rows(pool)
    torch.testing.assert_close(current[0], original[0])
    torch.testing.assert_close(current[1], original[1])


def test_order_numerical_oracle_and_successful_commit() -> None:
    events, pool, lease, _, _, wrapper = _fixture()
    hidden_states = _hidden()
    position_ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64)
    original_hidden = hidden_states.clone()
    original_positions = position_ids.clone()
    initial_conv, initial_recurrent = _pool_rows(pool)
    expected = _manual(
        hidden_states,
        initial_conv,
        initial_recurrent,
    )
    actual = wrapper(lease, position_ids, hidden_states)
    assert events == [
        "gather",
        "input_layernorm",
        "linear_attention",
        "post_attention_layernorm",
        "mlp",
        "commit",
    ]
    torch.testing.assert_close(actual, expected[0])
    committed_conv, committed_recurrent = _pool_rows(pool)
    torch.testing.assert_close(committed_conv, expected[1])
    torch.testing.assert_close(committed_recurrent, expected[2])
    torch.testing.assert_close(hidden_states, original_hidden)
    torch.testing.assert_close(position_ids, original_positions)


def test_bfloat16_and_noncontiguous_hidden_input() -> None:
    events, pool, lease, _, _, wrapper = _fixture(torch.bfloat16)
    hidden_states = (
        torch.arange(8, dtype=torch.float32)
        .reshape(4, 2)
        .t()
        .to(torch.bfloat16)
    )
    assert not hidden_states.is_contiguous()
    initial_conv, initial_recurrent = _pool_rows(pool)
    expected = _manual(
        hidden_states,
        initial_conv,
        initial_recurrent,
    )
    actual = wrapper(lease, torch.tensor([0, 1]), hidden_states)
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual.float(), expected[0].float())
    committed_conv, committed_recurrent = _pool_rows(pool)
    torch.testing.assert_close(
        committed_conv.float(),
        expected[1].float(),
    )
    torch.testing.assert_close(
        committed_recurrent.float(),
        expected[2].float(),
    )
    assert events[-1] == "commit"


def test_stale_lease_fails_before_layer_components() -> None:
    events, pool, _, _, _, wrapper = _fixture()
    original = _pool_rows(pool)
    stale = HybridStateLease(0, 0, 17)
    try:
        wrapper(stale, torch.tensor([0, 1]), _hidden())
    except RuntimeError as error:
        assert "lease mismatch" in str(error)
    else:
        raise AssertionError("stale lease was accepted")
    assert events == ["gather"]
    _assert_pool_unchanged(pool, original)


def test_layer_failures_after_gather_leave_pool_unchanged() -> None:
    cases = (
        ("input_layernorm", "input norm failure"),
        ("linear_attention", "mixer failure"),
        ("post_attention_layernorm", "post norm failure"),
        ("mlp", "mlp failure"),
    )
    for component, message in cases:
        events, pool, lease, _, decoder, wrapper = _fixture()
        original = _pool_rows(pool)
        setattr(
            decoder,
            component,
            _Failing(message, events, component),
        )
        try:
            wrapper(lease, torch.tensor([0, 1]), _hidden())
        except RuntimeError as error:
            assert message in str(error)
        else:
            raise AssertionError(f"{component} failure was swallowed")
        assert "commit" not in events
        _assert_pool_unchanged(pool, original)


def test_malformed_mixer_and_candidate_contracts_fail_before_commit() -> None:
    cases = (
        (
            lambda hidden, conv, recurrent: hidden,
            "three-item tuple",
        ),
        (
            lambda hidden, conv, recurrent: (
                torch.ones(2, 3),
                conv,
                recurrent,
            ),
            "linear_attention shape",
        ),
        (
            lambda hidden, conv, recurrent: (
                hidden,
                torch.ones(4, 2),
                recurrent,
            ),
            "convolution_state shape",
        ),
        (
            lambda hidden, conv, recurrent: (
                hidden,
                conv,
                torch.ones(2, 2, 3),
            ),
            "recurrent_state shape",
        ),
        (
            lambda hidden, conv, recurrent: (
                hidden,
                conv.to(torch.float64),
                recurrent,
            ),
            "convolution_state dtype",
        ),
    )
    for case_index, (callback, message) in enumerate(cases):
        events, pool, lease, _, decoder, wrapper = _fixture()
        original = _pool_rows(pool)

        class _MalformedMixer(nn.Module):

            def forward(self, hidden_states, conv_state, recurrent_state):
                events.append("linear_attention")
                return callback(hidden_states, conv_state, recurrent_state)

        decoder.linear_attention = _MalformedMixer()
        try:
            wrapper(lease, torch.tensor([0, 1]), _hidden())
        except ValueError as error:
            assert message in str(error), str(error)
        else:
            raise AssertionError("malformed mixer output was accepted")
        if case_index < 2:
            assert "commit" not in events
        else:
            assert events[-1] == "commit"
        _assert_pool_unchanged(pool, original)


def test_commit_second_copy_failure_rolls_back_pool() -> None:
    events, pool, lease, adapter, _, wrapper = _fixture()
    original = _pool_rows(pool)
    original_copy = adapter._copy_component
    calls = []

    def failing_copy(destination, source):
        calls.append(destination)
        if len(calls) == 2:
            raise RuntimeError("injected second copy failure")
        return original_copy(destination, source)

    adapter._copy_component = failing_copy
    try:
        wrapper(lease, torch.tensor([0, 1]), _hidden())
    except RuntimeError as error:
        assert "second copy failure" in str(error)
    else:
        raise AssertionError("commit copy failure was swallowed")
    assert events[-1] == "commit"
    _assert_pool_unchanged(pool, original)


def test_constructor_and_hidden_boundaries_fail_closed() -> None:
    _, _, _, adapter, _, _ = _fixture()
    full_decoder = Qwen35DecoderLayerShell(
        block_type="full_attention",
        input_layernorm=nn.Identity(),
        post_attention_layernorm=nn.Identity(),
        mlp=nn.Identity(),
        full_attention=_IdentityFullAttention(),
    )
    try:
        Qwen35StatefulLinearDecoderLayer(full_decoder, adapter)
    except ValueError as error:
        assert "linear_attention" in str(error)
    else:
        raise AssertionError("full-attention decoder was accepted")

    for hidden_states, message in (
        (torch.ones(2, 2, 2), "rank two"),
        (torch.ones(2, 4, dtype=torch.int64), "floating point"),
    ):
        _, pool, lease, _, _, wrapper = _fixture()
        original = _pool_rows(pool)
        try:
            wrapper(lease, torch.tensor([0, 1]), hidden_states)
        except ValueError as error:
            assert message in str(error), str(error)
        else:
            raise AssertionError("invalid hidden_states was accepted")
        _assert_pool_unchanged(pool, original)


def main() -> None:
    test_order_numerical_oracle_and_successful_commit()
    test_bfloat16_and_noncontiguous_hidden_input()
    test_stale_lease_fails_before_layer_components()
    test_layer_failures_after_gather_leave_pool_unchanged()
    test_malformed_mixer_and_candidate_contracts_fail_before_commit()
    test_commit_second_copy_failure_rolls_back_pool()
    test_constructor_and_hidden_boundaries_fail_closed()
    print("qwen35 stateful linear decoder layer tests passed")


if __name__ == "__main__":
    main()
