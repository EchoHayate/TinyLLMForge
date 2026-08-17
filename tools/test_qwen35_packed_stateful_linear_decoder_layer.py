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
packed_module = _load_module(
    "tinyvllm.layers.qwen35_packed_stateful_decoder_layer",
    "tinyvllm/layers/qwen35_packed_stateful_decoder_layer.py",
)

HybridStateComponentSpec = hybrid.HybridStateComponentSpec
HybridStateLayout = hybrid.HybridStateLayout
HybridStateLease = hybrid.HybridStateLease
HybridStateTensorPool = hybrid.HybridStateTensorPool
Qwen35LayerStateAdapter = adapter_module.Qwen35LayerStateAdapter
Qwen35DecoderLayerShell = decoder_module.Qwen35DecoderLayerShell
Qwen35PackedStatefulLinearDecoderLayer = (
    packed_module.Qwen35PackedStatefulLinearDecoderLayer
)


class _Affine(nn.Module):

    def __init__(self, name: str, events: list, scale: float, bias: float):
        super().__init__()
        self.name = name
        self.events = events
        self.scale = scale
        self.bias = bias

    def forward(self, tensor):
        self.events.append((self.name, tensor.shape[0]))
        return tensor * self.scale + self.bias


class _StatefulMixer(nn.Module):

    def __init__(self, events: list):
        super().__init__()
        self.events = events
        self.call_index = 0
        self.fail_on_call = None
        self.invalid_candidate_on_call = None

    def forward(self, hidden_states, convolution_state, recurrent_state):
        call_index = self.call_index
        self.call_index += 1
        self.events.append(("linear_attention", hidden_states.shape[0]))
        if call_index == self.fail_on_call:
            raise RuntimeError("injected packed request failure")
        delta = hidden_states.sum().to(convolution_state.dtype)
        candidate_convolution = convolution_state + delta
        candidate_recurrent = recurrent_state + delta * 0.25
        if call_index == self.invalid_candidate_on_call:
            candidate_recurrent = candidate_recurrent[:, :, :1]
        return (
            hidden_states * 0.5,
            candidate_convolution,
            candidate_recurrent,
        )


class _ForbiddenFull(nn.Module):

    def forward(self, position_ids, hidden_states):
        raise AssertionError("full attention must not be called")


def _fixture(dtype: torch.dtype = torch.float32):
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
    pool = HybridStateTensorPool(layout, capacity=3, device="cpu")
    leases = (
        HybridStateLease(0, 1, 17),
        HybridStateLease(1, 1, 18),
        HybridStateLease(2, 1, 19),
    )
    for lease in leases:
        pool.activate(lease)
    convolution = pool.component_tensor(3, "linear_convolution")
    recurrent = pool.component_tensor(3, "linear_recurrent")
    for slot_id in range(3):
        convolution[slot_id].copy_(
            (
                torch.arange(12, dtype=torch.float32).reshape(4, 3)
                + slot_id * 100
            ).to(dtype)
        )
        recurrent[slot_id].copy_(
            (
                torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
                + slot_id * 1000
            ).to(dtype)
        )
    adapter = Qwen35LayerStateAdapter(pool, layer_index=3)
    mixer = _StatefulMixer(events)
    decoder = Qwen35DecoderLayerShell(
        block_type="linear_attention",
        input_layernorm=_Affine("input_layernorm", events, 2.0, 1.0),
        post_attention_layernorm=_Affine(
            "post_attention_layernorm",
            events,
            1.0,
            -0.25,
        ),
        mlp=_Affine("mlp", events, 0.2, 0.0),
        full_attention=_ForbiddenFull(),
        linear_attention=mixer,
    )
    wrapper = Qwen35PackedStatefulLinearDecoderLayer(decoder, adapter)
    return events, pool, leases, adapter, mixer, wrapper


def _hidden(dtype: torch.dtype = torch.float32):
    return torch.tensor(
        [
            [1.0, -2.0, 0.5, 3.0],
            [-1.0, 0.25, 2.0, -0.5],
            [0.5, 1.5, -1.0, 2.0],
            [2.0, 0.0, -0.5, 1.0],
            [-2.0, 1.0, 0.25, 0.75],
            [1.25, -0.75, 2.5, -1.5],
        ],
        dtype=dtype,
    )


def _pool_rows(pool):
    return (
        pool.component_tensor(3, "linear_convolution").clone(),
        pool.component_tensor(3, "linear_recurrent").clone(),
    )


def _manual(hidden_states, token_counts, convolution, recurrent):
    outputs = []
    candidate_convolution = []
    candidate_recurrent = []
    offset = 0
    for request_index, token_count in enumerate(token_counts):
        segment = hidden_states[offset:offset + token_count]
        normalized = segment * 2.0 + 1.0
        mixed = normalized * 0.5
        after_mixer = segment + mixed
        post_normalized = after_mixer - 0.25
        outputs.append(after_mixer + post_normalized * 0.2)
        delta = normalized.sum().to(convolution.dtype)
        candidate_convolution.append(
            convolution[request_index] + delta
        )
        candidate_recurrent.append(
            recurrent[request_index] + delta * 0.25
        )
        offset += token_count
    return (
        torch.cat(outputs),
        torch.stack(candidate_convolution),
        torch.stack(candidate_recurrent),
    )


def test_packed_segments_match_oracle_and_commit_once() -> None:
    events, pool, leases, adapter, _, wrapper = _fixture()
    hidden_states = _hidden()
    position_ids = torch.arange(6, dtype=torch.int64)
    original_hidden = hidden_states.clone()
    original_positions = position_ids.clone()
    initial = _pool_rows(pool)
    expected = _manual(hidden_states, (2, 1, 3), *initial)
    commit_calls = []
    original_commit = adapter.commit_batch

    def commit_batch(*args):
        commit_calls.append(args)
        return original_commit(*args)

    adapter.commit_batch = commit_batch
    actual = wrapper(
        leases,
        (2, 1, 3),
        position_ids,
        hidden_states,
    )
    torch.testing.assert_close(actual, expected[0])
    committed = _pool_rows(pool)
    torch.testing.assert_close(committed[0], expected[1])
    torch.testing.assert_close(committed[1], expected[2])
    assert len(commit_calls) == 1
    assert [event[1] for event in events if event[0] == "linear_attention"] == [
        2,
        1,
        3,
    ]
    torch.testing.assert_close(hidden_states, original_hidden)
    torch.testing.assert_close(position_ids, original_positions)


def test_bfloat16_noncontiguous_and_out_of_slot_order() -> None:
    _, pool, leases, _, _, wrapper = _fixture(torch.bfloat16)
    hidden_states = (
        torch.arange(24, dtype=torch.float32)
        .reshape(4, 6)
        .t()
        .to(torch.bfloat16)
    )
    assert not hidden_states.is_contiguous()
    ordered_leases = (leases[2], leases[0], leases[1])
    initial = _pool_rows(pool)
    ordered_initial = (
        torch.stack((initial[0][2], initial[0][0], initial[0][1])),
        torch.stack((initial[1][2], initial[1][0], initial[1][1])),
    )
    expected = _manual(
        hidden_states,
        (2, 1, 3),
        *ordered_initial,
    )
    actual = wrapper(
        ordered_leases,
        (2, 1, 3),
        torch.arange(6),
        hidden_states,
    )
    torch.testing.assert_close(actual.float(), expected[0].float())
    committed = _pool_rows(pool)
    torch.testing.assert_close(committed[0][2], expected[1][0])
    torch.testing.assert_close(committed[0][0], expected[1][1])
    torch.testing.assert_close(committed[0][1], expected[1][2])
    torch.testing.assert_close(committed[1][2], expected[2][0])
    torch.testing.assert_close(committed[1][0], expected[2][1])
    torch.testing.assert_close(committed[1][1], expected[2][2])


def _expect_error(function, error_type, message):
    try:
        function()
    except error_type as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(
            f"expected {error_type.__name__} containing {message!r}"
        )


def test_metadata_boundaries_fail_before_pool_mutation() -> None:
    cases = (
        ([], (2, 1, 3), torch.arange(6), _hidden(), "leases"),
        ((1, 2, 3), (2, 1, 3), torch.arange(6), _hidden(), "leases"),
        (None, (2, 1, 3), torch.arange(6), _hidden(), "leases"),
    )
    for leases_value, counts, positions, hidden, message in cases:
        _, pool, leases, _, _, wrapper = _fixture()
        original = _pool_rows(pool)
        actual_leases = leases if leases_value is None else leases_value
        if leases_value is None:
            actual_leases = ()
        _expect_error(
            lambda: wrapper(actual_leases, counts, positions, hidden),
            ValueError,
            message,
        )
        current = _pool_rows(pool)
        torch.testing.assert_close(current[0], original[0])
        torch.testing.assert_close(current[1], original[1])

    for counts, positions, hidden, message in (
        ([2, 1, 3], torch.arange(6), _hidden(), "token_counts"),
        ((2, 1), torch.arange(6), _hidden(), "batch size"),
        ((2, 0, 4), torch.arange(6), _hidden(), "positive integers"),
        ((2, True, 3), torch.arange(6), _hidden(), "positive integers"),
        ((2, 1, 2), torch.arange(6), _hidden(), "sum"),
        ((2, 1, 3), torch.arange(5), _hidden(), "position_ids"),
        (
            (2, 1, 3),
            torch.zeros(2, 6, dtype=torch.int64),
            _hidden(),
            "position_ids",
        ),
        (
            (2, 1, 3),
            torch.arange(6),
            torch.ones(2, 3, 4),
            "rank two",
        ),
    ):
        _, pool, leases, _, _, wrapper = _fixture()
        original = _pool_rows(pool)
        _expect_error(
            lambda counts=counts, positions=positions,
            hidden=hidden: wrapper(leases, counts, positions, hidden),
            ValueError,
            message,
        )
        current = _pool_rows(pool)
        torch.testing.assert_close(current[0], original[0])
        torch.testing.assert_close(current[1], original[1])


def test_later_request_and_candidate_failures_skip_commit() -> None:
    for failure_kind in ("request", "candidate"):
        _, pool, leases, adapter, mixer, wrapper = _fixture()
        original = _pool_rows(pool)
        commit_calls = []
        original_commit = adapter.commit_batch

        def commit_batch(*args):
            commit_calls.append(args)
            return original_commit(*args)

        adapter.commit_batch = commit_batch
        if failure_kind == "request":
            mixer.fail_on_call = 2
            expected_type = RuntimeError
            expected_message = "packed request failure"
        else:
            mixer.invalid_candidate_on_call = 1
            expected_type = ValueError
            expected_message = "recurrent_state shape"
        _expect_error(
            lambda: wrapper(
                leases,
                (2, 1, 3),
                torch.arange(6),
                _hidden(),
            ),
            expected_type,
            expected_message,
        )
        assert not commit_calls
        current = _pool_rows(pool)
        torch.testing.assert_close(current[0], original[0])
        torch.testing.assert_close(current[1], original[1])


def test_commit_copy_failure_rolls_back_full_batch() -> None:
    _, pool, leases, adapter, _, wrapper = _fixture()
    original = _pool_rows(pool)
    original_copy = adapter._copy_component
    calls = []

    def failing_copy(destination, source):
        calls.append(destination)
        if len(calls) == 5:
            raise RuntimeError("injected packed commit failure")
        return original_copy(destination, source)

    adapter._copy_component = failing_copy
    _expect_error(
        lambda: wrapper(
            leases,
            (2, 1, 3),
            torch.arange(6),
            _hidden(),
        ),
        RuntimeError,
        "packed commit failure",
    )
    current = _pool_rows(pool)
    torch.testing.assert_close(current[0], original[0])
    torch.testing.assert_close(current[1], original[1])


def main():
    test_packed_segments_match_oracle_and_commit_once()
    test_bfloat16_noncontiguous_and_out_of_slot_order()
    test_metadata_boundaries_fail_before_pool_mutation()
    test_later_request_and_candidate_failures_skip_commit()
    test_commit_copy_failure_rolls_back_full_batch()
    print("qwen35 packed stateful linear decoder layer tests passed")


if __name__ == "__main__":
    main()
