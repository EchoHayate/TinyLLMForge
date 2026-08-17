import importlib.util
from pathlib import Path
import sys
import types

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]


def _load_module(module_name, relative_path):
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

hybrid = _load_module("tinyvllm.engine.hybrid_state", "tinyvllm/engine/hybrid_state.py")
adapter_module = _load_module("tinyvllm.engine.qwen35_layer_state", "tinyvllm/engine/qwen35_layer_state.py")
transaction_module = _load_module("tinyvllm.engine.qwen35_state_transaction", "tinyvllm/engine/qwen35_state_transaction.py")
decoder_module = _load_module("tinyvllm.layers.qwen35_decoder_layer", "tinyvllm/layers/qwen35_decoder_layer.py")
stack_module = _load_module("tinyvllm.layers.qwen35_packed_layer_stack", "tinyvllm/layers/qwen35_packed_layer_stack.py")
from tinyvllm.utils.context import get_context, reset_context, set_context

HybridStateComponentSpec = hybrid.HybridStateComponentSpec
HybridStateLayout = hybrid.HybridStateLayout
HybridStateLease = hybrid.HybridStateLease
HybridStateTensorPool = hybrid.HybridStateTensorPool
Qwen35LayerStateAdapter = adapter_module.Qwen35LayerStateAdapter
Qwen35CrossLayerStateTransaction = transaction_module.Qwen35CrossLayerStateTransaction
Qwen35DecoderLayerShell = decoder_module.Qwen35DecoderLayerShell
Qwen35PackedHeterogeneousLayerStack = stack_module.Qwen35PackedHeterogeneousLayerStack


class _Affine(nn.Module):
    def __init__(self, label, events, scale, bias):
        super().__init__()
        self.label, self.events, self.scale, self.bias = label, events, scale, bias
        self.call_index = 0
        self.fail_on_call = None

    def forward(self, tensor):
        call = self.call_index
        self.call_index += 1
        self.events.append((self.label, tensor.shape[0]))
        if call == self.fail_on_call:
            raise RuntimeError(f"injected {self.label} failure")
        return tensor * self.scale + self.bias


class _LinearMixer(nn.Module):
    def __init__(self, label, events, state_scale):
        super().__init__()
        self.label, self.events, self.state_scale = label, events, state_scale
        self.call_index = 0
        self.fail_on_call = None
        self.invalid_candidate_on_call = None

    def forward(self, hidden, convolution, recurrent):
        call = self.call_index
        self.call_index += 1
        self.events.append((self.label, hidden.shape[0]))
        if call == self.fail_on_call:
            raise RuntimeError(f"injected {self.label} failure")
        delta = hidden.sum().to(convolution.dtype) * self.state_scale
        next_recurrent = recurrent + delta * 0.25
        if call == self.invalid_candidate_on_call:
            next_recurrent = next_recurrent[:, :, :1]
        return hidden * 0.5, convolution + delta, next_recurrent

    def forward_with_state_trace(
        self,
        hidden,
        convolution,
        recurrent,
    ):
        output = hidden * 0.5
        cumulative = hidden.reshape(
            hidden.shape[0],
            -1,
        ).sum(dim=-1).cumsum(dim=0).to(convolution.dtype)
        convolution_trace = torch.stack(tuple(
            convolution + value * self.state_scale
            for value in cumulative
        ))
        recurrent_trace = torch.stack(tuple(
            recurrent + value * self.state_scale * 0.25
            for value in cumulative
        ))
        trace = types.SimpleNamespace(
            convolution=convolution_trace,
            recurrent=recurrent_trace,
        )
        return (
            output,
            convolution_trace[-1],
            recurrent_trace[-1],
            trace,
        )


class _PrefillContextValidatingLinearMixer(_LinearMixer):
    def __init__(self, label, events, state_scale):
        super().__init__(label, events, state_scale)
        self.contexts = []

    def forward(self, hidden, convolution, recurrent):
        context = get_context()
        q_offsets = context.cu_seqlens_q.tolist()
        k_offsets = context.cu_seqlens_k.tolist()
        projected_rows = sum(
            max(
                int(q_offsets[index + 1]) - int(q_offsets[index]),
                int(k_offsets[index + 1]) - int(k_offsets[index]),
            )
            for index in range(len(q_offsets) - 1)
        )
        if projected_rows != hidden.shape[0]:
            raise ValueError(
                f"in_proj_qkv token count must equal {hidden.shape[0]}"
            )
        self.contexts.append({
            "cu_seqlens_q": context.cu_seqlens_q.clone(),
            "cu_seqlens_k": context.cu_seqlens_k.clone(),
            "max_seqlen_q": context.max_seqlen_q,
            "max_seqlen_k": context.max_seqlen_k,
        })
        return super().forward(hidden, convolution, recurrent)


class _PrefillContextValidatingMLP(_Affine):
    def __init__(self, label, events, scale, bias):
        super().__init__(label, events, scale, bias)
        self.contexts = []

    def forward(self, tensor):
        context = get_context()
        q_offsets = context.cu_seqlens_q.tolist()
        k_offsets = context.cu_seqlens_k.tolist()
        projected_rows = sum(
            max(
                int(q_offsets[index + 1]) - int(q_offsets[index]),
                int(k_offsets[index + 1]) - int(k_offsets[index]),
            )
            for index in range(len(q_offsets) - 1)
        )
        if projected_rows != tensor.shape[0]:
            raise ValueError(
                f"mlp token count must equal {tensor.shape[0]}"
            )
        self.contexts.append({
            "cu_seqlens_q": context.cu_seqlens_q.clone(),
            "cu_seqlens_k": context.cu_seqlens_k.clone(),
        })
        return super().forward(tensor)


class _DecodeContextRecordingLinearMixer(_LinearMixer):
    def __init__(self, label, events, state_scale):
        super().__init__(label, events, state_scale)
        self.contexts = []

    def forward(self, hidden, convolution, recurrent):
        context = get_context()
        self.contexts.append({
            "slot_mapping": context.slot_mapping.clone(),
            "context_lens": context.context_lens.clone(),
            "block_tables": context.block_tables.clone(),
        })
        return super().forward(hidden, convolution, recurrent)


class _FullMixer(nn.Module):
    def __init__(self, events):
        super().__init__()
        self.events = events
        self.call_index = 0
        self.fail_on_call = None

    def forward(self, positions, hidden):
        call = self.call_index
        self.call_index += 1
        self.events.append(("full1", hidden.shape[0]))
        if call == self.fail_on_call:
            raise RuntimeError("injected full1 failure")
        pos = positions[0] if positions.ndim == 2 else positions
        return hidden + hidden.mean(0, keepdim=True) + pos.to(hidden.dtype).unsqueeze(-1)


class _ContextRecordingFullMixer(_FullMixer):
    def __init__(self, events):
        super().__init__(events)
        self.contexts = []

    def forward(self, positions, hidden):
        context = get_context()
        self.contexts.append({
            "slot_mapping": context.slot_mapping.clone(),
            "cu_seqlens_q": context.cu_seqlens_q.clone(),
            "cu_seqlens_k": context.cu_seqlens_k.clone(),
            "block_tables": context.block_tables.clone(),
            "prefill_attention_reference_lens": (
                context.prefill_attention_reference_lens
            ),
        })
        return super().forward(positions, hidden)


class _DecodeContextRecordingFullMixer(_FullMixer):
    def __init__(self, events):
        super().__init__(events)
        self.contexts = []

    def forward(self, positions, hidden):
        context = get_context()
        self.contexts.append({
            "slot_mapping": context.slot_mapping.clone(),
            "context_lens": context.context_lens.clone(),
            "block_tables": context.block_tables.clone(),
            "kv_offload_logical_block_tables": (
                context.kv_offload_logical_block_tables
            ),
            "kv_offload_context_lens": (
                context.kv_offload_context_lens
            ),
            "kv_offload_write_blocks": (
                context.kv_offload_write_blocks
            ),
        })
        return super().forward(positions, hidden)


class _SpecVerifyContextRecordingFullMixer(_FullMixer):
    def __init__(self, events):
        super().__init__(events)
        self.contexts = []

    def forward(self, positions, hidden):
        context = get_context()
        self.contexts.append({
            "mode": context.mode,
            "slot_mapping": context.slot_mapping.clone(),
            "context_lens": context.context_lens.clone(),
            "block_tables": context.block_tables.clone(),
            "spec_verify_query_lens": (
                context.spec_verify_query_lens
            ),
            "kv_offload_logical_block_tables": (
                context.kv_offload_logical_block_tables
            ),
            "kv_offload_context_lens": (
                context.kv_offload_context_lens
            ),
            "kv_offload_write_blocks": (
                context.kv_offload_write_blocks
            ),
        })
        return super().forward(positions, hidden)


def _decoder(block_type, layer, events, mixer):
    return Qwen35DecoderLayerShell(
        block_type=block_type,
        input_layernorm=_Affine(f"in{layer}", events, 1.1 + layer * 0.1, 0.1 * (layer + 1)),
        post_attention_layernorm=_Affine(f"post{layer}", events, 0.9, -0.05 * (layer + 1)),
        mlp=_Affine(f"mlp{layer}", events, 0.2 + 0.05 * layer, 0.0),
        full_attention=mixer if block_type == "full_attention" else None,
        linear_attention=mixer if block_type == "linear_attention" else None,
    )


def _fixture(dtype=torch.float32):
    events = []
    layout = HybridStateLayout(tuple(
        component
        for layer in (0, 2)
        for component in (
            HybridStateComponentSpec(layer, "linear_convolution", (4, 3), dtype),
            HybridStateComponentSpec(layer, "linear_recurrent", (2, 3, 2), dtype),
        )
    ))
    pool = HybridStateTensorPool(layout, 3, "cpu")
    leases = tuple(HybridStateLease(i, 1, 17 + i) for i in range(3))
    for lease in leases:
        pool.activate(lease)
    for layer_offset, layer in enumerate((0, 2)):
        for slot in range(3):
            pool.component_tensor(layer, "linear_convolution")[slot].copy_(
                (torch.arange(12).reshape(4, 3) + layer_offset * 10000 + slot * 100).to(dtype)
            )
            pool.component_tensor(layer, "linear_recurrent")[slot].copy_(
                (torch.arange(12).reshape(2, 3, 2) + layer_offset * 20000 + slot * 1000).to(dtype)
            )
    adapters = (Qwen35LayerStateAdapter(pool, 0), Qwen35LayerStateAdapter(pool, 2))
    transaction = Qwen35CrossLayerStateTransaction(adapters)
    linear0 = _LinearMixer("linear0", events, 1.0)
    full1 = _FullMixer(events)
    linear2 = _LinearMixer("linear2", events, 2.0)
    layers = (
        _decoder("linear_attention", 0, events, linear0),
        _decoder("full_attention", 1, events, full1),
        _decoder("linear_attention", 2, events, linear2),
    )
    stack = Qwen35PackedHeterogeneousLayerStack(layers, transaction)
    return events, pool, leases, adapters, (linear0, full1, linear2), stack


def _hidden(dtype=torch.float32):
    return torch.arange(24, dtype=torch.float32).reshape(6, 4).div(7).sub(1).to(dtype)


def _snap(pool):
    return tuple(
        (
            pool.component_tensor(layer, "linear_convolution").clone(),
            pool.component_tensor(layer, "linear_recurrent").clone(),
        )
        for layer in (0, 2)
    )


def _manual_layer(hidden, counts, positions, block_type, layer, states=None, state_scale=1.0):
    outputs, convs, recs = [], [], []
    offset = 0
    position_base = positions[0] if positions.ndim == 2 else positions
    for request, count in enumerate(counts):
        segment = hidden[offset:offset + count]
        normalized = segment * (1.1 + layer * 0.1) + 0.1 * (layer + 1)
        if block_type == "linear_attention":
            mixed = normalized * 0.5
            delta = normalized.sum().to(states[0].dtype) * state_scale
            convs.append(states[0][request] + delta)
            recs.append(states[1][request] + delta * 0.25)
        else:
            mixed = normalized + normalized.mean(0, keepdim=True) + position_base[offset:offset + count].to(hidden.dtype).unsqueeze(-1)
        after = segment + mixed
        post = after * 0.9 - 0.05 * (layer + 1)
        outputs.append(after + post * (0.2 + 0.05 * layer))
        offset += count
    candidates = None if block_type == "full_attention" else (torch.stack(convs), torch.stack(recs))
    return torch.cat(outputs), candidates


def test_schedule_output_state_and_call_order():
    events, pool, leases, _, _, stack = _fixture()
    hidden, positions, counts = _hidden(), torch.arange(6), (2, 1, 3)
    original_hidden = hidden.clone()
    original_positions = positions.clone()
    initial = _snap(pool)
    expected0, candidate0 = _manual_layer(hidden, counts, positions, "linear_attention", 0, initial[0], 1.0)
    expected1, _ = _manual_layer(expected0, counts, positions, "full_attention", 1)
    expected2, candidate2 = _manual_layer(expected1, counts, positions, "linear_attention", 2, initial[1], 2.0)
    gather_calls = []
    commit_calls = []
    original_gather = stack.state_transaction.gather
    original_commit = stack.state_transaction.commit

    def recording_gather(gather_leases):
        gather_calls.append(gather_leases)
        return original_gather(gather_leases)

    def recording_commit(commit_leases, candidates):
        commit_calls.append((commit_leases, candidates))
        return original_commit(commit_leases, candidates)

    stack.state_transaction.gather = recording_gather
    stack.state_transaction.commit = recording_commit
    actual = stack(leases, counts, positions, hidden)
    torch.testing.assert_close(actual, expected2)
    torch.testing.assert_close(hidden, original_hidden)
    torch.testing.assert_close(positions, original_positions)
    assert gather_calls == [leases]
    assert len(commit_calls) == 1
    assert commit_calls[0][0] == leases
    assert len(commit_calls[0][1]) == 2
    current = _snap(pool)
    for actual_pair, expected_pair in zip(current, (candidate0, candidate2)):
        torch.testing.assert_close(actual_pair[0], expected_pair[0])
        torch.testing.assert_close(actual_pair[1], expected_pair[1])
    assert [event for event in events if event[0] in ("linear0", "full1", "linear2")] == [
        ("linear0", 2), ("linear0", 1), ("linear0", 3),
        ("full1", 2), ("full1", 1), ("full1", 3),
        ("linear2", 2), ("linear2", 1), ("linear2", 3),
    ]


def test_transactional_prepare_captures_cross_layer_prefix_candidates():
    _, pool, leases, _, _, stack = _fixture()
    hidden = _hidden()
    positions = torch.arange(6)
    counts = (2, 1, 3)
    before = _snap(pool)

    prepared = stack.prepare_transactional(
        leases,
        counts,
        positions,
        hidden,
        capture_prefix_states=True,
    )

    assert prepared.hidden_states.shape == hidden.shape
    assert len(prepared.final_candidates) == 2
    assert prepared.prefix_candidates is not None
    assert tuple(
        len(sequence_prefixes)
        for sequence_prefixes in prepared.prefix_candidates
    ) == counts
    for sequence_index, sequence_prefixes in enumerate(
        prepared.prefix_candidates
    ):
        for prefix in sequence_prefixes:
            assert len(prefix) == 2
        for layer_index, final_pair in enumerate(
            prepared.final_candidates
        ):
            torch.testing.assert_close(
                sequence_prefixes[-1][layer_index][0],
                final_pair[0][sequence_index],
            )
            torch.testing.assert_close(
                sequence_prefixes[-1][layer_index][1],
                final_pair[1][sequence_index],
            )
    current = _snap(pool)
    for current_pair, before_pair in zip(current, before):
        torch.testing.assert_close(current_pair[0], before_pair[0])
        torch.testing.assert_close(current_pair[1], before_pair[1])


def test_transactional_prepare_uses_supplied_candidates_and_fast_path():
    _, pool, leases, _, _, stack = _fixture()
    hidden = _hidden()
    positions = torch.arange(6)
    counts = (2, 1, 3)
    supplied = tuple(
        (
            convolution + 1000,
            recurrent + 2000,
        )
        for convolution, recurrent
        in stack.state_transaction.gather(leases)
    )
    before = _snap(pool)

    prepared = stack.prepare_transactional(
        leases,
        counts,
        positions,
        hidden,
        initial_candidates=supplied,
        capture_prefix_states=False,
    )

    assert prepared.prefix_candidates is None
    assert all(
        torch.all(final_pair[0] > before_pair[0])
        and torch.all(final_pair[1] > before_pair[1])
        for final_pair, before_pair in zip(
            prepared.final_candidates,
            before,
        )
    )
    current = _snap(pool)
    for current_pair, before_pair in zip(current, before):
        torch.testing.assert_close(current_pair[0], before_pair[0])
        torch.testing.assert_close(current_pair[1], before_pair[1])


def test_transactional_prepare_rejects_candidate_inventory_mismatch():
    _, _, leases, _, _, stack = _fixture()
    hidden = _hidden()
    positions = torch.arange(6)
    counts = (2, 1, 3)
    gathered = stack.state_transaction.gather(leases)

    _expect_error(
        lambda: stack.prepare_transactional(
            leases,
            counts,
            positions,
            hidden,
            initial_candidates=gathered[:1],
        ),
        ValueError,
        "candidate count",
    )
    _expect_error(
        lambda: stack.prepare_transactional(
            leases,
            counts,
            positions,
            hidden,
            initial_candidates=(
                (
                    gathered[0][0][:2],
                    gathered[0][1][:2],
                ),
                gathered[1],
            ),
        ),
        ValueError,
        "batched component",
    )


def test_full_layer_request_isolation_and_bfloat16_noncontiguous():
    _, _, leases, _, _, stack = _fixture(torch.bfloat16)
    hidden = torch.arange(24, dtype=torch.float32).reshape(4, 6).t().to(torch.bfloat16)
    assert not hidden.is_contiguous()
    positions, counts = torch.arange(6), (2, 1, 3)
    first = stack(leases, counts, positions, hidden)
    _, _, leases2, _, _, stack2 = _fixture(torch.bfloat16)
    changed = hidden.clone()
    changed[2:].add_(1000)
    second = stack2(leases2, counts, positions, changed)
    torch.testing.assert_close(first[:2].float(), second[:2].float())


def test_full_layer_installs_request_local_prefill_context():
    _, _, leases, _, _, stack = _fixture()
    recorder = _ContextRecordingFullMixer([])
    stack.layers[1].full_attention = recorder
    slot_mapping = torch.tensor([10, 11, 20, 21, 22], dtype=torch.int64)
    cu_seqlens_q = torch.tensor([0, 2, 5], dtype=torch.int32)
    cu_seqlens_k = torch.tensor([0, 4, 10], dtype=torch.int32)
    block_tables = torch.tensor([[1, 2, -1], [4, 5, 6]], dtype=torch.int32)
    set_context(
        True,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=3,
        max_seqlen_k=6,
        slot_mapping=slot_mapping,
        block_tables=block_tables,
        prefill_attention_reference_lens=(4, 6),
    )
    original_context = get_context()
    try:
        stack(
            leases[:2],
            (2, 3),
            torch.arange(5),
            _hidden()[:5],
        )
        assert get_context() is original_context
    finally:
        reset_context()

    assert len(recorder.contexts) == 2
    expected = (
        {
            "slot_mapping": slot_mapping[:2],
            "cu_seqlens_q": torch.tensor([0, 2], dtype=torch.int32),
            "cu_seqlens_k": torch.tensor([0, 4], dtype=torch.int32),
            "block_tables": block_tables[:1],
            "prefill_attention_reference_lens": (4,),
        },
        {
            "slot_mapping": slot_mapping[2:],
            "cu_seqlens_q": torch.tensor([0, 3], dtype=torch.int32),
            "cu_seqlens_k": torch.tensor([0, 6], dtype=torch.int32),
            "block_tables": block_tables[1:2],
            "prefill_attention_reference_lens": (6,),
        },
    )
    for actual, wanted in zip(recorder.contexts, expected):
        torch.testing.assert_close(
            actual.pop("slot_mapping"),
            wanted.pop("slot_mapping"),
        )
        torch.testing.assert_close(
            actual.pop("cu_seqlens_q"),
            wanted.pop("cu_seqlens_q"),
        )
        torch.testing.assert_close(
            actual.pop("cu_seqlens_k"),
            wanted.pop("cu_seqlens_k"),
        )
        torch.testing.assert_close(
            actual.pop("block_tables"),
            wanted.pop("block_tables"),
        )
        assert actual == wanted


def test_full_layer_installs_request_local_decode_context():
    _, _, leases, _, _, stack = _fixture()
    recorder = _DecodeContextRecordingFullMixer([])
    stack.layers[1].full_attention = recorder
    slot_mapping = torch.tensor([10, 20], dtype=torch.int64)
    context_lens = torch.tensor([3, 5], dtype=torch.int32)
    block_tables = torch.tensor([[1, 2, -1], [4, 5, 6]], dtype=torch.int32)
    set_context(
        False,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        block_tables=block_tables,
        kv_offload_logical_block_tables=[
            [1, 2],
            [4, 5, 6],
        ],
        kv_offload_context_lens=[3, 5],
        kv_offload_write_blocks=[2, 6],
    )
    original_context = get_context()
    try:
        stack(
            leases[:2],
            (1, 1),
            torch.arange(2),
            _hidden()[:2],
        )
        assert get_context() is original_context
    finally:
        reset_context()

    assert len(recorder.contexts) == 2
    expected = (
        {
            "slot_mapping": slot_mapping[:1],
            "context_lens": context_lens[:1],
            "block_tables": block_tables[:1],
            "kv_offload_logical_block_tables": [[1, 2]],
            "kv_offload_context_lens": [3],
            "kv_offload_write_blocks": [2, 6],
        },
        {
            "slot_mapping": slot_mapping[1:2],
            "context_lens": context_lens[1:2],
            "block_tables": block_tables[1:2],
            "kv_offload_logical_block_tables": [[4, 5, 6]],
            "kv_offload_context_lens": [5],
            "kv_offload_write_blocks": [2, 6],
        },
    )
    for actual, wanted in zip(recorder.contexts, expected):
        for name in wanted:
            torch.testing.assert_close(actual[name], wanted[name])


def test_full_layer_installs_request_local_spec_verify_context():
    _, _, leases, _, _, stack = _fixture()
    recorder = _SpecVerifyContextRecordingFullMixer([])
    stack.layers[1].full_attention = recorder
    slot_mapping = torch.tensor(
        [10, 11, 12, 20, 21, 22],
        dtype=torch.int64,
    )
    context_lens = torch.tensor([7, 11], dtype=torch.int32)
    block_tables = torch.tensor(
        [[1, 2, -1], [4, 5, 6]],
        dtype=torch.int32,
    )
    set_context(
        mode="spec_verify",
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        block_tables=block_tables,
        spec_verify_query_lens=(3, 3),
        kv_offload_logical_block_tables=[
            [1, 2],
            [4, 5, 6],
        ],
        kv_offload_context_lens=[7, 11],
        kv_offload_write_blocks=[2, 6],
    )
    original_context = get_context()
    try:
        stack(
            leases[:2],
            (3, 3),
            torch.arange(6),
            _hidden(),
        )
        assert get_context() is original_context
    finally:
        reset_context()

    assert len(recorder.contexts) == 2
    expected = (
        {
            "mode": "spec_verify",
            "slot_mapping": slot_mapping[:3],
            "context_lens": context_lens[:1],
            "block_tables": block_tables[:1],
            "spec_verify_query_lens": (3,),
            "kv_offload_logical_block_tables": [[1, 2]],
            "kv_offload_context_lens": [7],
            "kv_offload_write_blocks": [2, 6],
        },
        {
            "mode": "spec_verify",
            "slot_mapping": slot_mapping[3:],
            "context_lens": context_lens[1:2],
            "block_tables": block_tables[1:2],
            "spec_verify_query_lens": (3,),
            "kv_offload_logical_block_tables": [[4, 5, 6]],
            "kv_offload_context_lens": [11],
            "kv_offload_write_blocks": [2, 6],
        },
    )
    for actual, wanted in zip(recorder.contexts, expected):
        assert actual.pop("mode") == wanted.pop("mode")
        assert (
            actual.pop("spec_verify_query_lens")
            == wanted.pop("spec_verify_query_lens")
        )
        for name in wanted:
            torch.testing.assert_close(actual[name], wanted[name])


def test_linear_layer_installs_request_local_exact_restore_prefill_context():
    request_count = 8
    token_count = 64
    events = []
    layout = HybridStateLayout((
        HybridStateComponentSpec(
            0,
            "linear_convolution",
            (4, 3),
            torch.float32,
        ),
        HybridStateComponentSpec(
            0,
            "linear_recurrent",
            (2, 3, 2),
            torch.float32,
        ),
    ))
    pool = HybridStateTensorPool(layout, request_count, "cpu")
    leases = tuple(
        HybridStateLease(slot, 1, 100 + slot)
        for slot in range(request_count)
    )
    for lease in leases:
        pool.activate(lease)
    adapter = Qwen35LayerStateAdapter(pool, 0)
    mixer = _PrefillContextValidatingLinearMixer(
        "linear0",
        events,
        1.0,
    )
    stack = Qwen35PackedHeterogeneousLayerStack(
        (_decoder("linear_attention", 0, events, mixer),),
        Qwen35CrossLayerStateTransaction((adapter,)),
    )
    mlp = _PrefillContextValidatingMLP(
        "mlp0",
        events,
        0.2,
        0.0,
    )
    stack.layers[0].mlp = mlp
    total_tokens = request_count * token_count
    cu_seqlens = torch.arange(
        0,
        total_tokens + 1,
        token_count,
        dtype=torch.int32,
    )
    set_context(
        True,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens.clone(),
        max_seqlen_q=token_count,
        max_seqlen_k=token_count,
    )
    original_context = get_context()
    try:
        output = stack(
            leases,
            (token_count,) * request_count,
            torch.arange(total_tokens),
            torch.zeros(total_tokens, 4),
        )
        assert get_context() is original_context
    finally:
        reset_context()

    assert output.shape == (total_tokens, 4)
    assert len(mixer.contexts) == request_count
    assert len(mlp.contexts) == request_count
    for context in mixer.contexts:
        torch.testing.assert_close(
            context["cu_seqlens_q"],
            torch.tensor([0, token_count], dtype=torch.int32),
        )
        torch.testing.assert_close(
            context["cu_seqlens_k"],
            torch.tensor([0, token_count], dtype=torch.int32),
        )
        assert context["max_seqlen_q"] == token_count
        assert context["max_seqlen_k"] == token_count
    for context in mlp.contexts:
        torch.testing.assert_close(
            context["cu_seqlens_q"],
            torch.tensor([0, token_count], dtype=torch.int32),
        )
        torch.testing.assert_close(
            context["cu_seqlens_k"],
            torch.tensor([0, token_count], dtype=torch.int32),
        )


def test_linear_layer_installs_request_local_decode_context():
    _, _, leases, _, _, stack = _fixture()
    recorder = _DecodeContextRecordingLinearMixer(
        "linear0",
        [],
        1.0,
    )
    stack.layers[0].linear_attention = recorder
    slot_mapping = torch.tensor([10, 20], dtype=torch.int64)
    context_lens = torch.tensor([3, 5], dtype=torch.int32)
    block_tables = torch.tensor(
        [[1, 2, -1], [4, 5, 6]],
        dtype=torch.int32,
    )
    set_context(
        False,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        block_tables=block_tables,
    )
    original_context = get_context()
    try:
        stack(
            leases[:2],
            (1, 1),
            torch.arange(2),
            _hidden()[:2],
        )
        assert get_context() is original_context
    finally:
        reset_context()

    assert len(recorder.contexts) == 2
    expected = (
        {
            "slot_mapping": slot_mapping[:1],
            "context_lens": context_lens[:1],
            "block_tables": block_tables[:1],
        },
        {
            "slot_mapping": slot_mapping[1:2],
            "context_lens": context_lens[1:2],
            "block_tables": block_tables[1:2],
        },
    )
    for actual, wanted in zip(recorder.contexts, expected):
        for name in wanted:
            torch.testing.assert_close(actual[name], wanted[name])


def _expect_error(function, error_type, message):
    try:
        function()
    except error_type as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(f"expected {error_type.__name__}: {message}")


def test_constructor_alignment_and_later_failures_preserve_all_state():
    events, pool, leases, adapters, mixers, stack = _fixture()
    bad_transaction = Qwen35CrossLayerStateTransaction((adapters[1], adapters[0]))
    _expect_error(
        lambda: Qwen35PackedHeterogeneousLayerStack(
            tuple(stack.layers),
            bad_transaction,
        ),
        ValueError,
        "linear layer indices",
    )
    _expect_error(
        lambda: Qwen35PackedHeterogeneousLayerStack(
            tuple(stack.layers[:2]),
            stack.state_transaction,
        ),
        ValueError,
        "linear layer indices",
    )
    for failure_target in ("full", "linear", "mlp", "candidate", "commit"):
        _, pool, leases, adapters, mixers, stack = _fixture()
        original = _snap(pool)
        error_type = RuntimeError
        if failure_target == "full":
            mixers[1].fail_on_call = 2
            expected = "full1 failure"
        elif failure_target == "linear":
            mixers[2].fail_on_call = 1
            expected = "linear2 failure"
        elif failure_target == "mlp":
            stack.layers[2].mlp.fail_on_call = 1
            expected = "mlp2 failure"
        elif failure_target == "candidate":
            mixers[2].invalid_candidate_on_call = 1
            error_type = ValueError
            expected = "recurrent_state shape"
        else:
            original_copy = adapters[1]._copy_component
            calls = []
            def failing_copy(destination, source):
                calls.append(destination)
                if len(calls) == 2:
                    raise RuntimeError("stack commit failure")
                return original_copy(destination, source)
            adapters[1]._copy_component = failing_copy
            expected = "stack commit failure"
        _expect_error(
            lambda: stack(leases, (2, 1, 3), torch.arange(6), _hidden()),
            error_type,
            expected,
        )
        current = _snap(pool)
        for current_pair, original_pair in zip(current, original):
            torch.testing.assert_close(current_pair[0], original_pair[0])
            torch.testing.assert_close(current_pair[1], original_pair[1])


def main():
    test_schedule_output_state_and_call_order()
    test_full_layer_request_isolation_and_bfloat16_noncontiguous()
    test_constructor_alignment_and_later_failures_preserve_all_state()
    print("qwen35 packed heterogeneous layer stack tests passed")


if __name__ == "__main__":
    main()
