from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import pytest
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


for package_name in (
    "tinyvllm",
    "tinyvllm.engine",
    "tinyvllm.layers",
    "tinyvllm.models",
):
    package = types.ModuleType(package_name)
    package.__path__ = [
        str(ROOT / package_name.replace(".", "/"))
    ]
    sys.modules[package_name] = package

hybrid_module = _load_module(
    "tinyvllm.engine.hybrid_state",
    "tinyvllm/engine/hybrid_state.py",
)
adapter_module = _load_module(
    "tinyvllm.engine.qwen35_layer_state",
    "tinyvllm/engine/qwen35_layer_state.py",
)
transaction_module = _load_module(
    "tinyvllm.engine.qwen35_state_transaction",
    "tinyvllm/engine/qwen35_state_transaction.py",
)
decoder_module = _load_module(
    "tinyvllm.layers.qwen35_decoder_layer",
    "tinyvllm/layers/qwen35_decoder_layer.py",
)
_load_module(
    "tinyvllm.layers.qwen35_packed_stateful_decoder_layer",
    "tinyvllm/layers/qwen35_packed_stateful_decoder_layer.py",
)
stack_module = _load_module(
    "tinyvllm.layers.qwen35_packed_layer_stack",
    "tinyvllm/layers/qwen35_packed_layer_stack.py",
)
root_module = _load_module(
    "tinyvllm.models.qwen35_packed",
    "tinyvllm/models/qwen35_packed.py",
)

HybridStateComponentSpec = hybrid_module.HybridStateComponentSpec
HybridStateLayout = hybrid_module.HybridStateLayout
HybridStateLease = hybrid_module.HybridStateLease
HybridStateTensorPool = hybrid_module.HybridStateTensorPool
Qwen35LayerStateAdapter = adapter_module.Qwen35LayerStateAdapter
Qwen35CrossLayerStateTransaction = (
    transaction_module.Qwen35CrossLayerStateTransaction
)
Qwen35DecoderLayerShell = decoder_module.Qwen35DecoderLayerShell
Qwen35PackedHeterogeneousLayerStack = (
    stack_module.Qwen35PackedHeterogeneousLayerStack
)
Qwen35PackedForCausalLM = root_module.Qwen35PackedForCausalLM


class _Identity(nn.Module):
    def __init__(self):
        super().__init__()
        self.fail = False

    def forward(self, tensor):
        if self.fail:
            raise RuntimeError("injected final norm failure")
        return tensor


class _Zero(nn.Module):
    def forward(self, tensor):
        return torch.zeros_like(tensor)


class _Embedding(nn.Module):
    def forward(self, input_ids):
        values = input_ids.to(torch.float32)
        return torch.stack((values, values + 1), dim=-1)


class _Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.fail = False

    def forward(self, hidden):
        if self.fail:
            raise RuntimeError("injected lm head failure")
        return torch.stack((
            hidden.sum(dim=-1),
            hidden[:, 0] - hidden[:, 1],
        ), dim=-1)


class _LinearMixer(nn.Module):
    def forward(self, hidden, convolution, recurrent):
        delta = hidden.sum().to(convolution.dtype)
        return (
            hidden * 0.5,
            convolution + delta,
            recurrent + delta * 0.25,
        )

    def forward_with_state_trace(
        self,
        hidden,
        convolution,
        recurrent,
    ):
        cumulative = hidden.reshape(
            hidden.shape[0],
            -1,
        ).sum(dim=-1).cumsum(dim=0).to(convolution.dtype)
        convolution_trace = torch.stack(tuple(
            convolution + value
            for value in cumulative
        ))
        recurrent_trace = torch.stack(tuple(
            recurrent + value * 0.25
            for value in cumulative
        ))
        trace = types.SimpleNamespace(
            convolution=convolution_trace,
            recurrent=recurrent_trace,
        )
        return (
            hidden * 0.5,
            convolution_trace[-1],
            recurrent_trace[-1],
            trace,
        )


def _fixture():
    layout = HybridStateLayout((
        HybridStateComponentSpec(
            0,
            "linear_convolution",
            (2, 2),
            torch.float32,
        ),
        HybridStateComponentSpec(
            0,
            "linear_recurrent",
            (2, 2, 2),
            torch.float32,
        ),
    ))
    pool = HybridStateTensorPool(layout, 2, "cpu")
    leases = (
        HybridStateLease(0, 1, 101),
        HybridStateLease(1, 1, 102),
    )
    for lease in leases:
        pool.activate(lease)
    pool.component_tensor(0, "linear_convolution").copy_(
        torch.arange(
            8,
            dtype=torch.float32,
        ).reshape(2, 2, 2)
    )
    pool.component_tensor(0, "linear_recurrent").copy_(
        torch.arange(
            16,
            dtype=torch.float32,
        ).reshape(2, 2, 2, 2)
    )
    adapter = Qwen35LayerStateAdapter(pool, 0)
    transaction = Qwen35CrossLayerStateTransaction((adapter,))
    decoder = Qwen35DecoderLayerShell(
        block_type="linear_attention",
        input_layernorm=_Identity(),
        post_attention_layernorm=_Identity(),
        mlp=_Zero(),
        linear_attention=_LinearMixer(),
    )
    stack = Qwen35PackedHeterogeneousLayerStack(
        (decoder,),
        transaction,
    )
    final_norm = _Identity()
    head = _Head()
    model = Qwen35PackedForCausalLM(
        _Embedding(),
        stack,
        final_norm,
        head,
    )
    return pool, leases, final_norm, head, model


def _snapshot(pool):
    return tuple(
        tensor.clone()
        for tensor in pool._tensors.values()
    )


def _assert_snapshot(pool, expected):
    for actual, value in zip(pool._tensors.values(), expected):
        torch.testing.assert_close(actual, value)


def _inputs():
    return (
        (2,),
        torch.tensor([3, 5], dtype=torch.int64),
        torch.tensor([0, 1], dtype=torch.int64),
    )


def test_prepare_step_is_read_only_and_commit_applies_candidates():
    pool, leases, _, _, model = _fixture()
    token_counts, input_ids, position_ids = _inputs()
    before = _snapshot(pool)

    prepared = model.prepare_step(
        (leases[0],),
        token_counts,
        input_ids,
        position_ids,
    )

    _assert_snapshot(pool, before)
    assert prepared.normalized.shape == (2, 2)
    assert prepared.logits.shape == (2, 2)
    assert prepared.state == "prepared"

    model.commit_prepared_step((leases[0],), prepared)

    assert prepared.state == "committed"
    assert any(
        not torch.equal(actual, expected)
        for actual, expected in zip(
            pool._tensors.values(),
            before,
        )
    )


def test_prepare_step_uses_initial_candidates_and_returns_prefix_states():
    pool, leases, _, _, model = _fixture()
    token_counts, input_ids, position_ids = _inputs()
    before = _snapshot(pool)
    initial_candidates = tuple(
        (
            convolution + 100,
            recurrent + 200,
        )
        for convolution, recurrent
        in model.layer_stack.state_transaction.gather((leases[0],))
    )

    prepared = model.prepare_step(
        (leases[0],),
        token_counts,
        input_ids,
        position_ids,
        initial_candidates=initial_candidates,
        capture_prefix_states=True,
    )

    _assert_snapshot(pool, before)
    assert prepared.prefix_candidates is not None
    assert tuple(
        len(sequence_prefixes)
        for sequence_prefixes in prepared.prefix_candidates
    ) == token_counts
    assert len(prepared.prefix_candidates[0][-1]) == 1
    torch.testing.assert_close(
        prepared.prefix_candidates[0][-1][0][0],
        prepared.final_candidates[0][0][0],
    )
    torch.testing.assert_close(
        prepared.prefix_candidates[0][-1][0][1],
        prepared.final_candidates[0][1][0],
    )
    assert torch.all(
        prepared.final_candidates[0][0]
        > before[0][0][:1]
    )
    assert torch.all(
        prepared.final_candidates[0][1]
        > before[0][1][:1]
    )


def test_prepared_step_rejects_second_commit_and_different_leases():
    _, leases, _, _, model = _fixture()
    token_counts, input_ids, position_ids = _inputs()
    prepared = model.prepare_step(
        (leases[0],),
        token_counts,
        input_ids,
        position_ids,
    )

    with pytest.raises(ValueError, match="lease identity"):
        model.commit_prepared_step((leases[1],), prepared)

    model.commit_prepared_step((leases[0],), prepared)
    with pytest.raises(RuntimeError, match="committed"):
        model.commit_prepared_step((leases[0],), prepared)


def test_run_step_matches_prepare_then_commit():
    pool_run, leases_run, _, _, run_model = _fixture()
    pool_prepared, leases_prepared, _, _, prepared_model = _fixture()
    token_counts, input_ids, position_ids = _inputs()

    run_hidden, run_logits = run_model.run_step(
        (leases_run[0],),
        token_counts,
        input_ids,
        position_ids,
    )
    prepared = prepared_model.prepare_step(
        (leases_prepared[0],),
        token_counts,
        input_ids,
        position_ids,
    )
    prepared_model.commit_prepared_step(
        (leases_prepared[0],),
        prepared,
    )

    torch.testing.assert_close(run_hidden, prepared.normalized)
    torch.testing.assert_close(run_logits, prepared.logits)
    for run_state, prepared_state in zip(
        pool_run._tensors.values(),
        pool_prepared._tensors.values(),
    ):
        torch.testing.assert_close(run_state, prepared_state)


def test_exact_cuda_graph_state_hooks_are_complete_and_lease_sealed():
    pool, leases, _, _, model = _fixture()
    before = _snapshot(pool)

    assert (
        model.exact_cuda_graph_state_schema_sha256()
        == pool.layout.fingerprint
    )
    first = model.exact_cuda_graph_lease_seal((leases[0], leases[1]))
    assert first == model.exact_cuda_graph_lease_seal(
        (leases[0], leases[1])
    )
    assert first != model.exact_cuda_graph_lease_seal(
        (leases[1], leases[0])
    )

    snapshot = model.snapshot_exact_cuda_graph_state((leases[0],))
    for tensor in pool._tensors.values():
        tensor[leases[0].slot_id].add_(1000)
    model.restore_exact_cuda_graph_state((leases[0],), snapshot)
    _assert_snapshot(pool, before)

    pool.release(leases[0])
    changed_generation = HybridStateLease(
        leases[0].slot_id,
        leases[0].generation + 1,
        leases[0].request_id,
    )
    pool.activate(changed_generation)
    assert first != model.exact_cuda_graph_lease_seal(
        (changed_generation, leases[1])
    )


def test_exact_cuda_graph_step_matches_run_step_output_and_state():
    graph_pool, graph_leases, _, _, graph_model = _fixture()
    eager_pool, eager_leases, _, _, eager_model = _fixture()
    token_counts, input_ids, position_ids = _inputs()

    graph_logits = graph_model.run_exact_cuda_graph_step(
        (graph_leases[0],),
        token_counts,
        input_ids,
        position_ids,
    )
    _, eager_logits = eager_model.run_step(
        (eager_leases[0],),
        token_counts,
        input_ids,
        position_ids,
    )

    torch.testing.assert_close(graph_logits, eager_logits)
    for graph_state, eager_state in zip(
        graph_pool._tensors.values(),
        eager_pool._tensors.values(),
    ):
        torch.testing.assert_close(graph_state, eager_state)


@pytest.mark.parametrize("failure", ("norm", "head"))
def test_prepare_step_failure_leaves_live_state_unchanged(failure):
    pool, leases, final_norm, head, model = _fixture()
    token_counts, input_ids, position_ids = _inputs()
    before = _snapshot(pool)
    if failure == "norm":
        final_norm.fail = True
    else:
        head.fail = True

    with pytest.raises(RuntimeError, match=f"injected .* failure"):
        model.prepare_step(
            (leases[0],),
            token_counts,
            input_ids,
            position_ids,
        )

    _assert_snapshot(pool, before)
