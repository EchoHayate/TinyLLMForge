from __future__ import annotations

import torch

import tinyvllm.layers.embed_head as embed_head
from tinyvllm.layers.embed_head import ParallelLMHead
from tinyvllm.utils.context import reset_context, set_context


def test_parallel_lm_head_prefill_selects_rows_after_linear(monkeypatch):
    monkeypatch.setattr(embed_head.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(embed_head.dist, "get_world_size", lambda: 1)
    head = ParallelLMHead(4, 2)
    with torch.no_grad():
        head.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                    [-1.0, 1.0],
                ]
            )
        )
    hidden = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 2.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
    )
    linear_input_shapes = []
    real_linear = embed_head.F.linear

    def recording_linear(input_tensor, weight, bias=None):
        linear_input_shapes.append(tuple(input_tensor.shape))
        return real_linear(input_tensor, weight, bias)

    monkeypatch.setattr(embed_head.F, "linear", recording_linear)
    set_context(
        True,
        cu_seqlens_q=torch.tensor([0, 2, 5, 6], dtype=torch.int32),
        logits_indices=torch.tensor([1, 5], dtype=torch.int64),
    )
    try:
        logits = head(hidden)
    finally:
        reset_context()

    expected_all = real_linear(hidden, head.weight)
    assert linear_input_shapes == [(6, 2)]
    assert logits.shape == (2, 4)
    assert torch.equal(logits, expected_all[[1, 5]])


def test_parallel_lm_head_exact_mode_projects_full_vocab_once(monkeypatch):
    monkeypatch.setattr(embed_head.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(embed_head.dist, "get_world_size", lambda: 2)
    head = ParallelLMHead(4, 2, exact_full_vocab=True)
    with torch.no_grad():
        head.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ]
            )
        )

    remote_weight = torch.tensor(
        [
            [1.0, 1.0],
            [-1.0, 1.0],
        ]
    )

    def fake_gather(input_tensor, output_tensors, destination):
        assert destination == 0
        output_tensors[0].copy_(input_tensor)
        output_tensors[1].copy_(remote_weight)

    monkeypatch.setattr(embed_head.dist, "gather", fake_gather)
    linear_weight_shapes = []
    real_linear = embed_head.F.linear

    def recording_linear(input_tensor, weight, bias=None):
        linear_weight_shapes.append(tuple(weight.shape))
        return real_linear(input_tensor, weight, bias)

    monkeypatch.setattr(embed_head.F, "linear", recording_linear)
    hidden = torch.tensor([[2.0, 3.0]])
    set_context(False)
    try:
        logits = head(hidden)
    finally:
        reset_context()

    expected_weight = torch.cat((head.weight, remote_weight), dim=0)
    assert linear_weight_shapes == [(4, 2)]
    assert torch.equal(logits, real_linear(hidden, expected_weight))
