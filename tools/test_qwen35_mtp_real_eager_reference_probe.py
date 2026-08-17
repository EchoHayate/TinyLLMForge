from __future__ import annotations

import math
from pathlib import Path
import sys
import types

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = ROOT / "tools"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TOOLS_DIR))
for package_name in (
    "tinyvllm",
    "tinyvllm.engine",
    "tinyvllm.layers",
    "tinyvllm.speculative",
):
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / package_name.replace(".", "/"))]
    sys.modules.setdefault(package_name, package)

import qwen35_mtp_real_checkpoint_gate as gate_module
from tinyvllm.engine.proposal_kv_cache import ProposalKVCache
from tinyvllm.engine.qwen35_mtp_executor import (
    Qwen35MTPProposalExecutor,
)
from tinyvllm.engine.qwen35_mtp_registration import (
    Qwen35MTPPhysicalSlotStore,
)
from tinyvllm.layers import qwen35_full_attention
from tinyvllm.utils.context import get_context


class _AttentionBackedMTP:

    def __init__(self, store, vocab_size: int = 32):
        self.store = store
        self.vocab_size = vocab_size

    def forward_hidden(self, input_ids, positions, hidden_states):
        del positions
        context = get_context()
        token_count = int(input_ids.shape[0])
        query = hidden_states.reshape(token_count, 2, 2)
        key = (
            hidden_states[:, :2]
            .reshape(token_count, 1, 2)
        )
        value = (
            hidden_states[:, 2:]
            .reshape(token_count, 1, 2)
        )
        if context.is_prefill:
            attention_output = (
                qwen35_full_attention
                .qwen35_prefill_eager_attention(
                    query,
                    key,
                    value,
                    context,
                    num_heads=2,
                    head_dim=2,
                    scale=2 ** -0.5,
                    key_cache=self.store.key_cache,
                    value_cache=self.store.value_cache,
                )
            )
        else:
            attention_output = (
                qwen35_full_attention
                .qwen35_cached_decode_eager_attention(
                    query,
                    key,
                    value,
                    self.store.key_cache,
                    self.store.value_cache,
                    context,
                    num_heads=2,
                    head_dim=2,
                    scale=2 ** -0.5,
                )
            )
        return attention_output

    def forward_step(self, input_ids, positions, hidden_states):
        attention_output = self.forward_hidden(
            input_ids,
            positions,
            hidden_states,
        )
        token_count = int(input_ids.shape[0])
        logits = torch.full(
            (token_count, self.vocab_size),
            -1000.0,
            dtype=torch.float32,
            device=hidden_states.device,
        )
        positive_token = torch.full(
            (token_count, 1),
            7,
            dtype=torch.int64,
            device=hidden_states.device,
        )
        negative_token = torch.full_like(positive_token, 11)
        next_token = torch.where(
            attention_output[:, :1] >= 0,
            positive_token,
            negative_token,
        )
        logits.scatter_(1, next_token, 1000.0)
        return attention_output, logits


def _runtime():
    store = Qwen35MTPPhysicalSlotStore(
        capacity=256,
        num_kv_heads=1,
        head_dim=2,
        dtype=torch.float32,
        device="cpu",
    )
    module = _AttentionBackedMTP(store)
    executor = Qwen35MTPProposalExecutor(
        module=module,
        proposal_kv_cache=ProposalKVCache(store),
        max_proposal_tokens=4,
    )
    hf_config = types.SimpleNamespace(
        text_config=types.SimpleNamespace(
            hidden_size=4,
            vocab_size=module.vocab_size,
        )
    )
    return executor, module, store, hf_config


@pytest.mark.parametrize(
    ("q", "batch_size"),
    (
        (1, 1),
        (4, 1),
        (2, 4),
        (4, 4),
    ),
)
def test_real_eager_reference_probe_matches_independent_attention(
    q,
    batch_size,
):
    executor, module, store, hf_config = _runtime()
    build_probe = getattr(
        gate_module,
        "_build_real_eager_reference_probe",
        None,
    )
    assert callable(build_probe)
    probe = build_probe(
        executor=executor,
        module=module,
        physical_store=store,
        hf_config=hf_config,
    )

    result = probe(q, batch_size)

    assert result["argmax_equal"] is True
    assert result["max_abs_diff"] >= 0.0
    assert math.isfinite(result["max_abs_diff"])
    assert all(
        store.is_allocated(slot_id) is False
        for slot_id in range(store.capacity)
    )


def test_real_eager_reference_probe_detects_reference_corruption(
    monkeypatch,
):
    executor, module, store, hf_config = _runtime()
    reference_decode = getattr(
        gate_module,
        "_qwen35_reference_cached_decode_attention",
        None,
    )
    assert callable(reference_decode)

    def corrupted_reference(*args, **kwargs):
        return -reference_decode(*args, **kwargs)

    monkeypatch.setattr(
        gate_module,
        "_qwen35_reference_cached_decode_attention",
        corrupted_reference,
    )
    probe = gate_module._build_real_eager_reference_probe(
        executor=executor,
        module=module,
        physical_store=store,
        hf_config=hf_config,
    )

    result = probe(4, 1)

    assert result["argmax_equal"] is False
    assert all(
        store.is_allocated(slot_id) is False
        for slot_id in range(store.capacity)
    )

