from __future__ import annotations

from pathlib import Path
import sys
import types
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = ROOT / "tools"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TOOLS_DIR))
for package_name in (
    "tinyvllm",
    "tinyvllm.engine",
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
from tinyvllm.utils.context import get_context


class _TensorWritingMTP:

    def __init__(self, store, vocab_size: int = 64):
        self.store = store
        self.vocab_size = vocab_size

    def forward_hidden(self, input_ids, positions, hidden_states):
        context = get_context()
        slot_ids = context.slot_mapping.to(torch.long)
        for row_index, slot_id in enumerate(slot_ids.tolist()):
            value = float(
                int(input_ids[row_index].item())
                + int(positions[..., row_index].reshape(-1)[0].item())
            )
            self.store.key_cache[slot_id, 0].fill_(value)
            self.store.value_cache[slot_id, 0].fill_(value + 0.5)
        output_hidden = hidden_states + input_ids.to(
            hidden_states.dtype
        ).unsqueeze(-1)
        return output_hidden

    def forward_step(self, input_ids, positions, hidden_states):
        output_hidden = self.forward_hidden(
            input_ids,
            positions,
            hidden_states,
        )
        logits = torch.full(
            (input_ids.shape[0], self.vocab_size),
            -1000.0,
            dtype=torch.float32,
            device=input_ids.device,
        )
        next_tokens = (input_ids + 1) % self.vocab_size
        logits.scatter_(1, next_tokens.unsqueeze(-1), 1000.0)
        return output_hidden, logits


def _runtime():
    store = Qwen35MTPPhysicalSlotStore(
        capacity=256,
        num_kv_heads=2,
        head_dim=4,
        dtype=torch.float32,
        device="cpu",
    )
    module = _TensorWritingMTP(store)
    executor = Qwen35MTPProposalExecutor(
        module=module,
        proposal_kv_cache=ProposalKVCache(store),
        max_proposal_tokens=4,
    )
    hf_config = SimpleNamespace(
        text_config=SimpleNamespace(
            hidden_size=4,
            vocab_size=64,
        )
    )
    return executor, module, store, hf_config


@pytest.mark.parametrize(
    ("q", "batch_size", "accepted"),
    (
        (4, 1, 2),
        (3, 4, 0),
        (1, 4, 1),
    ),
)
def test_real_transaction_probe_preserves_identity_and_rollback(
    q,
    batch_size,
    accepted,
):
    executor, module, store, hf_config = _runtime()
    build_probe = getattr(
        gate_module,
        "_build_real_transaction_probe",
        None,
    )
    assert callable(build_probe)
    probe = build_probe(
        executor=executor,
        module=module,
        physical_store=store,
        hf_config=hf_config,
    )

    result = probe(q, batch_size, accepted)

    staged_count = batch_size * max(q - 1, 0)
    committed_count = batch_size * max(accepted - 1, 0)
    assert result["q"] == q
    assert result["batch_size"] == batch_size
    assert result["accepted_proposal_tokens"] == accepted
    assert len(result["staged_slot_ids"]) == staged_count
    assert len(result["committed_slot_ids"]) == committed_count
    assert len(result["released_slot_ids"]) == (
        staged_count - committed_count
    )
    assert result["accepted_slot_identity_preserved"] is True
    assert result["rejected_slots_released"] is True
    assert result["post_rollback_continuation_equal"] is True
    assert all(
        store.is_allocated(slot_id) is False
        for slot_id in range(store.capacity)
    )
