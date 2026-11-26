from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

import torch
from torch import nn


@dataclass(frozen=True)
class PromotedPrefixEntry:
    slot_id: int
    long_prefix: tuple[int, ...]
    short_prefix: tuple[int, ...]
    name: str


class PrefixLoRARegistry:
    def __init__(self, max_slots: int):
        if isinstance(max_slots, bool) or not isinstance(max_slots, int) or max_slots < 0:
            raise ValueError("max_slots must be a non-negative integer")
        self.max_slots = max_slots
        self._by_slot: dict[int, PromotedPrefixEntry] = {}
        self._by_prefix: dict[tuple[int, ...], PromotedPrefixEntry] = {}
        self._sorted_lengths: list[int] = []

    def _rebuild_lengths(self) -> None:
        self._sorted_lengths = sorted({len(prefix) for prefix in self._by_prefix}, reverse=True)

    def _validate_slot(self, slot_id: int) -> int:
        if isinstance(slot_id, bool) or not isinstance(slot_id, int):
            raise ValueError("slot_id must be an integer")
        if self.max_slots <= 0 or not (1 <= slot_id <= self.max_slots):
            raise ValueError(f"slot_id must be in [1, {self.max_slots}]")
        return slot_id

    @staticmethod
    def _normalize_tokens(tokens: Sequence[int], label: str) -> tuple[int, ...]:
        out = tuple(tokens)
        if not out or any(isinstance(tok, bool) or not isinstance(tok, int) or tok < 0 for tok in out):
            raise ValueError(f"{label} must be a non-empty sequence of non-negative integer token IDs")
        return out

    def register_promoted_prefix(
        self,
        slot_id: int,
        long_prefix_ids: Sequence[int],
        short_prefix_ids: Sequence[int] = (),
        *,
        name: str = "",
    ) -> PromotedPrefixEntry:
        valid_slot = self._validate_slot(slot_id)
        long_tup = self._normalize_tokens(long_prefix_ids, "long_prefix_ids")
        short_tup = tuple(short_prefix_ids)
        if any(isinstance(tok, bool) or not isinstance(tok, int) or tok < 0 for tok in short_tup):
            raise ValueError("short_prefix_ids must contain non-negative integer token IDs")
        if len(long_tup) <= len(short_tup):
            raise ValueError("long_prefix_ids must be strictly longer than short_prefix_ids")
        existing = self._by_prefix.get(long_tup)
        if existing is not None and existing.slot_id != valid_slot:
            raise ValueError(f"long_prefix_ids already bound to slot {existing.slot_id}")

        if valid_slot in self._by_slot:
            old_entry = self._by_slot.pop(valid_slot)
            self._by_prefix.pop(old_entry.long_prefix, None)

        entry = PromotedPrefixEntry(
            slot_id=valid_slot,
            long_prefix=long_tup,
            short_prefix=short_tup,
            name=str(name),
        )
        self._by_slot[valid_slot] = entry
        self._by_prefix[long_tup] = entry
        self._rebuild_lengths()
        return entry

    def unregister_slot(self, slot_id: int) -> None:
        valid_slot = self._validate_slot(slot_id)
        old_entry = self._by_slot.pop(valid_slot, None)
        if old_entry is not None:
            self._by_prefix.pop(old_entry.long_prefix, None)
            self._rebuild_lengths()

    def match_and_rewrite(
        self,
        prompt_token_ids: list[int],
        explicit_slot_id: int = 0,
    ) -> tuple[list[int], int]:
        if isinstance(explicit_slot_id, bool) or not isinstance(explicit_slot_id, int) or explicit_slot_id < 0:
            raise ValueError("explicit_slot_id must be a non-negative integer")
        if explicit_slot_id > 0:
            self._validate_slot(explicit_slot_id)
            return list(prompt_token_ids), explicit_slot_id
        if not self._sorted_lengths:
            return list(prompt_token_ids), 0

        prompt_len = len(prompt_token_ids)
        for cand_len in self._sorted_lengths:
            if prompt_len <= cand_len:
                continue
            head_key = tuple(prompt_token_ids[:cand_len])
            matched = self._by_prefix.get(head_key)
            if matched is not None:
                rewritten = list(matched.short_prefix) + list(prompt_token_ids[cand_len:])
                return rewritten, matched.slot_id
        return list(prompt_token_ids), 0


class QKVLoRASlotPool(nn.Module):
    def __init__(
        self,
        max_slots: int,
        rank: int,
        hidden_size: int,
        q_shard_size: int,
        kv_shard_size: int,
        *,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        if max_slots <= 0 or rank <= 0:
            raise ValueError("max_slots and rank must be positive integers")
        self.max_slots = int(max_slots)
        self.rank = int(rank)
        self.hidden_size = int(hidden_size)
        self.q_shard_size = int(q_shard_size)
        self.kv_shard_size = int(kv_shard_size)
        self.total_out_size = self.q_shard_size + 2 * self.kv_shard_size

        total_rows = self.max_slots + 1
        self.register_buffer(
            "lora_a_q",
            torch.zeros(total_rows, self.rank, self.hidden_size, dtype=dtype, device=device),
            persistent=False,
        )
        self.register_buffer(
            "lora_b_q",
            torch.zeros(total_rows, self.q_shard_size, self.rank, dtype=dtype, device=device),
            persistent=False,
        )
        self.register_buffer(
            "lora_a_v",
            torch.zeros(total_rows, self.rank, self.hidden_size, dtype=dtype, device=device),
            persistent=False,
        )
        self.register_buffer(
            "lora_b_v",
            torch.zeros(total_rows, self.kv_shard_size, self.rank, dtype=dtype, device=device),
            persistent=False,
        )

    def _check_slot(self, slot_id: int) -> int:
        if isinstance(slot_id, bool) or not isinstance(slot_id, int) or not (1 <= slot_id <= self.max_slots):
            raise ValueError(f"slot_id must be in [1, {self.max_slots}]; slot 0 is immutable sentinel")
        return slot_id

    @torch.no_grad()
    def load_slot(
        self,
        slot_id: int,
        a_q: torch.Tensor,
        b_q: torch.Tensor,
        a_v: torch.Tensor,
        b_v: torch.Tensor,
        *,
        scale: float = 2.0,
    ) -> None:
        valid_slot = self._check_slot(slot_id)
        if tuple(a_q.shape) != (self.rank, self.hidden_size) or tuple(b_q.shape) != (self.q_shard_size, self.rank):
            raise ValueError("Q LoRA tensor shape mismatch")
        if tuple(a_v.shape) != (self.rank, self.hidden_size) or tuple(b_v.shape) != (self.kv_shard_size, self.rank):
            raise ValueError("V LoRA tensor shape mismatch")
        target_dtype = self.lora_a_q.dtype
        target_dev = self.lora_a_q.device
        self.lora_a_q[valid_slot].copy_(a_q.to(device=target_dev, dtype=target_dtype))
        self.lora_b_q[valid_slot].copy_((b_q.float() * float(scale)).to(device=target_dev, dtype=target_dtype))
        self.lora_a_v[valid_slot].copy_(a_v.to(device=target_dev, dtype=target_dtype))
        self.lora_b_v[valid_slot].copy_((b_v.float() * float(scale)).to(device=target_dev, dtype=target_dtype))
        self.lora_a_q[0].zero_()
        self.lora_b_q[0].zero_()
        self.lora_a_v[0].zero_()
        self.lora_b_v[0].zero_()

    @torch.no_grad()
    def clear_slot(self, slot_id: int) -> None:
        valid_slot = self._check_slot(slot_id)
        self.lora_a_q[valid_slot].zero_()
        self.lora_b_q[valid_slot].zero_()
        self.lora_a_v[valid_slot].zero_()
        self.lora_b_v[valid_slot].zero_()

    def apply(
        self,
        x: torch.Tensor,
        base_qkv: torch.Tensor,
        slot_ids: torch.Tensor,
    ) -> torch.Tensor:
        orig_out_shape = base_qkv.shape
        x_flat = x.reshape(-1, self.hidden_size)
        out_flat = base_qkv.reshape(-1, self.total_out_size)
        num_tokens = x_flat.shape[0]
        slot_long = slot_ids[:num_tokens].to(device=x_flat.device, dtype=torch.long)

        work_dtype = self.lora_a_q.dtype
        x_col = x_flat.to(dtype=work_dtype).unsqueeze(-1)

        aq_sel = torch.index_select(self.lora_a_q, 0, slot_long)
        bq_sel = torch.index_select(self.lora_b_q, 0, slot_long)
        dq = torch.bmm(bq_sel, torch.bmm(aq_sel, x_col)).squeeze(-1).to(dtype=out_flat.dtype)

        av_sel = torch.index_select(self.lora_a_v, 0, slot_long)
        bv_sel = torch.index_select(self.lora_b_v, 0, slot_long)
        dv = torch.bmm(bv_sel, torch.bmm(av_sel, x_col)).squeeze(-1).to(dtype=out_flat.dtype)

        v_offset = self.q_shard_size + self.kv_shard_size
        out_q = out_flat[:, :self.q_shard_size] + dq
        out_k = out_flat[:, self.q_shard_size:v_offset]
        out_v = out_flat[:, v_offset:] + dv
        merged = torch.cat((out_q, out_k, out_v), dim=-1)
        return merged.reshape(orig_out_shape)


def _discover_qkv_modules(model: nn.Module) -> list[tuple[int, nn.Module]]:
    discovered: list[tuple[int, nn.Module]] = []
    pattern = re.compile(r"(?:^|\.)layers\.(\d+)\.self_attn\.qkv_proj$")
    for mod_name, mod_obj in model.named_modules():
        matched = pattern.search(mod_name)
        if matched is not None:
            discovered.append((int(matched.group(1)), mod_obj))
    discovered.sort(key=lambda pair: pair[0])
    if not discovered:
        raise ValueError("no layers.{idx}.self_attn.qkv_proj modules found on model")
    expected_indices = list(range(len(discovered)))
    if [idx for idx, _ in discovered] != expected_indices:
        raise ValueError("non-contiguous layer indices in discovered qkv_proj modules")
    return discovered


def install_qkv_lora_pools(model: nn.Module, max_slots: int, rank: int) -> int:
    discovered = _discover_qkv_modules(model)
    for _layer_idx, qkv_mod in discovered:
        weight_param = qkv_mod.weight
        q_shard = int(qkv_mod.num_heads * qkv_mod.head_size)
        kv_shard = int(qkv_mod.num_kv_heads * qkv_mod.head_size)
        hidden_sz = int(weight_param.shape[1])
        qkv_mod.qkv_lora_pool = QKVLoRASlotPool(
            max_slots=max_slots,
            rank=rank,
            hidden_size=hidden_sz,
            q_shard_size=q_shard,
            kv_shard_size=kv_shard,
            dtype=weight_param.dtype,
            device=weight_param.device,
        )
    return len(discovered)


@torch.no_grad()
def load_hf_lora_into_tinyvllm_slot(
    model: nn.Module,
    slot_id: int,
    adapter_tensors: dict[str, torch.Tensor],
    *,
    alpha: float = 16.0,
    rank: int = 8,
) -> dict[str, int]:
    discovered = _discover_qkv_modules(model)
    scale = float(alpha) / float(rank)
    loaded_bytes = 0
    for layer_idx, qkv_mod in discovered:
        pool = getattr(qkv_mod, "qkv_lora_pool", None)
        if pool is None:
            raise ValueError("install_qkv_lora_pools must be called before loading slots")
        q_pref = f"model.layers.{layer_idx}.self_attn.q_proj"
        v_pref = f"model.layers.{layer_idx}.self_attn.v_proj"
        a_q = adapter_tensors[f"{q_pref}.adapter_a"]
        b_q_full = adapter_tensors[f"{q_pref}.adapter_b"]
        a_v = adapter_tensors[f"{v_pref}.adapter_a"]
        b_v_full = adapter_tensors[f"{v_pref}.adapter_b"]

        tp_size = int(getattr(qkv_mod, "tp_size", 1))
        tp_rank = int(getattr(qkv_mod, "tp_rank", 0))
        b_q_shard = b_q_full.chunk(tp_size, dim=0)[tp_rank]
        b_v_shard = b_v_full.chunk(tp_size, dim=0)[tp_rank]
        pool.load_slot(slot_id, a_q, b_q_shard, a_v, b_v_shard, scale=scale)
        loaded_bytes += (
            (a_q.numel() + b_q_shard.numel() + a_v.numel() + b_v_shard.numel())
            * pool.lora_a_q.element_size()
        )
    return {
        "slot_id": int(slot_id),
        "layers_loaded": len(discovered),
        "slot_resident_bytes": loaded_bytes,
    }


@torch.no_grad()
def clear_tinyvllm_lora_slot(model: nn.Module, slot_id: int) -> None:
    for _layer_idx, qkv_mod in _discover_qkv_modules(model):
        pool = getattr(qkv_mod, "qkv_lora_pool", None)
        if pool is not None:
            pool.clear_slot(slot_id)
