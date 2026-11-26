import os
import pickle
import sys
import types
import unittest
from unittest import mock
import torch
from torch import nn

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if "tinyvllm" not in sys.modules:
    _pkg = types.ModuleType("tinyvllm")
    _pkg.__path__ = [os.path.join(_REPO_ROOT, "tinyvllm")]
    sys.modules["tinyvllm"] = _pkg

from tinyvllm.lora import (
    LinearLoRASlotPool,
    PrefixLoRARegistry,
    QKVLoRASlotPool,
    install_mlp_down_lora_pools,
    install_qkv_lora_pools,
    load_hf_lora_into_tinyvllm_slot,
)
from tinyvllm.sampling_params import SamplingParams
from tinyvllm.engine.sequence import Sequence
from tinyvllm.utils.context import get_context, reset_context, set_context
from tinyvllm.layers.linear import QKVParallelLinear, RowParallelLinear


class PrefixLoRARuntimeUnitTests(unittest.TestCase):
    def test_prefix_registry_longest_match_rewrite_and_validation(self):
        registry = PrefixLoRARegistry(max_slots=4)
        long_prefix = [10, 11, 12, 13, 14, 15]
        short_prefix = [10, 11]
        registry.register_promoted_prefix(
            1, long_prefix, short_prefix, name="code_repair_v2"
        )

        prompt_hit = [10, 11, 12, 13, 14, 15, 90, 91, 92]
        rewritten, slot_id = registry.match_and_rewrite(prompt_hit)
        self.assertEqual(slot_id, 1)
        self.assertEqual(rewritten, [10, 11, 90, 91, 92])

        prompt_miss = [10, 11, 99, 90, 91]
        untouched, miss_slot = registry.match_and_rewrite(prompt_miss)
        self.assertEqual(miss_slot, 0)
        self.assertEqual(untouched, prompt_miss)

        explicit_prompt, exp_slot = registry.match_and_rewrite(prompt_hit, explicit_slot_id=2)
        self.assertEqual(exp_slot, 2)
        self.assertEqual(explicit_prompt, prompt_hit)

        with self.assertRaises(ValueError):
            registry.register_promoted_prefix(0, [1, 2, 3], [1])
        with self.assertRaises(ValueError):
            registry.register_promoted_prefix(5, [1, 2, 3], [1])
        with self.assertRaises(ValueError):
            registry.register_promoted_prefix(2, [1, 2], [1, 2, 3])

    def test_qkv_lora_slot_pool_mixed_batch_exactness_and_zero_slot_immutability(self):
        torch.manual_seed(7)
        hidden_size = 16
        q_size = 16
        kv_size = 8
        rank = 4
        pool = QKVLoRASlotPool(
            max_slots=3,
            rank=rank,
            hidden_size=hidden_size,
            q_shard_size=q_size,
            kv_shard_size=kv_size,
            dtype=torch.float32,
            device="cpu",
        )

        a_q1 = torch.randn(rank, hidden_size) * 0.2
        b_q1 = torch.randn(q_size, rank) * 0.2
        a_v1 = torch.randn(rank, hidden_size) * 0.2
        b_v1 = torch.randn(kv_size, rank) * 0.2
        pool.load_slot(1, a_q1, b_q1, a_v1, b_v1, scale=2.0)

        a_q2 = torch.randn(rank, hidden_size) * 0.3
        b_q2 = torch.randn(q_size, rank) * 0.3
        a_v2 = torch.randn(rank, hidden_size) * 0.3
        b_v2 = torch.randn(kv_size, rank) * 0.3
        pool.load_slot(2, a_q2, b_q2, a_v2, b_v2, scale=2.0)

        x_batch = torch.randn(4, hidden_size, dtype=torch.float32)
        base_qkv = torch.randn(4, q_size + 2 * kv_size, dtype=torch.float32)
        slot_ids = torch.tensor([0, 1, 2, 0], dtype=torch.int64)

        out_qkv = pool.apply(x_batch, base_qkv.clone(), slot_ids)

        self.assertTrue(torch.equal(out_qkv[0], base_qkv[0]))
        self.assertTrue(torch.equal(out_qkv[3], base_qkv[3]))
        self.assertTrue(torch.equal(out_qkv[:, q_size:q_size + kv_size], base_qkv[:, q_size:q_size + kv_size]))

        expected_dq1 = 2.0 * (x_batch[1] @ a_q1.t() @ b_q1.t())
        expected_dv1 = 2.0 * (x_batch[1] @ a_v1.t() @ b_v1.t())
        self.assertTrue(torch.allclose(out_qkv[1, :q_size], base_qkv[1, :q_size] + expected_dq1, atol=1e-5))
        self.assertTrue(torch.allclose(out_qkv[1, q_size + kv_size:], base_qkv[1, q_size + kv_size:] + expected_dv1, atol=1e-5))

        expected_dq2 = 2.0 * (x_batch[2] @ a_q2.t() @ b_q2.t())
        expected_dv2 = 2.0 * (x_batch[2] @ a_v2.t() @ b_v2.t())
        self.assertTrue(torch.allclose(out_qkv[2, :q_size], base_qkv[2, :q_size] + expected_dq2, atol=1e-5))
        self.assertTrue(torch.allclose(out_qkv[2, q_size + kv_size:], base_qkv[2, q_size + kv_size:] + expected_dv2, atol=1e-5))

    def test_sequence_sampling_params_and_pickle_preserve_lora_slot_id(self):
        default_seq = Sequence([10, 20, 30], SamplingParams())
        self.assertEqual(default_seq.lora_slot_id, 0)
        self.assertEqual(len(default_seq.__getstate__()), 17)
        restored_default = pickle.loads(pickle.dumps(default_seq))
        self.assertEqual(restored_default.lora_slot_id, 0)

        lora_seq = Sequence([10, 20, 30], SamplingParams(temperature=0.0, max_tokens=32, lora_slot_id=3))
        self.assertEqual(lora_seq.lora_slot_id, 3)
        restored_lora = pickle.loads(pickle.dumps(lora_seq))
        self.assertEqual(restored_lora.lora_slot_id, 3)

    def test_qkv_parallel_linear_end_to_end_context_hook(self):
        class MiniDecoderModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Module()
                layer = nn.Module()
                layer.self_attn = nn.Module()
                layer.self_attn.qkv_proj = QKVParallelLinear(16, 8, 2, 1, bias=False)
                self.model.layers = nn.ModuleList([layer])

        torch.manual_seed(19)
        with mock.patch("torch.distributed.get_world_size", return_value=1), mock.patch(
            "torch.distributed.get_rank", return_value=0
        ):
            mini = MiniDecoderModel().to(dtype=torch.float32)
        nn.init.normal_(mini.model.layers[0].self_attn.qkv_proj.weight, std=0.05)

        installed = install_qkv_lora_pools(mini, max_slots=2, rank=4)
        self.assertEqual(installed, 1)

        hf_tensors = {
            "model.layers.0.self_attn.q_proj.adapter_a": torch.randn(4, 16) * 0.2,
            "model.layers.0.self_attn.q_proj.adapter_b": torch.randn(16, 4) * 0.2,
            "model.layers.0.self_attn.v_proj.adapter_a": torch.randn(4, 16) * 0.2,
            "model.layers.0.self_attn.v_proj.adapter_b": torch.randn(8, 4) * 0.2,
        }
        load_hf_lora_into_tinyvllm_slot(mini, 1, hf_tensors, alpha=8.0, rank=4)

        x_in = torch.randn(3, 16, dtype=torch.float32)
        reset_context()
        base_out = mini.model.layers[0].self_attn.qkv_proj(x_in)

        try:
            set_context(True, lora_token_slot_ids=torch.tensor([0, 1, 0], dtype=torch.int64))
            self.assertIsNotNone(get_context().lora_token_slot_ids)
            hooked_out = mini.model.layers[0].self_attn.qkv_proj(x_in)
        finally:
            reset_context()

        self.assertTrue(torch.equal(hooked_out[0], base_out[0]))
        self.assertTrue(torch.equal(hooked_out[2], base_out[2]))
        self.assertFalse(torch.allclose(hooked_out[1, :16], base_out[1, :16]))
        self.assertTrue(torch.equal(hooked_out[1, 16:24], base_out[1, 16:24]))
        self.assertFalse(torch.allclose(hooked_out[1, 24:], base_out[1, 24:]))

    def test_mlp_down_lora_pool_and_heterogeneous_rank_co_residency(self):
        class MiniDecoderWithMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Module()
                layer = nn.Module()
                layer.self_attn = nn.Module()
                layer.self_attn.qkv_proj = QKVParallelLinear(16, 8, 2, 1, bias=False)
                layer.mlp = nn.Module()
                layer.mlp.down_proj = RowParallelLinear(32, 16, bias=False)
                self.model.layers = nn.ModuleList([layer])

        torch.manual_seed(31)
        with mock.patch("torch.distributed.get_world_size", return_value=1), mock.patch(
            "torch.distributed.get_rank", return_value=0
        ):
            mini = MiniDecoderWithMLP().to(dtype=torch.float32)
        nn.init.normal_(mini.model.layers[0].self_attn.qkv_proj.weight, std=0.05)
        nn.init.normal_(mini.model.layers[0].mlp.down_proj.weight, std=0.05)

        install_qkv_lora_pools(mini, max_slots=3, rank=8)
        install_mlp_down_lora_pools(mini, max_slots=3, rank=8)
        self.assertIsInstance(mini.model.layers[0].mlp.down_proj.mlp_lora_pool, LinearLoRASlotPool)

        slot1_r4_qv_only = {
            "model.layers.0.self_attn.q_proj.adapter_a": torch.randn(4, 16) * 0.2,
            "model.layers.0.self_attn.q_proj.adapter_b": torch.randn(16, 4) * 0.2,
            "model.layers.0.self_attn.v_proj.adapter_a": torch.randn(4, 16) * 0.2,
            "model.layers.0.self_attn.v_proj.adapter_b": torch.randn(8, 4) * 0.2,
        }
        load_hf_lora_into_tinyvllm_slot(mini, 1, slot1_r4_qv_only, alpha=8.0, rank=4)

        a_down = torch.randn(8, 32) * 0.25
        b_down = torch.randn(16, 8) * 0.25
        slot2_r8_qv_and_mlp = {
            "model.layers.0.self_attn.q_proj.adapter_a": torch.randn(8, 16) * 0.2,
            "model.layers.0.self_attn.q_proj.adapter_b": torch.randn(16, 8) * 0.2,
            "model.layers.0.self_attn.v_proj.adapter_a": torch.randn(8, 16) * 0.2,
            "model.layers.0.self_attn.v_proj.adapter_b": torch.randn(8, 8) * 0.2,
            "model.layers.0.mlp.down_proj.adapter_a": a_down,
            "model.layers.0.mlp.down_proj.adapter_b": b_down,
        }
        load_hf_lora_into_tinyvllm_slot(mini, 2, slot2_r8_qv_and_mlp, alpha=16.0, rank=8)

        x_mlp = torch.randn(4, 32, dtype=torch.float32)
        reset_context()
        base_mlp = mini.model.layers[0].mlp.down_proj(x_mlp)

        try:
            set_context(True, lora_token_slot_ids=torch.tensor([0, 1, 2, 0], dtype=torch.int64))
            mixed_mlp = mini.model.layers[0].mlp.down_proj(x_mlp)
        finally:
            reset_context()

        self.assertTrue(torch.equal(mixed_mlp[0], base_mlp[0]))
        self.assertTrue(torch.equal(mixed_mlp[1], base_mlp[1]))
        self.assertTrue(torch.equal(mixed_mlp[3], base_mlp[3]))
        expected_delta2 = 2.0 * (x_mlp[2] @ a_down.t() @ b_down.t())
        self.assertTrue(torch.allclose(mixed_mlp[2], base_mlp[2] + expected_delta2, atol=1e-5))

    def test_template_prefix_registry_preserves_dynamic_date_and_user_variables(self):
        registry = PrefixLoRARegistry(max_slots=4)
        head_anchor = [101, 102]
        tail_anchor = [201, 202, 203, 204, 205, 206, 207, 208]
        short_replacement = [901, 902]
        registry.register_template_prefix(
            2,
            head_anchor,
            tail_anchor,
            max_var_tokens=8,
            short_prefix_ids=short_replacement,
            preserve_var_span=True,
            name="med_sop_dynamic_header",
        )

        req_alice = [101, 102, 71, 72, 201, 202, 203, 204, 205, 206, 207, 208, 500, 501]
        rew_a, slot_a = registry.match_and_rewrite(req_alice)
        self.assertEqual(slot_a, 2)
        self.assertEqual(rew_a, [101, 102, 71, 72, 901, 902, 500, 501])

        req_bob = [101, 102, 81, 82, 83, 84, 85, 201, 202, 203, 204, 205, 206, 207, 208, 600]
        rew_b, slot_b = registry.match_and_rewrite(req_bob)
        self.assertEqual(slot_b, 2)
        self.assertEqual(rew_b, [101, 102, 81, 82, 83, 84, 85, 901, 902, 600])

        too_long_var = [101, 102] + list(range(10)) + tail_anchor + [500]
        rew_miss, slot_miss = registry.match_and_rewrite(too_long_var)
        self.assertEqual(slot_miss, 0)
        self.assertEqual(rew_miss, too_long_var)


if __name__ == "__main__":
    unittest.main()
