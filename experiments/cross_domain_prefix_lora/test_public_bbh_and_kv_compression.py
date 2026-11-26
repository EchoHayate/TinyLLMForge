from __future__ import annotations

import importlib.util
import unittest

from experiments.cross_domain_prefix_lora.public_bbh_and_kv_compression_study import (
    build_public_bbh_suite,
    exact_mcnemar_pvalue,
    quantize_dequantize_tensor_int4_group32,
    quantize_dequantize_tensor_int8_per_token,
    score_bbh_prediction,
    select_snapkv_sparse_prefix_indices,
)


class PublicBBHAndKVCompressionTests(unittest.TestCase):
    def test_bbh_suite_disjoint_and_exact_mcnemar(self):
        suite = build_public_bbh_suite(seed=2026)
        self.assertEqual(len(suite["exemplars"]), 12)
        self.assertEqual(len(suite["train_unlabeled"]), 64)
        self.assertEqual(len(suite["test_items"]), 120)

        ex_q = {x["question"] for x in suite["exemplars"]}
        tr_q = {x["question"] for x in suite["train_unlabeled"]}
        te_q = {x["question"] for x in suite["test_items"]}
        self.assertEqual(len(ex_q & te_q), 0)
        self.assertEqual(len(tr_q & te_q), 0)

        for ex in suite["exemplars"]:
            self.assertTrue(score_bbh_prediction(ex["cot_target"], ex["answer"]))

        self.assertAlmostEqual(exact_mcnemar_pvalue(3, 2), 1.0, places=4)
        self.assertLess(exact_mcnemar_pvalue(28, 3), 1e-5)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires PyTorch")
    def test_kv_quantizers_and_snapkv_selector(self):
        import torch

        torch.manual_seed(9)
        kv = torch.randn(1, 8, 200, 128, dtype=torch.bfloat16)
        deq8, b8 = quantize_dequantize_tensor_int8_per_token(kv)
        deq4, b4 = quantize_dequantize_tensor_int4_group32(kv)
        self.assertEqual(deq8.shape, kv.shape)
        self.assertEqual(deq4.shape, kv.shape)
        self.assertLess(b8, kv.numel() * 2 * 0.55)
        self.assertLess(b4, kv.numel() * 2 * 0.32)
        self.assertLess(float((deq8.float() - kv.float()).abs().mean()), 0.02)

        scores = torch.rand(1, 8, 16, 200)
        scores[:, :, :, 77] = 10.0
        kept = select_snapkv_sparse_prefix_indices(scores, 200, keep_ratio=0.15, sink_tokens=16)
        self.assertEqual(kept.shape[0], 30)
        self.assertIn(0, kept.tolist())
        self.assertIn(77, kept.tolist())


if __name__ == "__main__":
    unittest.main()
