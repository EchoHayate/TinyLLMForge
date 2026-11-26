from __future__ import annotations

import importlib.util
import types
import unittest


class MockCharTokenizer:
    eos_token_id = 999

    def __init__(self):
        self._s2i = {}
        self._i2s = {}

    def _tok(self, ch_val: str) -> int:
        if ch_val not in self._s2i:
            nid = len(self._s2i) + 1
            self._s2i[ch_val] = nid
            self._i2s[nid] = ch_val
        return self._s2i[ch_val]

    def encode_answer(self, text_val: str) -> list[int]:
        return [self._tok(ch_val) for ch_val in text_val] + [self.eos_token_id]

    def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) -> str:
        return "".join(self._i2s[tid] for tid in token_ids if tid != self.eos_token_id)


class SurgicalJsonCSDTrainerTests(unittest.TestCase):
    def test_json_value_surgical_mask_zeros_keys_braces_and_eos(self):
        from experiments.cross_domain_prefix_lora import surgical_json_csd_trainer as trainer

        sample = (
            '{"renal_tier": "R4_SEVERE", "action": "HOLD_STAT", '
            '"regimen": "NONE", "dose_mg": 0, "rule_id": "MED-R01"}'
        )
        spans = trainer.find_json_value_char_spans(sample)
        extracted = [sample[lo:hi] for lo, hi in spans]
        self.assertEqual(extracted, ['"R4_SEVERE"', '"HOLD_STAT"', '"NONE"', "0", '"MED-R01"'])

        tok = MockCharTokenizer()
        target_ids = tok.encode_answer(sample)
        kl_wts, surg_wts, val_cnt = trainer.build_json_value_surgical_weights(
            tok, sample, target_ids, diff_kl_weight=4.0
        )
        self.assertEqual(len(kl_wts), len(target_ids))
        self.assertEqual(len(surg_wts), len(target_ids))
        self.assertGreater(val_cnt, 25)
        self.assertAlmostEqual(sum(kl_wts) / len(kl_wts), 1.0, places=6)
        self.assertAlmostEqual(sum(surg_wts) / len(surg_wts), 1.0, places=6)
        self.assertEqual(surg_wts[0], 0.0)
        self.assertEqual(surg_wts[-1], 0.0)
        key_pos = sample.index("renal_tier")
        self.assertEqual(surg_wts[key_pos + 2], 0.0)
        val_pos = sample.index("R4_SEVERE")
        self.assertGreater(surg_wts[val_pos + 2], 1.5)

        valid_med = trainer.validate_unlabeled_sop_candidate("med_triage", sample)
        self.assertEqual(valid_med, sample)
        invalid_med = trainer.validate_unlabeled_sop_candidate(
            "med_triage",
            '{"renal_tier": "INVALID_TIER", "action": "HOLD_STAT", "regimen": "NONE", "dose_mg": 0, "rule_id": "MED-R01"}',
        )
        self.assertIsNone(invalid_med)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires PyTorch")
    def test_train_domain_sop_lora_adapter_preserves_base_and_converges(self):
        import sys
        from pathlib import Path
        import torch

        repo_lora = Path(__file__).resolve().parents[1] / "lora_necessity"
        if str(repo_lora) not in sys.path:
            sys.path.insert(0, str(repo_lora))
        from confirmatory_v2_converged_ablation import mounted_converged_adapter
        from training_worker import _state, _tensor_digest
        from experiments.cross_domain_prefix_lora import surgical_json_csd_trainer as trainer

        class TinyQwen3(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = types.SimpleNamespace(
                    model_type="qwen3", num_hidden_layers=1, return_dict=True, use_cache=False
                )
                self.embedding = torch.nn.Embedding(32, 16)
                self.model = torch.nn.Module()
                layer = torch.nn.Module()
                layer.self_attn = torch.nn.Module()
                layer.self_attn.q_proj = torch.nn.Linear(16, 16)
                layer.self_attn.v_proj = torch.nn.Linear(16, 16)
                self.model.layers = torch.nn.ModuleList([layer])
                self.lm_head = torch.nn.Linear(16, 32)

            def forward(self, input_ids, attention_mask=None, use_cache=False, return_dict=True, past_key_values=None):
                hidden = self.embedding(input_ids)
                attn = self.model.layers[0].self_attn
                hidden = hidden + attn.q_proj(hidden) + attn.v_proj(hidden)
                return types.SimpleNamespace(logits=self.lm_head(hidden), past_key_values=None)

        torch.manual_seed(17)
        tiny = TinyQwen3().to(dtype=torch.bfloat16).eval().requires_grad_(False)
        before_sha = _tensor_digest(_state(tiny))

        pos_trajs = []
        for idx_n in range(10):
            s_ids = [1, 3 + (idx_n % 5), 14, 26, 15]
            t_log = torch.zeros(3, 32, dtype=torch.bfloat16)
            t_log[0, 14] = 4.0
            t_log[1, 26] = 6.0
            t_log[2, 15] = 4.0
            is_seed = idx_n < 8
            pos_trajs.append({
                "task_id": f"M_P{idx_n:02d}",
                "source_pool": "seed" if is_seed else "unlabeled_verified",
                "student_full_ids": s_ids,
                "student_prompt_len": 2,
                "answer_len": 3,
                "target_ids": [14, 26, 15],
                "target_weights": [0.5, 2.0, 0.5],
                "surgical_ce_weights": [0.0, 3.0, 0.0] if is_seed else [0.0, 0.0, 0.0],
                "teacher_answer_logits_bf16": t_log,
            })

        bg_trajs = []
        with torch.no_grad():
            for bg_i in range(16):
                b_ids = [1, 11 + (bg_i % 5), 21, 22, 23]
                inp_t = torch.tensor([b_ids[:-1]], dtype=torch.long)
                b_sl = tiny(input_ids=inp_t).logits[0, 1:4].detach().clone()
                bg_trajs.append({
                    "task_id": f"BG{bg_i:02d}",
                    "full_ids": b_ids,
                    "prompt_len": 2,
                    "answer_len": 3,
                    "base_answer_logits_bf16": b_sl,
                })

        out = trainer.train_domain_sop_lora_adapter(
            tiny,
            pos_trajs,
            bg_trajs,
            hyperparams={**trainer.LOCKED_CROSS_DOMAIN_HYPERPARAMS, "learning_rate": 0.02, "steps": 32},
        )
        self.assertEqual(_tensor_digest(_state(tiny)), before_sha)
        seed_steps = [st for st in out["steps"] if st["source_pool"] == "seed"]
        self.assertLess(seed_steps[-1]["surgical_ce_loss"], seed_steps[0]["surgical_ce_loss"] * 0.5)
        with mounted_converged_adapter(tiny, out) as adapted:
            self.assertIsNotNone(adapted)


if __name__ == "__main__":
    unittest.main()
