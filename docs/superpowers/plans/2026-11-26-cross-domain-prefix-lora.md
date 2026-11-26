# Cross-Domain Parametric Prefix LoRA & Multi-Slot Co-Serving Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and execute a zero-tuning single-shot blind cross-domain evaluation (`med_triage`, `legal_compliance`, `code_bugfix`, `general_retention`) and `tinyvllm` 4-domain multi-slot co-serving benchmark on `Qwen3-8B`.

**Architecture:** Deterministic pure-Python SOP rule engines generate disjoint seed (`8`), unlabeled (`20`), and offline-sealed blind test (`16`) sets for Medical and Legal SOPs (`~3,100` tokens each), paired with `16` multimodal background anchors and `16` blind general-reasoning holdout tasks. A domain-agnostic trainer applies JSON-value surgical CE masking + Teacher soft KL self-distillation + null-space background KL under locked Gate-G12 hyperparameters (`rank=8, alpha=16, steps=96, lr=2.2e-4`), followed by an A100 `4 × 4` cross-domain interference matrix and native `tinyvllm.LLMEngine` 4-domain CUDA Graph mixed-batch verification.

**Tech Stack:** Python 3.12, PyTorch 2.4+ (`bfloat16`), HuggingFace `AutoTokenizer` / `AutoModelForCausalLM` (`Qwen3-8B`), `tinyvllm` (`PrefixLoRARegistry`, `QKVLoRASlotPool`, Paged KV Cache, FlashAttention, CUDA Graphs), `bwrap` + `seccomp` sandbox.

## Global Constraints

- Work exclusively inside isolated worktree `/Users/bytedance/dev/TinyLLMForge/.trae/worktrees/cross-domain-prefix-lora` on branch `experiments/cross-domain-prefix-lora`.
- Pre-registered locked hyperparameters: `rank=8, alpha=16, targets=["q_proj", "v_proj"], steps=96, learning_rate=0.00022, lambda_pos_kl=1.5, lambda_seed_surgical_ce=0.55, lambda_bg_kl=2.5, diff_token_weight=4.0, scaffold_token_weight=1.0` — zero post-hoc tuning on blind test sets.
- `blind_oracles.json` (ground-truth answers for `M_T01..M_T16`, `L_T01..L_T16`, `G_R01..G_R16`) MUST remain strictly local; `zero_blind_oracle_in_gpu_payload` MUST evaluate to `True` before any remote GPU dispatch.
- Reuse sealed Gate G12 code adapter (`artifact_sha256 = 4d03f75048c1646dc7c8f32bf943d02e2009c57d4d1ea55a1ff5b6cc6d09fa80`) without retraining.
- Zero internal hostnames, usernames, IPs, numeric UIDs, GPU UUIDs, or remote user paths in committed code, reports, or JSON artifacts.

---

### Task 1: Deterministic SOP Rule Engines, Blind Dataset Seal, and Scorer

**Files:**
- Create: `experiments/cross_domain_prefix_lora/__init__.py`
- Create: `experiments/cross_domain_prefix_lora/domain_sop_contracts.py`
- Create: `experiments/cross_domain_prefix_lora/test_cross_domain_contracts.py`
- Create (generated & pinned): `experiments/cross_domain_prefix_lora/data/cross_domain_v1/{public_bundle.json, blind_oracles.json, manifest.json}`

**Interfaces:**
- Produces:
  - `med_sop_evaluate(case_input: dict) -> dict` returning `{"renal_tier": str, "action": str, "regimen": str, "dose_mg": int, "rule_id": str}`
  - `legal_sop_evaluate(case_input: dict) -> dict` returning `{"risk_grade": str, "routing": str, "min_cap_x": int, "cure_days": int, "primary_rule": str}`
  - `build_and_verify_cross_domain_dataset() -> dict` returning `{"public_bundle": dict, "blind_oracles": dict, "manifest": dict}`
  - `render_domain_long_messages(domain: str, case_obj: dict, public_bundle: dict, exclude_seed_id: str | None = None) -> list[dict]`
  - `render_domain_short_messages(domain: str, case_obj: dict) -> list[dict]`
  - `score_structured_sop_prediction(domain: str, raw_text: str, expected_dict: dict) -> dict`
  - `score_general_retention_prediction(task_obj: dict, raw_text: str, expected_answer: str) -> dict`
  - `detect_cross_domain_contamination(target_domain: str, raw_text: str) -> bool`

- [ ] **Step 1: Write failing contract tests in `test_cross_domain_contracts.py`**

```python
import json
from pathlib import Path
import unittest

from experiments.cross_domain_prefix_lora import domain_sop_contracts as contracts


class CrossDomainContractsTests(unittest.TestCase):
    def test_med_and_legal_rule_engines_deterministic_and_nontrivial(self):
        built = contracts.build_and_verify_cross_domain_dataset()
        pub = built["public_bundle"]
        orc = built["blind_oracles"]
        self.assertEqual(len(pub["med_triage"]["seeds"]), 8)
        self.assertEqual(len(pub["med_triage"]["unlabeled"]), 20)
        self.assertEqual(len(pub["med_triage"]["blind_test_inputs"]), 16)
        self.assertEqual(len(pub["legal_compliance"]["blind_test_inputs"]), 16)
        self.assertEqual(len(pub["general_retention"]["blind_test_inputs"]), 16)
        self.assertEqual(len(pub["background_anchors"]), 16)
        pub_raw = json.dumps(pub, sort_keys=True)
        for tid, ans in orc["med_triage"].items():
            self.assertIn(tid, {x["id"] for x in pub["med_triage"]["blind_test_inputs"]})
            self.assertNotIn(json.dumps(ans, sort_keys=True), pub_raw)

    def test_leave_one_out_and_contamination_detectors(self):
        built = contracts.build_and_verify_cross_domain_dataset()
        pub = built["public_bundle"]
        s01 = pub["med_triage"]["seeds"][0]
        loo_msgs = contracts.render_domain_long_messages(
            "med_triage", s01, pub, exclude_seed_id=s01["id"]
        )
        self.assertNotIn(s01["id"], loo_msgs[0]["content"])
        self.assertTrue(
            contracts.detect_cross_domain_contamination(
                "general_retention", '{"renal_tier": "R4_SEVERE", "dose_mg": 500}'
            )
        )
        self.assertFalse(
            contracts.detect_cross_domain_contamination("general_retention", "FINAL_ANSWER: 42")
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/homebrew/bin/python3.12 -B -m unittest experiments/cross_domain_prefix_lora/test_cross_domain_contracts.py -v`
Expected: FAIL (`ImportError` / `ModuleNotFoundError`)

- [ ] **Step 3: Implement `domain_sop_contracts.py` and materialize `data/cross_domain_v1/*.json`**

Implement full `MED-SOP-2026-V4` and `LEGAL-SOP-2026-V3` manuals (`~3,100` Qwen3 tokens each), disjoint case generators (`M_S01..08`, `M_U01..20`, `M_T01..16`, `L_S01..08`, `L_U01..20`, `L_T01..16`, `BG01..16`, `G_R01..16`), JSON schema parsers, and SHA256-sealed files on disk.

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/homebrew/bin/python3.12 -B -m unittest experiments/cross_domain_prefix_lora/test_cross_domain_contracts.py -v`
Expected: PASS

---

### Task 2: Domain-Agnostic JSON Value Surgical Masker & Locked Self-Distillation Trainer

**Files:**
- Create: `experiments/cross_domain_prefix_lora/surgical_json_csd_trainer.py`
- Create: `experiments/cross_domain_prefix_lora/test_surgical_json_csd_trainer.py`

**Interfaces:**
- Consumes: `render_domain_long_messages`, `render_domain_short_messages` from `domain_sop_contracts.py`; `LoRAProjection`, `_compute_trajectory_kl_loss` from `experiments/lora_necessity/`.
- Produces:
  - `LOCKED_CROSS_DOMAIN_HYPERPARAMS: dict`
  - `find_json_value_char_spans(canonical_json_str: str) -> list[tuple[int, int]]`
  - `build_json_value_surgical_weights(tokenizer, canonical_json_str: str, target_ids: list[int], diff_kl_weight: float = 4.0) -> tuple[list[float], list[float], int]` returning `(kl_weights, surgical_ce_weights, value_token_count)`
  - `validate_unlabeled_sop_candidate(domain: str, raw_text: str) -> str | None`
  - `prepare_domain_distillation_trajectories(model, tokenizer, domain: str, public_bundle: dict, hyperparams: dict | None = None) -> tuple[list[dict], dict]`
  - `train_domain_sop_lora_adapter(model, positive_trajectories: list[dict], background_trajectories: list[dict], hyperparams: dict | None = None) -> dict`

- [ ] **Step 1: Write failing trainer & masker tests in `test_surgical_json_csd_trainer.py`**

```python
import unittest
from experiments.cross_domain_prefix_lora import surgical_json_csd_trainer as trainer


class SurgicalJsonCSDTrainerTests(unittest.TestCase):
    def test_json_value_surgical_mask_zeros_keys_and_braces(self):
        sample = '{"renal_tier": "R4_SEVERE", "action": "HOLD_STAT", "regimen": "NONE", "dose_mg": 0, "rule_id": "MED-R01"}'
        spans = trainer.find_json_value_char_spans(sample)
        extracted = [sample[lo:hi] for lo, hi in spans]
        self.assertEqual(extracted, ['"R4_SEVERE"', '"HOLD_STAT"', '"NONE"', "0", '"MED-R01"'])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/homebrew/bin/python3.12 -B -m unittest experiments/cross_domain_prefix_lora/test_surgical_json_csd_trainer.py -v`
Expected: FAIL (`ImportError`)

- [ ] **Step 3: Implement `surgical_json_csd_trainer.py` + tiny-Qwen3 PyTorch training test**

Ensure `surgical_ce_weights` assigns `0.0` to `{`, `}`, JSON keys, `:`, `,`, whitespace, and EOS, normalizing positive weights over value tokens to mean `1.0`; verify base weights remain bit-exact after training.

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/homebrew/bin/python3.12 -B -m unittest experiments/cross_domain_prefix_lora/test_surgical_json_csd_trainer.py -v`
Expected: PASS

---

### Task 3: Blind GPU Execution & `4 × 4` Cross-Domain Interference Matrix

**Files:**
- Create: `experiments/cross_domain_prefix_lora/cross_domain_remote_runner.py`
- Create: `experiments/cross_domain_prefix_lora/score_cross_domain_blind_run.py`

**Interfaces:**
- Consumes: `public_bundle.json` (blind inputs ONLY; zero `blind_oracles.json` on GPU!), `surgical_json_csd_trainer.py`, sealed Gate G12 `hybrid_diff_csd_adapter.pt`.
- Produces:
  - Remote sealed BF16/FP32 adapters `med_sop_adapter.pt` and `legal_sop_adapter.pt`
  - Blind raw generation outputs for:
    1) In-domain `text` (`~3,400` tok SOP prefix) on `M_T01..16` and `L_T01..16`
    2) Full `4 × 4` adapter-by-domain matrix: `[base, code_lora, med_lora, legal_lora] × [code_wt12, med_t16, legal_t16, general_r16]`
  - Offline scored artifact `artifacts/cross_domain_v1/cross-domain-blind-matrix-report.json`

- [ ] **Step 1: Implement `cross_domain_remote_runner.py` and `score_cross_domain_blind_run.py` with programmatic payload leak guard**
- [ ] **Step 2: Execute single-shot blind run on idle remote A100 + `bwrap+seccomp` sandbox for `WT01..WT12`**
- [ ] **Step 3: Verify `zero_blind_oracle_in_gpu_payload == True` and archive redacted JSON report**

---

### Task 4: Native `tinyvllm` 4-Domain Multi-Slot Co-Serving Benchmark & Final Report

**Files:**
- Create: `tools/verify_tinyvllm_4domain_coserving.py`
- Create: `experiments/cross_domain_prefix_lora/20261126-cross-domain-blind-and-coserving-results.md`

**Interfaces:**
- Consumes: `tinyvllm.engine.llm_engine.LLMEngine`, sealed `code_g12` (`Slot 1`), `med_sop` (`Slot 2`), and `legal_sop` (`Slot 3`) adapters.
- Produces:
  - `artifacts/cross_domain_v1/tinyvllm-4domain-coserving-e2e-report.json`
  - `experiments/cross_domain_prefix_lora/20261126-cross-domain-blind-and-coserving-results.md`

- [ ] **Step 1: Implement `tools/verify_tinyvllm_4domain_coserving.py`**
  Register all 3 domain `~3,100`-token prefixes into `LLMEngine` (`Slot 1, 2, 3`, `21.94 MiB` total BF16 residency), run 12-request 4-domain mixed CUDA Graph equivalence test (`3 × 4`), and benchmark 32-request 4-domain mixed burst (`8 × 4`) vs native `tinyvllm` 3-domain Warm APC.
- [ ] **Step 2: Execute single-shot A100 `tinyvllm` 4-domain benchmark and archive report**
- [ ] **Step 3: Run full local unit test suite & sensitive-string leak scan**
