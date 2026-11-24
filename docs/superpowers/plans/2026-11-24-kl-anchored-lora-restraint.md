# Gate G7: Route B KL-Anchored Standard LoRA Restraint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and execute Route B (`kl_anchored_lora_ablation.py`), training a 100% standard `14.62 MiB` `rank-8` `(A, B)` LoRA on `V01..V08` (`64` steps = `8` epochs, `lr=2e-4 -> 2e-5`, `weight_decay=0.01`) jointly anchored by exact full-vocabulary forward KL divergence (`lambda_kl=2.0`) against frozen base-model self-distillation trajectories on 16 unrelated background tasks (`R01..R16`), auditing out-of-domain KL restraint on unseen `D01..D04`, and evaluating 24-task transfer/retention accuracy and latency on `confirmatory_v2_r1`.

**Architecture:** `kl_anchored_lora_ablation.py` prepares sanitized base-model greedy anchor trajectories (`<= 128` tokens) and exact BF16 target-slice base logits on CPU for `R01..R16` and held-out `D01..D04`, trains standard `LoRAProjection` wrappers via sequential gradient accumulation (`1` supervision CE pass + `2` background KL passes per step), audits post-training KL drift and top-1 token agreement on `D01..D04`, runs the 26-episode schedule (`2` `base_canary` + `24` `lora_kl` inside a single `mounted_converged_adapter` block) with zero private oracles in Phase 1, scores all `416` positions in Phase 2 `bwrap`+`seccomp` CPU sandboxes, and produces paired comparisons against `base`, `lora` (`16`-step), `lora_conv` (`96`-step unconstrained), and `text`.

**Tech Stack:** Python `3.12` stdlib, PyTorch `2.5+` (`F.kl_div(..., log_target=True, reduction="batchmean")`), `bwrap` + `seccomp` isolated task scorer.

## Global Constraints

- Zero `#` comments (`tokenize.COMMENT`) and zero single-letter AST identifiers across all Python files.
- Preserve all 205 baseline files, all Gate G4/G5 files, and all Gate G6-A artifacts byte-for-byte.
- Zero `oracles.json`, zero `H01..H06`, and zero `v1` ground-truth reference solutions in Phase 1.
- Standard `LoRAProjection` tensor layout (`72` FP32 tensors, `15,335,424` bytes) unchanged at inference time.

---

### Task 1: KL-Anchored LoRA Trainer, Out-of-Domain Drift Auditor & Orchestrator (`kl_anchored_lora_ablation.py`)

**Files:**
- Create: `experiments/lora_necessity/kl_anchored_lora_ablation.py`
- Test: `experiments/lora_necessity/test_kl_anchored_lora_ablation.py`

- [ ] **Step 1: Write failing unit & PyTorch gradient tests (`test_kl_anchored_lora_ablation.py`)**
  Verify AST/comment hygiene, zero-oracle Phase 1 payload, paired scoring against `base`/`lora_conv`/`text`, and real PyTorch tiny-Qwen3 training proving `lambda_kl=2.0` converges supervision CE loss while suppressing background KL divergence by `>= 5x` relative to `lambda_kl=0.0`.
- [ ] **Step 2: Run RED test check**
- [ ] **Step 3: Implement `kl_anchored_lora_ablation.py`**
- [ ] **Step 4: Run isolated stdlib (37 tests) and PyTorch venv suites to verify GREEN**

---

### Task 2: Gate G7 Package Audit, Single-Shot Live A100 Execution & Comprehensive Report

**Files:**
- Create (private `/private/tmp/`): Gate G7 package, run root, and caller capture root (`0700`/`0600`)
- Create (repo redacted): `artifacts/lora_necessity_v2/20261124-restraint-ablation/gate-g7-kl-anchored-ablation-report.json`, `experiments/lora_necessity/20261124-lora-restraint-ablation-results.md`

- [ ] **Step 1: Materialize & locally audit Gate G7 launch package (byte reconstruction, zero oracle leak, redaction, valid Kerberos ticket, 205 baseline files intact)**
- [ ] **Step 2: Execute single-shot Gate G7 on 1× idle A100 (Phase 1 anchor prep + 64-step KL-LoRA training + D01..D04 audit + 52 greedy generations; Phase 2 isolated 8-case CPU scoring)**
- [ ] **Step 3: Verify canary bit-exactness, archive redacted G7 report, and write complete Route A + Route B empirical report**
