# Gate G4: v2 Confirmatory Matched-Information LoRA Four-Arm Comparison Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construct and validate the unseen 32-task multi-obligation `confirmatory_v2_r1` benchmark (`V01..V08` supervision, `WT01..WT12` transfer, `WR01..WR12` retention), pass real remote `bwrap`+`seccomp` dynamic source validation and cold-process Qwen3-8B tokenizer length audit (`<= 4096` input, `<= 256` output), then execute the single-shot 16-step Qwen3-8B LoRA training + 192-generation four-arm (`base`, `text`, `lora`, `lora_text`) evaluation + 1,536-position isolated scoring and independent offline reconstruction review.

**Architecture:**
- `data/confirmatory_v2_r1/{manifest.json,supervision.json,prompts.json,oracles.json,structure.json}`: 32 multi-obligation pure-Python single-function repair tasks completely disjoint from `C01..C12` and all v1 tasks (`8` supervision across `bounds`/`reset`/`multiplicity`/`order`, `12` transfer across the same 4 families, `12` retention across disjoint `dispatch`/`window`/`balance`/`dedup`).
- `confirmatory_v2_contract.py`: Frozen SHA256 pins, AST/outcome/distinct-input validation, prompt rendering (`first_round`, `second_round`, `training_jobs`), dynamic validation batch builder (`32 × 2 references + 32 × 3 incorrect variants`), and tokenizer audit projection builder (`8` training rows + `96` first-round + `96` reference + `96` sampled second-round = `296` rows).
- `confirmatory_v2_training.py`: Exact v2 counterpart to `training_worker.train_adapter` + `training_inputs.prepare_training` bound to `V01..V08`, producing `LORA_TRAINING_CORE_RESULT_UNATTESTED` compatible with frozen `adapter_artifact.{save_adapter,load_adapter,mounted_adapter}`.
- `confirmatory_v2_scoring.py`, `confirmatory_v2_remote.py`, `confirmatory_v2_controller.py`: Counterbalanced 96-episode / 192-generation schedule (`base`, `text`, `lora`, `lora_text`), remote Phase 1 GPU trainer + generator + `bwrap`/`seccomp` public-case runner (zero private oracles in Phase 1), Phase 2 `close_fds=True` isolated 8-case scorer (`<= 48` batches of 32), paired round-1 and round-2 four-arm comparison (`lora_text` vs `text`, `lora` vs `text`, `lora` vs `base`), retention regression check, and exact training/inference wall-time break-even amortization.

**Global Constraints:**
- Preserve all 205 v1/recalibration baseline files and all Gate G1/G2/G3 artifacts byte-for-byte.
- Zero `#` inline comments and zero single-letter AST identifiers in all Python files.
- Complete all 192 generations locally sealed before executing any private scoring case (`H01..H06`).

---

### Task 1: `confirmatory_v2_r1` 32-Task Dataset & Contract (`confirmatory_v2_contract.py`)
- [x] **Step 1: Write failing contract tests (`test_confirmatory_v2_contract.py`)**
- [x] **Step 2: Run RED and save log**
- [x] **Step 3: Generate `data/confirmatory_v2_r1/*` and implement `confirmatory_v2_contract.py`**
- [x] **Step 4: Run GREEN and independent static review**

---

### Task 2: Gate G4-A Live Dynamic Validation & Cold-Process Qwen3-8B Tokenizer Length Audit
- [x] **Step 1: Build and test dynamic validation + 296-row tokenizer audit controller/driver**
- [x] **Step 2: Execute remote `bwrap`+`seccomp` dynamic validation of all 32 tasks (references + witnesses)**
- [x] **Step 3: Execute cold-process Qwen3-8B tokenizer length audit of all 296 rows (`<= 4096` / `<= 256`)**
- [x] **Step 4: Independent post-run verification of Gate G4-A**

---

### Task 3: Gate G4-B LoRA Training, Four-Arm Generator, Scorer & Controller
- [x] **Step 1: Write failing tests (`test_confirmatory_v2_training.py`, `test_confirmatory_v2_scoring_and_controller.py`) and save RED log**
- [x] **Step 2: Implement `confirmatory_v2_training.py`, `confirmatory_v2_scoring.py`, `confirmatory_v2_remote.py`, `confirmatory_v2_controller.py`**
- [x] **Step 3: Run GREEN and build frozen Gate G4-B launch package**
- [x] **Step 4: Independent pre-dispatch code & package review (`APPROVE`, 0 P0/P1/P2)**

---

### Task 4: Gate G4-B Live Execution, Post-Run Independent Reconstruction & Final Report
- [x] **Step 1: Execute single-shot Gate G4-B (16-step Qwen3-8B LoRA training + 192 generations + Phase 2 1,536-item isolated scoring)**
- [x] **Step 2: Independent read-only offline reconstruction & verification (`APPROVE`, 0 P0/P1/P2)**
- [x] **Step 3: Write `experiments/lora_necessity/20261122-lora-v2-confirmatory-final-results.md` and update SDD ledger**
