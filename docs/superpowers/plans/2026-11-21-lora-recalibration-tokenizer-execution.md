# 120-Row Real Tokenizer Length Execution And Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the cold-process remote bootstrap (`recalibration_tokenizer_remote.py`) and local launch/reconstruction controller (`recalibration_tokenizer_controller.py`) to measure all 120 rows constructed from the accepted 480-item dynamic validation against the real Qwen3-8B fast tokenizer (`4096` input cap, `256` target cap including unique EOS, `enable_thinking=False`) without altering existing synthetic-only modules or loading model weights/GPUs.

**Architecture:** Locally, `recalibration_tokenizer_controller.py` reconstructs the 120-row inventory from the 480-item raw validation inputs via `recalibration_tokenizer_inventory.build_synthetic_inventory`, projects only `{ordinal, messages, answer, row_sha256}` (no full oracles or private dynamic receipts), builds a non-cyclic launch package (`config_sha256` -> SSH `argv` -> `package_sha256`) with the 7 pure session modules and fresh `TOKENIZER_RUNTIME_PROFILE`, conducts a `TLFH1` (`READY / RESULT / STDOUT / STDERR / DONE`) handshake over an owned SSH process group (`1260s` SSH work + `2s` cleanup), and reconstructs the 120-row report locally via `recalibration_tokenizer_report.build_synthetic_report` wrapped in an outer `LORA_RECALIBRATION_TOKENIZER_EXECUTION_EVIDENCE` envelope. Remotely, `recalibration_tokenizer_remote.py` runs in a cold `-I -S -B` Python 3.12.13 process, enforces `RLIMIT_CORE=(0,0)`, offline environment flags (`USE_TORCH=0`, `CUDA_VISIBLE_DEVICES=""`), and `seccomp` fork/clone/thread denial (`install_task_denial`) before emitting `READY`, validates `grant` and the 7-module + counting-projection + profile bundle under a single `1200s` pre-`READY` monotonic anchor, creates the read-only 5-file tokenizer snapshot via `create_snapshot`, runs `open_session` -> 120 `measure` calls -> `finish` (or `abort`), and emits the raw measurement chain and finalization in `RESULT` + `DONE`.

**Tech Stack:** Python 3.9+ standard library and pinned repository modules (`pilot_protocol`, `validation_contract`, `tokenizer_measurement`, `tokenizer_runtime_guard`, `tokenizer_runtime_profile`, `tokenizer_snapshot`, `tokenizer_session`, `recalibration_tokenizer_inventory`, `recalibration_tokenizer_report`), tested via `unittest` with artificial process/session boundaries.

## Global Constraints

- Authority: `experiments/lora_necessity/20261120-recalibration-launch-readiness.md` Section 5.C and `research-path-audit.md` Gate G2.
- Private verification root: `/private/tmp/lora-recal-tokenizer-exec-20261121-cqMfmn` (205 baseline LoRA files frozen).
- Do not modify any of the 205 frozen baseline files in `experiments/lora_necessity/`.
- Do not import `transformers`, `torch`, `validation_remote`, or `validation_task_worker` in host tests.
- Keep `input_tokens <= 4096`, `answer_tokens_including_eos <= 256`, `enable_thinking=False`.
- No `session_factory`, `transport_callback`, `real=True`, or `approved=True` parameters on production entrypoints.
- No inline comments (`#`) and no single-letter variable names in Python source or test files.

---

### Task 1: Cold-Process Remote Tokenizer Bootstrap (`recalibration_tokenizer_remote.py`)

**Files:**
- Create: `experiments/lora_necessity/recalibration_tokenizer_remote.py`
- Test: `experiments/lora_necessity/test_recalibration_tokenizer_remote.py`

- [x] **Step 1: Write failing tests (`test_recalibration_tokenizer_remote.py`)**
- [x] **Step 2: Run focused RED and save log**
- [x] **Step 3: Implement `recalibration_tokenizer_remote.py`**
- [x] **Step 4: Run focused GREEN and save log**

---

### Task 2: Local 120-Row Tokenizer Package And Controller (`recalibration_tokenizer_controller.py`)

**Files:**
- Create: `experiments/lora_necessity/recalibration_tokenizer_controller.py`
- Test: `experiments/lora_necessity/test_recalibration_tokenizer_controller.py`

- [x] **Step 1: Write failing tests (`test_recalibration_tokenizer_controller.py`)**
- [x] **Step 2: Run focused RED and save log**
- [x] **Step 3: Implement `recalibration_tokenizer_controller.py`**
- [x] **Step 4: Run focused GREEN and combined regression**
- [x] **Step 5: Independent review and build real 120-row launch package**
