# KV8 + Quest Selective-Dequant Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether Quest top-16 selective dequantization materially recovers the current KV8 decode regression without unacceptable long-context retrieval loss.

**Architecture:** Extend the existing step-scaling harness with explicit Quest configuration and provenance, then run source-bound eager-only performance arms plus a fixed-prompt needle quality comparison. A small independent analyzer validates arm identity and applies the predeclared speed and quality gates.

**Tech Stack:** Python 3, pytest, Bash, TinyLLMForge `LLM`, JSON artifacts, SSH to the existing A100 host.

## Global Constraints

- Use only `/Users/bytedance/Desktop/TinyLLMForge`.
- Do not reset, clean, or include unrelated untracked SLO/KV-capacity artifacts.
- Do not create another worktree or use subagents.
- Do not run `klist` or wait on Kerberos TTL; use the existing FILE cache for SSH.
- Write remote run data only below `/data00/home/sitian/tllm/kvcapacity-runs/`.
- Use Qwen3-8B, A100 80GB PCIe, TP1, eager decode, and `KV_BLOCKS=640`.
- Use context 8192 with batches 4, 8, 12, 16, and 19.
- Keep Quest top-k fixed at 16 and `quest_min_seq_len=512`.
- Do not claim graph-path, production, TP2/TP4, or cross-model benefit.
- A failed or partial tag is immutable and must not be reused.

---

### Task 1: Add auditable Quest configuration to the step-scaling harness

**Files:**
- Modify: `tools/test_kvcapacity_step_scaling.py`
- Modify: `tools/kvcapacity_step_scaling_worker.py`
- Modify: `tools/run_kvcapacity_step_scaling_remote.sh`

**Interfaces:**
- Consumes: `quest_top_k_blocks: int`, `quest_min_seq_len: int`
- Produces: worker CLI flags, runner environment passthrough, requested payload configuration, and resolved engine identity

- [x] **Step 1: Write failing worker CLI and payload tests**

Add tests asserting:

```python
args = worker.parse_args([
    "--model-path", "m",
    "--out", "o",
    "--kv-quant-bits", "8",
    "--quest-top-k-blocks", "16",
    "--quest-min-seq-len", "512",
])
assert args.quest_top_k_blocks == 16
assert args.quest_min_seq_len == 512
```

and asserting that `build_payload(...)["configuration"]` contains
`kv_quant_bits`, `quest_top_k_blocks`, and `quest_min_seq_len`.

- [x] **Step 2: Write a failing engine-construction passthrough test**

Replace the imported `tinyvllm.LLM` with a capturing fake and assert that
`_load_engine(...)` passes:

```python
kv_quant_bits=8
quest_top_k_blocks=16
quest_min_seq_len=512
```

- [x] **Step 3: Write a failing runner-source test**

Read `tools/run_kvcapacity_step_scaling_remote.sh` and assert that non-default
`QUEST_TOP_K_BLOCKS` and `QUEST_MIN_SEQ_LEN` become
`--quest-top-k-blocks` and `--quest-min-seq-len` worker arguments.

- [x] **Step 4: Run RED**

Run:

```bash
python3 -m pytest -q \
  tools/test_kvcapacity_step_scaling.py \
  -k 'quest or quant_config'
```

Expected: failures because the new CLI, payload, and passthrough do not exist.

- [x] **Step 5: Implement the minimum passthrough**

Thread the two Quest values through `parse_args`, `main`, `run`,
`_load_engine`, `_engine_identity`, and `build_payload`. Add runner defaults
and append the worker flags only when Quest is enabled.

- [x] **Step 6: Run GREEN and adjacent tests**

Run:

```bash
python3 -m pytest -q \
  tools/test_kvcapacity_step_scaling.py \
  tools/test_kv_slot_snapshot_quantised.py \
  tools/test_eval_needle_fixed_prompts.py
```

Expected: all tests pass.

- [x] **Step 7: Commit and push the source-bound harness**

Stage only the three task files plus this design and plan. Commit with:

```text
test(kvcapacity): add KV8 Quest gate plumbing
```

Push and verify the exact remote SHA before launching experiments.

### Task 2: Run the eager performance diagnostic

**Files:**
- Create after execution: `experiments/kvcapacity_step_scaling/<bf16-tag>/`
- Create after execution: `experiments/kvcapacity_step_scaling/<kv8-tag>/`
- Create after execution: `experiments/kvcapacity_step_scaling/<kv8-quest-tag>/`

**Interfaces:**
- Consumes: pushed Task-1 source SHA
- Produces: three source-bound sweep bundles

- [x] **Step 1: Check the remote GPU state**

Use a read-only SSH command with the existing Kerberos cache. Select one A100
with enough free memory and no active foreign compute process. Do not terminate
any process.

- [x] **Step 2: Run bf16 eager**

Run the existing remote runner with:

```text
MODE=sweep
EXECUTION_PATHS_OVERRIDE=eager
KV_BLOCKS=640
KV_QUANT_BITS=0
SWEEP_GRID=8192:4,8,12,16,19
WARMUP_STEPS=24
MEASURED_STEPS=24
```

- [x] **Step 3: Run KV8 eager**

Use the same values with `KV_QUANT_BITS=8`.

- [x] **Step 4: Run KV8+Quest eager**

Use the same values with:

```text
KV_QUANT_BITS=8
QUEST_TOP_K_BLOCKS=16
QUEST_MIN_SEQ_LEN=512
```

- [x] **Step 5: Validate every raw artifact**

Require all five cells per arm to be measured, stable at their target batch,
and fully eager. Verify source SHA, model, pool, grid, seed, warmup, and sample
counts match.

### Task 3: Run the fixed-prompt quality diagnostic

**Files:**
- Create after execution: compact bf16 and KV8/Quest needle JSON artifacts

**Interfaces:**
- Consumes: pushed Task-1 source SHA and the same Qwen3-8B checkpoint
- Produces: paired accuracy and throughput rows for bf16, KV8 full, and
  KV8+Quest

- [x] **Step 1: Run bf16 baseline**

Use `tools/eval_needle.py` with context 8192, five fixed depths, five trials,
greedy decoding, and `top_k=-1`.

- [x] **Step 2: Run KV8 full plus KV8+Quest**

Use one KV8 engine with fixed prompts and `top_k=-1 16`, clearing prefix-cache
metadata between settings.

- [x] **Step 3: Validate pairing and completeness**

Require 25 cases per setting, identical `(context, depth, trial, magic)` keys,
and nonempty outputs.

### Task 4: Independently classify the gate

**Files:**
- Create: `tools/kvcapacity_kv8_quest_gate.py`
- Create: `tools/test_kvcapacity_kv8_quest_gate.py`
- Create after execution: compact gate JSON and Markdown report

**Interfaces:**
- Consumes: three performance sweep bundles and two quality bundles
- Produces: `GO_TO_FUSED_KERNEL_SCOPE`, `NO_GO_KV8_QUEST`, or `INCONCLUSIVE`

- [x] **Step 1: Write failing mutation tests**

Cover mismatched source identity, pool, grid, execution path, incomplete
cells, non-improving Quest cells, insufficient wall-cell excess-latency
recovery, overall accuracy loss, and per-depth accuracy loss.

- [x] **Step 2: Run RED**

Run:

```bash
python3 -m pytest -q tools/test_kvcapacity_kv8_quest_gate.py
```

- [x] **Step 3: Implement independent validation and classification**

Compute per-cell latency ratios, wall-cell excess-latency recovery, overall
and per-depth accuracy deltas, and fixed failure precedence.

- [x] **Step 4: Run GREEN and the complete adjacent suite**

Run:

```bash
python3 -m pytest -q \
  tools/test_kvcapacity_kv8_quest_gate.py \
  tools/test_kvcapacity_step_scaling.py \
  tools/test_kvcapacity_batch_sweep_analysis.py \
  tools/test_kv_slot_snapshot_quantised.py \
  tools/test_eval_needle_fixed_prompts.py
```

- [x] **Step 5: Classify the real artifacts**

Write a compact JSON result and Markdown report. Do not copy large raw remote
artifacts into the repository.

### Task 5: Close the evidence chain

**Files:**
- Modify: `docs/superpowers/audits/2026-08-16-phase1-completion-audit.md`
- Modify: `AGENT_HANDOFF_STATE.md`

- [x] **Step 1: Append the prompt-to-artifact checklist**

Record source SHA, immutable tags, remote paths, model/hardware identity,
performance and quality metrics, all gate checks, and explicit unsupported
claims.

- [x] **Step 2: Verify the complete task**

Run all Task-4 tests, parse every JSON artifact, run `git diff --check`, and
confirm the staged-path manifest contains only task files.

- [ ] **Step 3: Commit and push**

Use exact-path staging, push `feat/kv-sparse-attention`, and verify local HEAD
equals `git ls-remote` for the branch.
