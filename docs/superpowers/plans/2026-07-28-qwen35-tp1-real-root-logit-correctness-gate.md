# Qwen3.5 TP1 Real Root-Logit Correctness Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Load the approved Qwen3.5-2B checkpoint into an exact TP1 native root model, execute deterministic one-shot prompts, and independently compare full-vocabulary final-token logits with the official Transformers reference.

**Architecture:** Reuse the authorized real-checkpoint candidate stack to build a TP1 `Qwen35PackedForCausalLM`, add one gate-owned FP32 causal-attention backend, and run official and native models in separate fresh GPU processes. Publish JSON plus two CPU-FP32 tensor maps and verify all metrics independently without importing TinyLLMForge producer code.

**Tech Stack:** Python 3.9+, PyTorch 2.4.1 CUDA 12.1, Transformers 5.8.1, Qwen3.5-2B BF16 checkpoint, safetensors, JSON/SHA256 evidence, SSH source-bound execution.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Inline execution only; do not use subagents.
- Do not stage, commit, merge, create a PR, delete evidence, or overwrite a run.
- Use only `sitian@10.232.195.203`.
- Use `KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian`.
- Use SSH ControlMaster `/tmp/ssh-sitian-10.232.195.203`.
- Use remote Python `/data00/home/sitian/sitian-workspace01/tllm/env/bin/python`.
- Use only the approved checkpoint and model manifest from the design.
- Reference and native models must run in separate fresh processes and must not coexist on one GPU.
- Require at least 24 GiB free GPU memory before each model process.
- Never kill unrelated GPU processes.
- Do not construct `LLMEngine`, `ModelRunner`, Scheduler, sampler, or generation.
- Do not modify the immutable schema-v2 canonical `NO_GO`.
- Do not claim TP4, cached decode, Engine correctness, speed, cache, memory, compression, or quality improvement.

---

### Task 1: Frozen Corpus and Comparison Contract

**Files:**
- Create: `tools/qwen35_tp1_real_root_logit_correctness_contract.py`
- Create: `tools/test_qwen35_tp1_real_root_logit_correctness_contract.py`

**Interfaces:**
- Produces: `PromptCase`, `ComparisonTolerance`, `prompt_cases()`, `compare_logits(...)`, `classify_rows(...)`.
- Consumes: p17/p65 arrays reconstructed once from the frozen schema-v2
  deterministic generator and an explicit tokenizer-range-safe synthetic
  token array.

- [ ] **Step 1: Write RED tests for exact prompt identity**

  Require exactly three IDs:

  ```python
  assert tuple(case.case_id for case in prompt_cases()) == (
      "p17",
      "p65",
      "synthetic",
  )
  ```

  Require positive valid token IDs, exact lengths, exact token SHA256, no
  duplicate case IDs, every token below tokenizer vocab size `248044`, and no
  tokenizer or old-probe call.

- [ ] **Step 2: Run the contract tests and require RED**

  ```bash
  python3 tools/test_qwen35_tp1_real_root_logit_correctness_contract.py
  ```

  Expected: import failure because the contract module does not exist.

- [ ] **Step 3: Implement the frozen dataclasses and corpus**

  Define:

  ```python
  @dataclass(frozen=True)
  class PromptCase:
      case_id: str
      token_ids: tuple[int, ...]
      token_sha256: str

  @dataclass(frozen=True)
  class ComparisonTolerance:
      atol: float
      rtol: float
  ```

  Reconstruct p17/p65 once with the frozen schema-v2 call:

  ```python
  deterministic_token_ids(
      length=prompt_length,
      vocab_size=248044,
      seed=prompt_length,
      forbidden_ids=set(),
  )
  ```

  Assert those values match the exact arrays frozen in the design, then
  hard-code those arrays and canonical token SHA256 values in the new
  contract. Do not import the old probe or generator from the new runtime
  contract. Define synthetic exactly as:

  ```python
  (128, 129, 255, 256, 1024, 32768, 65536, 124022, 186033, 247787, 248043)
  ```

  Hash token arrays as UTF-8 compact JSON lists using
  `ensure_ascii=True, separators=(",", ":")`.

- [ ] **Step 4: Write RED metric-reconstruction tests**

  Use small controlled FP32 tensors to require exact:

  ```text
  full-logit SHA256
  top-20 ordering with token-ID tie break
  winner and runner-up
  winner margin
  max/mean/percentile absolute error
  cosine similarity
  allclose violation count
  maximum scaled error
  tie-preservation behavior
  ```

- [ ] **Step 5: Implement comparison and classification**

  Reuse the frozen schema-v2 BF16 decision-preserving semantics without
  importing the old producer. Return `NO_GO_LOGIT` for winner/top-k/margin
  violations and `PASS` otherwise.

- [ ] **Step 6: Run GREEN and compile**

  ```bash
  python3 tools/test_qwen35_tp1_real_root_logit_correctness_contract.py
  python3 -m py_compile \
    tools/qwen35_tp1_real_root_logit_correctness_contract.py \
    tools/test_qwen35_tp1_real_root_logit_correctness_contract.py
  ```

### Task 2: TP1 Causal Attention and Native Runner

**Files:**
- Create: `tools/qwen35_tp1_real_root_logit_correctness_preflight.py`
- Create: `tools/test_qwen35_tp1_real_root_logit_correctness_preflight.py`

**Interfaces:**
- Produces: `Qwen35TP1CausalAttentionBackend`.
- Produces: `run_native_case(...) -> NativeCaseResult`.
- Reuses: authorized checkpoint metadata, target factory, streamed loader, and transaction validator.

- [ ] **Step 1: Write causal-attention RED tests**

  Compare the backend against a manual FP32 causal-attention calculation for:

  ```text
  one query head / one KV head
  four query heads / one replicated KV head
  multiple tokens with future-token poisoning
  BF16 input and FP32 accumulation
  malformed shape and dtype rejection
  ```

- [ ] **Step 2: Run focused tests and require RED**

  ```bash
  python3 tools/test_qwen35_tp1_real_root_logit_correctness_preflight.py
  ```

  Expected: missing backend or preflight module.

- [ ] **Step 3: Implement the gate-owned attention backend**

  Implement strict lower-triangular FP32 attention and cast the flattened
  output to the query dtype. Do not call FlashAttention or TinyLLMForge
  production attention kernels.

- [ ] **Step 4: Write native-runner RED tests with dependency-light fakes**

  Require:

  ```text
  exact TP context (1,0)
  exact checkpoint and manifest identity
  exact Qwen35PackedForCausalLM
  one fresh lease per case
  prepare-before-commit state boundary
  final-token logits extraction
  release zeroing
  no Engine/Runner/Scheduler/sampler construction
  ```

- [ ] **Step 5: Implement the real native runner**

  Reuse the existing producer-component builder, but request TP1 and inject
  `Qwen35TP1CausalAttentionBackend`. Move the complete candidate to one chosen
  idle GPU only after the checkpoint transaction succeeds, then execute
  `run_step()` under `torch.no_grad()`.

- [ ] **Step 6: Add state-failure tests**

  Inject an lm-head exception into a small real packed fixture. Require pool
  values, bindings, storage identity, and tensor versions to remain unchanged.
  Require successful execution to change all 18 linear-layer state pairs and
  release to zero all pairs.

- [ ] **Step 7: Run GREEN and adjacent model tests**

  Run:

  ```text
  TP1 correctness preflight tests
  transactional root causal-LM tests
  packed layer-stack tests
  state transaction tests
  concrete component factory tests
  checkpoint binding/assignment/loader tests
  full-attention shell tests
  linear-attention shell tests
  ```

### Task 3: Isolated Official Reference Worker

**Files:**
- Modify: `tools/qwen35_tp1_real_root_logit_correctness_preflight.py`
- Modify: `tools/test_qwen35_tp1_real_root_logit_correctness_preflight.py`

**Interfaces:**
- Produces: `run_reference_worker(...) -> ReferenceWorkerResult`.
- Produces: one CPU-FP32 tensor row per prompt case.

- [ ] **Step 1: Write command and resource RED tests**

  Require `local_files_only=True`, `trust_remote_code=False`, BF16, eager
  attention, `use_cache=False`, no network environment, one selected GPU, and
  a 24 GiB free-memory preflight.

- [ ] **Step 2: Implement isolated worker execution**

  Spawn a fresh subprocess, load the official model, run all frozen prompts,
  atomically write a temporary CPU tensor map and JSON process row, unload the
  model, empty the process-owned CUDA allocator, and exit.

- [ ] **Step 3: Add fail-closed tests**

  Reject missing model files, wrong model manifest, insufficient GPU memory,
  non-finite logits, wrong vocabulary width, worker timeout, non-zero exit,
  missing tensor output, or a surviving child PID.

- [ ] **Step 4: Run focused GREEN**

  Run the dependency-light tests locally and one remote reference-only smoke
  with a unique tag. Preserve a failed smoke and never reuse its directory.

### Task 4: Artifact Builder and Independent Verifier

**Files:**
- Modify: `tools/qwen35_tp1_real_root_logit_correctness_preflight.py`
- Create: `tools/verify_qwen35_tp1_real_root_logit_correctness_gate.py`
- Create: `tools/test_verify_qwen35_tp1_real_root_logit_correctness_gate.py`

**Interfaces:**
- Produces: exact four-file authoritative artifact.
- Produces: `verify_run(...) -> dict`.

- [ ] **Step 1: Write artifact refusal RED tests**

  Refuse publication unless reference and native process rows are complete,
  separated, cleanup-proven, source-bound, model-bound, and classified.

- [ ] **Step 2: Implement atomic publication**

  Publish:

  ```text
  tp1_real_root_logit_correctness.json
  reference_logits.pt
  native_logits.pt
  source_manifest.json
  ```

  Write `.partial` files first and rename only after every SHA256 is known.

- [ ] **Step 3: Write verifier tamper matrix**

  Reject:

  ```text
  extra inventory
  source or checkpoint drift
  prompt-token drift
  tensor replacement or shape drift
  re-signed derived metrics
  relaxed tolerance
  missing state evidence
  shared reference/native PID
  insufficient memory preflight
  non-zero forbidden counter
  false PASS classification
  ```

- [ ] **Step 4: Implement independent verification**

  The verifier imports PyTorch only for CPU tensor loading. It must not import
  TinyLLMForge, the producer, Transformers, or CUDA.

- [ ] **Step 5: Run GREEN, compile, and diff check**

  ```bash
  python3 tools/test_verify_qwen35_tp1_real_root_logit_correctness_gate.py
  python3 -m py_compile \
    tools/qwen35_tp1_real_root_logit_correctness_preflight.py \
    tools/test_qwen35_tp1_real_root_logit_correctness_preflight.py \
    tools/verify_qwen35_tp1_real_root_logit_correctness_gate.py \
    tools/test_verify_qwen35_tp1_real_root_logit_correctness_gate.py
  git diff --check
  ```

### Task 5: Source-Bound Remote Gate

**Files:**
- Modify: `tools/qwen35_tp1_real_root_logit_correctness_preflight.py`
- Modify: `tools/test_qwen35_tp1_real_root_logit_correctness_preflight.py`
- Create: `experiments/qwen35_hybrid_state/<unique-tag>/tp1_real_root_logit_correctness.json`
- Create: `experiments/qwen35_hybrid_state/<unique-tag>/reference_logits.pt`
- Create: `experiments/qwen35_hybrid_state/<unique-tag>/native_logits.pt`
- Create: `experiments/qwen35_hybrid_state/<unique-tag>/source_manifest.json`

**Interfaces:**
- Consumes: exact source tar, approved checkpoint, frozen prompts, remote GPU.
- Produces: one authoritative PASS, NO_GO, or incomplete run without overwriting evidence.

- [ ] **Step 1: Implement CLI modes**

  Add:

  ```text
  run
  internal-reference
  internal-native
  validate
  ```

- [ ] **Step 2: Implement deterministic staging**

  Stage the exact source closure and contract, rehash remotely, create one
  unique run root, and prohibit existing tags.

- [ ] **Step 3: Select one resource-safe GPU**

  Read all GPU UUIDs/free memory, choose the lowest-index GPU with at least
  24 GiB free, record the decision, and recheck immediately before each
  worker. Do not reserve or kill unrelated processes.

- [ ] **Step 4: Execute reference then native**

  Require reference exit and PID disappearance before native startup. Use
  unique ports even though this TP1 gate must not initialize a process group.

- [ ] **Step 5: Finalize and download exact artifacts**

  Compare, classify, clean temporary worker files, publish exact inventory,
  download all four files atomically, and verify locally.

- [ ] **Step 6: Verify remotely through a read-only view**

  Run the independent verifier outside the authoritative directory and require
  the same classification, checks, result SHA256, and tensor SHA256 values.

### Task 6: Completion Audit and Next Boundary

**Files:**
- Modify: `docs/superpowers/specs/2026-07-28-qwen35-tp1-real-root-logit-correctness-gate-design.md`
- Modify: `docs/superpowers/plans/2026-07-28-qwen35-tp1-real-root-logit-correctness-gate.md`
- Modify: `AGENT_HANDOFF_STATE.md`

**Interfaces:**
- Produces: authoritative interpretation and next TODO.

- [ ] **Step 1: Run the full focused and adjacent matrix**

  Include the constructed ownership gate/verifier, schema-v2
  contract/verifier, root shell, component factory, checkpoint loader, and
  exact worker rejection.

- [ ] **Step 2: Audit every spec requirement**

  Map each requirement to source, test, raw artifact field, tensor artifact,
  and verifier check.

- [ ] **Step 3: Record the exact conclusion**

  If `PASS`, state only TP1 one-shot root-logit equivalence. If
  `NO_GO_LOGIT` or `NO_GO_STATE`, preserve the evidence and record the first
  failing case/metric without weakening thresholds.

- [ ] **Step 4: Keep the long-term goal active**

  On PASS, next TODO is TP4 distributed one-shot correctness. On failure, next
  TODO is layer/component divergence localization. Performance and cache
  benchmarks remain pending in both cases.

## Execution Record

All six tasks are complete for the TP1 boundary.

```text
final authoritative run:
  qwen35-tp1-authority-20260728-195153-r2
classification:
  PASS
independent verification:
  PASS, 179 checks
source tree:
  e5da50970951b32a61ecc9f85cf1cd447dc95950856345d780cd9e089bdf11ab
result:
  39c4bbc548a82e915609dccb57101a6e64cbc92319d5ba69c44d427f8f4aa519
reference logits:
  3373ab6038d6f21cf6421aa0e1c9146cc001e6e8bcbd06e6ab59c65ecc709e5a
native logits:
  5d3fcb3a204a1b1bd673026d83f06ddc76422ca49d51d6af32671ba153ecb4d4
```

The result is limited to TP1 one-shot final-token
`bf16_decision_preserving` correctness. It is not an elementwise allclose
result and does not authorize any performance, cache, memory, compression, or
quality claim.

The first authority run
`qwen35-tp1-authority-20260728-194155` remains preserved as superseded
evidence. The final r2 run adds distinct dynamic
`TINYVLLM_DIST_PORT`/`MASTER_PORT` environments for the two isolated workers
without initializing distributed execution.

Next plan boundary:

1. implement TP4 distributed one-shot root-logit correctness against the same
   official oracle;
2. prove cached full-attention continuation;
3. integrate the proven owner into ModelRunner and bounded Engine execution;
4. only then benchmark latency, throughput, cache footprint, and GPU memory.
