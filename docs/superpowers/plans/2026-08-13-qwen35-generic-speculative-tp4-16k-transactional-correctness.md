# Qwen3.5 Generic Speculative TP4 16K Transactional Correctness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: execute this plan inline with strict test-driven development. Subagents, worktrees, commits, staging, pushes, stashes, resets, and cleans are forbidden by the active user constraints.

**Goal:** Build and run an independent source-bound Qwen3.5 TP4/16K transactional correctness authority without changing the frozen TP4/4K authority.

**Architecture:** Add a narrow 16K overlay that reuses the frozen 4K authority's side-effect-free validators and orchestration while replacing all authority identity and capacity constants before execution. Add a worker overlay that delegates to the established worker behavior but forces the approved long-context Engine configuration, then bind the independent verifier and remote runner to the new source set and artifact namespace.

**Tech Stack:** Python 3, pytest-compatible direct test functions, TinyLLMForge Engine, PyTorch distributed TP4, Bash, SSH/rsync, JSON/SHA-256 source binding.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Do not modify the existing 4K gate, worker, verifier, runner, tests, or authority artifacts.
- Do not stage, commit, push, switch branches, create worktrees, stash, reset, or clean.
- Use `sitian@10.232.195.203` only.
- Use `KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian`.
- Use `ControlMaster=no` and `ControlPath=none`.
- Keep SSH, rsync, launch, and polling serial with finite retries.
- Use strict RED, observed failure, minimal GREEN, and observed pass for every behavior change.
- Freeze schema `qwen35.generic-speculative-tp4-16k-transactional-correctness.v1`.
- Freeze classification `SECOND_MODEL_TP4_16K_ESTABLISHED`.
- Freeze scope `second_model_tp4_16k_only`.
- Freeze TP4, batch `(1, 4)`, context `16384`, output `8`, n-gram `3`, proposal `4`.
- Freeze `max_model_len=33024` and `max_num_batched_tokens=132096`.
- Freeze prefill step `1024`, GPU blocks `68`, logical blocks `640`, blockwise blocks `8`.
- Require positive production H2D copies and bytes in the n-gram batch-4 cell.
- Require exact greedy parity and zero accepted-prefix replay.
- Do not claim performance, 32K, learned drafter, KV quantization, production readiness, or Phase 1 completion.

---

### Task 1: Freeze the independent 16K gate contract

**Files:**
- Create: `tools/test_qwen35_generic_speculative_tp4_16k_gate.py`
- Create: `tools/qwen35_generic_speculative_tp4_16k_gate.py`

**Interfaces:**
- Consumes: the public and private validator functions in `tools/qwen35_generic_speculative_tp4_gate.py`.
- Produces: the 16K constants, `validate_result(value) -> dict`, campaign CLI, hashing helpers, and artifact helpers used by later tasks.

- [ ] **Step 1: Write the failing constants and source-isolation tests**

Add tests that import the new module and assert:

```python
assert gate.SCHEMA_VERSION == (
    "qwen35.generic-speculative-tp4-16k-"
    "transactional-correctness.v1"
)
assert gate.CLASSIFICATION == "SECOND_MODEL_TP4_16K_ESTABLISHED"
assert gate.CLAIM_SCOPE == "second_model_tp4_16k_only"
assert gate.WORLD_SIZE == 4
assert gate.BATCH_SIZES == (1, 4)
assert gate.CONTEXT_TOKENS == 16384
assert gate.MAX_OUTPUT_TOKENS == 8
assert "context_16k_not_established" not in gate.LIMITATIONS
assert "context_32k_not_established" in gate.LIMITATIONS
assert "tools/qwen35_generic_speculative_tp4_gate.py" in (
    gate.DEFAULT_SOURCE_FILES
)
assert "tools/qwen35_generic_speculative_tp4_16k_gate.py" in (
    gate.DEFAULT_SOURCE_FILES
)
```

Also hash `tools/qwen35_generic_speculative_tp4_gate.py` before and after
loading the 16K module and assert the digest is unchanged.

- [ ] **Step 2: Run the focused test and observe RED**

Run:

```bash
python3 -m pytest \
  tools/test_qwen35_generic_speculative_tp4_16k_gate.py::test_contract_constants_are_frozen \
  -q
```

Expected: collection/import failure because the 16K gate does not exist.

- [ ] **Step 3: Implement the minimal 16K overlay**

Create a module that loads the frozen gate under a private module name,
overrides its authority constants before campaign use, extends
`DEFAULT_SOURCE_FILES`, and re-exports its functions. Do not write to the 4K
source file or import it under its normal module name.

- [ ] **Step 4: Run the focused test and observe GREEN**

Run the command from Step 2.

Expected: `1 passed`.

### Task 2: Enforce the 16K-only H2D rule

**Files:**
- Modify: `tools/test_qwen35_generic_speculative_tp4_16k_gate.py`
- Modify: `tools/qwen35_generic_speculative_tp4_16k_gate.py`

**Interfaces:**
- Consumes: the frozen gate's normalized result and movement summaries.
- Produces: `validate_result(value) -> dict` that additionally rejects a
  candidate batch-4 cell with zero H2D copies or bytes.

- [ ] **Step 1: Write failing movement tests**

Build a valid synthetic result using the frozen test fixtures adapted to the
16K constants. Assert that:

```python
result["cells"]["ngram:b4"]["movement"]["h2d_copies"] = 0
```

raises `ValueError("16K batch-4 candidate requires real H2D copies")`, and
that zero `h2d_bytes` raises the corresponding byte error. Include a passing
case with both values positive.

- [ ] **Step 2: Run the three tests and observe RED**

Run:

```bash
python3 -m pytest tools/test_qwen35_generic_speculative_tp4_16k_gate.py \
  -k 'h2d' -q
```

Expected: the zero-H2D cases do not yet fail.

- [ ] **Step 3: Add the minimal post-validation rule**

Wrap the frozen `validate_result`, select `ngram:b4`, read its normalized
movement aggregate, require integer `h2d_copies > 0` and `h2d_bytes > 0`, and
return the normalized result unchanged otherwise.

- [ ] **Step 4: Run the three tests and observe GREEN**

Run the command from Step 2.

Expected: all selected tests pass.

### Task 3: Add the long-context worker overlay

**Files:**
- Modify: `tools/test_qwen35_generic_speculative_tp4_16k_gate.py`
- Create: `tools/qwen35_generic_speculative_tp4_16k_worker.py`

**Interfaces:**
- Consumes: the frozen 4K worker's generation, receipt capture, profiling,
  cleanup, and campaign helpers.
- Produces: `run_cell(...) -> dict` and `main(argv=None) -> int` using the
  16K gate and long-context Engine configuration.

- [ ] **Step 1: Write a failing Engine configuration test**

Load the worker with injected fake dependencies, invoke `run_cell`, capture
the keyword arguments received by the fake Engine factory, and assert:

```python
assert kwargs["max_model_len"] == 33024
assert kwargs["max_num_batched_tokens"] == 132096
assert kwargs["max_num_prefill_tokens_per_step"] == 1024
assert kwargs["chunked_prefill_decode_first"] is False
assert kwargs["chunked_prefill_mixed_batch"] is False
assert kwargs["kv_offload_gpu_blocks"] == 68
assert kwargs["kv_offload_logical_blocks"] == 640
assert kwargs["kv_offload_blockwise_blocks"] == 8
```

Assert prompt construction uses exactly `16384` context tokens.

- [ ] **Step 2: Run the focused worker test and observe RED**

Run:

```bash
python3 -m pytest tools/test_qwen35_generic_speculative_tp4_16k_gate.py \
  -k 'worker_uses_frozen_long_context_configuration' -q
```

Expected: import failure because the worker does not exist.

- [ ] **Step 3: Implement the minimal worker overlay**

Load the frozen worker privately, replace its `gate` reference with the 16K
gate, delegate all unchanged helpers, and wrap the Engine factory so the
frozen worker's call receives the approved long-context values. Ensure
campaign dispatch resolves to the wrapped `run_cell`, not the frozen one.

- [ ] **Step 4: Run the focused worker test and observe GREEN**

Run the command from Step 2.

Expected: the selected test passes.

### Task 4: Bind an independent verifier

**Files:**
- Modify: `tools/test_qwen35_generic_speculative_tp4_16k_gate.py`
- Create: `tools/verify_qwen35_generic_speculative_tp4_16k_gate.py`

**Interfaces:**
- Consumes: `validate_result`, source hashing, and artifact hashing from the
  16K gate.
- Produces: `verify_run(run_dir, source_root=None) -> dict` and CLI exit status.

- [ ] **Step 1: Write failing verifier tests**

Create a valid run directory and assert PASS. Then independently tamper with:

- `result.json`;
- the source manifest schema;
- the approved model digest;
- one bound source file; and
- the n-gram batch-4 H2D counters.

Assert every tampered case returns `classification == "FAIL"` with the
specific failure in `failures[0]`.

- [ ] **Step 2: Run verifier tests and observe RED**

Run:

```bash
python3 -m pytest tools/test_qwen35_generic_speculative_tp4_16k_gate.py \
  -k 'verifier' -q
```

Expected: import failure because the verifier does not exist.

- [ ] **Step 3: Implement the verifier**

Mirror the frozen verifier structure, but load only
`qwen35_generic_speculative_tp4_16k_gate.py`. Validate JSON object shape,
schema, source tree, model digest, result digest, and optional live source
identity. Return PASS only when `failures` is empty.

- [ ] **Step 4: Run verifier tests and observe GREEN**

Run the command from Step 2.

Expected: all selected verifier tests pass.

### Task 5: Add the bounded remote runner

**Files:**
- Modify: `tools/test_qwen35_generic_speculative_tp4_16k_gate.py`
- Create: `tools/run_qwen35_generic_speculative_tp4_16k_gate_remote.sh`

**Interfaces:**
- Consumes: the 16K gate, worker, verifier, approved checkpoint, and remote
  Python.
- Produces: one non-replayable local/remote campaign directory and terminal
  `authority` or `authority.failed`.

- [ ] **Step 1: Write the failing source-contract test**

Assert the script contains:

```text
sitian@10.232.195.203
FILE:/Users/bytedance/krb5cc_sitian
ControlMaster=no
ControlPath=none
qwen35_generic_speculative_tp4_16k_gate.py
qwen35_generic_speculative_tp4_16k_worker.py
verify_qwen35_generic_speculative_tp4_16k_gate.py
qwen35_generic_speculative_tp4_16k
campaign.status
campaign.pid
campaign.exit_code
authority.failed
REMOTE_COMMAND_RETRY_ATTEMPTS
REMOTE_RSYNC_RETRY_ATTEMPTS
POLL_INTERVAL_SECONDS
```

Also assert four GPUs are selected with `head -n 4` and existing terminal or
running campaigns are not replayed.

- [ ] **Step 2: Run the source-contract test and observe RED**

Run:

```bash
python3 -m pytest tools/test_qwen35_generic_speculative_tp4_16k_gate.py \
  -k 'remote_runner_source_contract' -q
```

Expected: failure because the runner does not exist.

- [ ] **Step 3: Implement the runner**

Copy the established bounded orchestration semantics into the new namespace.
Tar only `tinyvllm` and the 16K authority plus its explicitly bound frozen
authority dependencies. Preflight checkpoint identity, GPU capacity, and
fresh ports; launch once; poll terminal status; rsync artifacts; run the
independent verifier; and write terminal authority state.

- [ ] **Step 4: Run the source-contract test and shell syntax check**

Run:

```bash
python3 -m pytest tools/test_qwen35_generic_speculative_tp4_16k_gate.py \
  -k 'remote_runner_source_contract' -q
bash -n tools/run_qwen35_generic_speculative_tp4_16k_gate_remote.sh
```

Expected: test pass and `bash -n` exit 0.

### Task 6: Complete local regression validation

**Files:**
- Modify only if a focused RED exposes a real defect in the new 16K files.

**Interfaces:**
- Consumes: all new 16K authority files and frozen 4K regressions.
- Produces: fresh local correctness evidence before remote execution.

- [ ] **Step 1: Run all 16K authority tests**

```bash
python3 -m pytest \
  tools/test_qwen35_generic_speculative_tp4_16k_gate.py -q
```

- [ ] **Step 2: Run frozen authority regressions**

```bash
python3 -m pytest \
  tools/test_qwen35_generic_speculative_tp4_gate.py \
  tools/test_qwen35_generic_speculative_tp4_gate.py \
  tools/test_qwen35_packed_layer_stack.py -q
```

Deduplicate the first path if the shell command is edited; the required point
is to run the frozen 4K gate and packed-layer regressions without changing
them.

- [ ] **Step 3: Compile and syntax-check changed authority files**

```bash
python3 -m py_compile \
  tools/qwen35_generic_speculative_tp4_16k_gate.py \
  tools/qwen35_generic_speculative_tp4_16k_worker.py \
  tools/verify_qwen35_generic_speculative_tp4_16k_gate.py \
  tools/test_qwen35_generic_speculative_tp4_16k_gate.py
bash -n tools/run_qwen35_generic_speculative_tp4_16k_gate_remote.sh
git diff --check
```

Expected: every command exits 0.

### Task 7: Run remote direct tests and the real TP4/16K campaign

**Files:**
- Generate: `artifacts/qwen35_generic_speculative_tp4_16k/opaque-*/`

**Interfaces:**
- Consumes: a source tarball from the exact local tree and the approved remote
  checkpoint.
- Produces: real rank-local TP4 GPU evidence, source manifest, result, verifier
  output, and terminal authority marker.

- [ ] **Step 1: Run remote direct test functions**

Copy the source to a fresh remote directory and invoke the 16K test functions
with `runpy.run_path()` using the approved remote Python because remote pytest
is unavailable. At minimum execute the long-context worker configuration and
H2D fail-closed tests.

- [ ] **Step 2: Launch one fresh campaign**

```bash
KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian \
  bash tools/run_qwen35_generic_speculative_tp4_16k_gate_remote.sh
```

Do not reuse a previous `RUN_ID`.

- [ ] **Step 3: Audit the terminal evidence**

Require:

- independent verifier PASS;
- classification `SECOND_MODEL_TP4_16K_ESTABLISHED`;
- four ranks in every cell;
- exact baseline/candidate token parity;
- accepted and rejected proposals in batch 1 and batch 4;
- zero accepted-prefix replay;
- positive n-gram batch-4 H2D copies and bytes;
- complete KV and recurrent transaction receipts;
- no poison, lease, prepared transaction, process group, worker, or Engine
  cleanup failure; and
- matching result, source-tree, source-file, and model hashes.

If any requirement fails, preserve `authority.failed`, diagnose with a new RED
test, and repeat the RED/GREEN cycle before launching a fresh run ID.

### Task 8: Record the authority and remaining boundary

**Files:**
- Modify: `docs/superpowers/audits/2026-08-12-phase1-objective-coverage.md`
- Modify: `AGENT_HANDOFF_STATE.md`

**Interfaces:**
- Consumes: the verified authority directory and exact audit values.
- Produces: durable handoff state that distinguishes 4K and 16K authority and
  leaves 32K/performance work explicitly open.

- [ ] **Step 1: Add the 16K authority row**

Record run path, result/source/model hashes, batch acceptance/rejection totals,
H2D copies/bytes, exact-parity status, replay count, verifier status, and the
classification.

- [ ] **Step 2: Update the handoff**

Mark Qwen3.5 TP4/16K correctness established only if the independent verifier
passed. Keep overall status `NOT_PROMOTABLE`; keep TP4/32K, performance,
learned drafter, and KV quantization pending.

- [ ] **Step 3: Run final fresh verification**

```bash
python3 -m pytest \
  tools/test_qwen35_generic_speculative_tp4_16k_gate.py \
  tools/test_qwen35_generic_speculative_tp4_gate.py \
  tools/test_qwen35_packed_layer_stack.py -q
python3 -m py_compile \
  tools/qwen35_generic_speculative_tp4_16k_gate.py \
  tools/qwen35_generic_speculative_tp4_16k_worker.py \
  tools/verify_qwen35_generic_speculative_tp4_16k_gate.py \
  tools/test_qwen35_generic_speculative_tp4_16k_gate.py
bash -n tools/run_qwen35_generic_speculative_tp4_16k_gate_remote.sh
git diff --check
```

Read the full outputs before reporting any completion claim.
