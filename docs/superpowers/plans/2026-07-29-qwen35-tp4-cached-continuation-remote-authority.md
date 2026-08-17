# Qwen3.5 TP4 Cached-Continuation Remote Authority Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use Markdown checkboxes for tracking.

**Goal:** Implement a CPU-testable, independently verified remote execution protocol for the standalone Qwen3.5 TP4 cached-continuation authority.

**Architecture:** Keep the frozen Engine two-phase receipt unchanged. Add cached-specific plan and receipt modules while reusing deterministic source staging, the strict four-GPU guard, single-use authorization, the process-free executor pattern, and the isolated subprocess adapter.

**Tech Stack:** Python 3 standard library, canonical JSON, SHA-256, tar, SSH/SCP command construction, file-local fake-runtime tests.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Do not stage, commit, merge, create a branch, or create a PR.
- Do not execute SSH, `scp`, `nvidia-smi`, Torch, Transformers, CUDA, or a GPU workload.
- Use only `sitian@10.232.195.203`.
- Preserve exact `KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian`.
- Preserve the active-compute-process guard and do not kill remote processes.
- Do not modify the frozen Engine remote receipt semantics.

---

### Task 1: Cached Remote Plan

**Files:**
- Create: `tools/qwen35_tp4_cached_continuation_remote_execution_plan.py`
- Create: `tools/test_qwen35_tp4_cached_continuation_remote_execution_plan.py`

**Interfaces:**
- Consumes: Engine authority configuration bundle and immutable source inventory.
- Produces: `build_remote_execution_plan(...) -> dict` and `verify_remote_execution_plan(path) -> dict`.

- [x] **Step 1: Write failing plan tests**

Require exact source identities, resource guard, standalone cached driver
argv, two-entry package inventory, safe extraction, and local cached verifier.

- [x] **Step 2: Run RED**

```bash
python3 tools/test_qwen35_tp4_cached_continuation_remote_execution_plan.py
```

- [x] **Step 3: Implement the minimal process-free plan**

Reuse helper functions from the Engine remote plan only where their semantics
are identical. Use cached-specific paths, package entries, and verifier argv.

- [x] **Step 4: Run GREEN**

Run the focused plan test and `py_compile`.

### Task 2: Cached Receipt

**Files:**
- Create: `tools/qwen35_tp4_cached_continuation_remote_execution_receipt.py`
- Create: `tools/test_qwen35_tp4_cached_continuation_remote_execution_receipt.py`

**Interfaces:**
- Consumes: frozen cached plan, command results, consumed authorization.
- Produces: `produce_execution_receipt(...)` and `validate_execution_receipt(...)`.

- [x] **Step 1: Write failing receipt tests**

Require matching remote/local cached verification payloads, resource UUID
parity, command hashes, package identity, and authorization binding.

- [x] **Step 2: Run RED**

```bash
python3 tools/test_qwen35_tp4_cached_continuation_remote_execution_receipt.py
```

- [x] **Step 3: Implement cached-specific PASS validation**

Do not accept or require Engine two-phase classification fields.

- [x] **Step 4: Run GREEN**

Run focused receipt tests.

### Task 3: Cached Executor and Safety Closure

**Files:**
- Create: `tools/qwen35_tp4_cached_continuation_remote_execution_executor.py`
- Create: `tools/test_qwen35_tp4_cached_continuation_remote_execution_executor.py`
- Modify: `tools/qwen35_tp4_engine_remote_execution_source_contract.py`
- Modify: `tools/test_qwen35_tp4_engine_remote_execution_source_contract.py`
- Modify: `tools/build_qwen35_tp4_engine_authority_configuration.py`
- Modify: `tools/test_build_qwen35_tp4_engine_authority_configuration.py`

**Interfaces:**
- Consumes: cached plan, single-use authorization, injected command runner.
- Produces: authorization-bound PASS receipt or bounded FAILED evidence.

- [x] **Step 1: Write failing executor and AST tests**

Require exact Kerberos environment, plan verification before authorization
consumption, frozen command order, no default runner, and no subprocess use.

- [x] **Step 2: Run RED**

Run cached executor and source-contract tests.

- [x] **Step 3: Implement the process-free executor**

Reuse the established command-shape dispatcher while binding the cached
receipt module.

- [x] **Step 4: Extend immutable source inventory**

Include cached plan, receipt, and executor in the authority source tree.

- [x] **Step 5: Run complete CPU-safe gate**

Run focused tests, the selected authority suite, `py_compile`, and
`git diff --check`.

### Task 4: Durable Evidence

**Files:**
- Modify: `docs/superpowers/plans/2026-07-29-qwen35-tp4-hybrid-prefix-performance-cache-authority.md`
- Modify: `AGENT_HANDOFF_STATE.md`

- [x] **Step 1: Record exact commands and test counts**

- [x] **Step 2: Preserve strict claim boundary**

State that no real remote/GPU execution or performance benefit has been
measured.

## Completion Evidence

Completed on 2026-07-29 under strict CPU-safe dependency injection.

RED evidence:

```text
cached remote plan:
  FileNotFoundError for missing plan module
cached remote receipt:
  FileNotFoundError for missing receipt module
cached remote executor:
  FileNotFoundError for missing executor module
source AST contract:
  rejected missing cached executor
authority source inventory:
  rejected missing cached remote modules
```

Focused GREEN evidence:

```text
cached remote plan:                     4 passed
cached remote receipt:                  5 passed
cached remote executor:                 5 passed
execution source AST contract:          4 passed
configuration builder:                  4 passed
```

Complete selected CPU-safe gate:

```text
244 passed across 34 file-local test files
focused py_compile: passed
git diff --check: passed
staged files: 0
```

The complete gate was run by reading the following exact file list and
executing `PYTHONDONTWRITEBYTECODE=1 python3 "$test_file"` for each entry:

```text
tools/test_qwen35_tp4_cached_continuation_correctness_contract.py
tools/test_verify_qwen35_tp4_cached_continuation_correctness_gate.py
tools/test_qwen35_tp4_cached_continuation_correctness_executor.py
tools/test_qwen35_tp4_cached_continuation_correctness_producer.py
tools/test_qwen35_tp4_cached_continuation_backend_session.py
tools/test_run_qwen35_tp4_cached_continuation_authority.py
tools/test_qwen35_tp4_cached_continuation_remote_execution_plan.py
tools/test_qwen35_tp4_cached_continuation_remote_execution_receipt.py
tools/test_qwen35_tp4_cached_continuation_remote_execution_executor.py
tools/test_qwen35_tp4_engine_correctness_contract.py
tools/test_verify_qwen35_tp4_engine_correctness_gate.py
tools/test_qwen35_tp4_engine_correctness_executor.py
tools/test_qwen35_tp4_engine_correctness_producer.py
tools/test_qwen35_tp4_engine_backend_source_contract.py
tools/test_qwen35_tp4_engine_backend_session.py
tools/test_qwen35_tp4_engine_reference_tokens.py
tools/test_verify_qwen35_tp4_engine_reference_tokens.py
tools/test_qwen35_tp4_engine_reference_tokens_producer.py
tools/test_qwen35_tp4_engine_official_reference_executor.py
tools/test_build_qwen35_tp4_engine_authority_configuration.py
tools/test_qwen35_tp4_engine_remote_execution_plan.py
tools/test_qwen35_tp4_engine_remote_execution_authorization.py
tools/test_qwen35_tp4_engine_remote_execution_receipt.py
tools/test_qwen35_tp4_engine_remote_execution_executor.py
tools/test_qwen35_tp4_engine_remote_execution_source_contract.py
tools/test_qwen35_tp4_engine_remote_subprocess_adapter.py
tools/test_run_qwen35_tp4_engine_correctness_authority.py
tools/test_verify_qwen35_tp4_engine_correctness_authority.py
tools/test_qwen35_tp4_hybrid_prefix_benchmark_contract.py
tools/test_qwen35_tp4_hybrid_prefix_benchmark_worker.py
tools/test_verify_qwen35_tp4_hybrid_prefix_benchmark.py
tools/test_run_qwen35_tp4_hybrid_prefix_benchmark_remote.py
tools/test_build_qwen35_tp4_performance_prerequisites.py
tools/test_qwen35_tp4_cache_snapshot_transport.py
```

No SSH, `scp`, `nvidia-smi`, remote directory creation, subprocess adapter,
Torch, Transformers, CUDA, model load, Engine run, or GPU workload was
executed.

Strict boundary:

```text
cached-specific remote plan/receipt/executor:
  implemented and CPU-tested
single-use authorization binding:
  implemented and CPU-tested
isolated subprocess adapter:
  exists but was not invoked
real cached exact-five artifact:
  absent
real TP4 Engine exact-four authority:
  absent
correctness prerequisite bundle:
  incomplete
canonical performance/cache authority:
  not run
latency/throughput/cache/GPU-memory/compression/quality benefit:
  unmeasured and not claimable
```
