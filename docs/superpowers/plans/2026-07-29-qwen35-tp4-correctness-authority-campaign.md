# Qwen3.5 TP4 Correctness Authority Campaign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a single-use, receipt-bound local coordinator that executes the
existing root-logit, cached-continuation, and Engine correctness authority
protocols serially and publishes a validated v2 prerequisite bundle only
after all three receipt chains pass.

**Architecture:** Build a pure immutable campaign plan around three existing
child plans, then add campaign authorization, receipt, and a
dependency-injected executor. The executor owns no subprocess; it delegates
all child execution and bundle operations to explicit callbacks and publishes
PASS only after independent prerequisite authorization.

**Tech Stack:** Python standard library, canonical JSON/SHA256, atomic rename,
existing child plan/receipt verifiers, real-authority adapter, v2 prerequisite
builder and validator, file-local CPU-only tests.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Do not stage, commit, merge, create a branch, or open a PR.
- Do not execute SSH, `scp`, `nvidia-smi`, Torch, Transformers, CUDA, model
  loading, Engine construction, or GPU workloads.
- Future remote execution target remains exactly
  `sitian@10.232.195.203`.
- Future execution environment remains exactly
  `KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian`.
- Child authority execution is strictly serial.
- Preserve every child authority's existing resource guard, authorization,
  source inventory, receipt, and failure semantics.
- Add no subprocess import or default command/stage runner.
- The campaign must freeze
  `benchmark_execution_authorized=false`.
- Do not claim latency, throughput, cache, memory, compression, quality, or
  accuracy benefit from CPU-only protocol tests.

---

### Task 1: Immutable Campaign Plan

**Files:**
- Create: `tools/qwen35_tp4_correctness_authority_campaign_plan.py`
- Create: `tools/test_qwen35_tp4_correctness_authority_campaign_plan.py`

**Interfaces:**
- Consumes:
  `build_campaign_plan(*, repo_root, output_dir, campaign_tag, children,
  adapter_output_dir, bundle_output_dir)`.
- Produces:
  `campaign_plan.json` and
  `verify_campaign_plan(path, *, child_plan_verifiers) -> dict`.

- [x] **Step 1: Write failing plan tests**

Require exact child order:

```python
assert plan["child_order"] == [
    "tp4_root_logit",
    "cached_continuation",
    "engine_correctness",
]
assert plan["stage_order"] == [
    "root_logit",
    "cached_continuation",
    "engine_correctness",
    "adapt_authorities",
    "build_bundle",
    "verify_bundle",
]
assert plan["ssh_target"] == "sitian@10.232.195.203"
assert plan["execution_env"] == {
    "KRB5CCNAME": "FILE:/Users/bytedance/krb5cc_sitian",
}
assert plan["benchmark_execution_authorized"] is False
```

Each child row must freeze:

```text
name
run_tag
plan_path
plan_sha256
source_tree_sha256
model_manifest_sha256
authority_dir
authorization_path
consumed_authorization_path
receipt_path
failure_path
```

Reject unsafe campaign tags, wrong child inventory/order, missing or symlinked
child plans, child plan verifier failure, child target/env/source/model drift,
duplicate output paths, non-absolute paths, an existing plan output, an
existing adapter/bundle output, existing child receipt/failure/consumed
authorization, or `benchmark_execution_authorized=true`.

- [x] **Step 2: Run plan tests and require RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 \
python3 tools/test_qwen35_tp4_correctness_authority_campaign_plan.py
```

Expected: missing-module `FileNotFoundError`.

- [x] **Step 3: Implement the minimal plan**

Use canonical JSON, regular non-symlink file checks, exact closed schemas, and
atomic publication. The builder receives already verified child-plan
summaries from explicit `child_plan_verifiers` and stores only paths and
identities required by later stages.

- [x] **Step 4: Run plan tests and py_compile**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 \
python3 tools/test_qwen35_tp4_correctness_authority_campaign_plan.py
python3 -m py_compile \
  tools/qwen35_tp4_correctness_authority_campaign_plan.py \
  tools/test_qwen35_tp4_correctness_authority_campaign_plan.py
```

Expected: all tests pass.

### Task 2: Single-Use Campaign Authorization

**Files:**
- Create:
  `tools/qwen35_tp4_correctness_authority_campaign_authorization.py`
- Create:
  `tools/test_qwen35_tp4_correctness_authority_campaign_authorization.py`

**Interfaces:**
- Consumes: a verified campaign plan.
- Produces:
  `produce_authorization(...)`, `validate_authorization(...)`, and
  `consume_authorization(...)`.

- [x] **Step 1: Write failing authorization tests**

Bind:

```text
canonical campaign plan SHA
campaign tag
child order and exact child plan SHAs
SSH target
execution environment
adapter output directory
bundle output directory
safe nonce
consumed=false
benchmark_execution_authorized=false
```

Reject unsafe nonce, plan drift, child plan drift, output drift, target/env
drift, cross-directory consumption, pre-existing consumed path, or reuse.

- [x] **Step 2: Run tests and require RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 \
python3 tools/test_qwen35_tp4_correctness_authority_campaign_authorization.py
```

Expected: missing-module `FileNotFoundError`.

- [x] **Step 3: Implement atomic authorization**

Use canonical bytes and SHA256. Consumption must claim by atomic rename before
rewriting `consumed=true`, and never recreate the active record.

- [x] **Step 4: Run tests and py_compile**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 \
python3 tools/test_qwen35_tp4_correctness_authority_campaign_authorization.py
python3 -m py_compile \
  tools/qwen35_tp4_correctness_authority_campaign_authorization.py \
  tools/test_qwen35_tp4_correctness_authority_campaign_authorization.py
```

Expected: all tests pass.

### Task 3: Campaign Receipt and Failure Evidence

**Files:**
- Create: `tools/qwen35_tp4_correctness_authority_campaign_receipt.py`
- Create:
  `tools/test_qwen35_tp4_correctness_authority_campaign_receipt.py`

**Interfaces:**
- Consumes: verified plan, consumed authorization, six ordered stage results,
  child receipt verifier callbacks, and prerequisite validator callback.
- Produces:
  `produce_campaign_receipt(...)`,
  `validate_campaign_receipt(...)`, and
  `verify_campaign_receipt_files(...)`.

- [x] **Step 1: Write failing receipt tests**

Require:

- exact six-stage order and canonical result hashes;
- consumed campaign authorization;
- each child stage result binds child plan, consumed authorization, receipt,
  authority directory, run tag, source, and model;
- explicit child production receipt verifier returns `PASS`;
- adapter output contains exactly three `AuthorityInput`-equivalent rows;
- bundle stage binds the regular prerequisite file, SHA, and sorted regular
  file inventory;
- independent validator returns `PASS` and `authorized=true`;
- `benchmark_execution_authorized=false`;
- atomic PASS publication.

Reject missing or symlinked evidence, child receipt/run/source/model drift,
child order drift, adapter inventory drift, bundle hash/inventory drift,
validator BLOCKED result, plan/authorization drift, extra stages, or
benchmark authorization drift.

- [x] **Step 2: Run tests and require RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 \
python3 tools/test_qwen35_tp4_correctness_authority_campaign_receipt.py
```

Expected: missing-module `FileNotFoundError`.

- [x] **Step 3: Implement receipt validation**

Reopen and hash every referenced evidence file. Keep child receipt semantics
delegated to the explicit production-verifier callbacks. Do not trust stage
summaries without on-disk evidence.

- [x] **Step 4: Run tests and py_compile**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 \
python3 tools/test_qwen35_tp4_correctness_authority_campaign_receipt.py
python3 -m py_compile \
  tools/qwen35_tp4_correctness_authority_campaign_receipt.py \
  tools/test_qwen35_tp4_correctness_authority_campaign_receipt.py
```

Expected: all tests pass.

### Task 4: Dependency-Injected Campaign Executor

**Files:**
- Create: `tools/qwen35_tp4_correctness_authority_campaign_executor.py`
- Create:
  `tools/test_qwen35_tp4_correctness_authority_campaign_executor.py`

**Interfaces:**
- Consumes:
  `execute_verified_campaign_file(...)` with explicit plan verifier,
  child-executor callbacks, child receipt verifiers, adapter callback, builder
  callback, prerequisite validator, and exact execution environment.
- Produces: campaign PASS receipt or prefix-preserving FAILED evidence.

- [x] **Step 1: Write failing executor tests**

Require:

- exact execution environment;
- no default callback;
- campaign authorization consumed before the first child callback;
- exact serial callback order;
- child callback result must prove `PASS`;
- production child receipt verification after each callback;
- adapter is not called before all three child receipts pass;
- builder is not called before adapter success;
- campaign receipt is not written before independent bundle authorization;
- existing receipt/failure/consumed authorization/adapter/bundle outputs fail
  before consumption;
- first failure stops later callbacks and writes a canonical completed-stage
  prefix;
- callback exception text is bounded;
- benchmark execution remains unauthorized.

- [x] **Step 2: Run executor and AST tests and require RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 \
python3 tools/test_qwen35_tp4_correctness_authority_campaign_executor.py
PYTHONDONTWRITEBYTECODE=1 \
python3 tools/test_qwen35_tp4_engine_remote_execution_source_contract.py
```

Expected: missing executor module and campaign modules absent from the source
contract inventory.

- [x] **Step 3: Implement the executor**

Use explicit callbacks only. Store each successful stage as:

```python
{
    "name": stage_name,
    "result_sha256": canonical_sha(result),
    "result": result,
}
```

On failure, atomically write:

```text
schema_version
classification=FAILED
plan_sha256
authorization_sha256
authorization_nonce
campaign_tag
failed_stage
completed_stages
bounded error
benchmark_execution_authorized=false
```

Do not delete or rewrite child evidence.

- [x] **Step 4: Extend the shared execution source contract**

Add the four campaign production modules. Require executor callables to be
explicit constructor/function parameters and reject `subprocess`, `os.system`,
shell execution, dynamic `exec`, or default runners.

- [x] **Step 5: Run executor/source tests and py_compile**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 \
python3 tools/test_qwen35_tp4_correctness_authority_campaign_executor.py
PYTHONDONTWRITEBYTECODE=1 \
python3 tools/test_qwen35_tp4_engine_remote_execution_source_contract.py
python3 -m py_compile \
  tools/qwen35_tp4_correctness_authority_campaign_executor.py \
  tools/test_qwen35_tp4_correctness_authority_campaign_executor.py \
  tools/qwen35_tp4_engine_remote_execution_source_contract.py \
  tools/test_qwen35_tp4_engine_remote_execution_source_contract.py
```

Expected: all tests pass.

### Task 5: Production Callback Adapter and End-to-End Bundle Fixture

**Files:**
- Create:
  `tools/qwen35_tp4_correctness_authority_campaign_callbacks.py`
- Create:
  `tools/test_qwen35_tp4_correctness_authority_campaign_callbacks.py`
- Modify:
  `tools/qwen35_tp4_engine_remote_execution_source_contract.py`
- Modify:
  `tools/test_qwen35_tp4_engine_remote_execution_source_contract.py`

**Interfaces:**
- Consumes: existing child plan/executor/receipt modules, real-authority
  adapter, prerequisite builder and validator.
- Produces:
  `CampaignCallbacks` containing explicit root/cached/Engine execute and
  verify callbacks plus adapter/build/validate callbacks.

- [x] **Step 1: Write failing callback and end-to-end tests**

Use only synthetic files and injected no-op child executors. Require:

- imports are lazy and CPU-safe;
- root stage callback maps the semantic four-stage runner correctly;
- cached/Engine callbacks use the existing audited command runner interface;
- exact `RealAuthorityRun` rows are produced from plan-bound paths;
- adapter output builds a valid v2 bundle;
- final prerequisite validator returns `PASS` and `authorized=true`;
- the campaign receipt binds the final bundle;
- campaign modules remain local-only and are not added to any remote source
  tar.

- [x] **Step 2: Run tests and require RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 \
python3 tools/test_qwen35_tp4_correctness_authority_campaign_callbacks.py
```

Expected: missing-module `FileNotFoundError`.

- [x] **Step 3: Implement the production callback adapter**

Keep all imports lazy. Do not create a CLI that executes immediately. Export
one constructor that requires the already audited command/stage runner and
returns explicit callbacks for the campaign executor.

- [x] **Step 4: Run focused campaign tests**

Run all five new campaign suites plus:

```text
tools/test_qwen35_tp4_real_prerequisite_authority_adapter.py
tools/test_build_qwen35_tp4_performance_prerequisites.py
tools/test_qwen35_tp4_hybrid_prefix_benchmark_contract.py
tools/test_qwen35_tp4_engine_remote_execution_source_contract.py
```

Record exact test/file counts.

### Task 6: Expanded Gate and Handoff

**Files:**
- Modify:
  `docs/superpowers/plans/2026-07-29-qwen35-tp4-hybrid-prefix-performance-cache-authority.md`
- Modify: `AGENT_HANDOFF_STATE.md`

- [x] **Step 1: Run the expanded selected authority gate**

Start from the current exact 45-file inventory and add the five campaign test
files. Execute each file with:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 "$test_file"
```

- [x] **Step 2: Run static and boundary checks**

Run:

```bash
python3 -m py_compile <all campaign and affected files>
git diff --check
test -z "$(git diff --cached --name-only)"
```

Search production campaign modules and require no:

```text
subprocess
os.system
shell=True
exec(
default command runner
benchmark execution authorization
```

- [x] **Step 3: Update completion audit**

Record:

```text
campaign coordinator:
  implemented and CPU-verified
real campaign execution:
  not run
real three-authority v2 bundle:
  absent
canonical benchmark:
  not run
performance/cache/memory/quality/accuracy benefit:
  unmeasured and not claimable
```

Do not mark the long-term goal complete.

## 2026-07-29 Completion Record

All six implementation tasks and all 24 checklist steps are complete under
the CPU-only boundary.

The campaign is a local semantic coordinator over the three existing
receipt-bound authorities. It freezes:

```text
child order:
  tp4_root_logit
  cached_continuation
  engine_correctness
stage order:
  root_logit
  cached_continuation
  engine_correctness
  adapt_authorities
  build_bundle
  verify_bundle
remote target:
  sitian@10.232.195.203
execution environment:
  KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian
benchmark_execution_authorized:
  false
```

The campaign authorization is canonical-plan-SHA bound and single-use. The
executor consumes it before the first callback, executes child callbacks
strictly serially, independently verifies every child receipt before moving
forward, and publishes bounded prefix-preserving FAILED evidence on the first
error. It owns no subprocess or default runner.

The production callback bridge lazily loads the existing root semantic-stage
executor, cached/Engine command executors, receipt verifiers, real-authority
adapter, v2 prerequisite builder, and independent prerequisite validator.
The final campaign receipt reopens and hashes child plans, consumed
authorizations, receipts, adapter evidence, bundle inventory, and the
independent authorized prerequisite result.

Fresh CPU-only evidence:

```text
campaign suites:                       15 passed across 5 files
focused campaign integration gate:     50 passed across 9 files
expanded campaign authority gate:     318 passed across 50 files
clean-namespace dependency probe:      passed, 12 dependency keys
campaign and affected py_compile:      passed
forbidden execution surfaces:          0 matches
git diff --check:                       passed
staged files:                           0
```

The expanded inventory is the exact previous 45-file CPU authority gate plus
the five campaign suites. No Torch-dependent semantic runtime test was
substituted into that CPU-only inventory.

No SSH, `scp`, `nvidia-smi`, remote directory creation, subprocess adapter
invocation, Torch, Transformers, CUDA, model load, Engine construction, or GPU
workload occurred.

Completion audit remains negative for the long-term performance objective:

```text
campaign coordinator:
  implemented and CPU-verified
real campaign execution:
  not run
real root-logit receipt:
  absent
real cached-continuation receipt:
  absent
real Engine correctness receipt:
  absent
real three-authority v2 prerequisite bundle:
  absent
canonical 70-case TP4 benchmark:
  not run
real all-rank cache/CUDA allocator evidence:
  absent
latency/throughput/cache/GPU-memory/compression/quality/accuracy benefit:
  unmeasured and not claimable
```

