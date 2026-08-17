# Qwen3.5 TP4 Root-Logit Remote Receipt Authority Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wrap the existing TP4 real root-logit remote runner in an immutable,
single-use, receipt-bound four-stage protocol and remove the root-logit
provenance gap from the performance prerequisite bundle.

**Architecture:** Preserve the frozen root-logit source and mature runner.
Add pure plan, authorization, receipt, and dependency-injected executor
modules around its `preflight`, `run`, `download`, and `verify` callbacks.
Then require the verified receipt chain in the real prerequisite adapter and
copy it into the self-contained v2 bundle.

**Tech Stack:** Python standard library, canonical JSON/SHA256, atomic rename,
existing root-logit runner/verifier, shared prerequisite semantic validator,
file-local CPU-only tests.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Do not stage, commit, merge, create a branch, or open a PR.
- Do not modify the frozen root-logit source identified by
  `qwen35-tp4-source-prep-20260729-010400`.
- Do not execute SSH, `scp`, `nvidia-smi`, Torch, Transformers, CUDA, model
  loading, Engine construction, or GPU workloads.
- Future remote execution target remains exactly
  `sitian@10.232.195.203`.
- Future execution environment remains exactly
  `KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian`.
- Preserve the four-unique-GPU requirement, 24 GiB free-memory floor, and
  active-compute-process rejection.
- Do not claim performance, cache, memory, compression, quality, or accuracy
  benefit from CPU-only protocol tests.

---

### Task 1: Immutable Root-Logit Execution Plan

**Files:**
- Create: `tools/qwen35_tp4_root_logit_remote_execution_plan.py`
- Create: `tools/test_qwen35_tp4_root_logit_remote_execution_plan.py`

**Interfaces:**
- Consumes:
  `build_remote_execution_plan(*, repo_root, output_dir, run_tag)`.
- Produces:
  `remote_execution_plan.json` and
  `verify_remote_execution_plan(path) -> dict`.

- [x] **Step 1: Write the failing plan tests**

Require:

```python
assert plan["stage_order"] == [
    "preflight",
    "run",
    "download",
    "verify",
]
assert plan["ssh_target"] == "sitian@10.232.195.203"
assert plan["frozen_source_tree_sha256"] == (
    "b2d0b77de953e273dbf62f0e7b2bbe689ef33c183edf65830940e43123bb485f"
)
assert plan["exact_artifact_names"] == sorted(
    runner.EXACT_ARTIFACT_NAMES
)
assert plan["execution_performed"] is False
```

Also reject unsafe run tags, an existing plan output, an existing local run
directory, changed runner constants, changed stage order, changed model SHA,
or changed resource policy.

- [x] **Step 2: Run the test and require RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 \
python3 tools/test_qwen35_tp4_root_logit_remote_execution_plan.py
```

Expected: missing-module `FileNotFoundError`.

- [x] **Step 3: Implement the minimal plan builder/verifier**

Use only standard-library imports and lazy-load:

```python
run_qwen35_tp4_real_root_logit_gate_remote.py
qwen35_tp4_hybrid_prefix_benchmark_contract.py
```

Freeze:

```text
schema_version
run_tag
repo_root
local_run_dir
ssh_target
remote_run_dir
frozen_source_tag
frozen_source_tree_sha256
model_manifest_sha256
exact_artifact_names
minimum_free_bytes_per_gpu
requires_no_active_compute_processes
stage_order
stage_inputs
execution_performed
claim_boundary
```

Publish atomically and verify the canonical closed schema.

- [x] **Step 4: Run plan tests and py_compile**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 \
python3 tools/test_qwen35_tp4_root_logit_remote_execution_plan.py
python3 -m py_compile \
  tools/qwen35_tp4_root_logit_remote_execution_plan.py \
  tools/test_qwen35_tp4_root_logit_remote_execution_plan.py
```

Expected: all tests pass.

### Task 2: Single-Use Root-Logit Authorization

**Files:**
- Create:
  `tools/qwen35_tp4_root_logit_remote_execution_authorization.py`
- Create:
  `tools/test_qwen35_tp4_root_logit_remote_execution_authorization.py`

**Interfaces:**
- Consumes: a verified plan dictionary.
- Produces:
  `produce_authorization(...)`, `validate_authorization(...)`, and
  `consume_authorization(...)`.

- [x] **Step 1: Write failing authorization tests**

Require the payload to bind:

```text
canonical plan SHA
run tag
SSH target
frozen source SHA
model manifest SHA
exact stage order
safe nonce
consumed=false
```

Reject unsafe nonce, plan/source/model/stage drift, reuse, cross-directory
consumption, and a pre-existing consumed path.

- [x] **Step 2: Run the test and require RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 \
python3 tools/test_qwen35_tp4_root_logit_remote_execution_authorization.py
```

Expected: missing-module `FileNotFoundError`.

- [x] **Step 3: Implement closed-schema atomic authorization**

Use canonical JSON bytes and SHA256. `consume_authorization()` must:

1. validate the active record;
2. require active and consumed paths in one directory;
3. atomically rename active to consumed;
4. rewrite the claimed record with `consumed=true`;
5. never recreate the active path.

- [x] **Step 4: Run authorization tests and py_compile**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 \
python3 tools/test_qwen35_tp4_root_logit_remote_execution_authorization.py
python3 -m py_compile \
  tools/qwen35_tp4_root_logit_remote_execution_authorization.py \
  tools/test_qwen35_tp4_root_logit_remote_execution_authorization.py
```

Expected: all tests pass.

### Task 3: Root-Logit Execution Receipt

**Files:**
- Create: `tools/qwen35_tp4_root_logit_remote_execution_receipt.py`
- Create:
  `tools/test_qwen35_tp4_root_logit_remote_execution_receipt.py`

**Interfaces:**
- Consumes: verified plan, consumed authorization, four canonical stage
  result dictionaries, and the plan-local run directory.
- Produces:
  `produce_execution_receipt(...)`,
  `validate_execution_receipt(...)`, and
  `verify_receipt_files(...)`.

- [x] **Step 1: Build a complete synthetic local run fixture**

Create:

```text
<local_run_dir>/remote_resource_preflight.json
<local_run_dir>/remote_run.json
<local_run_dir>/download.json
<local_run_dir>/independent_verification.json
<local_run_dir>/artifacts/<exact five files>
```

Use the existing root verifier test fixture helpers or a minimal valid fixture
accepted by `validate_authority_documents(...)`; do not mock the shared
semantic validator.

- [x] **Step 2: Write failing receipt tests**

Require:

- exact stage order and canonical stage-result hashes;
- consumed authorization SHA and nonce;
- preflight `READY`, four ranks, four unique GPUs/UUIDs, >=24 GiB free, and
  empty compute processes;
- run/download exact-five inventory;
- only regular non-symlink artifact files;
- on-disk evidence equals each supplied stage result;
- verifier payload `PASS`, exact case IDs, ranks `[0,1,2,3]`, positive checks;
- root artifact passes `validate_authority_documents(...)`;
- source manifest source SHA equals the plan;
- atomic PASS publication.

Reject blocked preflight, changed GPU identity, extra/missing artifact, changed
verification payload, changed root semantic evidence, source drift, plan
drift, and authorization drift.

- [x] **Step 3: Run the test and require RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 \
python3 tools/test_qwen35_tp4_root_logit_remote_execution_receipt.py
```

Expected: missing-module `FileNotFoundError`.

- [x] **Step 4: Implement the minimal receipt**

Lazy-load the benchmark contract and validate:

```python
contract.validate_authority_documents(
    "tp4_root_logit",
    artifact_payload,
    verification_payload,
    plan["frozen_source_tree_sha256"],
)
```

The receipt summary must include:

```text
classification=PASS
run_tag
plan_sha256
authorization_sha256
authorization_nonce
source_tree_sha256
model_manifest_sha256
case_ids
ranks
checks
artifact_names
stage_count
```

- [x] **Step 5: Run receipt tests and py_compile**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 \
python3 tools/test_qwen35_tp4_root_logit_remote_execution_receipt.py
python3 -m py_compile \
  tools/qwen35_tp4_root_logit_remote_execution_receipt.py \
  tools/test_qwen35_tp4_root_logit_remote_execution_receipt.py
```

Expected: all tests pass.

### Task 4: Dependency-Injected Root-Logit Executor

**Files:**
- Create: `tools/qwen35_tp4_root_logit_remote_execution_executor.py`
- Create:
  `tools/test_qwen35_tp4_root_logit_remote_execution_executor.py`
- Modify: `tools/qwen35_tp4_engine_remote_execution_source_contract.py`
- Modify:
  `tools/test_qwen35_tp4_engine_remote_execution_source_contract.py`

**Interfaces:**
- Consumes:
  `execute_verified_plan_file(*, plan_path, authorization_path,
  consumed_authorization_path, receipt_path, failure_path, stage_runner,
  plan_verifier, execution_env)`.
- Produces: PASS receipt or bounded prefix-preserving FAILED evidence.

- [x] **Step 1: Write failing executor tests**

Define an explicit callback:

```python
def stage_runner(*, name, plan, execution_env):
    ...
```

Require:

- no default runner;
- exact KRB5CCNAME;
- plan verification before authorization consumption;
- authorization consumption before the first callback;
- exact callback order;
- rejection of pre-existing receipt/failure/consumed/local-run targets;
- PASS publication only after all four stages;
- blocked preflight and any exception produce no PASS;
- FAILED evidence contains exactly the completed prefix and next failed stage.

- [x] **Step 2: Run the executor and AST tests and require RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 \
python3 tools/test_qwen35_tp4_root_logit_remote_execution_executor.py
PYTHONDONTWRITEBYTECODE=1 \
python3 tools/test_qwen35_tp4_engine_remote_execution_source_contract.py
```

Expected: missing module and missing AST inventory failures.

- [x] **Step 3: Implement the executor and source contract**

The executor imports no subprocess module and has no `main`. It records each
stage as:

```python
{
    "name": name,
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
run_tag
failed_stage
completed_stages
error
```

Add all four root protocol modules to `SOURCE_NAMES` and require explicit
runner, plan verifier, and execution environment in the AST contract.

- [x] **Step 4: Run executor/source tests and py_compile**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 \
python3 tools/test_qwen35_tp4_root_logit_remote_execution_executor.py
PYTHONDONTWRITEBYTECODE=1 \
python3 tools/test_qwen35_tp4_engine_remote_execution_source_contract.py
python3 -m py_compile \
  tools/qwen35_tp4_root_logit_remote_execution_executor.py \
  tools/test_qwen35_tp4_root_logit_remote_execution_executor.py \
  tools/qwen35_tp4_engine_remote_execution_source_contract.py \
  tools/test_qwen35_tp4_engine_remote_execution_source_contract.py
```

Expected: all tests pass.

### Task 5: Receipt-Bound Prerequisite Adapter Integration

**Files:**
- Modify: `tools/qwen35_tp4_real_prerequisite_authority_adapter.py`
- Modify: `tools/test_qwen35_tp4_real_prerequisite_authority_adapter.py`
- Modify: `tools/build_qwen35_tp4_performance_prerequisites.py`
- Modify: `tools/test_build_qwen35_tp4_performance_prerequisites.py`
- Modify: `tools/qwen35_tp4_hybrid_prefix_benchmark_contract.py`
- Modify:
  `tools/test_qwen35_tp4_hybrid_prefix_benchmark_contract.py`

**Interfaces:**
- Consumes: root `RealAuthorityRun` with plan, consumed authorization, and
  receipt paths.
- Produces: root provenance with
  `binding_kind=remote_execution_receipt` and
  `root_logit_receipt_gap=false`.

- [x] **Step 1: Write failing adapter and bundle tests**

Require root behavior to match cached/Engine:

```python
assert provenance["binding_kind"] == "remote_execution_receipt"
assert provenance["root_logit_receipt_gap"] is False
assert provenance["plan_path"]
assert provenance["authorization_path"]
assert provenance["receipt_path"]
```

Reject missing root receipt chain, unconsumed authorization, plan-local
directory drift, run-tag/source/model mismatch, and legacy
`complete_directory_only` provenance.

- [x] **Step 2: Run focused tests and require RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 \
python3 tools/test_qwen35_tp4_real_prerequisite_authority_adapter.py
PYTHONDONTWRITEBYTECODE=1 \
python3 tools/test_build_qwen35_tp4_performance_prerequisites.py
PYTHONDONTWRITEBYTECODE=1 \
python3 tools/test_qwen35_tp4_hybrid_prefix_benchmark_contract.py
```

Expected: root receipt/provenance assertions fail.

- [x] **Step 3: Implement one receipt-bound provenance path**

Extend `VerifierDependencies` with root plan/receipt verifiers. Use
`_receipt_bound(...)` for root, compare receipt source/model/run-tag summary
with the independently verified authority, copy root plan/authorization/
receipt into verification output, and remove the root special case from:

```text
_provenance_payload(...)
build_prerequisite_bundle(...)
_validate_prerequisite_provenance(...)
```

All three authorities must now require and copy the same three evidence file
types.

- [x] **Step 4: Run adapter/builder/contract tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 \
python3 tools/test_qwen35_tp4_real_prerequisite_authority_adapter.py
PYTHONDONTWRITEBYTECODE=1 \
python3 tools/test_build_qwen35_tp4_performance_prerequisites.py
PYTHONDONTWRITEBYTECODE=1 \
python3 tools/test_qwen35_tp4_hybrid_prefix_benchmark_contract.py
```

Expected: all tests pass.

### Task 6: Expanded Authority Gate and Handoff

**Files:**
- Modify:
  `docs/superpowers/plans/2026-07-29-qwen35-tp4-hybrid-prefix-performance-cache-authority.md`
- Modify: `AGENT_HANDOFF_STATE.md`

**Interfaces:**
- Consumes: Tasks 1-5.
- Produces: exact CPU-only validation record and updated next blocker.

- [x] **Step 1: Run focused root protocol gate**

Run all new root protocol suites plus:

```text
tools/test_run_qwen35_tp4_real_root_logit_gate_remote.py
tools/test_qwen35_tp4_real_prerequisite_authority_adapter.py
tools/test_build_qwen35_tp4_performance_prerequisites.py
tools/test_qwen35_tp4_hybrid_prefix_benchmark_contract.py
tools/test_qwen35_tp4_engine_remote_execution_source_contract.py
```

- [x] **Step 2: Run the expanded selected authority gate**

Start from the current exact 41-file inventory and add the four new root
protocol test files. Execute every file with:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 "$test_file"
```

Record exact test and file counts from command output.

- [x] **Step 3: Run static checks**

Run:

```bash
python3 -m py_compile <all changed production and test files>
git diff --check
test -z "$(git diff --cached --name-only)"
```

Expected:

```text
py_compile: passed
git diff --check: passed
staged files: 0
```

- [x] **Step 4: Update completion audit**

Record:

```text
root-logit receipt protocol:
  implemented and CPU-verified
root_logit_receipt_gap:
  structurally closed for future real runs
real root-logit receipt:
  absent
real three-authority v2 bundle:
  absent
canonical benchmark:
  not run
performance/cache/memory/quality/accuracy benefit:
  unmeasured and not claimable
```

Do not mark the long-term goal complete.

## 2026-07-29 Completion Record

All six implementation tasks are complete under the CPU-only boundary.

The root-logit authority now uses the same provenance contract as cached
continuation and Engine correctness:

```text
binding_kind:
  remote_execution_receipt
root_logit_receipt_gap:
  false
required copied evidence:
  execution plan
  consumed authorization
  execution receipt
```

The adapter verifies the immutable root plan and execution receipt, binds the
receipt run tag/source/model identities to the independently verified root
authority, and copies all three evidence files. The prerequisite builder,
runtime contract, benchmark worker fixtures, assembler, and final independent
verifier all require, preserve, and rehash the same evidence for all three
authorities. Production code contains no remaining root-only
`complete_directory_only` or evidence-skip path; legacy provenance appears
only in rejection tests.

Strict TDD and integration evidence:

```text
root plan RED:
  missing module FileNotFoundError
root authorization RED:
  missing module FileNotFoundError
root receipt RED:
  missing module FileNotFoundError
root executor RED:
  missing module FileNotFoundError
adapter integration RED:
  VerifierDependencies rejected root_plan_verify
builder/contract RED:
  tp4_root_logit provenance is invalid
expanded integration RED:
  worker fixture still emitted legacy root provenance
assembler/verifier RED:
  root receipt evidence was missing from final nested artifact
focused root/authority gate:
  56 passed across 9 files
adapter + builder + contract:
  31 passed across 3 files
expanded selected authority gate:
  303 passed across 45 files
focused py_compile:
  passed
git diff --check:
  passed
staged files:
  0
```

No SSH, `scp`, `nvidia-smi`, remote directory creation, subprocess execution,
Torch, Transformers, CUDA, model load, Engine construction, or GPU workload
occurred.

Completion audit remains negative for the long-term performance goal:

```text
root-logit receipt protocol:
  implemented and CPU-verified
root_logit_receipt_gap:
  structurally closed for future real runs
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
performance/cache/memory/quality/accuracy benefit:
  unmeasured and not claimable
```

The next blocker is real correctness authority production on the approved
remote target, followed by adapter/bundle publication. Benchmark execution
must remain unauthorized until that real v2 bundle independently validates.
