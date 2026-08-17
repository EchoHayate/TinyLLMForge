# Qwen3.5 TP4 Correctness Campaign Preparation Bundle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a pure-local builder and independent verifier that publish one
READY bundle containing three verified child plans/authorizations and one
verified campaign plan/authorization without executing any remote or GPU
operation.

**Architecture:** Reuse the existing production plan and authorization
modules through explicit dependency injection. Build into the final unique
output root, publish `preparation_manifest.json` last, and remove the entire
root on any failure.

**Tech Stack:** Python standard library, canonical JSON/SHA256, atomic file
publication, existing child/campaign plan and authorization modules,
file-local CPU-only tests.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Do not stage, commit, merge, create a branch, or open a PR.
- Do not execute SSH, `scp`, `nvidia-smi`, Torch, Transformers, CUDA, model
  loading, Engine construction, subprocess adapters, or GPU workloads.
- Future remote target remains exactly `sitian@10.232.195.203`.
- Future execution environment remains exactly
  `KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian`.
- Require explicit configuration, source inventory, remote model paths, run
  tags, and authorization nonces.
- Add no subprocess import, runner, executor callback, or execution flag.
- Preserve `benchmark_execution_authorized=false`.
- Do not claim latency, throughput, cache, memory, compression, quality, or
  accuracy benefit.

---

### Task 1: Preparation Manifest Contract

**Files:**
- Create:
  `tools/qwen35_tp4_correctness_authority_campaign_preparation.py`
- Create:
  `tools/test_qwen35_tp4_correctness_authority_campaign_preparation.py`

**Interfaces:**
- Produces:
  `verify_preparation_bundle(path, *, dependencies=None) -> dict`.
- Manifest schema:
  `qwen35.tp4-correctness-campaign-preparation.v1`.

- [x] **Step 1: Write failing verifier tests**

Create synthetic plan/authorization fixtures and require exact closed schema,
root-contained regular paths, exact child/stage order, target/environment,
on-disk SHA checks, active authorization validation, absent future outputs,
and both execution flags false.

- [x] **Step 2: Run verifier tests and require RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 \
python3 tools/test_qwen35_tp4_correctness_authority_campaign_preparation.py
```

Expected: missing-module `FileNotFoundError`.

- [x] **Step 3: Implement the independent verifier**

Use lazy dependency loading and explicit injectable module dependencies.
Reopen child and campaign plans with production verifiers and validate every
active authorization against its reopened plan.

- [x] **Step 4: Run verifier tests and py_compile**

Run the preparation test file and compile the new production/test modules.

### Task 2: Pure-Local Preparation Builder

**Files:**
- Modify:
  `tools/qwen35_tp4_correctness_authority_campaign_preparation.py`
- Modify:
  `tools/test_qwen35_tp4_correctness_authority_campaign_preparation.py`

**Interfaces:**
- Produces:
  `prepare_campaign_bundle(*, repo_root, output_dir, campaign_tag,
  root_run_tag, cached_run_tag, engine_run_tag, configuration_path,
  source_inventory_path, remote_model_dir, remote_model_manifest,
  root_authorization_nonce, cached_authorization_nonce,
  engine_authorization_nonce, campaign_authorization_nonce,
  dependencies=None) -> dict`.

- [x] **Step 1: Write failing builder tests**

Require exact layout, fixed builder order, shared cached/Engine configuration,
fixed runtime output paths, pairwise-distinct tags/nonces, final-manifest-last
publication, and full cleanup after injected failure.

- [x] **Step 2: Run builder tests and require RED**

Expected: verifier-only module lacks `prepare_campaign_bundle`.

- [x] **Step 3: Implement the minimal builder**

Call only existing plan builders/verifiers and authorization
producers/validators. Build the campaign child inventory from fixed
root-contained future paths. Call `verify_preparation_bundle` before
publishing the final manifest.

- [x] **Step 4: Run preparation tests and py_compile**

Expected: all preparation tests pass.

### Task 3: Static Execution-Surface Contract

**Files:**
- Modify:
  `tools/qwen35_tp4_engine_remote_execution_source_contract.py`
- Modify:
  `tools/test_qwen35_tp4_engine_remote_execution_source_contract.py`

**Interfaces:**
- Adds the preparation module to the local AST audit inventory.
- Keeps it excluded from all remote source tar inventories.

- [x] **Step 1: Write failing AST contract test**

Require the module to contain no subprocess import, shell call, dynamic
`exec`, runner/default runner, child executor import, campaign callback
import, or campaign execution call.

- [x] **Step 2: Run source-contract test and require RED**

Expected: preparation module absent from the required local audit inventory.

- [x] **Step 3: Extend the source contract**

Audit the preparation module as a local-only builder while preserving all
existing execution-module checks.

- [x] **Step 4: Run source-contract and preparation tests**

Expected: all tests pass.

### Task 4: Expanded Gate and Handoff

**Files:**
- Modify:
  `docs/superpowers/plans/2026-07-29-qwen35-tp4-correctness-campaign-preparation-bundle.md`
- Modify:
  `docs/superpowers/plans/2026-07-29-qwen35-tp4-hybrid-prefix-performance-cache-authority.md`
- Modify: `AGENT_HANDOFF_STATE.md`

- [x] **Step 1: Run focused integration gate**

Run the preparation suite plus child plan/authorization, campaign
plan/authorization, source-contract, and configuration-builder suites.
Record exact test/file counts.

- [x] **Step 2: Run expanded authority gate**

Start from the exact 50-file campaign authority inventory and add the one
preparation suite. Record exact test/file counts.

- [x] **Step 3: Run static and boundary checks**

Run:

```bash
python3 -m py_compile <preparation and affected files>
git diff --check
test -z "$(git diff --cached --name-only)"
```

Require zero forbidden execution surfaces in the preparation module.

- [x] **Step 4: Update completion audit**

Record that preparation is implemented and CPU-verified, but no real
preparation bundle was produced without explicit real model/configuration
inputs, no campaign executed, no v2 prerequisite bundle exists, and no
performance or accuracy benefit is claimable.

## 2026-07-29 Completion Record

All four implementation tasks and all 16 checklist steps are complete under
the CPU-only boundary.

Implemented:

```text
tools/qwen35_tp4_correctness_authority_campaign_preparation.py
tools/test_qwen35_tp4_correctness_authority_campaign_preparation.py
```

The preparation builder requires explicit configuration, source inventory,
remote model paths, three distinct child run tags, and four distinct
authorization nonces. It freezes copies of the external configuration and
source inventory under `inputs/`, then generates and reopens:

```text
root child plan + active authorization
cached child plan + active authorization
Engine child plan + active authorization
campaign plan + active authorization
```

The final `preparation_manifest.json` is published last. It binds all plan and
authorization paths and SHAs, future runtime output paths, the frozen input
SHAs, source/model/workload/GPU/port identities, exact target/environment,
and both execution flags as false. Any failure removes the entire incomplete
output root.

The independent verifier no longer depends on the original external
configuration or source inventory. It reopens the frozen copies, all child
plans and authorizations, and the campaign plan and authorization. It rejects
any identity drift or any pre-existing consumed authorization, receipt,
failure, authority, adapter, or bundle output.

Authority directories are derived from the verified child plans rather than
from a preparation placeholder: root uses
`stage_inputs.verify.local_artifact_dir`, cached uses its plan-local
`downloaded_cached_authority`, and Engine uses its plan-local
`downloaded_authority`. The verifier re-derives and compares these paths so
the later adapter receives the exact executor output.

TDD and validation evidence:

```text
initial RED:
  preparation module FileNotFoundError
builder RED:
  prepare_campaign_bundle absent
AST contract RED:
  inspect_preparation_source absent
production schema RED:
  external configuration/source inventory were not self-contained
identity closure RED:
  configuration/source identity drift was accepted
execution-path audit RED:
  campaign authority_dir used a preparation placeholder instead of the
  child plan's real downloaded authority path
preparation suite:                     5 passed
source execution contract:             5 passed
focused preparation integration gate: 40 passed across 10 files
expanded preparation authority gate: 324 passed across 51 files
clean-namespace dependency probe:      passed, 8 dependency keys
preparation and affected py_compile:   passed
forbidden execution surfaces:          0 matches
remote source inventory inclusion:     0 matches
git diff --check:                       passed
staged files:                           0
```

No real preparation bundle was produced because no explicit real
configuration/model input was supplied to the builder. No SSH, `scp`,
`nvidia-smi`, subprocess adapter, Torch, Transformers, CUDA, model load,
Engine construction, campaign execution, or GPU workload occurred.

Completion audit:

```text
preparation bundle builder/verifier:
  implemented and CPU-verified
real preparation bundle:
  absent
real campaign execution:
  not run
real root-logit/cached/Engine receipts:
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
