# Qwen3.5 TP4 Remote Authority Configuration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a manifest-bound remote TP4 authority configuration and use it to publish a real local `READY` correctness-campaign preparation bundle without local model weights or remote execution.

**Architecture:** Add a separate `build_remote_configuration(...)` entry point beside the unchanged local builder. Both paths share deterministic private publication code only after mode-specific model-path validation. The real artifact flow passes every runtime parameter explicitly, then invokes and independently verifies the existing pure-local preparation builder.

**Tech Stack:** Python standard library, canonical JSON, SHA-256, existing TinyLLMForge authority configuration and campaign preparation modules.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Do not stage, commit, merge, create a branch, or open a PR.
- Do not use SSH, `scp`, `nvidia-smi`, Torch, Transformers, CUDA, model loading, Engine construction, subprocess adapter execution, or GPU workloads.
- Preserve `build_configuration(...)` local model-directory validation unchanged.
- Require explicit values for GPU indices, ports, cache limits, timeout, and model fingerprint.
- Keep remote target `sitian@10.232.195.203` and execution environment `KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian`.
- Do not modify `experiments/qwen35_hybrid_state/qwen35-tp4-source-prep-20260729-010400`.
- Do not claim correctness, accuracy, performance, cache, memory, compression, or quality gains.

---

### Task 1: Remote Manifest-Bound Builder

**Files:**
- Modify: `tools/test_build_qwen35_tp4_engine_authority_configuration.py`
- Modify: `tools/build_qwen35_tp4_engine_authority_configuration.py`

**Interfaces:**
- Consumes: a regular local model manifest whose JSON object contains an absolute `remote_model_dir`.
- Produces: `build_remote_configuration(*, repo_root, output_dir, model_manifest_path, remote_model_dir, model_fingerprint, gpu_indices, dist_port, master_port, max_cache_entries, max_cache_bytes, timeout_s) -> dict`.

- [x] **Step 1: Write failing success and identity tests**

Add a test that creates only a repository file and model manifest, deliberately
does not create local model weights, calls `build_remote_configuration(...)`,
and asserts the emitted `ExecutorConfiguration` contains the explicit remote
directory plus exact manifest, source, workload, GPU, port, cache, timeout,
and fingerprint identities.

- [x] **Step 2: Run the focused test and verify RED**

Run:

```bash
python tools/test_build_qwen35_tp4_engine_authority_configuration.py
```

Expected: FAIL because `build_remote_configuration` does not exist.

- [x] **Step 3: Implement minimal remote builder and shared publication**

Add strict JSON-object loading and POSIX absolute-path validation. Refactor the
existing deterministic output body into a private helper that accepts an
already validated model-directory string. Keep the existing
`build_configuration(...)` preconditions and signature intact.

- [x] **Step 4: Run the focused test and verify GREEN**

Run:

```bash
python tools/test_build_qwen35_tp4_engine_authority_configuration.py
```

Expected: all builder tests pass.

- [x] **Step 5: Write failing rejection tests**

Add separate tests for malformed manifest JSON, missing manifest
`remote_model_dir`, relative explicit/manifest paths, path mismatch, symlinked
manifest, and cleanup after a source-inventory failure. Each test must assert
that no output directory remains.

- [x] **Step 6: Run rejection tests and verify RED**

Run the builder test file and confirm at least one new rejection assertion
fails because validation is not yet complete.

- [x] **Step 7: Implement strict fail-closed validation**

Make every invalid manifest/path case raise `ValueError` before publication,
and preserve temporary-directory cleanup for failures after publication starts.

- [x] **Step 8: Run builder tests and verify GREEN**

Run:

```bash
python tools/test_build_qwen35_tp4_engine_authority_configuration.py
```

Expected: all tests pass with no warnings.

### Task 2: Explicit CLI Mode Selection

**Files:**
- Modify: `tools/test_build_qwen35_tp4_engine_authority_configuration.py`
- Modify: `tools/build_qwen35_tp4_engine_authority_configuration.py`

**Interfaces:**
- Consumes: exactly one of `--model-dir` or `--remote-model-dir`.
- Produces: the existing canonical JSON stdout payload and atomically published configuration directory.

- [x] **Step 1: Write failing CLI dispatch tests**

Call `main([...])` with local mode and remote mode, assert each dispatches to
the correct builder, and assert argparse rejects supplying both or neither.

- [x] **Step 2: Run focused tests and verify RED**

Run the builder test file. Expected: FAIL because the current parser requires
`--model-dir` and has no remote mode.

- [x] **Step 3: Implement mutually exclusive CLI arguments**

Use one required mutually exclusive argparse group containing `--model-dir`
and `--remote-model-dir`. Dispatch to `build_configuration(...)` or
`build_remote_configuration(...)` while passing all other explicit arguments
unchanged.

- [x] **Step 4: Run focused tests and verify GREEN**

Run the builder test file. Expected: all tests pass.

### Task 3: Real Local Configuration and READY Bundle

**Files:**
- Create: `experiments/qwen35_hybrid_state/qwen35-tp4-remote-authority-config-20260729-132532/executor_configuration.json`
- Create: `experiments/qwen35_hybrid_state/qwen35-tp4-remote-authority-config-20260729-132532/workload_manifest.json`
- Create: `experiments/qwen35_hybrid_state/qwen35-tp4-remote-authority-config-20260729-132532/source_inventory.json`
- Create: `experiments/qwen35_hybrid_state/qwen35-tp4-correctness-campaign-preparation-20260729-132532/preparation_manifest.json`
- Update: `docs/superpowers/plans/2026-07-29-qwen35-tp4-remote-authority-configuration.md`

**Interfaces:**
- Consumes: canonical local manifest SHA `3e650a908234771c3cf1ac4e20c4d38fe69982efedaf4a3e631ad0b14aad7dd0`, canonical remote model and manifest paths, fresh run tags, and four unique nonces.
- Produces: a real local authority configuration and an independently verified `qwen35.tp4-correctness-campaign-preparation.v1` `READY` bundle.

- [x] **Step 1: Generate the real local remote configuration**

Invoke `build_remote_configuration(...)` from Python with explicit:

```text
gpu_indices=(0,1,2,3)
dist_port=31001
master_port=31002
max_cache_entries=8
max_cache_bytes=1073741824
timeout_s=600.0
model_fingerprint=3e650a908234771c3cf1ac4e20c4d38fe69982efedaf4a3e631ad0b14aad7dd0
```

Use the canonical local manifest and its exact manifest-bound remote model
directory.

- [x] **Step 2: Verify configuration identities**

Load the emitted payload through `ExecutorConfiguration`, verify the canonical
manifest SHA, verify the source inventory matches the configuration source
SHA, and verify the workload manifest SHA.

- [x] **Step 3: Generate the real local preparation bundle**

Call `prepare_campaign_bundle(...)` with:

```text
remote_model_dir=/data00/home/sitian/sitian-workspace01/tllm/qwen35-hybrid-state-runs/qwen35-2b-hybrid-acquire-20260723-222004/model
remote_model_manifest=/data00/home/sitian/sitian-workspace01/tllm/qwen35-hybrid-state-runs/qwen35-2b-hybrid-acquire-20260723-222004/model_manifest.json
```

Use three pairwise-distinct fresh child run tags and four pairwise-distinct
fresh authorization nonces.

- [x] **Step 4: Independently reopen the READY manifest**

Start a fresh Python process, call `verify_preparation_bundle(...)`, and assert:

```text
classification=READY
execution_performed=false
benchmark_execution_authorized=false
ssh_target=sitian@10.232.195.203
model_manifest_sha256=3e650a908234771c3cf1ac4e20c4d38fe69982efedaf4a3e631ad0b14aad7dd0
```

- [x] **Step 5: Record exact artifact paths and hashes**

Append a completion record to this plan containing the configuration directory,
preparation directory, configuration SHA, source-tree SHA, workload SHA,
preparation manifest SHA, child run tags, and the no-execution claim boundary.

Completion record:

```text
configuration_dir:
  experiments/qwen35_hybrid_state/qwen35-tp4-remote-authority-config-20260729-132532
configuration_sha256:
  1a524173e1be49c8b6e7fc9540e5827d55d278c14184727711f5735635d2712c
model_manifest_sha256:
  3e650a908234771c3cf1ac4e20c4d38fe69982efedaf4a3e631ad0b14aad7dd0
source_tree_sha256:
  935e6406a8eda96566094affb8ee3b054cf31c4f3b9c44045fb9db4c1a5b3dce
workload_manifest_sha256:
  d8c81d6efa73f9b5e20dd0019e7e2dbf34e9f2ce4cef60658b0c44f3ca9648c2
preparation_dir:
  experiments/qwen35_hybrid_state/qwen35-tp4-correctness-campaign-preparation-20260729-132532
preparation_manifest_sha256:
  b3d566d7a3877570577e97eacd39c3acfbe59e79b0049142a7d9d5f8fa707e5c
campaign_tag:
  qwen35-tp4-correctness-campaign-20260729-132532
child_run_tags:
  qwen35-tp4-root-logit-20260729-132532
  qwen35-tp4-cached-continuation-20260729-132532
  qwen35-tp4-engine-correctness-20260729-132532
classification:
  READY
execution_performed:
  false
benchmark_execution_authorized:
  false
claim_boundary:
  preparation only; no remote execution, correctness, performance, cache,
  memory, compression, or quality claim
```

### Task 4: Regression and Handoff Audit

**Files:**
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify: `docs/superpowers/plans/2026-07-29-qwen35-tp4-hybrid-prefix-performance-cache-authority.md`

**Interfaces:**
- Consumes: the completed builder tests and real READY artifact identities.
- Produces: a precise continuation record that distinguishes preparation from execution evidence.

- [x] **Step 1: Run focused and adjacent authority tests**

Run the builder, preparation, remote-plan, source-contract, campaign-plan, and
campaign-authorization test files.

- [x] **Step 2: Run the expanded previously established authority gate**

Run the same 51-file expanded CPU-only gate used by the preparation completion
audit, plus the modified builder test.

- [x] **Step 3: Run static safety checks**

Run `py_compile`, AST/forbidden-surface checks, `git diff --check`, confirm no
staged files, and confirm no SSH/GPU command was invoked.

- [x] **Step 4: Update handoff and objective audit**

Record that the real local preparation bundle is now present and READY, but the
real correctness campaign, three authority receipts, v2 prerequisite bundle,
canonical benchmark, and all performance/accuracy conclusions remain absent.

- [x] **Step 5: Final plan self-review**

Check every task box only after fresh evidence exists, remove any ambiguity or
placeholder language, and append exact verification counts.

## Completion Audit

All four tasks and all 17 checklist steps are complete under the local-only
boundary.

Fresh verification:

```text
remote configuration builder suite:   8 passed
preparation suite:                     5 passed
focused authority gate:               44 passed across 10 files
expanded authority gate:              328 passed across 51 files
clean-namespace dependency probe:      passed, 8 dependency keys
real READY manifest fresh-process reopen:
  passed
py_compile:
  passed
AST forbidden imports:
  0 matches
forbidden execution surfaces:
  0 matches
git diff --check:
  passed
staged files:
  0
```

The real local artifacts are present:

```text
remote authority configuration:
  experiments/qwen35_hybrid_state/qwen35-tp4-remote-authority-config-20260729-132532
READY campaign preparation:
  experiments/qwen35_hybrid_state/qwen35-tp4-correctness-campaign-preparation-20260729-132532
```

The preparation manifest binds three child plans, three child authorizations,
one campaign plan, and one campaign authorization. The manifest independently
reopens as `READY`, but both execution flags remain false.

Objective boundary after this task:

```text
manifest-bound remote configuration:
  implemented and verified
real local READY preparation:
  present and independently verified
real correctness campaign:
  not run
real root-logit/cached/Engine receipts:
  absent
real three-authority v2 prerequisite bundle:
  absent
canonical 70-case TP4 benchmark:
  not run
speed improvement:
  unmeasured
cache or physical-memory reduction:
  unmeasured
quality or accuracy preservation:
  not established by a real run
```
