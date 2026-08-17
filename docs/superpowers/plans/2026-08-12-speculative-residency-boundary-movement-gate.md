# Speculative Residency Boundary and Movement Gate Implementation Plan

> **For agentic workers:** Execute inline in the current worktree. Follow every RED/GREEN step in order. Do not commit, stage, switch branches, stash, reset, push, or clean.

**Goal:** Prove loaded-model accepted and rejected speculative reserved-block behavior at a 256-token boundary together with real MVP-0 H2D reload.

**Architecture:** Add a validated clean-resident eviction primitive to `KVOffloadMVP0`. Build a separate TP1 gate using pretokenized 254-token prompts and deterministic source-agnostic adapters that force one accepted or rejected proposal after prefill. Preserve the existing schema-v2 parity artifact and produce a new independently verified correctness-only artifact.

**Tech Stack:** Python 3, PyTorch/CUDA, TinyLLMForge generic speculative runtime, `KVOffloadMVP0`, JSON, SHA-256, pytest, SSH ControlMaster.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Keep generic runtime, Scheduler, verifier, and allocator model-name-free.
- Use only real `KVOffloadMVP0` movement counters.
- Accepted KV commits in place; rejected suffix rolls back without D2H.
- Do not claim TPOT, TTFT, throughput, memory, long-context, TP4, learned-drafter, or MTP improvement.
- Keep classification `NOT_PROMOTABLE`.
- Use `sitian@10.232.195.203`, GPU 0, and the existing SSH control socket.
- Do not commit, stage, switch branches, stash, reset, push, or run `git clean`.

---

### Task 1: Add Clean CPU-Backed Eviction

**Files:**
- Modify: `tinyvllm/engine/model_runner.py`
- Modify: `tools/test_kv_offload.py`

**Interfaces:**
- Consumes: generation-bound `KVOffloadMVP0` mappings and existing CPU-valid/writeback state.
- Produces: `KVOffloadMVP0.evict_clean_resident_blocks(block_identities)`.

- [x] **Step 1: Add failing direct-manager tests**

Cover:

```text
clean CPU-valid block unmaps while generation and CPU validity remain
batch validation is atomic
dirty block fails before mutation
stale generation fails before mutation
missing CPU backing fails before mutation
duplicate identity fails before mutation
evictions and evict_clean increment exactly once
```

- [x] **Step 2: Run focused RED**

```bash
/opt/homebrew/bin/python3.12 tools/test_kv_offload.py
```

If local `flash_attn` is unavailable, load the dependency-light
`KVOffloadMVP0` class through the existing AST pattern used by
`tools/test_kv_offload_generation_metadata.py`, and run the focused pytest
locally before remote direct validation.

- [x] **Step 3: Implement the minimal primitive**

Validate the full identity tuple before changing mappings. Require exact
generation, residency, clean state, CPU validity, and no pending H2D. Remove
the logical/slot mapping and completed event metadata, preserve
`bound_generations` and `cpu_valid`, and update `evictions` plus
`evict_clean`.

- [x] **Step 4: Run GREEN**

```bash
python3 -m pytest tools/test_kv_offload_generation_metadata.py -q
```

Then synchronize `tools/test_kv_offload.py` and run it with the remote CUDA
Python.

---

### Task 2: Build Deterministic Boundary Case Logic

**Files:**
- Create: `tools/speculative_residency_boundary_gate.py`
- Create: `tools/test_speculative_residency_boundary_gate.py`

**Interfaces:**
- Produces: `BoundaryDraftAdapter(mode)`, `run_boundary_case(...)`, and
  `build_boundary_artifact(...)`.

- [x] **Step 1: Add failing dependency-light tests**

Test:

```text
accept adapter proposes exactly first_target_token
reject adapter always proposes a different valid token
prompt token input must contain exactly 254 tokens
prefill transition must expose one live sequence at length 255
eviction orchestration calls writeback -> synchronize -> clean eviction
accepted case requires committed_blocks > 0 and exact parity
rejected case requires rejected_blocks > 0 and rejected_d2h_copies == 0
both speculative cases require h2d_copies/h2d_bytes > 0
```

- [x] **Step 2: Run RED**

```bash
python3 -m pytest tools/test_speculative_residency_boundary_gate.py -q
```

- [x] **Step 3: Implement the gate core**

Use a three-token `DraftCapabilities` contract with source type
`boundary_fixture`. Pass prompt IDs directly to `LLMEngine.add_request()`.
Run one baseline engine, derive its three-token continuation suffix after the
prefill output, and pass that suffix to separate accepted/rejected speculative
engines. After the first step, validate length 255 and force clean eviction
through the new public manager primitive. Three tokens are required because
allocator reservation covers `proposal_count - 1` materialized tail tokens.

- [x] **Step 4: Run GREEN**

```bash
python3 -m pytest tools/test_speculative_residency_boundary_gate.py -q
```

---

### Task 3: Add Independent Artifact Verification

**Files:**
- Create: `tools/verify_speculative_residency_boundary_gate.py`
- Modify: `tools/test_speculative_residency_boundary_gate.py`

**Interfaces:**
- Produces: `verify_boundary_artifact(artifact_path, repo_root)`.

- [x] **Step 1: Add failing schema/verifier tests**

Reject token divergence, missing source files, source hash mismatch,
non-positive H2D, missing committed/rejected evidence, nonzero rejected D2H,
negative/non-integer counters, and interpreted performance claims.

- [x] **Step 2: Run RED**

```bash
python3 -m pytest tools/test_speculative_residency_boundary_gate.py -q
```

- [x] **Step 3: Implement schema version 1 and verifier**

The verifier imports only the gate's pure validation function and recomputes
every recorded source SHA-256.

- [x] **Step 4: Run GREEN and compile**

```bash
python3 -m pytest tools/test_speculative_residency_boundary_gate.py -q
python3 -m py_compile \
  tools/speculative_residency_boundary_gate.py \
  tools/verify_speculative_residency_boundary_gate.py
```

---

### Task 4: Run the Real Remote Boundary Gate

**Files:**
- Create: `tools/run_speculative_residency_boundary_gate_remote.sh`

**Interfaces:**
- Produces: one tagged artifact directory containing `result.json`,
  `remote.log`, `verify.remote.json`, and `verify.json`.

- [x] **Step 1: Add remote-runner source tests**

Require full `tinyvllm/` sync, gate/verifier/test sync, fixed remote Python,
model path, GPU 0, always-downloaded logs, and independent remote/local
verification.

- [x] **Step 2: Implement the runner**

Use:

```text
REMOTE_HOST=sitian@10.232.195.203
CONTROL_SOCKET=/tmp/ssh-sitian-10.232.195.203
REMOTE_PYTHON=/data00/home/sitian/sitian-workspace01/tllm/env/bin/python
MODEL_PATH=/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0.6B
GPU_ID=0
```

- [x] **Step 3: Run local tests**

```bash
python3 -m pytest tools/test_speculative_residency_boundary_gate.py -q
```

- [x] **Step 4: Run the remote gate**

Set the tag in a prior shell statement so `LOCAL_OUT` does not expand an
unset same-line variable:

```bash
RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
LOCAL_OUT="artifacts/speculative_residency_boundary/${RUN_TAG}"
RUN_TAG="${RUN_TAG}" LOCAL_OUT="${LOCAL_OUT}" \
  bash tools/run_speculative_residency_boundary_gate_remote.sh
```

Required artifact evidence:

```text
baseline == accepted-boundary == rejected-boundary output IDs
accepted committed_blocks > 0
rejected rejected_blocks > 0
rejected rejected_d2h_copies == 0
accepted and rejected h2d_copies > 0
accepted and rejected h2d_bytes > 0
remote and local source-hash verifier PASS
```

---

### Task 5: Full Regression and Authoritative Evidence

**Files:**
- Modify: `docs/superpowers/audits/2026-08-12-generic-inference-optimization-goal-audit.md`
- Modify: `AGENT_HANDOFF_STATE.md`

- [x] **Step 1: Run focused and broad regressions**

```bash
python3 -m pytest \
  tools/test_speculative_residency.py \
  tools/test_model_runner_spec_verify.py \
  tools/test_engine_speculative_execution.py \
  tools/test_engine_speculative_runtime.py \
  tools/test_speculative_residency_boundary_gate.py \
  -q
```

Run the remote direct `tools/test_kv_offload.py` suite after synchronizing the
current test file.

- [x] **Step 2: Run static checks**

```bash
python3 -m py_compile \
  tinyvllm/engine/model_runner.py \
  tools/speculative_residency_boundary_gate.py \
  tools/verify_speculative_residency_boundary_gate.py
git diff --check
git diff --cached --quiet
```

- [x] **Step 3: Update audit and handoff**

Record exact commands, counts, artifact path, all movement/residency counters,
elapsed observations without interpretation, and the remaining
`NOT_PROMOTABLE` dimensions.

- [x] **Step 4: Final evidence check**

Re-run the independent local verifier from the final tagged artifact and
confirm no staged changes.

