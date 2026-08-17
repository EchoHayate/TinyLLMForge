# Autoregressive Draft Exact-Shape CUDA Graph Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a default-off exact Q4/B4 TP4 CUDA Graph path for independent Qwen3 draft proposal generation while preserving exact-greedy and transactional Proposal-KV semantics.

**Architecture:** A focused graph policy runner admits immutable exact identities only after successful eager observations. A Qwen3 CUDA backend captures three model-forward/argmax/TP-broadcast steps using private scratch Proposal-KV state, while the existing executor remains authoritative for lifecycle registration and accepted-prefix finalization.

**Tech Stack:** Python, PyTorch CUDA Graphs, torch.distributed/NCCL, pytest, TinyLLMForge ProposalKVCache and speculative runtime.

## Global Constraints

- Work only in `/Users/bytedance/Desktop/TinyLLMForge`.
- First slice is exactly TP4, batch size 4, Q4, greedy, dense direct Proposal-KV, and KV offload disabled.
- Exact identities must not use padding or rounding.
- Capture starts only after a successful eager observation.
- Pre-replay failure may fall back eagerly; replay-started failure must quarantine and fail closed without eager retry.
- Preserve exact greedy tokens, accepted-prefix commit, rejected-suffix abort, transaction ownership, and TP failure convergence.
- Keep selected tokens on GPU across all three proposal steps and perform one final host readback.
- Do not claim speedup until source-bound controlled before/after evidence exists.

---

### Task 1: Exact graph policy and state machine

**Files:**
- Create: `tinyvllm/engine/autoregressive_draft_graph.py`
- Create: `tools/test_autoregressive_draft_graph.py`

**Interfaces:**
- Produces: `AutoregressiveDraftGraphIdentity`, `AutoregressiveDraftGraphEntry`, `AutoregressiveDraftGraphPreReplayError`, `AutoregressiveDraftGraphReplayError`, and `AutoregressiveDraftExactGraphRunner.run(exact_q, rows, eager)`.
- Consumes: backend methods `estimate_static_bytes`, `capture`, and `replay`; scratch-owner methods `acquire` and `rollback`.

- [ ] **Step 1: Write failing identity and state-machine tests**

Add tests that vary every identity field, prove failed eager calls do not
advance observations, prove successful eager calls precede capture, and prove
the third exact call replays.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
python3 -m pytest -q tools/test_autoregressive_draft_graph.py
```

Expected: collection fails because `tinyvllm.engine.autoregressive_draft_graph`
does not exist.

- [ ] **Step 3: Implement the minimal policy runner**

Implement immutable JSON/SHA identity, budget validation, observation counts,
private scratch capture, permanent quarantine, pre-replay eager fallback, and
post-replay `AutoregressiveDraftGraphReplayError`.

- [ ] **Step 4: Verify GREEN**

Run the same pytest command. Expected: all policy tests pass.

### Task 2: Configuration contract

**Files:**
- Modify: `tinyvllm/config.py`
- Create: `tools/test_autoregressive_draft_cuda_graph_config.py`

**Interfaces:**
- Produces: the nine `autoregressive_draft_cuda_graph_*` configuration fields
  specified by the design.
- Consumes: existing autoregressive draft enablement, topology, and offload
  fields.

- [ ] **Step 1: Write failing default, canonicalization, and rejection tests**

Cover default-off values, sorted unique Q/B allowlists, nonpositive budgets,
feature-without-draft, feature-with-offload, and runtime TP4 enforcement.

- [ ] **Step 2: Run and verify RED**

```bash
python3 -m pytest -q tools/test_autoregressive_draft_cuda_graph_config.py
```

Expected: missing configuration attributes.

- [ ] **Step 3: Add minimal validated fields**

Use the existing canonical positive-integer tuple helper. Reject enabled graph
mode when learned draft is disabled or Proposal-KV offload is enabled.

- [ ] **Step 4: Verify GREEN and existing config compatibility**

```bash
python3 -m pytest -q \
  tools/test_autoregressive_draft_cuda_graph_config.py \
  tools/test_qwen35_config_compatibility.py
```

### Task 3: Executor graph integration and shared registration

**Files:**
- Modify: `tinyvllm/engine/autoregressive_draft_executor.py`
- Modify: `tools/test_autoregressive_draft_executor.py`

**Interfaces:**
- Consumes: optional `graph_runner`.
- Produces: one shared registration helper for eager and graph proposals and a
  tensor-free graph summary in `authority_snapshot()`.

- [ ] **Step 1: Write failing fake-runner tests**

Add tests proving Q4/B4 dispatches to the runner, Q1 stays eager, unsupported
families are passed to the runner without padding, fake graph proposals use
the same lifecycle registration, and finalize commit/abort outcomes match
eager.

- [ ] **Step 2: Run the focused tests and verify RED**

```bash
python3 -m pytest -q tools/test_autoregressive_draft_executor.py \
  -k 'graph or exact_q'
```

Expected: constructor rejects `graph_runner` or graph dispatch assertions fail.

- [ ] **Step 3: Refactor registration only after RED**

Split current `_run_exact_q_group()` into:

```text
_run_exact_q_group_eager()
_register_exact_q_group()
_run_exact_q_group()
```

The wrapper dispatches through `graph_runner.run(...)`; both result paths use
the same logical-authority assertion and lifecycle registration.

- [ ] **Step 4: Verify focused and full executor GREEN**

```bash
python3 -m pytest -q tools/test_autoregressive_draft_executor.py
```

### Task 4: Private scratch Proposal-KV owner

**Files:**
- Create: `tinyvllm/engine/qwen3_draft_graph_scratch.py`
- Create: `tools/test_qwen3_draft_graph_scratch.py`

**Interfaces:**
- Produces: `Qwen3DraftGraphScratchOwner.acquire(identity, rows)` and
  `.rollback(lease)`.
- Consumes: live `ProposalKVCache`, shared `Qwen3DraftProposalKVStorage`, and
  graph capture row records.

- [ ] **Step 1: Write failing ownership and rollback tests**

Prove scratch transactions use a private namespace, preserve live committed
state, allocate exactly three staged entries per row, never enter executor
maps, and release every scratch sequence on success and injected failure.

- [ ] **Step 2: Verify RED**

```bash
python3 -m pytest -q tools/test_qwen3_draft_graph_scratch.py
```

- [ ] **Step 3: Implement the owner**

Reuse the production physical store but construct a distinct direct allocator
and ProposalKVCache. Copy only logical committed ownership needed to form
capture rows; never share transaction dictionaries.

- [ ] **Step 4: Verify GREEN**

Run the focused test and `tools/test_qwen3_draft_proposal_kv_storage.py`.

### Task 5: Qwen3 CUDA Graph backend

**Files:**
- Create: `tinyvllm/engine/qwen3_draft_cuda_graph_backend.py`
- Create: `tools/test_qwen3_draft_cuda_graph_backend.py`
- Modify: `tinyvllm/engine/qwen3_draft_backend.py`

**Interfaces:**
- Produces: static tensor allocation, exact preflight, capture, replay, one
  final token readback, and live unregistered DraftProposal objects.
- Consumes: loaded Qwen3 model, backend identity fields, direct
  ProposalKVCache, TP rank/size, and `torch.distributed.broadcast`.

- [ ] **Step 1: Write fake-torch failing tests**

Cover exact tensor shapes/dtypes, three forward calls, three root
argmax/broadcast steps, GPU-resident token chaining, one final `.tolist()`,
live transaction abort on failure, and pre-replay validation before
`graph.replay()`.

- [ ] **Step 2: Verify RED**

```bash
python3 -m pytest -q tools/test_qwen3_draft_cuda_graph_backend.py
```

- [ ] **Step 3: Add graph-safe backend primitives**

Expose a Qwen3 backend method that executes one decode step from supplied
static tensors without constructing tensors or extracting CPU scalars. Keep
the existing `decode_step_batch()` behavior unchanged for eager callers.

- [ ] **Step 4: Implement capture and replay**

Allocate fixed Q4/B4 buffers, prepare scratch/live metadata before graph
entry, capture three decode/argmax/broadcast steps, read tokens once after
replay, mark three entries materialized, and return unregistered proposals.

- [ ] **Step 5: Verify focused backend GREEN**

```bash
python3 -m pytest -q \
  tools/test_qwen3_draft_cuda_graph_backend.py \
  tools/test_qwen3_draft_backend.py
```

### Task 6: TP convergence around replay

**Files:**
- Modify: `tinyvllm/engine/autoregressive_draft_executor.py`
- Modify: `tinyvllm/engine/autoregressive_draft_graph.py`
- Modify: `tools/test_autoregressive_draft_executor.py`
- Modify: `tools/test_autoregressive_draft_graph.py`

**Interfaces:**
- Produces: one `graph_pre_replay` convergence before entry and one
  `graph_replay_complete` convergence after final token readback.

- [ ] **Step 1: Write failure-injection tests**

Inject one-rank preflight failure and assert all ranks take one eager fallback.
Inject post-entry failure and assert quarantine, transaction abort, and no
eager retry.

- [ ] **Step 2: Verify RED**

Run focused `-k graph` tests and confirm missing convergence stages.

- [ ] **Step 3: Implement convergence callbacks**

Pass executor callbacks into the runner/backend. Include exact Q, ordered
sequence IDs, transaction IDs, and final token rows in the authority records.

- [ ] **Step 4: Verify GREEN**

Run the full executor, TP, and graph policy test files.

### Task 7: Production registration

**Files:**
- Modify: `tinyvllm/engine/autoregressive_draft_registration.py`
- Modify: `tinyvllm/engine/model_runner.py`
- Modify: `tools/test_autoregressive_draft_registration.py`
- Modify: `tools/test_autoregressive_draft_model_runner_integration.py`

**Interfaces:**
- Consumes: validated graph configuration and the constructed Qwen3 backend.
- Produces: graph backend, scratch owner, policy runner, and executor wiring.

- [ ] **Step 1: Write failing default-off and enabled-wiring tests**

Prove default-off creates no graph owner, enabled TP4/dense mode wires all
three objects, and unsupported topology/offload fails before publication.

- [ ] **Step 2: Verify RED**

Run both registration test files.

- [ ] **Step 3: Wire production construction**

Construct graph components only after model, cache, and backend construction.
Publish nothing if any rank fails registration consensus.

- [ ] **Step 4: Verify GREEN**

Run registration, model-runner integration, TP4 local gate, and authority
snapshot tests.

### Task 8: Source-bound correctness and performance gate

**Files:**
- Create: `tools/autoregressive_draft_cuda_graph_contract.py`
- Create: `tools/run_autoregressive_draft_cuda_graph_gate_remote.py`
- Create: `tools/verify_autoregressive_draft_cuda_graph_gate.py`
- Create: `tools/test_autoregressive_draft_cuda_graph_gate.py`

**Interfaces:**
- Produces: paired eager/graph raw rows, correctness summary, graph counters,
  transaction digests, controlled performance summary, archived verifier
  receipt, and manifest.

- [ ] **Step 1: Write verifier tamper tests**

Reject changed proposal tokens, accepted counts, transaction digests, graph
counters, source hashes, pair order, memory rows, or performance aggregates.

- [ ] **Step 2: Verify RED then implement the contract**

The verifier must recompute every aggregate from raw rows and require graph
replay on all four ranks.

- [ ] **Step 3: Check remote GPU state without changing other processes**

Run SSH/Kerberos reachability and `nvidia-smi`. Do not kill or reassign any
unrelated process.

- [ ] **Step 4: Run local source-bound preflight**

Archive only tracked source and the focused untracked implementation files.
Record the exact local commit, patch SHA, Python, PyTorch, CUDA, NCCL, model
fingerprints, and GPU UUIDs.

- [ ] **Step 5: Run correctness gate**

Require eager/graph equality for target tokens, proposal tokens, accepted
prefixes, transaction digests, and final active transaction count.

- [ ] **Step 6: Run controlled paired performance gate**

Use two warmups and at least eight measured position-balanced pairs for
TP4/B4/Q4, prompt length 256, output length 16.

- [ ] **Step 7: Verify source-bound artifacts twice**

Run the archived verifier in the remote bundle and the current local verifier
after download, then run:

```bash
sha256sum -c manifest.sha256
```

### Task 9: Completion audit, handoff, and version control

**Files:**
- Modify: `docs/superpowers/audits/2026-08-16-phase1-completion-audit.md`
- Modify: `AGENT_HANDOFF_STATE.md`
- Create: `docs/superpowers/audits/2026-08-17-autoregressive-draft-cuda-graph-completion-audit.md`

**Interfaces:**
- Produces: prompt-to-artifact checklist and an honest `GO`,
  `NO_GO_PERFORMANCE`, `NO_GO_CORRECTNESS`, or
  `INCONCLUSIVE_ENVIRONMENT` classification.

- [ ] **Step 1: Run fresh local verification**

Run all focused tests, related regression suites, compileall, staged diff
checks, and source-bound artifact verification.

- [ ] **Step 2: Build the prompt-to-artifact checklist**

Map every design requirement to a file, test, command output, or artifact.
Mark uncertainty and missing remote evidence as not achieved.

- [ ] **Step 3: Update handoff and Phase-1 audit**

Record what passed, what it proves, what it does not prove, performance
classification, exact artifact paths, and next action.

- [ ] **Step 4: Stage only versionable focused files**

Do not stage the existing deep untracked experiment tree. Inspect the exact
path list before commit.

- [ ] **Step 5: Commit with the required single trailer and push**

Use hooks disabled to prevent duplicate attribution:

```bash
git -c core.hooksPath=/dev/null commit \
  -m "feat(runtime): graph learned draft proposal families" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push -u origin feat/kv-sparse-attention
```

- [ ] **Step 6: Verify local and remote SHA and trailer**

Confirm the remote branch resolves to the local commit and the required
trailer appears exactly once as the final non-empty commit-message line.
