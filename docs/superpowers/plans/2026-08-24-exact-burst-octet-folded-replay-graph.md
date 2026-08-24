# Exact-Burst Octet-Folded Replay Graph Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking. This repository's active constraint
> forbids subagents and additional worktrees, so execute inline in the
> authoritative checkout with `executing-plans`.

**Goal:** Capture eight ordered exact-greedy complete-token steps in one CUDA
Graph so K8 uses one physical graph launch and K16 uses two, while preserving
the accepted token, KV, scheduler, D2H, rollback, and fallback contracts.

**Architecture:** Add a default-disabled folded-graph capability beside the
existing one-token graph. The folded graph unrolls the unchanged complete-step
body eight times during capture, reports logical token work separately from
physical graph launches, and falls back to the one-token graph before replay
when ineligible or unhealthy. A source-bound ceiling probe runs before any
terminal benchmark work.

**Tech Stack:** Python 3, dataclasses, PyTorch CUDA Graphs, pytest,
TinyLLMForge model runner and exact-burst runtime, JSON/JSONL evidence,
SHA256 manifests, SSH remote controller, Qwen3-0.6B on one strict-clean A100.

## Global Constraints

- The only authoritative checkout is
  `/Users/bytedance/Desktop/TinyLLMForge`, resolving to
  `/Users/bytedance/dev/TinyLLMForge`.
- Stay on `feat/kv-sparse-attention`; do not create a worktree or use a
  subagent.
- Do not begin runtime implementation until the r10 journal gate,
  generation-sealed identity gate, and elastic K8/K16 ceiling are reconciled.
- Preserve every unrelated dirty or untracked file. Stage only exact task
  paths.
- Use `git -c core.hooksPath=/dev/null commit`.
- Every commit has exactly one
  `Co-authored-by: TRAE CLI <noreply@bytedance.com>` trailer.
- Push only to `origin/feat/kv-sparse-attention`.
- Use `python3 -m pytest`; `python` is unavailable locally.
- Keep the feature default-disabled.
- Preserve exact tokens, text, logits, argmax, target forwards, KV slots,
  one final token D2H per burst, scheduler ownership, and rollback behavior.
- Keep the existing one-token graph as the complete fallback.
- Never retry after a folded launch begins in the same engine step.
- Hardware data may be written only below
  `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818`.
- Do not run `kinit`, kill external processes, or reuse an attempted run tag.
- Strict-clean GPU admission is memory `<=1024 MiB`, utilization `<=5%`, and
  no compute process.
- A GPU gate source SHA must equal the already-pushed branch HEAD at launch.
- Partial rows never authorize a performance classification.
- Report benefit and cost together.

---

## File Map

- Modify `tinyvllm/config.py`: add the default-off folded-graph flag and
  dependency validation.
- Modify `tinyvllm/engine/exact_greedy_decode_burst.py`: add folded capture,
  receipts, health, counters, validation, and replay.
- Modify `tinyvllm/engine/model_runner.py`: capture and route the production
  and correctness folded graphs.
- Modify `tinyvllm/engine/llm_engine.py`: expose capability and preserve
  fail-closed fallback.
- Modify `tools/test_exact_greedy_decode_burst.py`: pure capture/replay,
  accounting, and failure tests.
- Modify `tools/test_model_runner_spec_verify.py`: config, capability,
  routing, and serialization tests.
- Modify `tools/test_scheduler_prepared_postprocess.py`: unchanged scheduler
  and rollback authority tests.
- Create `tools/profile_exact_burst_octet_folded_graph.py`: common paired
  workload producer.
- Create `tools/test_profile_exact_burst_octet_folded_graph.py`.
- Create `tools/exact_burst_octet_folded_graph_ceiling.py`: ceiling
  classifier.
- Create `tools/test_exact_burst_octet_folded_graph_ceiling.py`.
- Create `tools/exact_burst_octet_folded_graph_gate.py`: terminal producer
  and gate.
- Create `tools/test_exact_burst_octet_folded_graph_gate.py`.
- Create `tools/exact_burst_octet_folded_graph_verify.py`: independent
  verifier.
- Create `tools/test_exact_burst_octet_folded_graph_verify.py`.
- Create `tools/run_exact_burst_octet_folded_graph_remote.py`: mounted-only
  remote controller.
- Create `tools/test_run_exact_burst_octet_folded_graph_remote.py`.
- Create
  `docs/superpowers/audits/2026-08-24-exact-burst-octet-folded-replay-graph-audit.md`.
- Modify `docs/superpowers/audits/2026-08-16-phase1-completion-audit.md`.
- Modify `AGENT_HANDOFF_STATE.md`.

### Task 1: Freeze the Configuration and Accounting Contract

**Files:**

- Modify: `tinyvllm/config.py`
- Modify: `tinyvllm/engine/exact_greedy_decode_burst.py`
- Modify: `tools/test_exact_greedy_decode_burst.py`
- Modify: `tools/test_model_runner_spec_verify.py`

**Interfaces:**

- Produces:
  `Config.exact_greedy_decode_burst_octet_folded_graph: bool = False`
- Produces:
  `ExactGreedyDecodeBurstFoldedHealth`
- Extends `ExactGreedyDecodeBurstStats.summary()` with physical launch and
  folded-health fields.

- [ ] **Step 1: Write RED configuration tests**

Require the default to be false and reject:

```text
non-bool folded flag
folded graph without exact burst
folded graph with base width other than eight
folded graph with split phase
folded graph with ragged coalescing
```

Allow elastic K16 either false or true.

- [ ] **Step 2: Write RED folded-health tests**

Require a host-side health object with:

```python
health = ExactGreedyDecodeBurstFoldedHealth()
assert health.generation == 0
assert health.quarantine_reason is None

health.record_attempt(width=8)
health.record_launch(logical_steps=8)
health.quarantine("replay_failure:RuntimeError")

assert health.generation == 1
assert health.quarantine_reason == "replay_failure:RuntimeError"
```

Reject boolean integers, unsupported widths, non-positive logical steps,
empty reasons, repeated quarantine with a different reason, and generation
overflow before mutation.

- [ ] **Step 3: Write RED summary tests**

Require exact fields:

```text
one_token_cuda_graph_launches
folded_cuda_graph_launches
folded_logical_steps
folded_k8_bursts
folded_k16_bursts
folded_fallback_counts
folded_health_generation
folded_quarantine_reason
```

Keep `graph_replays` as logical token-step accounting.

- [ ] **Step 4: Run RED**

```bash
python3 -m pytest \
  tools/test_exact_greedy_decode_burst.py \
  tools/test_model_runner_spec_verify.py \
  -q
```

Expected: only new folded configuration, health, and summary tests fail.

- [ ] **Step 5: Implement the minimum contract**

Add the config field and validation. Add a dataclass whose mutators validate
before changing state:

```python
@dataclass
class ExactGreedyDecodeBurstFoldedHealth:
    generation: int = 0
    quarantine_reason: Optional[str] = None
    attempts: int = 0
    launches: int = 0
    logical_steps: int = 0

    @property
    def healthy(self) -> bool:
        return self.quarantine_reason is None
```

Add stats recorders that increment physical launches separately from logical
replays.

- [ ] **Step 6: Run GREEN and adjacent tests**

```bash
python3 -m pytest \
  tools/test_exact_greedy_decode_burst.py \
  tools/test_model_runner_spec_verify.py \
  tools/test_scheduler_prepared_postprocess.py \
  -q
git diff --check -- \
  tinyvllm/config.py \
  tinyvllm/engine/exact_greedy_decode_burst.py \
  tools/test_exact_greedy_decode_burst.py \
  tools/test_model_runner_spec_verify.py
```

- [ ] **Step 7: Commit and push**

Stage only the four task files. Commit:

```text
feat(runtime): add folded exact-burst contract

Co-authored-by: TRAE CLI <noreply@bytedance.com>
```

Push to `origin/feat/kv-sparse-attention`.

### Task 2: Capture Eight Complete Steps in One Graph

**Files:**

- Modify: `tinyvllm/engine/exact_greedy_decode_burst.py`
- Modify: `tools/test_exact_greedy_decode_burst.py`

**Interfaces:**

- Consumes: existing `_run_complete_step(...)`
- Produces:
  `ExactGreedyDecodeBurstFoldedGraph.capture(..., steps_per_launch=8)`
- Produces a capture receipt containing `steps_per_launch` and
  `retained_output_tensor_count`.

- [ ] **Step 1: Write RED capture-count tests**

Use fake graph/capture factories and a counted complete-step body. Require:

```text
warmup complete-step calls == 8
captured complete-step calls == 8
steps_per_launch == 8
retained output groups == 8
capture receipt identity includes steps_per_launch
```

Changing `steps_per_launch` in a copied fixture must change graph identity.

- [ ] **Step 2: Write RED state-progression tests**

With fake mutable tensors, verify one folded capture body advances:

```text
input token feedback: eight ordered updates
position: +8
context length: +8
slot mapping: +8
history index: +8
token history: eight ordered positions
```

Require scratch capacity for at least eight writable slots and history
capacity for the configured maximum lease.

- [ ] **Step 3: Run RED**

```bash
python3 -m pytest \
  tools/test_exact_greedy_decode_burst.py \
  -k 'folded and capture' \
  -q
```

Expected: failures because the folded graph and receipt fields do not exist.

- [ ] **Step 4: Implement folded capture**

Keep `_run_complete_step(...)` unchanged. In the capture context:

```python
retained_groups = []
for _ in range(steps_per_launch):
    retained_groups.append(
        cls._run_complete_step(
            tensors=tensors,
            model=model,
            compute_logits=compute_logits,
            float32_dtype=float32_dtype,
            correctness_trace=correctness_trace,
        )
    )
retained_outputs = tuple(
    tensor
    for group in retained_groups
    for tensor in group
)
```

Use the same eight-step loop for warmup. Reset static state in `finally`.
Include the fold width in identity and receipt validation.

- [ ] **Step 5: Run GREEN and legacy capture tests**

```bash
python3 -m pytest \
  tools/test_exact_greedy_decode_burst.py \
  -q
```

Expected: all tests pass; existing one-token capture expectations are
unchanged.

- [ ] **Step 6: Commit and push**

Commit exact task files as:

```text
feat(runtime): capture folded exact-burst graph

Co-authored-by: TRAE CLI <noreply@bytedance.com>
```

### Task 3: Route K8 and K16 Through the Folded Capability

**Files:**

- Modify: `tinyvllm/engine/exact_greedy_decode_burst.py`
- Modify: `tinyvllm/engine/model_runner.py`
- Modify: `tinyvllm/engine/llm_engine.py`
- Modify: `tools/test_exact_greedy_decode_burst.py`
- Modify: `tools/test_model_runner_spec_verify.py`
- Modify: `tools/test_scheduler_prepared_postprocess.py`

**Interfaces:**

- Produces:
  `ExactGreedyDecodeBurstFoldedGraph.replay(...)`
- Produces model-runner capability fields for production and correctness
  folded graphs.
- Preserves the existing lease and result schemas.

- [ ] **Step 1: Write RED routing tests**

Table-drive:

```text
authorized=8, folded healthy  -> 1 folded launch, 0 one-token launches
authorized=16, folded healthy -> 2 folded launches, 0 one-token launches
authorized=7                  -> 0 folded launches, 7 one-token launches
authorized=15                 -> 0 folded launches, 15 one-token launches
folded unavailable            -> one-token launches
folded quarantined            -> one-token launches
identity mismatch             -> one-token launches before replay
```

- [ ] **Step 2: Write RED K16 continuation tests**

Require the second folded launch to observe state produced by the first:

```text
history cursor before launch 1 == 0
history cursor before launch 2 == 8
history cursor after launch 2 == 16
no static reset between launches
one final token D2H
sixteen logical forwards/replays
two physical folded launches
```

- [ ] **Step 3: Write RED failure tests**

Require:

- pre-launch validation failure falls back to one-token graph;
- first-launch exception commits no host tokens and quarantines folded;
- second-launch exception commits no host tokens and quarantines folded;
- D2H exception commits no host tokens and quarantines folded;
- one-token graph health remains unchanged;
- no same-step retry occurs after any folded launch begins.

- [ ] **Step 4: Run RED**

```bash
python3 -m pytest \
  tools/test_exact_greedy_decode_burst.py \
  tools/test_model_runner_spec_verify.py \
  tools/test_scheduler_prepared_postprocess.py \
  -k 'folded or exact_greedy_decode_burst' \
  -q
```

- [ ] **Step 5: Implement replay and routing**

Validate capability before replay. For a divisible width:

```python
launch_count = lease.authorized_token_count // 8
for _ in range(launch_count):
    folded_graph.graph.replay()
    completed_launches += 1
```

After all launches, read the existing token-history slice once and construct
the existing `ExactGreedyDecodeBurstResult`. Record eight logical replay
steps per physical folded launch.

If folded validation fails before the first launch, dispatch unchanged to the
one-token graph. If any launch has begun, propagate the error after
quarantining folded only.

- [ ] **Step 6: Capture production and correctness capabilities**

Create folded graphs only when the flag is enabled. Bind them to the same
model, static input shapes, block capacity, and source generation as their
one-token counterparts. Use a shared graph pool only when capture/replay
ordering is statically non-concurrent.

- [ ] **Step 7: Run GREEN and adjacent suites**

```bash
python3 -m pytest \
  tools/test_exact_greedy_decode_burst.py \
  tools/test_model_runner_spec_verify.py \
  tools/test_scheduler_prepared_postprocess.py \
  tools/test_context_gated_elastic_exact_burst_ceiling.py \
  -q
git diff --check -- \
  tinyvllm/config.py \
  tinyvllm/engine/exact_greedy_decode_burst.py \
  tinyvllm/engine/model_runner.py \
  tinyvllm/engine/llm_engine.py
```

- [ ] **Step 8: Commit and push**

Commit exact task paths as:

```text
feat(runtime): route exact bursts through folded graph

Co-authored-by: TRAE CLI <noreply@bytedance.com>
```

### Task 4: Build the Source-Bound Ceiling Probe

**Files:**

- Create: `tools/profile_exact_burst_octet_folded_graph.py`
- Create: `tools/test_profile_exact_burst_octet_folded_graph.py`
- Create: `tools/exact_burst_octet_folded_graph_ceiling.py`
- Create: `tools/test_exact_burst_octet_folded_graph_ceiling.py`

**Interfaces:**

- Produces performance and correctness rows for `one_token_graph` and
  `octet_folded_graph`.
- Produces `ceiling.json` with `GO_CEILING` or `NO_GO_CEILING`.

- [ ] **Step 1: Write RED schema and inventory tests**

Freeze:

```text
contexts: 256, 2048, 8192
K8 repetitions: 5 per arm and context
optional K16 contexts after elastic promotion: 256, 2048
correctness points: prefill-final, decode-first, decode-middle, decode-final
```

Every row carries source SHA, patch SHA256, graph identities, logical
forwards/replays, physical launches, D2H calls/bytes, capture cost, peak CUDA
memory, TPOT, TTFT, E2E, throughput, and host-visible gap.

- [ ] **Step 2: Write RED classifier tests**

Require `NO_GO_CEILING` for:

- missing or duplicate rows;
- unmatched workload identity or execution order;
- token/text/logit/argmax mismatch;
- logical-forward or logical-replay drift;
- insufficient physical-launch reduction;
- median TPOT improvement below 1.0%;
- P95 TPOT improvement below 0.5%;
- protected latency/throughput regression above 2%;
- allocated/reserved capture cost above 1% of baseline peak;
- retained static delta above 128 MiB;
- folded capture duration above 120 seconds;
- NaN, infinity, malformed digest, or source mismatch; and
- any unexpected fallback, rollback, or quarantine.

- [ ] **Step 3: Implement producer and classifier**

Reuse the accepted exact-burst workload harness. Change only the folded flag
between paired arms. Keep the one-token fallback enabled in both arms and
record whether it was actually used.

- [ ] **Step 4: Run local GREEN**

```bash
python3 -m pytest \
  tools/test_profile_exact_burst_octet_folded_graph.py \
  tools/test_exact_burst_octet_folded_graph_ceiling.py \
  -q
```

- [ ] **Step 5: Commit and push**

Commit exact task files as:

```text
test(runtime): add folded graph ceiling probe

Co-authored-by: TRAE CLI <noreply@bytedance.com>
```

### Task 5: Add Independent Verification and Safe Remote Execution

**Files:**

- Create: `tools/exact_burst_octet_folded_graph_verify.py`
- Create: `tools/test_exact_burst_octet_folded_graph_verify.py`
- Create: `tools/run_exact_burst_octet_folded_graph_remote.py`
- Create: `tools/test_run_exact_burst_octet_folded_graph_remote.py`

**Interfaces:**

- Produces a verifier receipt independently reconstructed from rows.
- Produces controller manifests and mounted-only launch/download receipts.

- [ ] **Step 1: Write RED verifier tests**

Require independent reconstruction of every ceiling metric and reject:

```text
tampered rows
missing rows
duplicate rows
source drift
threshold drift
NaN or infinity
producer/verifier disagreement
logical versus physical counter disagreement
```

- [ ] **Step 2: Write RED controller safety tests**

Require:

- approved `/data00/home/sitian/.../command-timeline-20260818` root only;
- strict-clean GPU admission;
- Kerberos TTL fail-fast without `kinit`;
- pushed HEAD equality;
- fresh immutable run tag;
- runtime caches and logs below the approved mounted root;
- no `/`, `/tmp`, or `/private/tmp` runtime writes;
- resumable worker receipts; and
- partial-preserving downloads.

- [ ] **Step 3: Implement verifier and controller**

Follow the accepted r10 controller structure. Keep worker, controller
verification, and local verification as separate receipts.

- [ ] **Step 4: Run GREEN**

```bash
python3 -m pytest \
  tools/test_exact_burst_octet_folded_graph_verify.py \
  tools/test_run_exact_burst_octet_folded_graph_remote.py \
  -q
```

- [ ] **Step 5: Commit and push**

Commit exact task files as:

```text
test(runtime): verify folded graph artifacts

Co-authored-by: TRAE CLI <noreply@bytedance.com>
```

### Task 6: Run the GPU Ceiling and Enforce the Stop Rule

**Files:**

- Create under local artifacts:
  `artifacts/exact_burst_octet_folded_graph/<fresh-tag>/`
- Write remotely only below:
  `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/exact-burst-octet-folded-graph/`

- [ ] **Step 1: Verify launch preconditions**

Verify pushed HEAD equality, Kerberos lifetime, remote requirements, fresh
destinations, and one strict-clean A100. Do not launch if any precondition
fails.

- [ ] **Step 2: Launch one fresh-tag ceiling worker**

Use the controller only. Record PID, PGID, GPU UUID/index, source SHA, patch
hash, distribution port, and all remote paths.

- [ ] **Step 3: Complete both verifiers**

Require complete row inventory and byte-identical producer, remote verifier,
and frozen-source local verifier classifications.

- [ ] **Step 4: Apply the stop rule**

If classification is `NO_GO_CEILING`, stop this optimization and write the
audit with measured benefit and cost. Do not create terminal-gate code.

If classification is `GO_CEILING`, continue to Task 7.

### Task 7: Build and Run the Terminal Gate Only After GO

**Files:**

- Create: `tools/exact_burst_octet_folded_graph_gate.py`
- Create: `tools/test_exact_burst_octet_folded_graph_gate.py`
- Modify: `tools/exact_burst_octet_folded_graph_verify.py`
- Modify: `tools/test_exact_burst_octet_folded_graph_verify.py`
- Modify: `tools/run_exact_burst_octet_folded_graph_remote.py`
- Modify: `tools/test_run_exact_burst_octet_folded_graph_remote.py`

- [ ] **Step 1: Freeze terminal inventory**

Require 40 complete paired performance rows and 32 correctness rows across
eligible K8 and, if promoted, K16 contexts.

- [ ] **Step 2: Write RED terminal-classifier tests**

Require:

```text
aggregate median TPOT improvement >= 1.5%
aggregate P95 TPOT improvement >= 1.0%
per-context median/P95 regression <= 2%
TTFT/E2E/TPOT-P99 regression <= 2%
throughput regression <= 2%
peak allocated/reserved memory regression <= 1%
exact parity
logical target forwards unchanged
K8 physical launches reduced >= 85%
K16 physical launches reduced >= 80%
unexpected fallback/rollback/quarantine == 0
```

- [ ] **Step 3: Implement and run local GREEN**

```bash
python3 -m pytest \
  tools/test_exact_burst_octet_folded_graph_gate.py \
  tools/test_exact_burst_octet_folded_graph_verify.py \
  tools/test_run_exact_burst_octet_folded_graph_remote.py \
  -q
```

- [ ] **Step 4: Commit and push gate source**

Commit exact task paths, push, and verify remote HEAD before launch.

- [ ] **Step 5: Launch one fresh-tag terminal worker**

Use one strict-clean A100 and the mounted-only controller. Never reuse the
ceiling tag or any interrupted tag.

- [ ] **Step 6: Verify terminal evidence**

Require all rows, source manifests, runner receipt, gate result, remote
verification, controller completion, downloaded manifest, and frozen-source
local verification.

### Task 8: Reconcile Audits and Handoff

**Files:**

- Create:
  `docs/superpowers/audits/2026-08-24-exact-burst-octet-folded-replay-graph-audit.md`
- Modify:
  `docs/superpowers/audits/2026-08-16-phase1-completion-audit.md`
- Modify: `AGENT_HANDOFF_STATE.md`

- [ ] **Step 1: Write prompt-to-artifact reconciliation**

Map each design requirement to implementation, tests, GPU rows, verifier
receipts, and final status.

- [ ] **Step 2: Report benefit and cost**

Include:

- TPOT median/P95/P99 and throughput;
- TTFT and E2E;
- logical forwards/replays and physical launch counts;
- D2H calls/bytes;
- host-visible gap;
- capture duration;
- allocated/reserved and retained-static memory;
- fallback, rollback, and quarantine counts; and
- the exact claim boundary.

- [ ] **Step 3: Run final verification**

Run focused folded suites, adjacent exact-burst suites, source-manifest
verification, and:

```bash
git diff --check -- \
  docs/superpowers/audits/2026-08-24-exact-burst-octet-folded-replay-graph-audit.md \
  docs/superpowers/audits/2026-08-16-phase1-completion-audit.md \
  AGENT_HANDOFF_STATE.md
```

- [ ] **Step 4: Commit and push**

Stage only the folded audit, Phase 1 audit, and handoff. Commit:

```text
docs(runtime): reconcile folded graph gate

Co-authored-by: TRAE CLI <noreply@bytedance.com>
```

Push to `origin/feat/kv-sparse-attention` and verify local HEAD equals remote.
