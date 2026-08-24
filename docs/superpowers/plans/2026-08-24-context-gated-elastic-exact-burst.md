# Context-Gated Elastic Exact-Burst Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking. This repository's active constraint
> forbids subagents and additional worktrees, so execute inline in the
> authoritative checkout with `executing-plans`.

**Goal:** Add a default-disabled deterministic policy that uses a separately
captured K16 exact-greedy graph for contexts up to 2,048 tokens and otherwise
preserves the current K8 path, then admit the feature only if a source-bound
GPU ceiling probe and terminal paired gate show benefit without violating the
40 ms visibility budget.

**Architecture:** Keep K8 as the baseline and complete fallback. Add a pure
K16 eligibility decision, separate K8/K16 graph identities and quarantine
state, and width-aware scheduler/journal accounting. First implement only the
minimum runtime surface needed for an exact K8/K16 ceiling probe; proceed to
the full paired gate only when the frozen probe thresholds pass.

**Tech Stack:** Python 3, dataclasses, PyTorch CUDA Graphs, pytest,
TinyLLMForge scheduler/model runner, JSON/JSONL evidence, SHA256 manifests,
SSH remote controller, Qwen3-0.6B TP1 on one strict-clean A100.

## Global Constraints

- The only authoritative checkout is
  `/Users/bytedance/Desktop/TinyLLMForge`, resolving to
  `/Users/bytedance/dev/TinyLLMForge`.
- Stay on `feat/kv-sparse-attention`; do not create a worktree or use a
  subagent.
- Complete and reconcile the r10 one-phase lease-local-journal gate before
  modifying any source file in its frozen source manifest.
- Complete the generation-sealed block-table identity gate before beginning
  elastic-width runtime changes, so the elastic gate measures the accepted
  predecessor stack.
- Preserve every unrelated dirty or untracked file. Stage only exact task
  paths.
- Use `git -c core.hooksPath=/dev/null commit`.
- Every commit has exactly one
  `Co-authored-by: TRAE CLI <noreply@bytedance.com>` trailer.
- Push only to `origin/feat/kv-sparse-attention`.
- Use `python3 -m pytest`; `python` is unavailable locally.
- The feature flag is `exact_greedy_decode_burst_elastic_k16` and remains
  default-disabled.
- The base width remains exactly eight; the feature requires
  `exact_greedy_decode_burst=true`,
  `exact_greedy_decode_burst_tokens=8`, and
  `exact_greedy_decode_burst_split_phase=false`.
- K16 eligibility is frozen to initial sequence length `<=2048`, remaining
  output tokens `>=16`, writable positions in the current physical block
  `>=16`, a healthy K16 graph, and no incompatible mode.
- Contexts 4,096 and 8,192 always select K8.
- K16 replay or validation failure quarantines only K16 and never retries
  work in the same engine step.
- Preserve exact output tokens, decoded text, sampled logits, argmax, target
  forwards, graph replays, D2H behavior, KV slots, ordering, fairness, and
  rollback semantics.
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

- Modify `tinyvllm/config.py`: add the flag and strict dependency checks.
- Modify `tinyvllm/engine/exact_greedy_decode_burst.py`: add the pure elastic
  selector, graph-width identity, K16 counters, and width-aware contracts.
- Modify `tinyvllm/engine/model_runner.py`: own, capture, select, quarantine,
  and invalidate separate K8 and K16 graphs.
- Modify `tinyvllm/engine/scheduler.py`: request K16 only after deterministic
  eligibility and generalize the one-phase journal from fixed K8 to `{8,16}`.
- Modify `tinyvllm/engine/llm_engine.py`: pass both graph capabilities into
  scheduling and preserve fail-closed next-step fallback.
- Modify `tools/test_exact_greedy_decode_burst.py`: selector, graph identity,
  replay, D2H, and quarantine unit tests.
- Modify `tools/test_model_runner_spec_verify.py`: graph ownership, capture,
  routing, config, and serialization tests.
- Modify `tools/test_scheduler_prepared_postprocess.py`: width-aware K16
  transaction, rollback, publication, and counter tests.
- Create `tools/profile_context_gated_elastic_exact_burst.py`: paired K8/K16
  workload producer shared by the ceiling and terminal gates.
- Create `tools/test_profile_context_gated_elastic_exact_burst.py`.
- Create `tools/context_gated_elastic_exact_burst_ceiling.py`: strict
  pre-promotion ceiling classifier.
- Create `tools/test_context_gated_elastic_exact_burst_ceiling.py`.
- Create `tools/context_gated_elastic_exact_burst_gate.py`: terminal producer
  and classification.
- Create `tools/test_context_gated_elastic_exact_burst_gate.py`.
- Create `tools/context_gated_elastic_exact_burst_verify.py`: independent
  artifact reconstruction.
- Create `tools/test_context_gated_elastic_exact_burst_verify.py`.
- Create `tools/run_context_gated_elastic_exact_burst_remote.py`: mounted-only
  strict-clean remote controller.
- Create `tools/test_run_context_gated_elastic_exact_burst_remote.py`.
- Create
  `docs/superpowers/audits/2026-08-24-context-gated-elastic-exact-burst-audit.md`.
- Modify `docs/superpowers/audits/2026-08-16-phase1-completion-audit.md`.
- Modify `AGENT_HANDOFF_STATE.md`.

### Task 1: Add the Frozen Elastic Policy Contract

**Files:**

- Modify: `tinyvllm/config.py`
- Modify: `tinyvllm/engine/exact_greedy_decode_burst.py`
- Modify: `tools/test_exact_greedy_decode_burst.py`
- Modify: `tools/test_model_runner_spec_verify.py`

**Interfaces:**

- Produces:
  `select_context_gated_elastic_exact_burst_width(...) -> ElasticExactBurstWidthDecision`
- Produces immutable fields:
  `requested_width`, `selected_width`, `k16_eligible`,
  `k16_fallback_reason`.
- Preserves the existing `select_exact_greedy_decode_burst_width` result when
  the new flag is disabled.

- [ ] **Step 1: Write RED selector tests**

Add a table-driven test with these exact expectations:

```python
cases = (
    # enabled, context, remaining, writable, healthy, width, reason
    (False, 256, 128, 128, True, 8, "disabled"),
    (True, 2048, 16, 16, True, 16, None),
    (True, 2049, 128, 128, True, 8, "context_above_2048"),
    (True, 256, 15, 128, True, 8, "output_budget_below_16"),
    (True, 256, 128, 15, True, 8, "write_block_capacity_below_16"),
    (True, 256, 128, 128, False, 8, "k16_graph_unavailable"),
    (True, 4096, 128, 128, True, 8, "context_above_2048"),
    (True, 8192, 128, 128, True, 8, "context_above_2048"),
)
```

For each row assert the selected width and exact fallback reason. Add invalid
input tests for booleans-as-integers, non-positive context/block size,
negative output budget, base width other than eight, and a malformed
incompatible-mode tuple.

- [ ] **Step 2: Write RED configuration tests**

Construct the normal config fixture and assert:

```python
assert config.exact_greedy_decode_burst_elastic_k16 is False
```

Then require `ValueError` for each invalid composition:

```text
elastic K16 without exact burst
elastic K16 with exact_greedy_decode_burst_tokens != 8
elastic K16 with split phase
elastic K16 with ragged coalescing
elastic K16 with a non-bool flag value
```

- [ ] **Step 3: Run RED**

Run:

```bash
python3 -m pytest \
  tools/test_exact_greedy_decode_burst.py \
  tools/test_model_runner_spec_verify.py \
  -q
```

Expected: only the new selector/config tests fail because the flag, decision
type, and selector do not exist.

- [ ] **Step 4: Implement the pure selector**

Add:

```python
@dataclass(frozen=True)
class ElasticExactBurstWidthDecision:
    requested_width: int
    selected_width: int
    k16_eligible: bool
    k16_fallback_reason: Optional[str]
```

The selector validates all inputs, returns K8 immediately when disabled, and
checks conditions in this stable order:

```text
incompatible mode
context above 2048
output budget below 16
write-block capacity below 16
K16 graph unavailable or quarantined
```

It returns K16 only if no reason is present. It never changes or calls the
existing ragged-coalescing selector.

- [ ] **Step 5: Implement config validation**

Add the default-off dataclass field adjacent to the other exact-burst flags.
Validate it as `bool`; when true, require exact burst, fixed K8 base width,
split phase off, and ragged coalescing off.

- [ ] **Step 6: Run GREEN and adjacent tests**

Run:

```bash
python3 -m pytest \
  tools/test_exact_greedy_decode_burst.py \
  tools/test_model_runner_spec_verify.py \
  tools/test_exact_greedy_decode_burst_split_phase.py \
  tools/test_exact_burst_ragged_coalescing_gate.py \
  -q
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit and push Task 1**

Stage only the four Task 1 paths, inspect `git diff --cached`, commit with:

```text
feat(runtime): define elastic exact burst policy

Co-authored-by: TRAE CLI <noreply@bytedance.com>
```

Push to `origin/feat/kv-sparse-attention`.

### Task 2: Give K16 an Independent Graph Identity

**Files:**

- Modify: `tinyvllm/engine/exact_greedy_decode_burst.py`
- Modify: `tinyvllm/engine/model_runner.py`
- Modify: `tools/test_exact_greedy_decode_burst.py`
- Modify: `tools/test_model_runner_spec_verify.py`

**Interfaces:**

- Extends `ExactGreedyDecodeBurstCaptureReceipt` with
  `maximum_burst_width: int`.
- Extends `ExactGreedyDecodeBurstGraph.capture` with
  `maximum_burst_width: int`.
- Produces model-runner graph owners:
  `exact_greedy_decode_burst_k16_graph` and
  `exact_greedy_decode_burst_k16_correctness_graph`.
- Produces a capability mapping keyed by width:
  `exact_greedy_decode_burst_capabilities(...) -> {8: dict, 16: dict}`.

- [ ] **Step 1: Write RED graph-identity tests**

Capture fake K8 and K16 graphs with otherwise identical tensor metadata and
assert:

```python
assert k8.receipt.maximum_burst_width == 8
assert k16.receipt.maximum_burst_width == 16
assert k8.receipt.graph_identity_sha256 != k16.receipt.graph_identity_sha256
```

Require capture rejection when the width is not exactly 8 or 16, when token
history capacity is below the declared width, or when a replay lease exceeds
the graph's declared width.

- [ ] **Step 2: Write RED model-runner ownership tests**

Require:

```text
flag off: no K16 graph is captured
flag on: production and correctness K16 graphs are independently captured
K16 lease selects only the K16 graph
K8 lease selects only the existing K8 graph
K16 quarantine leaves K8 capability available
continuation invalidation visits both width owners exactly once
```

- [ ] **Step 3: Run RED**

Run:

```bash
python3 -m pytest \
  tools/test_exact_greedy_decode_burst.py \
  tools/test_model_runner_spec_verify.py \
  -q
```

Expected: the new receipt field, K16 owners, and width routing are absent.

- [ ] **Step 4: Implement width-bound graph capture**

Include `maximum_burst_width` in the capture identity payload and receipt.
Set token-history capacity to at least the declared width. Preserve the
existing one-complete-step CUDA graph body; K16 means sixteen ordered replays
of a width-bound owner, not a sixteen-token fused model forward.

Capture K8 exactly as before with width eight. If the elastic flag is enabled,
capture a second production graph and, only for correctness runs, a second
correctness graph with width sixteen. Keep graph pools, static tensors,
receipts, and quarantine state independent.

- [ ] **Step 5: Implement width routing and quarantine isolation**

Select a graph by `lease.requested_token_count`. Reject a width mismatch
before static-state mutation. Add the K16 owners to continuation invalidation
without deduplicating distinct graph objects. A K16 failure calls quarantine
only on the selected K16 graph; K8 remains available for the next engine
step.

- [ ] **Step 6: Run GREEN**

Run:

```bash
python3 -m pytest \
  tools/test_exact_greedy_decode_burst.py \
  tools/test_model_runner_spec_verify.py \
  tools/test_llm_engine_exact_greedy_decode_burst.py \
  -q
```

Expected: all selected tests pass and flag-off K8 fixture receipts remain
unchanged except for the explicit schema field fixed to eight.

- [ ] **Step 7: Commit and push Task 2**

Commit exact Task 2 paths with:

```text
feat(runtime): capture independent K16 burst graph

Co-authored-by: TRAE CLI <noreply@bytedance.com>
```

### Task 3: Authorize and Commit Width-Aware K16 Leases

**Files:**

- Modify: `tinyvllm/engine/exact_greedy_decode_burst.py`
- Modify: `tinyvllm/engine/scheduler.py`
- Modify: `tinyvllm/engine/llm_engine.py`
- Modify: `tools/test_exact_greedy_decode_burst.py`
- Modify: `tools/test_scheduler_prepared_postprocess.py`
- Modify: `tools/test_llm_engine_exact_greedy_decode_burst.py`

**Interfaces:**

- Extends `Scheduler.prepare_exact_greedy_decode_burst` with
  `elastic_k16_enabled: bool` and `k16_graph_available: bool`.
- Generalizes one-phase lease-local journal eligibility to exact widths
  `{8,16}`.
- Adds K16 attempt, acceptance, K8-fallback, fallback-reason, and per-width
  commit counters to `ExactGreedyDecodeBurstStats.summary()`.

- [ ] **Step 1: Write RED scheduler tests**

Create a running single sequence for each policy boundary and assert:

```text
context 2048, 16 remaining, 16 writable, healthy graph -> K16 lease
context 2049 -> K8 lease and context fallback reason
15 remaining -> K8 lease and output-budget fallback reason
15 writable -> K8 lease and block-capacity fallback reason
K16 unavailable/quarantined -> K8 lease
flag off -> byte-for-byte existing K8 lease payload
```

Assert the K16 lease has requested and authorized count sixteen, a
single-block physical range, and a digest different from the corresponding
K8 lease.

- [ ] **Step 2: Write RED journal and failure tests**

For a non-terminal K16 row, require lease-local journal attempt, capture, and
commit counts to each increase once; generic journal capture remains zero.
Inject failures at token append, publication, progress update, and final
commit and assert exact restoration of sequence tokens, block table,
refcounts, queue position, progress state, pending lease, and graph
quarantine.

Require no K16 lease when sixteen writes cross a physical block boundary.
Require terminal K16 to use the existing safe generic journal unless a
separate exact terminal proof exists; do not weaken the current terminal
exclusion.

- [ ] **Step 3: Run RED**

Run:

```bash
python3 -m pytest \
  tools/test_scheduler_prepared_postprocess.py \
  tools/test_llm_engine_exact_greedy_decode_burst.py \
  tools/test_exact_greedy_decode_burst.py \
  -q
```

Expected: K16 selection and width-aware one-phase journal assertions fail.

- [ ] **Step 4: Implement scheduler selection**

Compute writable positions from the current write position and block size,
call the pure elastic selector, and pass the selected width into the existing
decision/lease builder. Record K16-specific reasons separately from ordinary
exact-burst fallback; a deterministic K16-to-K8 choice is not an
exact-burst fallback.

- [ ] **Step 5: Generalize the one-phase journal**

Replace hard-coded one-phase width eight checks with:

```python
one_phase_width = lease.authorized_token_count
supported = one_phase_width in (8, 16)
expected_token_count = one_phase_width
terminal = (
    sequence.num_completion_tokens + one_phase_width
    >= sequence.max_tokens
)
```

Keep split-prefix and split-suffix widths fixed at four. Preserve the
at-most-one-write-block publication assertion and all rollback state.

- [ ] **Step 6: Implement next-step fail-closed fallback**

Pass separate K8 and K16 capabilities from the model runner. If K16 replay,
D2H, result validation, prepare, or commit fails, quarantine K16, cancel the
lease, and let the existing engine loop attempt K8 only on a later step. Do
not perform a synchronous per-token or K8 replay after a K16 replay began.

- [ ] **Step 7: Run GREEN and adjacent regression**

Run:

```bash
python3 -m pytest \
  tools/test_exact_greedy_decode_burst.py \
  tools/test_scheduler_prepared_postprocess.py \
  tools/test_llm_engine_exact_greedy_decode_burst.py \
  tools/test_exact_greedy_decode_burst_split_phase.py \
  tools/test_exact_burst_ragged_coalescing_gate.py \
  tools/test_exact_burst_one_phase_lease_local_journal_gate.py \
  -q
```

Expected: all selected tests pass.

- [ ] **Step 8: Commit and push Task 3**

Commit exact Task 3 paths with:

```text
feat(runtime): execute context-gated K16 bursts

Co-authored-by: TRAE CLI <noreply@bytedance.com>
```

### Task 4: Build the Source-Bound K8/K16 Profiler

**Files:**

- Create: `tools/profile_context_gated_elastic_exact_burst.py`
- Create: `tools/test_profile_context_gated_elastic_exact_burst.py`

**Interfaces:**

- Defines `POLICIES = ("fixed_k8", "context_gated_elastic_k16")`.
- Defines `CONTEXT_LENGTHS = (256, 2048, 4096, 8192)`.
- Defines `GENERATED_TOKENS = 128`, `REPETITIONS = 5`, and four correctness
  sampling points.
- Emits one JSON performance row per policy/context/repetition and one
  correctness row per policy/context/sample point.

- [ ] **Step 1: Write RED profiler contract tests**

Require exact alternating policy order by repetition, exact prompt token
length, fixed output length, temperature zero, `ignore_eos=true`, TP1,
batch one, and completion-only execution.

Validate every row contains:

```text
source SHA and source-file hashes
policy, context, repetition, order
TTFT, E2E, TPOT samples, throughput
host-visible gap samples
allocated/reserved peak bytes
capture duration and retained static bytes for K8 and K16
requested/authorized width histograms
K16 attempt/accept/fallback reasons
forwards, replays, D2H calls/bytes
journal attempts/captures/commits/fallbacks/rollbacks
tokens, decoded text, argmax, and float32 logit sidecar references
```

Reject NaN/Inf, duplicate identities, missing sidecars, width-policy
mismatches, and K16 selection at 4,096 or 8,192.

- [ ] **Step 2: Run RED**

Run:

```bash
python3 -m pytest \
  tools/test_profile_context_gated_elastic_exact_burst.py \
  -q
```

Expected: module import fails because the profiler does not exist.

- [ ] **Step 3: Implement the profiler**

Reuse the source-bound prompt construction, timing, logits sidecar, memory,
and lifecycle extraction helpers from the accepted exact-burst gates.
Import helpers rather than copying policy logic. The profiler must accept
`--repetitions`, `--output-dir`, `--model`, and `--device`; it must create the
output directory with exclusive semantics and refuse an existing run.

- [ ] **Step 4: Run GREEN**

Run:

```bash
python3 -m pytest \
  tools/test_profile_context_gated_elastic_exact_burst.py \
  tools/test_profile_exact_burst_one_phase_lease_local_journal.py \
  -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit and push Task 4**

Commit the two profiler paths with:

```text
test(runtime): profile elastic exact burst width

Co-authored-by: TRAE CLI <noreply@bytedance.com>
```

### Task 5: Run the Mandatory Ceiling Probe

**Files:**

- Create: `tools/context_gated_elastic_exact_burst_ceiling.py`
- Create: `tools/test_context_gated_elastic_exact_burst_ceiling.py`
- Create: `tools/run_context_gated_elastic_exact_burst_remote.py`
- Create: `tools/test_run_context_gated_elastic_exact_burst_remote.py`

**Interfaces:**

- Produces terminal classifications:
  `CEILING_GO`, `NO_GO_INSUFFICIENT_INCREMENTAL_BENEFIT`,
  `NO_GO_BURST_GAP`, `NO_GO_CORRECTNESS`, and
  `NO_GO_EVIDENCE_INCOMPLETE`.
- Uses a fresh immutable run tag and the approved mounted remote root.

- [ ] **Step 1: Write RED ceiling-classifier tests**

Synthetic fixtures must independently prove each terminal classification.
`CEILING_GO` requires:

```text
at least 1.5% median TPOT improvement at either 256 or 2048
maximum selected-K16 host-visible gap <= 40,000,000 ns
exact output tokens and sampled logits
forwards == replays == emitted tokens
final token D2H calls == burst count
intermediate token D2H calls == 0
final token D2H bytes == emitted tokens * 8
K16 selected at both eligible contexts
K16 never selected at 4096 or 8192
```

Any incomplete or duplicate inventory must classify as evidence incomplete,
not performance failure.

- [ ] **Step 2: Write RED controller tests**

Require Kerberos TTL fail-fast, source SHA equal to pushed branch HEAD,
empty source patch, strict-clean GPU selection, exclusive remote/local paths,
mounted-only paths, deterministic distinct port, worker PID/PGID receipts,
bounded polling retries, remote verifier execution, bundle download, and
frozen-source local verification.

- [ ] **Step 3: Implement classifier and controller**

The ceiling run uses two policies, four contexts, and at least three paired
repetitions. It writes a complete manifest, source hashes, raw rows,
comparison, summary, gate, producer receipt, remote-verifier receipt, local
verifier receipt, controller receipt, logs, PID/PGID, and exit code.

- [ ] **Step 4: Run all CPU tests and source checks**

Run:

```bash
python3 -m pytest \
  tools/test_context_gated_elastic_exact_burst_ceiling.py \
  tools/test_run_context_gated_elastic_exact_burst_remote.py \
  tools/test_profile_context_gated_elastic_exact_burst.py \
  -q
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-elastic-k16-pycache \
  python3 -m py_compile \
  tools/context_gated_elastic_exact_burst_ceiling.py \
  tools/run_context_gated_elastic_exact_burst_remote.py \
  tools/profile_context_gated_elastic_exact_burst.py
git diff --check
```

Expected: all tests pass, compilation succeeds, and diff check is clean.

- [ ] **Step 5: Commit, push, and launch a fresh ceiling tag**

Commit the four Task 5 paths with:

```text
test(runtime): gate elastic K16 ceiling

Co-authored-by: TRAE CLI <noreply@bytedance.com>
```

Push first. Launch only when the source SHA equals the pushed branch HEAD and
a strict-clean GPU is available.

- [ ] **Step 6: Enforce the decision checkpoint**

If the complete source-bound result is not `CEILING_GO`, stop elastic K16
promotion and write the audit with measured benefit and cost. Do not create a
terminal-gate performance claim and do not tune the 2,048-token threshold.

If the result is `CEILING_GO`, continue to Task 6 without changing the
selector threshold or gate criteria.

### Task 6: Build and Run the Terminal Paired Gate

**Files:**

- Create: `tools/context_gated_elastic_exact_burst_gate.py`
- Create: `tools/test_context_gated_elastic_exact_burst_gate.py`
- Create: `tools/context_gated_elastic_exact_burst_verify.py`
- Create: `tools/test_context_gated_elastic_exact_burst_verify.py`
- Modify: `tools/run_context_gated_elastic_exact_burst_remote.py`
- Modify: `tools/test_run_context_gated_elastic_exact_burst_remote.py`

**Interfaces:**

- Requires exactly 40 performance rows and 32 correctness rows.
- Produces `GO_CONTEXT_GATED_ELASTIC_EXACT_BURST` or one explicit no-go
  classification.
- Producer and independent verifier reconstruct metrics from raw rows and
  float32 sidecars independently.

- [ ] **Step 1: Write RED producer and verifier tests**

Build valid synthetic artifacts exactly at every threshold and perturb one
requirement at a time. Cover:

```text
eligible aggregate median TPOT improvement >= 2%
eligible aggregate P95 TPOT improvement >= 1%
per-context median/P95 regression <= 2%
maximum gap <= 40 ms
TTFT/E2E/TPOT-P99 regression <= 2%
throughput regression <= 1%
allocated/reserved memory regression <= 3%
exact token/text/logit/argmax parity
width selection by context
forwards/replays/D2H invariants
zero unexpected fallback/rollback/quarantine
manifest and source-hash integrity
```

Add tamper tests showing the verifier rejects altered summary, gate,
sidecars, row inventory, source hash, and producer receipt.

- [ ] **Step 2: Run RED**

Run:

```bash
python3 -m pytest \
  tools/test_context_gated_elastic_exact_burst_gate.py \
  tools/test_context_gated_elastic_exact_burst_verify.py \
  tools/test_run_context_gated_elastic_exact_burst_remote.py \
  -q
```

Expected: terminal gate and verifier modules are absent.

- [ ] **Step 3: Implement producer and independent verifier**

The producer writes raw evidence before derived files. The verifier reads no
producer comparison fields when recomputing metrics. Both enforce exact
inventory and finite numeric values and agree on the final classification.
The local verifier runs from the downloaded frozen source, not the live
checkout.

- [ ] **Step 4: Run GREEN and regression**

Run:

```bash
python3 -m pytest \
  tools/test_context_gated_elastic_exact_burst_gate.py \
  tools/test_context_gated_elastic_exact_burst_verify.py \
  tools/test_run_context_gated_elastic_exact_burst_remote.py \
  tools/test_exact_burst_one_phase_lease_local_journal_gate.py \
  tools/test_exact_burst_one_phase_lease_local_journal_verify.py \
  -q
git diff --check
```

Expected: all selected tests pass and diff check is clean.

- [ ] **Step 5: Commit, push, and run the terminal gate**

Commit exact Task 6 paths with:

```text
test(runtime): verify elastic exact burst gate

Co-authored-by: TRAE CLI <noreply@bytedance.com>
```

Push first, use a fresh tag, and wait for all 40 performance rows, all 32
correctness rows, worker exit code, producer result, remote verifier, bundle
download, and frozen-source local verifier.

### Task 7: Reconcile Evidence and Claims

**Files:**

- Create:
  `docs/superpowers/audits/2026-08-24-context-gated-elastic-exact-burst-audit.md`
- Modify: `docs/superpowers/audits/2026-08-16-phase1-completion-audit.md`
- Modify: `AGENT_HANDOFF_STATE.md`

- [ ] **Step 1: Build the prompt-to-artifact checklist**

Map every objective, selector boundary, config dependency, graph owner,
failure rule, row count, metric threshold, invariant, remote-path rule,
receipt, and claim boundary to an exact source path, test, and artifact.
Mark uncertainty or partial rows as not verified.

- [ ] **Step 2: Report benefit and cost**

Report overall and per-context TPOT median/P95/P99, throughput, TTFT, E2E,
maximum/P95 visibility gap, capture duration, retained static bytes, allocated
and reserved memory, K8 fallback rate, lifecycle counters, correctness, and
the exact no-go reason if any threshold fails.

- [ ] **Step 3: Run final verification**

Run the full focused elastic suite, adjacent exact-burst suites,
`py_compile`, `git diff --check`, and the frozen-source local verifier. Confirm
the local branch SHA equals `origin/feat/kv-sparse-attention` after push.

- [ ] **Step 4: Commit and push reconciliation**

Stage only the three documentation paths and any intentionally committed
terminal artifact receipts named by the audit. Commit with:

```text
docs(runtime): audit elastic exact burst

Co-authored-by: TRAE CLI <noreply@bytedance.com>
```

Push to `origin/feat/kv-sparse-attention`.

## Self-Review

- Spec coverage: every policy boundary, runtime owner, failure path, CPU
  contract, ceiling threshold, terminal threshold, evidence count, remote
  safety rule, benefit/cost field, and claim boundary maps to a task.
- Placeholder scan: the plan contains no deferred implementation placeholders;
  the only conditional work is the explicit measured ceiling checkpoint.
- Type consistency: the selector decision, capture receipt width, K8/K16 graph
  owners, scheduler arguments, policy names, context inventory, and terminal
  classifications are stable across tasks.
- Scope boundary: the plan does not authorize streaming, EOS-aware generation,
  multi-sequence scheduling, tensor parallelism, Qwen3-8B, threshold retuning,
  or production-default enablement.
