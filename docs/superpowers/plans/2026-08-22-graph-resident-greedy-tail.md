# Graph-Resident Greedy Tail Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce batch-1 zero-temperature decode TPOT and tail latency by replaying LM-head, float32 conversion, and argmax inside a second CUDA Graph that reads the existing transformer's static hidden output.

**Architecture:** A model-agnostic module owns eligibility, source binding, capture/replay lifecycle, quarantine, and accounting. `ModelRunner` captures one tail graph against the existing batch-1 transformer graph output, decides eligibility before transformer replay, and returns an explicit retained-logits/token result so `_run_model_step` performs exactly one final token D2H. A source-bound three-arm gate compares legacy, host-greedy, and graph-greedy paths while reporting correctness, latency, capture cost, and CUDA memory cost.

**Tech Stack:** Python 3, PyTorch CUDA Graphs, TinyLLMForge `ModelRunner`, dependency-light script tests, JSON/JSONL evidence, binary float32 logit sidecars, SSH remote runner.

## Global Constraints

- Work only in `/Users/bytedance/dev/TinyLLMForge`, the target of the authoritative `/Users/bytedance/Desktop/TinyLLMForge` symlink.
- Do not create worktrees or use subagents.
- Never modify `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Preserve all unrelated dirty and untracked files.
- Stage exact paths only; never use broad `git add`, `git reset`, `git clean`, or mass formatting.
- Commit with `git -c core.hooksPath=/dev/null commit`.
- Every commit has exactly one `Co-authored-by: TRAE CLI <noreply@bytedance.com>` trailer.
- Push only to `origin/feat/kv-sparse-attention`.
- The feature flag is `graph_resident_greedy_tail` and defaults to `False`.
- The mechanism layer must not contain Qwen3, checkpoint, prompt, tokenizer, or workload names.
- Stage 1 covers Qwen3-0.6B, tensor parallel size one, rank zero, batch size one, ordinary CUDA-Graph decode, and exact numeric `temperature == 0.0`.
- The captured tail expression is exactly `compute_logits(static_hidden[:1])` followed by `logits.to(torch.float32).argmax(dim=-1)`.
- Eligibility must be decided before transformer replay; no fallback is allowed after current-step KV mutation.
- Replay failure quarantines the tail for the runner lifetime and propagates the error without replaying the transformer or resampling.
- The final one-element token tensor performs exactly one `.tolist()` because scheduler state remains host-owned.
- Output token IDs and decoded-text hashes must match exactly across all three arms.
- Logit correctness remains `max_abs <= 0.25`, per-pair `mean_abs <= 0.05`, and argmax equality.
- Graph-greedy must replay the tail on every measured decode step after warmup.
- Report benefit and cost together, including capture duration, retained static bytes, allocated/reserved deltas, and final token D2H cost.
- Every remote run tag is immutable.
- All remote task output stays under `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818`.
- Never write remote task data under `/`, `/tmp`, `/private/tmp`, or `/data00/home/sitian/tllm/TinyLLMForge`.
- Do not refresh Kerberos automatically.
- Do not terminate or interfere with unrelated GPU processes.
- GPU admission requires memory used `<=1024 MiB`, utilization `<=5%`, and no compute process.
- Qwen3-0.6B evidence cannot support Qwen3-8B or tensor-parallel claims.
- Do not launch Qwen3-8B unless the Qwen3-0.6B Stage-1 gate is GO.

---

## File Structure

- Create `tinyvllm/engine/graph_resident_greedy_tail.py`: generic eligibility, capture receipt, explicit replay result, lifecycle, quarantine, and accounting.
- Create `tools/test_graph_resident_greedy_tail.py`: dependency-light policy and fake-graph lifecycle tests.
- Modify `tinyvllm/config.py`: add and validate the default-disabled flag.
- Modify `tinyvllm/engine/model_runner.py`: capture the tail, decide before transformer replay, expose summaries, and consume the explicit replay result.
- Modify `tools/test_model_runner_spec_verify.py`: source-level and fake-runner integration tests.
- Create `tools/profile_graph_resident_greedy_tail.py`: three-arm performance worker and bounded logits sidecars.
- Create `tools/test_profile_graph_resident_greedy_tail.py`: worker schema, order, counter, and sidecar tests.
- Create `tools/graph_resident_greedy_tail_gate.py`: producer validation and fixed-precedence classification.
- Create `tools/graph_resident_greedy_tail_verify.py`: independent reconstruction without importing producer classification logic.
- Create `tools/test_graph_resident_greedy_tail_gate.py`: GO and every NO-GO producer fixture.
- Create `tools/test_graph_resident_greedy_tail_verify.py`: independent-verifier and tamper fixtures.
- Create `tools/run_graph_resident_greedy_tail_remote.py`: source-bound remote controller.
- Create `tools/test_run_graph_resident_greedy_tail_remote.py`: remote safety, preflight, immutable-tag, and download tests.
- Modify `AGENT_HANDOFF_STATE.md`: append the terminal result at EOF.
- Modify `docs/superpowers/audits/2026-08-16-phase1-completion-audit.md`: append reconciliation and prompt-to-artifact evidence at EOF.

### Task 1: Generic eligibility and accounting contract

**Files:**

- Create: `tinyvllm/engine/graph_resident_greedy_tail.py`
- Create: `tools/test_graph_resident_greedy_tail.py`

**Interfaces:**

- Produces:
  - `GraphResidentGreedyTailDecision`
  - `GraphResidentGreedyTailStats`
  - `GraphResidentGreedyTailCaptureReceipt`
  - `GraphResidentGreedyTailReplay`
  - `decide_graph_resident_greedy_tail(...)`
  - `tensor_identity(tensor) -> tuple[int, tuple[int, ...], tuple[int, ...], int, str, str]`

- [ ] **Step 1: Write failing eligibility tests**

Use a direct-execution test script with no Torch import. The accepted case is:

```python
decision = decide_graph_resident_greedy_tail(
    enabled=True,
    rank=0,
    tensor_parallel_size=1,
    is_prefill=False,
    enforce_eager=False,
    batch_kind=None,
    active_batch_size=1,
    selected_graph_batch_size=1,
    do_sample=True,
    temperatures=(0.0,),
    input_embeds_present=False,
    return_hidden=False,
    incompatible_modes=(),
    capture_available=True,
    quarantined=False,
    source_matches=True,
)
assert decision.optimized is True
assert decision.fallback_reason is None
```

Add exact fallback assertions, in validation order, for `disabled`,
`non_root_rank`, `tensor_parallel_unsupported`, `prefill_unsupported`,
`eager_unsupported`, `mixed_batch_unsupported`, `batch_size_unsupported`,
`selected_graph_batch_unsupported`, `sampling_disabled`,
`temperature_invalid`, `nonzero_temperature`,
`input_embeds_unsupported`, `return_hidden_unsupported`,
`incompatible_mode`, `capture_unavailable`, `quarantined`, and
`source_identity_drift`.

- [ ] **Step 2: Run the focused test and confirm RED**

Run:

```bash
python3 tools/test_graph_resident_greedy_tail.py
```

Expected: import failure because
`tinyvllm.engine.graph_resident_greedy_tail` does not exist.

- [ ] **Step 3: Implement immutable contracts and exact counters**

Implement these public shapes:

```python
@dataclass(frozen=True)
class GraphResidentGreedyTailDecision:
    optimized: bool
    fallback_reason: str | None


@dataclass(frozen=True)
class GraphResidentGreedyTailCaptureReceipt:
    source_identity: tuple[
        int,
        tuple[int, ...],
        tuple[int, ...],
        int,
        str,
        str,
    ]
    graph_generation: int
    rank: int
    capture_duration_ns: int
    allocated_delta_bytes: int
    reserved_delta_bytes: int
    retained_logits_bytes: int
    retained_float32_bytes: int
    retained_token_bytes: int


@dataclass(frozen=True)
class GraphResidentGreedyTailReplay:
    logits: object
    token_ids: object


@dataclass
class GraphResidentGreedyTailStats:
    eligible_steps: int = 0
    captured_graphs: int = 0
    replayed_steps: int = 0
    final_token_d2h_calls: int = 0
    avoided_external_compute_logits_calls: int = 0
    avoided_external_float32_conversions: int = 0
    avoided_external_argmax_calls: int = 0
    fallback_counts: dict[str, int] = field(default_factory=dict)
    quarantine_reason: str | None = None
```

Validate non-boolean integer counters, non-empty fallback reasons, and
non-negative byte/duration values. `summary()` must return only JSON-safe
values and include the capture receipt when available.

- [ ] **Step 4: Run GREEN and syntax checks**

```bash
python3 tools/test_graph_resident_greedy_tail.py
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-graph-tail-policy-pycache \
  python3 -m py_compile \
  tinyvllm/engine/graph_resident_greedy_tail.py \
  tools/test_graph_resident_greedy_tail.py
```

Expected: the script prints its pass marker and exits zero.

- [ ] **Step 5: Commit and push**

```bash
git add -- \
  tinyvllm/engine/graph_resident_greedy_tail.py \
  tools/test_graph_resident_greedy_tail.py
git -c core.hooksPath=/dev/null commit \
  -m "feat(perf): add graph greedy tail contracts" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

### Task 2: Generic capture, replay, and quarantine lifecycle

**Files:**

- Modify: `tinyvllm/engine/graph_resident_greedy_tail.py`
- Modify: `tools/test_graph_resident_greedy_tail.py`

**Interfaces:**

- Produces:
  - `GraphResidentGreedyTail.capture(...)`
  - `GraphResidentGreedyTail.matches(...)`
  - `GraphResidentGreedyTail.replay(...)`
  - `GraphResidentGreedyTail.mark_token_d2h()`
  - `GraphResidentGreedyTail.summary()`

- [ ] **Step 1: Add failing fake-graph lifecycle tests**

Create fake tensor, logits, graph, capture context, synchronizer, memory
snapshot, and clock objects. Prove:

```python
tail = GraphResidentGreedyTail.capture(
    static_hidden=hidden,
    compute_logits=compute_logits,
    float32_dtype="float32",
    graph_generation=7,
    rank=0,
    graph_factory=FakeGraph,
    capture_context_factory=fake_capture_context,
    synchronize=synchronize,
    memory_snapshot=memory_snapshot,
    clock_ns=clock_ns,
)
assert tail.matches(
    static_hidden=hidden,
    graph_generation=7,
    rank=0,
)
result = tail.replay(
    static_hidden=hidden,
    graph_generation=7,
    rank=0,
)
assert result.logits is tail.logits
assert result.token_ids is tail.token_ids
```

Assert capture runs one warmup sequence and one captured sequence, the graph
body calls `compute_logits`, `.to("float32")`, and `.argmax(dim=-1)`, and
does not clone or copy the hidden tensor.

- [ ] **Step 2: Run the lifecycle tests and confirm RED**

```bash
python3 tools/test_graph_resident_greedy_tail.py
```

Expected: failure on the missing `GraphResidentGreedyTail` lifecycle API.

- [ ] **Step 3: Implement capture and replay**

`tensor_identity()` must use `data_ptr`, shape, stride, storage offset,
dtype, and device so a fresh Python slice object over the same static storage
still matches. `capture()` must:

1. validate the stable source identity and positive graph generation;
2. synchronize and record allocated/reserved memory before warmup;
3. run one uncaptured warmup of the exact tail expression;
4. synchronize, start the clock, and capture the same expression;
5. synchronize, stop the clock, and record memory deltas;
6. retain the captured logits and one-element token tensors;
7. record exactly one captured graph and a complete receipt.

The tail graph must use its own CUDA Graph pool rather than the transformer
graphs' shared pool. `replay()` must verify source identity, graph generation, and rank before
calling `graph.replay()`. Any replay exception must set one permanent
quarantine reason formed as `"replay_failure:" + type(error).__name__` and
re-raise the original exception. A quarantined object must reject later
replay without calling the graph again.

- [ ] **Step 4: Add capture-failure and drift tests**

Prove capture failures produce no usable object, replay failure quarantines,
and identity, shape, dtype, device, graph-generation, and rank drift are
rejected before `graph.replay()`. Prove `mark_token_d2h()` increments exactly
once per successful host conversion and rejects duplicate accounting for one
replay.

- [ ] **Step 5: Run GREEN and syntax checks**

```bash
python3 tools/test_graph_resident_greedy_tail.py
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-graph-tail-lifecycle-pycache \
  python3 -m py_compile \
  tinyvllm/engine/graph_resident_greedy_tail.py \
  tools/test_graph_resident_greedy_tail.py
```

- [ ] **Step 6: Commit and push**

```bash
git add -- \
  tinyvllm/engine/graph_resident_greedy_tail.py \
  tools/test_graph_resident_greedy_tail.py
git -c core.hooksPath=/dev/null commit \
  -m "feat(perf): capture graph-resident greedy tail" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

### Task 3: Default-disabled ModelRunner integration

**Files:**

- Modify: `tinyvllm/config.py`
- Modify: `tinyvllm/engine/model_runner.py`
- Modify: `tools/test_model_runner_spec_verify.py`

**Interfaces:**

- Consumes the Task 1-2 graph-tail module.
- Produces:
  - `Config.graph_resident_greedy_tail: bool`
  - `ModelRunner.graph_resident_greedy_tail_summary() -> dict`
  - `_capture_graph_resident_greedy_tail()`
  - `_graph_resident_greedy_tail_request(...)`
  - explicit `GraphResidentGreedyTailReplay` handling in `run_model()` and `_run_model_step()`

- [ ] **Step 1: Add failing config and source-contract tests**

Assert:

```python
assert (
    Config.__dataclass_fields__[
        "graph_resident_greedy_tail"
    ].default
    is False
)
```

A non-boolean value must raise:

```text
graph_resident_greedy_tail must be a bool
```

Source assertions must prove the ordinary batch-1 graph branch decides the
tail before `graph.replay()`, returns `GraphResidentGreedyTailReplay` only
after successful transformer and tail replay, and preserves the old
`compute_logits(graph_vars["outputs"][:bs])` fallback.

- [ ] **Step 2: Run and confirm RED**

```bash
python3 tools/test_model_runner_spec_verify.py
```

Expected: failure on the missing config field and graph-tail integration.

- [ ] **Step 3: Add config and runner-owned lifecycle**

Add:

```python
graph_resident_greedy_tail: bool = False
```

Initialize:

```python
self.graph_resident_greedy_tail = None
self.graph_resident_greedy_tail_stats = (
    GraphResidentGreedyTailStats()
)
self._ordinary_graph_generation = 0
```

At the end of `capture_cudagraph()`, increment the generation and, only when
the flag is enabled, TP size is one, rank is zero, graph batch size one
exists, and no static incompatible mode is enabled, capture against:

```python
static_hidden = self.graph_vars["outputs"][:1]
```

Supply `self.model.compute_logits`, `torch.float32`,
`torch.cuda.CUDAGraph`, a dedicated tail capture context, CUDA synchronization,
`torch.cuda.memory_allocated`, `torch.cuda.memory_reserved`, and
`time.perf_counter_ns` through the generic capture API. Capture failure must
record `"capture_failure:" + type(error).__name__`, leave the tail
unavailable, and preserve startup.

- [ ] **Step 4: Decide before transformer replay**

Extend `run_model()` with one keyword-only request carrying:

```python
graph_tail_temperatures: tuple[object, ...] | None = None
graph_tail_do_sample: bool = False
graph_tail_batch_kind: str | None = None
```

Immediately before the existing ordinary graph replay:

1. determine active and selected graph batch sizes;
2. build `incompatible_modes` from Quest, AM compact, C4, CPU offload,
   KV offload, input embeddings, hidden-state return, and non-ordinary mode;
3. call `decide_graph_resident_greedy_tail(...)`;
4. record a stable fallback reason when rejected;
5. replay the transformer exactly once;
6. on an accepted decision, replay the tail on the same current stream and
   return `GraphResidentGreedyTailReplay`;
7. otherwise execute the existing external `compute_logits`.

Do not catch tail replay failure in `run_model()`.

- [ ] **Step 5: Consume the explicit result exactly once**

In `_run_model_step()`, pass sequence temperatures, `do_sample`, and
`batch_kind` only for ordinary decode. If the result is a
`GraphResidentGreedyTailReplay`:

```python
logits = result.logits
graph_tail_token_ids = result.token_ids
```

Preserve spec trace and step-logit recording from retained logits. On rank
zero, bypass `_sample_tokens_with_optional_greedy_fast_path()` and call:

```python
token_ids = graph_tail_token_ids.tolist()
self.graph_resident_greedy_tail.mark_token_d2h()
```

Every non-tail result keeps the current sampling code unchanged.

- [ ] **Step 6: Add integration and fallback tests**

Using fake runner components, prove:

- eligible graph decode performs one transformer replay, one tail replay,
  zero external `compute_logits`, zero sampler calls, and one `.tolist()`;
- step-logit recording reads retained graph logits;
- disabled, prefill, nonzero temperature, mixed batch, batch size above one,
  TP above one, non-root rank, eager mode, incompatible features, missing
  capture, and source drift execute the current path;
- tail replay failure performs no fallback, no second transformer replay,
  no sampling, and leaves the tail quarantined;
- default-disabled source behavior is unchanged.

- [ ] **Step 7: Run focused and neighboring regressions**

```bash
python3 tools/test_graph_resident_greedy_tail.py
python3 tools/test_greedy_sampling_fast_path.py
python3 tools/test_model_runner_spec_verify.py
python3 tools/test_decode_metadata_landing.py
python3 tools/test_chunked_prefill.py
python3 tools/test_multi_sequence_cuda_graph_gate.py
python3 tools/test_source_audit.py
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-graph-tail-runner-pycache \
  python3 -m py_compile \
  tinyvllm/config.py \
  tinyvllm/engine/model_runner.py \
  tinyvllm/engine/graph_resident_greedy_tail.py
```

Any Torch-dependent test unavailable locally remains required in remote
preflight and must be reported as environment-blocked, not passing.

- [ ] **Step 8: Commit and push**

```bash
git add -- \
  tinyvllm/config.py \
  tinyvllm/engine/model_runner.py \
  tinyvllm/engine/graph_resident_greedy_tail.py \
  tools/test_model_runner_spec_verify.py
git -c core.hooksPath=/dev/null commit \
  -m "feat(perf): replay greedy tail in cuda graph" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

### Task 4: Three-arm worker and logits sidecars

**Files:**

- Create: `tools/profile_graph_resident_greedy_tail.py`
- Create: `tools/test_profile_graph_resident_greedy_tail.py`

**Interfaces:**

- Produces schemas:
  - `graph-resident-greedy-tail.case.v1`
  - `graph-resident-greedy-tail.correctness.v1`
  - `graph-resident-greedy-tail.summary.v1`
  - `graph-resident-greedy-tail.workload.v1`
  - `graph-resident-greedy-tail.source.v1`
- Produces policies `legacy`, `host_greedy`, and `graph_greedy`.
- Produces a bounded `logits/` directory of little-endian float32 sidecars.

- [ ] **Step 1: Write failing pure worker-contract tests**

Assert:

```python
assert context_cases() == (
    ("short", 256, 128),
    ("medium", 2048, 128),
    ("long", 8192, 128),
)
assert policy_order(0) == (
    "legacy",
    "host_greedy",
    "graph_greedy",
)
assert policy_order(1) == (
    "graph_greedy",
    "host_greedy",
    "legacy",
)
```

Round-trip known float32 values and reject stale SHA256, wrong byte length,
non-finite values, reused case identity, wrong output length, or a
graph-greedy row whose tail replay and final token D2H counts are not 127.
The first of 128 generated tokens is sampled from prefill; only the remaining
127 tokens use ordinary decode.

- [ ] **Step 2: Run and confirm RED**

```bash
python3 tools/test_profile_graph_resident_greedy_tail.py
```

Expected: import failure because the worker does not exist.

- [ ] **Step 3: Implement three-arm performance rows**

Construct fresh `LLM` instances with only these flag differences:

```python
POLICY_FLAGS = {
    "legacy": (False, False),
    "host_greedy": (True, False),
    "graph_greedy": (True, True),
}
```

The tuple is
`(zero_temperature_greedy_fast_path, graph_resident_greedy_tail)`.
Reuse the canonical request loop and decode profiler from
`profile_zero_temperature_greedy_fast_path.py`. Record exact output IDs,
text hash, TTFT, E2E, 127 TPOT samples, decode host/CUDA samples, throughput,
peak allocated/reserved CUDA memory, greedy-fast-path delta, graph-tail
delta, capture duration, retained bytes, and final token D2H count.

- [ ] **Step 4: Implement three-point logits probes**

For every bucket and arm, run a separate deterministic request retaining:

```text
prefill-final
decode-first
decode-final
```

Write little-endian float32 bytes plus shape, element count, byte length,
SHA256, policy, bucket, and sampling point. Disable internal profiling during
correctness probes.

- [ ] **Step 5: Run GREEN and syntax checks**

```bash
python3 tools/test_profile_graph_resident_greedy_tail.py
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-graph-tail-worker-pycache \
  python3 -m py_compile \
  tools/profile_graph_resident_greedy_tail.py \
  tools/test_profile_graph_resident_greedy_tail.py
```

- [ ] **Step 6: Commit and push**

```bash
git add -- \
  tools/profile_graph_resident_greedy_tail.py \
  tools/test_profile_graph_resident_greedy_tail.py
git -c core.hooksPath=/dev/null commit \
  -m "test(perf): profile graph-resident greedy tail" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

### Task 5: Producer gate and independent verifier

**Files:**

- Create: `tools/graph_resident_greedy_tail_gate.py`
- Create: `tools/graph_resident_greedy_tail_verify.py`
- Create: `tools/test_graph_resident_greedy_tail_gate.py`
- Create: `tools/test_graph_resident_greedy_tail_verify.py`

**Interfaces:**

- Producer classifications:
  - `GO_GRAPH_RESIDENT_GREEDY_TAIL`
  - `NO_GO_CORRECTNESS`
  - `NO_GO_GRAPH_REPLAY_INCOMPLETE`
  - `NO_GO_LEGACY_TPOT_MEDIAN`
  - `NO_GO_LEGACY_TPOT_P95`
  - `NO_GO_HOST_GREEDY_INCREMENTAL`
  - `NO_GO_PROTECTED_REGRESSION`
  - `NO_GO_COST_INCOMPLETE`
  - `NO_GO_EVIDENCE_INCOMPLETE`
- The independent verifier must not import producer classification,
  percentile, pairing, comparison, or manifest functions.

- [ ] **Step 1: Write producer RED tests**

Build a complete synthetic fixture with 45 performance rows and 27 logits
sidecars. Prove GO plus each classification above. Also reject duplicate
identity, stale source hash, stale sidecar hash, non-finite metric, missing
capture cost, and inconsistent graph-tail counters.

- [ ] **Step 2: Implement producer validation and comparison**

The producer must:

1. require exactly 45 performance rows and 27 correctness rows;
2. pair performance by `(bucket, repetition)` across all three arms;
3. pair correctness by `(bucket, sampling_point)` across all three arms;
4. independently read float32 bytes and calculate max/mean absolute
   differences plus argmax;
5. reconstruct medians and nearest-rank P95/P99 from raw TPOT samples;
6. compare graph-greedy to both legacy and host-greedy;
7. apply the spec thresholds in the classification order above;
8. report capture duration, retained bytes, memory deltas, avoided calls,
   and token D2H count;
9. write `comparison.json`, `gate.json`, and `manifest.sha256`.

- [ ] **Step 3: Write independent-verifier RED tests**

Prove independent reconstruction of GO and rejection of producer comparison
drift, classification drift, missing sidecars, stale sidecar digests,
missing manifest entries, stale primary digests, and changed threshold
semantics.

- [ ] **Step 4: Implement the verifier separately**

Emit:

```python
{
    "schema_version":
        "graph-resident-greedy-tail.independent-verification.v1",
    "status": "PASS",
    "reconstructed_classification": classification,
    "comparison_sha256": comparison_digest,
    "manifest_sha256": manifest_digest,
}
```

Duplicate the small percentile, pairing, float32 decoding, threshold, and
manifest logic intentionally.

- [ ] **Step 5: Run GREEN and syntax checks**

```bash
python3 tools/test_graph_resident_greedy_tail_gate.py
python3 tools/test_graph_resident_greedy_tail_verify.py
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-graph-tail-gate-pycache \
  python3 -m py_compile \
  tools/graph_resident_greedy_tail_gate.py \
  tools/graph_resident_greedy_tail_verify.py \
  tools/test_graph_resident_greedy_tail_gate.py \
  tools/test_graph_resident_greedy_tail_verify.py
```

- [ ] **Step 6: Commit and push**

```bash
git add -- \
  tools/graph_resident_greedy_tail_gate.py \
  tools/graph_resident_greedy_tail_verify.py \
  tools/test_graph_resident_greedy_tail_gate.py \
  tools/test_graph_resident_greedy_tail_verify.py
git -c core.hooksPath=/dev/null commit \
  -m "test(perf): gate graph-resident greedy tail" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

### Task 6: Safe source-bound remote controller

**Files:**

- Create: `tools/run_graph_resident_greedy_tail_remote.py`
- Create: `tools/test_run_graph_resident_greedy_tail_remote.py`

**Interfaces:**

- Reuses the established SSH, Kerberos, GPU admission, upload, polling, and
  chunked-download helpers.
- Uses only:

```text
/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/
  graph-resident-greedy-tail/
```

- [ ] **Step 1: Write controller RED tests**

Cover immutable valid tags, approved-root confinement, Kerberos lifetime at
least 5,400 seconds, strict-clean GPU selection, fixed remote Python and
model checks, source commit equality with pushed HEAD, source archive
limited to `tinyvllm/` and `tools/`, isolated runtime/cache environment,
dependency-light and Torch-dependent preflight, second GPU admission before
launch, complete chunked download, and tamper rejection.

- [ ] **Step 2: Run and confirm RED**

```bash
python3 tools/test_run_graph_resident_greedy_tail_remote.py
```

Expected: import failure because the controller does not exist.

- [ ] **Step 3: Implement the controller**

Preflight must run:

```text
tools/test_graph_resident_greedy_tail.py
tools/test_greedy_sampling_fast_path.py
tools/test_model_runner_spec_verify.py
tools/test_multi_sequence_cuda_graph_gate.py
tools/test_chunked_prefill.py
tools/test_profile_graph_resident_greedy_tail.py
tools/test_graph_resident_greedy_tail_gate.py
tools/test_graph_resident_greedy_tail_verify.py
```

Launch only after a second strict-clean check confirms the same GPU UUID.
The local controller must poll completion, run producer and independent
verification remotely, download every manifest-listed file including
sidecars, and rerun the independent verifier locally.

- [ ] **Step 4: Run GREEN and syntax checks**

```bash
python3 tools/test_run_graph_resident_greedy_tail_remote.py
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-graph-tail-controller-pycache \
  python3 -m py_compile \
  tools/run_graph_resident_greedy_tail_remote.py \
  tools/test_run_graph_resident_greedy_tail_remote.py
```

- [ ] **Step 5: Commit and push**

```bash
git add -- \
  tools/run_graph_resident_greedy_tail_remote.py \
  tools/test_run_graph_resident_greedy_tail_remote.py
git -c core.hooksPath=/dev/null commit \
  -m "test(perf): add graph greedy tail remote gate" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

### Task 7: Local regression and Qwen3-0.6B Stage 1

**Files:**

- Create: `artifacts/graph_resident_greedy_tail/20260822-qwen3-06b-graph-greedy-tail-r1/`

- [ ] **Step 1: Run the complete local regression**

```bash
python3 tools/test_graph_resident_greedy_tail.py
python3 tools/test_greedy_sampling_fast_path.py
python3 tools/test_model_runner_spec_verify.py
python3 tools/test_decode_metadata_landing.py
python3 tools/test_chunked_prefill.py
python3 tools/test_multi_sequence_cuda_graph_gate.py
python3 tools/test_profile_graph_resident_greedy_tail.py
python3 tools/test_graph_resident_greedy_tail_gate.py
python3 tools/test_graph_resident_greedy_tail_verify.py
python3 tools/test_run_graph_resident_greedy_tail_remote.py
python3 tools/test_source_audit.py
git diff --check
```

- [ ] **Step 2: Launch one immutable Stage-1 run**

Use a fresh tag that has never been used locally or remotely:

```bash
KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian \
TINYLLMFORGE_SSH_CONTROL_PATH=/tmp/ssh-sitian-10.232.195.203 \
python3 tools/run_graph_resident_greedy_tail_remote.py \
  --run-tag 20260822-qwen3-06b-graph-greedy-tail-r1 \
  --model-tier qwen3-0.6b \
  --source-commit "$(git rev-parse HEAD)"
```

The controller must fail fast before remote mutation if Kerberos TTL is
below 5,400 seconds. It may wait for one strict-clean GPU and launch
automatically when admission passes.

- [ ] **Step 3: Reconstruct locally**

```bash
PYTHONPATH=. python3 tools/graph_resident_greedy_tail_gate.py \
  --run-dir \
  artifacts/graph_resident_greedy_tail/\
20260822-qwen3-06b-graph-greedy-tail-r1/primary
PYTHONPATH=. python3 tools/graph_resident_greedy_tail_verify.py \
  --run-dir \
  artifacts/graph_resident_greedy_tail/\
20260822-qwen3-06b-graph-greedy-tail-r1/primary
```

Producer and verifier must agree on classification, comparison digest, and
manifest digest.

- [ ] **Step 4: Apply the promotion boundary**

On `GO_GRAPH_RESIDENT_GREEDY_TAIL`, keep the proven scope explicit and run a
fresh confirmation before default enablement or Qwen3-8B. On any NO-GO,
leave the flag default-disabled, preserve the complete negative result, and
do not run Qwen3-8B.

### Task 8: Audit, handoff, final verification, and push

**Files:**

- Modify: `AGENT_HANDOFF_STATE.md`
- Modify: `docs/superpowers/audits/2026-08-16-phase1-completion-audit.md`

- [ ] **Step 1: Append exact evidence at EOF**

Record source and documentation commits, immutable tag, local/remote paths,
GPU UUID and admission samples, all artifact hashes, producer and independent
classifications, token/text and logit metrics, per-bucket median/P95/P99
TPOT, TTFT, E2E, throughput, CUDA memory, capture duration, retained bytes,
avoided work, final token D2H, and Qwen3-8B eligibility.

- [ ] **Step 2: Build the prompt-to-artifact checklist**

Map every design requirement, plan task, file, test, command, gate field,
sidecar, manifest entry, remote assertion, and claim boundary to inspected
evidence. Mark every uncertainty incomplete.

- [ ] **Step 3: Run final verification**

```bash
python3 tools/test_graph_resident_greedy_tail.py
python3 tools/test_greedy_sampling_fast_path.py
python3 tools/test_model_runner_spec_verify.py
python3 tools/test_decode_metadata_landing.py
python3 tools/test_chunked_prefill.py
python3 tools/test_multi_sequence_cuda_graph_gate.py
python3 tools/test_profile_graph_resident_greedy_tail.py
python3 tools/test_graph_resident_greedy_tail_gate.py
python3 tools/test_graph_resident_greedy_tail_verify.py
python3 tools/test_run_graph_resident_greedy_tail_remote.py
python3 tools/test_source_audit.py
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-graph-tail-final-pycache \
  python3 -m py_compile \
  tinyvllm/config.py \
  tinyvllm/engine/model_runner.py \
  tinyvllm/engine/graph_resident_greedy_tail.py \
  tools/profile_graph_resident_greedy_tail.py \
  tools/graph_resident_greedy_tail_gate.py \
  tools/graph_resident_greedy_tail_verify.py \
  tools/run_graph_resident_greedy_tail_remote.py
git diff --check
```

- [ ] **Step 4: Commit exact documentation paths and push**

```bash
git add -- \
  AGENT_HANDOFF_STATE.md \
  docs/superpowers/audits/2026-08-16-phase1-completion-audit.md
git -c core.hooksPath=/dev/null commit \
  -m "docs(perf): record graph greedy tail evidence" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

- [ ] **Step 5: Completion audit**

Verify all of these against current artifacts:

1. independently motivated optimization with a model-agnostic mechanism;
2. default-disabled implementation and fail-closed eligibility;
3. no fallback after transformer replay or current-step KV mutation;
4. exact outputs and bounded float32 logits evidence;
5. Qwen3-0.6B three-arm performance evidence;
6. benefit and cost reported together;
7. producer and independent reconstruction agreement;
8. immutable local and remote artifacts;
9. EOF audit/handoff, exact commits, and pushed remote HEAD;
10. no Qwen3-8B claim without a verified Stage-1 GO.

Do not claim completion until every item maps to fresh evidence and local
HEAD equals `origin/feat/kv-sparse-attention`.
