# Exact-Burst Medium-Context Split-K Graph Design

**Date:** 2026-08-24

**Status:** Selected for implementation under the standing approval to continue
performance work without another approval round

**Stage-1 model and hardware:** Qwen3-0.6B, TP1, NVIDIA A100 80GB PCIe,
FlashAttention 2.6.3

**Primary target:** reduce completion-only batch-1 greedy decode TPOT in the
medium-context range without changing output tokens, scheduler ownership, KV
ownership, or behavior outside that range

## Objective

Capture one additional exact-greedy-decode-burst CUDA Graph whose decode
attention calls use `num_splits=12`. Select that graph only when the complete
authorized burst stays in the measured medium-context range. Preserve the
existing `num_splits=0` FlashAttention auto graph as the default and fallback
for every other request.

This is a runtime-data-flow-specific original engineering design. It does not
claim academic novelty. The isolated attention measurements establish only a
mechanism and a promising operating range. The implementation is retained
only if a source-bound full-model paired gate demonstrates useful TPOT benefit
and reports its capture, memory, correctness, TTFT, E2E, and throughput costs.

## Measured Opportunity

The current exact-burst graph is captured through `set_context()` without an
explicit `flash_attn_num_splits`, so every replay uses the default value `0`.
The normal multi-sequence CUDA Graph path already binds its split choice into
the graph identity, but exact burst does not.

A source-bound standalone decode-attention CUDA Graph sweep at commit
`8e129171007cdec777a2638628bad9c08b0a6985` used the Qwen3-0.6B shape:

```text
batch                 1
query heads          16
KV heads              8
head dimension      128
KV page size        256
GPU          A100 80GB PCIe
FlashAttention      2.6.3
```

The complete artifact is:

```text
/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/
  exact-burst-splitk-context-map/probes/
  20260824-exact-burst-splitk-context-map-r1/summary.json
```

For one fixed explicit value, `num_splits=12`, the measured median attention
replay changes versus auto were:

| Context length | Auto median | Split-12 median | Improvement |
|---:|---:|---:|---:|
| 1,537 | 22.090 us | 21.614 us | 2.16% |
| 1,793 | 22.408 us | 21.471 us | 4.18% |
| 2,049 | 23.073 us | 21.607 us | 6.36% |
| 2,305 | 23.557 us | 21.724 us | 7.78% |
| 2,561 | 23.928 us | 21.770 us | 9.02% |
| 3,073 | 25.213 us | 24.333 us | 3.49% |
| 3,585 | 26.957 us | 24.576 us | 8.83% |
| 4,097 | 27.379 us | 24.643 us | 10.00% |

The sampled output differences were small but not bitwise:

```text
maximum absolute difference across listed points: 0.0009765625
per-point mean absolute difference: about 2.1e-5 to 3.5e-5
```

This is expected from a different split-K reduction order. It means the
production gate must require exact generated token IDs and equal sampled-logit
argmax, while applying the existing exact-burst sampled-logit limits rather
than inventing a bitwise-logit claim.

Negative controls matter:

- `num_splits=1` improved the 257-token attention probe by only 1.89%, then
  regressed 2,049 tokens by about 3.19x and 8,193 tokens by about 7.06x;
- the existing FlashAttention 2.6.3 heuristic was effectively neutral versus
  auto at 2,049 and 8,193 tokens;
- fixed `num_splits=10` improved most medium points but regressed the
  2,561-token point by 0.30%;
- split 12 regressed the 1,281-token point by 0.79%, was neutral-to-negative at
  6,145 tokens, and therefore must not replace auto globally.

The prior probe measured roughly 0.914 MB of retained static graph state and
typically about 2 MB of reserved-memory delta for one exact-burst graph
variant. Those figures are planning evidence, not the final cost result.

## Scope

Stage 1 is intentionally narrow:

- exact greedy decode burst only;
- TP1 only, matching the existing exact-burst eligibility contract;
- Qwen3-0.6B and FlashAttention 2.6.3 for the performance claim;
- an explicit default-disabled feature flag;
- one additional production graph and, only while correctness tracing is
  active, one corresponding correctness graph;
- fixed `num_splits=12`;
- selection only when every model step in the authorized burst has an
  attention context length in `[1537, 4097]`; and
- auto graph fallback for all other contexts and all specialized-capture
  failures.

The change must not modify:

- normal eager decode;
- ordinary single-step or multi-sequence CUDA Graph selection;
- prefill or chunked-prefill behavior;
- scheduler burst eligibility or lease contents;
- block allocation, logical/physical KV ownership, or scratch-block count;
- sampling semantics;
- split-phase mailbox ownership;
- continuation correctness rules; or
- the shared FlashAttention 2.6.3 heuristic.

The fixed range and split count are evidence-bound constants, not public
tuning knobs. A later model/GPU/FlashAttention version needs new evidence and
a separate design.

## Considered Approaches

### A. One medium-context split-12 graph plus auto fallback

Capture the existing auto graph and one additional graph with
`flash_attn_num_splits=12`. Select split 12 only when the entire lease remains
inside the measured interval.

Advantages:

- all eight measured medium-context points improve;
- one extra graph keeps capture latency, retained tensors, and memory bounded;
- auto remains available for short, long, unsupported, and failed-capture
  cases;
- selection is outside replay and adds no per-layer host decision; and
- graph identity can make the numerical variant auditable.

Costs and risks:

- one more full-model graph capture at startup;
- one more set of retained graph/static allocations;
- the attention-only gain may dilute below a useful full-model TPOT gain;
- a different reduction order changes low bits of logits; and
- a burst crossing a range boundary must use auto for the complete burst.

This is the selected approach.

### B. Three specialized graphs for split 8, 10, and 12

Use narrower buckets to select the locally fastest measured split.

This raises the attention-only ceiling, but triples specialized capture and
memory cost, enlarges continuation and invalidation state, and overfits a
sparse context sweep. The expected end-to-end increment over one split-12
graph is too small to justify that state surface. Rejected for Stage 1.

### C. Replace auto with one explicit split globally

Capture only one explicit-split graph for all exact bursts.

This minimizes graph count but contradicts measured short- and long-context
regressions. It also removes the safest fallback. Rejected.

## Architecture

### 1. Configuration and fixed policy

Add one boolean:

```text
Config.exact_greedy_decode_burst_medium_split_k: bool = False
```

It requires `exact_greedy_decode_burst=True`. The selected constants live in
the exact-burst runtime module:

```text
MEDIUM_SPLIT_K_NUM_SPLITS = 12
MEDIUM_SPLIT_K_MIN_CONTEXT_LENGTH = 1537
MEDIUM_SPLIT_K_MAX_CONTEXT_LENGTH = 4097
```

A pure selector receives the lease's initial context length and authorized
token count. It returns 12 only when:

```text
initial_context >= 1537
initial_context + authorized_token_count - 1 <= 4097
```

Otherwise it returns 0. Checking the complete burst prevents one graph replay
from crossing into an unmeasured context region.

### 2. Graph identity and capture receipt

`ExactGreedyDecodeBurstGraph.capture()` gains an explicit non-negative
`flash_attn_num_splits` argument, defaulting to 0 for compatibility.

The value is:

- passed to `set_context()` during warmup and graph capture;
- included in the graph identity payload;
- stored in `ExactGreedyDecodeBurstCaptureReceipt`;
- exposed by graph capability/summary evidence; and
- validated as part of the expected graph identity before replay.

Two graphs with identical tensor storage but different split values must have
different graph identities. A receipt that omits the split value is
insufficient evidence.

### 3. ModelRunner graph ownership

The existing attributes continue to own the auto graphs:

```text
exact_greedy_decode_burst_graph
exact_greedy_decode_burst_correctness_graph
```

When the new flag is enabled, add:

```text
exact_greedy_decode_burst_medium_split_k_graph
exact_greedy_decode_burst_medium_split_k_correctness_graph
```

Production initialization captures auto first. It then attempts the
specialized graph. A specialized capture failure records a specific fallback
reason and leaves auto usable; it must not disable exact burst globally.

Correctness graph capture follows the same policy but is invoked only by the
gate's correctness path.

The split-phase mailbox backend remains shared by graph variants because it
owns host/device publication state, not attention kernel scheduling.

### 4. Dispatch and replay

Before capability validation, `_run_exact_greedy_decode_burst()` selects the
graph from immutable lease data:

```text
feature disabled                         -> auto
burst fully within [1537, 4097]          -> split12 if captured
eligible but split12 capture unavailable -> auto
all other contexts                       -> auto
```

The selected graph's own capability and identity are used for replay. There
is no graph switch during a burst.

Continuation state remains graph-owned. Crossing into or out of the
specialized range can therefore cause a cold bind on the newly selected graph
but cannot reuse continuation state from another graph identity.

Invalidation visits every distinct auto/specialized production/correctness
graph exactly once.

### 5. Observability

Capture receipts provide the authoritative mapping:

```text
graph_identity_sha256 -> flash_attn_num_splits
```

Performance and correctness rows must carry the replayed graph identity. The
gate resolves that identity against the receipts and verifies:

- specialized rows used split 12;
- control and out-of-range rows used auto;
- every replay identity belongs to the current process's receipts; and
- no row silently falls back after partial replay.

Summary evidence reports per-variant capture duration, allocated/reserved
delta, retained static bytes, replay count, fallback count, and context
coverage.

## Failure Handling

- Invalid split values fail during capture construction.
- Missing auto capture preserves the existing `capture_unavailable` behavior.
- Missing specialized capture falls back to auto and records
  `medium_split_k_capture_unavailable`.
- A graph identity mismatch fails closed before replay.
- A lease that crosses either range boundary uses auto for all of its tokens.
- Any failure after replay begins follows the existing exact-burst quarantine
  and transactional rules; the new selector does not create a recovery path.
- No failed, interrupted, or partial benchmark tag is reused.

## Testing Strategy

### Unit and contract tests

Tests must prove:

- configuration validation and dependency on exact burst;
- selector behavior below, at, inside, crossing, and above both boundaries;
- auto and split-12 identity hashes differ;
- receipt serialization includes `flash_attn_num_splits`;
- capture passes the selected split into every context installation;
- specialized capture failure preserves the auto graph;
- dispatch chooses the expected production and correctness graph;
- out-of-range and boundary-crossing leases use auto;
- capability and replay use the selected graph's identity;
- continuation invalidation covers all distinct variants; and
- disabled behavior is structurally identical to the current auto-only path.

Tests are written RED before implementation and then made GREEN.

### Source-bound GPU microgate

Before a long canonical run, run a paired full-model Qwen3-0.6B K8 microgate
from an already-pushed source SHA on a strict-clean A100. Compare:

```text
control:   exact burst with auto graph only
candidate: exact burst with medium split-K enabled
```

Use identical model, GPU, prompts, generated-token counts, ordering, warmup,
and sampling. Include at least:

- one short out-of-range control;
- 1,537;
- 2,049;
- 2,561;
- 3,073;
- 3,585;
- 4,097 or the largest legal context whose whole K8 burst remains at or below
  4,097; and
- one long out-of-range control when model length permits.

The microgate proceeds to the canonical gate only if:

- output token IDs and sampled-logit argmax are exact;
- sampled logits satisfy existing exact-burst limits
  (`max_abs <= 0.25`, per-pair `mean_abs <= 0.05`);
- every intended medium row resolves to a split-12 graph identity;
- every out-of-range row resolves to auto;
- aggregate target-range median TPOT improves by at least 1%;
- no target point's median TPOT regresses by more than 2%;
- out-of-range median TPOT does not regress by more than 1%;
- no fallback, quarantine, replay, D2H, or KV lifecycle invariant changes; and
- all benefit and cost fields are present.

Failure produces a specific NO-GO classification and withdraws the runtime
change.

### Canonical paired gate

The canonical gate uses fresh run tags and multiple alternating/reversed-order
pairs. It must reconstruct all aggregates from raw rows and bind:

- source SHA and clean source snapshot;
- model path and model fingerprint;
- GPU UUID/model and software versions;
- workload and prompt digests;
- arm order and repetition;
- graph receipt/identity/split mapping;
- exact-burst forward, replay, D2H, lease, and commit inventories;
- raw TPOT samples plus median/P95/P99;
- TTFT, E2E, and output throughput;
- allocated, reserved, and retained graph memory;
- capture duration; and
- output IDs, sampled logits, and argmax.

Two independent verifiers must reject missing, duplicated, tampered, partial,
or source-mismatched evidence.

## GO / NO-GO Contract

Classify `GO_EXACT_BURST_MEDIUM_SPLIT_K` only if complete canonical evidence
shows:

1. exact output token IDs and sampled-logit argmax for every pair;
2. sampled-logit `max_abs <= 0.25` and per-pair `mean_abs <= 0.05`;
3. complete replay, forward, D2H, lease, commit, and KV lifecycle invariants;
4. split-12 identity coverage for every eligible candidate row;
5. auto identity coverage for every control and out-of-range row;
6. at least 1% aggregate median TPOT improvement in the target range;
7. no target-range bucket median or P95 TPOT regression above 2%;
8. no out-of-range median or P95 TPOT regression above 1%;
9. aggregate TTFT and E2E regression no greater than 2%;
10. throughput regression no greater than 2%;
11. no additional scratch or reserved KV blocks;
12. additional retained static bytes no greater than 8 MiB per production
    graph variant;
13. additional CUDA reserved-memory delta no greater than 64 MiB;
14. additional capture duration no greater than 5 seconds; and
15. complete cost, manifest, and dual-verifier evidence.

Possible final classifications are:

```text
GO_EXACT_BURST_MEDIUM_SPLIT_K
NO_GO_PERFORMANCE
NO_GO_CORRECTNESS
NO_GO_GRAPH_SELECTION
NO_GO_LIFECYCLE
NO_GO_MEMORY
NO_GO_CAPTURE_COST
NO_GO_EVIDENCE_INCOMPLETE
```

For any NO-GO, retain the benchmark artifacts and design record, revert the
runtime implementation, and state the failed threshold. A positive
attention-only probe is never reported as an end-to-end runtime win.

## Benefit and Cost Reporting

Every result must report both sides:

- benefit: target-range TPOT median/P95/P99 and throughput changes;
- collateral behavior: out-of-range TPOT, TTFT, and E2E changes;
- numerical cost: maximum/mean sampled-logit difference and argmax status;
- startup cost: added capture duration;
- memory cost: added allocated/reserved bytes and retained static bytes; and
- complexity cost: one extra graph identity, two extra ModelRunner references
  when correctness tracing is included, and boundary-driven cold binds.

The final claim is limited to the exact measured model, hardware, software,
context range, burst width, and workload. It must not be generalized to all
Qwen models, all A100 workloads, all context lengths, or all FlashAttention
versions.
