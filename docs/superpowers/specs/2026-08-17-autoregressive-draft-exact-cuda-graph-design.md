# Autoregressive Draft Exact-Shape CUDA Graph Design

**Date:** 2026-08-17
**Status:** Approved for implementation

## Goal

Reduce the TP4 batch-4 learned Qwen3 proposal-forward bottleneck by capturing
the complete fixed Q4 proposal family as one exact-shape CUDA Graph:

```text
draft forward step 0
  -> rank-0 greedy argmax
  -> TP broadcast
  -> draft forward step 1
  -> rank-0 greedy argmax
  -> TP broadcast
  -> draft forward step 2
  -> rank-0 greedy argmax
  -> TP broadcast
  -> one final host token readback
```

The first implementation slice is deliberately limited to:

```text
independent Qwen3 draft model
TP = 4
exact batch size = 4
exact Q = 4
temperature = 0 / exact greedy
dense direct Proposal-KV allocator
KV offload disabled
```

The feature is default-off. Unsupported families continue through the current
eager path without padding, rounding, or grouping identities together.

## Evidence and Claim Boundary

The retained TP4 batch-4 diagnostic reports:

- proposal-forward is `53.4176%` of median end-to-end time;
- backend submission, selection collective, and token readback account for
  `94.45%` of proposal-forward;
- decode authority and metadata/residual account for only about `5.8%`.

This supports graphing the fixed proposal chain as the first optimization
candidate. It does not prove that CUDA launch overhead is the sole cause, that
NCCL graph capture will be stable on the production topology, or that the
graph will improve end-to-end performance. Those claims require the controlled
before/after gate defined below.

## Non-Goals

This slice does not:

- enable TP1, TP2, or any topology other than TP4;
- support batch sizes other than exactly four;
- support Q1, Q2, Q3, or Q greater than four;
- support Proposal-KV offload or blockwise residency;
- support sampling, temperature, or stochastic token selection;
- pad shorter contexts or proposal families into the Q4/B4 identity;
- change target verification, accepted-prefix commit, rejected-suffix abort,
  or target-KV ownership;
- retry eagerly after graph replay has started;
- claim performance improvement before a source-bound controlled result;
- change Phase-1 classification by itself.

## Considered Approaches

### Optimize authority or lifecycle metadata first

Rejected as the first target. The measured authority and metadata/residual
fractions are too small to explain the current regression.

### Delay only the per-step token readback

This is lower risk, but it still leaves three Python model submissions and
three host-issued TP collectives. It is retained as the fallback optimization
if NCCL graph capture is unsupported or unstable.

### Capture the complete exact Q4/B4 proposal family

Selected. The graph keeps selected tokens on the GPU between proposal steps
and removes the repeated Python submission/readback chain. Exact identity,
private scratch capture, fail-closed replay, and default-off admission keep
the semantic risk bounded.

## Architecture

### Policy runner

`tinyvllm/engine/autoregressive_draft_graph.py` owns:

- immutable `AutoregressiveDraftGraphIdentity`;
- exact allowlists;
- successful-eager observation counts;
- capture byte/time budgets;
- ready entries and permanent quarantine;
- pre-replay eager fallback;
- post-replay fail-closed behavior and no duplicate eager retry;
- graph counters and a tensor-free authority summary.

Its public operation is:

```python
run(*, exact_q: int, rows: tuple, eager)
```

Cold identities run eagerly. Capture admission occurs only after the eager
call returns successfully. Ready identities replay before any live result is
materialized.

### Executor integration

`AutoregressiveDraftProposalExecutor` receives an optional graph runner.
Exact-Q groups use:

```text
Q1 or no graph runner -> existing eager group
Q2/Q3/unsupported Q4 -> graph runner returns eager result
ready Q4/B4 identity -> graph replay result
```

Eager and graph paths both return unregistered `DraftProposal` objects backed
by live `ProposalKVCache` transactions. One shared executor helper performs:

1. proposal-materialized logical-authority assertion;
2. lifecycle batch registration;
3. transaction-to-Q and transaction-to-token publication;
4. proposal-forward materialize/register timing.

This keeps accepted-prefix commit and rejected-suffix rollback authoritative
in the existing lifecycle coordinator.

### CUDA backend

`tinyvllm/engine/qwen3_draft_cuda_graph_backend.py` owns:

```python
estimate_static_bytes(identity, rows) -> int
capture(identity, rows, eager, scratch_lease) -> GraphEntry
replay(entry, rows) -> tuple[DraftProposal, ...]
```

The backend owns graph-private CUDA tensors and a capture stream. It invokes
the loaded Qwen3 draft model directly under static decode contexts. Rank zero
computes greedy token IDs; every step broadcasts a contiguous `int64[B]`
token tensor to the other three ranks inside the captured graph.

### Scratch owner

Capture must not consume or mutate live proposal transactions. A dedicated
scratch owner:

1. creates an owner-private `ProposalKVCache` sharing only the production
   physical Qwen3 KV store;
2. maps source rows to positive synthetic sequence IDs;
3. snapshots committed source entries into scratch sequence ownership without
   publishing scratch transactions to the executor;
4. reserves exactly three staged entries per row;
5. returns capture rows bound to scratch transactions;
6. aborts every scratch transaction and releases every synthetic sequence
   after capture;
7. treats rollback failure as a hard capture-lifecycle failure.

## Exact Identity

Every graph identity contains:

```text
exact_q
exact_batch_size
tensor_parallel_size
tensor_parallel_rank
device_index
compute_dtype
backend_identity
model_fingerprint
tokenizer_fingerprint
local_query_heads
local_kv_heads
kv_block_table_width
proposal_kv_capacity
blockwise_offload
```

The first slice admits only:

```text
exact_q == 4
exact_batch_size == 4
tensor_parallel_size == 4
blockwise_offload is false
```

No identity field is rounded. Each rank has a distinct identity because its
model shard, local head geometry, and device are distinct. All ranks must
independently admit the same logical Q4/B4 family before replay.

## Static Tensor Layout

For `B=4`, `Q=4`, and block-table width `W`, one entry owns:

```text
first_tokens:       int64 [B]
current_tokens:     int64 [B]
proposal_tokens:    int64 [B, Q]
positions:          int64 [Q - 1, B]
slot_mapping:       int32 [Q - 1, B]
context_lens:       int32 [Q - 1, B]
block_tables:       int32 [Q - 1, B, W]
```

Unused block-table columns contain `-1`. The exact committed and staged slot
IDs are copied into the static buffers before replay. Shapes and dtypes never
change after capture.

The graph performs, for each of three proposal steps:

1. model forward with `current_tokens`, the static position row, slot mapping,
   context lengths, and block tables;
2. rank-0 `argmax(logits, dim=-1)` into the next token buffer;
3. TP broadcast of that token buffer;
4. copy into `proposal_tokens[:, step + 1]`;
5. use the selected token buffer as the next step input.

No `.item()`, `.tolist()`, Python token publication, or per-step authority
snapshot occurs inside this chain.

## Replay State Machine

### Cold path

1. Execute eager proposal generation.
2. If eager fails, do not advance graph admission.
3. Record one successful observation for the exact identity.
4. After the observation threshold, acquire scratch rows and attempt capture.
5. Always roll scratch state back.
6. Return the already-produced eager result.

### Ready path before replay

1. Validate row count, Q, topology, device, dtype, backend/model/tokenizer
   identity, direct allocator, and exact block-table capacity.
2. Begin live Proposal-KV transactions and prepare all three steps of tensor
   metadata.
3. Run TP `graph_pre_replay` convergence before any rank calls
   `graph.replay()`.
4. If any rank reports a pre-replay error, abort all newly opened live
   transactions and execute one ordinary eager attempt on all ranks.

### Ready path after replay starts

1. All ranks call `graph.replay()` exactly once.
2. One final `proposal_tokens.tolist()` synchronizes and reads all four token
   rows.
3. Each rank performs one `graph_replay_complete` convergence carrying the
   same sequence IDs, transaction IDs, exact Q, and token rows.
4. A local or peer failure quarantines the identity, aborts all live
   transactions, and propagates `AutoregressiveDraftGraphReplayError`.
5. The executor must not retry eagerly after replay starts.

## Transactional Semantics

Graph execution preserves the existing contract:

- each row opens one live transaction with exactly three staged entries;
- graph replay writes only those staged entries;
- successful replay marks exactly three entries materialized;
- lifecycle registration remains executor-owned;
- verifier acceptance commits only the accepted staged prefix;
- every rejected staged suffix is aborted;
- any pre-registration failure aborts the full new transaction;
- capture scratch transactions never appear in executor lifecycle maps.

## Configuration

Add default-off controls:

```text
autoregressive_draft_cuda_graphs = false
autoregressive_draft_cuda_graph_q_allowlist = (4,)
autoregressive_draft_cuda_graph_batch_allowlist = (4,)
autoregressive_draft_cuda_graph_min_observations = 2
autoregressive_draft_cuda_graph_max_entries = 4
autoregressive_draft_cuda_graph_max_static_bytes = 64 MiB
autoregressive_draft_cuda_graph_max_reserved_bytes = 512 MiB
autoregressive_draft_cuda_graph_max_total_capture_ns = 5 s
autoregressive_draft_cuda_graph_max_single_capture_ns = 4 s
```

Enabling the feature with TP other than four or with Proposal-KV offload is a
configuration error rather than a silent behavior change.

The single-capture ceiling is calibrated above the observed TP4/B4/Q4
real-checkpoint capture range of 2.741-2.750 seconds while remaining below the
five-second cumulative ceiling. The same diagnostic measured only 8,520,704
retained reserved bytes and 53,408 static bytes per rank, so the existing
memory ceilings remain unchanged.

## Validation Gates

### Local deterministic tests

Tests must establish:

- every identity field changes the SHA;
- no padding or rounding;
- failed eager calls do not advance admission;
- capture occurs only after a successful eager observation;
- scratch rollback occurs on capture success and failure;
- pre-replay failure performs one eager fallback;
- replay-started failure quarantines and never retries eagerly;
- executor eager and fake-graph paths produce identical token rows,
  transaction counts, registration metadata, and finalize commit/abort state;
- TP pre-replay and post-replay convergence stages are called exactly once per
  graph family rather than once per proposal step;
- unsupported Q/B/offload families stay eager;
- authority summaries contain no tensors.

### Real-checkpoint correctness gate

On TP4 GPUs, run paired eager and graph cases with identical:

```text
Qwen3 draft checkpoint
target checkpoint
four prompts
prompt length 256
output length 16
max_proposal_tokens 4
temperature 0
```

The gate requires:

- exact target output token equality;
- exact draft proposal token equality for every proposal call;
- exact accepted-prefix lengths;
- identical committed Proposal-KV logical identities and digests;
- zero active transactions after completion;
- at least one successful capture and replay on every rank;
- no quarantine or eager fallback during measured graph repeats;
- source archive, current-source hashes, dual verifier receipts, and checksum
  manifest.

### Controlled performance gate

Use process-position-balanced paired repeats so eager and graph alternate
first/second position. Require at least two warmups and eight measured pairs.
Each fresh eager or graph worker must also execute exactly one unmeasured
in-process warmup batch before its measured batch. The eager warmup establishes
the same model, allocator, and kernel-warmth lifecycle as graph mode. The graph
warmup must complete the successful eager observation, capture, and at least
one replay before measurement begins.

The measured graph batch is valid steady-state evidence only when, on every
rank:

- cumulative capture attempts and successful captures are unchanged from the
  end of the in-process warmup;
- cumulative replay count increases;
- cumulative quarantine and pre-replay fallback counts do not increase;
- retained graph entry count remains one; and
- cumulative capture duration and retained graph bytes are unchanged.

The gate payload and verifier must retain both the warmup-end and measured-end
counter/resource snapshots. Pair-level warmups remain excluded from the
performance aggregate; they are not a substitute for the same-engine
in-process warmup because every pair member runs in a fresh worker process.
This state transition is canonical gate schema version 2; schema version 1
evidence cannot support a steady-state performance classification.

Record:

- E2E latency and throughput;
- TTFT and TPOT;
- acceptance rate and accepted tokens per target call;
- proposal-forward parent and detail timings;
- graph capture/replay/fallback counters;
- peak allocated and reserved GPU memory;
- per-rank raw rows and environment/GPU snapshots.

The result is classified:

- `GO` only if correctness passes, every rank replays, median batch-4
  throughput improves, median TPOT does not regress, and the paired bootstrap
  confidence interval excludes zero in the favorable direction;
- `NO_GO_PERFORMANCE` if correctness passes but the controlled delta is not
  favorable;
- `NO_GO_CORRECTNESS` for any token, transaction, authority, or verifier
  mismatch;
- `INCONCLUSIVE_ENVIRONMENT` for unstable placement, GPU interference,
  insufficient valid pairs, or source mismatch.

## Promotion Boundary

Even a `GO` result establishes only the TP4/B4/Q4 dense learned-Qwen3 path.
It does not establish TP1, other batch sizes, other proposal lengths, offload,
long context, another draft architecture, or Phase-1 promotion.
