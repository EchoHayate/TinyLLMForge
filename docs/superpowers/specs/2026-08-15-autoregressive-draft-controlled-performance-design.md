# Autoregressive Draft Controlled Performance Design

## Scope

Measure the already-established Qwen3 target plus independent Qwen3 draft
runtime without changing the TP1 or TP4 correctness producers. The first
campaign is a narrow TP4 direct-Proposal-KV performance pilot on the same
checkpoints and GPUs as the passing TP4 authority.

This campaign is not a Phase 1 promotion gate. It produces synchronized
latency, throughput, memory, acceptance, and Proposal-KV movement evidence
with exact output parity. Long-context, real Proposal-KV offload, a second
model structure, and statistical significance remain separate requirements.

## Considered Approaches

### 1. Extend the TP4 correctness producer

Rejected. The correctness producer intentionally records
`performance_pass_criterion=false`, performs only one run, and has no warmup
or synchronized per-token timing. Adding benchmark behavior would weaken its
small, source-bound correctness contract.

### 2. Time the existing correctness command externally

Rejected. Shell wall time includes model load, distributed initialization,
source verification, and cleanup. It cannot provide TTFT or per-request TPOT
and would repeat the cold-start distortion already visible in the TP1 result.

### 3. Add a learned-drafter performance producer using existing measurement
primitives

Selected. Reuse `build_run_metrics()` and `aggregate_measurements()` from
`tools/speculative_runtime_performance_gate.py`, use the passing TP4 adapter
to create target and learned engines, and measure only post-load request
execution after warmup. Keep the new artifact and verifier independent from
the correctness authorities.

## Architecture

### Worker

`tools/autoregressive_draft_performance_worker.py` owns one isolated policy
cell: `target` or `learned`, batch 1 or batch 4.

For each cell it:

1. Builds deterministic prompt token rows from the target tokenizer.
2. Creates one TP4 engine using the existing Qwen3 TP4 adapter.
3. Executes one unmeasured warmup.
4. Executes three measured runs.
5. Synchronizes distributed request completion through the synchronous Engine
   step boundary.
6. Records token emission timestamps from
   `last_step_observation["new_completion_tokens_by_seq"]`.
7. Resets and reads distributed CUDA peak-memory snapshots around each run.
8. Deltas learned Proposal-KV allocator counters from authority snapshots.
9. Records proposal/accepted token counts and exact output IDs.
10. Exits the Engine and emits one source-bound worker JSON.

### Gate

`tools/autoregressive_draft_performance_gate.py` launches four isolated worker
cells:

- target batch 1;
- learned batch 1;
- target batch 4;
- learned batch 4.

It validates three measured runs per cell, exact target/learned output parity
for every repeat, checkpoint/tokenizer identity, synchronized timing
evidence, distributed memory rows, acceptance evidence, and monotonic
Proposal-KV counters.

It derives median and raw-distribution comparisons for:

- TTFT;
- TPOT;
- end-to-end latency;
- output-token throughput;
- peak allocated and reserved CUDA memory;
- Proposal-KV H2D/D2H bytes;
- proposal and accepted token counts.

The classification remains `PILOT_ONLY` regardless of direction. A positive
direction cannot be promoted without 4K/16K/32K, real Proposal-KV movement,
and a second model structure.

### Verifier

`tools/verify_autoregressive_draft_performance_gate.py` recomputes all derived
aggregates and direction fields from raw worker rows, checks source hashes,
and fails closed on drift or missing raw evidence.

### Remote Runner

`tools/run_autoregressive_draft_performance_gate_remote.sh` uses
`sitian@10.232.195.203`, the existing Qwen3 target/draft checkpoints, GPUs
3/4/6/7, fixed free ports, a frozen source bundle, a hard timeout, and local
plus remote verifier receipts. It never terminates unrelated processes.

## Initial Workload

- TP: 4
- allocator: direct Proposal-KV
- prompt count: 4
- prompt length: 256 tokens for the first pilot
- output length: 16 greedy tokens
- batch sizes: 1 and 4
- warmup runs: 1
- measured runs: 3
- max proposal tokens: 4
- exact output parity: required for every measured repeat

The 256/16 pilot keeps runtime bounded while validating the measurement path.
It cannot satisfy the 4K promotion requirement. A later 4K campaign may reuse
the same producer after the pilot verifier passes.

## Failure Handling

- Any worker nonzero exit fails the parent gate.
- Any missing token event, finish timestamp, rank memory row, or learned
  acceptance row fails validation.
- Any target/learned output mismatch fails validation.
- Any source hash mismatch fails verification.
- Missing Proposal-KV movement is recorded as zero and prevents an offload
  claim; it is not synthesized.
- GPU inventory is captured before and after. Unrelated processes are never
  killed.

## Test Strategy

Dependency-light tests cover:

- synchronized TTFT/TPOT/throughput calculation;
- Proposal-KV counter deltas;
- memory aggregation;
- warmup exclusion and exactly three measured runs;
- four-cell parent orchestration;
- exact parity and acceptance validation;
- aggregate recomputation and tamper rejection;
- source-hash verifier rejection;
- remote runner command, host, GPU, timeout, and dual-verifier contract.

The real-GPU pilot runs only after all local tests and remote dependency-light
tests pass.

## Claim Boundary

Even a faster pilot remains `PILOT_ONLY`. It does not establish:

- 4K, 16K, or 32K performance;
- real Proposal-KV offload benefit;
- a second model structure;
- production scheduler or queueing behavior;
- CUDA Graph benefit;
- statistically significant performance improvement;
- Phase 1 promotion.
