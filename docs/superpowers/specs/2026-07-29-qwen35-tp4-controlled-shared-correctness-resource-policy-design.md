# Qwen3.5 TP4 Controlled-Shared Correctness Resource Policy Design

## Objective

Permit the real Qwen3.5 TP4 correctness campaign to run on a node whose GPUs
have known long-running baseline processes, without weakening the existing
strict-exclusive policy or treating shared-resource observations as
performance evidence.

The approved correctness-only GPU set is `2,4,5,6` on
`sitian@10.232.195.203`. Canonical performance benchmarking remains blocked
until four genuinely exclusive GPUs satisfy the existing zero-compute guard.

## Policy Modes

Existing plans retain `strict_exclusive` semantics:

- each selected GPU has at least 24 GiB free;
- no active compute process is allowed;
- the final recheck applies the same rule.

New correctness plans may explicitly select `controlled_shared`:

- exactly four GPU indices and UUIDs are frozen;
- a regular baseline manifest freezes the external compute-process inventory;
- every selected GPU has at least 24 GiB free at baseline and at each guard;
- every observed external process must be in the frozen baseline;
- a frozen baseline process may disappear, but no new external process may
  appear;
- PID reuse is rejected by binding PID, process name, and process start time;
- authority-owned process descendants are allowed only during the guarded
  authority command;
- a final guard runs after the authority exits and again permits only the
  surviving subset of frozen baseline processes;
- resource observations are evidence for correctness execution safety only.

The strict-exclusive schema, command generation, receipt validation, and
benchmark protocol remain unchanged.

## Baseline Manifest

A new remote read-only baseline capture produces canonical JSON with:

```text
schema_version
classification=READY
ssh_target
captured_at
gpu_indices
selected[
  gpu_index
  gpu_uuid
  free_bytes
  compute_processes[
    pid
    process_name
    used_memory_mib
    start_time_ticks
  ]
]
minimum_free_bytes_per_gpu
benchmark_execution_authorized=false
```

The local builder reopens and validates the manifest, binds its SHA256 into
the executor configuration and all three child plans, and copies it into the
new preparation bundle. Symlinks, duplicate GPUs or PIDs, malformed process
identities, insufficient memory, target drift, or unexpected fields are
rejected.

## Guard Execution

The controlled-shared guard queries:

- `nvidia-smi --query-gpu=index,uuid,memory.free`;
- `nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory`;
- `/proc/<pid>/stat` for process start time.

Before the authority starts, the observed process identity set on each GPU
must be a subset of the frozen baseline. During final recheck, authority
descendants have already exited, so the same subset rule applies. The guard
prints canonical JSON containing `resource_policy=controlled_shared`,
baseline SHA256, selected GPUs, and observed processes.

No process is killed, paused, reprioritized, or modified.

## Evidence Binding

The following identities are immutable and propagated through configuration,
plans, authorizations, receipts, campaign preparation, and campaign receipt:

- `resource_policy`;
- baseline manifest path;
- baseline manifest SHA256;
- selected GPU indices and UUIDs;
- minimum free bytes;
- `benchmark_execution_authorized=false`.

Receipt validators reopen the baseline and reject policy, GPU, UUID, SHA,
process, or memory drift. A controlled-shared PASS can authorize the
correctness prerequisite bundle, but cannot authorize a benchmark.

## Failure Semantics

Execution stops before consuming a child authorization when the initial guard
fails. After authorization consumption, any command, authority, final guard,
receipt, or independent verification failure publishes failure evidence and
does not publish PASS.

Resource failures include:

- selected GPU missing or UUID changed;
- free memory below 24 GiB;
- new external PID;
- PID reuse or process-name drift;
- malformed or unavailable process start time;
- baseline manifest drift;
- policy drift.

## Testing

CPU-only tests cover:

- strict-exclusive output remains byte-for-byte stable;
- controlled-shared manifest validation;
- baseline subset acceptance and new-process rejection;
- PID-reuse rejection;
- free-memory and UUID rejection;
- plan, authorization, receipt, preparation, and campaign binding;
- benchmark plans rejecting controlled-shared resources;
- fresh-process verification of the generated READY preparation.

After local gates pass, the real campaign runs strictly serially:

1. TP4 root-logit;
2. cached continuation;
3. Engine correctness;
4. authority adaptation;
5. v2 prerequisite bundle build and verification.

Only real receipts may support correctness claims.
