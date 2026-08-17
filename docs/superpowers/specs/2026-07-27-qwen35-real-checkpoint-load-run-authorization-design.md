# Qwen3.5 Real Checkpoint Load Run Authorization Design

## Status

Approved for local-only implementation. This gate does not implement or launch
the checkpoint worker and does not authorize payload access while the live
preflight is `INCOMPLETE`.

## Goal

Add an explicit fail-closed decision between read-only preflight and any future
real checkpoint-load worker implementation:

```text
BLOCKED
AUTHORIZED
```

The gate must prove that the preflight is `READY`, its staged source exactly
matches the current frozen owned-source files, and the owned source is clean
and immutable. A hash-matched but dirty working tree is not sufficient.

## Why This Gate Is Required

The read-only preflight currently proves:

- exact local/remote source-file hash equality;
- approved model/config/index/shard identity;
- CPU-only execution environment;
- payload-zero behavior;
- GPU0 occupancy at observation time.

It does not independently prove that the owned source files are committed and
clean. The safety design requires clean source before real execution, so a
separate authorization decision must reject:

- any `INCOMPLETE` preflight;
- stale source hashes after local edits;
- a source manifest whose local/remote/preflight maps disagree;
- an owned source file with staged, unstaged, or untracked changes;
- branch or commit mismatch;
- missing or malformed provenance.

## Inputs

The mode reads only:

```text
experiments/qwen35_hybrid_state/<run-tag>/preflight.json
experiments/qwen35_hybrid_state/<run-tag>/source_manifest.json
```

It also computes current local hashes and queries local Git metadata for the
five frozen `OWNED_SOURCE_FILES`. It performs no SSH and never reads model
payloads.

## Clean-Source Definition

Every owned source path must:

- be tracked by Git;
- have no staged diff;
- have no unstaged diff;
- not be untracked;
- match the exact SHA256 map recorded by the source manifest and preflight.

The source manifest branch and commit must equal the current branch and
`HEAD`. Other unrelated working-tree changes do not block this gate.

## Decision Rules

`AUTHORIZED` requires all of:

1. both input files exist and parse;
2. `contract.validate_preflight(preflight)` succeeds;
3. `preflight.status == "READY"`;
4. source manifest schema, target, branch, commit, and source-tree digest are
   valid;
5. local, remote, preflight-local, and preflight-remote hash maps are exactly
   equal;
6. the current owned-source hash map equals the recorded map;
7. every owned source path is Git-tracked and clean;
8. payload-zero, CUDA-disabled, GPU-idle, model identity, and runtime checks
   remain true in the preflight record.

Any failure returns `BLOCKED` with explicit reasons. It never returns a partial
authorization.

## Output

The result contains:

```text
schema_version
decision
run_tag
source_tree_sha256
current_source_tree_sha256
current_branch
current_commit
checks
reasons
worker_implementation_authorized
worker_execution_authorized
claim_boundary
```

For this gate:

- `worker_implementation_authorized` equals the `AUTHORIZED` decision;
- `worker_execution_authorized` is always `false`.

Execution remains a separate future gate because GPU state can change after
preflight and the worker does not yet exist.

## Runner Mode

Add:

```text
authorization-only
```

The mode reads local evidence, performs Git/hash checks, optionally atomically
writes `--output-json`, and performs zero SSH/subprocess worker/payload
actions. Git commands are local provenance reads only.

## Test Strategy

Dependency-light tests cover:

- synthetic `READY` plus exact clean source -> `AUTHORIZED`;
- current `INCOMPLETE` GPU-busy preflight -> `BLOCKED`;
- stale current source hash;
- local/remote/preflight hash mismatch;
- branch and commit mismatch;
- staged, unstaged, and untracked owned-source paths;
- malformed/missing artifacts;
- CLI JSON persistence and no SSH/worker/payload behavior.

## Claim Boundary

`AUTHORIZED` would permit implementing the worker, not launching it. It does
not prove inference speed, cache or memory reduction, compression safety,
quality retention, native forward execution, or checkpoint-load performance.
