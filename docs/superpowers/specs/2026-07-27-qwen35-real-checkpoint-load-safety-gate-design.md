# Qwen3.5 Real Checkpoint Load Safety Gate Design

## Status

Approved for implementation and local dry-run validation only. This design
does not authorize SSH execution or opening the real checkpoint payload in the
current session.

## Goal

Create a fail-closed, source-bound remote gate for the first real Qwen3.5
checkpoint payload comparison:

```text
TP=2 rank0
8 MiB tile budget
16 MiB tile budget
```

The future remote run must measure load time and host-memory/page-fault
telemetry while proving exact assignment correctness, strict GPU non-use, and
complete artifact provenance.

## Fixed Remote Identity

```text
target: sitian@10.232.195.203
SSH control path: /tmp/ssh-sitian-10.232.195.203
Python: /data00/home/sitian/sitian-workspace01/tllm/env/bin/python
GPU visibility: CUDA_VISIBLE_DEVICES=""
forbidden GPU compute processes: any unrelated process on GPU0
model repository: Qwen/Qwen3.5-2B
model revision: 15852e8c16360a2fea060d615a32b45270f8a8fc
```

The worker is CPU-only. GPU0 is inspected because the approved host is shared,
but the worker must not initialize CUDA or allocate GPU memory.

## Gate Layers

### Contract

Create:

```text
tools/qwen35_real_checkpoint_load_contract.py
```

It freezes:

- schema version;
- exact two-case matrix;
- remote/model identities;
- required artifacts;
- allowed classification values;
- result and telemetry validation;
- source/config/model identity checks;
- exact checksum/assignment requirements.

### Worker

Create:

```text
tools/qwen35_real_checkpoint_load_worker.py
```

The future `run` mode:

1. validates immutable checkpoint config/index/header and all shard paths;
2. builds one fresh TP=2 rank0 CPU candidate for each case;
3. uses the existing policy/tiled loading path with an exact forced budget;
4. samples `/proc/self/status`, `/proc/self/stat`, `resource.getrusage()`, and
   monotonic time before/after each phase;
5. records wall time, user/system CPU time, minor/major faults, voluntary and
   involuntary context switches, `VmRSS`, `VmHWM`, and process exit status;
6. proves every binding destination matches a deterministic independent
   digest/accounting contract;
7. discards each candidate before the next case;
8. runs cases in alternating order across repeats to reduce order bias.

The worker must not call `.cuda()`, initialize CUDA, run forward, or publish an
owner.

### Independent Verifier

Create:

```text
tools/verify_qwen35_real_checkpoint_load_gate.py
```

It must reject:

- missing or extra cases;
- source/model/config mismatch;
- dirty or mismatched staged source;
- non-empty `CUDA_VISIBLE_DEVICES`;
- any CUDA initialization/allocation;
- unrelated GPU compute occupancy;
- missing/invalid telemetry;
- non-exact assignment/digest evidence;
- unbalanced shard handles;
- budget or tile-plan mismatch;
- any failed process or incomplete artifact.

The verifier reports only:

```text
READY
INCOMPLETE
NO_GO
GO
```

`GO` requires exact correctness for both budgets and a stable, predeclared
resource/performance comparison. The first run may legitimately be
`INCOMPLETE` or `NO_GO`.

Implementation status update:

- local `verify-only` invokes only the independent local verifier;
- `download-only` is manifest-first, fixed-inventory, size/SHA256 checked,
  chunk-persisted, atomically published, then independently verified locally;
- local `authorization-only` requires a `READY` preflight, exact current
  owned-source hashes, matching branch/commit, and tracked clean owned source;
- neither mode launches the worker or opens checkpoint payloads;
- `run` remains fail-closed until a fresh approved preflight reaches `READY`.

### Remote Runner

Create:

```text
tools/run_qwen35_real_checkpoint_load_gate_remote.py
```

Modes:

```text
preflight
run
download-only
verify-only
dry-run
```

The runner:

- owns an exact source-file list;
- requires those files to be clean before remote execution;
- stages an immutable tar snapshot and verifies remote SHA256 values;
- creates unique non-destructive run directories;
- never uses `rm -rf`, `pkill`, `killall`, `git reset`, or `git clean`;
- runs read-only preflight before any payload access;
- requires a matching `READY` preflight artifact for `run`;
- records process and GPU occupancy attempts;
- downloads artifacts atomically/chunked;
- invokes the local independent verifier.

## Read-Only Preflight

Preflight opens no `.safetensors` payload. It records:

- host/user/hostname;
- Python, PyTorch, safetensors, Transformers versions;
- model revision and snapshot path;
- config/index/header SHA256 identities;
- every shard path, declared byte count, file size, inode, and device;
- free disk and memory;
- `/proc/meminfo`;
- GPU0 name/UUID/driver/process list;
- `CUDA_VISIBLE_DEVICES`;
- source manifest and remote hashes;
- filesystem type/mount for model and run roots;
- whether Linux `/proc` telemetry fields are available.

Preflight is `READY` only when all identities match, all files exist, no
unrelated GPU compute process is present, worker dependencies exist, and the
run root has sufficient free space for artifacts.

## Future Run Matrix

```text
warm-up: 1 per budget
measured repeats: 3 per budget
order: 8,16,16,8,8,16 MiB
TP: size=2 rank=0
```

No OS cache dropping is allowed. The gate labels runs as observed-cache-state
and records per-repeat page faults and file residency telemetry. It does not
claim cold-cache measurements.

## Artifacts

Required:

```text
manifest.json
source_manifest.json
preflight.json
environment.json
model_manifest.json
processes.json
gpu_processes.json
case_rows.jsonl
telemetry.jsonl
summary.json
independent_verification.json
report.md
stdout/worker.log
stderr/worker.log
```

## Promotion Rules

Correctness is non-negotiable:

- all 320 bindings assigned;
- exact destination byte accounting;
- deterministic digest equality for every destination;
- no missing/extra sources;
- no CUDA use;
- all handles closed.

Performance/resource promotion is intentionally conservative:

- compare medians across three measured repeats;
- report 8→16 MiB wall-time, `VmHWM`, minor faults, and major faults;
- `GO` only when 16 MiB has non-negative correctness/resource safety and a
  stable wall-time improvement exceeding 5%;
- `NO_GO` when correctness passes but improvement is below 5%, unstable, or
  resource regression exceeds 16 MiB;
- `INCOMPLETE` for any provenance, process, telemetry, or correctness gap.

These thresholds are frozen before the first payload read.

## Current-Session Deliverable

Implement contract validation, runner command construction, preflight/result
schemas, dry-run artifact planning, and tests. Do not SSH, inspect the remote
host live, or open model payloads in this session.

