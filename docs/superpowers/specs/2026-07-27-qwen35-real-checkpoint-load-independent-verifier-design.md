# Qwen3.5 Real Checkpoint Load Independent Verifier Design

## Status

Approved for inline implementation. This verifier is local-only and does not
authorize SSH, checkpoint payload access, worker implementation, or worker
execution.

## Goal

Implement an independent, dependency-light verifier for a completed Qwen3.5
real-checkpoint-load artifact directory. The verifier must distinguish:

```text
READY
INCOMPLETE
NO_GO
GO
```

without trusting the worker's summary or performing any remote operation.

## Scope And Boundaries

The verifier:

- reads one local artifact directory;
- loads only JSON, JSONL, Markdown, and log files;
- never opens a `.safetensors` payload;
- never imports Torch, Transformers, safetensors, or TinyLLMForge production
  modules;
- never invokes SSH, `nvidia-smi`, or the worker;
- writes only `independent_verification.json` and `report.md`;
- preserves the worker-owned input inventory across repeated verification.

The worker remains an intentional rejection placeholder until a separate live
preflight reaches `READY`.

## Architecture

`tools/verify_qwen35_real_checkpoint_load_gate.py` dynamically loads
`tools/qwen35_real_checkpoint_load_contract.py`, validates the run in ordered
layers, then delegates completed performance classification to
`contract.classify_case_rows(rows)`.

Validation layers are:

1. safe artifact inventory and SHA256 validation;
2. source, model, preflight, and execution-environment provenance;
3. process, GPU, logs, and summary completion;
4. telemetry-to-case-row consistency;
5. exact case coverage, correctness, tile-budget, digest, and handle checks;
6. frozen `GO` versus `NO_GO` performance/resource thresholds.

Any missing, malformed, contradictory, or unverifiable evidence is
`INCOMPLETE`. `NO_GO` is reserved for a complete and correct run that misses
the frozen performance/resource threshold. `GO` is reserved for a complete
and correct run that meets it.

## Canonical Input Inventory

The manifest lists every worker-owned input artifact except `manifest.json`
itself:

```text
source_manifest.json
preflight.json
environment.json
model_manifest.json
processes.json
gpu_processes.json
case_rows.jsonl
telemetry.jsonl
summary.json
stdout/worker.log
stderr/worker.log
```

Each manifest entry contains:

```text
path
size
sha256
```

The verifier rejects duplicate, absolute, parent-traversing, absent, unlisted,
or hash/size-mismatched inputs. Its two output files are excluded from input
inventory comparison so repeated `--write-report` runs are idempotent.

## Provenance Contract

### Source

`source_manifest.json` must report:

- `clean: true`;
- a 40-character immutable source commit;
- non-empty `local_file_sha256` and `remote_file_sha256` maps;
- exact equality between local and remote maps;
- a `source_tree_sha256` matching `manifest.json` and `preflight.json`.

Every source digest must be lowercase SHA256. The verifier does not inspect the
current checkout and does not accept mutable working-tree state as evidence.

### Model

`model_manifest.json` must exactly match the frozen:

```text
repository: Qwen/Qwen3.5-2B
resolved_revision: 15852e8c16360a2fea060d615a32b45270f8a8fc
trust_remote_code: false
manifest SHA256:
  3e650a908234771c3cf1ac4e20c4d38fe69982efedaf4a3e631ad0b14aad7dd0
config SHA256:
  ed1c1723241f23f7f4e23430759cbd7dcfb4103cbdfe052bfe7626b57c2615b4
index SHA256:
  aca8afed9da75b0f050b408d270766fd77627f1af401e240f61c3b47d0db02f9
```

The model file map, shard filename, shard size, local path, and remote path
must equal the approved contract. The verifier hashes only the local
`model_manifest.json` artifact, never checkpoint files.

### Preflight

The embedded `preflight.json` must pass
`contract.validate_preflight(record)` and have status `READY`. Its source tree,
source file maps, model identities, empty CUDA visibility, zero payload reads,
and empty GPU process list must agree with the other artifacts.

### Environment

`environment.json` must bind execution to:

```text
remote_target: sitian@10.232.195.203
remote_python: /data00/home/sitian/sitian-workspace01/tllm/env/bin/python
user: sitian
CUDA_VISIBLE_DEVICES: ""
cuda_initialized: false
cuda_allocated_bytes: 0
```

It must also repeat the frozen model repository/revision and the preflight
source-tree digest.

## Process And GPU Evidence

`processes.json` contains exactly one worker process record. The record must
show:

- role `worker`;
- attempted and started;
- exited normally with return code zero;
- no signal or timeout;
- empty CUDA visibility;
- CUDA never initialized and zero CUDA allocation;
- stdout/stderr paths equal to the canonical worker log paths.

`gpu_processes.json` contains `before` and `after` snapshots. Both snapshots
must be empty lists. Any GPU occupancy before or after the worker makes the run
`INCOMPLETE`.

`stdout/worker.log` and `stderr/worker.log` must exist. The verifier permits an
empty stderr log only when the process record is successful.

## Telemetry And Case Rows

The frozen measured order is:

```text
8, 16, 16, 8, 8, 16 MiB
```

There are exactly six measured case rows and six telemetry rows. Every
telemetry row is keyed by the same `case_id`, `order_index`, `repeat_index`,
and `budget_bytes` as its case row and repeats these measured fields exactly:

```text
wall_seconds
user_cpu_seconds
system_cpu_seconds
minor_faults
major_faults
vmrss_bytes
vmhwm_bytes
voluntary_context_switches
involuntary_context_switches
```

Telemetry values must be finite and non-negative; wall and CPU durations must
meet the stricter positive-duration checks already frozen in the contract.
Duplicate or missing case IDs, extra telemetry rows, or metric disagreement is
`INCOMPLETE`.

Each case row must pass the contract's exact checks:

- TP size two and rank zero;
- correct budget/order/repeat;
- 320 assigned bindings and 320 source tensors;
- destination bytes `1,881,935,712`;
- peak tile bytes at or below the selected budget;
- actual and expected assignment digests equal;
- every shard handle closed;
- CUDA uninitialized and allocation zero;
- complete process and `/proc` telemetry;
- return code zero.

## Summary Consistency

`summary.json` is evidence, not authority. It must repeat:

- schema version;
- status `COMPLETE`;
- six case rows and six telemetry rows;
- successful worker process count one;
- zero GPU process count before and after;
- assigned bindings, source tensors, and destination bytes;
- all handles closed;
- no CUDA initialization/allocation;
- the same classification and aggregate metrics independently returned by
  `contract.classify_case_rows`.

Any disagreement is `INCOMPLETE`.

## Classification

The verifier first proves completeness and correctness. It then calls:

```python
contract.classify_case_rows(case_rows)
```

Results map directly:

- contract `GO` -> verifier `GO`;
- contract `NO_GO` -> verifier `NO_GO`;
- contract `INCOMPLETE` -> verifier `INCOMPLETE`.

`READY` is emitted only when all non-performance evidence is present and valid
but no measured case rows have been produced yet. A canonical completed run
with partial case or telemetry coverage is `INCOMPLETE`, not `READY`.

## Outputs

With `--write-report`, outputs are atomically replaced:

```text
independent_verification.json
report.md
```

The JSON includes classification, expected/observed case counts, guard
booleans, reasons, frozen aggregate metrics, and a strict claim boundary.

The Markdown report states the classification and makes no inference-speed,
cache-size, memory-reduction, compression, or quality claim beyond the
verified checkpoint-load comparison.

The CLI prints the JSON result and exits zero for all four classifications.
Invalid CLI arguments remain argparse errors.

## Test Strategy

A dependency-light test creates complete synthetic artifact directories for:

- a correct `GO` fixture;
- a correct but sub-threshold `NO_GO` fixture;
- repeated report generation;
- provenance, inventory, process, GPU, telemetry, case, digest, handle,
  summary, and log tampering.

Each tamper case refreshes the manifest entry for the changed file so the test
reaches the intended semantic guard rather than failing only at the inventory
hash layer.

## Claim Boundary

Synthetic verifier fixtures prove verifier behavior only. They do not prove a
real checkpoint was opened, a native Qwen3.5 forward pass works, accuracy is
preserved, inference is faster, or production cache/memory is reduced.
