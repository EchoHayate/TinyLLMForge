# Autoregressive Draft Learned A/A Process-Boundary Design

## Goal

Measure whether two source-identical, configuration-identical learned-policy
epochs launched sequentially in one remote bundle exhibit a reproducible
process-position effect.

This experiment is an A/A control for the r7/r8 learned-policy reversal. It
does not optimize the runtime, establish a host-resource cause, or contribute
directly to Phase-1 promotion.

## Motivation

The source-bound r7/r8 host-semantic comparison produced valid, repeat-aligned
campaigns with opposite target/learned order, but the learned-policy direction
reversed:

```text
learned first E2E median:   6.1944397425 s
learned second E2E median:  9.07202871925 s
relative delta:            -31.7193547971%

classification:
  HOST_ALIGNMENT_INCONCLUSIVE
```

The comparison verifier passed, the six source hashes matched, exact greedy
parity held, each policy had eight measured repeats, and repeat-local host
sampling gaps remained below `0.6 s`. The reversal is therefore not explained
by a missing source receipt, policy artifact, repeat, or host-sample coverage
gate.

Before attempting another host-causal attribution, the learned policy must be
compared against itself across the same process boundary. This separates
same-policy launch/order variance from target-versus-learned policy effects.

## Selected Approach

Create a dedicated learned A/A remote runner that launches two independent
learned-policy process epochs in one bundle:

```text
epoch order:
  learned_a
  learned_b
```

Both epochs invoke the existing worker with:

```text
--policy learned
--batch-size 4
--warmup-runs 2
--measured-runs 8
```

Each measured epoch is preceded by its own isolated learned-policy prime
process:

```text
--policy learned
--batch-size 4
--warmup-runs 2
--measured-runs 1
```

`learned_a` and `learned_b` are artifact identities only. They are not new
runtime policies and must never be passed to the worker as `--policy`.

The runner, diagnostic, verifier, and artifact schema are dedicated to learned
A/A. The existing target/learned timing, instability, and host-semantic
diagnostics retain their current semantics.

## Why a Dedicated Runner

The current runner accepts only:

```text
target,learned
learned,target
```

It also derives artifact paths from the runtime policy name:

```text
workers/learned-b4.json
telemetry/learned-gpu.csv
host-semantic/learned-host.jsonl
```

Allowing `learned,learned` would overwrite the first epoch and would still
feed two semantically distinct positions through target/learned-only
assemblers. Adding a mode switch to the existing runner would couple two
different artifact contracts and make accidental cross-use harder to reject.

A dedicated runner preserves the existing authority chain and gives A/A
artifacts unambiguous identities.

## Rejected Alternatives

### Two Independent Bundles

Running one learned epoch per bundle avoids path collisions, but it adds SSH,
source extraction, preflight, and bundle-start variation between A and B. The
first control should keep both epochs under one bundle-level environment and
one source snapshot.

Two independent bundles remain useful only as the required replication after
a candidate effect is observed.

### One Persistent Learned Process

Running A and B inside one persistent model process would remove the process
boundary under investigation. It could measure request-order drift, but it
cannot answer whether independent process epochs differ by launch position.

### Reuse the Target/Learned Diagnostic

Relabeling one learned epoch as target would violate worker-policy validation,
misstate the experiment in the artifact, and risk treating a structurally
invalid comparison as authority. A/A requires a separate schema and verifier.

## Files and Responsibilities

The implementation plan may refine exact internal function boundaries, but it
must preserve these file-level responsibilities:

```text
tools/run_autoregressive_draft_learned_aa_remote.sh
  package the source snapshot, launch both remote learned epochs, run remote
  verification, download the bundle, run local verification, and build the
  manifest

tools/autoregressive_draft_learned_aa_diagnostic.py
  validate the two worker results and raw telemetry inputs, align samples to
  measured repeats, compute stationarity and A/B deltas, and emit the
  canonical learned A/A artifact

tools/verify_autoregressive_draft_learned_aa_diagnostic.py
  recompute the canonical artifact from its hash-bound raw inputs and source
  files, require structural equality, and emit a verification receipt

tools/test_autoregressive_draft_learned_aa_diagnostic.py
  cover schema validation, exact parity, alignment, stationarity,
  classification, source identity, and tamper rejection

tools/test_autoregressive_draft_instability_telemetry.py
  retain runner source-contract tests if that file remains the established
  home for remote-runner contract assertions
```

The implementation must not modify runtime execution semantics merely to run
this control.

## Runner Interface

The dedicated runner uses the same environment defaults as the current r8
authority runner:

```text
REMOTE_HOST=sitian@10.232.195.203
REMOTE_PYTHON=/data00/home/sitian/miniconda3/envs/py311/bin/python
REMOTE_BASE=/dev/shm/sitian/tllm-qwen35-target-qwen3-draft-20260815
TARGET_MODEL=${REMOTE_BASE}/target-qwen3-1.7b
DRAFT_MODEL=${REMOTE_BASE}/draft
GPU_INDICES=3,4,6,7
```

It supports the existing optional control socket:

```text
--ssh-control-path <existing-control-socket>
```

It must derive `LOCAL_RUN` and `REMOTE_RUN` only after parsing CLI arguments,
must refuse to overwrite either path, and must record all effective values in
`command.txt`.

The epoch order is fixed as:

```text
learned_a,learned_b
```

The first bundle does not expose a user-selectable order because both epochs
are semantically identical. Replication uses a fresh bundle and a separate
replication role, not a misleading policy reversal.

## Remote Data Flow

The remote bundle performs these steps:

1. Record bundle start time, GPU state, source archive, source hashes, command,
   fixed epoch order, and preflight results.
2. Run Python compilation and the focused local test set from the packaged
   source.
3. Launch an isolated learned prime process for `learned_a`.
4. Start script-owned GPU, `/proc`, `vmstat`, `mpstat`, and `pidstat` samplers
   using `learned_a` paths.
5. Launch the measured `learned_a` worker as an independent process.
6. Stop and reap only the samplers owned by the script.
7. Launch an isolated learned prime process for `learned_b`.
8. Start a fresh set of script-owned samplers using `learned_b` paths.
9. Launch the measured `learned_b` worker as an independent process.
10. Stop and reap only the second sampler set.
11. Assemble the canonical learned A/A artifact.
12. Run the independent remote verifier.
13. Record all exit codes, final GPU state, and bundle finish time.
14. Download the bundle, run the independent local verifier, and build and
    verify `manifest.sha256`.

No worker process, sampler process, model state, Python interpreter, or CUDA
context is shared between `learned_a` and `learned_b`.

## Artifact Layout

Every epoch has a unique path:

```text
epoch-order.txt
prime-each-epoch.txt

prime-workers/learned-a-prime-b4.json
prime-workers/learned-b-prime-b4.json
prime-logs/learned-a-prime-b4.log
prime-logs/learned-b-prime-b4.log

workers/learned-a-b4.json
workers/learned-b-b4.json
logs/learned-a-b4.log
logs/learned-b-b4.log

telemetry/learned-a-gpu.csv
telemetry/learned-b-gpu.csv

host-semantic/learned-a-host.jsonl
host-semantic/learned-b-host.jsonl
host-semantic/learned-a-host.stderr.log
host-semantic/learned-b-host.stderr.log

host/learned-a-vmstat.log
host/learned-a-mpstat.log
host/learned-a-pidstat.log
host/learned-b-vmstat.log
host/learned-b-mpstat.log
host/learned-b-pidstat.log

learned-aa.json
verify.learned-aa.remote.json
verify.learned-aa.local.json
manifest.sha256
```

Prime artifacts are evidence that both positions received the same priming
protocol. Prime timings are not included in the A/B performance comparison.

## Canonical Artifact Contract

`learned-aa.json` uses schema version 1 and includes:

```text
schema_version
status
classification
classification_reasons
claim_state
epoch_order
prime_each_epoch
exact_parity
workload_identity
input_files
source_files
epochs
comparison
thresholds
limitations
```

`epoch_order` must equal:

```json
["learned_a", "learned_b"]
```

`claim_state` must always include:

```text
candidate_process_boundary_effect
process_boundary_effect_established
```

For every single-bundle artifact,
`process_boundary_effect_established` must be `false`.

Each `epochs` row includes:

```text
artifact_identity
worker_policy
worker_sha256
gpu_csv_sha256
host_jsonl_sha256
measured_runs
stationarity
coverage
```

`worker_policy` must be `learned` for both epochs. Each epoch must have two
warmups, eight measured repeats, batch size four, temperature zero, and the
same workload and model identity.

Each measured repeat includes:

```text
repeat
outputs
timing
runtime
gpu_summary
host_metrics
coverage
```

The primary timing metrics are:

```text
e2e_s
tpot_s
executor_proposal_forward_ms
```

The artifact also preserves TTFT, output throughput, acceptance, Proposal-KV
capacity and lifecycle counters, GPU memory, and real KV movement counters
already present in the worker result. Missing counters are represented as
missing evidence, not synthesized as zero.

## Input and Source Binding

The canonical artifact hash-binds at least:

```text
prime-workers/learned-a-prime-b4.json
prime-workers/learned-b-prime-b4.json
workers/learned-a-b4.json
workers/learned-b-b4.json
telemetry/learned-a-gpu.csv
telemetry/learned-b-gpu.csv
host-semantic/learned-a-host.jsonl
host-semantic/learned-b-host.jsonl
epoch-order.txt
prime-each-epoch.txt
```

It also binds the implementation sources required to reproduce the artifact,
including the worker, sampler, diagnostic, verifier, and dedicated runner.

The independent verifier reloads those paths relative to the artifact
directory, checks that no path is absolute or contains `..`, checks every
digest, recomputes the artifact, and requires exact structural equality. The
verification receipt is a separate file and is never embedded in the
canonical artifact.

Distinct A/B input digests are required for worker and raw telemetry files.
Source digests must be identical because both epochs come from one packaged
source snapshot.

## Exact Parity and Workload Identity

For every repeat index `0..7`, `learned_a` and `learned_b` must produce exactly
the same output token IDs for every request. The diagnostic rejects the
campaign rather than classifying it if parity fails.

The diagnostic also requires equality for:

```text
target model identity
draft model identity
prompt token IDs
requested output length
batch size
temperature
MAX_PROPOSAL_TOKENS
Proposal-KV capacity derivation inputs
GPU index set
TP world size
```

The Proposal-KV capacity must remain the exact workload-derived upper bound.
The A/A control must not substitute a fixed or oversized capacity.

## Coverage and Stationarity

Host samples are aligned independently to each measured repeat using the
existing repeat-local bracketing semantics:

```text
sample cadence:             0.2 s
maximum repeat-local gap:   0.6 s
host boundary allowance:    0.4 s
```

GPU samples use a `0.6 s` boundary-nearest allowance. Every repeat must contain
at least five samples for each of GPU indices `3,4,6,7` after boundary-nearest
completion.

Stationarity is evaluated independently for each epoch and each primary
timing metric using the existing thresholds:

```text
range over median <= 0.25
first-half/second-half drift <= 0.20
```

Coverage, counter monotonicity, sample ordering, source identity, or exact
parity failures are verifier failures. They are not converted into an
inconclusive performance classification.

## Comparison

For each primary timing metric:

```text
relative_delta = (median_a - median_b) / median_b
absolute_relative_delta = abs(relative_delta)
```

The artifact records medians, absolute differences, relative deltas, sign,
per-repeat values, and stationarity for both epochs.

The comparison also records the same A/B summaries for:

```text
TTFT
output throughput
acceptance
GPU memory
KV H2D bytes
KV D2H bytes
Proposal-KV movement
primary host metrics
```

These secondary metrics provide diagnostic context. They do not independently
establish a process-boundary effect.

## Classification

### `LEARNED_AA_STABLE`

Classify stable when:

- both epochs pass stationarity for all three primary timing metrics; and
- `abs(e2e relative_delta) < 0.10`.

TPOT and proposal-forward direction are still reported. A sub-10% E2E delta
does not establish a meaningful process-position effect even if a secondary
metric exceeds 10%.

### `LEARNED_AA_PROCESS_BOUNDARY_EFFECT`

Classify a candidate process-boundary effect when:

- both epochs pass stationarity for all three primary timing metrics;
- `abs(e2e relative_delta) >= 0.10`;
- TPOT has the same non-zero sign as E2E; and
- proposal-forward has the same non-zero sign as E2E.

Within one bundle this classification means only:

```text
CANDIDATE_PROCESS_BOUNDARY_EFFECT=OBSERVED
PROCESS_BOUNDARY_EFFECT_ESTABLISHED=NO
```

It does not identify process startup, page cache, allocator state, host
pressure, GPU state, or another specific mechanism.

### `LEARNED_AA_INCONCLUSIVE`

Classify inconclusive when structurally valid evidence remains unsuitable for
either stable or candidate-effect classification, including:

- either epoch fails a primary stationarity threshold; or
- E2E exceeds 10% but TPOT or proposal-forward has a conflicting or zero
  direction.

The artifact lists the exact failed conditions.

## Replication Gate

One bundle cannot establish a process-boundary effect.

If the first bundle is `LEARNED_AA_PROCESS_BOUNDARY_EFFECT`, run a second
source-identical bundle under the same fixed A-then-B protocol. The second
bundle assigns replication roles rather than inventing a policy reversal:

```text
bundle 1:
  role: discovery

bundle 2:
  role: replication
```

Establish a process-boundary effect only if:

- both bundles pass independent remote and local verification;
- both classify `LEARNED_AA_PROCESS_BOUNDARY_EFFECT`;
- E2E, TPOT, and proposal-forward have the same A-versus-B sign in both
  bundles; and
- E2E has at least 10% absolute relative delta in both bundles.

If the sign reverses, either bundle is inconclusive, or the replication falls
below 10%, classify the cross-bundle conclusion:

```text
PROCESS_BOUNDARY_EFFECT_NOT_ESTABLISHED
```

A cross-bundle comparison artifact and independent verifier are required
before any established-effect claim. They belong in a follow-up design or an
explicit extension approved during implementation planning; the first
implementation slice must not silently promote a one-bundle result.

## Error Handling

The runner must:

- fail before remote execution if the local bundle path exists;
- fail before workload execution if the remote bundle path exists;
- preserve preflight, campaign, diagnostic, remote-verifier, transfer, and
  local-verifier exit codes separately;
- stop and reap only sampler PIDs it created;
- preserve partial raw artifacts after a remote failure;
- never recover by rerunning only one measured epoch in the same bundle;
- never overwrite an A artifact with a B artifact;
- never classify a bundle whose canonical artifact or verifier failed.

Postprocessing-only recovery is allowed only when both measured worker outputs
and all required raw telemetry inputs already exist and remain hash-bound. Any
recovery must record its provenance and retain the original non-zero exit
receipts.

## Safety and Correctness Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Use `sitian@10.232.195.203`.
- Write new experiment artifacts under
  `/dev/shm/sitian/tllm-qwen35-target-qwen3-draft-20260815`.
- Do not write experiment artifacts under `/data00`; only the existing Python
  executable may be used from that filesystem.
- Preserve `MAX_PROPOSAL_TOKENS=4`.
- Preserve temperature zero, accepted-prefix semantics, and exact greedy
  parity.
- Preserve workload-derived exact Proposal-KV capacity.
- Do not add `torch.cuda.synchronize()` to the measured request path.
- Do not terminate unrelated GPU processes.
- Preserve GPU-7 PID `703088`.
- Do not present synthetic or fake KV copy as real KV movement.
- Do not stage, commit, push, stash, reset, clean, create a branch, switch a
  branch, or create a worktree.
- Do not start background watchers; use bounded foreground commands and reuse
  existing execution sessions where possible.

## Verification Requirements

Focused tests must prove:

- the runner fixes epoch order to `learned_a,learned_b`;
- both worker commands pass `--policy learned`;
- A and B prime, worker, GPU, host, and log paths are distinct;
- each epoch primes before its measured worker;
- prime workers use two warmups and one measured repeat;
- measured workers use two warmups and eight measured repeats;
- no worker, interpreter, CUDA context, or sampler is intentionally shared;
- both epochs use batch four and temperature zero;
- `MAX_PROPOSAL_TOKENS=4` remains source-bound;
- Proposal-KV capacity remains workload-derived;
- exact per-repeat output parity is mandatory;
- workload and model identities must match;
- host and GPU coverage gates apply per repeat and per epoch;
- stationarity thresholds are `0.25` and `0.20`;
- the 10% E2E threshold uses `< 0.10` for stable and `>= 0.10` for a
  candidate effect;
- stable, candidate-effect, and inconclusive classifications are covered;
- conflicting TPOT or proposal-forward direction is inconclusive;
- source or raw-input tampering fails verification;
- absolute and parent-traversing artifact paths fail verification;
- one-bundle candidate classification cannot set
  `PROCESS_BOUNDARY_EFFECT_ESTABLISHED=YES`;
- no `torch.cuda.synchronize()` text is introduced in the measured path.

Before a remote launch, local validation must include:

```bash
python3 -m pytest -q \
  tools/test_autoregressive_draft_executor.py \
  tools/test_autoregressive_draft_performance_gate.py \
  tools/test_autoregressive_draft_instability_telemetry.py \
  tools/test_autoregressive_draft_host_sampler.py \
  tools/test_autoregressive_draft_host_semantic_diagnostic.py \
  tools/test_autoregressive_draft_learned_aa_diagnostic.py

python3 -m py_compile \
  tools/autoregressive_draft_performance_worker.py \
  tools/autoregressive_draft_host_sampler.py \
  tools/autoregressive_draft_learned_aa_diagnostic.py \
  tools/verify_autoregressive_draft_learned_aa_diagnostic.py

bash -n tools/run_autoregressive_draft_learned_aa_remote.sh
```

After a remote launch, authority requires:

```text
remote campaign exit:       0
remote verifier exit:       0
local verifier exit:        0
manifest verification:      PASS
exact greedy parity:        true
measured repeats per epoch: 8
host/GPU repeat coverage:   PASS
source identity:            PASS
```

## Claim Boundary

Even a replicated learned A/A process-boundary effect does not establish:

- a specific host or GPU root cause;
- a stable long-context performance baseline;
- 4K, 16K, or 32K authority;
- Proposal-KV offload benefit;
- real KV H2D reduction;
- a second learned model structure;
- TP1 or TP4 promotion beyond the exact tested workload;
- Phase-1 completion; or
- promotion of Generic MTP/Speculative Runtime + Transactional KV Cache.

The next scientific action depends on the result:

```text
LEARNED_AA_STABLE:
  treat the r7/r8 reversal as not reproduced by a same-policy process
  boundary and return to controlled runtime/host attribution

LEARNED_AA_PROCESS_BOUNDARY_EFFECT:
  run the required second source-identical replication bundle before
  identifying or optimizing a mechanism

LEARNED_AA_INCONCLUSIVE:
  repair stationarity or metric-direction uncertainty before any causal
  experiment
```
