# Autoregressive Draft Paired Stability Design

## Goal

Determine whether the apparent learned-policy process-boundary latency effect
can be reproduced under a precommitted, stationarity-admitted, label-balanced
protocol.

This is a repaired evidence protocol for the completed learned A/A discovery
bundle. It is not a runtime optimization, a host-resource root-cause study, a
replication campaign, or Phase-1 promotion evidence.

The strongest classification available from one bundle is:

```text
CANDIDATE_PROCESS_BOUNDARY_EFFECT
```

Every output must retain:

```text
process_boundary_effect_established=false
```

No classification from this protocol authorizes an automatic replication
bundle.

## Motivation

The source-bound learned A/A discovery bundle:

```text
tp4-qwen3-b4-learned-aa-discovery-20260815T215015Z
```

preserved exact greedy parity, complete source identity, telemetry coverage,
eight measured repeats per epoch, and byte-identical remote and local
verification. Its canonical classification was:

```text
LEARNED_AA_INCONCLUSIVE
```

The observed A-relative-to-B medians were large:

```text
E2E:               +29.9893%
TPOT:              +28.3472%
proposal-forward:  +21.2047%
```

They were not interpretable because five of the six primary stationarity
range gates failed. Offline analysis also showed that E2E variation tracked
proposal-forward and backend-submit variation closely:

```text
correlation(E2E, proposal-forward):
  A: 0.9655
  B: 0.9557

correlation(proposal-forward, backend-submit):
  A: 0.9908
  B: 0.9747
```

The evidence therefore localizes the visible variance to proposal execution
latency, but does not identify a host, GPU, or runtime cause. Repeating the
same sequential A-then-B protocol with more samples would preserve the
process-position and time-drift confound.

## Selected Approach

Use epoch admission followed by four balanced paired blocks.

The block schedule is fixed before execution:

```text
block 0: A -> B
block 1: B -> A
block 2: B -> A
block 3: A -> B
```

The schedule text and its SHA-256 digest are source-bound inputs to the
canonical artifact and independent verifier. The runner must not accept an
alternative schedule through CLI arguments or environment variables.

`A` and `B` are artifact labels only. Both labels invoke the same learned
runtime policy, source snapshot, model checkpoints, prompts, proposal
configuration, batch size, warmup settings, and measurement settings.

The complete bundle contains:

```text
4 blocks
8 measured epochs
8 isolated prime processes
8 isolated measured worker processes
5 measured repeats per epoch
40 measured repeats total
```

Each measured epoch is a fresh process with a fresh Python interpreter, CUDA
context, model load, and telemetry sampler set. No worker or model state is
shared across epochs.

## Why This Repairs the Discovery Protocol

### Epoch Admission

The discovery bundle allowed a high-variance epoch to contribute to the
between-position comparison. The repaired protocol first decides whether
each epoch is measurable on its own. A bundle with any inadmissible epoch is
classified as unstable before effect estimation can influence the result.

### Balanced Labels

The discovery bundle always ran A before B. The repaired schedule places each
label first twice and second twice. A label-specific artifact can therefore
be separated from a process-position effect.

### Symmetric Block Order

The `AB, BA, BA, AB` schedule is symmetric around the bundle midpoint. It
reduces sensitivity to monotonic bundle-wide drift and permits an explicit
AB-versus-BA sequence-interaction check.

### Precommitted Exclusion Rules

No measured repeat, epoch, or block may be removed after execution. An
admission failure invalidates effect interpretation for the entire bundle.
This prevents a stable-looking result from being manufactured by deleting
high-variance evidence.

## Rejected Alternatives

### Increase Repeats in A-Then-B

More repeats reduce sampling error but do not remove the process-position,
label, and elapsed-time confound. The prior discovery already showed that a
large median delta can coexist with failed stationarity.

### Interleave AB and BA Without Admission

Balanced ordering alone does not make an unstable epoch interpretable. A
single noisy process can dominate a five-repeat median and produce a
directionally consistent paired ratio by accident.

### Run Separate Bundles Per Pair

Separate bundles add SSH, source extraction, preflight, and bundle-start
variation between pairs. The first repaired protocol keeps all four blocks
under one source snapshot and one bundle-level environment. A later
replication, if explicitly approved, must use a fresh bundle.

### Reuse the Existing Learned A/A Artifact Schema

The existing schema represents two fixed-position epochs and cannot express
eight epoch admissions, a precommitted crossover schedule, block-local
effects, label effects, or sequence interactions. Reinterpreting it would
weaken tamper detection and blur claim boundaries.

## Fixed Runtime Configuration

The runner preserves the authority configuration used by the learned A/A
discovery unless the implementation plan identifies a source-bound value
already recorded by that runner:

```text
remote host:       sitian@10.232.195.203
remote Python:     /data00/home/sitian/miniconda3/envs/py311/bin/python
remote base:       /dev/shm/sitian/tllm-qwen35-target-qwen3-draft-20260815
target model:      ${REMOTE_BASE}/target-qwen3-1.7b
draft model:       ${REMOTE_BASE}/draft
GPU indices:       3,4,6,7
runtime policy:    learned
batch size:        4
proposal limit:    MAX_PROPOSAL_TOKENS=4
temperature:       0
prime protocol:    one isolated process with 2 warmups + 1 measured run
worker warmups:    2 per measured epoch
measured repeats:  5 per epoch
```

`/data00` may only supply the existing Python interpreter. Source archives,
temporary files, logs, artifacts, and experiment output must remain under
the remote base in `/dev/shm`.

The runner must not terminate unrelated GPU processes and must preserve the
pre-existing GPU 7 process with PID `703088`. A changed unrelated-process
inventory is an invariant failure, not permission to clean the host.

The measured request path must not add `torch.cuda.synchronize()` or alter
runtime semantics to reduce measurement variance.

## Prime and Measured Epoch Semantics

For every epoch:

1. Launch one isolated learned-policy prime process.
2. Run two prime warmups followed by one prime measured workload.
3. Exit and reap the prime process completely.
4. Capture the pre-epoch GPU and host invariant snapshot.
5. Start fresh script-owned GPU and host samplers.
6. Launch a fresh learned-policy measured worker.
7. Execute exactly two worker warmups followed by five measured repeats.
8. Exit and reap the worker.
9. Stop and reap only samplers owned by the runner.
10. Capture the post-epoch invariant snapshot.
11. Assemble the epoch admission inputs without deciding admission in shell.

Prime timings are recorded but excluded from stationarity, effect, and
classification calculations.

## Epoch Admission

All eight epochs must pass every gate below. Gates are conjunctive and are
evaluated independently for each epoch.

### Identity and Semantic Gates

Each epoch must have:

- the expected artifact label and schedule position;
- runtime policy exactly `learned`;
- identical source hashes, model identities, prompts, request order, batch
  size, proposal limit, temperature, prime protocol, warmup protocol, and
  measured-repeat count;
- exact greedy output identity against the source-bound baseline;
- identical accepted-token outputs and accepted-prefix semantics;
- identical proposal counts, proposal lengths, and total verified-token
  counts across labels and positions;
- no accepted-KV rematerialization or other runtime-semantic change introduced
  solely for this experiment.

Any mismatch fails admission even if timing metrics appear stable.

### Coverage Gates

Each measured repeat must have:

- one complete worker timing record;
- one proposal-forward timing record;
- one backend-submit timing record;
- aligned GPU telemetry covering the measured interval;
- aligned host-semantic telemetry covering the measured interval;
- nonnegative, monotonic timestamps;
- no missing or duplicate repeat index.

The epoch must contain exactly five measured repeats. Extra, missing, or
unmatched records fail admission.

### GPU and Process Invariant Gates

The verifier must confirm:

- the same expected GPU UUID set is visible before and after every epoch;
- the runner uses only the declared GPU indices;
- no GPU reset, Xid, throttle violation, or unavailable telemetry interval is
  recorded during a measured repeat;
- clock, P-state, and throttle fields satisfy the source-bound validity rules;
- the protected unrelated GPU 7 PID `703088` remains present;
- no script-owned worker or sampler leaks into the next epoch;
- no unrelated process is terminated by the runner.

These gates establish measurement admissibility only. Passing them does not
attribute an effect to GPU state.

### Primary Stationarity Gates

The primary metrics are:

```text
E2E batch elapsed seconds
TPOT seconds
proposal-forward milliseconds
```

For each metric with measured values `x[0:5]`, define:

```text
epoch_median = median(x[0:5])
epoch_mad = median(abs(x[i] - epoch_median) for i in 0..4)
robust_dispersion = epoch_mad / epoch_median

first_half_median = median(x[0:2])
second_half_median = median(x[3:5])
half_drift =
  abs(second_half_median - first_half_median) / epoch_median
```

The center repeat `x[2]` contributes to `epoch_median` and `epoch_mad` but is
excluded from the two half medians so the halves have equal cardinality.

Every primary metric must satisfy:

```text
epoch_median > 0
robust_dispersion <= 0.10
half_drift <= 0.15
```

Threshold equality passes. Non-finite values fail.

### Bundle Admission Rule

If any gate fails for any epoch:

```text
classification = PAIRED_PROTOCOL_UNSTABLE
candidate_process_boundary_effect = false
process_boundary_effect_established = false
```

The artifact still records all raw inputs, per-epoch admission results, and
failure reasons. It must not emit an authoritative candidate/no-candidate
effect interpretation.

No implementation may:

- delete a failed epoch;
- rerun one epoch in place;
- remove a measured repeat;
- substitute a prime result;
- retain only passing blocks;
- relax a threshold after inspecting results.

A new attempt requires a fresh run tag and a complete new bundle.

## Effect Estimation

Effect estimation is authoritative only after all eight epochs pass
admission.

For primary metric `m`, block `b`, label `L`, and position `p`, let:

```text
y[b,L,p,m] = log(median of the five measured repeats)
```

The block-local process-position effect is:

```text
position_effect[b,m] = y[b,second,m] - y[b,first,m]
```

A positive value means the second process is slower. The relative effect is:

```text
position_relative[b,m] = exp(position_effect[b,m]) - 1
```

The block-local label effect is:

```text
label_effect[b,m] = y[b,A,m] - y[b,B,m]
label_relative[b,m] = exp(label_effect[b,m]) - 1
```

A positive value means label A is slower. Because A and B invoke identical
runtime policies, a large reproducible label effect is evidence of an
unresolved protocol artifact.

The aggregate effects are the medians of the four block-local log effects:

```text
aggregate_position_effect[m] = median(position_effect[b,m])
aggregate_label_effect[m] = median(label_effect[b,m])
```

The artifact must also report, without using them as primary classification
metrics:

- backend-submit block-local and aggregate effects;
- per-repeat raw ratios;
- AB-only and BA-only position-effect medians;
- chronological block-index trend;
- GPU and host telemetry summaries;
- acceptance, proposal-length, and verified-token summaries.

## Order-Effect Check

For each primary metric, compute:

```text
ab_position_effect[m] =
  median(position_effect[b,m] for b in {0,3})

ba_position_effect[m] =
  median(position_effect[b,m] for b in {1,2})

sequence_interaction[m] =
  ab_position_effect[m] - ba_position_effect[m]
```

The E2E candidate direction passes the order-effect check only if:

1. `ab_position_effect[E2E]` and `ba_position_effect[E2E]` have the same
   nonzero sign as `aggregate_position_effect[E2E]`;
2. at least one AB block and at least one BA block have E2E position effects
   in that same direction with absolute relative magnitude at least `10%`;
3. the aggregate E2E label effect has absolute relative magnitude below
   `10%`;
4. the four E2E label effects do not independently satisfy the candidate
   direction-and-magnitude rule.

This check rejects a result that appears only in one label order or can be
explained by the synthetic A/B labels. `sequence_interaction` is reported as
diagnostic evidence; it is not itself interpreted as a host or GPU cause.

## Classification

Classification precedence is fixed in the following order.

### `PAIRED_PROTOCOL_UNSTABLE`

Use when any source, identity, parity, coverage, invariant, repeat-count, or
epoch-stationarity gate fails.

Required fields:

```text
candidate_process_boundary_effect=false
process_boundary_effect_established=false
```

### `CANDIDATE_PROCESS_BOUNDARY_EFFECT`

Use only when all epochs pass admission and all of the following hold:

1. At least three of four E2E block-local position effects have the same
   nonzero sign.
2. The aggregate E2E position effect has absolute relative magnitude at least
   `10%`.
3. The aggregate TPOT and proposal-forward position effects have the same
   sign as the aggregate E2E position effect.
4. The E2E order-effect check passes.

Required fields:

```text
candidate_process_boundary_effect=true
process_boundary_effect_established=false
```

The `10%` boundary is inclusive.

### `NO_REPRODUCIBLE_PROCESS_EFFECT`

Use when all epochs pass admission but any candidate condition fails.

This includes:

- aggregate E2E magnitude below `10%`;
- fewer than three E2E blocks with a common direction;
- TPOT or proposal-forward aggregate direction disagreement;
- an order-effect or label-effect explanation that is not excluded.

Required fields:

```text
candidate_process_boundary_effect=false
process_boundary_effect_established=false
```

This classification means the repaired bundle did not produce a reproducible
candidate. It does not prove that no process-boundary effect exists.

## Canonical Artifact

The canonical JSON must be self-describing and include at least:

```text
schema_version
classification
candidate_process_boundary_effect
process_boundary_effect_established
claim_boundary

run_tag
bundle_start_utc
bundle_finish_utc
remote_host
remote_base
schedule
schedule_sha256

configuration
source_files
source_sha256
model_identity
prompt_identity
command_identity

blocks[4]
epochs[8]
measured_repeat_count_total

epoch_admission
bundle_admission
primary_stationarity
coverage
gpu_invariants
process_invariants
exact_parity

block_local_position_effects
block_local_label_effects
aggregate_position_effects
aggregate_label_effects
ab_position_effects
ba_position_effects
sequence_interactions
diagnostic_effects

raw_input_files
raw_input_sha256
```

Each admission failure must have a stable machine-readable code, affected
block, label, position, epoch, metric if applicable, observed value, expected
condition, and source path.

## Artifact Layout

The implementation plan may refine names while preserving unique epoch
identity and hash binding. The intended layout is:

```text
schedule.txt
command.txt
source.sha256
preflight/

blocks/block-0-ab/
  a-first/
  b-second/
blocks/block-1-ba/
  b-first/
  a-second/
blocks/block-2-ba/
  b-first/
  a-second/
blocks/block-3-ab/
  a-first/
  b-second/

paired-stability.json
verify.paired-stability.remote.json
verify.paired-stability.local.json
manifest.sha256
```

Every epoch directory contains:

```text
prime-worker.json
prime.log
worker.json
worker.log
gpu.csv
host-semantic.jsonl
host-semantic.stderr.log
vmstat.log
mpstat.log
pidstat.log
gpu.before.txt
gpu.after.txt
process.before.txt
process.after.txt
```

## Runner Data Flow

The future dedicated runner must:

1. Parse arguments before deriving local or remote run paths.
2. Refuse to overwrite an existing local or remote run directory.
3. Materialize the fixed schedule and bind its digest.
4. Package the complete source and verifier dependency closure.
5. Record source, model, prompt, command, environment, GPU, and process
   identities.
6. Run compilation and the focused preflight test suite remotely.
7. Execute all four blocks and all eight epochs in schedule order.
8. Stop and reap only runner-owned processes after every epoch.
9. Preserve all partial evidence if an epoch or sampler fails.
10. Assemble the canonical artifact once, after the execution attempt ends.
11. Run the independent remote verifier.
12. Download the complete bundle.
13. Run the independent local verifier.
14. Build and verify `manifest.sha256`.

The runner must not stop after the first unstable epoch, because the complete
precommitted schedule is evidence about the environment. It may stop only for
a safety condition that makes continued execution invalid, such as loss of
the expected GPU set, protected-process disappearance, source corruption, or
insufficient remote storage. Such a stop still produces
`PAIRED_PROTOCOL_UNSTABLE` with a partial-execution reason.

## Independent Verifier

The verifier must recompute the canonical artifact from hash-bound raw inputs.
It must not trust derived admission, effect, or classification fields.

It must reject:

- schedule changes or a schedule digest mismatch;
- fewer or more than four blocks, eight epochs, or forty measured repeats;
- duplicated or reordered block, label, position, epoch, or repeat identity;
- source, model, prompt, command, or raw-input hash mismatch;
- malformed, missing, non-finite, or misaligned timing and telemetry data;
- incorrect median, MAD, drift, log-ratio, aggregate, sequence-interaction, or
  threshold-boundary calculations;
- incorrect admission precedence or classification;
- any canonical artifact with
  `process_boundary_effect_established=true`;
- remote and local receipts that are not structurally and byte-for-byte
  equivalent apart from explicitly permitted verifier metadata.

The remote and local verifier receipts must bind the canonical artifact hash,
manifest hash, verifier source hash, and all authoritative raw inputs.

## Test Requirements

The implementation plan must include focused tests for:

- the exact `AB, BA, BA, AB` schedule and schedule hash binding;
- unique paths and identities for all eight epochs;
- exactly five measured repeats per epoch and forty total;
- source-identical A/B runtime configuration;
- prime exclusion from measured statistics;
- exact parity and accepted-prefix semantic rejection;
- telemetry alignment and coverage rejection;
- GPU and protected-process invariant rejection;
- `MAD / median` pass, equality, and fail boundaries;
- five-repeat half-drift calculation with the center repeat excluded;
- half-drift pass, equality, and fail boundaries;
- all-or-nothing bundle admission;
- rejection of post-hoc repeat, epoch, or block deletion;
- block-local position and label log ratios for both AB and BA blocks;
- aggregate position and label effects;
- AB/BA sequence-interaction calculation;
- `10%` magnitude equality;
- three-of-four direction equality;
- TPOT and proposal-forward direction disagreement;
- order-effect check failure;
- all three classifications and their precedence;
- forced `process_boundary_effect_established=false`;
- source and raw-input tamper rejection;
- remote runner executable mode and source-package dependency closure;
- local and remote verifier equivalence;
- manifest completeness and verification.

## Implementation File Boundaries

The implementation plan should prefer dedicated files:

```text
tools/run_autoregressive_draft_paired_stability_remote.sh
  own packaging, fixed schedule execution, remote/local verification, and
  manifest construction

tools/autoregressive_draft_paired_stability_diagnostic.py
  own schema validation, epoch admission, paired effect estimation, and
  canonical classification

tools/verify_autoregressive_draft_paired_stability_diagnostic.py
  independently recompute and verify the canonical artifact

tools/test_autoregressive_draft_paired_stability_diagnostic.py
  cover diagnostic and verifier semantics

tools/test_autoregressive_draft_instability_telemetry.py
  retain shared runner executable and source-inventory closure contracts if
  this remains the established contract-test home
```

Existing learned A/A artifacts and verifiers retain their original semantics.
The repaired protocol must not silently change or reinterpret them.

## Failure Handling

- SSH, preflight, source-package, storage, worker, sampler, assembler,
  verifier, download, or manifest failure preserves a failed or partial
  artifact under a unique run tag.
- A workload exit code must never be rewritten as success.
- Missing dependencies are environment failures, not passing tests.
- A safety stop must record which precommitted blocks and epochs did and did
  not execute.
- A failed bundle cannot be repaired in place.
- The runner must not launch a replacement epoch or a replication bundle.

## Claim Boundary

This protocol can establish only that one source-bound bundle was:

1. internally admissible under the precommitted stationarity and invariant
   gates; and
2. either did or did not exhibit a balanced paired candidate process-position
   effect.

It cannot establish:

- a host-resource cause;
- a GPU-resource cause;
- a learned-versus-target policy effect;
- a production runtime regression;
- a performance improvement;
- generalization to TP1, another model, another batch size, another context
  length, or another host;
- Phase-1 completion or promotion readiness.

Even for `CANDIDATE_PROCESS_BOUNDARY_EFFECT`:

```text
process_boundary_effect_established=false
```

must remain authoritative until a separately designed and explicitly approved
replication protocol succeeds.

## Execution Gate

Writing this design does not authorize implementation or remote execution.

The next allowed step is user review of this written specification. Only
after explicit approval may a separate implementation plan be written. The
implementation plan itself does not authorize a remote workload; remote
execution requires the later implementation and validation gates to pass and
must not automatically launch a replication bundle.
