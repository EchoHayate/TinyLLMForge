# Qwen3.5 TP4 Hybrid-Prefix Performance and Cache Authority Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a source-bound, correctness-gated TP4 authority that measures exact Qwen3.5 hybrid-prefix speed and cache cost before any compressed-state candidate is evaluated.

**Architecture:** A dependency-light pure contract freezes prerequisites, workloads, policies, artifacts, thresholds, and classifications. A separate TP4 worker records raw engine, output, timing, memory, and cache observations; a remote runner stages immutable source and executes paired isolated processes; an independent verifier reconstructs correctness and `GO | NO_GO | INVALID` without trusting producer summaries. The first implementation stops after CPU-safe contract/tests while correctness or GPU resources remain blocked.

**Tech Stack:** Python 3, standard library, PyTorch only in GPU worker/verifier paths, TinyLLMForge `LLMEngine`, JSON/JSONL, SHA-256, SSH/Kerberos.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Do not modify the frozen `qwen35-tp4-source-prep-20260729-010400` correctness source.
- Do not stage, commit, merge, create a branch, or create a PR.
- Use only `sitian@10.232.195.203` with `KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian`.
- Never kill or signal pre-existing remote processes.
- Require four unique GPUs with at least `24 * 1024**3` free bytes and no active compute processes.
- Preserve failed, negative, blocked, and superseded evidence under unique run tags.
- Do not weaken schema-v2 canonical `NO_GO`.
- Do not claim latency, throughput, cache, memory, compression, or quality improvement before independent canonical `GO`.
- P0 and P1 use identical model, source, TP4 topology, KV capacity, workload, and sampling configuration.
- The first candidate is exact full-fidelity restore; int4, token sparse, low rank, and Gist layer sharing remain excluded.

---

## File Map

- Create `tools/qwen35_tp4_hybrid_prefix_benchmark_contract.py`: pure constants, schemas, prerequisite validation, matrix construction, pairing, and classification.
- Create `tools/test_qwen35_tp4_hybrid_prefix_benchmark_contract.py`: dependency-light RED/GREEN tests and tamper cases.
- Create `tools/qwen35_tp4_hybrid_prefix_benchmark_worker.py`: later GPU-only TP4 engine worker and raw observation writer.
- Create `tools/test_qwen35_tp4_hybrid_prefix_benchmark_worker.py`: dependency-light command/config/row tests with fake engine objects.
- Create `tools/verify_qwen35_tp4_hybrid_prefix_benchmark.py`: independent artifact and metric verifier.
- Create `tools/test_verify_qwen35_tp4_hybrid_prefix_benchmark.py`: complete synthetic fixture and tamper rejection.
- Create `tools/run_qwen35_tp4_hybrid_prefix_benchmark_remote.py`: source staging, prerequisite/resource preflight, paired orchestration, download, and verify modes.
- Create `tools/test_run_qwen35_tp4_hybrid_prefix_benchmark_remote.py`: no-GPU runner tests.
- Modify `AGENT_HANDOFF_STATE.md`: exact status, prerequisites, blocked evidence, commands, and claim boundary.

### Task 1: Freeze the Dependency-Light Contract

**Files:**
- Create: `tools/qwen35_tp4_hybrid_prefix_benchmark_contract.py`
- Create: `tools/test_qwen35_tp4_hybrid_prefix_benchmark_contract.py`

**Interfaces:**
- Produces: `BenchmarkCase`, `PrerequisiteStatus`, `build_case_matrix()`, `validate_prerequisites()`, `pair_order()`, `classify_run()`, and canonical constants used by every later task.

- [x] **Step 1: Write failing tests for exact constants and matrix**

Define tests that require:

```python
POLICIES = ("recompute", "exact_restore")
WORKLOADS = ("w0_short_control", "w1_medium_reuse", "w2_long_reuse", "w3_batched_fanout", "w4_miss_invalidation")
WARMUP_REPETITIONS = 1
CORRECTNESS_REPETITIONS = 1
MEASURED_REPETITIONS = 5
MODEL_MANIFEST_SHA256 = "3e650a908234771c3cf1ac4e20c4d38fe69982efedaf4a3e631ad0b14aad7dd0"
TP4_ROOT_SOURCE_TREE_SHA256 = "b2d0b77de953e273dbf62f0e7b2bbe689ef33c183edf65830940e43123bb485f"
```

Assert the matrix contains both policies for every phase, workload, and
repetition, has unique deterministic IDs, and alternates pair order by
repetition.

- [x] **Step 2: Run the focused test and confirm RED**

Run:

```bash
python3 tools/test_qwen35_tp4_hybrid_prefix_benchmark_contract.py
```

Expected: failure because the contract module does not exist.

- [x] **Step 3: Implement immutable cases, thresholds, and schemas**

Implement frozen workload fields, exact artifact inventory, raw row fields,
process row fields, threshold values, canonical JSON bytes, SHA-256 helpers,
and deterministic case IDs. The module must import only the standard library.

- [x] **Step 4: Add prerequisite fail-closed tests**

Cover:

- missing prerequisite file;
- wrong model manifest;
- root-logit source tree mismatch;
- absent independent verification;
- any classification other than `PASS`;
- artifact hash mismatch;
- complete three-authority `PASS`.

`validate_prerequisites()` must return `BLOCKED_CORRECTNESS` for missing or
non-passing authorities and must never authorize a worker launch.

- [x] **Step 5: Add pure classifier tests**

Cover:

- valid full metrics -> `GO`;
- correctness mismatch -> `INVALID`;
- missing measured repetition -> `INVALID`;
- W1/W2/W3 threshold failure -> `NO_GO`;
- capacity mismatch -> `NO_GO`;
- cache accounting mismatch -> `INVALID`;
- absent eligible GPUs before execution -> `BLOCKED_RESOURCES`.

- [x] **Step 6: Run contract tests**

Run:

```bash
python3 tools/test_qwen35_tp4_hybrid_prefix_benchmark_contract.py
python3 -m py_compile \
  tools/qwen35_tp4_hybrid_prefix_benchmark_contract.py \
  tools/test_qwen35_tp4_hybrid_prefix_benchmark_contract.py
```

Expected: all tests pass.

### Task 2: Add a CPU-Testable Worker Boundary

**Files:**
- Create: `tools/qwen35_tp4_hybrid_prefix_benchmark_worker.py`
- Create: `tools/test_qwen35_tp4_hybrid_prefix_benchmark_worker.py`

**Interfaces:**
- Consumes: Task 1 case and row schemas.
- Produces: `build_engine_configuration(policy, case)`, `run_case(...)`, atomic JSON/JSONL writers, and raw row normalization.

- [x] **Step 1: Write fake-engine tests**

Use fake engine/model-runner/cache objects to require:

- identical TP4, KV-capacity, sampling, and workload settings for P0/P1;
- P0 never configures hybrid-prefix publication/restore;
- P1 configures exact full-fidelity publication/restore;
- one excluded warmup, one correctness phase, and five measured phases;
- CUDA timing hooks are injectable;
- P0 cache observations are exactly zero;
- P1 cache accounting is copied from `observation_snapshot()`;
- output IDs, logits paths, memory snapshots, and raw nanoseconds are emitted;
- exceptions publish failure evidence atomically.

- [x] **Step 2: Run focused tests and confirm RED**

Run:

```bash
python3 tools/test_qwen35_tp4_hybrid_prefix_benchmark_worker.py
```

Expected: failure because the worker does not exist.

- [x] **Step 3: Implement only dependency-light worker helpers**

Implement configuration construction, schema normalization, timing injection,
atomic writers, and fake-engine-compatible orchestration. Keep Torch and
TinyLLMForge imports inside the real worker entry point so CPU tests remain
dependency-light.

- [x] **Step 4: Run worker tests**

Run:

```bash
python3 tools/test_qwen35_tp4_hybrid_prefix_benchmark_worker.py
python3 -m py_compile \
  tools/qwen35_tp4_hybrid_prefix_benchmark_worker.py \
  tools/test_qwen35_tp4_hybrid_prefix_benchmark_worker.py
```

Expected: all tests pass without GPU initialization.

### Task 3: Build the Independent Verifier

**Files:**
- Create: `tools/verify_qwen35_tp4_hybrid_prefix_benchmark.py`
- Create: `tools/test_verify_qwen35_tp4_hybrid_prefix_benchmark.py`

**Interfaces:**
- Consumes: Task 1 constants plus on-disk raw artifacts.
- Produces: `verify_run(run_dir: Path) -> dict`, atomic `independent_verification.json`, and `report.md`.

- [x] **Step 1: Create a complete synthetic fixture**

The fixture must include exact top-level inventory, nested logits/log files,
an `artifact_manifest.json` covering every producer/raw input and nested file,
all policies, workloads, phases, and repetitions, valid hashes, exact outputs,
valid cache equations, capacity parity, and metrics satisfying every
threshold. The artifact manifest must exclude itself and verifier outputs to
avoid a self-referential hash.

- [x] **Step 2: Write RED tests for independent reconstruction**

Require the verifier to recompute:

- prerequisite hashes and classifications;
- exact case matrix and pair order;
- output-token identity;
- logits tensor hashes and tolerance;
- medians, throughput, TTFT, decode latency, initialization, and memory ratios;
- cache accounting and capacity parity;
- final `GO | NO_GO | INVALID`.

- [x] **Step 3: Implement verifier without producer aggregation imports**

The verifier may import only immutable constants and schema definitions from
the contract. It must implement its own row pairing, statistics, equations,
and classification checks.

- [x] **Step 4: Add tamper tests**

Reject:

- extra or missing files;
- symlinks;
- duplicate or reordered row IDs;
- changed prerequisite/source/model/workload hash;
- changed command or policy order;
- changed output IDs or logits bytes;
- non-finite metrics;
- changed cache counters;
- missing repetition;
- mixed capacity;
- worker log traceback or missing completion marker.

- [x] **Step 5: Run verifier tests**

Run:

```bash
python3 tools/test_verify_qwen35_tp4_hybrid_prefix_benchmark.py
python3 -m py_compile \
  tools/verify_qwen35_tp4_hybrid_prefix_benchmark.py \
  tools/test_verify_qwen35_tp4_hybrid_prefix_benchmark.py
```

Expected: all tests pass.

### Task 4: Add the Fail-Closed Remote Runner

**Files:**
- Create: `tools/run_qwen35_tp4_hybrid_prefix_benchmark_remote.py`
- Create: `tools/test_run_qwen35_tp4_hybrid_prefix_benchmark_remote.py`

**Interfaces:**
- Consumes: Tasks 1-3 and the existing strict TP4 GPU selector/cleanup contract.
- Produces: modes `preflight`, `smoke`, `canonical`, `download-only`, and `verify-only`.

- [x] **Step 1: Write runner RED tests**

Cover:

- safe unique run tags;
- exact SSH target and Kerberos inheritance;
- prerequisite check occurs before SSH resource query;
- `BLOCKED_CORRECTNESS` creates no remote path;
- `BLOCKED_RESOURCES` creates no work/publish path;
- source and model identities are checked before model construction;
- unique dynamic port pairs;
- alternating P0/P1 order;
- exact artifact download and path safety;
- local verifier invocation;
- no forbidden arbitrary process signaling.

- [ ] **Step 2: Implement source-bound staging and preflight**

Reuse the existing TP4 resource query and selector without weakening it. Stage
a deterministic tar from an explicit owned-source list, verify the remote tree,
and keep every run under a new unique remote directory.

CPU-safe contract complete:

- deterministic USTAR bundle with fixed metadata;
- source-tree SHA over sorted relative path plus file bytes;
- fail-closed missing, traversal, symlink, and repository-escape checks;
- non-destructive unique remote stage command with tar and extracted-tree SHA
  verification;
- exact canonical download inventory validation.
- prerequisite-first local source preparation: the benchmark tar is built only
  after correctness authorization, and its source-tree SHA is carried into
  both `READY` and `BLOCKED_RESOURCES` evidence;
- frozen owned-source inventory excludes interpreter caches and rejects a tar
  output inside the owned tree.

Remaining before this step is fully complete: wire the tested builders into
upload/download execution after correctness prerequisites and the real worker
are available. No SSH, remote path creation, or GPU workload was executed.

READY-only launch planning is also CPU-complete:

- a blocked preflight cannot construct a launch plan;
- source bundle and worker authorization source SHA must match;
- runtime artifact paths are explicit absolute paths;
- one safe stage command and all 70 canonical case commands are generated;
- every process receives a unique dist/master port pair;
- remote tar extraction validates every member as a relative regular file and
  rejects traversal, symlinks, and hardlinks before extraction.

The plan is data only. It does not run `scp`, `ssh`, stage commands, or worker
commands.

- [x] **Step 3: Implement blocked modes first**

Before any GPU worker code is enabled, make `preflight`, `smoke`, and
`canonical` return structured `BLOCKED_CORRECTNESS` while prerequisite
authorities are absent. Assert that no remote model command can be built from
that state.

- [x] **Step 4: Run runner tests**

Run:

```bash
python3 tools/test_run_qwen35_tp4_hybrid_prefix_benchmark_remote.py
python3 -m py_compile \
  tools/run_qwen35_tp4_hybrid_prefix_benchmark_remote.py \
  tools/test_run_qwen35_tp4_hybrid_prefix_benchmark_remote.py
```

Expected: all dependency-light tests pass and no SSH model workload runs.

### Task 4A: Close the Runner/Worker Execution Boundary

CPU-safe boundary complete:

- the runner's 70 canonical commands now map to exactly one worker case each;
- the worker rejects any non-canonical
  `(policy, workload, phase, repetition)` tuple before engine construction;
- model directory, model manifest, and correctness-prerequisite files are
  explicit worker inputs;
- actual model-manifest and prerequisite file SHA-256 values are checked
  before any lazy Torch/Engine import;
- the model manifest SHA must equal the frozen canonical model identity;
- the complete three-authority prerequisite bundle is revalidated through the
  independent contract, so a handwritten top-level `PASS` is rejected;
- the runner requires absolute, traversal-free runtime artifact paths and
  carries them into every worker command;
- the Engine now exposes acknowledged all-rank cache and memory snapshot
  transports with exact contiguous rank-inventory validation;
- the worker sums TP4 byte-valued cache and memory fields across ranks rather
  than silently reporting rank zero only;
- replicated cache counters require exact rank parity before aggregation;
- the default runtime loader remains fail-closed until the real TP4 Engine
  adapter is correctness-proven.

This closes a discovered matrix mismatch: the runner already emitted 70
single-case commands, while the original worker helper executed all seven
phases/repetitions for each command. Direct wiring would therefore have run
490 cases and broken process isolation. The new `run_benchmark_case` API
preserves the frozen 70-process authority.

The all-rank observation boundary is covered by dependency-light tests:

```text
contract:
  13 tests passed
worker:
  20 tests passed
independent verifier:
  8 tests passed
remote runner:
  20 tests passed
prerequisite builder:
  5 tests passed
cache/memory transport:
  5 tests passed
total:
  71 tests passed
py_compile:
  passed
git diff --check:
  passed
```

This proves CPU-side schema, rank inventory, aggregation, and fail-closed
transport behavior. It does not prove that a real TP4 process group reaches
the transport, nor does it measure GPU memory, cache savings, latency,
throughput, or quality.

The previously approved deterministic token-ID workload requirement is now
also implemented rather than represented by lengths alone:

```text
workload manifest bytes:
  60239
canonical JSON-file SHA-256:
  d8c81d6efa73f9b5e20dd0019e7e2dbf34e9f2ce4cef60658b0c44f3ca9648c2
```

For every workload the manifest freezes:

- the exact shared-prefix token IDs;
- the exact source suffix token IDs;
- one distinct suffix-token vector per continuation;
- the deterministic token seed;
- the original workload shape and generation settings.

W4 additionally freezes three different controls:

```text
continuation 0:
  one token mismatch at shared-prefix index 512
continuation 1:
  stale block-generation identity
continuation 2:
  explicit cache clear
```

The independent verifier compares the complete manifest with the contract and
rejects a re-signed single-token mutation. The worker now records the SHA-256
of the canonical JSON file bytes, including the trailing newline, rather than
the hash of the in-memory payload bytes. This closes a producer/verifier hash
domain mismatch that would otherwise invalidate a real run even when the
manifest content was correct.

Fresh CPU validation after the token manifest change:

```text
contract:
  15 tests passed
worker:
  22 tests passed
independent verifier:
  9 tests passed
remote runner:
  20 tests passed
prerequisite builder:
  5 tests passed
cache/memory transport:
  5 tests passed
total:
  76 tests passed
py_compile:
  passed
git diff --check:
  passed
```

The future real Engine adapter boundary is now fail-closed before any request
row or logits artifact is written. `validate_benchmark_requests(...)`
requires:

- exactly the frozen continuation count;
- a complete, closed request schema and unique non-empty request IDs;
- an actual boolean `restored_hybrid_state` rather than Python truthiness;
- exact prompt, reused-token, executed-prefill, and generated-token counts;
- exact generated output length with non-negative integer token IDs;
- exact decode-step timing count (`generated_tokens - 1`);
- non-negative integer TTFT/E2E/decode timings with `E2E >= TTFT`.

This closes a real producer-boundary bug where a value such as the string
`"false"` would previously become `True` through `bool(value)`, and malformed
adapter accounting could be written before independent verification.

The independent verifier does not import or trust the worker validator. It
separately reconstructs prompt, prefill, generation, output, decode-timing,
request-identity, and timing-order invariants from the frozen contract. New
re-signed tamper fixtures prove prompt, prefill, and generation-shape changes
are rejected.

Fresh CPU validation after request-boundary hardening:

```text
contract:
  15 tests passed
worker:
  25 tests passed
independent verifier:
  11 tests passed
remote runner:
  20 tests passed
prerequisite builder:
  5 tests passed
cache/memory transport:
  5 tests passed
total:
  81 tests passed
py_compile:
  passed
git diff --check:
  passed
```

The deterministic workload manifest is now an explicit runtime artifact
rather than an implicit worker default:

1. the source stage command first verifies the extracted source-tree SHA;
2. it imports the contract from that verified staged source;
3. it materializes canonical compact `workload_manifest.json` bytes;
4. it verifies the canonical file SHA before and after writing;
5. READY preflight authorization carries the exact workload-manifest SHA;
6. every one of the 70 worker commands receives the manifest path and SHA;
7. the worker verifies canonical SHA, actual file SHA, parsed content, and
   contract equality before lazy Torch or Engine imports.

`WorkerAuthorization.from_dict()` requires the workload SHA explicitly. It
does not fill missing legacy preflight data with a default.

Fresh CPU validation after runtime-artifact binding:

```text
contract:
  15 tests passed
worker:
  26 tests passed
independent verifier:
  11 tests passed
remote runner:
  20 tests passed
prerequisite builder:
  5 tests passed
cache/memory transport:
  5 tests passed
total:
  82 tests passed
py_compile:
  passed
git diff --check:
  passed
```

### Task 5: Complete Correctness Prerequisites Before GPU Benchmarking

**Files:**
- Modify only the separately approved correctness gate files and evidence directories.
- Do not modify the benchmark thresholds or workload after seeing results.

**Interfaces:**
- Produces: immutable `PASS` artifacts for TP4 root logits, cached continuation, and Engine/ModelRunner correctness.

- [ ] **Step 1: Wait for four truly idle GPUs or an explicitly approved equivalent node**

Run the existing strict TP4 preflight with a new unique tag. A blocked result
is preserved and does not authorize later steps.

- [ ] **Step 2: Run native-only smoke**

Require exact rank topology, rank-zero logits, non-root `None`, state cleanup,
process-group destruction, and bounded gate-owned child cleanup.

- [ ] **Step 3: Run TP4 root-logit authority**

Require the exact-five artifact inventory and independent verification.

- [ ] **Step 4: Implement and run cached-continuation correctness authority**

Use exact full-fidelity publication/restore across at least W1/W2/W3-shaped
continuations and require output-token equality, registered logits tolerance,
cache identity checks, and W4 miss/invalidation behavior.

CPU-safe authority contract and independent verifier complete:

```text
tools/qwen35_tp4_cached_continuation_correctness_contract.py
tools/test_qwen35_tp4_cached_continuation_correctness_contract.py
tools/verify_qwen35_tp4_cached_continuation_correctness_gate.py
tools/test_verify_qwen35_tp4_cached_continuation_correctness_gate.py
```

The frozen matrix contains exactly 19 continuation rows:

```text
W1:
  4 exact hits
W2:
  4 exact hits
W3:
  8 exact batched-fanout hits
W4:
  token_mismatch
  stale_block_generation
  cache_clear
```

Every row requires exact output-token equality, registered-logits tolerance,
cache identity, prompt/reused/executed-prefill accounting, rank inventory
`0..3`, process-group destruction, and zero surviving gate-owned children.

The exact-five artifact contract is:

```text
cached_continuation_correctness.json
reference_outputs.json
restored_outputs.json
registered_logits.json
source_manifest.json
```

The independent verifier checks closed inventory, regular files, source/model/
workload identities, every artifact SHA, cross-file output/logits consistency,
and independently recomputes `PASS`. Re-signed output and restore-semantic
tampering is rejected.

`verify_and_write(run_dir, output_path=...)` persists the independent result
as a canonical JSON file outside the exact-five run. It refuses run-internal
paths and existing outputs, writes atomically, and includes the required
`classification` plus canonical `model_manifest_sha256` fields consumed by
the prerequisite builder. The exact-five inventory therefore remains closed
while the independent verification has a separately hashable identity.

Current boundary:

```text
contract:
  complete
independent verifier:
  complete
CPU-testable producer boundary:
  complete
real TP4 executor:
  absent and fail-closed
exact-five real artifact:
  absent
cached_continuation prerequisite PASS:
  absent
```

Producer boundary:

```text
tools/qwen35_tp4_cached_continuation_correctness_producer.py
tools/test_qwen35_tp4_cached_continuation_correctness_producer.py
```

It executes the 19-row matrix through an injected executor, classifies before
publishing, writes all five artifacts into a sibling temporary directory,
runs the independent verifier, and atomically renames the directory only after
independent `PASS`. Invalid rows remove the temporary directory and never
create the requested output path. Existing output directories are never
overwritten. The default executor factory raises:

```text
real Qwen3.5 TP4 cached-continuation executor is not implemented
```

- [ ] **Step 5: Implement and run Engine/ModelRunner correctness authority**

Construct the real TP4 `LLMEngine`, execute scheduler/model-runner paths, and
prove rank completion, output equality, cache lifecycle, and cleanup.

CPU-safe authority contract, independent verifier, and exact-four producer
boundary complete:

```text
tools/qwen35_tp4_engine_correctness_contract.py
tools/test_qwen35_tp4_engine_correctness_contract.py
tools/verify_qwen35_tp4_engine_correctness_gate.py
tools/test_verify_qwen35_tp4_engine_correctness_gate.py
tools/qwen35_tp4_engine_correctness_executor.py
tools/test_qwen35_tp4_engine_correctness_executor.py
tools/qwen35_tp4_engine_correctness_producer.py
tools/test_qwen35_tp4_engine_correctness_producer.py
```

The frozen six-scenario matrix covers:

```text
construct_and_bind
publish_source
restore_w1
miss_w4_token
miss_w4_stale
miss_w4_clear
```

Every row binds the concrete `LLMEngine` and `ModelRunner` class identities,
all four ranks, non-root acknowledgements, exact scheduler/model-runner call
counts, output-token equality, publication/restore/cache lifecycle counts,
process-group destruction, zero rank failures, and zero surviving owned
children.

The exact-four artifact inventory is:

```text
engine_correctness.json
scheduler_observations.json
rank_events.json
source_manifest.json
```

The producer executes the matrix through an injected executor, classifies
before publication, writes into a sibling temporary directory, verifies the
closed exact-four inventory independently, and atomically renames only after
`PASS`. Invalid rank inventory or any other row failure leaves no target;
existing outputs are never overwritten. The default real executor remains
fail-closed:

```text
real Qwen3.5 TP4 Engine correctness executor is not implemented
```

The CPU-safe executor adapter now freezes every runtime input that a later
real implementation must receive explicitly:

```text
model directory and model-manifest path/SHA
source-tree SHA
workload-manifest path/SHA
model fingerprint
four unique GPU indices
distinct distributed and master ports
cache entry/byte limits
operation timeout
```

It also freezes the ordered action plan for all six scenarios. Runtime
construction is lazy and cannot happen before the first correctly ordered
scenario. Out-of-order scenarios fail before runtime construction. The
adapter imports neither Torch nor TinyLLMForge, and its default runtime factory
raises:

```text
real Qwen3.5 TP4 Engine runtime is not implemented
```

Each scenario is now explicitly isolated in a fresh Engine/TP4 process group.
This is required because every authoritative row asserts that the process
group has already been destroyed and all four ranks have exited. Restore and
miss scenarios create their seed publication fixture before
`begin_observation`, so fixture setup cannot inflate the scenario counters.
Every plan starts with `construct_engine` and ends with:

```text
close_engine
verify_cleanup
```

`AuditedScenarioRuntime` opens one backend session per scenario, executes every
listed action in order, merges only returned structured evidence, rejects
duplicate/missing fields through the existing row contract, and closes the
session on success or failure. It never fills evidence from expected values.

`model_runner_calls` is now explicitly defined as model execution calls issued
by `LLMEngine.step()`. Control-plane acknowledged commands used for restore
owner setup, runtime identity, snapshots, and cleanup are excluded and are
proved separately by rank acknowledgements/evidence. Therefore
`construct_and_bind` has:

```text
scheduler_steps = 0
model_runner_calls = 0
generated_tokens = 0
```

The other five scenarios retain
`model_runner_calls == scheduler_steps == generated_tokens`. This fixes the
earlier impossible interpretation in which configuration commands would have
been mixed with model-forward calls.

`build_configured_executor_factory(...)` bridges this adapter into the
exact-four producer without reading environment variables. It rebuilds the
configuration from a canonical payload across dynamic module identities,
preserving strict field validation instead of relying on Python class identity.
`build_audited_executor_factory(...)` additionally wires a backend session
factory directly into the producer while preserving the same fail-closed
configuration and evidence checks.

Current boundary:

```text
contract:
  complete
independent verifier:
  complete
CPU-testable producer boundary:
  complete
CPU-safe executor configuration/action-plan boundary:
  complete
real TP4 Engine executor:
  absent and fail-closed
exact-four real artifact:
  absent
independent real PASS:
  absent
```

- [ ] **Step 6: Freeze `correctness_prerequisites.json`**

Hash the three authoritative artifacts and independent verification files.
Do not hand-edit a `PASS`.

CPU-safe builder complete:

```text
tools/build_qwen35_tp4_performance_prerequisites.py
tools/test_build_qwen35_tp4_performance_prerequisites.py
```

The builder:

- requires exactly the three named authorities;
- binds each source tree, approved artifact SHA, and approved independent
  verification SHA;
- requires both copied JSON payloads to prove canonical-model `PASS`;
- rejects symlinks, missing files, duplicate/missing authorities, wrong
  root-logit source, tamper, and output overwrite;
- copies into a closed `prerequisites/<authority>/` inventory;
- writes canonical `correctness_prerequisites.json`;
- revalidates the final bundle through the independent contract before
  returning `PASS`.

The final real bundle remains pending because the real TP4 root-logit,
cached-continuation, and Engine/ModelRunner authority artifacts and independent
`PASS` files do not yet all exist.

Fresh CPU validation including both cached-continuation and Engine correctness
authority boundaries:

```text
cached-continuation contract:
  6 tests passed
cached-continuation verifier:
  6 tests passed
cached-continuation producer:
  4 tests passed
Engine correctness contract:
  6 tests passed
Engine correctness verifier:
  4 tests passed
Engine correctness executor:
  9 tests passed
Engine correctness producer:
  6 tests passed
performance contract:
  15 tests passed
performance worker:
  26 tests passed
performance verifier:
  11 tests passed
remote runner:
  20 tests passed
prerequisite builder:
  5 tests passed
cache/memory transport:
  5 tests passed
total:
  123 tests passed
py_compile:
  passed
git diff --check:
  passed
```

### Task 6: Run Smoke and Canonical Performance/Cache Authority

**Files:**
- Create: `experiments/qwen35_hybrid_state/<unique-run-tag>/...`
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify `README.md` only after independent canonical `GO`.

**Interfaces:**
- Consumes: Tasks 1-5.
- Produces: source-bound `GO | NO_GO | INVALID` and durable evidence.

- [ ] **Step 1: Run one reduced paired smoke**

Use W1 only, one excluded warmup, one correctness repetition, and one measured
repetition. Smoke must exercise a real P1 restore hit and exact P0/P1 output
equality. Do not interpret smoke latency.

- [ ] **Step 2: Review smoke evidence**

Require exact artifacts, unique PIDs/ports, four rank exits, cache equations,
capacity parity, and independent smoke verification before canonical launch.

- [ ] **Step 3: Run the frozen canonical matrix**

Run all five workloads, both policies, alternating order, one warmup, one
correctness repetition, and five measured repetitions in isolated processes.

- [ ] **Step 4: Download and independently verify**

Run:

```bash
KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian \
python3 tools/run_qwen35_tp4_hybrid_prefix_benchmark_remote.py \
  verify-only \
  --run-tag "<canonical-run-tag>"
```

Expected: one immutable `GO`, valid `NO_GO`, or `INVALID` result. Only `GO`
authorizes a positive measured claim.

- [ ] **Step 5: Perform a completion audit**

Map every design requirement to:

- raw artifact;
- independent verifier check;
- exact test;
- command output;
- handoff statement.

Treat every missing mapping as incomplete.

- [ ] **Step 6: Update durable documentation**

Always update `AGENT_HANDOFF_STATE.md` with:

- exact run tags and hashes;
- prerequisite identities;
- source/model/workload identities;
- test and verifier results;
- classification;
- what the result proves and does not prove;
- next best optimization branch.

Update `README.md` only for canonical `GO`, copying every number from
`independent_verification.json`. For `NO_GO`, `INVALID`, or either blocked
classification, leave README performance claims unchanged.

## 2026-07-29 Engine Backend Session and Cleanup Receipt Update

The dependency-injected real Engine backend boundary is now implemented in:

```text
tools/qwen35_tp4_engine_backend_session.py
tools/test_qwen35_tp4_engine_backend_session.py
```

It remains CPU-safe under tests and constructs `LLMEngine` only after the
audited `construct_engine` action. The session maps the frozen six-scenario
action plan to:

- publication-runtime configuration;
- exact TP4 rank/process inventory checks;
- frozen source and continuation token payload submission;
- one `LLMEngine.step()` count per model execution;
- independently injected reference output tokens;
- authority-only lifecycle snapshots and baseline deltas;
- all-rank stale-block invalidation and cache clear;
- structured Engine shutdown and cleanup verification.

The backend refuses to copy its own output into
`reference_output_token_ids`. A caller must inject an independent reference
token provider through:

```text
build_real_backend_factory(
  engine_factory=...,
  reference_token_provider=...,
)
```

No provider means fail-closed. This prevents circular correctness evidence.
Seed publication runs before the observation baseline and does not contribute
to measured publication/hit/miss/release counters. Evidence ownership is
single-action: publish commits belong to `verify_publication_commit`, restore
release events belong to `drain_release_events`, and the remaining lifecycle
fields belong to `snapshot_cache`.

Cleanup is now an Engine/ModelRunner transport property rather than a backend
assumption. Every `ModelRunner.exit()` returns:

```text
rank
process_group_destroyed
```

Workers return this receipt over the existing acknowledgement channel.
`LLMEngine.exit()` joins every owned worker and returns:

```text
process_group_destroyed
rank_exit_codes
owned_children_remaining
rank_cleanup_receipts
```

The aggregate cleanup claim is true only when all rank receipts are present,
every rank reports process-group destruction, every child exit code is zero,
and no child remains alive. Missing receipts, nonzero exit codes, live
children, or an unstructured return are rejected.

The source compatibility gate now binds correctness actions to
`qwen35_hybrid_prefix_authority_snapshots(...)`, not the narrower performance
snapshot schema.

Fresh complete CPU-safe validation:

```text
cached-continuation contract:       6 passed
cached-continuation verifier:       6 passed
cached-continuation producer:       4 passed
Engine correctness contract:        6 passed
Engine correctness verifier:        4 passed
Engine correctness executor:       10 passed
Engine correctness producer:        7 passed
Engine backend source contract:      3 passed
Engine backend session:              6 passed
performance contract:               15 passed
performance worker:                 26 passed
performance verifier:               11 passed
remote runner contract:             20 passed
prerequisite builder:                5 passed
cache/authority transport:          11 passed
total:                             140 passed
```

Adjacent local regressions also passed:

```text
hybrid state runtime bridge
model runner live acknowledgement wiring: 11 passed
model runner command acknowledgement:     14 passed
py_compile
git diff --check
```

Current strict boundary:

```text
real backend session implementation:
  present and dependency-injected
real TP4 Engine execution:
  not executed
real exact-four correctness artifact:
  absent
correctness prerequisite bundle:
  incomplete
SSH or GPU workload:
  not executed
latency/throughput/cache/memory/quality benefit:
  unmeasured and not claimable
```

## 2026-07-29 Source-Bound Official Reference Token Authority

The Engine backend no longer depends on an unbound callable for
`reference_output_token_ids`. A source-bound reference authority chain is now
implemented:

```text
tools/qwen35_tp4_engine_reference_tokens.py
tools/qwen35_tp4_engine_reference_tokens_producer.py
tools/verify_qwen35_tp4_engine_reference_tokens.py
tools/qwen35_tp4_engine_official_reference_executor.py
```

The exact-two authority directory contains only:

```text
reference_tokens.json
source_manifest.json
```

An independent verification file must live outside that directory. The
authority and verification bind:

- model-manifest SHA;
- source-tree SHA;
- workload-manifest SHA;
- official reference backend identity;
- greedy generation policy;
- exact five generation scenarios;
- prompt token count and SHA for every scenario;
- generated-token count and exact output token IDs;
- both authority artifact hashes.

The verifier rebuilds the frozen scenario payloads from the current executor
contract and recomputes every prompt hash. It does not trust prompt identities
reported by the producer. Extra files, symlinks, tamper, scenario reordering,
prompt drift, output-length drift, or identity mismatch fail closed.

The atomic producer accepts a separately injected official reference executor,
generates all five references, runs the independent verifier, and publishes
the authority plus external verification only after `PASS`. Existing output
or verification targets are never overwritten.

`build_source_bound_real_backend_factory(...)` now consumes this verified
authority once and shares one immutable provider across the six fresh Engine
sessions. At request time the provider again verifies scenario, prompt hash,
and requested generation length before returning reference tokens.

The official reference executor/backend boundary is implemented with:

```text
AutoModelForCausalLM.from_pretrained(
  local_files_only=True,
  trust_remote_code=False,
  dtype=bfloat16,
  attn_implementation="eager",
)
```

It uses the first frozen GPU in a dedicated spawn worker process. The child
sets `CUDA_VISIBLE_DEVICES` before importing Torch or Transformers, loads the
model once, and serves all five generation requests over a bounded
request/response pipe. The parent requires a ready handshake, structured
responses, zero exit code, and an explicit cleanup receipt. Startup timeout,
generation timeout, worker error, malformed response, or cleanup failure
poisons the backend and kills/joins/closes the worker transport.

Within that worker it loads lazily, verifies the model manifest and every
listed model-file SHA, runs under inference mode, disables sampling, forces equal
`min_new_tokens/max_new_tokens`, validates the returned prompt prefix and
completion length, then explicitly releases the model and CUDA cache.
CPU tests use injected fake Torch/Transformers backends; no real model or GPU
was loaded in this update.

Fresh complete CPU-safe validation:

```text
cached-continuation contract:        6 passed
cached-continuation verifier:        6 passed
cached-continuation producer:        4 passed
Engine correctness contract:         6 passed
Engine correctness verifier:         4 passed
Engine correctness executor:        10 passed
Engine correctness producer:         8 passed
Engine backend source contract:       3 passed
Engine backend session:               6 passed
Engine reference provider:            3 passed
Engine reference verifier:            3 passed
Engine reference producer:            3 passed
official reference executor/backend: 10 passed
performance contract:                15 passed
performance worker:                  26 passed
performance verifier:                11 passed
remote runner contract:              20 passed
prerequisite builder:                 5 passed
cache/authority transport:           11 passed
total:                              160 passed
```

Adjacent regressions:

```text
hybrid state runtime bridge: passed
model runner live acknowledgement wiring: 11 passed
model runner command acknowledgement: 14 passed
py_compile: passed
git diff --check: passed
```

Strict boundary:

```text
official reference producer/executor code:
  implemented
real official M8 reference generation:
  not executed
real source-bound reference authority:
  absent
real TP4 Engine exact-four authority:
  absent
correctness prerequisite bundle:
  incomplete
performance/cache/memory/quality benefit:
  still unmeasured and not claimable
```

## 2026-07-29 Atomic Two-Phase Engine Authority Driver

The source-bound reference and Engine exact-four phases now have one
reproducible entry point:

```text
tools/run_qwen35_tp4_engine_correctness_authority.py
tools/verify_qwen35_tp4_engine_correctness_authority.py
```

The driver accepts one strict `ExecutorConfiguration` JSON and a new output
root. Before constructing any GPU executor it verifies that the configured
model and workload manifests are regular non-symlink files whose real
SHA-256 values match the configuration.

Execution order is fixed:

```text
official single-GPU reference worker
  ->
exact-two reference authority
  ->
external reference independent verification
  ->
source-bound reference provider
  ->
six fresh TP4 Engine sessions
  ->
exact-four Engine authority
  ->
complete-root independent verification
  ->
atomic root publication
```

The final root inventory is exactly:

```text
reference_authority/
reference_independent_verification.json
engine_authority/
authority_summary.json
```

All work occurs under a temporary sibling root. Any reference failure,
Engine failure, verification failure, existing destination, manifest drift,
or cross-phase identity mismatch leaves the final root absent.

The complete-root verifier:

- reruns the reference independent verifier;
- requires the external reference verification JSON to equal the recomputed
  result exactly;
- reruns the Engine exact-four independent verifier;
- requires source/model identity parity across summary and both phases;
- requires workload identity parity between summary and reference phase;
- rejects extra root entries, symlinks, wrong entry types, or non-`PASS`
  summary classifications.

The official reference backend now uses a persistent spawn worker. The child
sets the selected physical GPU before importing Torch/Transformers and reuses
one model load for all five references. Startup, request, response, or cleanup
transport failures poison the backend and kill/join/close the worker.

Fresh complete CPU-safe validation:

```text
cached-continuation contract:         6 passed
cached-continuation verifier:         6 passed
cached-continuation producer:         4 passed
Engine correctness contract:          6 passed
Engine correctness verifier:          4 passed
Engine correctness executor:         10 passed
Engine correctness producer:          8 passed
Engine backend source contract:        3 passed
Engine backend session:                6 passed
Engine reference provider:             3 passed
Engine reference verifier:             3 passed
Engine reference producer:             3 passed
official reference executor/backend:  10 passed
two-phase authority driver:            3 passed
complete-root authority verifier:      2 passed
performance contract:                 15 passed
performance worker:                   26 passed
performance verifier:                 11 passed
remote runner contract:               20 passed
prerequisite builder:                  5 passed
cache/authority transport:            11 passed
total:                               165 passed
```

Adjacent regressions and checks:

```text
hybrid state runtime bridge: passed
model runner live acknowledgement wiring: 11 passed
model runner command acknowledgement: 14 passed
py_compile: passed
git diff --check: passed
```

Strict boundary remains:

```text
two-phase local driver/verifier:
  implemented
real official reference worker run:
  not executed
real TP4 Engine authority run:
  not executed
real correctness artifacts:
  absent
SSH/GPU:
  not used
performance/cache/memory/quality gain:
  unmeasured and not claimable
```

## 2026-07-29 Engine Authority Remote Execution Plan

Add a CPU-only, non-executing authorization builder:

```text
tools/qwen35_tp4_engine_remote_execution_plan.py
tools/test_qwen35_tp4_engine_remote_execution_plan.py
```

The builder consumes the exact configuration bundle and source inventory,
recomputes the owned-source tree and deterministic source tar, rewrites only
the remote runtime paths, and emits one canonical JSON plan. The plan binds:

- `sitian@10.232.195.203`;
- a new safe run tag and unique local/remote destinations;
- source, model-manifest, workload-manifest, configuration, inventory, and tar
  SHA-256 identities;
- the exact four configured GPU indices and configured dist/master ports;
- a staged-source resource guard requiring four unique GPU UUIDs, at least
  `24 * 1024**3` free bytes per GPU, and exactly zero active compute
  processes;
- ordered upload, stage, resource-guard, authority, exact-root packaging,
  download, safe extraction, and local independent-verification commands.

The builder and verifier are pure planning code. They do not import
`subprocess`, open SSH, create remote paths, query GPUs, launch Torch, or run
any emitted command.

Implemented details:

- the configuration builder now uses an Engine-authority-specific owned-source
  inventory rather than the narrower performance-worker inventory;
- the deterministic tar inventory is exactly the sorted source inventory and
  its tree SHA must equal the configuration SHA;
- the output directory contains exactly the plan, rewritten remote
  configuration, and deterministic source tar;
- the remote configuration changes only model/workload runtime paths while
  preserving all source/model/workload identities, TP4 indices, ports, cache
  limits, and timeout;
- command order is frozen as remote reservation, four-file upload, uploaded
  SHA verification/source extraction, lightweight `nvidia-smi` resource
  guard, guarded authority execution, exact-root packaging/download, safe
  local extraction, and local independent verification;
- the resource guard imports neither Torch nor the Engine and rejects missing,
  duplicate, low-memory, or actively used configured GPUs;
- authority cannot be launched through a standalone SSH argv in the plan: the
  emitted `guarded_authority` shell reruns the same `nvidia-smi` guard and
  immediately `exec`s the authority driver only after that final recheck;
- the complete-root verifier now has a read-only positional CLI used by the
  plan;
- local downloaded-artifact verification no longer invokes the mutable current
  checkout: the plan safely extracts the same SHA-bound source tar into a new
  local verifier-source directory, recomputes the source-tree SHA, and invokes
  that snapshot's complete-root verifier;
- the independent plan verifier rehashes every local input and reconstructs
  every command exactly, so identity, GPU, port, path, stage, guard, authority,
  packaging, extraction, verifier-source preparation, or verifier-command
  tampering fails closed.

Fresh CPU-safe validation:

```text
cached-continuation contract:         6 passed
cached-continuation verifier:         6 passed
cached-continuation producer:         4 passed
Engine correctness contract:          6 passed
Engine correctness verifier:          4 passed
Engine correctness executor:         10 passed
Engine correctness producer:          8 passed
Engine backend source contract:        3 passed
Engine backend session:                6 passed
Engine reference provider:             3 passed
Engine reference verifier:             3 passed
Engine reference producer:             3 passed
official reference executor/backend:  10 passed
configuration builder:                 4 passed
remote execution plan/verifier:        5 passed
two-phase authority driver:            3 passed
complete-root authority verifier:      2 passed
performance contract:                 15 passed
performance worker:                   26 passed
performance verifier:                 11 passed
remote runner contract:               20 passed
prerequisite builder:                  5 passed
cache/authority transport:            11 passed
total:                               174 passed
```

Static checks:

```text
py_compile: passed
git diff --check: passed
```

The adjacent Torch-dependent runtime bridge and acknowledgement tests were not
rerun in this local shell because `/usr/bin/python3` cannot import `torch`.
The core 174-test batch completed before that environment-only dependency
boundary; no test assertion failed.

Strict boundary remains:

```text
remote authorization/command plan:
  implemented and independently verified
SSH, scp, remote directory creation, nvidia-smi, or GPU workload:
  not executed
real official M8 reference authority:
  absent
real TP4 Engine exact-four authority:
  absent
real cached-continuation authority:
  absent
correctness prerequisite bundle:
  incomplete
latency/throughput/cache/GPU-memory/compression/quality benefit:
  unmeasured and not claimable
```

## 2026-07-29 Real Prerequisite Provenance and Remote Stage Closure

The benchmark correctness prerequisite boundary is now upgraded from copied
summary documents to a self-contained provenance-bearing bundle:

```text
prerequisite schema:
  qwen35.tp4-performance-prerequisites.v2
provenance schema:
  qwen35.tp4-performance-prerequisite-provenance.v1
```

`qwen35_tp4_real_prerequisite_authority_adapter.py` read-only adapts complete
production authority directories. It invokes the real independent verifier
for all three authorities and verifies cached/Engine remote plan,
authorization, and receipt chains. The final builder copies provenance plus
the referenced plan, consumed authorization, and receipt evidence into the
bundle. The runtime contract independently validates every path and SHA.

Root-logit remains honestly different:

```text
binding_kind:
  complete_directory_only
root_logit_receipt_gap:
  true
plan/authorization/receipt:
  null
```

The canonical 11-step benchmark plan now transports the complete bundle. It
builds and freezes `correctness_prerequisites.tar`, uploads it alongside the
source tar, validates its SHA and exact regular-file inventory remotely,
safely extracts it to the one prerequisite path used by all 70 workers, and
rehashes the staged main JSON. The local plan verifier independently opens
the tar and requires its real inventory to equal
`prerequisites_owned_files`.

Fresh CPU-safe evidence:

```text
focused affected gate:                116 passed across 12 files
expanded selected authority gate:     288 passed across 41 files
focused py_compile:                    passed
embedded command-script compile:       passed
git diff --check:                      passed
staged files:                          0
```

Completion audit:

1. **No accuracy regression**
   - Structural authority: complete-directory adapter, v2 provenance, semantic
     validators, and receipt evidence transport are implemented.
   - Missing artifact: real root-logit, cached-continuation, and Engine
     correctness bundle.
2. **Faster inference**
   - Structural authority: canonical 70-case worker/assembler/verifier and
     single-use 11-step execution protocol are implemented.
   - Missing artifact: real independently verified TP4 latency/throughput run.
3. **Less cache / physical memory**
   - Structural authority: all-rank cache and CUDA allocator schemas and
     independent reconstruction are implemented.
   - Missing artifact: real all-rank cache/allocator observations.
4. **Safe remote execution**
   - Authorization, consumed receipt chain, dual immutable upload, resource
     guards, safe extraction, remote verification, and frozen-source local
     verification are implemented and CPU-tested.
   - Execution remains intentionally unperformed.

The project objective is therefore not complete. The next evidence-producing
step is to run the three real correctness authorities on the approved remote
server, build and verify the v2 prerequisite bundle, and only then issue a
single-use canonical benchmark authorization.

## 2026-07-29 Benchmark Remote Execution Protocol Primitives

A prompt-to-artifact completion audit separated the long-term objective into
three required authorities:

```text
no accuracy regression:
  real root-logit + cached-continuation + Engine prerequisite bundle
faster inference:
  real canonical 70-case latency/throughput artifact
less cache / physical memory:
  real all-rank cache and CUDA allocator evidence
```

All three remain incomplete. The closest structural blocker was that the
benchmark runner produced a data-only READY launch plan but had no single-use
execution authorization, bounded execution receipt, or dependency-injected
executor. Starting the 70 GPU cases from that state would bypass the remote
authority safety model already established for Engine and cached-continuation
correctness.

Added:

```text
tools/qwen35_tp4_hybrid_prefix_benchmark_remote_execution_authorization.py
tools/qwen35_tp4_hybrid_prefix_benchmark_remote_execution_receipt.py
tools/qwen35_tp4_hybrid_prefix_benchmark_remote_execution_executor.py
```

The authorization binds:

- canonical launch-plan SHA and run tag;
- prerequisite, benchmark source, model, and workload identities;
- exact four GPU indices;
- the ordered 70-case inventory and all 140 unique dist/master ports;
- a safe operator nonce and explicit unconsumed state.

Consumption atomically renames the active record before rewriting it as
consumed. Reuse, cross-directory claims, identity drift, duplicate ports,
non-canonical case count, and unsafe nonces fail closed.

The execution receipt requires the exact frozen 11-step order:

```text
reserve_remote
upload
stage
resource_guard
workers
assembly
remote_verify
final_resource_guard
package_download
safe_extract
local_verify
```

It validates command hashes, bounded logs, all 70 worker identities, exact
`280` case rows and `70` process rows, package SHA/size, preflight/final GPU
UUID stability, zero active compute processes, and byte-for-byte canonical
remote/local verification payload equality. A verified `NO_GO` benchmark is a
valid execution PASS and remains `NO_GO`; the protocol does not discard or
mislabel negative performance evidence.

The offline executor:

- has no default runner and contains no subprocess import;
- requires exact
  `KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian`;
- consumes authorization before the first injected command;
- rejects pre-existing receipt, failure, package, extraction, or consumed
  authorization targets;
- publishes only bounded prefix-preserving FAILED evidence after any command
  error;
- reuses the existing audited
  `qwen35_tp4_engine_remote_subprocess_adapter.py` as the single `Popen`
  owner.

The new modules are included in both the deterministic benchmark source bundle
and the shared AST execution-source contract.

Strict TDD evidence:

```text
authorization RED:                     missing module FileNotFoundError
receipt RED:                           missing module FileNotFoundError
executor RED:                          missing module FileNotFoundError
source-contract RED:                   new modules absent from AST inventory
source-bundle RED:                     new modules absent from owned sources
benchmark authorization:               5 passed
benchmark receipt:                     5 passed
benchmark executor:                    4 passed
shared execution source contract:      4 passed
shared subprocess adapter:             7 passed
benchmark remote runner:              20 passed
focused protocol gate:                45 passed
expanded selected authority gate:    275 passed across 39 files
focused py_compile:                    passed
```

No SSH, `scp`, `nvidia-smi`, remote directory creation, subprocess invocation,
Torch, Transformers, CUDA, model load, Engine construction, or GPU workload
was executed.

Remaining nearest blocker:

```text
benchmark 11-step executable plan builder/verifier:
  absent
benchmark-specific safe package/extract/local-verifier commands:
  absent
real correctness prerequisite bundle:
  absent
real canonical benchmark:
  not run
```

Therefore the protocol primitives are not yet authorization to launch the
remote benchmark, and no latency, throughput, cache, GPU-memory, compression,
quality, or accuracy improvement is claimable.

## 2026-07-29 Benchmark 11-Step Execution Plan Authority

Added and independently verified:

```text
tools/qwen35_tp4_hybrid_prefix_benchmark_remote_execution_plan.py
tools/test_qwen35_tp4_hybrid_prefix_benchmark_remote_execution_plan.py
```

The builder freezes the launch plan into the exact receipt-compatible order:

```text
reserve_remote
upload
stage
resource_guard
workers
assembly
remote_verify
final_resource_guard
package_download
safe_extract
local_verify
```

The emitted plan is self-contained. It copies the deterministic source tar,
correctness prerequisites, model manifest, and canonical assembly metadata
into the plan directory, then rewrites all local command paths to those frozen
copies. Removing the original builder inputs no longer invalidates the plan.

Source-bound local verification does not import from the active checkout. The
single `local_verify` command:

1. checks the frozen source-tar SHA;
2. rejects duplicate, unsorted, linked, non-file, absolute, or parent-escaping
   tar members;
3. safely extracts into a new plan-local verifier tree;
4. recomputes the canonical source-tree SHA over member names and bytes;
5. loads `verify_qwen35_tp4_hybrid_prefix_benchmark.py` only from that frozen
   verified tree;
6. verifies the downloaded artifact and emits the provenance-complete
   canonical verification envelope.

The plan verifier independently rechecks the copied input hashes, source-tar
inventory, source-tree identity, launch-plan bindings, metadata bytes, and all
11 deterministic command shapes. The existing Engine resource guard remains
unchanged: exact GPUs `0,1,2,3`, at least `24 GiB` free per GPU, and no active
compute processes at both guard points.

Strict TDD evidence:

```text
self-contained-plan RED:
  ValueError: source tar must be a regular file
source-tree-binding RED:
  source tree identity mismatch was accepted
execution plan suite:                  6 passed
benchmark authorization:               5 passed
benchmark receipt:                     5 passed
benchmark executor:                    4 passed
shared execution source contract:      4 passed
shared subprocess adapter:             7 passed
benchmark remote runner:              20 passed
focused benchmark protocol gate:      51 passed
expanded selected authority gate:    281 passed across 40 files
embedded command-script compile:       passed
```

The plan module is now included in both the deterministic benchmark source
bundle and the shared AST execution-source contract. No second subprocess
surface was added; the existing audited Engine adapter remains the sole
`Popen` owner.

No SSH, `scp`, `nvidia-smi`, remote directory creation, subprocess invocation,
Torch, Transformers, CUDA, model load, Engine construction, or GPU workload
was executed while producing this authority.

Current boundary:

```text
benchmark execution plan builder/verifier:
  implemented and CPU-verified
safe frozen-source local verifier:
  implemented and CPU-verified
real correctness prerequisite bundle:
  absent
real canonical 70-case benchmark:
  not run
latency/throughput/cache/GPU-memory/compression/quality/accuracy benefit:
  unmeasured and not claimable
```

## 2026-07-29 Real LLMEngine Benchmark Adapter

The benchmark worker no longer stops at an unimplemented default runtime
loader. It now lazily constructs:

```text
tools/qwen35_tp4_hybrid_prefix_benchmark_engine_adapter.py
  BenchmarkEngineAdapter
```

through the production `tinyvllm.engine.llm_engine.LLMEngine` and
`tinyvllm.sampling_params.SamplingParams` APIs. The import remains lazy so all
contract and orchestration tests stay CPU-safe and do not import Torch.

The adapter:

- forwards the frozen TP4 Engine configuration, including
  `max_model_len=4352`, `max_num_batched_tokens=17408`, and
  `max_num_seqs=8`;
- binds each admitted request to its real sequence identity by taking the
  `scheduler.waiting` set difference before and after `add_request()`;
- derives first-token, end-to-end, and decode-step timing only from
  `last_step_observation.step_end_ns`, excluding source publication from
  continuation admission timing;
- rejects missing, malformed, pre-admission, or non-monotonic step timestamps
  before benchmark rows can be emitted;
- derives executed prefill from the exact observed
  `prefill_chunk_start`/`prefill_chunk_end` ranges and computes restored/reused
  tokens from that execution evidence;
- checks every completion delta against final Engine outputs;
- records rank-zero step logits in the same sample-row order used by
  `ModelRunner._select_sample_rows()` and only associates logits with rows that
  emitted a completion token;
- runs W1/W2 continuations sequentially, admits all eight W3 continuations
  before the first step, and applies W4 token mismatch, stale-block
  invalidation, and cache-clear miss controls after a fresh source publication
  for each request;
- forwards all-rank memory/cache snapshots and the scheduler-visible capacity
  snapshot;
- accepts cleanup only when all four rank receipts, exit codes, child-process
  inventory, and process-group destruction evidence match the production
  `LLMEngine.exit()` contract.

The two public capacity increases are required by the frozen workloads:

```text
W2 total sequence length:
  4096 shared-prefix + 64 suffix + 64 generated = 4224
W3 single admission:
  8 * (2048 shared-prefix + 64 suffix) = 16896 prefill tokens
```

The old defaults of 4096 model length and 16384 batched tokens could not admit
those canonical cases. The selected values leave a small explicit bound rather
than silently changing the workload.

TDD and source-audit evidence:

```text
non-monotonic timestamp RED:
  adapter accepted step_end_ns=800 after admission_ns=910
minimal timestamp guard GREEN:
  adapter rejects step timestamps before admission or earlier than the prior
  step
Engine adapter:                         9 passed
benchmark worker:                      27 passed
benchmark assembler:                    5 passed
benchmark verifier:                    11 passed
remote runner:                         20 passed
complete selected authority gate:     261 passed across 36 files
```

The source audit additionally confirmed:

```text
logits row order:
  production ModelRunner filters the scheduled seq list in-order
W4 invalidation API:
  exact timeout-bound production call
cleanup receipt:
  exact four-rank LLMEngine.exit schema
```

No SSH, `scp`, `nvidia-smi`, remote directory creation, Torch/Transformers
import, CUDA initialization, model load, Engine construction, Engine step, or
GPU workload was executed during this adapter validation.

Strict boundary:

```text
real production API benchmark adapter:
  implemented and CPU-tested through dependency-injected fake Engine
real TP4 adapter execution:
  not run
real correctness prerequisite bundle:
  absent
canonical performance/cache authority:
  not run
latency/throughput/cache/GPU-memory/compression/quality benefit:
  unmeasured and not claimable
```

## 2026-07-29 Canonical Benchmark Artifact Assembler

Completion audit found that the 70 single-case workers and the independent
verifier did not yet have a producer boundary between them. Each worker wrote
only a local `case_rows.jsonl`, `process_rows.jsonl`, optional logits, and
`summary.json`; no component could construct the verifier's frozen top-level
and nested artifact inventory. Therefore, successful worker processes still
could not produce a canonical benchmark authority.

Added:

```text
tools/qwen35_tp4_hybrid_prefix_benchmark_assembler.py
tools/test_qwen35_tp4_hybrid_prefix_benchmark_assembler.py
```

The assembler:

- requires the exact 70-case directory inventory and canonical matrix order;
- validates every worker summary, exact case/process row schema and count,
  case identity, source/model/workload/prerequisite provenance, output-token
  hash, and correctness-logits SHA;
- rejects worker failure files, missing completion markers, tracebacks,
  missing/extra cases, duplicate logits paths, and non-canonical command/port
  inventories;
- revalidates and copies the complete three-authority prerequisite bundle
  rather than trusting a producer classification;
- copies every worker log and correctness logits into the frozen nested
  inventory;
- emits `source_manifest.json`, `environment.json`,
  `gpu_assignments.json`, `commands.json`, aggregate rows, manifests,
  untrusted producer summary, and the exact artifact manifest;
- publishes the entire run directory with one atomic rename and leaves no
  target after validation failure;
- supports both a direct Python API and an explicit-file CLI.

The worker now prints
`QWEN35_TP4_BENCHMARK_WORKER_COMPLETE` only after all successful atomic
case outputs are written. The assembler and independent verifier both require
that real worker marker; the orchestration layer does not synthesize it.

The READY launch plan now:

- includes the assembler in the deterministic benchmark source tar;
- assigns one unique log path to every case command;
- freezes source/environment/GPU/command/log metadata files;
- carries an explicit assembler command producing the canonical remote
  artifact directory.

Fresh CPU-safe evidence:

```text
assembler RED:
  FileNotFoundError for the missing assembler module
worker marker RED:
  successful CLI produced no required completion marker
assembler:                              5 passed
benchmark worker:                      26 passed
benchmark verifier:                    11 passed
remote runner:                         20 passed
extended selected authority gate:     251 passed across 35 files
focused py_compile:                    passed
git diff --check:                      passed
staged files:                          0
```

One assembler fixture splits a complete canonical synthetic run into 70
worker directories, reassembles it through the production API/CLI, and then
requires the independent verifier to reconstruct `GO`. Negative fixtures
prove missing cases, traceback logs, and source-provenance drift publish no
output.

No SSH, `scp`, `nvidia-smi`, remote directory creation, subprocess adapter,
Torch, Transformers, CUDA, model load, Engine run, or GPU workload was
executed.

Strict boundary:

```text
70-case canonical artifact assembly:
  implemented and CPU-tested
worker completion log contract:
  implemented and CPU-tested
READY launch-plan finalizer metadata:
  implemented as data only
real benchmark Engine adapter:
  absent; _default_runtime_loader remains fail-closed
real correctness prerequisite bundle:
  absent
canonical performance/cache authority:
  not run
latency/throughput/cache/GPU-memory/compression/quality benefit:
  unmeasured and not claimable
```

## 2026-07-29 Cached-Continuation Remote Authority Protocol

Added the independent cached-specific remote execution chain:

```text
tools/qwen35_tp4_cached_continuation_remote_execution_plan.py
tools/qwen35_tp4_cached_continuation_remote_execution_receipt.py
tools/qwen35_tp4_cached_continuation_remote_execution_executor.py
```

The plan binds the immutable configuration/source/model/workload identities,
exact TP4 GPU indices and ports, `sitian@10.232.195.203`, the strict
no-active-compute-process resource guard, the standalone cached authority
driver, and an exact two-entry downloaded package:

```text
cached_continuation_authority
cached_continuation_independent_verification.json
```

Safe extraction rejects absolute paths, traversal, symlinks, hard links, or
package-root drift. The local verification command loads the verifier only
from the independently reconstructed immutable source tree, recomputes the
exact-five result, and requires byte-semantically equal canonical payloads
from remote and local verification.

The cached receipt intentionally does not accept or require the Engine
two-phase `reference_classification` or `engine_classification` fields. It
requires the cached verification fields:

```text
classification
schema_version
source_tree_sha256
model_manifest_sha256
workload_manifest_sha256
checks
```

It also binds command hashes, nonzero package SHA/size, consumed authorization
SHA/nonce, and identical preflight/final GPU index-to-UUID mappings.

The process-free executor requires an injected runner, exact
`KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian`, independent plan
verification before authorization consumption, non-existing local outputs,
and frozen command order. PASS receipts and bounded FAILED evidence both bind
the consumed authorization. The AST safety contract and immutable authority
source inventory now cover all three cached modules. The only `Popen` owner
remains the existing isolated adapter.

Fresh CPU-safe evidence:

```text
cached remote plan:                     4 passed
cached remote receipt:                  5 passed
cached remote executor:                 5 passed
execution source AST contract:          4 passed
configuration builder:                  4 passed
complete selected authority gate:     244 passed across 34 files
focused py_compile:                    passed
git diff --check:                      passed
staged files:                          0
```

The exact 34-file command inventory is recorded in
`2026-07-29-qwen35-tp4-cached-continuation-remote-authority.md`.

No SSH, `scp`, `nvidia-smi`, remote directory creation, subprocess adapter,
Torch, Transformers, CUDA, model load, Engine run, or GPU workload was
executed.

Strict boundary:

```text
independent cached remote protocol:
  implemented and CPU-tested
real cached exact-five artifact:
  absent
real TP4 Engine exact-four authority:
  absent
correctness prerequisite bundle:
  incomplete
canonical performance/cache authority:
  not run
latency/throughput/cache/GPU-memory/compression/quality benefit:
  unmeasured and not claimable
```

## 2026-07-29 Cached-Continuation Real-Engine Wiring

The previously missing CPU-testable wiring now exists:

```text
configured producer
  -> ordered 19-row CachedContinuationExecutor
  -> fresh CachedContinuationBackendSession per row
  -> independent official Transformers greedy reference
  -> real LLMEngine-shaped source publication / restore / invalidation
  -> rank-zero raw step-logit comparison
  -> structured all-rank cleanup evidence
```

The official reference backend now has a stepwise API that returns both exact
completion token IDs and one raw `[1, vocab]` logit tensor per generated
token. It runs a direct cached model forward for the prompt, then feeds one
greedy token at a time with `past_key_values`. This avoids processed
generation `scores` and avoids depending on the newer Transformers
`output_logits` generate API. EOS remains an ordinary token during the
fixed-length authority loop.

The process-isolated reference worker transports the step logits through the
existing spawn boundary. For cached-continuation rows, the reference process
is constructed, queried, and closed before the TP4 Engine is constructed.
This prevents the independent reference model from retaining GPU 0 while the
four-rank Engine attempts to reserve the exact four configured GPUs.

The cached session proves:

- source publication before every continuation;
- W1/W2/W3 hits and W4 token/stale/clear misses from all-rank counter deltas;
- executed prefill tokens from positive `LLMEngine.step()` token deltas;
- completion tokens against the independent official reference;
- raw rank-zero step logits at `atol=2e-5`, `rtol=0`;
- process-group destruction and no owned child leakage.

The producer now exposes `produce_configured_authority(...)`. It constructs
the official reference executor factory by default, builds the real cached
session factory, and publishes only after the existing independent cached
verifier passes. The old parameterless default executor remains fail-closed;
there is still no import-time execution or implicit remote/GPU launch.

The cached contract, executor, backend, producer, and verifier are included in
the immutable authority source inventory.

Fresh CPU-safe validation:

```text
cached-continuation contract:           6 passed
cached-continuation verifier:           6 passed
cached-continuation backend session:     5 passed
cached-continuation executor:            6 passed
cached-continuation producer:            7 passed
official reference executor/backend:    13 passed
model-runner live logits ack wiring:     14 passed
extended complete selected gate:       237 passed across 31 files
focused py_compile:                   passed
git diff --check:                     passed
staged files:                         0
```

No real `subprocess.Popen`, SSH, `scp`, `nvidia-smi`, remote directory,
Transformers model load, Torch CUDA allocation, TP4 Engine, or GPU workload
was executed.

Strict claim boundary:

```text
cached producer -> executor -> backend -> stepwise official reference:
  implemented and CPU-tested with fake runtime/model/process boundaries
real cached-continuation artifact:
  absent
real official M8 step-logit reference:
  absent
real TP4 Engine exact-four authority:
  absent
canonical performance/cache authority:
  not run
latency/throughput/cache/GPU-memory/compression/quality/accuracy benefit:
  unmeasured and not claimable
```

## 2026-07-29 Cached-Continuation Standalone Authority Driver

Added:

```text
tools/run_qwen35_tp4_cached_continuation_authority.py
tools/test_run_qwen35_tp4_cached_continuation_authority.py
```

The cached-continuation chain now has an explicit standalone execution entry:

```text
python3 tools/run_qwen35_tp4_cached_continuation_authority.py \
  --configuration <bundle>/executor_configuration.json \
  --source-inventory <bundle>/source_inventory.json \
  --output-dir <new-exact-five-directory> \
  --verification-path <new-external-verification.json>
```

Before constructing the reference backend or Engine, the driver reuses the
Engine authority configuration loader to verify:

- exact configuration schema and TP4 topology;
- model-manifest file identity;
- frozen workload-manifest file identity;
- immutable source-inventory schema and tree SHA;
- cached contract workload SHA equality.

The driver runs the configured cached producer in a sibling temporary
directory, requires producer `PASS`, writes the independent verification
outside the exact-five directory, and publishes both requested outputs only
after verification `PASS`. Existing outputs and run-internal verification
paths are rejected before execution. Producer or verifier failure removes all
temporary and partially published output.

The driver is included in `AUTHORITY_OWNED_SOURCE_PATHS`; therefore the
configuration bundle's source-tree SHA now covers the callable cached
authority entrypoint itself.

Fresh CPU-safe validation:

```text
standalone cached authority driver:      4 passed
extended selected authority gate:      241 passed across 32 files
focused py_compile:                    passed
git diff --check:                      passed
```

No real reference model, CUDA allocation, TP4 Engine, SSH, subprocess adapter,
or remote workload was executed.

Current boundary:

```text
local/remote-callable cached authority CLI:
  implemented and CPU-tested
remote orchestration plan for this standalone CLI:
  absent
real exact-five cached authority:
  absent
cached_continuation prerequisite PASS:
  absent
performance/cache/accuracy benefit:
  unmeasured and not claimable
```

### Cached Step-Logit Semantic Corrections

The official reference and Engine collection paths were tightened after
implementation-level review:

- the direct Transformers forward now passes `return_dict=True` and exact
  `cache_position` values, matching the repository's existing Qwen3.5
  `ReferenceStateAdapter` contract;
- prompt positions are `0..prompt_length-1`, then each cached decode receives
  the next single absolute position;
- reference logits stay shaped `[1, vocab]`, matching the Engine's selected
  single-request sample row;
- Engine collection no longer assumes logits exist only on negative
  decode-token steps;
- the first completion token can be sampled on the final positive prefill
  step, so collection uses `last_step_observation.do_sample` plus the actual
  completion-token delta.

The new prefill-sample regression would fail with one missing logit row under
the previous implementation.

Fresh CPU-safe validation:

```text
cached backend session:                  6 passed
official reference executor/backend:    13 passed
extended selected authority gate:      242 passed across 32 files
focused py_compile:                    passed
git diff --check:                      passed
```

### Single-Load Official Reference Corpus

The standalone cached authority no longer reloads the official Transformers
model once per continuation row. Before constructing any TP4 Engine session,
the stateful session factory now:

1. lazily constructs one process-isolated official reference executor;
2. computes the frozen 19-row W1/W2/W3/W4 reference corpus;
3. stores completion IDs and cloned CPU raw logits keyed by
   `(workload, request_index)`;
4. closes the reference executor and releases its GPU allocation;
5. only then permits fresh TP4 Engine sessions to start.

Every lookup revalidates the exact frozen prompt and generated-token count.
Configuration mismatch is rejected before loading the reference model.
Reference failure closes the executor, leaves no partial corpus, and permits a
clean retry.

This reduces authority setup from nineteen model loads to one. It is an
authority-execution resource improvement, not measured production inference
latency or throughput.

Fresh CPU-safe validation:

```text
cached backend session:                  8 passed
extended selected authority gate:      244 passed across 32 files
focused py_compile:                    passed
git diff --check:                      passed
```

## 2026-07-29 Isolated Subprocess Adapter

Added:

```text
tools/qwen35_tp4_engine_remote_subprocess_adapter.py
tools/test_qwen35_tp4_engine_remote_subprocess_adapter.py
```

The previously absent local process boundary is now implemented as a separate
explicit runner adapter. The process-free plan, authorization, executor, and
receipt modules remain free of subprocess execution. The adapter:

- has no CLI, `main`, `__main__`, import-time execution, or executor default;
- accepts only `ssh`, `scp`, and the exact current local Python executable;
- requires exactly
  `KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian`;
- invokes the injected `Popen` factory with `shell=False`;
- captures normal stdout/stderr through bounded UTF-8 file-backed logs;
- streams `package_download` stdout to a new binary file;
- independently computes successful package SHA-256 and byte size;
- deletes partial package output on nonzero return, process creation failure,
  invalid/oversized stderr, or empty output.

The executor package protocol was corrected so failed downloads do not need a
success-only package identity. Nonzero package results are preserved as
authorization-bound FAILED evidence; only zero-return downloads must provide
and pass independent file SHA/size verification.

The adapter is now included in the immutable Engine authority source
inventory. The AST source contract keeps subprocess forbidden in core modules
while separately proving that the adapter exposes only `run_command`, has no
CLI, never uses `shell=True`, and calls no subprocess helper other than
`Popen`.

Fresh complete CPU-safe validation:

```text
cached-continuation contract:           6 passed
cached-continuation verifier:           6 passed
cached-continuation producer:           4 passed
Engine correctness contract:            6 passed
Engine correctness verifier:            4 passed
Engine correctness executor:           10 passed
Engine correctness producer:            8 passed
Engine backend source contract:          3 passed
Engine backend session:                  6 passed
Engine reference provider:               3 passed
Engine reference verifier:               3 passed
Engine reference producer:               3 passed
official reference executor/backend:    10 passed
configuration builder:                   4 passed
remote execution plan/verifier:          6 passed
remote execution authorization:          4 passed
remote execution receipt:                5 passed
offline execution executor:             11 passed
execution source AST contract:           4 passed
remote subprocess adapter:               7 passed
two-phase authority driver:              3 passed
complete-root authority verifier:        2 passed
performance contract:                   15 passed
performance worker:                     26 passed
performance verifier:                   11 passed
remote runner contract:                 20 passed
prerequisite builder:                    5 passed
cache/authority transport:              11 passed
total:                                 206 passed
```

Static checks:

```text
focused py_compile: passed
git diff --check: passed
staged files: 0
```

Strict claim boundary:

```text
subprocess adapter implementation:
  complete and CPU-tested through an injected fake Popen factory
real adapter invocation:
  not performed
SSH/scp/nvidia-smi/remote directory/GPU workload:
  not executed
real official M8 reference authority:
  absent
real TP4 Engine exact-four authority:
  absent
real cached-continuation authority:
  absent
correctness prerequisite bundle:
  incomplete
latency/throughput/cache/GPU-memory/compression/quality benefit:
  unmeasured and not claimable
```

### Remote Execution Receipt and Offline Executor

Added CPU-only modules:

```text
tools/qwen35_tp4_engine_remote_execution_receipt.py
tools/test_qwen35_tp4_engine_remote_execution_receipt.py
tools/qwen35_tp4_engine_remote_execution_executor.py
tools/test_qwen35_tp4_engine_remote_execution_executor.py
tools/qwen35_tp4_engine_remote_execution_source_contract.py
tools/test_qwen35_tp4_engine_remote_execution_source_contract.py
```

The execution receipt contract requires:

- exact canonical plan SHA and run tag;
- exact frozen step order and command SHA for every step;
- zero return code and bounded stdout/stderr for every completed step;
- four configured, unique, idle GPUs in both preflight and the final
  `guarded_authority` resource marker;
- identity-stable authority and immutable-local-verifier `PASS` JSON;
- independently recomputed package tar SHA and byte size;
- no extra receipt fields or partial-success classification.

`guarded_authority` now emits a unique
`QWEN35_FINAL_RESOURCE_JSON=<json>` marker immediately before `exec`. Ordinary
authority logs may be present; the receipt verifier finds exactly one final
resource marker and the final valid PASS JSON. It permits free-memory drift
while requiring identical GPU index/UUID identities and zero compute
processes.

The offline executor has no subprocess implementation or default runner. It:

- requires an explicitly injected command runner;
- verifies a plan file before any command;
- requires exactly
  `KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian`;
- rejects existing receipt, failure, package, extracted-authority, or local
  verifier-source destinations before remote reservation;
- maps upload fan-out, guarded authority, and binary package download without
  trusting package SHA/size reported by the runner;
- atomically publishes a PASS receipt only after independent validation;
- atomically preserves bounded `FAILED` evidence with exact completed-prefix
  command receipts when a step fails.

The source AST gate proves that plan/receipt/executor modules import no
`subprocess`, call no process-execution APIs, and that the executor has no
`main`/`__main__` execution surface. It also proves both executor entrypoints
require explicit runner/verifier/environment authorities. Plan and receipt
retain only data-builder/read-only-verifier CLIs.

Fresh CPU-safe validation:

```text
cached-continuation contract:          6 passed
cached-continuation verifier:          6 passed
cached-continuation producer:          4 passed
Engine correctness contract:           6 passed
Engine correctness verifier:           4 passed
Engine correctness executor:          10 passed
Engine correctness producer:           8 passed
Engine backend source contract:         3 passed
Engine backend session:                 6 passed
Engine reference provider:              3 passed
Engine reference verifier:              3 passed
Engine reference producer:              3 passed
official reference executor/backend:   10 passed
configuration builder:                  4 passed
remote execution plan/verifier:         6 passed
remote execution receipt:               5 passed
offline execution executor:             9 passed
execution source AST contract:          3 passed
two-phase authority driver:             3 passed
complete-root authority verifier:       2 passed
performance contract:                  15 passed
performance worker:                    26 passed
performance verifier:                  11 passed
remote runner contract:                20 passed
prerequisite builder:                   5 passed
cache/authority transport:             11 passed
total:                                192 passed
```

Static checks:

```text
py_compile: passed
git diff --check: passed
```

No process runner, SSH, `scp`, `nvidia-smi`, remote path, Torch load, or GPU
workload was executed.

## 2026-07-29 Reproducible Engine Authority Configuration

The two-phase driver no longer requires a hand-written fourteen-field
configuration. A CPU-safe builder now produces the exact configuration bundle:

```text
tools/build_qwen35_tp4_engine_authority_configuration.py
```

Its output inventory is exactly:

```text
executor_configuration.json
workload_manifest.json
source_inventory.json
```

The builder reuses the benchmark runner's deterministic owned-source file
inventory and tree-hash implementation. It does not invent another source
identity algorithm. It writes the benchmark contract's canonical workload
manifest, hashes the real user-provided model manifest, validates all values
through `ExecutorConfiguration`, and publishes the bundle atomically.

The authority driver now requires the source inventory on its CLI:

```text
python3 tools/run_qwen35_tp4_engine_correctness_authority.py \
  --configuration <bundle>/executor_configuration.json \
  --source-inventory <bundle>/source_inventory.json \
  --output-root <new-authority-root>
```

Before any worker or Engine construction, the driver verifies:

- exact configuration schema and TP4 world size;
- model manifest is a regular non-symlink file with matching SHA;
- workload manifest is a regular non-symlink file with matching SHA;
- source inventory exact schema;
- non-empty unique safe relative source paths;
- source inventory tree SHA equals the configuration tree SHA.

Any builder failure leaves no partial output, and existing bundle/authority
targets are never overwritten.

Fresh complete CPU-safe validation:

```text
cached-continuation contract:         6 passed
cached-continuation verifier:         6 passed
cached-continuation producer:         4 passed
Engine correctness contract:          6 passed
Engine correctness verifier:          4 passed
Engine correctness executor:         10 passed
Engine correctness producer:          8 passed
Engine backend source contract:        3 passed
Engine backend session:                6 passed
Engine reference provider:             3 passed
Engine reference verifier:             3 passed
Engine reference producer:             3 passed
official reference executor/backend:  10 passed
configuration builder:                 4 passed
two-phase authority driver:            3 passed
complete-root authority verifier:      2 passed
performance contract:                 15 passed
performance worker:                   26 passed
performance verifier:                 11 passed
remote runner contract:               20 passed
prerequisite builder:                  5 passed
cache/authority transport:            11 passed
total before remote-plan addition:    169 passed
```

Adjacent regressions and checks remain green:

```text
hybrid state runtime bridge
model runner live acknowledgement wiring: 11 passed
model runner command acknowledgement: 14 passed
py_compile
git diff --check
```

This removes configuration identity ambiguity only. No real reference,
TP4 Engine, performance, cache, memory, compression, or quality result was
created.

## 2026-07-29 Single-Use Remote Execution Authorization

Added:

```text
tools/qwen35_tp4_engine_remote_execution_authorization.py
tools/test_qwen35_tp4_engine_remote_execution_authorization.py
```

The CPU-safe execution boundary now requires a canonical single-use
authorization before the first injected runner call. The authorization binds:

- canonical plan SHA and run tag;
- exact source-tree, model-manifest, and workload-manifest SHA values;
- exact four GPU indices and dist/master ports;
- a safe operator-provided nonce;
- an explicit unconsumed state.

`consume_authorization()` validates the active record, requires active and
consumed paths in one directory, atomically renames the active file to claim
it, and only then rewrites the claimed record with `consumed: true`. Reuse,
cross-directory consumption, pre-existing consumed output, identity
tampering, and model-manifest drift are rejected.

The verified-plan executor consumes this authorization before the first
runner invocation. Both PASS receipts and bounded FAILED evidence bind the
consumed authorization SHA and nonce. The authorization module is included in
the immutable source inventory and the AST safety gate; no subprocess adapter
or default runner was introduced.

Fresh complete CPU-safe validation:

```text
cached-continuation contract:           6 passed
cached-continuation verifier:           6 passed
cached-continuation producer:           4 passed
Engine correctness contract:            6 passed
Engine correctness verifier:            4 passed
Engine correctness executor:           10 passed
Engine correctness producer:            8 passed
Engine backend source contract:          3 passed
Engine backend session:                  6 passed
Engine reference provider:               3 passed
Engine reference verifier:               3 passed
Engine reference producer:               3 passed
official reference executor/backend:    10 passed
configuration builder:                   4 passed
remote execution plan/verifier:          6 passed
remote execution authorization:          4 passed
remote execution receipt:                5 passed
offline execution executor:             10 passed
execution source AST contract:           3 passed
two-phase authority driver:              3 passed
complete-root authority verifier:        2 passed
performance contract:                   15 passed
performance worker:                     26 passed
performance verifier:                   11 passed
remote runner contract:                 20 passed
prerequisite builder:                    5 passed
cache/authority transport:              11 passed
total:                                 197 passed
```

Static checks:

```text
focused py_compile: passed
git diff --check: passed
staged files: 0
```

The complete gate directly executed 27 custom test files; `python -m
unittest` is not a valid collector for these file-local `_run()` entrypoints.
No SSH, `scp`, `nvidia-smi`, remote directory creation, Torch load, remote
process, or GPU workload was executed.

Strict claim boundary remains:

```text
single-use execution authorization:
  implemented and independently verified under dependency injection
actual subprocess/SSH adapter:
  absent
real official M8 reference authority:
  absent
real TP4 Engine exact-four authority:
  absent
real cached-continuation authority:
  absent
correctness prerequisite bundle:
  incomplete
latency/throughput/cache/GPU-memory/compression/quality benefit:
  unmeasured and not claimable
```

## 2026-07-29 Performance Prerequisite Trust-Chain Hardening

A completion audit found that the prerequisite builder and runtime contract
previously trusted only a top-level `classification == "PASS"` plus model
identity. Therefore, the following forged two-field document could be copied
into all three authority slots and incorrectly authorize the benchmark:

```json
{
  "classification": "PASS",
  "model_manifest_sha256": "<expected sha256>"
}
```

This authorization path is now closed. The benchmark contract owns one
shared authority-specific validator:

```text
validate_authority_documents(
  name,
  artifact,
  verification,
  source_tree_sha256,
)
```

It fail-closes on exact schemas and semantic evidence:

- root-logit requires the frozen source SHA, exact three case IDs and four
  ranks, decision-preserving winner/top-k/margin evidence, forbidden-counter
  closure, and a nonzero independent check count;
- cached continuation requires the exact 19-row workload matrix, restore
  hit/miss reasons, token accounting, output/logit equality, cache identity,
  rank inventory, and cleanup evidence;
- Engine correctness requires the exact six-scenario matrix, scheduler and
  model-runner counts, output identity, publication/restore/cache accounting,
  rank exits, and cleanup evidence;
- every independent verification document must carry the authority-specific
  schema, source/model/workload identity where applicable, and exact semantic
  check count.

Both `validate_prerequisites()` and
`build_qwen35_tp4_performance_prerequisites.py` call this shared validator.
The builder no longer dynamically imports cached/Engine contracts and no
longer contains duplicate authority validators.

TDD evidence:

```text
builder forged-PASS RED:
  expected schema rejection, but the old builder accepted the document
runtime contract forged-PASS RED:
  expected BLOCKED_CORRECTNESS, but the old contract authorized execution
builder tests:                         6 passed
performance contract tests:           16 passed
performance worker tests:             26 passed
performance verifier tests:           11 passed
remote runner tests:                   20 passed
complete selected authority gate:     246 passed across 34 files
focused py_compile:                    passed
git diff --check:                      passed
staged files:                          0
```

The selected gate increased from 244 to 246 tests because it now includes the
two forged-PASS regressions. The exact 34-file inventory remains recorded in
`2026-07-29-qwen35-tp4-cached-continuation-remote-authority.md`.

No SSH, `scp`, `nvidia-smi`, remote directory creation, subprocess adapter,
Torch, Transformers, CUDA, model load, Engine run, or GPU workload was
executed.

Strict boundary:

```text
forged two-field PASS authorization:
  rejected by builder and runtime prerequisite validation
authority-specific evidence semantics:
  implemented and CPU-tested
real root-logit authority artifact:
  absent from the performance prerequisite bundle
real cached exact-five artifact:
  absent
real TP4 Engine exact-four authority:
  absent
canonical performance/cache authority:
  not run
latency/throughput/cache/GPU-memory/compression/quality benefit:
  unmeasured and not claimable
```

## 2026-07-29 Root-Logit Receipt Provenance Closure

The remaining structural provenance asymmetry is closed. Root-logit,
cached-continuation, and Engine correctness now all require:

```text
binding_kind:
  remote_execution_receipt
root_logit_receipt_gap:
  false
evidence:
  immutable execution plan
  consumed single-use authorization
  verified execution receipt
```

The root protocol wraps the frozen mature runner at the semantic four-stage
boundary:

```text
preflight
run
download
verify
```

The executor is dependency-injected, consumes authorization before the first
callback, requires exact
`KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian`, and preserves bounded
FAILED evidence. The receipt verifies four stable unique GPUs, the 24 GiB
free-memory floor, no active compute processes, exact-five regular-file
inventory, source/model/run identity, consumed authorization, and independent
root semantic PASS evidence.

The receipt chain is now preserved through:

```text
real authority adapter
v2 prerequisite builder
runtime prerequisite validator
benchmark worker runtime validation
benchmark assembler
final independent benchmark verifier
```

Fresh CPU-only validation:

```text
focused root/authority gate:           56 passed across 9 files
adapter + builder + contract:          31 passed across 3 files
expanded selected authority gate:     303 passed across 45 files
focused py_compile:                    passed
production legacy-root scan:          no matches
git diff --check:                      passed
staged files:                          0
```

No real remote or GPU execution occurred. Therefore the performance objective
is still not complete:

```text
real three-authority v2 bundle:
  absent
canonical 70-case benchmark:
  not run
speed improvement:
  unmeasured
cache or physical-memory reduction:
  unmeasured
quality or accuracy preservation:
  not established by a real run
```

## 2026-07-29 Manifest-Bound Remote Configuration and Real READY Preparation

The local-weight dependency before campaign preparation is removed without
weakening the existing local configuration builder:

```text
tools/build_qwen35_tp4_engine_authority_configuration.py
tools/test_build_qwen35_tp4_engine_authority_configuration.py
```

`build_configuration(...)` still requires an existing local model directory.
The separate `build_remote_configuration(...)` requires a regular local model
manifest and an explicit absolute remote model directory that exactly matches
the manifest-bound `remote_model_dir`. It does not inspect or require local
model weights. GPU indices, ports, cache limits, timeout, and fingerprint
remain explicit.

The real local configuration and preparation artifacts now exist:

```text
configuration:
  experiments/qwen35_hybrid_state/qwen35-tp4-remote-authority-config-20260729-132532
configuration_sha256:
  1a524173e1be49c8b6e7fc9540e5827d55d278c14184727711f5735635d2712c
source_tree_sha256:
  935e6406a8eda96566094affb8ee3b054cf31c4f3b9c44045fb9db4c1a5b3dce
workload_manifest_sha256:
  d8c81d6efa73f9b5e20dd0019e7e2dbf34e9f2ce4cef60658b0c44f3ca9648c2
preparation:
  experiments/qwen35_hybrid_state/qwen35-tp4-correctness-campaign-preparation-20260729-132532
preparation_manifest_sha256:
  b3d566d7a3877570577e97eacd39c3acfbe59e79b0049142a7d9d5f8fa707e5c
classification:
  READY
execution_performed:
  false
benchmark_execution_authorized:
  false
```

The first real preparation attempt exposed a synthetic-test blind spot:
cached-continuation verification reads two paths below
`downloaded_cached_authority` rather than passing the root directory as a
standalone argv element. The preparation authority binding now accepts only
an absolute verifier argv path structurally equal to or below the plan-bound
authority root, using `Path.relative_to` rather than string-prefix matching.
The synthetic dependency was changed to mirror the production command shape.

Fresh local-only evidence:

```text
remote configuration builder suite:   8 passed
preparation suite:                     5 passed
focused authority gate:               44 passed across 10 files
expanded authority gate:              328 passed across 51 files
clean-namespace dependency probe:      passed, 8 dependency keys
real READY manifest fresh-process reopen:
  passed
py_compile/AST/forbidden surfaces:
  passed
git diff --check:
  passed
staged files:
  0
```

No SSH, `scp`, `nvidia-smi`, Torch, Transformers, CUDA, model load, Engine
construction, subprocess adapter, campaign execution, or GPU workload was
performed. Therefore the engine objective remains incomplete:

```text
real correctness campaign:
  not run
real three-authority v2 prerequisite bundle:
  absent
canonical 70-case TP4 benchmark:
  not run
speed/cache/GPU-memory benefit:
  unmeasured
quality/accuracy preservation:
  not established by a real run
```

Next exact TODO:

```text
on sitian@10.232.195.203, produce the real root-logit,
cached-continuation, and Engine correctness receipt-bound authorities;
adapt and independently validate the v2 bundle before issuing any canonical
benchmark authorization
```

## 2026-07-29 Receipt-Bound Correctness Campaign Coordinator

The three real correctness prerequisite chains now have a single local,
single-use, receipt-bound coordinator:

```text
tools/qwen35_tp4_correctness_authority_campaign_plan.py
tools/qwen35_tp4_correctness_authority_campaign_authorization.py
tools/qwen35_tp4_correctness_authority_campaign_receipt.py
tools/qwen35_tp4_correctness_authority_campaign_executor.py
tools/qwen35_tp4_correctness_authority_campaign_callbacks.py
```

The coordinator does not duplicate child command construction and owns no
subprocess surface. It delegates to the existing root stage runner and
cached/Engine command runners, in exact serial order, then adapts the three
verified authorities, builds the v2 prerequisite bundle, and requires an
independent authorized validation before publishing campaign PASS.

Safety and identity are frozen at the campaign level:

```text
remote target:
  sitian@10.232.195.203
execution environment:
  KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian
child execution:
  strictly serial
benchmark_execution_authorized:
  false
```

Fresh CPU-only validation:

```text
campaign suites:                       15 passed across 5 files
focused campaign integration gate:     50 passed across 9 files
expanded campaign authority gate:     318 passed across 50 files
clean-namespace dependency probe:      passed, 12 dependency keys
campaign and affected py_compile:      passed
forbidden execution surfaces:          0 matches
git diff --check:                       passed
staged files:                           0
```

No real campaign or benchmark was launched. Current objective audit:

```text
campaign coordinator:
  implemented and CPU-verified
real campaign execution:
  not run
real three-authority v2 prerequisite bundle:
  absent
canonical 70-case TP4 benchmark:
  not run
real all-rank cache/CUDA allocator evidence:
  absent
speed improvement:
  unmeasured
cache or physical-memory reduction:
  unmeasured
quality or accuracy preservation:
  not established by a real run
```

Next exact evidence-producing step, when the remote/GPU boundary is approved,
is to instantiate the three immutable child plans, issue one campaign
authorization, and execute the coordinator on
`sitian@10.232.195.203`. The canonical benchmark must remain unauthorized
until the resulting v2 prerequisite bundle independently validates.

## 2026-07-29 Correctness Campaign Preparation Bundle

The remaining manual assembly gap before campaign execution is closed by:

```text
tools/qwen35_tp4_correctness_authority_campaign_preparation.py
```

This pure-local builder accepts explicit configuration, source inventory,
remote model paths, run tags, and nonces. It atomically prepares and
independently verifies the three child plan/authorization pairs plus the
campaign plan/authorization pair. It freezes configuration and source
inventory copies into the bundle, publishes the READY manifest last, and
removes all partial output on failure.

The campaign authority directories are taken from each verified child plan's
actual local output binding, not from a preparation placeholder. This is
required for the real-authority adapter to accept the later receipts.

The module has no subprocess, runner, executor, callback, or remote-source-tar
surface. It cannot consume authorization or start the campaign.

Fresh CPU-only evidence:

```text
preparation suite:                     5 passed
source execution contract:             5 passed
focused preparation integration gate: 40 passed across 10 files
expanded preparation authority gate: 324 passed across 51 files
clean-namespace dependency probe:      passed, 8 dependency keys
preparation and affected py_compile:   passed
forbidden execution surfaces:          0 matches
remote source inventory inclusion:     0 matches
git diff --check:                       passed
staged files:                           0
```

No real preparation bundle was produced because this local session was not
given the explicit real configuration/model inputs. The objective audit
therefore remains:

```text
preparation builder/verifier:
  implemented and CPU-verified
real preparation bundle:
  absent
real correctness campaign:
  not run
real three-authority v2 bundle:
  absent
canonical benchmark:
  not run
speed improvement:
  unmeasured
cache or physical-memory reduction:
  unmeasured
quality or accuracy preservation:
  not established by a real run
```
