# Qwen3.5 Constructed Engine/ModelRunner Ownership Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construct one exact production `LLMEngine` and four exact production `ModelRunner` instances under an audited inert dependency capsule, then bind four approved real TP4 checkpoint candidates through the production zero-payload all-rank Engine method without scheduler, step, CUDA, forward, or inference.

**Architecture:** Import the frozen production classes and execute their attached constructors while a closed context manager replaces only external side-effect dependencies and three forbidden runner methods with ledgered inert implementations. After all module globals are restored, reuse the proven real-candidate producer to publish one pristine-oracle-matching candidate per constructed runner, invoke the production Engine aggregate binding method, clean the whole graph, and publish a two-file independently verifiable artifact.

**Tech Stack:** Python 3.9+, PyTorch CPU, production TinyLLMForge classes, standard-library context managers and weak references, bounded safetensors checkpoint loading, JSON/SHA256 evidence, SSH source-bound remote execution.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Inline execution only; do not use subagents.
- Use only `sitian@10.232.195.203` for remote execution.
- Do not stage, commit, merge, create a PR, delete evidence, or overwrite evidence.
- Use provenance `real-checkpoint-derived-constructed-engine-model-runner-ownership`.
- Use claim boundary `no-scheduler-step-forward-or-inference`.
- Execute the attached production `LLMEngine.__init__` exactly once.
- Execute the attached production `ModelRunner.__init__` exactly four times.
- Never use `object.__new__`, constructor AST compilation, subclass construction, or class replacement.
- Never construct or execute production Scheduler.
- Never execute `LLMEngine.step()`, `ModelRunner.run()`, CUDA, NCCL, model or attention forward, tokenization, sampling, generation, or inference.
- Candidate command transport must remain zero-payload.
- Preserve the exact real worker hard rejection:
  `RuntimeError: real checkpoint load worker execution is not implemented; only the local safety dry-run is authorized`.
- Preserve schema-v2 canonical `NO_GO`.
- Do not claim accuracy, quality, latency, throughput, cache, memory, or compression improvement.

---

### Task 1: Constructor Capsule and Exact Class Construction

**Files:**
- Create: `tools/qwen35_constructed_engine_model_runner_ownership_preflight.py`
- Create: `tools/test_qwen35_constructed_engine_model_runner_ownership_preflight.py`

**Interfaces:**
- Consumes: frozen `LLMEngine`, `ModelRunner`, `Config`, constructor source hashes, and an inert TP4 config.
- Produces: `construct_engine_runtime_under_inert_capsule(...) -> ConstructedRuntimeScope`, an exact dependency ledger, four constructed runners, and complete restoration evidence.

- [x] **Step 1: Write source and allowlist RED tests**

  Add tests that call:

  ```python
  contract = inspect_constructed_runtime_source_contract(source_root)
  ```

  Require exact file hashes, exact method hashes, exact constructor signatures,
  closed replacement names, absence of `object.__new__` and constructor AST
  execution, and rejection of one re-signed source or one extra replacement.

- [x] **Step 2: Run the focused test and require RED**

  Run:

  ```bash
  python3 tools/test_qwen35_constructed_engine_model_runner_ownership_preflight.py
  ```

  Expected: failure because the preflight module or source-contract function
  does not exist.

- [x] **Step 3: Implement the dependency ledger and context manager**

  Define:

  ```python
  @dataclass
  class DependencyCall:
      sequence: int
      dependency: str
      rank: int | None
      arguments: dict
      result_identity: str

  class InertConstructorDependencyCapsule:
      def __enter__(self) -> "InertConstructorDependencyCapsule": ...
      def __exit__(self, exc_type, exc, traceback) -> bool: ...
      def summary(self) -> dict: ...
  ```

  Save and restore every patched attribute identity in `finally`. Reject
  duplicate installation, nested use, missing restoration, unapproved calls,
  wrong arguments, and any call after capsule exit. Do not patch either
  production class binding; the inert process factory must require the exact
  `ModelRunner` class as its target.

- [x] **Step 4: Write exact constructor-call RED tests**

  Require the call counts frozen in the spec:

  ```text
  Engine __init__: 1
  Runner __init__: 4
  init_process_group/set_device/model/load/sampler/warmup/KV: 4 each
  SharedMemory/barrier: 4 each
  worker loop: 3
  Process/Pipe/Event/start/sender-close: 3 each
  Config/context/tokenizer/Scheduler/ack-collector/atexit: 1 each
  capture_cudagraph/apply_cpu_offload: 0
  ```

  Require exact Engine and Runner types, ranks `(0,1,2,3)`, world size four,
  unique runner identities, and original module-global identities restored.

- [x] **Step 5: Implement inert process, transport, tokenizer, scheduler, and model dependencies**

  Implement focused private types:

  ```python
  class _InertSpawnContext: ...
  class _DeferredModelRunnerProcess: ...
  class _InertOneWayPipe: ...
  class _InertEvent: ...
  class _InertSharedMemory: ...
  class _InertTokenizer: ...
  class _InertSchedulerSentinel: ...
  class _InProcessAckCollector: ...
  ```

  `process.start()` records only. The ack collector executes deferred rank
  `1..3` production runner constructors before returning from the Engine
  constructor. No operating-system process or shared memory may be created.

- [x] **Step 6: Implement `ConstructedRuntimeScope` and run GREEN**

  The scope exposes:

  ```python
  scope.engine
  scope.runners_by_rank
  scope.constructor_evidence()
  scope.close_inert_resources()
  ```

  Run:

  ```bash
  python3 tools/test_qwen35_constructed_engine_model_runner_ownership_preflight.py
  python3 -m py_compile \
    tools/qwen35_constructed_engine_model_runner_ownership_preflight.py \
    tools/test_qwen35_constructed_engine_model_runner_ownership_preflight.py
  ```

  Expected: constructor-focused tests pass and compilation succeeds.

### Task 2: Four Real Candidates and Production All-Rank Binding

**Files:**
- Modify: `tools/qwen35_constructed_engine_model_runner_ownership_preflight.py`
- Modify: `tools/test_qwen35_constructed_engine_model_runner_ownership_preflight.py`

**Interfaces:**
- Consumes: `ConstructedRuntimeScope`, pristine TP4 oracle, approved checkpoint, and proven real-candidate producer components.
- Produces: `prepare_and_bind_constructed_runtime(...) -> BoundConstructedRuntimeScope` with four complete payload rows and production Engine aggregate state.

- [x] **Step 1: Write candidate transfer RED tests**

  Use dependency-light fake candidates to require:

  - load order `(0,1,2,3)`;
  - exact runner rank/type/world-size checks;
  - constructor placeholder replaced only by `candidate.owner.model`;
  - one production publication call per rank;
  - no direct candidate-binding call before Engine dispatch;
  - wrong-rank, duplicate, or payload-mismatched candidate rejection.

- [x] **Step 2: Implement candidate preparation adapter**

  Reuse the proven producer component builder from:

  ```text
  tools/qwen35_tp4_real_candidate_provenance_replay_preflight.py
  tools/qwen35_tp4_live_concurrent_candidate_ownership_preflight.py
  ```

  Define:

  ```python
  prepare_constructed_runner_candidate(
      *,
      scope,
      rank,
      checkpoint_dir,
      source_root,
      pristine_row,
  ) -> ConstructedRankCandidateState
  ```

  Retain the runner, slot, request, candidate, owner, runtime bridge, runtime
  identity after binding, model, pool, target, selected tensors, identity
  snapshot, and pool snapshot.

- [x] **Step 3: Write zero-payload all-rank RED tests**

  Require one production Engine call:

  ```python
  rows = scope.engine.bind_qwen35_loaded_checkpoint_candidates(
      timeout_s=0.25,
  )
  ```

  Assert one exact envelope with empty args, worker ranks `(1,2,3)`, four
  `bound` rows, homogeneous identity, exact Engine completion tuple, and
  rank-local publication/candidate-bind/owner-bind counts of one.

- [x] **Step 4: Implement in-process acknowledged dispatch**

  The production rank0 `dispatch_command` writes the exact envelope to the
  inert buffer. `_InProcessAckCollector.collect(...)` reads that envelope,
  checks `expected_ranks`, `timeout_s`, and `is_rank_alive`, then invokes the
  exact production worker binding method and returns production
  `ModelRunnerCommandAck` objects.

- [x] **Step 5: Add exact-repeat and participant-error tests**

  Exact repeat must return the identical stored row tuple with zero new
  envelope, collect, or bind calls. An injected worker participant error must
  leave Engine aggregate completion fields `None`, preserve every candidate,
  and allow one corrected retry.

- [x] **Step 6: Implement full pristine payload validation and GREEN**

  Require each rank's 320 binding hashes, 26 phase hashes, 24 alias groups,
  aggregate hash, loader statistics, model/layout/dtype, and model manifest to
  equal the frozen pristine oracle before Engine binding.

  Run all focused tests and require PASS.

### Task 3: Cleanup, Artifact, and Independent Verifier

**Files:**
- Modify: `tools/qwen35_constructed_engine_model_runner_ownership_preflight.py`
- Modify: `tools/test_qwen35_constructed_engine_model_runner_ownership_preflight.py`
- Create: `tools/verify_qwen35_constructed_engine_model_runner_ownership_gate.py`
- Create: `tools/test_verify_qwen35_constructed_engine_model_runner_ownership_gate.py`

**Interfaces:**
- Consumes: bound constructed runtime, constructor ledger, raw memory points, source identities, and pristine prerequisites.
- Produces: canonical two-file evidence and a standard-library-only independent verification result.

- [x] **Step 1: Write cleanup and forbidden-counter RED tests**

  Track Engine, four runners, process records, channels, shared buffer,
  tokenizer, scheduler sentinel, ack collector, candidates, owners, runtime
  bridges, runtime identities, models, pools, targets, requests, slots, and
  constructor placeholders.

  Require reverse-rank clear, selected tensors zero, non-selected values and
  identities unchanged, pools unchanged, all inert resources closed, all
  weak references dead, and every forbidden counter zero.

- [x] **Step 2: Implement bounded cleanup**

  Define:

  ```python
  release_constructed_runtime(
      bound_scope,
  ) -> dict
  ```

  Do not call production Engine or Runner `exit()`. Serialize evidence before
  clearing slots and registries, clear tensors and references, close inert
  resources, run `gc.collect()`, and return exact collection evidence.

- [x] **Step 3: Implement artifact builder and atomic publication**

  Define:

  ```python
  build_constructed_runtime_artifact(...) -> dict
  finalize_constructed_runtime_artifact(...) -> tuple[Path, Path]
  ```

  Enforce the exact two-file inventory:

  ```text
  constructed_engine_model_runner_ownership.json
  source_manifest.json
  ```

  Refuse publication unless constructor restoration, all-rank binding,
  forbidden counters, cleanup, and memory contracts all pass.

- [x] **Step 4: Write verifier tamper RED tests**

  Reject changed class identity, constructor count, replacement allowlist,
  call order, restoration identity, payload hash, non-empty command args,
  missing acknowledgement, damaged Engine completion state, repeat dispatch,
  non-zero forbidden counter, memory re-signing, leaked object, source drift,
  or extra inventory file.

- [x] **Step 5: Implement stdlib-only verifier**

  The verifier must not import TinyLLMForge, torch, or the preflight. It reads
  raw JSON, rehashes the exact source closure, reconstructs all counts and
  aggregate hashes, and prints:

  ```text
  PASS, <N> checks
  ```

  only after every independent check succeeds.

- [x] **Step 6: Run focused GREEN, compile, and diff check**

  Run:

  ```bash
  python3 tools/test_qwen35_constructed_engine_model_runner_ownership_preflight.py
  python3 tools/test_verify_qwen35_constructed_engine_model_runner_ownership_gate.py
  python3 -m py_compile \
    tools/qwen35_constructed_engine_model_runner_ownership_preflight.py \
    tools/test_qwen35_constructed_engine_model_runner_ownership_preflight.py \
    tools/verify_qwen35_constructed_engine_model_runner_ownership_gate.py \
    tools/test_verify_qwen35_constructed_engine_model_runner_ownership_gate.py
  git diff --check
  ```

  Expected: all focused tests pass, compilation succeeds, and diff check is
  clean.

### Task 4: Source-Bound Remote Gate and Completion Audit

**Files:**
- Modify: `tools/qwen35_constructed_engine_model_runner_ownership_preflight.py`
- Modify: `tools/test_qwen35_constructed_engine_model_runner_ownership_preflight.py`
- Modify: `tools/verify_qwen35_constructed_engine_model_runner_ownership_gate.py`
- Modify: `tools/test_verify_qwen35_constructed_engine_model_runner_ownership_gate.py`
- Create: `experiments/qwen35_hybrid_state/<unique-run-tag>/constructed_engine_model_runner_ownership.json`
- Create: `experiments/qwen35_hybrid_state/<unique-run-tag>/source_manifest.json`
- Modify: `docs/superpowers/specs/2026-07-28-qwen35-constructed-engine-model-runner-ownership-gate-design.md`
- Modify: `docs/superpowers/plans/2026-07-28-qwen35-constructed-engine-model-runner-ownership-gate.md`
- Modify: `AGENT_HANDOFF_STATE.md`

**Interfaces:**
- Consumes: exact source tar, frozen prerequisites, approved checkpoint, remote Python, Kerberos ticket, and live host memory.
- Produces: one authoritative remote run, local and remote independent verification, requirement audit, and the next correctness TODO.

- [x] **Step 1: Implement source-bound CLI**

  Add exact modes:

  ```text
  run
  internal-run
  validate
  ```

  `run` stages deterministic sources and prerequisite artifacts, rehashes them
  remotely, performs host-memory preflight, invokes `internal-run`, downloads
  only the two authoritative files, and re-verifies locally.

- [x] **Step 2: Run local focused and adjacent regressions**

  Run constructed gate tests plus:

  ```text
  live-concurrent ownership tests and verifier tests
  serial provenance tests and verifier tests
  Engine all-rank binding tests
  ModelRunner published-candidate binding tests
  live acknowledgement wiring tests
  checkpoint assignment/binding/factory tests
  full-attention and KV-head replication tests
  exact worker rejection tests
  schema-v2 hybrid-state contract/verifier
  ```

  Do not fix unrelated failures.

- [x] **Step 3: Execute one unique authoritative remote tag**

  Use:

  ```text
  target:
    sitian@10.232.195.203
  KRB5CCNAME:
    FILE:/Users/bytedance/krb5cc_sitian
  ControlMaster:
    /tmp/ssh-sitian-10.232.195.203
  Python:
    /data00/home/sitian/sitian-workspace01/tllm/env/bin/python
  ```

  Preserve every failed or superseded run. Never reuse a tag.

- [x] **Step 4: Verify locally and remotely**

  Run the verifier locally against the downloaded two-file directory. Create a
  separate remote read-only two-file verifier view and require the same result
  and manifest SHA256 values plus the same `PASS, <N> checks` count.

- [x] **Step 5: Complete requirement audit and update docs**

  Map each spec requirement to exact artifact fields, source hashes, focused
  tests, adjacent tests, and verifier checks. Mark plan tasks complete only
  after evidence exists. Add authoritative identities, memory observations,
  exact claim boundary, and limitations to the spec and
  `AGENT_HANDOFF_STATE.md`.

- [x] **Step 6: Keep the long-term goal active**

  Record the next TODO as real output/logit correctness without yet claiming
  accuracy or performance. Keep latency/throughput/cache benchmarking pending
  until correctness passes.

## Completion Record

Authoritative run:

```text
qwen35-constructed-engine-model-runner-ownership-20260728-181454
```

Verification:

```text
local independent verifier:
  PASS, 281 checks
remote independent verifier:
  PASS, 281 checks
result SHA256:
  a3f499eacd19f80c676d71351f4c9904f6dd1be0bcb2cb4023dbefdebe029d0a
manifest SHA256:
  0c709933940cf9b293457308e688b9a44ea98c32a1c4d46ef766b8599906122e
source tree SHA256:
  a2bf242ed69fe556419b0b340602a5293d9d849861aaad2ada0e232e7b4e4717
```

Fresh focused and adjacent validation:

```text
constructed preflight:
  34 tests passed
constructed independent verifier:
  2 tests passed
live-concurrent ownership + verifier:
  18 + 7 tests passed
real-candidate provenance + verifier:
  19 + 5 tests passed
Engine/ModelRunner acknowledgement and binding:
  9 + 4 + 11 + 14 tests passed
checkpoint assignment/binding/factory/loader/request:
  5 + 4 + 6 + 5 + 4 tests passed remotely
full-attention and KV-head:
  passed remotely
real worker rejection:
  6 tests passed remotely
real checkpoint safety gate:
  23 tests passed in the real Git worktree
schema-v2 contract and independent verifier:
  passed
four-file py_compile:
  passed
focused git diff --check:
  passed
```

The system `python3` does not provide torch, so torch-dependent adjacent tests
ran through the approved remote Python against a unique source snapshot. The
safety harness requires Git metadata and therefore ran in the real local
worktree after the temporary remote snapshot correctly rejected the missing
`.git` directory.

Next TODO:

```text
real output/logit correctness under an explicitly bounded runtime path
```

Latency, throughput, cache, GPU-memory, and compression benchmarks remain
pending until that correctness gate passes.
