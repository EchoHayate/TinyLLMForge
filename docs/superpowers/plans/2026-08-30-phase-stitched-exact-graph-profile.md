# Phase-Stitched Exact Graph Profile Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure the removable host-controlled gap between first-token
availability after exact prefill and the first Exact Greedy K8 dispatch, then
produce a source-bound GO/NO_GO decision before any stitched runtime is built.

**Architecture:** Add a dependency-light event recorder owned by `LLMEngine`.
Instrument existing prefill completion, Scheduler commit, next scheduling, K8
lease preparation, and K8 dispatch boundaries only when profiling is enabled.
Run isolated instrumentation-off/on cases, reconstruct the gate from raw
events, and require the profiler-overhead and exact-token controls to pass.

**Tech Stack:** Python 3, `time.perf_counter_ns`, pytest, existing
TinyLLMForge Exact Prefill Graph and Exact Greedy K8 runtime, JSON/JSONL,
SHA-256 manifests, remote A100 execution over SSH.

## Global Constraints

- Work only in `/Users/bytedance/dev/TinyLLMForge`, whose Desktop path is the
  authoritative symlink.
- Do not create a worktree or use a subagent.
- Stage only exact paths; never use broad `git add`, `git reset`, or
  `git clean`.
- Keep remote artifacts under
  `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/`.
- Do not write task data to remote `/`, remote `/tmp`, the retired checkout,
  or local `experiments/`.
- Reuse `/data00/home/sitian/.ms_cache/Qwen/Qwen3-0___6B`; do not download a
  model.
- Use `KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian`; never run `kinit` or
  `krenew`.
- Do not terminate or take over external GPU processes.
- Require a clean A100 80GB GPU before launching the benchmark.
- Preserve exact generated token IDs and text.
- Keep profiling disabled by default and add no synchronization to an
  instrumentation-off run.
- Do not implement the stitched runtime unless this profile returns
  `GO_PHASE_STITCH_PROFILE`.
- Do not implement or claim sentinel-filled prefill graph buckets.

---

### Task 1: Define the profile event contract

**Files:**

- Create: `tinyvllm/engine/phase_stitch_profile.py`
- Create: `tools/test_phase_stitch_profile.py`

**Interfaces:**

- Produces:
  `PhaseStitchProfileRecorder(enabled: bool, clock_ns: Callable[[], int])`
- Produces:
  `begin_request(sequence_id: int, prompt_tokens: int) -> None`
- Produces:
  `mark(sequence_id: int, event: str) -> None`
- Produces:
  `finish_request(sequence_id: int, output_token_ids: tuple[int, ...]) -> dict`
- Produces: `snapshot() -> dict`

- [ ] **Step 1: Write validation and lifecycle tests**

```python
def test_profile_reconstructs_one_prefill_to_k8_handoff():
    clock = iter((100, 120, 150, 190, 230, 260, 300)).__next__
    recorder = PhaseStitchProfileRecorder(enabled=True, clock_ns=clock)
    recorder.begin_request(sequence_id=7, prompt_tokens=256)
    for event in PHASE_STITCH_EVENTS:
        recorder.mark(7, event)
    row = recorder.finish_request(7, (11,) * 128)
    assert row["removable_host_gap_ns"] == (
        row["first_k8_dispatch_started_ns"]
        - row["first_token_host_available_ns"]
    )
    assert row["event_coverage_complete"] is True


def test_profile_rejects_duplicate_or_out_of_order_events():
    recorder = PhaseStitchProfileRecorder(
        enabled=True,
        clock_ns=iter((100, 120, 130)).__next__,
    )
    recorder.begin_request(sequence_id=7, prompt_tokens=256)
    recorder.mark(7, "prefill_dispatch_finished")
    with pytest.raises(ValueError, match="event order"):
        recorder.mark(7, "prefill_dispatch_finished")


def test_disabled_profile_is_a_noop():
    recorder = PhaseStitchProfileRecorder(
        enabled=False,
        clock_ns=lambda: (_ for _ in ()).throw(
            AssertionError("disabled recorder read the clock")
        ),
    )
    recorder.begin_request(sequence_id=7, prompt_tokens=256)
    recorder.mark(7, "prefill_dispatch_finished")
    assert recorder.snapshot()["rows"] == []
```

- [ ] **Step 2: Run the new tests and confirm RED**

Run:

```bash
python3 -m pytest -q tools/test_phase_stitch_profile.py
```

Expected: collection fails because
`tinyvllm.engine.phase_stitch_profile` does not exist.

- [ ] **Step 3: Implement the dependency-light recorder**

Define the frozen event order:

```python
PHASE_STITCH_EVENTS = (
    "prefill_dispatch_finished",
    "first_token_host_available",
    "prefill_scheduler_commit_finished",
    "next_schedule_started",
    "next_schedule_finished",
    "k8_lease_prepare_finished",
    "first_k8_dispatch_started",
)
```

Each completed row must contain the seven timestamps, six adjacent interval
durations, `removable_host_gap_ns`, `prompt_tokens`, `sequence_id`,
`output_token_ids_sha256`, and `event_coverage_complete`. Reject duplicate,
missing, non-monotonic, cross-sequence, and post-finalization events.

- [ ] **Step 4: Run the focused tests and confirm GREEN**

Run:

```bash
python3 -m pytest -q tools/test_phase_stitch_profile.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit the recorder**

```bash
git add -- tinyvllm/engine/phase_stitch_profile.py \
  tools/test_phase_stitch_profile.py
git commit -m "feat(profiler): add phase-stitch event contract"
```

### Task 2: Instrument the existing engine path

**Files:**

- Modify: `tinyvllm/config.py`
- Modify: `tinyvllm/engine/llm_engine.py`
- Modify: `tools/test_phase_stitch_profile.py`

**Interfaces:**

- Consumes: `PhaseStitchProfileRecorder`
- Produces:
  `LLMEngine.configure_phase_stitch_profile(enabled: bool) -> dict`
- Produces: `LLMEngine.phase_stitch_profile_snapshot() -> dict`

- [ ] **Step 1: Write failing integration tests with fake components**

Cover:

```python
def test_engine_marks_final_prefill_and_first_k8_boundaries():
    engine = make_profiled_fake_engine(
        prefill_tokens=(101,),
        burst_tokens=(102, 103, 104, 105, 106, 107, 108),
    )
    engine.step(completion_only=True)
    engine.step(completion_only=True)
    assert engine.phase_stitch_profile_snapshot()["active"][0][
        "events"
    ] == list(PHASE_STITCH_EVENTS)
    assert events == list(PHASE_STITCH_EVENTS)


def test_engine_does_not_touch_clock_when_profile_disabled():
    engine = make_profiled_fake_engine(profile_enabled=False)
    engine.phase_stitch_profile._clock_ns = lambda: (
        (_ for _ in ()).throw(
            AssertionError("disabled profile read the clock")
        )
    )
    engine.step(completion_only=True)


def test_non_k8_followup_finishes_as_ineligible_without_fake_gap():
    engine = make_profiled_fake_engine(
        exact_burst_available=False,
    )
    engine.step(completion_only=True)
    engine.step(completion_only=True)
    row = engine.phase_stitch_profile_snapshot()["rows"][0]
    assert row["status"] == "ineligible"
    assert row["removable_host_gap_ns"] is None
```

The fake runner must emit one prefill token and then an accepted K8 result.
The test must assert exact event order and sequence identity, not only event
count.

- [ ] **Step 2: Run the integration tests and confirm RED**

Run:

```bash
python3 -m pytest -q \
  tools/test_phase_stitch_profile.py::test_engine_marks_final_prefill_and_first_k8_boundaries \
  tools/test_phase_stitch_profile.py::test_engine_does_not_touch_clock_when_profile_disabled \
  tools/test_phase_stitch_profile.py::test_non_k8_followup_finishes_as_ineligible_without_fake_gap
```

Expected: fail because the engine configuration and hooks are absent.

- [ ] **Step 3: Add the disabled-by-default configuration**

Add:

```python
phase_stitch_profile: bool = False
```

Validate that it is exactly `bool`; it must not imply
`phase_stitched_exact_graph_runtime`.

- [ ] **Step 4: Add narrowly placed timing hooks**

In `LLMEngine.step`, mark:

```python
if is_final_single_sequence_prefill:
    profiler.begin_request(seq.seq_id, seq.num_prompt_tokens)

# immediately after ModelRunner returns the sampled prefill token
profiler.mark(seq.seq_id, "prefill_dispatch_finished")
profiler.mark(seq.seq_id, "first_token_host_available")

# immediately after the existing prepared Scheduler postprocess commits
profiler.mark(seq.seq_id, "prefill_scheduler_commit_finished")

# around the next scheduler.schedule call for the same request
profiler.mark(seq.seq_id, "next_schedule_started")
scheduled = self.scheduler.schedule(decision_now_ns)
profiler.mark(seq.seq_id, "next_schedule_finished")

# after a K8 lease is accepted and before ModelRunner dispatch
profiler.mark(seq.seq_id, "k8_lease_prepare_finished")
profiler.mark(seq.seq_id, "first_k8_dispatch_started")
```

Do not add `torch.cuda.synchronize()`. The first-token host timestamp is taken
after the existing token materialization has already made the result
host-visible.

- [ ] **Step 5: Run focused and adjacent tests**

Run:

```bash
python3 -m pytest -q \
  tools/test_phase_stitch_profile.py \
  tools/test_exact_prefill_cuda_graph.py \
  tools/test_exact_prefill_cuda_graph_benchmark.py \
  tools/test_graph_resident_greedy_tail.py
```

Expected: all tests pass.

- [ ] **Step 6: Commit engine instrumentation**

```bash
git add -- tinyvllm/config.py tinyvllm/engine/llm_engine.py \
  tools/test_phase_stitch_profile.py
git commit -m "feat(profiler): trace prefill to K8 handoff"
```

### Task 3: Freeze the paired profile contract and worker

**Files:**

- Create: `tools/phase_stitch_profile_contract.py`
- Create: `tools/phase_stitch_profile_worker.py`
- Create: `tools/test_phase_stitch_profile_benchmark.py`

**Interfaces:**

- Produces: `build_case_matrix() -> list[dict]`
- Produces: `contract_sha256() -> str`
- Produces: `run_worker(spec: dict, model: str, output_dir: Path) -> dict`

- [ ] **Step 1: Write failing contract tests**

Freeze:

```python
ARMS = ("instrumentation_off", "instrumentation_on")
PROMPT_TOKEN_COUNTS = (256, 2048)
ROUNDS = 2
WARMUP_REPETITIONS = 2
MEASURED_REPETITIONS = 5
GENERATED_TOKENS = 128
```

Require AB/BA arm order, fresh engine per case, exact prompt hashes, and:

```python
engine_config = {
    "tensor_parallel_size": 1,
    "max_num_seqs": 1,
    "prefill_cuda_graphs": True,
    "prefill_cuda_graph_token_allowlist": [256, 2048],
    "exact_greedy_decode_burst": True,
    "exact_greedy_decode_burst_tokens": 8,
    "phase_stitch_profile": arm == "instrumentation_on",
}
```

- [ ] **Step 2: Run contract tests and confirm RED**

Run:

```bash
python3 -m pytest -q tools/test_phase_stitch_profile_benchmark.py
```

Expected: import failure for the new contract module.

- [ ] **Step 3: Implement the contract and isolated worker**

The worker must:

- use deterministic synthetic token prompts;
- run warmups before resetting profiler and peak-memory state;
- call `engine.step(completion_only=True)`;
- retain all per-request timestamps for the instrumentation-on arm;
- retain TTFT, TPOT samples, E2E, throughput, tokens, text hash, graph replay
  deltas, and peak memory for both arms;
- require exactly 128 generated tokens;
- write one immutable `result.json` with `allow_nan=False`;
- always call `engine.exit()`.

- [ ] **Step 4: Test malformed and valid worker results**

Reject missing events, duplicate samples, non-positive latency, invalid hashes,
wrong token counts, graph replay absence, and arm/config drift.

- [ ] **Step 5: Run tests and commit**

```bash
python3 -m pytest -q tools/test_phase_stitch_profile_benchmark.py
git add -- tools/phase_stitch_profile_contract.py \
  tools/phase_stitch_profile_worker.py \
  tools/test_phase_stitch_profile_benchmark.py
git commit -m "test(profiler): add phase-stitch benchmark contract"
```

### Task 4: Implement producer gate and independent verifier

**Files:**

- Create: `tools/phase_stitch_profile_gate.py`
- Create: `tools/phase_stitch_profile_verify.py`
- Modify: `tools/test_phase_stitch_profile_benchmark.py`

**Interfaces:**

- Produces: `produce_gate(run_dir: Path) -> dict`
- Produces: `verify_bundle(run_dir: Path) -> dict`

- [ ] **Step 1: Write fail-closed gate tests**

Cover:

```python
def test_gate_returns_go_only_when_ceiling_and_overhead_pass():
    run_dir = write_profile_fixture(
        median_gap_ns=700_000,
        p95_gap_ns=900_000,
        profile_overhead_fraction=0.005,
    )
    assert produce_gate(run_dir)["classification"] == (
        "GO_PHASE_STITCH_PROFILE"
    )


def test_gate_returns_no_go_when_gap_is_below_ceiling():
    run_dir = write_profile_fixture(
        median_gap_ns=100_000,
        p95_gap_ns=200_000,
        profile_overhead_fraction=0.005,
    )
    assert produce_gate(run_dir)["classification"] == (
        "NO_GO_PHASE_STITCH_CEILING"
    )


def test_gate_rejects_missing_or_nonfinite_timestamps():
    run_dir = write_profile_fixture()
    mutate_first_profile_row(
        run_dir,
        "first_k8_dispatch_started_ns",
        float("nan"),
    )
    with pytest.raises(ValueError, match="finite"):
        produce_gate(run_dir)


def test_gate_rejects_token_or_source_hash_drift():
    run_dir = write_profile_fixture()
    mutate_first_profile_row(
        run_dir,
        "output_token_ids_sha256",
        "0" * 64,
    )
    with pytest.raises(ValueError, match="hash"):
        produce_gate(run_dir)


def test_verifier_reconstructs_without_importing_producer():
    run_dir = write_profile_fixture()
    source = Path(
        "tools/phase_stitch_profile_verify.py"
    ).read_text(encoding="utf-8")
    assert "phase_stitch_profile_gate import" not in source
    assert verify_bundle(run_dir)["verified"] is True
```

- [ ] **Step 2: Run tests and confirm RED**

Run:

```bash
python3 -m pytest -q tools/test_phase_stitch_profile_benchmark.py
```

Expected: fail because producer and verifier modules do not exist.

- [ ] **Step 3: Implement frozen classification**

Return `GO_PHASE_STITCH_PROFILE` only when:

```python
ceiling_pass = (
    any(shape.median_gap_ns >= 150_000 for shape in shapes)
    and any(
        shape.median_gap_ns / shape.e2e_median_ns >= 0.03
        or shape.p95_gap_ns >= 500_000
        for shape in shapes
    )
)
overhead_pass = all(
    abs(shape.profile_on_e2e_median_ns
        / shape.profile_off_e2e_median_ns - 1.0) <= 0.01
    for shape in shapes
)
```

Also require exact token/text equality, complete event coverage, positive
finite timings, prefill replay and K8 acceptance/replay evidence, zero
quarantine/failure counts, complete source hashes, and complete case inventory.

- [ ] **Step 4: Implement an independent verifier**

The verifier must duplicate validation and aggregation logic instead of
importing `phase_stitch_profile_gate`. It must reconstruct every median/P95,
contract hash, source hash, output hash, case identity, and final
classification from raw result files.

- [ ] **Step 5: Run tests and commit**

```bash
python3 -m pytest -q tools/test_phase_stitch_profile_benchmark.py
git add -- tools/phase_stitch_profile_gate.py \
  tools/phase_stitch_profile_verify.py \
  tools/test_phase_stitch_profile_benchmark.py
git commit -m "feat(profiler): gate phase-stitch latency ceiling"
```

### Task 5: Add the remote clean-GPU controller

**Files:**

- Create: `tools/run_phase_stitch_profile_remote.py`
- Create: `tools/test_run_phase_stitch_profile_remote.py`

**Interfaces:**

- Produces a source-bound remote run under the approved `/data00` root.
- Produces `run_manifest.json`, per-case `result.json`, `summary.json`,
  `manifest.json`, producer and verifier exit receipts, and a compact local
  evidence bundle.

- [ ] **Step 1: Write controller safety tests**

Assert:

- remote root is exactly under
  `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/`;
- the controller never names `/tmp` or local `experiments/`;
- GPU admission requires one clean A100;
- external GPU processes are reported but never killed;
- the model path is reused and no download command exists;
- a run tag cannot overwrite an existing remote or local directory;
- Kerberos expiry is checked and no renewal command exists;
- source archive contains only the frozen source allowlist;
- every subprocess exit code is captured.

- [ ] **Step 2: Run tests and confirm RED**

Run:

```bash
python3 -m pytest -q tools/test_run_phase_stitch_profile_remote.py
```

Expected: import failure for the new controller.

- [ ] **Step 3: Implement the controller**

Use:

```text
host: sitian@10.232.195.203
credential cache: FILE:/Users/bytedance/krb5cc_sitian
model: /data00/home/sitian/.ms_cache/Qwen/Qwen3-0___6B
remote task root:
  /data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/
    phase-stitch-profile-20260830-qwen3-06b-r1
```

The controller must poll for a clean A100 without terminating foreign
processes, upload a compact source archive, run every case in a separate
process, run producer and verifier remotely, then download only the compact
allowlisted evidence.

- [ ] **Step 4: Run controller tests and commit**

```bash
python3 -m pytest -q tools/test_run_phase_stitch_profile_remote.py
git add -- tools/run_phase_stitch_profile_remote.py \
  tools/test_run_phase_stitch_profile_remote.py
git commit -m "feat(profiler): add remote phase-stitch gate"
```

### Task 6: Run local verification and the real A100 gate

**Files:**

- Create:
  `artifacts/phase_stitch_profile/20260830-qwen3-06b-r1/`
- Modify: `AGENT_HANDOFF_STATE.md`

**Interfaces:**

- Consumes all Stage-0 tools.
- Produces the terminal `GO_PHASE_STITCH_PROFILE` or
  `NO_GO_PHASE_STITCH_CEILING` evidence.

- [ ] **Step 1: Run all focused local tests**

```bash
python3 -m pytest -q \
  tools/test_phase_stitch_profile.py \
  tools/test_phase_stitch_profile_benchmark.py \
  tools/test_run_phase_stitch_profile_remote.py \
  tools/test_exact_prefill_cuda_graph.py \
  tools/test_exact_prefill_cuda_graph_benchmark.py \
  tools/test_graph_resident_greedy_tail.py
```

Expected: all tests pass.

- [ ] **Step 2: Run compilation and exact diff checks**

```bash
python3 -m py_compile \
  tinyvllm/engine/phase_stitch_profile.py \
  tinyvllm/config.py \
  tinyvllm/engine/llm_engine.py \
  tools/phase_stitch_profile_contract.py \
  tools/phase_stitch_profile_worker.py \
  tools/phase_stitch_profile_gate.py \
  tools/phase_stitch_profile_verify.py \
  tools/run_phase_stitch_profile_remote.py

git diff --check -- \
  tinyvllm/engine/phase_stitch_profile.py \
  tinyvllm/config.py \
  tinyvllm/engine/llm_engine.py \
  tools/phase_stitch_profile_contract.py \
  tools/phase_stitch_profile_worker.py \
  tools/phase_stitch_profile_gate.py \
  tools/phase_stitch_profile_verify.py \
  tools/run_phase_stitch_profile_remote.py
```

- [ ] **Step 3: Check Kerberos and launch the remote run**

```bash
KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian klist
python3 -m tools.run_phase_stitch_profile_remote \
  --host sitian@10.232.195.203 \
  --model /data00/home/sitian/.ms_cache/Qwen/Qwen3-0___6B \
  --run-tag 20260830-qwen3-06b-r1
```

Do not launch if credentials are expired or no clean A100 is available.

- [ ] **Step 4: Independently verify the downloaded bundle**

```bash
python3 -m tools.phase_stitch_profile_verify \
  --run-dir artifacts/phase_stitch_profile/20260830-qwen3-06b-r1
```

Expected: `verified=true` and exactly one terminal classification.

- [ ] **Step 5: Record the decision**

Append a reconciliation to `AGENT_HANDOFF_STATE.md` with:

- source commit and source hashes;
- remote run path and GPU identity;
- exact workload and case inventory;
- measured removable gap and profiler overhead per shape;
- correctness and graph-use evidence;
- producer and independent-verifier results;
- benefit and cost;
- the explicit next action:
  - write the Stage-1 runtime plan only for `GO_PHASE_STITCH_PROFILE`;
  - stop runtime work for `NO_GO_PHASE_STITCH_CEILING`.

- [ ] **Step 6: Commit, push, and verify the remote SHA**

Stage only the Stage-0 source, tests, plan/spec update, handoff reconciliation,
and compact terminal evidence:

```bash
git add -- \
  AGENT_HANDOFF_STATE.md \
  artifacts/phase_stitch_profile/20260830-qwen3-06b-r1 \
  docs/superpowers/audits/2026-08-30-phase-stitch-profile-audit.md \
  docs/superpowers/plans/2026-08-30-phase-stitched-exact-graph-profile.md \
  docs/superpowers/specs/2026-08-30-phase-stitched-exact-graph-runtime-design.md \
  tinyvllm/config.py \
  tinyvllm/engine/llm_engine.py \
  tinyvllm/engine/phase_stitch_profile.py \
  tools/phase_stitch_profile_contract.py \
  tools/phase_stitch_profile_gate.py \
  tools/phase_stitch_profile_verify.py \
  tools/phase_stitch_profile_worker.py \
  tools/run_phase_stitch_profile_remote.py \
  tools/test_phase_stitch_profile.py \
  tools/test_phase_stitch_profile_benchmark.py \
  tools/test_run_phase_stitch_profile_remote.py
git diff --cached --check
git commit -m "perf(runtime): profile prefill to K8 handoff"
git push -u origin feat/kv-sparse-attention
test "$(git rev-parse HEAD)" = \
  "$(git ls-remote origin refs/heads/feat/kv-sparse-attention | awk '{print $1}')"
```

### Task 7: Completion audit

**Files:**

- Create:
  `docs/superpowers/audits/2026-08-30-phase-stitch-profile-audit.md`

- [ ] **Step 1: Build the prompt-to-artifact checklist**

Map every Global Constraint and Tasks 1-6 requirement to:

- source path and line;
- focused test;
- raw artifact;
- manifest field;
- producer result;
- independent-verifier result;
- commit and remote SHA.

- [ ] **Step 2: Classify uncovered requirements as incomplete**

Do not use passing tests, a manifest, or the producer summary as a proxy for
missing runtime evidence. Any absent timestamp, graph counter, raw row, source
hash, or cost field fails the profile gate.

- [ ] **Step 3: Commit and push the audit**

```bash
git add -- \
  docs/superpowers/audits/2026-08-30-phase-stitch-profile-audit.md \
  AGENT_HANDOFF_STATE.md
git commit -m "docs(audit): reconcile phase-stitch profile"
git push -u origin feat/kv-sparse-attention
```
