# Zero-Temperature Greedy Fast Path Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce Qwen3-0.6B batch-1 TTFT and decode TPOT by preserving the existing float32 greedy argmax while bypassing all stochastic-sampling work for exact zero-temperature requests.

**Architecture:** A dependency-light policy module owns eligibility and accounting. `ModelRunner` routes exactly eligible batch-1 sampling points to `logits.to(torch.float32).argmax(dim=-1).tolist()` before temperature materialization; every other case executes the current sampler unchanged. A source-bound OFF/ON worker, producer gate, independent verifier, and safe remote controller establish exact logits/output parity, latency benefit, and resource cost.

**Tech Stack:** Python 3, PyTorch, TinyLLMForge `ModelRunner`, dependency-light script tests, binary float32 logit sidecars, JSON/JSONL evidence, SSH remote runner.

## Global Constraints

- Work only in `/Users/bytedance/dev/TinyLLMForge`, the target of the authoritative `/Users/bytedance/Desktop/TinyLLMForge` symlink.
- Do not create worktrees or use subagents.
- Never modify `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Preserve all unrelated dirty and untracked files.
- Stage exact paths only; never use broad `git add`, `git reset`, `git clean`, or mass formatting.
- Commit with `git -c core.hooksPath=/dev/null commit`.
- Every commit has exactly one `Co-authored-by: TRAE CLI <noreply@bytedance.com>` trailer.
- Push only to `origin/feat/kv-sparse-attention`.
- The feature flag is `zero_temperature_greedy_fast_path` and defaults to `False`.
- Stage 1 covers Qwen3-0.6B, tensor parallel size one, batch size one, ordinary generation, and exact `temperature == 0.0`.
- The optimized expression is exactly `logits.to(torch.float32).argmax(dim=-1)`.
- Nonzero temperature, mixed batches, batch sizes above one, and shape drift fall back to the existing sampler.
- Every 128-token ON row must record exactly 128 optimized sampling steps.
- Output token IDs and decoded-text hashes must match exactly.
- Logit correctness remains `max_abs <= 0.25`, `mean_abs <= 0.05`, and argmax equality.
- Report benefit and cost together.
- Every remote run tag is immutable.
- All remote task output stays under `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818`.
- Never write remote task data under `/`, `/tmp`, `/private/tmp`, or `/data00/home/sitian/tllm/TinyLLMForge`.
- Do not refresh Kerberos automatically.
- Do not terminate or interfere with unrelated GPU processes.
- GPU admission requires memory used `<=1024 MiB`, utilization `<=5%`, and no compute process.
- Qwen3-0.6B evidence cannot support Qwen3-8B claims.
- Do not launch Qwen3-8B unless the Qwen3-0.6B Stage-1 gate is GO.

---

## File structure

- Create `tinyvllm/engine/greedy_sampling_fast_path.py`: pure eligibility and accounting.
- Create `tools/test_greedy_sampling_fast_path.py`: dependency-light policy tests.
- Modify `tinyvllm/config.py`: add the default-disabled flag and strict validation.
- Modify `tinyvllm/engine/model_runner.py`: own stats and route eligible sampling.
- Modify `tools/test_model_runner_spec_verify.py`: integration and fallback tests.
- Create `tools/profile_zero_temperature_greedy_fast_path.py`: performance worker and bounded logits probes.
- Create `tools/test_profile_zero_temperature_greedy_fast_path.py`: worker contracts and binary-sidecar tests.
- Create `tools/zero_temperature_greedy_fast_path_gate.py`: producer validation and classification.
- Create `tools/zero_temperature_greedy_fast_path_verify.py`: independent reconstruction.
- Create `tools/test_zero_temperature_greedy_fast_path_gate.py`: producer and tamper tests.
- Create `tools/test_zero_temperature_greedy_fast_path_verify.py`: independent-verifier tests.
- Create `tools/run_zero_temperature_greedy_fast_path_remote.py`: safe source-bound controller.
- Create `tools/test_run_zero_temperature_greedy_fast_path_remote.py`: controller safety tests.
- Modify `AGENT_HANDOFF_STATE.md`: append the terminal result at EOF.
- Modify `docs/superpowers/audits/2026-08-16-phase1-completion-audit.md`: append the full reconciliation and prompt-to-artifact checklist at EOF.

### Task 1: Pure eligibility and accounting

**Files:**

- Create: `tinyvllm/engine/greedy_sampling_fast_path.py`
- Create: `tools/test_greedy_sampling_fast_path.py`

**Interfaces:**

- Produces:
  - `GreedySamplingFastPathDecision`
  - `GreedySamplingFastPathStats`
  - `decide_greedy_sampling_fast_path(...)`
  - `stats.record_optimized(batch_size)`
  - `stats.record_fallback(reason)`
  - `stats.summary()`

- [ ] **Step 1: Write failing decision tests**

```python
decision = decide_greedy_sampling_fast_path(
    enabled=True,
    rank=0,
    temperatures=(0.0,),
    batch_kind=None,
    logits_shape=(1, 151936),
)
assert decision.optimized is True
assert decision.fallback_reason is None
```

Parameterize disabled, non-root rank, empty/two-row batch, mixed batch,
nonzero temperature, nonnumeric temperature, and logits-shape mismatch. Each
must return a stable fallback reason without importing Torch.

- [ ] **Step 2: Run the focused test and confirm RED**

Run:

```bash
python3 tools/test_greedy_sampling_fast_path.py
```

Expected: import failure because the policy module does not exist.

- [ ] **Step 3: Implement immutable decision and mutable counters**

```python
@dataclass(frozen=True)
class GreedySamplingFastPathDecision:
    optimized: bool
    fallback_reason: str | None


@dataclass
class GreedySamplingFastPathStats:
    eligible_steps: int = 0
    optimized_steps: int = 0
    avoided_temperature_h2d_bytes: int = 0
    avoided_softmax_calls: int = 0
    avoided_gumbel_rng_calls: int = 0
    avoided_stochastic_divisions: int = 0
    avoided_stochastic_argmax_calls: int = 0
    avoided_where_calls: int = 0
    fallback_counts: dict[str, int] = field(default_factory=dict)

    def record_optimized(self, batch_size: int) -> None:
        self.eligible_steps += 1
        self.optimized_steps += 1
        self.avoided_temperature_h2d_bytes += 4 * batch_size
        self.avoided_softmax_calls += 1
        self.avoided_gumbel_rng_calls += 1
        self.avoided_stochastic_divisions += 2
        self.avoided_stochastic_argmax_calls += 1
        self.avoided_where_calls += 1
```

`decide_greedy_sampling_fast_path()` validates host values in a fixed order
and never touches a tensor value.

- [ ] **Step 4: Run GREEN and syntax checks**

```bash
python3 tools/test_greedy_sampling_fast_path.py
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-greedy-fast-path-pycache \
  python3 -m py_compile \
  tinyvllm/engine/greedy_sampling_fast_path.py \
  tools/test_greedy_sampling_fast_path.py
```

Expected: PASS and zero exit status.

- [ ] **Step 5: Commit and push**

```bash
git add -- \
  tinyvllm/engine/greedy_sampling_fast_path.py \
  tools/test_greedy_sampling_fast_path.py
git -c core.hooksPath=/dev/null commit \
  -m "feat(perf): add greedy fast path policy" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

### Task 2: Default-disabled ModelRunner integration

**Files:**

- Modify: `tinyvllm/config.py`
- Modify: `tinyvllm/engine/model_runner.py`
- Modify: `tools/test_model_runner_spec_verify.py`

**Interfaces:**

- Consumes `decide_greedy_sampling_fast_path()` and
  `GreedySamplingFastPathStats`.
- Produces:
  - `Config.zero_temperature_greedy_fast_path: bool`
  - `ModelRunner.zero_temperature_greedy_fast_path_summary() -> dict`
  - `_sample_tokens_with_optional_greedy_fast_path(...) -> list[int]`

- [ ] **Step 1: Add failing config and disabled-path tests**

Assert:

```python
assert (
    Config.__dataclass_fields__[
        "zero_temperature_greedy_fast_path"
    ].default
    is False
)
```

A non-boolean value must raise:

```text
zero_temperature_greedy_fast_path must be a bool
```

With the flag disabled, assert exactly one call each to
`prepare_sample()` and `self.sampler()`, and zero optimized steps.

- [ ] **Step 2: Run and confirm RED**

```bash
python3 tools/test_model_runner_spec_verify.py
```

Expected: failure on the missing config field or router.

- [ ] **Step 3: Add config, runner-owned stats, and router**

Add:

```python
zero_temperature_greedy_fast_path: bool = False
```

Construct:

```python
self.greedy_sampling_fast_path_stats = (
    GreedySamplingFastPathStats()
)
```

Route rank-zero sampling through:

```python
def _sample_tokens_with_optional_greedy_fast_path(
    self,
    logits,
    sample_seqs,
    *,
    batch_kind,
):
    decision = decide_greedy_sampling_fast_path(
        enabled=self.config.zero_temperature_greedy_fast_path,
        rank=self.rank,
        temperatures=tuple(
            seq.temperature for seq in sample_seqs
        ),
        batch_kind=batch_kind,
        logits_shape=tuple(logits.shape),
    )
    if decision.optimized:
        self.greedy_sampling_fast_path_stats.record_optimized(
            len(sample_seqs)
        )
        return (
            logits.to(torch.float32)
            .argmax(dim=-1)
            .tolist()
        )
    self.greedy_sampling_fast_path_stats.record_fallback(
        decision.fallback_reason
    )
    temperatures = self.prepare_sample(sample_seqs)
    return self.sampler(logits, temperatures).tolist()
```

Replace only the two current sampling lines with this router. Preserve the
existing pre-sampling logit capture before the router.

- [ ] **Step 4: Add eligible and fallback integration tests**

Eligible test requirements:

- feature enabled;
- one sequence at temperature zero;
- logits fake records `.to(torch.float32)`, one argmax, and one `.tolist()`;
- `prepare_sample` and stochastic sampler are never called;
- input logits are not mutated;
- summary records one optimized step and exact avoided-operation counts.

Fallback tests cover disabled, temperature `0.7`, two sequences, mixed batch,
and logits shape mismatch. Each must execute the legacy sampler once.

- [ ] **Step 5: Run focused and neighboring regressions**

```bash
python3 tools/test_greedy_sampling_fast_path.py
python3 tools/test_model_runner_spec_verify.py
python3 tools/test_decode_metadata_landing.py
python3 tools/test_chunked_prefill.py
```

The local Torch-dependent suite may stop with
`ModuleNotFoundError: No module named 'torch'`; record it as environment
blocked and require it in remote preflight.

- [ ] **Step 6: Commit and push**

```bash
git add -- \
  tinyvllm/config.py \
  tinyvllm/engine/model_runner.py \
  tools/test_model_runner_spec_verify.py
git -c core.hooksPath=/dev/null commit \
  -m "feat(perf): bypass stochastic work for greedy sampling" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

### Task 3: Source-bound worker and logits sidecars

**Files:**

- Create: `tools/profile_zero_temperature_greedy_fast_path.py`
- Create: `tools/test_profile_zero_temperature_greedy_fast_path.py`

**Interfaces:**

- Produces schemas:
  - `zero-temperature-greedy-fast-path.case.v1`
  - `zero-temperature-greedy-fast-path.correctness.v1`
  - `zero-temperature-greedy-fast-path.summary.v1`
- Produces a bounded `logits/` directory of little-endian float32 sidecars.

- [ ] **Step 1: Write failing pure worker-contract tests**

Test:

```python
assert context_cases() == (
    ("short", 256, 128),
    ("medium", 2048, 128),
    ("long", 8192, 128),
)
assert policy_order(0) == ("off", "on")
assert policy_order(1) == ("on", "off")
```

Round-trip a known float tuple through the binary writer/reader and reject a
stale SHA256, wrong byte length, non-finite value, reused case identity,
wrong output length, or ON row whose optimized-step count is not 128.

- [ ] **Step 2: Run and confirm RED**

```bash
python3 tools/test_profile_zero_temperature_greedy_fast_path.py
```

Expected: import failure because the worker does not exist.

- [ ] **Step 3: Implement performance rows**

Construct fresh OFF/ON `LLM` instances differing only in:

```python
zero_temperature_greedy_fast_path=enabled
```

Reuse the proven request loop and internal decode profiler contract from
`profile_replay_aware_decode_metadata.py`. Capture exact output IDs, text
hash, TTFT, E2E, 127 TPOT samples, decode host/CUDA samples, throughput,
memory, and the delta of the greedy-fast-path summary.

- [ ] **Step 4: Implement three-point logits probes**

For each bucket and policy, run a separate deterministic two-phase request
that retains pre-sampling logits at:

```text
prefill-final
decode-first
decode-final
```

Serialize each row as little-endian float32 bytes. Record path, element count,
shape, byte length, SHA256, policy, bucket, and sampling point in
`correctness_rows.jsonl`. Keep profiling disabled during probes.

- [ ] **Step 5: Run GREEN and syntax checks**

```bash
python3 tools/test_profile_zero_temperature_greedy_fast_path.py
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-greedy-worker-pycache \
  python3 -m py_compile \
  tools/profile_zero_temperature_greedy_fast_path.py
```

- [ ] **Step 6: Commit and push**

```bash
git add -- \
  tools/profile_zero_temperature_greedy_fast_path.py \
  tools/test_profile_zero_temperature_greedy_fast_path.py
git -c core.hooksPath=/dev/null commit \
  -m "test(perf): profile greedy sampling fast path" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

### Task 4: Producer gate and independent verifier

**Files:**

- Create: `tools/zero_temperature_greedy_fast_path_gate.py`
- Create: `tools/zero_temperature_greedy_fast_path_verify.py`
- Create: `tools/test_zero_temperature_greedy_fast_path_gate.py`
- Create: `tools/test_zero_temperature_greedy_fast_path_verify.py`

**Interfaces:**

- Producer classifications exactly match the design.
- Independent verifier must not import the producer gate or worker
  summarization/classification functions.

- [ ] **Step 1: Write producer RED tests**

Use a complete synthetic 30-row fixture plus 18 binary logits sidecars. Prove
GO, token mismatch, text mismatch, logit max/mean/argmax failures, incomplete
optimized steps, median failure, p95 failure, protected regression, duplicate
identity, stale source hash, stale sidecar hash, and non-finite metric.

- [ ] **Step 2: Implement producer validation**

The producer must:

1. require exactly 30 performance rows and 18 correctness rows;
2. pair performance rows by `(bucket, repetition)`;
3. pair correctness rows by `(bucket, sampling_point)`;
4. read the retained float32 bytes and independently calculate max/mean
   absolute difference and argmax;
5. reconstruct medians and nearest-rank P95/P99 from raw TPOT samples;
6. apply thresholds in fixed precedence order;
7. report benefit and cost;
8. write `comparison.json`, `gate.json`, and `manifest.sha256`.

- [ ] **Step 3: Write independent-verifier RED tests**

Prove that it independently reconstructs the GO fixture and rejects producer
comparison drift, classification drift, missing sidecars, stale sidecar
digests, missing manifest entries, and stale primary digests.

- [ ] **Step 4: Implement the verifier separately**

Duplicate the small percentile, pairing, binary decoding, threshold, and
manifest logic intentionally. Emit:

```python
{
    "schema_version":
        "zero-temperature-greedy-fast-path.independent-verification.v1",
    "status": "PASS",
    "reconstructed_classification": classification,
    "comparison_sha256": comparison_digest,
    "manifest_sha256": manifest_digest,
}
```

- [ ] **Step 5: Run GREEN and syntax checks**

```bash
python3 tools/test_zero_temperature_greedy_fast_path_gate.py
python3 tools/test_zero_temperature_greedy_fast_path_verify.py
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-greedy-gate-pycache \
  python3 -m py_compile \
  tools/zero_temperature_greedy_fast_path_gate.py \
  tools/zero_temperature_greedy_fast_path_verify.py
```

- [ ] **Step 6: Commit and push**

```bash
git add -- \
  tools/zero_temperature_greedy_fast_path_gate.py \
  tools/zero_temperature_greedy_fast_path_verify.py \
  tools/test_zero_temperature_greedy_fast_path_gate.py \
  tools/test_zero_temperature_greedy_fast_path_verify.py
git -c core.hooksPath=/dev/null commit \
  -m "test(perf): gate greedy sampling fast path" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

### Task 5: Safe remote controller

**Files:**

- Create: `tools/run_zero_temperature_greedy_fast_path_remote.py`
- Create: `tools/test_run_zero_temperature_greedy_fast_path_remote.py`

**Interfaces:**

- Reuses the import-safe SSH, Kerberos, admission, upload, polling, and
  chunked-download helpers from the replay-aware controller.
- Uses a distinct remote namespace:

```text
/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/
  zero-temperature-greedy-fast-path/
```

- [ ] **Step 1: Write controller RED tests**

Cover:

- immutable and valid run tags;
- every remote path below the approved root;
- Kerberos lifetime at least 5,400 seconds;
- strict clean-GPU selection;
- fixed remote Python and model existence;
- source commit equal to pushed HEAD;
- source archive limited to `tinyvllm/` and `tools/`;
- isolated runtime/cache environment;
- dependency-light and Torch-dependent preflight;
- incomplete and tampered download rejection.

- [ ] **Step 2: Implement the controller**

Use the existing ControlMaster-aware SSH helper. Preflight runs:

```text
tools/test_greedy_sampling_fast_path.py
tools/test_model_runner_spec_verify.py
tools/test_multi_sequence_cuda_graph_gate.py
tools/test_chunked_prefill.py
```

Launch the worker only after a second admission check confirms the selected
GPU UUID is still strict-clean. Poll from the local controller, then run both
gates remotely, download all files including logits sidecars, and rerun the
independent verifier locally.

- [ ] **Step 3: Run GREEN and syntax checks**

```bash
python3 tools/test_run_zero_temperature_greedy_fast_path_remote.py
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-greedy-controller-pycache \
  python3 -m py_compile \
  tools/run_zero_temperature_greedy_fast_path_remote.py
```

- [ ] **Step 4: Commit and push**

```bash
git add -- \
  tools/run_zero_temperature_greedy_fast_path_remote.py \
  tools/test_run_zero_temperature_greedy_fast_path_remote.py
git -c core.hooksPath=/dev/null commit \
  -m "test(perf): add greedy fast path remote gate" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

### Task 6: Execute Qwen3-0.6B Stage 1

**Files:**

- Create: `artifacts/zero_temperature_greedy_fast_path/<immutable-tag>/`

- [ ] **Step 1: Run local regression**

```bash
python3 tools/test_greedy_sampling_fast_path.py
python3 tools/test_model_runner_spec_verify.py
python3 tools/test_profile_zero_temperature_greedy_fast_path.py
python3 tools/test_zero_temperature_greedy_fast_path_gate.py
python3 tools/test_zero_temperature_greedy_fast_path_verify.py
python3 tools/test_run_zero_temperature_greedy_fast_path_remote.py
python3 tools/test_source_audit.py
git diff --check
```

- [ ] **Step 2: Launch one immutable Stage-1 run**

```bash
KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian \
TINYLLMFORGE_SSH_CONTROL_PATH=/tmp/ssh-sitian-10.232.195.203 \
python3 tools/run_zero_temperature_greedy_fast_path_remote.py \
  --run-tag 20260822-qwen3-06b-greedy-fast-r1 \
  --model-tier qwen3-0.6b \
  --source-commit "$(git rev-parse HEAD)"
```

The local controller waits and launches automatically when one strict-clean
GPU is available.

- [ ] **Step 3: Reconstruct locally**

```bash
PYTHONPATH=. python3 tools/zero_temperature_greedy_fast_path_gate.py \
  --run-dir \
  artifacts/zero_temperature_greedy_fast_path/\
20260822-qwen3-06b-greedy-fast-r1/primary
PYTHONPATH=. python3 tools/zero_temperature_greedy_fast_path_verify.py \
  --run-dir \
  artifacts/zero_temperature_greedy_fast_path/\
20260822-qwen3-06b-greedy-fast-r1/primary
```

Producer and verifier must agree on classification, comparison digest, and
manifest digest.

- [ ] **Step 4: Apply the promotion boundary**

On GO, enable only the proven batch-1 zero-temperature scope by default and
run a fresh confirmation bundle before any 8B work. On NO-GO, leave the flag
default-disabled, preserve the complete negative result, and do not run 8B.

### Task 7: Audit, handoff, final verification, and push

**Files:**

- Modify: `AGENT_HANDOFF_STATE.md`
- Modify: `docs/superpowers/audits/2026-08-16-phase1-completion-audit.md`

- [ ] **Step 1: Append exact evidence at EOF**

Record source and documentation commits, immutable tag, local/remote paths,
GPU admission, all artifact hashes, producer and independent classifications,
exact token/text and logit metrics, per-bucket median/P95/P99 TPOT, TTFT,
E2E, throughput, CUDA memory, avoided work, and Qwen3-8B eligibility.

- [ ] **Step 2: Build the prompt-to-artifact checklist**

Map every design requirement, file, test, command, gate field, sidecar,
manifest entry, and remote assertion to inspected evidence. Treat every
uncertainty as incomplete.

- [ ] **Step 3: Run final verification**

```bash
python3 tools/test_greedy_sampling_fast_path.py
python3 tools/test_model_runner_spec_verify.py
python3 tools/test_profile_zero_temperature_greedy_fast_path.py
python3 tools/test_zero_temperature_greedy_fast_path_gate.py
python3 tools/test_zero_temperature_greedy_fast_path_verify.py
python3 tools/test_run_zero_temperature_greedy_fast_path_remote.py
python3 tools/test_source_audit.py
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-greedy-final-pycache \
  python3 -m py_compile \
  tinyvllm/engine/greedy_sampling_fast_path.py \
  tinyvllm/engine/model_runner.py \
  tinyvllm/config.py \
  tools/profile_zero_temperature_greedy_fast_path.py \
  tools/zero_temperature_greedy_fast_path_gate.py \
  tools/zero_temperature_greedy_fast_path_verify.py \
  tools/run_zero_temperature_greedy_fast_path_remote.py
git diff --check
```

- [ ] **Step 4: Commit exact documentation paths and push**

```bash
git add -- \
  AGENT_HANDOFF_STATE.md \
  docs/superpowers/audits/2026-08-16-phase1-completion-audit.md
git -c core.hooksPath=/dev/null commit \
  -m "docs(perf): record greedy fast path evidence" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

- [ ] **Step 5: Completion audit**

Restate and verify:

1. one independently motivated optimization;
2. default-disabled safe implementation;
3. exact output and bounded logit evidence;
4. Qwen3-0.6B paired performance evidence;
5. benefit and cost metrics;
6. independent reconstruction;
7. immutable local and remote artifacts;
8. EOF audit/handoff, exact commits, and remote push;
9. Qwen3-8B only after a verified Stage-1 GO.

Do not claim completion until every item maps to current evidence and local
HEAD equals `origin/feat/kv-sparse-attention`.
