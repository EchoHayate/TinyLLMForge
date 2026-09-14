# KV8 + Quest Amortization-Aware Activation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a default-off host policy that enables Quest only when the current decode batch avoids at least a configured number of KV-block dequantizations, then classify it on a source-bound Qwen3-8B/A100 gate.

**Architecture:** A pure policy helper resolves Quest activation from host metadata and returns an auditable decision. `ModelRunner.prepare_decode()` publishes latest-step and cumulative telemetry, while the existing performance and needle harnesses carry the configuration and evidence into a new independent analyzer.

**Tech Stack:** Python 3, dataclasses, pytest, Bash, TinyLLMForge `LLM`, JSON/Markdown artifacts, SSH to the existing A100 host.

## Global Constraints

- Use only `/Users/bytedance/Desktop/TinyLLMForge`.
- Do not create a worktree or use subagents.
- Preserve all unrelated dirty/untracked files; use exact-path staging only.
- Do not run `klist`, wait on Kerberos TTL, or ask for `kinit`.
- Use `KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian` for SSH.
- Write remote data only below `/data00/home/sitian/tllm/kvcapacity-runs/`.
- Do not terminate foreign GPU or CPU processes.
- Keep `quest_min_saved_blocks=0` behavior-identical to the current runtime.
- Freeze the candidate at `quest_top_k_blocks=16`, `quest_min_seq_len=512`, and `quest_min_saved_blocks=128`.
- Freeze performance at Qwen3-8B, A100 80GB PCIe, TP1, eager decode, 640 KV blocks, context 8192, and batches `4,6,8,10,12,16,19`.
- Never reuse a failed or partial run tag.
- Report both benefit and cost; do not claim graph, TP, production, other-model, or universal-threshold results.

---

### Task 1: Implement and validate the pure activation policy

**Files:**
- Create: `tinyvllm/engine/quest_activation.py`
- Create: `tools/test_quest_activation_policy.py`
- Modify: `tinyvllm/config.py`

**Interfaces:**
- Consumes: requested top-k, minimum sequence length, minimum saved blocks, block size, per-sequence lengths/block counts, and an incompatibility flag
- Produces: `QuestActivationDecision` with `requested_top_k`, `resolved_top_k`, `min_seq_len`, `min_saved_blocks`, `saved_blocks`, `batch_size`, and `reason`

- [ ] **Step 1: Write policy RED tests**

Add table-driven tests covering:

```python
case = resolve_quest_activation(
    requested_top_k=16,
    min_seq_len=512,
    min_saved_blocks=128,
    block_size=256,
    sequence_lengths=[8192] * 4,
    sequence_block_counts=[32] * 4,
    incompatible_feature=False,
)
assert case.resolved_top_k == -1
assert case.saved_blocks == 64
assert case.reason == "below_saved_blocks"
```

and the B=6 fallback, B=8 activation, default-zero compatibility,
disabled, incompatible, short-sequence, insufficient-block, and
insufficient-pruning reasons.

- [ ] **Step 2: Run RED**

Run:

```bash
python3 -m pytest -q tools/test_quest_activation_policy.py
```

Expected: collection fails because `tinyvllm.engine.quest_activation` does not
exist.

- [ ] **Step 3: Implement the pure resolver**

Create:

```python
@dataclass(frozen=True)
class QuestActivationDecision:
    requested_top_k: int
    resolved_top_k: int
    min_seq_len: int
    min_saved_blocks: int
    saved_blocks: int | None
    batch_size: int
    reason: str
```

and:

```python
def resolve_quest_activation(
    *,
    requested_top_k: int,
    min_seq_len: int,
    min_saved_blocks: int,
    block_size: int,
    sequence_lengths: Sequence[int],
    sequence_block_counts: Sequence[int],
    incompatible_feature: bool,
) -> QuestActivationDecision:
```

Apply the reason order frozen in the spec and compute:

```python
saved_blocks = sum(
    max(0, blocks - requested_top_k)
    for blocks in sequence_block_counts
)
```

- [ ] **Step 4: Add config validation**

Add:

```python
quest_min_saved_blocks: int = 0
```

and reject booleans, non-integers, and negative values with:

```python
raise ValueError("quest_min_saved_blocks must be a non-negative integer")
```

Extend the RED tests to construct a minimal `Config` fixture and assert zero,
128, and invalid values.

- [ ] **Step 5: Run GREEN**

Run:

```bash
python3 -m pytest -q tools/test_quest_activation_policy.py
```

Expected: all policy and validation tests pass.

- [ ] **Step 6: Commit**

Stage only the three task files and commit:

```text
feat(quest): add amortization activation policy
```

### Task 2: Wire the policy and auditable telemetry into ModelRunner

**Files:**
- Modify: `tinyvllm/engine/model_runner.py`
- Modify: `tools/test_quest_activation_policy.py`

**Interfaces:**
- Consumes: `resolve_quest_activation(...)`
- Produces:
  - `ModelRunner.quest_activation_observation() -> dict | None`
  - `ModelRunner.quest_activation_summary() -> dict`

- [ ] **Step 1: Write ModelRunner telemetry RED tests**

Build a lightweight runner with fake sequences and assert that successive
publication calls:

```python
runner._publish_quest_activation(decision)
event = runner.quest_activation_observation()
assert event["observation_id"] == 1
assert event["resolved_top_k"] == -1
assert event["reason"] == "below_saved_blocks"
```

Then publish an active decision and require cumulative output:

```python
summary = runner.quest_activation_summary()
assert summary["steps"] == 2
assert summary["reason_counts"] == {
    "active": 1,
    "below_saved_blocks": 1,
}
assert summary["saved_blocks_min"] == 64
assert summary["saved_blocks_max"] == 128
```

- [ ] **Step 2: Run RED**

Run:

```bash
python3 -m pytest -q tools/test_quest_activation_policy.py \
  -k 'model_runner or telemetry'
```

Expected: failures because the publication/read APIs do not exist.

- [ ] **Step 3: Implement publication APIs**

Initialize a monotonically increasing id, latest event, and cumulative counts
in `ModelRunner.__init__`. Publish only host integers/strings and return copies
from readers so evidence consumers cannot mutate runtime state.

- [ ] **Step 4: Replace the inline Quest branch**

In `prepare_decode()`, call `resolve_quest_activation()` with:

```python
sequence_lengths=[len(seq) for seq in seqs]
sequence_block_counts=[seq.num_blocks for seq in seqs]
incompatible_feature=cartridge_active or am_compact_active
```

Publish the decision and pass only `decision.resolved_top_k` into
`set_context()`. Do not add `.item()`, CUDA events, or device reads.

- [ ] **Step 5: Prove default compatibility and threshold behavior**

Add source/runtime tests showing:

- threshold zero produces the same resolved top-k as the old branch;
- B=4/6 resolve `-1`;
- B=8 resolves `16`;
- no policy state is sent into the attention layer.

- [ ] **Step 6: Run GREEN and adjacent runtime tests**

Run:

```bash
python3 -m pytest -q \
  tools/test_quest_activation_policy.py \
  tools/test_model_runner_spec_verify.py \
  tools/test_native_verifier_attention.py
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit**

Stage only the two task files and commit:

```text
feat(quest): resolve adaptive activation in decode
```

### Task 3: Add activation provenance to the step-scaling harness

**Files:**
- Modify: `tools/test_kvcapacity_step_scaling.py`
- Modify: `tools/kvcapacity_step_scaling_worker.py`
- Modify: `tools/run_kvcapacity_step_scaling_remote.sh`

**Interfaces:**
- Consumes: `quest_min_saved_blocks` and ModelRunner telemetry
- Produces: requested configuration, resolved engine identity, per-step
  activation trace labels, and measured-window activation summary

- [ ] **Step 1: Write CLI/configuration RED tests**

Assert parsing and payload identity for:

```text
--quest-min-saved-blocks 128
```

and runner passthrough from:

```text
QUEST_MIN_SAVED_BLOCKS=128
```

- [ ] **Step 2: Write telemetry tracker RED tests**

Mirror `DispatchTracker` with `QuestActivationTracker`. Require:

- a new observation id is recorded;
- a repeated id becomes `unpublished`;
- no hook becomes `unobserved`;
- summaries contain reason counts, resolved-top-k counts, step count, and
  saved-block min/max;
- an event whose `resolved_top_k` contradicts its reason is marked invalid.

- [ ] **Step 3: Run RED**

Run:

```bash
python3 -m pytest -q tools/test_kvcapacity_step_scaling.py \
  -k 'quest or activation'
```

Expected: failures for the missing CLI, identity, and telemetry fields.

- [ ] **Step 4: Implement passthrough and evidence**

Thread the value through `parse_args`, `main`, `run`, `_load_engine`,
`_engine_identity`, and `build_payload`. Record activation observations beside
dispatch observations in `_measure_cell()` and add:

```python
"quest_activation_measured": summarise_quest_activation(measured_events)
```

- [ ] **Step 5: Run GREEN**

Run:

```bash
python3 -m pytest -q tools/test_kvcapacity_step_scaling.py
bash -n tools/run_kvcapacity_step_scaling_remote.sh
```

Expected: all tests pass and Bash syntax is valid.

- [ ] **Step 6: Commit**

Stage only the three task files and commit:

```text
test(kvcapacity): record adaptive Quest activation
```

### Task 4: Extend the fixed-prompt quality runner

**Files:**
- Modify: `tools/eval_needle.py`
- Modify: `tools/test_eval_needle_fixed_prompts.py`
- Create: `tools/run_kv8_quest_adaptive_quality_remote.sh`

**Interfaces:**
- Consumes: `quest_min_saved_blocks=128`
- Produces: source-bound KV8 full/adaptive quality artifacts with before/after
  activation-summary deltas

- [ ] **Step 1: Write quality RED tests**

Assert:

- `--quest-min-saved-blocks 128` reaches `LLM`;
- each result setting records an activation-summary delta;
- a fake runner whose counts change from `{"active": 2}` to `{"active": 7,
  "below_saved_blocks": 3}` yields per-setting counts of five active and three
  fallback steps;
- missing telemetry is represented as `null`, never silently as zero.

- [ ] **Step 2: Run RED**

Run:

```bash
python3 -m pytest -q tools/test_eval_needle_fixed_prompts.py \
  -k 'quest or activation'
```

Expected: failures because the new argument and summary snapshots do not exist.

- [ ] **Step 3: Implement summary snapshots**

Resolve the ModelRunner defensively, snapshot
`quest_activation_summary()` before and after each top-k setting, and subtract
numeric counters without affecting generation. Include requested threshold in
the JSON `args` block.

- [ ] **Step 4: Add the source-bound remote runner**

Copy only the established safety structure from
`run_kv8_quest_quality_remote.sh`, but run two engines:

```text
KV8 full: quest_top_k_blocks=-1, quest_min_saved_blocks=0
adaptive: quest_top_k_blocks=16, quest_min_saved_blocks=128
```

Archive source from the pushed revision, reject dirty source paths and reused
tags, keep all remote files below the approved root, and copy back only JSON,
source provenance, hashes, and compact logs needed for diagnosis.

- [ ] **Step 5: Run GREEN**

Run:

```bash
python3 -m pytest -q tools/test_eval_needle_fixed_prompts.py
bash -n tools/run_kv8_quest_adaptive_quality_remote.sh
```

Expected: all tests pass and Bash syntax is valid.

- [ ] **Step 6: Commit**

Stage only the three task files and commit:

```text
test(kvcapacity): add adaptive Quest quality evidence
```

### Task 5: Build the independent adaptive-policy gate analyzer

**Files:**
- Create: `tools/kvcapacity_kv8_quest_adaptive_gate.py`
- Create: `tools/test_kvcapacity_kv8_quest_adaptive_gate.py`

**Interfaces:**
- Consumes: KV8 full, fixed Quest, adaptive Quest performance bundles;
  source provenance; KV8 full/adaptive quality bundles; prior or rerun bf16
  wall-cell evidence
- Produces: `GO_KV8_QUEST_AMORTIZATION_POLICY`,
  `NO_GO_KV8_QUEST_AMORTIZATION_POLICY`, or
  `INCONCLUSIVE_KV8_QUEST_AMORTIZATION_POLICY`

- [ ] **Step 1: Write mutation RED tests**

Start with a passing fixture and mutate one property per test:

- missing B=6 or B=10;
- source/model/pool/seed/sample mismatch;
- adaptive B=4 active;
- adaptive B=8 fallback;
- fallback regression above 3%;
- active regression versus fixed Quest above 3%;
- B=19 excess-latency recovery below 50%;
- missing/repeated/invalid activation telemetry;
- adaptive quality contains no active step;
- overall quality loss above 5 pp;
- one depth loses above 20 pp;
- producer `hit` disagrees with independently recomputed `answer == magic`.

- [ ] **Step 2: Run RED**

Run:

```bash
python3 -m pytest -q tools/test_kvcapacity_kv8_quest_adaptive_gate.py
```

Expected: collection fails because the analyzer does not exist.

- [ ] **Step 3: Implement identity and telemetry verification**

Independently recompute expected activation with:

```python
expected_active = saved_blocks >= 128
```

Require all measured steps in B=4/6 to resolve full and all measured steps in
B>=8 to resolve top-16. Do not trust producer summary booleans.

- [ ] **Step 4: Implement performance and quality classification**

Compute:

```python
fallback_regression = adaptive_ms / kv8_ms - 1.0
active_regression = adaptive_ms / fixed_quest_ms - 1.0
recovery = (kv8_b19_ms - adaptive_b19_ms) / (kv8_b19_ms - bf16_b19_ms)
```

Recompute quality hits from normalized extracted answers and apply fixed
failure precedence: identity/completeness first, then performance/quality
thresholds.

- [ ] **Step 5: Emit JSON and Markdown**

Include every cell's three latencies, activation evidence, ratios, recovery,
quality deltas, failure lists, classification, and claim boundary.

- [ ] **Step 6: Run GREEN**

Run:

```bash
python3 -m pytest -q tools/test_kvcapacity_kv8_quest_adaptive_gate.py
python3 -m py_compile tools/kvcapacity_kv8_quest_adaptive_gate.py
```

Expected: all mutation tests pass.

- [ ] **Step 7: Commit and push the source-bound implementation**

Stage only Tasks 1-5 source/tests plus this plan if not already committed.
Commit any remaining cohesive analyzer changes as:

```text
test(kvcapacity): add adaptive Quest gate
```

Push and verify the exact remote SHA before launching remote experiments.

### Task 6: Run local verification and focused code review

**Files:**
- Review: all files changed by Tasks 1-5
- Create outside repository: temporary review report

**Interfaces:**
- Consumes: committed implementation
- Produces: local test evidence and a P0-P2 defect disposition

- [ ] **Step 1: Run focused suites**

Run:

```bash
python3 -m pytest -q \
  tools/test_quest_activation_policy.py \
  tools/test_kvcapacity_step_scaling.py \
  tools/test_eval_needle_fixed_prompts.py \
  tools/test_kvcapacity_kv8_quest_adaptive_gate.py
```

- [ ] **Step 2: Run adjacent suites**

Run:

```bash
python3 -m pytest -q \
  tools/test_model_runner_spec_verify.py \
  tools/test_native_verifier_attention.py \
  tools/test_kvcapacity_kv8_quest_gate.py
```

- [ ] **Step 3: Run static checks**

Run:

```bash
python3 -m py_compile \
  tinyvllm/engine/quest_activation.py \
  tinyvllm/engine/model_runner.py \
  tools/kvcapacity_step_scaling_worker.py \
  tools/eval_needle.py \
  tools/kvcapacity_kv8_quest_adaptive_gate.py
bash -n \
  tools/run_kvcapacity_step_scaling_remote.sh \
  tools/run_kv8_quest_adaptive_quality_remote.sh
git diff --check
```

- [ ] **Step 4: Review the exact diff**

Run the repository code-review workflow against the Task 1-5 commit range.
Fix all P0-P2 findings, rerun affected tests, and record rejected findings with
concrete evidence.

- [ ] **Step 5: Push and verify**

Push the reviewed source and require local HEAD to equal:

```bash
git ls-remote origin refs/heads/feat/kv-sparse-attention
```

### Task 7: Execute the remote three-arm and quality gate

**Files:**
- Create after execution: four new immutable run directories below
  `experiments/kvcapacity_step_scaling/`, named from the pushed source SHA and
  the `kv8`, `fixed`, `adaptive`, and `quality` arm labels

**Interfaces:**
- Consumes: exact pushed source SHA
- Produces: source-bound raw bundles and final adaptive classification

- [ ] **Step 1: Select a usable GPU without destructive action**

Use read-only SSH/NVIDIA queries. A card may have small incidental memory use
if the pinned 640-block engine can construct and the artifact records pre-load
free/foreign memory. Do not kill or modify another workload.

- [ ] **Step 2: Run the performance arms sequentially**

Use fresh tags with:

```text
SWEEP_GRID=8192:4,6,8,10,12,16,19
EXECUTION_PATHS_OVERRIDE=eager
KV_BLOCKS=640
WARMUP_STEPS=24
MEASURED_STEPS=24
```

Arms:

```text
KV8 full: KV_QUANT_BITS=8, QUEST_TOP_K_BLOCKS=-1
fixed Quest: KV_QUANT_BITS=8, QUEST_TOP_K_BLOCKS=16, QUEST_MIN_SAVED_BLOCKS=0
adaptive Quest: KV_QUANT_BITS=8, QUEST_TOP_K_BLOCKS=16, QUEST_MIN_SAVED_BLOCKS=128
```

- [ ] **Step 3: Run adaptive fixed-prompt quality**

Run the source-bound quality script with a fresh tag and the same GPU/model.
Require 25 paired scored cases in each arm and nonempty activation-summary
deltas.

- [ ] **Step 4: Verify raw artifacts before classification**

Check JSON parsing, local/remote SHA256 equality, exact source revision,
requested/resolved engine fields, seven complete cells per performance arm,
24 measured samples per cell, stable target batch, fully eager dispatch, and
policy-consistent activation evidence.

- [ ] **Step 5: Run the independent analyzer**

Generate:

```text
gate_report.json
gate_report.md
```

under a new compact repository evidence directory. Preserve the measured
classification even if it is NO_GO or INCONCLUSIVE.

### Task 8: Seal evidence, audit, and handoff

**Files:**
- Modify: `docs/kv-sparse-attention.md`
- Modify: `docs/qwen3-8b-fixes.md`
- Modify: `docs/superpowers/audits/2026-08-16-phase1-completion-audit.md`
- Modify: `AGENT_HANDOFF_STATE.md`
- Add: compact gate report and source provenance files

**Interfaces:**
- Consumes: verified local and remote evidence
- Produces: final claim, benefit/cost table, reproducible handoff, commit, push,
  and remote SHA proof

- [ ] **Step 1: Document benefit and cost**

Record:

- B=4/6 fallback overhead versus KV8 full;
- B>=8 adaptive overhead versus fixed Quest;
- B=19 excess-latency recovery;
- quality deltas overall and per depth;
- selector/telemetry/runtime complexity cost;
- exact hardware/model/topology/dispatch boundary.

- [ ] **Step 2: Update the audit and handoff**

Include immutable tags, source SHA, artifact hashes, commands, test counts,
review disposition, final classification, excluded untracked files, and the
next justified action.

- [ ] **Step 3: Run final verification**

Run the focused suites, analyzer against real artifacts, `git diff --check`,
JSON parsing, and shell/Python syntax checks again from the final tree.

- [ ] **Step 4: Commit with exact staging**

Stage only the runtime, tests, compact evidence, docs, audit, handoff, spec,
and plan belonging to this gate. Do not stage logs or older experiment
directories.

- [ ] **Step 5: Push and verify remote SHA**

Push `feat/kv-sparse-attention`, compare local HEAD with the remote branch SHA,
and report the final classification separately from implementation/test
completion.
