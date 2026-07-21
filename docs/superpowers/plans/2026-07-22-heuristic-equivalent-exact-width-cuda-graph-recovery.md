# Heuristic-Equivalent Exact-Width CUDA Graph Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and independently verify a diagnostic-only FlashAttention 2.6.3 CUDA Graph candidate whose explicit split and exact page-table width reproduce legacy auto-eager semantics for ragged multi-sequence decode.

**Architecture:** Add a Torch-free mirror of the FlashAttention 2.6.3 decode split heuristic and make every diagnostic step emit a complete graph identity. Replace the diagnostic's single padded-width graph with an exact-width graph cache keyed by batch, page-table width, effective split, model heads, page size, GPU SM count, and FlashAttention version. Preserve the 315-process Gate A/Gate B matrix and require the independent verifier to recompute every per-step policy decision before any production code is considered.

**Tech Stack:** Python 3.11, PyTorch 2.4.1+cu121, FlashAttention 2.6.3, CUDA Graphs, dependency-light Python tests, Qwen3-0.6B BF16 on A100 80GB PCIe.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Preserve commit `d130efb` and the approved design at `docs/superpowers/specs/2026-07-22-heuristic-equivalent-exact-width-cuda-graph-recovery-design.md`.
- Do not modify `/Users/bytedance/dev/TinyLLMForge`.
- Use Inline Execution; do not dispatch subagents.
- Keep the production batch-greater-than-one eager guard unchanged until a fresh independent canonical returns `EXACT_REPLAY_CORRECT`, `LEGACY_COMPATIBLE`, `POLICY_EXACT`, and zero structural failures.
- Do not weaken batches, trajectories, repetitions, warmups, measured steps, tensor/KV observations, token checks, tolerances, hashes, or thresholds.
- Gate A remains 189 isolated processes; Gate B remains 126 isolated processes; total remains 315.
- Rounded graph replay remains diagnostic-only and production-disabled.
- Do not expose arbitrary split tuning as a public configuration option.
- Remote GPU/model work runs only as `sitian@10.232.195.203` with `CUDA_VISIBLE_DEVICES=0`.
- Use SSH ControlMaster `/tmp/ssh-sitian-10.232.195.203`.
- Do not modify or synchronize the remote checkout, kill shared processes, clean shared `/tmp`, or move experiments to another GPU.
- Every remote model process uses unique dynamic `TINYVLLM_DIST_PORT` and `MASTER_PORT`.
- Selectively stage files; never use `git add -A`; never stage `experiments/`.
- Do not push unless explicitly requested.
- README and performance claims remain blocked until an independent production `GO`.

---

## File Structure

### Create

- `tinyvllm/engine/flash_attn_split_policy.py`
  - Torch-free FlashAttention 2.6.3 heuristic, graph identity, canonical serialization, and validation.

### Modify

- `tools/multi_sequence_cuda_graph_contract.py`
  - Replace fixed16 diagnostic identities with heuristic-exact-width identities while preserving matrix cardinality and frozen thresholds.
- `tools/diagnose_multi_sequence_cuda_graph.py`
  - Compute per-step policy, capture/reuse exact-width graphs, and emit per-step identity evidence.
- `tools/verify_multi_sequence_cuda_graph_diagnostic.py`
  - Recompute heuristic and graph identity from raw evidence; classify policy integrity independently.
- `tools/run_multi_sequence_cuda_graph_diagnostic_remote.py`
  - Rename run modes/artifact kind and preserve source-bound resumable orchestration.
- `tools/test_multi_sequence_cuda_graph_gate.py`
  - Add heuristic, graph identity, matrix, producer, verifier, and runner tests.
- `AGENT_HANDOFF_STATE.md`
  - Record the root cause, design boundary, commands, canonical result, and next conditional action.

### Must Remain Unchanged

- `tinyvllm/engine/model_runner.py`
  - The production `multi_sequence_decode` eager guard.
- `README.md`
  - No claim before production `GO`.

---

### Task 1: Implement the Torch-Free FlashAttention 2.6.3 Split Policy

**Files:**
- Create: `tinyvllm/engine/flash_attn_split_policy.py`
- Modify: `tools/test_multi_sequence_cuda_graph_gate.py`

**Interfaces:**
- Produces:

```python
@dataclass(frozen=True)
class FlashAttentionSplitInputs:
    batch_size: int
    num_query_heads: int
    num_kv_heads: int
    head_dim: int
    page_block_size: int
    page_table_width: int
    max_seqlen_q: int
    multi_processor_count: int


@dataclass(frozen=True)
class FlashAttentionGraphIdentity:
    graph_batch_size: int
    active_batch_size: int
    page_table_width: int
    effective_num_splits: int
    flash_attn_version: str
    multi_processor_count: int
    num_query_heads: int
    num_kv_heads: int
    head_dim: int
    page_block_size: int
    max_seqlen_q: int

    @property
    def sha256(self) -> str: ...


def flash_attn_263_decode_num_splits(
    inputs: FlashAttentionSplitInputs,
) -> int: ...


def build_flash_attn_263_graph_identity(
    *,
    graph_batch_size: int,
    inputs: FlashAttentionSplitInputs,
    flash_attn_version: str,
) -> FlashAttentionGraphIdentity: ...
```

- Consumes no Torch objects and imports only the Python standard library.

- [ ] **Step 1: Add failing known-vector tests**

Add tests that assert:

```python
def test_flash_attn_263_known_qwen3_a100_vectors():
    vectors = {
        (2, 1): 2,
        (3, 1): 2,
        (4, 1): 2,
        (5, 1): 2,
        (8, 2): 2,
        (9, 2): 2,
        (16, 3): 3,
    }
    for (batch_size, width), expected in vectors.items():
        inputs = split_policy.FlashAttentionSplitInputs(
            batch_size=batch_size,
            num_query_heads=16,
            num_kv_heads=8,
            head_dim=128,
            page_block_size=256,
            page_table_width=width,
            max_seqlen_q=1,
            multi_processor_count=108,
        )
        assert split_policy.flash_attn_263_decode_num_splits(inputs) == expected
```

Also test:

- `batch_size=22`, width `3` takes the upstream early-return path and returns `1`;
- identical graph identities have identical SHA256;
- width `1` versus `2`, split `2` versus `3`, and graph batch `8` versus `16` produce distinct identities;
- zero/negative sizes, unsupported page size, unsupported head dimension, non-divisible GQA, and `max_seqlen_q != 1` fail closed.

- [ ] **Step 2: Run the focused tests and verify failure**

Run:

```bash
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python \
  tools/test_multi_sequence_cuda_graph_gate.py
```

Expected: FAIL because `tinyvllm.engine.flash_attn_split_policy` does not exist.

- [ ] **Step 3: Implement exact upstream arithmetic**

Implement:

```python
def _ceildiv(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def flash_attn_263_decode_num_splits(inputs):
    inputs.validate()
    swapped = (
        inputs.max_seqlen_q == 1
        and inputs.num_query_heads > inputs.num_kv_heads
    )
    effective_heads = (
        inputs.num_kv_heads if swapped else inputs.num_query_heads
    )
    effective_seqlen_q = (
        inputs.num_query_heads // inputs.num_kv_heads
        if swapped
        else inputs.max_seqlen_q
    )
    block_n = (
        256 if inputs.head_dim <= 64
        else 128 if inputs.head_dim <= 128
        else 64
    )
    seqlen_k = inputs.page_table_width * inputs.page_block_size
    num_n_blocks = _ceildiv(seqlen_k, block_n)
    num_m_blocks = _ceildiv(effective_seqlen_q, 64)
    work = inputs.batch_size * effective_heads * num_m_blocks
    num_sms = inputs.multi_processor_count * 2
    if work >= 0.8 * num_sms:
        return 1
    max_splits = min(128, num_sms, num_n_blocks)
    candidates = []
    best = 0.0
    for num_splits in range(1, max_splits + 1):
        if (
            num_splits > 1
            and _ceildiv(num_n_blocks, num_splits)
            == _ceildiv(num_n_blocks, num_splits - 1)
        ):
            continue
        waves = work * num_splits / num_sms
        efficiency = waves / math.ceil(waves)
        candidates.append((num_splits, efficiency))
        best = max(best, efficiency)
    return next(
        num_splits
        for num_splits, efficiency in candidates
        if efficiency >= 0.85 * best
    )
```

Validation must explicitly require the frozen FA2 path:

```text
page_block_size % 256 == 0
head_dim <= 256
num_query_heads % num_kv_heads == 0
max_seqlen_q == 1
all dimensions positive
```

Use canonical JSON with sorted keys and compact separators for identity hashing.

- [ ] **Step 4: Run focused tests**

Run the same command.

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tinyvllm/engine/flash_attn_split_policy.py \
  tools/test_multi_sequence_cuda_graph_gate.py
git commit -m "feat: mirror flash attention split heuristic"
```

---

### Task 2: Replace Fixed16 Contracts with Per-Step Heuristic Identity

**Files:**
- Modify: `tools/multi_sequence_cuda_graph_contract.py`
- Modify: `tools/test_multi_sequence_cuda_graph_gate.py`

**Interfaces:**
- Produces policy names:

```text
legacy_eager_auto
candidate_eager_heuristic
exact_graph_heuristic
rounded_graph_heuristic
```

- Case IDs bind the policy family but do not claim one case-level split:

```text
b8__ragged-context__exact_graph_heuristic__fa2-263-exact-width__r0
```

- [ ] **Step 1: Add failing matrix tests**

Assert:

```text
Gate A processes = 189
Gate B processes = 126
unique case IDs = 315
batch sizes = 2,3,4,5,8,9,16
trajectories unchanged
repetitions = 0,1,2
warmups = 2
measured steps = 16
rtol = 1e-3
atol = 1e-2
```

Assert no heuristic candidate case stores `flash_attn_num_splits=16`.

- [ ] **Step 2: Run tests and verify failure**

Expected: failures reference old `fixed16` mode and policy names.

- [ ] **Step 3: Modify dataclasses and policy definitions**

Use:

```python
HEURISTIC_POLICY_NAME = "fa2_263_heuristic_exact_width"
SAME_POLICY_MODES = (
    "candidate_eager_heuristic",
    "exact_graph_heuristic",
    "rounded_graph_heuristic",
)
LEGACY_COMPATIBILITY_POLICIES = (
    "legacy_eager_auto",
    "candidate_eager_heuristic",
)
```

Keep `AUTO_FLASH_ATTN_NUM_SPLITS = 0` only for legacy eager. Candidate cases
must not have a frozen positive split at case construction time; their
effective split is derived and emitted per step.

- [ ] **Step 4: Run tests**

Expected: PASS with cardinalities unchanged.

- [ ] **Step 5: Commit**

```bash
git add tools/multi_sequence_cuda_graph_contract.py \
  tools/test_multi_sequence_cuda_graph_gate.py
git commit -m "test: freeze heuristic cuda graph contracts"
```

---

### Task 3: Add Exact-Width Per-Step Policy Construction

**Files:**
- Modify: `tools/diagnose_multi_sequence_cuda_graph.py`
- Modify: `tools/test_multi_sequence_cuda_graph_gate.py`

**Interfaces:**
- Produces:

```python
@dataclass(frozen=True)
class StepSplitPolicy:
    inputs: FlashAttentionSplitInputs
    identity: FlashAttentionGraphIdentity

    @property
    def effective_num_splits(self) -> int: ...


def build_step_split_policy(
    *,
    runner,
    dynamic_context: dict,
    active_batch_size: int,
    graph_batch_size: int,
    flash_attn_version: str,
) -> StepSplitPolicy: ...
```

- [ ] **Step 1: Add failing fake-runner tests**

Use dependency-light fake runner/config objects and fake tensors exposing
`.size()` to prove:

- page-table width comes from `dynamic_context["block_tables"].size(1)`;
- graph batch and active batch remain distinct fields;
- A100/Qwen3 vectors derive `2/2/2/3`;
- version other than exactly `2.6.3` is rejected;
- missing `num_attention_heads`, `num_key_value_heads`, `head_dim`, block size,
  or SM count is rejected.

- [ ] **Step 2: Run tests and verify failure**

Expected: `build_step_split_policy` is missing.

- [ ] **Step 3: Implement policy construction**

Read:

```text
runner.config.hf_config.num_attention_heads
runner.config.hf_config.num_key_value_heads
runner.config.hf_config.head_dim
runner.block_size
torch.cuda.get_device_properties(runner.kv_cache.device).multi_processor_count
dynamic_context["block_tables"].size(1)
```

Build the immutable policy and recompute the identity SHA256.

- [ ] **Step 4: Run tests**

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/diagnose_multi_sequence_cuda_graph.py \
  tools/test_multi_sequence_cuda_graph_gate.py
git commit -m "feat: derive per-step flash attention graph identity"
```

---

### Task 4: Replace the Single Diagnostic Graph with an Exact-Width Graph Cache

**Files:**
- Modify: `tools/diagnose_multi_sequence_cuda_graph.py`
- Modify: `tools/test_multi_sequence_cuda_graph_gate.py`

**Interfaces:**
- Replace:

```python
captured: CapturedDecodeGraph | None
```

with:

```python
graph_cache: dict[
    FlashAttentionGraphIdentity,
    CapturedDecodeGraph,
]
```

- `CapturedDecodeGraph` stores its immutable identity.

- [ ] **Step 1: Add failing cache and shape tests**

Test:

- exact graph capture allocates `block_tables` with exactly
  `identity.page_table_width` columns;
- two steps with identical identity reuse one graph;
- a width transition creates a second graph;
- a split transition creates a second graph;
- replay rejects an identity mismatch;
- capture snapshots and restores all active write slots;
- rounded graph identity uses rounded graph batch but exact runtime width.

- [ ] **Step 2: Run tests and verify failure**

Expected: current capture always allocates
`ceil(max_model_len / block_size)` columns and has no graph cache.

- [ ] **Step 3: Implement exact-width capture**

Change `_capture_decode_graph()` so:

```python
block_tables = torch.zeros(
    identity.graph_batch_size,
    identity.page_table_width,
    dtype=torch.int32,
    device=device,
)
```

Copy only the exact width and reject any dynamic width mismatch.

Install:

```python
with temporary_flash_attn_num_splits(
    identity.effective_num_splits
):
    warmup_and_capture()
```

Snapshot write slots before warmup/capture and restore them in `finally`.

- [ ] **Step 4: Implement cache lookup per decode step**

Before each eager or graph execution:

```python
step_policy = build_step_split_policy(...)
captured = graph_cache.get(step_policy.identity)
if captured is None:
    captured = _capture_decode_graph(..., step_policy.identity)
    graph_cache[step_policy.identity] = captured
```

Candidate eager also uses the step's explicit split. Legacy eager remains
`num_splits=0`.

- [ ] **Step 5: Run tests**

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tools/diagnose_multi_sequence_cuda_graph.py \
  tools/test_multi_sequence_cuda_graph_gate.py
git commit -m "feat: cache exact-width diagnostic cuda graphs"
```

---

### Task 5: Emit Complete Per-Step Policy Evidence

**Files:**
- Modify: `tools/diagnose_multi_sequence_cuda_graph.py`
- Modify: `tools/test_multi_sequence_cuda_graph_gate.py`

**Interfaces:**
- Every raw, layer, and KV row includes:

```python
{
    "split_policy_name": "fa2_263_heuristic_exact_width",
    "flash_attn_version": "2.6.3",
    "page_table_width": int,
    "effective_num_splits": int,
    "heuristic_batch_size": int,
    "heuristic_num_query_heads": int,
    "heuristic_num_kv_heads": int,
    "heuristic_head_dim": int,
    "heuristic_page_block_size": int,
    "heuristic_max_seqlen_q": 1,
    "heuristic_multi_processor_count": int,
    "graph_batch_size": int,
    "graph_identity_sha256": str,
}
```

- [ ] **Step 1: Add failing schema tests**

Assert every measured step has the fields above and that raw/layer/KV rows for
the same case and step agree exactly.

- [ ] **Step 2: Run tests and verify failure**

Expected: current rows only contain case-level fixed split evidence.

- [ ] **Step 3: Add `step_policy_evidence()`**

Serialize directly from the immutable dataclasses. Do not accept caller-owned
dictionaries.

- [ ] **Step 4: Add case-level graph identity summary**

`case_result.json` records:

```python
"graph_identities": [
    {
        "sha256": identity.sha256,
        "page_table_width": identity.page_table_width,
        "effective_num_splits": identity.effective_num_splits,
        "graph_batch_size": identity.graph_batch_size,
    },
]
```

ordered by first use.

- [ ] **Step 5: Run tests**

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tools/diagnose_multi_sequence_cuda_graph.py \
  tools/test_multi_sequence_cuda_graph_gate.py
git commit -m "feat: emit per-step cuda graph policy evidence"
```

---

### Task 6: Recompute Policy Integrity in the Independent Verifier

**Files:**
- Modify: `tools/verify_multi_sequence_cuda_graph_diagnostic.py`
- Modify: `tools/test_multi_sequence_cuda_graph_gate.py`

**Interfaces:**
- Produces:

```python
"policy_integrity": "POLICY_EXACT" | "POLICY_DRIFT" | "INCOMPLETE"
```

- [ ] **Step 1: Add failing verifier fixtures**

Create complete synthetic fixtures and mutate one field at a time:

- missing page-table width;
- wrong effective split;
- wrong SM count;
- graph identity hash mismatch;
- raw/layer/KV policy disagreement;
- wrong FlashAttention version;
- graph candidate using auto split;
- candidate eager and graph using different explicit split;
- case-level graph identity summary omits a used identity.

Each mutation must return `POLICY_DRIFT` or `INCOMPLETE`, never `GO`.

- [ ] **Step 2: Run tests and verify failure**

Expected: verifier does not know the new policy fields.

- [ ] **Step 3: Implement independent recomputation**

For every measured row:

1. construct `FlashAttentionSplitInputs` from serialized primitive fields;
2. recompute the expected split;
3. construct the expected graph identity;
4. compare every serialized field and SHA256;
5. compare policy evidence across raw/layer/KV rows;
6. compare used identities with the case-level summary.

Do not import or trust producer helper functions other than the standalone pure
heuristic module.

- [ ] **Step 4: Update classifications**

Diagnostic `GO` requires:

```python
exact_classification == "EXACT_REPLAY_CORRECT"
legacy_compatibility == "LEGACY_COMPATIBLE"
policy_integrity == "POLICY_EXACT"
not structural_failures
```

- [ ] **Step 5: Run tests**

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tools/verify_multi_sequence_cuda_graph_diagnostic.py \
  tools/test_multi_sequence_cuda_graph_gate.py
git commit -m "feat: verify heuristic cuda graph policy integrity"
```

---

### Task 7: Update Source-Bound Remote Orchestration

**Files:**
- Modify: `tools/run_multi_sequence_cuda_graph_diagnostic_remote.py`
- Modify: `tools/test_multi_sequence_cuda_graph_gate.py`

**Interfaces:**
- Run modes:

```text
heuristic-exact-width-preflight
heuristic-exact-width-smoke
heuristic-exact-width-canonical
```

- [ ] **Step 1: Add failing runner tests**

Assert:

- canonical schedules exactly 315 cases;
- source manifest includes the new heuristic module;
- process rows require per-step policy artifacts;
- resume rejects old fixed16 artifacts;
- 630 ports are unique;
- remote command fixes `CUDA_VISIBLE_DEVICES=0`;
- SSH user is `sitian`;
- no `rsync`, remote checkout mutation, shared cleanup, or kill command exists.

- [ ] **Step 2: Run tests and verify failure**

Expected: old mode names and fixed16 manifest metadata fail.

- [ ] **Step 3: Update runner metadata**

Use:

```python
"kind": "heuristic_exact_width_recovery"
"flash_attn_version": "2.6.3"
"policy_name": "fa2_263_heuristic_exact_width"
```

Bind the source tree hash to all modified source files.

- [ ] **Step 4: Run tests**

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/run_multi_sequence_cuda_graph_diagnostic_remote.py \
  tools/test_multi_sequence_cuda_graph_gate.py
git commit -m "feat: orchestrate heuristic cuda graph gates"
```

---

### Task 8: Run the Full Local Verification Suite

**Files:**
- No source changes unless a test exposes a defect.

- [ ] **Step 1: Run focused dependency-light tests**

```bash
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python \
  tools/test_multi_sequence_cuda_graph_gate.py
```

Expected: PASS.

- [ ] **Step 2: Run context and model-runner regression tests**

```bash
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python \
  tools/test_context_modes.py
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python \
  tools/test_model_runner_spec_verify.py
```

Expected: PASS.

- [ ] **Step 3: Compile changed Python files**

```bash
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python -m py_compile \
  tinyvllm/engine/flash_attn_split_policy.py \
  tools/multi_sequence_cuda_graph_contract.py \
  tools/diagnose_multi_sequence_cuda_graph.py \
  tools/verify_multi_sequence_cuda_graph_diagnostic.py \
  tools/run_multi_sequence_cuda_graph_diagnostic_remote.py
```

Expected: exit `0`.

- [ ] **Step 4: Check repository whitespace and production guard**

```bash
git diff --check
rg -n \
  'multi_sequence_decode = mode == "decode" and input_ids.size\\(0\\) > 1' \
  tinyvllm/engine/model_runner.py
```

Expected: no diff errors and exactly one unchanged production guard.

---

### Task 9: Run Remote Preflight and Direct Kernel Probes

**Files:**
- Artifacts only under `experiments/cuda_graph/`; never stage them.

- [ ] **Step 1: Verify SSH and environment**

```bash
ssh -S /tmp/ssh-sitian-10.232.195.203 \
  -o ControlMaster=no -o BatchMode=yes \
  sitian@10.232.195.203 \
  'CUDA_VISIBLE_DEVICES=0 /data00/home/sitian/sitian-workspace01/tllm/env/bin/python - <<'"'"'PY'"'"'
import flash_attn, torch
print(flash_attn.__version__)
print(torch.__version__)
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_properties(0).multi_processor_count)
PY'
```

Expected:

```text
2.6.3
2.4.1+cu121
NVIDIA A100 80GB PCIe
108
```

- [ ] **Step 2: Run source-bound preflight**

```bash
RUN_TAG="qwen3-06b-heuristic-exact-width-preflight-$(date +%Y%m%d-%H%M%S)"
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python \
  tools/run_multi_sequence_cuda_graph_diagnostic_remote.py \
  heuristic-exact-width-preflight \
  --run-tag "$RUN_TAG" \
  --ssh-control-path /tmp/ssh-sitian-10.232.195.203
```

Expected: environment and source identity PASS without loading the full model
matrix.

- [ ] **Step 3: Run direct auto-versus-explicit probes**

Cover at least:

```text
(batch,width,expected split)
(2,1,2)
(5,1,2)
(8,2,2)
(9,2,2)
(16,3,3)
```

Require bitwise equality between `num_splits=0` and the pure-function result.

- [ ] **Step 4: Run attention-level CUDA Graph positive and negative controls**

Positive:

```text
capture width == runtime width -> bitwise equal
```

Negative:

```text
capture width padded larger than runtime width -> at least one BF16 difference
```

The negative control proves the probe remains sensitive to the original
failure mechanism.

---

### Task 10: Run a Model Smoke with a Page-Boundary Transition

**Files:**
- Artifacts only under `experiments/cuda_graph/`; never stage them.

- [ ] **Step 1: Run smoke matrix**

```bash
RUN_TAG="qwen3-06b-heuristic-exact-width-smoke-$(date +%Y%m%d-%H%M%S)"
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python \
  tools/run_multi_sequence_cuda_graph_diagnostic_remote.py \
  heuristic-exact-width-smoke \
  --run-tag "$RUN_TAG" \
  --ssh-control-path /tmp/ssh-sitian-10.232.195.203
```

Smoke must include:

```text
batch 5, width 1
batch 8, width 2
batch 16, width 3
one sequence that crosses a 256-token page boundary during measured decode
legacy auto eager
candidate explicit heuristic eager
exact-width graph
rounded graph negative classification
```

- [ ] **Step 2: Run independent verifier**

```bash
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python \
  tools/verify_multi_sequence_cuda_graph_diagnostic.py \
  "experiments/cuda_graph/$RUN_TAG"
```

Expected for smoke:

```text
POLICY_EXACT
EXACT_REPLAY_CORRECT
LEGACY_COMPATIBLE
zero structural failures
```

Rounded replay may remain corrupt.

- [ ] **Step 3: Stop on any smoke failure**

Return to systematic debugging. Do not run canonical on:

- policy drift;
- missing identity transition;
- exact replay mismatch;
- auto-versus-explicit mismatch;
- token mismatch;
- structural failure.

---

### Task 11: Run the Fresh 315-Process Canonical Hard Checkpoint

**Files:**
- Artifacts only under `experiments/cuda_graph/`; never stage them.

- [ ] **Step 1: Launch/resume canonical**

```bash
RUN_TAG="qwen3-06b-heuristic-exact-width-canonical-$(date +%Y%m%d-%H%M%S)"
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python \
  tools/run_multi_sequence_cuda_graph_diagnostic_remote.py \
  heuristic-exact-width-canonical \
  --run-tag "$RUN_TAG" \
  --ssh-control-path /tmp/ssh-sitian-10.232.195.203
```

Poll the same run until all 315 isolated cases are terminal. Resume only
source-identical completed rows.

- [ ] **Step 2: Run independent verifier**

```bash
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python \
  tools/verify_multi_sequence_cuda_graph_diagnostic.py \
  "experiments/cuda_graph/$RUN_TAG"
```

- [ ] **Step 3: Audit completeness independently**

Verify:

```text
315 unique case IDs
630 unique ports
189 Gate A processes
126 Gate B processes
5040 raw rows
5040 layer rows
5040 KV rows
all referenced artifacts exist
all hashes recompute
all policy rows recompute
```

- [ ] **Step 4: Apply the hard checkpoint**

Proceed only if:

```text
exact_classification == EXACT_REPLAY_CORRECT
legacy_compatibility == LEGACY_COMPATIBLE
policy_integrity == POLICY_EXACT
structural_failures == 0
```

If any condition fails:

- classify the result `NO_GO`;
- leave production unchanged;
- do not run a production performance gate;
- record the failure in `AGENT_HANDOFF_STATE.md`.

---

### Task 12: Record the Diagnostic Result and Stop at the Production Gate

**Files:**
- Modify: `AGENT_HANDOFF_STATE.md`

- [ ] **Step 1: Record authoritative artifacts**

Include:

- branch and commit;
- source tree SHA256;
- run directory;
- environment identity;
- exact commands;
- matrix cardinality and completeness audit;
- heuristic vectors;
- graph identity count and transitions;
- exact, rounded, compatibility, and policy classifications;
- token mismatches, if any;
- what the result proves and does not prove.

- [ ] **Step 2: State the production boundary**

Explicitly record:

```text
No production performance number exists yet.
The batch>1 eager guard remains unchanged.
README remains unchanged.
```

- [ ] **Step 3: Run final local verification**

Repeat Task 8 commands and:

```bash
git status --short --branch
git diff --check
```

- [ ] **Step 4: Commit only tracked diagnostic/docs changes**

```bash
git add AGENT_HANDOFF_STATE.md
git commit -m "docs: record heuristic cuda graph diagnostic"
```

Do not stage `experiments/`.

---

## Conditional Follow-Up

Only after Task 11 returns diagnostic `GO`, write a new production design and
implementation plan covering:

```text
exact identity allowlist
lazy versus startup graph capture
maximum graph count
maximum page-table width
graph memory and initialization budgets
identity hit/fallback telemetry
source-bound arrival-load performance gate
rollback and default-off configuration
```

Do not implement production dispatch in this plan.

