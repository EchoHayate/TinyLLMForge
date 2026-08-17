# Blockwise Speculative Verifier KV-Offload Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task inline. Steps use checkbox (`- [ ]`) syntax for tracking. Do not dispatch subagents because the user has not authorized delegation.

**Goal:** Make generic fixed-Q speculative verification stream logical KV history through a bounded GPU staging cache, preserving exact attention, transactional KV commit/rollback, and real offload movement so 16K/32K batch-1/batch-4 correctness gates can run with 68 GPU KV slots.

**Architecture:** Add an opt-in streaming policy to speculative residency preparation, preserve logical verifier block tables in blockwise mode, and route spec-verify attention through a new multi-query blockwise online-softmax helper. Validate the core with dependency-light tests and dense tensor parity, then run an isolated loaded-model correctness/movement campaign with independent source-hash verification.

**Tech Stack:** Python 3, PyTorch, CUDA, TinyLLMForge `KVOffloadMVP0`, generic speculative runtime, online softmax, JSON, SHA-256, pytest, Bash, Kerberos/GSSAPI SSH.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Do not stage, commit, switch branches, stash, reset, push, or run `git clean`.
- Keep runtime, Scheduler, verifier, residency, and attention code free of model-name and proposal-source branches.
- Keep variable-Q behavior unchanged: distinct fixed Q values are grouped separately without padding.
- Keep accepted KV in place; reject only the unused reserved suffix; do not replay accepted tokens or copy accepted KV.
- Keep recurrent/convolution non-KV state fail-closed without transaction support.
- Keep nonzero temperature fail-closed.
- Use only real `KVOffloadMVP0` counters for H2D/D2H evidence.
- Keep classification `NOT_PROMOTABLE`.
- Use `sitian@10.232.195.203`, `KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian`, GPU 7, `ControlMaster=no`, and `ControlPath=none`.
- The remote model environment has no pytest; local pytest is test authority and remote preflight uses `py_compile`.
- Do not use `tools/profile_ngram_commit.py` as evidence.

---

### Task 1: Add Streaming Speculative Residency Preparation

**Files:**
- Modify: `tinyvllm/engine/speculative_residency.py`
- Modify: `tinyvllm/engine/model_runner.py`
- Modify: `tools/test_speculative_residency.py`
- Modify: `tools/test_model_runner_spec_verify.py`

**Interfaces:**
- Extends: `SpeculativeResidencyParticipant.prepare_batch(ticket_id, rows, *, stage_all_original_blocks: bool = True)`.
- Preserves: existing default prepare behavior.
- Produces: `ModelRunner.prepare_speculative_residency_batch()` choosing streaming mode from generic KV-offload blockwise configuration.

- [x] **Step 1: Write the failing participant streaming tests**

Add a manager fixture that records generation binds and `ensure_resident`
arguments. Add:

```python
def test_prepare_batch_streaming_stages_only_materialized_write_blocks():
    manager = RecordingManager()
    participant = SpeculativeResidencyParticipant(
        participant_id=0,
        manager=manager,
        block_size=4,
    )
    row = SpeculativeResidencyPrepareRow(
        sequence_id=7,
        original_block_identities=((10, 1), (11, 1), (12, 1)),
        reserved_block_identities=((20, 2),),
        proxy_block_table=(10, 11, 12, 20),
        logical_slots=(11, 12),
    )

    participant.prepare_batch(
        9,
        (row,),
        stage_all_original_blocks=False,
    )

    assert manager.bind_calls == [
        (10, 1), (11, 1), (12, 1), (20, 2)
    ]
    assert manager.ensure_calls == [
        {
            "blocks": [12],
            "require_valid": True,
            "protected_logical_blocks": {12, 20},
            "wait": True,
        },
        {
            "blocks": [20],
            "require_valid": False,
            "protected_logical_blocks": {12, 20},
            "wait": True,
        },
    ]
```

Add separate tests proving:

```text
default stage_all_original_blocks=True stages [10, 11, 12]
logical slots wholly inside an original block stage that original block
multiple rows stage their own materialized original/reserved write blocks
non-bool stage_all_original_blocks is rejected
failure cleanup discards only resident reserved identities
```

- [x] **Step 2: Run participant RED**

Run:

```bash
python3 -m pytest \
  tools/test_speculative_residency.py \
  -k 'streaming or stage_all_original' \
  -q
```

Expected: FAIL because `prepare_batch()` does not accept
`stage_all_original_blocks`.

- [x] **Step 3: Implement minimal streaming prepare**

In `SpeculativeResidencyParticipant.prepare_batch()`:

```python
if not isinstance(stage_all_original_blocks, bool):
    raise ValueError(
        "stage_all_original_blocks must be a boolean"
    )
```

Derive:

```python
materialized_original = [
    block_id
    for block_id, _ in materialized_ids
    if block_id in original_identity_by_block
]
reserved_materialized = [
    block_id
    for block_id, _ in materialized_ids
    if block_id in reserved_identity_by_block
]
original_read_blocks = (
    original_blocks
    if stage_all_original_blocks
    else materialized_original
)
protected = set(
    original_read_blocks + reserved_materialized
)
```

Call `ensure_resident()` only when the corresponding list is non-empty.
Preserve ticket materialization metadata and cleanup behavior.

- [x] **Step 4: Run participant GREEN**

Run:

```bash
python3 -m pytest tools/test_speculative_residency.py -q
```

Expected: all speculative residency tests PASS.

- [x] **Step 5: Write the failing ModelRunner policy test**

Add an AST-loaded test:

```python
def test_model_runner_blockwise_residency_prepare_streams_original_history():
    calls = []
    runner = SimpleNamespace(
        config=SimpleNamespace(
            kv_offload_mvp0=True,
            kv_offload_blockwise_decode=True,
        ),
        _speculative_residency_participant=lambda: SimpleNamespace(
            prepare_batch=lambda ticket_id, rows, **kwargs: (
                calls.append((ticket_id, rows, kwargs))
                or FakeResult()
            ),
        ),
        _speculative_residency_result_dict=lambda result: {
            "status": "prepared",
        },
    )

    method = _load_model_runner_method(
        "prepare_speculative_residency_batch"
    )
    method(runner, 4, ("row",))

    assert calls == [
        (
            4,
            ("row",),
            {"stage_all_original_blocks": False},
        )
    ]
```

Add a full-attention test expecting `True`.

- [x] **Step 6: Run ModelRunner RED**

Run:

```bash
python3 -m pytest \
  tools/test_model_runner_spec_verify.py \
  -k speculative_residency_batch \
  -q
```

Expected: FAIL because ModelRunner does not pass the policy.

- [x] **Step 7: Implement and run ModelRunner GREEN**

Pass:

```python
stage_all_original_blocks=not bool(
    self.config.kv_offload_mvp0
    and self.config.kv_offload_blockwise_decode
)
```

Run:

```bash
python3 -m pytest \
  tools/test_model_runner_spec_verify.py \
  -k speculative_residency_batch \
  -q
```

Expected: PASS.

---

### Task 2: Preserve Logical Spec-Verify Metadata in Blockwise Mode

**Files:**
- Modify: `tinyvllm/utils/context.py`
- Modify: `tinyvllm/engine/model_runner.py`
- Modify: `tools/test_context_modes.py`
- Modify: `tools/test_model_runner_spec_verify.py`

**Interfaces:**
- Extends: `Context` with verifier blockwise plan-cache state.
- Preserves: existing `set_context()` call compatibility.
- Produces: blockwise `prepare_spec_verify_batch()` that maps only write slots.

- [x] **Step 1: Write failing context-field tests**

Add:

```python
def test_spec_verify_context_preserves_blockwise_logical_metadata():
    set_context(
        mode="spec_verify",
        kv_offload_manager="manager",
        kv_offload_blockwise_decode=True,
        kv_offload_blockwise_blocks=8,
        kv_offload_logical_block_tables=[[10, 11]],
        kv_offload_context_lens=[300],
        kv_offload_write_blocks=[11],
        spec_verify_query_lens=(4,),
    )
    context = get_context()
    assert context.kv_offload_blockwise_decode is True
    assert context.kv_offload_logical_block_tables == [[10, 11]]
    assert context.kv_offload_context_lens == [300]
    assert context.kv_offload_write_blocks == [11]
    assert context.spec_verify_query_lens == (4,)
    assert context.kv_offload_spec_verify_window_plan_cache is None
```

- [x] **Step 2: Run context RED**

Run:

```bash
python3 -m pytest \
  tools/test_context_modes.py \
  -k blockwise_logical_metadata \
  -q
```

Expected: FAIL because
`kv_offload_spec_verify_window_plan_cache` is absent.

- [x] **Step 3: Add minimal context cache fields**

Add:

```python
kv_offload_spec_verify_window_plan_cache: object | None = None
kv_offload_spec_verify_position_template_cache: object | None = None
kv_offload_spec_verify_window_mask_cache: object | None = None
```

The fields are internal caches and do not require new `set_context()` input
parameters.

- [x] **Step 4: Run context GREEN**

Run:

```bash
python3 -m pytest tools/test_context_modes.py -q
```

Expected: PASS.

- [x] **Step 5: Write failing blockwise preparation tests**

Extend the existing offload spec-verify fixtures. For one row with four
logical history blocks and two verifier write positions, assert:

```python
def test_blockwise_offload_spec_verify_keeps_logical_rows_and_maps_only_writes():
    _, _, metadata = runner.prepare_spec_verify_batch(
        (item,),
        residency_ticket_id=5,
    )

    assert manager.map_block_rows_calls == []
    assert manager.map_slots_calls == [
        ([10, 11, 12, 20], [14, 15])
    ]
    assert metadata.rows[0].block_table == (
        10, 11, 12, 20
    )
    context = current_context()
    assert context.block_tables.tolist() == [
        [10, 11, 12, 20]
    ]
    assert context.kv_offload_logical_block_tables == [
        [10, 11, 12, 20]
    ]
    assert context.kv_offload_context_lens == [16]
    assert context.kv_offload_write_blocks == [20]
    assert context.kv_offload_blockwise_decode is True
```

Add tests proving:

```text
full-attention offload still maps full block rows
blockwise prefill no longer fails compatibility by itself
blockwise decode requires KVOffloadMVP0
quantized KV remains rejected
mixed batch remains rejected
missing prepared ticket remains rejected
```

- [x] **Step 6: Run preparation RED**

Run:

```bash
python3 -m pytest \
  tools/test_model_runner_spec_verify.py \
  -k 'blockwise_offload_spec_verify or blockwise_prefill' \
  -q
```

Expected: FAIL because compatibility rejects blockwise mode and preparation
maps all visible rows.

- [x] **Step 7: Implement blockwise preparation**

In `_validate_spec_verify_compatibility()`:

- remove `kv_offload_blockwise_decode` and
  `kv_offload_blockwise_prefill` from the unconditional unsupported tuple;
- reject blockwise decode when `kv_offload_mvp0` is false;
- retain all other existing guards.

In `prepare_spec_verify_batch()`:

```python
blockwise_offload = bool(
    self.config.kv_offload_mvp0
    and self.config.kv_offload_blockwise_decode
)
```

When blockwise:

- keep `visible_block_table` logical;
- map only `plan.logical_slots`;
- derive ordered unique write block IDs from those logical slots;
- call `set_context()` with manager, logical rows, host context lengths,
  write blocks, blockwise window size, and fixed query lengths.

When non-blockwise, preserve the existing full mapping path.

- [x] **Step 8: Run preparation GREEN**

Run:

```bash
python3 -m pytest \
  tools/test_model_runner_spec_verify.py \
  tools/test_context_modes.py \
  -q
```

Expected: PASS.

---

### Task 3: Add Exact Blockwise Multi-Query Verifier Attention

**Files:**
- Modify: `tinyvllm/layers/attention.py`
- Modify: `tools/test_native_verifier_attention.py`
- Modify: `tools/test_kv_offload.py`

**Interfaces:**
- Produces: `_blockwise_online_spec_verify_attention(...) -> torch.Tensor`.
- Reuses: blockwise staging, window planning, GQA value weighting, and online-softmax merge helpers.
- Preserves: `_flash_attn_spec_verify()` for non-blockwise execution.

- [x] **Step 1: Write a dense causal oracle**

In `tools/test_native_verifier_attention.py`, add a local oracle:

```python
def _dense_spec_verify_reference(
    q,
    logical_k,
    logical_v,
    context_lens,
    query_len,
    scale,
):
    outputs = []
    for row_index, context_len in enumerate(context_lens):
        row_q = q[
            row_index * query_len:
            (row_index + 1) * query_len
        ].float()
        row_k = logical_k[row_index][:context_len].float()
        row_v = logical_v[row_index][:context_len].float()
        scores = torch.einsum(
            "qhd,khd->qhk",
            row_q,
            _repeat_kv_heads(row_k, row_q.size(1)),
        ) * scale
        query_start = context_len - query_len
        mask = (
            torch.arange(context_len)
            .view(1, 1, -1)
            <= torch.arange(
                query_start,
                context_len,
            ).view(query_len, 1, 1)
        )
        probs = torch.softmax(
            scores.masked_fill(~mask, float("-inf")),
            dim=-1,
        )
        outputs.append(torch.einsum(
            "qhk,khd->qhd",
            probs,
            _repeat_kv_heads(row_v, row_q.size(1)),
        ))
    return torch.cat(outputs).to(q.dtype)
```

- [x] **Step 2: Write failing mathematical parity tests**

Add parametrized tests for:

```text
(batch=1, query_len=2, context_lens=(11,))
(batch=1, query_len=4, context_lens=(19,))
(batch=4, query_len=2, context_lens=(9, 13, 17, 21))
(batch=4, query_len=4, context_lens=(12, 16, 20, 24))
```

Use:

```text
block_size=4
window_blocks=2
gpu_blocks=12
num_heads=4
num_kv_heads=2
head_dim=8
```

Populate logical CPU backing, stage only write blocks, run the helper for even
and odd `layer_idx`, and compare:

```python
torch.testing.assert_close(
    actual.float(),
    expected.float(),
    rtol=2e-4,
    atol=2e-4,
)
```

Assert:

```text
positive H2D copies
more visible logical blocks than initially resident blocks
write blocks remain resident
forward and reverse window orders match the same dense oracle
```

- [x] **Step 3: Run attention RED**

Run:

```bash
python3 -m pytest \
  tools/test_native_verifier_attention.py \
  -k blockwise_spec_verify \
  -q
```

Expected: FAIL because the helper is absent.

- [x] **Step 4: Implement plan identity and causal-mask helpers**

Add a frozen identity including:

```python
block_rows
context_lens
query_lens
block_size
window_blocks
write_blocks
gpu_blocks
```

Add a cached absolute-position mask builder returning
`[B, Q, max_window_tokens]`.

Reject:

```text
empty rows
heterogeneous Q
flattened query mismatch
query_len > context_len
window_blocks > GPU slots
```

- [x] **Step 5: Implement the online-softmax helper**

Use FP32 accumulators with shapes:

```python
running_m = [B, Q, H]
running_l = [B, Q, H]
running_o = [B, Q, H, D]
```

For each existing blockwise decode window:

1. stage required logical blocks;
2. map each window row to physical slots;
3. gather dense K/V;
4. compute `einsum("bqhd,bkhd->bqhk") * scale`;
5. apply the absolute causal mask and per-row window-length mask;
6. merge with the existing stable recurrence;
7. preserve protected verifier write blocks.

Return:

```python
(running_o / running_l.unsqueeze(-1)).to(q.dtype).view_as(q)
```

- [x] **Step 6: Route spec-verify attention**

In `Attention.forward()`:

```python
if context.mode == "spec_verify":
    if self.kv_quant_bits != 0:
        raise RuntimeError(
            "spec_verify requires FP16/BF16 KV"
        )
    if context.kv_offload_blockwise_decode:
        o = _blockwise_online_spec_verify_attention(...)
    else:
        o = _flash_attn_spec_verify(...)
```

Keep the final output reshape identical.

- [x] **Step 7: Run attention GREEN**

Run:

```bash
python3 -m pytest \
  tools/test_native_verifier_attention.py \
  tools/test_kv_offload.py \
  -q
```

Expected: PASS.

- [x] **Step 8: Run focused core regression**

Run:

```bash
python3 -m pytest \
  tools/test_speculative_residency.py \
  tools/test_model_runner_spec_verify.py \
  tools/test_native_verifier_attention.py \
  tools/test_native_verifier_contract.py \
  tools/test_context_modes.py \
  tools/test_kv_offload.py \
  -q
```

Expected: PASS.

---

### Task 4: Add the Loaded 16K/32K Blockwise Correctness Worker

**Files:**
- Create: `tools/blockwise_speculative_verifier_worker.py`
- Create: `tools/blockwise_speculative_verifier_gate.py`
- Create: `tools/test_blockwise_speculative_verifier_gate.py`

**Interfaces:**
- Produces: one isolated worker JSON per `(policy, context, batch)` cell.
- Produces: pure prompt, counter-delta, schema-validation, and aggregation helpers.
- Uses: one warmup run and one recorded correctness run per worker.

- [x] **Step 1: Write failing pure-helper tests**

Define constants:

```python
SCHEMA_VERSION = 1
CLASSIFICATION = "NOT_PROMOTABLE"
POLICIES = ("baseline", "ngram")
CONTEXT_TOKENS = (16384, 32768)
BATCH_SIZES = (1, 4)
MAX_OUTPUT_TOKENS = 8
NGRAM_SIZE = 3
MAX_PROPOSAL_TOKENS = 4
GPU_BLOCKS = 68
LOGICAL_BLOCKS = 640
BLOCKWISE_BLOCKS = 8
```

Test:

```text
prompt builder returns exact requested lengths and stable SHA-256 digests
worker key is policy/context/batch unique
movement subtraction rejects negative/non-integer counters
worker validation requires exact output shape
candidate validation requires proposal, acceptance, callbacks, and H2D
all cells require rejected_d2h_copies=0
artifact validation requires eight cells
classification cannot differ from NOT_PROMOTABLE
```

- [x] **Step 2: Run helper RED**

Run:

```bash
python3 -m pytest \
  tools/test_blockwise_speculative_verifier_gate.py \
  -q
```

Expected: collection FAIL because the gate module is absent.

- [x] **Step 3: Implement pure gate helpers**

Create dependency-light functions:

```python
build_prompt_token_batches(...)
subtract_counter_summaries(...)
validate_worker_result(...)
build_artifact(...)
validate_artifact(...)
sha256_file(...)
atomic_write_json(...)
```

The artifact claim scope must be:

```text
TP1 Qwen3-0.6B 16K/32K blockwise KV-offload correctness,
batch 1/4, baseline versus generic n-gram runtime
```

- [x] **Step 4: Run helper GREEN**

Run:

```bash
python3 -m pytest \
  tools/test_blockwise_speculative_verifier_gate.py \
  -k 'prompt or movement or validate or artifact' \
  -q
```

Expected: PASS.

- [x] **Step 5: Write failing worker lifecycle tests**

Use injected fake engine/tokenizer factories and assert exact configuration:

```python
engine_factory(
    model_path,
    tensor_parallel_size=1,
    enforce_eager=True,
    max_model_len=33024,
    max_num_batched_tokens=132096,
    max_num_seqs=batch_size,
    max_num_prefill_tokens_per_step=1024,
    chunked_prefill_mixed_batch=False,
    kv_offload_mvp0=True,
    kv_offload_gpu_blocks=68,
    kv_offload_logical_blocks=640,
    kv_offload_blockwise_decode=True,
    kv_offload_blockwise_prefill=True,
    kv_offload_blockwise_blocks=8,
)
```

Assert:

```text
baseline does not install a runtime
ngram installs EngineSpeculativeRuntime(NGramDraftAdapter)
temperature=0.0, max_tokens=8, ignore_eos=True
one warmup and one recorded run
engine exits in finally
real before/after counter deltas are recorded
output IDs and prompt digests are recorded
```

- [x] **Step 6: Run worker RED**

Run:

```bash
python3 -m pytest \
  tools/test_blockwise_speculative_verifier_gate.py \
  -k worker \
  -q
```

Expected: FAIL because the worker is absent.

- [x] **Step 7: Implement worker**

The worker:

1. builds exact deterministic prompt rows;
2. loads one engine for one cell;
3. installs n-gram runtime only for candidate;
4. runs one warmup;
5. clears reusable prefix cache;
6. snapshots real MVP-0 counters;
7. runs one synchronized recorded generation;
8. snapshots counters and stores deltas;
9. writes JSON atomically;
10. exits in `finally`.

Do not force extra benchmark-only eviction. Natural blockwise execution must
produce H2D because visible history exceeds 68 slots.

- [x] **Step 8: Run worker GREEN and compile**

Run:

```bash
python3 -m pytest \
  tools/test_blockwise_speculative_verifier_gate.py \
  -k worker \
  -q
python3 -m py_compile \
  tools/blockwise_speculative_verifier_gate.py \
  tools/blockwise_speculative_verifier_worker.py
```

Expected: PASS.

---

### Task 5: Add Parent Orchestration and Independent Verification

**Files:**
- Modify: `tools/blockwise_speculative_verifier_gate.py`
- Create: `tools/verify_blockwise_speculative_verifier_gate.py`
- Modify: `tools/test_blockwise_speculative_verifier_gate.py`

**Interfaces:**
- Produces: eight isolated worker subprocesses.
- Produces: schema-v1 `result.json`.
- Produces: independent source-hash verifier receipt.

- [x] **Step 1: Write failing orchestration tests**

Assert the parent launches exactly:

```text
baseline:16K:b1
baseline:16K:b4
baseline:32K:b1
baseline:32K:b4
ngram:16K:b1
ngram:16K:b4
ngram:32K:b1
ngram:32K:b4
```

Test nonzero worker exit, missing JSON, malformed worker result, parity
mismatch, and source-hash drift.

- [x] **Step 2: Run orchestration RED**

Run:

```bash
python3 -m pytest \
  tools/test_blockwise_speculative_verifier_gate.py \
  -k 'orchestrate or verifier or source_hash' \
  -q
```

Expected: FAIL because orchestration/verifier interfaces are absent.

- [x] **Step 3: Implement parent and verifier**

The parent stores:

```text
environment
campaign matrix
engine configuration
prompts and digests
eight worker payloads
exact parity result per context/batch
candidate proposal/acceptance/callback counts
real movement totals
source hashes
limitations
classification
```

The verifier imports only dependency-light gate helpers, recomputes all
derived fields, and writes:

```json
{
  "schema_version": 1,
  "status": "PASS",
  "classification": "NOT_PROMOTABLE",
  "artifact_sha256": "...",
  "cells": {
    "16384:b1": "PASS",
    "16384:b4": "PASS",
    "32768:b1": "PASS",
    "32768:b4": "PASS"
  }
}
```

- [x] **Step 4: Run orchestration GREEN**

Run:

```bash
python3 -m pytest \
  tools/test_blockwise_speculative_verifier_gate.py \
  -q
python3 -m py_compile \
  tools/blockwise_speculative_verifier_gate.py \
  tools/blockwise_speculative_verifier_worker.py \
  tools/verify_blockwise_speculative_verifier_gate.py
```

Expected: PASS.

---

### Task 6: Add the Fixed Remote Runner

**Files:**
- Create: `tools/run_blockwise_speculative_verifier_gate_remote.sh`
- Modify: `tools/test_blockwise_speculative_verifier_gate.py`

**Interfaces:**
- Produces: tagged local artifact directory with worker logs/JSON, remote log,
  `result.json`, `verify.remote.json`, and `verify.json`.

- [x] **Step 1: Write failing runner-source tests**

Require:

```text
sitian@10.232.195.203
KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian
ControlMaster=no
ControlPath=none
/data00/home/sitian/sitian-workspace01/tllm/env/bin/python
/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0.6B
/data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge
CUDA_VISIBLE_DEVICES=7
remote py_compile preflight
unconditional artifact download
remote independent verification
local independent verification
```

- [x] **Step 2: Run runner RED**

Run:

```bash
python3 -m pytest \
  tools/test_blockwise_speculative_verifier_gate.py \
  -k remote_runner \
  -q
```

Expected: FAIL because the script is absent.

- [x] **Step 3: Implement runner**

Synchronize:

```text
tinyvllm/
blockwise gate
worker
verifier
test file
```

Use one remote tagged output directory. Wrap only the campaign in `set +e`,
always download partial state, print `remote.log`, and propagate the remote
status.

- [x] **Step 4: Run runner GREEN**

Run:

```bash
python3 -m pytest \
  tools/test_blockwise_speculative_verifier_gate.py \
  -k remote_runner \
  -q
bash -n \
  tools/run_blockwise_speculative_verifier_gate_remote.sh
```

Expected: PASS.

---

### Task 7: Run the Loaded 16K/32K Gate

**Files:**
- Produce: `artifacts/blockwise_speculative_verifier/${RUN_TAG}/result.json`
- Produce: `artifacts/blockwise_speculative_verifier/${RUN_TAG}/verify.remote.json`
- Produce: `artifacts/blockwise_speculative_verifier/${RUN_TAG}/verify.json`

**Interfaces:**
- Consumes: completed core and remote runner.
- Produces: authoritative correctness/movement evidence, not performance evidence.

- [x] **Step 1: Run full local dependency-light regression**

Run:

```bash
python3 -m pytest \
  tools/test_speculative_residency.py \
  tools/test_model_runner_spec_verify.py \
  tools/test_native_verifier_attention.py \
  tools/test_native_verifier_contract.py \
  tools/test_context_modes.py \
  tools/test_kv_offload.py \
  tools/test_blockwise_speculative_verifier_gate.py \
  -q
```

Expected: PASS.

- [x] **Step 2: Run remote campaign**

Run:

```bash
RUN_TAG="blockwise-tp1-opaque-17786-19070"
LOCAL_OUT="artifacts/blockwise_speculative_verifier/${RUN_TAG}"
KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian \
CUDA_VISIBLE_DEVICES=7 \
RUN_TAG="${RUN_TAG}" LOCAL_OUT="${LOCAL_OUT}" \
  bash tools/run_blockwise_speculative_verifier_gate_remote.sh
```

Wait for all eight workers and both verification receipts.

- [x] **Step 3: Require authoritative conditions**

Require:

```text
status=PASS
classification=NOT_PROMOTABLE
all eight worker cells present
exact baseline/candidate token parity for all four context/batch pairs
candidate proposed_tokens > 0 in every pair
candidate accepted_draft_tokens > 0 in every pair
candidate first_target_callbacks > 0
candidate tail_callbacks > 0
positive real h2d_copies and h2d_bytes in every cell whose visible logical
history exceeds the 68-slot GPU staging budget
speculative_residency_rejected_d2h_copies=0
visible logical blocks > 68 for 16K:b4, 32K:b1, and 32K:b4
remote verifier PASS
local verifier PASS
```

- [x] **Step 4: Run remote direct KV regression**

Run:

```bash
KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian ssh \
  -o ControlMaster=no \
  -o ControlPath=none \
  -o BatchMode=yes \
  sitian@10.232.195.203 \
  "cd /data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge && \
   CUDA_VISIBLE_DEVICES=7 \
   PYTHONPATH=/data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge \
   /data00/home/sitian/sitian-workspace01/tllm/env/bin/python \
   tools/test_kv_offload.py"
```

Expected: `kv offload tests passed`.

---

### Task 8: Record Evidence and Prepare the Performance Expansion

**Files:**
- Modify: `docs/superpowers/specs/2026-08-12-blockwise-speculative-verifier-kv-offload-design.md`
- Modify: `docs/superpowers/audits/2026-08-12-phase1-objective-coverage.md`
- Modify: `AGENT_HANDOFF_STATE.md`

**Interfaces:**
- Consumes: authoritative artifact and validation receipts.
- Produces: exact continuation boundary for the 16K/32K performance campaign.

- [x] **Step 1: Update design execution record**

Record:

```text
artifact path and SHA-256
source HEAD and dirty-tree limitation
all four context/batch parity outcomes
proposal and acceptance totals
real H2D/D2H/copy-wait/eviction totals
committed/rejected residency counters
remote/local verifier status
```

- [x] **Step 2: Update audit and handoff**

State what is established:

```text
fixed-budget blockwise speculative verification at TP1
16K/32K batch 1/4 exact greedy parity
real movement with visible logical history larger than GPU slots
transactional accepted/rejected residency behavior
```

State what remains unestablished:

```text
16K/32K performance direction
TP4
second model
learned drafter/MTP
KV8/KV4 speculative verification
promotion
```

- [ ] **Step 3: Run final verification**

Run:

```bash
python3 tools/verify_blockwise_speculative_verifier_gate.py \
  "artifacts/blockwise_speculative_verifier/${RUN_TAG}/result.json" \
  . \
  --output \
  "artifacts/blockwise_speculative_verifier/${RUN_TAG}/verify.json"

python3 -m pytest \
  tools/test_speculative_residency.py \
  tools/test_model_runner_spec_verify.py \
  tools/test_native_verifier_attention.py \
  tools/test_native_verifier_contract.py \
  tools/test_context_modes.py \
  tools/test_kv_offload.py \
  tools/test_blockwise_speculative_verifier_gate.py \
  -q

python3 -m py_compile \
  tinyvllm/engine/speculative_residency.py \
  tinyvllm/engine/model_runner.py \
  tinyvllm/layers/attention.py \
  tinyvllm/utils/context.py \
  tools/blockwise_speculative_verifier_gate.py \
  tools/blockwise_speculative_verifier_worker.py \
  tools/verify_blockwise_speculative_verifier_gate.py

bash -n \
  tools/run_blockwise_speculative_verifier_gate_remote.sh

git diff --check
git diff --cached --quiet
```

Expected: all tests and static checks PASS, both independent verification
receipts remain PASS, and staged diff remains empty.

Execution record on 2026-08-13:

```text
strict downloaded-artifact verifier:       PASS
dependency-light tests, isolated:           124 passed
remote direct tools/test_kv_offload.py:     PASS
Python compilation:                         PASS
remote-wrapper shell syntax:                PASS
runner lifecycle repetition:                10 / 10 passed
authoritative artifact conditions:          PASS
documentation consistency:                  PASS
staged diff:                                empty
scoped diff check for this gate:             PASS
repo-global git diff --check:                BLOCKED
```

The repo-global diff check reports pre-existing trailing whitespace in
`tinyvllm/engine/model_runner.py` around the unrelated `warmup_model()` and
another blank line. Those lines are outside this blockwise gate's edit scope
and were not modified as cleanup. Keep this final checkbox open until the
owner of that concurrent code either removes the whitespace or explicitly
authorizes a global hygiene edit.

### 2026-08-15 Fresh Final-Verification Reconciliation

The historical repo-global whitespace blocker is now gone:

```text
git diff --check:         PASS
git diff --cached --quiet: PASS
```

The retained authority directory is:

```text
artifacts/blockwise_speculative_verifier/
  blockwise-tp1-opaque-17786-19070/
```

Fresh payload and receipt checks establish:

```text
result payload validation:                    PASS
required cells:                               16384:b1, 16384:b4,
                                              32768:b1, 32768:b4
artifact SHA-256:                             2ecde42b605f6147a36a424ba12f71cff8a5714ef858bd0a28011706ea1dc600
historical local/remote receipts equal:       PASS
historical local/remote receipt status:       PASS / NOT_PROMOTABLE
source inventory:                             7 files
```

Fresh current-checkout independent verification fails closed at:

```text
ValueError: source hash drift: tinyvllm/engine/model_runner.py
```

The artifact retains per-file source hashes but no `source.tar` or other
frozen source archive. Later runtime work changed `model_runner.py`, so the
original source-bound verifier cannot be rerun from the current checkout and
the exact historical source cannot be reconstructed from this authority
directory alone.

The six dependency-light files were rerun in isolated pytest processes to
avoid their deliberate `sys.modules` stubs contaminating later collection:

```text
tools/test_speculative_residency.py:                 12 passed
tools/test_model_runner_spec_verify.py:             104 passed
tools/test_native_verifier_attention.py:              4 passed
tools/test_native_verifier_contract.py:               6 passed
tools/test_context_modes.py:                          9 passed
tools/test_blockwise_speculative_verifier_gate.py:    12 passed
total:                                               147 passed
```

The exact combined pytest command first exposed test-module stub pollution,
and isolated `tools/test_kv_offload.py` then failed collection on the actual
host dependency boundary:

```text
ModuleNotFoundError: No module named 'flash_attn'
```

No fake CUDA or FlashAttention module was installed. Static verification is
fresh:

```text
seven-file py_compile:                 PASS
remote-runner bash syntax:             PASS
repo-global git diff --check:          PASS
staged diff:                           empty
```

Task 8 Step 3 remains unchecked. Closing it requires either:

1. the exact historical seven-file source snapshot so the retained artifact
   verifier can be rerun, or a newly authorized source-archived authority
   campaign; and
2. a real environment containing the production `flash_attn` dependency for
   `tools/test_kv_offload.py`.

The retained artifact remains useful historical evidence within its frozen
scope, but it is not upgraded into fresh current-source authority.

### 2026-08-15 Frozen-Source Recovery Addendum

The exact historical seven-file source set has now been recovered from the
local AI contribution content-addressed snapshot store, without modifying the
current checkout. The previously missing file was:

```text
snapshot blob:
  /Users/bytedance/.trae/hooks/ai-contribution-sdk/snapshot-blobs/
    037bf4f4a6aff6e19b19493e8fb6b316abdf827a74564b76291089ae83d12f42

recovered path:
  tinyvllm/engine/model_runner.py

size:
  278700 bytes

SHA-256:
  037bf4f4a6aff6e19b19493e8fb6b316abdf827a74564b76291089ae83d12f42
```

All seven artifact source hashes were present as exact content-addressed
blobs. They were reconstructed under:

```text
/tmp/blockwise-speculative-verifier-frozen-2026-08-15
```

Fresh verification from the recovered historical verifier and gate produced:

```text
status:                  PASS
classification:          NOT_PROMOTABLE
artifact SHA-256:        2ecde42b605f6147a36a424ba12f71cff8a5714ef858bd0a28011706ea1dc600
16384:b1:                PASS
16384:b4:                PASS
32768:b1:                PASS
32768:b4:                PASS
fresh == verify.json:    true
fresh == verify.remote.json: true
```

This closes the missing historical-source reconstruction subproblem. It does
not make the authority directory self-contained: the retained artifact still
has no `source.tar`, and replay currently depends on an external local hook
blob store.

Task 8 Step 3 remains unchecked for two independent reasons:

1. the prescribed verifier against the current checkout still fails closed
   because current `tinyvllm/engine/model_runner.py` is not the historical
   artifact source; and
2. production `tools/test_kv_offload.py` still has no successful local
   collection: current `/usr/bin/python3` lacks `torch`, while the earlier
   Torch-enabled entry point reached the next missing dependency,
   `flash_attn`.

No GPU, remote, NCCL, loaded-checkpoint, performance, or runtime-path
observation was run. The updated boundary is:

```text
BLOCKWISE_TP1_FROZEN_SOURCE_RECOVERY=ESTABLISHED_LOCAL_EXTERNAL_SNAPSHOT
BLOCKWISE_TP1_FROZEN_SOURCE_REPLAY=PASS
BLOCKWISE_TP1_FROZEN_RECEIPT_EQUAL_HISTORICAL_LOCAL_REMOTE=PASS
BLOCKWISE_TP1_FRESH_CURRENT_SOURCE_VERIFICATION=BLOCKED_SOURCE_DRIFT
BLOCKWISE_CURRENT_SYSTEM_PYTHON_KV_OFFLOAD_TEST=NOT_COLLECTED_MISSING_TORCH
BLOCKWISE_TORCH_ENABLED_KV_OFFLOAD_TEST=NOT_COLLECTED_MISSING_FLASH_ATTN
PROMOTION=NOT_PROMOTABLE
```
