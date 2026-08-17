# Qwen3.5 TP4 Real Root-Logit Correctness Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove Qwen3.5 TP4 real-checkpoint one-shot final-token decision preservation with four real NCCL ranks, rank-0 full-vocabulary logits, all-rank state/cleanup evidence, and an independent verifier.

**Architecture:** Reuse the frozen TP1 prompt/comparison contract and authorized checkpoint stack. Run the official model in one isolated process, then run four concurrent rank-local native model roots on four GPUs with real production collectives; publish rank-0 logits plus all-rank evidence only after every process exits and cleanup succeeds.

**Tech Stack:** Python 3.11, PyTorch BF16/FP32, `torch.distributed` NCCL, TinyLLMForge Qwen3.5 packed root and checkpoint loader, pytest, JSON/SHA256 artifact binding, SSH/rsync remote execution.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Do not stage, commit, merge, create a branch, or create a PR.
- Use only `sitian@10.232.195.203` for remote execution.
- Use `KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian` and SSH ControlMaster `/tmp/ssh-sitian-10.232.195.203`.
- Use remote Python `/data00/home/sitian/sitian-workspace01/tllm/env/bin/python`.
- Preserve every failed or superseded run under a new unique tag.
- Do not weaken or rewrite the immutable schema-v2 canonical `NO_GO`.
- Do not claim latency, throughput, cache, GPU-memory, compression, or quality improvement from this correctness gate.
- Native execution must not construct `LLMEngine`, `ModelRunner`, Scheduler, sampler, tokenizer, or generation.
- Official reference must exit before the four-rank native group starts.
- Native execution requires four distinct GPUs, each with at least `24 * 1024**3` free bytes.
- Comparison remains `bf16_decision_preserving` with `atol=2e-5` and `rtol=1e-5`.

---

### Task 1: Distributed-Aware Root Logit Contract

**Files:**
- Modify: `tinyvllm/models/qwen35_packed.py`
- Modify: `tools/test_qwen35_transactional_root_causal_lm.py`

**Interfaces:**
- Consumes: initialized or uninitialized `torch.distributed` state and the existing `ParallelLMHead` root/non-root semantics.
- Produces: `_validate_logits(logits, hidden_states)` that accepts `None` only on non-root ranks of an initialized multi-rank process group.

- [ ] **Step 1: Write failing TP4 root tests**

Add tests with a narrow fake distributed state:

```python
def test_tp4_non_root_accepts_none_and_commits(monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 4)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)
    root, stack = make_root(lm_head_output=None)
    _, logits = root.run_step(...)
    assert logits is None
    assert stack.commit_calls == 1


def test_tp4_rank0_rejects_none_without_commit(monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 4)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    root, stack = make_root(lm_head_output=None)
    with pytest.raises(ValueError, match="rank zero"):
        root.run_step(...)
    assert stack.commit_calls == 0
```

Also add:

- TP4 non-root rejects a tensor;
- TP1/uninitialized execution still rejects `None`;
- initialized world-size-one execution still requires a tensor.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
python -m pytest -q tools/test_qwen35_transactional_root_causal_lm.py
```

Expected: new non-root `None` success test fails because the current root requires a tensor.

- [ ] **Step 3: Implement the minimal distributed-aware contract**

Use a helper that fails closed if distributed metadata is inconsistent:

```python
@staticmethod
def _distributed_output_role() -> tuple[int, int] | None:
    if not torch.distributed.is_available():
        return None
    if not torch.distributed.is_initialized():
        return None
    world_size = int(torch.distributed.get_world_size())
    rank = int(torch.distributed.get_rank())
    if world_size <= 0 or rank < 0 or rank >= world_size:
        raise ValueError("distributed output role is invalid")
    return rank, world_size
```

Then enforce:

```python
role = self._distributed_output_role()
if role is not None and role[1] > 1 and role[0] != 0:
    if logits is not None:
        raise ValueError("non-root lm_head output must be None")
    return
if logits is None:
    raise ValueError("rank zero lm_head output must be a tensor")
```

Keep all existing tensor shape, dtype, row-count, vocabulary, and device checks unchanged.

- [ ] **Step 4: Run root and TP1 regression tests**

Run:

```bash
python -m pytest -q \
  tools/test_qwen35_transactional_root_causal_lm.py \
  tools/test_qwen35_tp1_real_root_logit_correctness_preflight.py
```

Expected: all tests pass.

### Task 2: TP4 Contract and Correctness-Only Attention Backend

**Files:**
- Create: `tools/qwen35_tp4_real_root_logit_correctness_contract.py`
- Create: `tools/test_qwen35_tp4_real_root_logit_correctness_contract.py`
- Create: `tools/qwen35_tp4_real_root_logit_correctness_preflight.py`
- Create: `tools/test_qwen35_tp4_real_root_logit_correctness_preflight.py`

**Interfaces:**
- Consumes: `prompt_cases`, `compare_logits`, `classify_rows`, and BF16 tolerance from `qwen35_tp1_real_root_logit_correctness_contract.py`.
- Produces: TP4 artifact constants, rank-output validators, topology validators, `Qwen35TP4CausalAttentionBackend`, and candidate construction helpers.

- [ ] **Step 1: Write failing contract tests**

Cover these exact guards:

```python
def test_rank_output_contract_accepts_only_root_tensor():
    row = torch.zeros(248320, dtype=torch.float32)
    validate_rank_logits(rank=0, world_size=4, logits=row)
    for rank in (1, 2, 3):
        validate_rank_logits(rank=rank, world_size=4, logits=None)


def test_rank_output_contract_rejects_non_root_tensor():
    with pytest.raises(ValueError, match="non-root"):
        validate_rank_logits(
            rank=2,
            world_size=4,
            logits=torch.zeros(248320),
        )
```

Also reject:

- world size other than four;
- duplicate GPU indices or UUIDs;
- local topology other than 2 query heads and 1 KV head;
- rank-0 vocabulary width other than 248320;
- relaxed comparison tolerance.

- [ ] **Step 2: Run contract tests and verify RED**

Run:

```bash
python -m pytest -q tools/test_qwen35_tp4_real_root_logit_correctness_contract.py
```

Expected: import failure because the module does not exist.

- [ ] **Step 3: Implement the frozen TP4 contract**

The module must import the TP1 contract by sibling file path, then expose:

```python
SCHEMA_VERSION = "qwen35.tp4-real-root-logit-correctness.v1"
WORLD_SIZE = 4
MODEL_VOCAB_SIZE = 248320
LOCAL_QUERY_HEADS = 2
LOCAL_KV_HEADS = 1
MIN_GPU_FREE_BYTES = 24 * 1024**3

def validate_rank_logits(*, rank: int, world_size: int, logits):
    ...

def validate_rank_topology(row: Mapping[str, object]) -> dict[str, object]:
    ...

def validate_gpu_assignments(rows: Iterable[Mapping[str, object]]) -> tuple[dict, ...]:
    ...
```

Re-export the exact TP1 prompt cases, tolerance, comparison, and classification behavior without copying token arrays.

- [ ] **Step 4: Write failing backend tests**

Use a two-head manual oracle:

```python
backend = Qwen35TP4CausalAttentionBackend(
    local_query_heads=2,
    local_kv_heads=1,
    head_dim=4,
)
actual = backend(query_bf16, key_bf16, value_bf16)
expected = manual_fp32_causal_attention(...)
torch.testing.assert_close(actual.float(), expected.float(), atol=2e-2, rtol=2e-2)
```

Also test future-token poisoning, BF16 preservation, malformed widths, and unsupported head replication.

- [ ] **Step 5: Run backend tests and verify RED**

Run:

```bash
python -m pytest -q \
  tools/test_qwen35_tp4_real_root_logit_correctness_preflight.py \
  -k "attention"
```

Expected: missing backend class.

- [ ] **Step 6: Implement the minimal TP4 backend**

Reshape to local heads, repeat KV from one to two heads, perform FP32 causal attention, and cast back:

```python
scores = torch.einsum("thd,shd->hts", query.float(), key.float())
scores = scores * (head_dim ** -0.5)
scores = scores.masked_fill(~causal_mask, float("-inf"))
probabilities = torch.softmax(scores, dim=-1)
output = torch.einsum("hts,shd->thd", probabilities, value.float())
return output.to(query.dtype).reshape(token_count, -1)
```

- [ ] **Step 7: Run contract/backend tests**

Run:

```bash
python -m pytest -q \
  tools/test_qwen35_tp4_real_root_logit_correctness_contract.py \
  tools/test_qwen35_tp4_real_root_logit_correctness_preflight.py \
  -k "contract or topology or attention"
```

Expected: all selected tests pass.

### Task 3: Rank-Local Real Candidate and Native Case Execution

**Files:**
- Modify: `tools/qwen35_tp4_real_root_logit_correctness_preflight.py`
- Modify: `tools/test_qwen35_tp4_real_root_logit_correctness_preflight.py`

**Interfaces:**
- Consumes: approved checkpoint constants and authorized loader stack proven by TP1/TP4 prerequisite gates.
- Produces: `build_real_tp4_cpu_candidate(rank)`, `move_loaded_candidate_to_device(...)`, and `run_tp4_native_cases(...)`.

- [ ] **Step 1: Write failing candidate-construction tests**

Inject fake dependencies and assert:

```python
candidate = build_real_tp4_cpu_candidate(
    rank=2,
    dependencies=fakes,
)
assert fakes.layout_call["tensor_parallel_size"] == 4
assert fakes.target_call["tensor_parallel_size"] == 4
assert fakes.target_call["tensor_parallel_rank"] == 2
assert fakes.target_call["build_attention_backend"] is not None
```

Reject rank `-1`, rank `4`, wrong pool identity, repeated target-provider calls, and payload/model fingerprint mismatch.

- [ ] **Step 2: Run candidate tests and verify RED**

Run:

```bash
python -m pytest -q \
  tools/test_qwen35_tp4_real_root_logit_correctness_preflight.py \
  -k "candidate"
```

Expected: missing candidate builder.

- [ ] **Step 3: Implement rank-local candidate construction**

Generalize the TP1 flow with:

```python
layout = dependencies.build_layout(
    metadata.hf_config,
    tensor_parallel_size=4,
    dtype=torch.bfloat16,
    recurrent_dtype=torch.float32,
    speculative_tokens=1,
)
target = dependencies.prepare_target(
    metadata.hf_config,
    tensor_plan,
    pool=pool,
    tensor_parallel_size=4,
    tensor_parallel_rank=rank,
    build_attention_backend=_build_tp4_attention_backend,
    parameter_device="cpu",
)
```

Reuse the approved authorization, tensor-byte ceiling, transactional loader, and device migration invariants.

- [ ] **Step 4: Write failing native-case tests**

Use fake roots for ranks 0-3:

- rank 0 returns one full-vocabulary row;
- ranks 1-3 return `None`;
- all ranks mutate 36 state components;
- all ranks release and zero their pool;
- commit occurs once per case;
- a non-root tensor or root `None` fails.

- [ ] **Step 5: Implement `run_tp4_native_cases`**

For each case:

```python
set_context(
    is_prefill=True,
    mode="prefill",
    cu_seqlens_q=cumulative,
    cu_seqlens_k=cumulative.clone(),
    max_seqlen_q=token_count,
    max_seqlen_k=token_count,
    logits_indices=torch.tensor([token_count - 1], device=device),
)
normalized, logits = model.run_step(...)
validate_rank_logits(rank=rank, world_size=4, logits=logits)
```

Only rank 0 moves logits to contiguous CPU FP32. Every rank records normalized output, state mutation, lease generation, release, and binding cleanup.

- [ ] **Step 6: Run candidate/native tests**

Run:

```bash
python -m pytest -q tools/test_qwen35_tp4_real_root_logit_correctness_preflight.py \
  -k "candidate or native_case or state"
```

Expected: all selected tests pass.

### Task 4: Distributed Worker and Source-Bound Coordinator

**Files:**
- Modify: `tools/qwen35_tp4_real_root_logit_correctness_preflight.py`
- Modify: `tools/test_qwen35_tp4_real_root_logit_correctness_preflight.py`

**Interfaces:**
- Consumes: reference worker behavior from the TP1 preflight, TP4 candidate/native-case functions, four selected GPU rows, and two fresh ports.
- Produces: CLI modes `run`, `validate`, `internal-reference`, and `internal-native-rank`; exact-five artifact inputs.

- [ ] **Step 1: Write failing process-group lifecycle tests**

With fake `torch.distributed`, assert one worker:

```python
execute_native_rank_worker(rank=3, world_size=4, ...)
assert calls == [
    ("set_device", 3),
    ("init_process_group", "nccl", 4, 3, rendezvous),
    ("barrier", "case:p17"),
    ("barrier", "case:p65"),
    ("barrier", "case:synthetic"),
    ("barrier", "final"),
    ("destroy_process_group",),
]
```

Failure tests must prove destruction occurs after candidate, case, or barrier exceptions and that success-only final barrier is skipped after a local failure.

- [ ] **Step 2: Implement the rank worker**

The worker must:

- validate rank/world-size/environment;
- verify its GPU index and UUID;
- initialize real NCCL;
- install delegate-only collective counters;
- build/load/migrate its candidate;
- execute all cases;
- perform per-case and final barriers;
- destroy the group in `finally`;
- atomically write one rank row and, for rank 0, one logits tensor map.

- [ ] **Step 3: Write failing coordinator tests**

Use fake process objects to assert:

- reference exits before any native rank starts;
- all four native ranks are started before any join;
- `CUDA_VISIBLE_DEVICES` lists exactly four unique GPUs in rank order;
- all ranks share the rendezvous port and process-group nonce;
- `TINYVLLM_DIST_PORT != MASTER_PORT`;
- one rank timeout or early exit prevents finalization;
- all native PIDs must disappear before publication.

- [ ] **Step 4: Implement GPU selection and native launch**

Add:

```python
def select_tp4_gpu_resources(rows, *, minimum_free_bytes) -> tuple[dict, ...]:
    ...

def fresh_port_pair() -> tuple[int, int]:
    ...

def launch_native_rank_group(...):
    ...
```

Use one `CUDA_VISIBLE_DEVICES` list for the group and logical local ranks
`0..3`. Recheck physical indices, UUIDs, free memory, and active compute
processes after the reference PID is absent.

- [ ] **Step 5: Implement rank evidence aggregation**

Require:

```python
set(rank_rows) == {0, 1, 2, 3}
len({row["pid"] for row in rank_rows.values()}) == 4
len({row["gpu_uuid"] for row in rank_rows.values()}) == 4
all(row["world_size"] == 4 for row in rank_rows.values())
all(row["process_group_destroyed"] for row in rank_rows.values())
```

Validate rank-0 tensor map and rank1-3 `None` evidence before finalization.

- [ ] **Step 6: Run coordinator tests**

Run:

```bash
python -m pytest -q tools/test_qwen35_tp4_real_root_logit_correctness_preflight.py \
  -k "process_group or coordinator or gpu or port or launch or cleanup"
```

Expected: all selected tests pass.

### Task 5: Exact-Five Artifact Finalizer

**Files:**
- Modify: `tools/qwen35_tp4_real_root_logit_correctness_preflight.py`
- Modify: `tools/test_qwen35_tp4_real_root_logit_correctness_preflight.py`

**Interfaces:**
- Consumes: official tensor map/process row, rank-0 tensor map, four rank rows, source manifest, checkpoint identity, and forbidden counters.
- Produces: the exact five authoritative artifact files.

- [ ] **Step 1: Write failing finalizer tests**

Cover:

- successful exact-five publication;
- non-empty run directory rejection;
- missing or duplicate rank rejection;
- non-root tensor evidence rejection;
- incomplete process-group cleanup rejection;
- forbidden counter rejection;
- `.partial` cleanup after injected write failure;
- no authoritative output for non-PASS/incomplete input.

- [ ] **Step 2: Implement atomic finalization**

Build the result with:

```python
comparisons = [
    {
        "case_id": case_id,
        **contract.compare_logits(
            native_rank0[case_id],
            reference[case_id],
            tolerance=contract.BF16_DECISION_TOLERANCE,
        ),
    }
    for case_id in case_ids
]
classification = contract.classify_rows(comparisons)
```

Write `.partial` files, hash them into the source manifest, then `os.replace`
in the frozen five-file order. Remove partial and already-published files on
failure.

- [ ] **Step 3: Run finalizer tests**

Run:

```bash
python -m pytest -q tools/test_qwen35_tp4_real_root_logit_correctness_preflight.py \
  -k "artifact or finalizer or publication"
```

Expected: all selected tests pass.

### Task 6: Independent TP4 Verifier

**Files:**
- Create: `tools/verify_qwen35_tp4_real_root_logit_correctness_gate.py`
- Create: `tools/test_verify_qwen35_tp4_real_root_logit_correctness_gate.py`

**Interfaces:**
- Consumes: exact-five run directory and current source root.
- Produces: `verify_run(run_dir, source_root=...) -> {"checks": int, ...}` and CLI output `PASS, <N> checks`.

- [ ] **Step 1: Write a minimal valid fixture and tamper tests**

Generate a small valid fixture with four rank rows and short finite tensor
rows, then cover:

- baseline PASS;
- changed rank/GPU/UUID/nonce/world-size failure;
- rank2 tensor-output claim failure;
- missing gather/collective evidence failure;
- state or cleanup failure;
- tensor replacement or metric re-signing failure;
- source hash drift;
- extra-file failure.

- [ ] **Step 2: Run verifier tests and verify RED**

Run:

```bash
python -m pytest -q tools/test_verify_qwen35_tp4_real_root_logit_correctness_gate.py
```

Expected: import failure because the verifier does not exist.

- [ ] **Step 3: Implement the independent verifier**

The verifier must use only standard library and PyTorch. It must not import
TinyLLMForge or the producer. Reimplement:

- canonical JSON and file SHA256;
- prompt token SHA256 checks;
- CPU FP32 tensor-map validation;
- top-k and comparison metrics;
- BF16 decision preservation;
- source-tree verification;
- four-rank topology, collective, state, cleanup, and PID checks;
- exact-five inventory.

- [ ] **Step 4: Run verifier tests**

Run:

```bash
python -m pytest -q tools/test_verify_qwen35_tp4_real_root_logit_correctness_gate.py
```

Expected: all tests pass.

### Task 7: Local Completion Matrix

**Files:**
- Modify: `docs/superpowers/specs/2026-07-28-qwen35-tp4-real-root-logit-correctness-gate-design.md`
- Modify: `docs/superpowers/plans/2026-07-28-qwen35-tp4-real-root-logit-correctness-gate.md`

**Interfaces:**
- Consumes: all new and adjacent tests.
- Produces: current-source validation evidence before remote execution.

- [ ] **Step 1: Run focused tests**

Run:

```bash
python -m pytest -q \
  tools/test_qwen35_transactional_root_causal_lm.py \
  tools/test_qwen35_tp1_real_root_logit_correctness_contract.py \
  tools/test_qwen35_tp1_real_root_logit_correctness_preflight.py \
  tools/test_qwen35_tp4_real_root_logit_correctness_contract.py \
  tools/test_qwen35_tp4_real_root_logit_correctness_preflight.py \
  tools/test_verify_qwen35_tp4_real_root_logit_correctness_gate.py \
  tools/test_qwen35_tp4_live_concurrent_candidate_ownership_preflight.py \
  tools/test_qwen35_tp4_real_candidate_provenance_replay_preflight.py \
  tools/test_qwen35_concrete_component_factory.py \
  tools/test_qwen35_kv_head_parallel_linear.py
```

Expected: all tests pass.

- [ ] **Step 2: Run syntax and diff checks**

Run:

```bash
python -m py_compile \
  tinyvllm/models/qwen35_packed.py \
  tools/qwen35_tp4_real_root_logit_correctness_contract.py \
  tools/qwen35_tp4_real_root_logit_correctness_preflight.py \
  tools/verify_qwen35_tp4_real_root_logit_correctness_gate.py
git diff --check -- \
  tinyvllm/models/qwen35_packed.py \
  tools/test_qwen35_transactional_root_causal_lm.py \
  tools/qwen35_tp4_real_root_logit_correctness_contract.py \
  tools/test_qwen35_tp4_real_root_logit_correctness_contract.py \
  tools/qwen35_tp4_real_root_logit_correctness_preflight.py \
  tools/test_qwen35_tp4_real_root_logit_correctness_preflight.py \
  tools/verify_qwen35_tp4_real_root_logit_correctness_gate.py \
  tools/test_verify_qwen35_tp4_real_root_logit_correctness_gate.py
```

Expected: both commands exit zero.

### Task 8: Remote Smoke and Authoritative Run

**Files:**
- Modify: `docs/superpowers/specs/2026-07-28-qwen35-tp4-real-root-logit-correctness-gate-design.md`
- Modify: `AGENT_HANDOFF_STATE.md`
- Create: `experiments/qwen35_hybrid_state/<unique-tp4-tag>/`

**Interfaces:**
- Consumes: source-bound CLI, remote checkpoint, four available GPUs, and the independent verifier.
- Produces: preserved remote smoke evidence, one immutable authoritative exact-five artifact, updated spec, and handoff.

- [ ] **Step 1: Verify SSH and GPU resources**

Run:

```bash
KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian \
ssh -S /tmp/ssh-sitian-10.232.195.203 \
  -o BatchMode=yes -o ConnectTimeout=10 \
  sitian@10.232.195.203 \
  'nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.free --format=csv,noheader,nounits'
```

Expected: at least four distinct GPUs satisfy the 24-GiB free-memory floor.

- [ ] **Step 2: Sync the exact source closure**

Use the coordinator's source manifest and `rsync --relative`; do not sync the
whole dirty worktree or flatten nested paths.

- [ ] **Step 3: Run one native-only distributed smoke**

Use a new tag:

```text
qwen35-tp4-native-smoke-YYYYMMDD-HHMMSS
```

Expected:

- four native ranks initialize and exit;
- rank0 produces three `[248320]` FP32 rows;
- ranks1-3 report exact `None`;
- all ranks report 18 layers / 36 changed components per case;
- all pools zero and all process groups destroy.

Preserve the run whether it passes or fails.

- [ ] **Step 4: Run one full source-bound authority**

Use a second new tag:

```text
qwen35-tp4-authority-YYYYMMDD-HHMMSS
```

Run reference first, then the native group, then exact-five finalization.

- [ ] **Step 5: Download and independently verify**

Copy the exact-five run directory to:

```text
experiments/qwen35_hybrid_state/<authority-tag>/
```

Run:

```bash
python tools/verify_qwen35_tp4_real_root_logit_correctness_gate.py \
  experiments/qwen35_hybrid_state/<authority-tag> \
  --source-root .
```

Expected: `PASS, <N> checks`.

- [ ] **Step 6: Complete the prompt-to-artifact audit**

Append to the design spec:

- run tag and remote/local paths;
- source tree and artifact SHA256 values;
- reference and four native PIDs;
- four GPU indices and UUIDs;
- per-case winner, margins, max absolute difference, cosine, and allclose
  violations;
- per-rank topology, state, collective, cleanup, and process-group evidence;
- exact proven and unproven conclusions.

Update `AGENT_HANDOFF_STATE.md` with the next boundary:

```text
full-attention cache contract and cached continuation correctness
```

Do not claim performance or cache improvement.

