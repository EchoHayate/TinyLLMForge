# Qwen3.5 MTP Eager/Reference Parity Implementation Plan

> **Execution mode:** Inline execution only. The active workspace constraints
> forbid subagents, staging, commits, branch changes, and new worktrees.

**Goal:** Install an independent real-checkpoint eager/reference probe for
Qwen3.5 native MTP across exact Q `(1,2,3,4)` and batch `(1,4)`.

**Architecture:** Execute the same loaded checkpoint twice with fresh
sequences and identical deterministic inputs. The production side uses the
existing Qwen3.5 eager attention helpers; the reference side temporarily
replaces those helper entry points with independent PyTorch SDPA equations,
then compares every proposal-step logits tensor and greedy argmax.

**Tech Stack:** Python, PyTorch, pytest, existing Qwen3.5 MTP executor and
physical proposal-KV store.

## Global Constraints

- TP1 only.
- KV offload disabled.
- One Qwen3.5 MTP layer.
- Greedy proposals only.
- Shared target embedding and LM head.
- Exact Q values `(1,2,3,4)`; no padding, rounding, or family merging.
- Batch sizes `(1,4)`.
- Generic Engine, Scheduler, verifier, residency, and target-KV code remains
  source-neutral.
- CUDA Graph behavior is not changed.
- Overall classification remains `FAIL / NOT_PROMOTABLE`.

---

### Task 1: Independent Reference Probe Contract

**Files:**
- Create: `tools/test_qwen35_mtp_real_eager_reference_probe.py`
- Modify: `tools/test_qwen35_mtp_real_checkpoint_gate.py`

**Interfaces:**
- Produces:
  - `_build_real_eager_reference_probe(...)`
  - `probe(q, batch_size) -> Mapping[str, object]`
- Preserves:
  - `Qwen35MTPProposalExecutor` public runtime interface;
  - physical transaction commit/rollback behavior.

- [x] **Step 1: Write the failing algorithm tests**

Build a deterministic fake MTP module that calls the production
`qwen35_prefill_eager_attention()` and
`qwen35_cached_decode_eager_attention()` entry points. Parameterize:

```python
@pytest.mark.parametrize(
    ("q", "batch_size"),
    ((1, 1), (4, 1), (2, 4), (4, 4)),
)
```

Require:

```python
result = probe(q, batch_size)
assert result["argmax_equal"] is True
assert result["max_abs_diff"] >= 0.0
assert math.isfinite(result["max_abs_diff"])
assert all(
    store.is_allocated(slot_id) is False
    for slot_id in range(store.capacity)
)
```

Also corrupt the reference decode output in one test and require
`argmax_equal=False`, proving the probe does not compare a path with itself.

- [x] **Step 2: Run the focused test and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/tmp/tinyllmforge-pytest312-shim:$PWD \
/opt/homebrew/bin/python3.12 -m pytest \
  tools/test_qwen35_mtp_real_eager_reference_probe.py -q
```

Expected: fail because `_build_real_eager_reference_probe` and independent
reference equations do not exist.

- [x] **Step 3: Implement independent reference equations**

Add gate-local causal prefill and cached decode functions using:

```python
torch.nn.functional.scaled_dot_product_attention(
    query,
    key,
    value,
    is_causal=...,
    scale=scale,
)
```

The functions must independently:

- write current K/V to `context.slot_mapping`;
- gather visible K/V from `context.block_tables`;
- repeat K/V heads for GQA;
- return `[tokens, num_heads * head_dim]`;
- avoid calling any production Qwen3.5 eager-attention helper.

- [x] **Step 4: Implement the scenario runner**

For each production/reference side:

- create fresh sequence IDs;
- use identical token/position/hidden values independent of sequence ID;
- wrap `module.forward_step` to capture one-token logits;
- run `executor.observe_target_prefill()` and `executor.propose_batch()`;
- validate exactly `batch_size * max(q - 1, 0)` captured logits;
- abort proposal transactions and release all sequences in `finally`;
- restore helper entry points in `finally`.

Return the maximum float32 absolute logits difference and equality of all
proposal tokens plus all logits argmax values.

- [x] **Step 5: Run focused GREEN tests**

Run the Step 2 command.

Expected: all parameterized parity and corruption tests pass.

---

### Task 2: Real Runtime Installation

**Files:**
- Modify: `tools/test_qwen35_mtp_real_checkpoint_gate.py`
- Modify: `tools/qwen35_mtp_real_checkpoint_gate.py`

**Interfaces:**
- Consumes:
  - `_build_real_eager_reference_probe(...)`
- Produces:
  - runtime field `eager_reference_probe`;
  - no `eager_reference` blocker after successful construction.

- [x] **Step 1: Write the failing installation test**

Extend the fake `ModelRunner` runtime test to require:

```python
assert runtime["eager_reference_probe"] is eager_probe
assert set(runtime["blockers"]) == {"graph_eager"}
```

Require the builder to receive `executor`, `module`, `physical_store`, and
`hf_config`.

- [x] **Step 2: Run the gate contract test and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/tmp/tinyllmforge-pytest312-shim:$PWD \
/opt/homebrew/bin/python3.12 -m pytest \
  tools/test_qwen35_mtp_real_checkpoint_gate.py \
  -k 'real_runtime_installs' -q
```

Expected: fail because runtime still retains the eager/reference blocker and
does not expose the probe.

- [x] **Step 3: Install the probe fail-closed**

Construct the eager/reference and transaction probes independently. Record
the builder exception under its own blocker domain. Add only successfully
constructed callables to the runtime mapping.

- [x] **Step 4: Run focused GREEN tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/tmp/tinyllmforge-pytest312-shim:$PWD \
/opt/homebrew/bin/python3.12 -m pytest \
  tools/test_qwen35_mtp_real_eager_reference_probe.py \
  tools/test_qwen35_mtp_real_checkpoint_gate.py -q
```

Expected: all tests pass.

---

### Task 3: Real Checkpoint Evidence and Documentation

**Files:**
- Modify: `docs/superpowers/audits/2026-08-12-phase1-objective-coverage.md`
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify: this plan

- [x] **Step 1: Run the complete local MTP regression**

Use the same three isolated pytest groups as the physical-KV transaction
gate, adding the new eager/reference probe test to the first group.

- [x] **Step 2: Run the isolated remote real-checkpoint gate**

Run:

```bash
KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian \
bash tools/run_qwen35_mtp_real_checkpoint_gate_remote.sh
```

Use only GPU 7, `ControlMaster=no`, `ControlPath=none`, and a unique remote
run root.

- [x] **Step 3: Validate the downloaded artifact**

Require:

```text
eager_reference_argmax_equal=true
eager_reference_max_abs_diff is finite and nonnegative
accepted_slot_identity_preserved=true
rejected_slots_released=true
post_rollback_continuation_equal=true
backend failure domains=[graph_eager]
status=FAIL
promotion_classification=NOT_PROMOTABLE
```

- [x] **Step 4: Run static gates**

Run `py_compile`, wrapper `bash -n`, scoped whitespace checks, plan
checkbox validation, artifact schema validation, and the source-neutrality
scan.

- [x] **Step 5: Record evidence and limitations**

Record exact local counts, remote run root, local artifact path, checkpoint
hash, device/runtime, maximum logits difference, argmax result, the remaining
graph blocker, and explicit non-claims for TP4, KV offload, long context,
second model, CUDA Graph correctness, and performance.

## Evidence

TDD RED:

```text
algorithm probe:     5 failed because builder/reference functions were absent
runtime installation: 1 failed because eager_reference_probe was absent
```

Focused GREEN:

```text
tools/test_qwen35_mtp_real_eager_reference_probe.py
tools/test_qwen35_mtp_real_checkpoint_gate.py
result: 32 passed in 1.53s
```

Complete isolated local MTP regression:

```text
group 1: 51 passed in 3.50s
group 2: 48 passed in 3.29s
group 3: 27 passed in 0.11s
total:  126 passed
```

Real-checkpoint evidence:

```text
remote run root:
  /data00/home/sitian/sitian-workspace01/tllm/qwen35-mtp-runs/
    qwen35-mtp-20260813-044311-35753

local artifact:
  artifacts/qwen35-mtp-runs/
    qwen35-mtp-20260813-044311-35753/
      qwen35_mtp_real_checkpoint_gate.json

eager_reference_argmax_equal:   true
eager_reference_max_abs_diff:   0.171875
accepted slot identity:          true
rejected slots released:         true
rollback continuation equal:     true
remaining blocker:               graph_eager
status:                          FAIL
promotion classification:        NOT_PROMOTABLE
```

The `20260813` directory token is future-dated relative to the current date,
2026-08-12. It is an opaque remote-generated run identifier, not chronology
evidence.

The `0.171875` maximum is a bfloat16 full-logits difference between the
existing eager equations and an independent PyTorch SDPA reference. Greedy
argmax is equal for every required Q/batch case. This does not establish CUDA
Graph, TP4, KV-offload, long-context, second-model, or performance evidence.

