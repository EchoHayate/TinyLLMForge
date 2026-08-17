# Qwen3.5 Decode Row-Parallel Projection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Qwen3.5 attention output-projection input AllGather plus replicated weight execution with true row-parallel weight shards and validate the resulting TP4 decode performance.

**Architecture:** Reuse the existing `RowParallelLinear` boundary: each rank owns the checkpoint columns corresponding to its local attention heads, computes a local partial hidden-size output, and AllReduces that output. Preserve replicated decoder hidden states so the change is isolated to two projection sites and their checkpoint-binding contract.

**Tech Stack:** Python, PyTorch distributed/NCCL, pytest, TinyLLMForge Qwen3.5 checkpoint binding, CUDA Event/NVTX profiling, Nsight Systems, SSH remote execution.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Do not switch branches, stage, commit, stash, reset, push, or run `git clean`.
- Use GPUs `2,4,5,6` only for the real TP4 run.
- Before a real run, each selected GPU must have at least 25 GiB free and utilization no greater than 10 percent.
- Shared low-utilization GPUs are allowed; label results shared and non-exclusive.
- Do not create dummy reservations and do not kill unrelated processes.
- Every new full run uses a fresh tag and preserves earlier attempts.
- Cleanup may match only processes carrying the current attempt tag.
- Do not modify canonical manifests, case-matrix schemas, existing `profile.json` schemas, or r607-r620 artifacts.
- Do not claim a performance improvement from structural evidence alone.

---

### Task 1: Lock the row-parallel layout contract with failing tests

**Files:**
- Create: `tools/test_qwen35_output_projection_row_parallel.py`
- Modify: `tools/test_qwen35_concrete_component_factory.py`
- Modify: `tools/test_qwen35_checkpoint_target_binding.py`
- Modify: `tools/test_qwen35_checkpoint_assignment.py`

**Interfaces:**
- Consumes: `RowParallelLinear(input_size, output_size, bias=False)` and its axis-1 `weight_loader`.
- Produces: test coverage proving local head order, local checkpoint columns, local weight shapes, and summed-output equivalence.

- [ ] **Step 1: Add a synthetic mathematical-equivalence test**

Create `tools/test_qwen35_output_projection_row_parallel.py` with:

```python
from contextlib import contextmanager

import torch
import torch.nn.functional as F

from tinyvllm.layers.linear import RowParallelLinear


@contextmanager
def _tp_layout(rank: int, world_size: int):
    original_rank = torch.distributed.get_rank
    original_world_size = torch.distributed.get_world_size
    torch.distributed.get_rank = lambda: rank
    torch.distributed.get_world_size = lambda: world_size
    try:
        yield
    finally:
        torch.distributed.get_rank = original_rank
        torch.distributed.get_world_size = original_world_size


def test_qwen35_output_projection_shards_checkpoint_columns_and_sums():
    world_size = 4
    input_size = 16
    output_size = 8
    full_input = torch.arange(
        3 * input_size,
        dtype=torch.float32,
    ).reshape(3, input_size).div(11)
    full_weight = torch.arange(
        output_size * input_size,
        dtype=torch.float32,
    ).reshape(output_size, input_size).sub(17).div(13)

    partial_outputs = []
    for rank in range(world_size):
        with _tp_layout(rank, world_size):
            projection = RowParallelLinear(
                input_size,
                output_size,
                bias=False,
            )
        projection.weight.weight_loader(
            projection.weight,
            full_weight,
        )
        local_width = input_size // world_size
        local_input = full_input.narrow(
            1,
            rank * local_width,
            local_width,
        )
        expected_weight = full_weight.narrow(
            1,
            rank * local_width,
            local_width,
        )
        torch.testing.assert_close(
            projection.weight,
            expected_weight,
        )
        partial_outputs.append(
            F.linear(local_input, projection.weight)
        )

    torch.testing.assert_close(
        torch.stack(partial_outputs).sum(dim=0),
        F.linear(full_input, full_weight),
        rtol=1e-5,
        atol=1e-5,
    )
```

- [ ] **Step 2: Change component-factory expectations to the target layout**

In `tools/test_qwen35_concrete_component_factory.py`, replace the two
`ReplicatedWeightRowParallelLinear` assertions with:

```python
assert type(linear.out_proj) is RowParallelLinear
assert linear.out_proj.weight.shape == (
    config.hidden_size,
    (
        config.linear_num_value_heads
        * config.linear_value_head_dim
        // world_size
    ),
)

assert type(full.output_projection) is RowParallelLinear
assert full.output_projection.weight.shape == (
    config.hidden_size,
    (
        config.num_attention_heads
        * config.head_dim
        // world_size
    ),
)
```

- [ ] **Step 3: Change binding fixtures to the target layout**

In `tools/test_qwen35_checkpoint_target_binding.py`, construct both affected
fixtures with `RowParallelLinear`:

```python
out_proj=_bf16(RowParallelLinear(
    global_value_width,
    HIDDEN_SIZE,
    bias=False,
))

output_projection=_bf16(RowParallelLinear(
    FULL_QUERY_HEADS * FULL_HEAD_DIM,
    HIDDEN_SIZE,
    bias=False,
))
```

Add assertions that each binding destination has the input dimension divided
by `WORLD_SIZE`.

- [ ] **Step 4: Require axis-1 local assignment for both targets**

In `tools/test_qwen35_checkpoint_assignment.py`, make
`linear_attention.out_proj.weight` use the same local-column oracle already
used for full-attention output projection:

```python
if target.endswith((
    "linear_attention.out_proj.weight",
    "full_attention.output_projection.weight",
)):
    local_columns = transformed.shape[1] // world_size
    return transformed.narrow(
        1,
        rank * local_columns,
        local_columns,
    )
```

- [ ] **Step 5: Run the tests and verify the expected RED state**

Run in a PyTorch-enabled environment:

```bash
python -m pytest -q \
  tools/test_qwen35_output_projection_row_parallel.py \
  tools/test_qwen35_concrete_component_factory.py \
  tools/test_qwen35_checkpoint_target_binding.py \
  tools/test_qwen35_checkpoint_assignment.py
```

Expected: failures show the production factory and binding still construct or
require `ReplicatedWeightRowParallelLinear`; the synthetic standalone
`RowParallelLinear` test passes.

---

### Task 2: Migrate the two Qwen3.5 output projections

**Files:**
- Modify: `tinyvllm/models/qwen35_components.py`
- Modify: `tinyvllm/models/qwen35_checkpoint_binding.py`

**Interfaces:**
- Consumes: local full-attention query-head output and local linear-attention value-head output.
- Produces: replicated hidden-size outputs through `row_parallel_all_reduce`.

- [ ] **Step 1: Replace the component constructors**

In `tinyvllm/models/qwen35_components.py`, remove the unused
`ReplicatedWeightRowParallelLinear` import and construct:

```python
out_proj=RowParallelLinear(
    global_value_width,
    hidden_size,
    bias=False,
).to(dtype=torch.bfloat16)
```

and:

```python
output_projection=RowParallelLinear(
    query_heads * head_dim,
    hidden_size,
    bias=False,
).to(dtype=torch.bfloat16)
```

- [ ] **Step 2: Update exact-type binding requirements**

In `tinyvllm/models/qwen35_checkpoint_binding.py`, map both target suffixes to
`RowParallelLinear`:

```python
elif target.endswith("linear_attention.out_proj.weight"):
    expected_type = RowParallelLinear
elif target.endswith("full_attention.output_projection.weight"):
    expected_type = RowParallelLinear
```

Remove the specialized replicated-row shape check:

```python
if type(parent) is ReplicatedWeightRowParallelLinear:
    ...
```

The existing axis-one binding and `RowParallelLinear.weight_loader` become
the sole local-shape and assignment authority.

- [ ] **Step 3: Run the focused tests and verify GREEN**

Run:

```bash
python -m pytest -q \
  tools/test_qwen35_output_projection_row_parallel.py \
  tools/test_qwen35_concrete_component_factory.py \
  tools/test_qwen35_checkpoint_target_binding.py \
  tools/test_qwen35_checkpoint_assignment.py
```

Expected: all selected tests pass.

- [ ] **Step 4: Run source-level consistency checks**

Run:

```bash
rg -n "ReplicatedWeightRowParallelLinear" \
  tinyvllm/models/qwen35_components.py \
  tinyvllm/models/qwen35_checkpoint_binding.py
```

Expected: no output.

Run:

```bash
python3 -m py_compile \
  tinyvllm/models/qwen35_components.py \
  tinyvllm/models/qwen35_checkpoint_binding.py \
  tools/test_qwen35_output_projection_row_parallel.py
```

Expected: exit code 0.

---

### Task 3: Make the structural performance change auditable

**Files:**
- Modify: `tools/test_decode_internal_profile_wiring.py`
- Modify: `tools/test_run_qwen35_tp4_decode_internal_profile.py`
- Create: `tools/qwen35_decode_row_parallel_comparison.py`
- Create: `tools/test_qwen35_decode_row_parallel_comparison.py`

**Interfaces:**
- Consumes: r620 baseline `decode_summary.json`, candidate summary, and per-case `decode_profile.json`.
- Produces: `row_parallel_comparison.json` with structural, parity, and performance classifications.

- [ ] **Step 1: Add a failing profiler-wiring assertion**

Update `tools/test_decode_internal_profile_wiring.py` to assert:

```python
components_source = (
    root / "tinyvllm/models/qwen35_components.py"
).read_text()
assert "ReplicatedWeightRowParallelLinear(" not in components_source
assert components_source.count("RowParallelLinear(") >= 2

linear_source = (
    root / "tinyvllm/layers/linear.py"
).read_text()
assert '"row_parallel_all_reduce"' in linear_source
```

- [ ] **Step 2: Define the comparison API with tests**

Create `tools/qwen35_decode_row_parallel_comparison.py` with:

```python
def compare_decode_attempts(
    baseline_root,
    candidate_root,
    *,
    minimum_speedup_percent=5.0,
    maximum_tail_regression_percent=2.0,
):
    ...
```

The returned dictionary must include:

```python
{
    "schema_version": "qwen35.decode-row-parallel-comparison.v1",
    "baseline_root": str(baseline_root),
    "candidate_root": str(candidate_root),
    "legacy_all_gather_rows": {
        "baseline": int,
        "candidate": int,
    },
    "row_parallel_all_reduce_rows": {
        "baseline": int,
        "candidate": int,
    },
    "output_parity": bool,
    "steady_decode_wall_speedup_percent": float,
    "steady_decode_cuda_speedup_percent": float,
    "collective_cuda_speedup_percent": float,
    "classification": (
        "PERFORMANCE_PASS"
        | "STRUCTURAL_ONLY"
        | "NO_GO"
    ),
    "reasons": list[str],
}
```

Create tests covering:

- candidate legacy AllGather rows are nonzero -> `NO_GO`
- parity false -> `NO_GO`
- wall speedup at least 5 percent, CUDA improves, tail regression no greater
  than 2 percent -> `PERFORMANCE_PASS`
- structural gates pass but speedup is below 5 percent -> `STRUCTURAL_ONLY`

- [ ] **Step 3: Implement comparison from existing artifacts**

Read, without changing their schemas:

```text
<attempt>/decode_summary.json
<attempt>/download/cases/*/decode_profile.json
<attempt>/download/cases/*/case_rows.jsonl
```

Count operation names directly from per-rank collective rows. Compare the
same policy and steady-state field between r620 and the candidate. Reject
missing measured repetitions or mismatched generated-token sets.

- [ ] **Step 4: Run focused tests**

Run:

```bash
python3 -m pytest -q \
  tools/test_decode_internal_profile_wiring.py \
  tools/test_qwen35_decode_row_parallel_comparison.py \
  tools/test_run_qwen35_tp4_decode_internal_profile.py
```

Expected: all selected tests pass.

---

### Task 4: Run the focused regression suite

**Files:**
- Verify only; do not modify unrelated failures.

**Interfaces:**
- Consumes: Tasks 1-3.
- Produces: local and PyTorch-enabled regression evidence.

- [ ] **Step 1: Run local source/static tests**

Run:

```bash
python3 -m pytest -q \
  tools/test_decode_internal_profile_wiring.py \
  tools/test_qwen35_decode_row_parallel_comparison.py \
  tools/test_run_qwen35_tp4_decode_internal_profile.py \
  tools/test_qwen35_tp4_decode_internal_profile.py
```

Expected: all selected local tests pass.

- [ ] **Step 2: Run PyTorch-dependent tests in the remote environment**

Stage the current source through the existing fresh-attempt runner, then use
the staged remote source and:

```bash
/data00/home/sitian/sitian-workspace01/tllm/env/bin/python -m pytest -q \
  tools/test_qwen35_output_projection_row_parallel.py \
  tools/test_qwen35_concrete_component_factory.py \
  tools/test_qwen35_checkpoint_target_binding.py \
  tools/test_qwen35_checkpoint_assignment.py \
  tools/test_qwen35_full_attention_shell.py \
  tools/test_qwen35_linear_attention_shell.py
```

Expected: all selected tests pass. Missing dependencies are an environment
blocker, not a pass.

- [ ] **Step 3: Run syntax and whitespace checks**

Run:

```bash
python3 -m py_compile \
  tinyvllm/models/qwen35_components.py \
  tinyvllm/models/qwen35_checkpoint_binding.py \
  tools/qwen35_decode_row_parallel_comparison.py \
  tools/test_qwen35_decode_row_parallel_comparison.py

git diff --check -- \
  tinyvllm/models/qwen35_components.py \
  tinyvllm/models/qwen35_checkpoint_binding.py \
  tools/test_qwen35_output_projection_row_parallel.py \
  tools/test_qwen35_concrete_component_factory.py \
  tools/test_qwen35_checkpoint_target_binding.py \
  tools/test_qwen35_checkpoint_assignment.py \
  tools/test_decode_internal_profile_wiring.py \
  tools/qwen35_decode_row_parallel_comparison.py \
  tools/test_qwen35_decode_row_parallel_comparison.py
```

Expected: both commands exit 0.

---

### Task 5: Execute a fresh TP4 decode attempt

**Files:**
- Use: `tools/run_qwen35_tp4_decode_internal_profile.py`
- Create artifacts under:
  `experiments/qwen35_hybrid_state/qwen35-tp4-decode-row-parallel-20260811-r621-attempt001/`

**Interfaces:**
- Consumes: fixed GPUs, real checkpoint, r620-compatible workload and schema.
- Produces: structured decode profiles, Nsight evidence, parity rows, guards, and cleanup receipts.

- [ ] **Step 1: Execute the existing guarded runner with a fresh tag**

Run:

```bash
KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian \
python3 tools/run_qwen35_tp4_decode_internal_profile.py \
  --run-tag qwen35-tp4-decode-row-parallel-20260811-r621-attempt001
```

Expected:

- entry and pre-launch guards classify `READY`
- fixed GPUs are exactly `2,4,5,6`
- resource policy is shared low utilization and non-exclusive
- all structured cases and representative Nsight replay complete
- cleanup receipt is `CLEAN`

- [ ] **Step 2: Verify structural evidence**

Run a read-only artifact check that asserts:

```text
candidate replicated_weight_row_parallel_all_gather rows == 0
candidate row_parallel_all_reduce rows > 0
all four ranks present
all measured repetitions present
generated-token parity passes
cleanup is clean
```

Expected: all assertions pass.

- [ ] **Step 3: Preserve failure evidence if the attempt is blocked**

If resource, SSH, Kerberos, dependency, or runtime gates fail:

- retain the attempt directory
- write the exact blocker into `attempt_receipt.json`
- do not reuse the run tag
- fix only the diagnosed blocker
- rerun with the next fresh tag

---

### Task 6: Classify performance and complete the evidence audit

**Files:**
- Create:
  `experiments/qwen35_hybrid_state/<candidate>/row_parallel_comparison.json`
- Create:
  `experiments/qwen35_hybrid_state/<candidate>/completion_audit.json`
- Create:
  `experiments/qwen35_hybrid_state/<candidate>/completion_report.md`
- Modify: `AGENT_HANDOFF_STATE.md`

**Interfaces:**
- Consumes: r620 baseline and the first complete row-parallel candidate.
- Produces: final structural/correctness/performance classification and next-step decision.

- [ ] **Step 1: Generate the before/after comparison**

Run:

```bash
python3 tools/qwen35_decode_row_parallel_comparison.py \
  --baseline \
  experiments/qwen35_hybrid_state/qwen35-tp4-decode-internal-profile-20260811-r620-attempt001 \
  --candidate \
  experiments/qwen35_hybrid_state/qwen35-tp4-decode-row-parallel-20260811-r621-attempt001 \
  --output \
  experiments/qwen35_hybrid_state/qwen35-tp4-decode-row-parallel-20260811-r621-attempt001/row_parallel_comparison.json
```

Expected: classification is one of `PERFORMANCE_PASS`,
`STRUCTURAL_ONLY`, or `NO_GO`, with explicit reasons.

- [ ] **Step 2: Build the prompt-to-artifact completion audit**

Write `completion_audit.json` with one row for every requirement:

```json
{
  "objective": "Optimize Qwen3.5 decode TP AllGather, weight layout, and communication rather than restore.",
  "checks": [
    {
      "requirement": "two Qwen3.5 output projections use local axis-1 weight shards",
      "status": "PASS",
      "evidence": ["path and exact assertion"]
    },
    {
      "requirement": "legacy decode input AllGather is absent",
      "status": "PASS",
      "evidence": ["candidate collective count"]
    },
    {
      "requirement": "real TP4 output parity passes",
      "status": "PASS",
      "evidence": ["case-row parity artifact"]
    },
    {
      "requirement": "performance claim is supported by repeated E2E measurements",
      "status": "PASS or STRUCTURAL_ONLY or NO_GO",
      "evidence": ["row_parallel_comparison.json"]
    }
  ],
  "overall": "PASS or INCOMPLETE"
}
```

The audit may report overall `PASS` only when every correctness, structural,
resource, artifact, and cleanup gate is covered. A `STRUCTURAL_ONLY`
performance classification is an honest completed experiment, but not a
performance-win claim.

- [ ] **Step 3: Write the completion report**

`completion_report.md` must state:

- what changed
- what the result proves
- what it does not prove
- old and new collective counts
- old and new steady decode wall/CUDA/collective medians
- token parity result
- shared/non-exclusive resource caveat
- whether the next step is end-to-end hidden sharding, an AllReduce
  optimization, or no further change

- [ ] **Step 4: Update the handoff**

Append the candidate tag, classification, exact test commands, artifact
paths, limitations, and next action to `AGENT_HANDOFF_STATE.md`.

- [ ] **Step 5: Run the final audit checks**

Run:

```bash
python3 - <<'PY'
import json
from pathlib import Path

root = Path(
    "experiments/qwen35_hybrid_state/"
    "qwen35-tp4-decode-row-parallel-20260811-r621-attempt001"
)
audit = json.loads((root / "completion_audit.json").read_text())
assert audit["overall"] == "PASS"
assert all(row["status"] != "MISSING" for row in audit["checks"])
comparison = json.loads(
    (root / "row_parallel_comparison.json").read_text()
)
assert comparison["legacy_all_gather_rows"]["candidate"] == 0
assert comparison["output_parity"] is True
print(comparison["classification"])
PY

git diff --check -- \
  docs/superpowers/specs/2026-08-11-qwen35-decode-row-parallel-design.md \
  docs/superpowers/plans/2026-08-11-qwen35-decode-row-parallel.md \
  AGENT_HANDOFF_STATE.md \
  tinyvllm/models/qwen35_components.py \
  tinyvllm/models/qwen35_checkpoint_binding.py \
  tools/test_qwen35_output_projection_row_parallel.py \
  tools/qwen35_decode_row_parallel_comparison.py \
  tools/test_qwen35_decode_row_parallel_comparison.py
```

Expected: the audit prints the honest performance classification and all
checks exit 0.
