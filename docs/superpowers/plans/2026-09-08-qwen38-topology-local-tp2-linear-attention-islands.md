# Qwen3.8 Topology-Local TP2 Linear-Attention Islands Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and formally qualify a real-checkpoint four-GPU Stage-0
microgate that replaces Qwen3.8 linear-attention TP4 mixer execution with two
topology-local TP2 replicas and reports the complete benefit and cost.

**Architecture:** A model-neutral pair-layout contract defines two disjoint
rank pairs and logical TP2 identities. A Qwen3.8-specific experimental worker
loads the pinned checkpoint, constructs baseline TP4 and candidate TP2 views
for layer 0, transforms TP4 state quarters into replicated TP2 halves, and
benchmarks the complete mixer on active-token groups 1/4/8. Separate
assembler, verifier, and controller modules freeze evidence, independently
reconstruct classification, enforce remote storage/admission, and seal a
compact bundle.

**Tech Stack:** Python 3.11, PyTorch distributed/NCCL, CUDA events, pytest,
JSON/JSONL, SHA-256 manifests, SSH/Kerberos controller.

**Design spec:**
`docs/superpowers/specs/2026-09-08-qwen38-topology-local-tp2-linear-attention-islands-design.md`
at source anchor `3521872de3e48ba17872a458ce894a7c4b13189b`.

## Global Constraints

- Work only in `/Users/bytedance/Desktop/TinyLLMForge`.
- Branch is `feat/kv-sparse-attention`; push only
  `origin/feat/kv-sparse-attention`.
- Preserve unrelated tracked and untracked files; stage exact paths only.
- Do not create a worktree and do not dispatch subagents.
- Do not edit `tinyvllm/layers/linear.py` or integrate the candidate into the
  production model path during Stage 0.
- Keep the feature disabled by default.
- Model is `Qwen/Qwen3.8-27B` at revision
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- Formal execution uses exactly four GPUs and the immutable active-token
  matrix 1/4/8, two warmup pairs, and 15 measured pairs.
- Candidate pair groups are frozen before attempt creation; all ranks create
  groups in identical order.
- No candidate timed path may contain a global four-rank AllReduce,
  AllGather, barrier, device-wide synchronization, cross-pair correctness
  check, host polling, or hidden synchronous fallback.
- Candidate output/state tolerance against baseline is
  `atol=2e-2, rtol=2e-3`; pair-replica agreement is
  `atol=2e-4, rtol=2e-4`.
- Performance gates are: token-1 median speedup at least 5%; token-4/8
  geometric aggregate speedup at least 5%; token-4 and token-8 medians do not
  regress; every P99 regression is at most 3%; at least 11/15 improving pairs
  for token 4 and token 8; host median regression at most 10%; migration
  break-even at most 32 generated tokens.
- Memory gates are: projected integrated steady-state increment at most
  1,920 MiB per rank at capacity eight; peak allocated memory below 98% of
  physical memory; migration temporaries released before steady timing.
- Do not run `kinit` or `krenew`. Kerberos commands use
  `KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian`.
- Require at least 1,800 seconds of TGT lifetime at launch. This preserves a
  thirty-minute startup window but explicitly accepts that credentials may
  expire during the two-hour worker timeout, staging, verification, or compact
  download. External automatic cache refresh is allowed; the controller does
  not perform renewal itself.
- Never terminate, pause, adopt, or modify foreign GPU/process workloads.
- Remote task-owned files must remain below
  `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/`;
  never write them to remote `/` or `/tmp`.
- Large artifacts remain remote; download only the compact final bundle.
- A fresh formal attempt tag is immutable and may not be repaired or reused.
- Every commit uses
  `git -c core.hooksPath=/dev/null commit` and exactly one
  `Co-authored-by: TRAE CLI <noreply@bytedance.com>` trailer.
- A PID, running shell, complete manifest, passing verifier, or profiler trace
  is not performance evidence.

---

### Task 1: Pair topology and logical-rank contract

**Files:**

- Create: `tinyvllm/engine/topology_local_tp2_island.py`
- Create: `tools/test_topology_local_tp2_island.py`

**Interfaces:**

- Produces:
  `TopologyLocalTP2PairMap`,
  `TopologyLocalTP2RankIdentity`,
  `validate_pair_groups(pair_groups, world_size=4)`,
  `logical_half_bounds(total_width, logical_rank)`, and
  `select_best_pair_groups(topology_rows)`.
- Consumes no Qwen-specific objects and performs no process-group creation.

- [ ] **Step 1: Write failing validation and topology-selection tests**

```python
from tinyvllm.engine.topology_local_tp2_island import (
    TopologyLocalTP2PairMap,
    logical_half_bounds,
    select_best_pair_groups,
)


def test_pair_map_assigns_two_replicas_and_logical_ranks():
    mapping = TopologyLocalTP2PairMap(((0, 1), (2, 3)))
    assert mapping.identity(0).pair_id == 0
    assert mapping.identity(0).logical_rank == 0
    assert mapping.identity(1).logical_rank == 1
    assert mapping.identity(2).pair_id == 1
    assert mapping.identity(2).logical_rank == 0
    assert mapping.identity(3).logical_rank == 1


def test_pair_map_rejects_overlap_missing_rank_and_non_four_world():
    for value in (
        ((0, 1), (1, 3)),
        ((0, 1), (2, 4)),
        ((0, 1),),
    ):
        with pytest.raises(ValueError):
            TopologyLocalTP2PairMap(value)


def test_logical_half_bounds_are_contiguous_and_complete():
    assert logical_half_bounds(6144, 0) == (0, 3072)
    assert logical_half_bounds(6144, 1) == (3072, 3072)


def test_topology_selection_avoids_sys_links():
    rows = [
        {"left_rank": 0, "right_rank": 1, "link": "PXB"},
        {"left_rank": 1, "right_rank": 0, "link": "PXB"},
        {"left_rank": 2, "right_rank": 3, "link": "PIX"},
        {"left_rank": 3, "right_rank": 2, "link": "PIX"},
        {"left_rank": 0, "right_rank": 2, "link": "SYS"},
        {"left_rank": 2, "right_rank": 0, "link": "SYS"},
        {"left_rank": 1, "right_rank": 3, "link": "SYS"},
        {"left_rank": 3, "right_rank": 1, "link": "SYS"},
        {"left_rank": 0, "right_rank": 3, "link": "SYS"},
        {"left_rank": 3, "right_rank": 0, "link": "SYS"},
        {"left_rank": 1, "right_rank": 2, "link": "SYS"},
        {"left_rank": 2, "right_rank": 1, "link": "SYS"},
    ]
    assert select_best_pair_groups(rows) == ((0, 1), (2, 3))
```

- [ ] **Step 2: Run the tests and confirm RED**

Run:

```bash
pytest -q tools/test_topology_local_tp2_island.py
```

Expected: collection fails because
`tinyvllm.engine.topology_local_tp2_island` does not exist.

- [ ] **Step 3: Implement the immutable pair contract**

```python
from dataclasses import dataclass

_LINK_COST = {
    "PIX": 0,
    "PXB": 1,
    "PHB": 2,
    "NODE": 3,
    "SYS": 4,
}


@dataclass(frozen=True)
class TopologyLocalTP2RankIdentity:
    global_rank: int
    pair_id: int
    logical_rank: int
    pair_ranks: tuple[int, int]


@dataclass(frozen=True)
class TopologyLocalTP2PairMap:
    pair_groups: tuple[tuple[int, int], tuple[int, int]]

    def __post_init__(self):
        groups = tuple(tuple(group) for group in self.pair_groups)
        if len(groups) != 2 or any(len(group) != 2 for group in groups):
            raise ValueError("pair_groups must contain two rank pairs")
        flattened = tuple(rank for group in groups for rank in group)
        if sorted(flattened) != [0, 1, 2, 3]:
            raise ValueError("pair_groups must partition ranks 0..3")
        object.__setattr__(self, "pair_groups", groups)

    def identity(self, global_rank: int) -> TopologyLocalTP2RankIdentity:
        for pair_id, ranks in enumerate(self.pair_groups):
            if global_rank in ranks:
                return TopologyLocalTP2RankIdentity(
                    global_rank=global_rank,
                    pair_id=pair_id,
                    logical_rank=ranks.index(global_rank),
                    pair_ranks=ranks,
                )
        raise ValueError("global_rank is not present in pair_groups")


def logical_half_bounds(total_width: int, logical_rank: int) -> tuple[int, int]:
    if total_width <= 0 or total_width % 2:
        raise ValueError("total_width must be positive and divisible by two")
    if logical_rank not in (0, 1):
        raise ValueError("logical_rank must be zero or one")
    width = total_width // 2
    return logical_rank * width, width


def select_best_pair_groups(topology_rows) -> tuple[tuple[int, int], ...]:
    directed = {}
    for row in topology_rows:
        left = row["left_rank"]
        right = row["right_rank"]
        link = row["link"]
        if left == right or link not in _LINK_COST:
            raise ValueError("topology row is invalid")
        key = (left, right)
        if key in directed:
            raise ValueError("topology row is duplicated")
        directed[key] = _LINK_COST[link]
    costs = {}
    for left in range(4):
        for right in range(left + 1, 4):
            forward = directed.get((left, right))
            reverse = directed.get((right, left))
            if forward is None or reverse is None or forward != reverse:
                raise ValueError("topology rows are incomplete or asymmetric")
            costs[(left, right)] = forward
    matchings = (
        ((0, 1), (2, 3)),
        ((0, 2), (1, 3)),
        ((0, 3), (1, 2)),
    )
    return min(
        matchings,
        key=lambda matching: (
            tuple(sorted(costs[tuple(sorted(pair))] for pair in matching)),
            matching,
        ),
    )
```

The topology test fixture contains both directions for every rank pair.

- [ ] **Step 4: Run GREEN and adjacent import checks**

Run:

```bash
pytest -q tools/test_topology_local_tp2_island.py
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-tp2-pair-pycache \
  python3 -m py_compile \
  tinyvllm/engine/topology_local_tp2_island.py \
  tools/test_topology_local_tp2_island.py
```

Expected: all tests pass and compilation exits zero.

- [ ] **Step 5: Commit the pair contract**

```bash
git add -- \
  tinyvllm/engine/topology_local_tp2_island.py \
  tools/test_topology_local_tp2_island.py
git -c core.hooksPath=/dev/null commit \
  -m "feat(tp4): add topology-local TP2 pair contract" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

---

### Task 2: TP4-quarter to replicated-TP2 state transformation

**Files:**

- Modify: `tinyvllm/engine/topology_local_tp2_island.py`
- Modify: `tools/test_topology_local_tp2_island.py`

**Interfaces:**

- Consumes: `TopologyLocalTP2RankIdentity`.
- Produces:
  `TopologyLocalTP2StateIdentity`,
  `assemble_logical_state_half(quarters, logical_rank)`, and
  `validate_state_publication(source_identity, candidate_identity)`.

- [ ] **Step 1: Add failing state-order, identity, and immutability tests**

```python
def test_assemble_logical_state_half_uses_global_head_order():
    quarters = tuple(torch.full((1, 2), rank) for rank in range(4))
    assert torch.equal(
        assemble_logical_state_half(quarters, 0),
        torch.tensor([[0, 0], [1, 1]]),
    )
    assert torch.equal(
        assemble_logical_state_half(quarters, 1),
        torch.tensor([[2, 2], [3, 3]]),
    )


def test_state_assembly_does_not_mutate_source_quarters():
    quarters = tuple(torch.arange(4).reshape(2, 2) + rank for rank in range(4))
    snapshots = tuple(tensor.clone() for tensor in quarters)
    assemble_logical_state_half(quarters, 0)
    assert all(torch.equal(a, b) for a, b in zip(quarters, snapshots))


def test_state_publication_rejects_generation_or_request_drift():
    source = TopologyLocalTP2StateIdentity(
        request_id=7, generation=3, slot_id=2, layer_index=0
    )
    with pytest.raises(RuntimeError):
        validate_state_publication(
            source,
            dataclasses.replace(source, generation=4),
        )
```

- [ ] **Step 2: Run focused RED**

Run:

```bash
pytest -q tools/test_topology_local_tp2_island.py -k state
```

Expected: failures identify the missing state APIs.

- [ ] **Step 3: Implement pure state transformation and identity checks**

```python
@dataclass(frozen=True)
class TopologyLocalTP2StateIdentity:
    request_id: int
    generation: int
    slot_id: int
    layer_index: int


def assemble_logical_state_half(
    quarters: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    logical_rank: int,
) -> torch.Tensor:
    if len(quarters) != 4:
        raise ValueError("quarters must contain four tensors")
    first = 2 * logical_rank
    selected = quarters[first:first + 2]
    reference = selected[0]
    if any(
        tensor.shape[1:] != reference.shape[1:]
        or tensor.dtype != reference.dtype
        or tensor.device != reference.device
        for tensor in quarters
    ):
        raise ValueError("state quarters must have compatible layouts")
    return torch.cat(tuple(tensor.clone() for tensor in selected), dim=0)


def validate_state_publication(source, candidate) -> None:
    if source != candidate:
        raise RuntimeError("state publication identity mismatch")
```

The distributed worker later gathers four quarters into a tuple and calls the
pure transformer for convolution and recurrent components separately.

- [ ] **Step 4: Run GREEN**

Run:

```bash
pytest -q tools/test_topology_local_tp2_island.py
```

Expected: all pair and state tests pass.

- [ ] **Step 5: Commit the state contract**

```bash
git add -- \
  tinyvllm/engine/topology_local_tp2_island.py \
  tools/test_topology_local_tp2_island.py
git -c core.hooksPath=/dev/null commit \
  -m "feat(tp4): add TP2 island state transformation" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

---

### Task 3: Real-checkpoint mixer worker and paired timing

**Files:**

- Create: `tools/qwen38_topology_local_tp2_island_worker.py`
- Create: `tools/test_qwen38_topology_local_tp2_island_worker.py`

**Interfaces:**

- Consumes:
  `TopologyLocalTP2PairMap`,
  `assemble_logical_state_half`,
  Qwen3.8 checkpoint metadata/load APIs, and the layer-0
  `Qwen35LinearAttentionShell`.
- Produces:
  `build_case_matrix()`,
  `build_logical_tp2_layer_view(layer, logical_rank, pair_group)`,
  `run_mixer_pair(*, attempt: str, case: dict, baseline_layer,
  candidate_view: LogicalTP2LinearAttentionView, baseline_states: tuple,
  candidate_states: tuple, downstream_weight: torch.Tensor) -> dict`,
  `run_worker_campaign(*, attempt: str, source_revision: str,
  model_root: Path, pair_groups: tuple[tuple[int, int], tuple[int, int]],
  output_root: Path, cases: tuple[dict, ...]) -> dict`, and schema
  `qwen38.topology-local-tp2-island-worker.v1`.

- [ ] **Step 1: Write failing case-matrix and view-construction tests**

```python
def test_case_matrix_has_frozen_shape_order_and_alternating_arms():
    rows = worker.build_case_matrix()
    assert len(rows) == 3 * (2 + 15)
    assert {row["active_tokens"] for row in rows} == {1, 4, 8}
    measured = [row for row in rows if row["phase"] == "measured"]
    assert all(
        row["arm_order"]
        == (("baseline", "candidate") if row["repetition"] % 2 == 0
            else ("candidate", "baseline"))
        for row in measured
    )


def test_logical_view_selects_contiguous_half_and_pair_group():
    view = worker.build_logical_tp2_layer_view(
        fake_linear_layer(),
        logical_rank=1,
        pair_group="pair-b",
    )
    assert view.key_head_range == (8, 16)
    assert view.value_head_range == (24, 48)
    assert view.output_input_range == (3072, 6144)
    assert view.process_group == "pair-b"
```

Add tests that reject a non-linear layer, wrong pinned dimensions, missing
`prefill_weight`, absent FP32 accumulation, quantized weights, and any
candidate setup attempted after warmup.

- [ ] **Step 2: Run worker RED**

Run:

```bash
pytest -q tools/test_qwen38_topology_local_tp2_island_worker.py
```

Expected: import failure for the new worker module.

- [ ] **Step 3: Implement the frozen case matrix and candidate view**

Use immutable data classes for:

```python
@dataclass(frozen=True)
class LogicalTP2LinearAttentionView:
    logical_parallel_size: int
    logical_rank: int
    key_head_range: tuple[int, int]
    value_head_range: tuple[int, int]
    output_input_range: tuple[int, int]
    pair_group: object
    qkv_weight: torch.Tensor
    qkv_weight_segments: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]
    z_weight: torch.Tensor
    z_weight_half: torch.Tensor
    b_weight: torch.Tensor
    b_weight_half: torch.Tensor
    a_weight: torch.Tensor
    a_weight_half: torch.Tensor
    ab_weight_half: torch.Tensor
    conv_weight: torch.Tensor
    A_log: torch.Tensor
    dt_bias: torch.Tensor
    norm_weight: torch.Tensor
    norm_eps: float
    output_accumulation_weight: torch.Tensor
```

Construct the candidate view before warmup:

- select key heads `[0:8]` or `[8:16]`;
- select value heads `[0:24]` or `[24:48]`;
- preserve the complete fused QKV and Z projections and select the logical
  TP2 output segments afterward so the candidate retains the baseline BF16
  GEMM numerical paths and exact greedy-token result;
- concatenate the selected A and B row views once before warmup into one
  contiguous 48-row BF16 weight, then split its single GEMM output into
  24-row A and B halves;
- materialize the matching contiguous FP32 output-projection input-column
  half;
- keep all parameters immutable; and
- hash every selected tensor.

The test must reject three separate Q/K/V `F.linear` calls in the candidate
timed path, require exactly one `F.linear(hidden, view.qkv_weight)` call, and
require the complete `view.z_weight` projection rather than
`view.z_weight_half`, and require one
`F.linear(hidden, view.ab_weight_half)` call rather than separate A/B calls.
The remote diagnostic must retain exact greedy argmax equality for token
groups 1, 4, and 8 before any performance classification is accepted.

The worker loads the complete pinned model through the existing Qwen3.8
checkpoint path, verifies that layer 0 is linear attention, and retains only
the references required by the microgate.

- [ ] **Step 4: Implement baseline and candidate mixer arms**

Factor shared operations so both arms execute the same mathematical sequence.
The candidate uses the logical TP2 view and pair group:

```python
def _pair_reduce(output: torch.Tensor, pair_group) -> torch.Tensor:
    torch.distributed.all_reduce(output, group=pair_group)
    return output


def run_candidate_mixer(
    hidden,
    convolution_state,
    recurrent_state,
    *,
    view,
):
    qkv = torch.cat(tuple(
        F.linear(hidden, weight)
        for weight in view.qkv_weight_segments
    ), dim=-1)
    z = F.linear(hidden, view.z_weight_half)
    a, b = F.linear(hidden, view.ab_weight_half).split((24, 24), dim=-1)
    convolved, next_convolution = qwen35_causal_depthwise_conv(
        qkv, convolution_state, view.conv_weight
    )
    gated, next_recurrent = _run_gated_delta_and_norm(
        convolved, z, a, b, recurrent_state, view
    )
    local = F.linear(
        gated.float(),
        view.output_accumulation_weight,
    )
    _pair_reduce(local, view.pair_group)
    return local.to(hidden.dtype), next_convolution, next_recurrent
```

Implement the helper with the same operation order as
`Qwen35LinearAttentionShell._forward`:

```python
def _run_gated_delta_and_norm(
    convolved,
    projected_z,
    projected_a,
    projected_b,
    recurrent_state,
    view,
):
    token_count = convolved.shape[0]
    key_width = 8 * 128
    value_width = 24 * 128
    query, key, value = convolved.split(
        (key_width, key_width, value_width),
        dim=-1,
    )
    query = query.reshape(token_count, 8, 128).repeat_interleave(3, dim=1)
    key = key.reshape(token_count, 8, 128).repeat_interleave(3, dim=1)
    value = value.reshape(token_count, 24, 128)
    delta_rule = (
        qwen35_gated_delta_recurrent
        if token_count == 1
        else qwen35_gated_delta_chunk
    )
    core, next_recurrent = delta_rule(
        query,
        key,
        value,
        projected_a,
        projected_b,
        view.A_log,
        view.dt_bias,
        recurrent_state,
    )
    norm_core = core.reshape(-1, 128)
    norm_gate = projected_z.reshape(-1, 128)
    if token_count == 1:
        padded_core = norm_core.repeat(view.logical_parallel_size, 1)
        padded_gate = norm_gate.repeat(view.logical_parallel_size, 1)
        gated = qwen35_gated_rmsnorm(
            padded_core,
            padded_gate,
            view.norm_weight,
            eps=view.norm_eps,
        )[:norm_core.shape[0]]
    else:
        gated = qwen35_gated_rmsnorm(
            norm_core,
            norm_gate,
            view.norm_weight,
            eps=view.norm_eps,
        )
    return gated.reshape(token_count, value_width), next_recurrent
```

Do not call the production `RowParallelLinear.forward` in the candidate arm,
because it is bound to the global process group. Do not modify the production
module.

- [ ] **Step 5: Implement state migration and cost accounting**

For each active request state, gather the four TP4 quarters outside the timed
steady-state interval, call `assemble_logical_state_half`, and retain the
logical half on both replicas. Record:

```python
{
    "latency_ns": migration_latency_ns,
    "source_bytes": source_bytes,
    "transferred_bytes": transferred_bytes,
    "retained_bytes": retained_bytes,
    "temporary_peak_allocated_bytes": temporary_peak_allocated_bytes,
    "steady_allocated_bytes": steady_allocated_bytes,
    "temporary_tensor_count": 8,
    "temporary_live_tensor_count_after_release": 0,
    "source_digest": source_digest,
    "candidate_digest": candidate_digest,
}
```

Synchronize the four participating ranks immediately before the migration
start event. This barrier is outside the measured migration interval and
prevents prior per-rank digest/GC work from contaminating the collective
latency. It is not permitted inside the candidate steady-state timed path.

Allocate an exact-size persistent reservation for the unmeasured 47
linear-attention output-projection increments and all capacity-eight state
increments. Track all eight `all_gather` destination tensors with weak
references, remove their strong references, synchronize, and require zero
live temporary tensors. Keep allocator-observed steady bytes separate from
logical retained bytes so CUDA allocation-bin padding cannot masquerade as a
live temporary. Call `torch.cuda.reset_peak_memory_stats()` only after this
release proof, and only then begin warmup.

- [ ] **Step 6: Implement paired CUDA timing and untimed correctness**

Each case:

1. clones digest-bound baseline and candidate inputs;
2. runs arms in frozen alternating order;
3. records CUDA events around the complete mixer;
4. synchronizes only after both arm submissions;
5. records per-rank device latency and host submission;
6. projects outputs and states outside timing;
7. compares both pair replicas and the TP4 baseline;
8. performs a fixed downstream greedy-argmax projection; and
9. writes one append-only rank row.

The row includes pair identity, physical GPU identity, tensor digests,
component diagnostic timings, allocation counters, fallback counters, and
all correctness booleans.

- [ ] **Step 7: Add cleanup and failure tests**

Test that:

- process groups are destroyed exactly once;
- candidate state is unpublished after an exception;
- stale request/generation identity is rejected;
- no row is marked complete if one rank is missing;
- cleanup reports every owned tensor reservation released; and
- no output file is written outside the provided attempt root.

- [ ] **Step 8: Run worker GREEN**

Run:

```bash
pytest -q \
  tools/test_topology_local_tp2_island.py \
  tools/test_qwen38_topology_local_tp2_island_worker.py
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-tp2-worker-pycache \
  python3 -m py_compile \
  tinyvllm/engine/topology_local_tp2_island.py \
  tools/qwen38_topology_local_tp2_island_worker.py \
  tools/test_qwen38_topology_local_tp2_island_worker.py
```

Expected: all tests pass and compilation exits zero.

- [ ] **Step 9: Commit the worker**

```bash
git add -- \
  tinyvllm/engine/topology_local_tp2_island.py \
  tools/qwen38_topology_local_tp2_island_worker.py \
  tools/test_topology_local_tp2_island.py \
  tools/test_qwen38_topology_local_tp2_island_worker.py
git -c core.hooksPath=/dev/null commit \
  -m "feat(tp4): add topology-local TP2 island worker" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

---

### Task 4: Independent assembler and frozen classifier

**Files:**

- Create: `tools/assemble_qwen38_topology_local_tp2_island.py`
- Create: `tools/test_assemble_qwen38_topology_local_tp2_island.py`

**Interfaces:**

- Consumes rank JSONL rows, admission, topology, model identity, source
  identity, state-migration rows, memory rows, and cleanup.
- Produces the compact final bundle and producer classification.

- [ ] **Step 1: Write failing classification tests**

Create `passing_inputs()` with 180 measured rank rows
(`3 shapes * 15 pairs * 4 ranks`), 60 measured migration rows
(`15 repetitions * 4 ranks`), complete pair identity,
and memory rows. Assert exact classifications for:

```python
def test_assembler_classifies_complete_gate_as_go(tmp_path):
    result = assemble_bundle(tmp_path, **passing_inputs())
    assert result["classification"] == (
        "GO_TOPOLOGY_LOCAL_TP2_ISLAND_MICROGATE"
    )


@pytest.mark.parametrize(
    ("mutation", "classification"),
    [
        ("pair_disagreement", "NO_GO_CORRECTNESS_OR_LIFECYCLE"),
        ("baseline_numeric_failure", "NO_GO_CORRECTNESS_OR_LIFECYCLE"),
        ("token1_speedup_0049", "NO_GO_PERFORMANCE"),
        ("token48_geomean_0049", "NO_GO_PERFORMANCE"),
        ("p99_regression_0031", "NO_GO_PERFORMANCE"),
        ("improving_pairs_10", "NO_GO_PERFORMANCE"),
        ("host_regression_0101", "NO_GO_PERFORMANCE"),
        ("break_even_33", "NO_GO_MIGRATION_AMORTIZATION"),
        ("steady_increment_over_1920_mib", "NO_GO_MEMORY"),
        ("peak_ratio_09801", "NO_GO_MEMORY"),
    ],
)
def test_assembler_enforces_frozen_boundaries(
    tmp_path, mutation, classification
):
    inputs = mutate(passing_inputs(), mutation)
    assert assemble_bundle(tmp_path, **inputs)["classification"] == classification
```

Also reject duplicate JSON keys, NaN/Infinity, missing ranks, wrong row count,
unfrozen pair maps, source/model drift, candidate global collectives,
non-zero fallback counts, incomplete cleanup, and extra output files.

- [ ] **Step 2: Run assembler RED**

Run:

```bash
pytest -q tools/test_assemble_qwen38_topology_local_tp2_island.py
```

Expected: import failure for the new assembler.

- [ ] **Step 3: Implement aggregation and classification**

Use maximum-rank latency per measured pair. Use nearest-rank percentile
selection, ratio-of-medians speedup, and:

```python
aggregate_4_8 = (
    (1.0 + speedup_by_tokens[4])
    * (1.0 + speedup_by_tokens[8])
) ** 0.5 - 1.0

break_even_tokens = math.inf if median_savings_ns <= 0 else math.ceil(
    median_migration_ns / median_savings_ns
)
```

Apply gate precedence:

1. evidence/identity invalid -> `INVALID_EVIDENCE`;
2. correctness/lifecycle failure ->
   `NO_GO_CORRECTNESS_OR_LIFECYCLE`;
3. memory failure -> `NO_GO_MEMORY`;
4. migration break-even failure -> `NO_GO_MIGRATION_AMORTIZATION`;
5. performance failure -> `NO_GO_PERFORMANCE`;
6. otherwise -> `GO_TOPOLOGY_LOCAL_TP2_ISLAND_MICROGATE`.

Write all bundle files named in the spec and bind them with
`manifest.sha256`.

- [ ] **Step 4: Run assembler GREEN**

Run:

```bash
pytest -q tools/test_assemble_qwen38_topology_local_tp2_island.py
```

Expected: all assembler tests pass.

- [ ] **Step 5: Commit the assembler**

```bash
git add -- \
  tools/assemble_qwen38_topology_local_tp2_island.py \
  tools/test_assemble_qwen38_topology_local_tp2_island.py
git -c core.hooksPath=/dev/null commit \
  -m "feat(tp4): assemble TP2 island evidence" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

---

### Task 5: Independent verifier and terminal sealing

**Files:**

- Create: `tools/verify_qwen38_topology_local_tp2_island.py`
- Create: `tools/test_verify_qwen38_topology_local_tp2_island.py`

**Interfaces:**

- Consumes only the sealed bundle files.
- Produces an independently reconstructed receipt without importing the
  assembler.

- [ ] **Step 1: Write failing independent-verifier tests**

Copy a complete synthetic bundle fixture generated by the test itself, then
assert:

```python
def test_verifier_reconstructs_go_without_assembler_import(tmp_path):
    root = write_passing_bundle(tmp_path)
    result = verify_bundle(root)
    assert result["classification"] == (
        "GO_TOPOLOGY_LOCAL_TP2_ISLAND_MICROGATE"
    )
    assert "assemble_qwen38_topology_local_tp2_island" not in sys.modules


def test_verifier_rejects_mutated_row_after_manifest(tmp_path):
    root = write_passing_bundle(tmp_path)
    mutate_first_timing_row(root)
    with pytest.raises(ValueError, match="manifest"):
        verify_bundle(root)


def test_verifier_rejects_producer_classifier_disagreement(tmp_path):
    root = write_passing_bundle(tmp_path)
    rewrite_producer_classification(root, "NO_GO_PERFORMANCE")
    rewrite_manifest(root)
    with pytest.raises(ValueError, match="classification"):
        verify_bundle(root)
```

Cover missing/extra files, duplicate keys, non-finite numbers, rank coverage,
pair-map drift, topology drift, row-count drift, forbidden global candidate
collectives, memory, migration, performance, cleanup, and both verifier
receipt names.

- [ ] **Step 2: Run verifier RED**

Run:

```bash
pytest -q tools/test_verify_qwen38_topology_local_tp2_island.py
```

Expected: import failure for the verifier.

- [ ] **Step 3: Implement independent reconstruction**

Duplicate the frozen equations and thresholds in the verifier; do not import
assembler helpers. Verify hashes before parsing semantic content. Compare the
reconstructed summary recursively against the producer result, including
float values with exact serialized representations.

`--seal-terminal` requires the remote verifier receipt, writes the local
receipt atomically, then rewrites the manifest. `--check-only` is
non-mutating and validates both existing receipts.

- [ ] **Step 4: Run verifier GREEN**

Run:

```bash
pytest -q \
  tools/test_assemble_qwen38_topology_local_tp2_island.py \
  tools/test_verify_qwen38_topology_local_tp2_island.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit the verifier**

```bash
git add -- \
  tools/verify_qwen38_topology_local_tp2_island.py \
  tools/test_verify_qwen38_topology_local_tp2_island.py
git -c core.hooksPath=/dev/null commit \
  -m "feat(tp4): verify TP2 island evidence" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

---

### Task 6: Kerberos-aware remote controller

**Files:**

- Create: `tools/run_qwen38_topology_local_tp2_island.py`
- Create: `tools/test_run_qwen38_topology_local_tp2_island.py`

**Interfaces:**

- Consumes the worker, assembler, verifier, pinned model root, explicit remote
  root, and a fresh attempt tag.
- Produces controller plan, admission, supervision, cleanup, download, and
  terminal receipts.

- [ ] **Step 1: Write failing controller-safety tests**

Test:

- all remote paths are strict descendants of the approved `/data00` root;
- attempts must not exist before creation;
- Kerberos principal and TGT match and lifetime is sufficient;
- the controller never invokes `kinit` or `krenew`;
- exactly four GPUs are frozen;
- topology selects the best perfect matching;
- a changed process inventory blocks launch;
- only exact-tag owned process groups may be signaled during cleanup;
- SSH status 255 retries only within the fixed retry budget, with bounded
  one-, two-, and four-second delays between attempts;
- non-255 SSH failures return immediately without sleeping;
- raw traces are excluded from compact download;
- remote and local independent verifiers must agree;
- failure paths write a terminal controller receipt; and
- `--dry-run` performs no remote write.

- [ ] **Step 2: Run controller RED**

Run:

```bash
pytest -q tools/test_run_qwen38_topology_local_tp2_island.py
```

Expected: import failure for the controller.

- [ ] **Step 3: Implement plan and admission**

Mirror the proven controller boundaries from
`tools/run_lease_sealed_state_commit_overlap.py`:

```python
DEFAULT_REMOTE_ROOT = (
    "/data00/home/sitian/tinyllmforge-workspaces/"
    "command-timeline-20260818"
)
DEFAULT_REMOTE_PYTHON = "/data00/home/sitian/tllm/env/bin/python"
EXPECTED_KERBEROS_PRINCIPAL = "sitian@BYTEDANCE.COM"
EXPECTED_KERBEROS_TGT = "krbtgt/BYTEDANCE.COM@BYTEDANCE.COM"
```

The plan freezes source revision/tree hash, model revision, four physical GPU
UUIDs, pair groups, distributed port, workload matrix, thresholds, output
paths, and expected artifact names.

- [ ] **Step 4: Implement source staging, launch, supervision, and cleanup**

Stage only committed source. Launch four workers with:

```text
WORLD_SIZE=4
RANK is one of 0, 1, 2, or 3
LOCAL_RANK equals RANK
CUDA_VISIBLE_DEVICES is the frozen comma-separated physical GPU list
PAIR_GROUPS is the frozen JSON pair map
```

Register every owned PID/process group before supervision. On timeout or
failure, signal only registered exact-attempt descendants. Validate that all
worker ranks destroy process groups and release candidate reservations.

- [ ] **Step 5: Implement assembly, dual verification, and compact download**

Remote order:

1. assemble raw rows;
2. run remote independent verifier;
3. seal remote manifest;
4. download compact bundle;
5. run local independent verifier with `--seal-terminal`;
6. run local `--check-only`;
7. compare producer, remote verifier, and local verifier classification.

- [ ] **Step 6: Run controller GREEN**

Run:

```bash
pytest -q tools/test_run_qwen38_topology_local_tp2_island.py
```

Expected: all controller tests pass without network access.

- [ ] **Step 7: Commit the controller**

```bash
git add -- \
  tools/run_qwen38_topology_local_tp2_island.py \
  tools/test_run_qwen38_topology_local_tp2_island.py
git -c core.hooksPath=/dev/null commit \
  -m "feat(tp4): orchestrate TP2 island gate" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

---

### Task 7: Complete CPU verification and implementation publication

**Files:**

- Modify only files created in Tasks 1-6 if failures require focused fixes.

**Interfaces:**

- Produces a pushed source revision eligible for a fresh GPU attempt.

- [ ] **Step 1: Run the complete focused suite**

Run:

```bash
pytest -q \
  tools/test_topology_local_tp2_island.py \
  tools/test_qwen38_topology_local_tp2_island_worker.py \
  tools/test_assemble_qwen38_topology_local_tp2_island.py \
  tools/test_verify_qwen38_topology_local_tp2_island.py \
  tools/test_run_qwen38_topology_local_tp2_island.py
```

Expected: all tests pass.

- [ ] **Step 2: Compile every new Python file**

Run:

```bash
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-tp2-island-pycache \
  python3 -m py_compile \
  tinyvllm/engine/topology_local_tp2_island.py \
  tools/qwen38_topology_local_tp2_island_worker.py \
  tools/assemble_qwen38_topology_local_tp2_island.py \
  tools/verify_qwen38_topology_local_tp2_island.py \
  tools/run_qwen38_topology_local_tp2_island.py
```

Expected: exit zero.

- [ ] **Step 3: Run integrity checks**

Run:

```bash
git diff --check
git status --short
```

Expected: no whitespace errors; only intentional task paths are tracked
changes. Existing unrelated untracked artifacts remain untouched.

- [ ] **Step 4: Commit any focused verification fixes**

If Task 7 required changes:

```bash
git add -- \
  tinyvllm/engine/topology_local_tp2_island.py \
  tools/qwen38_topology_local_tp2_island_worker.py \
  tools/assemble_qwen38_topology_local_tp2_island.py \
  tools/verify_qwen38_topology_local_tp2_island.py \
  tools/run_qwen38_topology_local_tp2_island.py \
  tools/test_topology_local_tp2_island.py \
  tools/test_qwen38_topology_local_tp2_island_worker.py \
  tools/test_assemble_qwen38_topology_local_tp2_island.py \
  tools/test_verify_qwen38_topology_local_tp2_island.py \
  tools/test_run_qwen38_topology_local_tp2_island.py
git -c core.hooksPath=/dev/null commit \
  -m "fix(tp4): close TP2 island verification gaps" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

If no files changed, do not create an empty commit.

- [ ] **Step 5: Push and verify SHA equality**

Run:

```bash
git push origin feat/kv-sparse-attention
test "$(git rev-parse HEAD)" = \
  "$(git ls-remote origin refs/heads/feat/kv-sparse-attention | awk '{print $1}')"
```

Expected: push succeeds and SHA equality exits zero.

---

### Task 8: Dry-run and launch one immutable formal GPU attempt

**Files:**

- Create remotely under the approved attempt root.
- Download only:
  `artifacts/qwen38_topology_local_tp2_islands/20260908-qwen38-topology-local-tp2-island-stage0-r1/final_bundle/`

**Interfaces:**

- Consumes the pushed implementation SHA and current Kerberos/GPU state.
- Produces one terminal, immutable Stage-0 attempt.

- [ ] **Step 1: Verify local Kerberos without renewing it**

Run:

```bash
KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian klist
```

Expected: the `sitian@BYTEDANCE.COM` TGT has at least 1,800 seconds remaining.
If not, stop before remote mutation and record `BLOCKED_KERBEROS`.

- [ ] **Step 2: Run a non-mutating dry-run**

Run:

```bash
KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian \
python3 tools/run_qwen38_topology_local_tp2_island.py \
  --attempt 20260908-qwen38-topology-local-tp2-island-stage0-r1 \
  --remote-root \
    /data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818 \
  --model-root \
    /data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/models/Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --dry-run
```

Expected: plan and admission queries pass, `remote_write_performed=false`, and
the attempt path remains absent.

- [ ] **Step 3: Launch the fresh formal attempt**

Use the same command without `--dry-run`. The controller waits for one
admissible four-GPU snapshot, freezes the best pair matching, creates the
attempt once, and owns execution through terminal cleanup.

Do not launch a second attempt while the first has an active controller or
owned workers.

- [ ] **Step 4: Inspect real progress, not shell status**

Require all of:

- local controller attempt root;
- remote attempt root;
- launch-admission receipt;
- four worker-start receipts;
- registered owned PIDs;
- append-only rank rows increasing; and
- no resource-identity drift.

A background shell or PID alone does not satisfy this step.

- [ ] **Step 5: Require terminal evidence**

The attempt is terminal only after:

- all expected rows exist;
- cleanup is complete;
- producer result exists;
- remote independent verification exists;
- compact bundle is downloaded;
- local independent verification exists;
- `--check-only` passes; and
- all three classifications agree.

---

### Task 9: Terminal audit, handoff, commit, and push

**Files:**

- Create:
  `docs/superpowers/audits/2026-09-08-qwen38-topology-local-tp2-linear-attention-islands-audit.md`
- Modify: `AGENT_HANDOFF_STATE.md` at true EOF only.

**Interfaces:**

- Consumes the sealed final bundle and fresh `--check-only` result.
- Produces the claim boundary and next authorized direction.

- [ ] **Step 1: Write a prompt-to-artifact completion checklist**

Map every design requirement to:

- source file and commit;
- focused test;
- raw evidence file and row count;
- assembler field;
- independent-verifier reconstruction;
- terminal classification; and
- unresolved or intentionally out-of-scope item.

Do not use a passing verifier as a substitute for this mapping.

- [ ] **Step 2: Write the terminal audit**

Report:

- exact source/model/attempt identities;
- topology and pair map;
- row and lifecycle coverage;
- output/state correctness;
- token-1/4/8 median and P99 benefit;
- improving-pair counts;
- host submission;
- migration latency and break-even;
- calculated and observed memory cost;
- cleanup;
- producer/remote/local agreement;
- borrowed components versus project-specific composition; and
- explicit mechanism-versus-end-to-end claim boundary.

If the result is a no-go, preserve the failed route and name the next
different optimization family. Do not weaken or reinterpret the gate.

- [ ] **Step 3: Append the handoff at true EOF**

Record the terminal attempt, classification, source SHA, final-bundle path,
fresh tests, verifier command/result, pushed audit commit, and the exact next
command. Preserve all prior handoff content.

- [ ] **Step 4: Run final verification**

Run:

```bash
pytest -q \
  tools/test_topology_local_tp2_island.py \
  tools/test_qwen38_topology_local_tp2_island_worker.py \
  tools/test_assemble_qwen38_topology_local_tp2_island.py \
  tools/test_verify_qwen38_topology_local_tp2_island.py \
  tools/test_run_qwen38_topology_local_tp2_island.py
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-tp2-final-pycache \
  python3 -m py_compile \
  tinyvllm/engine/topology_local_tp2_island.py \
  tools/qwen38_topology_local_tp2_island_worker.py \
  tools/assemble_qwen38_topology_local_tp2_island.py \
  tools/verify_qwen38_topology_local_tp2_island.py \
  tools/run_qwen38_topology_local_tp2_island.py
python3 tools/verify_qwen38_topology_local_tp2_island.py \
  artifacts/qwen38_topology_local_tp2_islands/\
20260908-qwen38-topology-local-tp2-island-stage0-r1/final_bundle \
  --check-only
git diff --check
```

Expected: focused tests and compilation pass, post-seal verification
reconstructs the terminal classification, and no whitespace errors remain.

- [ ] **Step 5: Commit exact audit paths**

```bash
git add -- \
  docs/superpowers/audits/2026-09-08-qwen38-topology-local-tp2-linear-attention-islands-audit.md \
  AGENT_HANDOFF_STATE.md
git -c core.hooksPath=/dev/null commit \
  -m "docs(tp4): record topology-local TP2 island result" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

- [ ] **Step 6: Push and verify final remote identity**

Run:

```bash
git push origin feat/kv-sparse-attention
test "$(git rev-parse HEAD)" = \
  "$(git ls-remote origin refs/heads/feat/kv-sparse-attention | awk '{print $1}')"
```

Expected: local HEAD, tracking branch, and GitHub branch SHA agree.

---

### Task 10: Post-r5 short-chunk gated-delta specialization

**Files:**

- Modify: `tools/qwen38_topology_local_tp2_island_worker.py`
- Modify: `tools/test_qwen38_topology_local_tp2_island_worker.py`
- Modify:
  `docs/superpowers/specs/2026-09-08-qwen38-topology-local-tp2-linear-attention-islands-design.md`
- Modify:
  `docs/superpowers/audits/2026-08-16-phase1-completion-audit.md`
- Modify: `AGENT_HANDOFF_STATE.md`

**Interfaces:**

- Produces:
  `candidate_gated_delta_chunk_size(token_count: int) -> int`
- Consumes:
  `qwen35_gated_delta_chunk(..., chunk_size: int)`
- Preserves:
  token-1 recurrent execution, the current TP4 baseline, exact-greedy
  validation, all frozen gates, and the default production runtime.

- [x] **Step 1: Add failing short-chunk policy tests**

Add tests that require:

```python
assert worker.candidate_gated_delta_chunk_size(2) == 2
assert worker.candidate_gated_delta_chunk_size(4) == 4
assert worker.candidate_gated_delta_chunk_size(8) == 8
assert worker.candidate_gated_delta_chunk_size(9) == 64
assert worker.candidate_gated_delta_chunk_size(64) == 64
assert worker.candidate_gated_delta_chunk_size(65) == 64
```

Reject booleans, non-integers, and values below two. Inspect
`_run_gated_delta_and_norm` to require that token-1 still calls
`qwen35_gated_delta_recurrent` and that the multi-token branch passes
`chunk_size=candidate_gated_delta_chunk_size(token_count)` explicitly.

- [x] **Step 2: Run RED**

Run:

```bash
pytest -q \
  tools/test_qwen38_topology_local_tp2_island_worker.py \
  -k 'short_chunk or gated_delta_chunk_size'
```

Expected: FAIL because `candidate_gated_delta_chunk_size` is absent and the
candidate still relies on the default chunk size 64.

- [x] **Step 3: Implement the minimal candidate-only policy**

Add:

```python
def candidate_gated_delta_chunk_size(token_count: int) -> int:
    if (
        isinstance(token_count, bool)
        or not isinstance(token_count, int)
        or token_count < 2
    ):
        raise ValueError("multi-token chunk size requires token_count >= 2")
    return token_count if token_count <= 8 else 64
```

Keep token-1 on `qwen35_gated_delta_recurrent`. In the multi-token branch,
call:

```python
core, next_recurrent = qwen35_gated_delta_chunk(
    query,
    key,
    value,
    projected_a,
    projected_b,
    view.A_log,
    view.dt_bias,
    recurrent_state,
    chunk_size=candidate_gated_delta_chunk_size(token_count),
)
```

Do not modify `tinyvllm/layers/gated_delta.py`,
`tinyvllm/layers/qwen35_linear_attention.py`, or
`tinyvllm/layers/linear.py`.

- [x] **Step 4: Run GREEN and adjacent CPU verification**

Run:

```bash
pytest -q \
  tools/test_qwen38_topology_local_tp2_island_worker.py
pytest -q \
  tools/test_topology_local_tp2_island.py \
  tools/test_qwen38_topology_local_tp2_island_worker.py \
  tools/test_assemble_qwen38_topology_local_tp2_island.py \
  tools/test_verify_qwen38_topology_local_tp2_island.py \
  tools/test_run_qwen38_topology_local_tp2_island.py
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-tp2-short-chunk-pycache \
  python3 -m py_compile \
  tools/qwen38_topology_local_tp2_island_worker.py
git diff --check
```

Expected: worker tests, the complete focused suite, compilation, and
whitespace checks pass.

- [ ] **Step 5: Commit and push the source revision**

Stage only the worker, worker test, and this plan:

```bash
git add -- \
  tools/qwen38_topology_local_tp2_island_worker.py \
  tools/test_qwen38_topology_local_tp2_island_worker.py \
  docs/superpowers/plans/2026-09-08-qwen38-topology-local-tp2-linear-attention-islands.md
git -c core.hooksPath=/dev/null commit \
  -m "perf(tp4): specialize short gated-delta chunks" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

- [ ] **Step 6: Run a fresh diagnostic**

After verifying at least 1,800 seconds of Kerberos lifetime and four
admissible GPUs, run the controller with:

```bash
KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian \
python3 tools/run_qwen38_topology_local_tp2_island.py \
  --attempt \
  20260908-qwen38-topology-local-tp2-island-stage0-diagnostic-r16 \
  --dist-port 29719 \
  --retry-count 20
```

Require all correctness and lifecycle gates, 180 measurement rows, 60
migration rows, four clean ranks, producer/remote/local agreement, and the
unchanged performance and cost thresholds.

- [ ] **Step 7: Run one fresh formal attempt**

Only if diagnostic-r16 returns
`GO_TOPOLOGY_LOCAL_TP2_ISLAND_MICROGATE`, use a new immutable tag and port:

```bash
KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian \
python3 tools/run_qwen38_topology_local_tp2_island.py \
  --attempt 20260908-qwen38-topology-local-tp2-island-stage0-r6 \
  --dist-port 29729 \
  --retry-count 20
```

Do not rerun r6 if it fails. Preserve its classification and evidence.

- [ ] **Step 8: Close the evidence loop**

Update the audit and append the handoff at true EOF with:

- r5 as the immutable `NO_GO_PERFORMANCE` predecessor;
- diagnostic-r16 and formal-r6 identities and classifications;
- both ingredients in the candidate claim;
- benefit and cost for every gate;
- producer, remote verifier, and local verifier agreement;
- exact source and remote SHA equality; and
- the one-layer microgate versus whole-model claim boundary.
