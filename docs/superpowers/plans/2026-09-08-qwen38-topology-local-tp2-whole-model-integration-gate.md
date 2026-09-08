# Qwen3.8 Topology-Local TP2 Whole-Model Integration Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans` to implement this plan task-by-task. The user
> has prohibited subagents and additional worktrees for this project. Steps
> use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate the default-off topology-local TP2-island linear-attention
path into real Qwen3.8-27B fixed-cohort requests and produce a source-bound
four-GPU A/B/B/A gate with correctness, TPOT, tail, TTFT, throughput, memory,
migration, cleanup, manifest, and dual-verifier evidence.

**Architecture:** Keep global TP4 ownership for prefill, full attention,
sampling, and engine coordination. At the prefill-to-decode boundary,
transactionally migrate every active request's 48 linear-attention state
quarters into replicated logical TP2 halves, switch only the eligible mixers
to pair-local execution, and expose independently verifiable hit telemetry.
Run baseline and candidate in separate engine epochs so allocator and state
layouts cannot contaminate each other.

**Tech Stack:** Python 3, PyTorch distributed/NCCL, TinyLLMForge
`LLMEngine`, pytest, safetensors checkpoint loading, JSON/JSONL evidence,
SHA-256 manifests, SSH/Kerberos remote orchestration.

## Global Constraints

- Work only in `/Users/bytedance/Desktop/TinyLLMForge`; the
  `/Users/bytedance/dev/TinyLLMForge` path is only a symlink to it.
- Do not create a worktree and do not use subagents.
- Do not modify `tinyvllm/layers/linear.py`.
- Keep `qwen38_topology_local_tp2_islands=False` by default.
- Support only frozen Qwen3.8-27B BF16, TP4, eager, fixed-cohort execution in
  this gate.
- Keep quantization, CPU offload, speculative decoding, prefix cache, KV
  offload, and dynamic arrivals disabled for the candidate.
- Preserve exact greedy output tokens and bitwise cross-pair BF16 mixer
  outputs.
- Treat ordinary exact-greedy decode as token-one segments; require zero
  short-chunk calls in the formal workload.
- Use strict-clean four-GPU admission: memory used at most `1,024 MiB`,
  utilization at most `5%`, and no compute processes.
- Never kill, pause, adopt, or modify a foreign process.
- Use `KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian`; never invoke `kinit`
  or `krenew`; launch floor is `1,800` seconds.
- Write remote task data only below
  `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/`.
- Keep large artifacts remote and download only a compact final bundle.
- Every attempt tag is immutable and may not be repaired or reused.
- Report benefit and cost together; a manifest or test pass is not a
  performance result.
- Exact-stage only named task files; never use `git add -A`, `git reset`, or
  `git clean`.
- Push only `origin/feat/kv-sparse-attention`.

---

## File map

### Runtime files

- Modify `tinyvllm/config.py`
  - add and validate the default-off candidate flag.
- Modify `tinyvllm/engine/topology_local_tp2_island.py`
  - add deterministic pair-group construction and pair-context identity.
- Create `tinyvllm/layers/qwen38_topology_local_tp2_linear_attention.py`
  - own logical TP2 parameter views and the pair-local mixer wrapper.
- Create `tinyvllm/engine/qwen38_topology_local_tp2_state.py`
  - own dual layouts, generation-sealed migration, publication, and release.
- Create `tinyvllm/engine/qwen38_topology_local_tp2_runtime.py`
  - coordinate phase transition, layer installation, telemetry, and cleanup.
- Modify `tinyvllm/engine/model_runner.py`
  - build the runtime after checkpoint load, transition before first decode,
    and expose acknowledged telemetry commands.
- Modify `tinyvllm/engine/llm_engine.py`
  - expose rank-complete runtime snapshots to gate workers.

### Gate files

- Create `tools/qwen38_topology_local_tp2_whole_model_worker.py`
  - run correctness, baseline, candidate, and TP2-x2 control cases.
- Create `tools/run_qwen38_topology_local_tp2_whole_model.py`
  - perform local/remote admission, immutable launch, download, and
    verification.
- Create `tools/assemble_qwen38_topology_local_tp2_whole_model.py`
  - reconcile raw rank/case rows and classify the result.
- Create `tools/verify_qwen38_topology_local_tp2_whole_model.py`
  - standard-library-only independent verification.

### Tests

- Create `tools/test_qwen38_topology_local_tp2_runtime.py`.
- Create `tools/test_qwen38_topology_local_tp2_state.py`.
- Create `tools/test_qwen38_topology_local_tp2_whole_model_worker.py`.
- Create `tools/test_run_qwen38_topology_local_tp2_whole_model.py`.
- Create `tools/test_assemble_qwen38_topology_local_tp2_whole_model.py`.
- Create `tools/test_verify_qwen38_topology_local_tp2_whole_model.py`.
- Modify focused existing Qwen3.8 tests only where a public constructor or
  management-command inventory changes.

### Documentation

- Modify
  `docs/superpowers/specs/2026-09-08-qwen38-topology-local-tp2-whole-model-integration-gate-design.md`
  only if implementation discovers a design-level contradiction.
- Append the terminal result to
  `docs/superpowers/audits/2026-08-16-phase1-completion-audit.md`.
- Append the terminal handoff to `AGENT_HANDOFF_STATE.md`.

---

### Task 1: Freeze the configuration and topology-group contract

**Files:**

- Modify: `tinyvllm/config.py`
- Modify: `tinyvllm/engine/topology_local_tp2_island.py`
- Create: `tools/test_qwen38_topology_local_tp2_runtime.py`

**Interfaces:**

- Produces:
  `Config.qwen38_topology_local_tp2_islands: bool`
- Produces:
  `create_topology_local_tp2_pair_context(distributed, global_rank,
  world_size, pair_map) -> TopologyLocalTP2PairContext`
- Consumes existing `TopologyLocalTP2PairMap`.

- [ ] **Step 1: Write failing configuration tests**

Load the real `Config` class with the same isolated `transformers.AutoConfig`
pattern used by `tools/test_model_runner_spec_verify.py`, then construct it
through this helper:

```python
def build_config(**overrides):
    Config = _load_real_config_class()
    model = tempfile.TemporaryDirectory()
    config = Config(model=model.name, **overrides)
    config._test_model_directory = model
    return config


config = build_config()
assert config.qwen38_topology_local_tp2_islands is False

with pytest.raises(
    ValueError,
    match="qwen38_topology_local_tp2_islands must be a bool",
):
    build_config(qwen38_topology_local_tp2_islands=1)

with pytest.raises(
    ValueError,
    match="requires tensor_parallel_size 4",
):
    build_config(
        tensor_parallel_size=2,
        enforce_eager=True,
        qwen38_topology_local_tp2_islands=True,
    )

with pytest.raises(
    ValueError,
    match="requires eager execution",
):
    build_config(
        tensor_parallel_size=4,
        enforce_eager=False,
        qwen38_topology_local_tp2_islands=True,
    )
```

Parametrize the incompatible Boolean modes and separately reject non-BF16
weight/KV modes:

```python
@pytest.mark.parametrize(
    "field",
    (
        "qwen35_mtp_enabled",
        "autoregressive_draft_enabled",
        "multi_sequence_cuda_graphs",
        "prefill_cuda_graphs",
        "cpu_offload",
        "kv_offload_mvp0",
        "exact_greedy_decode_burst",
        "graph_resident_greedy_tail",
    ),
)
def test_candidate_rejects_incompatible_runtime_modes(field):
    kwargs = {
        "tensor_parallel_size": 4,
        "enforce_eager": True,
        "qwen38_topology_local_tp2_islands": True,
        field: True,
    }
    with pytest.raises(ValueError, match="incompatible"):
        build_config(**kwargs)


@pytest.mark.parametrize(
    "overrides",
    (
        {"quantization": "int8"},
        {"kv_quant_bits": 8},
        {"act_quant_bits": 8},
        {"smoothquant_scale_path": "non-null"},
        {"quest_top_k_blocks": 1},
        {"kv_cartridge_blocks": 1},
        {"am_compact_blocks": 1},
    ),
)
def test_candidate_rejects_incompatible_numeric_modes(
    monkeypatch,
    tmp_path,
    overrides,
):
    if "smoothquant_scale_path" in overrides:
        scale = tmp_path / "sq.pt"
        scale.write_bytes(b"fixture")
        overrides = {"smoothquant_scale_path": str(scale)}
    with pytest.raises(ValueError, match="incompatible"):
        build_config(
            tensor_parallel_size=4,
            enforce_eager=True,
            qwen38_topology_local_tp2_islands=True,
            **overrides,
        )
```

- [ ] **Step 2: Run the RED tests**

Run:

```bash
python -m pytest \
  tools/test_qwen38_topology_local_tp2_runtime.py \
  -q
```

Expected: FAIL because the config field and pair-context constructor do not
exist.

- [ ] **Step 3: Implement the config field and fail-closed validation**

Add to `Config`:

```python
qwen38_topology_local_tp2_islands: bool = False
```

Insert this fail-closed block in `__post_init__` before the existing generic
quantization/offload assertions:

```python
if not isinstance(
    self.qwen38_topology_local_tp2_islands,
    bool,
):
    raise ValueError(
        "qwen38_topology_local_tp2_islands must be a bool"
    )
if self.qwen38_topology_local_tp2_islands:
    if self.tensor_parallel_size != 4:
        raise ValueError(
            "qwen38 topology-local TP2 islands requires "
            "tensor_parallel_size 4"
        )
    if not self.enforce_eager:
        raise ValueError(
            "qwen38 topology-local TP2 islands requires eager execution"
        )
    incompatible = any((
        self.qwen35_mtp_enabled,
        self.autoregressive_draft_enabled,
        self.multi_sequence_cuda_graphs,
        self.prefill_cuda_graphs,
        self.cpu_offload,
        self.kv_offload_mvp0,
        self.exact_greedy_decode_burst,
        self.graph_resident_greedy_tail,
        self.quantization is not None,
        self.kv_quant_bits != 0,
        self.act_quant_bits != 0,
        self.smoothquant_scale_path is not None,
        self.quest_top_k_blocks > 0,
        self.kv_cartridge_blocks > 0,
        self.am_compact_blocks > 0,
    ))
    if incompatible:
        raise ValueError(
            "qwen38 topology-local TP2 islands is incompatible "
            "with the selected runtime mode"
        )
```

- [ ] **Step 4: Add deterministic process-group construction**

Extend `topology_local_tp2_island.py` with:

```python
@dataclass(frozen=True)
class TopologyLocalTP2PairContext:
    identity: TopologyLocalTP2RankIdentity
    pair_map: TopologyLocalTP2PairMap
    pair_group: object
    all_pair_groups: tuple[object, object]


def create_topology_local_tp2_pair_context(
    distributed,
    *,
    global_rank: int,
    world_size: int,
    pair_map: TopologyLocalTP2PairMap,
) -> TopologyLocalTP2PairContext:
    if world_size != 4:
        raise ValueError(
            "topology-local TP2 pair context requires world_size 4"
        )
    identity = pair_map.identity(global_rank)
    groups = tuple(
        distributed.new_group(ranks=list(ranks))
        for ranks in pair_map.pair_groups
    )
    return TopologyLocalTP2PairContext(
        identity=identity,
        pair_map=pair_map,
        pair_group=groups[identity.pair_id],
        all_pair_groups=groups,
    )
```

Test that a fake distributed backend receives `new_group([0, 1])` before
`new_group([2, 3])` on every rank and that the returned logical rank is
correct.

- [ ] **Step 5: Run focused and adjacent tests**

Run:

```bash
python -m pytest \
  tools/test_topology_local_tp2_island.py \
  tools/test_qwen38_topology_local_tp2_runtime.py \
  tools/test_qwen38_model_runner_wiring.py \
  -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add -- \
  tinyvllm/config.py \
  tinyvllm/engine/topology_local_tp2_island.py \
  tools/test_qwen38_topology_local_tp2_runtime.py
git diff --cached --check
git commit -m "feat(tp4): add topology-local runtime admission"
```

---

### Task 2: Extract and qualify reusable logical TP2 parameter views

**Files:**

- Create:
  `tinyvllm/layers/qwen38_topology_local_tp2_linear_attention.py`
- Modify: `tools/qwen38_topology_local_tp2_island_worker.py`
- Modify: `tools/test_qwen38_topology_local_tp2_island_worker.py`
- Modify: `tools/test_qwen38_topology_local_tp2_runtime.py`

**Interfaces:**

- Produces:
  `LogicalTP2LinearAttentionView`
- Produces:
  `build_logical_tp2_linear_attention_view(layer, logical_rank,
  pair_group) -> LogicalTP2LinearAttentionView`
- Produces:
  `release_global_tp4_decode_accumulation(layer) -> dict`

- [ ] **Step 1: Move Stage-0 view tests to the runtime boundary**

Create tests for these exact properties:

```python
view = build_logical_tp2_linear_attention_view(
    fake_linear_attention(),
    logical_rank=1,
    pair_group=sentinel.pair_group,
)
assert view.logical_parallel_size == 2
assert view.logical_rank == 1
assert view.key_head_range == (8, 16)
assert view.value_head_range == (24, 48)
assert view.output_input_range == (3072, 6144)
assert view.qkv_weight_segments[0].shape == (1024, 5120)
assert view.z_weight.shape == (3072, 5120)
assert view.output_accumulation_weight.shape == (5120, 3072)
```

Also assert that the helper rejects quantized projections, missing
`prefill_weight`, wrong Qwen3.8 shapes, post-warmup construction, and a
logical rank outside `{0, 1}`.

- [ ] **Step 2: Run the RED tests**

```bash
python -m pytest \
  tools/test_qwen38_topology_local_tp2_runtime.py \
  tools/test_qwen38_topology_local_tp2_island_worker.py \
  -q
```

Expected: FAIL because the runtime layer module does not exist.

- [ ] **Step 3: Extract the Stage-0 implementation without semantic change**

Move the immutable view dataclass, checkpoint slicing, tensor digests, fused
A/B construction, output-half construction, and parameter reconstruction
proof into the new layer module. Preserve this public shape:

```python
@dataclass(frozen=True)
class LogicalTP2LinearAttentionView:
    logical_parallel_size: int
    logical_rank: int
    key_head_range: tuple[int, int]
    value_head_range: tuple[int, int]
    output_input_range: tuple[int, int]
    pair_group: object
    qkv_weight_segments: tuple[torch.Tensor, ...]
    z_weight: torch.Tensor
    b_weight: torch.Tensor
    a_weight: torch.Tensor
    conv_weight: torch.Tensor
    A_log: torch.Tensor
    dt_bias: torch.Tensor
    norm_weight: torch.Tensor
    output_accumulation_weight: torch.Tensor
    norm_eps: float
    tensor_digests: Mapping[str, str]
```

Implement release as:

```python
def release_global_tp4_decode_accumulation(layer) -> dict:
    weight = layer.out_proj.accumulation_weight
    if weight is None:
        raise RuntimeError(
            "global TP4 decode accumulation weight is missing"
        )
    released_bytes = weight.numel() * weight.element_size()
    reference = weakref.ref(weight)
    layer.out_proj.accumulation_weight = None
    del weight
    gc.collect()
    if reference() is not None:
        raise RuntimeError(
            "global TP4 decode accumulation weight remained live"
        )
    return {
        "released_bytes": int(released_bytes),
        "released": True,
    }
```

- [ ] **Step 4: Make the Stage-0 worker import the shared helpers**

Delete only the duplicated helper definitions from the worker and import:

```python
from tinyvllm.layers.qwen38_topology_local_tp2_linear_attention import (
    LogicalTP2LinearAttentionView,
    build_layer_parameter_identity,
    build_logical_tp2_linear_attention_view,
)
```

Keep Stage-0 schemas and output byte-for-byte compatible.

- [ ] **Step 5: Run regression tests**

```bash
python -m pytest \
  tools/test_topology_local_tp2_island.py \
  tools/test_qwen38_topology_local_tp2_island_worker.py \
  tools/test_assemble_qwen38_topology_local_tp2_island.py \
  tools/test_verify_qwen38_topology_local_tp2_island.py \
  -q
```

Expected: PASS with unchanged Stage-0 classification fixtures.

- [ ] **Step 6: Commit**

```bash
git add -- \
  tinyvllm/layers/qwen38_topology_local_tp2_linear_attention.py \
  tools/qwen38_topology_local_tp2_island_worker.py \
  tools/test_qwen38_topology_local_tp2_island_worker.py \
  tools/test_qwen38_topology_local_tp2_runtime.py
git diff --cached --check
git commit -m "refactor(tp4): share logical TP2 parameter views"
```

---

### Task 3: Build the generation-sealed TP4-to-TP2 state transaction

**Files:**

- Create: `tinyvllm/engine/qwen38_topology_local_tp2_state.py`
- Create: `tools/test_qwen38_topology_local_tp2_state.py`

**Interfaces:**

- Consumes `HybridStateLease`, `HybridStateTensorPool`,
  `Qwen35CrossLayerStateTransaction`, and `TopologyLocalTP2PairContext`.
- Produces:
  `Qwen38TopologyLocalTP2StateOwner`.
- Produces:
  `build_qwen38_topology_local_tp2_state_owner(*, hf_config, capacity: int,
  device, source_transaction: Qwen35CrossLayerStateTransaction,
  pair_identity: TopologyLocalTP2RankIdentity,
  all_gather: Callable) -> Qwen38TopologyLocalTP2StateOwner`.
- Produces:
  `migrate(leases: tuple[HybridStateLease, ...]) -> tuple[dict, ...]`,
  `gather(leases: tuple[HybridStateLease, ...]) -> tuple[dict, ...]`,
  `commit(leases: tuple[HybridStateLease, ...],
  candidates: tuple[dict, ...]) -> tuple[dict, ...]`,
  `release(leases: tuple[HybridStateLease, ...]) -> tuple[dict, ...]`,
  and `snapshot() -> dict`.

- [ ] **Step 1: Write state-layout and migration RED tests**

Use real CPU torch tensors and fake collectives. Cover:

```python
owner = build_qwen38_topology_local_tp2_state_owner(
    hf_config=frozen_qwen38_config(),
    capacity=8,
    device="cpu",
    source_transaction=tp4_transaction,
    pair_identity=TopologyLocalTP2PairMap(
        ((0, 1), (2, 3))
    ).identity(0),
    all_gather=fake_world_all_gather,
)
assert owner.destination_pool.layout.bytes_per_slot == (
    2 * owner.source_pool.layout.bytes_per_slot
)

rows = owner.migrate((lease,))
assert len(rows) == 48
assert all(row["published"] for row in rows)
assert owner.phase_for(lease) == "tp2_decode"
```

Also cover stale generation, duplicate migration, partial collective failure,
wrong layer inventory, wrong source shape, candidate commit before
publication, release, and rollback leaving TP4 authoritative.

- [ ] **Step 2: Run the RED tests**

```bash
python -m pytest \
  tools/test_qwen38_topology_local_tp2_state.py \
  -q
```

Expected: FAIL because the state owner does not exist.

- [ ] **Step 3: Implement dual-layout ownership**

Construct the destination layout with:

```python
destination_layout = build_qwen35_hybrid_state_layout(
    hf_config,
    tensor_parallel_size=2,
    dtype=torch.bfloat16,
    recurrent_dtype=torch.float32,
    speculative_tokens=1,
)
```

The owner must keep a per-lease state machine:

```text
tp4_prefill -> migrating -> tp2_decode -> released
```

For each component, world-gather four contiguous TP4 quarters, then select:

```python
first = 2 * pair_identity.logical_rank
logical_half = assemble_logical_state_half(
    tuple(gathered_quarters),
    pair_identity.logical_rank,
)
```

Publish all 48 layers atomically only after every destination copy succeeds.
On failure, erase the unpublished destination rows and restore
`tp4_prefill`.

- [ ] **Step 4: Add lifecycle and release evidence**

`snapshot()` must return:

```python
{
    "schema_version":
        "qwen38.topology-local-tp2-state-snapshot.v1",
    "source_layout_fingerprint": str,
    "destination_layout_fingerprint": str,
    "capacity": int,
    "leases": list,
    "migration_rows": list,
    "temporary_live_tensors": int,
    "publication_count": int,
    "rollback_count": int,
}
```

Require `temporary_live_tensors == 0` before candidate decode.

- [ ] **Step 5: Run tests**

```bash
python -m pytest \
  tools/test_qwen38_topology_local_tp2_state.py \
  tools/test_topology_local_tp2_island.py \
  -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add -- \
  tinyvllm/engine/qwen38_topology_local_tp2_state.py \
  tools/test_qwen38_topology_local_tp2_state.py
git diff --cached --check
git commit -m "feat(tp4): add generation-sealed TP2 state migration"
```

---

### Task 4: Implement the pair-local linear-attention wrapper

**Files:**

- Modify:
  `tinyvllm/layers/qwen38_topology_local_tp2_linear_attention.py`
- Modify: `tools/test_qwen38_topology_local_tp2_runtime.py`

**Interfaces:**

- Produces:
  `Qwen38TopologyLocalTP2LinearAttention`.
- Method:
  `forward(hidden_states, convolution_state, recurrent_state)`.
- Method:
  `telemetry_snapshot() -> dict`.

- [ ] **Step 1: Write wrapper RED tests**

Test baseline prefill dispatch, candidate decode dispatch, and counters:

```python
wrapper = Qwen38TopologyLocalTP2LinearAttention(
    baseline=baseline,
    candidate_view=view,
    pair_reduce=fake_pair_reduce,
)
assert wrapper.phase == "tp4_prefill"
assert wrapper(hidden, conv4, recurrent4) == baseline_result

wrapper.activate_tp2_decode()
output, conv2, recurrent2 = wrapper(
    hidden_one_token,
    conv2_state,
    recurrent2_state,
)
snapshot = wrapper.telemetry_snapshot()
assert snapshot["tp2_decode_calls"] == 1
assert snapshot["recurrent_token_one_calls"] == 1
assert snapshot["short_chunk_calls"] == 0
assert snapshot["global_tp4_decode_all_reduce_calls"] == 0
```

Test that token counts two through eight select the exact token count as chunk
size, token counts above eight retain 64, and a phase switch cannot be
reversed.

- [ ] **Step 2: Run the RED tests**

```bash
python -m pytest \
  tools/test_qwen38_topology_local_tp2_runtime.py \
  -q
```

Expected: FAIL because the wrapper does not exist.

- [ ] **Step 3: Implement candidate execution**

Use the Stage-0 operation order:

```python
def forward(
    self,
    hidden_states,
    convolution_state,
    recurrent_state,
):
    if self.phase == "tp4_prefill":
        self._telemetry.tp4_prefill_calls += 1
        return self.baseline(
            hidden_states,
            convolution_state,
            recurrent_state,
        )
    token_count = int(hidden_states.shape[0])
    projected = self._project_logical_tp2(hidden_states)
    core, next_recurrent = self._run_delta(
        projected,
        recurrent_state,
        token_count=token_count,
    )
    next_convolution = projected.next_convolution
    local = self._output_projection(core, projected.gate)
    self._pair_reduce(local, self.candidate_view.pair_group)
    output = local.to(dtype=hidden_states.dtype)
    self._telemetry.record_decode(token_count)
    return output, next_convolution, next_recurrent
```

Do not add a global collective, barrier, host poll, allocation, or
cross-pair comparison to this method.

- [ ] **Step 4: Prove exact cross-pair output support**

Add:

```python
def output_digest(output: torch.Tensor) -> str:
    if output.dtype is not torch.bfloat16:
        raise ValueError("candidate output digest requires BF16")
    return hashlib.sha256(
        output.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    ).hexdigest()
```

This helper is correctness-only and must not be called by the timed wrapper.
Tests assert identical inputs/views produce identical digests and a one-bit
change does not.

- [ ] **Step 5: Run focused tests**

```bash
python -m pytest \
  tools/test_qwen38_topology_local_tp2_runtime.py \
  tools/test_qwen38_topology_local_tp2_island_worker.py \
  -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add -- \
  tinyvllm/layers/qwen38_topology_local_tp2_linear_attention.py \
  tools/test_qwen38_topology_local_tp2_runtime.py
git diff --cached --check
git commit -m "feat(tp4): add pair-local Qwen3.8 mixer"
```

---

### Task 5: Install the candidate across the loaded 64-layer model

**Files:**

- Create: `tinyvllm/engine/qwen38_topology_local_tp2_runtime.py`
- Modify: `tinyvllm/engine/model_runner.py`
- Modify: `tools/test_qwen38_topology_local_tp2_runtime.py`
- Modify: `tools/test_qwen38_model_runner_wiring.py`

**Interfaces:**

- Produces:
  `Qwen38TopologyLocalTP2Runtime`.
- Produces:
  `install_qwen38_topology_local_tp2_runtime(model, owner, pair_context,
  capacity)`.
- Runtime methods:
  `prepare_decode(leases)`, `snapshot()`, and `close()`.

- [ ] **Step 1: Write installation RED tests**

Assert:

```python
runtime = install_qwen38_topology_local_tp2_runtime(
    model=model,
    owner=owner,
    pair_context=pair_context,
    capacity=8,
)
assert runtime.linear_layer_indices == tuple(
    index for index in range(64) if index % 4 != 3
)
assert len(runtime.mixers) == 48
assert all(
    type(layer.linear_attention)
    is Qwen38TopologyLocalTP2LinearAttention
    for layer in model.layer_stack.layers
    if layer.block_type == "linear_attention"
)
assert all(
    layer.linear_attention is original[index]
    for index, layer in enumerate(model.layer_stack.layers)
    if layer.block_type == "full_attention"
)
```

Reject non-Qwen3.8 topology, a profile repository other than
`Qwen/Qwen3.8-27B`, a profile revision other than
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, non-BF16, non-TP4, missing
dense prefill weights, wrong linear-layer count, wrong pair map, and a second
installation.

- [ ] **Step 2: Run RED tests**

```bash
python -m pytest \
  tools/test_qwen38_topology_local_tp2_runtime.py \
  tools/test_qwen38_model_runner_wiring.py \
  -q
```

Expected: FAIL on missing installer/runtime.

- [ ] **Step 3: Implement the runtime owner**

The runtime owns:

```python
class Qwen38TopologyLocalTP2Runtime:
    model: Qwen35PackedForCausalLM
    baseline_owner: Qwen35HybridModelOwner
    candidate_state_owner: Qwen38TopologyLocalTP2StateOwner
    pair_context: TopologyLocalTP2PairContext
    mixers: tuple[Qwen38TopologyLocalTP2LinearAttention, ...]
    phase: str
```

`prepare_decode(leases)` must:

1. reject an empty or changed fixed cohort;
2. migrate every lease while baseline state is still authoritative;
3. publish the candidate transaction;
4. switch the layer stack to candidate adapters;
5. activate all 48 wrappers;
6. release TP4 decode-only accumulation weights;
7. synchronize once outside the measured decode interval; and
8. return migration and release evidence.

Any failure before step 3 rolls back. Any failure after step 3 quarantines the
runtime and raises without baseline replay.

- [ ] **Step 4: Wire construction after checkpoint loading**

In `ModelRunner.__init__`, after binding the existing Qwen3.8 owner:

```python
self.qwen38_topology_local_tp2_runtime = None
if config.qwen38_topology_local_tp2_islands:
    if self.qwen38_text_profile is None:
        raise ValueError(
            "topology-local TP2 islands requires Qwen3.8"
        )
    pair_map = TopologyLocalTP2PairMap(((0, 1), (2, 3)))
    pair_context = create_topology_local_tp2_pair_context(
        dist,
        global_rank=self.rank,
        world_size=self.world_size,
        pair_map=pair_map,
    )
    self.qwen38_topology_local_tp2_runtime = (
        install_qwen38_topology_local_tp2_runtime(
            model=self.model,
            owner=self.qwen35_hybrid_model_owner,
            pair_context=pair_context,
            capacity=config.max_num_seqs,
        )
    )
```

The controller controls physical rank ordering through
`CUDA_VISIBLE_DEVICES`; runtime global pairs remain deterministic.

- [ ] **Step 5: Run focused and adjacent tests**

```bash
python -m pytest \
  tools/test_qwen38_topology_local_tp2_runtime.py \
  tools/test_qwen38_model_runner_wiring.py \
  tools/test_qwen35_packed_stateful_linear_decoder_layer.py \
  tools/test_qwen35_packed_full_decoder_layer.py \
  tools/test_qwen35_packed_layer_stack.py \
  -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add -- \
  tinyvllm/engine/qwen38_topology_local_tp2_runtime.py \
  tinyvllm/engine/model_runner.py \
  tools/test_qwen38_topology_local_tp2_runtime.py \
  tools/test_qwen38_model_runner_wiring.py
git diff --cached --check
git commit -m "feat(tp4): install Qwen3.8 TP2 islands"
```

---

### Task 6: Add the fixed-cohort transition and rank-complete telemetry API

**Files:**

- Modify: `tinyvllm/engine/model_runner.py`
- Modify: `tinyvllm/engine/llm_engine.py`
- Modify: `tools/test_qwen38_topology_local_tp2_runtime.py`
- Modify: `tools/test_model_runner_spec_verify.py`

**Interfaces:**

- `ModelRunner.qwen38_topology_local_tp2_snapshot() -> dict`
- `LLMEngine.qwen38_topology_local_tp2_snapshots(timeout_s) -> tuple[dict,
  ...]`
- Candidate transition occurs before the first decode model call.

- [ ] **Step 1: Write transition RED tests**

Use a fake runtime and assert:

```python
runner._prepare_hybrid_state_batch(seqs, ())
runner._prepare_qwen38_topology_local_tp2_decode(
    seqs,
    is_prefill=False,
)
runtime.prepare_decode.assert_called_once_with(
    runner._last_hybrid_state_leases
)
```

Also assert:

- prefill never transitions;
- a second decode with the same cohort does not migrate twice;
- changed request IDs after activation raise
  `"fixed candidate cohort changed"`;
- mixed batches and released leases raise before model execution;
- snapshot is included in the acknowledged management-command allowlist.

- [ ] **Step 2: Run RED tests**

```bash
python -m pytest \
  tools/test_qwen38_topology_local_tp2_runtime.py \
  tools/test_model_runner_spec_verify.py \
  -q
```

Expected: FAIL because transition and snapshot methods are missing.

- [ ] **Step 3: Add the pre-decode hook**

Immediately after `_prepare_hybrid_state_batch` has resolved leases and before
`_run_eager_logits` or graph dispatch:

```python
def _prepare_qwen38_topology_local_tp2_decode(
    self,
    seqs,
    *,
    is_prefill,
):
    runtime = self.qwen38_topology_local_tp2_runtime
    if runtime is None or is_prefill:
        return None
    request_ids = tuple(int(seq.seq_id) for seq in seqs)
    if request_ids != self._last_hybrid_state_request_ids:
        raise RuntimeError(
            "fixed candidate cohort changed before decode"
        )
    return runtime.prepare_decode(
        self._last_hybrid_state_leases
    )
```

Call it exactly once on the first decode step. Candidate mode must bypass all
CUDA Graph paths and remain eager.

- [ ] **Step 4: Add rank-complete snapshots**

ModelRunner:

```python
def qwen38_topology_local_tp2_snapshot(self):
    runtime = self.qwen38_topology_local_tp2_runtime
    if runtime is None:
        return {
            "schema_version":
                "qwen38.topology-local-tp2-runtime-snapshot.v1",
            "rank": self.rank,
            "enabled": False,
        }
    return runtime.snapshot()
```

LLMEngine uses `call_model_runner_acknowledged` and rejects missing,
duplicated, or mismatched ranks before returning ordered snapshots.

- [ ] **Step 5: Close pair groups and candidate state**

Invoke `runtime.close()` from `ModelRunner.exit()` before global process-group
destruction. The receipt must include:

```python
{
    "pair_groups_destroyed": 2,
    "candidate_state_released": True,
    "published_generations_remaining": 0,
    "temporary_live_tensors": 0,
}
```

- [ ] **Step 6: Run focused and adjacent tests**

```bash
python -m pytest \
  tools/test_qwen38_topology_local_tp2_runtime.py \
  tools/test_model_runner_spec_verify.py \
  tools/test_qwen38_model_runner_wiring.py \
  -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add -- \
  tinyvllm/engine/model_runner.py \
  tinyvllm/engine/llm_engine.py \
  tools/test_qwen38_topology_local_tp2_runtime.py \
  tools/test_model_runner_spec_verify.py
git diff --cached --check
git commit -m "feat(tp4): activate fixed-cohort TP2 decode"
```

---

### Task 7: Build the whole-model correctness and performance worker

**Files:**

- Create: `tools/qwen38_topology_local_tp2_whole_model_worker.py`
- Create: `tools/test_qwen38_topology_local_tp2_whole_model_worker.py`

**Interfaces:**

- `build_request_specs(prompt_tokens, output_tokens, concurrency,
  seed_namespace) -> tuple[dict, ...]`
- `run_engine_case(*, model_root: Path, arm: str, workload_id: str,
  request_specs: tuple[dict, ...], warmup: bool, epoch: int,
  repetition: int) -> dict`
- `run_correctness_campaign(*, model_root: Path, output_root: Path,
  workloads: Mapping[str, tuple], seed_namespace: str) -> dict`
- `run_performance_epoch(*, model_root: Path, output_root: Path, epoch: int,
  arm: str, workload_order: tuple[str, ...]) -> dict`
- `run_service_control(*, model_root: Path, output_root: Path,
  pair_devices: tuple[tuple[int, int], tuple[int, int]],
  workloads: tuple[str, ...]) -> dict`

- [ ] **Step 1: Write workload and timing RED tests**

Freeze:

```python
WORKLOADS = {
    "P0": ("causal", 256, 128, 1),
    "P1": ("causal", 2048, 128, 1),
    "Q0": ("online", 256, 128, 4),
    "Q1": ("online", 256, 128, 8),
    "Q2": ("online", 2048, 128, 4),
}
EPOCH_ARMS = ("baseline", "candidate", "candidate", "baseline")
MEASURED_REPETITIONS = 5
WARMUP_REPETITIONS = 2
```

Tests require deterministic disjoint correctness/timing prompts, exact 128
tokens, and:

```python
tpot_ns = (
    token_timestamps_ns[-1] - token_timestamps_ns[0]
) / 127
token_gaps_ns = [
    current - previous
    for previous, current in zip(
        token_timestamps_ns,
        token_timestamps_ns[1:],
    )
]
```

- [ ] **Step 2: Write candidate-evidence RED tests**

Fake engine snapshots must be rejected unless:

```python
assert row["tp2_decode_calls"] == expected_segments * 48
assert row["recurrent_token_one_calls"] == expected_segments * 48
assert row["short_chunk_calls"] == 0
assert row["ordinary_chunk_calls"] == 0
assert row["global_tp4_linear_decode_all_reduce_calls"] == 0
assert row["fallback_calls"] == 0
assert row["migration_publications"] == concurrency
assert row["full_attention_layer_count"] == 16
assert row["full_attention_tp4_collective_calls"] == (
    expected_segments * 16
)
assert row["pair_local_collective_sequence_match"] is True
assert row["post_warmup_request_path_allocations"] == 0
assert row["prefix_restore_calls"] == 0
assert row["prefix_publication_calls"] == 0
assert row["retry_after_mutation_calls"] == 0
assert row["duplicate_commit_calls"] == 0
```

Add negative cases for token mismatch, changed cohort, missing rank, duplicate
commit, nonfinite logits, missing layer, retained migration temporary, and
incomplete cleanup. Independently derive `expected_segments`, pair-local
collective calls/bytes, full-attention TP4 collective calls/bytes, and
migration publications from request rows, scheduler-step rows, token-count
rows, and the 64-layer model manifest; never accept producer summary totals
as authority.

- [ ] **Step 3: Run RED tests**

```bash
python -m pytest \
  tools/test_qwen38_topology_local_tp2_whole_model_worker.py \
  -q
```

Expected: FAIL because the worker does not exist.

- [ ] **Step 4: Implement real request execution**

Construct each engine as:

```python
engine = LLMEngine(
    str(model_root),
    tensor_parallel_size=4,
    enforce_eager=True,
    max_num_seqs=max(8, concurrency),
    max_model_len=prompt_tokens + output_tokens,
    max_num_batched_tokens=prompt_tokens * concurrency,
    qwen38_topology_local_tp2_islands=(arm == "candidate"),
)
```

Record admission time before `add_request`, capture
`engine.last_step_observation` after every `step()`, and require terminal
engine output to equal the accumulated token deltas. For candidate cases,
collect rank-complete snapshots before and after every measured request set.

- [ ] **Step 5: Implement correctness-only state checkpoints**

Enable correctness tracing only in `phase="correctness"`. Run five
deterministic request sets per workload, disjoint from all timing prompts.
Collect per-rank, per-layer device digests at:

```python
STATE_CHECKPOINTS = (
    "pre_migration",
    "post_migration",
    "token_1",
    "token_4",
    "token_8",
    "token_32",
    "token_128",
)
```

Require bitwise BF16 output digest equality across pair replicas and
canonical convolution/recurrent-state tolerance checks. Each row also binds
request, generation, lease, slot, layer, rank, pair, source revision, model
revision, exact argmax token, rank token agreement, finite logits, top-logit
ID/value tolerance, stop position/reason, and single-commit-per-step
evidence. Mark every correctness row `timing_authority=False`, and record its
additional retained-state/weight bytes separately from performance memory.

- [ ] **Step 6: Implement TP2-x2 control**

Launch two independent TP2 engine processes with disjoint
`CUDA_VISIBLE_DEVICES` pairs. Split request indexes by `index % 2`, synchronize
the shared start time, and compute aggregate makespan from the earliest
admission to latest completion. Label every row:

```python
{
    "arm": "TP2_X2_SERVICE_CONTROL",
    "classification_authority": False,
}
```

- [ ] **Step 7: Run worker tests**

```bash
python -m pytest \
  tools/test_qwen38_topology_local_tp2_whole_model_worker.py \
  tools/test_qwen38_topology_local_tp2_runtime.py \
  -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add -- \
  tools/qwen38_topology_local_tp2_whole_model_worker.py \
  tools/test_qwen38_topology_local_tp2_whole_model_worker.py
git diff --cached --check
git commit -m "feat(tp4): add TP2 whole-model worker"
```

---

### Task 8: Implement deterministic assembly and classification

**Files:**

- Create:
  `tools/assemble_qwen38_topology_local_tp2_whole_model.py`
- Create:
  `tools/test_assemble_qwen38_topology_local_tp2_whole_model.py`

**Interfaces:**

- `assemble_attempt(attempt_root: Path, output_root: Path) -> dict`
- `classify(summary: Mapping[str, object]) -> str`
- `nearest_rank_percentile(values, percentile) -> float`

- [ ] **Step 1: Write classifier RED tests**

Create a minimal passing fixture with:

```python
{
    "correctness_pass": True,
    "resource_identity_pass": True,
    "candidate_coverage_pass": True,
    "memory_pass": True,
    "measurement_complete": True,
    "tail_ttft_pass": True,
    "throughput_pass": True,
    "migration_pass": True,
    "aggregate_median_tpot_improvement_percent": 5.0,
    "improving_workload_count": 4,
    "median_regressing_workloads": [],
    "pair_direction_counts": {
        workload: 7 for workload in WORKLOADS
    },
}
```

Assert the precedence by mutating one field at a time and expecting:

```text
NO_GO_CORRECTNESS_OR_LIFECYCLE
NO_GO_RESOURCE_IDENTITY
NO_GO_CANDIDATE_NOT_EXERCISED
NO_GO_MEMORY_OR_ALLOCATION
INCONCLUSIVE_ENVIRONMENT_OR_MEASUREMENT
NO_GO_TAIL_OR_TTFT
NO_GO_THROUGHPUT
NO_GO_MIGRATION_AMORTIZATION
NO_GO_PERFORMANCE
GO_TOPOLOGY_LOCAL_TP2_WHOLE_MODEL_GATE
```

- [ ] **Step 2: Write reconciliation RED tests**

Require the assembler to reject:

- any epoch arm order other than A/B/B/A;
- anything other than ten baseline and ten candidate rows per workload;
- mismatched request-set digests;
- missing 127 token gaps per 128-token request;
- P99 computed from ten request summaries instead of raw token gaps;
- candidate hit counts not derivable from scheduler rows;
- nonzero short-chunk calls;
- a service-control row marked authoritative;
- missing/extra terminal files; and
- NaN, Inf, negative duration, or duplicate row identity.

Also reject a missing entry/pre-epoch/post-launch/terminal resource sample,
any task-created remote path outside the approved root, a reused or
overwritten attempt tag, an incomplete source archive, a foreign-process
action, a retained generation/lease/tensor/process-group/owned-process, or
remote/local verifier receipts whose semantic bytes differ.

- [ ] **Step 3: Run RED tests**

```bash
python -m pytest \
  tools/test_assemble_qwen38_topology_local_tp2_whole_model.py \
  -q
```

Expected: FAIL because the assembler does not exist.

- [ ] **Step 4: Implement metric reconstruction**

Use standard-library `statistics` and:

```python
def geometric_mean(values):
    values = tuple(float(value) for value in values)
    if not values or any(
        not math.isfinite(value) or value <= 0
        for value in values
    ):
        raise ValueError("geometric mean inputs must be finite and positive")
    return math.exp(
        sum(math.log(value) for value in values) / len(values)
    )


def nearest_rank_percentile(values, percentile):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile inputs must not be empty")
    rank = max(
        1,
        math.ceil((float(percentile) / 100.0) * len(ordered)),
    )
    return ordered[rank - 1]
```

Derive workload ratios, aggregate geometric ratios, raw-gap P95/P99, QPS,
output tokens/s, migration break-even, memory limits, and direction counts
without trusting producer summaries.

- [ ] **Step 5: Build the exact terminal bundle**

Write only the files listed in the design's artifact contract. Generate
`manifest.json` from sorted relative paths and then `manifest.sha256`.
`classification.json` contains every Boolean gate, absolute metric, ratio,
threshold, failed-gate name, and terminal classification.

- [ ] **Step 6: Run assembler tests**

```bash
python -m pytest \
  tools/test_assemble_qwen38_topology_local_tp2_whole_model.py \
  -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add -- \
  tools/assemble_qwen38_topology_local_tp2_whole_model.py \
  tools/test_assemble_qwen38_topology_local_tp2_whole_model.py
git diff --cached --check
git commit -m "feat(tp4): classify TP2 whole-model evidence"
```

---

### Task 9: Add the standard-library independent verifier

**Files:**

- Create: `tools/verify_qwen38_topology_local_tp2_whole_model.py`
- Create:
  `tools/test_verify_qwen38_topology_local_tp2_whole_model.py`

**Interfaces:**

- CLI:
  `python tools/verify_qwen38_topology_local_tp2_whole_model.py
  --bundle <path> --output <path>`
- Exit zero only for a semantically complete bundle, including a valid NO_GO
  or INCONCLUSIVE classification.

- [ ] **Step 1: Write import-isolation and mutation RED tests**

Assert the verifier source contains no imports of:

```text
tinyvllm
torch
tools.assemble_qwen38_topology_local_tp2_whole_model
tools.qwen38_topology_local_tp2_whole_model_worker
```

Generate one passing fixture, then mutate each of:

- source SHA;
- model revision;
- epoch arm;
- request token;
- raw token gap;
- candidate hit count;
- short-chunk count;
- collective byte count;
- state publication count;
- memory peak;
- cleanup ownership;
- service-control authority;
- classification; and
- manifest digest.

Every mutation must be rejected.

- [ ] **Step 2: Run RED tests**

```bash
python -m pytest \
  tools/test_verify_qwen38_topology_local_tp2_whole_model.py \
  -q
```

Expected: FAIL because the verifier does not exist.

- [ ] **Step 3: Implement independent parsing and classification**

Use imports only from:

```python
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
```

Reimplement schema checks, identities, percentile/geometric arithmetic,
thresholds, classification precedence, exact terminal inventory, and manifest
hashing. Do not call producer functions or trust producer aggregates.

Emit:

```python
{
    "schema_version":
        "qwen38.topology-local-tp2-whole-model-verification.v1",
    "classification": classification,
    "checks": checks,
    "semantic_digest": semantic_digest,
}
```

Do not include verifier location or timestamps so remote and local receipts
can be byte-identical.

- [ ] **Step 4: Run verifier and assembler tests**

```bash
python -m pytest \
  tools/test_verify_qwen38_topology_local_tp2_whole_model.py \
  tools/test_assemble_qwen38_topology_local_tp2_whole_model.py \
  -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -- \
  tools/verify_qwen38_topology_local_tp2_whole_model.py \
  tools/test_verify_qwen38_topology_local_tp2_whole_model.py
git diff --cached --check
git commit -m "feat(tp4): verify TP2 whole-model evidence"
```

---

### Task 10: Build immutable remote orchestration

**Files:**

- Create: `tools/run_qwen38_topology_local_tp2_whole_model.py`
- Create: `tools/test_run_qwen38_topology_local_tp2_whole_model.py`

**Interfaces:**

- `build_plan(*, attempt_tag: str, source_revision: str,
  gpu_inventory: tuple[dict, ...], topology: Mapping[str, object]) -> dict`
- `run_attempt(plan: Mapping[str, object]) -> dict`
- CLI supports `--dry-run`, `--check-only`,
  `--gpu-wait-timeout-s`, and `--gpu-poll-interval-s`.

- [ ] **Step 1: Write controller RED tests**

Reuse strict validators from the Stage-0 controller where their schemas are
identical. Test:

```python
assert MINIMUM_KERBEROS_LIFETIME_SECONDS == 1_800
assert APPROVED_REMOTE_ROOT == (
    "/data00/home/sitian/tinyllmforge-workspaces/"
    "command-timeline-20260818"
)
assert plan["campaign_epochs"] == [
    {"epoch": 0, "arm": "baseline", "workload_order": forward},
    {"epoch": 1, "arm": "candidate", "workload_order": reverse},
    {"epoch": 2, "arm": "candidate", "workload_order": forward},
    {"epoch": 3, "arm": "baseline", "workload_order": reverse},
]
```

Reject path escape, reused tag, fewer than four clean GPUs, non-optimal pair
matching, TTL 1,799, source drift, model drift, post-staging GPU drift,
foreign post-launch PID, missing epoch, and cleanup that touches an unowned
PID.

- [ ] **Step 2: Run RED tests**

```bash
python -m pytest \
  tools/test_run_qwen38_topology_local_tp2_whole_model.py \
  -q
```

Expected: FAIL because the controller does not exist.

- [ ] **Step 3: Implement the immutable plan**

The plan must freeze:

```python
{
    "schema_version":
        "qwen38.topology-local-tp2-whole-model-plan.v1",
    "source_revision": source_revision,
    "model_repository": "Qwen/Qwen3.8-27B",
    "model_revision":
        "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    "remote_root": APPROVED_REMOTE_ROOT,
    "attempt_tag": attempt_tag,
    "gpu_rank_mapping": rank_mapping,
    "pair_groups": pair_groups,
    "campaign_epochs": campaign_epochs,
    "workloads": workloads,
    "thresholds": frozen_thresholds,
}
```

Create remote files with an attempt-local temporary directory and atomic
rename. Never use remote `/tmp`.

- [ ] **Step 4: Implement launch, monitoring, and verification**

The controller must:

1. verify local source and Kerberos;
2. inventory topology and strict-clean GPUs;
3. select/freeze the best pair matching;
4. create the fresh remote attempt;
5. upload a source archive and hashes;
6. repeat admission immediately before every engine epoch and control
   launch;
7. launch the worker under the task-owned remote root;
8. capture post-launch identity and monitor owned PIDs and foreign GPU
   processes;
9. assemble the terminal bundle remotely;
10. run the frozen verifier remotely;
11. download only `final_bundle`;
12. run the same frozen verifier locally;
13. compare receipts byte-for-byte; and
14. verify the terminal manifest.

Transport exit 255 receives bounded `1, 2, 4` second retries. Non-255 failure
returns immediately.

The terminal controller receipt must enumerate every task-created remote
path and prove that each resolves below `APPROVED_REMOTE_ROOT`; it must also
record entry, every pre-epoch, every post-launch, and terminal GPU/process
inventories. Cleanup may signal only PIDs created by the immutable attempt.

- [ ] **Step 5: Run controller tests**

```bash
python -m pytest \
  tools/test_run_qwen38_topology_local_tp2_whole_model.py \
  tools/test_run_qwen38_topology_local_tp2_island.py \
  -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add -- \
  tools/run_qwen38_topology_local_tp2_whole_model.py \
  tools/test_run_qwen38_topology_local_tp2_whole_model.py
git diff --cached --check
git commit -m "feat(tp4): orchestrate TP2 whole-model gate"
```

---

### Task 11: Run the local integration verification

**Files:**

- Modify only files already named by Tasks 1-10 if failures expose defects.

- [ ] **Step 1: Run syntax checks**

```bash
python -m py_compile \
  tinyvllm/config.py \
  tinyvllm/engine/topology_local_tp2_island.py \
  tinyvllm/engine/qwen38_topology_local_tp2_state.py \
  tinyvllm/engine/qwen38_topology_local_tp2_runtime.py \
  tinyvllm/layers/qwen38_topology_local_tp2_linear_attention.py \
  tools/qwen38_topology_local_tp2_whole_model_worker.py \
  tools/run_qwen38_topology_local_tp2_whole_model.py \
  tools/assemble_qwen38_topology_local_tp2_whole_model.py \
  tools/verify_qwen38_topology_local_tp2_whole_model.py
```

Expected: no output and exit zero.

- [ ] **Step 2: Run all focused tests**

```bash
python -m pytest \
  tools/test_topology_local_tp2_island.py \
  tools/test_qwen38_topology_local_tp2_island_worker.py \
  tools/test_run_qwen38_topology_local_tp2_island.py \
  tools/test_assemble_qwen38_topology_local_tp2_island.py \
  tools/test_verify_qwen38_topology_local_tp2_island.py \
  tools/test_qwen38_topology_local_tp2_runtime.py \
  tools/test_qwen38_topology_local_tp2_state.py \
  tools/test_qwen38_topology_local_tp2_whole_model_worker.py \
  tools/test_run_qwen38_topology_local_tp2_whole_model.py \
  tools/test_assemble_qwen38_topology_local_tp2_whole_model.py \
  tools/test_verify_qwen38_topology_local_tp2_whole_model.py \
  tools/test_qwen38_model_runner_wiring.py \
  tools/test_model_runner_spec_verify.py \
  -q
```

Expected: all tests pass.

- [ ] **Step 3: Run adjacent Qwen3.8 suites**

```bash
python -m pytest \
  tools/test_qwen38_checkpoint_adoption.py \
  tools/test_qwen38_model_manifest.py \
  tools/test_qwen38_text_adopter.py \
  tools/test_qwen38_tp_correctness.py \
  tools/test_run_qwen38_tp_correctness.py \
  tools/test_qwen38_tp4_communication_profile_worker.py \
  tools/test_run_qwen38_tp4_communication_profile.py \
  tools/test_assemble_qwen38_tp4_communication_profile.py \
  tools/test_verify_qwen38_tp4_communication_profile.py \
  -q
```

Expected: all tests pass.

- [ ] **Step 4: Verify no forbidden production-file mutation**

```bash
git diff 35895293b98951816153124aaa86ef7d7a5bd19a -- \
  tinyvllm/layers/linear.py
```

Expected: no output.

- [ ] **Step 5: Verify repository hygiene**

```bash
git diff --check
git status --short --untracked-files=no
```

Expected: no whitespace errors; only intended tracked files are modified.

- [ ] **Step 6: Commit any bounded verification fixes**

If and only if Tasks 11.1-11.5 required changes:

```bash
git add -- <exact-files-fixed-in-task-11>
git diff --cached --check
git commit -m "fix(tp4): close whole-model gate gaps"
```

Do not create an empty commit.

---

### Task 12: Run diagnostic and formal four-GPU campaigns

**Files:**

- Create remote immutable attempt directories only under the approved root.
- Download compact bundles only under
  `artifacts/qwen38_topology_local_tp2_whole_model/`.

- [ ] **Step 1: Push and verify the implementation source**

```bash
git push origin feat/kv-sparse-attention
local_sha=$(git rev-parse HEAD)
remote_sha=$(
  git ls-remote origin refs/heads/feat/kv-sparse-attention |
  awk '{print $1}'
)
test "$local_sha" = "$remote_sha"
```

Expected: exact SHA equality.

- [ ] **Step 2: Run check-only preflight**

```bash
KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian \
python tools/run_qwen38_topology_local_tp2_whole_model.py \
  --attempt-tag \
  20260908-qwen38-topology-local-tp2-whole-model-diagnostic-r1 \
  --check-only
```

Expected: source/model/storage/Kerberos checks pass or a precise
`BLOCKED_ADMISSION` result is written without creating a GPU worker.

- [ ] **Step 3: Run one fresh diagnostic**

```bash
KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian \
python tools/run_qwen38_topology_local_tp2_whole_model.py \
  --attempt-tag \
  20260908-qwen38-topology-local-tp2-whole-model-diagnostic-r1 \
  --gpu-wait-timeout-s 0 \
  --gpu-poll-interval-s 30
```

If the tag was created by any earlier command, increment the suffix before
launch. Never reuse the tag.

Expected: a terminal diagnostic bundle. Diagnose any failure from artifacts;
change source only for a demonstrated cause.

- [ ] **Step 4: Verify and commit diagnostic fixes**

For each source correction:

```bash
python -m pytest <exact-focused-test-files> -q
git add -- <exact-changed-files>
git diff --cached --check
git commit -m "fix(tp4): <measured diagnostic cause>"
git push origin feat/kv-sparse-attention
```

Every rerun uses a fresh diagnostic tag.

- [ ] **Step 5: Run the formal immutable attempt**

Only after one complete diagnostic passes:

```bash
KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian \
python tools/run_qwen38_topology_local_tp2_whole_model.py \
  --attempt-tag \
  20260908-qwen38-topology-local-tp2-whole-model-r1 \
  --gpu-wait-timeout-s 0 \
  --gpu-poll-interval-s 30
```

Increment `r1` if that tag already exists. The formal run uses the exact
pushed source revision that passed the final local suite.

- [ ] **Step 6: Independently verify terminal evidence**

```bash
python tools/verify_qwen38_topology_local_tp2_whole_model.py \
  --bundle \
  artifacts/qwen38_topology_local_tp2_whole_model/<formal-tag>/final_bundle \
  --output \
  artifacts/qwen38_topology_local_tp2_whole_model/<formal-tag>/\
final_bundle/local_independent_verification.json
```

Then verify every manifest hash and compare remote/local verifier receipts.

Expected: one terminal classification with complete semantic coverage. A
NO_GO or INCONCLUSIVE result is still a valid completed experiment.

---

### Task 13: Reconcile audit, handoff, commit, and push

**Files:**

- Modify:
  `docs/superpowers/audits/2026-08-16-phase1-completion-audit.md`
- Modify: `AGENT_HANDOFF_STATE.md`
- Add: compact formal `final_bundle` files only

- [ ] **Step 1: Append the completion audit at true EOF**

Add:

- source and model SHA;
- diagnostic and formal tags;
- exact GPU UUID/rank/pair mapping;
- all correctness and candidate-hit counts;
- baseline/candidate absolute TPOT, P99, TTFT, QPS, and throughput;
- migration and memory cost;
- TP2-x2 service-control result;
- terminal classification and failed gates;
- remote/local verifier equality;
- exact claim boundary; and
- the prompt-to-artifact checklist with every row marked by artifact path.

- [ ] **Step 2: Append the terminal handoff at true EOF**

Record:

- current branch and HEAD;
- tracked worktree status;
- tests and exact commands;
- artifact locations;
- remote process and cleanup state;
- formal classification;
- what may and may not be claimed; and
- the next authorized stage only if the formal result is GO.

- [ ] **Step 3: Run the completion audit**

Map every requirement in the design spec to a concrete file and verifier
field. Explicitly mark missing evidence as incomplete; do not infer it from
the producer result or manifest.

Run:

```bash
git diff --check
python -m pytest \
  tools/test_qwen38_topology_local_tp2_runtime.py \
  tools/test_qwen38_topology_local_tp2_state.py \
  tools/test_qwen38_topology_local_tp2_whole_model_worker.py \
  tools/test_run_qwen38_topology_local_tp2_whole_model.py \
  tools/test_assemble_qwen38_topology_local_tp2_whole_model.py \
  tools/test_verify_qwen38_topology_local_tp2_whole_model.py \
  -q
```

Expected: PASS.

- [ ] **Step 4: Exact-stage final evidence and documentation**

```bash
git add -- \
  docs/superpowers/audits/2026-08-16-phase1-completion-audit.md \
  AGENT_HANDOFF_STATE.md \
  artifacts/qwen38_topology_local_tp2_whole_model/<formal-tag>/final_bundle
git diff --cached --check
git diff --cached --name-only
```

Confirm that no historical artifact or unrelated file is staged.

- [ ] **Step 5: Commit, push, and verify SHA**

```bash
git commit -m "docs(tp4): record TP2 whole-model gate"
git push origin feat/kv-sparse-attention
local_sha=$(git rev-parse HEAD)
remote_sha=$(
  git ls-remote origin refs/heads/feat/kv-sparse-attention |
  awk '{print $1}'
)
test "$local_sha" = "$remote_sha"
git status --short --untracked-files=no
```

Expected: local and remote SHA match; tracked worktree is clean.

---

## Plan self-review record

The plan was checked section-by-section against the approved design before
execution:

| Design requirement | Implementing task(s) | Verification authority |
| --- | --- | --- |
| Decision and fixed-cohort whole-model scope | 5-7 | real `LLMEngine` request/scheduler rows |
| Stage-0 reuse without projecting its timings | 2, 4, 7 | Stage-0 regression tests plus fresh whole-model rows |
| Default-off fail-closed feature boundary | 1, 5 | config and installation rejection tests |
| Deterministic TP2 pair groups and topology | 1, 10 | pair-context tests and frozen topology manifests |
| 48 linear-attention candidate layers | 2, 4, 5 | installation inventory and derived hit counts |
| Bitwise cross-pair BF16 output identity | 4, 7 | untimed per-layer device digests |
| Candidate weight ownership and TP4 release | 2, 5, 7 | weak-reference release plus memory rows |
| Generation-sealed TP4-to-TP2 state lifecycle | 3, 5, 6 | migration/publication/rollback/release tests and rows |
| Real candidate-hit authority | 4, 6-9 | raw counters independently joined to scheduler/model rows |
| Frozen P0/P1/Q0/Q1/Q2 workloads | 7, 8, 10 | workload and campaign manifests |
| Four-epoch A/B/B/A timing design | 7, 8, 10 | epoch manifest and exact row-count reconciliation |
| Non-authoritative TP2-x2 service control | 7-10 | labeled service-control rows and verifier rejection tests |
| Untimed and timed correctness | 7-9 | correctness/request rows and mutation tests |
| TPOT, tail, TTFT, QPS, throughput, and cost metrics | 7-9 | raw timestamps, gaps, memory, migration, and resource rows |
| Frozen GO thresholds and classification precedence | 8, 9 | one-field mutation matrix in producer and verifier tests |
| Kerberos/GPU/storage/process safety | 10, 12 | admission and controller receipts |
| Exact terminal artifact contract and immutable manifest | 8-10, 12 | exact-inventory and digest mutation tests |
| Prompt-to-artifact checklist and claim boundary | 13 | EOF audit/handoff entries tied to final paths |

The prompt-to-artifact checklist in design section 14 is covered as follows:

| Checklist row | Planned artifact/evidence |
| --- | --- |
| Real Qwen3.8-27B request path | `model_manifest.json`, `request_rows.jsonl`, `scheduler_step_rows.jsonl` |
| Frozen checkpoint | `model_manifest.json` |
| Baseline versus candidate | `campaign_epoch_manifest.json`, `request_rows.jsonl` |
| Exact greedy output identity | `correctness_rows.jsonl`, `request_rows.jsonl` |
| Whole-model execution | `feature_contract.json`, `candidate_hit_rows.jsonl` |
| Candidate actually exercised | `candidate_hit_rows.jsonl`, `migration_rows.jsonl`, `collective_rows.jsonl` |
| TPOT benefit and P99 protection | raw token timestamps/gaps in `request_rows.jsonl`, recomputed in `classification.json` |
| TTFT protection | admission/first-token timestamps in `request_rows.jsonl` |
| QPS and throughput | cohort timestamps in `request_rows.jsonl` |
| Migration cost | `migration_rows.jsonl` |
| Memory cost | `weight_layout_manifest.json`, `state_layout_manifest.json`, `memory_rows.jsonl` |
| TP2 pair locality | `gpu_topology.json`, `gpu_rank_manifest.json`, `pair_group_manifest.json` |
| Full-attention unchanged | `collective_rows.jsonl`, `candidate_hit_rows.jsonl` |
| No hidden TP4 linear collective | `collective_rows.jsonl`, `candidate_hit_rows.jsonl` |
| Fixed-cohort scope | `workload_manifest.json`, `scheduler_step_rows.jsonl` |
| TP2 x2 service control | `service_control_rows.jsonl` |
| Strict-clean GPUs | `resource_rows.jsonl` |
| No foreign-process action | `resource_rows.jsonl`, `cleanup.json` |
| Approved remote storage | controller path audit recorded in `cleanup.json` |
| Immutable attempts | `source_manifest.json`, controller no-overwrite receipt, `manifest.json` |
| Dual verification | `classification.json`, `remote_independent_verification.json`, `local_independent_verification.json` |
| Compact local storage | exact `final_bundle` inventory in `manifest.json` |
| Claim boundary | `report.md`, EOF audit, and `AGENT_HANDOFF_STATE.md` |

No design section remains without an implementation task or independent
verification path. Dynamic arrivals, recovery, preemption, prefix restore,
speculative rollback, production-default enablement, and cross-engine
comparisons remain explicit non-goals.

---

## Completion criteria

The implementation is complete only when all of the following are true:

- Tasks 1-11 are implemented with RED then GREEN evidence and focused
  commits.
- The implementation source is pushed and remote/local branch SHA matches.
- One fresh diagnostic attempt is terminal.
- One fresh formal attempt is terminal.
- The formal bundle contains the exact artifact inventory.
- Producer, remote independent verifier, and local independent verifier
  reconstruct the same classification.
- Every manifest hash matches.
- No owned GPU process or remote temporary remains.
- Audit and handoff entries are appended at the true EOF.
- Benefit and cost are both reported.
- The claim remains fixed-cohort whole-model only.
- Production-default enablement remains prohibited.
