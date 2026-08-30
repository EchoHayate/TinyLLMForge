# Qwen3.8 TP4 Peer-Reduction and Residual-Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Qualify and, only after a real four-GPU microgate pass, integrate a
default-disabled fixed-slot peer-memory reduction that replaces selected
Qwen3.8 TP4 attention-output NCCL AllReduces and fuses the baseline BF16 cast
plus residual addition.

**Architecture:** First repair the stale 130-collective evidence contract to
the implemented 66-site graph. Then build a dependency-light policy,
classifier, controller, and independent verifier around a small
`torch.utils.cpp_extension` CUDA IPC module. The CUDA candidate uses
two-entry per-layer FP32 slot rings and generation flags; runtime wiring is a
separate conditional task that is forbidden unless the isolated real-shape
microgate passes its frozen thresholds.

**Tech Stack:** Python 3.12, PyTorch distributed/NCCL,
`torch.utils.cpp_extension`, CUDA Runtime API, CUDA C++, pytest, JSON/JSONL,
SHA-256 manifests, four A100 GPUs.

## Global Constraints

- Work only in `/Users/bytedance/dev/TinyLLMForge`; the Desktop path is only
  a symlink to this checkout.
- Stay on `feat/kv-sparse-attention` and push only to
  `origin/feat/kv-sparse-attention`.
- Do not create a worktree or use subagents; execute inline.
- Use meaningful RED, minimal implementation, and GREEN for every code task.
- Stage exact paths only. Never use broad `git add`, `git reset`, `git clean`,
  or unrelated formatting.
- Commit with `git -c core.hooksPath=/dev/null commit` and exactly one
  `Co-authored-by: TRAE CLI <noreply@bytedance.com>` trailer.
- Put every remote task file, build cache, log, and artifact below
  `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/`.
- Do not write remote task data to `/`, `/tmp`, an old checkout, or the model
  cache.
- Do not run `kinit` or `krenew`; Kerberos refresh belongs to the user.
- Do not terminate, adopt, or clean external GPU processes.
- Require four strict-clean GPUs immediately before every real launch.
- Keep large traces remote and download only compact evidence.
- Report benefit and cost together. A microgate, profiler result, or
  theoretical ceiling is not an end-to-end speedup.
- Never revive the superseded R8/R16 AllGather plus full-GEMM route.
- Preserve immutable r10, r620, r621, r630, and r631 artifacts.

---

### Task 1: Repair the Qwen3.8 collective contract from 130 to 66

**Files:**

- Modify: `tools/qwen38_collective_reduction.py`
- Modify: `tools/qwen38_tp4_collective_reduction_worker.py`
- Modify: `tools/assemble_qwen38_tp4_collective_reduction.py`
- Modify: `tools/test_qwen38_collective_reduction.py`
- Modify: `tools/test_qwen38_tp4_collective_reduction_worker.py`
- Modify: `tools/test_assemble_qwen38_tp4_collective_reduction.py`
- Modify: `tools/test_verify_qwen38_tp4_collective_reduction.py`

**Interfaces:**

- Produces:
  `build_qwen38_static_collective_catalog(text_profile,
  tensor_parallel_size=4, layer_roles=...) -> tuple[dict, ...]`.
- Produces the exact sequence `embedding + 64 attention + broadcast`.
- Removes every expected MLP-output collective from the Qwen3.8 catalog.
- Keeps immutable old artifacts untouched; only code and fixtures change.

- [ ] **Step 1: Write the failing 66-site catalog tests**

Replace the old count assertions with:

```python
def test_qwen38_catalog_contains_exactly_66_decode_sites():
    catalog = _catalog()

    assert len(catalog) == 66
    assert catalog[0]["site_id"] == "embedding.input"
    assert sum(
        row["site_role"] == "row_parallel_attention_output"
        for row in catalog
    ) == 64
    assert not any(
        row["layer_role"] == "mlp"
        for row in catalog
    )
    assert catalog[-1]["site_id"] == "sampling.greedy_token"


def test_qwen38_catalog_preserves_checkpoint_layer_roles():
    roles = tuple(
        "full_attention" if index % 4 == 3 else "linear_attention"
        for index in range(64)
    )
    catalog = build_qwen38_static_collective_catalog(
        _profile(),
        tensor_parallel_size=4,
        layer_roles=roles,
    )

    attention = catalog[1:-1]
    assert [row["layer_role"] for row in attention] == list(roles)
    assert all(
        row["local_tensor_dtype"] == "torch.float32"
        for row in attention
    )
```

Update census fixtures so each decode step contains 66 entries and an FP32
attention payload of `active_tokens * 5120 * 4` bytes.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
/opt/homebrew/bin/python3.12 -m pytest \
  tools/test_qwen38_collective_reduction.py \
  tools/test_qwen38_tp4_collective_reduction_worker.py \
  tools/test_assemble_qwen38_tp4_collective_reduction.py \
  tools/test_verify_qwen38_tp4_collective_reduction.py -q
```

Expected: failures mention `130`, MLP rows, missing `layer_roles`, or BF16
attention payload assumptions.

- [ ] **Step 3: Implement the 66-site catalog**

Use:

```python
EXPECTED_DECODE_COLLECTIVE_COUNT = 66


def build_qwen38_static_collective_catalog(
    text_profile,
    *,
    tensor_parallel_size: int,
    layer_roles=None,
):
    # Existing frozen profile and TP validation remains.
    if layer_roles is None:
        layer_roles = tuple(
            "full_attention" if index % 4 == 3 else "linear_attention"
            for index in range(EXPECTED_PROFILE["num_hidden_layers"])
        )
    if (
        not isinstance(layer_roles, (list, tuple))
        or len(layer_roles) != EXPECTED_PROFILE["num_hidden_layers"]
        or any(
            role not in ("linear_attention", "full_attention")
            for role in layer_roles
        )
    ):
        raise ValueError("layer_roles must describe all 64 attention layers")

    rows = [_embedding_catalog_row()]
    for layer_index, layer_role in enumerate(layer_roles):
        rows.append(_catalog_row(
            site_id=f"layer.{layer_index:03d}.attention.output",
            module_path=(
                f"model.layers.{layer_index}."
                + (
                    "full_attention.output_projection"
                    if layer_role == "full_attention"
                    else "linear_attention.out_proj"
                )
            ),
            layer_index=layer_index,
            layer_role=layer_role,
            operation_name="row_parallel_all_reduce",
            collective_kind="all_reduce",
            local_tensor_shape_formula="[active_tokens, hidden_size]",
            local_tensor_dtype="torch.float32",
            producer="row-parallel attention output projection",
            first_consumer="attention residual addition",
            requires_replicated_result=True,
            packing_window=None,
            elimination_precondition=(
                "qualified peer-reduction residual-fusion group"
            ),
            classification="MATERIALIZATION_ALTERNATIVE",
            site_role="row_parallel_attention_output",
        ))
    rows.append(_greedy_broadcast_catalog_row())
    if len(rows) != EXPECTED_DECODE_COLLECTIVE_COUNT:
        raise AssertionError("Qwen3.8 collective catalog size mismatch")
    return tuple(rows)
```

Factor the existing embedding and broadcast row construction into private
helpers without changing their fields.

- [ ] **Step 4: Update worker, assembler, and verifier invariants**

Require:

```python
EXPECTED_COLLECTIVE_COUNT = 66
EXPECTED_ATTENTION_COLLECTIVE_COUNT = 64
EXPECTED_MLP_COLLECTIVE_COUNT = 0
```

The assembler and verifier must reject any catalog or dynamic step that has:

```python
len(sequence) != 66
attention_count != 64
mlp_count != 0
sequence[0] != "embedding.input"
sequence[-1] != "sampling.greedy_token"
```

- [ ] **Step 5: Run GREEN and syntax checks**

Run the four pytest files from Step 2, then:

```bash
/opt/homebrew/bin/python3.12 -m py_compile \
  tools/qwen38_collective_reduction.py \
  tools/qwen38_tp4_collective_reduction_worker.py \
  tools/assemble_qwen38_tp4_collective_reduction.py
git diff --check -- \
  tools/qwen38_collective_reduction.py \
  tools/qwen38_tp4_collective_reduction_worker.py \
  tools/assemble_qwen38_tp4_collective_reduction.py \
  tools/test_qwen38_collective_reduction.py \
  tools/test_qwen38_tp4_collective_reduction_worker.py \
  tools/test_assemble_qwen38_tp4_collective_reduction.py \
  tools/test_verify_qwen38_tp4_collective_reduction.py
```

Expected: all focused tests pass, compilation passes, and diff check is
empty.

- [ ] **Step 6: Commit and push**

```bash
git add -- \
  tools/qwen38_collective_reduction.py \
  tools/qwen38_tp4_collective_reduction_worker.py \
  tools/assemble_qwen38_tp4_collective_reduction.py \
  tools/test_qwen38_collective_reduction.py \
  tools/test_qwen38_tp4_collective_reduction_worker.py \
  tools/test_assemble_qwen38_tp4_collective_reduction.py \
  tools/test_verify_qwen38_tp4_collective_reduction.py
git -c core.hooksPath=/dev/null commit \
  -m "fix(profiler): correct Qwen3.8 decode collective graph" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

### Task 2: Add the dependency-light peer-reduction contract and classifier

**Files:**

- Create: `tools/qwen38_tp4_peer_reduction.py`
- Create: `tools/test_qwen38_tp4_peer_reduction.py`

**Interfaces:**

- Produces `PeerReductionPolicy`.
- Produces `validate_peer_topology(rows)`.
- Produces `classify_peer_microgate(rows, cleanup, memory)`.
- Contains no torch import so local policy tests run on macOS.

- [ ] **Step 1: Write failing policy and classification tests**

Cover exact validation for:

```python
def test_policy_freezes_supported_shape_and_ring():
    policy = PeerReductionPolicy()
    assert policy.world_size == 4
    assert policy.hidden_size == 5120
    assert policy.max_active_tokens == 8
    assert policy.slot_ring_size == 2
    assert policy.maximum_allocated_delta_bytes == 48 * 1024 * 1024


def test_topology_requires_all_twelve_directed_peer_edges():
    rows = [
        {"source_rank": source, "destination_rank": destination,
         "can_access": True, "ipc_roundtrip": True}
        for source in range(4)
        for destination in range(4)
        if source != destination
    ]
    assert validate_peer_topology(rows)["classification"] == "PASS"
    rows[-1]["can_access"] = False
    with pytest.raises(ValueError, match="peer topology"):
        validate_peer_topology(rows)


def test_microgate_requires_benefit_cost_correctness_and_cleanup():
    result = classify_peer_microgate(
        rows=_passing_microgate_rows(),
        cleanup={"classification": "CLEAN"},
        memory={"maximum_allocated_delta_bytes": 48 * 1024 * 1024},
    )
    assert result["classification"] == "PASS"


@pytest.mark.parametrize(
    ("mutation", "expected"),
    (
        ("correctness", "NO_GO_CORRECTNESS"),
        ("median", "NO_GO_MICROGATE"),
        ("p99", "NO_GO_MICROGATE"),
        ("timeout", "NO_GO_MICROGATE"),
        ("memory", "NO_GO_MEMORY"),
        ("cleanup", "INCONCLUSIVE_EVIDENCE"),
    ),
)
def test_microgate_fails_closed(mutation, expected):
    rows, cleanup, memory = mutated_case(mutation)
    assert classify_peer_microgate(
        rows=rows,
        cleanup=cleanup,
        memory=memory,
    )["classification"] == expected
```

- [ ] **Step 2: Run the new tests and verify RED**

```bash
/opt/homebrew/bin/python3.12 -m pytest \
  tools/test_qwen38_tp4_peer_reduction.py -q
```

Expected: import failure because
`tools/qwen38_tp4_peer_reduction.py` does not exist.

- [ ] **Step 3: Implement exact policy and fail-closed precedence**

Implement immutable constants:

```python
WORLD_SIZE = 4
HIDDEN_SIZE = 5120
MAX_ACTIVE_TOKENS = 8
SLOT_RING_SIZE = 2
MAX_ALLOCATED_DELTA_BYTES = 48 * 1024 * 1024
MIN_MEDIAN_SPEEDUP = 0.10
MAX_TOKENS8_REGRESSION = 0.02
MAX_P99_REGRESSION = 0.03
CROSS_RANK_ATOL = 2e-4
CROSS_RANK_RTOL = 2e-4
BASELINE_ATOL = 2e-2
BASELINE_RTOL = 2e-3
```

Classification precedence must be:

```text
NO_GO_CORRECTNESS
NO_GO_MEMORY
NO_GO_MICROGATE
INCONCLUSIVE_EVIDENCE
PASS
```

Reject duplicate or missing active-token groups, fewer than 200 paired
measurements, non-finite metrics, missing four-rank agreement, any timeout,
or non-clean cleanup.

- [ ] **Step 4: Run GREEN and commit**

```bash
/opt/homebrew/bin/python3.12 -m pytest \
  tools/test_qwen38_tp4_peer_reduction.py -q
/opt/homebrew/bin/python3.12 -m py_compile \
  tools/qwen38_tp4_peer_reduction.py
git diff --check -- \
  tools/qwen38_tp4_peer_reduction.py \
  tools/test_qwen38_tp4_peer_reduction.py
git add -- \
  tools/qwen38_tp4_peer_reduction.py \
  tools/test_qwen38_tp4_peer_reduction.py
git -c core.hooksPath=/dev/null commit \
  -m "feat(runtime): define TP4 peer reduction gate" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

### Task 3: Add the CUDA IPC extension and Python lifecycle owner

**Files:**

- Create: `tinyvllm/engine/tp4_peer_reduction.py`
- Create: `tinyvllm/engine/csrc/tp4_peer_reduction.cpp`
- Create: `tinyvllm/engine/csrc/tp4_peer_reduction_kernel.cu`
- Create: `tools/test_tp4_peer_reduction_runtime.py`

**Interfaces:**

- Produces `TP4PeerReductionGroup.create(...)`.
- Produces `TP4PeerReductionGroup.reduce_add_residual(...)`.
- Produces idempotent `TP4PeerReductionGroup.close()`.
- Extension loading uses `TORCH_EXTENSIONS_DIR`; no build output may enter
  the repository or remote `/tmp`.

- [ ] **Step 1: Write dependency-light lifecycle RED tests**

Load the Python module by file path with injected fake torch/distributed and
fake extension objects. Test:

```python
def test_create_rejects_non_tp4():
    with pytest.raises(ValueError, match="world_size must be 4"):
        TP4PeerReductionGroup.create(
            rank=0,
            world_size=2,
            device=FakeDevice("cuda", 0),
            layer_count=16,
            max_active_tokens=8,
            hidden_size=5120,
        )


def test_reduce_rejects_wrong_generation_shape_and_dtype():
    group = ready_group()
    with pytest.raises(ValueError, match="generation"):
        group.reduce_add_residual(
            layer_index=0,
            generation=-1,
            local_partial=fake_fp32((1, 5120)),
            residual=fake_bf16((1, 5120)),
        )
    with pytest.raises(ValueError, match="local_partial"):
        group.reduce_add_residual(
            layer_index=0,
            generation=1,
            local_partial=fake_fp32((1, 4096)),
            residual=fake_bf16((1, 5120)),
        )


def test_close_releases_imports_then_owned_allocations_once():
    group, extension = ready_group_with_extension()
    group.close()
    group.close()
    assert extension.calls == [
        ("close_mapping", 0),
        ("close_mapping", 1),
        ("close_mapping", 2),
        ("release_owned",),
    ]
```

- [ ] **Step 2: Run the lifecycle tests and verify RED**

```bash
/opt/homebrew/bin/python3.12 -m pytest \
  tools/test_tp4_peer_reduction_runtime.py -q
```

Expected: import failure because the lifecycle module is absent.

- [ ] **Step 3: Implement the Python owner**

The owner must:

- lazily load the extension only after CUDA/TP4 admission;
- allocate `[layer_count, 2, max_active_tokens, hidden_size]` FP32 slots;
- allocate `[layer_count, 2]` uint64 generation flags;
- exchange fixed-size handle bytes with `dist.all_gather_object()` only
  during `create()`;
- open each peer allocation once;
- keep an explicit state machine `NEW -> READY -> POISONED -> CLOSED`;
- select `slot_index = generation % 2`;
- call `torch.mm(..., out=local_slot_slice)` under inference mode;
- invoke publish then fused reduction on the current stream;
- reject active-token counts outside `1..8`; and
- never silently use the candidate after poisoning.

Expose a constructor injection point for the fake extension and distributed
adapter used by tests.

- [ ] **Step 4: Implement the extension lifecycle**

The C++ binding validates CUDA tensors and exposes opaque capsules. The CUDA
implementation must:

- call `cudaIpcGetMemHandle` for owned slot/flag allocations;
- call `cudaIpcOpenMemHandle(..., cudaIpcMemLazyEnablePeerAccess)` once per
  imported allocation;
- call `cudaIpcCloseMemHandle` exactly once per imported allocation;
- write the selected generation only after `__threadfence_system()`;
- compare peer flags against the exact expected generation;
- terminate the wait after `timeout_clocks`;
- set one device status value on timeout;
- sum peer FP32 values in rank order;
- convert the sum to BF16;
- add the BF16 residual using BF16-equivalent rounding; and
- write BF16 output without dynamic allocation.

Compilation flags are:

```python
extra_cflags=["-O3"]
extra_cuda_cflags=["-O3", "-lineinfo"]
with_cuda=True
```

NVCC defaults to precise division/square-root behavior when
`--use_fast_math` is absent. The spelling `--use_fast_math=false` is invalid
and must not be used.

- [ ] **Step 5: Run local GREEN and source-contract checks**

```bash
/opt/homebrew/bin/python3.12 -m pytest \
  tools/test_tp4_peer_reduction_runtime.py -q
/opt/homebrew/bin/python3.12 -m py_compile \
  tinyvllm/engine/tp4_peer_reduction.py
rg -n "/tmp|mkdtemp|TemporaryDirectory|load_inline" \
  tinyvllm/engine/tp4_peer_reduction.py \
  tinyvllm/engine/csrc/tp4_peer_reduction.cpp \
  tinyvllm/engine/csrc/tp4_peer_reduction_kernel.cu
git diff --check -- \
  tinyvllm/engine/tp4_peer_reduction.py \
  tinyvllm/engine/csrc/tp4_peer_reduction.cpp \
  tinyvllm/engine/csrc/tp4_peer_reduction_kernel.cu \
  tools/test_tp4_peer_reduction_runtime.py
```

Expected: pytest and compilation pass; the path search has no matches.

- [ ] **Step 6: Commit and push**

```bash
git add -- \
  tinyvllm/engine/tp4_peer_reduction.py \
  tinyvllm/engine/csrc/tp4_peer_reduction.cpp \
  tinyvllm/engine/csrc/tp4_peer_reduction_kernel.cu \
  tools/test_tp4_peer_reduction_runtime.py
git -c core.hooksPath=/dev/null commit \
  -m "feat(runtime): add TP4 peer reduction primitive" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

### Task 4: Build the four-GPU topology and microgate worker

**Files:**

- Create: `tools/qwen38_tp4_peer_reduction_microgate_worker.py`
- Create: `tools/test_qwen38_tp4_peer_reduction_microgate_worker.py`

**Interfaces:**

- Consumes `TP4PeerReductionGroup` and `PeerReductionPolicy`.
- Produces `peer_access_matrix.json`, `ipc_roundtrip.jsonl`,
  `microgate_rows.jsonl`, `memory_summary.json`, and `cleanup.json`.
- Does not import or construct the Qwen3.8 model.

- [ ] **Step 1: Write RED tests for workload and artifact construction**

Test exact workload order `(1, 4, 8)`, two warmups, 200 alternating measured
pairs, four rank rows per pair, immutable seeds, finite metrics, and rejection
of missing timeout/status evidence.

The worker argument parser must require:

```text
--attempt
--source-revision
--output-dir
--rank
--world-size
--dist-port
```

- [ ] **Step 2: Run RED**

```bash
/opt/homebrew/bin/python3.12 -m pytest \
  tools/test_qwen38_tp4_peer_reduction_microgate_worker.py -q
```

Expected: missing-module failure.

- [ ] **Step 3: Implement the worker**

For each active-token count:

1. seed on rank 0 and broadcast frozen BF16 input, FP32 local weights, and
   BF16 residual;
2. warm both arms twice;
3. alternate `AB` and `BA` order by pair index;
4. record complete transaction CUDA events without synchronizing inside the
   transaction;
5. synchronize only after both arm launches for the pair;
6. call `candidate_group.check_status()` after that synchronization;
7. record host submission and CUDA durations;
8. gather compact correctness digests and max errors;
9. record allocator deltas and timeout status; and
10. close the peer group in `finally`.

The baseline is exactly:

```python
partial = torch.nn.functional.linear(
    x.float(),
    local_weight,
)
torch.distributed.all_reduce(partial)
baseline = partial.to(torch.bfloat16) + residual
```

- [ ] **Step 4: Run GREEN, compile, and commit**

```bash
/opt/homebrew/bin/python3.12 -m pytest \
  tools/test_qwen38_tp4_peer_reduction_microgate_worker.py -q
/opt/homebrew/bin/python3.12 -m py_compile \
  tools/qwen38_tp4_peer_reduction_microgate_worker.py
git diff --check -- \
  tools/qwen38_tp4_peer_reduction_microgate_worker.py \
  tools/test_qwen38_tp4_peer_reduction_microgate_worker.py
git add -- \
  tools/qwen38_tp4_peer_reduction_microgate_worker.py \
  tools/test_qwen38_tp4_peer_reduction_microgate_worker.py
git -c core.hooksPath=/dev/null commit \
  -m "test(runtime): add TP4 peer reduction microgate" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

### Task 5: Add controller, assembler, and independent verifier

**Files:**

- Create: `tools/run_qwen38_tp4_peer_reduction_microgate.py`
- Create: `tools/assemble_qwen38_tp4_peer_reduction_microgate.py`
- Create: `tools/verify_qwen38_tp4_peer_reduction_microgate.py`
- Create: `tools/test_run_qwen38_tp4_peer_reduction_microgate.py`
- Create: `tools/test_assemble_qwen38_tp4_peer_reduction_microgate.py`
- Create: `tools/test_verify_qwen38_tp4_peer_reduction_microgate.py`

**Interfaces:**

- Reuses SSH, Kerberos, GPU inventory, topology, environment, source archive,
  supervisor, cleanup, and manifest helpers from the existing Qwen3.8
  collective-reduction tooling.
- Produces one immutable compact bundle and a byte-independent verifier
  result.

- [ ] **Step 1: Write RED controller tests**

Cover:

- expired Kerberos stops before remote write;
- all remote paths are descendants of the approved root;
- `TMPDIR`, `TORCH_EXTENSIONS_DIR`, CUDA cache, and logs are attempt-local;
- the attempt tag must be absent before launch;
- exactly four strict-clean GPUs are selected;
- a second strict-clean check occurs immediately before launch;
- SSH return code 255 is retried only within the fixed retry count; and
- no external process is terminated.

- [ ] **Step 2: Write RED assembler/verifier tests**

Build synthetic passing and failing bundles. Require:

```text
12 directed peer edges
3 active-token groups
>= 200 paired rows per group
4 ranks per pair
zero timeouts
cross-rank and baseline tolerances
<= 48 MiB allocated delta
CLEAN cleanup
manifest agreement
```

Mutate each field independently and assert the exact fail-closed
classification.

- [ ] **Step 3: Run RED**

```bash
/opt/homebrew/bin/python3.12 -m pytest \
  tools/test_run_qwen38_tp4_peer_reduction_microgate.py \
  tools/test_assemble_qwen38_tp4_peer_reduction_microgate.py \
  tools/test_verify_qwen38_tp4_peer_reduction_microgate.py -q
```

Expected: missing-module failures.

- [ ] **Step 4: Implement controller and evidence pipeline**

The controller stages a committed `git archive`, creates all runtime/cache
directories below the attempt root, launches four ranks through the existing
supervisor pattern, polls without signals, downloads only compact JSON/JSONL,
runs producer assembly, runs the independent verifier from frozen source,
compares both classifications, writes `manifest.sha256`, and performs three
read-only exact-tag process scans after cleanup.

The assembler delegates threshold logic to
`classify_peer_microgate(...)`. The verifier recomputes raw paired
percentiles and errors without trusting `microgate_summary.json`.

- [ ] **Step 5: Run GREEN and commit**

```bash
/opt/homebrew/bin/python3.12 -m pytest \
  tools/test_run_qwen38_tp4_peer_reduction_microgate.py \
  tools/test_assemble_qwen38_tp4_peer_reduction_microgate.py \
  tools/test_verify_qwen38_tp4_peer_reduction_microgate.py -q
/opt/homebrew/bin/python3.12 -m py_compile \
  tools/run_qwen38_tp4_peer_reduction_microgate.py \
  tools/assemble_qwen38_tp4_peer_reduction_microgate.py \
  tools/verify_qwen38_tp4_peer_reduction_microgate.py
git diff --check -- \
  tools/run_qwen38_tp4_peer_reduction_microgate.py \
  tools/assemble_qwen38_tp4_peer_reduction_microgate.py \
  tools/verify_qwen38_tp4_peer_reduction_microgate.py \
  tools/test_run_qwen38_tp4_peer_reduction_microgate.py \
  tools/test_assemble_qwen38_tp4_peer_reduction_microgate.py \
  tools/test_verify_qwen38_tp4_peer_reduction_microgate.py
git add -- \
  tools/run_qwen38_tp4_peer_reduction_microgate.py \
  tools/assemble_qwen38_tp4_peer_reduction_microgate.py \
  tools/verify_qwen38_tp4_peer_reduction_microgate.py \
  tools/test_run_qwen38_tp4_peer_reduction_microgate.py \
  tools/test_assemble_qwen38_tp4_peer_reduction_microgate.py \
  tools/test_verify_qwen38_tp4_peer_reduction_microgate.py
git -c core.hooksPath=/dev/null commit \
  -m "feat(runtime): automate TP4 peer reduction qualification" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

### Task 6: Run the real four-GPU microgate and enforce the stop rule

**Files:**

- Create after execution:
  `artifacts/qwen38_tp4_peer_reduction/20260830-qwen38-tp4-peer-reduction-r1/controller/`
- Create after execution:
  `artifacts/qwen38_tp4_peer_reduction/20260830-qwen38-tp4-peer-reduction-r1/final_bundle/`
- Create after execution:
  `docs/superpowers/audits/2026-08-30-qwen38-tp4-peer-reduction-microgate-audit.md`

**Interfaces:**

- Consumes a fresh valid Kerberos TGT and four strict-clean A100 GPUs.
- Produces one terminal classification.
- Runtime integration is forbidden unless the classification is `PASS`.

- [ ] **Step 1: Perform read-only preflight**

Run:

```bash
klist
ssh -o BatchMode=yes -o ConnectTimeout=8 sitian@10.232.195.203 \
  'hostname; df -h /data00/home/sitian; nvidia-smi topo -m; nvidia-smi'
```

Expected: valid TGT, approved storage available, one host, no MIG, complete
selected-GPU peer access, and four strict-clean GPUs.

- [ ] **Step 2: Launch one fresh immutable attempt**

Use:

```bash
/opt/homebrew/bin/python3.12 \
  tools/run_qwen38_tp4_peer_reduction_microgate.py \
  --attempt 20260830-qwen38-tp4-peer-reduction-r1 \
  --remote-root \
  /data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818 \
  --ssh-target sitian@10.232.195.203
```

The controller, not the user, waits for GPU admission and launches
immediately when four strict-clean GPUs are available.

- [ ] **Step 3: Verify independently**

Run the verifier from the committed source revision recorded in the bundle
and require byte-identical classification inputs plus manifest agreement.

- [ ] **Step 4: Write the audit**

The audit must state:

- topology and source identities;
- per-shape median and P99 benefit;
- host submission cost;
- timeout count;
- numerical deltas;
- memory delta;
- cleanup evidence;
- producer and independent-verifier classifications; and
- whether runtime integration is authorized.

- [ ] **Step 5: Commit compact evidence and audit**

Stage only the compact controller receipts, final bundle, and audit. Do not
stage raw build objects, large logs, or traces.

```bash
git -c core.hooksPath=/dev/null commit \
  -m "perf(runtime): qualify TP4 peer reduction" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

If classification is not `PASS`, stop here and preserve the measured
`NO_GO` or `INELIGIBLE_TOPOLOGY`.

### Task 7: Conditionally integrate F16, then qualify A64 and E2E

**Files:**

- Modify: `tinyvllm/layers/linear.py`
- Modify: `tinyvllm/layers/qwen35_full_attention.py`
- Modify: `tinyvllm/layers/qwen35_linear_attention.py`
- Modify: `tinyvllm/layers/qwen35_decoder_layer.py`
- Modify: `tinyvllm/models/qwen35_components.py`
- Modify: `tinyvllm/engine/model_runner.py`
- Modify: `tinyvllm/config.py`
- Create: `tools/test_qwen38_tp4_peer_reduction_integration.py`
- Create: `tools/qwen38_tp4_peer_reduction_worker.py`
- Create: `tools/run_qwen38_tp4_peer_reduction.py`
- Create: `tools/assemble_qwen38_tp4_peer_reduction.py`
- Create: `tools/verify_qwen38_tp4_peer_reduction.py`
- Create: `docs/superpowers/audits/2026-08-30-qwen38-tp4-peer-reduction-audit.md`

**Interfaces:**

- Consumes a Stage-1 `PASS` bundle hash.
- Adds a default-disabled policy selecting exactly `F16` or `A64`.
- Prefill and unsupported decode retain the baseline implementation.
- Produces the terminal classification
  `GO_TP4_PEER_REDUCTION_RESIDUAL_FUSION` or a fail-closed alternative.

- [ ] **Step 1: Write integration RED tests**

Require:

- default configuration never constructs a peer group;
- only Qwen3.8-27B TP4 BF16 may enable the policy;
- F16 selects exactly layers
  `3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63`;
- prefill always calls `forward_prefill()`;
- decode with active tokens above eight calls baseline `forward()`;
- selected decode calls `forward_local_partial_into(...)` followed by
  `reduce_add_residual(...)`;
- the decoder shell does not add the first residual twice;
- unselected layers retain the current residual path;
- graph mode uses baseline;
- a poisoned group rejects further candidate work; and
- close is invoked during model-runner teardown.

- [ ] **Step 2: Run RED**

Run each shell/factory test file in an isolated pytest process to avoid
intentional `sys.modules` stub pollution.

- [ ] **Step 3: Implement the minimum F16 integration**

Add:

```python
RowParallelLinear.forward_local_partial_into(
    x: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor
```

Add an optional residual-aware projection protocol to both attention shells.
The decoder shell accepts either:

```text
ordinary mixed tensor -> shell adds first residual
residual-fused tensor receipt -> shell does not add first residual again
```

The receipt is a typed dataclass, not a tuple or magic attribute. The model
factory receives the frozen layer index and injects peer reduction only into
selected output projections.

- [ ] **Step 4: Run focused GREEN**

Run:

```bash
/opt/homebrew/bin/python3.12 -m pytest \
  tools/test_qwen38_tp4_peer_reduction_integration.py -q
/opt/homebrew/bin/python3.12 -m pytest \
  tools/test_qwen35_output_projection_row_parallel.py -q
/opt/homebrew/bin/python3.12 -m pytest \
  tools/test_qwen35_full_attention_shell.py -q
/opt/homebrew/bin/python3.12 -m pytest \
  tools/test_qwen35_linear_attention_shell.py -q
/opt/homebrew/bin/python3.12 -m pytest \
  tools/test_qwen35_decoder_layer_shell.py -q
/opt/homebrew/bin/python3.12 -m pytest \
  tools/test_qwen35_concrete_component_factory.py -q
/opt/homebrew/bin/python3.12 -m pytest \
  tools/test_qwen35_checkpoint_target_binding.py -q
```

Also run `py_compile` for every modified Python file and a scoped
`git diff --check`.

- [ ] **Step 5: Commit the default-disabled runtime candidate**

Stage only the files named in this task and commit:

```text
feat(runtime): integrate TP4 peer reduction candidate

Co-authored-by: TRAE CLI <noreply@bytedance.com>
```

Push to `origin/feat/kv-sparse-attention`.

- [ ] **Step 6: Run F16**

Use the frozen P0/P1/Q0/Q1/Q2 matrix, two warmups, at least five alternating
measured repetitions, exact token parity, memory telemetry, collective
census, cleanup, producer verification, and independent verification.

- [ ] **Step 7: Conditionally run A64**

Run A64 only if F16 preserves exact tokens, improves TPOT by at least 3%, has
no workload P99 regression above 3%, and stays within memory budget.

- [ ] **Step 8: Final audit and commit**

The final audit reports, for baseline/F16/A64:

- TPOT median and P99;
- TTFT median and P99;
- request and output-token throughput;
- selected collective count;
- timeout count;
- allocated/reserved memory;
- numerical error;
- topology limitation;
- cleanup; and
- terminal classification.

Commit and push exact compact evidence plus the audit only after fresh
verification.

## Plan self-review

- Spec coverage: Tasks 1–7 cover stale-count repair, topology qualification,
  CUDA IPC lifecycle, bounded generation protocol, real-shape microgate,
  conditional runtime integration, paired E2E, dual verification, cleanup,
  and benefit-plus-cost reporting.
- Placeholder scan: no `TBD`, `TODO`, “implement later,” or unspecified test
  step remains.
- Type consistency: `TP4PeerReductionGroup.create`,
  `reduce_add_residual`, `close`, `PeerReductionPolicy`,
  `validate_peer_topology`, and `classify_peer_microgate` use the same names
  across producer, worker, controller, integration, and verifier tasks.
- Scope: runtime integration is explicitly conditional on the standalone
  microgate; a failed microgate terminates the plan without speculative model
  edits.
