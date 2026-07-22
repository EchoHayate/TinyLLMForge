# Production Exact-Width Multi-Sequence CUDA Graph Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. This repository's approved execution mode is inline; do not dispatch subagents.

**Goal:** Add a default-off, bounded exact-identity CUDA Graph cache for production multi-sequence decode, then independently determine with source-bound correctness and arrival-load evidence whether it provides a real throughput/latency benefit without correctness, memory, startup, or capacity regressions.

**Architecture:** Keep the existing batch-one startup graph path and broad batch-greater-than-one eager guard. A separate `ExactCudaGraphCache` module owns immutable identities, observations, admission/rejection state, five independent budgets, entry-private tensors, and closed-enum telemetry; `ModelRunner` owns exact runtime-input preparation, scratch-KV capture, replay, and fail-closed eager fallback. A source-bound remote runner executes the existing 315-case diagnostic plus actual `ModelRunner` correctness and paired arrival-load workloads, while an independent verifier reconstructs provenance, exact identities, output equality, hit/fallback behavior, capacity, metrics, and the final `GO | NO_GO | INCOMPLETE` classification from immutable artifacts.

**Tech Stack:** Python 3, dataclasses, PyTorch CUDA Graphs, FlashAttention 2.6.3 paged decode, TinyLLMForge `ModelRunner`/`LLMEngine`, JSON/JSONL, SHA-256, dependency-light script tests, Qwen3-0.6B BF16 on A100 GPU 0, SSH ControlMaster.

## Global Constraints

- Work only in `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`; never modify `/Users/bytedance/dev/TinyLLMForge`.
- Execute inline in the current session; do not dispatch subagents.
- Keep `multi_sequence_cuda_graphs=False` by default.
- Preserve disabled behavior, including the current broad `multi_sequence_decode` eager guard.
- Preserve the existing batch-one CUDA Graph fast path.
- When the feature is enabled, startup capture is batch one only; batch-greater-than-one entries are exact-identity lazy captures.
- Production exact identity requires `graph_batch_size == active_batch_size`.
- Never round active batch size or page-table width and never expose rounded replay through configuration.
- Reuse `FlashAttentionGraphIdentity` and require FlashAttention version exactly `2.6.3`.
- The initial batch allowlist is exactly `(2, 4, 8)`.
- Require exactly `3` successful eager observations before one capture attempt.
- Enforce five independent default ceilings: `8` ready entries, `64 MiB` static bytes, `512 MiB` incremental reserved CUDA bytes, `2_000_000_000 ns` single capture, and `5_000_000_000 ns` cumulative capture.
- Do not evict, recapture, asynchronously capture, or rely on allocator reclamation after an overshoot.
- Every cache miss, unsupported condition, identity failure, budget failure, capture failure, or rejected entry stays eager.
- Graph entries must own separate static tensors; only the CUDA graph pool may be shared.
- Capture must use scheduler-invisible scratch KV blocks and must snapshot/restore every scratch write slot.
- Candidate and baseline production processes must expose the same scheduler-visible KV block count.
- A replay exception after possible live KV mutation must fail the current step; do not silently rerun eager.
- Keep Light Doc Cache, Gist KV sharing, token sparsity, low rank, KV quantization/offload, Quest, KV-Cartridge, Attention Matching, speculative verification, mixed prefill/decode, input embeddings, hidden-state return, and prefill out of this feature.
- GPU/model work runs only on `sitian@10.232.195.203` as user `sitian`.
- Use SSH ControlMaster `/tmp/ssh-sitian-10.232.195.203`.
- Use remote Python `/data00/home/sitian/sitian-workspace01/tllm/env/bin/python`.
- Use model `/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B`.
- Set `CUDA_VISIBLE_DEVICES=0` and give every model process unique dynamic `TINYVLLM_DIST_PORT` and `MASTER_PORT`.
- Do not modify the remote checkout, use rsync, kill unrelated processes, clear shared `/tmp`, or reuse fixed ports.
- Retry only `EADDRINUSE`, with a fresh port pair.
- Preserve unrelated untracked `experiments/` directories; stage exact paths only and never use `git add -A`.
- Do not update `README.md` before an independently verified production `GO`.
- Production thresholds are conjunctive: aggregate decode throughput `>=1.15x`, stable-exact decode throughput `>=1.25x`, every request-throughput ratio `>=0.95x`, every p95 ITL ratio `<=1.05x`, every p99 ITL ratio `<=1.10x`, peak reserved-memory ratio `<=1.02x`, initialization-duration ratio `<=1.05x`, and stable-exact graph hit rate `>=0.60`.

---

## File Map

- Modify `tinyvllm/config.py`: add the default-off feature and seven bounded controls; canonicalize and validate the exact allowlist and all positive ceilings.
- Modify `tinyvllm/engine/flash_attn_split_policy.py`: tighten production exact-identity construction so graph and active batch equality can be required explicitly.
- Create `tinyvllm/engine/exact_cuda_graph_cache.py`: immutable configuration snapshot, closed fallback enum, entry/cache state, observation/admission decisions, five budgets, counters, and event serialization.
- Modify `tinyvllm/engine/model_runner.py`: reserve scheduler-invisible scratch KV blocks, preserve batch-one startup capture, build exact identities, perform eager observation, capture at the post-step boundary, replay ready entries, and publish one dispatch event per multi-sequence decode step.
- Modify `tools/multi_sequence_cuda_graph_contract.py`: freeze production defaults, fallback reasons, workload matrix, artifact names, capacity contract, and independent classification inputs.
- Modify `tools/test_multi_sequence_cuda_graph_gate.py`: dependency-light TDD for configuration, identity, cache lifecycle, all budgets, scratch isolation, exact dispatch, artifacts, verifier tamper rejection, and remote command discipline.
- Modify `tools/test_model_runner_spec_verify.py`: preserve the old default-off eager assertion and add focused real-dispatch doubles for exact ready hits and fail-closed paths.
- Create `tools/run_multi_sequence_cuda_graph_production_gate_remote.py`: source snapshot, preflight, local worker mode, paired capacity calibration, actual `LLMEngine` correctness/arrival workloads, remote orchestration, downloads, and local verification.
- Create `tools/verify_multi_sequence_cuda_graph_production.py`: independent artifact/provenance/correctness/identity/capacity/performance reconstruction and final classification.
- Modify `tools/arrival_load_gate.py`: expose shared percentile/ratio helpers and freeze the exact CUDA Graph production thresholds without changing existing P0/P4/P5 policy semantics.
- Modify `AGENT_HANDOFF_STATE.md`: record implementation state and canonical result only after real evidence exists.
- Modify `README.md` only after independent production `GO`.
- Create raw run directories under `experiments/cuda_graph/<run-tag>/`; never stage them.

## Shared Interfaces

Use these exact configuration fields:

```python
multi_sequence_cuda_graphs: bool = False
multi_sequence_cuda_graph_batch_allowlist: tuple = (2, 4, 8)
multi_sequence_cuda_graph_min_observations: int = 3
multi_sequence_cuda_graph_max_entries: int = 8
multi_sequence_cuda_graph_max_static_bytes: int = 64 * 1024 * 1024
multi_sequence_cuda_graph_max_reserved_bytes: int = 512 * 1024 * 1024
multi_sequence_cuda_graph_max_total_capture_ns: int = 5_000_000_000
multi_sequence_cuda_graph_max_single_capture_ns: int = 2_000_000_000
```

Use these exact cache interfaces:

```python
FALLBACK_REASONS = (
    "feature_disabled",
    "enforce_eager",
    "unsupported_mode",
    "incompatible_feature",
    "batch_not_allowlisted",
    "identity_invalid",
    "cold_identity",
    "entry_limit",
    "static_byte_budget",
    "reserved_byte_budget",
    "single_capture_budget",
    "total_capture_budget",
    "scratch_unavailable",
    "capture_failed",
    "identity_drift",
    "replay_disabled",
)


@dataclass(frozen=True)
class ExactCudaGraphCacheConfig:
    enabled: bool
    batch_allowlist: tuple
    min_observations: int
    max_entries: int
    max_static_bytes: int
    max_reserved_bytes: int
    max_total_capture_ns: int
    max_single_capture_ns: int


@dataclass
class ExactCudaGraphEntry:
    identity: FlashAttentionGraphIdentity
    identity_sha256: str
    graph: object | None
    tensors: dict[str, object]
    static_bytes: int
    capture_duration_ns: int
    allocated_delta_bytes: int
    reserved_delta_bytes: int
    replay_count: int = 0
    last_replay_step: int | None = None
    state: str = "ready"
    rejection_reason: str | None = None


@dataclass(frozen=True)
class AdmissionDecision:
    should_capture: bool
    cache_state: str
    fallback_reason: str
    observation_count: int


ExactCudaGraphCache.observe_success(
    identity: FlashAttentionGraphIdentity,
    *,
    estimated_static_bytes: int,
) -> AdmissionDecision

ExactCudaGraphCache.ready_entry(
    identity: FlashAttentionGraphIdentity,
) -> ExactCudaGraphEntry | None

ExactCudaGraphCache.commit_capture(
    entry: ExactCudaGraphEntry,
) -> None

ExactCudaGraphCache.reject(
    identity: FlashAttentionGraphIdentity,
    reason: str,
    *,
    retained_reserved_bytes: int = 0,
) -> None

ExactCudaGraphCache.disable_entry(
    identity_sha256: str,
    reason: str,
) -> None

ExactCudaGraphCache.summary() -> dict
```

Use these exact `ModelRunner` helper interfaces:

```python
def _multi_sequence_graph_incompatible_reason(
    self,
    *,
    mode: str,
    is_prefill: bool,
    input_embeds,
    return_hidden: bool,
) -> str | None


def _build_multi_sequence_graph_identity(
    self,
    input_ids,
    context,
) -> FlashAttentionGraphIdentity


def _estimate_exact_graph_static_bytes(
    self,
    *,
    batch_size: int,
    page_table_width: int,
) -> int


def _capture_exact_multi_sequence_graph(
    self,
    *,
    identity: FlashAttentionGraphIdentity,
    input_ids,
    positions,
    context,
) -> ExactCudaGraphEntry


def _replay_exact_multi_sequence_graph(
    self,
    entry: ExactCudaGraphEntry,
    *,
    input_ids,
    positions,
    context,
) -> object


def cuda_graph_dispatch_observation(self) -> dict | None
```

Every multi-sequence decode dispatch event uses these exact keys:

```python
DISPATCH_EVENT_FIELDS = (
    "step_id",
    "request_ids_hash",
    "mode",
    "active_batch_size",
    "page_table_width",
    "effective_num_splits",
    "graph_identity_sha256",
    "feature_enabled",
    "dispatch",
    "cache_state",
    "observation_count",
    "fallback_reason",
    "capture_attempted",
    "capture_duration_ns",
    "capture_static_bytes",
    "capture_allocated_delta_bytes",
    "capture_reserved_delta_bytes",
    "cache_ready_entries",
    "cache_static_bytes",
    "cache_reserved_delta_bytes",
    "cache_total_capture_ns",
    "source_sha256",
)
```

The production runner CLI is:

```text
python tools/run_multi_sequence_cuda_graph_production_gate_remote.py
  preflight|local-contracts|correctness-smoke|correctness-canonical|arrival-smoke|arrival-canonical|download-only|verify-only
  --run-tag RUN_TAG
  [--diagnostic-run-tag DIAGNOSTIC_RUN_TAG]
```

The independent verifier CLI is:

```text
python tools/verify_multi_sequence_cuda_graph_production.py
  --run-dir RUN_DIR
  --write-report
```

---

### Task 1: Add Fail-Closed Configuration and Exact Identity Validation

**Files:**
- Modify: `tinyvllm/config.py`
- Modify: `tinyvllm/engine/flash_attn_split_policy.py`
- Test: `tools/test_multi_sequence_cuda_graph_gate.py`

**Interfaces:**
- Consumes: existing `Config`, `FlashAttentionSplitInputs`, and `FlashAttentionGraphIdentity`.
- Produces: eight validated configuration fields and an added `require_exact_batch: bool = False` keyword on `build_flash_attn_263_graph_identity()`.

- [ ] **Step 1: Write failing configuration default and canonicalization tests**

Add dependency-light tests that instantiate the real `Config` with a temporary model directory and a fake `AutoConfig`:

```python
def test_multi_sequence_cuda_graph_config_defaults_and_allowlist():
    Config = load_real_config_class()
    with tempfile.TemporaryDirectory() as model:
        config = Config(model=model)
        assert config.multi_sequence_cuda_graphs is False
        assert config.multi_sequence_cuda_graph_batch_allowlist == (2, 4, 8)
        assert config.multi_sequence_cuda_graph_min_observations == 3
        assert config.multi_sequence_cuda_graph_max_entries == 8
        assert config.multi_sequence_cuda_graph_max_static_bytes == 64 * 1024 * 1024
        assert config.multi_sequence_cuda_graph_max_reserved_bytes == 512 * 1024 * 1024
        assert config.multi_sequence_cuda_graph_max_total_capture_ns == 5_000_000_000
        assert config.multi_sequence_cuda_graph_max_single_capture_ns == 2_000_000_000

        canonical = Config(
            model=model,
            multi_sequence_cuda_graph_batch_allowlist=(8, 2, 4, 4),
        )
        assert canonical.multi_sequence_cuda_graph_batch_allowlist == (2, 4, 8)
```

- [ ] **Step 2: Write failing rejection tests for booleans, non-integers, batch one, and non-positive budgets**

```python
def test_multi_sequence_cuda_graph_config_rejects_invalid_controls():
    Config = load_real_config_class()
    invalid = (
        {"multi_sequence_cuda_graph_batch_allowlist": ()},
        {"multi_sequence_cuda_graph_batch_allowlist": (1, 2)},
        {"multi_sequence_cuda_graph_batch_allowlist": (2, True)},
        {"multi_sequence_cuda_graph_batch_allowlist": (2, 4.0)},
        {"multi_sequence_cuda_graph_min_observations": 0},
        {"multi_sequence_cuda_graph_max_entries": 0},
        {"multi_sequence_cuda_graph_max_static_bytes": 0},
        {"multi_sequence_cuda_graph_max_reserved_bytes": 0},
        {"multi_sequence_cuda_graph_max_total_capture_ns": 0},
        {"multi_sequence_cuda_graph_max_single_capture_ns": 0},
    )
    with tempfile.TemporaryDirectory() as model:
        for overrides in invalid:
            try:
                Config(model=model, **overrides)
            except AssertionError:
                pass
            else:
                raise AssertionError(f"invalid CUDA Graph config accepted: {overrides}")
```

- [ ] **Step 3: Write failing exact-batch identity tests**

```python
def test_production_identity_requires_graph_batch_equal_active_batch():
    split_policy = load_split_policy()
    inputs = split_policy.FlashAttentionSplitInputs(
        batch_size=4,
        num_query_heads=16,
        num_kv_heads=8,
        head_dim=128,
        page_block_size=256,
        page_table_width=2,
        max_seqlen_q=1,
        multi_processor_count=108,
    )
    exact = split_policy.build_flash_attn_263_graph_identity(
        graph_batch_size=4,
        inputs=inputs,
        flash_attn_version="2.6.3",
        require_exact_batch=True,
    )
    assert exact.graph_batch_size == exact.active_batch_size == 4
    with pytest.raises(ValueError, match="equal"):
        split_policy.build_flash_attn_263_graph_identity(
            graph_batch_size=8,
            inputs=inputs,
            flash_attn_version="2.6.3",
            require_exact_batch=True,
        )
```

- [ ] **Step 4: Run the focused tests and verify RED**

Run:

```bash
python tools/test_multi_sequence_cuda_graph_gate.py
```

Expected: FAIL because the new fields and `require_exact_batch` parameter do not exist.

- [ ] **Step 5: Implement the fields, canonicalization, assertions, and exact-batch option**

Add the eight fields beside `enforce_eager`, canonicalize the allowlist before validation, reject `bool`, and add:

```python
allowlist = self.multi_sequence_cuda_graph_batch_allowlist
assert isinstance(allowlist, (tuple, list))
assert allowlist
assert all(
    isinstance(value, int)
    and not isinstance(value, bool)
    and value > 1
    for value in allowlist
)
self.multi_sequence_cuda_graph_batch_allowlist = tuple(
    sorted(set(allowlist))
)
for value in (
    self.multi_sequence_cuda_graph_min_observations,
    self.multi_sequence_cuda_graph_max_entries,
    self.multi_sequence_cuda_graph_max_static_bytes,
    self.multi_sequence_cuda_graph_max_reserved_bytes,
    self.multi_sequence_cuda_graph_max_total_capture_ns,
    self.multi_sequence_cuda_graph_max_single_capture_ns,
):
    assert isinstance(value, int) and not isinstance(value, bool)
    assert value > 0
```

Extend the identity builder with:

```python
if require_exact_batch and graph_batch_size != inputs.batch_size:
    raise ValueError(
        "production graph_batch_size must equal active batch size"
    )
```

- [ ] **Step 6: Run the focused tests and verify GREEN**

Run:

```bash
python tools/test_multi_sequence_cuda_graph_gate.py
```

Expected: PASS, including all pre-existing 315-case contract tests.

- [ ] **Step 7: Commit**

```bash
git add tinyvllm/config.py tinyvllm/engine/flash_attn_split_policy.py tools/test_multi_sequence_cuda_graph_gate.py
git commit -m "feat: validate exact cuda graph configuration"
```

### Task 2: Implement the Pure Cache Lifecycle and Five Budgets

**Files:**
- Create: `tinyvllm/engine/exact_cuda_graph_cache.py`
- Modify: `tools/multi_sequence_cuda_graph_contract.py`
- Modify: `tools/test_multi_sequence_cuda_graph_gate.py`

**Interfaces:**
- Consumes: `FlashAttentionGraphIdentity` and Task 1 configuration values.
- Produces: `ExactCudaGraphCacheConfig`, `ExactCudaGraphEntry`, `AdmissionDecision`, `ExactCudaGraphCache`, `FALLBACK_REASONS`, and production defaults in the contract module.

- [ ] **Step 1: Write failing tests for three observations and the first later hit**

```python
def test_exact_cache_observes_three_eager_steps_before_capture():
    cache_module = load_exact_cache()
    identity = make_identity(batch=4, width=2, splits=2)
    cache = cache_module.ExactCudaGraphCache(make_cache_config())

    first = cache.observe_success(identity, estimated_static_bytes=4096)
    second = cache.observe_success(identity, estimated_static_bytes=4096)
    third = cache.observe_success(identity, estimated_static_bytes=4096)
    assert [first.should_capture, second.should_capture, third.should_capture] == [
        False, False, True,
    ]
    assert [first.observation_count, second.observation_count, third.observation_count] == [
        1, 2, 3,
    ]

    entry = make_entry(identity, static_bytes=4096)
    cache.commit_capture(entry)
    assert cache.ready_entry(identity) is entry
```

- [ ] **Step 2: Write failing tests for all five ceilings**

Use one independent cache instance per ceiling and assert exact reasons:

```python
def test_every_exact_cache_budget_blocks_admission_independently():
    cases = (
        ("max_entries", 1, "entry_limit"),
        ("max_static_bytes", 4095, "static_byte_budget"),
        ("max_reserved_bytes", 1023, "reserved_byte_budget"),
        ("max_single_capture_ns", 99, "single_capture_budget"),
        ("max_total_capture_ns", 199, "total_capture_budget"),
    )
    for field, value, expected_reason in cases:
        cache = ExactCudaGraphCache(make_cache_config(**{field: value}))
        exercise_budget(cache, expected_reason)
```

`exercise_budget()` must commit one valid entry or rejection measurement as needed so only the named ceiling is crossed.

- [ ] **Step 3: Write failing tests for stable rejection, no recapture, no eviction, and exact SHA lookup**

```python
def test_rejected_identity_is_terminal_and_exact_lookup_only():
    cache = ExactCudaGraphCache(make_cache_config())
    identity = make_identity(batch=4, width=2, splits=2)
    wider = make_identity(batch=4, width=3, splits=3)
    cache.reject(identity, "capture_failed", retained_reserved_bytes=8192)
    assert cache.ready_entry(identity) is None
    assert cache.ready_entry(wider) is None
    for _ in range(10):
        decision = cache.observe_success(identity, estimated_static_bytes=4096)
        assert decision.should_capture is False
        assert decision.cache_state == "rejected"
        assert decision.fallback_reason == "capture_failed"
    assert cache.summary()["capture_attempts"] == 0
```

- [ ] **Step 4: Write failing tests for entry-private tensor ownership and closed enums**

```python
def test_entries_do_not_share_static_tensor_objects():
    first = make_entry(make_identity(batch=2, width=1, splits=2))
    second = make_entry(make_identity(batch=4, width=1, splits=2))
    assert set(first.tensors) == set(second.tensors)
    assert all(first.tensors[name] is not second.tensors[name] for name in first.tensors)


def test_fallback_reason_contract_is_closed_and_complete():
    cache_module = load_exact_cache()
    assert cache_module.FALLBACK_REASONS == contract.FALLBACK_REASONS
    assert len(set(cache_module.FALLBACK_REASONS)) == len(cache_module.FALLBACK_REASONS)
```

- [ ] **Step 5: Run the focused tests and verify RED**

Run:

```bash
python tools/test_multi_sequence_cuda_graph_gate.py
```

Expected: FAIL because `exact_cuda_graph_cache.py` and production constants do not exist.

- [ ] **Step 6: Implement the dependency-light cache module**

The module may import dataclasses, enum-free string constants, and the split-policy identity, but must not import `ModelRunner`. Store:

```python
self.observation_counts: dict[str, int] = {}
self.ready_entries: dict[str, ExactCudaGraphEntry] = {}
self.rejected: dict[str, str] = {}
self.capturing: set[str] = set()
self.static_bytes = 0
self.reserved_delta_bytes = 0
self.total_capture_ns = 0
self.counters = collections.Counter()
```

`observe_success()` returns `cold_identity` until the exact count reaches `min_observations`, checks all five pre-capture ceilings, adds the SHA to `capturing` exactly once, and never mutates a rejected identity. `commit_capture()` validates SHA equality, terminal state, actual budget values, and converts any overshoot into permanent rejection. `summary()` returns only JSON-serializable values sorted by SHA.

- [ ] **Step 7: Run tests and verify GREEN**

Run:

```bash
python tools/test_multi_sequence_cuda_graph_gate.py
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add tinyvllm/engine/exact_cuda_graph_cache.py tools/multi_sequence_cuda_graph_contract.py tools/test_multi_sequence_cuda_graph_gate.py
git commit -m "feat: add bounded exact cuda graph cache"
```

### Task 3: Reserve Scratch KV Capacity and Preserve Batch-One Startup

**Files:**
- Modify: `tinyvllm/engine/model_runner.py`
- Modify: `tools/test_multi_sequence_cuda_graph_gate.py`
- Modify: `tools/test_model_runner_spec_verify.py`

**Interfaces:**
- Consumes: Task 1 config and Task 2 cache.
- Produces: scheduler-visible versus physical KV capacity, scratch block/slot helpers, exact identity/static-byte helpers, and feature-aware startup capture.

- [ ] **Step 1: Write failing capacity-resolution tests**

Test a pure helper:

```python
def test_exact_graph_capacity_reserves_scheduler_invisible_scratch():
    assert resolve_exact_graph_kv_capacity(
        auto_blocks=100,
        requested_visible_blocks=-1,
        feature_enabled=True,
        scratch_blocks=8,
    ) == (92, 100)
    assert resolve_exact_graph_kv_capacity(
        auto_blocks=100,
        requested_visible_blocks=80,
        feature_enabled=True,
        scratch_blocks=8,
    ) == (80, 88)
    assert resolve_exact_graph_kv_capacity(
        auto_blocks=100,
        requested_visible_blocks=-1,
        feature_enabled=False,
        scratch_blocks=8,
    ) == (100, 100)
```

Also assert that requesting `96` visible plus `8` scratch against `100` physical raises before allocation.

- [ ] **Step 2: Write failing scratch-ID and scheduler isolation tests**

```python
def test_scratch_blocks_are_above_scheduler_visible_range():
    runner = make_capacity_runner(
        visible_blocks=92,
        physical_blocks=100,
        scratch_blocks=tuple(range(92, 100)),
    )
    assert runner._exact_graph_scratch_block_ids == tuple(range(92, 100))
    assert max(range(runner.config.num_kvcache_blocks)) < min(
        runner._exact_graph_scratch_block_ids
    )
    slots = runner._exact_graph_scratch_slots(batch_size=8)
    assert slots == tuple(block * runner.block_size for block in range(92, 100))
```

- [ ] **Step 3: Write failing startup-capture tests**

```python
def test_feature_enabled_startup_captures_only_batch_one():
    runner = make_capture_runner(feature_enabled=True)
    runner.capture_cudagraph()
    assert tuple(runner.graph_bs) == (1,)
    assert set(runner.graphs) == {1}


def test_feature_disabled_startup_inventory_is_unchanged():
    runner = make_capture_runner(feature_enabled=False)
    runner.capture_cudagraph()
    assert runner.graph_bs[:4] == [1, 2, 4, 8]
```

- [ ] **Step 4: Write failing identity/static-byte tests**

Use fake context tensors and fake A100 properties to assert:

```python
identity = runner._build_multi_sequence_graph_identity(
    FakeTensor([10, 20, 30, 40]),
    make_context(width=2),
)
assert identity.graph_batch_size == identity.active_batch_size == 4
assert identity.page_table_width == 2
assert identity.flash_attn_version == "2.6.3"
assert identity.multi_processor_count == 108
assert runner._estimate_exact_graph_static_bytes(
    batch_size=4,
    page_table_width=2,
) > 0
```

Changing batch, width, split inputs, SM count, model heads, head dimension, or FlashAttention version must change the SHA or fail construction.

- [ ] **Step 5: Run focused tests and verify RED**

Run:

```bash
python tools/test_multi_sequence_cuda_graph_gate.py
python tools/test_model_runner_spec_verify.py
```

Expected: FAIL because capacity, scratch, identity, and feature-aware startup helpers do not exist.

- [ ] **Step 6: Implement capacity reservation before KV allocation**

Derive `scratch_blocks = max(config.multi_sequence_cuda_graph_batch_allowlist)` only when the feature is enabled. For automatic sizing, allocate `auto_num_blocks` physically and expose `auto_num_blocks - scratch_blocks`; for explicit sizing, treat `config.num_kvcache_blocks` as scheduler-visible and allocate `visible + scratch_blocks`. Set:

```python
self._physical_num_kvcache_blocks = physical_blocks
self._exact_graph_scratch_block_ids = tuple(
    range(visible_blocks, physical_blocks)
)
config.num_kvcache_blocks = visible_blocks
```

Allocate `self.kv_cache` with `physical_blocks`; the later `Scheduler(config)` therefore sees only `visible_blocks`.

- [ ] **Step 7: Initialize the cache and preserve startup semantics**

After KV allocation, create `ExactCudaGraphCache` from an immutable config snapshot. If the feature is enabled, `capture_cudagraph()` sets `self.graph_bs = [1]`; otherwise retain the existing list exactly. Keep all prior skip conditions authoritative and let `enforce_eager=True` leave the feature unreachable.

- [ ] **Step 8: Run tests and verify GREEN**

Run:

```bash
python tools/test_multi_sequence_cuda_graph_gate.py
python tools/test_model_runner_spec_verify.py
```

Expected: PASS, including the original default-off multi-sequence eager test and batch-one replay test.

- [ ] **Step 9: Commit**

```bash
git add tinyvllm/engine/model_runner.py tools/test_multi_sequence_cuda_graph_gate.py tools/test_model_runner_spec_verify.py
git commit -m "feat: reserve exact graph scratch capacity"
```

### Task 4: Integrate Eager Observation, Safe Capture, Exact Replay, and Telemetry

**Files:**
- Modify: `tinyvllm/engine/model_runner.py`
- Modify: `tools/test_multi_sequence_cuda_graph_gate.py`
- Modify: `tools/test_model_runner_spec_verify.py`

**Interfaces:**
- Consumes: Tasks 2-3 cache, scratch blocks, identity, and static-byte estimate.
- Produces: exact capture/replay helpers, fail-closed `run_model()` dispatch, `last_cuda_graph_dispatch_event`, and `cuda_graph_dispatch_observation()`.

- [ ] **Step 1: Write failing dispatch-state tests**

Cover this exact sequence for batch `4`, width `2`:

```python
def test_three_successful_eager_steps_capture_post_step_and_fourth_replays():
    runner = make_exact_dispatch_runner(min_observations=3)
    outputs = [run_decode(runner) for _ in range(4)]
    assert [row["dispatch"] for row in outputs] == [
        "eager", "eager", "eager", "graph",
    ]
    assert [row["capture_attempted"] for row in outputs] == [
        False, False, True, False,
    ]
    assert outputs[2]["fallback_reason"] == "cold_identity"
    assert outputs[3]["cache_state"] == "ready"
    assert outputs[3]["graph_identity_sha256"] == expected_identity.sha256
```

The third call must return the eager logits even though capture succeeds afterward.

- [ ] **Step 2: Write failing exact-match and no-rounding tests**

Create ready entries for `(batch=4,width=2)` and assert batch `3`, batch `5`, width `1`, and width `3` all execute eager. Assert no code path selects a larger batch or wider page table.

- [ ] **Step 3: Write failing incompatibility and guard-authority tests**

For each condition below, assert eager dispatch and the exact fallback reason:

```text
feature disabled -> feature_disabled
enforce_eager -> enforce_eager
prefill or spec_verify -> unsupported_mode
Quest/AM/C4/CPU offload/KV offload/input_embeds/return_hidden -> incompatible_feature
batch outside (2,4,8) -> batch_not_allowlisted
invalid width/version/shape -> identity_invalid
no ready entry -> cold_identity or terminal budget/capture reason
```

Retain the original `multi_sequence_decode` predicate and assert its only exception is a feature-enabled ready exact entry.

- [ ] **Step 4: Write failing scratch snapshot/restore and context-finally tests**

Use fake KV tensors and a fake graph:

```python
def test_capture_snapshots_and_restores_all_scratch_slots():
    before = clone_scratch_bytes(runner)
        runner._capture_exact_multi_sequence_graph(
            identity=identity,
            input_ids=input_ids,
            positions=positions,
            context=runtime_context,
        )
    after = clone_scratch_bytes(runner)
    assert after == before


def test_replay_resets_context_on_success_and_exception():
    for graph in (SuccessfulGraph(), RaisingGraph()):
        context.reset_context()
        try:
            runner._replay_exact_multi_sequence_graph(
                entry_with(graph),
                input_ids=input_ids,
                positions=positions,
                context=runtime_context,
            )
        except RuntimeError:
            pass
        assert context.get_context().is_prefill is False
        assert context.get_context().slot_mapping is None
```

- [ ] **Step 5: Write failing terminal-failure tests**

Independently inject capture exception, identity drift, static overshoot, reserved overshoot, single-duration overshoot, cumulative-duration overshoot, scratch restore failure, copy-shape mismatch, context failure, and replay failure. Assert permanent rejection, no later capture/replay, and exact fallback telemetry. For replay failure, assert the current call raises instead of rerunning eager.

- [ ] **Step 6: Write failing complete-event tests**

For every fallback reason and one graph hit, assert:

```python
assert tuple(event) == DISPATCH_EVENT_FIELDS
assert event["fallback_reason"] in FALLBACK_REASONS or event["fallback_reason"] is None
assert event["dispatch"] in {"eager", "graph"}
assert event["source_sha256"] == os.environ["TINYVLLM_SOURCE_SHA256"]
```

`request_ids_hash` is SHA-256 of canonical sorted scheduled sequence IDs, set by `run()` before `run_model()`. `step_id` is a process-local monotonically increasing integer.

- [ ] **Step 7: Run focused tests and verify RED**

Run:

```bash
python tools/test_multi_sequence_cuda_graph_gate.py
python tools/test_model_runner_spec_verify.py
```

Expected: FAIL because exact dispatch/capture/replay is not integrated.

- [ ] **Step 8: Implement fail-closed dispatch**

Restructure `run_model()` into:

```python
if not multi_sequence_decode:
    return self._run_existing_single_sequence_or_eager_path(
        input_ids=input_ids,
        positions=positions,
        is_prefill=is_prefill,
        input_embeds=input_embeds,
        return_hidden=return_hidden,
        mode=mode,
    )

reason = self._multi_sequence_graph_incompatible_reason(
    mode=mode,
    is_prefill=is_prefill,
    input_embeds=input_embeds,
    return_hidden=return_hidden,
)
if reason is not None:
    return self._run_eager_with_dispatch_event(
        reason,
        input_ids=input_ids,
        positions=positions,
        input_embeds=input_embeds,
        return_hidden=return_hidden,
        mode=mode,
    )

try:
    identity = self._build_multi_sequence_graph_identity(input_ids, context)
except (ValueError, RuntimeError):
    return self._run_eager_with_dispatch_event(
        "identity_invalid",
        input_ids=input_ids,
        positions=positions,
        input_embeds=input_embeds,
        return_hidden=return_hidden,
        mode=mode,
    )

entry = self.exact_cuda_graph_cache.ready_entry(identity)
if entry is not None:
    return self._replay_exact_multi_sequence_graph(
        entry,
        input_ids=input_ids,
        positions=positions,
        context=context,
    )

logits = self._run_eager_logits(
    input_ids=input_ids,
    positions=positions,
    input_embeds=input_embeds,
)
decision = self.exact_cuda_graph_cache.observe_success(
    identity,
    estimated_static_bytes=self._estimate_exact_graph_static_bytes(
        batch_size=int(input_ids.size(0)),
        page_table_width=int(context.block_tables.size(1)),
    ),
)
if decision.should_capture:
    self._attempt_post_step_capture(
        identity=identity,
        input_ids=input_ids,
        positions=positions,
        context=context,
    )
return logits
```

The ready-entry branch is the narrow exception to the broad guard. All feature-off, miss, not-ready, rejected, and unsupported branches still use eager.

- [ ] **Step 9: Implement safe capture**

Allocate a fresh tensor dictionary per identity with exact shapes for `input_ids`, `positions`, `slot_mapping`, `context_lens`, `block_tables`, and `outputs`. Snapshot scratch slots using `snapshot_kv_slots()`, substitute scratch slot mappings only during warmup/capture, synchronize around `perf_counter_ns()` and CUDA memory counters, restore scratch in `finally`, rebuild the identity after capture, and commit only if every invariant holds.

- [ ] **Step 10: Implement exact replay**

Recompute identity, compare every dataclass field and SHA, overwrite or zero every graph-private input region, copy only exact-shaped tensors, install graph-private context, replay, compute logits from exact output rows, reset context in `finally`, increment replay counters, and never slice/pad a larger graph.

- [ ] **Step 11: Run focused tests and verify GREEN**

Run:

```bash
python tools/test_multi_sequence_cuda_graph_gate.py
python tools/test_model_runner_spec_verify.py
```

Expected: PASS.

- [ ] **Step 12: Commit**

```bash
git add tinyvllm/engine/model_runner.py tools/test_multi_sequence_cuda_graph_gate.py tools/test_model_runner_spec_verify.py
git commit -m "feat: dispatch exact multi sequence cuda graphs"
```

### Task 5: Freeze the Production Workload and Artifact Contract

**Files:**
- Modify: `tools/multi_sequence_cuda_graph_contract.py`
- Modify: `tools/arrival_load_gate.py`
- Modify: `tools/test_multi_sequence_cuda_graph_gate.py`

**Interfaces:**
- Consumes: the approved design thresholds and existing arrival-load metric helpers.
- Produces: production case matrix, immutable artifact names, workload hashes, capacity pairing rules, and `classify_production_gate()` inputs with no batch-size-only hit inference.

- [ ] **Step 1: Write failing matrix tests**

Freeze these workload families:

```python
PRODUCTION_WORKLOADS = (
    "stable_exact_reuse",
    "mixed_allowlist_and_fallback",
    "page_width_transition",
    "short_capture_cold_cost",
    "long_decode",
    "burst_arrivals",
    "near_stable_service_rate",
    "long_prompt_pressure",
)
PRODUCTION_POLICIES = ("baseline", "candidate")
PRODUCTION_MEASURED_REPETITIONS = 5
PRODUCTION_WARMUP_REPETITIONS = 1
```

Assert `8 × 2 × (1 warmup + 5 measured) = 96` isolated cases, with randomized paired order recorded per repetition.

- [ ] **Step 2: Write failing artifact-contract tests**

Freeze:

```python
PRODUCTION_ARTIFACT_FILES = (
    "manifest.json",
    "environment.json",
    "source_manifest.json",
    "dispatch_events.jsonl",
    "capture_events.jsonl",
    "request_metrics.jsonl",
    "model_step_metrics.jsonl",
    "memory_trace.jsonl",
    "correctness_rows.jsonl",
    "case_summaries.json",
    "summary.json",
    "report.md",
    "independent_verification.json",
)
```

Assert no duplicates, stable order, and that the manifest binds source tree, copied files, model/config hash, commands, workload/arrival hashes, baseline/candidate order, PIDs, ports, configs, capacity, and thresholds.

- [ ] **Step 3: Write failing classification tests**

Build one complete synthetic `GO` row set and independently tamper each requirement:

- aggregate decode ratio below `1.15`;
- stable-exact ratio below `1.25`;
- one request ratio below `0.95`;
- one p95 ITL ratio above `1.05`;
- one p99 ITL ratio above `1.10`;
- reserved ratio above `1.02`;
- initialization ratio above `1.05`;
- hit rate below `0.60`;
- missing allowlisted replay;
- fewer than two replayed widths;
- missing non-allowlisted eager fallback;
- unknown fallback reason;
- replay after rejection;
- missing measured repetition;
- output mismatch;
- capacity mismatch;
- producer/independent summary mismatch.

Each complete regression returns `NO_GO`; missing or unverifiable evidence returns `INCOMPLETE`.

- [ ] **Step 4: Run focused tests and verify RED**

Run:

```bash
python tools/test_multi_sequence_cuda_graph_gate.py
```

Expected: FAIL because the production matrix/artifact contract is absent.

- [ ] **Step 5: Implement frozen contracts and shared ratio helpers**

Keep the existing diagnostic constants unchanged. Add production dataclasses and pure builders. Reuse or extract percentile/ratio arithmetic from `arrival_load_gate.py` without changing existing P0/P4/P5 outputs. A graph hit is valid only when `dispatch == "graph"`, `graph_identity_sha256` is a 64-character hash, and the independently rebuilt identity has the same SHA.

- [ ] **Step 6: Run focused tests and verify GREEN**

Run:

```bash
python tools/test_multi_sequence_cuda_graph_gate.py
python tools/test_arrival_load_gate.py
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add tools/multi_sequence_cuda_graph_contract.py tools/arrival_load_gate.py tools/test_multi_sequence_cuda_graph_gate.py
git commit -m "test: freeze exact cuda graph production gate"
```

### Task 6: Implement the Independent Production Verifier

**Files:**
- Create: `tools/verify_multi_sequence_cuda_graph_production.py`
- Modify: `tools/test_multi_sequence_cuda_graph_gate.py`

**Interfaces:**
- Consumes: Task 5 immutable artifacts and raw JSONL rows.
- Produces: `verify_run(run_dir: Path) -> dict`, `independent_verification.json`, and `report.md`.

- [ ] **Step 1: Write a complete synthetic artifact fixture**

The fixture must contain all 96 case summaries, raw dispatch/capture/request/model-step/memory/correctness rows, exact source hashes, paired capacity, two replayed widths, one non-allowlisted eager fallback, and finite metrics that satisfy every threshold.

- [ ] **Step 2: Write failing provenance and hash tamper tests**

Independently test missing file, duplicate row ID, reordered manifest row, mixed source SHA, wrong file hash, wrong workload hash, wrong model hash, wrong command, non-unique port, and non-finite metric. Every tamper must be rejected without trusting `summary.json`.

- [ ] **Step 3: Write failing identity and lifecycle tamper tests**

Change one identity field without updating the SHA, forge a graph hit from batch size alone, replay a rounded width, replay after rejection, alter observation count, share an entry SHA across incompatible tensors, omit a capture event, or report capture success after a budget overshoot. Each must fail independent reconstruction.

- [ ] **Step 4: Write failing correctness/capacity/metric tamper tests**

Change one token, logit tolerance result, live-slot KV hash, scheduler-visible capacity, request throughput, ITL percentile, reserved peak, initialization duration, hit rate, or producer classification. The verifier must recompute and disagree explicitly.

- [ ] **Step 5: Run focused tests and verify RED**

Run:

```bash
python tools/test_multi_sequence_cuda_graph_gate.py
```

Expected: FAIL because the production verifier does not exist.

- [ ] **Step 6: Implement independent reconstruction**

The verifier reads raw files directly, checks exact required filenames, hashes every closed file, validates unique IDs and source identity, reconstructs every `FlashAttentionGraphIdentity`, pairs baseline/candidate by workload and repetition, recomputes request throughput and p95/p99 ITL, validates capacity and memory ratios, derives stable-exact slices only after a ready capture event, and calls the pure classifier with reconstructed rows.

Write `independent_verification.json` atomically, then render `report.md` from that result. Return exactly one top-level classification: `GO`, `NO_GO`, or `INCOMPLETE`.

- [ ] **Step 7: Run focused tests and verify GREEN**

Run:

```bash
python tools/test_multi_sequence_cuda_graph_gate.py
```

Expected: PASS, including every independent tamper.

- [ ] **Step 8: Commit**

```bash
git add tools/verify_multi_sequence_cuda_graph_production.py tools/test_multi_sequence_cuda_graph_gate.py
git commit -m "feat: verify exact cuda graph production evidence"
```

### Task 7: Implement Source-Bound Correctness and Arrival-Load Orchestration

**Files:**
- Create: `tools/run_multi_sequence_cuda_graph_production_gate_remote.py`
- Modify: `tools/test_multi_sequence_cuda_graph_gate.py`

**Interfaces:**
- Consumes: existing source-audit helpers, existing 315-case diagnostic runner/verifier, Tasks 4-6 runtime and contracts.
- Produces: remote preflight, local worker, correctness and arrival modes, paired capacity calibration, resumable downloads, and local independent verification.

- [ ] **Step 1: Write failing CLI and prohibited-operation tests**

Assert all eight modes, required `--run-tag` behavior, fixed remote user/host/Python/model/GPU, ControlMaster path, dynamic port environment, and absence of `rsync`, `kill`, `pkill`, shared `/tmp` deletion, remote checkout writes, and fixed ports.

- [ ] **Step 2: Write failing source snapshot and command-binding tests**

Require exact local files in the snapshot, `TINYVLLM_SOURCE_SHA256` in every worker environment, source validation on remote before execution, and source artifacts copied into the final run before hashing.

- [ ] **Step 3: Write failing capacity-pairing tests**

The candidate calibration process reports physical blocks, scratch blocks, and scheduler-visible blocks. Both paired policies then launch with:

```python
baseline_config = {
    "multi_sequence_cuda_graphs": False,
    "num_kvcache_blocks": candidate_visible_blocks,
}
candidate_config = {
    "multi_sequence_cuda_graphs": True,
    "num_kvcache_blocks": candidate_visible_blocks,
}
```

Assert both runtime `capacity_snapshot()` values are identical and candidate physical capacity equals visible plus scratch.

- [ ] **Step 4: Write failing correctness-worker tests**

The worker must cover allowlisted `2,4,8`, adjacent non-allowlisted `3,5,7,9`, stable widths, one `255 -> 256 -> 257` token page transition, cold observations, capture, later replay, and each budget fallback. For each compared step record eager/candidate tokens, logits comparison, live-slot KV hashes, exact identity, dispatch, and source SHA.

- [ ] **Step 5: Write failing arrival-worker tests**

The worker uses real `LLMEngine.step()` and reads `engine.model_runner.cuda_graph_dispatch_observation()` immediately after each step. It records arrivals, per-request token timestamps, step durations, initialization, memory, dispatch/capture events, output tokens, and complete case summaries for all Task 5 workloads.

- [ ] **Step 6: Run focused tests and verify RED**

Run:

```bash
python tools/test_multi_sequence_cuda_graph_gate.py
```

Expected: FAIL because the production runner does not exist.

- [ ] **Step 7: Implement remote orchestration**

Use `subprocess.run()` argument arrays, `ssh` with:

```text
-o ControlMaster=auto
-o ControlPath=/tmp/ssh-sitian-10.232.195.203
-o ControlPersist=600
sitian@10.232.195.203
```

Create a unique remote directory under the user's workspace, stream the tarred source snapshot with `tar -czf - <owned-files> | ssh <ssh-options> sitian@10.232.195.203 'tar -xzf - -C <remote-run-dir>'`, validate it, and execute worker subprocesses with fresh ports. Record GPU occupancy before and after every process; non-idle unrelated occupancy marks the case incomplete.

`correctness-canonical` first invokes the existing 315-case diagnostic against the candidate source and requires the independent diagnostic classifications `EXACT_REPLAY_CORRECT`, `LEGACY_COMPATIBLE`, `POLICY_EXACT`, and `ROUNDED_REPLAY_CORRUPT`, then executes actual `ModelRunner` correctness.

- [ ] **Step 8: Implement finalization and local verification**

Download to `experiments/cuda_graph/<run-tag>/`, validate artifact completeness, run:

```bash
python tools/verify_multi_sequence_cuda_graph_production.py \
  --run-dir experiments/cuda_graph/<run-tag> \
  --write-report
```

Only after verifier success write `artifact_hashes.json`; `verify-only` must not contact the GPU.

- [ ] **Step 9: Run focused tests and verify GREEN**

Run:

```bash
python tools/test_multi_sequence_cuda_graph_gate.py
```

Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add tools/run_multi_sequence_cuda_graph_production_gate_remote.py tools/test_multi_sequence_cuda_graph_gate.py
git commit -m "feat: add exact cuda graph production gate"
```

### Task 8: Run Complete Local Validation

**Files:**
- No source changes unless a validation failure is directly caused by Tasks 1-7.

**Interfaces:**
- Consumes: all implementation commits.
- Produces: clean local evidence before GPU use.

- [ ] **Step 1: Run the exact focused suites**

```bash
python tools/test_multi_sequence_cuda_graph_gate.py
python tools/test_model_runner_spec_verify.py
python tools/test_arrival_load_gate.py
```

Expected: all PASS.

- [ ] **Step 2: Run source and syntax validation**

```bash
python -m py_compile \
  tinyvllm/config.py \
  tinyvllm/engine/flash_attn_split_policy.py \
  tinyvllm/engine/exact_cuda_graph_cache.py \
  tinyvllm/engine/model_runner.py \
  tools/multi_sequence_cuda_graph_contract.py \
  tools/run_multi_sequence_cuda_graph_production_gate_remote.py \
  tools/verify_multi_sequence_cuda_graph_production.py
git diff --check
```

Expected: exit code `0`.

- [ ] **Step 3: Prove default-off and README boundaries**

```bash
git diff 91af4d6 -- README.md
rg -n "multi_sequence_cuda_graphs: bool = False" tinyvllm/config.py
```

Expected: README diff is empty and exactly one default-off field exists.

- [ ] **Step 4: Inspect repository status**

```bash
git status --short
```

Expected: only intentional tracked changes plus pre-existing untracked `experiments/` artifacts.

### Task 9: Run Remote Preflight and Smoke Gates

**Files:**
- Raw artifacts only under `experiments/cuda_graph/`.

**Interfaces:**
- Consumes: locally validated source.
- Produces: source-bound smoke evidence; no performance claim.

- [ ] **Step 1: Verify the ControlMaster and remote environment**

```bash
ssh -o ControlPath=/tmp/ssh-sitian-10.232.195.203 \
  sitian@10.232.195.203 true
python tools/run_multi_sequence_cuda_graph_production_gate_remote.py \
  preflight \
  --run-tag qwen3-06b-exact-cuda-graph-preflight-$(date +%Y%m%d-%H%M%S)
```

Expected: remote Python/model/GPU/FlashAttention checks pass and GPU 0 is available.

- [ ] **Step 2: Run correctness smoke**

```bash
RUN_TAG=qwen3-06b-exact-cuda-graph-correctness-smoke-$(date +%Y%m%d-%H%M%S)
python tools/run_multi_sequence_cuda_graph_production_gate_remote.py \
  correctness-smoke \
  --run-tag "${RUN_TAG}"
```

Expected: exact output/logit/KV checks pass for allowlisted and adjacent fallback cases; capture/replay telemetry is exact; rounded replay is unavailable.

- [ ] **Step 3: Run arrival smoke**

```bash
RUN_TAG=qwen3-06b-exact-cuda-graph-arrival-smoke-$(date +%Y%m%d-%H%M%S)
python tools/run_multi_sequence_cuda_graph_production_gate_remote.py \
  arrival-smoke \
  --run-tag "${RUN_TAG}"
```

Expected: baseline/candidate capacities match, at least one exact graph hit occurs, at least one non-allowlisted eager fallback occurs, and independent verification returns a complete smoke result. Smoke does not authorize a performance claim.

- [ ] **Step 4: Stop on any smoke failure**

If either smoke is incorrect or incomplete, record the exact failing artifact and do not run canonical performance. Fix only the demonstrated implementation or harness defect, rerun local validation, and repeat smoke with a new run tag.

### Task 10: Run Canonical Correctness and Arrival-Load Gates

**Files:**
- Raw artifacts only under `experiments/cuda_graph/`.

**Interfaces:**
- Consumes: passing smoke evidence.
- Produces: independently verified canonical `GO | NO_GO | INCOMPLETE`.

- [ ] **Step 1: Run the complete source-bound correctness gate**

```bash
CORRECTNESS_TAG=qwen3-06b-exact-cuda-graph-correctness-canonical-$(date +%Y%m%d-%H%M%S)
python tools/run_multi_sequence_cuda_graph_production_gate_remote.py \
  correctness-canonical \
  --run-tag "${CORRECTNESS_TAG}"
```

Expected: the complete 315-case diagnostic remains `EXACT_REPLAY_CORRECT`, `LEGACY_COMPATIBLE`, `POLICY_EXACT`, and `ROUNDED_REPLAY_CORRUPT`; actual `ModelRunner` correctness, page transitions, cold/capture/replay lifecycle, and all budget fallbacks pass.

- [ ] **Step 2: Stop if correctness is not complete and correct**

Do not run canonical arrival-load performance unless the independent correctness result is complete and correct.

- [ ] **Step 3: Run the paired canonical arrival-load gate**

```bash
ARRIVAL_TAG=qwen3-06b-exact-cuda-graph-arrival-canonical-$(date +%Y%m%d-%H%M%S)
python tools/run_multi_sequence_cuda_graph_production_gate_remote.py \
  arrival-canonical \
  --run-tag "${ARRIVAL_TAG}" \
  --diagnostic-run-tag "${CORRECTNESS_TAG}"
```

Expected: one warmup plus five isolated measured repetitions for all eight workloads and both policies, with randomized pair order and exact capacity parity.

- [ ] **Step 4: Re-run independent verification locally**

```bash
python tools/verify_multi_sequence_cuda_graph_production.py \
  --run-dir "experiments/cuda_graph/${ARRIVAL_TAG}" \
  --write-report
```

Expected: the local result exactly matches the stored independent result.

- [ ] **Step 5: Inspect the conjunctive decision**

Confirm all of:

```text
correctness complete and exact
aggregate decode throughput ratio >= 1.15
stable-exact decode throughput ratio >= 1.25
minimum request-throughput ratio >= 0.95
maximum p95 ITL ratio <= 1.05
maximum p99 ITL ratio <= 1.10
peak reserved-memory ratio <= 1.02
initialization-duration ratio <= 1.05
stable-exact hit rate >= 0.60
at least one allowlisted batch replays
at least two exact widths replay
at least one non-allowlisted batch falls back eager
no unknown fallback
no replay after rejection
all five measured repetitions complete
producer and independent summaries match
```

Any failed complete requirement is `NO_GO`; missing/unverifiable evidence is `INCOMPLETE`.

### Task 11: Record the Result Without Overclaiming

**Files:**
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify: `README.md` only on independent `GO`.

**Interfaces:**
- Consumes: canonical artifacts and independent classification.
- Produces: durable handoff, and only when authorized, measured README guidance.

- [ ] **Step 1: Append the canonical evidence to the handoff**

Record source commit/tree SHA, run tags, commands, exact classification, all threshold ratios, correctness result, capacity parity, replayed batches/widths, fallback counts, memory/startup results, limitations, and artifact paths.

- [ ] **Step 2: Apply the classification boundary**

For `NO_GO`, keep the feature default-off and document that the narrow candidate remains experimental and unsupported for README guidance. For `INCOMPLETE`, document the missing evidence and grant no production authorization.

For `GO` only, update `README.md` with:

- the default-off opt-in fields;
- exact supported model/GPU/FlashAttention/workload scope;
- measured ratios copied from independent evidence;
- the fail-closed eager fallback behavior;
- the fact that default-on rollout is out of scope.

- [ ] **Step 3: Validate documentation claims**

```bash
git diff --check
python tools/verify_multi_sequence_cuda_graph_production.py \
  --run-dir "experiments/cuda_graph/${ARRIVAL_TAG}" \
  --write-report
```

Expected: verifier result is unchanged and every README number is present in `independent_verification.json`.

- [ ] **Step 4: Selectively commit tracked documentation**

For `NO_GO` or `INCOMPLETE`:

```bash
git add AGENT_HANDOFF_STATE.md
git commit -m "docs: record exact cuda graph production gate"
```

For `GO`:

```bash
git add AGENT_HANDOFF_STATE.md README.md
git commit -m "docs: publish exact cuda graph production evidence"
```

Never stage `experiments/`.

---

## Completion Audit

Before claiming implementation completion, map every approved design item to direct evidence:

1. Default-off config: `Config` test and source line.
2. Disabled behavior unchanged: original multi-sequence eager test.
3. Batch-one fast path retained: batch-one replay test.
4. Exact heuristic identity: identity-field/SHA tests and raw dispatch rows.
5. No rounded batch/width replay: local miss tests and independent verifier.
6. Three eager observations before capture: lifecycle test and canonical dispatch sequence.
7. Scratch KV is non-live: visible/physical capacity test and scratch IDs.
8. Entry-private buffers: object-identity test and per-entry metadata.
9. Five fail-closed ceilings: five independent local tests and remote budget cases.
10. No eviction/recapture: terminal-state tests and absence of later capture events.
11. Auditable dispatch reason: complete event-field and closed-enum tests.
12. Local contracts pass: exact command output from Task 8.
13. 315-case diagnostic remains correct: canonical diagnostic artifacts.
14. Actual `ModelRunner` correctness: source-bound correctness rows.
15. Arrival-load thresholds: independently recomputed canonical ratios.
16. Independent verification matches: local rerun and stored result.
17. Broad guard remains authoritative: source inspection plus default/miss tests.
18. README boundary: zero diff before `GO`, or evidence-bound update after `GO`.

Do not treat passing unit tests, a producer summary, a manifest, or elapsed effort as completion by itself. Any missing row in this audit means the feature is not production-proven.
