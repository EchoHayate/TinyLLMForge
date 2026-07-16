# Native Multi-Token Verifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace profiler-owned prefill-tail verification plus accepted-KV decode replay with one eager, decode-equivalent multi-query target forward that writes accepted-prefix KV directly into final slots and is proven against a serialized single-token decode oracle.

**Architecture:** Add a pure verifier-contract module, an explicit `prefill` / `decode` / `spec_verify` context mode, and a single-sequence `ModelRunner.prepare_spec_verify()` boundary. Dispatch `spec_verify` through `flash_attn_with_kvcache` with query shape `[1, Q, num_heads, head_dim]`, retain the existing metadata-only `BlockManager.commit_accepted_tokens()` lifecycle, and keep the legacy rematerializing path only as a measured comparison. Build a separate row-expanded oracle and isolated remote gate that compare logits, KV, acceptance, metadata, 16-token continuation, stable baseline output, forward counts, and timing before classifying the result.

**Tech Stack:** Python 3 dataclasses and standard library, PyTorch, FlashAttention `flash_attn_with_kvcache`, existing paged KV cache and `BlockManager`, dependency-light test scripts, Bash, SSH/SCP, Qwen3-0.6B on the existing remote CUDA host.

## Global Constraints

- The normative design is `docs/superpowers/specs/2026-07-15-native-multi-token-verifier-design.md`.
- Modify only the isolated worktree `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`; do not modify `/Users/bytedance/dev/TinyLLMForge`.
- Execute this plan inline in the current session; do not dispatch subagents.
- The implementation remains profiler-owned, greedy, single-sequence, linear-draft, eager-only, and FP16/BF16-KV-only.
- `spec_verify` is an explicit context mode; attention dispatch must not infer it from `is_prefill=False` plus auxiliary tensors.
- For history length `H`, draft `d[0:K]`, and `Q=max(0,K-1)`, use `input_ids=d[0:Q]`, model positions `[H+1,...,H+Q]`, logical slots `[H,...,H+Q-1]`, `query_lens=[Q]`, and `context_lens=[H+Q]`.
- `K=1` performs the existing first-target decode and no tail verifier forward.
- The native path writes verifier K/V directly to final current/reserved slots and must not copy, replay, or recompute accepted-token KV.
- `BlockManager.commit_accepted_tokens()` remains metadata-only and leaves the final accepted token pending through `materialized_tokens=final_len-1`.
- Unsupported KV offload, blockwise attention, Quest, Attention Matching, KV cartridge, C4/C8 KV, mixed prefill/decode, CUDA graph, multi-sequence, non-linear draft, or non-greedy use must fail before verifier KV mutation.
- The row-expanded oracle must use isolated cache state and established single-token decode; it is never included in performance conclusions.
- Exact token equality is a hard gate. Numerical tolerances diagnose logit/KV drift and never excuse token, acceptance, lifecycle, or continuation mismatch.
- Required draft lengths are `K in {1,4,8,16}`; required acceptance cases are zero, one, partial, and full, plus EOS, output-budget truncation, current-block, one-new-block, multiple-new-block, and rollback boundaries.
- Every native event must report `decode_calls=0`, `rematerialized_tokens=[]`, and `accepted_kv_rematerialize_ms=0`.
- GPU/model experiments run only on `sitian@10.232.195.203` with Python `/data00/home/sitian/sitian-workspace01/tllm/env/bin/python` and model `/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B`.
- Every remote run uploads a source snapshot to a unique directory, leaves the original remote checkout untouched, and allocates distinct dynamic `TINYVLLM_DIST_PORT` and `MASTER_PORT` values for every model process.
- Do not claim production batching, ragged verification, tree verification, non-greedy equivalence, queue-tail improvement, CUDA graph support, KV-offload support, quantized-KV support, memory reduction, multi-model generalization, or production readiness.
- A passing first-phase smoke may classify only as `READY_FOR_PERFORMANCE_GATE`; it is not a production `GO`.
- Important commands, exact results, limitations, and next direction must be written to `README.md` and `AGENT_HANDOFF_STATE.md`.

## File Structure

- Create `tinyvllm/speculative/verifier.py`: pure tensor-plan contract, immutable host metadata, supported-mode validation helpers, and JSON conversion.
- Create `tools/test_native_verifier_contract.py`: dependency-light contract, compatibility, `Q=0`, slot-boundary, and invalid-input tests.
- Modify `tinyvllm/utils/context.py`: explicit attention mode, legacy compatibility, mode validation, and reset behavior.
- Create `tools/test_context_modes.py`: explicit-mode, legacy caller, mismatch, and reset tests.
- Modify `tinyvllm/engine/model_runner.py`: `prepare_spec_verify()`, eager `spec_verify` forward, final-slot tensor setup, and verifier metadata return.
- Create `tools/test_model_runner_spec_verify.py`: fake-runner preparation, fail-closed-before-upload, eager dispatch, and all-logit-row tests.
- Modify `tinyvllm/layers/attention.py`: dedicated multi-query KV-cache branch and a small dispatch helper that is independently testable.
- Create `tools/test_native_verifier_attention.py`: deterministic multi-query versus serialized decode, future-row masking, block-boundary, GQA, FP16, and BF16 tests.
- Modify `tools/profile_ngram_commit.py`: native verifier mode, structured event schema, no-rematerialization path, rollback ownership, and legacy comparator retention.
- Modify `tools/test_ngram_speculative.py`: native lifecycle, acceptance/truncation, rollback, instrumentation, and legacy-regression tests.
- Modify `tools/test_chunked_prefill.py`: final-block hash publication and reserved-block visibility regressions for native-originated accepted prefixes.
- Create `tools/native_verifier_oracle.py`: isolated serialized-decode oracle, KV snapshot comparison, continuation comparison, and dtype tolerances.
- Create `tools/test_native_verifier_oracle.py`: synthetic oracle/native comparison and mismatch classification tests.
- Create `tools/native_verifier_gate.py`: manifest, deterministic case matrix, isolated process driver, event reconciliation, gate classification, artifact verification, and report rendering.
- Create `tools/test_native_verifier_gate.py`: complete `READY_FOR_PERFORMANCE_GATE`, semantic `NO_GO`, infrastructure `INCOMPLETE`, rematerialization, and artifact-tamper tests.
- Create `tools/run_native_verifier_gate_remote.sh`: isolated upload, FlashAttention capability run, exactness matrix, legacy timing comparison, artifact download, SHA-256 verification, and local verifier execution.
- Create `experiments/native_verifier/${RUN_TAG}/{manifest.json,capability.json,case_rows.json,event_rows.json,summary.json,report.md}` only after the remote runner has assigned `RUN_TAG=qwen3-06b-$(date +%Y%m%d-%H%M%S)-$$`.
- Modify `README.md`: implementation command, remote command, measured classification, evidence paths, and claim boundaries.
- Modify `AGENT_HANDOFF_STATE.md`: source SHA, local/remote paths, exact commands, gate evidence, unresolved items, and the next written performance-gate requirement.

---

### Task 1: Pure Verifier Tensor Contract

**Files:**
- Create: `tinyvllm/speculative/verifier.py`
- Create: `tools/test_native_verifier_contract.py`

**Interfaces:**
- Produces: `AttentionMode = Literal["prefill", "decode", "spec_verify"]`
- Produces: `SpecVerifyPlan(input_tokens, positions, logical_slots, context_len, visible_block_count)`
- Produces: `SpecVerifyMetadata(query_len, input_tokens, positions, logical_slots, physical_slots, context_len, block_table)`
- Produces: `build_spec_verify_plan(history_len: int, draft_tokens: list[int], block_size: int) -> SpecVerifyPlan`
- Produces: `validate_spec_verify_slots(plan: SpecVerifyPlan, proxy_block_table: list[int], block_size: int) -> tuple[int, ...]`
- Produces: `spec_verify_metadata_to_dict(metadata: SpecVerifyMetadata) -> dict[str, object]`

- [ ] **Step 1: Write failing tensor-contract tests**

Create `tools/test_native_verifier_contract.py` with direct imports and exact cases:

```python
from __future__ import annotations

from tinyvllm.speculative.verifier import (
    SpecVerifyMetadata,
    build_spec_verify_plan,
    spec_verify_metadata_to_dict,
    validate_spec_verify_slots,
)


def test_reference_h52_k4_contract():
    plan = build_spec_verify_plan(
        history_len=52,
        draft_tokens=[10, 20, 30, 40],
        block_size=256,
    )
    assert plan.input_tokens == (10, 20, 30)
    assert plan.positions == (53, 54, 55)
    assert plan.logical_slots == (52, 53, 54)
    assert plan.context_len == 55
    assert plan.visible_block_count == 1


def test_k1_has_zero_tail_queries():
    plan = build_spec_verify_plan(
        history_len=52,
        draft_tokens=[10],
        block_size=256,
    )
    assert plan.query_len == 0
    assert plan.input_tokens == ()
    assert plan.positions == ()
    assert plan.logical_slots == ()
    assert plan.context_len == 52


def test_required_k_values_have_consecutive_positions_and_slots():
    for draft_len in (1, 4, 8, 16):
        draft = list(range(100, 100 + draft_len))
        plan = build_spec_verify_plan(255, draft, block_size=256)
        assert plan.query_len == max(0, draft_len - 1)
        assert plan.positions == tuple(range(256, 256 + plan.query_len))
        assert plan.logical_slots == tuple(range(255, 255 + plan.query_len))
        assert plan.visible_block_count == (plan.context_len + 255) // 256


def test_slot_validation_maps_current_and_reserved_blocks():
    plan = build_spec_verify_plan(255, [1, 2, 3, 4], block_size=256)
    assert validate_spec_verify_slots(plan, [7, 11], 256) == (
        7 * 256 + 255,
        11 * 256,
        11 * 256 + 1,
    )


def test_invalid_contract_inputs_fail():
    invalid_calls = (
        lambda: build_spec_verify_plan(-1, [1], 256),
        lambda: build_spec_verify_plan(4, [], 256),
        lambda: build_spec_verify_plan(4, [1], 0),
        lambda: validate_spec_verify_slots(
            build_spec_verify_plan(255, [1, 2, 3, 4], 256),
            [7],
            256,
        ),
    )
    for call in invalid_calls:
        try:
            call()
        except ValueError:
            pass
        else:
            raise AssertionError("invalid verifier contract must fail")


def test_metadata_is_json_friendly():
    metadata = SpecVerifyMetadata(
        query_len=3,
        input_tokens=(10, 20, 30),
        positions=(53, 54, 55),
        logical_slots=(52, 53, 54),
        physical_slots=(52, 53, 54),
        context_len=55,
        block_table=(0,),
    )
    assert spec_verify_metadata_to_dict(metadata) == {
        "query_len": 3,
        "input_tokens": [10, 20, 30],
        "positions": [53, 54, 55],
        "logical_slots": [52, 53, 54],
        "physical_slots": [52, 53, 54],
        "context_len": 55,
        "block_table": [0],
    }
```

Add a `main()` that invokes every test and prints `native verifier contract tests passed`.

- [ ] **Step 2: Run the contract test and verify the missing module fails**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_native_verifier_contract.py
```

Expected: import failure for `tinyvllm.speculative.verifier`.

- [ ] **Step 3: Implement immutable plans and validation**

Create `tinyvllm/speculative/verifier.py`:

```python
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


AttentionMode = Literal["prefill", "decode", "spec_verify"]


@dataclass(frozen=True)
class SpecVerifyPlan:
    input_tokens: tuple[int, ...]
    positions: tuple[int, ...]
    logical_slots: tuple[int, ...]
    context_len: int
    visible_block_count: int

    @property
    def query_len(self) -> int:
        return len(self.input_tokens)


@dataclass(frozen=True)
class SpecVerifyMetadata:
    query_len: int
    input_tokens: tuple[int, ...]
    positions: tuple[int, ...]
    logical_slots: tuple[int, ...]
    physical_slots: tuple[int, ...]
    context_len: int
    block_table: tuple[int, ...]


def build_spec_verify_plan(
    history_len: int,
    draft_tokens: list[int],
    block_size: int,
) -> SpecVerifyPlan:
    history_len = int(history_len)
    block_size = int(block_size)
    if history_len < 1:
        raise ValueError("spec_verify requires history_len >= 1")
    if not draft_tokens:
        raise ValueError("spec_verify requires at least one draft token")
    if block_size <= 0:
        raise ValueError("block_size must be > 0")
    input_tokens = tuple(int(token_id) for token_id in draft_tokens[:-1])
    query_len = len(input_tokens)
    positions = tuple(range(history_len + 1, history_len + 1 + query_len))
    logical_slots = tuple(range(history_len, history_len + query_len))
    context_len = history_len + query_len
    visible_block_count = (
        (context_len + block_size - 1) // block_size
        if context_len > 0
        else 0
    )
    return SpecVerifyPlan(
        input_tokens=input_tokens,
        positions=positions,
        logical_slots=logical_slots,
        context_len=context_len,
        visible_block_count=visible_block_count,
    )


def validate_spec_verify_slots(
    plan: SpecVerifyPlan,
    proxy_block_table: list[int],
    block_size: int,
) -> tuple[int, ...]:
    block_size = int(block_size)
    if block_size <= 0:
        raise ValueError("block_size must be > 0")
    if len(proxy_block_table) < plan.visible_block_count:
        raise ValueError("proxy block table does not cover verifier context")
    physical_slots = []
    for logical_slot in plan.logical_slots:
        block_index = logical_slot // block_size
        if block_index >= len(proxy_block_table):
            raise ValueError("logical verifier slot is out of range")
        block_id = int(proxy_block_table[block_index])
        if block_id < 0:
            raise ValueError("verifier block table contains an invalid block")
        physical_slots.append(
            block_id * block_size + logical_slot % block_size
        )
    return tuple(physical_slots)


def spec_verify_metadata_to_dict(
    metadata: SpecVerifyMetadata,
) -> dict[str, object]:
    payload = asdict(metadata)
    return {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in payload.items()
    }
```

- [ ] **Step 4: Run the contract test**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_native_verifier_contract.py
```

Expected: `native verifier contract tests passed`.

- [ ] **Step 5: Commit the pure contract**

```bash
git add tinyvllm/speculative/verifier.py tools/test_native_verifier_contract.py
git commit -m "Add native verifier tensor contract"
```

---

### Task 2: Explicit Context Modes

**Files:**
- Modify: `tinyvllm/utils/context.py`
- Create: `tools/test_context_modes.py`

**Interfaces:**
- Consumes: `AttentionMode`
- Produces: `Context.mode: AttentionMode`
- Produces: `Context.is_prefill: bool` retained for transition compatibility
- Produces: the existing full `set_context()` signature with `is_prefill: bool | None = None` as its first argument and `mode: AttentionMode | None = None` appended as its final argument.
- Produces: `resolve_attention_mode(is_prefill: bool | None, mode: AttentionMode | None) -> AttentionMode`

- [ ] **Step 1: Write failing explicit-mode and legacy tests**

Create `tools/test_context_modes.py`:

```python
from tinyvllm.utils.context import (
    get_context,
    reset_context,
    resolve_attention_mode,
    set_context,
)


def test_explicit_modes_are_preserved():
    for mode, expected_prefill in (
        ("prefill", True),
        ("decode", False),
        ("spec_verify", False),
    ):
        set_context(mode=mode)
        context = get_context()
        assert context.mode == mode
        assert context.is_prefill is expected_prefill


def test_legacy_boolean_callers_keep_current_behavior():
    set_context(True)
    assert get_context().mode == "prefill"
    set_context(False)
    assert get_context().mode == "decode"


def test_conflicting_mode_and_boolean_fail():
    for is_prefill, mode in ((True, "decode"), (False, "prefill"), (True, "spec_verify")):
        try:
            resolve_attention_mode(is_prefill, mode)
        except ValueError as exc:
            assert "conflicting attention mode" in str(exc)
        else:
            raise AssertionError((is_prefill, mode))


def test_reset_context_returns_decode_default():
    set_context(mode="spec_verify")
    reset_context()
    assert get_context().mode == "decode"
    assert get_context().is_prefill is False
```

Add a `main()` that runs all tests and prints `context mode tests passed`.

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_context_modes.py
```

Expected: failure because `Context.mode` and `resolve_attention_mode()` do not exist.

- [ ] **Step 3: Add explicit mode while preserving old call sites**

Modify `tinyvllm/utils/context.py` to import `AttentionMode`, add `mode`, and resolve legacy calls:

```python
from tinyvllm.speculative.verifier import AttentionMode


def resolve_attention_mode(
    is_prefill: bool | None,
    mode: AttentionMode | None,
) -> AttentionMode:
    if mode is None:
        return "prefill" if bool(is_prefill) else "decode"
    if mode not in ("prefill", "decode", "spec_verify"):
        raise ValueError(f"unsupported attention mode: {mode}")
    expected_prefill = mode == "prefill"
    if is_prefill is not None and bool(is_prefill) != expected_prefill:
        raise ValueError(
            f"conflicting attention mode: is_prefill={is_prefill}, mode={mode}"
        )
    return mode
```

Define the first fields of `Context` as:

```python
@dataclass
class Context:
    mode: AttentionMode = "decode"
    is_prefill: bool = False
```

Change `set_context()` so `is_prefill` defaults to `None`, `mode` is a final keyword argument, and construction begins with:

```python
resolved_mode = resolve_attention_mode(is_prefill, mode)
_CONTEXT = Context(
    mode=resolved_mode,
    is_prefill=resolved_mode == "prefill",
    cu_seqlens_q=cu_seqlens_q,
    cu_seqlens_k=cu_seqlens_k,
    max_seqlen_q=max_seqlen_q,
    max_seqlen_k=max_seqlen_k,
    slot_mapping=slot_mapping,
    context_lens=context_lens,
    block_tables=block_tables,
    logits_indices=logits_indices,
)
```

Continue passing every existing Quest, Attention Matching, and KV-offload keyword field into `Context` exactly as the current function does. Do not change the order of existing positional tensor arguments.

- [ ] **Step 4: Run context and existing focused tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_context_modes.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_ngram_speculative.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_chunked_prefill.py
```

Expected: all three scripts pass.

- [ ] **Step 5: Commit explicit context modes**

```bash
git add tinyvllm/utils/context.py tools/test_context_modes.py
git commit -m "Add explicit verifier attention mode"
```

---

### Task 3: ModelRunner Preparation and Eager Forward

**Files:**
- Modify: `tinyvllm/engine/model_runner.py`
- Create: `tools/test_model_runner_spec_verify.py`

**Interfaces:**
- Consumes: `SpecVerifyPlan`, `SpecVerifyMetadata`, `validate_spec_verify_slots()`
- Produces: `ModelRunner._validate_spec_verify_compatibility(seq_count: int, linear_draft: bool, greedy: bool, mixed_batch: bool) -> None`
- Produces: `ModelRunner.prepare_spec_verify(seq: Sequence, input_tokens: list[int], proxy_block_table: list[int], slot_positions: list[int]) -> tuple[torch.Tensor, torch.Tensor, SpecVerifyMetadata]`
- Extends: `ModelRunner.run_model(..., execution_mode: AttentionMode | None = None)`
- Guarantees: `execution_mode="spec_verify"` always runs eager and returns all `Q` logits rows.

- [ ] **Step 1: Write failing preparation tests with a fake tensor uploader**

Create `tools/test_model_runner_spec_verify.py` without constructing a model:

```python
from __future__ import annotations

from types import SimpleNamespace

from tinyvllm.engine.model_runner import ModelRunner
from tinyvllm.utils.context import get_context, reset_context


class FakeTensor:
    def __init__(self, values):
        self.values = values


def make_runner(**overrides):
    runner = object.__new__(ModelRunner)
    runner.block_size = 256
    runner.kv_offload = None
    runner.enforce_eager = False
    config = {
        "kv_quant_bits": 0,
        "kv_offload_mvp0": False,
        "kv_offload_blockwise_decode": False,
        "kv_offload_blockwise_prefill": False,
        "quest_top_k_blocks": -1,
        "am_compact_blocks": 0,
        "kv_cartridge_blocks": 0,
        "chunked_prefill_mixed_batch": False,
        "cpu_offload": False,
    }
    config.update(overrides)
    runner.config = SimpleNamespace(**config)
    runner._list_to_cuda = lambda data, name, dtype: FakeTensor(list(data))
    runner.prepare_block_tables_from_rows = (
        lambda rows, name="block_tables": FakeTensor([list(row) for row in rows])
    )
    return runner


def test_prepare_spec_verify_installs_reference_context():
    runner = make_runner()
    seq = SimpleNamespace(block_size=256)
    input_ids, positions, metadata = runner.prepare_spec_verify(
        seq,
        input_tokens=[10, 20, 30],
        proxy_block_table=[0],
        slot_positions=[52, 53, 54],
    )
    context = get_context()
    assert input_ids.values == [10, 20, 30]
    assert positions.values == [53, 54, 55]
    assert metadata.query_len == 3
    assert metadata.logical_slots == (52, 53, 54)
    assert metadata.physical_slots == (52, 53, 54)
    assert metadata.context_len == 55
    assert context.mode == "spec_verify"
    assert context.context_lens.values == [55]
    assert context.block_tables.values == [[0]]


def test_prepare_spec_verify_rejects_nonconsecutive_slots_before_upload():
    runner = make_runner()
    runner._list_to_cuda = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("upload must not run")
    )
    try:
        runner.prepare_spec_verify(
            SimpleNamespace(block_size=256),
            input_tokens=[10, 20],
            proxy_block_table=[0],
            slot_positions=[52, 54],
        )
    except ValueError as exc:
        assert "consecutive" in str(exc)
    else:
        raise AssertionError("nonconsecutive slots must fail")


def test_every_unsupported_feature_fails_closed():
    unsupported = {
        "kv_quant_bits": 4,
        "kv_offload_mvp0": True,
        "kv_offload_blockwise_decode": True,
        "kv_offload_blockwise_prefill": True,
        "quest_top_k_blocks": 1,
        "am_compact_blocks": 1,
        "kv_cartridge_blocks": 1,
        "chunked_prefill_mixed_batch": True,
    }
    for name, value in unsupported.items():
        runner = make_runner(**{name: value})
        try:
            runner._validate_spec_verify_compatibility(
                seq_count=1,
                linear_draft=True,
                greedy=True,
                mixed_batch=False,
            )
        except RuntimeError as exc:
            assert name in str(exc)
        else:
            raise AssertionError(name)


def test_multi_sequence_nonlinear_and_nongreedy_fail():
    runner = make_runner()
    invalid = (
        dict(seq_count=2, linear_draft=True, greedy=True, mixed_batch=False),
        dict(seq_count=1, linear_draft=False, greedy=True, mixed_batch=False),
        dict(seq_count=1, linear_draft=True, greedy=False, mixed_batch=False),
        dict(seq_count=1, linear_draft=True, greedy=True, mixed_batch=True),
    )
    for arguments in invalid:
        try:
            runner._validate_spec_verify_compatibility(**arguments)
        except RuntimeError:
            pass
        else:
            raise AssertionError(arguments)
```

Reset context after each test in `main()` and print `model runner spec_verify tests passed`.

- [ ] **Step 2: Run the preparation test and verify it fails**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_model_runner_spec_verify.py
```

Expected: failure because `prepare_spec_verify()` does not exist.

- [ ] **Step 3: Implement fail-closed validation and preparation**

In `tinyvllm/engine/model_runner.py`, import the verifier contract and add:

```python
    def _validate_spec_verify_compatibility(
        self,
        *,
        seq_count: int,
        linear_draft: bool,
        greedy: bool,
        mixed_batch: bool,
    ) -> None:
        if seq_count != 1:
            raise RuntimeError("spec_verify requires exactly one sequence")
        if not linear_draft:
            raise RuntimeError("spec_verify requires a linear draft")
        if not greedy:
            raise RuntimeError("spec_verify requires greedy acceptance")
        if mixed_batch or self.config.chunked_prefill_mixed_batch:
            raise RuntimeError("chunked_prefill_mixed_batch is unsupported by spec_verify")
        unsupported = (
            ("kv_quant_bits", self.config.kv_quant_bits != 0),
            ("kv_offload_mvp0", self.config.kv_offload_mvp0),
            ("kv_offload_blockwise_decode", self.config.kv_offload_blockwise_decode),
            ("kv_offload_blockwise_prefill", self.config.kv_offload_blockwise_prefill),
            ("quest_top_k_blocks", self.config.quest_top_k_blocks > 0),
            ("am_compact_blocks", self.config.am_compact_blocks > 0),
            ("kv_cartridge_blocks", self.config.kv_cartridge_blocks > 0),
        )
        for name, active in unsupported:
            if active:
                raise RuntimeError(f"{name} is unsupported by spec_verify")
```

Implement `prepare_spec_verify()` so it:

1. validates compatibility before any `_list_to_cuda()` call;
2. requires non-empty input tokens because `K=1` bypasses this method;
3. requires `slot_positions` to be consecutive and equal in length to inputs;
4. derives positions as `slot+1`;
5. validates and maps final slots through the proxy block table;
6. installs `mode="spec_verify"` with one `context_lens` row and one block-table row;
7. returns all host metadata in `SpecVerifyMetadata`.

Use this context setup:

```python
set_context(
    mode="spec_verify",
    slot_mapping=slot_mapping,
    context_lens=context_lens,
    block_tables=block_tables,
)
```

- [ ] **Step 4: Force `spec_verify` eager execution**

Extend `run_model()`:

```python
    def run_model(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        is_prefill: bool,
        input_embeds: torch.Tensor | None = None,
        return_hidden: bool = False,
        execution_mode: AttentionMode | None = None,
    ):
        mode = execution_mode or get_context().mode
        if mode == "spec_verify" and is_prefill:
            raise ValueError("spec_verify cannot use prefill execution")
        spec_verify_active = mode == "spec_verify"
        quest_active = mode == "decode" and get_context().quest_top_k_blocks > 0
        am_active = mode == "decode" and get_context().am_compact_blocks > 0
        c4_active = self.config.kv_quant_bits == 4
        offload_active = self.config.cpu_offload
        kv_offload_active = self.config.kv_offload_mvp0
        eager = (
            is_prefill
            or spec_verify_active
            or self.enforce_eager
            or input_ids.size(0) > 512
            or quest_active
            or am_active
            or c4_active
            or offload_active
            or kv_offload_active
            or input_embeds is not None
            or return_hidden
        )
        if eager:
            hidden_states = self.model(
                input_ids,
                positions,
                input_embeds=input_embeds,
            )
            logits = self.model.compute_logits(hidden_states)
            if return_hidden:
                return logits, hidden_states
            return logits
```

Keep the current CUDA-graph `else` branch unchanged after this eager branch. Do not select or discard logit rows in `run_model()`.

- [ ] **Step 5: Add an eager-path regression test**

Extend `tools/test_model_runner_spec_verify.py` with a fake model and a `graphs` object that raises if accessed:

```python
def test_spec_verify_run_model_uses_eager_and_keeps_all_rows():
    runner = make_runner()
    class FakeModel:
        def __call__(self, input_ids, positions, input_embeds=None):
            return FakeTensor([[1], [2], [3]])

        def compute_logits(self, hidden):
            return hidden

    runner.model = FakeModel()
    runner.graphs = {"forbidden": True}
    logits = runner.run_model(
        FakeTensor([10, 20, 30]),
        FakeTensor([53, 54, 55]),
        is_prefill=False,
        execution_mode="spec_verify",
    )
    assert logits.values == [[1], [2], [3]]
```

Adapt `FakeTensor` only as needed to satisfy the eager branch; do not mock a graph replay.

- [ ] **Step 6: Run focused tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_model_runner_spec_verify.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_context_modes.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_ngram_speculative.py
```

Expected: all pass.

- [ ] **Step 7: Commit ModelRunner support**

```bash
git add tinyvllm/engine/model_runner.py tools/test_model_runner_spec_verify.py
git commit -m "Prepare eager native verifier batches"
```

---

### Task 4: Multi-Query KV-Cache Attention Dispatch

**Files:**
- Modify: `tinyvllm/layers/attention.py`
- Create: `tools/test_native_verifier_attention.py`

**Interfaces:**
- Produces: `_flash_attn_spec_verify(q, k_cache, v_cache, context, scale) -> torch.Tensor`
- Dispatches: `Context.mode == "spec_verify"` before ordinary decode optional-feature logic.
- Guarantees: query shape `[1,Q,num_heads,head_dim]`, `cache_seqlens=[H+Q]`, one paged block-table row, `causal=True`.

- [ ] **Step 1: Write a dispatch-shape test with a patched FlashAttention function**

Create `tools/test_native_verifier_attention.py`:

```python
from __future__ import annotations

from types import SimpleNamespace

import torch

import tinyvllm.layers.attention as attention_module


def test_spec_verify_helper_uses_single_multi_query_row():
    captured = {}

    def fake_flash(q, k_cache, v_cache, **kwargs):
        captured["q_shape"] = tuple(q.shape)
        captured["cache_seqlens"] = kwargs["cache_seqlens"].tolist()
        captured["block_table"] = kwargs["block_table"].tolist()
        captured["causal"] = kwargs["causal"]
        return q

    original = attention_module.flash_attn_with_kvcache
    attention_module.flash_attn_with_kvcache = fake_flash
    try:
        q = torch.zeros(3, 4, 8)
        cache = torch.zeros(2, 256, 2, 8)
        context = SimpleNamespace(
            context_lens=torch.tensor([55], dtype=torch.int32),
            block_tables=torch.tensor([[0]], dtype=torch.int32),
        )
        output = attention_module._flash_attn_spec_verify(
            q, cache, cache, context, 0.125
        )
    finally:
        attention_module.flash_attn_with_kvcache = original
    assert tuple(output.shape) == (3, 4, 8)
    assert captured == {
        "q_shape": (1, 3, 4, 8),
        "cache_seqlens": [55],
        "block_table": [[0]],
        "causal": True,
    }
```

- [ ] **Step 2: Run the test and verify the helper is missing**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_native_verifier_attention.py
```

Expected: failure because `_flash_attn_spec_verify()` does not exist.

- [ ] **Step 3: Implement the dedicated helper and dispatch branch**

Add:

```python
def _flash_attn_spec_verify(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    context,
    scale: float,
) -> torch.Tensor:
    if context.context_lens is None or context.context_lens.numel() != 1:
        raise RuntimeError("spec_verify requires one context length")
    if context.block_tables is None or context.block_tables.size(0) != 1:
        raise RuntimeError("spec_verify requires one block-table row")
    output = flash_attn_with_kvcache(
        q.unsqueeze(0),
        k_cache,
        v_cache,
        cache_seqlens=context.context_lens,
        block_table=context.block_tables,
        softmax_scale=scale,
        causal=True,
    )
    return output.view_as(q)
```

In `Attention.forward()`, retain the existing K/V write before dispatch, then branch:

```python
        if context.mode == "spec_verify":
            if self.kv_quant_bits != 0:
                raise RuntimeError("spec_verify requires FP16/BF16 KV")
            o = _flash_attn_spec_verify(
                q, k_cache, v_cache, context, self.scale
            )
```

Immediately after this branch, change the current `if context.is_prefill:` to `elif context.mode == "prefill":`, retain its full body unchanged, change its current `else:` to `elif context.mode == "decode":`, retain that full body unchanged, and finish with:

```python
        else:
            raise RuntimeError(f"unsupported attention mode: {context.mode}")
```

The `spec_verify` branch must occur before Quest, Attention Matching, blockwise, and KV-quant decode branches.

- [ ] **Step 4: Add deterministic CUDA numerical tests**

Extend `tools/test_native_verifier_attention.py` with a CUDA guard:

```python
def _cuda_flash_available():
    return torch.cuda.is_available() and hasattr(
        attention_module, "flash_attn_with_kvcache"
    )
```

For each supported dtype in `(torch.float16, torch.bfloat16)`, each `Q` in `(1,3,7,15)`, and both `(prefix_len, block_size)=(31,256)` and `(255,256)`:

1. seed PyTorch with `20260715 + Q`;
2. create one prefix KV cache, one query tensor, and `Q` new K/V rows;
3. clone the prefix into native and serialized caches;
4. write native K/V to consecutive slots and call `_flash_attn_spec_verify()` once;
5. serialize `Q` calls with query shape `[1,1,H,D]`, increasing `cache_seqlens`;
6. compare output with `torch.testing.assert_close()` using FP16 `rtol=2e-3, atol=2e-3` and BF16 `rtol=8e-3, atol=8e-3`;
7. compare written K/V exactly because both paths receive the same projected K/V;
8. perturb future K/V rows and assert earlier query outputs do not change, proving each verifier query cannot attend to future verifier rows;
9. use `num_heads=4`, `num_kv_heads=2` to cover GQA.

When CUDA or BF16 is unavailable, print a clear skip line and leave the remote capability gate responsible for the hard decision.

- [ ] **Step 5: Run local dispatch tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_native_verifier_attention.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_context_modes.py
```

Expected: CPU dispatch test passes; CUDA numerical cases either pass or explicitly skip.

- [ ] **Step 6: Commit attention support**

```bash
git add tinyvllm/layers/attention.py tools/test_native_verifier_attention.py
git commit -m "Dispatch native multi-query verifier attention"
```

---

### Task 5: Native Verify-and-Commit Lifecycle

**Files:**
- Modify: `tools/profile_ngram_commit.py`
- Modify: `tools/test_ngram_speculative.py`
- Modify: `tools/test_chunked_prefill.py`

**Interfaces:**
- Consumes: `ModelRunner.prepare_spec_verify()`
- Produces: `VerifierMode = Literal["legacy_rematerialize", "native"]`
- Extends: `verify_and_commit_block(..., verifier_mode: str = "legacy_rematerialize") -> dict`
- Produces: `_empty_rematerialization_event() -> dict`
- Produces: native event fields `verifier_mode`, `query_len`, positions/slots/context/block table, reserved/committed/released IDs, truncation flags, phase, forward counts, and zero rematerialization.

- [ ] **Step 1: Write failing native no-rematerialization test**

Add to `tools/test_ngram_speculative.py` a fake runner/block manager whose native tail returns controlled logits:

```python
def test_native_verify_commits_without_decode_rematerialization():
    llm, seq = make_native_verify_fixture(
        history=[1, 2, 3],
        first_target=4,
        tail_targets=[5, 6],
        eos=99,
        max_tokens=16,
        block_size=4,
    )
    event = profile_ngram.verify_and_commit_block(
        llm,
        seq,
        [4, 5, 6],
        draft_source="ngram",
        verifier_mode="native",
    )
    assert event["accepted_tokens"] == [4, 5, 6]
    assert event["verifier_mode"] == "native"
    assert event["query_len"] == 2
    assert event["accepted_kv_rematerialization"] == {
        "rematerialized_tokens": [],
        "decode_calls": 0,
        "elapsed_ms": 0.0,
    }
    assert event["timing_ms"]["accepted_kv_rematerialize_ms"] == 0.0
    assert llm.model_runner.normal_decode_calls == 1
    assert llm.model_runner.spec_verify_calls == 1
```

The fixture must record sequence metadata before and after each phase and must raise if `rematerialize_accepted_kv()` is invoked.

- [ ] **Step 2: Add failing acceptance and pending-token matrix**

Add table-driven cases for:

```python
cases = (
    ("zero", [9, 5, 6], 0),
    ("one", [4, 9, 6], 1),
    ("partial", [4, 5, 9], 2),
    ("full", [4, 5, 6], 3),
)
```

For each case assert:

- sequence length grows by exactly the accepted count;
- zero acceptance publishes no blocks and releases all reservations;
- one acceptance performs no tail forward when `K=1`;
- the final accepted token remains pending;
- only blocks covered by `materialized_tokens=final_len-1` become visible;
- unused reserved blocks are released.

- [ ] **Step 3: Add failing EOS, budget, and block-boundary tests**

Cover:

- EOS as the first, middle, and final accepted token;
- remaining output budgets `0`, `1`, and less than the greedy accepted prefix;
- current-block-only writes;
- exactly one newly reserved block;
- multiple newly reserved blocks;
- exact full-block hash publication at `materialized_tokens` boundaries.

Use the real `BlockManager` for block/hash assertions and a fake model runner for logits.

- [ ] **Step 4: Add failing rollback tests for every phase**

Inject failures:

```python
failure_phases = (
    "before_tail_forward",
    "after_tail_kv_write",
    "during_acceptance",
    "during_metadata_commit",
    "new_reserved_block_boundary",
)
```

For each failure assert:

- original sequence tokens and block table are unchanged;
- all still-owned reserved blocks return to `free_block_ids`;
- no prefix hash is published;
- context resets to `mode=="decode"`;
- stale unpublished KV is tolerated but never made visible;
- event/error text includes the failing phase.

- [ ] **Step 5: Implement native branching and ownership accounting**

Refactor `verify_and_commit_block()` so:

1. unsupported native mode is validated before `reserve_append_blocks()`;
2. reservations remain in a local `owned_reserved_blocks` list;
3. the normal first-target decode runs once;
4. `K=1` bypasses `prepare_spec_verify()` and tail forward;
5. `K>1` calls:

```python
input_ids, positions, verifier_metadata = (
    llm.model_runner.prepare_spec_verify(
        seq,
        input_tokens=tail_plan["input_tokens"],
        proxy_block_table=proxy_block_table,
        slot_positions=tail_plan["slot_positions"],
    )
)
logits = llm.model_runner.run_model(
    input_ids,
    positions,
    is_prefill=False,
    execution_mode="spec_verify",
)
```

6. native acceptance uses all `logits.argmax(dim=-1)` rows;
7. native mode uses:

```python
def _empty_rematerialization_event() -> dict:
    return {
        "rematerialized_tokens": [],
        "decode_calls": 0,
        "elapsed_ms": 0.0,
    }
```

8. legacy mode retains the existing prefill tail and `rematerialize_accepted_kv()` for timing comparison only;
9. after successful `commit_accepted_tokens()`, ownership transfers and the local owned list becomes empty;
10. exception cleanup releases only still-owned blocks;
11. context resets in `finally`.

- [ ] **Step 6: Emit the full structured native event**

Every native event must include:

```python
{
    "verifier_mode": "native",
    "query_len": verifier_metadata.query_len,
    "history_len": history_len,
    "draft_len": len(draft_tokens),
    "input_tokens": list(verifier_metadata.input_tokens),
    "positions": list(verifier_metadata.positions),
    "logical_slots": list(verifier_metadata.logical_slots),
    "physical_slots": list(verifier_metadata.physical_slots),
    "context_len": verifier_metadata.context_len,
    "proxy_block_table": list(verifier_metadata.block_table),
    "reserved_blocks": event_reserved_blocks,
    "committed_blocks": committed_blocks,
    "released_blocks": released_blocks,
    "target_tokens": target_tokens,
    "accepted_tokens": accepted_tokens,
    "accepted_count": len(accepted_tokens),
    "eos_truncated": eos_truncated,
    "output_budget_truncated": output_budget_truncated,
    "target_forward_count": 1 + int(query_len > 0),
    "accepted_kv_rematerialization": _empty_rematerialization_event(),
}
```

Timing must include `reserve_blocks_ms`, `decode_first_target_ms`, `verify_prepare_ms`, `target_forward_ms`, `accept_sample_ms`, `accepted_kv_rematerialize_ms`, `commit_metadata_ms`, `finish_check_ms`, and `verify_commit_total_ms`.

- [ ] **Step 7: Extend chunked-prefill/block-manager regressions**

In `tools/test_chunked_prefill.py`, add real `BlockManager` tests proving:

- a native-originated accepted prefix crossing one and two block boundaries publishes only fully materialized blocks;
- a just-filled final accepted-token block remains pending and is published by the next normal scheduler append, not early;
- rejected reserved blocks never enter `seq.block_table` or `hash_to_block_id`.

- [ ] **Step 8: Run lifecycle and block-manager tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_ngram_speculative.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_chunked_prefill.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_model_runner_spec_verify.py
```

Expected: all pass, including legacy rematerialization regression tests.

- [ ] **Step 9: Commit native lifecycle**

```bash
git add tools/profile_ngram_commit.py tools/test_ngram_speculative.py tools/test_chunked_prefill.py
git commit -m "Commit native verifier KV without replay"
```

---

### Task 6: Row-Expanded Decode Oracle

**Files:**
- Create: `tools/native_verifier_oracle.py`
- Create: `tools/test_native_verifier_oracle.py`
- Modify: `tinyvllm/engine/model_runner.py`

**Interfaces:**
- Produces: `ModelRunner.snapshot_kv_slots(physical_slots: list[int]) -> dict[str, torch.Tensor]`
- Produces: `DTypeTolerance(logits_rtol, logits_atol, kv_rtol, kv_atol)`
- Produces: `dtype_tolerance(dtype_name: str) -> DTypeTolerance`
- Produces: `compare_native_and_oracle(native: dict, oracle: dict) -> dict`
- Produces CLI: `python tools/native_verifier_oracle.py run-case --policy native|oracle --case-json PATH --out PATH --model PATH --continuation-steps 16`

- [ ] **Step 1: Write failing tolerance and comparison tests**

Create `tools/test_native_verifier_oracle.py`:

```python
from tools.native_verifier_oracle import (
    compare_native_and_oracle,
    dtype_tolerance,
)


def test_dtype_tolerances_are_fixed():
    assert dtype_tolerance("torch.float16").logits_atol == 2e-3
    assert dtype_tolerance("torch.bfloat16").logits_atol == 8e-3


def test_comparison_requires_tokens_acceptance_metadata_and_continuation():
    payload = {
        "dtype": "torch.float16",
        "target_tokens": [4, 5, 6],
        "accepted_tokens": [4, 5],
        "sequence_tokens_after": [1, 2, 3, 4, 5],
        "block_table_after": [0],
        "continuation_tokens": list(range(16)),
        "logits": [[0.0, 1.0]],
        "kv": [[0.0, 1.0]],
        "finite": True,
    }
    comparison = compare_native_and_oracle(payload, dict(payload))
    assert comparison["status"] == "PASS"
    assert comparison["token_match"] is True
    assert comparison["continuation_steps"] == 16


def test_token_mismatch_is_no_go_even_when_numeric_error_is_small():
    native = make_comparison_fixture()
    oracle = make_comparison_fixture()
    oracle["continuation_tokens"][-1] += 1
    comparison = compare_native_and_oracle(native, oracle)
    assert comparison["status"] == "NO_GO"
    assert "continuation token mismatch" in comparison["reasons"]


def test_missing_or_nonfinite_evidence_is_incomplete():
    native = make_comparison_fixture()
    oracle = make_comparison_fixture()
    native["finite"] = False
    assert compare_native_and_oracle(native, oracle)["status"] == "NO_GO"
    del oracle["kv"]
    assert compare_native_and_oracle(native, oracle)["status"] == "INCOMPLETE"
```

- [ ] **Step 2: Run the test and verify the module is missing**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_native_verifier_oracle.py
```

Expected: import failure for `tools.native_verifier_oracle`.

- [ ] **Step 3: Add a safe KV snapshot API**

In `ModelRunner`, add:

```python
    def snapshot_kv_slots(
        self,
        physical_slots: list[int],
    ) -> dict[str, torch.Tensor]:
        if self.config.kv_quant_bits != 0:
            raise RuntimeError("KV snapshot requires FP KV")
        block_ids = torch.tensor(
            [slot // self.block_size for slot in physical_slots],
            device=self.kv_cache.device,
            dtype=torch.long,
        )
        offsets = torch.tensor(
            [slot % self.block_size for slot in physical_slots],
            device=self.kv_cache.device,
            dtype=torch.long,
        )
        keys = self.kv_cache[0, :, block_ids, offsets].detach().cpu().clone()
        values = self.kv_cache[1, :, block_ids, offsets].detach().cpu().clone()
        return {"keys": keys, "values": values}
```

The method is debug/test-only and must not run in normal profiler mode unless oracle evidence is requested.

- [ ] **Step 4: Implement isolated serialized decode oracle**

`tools/native_verifier_oracle.py run-case` must:

1. create a fresh eager `LLM` for one case;
2. prefill the same prompt and reach the same pre-verification history;
3. clone only sequence metadata for the oracle path while using a fresh model/cache process;
4. obtain the same first target through normal decode;
5. for each tail input token, append it as the pending token and call normal `prepare_decode()` / `run_model(..., execution_mode="decode")`;
6. capture every tail logit row and every written KV slot with `snapshot_kv_slots()`;
7. apply the same greedy acceptance, EOS, output-budget, and metadata commit logic;
8. continue normal greedy decode for at least 16 tokens;
9. emit one JSON payload containing logits, KV summaries or tensors, argmax targets, accepted prefix, sequence metadata, block visibility, continuation tokens/logits/KV, finite checks, dtype, and exact tolerances.

The native case must run in a separate fresh process from the oracle case so candidate slots are never shared.

- [ ] **Step 5: Implement strict comparison**

`compare_native_and_oracle()` returns:

```python
{
    "status": "PASS" | "NO_GO" | "INCOMPLETE",
    "reasons": ["exact human-readable reason strings"],
    "target_token_match": bool,
    "accepted_prefix_match": bool,
    "metadata_match": bool,
    "continuation_token_match": bool,
    "continuation_steps": int,
    "finite": bool,
    "max_logit_abs_error": float,
    "max_kv_abs_error": float,
    "logits_within_tolerance": bool,
    "kv_within_tolerance": bool,
}
```

Missing evidence is `INCOMPLETE`. Any token, acceptance, metadata, continuation, NaN, or infinity mismatch is `NO_GO`. Numerical tolerance failure with identical tokens is still `NO_GO` for Gate 2.

- [ ] **Step 6: Run oracle unit tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_native_verifier_oracle.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_model_runner_spec_verify.py
```

Expected: both pass.

- [ ] **Step 7: Commit oracle support**

```bash
git add tinyvllm/engine/model_runner.py tools/native_verifier_oracle.py tools/test_native_verifier_oracle.py
git commit -m "Add row-expanded verifier oracle"
```

---

### Task 7: Reproducible Gate and Artifact Verifier

**Files:**
- Create: `tools/native_verifier_gate.py`
- Create: `tools/test_native_verifier_gate.py`

**Interfaces:**
- Produces: `CASE_MATRIX`
- Produces: `build_manifest(source_commit: str, source_dirty: bool, model_path: str, model_identifier: str, host: str, python_bin: str, torch_version: str, cuda_version: str, flash_attn_version: str, gpu_name: str, bf16_supported: bool, run_tag: str) -> dict`
- Produces: `classify_gate(manifest, capability, case_rows, event_rows) -> dict`
- Produces: `verify_artifacts(out_dir: Path) -> dict`
- Produces CLI: `run`, `verify`, and `render-report`.

- [ ] **Step 1: Define the committed case matrix in failing tests**

Create `tools/test_native_verifier_gate.py` and require these dimensions:

```python
def test_case_matrix_covers_required_dimensions():
    cases = gate.CASE_MATRIX
    assert {case["draft_len"] for case in cases} == {1, 4, 8, 16}
    assert {"zero", "one", "partial", "full"} <= {
        case["acceptance_case"] for case in cases
    }
    assert any(case["eos_case"] for case in cases)
    assert any(case["output_budget_case"] for case in cases)
    assert {"current_block", "one_new_block", "multi_block_context"} <= {
        case["block_case"] for case in cases
    }
    assert all(case["continuation_steps"] >= 16 for case in cases)
```

Use deterministic draft construction:

- `full`: baseline target stream unchanged;
- `partial`: mutate the token after a fixed accepted prefix;
- `one`: retain only the first target match;
- `zero`: mutate the first draft token;
- EOS: place the real EOS token at a preregistered accepted position;
- budget: set remaining budget below the accepted prefix;
- block cases: choose history offsets relative to block size `256`.
  With `K <= 16`, a real verifier event can enter at most one new block;
  `multi_block_context` therefore means multiple visible history blocks plus
  a tail that crosses one block boundary. Multiple newly reserved blocks stay
  covered by dependency-light lifecycle tests using a smaller block size.

- [ ] **Step 2: Add failing classification tests**

Synthetic complete evidence must classify:

- `READY_FOR_PERFORMANCE_GATE` only when capability passes, every native/oracle/baseline exactness row passes, every native event has zero rematerialization/copy/replay, and timing direction is positive;
- `NO_GO` for token, acceptance, lifecycle, continuation, rematerialization, copy/replay, or native semantic failures;
- `INCOMPLETE` for missing cases, duplicate rows, process failures, unavailable `seqlen_q>1`, missing tensors, hash mismatch, or absent performance evidence after exactness.

Also assert:

- `K=1` native versus baseline regression must be `<=1%`;
- accepted `K>1` events have lower median verifier-plus-commit time than legacy;
- target forward reduction equals removed legacy decode replay calls;
- zero-accept events remain in end-to-end throughput;
- memory is reported but never used as a success criterion.

- [ ] **Step 3: Add failing six-file artifact integrity tests**

The exact artifact set is:

```python
REQUIRED_ARTIFACTS = (
    "manifest.json",
    "capability.json",
    "case_rows.json",
    "event_rows.json",
    "summary.json",
    "report.md",
)
```

Tests must reject:

- a missing file;
- altered manifest thresholds/case matrix;
- source-dirty canonical evidence;
- duplicate or missing case keys;
- report classification differing from `summary.json`;
- SHA-256 mismatch in `manifest.json["artifact_hashes"]`;
- a native event without explicit `decode_calls`, copy, and replay fields.

- [ ] **Step 4: Implement manifest and process isolation**

`build_manifest()` records:

- source commit and dirty flag;
- exact model path and model identifier;
- host, Python, CUDA device, FlashAttention/PyTorch/CUDA versions;
- fixed dtype tolerances;
- full deterministic `CASE_MATRIX`;
- policy names `baseline`, `legacy_rematerialize`, `native`, `oracle`;
- required artifacts;
- claim boundaries from the design;
- dynamic port pair for each process;
- prompt/token history hashes;
- `created_unix_s` and run tag.

Each policy/case runs in a fresh process with distinct `TINYVLLM_DIST_PORT` and `MASTER_PORT`.

- [ ] **Step 5: Implement gate ordering**

`classify_gate()` applies gates in this order:

1. artifact/process completeness;
2. FlashAttention capability;
3. native-versus-oracle exactness;
4. stable baseline exactness;
5. rematerialization/copy/replay elimination;
6. diagnostic performance qualification.

Do not calculate a favorable performance classification when Gates 1-4 fail.

- [ ] **Step 6: Implement report rendering**

`report.md` must include:

- source/environment table;
- capability matrix for `Q in {1,3,7,15}`;
- exactness matrix by case;
- baseline/native token-stream hash;
- native/oracle max logit and KV errors with tolerances;
- accepted-prefix and 16-token continuation status;
- native versus legacy median verifier-plus-commit by `K`;
- target forward counts;
- end-to-end time/tokens-per-second including zero accepts;
- maximum allocated GPU memory labeled diagnostic;
- final classification and exact reasons;
- explicit non-claims.

- [ ] **Step 7: Run gate unit tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_native_verifier_gate.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_native_verifier_oracle.py
```

Expected: both pass.

- [ ] **Step 8: Commit the gate**

```bash
git add tools/native_verifier_gate.py tools/test_native_verifier_gate.py
git commit -m "Add native verifier evidence gate"
```

---

### Task 8: Isolated Remote Capability and Exactness Runner

**Files:**
- Create: `tools/run_native_verifier_gate_remote.sh`
- Modify: `tools/native_verifier_gate.py`

**Interfaces:**
- Produces commands:
  - `tools/run_native_verifier_gate_remote.sh preflight`
  - `tools/run_native_verifier_gate_remote.sh smoke`
- Produces local evidence under `experiments/native_verifier/${RUN_TAG}/`, where the script itself assigns the unique `RUN_TAG`.

- [ ] **Step 1: Write shell preflight and isolation checks**

Create `tools/run_native_verifier_gate_remote.sh` with:

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REMOTE_HOST="${REMOTE_HOST:-sitian@10.232.195.203}"
REMOTE_PYTHON="${REMOTE_PYTHON:-/data00/home/sitian/sitian-workspace01/tllm/env/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B}"
REMOTE_BASE="${REMOTE_BASE:-/data00/home/sitian/sitian-workspace01/tllm/native-verifier-gates}"
CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-7}"
MODE="${1:-smoke}"
RUN_TAG="${RUN_TAG:-qwen3-06b-$(date +%Y%m%d-%H%M%S)-$$}"
REMOTE_DIR="${REMOTE_BASE}/${RUN_TAG}"
LOCAL_OUT="${LOCAL_OUT:-${REPO_ROOT}/experiments/native_verifier/${RUN_TAG}}"
CONTROL_SOCKET="${CONTROL_SOCKET:-/tmp/ssh-sitian-10.232.195.203}"
```

Use `BatchMode=yes`, add `-S` only when the ControlMaster socket exists, require a clean local source for smoke, upload a tar snapshot, and verify the unique remote directory is empty before extraction.

- [ ] **Step 2: Add exact remote preflight**

Remote preflight must print and save:

```bash
"${REMOTE_PYTHON}" - <<'PY'
import flash_attn
import torch
print({
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "flash_attn": getattr(flash_attn, "__version__", "unknown"),
    "bf16_supported": torch.cuda.is_bf16_supported(),
    "gpu": torch.cuda.get_device_name(0),
})
PY
```

It must also verify:

- model directory and `config.json`;
- Python can import `tinyvllm`;
- `tools/test_native_verifier_attention.py` CPU dispatch test passes;
- all uploaded Python files compile;
- no pre-existing remote checkout is modified.

- [ ] **Step 3: Run the capability matrix before model cases**

Invoke the attention capability mode for:

- `Q in {1,3,7,15}`;
- FP16 and BF16 when supported;
- one-block and cross-block;
- GQA;
- output and written-KV comparison;
- future-row masking.

If `seqlen_q>1` is unavailable or divergent, write `capability.json` with `status="INCOMPLETE"` and stop before performance cases.

- [ ] **Step 4: Run isolated baseline, legacy, native, and oracle cases**

For every committed case:

1. allocate two distinct free ports with Python socket binds;
2. export both `TINYVLLM_DIST_PORT` and `MASTER_PORT`;
3. run one fresh process for exactly one policy/case;
4. retry only `EADDRINUSE`/address-in-use errors, at most three times, with new ports;
5. never reuse a model process across policies;
6. record return code, stdout/stderr paths, ports, elapsed time, max GPU memory, and source SHA.

- [ ] **Step 5: Download and verify evidence**

Download exactly:

```bash
manifest.json
capability.json
case_rows.json
event_rows.json
summary.json
report.md
```

Compare remote and local SHA-256 values, then run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 \
  "${REPO_ROOT}/tools/native_verifier_gate.py" verify \
  --out-dir "${LOCAL_OUT}"
```

- [ ] **Step 6: Validate the runner locally without starting a GPU job**

Run:

```bash
bash -n tools/run_native_verifier_gate_remote.sh
PYTHONDONTWRITEBYTECODE=1 python3 tools/native_verifier_gate.py --help
PYTHONDONTWRITEBYTECODE=1 python3 tools/native_verifier_gate.py verify --help
```

Expected: syntax and help checks pass.

- [ ] **Step 7: Commit the remote runner**

```bash
git add tools/run_native_verifier_gate_remote.sh tools/native_verifier_gate.py
git commit -m "Add isolated native verifier remote runner"
```

---

### Task 9: Full Local Regression Gate

**Files:**
- Potentially modify: `tinyvllm/speculative/verifier.py`
- Potentially modify: `tinyvllm/utils/context.py`
- Potentially modify: `tinyvllm/engine/model_runner.py`
- Potentially modify: `tinyvllm/layers/attention.py`
- Potentially modify: `tools/profile_ngram_commit.py`
- Potentially modify: `tools/native_verifier_oracle.py`
- Potentially modify: `tools/native_verifier_gate.py`
- Potentially modify: `tools/test_native_verifier_contract.py`
- Potentially modify: `tools/test_context_modes.py`
- Potentially modify: `tools/test_model_runner_spec_verify.py`
- Potentially modify: `tools/test_native_verifier_attention.py`
- Potentially modify: `tools/test_native_verifier_oracle.py`
- Potentially modify: `tools/test_native_verifier_gate.py`
- Potentially modify: `tools/test_ngram_speculative.py`
- Potentially modify: `tools/test_chunked_prefill.py`

**Interfaces:**
- Verifies all focused native tests plus existing speculative, SAM, and chunked-prefill suites.

- [ ] **Step 1: Run static checks**

Run:

```bash
python3 -m py_compile \
  tinyvllm/speculative/verifier.py \
  tinyvllm/utils/context.py \
  tinyvllm/engine/model_runner.py \
  tinyvllm/layers/attention.py \
  tools/profile_ngram_commit.py \
  tools/native_verifier_oracle.py \
  tools/native_verifier_gate.py
git diff --check
```

Expected: no output from `git diff --check`; all files compile.

- [ ] **Step 2: Run all new focused tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_native_verifier_contract.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_context_modes.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_model_runner_spec_verify.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_native_verifier_attention.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_native_verifier_oracle.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_native_verifier_gate.py
```

Expected: all dependency-light tests pass; unavailable local CUDA cases explicitly skip.

- [ ] **Step 3: Run existing regression suites**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_ngram_speculative.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_sam_speculative.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_sam_drafter_gate.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_adaptive_ngram_gate.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_chunked_prefill.py
```

Expected: all pass without changing old gate decisions or artifact schemas.

- [ ] **Step 4: Audit fail-closed mutation ordering**

Run:

```bash
rg -n "prepare_spec_verify|_validate_spec_verify_compatibility|store_kvcache|reserve_append_blocks" \
  tinyvllm/engine/model_runner.py tinyvllm/layers/attention.py tools/profile_ngram_commit.py
```

Manually verify from the displayed order that:

- compatibility checks precede tensor upload and KV write;
- native `K=1` never calls the tail verifier;
- optional feature branches cannot run under `spec_verify`;
- exception cleanup owns and releases every uncommitted reservation.

- [ ] **Step 5: Commit any focused regression corrections**

If no corrections are needed, do not create an empty commit. If corrections are needed, stage only the changed paths from the explicit Task 9 file list:

```bash
git add tinyvllm tools
git commit -m "Harden native verifier regression gate"
```

---

### Task 10: Remote Smoke, Completion Audit, and Documentation

**Files:**
- Create: `experiments/native_verifier/${RUN_TAG}/manifest.json`
- Create: `experiments/native_verifier/${RUN_TAG}/capability.json`
- Create: `experiments/native_verifier/${RUN_TAG}/case_rows.json`
- Create: `experiments/native_verifier/${RUN_TAG}/event_rows.json`
- Create: `experiments/native_verifier/${RUN_TAG}/summary.json`
- Create: `experiments/native_verifier/${RUN_TAG}/report.md`
- Modify: `README.md`
- Modify: `AGENT_HANDOFF_STATE.md`

**Interfaces:**
- Produces final first-phase classification: `READY_FOR_PERFORMANCE_GATE`, `NO_GO`, or `INCOMPLETE`.
- Produces a prompt-to-artifact completion checklist mapping every design requirement to evidence.

- [ ] **Step 1: Confirm remote reachability and GPU selection**

Run:

```bash
CONTROL_SOCKET=/tmp/ssh-sitian-10.232.195.203
SSH_SOCKET_ARGS=()
if [[ -S "${CONTROL_SOCKET}" ]]; then
  SSH_SOCKET_ARGS=(-S "${CONTROL_SOCKET}")
fi
ssh -o BatchMode=yes "${SSH_SOCKET_ARGS[@]}" sitian@10.232.195.203 \
  'hostname; nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader'
```

If the current shell cannot authenticate, report the exact `klist`/SSH error and use the existing Terminal-created ControlMaster socket `/tmp/ssh-sitian-10.232.195.203`; do not switch to user `bytedance`.

- [ ] **Step 2: Run remote preflight**

Run:

```bash
GPU_INDEX="$(
  ssh -o BatchMode=yes "${SSH_SOCKET_ARGS[@]}" sitian@10.232.195.203 \
    "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits" |
  awk -F',' '{gsub(/ /,\"\",$1); gsub(/ /,\"\",$2); print $2, $1}' |
  sort -n |
  head -1 |
  awk '{print $2}'
)"
CUDA_VISIBLE_DEVICES="${GPU_INDEX}" \
tools/run_native_verifier_gate_remote.sh preflight
```

Expected: isolated upload and capability environment preflight pass.

- [ ] **Step 3: Run the first-phase smoke**

Run:

```bash
CUDA_VISIBLE_DEVICES="${GPU_INDEX}" \
tools/run_native_verifier_gate_remote.sh smoke
```

Keep polling until the process completes or a real blocker occurs. Do not stop after upload or the first case.

- [ ] **Step 4: Independently verify downloaded artifacts**

Run:

```bash
RUN_TAG="$(find experiments/native_verifier -mindepth 1 -maxdepth 1 -type d -name 'qwen3-06b-*' -print | sort | tail -1 | xargs basename)"
PYTHONDONTWRITEBYTECODE=1 python3 tools/native_verifier_gate.py verify \
  --out-dir "experiments/native_verifier/${RUN_TAG}"
```

Expected: verifier prints the same classification and artifact hashes recorded in `summary.json`.

- [ ] **Step 5: Build the prompt-to-artifact completion checklist**

Add a `completion_audit` object to `summary.json` and a matching section in `report.md` with one row per requirement:

```text
explicit context mode
single-sequence linear greedy eager FP-KV scope
H/K/Q tensor contract
K=1 zero-tail behavior
multi-query causal attention
direct final-slot KV writes
metadata-only commit and pending final token
zero/one/partial/full acceptance
EOS and output-budget truncation
current/one-new/multiple-new block boundaries
rollback before/after KV write and during commit
row-expanded isolated oracle
argmax and accepted-prefix equality
metadata/block visibility equality
16-token continuation equality
stable normal greedy equality
finite logits/KV and numeric tolerances
decode_calls==0
accepted_kv_rematerialize_ms==0
no accepted-token copy/replay
K=1 <=1% control regression
K>1 native versus legacy timing direction
target forward-count reduction
zero-accept inclusion in throughput
max allocated memory diagnostic
remote source/port/isolation evidence
claim boundaries
README and handoff update
```

Each row contains `status`, exact artifact path, JSON key or command output, and a short evidence note. Any missing or uncertain row prevents `READY_FOR_PERFORMANCE_GATE`.

- [ ] **Step 6: Update README with measured evidence**

Add:

- local focused test commands and status;
- exact remote smoke command;
- artifact directory;
- capability result;
- exactness matrix summary;
- native-versus-legacy timing and forward-count summary;
- final classification;
- what the result proves;
- all non-claims;
- requirement for a separate preregistered performance-gate spec before production recommendation.

- [ ] **Step 7: Update handoff state**

Append to `AGENT_HANDOFF_STATE.md`:

- branch and final source commit;
- implementation commits;
- remote host/user/Python/model;
- remote isolated directory and local artifact directory;
- selected GPU and process port evidence;
- all validation commands and outcomes;
- classification and exact reasons;
- failed or incomplete cases;
- whether rematerialization was fully eliminated;
- next action.

- [ ] **Step 8: Run final completion audit**

Run:

```bash
git status --short
git diff --check
PYTHONDONTWRITEBYTECODE=1 python3 tools/native_verifier_gate.py verify \
  --out-dir "experiments/native_verifier/${RUN_TAG}"
rg -n "READY_FOR_PERFORMANCE_GATE|NO_GO|INCOMPLETE|native verifier|rematerial" \
  README.md AGENT_HANDOFF_STATE.md \
  "experiments/native_verifier/${RUN_TAG}/summary.json" \
  "experiments/native_verifier/${RUN_TAG}/report.md"
```

Inspect every completion-audit row against the actual artifacts. Do not rely on a green summary alone.

- [ ] **Step 9: Commit evidence and documentation**

```bash
git add \
  "experiments/native_verifier/${RUN_TAG}" \
  README.md \
  AGENT_HANDOFF_STATE.md
git commit -m "Record native verifier first-phase gate"
```

- [ ] **Step 10: Report the bounded conclusion**

If all exactness/elimination rows pass and timing moves in the expected direction, report only `READY_FOR_PERFORMANCE_GATE`. If a semantic/lifecycle/replay mismatch exists, report `NO_GO`. If capability, environment, artifacts, or coverage are insufficient, report `INCOMPLETE`.

Do not call the first-phase result a production `GO`.
