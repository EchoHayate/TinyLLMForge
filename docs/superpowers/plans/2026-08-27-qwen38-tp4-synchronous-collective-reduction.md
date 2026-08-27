# Qwen3.8 TP4 Synchronous Collective Reduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a source-bound, low-disturbance TP4 qualification gate
that determines whether one exact synchronous collective reduction candidate
has at least 5% attributable decode-TPOT opportunity.

**Architecture:** Add a model-agnostic synchronous-collective observer beside
the existing decode profiler, route every steady-decode collective through
the common wrapper, and keep Qwen3.8 topology policy in a pure `tools/`
catalog/classifier. Reuse the existing Qwen3.8 controller's remote identity,
Kerberos, strict-clean GPU, process ownership, storage, and cleanup contracts;
do not reuse its Nsight path. Assemble a small immutable bundle and require
producer, remote verifier, and local verifier agreement before authorizing a
candidate-specific design.

**Tech Stack:** Python 3.12, PyTorch distributed/CUDA events, pytest, JSON and
JSONL evidence, SSH, NVIDIA NVML/nvidia-smi, Git.

## Global Constraints

- The only authoritative checkout is `/Users/bytedance/Desktop/TinyLLMForge`.
- Do not create a worktree or use subagents.
- Keep and push only `origin/feat/kv-sparse-attention`.
- Stage exact task paths only; do not use broad `git add`, `git reset`,
  `git clean`, or mass formatting.
- Use `git -c core.hooksPath=/dev/null commit`.
- Every commit has exactly one
  `Co-authored-by: TRAE CLI <noreply@bytedance.com>` trailer.
- Remote task data must stay below
  `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818`.
- Do not write task data to remote `/`, remote `/tmp`, an old checkout, or
  the retired adaptive-ngram checkout.
- Do not run `kinit` or `krenew`.
- Do not signal, terminate, reserve, or take over unrelated processes or
  GPUs.
- Require four strict-clean GPUs at every worker entry: memory used at most
  1,024 MiB, utilization at most 5%, and no compute process.
- Preserve `OVERLAP_DESIGN_AUTHORIZED=false`.
- Preserve `ASYNC_COLLECTIVES_AUTHORIZED=false`.
- Do not introduce `async_op=True`, pending work handles, a communication
  stream, dual-stream execution, event-based overlap dependencies, chunked
  ReduceScatter/AllGather, or a new Nsight campaign.
- The qualification thresholds are frozen:
  median matched profiler overhead at most 3.0%, maximum at most 5.0%, and
  candidate lower-bound attributable TPOT opportunity at least 5.0%.
- Qualification changes observation only. Production collective execution
  may not change in this plan.
- Report benefit and cost together. A qualification `GO` authorizes a second
  candidate-specific design; it is not a speedup claim.

---

## File Map

### Runtime mechanism

- Create `tinyvllm/engine/synchronous_collective_census.py`: model-agnostic
  count/byte observer, deterministic sampled-event policy, step/layer
  contexts, finalization, and fail-closed validation.
- Modify `tinyvllm/engine/decode_internal_profiler.py`: fan existing
  `profile_layer`, `run_profiled_step`, and `profile_collective` calls into
  the census without changing the existing profiler schema.
- Modify `tinyvllm/engine/model_runner.py`: configure, reset, finalize, and
  activate the census on every rank.
- Modify `tinyvllm/engine/llm_engine.py`: expose acknowledged all-rank census
  configuration/finalization methods.
- Modify `tinyvllm/layers/linear.py`: attach stable site roles to row-parallel
  and prefill collectives.
- Modify `tinyvllm/layers/embed_head.py`: attach stable embedding site
  metadata and classify initialization gathers as `startup`.
- Modify `tinyvllm/engine/tensor_parallel_greedy.py`: route the blocking
  token-ID broadcast through the common synchronous wrapper.

### Qualification policy and execution

- Create `tools/qwen38_collective_reduction.py`: static 130-site catalog,
  consumer proofs, census reconciliation, overhead selection, reduction
  ceiling, and terminal classification.
- Create `tools/qwen38_tp4_collective_reduction_worker.py`: source-bound TP4
  workload runner that emits bounded census/timing/correctness/memory
  receipts.
- Create `tools/run_qwen38_tp4_collective_reduction.py`: local controller,
  immutable plan, strict-clean monitor, remote worker launch, exact attempt
  ownership, cleanup, and artifact retrieval.
- Create `tools/assemble_qwen38_tp4_collective_reduction.py`: producer bundle
  assembler and manifest writer.
- Create `tools/verify_qwen38_tp4_collective_reduction.py`: independent
  semantic verifier and classification recomputation.

### Tests

- Create `tools/test_synchronous_collective_census.py`.
- Modify `tools/test_decode_internal_profiler.py`.
- Modify `tools/test_decode_internal_profile_wiring.py`.
- Modify `tools/test_tensor_parallel_greedy.py`.
- Create `tools/test_qwen38_collective_reduction.py`.
- Create `tools/test_qwen38_tp4_collective_reduction_worker.py`.
- Create `tools/test_run_qwen38_tp4_collective_reduction.py`.
- Create `tools/test_assemble_qwen38_tp4_collective_reduction.py`.
- Create `tools/test_verify_qwen38_tp4_collective_reduction.py`.

### Terminal evidence

- Create
  `artifacts/qwen38_tp4_collective_reduction/20260827-qwen38-tp4-collective-reduction-r1/controller/`
  only through controller execution.
- Create
  `docs/superpowers/audits/2026-08-27-qwen38-tp4-synchronous-collective-reduction-audit.md`
  only after terminal producer and both verifiers finish.
- Append the terminal reconciliation to `AGENT_HANDOFF_STATE.md` only after
  the audit is complete.

---

### Task 1: Pure census policy and record contract

**Files:**

- Create: `tinyvllm/engine/synchronous_collective_census.py`
- Create: `tools/test_synchronous_collective_census.py`

**Interfaces:**

- Produces:
  `CollectiveCensusPolicy(sample_budget: int, cohort_count: int,
  expected_collective_count: int, source_revision: str, attempt: str,
  workload: str, repetition: int)`.
- Produces:
  `SynchronousCollectiveCensus(rank, policy, event_factory, synchronize,
  stream_resolver)`.
- Produces context helpers `active_synchronous_collective_census`,
  `run_census_step`, `census_layer`, and
  `observe_synchronous_collective`.
- Produces `finalize(already_synchronized=False) -> dict`.

- [ ] **Step 1: Write failing validation and deterministic-cohort tests**

```python
def test_policy_selects_a_stable_bounded_cohort():
    policy = CollectiveCensusPolicy(
        sample_budget=8,
        cohort_count=17,
        expected_collective_count=130,
        source_revision="a" * 40,
        attempt="attempt-r1",
        workload="P0",
        repetition=2,
    )
    first = policy.sampled_ordinals(
        decode_ordinal=5,
        collective_count=130,
    )
    second = policy.sampled_ordinals(
        decode_ordinal=5,
        collective_count=130,
    )
    assert first == second
    assert len(first) == 8
    assert len(set(first)) == 8
    assert min(first) >= 0
    assert max(first) < 130


@pytest.mark.parametrize("budget", [-1, 33])
def test_policy_rejects_unsupported_event_budget(budget):
    with pytest.raises(ValueError, match="sample_budget"):
        CollectiveCensusPolicy(
            sample_budget=budget,
            cohort_count=17,
            expected_collective_count=130,
            source_revision="a" * 40,
            attempt="attempt-r1",
            workload="P0",
            repetition=0,
        )
```

- [ ] **Step 2: Run the focused tests and confirm RED**

Run:

```bash
/opt/homebrew/bin/python3.12 -m pytest \
  tools/test_synchronous_collective_census.py -q
```

Expected: collection fails because
`tinyvllm.engine.synchronous_collective_census` does not exist.

- [ ] **Step 3: Implement the immutable policy and schema validators**

Implement these exact public types:

```python
@dataclass(frozen=True)
class CollectiveCensusPolicy:
    sample_budget: int
    cohort_count: int
    expected_collective_count: int
    source_revision: str
    attempt: str
    workload: str
    repetition: int

    def sampled_ordinals(
        self,
        *,
        decode_ordinal: int,
        collective_count: int,
    ) -> Sequence[int]:
        seed = (
            f"{self.source_revision}\0{self.attempt}\0"
            f"{self.workload}\0{self.repetition}\0{decode_ordinal}"
        ).encode("utf-8")
        cohort = int.from_bytes(
            hashlib.sha256(seed).digest()[:8],
            "big",
        ) % self.cohort_count
        cohort_width = math.ceil(
            self.expected_collective_count / self.cohort_count
        )
        start = (
            cohort * cohort_width
        ) % self.expected_collective_count
        return tuple(
            sorted(
                (start + offset) % self.expected_collective_count
                for offset in range(
                    min(
                        self.sample_budget,
                        self.expected_collective_count,
                    )
                )
            )
        )


class SynchronousCollectiveCensus:
    def begin_step(
        self,
        *,
        batch_kind: str,
        is_decode: bool,
        active_sequence_count: int,
        request_set_sha256: str,
        dispatch: str,
    ) -> None:
        self._require_mutable()
        if self._active_step is not None:
            raise RuntimeError("collective census step is active")
        self._active_step = self._new_step(
            batch_kind=batch_kind,
            is_decode=is_decode,
            active_sequence_count=active_sequence_count,
            request_set_sha256=request_set_sha256,
            dispatch=dispatch,
        )

    def end_step(self) -> None:
        if self._active_step is None:
            raise RuntimeError("collective census step is not active")
        self._steps.append(self._active_step)
        self._active_step = None

    @contextmanager
    def layer(self, layer_index: int, layer_role: str):
        previous = self._active_layer
        self._active_layer = {
            "layer_index": int(layer_index),
            "layer_role": str(layer_role),
        }
        try:
            yield
        finally:
            self._active_layer = previous

    def observe(
        self,
        *,
        site_role: str,
        operation: str,
        tensor,
        call,
        collective_kind: str,
        process_group: str,
        execution_phase: str,
        async_mode: bool,
        source_rank: int | None,
        destination_rank: int | None,
    ):
        row = self._prepare_record(
            site_role=site_role,
            operation=operation,
            tensor=tensor,
            collective_kind=collective_kind,
            process_group=process_group,
            execution_phase=execution_phase,
            async_mode=async_mode,
            source_rank=source_rank,
            destination_rank=destination_rank,
        )
        start_event = self._event_factory() if row["sampled"] else None
        end_event = self._event_factory() if row["sampled"] else None
        if start_event is not None:
            start_event.record()
        try:
            return call(tensor)
        finally:
            if end_event is not None:
                end_event.record()
            self._finish_record(row, start_event, end_event)

    def finalize(self, *, already_synchronized: bool = False) -> dict:
        if self._active_step is not None or self._active_layer is not None:
            raise RuntimeError("cannot finalize an open census scope")
        if self._finalized is None:
            if self._timed_records and not already_synchronized:
                self._synchronize()
            self._finalized = self._build_snapshot()
        return copy.deepcopy(self._finalized)
```

Use SHA-256 over canonical JSON for cohort selection. Permit only budgets
`0`, `8`, `16`, and `32`. Record tensor shape/dtype/bytes as plain values and
never retain the tensor. Create events only for sampled ordinals. Resolve
events after one final synchronization. Return schema
`tinyllmforge.synchronous-collective-census.v1`.

- [ ] **Step 4: Add behavior, failure, and no-double-call tests**

Add concrete assertions following these forms:

```python
def test_async_mode_is_rejected_before_collective_call():
    census, calls = _active_census(sample_budget=0)
    with pytest.raises(ValueError, match="async_mode"):
        census.observe(
            site_role="row_parallel_output",
            operation="row_parallel_all_reduce",
            tensor=FakeTensor(),
            call=lambda tensor: calls.append(tensor),
            collective_kind="all_reduce",
            process_group="tensor_parallel",
            execution_phase="decode",
            async_mode=True,
            source_rank=None,
            destination_rank=None,
        )
    assert calls == []


def test_finalize_synchronizes_once_and_is_idempotent():
    census, synchronizations = _completed_sampled_census()
    first = census.finalize()
    second = census.finalize()
    assert first == second
    assert synchronizations == [True]
    assert all("tensor" not in row for row in first["collectives"])
```

Use the same fixture style to assert count-only creates zero events,
sampled observation creates exactly two events, non-decode creates no row,
missing site/layer fails before the call, an operation exception is called
once, and open scopes cannot finalize.

- [ ] **Step 5: Run Task 1 GREEN verification**

Run:

```bash
/opt/homebrew/bin/python3.12 -m pytest \
  tools/test_synchronous_collective_census.py -q
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-collective-census-pycache \
  /opt/homebrew/bin/python3.12 -m py_compile \
  tinyvllm/engine/synchronous_collective_census.py \
  tools/test_synchronous_collective_census.py
git diff --check -- \
  tinyvllm/engine/synchronous_collective_census.py \
  tools/test_synchronous_collective_census.py
```

Expected: all focused tests pass; compile and whitespace checks exit zero.

- [ ] **Step 6: Commit and push Task 1**

```bash
git add -- \
  tinyvllm/engine/synchronous_collective_census.py \
  tools/test_synchronous_collective_census.py
git -c core.hooksPath=/dev/null commit \
  -m "feat(profiling): add synchronous collective census" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

---

### Task 2: Fan the census through existing profiler contexts

**Files:**

- Modify: `tinyvllm/engine/decode_internal_profiler.py`
- Modify: `tools/test_decode_internal_profiler.py`

**Interfaces:**

- Consumes Task 1 context helpers.
- Extends `profile_collective()` with keyword-only fields:
  `site_role`, `execution_phase="decode"`, `source_rank=None`, and
  `destination_rank=None`.
- Preserves all existing callers when no census is active.

- [ ] **Step 1: Write failing composition tests**

```python
def test_profile_collective_composes_profiler_and_census_once(monkeypatch):
    calls = []
    result = profile_collective(
        "row_parallel_all_reduce",
        FakeTensor(),
        lambda tensor: calls.append(tensor) or "done",
        site_role="row_parallel_output",
        collective_kind="all_reduce",
        process_group="tensor_parallel",
    )
    assert result == "done"
    assert len(calls) == 1


def test_existing_profiler_snapshot_schema_is_unchanged():
    profiler, _ = _profiler()
    snapshot = _run_one_collective_and_finalize(profiler)
    assert set(snapshot) == {
        "rank",
        "enabled",
        "finalization_status",
        "steps",
        "layers",
        "operations",
        "collectives",
        "dropped_steps",
        "dropped_layers",
        "dropped_operations",
        "dropped_collectives",
    }
```

Use a fake active census and the existing fake decode profiler. Assert that
the same exception identity propagates and neither wrapper retries the call.

- [ ] **Step 2: Run the focused tests and confirm RED**

Run:

```bash
/opt/homebrew/bin/python3.12 -m pytest \
  tools/test_decode_internal_profiler.py -q
```

Expected: new tests fail because `profile_collective` does not accept
`site_role` and the census is not entered.

- [ ] **Step 3: Implement `ExitStack`-based context composition**

Use this structure:

```python
def profile_layer(layer_index, layer_role):
    stack = ExitStack()
    profiler = active_decode_internal_profiler()
    if profiler is not None:
        stack.enter_context(profiler.layer(layer_index, layer_role))
    stack.enter_context(census_layer(layer_index, layer_role))
    return stack


def run_profiled_step(
    profiler,
    *,
    batch_kind,
    is_decode,
    active_sequence_count,
    request_set_sha256,
    dispatch,
    call,
):
    profiler.begin_step(
        batch_kind=batch_kind,
        is_decode=is_decode,
        active_sequence_count=active_sequence_count,
        request_set_sha256=request_set_sha256,
        dispatch=dispatch,
    )
    census = active_synchronous_collective_census()
    if census is not None:
        census.begin_step(
            batch_kind=batch_kind,
            is_decode=is_decode,
            active_sequence_count=active_sequence_count,
            request_set_sha256=request_set_sha256,
            dispatch=dispatch,
        )
    try:
        return call()
    finally:
        try:
            if census is not None:
                census.end_step()
        finally:
            profiler.end_step()
```

`profile_collective` must create one inner callable for the existing profiler
and pass that callable once to `observe_synchronous_collective`.

- [ ] **Step 4: Run Task 2 GREEN and adjacent regression tests**

Run:

```bash
/opt/homebrew/bin/python3.12 -m pytest \
  tools/test_synchronous_collective_census.py \
  tools/test_decode_internal_profiler.py \
  tools/test_communication_exposure_event_schema.py -q
git diff --check -- \
  tinyvllm/engine/decode_internal_profiler.py \
  tools/test_decode_internal_profiler.py
```

Expected: all tests pass and the old profiler schema remains byte-compatible
for existing fixtures.

- [ ] **Step 5: Commit and push Task 2**

```bash
git add -- \
  tinyvllm/engine/decode_internal_profiler.py \
  tools/test_decode_internal_profiler.py
git -c core.hooksPath=/dev/null commit \
  -m "feat(profiling): compose collective census contexts" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

---

### Task 3: Wire all production decode collective sites

**Files:**

- Modify: `tinyvllm/layers/linear.py`
- Modify: `tinyvllm/layers/embed_head.py`
- Modify: `tinyvllm/engine/tensor_parallel_greedy.py`
- Modify: `tools/test_decode_internal_profile_wiring.py`
- Modify: `tools/test_tensor_parallel_greedy.py`

**Interfaces:**

- Consumes the extended `profile_collective`.
- Emits stable `site_role` values:
  `row_parallel_output`, `row_parallel_prefill_materialization`,
  `replicated_weight_input_materialization`,
  `vocab_parallel_embedding`, `lm_head_parameter_materialization`, and
  `greedy_token_broadcast`.

- [ ] **Step 1: Write failing AST and behavior tests**

Add assertions that every `dist.all_reduce`, `dist.all_gather`,
`dist.gather`, and `dist.broadcast` in the three production files is either:

1. inside `profile_collective`, or
2. explicitly classified as startup-only and routed through a
   `profile_collective` call whose `execution_phase` is `"startup"`.

Add:

```python
def test_tp_greedy_broadcast_uses_profile_collective_once():
    calls = []

    def broadcast(tensor, src):
        calls.append((tensor.clone(), src))

    tokens = select_tensor_parallel_greedy_tokens(
        logits=None,
        rank=1,
        world_size=4,
        batch_size=2,
        device=torch.device("cpu"),
        broadcast=broadcast,
    )
    assert tokens.shape == (2,)
    assert len(calls) == 1
    assert calls[0][1] == 0
```

- [ ] **Step 2: Run the focused tests and confirm RED**

Run:

```bash
/opt/homebrew/bin/python3.12 -m pytest \
  tools/test_decode_internal_profile_wiring.py \
  tools/test_tensor_parallel_greedy.py -q
```

Expected: AST coverage fails for raw gather/broadcast callsites.

- [ ] **Step 3: Attach stable metadata without changing execution**

Use wrappers of this form:

```python
profile_collective(
    "greedy_token_broadcast",
    token_ids,
    lambda tensor: operation(tensor, src=0),
    site_role="greedy_token_broadcast",
    collective_kind="broadcast",
    process_group="tensor_parallel",
    source_rank=0,
    async_mode=False,
)
```

Initialization-only LM-head parameter gathers use
`execution_phase="startup"` and `destination_rank=0`. Row-parallel and
embedding callsites use `execution_phase="decode_or_prefill"`; the active
step context resolves the actual phase. Do not alter tensor shapes, dtypes,
return values, source/destination ranks, or number of distributed calls.

- [ ] **Step 4: Run Task 3 GREEN and model-runner regressions**

Run isolated pytest processes:

```bash
/opt/homebrew/bin/python3.12 -m pytest \
  tools/test_decode_internal_profile_wiring.py -q
/opt/homebrew/bin/python3.12 -m pytest \
  tools/test_tensor_parallel_greedy.py -q
/opt/homebrew/bin/python3.12 -m pytest \
  tools/test_model_runner_spec_verify.py -q
git diff --check -- \
  tinyvllm/layers/linear.py \
  tinyvllm/layers/embed_head.py \
  tinyvllm/engine/tensor_parallel_greedy.py \
  tools/test_decode_internal_profile_wiring.py \
  tools/test_tensor_parallel_greedy.py
```

Expected: all tests pass in separate processes.

- [ ] **Step 5: Commit and push Task 3**

```bash
git add -- \
  tinyvllm/layers/linear.py \
  tinyvllm/layers/embed_head.py \
  tinyvllm/engine/tensor_parallel_greedy.py \
  tools/test_decode_internal_profile_wiring.py \
  tools/test_tensor_parallel_greedy.py
git -c core.hooksPath=/dev/null commit \
  -m "feat(profiling): identify synchronous collective sites" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

---

### Task 4: Add all-rank census lifecycle to ModelRunner and LLMEngine

**Files:**

- Modify: `tinyvllm/engine/model_runner.py`
- Modify: `tinyvllm/engine/llm_engine.py`
- Modify: `tools/test_model_runner_spec_verify.py`
- Modify: `tools/test_decode_internal_profile_wiring.py`

**Interfaces:**

- Adds `ModelRunner.configure_synchronous_collective_census(policy: dict)`.
- Adds `ModelRunner.reset_synchronous_collective_census()`.
- Adds
  `ModelRunner.finalize_synchronous_collective_census(
  already_synchronized=False, already_synchronized_rank=None)`.
- Adds matching acknowledged all-rank methods on `LLMEngine`.

- [ ] **Step 1: Write failing lifecycle tests**

Add concrete tests following these forms:

```python
def test_model_runner_configures_rank_local_census_from_exact_policy():
    runner = _runner(rank=2, world_size=4)
    receipt = _configure_method(runner, _policy_payload())
    assert receipt == {
        "rank": 2,
        "enabled": True,
        "sample_budget": 8,
        "cohort_count": 17,
    }
    assert runner.synchronous_collective_census.rank == 2


def test_llm_engine_returns_ordered_rank_inventory_0_to_3():
    engine = _engine_with_ack_rows(
        [
            {"rank": 0, "enabled": True},
            {"rank": 1, "enabled": True},
            {"rank": 2, "enabled": True},
            {"rank": 3, "enabled": True},
        ]
    )
    receipt = _configure_engine_method(engine, _policy_payload())
    assert receipt["rank_inventory"] == [0, 1, 2, 3]
```

Add exact assertions that disabled finalization returns a complete empty
snapshot, reset creates a distinct observer, only one rank may claim prior
synchronization, and rank policy disagreement raises before execution.

The policy payload schema is:

```python
{
    "enabled": True,
    "sample_budget": 8,
    "cohort_count": 17,
    "expected_collective_count": 130,
    "source_revision": "a" * 40,
    "attempt": "attempt-r1",
    "workload": "P0",
    "repetition": 0,
}
```

- [ ] **Step 2: Run tests and confirm RED**

Run:

```bash
/opt/homebrew/bin/python3.12 -m pytest \
  tools/test_model_runner_spec_verify.py \
  tools/test_decode_internal_profile_wiring.py -q
```

Expected: failures name the missing census lifecycle methods.

- [ ] **Step 3: Implement lifecycle and acknowledged fan-out**

Initialize a disabled census beside `decode_internal_profiler`. Keep the
configured policy as immutable plain data. `LLMEngine` must use the existing
`call_model_runner_acknowledged` path and reject rank-order, policy, or
enabled-state disagreement.

- [ ] **Step 4: Run Task 4 GREEN**

Run:

```bash
/opt/homebrew/bin/python3.12 -m pytest \
  tools/test_synchronous_collective_census.py \
  tools/test_decode_internal_profiler.py \
  tools/test_decode_internal_profile_wiring.py \
  tools/test_model_runner_spec_verify.py -q
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-collective-lifecycle-pycache \
  /opt/homebrew/bin/python3.12 -m py_compile \
  tinyvllm/engine/model_runner.py \
  tinyvllm/engine/llm_engine.py
```

Expected: all tests pass and compilation exits zero.

- [ ] **Step 5: Commit and push Task 4**

```bash
git add -- \
  tinyvllm/engine/model_runner.py \
  tinyvllm/engine/llm_engine.py \
  tools/test_model_runner_spec_verify.py \
  tools/test_decode_internal_profile_wiring.py
git -c core.hooksPath=/dev/null commit \
  -m "feat(profiling): expose TP collective census lifecycle" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

---

### Task 5: Build the Qwen3.8 catalog, verifier math, and classifier

**Files:**

- Create: `tools/qwen38_collective_reduction.py`
- Create: `tools/test_qwen38_collective_reduction.py`

**Interfaces:**

- Produces
  `build_qwen38_static_collective_catalog(
  text_profile, *, tensor_parallel_size: int) -> Sequence[dict]`.
- Produces `validate_collective_census(rows, catalog) -> dict`.
- Produces `select_event_budget(calibration_rows) -> int | None`.
- Produces
  `build_consumer_dependency_proofs(catalog) -> Sequence[dict]`.
- Produces
  `estimate_reduction_ceiling(census, timing, proofs, online) -> dict`.
- Produces `classify_collective_reduction(summary) -> str`.

- [ ] **Step 1: Write failing catalog tests**

```python
def test_qwen38_catalog_contains_exactly_130_decode_sites():
    catalog = build_qwen38_static_collective_catalog(
        _profile(),
        tensor_parallel_size=4,
    )
    assert len(catalog) == 130
    assert catalog[0]["site_role"] == "vocab_parallel_embedding"
    assert sum(
        row["site_role"] == "row_parallel_output"
        for row in catalog
    ) == 128
    assert catalog[-1]["site_role"] == "greedy_token_broadcast"


def test_catalog_assigns_conservative_consumer_classes():
    catalog = build_qwen38_static_collective_catalog(
        _profile(),
        tensor_parallel_size=4,
    )
    row_parallel = [
        row for row in catalog
        if row["site_role"] == "row_parallel_output"
    ]
    assert {
        row["classification"] for row in row_parallel
    } == {"MANDATORY_IMMEDIATE_CONSUMER"}
    assert catalog[0]["classification"] == "MATERIALIZATION_ALTERNATIVE"
    assert catalog[-1]["classification"] == (
        "MANDATORY_IMMEDIATE_CONSUMER"
    )
```

- [ ] **Step 2: Run catalog tests and confirm RED**

Run:

```bash
/opt/homebrew/bin/python3.12 -m pytest \
  tools/test_qwen38_collective_reduction.py -q
```

Expected: import fails because the policy module does not exist.

- [ ] **Step 3: Implement the pure catalog and proof builder**

Create deterministic site IDs:

```text
embedding.input
layer.000.attention.output
layer.000.mlp.output
the same two ordered sites for every layer index 001 through 062
layer.063.attention.output
layer.063.mlp.output
sampling.greedy_token
```

Each row includes every field frozen in the design. Reject any Qwen3.8
profile that is not 64 layers, hidden size 5,120, vocabulary 248,320, BF16,
or TP4.

- [ ] **Step 4: Write RED tests for rank reconciliation and overhead**

Add concrete reconciliation and budget tests:

```python
def test_select_event_budget_chooses_largest_passing_budget():
    rows = _calibration_rows({
        0: [0.01, 0.02],
        8: [0.02, 0.03],
        16: [0.03, 0.05],
        32: [0.04, 0.06],
    })
    assert select_event_budget(rows) == 16


def test_census_requires_four_identical_rank_sequences():
    rows = _four_rank_rows()
    rows[-1]["collectives"][7]["tensor_bytes"] += 2
    with pytest.raises(ValueError, match="rank collective sequence"):
        validate_collective_census(rows, _catalog())
```

Add mutations for missing, extra, and duplicate sites. Add exact threshold
cases proving that 3% median and 5% maximum pass, values above either limit
fail, zero-budget failure invalidates count-only observation, and absence of
a passing nonzero budget yields no timed budget.

- [ ] **Step 5: Implement reconciliation and frozen event-budget selection**

Use measured ratios:

```python
ratio = instrumented_ns / control_ns - 1.0
```

Return the largest passing value from `(0, 8, 16, 32)`. A non-passing
zero-event arm invalidates the count-only mechanism. A passing zero arm with
no passing nonzero arm returns `None` for timed qualification.

- [ ] **Step 6: Write RED tests for ceilings and all terminal classes**

Cover exact boundary cases at 3%, 5%, and 5% opportunity. Require lower-bound
replacement and uncertainty costs to be subtracted. Classification
precedence is:

```text
INVALID_CORRECTNESS
INVALID_RESOURCE_IDENTITY
INCONCLUSIVE_INCOMPLETE_COVERAGE
INCONCLUSIVE_PROFILER_OVERHEAD
GO_SYNC_COLLECTIVE_REDUCTION
NO_GO_NO_REDUCIBLE_COLLECTIVE
```

- [ ] **Step 7: Implement ceiling arithmetic and classifier**

The `GO` branch requires one named proof with `status == "PASS"` and
`lower_bound_tpot_opportunity_ratio >= 0.05`. Do not use upper-bound
opportunity for promotion.

- [ ] **Step 8: Run Task 5 GREEN**

Run:

```bash
/opt/homebrew/bin/python3.12 -m pytest \
  tools/test_qwen38_collective_reduction.py -q
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-collective-policy-pycache \
  /opt/homebrew/bin/python3.12 -m py_compile \
  tools/qwen38_collective_reduction.py \
  tools/test_qwen38_collective_reduction.py
git diff --check -- \
  tools/qwen38_collective_reduction.py \
  tools/test_qwen38_collective_reduction.py
```

Expected: all tests pass; compile and whitespace checks exit zero.

- [ ] **Step 9: Commit and push Task 5**

```bash
git add -- \
  tools/qwen38_collective_reduction.py \
  tools/test_qwen38_collective_reduction.py
git -c core.hooksPath=/dev/null commit \
  -m "feat(profiling): classify TP collective reduction" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

---

### Task 6: Implement the source-bound worker

**Files:**

- Create: `tools/qwen38_tp4_collective_reduction_worker.py`
- Create: `tools/test_qwen38_tp4_collective_reduction_worker.py`

**Interfaces:**

- Reuses request construction and lifecycle measurement semantics from
  `tools/qwen38_tp4_communication_profile_worker.py`.
- Produces schema
  `qwen38.tp4-collective-reduction-worker.v1`.
- Produces one bounded case artifact per arm and a cleanup receipt.

- [ ] **Step 1: Write failing worker-contract tests**

Add concrete worker-contract tests:

```python
def test_build_cases_freezes_calibration_and_terminal_matrix():
    cases = build_collective_reduction_cases(selected_budget=16)
    assert {row["budget"] for row in cases["calibration"]} == {
        0, 8, 16, 32
    }
    assert {
        row["workload"] for row in cases["calibration"]
    } == {"P0", "P1", "Q1"}
    assert {
        row["workload"] for row in cases["terminal"]
    } == {"P0", "P1", "Q0", "Q1", "Q2"}


def test_worker_preserves_exact_request_outputs_across_pair():
    result = _run_pair(control_tokens=[7, 8], instrumented_tokens=[7, 8])
    assert result["classification"] == "PASS"
    with pytest.raises(RuntimeError, match="output mismatch"):
        _run_pair(control_tokens=[7, 8], instrumented_tokens=[7, 9])
```

Add exact fake-engine assertions that the control arm disables both
profilers, budget zero creates count-only rows, the timed arm uses the
selected budget, snapshots are ordered ranks 0 through 3, `engine.exit()` is
called once after failure, and the case sink receives the full artifact while
the campaign retains only bounded receipts.

Freeze calibration workloads to `P0`, `P1`, and `Q1`, budgets to
`0, 8, 16, 32`, and terminal workloads to all five frozen workloads.

- [ ] **Step 2: Run worker tests and confirm RED**

Run:

```bash
/opt/homebrew/bin/python3.12 -m pytest \
  tools/test_qwen38_tp4_collective_reduction_worker.py -q
```

Expected: collection fails because the worker does not exist.

- [ ] **Step 3: Implement paired request execution**

For every pair:

1. reset sequence IDs;
2. configure the control or census policy on all ranks;
3. reset peak memory;
4. add deterministic requests;
5. record TTFT, token timestamps, TPOT, E2E, output IDs, and decode wall time;
6. finalize census after request completion;
7. collect rank-local memory;
8. atomically stream the case JSON;
9. preserve one engine for compatible adjacent cases;
10. call `engine.exit()` exactly once in `finally`.

Do not enable `DecodeInternalProfiler` or Nsight.

- [ ] **Step 4: Run Task 6 GREEN**

Run:

```bash
/opt/homebrew/bin/python3.12 -m pytest \
  tools/test_qwen38_tp4_collective_reduction_worker.py \
  tools/test_qwen38_tp4_communication_profile_worker.py -q
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-collective-worker-pycache \
  /opt/homebrew/bin/python3.12 -m py_compile \
  tools/qwen38_tp4_collective_reduction_worker.py
```

Expected: new worker tests and adjacent communication-worker tests pass.

- [ ] **Step 5: Commit and push Task 6**

```bash
git add -- \
  tools/qwen38_tp4_collective_reduction_worker.py \
  tools/test_qwen38_tp4_collective_reduction_worker.py
git -c core.hooksPath=/dev/null commit \
  -m "feat(profiling): add TP4 collective census worker" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

---

### Task 7: Implement the controller and safe remote plan

**Files:**

- Create: `tools/run_qwen38_tp4_collective_reduction.py`
- Create: `tools/test_run_qwen38_tp4_collective_reduction.py`

**Interfaces:**

- Reuses public remote helpers and constants from
  `tools/run_qwen38_tp4_communication_profile.py`.
- Produces plan schema `qwen38.tp4-collective-reduction-plan.v1`.
- Supports `--plan-only`, `--dry-run`, bounded monitoring, execution,
  resumable case receipts, assembly, remote verification, and download.

- [ ] **Step 1: Write failing plan and safety tests**

Add concrete plan and safety tests:

```python
def test_plan_keeps_every_path_below_approved_remote_root():
    plan = _build_plan()
    paths = _all_remote_paths(plan)
    assert paths
    assert all(
        PurePosixPath(path).is_relative_to(
            PurePosixPath(APPROVED_REMOTE_ROOT)
        )
        for path in paths
    )


def test_plan_contains_no_nsys_command_or_path():
    encoded = json.dumps(_build_plan(), sort_keys=True).lower()
    assert "nsys" not in encoded
    assert "async_op" not in encoded
    assert "communication_stream" not in encoded


def test_controller_never_runs_forbidden_auth_or_signal_commands():
    commands = _flatten_remote_commands(_build_plan())
    basenames = {PurePosixPath(row[0]).name for row in commands}
    assert basenames.isdisjoint({"kinit", "krenew", "kill", "pkill", "killall"})
```

Add mutations for reused tags, symlink escape, source/model revision drift,
low Kerberos TTL, changed GPU UUIDs, memory above 1,024 MiB, utilization above
5%, and unrelated compute processes. Assert each mutation fails before worker
launch.

- [ ] **Step 2: Run controller tests and confirm RED**

Run:

```bash
/opt/homebrew/bin/python3.12 -m pytest \
  tools/test_run_qwen38_tp4_collective_reduction.py -q
```

Expected: collection fails because the controller does not exist.

- [ ] **Step 3: Implement immutable plan and dry-run**

The plan contains:

```python
{
    "schema_version": "qwen38.tp4-collective-reduction-plan.v1",
    "attempt_tag": attempt_tag,
    "source_revision": source_revision,
    "model_revision": model_revision,
    "remote_root": APPROVED_REMOTE_ROOT,
    "event_budgets": [0, 8, 16, 32],
    "median_overhead_ceiling": 0.03,
    "maximum_overhead_ceiling": 0.05,
    "minimum_lower_bound_opportunity": 0.05,
    "overlap_design_authorized": False,
    "async_collectives_authorized": False,
}
```

All writes use attempt-scoped atomic paths. Plan-only and dry-run start no
GPU worker.

- [ ] **Step 4: Write RED tests for execution ordering and cleanup**

Require:

```text
correctness
calibration controls and budgets
budget selection
terminal warmups and measured pairs
assembly
remote independent verification
download
local independent verification
cleanup validation
```

Reject partial case success, dirty entry inventory, rank drift, owned
children remaining, missing artifact, or producer/verifier disagreement.

- [ ] **Step 5: Implement resumable execution**

Resume only immutable completed case receipts from the same source, model,
attempt, rank map, and policy identity. Never duplicate a live worker.
Cleanup may act only on exact owned PIDs recorded by this attempt.

- [ ] **Step 6: Run Task 7 GREEN and existing controller regressions**

Run separate processes:

```bash
/opt/homebrew/bin/python3.12 -m pytest \
  tools/test_run_qwen38_tp4_collective_reduction.py -q
/opt/homebrew/bin/python3.12 -m pytest \
  tools/test_run_qwen38_tp4_communication_profile.py -q
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-collective-controller-pycache \
  /opt/homebrew/bin/python3.12 -m py_compile \
  tools/run_qwen38_tp4_collective_reduction.py
```

Expected: both controller suites pass and compile exits zero.

- [ ] **Step 7: Commit and push Task 7**

```bash
git add -- \
  tools/run_qwen38_tp4_collective_reduction.py \
  tools/test_run_qwen38_tp4_collective_reduction.py
git -c core.hooksPath=/dev/null commit \
  -m "feat(profiling): orchestrate TP4 collective gate" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

---

### Task 8: Assemble and independently verify the terminal bundle

**Files:**

- Create: `tools/assemble_qwen38_tp4_collective_reduction.py`
- Create: `tools/verify_qwen38_tp4_collective_reduction.py`
- Create: `tools/test_assemble_qwen38_tp4_collective_reduction.py`
- Create: `tools/test_verify_qwen38_tp4_collective_reduction.py`

**Interfaces:**

- Assembler consumes immutable case artifacts and writes the exact artifact
  contract from the spec.
- Verifier consumes only the terminal bundle and independently reconstructs
  every derived metric and classification.

- [ ] **Step 1: Write failing assembler tests**

Create a minimal four-rank, five-workload fixture. Assert exact output names:

```text
source_identity.json
model_manifest.json
gpu_topology.json
workload_manifest.json
static_collective_catalog.json
consumer_dependency_proofs.json
profiler_calibration.json
collective_census.jsonl
collective_timing_samples.jsonl
paired_online_metrics.json
correctness.jsonl
resource_samples.jsonl
reduction_ceiling.json
classification.json
cleanup.json
manifest.sha256
```

Reject missing/extra cases, output drift, rank divergence, duplicate JSON
keys, non-finite values, and incomplete cleanup.

- [ ] **Step 2: Run assembler tests and confirm RED**

Run:

```bash
/opt/homebrew/bin/python3.12 -m pytest \
  tools/test_assemble_qwen38_tp4_collective_reduction.py -q
```

Expected: import fails because the assembler does not exist.

- [ ] **Step 3: Implement producer assembly**

Use canonical compact JSON with `allow_nan=False`, atomic writes, explicit
schema versions, and a manifest that rejects extra terminal files. Invoke
only pure functions from `tools/qwen38_collective_reduction.py` for catalog,
proof, ceiling, and classification.

- [ ] **Step 4: Write failing independent-verifier tests**

Test a valid bundle and one mutation for each authority:

```text
source/model identity
GPU rank map
workload matrix
catalog
consumer proof
count/byte sequence
timing cohort
overhead
correctness
resource identity
cleanup
classification
manifest
```

Assert producer and verifier JSON classification equality.

- [ ] **Step 5: Implement independent verification**

Do not import assembler functions. The verifier may import constants and pure
schema validators from `tools/qwen38_collective_reduction.py`, but it must
independently load, normalize, aggregate, hash, and classify artifacts. Write
`independent_verification.json` atomically, then rewrite the manifest to
include it.

- [ ] **Step 6: Run Task 8 GREEN**

Run:

```bash
/opt/homebrew/bin/python3.12 -m pytest \
  tools/test_qwen38_collective_reduction.py \
  tools/test_assemble_qwen38_tp4_collective_reduction.py \
  tools/test_verify_qwen38_tp4_collective_reduction.py -q
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-collective-bundle-pycache \
  /opt/homebrew/bin/python3.12 -m py_compile \
  tools/assemble_qwen38_tp4_collective_reduction.py \
  tools/verify_qwen38_tp4_collective_reduction.py
git diff --check -- \
  tools/assemble_qwen38_tp4_collective_reduction.py \
  tools/verify_qwen38_tp4_collective_reduction.py \
  tools/test_assemble_qwen38_tp4_collective_reduction.py \
  tools/test_verify_qwen38_tp4_collective_reduction.py
```

Expected: all tests pass; compile and whitespace checks exit zero.

- [ ] **Step 7: Commit and push Task 8**

```bash
git add -- \
  tools/assemble_qwen38_tp4_collective_reduction.py \
  tools/verify_qwen38_tp4_collective_reduction.py \
  tools/test_assemble_qwen38_tp4_collective_reduction.py \
  tools/test_verify_qwen38_tp4_collective_reduction.py
git -c core.hooksPath=/dev/null commit \
  -m "feat(profiling): verify TP4 collective reduction bundle" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

---

### Task 9: Full local qualification preflight

**Files:**

- Modify only if a verified defect is found in Task 1-8 files.
- Create attempt-scoped local controller receipts under
  `artifacts/qwen38_tp4_collective_reduction/20260827-qwen38-tp4-collective-reduction-r1/controller/`.

**Interfaces:**

- Consumes all Task 1-8 public interfaces.
- Produces a frozen source identity and a plan-only/dry-run receipt.

- [ ] **Step 1: Run the complete focused test matrix**

Run tests in isolated processes where existing module stubs require it:

```bash
/opt/homebrew/bin/python3.12 -m pytest tools/test_synchronous_collective_census.py -q
/opt/homebrew/bin/python3.12 -m pytest tools/test_decode_internal_profiler.py -q
/opt/homebrew/bin/python3.12 -m pytest tools/test_decode_internal_profile_wiring.py -q
/opt/homebrew/bin/python3.12 -m pytest tools/test_tensor_parallel_greedy.py -q
/opt/homebrew/bin/python3.12 -m pytest tools/test_model_runner_spec_verify.py -q
/opt/homebrew/bin/python3.12 -m pytest tools/test_qwen38_collective_reduction.py -q
/opt/homebrew/bin/python3.12 -m pytest tools/test_qwen38_tp4_collective_reduction_worker.py -q
/opt/homebrew/bin/python3.12 -m pytest tools/test_run_qwen38_tp4_collective_reduction.py -q
/opt/homebrew/bin/python3.12 -m pytest tools/test_assemble_qwen38_tp4_collective_reduction.py -q
/opt/homebrew/bin/python3.12 -m pytest tools/test_verify_qwen38_tp4_collective_reduction.py -q
```

Expected: every process exits zero.

- [ ] **Step 2: Run static prohibition and source checks**

```bash
rg -n \
  "async_op=True|new_stream|Stream\\(|reduce_scatter|all_gather_into_tensor" \
  tinyvllm/engine/synchronous_collective_census.py \
  tools/qwen38_collective_reduction.py \
  tools/qwen38_tp4_collective_reduction_worker.py \
  tools/run_qwen38_tp4_collective_reduction.py
git diff --check -- \
  tinyvllm/engine/synchronous_collective_census.py \
  tinyvllm/engine/decode_internal_profiler.py \
  tinyvllm/engine/model_runner.py \
  tinyvllm/engine/llm_engine.py \
  tinyvllm/engine/tensor_parallel_greedy.py \
  tinyvllm/layers/linear.py \
  tinyvllm/layers/embed_head.py \
  tools/qwen38_collective_reduction.py \
  tools/qwen38_tp4_collective_reduction_worker.py \
  tools/run_qwen38_tp4_collective_reduction.py \
  tools/assemble_qwen38_tp4_collective_reduction.py \
  tools/verify_qwen38_tp4_collective_reduction.py
```

Expected: the prohibition search has no implementation hit; documentation or
explicit rejection strings are reviewed manually. `git diff --check` exits
zero.

- [ ] **Step 3: Commit any verified preflight fixes**

Stage only the exact files changed by a reproduced defect. Use the focused
subject `fix(profiling): correct TP4 collective gate defect` with the required
single trailer, then push. Skip this step when there is no change.

- [ ] **Step 4: Freeze a new attempt and run plan-only**

Use a never-before-used tag:

```text
20260827-qwen38-tp4-collective-reduction-r1
```

Run the controller with the immutable Qwen3.8 model revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, current source revision, approved
remote root, and `--plan-only`. Verify that no remote worker starts.

- [ ] **Step 5: Run read-only preflight and dry-run**

Check:

```text
Kerberos TGT remaining lifetime >= 5,400 seconds
approved remote root resolves below itself
model manifest and immutable revision match
four strict-clean GPUs selected by UUID
no task-owned worker already active for the tag
no attempt directory already exists
```

If any check fails, preserve the receipt and stop without a remote write or
GPU worker.

- [ ] **Step 6: Commit and push the source-bound preflight receipts**

Stage only bounded controller receipts intended for version control. Do not
stage raw logs, model data, or unrelated artifacts.

---

### Task 10: Execute the remote qualification and classify

**Files:**

- Create terminal attempt artifacts only beneath the approved remote root and
  the matching local artifact directory.
- Modify implementation files only for reproduced defects, each through a
  focused RED/GREEN cycle.

**Interfaces:**

- Consumes the frozen plan and source.
- Produces producer classification, remote independent verification, local
  independent verification, cleanup receipt, and immutable manifest.

- [ ] **Step 1: Start or resume exactly one frozen attempt**

Before launch, repeat strict-clean UUID admission. If the exact attempt
already has a live owned worker, monitor it instead of launching another.
Never substitute a new source revision into the same attempt.

- [ ] **Step 2: Complete calibration before terminal timing**

Run matched control/count/timed pairs for budgets `0`, `8`, `16`, and `32`.
Select the largest nonzero budget satisfying median overhead at most 3% and
maximum overhead at most 5%. If none passes, finish the bundle as
`INCONCLUSIVE_PROFILER_OVERHEAD`; do not launch unnecessary terminal timing.

- [ ] **Step 3: Complete all five terminal workloads**

For P0, P1, Q0, Q1, and Q2, run two warmups and five alternating measured
control/instrumented pairs. Require exact output equality and four-rank
census agreement.

- [ ] **Step 4: Assemble and run the remote verifier**

Require a complete exact manifest and matching producer/verifier
classification. Do not interpret partial artifacts.

- [ ] **Step 5: Download the bounded bundle and run the local verifier**

The bundle contains no Nsight SQLite traces. Hash every artifact, run the
independent verifier locally, and require byte-identical remote/local
verification JSON.

- [ ] **Step 6: Confirm terminal cleanup**

Check exact owned PIDs, process-group destruction, temporary staging removal,
and three read-only exact-tag process scans. Do not signal unrelated
processes.

---

### Task 11: Audit, reconcile, commit, and push the terminal result

**Files:**

- Create:
  `docs/superpowers/audits/2026-08-27-qwen38-tp4-synchronous-collective-reduction-audit.md`
- Modify: `AGENT_HANDOFF_STATE.md`
- Add only bounded terminal controller/verifier artifacts explicitly selected
  for source control.

**Interfaces:**

- Consumes the immutable terminal bundle and both verifier outputs.
- Produces the final claim boundary and exact next command.

- [ ] **Step 1: Write the prompt-to-artifact audit**

Include:

- source/model/GPU/workload identities;
- 130-site expected versus observed inventory;
- calibration overhead by budget;
- selected event budget;
- each consumer proof;
- candidate benefit lower/upper bound and all costs;
- correctness, resource, and cleanup results;
- producer/remote/local classifications and hashes;
- explicit overlap and async prohibition;
- exact authorized next action.

- [ ] **Step 2: Append the handoff reconciliation**

Use exactly one terminal block. Write the actual value from
`classification.json` for `PRODUCER_CLASSIFICATION`. Set candidate-design
authorization to `true` only when that value is
`GO_SYNC_COLLECTIVE_REDUCTION`; otherwise set it to `false`. Select the next
command from the two literal values shown:

```text
QWEN38_TP4_COLLECTIVE_REDUCTION_QUALIFICATION=COMPLETE
PRODUCER_CLASSIFICATION=one of GO_SYNC_COLLECTIVE_REDUCTION, NO_GO_NO_REDUCIBLE_COLLECTIVE, INCONCLUSIVE_PROFILER_OVERHEAD, INCONCLUSIVE_INCOMPLETE_COVERAGE
REMOTE_VERIFIER=PASS
LOCAL_VERIFIER=PASS
OVERLAP_DESIGN_AUTHORIZED=false
ASYNC_COLLECTIVES_AUTHORIZED=false
SYNC_COLLECTIVE_CANDIDATE_DESIGN_AUTHORIZED=true or false by the rule above
NEXT_COMMAND=write the named candidate-specific design or select another optimization
```

- [ ] **Step 3: Run final verification**

Run all ten focused test files from Task 9, `py_compile` for every new Python
file, `git diff --check`, prompt-to-artifact checks, manifest verification,
local verifier, exact tracked status, and exact staged-file inventory.

- [ ] **Step 4: Commit and push the terminal reconciliation**

```bash
git add -- \
  docs/superpowers/audits/2026-08-27-qwen38-tp4-synchronous-collective-reduction-audit.md \
  AGENT_HANDOFF_STATE.md \
  artifacts/qwen38_tp4_collective_reduction/20260827-qwen38-tp4-collective-reduction-r1/controller/source_identity.json \
  artifacts/qwen38_tp4_collective_reduction/20260827-qwen38-tp4-collective-reduction-r1/controller/plan.json \
  artifacts/qwen38_tp4_collective_reduction/20260827-qwen38-tp4-collective-reduction-r1/controller/plan_audit.json \
  artifacts/qwen38_tp4_collective_reduction/20260827-qwen38-tp4-collective-reduction-r1/controller/ssh_storage_preflight.json \
  artifacts/qwen38_tp4_collective_reduction/20260827-qwen38-tp4-collective-reduction-r1/controller/strict_clean_admission.json \
  artifacts/qwen38_tp4_collective_reduction/20260827-qwen38-tp4-collective-reduction-r1/controller/producer_result.json \
  artifacts/qwen38_tp4_collective_reduction/20260827-qwen38-tp4-collective-reduction-r1/controller/remote-independent-verification.json \
  artifacts/qwen38_tp4_collective_reduction/20260827-qwen38-tp4-collective-reduction-r1/controller/local-independent-verification.json \
  artifacts/qwen38_tp4_collective_reduction/20260827-qwen38-tp4-collective-reduction-r1/controller/remote-post-verification-manifest.json \
  artifacts/qwen38_tp4_collective_reduction/20260827-qwen38-tp4-collective-reduction-r1/controller/prompt_to_artifact_checklist.md
git -c core.hooksPath=/dev/null commit \
  -m "docs(profiling): audit TP4 collective reduction" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

Verify local HEAD, tracking branch, and GitHub branch SHA are identical.

## Plan Completion Boundary

This plan completes only when the terminal bundle, producer classification,
remote verifier, local verifier, manifest, audit, handoff, commit, and push
all exist and agree.

If the result is `GO_SYNC_COLLECTIVE_REDUCTION`, stop before changing
production collective semantics and write the candidate-specific design
named by the evidence.

If the result is `NO_GO_NO_REDUCIBLE_COLLECTIVE`,
`INCONCLUSIVE_PROFILER_OVERHEAD`, or
`INCONCLUSIVE_INCOMPLETE_COVERAGE`, do not implement a collective mutation
from this evidence.
