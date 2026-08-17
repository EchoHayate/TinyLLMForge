# Generic Speculative Runtime TP4 Correctness and Collective Authority Implementation Plan

> **For agentic workers:** Execute this plan inline in the current session. Do not dispatch subagents, create a worktree, switch branches, stage, commit, stash, reset, clean, or push.

**Goal:** Build and run a source-bound real TP4 authority proving exact baseline-versus-generic-n-gram output parity, four-rank speculative callback and collective execution, four-rank residency acknowledgements, and clean shutdown.

**Architecture:** Extend the existing acknowledged internal profiler with explicit scopes around the two speculative ModelRunner callbacks. Add a new TP4-only contract, worker, orchestrator, verifier, and serial Kerberos remote runner; keep the TP1 blockwise schema and artifacts unchanged.

**Tech Stack:** Python 3, PyTorch CUDA/NCCL, TinyLLMForge `LLMEngine`, `DecodeInternalProfiler`, generic `NGramDraftAdapter`, JSON artifacts, Bash, SSH/GSSAPI.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Use `apply_patch` for every file modification.
- Follow strict RED then GREEN for every behavior change.
- Do not use subagents.
- Do not create or switch branches or worktrees.
- Do not stage, commit, stash, reset, clean, or push.
- Do not modify unrelated trailing whitespace in `tinyvllm/engine/model_runner.py`.
- Remote host is exactly `sitian@10.232.195.203`.
- Remote authentication is `KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian`.
- Every SSH command uses `ControlMaster=no` and `ControlPath=none`.
- Keep SSH actions serial and do not create persistent local TTY sessions.
- Perform a real four-GPU preflight; do not assume the prior GPU 7 path expands to TP4.
- Use a new TP4 schema and artifact directory; do not change the TP1 blockwise schema.
- Classification remains `NOT_PROMOTABLE`.
- Do not claim TP4 performance, 16K/32K performance direction, second-model support, learned-drafter or MTP plus offload support, KV8/KV4 support, or Phase 1 completion.

---

## File Map

- Modify `tinyvllm/engine/model_runner.py`
  - Enter the existing internal profiler around speculative first-target and
    verification callbacks without changing rank-0 token authority.
- Modify `tools/test_model_runner_spec_verify.py`
  - Add isolated dependency-light tests for callback profile scopes and worker
    `None` results.
- Create `tools/generic_speculative_tp4_gate.py`
  - Own constants, schema validation, source hashing, prompt construction,
    parity classification, atomic publication, and subprocess orchestration.
- Create `tools/generic_speculative_tp4_worker.py`
  - Construct one TP4 Engine cell, capture runtime/KV/profile/residency/cleanup
    evidence, and write one validated cell result.
- Create `tools/verify_generic_speculative_tp4_gate.py`
  - Independently reload and verify the final artifact.
- Create `tools/run_generic_speculative_tp4_gate_remote.sh`
  - Perform serial Kerberos SSH, four-GPU preflight, fresh-port selection,
    detached campaign lifecycle, download, and remote/local verification.
- Create `tools/test_generic_speculative_tp4_gate.py`
  - Cover the new contract, worker seams, verifier, and remote runner text
    contract.
- Modify `docs/superpowers/audits/2026-08-12-phase1-objective-coverage.md`
  - Record the exact TP4 result and remaining limits only after the real run.
- Modify `AGENT_HANDOFF_STATE.md`
  - Record commands, artifact paths, hashes, evidence, limitations, and the
    next clear action only after the real run.

---

### Task 1: Profile Speculative Callbacks on Every Rank

**Files:**
- Modify: `tools/test_model_runner_spec_verify.py`
- Modify: `tinyvllm/engine/model_runner.py:5877`
- Modify: `tinyvllm/engine/model_runner.py:6257`

**Interfaces:**
- Consumes: `run_profiled_step(profiler, *, batch_kind, is_decode, active_sequence_count, request_set_sha256, dispatch, call)`.
- Produces: profile step rows with `batch_kind` equal to
  `spec_first_target` or `spec_verify`, one row per callback invocation per
  rank.

- [x] **Step 1: Add failing tests for first-target profile scope**

Add focused tests that install a fake profiler and stub the existing
first-target dependencies:

```python
def test_run_spec_first_target_batch_profiles_rank_zero_callback():
    runner, profiler = _profiled_first_target_runner(rank=0)
    rows = runner.run_spec_first_target_batch(
        tuple(_first_target_sequences(0, 0)),
    )
    assert rows is not None
    assert profiler.steps == [{
        "batch_kind": "spec_first_target",
        "is_decode": True,
        "active_sequence_count": 2,
        "request_set_sha256": _request_set_sha256((0, 1)),
        "dispatch": "eager",
    }]


def test_run_spec_first_target_batch_profiles_worker_before_none_result():
    runner, profiler = _profiled_first_target_runner(rank=1)
    rows = runner.run_spec_first_target_batch(
        tuple(_first_target_sequences(0, 0)),
    )
    assert rows is None
    assert profiler.steps[0]["batch_kind"] == "spec_first_target"
```

The helper computes the digest exactly as production code:

```python
def _request_set_sha256(sequence_ids):
    return hashlib.sha256(
        json.dumps(
            sorted(int(value) for value in sequence_ids),
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
```

- [ ] **Step 2: Run the isolated first-target tests and verify RED**

Run:

```bash
PYTHONPATH=$PWD python3 -m pytest \
  tools/test_model_runner_spec_verify.py \
  -k 'profiles_rank_zero_callback or profiles_worker_before_none_result' \
  -q
```

Expected: both tests fail because no `spec_first_target` profile step is
created.

- [x] **Step 3: Add failing tests for verification profile scope**

Add:

```python
def test_run_spec_verify_batch_profiles_rank_zero_callback():
    runner, profiler = _profiled_verify_runner(rank=0)
    rows = runner.run_spec_verify_batch(tuple(_tail_items()))
    assert rows is not None
    assert profiler.steps == [{
        "batch_kind": "spec_verify",
        "is_decode": True,
        "active_sequence_count": len(_tail_items()),
        "request_set_sha256": _request_set_sha256(
            item.sequence_id for item in _tail_items()
        ),
        "dispatch": "eager",
    }]


def test_run_spec_verify_batch_profiles_worker_before_none_result():
    runner, profiler = _profiled_verify_runner(rank=2)
    rows = runner.run_spec_verify_batch(tuple(_tail_items()))
    assert rows is None
    assert profiler.steps[0]["batch_kind"] == "spec_verify"
```

- [ ] **Step 4: Run the isolated verification tests and verify RED**

Run:

```bash
PYTHONPATH=$PWD python3 -m pytest \
  tools/test_model_runner_spec_verify.py \
  -k 'run_spec_verify_batch_profiles' \
  -q
```

Expected: both tests fail because no `spec_verify` profile step is created.

- [x] **Step 5: Add a canonical request-set digest helper**

Add near the existing ModelRunner callback helpers:

```python
def _profile_request_set_sha256(sequence_ids) -> str:
    return hashlib.sha256(
        json.dumps(
            sorted(int(value) for value in sequence_ids),
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
```

- [x] **Step 6: Split first-target execution from the profile wrapper**

Use `apply_patch` to rename the current
`run_spec_first_target_batch()` definition to
`_run_spec_first_target_batch()`. Keep its complete body, including the
`finally: reset_context()` block, byte-for-byte unchanged. Then add this
public wrapper immediately after it:

```python
def run_spec_first_target_batch(
    self,
    seqs,
    return_hidden=False,
    return_logits=False,
    kv_block_identity_rows=(),
):
    return run_profiled_step(
        self.decode_internal_profiler,
        batch_kind="spec_first_target",
        is_decode=True,
        active_sequence_count=len(seqs),
        request_set_sha256=_profile_request_set_sha256(
            seq.seq_id for seq in seqs
        ),
        dispatch="eager",
        call=lambda: self._run_spec_first_target_batch(
            seqs,
            return_hidden,
            return_logits,
            kv_block_identity_rows,
        ),
    )
```

- [x] **Step 7: Split verification execution from the profile wrapper**

Use `apply_patch` to rename the current `run_spec_verify_batch()` definition
to `_run_spec_verify_batch()`. Keep its complete body, including residency
materialization, rank-0-only result construction, worker `None`, and
`finally: reset_context()`, byte-for-byte unchanged. Then add:

```python
def run_spec_verify_batch(self, items, residency_ticket_id=None):
    return run_profiled_step(
        self.decode_internal_profiler,
        batch_kind="spec_verify",
        is_decode=True,
        active_sequence_count=len(items),
        request_set_sha256=_profile_request_set_sha256(
            item.sequence_id for item in items
        ),
        dispatch="eager",
        call=lambda: self._run_spec_verify_batch(
            items,
            residency_ticket_id,
        ),
    )
```

- [x] **Step 8: Run focused tests and verify GREEN**

Run:

```bash
PYTHONPATH=$PWD python3 -m pytest \
  tools/test_model_runner_spec_verify.py \
  -k 'profiles_rank_zero_callback or profiles_worker_before_none_result or run_spec_verify_batch_profiles' \
  -q
```

Expected: all new tests pass.

- [x] **Step 9: Run existing callback regression tests**

Run:

```bash
PYTHONPATH=$PWD python3 -m pytest \
  tools/test_model_runner_spec_verify.py \
  -k 'run_spec_first_target_batch or run_spec_verify_batch' \
  -q
```

Expected: all selected existing and new tests pass.

---

### Task 2: Define the TP4 Authority Contract

**Files:**
- Create: `tools/generic_speculative_tp4_gate.py`
- Create: `tools/test_generic_speculative_tp4_gate.py`

**Interfaces:**
- Produces:
  - `validate_cell_result(value: object) -> dict`
  - `validate_rank_profile(value: object, *, policy: str) -> dict`
  - `validate_cleanup_receipt(value: object) -> dict`
  - `validate_result(value: object) -> dict`
  - `source_tree_sha256(root: Path, files: tuple[str, ...]) -> str`
  - `atomic_write_json(path: Path, value: object) -> None`

- [x] **Step 1: Add failing tests for constants and canonical cell keys**

Add:

```python
def test_tp4_contract_is_independent_and_not_promotable():
    assert gate.SCHEMA_VERSION == 1
    assert gate.CLASSIFICATION == "NOT_PROMOTABLE"
    assert gate.WORLD_SIZE == 4
    assert gate.POLICIES == ("baseline", "ngram")
    assert gate.CONTEXT_TOKENS == 4096
    assert gate.BATCH_SIZES == (1, 4)
    assert gate.MAX_OUTPUT_TOKENS == 8
    assert gate.cell_key("ngram", 4) == "ngram:b4"
```

- [ ] **Step 2: Run the contract test and verify RED**

Run:

```bash
PYTHONPATH=$PWD python3 -m pytest \
  tools/test_generic_speculative_tp4_gate.py::test_tp4_contract_is_independent_and_not_promotable \
  -q
```

Expected: import or attribute failure because the gate module does not exist.

- [x] **Step 3: Add the minimal constants and cell-key implementation**

Create:

```python
SCHEMA_VERSION = 1
CLASSIFICATION = "NOT_PROMOTABLE"
WORLD_SIZE = 4
POLICIES = ("baseline", "ngram")
CONTEXT_TOKENS = 4096
BATCH_SIZES = (1, 4)
MAX_OUTPUT_TOKENS = 8
NGRAM_SIZE = 3
MAX_PROPOSAL_TOKENS = 4


def cell_key(policy: str, batch_size: int) -> str:
    if policy not in POLICIES:
        raise ValueError("unsupported policy")
    if batch_size not in BATCH_SIZES:
        raise ValueError("unsupported batch size")
    return f"{policy}:b{batch_size}"
```

- [x] **Step 4: Add failing rank-profile tests**

Cover:

```python
def test_validate_rank_profile_requires_four_matching_ranks():
    profile = _valid_rank_profile(policy="ngram")
    assert gate.validate_rank_profile(
        profile,
        policy="ngram",
    )["rank_inventory"] == [0, 1, 2, 3]


@pytest.mark.parametrize(
    "mutation, message",
    [
        (_drop_rank(3), "rank inventory"),
        (_duplicate_rank(2), "rank inventory"),
        (_drop_callback("spec_verify"), "callback"),
        (_change_request_hash(rank=1), "callback identity"),
        (_drop_collective(rank=2), "collective"),
        (_change_collective_operation(rank=3), "collective identity"),
    ],
)
def test_validate_rank_profile_fails_closed(mutation, message):
    with pytest.raises(ValueError, match=message):
        gate.validate_rank_profile(
            mutation(_valid_rank_profile(policy="ngram")),
            policy="ngram",
        )
```

The baseline fixture contains no `spec_first_target` or `spec_verify` rows.

- [ ] **Step 5: Run rank-profile tests and verify RED**

Run:

```bash
PYTHONPATH=$PWD python3 -m pytest \
  tools/test_generic_speculative_tp4_gate.py \
  -k 'rank_profile' \
  -q
```

Expected: failures because `validate_rank_profile` is missing.

- [x] **Step 6: Implement fail-closed rank and collective validation**

Implement normalization around these exact identities:

```python
SPECULATIVE_BATCH_KINDS = (
    "spec_first_target",
    "spec_verify",
)


def _callback_identity(step):
    return (
        step["step_index"],
        step["batch_kind"],
        step["decode_ordinal"],
        step["active_sequence_count"],
        step["request_set_sha256"],
        step["dispatch"],
    )


def _collective_identity(row):
    return (
        row["step_index"],
        row["decode_ordinal"],
        row["operation"],
        tuple(row["tensor_shape"]),
        row["tensor_dtype"],
    )
```

Require candidate callback and collective identities to match rank 0 after
removing only rank and timing fields. Require the baseline to contain no
speculative callback rows.

- [x] **Step 7: Add failing residency and cleanup tests**

Add:

```python
def test_validate_cell_requires_ordered_four_rank_residency_phases():
    cell = _valid_cell(policy="ngram", batch_size=1)
    assert [
        phase["operation"]
        for phase in gate.validate_cell_result(cell)["residency_phases"]
    ] == ["prepare", "precommit", "seal"]


def test_validate_cleanup_requires_all_four_rank_receipts():
    receipt = _valid_cleanup_receipt()
    assert gate.validate_cleanup_receipt(receipt)[
        "process_group_destroyed"
    ] is True
    receipt["rank_cleanup_receipts"].pop()
    with pytest.raises(ValueError, match="cleanup rank inventory"):
        gate.validate_cleanup_receipt(receipt)
```

- [ ] **Step 8: Run residency and cleanup tests and verify RED**

Run:

```bash
PYTHONPATH=$PWD python3 -m pytest \
  tools/test_generic_speculative_tp4_gate.py \
  -k 'residency or cleanup' \
  -q
```

Expected: failures because cell and cleanup validators are missing.

- [x] **Step 9: Implement cell and cleanup validation**

Require:

```python
EXPECTED_RANKS = (0, 1, 2, 3)
EXPECTED_ACK_RANKS = (1, 2, 3)
SUCCESSFUL_RESIDENCY_OPERATIONS = (
    "prepare",
    "precommit",
    "seal",
)
```

For n-gram cells with positive verification callbacks, require successful
residency phases in that order. Each phase must contain exactly four rows and
the production fields:

```python
RESIDENCY_ROW_FIELDS = {
    "ticket_id",
    "participant_id",
    "operation",
    "status",
    "sequence_ids",
    "committed_block_identities",
    "rejected_block_identities",
    "detail",
}
```

Require cleanup values:

```python
receipt["process_group_destroyed"] is True
receipt["rank_exit_codes"] == [0, 0, 0, 0]
receipt["owned_children_remaining"] == []
[row["rank"] for row in receipt["rank_cleanup_receipts"]] == [0, 1, 2, 3]
```

- [x] **Step 10: Add failing top-level parity and source-binding tests**

Add:

```python
def test_validate_result_requires_exact_policy_parity():
    result = _valid_result()
    assert gate.validate_result(result)["parity"]["b1"] is True
    result["cells"]["ngram:b1"]["outputs"][0][0] += 1
    with pytest.raises(ValueError, match="output parity"):
        gate.validate_result(result)


def test_source_tree_hash_changes_with_bound_source(tmp_path):
    source = tmp_path / "a.py"
    source.write_text("one\n", encoding="utf-8")
    before = gate.source_tree_sha256(tmp_path, ("a.py",))
    source.write_text("two\n", encoding="utf-8")
    after = gate.source_tree_sha256(tmp_path, ("a.py",))
    assert before != after
```

- [x] **Step 11: Implement result validation, hashing, and atomic JSON**

Use canonical JSON and lowercase SHA-256:

```python
def atomic_write_json(path: Path, value: object) -> None:
    payload = (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)
```

The final validator compares `baseline:b1` with `ngram:b1` and
`baseline:b4` with `ngram:b4` exactly.

- [x] **Step 12: Run the complete contract test file and verify GREEN**

Run:

```bash
PYTHONPATH=$PWD python3 -m pytest \
  tools/test_generic_speculative_tp4_gate.py \
  -q
```

Expected: all contract tests implemented so far pass.

---

### Task 3: Implement One Real TP4 Cell Worker

**Files:**
- Create: `tools/generic_speculative_tp4_worker.py`
- Modify: `tools/test_generic_speculative_tp4_gate.py`

**Interfaces:**
- Consumes:
  - `gate.validate_cell_result`
  - `EngineSpeculativeRuntime`
  - `NGramDraftAdapter`
  - `engine.configure_decode_internal_profile`
  - `engine.finalize_decode_internal_profile`
  - `engine.kv_offload_summaries`
- Produces:
  - `run_policy_cell(..., engine_factory, sampling_params_type, runtime_type, adapter_type, synchronize) -> dict`

- [x] **Step 1: Add a failing worker lifecycle test**

Use a fake Engine that records method order:

```python
def test_worker_captures_profile_residency_kv_and_cleanup():
    engine = FakeTP4Engine()
    cell = worker.run_policy_cell(
        model_path="/model",
        gpu_indices=(0, 1, 2, 3),
        policy="ngram",
        batch_size=1,
        dist_port=29001,
        master_port=29002,
        engine_factory=lambda *args, **kwargs: engine,
        sampling_params_type=FakeSamplingParams,
        runtime_type=FakeRuntime,
        adapter_type=FakeAdapter,
        synchronize=lambda: None,
    )
    assert engine.calls == [
        "configure_profile",
        "warmup_generation",
        "clear_reusable_prefix_cache",
        "kv_before",
        "recorded_generation",
        "kv_after",
        "finalize_profile",
        "exit",
    ]
    assert [row["rank"] for row in cell["kv_rank_deltas"]] == [
        0, 1, 2, 3
    ]
    assert [
        row["operation"] for row in cell["residency_phases"]
    ] == ["prepare", "precommit", "seal"]
```

- [ ] **Step 2: Run the worker lifecycle test and verify RED**

Run:

```bash
PYTHONPATH=$PWD python3 -m pytest \
  tools/test_generic_speculative_tp4_gate.py::test_worker_captures_profile_residency_kv_and_cleanup \
  -q
```

Expected: import failure because the worker module does not exist.

- [x] **Step 3: Implement generation observation collection**

Reuse TP1 behavior with a local helper:

```python
def run_generation(
    *,
    engine,
    prompt_rows,
    sampling_params,
    expected_output_tokens,
    synchronize,
):
    for prompt_row in prompt_rows:
        engine.add_request(prompt_row["token_ids"], sampling_params)
    outputs_by_id = {}
    observations = []
    while not engine.is_finished():
        step_outputs, _ = engine.step()
        synchronize()
        observations.append(dict(engine.last_step_observation))
        for sequence_id, token_ids in step_outputs:
            outputs_by_id[int(sequence_id)] = list(token_ids)
    return {
        "outputs": [
            outputs_by_id[index]
            for index in sorted(outputs_by_id)
        ],
        "runtime": summarize_step_observations(observations),
    }
```

Validate idle state, output row count, output token count, and observation
shape before returning.

- [x] **Step 4: Implement production residency evidence wrapping**

Wrap only the Engine instance in the worker:

```python
@contextmanager
def capture_residency_phases(engine):
    captured = []
    original = engine._call_speculative_residency_phase

    def recorded(method_name, ticket_id, payload=None, **kwargs):
        rows = original(
            method_name,
            ticket_id,
            payload,
            **kwargs,
        )
        captured.append({
            "ticket_id": ticket_id,
            "operation": kwargs["expected_operation"],
            "status": kwargs["expected_status"],
            "rows": [dict(row) for row in rows],
        })
        return rows

    engine._call_speculative_residency_phase = recorded
    try:
        yield captured
    finally:
        engine._call_speculative_residency_phase = original
```

Handle the production call shape where `payload` is omitted by preserving the
original positional behavior instead of forcing `None` as an extra argument.

- [x] **Step 5: Implement TP4 Engine construction and cell execution**

Construct:

```python
engine = engine_factory(
    model_path,
    tensor_parallel_size=4,
    enforce_eager=True,
    max_model_len=4352,
    max_num_batched_tokens=16384,
    max_num_seqs=batch_size,
    max_num_prefill_tokens_per_step=1024,
    chunked_prefill_decode_first=False,
    chunked_prefill_mixed_batch=False,
    kv_offload_mvp0=True,
    kv_offload_gpu_blocks=68,
    kv_offload_logical_blocks=640,
    kv_offload_blockwise_decode=True,
    kv_offload_blockwise_prefill=True,
    kv_offload_blockwise_blocks=8,
)
```

Set `CUDA_VISIBLE_DEVICES`, `TINYVLLM_DIST_PORT`, and `MASTER_PORT` before
construction and restore the prior environment after cleanup.

For `policy == "ngram"` activate:

```python
runtime_type(
    adapter_type(
        ngram_size=3,
        max_proposal_tokens=4,
    )
)
```

Always place `cleanup_receipt = engine.exit()` in `finally`, attach it to the
cell, and validate the complete cell before returning.

- [x] **Step 6: Add failing tests for environment restoration and exit failure**

Add:

```python
def test_worker_restores_distributed_environment(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "9")
    monkeypatch.setenv("TINYVLLM_DIST_PORT", "19991")
    monkeypatch.setenv("MASTER_PORT", "19992")
    _run_fake_cell(FakeTP4Engine())
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "9"
    assert os.environ["TINYVLLM_DIST_PORT"] == "19991"
    assert os.environ["MASTER_PORT"] == "19992"


def test_worker_does_not_publish_cell_without_clean_exit():
    engine = FakeTP4Engine(rank_exit_codes=[0, 1, 0, 0])
    with pytest.raises(ValueError, match="rank exit codes"):
        _run_fake_cell(engine)
```

- [x] **Step 7: Run worker tests and verify GREEN**

Run:

```bash
PYTHONPATH=$PWD python3 -m pytest \
  tools/test_generic_speculative_tp4_gate.py \
  -k 'worker_' \
  -q
```

Expected: all worker seam tests pass.

---

### Task 4: Orchestrate Cells and Independently Verify the Artifact

**Files:**
- Modify: `tools/generic_speculative_tp4_gate.py`
- Create: `tools/verify_generic_speculative_tp4_gate.py`
- Modify: `tools/test_generic_speculative_tp4_gate.py`

**Interfaces:**
- Produces:
  - `run_campaign(*, model_path: str, gpu_indices: tuple[int, ...], output_dir: Path, dist_port_base: int, master_port_base: int) -> dict`
  - `verify_run(run_dir: Path, *, source_root: Path | None = None) -> dict`

- [x] **Step 1: Add a failing subprocess isolation test**

Add:

```python
def test_campaign_runs_each_cell_in_a_fresh_subprocess(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        gate.subprocess,
        "run",
        _fake_worker_subprocess(calls),
    )
    result = gate.run_campaign(
        model_path="/model",
        gpu_indices=(0, 1, 2, 3),
        output_dir=tmp_path / "run",
        dist_port_base=29100,
        master_port_base=29200,
    )
    assert [call["cell"] for call in calls] == [
        "baseline:b1",
        "ngram:b1",
        "baseline:b4",
        "ngram:b4",
    ]
    assert len({call["dist_port"] for call in calls}) == 4
    assert result["classification"] == "NOT_PROMOTABLE"
```

- [ ] **Step 2: Run the isolation test and verify RED**

Run:

```bash
PYTHONPATH=$PWD python3 -m pytest \
  tools/test_generic_speculative_tp4_gate.py::test_campaign_runs_each_cell_in_a_fresh_subprocess \
  -q
```

Expected: failure because `run_campaign` is missing.

- [x] **Step 3: Implement subprocess orchestration and atomic publication**

For each cell invoke:

```bash
MODEL_PATH="${MODEL_PATH:?MODEL_PATH must name the Qwen3-0.6B checkpoint}"
CELL_POLICY="${CELL_POLICY:?CELL_POLICY must be baseline or ngram}"
CELL_BATCH_SIZE="${CELL_BATCH_SIZE:?CELL_BATCH_SIZE must be 1 or 4}"
CELL_DIST_PORT="${CELL_DIST_PORT:?CELL_DIST_PORT must be unique}"
CELL_MASTER_PORT="${CELL_MASTER_PORT:?CELL_MASTER_PORT must be unique}"
CELL_RESULT_PATH="${CELL_RESULT_PATH:?CELL_RESULT_PATH must be temporary}"
python3 tools/generic_speculative_tp4_worker.py \
  --model "$MODEL_PATH" \
  --gpu-indices 0,1,2,3 \
  --policy "$CELL_POLICY" \
  --batch-size "$CELL_BATCH_SIZE" \
  --dist-port "$CELL_DIST_PORT" \
  --master-port "$CELL_MASTER_PORT" \
  --out "$CELL_RESULT_PATH"
```

Load each cell through `validate_cell_result`. Build the final result only
after all four cells pass. Write:

```text
$RUN_DIR/result.json
$RUN_DIR/source_manifest.json
$RUN_DIR/verify.json
```

Use a temporary sibling directory and `os.replace()` for final publication.

- [x] **Step 4: Add failing verifier tamper tests**

Add:

```python
def test_verifier_rejects_tampered_collective(tmp_path):
    run_dir = _write_valid_run(tmp_path)
    result = json.loads((run_dir / "result.json").read_text())
    result["cells"]["ngram:b1"]["profile"]["ranks"][2][
        "collectives"
    ][0]["operation"] = "tampered"
    (run_dir / "result.json").write_text(json.dumps(result))
    verification = verifier.verify_run(run_dir)
    assert verification["classification"] == "FAIL"
    assert "collective identity" in verification["failures"][0]


def test_verifier_rejects_source_hash_mismatch(tmp_path):
    run_dir = _write_valid_run(tmp_path)
    manifest = json.loads(
        (run_dir / "source_manifest.json").read_text()
    )
    manifest["source_tree_sha256"] = "0" * 64
    (run_dir / "source_manifest.json").write_text(
        json.dumps(manifest)
    )
    assert verifier.verify_run(run_dir)["classification"] == "FAIL"
```

- [ ] **Step 5: Run verifier tests and verify RED**

Run:

```bash
PYTHONPATH=$PWD python3 -m pytest \
  tools/test_generic_speculative_tp4_gate.py \
  -k 'verifier_' \
  -q
```

Expected: import or behavior failure because the verifier is missing.

- [x] **Step 6: Implement independent verification**

The verifier:

```python
def verify_run(run_dir, *, source_root=None):
    failures = []
    try:
        result = gate.validate_result(_read_json(run_dir / "result.json"))
        manifest = _validate_manifest(
            _read_json(run_dir / "source_manifest.json")
        )
        _verify_artifact_hashes(run_dir, manifest)
        if source_root is not None:
            _verify_current_source_tree(source_root, manifest)
    except Exception as error:
        failures.append(str(error))
    return {
        "classification": "PASS" if not failures else "FAIL",
        "failures": failures,
    }
```

It must not import the worker module or trust worker-derived classification.

- [x] **Step 7: Run orchestrator and verifier tests and verify GREEN**

Run:

```bash
PYTHONPATH=$PWD python3 -m pytest \
  tools/test_generic_speculative_tp4_gate.py \
  -k 'campaign_ or verifier_' \
  -q
```

Expected: all selected tests pass.

---

### Task 5: Add the Serial Kerberos TP4 Remote Runner

**Files:**
- Create: `tools/run_generic_speculative_tp4_gate_remote.sh`
- Modify: `tools/test_generic_speculative_tp4_gate.py`

**Interfaces:**
- Consumes: gate CLI and verifier CLI from Task 4.
- Produces: downloaded run directory with `result.json`,
  `source_manifest.json`, `verify.json`, and `verify.remote.json`.

- [x] **Step 1: Add a failing remote-runner contract test**

Add:

```python
def test_remote_runner_uses_required_host_auth_and_no_controlmaster():
    text = RUNNER.read_text(encoding="utf-8")
    assert "sitian@10.232.195.203" in text
    assert "FILE:/Users/bytedance/krb5cc_sitian" in text
    assert "ControlMaster=no" in text
    assert "ControlPath=none" in text
    assert "CUDA_VISIBLE_DEVICES" in text
    assert "nvidia-smi" in text
    assert "verify_generic_speculative_tp4_gate.py" in text
    assert "stale" in text.lower()
```

- [ ] **Step 2: Run the runner contract test and verify RED**

Run:

```bash
PYTHONPATH=$PWD python3 -m pytest \
  tools/test_generic_speculative_tp4_gate.py::test_remote_runner_uses_required_host_auth_and_no_controlmaster \
  -q
```

Expected: failure because the runner does not exist.

- [x] **Step 3: Implement common serial SSH options**

Use:

```bash
export KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian
REMOTE=sitian@10.232.195.203
SSH=(
  ssh
  -o ControlMaster=no
  -o ControlPath=none
  -o GSSAPIAuthentication=yes
  "$REMOTE"
)
SCP=(
  scp
  -o ControlMaster=no
  -o ControlPath=none
  -o GSSAPIAuthentication=yes
)
```

Do not launch SSH commands in parallel.

- [x] **Step 4: Implement real four-GPU preflight**

The remote preflight must:

```bash
nvidia-smi \
  --query-gpu=index,memory.free,memory.total,utilization.gpu \
  --format=csv,noheader,nounits
```

Parse four distinct GPUs meeting an explicit free-memory floor derived from a
single TP4 Qwen3-0.6B Engine cell. Persist the raw inventory and selected
indices. Reject fewer than four eligible GPUs.

Probe candidate ports with remote Python socket binds and reject collisions.

- [x] **Step 5: Implement detached campaign lifecycle**

Use a unique opaque run ID, remote status file, PID file, and log. Poll
serially. If the process appears stale, reread the terminal status and result
files for a bounded interval before classifying failure, matching the proven
TP1 runner behavior.

The remote command runs:

```bash
PYTHONPATH="$REMOTE_CHECKOUT" \
CUDA_VISIBLE_DEVICES="$GPU_CSV" \
python3 tools/generic_speculative_tp4_gate.py \
  --model "$MODEL_PATH" \
  --gpu-indices "$GPU_CSV" \
  --dist-port-base "$DIST_PORT_BASE" \
  --master-port-base "$MASTER_PORT_BASE" \
  --output-dir "$REMOTE_RUN_DIR"
```

- [x] **Step 6: Implement remote and local verification**

Run the verifier remotely, download the complete run directory, then run the
verifier locally against the current source tree. Do not report success unless
both classifications are `PASS`.

- [x] **Step 7: Run runner text-contract tests and shell syntax**

Run:

```bash
PYTHONPATH=$PWD python3 -m pytest \
  tools/test_generic_speculative_tp4_gate.py \
  -k 'remote_runner' \
  -q
bash -n tools/run_generic_speculative_tp4_gate_remote.sh
```

Expected: tests pass and `bash -n` exits zero.

---

### Task 6: Run Local Regression and Scoped Validation

**Files:**
- Test only; no planned edits.

**Interfaces:**
- Consumes all code from Tasks 1-5.
- Produces local evidence that dependency-light behavior and source hygiene
  pass before using remote GPUs.

- [x] **Step 1: Run ModelRunner callback tests in isolation**

Run:

```bash
PYTHONPATH=$PWD python3 -m pytest \
  tools/test_model_runner_spec_verify.py \
  -k 'run_spec_first_target_batch or run_spec_verify_batch' \
  -q
```

Expected: all selected tests pass.

- [x] **Step 2: Run the complete TP4 gate test module**

Run:

```bash
PYTHONPATH=$PWD python3 -m pytest \
  tools/test_generic_speculative_tp4_gate.py \
  -q
```

Expected: all tests pass.

- [x] **Step 3: Run adjacent profiler and acknowledgement tests**

Run each module in a fresh process:

```bash
PYTHONPATH=$PWD python3 -m pytest \
  tools/test_decode_internal_profiler.py \
  -q
PYTHONPATH=$PWD python3 -m pytest \
  tools/test_model_runner_command_ack.py \
  -q
PYTHONPATH=$PWD python3 -m pytest \
  tools/test_engine_speculative_execution.py \
  -q
```

Expected: all tests pass.

- [x] **Step 4: Compile the changed Python files**

Run:

```bash
python3 -m py_compile \
  tinyvllm/engine/model_runner.py \
  tools/generic_speculative_tp4_gate.py \
  tools/generic_speculative_tp4_worker.py \
  tools/verify_generic_speculative_tp4_gate.py \
  tools/test_generic_speculative_tp4_gate.py
```

Expected: exit zero.

- [x] **Step 5: Run scoped whitespace validation**

Run:

```bash
git diff --check -- \
  tinyvllm/engine/model_runner.py \
  tools/test_model_runner_spec_verify.py \
  tools/generic_speculative_tp4_gate.py \
  tools/generic_speculative_tp4_worker.py \
  tools/verify_generic_speculative_tp4_gate.py \
  tools/run_generic_speculative_tp4_gate_remote.sh \
  tools/test_generic_speculative_tp4_gate.py \
  docs/superpowers/specs/2026-08-13-generic-speculative-runtime-tp4-correctness-collective-authority-design.md \
  docs/superpowers/plans/2026-08-13-generic-speculative-runtime-tp4-correctness-collective-authority.md
```

Expected: exit zero. Do not fix unrelated repository-global whitespace.

---

### Task 7: Run the Real TP4 Authority Campaign

**Files:**
- Create under: `artifacts/generic_speculative_tp4/$RUN_ID/`

**Interfaces:**
- Consumes the validated runner.
- Produces the real authority and both verifier results.

- [x] **Step 1: Verify Kerberos and SSH reachability serially**

Run:

```bash
KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian klist
KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian \
ssh \
  -o ControlMaster=no \
  -o ControlPath=none \
  -o GSSAPIAuthentication=yes \
  sitian@10.232.195.203 \
  'hostname && pwd'
```

Expected: valid ticket and successful remote command. If SSH reports
`Connection closed by UNKNOWN port 65535`, retry only after rechecking
Kerberos and the proxy route; do not reinterpret it as a code failure.

- [x] **Step 2: Launch the remote runner**

Run:

```bash
bash tools/run_generic_speculative_tp4_gate_remote.sh
```

Expected: the runner prints the selected four GPU indices, opaque run ID,
remote PID, and artifact destination.

- [x] **Step 3: Poll the existing runner process**

Reuse the same process/session until terminal state. Do not launch duplicate
campaigns. Expected terminal state: campaign complete, remote verifier PASS,
download complete, local verifier PASS.

- [x] **Step 4: Inspect the final authority**

The remote runner writes the downloaded authority path to
`artifacts/generic_speculative_tp4/last_completed_run_path.txt`. Validate with
a dependency-light command:

```bash
RUN_DIR="$(
  cat artifacts/generic_speculative_tp4/last_completed_run_path.txt
)"
python3 tools/verify_generic_speculative_tp4_gate.py \
  "$RUN_DIR" \
  --source-root "$PWD"
```

Expected:

```json
{"classification":"PASS","failures":[]}
```

- [x] **Step 5: Record the exact evidence boundary**

Extract and retain:

- result SHA-256;
- source tree SHA-256;
- model manifest SHA-256;
- selected GPU indices;
- exact baseline/candidate outputs for batch 1 and 4;
- candidate proposal, accepted-token, first-target, and verification counts;
- per-rank speculative callback counts;
- per-rank collective counts and operation sequences;
- residency phase order and rank rows;
- KV movement totals;
- cleanup receipt; and
- all explicit limitations.

Do not infer performance direction from timing fields.

---

### Task 8: Update Audit and Handoff

**Files:**
- Modify: `docs/superpowers/audits/2026-08-12-phase1-objective-coverage.md`
- Modify: `AGENT_HANDOFF_STATE.md`

**Interfaces:**
- Consumes the real artifact from Task 7.
- Produces durable project state and the next ordered action.

- [x] **Step 1: Add the TP4 authority to the Phase 1 audit**

Record a bullet named
`Generic host n-gram TP4 correctness/collective authority`. Under it, write
the exact artifact path from `last_completed_run_path.txt`, the exact
`sha256sum "$RUN_DIR/result.json"` value, and classification
`PASS / NOT_PROMOTABLE`. State that it proves exact 4K batch 1/4 baseline
parity, four-rank first-target/spec-verify callbacks, four-rank collective
identity, successful residency acknowledgements, KV summary inventory, and
clean shutdown. State that it does not prove TP4 performance, 16K/32K
performance direction, second-model portability, model-runner proposal TP4,
learned-drafter/MTP plus offload, KV8/KV4, or Phase 1 completion.

- [x] **Step 2: Update the handoff**

Add:

- exact local and remote commands;
- artifact and verifier paths;
- hashes and rank inventory;
- important callback/collective/residency totals;
- any remote GPU or proxy caveat;
- local test results; and
- the next ordered action: second-model generic runtime correctness authority
  or the explicitly chosen 16K/32K TP4 performance campaign, according to the
  Phase 1 audit ordering after this result.

- [x] **Step 3: Run document and scoped diff checks**

Run:

```bash
git diff --check -- \
  docs/superpowers/audits/2026-08-12-phase1-objective-coverage.md \
  AGENT_HANDOFF_STATE.md
git status --short
```

Expected: scoped diff check passes. Status shows only intended current
worktree changes plus pre-existing unrelated changes.

- [x] **Step 4: Final completion verification**

Run the focused tests and both artifact verifiers one final time. Report:

- what passed;
- what the authority proves;
- what it does not prove;
- the exact artifact path and SHA-256; and
- the next ordered Phase 1 action.
