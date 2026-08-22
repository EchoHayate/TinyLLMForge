# Replay-Aware Decode Metadata Landing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce Qwen3-0.6B batch-1 decode TPOT and host-side tail latency by landing pinned-host decode metadata directly into exact CUDA Graph inputs and eliminating semantically unnecessary blanket clears.

**Architecture:** A dependency-light module builds immutable decode metadata plans and owns bounded pinned-host staging plus exact accounting. `ModelRunner` enables the path only for an exact batch-1 legacy graph and otherwise falls back to the existing implementation. A source-bound OFF/ON benchmark, producer gate, and independent verifier establish exact correctness, performance benefit, and memory cost before promotion.

**Tech Stack:** Python 3, PyTorch CUDA Graphs, TinyLLMForge `ModelRunner`, dependency-light script tests, JSON/JSONL evidence, SSH remote runner.

## Global Constraints

- Work only in `/Users/bytedance/dev/TinyLLMForge`.
- Do not create worktrees or use subagents.
- Preserve all unrelated dirty and untracked files.
- Stage exact paths only; never use broad `git add`, `git reset`, `git clean`, or mass formatting.
- Commit with `git -c core.hooksPath=/dev/null commit`.
- Every commit has exactly one `Co-authored-by: TRAE CLI <noreply@bytedance.com>` trailer.
- Push only to `origin/feat/kv-sparse-attention`.
- The feature flag is `replay_aware_decode_metadata` and defaults to `False`.
- Stage 1 covers Qwen3-0.6B, batch size one, and ordinary legacy CUDA Graph decode only.
- Unsupported paths fail closed to the existing behavior.
- Generated tokens and decoded text must match exactly.
- Do not weaken logit correctness thresholds: `max_abs <= 0.25`, `mean_abs <= 0.05`, and argmax equality.
- Report benefit and cost together.
- Every remote run tag is immutable.
- All remote task output stays under `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818`.
- Never write remote task data under `/`, `/tmp`, `/private/tmp`, or `/data00/home/sitian/tllm/TinyLLMForge`.
- Do not refresh Kerberos automatically.
- Do not terminate or interfere with unrelated GPU processes.
- GPU admission requires memory used `<=1024 MiB`, utilization `<=5%`, and no compute process.
- Qwen3-0.6B evidence cannot support Qwen3-8B claims.
- Do not launch Qwen3-8B unless the Qwen3-0.6B Stage-1 gate is GO.

---

## File structure

- Create `tinyvllm/engine/decode_metadata_landing.py`: immutable host plan, pinned staging arena, eligibility/result records, and accounting.
- Create `tools/test_decode_metadata_landing.py`: dependency-light unit tests for planning, staging, landing, fallback, and counters.
- Modify `tinyvllm/config.py`: add and validate the default-disabled feature flag.
- Modify `tinyvllm/engine/model_runner.py`: construct the arena, route eligible decode steps through direct landing, skip duplicate copy/zero work, and expose a summary.
- Modify `tools/test_model_runner_spec_verify.py`: integration tests around `prepare_decode`, graph selection, replay, fallback, and disabled behavior.
- Create `tools/profile_replay_aware_decode_metadata.py`: source-bound Qwen3-0.6B OFF/ON worker.
- Create `tools/test_profile_replay_aware_decode_metadata.py`: worker contracts and deterministic summary tests.
- Create `tools/replay_aware_decode_metadata_gate.py`: producer-side artifact validation and GO/NO-GO classification.
- Create `tools/replay_aware_decode_metadata_verify.py`: independently reconstruct the paired comparison and classification.
- Create `tools/test_replay_aware_decode_metadata_gate.py`: gate and tamper tests.
- Create `tools/test_replay_aware_decode_metadata_verify.py`: independent-verifier and disagreement tests.
- Create `tools/run_replay_aware_decode_metadata_remote.py`: safe remote admission, source upload, execution, polling, and artifact retrieval.
- Create `tools/test_run_replay_aware_decode_metadata_remote.py`: remote path, GPU admission, Kerberos TTL, and immutable-tag tests.
- Modify `AGENT_HANDOFF_STATE.md`: terminal result, evidence hashes, benefit, cost, and promotion boundary.
- Modify `docs/superpowers/audits/2026-08-16-phase1-completion-audit.md`: append the new optimization audit and prompt-to-artifact checklist.

### Task 1: Pure decode metadata plan and staging arena

**Files:**

- Create: `tinyvllm/engine/decode_metadata_landing.py`
- Create: `tools/test_decode_metadata_landing.py`

**Interfaces:**

- Produces:
  - `DecodeMetadataPlan`
  - `DecodeMetadataLandingStats`
  - `build_decode_metadata_plan(seqs, block_size) -> DecodeMetadataPlan`
  - `ReplayAwareDecodeMetadataArena(torch_module)`
  - `arena.land(plan, graph_vars, graph_batch_size) -> DecodeMetadataLandingResult`
  - `arena.summary() -> dict`

- [ ] **Step 1: Write failing plan-construction tests**

```python
def test_build_decode_metadata_plan_preserves_readable_rows():
    seq = FakeSequence(
        last_token=17,
        token_count=513,
        block_table=[4, 8, 15],
        block_size=256,
    )
    plan = build_decode_metadata_plan([seq], 256)
    assert plan.input_ids == (17,)
    assert plan.positions == (512,)
    assert plan.slot_mapping == (15 * 256,)
    assert plan.context_lens == (513,)
    assert plan.block_table_rows == ((4, 8, 15),)
    assert plan.active_batch_size == 1
    assert plan.readable_page_table_width == 3


def test_build_decode_metadata_plan_pads_rows_deterministically():
    first = FakeSequence(3, 257, [1, 2], 256)
    second = FakeSequence(5, 1, [7], 256)
    plan = build_decode_metadata_plan([first, second], 256)
    assert plan.block_table_rows == ((1, 2), (7, -1))
```

- [ ] **Step 2: Run the focused tests and confirm RED**

Run:

```bash
python3 tools/test_decode_metadata_landing.py
```

Expected: import failure because `decode_metadata_landing.py` does not exist.

- [ ] **Step 3: Implement immutable planning records**

```python
@dataclass(frozen=True)
class DecodeMetadataPlan:
    input_ids: tuple[int, ...]
    positions: tuple[int, ...]
    slot_mapping: tuple[int, ...]
    context_lens: tuple[int, ...]
    block_table_rows: tuple[tuple[int, ...], ...]
    active_batch_size: int
    readable_page_table_width: int


def build_decode_metadata_plan(seqs, block_size):
    if not seqs:
        raise ValueError("decode metadata requires at least one sequence")
    width = max(len(seq.block_table) for seq in seqs)
    rows = tuple(
        tuple(int(value) for value in seq.block_table)
        + (-1,) * (width - len(seq.block_table))
        for seq in seqs
    )
    return DecodeMetadataPlan(
        input_ids=tuple(int(seq.last_token) for seq in seqs),
        positions=tuple(len(seq) - 1 for seq in seqs),
        slot_mapping=tuple(
            int(seq.block_table[-1]) * int(block_size)
            + int(seq.last_block_num_tokens) - 1
            for seq in seqs
        ),
        context_lens=tuple(len(seq) for seq in seqs),
        block_table_rows=rows,
        active_batch_size=len(seqs),
        readable_page_table_width=width,
    )
```

- [ ] **Step 4: Add failing arena tests**

Use a fake torch module and fake pinned/device tensors to assert:

```python
result = arena.land(plan, graph_vars, graph_batch_size=1)
assert result.optimized is True
assert graph_vars["input_ids"].writes == [((slice(0, 1),), [17])]
assert graph_vars["block_tables"].writes == [
    ((slice(0, 1), slice(0, 3)), [[4, 8, 15]])
]
assert graph_vars["outputs"].zero_calls == 0
assert graph_vars["block_tables"].zero_calls == 0
assert torch_module.tensor_calls == 0
assert arena.summary()["avoided_temporary_cuda_tensors"] == 5
```

Also assert that `graph_batch_size=2`, insufficient destination capacity, and
non-batch-1 plans return explicit fallback reasons without modifying graph
buffers.

- [ ] **Step 5: Implement direct pinned staging and exact counters**

Implement:

```python
@dataclass(frozen=True)
class DecodeMetadataLandingResult:
    optimized: bool
    fallback_reason: str | None
    input_ids: object | None = None
    positions: object | None = None
    slot_mapping: object | None = None
    context_lens: object | None = None
    block_tables: object | None = None


class ReplayAwareDecodeMetadataArena:
    FIELD_DTYPES = {
        "input_ids": "int64",
        "positions": "int64",
        "slot_mapping": "int32",
        "context_lens": "int32",
        "block_tables": "int32",
    }

    def __init__(self, torch_module):
        self._torch = torch_module
        self._host = {}
        self._capacity = {}
        self._stats = DecodeMetadataLandingStats()

    def _stage_flat(self, name, values, dtype):
        required = len(values)
        buffer = self._host.get(name)
        if buffer is None or buffer.numel() < required:
            capacity = max(required, max(64, self._capacity.get(name, 0) * 2))
            buffer = self._torch.empty(
                capacity,
                dtype=dtype,
                device="cpu",
                pin_memory=True,
            )
            self._host[name] = buffer
            self._capacity[name] = capacity
            self._stats.record_growth(name, buffer)
        for index, value in enumerate(values):
            buffer[index] = value
        self._stats.record_stage(required * buffer.element_size())
        return buffer[:required]
```

`land()` must validate all shapes before the first destination copy, flatten
the block-table rows, copy with `non_blocking=True`, return active destination
views, increment one optimized step and five avoided temporary CUDA tensors,
and account only bytes that the old blanket clear would have written.

- [ ] **Step 6: Run unit tests and syntax checks**

Run:

```bash
python3 tools/test_decode_metadata_landing.py
python3 -m py_compile tinyvllm/engine/decode_metadata_landing.py tools/test_decode_metadata_landing.py
```

Expected: both commands exit zero and print the test success marker.

- [ ] **Step 7: Commit Task 1**

```bash
git add -- tinyvllm/engine/decode_metadata_landing.py tools/test_decode_metadata_landing.py
git -c core.hooksPath=/dev/null commit \
  -m "feat(perf): add decode metadata landing arena" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

### Task 2: Default-disabled ModelRunner integration

**Files:**

- Modify: `tinyvllm/config.py`
- Modify: `tinyvllm/engine/model_runner.py`
- Modify: `tools/test_model_runner_spec_verify.py`

**Interfaces:**

- Consumes `build_decode_metadata_plan()` and
  `ReplayAwareDecodeMetadataArena.land()`.
- Produces:
  - `Config.replay_aware_decode_metadata: bool`
  - `ModelRunner.replay_aware_decode_metadata_summary() -> dict`
  - `_prepare_replay_aware_decode(seqs) -> tuple | None`
  - `_replay_aware_decode_prelanded: bool`

- [ ] **Step 1: Add failing config and disabled-path tests**

Add assertions:

```python
assert Config.__dataclass_fields__["replay_aware_decode_metadata"].default is False

runner.config.replay_aware_decode_metadata = False
runner.prepare_decode = lambda seqs: calls.append(("legacy", seqs)) or inputs
runner._run_model_step(seqs, False)
assert calls[0][0] == "legacy"
assert runner.replay_aware_decode_metadata_summary()["optimized_steps"] == 0
```

Add a validation test that a non-boolean flag raises
`ValueError("replay_aware_decode_metadata must be a bool")`.

- [ ] **Step 2: Run the focused integration tests and confirm RED**

Run:

```bash
python3 tools/test_model_runner_spec_verify.py
```

Expected: failure on the missing config field or summary method.

- [ ] **Step 3: Add configuration and runner-owned arena**

Add to `Config`:

```python
replay_aware_decode_metadata: bool = False
```

Add to `Config.__post_init__()`:

```python
if not isinstance(self.replay_aware_decode_metadata, bool):
    raise ValueError(
        "replay_aware_decode_metadata must be a bool"
    )
```

Construct in `ModelRunner.__init__()`:

```python
self.replay_aware_decode_metadata_arena = (
    ReplayAwareDecodeMetadataArena(torch)
)
self._replay_aware_decode_prelanded = False
```

- [ ] **Step 4: Add failing eligible-path and fallback tests**

Cover:

```python
runner.config.replay_aware_decode_metadata = True
runner.graph_bs = [1]
runner.graphs = {1: FakeGraph()}
runner.graph_vars = make_graph_vars()
prepared = runner._prepare_replay_aware_decode([sequence])
assert prepared is not None
assert runner._replay_aware_decode_prelanded is True

logits = runner.run_model(prepared[0], prepared[1], False)
assert runner.graphs[1].replay_calls == 1
assert all(buffer.zero_calls == 0 for buffer in runner.graph_vars.values())
```

Then parameterize fail-closed cases for batch size two, eager mode, absent
graph state, graph size mismatch, KV offload, Quest, compact attention,
KV quantization, CPU offload, and cartridge mode.

- [ ] **Step 5: Implement eligible routing and zero-free replay**

In `_run_model_step()`:

```python
self._replay_aware_decode_prelanded = False
prepared = (
    self._prepare_replay_aware_decode(seqs)
    if not is_prefill and batch_kind != "mixed"
    else None
)
if prepared is None:
    input_ids, positions = (
        self.prepare_prefill(seqs)
        if is_prefill
        else self.prepare_decode(seqs)
    )
else:
    input_ids, positions = prepared
```

`_prepare_replay_aware_decode()` must:

1. validate the complete eligibility list before planning;
2. build the host plan;
3. land into `graph_vars`;
4. call `set_context()` with returned active destination views;
5. set `_replay_aware_decode_prelanded=True` only after success;
6. record and return `None` for any explicit fallback.

In the legacy graph replay branch:

```python
if self._replay_aware_decode_prelanded:
    if bs != 1 or selected_graph_bs != 1:
        raise RuntimeError("prelanded decode graph identity drift")
else:
    for key, value in graph_vars.items():
        if key != "outputs":
            value.zero_()
    graph_vars["input_ids"][:bs] = input_ids
    graph_vars["positions"][:bs] = positions
    graph_vars["slot_mapping"][:bs] = context.slot_mapping
    graph_vars["context_lens"][:bs] = context.context_lens
    graph_vars["block_tables"][
        :bs, :context.block_tables.size(1)
    ] = context.block_tables
graph.replay()
```

Do not change the disabled branch.

- [ ] **Step 6: Run focused and neighboring regressions**

Run:

```bash
python3 tools/test_decode_metadata_landing.py
python3 tools/test_model_runner_spec_verify.py
python3 tools/test_multi_sequence_cuda_graph_gate.py
python3 tools/test_chunked_prefill.py
python3 tools/test_profile_prefix_cache.py
python3 -m py_compile tinyvllm/config.py tinyvllm/engine/model_runner.py
```

Expected: all available tests pass. If `test_chunked_prefill.py` is blocked by
the local lack of Torch, record the exact environment failure and rerun it in
the remote source-bound preflight; do not count the local result as pass or
semantic failure.

- [ ] **Step 7: Commit Task 2**

```bash
git add -- tinyvllm/config.py tinyvllm/engine/model_runner.py tools/test_model_runner_spec_verify.py
git -c core.hooksPath=/dev/null commit \
  -m "feat(perf): land decode metadata into graph inputs" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

### Task 3: Source-bound benchmark worker

**Files:**

- Create: `tools/profile_replay_aware_decode_metadata.py`
- Create: `tools/test_profile_replay_aware_decode_metadata.py`

**Interfaces:**

- Produces:
  - schema `replay-aware-decode-metadata.case.v1`
  - one `case_rows.jsonl`
  - one `workload_manifest.json`
  - one `source_manifest.json`
  - one `summary.json`

- [ ] **Step 1: Write failing pure worker-contract tests**

Test deterministic helpers:

```python
assert context_cases() == (
    ("short", 256, 128),
    ("medium", 2048, 128),
    ("long", 8192, 128),
)
assert policy_order(0) == ("off", "on")
assert policy_order(1) == ("on", "off")
assert nearest_rank_percentile([1, 2, 3, 4, 5], 0.95) == 5
```

Construct fake OFF/ON rows and assert exact token mismatch, missing optimized
steps, or missing cost fields makes `summarize_rows()` fail.

- [ ] **Step 2: Run and confirm RED**

Run:

```bash
python3 tools/test_profile_replay_aware_decode_metadata.py
```

Expected: import failure because the worker does not exist.

- [ ] **Step 3: Implement the worker**

The worker CLI must require:

```text
--model
--out-dir
--source-commit
--run-tag
```

It must support:

```text
--repetitions 5
--warmup-repetitions 2
--prompt-lengths 256,2048,8192
--generated-tokens 128
--gpu-memory-utilization 0.5
```

For each repetition and context bucket:

1. alternate OFF/ON construction order;
2. create a fresh `LLM` with only
   `replay_aware_decode_metadata` differing;
3. use deterministic token IDs and greedy decoding;
4. reset peak CUDA memory before the measured request;
5. capture exact output IDs and text;
6. derive TTFT, E2E, per-step TPOT, host/CUDA decode timing, throughput, memory,
   and landing summary;
7. fsync each JSONL append.

Every row must include:

```python
{
    "schema_version": "replay-aware-decode-metadata.case.v1",
    "run_tag": run_tag,
    "source_commit": source_commit,
    "policy": "off" or "on",
    "repetition": repetition,
    "context_bucket": bucket,
    "prompt_tokens": prompt_tokens,
    "generated_tokens": 128,
    "output_token_ids": output_ids,
    "output_text_sha256": sha256_text(output_text),
    "ttft_ns": ttft_ns,
    "e2e_ns": e2e_ns,
    "tpot_samples_ns": tpot_samples_ns,
    "decode_host_ns": decode_host_ns,
    "decode_cuda_ns": decode_cuda_ns,
    "output_tokens_per_second": rate,
    "cuda_peak_allocated_bytes": allocated,
    "cuda_peak_reserved_bytes": reserved,
    "landing_summary": landing_summary,
}
```

- [ ] **Step 4: Run worker tests and syntax checks**

Run:

```bash
python3 tools/test_profile_replay_aware_decode_metadata.py
python3 -m py_compile tools/profile_replay_aware_decode_metadata.py
```

Expected: PASS and zero exit status.

- [ ] **Step 5: Commit Task 3**

```bash
git add -- tools/profile_replay_aware_decode_metadata.py tools/test_profile_replay_aware_decode_metadata.py
git -c core.hooksPath=/dev/null commit \
  -m "test(perf): add replay-aware metadata benchmark worker" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

### Task 4: Producer gate and independent verifier

**Files:**

- Create: `tools/replay_aware_decode_metadata_gate.py`
- Create: `tools/replay_aware_decode_metadata_verify.py`
- Create: `tools/test_replay_aware_decode_metadata_gate.py`
- Create: `tools/test_replay_aware_decode_metadata_verify.py`

**Interfaces:**

- Producer classification:
  - `GO_REPLAY_AWARE_METADATA`
  - `NO_GO_CORRECTNESS`
  - `NO_GO_OPTIMIZED_PATH_INCOMPLETE`
  - `NO_GO_TPOT_MEDIAN`
  - `NO_GO_TPOT_P95`
  - `NO_GO_PROTECTED_REGRESSION`
  - `NO_GO_EVIDENCE_INCOMPLETE`
- Independent verifier emits
  `replay-aware-decode-metadata.independent-verification.v1`.

- [ ] **Step 1: Write failing gate tests**

Create complete synthetic five-repetition fixtures and prove:

```python
assert classify(go_fixture)["classification"] == "GO_REPLAY_AWARE_METADATA"
assert classify(token_mismatch)["classification"] == "NO_GO_CORRECTNESS"
assert classify(no_optimized_steps)["classification"] == "NO_GO_OPTIMIZED_PATH_INCOMPLETE"
assert classify(flat_tpot)["classification"] == "NO_GO_TPOT_MEDIAN"
assert classify(p95_regression)["classification"] == "NO_GO_TPOT_P95"
assert classify(ttft_regression)["classification"] == "NO_GO_PROTECTED_REGRESSION"
```

Tamper with one row, one source hash, one duplicate case identity, and one
non-finite metric; each must be rejected.

- [ ] **Step 2: Implement producer validation and classification**

The gate must:

1. require exactly 30 measured rows: 3 buckets × 5 repetitions × 2 policies;
2. pair rows by `(context_bucket, repetition)`;
3. require exact output token and text-hash equality;
4. require ON-path optimized steps to equal measured decode steps;
5. reconstruct medians and nearest-rank P95/P99 from raw samples;
6. apply every threshold from the design in fixed order;
7. report per-bucket and aggregate benefit plus pinned/CUDA memory cost;
8. hash all primary evidence into `manifest.sha256`;
9. write `comparison.json` and `gate.json`.

- [ ] **Step 3: Write failing independent-verifier tests**

The verifier must not import the producer gate. Test that it:

- reconstructs the GO fixture independently;
- rejects producer comparison drift;
- rejects producer classification drift;
- rejects an omitted primary artifact;
- rejects a manifest whose file list or digest is stale.

- [ ] **Step 4: Implement the independent verifier**

Use separate percentile, pairing, threshold, and manifest code. Emit:

```python
{
    "schema_version":
        "replay-aware-decode-metadata.independent-verification.v1",
    "status": "PASS",
    "reconstructed_classification": classification,
    "comparison_sha256": sha256_json(reconstructed_comparison),
    "manifest_sha256": sha256_file(manifest_path),
}
```

- [ ] **Step 5: Run gate and verifier tests**

Run:

```bash
python3 tools/test_replay_aware_decode_metadata_gate.py
python3 tools/test_replay_aware_decode_metadata_verify.py
python3 -m py_compile \
  tools/replay_aware_decode_metadata_gate.py \
  tools/replay_aware_decode_metadata_verify.py
```

Expected: all commands pass.

- [ ] **Step 6: Commit Task 4**

```bash
git add -- \
  tools/replay_aware_decode_metadata_gate.py \
  tools/replay_aware_decode_metadata_verify.py \
  tools/test_replay_aware_decode_metadata_gate.py \
  tools/test_replay_aware_decode_metadata_verify.py
git -c core.hooksPath=/dev/null commit \
  -m "test(perf): gate replay-aware decode metadata" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

### Task 5: Safe remote controller

**Files:**

- Create: `tools/run_replay_aware_decode_metadata_remote.py`
- Create: `tools/test_run_replay_aware_decode_metadata_remote.py`

**Interfaces:**

- Reuses constants and pure admission helpers from
  `tools/run_staged_inference_benchmark_remote.py` where import-safe.
- Produces controller, primary, and independent-verification bundles under the
  approved remote root and local `artifacts/replay_aware_decode_metadata/`.

- [ ] **Step 1: Write failing controller safety tests**

Assert:

```python
paths = remote_paths("20260822-qwen3-06b-replay-meta-r1")
assert all(path.startswith(APPROVED_ROOT + "/") for path in paths.values())
assert strict_clean_gpus(clean_rows)[0]["index"] == 3
```

Also assert rejection of:

- mutable/reused run tags;
- paths under `/tmp` or `/`;
- Kerberos lifetime below 5400 seconds;
- GPU memory above 1024 MiB;
- utilization above 5%;
- any compute process;
- source commit not equal to the requested commit;
- missing remote Python/model path;
- incomplete artifact download.

- [ ] **Step 2: Implement the controller**

The controller must:

1. validate local branch and source commit;
2. check Kerberos TTL without refreshing credentials;
3. query remote filesystem capacity and GPU inventory;
4. select one strict-clean GPU;
5. create an immutable source archive from tracked files plus the exact
   committed tree;
6. upload only to approved staging;
7. run dependency-light tests and `test_chunked_prefill.py` before the worker;
8. launch the worker with a new tag and selected `CUDA_VISIBLE_DEVICES`;
9. poll without holding more than one reusable SSH process;
10. run producer and independent verification remotely;
11. download with chunked transfer and verify every digest locally;
12. leave all failed and partial evidence in place.

- [ ] **Step 3: Run controller tests**

Run:

```bash
python3 tools/test_run_replay_aware_decode_metadata_remote.py
python3 -m py_compile tools/run_replay_aware_decode_metadata_remote.py
```

Expected: PASS.

- [ ] **Step 4: Commit Task 5**

```bash
git add -- \
  tools/run_replay_aware_decode_metadata_remote.py \
  tools/test_run_replay_aware_decode_metadata_remote.py
git -c core.hooksPath=/dev/null commit \
  -m "test(perf): add replay-aware metadata remote gate" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

### Task 6: Execute Stage 1 and classify

**Files:**

- Create: `artifacts/replay_aware_decode_metadata/<immutable-run-tag>/`

**Interfaces:**

- Consumes the committed source tree and remote controller.
- Produces a terminal, independently verified Stage-1 bundle.

- [ ] **Step 1: Run the full local regression set**

Run:

```bash
python3 tools/test_decode_metadata_landing.py
python3 tools/test_model_runner_spec_verify.py
python3 tools/test_multi_sequence_cuda_graph_gate.py
python3 tools/test_profile_replay_aware_decode_metadata.py
python3 tools/test_replay_aware_decode_metadata_gate.py
python3 tools/test_replay_aware_decode_metadata_verify.py
python3 tools/test_run_replay_aware_decode_metadata_remote.py
python3 tools/test_source_audit.py
git diff --check
```

Expected: all available tests pass and `git diff --check` is empty.

- [ ] **Step 2: Launch one immutable Stage-1 run**

Use a fresh tag:

```bash
python3 tools/run_replay_aware_decode_metadata_remote.py \
  --run-tag 20260822-qwen3-06b-replay-meta-r1 \
  --model-tier qwen3-0.6b \
  --source-commit "$(git rev-parse HEAD)"
```

If the strict-clean GPU condition is unavailable, keep monitoring locally and
launch immediately when one GPU satisfies all admission criteria. Do not
terminate unrelated processes.

- [ ] **Step 3: Reconstruct evidence locally**

Run both:

```bash
python3 tools/replay_aware_decode_metadata_gate.py \
  --artifact-dir artifacts/replay_aware_decode_metadata/20260822-qwen3-06b-replay-meta-r1
python3 tools/replay_aware_decode_metadata_verify.py \
  --artifact-dir artifacts/replay_aware_decode_metadata/20260822-qwen3-06b-replay-meta-r1
```

Expected: producer and independent classification and comparison digests agree.

- [ ] **Step 4: Apply the promotion boundary**

If classification is `GO_REPLAY_AWARE_METADATA`, enable the flag by default
only for the proven batch-1 legacy graph scope and run a fresh verification
bundle. If classification is any NO-GO value, keep the flag default-disabled,
do not run Qwen3-8B, and preserve the negative result.

### Task 7: Audit, handoff, final verification, and push

**Files:**

- Modify: `AGENT_HANDOFF_STATE.md`
- Modify: `docs/superpowers/audits/2026-08-16-phase1-completion-audit.md`

**Interfaces:**

- Produces the terminal prompt-to-artifact audit and remote branch state.

- [ ] **Step 1: Append the exact result**

Record:

- source commit and final documentation commit;
- immutable run tag and artifact paths;
- producer and independent classifications;
- summary, comparison, manifest, and verifier SHA256 values;
- per-bucket median/P95/P99 TPOT changes;
- TTFT, E2E, throughput, and CUDA-memory regressions;
- pinned-host bytes, growth count, staged H2D bytes, avoided allocations, and
  avoided zero bytes;
- exact output parity;
- Qwen3-8B eligibility and whether it was run.

- [ ] **Step 2: Build the prompt-to-artifact checklist**

Map every requirement in the design to a concrete file, test output, raw row,
manifest entry, gate field, verifier field, or git remote assertion. Mark any
uncertainty as incomplete and rerun or repair the missing evidence.

- [ ] **Step 3: Run final verification**

Run:

```bash
python3 tools/test_decode_metadata_landing.py
python3 tools/test_model_runner_spec_verify.py
python3 tools/test_multi_sequence_cuda_graph_gate.py
python3 tools/test_profile_replay_aware_decode_metadata.py
python3 tools/test_replay_aware_decode_metadata_gate.py
python3 tools/test_replay_aware_decode_metadata_verify.py
python3 tools/test_run_replay_aware_decode_metadata_remote.py
python3 tools/test_source_audit.py
python3 -m py_compile \
  tinyvllm/engine/decode_metadata_landing.py \
  tinyvllm/engine/model_runner.py \
  tinyvllm/config.py \
  tools/profile_replay_aware_decode_metadata.py \
  tools/replay_aware_decode_metadata_gate.py \
  tools/replay_aware_decode_metadata_verify.py \
  tools/run_replay_aware_decode_metadata_remote.py
git diff --check
```

Then verify:

```bash
git rev-parse HEAD
git ls-remote origin refs/heads/feat/kv-sparse-attention
```

- [ ] **Step 4: Commit and push exact documentation paths**

```bash
git add -- \
  AGENT_HANDOFF_STATE.md \
  docs/superpowers/audits/2026-08-16-phase1-completion-audit.md
git -c core.hooksPath=/dev/null commit \
  -m "docs(perf): record replay-aware metadata evidence" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

- [ ] **Step 5: Completion audit**

Restate the objective as:

1. one new, independently motivated decode optimization;
2. default-disabled safe implementation;
3. exact correctness evidence;
4. Qwen3-0.6B paired performance evidence;
5. benefit and cost metrics;
6. independent verification;
7. immutable artifacts;
8. audit, commits, and remote push;
9. Qwen3-8B only after Stage-1 GO.

Do not claim completion until every item maps to inspected evidence and local
HEAD equals `origin/feat/kv-sparse-attention`.
