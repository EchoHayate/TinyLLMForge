# Lease-Sealed Persistent Decode MegaKernel Ceiling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a source-bound A100 qualification that rejects or
authorizes a future lease-scoped persistent decode segment before any
production megakernel is implemented.

**Architecture:** Reuse the existing Exact Greedy K8 entrypoint for
uninstrumented timing and exact-output authority. Add a dependency-light
Nsight SQLite parser, generic kernel-role classifier, contiguous-segment
reconstructor, optimistic zero-cost ceiling classifier, independent verifier,
and mounted-only remote controller. Raw profiler reports remain remote; the
local repository receives only compact rows, manifests, and verification
receipts.

**Tech Stack:** Python 3, sqlite3, dataclasses, pytest, PyTorch NVTX, Nsight
Systems 2024.7.1, JSON/JSONL, SHA-256 manifests, SSH, Qwen3-0.6B BF16, one
strict-clean NVIDIA A100 80GB PCIe.

## Global Constraints

- Work only in `/Users/bytedance/dev/TinyLLMForge`; its Desktop path is a
  symlink to the same checkout.
- Do not create a worktree or use a subagent.
- Use TDD for every production change: focused RED, minimal implementation,
  focused GREEN.
- Stage exact paths only. Never use `git add -A`, `git reset`, `git clean`,
  mass formatting, or unrelated cleanup.
- Commit with `git -c core.hooksPath=/dev/null commit`.
- Every commit has exactly one
  `Co-authored-by: TRAE CLI <noreply@bytedance.com>` trailer.
- Push only `origin/feat/kv-sparse-attention`.
- Use `KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian` for every SSH command.
- Never run `kinit` or `krenew`.
- Use non-persistent SSH sessions.
- Do not terminate, attach to, or reuse external GPU processes.
- Put every remote source, cache, temporary file, profiler report, log,
  manifest, and artifact below
  `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/`.
- Do not write remote task data under `/`, `/tmp`, `/private/tmp`, the
  historical checkout, or a model-cache directory.
- Reuse `/data00/home/sitian/.ms_cache/Qwen/Qwen3-0___6B`; do not download or
  duplicate the model.
- Require one A100 with zero compute processes, zero MiB reported used memory,
  and zero percent utilization immediately before launch.
- Keep profiling disabled by default and add no production runtime flag.
- Do not claim measured DRAM bytes, occupancy, cache hit rate, warp
  efficiency, or instruction mix because Nsight Compute is unavailable.
- Preserve exact output-token IDs, decoded-text digest, target-forward count,
  committed-token count, lease ownership, and fallback semantics.
- Heavy `.nsys-rep` and `.sqlite` files remain remote. A local verifier may
  use a bounded temporary download that is removed after verification.
- Do not implement a persistent kernel unless the frozen classifier returns
  `GO_PERSISTENT_DECODE_CEILING`.
- Report benefit and cost together.

---

### Task 1: Parse generic Nsight decode transactions

**Files:**

- Create: `tools/persistent_decode_kernel_trace.py`
- Create: `tools/test_persistent_decode_kernel_trace.py`

**Interfaces:**

- Produces:
  `KernelInterval(start_ns: int, end_ns: int, stream_id: int, name: str)`
- Produces:
  `TraceRange(identity: dict, start_ns: int, end_ns: int, global_tid: int)`
- Produces:
  `parse_trace_label(text: str) -> tuple[str, dict] | None`
- Produces:
  `read_decode_trace(path: Path) -> dict`
- Produces:
  `assign_kernels_to_ranges(ranges, kernels) -> list[dict]`

- [ ] **Step 1: Write failing SQLite parser tests**

Create a minimal SQLite fixture with:

```python
def test_read_decode_trace_maps_kernels_to_non_overlapping_ranges(tmp_path):
    trace = make_trace_sqlite(
        tmp_path,
        ranges=[
            (
                "persistent_decode_trace/"
                "attempt=a/workload=exact/repetition=0/"
                "context=256/burst=0/logical_tokens=8",
                100,
                300,
            ),
        ],
        kernels=[
            (120, 160, 7, "void rms_norm_kernel"),
            (170, 260, 7, "ampere_sgemm_128x64"),
        ],
    )
    parsed = read_decode_trace(trace)
    assert parsed["classification"] == "COMPLETE"
    assert [row["name"] for row in parsed["kernel_rows"]] == [
        "void rms_norm_kernel",
        "ampere_sgemm_128x64",
    ]
    assert {
        row["logical_tokens"]
        for row in parsed["kernel_rows"]
    } == {8}
```

Also cover:

```python
def test_trace_rejects_overlapping_transaction_ranges(tmp_path):
    trace = make_overlapping_trace_sqlite(tmp_path)
    with pytest.raises(ValueError, match="ranges overlap"):
        read_decode_trace(trace)


def test_trace_rejects_kernel_crossing_transaction_boundary(tmp_path):
    trace = make_boundary_crossing_trace_sqlite(tmp_path)
    with pytest.raises(ValueError, match="crosses transaction"):
        read_decode_trace(trace)


def test_trace_rejects_duplicate_identity(tmp_path):
    trace = make_duplicate_identity_trace_sqlite(tmp_path)
    with pytest.raises(ValueError, match="duplicate identity"):
        read_decode_trace(trace)


def test_trace_ignores_unrelated_nvtx_and_loader_kernels(tmp_path):
    trace = make_trace_with_unrelated_ranges(tmp_path)
    parsed = read_decode_trace(trace)
    assert {row["name"] for row in parsed["kernel_rows"]} == {
        "rms_norm_kernel",
    }


def test_trace_requires_supported_nsys_schema(tmp_path):
    trace = make_trace_missing_kernel_table(tmp_path)
    with pytest.raises(ValueError, match="missing table"):
        read_decode_trace(trace)
```

- [ ] **Step 2: Run the tests and confirm RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q \
  tools/test_persistent_decode_kernel_trace.py \
  -p no:cacheprovider
```

Expected: collection fails because
`tools.persistent_decode_kernel_trace` does not exist.

- [ ] **Step 3: Implement the minimal parser**

Use only Python standard-library `sqlite3`. Require:

```python
REQUIRED_TABLES = {
    "StringIds": {"id", "value"},
    "NVTX_EVENTS": {"start", "end", "text", "textId", "globalTid"},
    "CUPTI_ACTIVITY_KIND_KERNEL": {
        "start",
        "end",
        "streamId",
        "globalPid",
        "demangledName",
        "shortName",
    },
}
```

Accept labels only under:

```text
persistent_decode_trace/
attempt=<value>/
workload=<value>/
repetition=<non-negative integer>/
context=<positive integer>/
burst=<non-negative integer>/
logical_tokens=<positive integer>
```

Normalize every row to JSON-safe integers and strings. Reject overlapping
transaction ranges, duplicate identities, non-positive intervals, kernels
crossing a selected range, and unsupported schema. Ignore all kernels outside
selected ranges.

- [ ] **Step 4: Run focused GREEN**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q \
  tools/test_persistent_decode_kernel_trace.py \
  -p no:cacheprovider
```

Expected: all Task 1 tests pass.

- [ ] **Step 5: Commit Task 1**

Run:

```bash
git add -- \
  tools/persistent_decode_kernel_trace.py \
  tools/test_persistent_decode_kernel_trace.py
git diff --cached --check
git -c core.hooksPath=/dev/null commit \
  -m "feat(profiler): parse persistent decode traces" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

### Task 2: Classify kernels and reconstruct candidate segments

**Files:**

- Modify: `tools/persistent_decode_kernel_trace.py`
- Modify: `tools/test_persistent_decode_kernel_trace.py`

**Interfaces:**

- Produces:
  `classify_kernel(name: str) -> str`
- Produces:
  `classify_kernel_rows(rows: list[dict]) -> list[dict]`
- Produces:
  `build_candidate_segments(rows: list[dict]) -> list[dict]`
- Produces:
  `summarize_trace_coverage(rows: list[dict]) -> dict`

- [ ] **Step 1: Write failing role-classification tests**

Cover representative generic/library symbols:

```python
@pytest.mark.parametrize(
    ("name", "role"),
    [
        ("ampere_bf16_s16816gemm", "MATMUL"),
        ("flash_fwd_splitkv_kernel", "ATTENTION"),
        ("rms_norm_kernel", "NORMALIZATION"),
        ("silu_and_mul_kernel", "ELEMENTWISE"),
        ("reduce_kernel", "REDUCTION"),
        ("index_put_kernel", "INDEX_OR_STATE_UPDATE"),
        ("argmax_reduce_kernel", "TOKEN_SELECTION"),
        ("vectorized_memcpy", "COPY_OR_FILL"),
        ("cudaGraphLaunch", "RUNTIME_OR_GRAPH"),
        ("unrecognized_vendor_kernel", "UNKNOWN"),
    ],
)
def test_classify_kernel_uses_generic_roles(name, role):
    assert classify_kernel(name) == role
```

Cover segment behavior:

```python
def test_candidate_segment_stops_at_matmul_attention_copy_and_unknown():
    rows = kernel_rows(
        "NORMALIZATION",
        "ELEMENTWISE",
        "MATMUL",
        "INDEX_OR_STATE_UPDATE",
        "ATTENTION",
        "TOKEN_SELECTION",
        "UNKNOWN",
    )
    segments = build_candidate_segments(rows)
    assert [row["kernel_count"] for row in segments] == [2, 1, 1]
```

Also assert:

- duration sum and internal gaps are exact;
- cross-stream rows do not form one segment;
- negative or overlapping kernel intervals fail;
- signatures are stable under absolute timestamp shifts;
- unknown rows count against both launch and duration coverage;
- the generic module contains none of:
  `qwen`, `llama`, `k8`, `octet`, `a100`.

- [ ] **Step 2: Run the new tests and confirm RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q \
  tools/test_persistent_decode_kernel_trace.py \
  -k 'classify or segment or coverage or leakage' \
  -p no:cacheprovider
```

Expected: fail because the role and segment functions are absent.

- [ ] **Step 3: Implement ordered, fail-closed classification**

Define:

```python
KERNEL_ROLES = (
    "MATMUL",
    "ATTENTION",
    "NORMALIZATION",
    "ELEMENTWISE",
    "REDUCTION",
    "INDEX_OR_STATE_UPDATE",
    "TOKEN_SELECTION",
    "COPY_OR_FILL",
    "RUNTIME_OR_GRAPH",
    "UNKNOWN",
)

CANDIDATE_ROLES = frozenset({
    "NORMALIZATION",
    "ELEMENTWISE",
    "REDUCTION",
    "INDEX_OR_STATE_UPDATE",
    "TOKEN_SELECTION",
})
```

Apply specific patterns before broad patterns so `argmax_reduce_kernel`
becomes `TOKEN_SELECTION`, not `REDUCTION`. Treat `UNKNOWN` as excluded.

Build a candidate segment only from adjacent candidate kernels on the same
stream and within the same complete transaction identity. The segment wall
union is:

```python
wall_union_ns = last_end_ns - first_start_ns
internal_gap_sum_ns = wall_union_ns - kernel_duration_sum_ns
```

- [ ] **Step 4: Run the complete Task 1–2 suite**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q \
  tools/test_persistent_decode_kernel_trace.py \
  -p no:cacheprovider
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 2**

Run:

```bash
git add -- \
  tools/persistent_decode_kernel_trace.py \
  tools/test_persistent_decode_kernel_trace.py
git diff --cached --check
git -c core.hooksPath=/dev/null commit \
  -m "feat(profiler): classify persistent decode segments" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

### Task 3: Implement frozen ceiling arithmetic and classification

**Files:**

- Create: `tools/lease_sealed_persistent_decode_ceiling.py`
- Create: `tools/test_lease_sealed_persistent_decode_ceiling.py`

**Interfaces:**

- Produces:
  `validate_timing_rows(rows: list[dict]) -> list[dict]`
- Produces:
  `validate_trace_summary(payload: dict) -> dict`
- Produces:
  `compute_ceiling(timing_rows, trace_summary) -> dict`
- Produces:
  `classify_ceiling(payload: dict) -> str`

- [ ] **Step 1: Write failing threshold tests**

Use synthetic complete evidence and assert:

```python
def test_complete_headroom_returns_go():
    result = compute_ceiling(
        timing_rows=timing_rows(
            median_tpot_ns=2_000_000,
            profiled_median_tpot_ns=2_100_000,
        ),
        trace_summary=trace_summary(
            eligible_zero_cost_ns=140_000,
            candidate_cuda_duration_ns=100_000,
            classified_launch_ratio=0.99,
            classified_duration_ratio=0.995,
        ),
    )
    assert result["classification"] == (
        "GO_PERSISTENT_DECODE_CEILING"
    )
```

Add one independent test for every terminal boundary:

```text
aggregate optimistic median < 5%
one context optimistic median < 3%
candidate CUDA-duration share < 4%
no stable cross-context signature
launch classification < 98%
duration classification < 99%
median profiler perturbation > 10%
P95 profiler perturbation > 15%
token or text mismatch
source/runtime/workload identity mismatch
fallback/failure/rollback/quarantine nonzero
missing row or trace
non-finite number
```

Assert classification precedence:

```text
correctness
-> evidence completeness
-> trace coverage
-> profiler overhead
-> headroom GO/NO_GO
```

- [ ] **Step 2: Run the tests and confirm RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q \
  tools/test_lease_sealed_persistent_decode_ceiling.py \
  -p no:cacheprovider
```

Expected: collection fails because the ceiling module does not exist.

- [ ] **Step 3: Implement immutable constants and arithmetic**

Define:

```python
CONTEXT_LENGTHS = (256, 2048, 8192)
GENERATED_TOKENS = 128
REPETITIONS = 5
MIN_AGGREGATE_OPTIMISTIC_IMPROVEMENT_PCT = 5.0
MIN_CONTEXT_OPTIMISTIC_IMPROVEMENT_PCT = 3.0
MIN_CANDIDATE_CUDA_DURATION_SHARE_PCT = 4.0
MIN_CLASSIFIED_LAUNCH_RATIO = 0.98
MIN_CLASSIFIED_DURATION_RATIO = 0.99
MAX_MEDIAN_PROFILE_PERTURBATION_PCT = 10.0
MAX_P95_PROFILE_PERTURBATION_PCT = 15.0
```

Use nearest-rank P95 and ordinary median. Reject booleans as numbers,
negative durations, NaN/Infinity, duplicate identities, missing
context/repetition pairs, and profiled timings used as the main denominator.

Emit:

```text
lease-sealed-persistent-decode.ceiling.v1
```

with observed values, thresholds, per-context metrics, aggregate metrics,
failed conditions, and exactly one terminal classification.

- [ ] **Step 4: Run focused and adjacent GREEN**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q \
  tools/test_lease_sealed_persistent_decode_ceiling.py \
  tools/test_persistent_decode_kernel_trace.py \
  -p no:cacheprovider
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 3**

Run:

```bash
git add -- \
  tools/lease_sealed_persistent_decode_ceiling.py \
  tools/test_lease_sealed_persistent_decode_ceiling.py
git diff --cached --check
git -c core.hooksPath=/dev/null commit \
  -m "feat(profiler): gate persistent decode headroom" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

### Task 4: Build the Exact K8 timing and structural producer

**Files:**

- Create: `tools/profile_lease_sealed_persistent_decode_ceiling.py`
- Create: `tools/test_profile_lease_sealed_persistent_decode_ceiling.py`
- Reuse without production changes:
  `tools/profile_exact_greedy_decode_burst.py`

**Interfaces:**

- Produces:
  `build_timing_identities() -> Sequence[tuple[int, int]]`, where each
  element is `(repetition, prompt_tokens)`
- Produces:
  `build_trace_identities() -> tuple[int, int, int]`, containing the three
  frozen prompt lengths
- Produces:
  `trace_label(*, attempt: str, workload: str, repetition: int,
  context: int, burst: int, logical_tokens: int) -> str`
- Produces:
  `run_timing_case(*, model: str, run_tag: str, source_commit: str,
  repetition: int, prompt_tokens: int, generated_tokens: int,
  gpu_memory_utilization: float) -> dict`
- Produces:
  `run_structural_case(*, model: str, run_tag: str, source_commit: str,
  prompt_tokens: int, generated_tokens: int,
  gpu_memory_utilization: float) -> dict`
- CLI modes:
  `--mode timing`, `--mode structural`, and `--mode finalize`

- [ ] **Step 1: Write failing producer-contract tests**

Assert:

```python
def test_timing_inventory_is_five_repetitions_for_three_contexts():
    assert build_timing_identities() == tuple(
        (repetition, context)
        for repetition in range(5)
        for context in (256, 2048, 8192)
    )


def test_structural_inventory_is_one_matched_case_per_context():
    assert build_trace_identities() == (256, 2048, 8192)
```

Also cover:

- timing mode calls the existing Exact K8 production entrypoint;
- timing mode never enables `DecodeInternalProfiler`;
- structural mode surrounds only measured `llm.step()` calls with
  `persistent_decode_trace/attempt=<tag>/workload=exact_greedy_k8/`
  `repetition=0/context=<tokens>/burst=<ordinal>/`
  `logical_tokens=<count>` NVTX ranges;
- structural mode writes emitted logical-token counts separately and the
  finalizer matches them to trace ranges;
- warmup and model-load kernels are outside selected ranges;
- output token IDs and text digest are identical across arms;
- exactly 128 generated tokens;
- target-forward and committed-token counts are exact;
- source manifest includes every new qualification file;
- runtime manifest records Python, PyTorch, CUDA, GPU, Nsight, model path,
  checkpoint inventory, and feature configuration;
- no path escapes the supplied output directory.

- [ ] **Step 2: Run the tests and confirm RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q \
  tools/test_profile_lease_sealed_persistent_decode_ceiling.py \
  -p no:cacheprovider
```

Expected: collection fails because the producer does not exist.

- [ ] **Step 3: Implement timing mode**

Import and reuse:

```python
from tools.profile_exact_greedy_decode_burst import (
    _aggregate_memory,
    _capture_cost,
    _combined_summary,
    _construct_llm,
    _make_prompt,
    _run_request,
    _runner_summaries,
    sha256_text,
)
```

Construct policy `decode_burst_k8`, run two warmups, clear reusable prefix
cache between requests, and call:

```python
_run_request(
    llm,
    prompt=prompt,
    generated_tokens=128,
    policy="decode_burst_k8",
    profile_label=None,
)
```

The timing arm therefore adds no CUDA events or NVTX instrumentation.

Write one validated row to `timing_rows.jsonl` with schema:

```text
lease-sealed-persistent-decode.timing.v1
```

- [ ] **Step 4: Implement structural mode**

Construct the same Exact K8 configuration. After warmup, wrap every measured
engine step in:

```python
with torch.cuda.nvtx.range(
    trace_label(
        attempt=run_tag,
        workload="exact_greedy_k8",
        repetition=0,
        context=prompt_tokens,
        burst=burst_ordinal,
        logical_tokens=expected_emitted,
    )
):
    outputs, _ = llm.step(
        completion_only=True,
    )
```

Record actual emitted tokens, final output IDs/text digest, exact-burst
counters, and unprofiled-compatible wall timings in
`structural_rows.jsonl`. Reject expected-versus-actual emitted-token
mismatch.

- [ ] **Step 5: Implement finalize mode**

Read the three timing/structural outputs and three SQLite paths, call
`read_decode_trace()`, classify kernels, build segments, aggregate coverage,
compute the ceiling, and atomically write:

```text
timing_summary.json
trace_inventory.json
kernel_rows.jsonl
segment_rows.jsonl
ceiling.json
```

Do not copy `.nsys-rep` or `.sqlite` into a second remote directory.

- [ ] **Step 6: Run focused GREEN**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q \
  tools/test_profile_lease_sealed_persistent_decode_ceiling.py \
  tools/test_lease_sealed_persistent_decode_ceiling.py \
  tools/test_persistent_decode_kernel_trace.py \
  -p no:cacheprovider
```

Expected: all tests pass.

- [ ] **Step 7: Commit Task 4**

Run:

```bash
git add -- \
  tools/profile_lease_sealed_persistent_decode_ceiling.py \
  tools/test_profile_lease_sealed_persistent_decode_ceiling.py
git diff --cached --check
git -c core.hooksPath=/dev/null commit \
  -m "feat(profiler): produce persistent decode evidence" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

### Task 5: Add an independent verifier

**Files:**

- Create: `tools/verify_lease_sealed_persistent_decode_ceiling.py`
- Create: `tools/test_verify_lease_sealed_persistent_decode_ceiling.py`

**Interfaces:**

- Produces:
  `verify_artifact_directory(run_dir: Path) -> dict`
- Produces CLI:
  `python tools/verify_lease_sealed_persistent_decode_ceiling.py RUN_DIR`

- [ ] **Step 1: Write failing independent-verifier tests**

Build a complete fixture and assert:

```python
def test_verifier_reconstructs_ceiling_without_trusting_ceiling_json(tmp_path):
    run_dir = make_complete_artifact(tmp_path)
    corrupt_reported_aggregate(run_dir / "ceiling.json")
    with pytest.raises(ValueError, match="ceiling mismatch"):
        verify_artifact_directory(run_dir)
```

Add mutations for:

- one missing timing row;
- one missing structural context;
- one changed output token;
- one changed text digest;
- one changed source hash;
- one unclassified kernel causing coverage failure;
- one altered segment duration;
- one altered threshold;
- one changed terminal classification;
- duplicate artifact path;
- path traversal in manifest;
- extra undeclared file;
- missing raw-trace digest;
- NaN/Infinity.

- [ ] **Step 2: Run the tests and confirm RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q \
  tools/test_verify_lease_sealed_persistent_decode_ceiling.py \
  -p no:cacheprovider
```

Expected: collection fails because the verifier does not exist.

- [ ] **Step 3: Implement independent reconstruction**

Do not import `compute_ceiling()` or `classify_ceiling()` from the producer
classifier. Duplicate the small arithmetic deliberately. Import only
schema-neutral helpers such as file hashing if needed.

Require exactly:

```text
source_manifest.json
runtime_manifest.json
gpu_admission.json
workload_manifest.json
timing_rows.jsonl
structural_rows.jsonl
timing_summary.json
trace_inventory.json
kernel_rows.jsonl
segment_rows.jsonl
ceiling.json
manifest.json
```

Raw trace files may be represented by immutable remote path, byte length,
and SHA-256 rows in `trace_inventory.json`; they need not be duplicated into
the compact local artifact.

- [ ] **Step 4: Run focused GREEN**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q \
  tools/test_verify_lease_sealed_persistent_decode_ceiling.py \
  tools/test_profile_lease_sealed_persistent_decode_ceiling.py \
  tools/test_lease_sealed_persistent_decode_ceiling.py \
  tools/test_persistent_decode_kernel_trace.py \
  -p no:cacheprovider
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 5**

Run:

```bash
git add -- \
  tools/verify_lease_sealed_persistent_decode_ceiling.py \
  tools/test_verify_lease_sealed_persistent_decode_ceiling.py
git diff --cached --check
git -c core.hooksPath=/dev/null commit \
  -m "test(profiler): verify persistent decode ceiling" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

### Task 6: Add the mounted-only remote controller

**Files:**

- Create: `tools/run_lease_sealed_persistent_decode_ceiling_remote.py`
- Create: `tools/test_run_lease_sealed_persistent_decode_ceiling_remote.py`

**Interfaces:**

- Produces:
  `remote_paths(run_tag: str) -> dict[str, str]`
- Produces:
  `validate_kerberos(minimum_lifetime_seconds: int) -> dict`
- Produces:
  `strict_clean_a100s(rows: list[dict]) -> list[dict]`
- Produces:
  `build_nsys_command(*, source_dir: str, output_dir: str,
  run_tag: str, source_commit: str, gpu_index: int,
  prompt_tokens: int) -> list[str]`
- Produces:
  `build_worker_plan(*, paths: dict[str, str], run_tag: str,
  source_commit: str, gpu: dict) -> dict`
- Produces:
  `download_compact_bundle(*, remote_path: str,
  local_parent: Path) -> Path`
- Produces CLI with `--run-tag`, `--source-commit`, `--gpu-timeout-seconds`,
  and `--poll-interval-seconds`.

- [ ] **Step 1: Write failing controller tests**

Assert:

```python
def test_remote_paths_stay_below_approved_task_root():
    paths = remote_paths("20260830-persistent-decode-ceiling-test")
    assert all(
        value.startswith(
            "/data00/home/sitian/tinyllmforge-workspaces/"
            "command-timeline-20260818/"
        )
        for value in paths.values()
    )
```

Also cover:

- run-tag validation and immutable destination absence;
- exact pushed-HEAD equality;
- Kerberos FILE cache and minimum-TTL fail-fast;
- no `kinit`, `krenew`, `/tmp`, `/private/tmp`, old checkout, or local
  `experiments/` string in generated commands;
- source archive contains only committed `tinyvllm/` and `tools/`;
- one strict-clean A100 selection;
- second admission immediately before producer launch;
- no process-kill command;
- `nsys` exact path and version receipt;
- one timing producer and three structural Nsight invocations;
- `nsys export --type=sqlite` stays below the task root;
- remote verifier runs from frozen staged source;
- compact download excludes `.nsys-rep` and `.sqlite`;
- local verifier runs against the compact bundle plus streamed temporary raw
  traces;
- interrupted producer evidence is preserved and never overwritten.

- [ ] **Step 2: Run the tests and confirm RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q \
  tools/test_run_lease_sealed_persistent_decode_ceiling_remote.py \
  -p no:cacheprovider
```

Expected: collection fails because the controller does not exist.

- [ ] **Step 3: Implement source, authentication, storage, and GPU guards**

Reuse proven helpers from:

```python
from tools import run_staged_inference_benchmark_remote as base
from tools import run_phase_stitched_exact_graph_remote as transport
```

Freeze:

```python
REMOTE_HOST = "sitian@10.232.195.203"
REMOTE_PYTHON = "/data00/home/sitian/tllm/env/bin/python"
MODEL_PATH = "/data00/home/sitian/.ms_cache/Qwen/Qwen3-0___6B"
KRB5_CACHE = "FILE:/Users/bytedance/krb5cc_sitian"
APPROVED_ROOT = (
    "/data00/home/sitian/tinyllmforge-workspaces/"
    "command-timeline-20260818"
)
MINIMUM_KERBEROS_LIFETIME_SECONDS = 5_400
```

The controller must pass `KRB5CCNAME` explicitly in every local SSH process
environment.

- [ ] **Step 4: Implement the immutable execution plan**

The worker sequence is:

```text
upload committed source archive
-> verify source inventory
-> record first strict-clean admission
-> record immediate second strict-clean admission
-> run one timing producer
-> run three bounded Nsight structural producers
-> export each report to SQLite remotely
-> finalize compact evidence remotely
-> run frozen-source remote independent verifier
-> build manifest and exit receipt
-> stream raw SQLite files through a local temporary directory
-> run local independent verifier
-> remove temporary raw traces
-> download compact terminal bundle
```

Never reuse a tag after any remote destination is created.

- [ ] **Step 5: Run focused GREEN**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q \
  tools/test_run_lease_sealed_persistent_decode_ceiling_remote.py \
  tools/test_verify_lease_sealed_persistent_decode_ceiling.py \
  tools/test_profile_lease_sealed_persistent_decode_ceiling.py \
  tools/test_lease_sealed_persistent_decode_ceiling.py \
  tools/test_persistent_decode_kernel_trace.py \
  -p no:cacheprovider
```

Expected: all tests pass.

- [ ] **Step 6: Commit Task 6**

Run:

```bash
git add -- \
  tools/run_lease_sealed_persistent_decode_ceiling_remote.py \
  tools/test_run_lease_sealed_persistent_decode_ceiling_remote.py
git diff --cached --check
git -c core.hooksPath=/dev/null commit \
  -m "feat(profiler): run persistent decode ceiling remotely" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

### Task 7: Run the complete local qualification suite

**Files:**

- Verify only; modify a file only when a meaningful RED exposes a defect.

- [ ] **Step 1: Run all new tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q \
  tools/test_persistent_decode_kernel_trace.py \
  tools/test_lease_sealed_persistent_decode_ceiling.py \
  tools/test_profile_lease_sealed_persistent_decode_ceiling.py \
  tools/test_verify_lease_sealed_persistent_decode_ceiling.py \
  tools/test_run_lease_sealed_persistent_decode_ceiling_remote.py \
  -p no:cacheprovider
```

Expected: all tests pass.

- [ ] **Step 2: Run adjacent exact-burst and Nsight suites**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q \
  tools/test_profile_exact_greedy_decode_burst.py \
  tools/test_exact_greedy_decode_burst_gate.py \
  tools/test_exact_greedy_decode_burst_verify.py \
  tools/test_exact_burst_octet_folded_graph_ceiling.py \
  tools/test_exact_burst_octet_folded_graph_verify.py \
  tools/test_qwen38_nsys_intervals.py \
  tools/test_assemble_qwen38_tp4_communication_profile.py \
  -p no:cacheprovider
```

Expected: all tests pass.

- [ ] **Step 3: Verify source leakage and repository state**

Run:

```bash
rg -n -i \
  'qwen|llama|k8|octet|a100' \
  tools/persistent_decode_kernel_trace.py \
  tools/lease_sealed_persistent_decode_ceiling.py
git diff --check
git status --short --untracked-files=no
```

Expected: the source scan has no matches; diff check passes; tracked files
are clean after the task commits.

- [ ] **Step 4: Confirm local and remote commit identity**

Run:

```bash
git rev-parse HEAD
git ls-remote origin refs/heads/feat/kv-sparse-attention
```

Expected: both SHAs are identical.

### Task 8: Run the immutable A100 ceiling campaign

**Files:**

- Produce:
  `artifacts/lease_sealed_persistent_decode/`
  `20260830-qwen3-06b-persistent-decode-ceiling-r1/`
- Do not modify production runtime files.

- [ ] **Step 1: Perform read-only SSH and profiler preflight**

Run:

```bash
KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian \
ssh -o BatchMode=yes -o ControlMaster=no -o ControlPath=none \
  -o ConnectTimeout=20 sitian@10.232.195.203 \
  'hostname; id -un; findmnt -T /data00/home/sitian; \
   /usr/local/bin/nsys --version; \
   nvidia-smi --query-gpu=index,uuid,name,memory.used,utilization.gpu \
     --format=csv,noheader,nounits; \
   nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name \
     --format=csv,noheader,nounits'
```

Expected: SSH succeeds, mounted storage is visible, Nsight version is
recorded, and at least one strict-clean A100 exists. Do not launch if the
second controller admission later fails.

- [ ] **Step 2: Launch one fresh immutable tag**

Use a new tag:

```bash
PYTHONDONTWRITEBYTECODE=1 \
KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian \
python3 tools/run_lease_sealed_persistent_decode_ceiling_remote.py \
  --run-tag 20260830-qwen3-06b-persistent-decode-ceiling-r1 \
  --source-commit "$(git rev-parse HEAD)" \
  --gpu-timeout-seconds 7200 \
  --poll-interval-seconds 30
```

Expected: one producer lifecycle, remote verification exit `0`, local
verification exit `0`, compact bundle downloaded, and no raw profiler report
retained locally.

- [ ] **Step 3: Inspect terminal evidence**

Run:

```bash
python3 tools/verify_lease_sealed_persistent_decode_ceiling.py \
  artifacts/lease_sealed_persistent_decode/\
20260830-qwen3-06b-persistent-decode-ceiling-r1
```

Expected: verifier exit `0` and exactly one terminal classification.

- [ ] **Step 4: Apply the stop rule**

If classification is not `GO_PERSISTENT_DECODE_CEILING`, do not write
production kernel or runtime code.

If classification is `GO_PERSISTENT_DECODE_CEILING`, stop this plan after
publishing the evidence and write a separate runtime design before any CUDA
or Triton implementation.

### Task 9: Audit, reconcile, commit, and push terminal evidence

**Files:**

- Create:
  `docs/superpowers/audits/2026-08-30-lease-sealed-persistent-decode-megakernel-ceiling-audit.md`
- Modify:
  `docs/superpowers/audits/2026-08-16-phase1-completion-audit.md`
- Modify: `AGENT_HANDOFF_STATE.md`

- [ ] **Step 1: Write the prompt-to-artifact checklist**

The audit maps every design requirement to:

```text
requirement
implementation file and symbol
test name and result
artifact path and field
remote verifier field
local verifier field
classification consequence
```

Explicitly include:

- source and pushed-HEAD identity;
- Kerberos cache and TTL guard;
- approved remote storage root;
- two strict-clean admissions;
- exact timing/trace inventories;
- output token/text equality;
- target-forward and committed-token counts;
- kernel launch/duration coverage;
- segment signature stability;
- profiler perturbation;
- aggregate/per-context headroom;
- benefit and cost;
- raw trace remote-only handling;
- remote/local verifier equality;
- stop-rule compliance;
- originality and claim boundaries.

- [ ] **Step 2: Reconcile Phase 1 and handoff**

Append a dated section to both existing files. State:

- immutable tag and source SHA;
- selected GPU UUID;
- exact terminal classification;
- every observed threshold and pass/fail;
- profiler runtime and remote raw-byte cost;
- local compact artifact size;
- whether runtime implementation is authorized;
- exact next command.

- [ ] **Step 3: Run completion verification**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q \
  tools/test_persistent_decode_kernel_trace.py \
  tools/test_lease_sealed_persistent_decode_ceiling.py \
  tools/test_profile_lease_sealed_persistent_decode_ceiling.py \
  tools/test_verify_lease_sealed_persistent_decode_ceiling.py \
  tools/test_run_lease_sealed_persistent_decode_ceiling_remote.py \
  tools/test_profile_exact_greedy_decode_burst.py \
  tools/test_exact_greedy_decode_burst_gate.py \
  tools/test_exact_greedy_decode_burst_verify.py \
  tools/test_exact_burst_octet_folded_graph_ceiling.py \
  tools/test_exact_burst_octet_folded_graph_verify.py \
  tools/test_qwen38_nsys_intervals.py \
  -p no:cacheprovider

git diff --check
```

Expected: all tests pass and diff check is clean.

- [ ] **Step 4: Commit and push documentation**

Run:

```bash
git add -- \
  docs/superpowers/audits/2026-08-30-lease-sealed-persistent-decode-megakernel-ceiling-audit.md \
  docs/superpowers/audits/2026-08-16-phase1-completion-audit.md \
  AGENT_HANDOFF_STATE.md
git diff --cached --check
git -c core.hooksPath=/dev/null commit \
  -m "docs(runtime): reconcile persistent decode ceiling" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

- [ ] **Step 5: Confirm final repository identity**

Run:

```bash
git status --short --untracked-files=no
git rev-parse HEAD
git ls-remote origin refs/heads/feat/kv-sparse-attention
git log -1 --format='%B'
```

Expected:

- tracked worktree is clean;
- local and remote SHA match;
- the final commit has exactly one required co-author trailer;
- the audit and handoff identify the same source, tag, metrics, and terminal
  classification.
