# Verifier-Aware Quantized Draft Stage-0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:executing-plans to implement this plan task-by-task. This
> repository's active constraints require inline execution; do not create a
> worktree or dispatch subagents. Steps use checkbox (`- [ ]`) syntax for
> tracking.

**Goal:** Build and qualify an A100-compatible fused W4A16 drafter GEMM path
that consumes packed INT4 weights directly and either proves a real
microkernel opportunity or stops the larger distillation project.

**Architecture:** Add a model-neutral Triton W4A16 primitive beside the
existing storage-only quantization path, then compare it against BF16 and the
current dequantize-then-GEMM implementation on a frozen manifest of real draft
model linear shapes. A fail-closed controller, assembler, independent
verifier, compact artifact bundle, and audit provide the sole Stage-0
classification authority.

**Tech Stack:** Python 3.11, PyTorch 2.4.1+cu121, Triton from the existing
remote environment, NVIDIA A100 80GB, pytest, CUDA events, SSH, SHA-256
manifests.

## Global Constraints

- Work only in `/Users/bytedance/dev/TinyLLMForge` on
  `feat/kv-sparse-attention`.
- Do not create a worktree and do not use subagents.
- Use RED -> minimal implementation -> GREEN for every code task.
- Stage only exact paths; never use broad `git add`, `git reset`, `git clean`,
  or unrelated formatting.
- Commit with `git -c core.hooksPath=/dev/null commit` and exactly one
  `Co-authored-by: TRAE CLI <noreply@bytedance.com>` trailer.
- Push only to `origin/feat/kv-sparse-attention` and confirm local and remote
  SHA equality.
- All remote task files, compilation caches, logs, and raw artifacts must be
  below
  `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/`.
- Do not write task data to remote `/`, `/tmp`, the model cache, or an older
  checkout.
- Do not execute `kinit` or `krenew`.
- Do not terminate, take over, or clean up an external GPU process.
- Stage 0 needs one clean A100, not four. The controller may wait for one
  eligible device but must not reserve idle devices it does not use.
- Keep large traces remote. Download only the compact bundle, controller
  receipts, independent-verifier receipts, and audit.
- The target model and speculative runtime are not modified in Stage 0.
- The existing `tinyvllm/layers/linear.py` dispatch remains unchanged until a
  terminal Stage-0 GO.
- A packed checkpoint, lower allocated bytes, or a correct kernel is not a
  speedup claim.
- A Stage-0 NO-GO stops the INT4 route before distillation work.

---

## File Map

### Runtime primitive

- Create `tinyvllm/layers/fused_int4_linear.py`
  - lazy Triton import;
  - immutable support result;
  - shape and dtype validation;
  - packed W4A16 launch;
  - explicit unsupported-shape fallback result;
  - no integration into `LinearBase`.

### Gate policy and evidence

- Create `tools/quantized_draft_int4_microgate.py`
  - shape-manifest schema;
  - frozen thresholds;
  - row validation;
  - weighted and per-shape classification.
- Create `tools/quantized_draft_int4_microgate_worker.py`
  - real checkpoint shape extraction;
  - BF16, current INT4, and fused INT4 execution;
  - paired CUDA and host timing;
  - memory, correctness, graph, and cleanup evidence.
- Create `tools/assemble_quantized_draft_int4_microgate.py`
  - compact bundle assembly;
  - terminal classification;
  - complete-inventory SHA-256 manifest.
- Create `tools/verify_quantized_draft_int4_microgate.py`
  - independent raw-row recomputation;
  - manifest and identity validation.
- Create `tools/run_quantized_draft_int4_microgate.py`
  - local admission;
  - remote source staging under the approved root;
  - Kerberos TTL and clean-GPU monitoring;
  - launch, remote verification, compact download, and local verification.

### Tests

- Create `tools/test_fused_int4_linear.py`.
- Create `tools/test_quantized_draft_int4_microgate.py`.
- Create `tools/test_quantized_draft_int4_microgate_worker.py`.
- Create `tools/test_assemble_quantized_draft_int4_microgate.py`.
- Create `tools/test_verify_quantized_draft_int4_microgate.py`.
- Create `tools/test_run_quantized_draft_int4_microgate.py`.

### Terminal evidence

- Create
  `artifacts/quantized_draft_int4/20260831-quantized-draft-int4-stage0-r1/controller/`.
- Create
  `artifacts/quantized_draft_int4/20260831-quantized-draft-int4-stage0-r1/final_bundle/`.
- Create
  `docs/superpowers/audits/2026-08-31-quantized-draft-int4-stage0-audit.md`.

---

### Task 1: Freeze the shape and classification contract

**Files:**

- Create: `tools/quantized_draft_int4_microgate.py`
- Test: `tools/test_quantized_draft_int4_microgate.py`

**Interfaces:**

- Produces:

```python
@dataclass(frozen=True)
class DraftLinearShape:
    shape_id: str
    input_features: int
    output_features: int
    execution_count: int
    group_size: int

@dataclass(frozen=True)
class QuantizedDraftInt4Policy:
    minimum_pairs_per_shape: int = 200
    maximum_candidate_to_bf16_median_ratio: float = 0.75
    maximum_candidate_to_bf16_p99_ratio: float = 0.95
    maximum_weight_bytes_ratio: float = 0.40
    maximum_absolute_error: float = 0.08
    maximum_relative_error: float = 0.08

def validate_shape_manifest(payload: object) -> tuple[DraftLinearShape, ...]:
    raise NotImplementedError

def classify_int4_microgate(
    *,
    shapes: tuple[DraftLinearShape, ...],
    rows: object,
    memory: object,
    graph: object,
    cleanup: object,
) -> dict[str, object]:
    raise NotImplementedError
```

- Classification values:

```text
GO_FUSED_INT4_DRAFT_KERNEL
NO_GO_CORRECTNESS
NO_GO_PERFORMANCE
NO_GO_MEMORY
NO_GO_GRAPH
INCONCLUSIVE_EVIDENCE
```

- Precedence:

```text
NO_GO_CORRECTNESS
  > INCONCLUSIVE_EVIDENCE
  > NO_GO_MEMORY
  > NO_GO_GRAPH
  > NO_GO_PERFORMANCE
  > GO_FUSED_INT4_DRAFT_KERNEL
```

- A row identity is:

```text
(shape_id, pair_index)
```

- Every measured row contains:

```text
shape_id
pair_index
arm_order
bf16_cuda_ns
dequant_cuda_ns
fused_int4_cuda_ns
bf16_host_submission_ns
dequant_host_submission_ns
fused_int4_host_submission_ns
maximum_absolute_error
maximum_relative_error
fallback_reason
full_dequant_allocation_observed
```

- [ ] **Step 1: Write the failing contract tests**

Create tests that construct three shapes and 200 rows per shape. Cover:

```python
def test_shape_manifest_requires_unique_positive_aligned_shapes():
    shapes = validate_shape_manifest({
        "schema_version": 1,
        "shapes": [{
            "shape_id": "q_proj",
            "input_features": 1024,
            "output_features": 2048,
            "execution_count": 28,
            "group_size": 128,
        }],
    })
    assert shapes[0].shape_id == "q_proj"


def test_classifier_accepts_complete_profitable_evidence():
    result = classify_int4_microgate(
        shapes=passing_shapes(),
        rows=passing_rows(),
        memory=passing_memory(),
        graph=passing_graph(),
        cleanup={"classification": "CLEAN"},
    )
    assert result["classification"] == "GO_FUSED_INT4_DRAFT_KERNEL"


@pytest.mark.parametrize(
    ("mutation", "classification"),
    (
        ("error", "NO_GO_CORRECTNESS"),
        ("missing_pair", "INCONCLUSIVE_EVIDENCE"),
        ("duplicate_pair", "INCONCLUSIVE_EVIDENCE"),
        ("nonfinite", "INCONCLUSIVE_EVIDENCE"),
        ("full_dequant", "INCONCLUSIVE_EVIDENCE"),
        ("weight_bytes", "NO_GO_MEMORY"),
        ("graph", "NO_GO_GRAPH"),
        ("median", "NO_GO_PERFORMANCE"),
        ("p99", "NO_GO_PERFORMANCE"),
        ("cleanup", "INCONCLUSIVE_EVIDENCE"),
    ),
)
def test_classifier_fails_closed(mutation, classification):
    shapes = passing_shapes()
    rows = passing_rows()
    memory = passing_memory()
    graph = passing_graph()
    cleanup = {"classification": "CLEAN"}
    mutate_fixture(
        mutation,
        rows=rows,
        memory=memory,
        graph=graph,
        cleanup=cleanup,
    )
    assert classify_int4_microgate(
        shapes=shapes,
        rows=rows,
        memory=memory,
        graph=graph,
        cleanup=cleanup,
    )["classification"] == classification
```

- [ ] **Step 2: Run the tests and confirm RED**

Run:

```bash
python -m pytest -q tools/test_quantized_draft_int4_microgate.py
```

Expected: collection or import failure because
`tools.quantized_draft_int4_microgate` does not exist.

- [ ] **Step 3: Implement the minimal contract**

Implement strict validation with:

```python
def _finite_nonnegative(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )
```

Require:

- non-empty unique shape IDs;
- positive dimensions and execution counts;
- `input_features % group_size == 0`;
- `input_features % 2 == 0`;
- exactly 200 unique pairs per shape;
- both arm orders with a difference of at most one;
- finite positive timing;
- `fallback_reason is None`;
- `full_dequant_allocation_observed is False`;
- complete graph and cleanup receipts; and
- weight bytes computed from packed tensor plus scales rather than declared by
  the producer alone.

Aggregate performance is weighted by `execution_count`, but every individual
shape must pass median and P99 thresholds. A high-frequency shape may not hide
an unsupported or regressing low-frequency shape.

- [ ] **Step 4: Run the contract tests and confirm GREEN**

Run:

```bash
python -m pytest -q tools/test_quantized_draft_int4_microgate.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add -- \
  tools/quantized_draft_int4_microgate.py \
  tools/test_quantized_draft_int4_microgate.py
git -c core.hooksPath=/dev/null commit \
  -m "test(runtime): freeze INT4 draft microgate contract" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

---

### Task 2: Add a model-neutral fused W4A16 Triton primitive

**Files:**

- Create: `tinyvllm/layers/fused_int4_linear.py`
- Test: `tools/test_fused_int4_linear.py`

**Interfaces:**

- Consumes the existing packed layout from:

```python
quantize_int4(
    weight: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]
```

- Produces:

```python
@dataclass(frozen=True)
class FusedInt4Support:
    supported: bool
    reason: str | None

def fused_int4_support(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    scales: torch.Tensor,
    *,
    group_size: int,
) -> FusedInt4Support:
    raise NotImplementedError

def fused_int4_linear(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    scales: torch.Tensor,
    *,
    group_size: int,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    raise NotImplementedError
```

- Input layout:

```text
x:             [M, K], contiguous FP16 or BF16 CUDA
packed_weight: [N, K / 2], contiguous uint8 CUDA
scales:        [N, K / group_size], contiguous FP32 CUDA
output:        [M, N], same floating dtype and device as x
```

- [ ] **Step 1: Write failing support and dispatch tests**

Cover:

```python
def test_support_accepts_aligned_cuda_contract():
    result = fused_int4_support(
        fake_tensor((4, 1024), "bfloat16", "cuda"),
        fake_tensor((2048, 512), "uint8", "cuda"),
        fake_tensor((2048, 8), "float32", "cuda"),
        group_size=128,
    )
    assert result == FusedInt4Support(True, None)


@pytest.mark.parametrize(
    "mutation",
    (
        "cpu",
        "dtype",
        "rank",
        "packed_shape",
        "scale_shape",
        "group",
        "noncontiguous",
    ),
)
def test_support_rejects_invalid_contract(mutation):
    x, packed, scales = valid_fake_tensors()
    x, packed, scales, group_size = mutate_support_case(
        mutation,
        x=x,
        packed=packed,
        scales=scales,
        group_size=128,
    )
    result = fused_int4_support(
        x,
        packed,
        scales,
        group_size=group_size,
    )
    assert result.supported is False
    assert result.reason is not None


def test_launch_passes_packed_weight_without_full_dequantization(monkeypatch):
    calls = []
    monkeypatch.setattr(module, "_launch_triton", calls.append)
    monkeypatch.setattr(
        quantization,
        "dequantize_int4",
        lambda *args, **kwargs: pytest.fail("must not dequantize"),
    )
    fused_int4_linear(x, packed, scales, group_size=128, output=out)
    assert calls
```

- [ ] **Step 2: Run the tests and confirm RED**

Run:

```bash
python -m pytest -q tools/test_fused_int4_linear.py
```

Expected: import failure because `tinyvllm.layers.fused_int4_linear` does not
exist.

- [ ] **Step 3: Implement validation and lazy Triton loading**

The module must import on CPU-only test hosts. Import Triton only inside:

```python
def _triton_modules():
    import triton
    import triton.language as tl
    return triton, tl
```

Do not import or mutate `tinyvllm.layers.linear`.

Use stable fallback reasons:

```text
not_cuda
unsupported_activation_dtype
invalid_rank
noncontiguous
packed_shape_mismatch
scale_shape_mismatch
unsupported_group_size
unsupported_alignment
```

- [ ] **Step 4: Implement the packed W4A16 kernel**

Use one Triton program per `[BLOCK_M, BLOCK_N]` output tile. For each
`BLOCK_K` slice:

```python
packed = tl.load(packed_ptrs, mask=packed_mask, other=0)
low = (packed & 0x0F).to(tl.int8) - 8
high = ((packed >> 4) & 0x0F).to(tl.int8) - 8
q = tl.interleave(low, high)
scale = tl.load(scale_ptrs, mask=scale_mask, other=0.0)
w = q.to(tl.float32) * scale
accumulator += tl.dot(x_tile, w_transposed)
```

Required implementation properties:

- unpack only the current K tile into registers;
- never allocate `[N, K]` dequantized weights;
- accumulate in FP32;
- cast once when storing the output;
- support `M in {1, 2, 4, 8}`;
- initially support `group_size in {32, 64, 128}`;
- require K and N alignments selected by the autotune configurations;
- use a bounded configuration list, not unconstrained autotuning during
  measured rows; and
- expose a warmup function that compiles every frozen shape before timing.

- [ ] **Step 5: Run CPU contract tests**

Run:

```bash
python -m pytest -q tools/test_fused_int4_linear.py
```

Expected: all dependency-light tests pass without requiring CUDA.

- [ ] **Step 6: Run syntax and import checks**

Run:

```bash
python -m py_compile \
  tinyvllm/layers/fused_int4_linear.py \
  tools/test_fused_int4_linear.py
```

Expected: exit zero.

- [ ] **Step 7: Commit Task 2**

```bash
git add -- \
  tinyvllm/layers/fused_int4_linear.py \
  tools/test_fused_int4_linear.py
git -c core.hooksPath=/dev/null commit \
  -m "feat(runtime): add fused W4A16 draft primitive" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

---

### Task 3: Build the real-shape worker

**Files:**

- Create: `tools/quantized_draft_int4_microgate_worker.py`
- Test: `tools/test_quantized_draft_int4_microgate_worker.py`

**Interfaces:**

- Produces:

```python
def extract_linear_shape_manifest(
    *,
    model_path: Path,
    model_loader,
) -> dict[str, object]:
    raise NotImplementedError

def build_pair_schedule(
    *,
    shapes: tuple[DraftLinearShape, ...],
    warmup_pairs: int = 2,
    measured_pairs: int = 200,
) -> tuple[dict[str, object], ...]:
    raise NotImplementedError

def validate_worker_arguments(args: argparse.Namespace) -> None:
    raise NotImplementedError

def run_worker(args: argparse.Namespace) -> dict[str, object]:
    raise NotImplementedError
```

- Worker artifacts:

```text
environment.json
shape_manifest.json
microgate_rows.jsonl
memory.json
graph.json
cleanup.json
```

- [ ] **Step 1: Write failing worker tests**

Cover:

```python
def test_shape_extraction_coalesces_identical_linear_shapes():
    manifest = extract_linear_shape_manifest(
        model_path=Path("/model"),
        model_loader=fake_model_loader,
    )
    assert manifest["schema_version"] == 1
    assert manifest["shapes"] == [{
        "shape_id": "m1_k1024_n2048_g128",
        "input_features": 1024,
        "output_features": 2048,
        "execution_count": 28,
        "group_size": 128,
        "module_names_sha256": EXPECTED_SHA,
    }]


def test_schedule_is_position_balanced_and_complete():
    rows = build_pair_schedule(
        shapes=passing_shapes(),
        measured_pairs=200,
    )
    assert len(rows) == len(passing_shapes()) * 202
    assert sum(row["arm_order"][0] == "bf16" for row in rows) == (
        len(rows) // 2
    )


def test_worker_rejects_output_outside_approved_remote_root():
    args = valid_worker_args(
        output_dir=Path("/var/tmp/outside-approved-root")
    )
    with pytest.raises(ValueError, match="approved remote root"):
        validate_worker_arguments(args)


def test_worker_writes_cleanup_on_candidate_failure(
    tmp_path,
    monkeypatch,
):
    approved = tmp_path / "command-timeline-20260818"
    output = approved / "quantized-draft-int4" / "fixture"
    args = valid_worker_args(
        output_dir=output,
        approved_remote_root=approved,
    )
    monkeypatch.setattr(
        worker,
        "run_measured_candidate",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("candidate failed")
        ),
    )
    with pytest.raises(RuntimeError, match="candidate failed"):
        run_worker(args)
    cleanup = json.loads(
        (output / "cleanup.json").read_text(encoding="utf-8")
    )
    assert cleanup["classification"] != "CLEAN"
```

- [ ] **Step 2: Run the tests and confirm RED**

Run:

```bash
python -m pytest -q \
  tools/test_quantized_draft_int4_microgate_worker.py
```

Expected: import failure because the worker does not exist.

- [ ] **Step 3: Implement real shape extraction**

Load the existing draft checkpoint

```text
/data00/home/sitian/.ms_cache/Qwen/Qwen3-0___6B
```

through the established TinyLLMForge model builder. Enumerate linear modules
before quantization and group identical `[N, K]` shapes. Hash sorted module
names rather than storing all names in the compact summary.

Reject:

- missing or non-file checkpoint artifacts;
- an empty shape inventory;
- dimensions incompatible with packed INT4;
- tokenizer or checkpoint fingerprint failure; and
- a model load that writes into its source directory.

- [ ] **Step 4: Implement paired execution**

For each shape:

1. allocate one deterministic BF16 activation and weight;
2. create packed INT4 and scales once before warmup;
3. compile the fused candidate before measured rows;
4. run two warmup pairs;
5. run 200 alternating measured pairs;
6. measure each arm with dedicated CUDA start/end events;
7. synchronize only after all three arms in the pair are submitted;
8. record host submission separately;
9. compare the fused result with the current dequantized INT4 result;
10. track peak allocation around each arm; and
11. reject any candidate dispatch that reports fallback.

The current dequantize-then-GEMM arm is diagnostic only. Stage-0 promotion is
based on fused INT4 versus BF16.

- [ ] **Step 5: Implement graph and cleanup probes**

For every shape:

```python
graph = torch.cuda.CUDAGraph()
static_x = torch.empty_like(x)
static_out = torch.empty_like(reference)
with torch.cuda.graph(graph):
    fused_int4_linear(
        static_x,
        packed,
        scales,
        group_size=shape.group_size,
        output=static_out,
    )
```

Record capture, two successful replays, static pointers, and identical replay
output. Do not make graph capture a hidden warmup side effect.

In `finally`:

- synchronize the owned device;
- release model, weights, packed buffers, scales, and graph objects;
- call `torch.cuda.empty_cache()`;
- record final allocated and reserved bytes; and
- write `cleanup.json` atomically even after failure.

- [ ] **Step 6: Run worker unit tests and compile checks**

Run:

```bash
python -m pytest -q \
  tools/test_quantized_draft_int4_microgate_worker.py
python -m py_compile \
  tools/quantized_draft_int4_microgate_worker.py
```

Expected: all tests pass and compilation exits zero.

- [ ] **Step 7: Commit Task 3**

```bash
git add -- \
  tools/quantized_draft_int4_microgate_worker.py \
  tools/test_quantized_draft_int4_microgate_worker.py
git -c core.hooksPath=/dev/null commit \
  -m "test(runtime): add real-shape INT4 draft worker" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

---

### Task 4: Add bundle assembly and an independent verifier

**Files:**

- Create: `tools/assemble_quantized_draft_int4_microgate.py`
- Create: `tools/verify_quantized_draft_int4_microgate.py`
- Test: `tools/test_assemble_quantized_draft_int4_microgate.py`
- Test: `tools/test_verify_quantized_draft_int4_microgate.py`

**Interfaces:**

- Produces:

```python
def assemble_bundle(
    *,
    raw_dir: Path,
    output_dir: Path,
    source_revision: str,
    run_tag: str,
) -> dict[str, object]:
    raise NotImplementedError

def verify_bundle(bundle_dir: Path) -> dict[str, object]:
    raise NotImplementedError
```

- Final bundle inventory:

```text
source_identity.json
environment.json
shape_manifest.json
microgate_rows.jsonl
memory.json
graph.json
cleanup.json
summary.json
classification.json
independent_verification.json
manifest.sha256
```

- [ ] **Step 1: Write failing assembler tests**

Test:

- complete passing raw evidence produces
  `GO_FUSED_INT4_DRAFT_KERNEL`;
- the summary contains per-shape and execution-count-weighted aggregates;
- source revision and run tag are copied into every derived receipt;
- an incomplete raw inventory is rejected;
- a symlink is rejected;
- duplicate row identities are rejected;
- non-finite JSON values are rejected; and
- `manifest.sha256` contains every final file except itself exactly once.

- [ ] **Step 2: Write failing verifier tests**

Test that `verify_bundle()` independently:

- parses raw rows without importing assembler classification helpers;
- recomputes medians, nearest-rank P99, ratios, and weighted aggregates;
- recomputes the terminal classification;
- verifies source and run identity consistency;
- checks the complete manifest;
- rejects an added file;
- rejects a removed file;
- rejects a symlink;
- rejects one-byte mutation; and
- reports both `status: PASS` and the recomputed classification.

- [ ] **Step 3: Run tests and confirm RED**

Run:

```bash
python -m pytest -q \
  tools/test_assemble_quantized_draft_int4_microgate.py \
  tools/test_verify_quantized_draft_int4_microgate.py
```

Expected: import failures because assembler and verifier do not exist.

- [ ] **Step 4: Implement the assembler**

Use atomic writes and canonical JSON:

```python
json.dumps(
    payload,
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False,
)
```

Generate the checksum manifest only after all final files except the verifier
receipt exist. After the independent verifier writes its receipt, regenerate
the manifest and verify it once before publication.

- [ ] **Step 5: Implement the independent verifier**

Do not call `classify_int4_microgate()` from the verifier. Duplicate the
frozen formulas and constants intentionally so a producer bug cannot
self-validate.

Reject any path whose resolved parent escapes the supplied bundle directory.

- [ ] **Step 6: Run assembler/verifier tests and confirm GREEN**

Run:

```bash
python -m pytest -q \
  tools/test_assemble_quantized_draft_int4_microgate.py \
  tools/test_verify_quantized_draft_int4_microgate.py
```

Expected: all tests pass.

- [ ] **Step 7: Commit Task 4**

```bash
git add -- \
  tools/assemble_quantized_draft_int4_microgate.py \
  tools/verify_quantized_draft_int4_microgate.py \
  tools/test_assemble_quantized_draft_int4_microgate.py \
  tools/test_verify_quantized_draft_int4_microgate.py
git -c core.hooksPath=/dev/null commit \
  -m "feat(runtime): verify INT4 draft microgate evidence" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

---

### Task 5: Add the fail-closed remote controller

**Files:**

- Create: `tools/run_quantized_draft_int4_microgate.py`
- Test: `tools/test_run_quantized_draft_int4_microgate.py`

**Interfaces:**

- Produces:

```python
APPROVED_REMOTE_ROOT = (
    "/data00/home/sitian/tinyllmforge-workspaces/"
    "command-timeline-20260818"
)

def build_run_plan(
    *,
    run_tag: str,
    source_revision: str,
    remote_root: str = APPROVED_REMOTE_ROOT,
    remote_python: str = "/data00/home/sitian/tllm/env/bin/python",
    draft_model: str = (
        "/data00/home/sitian/.ms_cache/Qwen/Qwen3-0___6B"
    ),
) -> dict[str, object]:
    raise NotImplementedError

def classify_preflight(payload: object) -> str:
    raise NotImplementedError

def run_controller(args: argparse.Namespace) -> int:
    raise NotImplementedError
```

- Controller receipt inventory:

```text
plan.json
source_identity.json
ssh_storage_preflight.json
kerberos_preflight.json
gpu_admission.json
launch.json
remote_verification.json
download.json
local_verification.json
remote_cleanup_scan.json
```

- [ ] **Step 1: Write failing controller tests**

Cover:

```python
def test_plan_rejects_any_remote_path_outside_approved_root():
    with pytest.raises(ValueError, match="approved remote root"):
        build_run_plan(
            run_tag="fixture-r1",
            source_revision="a" * 40,
            remote_root="/tmp/not-approved",
        )


def test_preflight_requires_one_clean_a100_and_does_not_require_four():
    payload = ready_payload()
    payload["gpus"][1]["compute_process_count"] = 2
    assert classify_preflight(payload) == "READY"


def test_preflight_never_selects_an_occupied_gpu():
    payload = ready_payload()
    for row in payload["gpus"]:
        row["compute_process_count"] = 1
    assert classify_preflight(payload) == "WAIT_GPU"


def test_low_kerberos_ttl_fails_before_source_upload(monkeypatch):
    calls = []
    args = valid_controller_args(kerberos_lifetime_seconds=5399)
    monkeypatch.setattr(
        controller,
        "_upload_source",
        lambda *args, **kwargs: calls.append("upload"),
    )
    result = run_controller(args)
    assert result != 0
    assert calls == []


def test_controller_does_not_emit_kinit_krenew_or_remote_tmp():
    commands = build_remote_commands(valid_plan())
    encoded = "\n".join(commands)
    assert "kinit" not in encoded
    assert "krenew" not in encoded
    assert "/tmp" not in encoded


def test_download_inventory_contains_only_compact_artifacts():
    assert download_inventory("fixture-r1") == (
        "controller",
        "final_bundle",
    )
```

- [ ] **Step 2: Run tests and confirm RED**

Run:

```bash
python -m pytest -q \
  tools/test_run_quantized_draft_int4_microgate.py
```

Expected: import failure because the controller does not exist.

- [ ] **Step 3: Implement local admission and monitoring**

Use:

```text
SSH target: sitian@10.232.195.203
ProxyCommand: ssh -qW %h:%p jump-proxy-lf
Remote Python: /data00/home/sitian/tllm/env/bin/python
Minimum Kerberos lifetime: 5400 seconds
Maximum idle GPU memory: 1024 MiB
Maximum idle GPU utilization: 5 percent
```

The controller polls from the current Mac agent and launches immediately when
one clean A100 is available. It does not start a remote-only monitor that
cannot trigger the next action.

Before upload, require:

- local branch is `feat/kv-sparse-attention`;
- source revision is a full SHA;
- local and `origin/feat/kv-sparse-attention` SHAs agree;
- the worktree has no tracked modifications in source paths;
- Kerberos TTL passes;
- the remote root resolves below the approved root;
- remote Python and draft checkpoint exist;
- exactly one selected GPU is clean; and
- no exact-tag directory or process already exists.

- [ ] **Step 4: Implement source staging and worker launch**

Create a source archive from an explicit allowlist:

```text
tinyvllm/layers/fused_int4_linear.py
tinyvllm/layers/quantization.py
tinyvllm/layers/linear.py
tools/quantized_draft_int4_microgate.py
tools/quantized_draft_int4_microgate_worker.py
tools/assemble_quantized_draft_int4_microgate.py
tools/verify_quantized_draft_int4_microgate.py
```

Stage and extract it only below:

```text
/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/quantized-draft-int4/20260831-quantized-draft-int4-stage0-r1/source
```

Set all task-local cache variables explicitly below:

```text
/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/quantized-draft-int4/20260831-quantized-draft-int4-stage0-r1/cache
```

Launch one worker on the selected GPU. Record PID, PGID, command, environment,
selected GPU UUID, start time, and source SHA.

- [ ] **Step 5: Implement verification and compact download**

The controller must:

1. wait for worker terminal state;
2. assemble the remote final bundle;
3. run the remote independent verifier;
4. create the final manifest;
5. verify the manifest remotely;
6. download only controller receipts and `final_bundle`;
7. run the local independent verifier from the frozen source revision;
8. verify the local manifest;
9. perform three exact-tag remote process scans; and
10. record cleanup without killing unrelated processes.

- [ ] **Step 6: Run controller tests and confirm GREEN**

Run:

```bash
python -m pytest -q \
  tools/test_run_quantized_draft_int4_microgate.py
```

Expected: all tests pass.

- [ ] **Step 7: Commit Task 5**

```bash
git add -- \
  tools/run_quantized_draft_int4_microgate.py \
  tools/test_run_quantized_draft_int4_microgate.py
git -c core.hooksPath=/dev/null commit \
  -m "feat(runtime): automate INT4 draft qualification" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

---

### Task 6: Run the complete local regression gate

**Files:**

- No production-file changes expected.

- [ ] **Step 1: Run the complete Stage-0 suite**

```bash
python -m pytest -q \
  tools/test_fused_int4_linear.py \
  tools/test_quantized_draft_int4_microgate.py \
  tools/test_quantized_draft_int4_microgate_worker.py \
  tools/test_assemble_quantized_draft_int4_microgate.py \
  tools/test_verify_quantized_draft_int4_microgate.py \
  tools/test_run_quantized_draft_int4_microgate.py
```

Expected: all tests pass.

- [ ] **Step 2: Run adjacent quantization regressions**

Enumerate and run the tracked adjacent tests:

```bash
TEST_FILES="$(git ls-files \
  'tools/test_*quant*.py' \
  'tools/test_*linear*.py' |
  tr '\n' ' ')"
test -n "${TEST_FILES}"
python -m pytest -q ${TEST_FILES}
```

Expected: all collected tests pass. Record the exact file and test counts.

- [ ] **Step 3: Run import and diff checks**

```bash
python -m py_compile \
  tinyvllm/layers/fused_int4_linear.py \
  tools/quantized_draft_int4_microgate.py \
  tools/quantized_draft_int4_microgate_worker.py \
  tools/assemble_quantized_draft_int4_microgate.py \
  tools/verify_quantized_draft_int4_microgate.py \
  tools/run_quantized_draft_int4_microgate.py
git diff --check HEAD
```

Expected: both commands exit zero.

- [ ] **Step 4: Push implementation commits**

```bash
git push origin feat/kv-sparse-attention
test "$(git rev-parse HEAD)" = "$(
  git ls-remote origin refs/heads/feat/kv-sparse-attention |
  awk '{print $1}'
)"
```

Expected: push succeeds and the SHA equality check exits zero.

---

### Task 7: Execute the real A100 Stage-0 gate

**Files:**

- Create:
  `artifacts/quantized_draft_int4/20260831-quantized-draft-int4-stage0-r1/controller/`
- Create:
  `artifacts/quantized_draft_int4/20260831-quantized-draft-int4-stage0-r1/final_bundle/`

- [ ] **Step 1: Create a fresh immutable tag**

Use:

```text
20260831-quantized-draft-int4-stage0-r1
```

If any exact-tag local directory, remote directory, or active process exists,
increment only the `rN` suffix. Never resume or overwrite a partial tag.

- [ ] **Step 2: Run dry-run admission**

```bash
RUN_TAG=20260831-quantized-draft-int4-stage0-r1
python tools/run_quantized_draft_int4_microgate.py \
  --run-tag "${RUN_TAG}" \
  --dry-run
```

Expected:

```text
READY
```

If Kerberos TTL or clean-GPU admission is unavailable, start the controller's
local monitor. Do not launch a remote-only monitor and do not ask the user to
watch terminal output.

- [ ] **Step 3: Launch the gate**

```bash
RUN_TAG=20260831-quantized-draft-int4-stage0-r1
python tools/run_quantized_draft_int4_microgate.py \
  --run-tag "${RUN_TAG}" \
  --execute
```

Expected: terminal controller result with one of the frozen classifications.

- [ ] **Step 4: Verify the downloaded bundle independently**

```bash
RUN_TAG=20260831-quantized-draft-int4-stage0-r1
python tools/verify_quantized_draft_int4_microgate.py \
  "artifacts/quantized_draft_int4/${RUN_TAG}/final_bundle"
shasum -a 256 -c \
  "artifacts/quantized_draft_int4/${RUN_TAG}/final_bundle/manifest.sha256"
```

Expected:

- verifier status `PASS`;
- verifier classification equals producer classification; and
- every manifest entry reports `OK`.

- [ ] **Step 5: Apply the stop rule**

If classification is not:

```text
GO_FUSED_INT4_DRAFT_KERNEL
```

stop. Do not create training code, teacher datasets, or Stage-1 claims.

If it is GO, write a separate Stage-1 distillation design and plan before any
training implementation.

---

### Task 8: Write the terminal Stage-0 audit and publish evidence

**Files:**

- Create:
  `docs/superpowers/audits/2026-08-31-quantized-draft-int4-stage0-audit.md`
- Add exact compact files under:
  `artifacts/quantized_draft_int4/20260831-quantized-draft-int4-stage0-r1/controller/`
- Add exact compact files under:
  `artifacts/quantized_draft_int4/20260831-quantized-draft-int4-stage0-r1/final_bundle/`

- [ ] **Step 1: Write the prompt-to-artifact audit**

The audit must map:

| Requirement | Required evidence |
| --- | --- |
| Real packed INT4 consumption | kernel source, dispatch test, allocation row |
| No full dequantized weight | negative unit test and measured memory receipt |
| Real draft shapes | checkpoint fingerprint and shape manifest |
| Median and P99 thresholds | raw rows and independent recomputation |
| Numerical correctness | per-shape max absolute and relative errors |
| Graph behavior | capture and two-replay receipt per shape |
| Benefit and cost | latency, host submission, memory, compile time |
| Clean remote execution | GPU admission, cleanup, three exact-tag scans |
| Independent verification | remote and local verifier receipts |
| Complete compact inventory | passing SHA-256 manifest |
| Claim boundary | exact frozen classification and stop decision |

- [ ] **Step 2: Verify the audit against actual artifacts**

Run:

```bash
RUN_TAG=20260831-quantized-draft-int4-stage0-r1
test -f \
  docs/superpowers/audits/2026-08-31-quantized-draft-int4-stage0-audit.md
python tools/verify_quantized_draft_int4_microgate.py \
  "artifacts/quantized_draft_int4/${RUN_TAG}/final_bundle"
shasum -a 256 -c \
  "artifacts/quantized_draft_int4/${RUN_TAG}/final_bundle/manifest.sha256"
git diff --check -- \
  docs/superpowers/audits/2026-08-31-quantized-draft-int4-stage0-audit.md \
  "artifacts/quantized_draft_int4/${RUN_TAG}/controller" \
  "artifacts/quantized_draft_int4/${RUN_TAG}/final_bundle"
```

Expected: every command exits zero.

- [ ] **Step 3: Stage only terminal compact evidence**

```bash
RUN_TAG=20260831-quantized-draft-int4-stage0-r1
git add -- \
  docs/superpowers/audits/2026-08-31-quantized-draft-int4-stage0-audit.md \
  "artifacts/quantized_draft_int4/${RUN_TAG}/controller" \
  "artifacts/quantized_draft_int4/${RUN_TAG}/final_bundle"
```

Confirm:

```bash
git diff --cached --name-only
git diff --cached --check
```

Expected: only the named audit and compact artifact paths are staged.

- [ ] **Step 4: Commit and push terminal evidence**

```bash
git -c core.hooksPath=/dev/null commit \
  -m "perf(runtime): qualify fused INT4 drafting" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
test "$(git rev-parse HEAD)" = "$(
  git ls-remote origin refs/heads/feat/kv-sparse-attention |
  awk '{print $1}'
)"
```

Expected: commit and push succeed and local/remote SHA equality passes.

## Completion Conditions

Stage 0 is complete only when all of the following are true:

- the six focused test files pass;
- adjacent quantization tests pass;
- all new Python files compile;
- the implementation commits are on the remote branch;
- one fresh real A100 gate reaches a terminal classification;
- producer, remote verifier, and frozen-source local verifier agree;
- the complete compact manifest verifies;
- remote cleanup and three exact-tag process scans are recorded;
- the audit contains a complete prompt-to-artifact checklist;
- terminal evidence is committed and pushed; and
- local and remote branch SHAs match.

`NO_GO_*` is a valid completed Stage-0 result when evidence is complete.
`INCONCLUSIVE_EVIDENCE`, an unavailable GPU, an incomplete verifier, or a
missing manifest is not completion.

## Plan Self-Review

### Spec coverage

| Design requirement | Implementing task |
| --- | --- |
| Model-neutral packed W4A16 primitive | Task 2 |
| Real draft-checkpoint shape inventory | Task 3 |
| No full-weight dequantization | Tasks 2 and 3 |
| Frozen median, P99, memory, numerical, and graph thresholds | Task 1 |
| Paired real A100 measurement | Tasks 3 and 7 |
| Fail-closed remote execution under the approved root | Task 5 |
| Compact local evidence and remote large-artifact retention | Tasks 4, 5, and 8 |
| Independent metric and classification recomputation | Task 4 |
| Checksum-complete final inventory | Tasks 4, 7, and 8 |
| Stop before distillation on Stage-0 failure | Task 7 |
| Benefit and cost reported together | Tasks 3, 4, and 8 |

### Scope check

The plan intentionally ends at the fused-kernel terminal classification. It
does not implement teacher-data generation, quantization-aware distillation,
precision-profile search, speculative runtime integration, or branch-ahead
execution. Those remain separate plans that are authorized only by the
preceding gate.

### Interface consistency

- The worker and assembler consume the `DraftLinearShape` and row schemas
  defined in Task 1.
- The worker is the only consumer of `fused_int4_linear()` in Stage 0.
- The assembler is the producer authority; the verifier does not import its
  classification implementation.
- The controller invokes the worker, assembler, and verifier through their
  documented command-line boundaries.
- No task requires a production dispatch change in
  `tinyvllm/layers/linear.py`.
