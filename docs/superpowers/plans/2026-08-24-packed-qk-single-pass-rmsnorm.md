# Packed-QK Single-Pass RMSNorm Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. This repository's active user constraint forbids subagents and additional worktrees, so execute inline in the authoritative checkout with the executing-plans workflow.

**Goal:** Replace Qwen3's two per-layer Q/K RMSNorm dispatches with one
default-off packed-QK dispatch, then prove exactness and end-to-end benefit on
Qwen3-0.6B.

**Architecture:** Preserve the packed QKV projection and treat its contiguous
Q+K prefix as `[tokens, q_heads + kv_heads, head_dim]`. A compiled helper
performs the same FP32 per-head reduction and BF16 cast boundary once, while
selecting the original Q or K learned weight by head index; the result is
split into the unchanged Q and K RoPE inputs.

**Tech Stack:** Python 3, PyTorch, `torch.compile`, CUDA Graphs, pytest,
JSON/JSONL benchmark artifacts, source-bound SSH controller, Qwen3-0.6B on one
clean A100.

## Global Constraints

- The only authoritative checkout is `/Users/bytedance/Desktop/TinyLLMForge`.
- Stay on `feat/kv-sparse-attention`; do not create worktrees or use
  subagents.
- Preserve unrelated dirty and untracked files. Stage only exact task paths.
- Use `git -c core.hooksPath=/dev/null commit`.
- Every commit has exactly one
  `Co-authored-by: TRAE CLI <noreply@bytedance.com>` trailer.
- Push only to `origin/feat/kv-sparse-attention`.
- The feature is default-disabled.
- Preserve per-head reduction over exactly `head_dim`, separate Q/K learned
  weights, FP32 RMS computation, BF16 cast-before-weight behavior, and the
  existing RoPE/attention path.
- Do not materialize an activation-sized Q/K concatenation.
- Do not fuse RMSNorm with RoPE in Stage 1.
- The canonical gate requires 60 performance rows and 24 correctness rows.
- Sampled-logit maximum absolute difference must equal `0.0`.
- Remote task data may be written only below
  `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818`.
- Strict-clean GPU admission is memory `<=1024 MiB`, utilization `<=5%`, and
  no compute process.
- Do not run `kinit` or terminate external GPU processes.
- Never reuse an attempted run tag.
- Remote source SHA must equal the pushed branch HEAD at launch.
- Report both benefit and cost; do not classify partial evidence.

---

### Task 1: Add the Default-Off Configuration and Construction Contract

**Files:**
- Modify: `tinyvllm/config.py`
- Modify: `tinyvllm/engine/model_runner.py`
- Modify: `tinyvllm/models/qwen3.py`
- Modify: `tools/test_model_runner_spec_verify.py`
- Create: `tools/test_packed_qk_single_pass_rmsnorm.py`

**Interfaces:**
- Produces:
  `Config.packed_qk_single_pass_rmsnorm: bool = False`
- Produces:
  `Qwen3ForCausalLM(config, *, packed_qk_single_pass_rmsnorm=False)`
- Passes the same keyword through `Qwen3Model`, `Qwen3DecoderLayer`, and
  `QWen3Attention`

- [ ] **Step 1: Write RED configuration tests**

Add to the exact runtime configuration tests:

```python
fields = Config.__dataclass_fields__
assert fields["packed_qk_single_pass_rmsnorm"].default is False
for invalid in (None, 0, 1, "true"):
    with pytest.raises(
        ValueError,
        match="^packed_qk_single_pass_rmsnorm must be a bool$",
    ):
        Config(
            model=model,
            packed_qk_single_pass_rmsnorm=invalid,
        )
```

- [ ] **Step 2: Write RED construction-propagation tests**

Use dependency-light AST and constructor fakes to require
`model_runner.load_legacy_model()` to pass:

```python
Qwen3ForCausalLM(
    runner_config.hf_config,
    packed_qk_single_pass_rmsnorm=(
        runner_config.packed_qk_single_pass_rmsnorm
    ),
)
```

Require every Qwen3 constructor layer to expose and forward the same keyword.

- [ ] **Step 3: Run RED tests**

Run:

```bash
python3 -m pytest -q \
  tools/test_model_runner_spec_verify.py \
  tools/test_packed_qk_single_pass_rmsnorm.py
```

Expected: FAIL because the config field and constructor keyword do not exist.

- [ ] **Step 4: Implement minimal configuration and propagation**

Add to `Config`:

```python
packed_qk_single_pass_rmsnorm: bool = False
```

Add strict validation:

```python
if not isinstance(self.packed_qk_single_pass_rmsnorm, bool):
    raise ValueError(
        "packed_qk_single_pass_rmsnorm must be a bool"
    )
```

Pass the normalized boolean through model construction without changing any
other model family.

- [ ] **Step 5: Run GREEN tests**

Run the Task 1 command. Expected: PASS.

- [ ] **Step 6: Commit and push**

Stage only the five Task 1 paths and commit:

```text
feat(runtime): configure packed QK RMSNorm
```

---

### Task 2: Implement the Exact Packed-QK Kernel Path

**Files:**
- Modify: `tinyvllm/models/qwen3.py`
- Modify: `tools/test_packed_qk_single_pass_rmsnorm.py`

**Interfaces:**
- Produces:
  `QWen3Attention._packed_qk_rmsnorm(packed_qk) -> Tensor`
- Produces:
  `QWen3Attention.packed_qk_single_pass_rmsnorm_receipt() -> dict`
- Consumes the existing `q_norm.weight`, `k_norm.weight`, `head_dim`,
  `num_heads`, `num_kv_heads`, and RMS epsilon.

- [ ] **Step 1: Write RED structural and routing tests**

Require:

```python
assert attention.packed_qk_single_pass_rmsnorm is False
assert attention.packed_qk_single_pass_rmsnorm_receipt() == {
    "packed_qk_single_pass_rmsnorm_enabled": False,
    "q_heads": attention.num_heads,
    "kv_heads": attention.num_kv_heads,
    "head_dim": attention.head_dim,
}
```

With a fake packed tensor, verify the disabled route calls `q_norm` and
`k_norm` once each, while the enabled route calls `_packed_qk_rmsnorm` once
and does not call either module's `forward`.

- [ ] **Step 2: Write RED numerical tests**

On a torch-capable environment, use fixed BF16 inputs with distinct Q and K
weights. Compare:

```python
expected_q = attention.q_norm(q_view)
expected_k = attention.k_norm(k_view)
actual_q, actual_k = attention._normalize_qk(qkv)
assert torch.equal(actual_q, expected_q)
assert torch.equal(actual_k, expected_k)
```

Cover token counts `1`, `4`, and `17`, Q/K heads `(16, 8)` and `(8, 8)`,
and nontrivial learned weights.

- [ ] **Step 3: Run RED tests**

Run:

```bash
python3 -m pytest -q tools/test_packed_qk_single_pass_rmsnorm.py
```

Expected: FAIL because the helper and routing do not exist.

- [ ] **Step 4: Implement the compiled helper**

Implement a `@compile_if_enabled(dynamic=True)` method whose operation order
is:

```python
origin_dtype = packed_qk.dtype
normalized = packed_qk.view(
    packed_qk.size(0),
    self.num_heads + self.num_kv_heads,
    self.head_dim,
).to(torch.float32)
variance = normalized.pow(2).mean(dim=-1, keepdim=True)
normalized.mul_(torch.rsqrt(variance + self.q_norm.eps))
weights = torch.cat((
    self.q_norm.weight.expand(self.num_heads, -1),
    self.k_norm.weight.expand(self.num_kv_heads, -1),
), dim=0)
normalized = normalized.to(origin_dtype).mul_(weights)
return normalized.view(packed_qk.shape)
```

Do not register expanded weights or allocate a persistent activation-sized
buffer.

- [ ] **Step 5: Integrate without changing the disabled path**

In `forward()`, retain the current code byte-for-byte in the disabled branch.
In the enabled branch, slice the contiguous Q+K prefix from `qkv`, normalize
it once, split it into Q and K, and continue through the existing RoPE call.

- [ ] **Step 6: Run GREEN and adjacent tests**

Run:

```bash
python3 -m pytest -q \
  tools/test_packed_qk_single_pass_rmsnorm.py \
  tools/test_model_runner_spec_verify.py \
  tools/test_exact_greedy_decode_burst.py \
  tools/test_llm_engine_exact_greedy_decode_burst.py
```

Expected: all pass.

- [ ] **Step 7: Run a source-bound remote numerical microgate**

On one strict-clean A100, verify BF16 bitwise equality and compare CUDA Graph
time for 28 layers. Record:

```text
exact
max_abs_diff
baseline_us
candidate_us
improvement_pct
```

Reject the implementation if `exact` is false or `max_abs_diff != 0.0`.

- [ ] **Step 8: Commit and push**

Stage only `tinyvllm/models/qwen3.py` and the focused test, then commit:

```text
feat(runtime): normalize packed QK in one pass
```

---

### Task 3: Add the Paired Hardware Gate and Independent Verifier

**Files:**
- Create: `tools/packed_qk_single_pass_rmsnorm_gate.py`
- Create: `tools/test_packed_qk_single_pass_rmsnorm_gate.py`
- Create: `tools/packed_qk_single_pass_rmsnorm_verify.py`
- Create: `tools/test_packed_qk_single_pass_rmsnorm_verify.py`
- Create: `tools/run_packed_qk_single_pass_rmsnorm_remote.py`
- Create: `tools/test_run_packed_qk_single_pass_rmsnorm_remote.py`

**Interfaces:**
- Produces 60 performance rows:
  `10 repetitions x 3 contexts x 2 modes`
- Produces 24 correctness rows:
  `3 contexts x 2 modes x 4 sampling points`
- Produces classifications:
  `GO_PACKED_QK_SINGLE_PASS_RMSNORM`,
  `NO_GO_PACKED_QK_RMSNORM_CORRECTNESS`,
  `NO_GO_PACKED_QK_RMSNORM_PERFORMANCE`, or
  `NO_GO_PACKED_QK_RMSNORM_EVIDENCE_INCOMPLETE`

- [ ] **Step 1: Write RED gate-contract tests**

Require fixed row inventories, interleaved paired order, source/run identity,
mode receipts, exact output/logit checks, execution-inventory parity, and all
threshold boundaries from the design.

- [ ] **Step 2: Implement deterministic gate summarization**

Reuse the existing exact-burst gate's prompt generation, sampling points,
JSON writers, nearest-rank percentile convention, and paired aggregation.
Change only the compared feature flag and classification fields.

- [ ] **Step 3: Write RED verifier tamper tests**

Require rejection of:

- missing/extra performance or correctness rows;
- duplicate pair keys;
- mismatched prompt digests or order positions;
- mixed source SHA or run tag;
- non-empty source patch;
- artifact hash mismatch;
- summary recomputation mismatch; and
- non-finite numeric fields.

- [ ] **Step 4: Implement the independent verifier**

Parse raw JSONL independently, recompute the summary, validate SHA256
manifests, and emit one deterministic verification JSON receipt.

- [ ] **Step 5: Write RED controller tests**

Require:

- mounted-root-only runtime directories;
- exact pushed HEAD;
- strict-clean GPU admission;
- Kerberos fail-fast with minimum lifetime `5400`;
- run-tag-derived isolated distributed port;
- source-bound preflight and worker;
- no process-kill command;
- remote verification before download;
- local re-verification after download; and
- preservation of partial artifacts on failure.

- [ ] **Step 6: Implement the local controller**

Follow
`tools/run_exact_burst_lease_local_delta_journal_remote.py`, including the
fixed port-isolation behavior from commit `c685b9b`, but use a distinct remote
task root:

```text
.../packed-qk-single-pass-rmsnorm
```

- [ ] **Step 7: Run the complete local gate suite**

Run:

```bash
python3 -m pytest -q \
  tools/test_packed_qk_single_pass_rmsnorm.py \
  tools/test_packed_qk_single_pass_rmsnorm_gate.py \
  tools/test_packed_qk_single_pass_rmsnorm_verify.py \
  tools/test_run_packed_qk_single_pass_rmsnorm_remote.py
```

Expected: all pass.

- [ ] **Step 8: Commit and push**

Stage only the six Task 3 files and commit:

```text
test(runtime): gate packed QK RMSNorm
```

---

### Task 4: Run the Canonical Gate and Reconcile the Result

**Files:**
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify:
  `docs/superpowers/audits/2026-08-16-phase1-completion-audit.md`

- [ ] **Step 1: Verify launch preconditions**

Confirm exact local/pushed SHA equality, empty task-tracked diff, Kerberos
lifetime at least 5400 seconds, absent fresh tag paths, and one strict-clean
GPU.

- [ ] **Step 2: Launch through the local controller**

Use a fresh tag and allow the controller to monitor, launch, verify, and
download automatically. Never classify partial rows.

- [ ] **Step 3: Run both verifiers**

Require remote and local receipts to agree on:

```text
performance_row_count == 60
correctness_row_count == 24
verified == true
classification
```

- [ ] **Step 4: Recompute benefit and cost**

Report TPOT median/P95, TTFT, E2E, throughput, reserved memory, compile/capture
cost, and launch-inventory evidence. State the exactness and forward/replay/D2H
parity separately.

- [ ] **Step 5: Update audit and handoff**

Record the source SHA, run tag, remote/local artifact paths, verifier hashes,
classification, benefit, cost, and claim boundary.

- [ ] **Step 6: Run final verification**

Run the focused suite, adjacent exact-burst regression, `py_compile`,
`git diff --check`, and both artifact verifiers.

- [ ] **Step 7: Commit and push**

Stage only the two documentation files and commit:

```text
docs(runtime): reconcile packed QK RMSNorm gate
```
