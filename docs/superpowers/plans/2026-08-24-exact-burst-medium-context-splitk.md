# Exact-Burst Medium-Context Split-K Graph Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:executing-plans to implement this plan task-by-task. This task
> explicitly prohibits subagents and additional worktrees. Steps use checkbox
> (`- [ ]`) syntax for tracking.

**Goal:** Add one default-disabled split-12 CUDA Graph variant for exact greedy
decode bursts whose complete replay span stays within context lengths
`[1537, 4097]`, then retain it only if source-bound Qwen3-0.6B/A100 evidence
shows useful full-model TPOT benefit without correctness or collateral
regression.

**Architecture:** Preserve the existing auto exact-burst graph and add one
specialized sibling graph. A pure lease-span selector chooses split 12 before
replay; graph identity and capture receipts bind the split value so benchmark
rows can prove which graph executed. The scheduler lease, KV allocation,
sampling, prefill, ordinary decode, and shared FlashAttention heuristic remain
unchanged.

**Tech Stack:** Python 3, PyTorch 2.4.1 CUDA Graphs, FlashAttention 2.6.3,
pytest, Qwen3-0.6B, NVIDIA A100 80GB PCIe, JSON/JSONL evidence, SSH remote
controller

## Global Constraints

- The only authoritative checkout is `/Users/bytedance/Desktop/TinyLLMForge`.
- Do not create a worktree and do not modify
  `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Do not use subagents; execute this plan inline.
- Preserve all unrelated tracked and untracked files.
- Stage only named task files; never use broad `git add`, `git reset`, or
  `git clean`.
- Commit with `git -c core.hooksPath=/dev/null commit`.
- Every commit has exactly one trailer:
  `Co-authored-by: TRAE CLI <noreply@bytedance.com>`.
- Push only to `origin/feat/kv-sparse-attention`.
- Remote runtime data may be written only below
  `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818`.
- Do not run `kinit`, kill external GPU work, or reuse any attempted run tag.
- A GPU is strict-clean only when memory is at most 1024 MiB, utilization is
  at most 5%, and no compute process is attached.
- A GPU gate source SHA must equal the already-pushed branch HEAD.
- Never classify partial rows.
- Report benefit and cost together.
- The design is a runtime-data-flow-specific original engineering design, not
  a claim of academic novelty.
- Preserve exact output token IDs and sampled-logit argmax. Sampled logits
  must satisfy `max_abs <= 0.25` and per-pair `mean_abs <= 0.05`.
- GO requires at least 1% aggregate target-range median TPOT improvement,
  target bucket median/P95 regression no greater than 2%, out-of-range
  median/P95 regression no greater than 1%, TTFT/E2E/throughput regression no
  greater than 2%, no extra KV scratch blocks, at most 8 MiB retained static
  bytes per added production graph, at most 64 MiB added reserved memory, and
  at most 5 seconds added capture duration.

## File Structure

- `tinyvllm/config.py`: owns the default-disabled public configuration and
  dependency validation.
- `tinyvllm/engine/exact_greedy_decode_burst.py`: owns evidence-bound constants,
  the pure lease-span selector, split-bearing graph identity, capture receipt,
  and graph capability.
- `tinyvllm/engine/model_runner.py`: owns graph capture, graph selection,
  replay dispatch, fallback, and invalidation.
- `tools/test_exact_greedy_decode_burst.py`: unit tests for selector, identity,
  receipt, capture context, and compatibility.
- `tools/test_model_runner_spec_verify.py`: integration-contract tests for
  config, graph ownership, capture failure, dispatch, and invalidation.
- `tools/profile_exact_burst_medium_split_k.py`: source-bound worker that
  measures paired control/candidate performance and correctness.
- `tools/exact_burst_medium_split_k_gate.py`: canonical producer, validation,
  aggregation, thresholds, manifest, and classification.
- `tools/exact_burst_medium_split_k_verify.py`: independent reconstruction and
  artifact verification.
- `tools/run_exact_burst_medium_split_k_remote.py`: strict-clean GPU lease,
  source upload, remote preflight/run, remote verify, download, and local
  verify.
- Matching `tools/test_*.py` files own deterministic RED/GREEN contracts for
  all three evidence tools.
- `docs/superpowers/audits/2026-08-24-exact-burst-medium-context-splitk-audit.md`:
  final source/evidence/benefit/cost/prompt-to-artifact audit.
- `AGENT_HANDOFF_STATE.md`: append-only final state and exact continuation
  point.

---

### Task 1: Add the evidence-bound selector and configuration contract

**Files:**

- Modify: `tinyvllm/config.py`
- Modify: `tinyvllm/engine/exact_greedy_decode_burst.py`
- Modify: `tools/test_exact_greedy_decode_burst.py`
- Modify: `tools/test_model_runner_spec_verify.py`

**Interfaces:**

- Produces:

```python
MEDIUM_SPLIT_K_NUM_SPLITS = 12
MEDIUM_SPLIT_K_MIN_CONTEXT_LENGTH = 1537
MEDIUM_SPLIT_K_MAX_CONTEXT_LENGTH = 4097

def exact_greedy_decode_burst_flash_attn_num_splits(
    *,
    enabled: bool,
    initial_sequence_length: int,
    authorized_token_count: int,
) -> int:
    ...
```

- Produces:

```python
Config.exact_greedy_decode_burst_medium_split_k: bool = False
```

- [ ] **Step 1: Write selector boundary tests**

Add a table-driven test to `tools/test_exact_greedy_decode_burst.py`:

```python
def test_medium_split_k_selector_requires_complete_burst_in_range():
    select = exact_greedy_decode_burst_flash_attn_num_splits
    cases = (
        (False, 1537, 8, 0),
        (True, 1536, 1, 0),
        (True, 1537, 1, 12),
        (True, 1537, 8, 12),
        (True, 4090, 8, 12),
        (True, 4091, 8, 0),
        (True, 4097, 1, 12),
        (True, 4098, 1, 0),
    )
    for enabled, initial, count, expected in cases:
        assert select(
            enabled=enabled,
            initial_sequence_length=initial,
            authorized_token_count=count,
        ) == expected
```

Also assert that non-boolean `enabled`, non-positive lengths, and
non-positive token counts raise `ValueError`.

- [ ] **Step 2: Write config RED tests**

Extend `test_exact_greedy_decode_burst_config_is_strict_and_default_off()`:

```python
assert config.exact_greedy_decode_burst_medium_split_k is False
with pytest.raises(ValueError, match="must be a bool"):
    make_config(exact_greedy_decode_burst_medium_split_k=1)
with pytest.raises(
    ValueError,
    match="medium split-k requires exact_greedy_decode_burst",
):
    make_config(exact_greedy_decode_burst_medium_split_k=True)
```

Add a valid case with both flags enabled.

- [ ] **Step 3: Run RED tests**

Run:

```bash
python -m pytest \
  tools/test_exact_greedy_decode_burst.py \
  tools/test_model_runner_spec_verify.py \
  -k 'medium_split_k or exact_greedy_decode_burst_config' -q
```

Expected: failures because the selector, constants, and config field do not
exist.

- [ ] **Step 4: Implement the selector and config**

In `tinyvllm/engine/exact_greedy_decode_burst.py`, validate all inputs and
return 12 only when the full replay span is in range:

```python
def exact_greedy_decode_burst_flash_attn_num_splits(
    *,
    enabled: bool,
    initial_sequence_length: int,
    authorized_token_count: int,
) -> int:
    _require_bool(enabled, "enabled")
    _require_positive_int(
        initial_sequence_length,
        "initial_sequence_length",
    )
    _require_positive_int(
        authorized_token_count,
        "authorized_token_count",
    )
    final_context_length = (
        initial_sequence_length + authorized_token_count - 1
    )
    if (
        enabled
        and initial_sequence_length
        >= MEDIUM_SPLIT_K_MIN_CONTEXT_LENGTH
        and final_context_length
        <= MEDIUM_SPLIT_K_MAX_CONTEXT_LENGTH
    ):
        return MEDIUM_SPLIT_K_NUM_SPLITS
    return 0
```

Add the config field and fail-closed validation in `Config.__post_init__`.

- [ ] **Step 5: Run GREEN tests**

Run the command from Step 3. Expected: all selected tests pass.

- [ ] **Step 6: Commit and push**

Stage exactly the four task files, inspect the cached diff, commit:

```text
feat(runtime): select medium-context burst split-k
```

with the required trailer, then push the current branch.

---

### Task 2: Bind split-K to graph identity, receipt, and capture context

**Files:**

- Modify: `tinyvllm/engine/exact_greedy_decode_burst.py`
- Modify: `tools/test_exact_greedy_decode_burst.py`

**Interfaces:**

- Extends:

```python
ExactGreedyDecodeBurstCaptureReceipt.flash_attn_num_splits: int
```

- Extends:

```python
ExactGreedyDecodeBurstGraph.capture(
    ...,
    flash_attn_num_splits: int = 0,
) -> ExactGreedyDecodeBurstGraph
```

- `set_decode_context` receives `flash_attn_num_splits`.

- [ ] **Step 1: Write RED identity and receipt tests**

Create two graph fixtures with identical tensors and split values 0 and 12.
Assert:

```python
assert auto.receipt.flash_attn_num_splits == 0
assert split.receipt.flash_attn_num_splits == 12
assert (
    auto.receipt.graph_identity_sha256
    != split.receipt.graph_identity_sha256
)
```

Assert both warmup and capture context calls contain the selected split:

```python
assert context_calls == [
    {"flash_attn_num_splits": 12, ...},
    {"flash_attn_num_splits": 12, ...},
]
```

Assert `-1`, `True`, and non-integers are rejected.

- [ ] **Step 2: Run RED tests**

```bash
python -m pytest tools/test_exact_greedy_decode_burst.py \
  -k 'split_k or flash_attn_num_splits or capture_orders' -q
```

Expected: failures for the missing receipt field and capture parameter.

- [ ] **Step 3: Implement identity and receipt binding**

Add the normalized integer to `identity_payload`, the receipt dataclass, the
capture receipt constructor, `capability()`, and `set_decode_context()` calls.
Keep the default `0` so existing callers preserve behavior.

- [ ] **Step 4: Update existing fixtures explicitly**

Every direct `ExactGreedyDecodeBurstCaptureReceipt(...)` fixture includes:

```python
flash_attn_num_splits=0
```

Every fixture callback accepts and records the new keyword. Do not weaken
assertions with unrestricted `**kwargs`.

- [ ] **Step 5: Run focused and full module tests**

```bash
python -m pytest tools/test_exact_greedy_decode_burst.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit and push**

Stage exactly the runtime module and its test, commit:

```text
feat(runtime): bind burst split-k graph identity
```

with the required trailer, then push.

---

### Task 3: Capture and own auto plus split-12 graph variants

**Files:**

- Modify: `tinyvllm/engine/model_runner.py`
- Modify: `tools/test_model_runner_spec_verify.py`

**Interfaces:**

- Produces ModelRunner attributes:

```python
self.exact_greedy_decode_burst_medium_split_k_graph
self.exact_greedy_decode_burst_medium_split_k_correctness_graph
```

- Extends:

```python
ModelRunner._capture_exact_greedy_decode_burst(
    *,
    correctness_trace: bool = False,
    sampled_logit_ordinals: tuple[int, ...] = (),
    flash_attn_num_splits: int = 0,
)
```

- [ ] **Step 1: Write RED ownership and capture tests**

Extend the capture fixture to return markers keyed by split:

```python
markers = {0: object(), 12: object()}

class FakeBurstGraph:
    @classmethod
    def capture(cls, **kwargs):
        return markers[kwargs["flash_attn_num_splits"]]
```

Assert auto capture fills the existing attribute, enabled specialized capture
fills the new attribute, each call passes the exact split to
`set_context(..., flash_attn_num_splits=value)`, and both variants retain the
same one scratch block.

- [ ] **Step 2: Write specialized-capture failure test**

Make the fake capture raise only for split 12. Assert the auto marker remains
installed, the specialized attribute is `None`, and the stats contain:

```python
fallback_counts["medium_split_k_capture_unavailable"] == 1
```

- [ ] **Step 3: Run RED tests**

```bash
python -m pytest tools/test_model_runner_spec_verify.py \
  -k 'medium_split_k and capture' -q
```

Expected: failures because the new graph attributes and capture path do not
exist.

- [ ] **Step 4: Implement graph ownership and capture**

Initialize both new attributes to `None`. Refactor capture so each invocation
has an explicit target attribute and split value. Capture auto first; if the
feature is enabled, capture split 12 second without clearing auto on failure.
Pass:

```python
set_context(
    False,
    slot_mapping=slot_mapping,
    context_lens=context_length,
    block_tables=block_table,
    flash_attn_num_splits=flash_attn_num_splits,
)
```

Do not allocate another scratch KV block.

- [ ] **Step 5: Run focused tests**

```bash
python -m pytest tools/test_model_runner_spec_verify.py \
  -k 'exact_burst and (capture or scratch)' -q
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit and push**

Stage exactly the two files, commit:

```text
feat(runtime): capture medium split-k burst graph
```

with the required trailer, then push.

---

### Task 4: Dispatch the lease to the correct graph variant

**Files:**

- Modify: `tinyvllm/engine/model_runner.py`
- Modify: `tools/test_model_runner_spec_verify.py`

**Interfaces:**

- Produces:

```python
ModelRunner._select_exact_greedy_decode_burst_graph(
    lease: ExactGreedyDecodeBurstLease,
    *,
    correctness_trace: bool,
)
```

- The selected graph supplies capability, block-table width, graph identity,
  and replay.

- [ ] **Step 1: Write RED dispatch matrix tests**

Use four fake graphs with distinct identity hashes. Verify:

```text
disabled feature, in range, production       -> production auto
enabled feature, in range, production        -> production split12
enabled feature, in range, correctness       -> correctness split12
enabled feature, below range                 -> corresponding auto
enabled feature, upper-bound crossing        -> corresponding auto
enabled feature, split graph missing         -> corresponding auto
```

Assert replay receives the identity of the selected graph, not the auto
identity.

- [ ] **Step 2: Write RED invalidation test**

Install four distinct fake graphs plus one duplicated alias and assert
`invalidate_exact_greedy_decode_burst_continuation()` visits each object
exactly once.

- [ ] **Step 3: Run RED tests**

```bash
python -m pytest tools/test_model_runner_spec_verify.py \
  -k 'medium_split_k or invalidates' -q
```

Expected: dispatch and invalidation tests fail.

- [ ] **Step 4: Implement selection and replay integration**

Call the pure selector with:

```python
flash_attn_num_splits = (
    exact_greedy_decode_burst_flash_attn_num_splits(
        enabled=(
            self.config
            .exact_greedy_decode_burst_medium_split_k
        ),
        initial_sequence_length=lease.initial_sequence_length,
        authorized_token_count=lease.authorized_token_count,
    )
)
```

Choose split 12 only when its graph exists. Then derive capability,
block-table width, and expected identity from that graph. Extend invalidation
to all graph attributes with identity-based deduplication.

- [ ] **Step 5: Run the complete CPU contract suite**

```bash
python -m pytest \
  tools/test_exact_greedy_decode_burst.py \
  tools/test_model_runner_spec_verify.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit and push**

Stage exactly the two files, commit:

```text
feat(runtime): dispatch medium split-k burst graph
```

with the required trailer, then push.

---

### Task 5: Build the source-bound paired profile worker

**Files:**

- Create: `tools/profile_exact_burst_medium_split_k.py`
- Create: `tools/test_profile_exact_burst_medium_split_k.py`

**Interfaces:**

- Policies:

```python
POLICIES = {
    "auto": {
        "exact_greedy_decode_burst_medium_split_k": False,
    },
    "split12": {
        "exact_greedy_decode_burst_medium_split_k": True,
    },
}
```

- Contexts:

```python
CONTEXT_LENGTHS = (1025, 1537, 2049, 2561, 3073, 3585, 4090, 6145)
```

`4090` is the largest listed start for which a K8 burst remains within 4097.

- [ ] **Step 1: Write RED manifest and row-validation tests**

Test exact policy order, context inventory, five repetitions, two warmups,
K8, deterministic prompt digests, raw TPOT inventory, source SHA, graph
identity, capture receipts, memory fields, and output IDs.

Reject unknown policies, duplicate identities, missing rows, non-finite
numbers, malformed hashes, and a candidate medium row whose graph identity
does not resolve to split 12.

- [ ] **Step 2: Run RED tests**

```bash
python -m pytest \
  tools/test_profile_exact_burst_medium_split_k.py -q
```

Expected: import failure because the profile module does not exist.

- [ ] **Step 3: Implement the profile worker**

Reuse narrow public helpers from `profile_exact_greedy_decode_burst.py` for
prompt construction, request execution, percentile calculation, and
correctness sidecars. Do not copy scheduler or runtime logic into the tool.

Construct both arms with identical parameters except the new flag. Each
performance row must include:

```text
policy, repetition, order_position, context_length, prompt_sha256
output_token_ids, raw TPOT samples, median/P95/P99, TTFT, E2E, throughput
allocated/reserved/peak memory
capture receipts and replay graph identity
exact-burst lifecycle counters
```

Correctness rows sample prefill-final, decode-first, decode-middle, and
decode-final logits and include sidecar hashes.

- [ ] **Step 4: Run GREEN tests**

Run the command from Step 2. Expected: all tests pass.

- [ ] **Step 5: Commit and push**

Stage exactly the profile and test, commit:

```text
feat(bench): profile medium split-k burst graph
```

with the required trailer, then push.

---

### Task 6: Build the canonical gate and independent verifier

**Files:**

- Create: `tools/exact_burst_medium_split_k_gate.py`
- Create: `tools/test_exact_burst_medium_split_k_gate.py`
- Create: `tools/exact_burst_medium_split_k_verify.py`
- Create: `tools/test_exact_burst_medium_split_k_verify.py`

**Interfaces:**

- Producer classification:

```python
GO_EXACT_BURST_MEDIUM_SPLIT_K
NO_GO_PERFORMANCE
NO_GO_CORRECTNESS
NO_GO_GRAPH_SELECTION
NO_GO_LIFECYCLE
NO_GO_MEMORY
NO_GO_CAPTURE_COST
NO_GO_EVIDENCE_INCOMPLETE
```

- Independent verifier:

```python
def verify_artifact_directory(path: Path) -> dict:
    ...
```

- [ ] **Step 1: Write gate RED tests**

Create a complete synthetic GO fixture and one mutation test per threshold:
token mismatch, argmax mismatch, max/mean logit limit, wrong split mapping,
wrong auto mapping, lifecycle mismatch, target median/P95 regression,
out-of-range regression, TTFT/E2E/throughput regression, extra scratch block,
retained/reserved/capture cap, missing row, duplicate row, source mismatch,
and altered raw samples with stale summary.

- [ ] **Step 2: Run gate RED tests**

```bash
python -m pytest \
  tools/test_exact_burst_medium_split_k_gate.py -q
```

Expected: import failure.

- [ ] **Step 3: Implement producer aggregation**

Validate raw rows first, reconstruct nearest-rank percentiles, compare paired
arms by context and repetition, resolve graph hashes against receipts, and
write:

```text
performance_rows.jsonl
correctness_rows.jsonl
comparison.json
summary.json
manifest.json
report.md
```

The manifest binds SHA-256 for every artifact and source file. Classification
must be derived only after exact row-count completeness is proven.

- [ ] **Step 4: Run gate GREEN tests**

Run Step 2. Expected: all tests pass.

- [ ] **Step 5: Write independent-verifier RED tests**

The verifier must independently load raw rows and sidecars, reconstruct every
metric/classification, verify the manifest, and reject tampering. It must not
import producer aggregation or classification functions.

- [ ] **Step 6: Implement and run the verifier**

```bash
python -m pytest \
  tools/test_exact_burst_medium_split_k_verify.py -q
```

Expected: all tests pass, including raw-row, sidecar, summary, manifest, and
classification mutation cases.

- [ ] **Step 7: Commit and push**

Stage exactly the four files, commit:

```text
feat(bench): gate medium split-k burst graph
```

with the required trailer, then push.

---

### Task 7: Add the remote strict-clean controller

**Files:**

- Create: `tools/run_exact_burst_medium_split_k_remote.py`
- Create: `tools/test_run_exact_burst_medium_split_k_remote.py`

**Interfaces:**

- Remote root:

```text
/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/
exact-burst-medium-split-k
```

- Required flags:

```text
--source-commit
--run-tag
--model
--local-destination
```

- [ ] **Step 1: Write controller RED tests**

Test remote-root allowlisting, run-tag uniqueness, deterministic isolated
distributed port, strict-clean GPU parsing, second cleanliness check
immediately before launch, runtime environment rooted only under the allowed
remote path, source archive/hash verification, worker PID/PGID receipt,
terminal exit-code polling, remote verifier, terminal bundle download, local
verifier, and rejection of partial evidence.

- [ ] **Step 2: Run RED tests**

```bash
python -m pytest \
  tools/test_run_exact_burst_medium_split_k_remote.py -q
```

Expected: import failure.

- [ ] **Step 3: Implement the controller**

Follow the established delta-journal controller pattern but use the new
source file allowlist and gate command. Set `TMPDIR`, `TMP`, `TEMP`,
`PYTHONPYCACHEPREFIX`, `XDG_CACHE_HOME`, `HF_HOME`, and
`TORCH_EXTENSIONS_DIR` beneath the run staging directory. Never write remote
runtime state to `/`, `/tmp`, or the old checkout.

- [ ] **Step 4: Run GREEN tests**

Run Step 2. Expected: all tests pass.

- [ ] **Step 5: Run the complete local feature suite**

```bash
python -m pytest \
  tools/test_exact_greedy_decode_burst.py \
  tools/test_model_runner_spec_verify.py \
  tools/test_profile_exact_burst_medium_split_k.py \
  tools/test_exact_burst_medium_split_k_gate.py \
  tools/test_exact_burst_medium_split_k_verify.py \
  tools/test_run_exact_burst_medium_split_k_remote.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit and push**

Stage exactly the controller and its test, commit:

```text
feat(bench): run medium split-k gate remotely
```

with the required trailer, then push.

---

### Task 8: Run the source-bound GPU microgate

**Files:**

- No tracked source edits before classification.
- Remote artifact: a fresh unique tag under the allowed remote root.
- Local downloaded artifact:
  `artifacts/exact_burst_medium_split_k/<fresh-tag>/`

**Interfaces:**

- Consumes the already-pushed `origin/feat/kv-sparse-attention` SHA.
- Produces a complete paired microgate bundle and two verifier receipts.

- [ ] **Step 1: Verify source and credentials**

Run:

```bash
git status --short --branch
git rev-parse HEAD
git rev-parse origin/feat/kv-sparse-attention
klist -c FILE:/Users/bytedance/krb5cc_sitian
```

Require equal local/remote SHAs and a valid ticket. Do not run `kinit`.

- [ ] **Step 2: Check strict-clean GPU state**

Use the controller's parser against `nvidia-smi` output. Wait without killing
external processes until one GPU satisfies all three strict-clean conditions.

- [ ] **Step 3: Launch a fresh microgate tag**

Use one warmup and at least three alternating pairs over:

```text
1025,1537,2049,2561,3073,3585,4090,6145
```

with K8 and enough generated tokens to produce stable TPOT samples.

- [ ] **Step 4: Verify complete evidence**

Require worker exit code zero, complete expected row counts, remote verifier
success, downloaded hashes, and independent local verifier success.

- [ ] **Step 5: Apply the microgate decision**

If any correctness, selection, lifecycle, cost, or performance threshold
fails, record the exact NO-GO, revert runtime code in a dedicated commit,
preserve the benchmark tools/artifacts, push, and stop the candidate.

If all microgate thresholds pass, continue to Task 9.

---

### Task 9: Run the canonical paired gate and publish the audit

**Files:**

- Create:
  `docs/superpowers/audits/2026-08-24-exact-burst-medium-context-splitk-audit.md`
- Modify: `AGENT_HANDOFF_STATE.md`
- Add only the compact final evidence bundle under:
  `artifacts/exact_burst_medium_split_k/<canonical-tag>/`

**Interfaces:**

- Produces final GO/NO-GO classification, benefit/cost table, dual-verifier
  receipts, and prompt-to-artifact checklist.

- [ ] **Step 1: Launch a fresh canonical tag**

Use five repetitions, two warmups, alternating/reversed arm order, all target
and control contexts, K8, and the already-pushed source SHA. Never reuse the
microgate tag.

- [ ] **Step 2: Wait for terminal state while doing local audit work**

Poll the existing controller session rather than opening redundant PTYs.
While the worker runs, draft the source-boundary and requirement checklist.
Do not classify until every expected performance and correctness row exists.

- [ ] **Step 3: Run both verifiers**

Require the remote verifier and the independently executed local verifier to
reconstruct the same classification and metrics from raw evidence.

- [ ] **Step 4: Write the audit**

The audit restates the objective as concrete deliverables and maps each
requirement to:

```text
source file and line
unit/contract test
GPU raw row or sidecar
manifest hash
remote verifier receipt
local verifier receipt
final classification
```

It explicitly reports target TPOT median/P95/P99, controls, TTFT, E2E,
throughput, logit differences, capture duration, allocated/reserved memory,
retained bytes, scratch/KV capacity, and any residual limitation.

- [ ] **Step 5: Append handoff state**

Append the exact source SHA, artifact paths, classification, measured benefit
and cost, tests, commits, remote state, and next recommended experiment to
`AGENT_HANDOFF_STATE.md`. Do not rewrite prior history.

- [ ] **Step 6: Run final verification**

Run all feature tests, `git diff --check`, both artifact verifiers, exact
manifest hash checks, branch SHA checks, and a focused inspection proving the
feature-off path remains auto-only.

- [ ] **Step 7: Commit and push exact final files**

Stage only the audit, handoff append, compact evidence bundle, and any final
test corrections. Commit with a classification-specific message:

```text
perf(runtime): validate medium split-k burst graph
```

or:

```text
docs(runtime): record medium split-k burst no-go
```

Include exactly the required trailer and push
`origin/feat/kv-sparse-attention`.

- [ ] **Step 8: Fresh completion audit**

Re-read the user objective, design, plan, final diff, test outputs, raw GPU
rows, manifest, both verifier receipts, final audit, handoff, local HEAD, and
remote branch SHA. Treat every uncertainty or uncovered requirement as
incomplete and continue working until resolved.
