# Qwen3.5 TP4 Request-Level E2E64 Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run controlled r620-versus-r631 canonical 64-token request-level benchmarks and produce a parity-gated E2E latency and throughput conclusion.

**Architecture:** A thin local orchestrator dynamically imports the verified r631 attempt launcher, overrides immutable source and attempt identities, strips only the 8-token decode-profile arguments, and validates the resulting 12-command matrix before remote side effects. A separate comparison module validates paired request outputs and aggregates case makespan, request throughput, token throughput, request E2E, TTFT, and decode latency.

**Tech Stack:** Python 3, pytest, JSON/JSONL benchmark artifacts, SSH/SCP, PyTorch TP4/NCCL benchmark workers.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Do not switch branches, stage, commit, stash, reset, push, or run `git clean`.
- Use GPUs `2,4,5,6` only.
- Admission requires at least `26843545600` free bytes and utilization no greater than `10` percent on every selected GPU.
- Treat the run as shared and non-exclusive; allow unrelated low-utilization processes.
- Do not create dummy reservations and do not kill unrelated processes.
- Every real or failed attempt uses a fresh tag and remains preserved.
- Cleanup may match only the exact current attempt tag and descendants.
- Do not modify canonical manifests, case-matrix schemas, existing `profile.json` schemas, or old artifacts.
- Output-token parity failure remains `NO_GO`.
- Do not use decode-step metrics as request-level E2E evidence.
- The explicit no-commit constraint overrides the generic plan template's commit steps.

---

### Task 1: Lock canonical command derivation with tests

**Files:**
- Create: `tools/test_qwen35_tp4_request_e2e64_runner.py`
- Create: `tools/qwen35_tp4_request_e2e64_runner.py`

**Interfaces:**
- Consumes: the verified r631 `launch_w2.py` command template and frozen r620/r631 source specifications.
- Produces: `build_commands(source_spec) -> list[dict]` and `validate_commands(commands, source_spec) -> None`.

- [ ] **Step 1: Write failing tests**

Cover:

```python
def test_build_commands_produces_canonical_64_token_matrix():
    commands = runner.build_commands(runner.SOURCES["r620"])
    assert len(commands) == 12
    for row in commands:
        assert "--profile" in row["argv"]
        assert "--generated-tokens-override" not in row["argv"]
        assert "--decode-internal-profile" not in row["argv"]


def test_validate_commands_rejects_short_output_override():
    commands = runner.build_commands(runner.SOURCES["r631"])
    commands[0]["argv"].extend(["--generated-tokens-override", "8"])
    with pytest.raises(ValueError, match="generated-tokens-override"):
        runner.validate_commands(commands, runner.SOURCES["r631"])
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
python -m pytest -q tools/test_qwen35_tp4_request_e2e64_runner.py
```

Expected: import failure because the runner does not exist.

- [ ] **Step 3: Implement immutable source specs and command derivation**

Define:

```python
@dataclass(frozen=True)
class SourceSpec:
    name: str
    tag: str
    source_tree_sha256: str
    source_tar: Path
    source_tar_sha256: str


def build_commands(source: SourceSpec) -> list[dict]:
    ...


def validate_commands(
    commands: list[dict],
    source: SourceSpec,
) -> None:
    ...
```

Derive commands from the verified launcher and remove exactly the
`--generated-tokens-override 8` pair and the standalone
`--decode-internal-profile` flag.

- [ ] **Step 4: Run the focused test and verify GREEN**

Run:

```bash
python -m pytest -q tools/test_qwen35_tp4_request_e2e64_runner.py
```

Expected: all tests pass.

### Task 2: Implement parity-gated E2E comparison

**Files:**
- Create: `tools/test_qwen35_request_e2e64_comparison.py`
- Create: `tools/qwen35_request_e2e64_comparison.py`

**Interfaces:**
- Consumes: two complete attempt roots containing measured
  `profile.json` and `case_rows.jsonl` files.
- Produces: `compare_attempts(baseline_root, candidate_root) -> dict`.

- [ ] **Step 1: Write synthetic comparison tests**

Tests must cover:

- candidate 10 percent faster for both policies gives
  `E2E_PERFORMANCE_PASS`;
- token mismatch gives `NO_GO`;
- one improved and one unchanged policy gives `MIXED`;
- small changes give `NO_MATERIAL_E2E_CHANGE`;
- a 5 percent or larger regression gives `E2E_REGRESSION`;
- malformed row count, token count, or decode-step count is rejected.

- [ ] **Step 2: Run the comparison tests and verify RED**

Run:

```bash
python -m pytest -q tools/test_qwen35_request_e2e64_comparison.py
```

Expected: import failure because the comparison module does not exist.

- [ ] **Step 3: Implement case parsing and metrics**

For each measured case calculate:

```python
makespan_ns = max(row["e2e_ns"] for row in rows)
request_throughput_rps = 4e9 / makespan_ns
output_token_throughput_tps = 256e9 / makespan_ns
request_decode_ns = sum(row["decode_step_ns"])
```

Aggregate medians and dispersion by source and policy, compare paired
repetitions, and preserve all raw case metrics in the JSON result.

- [ ] **Step 4: Implement hard gates and classification**

Require exact case/request alignment and 64-token output parity. Return
`NO_GO` with explicit reasons for any correctness or completeness failure.
Apply the classification thresholds frozen in the design.

- [ ] **Step 5: Run the focused test and verify GREEN**

Run:

```bash
python -m pytest -q \
  tools/test_qwen35_request_e2e64_comparison.py
```

Expected: all tests pass.

### Task 3: Add attempt execution and static validation

**Files:**
- Modify: `tools/qwen35_tp4_request_e2e64_runner.py`

**Interfaces:**
- Consumes: `SourceSpec` and validated command list.
- Produces: `run_source(source_spec)` plus a CLI with `--source`,
  `--dry-run`, and `--run`.

- [ ] **Step 1: Add dry-run tests**

Verify the serialized dry-run payload contains the attempt tag, source
hashes, 12 commands, canonical output length, GPU policy, and no forbidden
flags.

- [ ] **Step 2: Implement launcher adaptation**

Load the verified attempt launcher with `importlib.util`, override:

```python
module.TAG = source.tag
module.OUTPUT = ROOT / "experiments/qwen35_hybrid_state" / source.tag
module.REMOTE = f"{module.runner.REMOTE_ROOT}/{source.tag}"
module.SOURCE = source.source_tree_sha256
module.SOURCE_TAR = source.source_tar
module.SOURCE_TAR_SHA = source.source_tar_sha256
```

Patch its `build_commands` function to return the already validated canonical
commands, then invoke its existing main execution flow.

- [ ] **Step 3: Enforce pre-side-effect freshness and hash gates**

Before upload, require:

- source tar SHA256 equals the frozen value;
- local output path does not exist;
- remote attempt path does not exist;
- prerequisite and manifest hashes match the frozen values.

- [ ] **Step 4: Run all local runner/comparison tests**

Run:

```bash
python -m pytest -q \
  tools/test_qwen35_tp4_request_e2e64_runner.py \
  tools/test_qwen35_request_e2e64_comparison.py
```

Expected: all tests pass.

- [ ] **Step 5: Produce static dry-run evidence**

Run:

```bash
python tools/qwen35_tp4_request_e2e64_runner.py --source all --dry-run \
  > experiments/qwen35_hybrid_state/qwen35-tp4-request-e2e64-dry-run-20260811.json
```

Inspect the JSON and independently search for forbidden flags.

### Task 4: Execute the r620 and r631 campaigns

**Files:**
- Create: `experiments/qwen35_hybrid_state/qwen35-tp4-request-e2e64-r620-baseline-attempt001/`
- Create: `experiments/qwen35_hybrid_state/qwen35-tp4-request-e2e64-r631-candidate-attempt001/`

**Interfaces:**
- Consumes: validated source tarballs and the existing prerequisites bundle.
- Produces: two preserved 12-worker attempt directories.

- [ ] **Step 1: Check local and remote freshness**

Verify both local attempt directories and both remote paths are absent.

- [ ] **Step 2: Run r620 baseline**

Run without a persistent PTY:

```bash
KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian \
python tools/qwen35_tp4_request_e2e64_runner.py --source r620 --run
```

Poll only the single process/session until completion.

- [ ] **Step 3: Audit r620 completeness and cleanup**

Require `RUN_COMPLETE`, all 12 workers, ten measured profile/row pairs,
`CLEAN` cleanup receipt, and zero exact-tag processes.

- [ ] **Step 4: Run r631 candidate**

Run:

```bash
KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian \
python tools/qwen35_tp4_request_e2e64_runner.py --source r631 --run
```

- [ ] **Step 5: Audit r631 completeness and cleanup**

Apply the same completeness and exact-tag process gates.

### Task 5: Compare, report, and complete the audit

**Files:**
- Create: `experiments/qwen35_hybrid_state/qwen35-tp4-request-e2e64-comparison-20260811/`
- Modify: `AGENT_HANDOFF_STATE.md`

**Interfaces:**
- Consumes: both complete attempt roots.
- Produces: machine-readable comparison, human report, completion audit, and
  updated handoff.

- [ ] **Step 1: Generate the comparison**

Run:

```bash
python tools/qwen35_request_e2e64_comparison.py \
  --baseline experiments/qwen35_hybrid_state/qwen35-tp4-request-e2e64-r620-baseline-attempt001 \
  --candidate experiments/qwen35_hybrid_state/qwen35-tp4-request-e2e64-r631-candidate-attempt001 \
  --output experiments/qwen35_hybrid_state/qwen35-tp4-request-e2e64-comparison-20260811/comparison.json
```

- [ ] **Step 2: Write the human-readable report**

Report, for both policies:

- median case makespan and percent change;
- median request throughput and percent change;
- median output-token throughput and percent change;
- median request E2E, TTFT, and decode latency and percent changes;
- per-repetition paired ratios and dispersion;
- parity and classification.

- [ ] **Step 3: Perform the completion audit**

Create a prompt-to-artifact checklist covering source identity, workload
identity, 64-token shape, GPU admission, run completeness, parity, every
requested metric, cleanup, and exact-tag zero-process checks.

- [ ] **Step 4: Update the handoff**

Append the request-level result, its proof boundary, artifact paths, exact
commands, and any remaining optimization opportunity to
`AGENT_HANDOFF_STATE.md`.

- [ ] **Step 5: Run final verification**

Run focused tests again, validate all JSON files parse, and independently
recompute at least one baseline and candidate makespan/throughput pair from
raw `case_rows.jsonl`.

