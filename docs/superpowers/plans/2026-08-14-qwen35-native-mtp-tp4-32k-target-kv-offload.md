# Qwen3.5 Native MTP TP4 32K Target-KV Offload Implementation Plan

> **For agentic workers:** Execute inline in the current session. Do not use
> subagents for this plan.

**Goal:** Establish an independent, source-bound Qwen3.5 native-MTP TP4/32K
production-Engine correctness authority with exact greedy parity and real
target-KV offload.

**Architecture:** Build a thin 32K authority overlay on the frozen, passing
native-MTP TP4/16K authority. Override only the frozen 32K constants, source
inventory, worker/verifier defaults, and the stronger b1/b4 movement contract.
Reuse the production runtime unchanged unless a focused RED test proves a
specific missing production connection.

**Tech Stack:** Python 3, pytest, Bash, PyTorch distributed TP4, TinyLLMForge
`LLMEngine`, native Qwen3.5 MTP executor, production `KVOffloadMVP0`, SSH,
Kerberos, rsync, JSON/SHA-256 source manifests.

## Global Constraints

- Work only in `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Do not stage, commit, push, switch branches/worktrees, stash, reset, or
  clean.
- Do not use subagents.
- Do not terminate unrelated GPU processes.
- Remote execution is limited to `sitian@10.232.195.203`.
- Use `KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian`.
- Use `ControlMaster=no` and `ControlPath=none`.
- Exact greedy parity is mandatory.
- Target-KV movement must come from `engine.kv_offload_summaries`.
- Keep `kv_offload_gpu_blocks=68`, `kv_offload_logical_blocks=640`, and
  `kvcache_block_size=256`.
- Keep proposal KV GPU-resident; proposal-KV offload is out of scope.
- Retain failed artifacts and never reinterpret them as authority.
- Do not claim performance, KV8/KV4, second learned structure, production
  readiness, or Phase 1 completion.

---

### Task 1: Freeze the Independent 32K Gate Contract

**Files:**
- Create:
  `tools/test_qwen35_native_mtp_tp4_32k_target_kv_offload_gate.py`
- Create:
  `tools/qwen35_native_mtp_tp4_32k_target_kv_offload_gate.py`
- Read-only baseline:
  `tools/qwen35_native_mtp_tp4_16k_target_kv_offload_gate.py`

**Interfaces:**
- Consumes the complete 16K gate module as `_frozen_gate`.
- Produces 32K constants, `DEFAULT_SOURCE_FILES`, `validate_result()`, and
  `run_campaign()`.

- [x] **Step 1: Write the frozen-constant RED test**

Add:

```python
def test_contract_constants_are_frozen():
    assert gate.SCHEMA_VERSION == (
        "qwen35.native-mtp-tp4-32k-target-kv-offload.v1"
    )
    assert gate.CLASSIFICATION == (
        "QWEN35_NATIVE_MTP_TP4_32K_TARGET_KV_OFFLOAD_ESTABLISHED"
    )
    assert gate.PROMPT_TOKENS == 32768
    assert gate.MAX_OUTPUT_TOKENS == 8
    assert gate.MAX_PROPOSAL_TOKENS == 4
    assert gate.WORLD_SIZE == 4
    assert gate.BATCH_SIZES == (1, 4)
    assert gate.POLICIES == ("baseline", "native_mtp")
    assert gate.MAX_MODEL_LEN == 33024
    assert gate.MAX_NUM_BATCHED_TOKENS == 132096
    assert gate.MAX_NUM_PREFILL_TOKENS_PER_STEP == 1024
    assert gate.KV_OFFLOAD_GPU_BLOCKS == 68
    assert gate.KV_OFFLOAD_LOGICAL_BLOCKS == 640
    assert gate.KV_OFFLOAD_BLOCKWISE_BLOCKS == 8
    assert gate.BLOCK_SIZE == 256
    assert gate.REQUIRED_LIMITATIONS == (
        "phase1_not_promotable",
        "proposal_kv_offload_not_established",
        "tp1_32k_not_established",
        "performance_not_established",
        "kv_quantization_not_established",
        "second_learned_structure_not_established",
    )
```

- [ ] **Step 2: Run the RED test**

Run:

```bash
python3 -m pytest -q \
  tools/test_qwen35_native_mtp_tp4_32k_target_kv_offload_gate.py::test_contract_constants_are_frozen
```

Expected: collection or import failure because the 32K gate does not exist.

- [x] **Step 3: Implement the minimal gate overlay**

Create the gate with this structure:

```python
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_TOOLS = Path(__file__).resolve().parent
_frozen_gate = _load_module(
    "_qwen35_native_mtp_tp4_32k_frozen_gate",
    _TOOLS
    / "qwen35_native_mtp_tp4_16k_target_kv_offload_gate.py",
)

_frozen_gate.SCHEMA_VERSION = (
    "qwen35.native-mtp-tp4-32k-target-kv-offload.v1"
)
_frozen_gate.CLASSIFICATION = (
    "QWEN35_NATIVE_MTP_TP4_32K_TARGET_KV_OFFLOAD_ESTABLISHED"
)
_frozen_gate.PROMPT_TOKENS = 32768
_frozen_gate.REQUIRED_LIMITATIONS = (
    "phase1_not_promotable",
    "proposal_kv_offload_not_established",
    "tp1_32k_not_established",
    "performance_not_established",
    "kv_quantization_not_established",
    "second_learned_structure_not_established",
)
```

Append the three 32K authority files to
`_frozen_gate.DEFAULT_SOURCE_FILES`, export non-dunder names, retain the
frozen `validate_result`, and provide 32K default worker/verifier paths in
`run_campaign()`.

- [x] **Step 4: Run the constant test GREEN**

Run the command from Step 2.

Expected: `1 passed`.

- [x] **Step 5: Prove 16K source isolation**

Add:

```python
def test_loading_32k_gate_does_not_modify_frozen_gate_source():
    before = hashlib.sha256(FROZEN_GATE_PATH.read_bytes()).hexdigest()
    _load_module(
        "qwen35_native_mtp_tp4_32k_gate_isolation",
        GATE_PATH,
    )
    after = hashlib.sha256(FROZEN_GATE_PATH.read_bytes()).hexdigest()
    assert after == before
```

Run it and require PASS.

### Task 2: Enforce the Stronger 32K Result Contract

**Files:**
- Modify:
  `tools/qwen35_native_mtp_tp4_32k_target_kv_offload_gate.py`
- Modify:
  `tools/test_qwen35_native_mtp_tp4_32k_target_kv_offload_gate.py`

**Interfaces:**
- Consumes `_frozen_validate_result(value)`.
- Produces a canonical 32K result that requires real H2D and D2H movement in
  both native cells.

- [x] **Step 1: Write parameterized movement RED tests**

Use the 16K test fixture module with `frozen_test.gate = gate`, then add:

```python
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("h2d_copies", "real H2D copies"),
        ("h2d_bytes", "real H2D bytes"),
        ("d2h_copies", "real D2H copies"),
        ("d2h_bytes", "real D2H bytes"),
    ],
)
def test_validate_result_rejects_zero_native_movement(
    batch_size,
    field,
    message,
):
    result = _valid_result()
    for row in result["cells"][
        f"native_mtp:b{batch_size}"
    ]["kv_rank_deltas"]:
        row[field] = 0
    with pytest.raises(ValueError, match=message):
        gate.validate_result(result)
```

- [ ] **Step 2: Run the movement tests RED**

Expected: the inherited 16K validator accepts at least the b1 zero-movement
mutations.

- [x] **Step 3: Implement the 32K movement validator**

Add:

```python
_frozen_validate_result = _frozen_gate.validate_result


def validate_result(value: object) -> dict:
    normalized = _frozen_validate_result(value)
    for batch_size in BATCH_SIZES:
        movement = normalized["cells"][
            f"native_mtp:b{batch_size}"
        ]["kv_rank_deltas"]
        required = (
            ("h2d_copies", "real H2D copies"),
            ("h2d_bytes", "real H2D bytes"),
            ("d2h_copies", "real D2H copies"),
            ("d2h_bytes", "real D2H bytes"),
        )
        for field, label in required:
            if sum(row[field] for row in movement) <= 0:
                raise ValueError(
                    f"32K batch-{batch_size} native cell "
                    f"requires {label}"
                )
    return normalized


_frozen_gate.validate_result = validate_result
```

- [x] **Step 4: Run movement and inherited canonical tests GREEN**

Run:

```bash
python3 -m pytest -q \
  tools/test_qwen35_native_mtp_tp4_32k_target_kv_offload_gate.py \
  -k 'validate_result or contract_constants or isolation'
```

Expected: all selected tests pass.

### Task 3: Add the 32K Worker Overlay

**Files:**
- Create:
  `tools/qwen35_native_mtp_tp4_32k_target_kv_offload_worker.py`
- Modify:
  `tools/test_qwen35_native_mtp_tp4_32k_target_kv_offload_gate.py`

**Interfaces:**
- Consumes the frozen 16K worker and the new 32K gate.
- Produces the unchanged `run_policy_cell()` and CLI using 32K constants.

- [x] **Step 1: Write the worker RED test**

Add a fake engine factory and assert:

```python
assert kwargs["tensor_parallel_size"] == 4
assert kwargs["max_model_len"] == 33024
assert kwargs["max_num_batched_tokens"] == 132096
assert kwargs["max_num_prefill_tokens_per_step"] == 1024
assert kwargs["kv_offload_gpu_blocks"] == 68
assert kwargs["kv_offload_logical_blocks"] == 640
assert kwargs["kv_offload_blockwise_blocks"] == 8
assert len(cell["prompt_rows"][0]["token_ids"]) == 32768
```

Also assert the cell keeps `max_output_tokens=8`,
`max_proposal_tokens=4`, and policy `native_mtp`.

- [ ] **Step 2: Run the worker test RED**

Expected: import failure because the worker does not exist.

- [x] **Step 3: Implement the minimal worker overlay**

Create:

```python
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_TOOLS = Path(__file__).resolve().parent
gate = _load_module(
    "qwen35_native_mtp_tp4_32k_target_kv_offload_gate",
    _TOOLS
    / "qwen35_native_mtp_tp4_32k_target_kv_offload_gate.py",
)
_frozen_worker = _load_module(
    "_qwen35_native_mtp_tp4_32k_frozen_worker",
    _TOOLS
    / "qwen35_native_mtp_tp4_16k_target_kv_offload_worker.py",
)
_frozen_worker.gate = gate

for name, value in vars(_frozen_worker).items():
    if not name.startswith("__") and name != "gate":
        globals()[name] = value

if __name__ == "__main__":
    sys.exit(main())
```

- [x] **Step 4: Run the worker test GREEN**

Expected: PASS with a 32,768-token prompt.

### Task 4: Add the Independent 32K Verifier

**Files:**
- Create:
  `tools/verify_qwen35_native_mtp_tp4_32k_target_kv_offload_gate.py`
- Modify:
  `tools/test_qwen35_native_mtp_tp4_32k_target_kv_offload_gate.py`

**Interfaces:**
- Consumes the frozen 16K verifier implementation and the 32K gate.
- Produces the same `verify_run()` and CLI with the 32K canonical contract.

- [x] **Step 1: Write verifier dispatch and tamper RED tests**

Add tests that:

- load the verifier and assert `verifier.gate is gate` semantically by schema;
- verify a valid 32K fixture as `PASS`;
- mutate schema to the 16K schema and require `FAIL`;
- mutate prompt length to 16,384 and require `FAIL`;
- remove one 32K source file from the manifest and require `FAIL`;
- zero native b1 H2D movement and require `FAIL`; and
- append an undeclared result or manifest field and require `FAIL`.

- [ ] **Step 2: Run verifier tests RED**

Expected: import failure because the verifier does not exist.

- [x] **Step 3: Implement the verifier overlay**

Create:

```python
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_TOOLS = Path(__file__).resolve().parent
gate = _load_module(
    "qwen35_native_mtp_tp4_32k_target_kv_offload_gate",
    _TOOLS
    / "qwen35_native_mtp_tp4_32k_target_kv_offload_gate.py",
)
_frozen_verifier = _load_module(
    "_verify_qwen35_native_mtp_tp4_32k_frozen_gate",
    _TOOLS
    / "verify_qwen35_native_mtp_tp4_16k_target_kv_offload_gate.py",
)
_frozen_verifier.gate = gate

for name, value in vars(_frozen_verifier).items():
    if not name.startswith("__") and name != "gate":
        globals()[name] = value

if __name__ == "__main__":
    sys.exit(main())
```

- [x] **Step 4: Run verifier tests GREEN**

Expected: all verifier and tamper tests pass.

### Task 5: Add the Bounded Remote Runner

**Files:**
- Create:
  `tools/run_qwen35_native_mtp_tp4_32k_target_kv_offload_remote.sh`
- Modify:
  `tools/test_qwen35_native_mtp_tp4_32k_target_kv_offload_gate.py`

**Interfaces:**
- Consumes the frozen 16K runner as source text.
- Produces a derived runner targeting the 32K gate, worker, verifier, artifact
  parent, and remote run parent.

- [x] **Step 1: Write runner contract RED tests**

Require the derived runner to contain:

```text
sitian@10.232.195.203
FILE:/Users/bytedance/krb5cc_sitian
ControlMaster=no
ControlPath=none
qwen35_native_mtp_tp4_32k_target_kv_offload_gate.py
qwen35_native_mtp_tp4_32k_target_kv_offload_worker.py
verify_qwen35_native_mtp_tp4_32k_target_kv_offload_gate.py
qwen35_native_mtp_tp4_16k_target_kv_offload_gate.py
qwen35_native_mtp_tp4_16k_target_kv_offload_worker.py
verify_qwen35_native_mtp_tp4_16k_target_kv_offload_gate.py
campaign.status
campaign.pid
campaign.exit_code
authority.failed
REMOTE_COMMAND_RETRY_ATTEMPTS
REMOTE_RSYNC_RETRY_ATTEMPTS
POLL_INTERVAL_SECONDS
head -n 4
```

Reject `ControlMaster=yes`. Require fresh run refusal and preservation of the
four-idle-GPU gate.

- [ ] **Step 2: Run runner tests RED**

Expected: runner file missing.

- [x] **Step 3: Implement the derived runner**

Use the generic 32K runner derivation pattern:

1. bind `REPO_ROOT` through an exported 32K variable;
2. read the frozen 16K native runner;
3. replace the authority prefix and remote/local artifact parents;
4. expand the source archive to include both frozen 16K and new 32K files;
5. validate required fragments before executing the generated script; and
6. execute the generated script with all caller arguments and environment
   overrides preserved.

- [x] **Step 4: Run runner tests and shell syntax GREEN**

Run:

```bash
python3 -m pytest -q \
  tools/test_qwen35_native_mtp_tp4_32k_target_kv_offload_gate.py \
  -k remote_runner
bash -n \
  tools/run_qwen35_native_mtp_tp4_32k_target_kv_offload_remote.sh
```

Expected: all selected tests pass; `bash -n` exits zero.

### Task 6: Run the Complete Local Authority Gate

**Files:**
- Modify only if a focused test exposes a scoped defect.

**Interfaces:**
- Consumes all Task 1–5 files.
- Produces local GREEN evidence before any GPU launch.

- [x] **Step 1: Run the complete new test file**

Run:

```bash
python3 -m pytest -q \
  tools/test_qwen35_native_mtp_tp4_32k_target_kv_offload_gate.py
```

Expected: all tests pass.

- [x] **Step 2: Run frozen-authority regressions**

Run:

```bash
python3 -m pytest -q \
  tools/test_qwen35_native_mtp_tp4_16k_target_kv_offload_gate.py \
  tools/test_qwen35_generic_speculative_tp4_32k_gate.py \
  tools/test_engine_speculative_runtime.py \
  tools/test_model_runner_spec_verify.py
```

Expected: all tests pass. Treat unrelated pre-existing failures separately;
do not hide them.

- [x] **Step 3: Run compilation and diff validation**

Run:

```bash
python3 -m py_compile \
  tools/qwen35_native_mtp_tp4_32k_target_kv_offload_gate.py \
  tools/qwen35_native_mtp_tp4_32k_target_kv_offload_worker.py \
  tools/verify_qwen35_native_mtp_tp4_32k_target_kv_offload_gate.py \
  tools/test_qwen35_native_mtp_tp4_32k_target_kv_offload_gate.py
bash -n \
  tools/run_qwen35_native_mtp_tp4_32k_target_kv_offload_remote.sh
git diff --check -- \
  docs/superpowers/specs/2026-08-14-qwen35-native-mtp-tp4-32k-target-kv-offload-design.md \
  docs/superpowers/plans/2026-08-14-qwen35-native-mtp-tp4-32k-target-kv-offload.md \
  tools/qwen35_native_mtp_tp4_32k_target_kv_offload_gate.py \
  tools/qwen35_native_mtp_tp4_32k_target_kv_offload_worker.py \
  tools/verify_qwen35_native_mtp_tp4_32k_target_kv_offload_gate.py \
  tools/run_qwen35_native_mtp_tp4_32k_target_kv_offload_remote.sh \
  tools/test_qwen35_native_mtp_tp4_32k_target_kv_offload_gate.py
```

Expected: all commands exit zero.

### Task 7: Run the Fresh TP4/32K Authority

**Files:**
- Create only under:
  `artifacts/qwen35_native_mtp_tp4_32k_target_kv_offload/<run-id>/`
- Modify after PASS:
  `AGENT_HANDOFF_STATE.md`
- Modify after PASS:
  `docs/superpowers/audits/2026-08-12-phase1-objective-coverage.md`

**Interfaces:**
- Consumes the complete source-bound 32K runner.
- Produces remote and local independent verification plus the Phase 1 status
  update.

- [x] **Step 1: Launch one fresh campaign**

Run:

```bash
RUN_ID=native-mtp-tp4-32k-$(date +%Y%m%d)-1 \
  bash \
  tools/run_qwen35_native_mtp_tp4_32k_target_kv_offload_remote.sh
```

The runner must stop safely if fewer than four idle GPUs have at least the
frozen free-memory threshold.

- [x] **Step 2: Reuse the recorded campaign**

Poll the existing PID/status through the runner. Do not launch a second
campaign while the first is `RUNNING`.

- [ ] **Step 3: Run a fresh explicit local verifier**

Run:

```bash
python3 \
  tools/verify_qwen35_native_mtp_tp4_32k_target_kv_offload_gate.py \
  artifacts/qwen35_native_mtp_tp4_32k_target_kv_offload/<run-id>/artifacts/authority \
  --source-root \
  artifacts/qwen35_native_mtp_tp4_32k_target_kv_offload/<run-id>/source
```

Expected:

```json
{"classification":"PASS","failures":[]}
```

- [x] **Step 4: Audit the result directly**

Confirm from `result.json`:

- `parity.baseline_native.b1 == true`;
- `parity.baseline_native.b4 == true`;
- prompt length is exactly 32,768 in every cell;
- all four ranks are present;
- native proposal/accepted/rejected totals are positive;
- accepted-prefix replay is zero;
- b1 and b4 release rows are complete on every rank;
- b1 and b4 target-KV H2D and D2H copies/bytes are positive;
- movement provenance is `engine.kv_offload_summaries`;
- capacity is `68 / 640` and peak residency is at most 68;
- target-KV, side-state, and residency receipt phases are complete;
- proposal transactions and physical slots are zero after completion;
- runtime poison is false;
- process, shared-memory, rank, and child cleanup pass;
- selected-GPU process inventory is unchanged; and
- target, MTP, and source-tree digests match the manifest.

- [ ] **Step 5: Update handoff and Phase 1 audit**

Record:

- run ID and authority path;
- remote, runner-local, and fresh-local verifier results;
- exact parity;
- proposal acceptance/rejection totals;
- b1/b4 per-rank target-KV movement;
- capacity and peak residency;
- release rows and zero-leak cleanup;
- source and checkpoint digests;
- retained failed runs and their exact failure boundaries; and
- non-claims.

The updated Phase 1 verdict remains `NOT_PROMOTABLE` until controlled
native-MTP performance and a second learned structure are established.

- [ ] **Step 6: Run final fresh verification**

Repeat Task 6 tests, the explicit verifier, `py_compile`, `bash -n`, and
scoped `git diff --check`. Use only this fresh output for the completion
report.
