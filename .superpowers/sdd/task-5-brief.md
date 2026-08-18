### Task 5: Canonical Exact-Identity Diagnostic

**Files:**
- Create: `tools/autoregressive_draft_command_timeline_diagnostic.py`
- Create: `tools/test_autoregressive_draft_command_timeline_diagnostic.py`

**Interfaces:**
- `EpochIdentity`.
- `expected_epoch_identities()`.
- `validate_epoch_worker(worker, identity)`.
- `join_repeat_timeline(worker, repeat_index)`.
- `compute_sync_debt(repeat)`.
- `build_epoch_admission(identity, raw_inputs)`.
- `compute_paired_boundary_effects(epochs)`.
- `classify_boundary(bundle_admission, effects)`.
- `build_command_timeline_artifact(*, metadata, epoch_raw_inputs, input_files, source_files)`.
- `validate_command_timeline_artifact(artifact)`.

- [ ] **Step 1: Write schedule, exact identity, and parity tests**

Use fixed constants:

```python
SCHEMA_VERSION = 1
BLOCK_SCHEDULE = (
    ("eager", "graph"),
    ("graph", "eager"),
    ("graph", "eager"),
    ("eager", "graph"),
)
MEASURED_RUNS_PER_EPOCH = 5
MEASURED_RUNS_TOTAL = 40
BATCH_SIZE = 4
MAX_PROPOSAL_TOKENS = 4
PROMPT_TOKENS = 256
OUTPUT_TOKENS = 16
TEMPERATURE = 0.0
ROBUST_DISPERSION_LIMIT = 0.10
HALF_DRIFT_LIMIT = 0.15
ABSOLUTE_CONSERVATION_NS = 2_000_000
RELATIVE_CONSERVATION_LIMIT = 0.01
BOUNDARY_EXPLANATION_THRESHOLD = 0.60
BOUNDARY_BLOCK_COUNT = 3
UNEXPLAINED_E2E_LIMIT = 0.10
```

Tests must independently reject:

- wrong schedule label or position;
- non-learned policy;
- TP not equal to four;
- batch not equal to four;
- proposal limit not equal to four;
- prompt length or prompt digest mismatch;
- output length or request order mismatch;
- nonzero temperature;
- non-direct Proposal-KV allocator;
- Proposal-KV offload enabled;
- graph/eager source, checkpoint, tokenizer, or GPU UUID mismatch;
- graph capture/replay/resource drift;
- eager capture, replay, or ready entry;
- target token, proposal row, accepted-prefix, accepted-token, transaction
  digest, acceptance, or active-transaction mismatch; and
- padded or oversized logical proposal rows.

- [ ] **Step 2: Write timeline join, debt, and conservation tests**

Create a valid fixture with four rank command rows, CUDA rows, engine spans,
and request timing. Assert:

```python
repeat = diagnostic.join_repeat_timeline(worker, 0)
assert repeat["critical_rank"] == 3
assert repeat["components_ns"] == {
    "worker_queue_debt": 60_000_000,
    "worker_cuda_execution": 400_000_000,
    "ack_wait": 20_000_000,
    "scheduler_postprocess": 100_000_000,
}
assert repeat["conservation"]["passed"] is True
```

Reject:

- mismatched boot IDs or clock metadata;
- duplicate, missing, or reordered command IDs;
- unknown command references in CUDA or engine rows;
- negative queue/ack/CUDA/phase duration;
- CUDA greater than method wall time;
- overlap greater than containing intervals;
- a non-ack command with ack timestamps;
- an ack command with missing ack wait;
- unexplained step residual above tolerance; and
- timeline rows outside the repeat campaign interval.

- [ ] **Step 3: Write stationarity and classification boundary tests**

Test exact threshold inclusivity:

```python
assert stationarity_for_values(
    [100.0, 100.0, 100.0, 100.0, 110.0]
)["robust_dispersion_passed"] is True
```

Construct four blocks where queue debt explains `60%` in exactly three blocks,
same-sign count is three, and residual is exactly `10%`; expect:

```text
BOUNDARY_LOCALIZED
localized_boundary=worker_queue_debt
```

Move each threshold one unit beyond the allowed boundary and expect:

```text
PAIRED_PROTOCOL_UNSTABLE
stable_but_unlocalized=true
```

Also cover precedence:

```text
identity/parity failure -> INVALID_IDENTITY_OR_CORRECTNESS
timeline/conservation failure -> TIMELINE_INCOMPLETE_OR_NONCONSERVING
stationarity failure -> PAIRED_PROTOCOL_UNSTABLE
```

- [ ] **Step 4: Run tests and confirm RED**

```bash
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-command-timeline-pycache \
python3 -m pytest -q \
  tools/test_autoregressive_draft_command_timeline_diagnostic.py \
  -k 'schedule or identity or parity or timeline or debt or conservation or stationarity or classification'
```

Expected: collection fails because the diagnostic module does not exist.

- [ ] **Step 5: Implement canonical diagnostic**

Reuse only pure validated helpers from:

- `autoregressive_draft_cuda_graph_contract.py` for exact graph counter and
  bounded logical-row rules;
- `autoregressive_draft_paired_stability_diagnostic.py` for stationarity and
  balanced block effect primitives;
- `autoregressive_draft_instability_telemetry.py` and
  `autoregressive_draft_host_semantic_diagnostic.py` for telemetry alignment.

Do not import or mutate completed schema-v2 payload state.

The canonical artifact must include these exact top-level keys:

```python
TOP_LEVEL_KEYS = (
    "schema_version",
    "schedule",
    "configuration",
    "provenance",
    "raw_input_files",
    "source_files",
    "epochs",
    "blocks",
    "admission",
    "effects",
    "classification",
    "localized_boundary",
    "stable_but_unlocalized",
    "runtime_optimization_authorized",
    "performance_improvement_established",
    "phase_1_complete",
    "promotion_ready",
)
```

`validate_command_timeline_artifact` recomputes all derived fields from the
embedded normalized epoch rows and rejects any mismatch.

- [ ] **Step 6: Run diagnostic tests and confirm GREEN**

Run the Step 4 command without `-k`.

Expected: all tests pass.

- [ ] **Step 7: Commit and push the diagnostic**

```bash
git add -- \
  tools/autoregressive_draft_command_timeline_diagnostic.py \
  tools/test_autoregressive_draft_command_timeline_diagnostic.py
git diff --cached --check
git -c core.hooksPath=/dev/null commit \
  -m "feat(runtime): classify command timeline boundaries" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

---
