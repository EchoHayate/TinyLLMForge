# Autoregressive Draft Same-Policy Priming Implementation Plan

> **For agentic workers:** Execute inline in the current checkout with
> `executing-plans`. Do not use subagents, create or switch branches/worktrees,
> stage, commit, push, stash, reset, or clean.

**Goal:** Add an optional same-policy prime worker before each measured policy
worker and run both primed policy orders to test the observed second-process
speedup.

**Architecture:** Keep the performance worker, timing artifact, telemetry
artifact, and both independent verifiers unchanged. Extend only the remote
runner control flow and its source-contract test so prime workers are retained
separately and excluded from measured telemetry and aggregation.

**Tech Stack:** Bash, Python 3.11, pytest, SSH, JSON, `nvidia-smi`, `vmstat`,
`mpstat`, `pidstat`.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Remote host is `sitian@10.232.195.203`.
- Remote base is
  `/dev/shm/sitian/tllm-qwen35-target-qwen3-draft-20260815`.
- Do not write new artifacts under `/data00`.
- Preserve temperature zero, exact greedy parity, accepted-prefix semantics,
  `MAX_PROPOSAL_TOKENS=4`, and workload-derived Proposal-KV capacity.
- Do not add `torch.cuda.synchronize()` to the measured request path.
- Do not terminate unrelated GPU processes; preserve the GPU-7 `python3`
  service.
- Do not stage, commit, push, stash, reset, clean, or switch branches.

---

### Task 1: Prime-Control Runner Contract

**Files:**
- Modify:
  `tools/test_autoregressive_draft_instability_telemetry.py`
- Modify:
  `tools/run_autoregressive_draft_b4_instability_telemetry_remote.sh`

**Interfaces:**
- Consumes: existing validated `POLICY_ORDER`.
- Produces: `PRIME_EACH_POLICY=0|1` and CLI `--prime-each-policy`.
- Produces separate `prime-workers/` and `prime-logs/` artifacts.
- Leaves measured worker paths and verifier inputs unchanged.

- [x] **Step 1: Write the failing source-contract test**

Add:

```python
def test_remote_runner_supports_same_policy_priming_control():
    script = RUNNER_PATH.read_text(encoding="utf-8")
    for expected in (
        'PRIME_EACH_POLICY="${PRIME_EACH_POLICY:-0}"',
        "--prime-each-policy",
        "PRIME_EACH_POLICY=1",
        "'${REMOTE_ARTIFACTS}/prime-workers'",
        "'${REMOTE_ARTIFACTS}/prime-logs'",
        'prime_policy "${policy}"',
        '--measured-runs 1',
        '"${artifacts}/prime-workers/${policy}-prime-b4.json"',
        '"${artifacts}/prime-logs/${policy}-prime-b4.log"',
        'if [[ "${prime_each_policy}" -eq 1 ]]; then',
    ):
        assert expected in script

    assert """for policy in "${policy_order[@]}"; do
  if [[ "${prime_each_policy}" -eq 1 ]]; then
    prime_policy "${policy}"
  fi
  run_policy "${policy}"
done
""" in script
```

Also keep the existing assertions that measured workers use
`--warmup-runs 2`, `--measured-runs 8`, and that no
`torch.cuda.synchronize` text exists.

- [x] **Step 2: Run the focused test and confirm RED**

Run:

```bash
python3 -m pytest -q \
  tools/test_autoregressive_draft_instability_telemetry.py \
  -k same_policy_priming
```

Expected: failure because the runner has no priming interface or artifacts.

- [x] **Step 3: Add the CLI and strict boolean state**

At the runner defaults add:

```bash
PRIME_EACH_POLICY="${PRIME_EACH_POLICY:-0}"
```

Add the flag parser:

```bash
--prime-each-policy)
  PRIME_EACH_POLICY=1
  shift
  ;;
```

Validate:

```bash
case "${PRIME_EACH_POLICY}" in
  0|1)
    ;;
  *)
    printf 'invalid prime-each-policy value: %s\n' \
      "${PRIME_EACH_POLICY}" >&2
    exit 2
    ;;
esac
```

Record it in `command.txt` and pass it through both remote heredoc argument
layers without changing the measured worker arguments.

- [x] **Step 4: Create separate remote prime directories**

Extend remote directory creation with:

```bash
'${REMOTE_ARTIFACTS}/prime-workers'
'${REMOTE_ARTIFACTS}/prime-logs'
```

Record the effective state:

```bash
printf '%s\n' "${prime_each_policy}" \
  > "${remote_artifacts}/prime-each-policy.txt"
```

- [x] **Step 5: Add the minimal prime worker**

Inside the campaign heredoc add:

```bash
prime_policy() {
  local policy="$1"
  "${python_executable}" \
    tools/autoregressive_draft_performance_worker.py \
      --target-model "${target_model}" \
      --draft-model "${draft_model}" \
      --policy "${policy}" \
      --batch-size 4 \
      --warmup-runs 2 \
      --measured-runs 1 \
      --out "${artifacts}/prime-workers/${policy}-prime-b4.json" \
      >"${artifacts}/prime-logs/${policy}-prime-b4.log" 2>&1
}
```

Change the policy loop to:

```bash
for policy in "${policy_order[@]}"; do
  if [[ "${prime_each_policy}" -eq 1 ]]; then
    prime_policy "${policy}"
  fi
  run_policy "${policy}"
done
```

Do not start samplers in `prime_policy`.

- [x] **Step 6: Run focused GREEN and syntax checks**

Run:

```bash
python3 -m pytest -q \
  tools/test_autoregressive_draft_instability_telemetry.py \
  -k runner
bash -n \
  tools/run_autoregressive_draft_b4_instability_telemetry_remote.sh
```

Expected: runner tests pass and Bash syntax is valid.

---

### Task 2: Full Local Regression Gate

**Files:**
- Verify:
  `tools/test_autoregressive_draft_performance_gate.py`
- Verify:
  `tools/test_autoregressive_draft_instability_telemetry.py`

**Interfaces:**
- Proves the measured worker, artifact schemas, and verifier behavior remain
  unchanged.

- [x] **Step 1: Run the complete local suite**

Run:

```bash
python3 -m pytest -q \
  tools/test_autoregressive_draft_performance_gate.py \
  tools/test_autoregressive_draft_instability_telemetry.py
```

Expected: all tests pass.

- [x] **Step 2: Run compilation and scoped whitespace checks**

Run:

```bash
python3 -m py_compile \
  tinyvllm/engine/autoregressive_draft_executor.py \
  tools/autoregressive_draft_performance_worker.py \
  tools/autoregressive_draft_b4_timing_diagnostic.py \
  tools/autoregressive_draft_instability_telemetry.py \
  tools/verify_autoregressive_draft_instability_telemetry.py

git diff --check -- \
  tools/run_autoregressive_draft_b4_instability_telemetry_remote.sh \
  tools/test_autoregressive_draft_instability_telemetry.py \
  docs/superpowers/specs/2026-08-15-autoregressive-draft-same-policy-priming-design.md \
  docs/superpowers/plans/2026-08-15-autoregressive-draft-same-policy-priming.md
```

Expected: compilation and whitespace checks pass.

---

### Task 3: Primed Target-Then-Learned Authority

**Files:**
- Create:
  `experiments/autoregressive_draft/tp4-qwen3-b4-instability-telemetry-primed-target-learned-gpu3467-r5-20260815/`

**Interfaces:**
- Produces the first same-policy-primed campaign.

- [x] **Step 1: Confirm remote resources**

Run a read-only SSH preflight proving:

```text
/dev/shm has sufficient free space
target and draft model directories exist
GPUs 3,4,6 are free
GPU7 retains only the known python3 service relevant to the selected set
```

- [x] **Step 2: Run the source-bound primed campaign**

Run:

```bash
bash tools/run_autoregressive_draft_b4_instability_telemetry_remote.sh \
  --remote-host sitian@10.232.195.203 \
  --remote-python /data00/home/sitian/miniconda3/envs/py311/bin/python \
  --remote-base /dev/shm/sitian/tllm-qwen35-target-qwen3-draft-20260815 \
  --target-model \
    /dev/shm/sitian/tllm-qwen35-target-qwen3-draft-20260815/target-qwen3-1.7b \
  --draft-model \
    /dev/shm/sitian/tllm-qwen35-target-qwen3-draft-20260815/draft \
  --gpu-indices 3,4,6,7 \
  --policy-order target,learned \
  --prime-each-policy \
  --run-tag \
    tp4-qwen3-b4-instability-telemetry-primed-target-learned-gpu3467-r5-20260815 \
  --local-run \
    experiments/autoregressive_draft/tp4-qwen3-b4-instability-telemetry-primed-target-learned-gpu3467-r5-20260815
```

- [x] **Step 3: Verify the downloaded authority**

Run both local verifiers and:

```bash
(
  cd experiments/autoregressive_draft/\
tp4-qwen3-b4-instability-telemetry-primed-target-learned-gpu3467-r5-20260815
  test "$(cat prime-each-policy.txt)" = "1"
  test "$(cat policy-order.txt)" = "target,learned"
  test -s prime-workers/target-prime-b4.json
  test -s prime-workers/learned-prime-b4.json
  test -s prime-logs/target-prime-b4.log
  test -s prime-logs/learned-prime-b4.log
  shasum -a 256 -c manifest.sha256
)
```

Expected: all checks pass.

---

### Task 4: Primed Learned-Then-Target Authority

**Files:**
- Create:
  `experiments/autoregressive_draft/tp4-qwen3-b4-instability-telemetry-primed-learned-target-gpu3467-r6-20260815/`

**Interfaces:**
- Produces the reversed same-policy-primed companion.

- [x] **Step 1: Reconfirm selected-GPU process inventory**

Run a read-only `nvidia-smi` preflight and preserve the GPU-7 service.

- [x] **Step 2: Run the reversed primed campaign**

Use the Task 3 command with:

```text
--policy-order learned,target
--run-tag tp4-qwen3-b4-instability-telemetry-primed-learned-target-gpu3467-r6-20260815
--local-run experiments/autoregressive_draft/tp4-qwen3-b4-instability-telemetry-primed-learned-target-gpu3467-r6-20260815
```

- [x] **Step 3: Verify the downloaded authority**

Run both local verifiers, require both prime worker JSON/log pairs, require
`prime-each-policy.txt` to equal `1`, require `policy-order.txt` to equal
`learned,target`, and verify the full manifest.

---

### Task 5: Four-Campaign Comparison and Persistence

**Files:**
- Create after results:
  `README.md` in both primed bundles.
- Modify:
  `docs/superpowers/audits/2026-08-15-phase1-prompt-to-artifact-coverage.md`
- Modify: `AGENT_HANDOFF_STATE.md`

**Interfaces:**
- Consumes r3, r4, primed r5, and primed r6.
- Produces a non-promotional cadence classification.

- [x] **Step 1: Recompute comparison values**

For every campaign and policy extract:

```text
TTFT values and median
TPOT values and median
E2E values and median
throughput values and median
learned proposal-forward values and median
range-over-median
half-drift fraction
sample count per repeat/GPU
clock, P-state, throttle, and temperature ranges
```

Assert the timing and telemetry verifier-bound source hash sets match across
all four bundles.

- [x] **Step 2: Apply exactly one priming classification**

Use the design rules:

```text
PROCESS_CADENCE_EFFECT_SUPPORTED
POSITION_EFFECT_REMAINS
PRIMING_CONTROL_INCONCLUSIVE
```

Do not identify a specific subsystem root cause.

- [x] **Step 3: Persist exact claim boundaries**

Each primed README and the phase audit/handoff must record:

- exact policy order and priming state;
- prime artifact paths;
- remote tests, verifier receipts, manifest, and sampler stderr;
- four-campaign timing comparison;
- telemetry coverage and invariants;
- selected-GPU process inventory;
- host-log semantic limitation;
- the next experiment;
- unchanged `PHASE_1=NOT_ACHIEVED`.

- [x] **Step 4: Run the final verification**

Run the complete local suite, compilation, runner syntax, both primed
verifiers, all four manifests, a source-hash identity audit, and scoped
`git diff --check`.

Expected: every command passes. Do not stage, commit, or push.

