# Autoregressive Draft Same-Policy Priming Control Design

## Goal

Determine whether the TP4 Qwen3 batch-4 second-process speedup is caused by
process-level cadence such as model page-cache reuse, CUDA/JIT cache reuse, or
other cross-process warm state before selecting a runtime optimization.

## Evidence That Motivates the Control

The source-bound r3 and r4 telemetry campaigns used opposite policy orders:

```text
r3: target,learned
r4: learned,target
```

Both policies were faster when executed second:

```text
target median E2E:
  first:   5.522148 s
  second:  3.958820 s
  change: -28.31%

learned median E2E:
  first:  11.641256 s
  second:  9.840383 s
  change: -15.47%

learned proposal-forward median:
  first:   6564.098539 ms
  second:  5323.351506 ms
  change: -18.90%
```

All verifier-bound runtime and telemetry source hashes matched. Sampled GPU
clocks, memory clocks, P-state, throttle mask, and temperature remained
stable. Host logs were retained but not semantically aligned by the verifier.

## Selected Approach

Before every measured policy worker, run an isolated worker process with the
same policy:

```text
prime(target)  -> measured(target)
prime(learned) -> measured(learned)
```

The prime worker uses the same model paths, TP4 GPU set, batch size, prompt
shape, proposal length, and temperature as the measured worker. It performs
two warmups and one completed request batch. Its JSON and stdout/stderr log
are retained under separate prime directories.

The prime worker is excluded from:

- measured timing medians;
- stationarity classification;
- measured GPU telemetry CSV files;
- timing and telemetry verifier inputs.

This keeps the existing timing and telemetry artifact schemas unchanged and
preserves direct comparability with r3 and r4.

## Rejected Alternatives

### Duplicate Each Measured Policy

Running each policy twice and treating both as measured cells would directly
show first-versus-second behavior, but it requires multi-cell timing and
telemetry schemas, new independent verifiers, and new aggregation rules. That
is unnecessary for the first cadence control.

### Switch Policies in One Persistent Process

A single persistent engine could eliminate process startup, but it changes
the current worker and engine ownership boundary. Policy switching may retain
state that production does not retain and would introduce a larger runtime
change than the phenomenon being tested.

## Runner Interface

Extend:

```text
tools/run_autoregressive_draft_b4_instability_telemetry_remote.sh
```

with:

```text
environment:
  PRIME_EACH_POLICY=0|1

CLI:
  --prime-each-policy
```

The default remains disabled so existing commands and bundles preserve their
meaning.

When enabled, the remote bundle records:

```text
prime-each-policy.txt
prime-workers/target-prime-b4.json
prime-workers/learned-prime-b4.json
prime-logs/target-prime-b4.log
prime-logs/learned-prime-b4.log
```

The command record must also include:

```text
PRIME_EACH_POLICY=1
```

## Remote Data Flow

For every policy in the validated policy order:

1. Run the same-policy prime worker without GPU or host samplers.
2. Require the prime worker to exit successfully.
3. Start the existing script-owned GPU and host samplers.
4. Run the existing measured worker with two warmups and eight measured
   repeats.
5. Stop only sampler PIDs owned by the script.
6. Continue with the unchanged timing diagnostic, telemetry assembler, remote
   verifiers, download, local verifiers, and manifest.

The prime worker command is:

```bash
"${python_executable}" \
  tools/autoregressive_draft_performance_worker.py \
    --target-model "${target_model}" \
    --draft-model "${draft_model}" \
    --policy "${policy}" \
    --batch-size 4 \
    --warmup-runs 2 \
    --measured-runs 1 \
    --out "${artifacts}/prime-workers/${policy}-prime-b4.json"
```

The measured worker remains:

```text
batch size:      4
warmup runs:     2
measured runs:   8
```

## Campaign Matrix

Run two new source-bound campaigns:

```text
campaign P1:
  policy order: target,learned
  prime each policy: enabled

campaign P2:
  policy order: learned,target
  prime each policy: enabled
```

Do not merge medians across P1 and P2.

Compare:

- P1 target against r3 target-first and r4 target-second;
- P2 target against the same references;
- P1 learned against r3 learned-second and r4 learned-first;
- P2 learned against the same references;
- learned proposal-forward sequences;
- within-campaign stationarity;
- per-repeat/per-GPU telemetry invariants.

## Interpretation Rules

### Priming Removes the Position Effect

If the same policy has materially similar medians in P1 and P2 and the
first-position result moves toward the historical second-position result,
classify:

```text
PROCESS_CADENCE_EFFECT_SUPPORTED
```

Do not identify JIT, page cache, allocator, or another sub-cause without a
separate controlled experiment.

### Position Effect Remains

If the first-position policy remains materially slower after same-policy
priming, classify:

```text
POSITION_EFFECT_REMAINS
```

The next experiment must inspect semantically aligned host conditions or a
more specific runtime boundary.

### Timing Remains Too Unstable

If range-over-median remains above the existing stationarity threshold and
the cross-order direction is not consistent, classify:

```text
PRIMING_CONTROL_INCONCLUSIVE
```

Do not select a performance optimization.

## Safety and Correctness

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Use `sitian@10.232.195.203`.
- Use `/dev/shm/sitian/tllm-qwen35-target-qwen3-draft-20260815`.
- Do not write experiment artifacts under `/data00`.
- Preserve `MAX_PROPOSAL_TOKENS=4`.
- Preserve temperature zero and exact greedy parity.
- Preserve workload-derived Proposal-KV capacity.
- Do not add `torch.cuda.synchronize()` to the measured request path.
- Do not terminate unrelated GPU processes.
- Preserve the existing GPU-7 `python3` service.
- Do not treat synthetic KV movement as real movement.
- Do not stage, commit, push, stash, reset, clean, or switch branches.

## Verification

Focused source-contract tests must prove:

- priming is disabled by default;
- `--prime-each-policy` enables it;
- the command record retains the flag;
- prime directories are created remotely;
- prime runs use the same policy immediately before its measured worker;
- prime runs use two warmups and one measured repeat;
- measured runs remain two warmups and eight measured repeats;
- prime JSON/log paths cannot overwrite measured paths;
- samplers start only after the prime worker;
- timing and telemetry assembler inputs remain the measured worker paths;
- no `torch.cuda.synchronize` text is introduced.

The full local gate remains:

```bash
python3 -m pytest -q \
  tools/test_autoregressive_draft_performance_gate.py \
  tools/test_autoregressive_draft_instability_telemetry.py

python3 -m py_compile \
  tinyvllm/engine/autoregressive_draft_executor.py \
  tools/autoregressive_draft_performance_worker.py \
  tools/autoregressive_draft_b4_timing_diagnostic.py \
  tools/autoregressive_draft_instability_telemetry.py \
  tools/verify_autoregressive_draft_instability_telemetry.py

bash -n \
  tools/run_autoregressive_draft_b4_instability_telemetry_remote.sh
```

## Claim Boundary

This control may establish that same-policy process priming removes or does
not remove the policy-position effect. It cannot by itself establish:

- a specific JIT, page-cache, allocator, CUDA, TP, or host root cause;
- a stable production performance baseline;
- a promotable runtime optimization;
- 4K/16K/32K performance;
- a second learned model structure;
- Phase 1 completion.

