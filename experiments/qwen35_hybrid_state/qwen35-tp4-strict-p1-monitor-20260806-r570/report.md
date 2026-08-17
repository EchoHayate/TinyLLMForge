# Qwen3.5 TP4 strict-P1 low-frequency monitor

## Active job

```text
launchd label:
com.bytedance.tinyllmforge.qwen35.strictp1.r570

monitor tag:
qwen35-tp4-strict-p1-monitor-20260806-r570

canonical run-tag base:
qwen35-tp4-strict-p1-canonical-20260806-r571

launch-attempt pattern:
qwen35-tp4-strict-p1-canonical-20260806-r571-attemptNNN
```

The launchd PID at startup was `32011`.

r562, r564, r566, and r568 were removed before this monitor was submitted.
They performed only read-only resource sampling and never started a launch
attempt.

## Admission

Only GPUs `2,4,5,6` are eligible. Each must expose:

```text
free bytes >= 25769803776
compute_processes == []
```

Two consecutive `READY` samples separated by at least 60 seconds are required.
The canonical executor then performs a fresh resource preflight before
authorization or remote path creation. A launch-time
`BLOCKED_RESOURCES` race is recorded and monitoring resumes.

No dummy process reserves GPUs. No pre-existing process is signaled, stopped,
reprioritized, or modified.

## Monitor health

At sample `21` (`2026-08-06T08:58:10.942195+00:00`), the detached monitor
had produced 21 contiguous samples:

```text
BLOCKED_RESOURCES samples: 21
SAMPLE_FAILED samples:      0
minimum interval:           61.836522 seconds
maximum interval:           64.346401 seconds
mean interval:              62.2569553 seconds
intervals below 60 seconds: 0
launch attempts:            0
```

Both launchd stdout and stderr logs remained empty. The job was still running
as PID `32011` and had never exited.

At approximately `2026-08-06T15:49:00+00:00`, the original ControlMaster
socket pathname disappeared while the detached monitor process remained
alive. An already authenticated replacement mux was independently verified:

```text
socket: /tmp/ssh-dynamic-token-ppo-a66cd65-attempt3
master PID: 66932
remote identity: sitian@n232-195-203
```

The original pathname was restored as a symbolic link to that verified mux:

```text
/tmp/ssh-dynamic-token-ppo-a66cd65
  -> /tmp/ssh-dynamic-token-ppo-a66cd65-attempt3
```

The active monitor was not restarted. Its own subsequent samples `403` and
`404` completed normally as `BLOCKED_RESOURCES`, with no `SAMPLE_FAILED`
record, proving that PID `32011` resumed its configured transport path.

## Current state

The first active sample was:

```text
observed_at: 2026-08-06T08:37:25.803089+00:00
classification: BLOCKED_RESOURCES

GPU 2: 68,789,731,328 free bytes, 5 compute processes
GPU 4: 50,963,939,328 free bytes, 5 compute processes
GPU 5: 84,979,744,768 free bytes, 0 compute processes
GPU 6: 58,502,152,192 free bytes, 3 compute processes
```

No benchmark, model load, CUDA worker, remote run directory, authorization, or
new GPU allocation has started.

## Transport and timeout

The monitor reuses:

```text
/tmp/ssh-dynamic-token-ppo-a66cd65
```

The local ControlMaster wrapper injects a `43200`-second timeout only for the
canonical `workers` command. Explicit per-call timeouts retain priority.

If the worker step stalls:

1. the subprocess adapter terminates and then kills the local SSH client;
2. the command returns code `124`;
3. the schema-v1 executor writes execution failure evidence;
4. the monitor enters scoped remote cleanup.

Other commands retain their existing timeout behavior.

## Cleanup

Cleanup anchors only on processes whose `/proc/<pid>/cmdline` contains both
the unique launch-attempt run tag and exact remote run root. It snapshots
parent links and process start-time ticks, expands those roots to the complete
descendant closure, and signals only PIDs whose start-time ticks still match.

This covers spawned TP workers whose own command lines omit the run tag while
protecting against PID reuse. Local `SIGTERM` and `SIGINT` during launch also
enter this cleanup path.

After cleanup:

```text
remaining_target_pids == []
matched_gpu_pids_after_cleanup == []
```

must both hold.

## Cleanup integration evidence

A remote no-GPU test created only a uniquely tagged `bash` root and one
`sleep` child:

```text
root_pids:                      [1223948]
descendants:                    [1223953]
target_pids:                    [1223948, 1223953]
remaining_target_pids:          []
matched_gpu_pids_after_cleanup: []
classification:                 CLEAN
```

The temporary log was removed. No pre-existing process or GPU allocation was
selected.

## Verification

```text
ControlMaster transport: 5 tests passed
monitor core:            6 tests passed
live monitor runner:     5 tests passed
py_compile:              PASS

strict-P1 source tree:
88ebda203decaf26a721f6299a5c9ed08b648841feb0b972f1395d6cc609eb9c

strict-P1 source tar:
8f807c0e51010fddd0bd474670ae41c09beda604c8247f672fc047f533c53a5d
```

Monitor files remain outside the frozen 91-file strict-P1 owned-source
inventory.

## Post-run acceptance boundary

The canonical independent verifier compares `exact_restore` against
`recompute` on the same TP4 run and requires all of the following before it
can classify the run as `GO`:

```text
output token IDs:                  exact match
correctness logits:                atol 2e-5, rtol 1e-5
W1 median TTFT ratio:              <= 0.85
W2 median TTFT ratio:              <= 0.75
W3 throughput ratio:               >= 1.15
reuse-workload repetition TTFT:    <= 1.05
all-workload decode latency ratio: <= 1.02
control/miss E2E ratio:            <= 1.05
initialization ratio:              <= 1.10
peak CUDA reserved ratio:          <= 1.10
KV capacity and visible blocks:    equal
required-workload evictions:       none
```

The artifact separately records:

```text
cuda_allocated_bytes
cuda_reserved_bytes
cuda_peak_allocated_bytes
cuda_peak_reserved_bytes
hybrid_cache_current_bytes
hybrid_cache_current_logical_bytes
hybrid_cache_deduplicated_bytes
hybrid_cache_peak_bytes
```

Therefore `GO` is authoritative for correctness preservation and the frozen
performance gates. It is not, by itself, proof that total physical
CUDA-resident cache is lower than `recompute`: the current contract permits up
to a `1.10` peak-CUDA-reserved ratio and does not impose a positive minimum on
the logical-to-physical snapshot ratio. After download, report physical CUDA
bytes, physical snapshot bytes, logical snapshot bytes, and their ratios
separately. Do not describe logical deduplication as physical VRAM savings.

## Post-trigger acceptance checklist

For the concrete attempt tag recorded in `monitor_result.json`, require all of
the following before reporting a successful experiment:

1. `monitor_result.json` exists and its `cleanup.classification` is `CLEAN`.
2. The attempt directory contains:
   - `plan/remote_execution_plan.json`
   - `runtime/consumed_authorization.json`
   - `runtime/execution_receipt.json`
   - `plan/downloaded_benchmark/artifact`
   - `plan/local_verifier_source`
3. `runtime/execution_failure.json` does not exist.
4. The execution receipt proves every canonical step completed, including
   `resource_guard`, `workers`, `remote_verify`, `final_resource_guard`,
   `package_download`, `safe_extract`, and `local_verify`.
5. Run the frozen, safely extracted local verifier again, independently of
   the receipt:

   ```bash
   ATTEMPT=experiments/qwen35_hybrid_state/<attempt-tag>
   /opt/homebrew/bin/python3.12 - "$ATTEMPT" <<'PY'
   import importlib.util
   import json
   from pathlib import Path
   import sys

   attempt = Path(sys.argv[1])
   source = attempt / "plan/local_verifier_source"
   artifact = attempt / "plan/downloaded_benchmark/artifact"
   verifier = (
       source
       / "tools/verify_qwen35_tp4_hybrid_prefix_benchmark.py"
   )
   sys.path.insert(0, str(verifier.parent))
   spec = importlib.util.spec_from_file_location(
       "strict_p1_postrun_verifier",
       verifier,
   )
   module = importlib.util.module_from_spec(spec)
   sys.modules[spec.name] = module
   spec.loader.exec_module(module)
   print(json.dumps(module.verify_run(artifact), sort_keys=True))
   PY
   ```

6. Require the independent classification to be `GO`; `ASSEMBLED`,
   `UNTRUSTED_PRODUCER_COMPLETE`, worker completion markers, and a successful
   execution receipt are not substitutes for `GO`.
7. Read `plan/downloaded_benchmark/artifact/independent_verification.json`,
   `case_rows.jsonl`, and `process_rows.jsonl`; report:
   - W1 and W2 median TTFT ratios;
   - W3 throughput ratio;
   - all-workload decode and E2E ratios;
   - initialization and peak-CUDA-reserved ratios;
   - physical snapshot bytes, logical snapshot bytes, deduplicated bytes,
     added CUDA bytes, and per-reused-token metrics.
8. Confirm cleanup independently:
   - no process whose command line contains both the exact attempt tag and
     exact remote run root remains;
   - none of those attempt PIDs appears in the remote GPU compute inventory;
   - do not require GPUs `2,4,5,6` to be globally empty, because unrelated
     jobs may start after this attempt releases them.
9. Record the result, limitations, and exact artifact paths in
   `AGENT_HANDOFF_STATE.md`.

## Operations

Status:

```bash
launchctl list com.bytedance.tinyllmforge.qwen35.strictp1.r570
tail -n 5 experiments/qwen35_hybrid_state/qwen35-tp4-strict-p1-monitor-20260806-r570/resource_samples.jsonl
```

Stop only when explicitly requested:

```bash
launchctl remove com.bytedance.tinyllmforge.qwen35.strictp1.r570
```

The monitor is bounded to 1440 samples, approximately 24 hours.

## 2026-08-07 ControlMaster recovery

At approximately `2026-08-07T01:25+08:00`, the previously reused mux
pathname disappeared:

```text
/tmp/ssh-dynamic-token-ppo-a66cd65-attempt3: missing
```

The compatibility pathname still pointed to that missing socket, while the
launchd monitor remained alive. An already-running authenticated mux was
found and verified without creating a new GPU process or restarting the
monitor:

```text
socket:          /tmp/ssh-dynamic-token-ppo-a66cd65-attempt4
master PID:      60348
remote identity: sitian@n232-195-203
```

The compatibility symlink was atomically switched to:

```text
/tmp/ssh-dynamic-token-ppo-a66cd65
  -> /tmp/ssh-dynamic-token-ppo-a66cd65-attempt4
```

An SSH round trip through the compatibility pathname passed. The next formal
monitor sample also passed, proving that the live monitor consumed the
recovered transport:

```text
sample id:      496
observed at:    2026-08-06T17:26:11.508373+00:00
classification: BLOCKED_RESOURCES

GPU 2: 68,789,731,328 free bytes, 5 compute processes
GPU 4: 50,961,842,176 free bytes, 5 compute processes
GPU 5: 84,979,744,768 free bytes, 0 compute processes
GPU 6: 57,139,003,392 free bytes, 3 compute processes
```

No launch attempt, monitor result, or monitor failure existed after sample
`496`. The fixed set was still ineligible because GPUs `2`, `4`, and `6`
had active compute processes. Do not kill those processes or restart r570 as
part of transport recovery.

## 2026-08-07 second ControlMaster recovery

The `attempt4` mux later disappeared. Formal samples `1002` through `1006`
correctly recorded `SAMPLE_FAILED` with:

```text
Connection closed by UNKNOWN port 65535
```

The monitor remained alive and did not launch while transport authority was
unavailable. The default API Kerberos cache was expired, but the existing
tmux environment identified a current file cache:

```text
KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian
principal: sitian@BYTEDANCE.COM
valid through: 2026-08-07T20:36:00+08:00
```

Using that current cache, a new persistent mux was established and verified:

```text
socket:          /tmp/ssh-dynamic-token-ppo-a66cd65-attempt5
ControlPersist:  12h
remote identity: sitian@n232-195-203
```

The compatibility symlink was atomically switched to `attempt5`. Formal
samples `1007` and `1008` then returned to `BLOCKED_RESOURCES`, proving the
live monitor recovered without restart. At sample `1008` the fixed set was:

```text
GPU 2: 68,789,731,328 free bytes, 5 compute processes
GPU 4: 50,961,842,176 free bytes, 5 compute processes
GPU 5: 84,979,744,768 free bytes, 0 compute processes
GPU 6: 57,139,003,392 free bytes, 3 compute processes
```

There was still no launch attempt, monitor result, or monitor failure.
