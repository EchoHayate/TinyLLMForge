# Qwen3.5 TP4 Strict-P1 Read-Only Resource Preflight

## Decision

`BLOCKED_RESOURCES`

The canonical current-source strict-P1 preflight queried the fixed GPU set
`2,4,5,6`. The policy requires at least 25 GiB free on every selected GPU and
no active compute process. Although every selected GPU had more than 25 GiB
free, GPUs `2`, `4`, and `6` had active compute processes, so the preflight
correctly refused authorization.

```text
authorized:            false
remote_query_executed: true
remote_path_created:   false
```

## Bound Source

```text
source tree:
88ebda203decaf26a721f6299a5c9ed08b648841feb0b972f1395d6cc609eb9c

source tar:
8f807c0e51010fddd0bd474670ae41c09beda604c8247f672fc047f533c53a5d

benchmark_preflight.json:
6ee9690915c0053a2aa92cb7e8af021b2249d5f0145a0211ec569e45d5309092
```

The source identity matches the r551 local readiness package.

## Fixed GPU Observations

| GPU | Free bytes | Active compute processes | Decision |
| ---: | ---: | ---: | --- |
| 2 | 68,789,731,328 | 5 | blocked |
| 4 | 50,963,939,328 | 5 | blocked |
| 5 | 84,979,744,768 | 0 | eligible |
| 6 | 63,191,384,064 | 3 | blocked |

The listed processes pre-existed this preflight. This operation did not start,
stop, signal, or mutate any remote process and did not allocate GPU memory.

## SSH Recovery Boundary

The first canonical invocation using a fresh SSH connection failed before the
GPU query with:

```text
Connection closed by UNKNOWN port 65535
```

Local diagnosis showed expired Kerberos host tickets dated July 29, 2026, and
`kinit -R` could not renew them because no renewable TGT was present. An
already-running ControlMaster for `sitian@10.232.195.203` was independently
checked with `ssh -O check` and a read-only `true` command. The successful
preflight reused that existing authenticated transport while preserving the
canonical `run_preflight()` source-bundle, prerequisite, GPU-selection, and
classification logic.

## Stop Boundary

Because the result is not `READY`:

- no remote directory was created;
- no source bundle was uploaded;
- no worker, model, CUDA workload, or benchmark was launched;
- no execution authorization or receipt was created; and
- strict-P1, calibration, and P2 remain unexecuted.

The next valid action is another separately approved read-only resource
preflight after GPUs `2`, `4`, and `6` have no active compute processes.
