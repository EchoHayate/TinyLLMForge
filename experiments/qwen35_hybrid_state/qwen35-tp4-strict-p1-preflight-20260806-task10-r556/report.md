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
e2afefc1540184d7724a6e9a1027d4ad95e8de4fcfc99f18e17ff3de43571578
```

The source identity matches both the r551 local readiness package and the
earlier r554 preflight.

## Fixed GPU Observations

| GPU | Free bytes | Active compute processes | Decision |
| ---: | ---: | ---: | --- |
| 2 | 68,789,731,328 | 5 | blocked |
| 4 | 50,963,939,328 | 5 | blocked |
| 5 | 84,979,744,768 | 0 | eligible |
| 6 | 63,191,384,064 | 3 | blocked |

The listed processes pre-existed this preflight. This operation did not start,
stop, signal, or mutate any remote process and did not allocate GPU memory.

## Transport Boundary

The existing authenticated ControlMaster
`/tmp/ssh-dynamic-token-ppo-a66cd65` was checked with `ssh -O check` and
reused only as the transport for the canonical `run_preflight()` source
bundle, prerequisite, GPU-selection, and classification logic.

The first local attempt for tag r555 omitted the canonical `bash -lc` and
`shlex.quote` wrapping around the remote Python command. The remote shell
therefore rejected the split command with a syntax error before the GPU query
ran. That incomplete local directory contains only the deterministic source
tar and is not a preflight result. The wrapping was validated with the
read-only command `print("ok")`, and r556 then completed the canonical
read-only query.

## Stop Boundary

Because the result is not `READY`:

- no remote directory was created;
- no source bundle was uploaded;
- no worker, model, CUDA workload, or benchmark was launched;
- no execution authorization or receipt was created; and
- strict-P1, calibration, and P2 remain unexecuted.

The next valid action is another separately approved read-only resource
preflight after GPUs `2`, `4`, and `6` have no active compute processes.
