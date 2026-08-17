# Qwen3.5 Recurrent INT8 Task 10 Current-Source Audit

## Decision

`BLOCKED_RESOURCES`

The current source tree now has both authoritative local prerequisites:

```text
correctness prerequisite: PASS
Gate-1 audit:             PASS
```

The current source identity is:

```text
e265b3ead9d9717d92d8bc0507ac051d93ec22f8403b7929c3625ee4153ccfd7
```

Strict-P1 `GO` and recurrent-INT8 calibration `PASS` remain missing.

## Current-Source Preflight

```text
run tag:
qwen35-tp4-strict-p1-preflight-20260805-225220-task10-r549

classification:       BLOCKED_RESOURCES
authorized:           false
remote query:         executed
remote path created:  false
```

Required GPU observations:

```text
GPU 2: 68,789,731,328 free bytes, 5 compute processes
GPU 4: 50,963,939,328 free bytes, 5 compute processes
GPU 5: 84,979,744,768 free bytes, 0 compute processes
GPU 6: 63,191,384,064 free bytes, 3 compute processes
```

The complete fixed set `2,4,5,6` is therefore not eligible. No existing
process was killed or modified.

Evidence:

```text
benchmark_preflight.json
SHA256 1a67b5d5b5d568dacd398fc8505ef63db6c2a5de048b7a1b25499dec4935cf02

benchmark_source.tar
SHA256 63f6a5f89c726cf0a60327f3cee3121f3d448216325a71c3d7b8c0404539f0f0

Gate-1 audit document
SHA256 ce0ad854a7589c53dd3235a4d1f418840f1538ad164ae6e5b691654cf6a1834e
```

## Safety Boundary

The read-only SSH/GPU query ran. The blocked path created no remote directory,
staged no source, launched no worker, created no execution plan, and created
or consumed no authorization.

Strict-P1 smoke, recurrent-INT8 calibration, v2 preflight, and P2 canonical
authority remain prohibited.

Gate 1 proves local integration only. No canonical accuracy, cache, capacity,
memory, latency, throughput, decode, compression, or speed benefit is
established.

## Bounded Follow-Up Poll

Five additional read-only resource queries were made at 60-second intervals.
Every observation was identical:

```text
eligible GPU indices: [5]
GPU 2 process count:  5
GPU 4 process count:  5
GPU 5 process count:  0
GPU 6 process count:  3
```

The stable result is recorded in `resource_poll.json`. Polling stopped after
the bounded window because continued high-frequency queries would not advance
the authority chain. No remote path or process side effect occurred.

## Process Ownership

A final read-only `ps` inventory established that the blocking processes are
not owned by the current benchmark authority:

- GPU 2 includes long-running `zengjun+` inference/KV services plus one
  `sitian` diagnostic process.
- GPU 4 includes multiple `root` `/opt/tiger/test/main_model.py` processes
  plus the diagnostic process child.
- GPU 6 includes `root` and `wangmin+` model-serving processes.

Even removing the two `sitian` diagnostic processes would leave GPUs 2 and 4
ineligible. No process may therefore be safely terminated or treated as
authority-owned cleanup. The normalized evidence is in
`process_inventory.json`.

## Exact Next Action

Wait for all fixed GPUs `2,4,5,6` to become eligible, then rerun a fresh
current-source strict-P1 read-only preflight. Execute strict-P1 smoke only
after `READY`.
