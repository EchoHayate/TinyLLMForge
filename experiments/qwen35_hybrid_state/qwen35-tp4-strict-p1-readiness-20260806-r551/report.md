# Qwen3.5 TP4 Strict-P1 Local Readiness Package

## Decision

`READY_FOR_RESOURCE_PREFLIGHT`

This package freezes the local inputs required before a fresh strict-P1
resource preflight. It does not authorize or perform the preflight.

## Bound Inputs

```text
benchmark-owned source files: 91
source tree SHA256:
88ebda203decaf26a721f6299a5c9ed08b648841feb0b972f1395d6cc609eb9c

benchmark source tar SHA256:
8f807c0e51010fddd0bd474670ae41c09beda604c8247f672fc047f533c53a5d

correctness prerequisite SHA256:
35b4bf092d5c4c84746b88ecd88b32bf14357a21d2923336d62653186cf352f8

correctness evidence inventory:
18 files, 198222 bytes

correctness evidence inventory SHA256:
4fd3253b38b43609455f3794e037322ad2c98a4fc58bb2fd984b4cf69222c2b1

model manifest SHA256:
3e650a908234771c3cf1ac4e20c4d38fe69982efedaf4a3e631ad0b14aad7dd0

workload manifest SHA256:
71909b825d1a8d162604f6cc3d34ad413b2af6c191425ec007859715a4d084e3

current-source Gate-1 audit SHA256:
9c1cd090753a064ebc81b68069f1faffc8b821d65bfad7732d11a24e0731573f

local readiness audit SHA256:
6c0c06a3b3e0e8dc4d3f9b6d9eae0bc2af35cc231b1aa2d63ddf1085900c59fa
```

The production schema-v1 contract accepts the correctness prerequisite with:

```text
classification: PASS
authorized: true
reasons:    []
```

The top-level prerequisite JSON uses paths relative to its own directory.
Therefore `correctness_prerequisites.json` and the sibling `prerequisites/`
tree are one indivisible bundle. This package contains all 18 referenced and
supporting evidence files, copied byte-for-byte from the accepted prerequisite
authority package. A top-level-only copy is invalid and is classified
`BLOCKED_CORRECTNESS`.

The independent local readiness audit reparses all 23 package JSON files,
recomputes all bound top-level hashes and the nested evidence inventory,
checks the production prerequisite validator, verifies the deterministic tar
against all 91 current owned source files, confirms the 70-case canonical
matrix, and checks that every execution authorization remains `false`. Its
classification is `PASS`.

The canonical matrix contains 70 cases. The fixed GPU set is `2,4,5,6`, with
a minimum of 25 GiB free on every GPU and no active compute process permitted.

## Authorization Boundary

The package explicitly records:

```text
execution_authorized:             false
ssh_authorized:                   false
gpu_authorized:                   false
remote_path_creation_authorized:  false
```

No SSH, SCP, remote query, remote directory, CUDA import, GPU operation,
process mutation, launch plan, execution authorization, or receipt was
created.

## Next Gate

After separate approval, run only a fresh read-only preflight:

```bash
PYTHONPATH=/Users/bytedance/dev/TinyLLMForge-adaptive-ngram/tools:/Users/bytedance/dev/TinyLLMForge-adaptive-ngram \
  /opt/homebrew/bin/python3.12 \
  tools/run_qwen35_tp4_hybrid_prefix_benchmark_remote.py \
  preflight \
  --run-tag qwen35-tp4-strict-p1-preflight-20260806-task10-r552 \
  --prerequisites \
  /Users/bytedance/dev/TinyLLMForge-adaptive-ngram/experiments/qwen35_hybrid_state/qwen35-tp4-strict-p1-readiness-20260806-r551/correctness_prerequisites.json
```

The preflight itself regenerates the deterministic source bundle and must
produce the same source-tree SHA. If it returns `BLOCKED_CORRECTNESS`,
`BLOCKED_RESOURCES`, an SSH error, or any source identity drift, stop without
creating a remote path or launching a worker.

Only a `READY` result permits inspection of a future strict-P1 smoke plan.
