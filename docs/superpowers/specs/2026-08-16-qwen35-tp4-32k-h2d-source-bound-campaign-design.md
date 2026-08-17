# Qwen3.5 TP4/32K Focused-H2D Source-Bound Campaign Design

## Status

The user approved designing and implementing the local source-bound campaign
runner plus provenance and authorization contract on 2026-08-16.

This approval is local-only. It does not authorize SSH, GPU, NCCL, remote
directory creation, model execution, or the four-cell campaign.

## Goal

Turn the existing focused-H2D worker, gate, and verifier into a reconstructable
future campaign with one authorization-enforced local executor boundary but no
built-in SSH/GPU transport.

The local contract must:

1. bind the complete producer and verifier source closure;
2. create a deterministic source inventory and tar archive;
3. freeze the four cells, repetitions per cell, GPU indices, ports, model
   paths, output paths, and remote root;
4. require the exact future execution authorization text;
5. make authorization canonical-plan-SHA bound, nonce bound, and single use;
6. reject source, plan, path, host, cell, repetition, GPU, port, or model
   tampering;
7. produce a command plan that contains no built-in subprocess/SSH execution
   path;
8. keep real execution separately unauthorized.

## Frozen Boundary

The only remote identity accepted by the local plan is:

```text
host: sitian@10.232.195.203
remote root: /dev/shm/sitian/tllm-qwen35-target-qwen3-draft-20260815
```

`/data00` may appear only in pre-existing read-only Python or model paths. New
experiment outputs, packages, plans, authorizations, receipts, and artifacts
must be under the frozen `/dev/shm` root.

The exact future authorization text is:

```text
允许只运行一个 source-bound focused-H2D four-cell campaign
```

Local implementation and validation must not consume that future execution
authorization.

## Components

### Source Bundle

`tools/qwen35_tp4_32k_h2d_slot_reuse_source_bundle.py` owns a conservative,
explicit source-root inventory. It includes the complete `tinyvllm` Python
package and the focused producer/verifier tools plus their frozen 32K authority
helpers. Files are regular, non-symlink `.py` sources. Inventory order,
per-file SHA-256, tree SHA-256, and tar SHA-256 are deterministic.

The tree digest is length-delimited over relative path and file bytes. The tar
contains exactly the inventory paths in sorted order and rejects links or
non-regular members.

### Campaign Plan

`tools/qwen35_tp4_32k_h2d_slot_reuse_campaign.py` creates and validates one
local immutable plan. The plan freezes:

- source inventory/tree/tar identities;
- checkpoint manifest identity;
- four exact cells: `observe:b1`, `observe:b4`, `control:b1`, `control:b4`;
- one positive repetition count applied to every cell;
- TP4 GPU indices and two distinct safe ports;
- exact worker/gate/verifier entry points;
- local and remote paths;
- the future authorization text;
- claim boundaries and all non-execution flags.

The plan exposes only inert command descriptors. It does not call subprocess,
SSH, rsync, scp, torch, CUDA, or NCCL.

Its CLI exposes only:

```text
prepare   build a local source bundle and immutable plan
validate  revalidate an existing local plan and its bound files
```

The CLI freezes GPU indices to `0,1,2,3`, creates no authorization, and invokes
no executor callback. Its machine-readable classifications are
`PREPARED_LOCAL_ONLY` and `VALID_LOCAL_PLAN`.

### Single-Use Authorization

`tools/qwen35_tp4_32k_h2d_slot_reuse_campaign_authorization.py` writes and
validates a canonical authorization payload bound to the exact plan SHA,
source tree/tar SHA, checkpoint manifest SHA, cells, repetitions, GPU indices,
ports, remote run path, and nonce.

Consumption is rename-first and writes an immutable consumed record. Reuse,
tampering, unsafe nonce values, existing outputs, or authorization text
mismatch fail closed.

### Authorized Executor Boundary

`tools/qwen35_tp4_32k_h2d_slot_reuse_campaign_executor.py` is the only local
execution boundary. It validates the plan, atomically consumes the single-use
authorization, and only then invokes an explicitly injected command runner.

The module contains no subprocess, SSH, socket, torch, CUDA, or NCCL transport.
An invalid or missing authorization invokes no callback. Once authorization is
consumed, callback failure does not restore it, so a partial attempt cannot be
silently replayed.

## Validation

Dependency-light tests must prove:

- deterministic complete source inventory and tar membership;
- dynamic focused worker and verifier dependencies are included;
- symlinks and unsafe paths are rejected;
- plan constants and output roots are frozen;
- no execution-capable import or call exists;
- every execution field is false;
- authorization is exact-text, plan-bound, tamper-resistant, and single-use;
- the injected command runner is called only after the active authorization
  disappears and the consumed record exists;
- callback failure leaves authorization consumed;
- no SSH/GPU/remote command is run during tests.

This local contract can establish only:

```text
FOCUSED_H2D_SOURCE_BOUND_LOCAL_RUNNER=ESTABLISHED
FOCUSED_H2D_CODE_ENFORCED_AUTHORIZATION=ESTABLISHED
FOCUSED_H2D_COMPLETE_PRODUCER_PROVENANCE_CONTRACT=ESTABLISHED
FOCUSED_H2D_BUILTIN_REMOTE_TRANSPORT=ABSENT
FOCUSED_H2D_GPU_DIAGNOSTIC=NOT_APPROVED
```

It cannot establish GPU causality, a production synchronization fix, TP4/32K
correctness, Phase 1 completion, or promotion.
