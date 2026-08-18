# TinyLLMForge Sitian Remote Transaction Engine Design

**Date:** 2026-08-18

**Status:** Approved architecture after Task 2 review escalation

## Problem

The first Task 2 implementation proved the transport, policy, and basic
remote workflow, but three independent review rounds found that an inline
shell transaction cannot make all of these guarantees simultaneously:

- `init` and `sync` must never leave the verified `source/` missing or
  partially updated;
- an SSH response loss after a remote commit must be distinguishable from a
  failed mutation;
- lock cleanup must survive process exit and signals without leaving a stale
  lock;
- source writes must not follow a parent symlink between a pathname check and
  the mutation;
- one strict commit predicate must govern cleanup, same-nonce reentry, and
  ambiguous-result confirmation; and
- test-only failure injection must not be activatable by ambient production
  environment variables.

Continuing to add shell flags, traps, and rollback markers would preserve the
same check/use and cleanup architecture that caused the findings. Task 2
therefore changes implementation architecture while preserving its public
CLI and remote-scratch policy.

## Selected Architecture

Create a dependency-free Python remote transaction helper:

```text
tools/sitian_remote_transaction.py
```

The local controller remains:

```text
tools/sitian_remote_scratch.py
```

The controller owns local validation, SSH transport, bounded output, and CLI
parsing. The transaction helper owns every mutation below the remote task
root. The helper is included in committed `HEAD`, extracted into each initial
staging tree, and invoked from that staging tree for `init`. A later change to
the transaction helper itself requires a fresh `init`; ordinary task files
continue to use incremental `sync`.

## Complete-Generation Commit Model

Neither `init` nor `sync` modifies the active `source/` tree in place.

Both operations create a unique private generation:

```text
<remote-root>/.transactions/<nonce>/generation/
```

For `init`, the streamed `git archive HEAD` is extracted into the private
generation with the shared forbidden policy.

For `sync`, the helper clones the active `source/` into the private generation
with symlinks preserved, then applies only the validated explicit delta inside
that private tree.

Before promotion, the helper writes canonical commit metadata inside the
generation:

```text
generation/.tinyllmforge-scratch/commit.json
generation/.tinyllmforge-scratch/source-files.sha256
generation/.tinyllmforge-scratch/explicit-paths.txt
```

The embedded commit metadata contains:

```text
schema_version
operation
nonce
source_head
explicit_paths
explicit_path_sha256
source_manifest_sha256
source_file_count
created_at_unix_ns
```

The helper verifies the generation and embedded metadata before promotion.
The embedded metadata is the canonical commit truth because it moves with the
generation in the same atomic operation as the source contents.

## Atomic Source Switch

The remote Linux host must support:

```text
renameat2(..., RENAME_EXCHANGE)
```

The helper calls `renameat2` through `ctypes` using directory file
descriptors. For the first initialization, when `source/` does not exist, a
single directory-fd-relative `rename` promotes the generation.

For replacement:

```text
source/      <atomic exchange>      .transactions/<nonce>/generation/
```

At every observable instant, `source/` names either the previous complete
generation or the new complete generation. A process exit before the exchange
leaves the previous source active. A process exit after the exchange leaves a
new source containing its canonical embedded commit receipt, so ambiguous
transport confirmation can classify the operation as committed.

The old generation remains at the transaction path after the exchange and is
deleted only as post-commit garbage collection. Failure to delete it is
reported as residue but does not invalidate or roll back the committed source.

If `RENAME_EXCHANGE` is unavailable, crosses filesystems, or fails its
preflight, the operation fails before replacing `source/`. There is no
two-rename fallback.

## Locking and Ownership

All `source/` writers use one advisory lock:

```text
<remote-root>/.transactions/source.lock
```

The helper opens the trusted `.transactions` directory with
`O_DIRECTORY | O_NOFOLLOW`, opens the lock file relative to that directory
with `O_CREAT | O_RDWR | O_NOFOLLOW`, and acquires `fcntl.flock(LOCK_EX)`.

The kernel releases the lock when the helper exits, including ordinary signal
termination. The lock file may persist, but a persistent unlocked file is not
a stale lock and does not block later operations. No lock-directory deletion,
owner-file rename, or stale-lock recovery protocol is needed.

The lock covers:

- recovery and garbage-collection inspection;
- sync generation cloning;
- generation verification;
- embedded commit metadata creation;
- atomic source promotion or exchange; and
- external receipt materialization.

## Non-Following Filesystem Operations

The helper opens the remote root and active source using directory file
descriptors with `O_DIRECTORY | O_NOFOLLOW`.

Explicit sync paths are traversed component by component with `os.open` and
`dir_fd`:

- every existing parent must open as a real directory with
  `O_DIRECTORY | O_NOFOLLOW`;
- a final existing object must be a real regular file opened with
  `O_NOFOLLOW`;
- missing or non-directory parent topology fails with a message requiring a
  full `init`;
- symlink-to-directory and symlink-to-file topology is never followed; and
- final replacement uses `os.replace` with source and destination directory
  file descriptors.

The active `source/` is never mutated through an unchecked absolute pathname.
The complete-generation exchange also uses root-relative directory
descriptors.

## Commit Predicate and Receipt Recovery

One function defines committed truth:

```python
read_committed_generation(
    remote_root: Path,
    *,
    expected_nonce: Optional[str] = None,
    expected_operation: Optional[str] = None,
    expected_head: Optional[str] = None,
    expected_paths: Optional[Sequence[str]] = None,
) -> CommitReceipt
```

It requires:

- `source/` to be a real directory and not a symlink;
- every embedded metadata file to be a real regular file and not a symlink;
- canonical JSON with the exact schema version;
- exact nonce, operation, source head, and explicit path list;
- explicit path hashes matching the committed source;
- the source manifest digest matching the embedded manifest; and
- the shared forbidden policy to pass.

The same function is used for:

- normal post-exchange success;
- same-nonce reentry;
- read-only confirmation after an ambiguous SSH result;
- `status`; and
- reconstruction of external receipts.

External files under `receipts/` are detached mirrors, not commit truth. They
are written atomically from the embedded receipt. If the helper exits after
the source exchange but before those mirrors are complete, read-only
confirmation acquires the same lock, validates the embedded receipt, recreates
the mirrors, and returns success.

Receipt symlinks, incomplete receipt sets, mismatched explicit paths, or
mismatched hashes fail closed.

## Signal and Failure Handling

The Python helper uses a single `try/finally` ownership scope.

Signal handlers for `SIGHUP`, `SIGINT`, and `SIGTERM` raise a
`TransactionInterrupted` exception carrying the conventional exit code. The
helper does not perform rollback from the signal handler.

Before the atomic exchange, cleanup removes only the current nonce's private
generation. The active source remains untouched.

After the exchange, the operation is committed because canonical metadata is
already inside the active source. Cleanup may remove the old generation but
must never exchange back merely because detached receipt materialization or
transport response failed.

## Test-Only Fault Injection

Production command builders never read a failure-point environment variable.

Internal transaction functions accept:

```python
fault_injector: Optional[Callable[[str], None]] = None
```

Unit tests pass an explicit injector. CLI entry points always pass `None`.
Remote integration fixtures may invoke a dedicated `--testing-fault-point`
flag only when paired with an exact repository-owned testing token generated
by the test harness; the production controller never supplies either option.

Required fault points are:

```text
after_lock
after_generation_ready
after_embedded_receipt
before_exchange
after_exchange
before_external_receipts
after_external_receipts
before_old_generation_cleanup
```

## Public Interface Compatibility

The following public interfaces remain:

```text
ScratchConfig
run_with_retries(...)
initial_snapshot_commands(config)
incremental_sync_commands(config, paths)
CLI: init
CLI: sync -- <explicit paths>
CLI: status
```

Successful local output remains bounded. No local archive, cache, review log,
validation log, or Python bytecode tree is created.

Task 3 may continue to rely on:

```text
<remote-root>/source
<remote-root>/logs
<remote-root>/receipts
```

## File Boundaries

```text
tools/sitian_remote_transaction.py
  Remote lock, generation preparation, dirfd traversal, atomic exchange,
  embedded receipt, strict commit predicate, detached receipt recovery.

tools/sitian_remote_scratch.py
  Local policy, SSH transport, CLI, bounded output, invocation of the remote
  helper.

tools/test_sitian_remote_transaction.py
  Pure and fixture tests for transaction primitives and fault boundaries.

tools/test_sitian_remote_scratch.py
  Existing policy/transport regressions plus controller integration tests.
```

The 971-line controller is reduced by deleting the inline init/sync shell
transaction state machines after the helper is integrated.

## Verification Requirements

The architecture is accepted only when remote tests prove:

1. the existing 25 Task 1/2 tests remain green or are replaced by stricter
   tests covering the same behavior;
2. `RENAME_EXCHANGE` preflight succeeds on the Sitian task filesystem;
3. every pre-exchange fault leaves the previous source and detached receipts
   byte-for-byte unchanged;
4. every post-exchange fault is confirmed as committed from embedded metadata
   and reconstructs exact detached receipts;
5. a second signal or exception during cleanup does not leave a held lock;
6. same-nonce reentry and ambiguous confirmation use the same strict commit
   predicate;
7. receipt symlinks and mismatched explicit path lists fail closed;
8. a concurrent `init` and `sync` serialize on the same `flock`;
9. a remote parent-symlink race cannot write outside the private generation or
   active source;
10. forbidden paths and AppleDouble files remain absent;
11. the real verified source and pre-existing receipts remain unchanged by
    failure-injection tests;
12. no matching local `/tmp` or `/private/tmp` item is created; and
13. no GPU, model, or performance gate is run.

After this architecture passes an independent task review, the remote-scratch
workflow may proceed to Task 3.
