# TinyLLMForge Sitian Remote Scratch Design

**Date:** 2026-08-18

**Status:** Approved design; implementation pending written-spec review

## Goal

Move cache-heavy TinyLLMForge development work off the nearly full macOS data
volume and onto the large remote filesystem owned by:

```text
host:          sitian@10.232.195.203
storage root:  /data00/home/sitian
```

The authoritative source and Git checkout remain local:

```text
/Users/bytedance/Desktop/TinyLLMForge
  -> /Users/bytedance/dev/TinyLLMForge
```

The remote host supplies isolated scratch source trees, temporary directories,
Python bytecode caches, test outputs, and review logs. It does not become Git
authority.

## Motivation and Existing Migration

On August 18, 2026, `/System/Volumes/Data` was at 96% capacity with about
39 GiB available. TinyLLMForge work from the current session had created:

```text
77 top-level temporary items
19,304 regular files
343,541,747 content bytes
367.6 MiB allocated size
```

Those items were copied without a local intermediate archive to:

```text
/data00/home/sitian/tinyllmforge-temp/2026-08-18/command-timeline-01a013cc
```

The local and remote trees matched in top-level item count, regular-file
count, content bytes, and the following sorted per-file SHA-256 manifest:

```text
3a06439f91c3cc2a8cad87d94039796486636f06835fbc3bb9779fa904b2f96d
```

The local 77-item source set was removed only after that verification.

## Selected Approach

Create a task-isolated, non-Git remote scratch workspace under:

```text
/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818
```

Populate the initial source tree by streaming the local committed `HEAD`
archive directly over SSH. This avoids a local archive and excludes untracked
review packages by construction.

After the initial snapshot, synchronize only explicitly named files needed by
the active task. Never use a whole-tree `rsync --delete`, never infer a broad
untracked-file set, and never update the retired adaptive-ngram checkout.

Cache-heavy CPU tests, compile checks, review commands, and generated logs run
on the remote host. The local checkout remains responsible for editing,
focused diffs, Git commits, and pushes.

## Considered Alternatives

### Continue using `/private/tmp`

Rejected. `/tmp` resolves to `/private/tmp`, which is on the same nearly full
macOS data volume. Moving names within that volume does not solve capacity
pressure.

### Update `/data00/home/sitian/tllm/TinyLLMForge` in place

Rejected. That 2 GiB directory is a historical source mirror without `.git`
metadata and contains old logs and result directories. In-place updates would
mix current source identity with retained historical state.

### Archive outputs remotely but keep running tests locally

Rejected as the default. It would move final logs but still create local
bytecode, pytest, compiler, and review caches. Local read-only checks remain
allowed only when they are explicitly configured not to write bytecode or
cache data.

## Workspace Layout

The remote task root is:

```text
/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/
  source/
  tmp/
  pycache/
  cache/
  logs/
  receipts/
  env/
```

Responsibilities are:

- `source/`: streamed local `HEAD` plus explicit active-task file updates;
- `tmp/`: `TMPDIR`, `TMP`, and `TEMP`;
- `pycache/`: `PYTHONPYCACHEPREFIX`;
- `cache/`: `XDG_CACHE_HOME` and any tool-specific cache directories;
- `logs/`: complete stdout and stderr from remote validation commands;
- `receipts/`: source identity, command, exit status, and output checksums;
- `env/`: optional task-local environment only when the implementation plan
  identifies a missing dependency.

No cache or result path may resolve outside the task root unless an existing
read-only model or runtime dependency is intentionally reused.

## Source Transport Contract

### Initial snapshot

The initial remote source is created from local committed `HEAD` using a
streaming archive:

```text
git archive HEAD
  -> SSH transport
  -> remote tar extraction into source/
```

The transport must disable macOS extended attributes and AppleDouble metadata.
The completed source tree must contain no `._*` files.

### Incremental updates

Incremental synchronization uses an explicit path allowlist and preserves
repository-relative paths. It must not include:

- `.git/`;
- `.superpowers/sdd/*review-package.diff`;
- `__pycache__/`, `.pytest_cache/`, or `*.pyc`;
- local logs, PIDs, raw remote output, model data, or generated results;
- `artifacts/`, experiment output trees, or source archives; or
- files from `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.

Every sync records the local `HEAD`, the explicit path list, and SHA-256 values
for transferred files in `receipts/`.

## Remote Execution Contract

Every cache-producing remote command receives:

```text
TMPDIR=<task-root>/tmp
TMP=<task-root>/tmp
TEMP=<task-root>/tmp
PYTHONPYCACHEPREFIX=<task-root>/pycache
XDG_CACHE_HOME=<task-root>/cache
PYTHONDONTWRITEBYTECODE=0
```

Additional framework caches, when applicable, also remain under
`<task-root>/cache`.

Complete command output is redirected to `<task-root>/logs`. The local caller
returns only a bounded status summary and a short failure tail, preventing
large validation logs from being duplicated into local tool-output storage.

The existing remote runtime may be used read-only after a dependency
preflight. Missing dependencies are reported as environment failures. The
workflow must not mutate a shared remote environment implicitly; any required
installation belongs in `<task-root>/env` and must be specified in the
implementation plan.

## Local Execution Contract

The normal workflow does not run cache-heavy Python tests locally.

Any unavoidable local Python probe must set:

```text
PYTHONDONTWRITEBYTECODE=1
```

It must also disable the test framework's persistent cache and must not set
`PYTHONPYCACHEPREFIX` to `/tmp` or `/private/tmp`. Local output is kept
bounded and is not redirected into a new local log file.

Small operating-system resources that must remain local, such as an SSH Unix
control socket, are allowed. They may not be used as storage for source
archives, test caches, or logs.

## SSH and Failure Semantics

The preferred transport reuses an already healthy SSH route when available.
The file Kerberos cache remains:

```text
FILE:/Users/bytedance/krb5cc_sitian
```

New SSH connections may fail intermittently with:

```text
Connection closed by UNKNOWN port 65535
```

Transport commands therefore use bounded retries and fail without deleting
the last verified remote source or local authoritative files. A failed
transfer writes only into a uniquely named remote staging directory. The
staging directory is promoted atomically after verification.

No workflow may refresh credentials, terminate unrelated SSH sessions, or
kill remote jobs without separate authority.

## GPU and Gate Boundary

This design authorizes remote storage, source staging, and CPU-only
development validation. It does not authorize:

- GPU allocation;
- model loading;
- a source-bound performance gate;
- resuming an interrupted gate;
- terminating an existing GPU process; or
- selecting a runtime optimization.

Those operations retain their existing separate authorization and evidence
requirements.

## Verification and Acceptance Criteria

The implementation is complete only when:

1. the remote task root exists on `/data00/home/sitian`;
2. the initial `HEAD` snapshot is streamed without a local archive;
3. source identity and content checksums are recorded remotely;
4. no AppleDouble `._*` files exist in the remote source;
5. remote temp, bytecode, cache, log, and receipt paths remain inside the task
   root;
6. an approved focused CPU test runs remotely and records its receipt;
7. the same operation creates no new TinyLLMForge cache or log tree under
   `/tmp` or `/private/tmp`;
8. review-package diffs and unrelated local files are absent remotely;
9. local Git status changes only by explicitly intended versioned files; and
10. the workflow can be resumed after an SSH failure without deleting the
    last verified source snapshot.

After these checks, Task 5 review and Tasks 6-8 may continue using the remote
scratch workflow. Remote GPU execution remains blocked pending separate
authorization.
