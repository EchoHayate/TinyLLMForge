# Sitian Remote Transaction Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the inline shell source transaction with a remote Python generation engine that atomically switches complete verified source trees and supports strict ambiguous-result confirmation.

**Architecture:** A dependency-free Python 3.7 helper builds private complete generations, embeds canonical commit metadata, serializes every source writer with `fcntl.flock`, and promotes with directory-fd-relative `renameat2(RENAME_EXCHANGE)`. The local controller keeps policy, transport, and bounded CLI output while using one strict helper commit predicate for success, reentry, status, and detached receipt recovery.

**Tech Stack:** Python 3.7 standard library, `ctypes`, Linux `renameat2`, `fcntl.flock`, dirfd-relative `os.open`/`os.replace`, `unittest`, SSH, streaming tar.

## Global Constraints

- The authoritative date is Tuesday, August 18, 2026.
- Modify only `/Users/bytedance/Desktop/TinyLLMForge`, whose physical path is `/Users/bytedance/dev/TinyLLMForge`.
- Never read from, modify, stage, commit, package, or synchronize `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- The local checkout remains the only Git authority.
- The remote host is exactly `sitian@10.232.195.203`.
- The remote task root is exactly `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818`.
- Do not modify the historical non-Git mirror `/data00/home/sitian/tllm/TinyLLMForge`.
- Do not create TinyLLMForge source archives, pycache trees, pytest caches, review logs, or validation logs under local `/`, `/tmp`, or `/private/tmp`.
- Any unavoidable local Python invocation sets `PYTHONDONTWRITEBYTECODE=1`, disables persistent test caches, produces bounded output, and does not redirect to a local log.
- All cache-producing tests and compile checks run on sitian with `TMPDIR`, `TMP`, `TEMP`, `PYTHONPYCACHEPREFIX`, and `XDG_CACHE_HOME` inside the remote task root.
- Initial source staging streams `git archive HEAD` directly to SSH; no local archive file is created.
- Incremental source staging accepts only explicit repository-relative paths.
- Never use `git add -A`, broad untracked-file discovery, or `rsync --delete`.
- Never synchronize `.git/`, `artifacts/`, `experiments/`, `__pycache__/`, `.pytest_cache/`, `*.pyc`, logs, PIDs, raw remote output, source archives, or `.superpowers/sdd/*review-package.diff`.
- Disable macOS xattrs and AppleDouble metadata during transport; the remote source must contain zero `._*` files.
- A failed pre-commit transfer writes only to a unique remote generation and never deletes or replaces the last verified source.
- All `source/` writers use the same kernel-released advisory lock.
- No two-rename fallback is allowed when `RENAME_EXCHANGE` is unavailable.
- Production commands contain no ambient-environment failure checkpoint.
- Do not refresh Kerberos credentials, kill SSH sessions, terminate remote processes, allocate GPUs, load models, or run a source-bound performance gate.
- Stage only explicit intended source, test, plan, audit, progress, and handoff files.
- Every commit uses `git -c core.hooksPath=/dev/null commit`.
- Every commit ends with exactly one `Co-authored-by: TRAE CLI <noreply@bytedance.com>` trailer.
- Push every completed slice to `origin/feat/kv-sparse-attention`.

## File Map

- Create `tools/sitian_remote_transaction.py`: remote mutation authority and strict commit predicate.
- Create `tools/test_sitian_remote_transaction.py`: transaction primitive, fault-injection, concurrency, and containment tests.
- Modify `tools/sitian_remote_scratch.py`: invoke the helper and remove inline transaction shell.
- Modify `tools/test_sitian_remote_scratch.py`: preserve policy/transport behavior and test controller integration.
- Modify `.superpowers/sdd/progress.md`: record the approved architecture and per-task review state.

---

### Task 1: Transaction Primitives and Atomic Exchange

**Files:**
- Create: `tools/sitian_remote_transaction.py`
- Create: `tools/test_sitian_remote_transaction.py`

**Interfaces:**
- Produces `CommitReceipt`.
- Produces `open_directory_no_follow(path) -> int`.
- Produces `locked_remote_root(remote_root)`.
- Produces `rename_exchange(parent_fd, left, right) -> None`.
- Produces `write_embedded_receipt(generation_fd, receipt) -> None`.
- Produces `read_committed_generation(remote_root, *, expected_nonce=None, expected_operation=None, expected_head=None, expected_paths=None) -> CommitReceipt`.
- Produces `promote_generation(remote_root, generation_name, receipt, fault_injector=None) -> CommitReceipt`.

- [ ] **Step 1: Add failing lock, no-follow, exchange, and strict-receipt tests**

Add tests with these exact behaviors:

```python
class TransactionPrimitiveTests(unittest.TestCase):
    def test_flock_releases_when_holder_exits(self):
        root = self.make_root()
        first = self.start_lock_holder(root)
        first.wait_until_locked()
        first.terminate()
        self.assertEqual(first.wait(), 0)
        with transaction.locked_remote_root(root):
            pass

    def test_directory_open_rejects_symlink(self):
        root = self.make_root()
        target = root / "real"
        target.mkdir()
        (root / "link").symlink_to(target, target_is_directory=True)
        with self.assertRaises(OSError):
            transaction.open_directory_no_follow(root / "link")

    def test_exchange_never_leaves_source_missing(self):
        root = self.make_root()
        self.write_generation(root / "source", marker="old")
        self.write_generation(root / ".transactions/n1/generation", marker="new")
        transaction.rename_exchange(
            transaction.open_directory_no_follow(root),
            "source",
            ".transactions/n1/generation",
        )
        self.assertEqual(self.read_marker(root / "source"), "new")
        self.assertEqual(
            self.read_marker(root / ".transactions/n1/generation"),
            "old",
        )

    def test_strict_receipt_rejects_symlink_and_wrong_paths(self):
        root = self.make_committed_root(paths=["tools/a.py"])
        receipt = root / "source/.tinyllmforge-scratch/commit.json"
        target = root / "receipt-target.json"
        receipt.replace(target)
        receipt.symlink_to(target)
        with self.assertRaises(transaction.TransactionError):
            transaction.read_committed_generation(
                root,
                expected_nonce="n1",
                expected_operation="sync",
                expected_head="abc",
                expected_paths=["tools/a.py"],
            )
```

- [ ] **Step 2: Run the tests remotely and confirm RED**

Run under:

```bash
root=/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818
cd "$root/red-task1"
TMPDIR="$root/tmp" TMP="$root/tmp" TEMP="$root/tmp" \
PYTHONPYCACHEPREFIX="$root/pycache" XDG_CACHE_HOME="$root/cache" \
PYTHONDONTWRITEBYTECODE=1 \
python3 -m unittest -v tools/test_sitian_remote_transaction.py
```

Expected: import failure because `sitian_remote_transaction.py` does not
exist.

- [ ] **Step 3: Implement the dependency-free primitive layer**

Define these exact public types and constants:

```python
SCHEMA_VERSION = 1
RENAME_EXCHANGE = 0x2

class TransactionError(RuntimeError):
    pass

class TransactionInterrupted(TransactionError):
    def __init__(self, signum):
        super().__init__("transaction interrupted by signal {}".format(signum))
        self.signum = signum

class CommitReceipt:
    def __init__(
        self,
        *,
        operation,
        nonce,
        source_head,
        explicit_paths,
        explicit_path_sha256,
        source_manifest_sha256,
        source_file_count,
        created_at_unix_ns,
    ):
        self.operation = operation
        self.nonce = nonce
        self.source_head = source_head
        self.explicit_paths = tuple(explicit_paths)
        self.explicit_path_sha256 = dict(explicit_path_sha256)
        self.source_manifest_sha256 = source_manifest_sha256
        self.source_file_count = source_file_count
        self.created_at_unix_ns = created_at_unix_ns
```

Implement:

- `os.open(..., O_DIRECTORY | O_NOFOLLOW)` for trusted directories;
- `fcntl.flock(fd, LOCK_EX)` in a context manager;
- `ctypes.CDLL(None, use_errno=True).renameat2` with no fallback;
- canonical JSON using `sort_keys=True` and compact separators;
- dirfd-relative regular-file reads with `O_NOFOLLOW`;
- exact embedded manifest and explicit-path hash validation; and
- signal handlers that raise `TransactionInterrupted` while cleanup remains
  in one Python `finally`.

- [ ] **Step 4: Run remote primitive tests and verify GREEN**

Expected: all primitive tests pass, including a real
`RENAME_EXCHANGE` preflight on the Sitian task filesystem.

- [ ] **Step 5: Commit and push Task 1**

```bash
git add -- \
  tools/sitian_remote_transaction.py \
  tools/test_sitian_remote_transaction.py
git diff --cached --check
git -c core.hooksPath=/dev/null commit \
  -m "feat(workflow): add sitian transaction engine" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

---

### Task 2: Atomic Initial Snapshot Promotion

**Files:**
- Modify: `tools/sitian_remote_transaction.py`
- Modify: `tools/test_sitian_remote_transaction.py`
- Modify: `tools/sitian_remote_scratch.py`
- Modify: `tools/test_sitian_remote_scratch.py`

**Interfaces:**
- Consumes `promote_generation(...)` and `read_committed_generation(...)`.
- Produces helper CLI `init-commit`.
- Preserves `initial_snapshot_commands(config)`.
- Preserves controller CLI `init`.

- [ ] **Step 1: Add failing init fault-boundary and controller tests**

Cover every exact fault point:

```python
INIT_FAULT_POINTS = (
    "after_lock",
    "after_generation_ready",
    "after_embedded_receipt",
    "before_exchange",
    "after_exchange",
    "before_external_receipts",
    "after_external_receipts",
    "before_old_generation_cleanup",
)
```

For each pre-exchange point, assert old `source/` and detached receipts are
byte-identical. For each post-exchange point, assert strict confirmation
classifies the new source as committed and recreates detached receipts.

Add a controller test asserting:

```python
commands = module.initial_snapshot_commands(config)
self.assertEqual(commands["archive"], ("git", "archive", "--format=tar", "HEAD"))
self.assertIn("sitian_remote_transaction.py", commands["remote_commit"])
self.assertNotIn("SITIAN_SYNC_FAIL_POINT", commands["remote_commit"])
```

- [ ] **Step 2: Run focused remote tests and confirm RED**

Expected: missing `init-commit` helper CLI and controller still builds the
inline shell promotion.

- [ ] **Step 3: Implement private-generation init**

The remote extraction command remains a minimal shell that:

```text
creates .transactions/<nonce>/generation
extracts stdin with shared forbidden excludes
invokes generation/tools/sitian_remote_transaction.py init-commit
```

The helper:

1. acquires the global `flock`;
2. verifies the private generation and forbidden policy;
3. computes and writes embedded commit metadata;
4. validates the embedded metadata through the strict predicate;
5. promotes first source with one dirfd-relative rename or replaces an
   existing source with `RENAME_EXCHANGE`;
6. materializes detached `source-head.txt`, `source-files.sha256`, and
   transaction receipt atomically;
7. removes the old exchanged generation as post-commit garbage collection;
8. confirms success through `read_committed_generation`; and
9. leaves committed source active if response loss occurs after exchange.

Delete `_initial_promotion_command` and its shell rollback state machine.

- [ ] **Step 4: Run remote init tests and full Task 1/2 regression**

Run:

```bash
python3 -m unittest -v \
  tools/test_sitian_remote_transaction.py \
  tools/test_sitian_remote_scratch.py
```

Expected: all tests pass; root transaction residue and held locks are zero.

- [ ] **Step 5: Commit and push Task 2**

```bash
git add -- \
  tools/sitian_remote_transaction.py \
  tools/test_sitian_remote_transaction.py \
  tools/sitian_remote_scratch.py \
  tools/test_sitian_remote_scratch.py
git diff --cached --check
git -c core.hooksPath=/dev/null commit \
  -m "refactor(workflow): atomically promote sitian source" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

---

### Task 3: Complete-Generation Incremental Sync

**Files:**
- Modify: `tools/sitian_remote_transaction.py`
- Modify: `tools/test_sitian_remote_transaction.py`
- Modify: `tools/sitian_remote_scratch.py`
- Modify: `tools/test_sitian_remote_scratch.py`

**Interfaces:**
- Produces helper CLI `sync-commit` and `confirm`.
- Preserves `incremental_sync_commands(config, paths)`.
- Preserves controller CLI `sync -- <paths>` and `status`.

- [ ] **Step 1: Add failing sync generation, concurrency, and confirmation tests**

Add exact tests for:

- clone active source with symlinks preserved;
- apply explicit files only inside the private generation;
- reject remote parent symlink without writing its target;
- init and sync serialize on the same `flock`;
- same-nonce reentry uses strict embedded commit truth;
- symlinked detached or embedded receipts fail closed;
- mismatched explicit path lists fail closed;
- response loss after exchange is confirmed and external receipts are rebuilt;
- a second signal during private-generation cleanup cannot leave a held lock;
  and
- production controller commands contain no testing fault-point input.

Use an explicit callable injector:

```python
def fail_at(expected):
    def inject(actual):
        if actual == expected:
            raise transaction.InjectedFailure(actual)
    return inject
```

The controller never passes an injector.

- [ ] **Step 2: Run focused remote tests and confirm RED**

Expected: old inline `_incremental_remote_command` remains and the helper has
no `sync-commit` or `confirm` mode.

- [ ] **Step 3: Implement clone/apply/verify/exchange**

The helper:

1. acquires the same global `flock`;
2. validates the current source through the strict commit predicate;
3. clones `source/` into the nonce generation with symlinks preserved;
4. opens generation parents with dirfd + `O_NOFOLLOW`;
5. applies only explicit files from an isolated delta tree;
6. verifies the shared forbidden policy and exact explicit hashes;
7. writes embedded canonical commit metadata;
8. exchanges generation and active source atomically;
9. materializes timestamped path/hash/state detached receipts;
10. confirms success with the shared predicate; and
11. removes the old generation after commit.

The controller performs exactly one side-effecting stream attempt. On an
ambiguous result it invokes helper `confirm`, which validates embedded truth
and repairs detached receipts without replaying the mutation.

Delete `_incremental_remote_command`, its shell transaction state machine,
ambient environment failure checkpoint, and shell commit-status predicate.

- [ ] **Step 4: Run complete remote tests and real non-mutating acceptance**

Run both test files remotely. Record before/after fingerprints for the real
verified source and pre-existing receipts. The fault-injection fixtures use a
separate remote test root and must leave:

```text
incoming_generations=0
held_locks=0
outside_writes=0
```

Expected: all tests pass and real fingerprints are unchanged.

- [ ] **Step 5: Commit and push Task 3**

```bash
git add -- \
  tools/sitian_remote_transaction.py \
  tools/test_sitian_remote_transaction.py \
  tools/sitian_remote_scratch.py \
  tools/test_sitian_remote_scratch.py
git diff --cached --check
git -c core.hooksPath=/dev/null commit \
  -m "refactor(workflow): atomically sync sitian generations" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

---

### Task 4: Real Acceptance and Task 2 Closure

**Files:**
- Modify: `.superpowers/sdd/progress.md`
- Create: `docs/superpowers/audits/2026-08-18-tinyllmforge-sitian-remote-scratch-audit.md`

**Interfaces:**
- Consumes controller `init`, `sync`, and `status`.
- Produces final Task 2 remote receipts and exact Task 3 resume boundary.

- [ ] **Step 1: Record local and remote pre-acceptance state**

Record bounded local matching temporary paths, Git HEAD, remote source
fingerprint, detached receipt fingerprint, transaction residue, and lock
availability.

- [ ] **Step 2: Run one real `init` at current committed HEAD**

Run locally with `PYTHONDONTWRITEBYTECODE=1`; the archive streams directly to
Sitian. Verify:

```text
source_head=<current HEAD>
forbidden=0
appledouble=0
embedded_receipt=valid
detached_receipts=valid
transaction_residue=0
lock_available=true
```

- [ ] **Step 3: Run one explicit two-file `sync`**

Synchronize:

```text
tools/sitian_remote_transaction.py
tools/test_sitian_remote_transaction.py
```

Verify exact path/hash receipts and strict confirmation. Do not run a GPU,
model, or performance command.

- [ ] **Step 4: Run final remote suites**

Run:

```bash
python3 -m unittest -v \
  tools/test_sitian_remote_transaction.py \
  tools/test_sitian_remote_scratch.py
```

Expected: all tests pass with caches inside the remote task root.

- [ ] **Step 5: Perform independent full Task 2 review**

Place the review package only under:

```text
/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/receipts/reviews/
```

The reviewer must verify every requirement in the transaction-engine design,
all prior review findings, the real receipts, and the no-local-write boundary.
Task 2 closes only with no Critical or Important finding.

- [ ] **Step 6: Update audit and progress, commit, and push**

Stage only the audit and progress files:

```bash
git add -- \
  docs/superpowers/audits/2026-08-18-tinyllmforge-sitian-remote-scratch-audit.md \
  .superpowers/sdd/progress.md
git diff --cached --check
git -c core.hooksPath=/dev/null commit \
  -m "docs(workflow): close sitian transaction audit" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

After the push, resume the original remote-scratch Task 3. Cache-producing
tests and review commands continue to run only through the Sitian scratch
workflow.
