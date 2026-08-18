# TinyLLMForge Sitian Remote Scratch and Transaction Audit

- Date: 2026-08-18
- Authoritative checkout: `/Users/bytedance/Desktop/TinyLLMForge`
- Physical checkout: `/Users/bytedance/dev/TinyLLMForge`
- Branch: `feat/kv-sparse-attention`
- Final transaction head: `52626becb3250ea9e5f631901e60367ff5622339`
- Remote host: `sitian@10.232.195.203`
- Remote task root:
  `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818`

## Conclusion

The Sitian source-staging transaction engine is complete and pushed. The
final independent whole-slice gate is:

```text
PASS
Critical 0
Important 0
Minor 0
```

The real remote source is bound to commit
`52626becb3250ea9e5f631901e60367ff5622339`, contains 1,302 files, passes
strict receipt confirmation, and has no forbidden cache material,
AppleDouble files, transaction residue, or held transaction lock.

All generated validation caches, receipts, packages, and review reports used
for the final workflow are under the mounted remote task root. No source
archive was written locally.

## Implemented Transaction Boundary

The final controller and helper provide:

- one side-effecting `init-commit` attempt;
- status-255-only read-only controller confirmation;
- one side-effecting streamed `sync-commit` attempt;
- complete private generations and atomic source promotion;
- one shared kernel-released source lock;
- directory-fd-relative no-follow traversal;
- strict embedded commit truth and detached receipt mirrors;
- pre-exchange init and sync receipt topology validation;
- fail-closed same-nonce sync receipt handling;
- committed-success preservation after init or sync exchange;
- committed-success preservation through managed signals, cleanup, receipt
  publication, handler teardown, and stdout;
- exact init reentry that validates source identity and receipt topology
  before publishing its committed marker;
- no rollback, exchange-back, or two-rename fallback; and
- Python 3.7 compatibility.

## Commits and Push State

The final review range is `4b10fa1..52626be`. The final corrective commits
are:

```text
915bdd2054b2e8be05bf5a735ec1376ed6cb208d
  fix(workflow): close transaction receipt gaps

52626becb3250ea9e5f631901e60367ff5622339
  fix(workflow): preserve committed init outcomes
```

Both commits are pushed to:

```text
https://github.com/EchoHayate/TinyLLMForge.git
refs/heads/feat/kv-sparse-attention
```

## TDD and Regression Evidence

The final review findings were closed through explicit RED to GREEN cycles:

```text
original three findings:
  RED: 5 methods, 10 expected failures
  focused GREEN: 5 passed
  full GREEN after first fix: 105 passed

fresh init post-commit boundary:
  RED: 5 methods, 16 expected failures, 0 errors
  focused GREEN: 5 passed
  first full GREEN: 110 passed

exact init reentry boundary:
  RED: 2 methods, 6 expected failures
  focused GREEN: 2 passed

final validation-root suite:
  111 passed in 6.195s

final real-source suite:
  111 passed in 6.830s
```

The final suites cover 82 transaction-helper tests and 29 controller tests.

## Real Remote Acceptance

The final explicit sync receipt is:

```text
/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/
  receipts/sync-1787072562-9376-1787072562374745000.sha256
```

The final acceptance receipt is:

```text
/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/
  receipts/acceptance/final-suite-52626be.txt
```

It records:

```text
source_head=52626becb3250ea9e5f631901e60367ff5622339
source_file_count=1302
synced_paths=3
strict_confirmation=valid
final_remote_suite=111_passed
forbidden=0
appledouble=0
transaction_residue=0
lock_available=true
```

The final synchronized hashes are:

```text
e1274ce17a10c43c4af2fe4d62f4c35d9fe621aae6711cb3a468dbca493d2fdf
  tools/sitian_remote_transaction.py

f4f426a819b2afa0ab74c2ddf8445e3608cdefdea922a171684769a4bb50c779
  tools/test_sitian_remote_transaction.py

a8ccd4b812e074d0c2740e116bf40ae6de557d66e932f773577c72dd78edeb78
  tools/test_sitian_remote_scratch.py
```

The unchanged final controller hash is:

```text
ed1e1abe8b5689baa87aa488c638098f0207917f6edf64fe8226d2fe4b8926e3
  tools/sitian_remote_scratch.py
```

## Independent Review Evidence

The final whole-slice package and report are:

```text
/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/
  receipts/reviews/transaction-engine-final-review-52626be.diff

/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/
  receipts/reviews/transaction-engine-final-review-52626be.md
```

The report independently closes:

- init mutation replay and status-255 confirmation;
- init receipt preflight;
- same-nonce sync receipt fail-closed behavior;
- fresh init post-exchange committed-result preservation; and
- exact init reentry marker ordering.

## Storage and Hygiene Accounting

Remote filesystem state at the audit boundary:

```text
remote /:
  32% used

/data00/home/sitian:
  41% used
```

A targeted scan of remote `/tmp` and `/var/tmp` found no TinyLLMForge or
command-timeline paths. Final cache directories are under the mounted task
root.

Two transient task-owned hygiene violations occurred and were corrected:

1. A manual helper invocation without the cache environment briefly created
   `source/tools/__pycache__` under the mounted remote source. The exact
   directory was deleted, strict confirmation was rerun with the required
   cache environment, and final forbidden count is zero.
2. A test subagent created
   `/var/folders/.../tmp.tWHF2jVw5H` locally. The exact task-owned directory
   was inspected and deleted. It is not part of the repository or final
   evidence.

These events are retained in the audit rather than hidden. Neither left
surviving state.

## Supersession and Resume Boundary

The transaction-engine plan replaces the unsafe inline mutation portion of
the original Sitian remote-scratch Task 2. The old remote-scratch Task 3
runner design is not resumed independently; its source-bound runner,
manifest, and completion responsibilities are owned by command-timeline
Tasks 6–8.

The exact active continuation is:

```text
Task 5:
  Canonical Exact-Identity Diagnostic
  review-1 fixes committed at a1e6f42
  independent re-review pending

Task 6:
  Independent Verifier and Manifest
  pending

Task 7:
  Safe Source-Bound Remote Runner Contract
  pending

Task 8:
  Expanded Local Verification and Completion Audit
  pending
```

No GPU/model/performance execution was authorized or performed during the
transaction-engine closure. Any later controlled performance campaign must
use a never-before-used tag and separate authorization.
