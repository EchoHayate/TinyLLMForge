# Fixed Read-Only Route Collector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enforce the reviewed one-shot identity/path query's local collection requirements without dispatching it during development.

**Architecture:** A private, standalone standard-library collector consumes only the exact r2 scope bytes. It validates current configuration closure, creates a durable exclusive claim in a fixed private parent, launches one owned process group, preserves bounded output and lifecycle metadata, and repeats configuration checks. Actual dispatch permission remains outside the program and is still missing.

**Tech Stack:** Python3.9 standard library, unittest with artificial process/selector/clock boundaries; no packages or real subprocess tests.

## Global Constraints

- Scope SHA256 `7c8bf4e3a4200f6dcedfb00f03444304a957081f79867be5fa019a34c2decdea`.
- Scope `/private/tmp/lora-recalibration-evidence-20261120-dtPPVL/route-observation-scope-r2.json` stays byte-identical and `DRAFT_NOT_AUTHORIZED_OR_EXECUTED`.
- One connection-chain attempt; work20s from before Popen, cleanup2s separately; stdout16384/stderr131072 bytes retained. Overflow is explicit, never successful truncation.
- Exact argv, replacement environment, stdin DEVNULL, cwd `/`, close_fds=True, start_new_session=True, no shell=True.
- No CLI or import-time execution, no arbitrary transport callback or approval boolean. A code invocation is not user authorization; all provenance/authorization/experiment flags stay false.
- No real SSH, network, credential read/query/refresh, stored-task/backend/worker import, tokenizer/model/GPU operation, installation, telemetry, bootstrap/flush, branch/worktree change, commit/push or cleanup.
- Only private source/test/helper files, this plan, result note and append-only ledger may change. Preserve all existing LoRA files.
- Private root for implementation evidence: `/private/tmp/lora-route-collector-20261120-QF2RTX`.
- Use apply_patch; no inline comments or single-letter variables. Existing unittest conventions override test-skill defaults. Python/default/basic/lite; prohibited telemetry/bootstrap/flush are replaced by retained local logs/artifacts.
- Do not rerun previous unchanged component suites.

## Task 1: Complete Fixed-Scope Collector

**Files:**
- Create private `route_observation_collector.py`.
- Create private `test_route_observation_collector.py`.
- Parent owns verification, review, result note and ledger.

**Interfaces:**

```python
def validate_scope(scope_raw):
    ...

def parse_observation(stdout_raw):
    ...

def collect_once(scope_raw):
    ...
```

`validate_scope`: strict builtin bytes <=65536, exact fixed SHA before JSON;
return fresh decoded dict. No caller-supplied expected digest or launch flag.

`parse_observation`: strict ASCII bytes <=16384; exact begin/end marker, ordered
hostname/uid/user/kernel/python3/python3.12/bwrap labels; the three command
resolutions may have one additional empty line emitted by the fixed shell.
No banner, duplicates, trailing bytes or missing fields. UID canonical
nonnegative decimal; identities nonempty ASCII without control characters;
command result is empty or an absolute path without control characters.
Return missing command paths as null; this does not qualify any executable.

`collect_once`: verify scope; check all6 pinned file hashes/sizes/resolved paths,
closed Include edges and directory/glob memberships without executing SSH
config. Read at most pinned length+1 per file. Reject unreadable/drifted data.
Use the fixed reviewed evidence parent, owned mode0700, canonical and no-follow.
Create fixed `route-attempt-<scope-sha>` exclusively; never delete or retry it.
Use directory descriptors, exclusive no-follow mode0600 files, fsync files and
directory. Save exact scope, source identity and claim before Popen; any
pre-dispatch write/check failure causes zero Popen calls.

The private process collector uses a single monotonic deadline established
before spawn. Nonblocking stdout/stderr share a selector. Read at most remaining
stream budget+1, retain only the limit and separately record observed overflow.
Poll/check work deadline even after EOF; no blocking unbounded communicate/read.
Launch once. On failure, signal only the owned local group (and its retained
direct child during incomplete construction, as refined below), wait within a
shared2s cleanup deadline, and observe group absence separately from direct-child
wait. Do not infer descendant cleanup from direct-child reaping or kill delivery.
Unexpected clock behavior fails closed. OS scheduling/system-call stalls are
not a hard-real-time guarantee; observed overruns cannot qualify completion.

Retain primary cancellation through cleanup and attempted persistence; then
re-raise the original BaseException. Secondary cleanup/persistence failures are
recorded by phase/type and never overwrite it. Ordinary failure returns failure
metadata when persistence succeeds; persistence failures raise and leave claim
consumed. Try independent stdout/stderr/receipt writes after an earlier write
failure; no subsequent connection or alternate output directory.

Returned metadata separates process status, parse status, configuration
identity checks and local lifecycle. `observation_complete` requires complete
valid output, exit0, both EOF, no overflow/error, verified direct-child reap and
local group absence, bounded observed work/cleanup and pre/post config checks.
It does not establish remote isolation, tokenizer lengths, model quality or
trusted collection. Remote state on failure remains UNKNOWN; even success does
not claim all remote processes are absent. Keep raw stdout/stderr as private
binary files with hashes in receipt; no raw diagnostics in public summaries.

Receipt publication refinement after independent review: a file cannot attest
its own final successful fsync. Therefore `receipt.json` is always an explicit
`UNFINALIZED_CANDIDATE` with `observation_complete=false`,
`persistence_completed=false`, and a separate `candidate_observation_complete`
value. Existence or parseability is not a completion receipt. Only a normal
collector return can report completed persistence, bind the candidate's exact
hash, and report observation completion; any later consumer must have a
separately captured successful caller return as well as matching raw files.
No caller capture exists or is implemented by this task. Receipt/close failures
attempt a separate exclusive phase/type failure note, without claiming storage
can reliably persist its own failure. Original artifacts are not rewritten.

Constructor-ownership refinement after third review: retain a preallocated
Popen instance before invoking its constructor exactly once. This is pinned
to the inspected local Python3.9.6/subprocess source, not a claim of portability
across unreviewed interpreter versions. If construction raises after a PID has
been recorded, keep that handle for bounded cleanup and close any acquired
streams. Preserve an earlier constructor cancellation exposed by its exception
context even when constructor unwinding raises a secondary exception.
If setsid may still be pending, cleanup may additionally signal that exact
retained direct child through its process handle; it never searches unrelated
processes or broadens the remote argv. Record direct-child signal attempts
separately from group signals. Before a PID is observable, retain unknown
ownership rather than infer no child from an exception. This does not promise
arbitrary instruction-level asynchronous-exception atomicity, recovery from
process termination, or interruptible operating-system calls.

- [x] **Step 1: Write failing tests**

Discovery-safe import and absent-implementation setUp assertion:

```python
def test_scope_digest_checked_before_parse(self):
    with self.assertRaisesRegex(ValueError, "scope digest"):
        collector.validate_scope(b"not json")
```

Test strict scope/digest; exact parser/missing paths/invalid output; success
with interleaved streams; timeout before/after EOF; delayed spawn consumes
budget; stdout/stderr limit inclusive and overflow; launch/read errors;
BlockingIOError; nonzero exit; cancellation and secondary failures; clock
rollback; unreaped child; surviving group; exact Popen args; config/file/glob
drift; exclusive claim; file modes/no-follow; pre-dispatch save failure yields
zero launches; post-dispatch save failure preserves attempt and stops; report
flags false and no import/CLI effects.

All fake processes live only in test memory; patch Popen and killpg at every
test boundary, including failure-path tests. No real process may be spawned.
Filesystem cases use new private temporary directories; no global config edits.

- [x] **Step 2: Save genuine RED**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/private/tmp/lora-route-collector-20261120-QF2RTX python3 -m unittest test_route_observation_collector -v
```

Save raw output and immediate exit in private `red.log/.exit`; expect absent
implementation assertion failures, not import errors.

- [x] **Step 3: Implement the collector**

Keep private configuration/persistence/process helpers in one file because
they serve exactly this frozen single-shot entry. Do not add a configurable
framework, alternate route, retries, approval parser, task execution or future
LoRA functionality. The three public interfaces above are the only entrypoints.

- [x] **Step 4: Verify GREEN and preservation**

Run the same explicit unittest command once per changed source/test identity;
save raw log/immediate exit. Verify previous source/receipt hashes, style/import
AST, exclusive test artifact behavior and no changes to the reviewed r2 scope.
Do not claim real process behavior from mocked boundaries.

- [x] **Step 5: Independent review and integration gate**

Fresh reviewer checks actual source/test diff, scope and raw saved logs, with
special attention to group lifecycle, cancellation and failure persistence.
Fix supported findings with failing tests and focused re-verification.
Record accepted source hashes and remaining authorization gate; do not launch,
rewrite old receipts, or mark the full LoRA research goal complete.

## Scope Check

This addresses the actual route review's missing bounded collector, not the
still-unimplemented 480-item supervisor/controller. The latter remains separate.
The overall research goal is unchanged. A current real observation will still
require explicit authorization, followed by independently verified original
results; this implementation alone is not a LoRA conclusion.
