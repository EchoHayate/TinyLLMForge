# Tokenizer bounded pipe reader

Date: 2026-11-13. Child of tokenizer admission section 7.

## Decision

Continue the approved local launch-wrapper direction without reopening sealed
components. Alternatives: unbounded file/communicate capture (cannot stop
before retaining excessive output); immediate SSH/process-group integration
(needs unapproved runtime and ownership facts); or real nonblocking pipe
reads coupled to the accepted output capture and launch window. Choose the
third. This is useful executable I/O, not a callback-only simulation.

The reader does not launch or kill a process. The future supervisor owns the
connection, audit process, root limits and independent cleanup. On reader
failure it must stop the producer; stopping reads alone is not termination.
Successful pipe EOFs do not authenticate output or prove producer exit.
No cgroup capability is inferred from historical remote observations.

## Global constraints

- Existing checkout only; no branch/worktree, commit or push.
- Local stdlib tests only; no network/SSH, installations, telemetry, real
  metadata discovery, tokenizer/model loading, GPU, training or task-source
  execution. Synthetic bytes through local anonymous pipes are permitted.
- Preserve every accepted source/test/data/document and sealed evidence root.
- Use `apply_patch`; no inline code comments or alphabetic one-letter names.
- Reuse `OutputCapture` unchanged: result 67,108,864 bytes, stdout and stderr
  1,048,576 bytes each; maximum read 65,536 bytes, no truncation or retries.
- Reuse an already AUDITING exact `LaunchWindow` unchanged. Never create a
  replacement window, claim connections/audits, reset deadlines or close it.
- Selector waits are at most 0.1 seconds and at most the observed remaining
  work budget. Before and after each wait and each read, verify work remains.
- No configurable budgets, root-write/resource enforcement, process signals
  or admission promotion. All ten capture flags stay false and cleanup stays
  `CLEANUP_UNVERIFIED`.

## API

Create `experiments/lora_necessity/tokenizer_output_reader.py` and paired
`test_tokenizer_output_reader.py`.

`OutputReader()` is inert: only local state and an owned `OutputCapture`.
No pipe/selector creation, clock sampling, reads or dependency discovery.
Every public operation checks actual main-thread identity before touching
arguments or state. Wrong-thread operations raise without changing anything.

Public methods:

- `run(window, streams)` returns None on success. One attempt per reader;
  mark RUNNING before validating arguments. Repeated calls in any terminal
  state raise without altering the first result/failure or accessing inputs.
- `snapshot()` returns a fresh primitive-only dictionary with exactly
  `kind`, `state`, `failure`, `local_handles_released`, `capture`.
  Kind is `LOCAL_TOKENIZER_PIPE_READER_ONLY`; states are NEW, RUNNING,
  FINISHED, FAILED. `capture` is the unchanged capture snapshot. `failure`
  is None or one fixed reason string; never payload, FD number or OS text.
  No clock read or payload copy.
- `private_outputs()` delegates to capture only in FINISHED or FAILED.
  In NEW reject without mutation. It returns private bytes, never safe logs.
  Copy failure sets reader FAILED and CAPTURE_ERROR if no prior failure,
  preserving the underlying capture failure. Cancellation identity survives.

`OutputReaderError(ValueError)` has the fixed message
`tokenizer output reader invalid`. Ordinary failures suppress their original
display chain. Non-Exception BaseExceptions are rethrown unchanged.
No malicious same-process tampering, concurrent use of the shared window,
asynchronous signal reentrancy or concurrent close/reuse of caller FDs is
supported. These are trusted internal interfaces, not a security sandbox.

## Input and ownership

Require exact `LaunchWindow` in AUDITING. Only after that validation bind it
for failure cleanup accounting. Wrong window type/state leaves it untouched.
`streams` must be an exact dict with exactly the built-in string keys result,
stdout and stderr. Validate exact key types before equality/set operations.
Values must be exact ints >= 3, distinct; booleans/subclasses are rejected.
Copy this three-entry mapping; no caller formatting/equality hooks.

Validate all borrowed FDs before reading: `fstat` must show pipes/FIFOs;
`fcntl(F_GETFL) & O_ACCMODE` must be O_RDONLY and O_NONBLOCK must be set.
Reject two FDs referring to the same `(st_dev, st_ino)` pipe. No changing
blocking flags: a duplicate shares those flags with the original.
Closed FD / syscall failures are READ_ERROR; structurally invalid arguments
or valid FDs with disallowed properties are PROTOCOL_ERROR.

Duplicate each borrowed read FD once with `os.dup`, then register owned
duplicates with `selectors.DefaultSelector` for read readiness. Caller
originals are never closed or modified. This is borrowing the read stream,
not independent stream contents: reads consume the same kernel pipe buffer.
Only descriptors created by this run and its selector are released.
Partial setup failure releases all handles successfully acquired so far.

## Loop and completion

Check positive `window.work_remaining_ns()` before setup, before every wait,
after every wait, immediately before a read, and immediately after a read.
Window must remain AUDITING; healthy expiry/cleanup/closure is WORK_DEADLINE,
clock failure is CLOCK_ERROR. Setup is bounded in count, not preemptible.

For each readiness event read at most 65,536 bytes using `os.read`. Feed that
chunk directly into the capture before the next read; retain no output queue.
One read per ready stream per wait cycle avoids draining one hot stream
indefinitely. BlockingIOError and InterruptedError mean no bytes/EOF and
return to bounded polling without extending the window. Other read/select/
dup/register/unregister syscall exceptions are READ_ERROR. Invalid os.read
return values are passed to capture validation, not coerced.

An actual empty bytes read declares that stream's EOF and unregisters it.
No readiness event alone is EOF. Remove EOF streams from future polling.
After all three EOFs check the work window again, then finish the capture.
Release owned handles before returning FINISHED. Successful completion leaves
the shared window AUDITING: only the future supervisor can decide whether
the bytes are a valid audit and begin AUDIT_COMPLETE cleanup.

If expiry is detected after a read, discard that unaccepted chunk and fail.
Already accepted bytes remain privately available. One in-flight read chunk,
kernel pipe buffers, bytearray overhead and private output copies are outside
the retained-payload cap. Poll timeout is not a hard wall-clock enforcement
guarantee against scheduler delays, OS hangs or Python signal handlers.

## Failure and release

First failure reasons: PROTOCOL_ERROR, WORK_DEADLINE, CLOCK_ERROR, READ_ERROR,
OUTPUT_LIMIT, CAPTURE_ERROR, CANCELLED, RELEASE_ERROR. Map capture capacity
failure to OUTPUT_LIMIT; other capture failure to CAPTURE_ERROR.
Latch the first reason, abort capture without overwriting its first failure,
and stop all reads. For a bound window call begin_cleanup(AUDIT_FAILED), or
CANCELLED for cancellation; an already reached deadline retains its own
reason and allowance. A cleanup-accounting exception must not hide the
primary failure or cancellation. No process cleanup is performed.

Always attempt release of the selector and every owned duplicate, once each,
even after errors in another close. No close retry on uncertain OS outcomes.
`local_handles_released` initially true (no ownership); false while acquired
handles remain; after release true only if every close succeeded. Any close
failure leaves it false, even if the OS might have closed the resource.
Record RELEASE_ERROR only if no earlier failure; abort a formerly successful
capture and begin failure cleanup accounting if release fails.
Preserve the earliest non-Exception cancellation object across finalization;
if release first introduces cancellation, rethrow it. Never silently suppress
KeyboardInterrupt/SystemExit to report success.

These flags are local release-call evidence, not independent process, socket,
filesystem or remote cleanup verification.

## Validation

TDD with bounded real anonymous pipes: binary interleaving, empty streams,
actual EOF, original descriptors remain open/unchanged, no duplicate alias
streams, partial setup release, no read after failed capacity or deadline,
no hidden payload in snapshots/errors, one-shot behavior and main-thread guard.
At least one integration test exceeds the real stdout 1,048,576-byte limit
without replacing caps; use an owned nonblocking producer thread with bounded
join/stop and no external programs. Existing capture tests retain all-three
production-capacity coverage.

Fake only nondeterministic clock and OS fault boundaries. Assert exact
nanosecond deadline edges, timeout <= min(0.1, remaining seconds), readiness
with no bytes, InterruptedError, post-wait/post-read expiry, clock errors,
selector/read/copy/close cancellation identity and continued handle release.
All fixtures release originals and owned producer threads even on assertion.
Never execute task sources, launch a subprocess workload or wait 540 seconds.

Main retains all 245 prior test IDs, checks syntax/naming and fresh-process
import/construction inertness, packages complete additions, and obtains task
and whole-component review before exclusive sealing. Historical receipts are
execution records, not independently witnessed TDD provenance.

## Design self-check

One reader/source-test pair; no supervisor facade implying termination.
Actual FD readiness/read behavior is tested, with no borrowed-FD mutation.
Budget and capture dependencies are sealed and not edited. Accepted local
release is explicitly narrower than audit acceptance or remote cleanup.
