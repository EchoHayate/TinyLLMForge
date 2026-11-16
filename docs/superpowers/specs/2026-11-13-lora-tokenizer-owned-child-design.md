# Tokenizer owned direct-child lifecycle

Date: 2026-11-13. Local child of tokenizer admission section 7.

## Decision and limits

The bounded pipe reader is sealed. Its failure stops reads but does not stop
the producer. Implement the next reusable primitive: exclusive handoff of one
trusted `subprocess.Popen` direct child, nonblocking kernel exit collection,
and bounded TERM/KILL escalation using the existing cleanup window.

Alternatives were raw-PID/process-name cleanup (unsafe ownership), process
group scanning/signalling (cannot prove descendant membership or prevent
group-ID reuse with the available contract), and exclusive direct-child
handoff. Choose the third; do not claim tree/group cleanup or isolation.
There is no launch API in this component. A future admitted launcher creates
the real process under verified hard limits, then transfers its handle.

Passing an exact Popen object is a trusted caller ownership assertion,
supplemented by kernel direct-child checks. It is not remote attestation or
a security boundary against malicious code in the same Python process.
Exclusive ownership forbids other calls to poll/wait/waitpid/send_signal on
that child, external reapers, concurrent Popen mutation, native SIGCHLD
changes, fork reuse, and asynchronous signal-handler reentrancy. Enforce
Python's visible SIGCHLD default disposition before each kernel check.
Reject observation loss; never infer exit code zero from ECHILD.

## Global constraints

- Existing checkout only; no branch/worktree, commit or push.
- No network/SSH, installation, telemetry, real runtime metadata discovery,
  tokenizer/model loading, GPU, training or candidate/task-source execution.
- Stdlib tests may create only their own fixed inert Python children with
  `-I -S`, empty environment, closed inherited FDs and no shell. Each fixture
  self-exits within 5 seconds, has bounded handshake/wait/cleanup, and creates
  no descendants. At most two such children live simultaneously.
- Preserve all accepted sources/tests/data/documents and sealed evidence.
- Use `apply_patch`; no inline code comments or alphabetic one-letter names.
- Production code never launches a process, signals a group, scans processes,
  reads/writes payload pipes, closes caller FDs, deletes paths or installs
  handlers. It signals only the exclusively transferred direct child.
- One attach attempt per instance and at most one termination attempt, one
  SIGTERM attempt and one SIGKILL attempt. No PID/PGID adoption API.
- Use exact `LaunchWindow` in CLEANUP/CLOSED; no new window, connection/audit
  claim, begin_cleanup or close. TERM grace is at most 1,000,000,000 ns;
  each sleep is at most 0.01 seconds and current cleanup time remaining.
- All ten enforcement/admission flags remain false; `cleanup_status` remains
  `CLEANUP_UNVERIFIED`, even when this one child has been reaped.

## Public contract

Files: `experiments/lora_necessity/tokenizer_owned_child.py` and
`test_tokenizer_owned_child.py`.

`OwnedChild()` constructs only local state. Module import and construction
must not inspect processes, sample clocks, create processes/FDs, send signals
or change environment/handlers. Every public operation first checks actual
main-thread identity, without touching caller inputs or state on rejection.

`OwnedChildError(ValueError)` has exactly `tokenizer owned child invalid`.
Suppress original ordinary exception display chains and never emit PID,
command, paths, environment or exception payload in public state/errors.
Non-Exception BaseExceptions preserve the original object.

States: NEW, OWNED, REAPED, LOST. First failure never changes.

### attach(process)

Only NEW, one attempt consumed before validation; success returns None.
Wrong/repeated calls raise. Repeated attach does not inspect inputs or change
an existing result. Initial validation failure sets LOST/PROTOCOL_ERROR.

Require exact `subprocess.Popen`, exact int PID > 1 and not this process's PID,
returncode is None, and no previous handoff of that same object. Use a module
weak set to retain handoff identity without retaining completed Popen objects.
Do not reset the weak-set entry after a failed or completed handoff.
Validate visible SIGCHLD default and direct-child status using waitpid below.
Hold a strong reference to the object and a private captured PID after binding.
Never read its command/environment/pipe payload. No Popen.poll/wait method.

### poll()

Return True only if REAPED; return False for a still-owned live/unreaped child.
REAPED is idempotent, with no new syscalls. NEW/LOST reject without new work.
Before each OS observation require unchanged exact PID, Popen.returncode None,
and `signal.getsignal(SIGCHLD) == SIG_DFL`. On mismatch set LOST and
OWNERSHIP_LOST, then raise. No restoration or mutation of a changed handler.

Use `os.waitpid(captured_pid, os.WNOHANG)` only. Zero PID means still waitable.
Exact PID with normal exited/signalled status means the kernel reaped this
child: compute `os.waitstatus_to_exitcode`, set Popen.returncode consistently,
store exit_code and set REAPED. Never report success based on kill returning.
Unexpected return shape/status/PID is WAIT_ERROR/LOST. ChildProcessError /
ECHILD is OWNERSHIP_LOST/LOST. Other errors, including InterruptedError, are
WAIT_ERROR/LOST; do not signal without a successful fresh ownership check.
Cancellation during observation also loses ownership, records CANCELLED if
first, and rethrows unchanged. These fail-closed decisions can leave the
child for the trusted caller's separately scoped cleanup.

### terminate(window)

REAPED returns True without inspecting window. NEW/LOST reject.
If termination was already attempted, reject without clock, wait or signal.
Validate exact window in CLEANUP/CLOSED before consuming termination attempt;
an invalid window raises without consuming or changing existing ownership.
Window pre-validation is outside the termination attempt. If its snapshot
raises a non-Exception cancellation, propagate that same object with owner
state unchanged and no clock, wait, signal or sleep. Incomplete validation
cannot authorize emergency signalling. A later valid call remains allowed.
Once consumed, no later call grants another termination/signalling attempt.
`poll()` remains usable after a timed-out attempt while ownership is retained.

First poll: if already exited, reap and return True with no signals/clock.
Otherwise sample remaining cleanup budget through the window. For a healthy
positive budget, set a local grace deadline from its last observed timestamp,
min(now + 1,000,000,000 ns, existing cleanup deadline), then attempt SIGTERM.

Before EVERY signal call, perform a fresh successful poll plus ownership
checks; if reaped, do not signal. Record the signal attempt before calling
`os.kill` so an exception cannot cause a duplicate attempt. Only positive
captured PID, never zero/negative/group functions. A child exiting between a
successful WNOHANG check and kill remains unreaped and reserves its PID under
the exclusive/default-reaping contract. This is not a claim of general
race-free signalling with other reapers or malicious same-process code.

Loop nonblocking polls, observing the shared remaining budget before sleeps.
Escalate to SIGKILL at grace expiry, at most once. Sleeps cannot exceed 0.01s,
remaining cleanup budget, or positive remaining TERM grace before escalation.
After KILL, poll/sleep only within the original cleanup allowance; do not
reset time or extend the cleanup deadline.

At exhausted or already CLOSED healthy window, skip TERM and immediately
attempt KILL if not attempted and ownership remains provable. Make one final
nonblocking poll, never sleep on stale/exhausted time. Return True if reaped,
otherwise False with first failure DEADLINE; state remains OWNED. Later
explicit poll may reap, but does not erase DEADLINE or create another kill.
This best-effort final signal/poll is not hard wall-clock preemption.

## Failures and cancellation

First failure reasons: PROTOCOL_ERROR, OWNERSHIP_LOST, WAIT_ERROR, CLOCK_ERROR,
SIGNAL_ERROR, SLEEP_ERROR, CANCELLED, DEADLINE.

Ordinary window observation errors record CLOCK_ERROR. A detected window
clock_failed flag also counts as CLOCK_ERROR even in CLOSED. Other ordinary
signal or sleep errors record SIGNAL_ERROR or SLEEP_ERROR. Preserve this
first reason and raise the fixed error after bounded emergency cleanup.

After successful window validation and consumption of the termination attempt,
on clock/signal/sleep error or any non-Exception cancellation in terminate,
attempt at most one not-yet-attempted KILL, only if fresh poll still proves
ownership, then one nonblocking poll. No sleeps or new time allowance.
If observation already set LOST, no more signal/wait attempts. Keep the
earliest non-Exception cancellation object even if emergency cleanup fails;
if emergency cleanup introduces the first cancellation, propagate it rather
than silently treating it as an ordinary failure. Earlier ordinary failure
reason remains recorded. No retry of a failed SIGKILL.

Any signal syscall exception, including ProcessLookupError, remains an error;
do not interpret it as successful cleanup without waitpid exit evidence.
Even if emergency collection reaps the child, an original operational error
still raises; child_reaped can be true while failure remains nonempty.

## Exact snapshot

Fresh primitive-only dict, no observation/clock call:

- `kind`: LOCAL_OWNED_CHILD_LIFECYCLE_ONLY
- `state`: NEW / OWNED / REAPED / LOST
- `attach_attempted`, `termination_attempted`: booleans
- `signals_attempted`: fresh ordered list of SIGTERM/SIGKILL strings
- `exit_code`: None or exact int obtained from kernel wait status
- `child_reaped`: true only in REAPED
- `failure`: None or the first fixed reason string
- `cleanup_status`: CLEANUP_UNVERIFIED
- `resource_limits_enforced`, `root_write_limit_enforced`, `deadline_enforced`,
  `process_termination_enforced`, `cleanup_verified`, `benchmark_eligible`,
  `formal_evaluation_ready`, `net_benefit_established`, `isolated`,
  `all_possible_second_rounds_validated`: always false.

The direct-child record is local execution evidence, not independently
verified tree cleanup. Descendants, transport connections, root deletion,
resource limits and actual audit acceptance remain separate.

## Validation

Test first, with initial missing-module assertion. Use real fixed child
fixtures for natural exit/nonzero status, normal TERM, TERM-ignoring KILL,
unchanged unrelated sibling, Popen status synchronization and no duplicate
signalling/reaping. Require a readiness handshake before TERM-ignore tests;
no guessed sleep used as readiness. Self-exit and cleanup bound every fixture,
including RED. Do not launch programs from experiment/task data.

Use synthetic OS/clock fault boundaries to cover exhausted budget, exact grace
and cleanup edges, no deadline reset, no sleeps after bad clock/cancellation,
wait/ECHILD/handler/PID/returncode drift, duplicate ownership, ordinary errors,
earliest cancellation and no repeated KILL. Never inject a forged positive
PID into a real kill; patch the signal boundary for adversarial cases.
Actual wrong-thread calls must cause zero state or OS changes.

Retain all prior 277 test IDs; main independently runs regression and import/
construction probe, hashes protected files, packages complete additions and
obtains task plus fresh whole-component review before exclusive sealing.
No new remote capability, group supervision or LoRA benefit is established.

## Design check

No generic launcher bypasses the still-missing resource backend. No raw PID
ownership promise. Exclusive parent wait ownership and default SIGCHLD are
explicit requirements. LOST is non-signallable; timeout keeps a safely
pollable child without retrying termination. Evidence flags do not overclaim.
