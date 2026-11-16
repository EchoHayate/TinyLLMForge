# Tokenizer launch-window accounting

Date: 2026-11-12. Child of `2026-11-12-lora-tokenizer-admission-design.md`,
sections 7–8. This is the next local component, not permission to launch.

## Scope and choice

Implement one small, synchronous `LaunchWindow` in
`experiments/lora_necessity/tokenizer_launch_window.py`, with a paired stdlib
unittest file. It accounts for one connection and one audit claim using a
shared absolute clock. No integration changes to sealed components.

Relative phase timeouts would permit budget resets. A complete transport and
resource supervisor would conflate timing with currently unverified hard
limits. Choose a standalone absolute-deadline primitive instead. The future
launcher must own exactly one instance for its approved run, claim immediately
before attempting each external operation, and enforce returned deadlines.
This class does not enforce process-global uniqueness, prevent replacement
objects, stop processes, interrupt blocking work, or authorize any operation.

## Global constraints

- Existing checkout only; no branch/worktree creation, commits or pushes.
- Local stdlib tests only; no SSH/network, installations, telemetry, real
  metadata discovery, tokenizer/model loading, weights, GPU, training, or
  candidate/task-source execution.
- Preserve all accepted source/tests/data/docs and every sealed evidence root.
- Use `apply_patch`; no inline code comments or alphabetic one-letter names.
- Connection claim fixes work deadline at start + 540 seconds and total
  deadline at start + 600 seconds, in integer nanoseconds.
- At most one connection claim and one audit claim per instance.
- Cleanup lasts at most 60 seconds and never beyond the total deadline;
  phase changes and repeated calls cannot reset either deadline.
- All five admission flags remain false; cleanup is `CLEANUP_UNVERIFIED`.
- Resource enforcement, process termination and cleanup verification remain
  explicitly false. Snapshot data is last-observed accounting, not live proof.

## API and state

`LaunchWindow()` takes no arguments and does not sample time. Construction and
every public method require `threading.current_thread() is
threading.main_thread()`. Wrong-thread calls raise `LaunchWindowError` before
any clock read or mutation. This avoids ordinary cross-thread claims, not
malicious access, fork reuse or asynchronous signal reentrancy.

All ordinary public failures use `LaunchWindowError(ValueError)` with the fixed
message `tokenizer launch window invalid`, without exception text or caller
payload. Successful mutators return `None`.

States: `NEW`, `CONNECTING`, `AUDITING`, `CLEANUP`, `CLOSED`.
Counters initially zero; nullable times and stop reason initially `None`;
`clock_failed` initially false. `CLOSED` is terminal. The first stop reason
is retained, including across later clock errors; `clock_failed` records
clock failure separately.

### Clock and deadline observation

Only `time.monotonic_ns()` supplies time. Each timed operation samples once.
Accept only exact built-in `int`, nonnegative and no less than the previous
sample; equal timestamps are valid. Store only valid samples. An ordinary
clock exception, invalid type/value or regression latches `clock_failed`,
sets `CLOSED`, sets first reason `CLOCK_ERROR` if absent, then raises the fixed
error. A clock `BaseException` outside `Exception` performs the same latch and
rethrows the original object. Clock failure never recovers or grants a
cleanup budget from stale time.

On every valid observation in CONNECTING/AUDITING, if now >= work deadline,
enter cleanup with first reason `WORK_DEADLINE` and cleanup deadline equal to
the total deadline. This represents the reserved final minute, not a newly
started minute. On observation in CLEANUP at or beyond its deadline, close.
Observing past total therefore closes immediately. No negative remaining time.

### Public methods

| Method | Behavior |
| --- | --- |
| `claim_connection()` | Only NEW. Consume connection counter (set 1) before sampling; therefore a failed initial clock still consumes it. A valid sample fixes start/work/total times and enters CONNECTING. |
| `claim_audit()` | Only CONNECTING. Observe time first. Only if still CONNECTING, consume audit counter (set 1) and enter AUDITING. Expired claims raise with no audit counter increment. |
| `work_remaining_ns()` | NEW is a protocol error. In CONNECTING/AUDITING/CLEANUP observe once, apply deadlines, then return max(work deadline - now, 0) only in CONNECTING/AUDITING, otherwise 0. CLOSED returns 0 without sampling. |
| `begin_cleanup(reason)` | Exact built-in string in `AUDIT_COMPLETE`, `AUDIT_FAILED`, `CONNECTION_FAILED`, `CANCELLED`. In NEW, close without clock or deadlines, retaining reason. In CONNECTING/AUDITING, observe; if still active enter CLEANUP with first reason and deadline min(now + 60 seconds, total). Existing CLEANUP observes but preserves reason/deadline. CLOSED is idempotent without clock. |
| `cleanup_remaining_ns()` | Only CLEANUP or CLOSED. In CLEANUP observe and return max(cleanup deadline - now, 0), or 0 if observation closed it. CLOSED returns 0 without sampling. Other states are protocol errors. |
| `close()` | NEW closes with first reason `CLOSED`, without time. Active/CLEANUP observe then close with first reason `CLOSED` if absent. CLOSED is idempotent without clock. This ends accounting only, not OS work or cleanup verification. |
| `snapshot()` | Fresh primitive-only dict of last-observed state. No clock reads, no mutation; wrong-thread check still applies. |

### Protocol errors

Wrong-order or duplicate claims, querying cleanup time before cleanup, and
invalid cleanup reasons all raise the fixed error. NEW becomes CLOSED with
`PROTOCOL_ERROR`, without consuming attempts or sampling. CONNECTING/AUDITING
observe once and, if still active, enter CLEANUP with `PROTOCOL_ERROR` and
min(now + 60 seconds, total). CLEANUP observes but retains its deadline and
first reason. CLOSED stays unchanged without sampling. Deadline expiry found
during that observation takes precedence over the protocol reason. Clock
errors take precedence as described above. Rejected calls do not add claims.

After a protocol failure with a healthy clock and an existing start, cleanup
can still use its remaining bounded allowance. After a clock failure no
timed allowance is trustworthy; external termination remains the future
supervisor's responsibility.

### Snapshot schema

Exact keys: `kind`, `state`, `connection_attempts`, `audit_attempts`,
`started_ns`, `last_observed_ns`, `work_deadline_ns`, `total_deadline_ns`,
`cleanup_deadline_ns`, `stop_reason`, `clock_failed`, `cleanup_status`,
`resource_limits_enforced`, `process_termination_enforced`, `cleanup_verified`,
`benchmark_eligible`, `formal_evaluation_ready`, `net_benefit_established`,
`isolated`, `all_possible_second_rounds_validated`.

`kind` is `LOCAL_TOKENIZER_LAUNCH_WINDOW_ACCOUNTING_ONLY`.
`cleanup_status` is `CLEANUP_UNVERIFIED`. The other eight evidence/admission
booleans are false at every state. No caller-provided text appears in output.
No computed remaining-time field: a snapshot must not look like a fresh lease.
Changing a returned dict must not affect the object.

## Validation and limits

Patch the module clock only in synthetic tests. Cover construction without
time, equal/large integer times, nanosecond boundaries around 540/600 seconds,
early and late cleanup, repeated cleanup, rejected claims, invalid reasons,
wrong threads, clock faults at start/work/cleanup/close, cancellation identity,
terminal behavior, snapshot independence and invariant flags.
Preserve and rerun the prior 191 tests. Fresh-process import must not sample
time, create files, initiate network/process activity or load third-party
model/tokenizer modules. Independent review covers the full new additions.

Acceptance proves only local accounting. Remaining gates include aggregate
4 GiB memory / 8 tasks, 128 MiB root / 64 MiB result / 1 MiB per output stream,
disabled core dumps, transport ownership/credential isolation, approved exact
runtime metadata and scope, and independent cleanup. None are implemented
by this child.

## Design check

One source/test pair; no caller clock or configurable budgets. Failed initial
claims cannot retry; healthy protocol failures retain bounded cleanup;
clock errors do not invent remaining time; terminal state is irreversible.
No API accepts a success flag capable of promoting scientific admission.
