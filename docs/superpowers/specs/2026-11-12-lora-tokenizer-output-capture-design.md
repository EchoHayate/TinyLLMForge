# Bounded tokenizer output capture

Date: 2026-11-12. Local child of tokenizer admission section 7.

## Decision and boundary

The prior timing component is sealed. Historical remote inspection on
2026-11-05 reported cgroup v1 and non-writable user slices
(`experiments/feedback_adaptation/20261105-remote-isolation-preflight.md`).
That is not current capability evidence or authorization to modify a daemon.
Do not invent a cgroup v2 backend or label per-process limits as aggregate.

Three options were considered: unbounded `communicate()` followed by a size
check (too late); transport/process integration immediately (depends on
unapproved remote/backend facts); or a standalone bounded binary receiver.
Choose the receiver as a necessary, locally testable part of the future
transport. It retains actual accepted bytes, not just counters.

This component receives caller-supplied chunks, not sockets or processes.
It does not read, poll, kill, write files, verify actual EOF, or integrate the
sealed window/controller. Its capacity limits apply only to its own retained
payload. They are not an RSS ceiling, root-write quota, pipe/network limit,
remote producer limit, deadline enforcement or launch permission.

The future transport must read chunks of at most 65,536 bytes, call the
receiver before retaining any further output, use its owned absolute window,
stop reads on rejection, terminate only owned work, and independently verify
cleanup. It must not buffer unbounded output first. Captured stdout/stderr
are private evidence, not safe-to-display logs. Nothing here sanitizes them.

## Global constraints

- Existing checkout only; no branch/worktree, commit, push, network/SSH,
  installation, telemetry, real metadata/tokenizer/model loading, GPU,
  training or candidate/task-source execution.
- Standard-library local tests only; preserve accepted sources, tests, data,
  documents and all sealed evidence.
- Use `apply_patch`; no inline code comments or alphabetic one-letter names.
- Fixed accepted payload caps: result 67,108,864 bytes, stdout 1,048,576 bytes,
  stderr 1,048,576 bytes. No configurable limits, truncation or retry.
- A feed chunk is exact built-in bytes, nonempty and at most 65,536 bytes.
- Public snapshots contain no output text, token IDs or content hashes.
- All five admission flags remain false; cleanup is `CLEANUP_UNVERIFIED`.
- Root/resource/deadline/termination enforcement and cleanup verification
  remain false; complete capture is not a valid audit or real admission.

## API and state machine

One source `experiments/lora_necessity/tokenizer_output_capture.py`, one paired
test file. Export `OutputCapture` and `OutputCaptureError(ValueError)` with
fixed message `tokenizer output capture invalid`. Constructor takes no
arguments. Construction and every public method require actual main-thread
identity before mutation, validation with caller values, or payload copying.
Wrong-thread calls raise without altering state. No reentrancy or malicious
same-process/private-state tampering guarantee; no process-global uniqueness.

Streams, in order: `result`, `stdout`, `stderr`. Accept only exact built-in
strings with these values; never invoke caller formatting/equality.
State is `OPEN`, `FINISHED` or `FAILED`, initially OPEN. Each stream has an
owned growable byte buffer and false `eof`. Byte counts derive from actual
buffer lengths, not an independent caller-provided counter. Avoid storing
one object per feed: arbitrarily fragmented input must not grow metadata in
proportion to feed-call count. No decoding, parsing, evaluation or printing.

### Methods

- `feed(stream, chunk)`: only OPEN, known stream not at EOF, and valid chunk.
  Validate before storing anything. If existing length + chunk length exceeds
  the stream cap, reject the *whole* chunk, latch FAILED/`OUTPUT_LIMIT` with
  the known stream, and raise. Do not fill the remaining space with a prefix.
  Exact cap is accepted, but not implicit EOF. Otherwise append unchanged
  bytes and return None. A bytes subclass, bytearray, memoryview, empty or
  oversized chunk is a protocol error, not a truncation request.
- `end(stream)`: only OPEN and known non-EOF stream; mark its EOF and return
  None. Duplicate EOF and feed-after-EOF are protocol errors. All EOFs alone
  do not finish the capture; explicit finish is required.
- `finish()`: in OPEN, require all three EOFs, otherwise latch
  FAILED/`PROTOCOL_ERROR` with stream None and raise. On success set FINISHED,
  return None. Repeated finish in FINISHED returns None without copying
  output. Empty result is allowed at this layer: validity belongs to the
  audit parser and cannot be inferred from transport completion.
- `abort()`: in OPEN or FINISHED latch FAILED/`ABORTED` with stream None,
  preserving all accepted bytes/EOFs. In FAILED do nothing. Return None.
- `snapshot()`: return a fresh primitive-only metadata dictionary, no clock
  sampling or payload copies; safe in all states after the thread guard.
- `private_outputs()`: only FINISHED or FAILED; return a new ordered dict
  mapping the three names to immutable bytes copies of their retained
  buffers. No aliases to mutable internal storage. In OPEN, latch
  FAILED/`PROTOCOL_ERROR`, stream None, then raise.

Feed/end in FINISHED invalidate capture by latching FAILED/`PROTOCOL_ERROR`;
they cannot silently append to a finished result. Once FAILED, feed/end/finish
raise without changing any bytes, EOFs or first failure; abort is idempotent.
Do not inspect hostile caller values once FAILED.

All protocol failures latch the fixed reason `PROTOCOL_ERROR` and known stream
if an exact valid stream was provided, otherwise None. Preserve the first
failure forever. Validation precedence in OPEN/FINISHED: state, stream,
stream EOF, chunk type/size, capacity. No caller payload appears in errors.

Unexpected ordinary exceptions during buffer append or private materialization
latch `CAPTURE_ERROR`, suppress the original error chain and raise the fixed
error. Non-Exception BaseExceptions at those boundaries latch the same failure
and rethrow the original object. Append failures retain whatever valid prefix
is actually in the buffer; they do not claim the whole feed succeeded.
Materialization fails without returning a partial dict. A later copying
failure cannot overwrite a previously retained failure reason.

## Exact public snapshot

Top-level keys:

- `kind`: `LOCAL_TOKENIZER_OUTPUT_CAPTURE_ONLY`
- `state`: the state string
- `streams`: ordered list of exactly three fresh records with keys
  `stream`, `limit_bytes`, `accepted_bytes`, `eof`
- `total_accepted_bytes`: sum of the three retained lengths
- `capture_complete`: true only in FINISHED
- `failure`: None or a fresh dict with `reason` and `stream`
- `cleanup_status`: `CLEANUP_UNVERIFIED`
- `resource_limits_enforced`, `root_write_limit_enforced`,
  `deadline_enforced`, `process_termination_enforced`, `cleanup_verified`:
  always false
- `benchmark_eligible`, `formal_evaluation_ready`, `net_benefit_established`,
  `isolated`, `all_possible_second_rounds_validated`: always false

Capture completeness describes only supplied chunks and declared EOFs. It does
not authenticate origin, report completeness or success, including empty
results. Buffer payload sum can never exceed 69,206,016 bytes on valid calls.
Python allocation overhead and returned private copies are additional memory;
no 4 GiB aggregate-memory or 128 MiB root-budget claim follows.

## Validation

Test-first stdlib unittest; use actual byte buffers and production cap values.
Cover each exact limit and one-byte overflow, whole-chunk rejection near a
limit, interleaved streams, all-three-stream maximum, binary preservation
including invalid UTF-8 and NUL, fragmented feeds, EOF/order errors,
terminal behavior, invalid exact types, first-failure retention, private
snapshot independence, all false flags, and actual wrong-thread calls.
Use bounded synthetic storage-fault fixtures for exception/cancellation paths;
do not cause real memory exhaustion. No model or task source is executed.

Main rechecks all 218 existing tests, full untracked addition packaging,
fresh-process import/construction inertness, protected hashes and Git state.
Fresh task review and whole-component review precede an exclusive local seal.
No remote/runtime or scientific result is established by these tests.
