# Command-Timeline Process-Isolated NVML Sampler Design

## Goal

Eliminate the remaining roughly two-second direct-NVML snapshot stall by
moving each frozen GPU's sequential query chain into its own persistent child
process, while preserving strict complete-snapshot telemetry and all existing
campaign invariants.

## Evidence and Root Cause

The thread-concurrent sampler first ran in immutable campaign
`20260818-command-timeline-tp4-b4-q4-r16`.

The first eager and graph epochs completed with strict telemetry, proving that
cross-GPU concurrency improved ordinary sampling. The third epoch,
`block-1:graph:first`, failed only during telemetry attachment. Worker status
was zero and both sampler stderr files were empty.

Measured repeat 2 lasted 1351 ms and contained seven host rows but no complete
four-GPU snapshot. Its surrounding complete GPU snapshots were 2124 ms apart.
The later snapshot showed four concurrent per-GPU chains lasting between 1973
and 2013 ms. On every GPU, most time accumulated in the SM-clock and
memory-clock calls, but the long durations were staggered across devices.

The important comparison is:

- r15, one process and serialized device chains: roughly two seconds;
- r16, one process and four device threads: still roughly two seconds.

Thread concurrency changed where wait time was recorded but did not reduce the
wall-clock stall. The evidence is consistent with serialization or lock
contention inside the process-local NVML library path. Adding more threads in
the same process would increase contention without crossing that boundary.

## Considered Approaches

### 1. Four persistent per-GPU child processes

Each child loads and initializes its own `libnvidia-ml.so.1`, validates one
frozen index/UUID pair, and executes that GPU's eight calls sequentially when
the parent sends a sample command. The parent sends all four commands before
receiving results, then timestamps and prints only a complete group.

This is the selected approach because it isolates process-local NVML state,
keeps per-device ordering, and retains one authoritative JSONL stream.

### 2. Thirty-two query threads

Flattening every `(GPU, API)` call into one thread pool would still share the
same loaded NVML library and its process-local synchronization. It would also
change same-device query concurrency without evidence that the NVML ABI or
driver benefits from it.

### 3. Independent uncoordinated GPU samplers

Four standalone samplers writing four files would isolate processes, but
their rows could represent different sampling iterations. Merging by nearest
timestamp would introduce borrowing and weaken complete-snapshot semantics.

## Architecture

The generated sampler uses the standard-library `multiprocessing` fork
context. Fork is explicit because both the production host and local contract
environment are POSIX, and the generated `python -c` program has no importable
module entrypoint for spawn.

The parent creates four duplex pipes and four persistent children. A child:

1. loads and configures NVML after the fork;
2. calls `nvmlInit_v2`;
3. resolves its assigned index and verifies the frozen UUID;
4. sends one ready message containing index, UUID, and child PID;
5. waits for `sample` or `stop`;
6. on `sample`, executes the existing eight timed calls sequentially and
   sends one primitive-only row;
7. on any failure, sends a structured error and exits;
8. calls `nvmlShutdown` exactly once in `finally`.

After all four ready messages pass validation, each parent iteration sends
`sample` to every child before receiving from any child. The parent receives
one row from each pipe in frozen inventory order. Only after all four rows are
available does it capture the shared Unix and monotonic timestamps and print
the four JSON rows.

Each raw row includes `sampler_process_pid` so tests and retained evidence can
prove that four distinct NVML processes served the snapshot. Telemetry
attachment continues selecting only repeat identity, timestamps, and UUID.

## Failure and Shutdown Semantics

- A child error message names the GPU index and original NVML API failure.
- EOF before a row is a fatal child failure.
- No child prints telemetry; therefore no partial snapshot can escape.
- SIGTERM and SIGINT set the parent's existing stop flag. A signal during a
  sample lets owned children finish, discards that in-progress group, sends
  `stop`, joins all children, and exits cleanly.
- If the outer campaign timeout terminates the sampler process group, all four
  owned children are in that same task-owned process group. No unrelated
  process is signaled.
- Inventory validation remains fail closed. Duplicate or malformed inventory
  is rejected before forking; UUID mismatch is rejected by the responsible
  child before sampling.
- Child process exit codes are checked after join. An unexpected nonzero exit
  is fatal even if a pipe message was lost.

## Preserved Invariants

- No synchronization or instrumentation enters the measured request path.
- Every GPU still executes the same eight calls in the same order.
- Every raw row retains all fields and all eight `query_duration_ns` values.
- Only complete four-GPU groups receive shared timestamps and are emitted.
- The 200 ms target cadence and monotonic deadline logic remain unchanged.
- Strict in-interval coverage, no boundary borrowing, and no synthetic
  duplication remain unchanged.
- Four-clean-GPU admission and ownership checks remain unchanged.
- All remote output remains beneath
  `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818`.

## Test Design

The fake-NVML subprocess contract will prove:

1. one complete snapshot contains four distinct `sampler_process_pid` values;
2. each child records exactly one frozen GPU identity;
3. each child preserves the eight-call per-device order;
4. an injected query failure emits no row and identifies the failing API;
5. UUID mismatch and invalid inventory remain fail closed;
6. SIGTERM causes four child shutdown markers and parent exit zero;
7. all existing duration, timestamp, startup, attachment, and complete
   command-timeline contracts remain green.

The old threaded implementation fails the distinct-process test before any
production edit.

## Campaign Validation

After local and remote contracts pass, the exact implementation files are
atomically synced with a detached receipt. The Mac Agent then uses the same
condition controller to start fresh immutable tag
`20260818-command-timeline-tp4-b4-q4-r17` immediately on `READY`.

Success at this layer requires complete strict telemetry for every measured
repeat reached by r17 and four distinct retained sampler PIDs per timestamp
group. Any later failure is preserved and diagnosed under another fresh tag.
