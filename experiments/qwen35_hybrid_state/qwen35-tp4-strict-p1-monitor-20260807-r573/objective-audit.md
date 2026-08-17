# strict-P1 TP4 objective audit

## Active policy supersession: 2026-08-10

The user explicitly relaxed the original zero-compute-process gate.
The active monitor policy is now `shared-low-utilization`:

- fixed GPUs remain exactly `2,4,5,6`;
- every fixed GPU must retain at least `25 GiB` free;
- unrelated compute processes are allowed;
- every fixed GPU must report at most `10%` instantaneous GPU
  utilization;
- two consecutive `READY` samples are still required;
- the benchmark performs a fresh launch-time guard with the same
  shared policy and threshold;
- results are explicitly bounded as non-exclusive shared-GPU
  observations and must not be represented as an uncontended
  strict-P1 performance baseline;
- cleanup remains limited to processes matching both the unique
  strict-P1 attempt tag and its remote run root, plus verified
  descendants.

All later references in this document to a mandatory zero-process gate
describe historical policy and are superseded by this section.

## Success criteria and evidence

1. Monitor fixed GPUs `2,4,5,6` at low frequency.
   - Runner constant: `GPU_INDICES = (2, 4, 5, 6)`.
   - Launch arguments: `--interval-s 60`.
   - Live ledger: `resource_samples.jsonl`.
   - Live restart verification continued sample IDs `8 -> 9` and
     `9 -> 10`.

2. Require at least `25 GiB` free on every fixed GPU.
   - Frozen threshold: `25,769,803,776` bytes.
   - The live resource record includes
     `minimum_free_bytes_per_gpu=25769803776`.
   - `_select_tp4_gpu_resources` rejects a fixed GPU below the
     threshold.

3. Admit shared GPUs only at low utilization.
   - `_select_shared_tp4_gpu_resources` permits unrelated compute
     processes but requires `utilization_percent <= 10`.
   - The read-only query records both compute-process inventory and
     `nvidia-smi` `utilization.gpu`.
   - Live records include
     `requires_no_active_compute_processes=false`,
     `maximum_gpu_utilization_percent=10`, and
     `resource_sharing_policy=shared-low-utilization`.
   - The same policy is carried through benchmark preflight,
     execution-plan resource guards, and execution-receipt validation.

4. Do not substitute other GPUs.
   - Selection requires every index in `(2,4,5,6)` and returns those
     indices in that order.
   - Failure reason is `eligible GPUs 2,4,5,6 are required`.

5. Avoid a transient launch race.
   - Two consecutive READY samples are required.
   - The benchmark runs a fresh preflight immediately before worker
     launch.
   - A preflight `BLOCKED_RESOURCES` result is persisted as a launch
     attempt and monitoring resumes only after scoped cleanup succeeds.

6. Launch the prepared strict-P1 TP4 experiment directly.
   - The runner calls `execute_benchmark_launch(mode="canonical", ...)`.
   - No dummy reservation path exists in the monitor or wrapper.
   - The run tag is
     `qwen35-tp4-strict-p1-canonical-20260806-r571`, with a unique
     attempt suffix.

7. Release only experiment-owned processes after success or failure.
   - Cleanup matches both the unique run tag and its remote run root.
   - It expands only descendants of those matched root processes.
   - It records PID start times to prevent PID-reuse mistakes.
   - It uses TERM followed by bounded KILL only for matched PIDs.
   - It does not use `pkill`, `killall`, or GPU reset.

8. Persist results and cleanup evidence.
   - Success/final classification writes `monitor_result.json`
     atomically.
   - Launch exceptions write `monitor_failure.json` atomically.
   - Resource observations append durably to
     `resource_samples.jsonl`.
   - Cleanup callback exceptions and incomplete cleanup are classified
     `CLEANUP_FAILED` and cannot be reported as a successful launch.

9. Remain operational across local process or launchd restarts.
   - The wrapper owns the SSH ControlMaster and fixed FILE Kerberos
     cache.
   - Resume accepts only a contiguous, parseable ledger with no
     terminal artifact.
   - The consecutive-READY window resets after restart.
   - launchd restarts non-terminal failures.
   - Controlled verification: wrapper PID `52890` was replaced by
     `54316`, and sample ID continued `9 -> 10`.

## Fresh validation

- ControlMaster transport: 5 tests passed.
- strict-P1 monitor: 10 tests passed.
- live monitor runner: 9 tests passed.
- hybrid-prefix remote runner: 25 tests passed.
- hybrid-prefix remote execution plan: 8 tests passed.
- hybrid-prefix remote execution receipt: 6 tests passed.
- hybrid-prefix remote execution executor: 4 tests passed.
- Python compilation passed.
- Shell syntax validation passed.
- launchd plist lint passed.
- `git diff --check` passed for the monitor changes.
- Real remote samples were captured after launchd reload and after a
  controlled local monitor-process failure.

## Remaining completion gate

The infrastructure and safety requirements are covered, but the overall
objective is not yet complete because fixed GPUs `2,4,5,6` have not
simultaneously remained at or below `10%` utilization for two
consecutive samples. The canonical TP4 experiment has therefore not
run, and no final `monitor_result.json` exists yet. The launchd service
remains active and will launch only after two consecutive qualifying
samples followed by the fresh shared-policy resource guard.

## Shared-policy live activation: 2026-08-10

The sole launchd service was safely restarted from PID `54316` to PID
`2947`. The durable ledger continued without reset:

- sample `3806`, observed at
  `2026-08-10T09:35:21.546552+00:00`;
- sample `3807`, observed at
  `2026-08-10T09:36:23.840701+00:00`.

Both samples contain the active shared-policy metadata:

- `requires_no_active_compute_processes=false`;
- `minimum_free_bytes_per_gpu=25769803776`;
- `maximum_gpu_utilization_percent=10`;
- `resource_sharing_policy=shared-low-utilization`;
- the non-exclusive performance claim boundary.

The fixed-GPU readings were:

- GPU `2`: `66.61 GiB` free, four unrelated compute processes,
  `0%` utilization;
- GPU `4`: `52.96 GiB` free, four unrelated compute processes,
  `0%` utilization;
- GPU `5`: about `52.85 GiB` free, two unrelated compute processes,
  `35%` then `56%` utilization;
- GPU `6`: `58.85 GiB` free, three unrelated compute processes,
  `0%` utilization.

The monitor therefore correctly remained `BLOCKED_RESOURCES` because
GPU `5` exceeded the `10%` utilization limit. No dummy reservation,
launch attempt, remote process signal, `monitor_result.json`, or
`monitor_failure.json` was produced. Launchd remained running as PID
`2947`.

Focused validation after the policy change:

- ControlMaster transport: `5` tests passed.
- strict-P1 monitor loop: `10` tests passed.
- live monitor runner and launch-policy wiring: `9` tests passed.
- hybrid-prefix remote runner: `25` tests passed.
- hybrid-prefix remote execution plan: `8` tests passed.
- hybrid-prefix remote execution receipt: `6` tests passed.
- hybrid-prefix remote execution executor: `4` tests passed.
- Python compilation and `git diff --check` passed.
- A real read-only ControlMaster query successfully parsed
  `utilization.gpu` and independently classified the same live state
  as `BLOCKED_RESOURCES`.

The broader engine remote-execution-plan test currently has an
unrelated pre-existing fixture mismatch: its guarded-authority PASS
payload omits the inventory now required by the engine receipt
validator. The shared helper itself is exercised by the passing
hybrid-prefix execution-plan test and the real remote query.

After the final launch-result claim-boundary change, launchd was safely
restarted again from PID `2947` to PID `15387`. Ledger sample `3812`
was appended immediately at
`2026-08-10T09:40:45.187744+00:00` with the same shared-policy fields.
GPU `5` reported `46%` utilization, so the monitor correctly remained
blocked.

The exact production launch-time guard command generated by
`qwen35_tp4_engine_remote_execution_plan._shared_low_utilization_resource_guard_command`
was also executed read-only through the existing ControlMaster. It
returned exit code `1`, empty stdout, and
`configured GPU utilization is too high` on stderr. This proves the
fresh guard enforces the same threshold independently of the monitor
sampling query. No experiment process was started by this check.

The launch result returned to the monitor now always carries:

- `resource_sharing_policy=shared-low-utilization`;
- the explicit non-exclusive performance claim boundary.

Therefore a future `monitor_result.json` cannot silently present a
shared-GPU run as an uncontended strict-P1 performance baseline.

At sample `3814`
(`2026-08-10T09:42:49.862537+00:00`), the monitor recorded its first
`READY` sample under the shared policy:

- all fixed GPUs had at least `52.96 GiB` free;
- all four fixed GPUs reported `0%` utilization;
- unrelated compute processes remained present and were intentionally
  tolerated by the approved shared policy.

The condition did not persist. At sample `3815`
(`2026-08-10T09:43:52.184933+00:00`), GPU `5` was back at `99%`
utilization with two compute processes, so the monitor reset the
consecutive-READY window and returned to `BLOCKED_RESOURCES`.

A read-only process snapshot around this transition identified the
transient GPU `5` load as a root-owned validation process running
`/mnt/eval_workspace/eval_entrypoint.py` plus the existing root-owned
`python server.py`. Neither process matched the strict-P1 attempt tag
and remote run root, so neither was eligible for cleanup.

No launch attempt or experiment process was created. This transition
is direct live evidence that the two-consecutive-READY requirement
prevents launching into a short resource gap immediately followed by a
new high-utilization workload.

## Live blocker ownership

A read-only remote process audit confirmed that the compute-process
records are backed by live `/proc` entries rather than stale
`nvidia-smi` rows.

- GPU 2 includes long-running processes owned by another user, including
  `inferencer_worker_0_0` and `m13p4_*` workers.
- GPUs 4 and 6 include active `root` processes running
  `/opt/tiger/test/main_model.py`.
- Two `sitian` processes are from the older
  `/data00/home/sitian/sitian-workspace01/tllm/qwen35-engine-native-entry-diagnostic-20260729-01`
  diagnostic, with `CUDA_VISIBLE_DEVICES=2,4,5,6`.
- Those diagnostic processes do not contain the strict-P1 run tag or
  strict-P1 remote output root and are therefore outside the authorized
  scoped-cleanup set.
- Even removing those older `sitian` processes would not make the fixed
  GPU set eligible because unrelated `root` and other-user processes
  remain.

No remote process was signaled or terminated during this audit.

## Live completion audit: 2026-08-08 08:32 CST

The objective remains incomplete. A fresh monitor sample and an
independent read-only `nvidia-smi` query agreed that every fixed GPU has
more than `25 GiB` free, but none of the four fixed GPUs has zero compute
processes.

- Ledger sample `504`, observed at
  `2026-08-08T00:32:22.329301+00:00`, is
  `BLOCKED_RESOURCES`.
- GPU `2`: `64.1 GiB` free, five compute processes.
- GPU `4`: `47.5 GiB` free, five compute processes.
- GPU `5`: `78.6 GiB` free, one compute process.
- GPU `6`: `59.8 GiB` free, two compute processes.
- The ledger is parseable and contiguous from sample `1` through
  sample `504`.
- No `launch_attempts.jsonl`, `monitor_result.json`, or
  `monitor_failure.json` exists, so no launch has been attempted and no
  terminal result has been claimed.

The live process ownership check showed:

- GPU `2`: four long-running workers owned by another user plus the
  older `sitian` diagnostic process.
- GPU `4`: two `root` model processes plus the older `sitian`
  diagnostic child.
- GPU `5`: a current `sitian` RL evaluation process started on
  2026-08-08.
- GPU `6`: two `root` model processes.

None of these processes matches both the unique strict-P1 run tag and
its remote run root, so none is eligible for the monitor's scoped
cleanup.

Fresh executable validation used the test files directly rather than
module discovery:

- ControlMaster transport: `5` tests passed.
- strict-P1 monitor: `10` tests passed.
- live monitor runner: `6` tests passed.
- Python compilation, shell syntax, and launchd plist lint passed.

The launchd monitor remains running. It will continue sampling every
`60` seconds and will directly launch the canonical strict-P1 TP4 run
only after two consecutive qualifying samples followed by a fresh
preflight.

A follow-up utilization query after sample `507` did not weaken this
gate. All fixed GPUs happened to report `0%` instantaneous GPU
utilization, but `nvidia-smi pmon` still listed compute contexts on
every fixed GPU. Several corresponding host processes also had active
CPU usage, including the older diagnostic worker, the current RL
evaluation, and `root` model processes. The objective explicitly
requires no compute processes, not merely a momentary `0%` utilization
sample, so the monitor correctly remained blocked.

At sample `517`, GPU `6` gained another compute process, PID `3936335`.
A read-only `/proc` audit identified it as a `wangmin`-owned Docker
model service running
`/opt/tiger/models/search.bert_qrec.8.8/pp_server/runner/main_model.py`.
It is unrelated to the strict-P1 run tag and remote run root, so it is
also outside the authorized cleanup scope.

Samples `518` through `527` remained `BLOCKED_RESOURCES`. The extra GPU
`6` worker changed PIDs several times. A parent-chain audit of PID
`4033570` traced it through `pilot_launcher.py`,
`lg_inference_launch.sh`, and `run_pp_server.sh` to a
`wangmingfa`-owned VS Code session inside Docker. This is a separate
search inference service that is supervising replacement workers, not a
strict-P1 residual process.

GPU `5` became eligible at sample `540`
(`2026-08-08T01:09:42.276555+00:00`) and remained free through sample
`545`. The prior RL evaluation process exited, its scheduler stage
wrote an `exit_code` artifact, and GPU `5` increased from about
`78.63 GiB` free with one compute process to about `79.14 GiB` free
with no compute processes. This confirms that the monitor detects a
real resource release without intervention. The overall fixed set
remained blocked because GPUs `2`, `4`, and `6` still had unrelated
compute processes.

Through sample `585` (`2026-08-08T01:56:21.702282+00:00`), GPU `5`
remained eligible while the resource signatures for GPUs `2`, `4`, and
`6` were unchanged for twenty consecutive additional samples. No
launch attempt or terminal monitor artifact was produced. The monitor
therefore continues to enforce the all-four-GPU gate rather than
launching from partial availability.

By sample `635` (`2026-08-08T02:48:08.172917+00:00`), the same partial
state had persisted for another fifty samples after sample `585`:
GPU `5` was eligible, while GPUs `2`, `4`, and `6` retained the same
unrelated compute-process sets. No launch or terminal artifact was
created. The launchd service remained the sole monitor and continued
the approved 60-second polling loop.

At sample `690` (`2026-08-08T03:45:15.296828+00:00`), GPU `5` became
ineligible again: free memory dropped to about `14.5 GiB` and PID
`2759617` appeared. A read-only parent-chain audit identified a
`root`-owned `sglang::scheduler` launched by `server.py` under
`nsys profile` in Docker. It is unrelated to strict-P1 and outside the
authorized cleanup scope. The monitor correctly enforced both the
`25 GiB` threshold and the zero-compute-process requirement.

The GPU `5` profiler was transient. Sample `691` showed both
`server.py` and `sglang::scheduler` with about `13.5 GiB` free; sample
`692` showed GPU `5` back at about `79.1 GiB` free with no compute
processes. No launch occurred because GPUs `2`, `4`, and `6` remained
blocked throughout.

At sample `701`, GPU `5` was occupied again by PID `2939905`, with
about `14.8 GiB` free. Its parent chain was another `root`-owned SGLang
server in the same unrelated Docker environment, this time launched
directly by `start_local.sh` rather than under `nsys`. It is outside the
strict-P1 cleanup scope.

During samples `765` through `885`, the older `sitian`
`engine_construct_diag.py` process and its child exited naturally from
GPUs `2` and `4`; they were not signaled by this monitor. Those GPUs
still retained four unrelated compute processes each. GPU `6` retained
three unrelated processes. At sample `885`, GPU `5` contained PID
`252431`, a `root`-owned `python server.py` process in another Docker
container. The fixed set therefore remained `BLOCKED_RESOURCES`.

Samples `886` through `1006` were all `BLOCKED_RESOURCES`. GPU `2`
varied between four and five compute processes, GPU `4` between four
and five, GPU `5` between one and four, and GPU `6` between three and
four. At sample `1006`, each fixed GPU still had at least one compute
process. No READY sample, launch attempt, or terminal monitor artifact
was produced.

Samples `1007` through `1127` remained `BLOCKED_RESOURCES` with an
unchanged process set for two hours: four processes on GPU `2`, four on
GPU `4`, one on GPU `5`, and three on GPU `6`. This is sustained
resource ownership rather than a transient sampling race.

That direct SGLang launch briefly expanded to two scheduler processes,
reducing GPU `5` free memory to about `1.9 GiB` at sample `702`. One
scheduler remained at sample `703`, and GPU `5` was fully released
again at sample `704`. The monitor did not mistake either transient
release boundary for all-four-GPU readiness.

Samples `1127` through `1367` were also entirely
`BLOCKED_RESOURCES`. Across that four-hour window, GPU `2` had at least
four compute processes, GPU `4` at least four, GPU `5` at least one,
and GPU `6` at least three. No READY sample or experiment artifact was
created.

At `2026-08-09 06:51 CST`, the durable ledger was contiguous from
sample `1` through sample `1795`. The latest sample remained
`BLOCKED_RESOURCES`: GPU `2` had four compute processes, GPU `4` had
four, GPU `5` had one, and GPU `6` had three. The launchd service was
still running, and no launch-attempt or terminal artifact existed.

At `2026-08-09 09:12 CST`, the ledger had advanced contiguously through
sample `1931`. The fixed-GPU process counts were unchanged at
`4,4,1,3` for GPUs `2,4,5,6`. The launchd service remained running,
there were no local foreground observer loops, and no READY, launch, or
terminal artifact existed.

At `2026-08-09 09:26 CST`, the ledger had advanced contiguously through
sample `1945`. The fixed GPUs all exceeded the `25 GiB` free-memory
threshold, but GPUs `2,4,5,6` still had `4,4,1,3` compute processes,
respectively. Those PID sets were unchanged across samples `1941`
through `1945`, so this was not a transient launch race. The launchd
service remained running as PID `54316`; no READY sample, launch
attempt, monitor result, or monitor failure artifact existed.

At `2026-08-09 10:27 CST`, the ledger had advanced through sample
`2004`. The same fixed-GPU PID sets and `4,4,1,3` process counts had
remained unchanged for more than one hour. All four GPUs still exceeded
the memory threshold, but none met the zero-compute-process gate. The
launchd service remained running as PID `54316`, and no launch or
terminal artifact existed.

At `2026-08-09 11:28 CST`, sample `2063` still showed exactly the same
fixed-GPU process sets, free-memory values, and `4,4,1,3` process
counts. The service remained running as PID `54316`, and no launch
attempt or terminal artifact existed. The monitor therefore continued
waiting rather than interfering with unrelated long-lived workloads.

At `2026-08-09 13:28 CST`, sample `2179` still showed the same PID sets
and `4,4,1,3` compute-process counts on GPUs `2,4,5,6`. All fixed GPUs
had more than `25 GiB` free, but the zero-process requirement remained
false on every card. The launchd service remained running as PID
`54316`; no launch attempt or terminal artifact existed.

At `2026-08-09 17:29 CST`, sample `2411` still showed exactly the same
PID sets, free-memory values, and `4,4,1,3` process counts on the fixed
GPUs. The launchd service remained running as PID `54316`; no launch
attempt, monitor result, or monitor failure artifact existed.

A read-only process audit after sample `2411` showed that the blockers
were long-lived unrelated services: the GPU `2` processes had run since
July 18, the GPU `4` services since July 27 and August 7, the GPU `5`
server since August 8, and the GPU `6` services since August 7 and
August 8. Their users included `zengjun*`, `root`, and `wangmingfa*`;
none matched the strict-P1 run tag or remote run root. They therefore
remain outside the authorized cleanup scope.

At `2026-08-09 23:31 CST`, sample `2760` showed additional compute
processes on every fixed GPU. GPU `2` had fallen to about `13.53 GiB`
free, while GPUs `4,5,6` had about `37.67`, `28.32`, and `44.70 GiB`
free. A read-only process audit identified the four new PIDs as
`root`-owned Qwen3.5-4B RULER evaluation shards launched from a shared
parent, not the strict-P1 run tag or remote run root. They are outside
the authorized cleanup scope. The launchd monitor and ControlMaster
remained healthy, and no launch or terminal artifact existed.

At `2026-08-10 01:36 CST`, sample `2881` showed that all four transient
RULER shards had exited naturally. The fixed GPUs returned to their
earlier long-lived blocker sets with `4,4,1,3` compute processes and
about `66.61`, `52.96`, `77.89`, and `58.85 GiB` free. The Kerberos
cache had renewed through `2026-08-10 11:33 CST`; both the existing
ControlMaster and an independent fresh batch-mode SSH connection
succeeded. The launch path therefore remained available, but the
zero-process gate was still not satisfied.

At `2026-08-10 05:38 CST`, sample `3114` still showed the same
long-lived blocker PID sets, `4,4,1,3` compute-process counts, and
free-memory values on GPUs `2,4,5,6`. The launchd service remained
running as PID `54316`, and no launch attempt or terminal artifact
existed.

At `2026-08-10 10:22 CST`, sample `3388` still showed the same
long-lived blocker sets with `4,4,1,3` compute processes and about
`66.61`, `52.96`, `77.89`, and `58.85 GiB` free on GPUs `2,4,5,6`.
The launchd service, Kerberos cache, and SSH ControlMaster were healthy.
No launch attempt or terminal artifact existed. The foreground sleep
observer was stopped to avoid consuming another unified exec slot; the
independent launchd monitor continued unchanged.

At `2026-08-10 16:00 CST`, the complete ledger through sample `3714`
was independently reparsed and the strict gate was recomputed from the
raw per-GPU rows rather than trusting the recorded classification.
All `3714` rows were parseable, all were `BLOCKED_RESOURCES`, and zero
rows satisfied the requirement that GPUs `2,4,5,6` each have at least
`25 GiB` free and zero compute processes. The ledger covered
`230915.78` seconds from `2026-08-07 23:51:34 CST` through
`2026-08-10 16:00:10 CST`; its median interval was `62.16` seconds and
its maximum interval was only `66.44` seconds, with no gap over
`180` seconds. This rules out a long sampling outage during the
monitored period, although an idle window shorter than one sampling
interval cannot be disproved.

The same full-ledger audit found a structural blocker rather than a
rare scheduling miss. GPU `2` never had fewer than four compute
processes, GPU `4` never had fewer than four, and GPU `6` never had
fewer than two. Four GPU `2` PIDs and four GPU `4` PIDs appeared in all
`3714` samples; two GPU `6` PIDs also appeared in every sample. A fresh
read-only `ps` query identified them as unrelated long-lived services
owned by `zengjun*` and `root`, with start times ranging from
`2026-07-18` through `2026-08-07`. They do not match the strict-P1 run
tag or remote run root and therefore cannot be terminated by the
monitor. Under the frozen fixed-GPU and zero-compute-process
requirements, the gate is unlikely to become reachable unless those
service owners release GPUs `2,4,6`. No launch-attempt, monitor-result,
or monitor-failure artifact existed at the time of this audit.

The strict-P1 workload is an inference correctness/performance
benchmark, not model training. Its frozen canonical matrix contains
`70` serial cases: five workloads, two profiles, one warmup repetition,
one correctness repetition, and five measured repetitions. The longest
prefix-plus-suffix input is `3904` tokens. Based on that workload shape,
the expected runtime after acquiring the four GPUs is approximately
`30-60` minutes, with `90` minutes as a conservative end-to-end
allowance for staging, assembly, download, and verification. The worker
command itself has a one-hour timeout.
