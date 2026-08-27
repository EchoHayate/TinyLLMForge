# Qwen3.8-27B TP4 Communication-Exposure Audit

## Outcome

The frozen producer campaign is complete and classifies the attempt as:

```text
INCONCLUSIVE_LOW_HEADROOM
```

The measured traces contain substantial exposed communication, but the
maximum matched Nsight overhead is `38.194519%`, far above the frozen `3%`
promotion ceiling. The trace therefore identifies a possible optimization
target but is too perturbative to authorize a communication/computation
overlap implementation. The remote and local independent verifiers agree
exactly after separately replaying the same 25 immutable traces.

Current immutable identities:

```text
authoritative checkout:
  /Users/bytedance/Desktop/TinyLLMForge
branch:
  feat/kv-sparse-attention
runtime source revision:
  549fef12dcfdab842af99ff09ce1847b623cdbad
runtime source-tree SHA-256:
  dfdf6e758cbaa52fa24d8fa99550a709a8bf8bf81f8bc6d3f53842ec9c1a0654
attempt:
  20260826-qwen38-tp4-communication-profile-r9
model repository:
  Qwen/Qwen3.8-27B
model revision:
  1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
model manifest SHA-256:
  8f1520d552b7c9bbbadabe27d0e9632f25da9c6fc1d7ff7e43fb7471edc1316a
remote approved root:
  /data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818
```

The artifact assembler and independent verifier are carried by later
controller-only commits and do not change the frozen runtime source tree:

```text
45a61acf834d2607af9c442c6ca14af0869bdc65
dad5ebea2a1ede6db331fa4d2f2a9876e6208d72
d14ff289effd3a1fd9cfb41c5e0434837ff959ff
6b9eae255165bfab4cdc843ff2e09b705844204e
026d8c6dec89beb89d0d3f0b54dcd76da810790f
cf68649bc98c571e5950f8ad39396616c75c7631
```

## Prompt-to-artifact checklist

| Requirement | Concrete evidence | Verification | Verdict |
| --- | --- | --- | --- |
| Use only the authoritative checkout and branch | Git branch, origin, and pushed commits above | Local Git inspection and GitHub push receipts | PASS |
| Keep all remote task data below the approved mounted root | The complete r9 attempt path is below the approved root | Controller path guards plus remote path inventory | PASS |
| Do not use the retired adaptive-ngram checkout | No r9 source, controller, trace, or bundle path names that checkout | Exact-path source and artifact inspection | PASS |
| Bind the official immutable Qwen3.8-27B checkpoint | `artifacts/correctness/model_manifest.json` and final bundle `model_manifest.json` | Hash and semantic verification | PASS |
| Bind the frozen runtime source | `controller/environment_identity.json`, correctness source manifest, and final bundle source manifest | Revision plus source-tree SHA-256 comparison | PASS |
| Establish official TP1, TinyLLMForge TP1, and TinyLLMForge TP4 correctness | `controller/correctness_controller_result.json` and `artifacts/correctness/` | Exact generated tokens, argmax/top-k, finite logits, shard and cleanup checks | PASS |
| Admit exactly four strict-clean GPUs at controller entry | `controller/gpu_admission_samples.jsonl`, structured admissions, and `controller/nsys-admission.json` | Four UUIDs, memory `<=1024 MiB`, utilization `<=5%`, no compute process | PASS |
| Prove worker-entry identity without synthesizing repeated snapshots | `controller/structured-resource-samples.raw.jsonl`, `controller/nsys-resource-samples.raw.jsonl`, and 61 timestamped entries in final `gpu_topology.json` | PID-linked pre-GPU-mutation sample reconstruction | PASS producer + remote verifier |
| Preserve the frozen P0/P1/Q0/Q1/Q2 matrix | `artifacts/structured/cases/` and final `workload_manifest.json` | Exact prompt/output/concurrency/order checks | PASS |
| Complete two warmups and five structured measured repetitions per workload | `controller/structured-resume-receipt.json` and 35 structured case JSON files | Exact case inventory and classifications | PASS |
| Complete all 25 Nsight replays | `artifacts/nsys_replay/cases/`, 25 `nsys/*.sqlite` files, and `controller/nsys-receipt.json` | Exact inventory, fresh `PRAGMA quick_check`, and required non-empty table checks | PASS |
| Correlate operations by structured identity, not kernel-name guesses | Final `profile_rows.jsonl` plus all 25 SQLite traces | Producer NVTX/operation/rank/layer correlation; independent recomputation | PASS producer + remote verifier |
| Compute interval unions, exposed NCCL, overlap, idle, and critical path | `layer_summary.json`, `communication_exposure_summary.json`, and 140 raw profile rows | Producer and independent recomputation | PASS producer + remote verifier |
| Report QPS, output tokens/s, TTFT, TPOT, E2E, memory, utilization, and power | `online_metrics.json`, `memory_summary.json`, and 100 `resource_samples.jsonl` rows | Inventory and aggregation checks | PASS producer + remote verifier |
| Pair all 25 profiled controls with identical unprofiled cases | `controller/nsys-receipt.json` and final `online_metrics.json` | Source/model/rank/GPU/workload/repetition equality | PASS |
| Keep profiler overhead at or below 3% for GO | Final producer and verifier summaries | Maximum of all 25 paired relative-overhead rows | FAIL promotion gate: `38.194519%` |
| Produce a complete immutable artifact manifest | Final `manifest.sha256` with 39 post-verification artifacts, including 25 traces | Remote and local verifiers rehash every required file and reject extras | PASS: 39/39 remote and local |
| Obtain producer/verifier agreement | `communication_exposure_summary.json`, remote verifier JSON, and local verifier JSON | Independent semantic recomputation plus byte/hash comparison | PASS: producer, remote verifier, and local verifier all reconstruct `INCONCLUSIVE_LOW_HEADROOM` |
| Reproduce the verifier from the local Mac without retaining 61.97 GiB of traces | Local non-Nsight bundle plus one-at-a-time gzip-over-SSH staging from remote `final_bundle/nsys` | Each trace is locally decompressed, hashed, parsed, compared, then deleted; final JSON is byte-identical to the remote verifier result | PASS: 25/25 traces, zero stderr, no retained temp trace |
| Report both benefit and cost | Final report and this audit | Exposure/headroom and online metrics paired with profiler overhead/resource cost | PASS producer + both verifiers |
| Enforce conditional continuation | Exact terminal classification | No overlap design unless both authorities return GO | ENFORCED |

## Frozen workloads and GPU mapping

```text
P0: prompt=256,  output=128, concurrency=1, family=causal
P1: prompt=2048, output=128, concurrency=1, family=causal
Q0: prompt=256,  output=128, concurrency=4, family=online
Q1: prompt=256,  output=128, concurrency=8, family=online
Q2: prompt=2048, output=128, concurrency=4, family=online

rank 0 -> physical GPU 2 -> GPU-63c05907-407b-8240-07a0-f38872840867
rank 1 -> physical GPU 3 -> GPU-f8904cb4-f9f0-c757-df36-e6fd971b3a9d
rank 2 -> physical GPU 4 -> GPU-56b882d2-6e6e-adb3-80e7-95f0a9e678f1
rank 3 -> physical GPU 5 -> GPU-687b7858-ca44-98ad-cfba-b6785eaf05e8
```

Every run uses BF16, TP4, greedy temperature-zero decoding, exactly 128
output tokens, and identical scheduler and CUDA Graph policy.

## Worker-entry evidence reconciliation

The first assembler draft repeated one `nsys-admission.json` inventory under
all 61 logical worker IDs. That representation passed the old schema but did
not prove independent entry observations and therefore was not accepted as
terminal evidence.

Commit `dad5ebea2a1ede6db331fa4d2f2a9876e6208d72` changes the bundle contract
to reconstruct each logical entry from timestamped raw evidence:

- correctness uses the final stage-specific row in
  `controller/gpu_admission_samples.jsonl`;
- structured cases are linked by their result PID to the corresponding launch
  in `controller/structured-resource-samples.raw.jsonl`;
- Nsight cases are linked by their result PID to the corresponding successful
  launch in `controller/nsys-resource-samples.raw.jsonl`;
- cases served by one persistent worker intentionally share the same captured
  entry timestamp instead of pretending that a new snapshot was taken;
- missing PID linkage, missing timestamp, dirty inventory, UUID drift, or
  incomplete logical coverage fails bundle assembly or independent
  verification.

The final producer bundle contains exactly 61 worker-entry inventories:
one correctness entry, 35 structured entries, and 25 Nsight replay entries.
Every entry names its raw capture source, pre-GPU-mutation stage, timestamp,
and the same four strict-clean UUIDs.

## Terminal producer profiler-overhead observation

All 25 matched profiled/unprofiled controls completed:

```text
all 25 pairs median:  30.773947%
all 25 pairs maximum: 38.194519%
P0 median:           28.060698%
P1 median:           29.556759%
Q0 median:           36.688667%
Q1 median:           34.557005%
Q2 median:           33.943138%
```

The maximum, not the median, is the frozen promotion statistic. It exceeds the
`<=3%` requirement by `35.194519` percentage points. The threshold was not
retuned after observing the result.

Commit `d14ff289effd3a1fd9cfb41c5e0434837ff959ff` preserves that threshold
while aligning verifier behavior with the approved terminal classifier:
profiler overhead above three percent prevents `GO`, but it does not suppress
`independent_verification.json`. The verifier must still reconstruct and
record the matching non-GO terminal classification.

## Terminal producer benefit and cost

Communication-exposure signal:

| Workload | Exposed communication | Overlap headroom lower bound | Representative repetition |
| --- | ---: | ---: | ---: |
| P0 | 21.300382% | 10.446015% | 3 |
| P1 | 13.359492% | 10.423242% | 4 |
| Q0 | 34.145316% | 10.644781% | 0 |
| Q1 | 38.818101% | 10.602486% | 2 |
| Q2 | 27.848422% | 10.773773% | 1 |

Unprofiled online baseline and resource cost:

| Workload | Median QPS | Median output tok/s | Median TTFT (ms) | Median TPOT (ms) | Median E2E (ms) | Max allocated / reserved (GiB) | Median GPU util. | Median power (W) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P0 | 0.032117 | 4.111012 | 635.805 | 245.164 | 31780.231 | 73.323 / 75.865 | 94% | 121.320 |
| P1 | 0.032570 | 4.168908 | 2457.572 | 241.760 | 33200.397 | 75.605 / 77.385 | 100% | 184.065 |
| Q0 | 0.034752 | 4.448205 | 3394.788 | 906.320 | 118497.385 | 75.699 / 76.484 | 100% | 130.355 |
| Q1 | 0.035022 | 4.482843 | 5818.871 | 1798.634 | 234317.963 | 75.862 / 77.418 | 100% | 141.370 |
| Q2 | 0.034509 | 4.417161 | 7152.671 | 912.689 | 123089.459 | 77.259 / 77.570 | 100% | 275.565 |

The benefit numbers are diagnostic overlap opportunity, not realized speedup.
The cost number is profiler perturbation: maximum paired overhead
`38.194519%`. Therefore this bundle does not prove that an overlap
implementation would improve production latency or throughput.

## Terminal classification and claim boundary

Producer:

```text
INCONCLUSIVE_LOW_HEADROOM
```

Remote independent verifier:

```text
status:                       PASS
producer classification:      INCONCLUSIVE_LOW_HEADROOM
reconstructed classification: INCONCLUSIVE_LOW_HEADROOM
profile rows:                 140
correctness rows:             100
Nsight traces:                25
strict-clean worker entries:  61
profiler overhead:            38.194519%
stderr bytes:                 0
```

Local independent verifier using bounded one-trace-at-a-time staging:

```text
status:                       PASS
producer classification:      INCONCLUSIVE_LOW_HEADROOM
reconstructed classification: INCONCLUSIVE_LOW_HEADROOM
profile rows:                 140
correctness rows:             100
Nsight traces:                25 / 25
strict-clean worker entries:  61
cleanup valid:                true
trace coverage complete:      true
stderr bytes:                 0
result SHA-256:
  c07348a63f02603fb8b1f99ae9b559dc80e1abe94ac2751a91288559e7e85e13
remote/local JSON:
  byte-identical
post-verification artifacts:  39 / 39
temporary trace directories:  0
```

Commit `026d8c6dec89beb89d0d3f0b54dcd76da810790f` closes every streamed
SQLite connection explicitly and preserves both fresh and resumed PASS
records. This prevents an unlinked multi-GiB trace from remaining open after
each validation and permits interruption-safe continuation without retaining
the full 61.97 GiB trace inventory on the Mac.

Commit `cf68649bc98c571e5950f8ad39396616c75c7631` adds the bounded
gzip-over-SSH copy mode used for the terminal local replay. The remote side
compresses one immutable trace to stdout, the local side streams it through a
decompressor into an attempt-scoped partial file, and only an exit-zero stream
is atomically promoted for hash and SQLite verification.

During the local replay on August 27, 2026, the existing
`jump-proxy-hl` ControlMaster disappeared while copying `Q1-r3`. The verifier
exhausted its bounded retries and removed the partial trace. The Kerberos
ticket remained valid; a read-only path probe localized the failure to the HL
proxy, while `jump-proxy-lf` reached the same host and user. A replacement
ControlMaster through LF resumed from the persisted 18/25 PASS records and
completed 25/25 without repeating the frozen campaign or retaining a failed
partial.

No overlap design or implementation is authorized. In particular, this
attempt does not authorize asynchronous collectives, a communication stream,
chunked ReduceScatter/AllGather, or new CUDA Event dependencies.
