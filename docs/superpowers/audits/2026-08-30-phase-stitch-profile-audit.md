# Phase-Stitch Profile Completion Audit

Date: 2026-08-30

## Terminal conclusion

The source-bound Stage 0 profile gate completed on one strictly clean
NVIDIA A100 80GB PCIe and returned:

```text
GO_PHASE_STITCH_PROFILE
```

The measured host-controlled interval between first-token host visibility and
the first Exact Greedy K8 dispatch has a median of 0.817 ms for the 256-token
prompt and 0.989 ms for the 2048-token prompt. The interval is only about
0.27-0.29% of end-to-end latency, but its P95 is 0.860 ms and 1.276 ms,
respectively. The frozen gate therefore passes through the P95 branch, not
through the 3% median-share branch.

Profiling changed median E2E by +0.568% and +0.838%, both within the 1%
ceiling. All generated token IDs and text hashes matched, all seven events
were present for every retained profiled request, both underlying graph paths
replayed, and capture, replay, lease, and quarantine failures were zero.

This result authorizes implementation of the default-disabled Stage 1
Phase-Stitched Exact Graph Runtime. It does not prove that Stage 1 will meet
its 3% per-shape and 2% aggregate E2E improvement thresholds. In particular,
the measured removable interval is smaller than those thresholds when viewed
as a fraction of the current 128-token request E2E.

## Prompt-to-artifact completion checklist

- [x] Used the sole authoritative checkout
      `/Users/bytedance/dev/TinyLLMForge`; the Desktop path is only its
      symlink.
- [x] Profiled the approved prefill-to-K8 boundary before modifying runtime
      transaction semantics.
- [x] Kept `phase_stitch_profile` disabled by default and strictly typed as a
      boolean.
- [x] Used `time.perf_counter_ns` and added no CUDA synchronization to either
      benchmark arm.
- [x] Captured the fixed seven-event lifecycle from final prefill completion
      through first K8 dispatch.
- [x] Defined the primary metric as
      `first_k8_dispatch_started_ns - first_token_host_available_ns`.
- [x] Used Qwen3-0.6B BF16, TP1, batch one, temperature zero,
      `ignore_eos=true`, completion-only execution, 128 generated tokens,
      Exact Prefill Graph, and Exact Greedy K8.
- [x] Used prompt lengths 256 and 2048, two AB/BA rounds, two warmups, and
      five measured repetitions per case.
- [x] Used a fresh engine for each of the eight cases.
- [x] Admitted physical GPU 0 only after observing 0 MiB used, 0% utilization,
      and no compute processes.
- [x] Reused the approved remote Python and Qwen3-0.6B model; no model was
      downloaded.
- [x] Kept all remote task state under
      `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/`.
- [x] Did not terminate, adopt, or alter an external GPU process.
- [x] Bound the run to pushed source commit
      `aa3fe145c8187eeee5d1ef8b6aeaf25bbb43b44f`.
- [x] Bound the run to ten exact source-file hashes and the frozen contract
      SHA-256.
- [x] Retained all eight case specifications, eight case results, and eight
      zero exit receipts.
- [x] Retained producer and independent-verifier JSON plus zero exit
      receipts.
- [x] Re-ran the independent verifier locally against the downloaded bundle.
- [x] Verified exact token/text equality, complete event coverage, positive
      prefill and K8 replay evidence, and zero failures/quarantines.
- [x] Evaluated both the absolute median-gap threshold and the alternative
      median-share/P95 threshold exactly as frozen.
- [x] Evaluated instrumentation overhead against the unprofiled control arm.
- [x] Reported both potential benefit and the observed measurement overhead.
- [x] Did not implement or claim sentinel-filled prefill graph buckets.
- [x] Preserved failed immutable attempts r1-r3 as non-authoritative history;
      only r4 is used as terminal evidence.

## Source and run identity

```text
branch:
  feat/kv-sparse-attention
source base commit:
  aa3fe145c8187eeee5d1ef8b6aeaf25bbb43b44f
run tag:
  20260830-qwen3-06b-r4
contract SHA-256:
  57897bbd8a083f3c519bb7d7dfa31d7ef1633217483f117072b3d5bb068ba6a6
manifest SHA-256:
  13bdc2aecb15ed1baede9044cfa32895b9541a3ddbb407448c75cd5ad36c484c
summary SHA-256 recorded by gate:
  d45f48347e4214fa96fc60eda810ab21d55f3235a1032467992888a7ce353493
```

The manifest's own file digest is not the same object as the
`summary_sha256` recorded by the producer gate. The former hashes the complete
manifest JSON; the latter is the producer's canonical summary identity. The
independent verifier reconstructed and accepted both identities.

## Environment and frozen workload

```text
host:
  sitian@10.232.195.203
remote Python:
  /data00/home/sitian/tllm/env/bin/python
model:
  /data00/home/sitian/.ms_cache/Qwen/Qwen3-0___6B
GPU:
  NVIDIA A100 80GB PCIe
physical GPU index:
  0
GPU UUID:
  GPU-57be086f-e967-c022-3832-93df4fc77bd0
clean admission:
  0 MiB used, 0% utilization, no compute processes
tensor parallel:
  1
batch size:
  1
prompt tokens:
  256, 2048
generated tokens:
  128
temperature:
  0.0
ignore EOS:
  true
completion-only:
  true
zero-temperature greedy fast path:
  true
rounds:
  2, AB/BA
warmups:
  2 per case
measured repetitions:
  5 per case
samples:
  10 per arm and prompt shape
```

## Measured benefit ceiling and profiling cost

| Prompt | Median removable gap | P95 removable gap | Median gap / E2E | Profile-off median E2E | Profile-on median E2E | Profiling cost |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 0.817 ms | 0.860 ms | 0.268% | 303.023 ms | 304.746 ms | +0.568% |
| 2048 | 0.989 ms | 1.276 ms | 0.294% | 333.824 ms | 336.621 ms | +0.838% |

The direct opportunity is roughly one millisecond at the measured boundary.
The Stage 1 composition may also avoid repeated metadata construction and
host launch work not perfectly represented by this interval, but such
secondary benefit is a hypothesis until the four-arm runtime gate measures
it.

## Gate evaluation

| Requirement | Threshold | Observed | Verdict |
| --- | ---: | ---: | --- |
| Median removable gap | at least 0.150 ms for one shape | 0.817 ms, 0.989 ms | PASS |
| Median share or P95 gap | at least 3% or at least 0.500 ms | 0.268% / 0.860 ms; 0.294% / 1.276 ms | PASS via P95 |
| Event coverage | complete | complete for every retained profiled row | PASS |
| Profiling E2E perturbation | absolute change at most 1% | +0.568%, +0.838% | PASS |
| Exact generated tokens/text | required | exact in all paired samples | PASS |
| Prefill and K8 graph evidence | required | positive replay deltas | PASS |
| Capture/replay/lease failures | zero | zero | PASS |
| Quarantines or pending leases | zero | zero | PASS |

The producer's complete check set is:

```text
complete_case_inventory=true
exact_output_equality=true
event_coverage_pass=true
graph_evidence_pass=true
zero_failures_pass=true
ceiling_pass=true
overhead_pass=true
```

## Independent verification and retained artifacts

The compact local evidence directory is:

```text
artifacts/phase_stitch_profile/20260830-qwen3-06b-r4/
```

It contains `run_manifest.json`, eight immutable case results, eight case
specifications, eight zero case-exit receipts, `summary.json`, `gate.json`,
`manifest.json`, and producer/verifier receipts with zero exit codes.

The local independent-verifier command was:

```bash
python3 -m tools.phase_stitch_profile_verify \
  --run-dir artifacts/phase_stitch_profile/20260830-qwen3-06b-r4
```

Its terminal result was:

```json
{
  "classification": "GO_PHASE_STITCH_PROFILE",
  "contract_sha256": "57897bbd8a083f3c519bb7d7dfa31d7ef1633217483f117072b3d5bb068ba6a6",
  "manifest_sha256": "13bdc2aecb15ed1baede9044cfa32895b9541a3ddbb407448c75cd5ad36c484c",
  "verified": true
}
```

The remote primary and controller roots are:

```text
/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/
  phase-stitch-profile/runs/20260830-qwen3-06b-r4

/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/
  phase-stitch-profile/controller-verification/20260830-qwen3-06b-r4
```

## Failed-attempt reconciliation

- r1 exposed a controller/source-bundle preflight mismatch.
- r2 first hit a transient SSH transport failure; its safe retry then exposed
  incorrect K8 acceptance-counter ownership in the worker.
- r3 hit another SSH preflight interruption.
- r4 used a task-owned SSH ControlMaster and the corrected scheduler-owned K8
  acceptance counter, then completed the entire immutable matrix.

No failed or partial attempt is included in the formal performance result.

## Claim boundary

Stage 0 proves only that a measurable host-controlled prefill-to-K8 boundary
exists under the frozen Qwen3-0.6B/A100/TP1/batch-one workload and that the
profile instrumentation is sufficiently low-overhead to authorize Stage 1.

It does not prove:

- a Stage 1 end-to-end speedup;
- superiority over another inference engine;
- benefit under batching, streaming, sampling, EOS, stop strings, TP greater
  than one, offload, quantization, compact/sparse attention, or stateful
  model-forward paths;
- support for arbitrary prompt shapes;
- that the full measured gap can be removed;
- that the measured P95 opportunity will remain stable under a different
  workload or machine.

## Final classification

```text
PHASE_STITCH_PROFILE=COMPLETE
PRODUCER_CLASSIFICATION=GO_PHASE_STITCH_PROFILE
INDEPENDENT_VERIFIER=PASS
EXACT_OUTPUT_PARITY=PASS
PROFILE_OVERHEAD_CEILING=PASS
RUNTIME_IMPLEMENTATION_AUTHORIZED=true
FEATURE_IMPLEMENTED=false
SENTINEL_BUCKET_CLAIM=false
NEXT_ACTION=implement the default-disabled Stage 1 runtime under the frozen four-arm gate
```
