# Cross-Engine Exact-Greedy K8 Benchmark Design

Date: 2026-08-29

## Status

Approved for design under the standing autonomous benchmark authorization.
Implementation and remote execution remain gated on this committed design and
its subsequent implementation plan.

## Objective

Determine whether TinyLLMForge's exact-greedy K8 runtime has a horizontal
performance advantage over a pinned vLLM release under the same physical GPU,
model checkpoint, numerical configuration, prompts, output lengths, sampling
semantics, warmup policy, and measurement procedure.

The benchmark must distinguish:

1. TinyLLMForge's within-engine K8 improvement over its own host-greedy path;
2. vLLM's within-engine multi-step improvement when the pinned public release
   exposes a supported multi-step control; and
3. absolute TinyLLMForge K8 performance versus the strongest eligible vLLM
   greedy configuration.

The first campaign is limited to Qwen3-0.6B, BF16, TP1, batch 1. It does not
authorize Qwen3-8B, TP4, SGLang, TensorRT-LLM, online concurrency, speculative
decoding, prefix caching, quantization, or production-default claims.

## Decision

Use two isolated remote Python environments under the approved `/data00`
campaign root, one for TinyLLMForge and one for vLLM. Both environments reuse
one existing read-only Qwen3-0.6B checkpoint. Full environments, package
caches, source snapshots, raw logs, and complete run bundles remain remote.
The Mac retains only source code and a compact allowlisted evidence bundle.

This is preferred over:

- containers, which provide stronger filesystem isolation but duplicate image
  layers and make the 20 GB campaign budget unnecessarily difficult; and
- one shared Python environment, which consumes less space but makes PyTorch,
  FlashAttention, Triton, and vLLM dependency conflicts part of the measured
  system.

## Storage Boundary

All new remote data must be below:

```text
/data00/home/sitian/tinyllmforge-workspaces/
  command-timeline-20260818/
    cross-engine-k8-qwen3-06b/
```

The layout is:

```text
cross-engine-k8-qwen3-06b/
  shared/
    model-pointer.json
    package-cache/
  envs/
    tinyllmforge/
    vllm/
  sources/
    tinyllmforge-<commit>/
    vllm-<commit-or-release>/
  attempts/
    <immutable-run-tag>/
      controller/
      tinyllmforge/
      vllm/
      remote-final/
  monitor/
```

The model pointer references the existing checkpoint:

```text
/data00/home/sitian/.ms_cache/Qwen/Qwen3-0___6B
```

The checkpoint must not be copied, hard-linked, repacked, or downloaded again.
The controller records its canonical path, revision metadata when available,
per-file inventory, and aggregate manifest digest.

No task data may be written to remote `/`, remote `/tmp`, an old remote
checkout, or the Mac's `experiments/` tree.

## Space Budget

The campaign has a 20 GiB hard limit for newly owned remote files, excluding
the pre-existing shared checkpoint target.

Budget enforcement uses actual allocated campaign bytes:

```text
warning threshold: 16 GiB
hard stop:          20 GiB
```

The controller records:

- `df` for `/` and `/data00/home/sitian`;
- campaign-root bytes before environment creation;
- bytes after each environment installation;
- bytes before and after every benchmark attempt;
- bytes before finalization; and
- the largest individual files and directories owned by the campaign.

At or above 20 GiB, no installation, source staging, model launch, benchmark,
or retry may begin. A running benchmark may finish writing its current atomic
artifact, after which the controller classifies the campaign
`INCOMPLETE_STORAGE_BUDGET`.

Every cache-bearing environment variable is explicitly redirected under the
campaign root:

```text
XDG_CACHE_HOME
HF_HOME
MODELSCOPE_CACHE
PIP_CACHE_DIR
UV_CACHE_DIR
TRITON_CACHE_DIR
TORCHINDUCTOR_CACHE_DIR
CUDA_CACHE_PATH
PYTHONPYCACHEPREFIX
TMPDIR
```

The process does not redefine `HOME`.

## Local Retention Boundary

The Mac must not retain:

- either remote virtual environment;
- Python wheels or package caches;
- model files;
- source archives;
- full stdout or stderr logs;
- Nsight traces;
- CUDA cache files;
- raw memory polling streams; or
- complete remote run directories.

The local allowlist is retained only below:

```text
artifacts/cross_engine_k8/<run-tag>/
controller_manifest.json
environment_manifest.json
workload_manifest.json
case_rows.jsonl
correctness_rows.jsonl
comparison.json
summary.json
gate.json
remote_verification.json
local_verification.json
manifest.sha256
```

The retained local bundle must be no more than 50 MiB. The expected size is
below 10 MiB. The local verifier may stream additional remote files one at a
time, hash and parse them, and discard the temporary copy immediately. It
must leave zero streamed temporary files after success or failure.

## Environment Identity

The controller creates separate remote environments:

```text
envs/tinyllmforge
envs/vllm
```

The TinyLLMForge environment is pinned to the authoritative source commit and
records Python, PyTorch, CUDA runtime, Triton, FlashAttention, and package
versions. Its probe is engine-specific and must not import vLLM from the
separate comparison environment.

The vLLM environment uses the newest stable upstream release that satisfies
all of the following compatibility checks on the target host:

1. supports the installed NVIDIA driver and CUDA runtime;
2. loads the shared Qwen3-0.6B checkpoint without conversion;
3. supports BF16, TP1, greedy decoding, explicit prompt token IDs, fixed
   output length, and disabled prefix caching;
4. exposes per-request timing sufficient to reconstruct TTFT, TPOT, and E2E;
5. does not require a model copy or container image; and
6. completes the frozen smoke workload with exact output length.

Candidate stable releases are evaluated newest-first. The first compatible
release is frozen by version, source revision, installed-distribution hashes,
and complete `pip freeze`. If no stable release passes, the campaign is
`INCOMPLETE_VLLM_COMPATIBILITY`; the controller must not silently patch vLLM
or fall back to an unpinned development head.

For the admitted host with NVIDIA driver `535.261.03`, candidate discovery is
bounded at vLLM `0.11.2`. Newer discovered releases either require a CUDA 13
dependency generation that the admitted driver cannot support or expose no
wheel matching the host's supported Python/platform tags. The vLLM candidate
environment is fully isolated from system site-packages and installs the
verified wheel together with its declared binary dependencies. `pip` caching
is disabled to avoid retaining both archives and installed files; all
temporary extraction remains under the campaign `TMPDIR`.

If the pinned public vLLM version exposes a supported multi-step toggle, the
matrix includes both disabled and enabled arms. Otherwise it records
`VLLM_MULTI_STEP_NOT_PUBLICLY_AVAILABLE` and retains only the strongest
supported default greedy arm.

## Benchmark Arms

Required arms:

```text
tinyllmforge_host_greedy
tinyllmforge_exact_k8
vllm_default_greedy
```

Conditional arm:

```text
vllm_public_multi_step
```

No engine source is modified to manufacture an equivalent arm. Public
configuration is allowed; private monkeypatching is not.

## Frozen Workload

The benchmark reuses the established exact-burst shape:

```text
model:               Qwen3-0.6B
precision:           BF16
tensor parallel:     1
batch size:          1
temperature:         0
ignore EOS:          true
generated tokens:    128
prompt token counts: 256, 2048, 8192
warmups:              2 per engine process
measured repetitions: 7 per arm and context bucket
```

Prompt token IDs are generated once, stored in `workload_manifest.json`, and
passed directly to both engines. No engine-specific tokenizer invocation may
change the measured prompt.

Each repetition uses one fresh engine process per arm. One process runs all
three context buckets after its warmups. Arm order follows a deterministic
balanced rotation so thermal or temporal drift does not always favor one
engine.

GPU admission requires:

- one physical NVIDIA A100 80GB PCIe;
- zero compute processes belonging to other users;
- zero active compute utilization at both admission checks;
- no unauthorized memory occupancy; and
- no process termination, eviction, or cleanup outside the campaign's own
  recorded process group.

If no GPU satisfies admission, the controller waits and monitors. It does not
kill external work.

## Metrics

Every measured row records:

- TTFT;
- median, P95, and P99 TPOT;
- E2E latency;
- output tokens per second;
- exact output length;
- externally sampled per-process peak GPU memory through NVML;
- engine-reported allocated and reserved GPU memory when exposed;
- process RSS;
- CPU time;
- GPU utilization samples;
- CUDA Graph launch or replay count when exposed without intrusive tracing;
- intermediate and final D2H counts when exposed;
- engine, source, environment, model, workload, GPU, and repetition identity;
  and
- raw monotonic timestamps used to reconstruct all latency metrics.

Metrics unavailable through a public engine interface are marked
`NOT_EXPOSED`; they are never imputed as zero.

No Nsight trace is required for the first campaign. Profiler instrumentation
whose paired overhead exceeds 3% is excluded from the performance verdict.

## Correctness

For every context bucket:

- TinyLLMForge K8 must exactly match TinyLLMForge host-greedy token IDs,
  decoded-text hash, sampled argmax, and retained float32 logits at the frozen
  sampling points.
- vLLM must emit exactly 128 tokens and match the frozen greedy token IDs.
- Any cross-engine token mismatch is a correctness failure, not a performance
  sample.

Performance rows from a failed correctness arm remain diagnostic and are
excluded from advantage classification.

## Comparisons

The result reports two independent views.

### Within-engine mechanism gain

```text
TinyLLMForge K8 / TinyLLMForge host-greedy
vLLM public multi-step / vLLM default greedy
```

This determines whether TinyLLMForge's K8 organization removes a larger
fraction of its own baseline overhead.

### Absolute engine result

```text
TinyLLMForge exact K8 / strongest eligible vLLM greedy arm
```

This determines whether a user receives better latency and throughput from
the complete TinyLLMForge path, regardless of which internal component is
responsible.

## Classification

`GO_CROSS_ENGINE_ADVANTAGE` requires:

- complete required workload and correctness matrices;
- producer, remote verifier, and frozen-source local verifier agreement;
- TinyLLMForge K8 aggregate median TPOT at least 5% lower than the strongest
  eligible vLLM arm;
- TinyLLMForge K8 aggregate output throughput at least 5% higher;
- no context bucket with worse median TPOT;
- TTFT, E2E, P95 TPOT, P99 TPOT, peak GPU memory, and process RSS regressions
  no greater than 2%;
- exact output equality; and
- complete storage-budget compliance.

`CROSS_ENGINE_PARITY` means aggregate TPOT and throughput are both within 5%
and all protected metrics remain within 2%.

`NO_CROSS_ENGINE_ADVANTAGE` means the required matrix is complete and valid,
but TinyLLMForge fails the GO and parity rules.

`INCOMPLETE` is used for environment incompatibility, GPU unavailability,
SSH failure, storage-budget refusal, missing rows, correctness mismatch,
worker failure, source drift, verifier disagreement, or absent terminal
receipts.

No threshold may be changed after measured rows exist.

## Controller and Failure Semantics

The local controller:

1. requires `KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian`;
2. acquires a non-blocking local file lock keyed by remote host and immutable
   attempt tag before any stage can mutate remote state;
3. fails fast when the Kerberos lifetime cannot cover the estimated remaining
   campaign duration plus 30 minutes;
4. performs read-only remote storage and environment discovery;
5. creates only the approved campaign root;
6. builds environments and source snapshots atomically;
7. checks storage and strict-clean GPU admission before every worker;
8. launches one owned process group;
9. records bounded heartbeat and resource state;
10. terminates only owned processes after timeout;
11. finalizes immutable artifacts under a fresh run tag;
12. runs the remote independent verifier;
13. streams the allowlisted evidence and any one-at-a-time verifier inputs;
14. runs the frozen-source local verifier; and
15. confirms zero local temporary retention and remote/local receipt equality.

An interrupted immutable run is never overwritten or resumed unless its
controller protocol explicitly supports safe resumption. A new attempt uses a
new tag and reuses the already installed, source-identified environments.

## Testing Strategy

Dependency-light local tests cover:

- approved path enforcement;
- rejection of `/`, `/tmp`, old checkouts, and local experiment destinations;
- cache-variable redirection;
- no `HOME` override;
- model-copy refusal;
- 16/20 GiB budget boundaries;
- local 50 MiB allowlist enforcement;
- version and environment manifests;
- deterministic arm rotation;
- metric reconstruction;
- correctness exclusion;
- classification thresholds;
- immutable attempt tags;
- owned-process-only cleanup;
- Kerberos cache and lifetime handling;
- SSH failure behavior;
- verifier tamper detection; and
- streamed temporary-file cleanup.

Remote execution proceeds in three stages:

1. storage and compatibility preflight with no GPU model load;
2. one-context, one-repetition smoke for all eligible arms; and
3. the full frozen canonical matrix.

The smoke is not performance evidence. Only the complete canonical matrix and
agreeing verifiers can classify horizontal advantage.

## Claim Boundary

A GO supports only:

> On the frozen Qwen3-0.6B BF16 TP1 batch-1 greedy workload and admitted A100,
> TinyLLMForge Exact K8 outperformed the pinned compatible vLLM greedy arm
> under the recorded thresholds.

It does not establish:

- global superiority over vLLM;
- online serving superiority;
- batching or concurrency superiority;
- Qwen3-8B or Qwen3.8-27B performance;
- tensor-parallel superiority;
- SGLang or TensorRT-LLM superiority;
- production-default readiness; or
- academic novelty.
