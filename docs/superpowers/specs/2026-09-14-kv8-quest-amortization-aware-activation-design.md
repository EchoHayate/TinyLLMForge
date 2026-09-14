# KV8 + Quest Amortization-Aware Activation Design

Date: 2026-09-14

## Objective

Turn the measured KV8 + Quest batch crossover into an auditable runtime
policy: use KV8 full attention when Quest's fixed selector cost cannot be
amortized, and enable Quest only when the current decode batch can avoid
enough KV-block dequantization work.

The candidate remains default-off. This gate is limited to deciding whether
the host-side policy preserves the useful half of the existing Quest result;
it is not a production qualification.

## Prior evidence

The fixed Quest top-16 gate on Qwen3-8B, A100 80GB PCIe, TP1, eager decode,
and 8192-token contexts established a clear crossover:

| Batch | KV8 full median step | Fixed Quest median step | Quest delta |
| ---: | ---: | ---: | ---: |
| 4 | 65.892 ms | 82.521 ms | 25.236% slower |
| 8 | — | — | 28.707% faster |
| 12 | — | — | 42.021% faster |
| 16 | — | — | 43.405% faster |
| 19 | — | — | 43.754% faster |

At batch 19, fixed Quest recovered 56.283% of KV8's excess latency over bf16
eager. The quality gate passed: bf16, KV8 full, and KV8 + Quest each retrieved
25 of 25 fixed needles.

The selector and top-k bookkeeping impose a fixed per-step cost. At batch 4,
that cost exceeds the dequantization saved by selecting 16 of 32 blocks. At
batch 8 and above, the avoided dequantization work dominates.

## Considered policies

### Fixed batch threshold

Enable Quest when `batch_size >= 8`.

This exactly matches the measured 8192-token grid, but it does not generalize
to different context lengths or top-k values. A batch of eight 17-block
sequences saves far less work than a batch of eight 64-block sequences.

### Cumulative avoided-dequant blocks

Enable Quest when the current batch can avoid at least a configured number of
block dequantizations:

```text
saved_blocks =
    sum(max(0, seq.num_blocks - quest_top_k_blocks) for seq in seqs)
```

This is the selected policy. It uses information already available on the
host, responds to both batch size and context length, has no historical state,
and is directly auditable.

For the frozen 8192/top-16 workload, each sequence has 32 visible blocks at
the start of decode. Therefore:

- batch 4 saves approximately 64 blocks;
- batch 6 saves approximately 96 blocks;
- batch 8 saves approximately 128 blocks.

The candidate threshold is frozen at `128`, so batches 4 and 6 fall back to
KV8 full attention while batches 8 and above activate Quest.

### Online latency model

Maintain an EMA of selector and dequantization cost and predict the cheaper
path for every step.

This could adapt to hardware and model changes, but it introduces cold-start
state, hysteresis, run-order dependence, and a substantially larger
verification surface. It is out of scope until the deterministic policy has
shown value.

## Runtime contract

Add one configuration field:

```python
quest_min_saved_blocks: int = 0
```

Its semantics are:

- `0` preserves the current Quest activation behavior exactly;
- a positive value adds an amortization gate after all existing Quest
  eligibility checks pass;
- negative values are invalid;
- the experimental candidate uses `128`.

`ModelRunner.prepare_decode()` remains the sole decision point. It first
applies the existing protections:

1. Quest is requested with `quest_top_k_blocks > 0`;
2. KV-Cartridge and AM compact are inactive;
3. all sequences meet `quest_min_seq_len`;
4. all sequences have more blocks than top-k;
5. top-k covers less than 80% of the longest sequence;
6. existing KV-offload incompatibility handling remains unchanged.

Only after those checks pass does the runner compute `saved_blocks`. Quest is
active for the step when:

```text
quest_min_saved_blocks == 0
or saved_blocks >= quest_min_saved_blocks
```

Otherwise the resolved `quest_top_k_blocks` sent through `set_context()` is
`-1`, and the step executes the existing KV8 full-attention path.

The decision is recomputed on every decode step. KV8 full and KV8 + Quest use
the same quantized KV cache, so changing the resolved path requires no cache
migration and changes no sequence state.

## Activation telemetry

Requested configuration is insufficient evidence because an adaptive arm can
silently execute either path. `ModelRunner` must publish a latest-step
host-derived activation observation containing:

- monotonically increasing observation id;
- decode batch size;
- requested top-k;
- resolved top-k;
- configured minimum sequence length;
- configured minimum saved blocks;
- computed saved blocks, or `null` if base eligibility failed;
- resolution reason.

Resolution reasons are frozen as:

- `disabled`;
- `incompatible_feature`;
- `below_min_seq_len`;
- `insufficient_blocks`;
- `insufficient_pruning`;
- `below_saved_blocks`;
- `active`.

The worker records one observation per decode step, including warmup, and
summarizes the measured window by reason, resolved top-k, and saved-block
range. Missing, repeated, or internally inconsistent observations make the
cell inconclusive. Telemetry must not add a GPU-to-host synchronization.

## Frozen performance experiment

Run three sequential arms on the same A100:

- KV8 full attention;
- fixed Quest top-16 with `quest_min_saved_blocks=0`;
- adaptive Quest top-16 with `quest_min_saved_blocks=128`.

All arms use:

- Qwen3-8B;
- TP1;
- eager decode;
- a pinned 640-block KV pool;
- context length 8192;
- batches 4, 6, 8, 10, 12, 16, and 19;
- identical seed, warmup count, measured-step count, and prompt construction;
- a fresh immutable tag per arm;
- identical resolved engine identity except for the requested Quest fields.

The added batches 6 and 10 bracket the activation boundary and prevent the
policy from being judged only on the five points used to derive it.

The adaptive arm must report:

- only `below_saved_blocks` with resolved top-k `-1` at batches 4 and 6;
- only `active` with resolved top-k `16` at batches 8, 10, 12, 16, and 19.

## Frozen quality experiment

Use the same 25 source-bound fixed needle prompts as the completed Quest gate:

- context length 8192;
- depths 0.0, 0.25, 0.5, 0.75, and 1.0;
- five scored trials per depth;
- greedy decoding;
- KV8 full and adaptive Quest arms built from the same source commit.

Run scored prompts in decode batches of eight so the adaptive arm reaches the
128-block threshold. For the final partial batch, add deterministic,
source-bound filler prompts until the physical decode batch is eight; filler
outputs are excluded from accuracy. Both arms receive the same batch grouping
and fillers.

Every scored adaptive decode step must resolve to top-k 16 with reason
`active`. A fallback step, missing observation, prompt/hash mismatch, or
different batch grouping makes quality evidence inconclusive.

The primary comparison remains adaptive Quest versus KV8 full on identical
scored prompts. No quality claim is inferred from the performance workload.

## Classification gate

The candidate is `GO_KV8_QUEST_AMORTIZATION_POLICY` only if all of the
following hold:

1. every performance and quality artifact is source-bound and identity
   matched;
2. every performance cell reaches its requested batch and contains complete,
   consistent activation telemetry;
3. adaptive batches 4 and 6 fall back on every measured step;
4. adaptive batches 4 and 6 regress no more than 3% versus KV8 full median
   step time;
5. adaptive batches 8, 10, 12, 16, and 19 activate on every measured step;
6. no active adaptive cell regresses more than 3% versus fixed Quest median
   step time;
7. at batch 19, adaptive Quest still removes at least 50% of KV8's excess
   latency over the matching bf16 eager baseline from the prior source-bound
   gate, or a newly rerun identity-equivalent bf16 arm;
8. adaptive Quest needle accuracy is no more than five percentage points
   below KV8 full overall;
9. no individual depth loses more than twenty percentage points;
10. all scored adaptive quality steps are telemetry-confirmed active.

An identity, telemetry, dispatch, completeness, or source-binding failure
yields `INCONCLUSIVE_KV8_QUEST_AMORTIZATION_POLICY`. A measured performance or
quality threshold failure yields `NO_GO_KV8_QUEST_AMORTIZATION_POLICY`.

## Implementation and harness scope

Runtime changes are limited to:

- configuration validation and passthrough;
- host-side per-step activation resolution;
- read-only activation telemetry.

The step-scaling worker and remote runner add:

- `quest_min_saved_blocks` CLI/environment passthrough;
- requested and resolved identity fields;
- per-step activation evidence and measured-window summaries.

The quality runner adds the same field, fixed eight-way batch grouping,
deterministic fillers, and activation evidence. The gate analyzer independently
recomputes expected activation from batch/block metadata and does not trust a
producer's classification.

No attention kernel, selector math, KV layout, graph policy, scheduler policy,
or default configuration changes in this candidate.

## Evidence and storage

Remote runs write only below:

`/data00/home/sitian/tllm/kvcapacity-runs/`

Repository evidence is limited to compact JSON/Markdown summaries, source
manifests, the spec, implementation plan, tests, and final audit/handoff.
Large logs, model files, traces, and duplicate sweep payloads remain remote.

Each failed or partial run receives a unique tag and is never reused.

## Claim boundary

A pass would show that a deterministic, workload-aware host policy retains
the measured Quest benefit while avoiding its small-batch regression on the
frozen Qwen3-8B/A100/TP1/eager/8192 workload.

It would not establish production QPS or P99 gains, CUDA-graph compatibility,
TP2/TP4 behavior, other models or GPUs, arbitrary context-length optimality,
online self-tuning, or a universally optimal threshold. The feature remains
default-off.
