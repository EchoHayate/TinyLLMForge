# Verifier-Aware Quantized Draft Runtime Design

**Date:** 2026-08-31

**Status:** Approved direction; implementation plan pending

**Scope:** Model-agnostic learned speculative decoding on NVIDIA A100

## 1. Objective

Determine whether TinyLLMForge can turn low-bit distillation into a real
end-to-end decode-speed improvement by optimizing a learned draft model for:

```text
accepted target-equivalent tokens
---------------------------------
        draft GPU time
```

rather than optimizing only standalone student-model perplexity.

The target model remains unchanged and remains the sole authority for emitted
tokens. The candidate drafter uses a genuinely accelerated A100-compatible
low-bit execution path and proposes multiple tokens per target verification.
The runtime chooses whether to speculate and how many tokens to propose using
frozen profitability evidence.

The work is not tied to Qwen3.8-27B. Stage 1 uses the smallest already
available target/draft pair that exercises the production speculative
interfaces. A larger or second model is attempted only after the small-model
gate proves a real opportunity.

## 2. Motivation

Recent model-compression work combines quantization-aware training or
distillation with weight reconstruction to recover much of the quality lost
by aggressive low-bit quantization. QUASAR applies loss-aware reconstruction
during low-bit training. Quantization-Aware Healing combines structural
compression, quantization, and direct teacher supervision. SlimQwen combines
progressive structural compression with language-model and multi-token
distillation.

Those results motivate this work, but TinyLLMForge should not directly copy
their full-model objective:

- a smaller checkpoint does not prove faster inference;
- an A100 does not provide Blackwell-native NVFP4 execution;
- compressing the target model introduces a model-quality claim that requires
  a broad evaluation suite;
- full-model training is expensive and weakly coupled to the existing
  inference-runtime contribution; and
- the repository already has an exact speculative verifier and a learned
  draft execution path whose main unresolved question is profitability.

The existing learned-drafter CUDA Graph gate established exact TP4/B4/Q4
execution and an acceptance rate of `0.7285714285714285`. Its median TPOT
improved from `94.181602 ms` to `72.495060 ms`, but the paired throughput
bootstrap interval crossed zero. It therefore ended as
`NO_GO_PERFORMANCE`.

That evidence shows that proposal quality is not zero and that proposal
execution can be accelerated, but it does not establish stable request-level
profitability. Reducing drafter cost while preserving accepted-prefix yield is
the next falsifiable mechanism.

## 3. Non-goals

This design does not:

- claim that quantization-aware distillation, speculative decoding, dynamic
  proposal length, mixed precision, or target verification is individually
  novel;
- quantize or retrain the target model;
- claim that reduced checkpoint size implies reduced TPOT;
- emulate NVFP4 and call it native A100 acceleration;
- use per-forward full-weight dequantization as the candidate low-bit path;
- change greedy acceptance, emitted-token authority, EOS handling, or output
  budget semantics;
- add speculative-speculative branch prediction in the first implementation;
- tune thresholds on final benchmark rows;
- use target-generated oracle tokens as runtime proposals; or
- promote a microkernel result as an end-to-end speedup.

## 4. Considered approaches

### 4.1 Full-target quantization-aware distillation

Quantize the complete target model and use the original model as teacher.

Advantages:

- large checkpoint and weight-bandwidth reduction;
- directly comparable with recent low-bit model releases; and
- potentially useful for fitting larger models on fewer GPUs.

Costs:

- changes the served model and therefore requires broad quality evaluation;
- A100 cannot reproduce native NVFP4 execution;
- training and calibration cost scales with the full target;
- existing TinyLLMForge INT4 currently dequantizes weights before
  `F.linear`, so storage reduction does not imply compute acceleration; and
- it does not use the repository's strongest speculative-runtime assets.

Decision: not selected for the next stage.

### 4.2 Structural compression plus healing

Prune layers, hidden dimensions, attention heads, or MoE experts, then distill
the compressed model from the original teacher and optionally quantize it.

Advantages:

- truly reduces parameter count and arithmetic work;
- offers a strong model-compression research result if successful; and
- can be evaluated independently of a speculative runtime.

Costs:

- architecture surgery and checkpoint conversion are model-specific;
- substantial training is required before the first end-to-end result;
- final quality is approximate rather than verifier-exact; and
- the work is primarily a model-training project.

Decision: retain as a later independent project, not the current runtime
optimization.

### 4.3 Verifier-aware quantized drafter

Keep the target unchanged. Train and execute a low-bit drafter whose objective
and precision allocation maximize accepted tokens per unit of draft GPU time.
Use the existing target verifier and transactional KV runtime to preserve exact
greedy output.

Advantages:

- target output quality is preserved by construction in greedy mode;
- training applies only to the smaller drafter;
- it directly attacks the measured `proposal_forward` cost;
- the existing source-neutral proposal, verification, and KV transaction
  boundaries are reusable;
- it works across target-model families; and
- it supports a clean comparison against ordinary decode and the BF16
  drafter.

Costs:

- a fused A100 low-bit kernel is a prerequisite;
- quantization may reduce acceptance enough to erase its compute benefit;
- training data and teacher-output provenance must be frozen; and
- request-level variance remains a separate risk after proposal acceleration.

Decision: selected.

## 5. Existing runtime boundaries to preserve

### 5.1 Proposal interface

`tinyvllm/speculative/adapter.py` already defines:

```python
DraftCapabilities
DraftContext
DraftProposal
DraftAdapter
```

The new drafter remains a `model_runner` execution-domain proposal source. It
must not introduce model-specific task names or data into the generic
speculative runtime.

The proposal metadata may add only source-neutral telemetry:

```text
precision_profile_id
requested_k
produced_k
draft_cuda_ns
draft_wall_ns
confidence_summary
profitability_decision
```

### 5.2 Verification and token authority

The target verifier remains the only authority for:

- accepted-prefix length;
- the first replacement target token after rejection;
- EOS truncation;
- output-budget truncation; and
- final emitted tokens.

The low-bit drafter cannot directly publish user-visible output.

### 5.3 KV transaction boundary

The existing transaction must continue to:

- reserve proposal KV before materialization;
- commit only the accepted prefix;
- release the rejected suffix;
- leave zero active transactions after completion or failure;
- preserve proposal lifecycle commit and rollback; and
- fail closed on any mismatched generation, ownership, or sequence identity.

No accepted KV may be recomputed merely because the proposal came from a
quantized drafter.

## 6. Candidate architecture

### 6.1 Teacher data producer

The teacher-data producer runs the unchanged target model over a frozen prompt
corpus and records only the tensors required by the chosen distillation loss:

```text
prompt identity and split
token positions
target top-k logits or compressed logit targets
selected hidden-state observations
greedy continuation tokens
source-model and tokenizer fingerprints
```

The producer writes large datasets only under the approved remote root:

```text
/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/
```

The local repository receives only compact manifests, summaries, verifier
receipts, and audit documents.

The corpus is divided before any precision or loss tuning:

- `train`: parameter updates;
- `calibration`: precision-profile and threshold selection;
- `holdout`: terminal quality and acceptance classification.

No prompt may cross split boundaries after normalization.

### 6.2 Distillation objective

The initial training objective is:

```text
L_total =
    lambda_ce * L_token_ce
  + lambda_kl * L_teacher_kl
  + lambda_hidden * L_hidden_alignment
  + lambda_accept * L_expected_accepted_prefix
```

`L_expected_accepted_prefix` assigns increasing value to correctly predicting
long contiguous prefixes. Matching tokens after the first mismatch receives
no speculative credit for that proposal.

The initial version does not backpropagate measured wall time. Runtime cost is
introduced through a frozen per-profile cost table:

```text
utility(profile, K) =
    expected_accepted_tokens(profile, K)
    / measured_draft_cuda_us(profile, K)
```

This avoids using noisy request timing as a training signal while still
aligning model selection with runtime profitability.

### 6.3 Precision profiles

Stage 1 supports a small frozen profile set:

```text
bf16_reference
int8_all
int4_all
int4_with_int8_sensitive_layers
```

Sensitivity is measured only on the calibration split. A layer is sensitive
when moving it from INT8 to INT4 causes the largest reduction in accepted
prefix per unit of memory or latency saved.

The first mixed profile may retain at most 20 percent of drafter linear-layer
weight bytes in INT8. Otherwise the profile is not meaningfully low-bit.

No per-request precision switching is implemented initially. Each benchmark
worker loads one immutable profile. Online precision routing is eligible only
after one fixed low-bit profile passes the end-to-end gate.

### 6.4 A100 fused low-bit execution

The candidate path must consume packed low-bit weights directly inside the
matrix multiplication. It may use a vetted CUTLASS-based dependency or a
repository-owned CUDA extension, but it must satisfy all of the following:

- no full BF16/FP16 weight reconstruction allocation;
- no full-weight dequantization kernel before each GEMM;
- scales are consumed in the GEMM mainloop or epilogue;
- supported shapes and alignment are checked before dispatch;
- unsupported shapes deterministically fall back to the frozen reference
  path;
- the fallback reason is recorded;
- graph-capture compatibility is measured rather than assumed; and
- kernel selection is independent of model family names.

The current `tinyvllm/layers/linear.py` INT4 path remains unchanged until the
isolated kernel gate passes.

### 6.5 Profitability router

The first runtime policy chooses only:

```text
baseline
draft K=2
draft K=4
draft K=8
```

Inputs are bounded, source-neutral features:

```text
batch size
context bucket
remaining output budget
recent accepted-prefix EMA
recent draft CUDA time EMA
recent verifier CUDA time EMA
active precision profile
```

The decision uses a frozen table or bounded arithmetic, not an online learned
model:

```text
expected_saved_target_us
    = expected_accepted_tokens * baseline_decode_us

expected_cost_us
    = draft_us + verifier_incremental_us + transaction_overhead_us

speculate only when:
    expected_saved_target_us
      >= expected_cost_us * safety_margin
```

The safety margin is calibrated before the terminal holdout run and is never
retuned from terminal results.

### 6.6 Optional branch-ahead extension

Speculative-speculative branch-ahead is intentionally Stage 3. It may execute
only after the fixed low-bit drafter produces a valid Stage-2 GO.

The future extension may predict the likely verifier outcome and prepare one
bounded next proposal while verification is in flight. It must use separate
generation identities and may never mutate authoritative sequence or KV state
before the current verifier transaction commits.

## 7. Qualification sequence

### Stage 0: low-bit kernel microgate

Use real drafter GEMM shapes taken from the existing model. Compare:

```text
BF16 torch/CUDA reference
current dequantize-then-GEMM INT4 path
candidate fused INT8 path
candidate fused INT4 path
```

Required metrics:

- CUDA-event median and P99;
- host submission median and P99;
- effective weight bandwidth;
- peak allocated and reserved GPU bytes;
- maximum absolute and relative output error;
- warmup stability;
- graph capture and replay status; and
- fallback count and reason.

Stage-0 GO requires:

- fused INT4 median CUDA time at most `0.75 * BF16`;
- fused INT4 P99 CUDA time at most `0.95 * BF16`;
- no full dequantized-weight allocation;
- persistent weight storage at most `0.40 * BF16`;
- finite outputs and frozen numerical tolerances;
- zero unexpected fallback; and
- producer plus independent verifier agreement.

Failure stops the INT4 route before distillation training.

### Stage 1: offline drafter quality gate

Compare the BF16 drafter with each frozen low-bit profile on the holdout split.

Required metrics:

- accepted tokens and proposed tokens;
- mean accepted prefix;
- full, partial, one-token, and zero-accept counts;
- target-token top-1 agreement by proposal position;
- draft CUDA time by K and batch;
- accepted tokens per draft millisecond;
- checkpoint bytes and loaded GPU bytes; and
- training tokens and GPU hours.

Stage-1 GO requires:

- low-bit acceptance at least `0.95 * BF16 acceptance`;
- low-bit mean accepted prefix at least
  `0.95 * BF16 mean accepted prefix`;
- accepted tokens per draft millisecond at least `1.25 * BF16`;
- loaded drafter weight bytes at most `0.50 * BF16`;
- every required acceptance class represented; and
- no holdout-driven retuning.

### Stage 2: exact end-to-end runtime gate

Compare three isolated policies:

```text
ordinary target decode
BF16 learned drafter
selected low-bit distilled drafter
```

Use fresh processes, position-balanced ordering, identical prompts, identical
output budgets, and at least eight measured pairs after warmup.

Correctness requirements:

- exact emitted greedy tokens;
- exact target-token rows;
- exact accepted-prefix recomputation;
- exact proposal lifecycle receipts;
- exact KV transaction digests;
- zero active transactions at worker exit;
- all TP ranks agree;
- no timeout, NaN, leaked allocation, or unexpected fallback; and
- source, model, tokenizer, dataset, and checkpoint identities are frozen.

Performance requirements relative to ordinary target decode:

- median TPOT improvement at least `10%`;
- paired throughput-delta bootstrap 95% CI lower bound greater than zero;
- TTFT regression no greater than `3%`;
- P99 TPOT regression no greater than `3%`;
- request throughput does not regress; and
- peak total GPU memory, including target and drafter, is reported.

Comparisons against the BF16 drafter must additionally report:

- draft-forward speedup;
- acceptance loss;
- accepted-tokens-per-draft-millisecond improvement; and
- drafter memory reduction.

A Stage-2 GO is model-, hardware-, workload-, batch-, context-, and precision-
profile-specific. It is not a universal performance claim.

### Stage 3: branch-ahead qualification

Only after Stage 2 passes:

- add one bounded predicted next branch;
- compare against the already-promoted low-bit drafter, not ordinary decode
  alone;
- charge all wasted branch work;
- report prediction accuracy and wasted compute;
- preserve exact output and transaction authority; and
- require an additional statistically supported E2E gain.

## 8. Evidence and artifact contract

Every terminal stage produces:

```text
source_identity.json
environment.json
preflight.json
frozen_config.json
raw_rows.jsonl
summary.json
classification.json
independent_verification.json
cleanup.json
manifest.sha256
```

Large traces, training tensors, and checkpoints remain remote. The compact
local bundle contains:

- immutable identities and configuration;
- aggregate and per-pair metrics;
- sampled diagnostic rows sufficient to audit formulas;
- independent-verifier output;
- cleanup and exact-tag scans; and
- a checksum manifest covering the complete compact inventory.

The independent verifier must recompute all derived metrics from raw rows and
must reject:

- missing or extra files;
- symlinks;
- duplicate row identities;
- non-finite values;
- incomplete order balance;
- insufficient pairs;
- identity drift;
- threshold drift;
- missing acceptance classes;
- result mutation after manifest creation; and
- a classification inconsistent with the frozen rules.

## 9. Failure and cleanup policy

All candidate features are default-disabled.

Any rank-local error must converge to a common failure result before process
group destruction. Failure handling must:

- stop new proposal admission;
- roll back active proposal and side-state transactions;
- synchronize owned streams when safe;
- release drafter KV and graph entries;
- destroy the process group in the established order;
- record unresolved external processes without terminating them;
- perform three exact-tag remote scans; and
- leave unrelated GPU processes and files untouched.

The controller must not run `kinit` or `krenew`. Kerberos TTL is checked before
launch and fails fast when insufficient.

## 10. Claim boundary

Before Stage 0:

```text
DESIGN_ONLY
```

After a Stage-0 GO:

```text
FUSED_LOW_BIT_DRAFT_KERNEL_OPPORTUNITY
```

After a Stage-1 GO:

```text
LOW_BIT_DRAFTER_QUALITY_AND_COST_OPPORTUNITY
```

Only after a complete Stage-2 GO:

```text
GO_VERIFIER_AWARE_QUANTIZED_DRAFT_RUNTIME
```

No intermediate state may be described as:

- a full-model compression win;
- a Qwen3.8-27B result;
- a production-ready optimization;
- a universal model-quality result; or
- an end-to-end speedup.

## 11. Originality statement

The design borrows established components:

- quantization-aware distillation and reconstruction;
- mixed-precision quantization;
- learned speculative drafting;
- dynamic proposal length;
- exact target verification; and
- optional speculative-speculative branch prediction.

The proposed TinyLLMForge contribution is the unified optimization target and
runtime protocol:

1. train for contiguous target-accepted prefix rather than standalone
   perplexity;
2. choose layer precision by acceptance lost per unit of measured draft cost;
3. express both training and runtime routing through accepted tokens per draft
   microsecond;
4. preserve target-authoritative output and transactional accepted-KV commit;
5. fail closed when the measured profitability envelope is absent; and
6. qualify the complete chain with a source-bound, independently recomputed
   end-to-end gate.

This is an original end-to-end combination and objective proposal. It has no
performance result until the required gates execute.

## 12. Prompt-to-artifact checklist

| Requirement | Planned evidence | Terminal status now |
| --- | --- | --- |
| Do not bind the method to Qwen3.8 | source-neutral interfaces and Stage-1 model selection rule | `DESIGNED` |
| Use recent distillation/compression ideas | references and motivation section | `DESIGNED` |
| Preserve exact target output | target-authoritative verifier and transaction invariants | `DESIGNED` |
| Produce real A100 speed rather than checkpoint-only compression | Stage-0 fused-kernel gate | `UNVERIFIED` |
| Report benefit and cost | Stage-1/2 latency, acceptance, memory, and training-cost metrics | `DESIGNED` |
| Support horizontal comparison | ordinary/BF16-draft/low-bit-draft three-policy matrix | `DESIGNED` |
| Preserve remote storage boundary | approved `/data00/home/sitian/...` root only | `DESIGNED` |
| Keep large artifacts remote | compact local artifact contract | `DESIGNED` |
| Provide independent verification | raw-row recomputation and manifest checks | `DESIGNED` |
| Establish an original end-to-end contribution | acceptance-per-draft-time objective plus unified precision/K routing | `DESIGN_ONLY` |

## 13. References

- QUASAR: Quantization-Aware Training with Loss-Aware Reconstruction,
  arXiv:2608.13966.
- Quantization-Aware Healing for Large Language Models,
  arXiv:2608.20953.
- SlimQwen: Progressive Structural Compression of Large Language Models,
  arXiv:2605.08738.
- Saguaro: Speculative Speculative Decoding for Efficient LLM Inference,
  arXiv:2603.03251.
- ML-SpecQD: Multi-Level Speculative Decoding with Quantized Draft Models,
  arXiv:2605.04062.
