# Qwen3 Frozen-CED Retrofit Phase-1 Probe Design

**Date:** 2026-09-21

**Status:** Approved design

**Target branch:** `feat/kv-sparse-attention`

**Source anchor:** `6b88d081c659e0dfc8d9758589f109708347056b`

**Stage-1 model:** Qwen3-0.6B

**Topology:** TP1 on one A100-class GPU

**Default:** disabled

**Runtime integration:** prohibited unless the offline probe returns
`PROBE_GO_FROZEN_CED_RETROFIT`

## 1. Decision

The next architecture-level experiment will test whether a frozen,
decoder-only Qwen3 checkpoint can approximate a small Causal
Encoder-Decoder-style prefill boundary without retraining the backbone.

For prompt prefill only, a lightweight trainable synthesizer consumes the
hidden-state sequence at a frozen boundary layer and predicts the prompt KV
entries that the final decoder layers would have produced. If those predicted
entries are sufficiently accurate and cheaper to produce, the prompt can skip
the corresponding upper decoder layers. Decode tokens still execute the
complete original model and append exact layer-local KV entries.

The first probe targets the final four layers:

```text
teacher:
  prompt
    -> frozen layers [0, ..., L-5]
    -> frozen layers [L-4, ..., L-1]
    -> exact upper prompt KV
    -> exact logits

student probe:
  prompt
    -> frozen layers [0, ..., L-5]
    -> trainable synthesizer
    -> predicted prompt KV for layers [L-4, ..., L-1]
    -> predicted final-position normalized state
```

This is an internal-state distillation experiment. It is not instruction
fine-tuning, not a claim that Qwen3 has become a native CED model, and not a
KV-capacity optimization.

## 2. Provenance and claim separation

DeepSeek-V4.1-Flash's native Causal Encoder-Decoder design trains the model so
that decoder global KV is projected from the final causal-encoder hidden
states. Its reported cache reduction also depends on separate mechanisms,
including Compressed Sparse Attention 2, low-precision KV, and bounded replay.

This design borrows only the high-level CED observation:

> prompt-side upper-layer KV might be generated from a lower boundary
> representation more cheaply than executing every original upper block.

TinyLLMForge's candidate differs materially:

- the source checkpoint is an existing decoder-only Qwen3 model;
- all original model weights remain frozen;
- a new auxiliary synthesizer approximates the omitted prompt computation;
- every skipped layer initially retains its own predicted KV;
- decode continues through every original layer;
- no DeepSeek performance or quality result is transferred to this candidate.

The term **Frozen-CED Retrofit** names this experiment only. It does not imply
architectural or numerical equivalence to native CED.

## 3. Objective and non-objectives

### 3.1 Objective

Determine, with a bounded single-GPU experiment, whether synthesized
upper-layer prompt KV can simultaneously provide:

1. usable internal-state fidelity;
2. stable continuation quality;
3. a credible prompt-prefill compute saving; and
4. a path to lower end-to-end TTFT after runtime integration.

### 3.2 Non-objectives

Phase 1 will not:

- modify or fine-tune the Qwen3 backbone;
- train on instruction-response supervision;
- claim exact token parity;
- claim reduced decode TPOT;
- share one KV representation across upper layers;
- implement CSA2, FP4 KV, or bounded replay;
- claim that KV capacity is halved;
- integrate the candidate into the production scheduler or default path;
- run Qwen3-8B, Qwen3.8-27B, TP2, or TP4;
- establish serving QPS, mixed-batch P99, or production readiness.

## 4. Why an offline probe comes first

The final layer KV of a decoder-only Transformer is not generally a
token-local linear function of an earlier hidden state. The omitted blocks
contain attention, MLP, residual, and normalization operations. A synthesizer
may therefore fail for either of two independent reasons:

- **representation failure:** it cannot produce KV that the frozen upper
  decoder layers can use without unacceptable output drift;
- **economic failure:** it can produce usable KV, but costs as much as or more
  than the omitted blocks.

An offline probe can reject either failure before scheduler, paged-cache,
rollback, or serving integration expands the implementation surface.

## 5. Frozen experiment matrix

### 5.1 Model identity

The runner must discover and record the real Qwen3-0.6B checkpoint path and
immutable model identity on the remote host. It must bind:

- checkpoint path and resolved revision or file manifest;
- tokenizer identity;
- model configuration hash;
- number of hidden layers;
- hidden size;
- attention head and KV-head counts;
- head dimension;
- dtype;
- source commit and source-tree digest.

The design assumes the commonly used Qwen3-0.6B architecture, but the runner
must derive all layer and tensor dimensions from the loaded configuration.
An identity mismatch makes the attempt inconclusive.

### 5.2 Skip depths

The probe evaluates final-layer skip depths:

```text
S in {2, 4, 6}
```

For a model with `L` layers, the boundary is the exact pair
`(hidden_states, residual)` returned by layer `L-S-1`, and the targets are the
prompt K/V entries of layers `[L-S, ..., L-1]`. The pair is required because
TinyLLMForge's fused residual/RMSNorm path does not represent a decoder-layer
boundary with one tensor.

Skip depth four is the primary candidate. Depths two and six establish the
local quality-versus-compute curve and prevent a single cherry-picked result.

### 5.3 Synthesizer families

The probe evaluates three bounded model families.

#### A. Token-local low-rank projector

Each prompt token's boundary hidden state is mapped independently to a shared
latent and then to layer-specific K/V heads:

```text
h_boundary[t]
  -> down projection
  -> bounded nonlinearity
  -> up projection
  -> {K_hat[l,t], V_hat[l,t]} for each skipped layer l
```

This is the cheapest lower bound. It cannot reproduce new cross-token mixing
performed by skipped layers.

#### B. One-layer sequence synthesizer

A single causal Gated DeltaNet or linear-attention block processes the
boundary hidden sequence before layer-specific low-rank K/V heads:

```text
H_boundary
  -> one causal lightweight sequence block
  -> H_synthetic
  -> layer-specific low-rank K/V heads
```

This is the primary candidate because it can model cross-token effects while
remaining substantially cheaper than four complete Transformer blocks.

The implementation plan must choose one already supported TinyLLMForge
primitive after measuring its standalone shape and cost. The design does not
authorize adding two sequence primitives merely to broaden the search.

#### C. Two-layer lightweight Transformer ceiling

A two-layer causal Transformer synthesizer is evaluated only as a
representation ceiling. It may establish that the target is learnable, but
it cannot pass the economic gate if its measured cost consumes the omitted
block saving.

The three families must use frozen parameter-count and hidden-width budgets
defined before training. Failed candidates are not enlarged iteratively after
holdout results are visible.

## 6. Teacher trace contract

### 6.1 Captured tensors

For each source sequence, the frozen teacher records:

- input token IDs and attention-relevant position metadata;
- both tensors in the boundary `(hidden_states, residual)` pair;
- target K after Qwen3 K-Norm and RoPE, and target V after projection, for
  every skipped layer;
- the exact physical cache-layout form of those K/V targets, including slot
  mapping, head layout, dtype, and any quantization metadata;
- target attention outputs for every skipped layer;
- final normalized hidden states at scored positions;
- final teacher logits or a lossless scored-logit subset sufficient to
  recompute the frozen metrics;
- greedy continuation token IDs for the fixed evaluation horizon.

The trace records tensor shapes, dtypes, layer indices, sequence lengths,
checkpoint identity, tokenizer identity, source SHA, and content hashes.

### 6.2 Storage

Large traces, checkpoints, optimizer states, and training logs remain remote
under:

```text
/data00/home/sitian/TinyLLMForge/ced-retrofit/
```

Nothing large is written to the remote root filesystem or synchronized into
the local repository. Repository artifacts are limited to compact manifests,
metric summaries, verifier outputs, and reports.

### 6.3 Data partitions

Text examples are partitioned by content hash before traces are generated:

- `train`;
- `calibration`;
- `holdout`.

The holdout partition is not used for optimizer updates, architecture
selection, loss-weight tuning, threshold selection, or early stopping.
Duplicate and prefix-overlapping examples across partitions are rejected.

Ordinary text is sufficient. No instruction answer is required because the
teacher's internal states and logits supply supervision.

## 7. Training contract

### 7.1 Frozen ownership

All original Qwen3 parameters are:

- loaded from the bound checkpoint;
- placed in evaluation mode;
- marked non-trainable;
- excluded from the optimizer;
- hashed before and after training.

Only synthesizer parameters may change. Any backbone hash or tensor mutation
makes the run invalid.

### 7.2 Losses

The candidate uses a staged, normalized objective:

```text
L =
  lambda_kv    * L_kv
  + lambda_attn  * L_attn
  + lambda_logit * L_logit
```

Where:

- `L_kv` compares predicted and teacher K/V per skipped layer after a frozen
  per-head normalization that prevents high-magnitude layers from dominating;
- `L_attn` compares the skipped layer's attention output when its historical
  prompt cache is replaced by predicted KV. During this training-only
  diagnostic, the Query and current-layer input come from the frozen teacher;
  they are not available to the replacement runtime and cannot count toward
  runtime correctness or TTFT evidence;
- `L_logit` is KL divergence between teacher and replacement-path logits at
  frozen scored positions.

The implementation plan must freeze the precise normalization, positions,
loss weights, and numerical precision before the holdout is opened.

KV reconstruction alone cannot qualify a candidate. A low tensor error that
does not preserve attention outputs and downstream logits is a representation
failure.

### 7.3 Teacher forcing and rollout evaluation

Training may use teacher-forced token positions. Qualification must include
closed-loop greedy generation in which prior generated tokens affect later
states. Teacher-forced next-token agreement alone is insufficient because it
does not expose cumulative continuation drift.

## 8. Replacement semantics

For a prompt with `N` tokens and skip depth `S`:

1. execute frozen layers `[0, ..., L-S-1]` normally;
2. retain the exact lower-layer prompt KV produced by those layers;
3. run the synthesizer once over the exact boundary
   `(hidden_states, residual)` sequence;
4. write predicted prompt K/V into the ordinary cache slots belonging to
   layers `[L-S, ..., L-1]`; predicted K is already K-normalized and
   RoPE-applied, predicted V is already in the value-cache representation,
   and neither may be transformed a second time after injection;
5. do not execute those `S` original layers for the historical prompt tokens;
6. begin decode from the exact final prompt token state required by the
   selected replacement protocol;
7. execute all `L` original layers for every newly decoded token;
8. append exact per-layer K/V for decoded tokens.

Step 6 is a critical feasibility question. Predicting upper-layer historical
KV does not by itself produce the final prompt hidden state used to sample the
first token. The offline probe must therefore compare two explicit protocols:

- **KV-only diagnostic:** execute the exact teacher prompt to obtain the first
  sampled token, then use predicted upper prompt KV for subsequent decode.
  This isolates cache usability but provides no TTFT claim.
- **KV-plus-terminal-state candidate:** the synthesizer also predicts the
  final normalized prompt state, or an equivalent final-position state from
  which the unchanged LM head computes first-token logits. This is the only
  protocol eligible for a future TTFT claim.

The terminal-state head consumes the synthesizer's causal state at the final
prompt position. It must not read teacher upper-layer activations, teacher
logits, future tokens, or any value unavailable to the replacement runtime.

A candidate that passes only the KV-only diagnostic is
`NO_GO_FIRST_TOKEN_PATH`, not a runtime GO.

## 9. Offline measurements

### 9.1 Representation metrics

For each skipped layer and aggregate:

- normalized K cosine similarity;
- normalized V cosine similarity;
- K and V relative L2 error;
- attention-output cosine similarity and relative L2 error;
- finite-value and shape checks.

These are diagnostic metrics, not sufficient gate conditions.

### 9.2 Behavioral metrics

On frozen holdout prompts:

- first-token top-1 agreement;
- first-token top-5 containment;
- teacher-versus-candidate logit KL;
- teacher top-1 margin and candidate probability assigned to that token;
- closed-loop greedy token match by position;
- exact-prefix length before first divergence;
- full-continuation exact match;
- task-level answer or retrieval accuracy on the frozen quality bank.

Results are stratified by prompt length and skip depth. Aggregate averages may
not hide a long-context collapse.

The frozen primary behavioral gate is the KV-plus-terminal-state candidate.
The KV-only diagnostic is reported separately and cannot satisfy the
first-token or TTFT conditions.

### 9.3 Economic metrics

Measure with CUDA events and explicit synchronization outside the timed
region:

- exact time of the omitted upper-layer prompt computation;
- synthesizer latency;
- cache-write and layout-conversion latency;
- terminal-state generation latency;
- total replacement-path latency;
- peak allocated and reserved GPU memory;
- synthesizer parameter and checkpoint sizes;
- bytes written to the KV cache.

The economic comparison is:

```text
net_saved_time =
  omitted_upper_layer_prefill_time
  - (
      synthesizer_time
      + conversion_time
      + cache_write_time
      + terminal_state_time
    )
```

Parameter-count or FLOP estimates cannot substitute for measured latency.

## 10. Frozen workloads

The initial probe uses:

- Qwen3-0.6B;
- TP1;
- BF16 backbone;
- greedy decoding;
- batch size one;
- prompt lengths `512`, `2048`, `8192`, and `16384`, subject to the verified
  native quality range of the checkpoint;
- fixed continuation horizons `1`, `16`, and `64`;
- skip depths `2`, `4`, and `6`;
- a source-bound general-text holdout;
- a source-bound long-context retrieval/needle bank;
- at least three isolated timing repetitions per cell after warmup.

If the baseline checkpoint itself fails a task or context-length cell, that
cell cannot support a Retrofit quality claim. Candidate quality is always
compared against the matching baseline output, and absolute task quality is
reported separately.

## 11. Fail-fast stages

### Stage 0: Static feasibility

Before training:

- verify target tensor shapes and cache layout;
- prove the synthesizer output can be written to isolated cache storage;
- estimate parameter count and memory;
- measure untrained forward cost;
- verify no source-backbone parameter is trainable.

Reject a family whose best-case forward plus mandatory conversion already
exceeds 50% of the omitted-layer prefill time at 8192 tokens.

### Stage 1: Teacher-forced representation probe

Train each admitted family on the frozen train partition and choose
hyperparameters using calibration only.

Reject a skip depth if:

- KV-plus-terminal-state first-token top-1 agreement is below 95%;
- median logit KL or attention error is non-finite;
- quality worsens monotonically with prompt length without a bounded plateau;
- backbone identity changes.

### Stage 2: Closed-loop continuation probe

Run greedy continuation using predicted historical upper KV.

Reject a candidate if:

- exact-prefix length collapses immediately for most prompts;
- task-level accuracy drops by more than 0.5 percentage points overall;
- any frozen length stratum drops by more than 2 percentage points;
- errors grow without bound across the 64-token horizon.

### Stage 3: Economic qualification

Only the KV-plus-terminal-state protocol is eligible.

Require:

- replacement-path median latency below 50% of omitted-layer median latency
  at 8192 and 16384 tokens;
- projected full-prefill median improvement of at least 20% at 8192 tokens;
- projected full-prefill median improvement of at least 25% at 16384 tokens;
- no increase above 5% in peak GPU memory during the replacement operation;
- no additional per-token decode work beyond normal exact KV append.

Projected improvements are still offline estimates. They authorize Shadow
Runtime work but are not end-to-end TTFT evidence.

## 12. Classification

The terminal offline classifier is one of:

- `PROBE_GO_FROZEN_CED_RETROFIT`: one frozen candidate passes Stage 0–3,
  including terminal-state generation, holdout quality, and measured economic
  thresholds;
- `NO_GO_REPRESENTATION`: no candidate preserves the frozen behavioral gate;
- `NO_GO_FIRST_TOKEN_PATH`: predicted historical KV is usable, but no cheap
  terminal-state path preserves first-token behavior;
- `NO_GO_ECONOMICS`: quality passes, but measured replacement cost consumes
  the omitted-layer saving;
- `INCONCLUSIVE_FROZEN_CED_RETROFIT`: required identity, data partition,
  timing, completeness, or source-binding evidence is missing or inconsistent.

Only `PROBE_GO_FROZEN_CED_RETROFIT` authorizes a separate Shadow Runtime
design. It does not authorize production integration or default enablement.

## 13. Artifact and verifier contract

Each attempt uses a fresh immutable tag. The compact evidence bundle contains:

- experiment manifest;
- source and checkpoint identities;
- dataset partition hashes;
- synthesizer configuration and parameter count;
- training configuration and seed;
- per-stage metric summaries;
- per-length and per-skip-depth quality rows;
- measured latency and memory rows;
- backbone before/after hashes;
- terminal classification;
- provenance receipt.

Two independent verifiers are required:

1. a structural verifier checks schemas, hashes, identities, partitions,
   completeness, finite metrics, and backbone immutability;
2. a decision verifier recomputes every threshold and classification from raw
   compact rows without trusting the producer's declared result.

The repository stores only compact JSON and Markdown evidence. Raw tensors,
optimizer states, checkpoints, and verbose logs remain under the remote
`/data00/home/sitian/TinyLLMForge/ced-retrofit/` tree.

## 14. Runtime follow-up boundary

If the offline result is GO, a separate design may add a default-off Shadow
Runtime that:

- runs exact and synthesized prompt paths side by side;
- publishes per-request identity and quality telemetry;
- never serves synthesized output during shadow qualification;
- measures real TTFT, prefill throughput, cache writes, and decode TPOT;
- fails closed to the unchanged baseline.

Only a later source-bound end-to-end gate may classify
`GO_CED_RETROFIT_RUNTIME`.

CSA2-style shared upper latent KV is a separate Phase 2 research question.
It requires its own quality, cache-capacity, and decode-cost design and is not
authorized by this document.

## 15. Claim boundary

A Phase-1 GO would establish only that, for the bound Qwen3-0.6B checkpoint
and frozen single-GPU workloads, a small auxiliary module can approximate the
last few layers' prompt KV and final prompt state cheaply enough to justify
an end-to-end shadow implementation.

It would not establish:

- native CED equivalence;
- exact output preservation;
- KV-cache capacity reduction;
- Qwen3-8B or Qwen3.8-27B transfer;
- TP2/TP4 behavior;
- mixed-batch serving improvement;
- production QPS or P99 gains;
- a production-safe or default-enabled path.

Negative results remain first-class evidence. In particular, a representation
GO with an economic NO-GO must be reported as `NO_GO_ECONOMICS`, not as a
successful performance optimization.
