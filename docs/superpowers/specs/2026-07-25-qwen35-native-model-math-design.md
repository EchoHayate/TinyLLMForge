# Qwen3.5 Native Model Math Design

## Status

Approved for inline execution under the standing instruction to continue
without per-step confirmation.

This design follows the completed layout/runtime bridge. It defines the native
model-math boundary but deliberately implements the first slice as CPU
reference math only. It does not start a GPU process, install a production
Qwen3.5 ModelRunner, or change the immutable schema-v2 canonical `NO_GO`.

## Objective

Build a correctness-first path from the existing request-state pool to native
Qwen3.5 inference:

1. exact CPU reference causal convolution and gated-delta recurrent updates;
2. TP-correct checkpoint projection loaders;
3. Qwen3.5-specific full-attention, normalization, rotary, decoder, and model
   modules;
4. deliberate ModelRunner model selection and tensor-pool installation;
5. remote layer/state/logit equivalence before any optimized kernel or
   performance claim.

## Source Basis

The design was checked against:

- frozen model revision
  `15852e8c16360a2fea060d615a32b45270f8a8fc`;
- local canonical architecture/state evidence under
  `experiments/qwen35_hybrid_state/`;
- Hugging Face `modeling_qwen3_5.py` snapshot SHA-256
  `15d5425ee6e771f8fbca10468c280fe62afa79fab3eff73ad1a8852162799d48`;
- Hugging Face `configuration_qwen3_5.py` snapshot SHA-256
  `3c01b3cdcff8d77cbafac9841bc48c41e5a5b38637231f1bde3d843cd198dbaf`;
- vLLM Qwen GDN snapshot SHA-256
  `1227d6f385a52296e9f08223544b1c5fdc7e8d9aa09a848e7a8e522a8dc51214`;
- vLLM Qwen3-Next model snapshot SHA-256
  `b642c10eb68978ca0df25f92ea866add08b31e654d99e78eaac0195d8bc6c74b`.

The remote environment cannot currently be refreshed because its Kerberos
ticket is expired. The saved canonical evidence remains authoritative for the
frozen model revision; current upstream source is used only to define
implementation semantics, not to rewrite the completed canonical result.

## Critical Compatibility Findings

### 1. Recurrent Storage Orientation Is Not Mathematical Orientation

The gated-delta recurrence uses a matrix shaped:

```text
[value_head, key_dim, value_dim]
```

because the update is:

```text
state = decay * state
memory = key @ state
delta = beta * (value - memory)
state = state + outer(key, delta)
output = query @ state
```

The vLLM-compatible physical request cache is shaped:

```text
[value_head, value_dim, key_dim]
```

The native path must transpose the last two dimensions when loading and
storing recurrent state. The canonical model has `key_dim == value_dim ==
128`, so shape-only evidence cannot detect an orientation bug. CPU tests must
use asymmetric dimensions.

### 2. Convolution Cache Width

The official reference cache stores `kernel_size` historical values for the
single-token update helper, while vLLM serving kernels use
`kernel_size - 1 + num_speculative_tokens`. The existing layout adapter follows
the serving-kernel convention.

The CPU primitive therefore treats the physical convolution row as the exact
window consumed by the serving path:

```text
history_width = kernel_size - 1 + speculative_tokens
```

For non-speculative execution this is `kernel_size`. The primitive updates the
row with the latest `history_width` projected values and computes causal
depthwise convolution without allocating token-growing state.

### 3. Existing Qwen3 Modules Are Not Drop-In Compatible

Qwen3.5 full-attention differs in several material ways:

- `q_proj` produces both query and a per-query output gate;
- q/k RMSNorm parameters use offset semantics `scale = 1 + weight`;
- only a configured fraction of each head receives rotary embedding;
- text rotary uses interleaved MRoPE configuration;
- the attention result is multiplied by `sigmoid(query_gate)` before
  `o_proj`.

The existing `QWen3Attention`, `RMSNorm`, and full-head rotary implementation
cannot be reused unchanged.

### 4. Linear Projection TP Sharding Must Be Segment-Aware

Qwen3.5 checkpoints store:

- one fused `in_proj_qkv` tensor with logical segments `[Q, K, V]`;
- separate `in_proj_z`, `in_proj_b`, and `in_proj_a`;
- depthwise `conv1d.weight`;
- sharded `A_log` and `dt_bias`;
- `norm.weight` and `out_proj.weight`.

A contiguous shard of the full `in_proj_qkv` output dimension is incorrect at
TP > 1 because it crosses logical Q/K/V segment boundaries. The loader must
split source segments, shard each independently, then concatenate local
segments. The same rule applies when fusing Q/K/V/Z or B/A for fewer GEMMs.

### 5. Cached Decode Cannot Be Validated Only by Final Tokens

The frozen canonical gate already found interleaved cached-path drift. Upstream
implementations also distinguish one-shot/chunk, cached single-token, and
packed recurrent kernels. Native correctness must compare:

- every layer output;
- convolution state after each chunk/token;
- recurrent state after each chunk/token;
- final hidden state and logits;
- one-shot, chunked continuation, cached decode, interleaved requests, and
  slot reuse.

A matching final token alone is insufficient.

## Architecture

### Phase A: CPU Reference GDN Primitive

Create `tinyvllm/layers/gated_delta.py` with pure PyTorch functions:

```python
def qwen35_l2norm(
    tensor: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    ...

def qwen35_causal_depthwise_conv(
    projected_qkv: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    *,
    activation: str = "silu",
) -> tuple[torch.Tensor, torch.Tensor]:
    ...

def qwen35_gated_delta_recurrent(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    recurrent_state_v_k: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    ...
```

Input row semantics:

- token-major tensors use `[tokens, local_heads, head_dim]`;
- `a`, `b` use `[tokens, local_value_heads]`;
- physical recurrent state uses
  `[local_value_heads, value_dim, key_dim]`;
- returned physical state preserves that orientation.

The recurrence:

```text
q = l2norm(q) / sqrt(key_dim)
k = l2norm(k)
beta = sigmoid(b)
log_decay = -exp(A_log) * softplus(a + dt_bias)
decay = exp(log_decay)

state_k_v = transpose(state_v_k)
for each token:
    state_k_v *= decay
    memory = key @ state_k_v
    delta = (value - memory) * beta
    state_k_v += outer(key, delta)
    output = query @ state_k_v
return output, transpose(state_k_v)
```

All recurrence accumulation is FP32. Outputs return to the query dtype; state
returns to the physical state dtype. The function is deterministic and
side-effect-free so tests can compare one-shot and continuation exactly within
declared tolerances.

### Phase B: Segmented TP Projection Layer

Add a focused segmented column-parallel linear:

```python
SegmentedColumnParallelLinear(
    input_size,
    output_sizes,
    bias=False,
)
```

Its checkpoint loader accepts a fused source tensor, independently shards each
logical output segment, and concatenates local shards. It must support:

- `[Q, K, V]`;
- `[Q, K, V, Z]`;
- `[B, A]`;

TP tests use unequal segment sizes and synthetic rank values to prove no
cross-segment slicing.

Qwen3.5 full-attention `q_proj` is an explicit exception. The official
checkpoint layout is head-major:

```text
[query_head_0, gate_head_0,
 query_head_1, gate_head_1,
 ...]
```

because the projection is reshaped to
`[..., num_heads, 2 * head_dim]` before query/gate are chunked along the last
dimension. It must therefore use a normal contiguous column-parallel shard
whose rank boundary contains complete `2 * head_dim` head pairs. Treating it
as a global `[Q_all, query_gate_all]` segmented tensor would assemble the
wrong rank-local rows.

### Phase C: Qwen3.5 Modules

Add `tinyvllm/models/qwen35.py` with:

- `Qwen35OffsetRMSNorm`;
- partial/interleaved text rotary embedding;
- gated full attention using the existing paged `Attention` backend;
- GDN linear attention consuming slot-indexed pool tensors;
- Qwen3.5 MLP and decoder residual structure;
- `Qwen35ForCausalLM` weight-name mapping.

The GDN module receives step-local slot ids from ModelRunner/context and
gathers physical convolution/recurrent rows. It writes updated rows back only
after successful layer math.

Mixed prefill/decode batches must preserve row boundaries. A flattened token
tensor without per-request offsets is insufficient for recurrent updates.

### Phase D: ModelRunner Selection and Pool Installation

Model selection must be explicit:

```text
Qwen3Config-compatible architecture -> Qwen3ForCausalLM
Qwen3_5TextConfig-compatible architecture -> Qwen35ForCausalLM
otherwise -> fail closed
```

For Qwen3.5 only:

1. build the validated TP-local layout;
2. create `HybridStateTensorPool`;
3. create `HybridStateRuntimeBridge`;
4. construct `Scheduler` with matching slot capacity;
5. publish slot ids and per-request token boundaries in `Context`;
6. exclude hybrid mode from unsupported CUDA Graph paths until independently
   validated.

Rank 0 and all TP workers must use the same layout fingerprint and capacity.

## Failure Semantics

Fail closed on:

- non-Qwen3.5 architecture routed to Qwen3.5 model;
- unsupported MRoPE configuration;
- missing or unexpected checkpoint weights;
- segmented TP divisibility failure;
- pool/layout fingerprint mismatch;
- missing slot ids or request boundaries;
- recurrent orientation mismatch;
- duplicate or stale lease;
- mixed-batch metadata that cannot preserve request isolation;
- hybrid CUDA Graph attempt before a dedicated gate.

No fallback may silently use Qwen3 math, full recomputation, an unsharded
projection, or token-growing fake recurrent state.

## Testing and Gates

### CPU Primitive Gate

- asymmetric `key_dim != value_dim` orientation test;
- manual scalar recurrence oracle;
- one-shot equals split continuation;
- zero state and non-zero initial state;
- two request states remain isolated;
- q/k L2 normalization and query scaling;
- BF16 inputs with FP32 recurrence accumulation;
- causal convolution one-shot equals chunk continuation;
- convolution state contains the exact latest physical window;
- invalid shapes and dtype contracts fail.

### Loader Gate

- TP=1 identity;
- TP=2/4 unequal-segment synthetic tensors;
- fused and separate checkpoint names;
- `A_log`, `dt_bias`, conv, norm, and output projection sharding;
- complete expected-weight inventory and unexpected-weight rejection.

### Module Gate

- offset RMSNorm against official formula;
- partial/interleaved rotary fixture;
- gated full-attention projection shapes;
- GDN layer state mutation against CPU primitive;
- decoder layer output against a dependency-light official fixture.

### Remote Correctness Gate

Only after GPU0 admission:

1. one linear layer with copied official weights;
2. one full-attention layer;
3. full 24-layer model;
4. one-shot versus cached decode;
5. one-shot versus chunked prefill;
6. interleaved multi-request;
7. completion release and slot reuse;
8. BF16 decision and FP32 elementwise evidence.

The canonical schema-v2 `NO_GO` remains immutable. New native evidence receives
a separate run tag and verifier.

## First Implementation Slice

This session implements only Phase A, the CPU reference GDN primitive and its
tests. This is the smallest independent deliverable that proves the state
orientation, numerical equations, and continuation semantics before model,
loader, or kernel complexity is introduced.

No performance claim follows from Phase A.
