# Qwen3.5 Root Model Assembly Factory Design

## Objective

Construct one complete `Qwen35PackedForCausalLM` ownership graph from:

- a validated Qwen3.5 config;
- one existing `HybridStateTensorPool`;
- explicitly supplied embedding/final-norm/lm-head modules;
- one dependency-injected decoder-layer factory.

This gate validates production-shaped composition without yet coupling to
tensor-parallel process groups, CUDA attention backends, or checkpoint weight
names.

## Why Dependency Injection Is Required Here

The existing Qwen3.5 shells are mathematically validated, but concrete
projection and attention modules currently depend on runtime TP/CUDA setup.
Constructing them inside a CPU/static factory would either start forbidden
runtime infrastructure or add fake production modules.

The assembly factory therefore owns graph topology and state ownership only.
The next checkpoint/runtime gate can provide concrete component factories.

## Factory

Create:

```text
tinyvllm/models/qwen35_factory.py
```

Expose:

```python
@dataclass(frozen=True)
class Qwen35PackedModelAssembly:
    model: Qwen35PackedForCausalLM
    layer_stack: Qwen35PackedHeterogeneousLayerStack
    state_transaction: Qwen35CrossLayerStateTransaction
    adapters: tuple[Qwen35LayerStateAdapter, ...]
    pool: HybridStateTensorPool

def assemble_qwen35_packed_model(
    hf_config,
    *,
    pool: HybridStateTensorPool,
    embed_tokens: nn.Module,
    final_norm: nn.Module,
    lm_head: nn.Module,
    build_decoder_layer: Callable[
        [int, str, Qwen35LayerStateAdapter | None],
        Qwen35DecoderLayerShell,
    ],
) -> Qwen35PackedModelAssembly
```

## Topology

Normalize `text_config` and validate:

```text
num_hidden_layers
layer_types
```

using the same supported values as `build_qwen35_hybrid_state_layout`.

For each `linear_attention` layer:

1. require both pool components for that layer;
2. build exactly one `Qwen35LayerStateAdapter`;
3. pass that exact adapter to `build_decoder_layer`.

For each `full_attention` layer, pass `None`.

The callback must return an exact `Qwen35DecoderLayerShell` whose
`block_type` matches the requested type. The callback cannot substitute,
reorder, or omit layers.

After all layers:

```text
transaction = Qwen35CrossLayerStateTransaction(adapters)
stack = Qwen35PackedHeterogeneousLayerStack(layers, transaction)
model = Qwen35PackedForCausalLM(embed_tokens, stack, final_norm, lm_head)
```

The returned assembly retains every exact identity.

## Pool Coherence

The factory never constructs `HybridStateTensorPool`.

It verifies:

- exact pool type;
- pool layout contains exactly the linear-layer indices in config;
- each linear layer has one convolution and one recurrent component;
- no full-attention layer has state components;
- every adapter references the supplied pool;
- root/stack/transaction/pool identities remain coherent.

Shape correctness remains owned by the existing config-to-layout builder and
pool constructor. This factory verifies topology, not checkpoint dimensions.

## Failure Boundary

Assembly is local and unpublished. Any callback or validation failure returns
no model and does not mutate pool tensor values or active leases.

No ModelRunner, Engine, Scheduler, checkpoint, or GPU code is called.

## Correctness Tests

Create:

```text
tools/test_qwen35_root_model_assembly_factory.py
```

Tests cover:

1. exact config layer order;
2. exact adapter identity only for linear layers;
3. exact root/stack/transaction/pool graph;
4. no pool storage allocation or mutation;
5. full-attention callback receives `None`;
6. wrong callback return type/block type fails closed;
7. pool/config layer-index mismatch fails closed;
8. missing/extra/full-layer state components fail closed;
9. callback failure leaves pool unchanged;
10. assembled model works with owner factory and transactional root execution.

## Acceptance Gate

Complete only when focused RED/GREEN passes under Python 3.9 and 3.12,
transactional root/owner/packed/hybrid regressions pass, the 97/1/0 matrix and
dual-version compile pass, and production selection remains unchanged.

Allowed conclusion:

> TinyLLMForge can assemble a complete Qwen3.5 root model graph from validated
> config topology and one existing state pool without duplicating state
> storage.

Not established:

- concrete TP/CUDA component construction;
- checkpoint loading;
- production ModelRunner selection;
- startup binding, Scheduler admission, GPU correctness, or performance.

