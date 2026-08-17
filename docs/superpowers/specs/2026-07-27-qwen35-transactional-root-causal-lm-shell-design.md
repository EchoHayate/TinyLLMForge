# Qwen3.5 Transactional Root Causal-LM Shell Design

## Objective

Add the smallest complete Qwen3.5 root execution shell above the existing
packed heterogeneous layer stack:

```text
embedding
packed full/linear decoder stack
final norm
lm head
```

The shell must preserve all-or-nothing recurrent-state semantics across the
complete logits path. It remains CPU/dependency-light and does not change
production model selection, checkpoint loading, Scheduler admission, or
Engine startup.

## Why the Layer Stack Cannot Be Selected Directly

`Qwen35PackedHeterogeneousLayerStack` is not a causal language model. It has no
token embedding, final norm, lm head, or `compute_logits` contract. Installing
it directly as `ModelRunner.model` would create a model that cannot satisfy the
current runtime call shape and would overstate native Qwen3.5 support.

The current stack also commits recurrent candidates before a caller can run
final norm or the lm head. A later failure would leave request state advanced
without a successful logits result.

## Staged Layer-Stack Execution

Extend `Qwen35PackedHeterogeneousLayerStack` with:

```python
def prepare(
    self,
    leases: tuple[HybridStateLease, ...],
    token_counts: tuple[int, ...],
    position_ids: torch.Tensor,
    hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor], ...]]

def commit(
    self,
    leases: tuple[HybridStateLease, ...],
    candidates: tuple[tuple[torch.Tensor, torch.Tensor], ...],
) -> None
```

`prepare()` performs the current gather and layer execution but does not mutate
persistent state. `commit()` delegates to the exact existing
`Qwen35CrossLayerStateTransaction`.

The existing `forward()` remains compatible:

```text
prepare -> commit -> return hidden states
```

## Root Shell

Create:

```text
tinyvllm/models/qwen35_packed.py
```

It exposes:

```python
class Qwen35PackedForCausalLM(nn.Module):
    def __init__(
        self,
        embed_tokens: nn.Module,
        layer_stack: Qwen35PackedHeterogeneousLayerStack,
        final_norm: nn.Module,
        lm_head: nn.Module,
    )

    def run_step(
        self,
        leases: tuple[HybridStateLease, ...],
        token_counts: tuple[int, ...],
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]
```

The root keeps exact strong references to all four components. The layer stack
is the unique state owner; the root allocates no state pool.

Execution order:

```text
validate public inputs
embed input_ids, unless input_embeds is supplied
layer_stack.prepare(...)
final_norm(hidden_states)
lm_head(hidden_states)
layer_stack.commit(leases, candidates)
return hidden_states, logits
```

Any failure before `commit()` leaves all persistent recurrent state unchanged.

## Public Validation

The shell fails closed on:

- non-module components;
- a non-exact packed heterogeneous layer stack;
- empty or malformed lease/token-count tuples;
- non-tensor or non-integral rank-1 `input_ids`;
- token total different from `input_ids.shape[0]`;
- `input_embeds` that are not floating rank-2 tensors with matching tokens;
- embedding, final-norm, or lm-head outputs that are not floating rank-2
  tensors with matching token count;
- final norm changing hidden width;
- lm head returning a non-positive vocabulary width;
- candidate preparation or commit failures.

`position_ids` shape/device validation remains owned by the packed stack's
existing input contract.

## Failure and Commit Boundary

Tests inject failures at:

```text
embedding
layer execution
final norm
lm head
commit
```

Embedding/layer/norm/head failures must leave the pool byte-for-byte unchanged.
A commit failure uses the existing cross-layer rollback behavior and therefore
also leaves the pool unchanged.

No unsafe hidden-only `forward()` or separate `compute_logits()` API is added
in this phase. Future ModelRunner integration must explicitly call
`run_step()` so logits success and recurrent-state commit remain one
transaction.

## Correctness Tests

Create:

```text
tools/test_qwen35_transactional_root_causal_lm.py
```

CPU tests cover:

1. exact embedding -> stack -> norm -> head call order;
2. independent manual output/logit equivalence;
3. input-embedding override skips token embedding;
4. state commits only after successful lm-head completion;
5. embedding, layer, norm, and head failures preserve state;
6. commit failure rolls back all layer state;
7. staged stack `prepare()` does not mutate state;
8. existing stack `forward()` retains prepare-then-commit behavior;
9. malformed component/input/output contracts fail closed;
10. no ModelRunner selection, Engine startup call, or Scheduler admission is
    added.

## Acceptance Gate

Complete only when:

- focused RED/GREEN passes under Python 3.9 and 3.12;
- existing packed stack, state transaction, owner binding, restore, and
  Qwen3.5 shell regressions pass;
- chunked-prefill remains 97 pass / 1 known skip / 0 fail;
- Python 3.9/3.12 `py_compile` and `git diff --check` pass;
- staged files remain empty and experiment evidence remains;
- handoff records that production model selection/checkpoint loading are still
  missing.

Allowed conclusion:

> TinyLLMForge has a locally tested Qwen3.5 root causal-LM execution shell that
> commits recurrent state only after the complete embedding-to-logits path
> succeeds.

Not established:

- production ModelRunner selecting this root shell;
- checkpoint weight-name traversal or loading;
- paged full-attention CUDA correctness;
- Engine startup, Scheduler admission, or prefix reuse;
- token/logit equivalence against a real checkpoint;
- performance, cache, memory, compression, or quality benefit.

