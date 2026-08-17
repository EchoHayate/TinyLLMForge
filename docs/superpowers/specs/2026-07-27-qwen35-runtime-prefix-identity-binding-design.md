# Qwen3.5 Runtime Prefix Identity Binding Design

## Objective

Bind one canonical cache-key identity to each installed Qwen3.5 model owner
before Engine can derive source-publication configuration.

## Identity Sources

The identity has three fields:

```text
model_fingerprint
layout_fingerprint
dtype
```

`layout_fingerprint` is derived from
`owner.pool.layout.fingerprint`.

`dtype` is derived from the recurrent-state pool layout. Every layout
component must use one identical supported dtype; mixed component dtypes are
rejected because the current cache key has one dtype field.

`model_fingerprint` must be a lowercase 64-character SHA256 supplied by the
verified checkpoint-manifest boundary. It is not inferred from a model path
and the runtime does not scan all weight tensors.

## ModelRunner Contract

`ModelRunner.bind_qwen35_hybrid_prefix_runtime_identity(
model_fingerprint)`:

1. requires an already-bound `Qwen35HybridModelOwner`;
2. validates the manifest SHA256;
3. derives layout fingerprint and state dtype from the owner pool;
4. creates and stores an immutable identity;
5. returns a pickle-safe rank identity row.

An exact repeated SHA256 is idempotent. A different SHA256, owner replacement,
layout drift, or dtype drift fails closed.

## Engine Contract

Engine calls the ModelRunner method through the acknowledged all-rank command
path and validates:

- one row per contiguous rank;
- inner participant ID equals outer rank;
- exact field schema;
- identical model fingerprint, layout fingerprint, and dtype name on all
  ranks.

The resulting canonical configuration may later be passed to the explicit
source-publisher installer. This gate does not install it automatically.

## Dtype Transport

Rows use the canonical strings:

```text
float16
bfloat16
float32
```

Engine converts the validated string back to the corresponding `torch.dtype`
only after all-rank agreement.

## Failure Semantics

Malformed or inconsistent all-rank identity poisons the acknowledgement
collector and leaves Engine identity unset. ModelRunner binding is
single-assignment and never mutates an existing identity.

## Tests

Tests cover:

- SHA256 validation;
- owner-required behavior;
- exact layout/dtype derivation;
- mixed-dtype rejection;
- ModelRunner idempotency and replacement rejection;
- all-rank Engine aggregation;
- rank/model/layout/dtype mismatch poison;
- no automatic use from `LLMEngine.step()`.

## Claim Boundary

Passing proves canonical identity binding and all-rank agreement. It does not
prove the real checkpoint worker passes the verified manifest SHA256, automatic
publisher installation, production cache hits, memory reduction, or speedup.
