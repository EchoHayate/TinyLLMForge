# Identity-Driven Source Publisher Installation Design

## Objective

Install the Qwen3.5 source publisher using only the Engine's already validated
canonical runtime identity.

## Interface

`LLMEngine` adds:

```text
install_configured_qwen35_hybrid_prefix_source_publisher()
```

The method takes no identity arguments.

## Contract

The method requires
`qwen35_hybrid_prefix_runtime_identity` to be a
`Qwen35HybridPrefixRuntimeIdentity`.

It delegates to:

```text
install_qwen35_hybrid_prefix_source_publisher(
    model_fingerprint=identity.model_fingerprint,
    layout_fingerprint=identity.layout_fingerprint,
    dtype=identity.dtype,
)
```

Therefore it inherits the existing stable-hook, installation atomicity,
idempotency, and replacement-rejection semantics.

## Failure Semantics

Missing or invalid Engine identity fails before Scheduler mutation.
If a manually installed publisher has different configuration, the existing
installer rejects the conflict without mutation.

## Default-Off Boundary

The method remains explicit. Neither `LLMEngine.__init__()` nor
`LLMEngine.step()` calls it.

## Tests

Tests prove missing-identity rejection, exact delegation, idempotency,
conflicting manual installation rejection, and `step()` disconnection.

## Claim Boundary

Passing removes duplicate caller-supplied layout/dtype configuration. It does
not connect checkpoint-worker identity handoff or automatic runtime enablement.
