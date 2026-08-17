# Qwen3.5 Source-Bound Publication Design

## Objective

Provide an explicit synchronous API that captures a live aligned prefix,
revalidates its source identity, and runs the Engine publication transaction.

## Concurrency Contract

TinyLLMForge currently has no request lifetime lock or pin. This API is valid
only when called synchronously on the same Engine control thread while the
source request is not being concurrently scheduled, preempted, or released.

The sequence is:

```text
capture exact candidate
revalidate Sequence, BlockManager, and allocator identity
build rank payload matrix
run synchronous Engine publication transaction
```

After all-rank prepare returns, every participant owns private state clones;
later source mutation no longer affects the transaction.

## Candidate Revalidation

`candidate.validate_source(...)` recaptures the exact current identity using
the candidate key parameters and requires equality with the frozen candidate.
Token, block, generation, hash, lease, model/layout fingerprint, TP size, and
dtype drift fail before Engine publication dispatch.

## Source Publisher

Add:

```text
Qwen35HybridPrefixSourcePublisher
```

It validates:

- the Engine has Scheduler block/state owners;
- the caller uses the exact Engine BlockManager and allocator indirectly;
- ticket IDs are monotonic within the publisher;
- only one call is active at a time;
- the Engine publication coordinator is installed.

`publish(sequence)` returns the Engine transaction boolean. Ineligible sources
raise before publication; cache oversize rejection returns false.

## Scope Boundary

The publisher is not installed or called from `LLMEngine.step()`. It does not
make cross-thread or asynchronous publication safe. It does not pin request
lifetime beyond synchronous control flow.

## Tests

Tests cover success, oversize false propagation, pre-dispatch source drift
rejection, missing Engine owners/coordinator, reentrant-call rejection,
monotonic ticket matrices, and runtime disconnection.

## Claim Boundary

Passing proves explicit single-thread synchronous source-bound publication.
It does not prove automatic postprocess wiring, concurrent safety, production
hit rate, memory reduction, or speedup.
