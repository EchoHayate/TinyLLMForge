# Explicit Prefill Publication Integration Design

## Objective

Prove the explicit opt-in path from real Scheduler complete-prefill commit
through real candidate capture, source-bound publisher, and Engine publication
coordinator while the request KV blocks and hybrid-state lease are still live.

## Test Architecture

The dependency-light integration uses:

- real `Scheduler`;
- real `BlockManager`;
- real `HybridStateSlotAllocator`;
- real `Qwen35HybridPrefixSourcePublisher`;
- real publication candidate and payload types;
- real `Qwen35HybridPrefixEnginePublicationCoordinator`;
- one-rank ModelRunner publication phase stubs.

The phase stubs return valid prepare/precommit/finalize/seal rows and inspect
the source request during every phase.

## Success Path

After explicit Engine installation, legacy full prompt prefill must:

1. commit complete block hashes;
2. invoke source publication once;
3. capture and revalidate exact prompt/block/lease identity;
4. complete prepare, precommit, finalize, and seal;
5. only then append the sampled token and release request storage.

The test asserts that every transaction phase observes no completion token,
live block ownership, and a valid hybrid-state lease.

## Default-Off Path

Without explicit installation, the same Scheduler prefill completes normally
and no publication phase runs.

## Failure Path

An injected publication phase failure must propagate through the Scheduler
hook before token append or release. Scheduler must retain live request
resources and poison all future scheduling.

## Scope Boundary

This gate does not run a real model, worker process, CUDA tensor pool, or
automatic Engine startup configuration. It proves synchronous lifecycle
integration only.

## Claim Boundary

Passing proves the explicit single-rank dependency-light hook-to-transaction
chain. It does not prove multi-process runtime behavior, actual recurrent
state tensor contents, production cache hits, memory savings, or speedup.
