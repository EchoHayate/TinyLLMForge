# Engine Hybrid Prefix Publication Transaction Design

## Objective

Compose the five acknowledged publication phases into one explicit Engine
transaction without wiring it into automatic runtime execution.

## Broadcast Rollback Prerequisite

The Engine command channel broadcasts to all ModelRunner ranks. After any
prepare, precommit, or finalize business failure, cleanup must therefore call
rollback on every rank.

`Qwen35HybridPrefixPublicationParticipant.rollback()` becomes an idempotent
abort operation for the same payload:

- prepared, precommitted, or finalized state is reverted normally;
- an already rolled-back ticket returns `rolled_back`;
- a rejected ticket transitions to `rolled_back` and returns `rolled_back`;
- a previously unseen ticket records `rolled_back` and returns `rolled_back`;
- a different payload reusing an existing ticket still returns `error`;
- a committed ticket still returns `error`.

This makes all-rank broadcast rollback truthful: every rank ends in a terminal
non-visible state even if it never acquired a prepared cache handle.

## Engine Coordinator

Add:

```text
Qwen35HybridPrefixEnginePublicationCoordinator
```

The coordinator owns:

- one Engine instance;
- a positive phase timeout;
- a poison reason;
- the last transaction record.

`publish(payloads)` performs:

```text
validate matrix
prepare all ranks
  rejected on any rank -> rollback all ranks -> return False
  error on any rank -> rollback all ranks -> raise
precommit all ranks
  non-precommitted -> rollback all ranks -> raise
finalize all ranks
  non-finalized -> rollback all ranks -> raise
seal all ranks
  non-committed -> poison and raise
return True
```

Rollback must return `rolled_back` on every rank. Any malformed or failed
rollback poisons the coordinator and the Engine acknowledgement collector.

## Transport Failure Boundary

This coordinator guarantees recovery for complete acknowledged phase results
that contain business failure statuses. A timeout, dead worker, process crash,
or transport collector poison can leave phase completion uncertain; the
coordinator poisons and fails closed rather than claiming rollback succeeded.
Durable consensus is outside this gate.

## Installation

`LLMEngine` gains an idempotent installation method and an explicit
`publish_qwen35_hybrid_prefix(payloads)` entry point. Installation validates
that the coordinator targets the same Engine. No automatic call is added to
`step()`, Scheduler, or ModelRunner `run()`.

## Tests

Dependency-light tests cover:

- broadcast rollback on rejected and unseen participant tickets;
- all-rank success;
- prepare rejection returns false after all-rank rollback;
- prepare error, precommit error, and finalize error rollback all ranks;
- rollback failure poisons and blocks reuse;
- seal error poisons and blocks reuse;
- phase transport exception poisons and blocks reuse;
- installation identity/idempotency;
- `LLMEngine.step()` remains publication-free.

## Claim Boundary

Passing proves explicit dependency-light Engine transaction orchestration for
acknowledged business outcomes. It does not prove crash recovery, automatic
runtime publication, real cache hit rate, CUDA memory reduction, or speedup.
