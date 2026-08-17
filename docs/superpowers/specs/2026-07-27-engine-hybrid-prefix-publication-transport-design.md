# Engine Hybrid Prefix Publication Transport Design

## Objective

Transport the five rank-local publication phases through the existing
acknowledged Engine-to-ModelRunner command channel and aggregate one strict
result per tensor-parallel rank.

## Payload Transport

The shared-memory command channel broadcasts one argument list to every rank,
but each publication payload is explicitly bound to one `participant_id`.
Therefore Engine phase APIs accept and broadcast a complete payload matrix:

```text
tuple[Qwen35HybridPrefixPublicationPayload, ...]
```

Before dispatch, Engine validates:

- exactly one payload per ModelRunner rank;
- participant IDs are contiguous and match matrix positions;
- ticket, request, key, tokens, and block identities match across ranks;
- key tensor-parallel size equals ModelRunner world size.

Each ModelRunner phase method accepts either its existing single rank-local
payload or the complete matrix. For a matrix, it validates rank coverage and
selects only the payload whose `participant_id` equals `self.rank`.

## Engine Phase APIs

Add explicit acknowledged methods:

```text
prepare_model_runner_hybrid_prefix_publication
precommit_model_runner_hybrid_prefix_publication
finalize_model_runner_hybrid_prefix_publication
seal_model_runner_hybrid_prefix_publication
rollback_model_runner_hybrid_prefix_publication
```

Each API:

1. validates the complete payload matrix before dispatch;
2. calls the corresponding ModelRunner method through
   `call_model_runner_acknowledged`;
3. validates exact outer-rank/inner-participant binding;
4. returns rows ordered by participant ID.

The allowed status matrix remains:

```text
prepare: prepared | rejected | error
precommit: precommitted | error
finalize: finalized | error
seal: committed | error
rollback: rolled_back | error
```

Valid business outcomes such as `rejected` or `error` are preserved for a
later transaction coordinator. Malformed result dictionaries, rank swaps,
wrong operations, unsupported statuses, and non-string details poison the
acknowledgement collector and fail closed.

## Scope Boundary

This phase does not add an Engine `publish()` transaction coordinator. It does
not automatically rollback after prepare/precommit/finalize errors and does
not poison merely because a valid nested row reports business status `error`.
It also does not call publication from `LLMEngine.step()`, Scheduler, or
ModelRunner `run()`.

## Tests

Dependency-light AST tests prove:

- ModelRunner selects its own payload from a complete rank matrix;
- malformed, incomplete, duplicate, or wrong-rank matrices fail before
  participant delegation;
- each Engine phase dispatches the exact method and aggregates ordered rows;
- payload identity mismatch fails before dispatch;
- malformed nested results poison the collector;
- publication remains absent from `LLMEngine.step()`.

## Claim Boundary

Passing proves acknowledged all-rank phase transport and strict aggregation.
It does not prove distributed publication transaction recovery, automatic
runtime publication, cache-hit rate, CUDA memory reduction, or speedup.
