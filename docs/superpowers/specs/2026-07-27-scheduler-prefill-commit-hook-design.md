# Scheduler Prefill Commit Hook Design

## Objective

Expose a default-off synchronous callback after complete prompt KV/hash
metadata is committed but before the first sampled token is appended or
request resources can be released.

## Hook Contract

Scheduler adds:

```text
install_prefill_commit_hook(hook)
```

The hook is either absent or a callable receiving one live `Sequence`.
Installation is idempotent for the same callable and rejects replacement.

The hook runs only when:

- the sequence has just completed its full prompt prefill;
- `num_computed_tokens >= num_prompt_tokens`;
- the request has not already been notified.

It runs after `BlockManager.commit_prefill()` and the
`num_computed_tokens` update, but before:

- appending a sampled completion token;
- changing the final prefill request to running/finished;
- releasing KV blocks or hybrid-state lease.

## Coverage

The same helper is called from:

- legacy prefill postprocess;
- final chunked prefill postprocess;
- final mixed prefill postprocess.

Non-final chunks do not trigger the hook. A full-prefix-cache hit that
recomputes the last prompt token for logits triggers at most once.

## Failure Semantics

If the hook raises, Scheduler stores a poison reason and re-raises before token
append or resource release. All future `schedule()` calls fail closed with the
same poison reason. Notification is recorded only after successful hook
completion.

## Scope Boundary

The hook is default-off and generic. This phase does not install the
source-bound publisher, change Engine `step()`, or publish automatically.

## Tests

Tests cover default-off behavior, installation validation/idempotency,
legacy/chunked/mixed ordering, non-final suppression, one-shot notification,
resource liveness during callback, and poison after callback failure.

## Claim Boundary

Passing proves a synchronous metadata-stable, resource-live callback boundary.
It does not prove Engine publisher wiring, production publication, cache hits,
memory reduction, or speedup.
