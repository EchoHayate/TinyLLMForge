# Qwen3.5 Runtime Prefix Publication Candidate Design

## Objective

Capture an immutable, exact publication candidate from a completed live
Qwen3.5 prefill without invoking cache publication or changing runtime flow.

## Exact Eligibility

A sequence is eligible only when all conditions hold:

- the source is a live `Sequence`;
- `num_prompt_tokens` is positive and exactly full-block aligned;
- `num_computed_tokens >= num_prompt_tokens`;
- every prompt block exists in `block_table`;
- every prompt block has non-negative hash metadata;
- block token IDs exactly match the frozen prompt tokens;
- block generations and prefix-hash chain are current;
- the Scheduler allocator still owns the exact request lease;
- sequence slot/generation metadata exactly matches that lease.

Non-aligned prompts are rejected rather than truncated. The live recurrent
state after prefill represents the complete prompt, so publishing only an
earlier aligned subset would pair the wrong state with the token prefix.

## Candidate

Add immutable:

```python
Qwen35HybridPrefixPublicationCandidate
```

Fields:

```text
request_id
key
token_ids
block_identities
lease
```

Capture freezes token IDs and block identities as tuples. Later mutations to
`Sequence.token_ids`, `Sequence.block_table`, or Block metadata cannot mutate
the candidate.

The key uses:

- terminal prefix hash for `token_hash` and `terminal_block_hash`;
- exact prompt token count;
- BlockManager block size;
- caller-supplied non-empty model fingerprint;
- caller-supplied non-empty layout fingerprint;
- exact Engine tensor-parallel size;
- caller-supplied supported state dtype.

## Payload Matrix

The candidate exposes:

```python
publication_payloads(ticket_id, world_size)
```

It returns one `Qwen35HybridPrefixPublicationPayload` per contiguous
participant ID, all sharing the frozen identity and live lease. It validates
that `world_size` equals the key tensor-parallel size.

## Scope Boundary

This primitive does not call the Engine publication coordinator. It is not
called from Scheduler postprocess, `LLMEngine.step()`, or ModelRunner `run()`.
It does not retain KV block references after capture; therefore callers must
revalidate or publish while the source request still owns the blocks and
lease. Automatic lifetime pinning is a later gate.

## Tests

Dependency-light tests cover:

- exact aligned completed prefill capture;
- sampled completion token exclusion from the frozen prompt;
- token/block mutation isolation;
- non-aligned, incomplete, missing-hash, stale-generation, stale-token, and
  released-lease rejection;
- exact payload matrix construction and TP mismatch rejection;
- no runtime wiring.

## Claim Boundary

Passing proves only immutable candidate eligibility and construction. It does
not prove safe delayed publication, runtime publication rate, cache hits,
memory reduction, or speedup.
