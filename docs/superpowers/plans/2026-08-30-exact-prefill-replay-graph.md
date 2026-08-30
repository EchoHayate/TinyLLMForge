# Exact Prefill Replay Graph Implementation Plan

Date: 2026-08-30

## Scope

Implement the approved exact-shape prefill CUDA Graph fast path for TP1,
batch-one, dense prefill. Preserve eager fallback and leave the feature off
by default.

## Tasks

1. Add failing configuration tests for enablement and canonical token
   allowlists.
2. Add failing contract tests for identity stability, eligibility, cache
   accounting, quarantine, and replay bookkeeping.
3. Implement the model-agnostic prefill graph contract and cache.
4. Add failing ModelRunner wiring tests for initialization, eligible replay,
   and fail-closed fallback.
5. Wire startup capture and replay into `ModelRunner`.
6. Add a focused benchmark worker and verifier that report exactness, TTFT,
   TPOT, E2E, capture duration, and memory cost.
7. Run local focused tests, then the relevant regression suite.
8. Run a remote clean-GPU smoke followed by the frozen paired gate for 256
   and 2048 prompt tokens.
9. Produce compact evidence, independent verification, an audit, and handoff
   reconciliation.
10. Stage exact paths, commit, push only to
    `origin/feat/kv-sparse-attention`, and verify the remote SHA.
