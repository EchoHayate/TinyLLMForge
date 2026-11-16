# Tokenizer Vocabulary Digest Implementation Plan

**Goal:** Reduce repeated canonicalization of unchanged vocabularies without
removing any observation, validation, message, token-ID or budget check.

**Architecture:** Keep a private one-entry vocabulary/digest memo per measurement
and per owned session. Each checkpoint still obtains and validates the complete
dictionary. At the digest boundary take an owned copy and validate its exact
entry types again, since intervening property getters can mutate the original.
Compare the copy before reusing the canonical digest; changed contents are
hashed normally and rejected by the
existing identity comparison. Public records and hashes stay unchanged.

**Tech Stack:** Existing Python unittest; pinned Python 3.12.13 /
Transformers 4.57.1 for the tokenizer-only remote diagnostic.

## Constraints And Alternatives

- Do not move checks to session boundaries: that would weaken intermediate
  mutation detection. Do not increase the full-audit timeout as a substitute
  for fixing the measured repeated work.
- Memoize only a complete validated `dict[str, int]`, never token outputs.
  Preserve exact types, nonnegative IDs, template/direct encoding agreement,
  complete answer-prefix/EOS checks, 4096/256 limits and fail-closed lifecycle.
- No global cache or cross-session reuse. Release the session memo when the
  tokenizer is released. A mapping returned by the tokenizer must not alias the
  stored copy.
- No dependency installation, telemetry, model/weight loading, generation,
  training, GPU work, full 324-row rerun, commits, pushes or worktree changes.
- Keep the previously sealed audit unchanged. Fresh diagnostic run identities
  refer separately to the existing read-only snapshot identity.

## Evidence Before Changes

Fresh public-input diagnostic `bbea229bd6514ce99eb254429e0f1e3f` completed in
18.916 seconds on the pinned host. Existing snapshot and runtime checks passed.
Loading took 2.406 seconds. Prompt-only / answer measurements took 3.075 / 4.386
seconds; checkpoint timers accounted for 3.038 / 4.362 seconds respectively.
A separate cProfile answer pass spent 2.409 seconds in JSON encoding,
1.111 seconds reading vocabulary and 0.371 seconds serializing the backend.
These are diagnostic samples, not a full-inventory acceptance result or proof
of the exact historical timeout phase.

## Steps

- [x] Capture current sources/ledger privately and verify the frozen baseline.
- [x] Add regression cases proving the old path repeatedly hashes an unchanged
  vocabulary, while retaining all vocabulary reads.
- [x] In `tokenizer_measurement.py`, add `_VocabularyDigest` with an owned
  dictionary copy; thread it through `_state` and `_render` for one measurement.
  Leave the public `measure(tokenizer, messages, answer=None)` API unchanged.
- [x] In `tokenizer_session.py`, pass a session-owned memo to `_observe`;
  standalone observation remains uncached. Clear the memo at release.
- [x] Test same-size vocabulary changes, reused mutable mapping, invalid bool /
  float / negative IDs, unchanged outputs, and all existing mutation checks.
- [x] Reproduce and fix the review finding: getter mutation after the earlier
  type check could exploit `1 == True == 1.0`. Six regression subcases first
  failed; the digest-boundary owned-copy validation closes that gap.
- [x] Run focused regression, then the full existing suite:
  `PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/opt/python@3.12/bin/python3.12 -m unittest discover -s experiments/lora_necessity -p 'test_*.py' -q`.
- [x] Obtain independent review of only the changed files against `before/`.
- [x] Freeze a new diagnostic candidate; run one matching public-input timing
  comparison on the same pinned environment, without replaying the full audit.
- [x] Record measured improvement, equal measurement hashes and remaining
  limits; preserve old evidence and leave the branch uncommitted.

Final evidence: 78 focused tests passed; 362 full-suite tests run, 360 passed,
two platform skips. Sixty-four synthetic complete records matched byte-for-byte.
The reviewed remote comparison completed, with unchanged measurement hashes and
checkpoint counts. Prompt/answer latency fell 27.3%/34.8% in single-sample
diagnostics. Full-inventory and benchmark acceptance remain unproven.

## Private Evidence

`/private/var/folders/0c/hl2r_50x28j2ysjvv2120t8h0000gn/T/tlf-tokenizer-timing-20261114-b3DpnuLS`
