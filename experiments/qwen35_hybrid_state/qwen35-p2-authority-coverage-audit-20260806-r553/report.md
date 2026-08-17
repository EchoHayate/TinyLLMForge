# Qwen3.5 P2 Authority Coverage Audit

## Decision

`LOCAL_AUTHORITY_HARDENED_REAL_RESULT_MISSING`

The schema-v2 independent verifier is now aligned with all 12 approved frozen
thresholds. This prevents a real P2 artifact from being classified `GO` while
silently ignoring an approved W1 or W2 TTFT gate.

This is local authority hardening, not a P2 performance result.

## Defects Found

The contract froze these thresholds:

```text
w1 int8 / recompute median TTFT <= 0.85
w2 int8 / exact median TTFT     <= 1.03
w2 int8 / exact every TTFT      <= 1.05
```

Before this audit, `_threshold_checks()` did not consume any of those three
keys. A structurally valid artifact could violate one of them and still be
classified `GO`.

Three negative tests first reproduced the problem:

1. W2 int8/exact median TTFT `1.05` incorrectly returned `GO`;
2. one W2 repetition at int8/exact TTFT `1.06` was not rejected;
3. W1 int8/recompute median TTFT `0.88` incorrectly returned `GO`.

The verifier now consumes all three thresholds. No threshold value, workload,
schema, profile, or classification precedence was changed.

## Verification

```text
schema-v2 frozen thresholds: 12
verifier-consumed thresholds: 12
new negative regressions:     3
full verifier suite:          19 passed in 609.20s
py_compile:                   PASS
diff check:                   PASS
```

Verifier SHA256:

```text
2ee9365c7fd0da731366a4ffba582d7702bb1ab67639d4786f6ef578c5953b0f
```

Verifier-test SHA256:

```text
d4cae19b34b1b37eaf183735cad6667cf31afeab1bb7b6dd38b7810e6c7c4711
```

The final full-suite rerun after report hardening was:

```text
19 passed in 609.20s
```

## Human-Readable Report Authority

The independent `report.md` generated for every legitimate classification now
includes:

- the classification and canonical token/logit correctness result;
- a claim boundary binding the result to the listed model, source, workload,
  configuration, thresholds, and artifact hashes;
- source-tree, model-manifest, and workload-manifest SHA256 values;
- TP world size, fixed GPU indices, profiles, workloads, sampling policy,
  repetition counts, hybrid-prefix limits, and W3 concurrency;
- W1/W2 exact-relative and recompute-relative TTFT ratios;
- W3 exact-relative and recompute-relative concurrent E2E proxy ratios;
- decode-latency, peak CUDA-reserved, unique physical-cache-byte, and
  same-budget-capacity ratios;
- all 12 frozen threshold names and values;
- a measured/operator/threshold/status row for every gate; and
- every producer artifact path and SHA256 from the validated artifact
  manifest.

Two test-first report regressions cover both sides:

1. a canonical `GO` report must expose the complete bound metric, threshold,
   configuration, hash, and W3 claim-boundary surface;
2. a legitimate `NO_GO_PERFORMANCE` caused only by CUDA peak-reserved ratio
   `1.10 > 1.05` must print that exact gate as `FAIL`.

This does not change the canonical result JSON schema, thresholds, workload,
profiles, or classification precedence. It makes the existing independent
decision auditable by a human without weakening machine authority.

The schema-v1 strict-P1 source bundle is unchanged because these schema-v2
verifier files are outside its 91-file owned-source domain:

```text
source tree:
88ebda203decaf26a721f6299a5c9ed08b648841feb0b972f1395d6cc609eb9c

source tar:
8f807c0e51010fddd0bd474670ae41c09beda604c8247f672fc047f533c53a5d
```

## Coverage of the User Goal

The canonical P2 authority has explicit gates for:

- exact output token equality and final-logit allclose;
- unique resident snapshot-storage bytes rather than logical bytes;
- same-budget cache entry capacity;
- W1/W2 TTFT relative to exact restore and recompute;
- W3 eight-request concurrent E2E ratios;
- decode latency;
- CUDA peak reserved bytes;
- forbidden corruption, fallback, mixed-representation, eviction, and OOM
  events.

The cache implementation derives resident bytes from owned tensor storages and
separately reports logical, metadata, deduplicated, encode-workspace, and
decode-workspace bytes. This is not copy-simulation accounting.

## Industry-Alignment Boundary

The current design follows the same broad principles used by production
engines:

- content/identity-safe prefix reuse;
- page or resident-storage accounting rather than logical-token accounting;
- explicit eviction and failure counters;
- paired TTFT, latency, capacity, and device-memory measurements;
- independent artifact verification rather than trusting producer summaries.

One boundary remains: W3's field named `throughput_ratio` is reconstructed as
the ratio of median per-request E2E latency under an eight-request concurrent
case. It does not record one batch makespan, requests per second, output tokens
per second, arrival-rate saturation, queueing delay, or long-running steady
state.

Therefore a canonical P2 `GO` may support the frozen claim:

```text
under the registered eight-request W3 workload, P2 preserved or improved the
approved concurrent E2E proxy ratios
```

It must not be expanded into:

```text
universal or sustained online serving throughput improved
```

A later serving stress authority should add controlled arrival rate,
batch-window makespan, requests/s, tokens/s, TTFT/ITL percentiles, queueing,
cache-hit rate, and allocator/device-memory telemetry. It must supplement, not
replace or weaken, canonical schema-v2.

## Remaining Critical Path

The user-level goal still requires:

1. fresh read-only resource preflight `READY`;
2. current-source strict-P1 independently classified `GO`;
3. real full-fidelity capture and calibration `PASS`;
4. canonical real P2 independently classified `GO`;
5. only then, an optional sustained-serving stress comparison.

No SSH, remote query, remote path, CUDA import, or GPU operation occurred in
this audit.
