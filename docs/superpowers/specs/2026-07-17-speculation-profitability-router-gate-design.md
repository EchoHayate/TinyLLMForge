# Speculation Profitability Router Gate Design

Date: 2026-07-17

## Objective

Preserve the existing native multi-token verifier where it is useful, bypass
it where it is predictably unprofitable, and build a source-auditable gate
that separates verifier efficiency from draft-source quality.

The first phase must:

1. Route `draft_len <= 1` to normal target decode without entering native
   speculative verification.
2. Route `draft_len >= 2` to the existing native verifier only when its
   compatibility contract passes.
3. Preserve token-identical greedy output, sequence lifecycle, block-table
   state, and continuation output relative to normal decode.
4. Preserve the native verifier's zero accepted-KV replay, copy, and
   rematerialization contract.
5. Measure a controlled acceptance envelope over `K in {2,4,8,16}` before
   making any claim about a real draft source.
6. Run a separate real-source gate whose draft tokens, source identity,
   source cost, and routing decisions are directly attributable.
7. Compare routed speculation against normal decode end to end, not only
   against the legacy rematerializing verifier.
8. Produce `READY_FOR_REAL_DRAFTER_GATE`, `GO`, `NO_GO`, or `INCOMPLETE`
   classifications with independently recomputable artifacts.

This design does not promote the current n-gram, adaptive n-gram, or SAM
draft sources. Their completed canonical performance results remain `NO_GO`.
It also does not claim production batching, non-greedy equivalence, tree
verification, CUDA graph support, KV-offload support, quantized-KV support,
or memory reduction.

## Evidence That Motivates This Work

The completed native verifier run
`qwen3-06b-native-verifier-split16-final-20260716-182334-a1` established:

- all nine exactness cases passed;
- logits, KV, accepted prefixes, metadata, and 16-token continuations matched
  the row-expanded oracle;
- accepted KV required zero replay, copy, or rematerialization;
- accepted `K > 1` verifier-plus-commit median latency was
  `130.24043291807175 ms`, versus
  `162.14731335639954 ms` for the legacy path;
- `K=1` end-to-end latency regressed by `4.349269214547502%`, above the
  preregistered `1%` limit.

The strict first-phase result was therefore `NO_GO`, not
`READY_FOR_PERFORMANCE_GATE`.

The later K1 experiment removed avoidable synchronization from the profiler,
but its 140-row source-auditable canonical gate also produced `NO_GO`.
Consequently, this design must not depend on another K1 micro-optimization.
The reliable conclusion is structural: one proposed token provides no
multi-token target-work reduction, so the normal decode path is the correct
fallback.

The positive `K > 1` native-versus-legacy result is necessary but not
sufficient. It does not prove:

- routed speculation beats normal decode;
- any existing draft source proposes enough accepted tokens;
- draft construction cost is recovered;
- a production-like prompt distribution benefits;
- throughput or latency improves under batching.

The new gate exists to answer those missing questions without reopening
already-settled verifier semantics.

## Alternatives Considered

### 1. Recommended: Deterministic Profitability Router Plus Two-Stage Gate

Add a narrow router in the profiler-owned speculative path:

- `draft_len <= 1` uses normal decode;
- compatible `draft_len >= 2` uses the native verifier;
- incompatible modes fail closed in the controlled envelope and use an
  explicitly recorded baseline fallback in the real-source gate.

First measure a controlled acceptance envelope. Then run a separate
source-attributed real-drafter gate.

Advantages:

- directly removes the only failure in the final native-verifier gate;
- retains the already-proven `K > 1` zero-replay implementation;
- prevents synthetic acceptance from being confused with real-source speed;
- changes little code before draft-source economics are known;
- provides the measurements needed to decide whether learned-drafter work is
  justified.

Risks:

- `draft_len >= 2` is only a safe first routing rule, not a proof of profit;
- a poor draft source can still lose even when every verifier call is routed
  correctly;
- single-sequence profiler results do not establish production batch gains.

### 2. Continue Optimizing K1

Keep every speculative call on the native path and reduce K1 overhead until
it meets the old one-percent threshold.

This is rejected:

- K1 cannot reduce the number of target tokens evaluated;
- the source-auditable K1 canonical gate already failed;
- further tuning risks optimizing profiler synchronization rather than model
  execution;
- it does not improve acceptance or draft quality.

### 3. Integrate a Learned Drafter Immediately

Start with an EAGLE-style or smaller-model drafter and connect it directly to
the current native verifier.

This is the likely high-value follow-up if the controlled envelope is
positive. It is not the first implementation step because:

- checkpoint availability and model compatibility must be established;
- hidden-state and tokenizer contracts are not yet implemented;
- draft-model cost could dominate on Qwen3-0.6B;
- without a profitability envelope, a failed learned-drafter run would not
  distinguish verifier cost from draft-source cost.

### 4. Expand Directly to Production Batching

Add ragged multi-request verification and scheduler integration now.

This is deferred. Batching broadens the compatibility and correctness
surface before single-request draft-source economics are established. It
would also mix scheduler policy, verifier implementation, and drafter quality
in one result.

## Decision

Implement Alternative 1.

The implementation must reuse the existing native verifier. It must not
introduce a second verifier, restore accepted-token replay, or alter the
established oracle tolerances.

The two evidence stages are intentionally different:

1. **Controlled envelope:** determines whether the router and native verifier
   have a profitable region under known acceptance patterns.
2. **Real-source gate:** determines whether a named, source-attributed draft
   source reaches that region often enough to beat normal decode after all
   source costs.

Only the real-source gate may produce a performance `GO`.

## Router Contract

### Inputs

The router receives:

- `draft_tokens`;
- `draft_source`;
- decoder and verifier compatibility state;
- the remaining output-token budget;
- whether the sequence is already finished;
- optional source-provided confidence, recorded for diagnostics only in the
  first phase.

The first-phase routing decision must not inspect target logits, accepted
count, or future timing. It must be reproducible before verification starts.

### Decisions

The router returns exactly one of:

- `baseline_short_draft`;
- `baseline_finished`;
- `baseline_output_budget`;
- `baseline_incompatible`;
- `native_multi_token`.

The fixed decision order is:

1. If the sequence is finished, return `baseline_finished` without another
   model forward.
2. If no output budget remains, return `baseline_output_budget` without
   another model forward.
3. If `len(draft_tokens) <= 1`, return `baseline_short_draft`.
4. If native compatibility fails:
   - controlled envelope: raise before KV mutation;
   - real-source gate: return `baseline_incompatible` and record the exact
     compatibility reason.
5. Otherwise return `native_multi_token`.

There is no adaptive threshold, acceptance EMA, online learning, or
prompt-specific exception in the first phase. Those mechanisms would add
tunable state before the fixed router has a trustworthy baseline.

### Baseline Fallback Semantics

`baseline_short_draft` must be equivalent to normal greedy decode:

- it must not reserve speculative blocks;
- it must not call `prepare_spec_verify()`;
- it must not enter `execution_mode="spec_verify"`;
- it must not create accepted-KV replay/copy/rematerialization events;
- it must not use the proposed token to override the target token;
- it must preserve the normal pending-token lifecycle.

The draft-source construction cost remains part of end-to-end elapsed time.
Bypassing verification must not hide or subtract the cost of producing a
short draft.

### Native Route Semantics

`native_multi_token` calls the existing `verify_and_commit_block()` native
mode. Its existing fail-closed compatibility matrix remains authoritative.
The router cannot silently enable unsupported combinations.

The event must retain:

- proposed and accepted tokens;
- target tokens;
- verifier query length;
- reserved and committed block metadata;
- `accepted_kv_rematerialization`;
- `accepted_kv_copy_calls`;
- `accepted_kv_replay_calls`;
- target-forward counts;
- verifier and commit timing.

## Controlled Acceptance Envelope

### Purpose

The controlled envelope measures the implementation's break-even region. It
is not a draft-source benchmark and cannot produce a product `GO`.

### Cases

For each `K in {2,4,8,16}`, include:

- zero acceptance;
- one-token acceptance;
- partial acceptance;
- full acceptance.

Also include:

- current-block and one-new-block cases;
- a multi-block context;
- EOS acceptance;
- output-budget truncation;
- continuation for at least 16 tokens.

Each case runs these policies in isolated model processes:

- `baseline`;
- `legacy_rematerialize`;
- `always_native`;
- `routed_native`;
- `oracle`.

`always_native` is diagnostic and may regress. `routed_native` is the
candidate. The row-expanded `oracle` is correctness-only and excluded from
performance aggregation.

### Controlled Draft Construction

Controlled drafts are derived from a serialized target-token probe and then
mutated to obtain the required accepted-prefix length. The probe is run in an
isolated process with its own dynamic ports.

This mechanism must be labelled `controlled_target_derived`. It must never be
called a real drafter, and its construction time must not be used in
end-to-end performance comparisons.

### Envelope Metrics

For every case and policy, record:

- end-to-end elapsed time;
- output tokens and output tokens per second;
- target forward count;
- verifier query length;
- proposed and accepted token counts;
- accepted tokens per target forward;
- verifier-plus-commit latency;
- maximum allocated GPU memory, diagnostic only;
- exact output and continuation hashes.

For each `(K, accepted_count)` pair, compute:

- routed-versus-baseline elapsed ratio;
- routed-versus-always-native elapsed ratio;
- native-versus-legacy verifier-plus-commit ratio;
- target-forward reduction;
- exactness and replay-elimination status.

### Envelope Classification

The controlled stage is
`READY_FOR_REAL_DRAFTER_GATE` only when:

1. every required row and capability row is present;
2. all process return codes are zero;
3. exact output, acceptance, metadata, KV, logits, and continuation checks
   pass;
4. every native event has zero accepted-KV replay, copy, and
   rematerialization;
5. every short-draft router case proves no speculative reservation, prepare,
   or forward;
6. at least one preregistered `K >= 2` accepted region has a routed-versus-
   baseline median elapsed ratio below `0.95`;
7. zero- and one-acceptance rows are included in the aggregate rather than
   discarded;
8. no required lifecycle case regresses by more than `5%`.

It is `NO_GO` when complete semantic evidence fails or no controlled region
beats baseline by at least `5%`.

It is `INCOMPLETE` for missing rows, failed processes, unavailable kernels,
invalid timings, missing source evidence, hash mismatch, or an unreconciled
artifact.

`READY_FOR_REAL_DRAFTER_GATE` is not a performance `GO`.

## Real-Source Gate

### Required Source Identity

The real-source gate accepts exactly one named source configuration per
canonical artifact. The source manifest records:

- source type and implementation path;
- source commit and dirty state;
- owned source-file SHA-256 values;
- model or checkpoint identifier, if any;
- model/checkpoint config SHA-256;
- tokenizer identity and vocabulary size;
- source hyperparameters fixed before the run;
- whether target hidden states are consumed;
- whether the source requires an additional model forward;
- the exact prompt-bank hash.

Debug stubs, target-derived drafts, forced acceptance, oracle tokens, and
post-hoc token replacement are forbidden in this stage.

### Eligible Sources

A source is eligible only when it produces tokens from information available
before the target verification result. Examples include:

- deterministic prompt lookup;
- a separately identified smaller draft model;
- a separately identified learned speculative head;
- an EAGLE-style hidden-state drafter with an attributable checkpoint.

The existing prompt lookup, n-gram, adaptive n-gram, and SAM implementations
may be used only as negative controls because their canonical results already
failed. A new `GO` requires a materially different source or checkpoint, not
threshold tuning on the same prompt bank.

### Policies

The canonical real-source run compares:

- `baseline`;
- `source_always_native`;
- `source_routed_native`.

All policies use the same prompts, output budgets, seeds, dtype, model,
warmup count, repetition count, and process isolation. Baseline does not
construct a draft. Source policies include all draft construction and
transfer costs.

### Prompt Matrix

The prompt bank must be fixed and hashed before the canonical run. It must
contain distinct buckets:

- natural conversational prompts;
- code or structured completion;
- repetitive text;
- transition-heavy text;
- low-match or adversarial text;
- EOS-sensitive requests;
- short and long contexts.

Calibration prompts, if any, are separate from evaluation prompts. No
threshold may be changed after evaluation rows are visible.

### Real-Source Metrics

Record and aggregate:

- end-to-end elapsed time and output tokens per second;
- median and per-prompt routed-versus-baseline ratios;
- draft construction latency;
- target verification latency;
- proposed, accepted, and rejected tokens;
- acceptance rate by proposed token and by verifier call;
- router decision counts;
- fallback reason counts;
- accepted tokens per target forward;
- target-forward reduction versus baseline;
- source overhead as a fraction of total elapsed time;
- exact output mismatch count;
- continuation mismatch count;
- maximum allocated GPU memory, diagnostic only.

The report must show natural and transition-heavy buckets separately. A gain
on repetitive prompts cannot hide a regression on natural prompts.

### Real-Source Classification

The canonical real-source stage is `GO` only when:

1. source identity and executed bytes are independently verified;
2. every required row succeeds and all dynamic port pairs are unique;
3. exact greedy output and lifecycle checks pass with zero mismatch;
4. native events preserve zero accepted-KV replay, copy, and
   rematerialization;
5. routed speculation improves aggregate median end-to-end elapsed time over
   baseline by at least `5%`;
6. routed speculation improves aggregate output tokens per second by at
   least `5%`;
7. natural-prompt and transition-heavy median elapsed ratios are each at
   most `1.00`;
8. no individual required prompt regresses by more than `10%`;
9. the routed policy is no slower than `source_always_native`;
10. the source is exercised with both native and baseline-fallback decisions;
11. the measured target-forward reduction is positive;
12. all thresholds were fixed before the canonical run.

It is `NO_GO` when complete and correct evidence fails any performance or
regression threshold.

It is `INCOMPLETE` for evidence, infrastructure, process, capability,
coverage, source-identity, or artifact-verification failures.

## Source-Auditable Artifact Contract

Each stage has a unique run directory:

```text
experiments/speculation_router/<run_tag>/
```

The canonical artifact contains:

- `source_evidence.json`;
- `source.patch`;
- `source_snapshot.tar.gz`;
- `manifest.json`;
- `capability.json`;
- `case_rows.json`;
- `event_rows.json`;
- `router_rows.json`;
- `summary.json`;
- `report.md`;
- `artifact_hashes.json`;
- `remote_exitcode`;
- `runner.log`.

The real-source stage additionally contains:

- `draft_source.json`;
- `prompt_bank.json`;
- `prompt_bank.sha256`;
- checkpoint/config hashes when applicable.

The source-evidence mechanism follows the existing adaptive n-gram
Source-Auditable Manifest contract:

- enumerate owned regular files deterministically;
- record size and SHA-256 for every file;
- record a canonical tree SHA-256;
- save a binary Git patch against the base commit;
- verify the local approved snapshot before upload;
- verify the remote snapshot before model initialization;
- reconstruct and verify the downloaded snapshot locally;
- reject untracked owned files unless the manifest format explicitly
  includes them.

Artifact verification recomputes classifications from raw rows. It must not
trust `summary.json` or `report.md` as authoritative.

## Process Isolation

Every model process receives a unique pair:

- `TINYVLLM_DIST_PORT`;
- `MASTER_PORT`.

The manifest records every pair and the verifier rejects duplicates.

Remote execution uses:

- host `sitian@10.232.195.203`;
- Python
  `/data00/home/sitian/sitian-workspace01/tllm/env/bin/python`;
- model
  `/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B`;
- a run-local remote directory;
- a run-local `TMPDIR`.

The runner must not kill unrelated GPU processes, modify system `/tmp`
contents, or reuse another run's source or artifacts.

## Error Handling

The gate fails closed:

- router exceptions before mutation are recorded as process or compatibility
  failures;
- exceptions after block reservation require explicit cleanup evidence;
- missing or non-finite timings produce `INCOMPLETE`;
- missing router events produce `INCOMPLETE`;
- output, acceptance, lifecycle, KV, logit, or continuation mismatch produces
  `NO_GO`;
- replay, copy, or rematerialization on the native route produces `NO_GO`;
- source or artifact hash disagreement produces `INCOMPLETE`;
- complete performance-threshold failure produces `NO_GO`;
- unavailable real-source checkpoint or incompatible tokenizer produces
  `INCOMPLETE`, not a verifier `NO_GO`.

Retries may replace only rows whose process failed or whose evidence is
incomplete. Successful rows are immutable under `--resume`. Replaced rows
must also replace their stale events and router decisions.

## Testing Strategy

### Dependency-Light Tests

Tests must cover:

- all router decisions and decision ordering;
- `K=0/1` baseline fallback with no reservation, prepare, spec forward, CUDA
  synchronization, or replay event;
- `K>=2` native dispatch;
- compatibility failure before mutation;
- explicit real-gate fallback for incompatible modes;
- exact event schema;
- synthetic complete `READY_FOR_REAL_DRAFTER_GATE`;
- synthetic real-source `GO`;
- semantic and replay `NO_GO`;
- performance `NO_GO`;
- structural and source-evidence `INCOMPLETE`;
- missing and duplicate row rejection;
- duplicate port rejection;
- non-finite timing rejection;
- artifact tamper detection;
- failed-row resume replacement and successful-row immutability.

Existing native verifier contract, model-runner, attention, oracle, and gate
tests remain required regression coverage.

### Remote Smoke

Run a reduced controlled envelope first. It must exercise:

- one baseline short-draft fallback;
- one zero-accept native case;
- one partial-accept native case;
- one full-accept native case;
- one block boundary;
- one continuation comparison.

Only after the smoke artifact verifies locally may the complete controlled
envelope run.

A real-source smoke is separate and cannot reuse controlled target-derived
drafts.

## Claim Boundaries

`READY_FOR_REAL_DRAFTER_GATE` proves:

- K1/short-draft negative work is bypassed;
- the router preserves normal-decode semantics;
- a controlled profitable `K>=2` acceptance region exists;
- the existing native verifier still eliminates accepted-KV replay.

It does not prove that any real source reaches that region.

A real-source `GO` proves only:

- the named source and fixed Qwen3-0.6B single-sequence workload beat the
  recorded baseline under the preregistered thresholds.

It does not prove:

- production scheduler throughput;
- multi-request or ragged verification gains;
- another model or checkpoint benefits;
- non-greedy sampling equivalence;
- reduced GPU memory;
- KV-offload compatibility.

## Follow-Up Decision

After the controlled envelope:

- `READY_FOR_REAL_DRAFTER_GATE`: evaluate checkpoint availability and design
  the smallest attributable learned-drafter integration, preferring an
  EAGLE-style or smaller-model source over more prompt-lookup threshold
  tuning;
- `NO_GO`: stop native-verifier performance work because even controlled
  accepted regions do not beat normal decode;
- `INCOMPLETE`: repair evidence or infrastructure without changing
  thresholds.

After the real-source gate:

- `GO`: design a separate production batch and scheduler integration gate;
- `NO_GO`: preserve the negative artifact and change draft architecture or
  checkpoint rather than tuning the same evaluation prompts;
- `INCOMPLETE`: repair source attribution or execution evidence before
  interpreting performance.

## Non-Goals

This phase does not:

- commit or promote the failed K1 fast path;
- reinterpret the adaptive n-gram or SAM `NO_GO`;
- train a learned drafter;
- download an unapproved checkpoint;
- modify the production scheduler;
- add tree speculation;
- add stochastic acceptance;
- alter KV-offload, Quest, Attention Matching, KV cartridge, or quantized-KV
  behavior;
- claim production readiness from a profiler-only gate.
