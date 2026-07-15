# Prompt+Dynamic SAM Drafter Gate Design

Date: 2026-07-15

## Objective

Determine whether a prompt-conditioned, dynamically updated suffix automaton
(SAM) can provide a materially better speculative draft source for
TinyLLMForge than the completed n-gram candidates, without requiring a draft
checkpoint or changing runtime scheduling.

The first phase must:

1. Preserve exact greedy output through the existing
   `verify_and_commit_block()` path.
2. Build one CPU-only token suffix automaton from each sequence's prompt and
   extend it only with target-verified generated tokens.
3. Match the longest suffix of the current history, then draft tokens from an
   earlier occurrence's continuation.
4. Compare normal greedy, fixed n-gram `K=4`, adaptive n-gram
   `K in {1,2,4}`, fixed SAM `K=16`, and match-aware SAM
   `K in {0,4,8,16}`.
5. Measure wall-clock throughput, target verify work, zero-accept work,
   drafted-token waste, lookup/build overhead, match quality, and selected-`K`
   trajectories.
6. Produce a strict `GO`, `NO_GO`, or `INCOMPLETE` decision from thresholds
   fixed before the canonical measurements.

This is a profiler-only, greedy, single-sequence gate. It does not implement
or claim production scheduler integration, ragged batched verification, batch
throughput, queue-tail improvement, non-greedy distribution equivalence, or
memory reduction.

## Motivation

The adaptive n-gram canonical gate completed 140 of 140 isolated process runs
on Qwen3-0.6B. It preserved exact output and reduced verification waste, but
its median throughput was `1.0222%` below the best fixed policy, narrowly
missing the preregistered `-1%` near-best fallback. Its decision was therefore
strictly `NO_GO`.

That result identifies draft quality, rather than verify/commit correctness,
as the next bottleneck:

- fixed `K=4` was faster than the adaptive n-gram policy;
- reducing rejected work did not compensate for weaker or shorter useful
  proposals;
- the available remote host has Qwen3-0.6B and Qwen3-8B, but no smaller
  tokenizer-compatible Qwen3 draft checkpoint;
- EAGLE-style and DFlash-style learned drafters require matched training and
  checkpoints that are not currently available.

A prompt+dynamic SAM is the next bounded candidate because it can index all
prompt substrings in linear space, find variable-length suffix matches, and
learn target-confirmed output structure online. It requires no external
corpus, no learned weights, and no change to target verification.

## Alternatives Considered

### 1. Recommended: Prompt+Dynamic Token SAM

Build a per-sequence suffix automaton over token IDs. Use the current history's
longest suffix match to select a previously observed continuation, with a
match-aware draft cap.

Advantages:

- no checkpoint, training pipeline, or tokenizer compatibility problem;
- variable-length matching is more expressive than one fixed n-gram order;
- prompt and generated-token indexing avoids external-corpus leakage;
- deterministic CPU data structures are independently testable;
- existing target verify/commit and canonical gate infrastructure can be
  reused.

Risks:

- online lookup and maintenance overhead may erase GPU step savings;
- repetitive prompts can create many possible occurrences with different
  continuations;
- a long structural match does not guarantee the continuation is still valid;
- Python implementation overhead may understate the value of a future native
  implementation.

### 2. Multi-Pattern Hash Table

Maintain rolling-hash tables for several suffix lengths, such as
`{2,3,4,6,8,12,16}`, and select the longest available match.

This is simpler and may have lower constants, but duplicates index entries
across lengths, requires a hand-selected length set, and does not represent all
substrings compactly. It remains a useful fallback if the SAM algorithm is
correct but profiler overhead dominates.

### 3. Learned Drafter: EAGLE-Style or DFlash-Style

Train or obtain a target-compatible draft head/model and use the existing block
verify/commit ABI.

This has the highest potential acceptance quality, but the current environment
lacks a compatible checkpoint and training artifacts. Starting there would
combine model-training risk, checkpoint provenance, hidden-state ABI work, and
runtime benchmarking in one gate. It is deferred until a weight source or
reproducible training path exists.

## Decision

Implement the first gate as a pure token-level suffix automaton owned by
`tools/profile_ngram_commit.py` and helper modules under
`tinyvllm/speculative/`.

The SAM changes only draft production and cap selection. It does not change:

- target logits or greedy token selection;
- `verify_and_commit_block()` acceptance semantics;
- KV reservation, accepted-token commit, or block release;
- EOS and output-budget handling;
- production scheduler or `Sequence` state;
- target model weights or hidden-state interfaces.

The canonical result will decide whether the approach deserves runtime/native
optimization. A `GO` does not itself authorize production integration.

## SAM Semantics

### Indexed Stream

Each candidate sequence owns one `SuffixAutomatonDraftIndex`.

Initialization:

1. tokenize the prompt through the normal profiler path;
2. insert prompt token IDs into the SAM in order;
3. retain the exact indexed token stream beside the automaton;
4. mark the prompt boundary in event metadata.

Online extension:

- after each normal target decode, append the newly generated target token;
- after a speculative verify, append only accepted target tokens;
- append the mismatch/fallback target token when the existing decode path
  commits it;
- never append rejected draft tokens;
- never update the index from unverified draft content.

The indexed stream must equal the sequence's target-verified token history at
every proposal boundary. A profiler invariant checks this before lookup.

### State Representation

Every SAM state contains:

```text
max_length: int
suffix_link: int
transitions: dict[token_id, state_id]
first_end_position: int
```

`first_end_position` is the end position of the state's earliest representative
occurrence. A newly created non-clone state receives the current indexed
position. A clone copies the source state's value.

The first gate deliberately uses one earliest occurrence rather than retaining
or propagating complete end-position sets. Standard SAM construction then
remains amortized linear, and an occurrence never becomes invalid as the
indexed stream grows. This gives deterministic continuation selection but does
not attempt to rank multiple occurrences by recency or continuation quality.

The initial implementation favors auditable behavior over maximal compression:

- token IDs are integer transition keys;
- state IDs are stable list indices;
- no Torch or CUDA objects are stored;
- no global or cross-request index exists;
- no external corpus is loaded.

### Query Boundary

The lookup query is the full target-verified history immediately before a
draft proposal. Because that exact history is already indexed, a naive lookup
could select the sequence's current terminal occurrence and have no future
continuation. The query must therefore exclude any candidate occurrence whose
continuation starts at or beyond the current history length.

Lookup starts at the SAM state for the complete indexed history and follows
suffix links from longer to shorter suffix classes. For each state, the
candidate match length is that state's `max_length`, and its representative
occurrence ends at `first_end_position`. A candidate occurrence is valid only
when:

```text
match_start = first_end_position - match_length + 1
match_start >= 0
continuation_start = first_end_position + 1
continuation_start < len(indexed_tokens)
occurrence does not end at the current terminal position
```

If a state has no usable earlier continuation, lookup follows its suffix link.
Suffix-link traversal is sufficient here because every linked state represents
a suffix of the current complete history. The result is the longest suffix
class with a usable earlier representative occurrence, not merely the longest
structural suffix.

### Continuation Choice

For the selected state, use its earliest representative occurrence. This is
deterministic and does not consult future target tokens. It may be less
adaptive than choosing among all occurrences; multi-occurrence ranking is
explicitly deferred until the simple source passes a canonical gate.

The proposal copies at most the selected cap from the indexed stream beginning
at `continuation_start`. It stops at the indexed-stream boundary; proposals
never wrap, synthesize, or repeat tokens beyond observed content.

The proposal record contains:

```text
source = "sam"
match_length
match_start
match_end
continuation_start
available_continuation_tokens
continuation_region
selected_k
draft_tokens
index_token_count
index_state_count
lookup_time_ms
```

`continuation_region` is `"prompt"` when `continuation_start` is before the
prompt boundary and `"generated"` otherwise. A proposal that begins in the
prompt and crosses the boundary remains prompt-sourced; the event separately
records whether its copied span crosses the boundary.

### Minimum Useful Match

Matches shorter than two tokens are treated as no usable match. A one-token
match is expected to be ambiguous enough that the target verify overhead is
unlikely to be justified.

This minimum is fixed for the first gate. The canonical run will not search or
retune it.

## Draft Policies

### Fixed SAM `K=16`

When a usable match exists, draft up to 16 observed continuation tokens. When
there is no usable match or no continuation, skip speculative verification and
perform the normal decode step.

This candidate measures the value and downside of the SAM source without a cap
controller.

### Match-Aware SAM `K in {0,4,8,16}`

The selected cap is a deterministic function of usable match length:

```text
match_length < 2  -> K = 0
2 <= length < 4  -> K = 4
4 <= length < 8  -> K = 8
length >= 8      -> K = 16
```

`K=0` means no proposal and no target speculative verify for that position.
The policy uses only information available before target verification.

The selected cap is a maximum. A continuation shorter than the cap produces a
shorter proposal. Empty proposals are recorded as bypass events, not verify
events.

The first gate deliberately avoids combining match length with an acceptance
EMA. This isolates whether variable-length source quality is useful before
adding another online controller. Acceptance-aware selection is a possible
follow-up only after canonical evidence.

## Profiler Integration

### Draft Source Interface

The profiler will support a SAM draft source alongside n-gram and existing toy
sources. A SAM proposal is normalized to the same draft object contract:

```text
tokens: list[int]
source: str
metadata: dict
```

SAM policies are valid only with:

- `--mode candidate-only`;
- `--temperature 0.0`;
- `--max-num-seqs 1`;
- a SAM draft source;
- profiler-owned execution.

Invalid combinations fail before model loading.

### Decode Data Flow

For every candidate decode position:

1. assert that the SAM indexed stream equals target-verified history;
2. query the longest usable suffix match;
3. select `K` from the configured fixed or match-aware policy;
4. if `K=0` or the proposal is empty, record a bypass and run normal decode;
5. otherwise call unchanged `verify_and_commit_block()`;
6. record actual proposed, accepted, rejected, and target timing values;
7. extend the SAM with every target token committed by the existing path;
8. assert index/history equality again.

SAM construction and lookup remain outside target verify timing. End-to-end
candidate timing includes them.

### No Runtime Mutation

Every SAM event and process summary must record:

```text
runtime_mutation = false
profiler_owned = true
```

The implementation must not edit `LLMEngine.step()`, scheduler policy, public
generation APIs, or persistent `Sequence` fields. A source scan test guards
the intended write scope, and the canonical verifier checks the event fields.

## Metrics

Each process row records at least:

- exact output-token equality with its paired baseline;
- generated token count and end-to-end elapsed time;
- output tokens per second;
- number of normal target decode steps;
- speculative verify attempts;
- target verify elapsed time;
- drafted, accepted, and rejected token counts;
- acceptance rate;
- zero-accept event count and cost;
- bypass count and reason;
- SAM build, extension, and lookup elapsed time;
- SAM state count and indexed token count;
- match-length distribution;
- available-continuation distribution;
- selected-`K` and actual proposal-length distributions;
- per-event source metadata;
- `runtime_mutation=false`.

Derived summaries include:

- median throughput by policy;
- paired per-run throughput ratios;
- paired median SAM speedup versus baseline and fixed n-gram `K=4`;
- verify-attempt and waste reductions versus fixed n-gram `K=4`;
- zero-accept target-cost reduction;
- SAM CPU overhead as a fraction of candidate elapsed time;
- prompt-class medians and worst prompt-class slowdown;
- match-aware `K=0/4/8/16` exercise counts;
- exact-output and trace-integrity rates.

Waste is defined as:

```text
drafted_tokens - accepted_tokens
```

Zero-accept cost is the sum of target verify time for events with
`accepted_count == 0`.

For any candidate/baseline pair with positive finite throughput:

```text
paired_speedup = candidate_tokens_per_s / reference_tokens_per_s - 1
```

The gate uses the median of the 35 paired speedups for each policy comparison
(five prompts times seven repetitions), rather than a ratio of independently
aggregated medians.

For verify attempts, drafted-token waste, and zero-accept cost, the gate first
sums the metric per process, computes:

```text
paired_reduction = 1 - candidate_metric / reference_metric
```

and then takes the median over pairs whose reference metric is positive.
Pairs with a zero reference metric are retained in artifacts but are not
silently assigned a perfect reduction. If no positive-reference pair exists
for a required reduction metric, the decision is `INCOMPLETE`.

Prompt-class regression uses the median paired speedup within that class. SAM
CPU overhead fraction is:

```text
(build_time + extension_time + lookup_time) / candidate_elapsed_time
```

and is diagnostic in v1 rather than a separate pass/fail threshold because it
is already reflected in end-to-end throughput.

## Canonical Experiment

### Policies

Each prompt and repetition runs these five isolated processes:

1. `baseline`: normal greedy decode;
2. `ngram_fixed_k4`: fixed n-gram `K=4`;
3. `ngram_adaptive`: completed adaptive n-gram `K in {1,2,4}`;
4. `sam_fixed_k16`: fixed SAM cap 16;
5. `sam_match_aware`: SAM `K in {0,4,8,16}`.

The n-gram implementations and settings are frozen from the completed adaptive
gate. They are not retuned for this comparison.

### Prompt Bank

The gate uses a committed prompt bank with five classes:

1. natural explanatory prose with low expected repetition;
2. structured checklist or code-like continuation;
3. long repeated prompt structure;
4. transition-heavy text that changes from repetition to prose;
5. prompt-copy/retrieval style text where a later instruction requests an
   earlier span or format.

Prompts, output budgets, names, class labels, and SHA-256 hashes are committed
before the canonical run. The gate must include at least one workload in which
the best usable match comes from generated tokens rather than only the prompt;
the verifier checks source-boundary metadata instead of assuming this occurred.

### Repetitions and Ordering

- Qwen3-0.6B only;
- exact model path:
  `/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B`;
- `temperature=0.0`;
- one sequence per model process;
- one untimed warmup per policy/prompt configuration;
- seven measured repetitions;
- deterministic randomized policy order within each repetition;
- every process receives distinct dynamic `TINYVLLM_DIST_PORT` and
  `MASTER_PORT`;
- transient port collisions may be retried up to three times and retain
  failure provenance;
- all runs execute from an isolated uploaded source directory on
  `sitian@10.232.195.203`;
- remote Python:
  `/data00/home/sitian/sitian-workspace01/tllm/env/bin/python`.

With five prompts, five policies, and seven repetitions, the canonical gate
expects exactly 175 valid process rows.

### Artifacts

The canonical directory contains exactly these top-level evidence files:

```text
manifest.json
raw_rows.json
event_rows.json
summary.json
report.md
```

Per-process logs and profiler JSON may remain in a remote `runs/` directory,
but only the five named files are canonical evidence and sufficient for local
verification.

`manifest.json` binds:

- source commit and dirty state;
- prompt bank and hashes;
- policies and fixed thresholds;
- exact model identifier/path;
- host, Python, and environment;
- deterministic run order;
- expected row count;
- claim scope.

`raw_rows.json` contains one unique process row for every run specification.
`event_rows.json` contains proposal, bypass, verify, commit, and index-integrity
events. `summary.json` contains derived metrics and the machine decision.
`report.md` explains the result and claim boundaries.

## Preregistered Decision Rule

All checks below are conjunctive unless an explicit alternative is stated.

### Completeness and Correctness

The result is `INCOMPLETE` unless:

1. all 175 expected unique rows exist;
2. every process exits successfully after allowed transient retries;
3. every timing and count used by the decision is finite and valid;
4. every candidate output exactly equals its paired greedy baseline;
5. each event trace reconciles with its process totals;
6. every SAM index/history invariant passes;
7. every SAM event records `runtime_mutation=false`;
8. all five canonical artifacts pass local verification and match downloaded
   remote SHA-256 hashes;
9. every required paired comparison has a matching prompt/repetition reference
   row and every required reduction has at least one positive reference value.

Missing rows, malformed output, invalid timing, unreconciled traces, unexpected
port failures, or source/artifact mismatches are not performance failures and
must never be classified as `NO_GO`.

### Primary Performance

`sam_match_aware` must satisfy:

```text
paired median speedup vs baseline >= +10%
```

and one of:

```text
paired median speedup vs ngram_fixed_k4 >= +3%
```

or:

```text
paired median speedup vs ngram_fixed_k4 >= -1%
and verify attempts reduced by >= 25%
and drafted-token waste reduced by >= 25%
```

The fixed SAM result is diagnostic and is not allowed to substitute for the
match-aware candidate in the primary decision.

### Regression Guard

For each critical non-synthetic prompt class:

```text
sam_match_aware median throughput vs baseline >= -5%
```

The critical classes are natural prose, structured/code-like, transition-heavy,
and prompt-copy/retrieval. The high-repeat class does not satisfy this guard on
behalf of another class.

### Policy Exercise

The canonical event set must exercise:

- at least one `K=0` bypass;
- at least one `K=4` proposal;
- at least one `K=8` proposal;
- at least one `K=16` proposal;
- at least one prompt-sourced continuation;
- at least one generated-token-sourced continuation;
- at least one zero-accept verify event;
- at least one fully accepted multi-token proposal.

If any required branch is absent, the result is `INCOMPLETE`, because the gate
did not cover the policy it claims to evaluate.

### Final Classification

- `GO`: all completeness, correctness, performance, regression, and exercise
  requirements pass.
- `NO_GO`: completeness and correctness pass, but at least one preregistered
  performance or regression threshold fails.
- `INCOMPLETE`: any required row, evidence field, correctness check, trace
  reconciliation, policy branch, or artifact check is missing or invalid.

Thresholds may not be relaxed after observing canonical measurements. Any
threshold change requires a new manifest, source commit, and experiment tag.

## Error Handling and Resume

The gate driver writes the manifest before measured runs and writes each
process result atomically. Resume behavior:

- a run key is reusable only if its row matches manifest source, prompt hash,
  policy, repetition, model identifier, and valid completion schema;
- partial or mismatched rows are rerun, never silently reused;
- non-port process failures are retained and make the canonical result
  `INCOMPLETE`;
- port collisions are separately classified and retried with newly allocated
  distinct ports;
- summary generation never drops failed or missing run specifications.

The remote wrapper uploads only the committed source paths required by the
manifest, records the remote source digest, runs in a unique experiment
directory, downloads the five canonical artifacts, verifies hashes locally,
and leaves the original local checkout untouched.

## Testing Strategy

### Pure SAM Unit Tests

Required tests include:

1. empty and one-token histories produce no usable match;
2. exact repeated substrings return the longest usable suffix;
3. terminal-only occurrences are rejected;
4. suffix-link fallback finds a shorter usable occurrence;
5. the earliest representative continuation is selected deterministically;
6. clone construction preserves valid transitions and
   `first_end_position`;
7. suffix-link traversal never returns a state that is not a suffix of the
   current indexed history;
8. proposals stop at the observed stream boundary;
9. rejected draft tokens are never indexed;
10. prompt and generated source-boundary metadata are correct;
11. index/history invariants detect missing or extra tokens;
12. Unicode text is irrelevant after tokenization because matching uses token
    IDs only.

### Policy Unit Tests

Required tests cover every boundary:

```text
length 0/1 -> K=0
length 2/3 -> K=4
length 4/7 -> K=8
length 8+  -> K=16
```

They also verify that selected caps may exceed available continuation length,
while actual proposals cannot.

### Profiler and Verify/Commit Tests

Tests must show:

- SAM policies reject invalid mode, temperature, or batch configurations;
- `K=0` bypass does not call target speculative verification;
- non-empty SAM proposals use unchanged `verify_and_commit_block()`;
- accepted tokens, fallback tokens, EOS, and output budgets keep exact greedy
  semantics;
- accepted speculative tokens crossing block boundaries preserve block/hash
  lifecycle invariants;
- every event includes source metadata and `runtime_mutation=false`;
- per-event counts and timings reconcile with process totals.

### Gate and Artifact Tests

Synthetic gate tests must cover:

- exactly 175 unique valid rows;
- duplicate, missing, failed, and mismatched rows;
- non-finite and non-positive timing rejection;
- prompt/source/commit mismatch rejection;
- output inequality;
- absent policy branches;
- malformed index traces;
- each side of the primary alternative threshold;
- each prompt-class regression;
- `GO`, performance `NO_GO`, and evidence `INCOMPLETE`;
- resumability and port-collision classification;
- canonical five-file hash verification.

### Remote Smoke Before Canonical Run

Run a small isolated Qwen3-0.6B smoke before the canonical gate. It must
exercise at least:

- one `K=0` bypass;
- one non-empty SAM proposal;
- one accepted multi-token commit;
- one zero-accept verify;
- one prompt-sourced and one generated-token-sourced continuation;
- one speculative append crossing a KV block boundary;
- exact baseline/candidate output equality;
- `runtime_mutation=false`.

The smoke validates plumbing only and cannot satisfy the canonical performance
gate.

## Scope Boundaries

### Included

- pure CPU token SAM and match-aware cap policy;
- prompt initialization and verified-token online extension;
- profiler-only single-sequence integration;
- exact greedy correctness checks;
- remote Qwen3-0.6B smoke and canonical gate;
- reproducible artifacts, verifier, README result, and handoff update.

### Excluded

- production scheduler integration;
- multiple candidates in one target forward;
- ragged batch metadata or masks;
- non-greedy speculative sampling;
- cross-request/shared-prefix SAM indexes;
- external retrieval corpora;
- learned continuation ranking;
- native C++/Rust optimization before evidence;
- draft-model checkpoints or training;
- claimed memory savings;
- claims for Qwen3-8B, other GPUs, or production traffic.

## Claim Boundaries

A successful gate would show only that, on the committed Qwen3-0.6B
single-sequence greedy prompt bank, the profiler-owned match-aware SAM reduces
measured decode time under the preregistered thresholds while preserving exact
tokens.

It would not show:

- improved production batch throughput or latency tails;
- correct or efficient ragged batched verification;
- non-greedy distribution preservation;
- lower model-weight or KV-cache memory;
- benefit under concurrent queue pressure;
- generalization to other models, tokenizers, hardware, or domains;
- that the Python SAM is the best production data structure.

If the gate is `GO`, the next decision is whether to optimize the lookup/index
implementation and design a separate batched runtime gate. If it is `NO_GO`,
retain the artifacts and pivot to a higher-quality learned drafter only when a
compatible checkpoint or reproducible training path is available. If it is
`INCOMPLETE`, repair evidence collection without changing performance
thresholds, then resume the same manifest-bound experiment.

## Deliverables After Written-Spec Approval

The implementation plan will map this design to:

- a focused SAM helper module and unit tests;
- profiler draft-source and policy integration;
- a canonical gate driver and synthetic verifier tests;
- an isolated remote runner using the fixed host, Python, and model path;
- smoke and 175-row canonical artifacts;
- README and `AGENT_HANDOFF_STATE.md` updates;
- a prompt-to-artifact completion audit before any final claim.

No implementation begins until this written specification is reviewed and
approved.
