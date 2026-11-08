# Online fixed-development model diagnostic

Date: 2026-11-08. Continues the approved five-strategy design and the user's
request to proceed through completion. This is a development diagnostic, not
training, arbitrary-code execution, or final benchmark admission.

## Decision

Use a persistent remote, inference-only model worker receiving public JSON
messages. The local controller runs the already reviewed fixed CPU probes
online when requested and scores only after sealing decisions.

Alternatives: precomputed tool replies would not measure online verification;
running controller and private scoring data on the model server would weaken
the information boundary. Neither alternative is used.

No additional worktree is needed: continue on the explicitly requested
`experiments/feedback-active-verification` branch. Preserve prior sources and
receipts. No shared runtime modifications, installation, downloads, LoRA,
optimizer, generated-code execution or final-test access.

## Frozen diagnostic

- Four existing task families, one deterministic candidate permutation per task.
- Two initial states and three feedback conditions: 24 paired cells, 120
  scheduled strategy records. Permutations are not independent task samples.
- Strategies: keep, direct, rethink, always_verify, active_verify. Rotate their
  execution order by cell; no warmup calls or outcome-dependent reordering.
- At most 144 model calls, 48 tool calls, 96 fixed probe executions.
- Qwen3-8B existing local model files, pinned hashes checked before loading.
  Existing Python 3.12 / torch 2.5.1+cu124 / Transformers 4.57.1 /
  safetensors 0.8.0 environment; versions rechecked, no installation.
- One previously used A100, UUID pinned in local launch data, idle admission
  requires matching index/UUID, <=100 MiB, 0% utilization, no compute process.
  Never select another card or interrupt someone else's process.
- BF16, eager attention, seed 42, thinking disabled, fresh per-request KV,
  explicit `use_model_defaults=False`, `do_sample=False`, max input 4096,
  max output 32 tokens. Verify effective greedy mode using installed library.
- One GPU launch only after local tests/review and CPU preflight. Whole worker
  timeout 595 seconds plus 5-second kill allowance. Controller admission budget
  600 seconds including loading; cleanup/transport observations reported
  separately rather than claiming an OS-level whole-controller hard deadline.
- No retries of model calls or tool calls. A transport/model failure ends the
  campaign; unattempted records remain missing in scheduled denominators.
  Invalid/truncated outputs terminate that episode and may continue the batch.

## Interfaces and evidence

The real protocol is separate from synthetic replay/scoring. Reuse only neutral
encoding/validation helpers where their contracts match, never change an
evidence label to make synthetic records qualify as real.

Controller sees private task registry. Model receives only specification,
initial source, neutral A/B/C candidates, visible probe inputs, feedback text,
method instructions and prior public responses. No task/family/condition names,
candidate roles, oracle arrays, local paths or private hashes in messages.

The first active decision accepts A/B/C/VERIFY; after one fixed probe only
A/B/C. Rethink uses two decisions with no new facts. Always-verify executes the
probe before its sole model call. A complete EOS is required; no extracting
letters from prose or retrying invalid output.

The local tool callback is bound to the frozen cell and initial program, not
caller-supplied code/input/path. Its public response includes only the two
declared rows and explicit status. Unknown executed-case counts remain unknown
on failed transport. Tool failure still consumes the one opportunity.

Model worker accepts bounded lines with exact `{id, messages}` fields and a
single user message. Accepted lines are at most 131072 UTF-8 bytes. Rejecting
an overlong line may consume one sentinel byte (or one text codepoint, at most
four bytes, for an injected text stream); this is not accepted input.
It returns raw input/output token IDs, EOS status, text,
prompt digest, request digest and nanosecond generation/call timings. It does
not import task modules or read oracle files. Source/model hashes and effective
generation configuration accompany readiness. A duplicate ID, malformed input
or over-budget call is rejected before generation. Shutdown is EOF, not a new
model request.

Local JSONL journals are exclusive, flushed/fsynced before dispatch and after
responses. Freeze source bytes, configuration, complete private schedule and
public initial requests before launch. Preserve raw partial output/stderr on
failure. Remote launch uses an exclusive directory and execution lock; only
worker source and model/launch identity reach that directory. Bind each reply
to the exact request, cell, method and round before accepting it.

## Scoring and acceptance

After decisions are sealed, score against the frozen real CPU audit, not
assumed candidate labels. Validate the source-bound audit and its admission.
Report fixed denominators, protocol failures/missing, repair and regression
separately, deployment fallback separately, and per-family/condition/initial
strata. Active-minus-all-comparator paired deltas use every scheduled pair.

Record input/output tokens, model calls, tool calls/cases, model generation,
model-call and transport roundtrip times, local tool/controller times, episode
walls, loading/shared preparation and PyTorch peak allocation/reservation.
Unknown CPU/GPU consumption, peak total host RSS and energy stay unknown.
No statistical independence is claimed for repeats or within-family cells.

Protocol completion means all 120 records are accounted for with correctly
bound receipts; an unsuccessful selection is still a completed episode.
Development result can be ceiling, failure, worse than baseline or inconclusive.
Never tune and rerun just to obtain a win. No formal effect threshold can be
inferred from four families: `benchmark_eligible=false`,
`formal_evaluation_ready=false`, `net_benefit_established=false`.

Resource hygiene is not a security sandbox. The model's remote host does not
receive local scoring files, but the trusted local fixed worker shares local
OS permissions. `isolated=false` remains. There is no independent human oracle
signoff. If live admission fails, finish local implementation and report the
specific blocker without inventing runtime results.
