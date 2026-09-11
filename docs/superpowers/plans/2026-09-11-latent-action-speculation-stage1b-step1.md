# Stage 1b step 1: the prompted drafter is dead, and the n-gram is not

Status: measured on one A100, inference only, no training.
Verdict: **NO-GO for the prompted-drafter line.** The pre-registered gate
was 0 for 30. The best number any prompted model produced anywhere in
the sweep is `p = 0.086`; the *loosest* threshold in the entire grid is
`0.091`, and that threshold assumes the drafter costs nothing at all.
The failure is not a tuning failure and it is not a 0.6B failure: an
8B actor on the same context also fails, and quadrupling the context
does not fix it.
The line is not dead, but the drafter that survives step 0's own
baselines is the training-free n-gram, not a language model.
Line: latent action speculation (`tinyvllm/agentspec/`).
Predecessor: `2026-09-11-latent-action-speculation-stage1b-step0b.md`.

## What was pre-registered

Step 0b fixed the thresholds before any GPU ran, so this document is
allowed to be short about what counts as a pass:

```text
regime        tool_seconds >= 5.0, rho = 0.6, rollback = 0.5s
drafter       prompted Qwen3-0.6B, greedy, no training
context       512-token compressed prompt budget
SWE cap       k = 11 tokens      APIGen cap  k = 27 tokens
required p    0.5304 @1024   0.4032 @4096   0.2064 @16384   (SWE, cap 11)
              0.7488 @1024   0.5765 @4096   0.3308 @16384   (APIGen, cap 27)
coverage      >= 0.60
stop line     prompted drafter at 4096 below 0.4032 -> stop
```

Costs come from the Stage 1a-bis serving-path artifact
`experiments/agentspec_engine_demand/engine-demand-measure-a100-20260910-2258/engine_demand.json`
(payload sha256 `1a9bb5c6...fb32ab`). Nothing about the cost side was
re-measured here; only `p` was unknown.

## How `p` was measured

`tools/agentspec_prompted_drafter_evalset.py` turns each raw trace into
`(prefix, gold next action)` rows. Eligibility reuses the side-effect
classifier from `tools/agentspec_trace_normalize.py`, so ineligible
steps are dropped before anything is scored. The evalsets are not
committed: they embed corpus text.

```text
evalset                        ctx  cap  kept  scanned  ineligible  gold_in_cap
evalset_swe_agent.jsonl        448   11  2500     9712        1192       0.7132
evalset_apigen.jsonl           448   27  1906     5167        1703       0.6684
evalset_swe_agent_ctx3584      3584  11  2500     9712        1192       0.7132
evalset_apigen_ctx3584         3584  27  1906     5167        1703       0.6684
```

`tools/agentspec_prompted_drafter_match_worker.py` runs greedy decoding
under a hard `max_new_tokens` cap, parses the emitted action, and scores
it against the gold action with the same canonical digest the router
would use at commit time (`tinyvllm.agentspec.action`). Three rates are
reported because they are not interchangeable:

- `coverage` — finished inside the cap and parsed.
- `p_speculated` — digest matches divided by steps actually spoken on.
  This is the cost model's `p`.
- `p_effective` — digest matches divided by all eligible steps. It
  charges every abstention as a miss, which overcharges, because an
  abstention never pays rollback. It is a floor, not the truth.

Prompt scaffolding is priced inside the budget, and the context tail is
right-trimmed so the total stays under it. The first smoke run leaked
over (mean 511.8, max 513 against a 512 budget) and was fixed before
any scored run.

## Prompt selection, done on different rows than scoring

Three styles were tried. `v1` simply asked for the next tool call and
the model answered `None` on most SWE rows. `v2` forbids `None`,
apologies, questions and explanations. `v3` adds hand-written format
examples — hand-written specifically so that no eval row leaks into the
prompt. Two variants: `tail` (trace tail only) and `tail_tools` (tail
plus the tool inventory seen in the prefix).

Selection ran on rows `[0, 256)`:

```text
run                        cap  coverage   p_spec    p_eff   tool_acc  declined
swe_agent_tail_v1           11    0.5664   0.0069   0.0039     0.2207        92
swe_agent_tail_v2           11    0.5195   0.0526   0.0273     0.3534        50
swe_agent_tail_v3           11    0.7383   0.0265   0.0195     0.2910         3
swe_agent_tail_tools_v3     11    0.6914   0.0113   0.0078     0.3446         2
apigen_tail_v2              27    0.4102   0.0286   0.0117     0.1905         0
apigen_tail_v3              27    0.4336   0.0541   0.0234     0.1622         0
apigen_tail_tools_v3        27    0.4180   0.0467   0.0195     0.1495         0
```

`v2` for SWE and `v3` for APIGen were selected. Scoring then ran on rows
`[256, 1256)` so that selection and measurement never touch the same
rows. Note that even at selection time nothing was close: the best cell
in this table is `0.054` against a `0.40` stop line. Selection was run
to completion anyway, because stopping a sweep at the first bad number
is how a tuning artifact gets mistaken for a result.

## Result 1: the 0.6B prompted drafter, 512-token context

1000 rows each, offset 256.

```text
run                       cap  coverage   p_spec    p_eff   tool_acc  declined
swe_agent_tail_v2          11    0.4430   0.0316   0.0140     0.0745       245
swe_agent_tail_tools_v2    11    0.5270   0.0133   0.0070     0.1063       180
apigen_tail_v3             27    0.3700   0.0486   0.0180     0.1514         0
apigen_tail_tools_v3       27    0.3610   0.0526   0.0190     0.1440         0
```

Against the pre-registered thresholds (`tools/agentspec_prompted_drafter_verdict.py`):

```text
run                     context  cap  required_p   p_spec  of_req  gate
swe_agent_tail_v2          1024   11      0.5304   0.0316  0.060x  FAIL
swe_agent_tail_v2          4096   11      0.4032   0.0316  0.078x  FAIL
swe_agent_tail_v2         16384   11      0.2064   0.0316  0.153x  FAIL
apigen_tail_tools_v3       1024   27      0.7488   0.0526  0.070x  FAIL
apigen_tail_tools_v3       4096   27      0.5765   0.0526  0.091x  FAIL
apigen_tail_tools_v3      16384   27      0.3308   0.0526  0.159x  FAIL
```

0 of 12. The shortfall is 6x to 40x, not a margin. Coverage also missed
its own 0.60 floor in every scored cell. The only configurations that
cleared 0.60 were the SWE `v3` styles at selection time, and they
cleared it by declining less while matching less — `v3` reached 0.7383
coverage at `p_spec` 0.0265, against `v2`'s 0.5195 coverage at 0.0526.
Coverage and accuracy trade against each other here, and neither side of
the trade is near its threshold.

## Result 2: it is not that 0.6B is too small

The obvious objection is capacity, so the same 512-token context was run
with Qwen3-8B as the drafter — a configuration that could never be
deployed, since an 8B drafter has roughly the actor's own cost, but that
is exactly why it is a clean diagnostic.

```text
run                cap  coverage   p_spec    p_eff   tool_acc
swe_agent_tail_v2   11    0.6914   0.0282   0.0195     0.0678
apigen_tail_v3      27    0.1836   0.1702   0.0312     0.2340
```

An 8B model on SWE is *not better* than the 0.6B (0.0282 vs 0.0316). On
APIGen it is 3.5x better and still 0.51x of the loosest threshold. So
the prompted drafter is not capacity-bound at this context length.

## Result 3: it is not that the context was too short

Second objection: 512 tokens of tail may simply not contain the answer.
So the context was widened to 3584 tokens with a 4096-token prompt
budget, 512 rows each, offset 256.

```text
run                             prompt_tok  coverage   p_spec    p_eff  tool_acc
swe_agent_tail_v2_ctx3584           3211.5    0.4648   0.0042   0.0020    0.0168
swe_agent_tail_tools_v2_ctx3584     3228.4    0.4629   0.0211   0.0098    0.0464
apigen_tail_v3_ctx3584               960.5    0.3086   0.0759   0.0234    0.2532
apigen_tail_tools_v3_ctx3584         968.5    0.3184   0.0859   0.0273    0.2393
```

APIGen improves 1.6x (0.0526 -> 0.0859) — its traces average about 960
tokens, so the 448-token window really was truncating them. SWE gets
*worse*: 0.0316 -> 0.0042 on the `tail` variant. SWE prefixes saturate
both windows (mean 508 at the small budget, 3211 at the large one), so
the extra 3000 tokens are more history, not the missing history, and the
0.6B degrades under them.

More context is also not free. Pricing the 3584-token run with a full
uncompressed drafter prefill raises the requirement rather than lowering
it, so the long-context arm loses on both axes at once.

## The kill that does not depend on cost

All of the above still prices a drafter. The cleanest statement removes
that variable. Set the drafter tax to zero — a drafter that consumes no
GPU at all — and ask what `p` the profitability guard still demands:

```text
tau = 0, tool = 5s, rho = 0.6, rollback = 0.5s
context    minimum p
   1024       0.2896
   4096       0.2108
  16384       0.0909
```

The best `p_speculated` any prompted model produced in this sweep is
`0.0859` (0.6B, APIGen, 3584 context). It does not clear the free
drafter floor at any context length. The 8B's `0.1702` clears only the
16384 floor, and an 8B drafter is not free by a wide margin.

This is the pre-registered stop line, reached with room to spare.
Prompted action speculation is NO-GO. Per step 0b, it does not get
rescued by training a pointer head: the pre-registration said explicitly
that a failure here ends the line rather than escalating it.

## What survives, and it is not a neural network

There is a finding here beyond the NO-GO, and it comes from comparing
this sweep against step 0's own training-free baselines on the same
corpora:

```text
predictor                      apigen   swe_agent   cost
trigram over previous 2         0.2078      0.1129   CPU table lookup
bigram over previous 1          0.2065      0.1111   CPU table lookup
prefix mode                     0.0047      0.1936   CPU table lookup
prompted 0.6B, best cell        0.0859      0.0316   1 prefill + k decodes
prompted 8B, best cell          0.1702      0.0282   actor-scale
```

The prompted 0.6B loses to a bigram counter by 2.4x on APIGen and 3.5x
on SWE, at several orders of magnitude more compute. It also loses on
tool identity alone — 0.07 to 0.25 here against 0.59 to 0.64 for a
tool-level trigram in step 0 — which means the language model is worse
than counting even at the part step 0 called easy.

Against the free-drafter floor, an n-gram drafter is not free but is
close enough to price at `tau ~ 0`:

```text
tau = 0, tool = 5s, rho = 0.6, rollback = 0.5s
context   min_p    trigram_apigen   trigram_swe   prefix_mode_swe
   1024  0.2896            fail          fail            fail
   4096  0.2108            fail          fail            fail
  16384  0.0909            PASS          PASS            PASS
```

At 16384-token actor context and 5-second tools, the training-free
n-gram clears the profitability floor on both corpora. That is one
surviving cell, it is the longest-context and slowest-tool corner, and
the n-gram numbers are themselves conservative: step 0 scored them over
all eligible steps, which is `p_effective`, not `p_speculated`. An
n-gram that abstains when it has no matching history would score higher
on the rate the cost model actually consumes.

## What this changes

- **Drop** the prompted-drafter arm. No prompt engineering follow-up, no
  larger drafter, no longer context. Three independent axes were varied
  and none of them moved the number to within 2x of a floor that assumes
  zero drafter cost.
- **Do not** start the pointer-head training project. It was already
  dropped in step 0b for being disproportionate, and nothing measured
  here argues that a learned head would cross a gap this wide; the
  8B diagnostic is evidence that the gap is not about model capacity.
- **Next measurable question**, if the line continues: re-score the
  step 0 n-gram predictors with an explicit abstention rule so
  `p_speculated` and `coverage` are directly comparable to this sweep,
  and confine the claim to the long-context slow-tool corner. That is
  CPU-only work on artifacts that already exist.
- **Honest framing of the ceiling**: even in the surviving corner the
  predictor is a counter over the agent's own recent history, which is
  a statement about repetitive agent loops, not about latent
  representations. The latent/discrete-code hypothesis that opened this
  line has not been supported by anything measured in Stage 1b.

## Artifacts

```text
tools/agentspec_prompted_drafter_evalset.py
tools/agentspec_prompted_drafter_match_worker.py
tools/agentspec_prompted_drafter_verdict.py
tools/run_agentspec_prompted_drafter_remote.sh

experiments/agentspec_prompted_drafter/prompted-drafter-preflight-20260911-144726/
experiments/agentspec_prompted_drafter/prompted-drafter-select-20260911-145253/
experiments/agentspec_prompted_drafter/prompted-drafter-measure-20260911-145528/   # 512 ctx, 0.6B
experiments/agentspec_prompted_drafter/prompted-drafter-diagnose-20260911-145756/  # 512 ctx, 8B
experiments/agentspec_prompted_drafter/prompted-drafter-measure-20260911-150230/   # 3584 ctx, 0.6B
```

Each measurement directory carries `verdict.json` with the recomputed
requirement and the PASS/FAIL per cell. Only derived statistics are
committed. The evalsets under `.agent_runtime/agentspec_evalsets/` and
the `runner_*.log` files are deliberately left out, because both embed
verbatim corpus text from APIGen-MT-5k (cc-by-nc-4.0) and
SWE-agent-trajectories (cc-by-4.0); each `match_*.json` records the
evalset digest as `evalset_rows_sha256` so the run is still pinned.

Environment: 1x NVIDIA A100 80GB PCIe, torch 2.4.1+cu121,
transformers 4.51.3, Qwen3-0.6B and Qwen3-8B, greedy decoding.
