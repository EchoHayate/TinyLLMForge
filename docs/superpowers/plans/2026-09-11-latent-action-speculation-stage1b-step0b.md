# Stage 1b step 0b: the training was never the requirement

Status: measured, CPU only, no training, no GPU time.
Verdict: **the pointer head is dropped from the critical path.** Step 0
assumed a compact learned output space was mandatory. It is mandatory
only when tool latency is around one second or less, and that regime
caps out at 1.21x with a margin thinner than one prefill of jitter. Once
tool latency reaches five seconds, a *prompted, untrained* drafter
writing plain text keeps 80 to 92 percent of a perfect one-token head's
benefit, and the training project is unnecessary to test the line.
Line: latent action speculation (`tinyvllm/agentspec/`).
Predecessor: `2026-09-11-latent-action-speculation-stage1b-step0.md`.

## The objection that triggered this

Step 0 falsified the fixed 4096-entry code head and proposed replacing
it with a tool classifier plus an argument copy pointer. That is a
training project: a new head, a training set, a loss, a checkpoint, and
an evaluation harness, all before the line's central unknown gets
measured. The objection was that this is disproportionate work, and the
objection was correct. Worse, it was disproportionate work aimed at the
wrong constraint.

Stage 1a-bis priced three arms and the cheapest was a code head. Reading
that as "the drafter must be a head" is a mistake I made when writing
step 0. What Stage 1a-bis actually priced was *few emitted tokens*. A
drafter costs one prefill over its context plus one decode step per
emitted token. The head was merely the limiting case of one token. The
real constraint is a token budget, and nobody had measured how large
that budget is or how long a real action is.

Both are arithmetic over artifacts that already exist.

## What the cost model actually permits

`tools/agentspec_output_token_budget.py` inverts the Stage 0 cost model
over the measured serving-path costs in
`experiments/agentspec_engine_demand/engine-demand-measure-a100-20260910-2258/engine_demand.json`
(payload sha256 `1a9bb5c6...fb32ab`). Nothing is re-measured. The
drafter cost becomes `compressed_prefill + k * drafter_step` and the
question is which `k` still pays.

Break-even `k`, compressed 512-token drafter context, `p=0.75`,
rollback 0.5s:

```text
context     tool.2s   tool1s   tool5s  tool20s
1024              0       11       27       27   rho=0.6
4096              0       11       42       42   rho=0.6
16384             0       10       67      102   rho=0.6
1024              0        0        8       10   rho=0.8
4096              0        0       11       17   rho=0.8
16384             0        0       14       43   rho=0.8
```

Read alone this table is misleading, and reading it alone was my second
mistake. Break-even is the point where the saving reaches zero. It is
the one operating point with no reason to exist. The decision needs the
decay, not the boundary:

```text
latency saving per action, seconds, compressed context, rho=0.6

context  tool_s      k=1      k=2      k=4      k=8     k=11     k=16     k=24
1024     0.2     -0.2409  -0.2712  -0.3344  -0.4714  -0.5849  -0.7984  -1.2221
1024     1.0      0.3591   0.3288   0.2656   0.1286   0.0151  -0.1984  -0.6221
1024     5.0      0.6488   0.6321   0.5974   0.5220   0.4596   0.3422   0.1092
4096     1.0      0.3668   0.3353   0.2705   0.1332   0.0229  -0.1767  -0.5447
4096     5.0      1.1368   1.1195   1.0839   1.0084   0.9477   0.8379   0.6355
16384    1.0      0.3713   0.3346   0.2603   0.1078  -0.0100  -0.2135  -0.5585
16384    5.0      3.3713   3.3346   3.2603   3.1078   2.9900   2.7865   2.4415
```

Three findings, and the middle one reverses the step 0 plan.

**Sub-actor tool latency kills the line outright.** At `tool=0.2s` every
`k` including zero is negative, and the verdict is
`infeasible_no_match_benefit`. A hit cannot save what the actor was
going to spend anyway. No drafter, however cheap or accurate, helps
here. This is a property of the cost model, not of any head.

**At `tool=1.0s` the value lives entirely at `k≈1`.** The saving falls
from 0.359s at one token to 0.015s at eleven, a 96 percent collapse. So
in the fast-tool regime the compact output space genuinely is mandatory,
which is what step 0 concluded. But 0.359s on a 2.23s baseline is a
1.21x ceiling that requires a trained head to reach at all, and the
whole margin is inside one prefill of jitter. This regime is not worth a
training project.

**At `tool>=5s` verbosity is nearly free.** Eight tokens keep 80 percent
of the `k=1` saving at 1024, 89 percent at 4096, 92 percent at 16384.
Sixteen tokens keep 53 / 74 / 83 percent. The absolute saving is larger
here too, because the actor's sojourn is fully hidden behind a longer
tool call. A drafter that writes the action as ordinary text is
admissible, and no head needs to exist.

## What a real action costs to write

`tools/agentspec_action_token_length.py` tokenises every action in both
step 0 corpora with the Qwen3 tokenizer, in three emission forms:
`verbatim` as it appears in the trace, `minimal` as tool name plus
argument values with JSON punctuation stripped, and `tool_only`.

```text
SWE-agent, 31703 eligible actions, Qwen3 tokens
form           mean    p50    p75    p90    p99
verbatim       31.9      8     23     72    414
tool_only       1.3      1      2      2      3

fraction of eligible actions within budget
form            <=8    <=11    <=16    <=27    <=42
verbatim     0.5146  0.6345  0.7039  0.7802  0.8451

APIGen, 16414 eligible actions
form           mean    p50    p75    p90    p99
verbatim       28.1     23     27     36    115
minimal        17.9     14     17     21    107
tool_only       3.5      3      4      6      6

fraction of eligible actions within budget
form            <=8    <=11    <=16    <=27    <=42
verbatim     0.0000  0.0000  0.0138  0.7514  0.9446
minimal      0.0706  0.1197  0.7157  0.9112  0.9544
```

The two corpora are structurally different and the difference decides
the design. SWE-agent actions are shell commands and half of them are
eight tokens or fewer: `python reproduce.py`, `open foo.py`,
`scroll_down`, `ls`. APIGen actions are typed JSON calls and the JSON
envelope alone costs more than the SWE-agent median action. Its long
tail is dominated by `edit` (median 51 tokens) and `think` (median 77),
which step 0 already excluded from the copyable set as authored content.

Crossing this against the decay table:

- SWE-agent, `tool>=5s`, `k=16`: covers 0.704 of eligible actions and
  retains 53 to 83 percent of the ideal saving. Above the 0.60 coverage
  floor pre-registered in step 0.
- APIGen, `tool>=5s`, `k=27`: covers 0.751 verbatim. Also viable, with
  less headroom.
- Either corpus at `tool=1s`: not viable in text at any coverage.

## The revised design, which trains nothing

```text
drafter        Qwen3-0.6B, prompted, greedy, no fine-tuning
context        512-token compressed prefix, as priced in Stage 1a-bis
output         the action as plain text, hard cap k tokens
abstain        if the action has not terminated within k tokens, do not
               speculate; the actor path is unchanged and the cost is
               one prefill plus k steps, already inside the budget
eligibility    unchanged from Stage 0: fail closed on irreversible and
               unclassified tools
regime         tool_seconds >= 5, declared per tool, not assumed
```

The length cap is what makes this work, and it is free. It converts an
unbounded cost risk into a bounded coverage loss, and coverage is
measurable without training anything. The actions the cap refuses are
exactly the long authored-content actions that step 0's copyability
analysis showed a pointer head could not produce either.

The only unmeasured quantity left is the one that was always the point:
the top-1 exact match probability `p` of a prompted, untrained 0.6B
drafter on real trace prefixes. That is one inference sweep on the A100
that is already reachable, with no training loop, no checkpoint and no
new head.

## Pre-registration for the sweep

Pinned from Stage 1a-bis, unchanged:

```text
D                0.4907s @1024, 0.7487s @4096, 2.0385s @16384
rho              0.6
rollback         0.5s
compressed       512 tokens
```

Changed by this document:

```text
regime           tool_seconds >= 5.0 only; the 1.0s reference point is
                 retired as not worth a training project
drafter          prompted Qwen3-0.6B, no training
token cap        k = 11
```

The required `p` must be recomputed, because step 0's 0.478-0.491 was
derived at `tool=1.0s` with a one-token head. At `tool=5s` with `k=11`
the cost model gives:

```text
context   draft tax   required p   coverage of eligible (SWE-agent)
   1024      0.1532       0.5304                             0.6345
   4096      0.1078       0.4032                             0.6345
  16384      0.0464       0.2064                             0.6345
```

The trade is explicit and it is not free. A longer cap buys coverage and
sells margin:

```text
cap   coverage   required p @1024   @4096   @16384
k=8     0.5146             0.4922  0.3719   0.1849
k=11    0.6345             0.5304  0.4032   0.2064
k=16    0.7039             0.5959  0.4563   0.2434
k=27    0.7802             0.7488  0.5765   0.3308
```

`k=11` is the chosen operating point because it is the smallest cap that
clears the 0.60 coverage floor pre-registered in step 0. Note that the
slow-tool regime does *not* lower the bar much at short contexts: 0.5304
at 1024 is harder than step 0's 0.491, because eleven emitted tokens
cost more than a one-token head. What the regime buys is that a text
drafter is admissible at all, and that at 16384 the bar collapses to
0.2064.

NO_GO condition, stated before the measurement:

```text
if a prompted untrained drafter cannot reach 0.4032 at 4096 with branch
width 1 and cap 11, the line stops. It does not get a training project
as a rescue, because step 0 already showed the achievable ceiling is
bounded by argument copyability at 0.53 to 0.76, and a trained head
would be spending real work to chase a bound that a prompt failed to
approach.
```

## What this does not establish

The regime restriction to `tool>=5s` is a genuine narrowing, not a
finesse. It excludes fast structured API agents, which is where the
APIGen corpus lives and plausibly where a lot of production traffic
lives. The honest claim is that action-level speculation pays for slow
tools and does not pay for fast ones.

`p=0.75` is still an assumption inside the cost model, used only to
compute the tables above. The sweep replaces it with a measurement.

Both corpora remain public proxies. Neither is the workload this would
ship against, and the tool latency distribution of a real deployment is
the input that decides whether any of this applies.

## Artifacts

```text
experiments/agentspec_output_budget/2026-09-11/output_token_budget.json
experiments/agentspec_trace_baseline/2026-09-11/token_length_apigen.json
  payload sha256 cbec5b8c9ce41d4bda9619e59e47a08a74ae1d48f128dc80346ce16c5506fc82
experiments/agentspec_trace_baseline/2026-09-11/token_length_swe_agent.json
  payload sha256 fe3f05ec9dc9ce67666d678c99c614037a4dbb06511689bdd6bb9884547ed8f7
```

Reproduce:

```bash
python3 tools/agentspec_output_token_budget.py \
  experiments/agentspec_engine_demand/engine-demand-measure-a100-20260910-2258/engine_demand.json

python3 tools/agentspec_action_token_length.py \
  --corpus swe_agent --input <shard0.parquet> \
  --tokenizer <qwen3 tokenizer.json> --limit 1200
```
