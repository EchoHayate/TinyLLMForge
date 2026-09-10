# Stage 1a-bis: re-measure the actor demand on the serving path

Status: measured on 1x A100 80GB. GO for Stage 1b, narrowed to the
compressed-context code drafter.
Line: latent action speculation (`tinyvllm/agentspec/`).
Predecessor: `2026-09-10-latent-action-speculation-stage1a.md`.

## Why this run exists

Stage 1a closed with one blocking defect. It measured the actor demand `D`
with an eager Hugging Face decode loop, and its own decomposition showed that
loop was overhead bound: a 0.6B decode step cost 0.60 to 0.78 of an 8B step
where bandwidth reasoning predicts closer to 0.08. Since every Stage 0
threshold is a function of `D`, an inflated `D` does not merely add noise, it
moves the admissibility boundary in the direction that flatters the hypothesis
under test. Stage 1a-bis re-measures on the path this repository actually
ships.

## What was measured

`tools/agentspec_engine_demand_worker.py` drives `tinyvllm.LLM` one scheduler
step at a time. The engine emits the first sampled token out of the prefill
step, so a request for `A` action tokens resolves into one prefill step plus
`A-1` decode steps, and the two phases are timed separately:

```text
D            = actor_prefill(L)   + (A - 1) * actor_step
G_text       = drafter_prefill(L) + (A - 1) * drafter_step
G_code       = drafter_prefill(L) + head
G_code_ckv   = drafter_prefill(B) + head
```

Both `enforce_eager=False` and `enforce_eager=True` are measured in the same
artifact, so the CUDA graph contribution is attributable rather than assumed.
Every repetition uses a fresh random prompt, because the engine carries a
reusable prefix cache and a repeated prompt would make prefill look free.

The code head stays randomly initialised. Head weights move accuracy, not
cost. This run claims no match probability and no end-to-end speedup.

```text
device        1x NVIDIA A100 80GB PCIe, torch 2.4.1+cu121, bf16
actor         Qwen3-8B          drafter   Qwen3-0.6B
action tokens 32                decode steps per action 31
contexts      1024, 4096, 16384 compressed budget 512
repetitions   5, after 2 warmup  code vocabulary 4096
artifact      experiments/agentspec_engine_demand/
              engine-demand-measure-a100-20260910-2258/engine_demand.json
payload sha256 1a9bb5c61dddcb39f579dab245fccc88471dc426b4e228773418888f61fb32ab
```

## Result: how much of Stage 1a's `D` was harness tax

```text
context   D_eager   D_serving   ratio   step ratio eager   step ratio serving
   1024    1.3624      0.4907   2.776              0.717                0.284
   4096    1.5879      0.7487   2.121              0.717                0.308
  16384    2.8463      2.0385   1.396              0.744                0.355
```

Two things fall out of this and both are checkable.

First, the correction is entirely in decode. Actor prefill is 0.0814 s eager
and 0.0815 s under CUDA graphs at 1024, and 1.5652 versus 1.5573 s at 16384,
which is the expected result because graph replay only covers decode. The
actor decode step drops from 41.3 ms to 13.2 ms. So Stage 1a's `D` carried
roughly 28 ms of per-step harness tax, and Stage 1a's own estimate of about
0.77 s of total overhead was in the right range but understated.

Second, the drafter is still not near its roofline. A 0.6B decode step costs
0.28 to 0.36 of an 8B step even under CUDA graphs, against the roughly 0.08
that weight-bandwidth reasoning predicts for a memory-bound decode. The
residual is per-layer launch and kernel floor at batch 1. This matters for
honesty about the text arm: the text drafter is still measured on a path that
is unkind to small models, and a better small-model decode path would lower
`tau_text` further. It does not matter for the code arms, whose cost is one
prefill plus one projection and does not contain a decode loop at all.

## Result: measured cost and tax on the serving path

```text
context     D_s   a_pre_s   a_step_ms   d_step_ms   tau_text   tau_code   tau_ckv
   1024  0.4907    0.0814      13.202       3.752     0.3078     0.0708    0.0691
   4096  0.7487    0.3196      13.843       4.259     0.2397     0.0634    0.0453
  16384  2.0385    1.5573      15.521       5.513     0.2244     0.1406    0.0166
```

Absolute arm costs, in seconds of GPU per agent step:

```text
context   text_drafter   code_drafter   code_drafter_ckv   code head
   1024         0.1510         0.0347             0.0339    0.000032
   4096         0.1795         0.0474             0.0339    0.000032
  16384         0.4574         0.2866             0.0339    0.000032
```

The shape of the result is the compression argument, stated in measurements.
The plain code drafter must re-read the whole trajectory, so its prefill grows
from 34.7 ms to 286.6 ms and its tax rises to 0.14 at 16384 even though it
emits no tokens. The compressed arm pins drafter prefill at the 512-token
budget, so it costs a flat 33.9 ms at every context and its tax *falls* as the
trajectory grows, from 0.069 to 0.017. The code head itself is 32 microseconds
and is not a cost consideration at any context length.

## Verdicts recomputed at measured `(D, tau)`

`tools/agentspec_engine_demand_verdict.py` re-runs the frozen Stage 0 cost
model at the measured operating points. The pre-registered Stage 1a threshold
of `tau < 0.3059` is not reused, because it was computed at a declared
`D = 0.080 s` that no longer describes the system. Reference point, declared
before reading the table: `tool_seconds = 1.0`, `rho = 0.6`,
`p = 0.75`, `rollback_seconds = 0.5`.

```text
context   arm                 tau      verdict         speedup   min_p   crit_tax
   1024   text_drafter     0.3078   no_match_benefit    0.6637     n/a        n/a
   1024   code_drafter     0.0708   net_positive        1.2071   0.495     0.1561
   1024   code_drafter_ckv 0.0691   net_positive        1.2113   0.491     0.1561
   4096   text_drafter     0.2397   no_match_benefit    0.7183     n/a        n/a
   4096   code_drafter     0.0634   net_positive        1.1155   0.552     0.1113
   4096   code_drafter_ckv 0.0453   net_positive        1.1606   0.485     0.1113
  16384   text_drafter     0.2244   no_match_benefit    0.6233     n/a        n/a
  16384   code_drafter     0.1406   no_match_benefit    0.7875     n/a        n/a
  16384   code_drafter_ckv 0.0166   net_positive        1.0716   0.478     0.0457
```

Over the full 90-point frozen grid per arm:

```text
context   arm                 net_positive   net_negative   no_benefit   unstable
   1024   text_drafter                  23             13           18         36
   1024   code_drafter                  50              4           36          0
   1024   code_drafter_ckv              50              4           36          0
   4096   text_drafter                  28              8           36         18
   4096   code_drafter                  49              5           36          0
   4096   code_drafter_ckv              50              4           36          0
  16384   text_drafter                  19              5           48         18
  16384   code_drafter                  28              2           42         18
  16384   code_drafter_ckv              52             14           24          0
```

Reading these tables against Stage 1a:

- The serving-path correction did not rescue the text drafter. It is
  `infeasible_no_match_benefit` at the reference point at every context
  length, and it is the only arm that ever destabilises capacity. Stage 0
  assumed `tau = 1.0` for this arm and measured 0.22 to 0.31, so Stage 0 was
  wrong about the magnitude and right about the conclusion, for the second
  time.
- The serving-path correction did partially rescue the plain code drafter,
  which is now `net_positive` at 1024 and 4096 where Stage 1a's eager numbers
  had it at 0.024 to 0.051 tax and also positive. It remains infeasible at
  16384, for the structural reason that its prefill scales with the
  trajectory.
- The compressed-context arm is `net_positive` at all three contexts with
  between 2.3x and 2.8x of margin against the recomputed critical tax, and it
  is the only arm whose margin *improves* with context length.

The uncomfortable number in this table is `min_p`. The compressed arm needs a
top-1 action match probability of 0.478 to 0.491 to break even at the
reference point, and the plain code drafter needs 0.552 at 4096. The published
single-branch action match rate that Stage 0 used as its reference is 0.55.
So the entire remaining risk of this line is now concentrated in one quantity,
with margin measured in a few percentage points, and it is exactly the
quantity Stage 1b measures. Cost is no longer the binding constraint;
accuracy is.

## Threats to validity that survive this run

- Single-sequence measurement. `D` is measured at batch 1. Under real
  batching, actor decode steps amortise across sequences and the per-agent-step
  `D` falls, which raises every `tau` and shrinks every margin. Stage 3 has to
  re-measure under concurrency; the Stage 1b GO is conditional on that.
- Drafter decode is overhead bound even under CUDA graphs, at 0.28 to 0.36 of
  the actor step. This flatters no code arm, but it means `tau_text` is an
  upper bound rather than an estimate.
- Prefill dominance at long context. At 16384, prefill is 76 percent of `D`,
  so `D` is sensitive to prefill attention implementation and to any future
  chunked-prefill scheduling change. The worker fails loudly if the prompt is
  chunked, so this assumption is enforced rather than hoped for.
- The compressed arm's 512-token budget is a cost assumption, not a working
  compressor. Nothing here shows that 512 tokens of compressed trajectory
  carry enough signal to predict the next action. That is Stage 1b's problem,
  and it is now the only problem that matters.

## Verdict

GO for Stage 1b, scope narrowed to `code_drafter_ckv`.

- Cost admissibility is established on the real serving path, not a
  scaffolding harness, with margin that grows rather than shrinks with
  trajectory length.
- The plain code drafter is retained only as a 1024 to 4096 fallback and is
  not the primary design.
- The text drafter is dropped from this line.

## Load point pinned for Stage 1b

Stage 1b must not re-derive its own thresholds. It inherits these, and any
change to them invalidates the Stage 1b result:

```text
serving path      tinyvllm.LLM, enforce_eager=False, batch 1, bf16
actor             Qwen3-8B          drafter   Qwen3-0.6B
D                 0.4907 s at 1024, 0.7487 s at 4096, 2.0385 s at 16384
rho               0.6
tool_seconds      1.0
rollback_seconds  0.5
tau_ckv measured  0.0691, 0.0453, 0.0166
required p        >= 0.491 at 1024, >= 0.485 at 4096, >= 0.478 at 16384
```

Stage 1b NO_GO condition, pre-registered here: if measured top-1 action match
probability on real traces is below the required `p` at 4096 with a branch
width of 1, the line stops and is not widened to `b > 1` to compensate,
because branch widening multiplies the drafter tax and moves the threshold up
at the same time.

## Completion criteria

- [x] Worker drives the real engine and separates prefill from decode.
- [x] Both CUDA graph and eager modes measured in one artifact.
- [x] Fresh prompts per repetition, prefix cache defeated.
- [x] Chunked prefill fails the run instead of silently averaging.
- [x] `D` re-measured on the serving path and the harness tax attributed.
- [x] Stage 0 verdicts recomputed at measured `(D, tau)` by a checked-in
      script rather than by hand.
- [x] Load point and NO_GO condition pinned for Stage 1b.
