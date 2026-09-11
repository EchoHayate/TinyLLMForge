# Stage 1b step 0: is the next action predictable at all?

Status: measured on two public agent corpora, CPU only, no training.
Verdict: **NO-GO for the fixed 4096-entry code head as specified in
Stage 1a-bis.** The line survives only if the drafter representation
changes from a fixed codebook to a copy/pointer head, and only if
authored-content actions are excluded from speculation.
Line: latent action speculation (`tinyvllm/agentspec/`).
Predecessor: `2026-09-10-latent-action-speculation-stage1a-bis.md`.

## Why this ran before any training

Stage 1a-bis proved cost admissibility on the real serving path and
pinned the one surviving question: the compressed-context code drafter
needs a top-1 exact action match probability of 0.491 at 1024, 0.485 at
4096 and 0.478 at 16384. Match is exact by construction, because
`tinyvllm/agentspec/action.py` only permits reusing a speculative
observation when the canonical `tool + arguments` digest is
byte-identical.

Training a head first and measuring accuracy second would have been the
expensive order. Three questions are answerable for the price of a CPU
pass over public traces, and each of them can kill or trivialise the
design before any GPU budget is spent:

1. What does a predictor with no model at all achieve? If a bigram
   already clears 0.485, the contribution is not a learned drafter, it
   is the observation that agent loops repeat.
2. What is the ceiling for a *fixed codebook*? Stage 1a-bis priced a
   4096-entry code head. If evaluation actions are mostly strings that
   never appear in training, no head over that codebook can emit them,
   and the achievable match rate is capped regardless of model quality.
3. How many actions may be speculated on at all? The Stage 0 router
   fails closed on irreversible and unclassified tools, and eligibility
   multiplies the match rate.

## Corpora

Both are public, and neither is the user's production workload. That
limitation is real and is revisited at the end.

```text
Salesforce/APIGen-MT-5k              CC BY-NC 4.0
  apigen-mt_5k.json
  sha256 5225b54198c1d4d2ae9ff14ddd98341751677138efe64ded10dcb770276b5841
  4977 traces, 21955 actions, tau-bench style airline and retail agents
  actions are typed calls: {"name", "arguments"}

nebius/SWE-agent-trajectories        CC BY 4.0
  data/train-00000-of-00012.parquet  (1 of 12 shards)
  sha256 5a395e8c7bb8ddc4b8f4d268506b3a0e2cf9b5ec3922600117322fe788067a13
  6669 traces, 174815 actions, real SWE-agent runs on real repositories
  actions are shell-style command lines, parsed head + command_line
```

Only derived statistics are committed. No corpus text is redistributed,
which also keeps the non-commercial corpus inside its licence.

`tools/agentspec_trace_normalize.py` converts both into the repository's
own action identity and a declared, fail-closed side-effect class.
Evaluation holds out one trace in five, split by trace and never by
step, because a bigram fitted on the trajectory it is scored on is not a
baseline, it is a lookup of the answer.

## Result 1: corpus shape

```text
                              apigen        swe_agent
traces                          4977             6669
actions                        21955           174815
actions per trace, median          4               16
distinct tools                   107              667
distinct exact actions         11765            49920
exact actions seen once         8449            36741
speculation eligible          0.7476           0.9013
```

Two thirds to three quarters of distinct exact actions occur exactly
once in the entire corpus. That is the first warning: the action space
is not a vocabulary, it is an open set.

## Result 2: training-free predictors and codebook ceilings

Exact match, held-out traces, steps with a prefix only. `elig` is the
speculation-eligible subset, `ronly` is read-only tools only.

```text
predictor                     apigen                     swe_agent
                       exact/elig  tool/elig      exact/elig  tool/elig
global_top1                0.0125     0.3593          0.0836     0.3441
repeat_last                0.0095     0.1175          0.0943     0.2751
prefix_mode                0.0047     0.0809          0.1936     0.3795
bigram_prev                0.2065     0.6127          0.1111     0.4369
trigram_prev2              0.2078     0.6446          0.1129     0.5885
oracle_in_own_prefix       0.0241     0.1648          0.4513     0.7476
oracle_in_train_vocab      0.6979     0.9927          0.2355     0.9965
oracle_in_codebook_4096    0.5813     0.9927          0.2052     0.9965
oracle_codebook_or_prefix  0.5990     0.9953          0.5361     0.9997
```

Reading these against the pinned requirement of 0.485 at 4096:

- **No training-free predictor comes close.** The best is 0.208 on
  APIGen and 0.194 on SWE-agent, both roughly 2.4x short. The line is
  not trivially true, which is the good news.
- **The fixed 4096-entry codebook is dead on real coding traces.** Its
  oracle ceiling on SWE-agent is 0.205 against a requirement of 0.478.
  Even a perfect classifier over that codebook fails by 2.3x, because
  only 0.236 of held-out actions were ever seen in training at all.
  Actions carry repository-specific paths and authored file contents;
  they cannot be enumerated in advance. This is a structural result, not
  a capacity result, and widening the codebook does not fix it: the
  gap between codebook 4096 and the *entire* training vocabulary is
  0.205 to 0.236.
- **On typed tool calls the codebook survives, but barely.** APIGen's
  ceiling is 0.581 against 0.485, so a learned head would have to reach
  84 percent of oracle-perfect performance within the codebook just to
  break even. That is not a margin, it is a coin flip on execution.
- **Tool identity is easy and arguments are the whole problem.** A
  trigram predicts the tool name at 0.59 to 0.64 with 0.99 coverage,
  while exact match collapses to 0.11 to 0.21. Every bit of difficulty
  lives in the arguments.
- **Trace-local repetition is the strongest free signal on real coding
  traces.** 0.451 of eligible SWE-agent actions appeared verbatim
  earlier in the same trajectory, and the union of codebook and copy
  reaches 0.536. That points at the representation the data actually
  supports.

## Result 3: can the arguments be copied from context?

`tools/agentspec_trace_copyability.py` asks the direct question. An
action is copyable when every scalar argument value already appears
verbatim in the text the drafter would have read: system prompt, user
turns, previous actions and their observations. It is an oracle over a
pointer head, not a method.

```text
                                apigen      swe_agent (1200 traces)
eligible steps scanned           11493                       30506
copyable from context           0.6342                      0.5289
argument-free                   0.0129                      0.0585
```

Per tool, the residue is concentrated and legible:

```text
apigen                          swe_agent
get_reservation_details 0.823   scroll_down   0.999
get_product_details     0.778   goto          0.980
get_user_details        0.739   python        0.961
get_order_details       0.582   open          0.746
search_direct_flight    0.559   search_dir    0.624
calculate               0.006   ls            0.619
think                   0.000   create        0.303
                                edit          0.000
```

`edit` is 28.8 percent of eligible SWE-agent steps and is copyable
0.0001 of the time. `think` and `calculate` are the same phenomenon on
APIGen. These are actions whose arguments are *authored* by the model:
a patch body, a chain of thought, an arithmetic expression. They cannot
be copied and they cannot be enumerated, so they bound every mechanism
in this family, not just this one.

Excluding authored-content actions and re-measuring the remainder:

```text
corpus      subset                     share of eligible   copy oracle   headroom
apigen      minus think, calculate                0.9172        0.6914      1.45x
swe_agent   minus edit, create, echo              0.6939        0.7558      1.58x
```

## Verdict

**The Stage 1a-bis design does not survive contact with the data, and
one narrower design does.**

- NO-GO, as specified: a fixed 4096-entry discrete code head predicting
  exact actions. Ceiling 0.205 on real coding traces against a 0.478
  requirement. This is falsified, not underpowered.
- NO-GO for training that head on typed tool calls as the primary bet:
  0.581 ceiling against 0.485 is 1.20x of oracle headroom, and the
  learned head would have to be near-oracle to pay for itself.
- CONDITIONAL GO for a revised design: predict the tool with a cheap
  classifier, copy the arguments from context with a pointer head, and
  refuse to speculate whenever the predicted tool is one whose arguments
  are authored rather than copied. Oracle headroom becomes 1.45x to
  1.58x on the remaining 69 to 92 percent of eligible steps.

The refusal rule is what makes this coherent rather than a retreat.
Tool identity is the easy prediction, 0.59 to 0.64 training-free, so the
drafter can decide *whether to speculate* far more reliably than it can
decide *what to speculate*, and Stage 0's router is already fail-closed.
Refusing costs the baseline path, which is exactly what happens today.

The cost side is unaffected: Stage 1a-bis measured the compressed-context
arm at 0.0166 to 0.0691 tax with a 32-microsecond head, and a pointer
head over the same compressed context is the same order of cost. What
changes is the head's output space, not its price.

## What this does not show

- **The copy oracle is an oracle.** It assumes the head knows which
  spans to copy. A real pointer head over a 512-token compressed context
  will be well below 0.69 and 0.76, and the compression itself may
  discard the very spans that must be copied. That interaction is
  unmeasured and it is the next thing to measure.
- **Neither corpus is the target workload.** APIGen is synthetic and
  verified, so its argument distribution is cleaner than production.
  The SWE-agent shard is one of twelve and was produced by a specific
  scaffold with older models, and 6.7 percent of its actions fall into
  the fail-closed `unknown` class.
- **First actions are excluded.** Every predictor is scored only where a
  prefix exists, which favours the hypothesis. The excluded counts are
  in the artifacts.
- **Substring copyability is strict and is a lower bound.** An
  argument reformatted between observation and call, `#W123` against
  `W123`, is scored as not copyable.
- **Eligibility is a declared table, not a verified property.** The
  side-effect classes come from prefix rules in the normaliser. They
  fail closed, but they have not been checked against the tools'
  actual semantics.

## Pre-registered revision for Stage 1b proper

The Stage 1a-bis NO_GO condition said the line stops if the measured
match at 4096 falls below 0.485 with branch width 1, and that it must
not be rescued by widening the branch. That condition is honoured: the
fixed-codebook head is stopped, and the branch is not widened.

The replacement is registered here, before any training:

```text
head              tool classifier + argument copy pointer over the
                  512-token compressed context
speculate only    when predicted tool is in a declared copyable set
declared set      apigen:    all eligible tools except think, calculate
                  swe_agent: all eligible heads except edit, create, echo
required p        unchanged: 0.491 @1024, 0.485 @4096, 0.478 @16384
                  measured over speculated steps only
required coverage speculated steps >= 0.60 of eligible steps, otherwise
                  the mechanism is too narrow to matter regardless of p
NO_GO             measured p below required at 4096, or coverage below
                  0.60, with branch width 1
```

The coverage floor is new and it exists to stop the obvious cheat. A
head that speculates only on `scroll_down` would score a high match rate
and accelerate nothing.

## Reproduction

```bash
python3 tools/agentspec_trace_normalize.py --corpus apigen \
    --input apigen-mt_5k.json --output apigen_actions.jsonl
python3 tools/agentspec_trace_match_baseline.py apigen_actions.jsonl \
    --output baseline_apigen.json
python3 tools/agentspec_trace_copyability.py --corpus apigen \
    --input apigen-mt_5k.json --output copyability_apigen.json
```

```text
baseline_apigen.json       sha256 197172d3514cf31d14e05af8183329eacc187f31a0fa4d99c137d7ab18ba5e7e
baseline_swe_agent.json    sha256 ed8bdc1816e92710cf3bf59535ec6dc29e7762635961920df706a8baec330042
copyability_apigen.json    sha256 61281efc40cd552c7d6e3517b6f69a6e9d75d8008ff1420914e50c0228226d45
copyability_swe_agent.json sha256 37e472c586ffd34ebe85a58671e1a201a0fc8ee0b9c480ff4f9d4b02dcc8aec5
```

## Completion criteria

- [x] Public traces normalised into the repository's own action identity.
- [x] Held-out split by trace, not by step.
- [x] Training-free predictors measured, so the line cannot claim credit
      for what a bigram already does.
- [x] Fixed-codebook ceiling measured and the specified design falsified.
- [x] Copy/pointer ceiling measured, including the authored-content
      residue that bounds it.
- [x] Revised head, coverage floor and NO_GO condition pre-registered.
- [ ] Pointer head measured against a real 512-token compressed context.
