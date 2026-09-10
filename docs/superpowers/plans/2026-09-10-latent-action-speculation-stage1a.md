# Stage 1a plan and results: measure the action drafter GPU tax

Status: measured on 1x A100 80GB. Conditional GO, superseded on the `D`
question by `2026-09-10-latent-action-speculation-stage1a-bis.md`, which
re-measures `D` on the serving path. The `tau` values in this document
come from an eager Hugging Face harness and are not the load point for
Stage 1b.
Line: latent action speculation (`tinyvllm/agentspec/`).
Predecessor: `2026-09-10-latent-action-speculation-stage0.md`.

## Why cost before accuracy

Stage 0 priced action-level speculation over two declared inputs: the actor
GPU demand `D` per agent step and the drafter tax `tau = G_draft / D`. Of the
270 frozen operating points, only 76 were `net_positive`, and the verdict was
far more sensitive to `tau` than to the match probability `p`. That ordering
decides the work order. Measuring `p` first would require agent traces, an
argument codebook, and a trained head, and all of that spend is wasted if the
drafter cannot be made cheap enough to be admissible at the target load.

Stage 1a therefore measures cost only. The code head is randomly initialised
on purpose: head weights move accuracy, not cost. Nothing in Stage 1a is
allowed to claim a match probability or a speedup.

## Pre-registered load point and NO_GO threshold

Stage 0 entry criteria require the target load point to be declared before
measurement, so it is fixed here and computed from the frozen cost model at
this revision.

```text
declared point B    rho = 0.6, tool_seconds = 1.0, rollback_seconds = 0.5
actor demand        D   = 0.080 s of GPU per agent step
stability bound     tau < 0.6667          capacity is unstable above this
critical_draft_tax  tau < 0.3059          at p = 0.90
minimum p           p >= 0.7600 at tau = 0.10
                    p >= 0.8537 at tau = 0.25
```

Two reference points bracket it:

```text
point A  rho = 0.3, tool = 0.2 s, rollback = 0.0    critical tau 0.385 .. 0.630
point C  rho = 0.8, tool = 5.0 s, rollback = 0.5    critical tau 0.117 .. 0.172
```

Pre-registered decision rule:

- `tau_measured > 0.3059` at point B, and Stage 1a is `NO_GO` for the shared
  serving story. The line may continue only against the dedicated-capacity
  story of point A, and the design doc must be amended to say so.
- `tau_measured <= 0.1722`, and the line also survives the heavy-load point C,
  which is the regime the design claims to target.
- A text drafter at `tau = 1.0` is already `unstable_capacity` at point B. If
  the measurement shows the text drafter is materially cheaper than that, the
  Stage 0 tax assumptions were wrong and the matrix must be re-run before any
  Stage 1b work starts.

## Arms

`tools/agentspec_drafter_tax_worker.py` measures four arms per agent step, at
context lengths 1024, 4096, and 16384, with 32 action tokens:

| arm | what it runs | what it tests |
| --- | --- | --- |
| `actor` | target prefill plus 32 greedy decode steps | denominator `D` |
| `text_drafter` | small-model prefill plus the same 32 decode steps | the published design |
| `code_drafter` | small-model prefill plus one linear head over a 4096-entry action code vocabulary | does dropping token decode actually move `tau` |
| `code_drafter_ckv` | same head over a 512-token compressed context | does drafter cost decouple from trajectory length |

The third and fourth arms are the two claims of this line. If `code_drafter`
does not beat `text_drafter` by a wide margin, the latent/discrete-code
framing has no cost advantage and the honest move is to retire it and use a
small text drafter. If `code_drafter_ckv` does not flatten the growth in
context length, the KV-compression half of the design is decoration.

## Harness state

- `tools/agentspec_drafter_tax_worker.py` — self-contained worker, CUDA-event
  timing with a wall-clock fallback, median over repetitions after warmup,
  deterministic JSON payload with `payload_sha256`, and an explicit
  `evidence_valid_for_gate` flag. The requested dtype is asserted against the
  dtype that actually loaded, because `transformers` 5 renamed `torch_dtype`
  to `dtype` and silently forwards the wrong keyword to the config, which
  would have produced float32 weights and a payload of wrong numbers.
- `tools/run_agentspec_drafter_tax_remote.sh` — `preflight | smoke | measure`.
  Uploads the worker, verifies the upload by sha256, runs it under the remote
  CUDA python, pulls back the payload, and refuses a synthetic payload.
- Smoke-tested end to end on CPU with `--synthetic`, which exercises every
  code path and is marked `evidence_valid_for_gate false`.

### Remote environment, and why it needed work

Neither remote interpreter worked as shipped. The venv site-packages carries
`transformers` 5.8.1 against its own `torch` 2.4.1, which dies at import
inside `torch.library.custom_op`. The user site carries `transformers` 4.51.3,
the right vintage, but sits next to a flash-attn wheel built against a
different torch C++ ABI, so importing any Qwen3 model pulls in an `.so` with
undefined `c10` symbols. The runner therefore builds a per-run symlink farm
over the user site with `flash_attn` and `torchvision` filtered out, puts it
on `PYTHONPATH` ahead of the venv, and disables user site. `torch` and `numpy`
still come from the venv. Nothing in the shared environment is mutated.

The GSSAPI ticket lives in a FILE credential cache at `~/krb5cc_sitian` that
the other local agents share. The macOS default API cache is per security
session and reads as empty from a non-interactive session even while a valid
ticket exists, which is what produced the earlier false "no ticket" reading.
The runner now selects the FILE cache explicitly.

## Measured result

One A100 80GB PCIe, GPU 7 idle, actor Qwen3-8B, drafter Qwen3-0.6B, bf16,
32 action tokens, 5 repetitions after 2 warmups, `torch` 2.4.1+cu121,
`transformers` 4.51.3. Payload
`de18005bcdc00d0415d6998f43548f59fb52abf88aa8371e44bf2e9392e55805`.

```text
context   actor_s   tau_text   tau_code   tau_code_ckv
   1024    1.1962     0.7275     0.0239         0.0231
   4096    1.4772     0.6137     0.0509         0.0176
  16384    3.4196     0.3554     0.1344         0.0085
```

Both design claims survive, and the second one turns out to be load bearing
rather than decorative.

1. The code drafter is 30x, 12x, and 2.6x cheaper than the text drafter at the
   three context lengths. Dropping token decode is where the cost goes.
2. Only the compressed-context arm decouples from trajectory length. Its tax
   *falls* from 0.0231 to 0.0085 as context grows 16x, because actor cost
   grows while the drafter stays pinned to a 512-token budget. The plain code
   drafter moves the other way, 0.0239 to 0.1344, because its prefill tracks
   the trajectory. KV compression is not an optimization for this design, it
   is the only thing that keeps the drafter admissible at long context.

## The pre-registered threshold was wrong, and the model caught it

The rule declared above was `tau > 0.3059 is NO_GO`. Every code-drafter number
clears it, so the naive reading is a clean pass. That reading is wrong.

`0.3059` was computed at a *declared* actor demand of `D = 0.080 s`. Measured
`D` is 1.20 to 3.42 s, 15x to 43x larger. The threshold is a function of `D`,
so it has to be recomputed:

```text
context   measured D   critical tau at p=0.90 and rollback 0.5 s
   1024      1.196 s   0.0971
   4096      1.477 s   0.0809
  16384      3.420 s   none: no tax is admissible at this point
```

Re-running the Stage 0 cost model at measured `D`, `rho = 0.6`,
`tool = 1.0 s`, `rollback = 0.5 s`, `p = 0.90`:

```text
context   text_drafter                  code_drafter                  code_drafter_ckv
   1024   unstable_capacity             net_positive                  net_positive
   4096   infeasible_no_match_benefit   net_positive                  net_positive
  16384   infeasible_no_match_benefit   infeasible_no_match_benefit   net_positive
```

So the pre-registration was under-specified: it pinned a threshold on `tau`
while letting `D` float, and `D` is what moved. This is the third place in
this line where the plan was wrong and the model was right, after the unit-tax
headroom error and the tool-latency speedup error in Stage 0. The rule for
Stage 1b is amended: a load point must pin `(D, rho, tool_seconds,
rollback_seconds)`, not just the last three.

## Threat to validity, stated against my own conclusion

The decode loop is an eager Python loop, so every step pays a fixed launch and
dispatch cost that is charged to the 0.6B and the 8B alike:

```text
context   actor step   drafter step   ratio
   1024     34.05 ms       26.18 ms   0.7688
   4096     33.39 ms       25.97 ms   0.7776
  16384     39.67 ms       23.66 ms   0.5963
```

A 0.6B decode step should be roughly an order of magnitude cheaper than an 8B
step. Measuring 0.60 to 0.78 means these steps are overhead bound, not compute
bound, and roughly 24 ms of each step is harness tax. That inflates `D` and it
inflates `tau_text`.

Both errors point the same way, in favour of the hypothesis under test, so the
sensitivity has to be published. Subtracting an assumed fixed per-step
overhead `o`:

```text
context   o = 0 ms                  o = 20 ms                 o = max
   1024   text .728 code .024       text .414 code .051       text .091 code .080
   4096   text .614 code .051       text .318 code .090       text .117 code .116
  16384   text .355 code .134       text .207 code .165       text .172 code .173
```

At the extreme the code drafter's advantage over the text drafter disappears
entirely. The compressed arm is the robust one: it stays at 0.011 to 0.077
across the whole sweep, an order of magnitude under the text drafter
everywhere. So the defensible claim from Stage 1a is narrower than the raw
table suggests:

- Supported: a compressed-context code drafter is cheap enough to be
  admissible, and its cost does not grow with the trajectory.
- Not supported yet: that a plain code drafter beats a text drafter. On a
  serving engine that does not pay 24 ms of per-step overhead, that gap may
  close.

## Verdict

Conditional GO, narrowed to the compressed-context arm.

- `code_drafter_ckv` is `net_positive` at every measured context length and is
  the only arm that survives at 16384. Proceed.
- `code_drafter` without compression is `infeasible` at 16384. Do not carry it
  forward as the primary design.
- `text_drafter` fails at every context length. Stage 0 assumed `tau = 1.0`;
  measured is 0.36 to 0.73, so Stage 0 was pessimistic about the magnitude and
  right about the conclusion.
- Blocking follow-up before Stage 1b: re-measure `D` on a serving path rather
  than an eager HF loop. Every threshold in this document is a function of
  `D`, and the current `D` carries roughly 0.77 s of harness overhead.

## Completion criteria

- [x] Worker and runner land, `bash -n` and `py_compile` clean.
- [x] Synthetic CPU smoke run passes and is marked as non-evidence.
- [x] Target load point and NO_GO threshold pre-registered above.
- [x] `preflight` returns GPU, torch, transformers, and both model paths.
- [x] `measure` returns a non-synthetic payload for all three context lengths.
- [x] Measured `tau` compared against the pre-registered threshold, the
      threshold corrected at measured `D`, and the result recorded including
      the arm it kills.
- [x] `D` re-measured on a serving path, and the thresholds recomputed.
      Done in `2026-09-10-latent-action-speculation-stage1a-bis.md`: the
      serving-path `D` is 1.4x to 2.8x smaller, the conclusions on the
      text and compressed arms hold, and the plain code drafter is
      revived at 1024 and 4096 only.

## Stage 1b entry criteria

Stage 1b, which measures top-`b` action prediction accuracy and calibration on
real agent traces, may not begin until `D` is re-measured on a serving path
and the compressed-context arm is still `net_positive` at the recomputed
threshold. Both conditions were met in Stage 1a-bis, and the pinned load
point for Stage 1b lives in that document, not this one. The load point for Stage 1b must pin `(D, rho, tool_seconds,
rollback_seconds)` in advance. Stage 1b measures accuracy for the
compressed-context arm only; the plain code drafter is out of scope unless the
serving-path measurement revives it.

