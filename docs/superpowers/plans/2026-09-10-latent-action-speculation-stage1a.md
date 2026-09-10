# Stage 1a plan: measure the action drafter GPU tax

Status: harness landed, measurement blocked on remote GPU access.
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
  `evidence_valid_for_gate` flag.
- `tools/run_agentspec_drafter_tax_remote.sh` — `preflight | smoke | measure`.
  Uploads the worker, verifies the upload by sha256, runs it under the remote
  CUDA python, pulls back the payload, and refuses a synthetic payload.
- Smoke-tested end to end on CPU with `--synthetic`, which exercises every
  code path and is marked `evidence_valid_for_gate false`.

## Blocker

`preflight` fails closed at connect:

```text
cannot reach sitian@10.232.195.203
  Connection closed by UNKNOWN port 65535
  hint: the jump proxy needs a Kerberos ticket; run kinit
```

`~/.ssh/config` routes the box through `jump-proxy-hl` with
`GSSAPIAuthentication yes`, and `klist` reports no credential cache. The fix
is one interactive `kinit`, which cannot be automated because it prompts for a
password. Everything downstream is ready to run unattended once the ticket
exists.

## Completion criteria

- [x] Worker and runner land, `bash -n` and `py_compile` clean.
- [x] Synthetic CPU smoke run passes and is marked as non-evidence.
- [x] Target load point and NO_GO threshold pre-registered above.
- [ ] `preflight` returns GPU, torch, transformers, and both model paths.
- [ ] `measure` returns a non-synthetic payload for all three context lengths.
- [ ] Measured `tau` compared against the pre-registered threshold, and the
      Stage 0 design doc amended with the result, including the case where the
      result kills the line.

## Stage 1b entry criteria

Stage 1b, which measures top-`b` action prediction accuracy and calibration on
real agent traces, may not begin until Stage 1a reports a measured `tau` at or
below the point B critical tax, or until the design is explicitly narrowed to
the dedicated-capacity regime of point A.
