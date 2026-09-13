# Latent KV capacity, Stage 1a GATE A: the Stage 0 decode model is falsified

Run: `experiments/kvcapacity_step_scaling/step-scaling-measure-20260913-191122`
Model: Qwen3-8B on one A100 80GB, `tinyvllm` serving path.
Grid (fully covered, 18/18 cells measured on both paths):

```
8192  : B=1,2,4,8,16,32
16384 : B=1,2,4,8,16
32768 : B=1,2,4,8
40448 : B=1,2,4
```

## Verdict

**FAIL on the eager path, FAIL on the graph path.** The failure is decisive, not a harness
artifact: coverage 18/18, window stability 18/18 within 5%, sample dispersion 18/18 under
25% of the median.

## What the Stage 0 model assumed

```
step_ms(L,B) = c0 + c1 * L * B      c0 = 13.05 ms   c1 = 0.151 us/token
```

## What the serving path actually does (eager, fit on all batches)

```
M1  step_ms = 39.789 + 0.0837e-3 * L*B      R^2 = 0.9543   [FAIL, threshold 0.98]
M2  + 0.1756 ms/seq pure batch term         R^2 = 0.9694   residual -33.0%
M3  + 1.363e-10 * (L*B)^2 curvature         R^2 = 0.9671   quadratic = 15.0% of the step
                                                            at L*B=262144  [FAIL, tol 10%]
```

Equal-product consistency, which M1 requires: 4/5 groups agree within 10%; `L*B=32768`
spreads 12.7% (45.500 / 43.554 / 40.029 ms), i.e. the same resident-token count costs
measurably different amounts depending on how it is split between L and B.

## Three separate ways Stage 0 was wrong

1. **The constant is ~3x off.** Measured `c0 = 39.789 ms` against `13.05 ms` assumed
   (ratio 3.049). The graph path shows why: batch 1 runs a CUDA graph fast path at
   15-18 ms, batch 2 jumps to 43-50 ms (ratio 2.58-2.68 at L=32768/40448). Stage 0's
   `c0` was fit on batch-1 data and therefore describes an execution path that the
   capacity argument itself never uses, since capacity gains only matter at B >= 2.
2. **The slope is ~0.55x off.** Measured `c1 = 0.0837 us/token` against `0.151`
   (ratio 0.554), eager; `0.1054` (ratio 0.698) on the graph path fit at B >= 2.
   Resident KV is *cheaper* per token than Stage 0 assumed.
3. **The functional form is wrong, not just the constants.** There is real curvature
   (15% of the step at the largest cell, eager; 19.8% graph) and a real pure-batch term
   (8.7% eager, 22.1% graph). An affine function of `L*B` alone cannot absorb either.

## Consequence for the latent KV capacity line

- The Stage 0 break-even artifact (`experiments/kvcapacity_stage0/2026-09-13/gate.json`)
  is computed on falsified constants and must not be cited.
- **Do not proceed to GATE B (`phi_probe` / head slicing) on the old numbers.**
- Two directions are honest from here:
  - Refit the cost model as M2 + curvature on the measured multi-batch regime, then re-run
    the Stage 0 break-even gate and see whether the capacity argument survives the larger
    constant. Note the direction of the two dominant errors: a 3x larger `c0` shrinks the
    fraction of the step that KV residency explains, which makes KV compression *less*
    valuable per unit of compression, and the smaller `c1` cuts the same way.
  - Or reframe: at these context lengths the step is dominated by a ~40 ms
    context-independent constant, and resident KV explains roughly 20 ms of a 60 ms step
    only at `L*B = 262144`. Compressing KV cannot beat that constant.

The second reading is the more likely one and should be checked first, because it decides
whether GATE B is worth building at all.
