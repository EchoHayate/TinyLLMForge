# Attention-Output Selective Compression Policy

Source run: `runs/am_qwen3_0_6b_s1536_b64_128_256_r1.0`
Source decision: **AM_WEAK**

best holdout highest fitV mean R2=0.207, coverage@0.5=4.46%

Regime: `holdout`; selector: `highest`; metric: `fitv_val_r2`; layer filter: `0`..`-1`.

constrained-head-quality-sweep

| R2 threshold | Compressed Heads | Head Fraction | Cache Entry Fraction | Cache Saving | Mean Compressed R2 | P10/P50/P90 R2 | Layers |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0.35 | 16 / 224 | 7.14% | 93.53% | 6.47% | 0.4488 | 0.407/0.448/0.495 | 1;2;3;6;8;10;11;16;18;19;21;22;23;24;25;26 |
| 0.40 | 16 / 224 | 7.14% | 93.60% | 6.40% | 0.4729 | 0.449/0.473/0.495 | 2;3;4;5;6;9;10;11;13;14;18;20;21;22;25;26 |
| 0.45 | 16 / 224 | 7.14% | 93.55% | 6.45% | 0.4888 | 0.457/0.479/0.535 | 2;3;4;5;6;8;9;10;11;14;18;22;23;24;25;26 |
| 0.50 | 9 / 224 | 4.02% | 96.48% | 3.52% | 0.5369 | 0.516/0.519/0.567 | 2;3;6;9;11;18;22;25;26 |

Interpretation:
- `Cache Entry Fraction` treats each KV head-token entry as one unit. A compacted head uses the selected AM budget; an uncompressed head keeps all sampled tokens.
- This is still an attention-output proxy, not an end-to-end generation quality measurement.
- A policy is useful only if the holdout policy gives meaningful cache savings at a quality threshold high enough for downstream tasks.
