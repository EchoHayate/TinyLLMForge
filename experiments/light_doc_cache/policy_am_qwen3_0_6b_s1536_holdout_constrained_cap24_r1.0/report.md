# Attention-Output Selective Compression Policy

Source run: `runs/am_qwen3_0_6b_s1536_b64_128_256_r1.0`
Source decision: **AM_WEAK**

best holdout highest fitV mean R2=0.207, coverage@0.5=4.46%

Regime: `holdout`; selector: `highest`; metric: `fitv_val_r2`; layer filter: `0`..`-1`.

constrained-cap24-task-quality-sweep

| R2 threshold | Compressed Heads | Head Fraction | Cache Entry Fraction | Cache Saving | Mean Compressed R2 | P10/P50/P90 R2 | Layers |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0.45 | 23 / 224 | 10.27% | 90.81% | 9.19% | 0.4816 | 0.454/0.475/0.518 | 2;3;4;5;6;8;9;10;11;14;18;20;22;23;24;25;26 |
| 0.50 | 11 / 224 | 4.91% | 95.70% | 4.30% | 0.5362 | 0.516/0.519/0.567 | 2;3;6;9;11;18;22;25;26 |

Interpretation:
- `Cache Entry Fraction` treats each KV head-token entry as one unit. A compacted head uses the selected AM budget; an uncompressed head keeps all sampled tokens.
- This is still an attention-output proxy, not an end-to-end generation quality measurement.
- A policy is useful only if the holdout policy gives meaningful cache savings at a quality threshold high enough for downstream tasks.
