# Policy Ablation Variants

Source policy rows: `experiments/light_doc_cache/policy_am_qwen3_0_6b_s1536_holdout_constrained_cap18_l1_r1.0/policy_rows.csv`
Source threshold: `0.45`
Mode: `leave_one_out`

Leave-one-out ablation for v3 non-A failure diagnosis.

| Threshold | Variant | Compressed Heads | Dropped Heads |
|---:|---|---:|---:|
| 0.451 | drop_l2_h7 | 16 | 1 |
| 0.452 | drop_l3_h0 | 16 | 1 |
| 0.453 | drop_l4_h6 | 16 | 1 |
| 0.454 | drop_l5_h2 | 16 | 1 |
| 0.455 | drop_l6_h7 | 16 | 1 |
| 0.456 | drop_l8_h1 | 16 | 1 |
| 0.457 | drop_l9_h1 | 16 | 1 |
| 0.458 | drop_l10_h3 | 16 | 1 |
| 0.459 | drop_l11_h2 | 16 | 1 |
| 0.460 | drop_l14_h5 | 16 | 1 |
| 0.461 | drop_l18_h0 | 16 | 1 |
| 0.462 | drop_l20_h2 | 16 | 1 |
| 0.463 | drop_l22_h1 | 16 | 1 |
| 0.464 | drop_l23_h7 | 16 | 1 |
| 0.465 | drop_l24_h2 | 16 | 1 |
| 0.466 | drop_l25_h3 | 16 | 1 |
| 0.467 | drop_l26_h1 | 16 | 1 |
