# Light Doc Cache Multi-Target Gate

Boundary: default-off restored-sidecar next-token comparison; no physical KV allocation or attention hot-path change.

- Decision: `NO_GO`
- Paired targets: `8/8`
- Holdout wins: `0`
- Aggregate relative improvement: `-99.86%`
- Worst relative regression: `3.1929122736749385`

## Conditions

- [x] all eight paired targets completed
- [x] holdout argmax rate not lower
- [ ] holdout wins at least five targets
- [ ] mean logit diff improves at least five percent
- [ ] worst relative regression no more than twenty five percent
- [ ] no correlated argmax match regressed
- [ ] actual prompt tokens match intended length buckets

## Per-Target Rows

| Target | Mode | Status | Tokens | Mean Logit Diff | Argmax Match |
|---|---|---|---:|---:|---|
| `code` | `calibration_holdout` | success | 79 | 1.22797 | True |
| `code` | `correlated_same_layer_target` | success | 79 | 0.292868 | True |
| `code` | `repeat_last_target` | success | 79 | 0.34493 | True |
| `cross_paragraph` | `calibration_holdout` | success | 141 | 1.32611 | True |
| `cross_paragraph` | `correlated_same_layer_target` | success | 141 | 0.757586 | False |
| `cross_paragraph` | `repeat_last_target` | success | 141 | 0.755837 | True |
| `document_qa` | `calibration_holdout` | success | 153 | 1.48067 | False |
| `document_qa` | `correlated_same_layer_target` | success | 153 | 0.891581 | True |
| `document_qa` | `repeat_last_target` | success | 153 | 1.3005 | True |
| `math` | `calibration_holdout` | success | 51 | 1.10947 | True |
| `math` | `correlated_same_layer_target` | success | 51 | 0.63399 | True |
| `math` | `repeat_last_target` | success | 51 | 0.640721 | True |
| `ood` | `calibration_holdout` | success | 52 | 0.950152 | True |
| `ood` | `correlated_same_layer_target` | success | 52 | 0.566026 | False |
| `ood` | `repeat_last_target` | success | 52 | 0.577385 | True |
| `repetitive` | `calibration_holdout` | success | 102 | 1.78887 | False |
| `repetitive` | `correlated_same_layer_target` | success | 102 | 0.505105 | True |
| `repetitive` | `repeat_last_target` | success | 102 | 0.544225 | True |
| `short_fact` | `calibration_holdout` | success | 31 | 1.23006 | False |
| `short_fact` | `correlated_same_layer_target` | success | 31 | 0.579231 | False |
| `short_fact` | `repeat_last_target` | success | 31 | 0.903192 | False |
| `structured` | `calibration_holdout` | success | 36 | 0.921632 | True |
| `structured` | `correlated_same_layer_target` | success | 36 | 0.794589 | True |
| `structured` | `repeat_last_target` | success | 36 | 0.523985 | True |
