# Light Doc Cache Multi-Target Gate

Boundary: default-off restored-sidecar next-token comparison; no physical KV allocation or attention hot-path change.

- Decision: `NO_GO`
- Paired targets: `8/8`
- Holdout wins: `0`
- Aggregate relative improvement: `-105.13%`
- Worst relative regression: `3.2297973221157115`

## Conditions

- [x] all eight paired targets completed
- [x] holdout argmax rate not lower
- [ ] holdout wins at least five targets
- [ ] mean logit diff improves at least five percent
- [ ] worst relative regression no more than twenty five percent
- [ ] no correlated argmax match regressed
- [x] actual prompt tokens match intended length buckets

## Per-Target Rows

| Target | Mode | Status | Tokens | Mean Logit Diff | Argmax Match |
|---|---|---|---:|---:|---|
| `code` | `calibration_holdout` | success | 79 | 1.22797 | True |
| `code` | `correlated_same_layer_target` | success | 79 | 0.292868 | True |
| `code` | `repeat_last_target` | success | 79 | 0.34493 | True |
| `cross_paragraph` | `calibration_holdout` | success | 229 | 1.42944 | True |
| `cross_paragraph` | `correlated_same_layer_target` | success | 229 | 0.86285 | False |
| `cross_paragraph` | `repeat_last_target` | success | 229 | 0.779818 | False |
| `document_qa` | `calibration_holdout` | success | 212 | 1.35235 | False |
| `document_qa` | `correlated_same_layer_target` | success | 212 | 0.860909 | False |
| `document_qa` | `repeat_last_target` | success | 212 | 0.755026 | False |
| `math` | `calibration_holdout` | success | 51 | 1.10947 | True |
| `math` | `correlated_same_layer_target` | success | 51 | 0.63399 | True |
| `math` | `repeat_last_target` | success | 51 | 0.640721 | True |
| `ood` | `calibration_holdout` | success | 52 | 0.950152 | True |
| `ood` | `correlated_same_layer_target` | success | 52 | 0.566026 | False |
| `ood` | `repeat_last_target` | success | 52 | 0.577385 | True |
| `repetitive` | `calibration_holdout` | success | 202 | 2.32133 | False |
| `repetitive` | `correlated_same_layer_target` | success | 202 | 0.548805 | True |
| `repetitive` | `repeat_last_target` | success | 202 | 0.571998 | True |
| `short_fact` | `calibration_holdout` | success | 31 | 1.23006 | False |
| `short_fact` | `correlated_same_layer_target` | success | 31 | 0.579231 | False |
| `short_fact` | `repeat_last_target` | success | 31 | 0.903192 | False |
| `structured` | `calibration_holdout` | success | 36 | 0.921632 | True |
| `structured` | `correlated_same_layer_target` | success | 36 | 0.794589 | True |
| `structured` | `repeat_last_target` | success | 36 | 0.523985 | True |
