# Light Doc Cache Multi-Target Gate

Boundary: default-off restored-sidecar next-token comparison; no physical KV allocation or attention hot-path change.

- Decision: `NO_GO`
- Paired targets: `2/8`
- Holdout wins: `0`
- Aggregate relative improvement: `-56.62%`
- Worst relative regression: `1.1236020740739827`

## Conditions

- [ ] all eight paired targets completed
- [x] holdout argmax rate not lower
- [ ] holdout wins at least five targets
- [ ] mean logit diff improves at least five percent
- [ ] worst relative regression no more than twenty five percent
- [x] no correlated argmax match regressed
- [x] actual prompt tokens match intended length buckets

## Per-Target Rows

| Target | Mode | Status | Tokens | Mean Logit Diff | Argmax Match |
|---|---|---|---:|---:|---|
| `short_fact` | `calibration_holdout` | success | 31 | 1.23006 | False |
| `short_fact` | `correlated_same_layer_target` | success | 31 | 0.579231 | False |
| `short_fact` | `repeat_last_target` | success | 31 | 0.903192 | False |
| `structured` | `calibration_holdout` | success | 36 | 0.921632 | True |
| `structured` | `correlated_same_layer_target` | success | 36 | 0.794589 | True |
| `structured` | `repeat_last_target` | success | 36 | 0.523985 | True |
