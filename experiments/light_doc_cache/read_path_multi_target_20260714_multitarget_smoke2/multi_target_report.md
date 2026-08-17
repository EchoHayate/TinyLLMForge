# Light Doc Cache Multi-Target Gate

Boundary: default-off restored-sidecar next-token comparison; no physical KV allocation or attention hot-path change.

- Decision: `NO_GO`
- Paired targets: `2/8`
- Holdout wins: `0`
- Aggregate relative improvement: `0.00%`
- Worst relative regression: `0.0`

## Conditions

- [ ] all eight paired targets completed
- [x] holdout argmax rate not lower
- [ ] holdout wins at least five targets
- [ ] mean logit diff improves at least five percent
- [x] worst relative regression no more than twenty five percent
- [x] no correlated argmax match regressed
- [x] actual prompt tokens match intended length buckets

## Per-Target Rows

| Target | Mode | Status | Tokens | Mean Logit Diff | Argmax Match |
|---|---|---|---:|---:|---|
| `short_fact` | `calibration_holdout` | success | 31 | 5.02742 | False |
| `short_fact` | `correlated_same_layer_target` | success | 31 | 5.02742 | False |
| `short_fact` | `repeat_last_target` | success | 31 | 5.02742 | False |
| `structured` | `calibration_holdout` | success | 36 | 4.28231 | False |
| `structured` | `correlated_same_layer_target` | success | 36 | 4.28231 | False |
| `structured` | `repeat_last_target` | success | 36 | 4.28231 | False |
