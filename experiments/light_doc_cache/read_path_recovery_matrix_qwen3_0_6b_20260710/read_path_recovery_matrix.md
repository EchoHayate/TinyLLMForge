# TinyLLM Read-Path Recovery Matrix

Boundary: default-off restored-sidecar read-path comparison; no attention hot-path or KV allocation lifetime change.

| Mode | Role | Missing MSE | Missing Max Abs | Max Logit Diff | Mean Logit Diff | Argmax Match | Restored Argmax |
|---|---|---:|---:|---:|---:|---|---:|
| `repeat_last` | baseline | 13.7399 | 224 | 5.5625 | 0.787285 | False | 3491 |
| `linear_tail` | baseline | 38.586 | 336 | 17.5 | 3.13622 | False | 50927 |
| `correlated_same_layer` | correlated | 11.5488 | 174 | 5.21875 | 0.886142 | False | 3491 |
| `correlated_prefix_fit` | correlated_ablation | 13.2058 | 198 | 6.9375 | 1.10957 | False | 13173 |
| `multi_correlated2` | multi_source_ablation | 18.3975 | 384 | 6.96875 | 1.04723 | False | 3972 |
| `oracle` | upper_bound | 0 | 0 | 0 | 0 | True | 1815 |

Common setup:

- Prompt tokens: `13`.
- Logical byte saving fraction: `18.99%`.
- Missing compact tokens: `553`.
- Original argmax: `1815`.

Interpretation:

- No non-oracle mode preserves argmax on this prompt.
- Oracle is exact, so layout, restore indexing, and temporary read-path pointer swap are correct.
- `repeat_last` has the best non-oracle mean logit diff.
- `correlated_same_layer` has the best non-oracle missing-token MSE.
