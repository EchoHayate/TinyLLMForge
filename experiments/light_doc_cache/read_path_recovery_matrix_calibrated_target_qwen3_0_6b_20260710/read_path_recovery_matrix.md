# TinyLLM Read-Path Recovery Matrix

Boundary: default-off restored-sidecar read-path comparison; no attention hot-path or KV allocation lifetime change.

| Mode | Role | Missing MSE | Missing Max Abs | Max Logit Diff | Mean Logit Diff | Argmax Match | Restored Argmax |
|---|---|---:|---:|---:|---:|---|---:|
| `repeat_last_target` | baseline | 15.2284 | 217 | 4.0625 | 0.598329 | True | 785 |
| `correlated_same_layer_target` | baseline | 11.4076 | 246 | 3.59375 | 0.507468 | True | 785 |
| `calibrated_multi_correlated_target` | trained | 13.5193 | 219 | 3.89062 | 0.684234 | True | 785 |

Common setup:

- Prompt tokens: `14`.
- Logical byte saving fraction: `17.63%`.
- Missing compact tokens: `553`.
- Original argmax: `785`.

Interpretation:

- Non-oracle argmax-preserving modes: `repeat_last_target`, `correlated_same_layer_target`, `calibrated_multi_correlated_target`.
- `correlated_same_layer_target` has the best non-oracle mean logit diff.
- `correlated_same_layer_target` has the best non-oracle missing-token MSE.
