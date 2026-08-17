# Provisional TinyLLM Calibrated-Target Read-Path Matrix

Boundary: this is **not** the final artifact-backed matrix. The two baseline
rows are copied from prior remote stdout and must be replaced by local
`tinyllm_sidecar_read_path_summary.json` artifacts after SSH credentials are
restored. The calibrated row is backed by the local summary JSON artifact.

| Mode | Role | Source | Max Logit Diff | Mean Logit Diff | Argmax Match | Original Argmax | Restored Argmax | Artifact Status |
|---|---|---|---:|---:|---|---:|---:|---|
| `repeat_last_target` | baseline | prior remote stdout | 4.0625 | 0.598329 | true | 785 | 785 | pending summary JSON pull |
| `correlated_same_layer_target` | baseline | prior remote stdout | 3.59375 | 0.507468 | true | 785 | 785 | pending summary JSON pull |
| `calibrated_multi_correlated_target` | trained | local summary JSON | 3.890625 | 0.684234 | true | 785 | 785 | local summary JSON available |

Calibrated artifact:

- `experiments/light_doc_cache/tinyllm_sidecar_read_path_calibrated_smoke_qwen3_0_6b_20260710/tinyllm_sidecar_read_path_summary.json`
- Prompt tokens: `14`
- Logical saving: `17.6339%`
- Missing compact tokens: `553`
- Missing-token MSE: `13.5193`
- Missing-token max abs: `219`

Interpretation:

- All three known target-prompt rows preserve argmax (`785 -> 785`).
- Based on stdout only, the trained calibrated row does not beat the simple
  `correlated_same_layer` baseline on mean or max logit diff for this single
  target prompt.
- Because baseline rows are not backed by local summary JSON yet, this file is
  only a progress/handoff artifact. The final matrix must be regenerated with
  `experiments/light_doc_cache/make_read_path_recovery_matrix.py` after pulling
  the two remote baseline artifact directories.
