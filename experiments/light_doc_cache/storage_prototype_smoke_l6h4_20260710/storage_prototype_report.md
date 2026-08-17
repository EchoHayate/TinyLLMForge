# Light Doc Cache Storage Prototype Smoke

Boundary: CPU/toy storage prototype; missing compact-head tokens are fill/repeat-last/linear-tail/oracle baselines, not trained model-quality tensors.

- Recovery mode: `linear_tail`
- KV pattern: `nonlinear`
- Missing-token MSE: `17.6471`
- Missing-token max abs error: `7.41228`
- Full tensor bytes: `57,344`
- Stored tensor bytes: `35,424`
- Saved tensor bytes: `21,920`
- Byte saving fraction: `38.23%`
- Compact heads: `79`
- Full heads: `145`
