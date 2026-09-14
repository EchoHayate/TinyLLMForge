# KV8 + Quest Amortization-Aware Activation Gate

Classification: `GO_KV8_QUEST_AMORTIZATION_POLICY`

## Performance

| Batch | bf16 ms | KV8 ms | Fixed Quest ms | Adaptive ms | Adaptive/KV8 | Adaptive/Fixed |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 43.802 | 66.060 | 82.802 | 67.054 | 1.015x | 0.810x |
| 6 | 44.262 | 89.942 | 83.230 | 90.904 | 1.011x | 1.092x |
| 8 | 45.395 | 114.480 | 86.366 | 83.939 | 0.733x | 0.972x |
| 10 | 47.087 | 139.320 | 86.521 | 85.971 | 0.617x | 0.994x |
| 12 | 48.989 | 165.244 | 97.399 | 98.050 | 0.593x | 1.007x |
| 16 | 51.694 | 215.172 | 122.898 | 123.538 | 0.574x | 1.005x |
| 19 | 52.291 | 252.341 | 142.405 | 142.566 | 0.565x | 1.001x |

## Quality

| Scope | KV8 full | Adaptive | Delta |
| :--- | ---: | ---: | ---: |
| Overall | 100.000% | 100.000% | +0.000 pp |
| Depth 0.00 | 100.000% | 100.000% | +0.000 pp |
| Depth 0.25 | 100.000% | 100.000% | +0.000 pp |
| Depth 0.50 | 100.000% | 100.000% | +0.000 pp |
| Depth 0.75 | 100.000% | 100.000% | +0.000 pp |
| Depth 1.00 | 100.000% | 100.000% | +0.000 pp |

## Failures

- Identity: none
- Threshold: none

## Claim boundary

Qwen3-8B, A100 80GB PCIe, TP1, eager decode, 8192-token synthetic contexts, 640 KV blocks, fixed needle prompts; default remains disabled
