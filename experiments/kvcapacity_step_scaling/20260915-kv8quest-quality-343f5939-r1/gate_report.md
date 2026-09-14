# KV8 + Quest Selective-Dequant Gate

Classification: **NO_GO_KV8_QUEST**

## Threshold failures

- `quest_not_faster_in_every_cell`

## Performance

| B | bf16 ms | KV8 ms | KV8+Quest ms | Quest/KV8 |
|---:|---:|---:|---:|---:|
| 4 | 41.568 | 65.892 | 82.521 | 1.252x |
| 8 | 44.221 | 114.430 | 81.580 | 0.713x |
| 12 | 49.077 | 166.306 | 96.423 | 0.580x |
| 16 | 51.654 | 215.853 | 122.162 | 0.566x |
| 19 | 56.268 | 252.753 | 142.164 | 0.562x |

B=19 excess-latency recovery: **56.3%**

## Quality

- bf16 overall: 100.0%
- KV8 full overall: 100.0%
- KV8+Quest overall: 100.0%
- Quest vs KV8 full: +0.0 pp

| Depth | KV8 full | KV8+Quest | Delta |
|---:|---:|---:|---:|
| 0.00 | 100.0% | 100.0% | +0.0 pp |
| 0.25 | 100.0% | 100.0% | +0.0 pp |
| 0.50 | 100.0% | 100.0% | +0.0 pp |
| 0.75 | 100.0% | 100.0% | +0.0 pp |
| 1.00 | 100.0% | 100.0% | +0.0 pp |

## Claim boundary

Qwen3-8B, A100 80GB PCIe, TP1, eager decode, context 8192, batches 4/8/12/16/19, pinned 640-block KV pool, synthetic fixed needle workload; no graph-path, production, TP2/TP4, or cross-model claim
