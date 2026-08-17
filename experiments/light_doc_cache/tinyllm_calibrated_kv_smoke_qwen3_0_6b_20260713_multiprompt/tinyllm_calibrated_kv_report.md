# TinyLLM Calibrated KV Smoke

Boundary: fits/applies a calibrated bank from TinyLLM ModelRunner.kv_cache; no attention hot-path or KV allocation lifetime change.

- Model: `/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B`
- Calibration tokens: `31`
- Calibration plan tokens: `31`
- Target tokens: `14`
- Effective plan tokens: `14`
- KV cache shape: `[2, 28, 1, 256, 8, 128]`
- Source count: `2`
- Recovery bank file: `/tmp/light_doc_cache_tinyllm_calibrated_kv_qwen3_0_6b_20260713_multiprompt/multi_source_recovery_bank.json`
- Missing-token MSE: `11.6785`
- Missing-token max abs error: `258`
- Stored tensor bytes: `1,322,496`
