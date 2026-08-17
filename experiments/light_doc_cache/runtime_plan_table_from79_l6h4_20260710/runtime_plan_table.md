# Light Doc Cache Runtime Plan Table

Boundary: planning/metrics only; this is not a runtime KV-cache compression or latency result.

| Doc | Task | Fallback | Recovered Heads | Recovered KV-Head Eq | Effective Saving | Compression Ratio |
|---|---|---|---:|---:|---:|---:|
| first | route_phase1 |  | 80 | 39.75 | 17.75% | 1.2157 |
| first | route_phase2 |  | 80 | 39.75 | 17.75% | 1.2157 |
| first | route_phase3 | task_override | 79 | 39.50 | 17.63% | 1.2141 |
| first | long_context_bottleneck |  | 80 | 39.75 | 17.75% | 1.2157 |
| first | quest_decode_selection | task_override | 79 | 39.50 | 17.63% | 1.2141 |
| first | quest_summary_form |  | 80 | 39.75 | 17.75% | 1.2157 |
| first | quest_dtype |  | 80 | 39.75 | 17.75% | 1.2157 |
| first | quest_default_enable |  | 80 | 39.75 | 17.75% | 1.2157 |
| first | quest_min_seq_len |  | 80 | 39.75 | 17.75% | 1.2157 |
| first | phase2_magic_number |  | 80 | 39.75 | 17.75% | 1.2157 |
| first | prefix_cache_bug |  | 80 | 39.75 | 17.75% | 1.2157 |
| first | topk8_quality | task_override | 79 | 39.50 | 17.63% | 1.2141 |
| first | sweet_spot | task_override | 79 | 39.50 | 17.63% | 1.2141 |
| second | qwen3_8b_model_choice |  | 80 | 39.75 | 17.75% | 1.2157 |
| second | w4a8_failure_symptom |  | 80 | 39.75 | 17.75% | 1.2157 |
| second | tp_memory_correction |  | 80 | 39.75 | 17.75% | 1.2157 |
| second | tp_true_weight_split | task_override | 79 | 39.50 | 17.63% | 1.2141 |
| second | gpu_utilization_semantics | task_override | 79 | 39.50 | 17.63% | 1.2141 |
| second | smoothquant_status | task_override | 79 | 39.50 | 17.63% | 1.2141 |

## Summary

- Tasks: `19`
- Average Effective Saving: `17.70%`
- Average Compression Ratio: `1.2151`
- Full KV Bytes: `176,160,768`
- Average Planned Recovered KV Bytes: `31,188,237`
- Average Planned Stored KV Bytes: `144,972,531`

Use this table to audit planned storage/recovery accounting before any ModelRunner hot-path integration.
