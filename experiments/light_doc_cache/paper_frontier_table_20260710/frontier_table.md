# Light Doc Cache Frontier Table

Quality-only recovery-bank simulation results. These rows preserve the strict task gates in the listed offline smoke tests, but they are not runtime KV-cache compression measurements.

| Frontier | Kind | Heads | First Doc Gate | Second Doc Gate | First Saving | Second Saving | First Delta | Second Delta | Fallback Tasks | Claim |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| static_79_head | fixed_global | 79 | 13/13 | 6/6 | 17.63% | 17.63% | 0.6782 | 1.4649 | 0 | quality-only |
| first_doc_adaptive_15h5 | task_adaptive | 80 | 13/13 | 6/6 | 17.71% | 17.75% | 0.9575 | 1.7281 | 4 | quality-only |
| two_doc_adaptive_6h6 | doc_task_adaptive | 80 | 13/13 | 6/6 | 17.71% | 17.69% | 0.7536 | 1.6298 | 7 | quality-only |
| two_doc_adaptive_5h5 | doc_task_adaptive | 80 | 13/13 | 6/6 | 17.71% | 17.69% | 0.6721 | 1.5364 | 7 | quality-only |
| two_doc_adaptive_7h7 | doc_task_adaptive | 80 | 13/13 | 6/6 | 17.71% | 17.69% | 0.7426 | 1.4269 | 7 | quality-only |
| two_doc_adaptive_6h4 | doc_task_adaptive | 80 | 13/13 | 6/6 | 17.71% | 17.69% | 0.6108 | 1.4672 | 7 | quality-only |

Claim boundary: report these as offline task/document-adaptive recovery-bank quality results with average effective KV head-token entry saving. Do not describe them as 2x+ runtime doc-cache compression.
