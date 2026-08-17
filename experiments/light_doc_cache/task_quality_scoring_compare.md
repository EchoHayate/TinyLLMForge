# Task Quality Choice Scoring Compare

Same Qwen3-0.6B task quality smoke, thresholds `0.35` and `0.50`, different candidate scoring strings.

| Scoring | Threshold | Baseline Gate | Baseline Acc | Compact Acc | Agreement | Baseline-Correct Agreement | Mean Answer LogP Delta | Compact Margin |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| letter | 0.35 | weak-baseline | 60.00% | 20.00% | 40.00% | 33.33% | -1.8541 | -0.7531 |
| letter | 0.50 | weak-baseline | 60.00% | 60.00% | 100.00% | 100.00% | 0.2862 | 0.4000 |
| space_letter | 0.35 | weak-baseline | 60.00% | 40.00% | 40.00% | 33.33% | -0.4697 | -0.5625 |
| space_letter | 0.50 | weak-baseline | 60.00% | 80.00% | 80.00% | 100.00% | -0.0263 | 0.5875 |
| letter_dot_text | 0.35 | pass | 80.00% | 60.00% | 80.00% | 75.00% | -4.4366 | 2.0983 |
| letter_dot_text | 0.50 | pass | 80.00% | 80.00% | 100.00% | 100.00% | 0.3970 | 1.7391 |
| text_only | 0.35 | pass | 80.00% | 80.00% | 100.00% | 100.00% | -3.2911 | 0.7557 |
| text_only | 0.50 | pass | 80.00% | 80.00% | 100.00% | 100.00% | 0.0311 | 2.7043 |
| space_text | 0.35 | pass | 80.00% | 60.00% | 80.00% | 75.00% | -3.6367 | -0.1070 |
| space_text | 0.50 | pass | 80.00% | 80.00% | 100.00% | 100.00% | -0.2264 | 1.7970 |

Interpretation:
- `letter` and `space_letter` leave the baseline below the 80% gate, so they are weak for this task smoke.
- `text_only` gives the cleanest compact-vs-baseline behavior in this toy set: both thresholds keep 100% agreement on baseline-correct tasks, while `0.35` still shows a large negative answer log-probability delta and lower compact margin.
- `threshold=0.50` remains the safer policy across scoring modes, but it compresses only 11 heads in the current policy and therefore has weak net compression upside.
- This is still only a 5-task smoke; expand task coverage before making go/no-go claims.
