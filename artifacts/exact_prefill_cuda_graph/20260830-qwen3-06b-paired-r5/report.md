# Exact Prefill CUDA Graph Gate

Classification: `GO_EXACT_PREFILL_GRAPH`

## Benefit and cost

- 256 tokens: TTFT improvement 0.8369657783447184; TPOT regression -0.013751147269480235; E2E regression -0.33319189297050655.
- 2048 tokens: TTFT improvement 0.37588076148615823; TPOT regression -0.006694240980542943; E2E regression -0.15262365343930462.
- Startup capture duration median ns: 727916919.0.
- Reserved-memory delta maximum bytes: 41943040.
