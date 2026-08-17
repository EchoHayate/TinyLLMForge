# Interrupted exact-shape CUDA Graph paired gate

Date: 2026-08-17

The source-bound TP4/B4/Q4 gate completed two warmup pairs and four measured
position-balanced pairs. Every completed eager/graph pair had exact target
tokens, exact proposal rows, exact accepted-prefix counts, the same
transaction digest, zero active transactions, and one successful capture and
replay on every rank without fallback or quarantine.

During `pair-4-eager`, an unrelated root-owned `VLLM::EngineCore` appeared on
physical GPU 3 and consumed approximately 73.3 GiB. The gate's rank 3 exited
and the remaining ranks stopped making progress. The run therefore did not
produce the required eight measured pairs.

Only the gate's own remote process group, PGID `3537615`, was terminated. The
external process was not signaled or modified. GPU 0 through GPU 2 returned
to idle; GPU 3 remained occupied by the external process.

The retained four-pair aggregate is diagnostic only:

- median eager throughput: `0.989071982502008 tok/s`;
- median graph throughput: `0.9749348881219722 tok/s`;
- mean paired throughput delta: `-0.002993426456363135 tok/s`;
- median eager TPOT: `1.9743923907000003 s`;
- median graph TPOT: `2.0375849077333337 s`; and
- observed capture range: `1.758702684-3.101219179 s`.

Because only four of eight measured pairs completed and the environment was
externally disturbed, this run is classified
`INCONCLUSIVE_ENVIRONMENT_PARTIAL_4_OF_8`. It is not a valid `GO` or
`NO_GO_PERFORMANCE` result.
