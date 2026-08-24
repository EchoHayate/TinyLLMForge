# Exact-Burst Medium-Context Split-K Audit

Date: 2026-08-24

## Objective

Validate a runtime-data-flow-specific original engineering design that keeps
the existing exact greedy K8 burst graph as the universal fallback and adds a
fixed FlashAttention `num_splits=12` graph only when the complete authorized
burst lies within context lengths 1537 through 4097.

Promotion requires all of the following:

- exact output-token and argmax parity;
- bounded logit drift;
- at least 1% aggregate median TPOT improvement in the target range;
- no target bucket median/P95 regression above 2%;
- no control-context median/P95 regression above 1%;
- no TTFT, E2E, or throughput regression above 2%;
- no extra KV scratch blocks;
- no more than 8 MiB added retained static bytes;
- no more than 64 MiB added reserved memory;
- no more than 5 seconds added capture duration;
- source-bound complete rows, manifest hashes, and agreeing remote/local
  independent-verifier receipts.

## Prompt-to-Artifact Checklist

| Requirement | Implementation evidence | Contract evidence | GPU/artifact evidence | Status |
| --- | --- | --- | --- | --- |
| Default-off feature flag | `tinyvllm/config.py` | `tools/test_model_runner_spec_verify.py` | r11 was explicitly enabled only for the candidate arm | PASS |
| Split12 only for complete bursts in `[1537, 4097]` | `tinyvllm/engine/exact_greedy_decode_burst.py` | selector boundary tests | r11 replay metadata selected 12 only for target contexts | PASS |
| Auto graph remains fallback | `tinyvllm/engine/model_runner.py` | dual-capture/failure/fallback tests | 1025 and 6145 used split count 0 in both arms | PASS |
| Paired auto/split12 workload | `tools/profile_exact_burst_medium_split_k.py` | profile contract tests | r11 has all 48 expected performance rows | PASS |
| Raw TPOT, TTFT, E2E, throughput, memory, capture, lifecycle | profile row schema | profile validation tests | r11 performance rows reconstruct successfully | PASS |
| Token IDs, argmax, and bounded logit drift | correctness sidecars and gate | producer/verifier mutation tests | five paired performance cases have different output-token IDs | **FAIL** |
| Complete-evidence classification only | producer gate | missing/duplicate/stale-summary tests | r11 stopped before writing 64 correctness rows or a gate receipt | PASS: no false receipt emitted |
| Independent reconstruction | independent verifier | manifest/raw/sidecar tamper tests | not run because r11 is incomplete and already violates exact-token parity | NOT RUN |
| Strict-clean remote execution and mounted runtime paths | remote controller | controller safety tests | preflight selected clean GPU 2 and mounted `/data00/home/sitian` paths | PASS |
| Benefit and cost reported together | gate summary/report | GO and threshold mutation tests | reconstructed below from all 48 performance rows | PASS |

## Source Boundary

- Authoritative checkout: `/Users/bytedance/Desktop/TinyLLMForge`
- Branch: `feat/kv-sparse-attention`
- r11 performance source commit:
  `1c80a7e3e8a249f04a3bef4157b6041d44dcec85`
- Audit-closeout source commit before this document:
  `c21cdb9727f4b6289cdae5b2d8c87af8a3b88b5f`
- Remote runtime root:
  `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/exact-burst-medium-split-k`
- Model: `/data00/home/sitian/.ms_cache/Qwen/Qwen3-0___6B`
- Hardware admission: one A100 with memory used at most 1024 MiB,
  utilization at most 5%, and no compute process.

## Unit and Contract Evidence

- Latest local feature suite after fixing tuple-logit serialization:
  `218 passed, 1 skipped`.
- The skipped test is the existing environment-dependent test and is not a
  pass claim for GPU behavior.

## Microgate

Run:
`20260824-qwen3-06b-medium-split-k-micro-r11`

Artifact:
`artifacts/exact_burst_medium_split_k/20260824-qwen3-06b-medium-split-k-micro-r11`

Inventory:

- performance rows: 48 of 48;
- correctness rows: 0 of 64;
- remote exit code: 1;
- terminal producer/verifier receipts: absent, as required for incomplete
  evidence.

The run completed the entire paired performance matrix before the correctness
probe failed while serializing a tuple-valued logits sample. The serializer
defect was fixed in `c21cdb9`, but a replacement run is not justified because
the already-complete performance rows contain exact-token failures.

### Performance Benefit

All values below are reconstructed from the complete 3-repetition paired
performance matrix. Positive values mean split12 is faster.

| Context | Median TPOT change | TTFT regression | E2E regression |
| ---: | ---: | ---: | ---: |
| 1025 control | -0.275% | -4.816% | -0.279% |
| 1537 | +0.298% | +1.950% | +0.131% |
| 2049 | +2.005% | -3.398% | -2.056% |
| 2561 | +3.534% | -3.950% | -3.612% |
| 3073 | +3.034% | **+2.331%** | -2.457% |
| 3585 | +2.621% | -0.481% | -2.563% |
| 4090 | -0.343% | +1.681% | +0.520% |
| 6145 control | -0.045% | -0.241% | -0.201% |

The target aggregate median TPOT improvement is **2.279%**, above the frozen
1% benefit threshold. However, the 3073-token TTFT regression is 2.331%,
above the frozen 2% limit.

### Correctness Failure

The performance rows contain the full generated token sequences for both
arms. Five of the 24 paired cases diverge:

| Repetition | Context | First differing token index | Differing token count |
| ---: | ---: | ---: | ---: |
| 0 | 2049 | 28 | 92 |
| 1 | 2049 | 7 | 121 |
| 2 | 2049 | 17 | 111 |
| 2 | 2561 | 17 | 1 |
| 2 | 3073 | 2 | 125 |

This directly violates the exact output-token requirement. Missing logits
sidecars prevent quantifying argmax and logit drift, but cannot reverse the
observed token mismatch.

### Resource and Capture Cost

The same 48 rows reconstruct these split12-minus-auto costs:

| Cost | Result | Frozen limit | Verdict |
| --- | ---: | ---: | --- |
| Added retained static bytes | 915,852 bytes | 8 MiB | PASS |
| Added reserved bytes | 2,097,152 bytes | 64 MiB | PASS |
| Added scratch blocks | 0 | 0 | PASS |
| Added capture duration | 18,359,333,218 ns | 5,000,000,000 ns | **FAIL** |
| Candidate peak allocated | 41,653,415,424 bytes | report only | — |
| Auto peak allocated | 41,828,656,640 bytes | report only | — |
| Candidate peak reserved | 41,957,720,064 bytes | report only | — |
| Auto peak reserved | 41,959,817,216 bytes | report only | — |

The added capture-duration metric uses the frozen gate definition: maximum
candidate capture duration minus maximum auto capture duration across the
complete matrix.

### Reproduction

The performance and cost reconstruction is reproducible without accepting an
incomplete artifact:

```bash
python3 - <<'PY'
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path("tools").resolve()))
import exact_burst_medium_split_k_gate as gate

path = Path(
    "artifacts/exact_burst_medium_split_k/"
    "20260824-qwen3-06b-medium-split-k-micro-r11/"
    "primary/performance_rows.jsonl"
)
rows = [
    gate.validate_case_row(json.loads(line))
    for line in path.read_text().splitlines()
    if line.strip()
]
print(gate._performance_metrics(rows, repetitions=3))
print(gate._cost_metrics(rows))
PY
```

## Canonical Gate

Not run. The microgate crossed hard correctness, TTFT, and capture-cost
boundaries. Running the 5-repetition canonical gate would consume additional
GPU time without a promotion path.

## Final Classification

`NO_GO_CORRECTNESS`

This is the candidate disposition, not a fabricated terminal gate receipt:
r11 is intentionally retained as an incomplete attempted artifact. The
split12 graph is not promoted or enabled by default. Its measured benefit is
approximately 2.279% target median TPOT, paid for with exact-token divergence,
a 2.331% TTFT regression at context 3073, and about 18.36 seconds of added
capture duration under the frozen cost definition.
