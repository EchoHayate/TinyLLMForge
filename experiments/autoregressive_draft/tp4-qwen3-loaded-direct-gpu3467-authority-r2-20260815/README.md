# Qwen3 Learned-Drafter TP4 Direct Proposal-KV Authority

This bundle preserves the successful real-GPU TP4 correctness run executed on
August 15, 2026 after fixing the FlashAttention incompatibility with the
draft backend's one-token physical Proposal-KV slots.

## Classification

`PASS_TP4_DIRECT_CORRECTNESS`

The canonical result is `run/result.json`. It reports schema v2,
`gate_pass=true`, and exact greedy output parity for batch 1 and batch 4.
The run used a Qwen3-1.7B target, a Qwen3-0.6B independent draft model,
TP4 on GPUs 3/4/6/7, `max_proposal_tokens=4`, `max_output_tokens=8`, and the
direct Proposal-KV allocator with capacity 90.

## Key Evidence

- Batch 1 and batch 4 exact output parity are both true.
- Acceptance rows are nonempty: 4 for batch 1 and 14 for batch 4.
- All four ranks participated in both cases.
- Per-rank decode forward counts are 8 for batch 1 and 19 for batch 4.
- Per-rank real draft forward counts are 9 for batch 1 and 21 for batch 4.
- Accepted-entry copy, replay, and rematerialization counts are zero on every
  rank.
- Every rank releases all direct Proposal-KV entries and returns to 90 free
  physical slots.
- `run/verify.json` records matching archived and current independent verifier
  receipts with classification `PASS`.

See `authority_summary.json` for the compact machine-readable summary and
`run/result.json` for token IDs, acceptance rows, per-rank snapshots,
checkpoint identities, tokenizer contract, logical authority digests, and
transaction records.

## FlashAttention Fix

The Qwen3 draft backend owns Proposal-KV as single-token physical slots. The
installed FlashAttention paged API rejects page sizes not divisible by 256.
The runtime now gathers only such incompatible ordinary FP decode pages into a
temporary dense read view and calls FlashAttention with `block_table=None`.
Writes and transactional ownership remain on the original one-token slots.
Normal 256-token paged caches, Quest, attention matching, and blockwise paths
remain unchanged.

Validation retained in the repository:

- `tools/test_native_verifier_attention.py`: 6 passed locally and remotely.
- `tools/test_blockwise_attention_planning.py`: 35 passed remotely.
- TP4 producer/local-gate/verifier/snapshot contracts: 56 passed locally and
  remotely.
- A short real-kernel smoke crossed the former page-size failure and failed
  only because `max_output_tokens=2` could not satisfy the positive decode
  count evidence rule.
- This final authority run exited 0 and passed the independent verifier.

## Provenance

- Canonical result SHA-256:
  `772a675ed10e1c8d9c2ca3c9f5475fa296214d65c455bf346b3eb3d21aba810f`
- Producer source-tree SHA-256:
  `4c689ef27b15f0052c4da6e35f647690611260b8875c82c1145f750421858870`
- Producer source archive SHA-256:
  `75d0625a63109864169bfd298ad50d5e9b7b65c22a8413a432c883815acd7777`
- Critical runtime source archive SHA-256:
  `fe046eebb7ed9475432fd10d6b0a7148d6b5d891e2c2654b087f24ec4cf52788`
- `run/critical-source-pre.sha256` and
  `run/critical-source-post.sha256` are identical, and
  `run/critical-source-stability.txt` records `match`.
- The critical archive includes `tinyvllm/layers/attention.py` and its two
  focused regression test files because the unchanged producer's 30-file
  source manifest does not include the attention module.
- The exact command, timestamps, exit code, GPU snapshots, full log, producer
  source archive, critical source archive, and verifier receipt are retained
  under `run/` and `logs/`.
- `manifest.sha256` covers every retained file in this directory except itself.

## Claim Boundary

The gate explicitly records `performance_pass_criterion=false`. This is TP4
direct-allocator correctness authority, not a performance promotion result and
not a real Proposal-KV offload movement result. The recorded context is short
(`max_model_len=24`), and the run does not establish 4K/16K/32K behavior, a
second model structure, TPOT/TTFT/throughput improvement, or H2D/D2H savings.
