# Qwen3 Learned-Drafter TP1 Proposal-KV Offload Authority

This compact bundle preserves the successful real-GPU TP1 correctness and
Proposal-KV movement run executed on August 15, 2026.

## Classification

`PASS_CORRECTNESS_AND_REAL_PROPOSAL_KV_MOVEMENT`

The canonical result is
`run/tp1-qwen3-loaded-offload-gpu4.json`. It reports schema v2,
`gate_pass=true`, and exact greedy output parity for batch 1 and batch 4.
The run used a Qwen3-1.7B target, a Qwen3-0.6B independent draft model,
`max_proposal_tokens=4`, `max_output_tokens=8`, TP1, and Proposal-KV offload
with GPU slot capacity 8.

## Key Evidence

- Batch 1 and batch 4 exact output parity are both true.
- Acceptance rows are nonempty: 4 for batch 1 and 14 for batch 4.
- `extra_target_forward_count=0`.
- `first_target_forward_count=8`,
  `tail_verification_forward_count=8`, and
  `real_draft_forward_count=45`.
- Accepted-entry copy, replay, and rematerialization counts are all zero.
- Proposal-KV and target-KV storage identities are distinct.
- Proposal-KV live slots fall from 46 before release to 0 after release.
- Real bidirectional movement is recorded:
  18,266 H2D entries / 2,094,891,008 bytes and
  83 D2H entries / 9,519,104 bytes.

See `authority_summary.json` for the compact machine-readable summary and the
canonical result for all token IDs, acceptance rows, timings, checkpoint
identities, tokenizer contract, and storage identities.

## Provenance

- Source inventory receipt:
  `4d5e8c856c712f5b97587f20bd85dacd481423d4f90f6ceacb973f7ec8f68b7d`
- Canonical result SHA-256:
  `da8dcf291c2650c0cd5400400e6ed017e1859c1096766f3e34ed1717e4b092d2`
- The exact launch command, start/finish timestamps, exit code, GPU snapshots,
  source inventory, transport manifests, tokenizer diagnostics,
  FlashAttention build logs, wheel/extension hashes, and real kernel smoke are
  retained under `run/`, `logs/`, and `flash_build/`.
- `manifest.sha256` covers every retained file in this directory except itself.

## Claim Boundary

The gate explicitly records `performance_pass_criterion=false`. This is a
correctness and real Proposal-KV movement authority, not a performance
promotion result. It does not establish TP4, 4K/16K/32K long-context coverage,
a second learned model structure, or controlled TPOT/TTFT/throughput gains.
The bundle preserves a source hash inventory and receipt, but it does not
contain a frozen source tarball or an independent current/archive verifier.
