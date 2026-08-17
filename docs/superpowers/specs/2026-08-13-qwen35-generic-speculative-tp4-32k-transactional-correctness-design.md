# Qwen3.5 Generic Speculative TP4 32K Transactional Correctness

## Status

Approved by the existing autonomous Phase 1 execution mandate and the ordered
handoff that places an independent TP4/32K authority immediately after the
established TP4/16K authority.

This specification creates a new 32K-only authority. It must not modify,
parameterize, replace, or reinterpret the established 4K and 16K authorities.
No staging, commit, push, branch switch, worktree, stash, reset, or clean is
allowed.

## Goal

Establish source-bound evidence that the production generic speculative
runtime executes the approved Qwen3.5 hybrid checkpoint at TP world size four
and 32,768 prompt tokens while preserving exact greedy output, transactional
full-attention KV, and transactional recurrent state on every rank.

Passing establishes only:

```text
SECOND_MODEL_TP4_32K_ESTABLISHED
```

Phase 1 remains:

```text
NOT_PROMOTABLE
```

## Frozen Inputs

- workspace: `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`;
- remote: `sitian@10.232.195.203`;
- Kerberos cache: `FILE:/Users/bytedance/krb5cc_sitian`;
- SSH: `ControlMaster=no`, `ControlPath=none`;
- checkpoint:
  `/data00/home/sitian/sitian-workspace01/tllm/qwen35-hybrid-state-runs/qwen35-2b-hybrid-acquire-20260723-222004/model`;
- remote Python:
  `/data00/home/sitian/sitian-workspace01/tllm/env/bin/python`;
- model manifest:
  `3e650a908234771c3cf1ac4e20c4d38fe69982efedaf4a3e631ad0b14aad7dd0`;
- schema:
  `qwen35.generic-speculative-tp4-32k-transactional-correctness.v1`;
- classification: `SECOND_MODEL_TP4_32K_ESTABLISHED`;
- scope: `second_model_tp4_32k_only`;
- world size: `4`;
- batch sizes: `1` and `4`;
- policies: `baseline` and `ngram`;
- context tokens: `32768`;
- output tokens: `8`;
- n-gram size: `3`;
- maximum proposal tokens: `4`;
- `max_model_len=33024`;
- `max_num_batched_tokens=132096`;
- `max_num_prefill_tokens_per_step=1024`;
- `chunked_prefill_decode_first=False`;
- `chunked_prefill_mixed_batch=False`;
- `kv_offload_gpu_blocks=68`;
- `kv_offload_logical_blocks=640`;
- `kv_offload_blockwise_blocks=8`;
- blockwise prefill and decode enabled.

The existing long-context authority already exercised this Engine envelope.
At block size 256 tokens, 32K batch 1 exposes 128 logical blocks and batch 4
exposes 512 logical blocks. Both exceed the 68 GPU staging slots. Therefore
both n-gram cells must record positive production H2D copies and bytes.

## Authority Surface

Create:

```text
tools/qwen35_generic_speculative_tp4_32k_gate.py
tools/qwen35_generic_speculative_tp4_32k_worker.py
tools/verify_qwen35_generic_speculative_tp4_32k_gate.py
tools/run_qwen35_generic_speculative_tp4_32k_gate_remote.sh
tools/test_qwen35_generic_speculative_tp4_32k_gate.py
```

The new authority may reuse side-effect-free orchestration and validation from
the frozen 4K authority, but it must replace all identity, source inventory,
worker dispatch, verifier dispatch, context, and movement requirements before
execution. It must not create a second speculative runtime.

## Required Evidence

Every successful result must prove:

1. the approved 24-layer Qwen3.5 hybrid checkpoint is loaded on four ranks;
2. baseline and candidate output token IDs are exactly equal per request;
3. batch 1 and batch 4 are real Engine runs;
4. every candidate cell has accepted and rejected proposal tokens;
5. all ranks agree on proposals, acceptance, committed-input mapping, KV
   decisions, and recurrent checkpoint selection;
6. accepted KV is committed from the batch-native verifier result;
7. accepted-prefix model replay count is zero;
8. rejected KV and recurrent suffix state remain unpublished;
9. callback, collective, residency, and cleanup evidence is complete;
10. candidate batch 1 has positive production H2D copies and bytes;
11. candidate batch 4 has positive production H2D copies and bytes;
12. rejected speculative D2H copies remain zero;
13. all workers, process groups, leases, prepared transactions, Engines, and
    owned children are cleaned up; and
14. an independent verifier fails closed for missing, inconsistent, tampered,
    or source-unbound evidence.

## Failure Semantics

Any mismatch writes `authority.failed`. Existing run directories cannot be
replayed. Only a fresh campaign followed by source-bound independent verifier
PASS may write `authority`.

The runner must use serial bounded SSH/rsync operations and finite retries.
Intermittent proxy/Kerberos failures may be retried only within those bounds.

## Explicit Non-Claims

This authority does not establish:

- TPOT, TTFT, throughput, peak-memory, or traffic improvement;
- lower H2D than baseline or 16K;
- learned drafter or native MTP;
- KV8 or KV4;
- production readiness; or
- Phase 1 completion.

## Validation Order

1. contract RED/GREEN for identity and source isolation;
2. RED/GREEN for positive candidate H2D in both batch sizes;
3. RED/GREEN for long-context worker configuration;
4. RED/GREEN for 32K worker/source/verifier campaign dispatch;
5. independent verifier tamper tests;
6. bounded runner source contract and `bash -n`;
7. complete 32K tests and frozen 4K/16K regressions;
8. Python compilation and `git diff --check`;
9. one fresh real TP4/32K GPU campaign;
10. fresh independent verification and raw evidence audit;
11. objective audit and handoff update.
