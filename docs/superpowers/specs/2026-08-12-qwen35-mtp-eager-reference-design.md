# Qwen3.5 MTP Eager/Reference Parity Design

## Scope

Close only the real-checkpoint gate's `eager_reference` blocker for the
approved first-version domain:

- tensor parallel size 1;
- KV offload disabled;
- one native Qwen3.5 MTP layer;
- greedy proposal tokens;
- exact Q values `(1, 2, 3, 4)`;
- batch sizes `(1, 4)`;
- shared target embedding and LM head.

The CUDA Graph blocker remains open. Overall status remains
`FAIL / NOT_PROMOTABLE`.

## Approaches Considered

### Reload two complete checkpoint modules

Run one module through the existing eager path and a second independently
loaded module through a reference path.

Rejected because it doubles checkpoint memory, increases remote gate setup
cost, and tests module duplication as much as attention parity.

### Compare the existing attention helper with itself

Capture the inputs and outputs of
`qwen35_cached_decode_eager_attention()` and call the same helper again.

Rejected because this is a self-comparison and cannot detect an error shared
by the production invocation and the comparison.

### Recommended: same module, fresh sequences, independent SDPA reference

Run two isolated scenarios through the same loaded checkpoint:

1. the production scenario uses the existing Qwen3.5 prefill/decode attention
   helpers;
2. the reference scenario temporarily replaces those helper entry points with
   independent PyTorch scaled-dot-product-attention equations.

Both scenarios use fresh sequence IDs, identical token IDs, positions, target
hidden tensors, exact Q, batch size, physical K/V allocation, and rollback.
The gate compares every proposal-step logits tensor and greedy argmax.

## Components

### Reference attention equations

`tools/qwen35_mtp_real_checkpoint_gate.py` owns gate-only functions for:

- causal prefill attention;
- cached single-token decode attention.

They write the current K/V rows into the real physical store, gather visible
rows from the exact block table, expand GQA K/V heads, and call
`torch.nn.functional.scaled_dot_product_attention`. They do not call the
production Qwen3.5 eager-attention helpers.

### Scenario runner

`_build_real_eager_reference_probe(...)` returns:

```python
probe(q: int, batch_size: int) -> {
    "max_abs_diff": float,
    "argmax_equal": bool,
}
```

For each side it:

1. creates fresh sequence IDs;
2. installs deterministic target-prefill observations;
3. executes the real `Qwen35MTPProposalExecutor`;
4. captures every one-token `module.forward_step()` logits result;
5. aborts active proposal transactions and releases every sequence;
6. verifies the physical store has no remaining allocation.

Q=1 has no generated MTP step, so the logits domain is empty and
`max_abs_diff=0.0`; proposal token equality still covers the first target
token.

### Runtime installation

`RealQwen35MTPGateBackend._load_real_runtime()` installs the probe only when
the real module, executor, and physical store exist. Probe construction
failure records a structured `eager_reference` blocker. Successful
installation removes that blocker while preserving `graph_eager`.

## Error Boundaries

- Reference helper replacement is process-local and restored in `finally`.
- Every sequence is released in `finally`.
- Active proposal transactions are aborted before release.
- A missing capture, shape/dtype/device mismatch, incomplete step count,
  remaining physical allocation, or non-finite difference fails the probe.
- No eager retry is introduced and CUDA Graph replay behavior is untouched.

## Validation

Local CPU-focused tests use a deterministic tensor-writing MTP fixture whose
forward path calls the same production attention entry points as the real
module. Tests must first fail because the probe and runtime installation are
absent, then pass after the minimal implementation.

The remote real-checkpoint gate must record:

```text
eager_reference_argmax_equal=true
eager_reference_max_abs_diff=<finite nonnegative value>
backend_failures=[graph_eager only]
status=FAIL
promotion_classification=NOT_PROMOTABLE
```

This is exact-greedy parity evidence for the stated TP1 domain. It is not
CUDA Graph, TP4, KV-offload, long-context, second-model, or performance
evidence.

