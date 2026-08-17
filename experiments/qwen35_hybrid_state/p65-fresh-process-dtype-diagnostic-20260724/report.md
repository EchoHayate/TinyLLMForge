# Qwen3.5 p65 Fresh-Process Dtype Diagnostic

## Scope

This is a targeted diagnostic of the exact p65 decision boundary that caused
the completed schema-v2 canonical run to classify `NO_GO`.

It uses:

- source commit `3217895019a26154270db40c432495c7657abcb1`;
- model revision `15852e8c16360a2fea060d615a32b45270f8a8fc`;
- the same deterministic 65-token prompt, with SHA256
  `2391c5bbc31e842e8c362e591458d05541b1566409f03672d192fe6a9702a264`;
- three fresh BF16 processes and two fresh FP32 processes;
- `sitian@10.232.195.203`, GPU0, and unique port pairs per process.

This diagnostic does not modify or relabel the immutable canonical evidence.

## BF16 Result

All three fresh processes were byte-identical:

- decoded tokens:
  `[62, 198, 248044, 2, 16, 15, 15, 15]`;
- all eight cached full-logit hashes matched across processes;
- step 5 at sequence length 70 reproduced:
  - actual top-2 token IDs: `[15, 17]`;
  - actual top-2 logits: `[13.8125, 13.8125]`;
  - actual winner margin: `0.0`;
  - oracle top-2 token IDs: `[15, 17]`;
  - oracle top-2 logits: `[13.75, 13.75]`;
  - oracle winner margin: `0.0`.

The tie is therefore deterministic across fresh BF16 processes. It is not an
incidental process-level numerical instability.

## FP32 Result

Both fresh FP32 processes were byte-identical:

- decoded tokens:
  `[62, 198, 248044, 271, 2, 220, 17, 15]`;
- all eight cached full-logit hashes matched across processes;
- step 5 at sequence length 70 produced:
  - actual top-2 token IDs: `[220, 2972]`;
  - actual winner margin: `1.6912260055541992`;
  - oracle winner margin: `1.6912221908569336`;
  - allclose violation count: `0`;
  - maximum allclose-scaled error: `0.4578429162502289`.

The FP32 continuation diverges from the BF16 continuation before the target
step and does not contain the zero-margin decision boundary.

## Conclusion

The p65 canonical rejection is a stable BF16 execution decision boundary:

- it is reproducible across fresh BF16 processes;
- it is absent from the FP32 diagnostic path;
- it is not caused by transient process noise;
- it is not a universal model-semantic tie independent of execution dtype.

The completed canonical status remains `NO_GO`. This result does not establish
TinyLLMForge native Qwen3.5 support, compression, quality retention, latency,
throughput, speedup, or physical-memory reduction.
