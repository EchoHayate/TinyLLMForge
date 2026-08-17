# Qwen3.5 TP4 Cached-Continuation Remote Authority Design

## Goal

Add an independently verifiable and single-use-authorized remote execution
protocol for the standalone cached-continuation authority without changing
the frozen TP4 Engine two-phase authority receipt schema.

## Architecture

Use a separate cached-authority plan and receipt. Reuse only the established
safe primitives:

- deterministic source bundle and source-tree identity;
- exact `sitian@10.232.195.203` SSH target;
- exact `KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian`;
- four-idle-GPU resource guard;
- isolated subprocess adapter;
- single-use authorization consumption;
- bounded command logs and binary package streaming.

The remote guarded command invokes:

```text
run_qwen35_tp4_cached_continuation_authority.py
```

It produces:

```text
cached_continuation_authority/
cached_continuation_independent_verification.json
```

The authority directory remains the existing exact-five closed inventory.

## Plan Contract

The plan binds:

- canonical configuration, source inventory, source tar, model manifest, and
  workload manifest identities;
- exact four GPU indices and dist/master ports;
- remote model paths and unique run root;
- ordered reserve, upload, stage, resource guard, guarded authority, package,
  safe extract, verifier-source preparation, and local verification commands;
- local package, extracted authority, external verification, verifier source,
  receipt, and failure destinations.

The package contains exactly two top-level entries:

```text
cached_continuation_authority
cached_continuation_independent_verification.json
```

## Receipt Contract

The cached receipt does not require Engine `reference_classification` or
`engine_classification`. It requires matching remote and local verification
payloads with:

```text
classification
schema_version
source_tree_sha256
model_manifest_sha256
workload_manifest_sha256
checks
```

It also proves:

- preflight and final resource guard use the same four GPU UUIDs;
- every command matches the frozen plan hash and returns zero;
- the downloaded package has an independently checked SHA and nonzero size;
- single-use authorization identity and nonce are bound into the receipt.

## Safety Boundary

The cached plan, receipt, and executor core contain no subprocess execution.
The existing adapter remains the only `Popen` owner. No module kills remote
processes or weakens the active-compute-process guard.

CPU-safe tests use fake runners only. They do not execute SSH, `scp`,
`nvidia-smi`, Torch, Transformers, CUDA, or a GPU workload.

## Claim Boundary

Completing this protocol proves only that the real cached authority has a
safe executable path. Performance, cache, memory, compression, and accuracy
benefits remain unclaimable until real exact-five artifacts pass independent
verification and the complete correctness prerequisite bundle authorizes the
canonical benchmark.
