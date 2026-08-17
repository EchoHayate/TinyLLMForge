# Qwen3.5 TP4 Real Prerequisite Authority Adapter Design

## Objective

Build a CPU-only adapter that converts complete, independently verified TP4
correctness authority runs into the existing benchmark prerequisite bundle.
The adapter must reject caller-authored summary JSON that is not anchored to
the full authority artifact inventory.

## Scope

The adapter covers:

- TP4 real root-logit correctness;
- TP4 cached-continuation correctness;
- TP4 Engine and ModelRunner correctness;
- cached-continuation and Engine remote execution plan, consumed
  authorization, and PASS receipt binding.

It does not execute SSH, GPU workloads, model loading, Torch, Transformers, or
CUDA. Root-logit currently has no receipt-compatible remote protocol, so the
adapter records that as an explicit remaining authority gap rather than
inventing a receipt.

## Architecture

Create one focused adapter module beside the existing prerequisite builder.
Each input names a complete authority directory. The adapter invokes the
authority's independent verifier directly, derives the canonical artifact
document from the verified directory, writes a canonical independent
verification document, and returns the existing `AuthorityInput` type.

For cached-continuation and Engine authorities, the adapter also validates:

1. the frozen execution plan;
2. the consumed single-use authorization;
3. the execution receipt;
4. the receipt's source, model, workload, run-tag, and plan identities;
5. that the supplied authority directory is the plan's exact downloaded
   local-verification target.

The existing bundle builder remains the final copying and validation boundary.

## Inputs

`RealAuthorityRun` contains:

- authority name;
- complete authority directory;
- run tag;
- optional plan path;
- optional consumed authorization path;
- optional execution receipt path.

Root-logit requires no remote receipt fields. Cached-continuation and Engine
require all three receipt-chain paths.

## Outputs

The adapter writes no data inside the source authority directories. It writes
canonical independent verification JSON into a caller-provided scratch
directory and returns three `AuthorityInput` rows accepted by
`build_prerequisite_bundle`.

## Failure Boundaries

Fail closed on:

- missing, linked, or incomplete authority directories;
- failed independent verification;
- mismatched artifact bytes or schemas;
- missing cached/Engine receipt chain;
- unconsumed or mismatched authorization;
- plan/receipt identity drift;
- authority directory not equal to the plan's frozen downloaded target;
- model, source, workload, or run-tag drift;
- pre-existing adapter output.

## Testing

Tests use complete synthetic directory inventories that pass the production
independent verifiers. They first prove that naked artifact and verification
JSON are insufficient, then cover receipt-chain requirements, directory
binding, identity tamper, deterministic output, and end-to-end compatibility
with the existing prerequisite bundle builder.

## Claim Boundary

Passing these tests proves only that future real correctness runs can be
assembled without dropping their complete-directory and receipt provenance.
It does not create real correctness evidence and does not prove latency,
throughput, cache, memory, compression, quality, or accuracy benefit.
