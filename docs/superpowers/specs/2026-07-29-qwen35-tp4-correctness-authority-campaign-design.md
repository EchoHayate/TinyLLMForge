# Qwen3.5 TP4 Correctness Authority Campaign Design

## Goal

Provide one receipt-bound coordinator that executes the existing real
root-logit, cached-continuation, and Engine correctness authority protocols in
a fixed order and publishes a self-contained v2 correctness prerequisite
bundle only after all three child receipt chains independently validate.

This is an execution-control improvement, not correctness or performance
evidence by itself.

## Current Gap

The repository already has:

- immutable plan, single-use authorization, executor, receipt, and verifier
  modules for all three correctness authorities;
- a real-authority adapter that accepts three completed receipt chains;
- a v2 prerequisite builder and runtime validator.

The remaining operational gap is orchestration. An operator must manually
assemble three child plans and authorizations, invoke three different
executors, map their downloaded authority directories into
`RealAuthorityRun`, invoke the adapter, invoke the builder, and independently
validate the resulting bundle. That manual sequence is not represented by
one immutable plan or one receipt.

## Considered Approaches

### A. Shell Script

Run the three CLIs and builder commands from Bash.

Rejected because shell state would become an unaudited second authorization
surface. It would not canonically bind child plan identities, could publish a
partial bundle, and would duplicate failure handling.

### B. New Monolithic Remote Runner

Rewrite all root, cached, and Engine SSH commands in one new runner.

Rejected because the existing child protocols already own mature resource,
source, authorization, and receipt semantics. Rewriting them would create a
second subprocess owner and a large correctness/safety review surface.

### C. Receipt-Bound Semantic Coordinator

Create an immutable campaign plan that binds the three existing child plans,
consume one campaign authorization, invoke dependency-injected child
executors sequentially, verify each child receipt, adapt all three completed
runs, build the v2 bundle, independently validate it, and publish one campaign
receipt.

Selected because it reuses every existing protocol and adds only the missing
cross-authority transaction boundary.

## Architecture

The campaign has four modules:

```text
qwen35_tp4_correctness_authority_campaign_plan.py
qwen35_tp4_correctness_authority_campaign_authorization.py
qwen35_tp4_correctness_authority_campaign_executor.py
qwen35_tp4_correctness_authority_campaign_receipt.py
```

The fixed campaign stage order is:

```text
root_logit
cached_continuation
engine_correctness
adapt_authorities
build_bundle
verify_bundle
```

Execution is strictly serial. All three child protocols target the same
four-GPU resource pool, so parallel child execution is forbidden.

## Immutable Campaign Plan

The plan binds:

- campaign schema and safe campaign tag;
- repository root and exact
  `KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian`;
- exact remote target `sitian@10.232.195.203`;
- fixed campaign stage order;
- canonical path and SHA256 of each verified child execution plan;
- child authority name, run tag, source tree SHA, model manifest SHA, and
  expected downloaded authority directory;
- expected active authorization, consumed authorization, child receipt, and
  child failure paths;
- adapter verification-output directory;
- final v2 bundle directory and
  `correctness_prerequisites.json` path;
- explicit `benchmark_execution_authorized=false`.

The plan builder requires all child plans to exist and pass their production
plan verifiers. It rejects pre-existing campaign outputs, child receipts,
child failures, consumed child authorizations, adapter output, or bundle
output.

## Campaign Authorization

The authorization binds the canonical campaign-plan SHA, campaign tag, exact
child plan SHAs/order, output paths, remote target, execution environment,
and a safe operator nonce.

Consumption follows the existing claim-before-rewrite rule:

1. validate the active authorization;
2. require active and consumed paths in one directory;
3. atomically rename active to consumed;
4. rewrite the claimed document with `consumed=true`;
5. never recreate the active path.

The campaign authorization does not replace child authorizations. It
authorizes the cross-authority transaction; each child executor must still
consume and validate its own authorization.

## Executor

The executor owns no subprocess and has no default callbacks. It requires:

```text
root_execute(child)
cached_execute(child)
engine_execute(child)
adapt_authorities(runs, verification_output_dir)
build_bundle(authorities, output_dir)
verify_bundle(prerequisite_path)
```

Before the first callback it:

- verifies the campaign plan;
- rejects every pre-existing output/failure/receipt target;
- requires the exact execution environment;
- consumes the campaign authorization.

For each child stage, it requires the callback result to be a dictionary with
`classification=PASS`, then verifies the child plan, consumed authorization,
receipt, downloaded authority directory, run tag, source identity, and model
identity through the child production receipt verifier.

After all three child stages pass, it constructs exactly three
`RealAuthorityRun` rows, invokes the existing adapter, invokes the existing
v2 builder, and requires `validate_prerequisites()` to return
`classification=PASS` and `authorized=true`.

Any failure stops the sequence. The executor writes bounded,
prefix-preserving FAILED evidence and never publishes a campaign PASS
receipt. Existing child artifacts remain immutable evidence and are not
deleted.

## Campaign Receipt

The PASS receipt binds:

- canonical campaign plan and consumed-authorization SHA;
- exact six-stage order;
- canonical result SHA for every completed stage;
- each child plan, consumed authorization, receipt, downloaded authority
  directory, source, model, and run tag;
- adapter output inventory and SHAs;
- final prerequisite path and SHA;
- final bundle inventory;
- independent prerequisite validation summary;
- `benchmark_execution_authorized=false`.

Receipt verification reopens every referenced regular non-symlink file and
rehashes it. A summary-only receipt is invalid.

## Failure Semantics

Failure evidence records:

- campaign plan and consumed-authorization identities;
- failed stage;
- exact completed-stage prefix;
- canonical completed-stage results;
- bounded error text.

The following never happen after a failure:

- execution of later child authorities;
- adapter publication;
- bundle publication;
- benchmark authorization.

## Security and Resource Boundary

- No new subprocess import or process owner.
- No default command/stage runner.
- Exact remote target and Kerberos environment remain frozen.
- Child resource guards remain authoritative and unmodified.
- No child source inventory or frozen root source is rewritten.
- Campaign code is local-only and excluded from remote source archives.
- The campaign cannot launch the canonical benchmark.

## Testing

CPU-only tests use dependency injection and synthetic complete authority
fixtures. They require:

- exact plan schema and child-plan SHA binding;
- rejection of unsafe tags, missing/drifted child plans, existing outputs,
  wrong target/env, and non-serial order;
- single-use campaign authorization;
- consume-before-first-callback;
- exact callback order;
- stop-on-first-failure and prefix-preserving FAILED evidence;
- rejection of missing/unconsumed/drifted child receipts;
- no adapter call before all three child receipts pass;
- no bundle call before adapter success;
- no receipt before independent bundle authorization;
- exact final bundle and receipt SHA binding;
- preservation of `benchmark_execution_authorized=false`;
- AST/source-contract proof that the coordinator owns no subprocess surface.

No test may import Torch, Transformers, CUDA, construct an Engine, connect to
SSH, or execute a GPU workload.

## Completion Boundary

This design is complete when the campaign protocol is implemented and
CPU-verified. The long-term inference objective remains incomplete until:

1. the three real receipt-bound authorities are produced on the approved
   remote target;
2. the real v2 bundle independently validates;
3. the canonical 70-case benchmark runs;
4. real latency, throughput, physical cache, CUDA memory, and accuracy
   evidence is available.

