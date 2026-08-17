# Qwen3.5 TP4 Correctness Campaign Preparation Bundle Design

## Goal

Provide one pure-local preparation entry that generates and mutually binds the
three child execution plans, their active single-use authorizations, the
campaign plan, and the campaign authorization.

The output is an immutable `READY` preparation bundle for a later explicitly
approved execution. Preparation performs no remote command, consumes no
authorization, and proves no correctness or performance result.

## Current Gap

The repository has complete protocol APIs for:

- root-logit plan, authorization, executor, and receipt;
- cached-continuation plan, authorization, executor, and receipt;
- Engine plan, authorization, executor, and receipt;
- campaign plan, authorization, callbacks, executor, and receipt.

The operator still has to call seven builders in the correct order and invent
all path relationships:

1. root child plan;
2. cached child plan;
3. Engine child plan;
4. three child authorizations;
5. campaign plan;
6. campaign authorization.

A path mistake is currently detected only when a later builder or executor
reopens the files. There is no single read-only manifest proving that the
prepared inputs form one closed campaign transaction.

## Considered Approaches

### A. Documentation-Only Command Recipe

Document seven Python invocations and path conventions.

Rejected because the operator remains responsible for manually preserving
plan order, nonces, output paths, and SHA relationships. A command recipe is
not independently verifiable evidence.

### B. Direct Execution CLI

Create one CLI that prepares and immediately runs the campaign.

Rejected because it would combine preparation authority with execution
authority, require runner ownership, and make accidental remote execution
possible. The existing campaign executor intentionally requires explicit
injected runners.

### C. Pure-Local Preparation Bundle

Create one builder/verifier that uses the existing production builders and
authorization modules, publishes a final preparation manifest only after all
artifacts independently verify, and otherwise removes the incomplete output.

Selected because it closes the operational assembly gap without adding a
subprocess surface or weakening the authorization boundary.

## Inputs

The builder requires explicit values for every environment-specific choice:

```text
repo_root
output_dir
campaign_tag
root_run_tag
cached_run_tag
engine_run_tag
configuration_path
source_inventory_path
remote_model_dir
remote_model_manifest
root_authorization_nonce
cached_authorization_nonce
engine_authorization_nonce
campaign_authorization_nonce
```

There are no defaults for model paths, remote model paths, run tags, ports,
cache limits, timeouts, or nonces.

The existing Engine authority configuration remains the single shared
configuration for cached-continuation and Engine plans. This is valid because
both protocols bind the same canonical workload manifest and source identity.
The preparation builder does not create or modify that configuration.

The three child run tags must be pairwise distinct. The four authorization
nonces must be pairwise distinct. Existing production validators still apply
their own safe-tag and safe-nonce rules.

## Output Layout

The output directory is fixed:

```text
<output_dir>/
  children/
    tp4_root_logit/
      plan/
        remote_execution_plan.json
      authorization.json
    cached_continuation/
      plan/
        remote_execution_plan.json
        ...
      authorization.json
    engine_correctness/
      plan/
        remote_execution_plan.json
        ...
      authorization.json
  campaign/
    plan/
      campaign_plan.json
    authorization.json
  preparation_manifest.json
```

Expected future receipt/failure outputs are bound under the same root and
remain absent after preparation:

```text
runtime/<child>/consumed_authorization.json
runtime/<child>/execution_receipt.json
runtime/<child>/execution_failure.json
runtime/campaign/consumed_authorization.json
runtime/campaign/campaign_receipt.json
runtime/campaign/campaign_failure.json
adapter/
bundle/
```

Authority directories are not invented by the preparation layout. They are
derived from each verified child plan:

```text
root:
  stage_inputs.verify.local_artifact_dir
cached:
  <cached plan directory>/downloaded_cached_authority
Engine:
  <Engine plan directory>/downloaded_authority
```

The derived absolute path must also be present in the child plan's frozen
local verifier binding. This makes the later real-authority adapter consume
the exact directory produced by the child executor.

`preparation_manifest.json` is written last. Its presence is the only
publication signal. If any build or verification fails, the entire output
directory is recursively removed.

## Builder

The builder performs:

1. reject an existing output directory;
2. validate pairwise-distinct run tags and nonces;
3. create the root child plan with the frozen root source identity;
4. create cached and Engine child plans from the same explicit configuration
   and source inventory;
5. reopen all three plans with their production verifiers;
6. create each active child authorization with its production authorization
   module;
7. construct the campaign child rows from fixed future runtime paths;
8. build and reopen the campaign plan with all production child verifiers;
9. create and validate the campaign authorization;
10. construct the preparation manifest from reopened on-disk artifacts;
11. validate the complete preparation bundle;
12. atomically write the final manifest.

The builder never imports or invokes:

```text
subprocess
os.system
os.popen
shell=True
exec
campaign callbacks
campaign executor
child executors
```

## Preparation Manifest

The manifest schema is
`qwen35.tp4-correctness-campaign-preparation.v1`.

It binds:

- `classification=READY`;
- canonical output root;
- campaign tag;
- exact remote target and execution environment;
- fixed child and campaign stage order;
- each child plan path and SHA;
- each child active authorization path and SHA;
- each child expected authority, consumed authorization, receipt, and failure
  path;
- campaign plan path and SHA;
- campaign active authorization path and SHA;
- campaign expected consumed authorization, receipt, and failure path;
- adapter and bundle output directories;
- source and model identities from the verified child plans;
- `execution_performed=false`;
- `benchmark_execution_authorized=false`;
- an explicit no-correctness/no-performance claim boundary.

The manifest does not require the original external configuration or source
inventory to remain present. Cached and Engine plan directories already
contain frozen configuration, workload, source archive, and verifier inputs.

## Independent Verifier

`verify_preparation_bundle(path)`:

1. requires a regular non-symlink manifest;
2. requires the exact closed manifest schema;
3. requires preparation-owned plans, authorizations, inputs, receipts, and
   failure paths to remain under the preparation root;
4. reopens all child plans with production verifiers;
5. rehashes all plans and active authorizations;
6. validates each authorization against its child plan;
7. reopens the campaign plan with production child verifiers;
8. validates the campaign authorization against the campaign plan;
9. re-derives each plan-bound authority directory and requires the manifest
   path to match it exactly;
10. requires all future runtime outputs, authority directories, adapter
   output, and bundle output to remain absent;
11. requires exact target, environment, order, source, and model identities;
12. requires both execution and benchmark authorization flags to remain
    false.

Any drift returns an exception; there is no degraded READY state.

## CLI Boundary

The module may expose a CLI for preparation only. Every input is required.
The CLI prints the verified preparation manifest and exits.

It does not accept an execution flag and cannot load campaign callbacks or
any runner.

## Testing

CPU-only tests use synthetic repositories and dependency injection for the
existing builders/verifiers. They require:

- exact deterministic layout and manifest bindings;
- production-like builder call order;
- one shared configuration for cached and Engine child plans;
- pairwise run-tag and nonce rejection;
- rejection of existing outputs and path escape;
- plan, authorization, source, model, target, and environment drift
  rejection;
- rejection if any consumed authorization, receipt, failure, authority,
  adapter, or bundle output exists;
- cleanup after an injected mid-build failure;
- final manifest publication only after full verification;
- no subprocess/default runner/execution import surface.

## Claim Boundary

After this increment:

```text
preparation bundle builder/verifier:
  implemented and CPU-verified
real preparation bundle:
  not necessarily produced without explicit model/configuration inputs
real campaign execution:
  not run
real three-authority v2 bundle:
  absent
canonical benchmark:
  not run
performance/cache/memory/quality/accuracy benefit:
  unmeasured and not claimable
```
