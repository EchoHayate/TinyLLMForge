# Qwen3.5 Authorized Prepared-Target Loader Adapter Design

## Objective

Bridge the proven rank-specific prepared-target factory to the existing
ModelRunner authorized loader injection point without enabling the real worker,
Engine dispatch, CUDA, or automatic runtime publication.

The adapter produces the exact `Qwen35LoadedCheckpointCandidate` type already
required by `ModelRunner.load_and_publish_qwen35_checkpoint_candidate(...)`.

## Chosen Approach

Create:

```text
tinyvllm/models/qwen35_checkpoint_candidate_loader.py
```

with:

```python
@dataclass(frozen=True)
class Qwen35AuthorizedCheckpointCandidateLoader:
    prepare_target: Callable[
        [],
        Qwen35PreparedCheckpointCandidateTarget,
    ]
    authorization_sha256: str

    def __call__(
        self,
        request: Qwen35CheckpointCandidateLoadRequest,
    ) -> Qwen35LoadedCheckpointCandidate:
        ...


def build_qwen35_authorized_checkpoint_candidate_loader(
    prepare_target,
    *,
    authorization_sha256,
) -> Qwen35AuthorizedCheckpointCandidateLoader:
    ...
```

The loader is installed into ModelRunner through the existing explicit method:

```text
install_qwen35_checkpoint_candidate_loader(...)
```

No ModelRunner or Engine method is changed by this gate.

## Why Fresh Targets

The adapter accepts a zero-argument target provider rather than one already
prepared target. Every load attempt receives a fresh model, binding plan, and
one-shot target.

This is required because a streamed load can fail after assigning some
destinations. Retrying with the same target would either reuse partially
assigned state or fail only because `take()` was already consumed. A fresh
provider preserves the existing ModelRunner retry model: failed publication
leaves the slot empty, and a later invocation can construct a new target.

## Data Flow

```text
bounded request
  -> exact request validation
  -> adapter authorization equality
  -> fresh prepared target provider
  -> exact prepared-target validation
  -> require CPU parameter device
  -> target.take()
  -> existing streamed checkpoint loader
  -> exact Qwen35LoadedCheckpointCandidate
```

The adapter passes:

```text
request.checkpoint_dir
request.max_tensor_bytes
request.model_fingerprint
```

directly to `load_qwen35_fresh_checkpoint_candidate(...)`.

## Authorization Boundary

The adapter stores one canonical lowercase authorization SHA256 and requires
the request to contain the exact same digest before target preparation or
payload access.

This duplicates the ModelRunner installation/request check intentionally as
defense in depth. The adapter can be tested or used directly without silently
bypassing its authorization identity.

This digest is not by itself a model-manifest verifier. Real worker execution
remains governed by the existing source-bound authorization and safety gate.

## Device and Candidate Boundary

The adapter accepts only an exact
`Qwen35PreparedCheckpointCandidateTarget`.

It rejects meta targets before calling the streamed loader. The existing
streamed loader already requires CPU non-meta registered destinations and
returns the exact candidate type required by ModelRunner.

The adapter does not convert tiled candidate wrapper types because ModelRunner
currently requires exact `Qwen35LoadedCheckpointCandidate`. Tiled loading can
be integrated only after defining an explicit common publication candidate
contract; silently coercing the current tiled type is out of scope.

## Failure Semantics

Before delegation, these failures perform no checkpoint payload access:

- invalid request type;
- authorization mismatch;
- non-callable provider;
- provider exception;
- non-exact prepared target;
- meta prepared target.

After delegation begins, the existing streamed loader owns file validation,
source budget enforcement, assignment, and candidate construction. An
exception returns no candidate. The consumed/possibly partially assigned
target is discarded by the caller; exact retry must invoke the provider again.

The adapter stores no completion state and performs no publication itself.

## Tests

Focused tests prove:

- builder validation and immutable callable configuration;
- exact request/authorization validation occurs before provider invocation;
- a fresh provider invocation occurs on every adapter call;
- exact CPU target is passed one-shot to the streamed loader;
- request path, budget, and fingerprint are forwarded exactly;
- meta/invalid targets fail before loader delegation;
- delegated failure returns no candidate and the next attempt receives a fresh
  target;
- success returns the exact existing loaded-candidate object unchanged.

Existing streamed/tiled, prepared-target, ModelRunner authorized-loader,
candidate publication, all-rank binding, authorization, and safety-gate suites
remain regression requirements.

## Alternatives Rejected

### Hold one prepared target in the adapter

Rejected because a failed load can consume or partially assign it, making retry
unsafe.

### Return tiled candidate wrappers

Rejected because ModelRunner requires exact
`Qwen35LoadedCheckpointCandidate`; changing that publication contract is a
separate design.

### Wire the adapter into Engine or the real worker

Rejected because real worker execution remains unauthorized and automatic
runtime enablement is explicitly out of scope.

## Claim Boundary

Passing proves a local, authorization-bound composition from a fresh CPU
prepared target to the existing streamed candidate loader.

It does not authorize or execute the real worker, validate the approved remote
model manifest, load the real checkpoint in this gate, run inference, enable
Engine dispatch, or establish speed, cache, compression, quality, or CUDA
memory benefit.
