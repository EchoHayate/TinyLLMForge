# ModelRunner Authorized Local Checkpoint Loader Boundary Design

## Objective

Add a runtime-local, dependency-injected checkpoint candidate loader boundary
that accepts only bounded metadata, requires an exact authorization digest,
and publishes a candidate only after complete load and provenance validation.

This gate does not implement or execute the real checkpoint worker.

## Why Dependency Injection Is Required

The current Qwen3.5 runtime contains:

- exact packed model and state ownership factories;
- checkpoint binding and streamed/tiled loading primitives;
- verified candidate provenance;
- local one-shot candidate publication;
- all-rank zero-payload candidate binding.

It does not yet contain a complete production attention-backend and
rank-specific candidate factory suitable for constructing the real model inside
`ModelRunner`. The existing
`tools/qwen35_real_checkpoint_load_worker.py` intentionally rejects execution,
and its authorization gate permits implementation only, not payload access.

The runtime boundary therefore receives an explicitly installed local loader.
Future authorized bootstrap code owns construction of that loader.

## Request Contract

Create:

```text
tinyvllm/models/qwen35_checkpoint_worker.py
```

with:

```python
@dataclass(frozen=True)
class Qwen35CheckpointCandidateLoadRequest:
    checkpoint_dir: str
    model_fingerprint: str
    max_tensor_bytes: int
    authorization_sha256: str
```

Validation requires:

- exact class, not a subclass;
- absolute normalized checkpoint path;
- UTF-8 path length from 1 through 4096 bytes;
- no NUL byte;
- canonical lowercase SHA256 model fingerprint;
- positive integer `max_tensor_bytes`;
- canonical lowercase SHA256 authorization digest.

The request contains no tensor, model owner, callable, config object, or
unbounded collection.

## Loader Installation

`ModelRunner` adds:

```python
install_qwen35_checkpoint_candidate_loader(
    loader,
    *,
    authorization_sha256,
)
```

The loader must be callable. Installation is explicit and single-assignment.
An exact repeat requires the same callable object and authorization digest.
Replacement or partial installation state fails closed.

`ModelRunner.__init__()` initializes the loader and authorization slots to
`None` but does not install a loader.

## Local Load and Publish

`ModelRunner` adds:

```python
load_and_publish_qwen35_checkpoint_candidate(request)
```

The method:

1. validates the exact request before calling the loader;
2. requires a coherent installed loader and authorization digest;
3. requires the request authorization digest to match installation;
4. requires the local publication slot to be empty;
5. calls `loader(request)`;
6. requires an exact `Qwen35LoadedCheckpointCandidate`;
7. requires candidate fingerprint to equal the request fingerprint;
8. publishes through the existing atomic one-shot slot;
9. returns a fixed result row.

Expected validation or load failures return:

```text
status=error
model_fingerprint=""
detail="TypeName: message"
```

and leave the publication slot empty. The detail is UTF-8 bounded to 4096
bytes. `BaseException` is not converted.

Success returns:

```text
{
    "participant_id": int,
    "operation": "load_checkpoint_candidate",
    "status": "published",
    "model_fingerprint": str,
    "detail": "",
}
```

An exact repeat after successful publication returns `published` only when the
slot candidate fingerprint and authorization request match the original
completed load configuration. Different requests fail without replacing the
candidate.

## Aggregate State

ModelRunner stores completed local load configuration only after slot
publication succeeds:

```text
(
    checkpoint_dir,
    model_fingerprint,
    max_tensor_bytes,
    authorization_sha256,
)
```

The completed request object is also retained. Exact repeat revalidates the
slot candidate and configuration before returning.

## Security and Execution Boundary

This gate:

- imports no `tools` authorization module;
- opens no checkpoint file;
- performs no SSH or network action;
- initializes no CUDA state;
- adds no Engine command;
- does not modify `tools/qwen35_real_checkpoint_load_worker.py`;
- does not change the canonical schema-v2 `NO_GO`;
- does not authorize real payload access.

## Tests

Dependency-light tests prove:

- strict bounded request validation;
- default-off loader state;
- exact installation idempotency and replacement rejection;
- request authorization mismatch rejection before loader call;
- loader failure leaves publication and completion state empty;
- invalid candidate/fingerprint leaves state empty;
- success publishes the exact candidate and stores completion only afterward;
- exact repeat has zero loader calls;
- conflicting repeat fails without replacement;
- `LLMEngine.step()` and Engine source contain no load invocation.

## Claim Boundary

Passing proves a safe local runtime API boundary for a future authorized
rank-specific loader. It does not prove a production candidate factory, real
checkpoint payload loading, multi-process execution, automatic publication,
or any memory or speed improvement.
