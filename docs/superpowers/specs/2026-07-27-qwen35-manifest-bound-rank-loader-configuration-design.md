# Qwen3.5 Manifest-Bound Rank Loader Configuration Design

## Objective

Bind immutable checkpoint metadata identity, one rank's topology, a fresh
state-pool provider, the exact tensor plan, and an attention-backend provider
into one explicit local loader configuration.

This gate constructs the already proven authorized prepared-target loader. It
does not open checkpoint files during configuration, modify Engine, or execute
the real worker.

## Components

Create:

```text
tinyvllm/models/qwen35_checkpoint_loader_configuration.py
```

with two frozen values:

```python
@dataclass(frozen=True)
class Qwen35CheckpointManifestIdentity:
    checkpoint_dir: str
    model_manifest_sha256: str
    config_sha256: str
    index_sha256: str
    config_index_header_sha256: str


@dataclass(frozen=True)
class Qwen35RankCheckpointLoaderConfiguration:
    manifest: Qwen35CheckpointManifestIdentity
    hf_config: object
    tensor_plan: Qwen35CheckpointTensorPlan
    tensor_parallel_size: int
    tensor_parallel_rank: int
    create_pool: Callable[[], HybridStateTensorPool]
    build_attention_backend: Callable
    authorization_sha256: str

    def build_loader(
        self,
    ) -> Qwen35ManifestBoundCheckpointCandidateLoader:
        ...
```

The produced loader is frozen and callable with
`Qwen35CheckpointCandidateLoadRequest`.

## Manifest Identity

`checkpoint_dir` must be an absolute normalized path bounded to 4096 UTF-8
bytes. All four identities must be canonical lowercase SHA256 values.

`model_manifest_sha256` is the runtime `model_fingerprint` passed through the
loaded candidate and publication identity.

The remaining digests are retained for exact configuration equality,
diagnostics, and future worker authorization. This gate accepts already
verified digests; it does not read or hash config, index, header, or payload
files.

## Rank Configuration

The configuration requires:

- an exact `Qwen35CheckpointTensorPlan`;
- a positive TP size and rank in range;
- a callable fresh-pool provider;
- a callable attention-backend provider;
- a canonical authorization SHA256.

`build_loader()` creates no pool and no model. It only captures the immutable
configuration.

On each loader invocation:

1. validate the exact bounded request;
2. require exact checkpoint path, manifest fingerprint, and authorization;
3. call `create_pool()` exactly once;
4. require an exact `HybridStateTensorPool`;
5. call `prepare_qwen35_checkpoint_candidate_target(...)` with
   `parameter_device="cpu"`;
6. delegate the fresh target through
   `Qwen35AuthorizedCheckpointCandidateLoader`;
7. return the exact `Qwen35LoadedCheckpointCandidate`.

## Freshness and Failure

Every invocation constructs a fresh pool and fresh prepared target. A failed
streamed load cannot reuse partially assigned model parameters or hybrid state.

Any request identity mismatch fails before pool creation. Invalid pool,
assembly, binding, backend, or streamed load returns no candidate. The
configuration stores no completion or publication state.

## Alternatives Rejected

### Store a prebuilt pool or target

Rejected because retry after partial assignment must use fresh ownership.

### Re-read manifest/config/index inside runtime configuration

Rejected because this gate is composition-only and must remain payload-zero
and dependency-light. Read-only evidence generation remains in the safety
harness.

### Import constants from `tools/`

Rejected because production modules must not depend on the gate harness or one
specific remote model snapshot.

## Tests

Focused tests prove:

- canonical path/SHA256/TP/callable/type validation;
- builder is allocation-free;
- mismatched path/fingerprint/authorization fail before pool creation;
- each invocation creates exactly one fresh pool;
- exact config/tensor plan/TP/backend/pool are forwarded to the target factory;
- CPU parameter device is forced;
- exact candidate result passes through unchanged;
- provider, target-preparation, and delegated load failures do not retain state
  and a retry creates a different pool.

Regression requires the prepared-target adapter/factory, streamed/tiled
loaders, ModelRunner publication, all-rank binding, real authorization, and
safety-gate suites.

## Claim Boundary

Passing proves explicit local binding of already verified immutable metadata
identities to fresh rank-specific CPU candidate preparation.

It does not verify files against those digests, open payloads during
configuration, authorize or execute the real worker, install the loader in
Engine, run inference, or establish performance or memory benefit.
