# Engine Resumable Prefix Publication Runtime Configuration Design

## Objective

Add one explicit `LLMEngine` entry point that composes the already validated
Qwen3.5 hybrid-prefix restore, canonical identity, Engine publication
coordinator, and Scheduler source-publisher hook configuration.

The entry point must remain default-off and support exact retry after a later
configuration stage fails.

## Interface

`LLMEngine` adds:

```python
configure_qwen35_hybrid_prefix_publication_runtime(
    *,
    model_fingerprint,
    max_entries,
    max_bytes,
    timeout_s,
)
```

On success it returns the stable installed
`Qwen35HybridPrefixSourcePublisher`.

## Aggregate Configuration

The canonical aggregate configuration is:

```text
(
    validated lowercase model_fingerprint,
    max_entries,
    max_bytes,
    float(timeout_s),
)
```

All arguments are validated before any child configuration method is called.
The aggregate configuration is stored only after every child stage succeeds.

If an aggregate configuration is already stored, an exact repeat returns the
existing publisher without calling any child stage, but only after revalidating
that every child slot still matches the aggregate configuration. A different
aggregate configuration or a damaged completed state fails before any child
stage is called.

## Stage Order

For a fresh or resumable exact configuration, the method calls these stages in
strict order:

1. `configure_qwen35_hybrid_prefix_restore(...)`
2. `configure_qwen35_hybrid_prefix_runtime_identity(...)`
3. create and install
   `Qwen35HybridPrefixEnginePublicationCoordinator(self, timeout_s=timeout_s)`
   if no coordinator is installed
4. `install_configured_qwen35_hybrid_prefix_source_publisher()`
5. store the aggregate configuration and stable publisher result

An already installed publication coordinator is reusable only when it targets
this Engine and has the exact requested `timeout_s`. A different installed
coordinator fails closed before restore or identity configuration is called.

## Retry Semantics

Restore configuration, runtime identity configuration, coordinator
installation, and source-publisher installation already provide exact
idempotency. The aggregate method deliberately does not roll back completed
child stages when a later stage fails.

Therefore:

- failure during restore leaves no aggregate configuration;
- failure after restore allows an exact retry to reuse restore state;
- failure after identity allows an exact retry to reuse restore and identity;
- failure while installing the hook allows an exact retry to reuse restore,
  identity, and coordinator;
- aggregate configuration is visible only after all stages succeed.

Conflicting retries fail before invoking any child stage by inspecting existing
aggregate, restore, identity, coordinator, and publisher configuration slots.
Partial child state that is not an exact prefix of the requested configuration
is rejected rather than replaced.

Configuration and object slots are validated as pairs. In particular, a
restore configuration requires its installed restore coordinator, an identity
configuration requires its canonical identity object, and a publisher
configuration requires its publisher object. Aggregate completion likewise
requires the stable aggregate publisher to be the installed source publisher.

## Failure Atomicity

The orchestrator is atomic only at the aggregate-completion boundary. It does
not promise cross-stage rollback because the child stages include acknowledged
all-rank state. Instead it exposes deterministic resumability:

```text
completed exact child stages remain installed
failed or incomplete aggregate state remains unset
exact retry resumes in the same strict order
conflicting retry fails before mutation
```

## Default-Off Boundary

Neither `LLMEngine.__init__()` nor `LLMEngine.step()` calls the orchestrator.
The method must be invoked explicitly by a future correctness-gated runtime
bootstrap.

## Tests

Dependency-light tests prove:

- complete argument validation precedes child calls;
- pre-existing conflicting child state fails before child calls;
- the four stages execute in strict order;
- successful configuration stores and returns a stable result;
- exact completed repeat has zero child-stage side effects;
- exact retry resumes after coordinator-stage failure;
- exact retry resumes after hook-stage failure;
- conflicting retry fails closed after partial or complete configuration;
- `LLMEngine.step()` has no orchestrator reference.

## Claim Boundary

Passing proves explicit, resumable Engine publication-runtime configuration
composition. It does not enable publication automatically, execute a real
checkpoint worker, establish production cache hit rate, reduce measured CUDA
memory, or prove inference speedup.
