# Engine Source Publisher Hook Installation Design

## Objective

Add an explicit, default-off `LLMEngine` method that creates one
`Qwen35HybridPrefixSourcePublisher` and installs its synchronous `publish`
callback at the Scheduler complete-prefill commit hook.

## Interface

`LLMEngine` adds:

```text
install_qwen35_hybrid_prefix_source_publisher(
    *,
    model_fingerprint,
    layout_fingerprint,
    dtype,
)
```

The method returns the installed publisher.

## Installation Contract

The first call:

1. constructs `Qwen35HybridPrefixSourcePublisher(self, ...)`;
2. installs the publisher's stable bound `publish` method through
   `Scheduler.install_prefill_commit_hook()`;
3. stores both the publisher and its hook callable on the Engine;
4. returns the publisher.

The stored hook callable is required because repeated bound-method attribute
access creates distinct method objects. Scheduler installation uses identity
for idempotency.

A repeated call with exactly the same model fingerprint, layout fingerprint,
and dtype returns the existing publisher without reinstalling the hook.
Different configuration fails closed and leaves the original installation
unchanged.

## Default-Off Boundary

`LLMEngine.__init__()` initializes publisher and hook attributes to `None`.
It does not construct or install a publisher.

`LLMEngine.step()` does not install, enable, or directly call the publisher.
Publication occurs only if an external caller explicitly invokes the install
method before scheduling.

## Failure Semantics

Constructor validation errors and Scheduler installation errors propagate.
Engine attributes are assigned only after Scheduler installation succeeds, so
a failed first installation does not leave a partial Engine-visible publisher.

Once installed, publication failures retain the existing Scheduler hook
fail-stop behavior: the current postprocess raises before append/release and
all future scheduling fails closed.

## Tests

Dependency-light tests extract the Engine method and use fake Scheduler and
publisher types to prove:

- default-off attributes;
- construction and exact Scheduler callback installation;
- same-configuration idempotency;
- different-configuration rejection without mutation;
- Scheduler installation failure leaves Engine attributes unset;
- `LLMEngine.step()` contains no publisher installation or direct publication.

The existing source publisher and Scheduler hook suites remain authoritative
for actual capture/transaction behavior and callback ordering.

## Claim Boundary

Passing proves an explicit opt-in Engine-to-Scheduler wiring primitive. It
does not prove automatic enablement, real-model identity derivation,
production publication frequency, cache hit rate, CUDA memory reduction, or
speedup.
