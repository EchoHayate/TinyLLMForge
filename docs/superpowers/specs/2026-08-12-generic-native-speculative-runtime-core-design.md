# Generic Native Speculative Runtime Core Design

## Objective

Move source-agnostic native speculative orchestration out of
`tools/profile_ngram_commit.py` into `tinyvllm/speculative/runtime.py`.

The runtime core must own:

- verifier plan construction;
- speculative KV transaction lifecycle;
- first-target and tail callback ordering;
- greedy prefix acceptance;
- EOS and output-budget truncation;
- accepted-token commit and rejected-suffix rollback;
- phase-labelled failure reporting.

It must not know model classes, draft sources, CUDA tensors, KV-offload
topology, profiler timing, or model names.

## Callback Boundary

```python
def execute_native_speculative_step(
    *,
    block_manager,
    seq,
    draft_tokens: list[int],
    eos_token: int,
    run_first_target: Callable[[], int],
    prepare_tail: Callable[
        [SpecVerifyPlan, tuple[int, ...]],
        object,
    ],
    run_tail: Callable[
        [object],
        NativeTailResult,
    ],
) -> NativeSpeculativeStepResult
```

`prepare_tail` and `run_tail` are skipped for K=1.

`NativeTailResult` carries:

```python
target_tokens: tuple[int, ...]
metadata: object | None
auxiliary: object | None
```

The runtime treats metadata and auxiliary evidence as opaque. The profiler
may use them for `SpecVerifyMetadata`, oracle KV snapshots, timing, or debug
payloads.

## Result

`NativeSpeculativeStepResult` carries:

- plan;
- target tokens including the first target;
- greedy accepted count before EOS/budget truncation;
- final accepted tokens;
- truncation flags;
- reserved, committed, and released block IDs;
- opaque tail metadata and auxiliary payload.

## Failure Model

`NativeSpeculativeStepError` records:

- phase: `reserve`, `first_target_decode`, `verify_prepare`,
  `tail_forward`, `kv_materialize`, `acceptance`, or `metadata_commit`;
- original cause;
- optional rollback error.

After transaction begin, every failure attempts rollback exactly once.
Rollback failure is preserved without hiding the original cause.

## Validation

Before transaction begin:

- draft tokens must be a non-empty list of integer, non-boolean token IDs;
- EOS must be an integer, non-boolean token ID;
- required callbacks must be callable.

After callbacks:

- first target must be an integer, non-boolean token ID;
- tail target count must equal `plan.query_len`;
- every tail target must be an integer, non-boolean token ID.

## Scope Boundary

This core remains single-sequence. Batch scheduling, MTP adapters, real KV
offload, CUDA Graphs, and performance evidence are later phases.
