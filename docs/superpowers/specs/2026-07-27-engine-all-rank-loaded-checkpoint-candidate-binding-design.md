# Engine All-Rank Loaded Checkpoint Candidate Binding Design

## Objective

Bind one locally loaded, verified `Qwen35LoadedCheckpointCandidate` on every
ModelRunner rank without serializing model owners through the Engine shared
memory command channel.

The operation is explicit, exact-retryable after acknowledged participant
rejection, provenance validating, and absent from `LLMEngine.step()`.

## Why Candidates Are Not Broadcast

`Qwen35LoadedCheckpointCandidate` owns a complete model graph and tensor
storage. The current ModelRunner command transport pickles arguments into a
fixed 1 MiB shared-memory buffer. Broadcasting a candidate would attempt to
serialize model-sized state, violate ownership locality, and exceed the
transport contract.

Each ModelRunner must instead load and publish its own candidate locally.
Engine orchestration broadcasts only a zero-payload public method name.

## Local Publication Slot

`Qwen35HybridModelOwnerPublicationSlot` retains the exact candidate in addition
to its already atomic owner and model fingerprint fields:

```text
slot.candidate
slot.owner
slot.model_fingerprint
```

All three begin as `None` and become visible together only after complete
candidate and owner-graph validation. The slot remains one-shot and has no
clear or replace operation.

`ModelRunner` owns one slot:

```text
qwen35_loaded_checkpoint_candidate_slot
```

and exposes:

```python
publish_qwen35_loaded_checkpoint_candidate(candidate)
```

This is the future local checkpoint-worker handoff boundary. It does not accept
candidate data from Engine transport.

## Zero-Payload Participant Bind

`ModelRunner` exposes:

```python
bind_published_qwen35_loaded_checkpoint_candidate()
```

It reads the exact candidate from the local slot and calls the already proven
atomic `bind_qwen35_loaded_checkpoint_candidate(candidate)`.

The method returns a fixed status row and does not raise for expected candidate
absence or binding conflict:

```text
{
    "participant_id": int,
    "operation": "bind_loaded_checkpoint_candidate",
    "status": "bound" | "error",
    "model_fingerprint": str,
    "layout_fingerprint": str,
    "dtype": str,
    "detail": str,
}
```

Successful exact repeats return `bound`. Expected participant errors return
`error` with empty identity fields and bounded textual detail. This lets all
ranks acknowledge one command and keeps the acknowledgement collector healthy
for an exact retry.

Base exceptions and transport failures are not converted into participant
errors. Existing acknowledgement fail-stop behavior remains authoritative for
timeout, worker death, malformed acknowledgement, or rank-zero failure before
a result row exists.

## Engine Orchestration

`LLMEngine` exposes:

```python
bind_qwen35_loaded_checkpoint_candidates(*, timeout_s)
```

The method validates `timeout_s`, then invokes:

```text
call_model_runner_acknowledged(
    "bind_published_qwen35_loaded_checkpoint_candidate",
    timeout_s=timeout_s,
)
```

It validates exactly one row per rank, exact row fields, outer/inner rank
agreement, exact operation name, supported status, and string detail.

If any rank returns `error`, the Engine raises a deterministic `RuntimeError`
without recording aggregate completion or poisoning the acknowledged command
transport. Ranks that already bound remain idempotently bound. After the
missing/conflicting local rank is corrected, an exact retry may converge.

When every rank returns `bound`, Engine requires homogeneous
`model_fingerprint`, `layout_fingerprint`, and `dtype`. It stores the ordered
rank rows and canonical configuration:

```text
(
    model_fingerprint,
    layout_fingerprint,
    dtype,
    float(timeout_s),
)
```

An exact completed repeat revalidates stored rows and returns them without
dispatching. A different timeout or damaged completed state fails before
dispatch.

## Ordering Boundary

This binding precedes publication-runtime configuration:

```text
local candidate load and publication on every rank
Engine all-rank candidate binding
Engine resumable publication-runtime configuration
```

The publication-runtime orchestrator remains a separate explicit call. This
gate does not combine them or call either path automatically.

## Tests

Dependency-light tests prove:

- slot candidate/owner/fingerprint atomic publication;
- ModelRunner local publication and zero-payload bind;
- missing candidate returns an acknowledged `error` row;
- first bind and exact repeat return `bound`;
- Engine dispatch carries no model candidate argument;
- rank/result schema and homogeneous provenance validation;
- participant error leaves aggregate state unset and permits exact retry;
- completed exact repeat has zero dispatch side effects;
- conflicting/damaged completed state fails closed;
- `LLMEngine.step()` remains disconnected.

## Claim Boundary

Passing proves dependency-light all-rank orchestration over locally published
verified candidates. It does not implement the real checkpoint worker that
loads/publishes those candidates, execute a real multi-process checkpoint
load, enable automatic runtime publication, or prove production memory or
speed improvement.
