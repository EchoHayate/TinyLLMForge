# Autoregressive Draft TP4 Source-Bound Bundle Design

Date: 2026-08-15

Repository: `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`

## Objective

Upgrade the existing dependency-light learned-drafter TP4 gate from a
producer-only JSON receipt to a schema-v2 authority bundle that can be
independently verified from archived source.

The bundle must make accepted-prefix identity reconstructable, freeze the
exact execution/verifier source inventory, publish a deterministic
`source.tar`, load the verifier from that archive, and atomically publish only
after current-source and archived-source verification agree.

This is evidence-layer work only. It does not authorize or claim GPU
execution, loaded-checkpoint parity, Proposal-KV movement, performance, or
Phase 1 promotion.

## Existing Boundary

`tools/autoregressive_draft_tp4_engine_gate.py` currently:

- runs target-only and independent-drafter TP4 cases for batch 1 and 4;
- validates exact output parity and four-rank direct-allocator snapshots;
- retains no-movement, no-performance, and no-promotion boundaries; and
- atomically writes one JSON file.

The JSON does not bind its source, does not carry an independent verifier
receipt, and does not identify each accepted prefix strongly enough to
reconstruct its prompt, generation step, and exact accepted token IDs.

## Considered Approaches

### 1. Extend the gate with a source-bound bundle layer

Add schema-v2 accepted-prefix rows, a standard-library-only verifier, and a
deterministic multi-file publisher next to the existing gate. Preserve the
legacy single-JSON CLI while adding `--output-dir` for authority bundles.

This is selected because it keeps runtime code unchanged, reuses the existing
gate validation, and makes the authority path explicit without coupling it to
native-MTP or generic n-gram campaign formats.

### 2. Reuse the generic speculative TP4 campaign bundle unchanged

This would inherit n-gram-specific cells, model identity, and result schema.
It is rejected because the learned drafter has two checkpoints, independent
tokenizer identity, rank snapshots, and accepted Proposal-KV evidence.

### 3. Publish only a manifest beside the current JSON

This would hash current files but still verify through the current checkout.
It is rejected because verifier drift remains possible and publication would
not prove that the archived source can reproduce the verification decision.

## Schema-v2 Accepted-Prefix Identity

Every learned acceptance event must contain:

```text
event_index
step_index
sequence_id
prompt_index
prompt_token_ids
output_token_count_before_step
proposal_token_ids
accepted_prefix_count
accepted_prefix_token_ids
```

The producer derives `accepted_prefix_token_ids` from
`speculative_accepted_draft_token_ids_by_seq` when present and otherwise from
the exact proposal prefix. The validator requires both representations to be
identical.

Sequence IDs must map bijectively to prompt indices within a case. The
adapter establishes the mapping from the first observed sequence-ID set and
fails closed if a later event introduces an unknown sequence ID, changes the
mapping, or reports an output length that moves backwards.

## Frozen Source Inventory

The schema-v2 bundle freezes this explicit source closure:

```text
tinyvllm/__init__.py
tinyvllm/llm.py
tinyvllm/config.py
tinyvllm/sampling_params.py
tinyvllm/engine/llm_engine.py
tinyvllm/engine/model_runner.py
tinyvllm/engine/model_runner_command_ack.py
tinyvllm/engine/autoregressive_draft_registration.py
tinyvllm/engine/autoregressive_draft_tp.py
tinyvllm/engine/autoregressive_draft_executor.py
tinyvllm/engine/qwen3_draft_backend.py
tinyvllm/engine/qwen3_draft_proposal_kv.py
tinyvllm/engine/proposal_kv_allocator.py
tinyvllm/engine/proposal_kv_cache.py
tinyvllm/engine/proposal_kv_lifecycle.py
tinyvllm/engine/proposal_kv_residency.py
tinyvllm/engine/speculative_proposal_executor.py
tinyvllm/engine/speculative_runtime.py
tinyvllm/engine/speculative_selection.py
tinyvllm/engine/tensor_parallel_greedy.py
tinyvllm/models/qwen3.py
tinyvllm/speculative/adapter.py
tinyvllm/speculative/batch_runtime.py
tinyvllm/speculative/verifier.py
tinyvllm/utils/context.py
tinyvllm/utils/loader.py
tools/autoregressive_draft_tp1_engine_gate.py
tools/autoregressive_draft_tp4_engine_gate.py
tools/autoregressive_draft_tp4_local_gate.py
tools/verify_autoregressive_draft_tp4_engine_gate.py
```

Every path must be a sorted POSIX relative path naming one regular,
non-symlink file below the source root.

## Deterministic Archive

`source.tar` contains exactly the frozen inventory. Every member uses:

```text
mode = 0644
uid = 0
gid = 0
uname = ""
gname = ""
mtime = 0
```

Members are sorted by path. The archive rejects absolute paths, empty or dot
components, `..`, duplicates, symlinks, hard links, directories, devices,
and FIFOs.

`source_manifest.json` records:

```text
schema_version
source_tree_sha256
source_files[path] = sha256
artifacts.result.json = sha256
artifacts.source.tar = sha256
```

The source-tree digest uses length-delimited path and payload framing, so it
does not depend on tar metadata.

## Independent Verification

`tools/verify_autoregressive_draft_tp4_engine_gate.py` is standard-library
only. It:

1. validates `result.json` through an explicitly loaded archived gate module;
2. validates manifest shape and all SHA-256 values;
3. validates every tar member before writing any extracted file;
4. reads each member payload, checks its expected SHA-256, and then writes it
   into a fresh temporary source root without `extractall()`;
5. loads the verifier and gate dependencies from the extracted root;
6. verifies the result against that extracted `source_root`; and
7. returns a bounded canonical PASS/FAIL receipt.

The publisher first runs the verifier from the current checkout, then safely
extracts `source.tar`, loads the verifier from the extracted archive, and
requires the two canonical receipts to match exactly.

## Atomic Publication

The bundle publisher creates a sibling temporary directory and writes:

```text
result.json
source.tar
source_manifest.json
verify.json
```

It refuses pre-existing success or `.failed` destinations. A PASS bundle is
published with one exclusive directory rename. Any failure moves the complete
temporary directory to `<output>.failed`; no partial success directory is
published.

The legacy `--output <result.json>` path remains available for compatibility.
The new `--output-dir <directory>` path is the authority-bundle surface.

## Validation Boundary

Dependency-light tests must prove:

- accepted-prefix rows are reconstructable and fail closed on drift;
- the source inventory is exact and every named file exists;
- two archives built from identical bytes are byte-identical;
- tar metadata is normalized;
- unsafe, duplicate, unexpected, missing, or hash-mismatched members fail;
- verification loads the archived verifier after current-source verification;
- verifier receipts must match;
- failed publication preserves `.failed` evidence;
- successful publication is atomic and refuses replacement; and
- loaded parity, movement, performance, and Phase 1 remain not established.

