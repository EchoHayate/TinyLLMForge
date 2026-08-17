# Qwen3.5 Checkpoint Weight-Name Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CPU-only immutable planner that classifies every verified Qwen3.5 checkpoint index key into a complete language-model load target or an explicit visual/MTP skip without opening tensor shards or mutating a model.

**Architecture:** Add one dependency-light Qwen3.5 checkpoint module that validates config topology and index grammar, generates the exact expected language-model source names, maps them to canonical TinyLLMForge logical targets, and publishes frozen tuples only after full coverage succeeds. Add one executable test suite that drives the public API with synthetic and official-topology fixtures and verifies fail-closed behavior.

**Tech Stack:** Python 3.9/3.12, standard-library dataclasses, `collections.abc.Mapping`, `pathlib.PurePosixPath`, dependency-light executable tests.

## Global Constraints

- Work only in `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Do not stage, commit, merge, or delete untracked experiment evidence.
- Do not open `.safetensors` shards, materialize tensors, start GPU work, or run remote compute.
- Keep production `ModelRunner` fixed to `Qwen3ForCausalLM`.
- Keep `LLMEngine.step()` and Scheduler Qwen3.5 admission unwired and fail-closed.
- Do not create or mutate a `HybridStateTensorPool`.
- Preserve the immutable Qwen3.5 schema-v2 canonical `NO_GO`.
- A language-model source must appear exactly once in the load plan.
- A checkpoint source must appear exactly once across loads and skips.
- Visual and MTP entries are explicit skips; unknown scopes are errors.
- This session intentionally omits all commit steps despite the generic skill template.

---

### Task 1: Write the Failing Weight-Plan Contract Suite

**Files:**
- Create: `tools/test_qwen35_checkpoint_weight_name_contract.py`

**Interfaces:**
- Consumes: no production implementation yet.
- Produces test expectations for:

```python
build_qwen35_checkpoint_weight_plan(
    hf_config,
    index_payload: Mapping[str, object],
) -> Qwen35CheckpointWeightPlan
```

- [x] **Step 1: Create dependency-light module loading and fixtures**

The test must load `tinyvllm/models/qwen35_checkpoint.py` with
`importlib.util.spec_from_file_location`. Define:

```python
def _config(layer_types=("linear_attention", "full_attention")):
    text_config = types.SimpleNamespace(
        num_hidden_layers=len(layer_types),
        layer_types=layer_types,
        tie_word_embeddings=True,
    )
    return types.SimpleNamespace(text_config=text_config)
```

Define fixture helpers that generate exact root/shared/linear/full source
names, attach a single safe shard, and optionally add visual/MTP keys.

- [x] **Step 2: Test exact interleaved mapping**

For a two-layer linear/full topology assert:

```python
plan = build_qwen35_checkpoint_weight_plan(
    _config(),
    _index_for(("linear_attention", "full_attention")),
)
assert len(plan.loads) == 27
assert len(plan.skips) == 0
assert plan.shards == (
    "model.safetensors-00001-of-00001.safetensors",
)
```

Assert the embedding/final norm targets, all layer prefixes, gate/up packed
slots `0` and `1`, linear buffers, and separate full-attention Q/K/V targets.

- [x] **Step 3: Test explicit skip and total coverage**

Add two visual and two MTP keys. Assert:

```python
assert [entry.scope for entry in plan.skips].count("visual") == 2
assert [entry.scope for entry in plan.skips].count("mtp") == 2
assert len(plan.loads) + len(plan.skips) == len(weight_map)
assert {
    entry.source.name for entry in plan.loads
} | {
    entry.source.name for entry in plan.skips
} == set(weight_map)
```

- [x] **Step 4: Test fail-closed language grammar**

Independently verify errors for:

```text
missing required text name
unexpected text name
linear mixer name on full layer
full mixer name on linear layer
layer index outside config
untied embeddings
unknown layer type
layer_types length mismatch
```

Use `_expect_error()` and assert stable message fragments.

- [x] **Step 5: Test malformed index and shard safety**

Verify errors for:

```text
missing/empty/non-mapping weight_map
empty source name
empty shard name
absolute shard path
parent traversal
non-safetensors shard
unknown top-level source scope
non-string source or shard
```

- [x] **Step 6: Test official Qwen3.5-2B topology counts**

Generate the fixed 24-layer schedule:

```python
layer_types = tuple(
    "full_attention" if (index + 1) % 4 == 0
    else "linear_attention"
    for index in range(24)
)
```

Generate 297 unique `model.visual.*` names and 15 unique `mtp.*` names.
Assert:

```python
assert len(plan.loads) == 320
assert len(plan.skips) == 312
assert sum(x.scope == "visual" for x in plan.skips) == 297
assert sum(x.scope == "mtp" for x in plan.skips) == 15
assert len(plan.loads) + len(plan.skips) == 632
```

- [x] **Step 7: Test read-only behavior**

Patch `builtins.open` to raise, call the planner with an already parsed
payload, and assert success. Assert no test object or input mapping is
mutated, and all public plan dataclasses reject field assignment.

- [x] **Step 8: Run RED under Python 3.9 and 3.12**

Run:

```bash
/usr/bin/python3 tools/test_qwen35_checkpoint_weight_name_contract.py
/opt/homebrew/bin/python3.12 tools/test_qwen35_checkpoint_weight_name_contract.py
```

Expected: both fail because
`tinyvllm/models/qwen35_checkpoint.py` does not exist.

### Task 2: Implement the Immutable Source Grammar and Plan

**Files:**
- Create: `tinyvllm/models/qwen35_checkpoint.py`

**Interfaces:**
- Consumes: config object and parsed index mapping.
- Produces:

```python
@dataclass(frozen=True)
class Qwen35CheckpointSource:
    name: str
    shard: str

@dataclass(frozen=True)
class Qwen35CheckpointLoadTarget:
    source: Qwen35CheckpointSource
    target: str
    packed_slot: str | int | None

@dataclass(frozen=True)
class Qwen35CheckpointSkip:
    source: Qwen35CheckpointSource
    scope: str

@dataclass(frozen=True)
class Qwen35CheckpointWeightPlan:
    loads: tuple[Qwen35CheckpointLoadTarget, ...]
    skips: tuple[Qwen35CheckpointSkip, ...]
    shards: tuple[str, ...]
```

- [x] **Step 1: Add frozen public data types**

Use only standard-library imports and `from __future__ import annotations`.
Do not import torch, safetensors, loader code, Engine code, or model modules.

- [x] **Step 2: Add strict config normalization**

Resolve `text_config`, require positive exact integer
`num_hidden_layers`, require an exact-length tuple/list of accepted layer
types, and require `tie_word_embeddings is True`.

- [x] **Step 3: Generate expected source-to-target records**

Generate:

```text
2 root entries
5 shared entries per layer
9 linear-attention entries per linear layer
6 full-attention entries per full layer
```

Represent each expected entry as:

```python
source_name -> (target_name, packed_slot)
```

Use packed slots only for MLP gate/up.

- [x] **Step 4: Validate and classify the index**

Require a non-empty mapping. Validate every source/shard string and safe
relative `.safetensors` path. Classify:

```text
model.language_model.* -> load candidate
model.visual.*         -> visual skip
mtp.*                  -> mtp skip
other                  -> error
```

Compare the complete observed language set with the generated expected set
before constructing plan entries.

- [x] **Step 5: Enforce uniqueness and publish atomically**

Reject duplicate `(target, packed_slot)` pairs. Verify source coverage across
loads/skips. Sort by source name and return frozen tuples plus sorted unique
shards.

- [x] **Step 6: Run GREEN under Python 3.9 and 3.12**

Run:

```bash
/usr/bin/python3 tools/test_qwen35_checkpoint_weight_name_contract.py
/opt/homebrew/bin/python3.12 tools/test_qwen35_checkpoint_weight_name_contract.py
```

Expected:

```text
qwen35 checkpoint weight-name contract tests passed
```

### Task 3: Validate Integration Boundaries and Regressions

**Files:**
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify: `docs/superpowers/plans/2026-07-27-qwen35-checkpoint-weight-name-contract.md`

**Interfaces:**
- Consumes: the completed contract and focused test output.
- Produces: durable gate evidence and checked plan boxes.

- [x] **Step 1: Run focused Qwen3.5 regressions**

Run under Python 3.9 and Python 3.12:

```text
tools/test_qwen35_checkpoint_weight_name_contract.py
tools/test_qwen35_root_model_assembly_factory.py
tools/test_qwen35_transactional_root_causal_lm.py
tools/test_qwen35_native_model_owner_binding.py
```

Expected: all pass.

- [x] **Step 2: Run dependency/static checks**

Run:

```bash
/usr/bin/python3 -m py_compile \
  tinyvllm/models/qwen35_checkpoint.py \
  tools/test_qwen35_checkpoint_weight_name_contract.py
/opt/homebrew/bin/python3.12 -m py_compile \
  tinyvllm/models/qwen35_checkpoint.py \
  tools/test_qwen35_checkpoint_weight_name_contract.py
rg -n 'safe_open|get_tensor|torch\\.load|load_model|cuda|distributed' \
  tinyvllm/models/qwen35_checkpoint.py
git diff --check
git diff --cached --name-only
```

Expected:

```text
both py_compile commands exit 0
forbidden dependency search returns no matches
git diff --check exits 0
staged file list is empty
```

- [x] **Step 3: Verify production wiring remains absent**

Run:

```bash
rg -n 'qwen35_checkpoint|build_qwen35_checkpoint_weight_plan' \
  tinyvllm/engine tinyvllm/utils/loader.py
rg -n 'Qwen3ForCausalLM\\(' tinyvllm/engine/model_runner.py
rg -n 'hybrid prefix reuse requires aligned state snapshot' \
  tinyvllm/engine/scheduler.py
```

Expected:

```text
no Engine/loader integration references
one production Qwen3 constructor remains
the Scheduler fail-closed guard remains
```

- [x] **Step 4: Update handoff**

Append a final section recording:

- verified index hash and `632 = 320 + 297 + 15`;
- public contract interface;
- exact mapping/skip behavior;
- RED and GREEN evidence;
- focused regression/static evidence;
- allowed conclusion and remaining gates;
- unchanged schema-v2 canonical `NO_GO`.

- [x] **Step 5: Mark all plan checkboxes complete and rerun final verification**

Run the complete commands from Steps 1–3 again after documentation changes.
Do not claim completion without fresh outputs.
