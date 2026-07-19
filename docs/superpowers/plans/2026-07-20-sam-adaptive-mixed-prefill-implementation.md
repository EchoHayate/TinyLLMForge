# SAM Backlog-Adaptive Mixed-Prefill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement disabled-by-default `P4` Scheduler-Adaptive Mixed-prefill (SAM), prove its state machine and evidence contract locally, then run the source-bound remote arrival-load chain before making any performance claim.

**Architecture:** Add five validated configuration fields and keep the controller entirely inside `Scheduler`. The controller samples `len(waiting)` once at the start of each scheduling decision, drives `INACTIVE / ACTIVE / DRAINING`, and reuses one transactional mixed-prefill/decode helper shared with `P3`. Extend the arrival-load policy identity and independent verifier so `P4` is the only promotable candidate while `P3` remains a diagnostic comparator.

**Tech Stack:** Python 3, dataclasses, deque-based scheduler state, TinyLLMForge block manager, JSON/JSONL source-bound artifacts, dependency-light script tests, Bash remote runner, Qwen3-0.6B on `sitian@10.232.195.203`.

## Global Constraints

- Work only in `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`; do not modify `/Users/bytedance/dev/TinyLLMForge`.
- Execute inline in the current session; do not dispatch subagents.
- The feature is disabled by default with `chunked_prefill_adaptive_mixed=False`.
- Use policy name `P4` and stable descriptive name `sam_backlog_adaptive_mixed_prefill`.
- Enter `ACTIVE` after two consecutive eligible observations with `waiting_depth >= 8`.
- Stop new admission after two consecutive eligible observations with `waiting_depth <= 2`.
- Permit at most two consecutive adaptive mixed steps while runnable decode work remains, then force one decode-only yield.
- Never emit an adaptive mixed branch unless the batch contains at least one prefill row and one decode row.
- Never allocate prefill KV blocks, call `may_append()`, or mutate queue ownership before proving that a mixed batch can contain a decode row.
- Keep the existing mixed model-runner and postprocess semantics shared by `P3` and `P4`; do not add a new kernel or attention preparation path.
- Adaptive mode is mutually exclusive with `chunked_prefill_mixed_batch=True` and `kv_offload_mvp0=True`.
- Preserve exact greedy fixed-length outputs, request lifecycle, queue ownership, prefix-cache metadata, and repository-default behavior when adaptive mode is disabled.
- Canonical comparison is exactly `P0 / P3 / P4`: 6 scenarios × 3 policies × 3 repetitions = 54 cases.
- `P3` is diagnostic. Only the independently recomputed `P4` result may determine the top-level canonical classification.
- Every remote model process must receive a unique dynamic `TINYVLLM_DIST_PORT` and `MASTER_PORT`.
- Remote GPU/model execution is only on `sitian@10.232.195.203` with Python `/data00/home/sitian/sitian-workspace01/tllm/env/bin/python` and model `/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B`.
- Use `KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian`; do not change the remote checkout, kill unrelated processes, or clear shared temporary directories.
- Preserve all untracked `experiments/` artifacts and use selective staging; never run `git add -A`.
- Do not update README or `AGENT_HANDOFF_STATE.md` until a source-bound remote result exists.

---

## File Map

- Modify `tinyvllm/config.py`: declare and fail-closed validate all five adaptive fields before model startup.
- Modify `tinyvllm/engine/scheduler.py`: own the adaptive state machine, transactional mixed scheduling, branch labels, counters, and immutable snapshots.
- Modify `tools/test_chunked_prefill.py`: dependency-light TDD coverage for configuration extraction, state transitions, queue/KV invariants, P3 compatibility, and engine observations.
- Modify `tools/arrival_load_gate.py`: freeze `P0/P3/P4`, include adaptive identity fields, strengthen smoke, preserve 54 canonical cases, and make only `P4` promotable.
- Modify `tools/test_arrival_load_gate.py`: prove policy identity, matrix order, resume identity, smoke activation workload, and P4-only promotion.
- Modify `tools/test_arrival_load_driver.py`: prove adaptive controller fields survive unchanged in `queue_before` and `queue_after`.
- Modify `tools/arrival_load_verify.py`: independently validate P4 configuration and reconstruct every controller transition from trace rows.
- Modify `tools/test_arrival_load_verify.py`: synthetic complete P4 artifacts plus fail-closed tampering cases.
- Modify `tools/run_arrival_load_gate_remote.sh`: preserve the source-bound `preflight → smoke → calibration → canonical → local verify` contract and explicit predecessor identities.
- Modify `tools/test_run_arrival_load_gate_remote.py`: assert exact remote host/runtime, immutable staging, predecessor run tags, dynamic ports, and prohibited operations.
- Create remote artifacts under `experiments/arrival_load/<run-tag>/`: never stage or commit them.

### Shared Scheduler Interfaces

The implementation tasks below use these exact scheduler interfaces:

- `ADAPTIVE_MIXED_INACTIVE = "inactive"`
- `ADAPTIVE_MIXED_ACTIVE = "active"`
- `ADAPTIVE_MIXED_DRAINING = "draining"`
- `_reset_adaptive_mixed_controller(self) -> None`
- `_adaptive_transition_eligible(self) -> bool`
- `_update_adaptive_mixed_state(self, waiting_depth: int) -> None`
- `_mixed_decode_reservation(self) -> tuple[int, int] | None`; the tuple is
  `(seq_id, required_free_blocks)` for one guaranteed decode row.
- `_schedule_mixed_prefill_decode(self, *, allow_waiting_admission: bool = True, require_decode: bool = False) -> tuple[list[Sequence], bool, bool, str] | tuple[list[Sequence], bool, bool] | None`
- `_schedule_chunked_prefill(self, max_prefill_seqs: int | None = None, max_prefill_tokens: int | None = None, *, allow_waiting_admission: bool = True, reserved_free_blocks: int = 0) -> tuple[list[Sequence], bool, bool] | None`
- `_schedule_adaptive_mixed(self, waiting_depth: int) -> tuple[list[Sequence], bool, bool] | tuple[list[Sequence], bool, bool, str]`

Complete executable bodies appear in Tasks 2–4; do not create stub
implementations.

The reservation tuple is computed read-only. `required_free_blocks` is `1` only when the selected decode sequence crosses a KV block boundary on its next token, otherwise `0`. The mixed helper passes that reserve into chunked-prefill admission, commits the prefill selection, then locates the reserved `seq_id` and calls `may_append()` for that row before adding optional extra decode rows.

---

### Task 1: Add Fail-Closed Adaptive Configuration

**Files:**
- Modify: `tinyvllm/config.py:55`
- Modify: `tinyvllm/config.py:133`
- Test: `tools/test_chunked_prefill.py`

**Interfaces:**
- Consumes: existing chunked-prefill and KV-offload configuration fields.
- Produces: five `Config` attributes with exact defaults and initialization-time assertions.

- [ ] **Step 1: Write failing configuration contract tests**

Add a dependency-light AST test that reads `tinyvllm/config.py`, extracts the five annotated defaults, and asserts:

```python
def test_adaptive_mixed_config_defaults_and_fail_closed_contract():
    source = open(
        os.path.join(_REPO_ROOT, "tinyvllm/config.py"),
        encoding="utf-8",
    ).read()
    tree = ast.parse(source)
    config_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Config"
    )
    defaults = {
        node.target.id: ast.literal_eval(node.value)
        for node in config_class.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.value is not None
    }
    assert defaults["chunked_prefill_adaptive_mixed"] is False
    assert defaults["chunked_prefill_adaptive_enter_waiting"] == 8
    assert defaults["chunked_prefill_adaptive_exit_waiting"] == 2
    assert defaults["chunked_prefill_adaptive_transition_steps"] == 2
    assert defaults["chunked_prefill_adaptive_max_mixed_steps"] == 2
    for fragment in (
        "chunked_prefill_adaptive_enter_waiting > 0",
        "chunked_prefill_adaptive_exit_waiting >= 0",
        "chunked_prefill_adaptive_transition_steps > 0",
        "chunked_prefill_adaptive_max_mixed_steps > 0",
        "chunked_prefill_adaptive_exit_waiting < self.chunked_prefill_adaptive_enter_waiting",
        "not (self.chunked_prefill_adaptive_mixed and self.chunked_prefill_mixed_batch)",
        "not (self.chunked_prefill_adaptive_mixed and self.kv_offload_mvp0)",
        "not self.chunked_prefill_adaptive_mixed or self.max_num_prefill_tokens_per_step > 0",
    ):
        assert fragment in source
```

Add an executable fail-closed test by loading the real dataclass with a fake
`transformers.AutoConfig`:

```python
def load_real_config_class():
    module_name = "tinyvllm_config_contract_under_test"
    module_path = os.path.join(_REPO_ROOT, "tinyvllm/config.py")
    fake_transformers = types.ModuleType("transformers")

    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(model):
            del model
            return SimpleNamespace(
                max_position_embeddings=4096,
                num_hidden_layers=4,
            )

    fake_transformers.AutoConfig = FakeAutoConfig
    original = sys.modules.get("transformers")
    sys.modules["transformers"] = fake_transformers
    try:
        spec = importlib.util.spec_from_file_location(
            module_name,
            module_path,
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module.Config
    finally:
        if original is None:
            sys.modules.pop("transformers", None)
        else:
            sys.modules["transformers"] = original


def test_adaptive_mixed_invalid_configurations_fail_before_model_start():
    Config = load_real_config_class()
    with tempfile.TemporaryDirectory() as model:
        common = {
            "model": model,
            "max_num_batched_tokens": 4096,
            "max_model_len": 4096,
            "kvcache_block_size": 256,
        }
        invalid = (
            {"chunked_prefill_adaptive_enter_waiting": 0},
            {"chunked_prefill_adaptive_exit_waiting": -1},
            {"chunked_prefill_adaptive_transition_steps": 0},
            {"chunked_prefill_adaptive_max_mixed_steps": 0},
            {
                "chunked_prefill_adaptive_enter_waiting": 2,
                "chunked_prefill_adaptive_exit_waiting": 2,
            },
            {
                "chunked_prefill_adaptive_mixed": True,
                "max_num_prefill_tokens_per_step": 0,
            },
            {
                "chunked_prefill_adaptive_mixed": True,
                "max_num_prefill_tokens_per_step": 128,
                "chunked_prefill_mixed_batch": True,
            },
            {
                "chunked_prefill_adaptive_mixed": True,
                "max_num_prefill_tokens_per_step": 128,
                "kv_offload_mvp0": True,
            },
        )
        for overrides in invalid:
            try:
                Config(**common, **overrides)
            except AssertionError:
                pass
            else:
                raise AssertionError(
                    f"invalid adaptive config accepted: {overrides}"
                )
```

Add `import tempfile` near the other test imports and append both test calls to
`tools/test_chunked_prefill.py::main`.

Append the test call to `tools/test_chunked_prefill.py::main`.

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
python3 tools/test_chunked_prefill.py
```

Expected: failure because `chunked_prefill_adaptive_mixed` is absent.

- [ ] **Step 3: Add the exact fields and assertions**

Add beside the existing mixed-prefill fields:

```python
chunked_prefill_adaptive_mixed: bool = False
chunked_prefill_adaptive_enter_waiting: int = 8
chunked_prefill_adaptive_exit_waiting: int = 2
chunked_prefill_adaptive_transition_steps: int = 2
chunked_prefill_adaptive_max_mixed_steps: int = 2
```

Add before `AutoConfig.from_pretrained()`:

```python
assert self.chunked_prefill_adaptive_enter_waiting > 0
assert self.chunked_prefill_adaptive_exit_waiting >= 0
assert self.chunked_prefill_adaptive_transition_steps > 0
assert self.chunked_prefill_adaptive_max_mixed_steps > 0
assert (
    self.chunked_prefill_adaptive_exit_waiting
    < self.chunked_prefill_adaptive_enter_waiting
)
assert not (
    self.chunked_prefill_adaptive_mixed
    and self.chunked_prefill_mixed_batch
), "adaptive mixed 和 always-on mixed 必须分开评测"
assert not (
    self.chunked_prefill_adaptive_mixed
    and self.kv_offload_mvp0
), "KV offload MVP-0 暂不支持 adaptive mixed prefill+decode"
assert (
    not self.chunked_prefill_adaptive_mixed
    or self.max_num_prefill_tokens_per_step > 0
), "adaptive mixed 需要开启 chunked prefill"
```

- [ ] **Step 4: Run the dependency-light suite**

Run:

```bash
python3 tools/test_chunked_prefill.py
```

Expected: `chunked prefill tests passed`.

- [ ] **Step 5: Commit the configuration contract**

```bash
git add tinyvllm/config.py tools/test_chunked_prefill.py
git commit -m "feat(scheduler): add adaptive mixed configuration"
```

---

### Task 2: Implement the Controller State Machine Without Changing Scheduling Yet

**Files:**
- Modify: `tinyvllm/engine/scheduler.py:9`
- Modify: `tinyvllm/engine/scheduler.py:35`
- Test: `tools/test_chunked_prefill.py`

**Interfaces:**
- Consumes: five validated config fields from Task 1.
- Produces: controller state/counters, transition helper, empty reset, and immutable observation fields; scheduling remains decode-first until Task 4.

- [ ] **Step 1: Write failing state-machine tests**

Add helpers:

```python
def make_running(scheduler, token_ids=(90, 91, 92, 93), max_tokens=8):
    seq = make_seq(token_ids, max_tokens=max_tokens)
    scheduler.block_manager.allocate(seq)
    seq.status = SequenceStatus.RUNNING
    seq.num_computed_tokens = len(seq)
    scheduler.running.append(seq)
    return seq


def add_waiting(scheduler, count, prompt_tokens=12):
    rows = []
    for offset in range(count):
        seq = make_seq(range(offset * 100, offset * 100 + prompt_tokens))
        scheduler.add(seq)
        rows.append(seq)
    return rows
```

Add tests that call `_update_adaptive_mixed_state(waiting_depth)` directly after constructing eligible queues:

```python
def test_adaptive_state_requires_two_high_observations_and_resets_streak():
    scheduler = Scheduler(make_config(
        chunked_prefill_adaptive_mixed=True,
        chunked_prefill_adaptive_enter_waiting=8,
        chunked_prefill_adaptive_exit_waiting=2,
        chunked_prefill_adaptive_transition_steps=2,
        chunked_prefill_adaptive_max_mixed_steps=2,
    ))
    make_running(scheduler)
    add_waiting(scheduler, 8)

    scheduler._update_adaptive_mixed_state(8)
    assert scheduler.adaptive_mixed_state == "inactive"
    assert scheduler.adaptive_high_streak == 1

    scheduler._update_adaptive_mixed_state(7)
    assert scheduler.adaptive_high_streak == 0

    scheduler._update_adaptive_mixed_state(8)
    scheduler._update_adaptive_mixed_state(8)
    assert scheduler.adaptive_mixed_state == "active"
    assert scheduler.adaptive_high_streak == 0
```

```python
def test_adaptive_state_low_hysteresis_enters_inactive_or_draining():
    scheduler = Scheduler(make_config(
        chunked_prefill_adaptive_mixed=True,
    ))
    make_running(scheduler)
    add_waiting(scheduler, 2)
    scheduler.adaptive_mixed_state = "active"

    scheduler._update_adaptive_mixed_state(2)
    assert scheduler.adaptive_mixed_state == "active"
    assert scheduler.adaptive_low_streak == 1
    scheduler._update_adaptive_mixed_state(2)
    assert scheduler.adaptive_mixed_state == "inactive"

    scheduler.adaptive_mixed_state = "active"
    prefilling = scheduler.waiting.popleft()
    scheduler.block_manager.allocate(prefilling)
    prefilling.status = SequenceStatus.PREFILLING
    scheduler.prefilling.append(prefilling)
    scheduler._update_adaptive_mixed_state(1)
    scheduler._update_adaptive_mixed_state(1)
    assert scheduler.adaptive_mixed_state == "draining"
```

```python
def test_adaptive_ineligible_decision_clears_transition_streaks():
    scheduler = Scheduler(make_config(
        chunked_prefill_adaptive_mixed=True,
    ))
    scheduler.adaptive_high_streak = 1
    scheduler.adaptive_low_streak = 1
    scheduler._update_adaptive_mixed_state(9)
    assert scheduler.adaptive_high_streak == 0
    assert scheduler.adaptive_low_streak == 0
```

```python
def test_adaptive_observation_and_empty_reset_are_exact():
    scheduler = Scheduler(make_config(
        chunked_prefill_adaptive_mixed=True,
    ))
    scheduler.adaptive_mixed_state = "draining"
    scheduler.adaptive_high_streak = 1
    scheduler.adaptive_low_streak = 0
    scheduler.adaptive_consecutive_mixed_steps = 2
    snapshot = scheduler.observation_snapshot()
    assert snapshot["adaptive_mixed_state"] == "draining"
    assert snapshot["adaptive_high_streak"] == 1
    assert snapshot["adaptive_low_streak"] == 0
    assert snapshot["adaptive_consecutive_mixed_steps"] == 2

    scheduler._maybe_reset_adaptive_mixed_controller()
    assert scheduler.adaptive_mixed_state == "inactive"
    assert scheduler.adaptive_high_streak == 0
    assert scheduler.adaptive_low_streak == 0
    assert scheduler.adaptive_consecutive_mixed_steps == 0
```

Append every test call to `main`.

- [ ] **Step 2: Run the tests and verify the first missing interface**

Run:

```bash
python3 tools/test_chunked_prefill.py
```

Expected: failure because `_update_adaptive_mixed_state` does not exist.

- [ ] **Step 3: Add state constants, config extraction, snapshots, and reset**

Add module constants:

```python
ADAPTIVE_MIXED_INACTIVE = "inactive"
ADAPTIVE_MIXED_ACTIVE = "active"
ADAPTIVE_MIXED_DRAINING = "draining"
```

Initialize:

```python
self.chunked_prefill_adaptive_mixed = getattr(
    config, "chunked_prefill_adaptive_mixed", False
)
self.chunked_prefill_adaptive_enter_waiting = getattr(
    config, "chunked_prefill_adaptive_enter_waiting", 8
)
self.chunked_prefill_adaptive_exit_waiting = getattr(
    config, "chunked_prefill_adaptive_exit_waiting", 2
)
self.chunked_prefill_adaptive_transition_steps = getattr(
    config, "chunked_prefill_adaptive_transition_steps", 2
)
self.chunked_prefill_adaptive_max_mixed_steps = getattr(
    config, "chunked_prefill_adaptive_max_mixed_steps", 2
)
self.adaptive_mixed_state = ADAPTIVE_MIXED_INACTIVE
self.adaptive_high_streak = 0
self.adaptive_low_streak = 0
self.adaptive_consecutive_mixed_steps = 0
```

Add snapshot scalars exactly as specified and implement:

```python
def _reset_adaptive_mixed_controller(self) -> None:
    self.adaptive_mixed_state = ADAPTIVE_MIXED_INACTIVE
    self.adaptive_high_streak = 0
    self.adaptive_low_streak = 0
    self.adaptive_consecutive_mixed_steps = 0


def _maybe_reset_adaptive_mixed_controller(self) -> None:
    if not self.waiting and not self.prefilling and not self.running:
        self._reset_adaptive_mixed_controller()


def _adaptive_transition_eligible(self) -> bool:
    return bool(
        self.chunked_prefill_adaptive_mixed
        and self.chunked_prefill_enabled
        and self.running
        and (self.waiting or self.prefilling)
    )
```

Implement `_update_adaptive_mixed_state()` with these exact rules:

```python
def _update_adaptive_mixed_state(self, waiting_depth: int) -> None:
    if not self._adaptive_transition_eligible():
        self.adaptive_high_streak = 0
        self.adaptive_low_streak = 0
        return

    transition_steps = self.chunked_prefill_adaptive_transition_steps
    if self.adaptive_mixed_state == ADAPTIVE_MIXED_INACTIVE:
        self.adaptive_low_streak = 0
        if waiting_depth >= self.chunked_prefill_adaptive_enter_waiting:
            self.adaptive_high_streak += 1
        else:
            self.adaptive_high_streak = 0
        if self.adaptive_high_streak >= transition_steps:
            self.adaptive_mixed_state = ADAPTIVE_MIXED_ACTIVE
            self.adaptive_high_streak = 0
        return

    if self.adaptive_mixed_state == ADAPTIVE_MIXED_ACTIVE:
        self.adaptive_high_streak = 0
        if waiting_depth <= self.chunked_prefill_adaptive_exit_waiting:
            self.adaptive_low_streak += 1
        else:
            self.adaptive_low_streak = 0
        if self.adaptive_low_streak >= transition_steps:
            self.adaptive_mixed_state = (
                ADAPTIVE_MIXED_DRAINING
                if self.prefilling
                else ADAPTIVE_MIXED_INACTIVE
            )
            self.adaptive_low_streak = 0
        return

    self.adaptive_low_streak = 0
    if waiting_depth >= self.chunked_prefill_adaptive_enter_waiting:
        self.adaptive_high_streak += 1
    else:
        self.adaptive_high_streak = 0
    if self.adaptive_high_streak >= transition_steps:
        self.adaptive_mixed_state = ADAPTIVE_MIXED_ACTIVE
        self.adaptive_high_streak = 0
    elif not self.prefilling:
        self.adaptive_mixed_state = ADAPTIVE_MIXED_INACTIVE
        self.adaptive_consecutive_mixed_steps = 0
```

Make the three postprocess exits explicit:

```python
if batch_kind == "mixed":
    self._postprocess_mixed(seqs, token_ids)
    self._maybe_reset_adaptive_mixed_controller()
    return
if is_prefill and self.chunked_prefill_enabled:
    self._postprocess_chunked_prefill(seqs, token_ids, do_sample)
    self._maybe_reset_adaptive_mixed_controller()
    return
```

At the final line of the normal postprocess path, call:

```python
self._maybe_reset_adaptive_mixed_controller()
```

- [ ] **Step 4: Run the scheduler tests**

Run:

```bash
python3 tools/test_chunked_prefill.py
```

Expected: `chunked prefill tests passed`.

- [ ] **Step 5: Commit the state machine**

```bash
git add tinyvllm/engine/scheduler.py tools/test_chunked_prefill.py
git commit -m "feat(scheduler): add adaptive mixed state machine"
```

---

### Task 3: Make Mixed Admission Transactional for P3 and P4

**Files:**
- Modify: `tinyvllm/engine/scheduler.py:195`
- Test: `tools/test_chunked_prefill.py`

**Interfaces:**
- Consumes: shared scheduler interfaces and block-manager read-only `estimate_admission()` / `can_append()`.
- Produces: a mixed helper that can require a decode row without prefill allocation or queue mutation on failure.

- [ ] **Step 1: Write failing transactional and compatibility tests**

Add:

```python
def test_required_mixed_decode_failure_is_transactional():
    scheduler = Scheduler(make_config(
        max_num_seqs=1,
        max_num_batched_tokens=4,
        chunked_prefill_mixed_batch=False,
        chunked_prefill_adaptive_mixed=True,
    ))
    running = make_running(scheduler)
    waiting = add_waiting(scheduler, 1)[0]
    queue_before = scheduler.observation_snapshot()
    waiting_blocks_before = list(waiting.block_table)

    result = scheduler._schedule_mixed_prefill_decode(
        allow_waiting_admission=True,
        require_decode=True,
    )

    assert result is None
    assert scheduler.observation_snapshot() == queue_before
    assert waiting.block_table == waiting_blocks_before
    assert list(scheduler.running) == [running]
    assert list(scheduler.waiting) == [waiting]
```

```python
def test_required_mixed_reserves_kv_block_for_decode_before_prefill():
    scheduler = Scheduler(make_config(
        max_num_seqs=2,
        max_num_batched_tokens=8,
        num_kvcache_blocks=3,
        kvcache_block_size=4,
        chunked_prefill_adaptive_mixed=True,
    ))
    running = make_running(
        scheduler,
        token_ids=(90, 91, 92, 93, 94),
    )
    waiting = add_waiting(scheduler, 1, prompt_tokens=8)[0]
    free_before = len(scheduler.block_manager.free_block_ids)

    result = scheduler._schedule_mixed_prefill_decode(
        allow_waiting_admission=True,
        require_decode=True,
    )

    assert result is None
    assert len(scheduler.block_manager.free_block_ids) == free_before
    assert list(scheduler.running) == [running]
    assert list(scheduler.waiting) == [waiting]
```

```python
def test_required_mixed_success_contains_both_roles():
    scheduler = Scheduler(make_config(
        max_num_seqs=4,
        max_num_batched_tokens=16,
        num_kvcache_blocks=16,
        chunked_prefill_adaptive_mixed=True,
    ))
    running = make_running(scheduler)
    waiting = add_waiting(scheduler, 1)[0]

    seqs, is_prefill, do_sample, batch_kind = (
        scheduler._schedule_mixed_prefill_decode(
            allow_waiting_admission=True,
            require_decode=True,
        )
    )

    assert is_prefill is True
    assert do_sample is True
    assert batch_kind == "mixed"
    assert waiting in seqs
    assert running in seqs
    assert any(not seq.step_is_decode for seq in seqs)
    assert any(seq.step_is_decode for seq in seqs)
```

Keep the existing `test_mixed_prefill_decode_schedules_prefill_chunk_with_decode`, short-prompt batching, token-budget, and postprocess tests unchanged. They are the P3 compatibility gate.

- [ ] **Step 2: Run the tests and verify the signature failure**

Run:

```bash
python3 tools/test_chunked_prefill.py
```

Expected: failure because `_schedule_mixed_prefill_decode()` does not accept `require_decode`.

- [ ] **Step 3: Add read-only decode reservation**

Implement:

```python
def _mixed_decode_reservation(self) -> tuple[int, int] | None:
    if self.max_num_seqs < 2 or self.max_num_batched_tokens < 2:
        return None
    for seq in self.running:
        required_free_blocks = int(
            len(seq) % self.block_manager.block_size == 1
        )
        if len(self.block_manager.free_block_ids) >= required_free_blocks:
            return seq.seq_id, required_free_blocks
    return None
```

- [ ] **Step 4: Add admission controls to `_schedule_chunked_prefill`**

At entry, continue an existing `prefilling` sequence regardless of `allow_waiting_admission`. If there is no `prefilling`, return `None` when `allow_waiting_admission` is false. For every new waiting admission, replace the free-block test with:

```python
if (
    len(self.block_manager.free_block_ids)
    - required_free_blocks
    < reserved_free_blocks
):
    break
```

For the first candidate, use this exact pre-allocation check:

```python
candidate = self.waiting[0]
_, required_free_blocks = self.block_manager.estimate_admission(
    candidate
)
if (
    len(self.block_manager.free_block_ids)
    - required_free_blocks
    < reserved_free_blocks
):
    return None
```

Only after it passes may the method call `popleft()` and `allocate()`.

- [ ] **Step 5: Refactor the mixed helper around the reservation**

When `require_decode=True`:

1. Call `_mixed_decode_reservation()` before any mutation.
2. Return `None` when it returns `None`.
3. Pass `reserved_free_blocks` and `allow_waiting_admission` into `_schedule_chunked_prefill`.
4. Return `None` when no prefill row can be selected.
5. Remove the reserved `seq_id` from `running`, call `may_append()`, and place it first in `decode_seqs`.
6. Add optional remaining decode rows with this loop:

```python
while (
    self.running
    and len(prefill_seqs) + len(decode_seqs) < self.max_num_seqs
    and prefill_tokens + len(decode_seqs) < self.max_num_batched_tokens
):
    seq = self.running.popleft()
    while not self.block_manager.can_append(seq):
        if self.running:
            self.preempt(self.running.pop())
        else:
            self.preempt(seq)
            seq = None
            break
    if seq is None:
        continue
    self.block_manager.may_append(seq)
    decode_seqs.append(seq)
```
7. Assert both roles exist before returning the four-element mixed tuple.

When `require_decode=False`, preserve current `P3` behavior: if no decode row survives, return the prefill tuple exactly as before.

- [ ] **Step 6: Run focused and full scheduler tests**

Run:

```bash
python3 tools/test_chunked_prefill.py
```

Expected: all new transactional tests and all existing P3 mixed tests pass.

- [ ] **Step 7: Commit the shared mixed scheduler refactor**

```bash
git add tinyvllm/engine/scheduler.py tools/test_chunked_prefill.py
git commit -m "fix(scheduler): make mixed admission transactional"
```

---

### Task 4: Route Adaptive Scheduling and Enforce Decode Service

**Files:**
- Modify: `tinyvllm/engine/scheduler.py:80`
- Test: `tools/test_chunked_prefill.py`

**Interfaces:**
- Consumes: state machine from Task 2 and transactional helper from Task 3.
- Produces: all five adaptive policy branches and exact scheduling behavior.

- [ ] **Step 1: Write failing end-to-end scheduler tests**

Add tests for the complete branch contract:

```python
def test_adaptive_disabled_matches_decode_first_schedule_and_snapshot():
    reset_sequence_state()
    baseline = Scheduler(make_config(
        chunked_prefill_decode_first=True,
        chunked_prefill_adaptive_mixed=False,
    ))
    baseline_running = make_running(baseline)
    baseline_waiting = add_waiting(baseline, 8)

    baseline_result = baseline.schedule()
    assert baseline.last_policy_branch == "decode_first"
    baseline_metadata = {
        "scheduled_roles": [
            (
                seq is baseline_running,
                seq.prefill_chunk_start,
                seq.prefill_chunk_end,
                seq.prefill_chunk_final,
            )
            for seq in baseline_result[0]
        ],
        "is_prefill": baseline_result[1],
        "do_sample": baseline_result[2],
        "branch": baseline.last_policy_branch,
        "waiting": list(baseline.waiting),
        "prefilling": list(baseline.prefilling),
        "running": list(baseline.running),
        "free_blocks": list(baseline.block_manager.free_block_ids),
        "used_blocks": list(baseline.block_manager.used_block_ids),
    }

    reset_sequence_state()
    candidate = Scheduler(make_config(
        chunked_prefill_decode_first=True,
        chunked_prefill_adaptive_mixed=False,
    ))
    candidate_running = make_running(candidate)
    candidate_waiting = add_waiting(candidate, 8)
    candidate_result = candidate.schedule()
    candidate_metadata = {
        "scheduled_roles": [
            (
                seq is candidate_running,
                seq.prefill_chunk_start,
                seq.prefill_chunk_end,
                seq.prefill_chunk_final,
            )
            for seq in candidate_result[0]
        ],
        "is_prefill": candidate_result[1],
        "do_sample": candidate_result[2],
        "branch": candidate.last_policy_branch,
        "waiting": list(candidate.waiting),
        "prefilling": list(candidate.prefilling),
        "running": list(candidate.running),
        "free_blocks": list(candidate.block_manager.free_block_ids),
        "used_blocks": list(candidate.block_manager.used_block_ids),
    }

    assert baseline_metadata["scheduled_roles"] == (
        candidate_metadata["scheduled_roles"]
    )
    assert baseline_metadata["is_prefill"] == candidate_metadata["is_prefill"]
    assert baseline_metadata["do_sample"] == candidate_metadata["do_sample"]
    assert baseline_metadata["branch"] == candidate_metadata["branch"]
    assert len(baseline_metadata["waiting"]) == len(
        candidate_metadata["waiting"]
    ) == 8
    assert not baseline_metadata["prefilling"]
    assert not candidate_metadata["prefilling"]
    assert baseline_metadata["free_blocks"] == candidate_metadata["free_blocks"]
    assert baseline_metadata["used_blocks"] == candidate_metadata["used_blocks"]
    assert baseline_running.status == candidate_running.status
    assert len(baseline_waiting) == len(candidate_waiting)
```

```python
def test_adaptive_second_high_observation_activates_and_mixes():
    scheduler = Scheduler(make_config(
        chunked_prefill_decode_first=True,
        chunked_prefill_adaptive_mixed=True,
    ))
    running = make_running(scheduler)
    add_waiting(scheduler, 8)

    first = scheduler.schedule()
    assert first[0] == [running]
    assert scheduler.last_policy_branch == "adaptive_mixed_decode_first"
    assert scheduler.adaptive_high_streak == 1

    second = scheduler.schedule()
    assert len(second) == 4
    assert second[3] == "mixed"
    assert scheduler.last_policy_branch == "adaptive_mixed_prefill_decode"
    assert scheduler.adaptive_mixed_state == "active"
    assert scheduler.adaptive_consecutive_mixed_steps == 1
```

```python
def test_adaptive_two_mixed_steps_force_decode_yield():
    scheduler = Scheduler(make_config(
        chunked_prefill_adaptive_mixed=True,
        chunked_prefill_adaptive_max_mixed_steps=2,
    ))
    make_running(scheduler, max_tokens=16)
    add_waiting(scheduler, 10)
    scheduler.adaptive_mixed_state = "active"

    first = scheduler.schedule()
    assert first[3] == "mixed"
    scheduler.postprocess(first[0], [101], first[1], first[2], first[3])
    second = scheduler.schedule()
    assert second[3] == "mixed"
    scheduler.postprocess(second[0], [102], second[1], second[2], second[3])
    third = scheduler.schedule()

    assert len(third) == 3
    assert third[1] is False
    assert scheduler.last_policy_branch == "adaptive_mixed_decode_yield"
    assert scheduler.adaptive_consecutive_mixed_steps == 0
    assert scheduler.adaptive_mixed_state == "active"
```

```python
def test_adaptive_draining_never_admits_waiting_and_returns_inactive():
    scheduler = Scheduler(make_config(
        chunked_prefill_adaptive_mixed=True,
    ))
    make_running(scheduler, max_tokens=16)
    prefilling = add_waiting(scheduler, 1)[0]
    scheduler.waiting.popleft()
    scheduler.block_manager.allocate(prefilling)
    prefilling.status = SequenceStatus.PREFILLING
    scheduler.prefilling.append(prefilling)
    waiting = add_waiting(scheduler, 1)[0]
    scheduler.adaptive_mixed_state = "draining"

    result = scheduler.schedule()
    assert result[3] == "mixed"
    assert result[0][0] is prefilling
    assert list(scheduler.waiting) == [waiting]
    assert scheduler.last_policy_branch == "adaptive_mixed_prefill_decode"
```

```python
def test_adaptive_no_running_uses_chunked_prefill_without_fake_decode():
    scheduler = Scheduler(make_config(
        chunked_prefill_adaptive_mixed=True,
    ))
    waiting = add_waiting(scheduler, 1)[0]
    result = scheduler.schedule()
    assert len(result) == 3
    assert result[0] == [waiting]
    assert scheduler.last_policy_branch == "adaptive_mixed_chunked_prefill"
    assert scheduler.adaptive_consecutive_mixed_steps == 0
```

```python
def test_adaptive_required_mixed_failure_falls_back_to_decode():
    scheduler = Scheduler(make_config(
        max_num_seqs=1,
        chunked_prefill_adaptive_mixed=True,
    ))
    running = make_running(scheduler)
    waiting = add_waiting(scheduler, 8)
    scheduler.adaptive_mixed_state = "active"
    queue_before = list(scheduler.waiting)
    result = scheduler.schedule()
    assert result[0] == [running]
    assert scheduler.last_policy_branch == "adaptive_mixed_decode_fallback"
    assert list(scheduler.waiting) == queue_before
    assert all(not seq.block_table for seq in waiting)
```

Update the engine observation test's `FakeScheduler.observation_snapshot()` to
return:

```python
return {
    "snapshot_index": self.snapshot_index,
    "adaptive_mixed_state": "active",
    "adaptive_high_streak": 0,
    "adaptive_low_streak": 1,
    "adaptive_consecutive_mixed_steps": 2,
}
```

Update the expected `queue_before` and `queue_after` dictionaries with those
four values. No production change to `tinyvllm/engine/llm_engine.py` is
planned because it already records complete snapshots returned by the
scheduler.

- [ ] **Step 2: Run tests and verify routing is absent**

Run:

```bash
python3 tools/test_chunked_prefill.py
```

Expected: adaptive branch assertions fail because `schedule()` still uses the legacy chunked path.

- [ ] **Step 3: Sample waiting depth once and route P4 first**

At the top of `schedule()`:

```python
waiting_depth = len(self.waiting)
self._maybe_reset_adaptive_mixed_controller()
if self.chunked_prefill_enabled and self.chunked_prefill_adaptive_mixed:
    return self._schedule_adaptive_mixed(waiting_depth)
```

Do not read `len(self.waiting)` again for threshold decisions in the same call.

- [ ] **Step 4: Implement `_schedule_adaptive_mixed`**

Use this branch order:

```python
def _schedule_adaptive_mixed(self, waiting_depth: int):
    self._update_adaptive_mixed_state(waiting_depth)

    if not self.running:
        self.adaptive_consecutive_mixed_steps = 0
        prefill = self._schedule_chunked_prefill()
        if prefill is not None:
            return self._return_schedule(
                prefill,
                "adaptive_mixed_chunked_prefill",
            )
        return self._return_schedule(
            (*self._schedule_decode(), True),
            "adaptive_mixed_decode_fallback",
        )

    if self.adaptive_mixed_state == ADAPTIVE_MIXED_INACTIVE:
        self.adaptive_consecutive_mixed_steps = 0
        return self._return_schedule(
            (*self._schedule_decode(), True),
            "adaptive_mixed_decode_first",
        )

    if (
        self.adaptive_consecutive_mixed_steps
        >= self.chunked_prefill_adaptive_max_mixed_steps
    ):
        self.adaptive_consecutive_mixed_steps = 0
        return self._return_schedule(
            (*self._schedule_decode(), True),
            "adaptive_mixed_decode_yield",
        )

    mixed = self._schedule_mixed_prefill_decode(
        allow_waiting_admission=(
            self.adaptive_mixed_state == ADAPTIVE_MIXED_ACTIVE
        ),
        require_decode=True,
    )
    if mixed is None:
        self.adaptive_consecutive_mixed_steps = 0
        return self._return_schedule(
            (*self._schedule_decode(), True),
            "adaptive_mixed_decode_fallback",
        )
    self.adaptive_consecutive_mixed_steps += 1
    return self._return_schedule(
        mixed,
        "adaptive_mixed_prefill_decode",
    )
```

Before selecting a branch, if state is `DRAINING` and `prefilling` is empty, change to `INACTIVE` and take decode-first. The fallback path must reset the consecutive mixed counter.

- [ ] **Step 5: Run scheduler and observation tests**

Run:

```bash
python3 tools/test_chunked_prefill.py
```

Expected: `chunked prefill tests passed`.

- [ ] **Step 6: Commit adaptive routing**

```bash
git add tinyvllm/engine/scheduler.py tools/test_chunked_prefill.py
git commit -m "feat(scheduler): route backlog adaptive mixed prefill"
```

---

### Task 5: Freeze P4 Policy Identity, Smoke, and Canonical Matrix

**Files:**
- Modify: `tools/arrival_load_gate.py`
- Modify: `tools/test_arrival_load_gate.py`
- Modify: `tools/test_arrival_load_driver.py`

**Interfaces:**
- Consumes: adaptive scheduler fields and branch observations.
- Produces: source-bound manifests with `P0/P3/P4`, a P4-activating smoke workload, 54 canonical cases, and P4-only promotion semantics.

- [ ] **Step 1: Write failing policy and matrix tests**

Replace old four-policy expectations with:

```python
ADAPTIVE_DEFAULTS = {
    "chunked_prefill_decode_first": True,
    "chunked_prefill_max_consecutive_chunks": 0,
    "chunked_prefill_mixed_batch": False,
    "chunked_prefill_mixed_min_prompt_tokens": 0,
    "chunked_prefill_adaptive_mixed": False,
    "chunked_prefill_adaptive_enter_waiting": 8,
    "chunked_prefill_adaptive_exit_waiting": 2,
    "chunked_prefill_adaptive_transition_steps": 2,
    "chunked_prefill_adaptive_max_mixed_steps": 2,
}
```

Assert:

```python
def test_p4_identity_contains_every_adaptive_field():
    resolved = {
        name: gate.resolve_policy_config(name, ADAPTIVE_DEFAULTS)
        for name in ("P0", "P3", "P4")
    }
    assert resolved["P4"] == {
        **gate.COMMON_ENGINE_CONFIG,
        **ADAPTIVE_DEFAULTS,
        "chunked_prefill_decode_first": True,
        "chunked_prefill_max_consecutive_chunks": 0,
        "chunked_prefill_mixed_batch": False,
        "chunked_prefill_mixed_min_prompt_tokens": 0,
        "chunked_prefill_adaptive_mixed": True,
        "chunked_prefill_adaptive_enter_waiting": 8,
        "chunked_prefill_adaptive_exit_waiting": 2,
        "chunked_prefill_adaptive_transition_steps": 2,
        "chunked_prefill_adaptive_max_mixed_steps": 2,
    }
    assert len({
        gate.policy_identity(resolved[name])
        for name in ("P0", "P3", "P4")
    }) == 3
```

Assert the matrix contains 54 unique rows and each repetition uses these rotations:

```python
{
    0: ("P0", "P3", "P4"),
    1: ("P3", "P4", "P0"),
    2: ("P4", "P0", "P3"),
}
```

Add classification tests where:

- `P3=GO`, `P4=NO_GO` produces top-level `NO_GO`;
- `P3=NO_GO`, `P4=GO` produces top-level `GO`;
- missing `P4` rows produce `INCOMPLETE`;
- a P4 identity collision with P0 or P3 fails closed.

- [ ] **Step 2: Write failing strengthened-smoke tests**

Assert `_smoke_workload()` creates at least ten simultaneous long-prompt requests, each with fixed-length greedy decoding, so the scheduler can observe `waiting >= 8` twice after the first request reaches decode:

```python
def test_smoke_workload_can_activate_p4_and_cross_ninth_token():
    workload = gate._smoke_workload(_prompt_bank())
    assert len(workload) >= 10
    assert sum(row["arrival_offset_ns"] == 0 for row in workload) >= 10
    assert sum(row["prompt_class"] == "long" for row in workload) >= 10
    assert all(row["sampling"]["temperature"] == 0.0 for row in workload)
    assert all(row["sampling"]["ignore_eos"] is True for row in workload)
    assert all(row["requested_output_tokens"] >= 16 for row in workload)
```

Assert smoke policies are exactly `["P0", "P4"]` and output correctness compares P4 directly with P0.

- [ ] **Step 3: Write failing driver passthrough test**

In `FakeEngine.last_step_observation`, include:

```python
"adaptive_mixed_state": "active",
"adaptive_high_streak": 0,
"adaptive_low_streak": 1,
"adaptive_consecutive_mixed_steps": 2,
```

inside both queue snapshots. Add:

```python
def test_driver_preserves_adaptive_controller_snapshots():
    temporary, output_dir, result = _run()
    try:
        assert result["status"] == "PASS"
        row = json.loads(
            (output_dir / "scheduler_trace.jsonl")
            .read_text()
            .splitlines()[0]
        )
        for snapshot_name in ("queue_before", "queue_after"):
            snapshot = row[snapshot_name]
            assert snapshot["adaptive_mixed_state"] == "active"
            assert snapshot["adaptive_high_streak"] == 0
            assert snapshot["adaptive_low_streak"] == 1
            assert snapshot["adaptive_consecutive_mixed_steps"] == 2
    finally:
        temporary.cleanup()
```

- [ ] **Step 4: Run the gate and driver tests and verify failures**

Run:

```bash
python3 tools/test_arrival_load_gate.py
python3 tools/test_arrival_load_driver.py
```

Expected: failures on absent P4 policy, old matrix, old smoke policies, and missing adaptive snapshot fields.

- [ ] **Step 5: Replace the frozen policy set**

In `arrival_load_gate.py`:

```python
POLICY_NAMES = ("P0", "P3", "P4")

POLICY_FIELDS = (
    "chunked_prefill_decode_first",
    "chunked_prefill_max_consecutive_chunks",
    "chunked_prefill_mixed_batch",
    "chunked_prefill_mixed_min_prompt_tokens",
    "chunked_prefill_adaptive_mixed",
    "chunked_prefill_adaptive_enter_waiting",
    "chunked_prefill_adaptive_exit_waiting",
    "chunked_prefill_adaptive_transition_steps",
    "chunked_prefill_adaptive_max_mixed_steps",
)
```

Define exact `P3` and `P4` overrides. Remove P1/P2 alias assumptions from new manifests, matrix validation, final classification, and tests. Keep historical artifacts untouched on disk.

Set:

```python
POLICY_ORDER_BY_REPETITION = {
    0: ("P0", "P3", "P4"),
    1: ("P3", "P4", "P0"),
    2: ("P4", "P0", "P3"),
}
```

The matrix length assertion remains `54`.

- [ ] **Step 6: Make P4 the sole promotion authority**

Compute diagnostic results for both candidates, then set:

```python
classification = candidate_results["P4"]["classification"]
```

Do not promote from `P3`. Include both results in `candidate_results` so P3 remains visible.

- [ ] **Step 7: Strengthen smoke and preserve immutable identity**

Make `_smoke_workload()` deterministic with at least ten long prompts at offset zero and fixed 16-token output. Set:

```python
smoke_policies = ("P0", "P4")
```

Update `_apply_output_correctness_smoke()` to compare `outputs["P4"]` with `outputs["P0"]`.

Ensure `_run_identity()` already includes `resolved_policy_config_by_name` and `policy_identity_by_name`; add a test that changing any one P4 threshold makes resume fail with `resume identity mismatch`.

- [ ] **Step 8: Run gate and driver suites**

Run:

```bash
python3 tools/test_arrival_load_gate.py
python3 tools/test_arrival_load_driver.py
```

Expected:

```text
arrival load gate tests passed
arrival load driver tests passed
```

- [ ] **Step 9: Commit policy and trace plumbing**

```bash
git add tools/arrival_load_gate.py tools/test_arrival_load_gate.py tools/test_arrival_load_driver.py
git commit -m "feat(gate): add source-bound P4 policy matrix"
```

---

### Task 6: Independently Reconstruct and Reject Invalid P4 State Transitions

**Files:**
- Modify: `tools/arrival_load_verify.py`
- Modify: `tools/test_arrival_load_verify.py`

**Interfaces:**
- Consumes: immutable P4 resolved config and scheduler trace rows.
- Produces: independent structural validation that does not import `arrival_load_gate.py`.

- [ ] **Step 1: Expand the synthetic complete artifact to P0/P3/P4**

Change the synthetic manifest to:

```python
"policy_identity_by_name": {
    "P0": _policy_id("P0"),
    "P3": _policy_id("P3"),
    "P4": _policy_id("P4"),
},
"canonical_policy_by_name": {
    "P0": "P0",
    "P3": "P3",
    "P4": "P4",
},
"resolved_policy_config_by_name": {
    "P0": dict(ADAPTIVE_DEFAULTS),
    "P3": {
        **ADAPTIVE_DEFAULTS,
        "chunked_prefill_decode_first": False,
        "chunked_prefill_mixed_batch": True,
    },
    "P4": {
        **ADAPTIVE_DEFAULTS,
        "chunked_prefill_adaptive_mixed": True,
    },
},
```

Generate P4 trace rows whose queue snapshots and branches form a legal sequence:

1. inactive/high streak 0 → `adaptive_mixed_decode_first` → inactive/high streak 1;
2. inactive/high streak 1 → `adaptive_mixed_prefill_decode` → active/mixed count 1;
3. active → `adaptive_mixed_prefill_decode` → active/mixed count 2;
4. active/mixed count 2 → `adaptive_mixed_decode_yield` → active/mixed count 0;
5. active/low streak 1 with existing prefilling → mixed drain → draining;
6. draining with empty prefilling → `adaptive_mixed_decode_first` → inactive.

Each mixed row must list at least one scheduled row with `"is_decode": False` and at least one with `"is_decode": True`.

- [ ] **Step 2: Write failing independent-verifier tamper tests**

Create one fresh artifact per mutation, refresh the changed artifact hash, and assert exact failures for:

```python
("adaptive_mixed_state", "illegal", "illegal adaptive state")
("adaptive_high_streak", -1, "invalid adaptive counter")
("adaptive_consecutive_mixed_steps", 3, "invalid adaptive counter")
```

Also add mutations for:

- `adaptive_mixed_prefill_decode` with no decode role;
- activation after only one high observation;
- three consecutive adaptive mixed branches without a yield;
- `DRAINING` queue-after containing a newly removed waiting ID and a new prefilling ID;
- P4 resolved threshold changed from `8` to `9` without changing identity;
- a P4 trace row missing one controller field;
- top-level recorded `GO` copied from a diagnostic P3 result while P4 is `NO_GO`.

Each case must call:

```python
verifier.verify_run(root, write_output=False)
```

and fail with `ValueError`.

- [ ] **Step 3: Run the verifier tests and verify missing validation**

Run:

```bash
python3 tools/test_arrival_load_verify.py
```

Expected: failure because the verifier currently ignores adaptive controller state.

- [ ] **Step 4: Add exact independent policy validation**

Define local constants in `arrival_load_verify.py`; do not import the harness:

```python
P4_FIELDS = (
    "chunked_prefill_adaptive_mixed",
    "chunked_prefill_adaptive_enter_waiting",
    "chunked_prefill_adaptive_exit_waiting",
    "chunked_prefill_adaptive_transition_steps",
    "chunked_prefill_adaptive_max_mixed_steps",
)

EXPECTED_P4 = {
    "chunked_prefill_decode_first": True,
    "chunked_prefill_max_consecutive_chunks": 0,
    "chunked_prefill_mixed_batch": False,
    "chunked_prefill_mixed_min_prompt_tokens": 0,
    "chunked_prefill_adaptive_mixed": True,
    "chunked_prefill_adaptive_enter_waiting": 8,
    "chunked_prefill_adaptive_exit_waiting": 2,
    "chunked_prefill_adaptive_transition_steps": 2,
    "chunked_prefill_adaptive_max_mixed_steps": 2,
}
```

Validate the manifest policy set is exactly `P0/P3/P4`, recompute each canonical JSON identity locally, and require every `EXPECTED_P4` key/value.

- [ ] **Step 5: Add trace reconstruction**

Implement:

```python
def _verify_p4_scheduler_trace(
    rows: list[dict],
    *,
    enter_waiting: int,
    exit_waiting: int,
    transition_steps: int,
    max_mixed_steps: int,
) -> None:
    if not rows:
        raise ValueError("missing P4 scheduler trace")
    expected_state = "inactive"
    expected_high = 0
    expected_low = 0
    expected_mixed = 0
    previous_controller_after = None
    for expected_step, row in enumerate(rows):
        if row.get("step_index") != expected_step:
            raise ValueError("invalid P4 scheduler step sequence")
        before = row.get("queue_before")
        after = row.get("queue_after")
        if not isinstance(before, dict) or not isinstance(after, dict):
            raise ValueError("missing P4 queue snapshot")
        required_fields = (
            "adaptive_mixed_state",
            "adaptive_high_streak",
            "adaptive_low_streak",
            "adaptive_consecutive_mixed_steps",
            "waiting_seq_ids",
            "prefilling_seq_ids",
            "running_seq_ids",
        )
        if any(field not in before for field in required_fields):
            raise ValueError("missing P4 controller field")
        if any(field not in after for field in required_fields):
            raise ValueError("missing P4 controller field")
        controller_before = tuple(
            before[field]
            for field in (
                "adaptive_mixed_state",
                "adaptive_high_streak",
                "adaptive_low_streak",
                "adaptive_consecutive_mixed_steps",
            )
        )
        if (
            previous_controller_after is not None
            and controller_before != previous_controller_after
        ):
            raise ValueError("P4 controller snapshots are not contiguous")
        for snapshot in (before, after):
            if snapshot["adaptive_mixed_state"] not in {
                "inactive",
                "active",
                "draining",
            }:
                raise ValueError("illegal adaptive state")
            counters = (
                snapshot["adaptive_high_streak"],
                snapshot["adaptive_low_streak"],
                snapshot["adaptive_consecutive_mixed_steps"],
            )
            if any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in counters
            ):
                raise ValueError("invalid adaptive counter")
            if counters[0] >= transition_steps:
                raise ValueError("invalid adaptive counter")
            if counters[1] >= transition_steps:
                raise ValueError("invalid adaptive counter")
            if counters[2] > max_mixed_steps:
                raise ValueError("invalid adaptive counter")
            queue_sets = [
                set(snapshot[name])
                for name in (
                    "waiting_seq_ids",
                    "prefilling_seq_ids",
                    "running_seq_ids",
                )
            ]
            if (
                queue_sets[0] & queue_sets[1]
                or queue_sets[0] & queue_sets[2]
                or queue_sets[1] & queue_sets[2]
            ):
                raise ValueError("duplicate P4 queue ownership")
        if (
            before["adaptive_mixed_state"] != expected_state
            or before["adaptive_high_streak"] != expected_high
            or before["adaptive_low_streak"] != expected_low
            or before["adaptive_consecutive_mixed_steps"]
            != expected_mixed
        ):
            raise ValueError("P4 controller continuity mismatch")

        waiting_depth = len(before["waiting_seq_ids"])
        eligible = bool(
            before["running_seq_ids"]
            and (
                before["waiting_seq_ids"]
                or before["prefilling_seq_ids"]
            )
        )
        state = expected_state
        high = expected_high
        low = expected_low
        if not eligible:
            high = 0
            low = 0
        elif state == "inactive":
            low = 0
            high = high + 1 if waiting_depth >= enter_waiting else 0
            if high >= transition_steps:
                state = "active"
                high = 0
        elif state == "active":
            high = 0
            low = low + 1 if waiting_depth <= exit_waiting else 0
            if low >= transition_steps:
                state = (
                    "draining"
                    if before["prefilling_seq_ids"]
                    else "inactive"
                )
                low = 0
        else:
            low = 0
            high = high + 1 if waiting_depth >= enter_waiting else 0
            if high >= transition_steps:
                state = "active"
                high = 0
            elif not before["prefilling_seq_ids"]:
                state = "inactive"
                expected_mixed = 0

        branch = row.get("policy_branch")
        scheduled = row.get("scheduled")
        if not isinstance(scheduled, list):
            raise ValueError("invalid P4 scheduled rows")
        has_prefill = any(
            item.get("is_decode") is False for item in scheduled
        )
        has_decode = any(
            item.get("is_decode") is True for item in scheduled
        )
        if branch == "adaptive_mixed_prefill_decode":
            if not has_prefill or not has_decode:
                raise ValueError("adaptive mixed branch role mismatch")
            expected_mixed += 1
            if expected_mixed > max_mixed_steps:
                raise ValueError("adaptive mixed service bound exceeded")
        elif branch in {
            "adaptive_mixed_decode_first",
            "adaptive_mixed_decode_yield",
            "adaptive_mixed_decode_fallback",
        }:
            if has_prefill:
                raise ValueError("decode-only adaptive branch has prefill")
            expected_mixed = 0
        elif branch == "adaptive_mixed_chunked_prefill":
            if before["running_seq_ids"] or has_decode:
                raise ValueError("adaptive chunked prefill has decode")
            expected_mixed = 0
        else:
            raise ValueError("illegal P4 policy branch")

        if state == "draining":
            newly_prefilling = (
                set(after["prefilling_seq_ids"])
                - set(before["prefilling_seq_ids"])
            )
            if newly_prefilling & set(before["waiting_seq_ids"]):
                raise ValueError("new waiting admission during draining")

        if (
            not after["waiting_seq_ids"]
            and not after["prefilling_seq_ids"]
            and not after["running_seq_ids"]
        ):
            state = "inactive"
            high = 0
            low = 0
            expected_mixed = 0

        if after["adaptive_mixed_state"] != state:
            raise ValueError("adaptive state transition mismatch")
        if after["adaptive_high_streak"] != high:
            raise ValueError("adaptive high streak mismatch")
        if after["adaptive_low_streak"] != low:
            raise ValueError("adaptive low streak mismatch")
        if (
            after["adaptive_consecutive_mixed_steps"]
            != expected_mixed
        ):
            raise ValueError("adaptive mixed counter mismatch")
        expected_state = state
        expected_high = high
        expected_low = low
        previous_controller_after = tuple(
            after[field]
            for field in (
                "adaptive_mixed_state",
                "adaptive_high_streak",
                "adaptive_low_streak",
                "adaptive_consecutive_mixed_steps",
            )
        )
```

For each P4 case:

1. Require contiguous `step_index`.
2. Require all four scalar fields in `queue_before` and `queue_after`.
3. Validate state names and integer counter ranges.
4. Derive `waiting_depth` only from `len(queue_before["waiting_seq_ids"])`.
5. Treat a row as eligible only when `queue_before` has running IDs and either waiting or prefilling IDs.
6. Recompute the expected transition before interpreting the branch.
7. Require branch/state compatibility.
8. Require both roles for `adaptive_mixed_prefill_decode`.
9. Require no prefill role for `adaptive_mixed_decode_yield`, `adaptive_mixed_decode_first`, and `adaptive_mixed_decode_fallback`.
10. Track mixed-branch count independently and reject a third mixed branch before a decode-only row.
11. In `DRAINING`, reject any new prefilling sequence whose ID came from `queue_before["waiting_seq_ids"]`.
12. Require queue ownership sets to be pairwise disjoint before and after every step.
13. Reset the reconstructed controller only when all three queue sets are empty.

Call this validation before recomputing case metrics. Any failure raises `ValueError`, causing independent verification exit code 1 and an `INCOMPLETE` artifact.

- [ ] **Step 6: Update classification and output equality**

Use canonical candidates from the manifest, compare both P3 and P4 output tokens against P0, compute both candidate results, and set top-level classification only from P4.

- [ ] **Step 7: Run verifier and harness suites**

Run:

```bash
python3 tools/test_arrival_load_verify.py
python3 tools/test_arrival_load_gate.py
```

Expected:

```text
arrival load verifier tests passed
arrival load gate tests passed
```

- [ ] **Step 8: Commit independent P4 verification**

```bash
git add tools/arrival_load_verify.py tools/test_arrival_load_verify.py
git commit -m "feat(verifier): reconstruct P4 scheduler state"
```

---

### Task 7: Harden the Remote Chain and Run Local Completion Tests

**Files:**
- Modify: `tools/run_arrival_load_gate_remote.sh`
- Modify: `tools/test_run_arrival_load_gate_remote.py`
- Test: all dependency-light arrival-load and scheduler suites

**Interfaces:**
- Consumes: P4 source snapshot, smoke artifact identity, frozen calibration, and remote environment evidence.
- Produces: an auditable command chain that cannot reuse evidence from another source or P4 identity.

- [ ] **Step 1: Write failing remote-runner assertions**

Add assertions that:

```python
def test_p4_chain_requires_explicit_predecessor_run_tags():
    runner = _runner()
    assert "calibration requires SMOKE_RUN_TAG" in runner
    assert "canonical requires SMOKE_RUN_TAG" in runner
    assert "canonical requires CALIBRATION_RUN_TAG" in runner
    assert "--smoke-run-dir" in runner
    assert "--calibration-run-dir" in runner
```

Keep and run the existing tests that prohibit `pkill`, `killall`, `rm -rf /tmp`, `git checkout`, `git reset`, `git clean`, `git add -A`, and `rsync`.

- [ ] **Step 2: Run the remote runner test and verify the missing canonical predecessor**

Run:

```bash
python3 tools/test_run_arrival_load_gate_remote.py
```

Expected: failure if canonical does not explicitly receive both predecessor run directories.

- [ ] **Step 3: Wire explicit predecessor identities**

For canonical mode require:

```bash
--smoke-run-dir "/data00/home/sitian/sitian-workspace01/tllm/arrival-load-runs/${SMOKE_RUN_TAG:?canonical requires SMOKE_RUN_TAG}/artifacts"
--calibration-run-dir "/data00/home/sitian/sitian-workspace01/tllm/arrival-load-runs/${CALIBRATION_RUN_TAG:?canonical requires CALIBRATION_RUN_TAG}/artifacts"
```

Add `_validate_predecessor_identity(current, predecessor, label)` in
`tools/arrival_load_gate.py`:

```python
def _validate_predecessor_identity(
    current: dict,
    predecessor: dict,
    label: str,
) -> None:
    for field in (
        "source_tree_sha256",
        "environment_sha256",
    ):
        if predecessor.get(field) != current.get(field):
            raise ValueError(f"{label} {field} identity mismatch")
    current_p4 = current.get(
        "policy_identity_by_name", {}
    ).get("P4")
    predecessor_p4 = predecessor.get(
        "policy_identity_by_name", {}
    ).get("P4")
    if predecessor_p4 != current_p4:
        raise ValueError(f"{label} P4 policy identity mismatch")
```

For calibration, call it on the smoke manifest before freezing the workload.
For canonical, call it on both smoke and calibration manifests, then also
require the calibration workload hash to equal the current frozen workload
hash before starting any case.

- [ ] **Step 4: Run every local dependency-light test**

Run:

```bash
python3 tools/test_chunked_prefill.py
python3 tools/test_arrival_load_gate.py
python3 tools/test_arrival_load_driver.py
python3 tools/test_arrival_load_verify.py
python3 tools/test_run_arrival_load_gate_remote.py
git diff --check
```

Expected: all five scripts print their pass sentinel and `git diff --check` emits no output.

- [ ] **Step 5: Confirm only intended tracked files changed**

Run:

```bash
git status --short
git diff --name-only
```

Expected tracked paths:

```text
tinyvllm/config.py
tinyvllm/engine/scheduler.py
tools/arrival_load_gate.py
tools/arrival_load_verify.py
tools/test_chunked_prefill.py
tools/test_arrival_load_gate.py
tools/test_arrival_load_driver.py
tools/test_arrival_load_verify.py
tools/run_arrival_load_gate_remote.sh
tools/test_run_arrival_load_gate_remote.py
```

`tinyvllm/engine/llm_engine.py` must remain unchanged. Existing untracked
`experiments/` directories remain present and unstaged.

- [ ] **Step 6: Commit remote-chain hardening**

```bash
git add tools/run_arrival_load_gate_remote.sh tools/test_run_arrival_load_gate_remote.py
git commit -m "test(gate): bind P4 remote evidence chain"
```

---

### Task 8: Execute the Source-Bound Remote Promotion Chain

**Files:**
- Create only untracked artifacts under `experiments/arrival_load/`
- Do not modify tracked source during this task

**Interfaces:**
- Consumes: a clean tracked source at the final implementation commit and a valid Kerberos cache.
- Produces: preflight, smoke, calibration, canonical, and independent-verifier artifacts for one immutable source tree.

- [ ] **Step 1: Verify Kerberos and SSH before creating a run**

Run:

```bash
export KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian
klist -s
ssh -n \
  -o BatchMode=yes \
  -o ConnectTimeout=20 \
  -o ControlMaster=auto \
  -o ControlPersist=600 \
  -S /tmp/ssh-sitian-10.232.195.203 \
  sitian@10.232.195.203 \
  'printf REMOTE_OK'
```

Expected: `klist -s` exit code 0 and `REMOTE_OK`.

If authentication is expired, stop the remote phase and ask the user to run:

```bash
KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian \
  kinit sitian@BYTEDANCE.COM
```

Do not classify an authentication artifact as a preflight result.

- [ ] **Step 2: Freeze the implementation source**

Run:

```bash
git status --short --branch
git rev-parse HEAD
git diff --check
```

Expected: no tracked changes; only preserved untracked `experiments/` roots.

- [ ] **Step 3: Run source-bound preflight**

```bash
export KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian
PREFLIGHT_RUN_TAG="qwen3-06b-sam-p4-preflight-$(date +%Y%m%d-%H%M%S)"
RUN_TAG="${PREFLIGHT_RUN_TAG}" \
  tools/run_arrival_load_gate_remote.sh preflight
```

Expected:

- remote dependency-light tests pass from the uploaded immutable snapshot;
- local `experiments/arrival_load/${PREFLIGHT_RUN_TAG}/source_preflight.json` exists;
- its source tree matches the locally staged source evidence.

- [ ] **Step 4: Run strengthened P0/P4 smoke**

```bash
SMOKE_RUN_TAG="qwen3-06b-sam-p4-smoke-$(date +%Y%m%d-%H%M%S)"
RUN_TAG="${SMOKE_RUN_TAG}" \
  tools/run_arrival_load_gate_remote.sh smoke
```

Expected:

- independent verifier exit code is `0`;
- smoke classification is `SMOKE_ONLY`;
- exact outputs and complete lifecycle are true;
- P4 trace contains `adaptive_mixed_prefill_decode`;
- no P4 structural failure exists.

- [ ] **Step 5: Run calibration bound to the smoke source**

```bash
CALIBRATION_RUN_TAG="qwen3-06b-sam-p4-calibration-$(date +%Y%m%d-%H%M%S)"
RUN_TAG="${CALIBRATION_RUN_TAG}" \
SMOKE_RUN_TAG="${SMOKE_RUN_TAG}" \
  tools/run_arrival_load_gate_remote.sh calibration
```

Expected: a frozen `lambda_ref`, workload manifest, source identity, environment identity, and P4 identity suitable for canonical.

- [ ] **Step 6: Run the full 54-case canonical matrix**

```bash
CANONICAL_RUN_TAG="qwen3-06b-sam-p4-canonical-$(date +%Y%m%d-%H%M%S)"
RUN_TAG="${CANONICAL_RUN_TAG}" \
SMOKE_RUN_TAG="${SMOKE_RUN_TAG}" \
CALIBRATION_RUN_TAG="${CALIBRATION_RUN_TAG}" \
  tools/run_arrival_load_gate_remote.sh canonical
```

Expected:

- exactly 54 unique case IDs;
- three measured repetitions for each `P0/P3/P4 × scenario`;
- unique dynamic port pairs for all model processes;
- local independent verification exit code `0`;
- complete scheduler, lifecycle, memory, source, environment, and policy evidence.

- [ ] **Step 7: Audit the independent result before making a claim**

Run:

```bash
python3 tools/arrival_load_verify.py \
  --run-dir "experiments/arrival_load/${CANONICAL_RUN_TAG}"
cat "experiments/arrival_load/${CANONICAL_RUN_TAG}/independent-verify/summary.json"
```

Acceptance checklist:

- top-level classification equals `candidate_results.P4.classification`;
- `candidate_results.P3` is diagnostic only;
- P4 has no structural or correctness failure;
- all p99 TTFT/ITL/E2E, maximum-decode-gap, service-bucket p95 E2E, and worst-repetition guards pass;
- one preregistered P4 benefit path passes against P0;
- P4 activation is material in burst and mostly inactive outside the intended backlog region.

If any guard fails, record `NO_GO`; do not call it an engine-wide speedup. If evidence is missing or contradictory, record `INCOMPLETE`.

- [ ] **Step 8: Update claims only after a verified result**

Only after the independent verifier succeeds:

- add the exact run tag, source tree, classification, ratios, limitations, and next action to `AGENT_HANDOFF_STATE.md`;
- update README only when the independently verified P4 classification is
  `GO`; otherwise leave README unchanged and record the negative result only
  in the handoff;
- keep all raw experiment artifacts untracked;
- selectively commit only the documentation changes.

---

## Final Completion Audit

Before declaring the implementation complete, verify each prompt-to-artifact mapping:

| Requirement | Concrete evidence |
|---|---|
| Disabled-by-default adaptive policy | `Config` default test and disabled schedule-equivalence test |
| Two high observations enter active | dependency-light scheduler transition and branch tests |
| Two low observations stop new admission | active-to-inactive/draining tests |
| Drain existing prefill only | draining queue-ownership test plus independent trace reconstruction |
| Two mixed steps force decode yield | scheduler test plus verifier rejection of third mixed row |
| No adaptive prefill-only batch with decode runnable | transactional helper test and branch-role verifier |
| No mutation before decode feasibility | queue/KV snapshot equality on failed required mixed attempt |
| Exact P3 shared behavior preserved | all pre-existing P3 mixed scheduler and postprocess tests |
| Immutable observation state | engine/driver snapshot tests |
| All five fields in identity | gate policy-identity test and verifier recomputation |
| P0/P3/P4 frozen 54-case matrix | matrix test and canonical manifest audit |
| P3 cannot promote P4 | harness and independent classification tests |
| Warmup lifecycle checked but excluded from metrics | existing verifier warmup tests remain green |
| Resume cannot cross P4 identity | threshold-drift resume test |
| Source changes require a new chain | predecessor source/identity checks and remote runner tests |
| Unique ports | gate allocator tests and final manifest audit |
| Remote-only GPU execution | exact host/runtime runner test and run commands |
| No performance overclaim | independent P4 `GO` is the sole promotion condition |

The work remains incomplete until every local test passes and the source-bound independent canonical verifier returns a classification for the final implementation source.
