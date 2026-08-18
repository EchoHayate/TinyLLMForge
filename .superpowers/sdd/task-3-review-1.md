### Spec Compliance

- ❌ Issues found: phase-exit clock failures can replace the original exception and leak active state; failed steps can mutate the previous step observation; conservation accepts out-of-envelope phases.
- ⚠️ Real GPU/CUDA runtime behavior is not part of this local task review.

### Strengths

- Immutable identity, fixed phase inventory, explicit skipped rows, bounded snapshots, and lifecycle rejection are present.
- Task 2 receives the live step/repeat identity with narrow import handling.
- Final batch kind and deterministic selected-sequence digest are bound before dispatch.
- Existing rollback, commit, poison, and seal boundaries remain structurally intact.

### Issues

#### Critical

- None.

#### Important

- Phase exit reads the clock before clearing `_active_phase`. A clock failure can replace the wrapped operation's exception, then prevent `finish_step()` cleanup and leak recorder/ContextVar state. Cleanup must be exception-safe and preserve the original operation exception.
- Failed-step finalization can attach the new failed step telemetry to a previous successful `last_step_observation`, producing mixed-step data. Publish a coherent failed-step observation or keep the prior observation untouched and expose failure telemetry separately.
- Conservation does not require executed phase intervals to lie inside the step interval. Reject out-of-envelope phase start/end values fail-closed.

#### Minor

- Default-off `LLMEngine.step()` still enters generator context managers for every phase. Add a true no-op/bypass path so disabled instrumentation does not construct phase contexts on the hot path.

### Assessment

**Task quality:** Needs fixes

**Reasoning:** Core instrumentation is promising, but exception cleanup, failed-observation ownership, and malformed conservation evidence are not yet trustworthy.
