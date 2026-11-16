# Standard Tokenizer Runtime Implementation Plan

> **For agentic workers:** Execute inline with `executing-plans`; use an independent final reviewer. This is the user's approved November 14 simplification, not another dependency-audit project.

**Goal:** Restore normal tokenizer imports without weakening input identity or token-accounting checks.

**Architecture:** Keep `RuntimeGuard.start/check/close` as a one-use, main-thread, fixed-environment lifecycle check. Remove its import finder, discovery wrapper, denied loader and post-import name blacklist. The only application loader remains `AutoTokenizer.from_pretrained` on the five pinned local snapshot files.

**Tech Stack:** Python 3.12, existing unittest suite, pinned remote Transformers 4.57.1.

## Global Constraints

- User approved the proposed simplification with “继续吧” after rejecting per-library revalidation.
- Work in the existing `experiments/feedback-active-verification` checkout. Preserve other work; no branch/worktree changes, installation, commit or push.
- Keep `USE_TORCH/USE_TF/USE_FLAX=0`, offline flags and single-thread settings; add empty `CUDA_VISIBLE_DEVICES` to checked environment.
- A fresh process must establish those settings before the tokenizer stack or model backends have been imported. Imports during the operation use Python's normal machinery.
- Normal imports are not evidence of model loading. This guard is not a security sandbox for hostile library code; do not claim that it prevents arbitrary model API calls or all file/network access.
- Retain exact runtime inventory, five-file snapshot, local-only loading, `trust_remote_code=False`, fast tokenizer identity, 4096/256 limits, no truncation, immutable observations, sequence binding and sticky session failures.
- No generation, weights, training or GPU workload. No tool telemetry or private inputs sent to a new recipient. Use existing local tests rather than install or run a separate reporting service.
- At most one new real token audit after tests/review, against the existing approved server, using a fresh run/profile/manifest and unchanged collector. Do not reuse a consumed launch.

## Task 1: Simplify The Runtime Guard

**Files:** `experiments/lora_necessity/tokenizer_runtime_guard.py`,
`experiments/lora_necessity/test_tokenizer_runtime_guard.py`,
`experiments/lora_necessity/test_tokenizer_session.py`.

**Interface:** Keep `RuntimeGuard()` with `active`, `failed`, `start()`, `check()`, `close()`. No application caller needs `find_spec` or `_forbidden`; only old guard tests used those implementation details.

- [x] Replace obsolete blacklist/wrapper tests with normal-import tests and retained lifecycle checks. A temporary real package must resolve `configuration_utils -> modeling_gguf_pytorch_utils -> safetensors` without calling its sentinel weight-loader function.
- [x] Assert `sys.meta_path` and `importlib.util.find_spec` remain identical through start/check/close. Assert an ordinary missing optional import does not poison the guard.
- [x] Run the new guard tests against the old source and retain the expected failures.
- [x] Reduce the implementation to environment/lifecycle ownership checks:

```python
def check(self):
    try:
        require(self.active and not self.failed, "runtime guard inactive or failed")
        require(threading.current_thread() is threading.main_thread(), "main thread required")
        require(all(os.environ.get(name) == value for name, value in ENVIRONMENT.items()),
                "runtime environment changed")
    except BaseException:
        self.failed = True
        raise
```

- [x] Replace session-test import-denial injections with caught environment-check failures. Obtain the existing owner through `_OWNER[0]()` rather than searching import finders. Keep failure sanitization, reentry and cleanup assertions.
- [x] Run `python3.12 -B -m unittest discover -s experiments/lora_necessity -p 'test_tokenizer_*.py' -q`, then the entire `test_*.py` suite. Expected: pass, with only pre-existing platform skips.

## Task 2: Verify The Actual Task

**Files:** append existing progress ledger and create
`experiments/lora_necessity/20261114-standard-tokenizer-results.md`.

- [x] Review the actual diff and tests independently; do not audit every Transformers dependency.
- [x] Build a fresh launch with `build_payload`/`build_launch` from current sources, the previously approved inventory and exact runtime/file pins; only the profile run ID changes.
- [x] Verify current credentials and run unchanged `execute_once` once. Retain transport failure if one occurs; no automatic retry.
- [x] Classify the actual return. Only READY arrived before termination at 539.023 seconds; no result exists to reconstruct. Actual checked/overflow/missing counts are unknown, not zero. A successful transport would not itself be a passed token audit.
- [x] Record source hashes, evidence paths and remaining experiment prerequisites. Preserve old failed runs unchanged.

## Plan Check

The proposed change removes the cause of name-based incompatibility rather than introducing a GGUF whitelist. Existing loaders, measurements and transport code do not change. The real audit, not synthetic module imports, decides tokenizer compatibility. No automatic commits or additional approval loop is needed for this already-approved direction.
