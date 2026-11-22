# Development Final Execution Implementation Plan

> **For agentic workers:** Use subagent-driven-development with independent review.

**Goal:** Admit and control final-development requests without changing the public domain.

**Architecture:** A separately named controller keeps the sealed lifecycle body.
Final admission reuses the same AST policy and pure lexical helpers, but binds the
new final request and records a distinct admission kind before launch.

**Tech Stack:** Existing Python 3.12 and unittest; trusted local control-frame helpers.

## Global Constraints

Follow `../specs/2026-11-17-lora-development-final-execution-design.md`.
Prerequisite: accepted `development_final_protocol.prepare_request`.
Only create the two final modules and their tests under `experiments/lora_necessity`.
No edits to sealed sources, tests, reports, data, runtime, or historical artifacts.
No remote operations, pretrained models, training, generated-source execution,
installations, telemetry, commits or pushes.
Local helpers are trusted literal programs that emit control frames; they must not
interpret the request's source.
No actual scoring or task/LoRA readiness claim follows from these tests.

## Task 1: Final Controller and Admission

**Create:**
- `experiments/lora_necessity/development_final_controller.py`
- `experiments/lora_necessity/development_final_admission.py`
- `experiments/lora_necessity/test_development_final_controller.py`
- `experiments/lora_necessity/test_development_final_admission.py`

**Interfaces:**
- Final controller retains `run_request(request, argv, *, artifact_dir, cwd, env,
  startup_timeout=10.0, candidate_timeout=10.0, output_limit=65536,
  result_limit=16384, reap_timeout=2.0)`.
- Admission retains `admit_request(request)` and `run_candidate_request(request,
  argv, *, artifact_dir, cwd, env, **limits)`, now exclusively final-domain.
- Both consume the accepted final `prepare_request`; worker/result lifecycle types
  still come from the sealed original protocol.

- [ ] Start with RED tests for missing final modules and this synthetic final request:

```python
request = {
    "kind": "LORA_DEVELOPMENT_FINAL_REQUEST", "version": 1, "round": 2,
    "run_id": "a" * 32, "attempt_id": "b" * 32,
    "generation_call_id": "c" * 32, "generation_sha256": "d" * 64,
    "case_set_sha256": "e" * 64, "task_id": "D01", "arm": "base",
    "case_group": "scoring", "case_id": "C01", "entry_point": "toy",
    "source": "def toy(value):\n    return value\n", "args": [7],
}
```

Before implementation run:

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/opt/python@3.12/bin/python3.12 -B -W error \
  -m unittest discover -s experiments/lora_necessity \
  -p 'test_development_final_*.py' -v
```

- [ ] Reuse original controller lifecycle tests through a module reference:

```python
import test_candidate_execution_controller as original_tests

class FinalControllerTests(original_tests.ControllerTests):
    def setUp(self):
        super().setUp()
        self.module = importlib.import_module("development_final_controller")
        self.request.update({
            "kind": "LORA_DEVELOPMENT_FINAL_REQUEST", "round": 2,
            "run_id": "a" * 32, "generation_call_id": "c" * 32,
            "generation_sha256": "d" * 64, "case_set_sha256": "e" * 64,
            "case_group": "scoring", "case_id": "C01",
        })
```

Read every inherited test before relying on it. Add tests that reject old/public
envelopes before launch, bind changes in final metadata to different receipt
hashes, and verify the helper reads exactly three payload fields. Assert the
complete final-controller text equals the exact transformation below.

- [ ] Create the controller using `apply_patch`, with its complete contents defined
  by this deterministic, independently checkable transformation:

```python
original = Path("experiments/lora_necessity/candidate_execution_controller.py").read_text()
assert hashlib.sha256(original.encode()).hexdigest() == (
    "a9c72892b97eb4cb69492cabd53cff3be2822a4837291fc59709d0be61ee4bc0")
old = (
    "from candidate_execution_protocol import (\n"
    "    InfrastructureError, READY, make_receipt, prepare_request,\n"
    ")\n"
)
new = (
    "from candidate_execution_protocol import InfrastructureError, READY, make_receipt\n"
    "from development_final_protocol import prepare_request\n"
)
assert original.count(old) == 1
expected_controller = original.replace(old, new)
```

No other controller text changes. This snippet defines the file content; use the
edit tool rather than writing files from a shell script.

- [ ] For final admission, import policy constants, `CandidateAdmissionError`,
  `_require`, `_plain_json`, `_bindings`, `_local_bindings`, `_check_scope` from
  `candidate_execution_admission`; import new `prepare_request`.
  Preserve the complete original `admit_request` function body except that its
  successful kind becomes `DEVELOPMENT_FINAL_AST_ADMISSION`. Because the imported
  validator is the new one, no old request conversion is involved. Include a
  source/AST equivalence test of the computation checks against the sealed
  function, excluding only this literal evidence-kind change.

- [ ] Write persistence-first run logic as follows:

```python
def _save_admission(path, admission):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        output = os.fdopen(descriptor, "wb")
    except BaseException:
        os.close(descriptor)
        raise
    with output:
        raw = encoded(admission)
        _require(output.write(raw) == len(raw), "incomplete admission record write")


def run_candidate_request(request, argv, *, artifact_dir, cwd, env, **limits):
    import development_final_controller

    try:
        _plain_json(request)
        prepare_request(request)
        snapshot = parse_json(encoded(request), limit=1052672)
    except (ValueError, RecursionError) as error:
        raise CandidateAdmissionError("request not qualified") from error
    admission = admit_request(snapshot)
    try:
        _save_admission(Path(artifact_dir) / "admission.json", admission)
    except (OSError, ValueError) as error:
        raise CandidateAdmissionError("admission record could not be persisted") from error
    result = development_final_controller.run_request(
        snapshot, argv, artifact_dir=artifact_dir, cwd=cwd, env=env, **limits)
    _require(result["receipt"]["request_sha256"] == admission["request_sha256"],
             "admitted request changed")
    return {**result, "admission": admission}
```

Imports required by these functions and the copied AST body are `ast`, `os`,
`Path`, `encoded`, `parse_json`, the policy constants/helpers above, and the new
`prepare_request`. Keep source and request snapshots native JSON; never invoke a
candidate while validating it.

- [ ] Test exact `DEVELOPMENT_FINAL_AST_ADMISSION`, policy v2 identity, final request
  binding, immutable snapshot and controller result binding. Use patched controller
  callbacks only to check ordering and capture request snapshots. In failure tests
  inject refusal, duplicate file, fdopen/write/short-write/close failures and assert
  controller was never called. Check fdopen failure does not leak the descriptor.

For security regressions use:

```python
denied = (
    "return __import__('os').getpid()",
    "return open('/tmp/x').read()",
    "return value.__class__",
    "return map(eval, value)",
    "items = [eval for eval in value]\n    return map(eval, value)",
    "len = value\n    return len(value)",
)
```

Use ordinary arithmetic, comprehensions, approved list/string methods and recursive
entry calls as positive static cases. Also test hostile builtin subclasses do not
invoke hooks, and public-domain input never produces final admission or a child.
End with one trusted literal helper integration through admission and controller;
it consumes but never executes the source and returns a fixed JSON outcome.

- [ ] Run GREEN with the same focused invocation. Then run the named old/new
  protocol/admission/controller suites, without candidate-worker execution tests
  or any remote diagnostic:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=experiments/lora_necessity \
  /opt/homebrew/opt/python@3.12/bin/python3.12 -B -W error -m unittest \
  test_candidate_execution_protocol test_candidate_execution_admission \
  test_candidate_execution_controller test_development_final_protocol \
  test_development_final_admission test_development_final_controller -v
```

The earlier broad `test_*execution*.py` pattern also selected worker tests that
execute their synthetic payload source. The explicit module list reconciles
regression coverage with this component's trusted-control-frame-only constraint.

- [ ] Report commands, tests, source hashes, source-equivalence checks and limits to
  `/tmp/tlf-candidate-executor-20261117-gb1gZX/development-final-execution-report.md`.
- [ ] Independent review checks policy equality, domain separation, persistence
  ordering and trusted-helper lifecycle; resolve concrete findings before real use.

## Task 2: Integration Boundary

- [ ] Record accepted new files in the progress ledger.
- [ ] Keep trusted full-case provenance, generation linkage, final scoring and
  development calibration in the following coordinator task. Do not execute
  real candidates or label these component tests as development results.
