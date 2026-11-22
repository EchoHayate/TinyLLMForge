# Development Final Request Implementation Plan

> **For agentic workers:** Use subagent-driven-development with independent review.

**Goal:** Give final development candidates a distinct, answer-free request domain.

**Architecture:** A new pure protocol module validates a complete final-request
envelope and produces the existing three-field worker payload. It never relabels
a final request as a round-one public request and never launches a process.

**Tech Stack:** Existing Python 3.12, unittest, standard library and sealed pure helpers.

## Global Constraints

Follow `../specs/2026-11-17-lora-development-final-request-design.md`.
Only create `development_final_protocol.py` and `test_development_final_protocol.py`
under `experiments/lora_necessity`.
Do not modify old protocol/admission/controller/worker, generation or frozen data.
No model imports, actual candidate execution, remote operations, training, installs,
telemetry, commits or pushes in this component stage.
Use synthetic requests only; do not open task source/oracle files.
Final requests are D01–D04 and base/text only; round=2.
The new protocol alone grants no execution/scoring/benchmark readiness.

## Task 1: Pure Final Request Domain

**Create:** `experiments/lora_necessity/development_final_protocol.py`

**Test:** `experiments/lora_necessity/test_development_final_protocol.py`

**Consumes:** sealed `candidate_execution_protocol._source`, `_json_budget`,
`pilot_protocol.fields/require`, `validation_contract.digest/encoded`.

**Produces:** `prepare_request(request) -> (sha256_string, payload_bytes)`.

- [ ] Write a synthetic request fixture and tests before creating the module:

```python
def request():
    return {
        "kind": "LORA_DEVELOPMENT_FINAL_REQUEST", "version": 1, "round": 2,
        "run_id": "a" * 32, "attempt_id": "b" * 32,
        "generation_call_id": "c" * 32, "generation_sha256": "d" * 64,
        "case_set_sha256": "e" * 64, "task_id": "D01", "arm": "base",
        "case_group": "scoring", "case_id": "C01", "entry_point": "identity",
        "source": "def identity(value):\n    return value\n", "args": [[1, 2]],
    }
```

Check output equals `(digest(value), encoded({name: value[name] for name in
("source", "entry_point", "args")}) + b"\n")`. Mutate each identity/input field
individually and check the full digest changes. Mutate the original nested args
after preparation and check the returned bytes remain unchanged.

For negative cases, iterate missing/extra keys; bool/float versions; bad hex IDs;
T01/R01/S01 tasks; lora/lora_text arms; round1; public C01 or scoring P01;
empty/oversized/non-ASCII case IDs; non-list args; NaN/Infinity; malformed,
multiple-function, annotated or oversized source; surrogate text; recursive
containers; builtin subclasses with observable hooks. Require ValueError and
zero hook calls. Use source with a `while True` body to demonstrate preparation
does not run it. Check a syntax-valid but disallowed computation stays a
syntax-only result, not an admission result.

Construct a separate ordinary old request containing only its original fields
and assert the old protocol still accepts it. Both protocols must reject the
other domain's envelope.

- [ ] Run and retain missing-component RED:

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/opt/python@3.12/bin/python3.12 -B -W error \
  -m unittest discover -s experiments/lora_necessity \
  -p test_development_final_protocol.py -v
```

- [ ] Implement the pure validator with this exact data flow:

```python
import re

from candidate_execution_protocol import _json_budget, _source
from pilot_protocol import fields, require
from validation_contract import digest, encoded


def prepare_request(request):
    try:
        _json_budget(request, 1052672)
        fields(request, (
            "kind", "version", "round", "run_id", "attempt_id", "generation_call_id",
            "generation_sha256", "case_set_sha256", "task_id", "arm",
            "case_group", "case_id", "entry_point", "source", "args",
        ))
        require(request["kind"] == "LORA_DEVELOPMENT_FINAL_REQUEST", "final request kind")
        require(type(request["version"]) is int and request["version"] == 1, "version")
        require(type(request["round"]) is int and request["round"] == 2, "final round")
        for name, width in (
            ("run_id", 32), ("attempt_id", 32), ("generation_call_id", 32),
            ("generation_sha256", 64), ("case_set_sha256", 64),
        ):
            require(type(request[name]) is str
                    and re.fullmatch("[0-9a-f]{" + str(width) + "}", request[name]),
                    "final request identity")
        require(type(request["task_id"]) is str
                and re.fullmatch("D0[1-4]", request["task_id"]), "development task")
        require(type(request["arm"]) is str and request["arm"] in ("base", "text"),
                "development arm")
        require(type(request["case_group"]) is str
                and request["case_group"] in ("public", "scoring"), "case group")
        case_id = request["case_id"]
        require(type(case_id) is str
                and re.fullmatch("[A-Za-z][A-Za-z0-9_-]{0,31}", case_id), "case identity")
        require((case_id in ("P01", "P02")) == (request["case_group"] == "public"),
                "case group identity")
        require(type(request["args"]) is list, "request args")
        _source(request["source"], request["entry_point"])
        payload = {name: request[name] for name in ("source", "entry_point", "args")}
        _json_budget(payload, 1048575)
        return digest(request), encoded(payload) + b"\n"
    except (SyntaxError, UnicodeError, RecursionError) as error:
        raise ValueError("invalid final request encoding, depth or syntax") from error
```

The full JSON walk must precede field comparison/serialization so subclass
hooks cannot run. `kind` also needs an exact string check if a synthetic test
demonstrates a path not already excluded by that walk; do not relax type checks.

- [ ] Run the same focused command to GREEN. Then run old/new protocol tests:

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/opt/python@3.12/bin/python3.12 -B -W error \
  -m unittest discover -s experiments/lora_necessity -p 'test_*protocol.py' -v
```

- [ ] Save RED/GREEN commands, logs, source hashes and limits to
  `/tmp/tlf-candidate-executor-20261117-gb1gZX/development-final-protocol-report.md`.
  Do not report execution, scoring or real task acceptance.
- [ ] Independent review checks domain separation, no-answer payload, bounded
  exact types, frozen helper compatibility and tests; repair concrete findings.

## Task 2: Handoff

- [ ] Record accepted source identities and test results in the private ledger.
- [ ] Keep final-domain admission/controller and trusted case-source/scoring
  binding as separate subsequent work. Do not run a candidate with this module.
