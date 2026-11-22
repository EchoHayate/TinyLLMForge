# Development final-request qualification

Date: 2026-11-17. Branch: `experiments/feedback-active-verification`.
This is component qualification, not a generated-candidate benchmark.

## Delivered

`development_final_protocol.py` provides a separate D01–D04, base/text,
round-two request domain. It binds run, generation, complete case-set, task,
arm and case identity into the canonical request digest. The worker receives
only the candidate source, entry point and one case's arguments.

The sealed first-round public protocol remains unchanged. Final requests are
not disguised as first-round public requests. Source parsing is syntax-only:
this component neither executes code nor establishes security admission.

An identity-only native JSON preflight rejects custom classes and metaclasses
before the reused byte-budget helper can invoke class equality behavior.
The new entry point is protected; no global repair of sealed helpers is claimed.

## Verification

- Implementation RED: 27 missing-component failures.
- Additional metaclass-hook regression: 1 failure before repair, 1 pass after.
- Final focused suite: 28 passed. Main independently reran all 28 successfully.
- Old/new protocol aggregate: 83 passed, including the new 28.
- Main full component regression: 735 run, 733 passed, 2 skipped; 221.689 seconds,
  exit 0. This run preceded addition of the final admission/controller files.
- Independent protocol review: specification PASS and scoped code-quality PASS.
- Preservation check: all 60 generation-stage, 51 executor-stage and 174 earlier
  manifest entries match their retained hashes.

The tests use synthetic requests. The existing protocol aggregate and full suite
also include their pre-existing frozen-data checks; those are not new model
outputs or repeats of the completed remote validation workload.

Source SHA256:

```
development_final_protocol.py
9315bfec3ed19604c21d8ec6a944d264828f8e92618416748d377d77f36ad141
test_development_final_protocol.py
97ea78b74279333c8ded1f8bb7f1aa46d4c15e811a142eee62cb17a2dcf0580f
```

Commands:

```sh
PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/opt/python@3.12/bin/python3.12 -B -W error \
  -m unittest discover -s experiments/lora_necessity \
  -p test_development_final_protocol.py -q
PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/opt/python@3.12/bin/python3.12 -B -W error \
  -m unittest discover -s experiments/lora_necessity -p 'test_*protocol.py' -v
PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/opt/python@3.12/bin/python3.12 -B -W error \
  -m unittest discover -s experiments/lora_necessity -p 'test_*.py' -q
```

## Evidence and limits

Retained local evidence under `/tmp/tlf-candidate-executor-20261117-gb1gZX`:

- `development-final-protocol-report.md`
- `development-final-protocol-review.md`
- `development-final-protocol-final-green.log`
- `development-final-protocol-combined.log`
- `development-final-protocol-full-regression.log`

Generation and case-set hashes are format-checked and digest-bound, not
independently authenticated here. Final admission/controller, frozen-case
reconstruction, scoring and coordinator bindings are separate components.
No real development candidate, final case or LoRA comparison was run in this
qualification. No task benefit or production inference capability is established.
