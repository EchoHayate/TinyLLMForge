# Development Startup Evidence Implementation Plan

> **For agentic workers:** Use test-driven-development and independent review.

**Goal:** Check the fixed launcher's saved pre-input stdout before a successful
execution callback returns, without rerunning the launcher or claiming remote
attestation.

**Architecture:** One pure bytes-to-record function. The fixed remote driver and
its offline verifier can reuse it. Neither imports nor calls execute a worker,
read task data, load a model, launch a process or change existing modules.

**Tech Stack:** Existing Python 3.12, standard library, strict `parse_json`,
existing unittest.

## Global Constraints

- Implements the startup-evidence part of
  `../specs/2026-11-17-lora-development-remote-driver-design.md`.
- Create only `experiments/lora_necessity/development_startup_evidence.py` and
  `experiments/lora_necessity/test_development_startup_evidence.py`.
- No changes to accepted worker/launcher/observer/coordinator or their tests.
- No remote access, candidate execution, real task data, installation, telemetry,
  training, commits or pushes.
- Saved diagnostics describe the pre-candidate-input phase. Their false
  candidate-execution flags must remain false in the returned record; they are
  not evidence of whether later candidate execution happened.

## Task 1: Pure Fixed Startup Record Check

Interface:

```python
def verify_startup(stdout_raw: bytes) -> dict:
    ...
```

Return the independently parsed original diagnostic record, not an attestation
or modified success report. Raise ValueError on invalid bytes/record.
Accept only exact bytes, nonempty, at most 65536 bytes, a final LF, and exactly
four LF-delimited lines:

1. `USERNS_POLICY_enabled` or `USERNS_POLICY_unavailable`.
2. `HOST_NO_NEW_PRIVILEGES`.
3. `ALL_CAPABILITIES_ZERO`.
4. A strict JSON object with the sealed worker's exact eight fields.

Validate exact kind `FIXED_HARDENING_SMOKE_ONLY`, Python `3.12.13`, passed=true,
the exact 17 check names each boolean true, exact five capability names each
the string `0000000000000000`, ordered allowed-syscall list matching the sealed
worker source, and both candidate-execution fields boolean false. Strict JSON
parsing rejects duplicate keys and nonfinite values. No callback or filesystem
dependency is part of this function.

- [ ] Write synthetic success records and mutation tests before implementation.
  Include both userns marker variants, extra/missing/reordered markers, CRLF,
  trailing output, malformed/duplicate/nonfinite JSON, wrong type/size, each
  check missing/false/integer-substituted, capability drift, syscall drift,
  incorrect version/kind/phase flags and unknown/missing fields.
- [ ] Keep initial missing-component and subsequent behavioral RED distinct.
- [ ] Implement the smallest validator, using the existing strict JSON parser
  and native type checks. The immutable allowed/check constants come from the
  inspected sealed worker source, not from a task or generated program.
- [ ] Run focused unittest with `-B -W error`, then named pure startup/protocol
  tests; save actual counts and exit statuses in the existing private stage.
- [ ] Main statically compares constants and marker sequence to sealed sources,
  without importing or executing those sources. Independent review checks
  record semantics, mutation coverage and phase-limited claims.

Commands use:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=experiments/lora_necessity \
  /opt/homebrew/opt/python@3.12/bin/python3.12 -B -W error -m unittest \
  test_development_startup_evidence -v
```

This component does not establish actual isolation or admit a real run. The
trusted driver must first bind the bytes to the actual pinned child execution,
and the final independent verifier must retain that provenance.
