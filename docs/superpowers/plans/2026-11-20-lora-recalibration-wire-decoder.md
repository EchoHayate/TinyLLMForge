# Recalibration Evidence Wire Decoder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decode the fixed five-frame evidence wire without losing bytes or admitting malformed, oversized or digest-mismatched records.

**Architecture:** Implement the byte-framing layer of the approved evidence reconstruction responsibility. A small shared pure helper lets the future controller inspect an isolated READY frame before payload delivery and lets the offline verifier split a saved complete wire. This is not the semantic READY/DONE validator or the complete evidence report.

**Tech Stack:** Python standard library (`hashlib`, `re`, `struct`), existing `pilot_protocol.require`, and unittest. No installation.

## Global Constraints

- Approved design: `docs/superpowers/specs/2026-11-20-lora-recalibration-fixed-execution-design.md`, SHA256 `88f69b71d60fc7165d4e67fa739f8a3a7e743ee0456551a1adf6d7e1ecd905c3`.
- Scope is the design's section 4.3 byte-framing layer only. The helper is shared by evidence/controller responsibilities, not a new execution path.
- Fixed magic ASCII `LRV01`, header network byte order `!5sBQ`, 14 bytes.
- Exactly five channels, in order: 1 READY, 2 BACKEND_STDOUT, 3 BACKEND_STDERR, 4 DIAGNOSTIC, 5 DONE.
- Payload limits respectively: 4096, 8388608, 131072, 65536, 4096 bytes. Total complete wire including all headers is at most 8388608 bytes.
- Single-frame decoding enforces its channel's payload limit, not the complete-wire aggregate limit. Consequently a channel-2 frame at its payload limit cannot fit in a valid complete wire.
- Inputs are strict built-in bytes; expected channel is strict built-in int 1–5 (not bool); whole-wire expected SHA256 is strict built-in str matching exactly 64 lowercase hex digits.
- No filesystem, network, subprocess, environment, clock, randomness, stored-source execution or tokenizer/model imports in production.
- No SSH, real preflight, task execution, tokenizer/model loading, generation, training, installation, telemetry, bootstrap/flush, branch/worktree creation, cleanup, commit or push.
- Preserve all 187 frozen baseline LoRA files, prior source/test/data pins and old receipts. Only the two new files below may be written by the implementer.
- Use apply_patch; no inline code comments or single-letter variable names. Existing unittest style overrides test-skill defaults; explicit no-network/no-telemetry/no-install requirements override those templates.
- READY/DONE bytes remain opaque here: empty bytes, malformed JSON, duplicate JSON fields and extra semantic fields are not accepted as valid READY/DONE messages, merely returned unchanged as frame payloads. The future semantic verifier must parse strict JSON and check exact schemas, identities and lifecycle. This decoder exposes no success, authorization or provenance flags.
- A caller-supplied matching digest is integrity relative to that supplied value, not authentication. No complete launch binding or real evidence acceptance follows from successful framing.
- Whole-wire decoder does not salvage a prefix on error; callers retain original failed bytes. Single-frame decoder does not permit trailing data.

---

### Task 1: Strict Byte Framing Decoder

**Files:**
- Create: `experiments/lora_necessity/recalibration_execution_wire.py`
- Create: `experiments/lora_necessity/test_recalibration_execution_wire.py`
- Parent owns plan/ledger/results and separate parent verification artifacts.

**Interfaces:**
- Consumes `pilot_protocol.require(condition, message)`, which raises ValueError.
- `decode_frame(frame_raw, *, expected_channel) -> bytes`: validate one exact full frame and return its opaque payload bytes. Wrong channel, bad magic, short header/body, over-limit declaration, or trailing bytes raises ValueError.
- `decode_wire(wire_raw, *, expected_sha256) -> dict`: check strict bytes/whole-wire budget and strict SHA256 syntax, then hash original bytes before any header parsing. Only then decode exactly channels 1..5 and reject all trailing bytes.
- Return keys exactly `ready_raw`, `backend_stdout_raw`, `backend_stderr_raw`, `diagnostic_raw`, `done_raw`; all values bytes. Do not decode text/JSON, canonicalize, trim, skip diagnostic bytes, convert to hex or add semantic flags.
- Constants exactly `MAGIC`, `HEADER`, `WIRE_LIMIT`, `FRAME_LIMITS`, `FRAME_NAMES` below. Limits/names are tuples, so no output aliases mutable constants.
- Tests use small synthetic opaque bytes, independent byte encoders and selective limit-sized buffers; never load the task dataset or execute stored programs.
- All Global Constraints apply. Evidence root is `/private/tmp/lora-recalibration-wire-20261120-lrR7Iy`.

- [x] **Step 1: Write complete failing tests**

Use discovery-safe import and a failing setUp assertion before production exists:

```python
import hashlib
import importlib.util
import unittest

if importlib.util.find_spec("recalibration_execution_wire") is not None:
    import recalibration_execution_wire as wire
else:
    wire = None


def frame(channel, payload, declared=None, magic=b"LRV01"):
    size = len(payload) if declared is None else declared
    return magic + bytes([channel]) + size.to_bytes(8, "big") + payload


class WireDecoderTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(wire, "wire decoder not implemented")

    def test_preserves_all_opaque_bytes(self):
        payloads = [b'{"unverified":true}', b"\x00\xff\n", b"\xfe",
                    b"diagnostic\x00", b'{"failed":true}']
        raw = b"".join(frame(channel, payload)
                       for channel, payload in enumerate(payloads, 1))
        decoded = wire.decode_wire(
            raw, expected_sha256=hashlib.sha256(raw).hexdigest())
        self.assertEqual(list(decoded.values()), payloads)
```

Write discoverable tests for these requirements before implementation:

1. Independent hand-encoded golden frame header: 5 magic bytes, one channel byte,
   eight big-endian length bytes; binary and embedded header-like payloads unchanged.
2. Exact five-key ordered output, fresh dict each call, deterministic values, no
   mutation or alias to later outputs. Empty payloads and non-JSON/control bytes
   are valid framing but imply no message validity or execution success.
3. `decode_frame` works for each channel, including just READY before any DONE;
   a whole-wire value is not accepted as one exact frame.
4. Wrong input types: bytes subclass, bytearray, memoryview, str, None; channel
   bool/int subclass, float, str, 0, -1, 6; digest subclasses, uppercase, wrong
   length, newline, invalid hex, non-str. Each raises ValueError.
5. Wrong magic (including old/near-match magic), unknown/repeated/skipped/swapped
   channels, each missing frame, duplicated DONE and trailing byte all rejected.
6. Every byte-prefix truncation of a small valid complete wire fails; frame
   truncation at each header/body boundary fails. A smaller declared length leaves
   trailing bytes; a larger declared length fails. No prefix-success result.
7. Header declarations at every channel limit + 1 and at `2**64-1` fail without
   allocation based on the declaration. Wrong little-endian multi-byte length fails.
8. Each single-frame payload limit is inclusive; limit+1 is rejected. These buffers
   are bounded by ~8 MiB, generated and released one at a time.
9. Whole wire exactly 8388608 bytes including 70 header bytes succeeds using empty
   other frames and channel-2 payload `8388608 - 70`; plus one byte fails even though
   the channel-2 payload remains within its individual limit.
10. Individually valid max channel-2 frame decodes alone but is rejected within a
    complete wire. No sum-of-frame-limits relaxation.
11. A stale digest rejects a one-byte same-length mutation even if framing remains
    valid. A matching digest never excuses malformed framing. For malformed bytes
    plus wrong digest, assert `wire digest` error to prove digest-before-header order.
12. Public API exactly the two stated functions (private helper excluded); signature
    keyword-only arguments. Inspect AST/imports and module observations: only pure
    dependencies, no I/O or forbidden modules; do not invoke forbidden APIs.

No mocks of the function or hash/parser are needed. Use subTest for bounded
adversarial cases, with assertions on actual bytes, return shape or ValueError.
Do not import old TestCase classes or duplicate their discovery.

- [x] **Step 2: Preserve genuine RED**

Run from repository root, save raw stdout/stderr to `task-1-red.log`, capture
immediate exit code in `task-1-red.exit`, and read the failure reason:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/experiments/lora_necessity" python3 -m unittest test_recalibration_execution_wire -v
```

Expected assertion failures `wire decoder not implemented`, not import errors.

- [x] **Step 3: Implement the minimal pure decoder**

Candidate implementation; correct demonstrated defects rather than treating this
candidate as infallible:

```python
import hashlib
import re
import struct

from pilot_protocol import require


MAGIC = b"LRV01"
HEADER = struct.Struct("!5sBQ")
WIRE_LIMIT = 8388608
FRAME_LIMITS = (4096, 8388608, 131072, 65536, 4096)
FRAME_NAMES = (
    "ready_raw", "backend_stdout_raw", "backend_stderr_raw", "diagnostic_raw", "done_raw",
)


def _read_frame(raw, offset, expected_channel):
    require(len(raw) - offset >= HEADER.size, "short frame header")
    magic, channel, size = HEADER.unpack_from(raw, offset)
    require(magic == MAGIC, "frame magic")
    require(channel == expected_channel, "frame channel order")
    require(size <= FRAME_LIMITS[channel - 1], "frame payload limit")
    start = offset + HEADER.size
    stop = start + size
    require(stop <= len(raw), "short frame payload")
    return raw[start:stop], stop


def decode_frame(frame_raw, *, expected_channel):
    require(type(expected_channel) is int and 1 <= expected_channel <= 5,
            "expected channel")
    require(type(frame_raw) is bytes
            and len(frame_raw) <= HEADER.size + FRAME_LIMITS[expected_channel - 1],
            "frame bytes")
    payload, stop = _read_frame(frame_raw, 0, expected_channel)
    require(stop == len(frame_raw), "trailing frame bytes")
    return payload


def decode_wire(wire_raw, *, expected_sha256):
    require(type(wire_raw) is bytes and len(wire_raw) <= WIRE_LIMIT, "wire bytes")
    require(type(expected_sha256) is str
            and re.fullmatch("[0-9a-f]{64}", expected_sha256) is not None,
            "expected wire digest")
    require(hashlib.sha256(wire_raw).hexdigest() == expected_sha256, "wire digest")
    offset = 0
    decoded = {}
    for channel, name in enumerate(FRAME_NAMES, 1):
        payload, offset = _read_frame(wire_raw, offset, channel)
        decoded[name] = payload
    require(offset == len(wire_raw), "trailing wire bytes")
    return decoded
```

- [x] **Step 4: Focused GREEN and one narrow integration regression**

Focused command is the same as Step 2. Save `task-1-green.log/.exit`.
Then run exactly:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/experiments/lora_necessity" python3 -m unittest test_recalibration_execution_package test_recalibration_execution_wire -v
```

Save `task-1-regression.log/.exit`; expected all discovered tests pass with no skips
or errors. This count includes the focused tests plus the existing 18 package tests.
Do not rerun the previous 66/157 suites. The regression imports both future-neighbor
modules in one interpreter; it is not a implemented launcher integration test.
Verify all 187 baseline file hashes and new source/test AST/style, saving evidence.

- [x] **Step 5: Report and fresh review**

No commit. Implementer writes `task-1-report.md` with exact commands, test counts,
raw logs/exit files, RED/GREEN sequence, final source/test hashes and concerns.
Parent creates actual added-file diff, independently checks golden frame bytes and
boundary behavior, and requests fresh spec/code review of the two additions.
Any Critical/Important fixes require focused verification and re-review.
Parent closes plan/ledger/results only after review and final preservation.

Completed 2026-11-20: genuine RED 31 assertion failures; GREEN 31 passed;
narrow regression 49 passed, including those 31. Independent parent byte checks
and 187-file preservation passed. Fresh reviewer Volta approved both spec
compliance and code quality with zero P0/P1/P2 findings against exact unchanged
source/test/diff hashes. Imported `require` uses a private `_require` alias to
preserve the exact two-public-function API. No code/test corrections after GREEN
or duplicate same-hash suite reruns were needed.
Result: `experiments/lora_necessity/20261120-recalibration-wire-decoder-results.md`.

## Plan Self-Review

- Covered: binary framing, strict types, channel order, per-frame/aggregate byte
  budgets, exact raw-byte preservation and whole-wire integrity before parsing.
- Not covered: strict READY/DONE JSON schemas, contextual identity and lifecycle
  binding, backend `_process_result` reconstruction, 480-row report integration,
  streaming process I/O, persistence, full launch config, preflight or authorization.
- Keeping payloads opaque avoids inventing a partially frozen semantic schema
  or conflating valid framing with valid execution evidence. It also permits
  retention of failure diagnostics without text decoding.
- The separate wire helper is required by both future READY-handshake and
  offline complete-record consumers. It does not add a fifth runtime responsibility.
- Code and signatures agree with the tests. Invalid channel is checked before
  indexing limits; header length and declared budget precede payload slicing.
- No user approval is inferred for real execution; current continuation advances
  the already approved local implementation only.
