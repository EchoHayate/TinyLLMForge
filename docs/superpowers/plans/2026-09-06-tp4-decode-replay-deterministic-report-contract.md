# TP4 Decode Replay Deterministic Report Contract Plan

Date: 2026-09-06

## Goal

Close the `report.md` producer/verifier blind spot without changing any frozen
qualification gate or mutating historical evidence.

## Steps

1. Add RED assembler coverage requiring `report.md`, its manifest hash, and
   deterministic content derived from strict-clean and shared-capacity
   fixtures.
2. Add RED verifier mutations for:
   - missing report;
   - report hash drift;
   - rehashed semantic drift;
   - producer/verifier implementation independence.
3. Add a deterministic assembler renderer and write `report.md` before the
   manifest.
4. Add a verifier-owned renderer and require exact byte equality after
   independent evidence reconstruction.
5. Run the focused assembler/verifier tests, the adjacent TP4 replay suite,
   `py_compile`, and `git diff --check`.
6. Review exact-path diff and status, commit with the required attribution,
   push only `feat/kv-sparse-attention`, and verify local/remote SHA equality.
7. Use only a fresh tag for subsequent remote qualification; do not rewrite
   r48.
