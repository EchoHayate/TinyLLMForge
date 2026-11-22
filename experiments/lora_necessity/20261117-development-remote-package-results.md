# Fixed development package and reconstruction

Date: 2026-11-17. Component status: accepted after independent R1–R3 review.
No actual development package or remote experiment is accepted by this document.

The private implementation packages the fixed driver and source snapshots over
stdin, preserves a one-shot transport claim and binary streams, and reconstructs
the fixed eight-episode/sixteen-generation development evidence independently.
It does not replay the consumed smoke, train, mount adapters, retry tasks or
launch the full evaluation automatically.

## Verification

Main's final named synthetic regression: **439 tests passed in73.008s, EXIT=0**.
This includes62 package/transport/reconstruction tests and377 existing regressions;
earlier targeted and aggregate runs overlap and must not be added.

Independent review closed:

- Unknown execution coverage incorrectly reported as a complete cost total.
- Unterminated bytes after DONE escaping the terminal boundary.
- Invalid or unavailable export/coordinator timing values escaping validation.
- A repair-induced schema mismatch with the driver's model-load memory counters.

The final synthetic preparation includes the actual producer's memory-field
shape. Durations and byte counters are validated separately. Unknown work stays
unknown; observed partial costs are retained without being called full totals.
The original failures, intermediate errors, fixes and reviewer reports remain.

## Source identity

The four private files are retained under
`/tmp/tlf-candidate-executor-20261117-gb1gZX`, with exact copies in
`development-remote-package-r3-snapshot/`.

| File | SHA256 |
| --- | --- |
| development_remote_package.py | 20deec2d06af6bbdbdbe8911178953683f7c3ac7d4c570ff0505cb235e467a8f |
| development_remote_transport.py | 550ef6c2e97e32c1a2f0dd96903045ce7f7df8c55f5182585d473acd54308a3f |
| verify_development_remote.py | 89a1ad7739cc7e6359e9891c13606d9a8dc74a0bf6b2e02b9b3f2866065efd4b |
| test_development_remote_package.py | 1e6502de7c22516eee1a4c826ff1d2d20f894cf58eb224839a43750a8c5df21a |

Evidence in that same private directory:

- `development-remote-package-report.md`: authorship, exact commands and complete
  retained RED/GREEN history, including the fixture blind spot.
- `development-remote-package-review-r3.md`: bounded Spec PASS/Quality APPROVED;
  all findings from the review sequence closed.
- `development-remote-package-review/final_comments-r3.json`: empty findings.
- `development-remote-package-review-r3-combined-green.log`: final439-test log,
  SHA256 `76edbbb377bbe6cd71362b97143b3e291847df11a6b79ee900e4dab4ada07266`.

The accepted driver/coordinator, source dependencies and sealed launcher/worker
were not modified. No real task data, remote model or generated candidate was
executed by these component tests.

## Claim boundary

This qualifies the reviewed synthetic component, not current SSH access, an
actual package's provenance, arbitrary hostile-code safety or LoRA effectiveness.
Process completion, evidence verification, functional scores and calibration
remain distinct, with all scientific/readiness flags false.

The next gate separately binds actual source/data/route/runtime/model identities
and reviews the exact single-use package. The workload remains D01–D04/base/text,
two rounds per episode. It is a development check, not a LoRA comparison result.
