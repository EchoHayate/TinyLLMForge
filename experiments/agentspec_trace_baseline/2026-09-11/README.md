# Stage 1b step 0 artifacts, 2026-09-11

Derived statistics only. No corpus text is stored here, which keeps the
non-commercial corpus inside its licence and keeps the artifacts
reviewable.

## Inputs, not committed

```text
Salesforce/APIGen-MT-5k  apigen-mt_5k.json                CC BY-NC 4.0
  sha256 5225b54198c1d4d2ae9ff14ddd98341751677138efe64ded10dcb770276b5841

nebius/SWE-agent-trajectories  data/train-00000-of-00012.parquet  CC BY 4.0
  sha256 5a395e8c7bb8ddc4b8f4d268506b3a0e2cf9b5ec3922600117322fe788067a13
```

Intermediate normalised action files, also not committed because they
are 9.8 MB and 69 MB:

```text
apigen_actions.jsonl      sha256 be66c9fa95ff1e51f83afd3b34291638016fa540f51d65779545817815bd6214
swe_agent_actions.jsonl   sha256 e81a4de1d47ed9025407009875381f300a240bb3e58d36268bddff046cf2e484
```

## Regenerate

```bash
huggingface-cli download Salesforce/APIGen-MT-5k apigen-mt_5k.json \
    --repo-type dataset --local-dir .
huggingface-cli download nebius/SWE-agent-trajectories \
    data/train-00000-of-00012.parquet --repo-type dataset --local-dir .

python3 tools/agentspec_trace_normalize.py --corpus apigen \
    --input apigen-mt_5k.json --output apigen_actions.jsonl
python3 tools/agentspec_trace_normalize.py --corpus swe_agent \
    --input data/train-00000-of-00012.parquet \
    --output swe_agent_actions.jsonl

python3 tools/agentspec_trace_match_baseline.py apigen_actions.jsonl \
    --output baseline_apigen.json
python3 tools/agentspec_trace_match_baseline.py swe_agent_actions.jsonl \
    --output baseline_swe_agent.json

python3 tools/agentspec_trace_copyability.py --corpus apigen \
    --input apigen-mt_5k.json --output copyability_apigen.json
python3 tools/agentspec_trace_copyability.py --corpus swe_agent \
    --input data/train-00000-of-00012.parquet --limit 1200 \
    --output copyability_swe_agent.json
```

`--limit 1200` on the SWE-agent copyability scan is a runtime bound, not
a sampling design. The scan is quadratic in trace length because the
context grows with every step, and 1200 traces already yields 30506
eligible steps.

## Outputs

```text
baseline_apigen.json       197172d3514cf31d14e05af8183329eacc187f31a0fa4d99c137d7ab18ba5e7e
baseline_swe_agent.json    ed8bdc1816e92710cf3bf59535ec6dc29e7762635961920df706a8baec330042
copyability_apigen.json    61281efc40cd552c7d6e3517b6f69a6e9d75d8008ff1420914e50c0228226d45
copyability_swe_agent.json 37e472c586ffd34ebe85a58671e1a201a0fc8ee0b9c480ff4f9d4b02dcc8aec5
```

Read `docs/superpowers/plans/2026-09-11-latent-action-speculation-stage1b-step0.md`
for the interpretation and the pre-registered revision.
