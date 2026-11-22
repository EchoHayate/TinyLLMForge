# LoRA Generation Core Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development for this
> single component and its review; no old sessions or agents are resumed.

**Goal:** Add a fixed greedy candidate generator without running a real evaluation.

**Architecture:** A pure orchestration function consumes a model, tokenizer and
messages. Existing tokenizer measurement performs template/input validation.
A no-cache torch forward loop implements only the frozen greedy policy.

**Tech Stack:** Existing Python 3.12, unittest, PyTorch and local synthetic Qwen3.

## Completion Record

- [x] Task 1 completed:24 expected RED failures,24 GREEN passes; pinned
  independent Spec PASS / Quality APPROVED.
- [x] Task 2 completed:random CPU Qwen3 output IDs match the built-in greedy
  reference for all256 tokens; full suite566 run/564 pass/2 skip, exit0.
  Old61 workspace/56 training-private/250 tail-private/51 tail-workspace files
  remain unchanged. Final result and progress records are bound in verification.

The checklists below retain the original task specification, not a request to
rerun the completed diagnostics. No remote or scored evaluation was performed.

## Global Constraints

- Follow `docs/superpowers/specs/2026-11-16-lora-generation-core-design.md`.
- Preserve every pre-existing implementation, data and evidence file.
- Add only the named component/test plus this stage's records.
- No remote action, pretrained loading or training, candidate execution or scoring.
  Existing small-tensor synthetic regression tests remain allowed.
- No installations, telemetry, memory updates, commits, pushes or new worktrees.
- CPU models and inputs must be synthetic. Keep all four admission flags false.
- Use existing unittest directly; do not run test-skill bootstrap/telemetry.

## Task 1: Single-call generation core

**Files:** create `experiments/lora_necessity/generation_core.py` and
`experiments/lora_necessity/test_generation_core.py`.
**Interface:** `generate_candidate(model, tokenizer, messages, *,
expected_tokenizer_sha256) -> dict`, with the exact fields in the design.

- [ ] Write RED tests first. Use a scripted `torch.nn.Module` with a frozen BF16
  parameter and inspect actual forward tensors, gradients and keyword arguments.
  Use the existing `MappingTokenizer` fixture plus a local test-only `decode`.
  At minimum cover EOS, 256-token cap, 4096/4097 input boundary, 3-role messages,
  repeated calls without prefix reuse, tokenizer mismatch before forward,
  tokenizer mutation during decode, malformed/nonfinite/out-of-vocab logits,
  caller train/grad/autocast rejection, Unicode/byte cap, model exception,
  exact candidate text/IDs/hashes, CPU null memory, timing scopes and cold import.
  Example assertion sequence:
  ```python
  result = generate_candidate(model, tokenizer, messages,
                              expected_tokenizer_sha256=identity)
  self.assertEqual(result["output_ids"], [ord("x") + 10, tokenizer.eos_token_id])
  self.assertEqual(result["candidate"], "x")
  self.assertEqual(result["forward_calls"], 2)
  self.assertEqual(result["model_calls"], 1)
  ```
- [ ] Run focused RED with
  `PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/opt/python@3.12/bin/python3.12 -B -m unittest discover -s experiments/lora_necessity -p test_generation_core.py -v`.
  Missing module must produce explicit expected assertion failures, not unrelated
  fixture/import failures.
- [ ] Implement the exact spec; no arbitrary decoding-policy options or callbacks.
  Central algorithm:
  ```python
  with torch.no_grad():
      for position in range(256):
          inputs = torch.tensor([prompt_ids + output_ids],
                                dtype=torch.long, device=device)
          output = model(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                         use_cache=False, return_dict=True)
          logits = output.logits
          # Validate tensor shape/type and finite final-position logits first.
          token = int(logits[0, -1].argmax().item())
          # Validate token against the declared tokenizer vocabulary.
          output_ids.append(token)
          if token == eos_token_id:
              break
  ```
  The two explanatory comments above describe implementation steps, not comments
  to copy into production code. Do not add inline comments.
- [ ] Run focused GREEN. Record exact commands/output and source hashes.
- [ ] Independent task review covers spec compliance and code quality; preserve
  failed review versions and re-review any important fixes.

## Task 2: Integration and closeout

**Files:** create only private CPU compatibility diagnostics and the public
`experiments/lora_necessity/20261116-generation-core-results.md`; append this
stage's `.superpowers/sdd/generation-core-20261116-progress.md`.

- [ ] Snapshot the new source with hashes. Build a random one-layer Qwen3 using
  existing Transformers and fixture tokenizer vocabulary; freeze/eval it.
  Run the generator and confirm finite bounded output, JSON receipt, unchanged
  parameter bytes and no accumulated gradients. No checkpoint reads or training.
- [ ] Run the whole `test_*.py` directory once after focused tests/review fixes.
- [ ] Hash old baseline files, earlier training artifact/evidence and seals;
  verify branch/HEAD/index unchanged and no evaluation data was consumed.
- [ ] Write report separating synthetic generator completion from real training
  already completed, and from still-unperformed remote generation/scoring.
  Record no-cache recomputation as a cost limitation, not engine optimization.
