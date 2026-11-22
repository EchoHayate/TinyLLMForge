# LoRA training core

Date: 2026-11-16.

This implements the training-worker portion of the already approved
`2026-11-10-lora-necessity-pilot-design.md`. It does not change the frozen
configuration, data, scoring rules or main inference engine. User continuation
after the validation checkpoint authorizes implementation, not a real model run.

## Scope and choice

Implement an explicit PyTorch q/v low-rank wrapper and an in-process training
core. Use existing dependencies; PEFT is not installed in the inspected local
environment. A PEFT integration would add an unnecessary dependency here, while
a generic injected fake optimizer would not test actual gradient updates.
Local verification uses small synthetic tensors, never a pretrained model.
Real Qwen execution remains remote and separately admitted.

The public entrypoint is `train_adapter(model, dataset)`. The caller owns model
loading, device admission, checkpoint identity and artifact persistence. The
core only accepts a dataset produced by `prepare_training(tokenizer,
supervision_raw)`. Its strict structural validation and digest are integrity
checks, not cryptographic attestation of tokenizer or data provenance.

## Training inputs

`training_inputs.py` imports no torch and uses `training_jobs` and `measure`.
Only the frozen eight supervision examples are accepted. It does not read
prompts, hidden oracles, reference inventories or evaluation results.

Each row contains `task_id`, `input_ids`, `attention_mask`, `labels`,
`prompt_tokens`, `target_tokens`, `messages_sha256`, `answer_sha256`.
Sequence = measured prompt IDs followed by measured answer IDs through its
unique EOS. Template whitespace after EOS is excluded. Labels are -100 for
prompt tokens and exactly answer IDs including EOS elsewhere; no pre-shifting.
Batch size one needs no padding. The prompt must be at most 4096 tokens and the
target at most 256 tokens; the combined training sequence may reach 4352.
No truncation or additional combined 4096-token cap is permitted.
Rows must be S01 through S08 in fixed order.
All eight measurements must use the same tokenizer-state digest.

Dataset fields are `kind="LORA_TRAINING_INPUTS_UNATTESTED"`, `version=1`,
`supervision_sha256`, `tokenizer_state_sha256`, `eos_token_id`, `pad_token_id`,
`rows`, `dataset_sha256`. The digest covers all fields except itself.
`validate_training(dataset)` checks exact keys/types, lengths, IDs, token
boundaries, masks, labels, unique terminal EOS, source pin and digest.
It returns the digest. Inputs are JSON-native and independently copied.

## Parameter updates

`training_worker.py` requires existing PyTorch. Model layout must expose
`model.layers.<index>.self_attn.{q_proj,v_proj}` for every declared decoder
layer. Targets must be ordinary nonshared `torch.nn.Linear` modules, with no
preexisting adapters, parametrizations or extra matching q/v projections.
The core requires base parameters to be BF16; floating buffers may be BF16 or
FP32 (including Qwen's nonpersistent rotary frequencies), and are hashed too.
All tensors must be on one CPU or CUDA device with finite base state, no
preexisting gradients, and no active inference-mode/no-grad/autocast context.
CPU is for synthetic tests only.

Wrap each target as `base(x) + (alpha/rank) * B(A(x.float()))` converted to the
base output dtype. Rank=8, alpha=16; A is seeded Kaiming-uniform FP32 and B
starts at zero. Preserve caller RNG state. Freeze all base parameters, disable
dropout using eval mode, keep autograd enabled, and allow only newly created
LoRA tensors into AdamW. q/v biases stay frozen.

Use fixed order S01..S08 twice. Exactly 16 optimizer steps; no accumulation,
checkpoint selection, retries or learning-rate search. AdamW lr=5e-5,
weight_decay=0, gradient clip=1. Forward has use_cache=False. Cross entropy
uses the preceding logit for each answer label, including EOS; prompt logits
do not contribute. Reject nonfinite loss, gradients, norm or adapter state.

Hash every base state tensor before and after; reject changed tensor names,
dtypes, shapes or bytes. Verify at least one adapter tensor changed. Do not
require every A tensor to have nonzero first-step gradients: zero B initially
blocks that path. Report losses as training-path evidence, not task success.
Return CPU clones of initial/final adapter tensors separately from a JSON
receipt. No files, model downloads, network calls or subprocesses.

Restore original projection modules, train/eval flags, requires_grad flags and
RNG state on success or failure; discard temporary adapters on failure. If base
state is mutated by an unexpected model forward, detect and fail but do not
claim to undo it: the caller must discard that model. No background cleanup or
aggregate resource-budget requirement is introduced.

## Receipt and evidence boundary

Return kind `LORA_TRAINING_CORE_RESULT_UNATTESTED`, exact input/config hashes,
target paths, initial/final adapter digests, base digest, per-step task/loss/
gradient-norm/elapsed time, preparation/training/verification/core wall times,
supervised token count and adapter/optimizer tensor bytes. CUDA timings use
device synchronization. Model load/save, launch, peak VRAM and end-to-end costs
are explicitly unknown, not zero. All scientific/admission flags remain false.

Tests cover real tiny-tensor forward/backward, zero initial delta, optimizer
allowlisting through frozen-state checks, answer-only causal alignment,
sixteen updates, deterministic initialization, no cache, bad layout/dtypes,
nonfinite failure and caller-state restoration. No pretrained model is loaded;
passing these tests does not qualify Qwen, GPU execution, generated code,
adapter serialization, model comparison or net benefit.
