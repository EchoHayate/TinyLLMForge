# LoRA adapter round trip

Date: 2026-11-16.

This is the persistence and frozen-inference part of the approved gradient-LoRA
pilot, after the completed tensor training core. It does not change training,
data, scoring, or the main engine. Real pretrained execution remains remote.
The fresh remote identity probe failed at jump-host GSSAPI authentication before
reaching the target; that blocks current GPU qualification, not local implementation.

## Choice and scope

Use existing PyTorch serialization with explicit `weights_only=True` loading,
an externally supplied file digest, and strict tensor/result validation.
Adding PEFT or another serialization dependency is unnecessary for this internal
experiment. Saving only tensor lists without metadata would lose the bindings
to the training inputs, configuration and base. This artifact is not a PEFT
adapter and does not claim interoperability or authenticity.

One new module, `adapter_artifact.py`, implements:

- `save_adapter(path, result) -> dict`: save initial and final states together,
  exclusively create a private 0600 file, never overwrite existing files.
  Return file SHA256, file size and save wall time. Partial failed writes remain
  evidence and must not be treated as successful artifacts.
- `load_adapter(path, *, expected_sha256) -> dict`: hash the exact bytes to be
  loaded, require the caller's digest, deserialize on CPU with weights-only,
  validate the envelope and training-result bindings. Return the training result
  and load duration. No unrestricted pickle fallback, imports, or allowlisting.
- `mounted_adapter(model, result, *, state="adapter")`: context manager selecting
  exactly `adapter` or `initial_adapter`; verify the base against the receipt,
  then temporarily install frozen copies of the selected q/v tensors. Yield the
  model in eval/no-grad mode. Restore modules, grad flags, training flags and RNG
  on success and failure. If corrupted state makes an individual restoration
  impossible, continue restoring the remaining state and require discarding the
  model, retaining the body or restoration exception as the explicit cause.
  Detect changed base state on exit, including buffers,
  and explicitly require discarding a mutated/unverifiable base. A body exception
  is preserved unless integrity failure needs to supersede it with a chained error.

## Validation

Require exactly receipt/initial_adapter/adapter result fields, the core's receipt
kind, fixed configuration hash, valid digest strings and false admission flags.
The saved envelope has an exact kind/version/result schema. Accept only finite,
dense, CPU FP32 adapter tensors with no gradients; clone on save and mount.
Receipt target paths must enumerate both q/v targets for consecutive decoder
layers. Initial and final keys, shapes, byte count and digests must agree with
the receipt; A is [8, in_features], B is [out_features, 8]. Initial B must be zero.
The final digest must differ from the initial one; do not require useful output
changes after BF16 rounding. Receipt metadata must be JSON-native and finite.

For mounting, check the existing core's Qwen layout constraints, base BF16
parameters, buffer precision, single CPU/CUDA device, no existing base gradients,
finite state, matching base digest, target names and exact projection dimensions
before changing modules. Reject active autocast; CPU is synthetic tests only.
Reject base inference tensors before mutation: restoring their grad flags
outside inference mode is unsupported. Ordinary tensors may still be mounted
while the caller is in inference mode.
Use the existing LoRAProjection formula, not a second implementation.
Freeze mounted parameters. Do not mutate the caller's result or merge into base.
Disallow nested mounting through the nonadapted-layout check.
Verify mounted adapter bytes too on exit so an illicit update cannot silently
qualify as frozen inference. Caller must use fresh KV state for each real request;
this context manager does not implement generation or reset external caches.
Single-owner model use only; it is not a concurrent serving hot-swap API.

## Acceptance and evidence

Tests use the existing tiny tensor model and synthetic dataset:
real 16-step training → exclusive save → checked reload → mounted forward.
Initial mounted logits must equal base logits; final mounted logits must match
the in-memory adapter before serialization. Unmount must reproduce the base.
Cover file tampering/wrong digest, existing-file refusal, malformed keys/shapes/
dtypes/nonfinite values/config/base identity, exception restoration, caller
RNG/modes/flags, selected state, adapter mutation and base mutation detection.
Default unittest discovery must run the new tensor cases in child processes so
their PyTorch import cannot contaminate the existing cold-tokenizer guard tests.
Preserve individual case counts and propagate child failures/timeouts. Direct
execution of the test file may run its tensor cases in that process.

Record save/load and mount/check wall times separately where measured. No
end-to-end, Qwen, GPU, generated-function or scientific claim follows. Keep all
four admission flags false, preserve sealed data and old evidence, install
nothing, do not retry SSH or rerun the 1416 observations. No new commit or push
is part of this implementation checkpoint.
