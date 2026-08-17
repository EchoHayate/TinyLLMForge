# Qwen3.5 Mixed Stable-Parameter Dtype Design

## Status

Approved under the standing inline-execution direction. This is a CPU-only
correctness gate for the existing Qwen3.5 reference shell.

## Goal

Make the existing Qwen3.5 linear-attention shell accept the verified
checkpoint dtype layout without changing its mathematical result, output
dtype, state dtype, state ownership, or runtime wiring:

```text
hidden/projections/states: BF16 compute dtype
conv_weight:               BF16 compute dtype
dt_bias:                   BF16 compute dtype
A_log:                     F32 stable dtype
norm_weight:               F32 stable dtype
```

The shell must preserve the F32 stable parameters. It must not downcast them
during construction or forward.

## Source Evidence

The verified fixed Qwen3.5-2B checkpoint stores every linear-attention layer
as:

```text
linear_attn.conv1d.weight BF16 [6144, 1, 4]
linear_attn.A_log         F32  [16]
linear_attn.dt_bias       BF16 [16]
linear_attn.norm.weight   F32  [128]
```

The completed tensor-metadata contract validates 18 `A_log` and 18 linear
normalization tensors as F32. The current shell incorrectly requires all four
parameter tensors to share one dtype, so the verified checkpoint cannot be
bound without an intentional mixed-dtype contract.

## Existing Math

`qwen35_gated_delta_recurrent()` already performs the recurrence in FP32:

```text
query/key/value/a/b/A_log/dt_bias/state -> float()
recurrent accumulation                  -> FP32
output                                  -> query dtype
candidate state                         -> input-state dtype
```

`qwen35_gated_rmsnorm()` already performs normalization, scale, and gate math
in FP32:

```text
core.float()
weight.float()
silu(gate.float())
output -> core dtype
```

Therefore no arithmetic rewrite or new cast is required. The blocker is only
an over-restrictive input validation rule.

## Alternatives Considered

### 1. Downcast all parameters to the compute dtype

Rejected. This discards the checkpoint's intentional F32 `A_log` and
normalization values and would make later checkpoint equivalence harder to
establish.

### 2. Promote all compute tensors and states to F32

Rejected. This changes the shell's storage/output contract, increases memory,
and does not match the verified checkpoint compute layout.

### 3. Separate compute and stable-parameter dtype contracts

Selected. `conv_weight` and `dt_bias` follow the hidden compute dtype.
`A_log` and `norm_weight` remain floating tensors on the same device and may
use F32. Existing FP32 accumulation consumes them without a lossy cast.

## Primitive Contract

`qwen35_gated_rmsnorm()` keeps this signature:

```python
def qwen35_gated_rmsnorm(
    core: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor
```

Validation becomes:

- `core`, `gate`, and `weight` must be floating tensors;
- `core` and `gate` must have exactly matching shape, dtype, and device;
- `weight` must be rank one and match the final dimension;
- `weight` must be on the same device as `core`;
- `weight.dtype` may differ from `core.dtype`;
- `eps` must be positive and finite.

Math remains FP32 and the result remains `core.dtype`. Inputs remain
unmodified.

## Linear-Attention Constructor Contract

All four parameter tensors must:

- be floating;
- have the existing exact shapes;
- be on one common device.

The dtype groups are:

```text
compute group: conv_weight, dt_bias
stable group:  A_log, norm_weight
```

`conv_weight.dtype` and `dt_bias.dtype` must match. `A_log` and
`norm_weight` may use any floating dtype, including F32, and are not required
to match each other or the compute group. This keeps the constructor general
while preserving the verified checkpoint tensors exactly.

## Forward Contract

Before projection work:

- hidden states and both supplied states retain the existing shape,
  floating, dtype, and device checks;
- `conv_weight` must match hidden-state dtype and device;
- `dt_bias` must match hidden-state dtype and device;
- `A_log` and `norm_weight` must remain floating and match the hidden-state
  device, but may differ in dtype.

This forward-time validation protects against post-construction buffer
replacement.

The output and candidates remain:

```text
output dtype/device:                      hidden_states dtype/device
candidate convolution state dtype/device: input convolution state
candidate recurrent state dtype/device:   input recurrent state
```

No persistent input state is mutated.

## Test Gate

### Primitive

Add a BF16 `core`/`gate` plus F32 `weight` case that:

- matches an independent FP32 formula;
- returns BF16;
- preserves all input tensors;
- still rejects a core/gate dtype mismatch;
- still rejects a weight device mismatch.

### Shell

Construct a checkpoint-like shell with:

```text
projection/conv/dt/output tensors: BF16
A_log/norm_weight:                 F32
hidden/states:                     BF16
```

Prove:

- construction succeeds;
- full output and candidate states match the independent FP32 oracle within
  BF16 tolerance;
- output and candidate states remain BF16;
- `A_log` and `norm_weight` remain F32 before and after forward;
- inputs and stable parameters remain unmodified;
- split continuation matches one-shot execution;
- constructor rejects `conv_weight`/`dt_bias` dtype mismatch;
- forward rejects compute parameters that do not match hidden dtype;
- forward rejects any parameter on the wrong device.

## Non-Goals

This gate does not add:

- safetensors payload materialization or assignment;
- convolution singleton-channel squeezing;
- TP-local target shape validation;
- embedding/LM-head aliasing;
- concrete CUDA/Triton/FLA kernels;
- production `ModelRunner` Qwen3.5 selection;
- Engine startup binding or Scheduler admission;
- checkpoint token/logit equivalence;
- any performance, memory, cache, compression, or quality claim.

The supplied state pool remains the only state owner. Engine and Scheduler
wiring remain fail-closed. The immutable Qwen3.5 schema-v2 canonical result
remains `NO_GO`.

## Success Criteria

The gate passes only when:

1. the new tests first fail on the current same-dtype restrictions;
2. the minimal validation changes make both primitive and shell suites pass;
3. existing FP32/BF16 same-dtype behavior remains green;
4. focused root/owner/full-shell regressions remain green;
5. static checks show no production wiring, staged files, or schema changes.

