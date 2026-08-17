# Qwen3.5 TP1 Real Root-Logit Correctness Gate Design

## Status

Approved for inline execution under the standing instruction to continue the
long-term inference-engine goal without per-step confirmation.

This gate follows the completed constructed Engine/ModelRunner ownership gate.
It deliberately narrows the first real output-correctness boundary to one
tensor-parallel rank and one-shot prompt execution.

## Objective

Load the approved real Qwen3.5-2B checkpoint into an exact
`Qwen35PackedForCausalLM` with `tensor_parallel_size=1`, execute the complete
native path:

```text
token embedding
24 heterogeneous decoder layers
final norm
lm head
```

and compare its full-vocabulary logits against the official Transformers
Qwen3.5 model loaded from the same immutable checkpoint.

The gate must produce independently verifiable evidence for:

- exact prompt tokens and position metadata;
- exact checkpoint and source identities;
- complete native model construction and checkpoint loading;
- full logits shape, dtype, finite status, SHA256, top-k, winner, and margin;
- full-vocabulary numerical differences against the official reference;
- recurrent-state mutation only after successful logits;
- cleanup and GPU-memory observations.

## Why TP1 Comes Before TP4

The existing component factory supports TP sharding, and production linear,
embedding, lm-head, and output-projection layers contain real distributed
collectives. A single TP4 rank is not a complete language model and cannot be
compared independently with the official model.

Starting with TP1 separates:

```text
native Qwen3.5 math and checkpoint mapping
```

from:

```text
distributed process-group construction
TP4 collectives
rank-local full-attention partitioning
all-rank output assembly
Engine and Scheduler integration
```

A TP1 failure therefore has a smaller and auditable root-cause surface. TP4
distributed correctness is a later gate and must not be inferred from TP1.

## Why One-Shot Comes Before Cached Decode

The current `Qwen35PackedForCausalLM.run_step()` owns recurrent linear state,
but the dependency-injected full-attention backend does not yet expose a
production paged-KV or explicit portable cache contract for token-by-token
continuation.

This gate therefore executes each prompt as one causal sequence and compares
the final prompt-token logits. It does not claim:

- cached full-attention correctness;
- token-by-token continuation equivalence;
- chunked-prefill equivalence;
- interleaved requests or slot reuse.

Those remain governed by the immutable schema-v2 canonical `NO_GO` and require
a separate full-attention cache integration gate.

## Frozen Inputs

Remote target:

```text
sitian@10.232.195.203
```

Remote Python:

```text
/data00/home/sitian/sitian-workspace01/tllm/env/bin/python
```

Approved checkpoint:

```text
/data00/home/sitian/sitian-workspace01/tllm/
qwen35-hybrid-state-runs/
qwen35-2b-hybrid-acquire-20260723-222004/model
```

Approved model manifest:

```text
3e650a908234771c3cf1ac4e20c4d38fe69982efedaf4a3e631ad0b14aad7dd0
```

The gate must rehash the complete model inventory and bind the result to the
same immutable revision already recorded by the schema-v2 and checkpoint-load
gates.

## Prompt Corpus

Use three deterministic token sequences without chat-template or generation
side effects:

```text
case p17:
  exact output of the frozen schema-v2 generator with
  length=17, vocab_size=248044, seed=17, forbidden_ids={}
case p65:
  exact output of the frozen schema-v2 generator with
  length=65, vocab_size=248044, seed=65, forbidden_ids={}
case synthetic:
  a fixed explicit list spanning low, middle, and high tokenizer ranges
  while remaining below tokenizer_vocab_size=248044
```

The canonical schema-v2 artifact does not store the original prompt-token
arrays. The authoritative reconstruction path is:

```python
qwen35_hybrid_state_probe._case_token_ids(
    adapter,
    prompt_length,
    seed=prompt_length,
)
```

which delegates to the frozen
`qwen35_hybrid_state_contract.deterministic_token_ids()` because the
authoritative tokenizer vocabulary is greater than 256. The new contract must
recompute these arrays once, assert the exact values below, then hard-code the
arrays and their canonical SHA256 values. Runtime correctness execution and
verification must not import the old probe or regenerate tokens.

Canonical token SHA serialization is UTF-8 JSON over a list with
`ensure_ascii=True` and compact separators `(",", ":")`.

```text
p17:
  tokens:
    [237734,105227,220508,88001,203282,70775,186056,53549,168830,
     36323,151604,19097,134378,1871,117152,232433,99926]
  token_sha256:
    be8a139b93467e0b0ed92999e8feec6de8fbaac4a2c4faf4786f798bb00cceb9

p65:
  tokens:
    [72098,187379,54872,170153,37646,152927,20420,135701,3194,
     118475,233756,101249,216530,84023,199304,66797,182078,49571,
     164852,32345,147626,15119,130400,245681,113174,228455,95948,
     211229,78722,194003,61496,176777,44270,159551,27044,142325,
     9818,125099,240380,107873,223154,90647,205928,73421,188702,
     56195,171476,38969,154250,21743,137024,4517,119798,235079,
     102572,217853,85346,200627,68120,183401,50894,166175,33668,
     148949,16442]
  token_sha256:
    2391c5bbc31e842e8c362e591458d05541b1566409f03672d192fe6a9702a264

synthetic:
  tokens:
    [128,129,255,256,1024,32768,65536,124022,186033,247787,248043]
  token_sha256:
    a36985347858070c7c917b110c793414192e691ffe160be66276b6022c940819
```

The artifact stores the exact token arrays and SHA256. It must not depend on a
mutable natural-language tokenizer rendering at verification time.

The p65 result is diagnostic. Reproducing the official BF16 tie is acceptable;
changing its winner or margin relative to the official model is not.

## Reference Model

Load the official model with:

```python
AutoModelForCausalLM.from_pretrained(
    checkpoint_dir,
    local_files_only=True,
    trust_remote_code=False,
    dtype=torch.bfloat16,
    attn_implementation="eager",
)
```

Requirements:

- exact local checkpoint only;
- no download or network access;
- evaluation mode and `torch.no_grad()`;
- one isolated fresh process;
- `use_cache=False`;
- output logits converted to contiguous CPU FP32 only after the GPU forward;
- the reference process exits before the native process starts.

The reference and native models must never coexist on one GPU. This bounds
memory and prevents one model from affecting the other's allocator state.

## Native Model

Build a TP1 real candidate using the existing authorized checkpoint stack:

```text
read bounded checkpoint metadata
build complete tensor plan
build TP1 hybrid-state layout and capacity-one pool
prepare exact Qwen35PackedForCausalLM target
stream and transactionally assign every checkpoint binding
validate complete candidate payload
```

The native process uses:

```text
tensor_parallel_size = 1
tensor_parallel_rank = 0
parameter_device = cuda:0
compute dtype = bfloat16
stable recurrent dtype = float32 where required by the frozen layout
```

No `LLMEngine`, `ModelRunner`, Scheduler, sampler, tokenizer I/O, generation,
or inference loop is constructed.

## TP1 Causal Full-Attention Backend

Add a gate-owned backend implementing the `Qwen35FullAttentionShell` contract:

```python
class Qwen35TP1CausalAttentionBackend(nn.Module):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        ...
```

For one sequence:

- reshape query to `[tokens, query_heads, head_dim]`;
- reshape key/value to `[tokens, kv_heads, head_dim]`;
- repeat KV heads only when `query_heads > kv_heads`;
- compute scaled dot-product attention in FP32;
- apply a strict lower-triangular causal mask;
- apply softmax in FP32;
- multiply values in FP32;
- cast the flattened result back to the input dtype.

This backend is correctness-only. It must not use FlashAttention, paged KV,
CUDA graphs, sparse attention, cache compaction, or optimized kernels.

## Native Execution

For each case:

1. allocate and activate one exact `HybridStateLease`;
2. snapshot all pool tensors and version counters;
3. construct exact 1D token and position tensors on GPU;
4. call `Qwen35PackedForCausalLM.run_step()` once;
5. require normalized hidden states and logits to be finite;
6. select the final prompt-token logits;
7. move that row to contiguous CPU FP32;
8. snapshot recurrent state after commit;
9. release the lease and prove the slot is zeroed.

The same native process may execute all three cases sequentially only if every
case receives a new lease generation and release is proven between cases.
Otherwise each case must use a fresh process.

## Comparison Contract

For each full-vocabulary final-token logit row, record:

```text
shape
source dtype
CPU comparison dtype
full_logit_sha256
top-20 token IDs and logits
winner and runner-up token IDs
winner and runner-up logits
winner margin
max absolute difference
mean absolute difference
p50/p95/p99/p99.9 absolute difference
cosine similarity
allclose violation count
maximum scaled allclose error
```

Use the existing schema-v2 comparison semantics:

```text
comparison policy:
  bf16_decision_preserving
full-vocabulary allclose:
  diagnostic, using the frozen dtype-derived tolerance
hard token guard:
  native winner token == official winner token
hard top-k guard:
  official winner appears in native top-20 and native winner appears in
  official top-20
hard margin guard:
  positive official margins remain positive and preserve the winner;
  an official zero-margin tie must remain the same tied top pair
```

Do not weaken the immutable schema-v2 thresholds. The independent verifier
reconstructs every derived metric from raw FP32 tensor artifacts.

## State Transaction Evidence

For every case the gate must prove:

- the pool is zero before activation;
- `layer_stack.prepare()` does not mutate persistent state;
- successful `run_step()` commits exactly once;
- all 18 linear-layer convolution and recurrent components change from zero
  after a non-empty prompt;
- the six full-attention layers do not create hidden state in this pool;
- release zeroes every component;
- a deliberately injected lm-head failure leaves every pool tensor and
  version unchanged.

The failure injection runs with a tiny synthetic fixture locally. The remote
real-checkpoint run remains unmodified.

## Process and Memory Safety

Use one fresh reference process and one fresh native process per authoritative
run. Each process records:

```text
PID
start and finish timestamps
CUDA device UUID
torch and transformers versions
VmRSS and VmHWM
torch.cuda.max_memory_allocated
torch.cuda.max_memory_reserved
exit code
```

Require before each process:

```text
GPU free memory >= 24 GiB
no child process from the gate remains alive
```

The gate must not kill unrelated GPU processes. If the free-memory floor is
not met, the run is `INCOMPLETE_RESOURCE`, not a correctness failure.

## Artifacts

Publish exactly:

```text
tp1_real_root_logit_correctness.json
reference_logits.pt
native_logits.pt
source_manifest.json
```

The JSON contains no full logits. Tensor artifacts contain a mapping from case
ID to contiguous CPU FP32 rows and are atomically published only after all
processes, cleanup, and hashes succeed.

The source manifest binds:

- exact source closure and hashes;
- exact checkpoint inventory and hashes;
- prompt corpus and token SHA256 values;
- runtime and GPU identity;
- result and tensor artifact SHA256 values.

## Independent Verifier

Create a standard-library-plus-PyTorch verifier that:

- imports neither TinyLLMForge nor the producer;
- requires the exact four-file inventory;
- rehashes source, checkpoint, JSON, and tensor artifacts;
- loads both tensor maps on CPU;
- checks exact case IDs, shapes, FP32 dtype, finite values, and vocabulary;
- recomputes SHA256, top-k, winners, margins, percentiles, cosine similarity,
  allclose violations, and scaled errors;
- validates process separation, cleanup, memory floor, state evidence, and
  forbidden counters;
- rejects extra files, source re-signing, tensor replacement, metric
  re-signing, prompt drift, missing cases, or relaxed thresholds.

It prints:

```text
PASS, <N> checks
```

only if every hard guard passes.

## Failure Classification

Use exact classifications:

```text
PASS
NO_GO_LOGIT
NO_GO_STATE
INCOMPLETE_RESOURCE
INCOMPLETE_REFERENCE
INCOMPLETE_NATIVE
INCOMPLETE_ARTIFACT
```

`NO_GO_LOGIT` means both processes completed but a hard output guard failed.
`NO_GO_STATE` means native transaction or release evidence failed.
Incomplete outcomes must not publish an authoritative PASS artifact.

## Forbidden Conclusions

Passing this gate does not prove:

- TP4 distributed correctness;
- `ModelRunner.run()` or `LLMEngine.step()` correctness;
- Scheduler, sampling, generation, or tokenization correctness;
- cached decode, chunked prefill, interleaving, or slot reuse;
- latency, throughput, cache savings, GPU-memory savings, compression, or
  production quality.

## Next Gates

If TP1 passes:

1. TP4 distributed one-shot logits against the same official oracle;
2. full-attention cache contract and cached continuation correctness;
3. production `ModelRunner.run()` correctness;
4. bounded `LLMEngine.step()` integration;
5. only then latency, throughput, cache, and GPU-memory benchmarks.

If TP1 fails, preserve the evidence and localize the first divergent layer or
component before any TP4 or performance work.

## Authoritative Result

The final source-bound remote run completed on 2026-07-28:

```text
run tag:
  qwen35-tp1-authority-20260728-195153-r2
remote root:
  /data00/home/sitian/sitian-workspace01/tllm/
  qwen35-tp1-root-logit-tests/
  qwen35-tp1-authority-20260728-195153-r2
local artifact:
  experiments/qwen35_hybrid_state/
  qwen35-tp1-authority-20260728-195153-r2/
classification:
  PASS
```

The exact source and artifact identities are:

```text
source files:
  77
source tree SHA256:
  e5da50970951b32a61ecc9f85cf1cd447dc95950856345d780cd9e089bdf11ab
result SHA256:
  39c4bbc548a82e915609dccb57101a6e64cbc92319d5ba69c44d427f8f4aa519
reference logits SHA256:
  3373ab6038d6f21cf6421aa0e1c9146cc001e6e8bcbd06e6ab59c65ecc709e5a
native logits SHA256:
  5d3fcb3a204a1b1bd673026d83f06ddc76422ca49d51d6af32671ba153ecb4d4
independent verifier:
  PASS, 179 checks
```

Reference PID `3832269` exited before native PID `3836576` started. Both
workers used GPU UUID
`GPU-57be086f-e967-c022-3832-93df4fc77bd0`, independently rechecked the
24-GiB free-memory floor, used distinct dynamically allocated
`TINYVLLM_DIST_PORT` and `MASTER_PORT` values, and did not initialize a
process group.

The three frozen cases preserved the official final-token winner:

| case | official winner | native winner | official margin | native margin | max abs diff | cosine | allclose violations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `p17` | 198 | 198 | 0.4375 | 0.5 | 0.130859375 | 0.9999015331 | 214474 |
| `p65` | 62 | 62 | 0.3125 | 0.25 | 0.12890625 | 0.9998495579 | 223900 |
| `synthetic` | 10992 | 10992 | 1.75 | 1.6875 | 0.1640625 | 0.9998912811 | 212647 |

This is a `bf16_decision_preserving` PASS, not an elementwise-logit allclose
PASS. The large diagnostic allclose-violation counts and exact raw FP32 rows
remain in the artifact. No tolerance was relaxed after observing the result.

For every case, all 18 linear layers changed both convolution and recurrent
state components, `prepare()` remained read-only, commit occurred once, and
release zeroed all 36 components and removed the pool binding. The independent
verifier also confirmed zero Engine, ModelRunner, Scheduler, sampler, and
generation counters.

Two runtime blockers found by the real native smoke were fixed through
test-first regression coverage:

1. `Qwen35PartialInterleavedRotaryEmbedding._apply` shadowed
   `nn.Module._apply`, preventing full-model device migration. The rotary
   helper is now `_apply_rotary`, and `.to("meta")` migration is covered.
2. `Qwen35PackedForCausalLM` rejected the production `ParallelLMHead`
   prefill contract, which intentionally returns only selected logits rows.
   The root now accepts a positive logits-row count no larger than the hidden
   token count while still rejecting zero or excess rows.

## Completion Audit

The prompt-to-artifact mapping is complete:

| requirement | implementation/test evidence | authoritative artifact/verifier evidence |
| --- | --- | --- |
| Frozen prompts and BF16 decision policy | contract module plus 7 tests | exact prompt arrays/SHA and recomputed metrics |
| Correct TP1 causal attention | manual FP32 oracle, GQA, poisoning, BF16, malformed-input tests | native full-vocabulary rows |
| Authorized real checkpoint and TP1 root | candidate factory/loader/binding/assignment tests | checkpoint hashes and model manifest |
| Separate fresh workers | coordinator ordering and PID tests | distinct PIDs and timestamps |
| Resource-safe GPU | GPU query/selection/recheck tests | UUID and free-memory rows |
| Source-bound execution | 77-file source manifest and hash validation | source tree SHA and artifact hashes |
| Transactional recurrent state | root failure and ownership tests | three 36-component state rows |
| Exact four-file publication | finalizer refusal/tamper tests | exact local and remote inventory |
| Independent recomputation | four verifier/tamper tests | 179-check read-only verification |
| Forbidden runtime objects | AST/source and worker tests | five zero counters |
| Existing ownership prerequisite | frozen authoritative snapshot | independent verifier PASS, 281 checks |
| Immutable schema-v2 result | existing contract/probe/verifier tests | canonical `NO_GO` remains unchanged |

The focused current-source matrix passed the TP1 contract, coordinator,
verifier, rotary, root, packed stack, state transaction, component factory,
checkpoint binding/assignment/loading, attention shells, hybrid-state
contract/probe/verifier, exact real-worker rejection, and constructed
ownership preflight tests. The historical constructed-ownership verifier must
use its immutable source snapshot because the later mixed-state dtype fix
intentionally changed its source tree; that frozen snapshot independently
passed 281 checks.

## Exact Conclusion

Proven:

```text
TP1 real-checkpoint one-shot final-token decision preservation:
  yes, for p17, p65, and synthetic
full-vocabulary raw logits and numerical diagnostics:
  preserved and independently recomputed
native recurrent-state transaction and release:
  proven for the three cases
```

Not proven:

```text
elementwise BF16 allclose:
  no
TP4 distributed logits:
  not tested
ModelRunner / LLMEngine / Scheduler correctness:
  not tested
cached decode / chunked prefill / interleaving:
  not tested
latency / throughput / cache / GPU-memory / compression / quality gains:
  not measured
```

The immutable schema-v2 canonical `NO_GO` remains unchanged. The next
correctness boundary is TP4 distributed one-shot final-token logits. Only
after TP4, cached-continuation, and Engine integration correctness pass may
the performance and cache benchmarks support improvement claims.
