# Qwen3.5 Constructed Engine/ModelRunner Ownership Gate Design

## Objective

Cross the next runtime boundary after the completed live-concurrent TP4
candidate gate by constructing one exact production `LLMEngine` and four
exact production `ModelRunner` instances, executing both real constructor
bodies, transferring one approved real TP4 candidate into each rank-local
runner, and invoking the production all-rank zero-payload binding method:

```text
real LLMEngine.__init__
  -> real ModelRunner.__init__ for ranks 0, 1, 2, 3
  -> four exact production class identities
  -> four approved real checkpoint-derived rank-local candidates
  -> rank-local publication and ownership transfer
  -> LLMEngine.bind_qwen35_loaded_checkpoint_candidates(timeout_s=...)
  -> production zero-payload all-rank aggregation
  -> four homogeneous bound rows
  -> reverse-rank cleanup and whole-scope collection
```

The gate executes on CPU and must not execute a production scheduler,
`LLMEngine.step()`, CUDA operation, model or attention forward, sampling,
tokenization, inference, or a real worker process loop.

Successful evidence may use:

```text
provenance:
  real-checkpoint-derived-constructed-engine-model-runner-ownership
claim boundary:
  no-scheduler-step-forward-or-inference
```

## Prerequisites

Freeze the authoritative live-concurrent ownership gate:

```text
run:
  qwen35-tp4-live-concurrent-ownership-20260728-163700
source tree:
  0a4ae63468b7f0bdccc0c41d4803e36d418e9966b5d66525ea7690f8203bfeb3
result:
  f2d38ca089a53a413236fbf18c057fb10df04b84a338248b2004d77f5060c280
manifest:
  d9d0166214d4e78d756f6c2a20306a0e537f5fc5f138adde155cd6c9f6b1b236
```

Freeze the pristine serial payload oracle:

```text
run:
  qwen35-tp4-real-candidate-replay-20260728-145713
oracle:
  d750d664219378c234a2127b708ec191feb9b2c9f1f2902c47d0ad5dc152d3ef
approved model manifest:
  3e650a908234771c3cf1ac4e20c4d38fe69982efedaf4a3e631ad0b14aad7dd0
checkpoint:
  /data00/home/sitian/sitian-workspace01/tllm/qwen35-hybrid-state-runs/
  qwen35-2b-hybrid-acquire-20260723-222004/model
```

Each constructed runner must ultimately expose the exact rank payload from
the pristine serial oracle: 320 binding hashes, 26 phase hashes, 24 alias
groups, aggregate destination hash, loader statistics, model fingerprint,
layout fingerprint, and `bfloat16` dtype.

## Approaches Considered

### `object.__new__` Followed by Field Injection

Rejected. It can create an object with production class identity, but it does
not execute either constructor and therefore cannot prove constructor control
flow, field initialization, dependency call ordering, or constructor drift.

### Extract and Recompile Constructor AST

Rejected as the authoritative path. It would execute copied constructor
syntax in a private namespace, not the methods attached to the imported
production classes. It would also allow the harness to omit module-global
dependencies without proving that the production lookup sites were reached.

AST inspection remains useful only as a static source audit.

### Real Classes with an Auditable Inert Dependency Capsule

Selected. Import the exact production classes, temporarily replace only the
module-global side-effect dependencies and three explicitly forbidden
`ModelRunner` methods, call `LLMEngine(model, ...)`, and restore every replaced
global before candidate loading or binding.

The selected design preserves:

- exact `type(engine) is LLMEngine`;
- exact `type(runner) is ModelRunner` for ranks `0..3`;
- execution of the attached production `LLMEngine.__init__`;
- execution of the attached production `ModelRunner.__init__` four times;
- production constructor branch ordering;
- production publication, rank-local candidate binding, acknowledged command
  dispatch, and Engine aggregate validation methods.

It replaces only unsafe external effects with inert implementations that
record exact arguments, ordering, return identities, and call counts.

## Frozen Production Sources

The preflight freezes these current files and method sources:

```text
tinyvllm/engine/model_runner.py:
  0cba7e97b2d3425186722f53a0c14e7b01e2c90e190e6cdf2c9472517bcda849
ModelRunner.__init__:
  8aa2747cff30e8398737cb024d375f9f04763efdd53cb23084c32c3d872f4edc
ModelRunner.dispatch_command:
  9a63e40ef7d16b6300e70d41c2f05575a50adbf6dc04942677034d6bee363342
ModelRunner.publish_qwen35_loaded_checkpoint_candidate:
  37f95954c287d5dd0e8883f299d7049e66dcb3c79806624eb6da3ca7d51a6d4f
ModelRunner.bind_qwen35_loaded_checkpoint_candidate:
  a14e6856ad74eb935116075ee6fe81516c8e212f89914e0fcd55bb39e86d63e0
ModelRunner.bind_published_qwen35_loaded_checkpoint_candidate:
  aa178f886d314893593039c5e890239fb954740f059b2d12fc697bd25790fbcd

tinyvllm/engine/llm_engine.py:
  6cf68dc76641bf772c01d31fd60ee42cbab82e3c62a0ee8aa154dbe802c727ae
LLMEngine.__init__:
  f770308d40248be4515838a720b288fd69f718d25746398bc145b4b43478fd9c
LLMEngine.call_model_runner_acknowledged:
  6eed126b80c9c823ceff37cc51273735d656c2b1be963bbea2bbd4ad9da9f14d
LLMEngine.bind_qwen35_loaded_checkpoint_candidates:
  82c0528d6b06ae8d67812d1a8802e8163aadb4886afc3894bf28a0cf35c3c84c

tinyvllm/config.py:
  9b860eafe88c1734e5135ab0f65188f025762f5d0d0a702eb4994157aabec076
```

Any source or structural drift must reject the gate before a candidate is
loaded.

## Constructor Dependency Capsule

The capsule is a context manager over exact attributes in the imported
`model_runner` and `llm_engine` modules. It must:

1. save every original object identity;
2. install one replacement per approved dependency;
3. record a monotonic call ledger with dependency name, normalized arguments,
   result identity, and caller rank when applicable;
4. reject an unapproved dependency lookup or call;
5. restore every original identity in `finally`;
6. prove restoration before candidate loading starts.

The capsule must not patch `LLMEngine`, `ModelRunner`,
`LLMEngine.__init__`, `ModelRunner.__init__`, publication methods, binding
methods, `call_model_runner_acknowledged`, or
`bind_qwen35_loaded_checkpoint_candidates`.

### ModelRunner Replacements

For exactly four constructor executions, the approved call counts are:

```text
dist.init_process_group:       4
torch.cuda.set_device:         4
torch.get_default_dtype:       4
torch.set_default_dtype:       8
torch.set_default_device:      8
set_quant_config:              4
Qwen3ForCausalLM:              4
load_model:                    4
Sampler:                       4
ModelRunner.warmup_model:      4
ModelRunner.allocate_kv_cache: 4
SharedMemory:                  4
dist.barrier:                  4
ModelRunner.loop:              3
```

`apply_cpu_offload` and `ModelRunner.capture_cudagraph` must each have zero
calls. The inert config sets:

```text
tensor_parallel_size = 4
enforce_eager = true
cpu_offload = false
kv_quant_bits = 0
am_compact_blocks = 0
kv_offload_mvp0 = false
multi_sequence_cuda_graphs = false
```

The inert `Qwen3ForCausalLM` constructor returns a unique rank-local
placeholder. `load_model`, warmup, KV allocation, and worker loop return
without touching tensors or CUDA. The production
`Qwen35HybridModelOwnerPublicationSlot` and
`ExactCudaGraphCache` constructors are not replaced.

The inert `SharedMemory` implementation provides one private in-process
1 MiB buffer named by the production constructor. It has no operating-system
shared-memory resource and cannot collide with another run.

### LLMEngine Replacements

For one Engine constructor, the approved call counts are:

```text
Config:                         1
mp.get_context("spawn"):        1
context.Pipe(duplex=False):     3
context.Event:                  3
context.Process:                3
process.start:                  3
parent sender close:            3
ModelRunner direct rank0 call:  1
ModelRunnerCommandAckCollector: 1
AutoTokenizer.from_pretrained:  1
Scheduler:                      1
atexit.register:                1
```

`Config` returns the audited TP4 inert config. The process objects require the
exact production `ModelRunner` class as their target and record its arguments
without creating operating-system processes. Their deferred rank `1..3`
exact-class constructor calls execute inside the acknowledgement-collector
replacement, after rank0 construction and before it returns. Rank0 is
constructed through the unmodified `llm_engine.ModelRunner` binding. No
callable proxy or class replacement is allowed.

The tokenizer replacement returns only an inert object with a fixed
`eos_token_id`; it performs no file or network access and no tokenization.
The Scheduler replacement returns an inert sentinel and does not execute the
production Scheduler constructor. The atexit replacement records
`engine.exit` but does not register it.

The exact call ledger must prove no process, tokenizer, scheduler, or exit
side effect escaped the capsule.

## In-Process Acknowledgement Transport

The three inert process records retain the exact worker runner instances.
The inert acknowledgement collector must implement the production collector
surface used by `LLMEngine.call_model_runner_acknowledged`:

```python
collect(
    command_id,
    *,
    expected_ranks,
    timeout_s,
    is_rank_alive,
) -> tuple[ModelRunnerCommandAck, ...]
```

Rank0 must execute the production `dispatch_command(...)`, which writes one
exact `ModelRunnerCommandEnvelope` into the inert shared buffer with:

```text
method_name:
  bind_published_qwen35_loaded_checkpoint_candidate
args:
  ()
requires_ack:
  true
```

The collector then requires ranks `(1, 2, 3)`, reads the exact envelope
identity, invokes the production worker method on each constructed runner,
and returns exact production acknowledgement objects. It must record:

- one collect call;
- one zero-payload envelope;
- one worker invocation per rank;
- one acknowledgement per rank;
- no timeout, poison, retry, or worker-death path.

This capsule is not a multiprocessing performance claim. It proves that the
constructed Engine's production all-rank method aggregates four constructed
rank-local participants without serializing a candidate through the command
payload.

## Candidate Ownership Transfer

After the capsule restores every production global, build four real
checkpoint-derived TP4 candidates sequentially in rank order `(0, 1, 2, 3)`.
Earlier candidates remain live while later candidates load.

For each rank:

1. identify the exact constructed runner;
2. require `type(runner) is ModelRunner`, exact `rank`, and exact
   `world_size == 4`;
3. replace only the inert constructor model placeholder with
   `candidate.owner.model`;
4. invoke the production
   `runner.publish_qwen35_loaded_checkpoint_candidate(candidate)` once;
5. retain candidate, owner, runtime bridge, model, pool, target, selected
   tensors, request, slot, and runner;
6. rehash the complete payload against the pristine rank oracle;
7. require the publication slot to retain the exact candidate and owner graph;
8. do not directly call the candidate-binding method.

The Engine then invokes exactly once:

```python
engine.bind_qwen35_loaded_checkpoint_candidates(
    timeout_s=0.25,
)
```

The result must be an ordered four-row tuple. Every row must be `bound`, use
the approved model fingerprint, share one layout fingerprint and dtype, and
match its rank. Engine completion state must contain the exact rows and:

```text
(
  approved_model_fingerprint,
  shared_layout_fingerprint,
  "bfloat16",
  0.25,
)
```

An exact repeat must return the same stored tuple with zero new dispatch,
collect, or rank-local bind calls.

## Construction and Binding Evidence

The authoritative artifact must contain:

- exact Engine and ModelRunner module, qualname, and file identity;
- exact `id(type(instance)) == id(production_class)` evidence;
- constructor entry and return count for Engine and every rank;
- complete capsule ledger and exact call-count summary;
- original and restored dependency object identities;
- unique inert placeholder identity per rank;
- candidate publication and owner transfer identity per rank;
- complete pristine payload evidence per rank;
- rank-local publication, candidate-bind, and owner-bind call counts;
- zero-payload command envelope and acknowledgement rows;
- first all-rank result and exact-repeat result;
- Engine stored rows and canonical completion configuration;
- explicit counters for scheduler step, Engine step, model forward, attention
  forward, sampler, tokenization, CUDA operations, and inference;
- cleanup and memory observations.

All forbidden execution counters must equal zero.

## Failure and Atomicity Matrix

Directed local tests must cover:

- constructor source hash drift;
- one dependency not restored;
- one unapproved dependency call;
- wrong constructor call count or ordering;
- real process creation attempted;
- real Scheduler constructor attempted;
- real tokenizer access attempted;
- `capture_cudagraph` or CPU offload called;
- worker loop not intercepted;
- wrong runner class, rank, world size, or duplicate runner identity;
- wrong candidate published to a rank;
- candidate payload drift from the pristine oracle;
- non-empty Engine command payload;
- missing, duplicate, reordered, malformed, or mismatched acknowledgement;
- participant binding error leaves Engine aggregate completion unset;
- exact repeat dispatches again;
- scheduler, step, CUDA, forward, tokenization, sampling, or inference counter
  becomes non-zero;
- one selected tensor cannot be cleared;
- one retained private object escapes collection.

Constructor or candidate-preparation failures must restore every global and
release every already-created candidate. Binding failures must leave the
failed run non-authoritative and preserve its evidence directory.

## Cleanup and Memory

The gate must not call production `LLMEngine.exit()` or
`ModelRunner.exit()` because those methods contain CUDA, distributed, shared
memory, and worker-join side effects outside this boundary.

Cleanup instead:

1. clears selected rank destinations in reverse rank and reverse unique-object
   order;
2. proves non-selected tensors, tensor identities, and pool state unchanged;
3. closes inert channels and buffers;
4. clears publication slots and capsule registries only after evidence is
   serialized;
5. drops Engine, all runners, candidates, owners, runtime identities, models,
   pools, targets, requests, sentinels, and transport objects;
6. runs garbage collection;
7. proves every tracked private object collected;
8. proves no OS child PID, OS shared-memory name, atexit handler, distributed
   process group, or CUDA context was created.

Use these conservative CPU correctness ceilings:

```text
process total VmHWM increment:
  12582912 KiB
ready VmRSS:
  8388608 KiB
host MemAvailable decrease:
  12582912 KiB
minimum preflight MemAvailable:
  16777216 KiB
```

These ceilings are safety limits, not memory-efficiency or cache-saving
claims.

## Static Safety

The preflight and independent verifier must reject unless all of the following
hold:

- the exact production classes are imported from the frozen source files;
- the attached constructor methods retain the frozen hashes;
- `object.__new__`, constructor AST compilation, subclass construction, and
  class replacement are absent from the execution path;
- the replacement allowlist is exact and closed;
- no production Scheduler instance is constructed;
- no OS process, real shared memory, tokenizer I/O, or atexit registration
  occurs;
- no call reaches `LLMEngine.step`, `ModelRunner.run`, model forward,
  attention forward, sampler, CUDA, or inference;
- only read-only `torch.cuda.is_initialized()` observations are allowed;
- the exact real worker rejection remains:
  `RuntimeError: real checkpoint load worker execution is not implemented; only the local safety dry-run is authorized`;
- schema-v2 canonical status remains `NO_GO`.

## Artifacts and Independent Verification

Publish only:

```text
constructed_engine_model_runner_ownership.json
source_manifest.json
```

inside one unique run directory after successful cleanup.

The standard-library-only verifier must not import TinyLLMForge, torch, or the
gate. It independently validates:

- exact file inventory and hashes;
- prerequisite result, manifest, and oracle identities;
- production file and method hashes;
- exact constructor class identities and counts;
- closed capsule replacement set, call ordering, and restoration;
- all four complete pristine payloads;
- zero-payload all-rank command and acknowledgements;
- Engine aggregate state and exact-repeat no-op;
- all forbidden counters equal zero;
- memory ceilings and cleanup;
- absence of child processes, real shared memory, distributed initialization,
  CUDA initialization, and leaked objects.

Remote verification must run outside the authoritative run directory through
a read-only two-file view so the run inventory remains exact.

## Allowed Conclusion

Passing proves that the frozen production `LLMEngine.__init__` and
`ModelRunner.__init__` bodies constructed one exact Engine and four exact
rank-local runners under an audited inert dependency capsule, after which four
approved real checkpoint-derived candidates were published, owned, and bound
through the production zero-payload all-rank Engine method.

It does not prove:

- a production Scheduler was constructed or executed;
- real multiprocessing, NCCL, CUDA, GPU KV allocation, or shared memory;
- `LLMEngine.step()`, `ModelRunner.run()`, model/attention forward, sampling,
  tokenization, generation, or inference;
- output or logit correctness, accuracy, quality, or no-regression;
- latency, throughput, cache savings, GPU-memory savings, or compression.

The next gate after this one is real output/logit correctness under an
explicit, still-bounded runtime path. Performance and cache benchmarks follow
only after correctness passes.

## Authoritative Result

The source-bound remote gate completed on 2026-07-28 with:

```text
run tag:
  qwen35-constructed-engine-model-runner-ownership-20260728-181454
remote evidence:
  /data00/home/sitian/sitian-workspace01/tllm/
  qwen35-constructed-engine-model-runner-ownership-runs/
  qwen35-constructed-engine-model-runner-ownership-20260728-181454/
  evidence/
  qwen35-constructed-engine-model-runner-ownership-20260728-181454
local evidence:
  experiments/qwen35_hybrid_state/
  qwen35-constructed-engine-model-runner-ownership-20260728-181454
```

The authoritative directory contains exactly:

```text
constructed_engine_model_runner_ownership.json
source_manifest.json
```

Authoritative identities:

```text
source files:
  63
source tree:
  a2bf242ed69fe556419b0b340602a5293d9d849861aaad2ada0e232e7b4e4717
result:
  a3f499eacd19f80c676d71351f4c9904f6dd1be0bcb2cb4023dbefdebe029d0a
manifest:
  0c709933940cf9b293457308e688b9a44ea98c32a1c4d46ef766b8599906122e
pristine prerequisite:
  d750d664219378c234a2127b708ec191feb9b2c9f1f2902c47d0ad5dc152d3ef
```

The exact source closure is the pristine 58-file producer closure plus:

```text
tinyvllm/config.py
tinyvllm/engine/exact_cuda_graph_cache.py
tools/qwen35_tp4_live_concurrent_candidate_ownership_preflight.py
tools/qwen35_constructed_engine_model_runner_ownership_preflight.py
tools/verify_qwen35_constructed_engine_model_runner_ownership_gate.py
```

The result proves:

```text
LLMEngine.__init__ executions:
  1
ModelRunner.__init__ executions:
  4
ModelRunner constructor ranks:
  0, 1, 2, 3
exact production class identity:
  true for Engine and every runner
dependency restoration:
  complete
production all-rank participants:
  0, 1, 2, 3
participant status:
  bound, bound, bound, bound
command args:
  []
worker acknowledgement ranks:
  1, 2, 3
exact repeat:
  identical rows, zero dispatch
forbidden counter sum:
  0
CUDA initialized:
  false
cleanup release order:
  3, 2, 1, 0
tracked private objects collected:
  all
production exit calls:
  0
```

Memory safety evidence:

```text
process VmHWM increment:
  5809116 / 12582912 KiB
ready process VmRSS:
  4331468 / 8388608 KiB
host MemAvailable decrease:
  5695784 / 12582912 KiB
preflight host MemAvailable:
  1670933232 / 16777216 KiB minimum
```

The standard-library-only verifier passed `281 checks` both locally and on
the remote staged source against the same immutable two-file evidence.

Superseded source-bound attempts are preserved:

```text
...-180702:
  SSH did not preserve the bash -c script as one remote command argument;
  no remote run directory was created.
...-180852:
  source closure omitted tinyvllm/config.py; remote Python stopped before
  construction and produced no evidence.
...-181019:
  source closure omitted the real ExactCudaGraphCache runtime source;
  production import stopped before construction and produced no evidence.
...-181348:
  source closure omitted the direct live-concurrent producer wrapper;
  exact constructors completed, but candidate preparation did not start and
  no evidence was published.
```

These failures led to regression coverage for SSH argument boundaries, exact
source closure, the real cache-constructor runtime source, the closed
import-only stub surface, and the direct producer wrapper.

This result remains strictly within
`no-scheduler-step-forward-or-inference`. It is not evidence of output
correctness, accuracy, quality, latency, throughput, cache savings,
GPU-memory savings, or compression. The next gate is real output/logit
correctness; performance and cache measurements remain pending.
