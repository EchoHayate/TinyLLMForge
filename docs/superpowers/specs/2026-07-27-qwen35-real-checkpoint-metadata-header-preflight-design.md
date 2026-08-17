# Qwen3.5 Real Checkpoint Metadata-Header Preflight Design

## Status

Approved for inline implementation and one live read-only run on
`sitian@10.232.195.203`.

This is a new evidence gate. It does not modify, reinterpret, or overwrite the
existing stat-only `preflight.json` contract, whose payload-open count must
remain zero.

## Goal

Use the bounded checkpoint metadata reader against the approved remote
Qwen3.5-2B snapshot and prove that:

- actual `config.json` and index bytes match retained SHA256 identities;
- the declared safetensors shard has the retained exact size;
- only its 8-byte prefix and bounded JSON header are read;
- the parsed metadata builds the exact Qwen3.5 checkpoint tensor plan;
- no tensor data-section byte is read;
- no loader, pool, model, CUDA runtime, Engine path, or inference executes.

The gate produces metadata and topology evidence only. It cannot establish
payload integrity, assignment correctness, model correctness, speed, memory,
cache, compression, or quality improvement.

## Fixed Identity

```text
remote target:
  sitian@10.232.195.203
SSH control path:
  /tmp/ssh-sitian-10.232.195.203
remote Python:
  /data00/home/sitian/sitian-workspace01/tllm/env/bin/python
approved model directory:
  /data00/home/sitian/sitian-workspace01/tllm/qwen35-hybrid-state-runs/
  qwen35-2b-hybrid-acquire-20260723-222004/model
model manifest SHA256:
  3e650a908234771c3cf1ac4e20c4d38fe69982efedaf4a3e631ad0b14aad7dd0
config SHA256:
  ed1c1723241f23f7f4e23430759cbd7dcfb4103cbdfe052bfe7626b57c2615b4
index SHA256:
  aca8afed9da75b0f050b408d270766fd77627f1af401e240f61c3b47d0db02f9
shard:
  model.safetensors-00001-of-00001.safetensors
shard size:
  4548221488
retained shard full-file SHA256:
  aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1
canonical config/index/shard composite:
  27da983f5ef3e38480d8b5d5976e5c434fc4b5d0c70d09511c35154beecd8db9
```

The shard SHA256 is retained manifest evidence only. The gate must not
recompute it.

## Architecture

Create:

```text
tools/qwen35_real_checkpoint_metadata_preflight.py
```

The module has three layers.

### Record Contract

Define a frozen schema name:

```text
qwen35.real-checkpoint-metadata-preflight.v1
```

and:

```python
validate_metadata_preflight(record) -> Mapping[str, object]
```

The validator requires exact fixed identities, `status == "PASS"`, source
binding, positive bounded metadata accounting, `payload_bytes_read == 0`,
`payload_hashes_recomputed is False`, an exact one-shard inventory, and
internally consistent tensor-plan counts.

### Remote Worker

The internal worker receives only:

- approved checkpoint directory;
- retained shard identity;
- config/index/composite identities;
- output path.

It loads `qwen35_checkpoint_metadata.py` and `qwen35_checkpoint.py` directly by
file path. It must not import `tinyvllm` package initialization.

The worker:

1. calls `read_qwen35_checkpoint_metadata`;
2. calls `build_qwen35_checkpoint_tensor_plan`;
3. derives counts and byte totals from the returned immutable values;
4. records Python/host/user and source-file SHA256 values;
5. atomically writes `metadata_preflight.json`.

No exception is converted to `PASS`. Failure exits nonzero and publishes no
final artifact.

### Local Orchestrator

The local CLI supports:

```text
run --run-tag <unique-tag>
validate --artifact <path>
```

`run`:

1. validates the run tag;
2. creates a deterministic tar containing only:
   - `tinyvllm/models/qwen35_checkpoint_metadata.py`;
   - `tinyvllm/models/qwen35_checkpoint.py`;
   - `tools/qwen35_real_checkpoint_metadata_preflight.py`;
3. creates a unique remote directory without deletion or overwrite;
4. stages and remotely verifies exact source hashes;
5. invokes the staged worker with `CUDA_VISIBLE_DEVICES=""`;
6. reads the two authorized JSON artifacts through SSH;
7. validates exact round-trip equality;
8. atomically publishes a unique local evidence directory.

The local directory contains exactly:

```text
metadata_preflight.json
source_manifest.json
```

## Header-Only I/O Boundary

The only reads from a `.safetensors` file are:

```text
read(8)
read(header_length)
```

The gate records:

```text
metadata_bytes_read
payload_bytes_read: 0
payload_hashes_recomputed: false
payload_identity_source: retained_approved_manifest
```

`metadata_bytes_read` must equal:

```text
config file size
+ index file size
+ 8-byte prefix
+ declared JSON header length
```

The worker must not use:

```text
safetensors.safe_open
torch.load
mmap
read_bytes on the shard
full-file shard hashing
CUDA
subprocess
network
Engine
LLMEngine.step
checkpoint loader construction
```

The local orchestrator may use subprocess only for SSH transport.

## Tensor-Plan Evidence

The PASS record contains:

```text
layer_count
linear_attention_layer_count
full_attention_layer_count
index_weight_count
header_tensor_count
load_count
skip_count
plan_payload_bytes
index_total_size
shard_count
```

The validator requires:

- 24 layers, 18 linear-attention, and 6 full-attention;
- one declared/observed shard;
- positive index/header/load counts;
- `load_count + skip_count == index_weight_count`;
- `header_tensor_count == index_weight_count`;
- `plan_payload_bytes == index_total_size`;
- metadata bundle payload reads equal zero.

These are metadata topology checks, not payload reads.

## Safety

- Work only in `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Use only the fixed SSH ControlMaster and remote user.
- Never stage, commit, merge, or clean evidence.
- Never overwrite an existing local or remote run directory.
- Never invoke `tools/qwen35_real_checkpoint_load_worker.py`.
- Keep that worker's `main()` hard rejection unchanged.
- Do not modify ModelRunner, Engine, Scheduler, publication, or schema-v2
  canonical evidence.
- Do not claim performance or memory improvement.

## Tests

Dependency-light tests prove:

- exact record validation and fail-closed mutations;
- deterministic source tar and source-tree identity;
- direct-file worker imports do not initialize `tinyvllm`;
- local snapshot execution produces exact topology and zero payload bytes;
- generated SSH command uses the fixed target/control path;
- unique local/remote path behavior;
- only two artifacts are published;
- forbidden API/source scans remain clean;
- worker execution stub remains rejected.

The live run must additionally prove exact remote source hashes and validate
the downloaded PASS artifact locally.

## Allowed Conclusion

Passing proves the approved checkpoint's bounded JSON metadata is internally
consistent with TinyLLMForge's Qwen3.5 tensor-plan contract while reading zero
tensor payload bytes.

It does not verify the retained full-shard SHA256, load or assign tensors,
construct a model, initialize CUDA, execute inference, or establish any
production speed/cache/memory benefit.
