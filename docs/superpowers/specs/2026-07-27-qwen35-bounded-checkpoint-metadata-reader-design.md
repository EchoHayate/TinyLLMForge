# Qwen3.5 Bounded Checkpoint Metadata Reader Design

## Objective

Read and verify only the metadata required to construct a Qwen3.5 checkpoint
tensor plan:

- `config.json`;
- `model.safetensors.index.json`;
- the 8-byte length prefix and JSON header of each declared safetensors shard.

No tensor data-section byte may be read.

## Interface

Create:

```text
tinyvllm/models/qwen35_checkpoint_metadata.py
```

with:

```python
@dataclass(frozen=True)
class Qwen35CheckpointShardIdentity:
    name: str
    size: int
    sha256: str


@dataclass(frozen=True)
class Qwen35CheckpointMetadataBundle:
    hf_config: object
    index_payload: Mapping[str, object]
    shard_headers: Mapping[str, Mapping[str, object]]
    config_sha256: str
    index_sha256: str
    config_index_header_sha256: str
    metadata_bytes_read: int
    payload_bytes_read: int


def read_qwen35_checkpoint_metadata(
    checkpoint_dir,
    *,
    shards,
    expected_config_sha256,
    expected_index_sha256,
    expected_config_index_header_sha256,
    max_config_bytes=1 << 20,
    max_index_bytes=16 << 20,
    max_header_bytes=64 << 20,
) -> Qwen35CheckpointMetadataBundle:
    ...
```

## Identity Semantics

The composite identity preserves the existing safety-harness definition:

```text
sha256(canonical_json({
  "config_sha256": expected_config_sha256,
  "index_sha256": expected_index_sha256,
  "shards": {
    shard_name: {"sha256": ..., "size": ...},
  },
}))
```

It is a manifest identity, not a hash of newly read header bytes.

The reader verifies actual config/index bytes against their expected SHA256
values and verifies each shard's observed file size against its retained
identity. It does not recompute a full shard hash.

## Header-Only I/O

For each safe relative shard path:

1. stat and verify exact file size;
2. open in binary mode;
3. read exactly 8 bytes;
4. decode an unsigned little-endian header length;
5. require `1 <= header_length <= max_header_bytes`;
6. require `8 + header_length <= shard_size`;
7. read exactly `header_length` bytes;
8. parse the JSON object;
9. close the file without any further read.

`metadata_bytes_read` counts config bytes, index bytes, 8-byte prefixes, and
header bytes. `payload_bytes_read` is always zero.

## Parsing

Config JSON must be an object and is converted recursively to immutable-style
attribute containers (`SimpleNamespace`, tuples for arrays) so existing
Qwen3.5 tensor-plan and component factories can consume it.

Index JSON and shard headers remain mappings. Returned top-level mappings are
read-only proxies; nested JSON values are not mutated by the reader.

## Safety

- checkpoint directory must exist and resolve exactly;
- shard names must be safe normalized relative `.safetensors` paths;
- shard identities must be exact frozen values with unique names;
- file-size and byte budgets are positive bounded integers;
- short reads, invalid UTF-8/JSON, non-object JSON, invalid header lengths,
  identity conflicts, and unexpected shard sets fail closed;
- no `safetensors.safe_open`, `torch.load`, mmap, CUDA, network, subprocess, or
  Engine reference.

## Tests

Synthetic files prove:

- exact config/index/composite verification;
- header-only byte accounting with payload sentinel bytes unread;
- recursive config namespace conversion;
- multiple shard coverage and read-only top-level mappings;
- invalid path, duplicate shard, digest, size, budget, short prefix/header,
  oversized header, malformed JSON, and composite mismatch rejection;
- no payload read even when parsing fails after a valid header.

The real `/tmp` metadata snapshots may be used for a local metadata-only
regression, but no real safetensors payload is opened in this gate.

## Claim Boundary

Passing proves bounded metadata/header parsing and retained identity
verification. It does not verify full shard SHA256 values, load tensor payload,
construct a CUDA model, execute the worker, run inference, or establish any
performance or memory improvement.
