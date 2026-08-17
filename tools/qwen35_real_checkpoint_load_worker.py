from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

build_qwen35_checkpoint_tensor_plan = None
Qwen35CheckpointManifestIdentity = None
Qwen35RankCheckpointLoaderConfiguration = None
Qwen35CheckpointMetadataBundle = None


def _resolve_construction_dependencies():
    global build_qwen35_checkpoint_tensor_plan
    global Qwen35CheckpointManifestIdentity
    global Qwen35RankCheckpointLoaderConfiguration

    if build_qwen35_checkpoint_tensor_plan is None:
        from tinyvllm.models.qwen35_checkpoint import (
            build_qwen35_checkpoint_tensor_plan as build_tensor_plan,
        )

        build_qwen35_checkpoint_tensor_plan = build_tensor_plan
    if (
        Qwen35CheckpointManifestIdentity is None
        or Qwen35RankCheckpointLoaderConfiguration is None
    ):
        from tinyvllm.models.qwen35_checkpoint_loader_configuration import (
            Qwen35CheckpointManifestIdentity as ManifestIdentity,
            Qwen35RankCheckpointLoaderConfiguration as LoaderConfiguration,
        )

        Qwen35CheckpointManifestIdentity = ManifestIdentity
        Qwen35RankCheckpointLoaderConfiguration = LoaderConfiguration
    return (
        build_qwen35_checkpoint_tensor_plan,
        Qwen35CheckpointManifestIdentity,
        Qwen35RankCheckpointLoaderConfiguration,
    )


def _resolve_metadata_dependency():
    global Qwen35CheckpointMetadataBundle

    if Qwen35CheckpointMetadataBundle is None:
        from tinyvllm.models.qwen35_checkpoint_metadata import (
            Qwen35CheckpointMetadataBundle as MetadataBundle,
        )

        Qwen35CheckpointMetadataBundle = MetadataBundle
    return Qwen35CheckpointMetadataBundle


def build_qwen35_real_checkpoint_rank_loader(
    hf_config,
    index_payload,
    shard_headers,
    *,
    checkpoint_dir,
    model_manifest_sha256,
    config_sha256,
    index_sha256,
    config_index_header_sha256,
    tensor_parallel_size,
    tensor_parallel_rank,
    create_pool,
    build_attention_backend,
    authorization_sha256,
):
    (
        build_tensor_plan,
        manifest_type,
        configuration_type,
    ) = _resolve_construction_dependencies()
    tensor_plan = build_tensor_plan(
        hf_config,
        index_payload,
        shard_headers,
    )
    manifest = manifest_type(
        checkpoint_dir=checkpoint_dir,
        model_manifest_sha256=model_manifest_sha256,
        config_sha256=config_sha256,
        index_sha256=index_sha256,
        config_index_header_sha256=config_index_header_sha256,
    )
    configuration = configuration_type(
        manifest=manifest,
        hf_config=hf_config,
        tensor_plan=tensor_plan,
        tensor_parallel_size=tensor_parallel_size,
        tensor_parallel_rank=tensor_parallel_rank,
        create_pool=create_pool,
        build_attention_backend=build_attention_backend,
        authorization_sha256=authorization_sha256,
    )
    return configuration.build_loader()


def build_qwen35_real_checkpoint_rank_loader_from_metadata(
    metadata,
    *,
    checkpoint_dir,
    model_manifest_sha256,
    tensor_parallel_size,
    tensor_parallel_rank,
    create_pool,
    build_attention_backend,
    authorization_sha256,
):
    metadata_type = _resolve_metadata_dependency()
    if type(metadata) is not metadata_type:
        raise ValueError(
            "metadata must be an exact Qwen35CheckpointMetadataBundle"
        )
    if (
        isinstance(metadata.payload_bytes_read, bool)
        or metadata.payload_bytes_read != 0
    ):
        raise ValueError(
            "metadata bundle must report zero payload bytes read"
        )
    return build_qwen35_real_checkpoint_rank_loader(
        metadata.hf_config,
        metadata.index_payload,
        metadata.shard_headers,
        checkpoint_dir=checkpoint_dir,
        model_manifest_sha256=model_manifest_sha256,
        config_sha256=metadata.config_sha256,
        index_sha256=metadata.index_sha256,
        config_index_header_sha256=(
            metadata.config_index_header_sha256
        ),
        tensor_parallel_size=tensor_parallel_size,
        tensor_parallel_rank=tensor_parallel_rank,
        create_pool=create_pool,
        build_attention_backend=build_attention_backend,
        authorization_sha256=authorization_sha256,
    )


def main():
    raise RuntimeError(
        "real checkpoint load worker execution is not implemented; "
        "only the local safety dry-run is authorized"
    )


if __name__ == "__main__":
    main()

