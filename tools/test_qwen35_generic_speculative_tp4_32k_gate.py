from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


TOOLS = Path(__file__).resolve().parent
FROZEN_GATE_PATH = (
    TOOLS / "qwen35_generic_speculative_tp4_gate.py"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gate = _load_module(
    "qwen35_generic_speculative_tp4_32k_gate",
    TOOLS / "qwen35_generic_speculative_tp4_32k_gate.py",
)
frozen_test = _load_module(
    "qwen35_generic_speculative_tp4_32k_fixtures",
    TOOLS / "test_qwen35_generic_speculative_tp4_gate.py",
)
frozen_test.gate = gate


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_contract_constants_are_frozen():
    assert gate.SCHEMA_VERSION == (
        "qwen35.generic-speculative-tp4-32k-"
        "transactional-correctness.v1"
    )
    assert gate.CLASSIFICATION == (
        "SECOND_MODEL_TP4_32K_ESTABLISHED"
    )
    assert gate.CLAIM_SCOPE == "second_model_tp4_32k_only"
    assert gate.WORLD_SIZE == 4
    assert gate.BATCH_SIZES == (1, 4)
    assert gate.POLICIES == ("baseline", "ngram")
    assert gate.CONTEXT_TOKENS == 32768
    assert gate.NGRAM_SIZE == 3
    assert gate.MAX_PROPOSAL_TOKENS == 4
    assert gate.MAX_OUTPUT_TOKENS == 8
    assert gate.MAX_MODEL_LEN == 33024
    assert gate.MAX_NUM_BATCHED_TOKENS == 132096
    assert gate.MAX_NUM_PREFILL_TOKENS_PER_STEP == 1024
    assert gate.KV_OFFLOAD_GPU_BLOCKS == 68
    assert gate.KV_OFFLOAD_LOGICAL_BLOCKS == 640
    assert gate.KV_OFFLOAD_BLOCKWISE_BLOCKS == 8
    assert gate.LIMITATIONS == (
        "phase1_not_promotable",
        "performance_not_established",
        "learned_drafter_not_established",
        "kv_quantization_not_established",
    )
    for source in (
        "tools/qwen35_generic_speculative_tp4_gate.py",
        "tools/qwen35_generic_speculative_tp4_worker.py",
        "tools/verify_qwen35_generic_speculative_tp4_gate.py",
        "tools/qwen35_generic_speculative_tp4_32k_gate.py",
        "tools/qwen35_generic_speculative_tp4_32k_worker.py",
        "tools/verify_qwen35_generic_speculative_tp4_32k_gate.py",
    ):
        assert source in gate.DEFAULT_SOURCE_FILES


def test_loading_32k_gate_does_not_modify_frozen_gate_source():
    before = _sha256(FROZEN_GATE_PATH)
    _load_module(
        "qwen35_generic_speculative_tp4_32k_gate_isolation",
        TOOLS / "qwen35_generic_speculative_tp4_32k_gate.py",
    )
    assert _sha256(FROZEN_GATE_PATH) == before


def _valid_result() -> dict:
    return frozen_test._valid_result()


def test_validate_result_accepts_positive_candidate_h2d():
    normalized = gate.validate_result(_valid_result())
    for batch_size in gate.BATCH_SIZES:
        movement = normalized["cells"][
            f"ngram:b{batch_size}"
        ]["kv_rank_deltas"]
        assert sum(row["h2d_copies"] for row in movement) > 0
        assert sum(row["h2d_bytes"] for row in movement) > 0


@pytest.mark.parametrize("batch_size", [1, 4])
def test_validate_result_rejects_zero_candidate_h2d_copies(
    batch_size,
):
    result = _valid_result()
    for row in result["cells"][
        f"ngram:b{batch_size}"
    ]["kv_rank_deltas"]:
        row["h2d_copies"] = 0
    with pytest.raises(
        ValueError,
        match=(
            f"32K batch-{batch_size} candidate requires "
            "real H2D copies"
        ),
    ):
        gate.validate_result(result)


@pytest.mark.parametrize("batch_size", [1, 4])
def test_validate_result_rejects_zero_candidate_h2d_bytes(
    batch_size,
):
    result = _valid_result()
    for row in result["cells"][
        f"ngram:b{batch_size}"
    ]["kv_rank_deltas"]:
        row["h2d_bytes"] = 0
    with pytest.raises(
        ValueError,
        match=(
            f"32K batch-{batch_size} candidate requires "
            "real H2D bytes"
        ),
    ):
        gate.validate_result(result)


def test_worker_uses_frozen_long_context_configuration():
    worker = _load_module(
        "qwen35_generic_speculative_tp4_32k_worker",
        TOOLS
        / "qwen35_generic_speculative_tp4_32k_worker.py",
    )
    factory_calls = []

    def engine_factory(model_path, **kwargs):
        factory_calls.append((model_path, dict(kwargs)))
        return frozen_test._FakeTP4Engine(
            kwargs["max_num_seqs"]
        )

    class Runtime:
        def __init__(self, adapter):
            self.adapter = adapter

    class Adapter:
        def __init__(
            self,
            *,
            ngram_size,
            max_proposal_tokens,
        ):
            self.ngram_size = ngram_size
            self.max_proposal_tokens = max_proposal_tokens

    class SamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    cell = worker.run_policy_cell(
        model_path="/model",
        gpu_indices=(0, 1, 2, 3),
        policy="ngram",
        batch_size=1,
        dist_port=31001,
        master_port=32001,
        engine_factory=engine_factory,
        sampling_params_type=SamplingParams,
        runtime_type=Runtime,
        adapter_type=Adapter,
        synchronize=lambda: None,
        run_generation_fn=frozen_test._fake_generation_runner,
    )

    kwargs = factory_calls[0][1]
    assert kwargs["tensor_parallel_size"] == 4
    assert kwargs["max_model_len"] == 33024
    assert kwargs["max_num_batched_tokens"] == 132096
    assert kwargs["max_num_prefill_tokens_per_step"] == 1024
    assert kwargs["chunked_prefill_decode_first"] is False
    assert kwargs["chunked_prefill_mixed_batch"] is False
    assert kwargs["kv_offload_gpu_blocks"] == 68
    assert kwargs["kv_offload_logical_blocks"] == 640
    assert kwargs["kv_offload_blockwise_blocks"] == 8
    assert len(cell["prompt_rows"][0]["token_ids"]) == 32768


def test_campaign_defaults_dispatch_32k_authority(
    tmp_path,
    monkeypatch,
):
    captured = {}

    def fake_run_campaign(**kwargs):
        captured.update(kwargs)
        return {"classification": gate.CLASSIFICATION}

    def fake_verifier(run_dir, source_root):
        return {"classification": "PASS", "failures": []}

    monkeypatch.setattr(
        gate,
        "_run_campaign_impl",
        fake_run_campaign,
    )
    monkeypatch.setattr(
        gate,
        "_load_32k_verifier",
        lambda: fake_verifier,
    )
    result = gate.run_campaign(
        model_path="/model",
        gpu_indices=(0, 1, 2, 3),
        output_dir=tmp_path / "authority",
        dist_port_base=31000,
        master_port_base=32000,
    )

    assert result["classification"] == (
        "SECOND_MODEL_TP4_32K_ESTABLISHED"
    )
    assert captured["worker_script"] == (
        TOOLS
        / "qwen35_generic_speculative_tp4_32k_worker.py"
    )
    assert captured["source_files"] == (
        gate.DEFAULT_SOURCE_FILES
    )
    assert captured["verifier"] is fake_verifier


def _load_verifier():
    return _load_module(
        "verify_qwen35_generic_speculative_tp4_32k_gate",
        TOOLS
        / "verify_qwen35_generic_speculative_tp4_32k_gate.py",
    )


def _write_valid_verifier_run(tmp_path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    source_path = source_root / "bound.py"
    source_path.write_text(
        "BOUND = True\n",
        encoding="utf-8",
    )
    source_files = ("bound.py",)
    result = _valid_result()
    result["source_tree_sha256"] = (
        gate.source_tree_sha256(
            source_root,
            source_files,
        )
    )
    run_dir = tmp_path / "authority"
    run_dir.mkdir()
    gate.atomic_write_json(run_dir / "result.json", result)
    gate.atomic_write_json(
        run_dir / "source_manifest.json",
        {
            "schema_version": gate.SCHEMA_VERSION,
            "source_tree_sha256": result[
                "source_tree_sha256"
            ],
            "model_manifest_sha256": result[
                "model_manifest_sha256"
            ],
            "source_files": gate.hash_source_files(
                source_root,
                source_files,
            ),
            "artifacts": {
                "result.json": gate.sha256_file(
                    run_dir / "result.json"
                ),
            },
        },
    )
    return run_dir, source_root


def _rewrite_result_and_manifest(run_dir, result):
    gate.atomic_write_json(run_dir / "result.json", result)
    manifest_path = run_dir / "source_manifest.json"
    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )
    manifest["artifacts"]["result.json"] = gate.sha256_file(
        run_dir / "result.json"
    )
    gate.atomic_write_json(manifest_path, manifest)


def test_verifier_accepts_valid_source_bound_32k_authority(
    tmp_path,
):
    run_dir, source_root = _write_valid_verifier_run(
        tmp_path
    )
    assert _load_verifier().verify_run(
        run_dir,
        source_root,
    ) == {
        "classification": "PASS",
        "failures": [],
    }


@pytest.mark.parametrize("batch_size", [1, 4])
def test_verifier_rejects_zero_candidate_h2d(
    tmp_path,
    batch_size,
):
    run_dir, _ = _write_valid_verifier_run(tmp_path)
    result = json.loads(
        (run_dir / "result.json").read_text(
            encoding="utf-8"
        )
    )
    for row in result["cells"][
        f"ngram:b{batch_size}"
    ]["kv_rank_deltas"]:
        row["h2d_copies"] = 0
    _rewrite_result_and_manifest(run_dir, result)

    verification = _load_verifier().verify_run(run_dir)

    assert verification["classification"] == "FAIL"
    assert (
        f"32K batch-{batch_size} candidate requires "
        "real H2D copies"
        in verification["failures"][0]
    )


def test_verifier_rejects_result_hash_tampering(tmp_path):
    run_dir, _ = _write_valid_verifier_run(tmp_path)
    result = json.loads(
        (run_dir / "result.json").read_text(
            encoding="utf-8"
        )
    )
    result["unbound_field"] = True
    gate.atomic_write_json(run_dir / "result.json", result)

    verification = _load_verifier().verify_run(run_dir)

    assert verification["classification"] == "FAIL"
    assert "result artifact SHA-256 mismatch" in (
        verification["failures"][0]
    )


def test_verifier_rejects_bound_source_tampering(tmp_path):
    run_dir, source_root = _write_valid_verifier_run(
        tmp_path
    )
    (source_root / "bound.py").write_text(
        "BOUND = False\n",
        encoding="utf-8",
    )

    verification = _load_verifier().verify_run(
        run_dir,
        source_root,
    )

    assert verification["classification"] == "FAIL"
    assert "current source file identity mismatch" in (
        verification["failures"][0]
    )


def test_remote_runner_source_contract():
    runner_path = (
        TOOLS
        / "run_qwen35_generic_speculative_tp4_32k_gate_remote.sh"
    )
    text = runner_path.read_text(encoding="utf-8")

    for required in (
        "sitian@10.232.195.203",
        "/data00/home/sitian/sitian-workspace01/tllm/env/bin/python",
        "/data00/home/sitian/sitian-workspace01/tllm/"
        "qwen35-hybrid-state-runs/"
        "qwen35-2b-hybrid-acquire-20260723-222004/model",
        "FILE:/Users/bytedance/krb5cc_sitian",
        "ControlMaster=no",
        "ControlPath=none",
        "qwen35_generic_speculative_tp4_32k_gate.py",
        "qwen35_generic_speculative_tp4_32k_worker.py",
        "verify_qwen35_generic_speculative_tp4_32k_gate.py",
        "qwen35_generic_speculative_tp4_gate.py",
        "qwen35_generic_speculative_tp4_worker.py",
        "verify_qwen35_generic_speculative_tp4_gate.py",
        "qwen35_generic_speculative_tp4_32k",
        "campaign.status",
        "campaign.pid",
        "campaign.exit_code",
        "authority.failed",
        "REMOTE_COMMAND_RETRY_ATTEMPTS",
        "REMOTE_RSYNC_RETRY_ATTEMPTS",
        "POLL_INTERVAL_SECONDS",
    ):
        assert required in text
    assert "head -n 4" in text
    assert "campaign already terminal" in text
    assert "campaign already running" in text
    assert "ControlMaster=yes" not in text
