from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT / "tools/qwen38_topology_local_tp2_island_worker.py"
)


def _load():
    assert MODULE_PATH.is_file(), (
        "Qwen3.8 topology-local TP2 island worker is missing"
    )
    sys.modules["torch"] = fake_torch
    spec = importlib.util.spec_from_file_location(
        "qwen38_topology_local_tp2_island_worker_under_test",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeTensor:

    def __init__(
        self,
        shape,
        *,
        label,
        dtype="torch.bfloat16",
        device="cuda:0",
        floating=True,
        contiguous=True,
    ):
        self.shape = tuple(shape)
        self.label = label
        self.dtype = dtype
        self.device = device
        self._floating = floating
        self._contiguous = contiguous

    @property
    def ndim(self):
        return len(self.shape)

    def is_floating_point(self):
        return self._floating

    def narrow(self, dim, start, length):
        shape = list(self.shape)
        assert 0 <= dim < len(shape)
        assert start >= 0
        assert length >= 0
        assert start + length <= shape[dim]
        shape[dim] = length
        return FakeTensor(
            shape,
            label=f"{self.label}.narrow({dim},{start},{length})",
            dtype=self.dtype,
            device=self.device,
            floating=self._floating,
            contiguous=False,
        )

    def contiguous(self):
        return FakeTensor(
            self.shape,
            label=f"{self.label}.contiguous",
            dtype=self.dtype,
            device=self.device,
            floating=self._floating,
            contiguous=True,
        )

    def to(self, *, dtype=None, device=None):
        return FakeTensor(
            self.shape,
            label=f"{self.label}.to",
            dtype=self.dtype if dtype is None else dtype,
            device=self.device if device is None else device,
            floating=self._floating,
            contiguous=self._contiguous,
        )

    def clone(self):
        return FakeTensor(
            self.shape,
            label=f"{self.label}.clone",
            dtype=self.dtype,
            device=self.device,
            floating=self._floating,
            contiguous=self._contiguous,
        )

    def squeeze(self, dim):
        shape = list(self.shape)
        assert shape[dim] == 1
        del shape[dim]
        return FakeTensor(
            shape,
            label=f"{self.label}.squeeze({dim})",
            dtype=self.dtype,
            device=self.device,
            floating=self._floating,
            contiguous=self._contiguous,
        )

    def repeat(self, *repeats):
        assert len(repeats) == len(self.shape)
        return FakeTensor(
            tuple(
                width * repeat
                for width, repeat in zip(self.shape, repeats)
            ),
            label=f"{self.label}.repeat{repeats}",
            dtype=self.dtype,
            device=self.device,
            floating=self._floating,
            contiguous=self._contiguous,
        )

    def __getitem__(self, item):
        if isinstance(item, slice):
            start, stop, step = item.indices(self.shape[0])
            assert step == 1
            shape = (max(0, stop - start),) + self.shape[1:]
            return FakeTensor(
                shape,
                label=f"{self.label}[{start}:{stop}]",
                dtype=self.dtype,
                device=self.device,
                floating=self._floating,
                contiguous=False,
            )
        raise TypeError("FakeTensor only implements leading slices")

    def numel(self):
        result = 1
        for width in self.shape:
            result *= width
        return result

    def element_size(self):
        return {
            "torch.bfloat16": 2,
            "torch.float16": 2,
            "torch.float32": 4,
            "torch.int8": 1,
        }[self.dtype]


fake_torch = types.ModuleType("torch")
fake_torch.Tensor = FakeTensor
fake_torch.bfloat16 = "torch.bfloat16"
fake_torch.float16 = "torch.float16"
fake_torch.float32 = "torch.float32"


def _cat(tensors, dim=0):
    tensors = tuple(tensors)
    shape = list(tensors[0].shape)
    shape[dim] = sum(tensor.shape[dim] for tensor in tensors)
    return FakeTensor(
        shape,
        label="cat(" + ",".join(tensor.label for tensor in tensors) + ")",
        dtype=tensors[0].dtype,
        device=tensors[0].device,
        floating=all(tensor._floating for tensor in tensors),
        contiguous=False,
    )


fake_torch.cat = _cat
sys.modules["torch"] = fake_torch
for package_name in ("tinyvllm", "tinyvllm.engine"):
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / package_name.replace(".", "/"))]
    sys.modules.setdefault(package_name, package)


class Qwen35LinearAttentionShell:
    pass


def fake_linear_layer(**overrides):
    layer = Qwen35LinearAttentionShell()
    values = {
        "local_key_heads": 4,
        "local_value_heads": 12,
        "key_head_dim": 128,
        "value_head_dim": 128,
        "norm_eps": 1e-6,
        "in_proj_qkv": SimpleNamespace(
            weight=FakeTensor(
                (10240, 5120),
                label="qkv",
            ),
            quant_method=None,
        ),
        "in_proj_z": SimpleNamespace(
            weight=FakeTensor(
                (6144, 5120),
                label="z",
            ),
            quant_method=None,
        ),
        "in_proj_b": SimpleNamespace(
            weight=FakeTensor(
                (48, 5120),
                label="b",
            ),
            quant_method=None,
        ),
        "in_proj_a": SimpleNamespace(
            weight=FakeTensor(
                (48, 5120),
                label="a",
            ),
            quant_method=None,
        ),
        "out_proj": SimpleNamespace(
            weight=FakeTensor(
                (5120, 1536),
                label="out-quarter",
                dtype="torch.bfloat16",
            ),
            prefill_weight=FakeTensor(
                (5120, 6144),
                label="out-full",
                dtype="torch.bfloat16",
            ),
            accumulation_weight=FakeTensor(
                (5120, 1536),
                label="out-quarter-fp32",
                dtype="torch.float32",
            ),
            quant_method=None,
            tp_size=4,
        ),
        "logical_tp2_conv_weight": FakeTensor(
            (10240, 4),
            label="conv-full",
        ),
        "logical_tp2_A_log": FakeTensor(
            (48,),
            label="A-log-full",
            dtype="torch.float32",
        ),
        "logical_tp2_dt_bias": FakeTensor(
            (48,),
            label="dt-bias-full",
        ),
        "norm_weight": FakeTensor(
            (128,),
            label="norm",
        ),
    }
    values.update(overrides)
    for name, value in values.items():
        setattr(layer, name, value)
    return layer


def test_case_matrix_has_frozen_shape_order_and_alternating_arms():
    worker = _load()

    rows = worker.build_case_matrix()

    assert len(rows) == 3 * (2 + 15)
    assert tuple(dict.fromkeys(
        row["active_tokens"] for row in rows
    )) == (1, 4, 8)
    assert [
        sum(
            row["active_tokens"] == active_tokens
            and row["phase"] == phase
            for row in rows
        )
        for active_tokens in (1, 4, 8)
        for phase in ("warmup", "measured")
    ] == [2, 15, 2, 15, 2, 15]
    measured = [row for row in rows if row["phase"] == "measured"]
    assert all(
        row["arm_order"]
        == (
            ("baseline", "candidate")
            if row["repetition"] % 2 == 0
            else ("candidate", "baseline")
        )
        for row in measured
    )
    assert worker.validate_case_matrix(rows) == rows


def test_case_matrix_rejects_missing_or_mutated_cases():
    worker = _load()
    rows = list(worker.build_case_matrix())

    with pytest.raises(ValueError, match="frozen"):
        worker.validate_case_matrix(rows[:-1])
    rows[0] = {**rows[0], "active_tokens": 2}
    with pytest.raises(ValueError, match="frozen"):
        worker.validate_case_matrix(rows)


def test_logical_view_selects_contiguous_half_and_pair_group():
    worker = _load()
    layer = fake_linear_layer()

    view = worker.build_logical_tp2_layer_view(
        layer,
        logical_rank=1,
        pair_group="pair-b",
    )

    assert view.logical_parallel_size == 2
    assert view.logical_rank == 1
    assert view.key_head_range == (8, 16)
    assert view.value_head_range == (24, 48)
    assert view.output_input_range == (3072, 6144)
    assert view.pair_group == "pair-b"
    assert view.qkv_weight is layer.in_proj_qkv.weight
    assert view.z_weight is layer.in_proj_z.weight
    assert view.b_weight is layer.in_proj_b.weight
    assert view.a_weight is layer.in_proj_a.weight
    assert view.qkv_weight.shape == (10240, 5120)
    assert view.z_weight.shape == (6144, 5120)
    assert view.b_weight.shape == (48, 5120)
    assert view.a_weight.shape == (48, 5120)
    assert view.conv_weight.shape == (5120, 4)
    assert view.A_log.shape == (24,)
    assert view.dt_bias.shape == (24,)
    assert view.output_accumulation_weight.shape == (5120, 3072)
    assert view.output_accumulation_weight.dtype == "torch.float32"
    assert tuple(view.tensor_digests) == (
        "A_log",
        "a_weight",
        "b_weight",
        "conv_weight",
        "dt_bias",
        "norm_weight",
        "output_accumulation_weight",
        "qkv_weight",
        "z_weight",
    )


def test_tensor_digest_is_device_independent():
    worker = _load()

    left = FakeTensor((2, 3), label="same", device="cuda:0")
    right = FakeTensor((2, 3), label="same", device="cuda:3")

    assert worker._tensor_digest(left) == worker._tensor_digest(right)


def test_parameter_identity_requires_exact_checkpoint_reconstruction():
    worker = _load()
    candidate = {"slice": "a" * 64}
    checkpoint = {"full": "b" * 64}

    record = worker.build_parameter_identity_record(
        candidate_slice_digests=candidate,
        checkpoint_full_digests=checkpoint,
        reconstructed_full_digests=dict(checkpoint),
    )

    assert record["checkpoint_reconstruction_match"] is True
    with pytest.raises(ValueError, match="reconstruct"):
        worker.build_parameter_identity_record(
            candidate_slice_digests=candidate,
            checkpoint_full_digests=checkpoint,
            reconstructed_full_digests={"full": "c" * 64},
        )


@pytest.mark.parametrize(
    "mutation",
    (
        lambda layer: object(),
        lambda layer: setattr(layer, "local_key_heads", 8) or layer,
        lambda layer: setattr(layer, "local_value_heads", 24) or layer,
        lambda layer: setattr(layer, "key_head_dim", 64) or layer,
        lambda layer: setattr(layer, "value_head_dim", 64) or layer,
        lambda layer: setattr(
            layer.out_proj,
            "prefill_weight",
            None,
        ) or layer,
        lambda layer: setattr(
            layer.out_proj,
            "accumulation_weight",
            None,
        ) or layer,
        lambda layer: setattr(
            layer.out_proj,
            "quant_method",
            object(),
        ) or layer,
        lambda layer: setattr(
            layer.in_proj_qkv,
            "quant_method",
            object(),
        ) or layer,
    ),
)
def test_logical_view_rejects_wrong_layer_or_weight_contract(mutation):
    worker = _load()
    layer = mutation(fake_linear_layer())

    with pytest.raises(ValueError):
        worker.build_logical_tp2_layer_view(
            layer,
            logical_rank=0,
            pair_group="pair-a",
        )


def test_candidate_setup_is_forbidden_after_warmup_starts():
    worker = _load()
    lifecycle = worker.CandidateSetupLifecycle()
    lifecycle.mark_warmup_started()

    with pytest.raises(RuntimeError, match="before warmup"):
        lifecycle.register_candidate_view(
            worker.build_logical_tp2_layer_view(
                fake_linear_layer(),
                logical_rank=0,
                pair_group="pair-a",
            )
        )


def test_candidate_reduce_uses_pair_group_not_global_group():
    worker = _load()
    calls = []
    tensor = object()

    result = worker._pair_reduce(
        tensor,
        "pair-a",
        distributed=SimpleNamespace(
            all_reduce=lambda value, *, group: calls.append(
                (value, group)
            )
        ),
    )

    assert result is tensor
    assert calls == [(tensor, "pair-a")]


def test_checkpoint_state_tensor_slices_preserve_fused_segment_order():
    worker = _load()

    assert worker.checkpoint_state_tensor_slices(0) == {
        "conv1d.weight": (
            (0, 1024),
            (2048, 1024),
            (4096, 3072),
        ),
        "A_log": ((0, 24),),
        "dt_bias": ((0, 24),),
    }
    assert worker.checkpoint_state_tensor_slices(1) == {
        "conv1d.weight": (
            (1024, 1024),
            (3072, 1024),
            (7168, 3072),
        ),
        "A_log": ((24, 24),),
        "dt_bias": ((24, 24),),
    }


def test_persistent_reservation_matches_frozen_integrated_cost():
    worker = _load()

    row = worker.project_persistent_reservation()

    assert row["unmeasured_output_projection_bytes"] == (
        47 * 30 * 1024 * 1024
    )
    assert row["capacity_eight_state_increment_bytes"] == int(
        295.5 * 1024 * 1024
    )
    assert row["unmeasured_parameter_increment_bytes"] == 47 * (
        (5120 * 4 * 2)
        + (12 * 4)
        + (12 * 2)
    )
    assert row["reservation_bytes"] == sum((
        row["unmeasured_output_projection_bytes"],
        row["capacity_eight_state_increment_bytes"],
        row["unmeasured_parameter_increment_bytes"],
    ))
    assert row["projected_integrated_increment_bytes"] <= (
        1920 * 1024 * 1024
    )


def test_case_seeds_share_hidden_but_distinguish_state_quarters():
    worker = _load()

    rows = [worker.case_seeds(1234, rank) for rank in range(4)]

    assert {row["hidden"] for row in rows} == {1234}
    assert len({row["state"] for row in rows}) == 4
    assert [row["state"] for row in rows] == [
        1234,
        101234,
        201234,
        301234,
    ]


def test_locate_layer_zero_requires_linear_attention():
    worker = _load()
    mixer = fake_linear_layer()
    model = SimpleNamespace(
        layer_stack=SimpleNamespace(
            layers=[
                SimpleNamespace(
                    block_type="linear_attention",
                    linear_attention=mixer,
                )
            ]
        )
    )

    assert worker.locate_layer_zero_linear_attention(model) is mixer

    model.layer_stack.layers[0].block_type = "full_attention"
    with pytest.raises(ValueError, match="linear attention"):
        worker.locate_layer_zero_linear_attention(model)


def test_checkpoint_state_loader_reads_only_layer_zero_sources(
    tmp_path,
    monkeypatch,
):
    worker = _load()
    monkeypatch.setattr(
        worker,
        "build_parameter_identity_record",
        lambda **kwargs: {
            **kwargs,
            "parameter_digests": kwargs["candidate_slice_digests"],
            "checkpoint_full_parameter_digests": (
                kwargs["checkpoint_full_digests"]
            ),
            "reconstructed_full_parameter_digests": (
                kwargs["reconstructed_full_digests"]
            ),
            "checkpoint_reconstruction_match": True,
        },
    )
    names = {
        "model.language_model.layers.0.linear_attn.conv1d.weight":
            "model-00001-of-00002.safetensors",
        "model.language_model.layers.0.linear_attn.A_log":
            "model-00002-of-00002.safetensors",
        "model.language_model.layers.0.linear_attn.dt_bias":
            "model-00002-of-00002.safetensors",
        "model.language_model.layers.1.linear_attn.conv1d.weight":
            "model-00001-of-00002.safetensors",
    }
    (tmp_path / "model.safetensors.index.json").write_text(
        __import__("json").dumps({"weight_map": names})
    )
    opened = []
    tensors = {
        "model.language_model.layers.0.linear_attn.conv1d.weight":
            FakeTensor((10240, 1, 4), label="conv", device="cpu"),
        "model.language_model.layers.0.linear_attn.A_log":
            FakeTensor((48,), label="A", device="cpu"),
        "model.language_model.layers.0.linear_attn.dt_bias":
            FakeTensor((48,), label="dt", device="cpu"),
    }

    class Handle:
        def __init__(self, path):
            self.path = path

        def __enter__(self):
            opened.append(self.path.name)
            return self

        def __exit__(self, *_args):
            return False

        def keys(self):
            return [
                name
                for name, shard in names.items()
                if shard == self.path.name
            ]

        def get_tensor(self, name):
            return tensors[name]

    result = worker.load_logical_tp2_state_parameters(
        tmp_path,
        logical_rank=1,
        device="cuda:3",
        safe_open_factory=lambda path, **_kwargs: Handle(Path(path)),
    )

    assert sorted(opened) == [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]
    assert result["conv_weight"].shape == (5120, 4)
    assert result["A_log"].shape == (24,)
    assert result["A_log"].dtype == "torch.float32"
    assert result["dt_bias"].shape == (24,)
    assert all(
        result[name].device == "cuda:3"
        for name in ("conv_weight", "A_log", "dt_bias")
    )
    assert result["parameter_identity"][
        "checkpoint_reconstruction_match"
    ] is True


def test_decode_norm_padding_uses_logical_parallel_size_two():
    worker = _load()
    calls = []

    result = worker._apply_candidate_gated_rmsnorm(
        FakeTensor((24, 128), label="core"),
        FakeTensor((24, 128), label="gate"),
        FakeTensor((128,), label="norm"),
        token_count=1,
        logical_parallel_size=2,
        eps=1e-6,
        gated_rmsnorm=lambda core, gate, weight, *, eps: (
            calls.append((core.shape, gate.shape, weight.shape, eps))
            or FakeTensor(core.shape, label="normalized")
        ),
    )

    assert calls == [((48, 128), (48, 128), (128,), 1e-6)]
    assert result.shape == (24, 128)


def test_migration_record_requires_release_before_steady_timing():
    worker = _load()
    record = worker.build_state_migration_record(
        latency_ns=100,
        source_bytes=40,
        transferred_bytes=20,
        retained_bytes=20,
        temporary_peak_allocated_bytes=60,
        steady_allocated_bytes=20,
        source_digest="a" * 64,
        candidate_digest="b" * 64,
        temporary_allocated_bytes_after_release=0,
    )

    assert record["temporary_released_before_timing"] is True
    assert set(record) == {
        "latency_ns",
        "source_bytes",
        "transferred_bytes",
        "retained_bytes",
        "temporary_peak_allocated_bytes",
        "steady_allocated_bytes",
        "source_digest",
        "candidate_digest",
        "temporary_allocated_bytes_after_release",
        "temporary_released_before_timing",
    }

    with pytest.raises(RuntimeError, match="1 bytes"):
        worker.build_state_migration_record(
            **{
                **record,
                "temporary_allocated_bytes_after_release": 1,
            }
        )


def test_campaign_rejects_missing_rank_and_path_escape(tmp_path):
    worker = _load()
    root = tmp_path / "attempt"

    with pytest.raises(RuntimeError, match="rank"):
        worker.validate_complete_rank_rows([
            {"rank": 0},
            {"rank": 1},
            {"rank": 2},
        ])
    assert worker.resolve_attempt_output(root, "rank-0.jsonl") == (
        root.resolve() / "rank-0.jsonl"
    )
    with pytest.raises(ValueError, match="outside"):
        worker.resolve_attempt_output(root, "../escaped.json")


def test_cleanup_is_idempotent_and_unpublishes_on_failure():
    worker = _load()
    destroyed = []
    released = []
    lifecycle = worker.CandidateStateLifecycle()
    lifecycle.publish(
        request_id=7,
        generation=3,
        slot_id=2,
        layer_index=0,
    )
    cleanup = worker.OwnedWorkerResources(
        process_groups=["pair-a", "pair-b"],
        tensor_reservations=["weight-reservation", "state-reservation"],
        destroy_group=lambda group: destroyed.append(group),
        release_tensor=lambda tensor: released.append(tensor),
        candidate_state_lifecycle=lifecycle,
    )

    receipt = cleanup.close(failed=True)
    repeated = cleanup.close(failed=True)

    assert receipt == repeated
    assert destroyed == ["pair-b", "pair-a"]
    assert released == ["state-reservation", "weight-reservation"]
    assert lifecycle.published_identity is None
    assert receipt["process_groups_destroyed"] == 2
    assert receipt["tensor_reservations_released"] == 2
    assert receipt["candidate_state_unpublished"] is True


def test_failure_record_preserves_stage_and_traceback():
    worker = _load()

    try:
        raise AssertionError("diagnostic sentinel")
    except AssertionError as error:
        record = worker.build_failure_record(
            error,
            stage="checkpoint_model_load",
        )

    assert record["type"] == "AssertionError"
    assert record["message"] == "diagnostic sentinel"
    assert record["stage"] == "checkpoint_model_load"
    assert "AssertionError: diagnostic sentinel" in record["traceback"]


def test_candidate_state_rejects_stale_identity():
    worker = _load()
    lifecycle = worker.CandidateStateLifecycle()
    lifecycle.publish(
        request_id=7,
        generation=3,
        slot_id=2,
        layer_index=0,
    )

    with pytest.raises(RuntimeError, match="identity"):
        lifecycle.require(
            request_id=7,
            generation=4,
            slot_id=2,
            layer_index=0,
        )


def test_success_cleanup_retires_published_candidate_state():
    worker = _load()
    lifecycle = worker.CandidateStateLifecycle()
    lifecycle.publish(
        request_id=7,
        generation=3,
        slot_id=2,
        layer_index=0,
    )
    cleanup = worker.OwnedWorkerResources(
        process_groups=[],
        tensor_reservations=[],
        destroy_group=lambda _group: None,
        release_tensor=lambda _tensor: None,
        candidate_state_lifecycle=lifecycle,
    )

    receipt = cleanup.close(failed=False)

    assert lifecycle.published_identity is None
    assert receipt["candidate_state_unpublished"] is True


def test_lifecycle_record_requires_observed_runtime_proofs():
    worker = _load()
    record = worker.build_lifecycle_record(
        state_identity_match=True,
        stale_generation_rejected=True,
        different_request_rejected=True,
        publish_after_success=True,
        baseline_state_unchanged=True,
        temporary_state_retired=True,
        fallback_count=0,
    )

    assert record["state_identity_match"] is True
    assert record["stale_generation_rejected"] is True
    assert record["different_request_rejected"] is True
    with pytest.raises(ValueError, match="proof"):
        worker.build_lifecycle_record(
            state_identity_match=True,
            stale_generation_rejected=False,
            different_request_rejected=True,
            publish_after_success=True,
            baseline_state_unchanged=True,
            temporary_state_retired=True,
            fallback_count=0,
        )


def test_timed_pair_preclones_inputs_and_syncs_after_both_submissions():
    worker = _load()
    source = inspect.getsource(worker.run_mixer_pair)
    timed_loop = source.split('for arm in case["arm_order"]:', 1)[1]
    timed_loop = timed_loop.split(
        "torch.cuda.current_stream().synchronize()", 1
    )[0]

    assert ".clone()" not in timed_loop
    assert timed_loop.count("_timed_arm(") == 2
    assert source.count(
        "torch.cuda.current_stream().synchronize()"
    ) == 1


def test_candidate_timed_path_has_only_pair_local_collective():
    worker = _load()
    source = inspect.getsource(worker.run_candidate_mixer)

    assert "_pair_reduce(local, view.pair_group)" in source
    assert "dist.all_reduce" not in source
    assert "barrier" not in source
    assert "synchronize" not in source


def test_campaign_binds_rank_device_before_cuda_use():
    worker = _load()
    source = inspect.getsource(worker.run_worker_campaign)

    device_assignment = source.index(
        'device = torch.device("cuda", rank)'
    )
    set_device = source.index("torch.cuda.set_device(device)")
    capability = source.index("_runtime_capability(rank, device)")

    assert device_assignment < set_device < capability


def test_campaign_only_registers_the_member_pair_group_for_cleanup():
    worker = _load()
    source = inspect.getsource(worker.run_worker_campaign)

    assert "process_groups=[None, pair_group]" in source
    assert "process_groups=[None, *created_groups]" not in source


def test_component_diagnostics_run_only_after_formal_pair_sync():
    worker = _load()
    source = inspect.getsource(worker.run_mixer_pair)

    synchronize = source.index(
        "torch.cuda.current_stream().synchronize()"
    )
    diagnostic = source.index("_run_candidate_diagnostic(")

    assert synchronize < diagnostic
