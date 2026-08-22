"""Chunked prefill scheduler/block-manager tests.

跑法：python3 tools/test_chunked_prefill.py
"""

import os
import sys
import types
import importlib.util
import ast
import hashlib
import pickle
import tempfile
from types import SimpleNamespace

try:
    import torch
    import torch.distributed as dist
except ModuleNotFoundError:
    torch = None
    dist = None

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

tinyvllm_pkg = types.ModuleType("tinyvllm")
tinyvllm_pkg.__path__ = [os.path.join(_REPO_ROOT, "tinyvllm")]
engine_pkg = types.ModuleType("tinyvllm.engine")
engine_pkg.__path__ = [os.path.join(_REPO_ROOT, "tinyvllm", "engine")]
utils_pkg = types.ModuleType("tinyvllm.utils")
utils_pkg.__path__ = [os.path.join(_REPO_ROOT, "tinyvllm", "utils")]
layers_pkg = types.ModuleType("tinyvllm.layers")
layers_pkg.__path__ = [os.path.join(_REPO_ROOT, "tinyvllm", "layers")]
sys.modules.setdefault("tinyvllm", tinyvllm_pkg)
sys.modules.setdefault("tinyvllm.engine", engine_pkg)
sys.modules.setdefault("tinyvllm.utils", utils_pkg)
sys.modules.setdefault("tinyvllm.layers", layers_pkg)


def build_engine_speculative_partition(
    record,
    seqs,
    *,
    expected_schedule_generation,
):
    del record
    return SimpleNamespace(
        schedule_generation=expected_schedule_generation,
        selected_sequence_ids=(),
        suppressed_sequence_ids=tuple(
            seq.seq_id for seq in seqs
        ),
        selected_sequences=(),
        suppressed_sequences=seqs,
    )


def load_module(module_name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(module_name, os.path.join(_REPO_ROOT, relative_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_class_method(relative_path: str, class_name: str, method_name: str):
    path = os.path.join(_REPO_ROOT, relative_path)
    tree = ast.parse(open(path).read(), filename=path)
    class_node = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method_node = next((
        node for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    ), None)
    assert method_node is not None, (
        f"{class_name}.{method_name} is missing from {relative_path}"
    )
    function_node = ast.FunctionDef(
        name=method_node.name,
        args=method_node.args,
        body=method_node.body,
        decorator_list=[],
        returns=method_node.returns,
        type_comment=method_node.type_comment,
    )
    namespace = {
        "build_engine_speculative_partition": (
            build_engine_speculative_partition
        ),
    }
    exec(compile(ast.fix_missing_locations(ast.Module(
        body=[function_node],
        type_ignores=[],
    )), path, "exec"), namespace)
    return namespace[method_name]


def load_function(relative_path: str, function_name: str):
    path = os.path.join(_REPO_ROOT, relative_path)
    tree = ast.parse(open(path).read(), filename=path)
    function_node = next((
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == function_name
    ), None)
    assert function_node is not None, (
        f"{function_name} is missing from {relative_path}"
    )
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(
        body=[function_node],
        type_ignores=[],
    )), path, "exec"), namespace)
    return namespace[function_name]


def test_explicit_kv_capacity_is_pinned_and_fails_closed():
    resolve = load_function(
        "tinyvllm/engine/model_runner.py",
        "_resolve_kv_cache_blocks",
    )
    assert resolve(-1, 1819) == 1819
    assert resolve(1819, 2075) == 1819
    assert resolve(1819, 1819) == 1819
    try:
        resolve(2075, 1819)
    except ValueError as exc:
        assert "exceeds available KV cache capacity" in str(exc)
    else:
        raise AssertionError(
            "explicit KV capacity above auto capacity was accepted"
        )


def test_adaptive_mixed_config_defaults_and_fail_closed_contract():
    expected_defaults = {
        "chunked_prefill_adaptive_mixed": False,
        "chunked_prefill_adaptive_enter_waiting": 8,
        "chunked_prefill_adaptive_exit_waiting": 2,
        "chunked_prefill_adaptive_transition_steps": 2,
        "chunked_prefill_adaptive_max_mixed_steps": 2,
    }
    source = open(
        os.path.join(_REPO_ROOT, "tinyvllm/config.py"),
        encoding="utf-8",
    ).read()
    tree = ast.parse(source)
    config_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Config"
    )
    defaults = {
        node.target.id: ast.literal_eval(node.value)
        for node in config_class.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.value is not None
        and node.target.id in expected_defaults
    }
    assert defaults == expected_defaults
    for fragment in (
        "chunked_prefill_adaptive_enter_waiting > 0",
        "chunked_prefill_adaptive_exit_waiting >= 0",
        "chunked_prefill_adaptive_transition_steps > 0",
        "chunked_prefill_adaptive_max_mixed_steps > 0",
        "chunked_prefill_adaptive_exit_waiting < self.chunked_prefill_adaptive_enter_waiting",
        "not (self.chunked_prefill_adaptive_mixed and self.chunked_prefill_mixed_batch)",
        "not (self.chunked_prefill_adaptive_mixed and self.kv_offload_mvp0)",
        "not self.chunked_prefill_adaptive_mixed or self.max_num_prefill_tokens_per_step > 0",
    ):
        assert fragment in source


def load_real_config_class():
    module_name = "tinyvllm_config_contract_under_test"
    module_path = os.path.join(_REPO_ROOT, "tinyvllm/config.py")
    fake_transformers = types.ModuleType("transformers")

    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(model):
            del model
            return SimpleNamespace(
                max_position_embeddings=4096,
                num_hidden_layers=4,
            )

    fake_transformers.AutoConfig = FakeAutoConfig
    original = sys.modules.get("transformers")
    sys.modules["transformers"] = fake_transformers
    try:
        module = types.ModuleType(module_name)
        module.__file__ = module_path
        sys.modules[module_name] = module
        source = open(module_path, encoding="utf-8").read()
        exec(
            compile(
                "from __future__ import annotations\n" + source,
                module_path,
                "exec",
            ),
            module.__dict__,
        )
        return module.Config
    finally:
        if original is None:
            sys.modules.pop("transformers", None)
        else:
            sys.modules["transformers"] = original


def test_adaptive_mixed_invalid_configurations_fail_before_model_start():
    Config = load_real_config_class()
    with tempfile.TemporaryDirectory() as model:
        common = {
            "model": model,
            "max_num_batched_tokens": 4096,
            "max_model_len": 4096,
            "kvcache_block_size": 256,
        }
        invalid = (
            {"chunked_prefill_adaptive_enter_waiting": 0},
            {"chunked_prefill_adaptive_exit_waiting": -1},
            {"chunked_prefill_adaptive_transition_steps": 0},
            {"chunked_prefill_adaptive_max_mixed_steps": 0},
            {
                "chunked_prefill_adaptive_enter_waiting": 2,
                "chunked_prefill_adaptive_exit_waiting": 2,
            },
            {
                "chunked_prefill_adaptive_mixed": True,
                "max_num_prefill_tokens_per_step": 0,
            },
            {
                "chunked_prefill_adaptive_mixed": True,
                "max_num_prefill_tokens_per_step": 128,
                "chunked_prefill_mixed_batch": True,
            },
            {
                "chunked_prefill_adaptive_mixed": True,
                "max_num_prefill_tokens_per_step": 128,
                "kv_offload_mvp0": True,
            },
        )
        for overrides in invalid:
            try:
                Config(**common, **overrides)
            except AssertionError:
                pass
            else:
                raise AssertionError(
                    f"invalid adaptive config accepted: {overrides}"
                )


def test_slo_mixed_config_defaults_and_fail_closed_contract():
    Config = load_real_config_class()
    with tempfile.TemporaryDirectory() as model:
        common = {
            "model": model,
            "max_num_batched_tokens": 4096,
            "max_model_len": 4096,
            "kvcache_block_size": 256,
        }
        config = Config(**common)
        assert config.chunked_prefill_slo_mixed is False
        assert config.chunked_prefill_slo_target_gap_ns == 0
        assert config.chunked_prefill_slo_reserve_ns == 0
        assert config.chunked_prefill_slo_cost_intercept_ns == 0
        assert config.chunked_prefill_slo_cost_per_prefill_token_ns == 0
        assert config.chunked_prefill_slo_min_chunk_tokens == 16

        enabled = {
            "chunked_prefill_slo_mixed": True,
            "chunked_prefill_slo_target_gap_ns": 64_000_000,
            "chunked_prefill_slo_reserve_ns": 8_000_000,
            "chunked_prefill_slo_cost_intercept_ns": 4_000_000,
            "chunked_prefill_slo_cost_per_prefill_token_ns": 100_000,
            "chunked_prefill_slo_min_chunk_tokens": 16,
            "max_num_prefill_tokens_per_step": 128,
        }
        Config(**common, **enabled)
        invalid = (
            {**enabled, "chunked_prefill_slo_target_gap_ns": 0},
            {**enabled, "chunked_prefill_slo_reserve_ns": 0},
            {**enabled, "chunked_prefill_slo_reserve_ns": 64_000_000},
            {**enabled, "chunked_prefill_slo_cost_intercept_ns": 0},
            {**enabled, "chunked_prefill_slo_cost_per_prefill_token_ns": 0},
            {**enabled, "chunked_prefill_slo_min_chunk_tokens": 0},
            {**enabled, "chunked_prefill_slo_min_chunk_tokens": 256},
            {**enabled, "max_num_prefill_tokens_per_step": 120},
            {**enabled, "chunked_prefill_mixed_batch": True},
            {**enabled, "chunked_prefill_adaptive_mixed": True},
            {**enabled, "kv_offload_mvp0": True},
            {
                **enabled,
                "chunked_prefill_slo_cost_intercept_ns": 1 << 63,
            },
            {
                **enabled,
                "chunked_prefill_slo_cost_per_prefill_token_ns":
                    ((1 << 63) - 1) // 128 + 1,
            },
        )
        for overrides in invalid:
            try:
                Config(**common, **overrides)
            except AssertionError:
                pass
            else:
                raise AssertionError(
                    f"invalid P5 config accepted: {overrides}"
                )


config_mod = types.ModuleType("tinyvllm.config")
config_mod.Config = object
sys.modules["tinyvllm.config"] = config_mod
xxhash_mod = types.ModuleType("xxhash")


class _FakeXXH64:
    def __init__(self):
        self._h = hashlib.blake2b(digest_size=8)

    def update(self, data):
        self._h.update(data)

    def intdigest(self):
        return int.from_bytes(self._h.digest(), "little")


xxhash_mod.xxh64 = _FakeXXH64
sys.modules.setdefault("xxhash", xxhash_mod)
sampling_mod = load_module("tinyvllm.sampling_params", "tinyvllm/sampling_params.py")
context_mod = load_module("tinyvllm.utils.context", "tinyvllm/utils/context.py") if torch is not None else None
sequence_mod = load_module("tinyvllm.engine.sequence", "tinyvllm/engine/sequence.py")
block_manager_mod = load_module("tinyvllm.engine.block_manager", "tinyvllm/engine/block_manager.py")
scheduler_mod = load_module("tinyvllm.engine.scheduler", "tinyvllm/engine/scheduler.py")
sam_mod = load_module("sam_chunked_prefill_test", "tinyvllm/speculative/sam.py")
if dist is not None:
    dist.get_rank = lambda: 0
    dist.get_world_size = lambda: 1
    embed_head_mod = load_module("tinyvllm.layers.embed_head", "tinyvllm/layers/embed_head.py")
else:
    embed_head_mod = None

BlockManager = block_manager_mod.BlockManager
Sequence = sequence_mod.Sequence
SuffixAutomatonDraftIndex = sam_mod.SuffixAutomatonDraftIndex
SequenceStatus = sequence_mod.SequenceStatus
Scheduler = scheduler_mod.Scheduler
SamplingParams = sampling_mod.SamplingParams
ParallelLMHead = embed_head_mod.ParallelLMHead if embed_head_mod is not None else None
set_context = context_mod.set_context if context_mod is not None else None
reset_context_global = context_mod.reset_context if context_mod is not None else None


def test_slo_chunk_ladder_and_largest_safe_boundary():
    ladder = scheduler_mod.build_slo_chunk_ladder(128, 16)
    assert ladder == (128, 112, 96, 80, 64, 48, 32, 16)
    assert scheduler_mod.select_slo_chunk(
        remaining_slack_ns=16_800_000,
        cost_intercept_ns=4_000_000,
        cost_per_prefill_token_ns=100_000,
        token_ladder=ladder,
    ) == (128, 16_800_000)
    assert scheduler_mod.select_slo_chunk(
        remaining_slack_ns=15_200_000,
        cost_intercept_ns=4_000_000,
        cost_per_prefill_token_ns=100_000,
        token_ladder=ladder,
    ) == (112, 15_200_000)
    assert scheduler_mod.select_slo_chunk(
        remaining_slack_ns=5_599_999,
        cost_intercept_ns=4_000_000,
        cost_per_prefill_token_ns=100_000,
        token_ladder=ladder,
    ) == (None, None)


def make_config(**overrides):
    cfg = dict(
        max_num_seqs=4,
        max_num_batched_tokens=64,
        max_model_len=64,
        eos=-1,
        num_kvcache_blocks=32,
        kvcache_block_size=4,
        max_num_prefill_tokens_per_step=4,
        chunked_prefill_decode_first=False,
        chunked_prefill_max_consecutive_chunks=0,
    )
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


def reset_sequence_state(block_size: int = 4):
    Sequence.block_size = block_size


def make_seq(token_ids, max_tokens: int = 4):
    return Sequence(list(token_ids), SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=False))


def make_running(scheduler, token_ids=(90, 91, 92, 93), max_tokens=8):
    seq = make_seq(token_ids, max_tokens=max_tokens)
    scheduler.block_manager.allocate(seq)
    seq.append_token(max(token_ids) + 1)
    seq.status = SequenceStatus.RUNNING
    seq.num_computed_tokens = len(seq)
    scheduler.running.append(seq)
    return seq


def make_slo_scheduler(**overrides):
    config = {
        "max_num_seqs": 4,
        "max_num_batched_tokens": 128,
        "max_model_len": 512,
        "num_kvcache_blocks": 256,
        "kvcache_block_size": 4,
        "max_num_prefill_tokens_per_step": 128,
        "chunked_prefill_decode_first": False,
        "chunked_prefill_max_consecutive_chunks": 0,
        "chunked_prefill_slo_mixed": True,
        "chunked_prefill_slo_target_gap_ns": 64_000_000,
        "chunked_prefill_slo_reserve_ns": 8_000_000,
        "chunked_prefill_slo_cost_intercept_ns": 4_000_000,
        "chunked_prefill_slo_cost_per_prefill_token_ns": 100_000,
        "chunked_prefill_slo_min_chunk_tokens": 16,
    }
    config.update(overrides)
    return Scheduler(make_config(**config))


def make_running_with_id(
    scheduler,
    seq_id,
    token_ids=(90, 91, 92, 93),
    max_tokens=8,
):
    seq = make_running(
        scheduler,
        token_ids=token_ids,
        max_tokens=max_tokens,
    )
    seq.seq_id = seq_id
    return seq


def scheduler_prefill_digest(scheduler):
    return {
        "waiting_seq_ids": [seq.seq_id for seq in scheduler.waiting],
        "prefilling_seq_ids": [seq.seq_id for seq in scheduler.prefilling],
        "waiting_state": [
            (
                seq.seq_id,
                seq.status,
                tuple(seq.block_table),
                seq.num_cached_tokens,
                seq.num_computed_tokens,
                seq.prefill_chunk_start,
                seq.prefill_chunk_end,
                seq.prefill_chunk_final,
            )
            for seq in scheduler.waiting
        ],
        "prefilling_state": [
            (
                seq.seq_id,
                seq.status,
                tuple(seq.block_table),
                seq.num_cached_tokens,
                seq.num_computed_tokens,
                seq.prefill_chunk_start,
                seq.prefill_chunk_end,
                seq.prefill_chunk_final,
            )
            for seq in scheduler.prefilling
        ],
    }


def add_waiting(scheduler, count, prompt_tokens=12):
    rows = []
    for offset in range(count):
        seq = make_seq(range(offset * 100, offset * 100 + prompt_tokens))
        scheduler.add(seq)
        rows.append(seq)
    return rows


def assert_queue_contains(queue, seq):
    assert list(queue) == [seq]


def test_adaptive_state_requires_two_high_observations_and_resets_streak():
    reset_sequence_state()
    scheduler = Scheduler(make_config(
        chunked_prefill_adaptive_mixed=True,
        chunked_prefill_adaptive_enter_waiting=8,
        chunked_prefill_adaptive_exit_waiting=2,
        chunked_prefill_adaptive_transition_steps=2,
        chunked_prefill_adaptive_max_mixed_steps=2,
    ))
    make_running(scheduler)
    add_waiting(scheduler, 8)

    scheduler._update_adaptive_mixed_state(8)
    assert scheduler.adaptive_mixed_state == "inactive"
    assert scheduler.adaptive_high_streak == 1

    scheduler._update_adaptive_mixed_state(7)
    assert scheduler.adaptive_high_streak == 0

    scheduler._update_adaptive_mixed_state(8)
    scheduler._update_adaptive_mixed_state(8)
    assert scheduler.adaptive_mixed_state == "active"
    assert scheduler.adaptive_high_streak == 0


def test_adaptive_state_low_hysteresis_enters_inactive_or_draining():
    reset_sequence_state()
    scheduler = Scheduler(make_config(
        chunked_prefill_adaptive_mixed=True,
    ))
    make_running(scheduler)
    add_waiting(scheduler, 2)
    scheduler.adaptive_mixed_state = "active"

    scheduler._update_adaptive_mixed_state(2)
    assert scheduler.adaptive_mixed_state == "active"
    assert scheduler.adaptive_low_streak == 1
    scheduler._update_adaptive_mixed_state(2)
    assert scheduler.adaptive_mixed_state == "inactive"

    scheduler.adaptive_mixed_state = "active"
    prefilling = scheduler.waiting.popleft()
    scheduler.block_manager.allocate(prefilling)
    prefilling.status = SequenceStatus.PREFILLING
    scheduler.prefilling.append(prefilling)
    scheduler._update_adaptive_mixed_state(1)
    scheduler._update_adaptive_mixed_state(1)
    assert scheduler.adaptive_mixed_state == "draining"


def test_adaptive_ineligible_decision_clears_transition_streaks():
    reset_sequence_state()
    scheduler = Scheduler(make_config(
        chunked_prefill_adaptive_mixed=True,
    ))
    scheduler.adaptive_high_streak = 1
    scheduler.adaptive_low_streak = 1
    scheduler._update_adaptive_mixed_state(9)
    assert scheduler.adaptive_high_streak == 0
    assert scheduler.adaptive_low_streak == 0


def test_adaptive_observation_and_empty_reset_are_exact():
    reset_sequence_state()
    scheduler = Scheduler(make_config(
        chunked_prefill_adaptive_mixed=True,
    ))
    scheduler.adaptive_mixed_state = "draining"
    scheduler.adaptive_high_streak = 1
    scheduler.adaptive_low_streak = 0
    scheduler.adaptive_consecutive_mixed_steps = 2
    snapshot = scheduler.observation_snapshot()
    assert snapshot["adaptive_mixed_state"] == "draining"
    assert snapshot["adaptive_high_streak"] == 1
    assert snapshot["adaptive_low_streak"] == 0
    assert snapshot["adaptive_consecutive_mixed_steps"] == 2

    scheduler._maybe_reset_adaptive_mixed_controller()
    assert scheduler.adaptive_mixed_state == "inactive"
    assert scheduler.adaptive_high_streak == 0
    assert scheduler.adaptive_low_streak == 0
    assert scheduler.adaptive_consecutive_mixed_steps == 0


def test_adaptive_disabled_matches_decode_first_schedule_and_snapshot():
    reset_sequence_state()
    baseline = Scheduler(make_config(
        chunked_prefill_decode_first=True,
        chunked_prefill_adaptive_mixed=False,
    ))
    baseline_running = make_running(baseline)
    baseline_waiting = add_waiting(baseline, 8)

    baseline_result = baseline.schedule()
    assert baseline.last_policy_branch == "decode_first"
    baseline_metadata = {
        "scheduled_roles": [
            (
                seq is baseline_running,
                seq.prefill_chunk_start,
                seq.prefill_chunk_end,
                seq.prefill_chunk_final,
            )
            for seq in baseline_result[0]
        ],
        "is_prefill": baseline_result[1],
        "do_sample": baseline_result[2],
        "branch": baseline.last_policy_branch,
        "waiting": list(baseline.waiting),
        "prefilling": list(baseline.prefilling),
        "running": list(baseline.running),
        "free_blocks": list(baseline.block_manager.free_block_ids),
        "used_blocks": list(baseline.block_manager.used_block_ids),
    }

    reset_sequence_state()
    candidate = Scheduler(make_config(
        chunked_prefill_decode_first=True,
        chunked_prefill_adaptive_mixed=False,
    ))
    candidate_running = make_running(candidate)
    candidate_waiting = add_waiting(candidate, 8)
    candidate_result = candidate.schedule()
    candidate_metadata = {
        "scheduled_roles": [
            (
                seq is candidate_running,
                seq.prefill_chunk_start,
                seq.prefill_chunk_end,
                seq.prefill_chunk_final,
            )
            for seq in candidate_result[0]
        ],
        "is_prefill": candidate_result[1],
        "do_sample": candidate_result[2],
        "branch": candidate.last_policy_branch,
        "waiting": list(candidate.waiting),
        "prefilling": list(candidate.prefilling),
        "running": list(candidate.running),
        "free_blocks": list(candidate.block_manager.free_block_ids),
        "used_blocks": list(candidate.block_manager.used_block_ids),
    }

    assert baseline_metadata["scheduled_roles"] == (
        candidate_metadata["scheduled_roles"]
    )
    assert baseline_metadata["is_prefill"] == candidate_metadata["is_prefill"]
    assert baseline_metadata["do_sample"] == candidate_metadata["do_sample"]
    assert baseline_metadata["branch"] == candidate_metadata["branch"]
    assert len(baseline_metadata["waiting"]) == len(
        candidate_metadata["waiting"]
    ) == 8
    assert not baseline_metadata["prefilling"]
    assert not candidate_metadata["prefilling"]
    assert baseline_metadata["free_blocks"] == candidate_metadata["free_blocks"]
    assert baseline_metadata["used_blocks"] == candidate_metadata["used_blocks"]
    assert baseline_running.status == candidate_running.status
    assert len(baseline_waiting) == len(candidate_waiting)


def test_adaptive_second_high_observation_activates_and_mixes():
    reset_sequence_state()
    scheduler = Scheduler(make_config(
        chunked_prefill_decode_first=True,
        chunked_prefill_adaptive_mixed=True,
    ))
    running = make_running(scheduler)
    add_waiting(scheduler, 8)

    first = scheduler.schedule()
    assert first[0] == [running]
    assert scheduler.last_policy_branch == "adaptive_mixed_decode_first"
    assert scheduler.adaptive_high_streak == 1
    scheduler.postprocess(first[0], [100], first[1], first[2])

    second = scheduler.schedule()
    assert len(second) == 4
    assert second[3] == "mixed"
    assert scheduler.last_policy_branch == "adaptive_mixed_prefill_decode"
    assert scheduler.adaptive_mixed_state == "active"
    assert scheduler.adaptive_consecutive_mixed_steps == 1


def test_adaptive_two_mixed_steps_force_decode_yield():
    reset_sequence_state()
    scheduler = Scheduler(make_config(
        chunked_prefill_adaptive_mixed=True,
        chunked_prefill_adaptive_max_mixed_steps=2,
    ))
    make_running(scheduler, max_tokens=16)
    add_waiting(scheduler, 10)
    scheduler.adaptive_mixed_state = "active"

    first = scheduler.schedule()
    assert first[3] == "mixed"
    scheduler.postprocess(first[0], [101], first[1], first[2], first[3])
    second = scheduler.schedule()
    assert second[3] == "mixed"
    scheduler.postprocess(second[0], [102], second[1], second[2], second[3])
    third = scheduler.schedule()

    assert len(third) == 3
    assert third[1] is False
    assert scheduler.last_policy_branch == "adaptive_mixed_decode_yield"
    assert scheduler.adaptive_consecutive_mixed_steps == 0
    assert scheduler.adaptive_mixed_state == "active"


def test_adaptive_draining_never_admits_waiting_and_returns_inactive():
    reset_sequence_state()
    scheduler = Scheduler(make_config(
        chunked_prefill_adaptive_mixed=True,
    ))
    make_running(scheduler, max_tokens=16)
    prefilling = add_waiting(scheduler, 1)[0]
    scheduler.waiting.popleft()
    scheduler.block_manager.allocate(prefilling)
    prefilling.status = SequenceStatus.PREFILLING
    scheduler.prefilling.append(prefilling)
    waiting = add_waiting(scheduler, 1)[0]
    scheduler.adaptive_mixed_state = "draining"

    result = scheduler.schedule()
    assert result[3] == "mixed"
    assert result[0][0] is prefilling
    assert list(scheduler.waiting) == [waiting]
    assert scheduler.last_policy_branch == "adaptive_mixed_prefill_decode"


def test_adaptive_no_running_uses_chunked_prefill_without_fake_decode():
    reset_sequence_state()
    scheduler = Scheduler(make_config(
        chunked_prefill_adaptive_mixed=True,
    ))
    waiting = add_waiting(scheduler, 1)[0]
    result = scheduler.schedule()
    assert len(result) == 3
    assert result[0] == [waiting]
    assert scheduler.last_policy_branch == "adaptive_mixed_chunked_prefill"
    assert scheduler.adaptive_consecutive_mixed_steps == 0


def test_adaptive_required_mixed_failure_falls_back_to_decode():
    reset_sequence_state()
    scheduler = Scheduler(make_config(
        max_num_seqs=1,
        chunked_prefill_adaptive_mixed=True,
    ))
    running = make_running(scheduler)
    waiting = add_waiting(scheduler, 8)
    scheduler.adaptive_mixed_state = "active"
    queue_before = list(scheduler.waiting)
    result = scheduler.schedule()
    assert result[0] == [running]
    assert scheduler.last_policy_branch == "adaptive_mixed_decode_fallback"
    assert list(scheduler.waiting) == queue_before
    assert all(not seq.block_table for seq in waiting)


def test_intermediate_chunk_does_not_sample_or_append():
    reset_sequence_state()
    scheduler = Scheduler(make_config(max_num_prefill_tokens_per_step=4))
    seq = make_seq(range(10))
    scheduler.add(seq)

    seqs, is_prefill, do_sample = scheduler.schedule()

    assert seqs == [seq]
    assert is_prefill is True
    assert do_sample is False
    assert scheduler.last_policy_branch == "chunked_prefill"
    assert seq.prefill_chunk_start == 0
    assert seq.prefill_chunk_end == 4
    scheduler.postprocess(seqs, None, is_prefill, do_sample)

    assert len(seq) == 10
    assert seq.completion_token_ids == []
    assert seq.num_computed_tokens == 4
    assert seq.status == SequenceStatus.PREFILLING
    assert_queue_contains(scheduler.prefilling, seq)
    assert list(scheduler.running) == []


def test_final_chunk_samples_once_and_moves_to_running():
    reset_sequence_state()
    scheduler = Scheduler(make_config(max_num_prefill_tokens_per_step=4))
    seq = make_seq(range(10), max_tokens=3)
    scheduler.add(seq)

    seqs, is_prefill, do_sample = scheduler.schedule()
    scheduler.postprocess(seqs, None, is_prefill, do_sample)
    seqs, is_prefill, do_sample = scheduler.schedule()
    scheduler.postprocess(seqs, None, is_prefill, do_sample)
    seqs, is_prefill, do_sample = scheduler.schedule()

    assert seqs == [seq]
    assert is_prefill is True
    assert do_sample is True
    assert seq.prefill_chunk_start == 8
    assert seq.prefill_chunk_end == 10
    scheduler.postprocess(seqs, [99], is_prefill, do_sample)

    assert seq.completion_token_ids == [99]
    assert seq.num_computed_tokens == 10
    assert seq.status == SequenceStatus.RUNNING
    assert_queue_contains(scheduler.running, seq)
    assert list(scheduler.prefilling) == []


def test_chunked_prefill_batches_multiple_short_final_prompts():
    reset_sequence_state()
    scheduler = Scheduler(make_config(
        max_num_seqs=4,
        max_num_batched_tokens=16,
        max_num_prefill_tokens_per_step=4,
    ))
    seq_a = make_seq([1, 2, 3, 4])
    seq_b = make_seq([5, 6, 7, 8])
    scheduler.add(seq_a)
    scheduler.add(seq_b)

    seqs, is_prefill, do_sample = scheduler.schedule()

    assert seqs == [seq_a, seq_b]
    assert is_prefill is True
    assert do_sample is True
    assert seq_a.prefill_chunk_start == 0
    assert seq_a.prefill_chunk_end == 4
    assert seq_b.prefill_chunk_start == 0
    assert seq_b.prefill_chunk_end == 4
    scheduler.postprocess(seqs, [91, 92], is_prefill, do_sample)

    assert seq_a.completion_token_ids == [91]
    assert seq_b.completion_token_ids == [92]
    assert list(scheduler.running) == [seq_a, seq_b]


def test_chunked_prefill_batches_warm_prompt_by_uncached_tokens():
    reset_sequence_state()
    scheduler = Scheduler(make_config(
        max_num_seqs=2,
        max_num_batched_tokens=10,
        max_model_len=10,
        max_num_prefill_tokens_per_step=4,
    ))
    _publish_and_release(
        scheduler.block_manager,
        list(range(1, 9)),
    )
    cold = make_seq([21, 22, 23, 24], max_tokens=1)
    warm = make_seq(list(range(1, 10)), max_tokens=1)
    scheduler.add(cold)
    scheduler.add(warm)

    seqs, is_prefill, do_sample = scheduler.schedule()

    assert seqs == [cold, warm]
    assert is_prefill is True
    assert do_sample is True
    assert cold.prefill_chunk_start == 0
    assert cold.prefill_chunk_end == 4
    assert warm.num_cached_tokens == 8
    assert warm.prefill_chunk_start == 8
    assert warm.prefill_chunk_end == 9


def test_decode_first_prioritizes_existing_running_sequence():
    reset_sequence_state()
    scheduler = Scheduler(make_config(
        max_num_prefill_tokens_per_step=4,
        chunked_prefill_decode_first=True,
    ))
    running = make_seq([1, 2, 3])
    scheduler.block_manager.allocate(running)
    running.status = SequenceStatus.RUNNING
    running.num_computed_tokens = len(running)
    scheduler.running.append(running)
    waiting = make_seq(range(12))
    scheduler.add(waiting)

    seqs, is_prefill, do_sample = scheduler.schedule()

    assert seqs == [running]
    assert is_prefill is False
    assert do_sample is True
    assert list(scheduler.waiting) == [waiting]
    assert scheduler.last_policy_branch == "decode_first"


def test_scheduler_observation_snapshot_reports_queue_and_kv_state():
    reset_sequence_state()
    scheduler = Scheduler(make_config(
        num_kvcache_blocks=8,
        kvcache_block_size=4,
        chunked_prefill_max_consecutive_chunks=2,
    ))
    waiting = make_seq([1, 2, 3])
    prefilling = make_seq([4, 5, 6])
    running = make_seq([7, 8, 9])
    scheduler.add(waiting)
    scheduler.block_manager.allocate(prefilling)
    scheduler.block_manager.allocate(running)
    prefilling.status = SequenceStatus.PREFILLING
    running.status = SequenceStatus.RUNNING
    scheduler.prefilling.append(prefilling)
    scheduler.running.append(running)
    scheduler._consecutive_prefill_chunks = 2

    snapshot = scheduler.observation_snapshot()

    assert snapshot == {
        "waiting_seq_ids": [waiting.seq_id],
        "prefilling_seq_ids": [prefilling.seq_id],
        "running_seq_ids": [running.seq_id],
        "free_kv_blocks": 6,
        "used_kv_blocks": 2,
        "total_kv_blocks": 8,
        "kv_block_size_tokens": 4,
        "consecutive_prefill_chunks": 2,
        "adaptive_mixed_state": "inactive",
        "adaptive_high_streak": 0,
        "adaptive_low_streak": 0,
        "adaptive_consecutive_mixed_steps": 0,
    }


def test_model_runner_memory_snapshot_is_read_only_and_counts_all_kv_storage():
    memory_snapshot = load_class_method(
        "tinyvllm/engine/model_runner.py",
        "ModelRunner",
        "memory_snapshot",
    )
    cuda_calls = []
    original_cuda = getattr(torch, "cuda", None) if torch is not None else None

    class FakeCuda:
        @staticmethod
        def memory_allocated():
            cuda_calls.append("memory_allocated")
            return 101

        @staticmethod
        def memory_reserved():
            cuda_calls.append("memory_reserved")
            return 202

        @staticmethod
        def max_memory_allocated():
            cuda_calls.append("max_memory_allocated")
            return 303

        @staticmethod
        def max_memory_reserved():
            cuda_calls.append("max_memory_reserved")
            return 404

        @staticmethod
        def synchronize():
            raise AssertionError("memory snapshot must not synchronize")

        @staticmethod
        def empty_cache():
            raise AssertionError("memory snapshot must not empty cache")

        @staticmethod
        def reset_peak_memory_stats():
            raise AssertionError("memory snapshot must not reset peaks")

    class FakeTensor:
        def __init__(self, elements, element_size):
            self.elements = elements
            self.bytes_per_element = element_size

        def numel(self):
            return self.elements

        def element_size(self):
            return self.bytes_per_element

    fake_torch = SimpleNamespace(cuda=FakeCuda)
    memory_snapshot.__globals__["torch"] = fake_torch
    runner = SimpleNamespace(
        kv_cache=FakeTensor(10, 2),
        kv_scale=FakeTensor(3, 4),
        kv_zero=FakeTensor(2, 4),
    )

    snapshot = memory_snapshot(runner)

    assert snapshot == {
        "cuda_allocated_bytes": 101,
        "cuda_reserved_bytes": 202,
        "cuda_peak_allocated_bytes": 303,
        "cuda_peak_reserved_bytes": 404,
        "kv_capacity_bytes": 40,
    }
    assert cuda_calls == [
        "memory_allocated",
        "memory_reserved",
        "max_memory_allocated",
        "max_memory_reserved",
    ]
    if torch is not None:
        assert torch.cuda is original_cuda


def test_model_runner_hybrid_prefix_cache_snapshot_is_rank_local():
    snapshot = load_class_method(
        "tinyvllm/engine/model_runner.py",
        "ModelRunner",
        "qwen35_hybrid_prefix_cache_snapshot",
    )

    class FakeCache:
        def observation_snapshot(self):
            return {
                "current_entries": 2,
                "current_bytes": 100,
                "current_logical_bytes": 150,
                "deduplicated_bytes": 50,
                "peak_entries": 3,
                "peak_bytes": 200,
                "publishes": 4,
                "hits": 5,
                "misses": 1,
                "evictions": 0,
                "validation_failures": 0,
                "failed_restores": 0,
                "current_interned_tensors": 4,
            }

    runner = SimpleNamespace(
        rank=2,
        qwen35_hybrid_prefix_restore_owner=SimpleNamespace(
            snapshot_cache=FakeCache(),
            representation="int8",
            representation_version="v1",
            codec="symmetric-per-token",
        ),
    )

    result = snapshot(runner)

    assert result["rank"] == 2
    assert result["representation"] == "int8"
    assert result["representation_version"] == "v1"
    assert result["codec"] == "symmetric-per-token"
    assert result["publishes"] == 4
    assert result["current_entries"] == 2
    assert result["current_bytes"] == 100
    assert result["current_logical_bytes"] == 150
    assert result["deduplicated_bytes"] == 50
    assert result["peak_entries"] == 3
    assert result["peak_bytes"] == 200
    assert result["hits"] == 5
    assert result["misses"] == 1
    assert result["evictions"] == 0
    assert result["validation_failures"] == 0
    assert result["failed_restores"] == 0


def test_engine_collects_all_rank_hybrid_prefix_cache_snapshots():
    snapshot = load_class_method(
        "tinyvllm/engine/llm_engine.py",
        "LLMEngine",
        "qwen35_hybrid_prefix_cache_snapshots",
    )
    local = {"rank": 0, "current_entries": 1}
    worker_rows = [
        SimpleNamespace(
            rank=rank,
            result={"rank": rank, "current_entries": 1},
        )
        for rank in (1, 2, 3)
    ]
    engine = SimpleNamespace(
        call_model_runner_acknowledged=lambda *args, **kwargs: (
            local,
            worker_rows,
        ),
        model_runner=SimpleNamespace(world_size=4),
    )

    assert snapshot(engine, timeout_s=12.0) == (
        local,
        worker_rows[0].result,
        worker_rows[1].result,
        worker_rows[2].result,
    )


def test_llm_engine_step_records_observation_without_changing_return_value():
    step = load_class_method(
        "tinyvllm/engine/llm_engine.py",
        "LLMEngine",
        "step",
    )

    class FakeSequence:
        def __init__(self):
            self.seq_id = 17
            self.status = SequenceStatus.RUNNING
            self.prefill_chunk_start = 0
            self.prefill_chunk_end = 3
            self.prefill_chunk_final = True
            self.step_is_decode = False
            self.step_do_sample = True
            self.completion_token_ids = []

        @property
        def is_finished(self):
            return self.status == SequenceStatus.FINISHED

    seq = FakeSequence()

    class FakeScheduler:
        last_policy_branch = "chunked_prefill"
        last_speculative_selection = None
        schedule_generation = 1

        def __init__(self):
            self.snapshot_index = 0
            self.timing = {}

        def observation_snapshot(self):
            self.snapshot_index += 1
            return {
                "snapshot_index": self.snapshot_index,
                "adaptive_mixed_state": "active",
                "adaptive_high_streak": 0,
                "adaptive_low_streak": 1,
                "adaptive_consecutive_mixed_steps": 2,
            }

        def schedule(self, decision_now_ns):
            assert decision_now_ns == 10
            return [seq], True, True

        def postprocess(
            self,
            seqs,
            token_ids,
            is_prefill,
            do_sample,
            batch_kind,
            *,
            decision_now_ns,
            step_end_ns,
        ):
            assert seqs == [seq]
            assert token_ids == [91]
            self.timing = {
                "decision_now_ns": decision_now_ns,
                "step_end_ns": step_end_ns,
                "actual_step_duration_ns": step_end_ns - decision_now_ns,
            }
            seq.completion_token_ids.append(91)
            seq.prefill_chunk_start = 3
            seq.prefill_chunk_end = 4
            seq.prefill_chunk_final = False
            seq.status = SequenceStatus.FINISHED

        def last_slo_observation(self):
            return dict(self.timing)

    class FakeModelRunner:
        def call(self, method_name, *args):
            assert method_name == "run"
            return [91]

        def memory_snapshot(self):
            return {"cuda_allocated_bytes": 123}

    engine = SimpleNamespace(
        _clock_ns=IntegerClock([10, 20]),
        scheduler=FakeScheduler(),
        model_runner=FakeModelRunner(),
        last_batch_kind=None,
        last_scheduled_seqs=[],
        last_step_observation=None,
    )

    result = step(engine)

    assert result == ([(17, [91])], 3)
    assert engine.last_step_observation == {
        "policy_branch": "chunked_prefill",
        "batch_kind": None,
        "is_prefill": True,
        "do_sample": True,
        "speculative_schedule_generation": 1,
        "speculative_selected_seq_ids": [],
        "speculative_suppressed_seq_ids": [17],
        "scheduled": [{
            "seq_id": 17,
            "is_decode": False,
            "do_sample": True,
            "prefill_chunk_start": 0,
            "prefill_chunk_end": 3,
            "prefill_chunk_final": True,
        }],
        "queue_before": {
            "snapshot_index": 1,
            "adaptive_mixed_state": "active",
            "adaptive_high_streak": 0,
            "adaptive_low_streak": 1,
            "adaptive_consecutive_mixed_steps": 2,
        },
        "queue_after": {
            "snapshot_index": 2,
            "adaptive_mixed_state": "active",
            "adaptive_high_streak": 0,
            "adaptive_low_streak": 1,
            "adaptive_consecutive_mixed_steps": 2,
        },
        "new_completion_tokens_by_seq": {17: [91]},
        "finished_seq_ids": [17],
        "speculative_output_token_counts": {},
        "speculative_accepted_draft_token_counts": {},
        "speculative_proposal_token_counts": {},
        "speculative_proposal_token_ids_by_seq": {},
        "speculative_accepted_draft_token_ids_by_seq": {},
        "speculative_proposal_row_count": 0,
        "speculative_first_target_callback_count": 0,
        "speculative_fixed_q_group_count": 0,
        "speculative_runtime_timing_ms": {},
        "exact_greedy_decode_burst_attempted": False,
        "exact_greedy_decode_burst_accepted": False,
        "exact_greedy_decode_burst_width": 0,
        "exact_greedy_decode_burst_lease_identity_sha256": None,
        "exact_greedy_decode_burst_result_identity_sha256": None,
        "exact_greedy_decode_burst_graph_identity_sha256": None,
        "exact_greedy_decode_burst_replay_count": 0,
        "exact_greedy_decode_burst_token_d2h_calls": 0,
        "exact_greedy_decode_burst_sampled_logit_d2h_calls": 0,
        "exact_greedy_decode_burst_sampled_logits": [],
        "exact_greedy_decode_burst_correctness_trace": False,
        "exact_greedy_decode_burst_host_visible_gap_ns": 0,
        "exact_greedy_decode_burst_fallback_reason": None,
        "exact_greedy_decode_burst_quarantine_reason": None,
        "exact_greedy_decode_burst_pending_lease_count": 0,
        "memory": {"cuda_allocated_bytes": 123},
        "decision_now_ns": 10,
        "step_end_ns": 20,
        "actual_step_duration_ns": 10,
    }


class IntegerClock:
    def __init__(self, values):
        self.values = iter(values)
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return next(self.values)


def test_engine_samples_one_decision_and_one_step_end_timestamp():
    step = load_class_method(
        "tinyvllm/engine/llm_engine.py",
        "LLMEngine",
        "step",
    )

    class FakeTimedScheduler:
        last_policy_branch = "legacy_decode"
        last_speculative_selection = None
        schedule_generation = 1

        def __init__(self):
            self.schedule_calls = []
            self.postprocess_calls = []

        def observation_snapshot(self):
            return {}

        def schedule(self, decision_now_ns):
            self.schedule_calls.append(decision_now_ns)
            return [], False, True

        def postprocess(
            self,
            seqs,
            token_ids,
            is_prefill,
            do_sample,
            batch_kind,
            *,
            decision_now_ns,
            step_end_ns,
        ):
            assert seqs == []
            assert token_ids == []
            assert is_prefill is False
            assert do_sample is True
            assert batch_kind is None
            self.postprocess_calls.append({
                "decision_now_ns": decision_now_ns,
                "step_end_ns": step_end_ns,
            })

        def last_slo_observation(self):
            return {
                "decision_now_ns": self.postprocess_calls[-1][
                    "decision_now_ns"
                ],
                "step_end_ns": self.postprocess_calls[-1]["step_end_ns"],
                "actual_step_duration_ns": (
                    self.postprocess_calls[-1]["step_end_ns"]
                    - self.postprocess_calls[-1]["decision_now_ns"]
                ),
            }

    class FakeTimedModelRunner:
        def call(self, method_name, *args):
            assert method_name == "run"
            return []

        def memory_snapshot(self):
            return {}

    engine = SimpleNamespace(
        _clock_ns=IntegerClock([100, 175]),
        scheduler=FakeTimedScheduler(),
        model_runner=FakeTimedModelRunner(),
        last_batch_kind=None,
        last_scheduled_seqs=[],
        last_step_observation=None,
    )

    outputs, _ = step(engine)

    assert outputs == []
    assert engine._clock_ns.calls == 2
    assert engine.scheduler.schedule_calls == [100]
    assert engine.scheduler.postprocess_calls == [{
        "decision_now_ns": 100,
        "step_end_ns": 175,
    }]
    assert engine.last_step_observation["decision_now_ns"] == 100
    assert engine.last_step_observation["step_end_ns"] == 175
    assert engine.last_step_observation["actual_step_duration_ns"] == 75


def test_p5_decision_snapshot_is_immutable_until_postprocess_copy():
    scheduler = make_slo_scheduler()
    scheduler._publish_slo_decision({
        "decision_now_ns": 100,
        "suppression_reason": "inactive",
    })
    snapshot = scheduler.last_slo_decision
    try:
        snapshot["decision_now_ns"] = 200
    except TypeError:
        pass
    else:
        raise AssertionError("P5 decision snapshot is mutable")


def test_decode_progress_updates_only_for_completion_tokens():
    reset_sequence_state()
    scheduler = make_slo_scheduler()
    first = make_seq([1, 2, 3, 4], max_tokens=2)
    first.seq_id = 1
    scheduler.block_manager.allocate(first)
    first.status = SequenceStatus.PREFILLING
    first.prefill_chunk_start = 0
    first.prefill_chunk_end = 4
    first.prefill_chunk_final = True
    scheduler._postprocess_chunked_prefill(
        [first],
        [101],
        True,
        step_end_ns=1_000,
    )
    assert scheduler.decode_progress_ns_by_seq_id == {1: 1_000}

    intermediate = make_seq(range(8), max_tokens=4)
    intermediate.seq_id = 2
    scheduler.block_manager.allocate(intermediate)
    intermediate.status = SequenceStatus.PREFILLING
    intermediate.prefill_chunk_start = 0
    intermediate.prefill_chunk_end = 4
    intermediate.prefill_chunk_final = False
    scheduler._postprocess_chunked_prefill(
        [intermediate],
        None,
        False,
        step_end_ns=1_100,
    )
    assert 2 not in scheduler.decode_progress_ns_by_seq_id

    scheduler.running.remove(first)
    first.step_is_decode = True
    scheduler._postprocess_mixed(
        [first],
        [102],
        step_end_ns=1_200,
    )
    assert first.is_finished
    assert 1 not in scheduler.decode_progress_ns_by_seq_id

    scheduler.prefilling.remove(intermediate)
    scheduler.block_manager.deallocate(intermediate)
    scheduler._maybe_reset_adaptive_mixed_controller()
    assert scheduler.decode_progress_ns_by_seq_id == {}
    assert scheduler._last_slo_decision_now_ns is None


def test_progress_survives_preemption_but_is_excluded_until_running():
    reset_sequence_state()
    scheduler = make_slo_scheduler()
    seq = make_running_with_id(scheduler, seq_id=7)
    scheduler.decode_progress_ns_by_seq_id[7] = 1_000
    scheduler.running.remove(seq)
    scheduler.preempt(seq)
    assert scheduler.decode_progress_ns_by_seq_id[7] == 1_000
    assert scheduler._oldest_runnable_decode(2_000) is None
    scheduler.waiting.remove(seq)
    scheduler.block_manager.allocate(seq)
    seq.status = SequenceStatus.RUNNING
    scheduler.running.append(seq)
    assert scheduler._oldest_runnable_decode(2_000) == (7, 1_000, 1_000)


def test_clock_regression_is_sticky_and_forces_decode_only():
    reset_sequence_state()
    scheduler = make_slo_scheduler()
    running = make_running_with_id(scheduler, seq_id=5, max_tokens=16)
    scheduler.decode_progress_ns_by_seq_id[running.seq_id] = 1_000

    first = scheduler.schedule(1_000)
    scheduler.postprocess(
        first[0],
        [101],
        first[1],
        first[2],
        decision_now_ns=1_000,
        step_end_ns=1_100,
    )
    assert scheduler.decode_progress_ns_by_seq_id[running.seq_id] == 1_100
    second = scheduler.schedule(999)
    scheduler.postprocess(
        second[0],
        [102],
        second[1],
        second[2],
        decision_now_ns=999,
        step_end_ns=1_200,
    )

    assert scheduler.slo_clock_invalid is True
    assert scheduler.slo_clock_invalid_reason == "decision_clock_regressed"
    assert scheduler.last_policy_branch == "slo_mixed_clock_invalid_decode"
    scheduler.schedule(2_000)
    assert scheduler.last_policy_branch == "slo_mixed_clock_invalid_decode"


def test_missing_runnable_progress_fails_closed():
    reset_sequence_state()
    scheduler = make_slo_scheduler()
    make_running_with_id(scheduler, seq_id=9, max_tokens=16)
    add_waiting(scheduler, 8)

    scheduler.schedule(10_000)

    assert scheduler.last_policy_branch == "slo_mixed_missing_progress_decode"
    assert scheduler.last_slo_decision["suppression_reason"] == (
        "missing_decode_progress"
    )


def test_active_demand_never_overrides_no_slack():
    reset_sequence_state()
    scheduler = make_slo_scheduler()
    running = make_running_with_id(scheduler, seq_id=1, max_tokens=16)
    scheduler.decode_progress_ns_by_seq_id[running.seq_id] = 1_000
    add_waiting(scheduler, 8, prompt_tokens=32)
    scheduler.adaptive_mixed_state = "active"
    before = scheduler_prefill_digest(scheduler)

    scheduler.schedule(56_001_001)

    assert scheduler.last_policy_branch == "slo_mixed_no_slack_decode"
    assert scheduler.last_slo_decision["remaining_slack_ns"] == -1
    assert scheduler.last_slo_decision["selected_chunk_tokens"] is None
    assert scheduler_prefill_digest(scheduler) == before


def test_largest_safe_chunk_is_selected_with_exact_integer_math():
    reset_sequence_state()
    scheduler = make_slo_scheduler()
    running = make_running_with_id(scheduler, seq_id=1, max_tokens=16)
    scheduler.decode_progress_ns_by_seq_id[running.seq_id] = 1_000
    add_waiting(scheduler, 8, prompt_tokens=128)
    scheduler.adaptive_mixed_state = "active"

    scheduler.schedule(41_800_000)

    decision = scheduler.last_slo_decision
    assert decision["oldest_decode_age_ns"] == 41_799_000
    assert decision["remaining_slack_ns"] == 14_201_000
    assert decision["candidate_chunk_tokens"] == [
        128, 112, 96, 80, 64, 48, 32, 16
    ]
    assert decision["selected_chunk_tokens"] == 96
    assert decision["predicted_step_ns"] == 13_600_000
    assert set(decision) == {
        "decision_now_ns",
        "target_gap_ns",
        "reserve_ns",
        "oldest_decode_seq_id",
        "oldest_decode_progress_ns",
        "oldest_decode_age_ns",
        "remaining_slack_ns",
        "cost_intercept_ns",
        "cost_per_prefill_token_ns",
        "candidate_chunk_tokens",
        "predicted_step_ns",
        "selected_chunk_tokens",
        "actual_prefill_tokens",
        "scheduled_decode_seq_ids",
        "demand_state_before",
        "demand_state_after",
        "suppression_reason",
        "clock_invalid",
        "clock_invalid_reason",
    }
    try:
        decision["selected_chunk_tokens"] = 128
    except TypeError:
        pass
    else:
        raise AssertionError("published P5 decision is mutable")


def test_mixed_batch_contains_exact_oldest_runnable_decode_row():
    reset_sequence_state()
    scheduler = make_slo_scheduler()
    older = make_running_with_id(scheduler, seq_id=11, max_tokens=16)
    younger = make_running_with_id(
        scheduler,
        seq_id=12,
        token_ids=(80, 81, 82, 83),
        max_tokens=16,
    )
    scheduler.running.clear()
    scheduler.running.extend([younger, older])
    scheduler.decode_progress_ns_by_seq_id.update({
        11: 1_000,
        12: 5_000,
    })
    add_waiting(scheduler, 8, prompt_tokens=64)
    scheduler.adaptive_mixed_state = "active"

    scheduled = scheduler.schedule(10_000_000)

    seqs = scheduled[0]
    decode_ids = [
        seq.seq_id for seq in seqs if seq.step_is_decode
    ]
    assert 11 in decode_ids
    assert scheduler.last_slo_decision["oldest_decode_seq_id"] == 11
    assert scheduler.last_slo_decision["scheduled_decode_seq_ids"] == (
        decode_ids
    )


def test_protected_row_reservation_failure_does_not_substitute_younger():
    reset_sequence_state()
    scheduler = make_slo_scheduler(num_kvcache_blocks=4)
    older = make_running_with_id(scheduler, seq_id=21, max_tokens=16)
    younger = make_running_with_id(
        scheduler,
        seq_id=22,
        token_ids=(80, 81, 82),
        max_tokens=16,
    )
    scheduler.running.clear()
    scheduler.running.extend([older, younger])
    scheduler.decode_progress_ns_by_seq_id.update({
        21: 1_000,
        22: 2_000,
    })
    add_waiting(scheduler, 1, prompt_tokens=8)
    scheduler.adaptive_mixed_state = "active"
    before = scheduler_prefill_digest(scheduler)

    scheduler.schedule(10_000_000)

    assert scheduler.last_policy_branch == (
        "slo_mixed_transaction_fallback_decode"
    )
    assert scheduler.last_slo_decision["selected_chunk_tokens"] is not None
    assert scheduler.last_slo_decision["actual_prefill_tokens"] == 0
    assert scheduler_prefill_digest(scheduler) == before


def test_p0_p3_p4_scheduling_is_unchanged_by_p5_support():
    reset_sequence_state()
    p0 = Scheduler(make_config(max_num_prefill_tokens_per_step=0))
    p0_waiting = add_waiting(p0, 1, prompt_tokens=4)[0]
    p0_result = p0.schedule()
    assert (
        [seq is p0_waiting for seq in p0_result[0]],
        p0_result[1:],
        p0.last_policy_branch,
        len(p0.waiting),
        len(p0.prefilling),
        [seq is p0_waiting for seq in p0.running],
    ) == (
        [True],
        (True, True),
        "legacy_prefill",
        0,
        0,
        [True],
    )

    reset_sequence_state()
    p3 = Scheduler(make_config(chunked_prefill_mixed_batch=True))
    p3_running = make_running(p3, max_tokens=16)
    p3_waiting = add_waiting(p3, 1, prompt_tokens=8)[0]
    p3_result = p3.schedule()
    assert (
        [
            "waiting" if seq is p3_waiting else "running"
            for seq in p3_result[0]
        ],
        p3_result[1:],
        p3.last_policy_branch,
        len(p3.waiting),
        len(p3.prefilling),
        len(p3.running),
    ) == (
        ["waiting", "running"],
        (True, True, "mixed"),
        "mixed_prefill_decode",
        0,
        0,
        0,
    )

    reset_sequence_state()
    p4 = Scheduler(make_config(chunked_prefill_adaptive_mixed=True))
    p4_running = make_running(p4, max_tokens=16)
    add_waiting(p4, 8)
    p4.adaptive_mixed_state = "active"
    p4_result = p4.schedule()
    assert (
        p4_result[0][-1] is p4_running,
        p4_result[1:],
        p4.last_policy_branch,
        len(p4.waiting),
        len(p4.prefilling),
        len(p4.running),
    ) == (
        True,
        (True, True, "mixed"),
        "adaptive_mixed_prefill_decode",
        7,
        0,
        0,
    )


def test_chunked_prefill_decode_fallback_reports_branch_without_changing_result():
    reset_sequence_state()
    scheduler = Scheduler(make_config(
        max_num_prefill_tokens_per_step=4,
        chunked_prefill_decode_first=False,
    ))
    running = make_seq([1, 2, 3], max_tokens=4)
    scheduler.block_manager.allocate(running)
    running.status = SequenceStatus.RUNNING
    running.num_computed_tokens = len(running)
    scheduler.running.append(running)

    seqs, is_prefill, do_sample = scheduler.schedule()

    assert seqs == [running]
    assert is_prefill is False
    assert do_sample is True
    assert list(scheduler.running) == [running]
    assert scheduler.last_policy_branch == "decode_fallback"


def test_legacy_prefill_reports_branch_without_changing_result():
    reset_sequence_state()
    scheduler = Scheduler(make_config(
        max_num_prefill_tokens_per_step=0,
    ))
    waiting = make_seq([1, 2, 3], max_tokens=4)
    scheduler.add(waiting)

    seqs, is_prefill, do_sample = scheduler.schedule()

    assert seqs == [waiting]
    assert is_prefill is True
    assert do_sample is True
    assert list(scheduler.waiting) == []
    assert list(scheduler.running) == [waiting]
    assert scheduler.last_policy_branch == "legacy_prefill"


def test_legacy_decode_reports_branch_without_changing_result():
    reset_sequence_state()
    scheduler = Scheduler(make_config(
        max_num_prefill_tokens_per_step=0,
    ))
    running = make_seq([1, 2, 3], max_tokens=4)
    scheduler.block_manager.allocate(running)
    running.status = SequenceStatus.RUNNING
    running.num_computed_tokens = len(running)
    scheduler.running.append(running)

    seqs, is_prefill, do_sample = scheduler.schedule()

    assert seqs == [running]
    assert is_prefill is False
    assert do_sample is True
    assert list(scheduler.running) == [running]
    assert scheduler.last_policy_branch == "legacy_decode"


def test_add_rejects_request_beyond_max_model_len():
    reset_sequence_state()
    scheduler = Scheduler(make_config(max_model_len=8))
    seq = make_seq(range(8), max_tokens=1)

    try:
        scheduler.add(seq)
    except ValueError as exc:
        message = str(exc)
    else:
        assert False, "expected max_model_len admission failure"

    assert "max_model_len" in message
    assert "prompt_tokens=8" in message
    assert "max_tokens=1" in message
    assert list(scheduler.waiting) == []


def test_add_rejects_prompt_beyond_logical_kv_capacity():
    reset_sequence_state()
    scheduler = Scheduler(make_config(
        max_model_len=64,
        num_kvcache_blocks=2,
        kvcache_block_size=4,
    ))
    seq = make_seq(range(9), max_tokens=1)

    try:
        scheduler.add(seq)
    except ValueError as exc:
        message = str(exc)
    else:
        assert False, "expected KV capacity admission failure"

    assert "KV cache capacity" in message
    assert "required_blocks=3" in message
    assert "available_blocks=2" in message
    assert list(scheduler.waiting) == []


def test_add_accounts_for_decode_kv_capacity_boundary():
    reset_sequence_state()
    scheduler = Scheduler(make_config(
        max_model_len=64,
        num_kvcache_blocks=2,
        kvcache_block_size=4,
    ))
    fits = make_seq(range(7), max_tokens=2)
    scheduler.add(fits)
    assert list(scheduler.waiting) == [fits]

    too_many_decode_tokens = make_seq(range(7), max_tokens=3)
    try:
        scheduler.add(too_many_decode_tokens)
    except ValueError as exc:
        message = str(exc)
    else:
        assert False, "expected decode KV capacity admission failure"

    assert "KV cache capacity" in message
    assert "kv_tokens=9" in message
    assert "required_blocks=3" in message
    assert list(scheduler.waiting) == [fits]


def test_add_capacity_boundary_for_multiple_logical_block_counts():
    for num_blocks in (1, 2, 4):
        reset_sequence_state()
        scheduler = Scheduler(make_config(
            max_model_len=64,
            num_kvcache_blocks=num_blocks,
            kvcache_block_size=4,
        ))
        capacity_tokens = num_blocks * 4
        fits = make_seq(range(capacity_tokens), max_tokens=1)
        scheduler.add(fits)
        assert list(scheduler.waiting) == [fits]

        too_long = make_seq(range(capacity_tokens + 1), max_tokens=1)
        try:
            scheduler.add(too_long)
        except ValueError as exc:
            message = str(exc)
        else:
            assert False, f"expected KV capacity failure for num_blocks={num_blocks}"

        assert "KV cache capacity" in message
        assert f"available_blocks={num_blocks}" in message
        assert list(scheduler.waiting) == [fits]


def test_chunked_prefill_progresses_with_varied_chunk_sizes():
    for chunk_tokens in (1, 2, 4):
        reset_sequence_state()
        scheduler = Scheduler(make_config(max_num_prefill_tokens_per_step=chunk_tokens))
        prompt_tokens = 9
        seq = make_seq(range(prompt_tokens), max_tokens=2)
        scheduler.add(seq)

        expected_start = 0
        while expected_start < prompt_tokens:
            seqs, is_prefill, do_sample = scheduler.schedule()
            expected_end = min(expected_start + chunk_tokens, prompt_tokens)

            assert seqs == [seq]
            assert is_prefill is True
            assert seq.prefill_chunk_start == expected_start
            assert seq.prefill_chunk_end == expected_end
            assert do_sample is (expected_end == prompt_tokens)
            scheduler.postprocess(seqs, [700 + chunk_tokens] if do_sample else None, is_prefill, do_sample)
            expected_start = expected_end

        assert seq.completion_token_ids == [700 + chunk_tokens]
        assert seq.num_computed_tokens == prompt_tokens
        assert list(scheduler.prefilling) == []
        assert list(scheduler.running) == [seq]


def test_short_prefill_batch_respects_sequence_and_token_limits():
    reset_sequence_state()
    scheduler = Scheduler(make_config(
        max_num_seqs=2,
        max_num_batched_tokens=8,
        max_num_prefill_tokens_per_step=4,
    ))
    seq_a = make_seq([1, 2, 3, 4], max_tokens=2)
    seq_b = make_seq([5, 6, 7, 8], max_tokens=2)
    seq_c = make_seq([9, 10, 11, 12], max_tokens=2)
    for seq in (seq_a, seq_b, seq_c):
        scheduler.add(seq)

    seqs, is_prefill, do_sample = scheduler.schedule()

    assert seqs == [seq_a, seq_b]
    assert is_prefill is True
    assert do_sample is True
    scheduler.postprocess(seqs, [101, 102], is_prefill, do_sample)
    assert seq_a.completion_token_ids == [101]
    assert seq_b.completion_token_ids == [102]
    assert list(scheduler.waiting) == [seq_c]
    assert list(scheduler.running) == [seq_a, seq_b]


def test_chunked_prefill_does_not_publish_future_block_hashes():
    reset_sequence_state()
    scheduler = Scheduler(make_config(max_num_prefill_tokens_per_step=4))
    seq = make_seq([1, 2, 3, 4, 5, 6, 7, 8])
    scheduler.add(seq)

    seqs, is_prefill, do_sample = scheduler.schedule()
    h0 = scheduler.block_manager.compute_hash(seq.block(0), -1)
    h1 = scheduler.block_manager.compute_hash(seq.block(1), h0)

    assert h0 not in scheduler.block_manager.hash_to_block_id
    assert h1 not in scheduler.block_manager.hash_to_block_id

    scheduler.postprocess(seqs, None, is_prefill, do_sample)

    assert h0 in scheduler.block_manager.hash_to_block_id
    assert h1 not in scheduler.block_manager.hash_to_block_id

    seqs, is_prefill, do_sample = scheduler.schedule()
    scheduler.postprocess(seqs, [42], is_prefill, do_sample)

    assert h1 in scheduler.block_manager.hash_to_block_id
    assert seq.completion_token_ids == [42]


def test_chunked_prefill_restores_reused_cached_block_metadata():
    reset_sequence_state()
    scheduler = Scheduler(make_config(max_num_prefill_tokens_per_step=4))
    cached = make_seq([1, 2, 3, 4])
    scheduler.block_manager.allocate(cached)
    h0 = scheduler.block_manager.compute_hash(cached.block(0), -1)
    block_id = cached.block_table[0]
    scheduler.block_manager.deallocate(cached)

    seq = make_seq([1, 2, 3, 4, 5, 6, 7, 8])
    scheduler.add(seq)
    seqs, is_prefill, do_sample = scheduler.schedule()

    assert seq.num_cached_tokens == 4
    assert seq.num_computed_tokens == 4
    assert seq.block_table[0] == block_id
    assert scheduler.block_manager.blocks[block_id].hash == h0
    assert scheduler.block_manager.blocks[block_id].token_ids == [1, 2, 3, 4]

    scheduler.postprocess(seqs, [77], is_prefill, do_sample)
    h1 = scheduler.block_manager.compute_hash(seq.block(1), h0)

    assert h1 in scheduler.block_manager.hash_to_block_id
    assert seq.completion_token_ids == [77]


def _publish_and_release(block_manager, token_ids):
    seq = make_seq(token_ids, max_tokens=1)
    block_manager.allocate(
        seq,
        publish_hashes=False,
        max_cached_tokens=0,
    )
    block_manager.commit_prefill(seq, 0, len(seq))
    block_table = list(seq.block_table)
    block_manager.deallocate(seq)
    return block_table


def test_max_reusable_tokens_keeps_one_sampleable_token():
    reset_sequence_state()
    block_manager = BlockManager(num_blocks=16, block_size=4)

    expected = {
        3: 0,
        4: 0,
        5: 4,
        8: 4,
        9: 8,
    }
    for prompt_tokens, reusable_tokens in expected.items():
        seq = make_seq(range(prompt_tokens), max_tokens=1)
        assert block_manager.max_reusable_tokens(seq) == reusable_tokens


def test_allocate_caps_exact_block_aligned_cache_hit():
    reset_sequence_state()
    block_manager = BlockManager(num_blocks=16, block_size=4)
    cached_blocks = _publish_and_release(
        block_manager,
        [1, 2, 3, 4],
    )

    seq = make_seq([1, 2, 3, 4], max_tokens=1)
    block_manager.allocate(
        seq,
        publish_hashes=False,
        max_cached_tokens=block_manager.max_reusable_tokens(seq),
    )

    assert seq.num_cached_tokens == 0
    assert seq.num_computed_tokens == 0
    assert seq.block_table[0] != cached_blocks[0]


def test_allocate_reuses_only_blocks_before_sampleable_suffix():
    reset_sequence_state()
    block_manager = BlockManager(num_blocks=16, block_size=4)
    cached_blocks = _publish_and_release(
        block_manager,
        list(range(1, 9)),
    )

    seq = make_seq(list(range(1, 9)), max_tokens=1)
    block_manager.allocate(
        seq,
        publish_hashes=False,
        max_cached_tokens=block_manager.max_reusable_tokens(seq),
    )

    assert seq.num_cached_tokens == 4
    assert seq.num_computed_tokens == 4
    assert seq.block_table[0] == cached_blocks[0]
    assert seq.block_table[1] != cached_blocks[1]


def test_can_allocate_excludes_live_prefix_hits_from_free_block_requirement():
    reset_sequence_state()
    block_manager = BlockManager(num_blocks=3, block_size=4)
    live = make_seq(list(range(1, 9)), max_tokens=2)
    block_manager.allocate(live, publish_hashes=False, max_cached_tokens=0)
    block_manager.commit_prefill(live, 0, len(live))

    warm = make_seq(list(range(1, 10)), max_tokens=1)
    assert len(block_manager.free_block_ids) == 1
    assert warm.num_blocks == 3
    assert block_manager.max_reusable_tokens(warm) == 8
    assert block_manager.can_allocate(warm) is True
    block_manager.allocate(
        warm,
        publish_hashes=False,
        max_cached_tokens=block_manager.max_reusable_tokens(warm),
    )
    assert warm.num_cached_tokens == 8
    assert len(block_manager.free_block_ids) == 0


def test_can_allocate_counts_idle_prefix_hits_as_free_block_requirement():
    reset_sequence_state()
    block_manager = BlockManager(num_blocks=3, block_size=4)
    _publish_and_release(block_manager, list(range(1, 9)))

    unrelated_live = make_seq([21, 22, 23, 24], max_tokens=2)
    block_manager.allocate(
        unrelated_live,
        publish_hashes=False,
        max_cached_tokens=0,
    )

    warm = make_seq(list(range(1, 10)), max_tokens=1)
    assert len(block_manager.free_block_ids) == 2
    assert warm.num_blocks == 3
    assert block_manager.can_allocate(warm) is False


def test_estimate_admission_is_read_only_for_live_and_idle_hits():
    reset_sequence_state()
    block_manager = BlockManager(num_blocks=4, block_size=4)
    source = make_seq(list(range(1, 9)), max_tokens=2)
    block_manager.allocate(source, publish_hashes=False, max_cached_tokens=0)
    block_manager.commit_prefill(source, 0, len(source))
    source_blocks = list(source.block_table)

    live_sharer = make_seq(list(range(1, 6)), max_tokens=2)
    block_manager.allocate(
        live_sharer,
        publish_hashes=False,
        max_cached_tokens=block_manager.max_reusable_tokens(live_sharer),
    )
    block_manager.deallocate(source)
    assert source_blocks[0] in block_manager.used_block_ids
    assert source_blocks[1] in block_manager.free_block_ids

    warm = make_seq(list(range(1, 10)), max_tokens=1)
    free_before = list(block_manager.free_block_ids)
    refs_before = [block.ref_count for block in block_manager.blocks]

    assert block_manager.estimate_admission(warm) == (8, 2)
    assert list(block_manager.free_block_ids) == free_before
    assert [block.ref_count for block in block_manager.blocks] == refs_before
    assert warm.block_table == []
    assert warm.num_cached_tokens == 0


def test_allocate_rejects_hash_collision_when_tokens_differ():
    reset_sequence_state()
    block_manager = BlockManager(num_blocks=8, block_size=4)
    cached = make_seq([1, 2, 3, 4], max_tokens=1)
    block_manager.allocate(
        cached,
        publish_hashes=False,
        max_cached_tokens=0,
    )
    block_manager.commit_prefill(cached, 0, len(cached))
    cached_block = cached.block_table[0]
    cached_hash = block_manager.blocks[cached_block].hash
    block_manager.deallocate(cached)

    original_compute_hash = block_manager.compute_hash
    block_manager.compute_hash = (
        lambda token_ids, prefix=-1: cached_hash
    )
    try:
        seq = make_seq([9, 8, 7, 6], max_tokens=1)
        assert block_manager.estimate_admission(seq)[0] == 0
        block_manager.allocate(
            seq,
            publish_hashes=False,
            max_cached_tokens=4,
        )
    finally:
        block_manager.compute_hash = original_compute_hash

    assert seq.num_cached_tokens == 0
    assert seq.block_table[0] != cached_block


def test_allocate_finds_token_match_behind_hash_collision_primary():
    reset_sequence_state()
    block_manager = BlockManager(num_blocks=3, block_size=4)
    original_compute_hash = block_manager.compute_hash
    block_manager.compute_hash = lambda token_ids, prefix=-1: 12345
    try:
        first = make_seq([1, 2, 3, 4], max_tokens=1)
        second = make_seq([5, 6, 7, 8], max_tokens=1)
        for seq in (first, second):
            block_manager.allocate(
                seq,
                publish_hashes=False,
                max_cached_tokens=0,
            )
            block_manager.commit_prefill(seq, 0, len(seq))
        first_block_id = first.block_table[0]
        second_block_id = second.block_table[0]
        assert block_manager.hash_to_block_id[12345] == second_block_id
        block_manager.deallocate(second)
        block_manager.deallocate(first)

        warm = make_seq([1, 2, 3, 4, 9], max_tokens=1)
        block_manager.allocate(
            warm,
            publish_hashes=False,
            max_cached_tokens=block_manager.max_reusable_tokens(warm),
        )
    finally:
        block_manager.compute_hash = original_compute_hash

    assert warm.num_cached_tokens == 4
    assert warm.block_table[0] == first_block_id


def test_clear_reusable_cache_preserves_live_block_metadata():
    reset_sequence_state()
    block_manager = BlockManager(num_blocks=8, block_size=4)
    free_cached = make_seq([1, 2, 3, 4], max_tokens=1)
    block_manager.allocate(
        free_cached,
        publish_hashes=False,
        max_cached_tokens=0,
    )
    block_manager.commit_prefill(free_cached, 0, len(free_cached))
    free_block_id = free_cached.block_table[0]
    block_manager.deallocate(free_cached)

    live = make_seq([5, 6, 7, 8], max_tokens=2)
    block_manager.allocate(
        live,
        publish_hashes=False,
        max_cached_tokens=0,
    )
    block_manager.commit_prefill(live, 0, len(live))
    live_block_id = live.block_table[0]
    live_hash = block_manager.blocks[live_block_id].hash
    live_tokens = list(block_manager.blocks[live_block_id].token_ids)

    cleared = block_manager.clear_reusable_cache()

    assert cleared == 1
    assert block_manager.blocks[free_block_id].hash == -1
    assert block_manager.blocks[free_block_id].token_ids == []
    assert block_manager.blocks[live_block_id].hash == live_hash
    assert block_manager.blocks[live_block_id].token_ids == live_tokens
    assert block_manager.blocks[live_block_id].ref_count == 1
    assert block_manager.hash_to_block_ids == {
        live_hash: {live_block_id},
    }


def test_reusing_idle_block_removes_stale_hash_mapping():
    reset_sequence_state()
    block_manager = BlockManager(num_blocks=1, block_size=4)
    first = make_seq([1, 2, 3, 4], max_tokens=1)
    block_manager.allocate(first, publish_hashes=False, max_cached_tokens=0)
    block_manager.commit_prefill(first, 0, len(first))
    block_id = first.block_table[0]
    first_hash = block_manager.blocks[block_id].hash
    block_manager.deallocate(first)

    second = make_seq([5, 6, 7, 8], max_tokens=1)
    block_manager.allocate(second, publish_hashes=False, max_cached_tokens=0)

    assert first_hash not in block_manager.hash_to_block_id
    block_manager.commit_prefill(second, 0, len(second))
    second_hash = block_manager.blocks[block_id].hash
    assert second_hash != first_hash
    assert block_manager.hash_to_block_id == {second_hash: block_id}
    assert block_manager.hash_to_block_ids == {
        second_hash: {block_id},
    }


def test_reusing_indexed_duplicate_preserves_equivalent_cache_mapping():
    reset_sequence_state()
    block_manager = BlockManager(num_blocks=2, block_size=4)
    first = make_seq([1, 2, 3, 4], max_tokens=1)
    second = make_seq([1, 2, 3, 4], max_tokens=1)
    block_manager.allocate(first, publish_hashes=False, max_cached_tokens=0)
    block_manager.allocate(second, publish_hashes=False, max_cached_tokens=0)
    block_manager.commit_prefill(first, 0, len(first))
    block_manager.commit_prefill(second, 0, len(second))
    first_block_id = first.block_table[0]
    second_block_id = second.block_table[0]
    shared_hash = block_manager.blocks[first_block_id].hash
    assert block_manager.hash_to_block_id[shared_hash] == second_block_id

    block_manager.deallocate(second)
    block_manager.deallocate(first)
    cold = make_seq([5, 6, 7, 8], max_tokens=1)
    block_manager.allocate(cold, publish_hashes=False, max_cached_tokens=0)

    assert cold.block_table == [second_block_id]
    assert block_manager.hash_to_block_id[shared_hash] == first_block_id
    block_manager.deallocate(cold)
    warm = make_seq([1, 2, 3, 4, 9], max_tokens=1)
    block_manager.allocate(
        warm,
        publish_hashes=False,
        max_cached_tokens=block_manager.max_reusable_tokens(warm),
    )
    assert warm.num_cached_tokens == 4
    assert warm.block_table[0] == first_block_id


def test_capacity_pressure_never_returns_live_shared_block():
    reset_sequence_state()
    block_manager = BlockManager(num_blocks=3, block_size=4)
    live = make_seq([1, 2, 3, 4], max_tokens=2)
    block_manager.allocate(
        live,
        publish_hashes=False,
        max_cached_tokens=0,
    )
    block_manager.commit_prefill(live, 0, len(live))
    live_block_id = live.block_table[0]

    shared = make_seq([1, 2, 3, 4, 5], max_tokens=1)
    block_manager.allocate(
        shared,
        publish_hashes=False,
        max_cached_tokens=block_manager.max_reusable_tokens(shared),
    )
    assert shared.block_table[0] == live_block_id
    assert block_manager.blocks[live_block_id].ref_count == 2

    other = make_seq([9, 8, 7, 6], max_tokens=1)
    block_manager.allocate(
        other,
        publish_hashes=False,
        max_cached_tokens=0,
    )

    assert other.block_table[0] != live_block_id
    assert live_block_id in block_manager.used_block_ids
    assert live_block_id not in block_manager.free_block_ids


def _block_manager_ownership_snapshot(block_manager):
    return {
        "free": tuple(block_manager.free_block_ids),
        "used": frozenset(block_manager.used_block_ids),
        "refs": tuple(
            block.ref_count for block in block_manager.blocks
        ),
        "generations": tuple(
            block.generation for block in block_manager.blocks
        ),
        "hashes": tuple(
            block.hash for block in block_manager.blocks
        ),
        "tokens": tuple(
            tuple(block.token_ids) for block in block_manager.blocks
        ),
        "hash_to_block_id": dict(
            block_manager.hash_to_block_id
        ),
        "hash_to_block_ids": {
            block_hash: frozenset(block_ids)
            for block_hash, block_ids
            in block_manager.hash_to_block_ids.items()
        },
    }


def test_block_generation_tracks_content_lifetime_not_refcount_lifetime():
    reset_sequence_state()
    block_manager = BlockManager(num_blocks=1, block_size=4)
    first = make_seq([1, 2, 3, 4], max_tokens=1)
    block_manager.allocate(
        first,
        publish_hashes=False,
        max_cached_tokens=0,
    )
    block_id = first.block_table[0]
    assert block_manager.blocks[block_id].generation == 1
    block_manager.commit_prefill(first, 0, len(first))
    block_manager.deallocate(first)
    assert block_manager.blocks[block_id].generation == 1

    reservation = block_manager.reserve_exact_prefix(
        (1, 2, 3, 4),
    )
    assert reservation is not None
    assert reservation.block_identities == ((
        block_id,
        1,
        block_manager.blocks[block_id].hash,
    ),)
    assert block_manager.blocks[block_id].generation == 1
    block_manager.release_prefix_reservation(reservation)

    second = make_seq([5, 6, 7, 8], max_tokens=1)
    block_manager.allocate(
        second,
        publish_hashes=False,
        max_cached_tokens=0,
    )
    assert second.block_table == [block_id]
    assert block_manager.blocks[block_id].generation == 2


def test_exact_prefix_reservation_holds_live_and_idle_multi_owner_refs():
    reset_sequence_state()
    block_manager = BlockManager(num_blocks=4, block_size=4)
    source = make_seq(list(range(1, 9)), max_tokens=1)
    block_manager.allocate(
        source,
        publish_hashes=False,
        max_cached_tokens=0,
    )
    block_manager.commit_prefill(source, 0, len(source))
    block_ids = tuple(source.block_table)
    block_manager.deallocate(source)

    live = make_seq([1, 2, 3, 4, 9], max_tokens=1)
    block_manager.allocate(
        live,
        publish_hashes=False,
        max_cached_tokens=4,
    )
    assert live.block_table[0] == block_ids[0]
    assert block_manager.blocks[block_ids[0]].ref_count == 1
    assert block_manager.blocks[block_ids[1]].ref_count == 0

    reservation = block_manager.reserve_exact_prefix(
        tuple(range(1, 9)),
        owner_count=2,
    )

    assert reservation is not None
    assert reservation.block_ids == block_ids
    assert reservation.token_count == 8
    assert reservation.owner_count == 2
    assert reservation.state == "reserved"
    assert block_manager.blocks[block_ids[0]].ref_count == 3
    assert block_manager.blocks[block_ids[1]].ref_count == 2
    assert set(block_ids).issubset(block_manager.used_block_ids)
    assert tuple(
        identity[1] for identity in reservation.block_identities
    ) == tuple(
        block_manager.blocks[block_id].generation
        for block_id in block_ids
    )

    block_manager.release_prefix_reservation(reservation)
    assert reservation.state == "released"
    assert block_manager.blocks[block_ids[0]].ref_count == 1
    assert block_manager.blocks[block_ids[1]].ref_count == 0
    assert block_ids[0] in block_manager.used_block_ids
    assert block_ids[1] in block_manager.free_block_ids


def test_exact_prefix_partial_miss_is_read_only():
    reset_sequence_state()
    block_manager = BlockManager(num_blocks=4, block_size=4)
    _publish_and_release(block_manager, [1, 2, 3, 4])
    before = _block_manager_ownership_snapshot(block_manager)

    reservation = block_manager.reserve_exact_prefix(
        (1, 2, 3, 4, 5, 6, 7, 8),
        owner_count=2,
    )

    assert reservation is None
    assert _block_manager_ownership_snapshot(block_manager) == before


def test_exact_prefix_reservation_rolls_back_partial_activation_failure():
    reset_sequence_state()
    block_manager = BlockManager(num_blocks=4, block_size=4)
    block_ids = tuple(_publish_and_release(
        block_manager,
        list(range(1, 9)),
    ))
    before = _block_manager_ownership_snapshot(block_manager)
    original_activate = block_manager._activate_cached_block
    calls = []

    def failing_activate(block_id, owner_count=1):
        calls.append(block_id)
        if len(calls) == 2:
            raise RuntimeError("injected block activation failure")
        return original_activate(block_id, owner_count)

    block_manager._activate_cached_block = failing_activate
    try:
        block_manager.reserve_exact_prefix(
            tuple(range(1, 9)),
            owner_count=2,
        )
    except RuntimeError as error:
        assert str(error) == "injected block activation failure"
    else:
        raise AssertionError("partial activation failure was swallowed")
    finally:
        block_manager._activate_cached_block = original_activate

    assert tuple(calls) == block_ids
    assert _block_manager_ownership_snapshot(block_manager) == before


def test_prefix_reservation_attachment_transfers_refs_to_destinations():
    reset_sequence_state()
    block_manager = BlockManager(num_blocks=4, block_size=4)
    block_ids = tuple(_publish_and_release(
        block_manager,
        list(range(1, 9)),
    ))
    reservation = block_manager.reserve_exact_prefix(
        tuple(range(1, 9)),
        owner_count=2,
    )
    first = make_seq(list(range(1, 10)), max_tokens=1)
    second = make_seq(list(range(1, 11)), max_tokens=1)

    block_manager.attach_prefix_reservation(
        reservation,
        (first, second),
    )

    assert reservation.state == "attached"
    for sequence in (first, second):
        assert sequence.block_table == list(block_ids)
        assert sequence.num_cached_tokens == 8
        assert sequence.num_computed_tokens == 8
    assert all(
        block_manager.blocks[block_id].ref_count == 2
        for block_id in block_ids
    )
    block_manager.deallocate(first)
    assert all(
        block_manager.blocks[block_id].ref_count == 1
        for block_id in block_ids
    )
    block_manager.deallocate(second)
    assert all(
        block_manager.blocks[block_id].ref_count == 0
        for block_id in block_ids
    )


def test_prefix_reservation_validation_and_terminal_states_fail_closed():
    reset_sequence_state()
    block_manager = BlockManager(num_blocks=4, block_size=4)
    _publish_and_release(block_manager, [1, 2, 3, 4])
    invalid_reservations = (
        lambda: block_manager.reserve_exact_prefix([1, 2, 3, 4]),
        lambda: block_manager.reserve_exact_prefix((1, 2, 3)),
        lambda: block_manager.reserve_exact_prefix(
            (1, 2, 3, 4),
            owner_count=0,
        ),
    )
    for operation in invalid_reservations:
        try:
            operation()
        except ValueError:
            pass
        else:
            raise AssertionError("invalid prefix reservation was accepted")

    reservation = block_manager.reserve_exact_prefix((1, 2, 3, 4))
    destination = make_seq([1, 2, 3, 4, 5], max_tokens=1)
    duplicate = block_manager.reserve_exact_prefix(
        (1, 2, 3, 4),
        owner_count=2,
    )
    before = _block_manager_ownership_snapshot(block_manager)
    try:
        block_manager.attach_prefix_reservation(
            duplicate,
            (destination, destination),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate destinations were accepted")
    assert _block_manager_ownership_snapshot(block_manager) == before
    block_manager.release_prefix_reservation(duplicate)

    dirty = make_seq([1, 2, 3, 4, 5], max_tokens=1)
    dirty.block_table.append(99)
    before = _block_manager_ownership_snapshot(block_manager)
    try:
        block_manager.attach_prefix_reservation(
            reservation,
            (dirty,),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("dirty destination was accepted")
    assert _block_manager_ownership_snapshot(block_manager) == before

    block_manager.release_prefix_reservation(reservation)
    for operation in (
        lambda: block_manager.release_prefix_reservation(reservation),
        lambda: block_manager.attach_prefix_reservation(
            reservation,
            (destination,),
        ),
    ):
        try:
            operation()
        except RuntimeError:
            pass
        else:
            raise AssertionError(
                "terminal reservation operation was accepted"
            )


def test_sequence_block_reservation_cold_is_private_until_attachment():
    reset_sequence_state()
    block_manager = BlockManager(num_blocks=6, block_size=4)
    sequence = make_seq(list(range(1, 10)), max_tokens=1)

    reservation = block_manager.reserve_sequence_blocks(
        sequence,
        max_cached_tokens=block_manager.max_reusable_tokens(sequence),
    )

    assert sequence.block_table == []
    assert sequence.num_cached_tokens == 0
    assert sequence.num_computed_tokens == 0
    assert reservation.cached_tokens == 0
    assert reservation.prefix_block_count == 0
    assert reservation.new_block_count == sequence.num_blocks
    assert len(reservation.block_ids) == sequence.num_blocks
    assert reservation.block_identities == ()
    assert all(
        block_manager.blocks[block_id].ref_count == 1
        for block_id in reservation.block_ids
    )

    block_manager.attach_sequence_reservation(
        reservation,
        sequence,
    )
    assert reservation.state == "attached"
    assert sequence.block_table == list(reservation.block_ids)
    assert sequence.num_cached_tokens == 0
    assert sequence.num_computed_tokens == 0
    block_manager.deallocate(sequence)


def test_sequence_block_reservation_matches_warm_allocate_semantics():
    reset_sequence_state()
    reserved_manager = BlockManager(num_blocks=8, block_size=4)
    allocated_manager = BlockManager(num_blocks=8, block_size=4)
    tokens = list(range(1, 10))
    for manager in (reserved_manager, allocated_manager):
        _publish_and_release(manager, list(range(1, 9)))

    reserved_sequence = make_seq(tokens, max_tokens=1)
    allocated_sequence = make_seq(tokens, max_tokens=1)
    max_cached_tokens = reserved_manager.max_reusable_tokens(
        reserved_sequence
    )
    reservation = reserved_manager.reserve_sequence_blocks(
        reserved_sequence,
        max_cached_tokens=max_cached_tokens,
    )
    allocated_manager.allocate(
        allocated_sequence,
        publish_hashes=False,
        max_cached_tokens=max_cached_tokens,
    )

    assert reservation.cached_tokens == allocated_sequence.num_cached_tokens
    assert reservation.prefix_block_count == 2
    assert reservation.new_block_count == 1
    assert reservation.block_ids[:2] == tuple(
        allocated_sequence.block_table[:2]
    )
    assert len(reservation.block_ids) == len(
        allocated_sequence.block_table
    )
    assert reserved_sequence.block_table == []
    reserved_manager.attach_sequence_reservation(
        reservation,
        reserved_sequence,
    )
    assert reserved_sequence.num_cached_tokens == 8
    assert reserved_sequence.num_computed_tokens == 8


def test_sequence_block_reservation_keeps_sampleable_token_cap():
    reset_sequence_state()
    block_manager = BlockManager(num_blocks=4, block_size=4)
    cached_block = _publish_and_release(
        block_manager,
        [1, 2, 3, 4],
    )[0]
    sequence = make_seq([1, 2, 3, 4], max_tokens=1)

    reservation = block_manager.reserve_sequence_blocks(
        sequence,
        max_cached_tokens=block_manager.max_reusable_tokens(sequence),
    )

    assert reservation.cached_tokens == 0
    assert reservation.prefix_block_count == 0
    assert reservation.new_block_count == 1
    assert reservation.block_ids[0] != cached_block
    block_manager.release_sequence_reservation(reservation)

    default_sequence = make_seq([1, 2, 3, 4], max_tokens=1)
    default_reservation = block_manager.reserve_sequence_blocks(
        default_sequence,
    )
    assert default_reservation.cached_tokens == 0
    assert default_reservation.prefix_block_count == 0
    block_manager.release_sequence_reservation(default_reservation)


def test_sequence_block_reservation_first_miss_allocates_all_suffix_blocks():
    reset_sequence_state()
    block_manager = BlockManager(num_blocks=6, block_size=4)
    first_block = _publish_and_release(
        block_manager,
        [1, 2, 3, 4],
    )[0]
    collision_hash = block_manager.blocks[first_block].hash
    original_compute_hash = block_manager.compute_hash
    block_manager.compute_hash = (
        lambda token_ids, prefix=-1: collision_hash
        if prefix == -1 else original_compute_hash(token_ids, prefix)
    )
    sequence = make_seq(
        [9, 8, 7, 6, 1, 2, 3, 4, 5],
        max_tokens=1,
    )
    try:
        reservation = block_manager.reserve_sequence_blocks(
            sequence,
            max_cached_tokens=8,
        )
    finally:
        block_manager.compute_hash = original_compute_hash

    assert reservation.cached_tokens == 0
    assert reservation.prefix_block_count == 0
    assert reservation.new_block_count == sequence.num_blocks
    assert reservation.block_identities == ()
    if first_block in reservation.block_ids:
        assert block_manager.blocks[first_block].generation == 2
        assert block_manager.blocks[first_block].hash == -1
    block_manager.release_sequence_reservation(reservation)


def test_sequence_block_reservation_capacity_and_suffix_failure_are_atomic():
    reset_sequence_state()
    capacity_manager = BlockManager(num_blocks=2, block_size=4)
    sequence = make_seq(list(range(1, 10)), max_tokens=1)
    before = _block_manager_ownership_snapshot(capacity_manager)
    try:
        capacity_manager.reserve_sequence_blocks(
            sequence,
            max_cached_tokens=0,
        )
    except RuntimeError as error:
        assert "insufficient KV blocks" in str(error)
    else:
        raise AssertionError("insufficient reservation capacity was accepted")
    assert _block_manager_ownership_snapshot(capacity_manager) == before
    assert sequence.block_table == []

    block_manager = BlockManager(num_blocks=6, block_size=4)
    _publish_and_release(block_manager, [1, 2, 3, 4])
    sequence = make_seq(list(range(1, 13)), max_tokens=1)
    before = _block_manager_ownership_snapshot(block_manager)
    original_allocate = block_manager._allocate_block
    calls = []

    def failing_allocate(block_id):
        calls.append(block_id)
        if len(calls) == 2:
            raise RuntimeError("injected suffix allocation failure")
        return original_allocate(block_id)

    block_manager._allocate_block = failing_allocate
    try:
        block_manager.reserve_sequence_blocks(
            sequence,
            max_cached_tokens=4,
        )
    except RuntimeError as error:
        assert str(error) == "injected suffix allocation failure"
    else:
        raise AssertionError("suffix allocation failure was swallowed")
    finally:
        block_manager._allocate_block = original_allocate
    assert len(calls) == 2
    assert _block_manager_ownership_snapshot(block_manager) == before
    assert sequence.block_table == []

    reset_sequence_state()
    duplicate_manager = BlockManager(num_blocks=6, block_size=4)
    duplicate_manager.compute_hash = (
        lambda token_ids, prefix=-1: 12345
    )
    first_cached = make_seq([1, 2, 3, 4], max_tokens=1)
    second_cached = make_seq([5, 6, 7, 8], max_tokens=1)
    for cached in (first_cached, second_cached):
        duplicate_manager.allocate(
            cached,
            publish_hashes=False,
            max_cached_tokens=0,
        )
        duplicate_manager.commit_prefill(cached, 0, len(cached))
        duplicate_manager.deallocate(cached)
    live_sequences = []
    for base in (20, 30, 40, 50):
        live = make_seq(
            [base, base + 1, base + 2, base + 3],
            max_tokens=1,
        )
        duplicate_manager.allocate(
            live,
            publish_hashes=False,
            max_cached_tokens=0,
        )
        live_sequences.append(live)
    assert len(duplicate_manager.free_block_ids) == 2
    before = _block_manager_ownership_snapshot(duplicate_manager)
    original_allocate = duplicate_manager._allocate_block
    calls = []

    def failing_duplicate_allocate(block_id):
        calls.append(block_id)
        if len(calls) == 2:
            raise RuntimeError("injected duplicate-hash failure")
        return original_allocate(block_id)

    duplicate_manager._allocate_block = failing_duplicate_allocate
    try:
        duplicate_manager.reserve_sequence_blocks(
            make_seq(list(range(100, 108)), max_tokens=1),
            max_cached_tokens=0,
        )
    except RuntimeError as error:
        assert str(error) == "injected duplicate-hash failure"
    else:
        raise AssertionError("duplicate-hash rollback failure was swallowed")
    finally:
        duplicate_manager._allocate_block = original_allocate
    assert len(calls) == 2
    assert _block_manager_ownership_snapshot(duplicate_manager) == before

    post_mutation_manager = BlockManager(num_blocks=4, block_size=4)
    post_mutation_sequence = make_seq(
        list(range(200, 208)),
        max_tokens=1,
    )
    before = _block_manager_ownership_snapshot(post_mutation_manager)
    original_allocate = post_mutation_manager._allocate_block
    calls = []

    def failing_after_allocate(block_id):
        calls.append(block_id)
        block = original_allocate(block_id)
        if len(calls) == 2:
            raise RuntimeError("injected post-mutation failure")
        return block

    post_mutation_manager._allocate_block = failing_after_allocate
    try:
        post_mutation_manager.reserve_sequence_blocks(
            post_mutation_sequence,
            max_cached_tokens=0,
        )
    except RuntimeError as error:
        assert str(error) == "injected post-mutation failure"
    else:
        raise AssertionError("post-mutation allocation failure was swallowed")
    finally:
        post_mutation_manager._allocate_block = original_allocate
    assert len(calls) == 2
    assert (
        _block_manager_ownership_snapshot(post_mutation_manager)
        == before
    )


def test_sequence_block_reservation_release_and_stale_attach_fail_closed():
    reset_sequence_state()
    block_manager = BlockManager(num_blocks=6, block_size=4)
    prefix_block = _publish_and_release(
        block_manager,
        [1, 2, 3, 4],
    )[0]
    prefix_hash = block_manager.blocks[prefix_block].hash
    sequence = make_seq([1, 2, 3, 4, 5], max_tokens=1)
    reservation = block_manager.reserve_sequence_blocks(
        sequence,
        max_cached_tokens=4,
    )
    refs_before_attach = tuple(
        block.ref_count for block in block_manager.blocks
    )
    block_manager.blocks[prefix_block].generation += 1
    try:
        block_manager.attach_sequence_reservation(
            reservation,
            sequence,
        )
    except RuntimeError as error:
        assert "stale" in str(error)
    else:
        raise AssertionError("stale reservation identity was accepted")
    assert sequence.block_table == []
    assert reservation.state == "reserved"
    block_manager.blocks[prefix_block].generation -= 1
    assert tuple(
        block.ref_count for block in block_manager.blocks
    ) == refs_before_attach

    block_manager.release_sequence_reservation(reservation)
    assert reservation.state == "released"
    assert block_manager.blocks[prefix_block].ref_count == 0
    assert block_manager.blocks[prefix_block].hash == prefix_hash
    assert block_manager.blocks[prefix_block].token_ids == [1, 2, 3, 4]
    assert all(
        block_manager.blocks[block_id].ref_count == 0
        for block_id in reservation.block_ids
    )
    for operation in (
        lambda: block_manager.release_sequence_reservation(reservation),
        lambda: block_manager.attach_sequence_reservation(
            reservation,
            sequence,
        ),
    ):
        try:
            operation()
        except RuntimeError:
            pass
        else:
            raise AssertionError(
                "terminal sequence reservation operation was accepted"
            )

    dirty = make_seq([1, 2, 3, 4, 5], max_tokens=1)
    dirty.block_table.append(99)
    try:
        block_manager.reserve_sequence_blocks(dirty)
    except ValueError:
        pass
    else:
        raise AssertionError("dirty sequence reservation was accepted")

    malformed = block_manager_mod.SequenceBlockReservation(
        block_ids=(999,),
        block_identities=(),
        cached_tokens=0,
        prefix_block_count=0,
        new_block_count=1,
    )
    before = _block_manager_ownership_snapshot(block_manager)
    try:
        block_manager.release_sequence_reservation(malformed)
    except ValueError:
        pass
    else:
        raise AssertionError("malformed sequence reservation was released")
    assert _block_manager_ownership_snapshot(block_manager) == before

    identity_sequence = make_seq([6, 7, 8, 9, 10], max_tokens=1)
    identity_reservation = block_manager.reserve_sequence_blocks(
        identity_sequence,
        max_cached_tokens=0,
    )
    malformed_identity = block_manager_mod.SequenceBlockReservation(
        block_ids=identity_reservation.block_ids,
        block_identities=((999, 1, 1),),
        cached_tokens=4,
        prefix_block_count=1,
        new_block_count=len(identity_reservation.block_ids) - 1,
    )
    before = _block_manager_ownership_snapshot(block_manager)
    try:
        block_manager.attach_sequence_reservation(
            malformed_identity,
            identity_sequence,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("malformed prefix identity was attached")
    assert identity_sequence.block_table == []
    assert _block_manager_ownership_snapshot(block_manager) == before
    block_manager.release_sequence_reservation(identity_reservation)


def test_normal_prefill_publishes_only_after_postprocess():
    reset_sequence_state()
    scheduler = Scheduler(
        make_config(max_num_prefill_tokens_per_step=0)
    )
    seq = make_seq([1, 2, 3, 4, 5], max_tokens=2)
    scheduler.add(seq)

    seqs, is_prefill, do_sample = scheduler.schedule()
    h0 = scheduler.block_manager.compute_hash([1, 2, 3, 4], -1)

    assert seqs == [seq]
    assert is_prefill is True
    assert do_sample is True
    assert seq.prefill_chunk_start == 0
    assert seq.prefill_chunk_end == 5
    assert h0 not in scheduler.block_manager.hash_to_block_id

    scheduler.postprocess(seqs, [99], is_prefill, do_sample)

    assert h0 in scheduler.block_manager.hash_to_block_id
    assert seq.num_computed_tokens == 5
    assert seq.completion_token_ids == [99]


def test_normal_prefill_does_not_reuse_prefix_created_in_same_batch():
    reset_sequence_state()
    scheduler = Scheduler(make_config(
        max_num_prefill_tokens_per_step=0,
        max_num_seqs=3,
        max_num_batched_tokens=32,
    ))
    seq_a = make_seq([1, 2, 3, 4, 5], max_tokens=1)
    seq_b = make_seq([9, 8, 7, 6, 5], max_tokens=1)
    seq_c = make_seq([1, 2, 3, 4, 5], max_tokens=1)
    for seq in (seq_a, seq_b, seq_c):
        scheduler.add(seq)

    seqs, is_prefill, do_sample = scheduler.schedule()

    assert seqs == [seq_a, seq_b, seq_c]
    assert is_prefill is True
    assert do_sample is True
    assert [seq.num_cached_tokens for seq in seqs] == [0, 0, 0]
    assert all(
        seq.prefill_chunk_end > seq.prefill_chunk_start
        for seq in seqs
    )
    assert seq_a.block_table[0] != seq_c.block_table[0]


def test_normal_prefill_exact_block_warm_hit_recomputes_final_block():
    reset_sequence_state()
    scheduler = Scheduler(make_config(
        max_num_prefill_tokens_per_step=0,
        max_num_batched_tokens=32,
    ))
    cold = make_seq([1, 2, 3, 4], max_tokens=1)
    scheduler.add(cold)
    seqs, is_prefill, do_sample = scheduler.schedule()
    scheduler.postprocess(seqs, [70], is_prefill, do_sample)

    warm = make_seq([1, 2, 3, 4], max_tokens=1)
    scheduler.add(warm)
    seqs, is_prefill, do_sample = scheduler.schedule()

    assert seqs == [warm]
    assert warm.num_cached_tokens == 0
    assert warm.prefill_chunk_start == 0
    assert warm.prefill_chunk_end == 4


def _seed_scheduler_cache(scheduler, token_ids):
    seq = make_seq(token_ids, max_tokens=1)
    scheduler.add(seq)
    seqs, is_prefill, do_sample = scheduler.schedule()
    while not do_sample:
        scheduler.postprocess(seqs, None, is_prefill, do_sample)
        seqs, is_prefill, do_sample = scheduler.schedule()
    scheduler.postprocess(seqs, [71], is_prefill, do_sample)


def test_normal_prefill_warm_hit_reuses_only_complete_prefix_blocks():
    reset_sequence_state()
    scheduler = Scheduler(make_config(
        max_num_prefill_tokens_per_step=0,
        max_num_batched_tokens=32,
    ))
    _seed_scheduler_cache(scheduler, list(range(1, 9)))

    warm = make_seq(list(range(1, 9)), max_tokens=1)
    scheduler.add(warm)
    seqs, is_prefill, do_sample = scheduler.schedule()

    assert warm.num_cached_tokens == 4
    assert warm.prefill_chunk_start == 4
    assert warm.prefill_chunk_end == 8
    assert do_sample is True


def test_normal_prefill_token_budget_counts_only_uncached_tokens():
    reset_sequence_state()
    scheduler = Scheduler(make_config(
        max_num_prefill_tokens_per_step=0,
        max_num_seqs=3,
        max_num_batched_tokens=10,
        max_model_len=10,
    ))
    _publish_and_release(
        scheduler.block_manager,
        list(range(1, 9)),
    )

    warm_seqs = [
        make_seq(list(range(1, 10)), max_tokens=1)
        for _ in range(3)
    ]
    for seq in warm_seqs:
        scheduler.add(seq)
    seqs, is_prefill, do_sample = scheduler.schedule()

    assert seqs == warm_seqs
    assert is_prefill is True
    assert do_sample is True
    assert [seq.num_cached_tokens for seq in seqs] == [8, 8, 8]
    assert [seq.prefill_chunk_start for seq in seqs] == [8, 8, 8]
    assert [seq.prefill_chunk_end for seq in seqs] == [9, 9, 9]


def test_normal_prefill_token_budget_still_limits_cold_prompts():
    reset_sequence_state()
    scheduler = Scheduler(make_config(
        max_num_prefill_tokens_per_step=0,
        max_num_seqs=3,
        max_num_batched_tokens=10,
        max_model_len=10,
    ))
    cold_seqs = [
        make_seq(range(offset, offset + 5), max_tokens=1)
        for offset in (0, 10, 20)
    ]
    for seq in cold_seqs:
        scheduler.add(seq)

    seqs, is_prefill, do_sample = scheduler.schedule()

    assert seqs == cold_seqs[:2]
    assert is_prefill is True
    assert do_sample is True
    assert list(scheduler.waiting) == cold_seqs[2:]


def test_chunked_prefill_uses_same_sampleable_prefix_cap():
    reset_sequence_state()
    scheduler = Scheduler(
        make_config(max_num_prefill_tokens_per_step=4)
    )
    _seed_scheduler_cache(scheduler, list(range(1, 9)))

    warm = make_seq(list(range(1, 9)), max_tokens=1)
    scheduler.add(warm)
    seqs, is_prefill, do_sample = scheduler.schedule()

    assert warm.num_cached_tokens == 4
    assert warm.prefill_chunk_start == 4
    assert warm.prefill_chunk_end == 8
    assert do_sample is True


def test_commit_accepted_tokens_appends_sequence_and_releases_unused_blocks():
    reset_sequence_state()
    block_manager = BlockManager(num_blocks=8, block_size=4)
    seq = make_seq([1, 2, 3])
    block_manager.allocate(seq)
    original_block = seq.block_table[0]

    reserved = block_manager.reserve_append_blocks(seq, 6)
    assert len(reserved) == 2

    block_manager.commit_accepted_tokens(seq, [4, 5], reserved)

    assert seq.token_ids == [1, 2, 3, 4, 5]
    assert seq.last_token == 5
    assert seq.num_tokens == 5
    assert seq.block_table == [original_block]
    assert block_manager.blocks[reserved[0]].ref_count == 0
    assert reserved[0] in block_manager.free_block_ids
    assert block_manager.blocks[reserved[1]].ref_count == 0
    assert reserved[1] in block_manager.free_block_ids
    h0 = block_manager.compute_hash([1, 2, 3, 4], -1)
    assert block_manager.blocks[original_block].hash == h0
    assert block_manager.blocks[original_block].token_ids == [1, 2, 3, 4]
    assert block_manager.hash_to_block_id[h0] == original_block


def test_commit_accepted_tokens_zero_accept_releases_all_reserved_blocks():
    reset_sequence_state()
    block_manager = BlockManager(num_blocks=8, block_size=4)
    seq = make_seq([1, 2, 3, 4])
    block_manager.allocate(seq)
    original_block_table = list(seq.block_table)
    original_token_ids = list(seq.token_ids)

    reserved = block_manager.reserve_append_blocks(seq, 5)
    assert len(reserved) == 2

    block_manager.commit_accepted_tokens(seq, [], reserved)

    assert seq.token_ids == original_token_ids
    assert seq.last_token == original_token_ids[-1]
    assert seq.num_tokens == len(original_token_ids)
    assert seq.block_table == original_block_table
    for block_id in reserved:
        assert block_manager.blocks[block_id].ref_count == 0
        assert block_id in block_manager.free_block_ids


def test_commit_accepted_tokens_publishes_multiple_full_block_hashes():
    reset_sequence_state()
    block_manager = BlockManager(num_blocks=8, block_size=4)
    seq = make_seq([1, 2, 3])
    block_manager.allocate(seq)

    reserved = block_manager.reserve_append_blocks(seq, 9)
    block_manager.commit_accepted_tokens(seq, [4, 5, 6, 7, 8, 9, 10, 11, 12], reserved)

    h0 = block_manager.compute_hash([1, 2, 3, 4], -1)
    h1 = block_manager.compute_hash([5, 6, 7, 8], h0)
    h2 = block_manager.compute_hash([9, 10, 11, 12], h1)
    assert seq.token_ids == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    assert len(seq.block_table) == 3
    assert block_manager.hash_to_block_id[h0] == seq.block_table[0]
    assert block_manager.hash_to_block_id[h1] == seq.block_table[1]
    assert block_manager.blocks[seq.block_table[0]].token_ids == [1, 2, 3, 4]
    assert block_manager.blocks[seq.block_table[1]].token_ids == [5, 6, 7, 8]
    assert block_manager.blocks[seq.block_table[2]].hash == -1

    block_manager.may_append(seq)

    assert block_manager.hash_to_block_id[h2] == seq.block_table[2]
    assert block_manager.blocks[seq.block_table[2]].token_ids == [9, 10, 11, 12]


def test_commit_accepted_tokens_keeps_scheduler_hash_state_after_crossing_boundary():
    reset_sequence_state()
    block_manager = BlockManager(num_blocks=8, block_size=4)
    seq = make_seq([1, 2, 3])
    block_manager.allocate(seq)

    reserved = block_manager.reserve_append_blocks(seq, 2)
    block_manager.commit_accepted_tokens(seq, [4, 5], reserved)
    block_manager.may_append(seq)

    assert seq.token_ids == [1, 2, 3, 4, 5]
    assert len(seq.block_table) == 2
    assert block_manager.blocks[seq.block_table[0]].hash != -1
    assert block_manager.blocks[seq.block_table[1]].hash == -1


def test_sam_originated_acceptance_crosses_block_boundary():
    reset_sequence_state()
    block_size = 4
    block_manager = BlockManager(num_blocks=8, block_size=block_size)
    prompt_tokens = [1, 2, 3]
    repeated_verified_prefix = [4, 5, 6, 7, 1, 2]
    index = SuffixAutomatonDraftIndex(
        prompt_tokens + repeated_verified_prefix
    )
    draft = index.propose(max_draft_tokens=16)
    current_block_offset = len(prompt_tokens) % block_size
    assert len(draft.tokens) > block_size - current_block_offset

    seq = make_seq(prompt_tokens)
    block_manager.allocate(seq)
    reserved = block_manager.reserve_append_blocks(
        seq,
        len(draft.tokens) + block_size,
    )
    expected_target_prefix = list(draft.tokens)
    block_manager.commit_accepted_tokens(
        seq,
        expected_target_prefix,
        reserved,
    )
    committed_tokens = seq.token_ids[len(prompt_tokens):]

    assert committed_tokens == expected_target_prefix
    assert len(seq.block_table) > 1
    assert block_manager.blocks[seq.block_table[0]].token_ids == seq.block(0)
    adopted = set(seq.block_table) & set(reserved)
    unused = set(reserved) - adopted
    assert adopted
    assert unused
    assert all(block_manager.blocks[block_id].ref_count == 1 for block_id in adopted)
    assert all(block_manager.blocks[block_id].ref_count == 0 for block_id in unused)
    assert all(block_id in block_manager.free_block_ids for block_id in unused)
    assert len(committed_tokens) > block_size - current_block_offset


def test_commit_accepted_tokens_leaves_just_filled_last_block_for_scheduler_publish():
    reset_sequence_state()
    block_manager = BlockManager(num_blocks=8, block_size=4)
    seq = make_seq([1, 2, 3])
    block_manager.allocate(seq)

    reserved = block_manager.reserve_append_blocks(seq, 1)
    block_manager.commit_accepted_tokens(seq, [4], reserved)
    block_manager.may_append(seq)

    assert seq.token_ids == [1, 2, 3, 4]
    assert len(seq.block_table) == 1
    assert block_manager.blocks[seq.block_table[0]].hash != -1


def test_max_consecutive_prefill_chunks_yields_to_decode():
    reset_sequence_state()
    scheduler = Scheduler(make_config(
        max_num_prefill_tokens_per_step=4,
        chunked_prefill_decode_first=False,
        chunked_prefill_max_consecutive_chunks=2,
    ))
    running = make_seq([90, 91, 92, 93], max_tokens=4)
    scheduler.block_manager.allocate(running)
    running.append_token(94)
    running.status = SequenceStatus.RUNNING
    running.num_computed_tokens = len(running)
    scheduler.running.append(running)
    long_prefill = make_seq(range(16), max_tokens=4)
    scheduler.add(long_prefill)

    seqs, is_prefill, do_sample = scheduler.schedule()
    assert seqs == [long_prefill]
    assert is_prefill is True
    assert do_sample is False
    scheduler.postprocess(seqs, None, is_prefill, do_sample)

    seqs, is_prefill, do_sample = scheduler.schedule()
    assert seqs == [long_prefill]
    assert is_prefill is True
    assert do_sample is False
    scheduler.postprocess(seqs, None, is_prefill, do_sample)

    seqs, is_prefill, do_sample = scheduler.schedule()
    assert seqs == [running]
    assert is_prefill is False
    assert do_sample is True
    assert scheduler.last_policy_branch == "bounded_prefill_yield"
    scheduler.postprocess(seqs, [123], is_prefill, do_sample)

    seqs, is_prefill, do_sample = scheduler.schedule()
    assert seqs == [long_prefill]
    assert is_prefill is True
    assert do_sample is False


def test_mixed_prefill_decode_schedules_prefill_chunk_with_decode():
    reset_sequence_state()
    scheduler = Scheduler(make_config(
        max_num_seqs=4,
        max_num_prefill_tokens_per_step=4,
        chunked_prefill_decode_first=False,
        chunked_prefill_mixed_batch=True,
    ))
    running = make_seq([90, 91, 92, 93], max_tokens=4)
    scheduler.block_manager.allocate(running)
    running.append_token(94)
    running.status = SequenceStatus.RUNNING
    running.num_computed_tokens = len(running)
    scheduler.running.append(running)
    long_prefill = make_seq(range(10), max_tokens=4)
    scheduler.add(long_prefill)

    seqs, is_prefill, do_sample, batch_kind = scheduler.schedule()

    assert seqs == [long_prefill, running]
    assert is_prefill is True
    assert do_sample is True
    assert batch_kind == "mixed"
    assert long_prefill.step_is_decode is False
    assert running.step_is_decode is True
    assert long_prefill.prefill_chunk_start == 0
    assert long_prefill.prefill_chunk_end == 4
    assert long_prefill.prefill_chunk_final is False
    assert list(scheduler.running) == []
    assert scheduler.last_policy_branch == "mixed_prefill_decode"


def test_required_mixed_decode_failure_is_transactional():
    reset_sequence_state()
    scheduler = Scheduler(make_config(
        max_num_seqs=1,
        max_num_batched_tokens=4,
        chunked_prefill_mixed_batch=False,
        chunked_prefill_adaptive_mixed=True,
    ))
    running = make_running(scheduler)
    waiting = add_waiting(scheduler, 1)[0]
    queue_before = scheduler.observation_snapshot()
    waiting_blocks_before = list(waiting.block_table)

    result = scheduler._schedule_mixed_prefill_decode(
        allow_waiting_admission=True,
        require_decode=True,
    )

    assert result is None
    assert scheduler.observation_snapshot() == queue_before
    assert waiting.block_table == waiting_blocks_before
    assert list(scheduler.running) == [running]
    assert list(scheduler.waiting) == [waiting]


def test_required_mixed_reserves_kv_block_for_decode_before_prefill():
    reset_sequence_state()
    scheduler = Scheduler(make_config(
        max_num_seqs=2,
        max_num_batched_tokens=8,
        num_kvcache_blocks=3,
        kvcache_block_size=4,
        chunked_prefill_adaptive_mixed=True,
    ))
    running = make_running(
        scheduler,
        token_ids=(90, 91, 92, 93, 94),
    )
    waiting = add_waiting(scheduler, 1, prompt_tokens=8)[0]
    free_before = len(scheduler.block_manager.free_block_ids)

    result = scheduler._schedule_mixed_prefill_decode(
        allow_waiting_admission=True,
        require_decode=True,
    )

    assert result is None
    assert len(scheduler.block_manager.free_block_ids) == free_before
    assert list(scheduler.running) == [running]
    assert list(scheduler.waiting) == [waiting]


def test_required_mixed_success_contains_both_roles():
    reset_sequence_state()
    scheduler = Scheduler(make_config(
        max_num_seqs=4,
        max_num_batched_tokens=16,
        num_kvcache_blocks=16,
        chunked_prefill_adaptive_mixed=True,
    ))
    running = make_running(scheduler)
    waiting = add_waiting(scheduler, 1)[0]

    seqs, is_prefill, do_sample, batch_kind = (
        scheduler._schedule_mixed_prefill_decode(
            allow_waiting_admission=True,
            require_decode=True,
        )
    )

    assert is_prefill is True
    assert do_sample is True
    assert batch_kind == "mixed"
    assert waiting in seqs
    assert running in seqs
    assert any(not seq.step_is_decode for seq in seqs)
    assert any(seq.step_is_decode for seq in seqs)


def test_mixed_short_prefill_batching_reserves_slot_for_decode():
    reset_sequence_state()
    scheduler = Scheduler(make_config(
        max_num_seqs=4,
        max_num_batched_tokens=32,
        max_num_prefill_tokens_per_step=4,
        chunked_prefill_decode_first=False,
        chunked_prefill_mixed_batch=True,
    ))
    running = make_seq([90, 91, 92, 93], max_tokens=4)
    scheduler.block_manager.allocate(running)
    running.append_token(94)
    running.status = SequenceStatus.RUNNING
    running.num_computed_tokens = len(running)
    scheduler.running.append(running)
    short_prompts = [make_seq([i, i + 1, i + 2, i + 3], max_tokens=4) for i in range(0, 16, 4)]
    for seq in short_prompts:
        scheduler.add(seq)

    seqs, is_prefill, do_sample, batch_kind = scheduler.schedule()

    assert is_prefill is True
    assert do_sample is True
    assert batch_kind == "mixed"
    assert running in seqs
    assert seqs[-1] == running
    assert len([seq for seq in seqs if not seq.step_is_decode]) == 3
    assert list(scheduler.waiting) == [short_prompts[-1]]


def test_mixed_prefill_reserves_token_budget_for_decode_queries():
    reset_sequence_state()
    scheduler = Scheduler(make_config(
        max_num_seqs=4,
        max_num_batched_tokens=12,
        max_num_prefill_tokens_per_step=4,
        chunked_prefill_decode_first=False,
        chunked_prefill_mixed_batch=True,
    ))
    running = make_seq([90, 91, 92, 93], max_tokens=4)
    scheduler.block_manager.allocate(running)
    running.append_token(94)
    running.status = SequenceStatus.RUNNING
    running.num_computed_tokens = len(running)
    scheduler.running.append(running)
    short_prompts = [make_seq([i, i + 1, i + 2, i + 3], max_tokens=4) for i in range(0, 12, 4)]
    for seq in short_prompts:
        scheduler.add(seq)

    seqs, is_prefill, do_sample, batch_kind = scheduler.schedule()

    assert is_prefill is True
    assert do_sample is True
    assert batch_kind == "mixed"
    assert seqs == [short_prompts[0], short_prompts[1], running]
    assert sum(seq.prefill_chunk_end - seq.prefill_chunk_start for seq in seqs if not seq.step_is_decode) == 8
    assert len([seq for seq in seqs if seq.step_is_decode]) == 1
    assert list(scheduler.waiting) == [short_prompts[2]]


def test_mixed_decode_rows_respect_remaining_token_budget():
    reset_sequence_state()
    scheduler = Scheduler(make_config(
        max_num_seqs=4,
        max_num_batched_tokens=9,
        max_num_prefill_tokens_per_step=4,
        chunked_prefill_decode_first=False,
        chunked_prefill_mixed_batch=True,
    ))
    running_seqs = []
    for offset in (80, 90):
        running = make_seq([offset, offset + 1, offset + 2, offset + 3], max_tokens=4)
        scheduler.block_manager.allocate(running)
        running.append_token(offset + 4)
        running.status = SequenceStatus.RUNNING
        running.num_computed_tokens = len(running)
        scheduler.running.append(running)
        running_seqs.append(running)
    short_prompts = [make_seq([i, i + 1, i + 2, i + 3], max_tokens=4) for i in range(0, 8, 4)]
    for seq in short_prompts:
        scheduler.add(seq)

    seqs, is_prefill, do_sample, batch_kind = scheduler.schedule()

    prefill_tokens = sum(seq.prefill_chunk_end - seq.prefill_chunk_start for seq in seqs if not seq.step_is_decode)
    decode_rows = [seq for seq in seqs if seq.step_is_decode]
    assert is_prefill is True
    assert do_sample is True
    assert batch_kind == "mixed"
    assert prefill_tokens + len(decode_rows) <= scheduler.max_num_batched_tokens
    assert seqs == [short_prompts[0], short_prompts[1], running_seqs[0]]
    assert list(scheduler.running) == [running_seqs[1]]


def test_mixed_first_prefill_chunk_shrinks_to_leave_decode_budget():
    reset_sequence_state()
    scheduler = Scheduler(make_config(
        max_num_seqs=4,
        max_num_batched_tokens=5,
        max_num_prefill_tokens_per_step=8,
        chunked_prefill_decode_first=False,
        chunked_prefill_mixed_batch=True,
    ))
    running = make_seq([90, 91, 92, 93], max_tokens=4)
    scheduler.block_manager.allocate(running)
    running.append_token(94)
    running.status = SequenceStatus.RUNNING
    running.num_computed_tokens = len(running)
    scheduler.running.append(running)
    long_prefill = make_seq(range(12), max_tokens=4)
    scheduler.add(long_prefill)

    seqs, is_prefill, do_sample, batch_kind = scheduler.schedule()

    assert is_prefill is True
    assert do_sample is True
    assert batch_kind == "mixed"
    assert seqs == [long_prefill, running]
    assert long_prefill.prefill_chunk_start == 0
    assert long_prefill.prefill_chunk_end == 4
    assert long_prefill.step_do_sample is False
    assert running.step_is_decode is True


def test_mixed_min_prompt_tokens_defers_short_waiting_prompt_to_decode():
    reset_sequence_state()
    scheduler = Scheduler(make_config(
        max_num_seqs=4,
        max_num_prefill_tokens_per_step=4,
        chunked_prefill_decode_first=False,
        chunked_prefill_mixed_batch=True,
        chunked_prefill_mixed_min_prompt_tokens=8,
    ))
    running = make_seq([90, 91, 92, 93], max_tokens=4)
    scheduler.block_manager.allocate(running)
    running.append_token(94)
    running.status = SequenceStatus.RUNNING
    running.num_computed_tokens = len(running)
    scheduler.running.append(running)
    short_prefill = make_seq([1, 2, 3, 4], max_tokens=4)
    scheduler.add(short_prefill)

    seqs, is_prefill, do_sample = scheduler.schedule()

    assert seqs == [running]
    assert is_prefill is False
    assert do_sample is True
    assert list(scheduler.waiting) == [short_prefill]


def test_mixed_min_prompt_tokens_still_admits_long_waiting_prompt():
    reset_sequence_state()
    scheduler = Scheduler(make_config(
        max_num_seqs=4,
        max_num_prefill_tokens_per_step=4,
        chunked_prefill_decode_first=False,
        chunked_prefill_mixed_batch=True,
        chunked_prefill_mixed_min_prompt_tokens=8,
    ))
    running = make_seq([90, 91, 92, 93], max_tokens=4)
    scheduler.block_manager.allocate(running)
    running.append_token(94)
    running.status = SequenceStatus.RUNNING
    running.num_computed_tokens = len(running)
    scheduler.running.append(running)
    long_prefill = make_seq(range(10), max_tokens=4)
    scheduler.add(long_prefill)

    seqs, is_prefill, do_sample, batch_kind = scheduler.schedule()

    assert seqs == [long_prefill, running]
    assert is_prefill is True
    assert do_sample is True
    assert batch_kind == "mixed"
    assert long_prefill.prefill_chunk_start == 0
    assert long_prefill.prefill_chunk_end == 4


def test_mixed_postprocess_commits_prefill_and_appends_decode_only_for_intermediate_chunk():
    reset_sequence_state()
    scheduler = Scheduler(make_config(
        max_num_seqs=4,
        max_num_prefill_tokens_per_step=4,
        chunked_prefill_decode_first=False,
        chunked_prefill_mixed_batch=True,
    ))
    running = make_seq([90, 91, 92, 93], max_tokens=4)
    scheduler.block_manager.allocate(running)
    running.append_token(94)
    running.status = SequenceStatus.RUNNING
    running.num_computed_tokens = len(running)
    scheduler.running.append(running)
    long_prefill = make_seq(range(10), max_tokens=4)
    scheduler.add(long_prefill)

    seqs, is_prefill, do_sample, batch_kind = scheduler.schedule()
    # Intermediate prefill chunks do not need a sampled token; mixed sampling
    # should only return tokens for rows whose step_do_sample is True.
    scheduler.postprocess(seqs, [123], is_prefill, do_sample, batch_kind)

    assert long_prefill.completion_token_ids == []
    assert long_prefill.num_computed_tokens == 4
    assert long_prefill.status == SequenceStatus.PREFILLING
    assert running.completion_token_ids == [94, 123]
    assert running.status == SequenceStatus.RUNNING
    assert list(scheduler.prefilling) == [long_prefill]
    assert list(scheduler.running) == [running]


def test_mixed_final_prefill_chunk_and_decode_consume_tokens_in_sequence_order():
    reset_sequence_state()
    scheduler = Scheduler(make_config(
        max_num_seqs=4,
        max_num_prefill_tokens_per_step=4,
        chunked_prefill_decode_first=False,
        chunked_prefill_mixed_batch=True,
    ))
    running_a = make_seq([90, 91, 92, 93], max_tokens=4)
    scheduler.block_manager.allocate(running_a)
    running_a.append_token(94)
    running_a.status = SequenceStatus.RUNNING
    running_a.num_computed_tokens = len(running_a)
    scheduler.running.append(running_a)
    running_b = make_seq([80, 81, 82, 83], max_tokens=4)
    scheduler.block_manager.allocate(running_b)
    running_b.append_token(84)
    running_b.status = SequenceStatus.RUNNING
    running_b.num_computed_tokens = len(running_b)
    scheduler.running.append(running_b)
    final_prefill = make_seq([1, 2, 3, 4], max_tokens=4)
    scheduler.add(final_prefill)

    seqs, is_prefill, do_sample, batch_kind = scheduler.schedule()
    assert seqs == [final_prefill, running_a, running_b]
    scheduler.postprocess(seqs, [111, 222, 333], is_prefill, do_sample, batch_kind)

    assert final_prefill.completion_token_ids == [111]
    assert running_a.completion_token_ids == [94, 222]
    assert running_b.completion_token_ids == [84, 333]
    assert list(scheduler.running) == [final_prefill, running_a, running_b]


def test_mixed_prefill_fallback_counts_toward_consecutive_prefill_limit():
    reset_sequence_state()
    scheduler = Scheduler(make_config(
        max_num_seqs=1,
        max_num_prefill_tokens_per_step=4,
        chunked_prefill_decode_first=False,
        chunked_prefill_mixed_batch=True,
        chunked_prefill_max_consecutive_chunks=1,
    ))
    running = make_seq([90, 91, 92, 93], max_tokens=4)
    scheduler.block_manager.allocate(running)
    running.append_token(94)
    running.status = SequenceStatus.RUNNING
    running.num_computed_tokens = len(running)
    scheduler.running.append(running)
    long_prefill = make_seq(range(12), max_tokens=4)
    scheduler.add(long_prefill)

    seqs, is_prefill, do_sample = scheduler.schedule()
    assert seqs == [long_prefill]
    assert is_prefill is True
    assert do_sample is False
    scheduler.postprocess(seqs, None, is_prefill, do_sample)

    seqs, is_prefill, do_sample = scheduler.schedule()
    assert seqs == [running]
    assert is_prefill is False
    assert do_sample is True


def test_sequence_pickle_preserves_mixed_step_metadata_for_tp_workers():
    reset_sequence_state()
    seq = make_seq([1, 2, 3, 4], max_tokens=4)
    seq.step_is_decode = True
    seq.step_do_sample = False
    seq.num_computed_tokens = 4
    seq.prefill_chunk_start = 3
    seq.prefill_chunk_end = 4
    seq.prefill_chunk_final = True
    seq.temperature = 0.6

    restored = pickle.loads(pickle.dumps(seq))

    assert restored.step_is_decode is True
    assert restored.step_do_sample is False
    assert restored.num_computed_tokens == 4
    assert restored.prefill_chunk_start == 3
    assert restored.prefill_chunk_end == 4
    assert restored.prefill_chunk_final is True
    assert restored.temperature == 0.6


def test_lm_head_prefill_returns_only_requested_logits_rows():
    if torch is None:
        return
    head = ParallelLMHead(4, 2)
    with torch.no_grad():
        head.weight.copy_(torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
        ]))
    hidden = torch.tensor([
        [0.0, 0.0],
        [1.0, 2.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [3.0, 4.0],
        [5.0, 6.0],
    ])
    set_context(True, cu_seqlens_q=torch.tensor([0, 2, 5, 6], dtype=torch.int32),
                logits_indices=torch.tensor([1, 5], dtype=torch.int64))

    logits = head(hidden)

    expected = torch.nn.functional.linear(hidden[[1, 5]], head.weight)
    assert logits.shape == (2, 4)
    assert torch.equal(logits, expected)
    reset_context_global()


def main():
    test_explicit_kv_capacity_is_pinned_and_fails_closed()
    test_adaptive_mixed_config_defaults_and_fail_closed_contract()
    test_adaptive_mixed_invalid_configurations_fail_before_model_start()
    test_slo_mixed_config_defaults_and_fail_closed_contract()
    test_slo_chunk_ladder_and_largest_safe_boundary()
    test_adaptive_state_requires_two_high_observations_and_resets_streak()
    test_adaptive_state_low_hysteresis_enters_inactive_or_draining()
    test_adaptive_ineligible_decision_clears_transition_streaks()
    test_adaptive_observation_and_empty_reset_are_exact()
    test_adaptive_disabled_matches_decode_first_schedule_and_snapshot()
    test_adaptive_second_high_observation_activates_and_mixes()
    test_adaptive_two_mixed_steps_force_decode_yield()
    test_adaptive_draining_never_admits_waiting_and_returns_inactive()
    test_adaptive_no_running_uses_chunked_prefill_without_fake_decode()
    test_adaptive_required_mixed_failure_falls_back_to_decode()
    test_intermediate_chunk_does_not_sample_or_append()
    test_final_chunk_samples_once_and_moves_to_running()
    test_chunked_prefill_batches_multiple_short_final_prompts()
    test_chunked_prefill_batches_warm_prompt_by_uncached_tokens()
    test_decode_first_prioritizes_existing_running_sequence()
    test_scheduler_observation_snapshot_reports_queue_and_kv_state()
    test_model_runner_memory_snapshot_is_read_only_and_counts_all_kv_storage()
    test_model_runner_hybrid_prefix_cache_snapshot_is_rank_local()
    test_engine_collects_all_rank_hybrid_prefix_cache_snapshots()
    test_llm_engine_step_records_observation_without_changing_return_value()
    test_engine_samples_one_decision_and_one_step_end_timestamp()
    test_p5_decision_snapshot_is_immutable_until_postprocess_copy()
    test_decode_progress_updates_only_for_completion_tokens()
    test_progress_survives_preemption_but_is_excluded_until_running()
    test_clock_regression_is_sticky_and_forces_decode_only()
    test_missing_runnable_progress_fails_closed()
    test_active_demand_never_overrides_no_slack()
    test_largest_safe_chunk_is_selected_with_exact_integer_math()
    test_mixed_batch_contains_exact_oldest_runnable_decode_row()
    test_protected_row_reservation_failure_does_not_substitute_younger()
    test_p0_p3_p4_scheduling_is_unchanged_by_p5_support()
    test_chunked_prefill_decode_fallback_reports_branch_without_changing_result()
    test_legacy_prefill_reports_branch_without_changing_result()
    test_legacy_decode_reports_branch_without_changing_result()
    test_add_rejects_request_beyond_max_model_len()
    test_add_rejects_prompt_beyond_logical_kv_capacity()
    test_add_accounts_for_decode_kv_capacity_boundary()
    test_add_capacity_boundary_for_multiple_logical_block_counts()
    test_chunked_prefill_progresses_with_varied_chunk_sizes()
    test_short_prefill_batch_respects_sequence_and_token_limits()
    test_chunked_prefill_does_not_publish_future_block_hashes()
    test_chunked_prefill_restores_reused_cached_block_metadata()
    test_max_reusable_tokens_keeps_one_sampleable_token()
    test_allocate_caps_exact_block_aligned_cache_hit()
    test_allocate_reuses_only_blocks_before_sampleable_suffix()
    test_can_allocate_excludes_live_prefix_hits_from_free_block_requirement()
    test_can_allocate_counts_idle_prefix_hits_as_free_block_requirement()
    test_estimate_admission_is_read_only_for_live_and_idle_hits()
    test_allocate_rejects_hash_collision_when_tokens_differ()
    test_allocate_finds_token_match_behind_hash_collision_primary()
    test_clear_reusable_cache_preserves_live_block_metadata()
    test_reusing_idle_block_removes_stale_hash_mapping()
    test_reusing_indexed_duplicate_preserves_equivalent_cache_mapping()
    test_capacity_pressure_never_returns_live_shared_block()
    test_block_generation_tracks_content_lifetime_not_refcount_lifetime()
    test_exact_prefix_reservation_holds_live_and_idle_multi_owner_refs()
    test_exact_prefix_partial_miss_is_read_only()
    test_exact_prefix_reservation_rolls_back_partial_activation_failure()
    test_prefix_reservation_attachment_transfers_refs_to_destinations()
    test_prefix_reservation_validation_and_terminal_states_fail_closed()
    test_sequence_block_reservation_cold_is_private_until_attachment()
    test_sequence_block_reservation_matches_warm_allocate_semantics()
    test_sequence_block_reservation_keeps_sampleable_token_cap()
    test_sequence_block_reservation_first_miss_allocates_all_suffix_blocks()
    test_sequence_block_reservation_capacity_and_suffix_failure_are_atomic()
    test_sequence_block_reservation_release_and_stale_attach_fail_closed()
    test_normal_prefill_publishes_only_after_postprocess()
    test_normal_prefill_does_not_reuse_prefix_created_in_same_batch()
    test_normal_prefill_exact_block_warm_hit_recomputes_final_block()
    test_normal_prefill_warm_hit_reuses_only_complete_prefix_blocks()
    test_normal_prefill_token_budget_counts_only_uncached_tokens()
    test_normal_prefill_token_budget_still_limits_cold_prompts()
    test_chunked_prefill_uses_same_sampleable_prefix_cap()
    test_commit_accepted_tokens_appends_sequence_and_releases_unused_blocks()
    test_commit_accepted_tokens_zero_accept_releases_all_reserved_blocks()
    test_commit_accepted_tokens_publishes_multiple_full_block_hashes()
    test_commit_accepted_tokens_keeps_scheduler_hash_state_after_crossing_boundary()
    test_sam_originated_acceptance_crosses_block_boundary()
    test_commit_accepted_tokens_leaves_just_filled_last_block_for_scheduler_publish()
    test_max_consecutive_prefill_chunks_yields_to_decode()
    test_mixed_prefill_decode_schedules_prefill_chunk_with_decode()
    test_required_mixed_decode_failure_is_transactional()
    test_required_mixed_reserves_kv_block_for_decode_before_prefill()
    test_required_mixed_success_contains_both_roles()
    test_mixed_short_prefill_batching_reserves_slot_for_decode()
    test_mixed_prefill_reserves_token_budget_for_decode_queries()
    test_mixed_decode_rows_respect_remaining_token_budget()
    test_mixed_first_prefill_chunk_shrinks_to_leave_decode_budget()
    test_mixed_min_prompt_tokens_defers_short_waiting_prompt_to_decode()
    test_mixed_min_prompt_tokens_still_admits_long_waiting_prompt()
    test_mixed_postprocess_commits_prefill_and_appends_decode_only_for_intermediate_chunk()
    test_mixed_final_prefill_chunk_and_decode_consume_tokens_in_sequence_order()
    test_mixed_prefill_fallback_counts_toward_consecutive_prefill_limit()
    test_sequence_pickle_preserves_mixed_step_metadata_for_tp_workers()
    test_lm_head_prefill_returns_only_requested_logits_rows()
    print("chunked prefill tests passed")


if __name__ == "__main__":
    main()
