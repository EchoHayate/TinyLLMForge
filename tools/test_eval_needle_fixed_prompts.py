"""eval_needle fixed-prompt 模式单测。

只测 prompt/magic 生成逻辑，不加载真实模型。

跑法：python tools/test_eval_needle_fixed_prompts.py
"""

import os
import sys
import types
from types import SimpleNamespace

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# eval_needle 的真实运行依赖 transformers/tinyvllm；本测试只验证 prompt 生成逻辑，
# 用轻量 stub 避免在本地无 torch/transformers 环境下无法 import。
transformers_stub = types.ModuleType("transformers")
transformers_stub.AutoTokenizer = object
sys.modules.setdefault("transformers", transformers_stub)

tinyvllm_stub = types.ModuleType("tinyvllm")
tinyvllm_stub.LLM = object
tinyvllm_stub.SamplingParams = object
sys.modules.setdefault("tinyvllm", tinyvllm_stub)

eval_needle = types.ModuleType("eval_needle_under_test")
eval_needle.__dict__["__file__"] = os.path.join(_THIS_DIR, "eval_needle.py")
with open(os.path.join(_THIS_DIR, "eval_needle.py"), "r") as f:
    source = f.read()
exec(compile(source, os.path.join(_THIS_DIR, "eval_needle.py"), "exec"), eval_needle.__dict__)


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


class FakeBlock:
    def __init__(self, ref_count, hash_value, token_ids):
        self.ref_count = ref_count
        self.hash = hash_value
        self.token_ids = token_ids


class FakeBlockManager:
    def __init__(self):
        self.hash_to_block_id = {11: 0, 22: 1}
        self.blocks = [
            FakeBlock(ref_count=0, hash_value=11, token_ids=[1, 2, 3]),
            FakeBlock(ref_count=1, hash_value=22, token_ids=[4, 5, 6]),
        ]


class FakeLLM:
    def __init__(self):
        block_manager = FakeBlockManager()
        self.scheduler = SimpleNamespace(block_manager=block_manager)


def _args(fixed_prompts: bool):
    return SimpleNamespace(
        context_lens=[128],
        depths=[0.0, 0.5],
        num_trials=3,
        seed=123,
        kv_quant_bits=8,
        fixed_prompts=fixed_prompts,
    )


def test_fixed_prompts_reuse_same_magic_across_topk():
    tokenizer = FakeTokenizer()
    base_prompts, base_metas = eval_needle.build_eval_batch(tokenizer, _args(True), top_k=-1)
    sparse_prompts, sparse_metas = eval_needle.build_eval_batch(tokenizer, _args(True), top_k=12)

    assert base_prompts == sparse_prompts
    assert base_metas == sparse_metas


def test_default_prompts_keep_topk_seed_offset():
    tokenizer = FakeTokenizer()
    base_prompts, base_metas = eval_needle.build_eval_batch(tokenizer, _args(False), top_k=-1)
    sparse_prompts, sparse_metas = eval_needle.build_eval_batch(tokenizer, _args(False), top_k=12)

    assert base_prompts != sparse_prompts
    assert [m["magic"] for m in base_metas] != [m["magic"] for m in sparse_metas]


def test_newline_needle_style_delimits_inserted_needle():
    tokenizer = FakeTokenizer()

    prompt = eval_needle.build_prompt(tokenizer, 128, 0.5, 12345, needle_style="newline")

    assert "\n\nThe magic number is 12345. Remember it.\n\n" in prompt


def test_clear_prefix_cache_drops_only_reusable_free_blocks():
    llm = FakeLLM()

    cleared = eval_needle.clear_prefix_cache(llm)

    bm = llm.scheduler.block_manager
    assert cleared == 1
    assert bm.hash_to_block_id == {}
    assert bm.blocks[0].hash == -1
    assert bm.blocks[0].token_ids == []
    assert bm.blocks[1].hash == 22
    assert bm.blocks[1].token_ids == [4, 5, 6]


def test_build_llm_kwargs_uses_configured_tp_size():
    args = SimpleNamespace(
        model="/tmp/model",
        enforce_eager=True,
        tp_size=2,
        max_model_len=4096,
        gpu_memory_utilization=0.7,
        max_num_seqs=8,
        quest_min_seq_len=512,
        quest_min_saved_blocks=128,
        kv_quant_bits=8,
        kv_quant_group_size=32,
        quantization="int4",
        quant_group_size=32,
        act_quant_bits=8,
        smoothquant_scale_path="/tmp/sq.pt",
        act_quant_skip_first=0,
        act_quant_skip_last=4,
        act_quant_skip_layers=None,
        kv_cartridge_blocks=8,
        kv_cartridge_min_seq_len=2048,
        kv_cartridge_mode="uniform",
        am_compact_blocks=32,
        am_compact_selector="highest",
        am_compact_min_seq_len=4096,
        am_compact_score_method="rms",
        am_compact_beta_bound=3.0,
        am_compact_ridge_lambda=1e-6,
        am_omp_candidate_pool_size=0,
        am_compact_cache_refresh_interval=0,
        am_prefill_cache_ref_query_stride=8,
        am_compact_num_clusters=1,
        am_compact_route_top_k=1,
        am_compact_num_key_spans=1,
        am_compact_decode_refit=False,
        am_compact_decode_refit_mode="selected",
        am_compact_decode_refit_interval=1,
        am_compact_skip_first_layers=0,
        am_compact_skip_last_layers=0,
        am_compact_enable_layers=None,
        am_compact_layer_stride=1,
    )

    kwargs = eval_needle.build_llm_kwargs(args, init_top_k=16)

    assert kwargs["tensor_parallel_size"] == 2
    assert kwargs["quest_top_k_blocks"] == 16
    assert kwargs["quest_min_saved_blocks"] == 128
    assert kwargs["act_quant_skip_last"] == 4
    assert kwargs["kv_cartridge_blocks"] == 8
    assert kwargs["kv_cartridge_min_seq_len"] == 2048
    assert kwargs["kv_cartridge_mode"] == "uniform"
    assert kwargs["am_compact_blocks"] == 32
    assert kwargs["am_compact_selector"] == "highest"
    assert kwargs["am_compact_min_seq_len"] == 4096
    assert kwargs["am_compact_score_method"] == "rms"
    assert kwargs["am_compact_beta_bound"] == 3.0
    assert kwargs["am_compact_ridge_lambda"] == 1e-6
    assert kwargs["am_compact_decode_refit"] is False
    assert kwargs["am_compact_decode_refit_mode"] == "selected"


def test_kv8_quest_quality_remote_runner_is_source_bound_and_paired():
    runner_path = os.path.join(_THIS_DIR, "run_kv8_quest_quality_remote.sh")
    source = open(runner_path, "r", encoding="utf-8").read()

    assert 'git archive "${SOURCE_REVISION}" -- tinyvllm tools/eval_needle.py' in source
    assert "/data00/home/sitian/tllm/kvcapacity-runs/" in source
    assert "--fixed-prompts" in source
    assert "--needle-style newline" in source
    assert "--context-lens 8192" in source
    assert "--depths 0.0 0.25 0.5 0.75 1.0" in source
    assert "--num-trials 5" in source
    assert "--kv-quant-bits 0" in source
    assert "--kv-quant-bits 8" in source
    assert "--top-k-blocks-list -1 16" in source


def test_quest_activation_summary_delta_preserves_missing_evidence():
    assert (
        eval_needle.quest_activation_summary_delta(None, None)
        is None
    )


def test_quest_activation_summary_delta_subtracts_counters():
    before = {
        "steps": 2,
        "reason_counts": {"active": 2},
        "resolved_top_k_counts": {"16": 2},
        "saved_blocks_min": 128,
        "saved_blocks_max": 256,
        "last_observation_id": 2,
        "events": [
            {"observation_id": 1, "reason": "active"},
            {"observation_id": 2, "reason": "active"},
        ],
        "events_dropped": 0,
    }
    after = {
        "steps": 10,
        "reason_counts": {
            "active": 7,
            "below_saved_blocks": 3,
        },
        "resolved_top_k_counts": {"-1": 3, "16": 7},
        "saved_blocks_min": 64,
        "saved_blocks_max": 400,
        "last_observation_id": 10,
        "events": [
            {"observation_id": value, "reason": "active"}
            for value in range(1, 11)
        ],
        "events_dropped": 0,
    }

    delta = eval_needle.quest_activation_summary_delta(
        before,
        after,
    )

    assert delta["steps"] == 8
    assert delta["reason_counts"] == {
        "active": 5,
        "below_saved_blocks": 3,
    }
    assert delta["resolved_top_k_counts"] == {
        "-1": 3,
        "16": 5,
    }
    assert delta["first_observation_id"] == 3
    assert delta["last_observation_id"] == 10
    assert delta["cumulative_saved_blocks_min"] == 64
    assert delta["cumulative_saved_blocks_max"] == 400
    assert [event["observation_id"] for event in delta["events"]] == list(
        range(3, 11)
    )
    assert delta["events_dropped"] == 0


def test_adaptive_quality_remote_runner_is_source_bound():
    runner_path = os.path.join(
        _THIS_DIR,
        "run_kv8_quest_adaptive_quality_remote.sh",
    )
    source = open(runner_path, "r", encoding="utf-8").read()

    assert (
        'git archive "${SOURCE_REVISION}" -- tinyvllm '
        "tools/eval_needle.py"
    ) in source
    assert "/data00/home/sitian/tllm/kvcapacity-runs/" in source
    assert "--fixed-prompts" in source
    assert "--kv-quant-bits 8" in source
    assert "--top-k-blocks-list -1 16" in source
    assert "--quest-min-saved-blocks 128" in source


def main():
    test_fixed_prompts_reuse_same_magic_across_topk()
    test_default_prompts_keep_topk_seed_offset()
    test_newline_needle_style_delimits_inserted_needle()
    test_clear_prefix_cache_drops_only_reusable_free_blocks()
    test_build_llm_kwargs_uses_configured_tp_size()
    test_kv8_quest_quality_remote_runner_is_source_bound_and_paired()
    test_quest_activation_summary_delta_preserves_missing_evidence()
    test_quest_activation_summary_delta_subtracts_counters()
    test_adaptive_quality_remote_runner_is_source_bound()
    print("eval_needle fixed-prompt tests passed")


if __name__ == "__main__":
    main()
