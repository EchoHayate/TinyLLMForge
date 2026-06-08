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
    source = "from __future__ import annotations\n" + f.read()
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
    )

    kwargs = eval_needle.build_llm_kwargs(args, init_top_k=16)

    assert kwargs["tensor_parallel_size"] == 2
    assert kwargs["quest_top_k_blocks"] == 16
    assert kwargs["act_quant_skip_last"] == 4
    assert kwargs["kv_cartridge_blocks"] == 8
    assert kwargs["kv_cartridge_min_seq_len"] == 2048
    assert kwargs["kv_cartridge_mode"] == "uniform"


def main():
    test_fixed_prompts_reuse_same_magic_across_topk()
    test_default_prompts_keep_topk_seed_offset()
    test_newline_needle_style_delimits_inserted_needle()
    test_clear_prefix_cache_drops_only_reusable_free_blocks()
    test_build_llm_kwargs_uses_configured_tp_size()
    print("eval_needle fixed-prompt tests passed")


if __name__ == "__main__":
    main()
