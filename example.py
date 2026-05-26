import os
import argparse
from tinyvllm import LLM, SamplingParams
from transformers import AutoTokenizer

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default=os.path.expanduser("../Qwen3-0.6B"))
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--enforce-eager", action="store_true", default=True)
    # 量化 / cpu-offload demo flag
    p.add_argument("--quantization", type=str, default=None,
                   choices=[None, "int8", "int8_bnb", "int2"], help="weight-only 量化方式")
    p.add_argument("--quant-group-size", type=int, default=128,
                   help="分组量化组大小")
    p.add_argument("--cpu-offload", action="store_true", default=False,
                   help="是否启用 cpu-offload (decoder layer 粒度)")
    p.add_argument("--cpu-offload-num-layers", type=int, default=-1,
                   help="卸载到 cpu 的 decoder 层数；-1 表示除最后两层外全部卸载")
    # Quest 动态稀疏 attention
    p.add_argument("--quest-top-k-blocks", type=int, default=-1,
                   help="Quest 每个 query 选 top-k block，-1 关闭")
    p.add_argument("--quest-min-seq-len", type=int, default=1024,
                   help="序列长度小于此值不启用 Quest")
    return p.parse_args()


def main():
    args = parse_args()
    path = args.model
    # 分词器，将句子分成多个token, 然后编码成数字
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(
        path,
        enforce_eager=args.enforce_eager,
        tensor_parallel_size=args.tensor_parallel_size,
        quantization=args.quantization,
        quant_group_size=args.quant_group_size,
        cpu_offload=args.cpu_offload,
        cpu_offload_num_layers=args.cpu_offload_num_layers,
        quest_top_k_blocks=args.quest_top_k_blocks,
        quest_min_seq_len=args.quest_min_seq_len,
    )

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "introduce your self",
        "list all prime numbers within 100",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role":"user", "content":prompt}], 
            tokenize = False,                          # True会输出token的id, False会输出token
            add_generation_prompt = True,              # True会输出生成提示, False不会输出生成提示
            enable_thinking = True,                    # 使用思维链，透明决策过程
        ) for prompt in prompts
    ]

    outputs = llm.generate(prompts, sampling_params)
    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")                    # !r表示强制显示 prompt的 __repr__，不对其进行转义
        print(f"Completion: {output['text']!r}")

if __name__ == "__main__":
    main()
