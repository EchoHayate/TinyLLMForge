import os
import time
import argparse
from random import randint, seed
from tinyvllm import LLM, SamplingParams
# from vllm import LLM, SamplingParams


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default=os.path.expanduser("../Qwen3-0.6B/"))
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--enforce-eager", action="store_true", default=False)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--num-seqs", type=int, default=256)
    p.add_argument("--max-input-len", type=int, default=1024)
    p.add_argument("--max-output-len", type=int, default=1024)
    # 量化 / cpu-offload demo flag
    p.add_argument("--quantization", type=str, default=None,
                   choices=[None, "int8", "int8_bnb", "int4", "int2"], help="weight-only 量化方式")
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
    # KV cache 量化（C4）
    p.add_argument("--kv-quant-bits", type=int, default=0,
                   choices=[0, 4, 8], help="KV cache 量化位宽（0=不量化）")
    p.add_argument("--kv-quant-group-size", type=int, default=128,
                   help="KV 量化的 group 大小（沿 head_dim 切）")
    p.add_argument("--kv-quant-symmetric", action="store_true", default=True,
                   help="对称量化（仅 scale，无 zero-point）")
    p.add_argument("--act-quant-bits", type=int, default=0,
                   choices=[0, 8], help="Activation 量化位宽（W4A8 用）")
    return p.parse_args()


def main():
    args = parse_args()
    seed(0)
    num_seqs = args.num_seqs
    max_input_len = args.max_input_len
    max_ouput_len = args.max_output_len

    path = args.model
    llm = LLM(
        path,
        enforce_eager=args.enforce_eager,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        quantization=args.quantization,
        quant_group_size=args.quant_group_size,
        cpu_offload=args.cpu_offload,
        cpu_offload_num_layers=args.cpu_offload_num_layers,
        quest_top_k_blocks=args.quest_top_k_blocks,
        quest_min_seq_len=args.quest_min_seq_len,
        kv_quant_bits=args.kv_quant_bits,
        kv_quant_group_size=args.kv_quant_group_size,
        kv_quant_symmetric=args.kv_quant_symmetric,
        act_quant_bits=args.act_quant_bits,
    )

    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)) for _ in range(num_seqs)]
    # uncomment the following line for vllm
    # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    llm.generate(["Benchmark: "], SamplingParams())
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = (time.time() - t)
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")


if __name__ == "__main__":
    main()
