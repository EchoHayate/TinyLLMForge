import argparse
from tinyvllm.llm import LLM
from tinyvllm.sampling_params import SamplingParams
from tinyvllm.config import Config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True, help="Path to model directory")
    args = parser.parse_args()

    # Enable INT8 KV Quantization
    config = Config(
        model=args.model_dir, 
        quantization="int8",      # 模型权重量化
        kv_quantization="int8",   # 开启我们刚刚手写的 KV Cache 量化!
        max_num_batched_tokens=4096, 
        max_num_seqs=4, 
        max_model_len=4096, 
        tensor_parallel_size=1
    )

    print("Initializing Engine with INT8 KV Quantization...")
    # 你应该能在控制台看到分配了原本两倍数量的 num_kvcache_blocks
    llm = LLM(config)
    
    prompts = [
        "Please explain the principles of Token-Level scale INT8 quantization for LLMs.",
        "用一段 Python 代码说明如何计算一个 Tensor 的最大值缩放系数。"
    ]

    sampling_params = SamplingParams(temperature=0.0, max_tokens=256)

    print(f"\nSubmitting requests to test INT8 KV generation coherence...")
    outputs = llm.generate(prompts, sampling_params)
    
    for i, output in enumerate(outputs):
        print(f"\n=== Output {i} (INT8 KV) ===")
        print(output["text"])
        print("==================\n")

if __name__ == "__main__":
    main()
