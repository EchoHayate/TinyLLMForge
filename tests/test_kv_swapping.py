import argparse
from tinyvllm.llm import LLM
from tinyvllm.sampling_params import SamplingParams
from tinyvllm.config import Config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True, help="Path to model directory")
    args = parser.parse_args()

    # 我们刻意将 gpu_memory_utilization 压低至 0.2
    # 这会导致 GPU 预留的 KV Block 极少。
    # 当同时涌入多个长 Context 且都需要解码时，引擎会耗尽 GPU Block，
    # 从而触发 Scheduler 自动将低优先级的 Sequence Swap-out 到 CPU 内存，
    # 并在轮到它时 Swap-in 回去继续生成，保证不发生 OOM。
    config = Config(
        model=args.model_dir, 
        quantization="int8",    # 取决于你测试时使用的模型，如果不开启可以删掉或者置空
        max_num_batched_tokens=4096, 
        max_num_seqs=4, 
        max_model_len=4096, 
        gpu_memory_utilization=0.2, # 核心：极低的 VRAM 利用率，强制触发 OOM 和 Swap
        swap_space_bytes=2 * 1024**3, # 分配 2GB CPU Pinned Memory 用于 Swap
        tensor_parallel_size=1
    )

    print("Initializing Engine with tight GPU memory to trigger KV Swapping...")
    llm = LLM(config)
    
    # 提交多个极其耗费 KV Cache 的并发长文本请求
    prompts = [
        "请详细介绍一下北京的故宫，包括它的历史、建筑布局、著名的宫殿以及背后的文化意义。尽量写得长一些，字数在500字以上。",
        "Please explain the history, architecture, and cultural significance of the Roman Colosseum in comprehensive detail.",
        "深入分析一下大语言模型在未来十年的演进路线，以及它对教育和医疗行业可能产生的颠覆性影响。",
        "Write a comprehensive and highly detailed sci-fi short story about humanity's first successful expedition to Alpha Centauri."
    ]

    sampling_params = SamplingParams(temperature=0.0, max_tokens=256)

    print(f"\nSubmitting {len(prompts)} heavy concurrent requests...")
    outputs = llm.generate(prompts, sampling_params)
    
    for i, output in enumerate(outputs):
        print(f"\n=== Output {i} ===")
        print(output["text"])
        print("==================\n")

if __name__ == "__main__":
    main()
