import os
import argparse
import torch
from tinyvllm.llm import LLM
from tinyvllm.sampling_params import SamplingParams
from tinyvllm.config import Config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True, 
                        help="Path to the DIRECTORY containing your quantized INT8 safetensors and config.json")
    parser.add_argument("--prompt", type=str, default="请用中文简要解释一下 Prefix Caching 和 KV Cache 的区别。")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    # The model-dir needs to contain config.json and model.safetensors or similar quantized weights.
    # Note: make sure you copied tokenizer.json, tokenizer_config.json, config.json from original model
    # to this model-dir along with your quantized .safetensors files.

    config = Config(
        model=args.model_dir, 
        quantization="int8", 
        max_num_batched_tokens=8192, 
        max_num_seqs=64, 
        max_model_len=4096, 
        gpu_memory_utilization=0.9, 
        tensor_parallel_size=1
    )

    print(f"Initializing Engine with INT8 Quantization from: {args.model_dir} ...")
    llm = LLM(config)

    sampling_params = SamplingParams(
        temperature=args.temperature, 
        max_tokens=args.max_tokens
    )

    print(f"\nPrompt: '{args.prompt}'")
    print("Generating...")
    
    # Normally LLMEngine takes batched prompts as strings or Token IDs. 
    # Adapting your runner structure here if `engine.generate` accepts prompt string:
    # Let's assume you have an add_request in your LLM class or direct generate method:
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": args.prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        print(f"Formatted prompt:\n{text}\n")
        
        outputs = llm.generate([text], sampling_params)
        
        for i, output in enumerate(outputs):
            generated_text = output["text"]
            print(f"=== Output {i} ===")
            print(generated_text)
            print("==================\n")
            
    except Exception as e:
        print(f"Error during generation: {e}")

if __name__ == "__main__":
    main()
