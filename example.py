import os
from tinyvllm import LLM, SamplingParams
from transformers import AutoTokenizer

def main():
    path = os.path.expanduser("../Qwen3-0.6B")
    # 分词器，将句子分成多个token, 然后编码成数字
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager = True, tensor_parallel_size = 1)

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