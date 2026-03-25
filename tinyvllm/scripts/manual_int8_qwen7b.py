import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# -----------------------------
# 1) 自己实现一个 weight-only INT8 的线性层
#    - 对称量化
#    - per-output-channel scale
#    - forward 时再反量化到 fp16/bf16 做 F.linear
# -----------------------------
class QuantLinearInt8(nn.Module):
    def __init__(self, in_features, out_features, bias=True, compute_dtype=torch.float16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.compute_dtype = compute_dtype

        # 量化后的权重: [out_features, in_features], int8
        self.register_buffer("qweight", torch.empty(out_features, in_features, dtype=torch.int8))
        # 每个输出通道一个 scale: [out_features, 1]
        self.register_buffer("scale", torch.empty(out_features, 1, dtype=torch.float16))

        if bias:
            self.register_buffer("bias", torch.empty(out_features, dtype=torch.float16))
        else:
            self.bias = None

    @staticmethod
    def from_linear(linear: nn.Linear, compute_dtype=torch.float16):
        q = QuantLinearInt8(
            linear.in_features,
            linear.out_features,
            bias=(linear.bias is not None),
            compute_dtype=compute_dtype,
        )

        # 取原始权重
        w = linear.weight.data.detach().to(torch.float32)  # [out, in]

        # 对称 per-output-channel 量化
        # 每一行一个 scale
        max_abs = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        scale = max_abs / 127.0
        qweight = torch.round(w / scale).clamp(-127, 127).to(torch.int8)

        q.qweight.copy_(qweight)
        q.scale.copy_(scale.to(torch.float16))

        if linear.bias is not None:
            q.bias.copy_(linear.bias.data.detach().to(torch.float16))

        return q

    def forward(self, x):
        # 反量化到 compute_dtype
        # 注意：这一步会引入额外开销，所以它不是最优性能方案
        w = (self.qweight.to(torch.float32) * self.scale.to(torch.float32)).to(self.compute_dtype)

        if x.dtype != self.compute_dtype:
            x = x.to(self.compute_dtype)

        bias = self.bias
        if bias is not None and bias.dtype != self.compute_dtype:
            bias = bias.to(self.compute_dtype)

        return F.linear(x, w, bias)


# -----------------------------
# 2) 替换模型中的 Linear
#    为了稳妥，我们先只量化 Transformer block 内部的 Linear：
#    - attention q/k/v/o
#    - mlp gate/up/down
#    保留 embedding、norm、lm_head 为 fp16
# -----------------------------
def should_quantize_module(module_name: str, module: nn.Module) -> bool:
    if not isinstance(module, nn.Linear):
        return False

    # 这些一般先不量化，稳定一些
    skip_keywords = [
        "lm_head",
        "embed_tokens",
        "norm",
    ]
    if any(k in module_name for k in skip_keywords):
        return False

    return True


def replace_linear_with_int8(model: nn.Module, compute_dtype=torch.float16):
    # 先拿到所有模块名，避免边遍历边改结构
    named_modules = list(model.named_modules())

    replaced = 0
    for module_name, module in named_modules:
        if should_quantize_module(module_name, module):
            # 找到父模块
            if "." in module_name:
                parent_name = module_name.rsplit(".", 1)[0]
                child_name = module_name.rsplit(".", 1)[1]
                parent = model.get_submodule(parent_name)
            else:
                parent = model
                child_name = module_name

            old_linear = getattr(parent, child_name)
            qlinear = QuantLinearInt8.from_linear(old_linear, compute_dtype=compute_dtype)

            setattr(parent, child_name, qlinear)
            replaced += 1

    return replaced


# -----------------------------
# 3) 打印模型参数 / 显存估算
# -----------------------------
def estimate_num_params(model: nn.Module):
    total = 0
    for _, p in model.named_parameters():
        total += p.numel()
    for _, b in model.named_buffers():
        total += b.numel()
    return total


# -----------------------------
# 4) 主流程
# -----------------------------
def main():
    assert torch.cuda.is_available(), "没有检测到 CUDA GPU"

    device = "cuda"
    compute_dtype = torch.float16

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading model on CPU (fp16)...")
    # 先在 CPU 上加载，避免直接把全精度权重压到 4070 上
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=compute_dtype,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
    )
    model.eval()

    print("Replacing Linear with manual INT8 QuantLinear...")
    replaced = replace_linear_with_int8(model, compute_dtype=compute_dtype)
    print(f"Replaced Linear layers: {replaced}")

    # 把模型转到 GPU
    print("Moving quantized model to GPU...")
    model.to(device)
    model.eval()

    # 尽量控制上下文长度，先跑通
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "请用中文简要解释一下 Prefix Caching 和 KV Cache 的区别。"}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer([text], return_tensors="pt").to(device)

    # 预热
    with torch.no_grad():
        _ = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            use_cache=True
        )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,   # 先别太大
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            use_cache=True
        )
    torch.cuda.synchronize()
    end = time.time()

    input_len = inputs["input_ids"].shape[1]
    new_tokens = outputs.shape[1] - input_len
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3

    answer = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

    print("\n=== Result ===")
    print(answer)
    print(f"\nGeneration time: {end - start:.3f}s")
    if end > start:
        print(f"Decode throughput: {new_tokens / (end - start):.2f} tokens/s")
    print(f"Peak GPU memory: {peak_mem:.2f} GB")


if __name__ == "__main__":
    main()