"""per-layer 激活 outlier 强度诊断：跑一小批 prompt，对每个 decoder layer 的
所有 LinearBase 输入激活，统计 outlier 指标，输出 per-layer 表格。

指标（都按 per-input-channel absmax 向量再聚合）：
  - amax        : 该层所有 channel absmax 的最大值（最强 outlier 通道）
  - p99/median  : channel absmax 的 p99 / median 比值，越大说明 outlier 越尖
  - kurtosis    : 激活展平后的峰度（fp 重尾程度）

目的：验证"W4A8 长文塌方的 outlier 集中在首/尾层"的假设，为 A8 skip 层数
消融提供先验。

用法：
    python tools/diag_layer_outlier.py --model <Qwen3-8B> --out /tmp/outlier.json
"""
import os, sys, argparse, json, re

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_PROMPTS = [
    "The capital of France is Paris, and the history of the city spans over two thousand years of",
    "In a typical transformer architecture, the self-attention mechanism allows each token to attend to",
    "深度学习近年来在自然语言处理、计算机视觉等领域取得了重大突破，其核心思想是通过多层非线性变换",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr)//2]\n    return",
    "Reinforcement learning differs from supervised learning because an agent interacts with an environment and",
    "The Roman Empire at its height stretched from Britain to Mesopotamia, encompassing roughly 60 million people across",
    "User: Explain the difference between a process and a thread.\nAssistant: A process is an independent instance of",
    "Climate change is primarily caused by greenhouse gases such as carbon dioxide and methane, which trap heat in",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--out", type=str, default="/tmp/outlier.json")
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    return p.parse_args()


def main():
    args = parse_args()
    import torch
    from tinyvllm import LLM, SamplingParams
    from tinyvllm.layers.linear import LinearBase

    llm = LLM(args.model, tensor_parallel_size=1, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization,
              enforce_eager=True, quantization=None, act_quant_bits=0)
    model = llm.model_runner.model

    # per-module 收集：channel absmax 向量 + 激活展平后的二阶/四阶矩（算 kurtosis）
    stat = {}  # name -> dict
    handles = []
    layer_re = re.compile(r"\.layers\.(\d+)\.")

    def make_hook(name):
        def _hook(_m, inputs):
            x = inputs[0]
            if x.numel() == 0:
                return
            xf = x.detach().reshape(-1, x.shape[-1]).float()
            cur_absmax = xf.abs().amax(dim=0)        # [in_dim]
            flat = xf.flatten()
            s = stat.get(name)
            if s is None:
                s = {"absmax": cur_absmax,
                     "n": 0, "sum2": 0.0, "sum4": 0.0}
                stat[name] = s
            else:
                s["absmax"] = torch.maximum(s["absmax"], cur_absmax)
            # 累计矩用于 kurtosis（均值近似 0，激活非零均值这里只做粗估）
            s["n"] += flat.numel()
            s["sum2"] += (flat * flat).sum().item()
            s["sum4"] += (flat * flat * flat * flat).sum().item()
        return _hook

    for name, mod in model.named_modules():
        if isinstance(mod, LinearBase) and layer_re.search(name):
            handles.append(mod.register_forward_pre_hook(make_hook(name)))

    sps = [SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=1) for _ in _PROMPTS]
    llm.generate(_PROMPTS, sps, use_tqdm=True)
    for h in handles:
        h.remove()

    # 聚合到 per-layer（同层多个 LinearBase 取 max outlier）
    per_layer = {}
    for name, s in stat.items():
        m = layer_re.search(name)
        idx = int(m.group(1))
        absmax = s["absmax"]
        amax = absmax.max().item()
        med = absmax.median().item()
        p99 = absmax.quantile(0.99).item()
        var = s["sum2"] / max(s["n"], 1)
        m4 = s["sum4"] / max(s["n"], 1)
        kurt = m4 / (var * var + 1e-12)  # 关于 0 的峰度（gaussian≈3）
        rec = per_layer.setdefault(idx, {"amax": 0.0, "p99_med": 0.0, "kurt": 0.0})
        rec["amax"] = max(rec["amax"], amax)
        rec["p99_med"] = max(rec["p99_med"], p99 / (med + 1e-9))
        rec["kurt"] = max(rec["kurt"], kurt)

    num_layers = max(per_layer.keys()) + 1
    print("\n=== per-layer activation outlier (max over LinearBase in layer) ===")
    print(f"{'layer':>5} | {'amax':>10} | {'p99/med':>8} | {'kurtosis':>10}")
    print("-" * 44)
    for i in range(num_layers):
        r = per_layer[i]
        print(f"{i:>5} | {r['amax']:>10.2f} | {r['p99_med']:>8.2f} | {r['kurt']:>10.1f}")

    json.dump({"num_layers": num_layers, "per_layer": per_layer}, open(args.out, "w"), indent=2)
    print(f"\nsaved -> {args.out}")

    # 自动给出首/尾 outlier 强度对比
    head = per_layer[0]["amax"], per_layer[1]["amax"]
    tail = per_layer[num_layers - 1]["amax"], per_layer[num_layers - 2]["amax"]
    mids = [per_layer[i]["amax"] for i in range(2, num_layers - 2)]
    import statistics
    print(f"\nhead(L0,L1) amax = {head[0]:.1f}, {head[1]:.1f}")
    print(f"tail(L{num_layers-1},L{num_layers-2}) amax = {tail[0]:.1f}, {tail[1]:.1f}")
    print(f"mid layers amax median = {statistics.median(mids):.1f}, max = {max(mids):.1f}")


if __name__ == "__main__":
    main()
