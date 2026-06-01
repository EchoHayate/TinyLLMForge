"""SmoothQuant 离线校准工具：跑一遍 prompt bank，记录每个 LinearBase 的
per-input-channel 激活 absmax，结合 fp 权重 absmax 计算 per-channel scale s：

    s[i] = max(|X[:,i]|)^alpha / max(|W[:,i]|)^(1-alpha)

把 {full_module_name -> tensor[in_dim] (cpu fp32)} 落盘，loader 在加载量化模型时
按 rank 切片注入：W <- W*s (loader)，x <- x/s (forward)。

约束：
  - 必须 TP=1（保存的 s 是 input-full 维；多卡加载时 RowParallel 按 rank narrow）。
  - 必须 quantization=None / act_quant_bits=0（校准要看 fp 权重 + 原始激活分布）。
  - 必须 enforce_eager=True（cuda graph 下 hook 不会按你想的方式 fire）。
  - 不下载任何外部数据集（磁盘约束）；prompt bank 内联，覆盖中英 / 代码 / 长短。

跑法：
    python tools/calibrate_smoothquant.py \
        --model /path/to/Qwen3-8B \
        --output /tmp/sq_scales_qwen3_8b.pt \
        --alpha 0.5
"""

import os
import sys
import argparse

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ============== prompt bank ==============
# 内联 ~96 条短/中/长混合 prompt：英文叙事 / 代码 / 中英混合 / 问答 / 技术解释。
# 论文显示 SmoothQuant 对校准集不敏感，几十条就够拿到 ≥95% 效果。
_PROMPTS = [
    # 短 - 自然续写
    "The capital of France is",
    "The largest planet in our solar system is",
    "When water freezes, it",
    "Photosynthesis is the process by which",
    "The first president of the United States was",
    "In machine learning, overfitting refers to",
    "The Great Wall of China was built",
    "DNA stands for",
    "Newton's second law states that",
    "The speed of light in vacuum is approximately",
    "Shakespeare's most famous tragedy is",
    "The mitochondria is known as",
    "Einstein's theory of relativity revolutionized",
    "The boiling point of water at sea level is",
    "Quantum mechanics describes",
    "The Pythagorean theorem states that",
    # 短 - 问答
    "Q: What is the difference between TCP and UDP? A:",
    "Q: How does a transformer model work? A:",
    "Q: Why is the sky blue? A:",
    "Q: What causes thunderstorms? A:",
    "Q: How does a CPU execute instructions? A:",
    "Q: What is the role of insulin? A:",
    # 中 - 故事 / 文学
    "Once upon a time, in a small village near the mountains, there lived an old wizard who",
    "The detective walked into the dimly lit room and immediately noticed three things that did not belong",
    "She opened the ancient book and a soft golden light began to emanate from its pages, revealing",
    "After traveling for forty days through the desert, the merchants finally arrived at the oasis where",
    "The spaceship's engines hummed softly as Captain Reyes considered her options. The alien artifact had",
    "In the year 2147, humanity finally made contact with an extraterrestrial civilization that",
    # 中 - 技术解释
    "In a typical transformer architecture, the self-attention mechanism allows each token to",
    "Gradient descent is an iterative optimization algorithm. At each step, we compute the gradient of",
    "Backpropagation works by applying the chain rule of calculus to compute gradients layer by layer",
    "When designing a REST API, the most important principles include statelessness, proper use of HTTP methods",
    "The CAP theorem in distributed systems states that you can only guarantee two of the following three:",
    "Garbage collection in modern programming languages typically uses one of several strategies, including",
    "Vector databases store high-dimensional embeddings and use approximate nearest neighbor search to",
    "In quantum computing, a qubit differs from a classical bit because it can",
    # 中 - 代码
    "def factorial(n):\n    if n <= 1:\n        return 1\n    return",
    "import torch\nimport torch.nn as nn\n\nclass MultiHeadAttention(nn.Module):\n    def __init__(self, d_model, num_heads):\n        super().__init__()\n        self.d_model = d_model\n        self.num_heads = num_heads\n        assert",
    "// Implement quicksort in C++\nvoid quicksort(int* arr, int low, int high) {\n    if (low < high) {\n        int pivot = arr[high];\n        int i = low - 1;\n        for",
    "# Compute the n-th Fibonacci number using dynamic programming\ndef fib(n):\n    if n <= 1:\n        return n\n    dp = [0] * (n + 1)\n    dp[1] = 1\n    for i in range(2, n + 1):",
    "function debounce(fn, delay) {\n    let timer = null;\n    return function(...args) {\n        if (timer)",
    "SELECT department, AVG(salary) AS avg_salary\nFROM employees\nWHERE hire_date >= '2020-01-01'\nGROUP BY department\nHAVING",
    "@app.route('/users/<int:user_id>', methods=['GET'])\ndef get_user(user_id):\n    user = User.query.get_or_404(user_id)\n    return",
    "use std::collections::HashMap;\n\nfn count_words(text: &str) -> HashMap<String, u32> {\n    let mut counts = HashMap::new();\n    for word",
    # 中 - 中文
    "中国的首都是",
    "唐朝是中国历史上最辉煌的时期之一，其文化、经济和政治影响远播",
    "在深度学习中，注意力机制（attention mechanism）的核心思想是",
    "量子计算与经典计算的本质区别在于",
    "人工智能的发展可以分为几个阶段，包括",
    "如何用 Python 实现一个简单的二分查找算法？答：",
    "Transformer 模型相对于 RNN 的主要优势包括",
    "光合作用是植物利用光能将",
    # 中 - 多领域知识
    "Climate change is primarily caused by the increasing concentration of greenhouse gases in the atmosphere, especially",
    "The human immune system has two main components: the innate immune system, which provides immediate but non-specific defense, and",
    "Inflation occurs when the general price level of goods and services rises over time. Central banks typically respond by",
    "The theory of plate tectonics explains how the Earth's lithosphere is divided into",
    "Machine translation has evolved from rule-based systems to statistical models to neural networks. Modern systems use",
    "The French Revolution began in 1789 and fundamentally changed European political thought by",
    # 长 - 段落延续
    "Artificial intelligence has progressed dramatically over the past decade. From narrow systems that could only perform a single task to large language models capable of conversing on virtually any topic, the field has expanded faster than even most experts predicted. However, this rapid progress has also raised serious concerns about safety, alignment, and economic disruption. Researchers and policymakers are now grappling with",
    "The history of computer science is filled with remarkable individuals whose contributions shaped the field as we know it today. Alan Turing's theoretical work laid the foundation for modern computing, while Grace Hopper pioneered the development of compilers and high-level programming languages. John von Neumann's architectural model still underpins virtually every computer in use today. Yet despite these foundational contributions,",
    "When optimizing the performance of a deep learning model on GPUs, several factors come into play. Memory bandwidth often becomes the limiting factor before compute capacity is fully utilized. Techniques such as kernel fusion, mixed-precision arithmetic, and careful management of tensor memory layout can yield substantial speedups. Additionally, for very large models, parallelism strategies like tensor parallelism, pipeline parallelism, and expert parallelism become essential because",
    "The COVID-19 pandemic accelerated the adoption of remote work technologies in ways that were previously thought impossible. Companies that had resisted remote work for decades suddenly found themselves operating fully distributed teams within weeks. This shift exposed both the strengths and weaknesses of existing collaboration tools, and it spurred massive investment in",
    "Reinforcement learning differs from supervised learning in fundamental ways. Rather than learning from labeled examples, an agent in reinforcement learning interacts with an environment, takes actions, and receives rewards or penalties based on the outcomes. Over time, the agent learns a policy that maximizes its cumulative reward. This paradigm has produced remarkable results in domains such as game playing, robotics, and recommendation systems, but it also presents unique challenges including",
    "The Roman Empire at its height stretched from Britain in the northwest to Mesopotamia in the east, encompassing a population of roughly 60 million people across three continents. Its road network, legal system, and engineering achievements set standards that influenced civilizations for over a millennium after its fall. Yet the question of why such a powerful empire eventually collapsed has been debated by historians for centuries. Among the proposed causes are",
    # 长 - 中文段落
    "深度学习近年来在自然语言处理、计算机视觉、语音识别等领域取得了重大突破。其核心思想是通过多层非线性变换从原始数据中自动学习特征表示，避免了传统机器学习方法中繁琐的人工特征工程。然而，深度学习模型也面临着诸多挑战，例如对大规模标注数据的依赖、对计算资源的高要求、模型可解释性差等。针对这些问题，研究者们提出了",
    "中国古代四大发明——造纸术、印刷术、火药和指南针——对世界文明的发展产生了深远影响。其中造纸术发明于东汉时期，由蔡伦改进后逐渐取代竹简和丝帛，使知识传播变得更加便捷。印刷术从雕版印刷发展到活字印刷，极大地推动了书籍的普及。火药最初用于炼丹，后来逐渐应用于军事，改变了战争的形态。指南针的发明则为大航海时代的开启",
    # 长 - 代码 / 数据
    "Below is a Python implementation of a transformer encoder layer. It uses multi-head self-attention followed by a feed-forward network, with residual connections and layer normalization around each sublayer. The forward pass processes a batch of token embeddings and returns contextualized representations.\n\nimport torch\nimport torch.nn as nn\n\nclass TransformerEncoderLayer(nn.Module):\n    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):\n        super().__init__()\n        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)\n        self.linear1 = nn.Linear(d_model, dim_feedforward)\n        self.dropout = nn.Dropout(dropout)\n        self.linear2 = nn.Linear(dim_feedforward, d_model)\n        self.norm1 = nn.LayerNorm(d_model)\n        self.norm2 = nn.LayerNorm(d_model)\n        self.dropout1 = nn.Dropout(dropout)\n        self.dropout2 = nn.Dropout(dropout)\n        self.activation = nn.ReLU()\n\n    def forward(self, src, src_mask=None, src_key_padding_mask=None):",
    "# Performance benchmark results on different hardware\n# GPU: NVIDIA A100 80GB\n# Batch size: 32, Sequence length: 2048\n# Model: Qwen3-8B with W4A8 + KV4 quantization\n#\n# Phase       | Throughput (tok/s) | Memory (GB)\n# ------------|--------------------|-----------\n# Prefill     |              12500 |        14.2\n# Decode (BS1)|                145 |         8.7\n# Decode (BS8)|                890 |         9.4\n# Decode (BS32)|              2840 |        12.1\n#\n# Comparing W4A8+KV4 to W4A16+KV16 baseline:",
    # 短 - 数学 / 推理
    "If a triangle has sides of length 3, 4, and 5, then it is a right triangle because",
    "The derivative of sin(x) with respect to x is",
    "The integral of e^x dx equals",
    "If you flip a fair coin 10 times, the probability of getting exactly 5 heads is",
    "The eigenvalues of the matrix [[2, 1], [1, 2]] are",
    "The sum of the first n positive integers is given by the formula",
    "log_2(1024) =",
    "The Taylor series expansion of e^x around x=0 is",
    # 中 - 技术对比
    "Comparing convolutional neural networks (CNNs) with transformers for image classification tasks: CNNs leverage spatial locality and translation invariance through convolution operations, while transformers use self-attention to model long-range dependencies. Recent work has shown that transformers can match or exceed CNN performance when",
    "The differences between functional programming and object-oriented programming go beyond syntax. Functional programming emphasizes immutability and pure functions, treating computation as the evaluation of mathematical expressions. Object-oriented programming, on the other hand, organizes code around objects that encapsulate state and behavior. Each paradigm has",
    "When choosing between PostgreSQL and MongoDB for a new application, several factors should be considered. PostgreSQL excels at structured data with complex relationships, ACID compliance, and rich SQL queries. MongoDB shines when dealing with flexible schemas, horizontal scaling, and document-oriented data models. The decision often comes down to",
    # 短 - 多语言
    "Bonjour, comment allez-vous?",
    "Hola, ¿cómo estás?",
    "こんにちは、お元気ですか？",
    "안녕하세요, 어떻게 지내세요?",
    "Guten Tag, wie geht es Ihnen?",
    "Привет, как дела?",
    # 长 - 反复对话
    "User: I'm trying to debug a memory leak in my Python application. The memory usage keeps growing even though I've called gc.collect() multiple times. What could be causing this?\nAssistant: Memory leaks in Python despite calling gc.collect() typically have several common causes. First, circular references that involve objects with __del__ methods cannot be collected by the garbage collector. Second, references held in module-level globals or class-level attributes will persist for the program's lifetime. Third, C extensions or libraries that allocate memory outside Python's control may not be tracked by gc. To diagnose,",
    "User: Can you explain the difference between a process and a thread?\nAssistant: A process is an independent instance of a running program with its own memory space, file descriptors, and system resources. Threads, on the other hand, are units of execution within a process that share the process's memory space and resources. Key differences include:",
    # 短 - 列表生成
    "List the top 10 programming languages used today: 1.",
    "Steps to deploy a machine learning model to production: 1.",
    "Five tips for improving the performance of a SQL query: 1.",
    "Three main causes of climate change: 1.",
    # 中 - 论证
    "Whether artificial general intelligence (AGI) is achievable within the next few decades remains a subject of intense debate. Optimists point to the rapid progress of large language models and the increasing scale of compute available. Skeptics argue that current systems",
    "The argument for universal basic income rests on several premises. First, automation is expected to displace a significant portion of the workforce. Second, existing welfare systems are often inefficient and stigmatizing. Third, providing a basic income would give people the freedom to",
    "Open source software has fundamentally changed how the technology industry operates. By making source code freely available, it has enabled rapid innovation, lowered barriers to entry, and created vibrant communities of contributors. However, the open source model also faces challenges, including",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True, help="HF 模型目录")
    p.add_argument("--output", type=str, required=True, help="保存路径，例如 /tmp/sq_scales.pt")
    p.add_argument("--alpha", type=float, default=0.5, help="SmoothQuant alpha (0~1)；典型 0.5–0.85")
    p.add_argument("--num-prompts", type=int, default=96, help="实际使用的 prompt 数（< 内联池大小则截断）")
    # scale 健壮性兜底：极端 channel 可能让 s 冲到 fp16 范围之外，loader 端做 1/s 会 overflow
    p.add_argument("--clamp-min", type=float, default=1e-3, help="s clamp 下界（fp16 friendly）")
    p.add_argument("--clamp-max", type=float, default=1e3,  help="s clamp 上界（fp16 friendly）")
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--max-output-len", type=int, default=1, help="只跑 prefill 即可，1 token 足够")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    return p.parse_args()


def main():
    args = parse_args()
    assert 0.0 <= args.alpha <= 1.0
    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    import torch
    from tinyvllm import LLM, SamplingParams
    from tinyvllm.layers.linear import LinearBase

    # 校准必须 fp16 / 不开 act 假量化 / 不开 cuda graph，且 TP=1
    llm = LLM(
        args.model,
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        quantization=None,
        act_quant_bits=0,
    )
    model = llm.model_runner.model

    # ---- 注册 forward_pre_hook 收集 per-input-channel 激活 absmax ----
    state: dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(name: str):
        def _hook(_mod, inputs):
            x = inputs[0]
            if x.numel() == 0:
                return
            x_flat = x.detach().reshape(-1, x.shape[-1]).float()
            cur = x_flat.abs().amax(dim=0)        # [in_dim]
            prev = state.get(name)
            if prev is None:
                state[name] = cur
            else:
                state[name] = torch.maximum(prev, cur)
        return _hook

    n_hooked = 0
    for name, mod in model.named_modules():
        if isinstance(mod, LinearBase):
            handles.append(mod.register_forward_pre_hook(make_hook(name)))
            n_hooked += 1
    print(f"[calib] hooked {n_hooked} LinearBase modules", flush=True)

    # ---- 跑 prompt bank（仅 prefill）----
    prompts = _PROMPTS[: args.num_prompts]
    print(f"[calib] running {len(prompts)} prompts (prefill only)...", flush=True)
    sps = [SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=args.max_output_len)
           for _ in prompts]
    llm.generate(prompts, sps, use_tqdm=True)

    for h in handles:
        h.remove()

    # ---- 聚合：s = act_max^alpha / w_max^(1-alpha) ----
    print(f"[calib] computing scales with alpha={args.alpha} ...", flush=True)
    scales: dict[str, torch.Tensor] = {}
    alpha = float(args.alpha)
    n_clamp_lo = 0
    n_clamp_hi = 0
    n_total = 0
    for name, mod in model.named_modules():
        if not isinstance(mod, LinearBase):
            continue
        if name not in state:
            print(f"[calib] WARN: no activation captured for {name} (skipped)", flush=True)
            continue
        # 校准 TP=1 → mod.weight 是全维 [out, in_full]，沿 dim=0 (out) 求 absmax
        w = mod.weight.data.detach().float()
        w_max = w.abs().amax(dim=0).clamp_min(1e-5)         # [in_full]
        a_max = state[name].clamp_min(1e-5).cpu()
        s = (a_max.pow(alpha) / w_max.cpu().pow(1.0 - alpha)).to(torch.float32)
        # 健壮性：clamp 到 fp16-friendly 范围；统计被夹的 channel 数
        n_total += s.numel()
        n_clamp_lo += int((s < args.clamp_min).sum().item())
        n_clamp_hi += int((s > args.clamp_max).sum().item())
        s = s.clamp_(args.clamp_min, args.clamp_max)
        # NaN / Inf 兜底（理论上不该出现，但 fp32 边界值乘除可能产生）
        if not torch.isfinite(s).all():
            n_bad = int((~torch.isfinite(s)).sum().item())
            print(f"[calib] WARN: {name} has {n_bad} non-finite scales, replaced with 1.0", flush=True)
            s = torch.where(torch.isfinite(s), s, torch.ones_like(s))
        scales[name] = s

    if n_total > 0:
        print(
            f"[calib] clamp stats: lo={n_clamp_lo}/{n_total} ({100*n_clamp_lo/n_total:.2f}%) "
            f"hi={n_clamp_hi}/{n_total} ({100*n_clamp_hi/n_total:.2f}%)",
            flush=True,
        )

    # ---- 落盘 ----
    bundle = {
        "scales": scales,
        "alpha": alpha,
        "num_prompts": len(prompts),
        "model_path": args.model,
        "clamp_min": args.clamp_min,
        "clamp_max": args.clamp_max,
        "format_version": 1,
    }
    torch.save(bundle, args.output)
    print(f"[calib] saved {len(scales)} scales -> {args.output}", flush=True)

    # 简单 sanity 统计
    if scales:
        all_s = torch.cat([s.flatten() for s in scales.values()])
        print(
            f"[calib] s stats: min={all_s.min().item():.3e} "
            f"median={all_s.median().item():.3e} "
            f"max={all_s.max().item():.3e}",
            flush=True,
        )


if __name__ == "__main__":
    main()
