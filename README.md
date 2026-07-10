# TinyLLMForge
build a tiny LLM engine from scratch

- [] apply APC(automatic prefix caching)

## 实验收益概览

从 2026-05 以来，本项目围绕 Qwen3-0.6B / Qwen3-8B 做了多轮推理引擎实验，当前比较明确的收益如下：

- **CUDA Graph decode**：0.6B decode 场景下主要收益来自减少 launch overhead，实测约 **8–9×** 吞吐提升。
- **Quest fused score kernel**：Quest score-only 微基准约 **9–22×** 加速；端到端 needle 16K 里，在 100% 召回下约 **+4.5%~6.8% TPS**。
- **W4A8 + SmoothQuant + skip-last**：把 8B 长上下文量化路径从不可用/低召回救到约 **96%+ needle 召回**，关键结论是尾部层比首部层更值得保留 fp activation。
- **KV8 + Quest full-stack**：KV cache 8bit 量化节省约 **50% KV 显存**；叠加 Quest sparse-dequant 后，在高召回配置下约 **+33% TPS**，高吞吐折中档约 **+46% TPS**。
- **n-gram speculative decoding**：在重复 prompt / 高 acceptance / 模拟 128MiB H2D upload 的长上下文场景中，wall-clock 约 **2.04×**，模拟 upload 成本约 **-54%**；该收益依赖场景，暂不作为通用默认路径。
- **KV offload / blockwise attention**：已打通 GPU staging、dirty writeback、H2D reload、prefetch/eviction 与 blockwise attention 正确性；局部 `gpu_blocks=4` matrix 中 H2D / eviction 计数约 **-33%**，但端到端 tok/s 收益仍需更严格 benchmark 证明。
- **DFlash profiler-only**：已完成 hidden-to-draft / draft-model-stub ABI、batch schema 与 contract 验证；当前不接 runtime，因此对现有推理速度 **0% 直接收益**，主要价值是降低未来接真实 draft model 的风险。

## 参考资料
```
https://github.com/GeeeekExplorer/nano-vllm
https://space.bilibili.com/362867186?spm_id_from=333.788.upinfo.detail.click
https://zhuanlan.zhihu.com/p/1932035278089987994
https://zhuanlan.zhihu.com/p/1932473745584394614
https://zhuanlan.zhihu.com/p/1925484783229698084
