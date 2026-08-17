"""
CPU-Offload：以 decoder layer 为粒度，把指定数量的层权重常驻 CPU(pinned memory)，
forward 时按需 swap 到 GPU。

设计：
    - 在 ModelRunner 构建并加载完模型后调用 apply_cpu_offload(model, num_layers)
    - 选择前 num_layers 层（即靠近 embedding 端）卸载到 cpu，剩余层留 GPU
    - 通过 forward_pre_hook / forward_hook 实现 透明的 H2D / D2H 拷贝
    - 使用一个独立的 cuda prefetch_stream，让 layer i+1 的 H2D 拷贝
      与 layer i 的 compute overlap
    - cpu pinned master tensor 永久驻留，evict 仅做 D2H copy_，避免重复 pin_memory

注意：
    - 量化后的层卸载的是 qweight / scales buffer，对外行为一致。
    - 第一次访问某层时，prefetch_stream 会同步发起 H2D（无 overlap 收益），
      之后 layer i 的 pre_hook 会顺手把 layer i+1 的 H2D 也提交，从而后续每层都能 overlap。
    - 设置环境变量 TLLM_OFFLOAD_NO_PREFETCH=1 可关闭 prefetch（用于 A/B 对比）。
"""

import os
import torch
from torch import nn

_NO_PREFETCH = os.environ.get("TLLM_OFFLOAD_NO_PREFETCH", "0") == "1"


def _iter_storage_tensors(layer: nn.Module):
    """遍历 layer 中所有 weight / qweight / scales / bias 张量。"""
    for name, p in layer.named_parameters(recurse=True):
        if p is not None:
            yield (name, p, "param")
    for name, b in layer.named_buffers(recurse=True):
        if b is not None and (name.endswith("qweight") or name.endswith("scales")):
            yield (name, b, "buffer")


class _PrefetchOffloadManager:
    """统一管理一组 offloaded decoder layer 的 swap-in / swap-out。

    核心思路：
      - 为每个 (layer, tensor) 维护一个 pinned-cpu master，永久驻留
      - prefetch_stream 上发起 H2D：master.to(gpu) 产出新的 gpu tensor
        并通过 t.data = gpu_t 让该层 forward 看到 GPU 权重
      - compute_stream.wait_event 等 H2D 完成，再开始本层 forward
      - 在 layer i 的 pre_hook 中提前发起 layer i+1 的 H2D，与 i 的 compute overlap
      - 在 post_hook 中以 master.copy_(gpu_t, non_blocking=True) 做 D2H，
        然后 t.data = master 释放 gpu tensor 引用
    """

    def __init__(self, layers: list[nn.Module], gpu_device: torch.device):
        self.layers = layers
        self.gpu_device = gpu_device
        self.prefetch_stream = torch.cuda.Stream(device=gpu_device)
        # entries[i] = list of (param_or_buffer, cpu_master_tensor)
        self.entries: list[list[tuple[torch.Tensor, torch.Tensor]]] = []
        self.gpu_resident: list[bool] = [False] * len(layers)
        self.prefetch_done: list[torch.cuda.Event | None] = [None] * len(layers)
        # 记录 evict 完成事件，确保下次 fetch 等到 master 写完
        self.evict_done: list[torch.cuda.Event | None] = [None] * len(layers)

        for layer in layers:
            layer_entries: list[tuple[torch.Tensor, torch.Tensor]] = []
            for _, t, _ in _iter_storage_tensors(layer):
                # init 阶段一次性 D2H（non_blocking + 末尾 sync），减少多层叠加同步开销
                cpu_master = torch.empty_like(t.data, device="cpu", pin_memory=True)
                cpu_master.copy_(t.data, non_blocking=True)
                t.data = cpu_master
                layer_entries.append((t, cpu_master))
            self.entries.append(layer_entries)
        # 等所有 init 期 D2H 完成
        torch.cuda.synchronize(gpu_device)

        for idx, layer in enumerate(layers):
            layer.register_forward_pre_hook(self._make_pre_hook(idx))
            layer.register_forward_hook(self._make_post_hook(idx))

    def _fetch_async(self, idx: int) -> None:
        """在 prefetch_stream 上发起第 idx 层的 H2D，并记录完成事件。"""
        if self.gpu_resident[idx]:
            return
        # 让 prefetch_stream 等 compute_stream 上当前已发起的工作（含可能未发起 evict_done event 的旧路径）
        self.prefetch_stream.wait_stream(torch.cuda.current_stream())
        # 显式等同一层上次 evict 的完成事件（保险，跨 step 也成立）
        evict_evt = self.evict_done[idx]
        if evict_evt is not None:
            self.prefetch_stream.wait_event(evict_evt)
            self.evict_done[idx] = None
        with torch.cuda.stream(self.prefetch_stream):
            for t, cpu_master in self.entries[idx]:
                gpu_t = cpu_master.to(self.gpu_device, non_blocking=True)
                t.data = gpu_t
            evt = torch.cuda.Event()
            evt.record(self.prefetch_stream)
        self.prefetch_done[idx] = evt
        self.gpu_resident[idx] = True

    def _evict(self, idx: int) -> None:
        """把第 idx 层的 GPU 权重拷回 CPU master，释放 GPU 内存。"""
        if not self.gpu_resident[idx]:
            return
        compute_stream = torch.cuda.current_stream()
        for t, cpu_master in self.entries[idx]:
            cpu_master.copy_(t.data, non_blocking=True)
            t.data = cpu_master
        # 记录 D2H 完成事件，下次 fetch 同一层时显式等它
        evt = torch.cuda.Event()
        evt.record(compute_stream)
        self.evict_done[idx] = evt
        self.gpu_resident[idx] = False
        self.prefetch_done[idx] = None

    def _make_pre_hook(self, idx: int):
        def pre_hook(module, inputs):
            # 1) 确保本层已经发起 H2D（首层 fallback；后续层一般已被上一层的 pre_hook 提前预取）
            if not self.gpu_resident[idx]:
                self._fetch_async(idx)
            evt = self.prefetch_done[idx]
            if evt is not None:
                torch.cuda.current_stream().wait_event(evt)
            # 2) 提前发起下一层的 H2D，与本层 compute overlap（可被环境变量关闭，用于 A/B）
            if _NO_PREFETCH:
                return
            nxt = idx + 1
            if nxt < len(self.layers) and not self.gpu_resident[nxt]:
                self._fetch_async(nxt)
        return pre_hook

    def _make_post_hook(self, idx: int):
        def post_hook(module, inputs, output):
            # D2H 在 compute_stream 上发起；与下一层的 compute 自然串行不影响 overlap
            self._evict(idx)
            return output
        return post_hook


def apply_cpu_offload(model: nn.Module, num_offload_layers: int,
                      gpu_device: torch.device | None = None):
    """对 model.model.layers 的前 num_offload_layers 层启用 cpu-offload (with prefetch).

    Args:
        model: Qwen3ForCausalLM 实例（含 model.layers）
        num_offload_layers: 卸载层数；-1 表示除最后 2 层外全部卸载
        gpu_device: 计算设备（默认当前 cuda device）
    """
    if num_offload_layers == 0:
        return None
    if gpu_device is None:
        gpu_device = torch.device(f"cuda:{torch.cuda.current_device()}")

    layers = model.model.layers
    total = len(layers)
    if num_offload_layers < 0:
        num_offload_layers = max(0, total - 2)
    num_offload_layers = min(num_offload_layers, total)
    if num_offload_layers == 0:
        return None

    offloaded = [layers[i] for i in range(num_offload_layers)]
    mgr = _PrefetchOffloadManager(offloaded, gpu_device)
    # 把 manager 挂在 model 上，便于调试 / 防 GC
    model._tllm_cpu_offload_mgr = mgr
    return mgr
