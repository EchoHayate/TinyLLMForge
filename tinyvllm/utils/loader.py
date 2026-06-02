import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open

from tinyvllm.layers.linear import LinearBase, RowParallelLinear, get_quant_method


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):    #copy to parm from loaded_weight
    param.data.copy_(loaded_weight)     


def _apply_smoothquant_scales(model: nn.Module, scales: dict):
    """SmoothQuant 注入：把 per-input-channel scale s 乘进 weight，并把 s 注册成 buffer 给 forward 用。

    必须在 weight load 之后、finalize_quantization 之前调用 —— 后者会把 self.weight 设为 None。

    TP 语义：
      - ColumnParallel / Replicated / QKV / Merged：input 维全量，s 全量、所有 rank 相同
      - RowParallel：input 维已分片，s 按 tp_rank narrow 出本 rank 的 [in_local] 段
    """
    n_applied = 0
    n_skipped_no_scale = 0
    for name, mod in model.named_modules():
        if not isinstance(mod, LinearBase):
            continue
        if name not in scales:
            n_skipped_no_scale += 1
            continue
        s_full = scales[name].to(torch.float32)
        # 健壮性：scale shape / 值域校验，提前失败比 finalize 后崩好
        if not torch.isfinite(s_full).all():
            raise ValueError(
                f"[smoothquant] scale for {name} contains NaN/Inf "
                f"(re-run calibration with proper --clamp-min/--clamp-max)"
            )
        if isinstance(mod, RowParallelLinear):
            in_local = mod.input_size_per_partition
            expected_in = in_local * mod.tp_size
            if s_full.numel() != expected_in:
                raise ValueError(
                    f"[smoothquant] {name}: scale len {s_full.numel()} != "
                    f"expected input dim {expected_in} (in_local={in_local} × tp={mod.tp_size})"
                )
            s_local = s_full.narrow(0, mod.tp_rank * in_local, in_local)
        else:
            if s_full.numel() != mod.weight.shape[1]:
                raise ValueError(
                    f"[smoothquant] {name}: scale len {s_full.numel()} != "
                    f"weight in-dim {mod.weight.shape[1]}"
                )
            s_local = s_full
        # 与 weight 同 device / dtype，避免 inplace mul 触发隐式转换
        s_dev = s_local.to(mod.weight.device, dtype=mod.weight.dtype)
        # W' = W * s，沿 axis=1 (input-channel) broadcast；注意 weight shape = [out, in]
        mod.weight.data.mul_(s_dev.view(1, -1))
        # 持久化为 buffer：persistent=False 不进 state_dict，且不会被 finalize_quantization 清掉
        mod.register_buffer("smooth_scale", s_dev, persistent=False)
        n_applied += 1
    # 至少要注入到一个模块，否则 scale 文件多半是错配的 model
    if n_applied == 0:
        raise ValueError(
            f"[smoothquant] no scales matched any LinearBase module "
            f"(scales keys sample: {list(scales.keys())[:3]}); model/scale mismatch"
        )
    print(
        f"[smoothquant] applied scales to {n_applied} modules "
        f"(skipped {n_skipped_no_scale} without matching key)",
        flush=True,
    )


def disable_act_quant_in_layers(model: nn.Module, skip_first: int, skip_last: int,
                                skip_layers: list[int] | None = None):
    """把指定 decoder 层 LinearBase 的 act_quant_bits 设为 0。

    层选择 = 首 skip_first 层 ∪ 末 skip_last 层 ∪ skip_layers（显式列表）。

    模块名规则按 Qwen3 系：`model.layers.{idx}.<sub>`，靠正则抽 idx。
    必须在 finalize_quantization 之前（act_quant_bits 是 forward 时读取的实例属性，
    finalize 不影响它，但放在 SQ 之后、量化之前最自然）。

    设计动机：W4A8+SQ 长文复读塌方的根因在 outlier 极端层，per-token A8 量化
    被这些层撑爆，整段 logits 噪声放大；让这些层保 fp16 激活、其余走 A8 能在
    几乎不动 TPS 的前提下大幅修复长文召回。

    skip_layers 用于"按 outlier 强度精准 skip"（诊断见 tools/diag_layer_outlier.py）：
    Qwen3-8B 上 L6 是 amax≈5952 的极端 outlier 层、尾部 L31-35 递增，首尾对称 skip
    会漏掉 L6 / 浪费在干净的 L0 上。详见 docs/qwen3-8b-fixes.md §28。
    """
    explicit = set(skip_layers) if skip_layers else set()
    if skip_first <= 0 and skip_last <= 0 and not explicit:
        return
    import re
    layer_re = re.compile(r"\.layers\.(\d+)\.")
    # 先扫一遍拿全 layer 数（取最大 idx + 1）
    max_idx = -1
    for name, _ in model.named_modules():
        m = layer_re.search(name)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    if max_idx < 0:
        # 找不到 layers，安静返回（lm_head / embedding 等非 transformer 块场景）
        return
    num_layers = max_idx + 1
    skip_set = set(range(min(skip_first, num_layers))) | \
               set(range(max(0, num_layers - skip_last), num_layers)) | \
               {i for i in explicit if 0 <= i < num_layers}

    n_disabled = 0
    for name, mod in model.named_modules():
        if not isinstance(mod, LinearBase):
            continue
        m = layer_re.search(name)
        if not m:
            continue
        idx = int(m.group(1))
        if idx in skip_set and getattr(mod, "act_quant_bits", 0) != 0:
            mod.act_quant_bits = 0
            n_disabled += 1
    print(
        f"[act-quant-skip] disabled A8 on {n_disabled} LinearBase modules "
        f"(layers={sorted(skip_set)}, total={num_layers})",
        flush=True,
    )


def load_model(model: nn.Module, path: str, smoothquant_scale_path: str | None = None,
               act_quant_skip_first: int = 0, act_quant_skip_last: int = 0,
               act_quant_skip_layers: list[int] | None = None):
    # 获取模型中的 packed_modules_mapping 属性，如果没有，那么返回空字典
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:     #先加载到cpu再gpu 防止vram溢出   不管什么并行 都是先读取到cpu 再分配到gpu上
            for weight_name in f.keys():
                # 如果k是压缩过的，那么需要从packed_modules_mapping中找到完整的，
                # 将weight_name中压缩过的k替换成完整的，才是正确的参数名
                for k in packed_modules_mapping: #匹配packed_modules_mapping和safetensor里的的key
                    if k in weight_name:                                        # shared_id 是因为模型是GQA, 一组q共享kv
                        v, shared_id = packed_modules_mapping[k]        
                        param_name = weight_name.replace(k, v)          #e.g. qkv_proj替换q_proj
                        param = model.get_parameter(param_name)         #通过这种方式实现了 3×[hidden_size, hidden_size] -> [3×hidden_size, hidden_size]
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shared_id)
                        break
                else:
                    param = model.get_parameter(weight_name)        #按名字找到模型中需要赋值的‘容器’ 获取模型中的参数对象
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)  # 获取参数的加载方法
                    weight_loader(param, f.get_tensor(weight_name))     #f.get_tensor按键取值

    # SmoothQuant：在 fp 权重已就位、量化未发生的窗口里把 s 迁移到 weight
    if smoothquant_scale_path is not None:
        bundle = torch.load(smoothquant_scale_path, map_location="cpu")
        scales = bundle["scales"] if isinstance(bundle, dict) and "scales" in bundle else bundle
        _apply_smoothquant_scales(model, scales)

    # 关 A8 的指定层：必须在 finalize 之前（虽然 finalize 不动 act_quant_bits，
    # 但和 SQ 一样属于"权重就位、forward 还没真跑"的注入窗口）
    disable_act_quant_in_layers(model, act_quant_skip_first, act_quant_skip_last,
                                act_quant_skip_layers)

    # 加载完整 fp 权重后，对所有线性层执行量化（如开启）
    if get_quant_method() is not None:
        for module in model.modules():
            if isinstance(module, LinearBase):
                module.finalize_quantization()
        torch.cuda.empty_cache()

