import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open

def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    # 获取模型中的 packed_modules_mapping 属性，如果没有，那么返回空字典
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                # 如果k是压缩过的，那么需要从packed_modules_mapping中找到完整的，
                # 将weight_name中压缩过的k替换成完整的，才是正确的参数名
                for k in packed_modules_mapping:
                    if k in weight_name:                                        # shared_id 是因为模型是GQA, 一组q共享kv
                        v, shared_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shared_id)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))

                    
