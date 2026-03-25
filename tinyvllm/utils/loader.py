import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open
def default_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor):    #copy to parm from loaded_weight
    param_data = param.data if hasattr(param, "data") else param
    param_data.copy_(loaded_weight)     


def load_model(model: nn.Module, path: str):
    # 获取模型中的 packed_modules_mapping 属性，如果没有，那么返回空字典
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:     #先加载到cpu再gpu 防止vram溢出   不管什么并行 都是先读取到cpu 再分配到gpu上
            for weight_name in f.keys():
                # For quantization, weight_name can be `layer.q_proj.weight` or `layer.q_proj.weight_scale`
                # So we extract the base projection name to match with packed_modules_mapping
                base_name = weight_name.replace(".weight_scale", "").replace(".weight", "")
                
                # 如果k是压缩过的，那么需要从packed_modules_mapping中找到完整的，
                # 将weight_name中压缩过的k替换成完整的，才是正确的参数名
                mapped = False
                for k in packed_modules_mapping: #匹配packed_modules_mapping和safetensor里的的key
                    if k in weight_name:                                        # shared_id 是因为模型是GQA, 一组q共享kv
                        v, shared_id = packed_modules_mapping[k]        
                        param_name = weight_name.replace(k, v)          #e.g. qkv_proj替换q_proj
                        
                        try:
                            # Use dict access for buffers as get_parameter might fail for buffers
                            param = dict(model.named_parameters(recurse=True)).get(param_name)
                            if param is None:
                                param = dict(model.named_buffers(recurse=True)).get(param_name)
                                
                            if param is not None:
                                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                                weight_loader(param, f.get_tensor(weight_name), shared_id)
                                mapped = True
                                break
                        except AttributeError:
                            pass
                
                if not mapped:
                    param = dict(model.named_parameters(recurse=True)).get(weight_name)
                    if param is None:
                        param = dict(model.named_buffers(recurse=True)).get(weight_name)
                        
                    if param is not None:
                        weight_loader = getattr(param, "weight_loader", default_weight_loader)
                        weight_loader(param, f.get_tensor(weight_name))


                    
