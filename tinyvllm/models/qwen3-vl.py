from tinyvllm.models.qwen3 import Qwen3Model      
from tinyvllm.modules.attention import Attention
from tinyvllm.modules.linear import Linear
import torch

class Qwen3VLModel(Qwen3Model):
    def quick_sort_in_place(nums,left,right):
        
        