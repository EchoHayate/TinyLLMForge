from tinyvllm.models.qwen3_vl import Qwen3VLModel, Qwen2VLVisionConfig

# Qwen2.5-VL shares the exact same architecture as Qwen2-VL
# We use the existing storage in qwen3-vl.py but provide proper class names for registry.

class Qwen2_5_VL_Model(Qwen3VLModel):
    """
    Qwen2.5-VL model wrapper.
    Architecture is identical to Qwen2-VL.
    """
    def __init__(self, config, vision_config):
        super().__init__(config, vision_config)

class Qwen2_5_VLVisionConfig(Qwen2VLVisionConfig):
    model_type = "qwen2_5_vl_vision"
