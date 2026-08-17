import os

import torch


def compile_if_enabled(*args, **kwargs):
    if os.environ.get("TORCH_COMPILE_DISABLE") == "1":
        return lambda function: function
    return torch.compile(*args, **kwargs)
