import os
import sys
import threading

from pilot_protocol import require


ENVIRONMENT = {
    "USE_TORCH": "0", "USE_TF": "0", "USE_FLAX": "0",
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1", "TOKENIZERS_PARALLELISM": "false",
    "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    "CUDA_VISIBLE_DEVICES": "",
}
COLD_STACK = {"transformers", "tokenizers", "huggingface_hub", "jinja2",
              "torch", "tensorflow", "jax", "jaxlib", "flax"}
_USED = False


class RuntimeGuard:
    """One-use tokenizer environment checks, not a model-execution sandbox."""

    def __init__(self):
        self.active = False
        self.failed = False

    def start(self):
        global _USED
        require(threading.current_thread() is threading.main_thread(), "main thread required")
        require(not _USED, "runtime attempt already consumed")
        _USED = True
        require(not any(name.split(".")[0] in COLD_STACK
                        for name in sys.modules), "cold tokenizer process required")
        os.environ.update(ENVIRONMENT)
        self.active = True
        self.check()

    def check(self):
        try:
            require(self.active and not self.failed, "runtime guard inactive or failed")
            require(threading.current_thread() is threading.main_thread(), "main thread required")
            require(all(os.environ.get(name) == value for name, value in ENVIRONMENT.items()),
                    "runtime environment changed")
        except BaseException:
            self.failed = True
            raise

    def close(self):
        self.active = False
