from pathlib import Path
import traceback

import torch.distributed as distributed
from torch.distributed.distributed_c10d import _get_default_group

from tinyvllm.engine.llm_engine import LLMEngine


_original_init = LLMEngine.__init__


def _diagnostic_init(self, model, **config_kwargs):
    config_kwargs["num_kvcache_blocks"] = 64
    return _original_init(self, model, **config_kwargs)


LLMEngine.__init__ = _diagnostic_init


_trace_directory = Path(
    "/data00/home/sitian/sitian-workspace01/tllm/"
    "qwen35-native-mtp-tp4-4k-engine-runs/"
    "opaque-00c220c1f7917a499af12f18/"
    "diagnostic-q4e/sequence-traces"
)
_trace_directory.mkdir(parents=True, exist_ok=True)
_trace_index = 0


def _sequence_number():
    if not distributed.is_initialized():
        return -1
    return int(_get_default_group()._get_sequence_number_for_group())


def _install_trace(name):
    operation = getattr(distributed, name, None)
    if not callable(operation):
        return

    def traced(*args, **kwargs):
        global _trace_index
        rank = (
            distributed.get_rank()
            if distributed.is_initialized()
            else -1
        )
        index = _trace_index
        _trace_index += 1
        before = _sequence_number()
        stack = traceback.extract_stack(limit=7)[:-1]
        location = " <- ".join(
            f"{Path(frame.filename).name}:{frame.lineno}:{frame.name}"
            for frame in stack[-4:]
        )
        trace_path = _trace_directory / f"rank{rank}.log"
        with trace_path.open("a", encoding="utf-8") as trace:
            trace.write(
                f"{index:04d} BEGIN {name} seq={before} "
                f"{location}\n"
            )
            trace.flush()
        result = operation(*args, **kwargs)
        after = _sequence_number()
        with trace_path.open("a", encoding="utf-8") as trace:
            trace.write(
                f"{index:04d} END {name} seq={after}\n"
            )
            trace.flush()
        return result

    setattr(distributed, name, traced)


for _collective_name in (
    "all_gather",
    "all_gather_into_tensor",
    "all_gather_object",
    "all_reduce",
    "all_to_all",
    "all_to_all_single",
    "barrier",
    "broadcast",
    "broadcast_object_list",
    "gather",
    "gather_object",
    "recv",
    "reduce",
    "reduce_scatter",
    "reduce_scatter_tensor",
    "scatter",
    "send",
):
    _install_trace(_collective_name)
