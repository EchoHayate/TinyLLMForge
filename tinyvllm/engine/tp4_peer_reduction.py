from __future__ import annotations

from pathlib import Path

import torch


DEFAULT_TIMEOUT_CLOCKS = 2_000_000_000
_EXTENSION = None


def _load_extension():
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION
    from torch.utils.cpp_extension import load

    source_root = Path(__file__).with_name("csrc")
    _EXTENSION = load(
        name="tinyllmforge_tp4_peer_reduction",
        sources=[
            str(source_root / "tp4_peer_reduction.cpp"),
            str(source_root / "tp4_peer_reduction_kernel.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=[
            "-O3",
            "-lineinfo",
        ],
        with_cuda=True,
        verbose=False,
    )
    return _EXTENSION


def _is_cuda_device(device):
    return getattr(device, "type", None) == "cuda"


class TP4PeerReductionGroup:
    def __init__(
        self,
        *,
        rank,
        device,
        layer_count,
        max_active_tokens,
        hidden_size,
        timeout_clocks,
        extension,
        torch_module,
        local_slots,
        local_flags,
        output_slots,
        status,
        peer_mappings,
    ):
        self.rank = rank
        self.device = device
        self.layer_count = layer_count
        self.max_active_tokens = max_active_tokens
        self.hidden_size = hidden_size
        self.timeout_clocks = timeout_clocks
        self._extension = extension
        self._torch = torch_module
        self._local_slots = local_slots
        self._local_flags = local_flags
        self._output_slots = output_slots
        self._status = status
        self._peer_mappings = tuple(peer_mappings)
        self.state = "READY"

    @classmethod
    def create(
        cls,
        *,
        rank,
        world_size,
        device,
        layer_count,
        max_active_tokens,
        hidden_size,
        timeout_clocks=DEFAULT_TIMEOUT_CLOCKS,
        extension=None,
        distributed=None,
        torch_module=None,
    ):
        if world_size != 4:
            raise ValueError("world_size must be 4")
        if type(rank) is not int or rank not in range(world_size):
            raise ValueError("rank must be in [0, world_size)")
        if not _is_cuda_device(device):
            raise ValueError("device must be CUDA")
        for name, value in (
            ("layer_count", layer_count),
            ("max_active_tokens", max_active_tokens),
            ("hidden_size", hidden_size),
            ("timeout_clocks", timeout_clocks),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

        torch_api = torch if torch_module is None else torch_module
        if distributed is None:
            import torch.distributed as distributed
        extension_api = (
            _load_extension() if extension is None else extension
        )
        local_slots = torch_api.empty(
            (
                layer_count,
                2,
                max_active_tokens,
                hidden_size,
            ),
            dtype=torch_api.float32,
            device=device,
        )
        local_flags = torch_api.zeros(
            (layer_count, 2),
            dtype=torch_api.uint64,
            device=device,
        )
        output_slots = torch_api.empty(
            (
                layer_count,
                2,
                max_active_tokens,
                hidden_size,
            ),
            dtype=torch_api.bfloat16,
            device=device,
        )
        status = torch_api.zeros(
            (1,),
            dtype=torch_api.int32,
            device=device,
        )
        peer_mappings = []
        try:
            payload = {
                "rank": rank,
                "slot_handle": extension_api.export_ipc_handle(
                    local_slots
                ),
                "flag_handle": extension_api.export_ipc_handle(
                    local_flags
                ),
            }
            gathered = [None] * world_size
            distributed.all_gather_object(gathered, payload)
            if {
                row.get("rank")
                for row in gathered
                if isinstance(row, dict)
            } != set(range(world_size)):
                raise RuntimeError("peer handle exchange is incomplete")
            by_rank = {row["rank"]: row for row in gathered}
            for peer_rank in range(world_size):
                if peer_rank == rank:
                    continue
                row = by_rank[peer_rank]
                peer_mappings.append(extension_api.open_mapping(
                    row["slot_handle"],
                    row["flag_handle"],
                    tuple(local_slots.shape),
                    tuple(local_flags.shape),
                    peer_rank,
                ))
        except BaseException:
            for mapping in reversed(peer_mappings):
                extension_api.close_mapping(mapping)
            extension_api.release_owned()
            raise
        return cls(
            rank=rank,
            device=device,
            layer_count=layer_count,
            max_active_tokens=max_active_tokens,
            hidden_size=hidden_size,
            timeout_clocks=timeout_clocks,
            extension=extension_api,
            torch_module=torch_api,
            local_slots=local_slots,
            local_flags=local_flags,
            output_slots=output_slots,
            status=status,
            peer_mappings=peer_mappings,
        )

    def _require_ready(self):
        if self.state != "READY":
            raise RuntimeError(
                f"peer reduction group is {self.state}"
            )

    def _validate_tensor(self, tensor, *, name, dtype):
        if (
            getattr(tensor, "ndim", None) != 2
            or tuple(getattr(tensor, "shape", ()))[:-1] == ()
            or tensor.shape[-1] != self.hidden_size
            or getattr(tensor, "dtype", None) != dtype
            or getattr(tensor, "device", None) != self.device
            or not tensor.is_contiguous()
        ):
            raise ValueError(f"{name} has invalid shape, dtype, or device")

    def reduce_add_residual(
        self,
        *,
        layer_index,
        generation,
        local_partial,
        residual,
    ):
        self._require_ready()
        if (
            type(layer_index) is not int
            or layer_index not in range(self.layer_count)
        ):
            raise ValueError("layer_index is out of range")
        if type(generation) is not int or generation <= 0:
            raise ValueError("generation must be a positive integer")
        self._validate_tensor(
            local_partial,
            name="local_partial",
            dtype=self._torch.float32,
        )
        self._validate_tensor(
            residual,
            name="residual",
            dtype=self._torch.bfloat16,
        )
        if tuple(local_partial.shape) != tuple(residual.shape):
            raise ValueError("local_partial and residual shapes differ")
        active_tokens = local_partial.shape[0]
        if not 1 <= active_tokens <= self.max_active_tokens:
            raise ValueError("active token count is out of range")

        slot_index = generation % 2
        local_slot = self._local_slots[
            layer_index,
            slot_index,
            :active_tokens,
        ]
        output = self._output_slots[
            layer_index,
            slot_index,
            :active_tokens,
        ]
        with self._torch.inference_mode():
            local_slot.copy_(local_partial)
            stream = self._torch.cuda.current_stream(
                self.device
            ).cuda_stream
            self._extension.publish(
                local_slot,
                self._local_flags,
                layer_index,
                slot_index,
                generation,
                stream,
            )
            status = self._extension.reduce_add_residual(
                self._peer_mappings,
                local_slot,
                self._local_flags,
                residual,
                output,
                self._status,
                self.rank,
                layer_index,
                slot_index,
                generation,
                active_tokens,
                self.hidden_size,
                self.timeout_clocks,
                stream,
            )
        if status != 0:
            self.state = "POISONED"
            raise RuntimeError("peer reduction launch failed")
        return output

    def check_status(self):
        self._require_ready()
        if int(self._status.item()) != 0:
            self.state = "POISONED"
            raise RuntimeError("peer reduction timed out")
        return 0

    def close(self):
        if self.state == "CLOSED":
            return
        for mapping in self._peer_mappings:
            self._extension.close_mapping(mapping)
        self._extension.release_owned()
        self._peer_mappings = ()
        self.state = "CLOSED"

    def __enter__(self):
        self._require_ready()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
