from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

import torch


def fingerprint_candidate_inventory(candidates) -> str:
    if not isinstance(candidates, tuple) or not candidates:
        raise ValueError(
            "trace candidates must be a non-empty tuple"
        )
    digest = hashlib.sha256()
    for layer_index, pair in enumerate(candidates):
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError(
                "trace candidate must contain a state pair"
            )
        for state_name, tensor in zip(
            ("convolution", "recurrent"),
            pair,
        ):
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(
                    "trace candidate state must be a tensor"
                )
            contiguous = tensor.detach().contiguous()
            header = {
                "layer_index": layer_index,
                "state_name": state_name,
                "dtype": str(contiguous.dtype),
                "shape": list(contiguous.shape),
            }
            digest.update(
                json.dumps(
                    header,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            digest.update(
                contiguous.view(torch.uint8)
                .cpu()
                .numpy()
                .tobytes()
            )
    return digest.hexdigest()


@dataclass(frozen=True)
class Qwen35SideStateTraceRow:
    sequence_id: int
    event: str
    checkpoint_index: int
    committed_input_count: int | None
    fingerprint: str

    def as_dict(self) -> dict:
        return asdict(self)


class Qwen35SpeculativeTraceRecorder:
    def __init__(self):
        self._enabled = False
        self._rows = []

    @property
    def enabled(self) -> bool:
        return self._enabled

    def enable(self, enabled: bool) -> dict:
        if not isinstance(enabled, bool):
            raise ValueError("trace enabled must be a boolean")
        self._enabled = enabled
        self._rows.clear()
        return {"enabled": enabled}

    def record_checkpoint(
        self,
        *,
        sequence_id: int,
        event: str,
        checkpoint_index: int,
        candidates,
    ) -> None:
        if not self._enabled:
            return
        if event not in (
            "first_target_checkpoint",
            "tail_checkpoint",
        ):
            raise ValueError(
                "trace checkpoint event is invalid"
            )
        if event == "first_target_checkpoint":
            if checkpoint_index != 1:
                raise ValueError(
                    "first-target checkpoint index must be one"
                )
        elif checkpoint_index < 2:
            raise ValueError(
                "tail checkpoint index must start at two"
            )
        self._rows.append(Qwen35SideStateTraceRow(
            sequence_id=int(sequence_id),
            event=event,
            checkpoint_index=int(checkpoint_index),
            committed_input_count=None,
            fingerprint=fingerprint_candidate_inventory(
                candidates
            ),
        ))

    def record_selection(
        self,
        *,
        sequence_id: int,
        committed_input_count: int,
        candidates,
    ) -> None:
        if not self._enabled:
            return
        if committed_input_count <= 0:
            raise ValueError(
                "selected checkpoint index must be positive"
            )
        self._rows.append(Qwen35SideStateTraceRow(
            sequence_id=int(sequence_id),
            event="selected_checkpoint",
            checkpoint_index=int(committed_input_count),
            committed_input_count=int(
                committed_input_count
            ),
            fingerprint=fingerprint_candidate_inventory(
                candidates
            ),
        ))

    def drain(self) -> tuple[dict, ...]:
        rows = tuple(row.as_dict() for row in self._rows)
        self._rows.clear()
        return rows
