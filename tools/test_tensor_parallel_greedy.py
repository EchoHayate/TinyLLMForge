from pathlib import Path
import sys
import types

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for package_name in (
    "tinyvllm",
    "tinyvllm.engine",
):
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / package_name.replace(".", "/"))]
    sys.modules.setdefault(package_name, package)

from tinyvllm.engine import tensor_parallel_greedy


select_tensor_parallel_greedy_tokens = (
    tensor_parallel_greedy.select_tensor_parallel_greedy_tokens
)


class _BroadcastBus:

    def __init__(self):
        self.payload = None
        self.root_calls = []
        self.worker_calls = []

    def root(self, tensor, src):
        assert src == 0
        self.payload = tensor.detach().clone()
        self.root_calls.append((
            tensor.dtype,
            tuple(tensor.shape),
            tensor.tolist(),
        ))

    def worker(self, tensor, src):
        assert src == 0
        assert self.payload is not None
        tensor.copy_(self.payload)
        self.worker_calls.append((
            tensor.dtype,
            tuple(tensor.shape),
            tensor.tolist(),
        ))


def _expect_error(function, message):
    try:
        function()
    except (TypeError, ValueError) as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(f"expected error containing {message!r}")


def test_tp1_selects_locally_without_broadcast():
    calls = []
    logits = torch.tensor([
        [0.0, 5.0, 1.0],
        [7.0, 2.0, 3.0],
    ])

    token_ids = select_tensor_parallel_greedy_tokens(
        logits,
        rank=0,
        world_size=1,
        batch_size=2,
        device=logits.device,
        broadcast=lambda *_args, **_kwargs: calls.append("broadcast"),
    )

    assert token_ids.dtype == torch.int64
    assert token_ids.is_contiguous()
    assert token_ids.tolist() == [1, 0]
    assert calls == []


def test_tp4_root_selects_and_workers_receive_only_token_ids():
    bus = _BroadcastBus()
    logits = torch.tensor([
        [0.0, 5.0, 1.0],
        [7.0, 2.0, 3.0],
    ])

    root = select_tensor_parallel_greedy_tokens(
        logits,
        rank=0,
        world_size=4,
        batch_size=2,
        device=logits.device,
        broadcast=bus.root,
    )
    workers = tuple(
        select_tensor_parallel_greedy_tokens(
            None,
            rank=rank,
            world_size=4,
            batch_size=2,
            device=logits.device,
            broadcast=bus.worker,
        )
        for rank in (1, 2, 3)
    )

    assert root.tolist() == [1, 0]
    assert all(torch.equal(worker, root) for worker in workers)
    assert bus.root_calls == [(torch.int64, (2,), [1, 0])]
    assert bus.worker_calls == [
        (torch.int64, (2,), [1, 0]),
        (torch.int64, (2,), [1, 0]),
        (torch.int64, (2,), [1, 0]),
    ]


def test_tp_greedy_broadcast_uses_profile_collective_once(monkeypatch):
    profile_calls = []
    broadcast_calls = []

    def profile_collective(operation, tensor, call, **metadata):
        profile_calls.append((operation, tensor, metadata))
        return call(tensor)

    def broadcast(tensor, src):
        broadcast_calls.append((tensor.clone(), src))

    monkeypatch.setattr(
        tensor_parallel_greedy,
        "profile_collective",
        profile_collective,
    )
    token_ids = select_tensor_parallel_greedy_tokens(
        torch.tensor([[0.0, 2.0]]),
        rank=0,
        world_size=4,
        batch_size=1,
        device=torch.device("cpu"),
        broadcast=broadcast,
    )

    assert token_ids.tolist() == [1]
    assert len(profile_calls) == 1
    assert profile_calls[0][0] == "greedy_token_broadcast"
    assert profile_calls[0][1] is token_ids
    assert profile_calls[0][2] == {
        "site_role": "greedy_token_broadcast",
        "collective_kind": "broadcast",
        "process_group": "tensor_parallel",
        "execution_phase": "decode",
        "async_mode": False,
        "source_rank": 0,
    }
    assert len(broadcast_calls) == 1
    assert broadcast_calls[0][1] == 0


def test_selector_rejects_invalid_topology_and_root_logits():
    logits = torch.ones(2, 3)
    cases = (
        (
            lambda: select_tensor_parallel_greedy_tokens(
                logits,
                rank=0,
                world_size=0,
                batch_size=2,
                device=logits.device,
            ),
            "world_size",
        ),
        (
            lambda: select_tensor_parallel_greedy_tokens(
                logits,
                rank=4,
                world_size=4,
                batch_size=2,
                device=logits.device,
            ),
            "rank",
        ),
        (
            lambda: select_tensor_parallel_greedy_tokens(
                None,
                rank=0,
                world_size=4,
                batch_size=2,
                device=logits.device,
                broadcast=lambda *_args, **_kwargs: None,
            ),
            "root logits",
        ),
        (
            lambda: select_tensor_parallel_greedy_tokens(
                torch.ones(2, 3, dtype=torch.int64),
                rank=0,
                world_size=4,
                batch_size=2,
                device=logits.device,
                broadcast=lambda *_args, **_kwargs: None,
            ),
            "floating",
        ),
        (
            lambda: select_tensor_parallel_greedy_tokens(
                torch.ones(1, 3),
                rank=0,
                world_size=4,
                batch_size=2,
                device=logits.device,
                broadcast=lambda *_args, **_kwargs: None,
            ),
            "row count",
        ),
        (
            lambda: select_tensor_parallel_greedy_tokens(
                torch.ones(2, 3, 1),
                rank=0,
                world_size=4,
                batch_size=2,
                device=logits.device,
                broadcast=lambda *_args, **_kwargs: None,
            ),
            "rank two",
        ),
    )
    for function, message in cases:
        _expect_error(function, message)


def test_selector_rejects_worker_logits_and_malformed_broadcast_result():
    logits = torch.ones(1, 3)
    _expect_error(
        lambda: select_tensor_parallel_greedy_tokens(
            logits,
            rank=1,
            world_size=4,
            batch_size=1,
            device=logits.device,
            broadcast=lambda *_args, **_kwargs: None,
        ),
        "non-root logits",
    )

    def write_negative(tensor, src):
        assert src == 0
        tensor.fill_(-1)

    _expect_error(
        lambda: select_tensor_parallel_greedy_tokens(
            None,
            rank=1,
            world_size=4,
            batch_size=1,
            device=logits.device,
            broadcast=write_negative,
        ),
        "nonnegative",
    )


def main():
    test_tp1_selects_locally_without_broadcast()
    test_tp4_root_selects_and_workers_receive_only_token_ids()
    test_selector_rejects_invalid_topology_and_root_logits()
    test_selector_rejects_worker_logits_and_malformed_broadcast_result()
    print("tensor parallel greedy tests passed")


if __name__ == "__main__":
    main()
