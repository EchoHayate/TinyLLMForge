from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import sys
import types
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_MODULE_NAMES = (
    "tinyvllm",
    "tinyvllm.engine",
    "tinyvllm.speculative",
    "tinyvllm.utils",
    "torch",
    "tinyvllm.engine.proposal_kv_cache",
    "tinyvllm.engine.proposal_kv_lifecycle",
    "tinyvllm.engine.speculative_proposal_executor",
    "tinyvllm.engine.tensor_parallel_greedy",
    "tinyvllm.speculative.adapter",
    "tinyvllm.utils.context",
    "tinyvllm.engine.qwen35_mtp_executor",
)
_MISSING = object()
_ORIGINAL_MODULES = {
    name: sys.modules.get(name, _MISSING)
    for name in _MODULE_NAMES
}
for package_name in _MODULE_NAMES[:4]:
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / package_name.replace(".", "/"))]
    sys.modules[package_name] = package


class _FakeTensor:

    def __init__(self, values, *, device="cpu"):
        self.values = values
        self.device = device
        self.shape = self._shape(values)

    @staticmethod
    def _shape(values):
        if not isinstance(values, (list, tuple)):
            return ()
        if not values:
            return (0,)
        return (len(values),) + _FakeTensor._shape(values[0])

    def tolist(self):
        def convert(value):
            if isinstance(value, (list, tuple)):
                return [convert(item) for item in value]
            return value

        return convert(self.values)


fake_torch = types.ModuleType("torch")
fake_torch.Tensor = _FakeTensor
fake_torch.int32 = object()
fake_torch.tensor = lambda values, **kwargs: _FakeTensor(
    values,
    device=kwargs.get("device", "cpu"),
)


def _inference_mode():
    def decorate(function):
        return function

    return decorate


fake_torch.inference_mode = _inference_mode
sys.modules["torch"] = fake_torch


cache_module = types.ModuleType(
    "tinyvllm.engine.proposal_kv_cache"
)
cache_module.ProposalKVCache = type("ProposalKVCache", (), {})
sys.modules[cache_module.__name__] = cache_module

lifecycle_module = types.ModuleType(
    "tinyvllm.engine.proposal_kv_lifecycle"
)
lifecycle_module.ProposalKVLifecycleCoordinator = type(
    "ProposalKVLifecycleCoordinator",
    (),
    {},
)
lifecycle_module.ProposalKVRegistration = type(
    "ProposalKVRegistration",
    (),
    {},
)
sys.modules[lifecycle_module.__name__] = lifecycle_module

executor_contracts = types.ModuleType(
    "tinyvllm.engine.speculative_proposal_executor"
)
for name in (
    "ModelRunnerProposalInput",
    "ProposalFinalizeRow",
    "TargetPrefillObservation",
):
    setattr(executor_contracts, name, type(name, (), {}))
executor_contracts.proposal_input_context_token_count = (
    lambda row: row.context_token_count
)
sys.modules[executor_contracts.__name__] = executor_contracts

greedy_module = types.ModuleType(
    "tinyvllm.engine.tensor_parallel_greedy"
)
greedy_module.select_tensor_parallel_greedy_tokens = lambda **_: ()
sys.modules[greedy_module.__name__] = greedy_module

adapter_module = types.ModuleType("tinyvllm.speculative.adapter")
adapter_module.DraftCapabilities = type("DraftCapabilities", (), {})
adapter_module.DraftProposal = type("DraftProposal", (), {})
sys.modules[adapter_module.__name__] = adapter_module

captured_contexts = []
context_module = types.ModuleType("tinyvllm.utils.context")


@contextmanager
def _temporary_context(**kwargs):
    captured_contexts.append(kwargs)
    yield


context_module.temporary_context = _temporary_context
sys.modules[context_module.__name__] = context_module

from tinyvllm.engine.qwen35_mtp_executor import (
    Qwen35MTPProposalExecutor,
)

for module_name, original in _ORIGINAL_MODULES.items():
    if original is _MISSING:
        sys.modules.pop(module_name, None)
    else:
        sys.modules[module_name] = original


@dataclass(frozen=True)
class _Identity:
    logical_entry_id: int
    generation: int


@dataclass(frozen=True)
class _Lease:
    identities: tuple[_Identity, ...]
    physical_slot_ids: tuple[int, ...]
    occupancy_generations: tuple[int, ...]


class _Allocator:

    def __init__(self, leases):
        self.leases = leases
        self.read_requests = []
        self.write_requests = []
        self.read_completions = []
        self.write_completions = []

    def ensure_readable(self, identities):
        identities = tuple(identities)
        self.read_requests.append(identities)
        return self.leases[("read", identities)]

    def ensure_writable(self, identities):
        identities = tuple(identities)
        self.write_requests.append(identities)
        return self.leases[("write", identities)]

    def record_read_complete(self, lease):
        self.read_completions.append(lease)

    def record_write_complete(self, lease):
        self.write_completions.append(lease)


class _Module:

    def __init__(self, *, fail=False):
        self.fail = fail

    def forward_hidden(self, input_ids, positions, hidden):
        if self.fail:
            raise RuntimeError("bootstrap failed")
        return hidden

    def forward_step(self, input_ids, positions, hidden):
        if self.fail:
            raise RuntimeError("step failed")
        return hidden, "logits"


def _executor(allocator, committed=()):
    executor = Qwen35MTPProposalExecutor.__new__(
        Qwen35MTPProposalExecutor
    )
    executor.proposal_kv_cache = SimpleNamespace(
        entry_allocator=allocator,
        committed_entry_identities=lambda sequence_id: committed,
    )
    executor.module = _Module()
    return executor


def test_bootstrap_uses_nonidentity_writable_lease_and_records_completion():
    staged = (_Identity(0, 1), _Identity(1, 1))
    writable = _Lease(staged, (7, 3), (4, 9))
    allocator = _Allocator({("write", staged): writable})
    executor = _executor(allocator)
    transaction = SimpleNamespace(staged_entry_identities=staged)
    captured_contexts.clear()

    result = executor._forward_bootstrap(
        executor.module,
        transaction,
        _FakeTensor([11, 12]),
        _FakeTensor([0, 1]),
        _FakeTensor([[1], [2]], device="cpu"),
    )

    assert result.tolist() == [[1], [2]]
    assert transaction.staged_entry_identities == staged
    assert captured_contexts[-1]["slot_mapping"].tolist() == [7, 3]
    assert allocator.write_requests == [staged]
    assert allocator.write_completions == [writable]


def test_proposal_step_preserves_visible_order_and_completion_coverage():
    committed = (_Identity(10, 1), _Identity(11, 1))
    staged = (
        _Identity(20, 1),
        _Identity(21, 1),
        _Identity(22, 1),
    )
    read_prefix = committed + staged[:1]
    read_lease = _Lease(read_prefix, (6, 2, 7), (1, 1, 1))
    write_identity = (staged[1],)
    write_lease = _Lease(write_identity, (3,), (9,))
    allocator = _Allocator({
        ("read", read_prefix): read_lease,
        ("write", write_identity): write_lease,
    })
    executor = _executor(allocator, committed=committed)
    transaction = SimpleNamespace(
        sequence_id=5,
        staged_entry_identities=staged,
    )
    captured_contexts.clear()

    executor._forward_proposal_step(
        transaction,
        step=1,
        input_ids=_FakeTensor([13]),
        positions=_FakeTensor([4]),
        current_hidden=_FakeTensor([[1]], device="cpu"),
    )

    context = captured_contexts[-1]
    assert context["slot_mapping"].tolist() == [3]
    assert context["block_tables"].tolist() == [[6, 2, 7, 3]]
    assert context["context_lens"].tolist() == [4]
    assert allocator.read_requests == [read_prefix]
    assert allocator.write_requests == [write_identity]
    assert allocator.read_completions == [read_lease]
    assert allocator.write_completions == [write_lease]
