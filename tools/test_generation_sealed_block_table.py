import importlib.util
import os
import pickle
import sys
import types

import pytest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

tinyvllm_pkg = types.ModuleType("tinyvllm")
tinyvllm_pkg.__path__ = [os.path.join(ROOT, "tinyvllm")]
engine_pkg = types.ModuleType("tinyvllm.engine")
engine_pkg.__path__ = [os.path.join(ROOT, "tinyvllm", "engine")]
sys.modules.setdefault("tinyvllm", tinyvllm_pkg)
sys.modules.setdefault("tinyvllm.engine", engine_pkg)


def load_module(module_name, relative_path):
    spec = importlib.util.spec_from_file_location(
        module_name,
        os.path.join(ROOT, relative_path),
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


sampling = load_module(
    "tinyvllm.sampling_params",
    "tinyvllm/sampling_params.py",
)
sequence_module = load_module(
    "tinyvllm.engine.sequence",
    "tinyvllm/engine/sequence.py",
)
SamplingParams = sampling.SamplingParams


def current_sequence_module():
    return sys.modules["tinyvllm.engine.sequence"]


def make_sequence():
    return current_sequence_module().Sequence(
        [1, 2, 3],
        SamplingParams(
            temperature=0.0,
            max_tokens=16,
        ),
    )


@pytest.mark.parametrize(
    ("initial", "mutate", "expected"),
    [
        ([], lambda table: table.append(1), [1]),
        ([1], lambda table: table.extend([2, 3]), [1, 2, 3]),
        ([1, 2], lambda table: table.insert(1, 9), [1, 9, 2]),
        ([1, 2], lambda table: table.__setitem__(1, 9), [1, 9]),
        (
            [1, 2, 3],
            lambda table: table.__setitem__(slice(1, 3), [8, 9]),
            [1, 8, 9],
        ),
        ([1, 2], lambda table: table.__delitem__(1), [1]),
        (
            [1, 2, 3],
            lambda table: table.__delitem__(slice(1, 3)),
            [1],
        ),
        ([1, 2], lambda table: table.pop(), [1]),
        ([1, 2], lambda table: table.remove(1), [2]),
        ([1, 2], lambda table: table.clear(), []),
        ([1, 2, 3], lambda table: table.reverse(), [3, 2, 1]),
        ([3, 1, 2], lambda table: table.sort(), [1, 2, 3]),
        ([1], lambda table: table.__iadd__([2, 3]), [1, 2, 3]),
        ([1, 2], lambda table: table.__imul__(2), [1, 2, 1, 2]),
    ],
)
def test_each_block_table_mutator_advances_revision_once(
    initial,
    mutate,
    expected,
):
    sequence = make_sequence()
    sequence.block_table = initial
    table = sequence.block_table
    start = table.revision

    mutate(table)

    assert sequence.block_table is table
    assert list(table) == expected
    assert table.revision == start + 1


def test_failed_block_table_mutation_still_invalidates_prior_revision():
    sequence = make_sequence()
    sequence.block_table = [1]
    table = sequence.block_table
    start = table.revision

    with pytest.raises(ValueError):
        table.remove(2)

    assert table.revision == start + 1


def test_property_augmented_assignment_preserves_table_and_single_bump():
    sequence = make_sequence()
    sequence.block_table = [1, 2]
    table = sequence.block_table
    start = table.revision

    sequence.block_table += [3]

    assert sequence.block_table is table
    assert list(table) == [1, 2, 3]
    assert table.revision == start + 1

    sequence.block_table *= 2

    assert sequence.block_table is table
    assert list(table) == [1, 2, 3, 1, 2, 3]
    assert table.revision == start + 2


def test_whole_block_table_replacement_uses_owned_revision_lineage():
    first = make_sequence()
    first.block_table = [1]
    prior = first.block_table
    prior_revision = prior.revision

    incoming = make_sequence()
    incoming.block_table = [7, 8]
    for value in range(100):
        incoming.block_table.append(value)
        incoming.block_table.pop()

    first.block_table = incoming.block_table

    assert first.block_table is not prior
    assert first.block_table is not incoming.block_table
    assert list(first.block_table) == [7, 8]
    assert first.block_table.revision == prior_revision + 1


def test_block_table_revision_is_read_only():
    sequence = make_sequence()
    table_type = getattr(
        current_sequence_module(),
        "VersionedBlockTable",
        None,
    )

    assert table_type is not None
    assert isinstance(sequence.block_table, table_type)
    with pytest.raises(AttributeError):
        sequence.block_table.revision = 7


def test_sequence_state_serializes_plain_ids_revision_and_last_payload():
    sequence = make_sequence()
    sequence.block_table = [7, 8]
    sequence.block_table.append(9)

    state = sequence.__getstate__()

    assert len(state) == 17
    assert type(state[3]) is list
    assert state[3] == [7, 8, 9]
    assert state[15] == sequence.block_table.revision
    assert state[-1] == [1, 2, 3]

    restored = pickle.loads(pickle.dumps(sequence))
    assert list(restored.block_table) == [7, 8, 9]
    assert restored.block_table.revision == sequence.block_table.revision


@pytest.mark.parametrize(
    "legacy_state",
    [
        (
            3, 3, 0, [7], 0, 0, 3, True, False, True,
            11, -1, 0, 0.0, 16, [1, 2, 3],
        ),
        (
            3, 3, 0, [7], 0, 0, 3, True, False, True,
            11, -1, 0, 0.0, [1, 2, 3],
        ),
        (
            3, 3, 0, [7], 0, 0, 3, True, False, True,
            11, -1, 0, [1, 2, 3],
        ),
        (
            3, 3, 0, [7], 0, 0, 3, True, False, True,
            5, 7, [1, 2, 3],
        ),
        (
            3, 3, 0, [7], 0, 0, 3, True, False, True,
            [1, 2, 3],
        ),
        (3, 3, 0, [7], [1, 2, 3]),
    ],
)
def test_legacy_sequence_states_restore_revision_zero(legacy_state):
    sequence_type = current_sequence_module().Sequence
    sequence = sequence_type.__new__(sequence_type)

    sequence.__setstate__(legacy_state)

    assert list(sequence.block_table) == [7]
    assert sequence.block_table.revision == 0


@pytest.mark.parametrize("revision", [True, -1, 1.5, "1"])
def test_new_sequence_state_rejects_invalid_block_table_revision(
    revision,
):
    sequence = make_sequence()
    state = list(sequence.__getstate__())
    state.insert(15, revision)
    sequence_type = current_sequence_module().Sequence
    restored = sequence_type.__new__(sequence_type)

    with pytest.raises(
        ValueError,
        match="block-table revision must be a non-negative integer",
    ):
        restored.__setstate__(tuple(state))


def test_replacement_from_list_subclass_does_not_call_python_iteration():
    class IterationCountingList(list):
        def __init__(self, values):
            super().__init__(values)
            self.iterations = 0

        def __iter__(self):
            self.iterations += 1
            return super().__iter__()

    sequence = make_sequence()
    incoming = IterationCountingList([7, 8])

    sequence.block_table = incoming

    assert list(sequence.block_table) == [7, 8]
    assert incoming.iterations == 0
