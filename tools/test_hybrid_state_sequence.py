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
Sequence = sequence_module.Sequence


def test_prompt_round_trip_preserves_hybrid_lease():
    sequence = Sequence(
        [1, 2, 3],
        SamplingParams(
            temperature=0.7,
            max_tokens=4,
        ),
    )
    sequence.hybrid_state_slot_id = 5
    sequence.hybrid_state_generation = 7
    restored = pickle.loads(pickle.dumps(sequence))
    assert restored.token_ids == [1, 2, 3]
    assert restored.seq_id == sequence.seq_id
    assert restored.hybrid_state_slot_id == 5
    assert restored.hybrid_state_generation == 7
    assert restored.temperature == 0.7
    assert restored.max_tokens == 4


def test_decode_round_trip_preserves_hybrid_lease():
    sequence = Sequence(
        [1, 2],
        SamplingParams(
            temperature=0.25,
            max_tokens=4,
        ),
    )
    sequence.append_token(9)
    sequence.hybrid_state_slot_id = 2
    sequence.hybrid_state_generation = 3
    restored = pickle.loads(pickle.dumps(sequence))
    assert restored.last_token == 9
    assert restored.seq_id == sequence.seq_id
    assert restored.hybrid_state_slot_id == 2
    assert restored.hybrid_state_generation == 3
    assert restored.temperature == 0.25
    assert restored.max_tokens == 4


def test_old_fourteen_field_state_defaults_temperature_to_greedy():
    sequence = Sequence.__new__(Sequence)
    old_state = (
        3,
        3,
        0,
        [],
        0,
        0,
        3,
        True,
        False,
        True,
        11,
        -1,
        0,
        [1, 2, 3],
    )

    sequence.__setstate__(old_state)

    assert sequence.token_ids == [1, 2, 3]
    assert sequence.seq_id == 11
    assert sequence.temperature == 0.0


@pytest.mark.parametrize(
    "temperature",
    [True, "0.0", float("nan"), float("inf")],
)
def test_new_fifteen_field_state_rejects_invalid_temperature(temperature):
    sequence = Sequence.__new__(Sequence)
    state = (
        3,
        3,
        0,
        [],
        0,
        0,
        3,
        True,
        False,
        True,
        11,
        -1,
        0,
        temperature,
        [1, 2, 3],
    )

    with pytest.raises(ValueError, match="temperature must be a finite number"):
        sequence.__setstate__(state)


def test_legacy_thirteen_field_state_defaults_temperature_to_greedy():
    sequence = Sequence.__new__(Sequence)
    legacy = (
        3,
        3,
        0,
        [],
        0,
        0,
        3,
        True,
        False,
        True,
        5,
        7,
        [1, 2, 3],
    )

    sequence.__setstate__(legacy)

    assert sequence.token_ids == [1, 2, 3]
    assert sequence.seq_id == -1
    assert sequence.hybrid_state_slot_id == 5
    assert sequence.hybrid_state_generation == 7
    assert sequence.temperature == 0.0


def test_legacy_eleven_field_state_uses_disabled_sentinel():
    sequence = Sequence.__new__(Sequence)
    legacy = (
        3,
        3,
        0,
        [],
        0,
        0,
        3,
        True,
        False,
        True,
        [1, 2, 3],
    )
    sequence.__setstate__(legacy)
    assert sequence.token_ids == [1, 2, 3]
    assert sequence.seq_id == -1
    assert sequence.hybrid_state_slot_id == -1
    assert sequence.hybrid_state_generation == 0
    assert sequence.temperature == 0.0


def test_older_state_uses_disabled_sentinel():
    sequence = Sequence.__new__(Sequence)
    older = (3, 3, 0, [], [1, 2, 3])
    sequence.__setstate__(older)
    assert sequence.token_ids == [1, 2, 3]
    assert sequence.seq_id == -1
    assert sequence.hybrid_state_slot_id == -1
    assert sequence.hybrid_state_generation == 0
    assert sequence.temperature == 0.0


if __name__ == "__main__":
    test_prompt_round_trip_preserves_hybrid_lease()
    test_decode_round_trip_preserves_hybrid_lease()
    test_old_fourteen_field_state_defaults_temperature_to_greedy()
    test_legacy_eleven_field_state_uses_disabled_sentinel()
    test_older_state_uses_disabled_sentinel()
    print("hybrid state sequence tests passed")
