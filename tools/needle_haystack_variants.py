"""Needle prompts that a top-k selector can actually fail on.

The existing harness fills the context with one sentence repeated thousands of times
and inserts a single unique needle. Under that construction the needle is the only
lexically distinctive region in the whole prompt, so *any* selector that looks at the
key vectors at all will find it, and every arm scores 100%. That is a property of the
prompt, not evidence about the selector.

Three variants are provided so the difference can be measured rather than asserted:

  repetitive  the current construction, kept as the control that should saturate
  natural     lexically diverse filler, so attention mass is spread and a selector
              must actually rank rather than spot the one odd region
  distractor  several decoy needles with the same surface form; the question names
              which one it wants, so a selector that retrieves "a needle" instead of
              "the needle" is now wrong and the metric can see it

Positions of the answer-bearing tokens are returned in token space, because the
selector gate needs to know whether those exact rows survived selection.
"""

from __future__ import annotations

import random
from typing import Sequence

REPETITIVE_SENTENCE = (
    "The grass is green. The sky is blue. The sun is yellow. "
    "Here we go. There and back again. "
)

_ADJ = (
    "quiet narrow copper hollow distant frozen bitter crowded ancient gentle "
    "restless amber crooked humid silver stubborn faded vivid brittle solemn"
).split()
_NOUN = (
    "harbor lantern courtyard ledger orchard tunnel bell rope canyon marsh "
    "workshop kettle archive foundry sparrow granary compass thicket beacon quarry "
    "mill parapet cistern trellis shale"
).split()
_VERB = (
    "settles drifts rattles gathers narrows warms rusts hums splits lingers "
    "tilts spreads cools creaks folds"
).split()
_PLACE = (
    "north bridge station ravine terrace pier boundary orchard hillside crossing "
    "reservoir escarpment"
).split()

LABELS = ("harbor", "ledger", "orchard", "foundry", "beacon", "quarry")


def natural_filler(rng: random.Random, n_sentences: int) -> str:
    out = []
    for _ in range(n_sentences):
        out.append(
            f"The {rng.choice(_ADJ)} {rng.choice(_NOUN)} {rng.choice(_VERB)} "
            f"near the {rng.choice(_PLACE)}. "
        )
    return "".join(out)


def _insert(ids: list[int], pieces: Sequence[tuple[int, list[int]]]) -> tuple[list[int], list[list[int]]]:
    """Insert (position, token_ids) pieces into `ids`, left to right.

    Returns the new id list and, for each piece in its original order, the absolute
    token positions it occupies in the result. Built by construction rather than by
    shifting arithmetic, because off-by-one here would silently corrupt every
    needle-coverage number downstream.
    """
    order = sorted(range(len(pieces)), key=lambda i: pieces[i][0])
    out: list[int] = []
    positions: list[list[int]] = [[] for _ in pieces]
    prev = 0
    for i in order:
        pos, chunk = pieces[i]
        pos = max(prev, min(pos, len(ids)))
        out.extend(ids[prev:pos])
        positions[i] = list(range(len(out), len(out) + len(chunk)))
        out.extend(chunk)
        prev = pos
    out.extend(ids[prev:])
    return out, positions


def build_variant(
    tokenizer,
    variant: str,
    seq_len: int,
    *,
    depth: float = 0.5,
    num_decoys: int = 4,
    seed: int = 0,
) -> dict:
    rng = random.Random(seed)
    magic = rng.randint(10000, 99999)

    if variant == "repetitive":
        question = "\n\nWhat is the magic number? Answer with only the digits, nothing else."
        needle = f"The magic number is {magic}. Remember it. "
        filler = REPETITIVE_SENTENCE * max(1, seq_len // 4)
        decoys: list[tuple[float, str]] = []
    elif variant == "natural":
        question = "\n\nWhat is the magic number? Answer with only the digits, nothing else."
        needle = f"The magic number is {magic}. Remember it. "
        filler = natural_filler(rng, max(1, seq_len // 3))
        decoys = []
    elif variant == "distractor":
        target = LABELS[0]
        question = (
            f"\n\nWhat is the magic number for the {target}? "
            "Answer with only the digits, nothing else."
        )
        needle = f"The magic number for the {target} is {magic}. Remember it. "
        filler = natural_filler(rng, max(1, seq_len // 3))
        decoys = []
        for i, label in enumerate(LABELS[1:1 + num_decoys]):
            other = rng.randint(10000, 99999)
            frac = (i + 1) / (num_decoys + 1)
            decoys.append((frac, f"The magic number for the {label} is {other}. Remember it. "))
    else:
        raise ValueError(f"unknown variant: {variant}")

    needle_ids = tokenizer.encode(needle, add_special_tokens=False)
    question_ids = tokenizer.encode(question, add_special_tokens=False)
    decoy_ids = [tokenizer.encode(text, add_special_tokens=False) for _, text in decoys]

    budget = seq_len - len(needle_ids) - len(question_ids) - sum(len(d) for d in decoy_ids)
    if budget < 64:
        raise ValueError(f"seq_len {seq_len} too small for variant {variant}")
    filler_ids = tokenizer.encode(filler, add_special_tokens=False)
    while len(filler_ids) < budget:
        filler_ids = filler_ids + filler_ids
    filler_ids = filler_ids[:budget]

    pieces: list[tuple[int, list[int]]] = [(int(depth * budget), needle_ids)]
    for (frac, _), ids in zip(decoys, decoy_ids):
        pieces.append((int(frac * budget), ids))

    body, positions = _insert(filler_ids, pieces)
    ids = body + question_ids

    return {
        "variant": variant,
        "ids": ids,
        "seq_len": len(ids),
        "answer": str(magic),
        "answer_needle_positions": positions[0],
        "decoy_positions": [p for group in positions[1:] for p in group],
        "question": question,
        "needle_text": needle,
        "num_decoys": len(decoy_ids),
        "depth": depth,
        "seed": seed,
    }
