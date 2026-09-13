"""Latent KV capacity: analytic serving-capacity model.

Scope and claim boundary
------------------------
This package answers one question and refuses to answer any other:

    Under what conditions does compressing the KV cache into a smaller latent
    representation (MLA-style) buy *serving capacity* for an agent workload,
    given that lossless prefix caching and lossless host offload already exist?

Everything here is analytic. No task in this package may import torch,
transformers, or numpy, load a checkpoint, touch a GPU, or reach the network.
Every number it emits is a consequence of declared inputs, never a measurement.
The two measured inputs it consumes are quoted from prior artifacts and carry
their provenance in `provenance.py`.

This package deliberately does not model:

  * generation quality under compression (Stage 1b concern),
  * the numerics of any particular decomposition (CARE / MLA / low-rank),
  * token-level distributional losslessness (out of scope, see
    `tinyvllm/speculative/`).

It is the entry gate that every later stage of the latent-kv-capacity line
re-runs before it is allowed to spend GPU time.
"""

from tinyvllm.kvcapacity.capacity_model import (
    CapacityPoint,
    CompressionSpec,
    DecodeStepFit,
    DeviceBudget,
    ModelGeometry,
    TurnProfile,
    capacity_gain,
    evaluate_point,
)

__all__ = [
    "CapacityPoint",
    "CompressionSpec",
    "DecodeStepFit",
    "DeviceBudget",
    "ModelGeometry",
    "TurnProfile",
    "capacity_gain",
    "evaluate_point",
]
