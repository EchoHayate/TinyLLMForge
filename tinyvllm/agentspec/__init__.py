"""Action-level speculative execution contracts for agent runtimes.

This package is contract-and-analysis only. It does not execute a
model, does not call a tool, and does not change any token-level
speculative decoding behaviour in ``tinyvllm.speculative``.

Scope boundary:

- ``action``          side-effect classification and action identity;
- ``latent_adapter``  latent-state action drafter contract;
- ``cost_model``      analytic break-even model for shared serving;
- ``router``          fail-closed route selection.

No module in this package claims a measured speedup.
"""

from __future__ import annotations

__all__ = [
    "action",
    "cost_model",
    "latent_adapter",
    "router",
]
