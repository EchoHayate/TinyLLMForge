import os
from pathlib import Path
from types import SimpleNamespace

import torch

from tinyvllm.layers.gated_delta import qwen35_gated_rmsnorm
from tinyvllm.layers.qwen35_linear_attention import (
    Qwen35LinearAttentionShell,
)


FIXTURE_PATH = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "qwen35_layer20_step38_norm_geometry_r457.pt"
)


def test_tp_decode_gated_rmsnorm_matches_official_global_geometry() -> None:
    fixture_path = Path(
        os.environ.get("QWEN35_NORM_FIXTURE", FIXTURE_PATH)
    )
    fixture = torch.load(
        fixture_path,
        map_location="cuda",
        weights_only=True,
    )
    core = fixture["core"].cuda()
    gate = fixture["gate"].cuda()
    weight = fixture["weight"].cuda()
    expected = fixture["expected"].cuda()
    tensor_parallel_size = int(fixture["tensor_parallel_size"])
    norm_eps = float(fixture["norm_eps"])

    local = qwen35_gated_rmsnorm(
        core,
        gate,
        weight,
        eps=norm_eps,
    )
    assert not torch.equal(local, expected)

    shell = SimpleNamespace(
        in_proj_qkv=SimpleNamespace(tp_size=tensor_parallel_size),
        local_value_heads=core.shape[0],
        norm_eps=norm_eps,
        norm_weight=weight,
    )
    apply_norm = getattr(
        Qwen35LinearAttentionShell,
        "_apply_decode_gated_rmsnorm",
        lambda self, value, value_gate: qwen35_gated_rmsnorm(
            value,
            value_gate,
            self.norm_weight,
            eps=self.norm_eps,
        ),
    )
    actual = apply_norm(shell, core, gate)

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the geometry regression")
    test_tp_decode_gated_rmsnorm_matches_official_global_geometry()
    print("PASS: TP decode gated RMSNorm global geometry")


if __name__ == "__main__":
    main()
