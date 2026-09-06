from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_module(module_name, relative_path):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


for package_name in ("tinyvllm", "tinyvllm.engine"):
    package = types.ModuleType(package_name)
    package.__path__ = [
        str(ROOT / package_name.replace(".", "/"))
    ]
    sys.modules[package_name] = package

hybrid_state = _load_module(
    "tinyvllm.engine.hybrid_state",
    "tinyvllm/engine/hybrid_state.py",
)
manifest_module = _load_module(
    "tinyvllm.engine.exact_cuda_graph_lease_manifest",
    "tinyvllm/engine/exact_cuda_graph_lease_manifest.py",
)
HybridStateLease = hybrid_state.HybridStateLease
build_exact_cuda_graph_lease_manifest = (
    manifest_module.build_exact_cuda_graph_lease_manifest
)


def test_manifest_preserves_order_and_hashes_canonically():
    leases = (
        HybridStateLease(4, 2, 101),
        HybridStateLease(1, 7, 202),
    )
    manifest = build_exact_cuda_graph_lease_manifest(
        leases=leases,
        expected_request_ids=(101, 202),
        validate_lease=lambda lease: lease,
    )

    assert manifest.slot_ids == (4, 1)
    assert tuple(row.batch_index for row in manifest.rows) == (0, 1)
    assert len(manifest.sha256) == 64
    assert manifest.sha256 == build_exact_cuda_graph_lease_manifest(
        leases=leases,
        expected_request_ids=(101, 202),
        validate_lease=lambda lease: lease,
    ).sha256
    assert manifest.sha256 != build_exact_cuda_graph_lease_manifest(
        leases=tuple(reversed(leases)),
        expected_request_ids=(202, 101),
        validate_lease=lambda lease: lease,
    ).sha256


@pytest.mark.parametrize(
    "leases,request_ids,message",
    [
        ((), (), "non-empty"),
        (
            (HybridStateLease(0, 1, 10),),
            (),
            "row counts",
        ),
        (
            (HybridStateLease(0, 1, 10),),
            (11,),
            "request order",
        ),
        (
            (
                HybridStateLease(0, 1, 10),
                HybridStateLease(0, 1, 11),
            ),
            (10, 11),
            "distinct slots",
        ),
    ],
)
def test_manifest_rejects_invalid_ownership_shape(
    leases,
    request_ids,
    message,
):
    with pytest.raises((ValueError, RuntimeError), match=message):
        build_exact_cuda_graph_lease_manifest(
            leases=leases,
            expected_request_ids=request_ids,
            validate_lease=lambda lease: lease,
        )


def test_manifest_propagates_stale_generation_before_returning():
    stale = HybridStateLease(0, 1, 10)
    calls = []

    def reject_stale(lease):
        calls.append(lease)
        raise RuntimeError("stale generation")

    with pytest.raises(RuntimeError, match="stale generation"):
        build_exact_cuda_graph_lease_manifest(
            leases=(stale,),
            expected_request_ids=(10,),
            validate_lease=reject_stale,
        )

    assert calls == [stale]


def test_manifest_rejects_validator_identity_rewrite():
    lease = HybridStateLease(0, 1, 10)

    with pytest.raises(RuntimeError, match="changed identity"):
        build_exact_cuda_graph_lease_manifest(
            leases=(lease,),
            expected_request_ids=(10,),
            validate_lease=lambda value: HybridStateLease(
                value.slot_id,
                value.generation + 1,
                value.request_id,
            ),
        )
