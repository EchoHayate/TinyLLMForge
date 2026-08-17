from __future__ import annotations

import json
import os
from pathlib import Path
import sys


def main():
    remote = Path(sys.argv[1])
    source = Path(sys.argv[2])
    inputs = remote / "inputs"
    output = remote / "output"
    sys.path.insert(0, str(source / "tools"))
    sys.path.insert(0, str(source))

    import run_qwen35_tp4_engine_correctness_authority as driver
    import qwen35_tp4_engine_official_reference_executor as official
    import qwen35_tp4_cached_first_divergence_probe as probe

    configuration = driver.load_configuration(
        inputs / "executor_configuration.json",
        source_inventory_path=inputs / "source_inventory.json",
    )
    factory = official.build_official_reference_executor_factory(
        configuration
    )
    result = probe.run_probe(
        configuration=configuration,
        engine_factory=probe.backend._default_engine_factory,
        reference_executor_factory=factory,
        workload="w1_medium_reuse",
        request_index=0,
        generated_tokens=1,
    )
    temporary = output / ".result.json.tmp"
    final = output / "result.json"
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, final)
    print(json.dumps({
        "classification": result["classification"],
        "result_path": str(final),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
