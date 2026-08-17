from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed-heads", required=True, help="Comma-separated layer:kv_head list.")
    p.add_argument("--candidate-heads", required=True, help="Comma-separated layer:kv_head list.")
    p.add_argument("--budget-fraction", type=float, default=0.5)
    p.add_argument("--threshold-start", type=float, default=0.901)
    p.add_argument("--threshold-step", type=float, default=0.001)
    p.add_argument("--policy-script", default=None)
    return p.parse_args()


def parse_heads(text):
    heads = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        layer_text, head_text = item.split(":", 1)
        heads.append((int(layer_text), int(head_text)))
    return heads


def format_heads(heads):
    return ",".join(f"{layer}:{head}" for layer, head in heads)


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    script = Path(args.policy_script) if args.policy_script else Path(__file__).with_name("make_recovery_task_policy.py")
    seed_heads = parse_heads(args.seed_heads)
    candidate_heads = [head for head in parse_heads(args.candidate_heads) if head not in set(seed_heads)]
    manifest = []

    for index, candidate in enumerate(candidate_heads, start=1):
        heads = seed_heads + [candidate]
        threshold = args.threshold_start + args.threshold_step * (index - 1)
        name = f"add_l{candidate[0]}_h{candidate[1]}"
        policy_dir = out / name
        cmd = [
            sys.executable,
            str(script),
            "--compact-heads",
            format_heads(heads),
            "--budget-fraction",
            str(args.budget_fraction),
            "--threshold",
            str(threshold),
            "--output-dir",
            str(policy_dir),
            "--note",
            name,
        ]
        subprocess.run(cmd, check=True)
        manifest.append(
            {
                "name": name,
                "candidate": {"layer": candidate[0], "kv_head": candidate[1]},
                "threshold": threshold,
                "policy_dir": str(policy_dir),
                "heads": format_heads(heads),
                "compressed_heads": len(heads),
                "budget_fraction": args.budget_fraction,
            }
        )

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    report = [
        "# Head Addition Policies",
        "",
        f"Seed heads: `{format_heads(seed_heads)}`",
        f"Budget fraction: `{args.budget_fraction:.2%}`",
        "",
        "| Threshold | Variant | Added Head | Compressed Heads | Policy Dir |",
        "|---:|---|---|---:|---|",
    ]
    for item in manifest:
        cand = item["candidate"]
        report.append(
            f"| {item['threshold']:.3f} | {item['name']} | {cand['layer']}:{cand['kv_head']} | "
            f"{item['compressed_heads']} | `{item['policy_dir']}` |"
        )
    (out / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
