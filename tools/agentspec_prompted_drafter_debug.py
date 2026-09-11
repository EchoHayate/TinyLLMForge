#!/usr/bin/env python3
"""Print raw prompted-drafter generations so the harness can be trusted.

A match rate near zero is a claim about the model or a bug in the
harness, and the two look identical in the summary numbers. This dumps
the prompt tail and the raw continuation for a handful of rows so the
difference is visible.
"""

from __future__ import annotations

import argparse
import sys

sys.path.insert(0, "tools")

import worker  # noqa: E402  the uploaded match worker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evalset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--variant", default="tail")
    parser.add_argument("--token-cap", type=int, default=11)
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--prompt-style", default="v2")
    parser.add_argument("--prompt-budget", type=int, default=512)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    header, rows = worker.load_evalset(args.evalset, args.rows)
    corpus = header["corpus"]
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to("cuda")
    model.eval()

    print("eos id %s pad id %s" % (tokenizer.eos_token_id, tokenizer.pad_token_id))
    for row in rows:
        prompt, prompt_tokens = worker.build_prompt(
            tokenizer,
            corpus,
            row,
            args.variant,
            args.prompt_style,
            args.prompt_budget,
        )
        encoded = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=args.token_cap,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_ids = generated[0, encoded["input_ids"].shape[1] :].tolist()
        print("=" * 70)
        print("PROMPT TOK  >>> %d" % prompt_tokens)
        print("PROMPT TAIL >>> %s" % prompt[-320:].replace("\n", "\\n"))
        print("GOLD        >>> %s" % row["gold_text"][:120])
        print("RAW IDS     >>> %s" % new_ids)
        print(
            "RAW TEXT    >>> %r"
            % tokenizer.decode(new_ids, skip_special_tokens=False)
        )
        print(
            "CLEAN       >>> %r"
            % tokenizer.decode(new_ids, skip_special_tokens=True)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
