#!/usr/bin/env python3
"""Stage 1a worker: measure the drafter GPU tax on real hardware.

Stage 0 priced action-level speculation over *declared* inputs. The
two inputs that decide the verdict are the actor GPU demand ``D`` and
the drafter tax ``tau = G_draft / D``. This worker measures both.

It measures cost only. The code-drafter head is randomly initialised,
because head weights change accuracy, not cost. Nothing here claims a
match probability; that is Stage 1b and needs real agent traces.

Arms, all evaluated per agent step:

- ``actor``               target model, prefill of a trajectory of
                          length ``L`` plus greedy decode of ``A``
                          action tokens;
- ``text_drafter``        small model, same prefill plus same decode,
                          i.e. the published design;
- ``code_drafter``        small model, prefill only, plus one linear
                          head over the last hidden state that scores
                          a tool-and-argument code vocabulary;
- ``code_drafter_ckv``    same as ``code_drafter`` but over a
                          compressed context budget, to test whether
                          drafter cost decouples from trajectory
                          length.

``--synthetic`` replaces both checkpoints with a small random stack so
that the harness itself can be smoke-tested on CPU. Synthetic runs are
marked in the payload and are never valid gate evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import time

import torch


SCHEMA_VERSION = 1
DEFAULT_ACTION_TOKENS = 32
DEFAULT_CONTEXT_LENGTHS = (1024, 4096, 16384)
DEFAULT_COMPRESSED_BUDGET = 512
DEFAULT_CODE_VOCABULARY = 4096
DEFAULT_REPETITIONS = 5
DEFAULT_WARMUP = 2


class _Timer:
    """CUDA-event timing when available, wall clock otherwise."""

    def __init__(self, device):
        self.device = device
        self.cuda = device.type == "cuda"

    def __enter__(self):
        if self.cuda:
            torch.cuda.synchronize(self.device)
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.stop_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()
        else:
            self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self.cuda:
            self.stop_event.record()
            torch.cuda.synchronize(self.device)
            self.elapsed_seconds = (
                self.start_event.elapsed_time(self.stop_event) / 1000.0
            )
        else:
            self.elapsed_seconds = time.perf_counter() - self.start_time
        return False


class SyntheticCausalModel(torch.nn.Module):
    """Small random decoder stack used only for harness smoke tests."""

    def __init__(self, hidden_size, layers, vocab_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=4,
            dim_feedforward=hidden_size * 2,
            batch_first=True,
        )
        self.stack = torch.nn.TransformerEncoder(layer, layers)
        self.head = torch.nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids):
        hidden = self.stack(self.embedding(input_ids))
        return hidden, self.head(hidden[:, -1, :])


def _digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _transformers_version():
    try:
        import transformers
    except ImportError:
        return None
    return transformers.__version__


def _load_real_model(path, device, dtype):
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    # transformers 5 renamed torch_dtype to dtype. Passing the wrong
    # one is silent: the loader forwards unknown keywords to the
    # config and the weights come back as float32, which would make
    # every number in this payload wrong. So pick by version and then
    # assert the dtype that actually landed.
    major = int(transformers.__version__.split(".")[0])
    dtype_keyword = "dtype" if major >= 5 else "torch_dtype"
    model = AutoModelForCausalLM.from_pretrained(
        path,
        trust_remote_code=True,
        **{dtype_keyword: dtype},
    )
    loaded_dtype = next(model.parameters()).dtype
    if loaded_dtype != dtype:
        raise RuntimeError(
            "requested %s but loaded %s from %s"
            % (dtype, loaded_dtype, path)
        )
    model.eval()
    model.to(device)
    identity = {
        "path": path,
        "hidden_size": int(getattr(config, "hidden_size", 0)),
        "num_hidden_layers": int(
            getattr(config, "num_hidden_layers", 0)
        ),
        "vocab_size": int(getattr(config, "vocab_size", 0)),
        "config_digest": _digest(config.to_json_string()),
    }
    return model, identity


def _load_synthetic_model(tag, device, hidden, layers, vocab):
    model = SyntheticCausalModel(hidden, layers, vocab)
    model.eval()
    model.to(device)
    identity = {
        "path": "synthetic:%s" % tag,
        "hidden_size": hidden,
        "num_hidden_layers": layers,
        "vocab_size": vocab,
        "config_digest": _digest("%s:%d:%d" % (tag, hidden, layers)),
    }
    return model, identity


def _random_ids(length, vocab, device):
    return torch.randint(
        low=0,
        high=max(2, vocab - 1),
        size=(1, length),
        device=device,
    )


@torch.no_grad()
def _run_prefill_decode(model, input_ids, action_tokens, synthetic):
    """One prefill plus ``action_tokens`` greedy decode steps."""

    if synthetic:
        hidden, logits = model(input_ids)
        current = input_ids
        for _ in range(action_tokens):
            nxt = torch.argmax(logits, dim=-1, keepdim=True)
            current = torch.cat([current, nxt], dim=1)
            hidden, logits = model(current[:, -256:])
        return
    outputs = model(input_ids=input_ids, use_cache=True)
    past = outputs.past_key_values
    token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
    for _ in range(action_tokens):
        outputs = model(
            input_ids=token.view(1, 1),
            past_key_values=past,
            use_cache=True,
        )
        past = outputs.past_key_values
        token = torch.argmax(outputs.logits[:, -1, :], dim=-1)


@torch.no_grad()
def _run_prefill_only(model, input_ids, synthetic):
    """One prefill and nothing else.

    This arm exists to separate compute from harness overhead. The
    decode loop below is an eager Python loop, so each step carries a
    fixed launch and dispatch cost that is charged to the small model
    and the large model alike. Subtracting the prefill isolates that
    per-step cost, which lets a reader recompute the tax for a serving
    engine that does not pay it.
    """

    if synthetic:
        model(input_ids)
        return
    model(input_ids=input_ids, use_cache=False)


@torch.no_grad()
def _run_prefill_head(model, input_ids, head, synthetic):
    """One prefill plus one code-head scoring pass."""

    if synthetic:
        hidden, _ = model(input_ids)
        head(hidden[:, -1, :])
        return
    outputs = model(
        input_ids=input_ids,
        use_cache=False,
        output_hidden_states=True,
    )
    head(outputs.hidden_states[-1][:, -1, :])


def _measure(callable_object, device, repetitions, warmup):
    for _ in range(warmup):
        callable_object()
    samples = []
    for _ in range(repetitions):
        with _Timer(device) as timer:
            callable_object()
        samples.append(timer.elapsed_seconds)
    samples.sort()
    return {
        "repetitions": repetitions,
        "warmup": warmup,
        "median_seconds": statistics.median(samples),
        "min_seconds": samples[0],
        "max_seconds": samples[-1],
        "samples_seconds": samples,
    }


def build_payload(args):
    device = torch.device(args.device)
    use_cuda = device.type == "cuda"
    dtype = torch.bfloat16 if use_cuda else torch.float32
    if args.synthetic:
        actor, actor_identity = _load_synthetic_model(
            "actor",
            device,
            args.synthetic_actor_hidden,
            args.synthetic_actor_layers,
            args.synthetic_vocab,
        )
        drafter, drafter_identity = _load_synthetic_model(
            "drafter",
            device,
            args.synthetic_drafter_hidden,
            args.synthetic_drafter_layers,
            args.synthetic_vocab,
        )
    else:
        actor, actor_identity = _load_real_model(
            args.actor_model,
            device,
            dtype,
        )
        drafter, drafter_identity = _load_real_model(
            args.drafter_model,
            device,
            dtype,
        )
    head = torch.nn.Linear(
        drafter_identity["hidden_size"],
        args.code_vocabulary,
    )
    head.eval()
    head.to(device=device, dtype=dtype if not args.synthetic else None)

    rows = []
    for context_length in args.context_lengths:
        actor_ids = _random_ids(
            context_length,
            actor_identity["vocab_size"],
            device,
        )
        drafter_ids = _random_ids(
            context_length,
            drafter_identity["vocab_size"],
            device,
        )
        compressed_length = min(args.compressed_budget, context_length)
        compressed_ids = drafter_ids[:, -compressed_length:]

        actor_row = _measure(
            lambda: _run_prefill_decode(
                actor,
                actor_ids,
                args.action_tokens,
                args.synthetic,
            ),
            device,
            args.repetitions,
            args.warmup,
        )
        text_row = _measure(
            lambda: _run_prefill_decode(
                drafter,
                drafter_ids,
                args.action_tokens,
                args.synthetic,
            ),
            device,
            args.repetitions,
            args.warmup,
        )
        code_row = _measure(
            lambda: _run_prefill_head(
                drafter,
                drafter_ids,
                head,
                args.synthetic,
            ),
            device,
            args.repetitions,
            args.warmup,
        )
        code_ckv_row = _measure(
            lambda: _run_prefill_head(
                drafter,
                compressed_ids,
                head,
                args.synthetic,
            ),
            device,
            args.repetitions,
            args.warmup,
        )
        actor_prefill_row = _measure(
            lambda: _run_prefill_only(actor, actor_ids, args.synthetic),
            device,
            args.repetitions,
            args.warmup,
        )
        drafter_prefill_row = _measure(
            lambda: _run_prefill_only(
                drafter,
                drafter_ids,
                args.synthetic,
            ),
            device,
            args.repetitions,
            args.warmup,
        )
        actor_seconds = actor_row["median_seconds"]
        actor_decode_step = (
            actor_seconds - actor_prefill_row["median_seconds"]
        ) / args.action_tokens
        drafter_decode_step = (
            text_row["median_seconds"]
            - drafter_prefill_row["median_seconds"]
        ) / args.action_tokens
        rows.append(
            {
                "context_length": context_length,
                "compressed_context_length": compressed_length,
                "action_tokens": args.action_tokens,
                "actor_seconds": actor_seconds,
                "arms": {
                    "actor": actor_row,
                    "text_drafter": text_row,
                    "code_drafter": code_row,
                    "code_drafter_ckv": code_ckv_row,
                    "actor_prefill": actor_prefill_row,
                    "text_drafter_prefill": drafter_prefill_row,
                },
                "decomposition": {
                    "actor_prefill_seconds": actor_prefill_row[
                        "median_seconds"
                    ],
                    "actor_decode_step_seconds": actor_decode_step,
                    "drafter_prefill_seconds": drafter_prefill_row[
                        "median_seconds"
                    ],
                    "drafter_decode_step_seconds": drafter_decode_step,
                    "decode_share_of_actor": (
                        actor_seconds
                        - actor_prefill_row["median_seconds"]
                    )
                    / actor_seconds,
                    "decode_step_ratio_drafter_over_actor": (
                        drafter_decode_step / actor_decode_step
                        if actor_decode_step > 0
                        else None
                    ),
                },
                "measured_draft_gpu_tax": {
                    "text_drafter": (
                        text_row["median_seconds"] / actor_seconds
                    ),
                    "code_drafter": (
                        code_row["median_seconds"] / actor_seconds
                    ),
                    "code_drafter_ckv": (
                        code_ckv_row["median_seconds"] / actor_seconds
                    ),
                },
            }
        )

    payload = {
        "worker": "agentspec_drafter_tax",
        "schema_version": SCHEMA_VERSION,
        "synthetic": bool(args.synthetic),
        "evidence_valid_for_gate": not bool(args.synthetic),
        "claim_boundary": (
            "cost measurement only; the code head is randomly "
            "initialised and no match probability is measured"
        ),
        "device": str(device),
        "torch_version": torch.__version__,
        "transformers_version": _transformers_version(),
        "python_version": platform.python_version(),
        "cuda_device_name": (
            torch.cuda.get_device_name(device) if use_cuda else None
        ),
        "dtype": str(dtype),
        "code_vocabulary": args.code_vocabulary,
        "actor_identity": actor_identity,
        "drafter_identity": drafter_identity,
        "rows": rows,
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    payload["payload_sha256"] = hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()
    return payload


def summarise(payload):
    lines = [
        "worker          %s" % payload["worker"],
        "synthetic       %s" % payload["synthetic"],
        "gate evidence   %s" % payload["evidence_valid_for_gate"],
        "device          %s (%s)"
        % (payload["device"], payload["cuda_device_name"]),
        "actor           %s" % payload["actor_identity"]["path"],
        "drafter         %s" % payload["drafter_identity"]["path"],
        "",
        "context  actor_s  tau_text  tau_code  tau_code_ckv",
    ]
    for row in payload["rows"]:
        tax = row["measured_draft_gpu_tax"]
        lines.append(
            "%7d  %7.4f  %8.4f  %8.4f  %12.4f"
            % (
                row["context_length"],
                row["actor_seconds"],
                tax["text_drafter"],
                tax["code_drafter"],
                tax["code_drafter_ckv"],
            )
        )
    lines.append("")
    lines.append(
        "context  a_prefill  a_step_ms  d_step_ms  step_ratio  dec_share"
    )
    for row in payload["rows"]:
        d = row["decomposition"]
        ratio = d["decode_step_ratio_drafter_over_actor"]
        lines.append(
            "%7d  %9.4f  %9.3f  %9.3f  %10s  %9.3f"
            % (
                row["context_length"],
                d["actor_prefill_seconds"],
                d["actor_decode_step_seconds"] * 1000.0,
                d["drafter_decode_step_seconds"] * 1000.0,
                "%.4f" % ratio if ratio is not None else "n/a",
                d["decode_share_of_actor"],
            )
        )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Measure action drafter GPU tax",
    )
    parser.add_argument("--actor-model", default=None)
    parser.add_argument("--drafter-model", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--synthetic-actor-hidden", type=int, default=256)
    parser.add_argument("--synthetic-actor-layers", type=int, default=4)
    parser.add_argument(
        "--synthetic-drafter-hidden",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--synthetic-drafter-layers",
        type=int,
        default=2,
    )
    parser.add_argument("--synthetic-vocab", type=int, default=512)
    parser.add_argument(
        "--action-tokens",
        type=int,
        default=DEFAULT_ACTION_TOKENS,
    )
    parser.add_argument(
        "--context-lengths",
        type=int,
        nargs="+",
        default=list(DEFAULT_CONTEXT_LENGTHS),
    )
    parser.add_argument(
        "--compressed-budget",
        type=int,
        default=DEFAULT_COMPRESSED_BUDGET,
    )
    parser.add_argument(
        "--code-vocabulary",
        type=int,
        default=DEFAULT_CODE_VOCABULARY,
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=DEFAULT_REPETITIONS,
    )
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    if not args.synthetic:
        if not args.actor_model or not args.drafter_model:
            parser.error(
                "--actor-model and --drafter-model are required "
                "unless --synthetic is set"
            )
    payload = build_payload(args)
    if args.output is not None:
        directory = os.path.dirname(os.path.abspath(args.output))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(payload, indent=2, sort_keys=True) + "\n"
            )
    print(summarise(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
