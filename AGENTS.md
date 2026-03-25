# AGENTS.md (Draft)

## Project Overview
This repository is a **lightweight VLM/LLM inference stack** focused on simple, readable, and modular inference-time components.

## Core Files and Responsibilities
- `config.py`: Centralized runtime/model configuration definitions and parameter wiring.
- `qwen3.py`: Qwen3 model-side logic and integration entry points used during inference.
- `sampler.py`: Token sampling strategies (e.g., temperature/top-k/top-p style decoding behavior).
- `loader.py`: Model/tokenizer/checkpoint loading and initialization flow.

## Engineering Rules
- Make the **smallest possible change** to solve the task.
- Do **not** change public APIs casually; preserve backward compatibility by default.
- For bug fixes, explain the **root cause first**, then describe the fix.
- Any behavior change must include or update tests to cover the new behavior.
