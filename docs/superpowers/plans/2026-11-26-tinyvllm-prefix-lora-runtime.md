# `tinyvllm` Native Prefix-Promoted Multi-LoRA Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate prefix-hash-triggered `7.31 MiB` BF16 Multi-LoRA slot execution natively into `tinyvllm/` (`PrefixLoRARegistry`, `QKVLoRASlotPool`, `QKVParallelLinear`, `ModelRunner` CUDA Graph capture/replay, and `LLMEngine.add_request`), then verify mixed-batch correctness and throughput on `Qwen3-8B` on 1× A100 80GB.

**Architecture:** `tinyvllm/lora/prefix_lora_manager.py` provides `PrefixLoRARegistry` (exact token-prefix matching and transparent prompt trimming at `LLMEngine.add_request`) and `QKVLoRASlotPool` (pre-allocated `[max_slots + 1, ...]` BF16 slot tensors with `slot=0` permanently zeroed as the unadapted sentinel, computing mixed-batch `dq, dv` via `index_select + torch.bmm`). `Context.lora_token_slot_ids` and `ModelRunner.graph_vars["lora_token_slot_ids"]` propagate per-token slot IDs across prefill, decode, and captured CUDA Graphs without graph recapture when switching adapters.

**Tech Stack:** Python `3.12`, PyTorch `2.5+` (`torch.bmm`, `torch.cuda.CUDAGraph`), `tinyvllm` Paged KV Cache & FlashAttention runtime.

## Global Constraints

- Default `Config.prefix_lora_max_slots = 0` must incur zero VRAM allocation and zero runtime overhead on existing `tinyvllm` code paths.
- Slot `0` in `QKVLoRASlotPool` is permanently zeroed and immutable; valid user adapter slots are `1 .. prefix_lora_max_slots`.
- Preserve all existing `tinyvllm` tests (`tools/test_chunked_prefill.py`, `tools/test_profile_prefix_cache.py`) and all `experiments/lora_necessity/` files untouched.
- Zero internal hostnames, usernames, UIDs, IPs, or remote paths in any committed file.

---

### Task 1: Standalone `PrefixLoRARegistry` & `QKVLoRASlotPool` (`tinyvllm/lora/`)

**Files:**
- Create: `tinyvllm/lora/__init__.py`
- Create: `tinyvllm/lora/prefix_lora_manager.py`
- Test: `tools/test_prefix_lora_runtime.py`

**Interfaces:**
- Consumes: `torch`, `torch.nn`
- Produces:
  - `PrefixLoRARegistry(max_slots: int)` with `.register_promoted_prefix(slot_id, long_prefix_ids, short_prefix_ids, *, name="")`, `.unregister_slot(slot_id)`, `.match_and_rewrite(prompt_token_ids, explicit_slot_id=0) -> tuple[list[int], int]`
  - `QKVLoRASlotPool(max_slots: int, rank: int, hidden_size: int, q_shard_size: int, kv_shard_size: int, *, dtype=torch.bfloat16, device="cpu")` with `.load_slot(slot_id, a_q, b_q, a_v, b_v, *, scale)`, `.clear_slot(slot_id)`, `.apply(x: torch.Tensor, base_qkv: torch.Tensor, slot_ids: torch.Tensor) -> torch.Tensor`
  - `install_qkv_lora_pools(model: torch.nn.Module, max_slots: int, rank: int) -> int`
  - `load_hf_lora_into_tinyvllm_slot(model: torch.nn.Module, slot_id: int, adapter_tensors: dict, *, alpha: float = 16.0, rank: int = 8) -> dict`

- [ ] **Step 1: Write failing unit tests in `tools/test_prefix_lora_runtime.py` covering `PrefixLoRARegistry` and `QKVLoRASlotPool`**
- [ ] **Step 2: Run `/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python -B -m unittest tools/test_prefix_lora_runtime.py -v` and confirm RED failure**
- [ ] **Step 3: Implement `tinyvllm/lora/__init__.py` and `tinyvllm/lora/prefix_lora_manager.py`**
- [ ] **Step 4: Run `tools/test_prefix_lora_runtime.py` and confirm GREEN**

---

### Task 2: Hook `Config`, `SamplingParams`, `Sequence`, `Context`, `QKVParallelLinear`, `ModelRunner`, & `LLMEngine`

**Files:**
- Modify: `tinyvllm/config.py`
- Modify: `tinyvllm/sampling_params.py`
- Modify: `tinyvllm/engine/sequence.py`
- Modify: `tinyvllm/utils/context.py`
- Modify: `tinyvllm/layers/linear.py`
- Modify: `tinyvllm/engine/model_runner.py`
- Modify: `tinyvllm/engine/llm_engine.py`
- Test: `tools/test_prefix_lora_runtime.py`

**Interfaces:**
- Consumes: `tinyvllm.lora.{PrefixLoRARegistry, QKVLoRASlotPool, install_qkv_lora_pools, load_hf_lora_into_tinyvllm_slot}`
- Produces:
  - `Config.prefix_lora_max_slots: int = 0`, `Config.prefix_lora_rank: int = 8`
  - `SamplingParams.lora_slot_id: int = 0`, `Sequence.lora_slot_id: int`
  - `Context.lora_token_slot_ids: torch.Tensor | None = None`
  - `LLMEngine.register_promoted_prefix_lora(slot_id, adapter_tensors, *, long_prefix_ids=None, short_prefix_ids=None, alpha=16.0, rank=8, name="")`

- [ ] **Step 1: Add failing integration tests in `tools/test_prefix_lora_runtime.py` testing `QKVParallelLinear` + `Context.lora_token_slot_ids` mixed-batch equivalence (`slot=[0, 1, 2, 0]`), `Sequence` serialization, and `Config` validation**
- [ ] **Step 2: Run tests to confirm RED failure**
- [ ] **Step 3: Add minimal hooks across the 7 `tinyvllm` files (including `ModelRunner.prepare_prefill`, `prepare_decode`, `capture_cudagraph`, and `run_model` `graph_vars` copy)**
- [ ] **Step 4: Run `tools/test_prefix_lora_runtime.py`, `tools/test_chunked_prefill.py`, and `tools/test_profile_prefix_cache.py` to confirm 100% GREEN**

---

### Task 3: Live Remote A100 `tinyvllm` End-to-End Multi-LoRA + CUDA Graph Verification & Benchmark

**Files:**
- Create: `experiments/lora_necessity/tinyvllm_native_lora_e2e_runner.py`
- Create: `experiments/lora_necessity/20261126-tinyvllm-native-prefix-lora-results.md`

- [ ] **Step 1: Build and locally syntax/unit-test `tinyvllm_native_lora_e2e_runner.py` (trains/loads `lora_diff_kl` into `slot=1` and `lora_csd` into `slot=2` inside real `tinyvllm.LLMEngine` on `Qwen3-8B` with CUDA Graph enabled, verifies transparent 3,500->450 prefix rewriting, mixed-batch multi-LoRA exactness vs single-request, and native `tinyvllm` Paged-KV throughput)**
- [ ] **Step 2: Execute single-shot live A100 `tinyvllm` end-to-end verification and record results in `experiments/lora_necessity/20261126-tinyvllm-native-prefix-lora-results.md`**
