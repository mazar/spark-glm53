# Changelog

All notable changes, hardware fixes, and benchmark updates to **`Spark-GLM53`** are documented in this file.

---

## [2026-10-23] — Cold-Boot UMA Fix, Long-Context Boundary-Checkpoint UMA Fix & Dual-Node Power/Thermal Telemetry

### Fixed
- **Long-Context Multi-Turn Boundary-Checkpoint UMA Exhaustion (`140K–313K+` Tokens) (`worker/b12x_dflash_runner.py`, `scripts/start_engine.sh`)**:
  - **2-Slot `mHC` Ping-Pong Buffer & `1 GiB` Layer-Boundary Reserve Cap:** Patched `b12x.norm.mhc._impl._b12x_mhc_post_pre_impl` and `Glm5NextModel.forward` (`n_toks > 64`) to reuse a 2-slot ping-pong buffer for `(residual_out, y_out, post_out, comb_out)` across all 160 `hc_fused_post_pre` calls per forward pass and capped unallocated `CUDACachingAllocator` reserve at `1 GiB` (`1024 MiB`), eliminating the `+64 MiB/call` (`+10.24 GiB`/forward) segment-splitting cascade at `n_toks ≈ 2,000–2,048` with zero impact on `llama-benchy` throughput (`c=1..8`).
  - **Zero-Allocation In-Place KV Block CoW (`copy_kv_cache_blocks_inplace`):** Replaced advanced-indexing `blocks[dst] = blocks[src]` (which allocated `K × 29.6 MiB` temporary GPU buffers on prefix-cache copy-on-write) with in-place `blocks[d_blk].copy_(blocks[s_blk], non_blocking=True)`.
  - **Direct 2D UVA NumPy `stage_write` & Pinned Host Cache Release:** Patched `StagedWriteTensor.stage_write` (`all_token_ids`, `prefill_positions`, `trace_token_ids`) to write directly into `_uva_buf.np[index, start:end]` on CPU without growing `UvaBufferPool` (`CachingHostAllocator`) and added `torch._C._host_emptyCache()` to `_spark_release_idle_uma_caches`.
  - **Boot UMA Headroom & `start_engine.sh` Hygiene:** Tuned default `KV_CACHE_MEMORY_BYTES` to `14600000000` (`13.60 GiB`, `2.36M` FP8 KV tokens, leaving `~4.0–4.3 GiB` free UMA on Head and `~6.3–6.6 GiB` on Worker) and bracket-escaped `pkill -9 -f "[_]offline_compiler_spawn_environment"` so step 1b never matches its own `sh -c` subshell.
- **Multi-Chunk Prefill UMA Exhaustion (`earlyoom` on 18K–30K+ Token Prompts) (`worker/b12x_dflash_runner.py`)**:
  - Synchronized the CUDA stream (`torch.cuda.synchronize()`) before `torch.cuda.empty_cache()` on prefill chunks (`n_toks > 64`) in `_patched_execute_model` and `_patched_sample_tokens` so intermediate prefill chunks (`num_logits == 0`) do not queue ahead of the GPU and accumulate `CUDACachingAllocator` scratch buffers across chunks.
  - Lowered the `nvfp4_materialize_intermediate = False` (`dynamic_tile_m = 64`) threshold in `_sanitize_moe_assignment` and `_patched_job_lookup` from `> 2048` to `>= 256` tokens so prefill tail chunks between `256` and `2,048` tokens do not allocate intermediate materialization buffers.
- **Cold-Boot `/dev/shm` Duplicate Dense Allocation (`earlyoom` after `./stop.sh --purge-shm`)** ([#4](https://github.com/mazar/spark-glm53/pull/4), `2c952af`):
  - Prevented `_bind_shm_views_to_model(..., is_cold_rebind=True)` in `worker/shm_safetensors.py` and `scripts/start_engine.sh` (`SPARK_SHM_MATERIALIZE_DENSE=0`) from cloning an extra `4.75 GiB` of dense/MXFP8 weights into `cudaMalloc` while `85.28 GiB` is still resident in `/dev/shm` during a first-time cold export (before `spark-cuda-ipc-weights` migrates slabs into GPU memory and punches holes in `/dev/shm`).
- **Repository & Benchmark Links in `README.md`** ([#3](https://github.com/mazar/spark-glm53/pull/3), `187b69e`):
  - Updated and verified links to [`SeraphimSerapis/tool-eval-bench`](https://github.com/SeraphimSerapis/tool-eval-bench), [`eugr/llama-benchy`](https://github.com/eugr/llama-benchy), [`MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks`](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks), [`MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold`](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold), and [`eugr/spark-vllm-docker`](https://github.com/eugr/spark-vllm-docker).

### Added
- **Strict Served-Model Validation, Runtime Alias Config (`config/model_aliases.json`), Effort Mapping & `/v1/messages/count_tokens` (`pkg/gateway/main.go`)**:
  - Enforced strict served-model validation (`GLM-5.3-Flash` by default) across `/v1/chat/completions`, `/v1/responses`, `/v1/messages`, `/v1/messages/count_tokens`, and `/v1/models/{id}`, returning HTTP `404` when an unconfigured model name is requested while allowing users to define runtime model aliases (and optional per-alias default effort) via `config/model_aliases.json` (`ModTime` hot-reload) or `--served-models` / `SERVED_MODEL_NAMES`.
  - Added support for `output_config.effort`, `thinking.type: "adaptive"`, `"xhigh"` $\rightarrow$ `"max"` effort mapping, HTTP `400` validation for unsupported effort values, `supported_reasoning_efforts` metadata on `/v1/models`, and `POST /v1/messages/count_tokens`.
- **Side-by-Side `Spark-GLM53` vs. `TensorFold` Throughput, Power & Thermal Benchmarks** ([#2](https://github.com/mazar/spark-glm53/pull/2), `c708561`):
  - Benchmarked `MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold` (`PARALLEL=8`) head-to-head against `Spark-GLM53` on the exact same 2x DGX Spark cluster using `llama-benchy` (`--pp 512 --tg 256 --exact-tg --no-cache --concurrency 1 2 4 8`) and `scripts/run_power_temp_bench.py`:
    - **Cluster Active Average GPU Power:** **`70.53 W`** (`Spark-GLM53`) vs. `122.05 W` (`TensorFold`) — **`42.2%` lower power (`-51.5 W`)**.
    - **Peak GPU Temperature:** **`64.0–69.0°C`** (`Spark-GLM53`) vs. `78.0–79.0°C` (`TensorFold`) — **`10°C–14°C` cooler GPUs** with zero thermal SM clock throttling.
    - **`c=8` Prefill & Peak Decode:** **`2,237.61 tok/s` prefill / `165.00 tok/s` peak decode** (`Spark-GLM53`) vs. `1,686.10 tok/s` prefill / `152.33 tok/s` peak decode (`TensorFold`).
- **1 Hz Dual-Node Power (`W`/`Wh`/`J`) & Thermal (`°C`) Telemetry Wrapper (`scripts/run_power_temp_bench.py`)** ([#1](https://github.com/mazar/spark-glm53/pull/1), `277dd8f`):
  - Added `scripts/run_power_temp_bench.py` to sample `nvidia-smi` (`power.draw,temperature.gpu,clocks.current.sm,utilization.gpu`) and `/sys/class/thermal/thermal_zone0/temp` at 1 Hz across both Head (`spark-5a4e`) and Worker (`spark-8a27`) over a single persistent SSH stream.

---

## [2026-10-22] — Initial Public Release, Universal 2x DGX Spark Scripts & `95/100` `tool-eval-bench` Hardmode

### Added
- **Apache 2.0 Public Release & Universal 2x DGX Spark Scripts** (`e98ab27`, `1773e08`, `09c03be`, `eac50e2`, `647422e`, `2c069c0`):
  - Made `scripts/start_engine.sh`, `scripts/tune_sparks.sh`, `scripts/fetch_nvidia_ckpt.sh`, `stop.sh`, and `pkg/shmdaemon/main.go` universal across any 2x DGX Spark cluster (`$HOME`/`$HOST_CACHE_DIR` auto-detection, dynamic HuggingFace snapshot resolution, automatic `WORKER_HOST` discovery, auto-building `bin/spark-gateway` and `bin/spark-shm-weights`, and counting already-pinned `spark-cuda-ipc-weights` / `spark-shm-weights` memory in `tune_sparks.sh --verify`).
- **Compiled Go Control Plane (`pkg/gateway/main.go` on `:8888`)**:
  - **256-Token Block-Aligned Radix Prefix Router & NVMe `O_DIRECT` Spill Tier:** Pins shared system prompts (`kept-state` protection) and spills evicted long-context KV states (`64K` multi-turn reuse TTFT in **`57.8–61.3 ms`**).
  - **Prefix-Cache-Safe `reasoning_effort` & Guided Decoding (`xgrammar`):** Supports per-request `reasoning_effort` (`none`, `low`, `medium`, `high`, `max`) and `enable_thinking: false` while preserving `<|system|>Reasoning Effort: Max` at token 0 so multi-turn tool $\rightarrow$ `json_schema` workflows retain a 100% KV prefix-cache hit and close `<think></think>` before `xgrammar` decoding starts.
  - **Schema-Aware `glm47_repair` Tool Parser & Loop Guard:** Parses `<think>` and `<tool_call>` blocks on both streaming SSE and non-streaming responses, repairs malformed JSON arguments, coerces XML scalar types against the request's JSON schema, and aborts runaway repetition loops (`6.9 ms` client abort latency).
  - **Dual OpenAI & Anthropic API Compatibility:** Serves `/v1/chat/completions`, `/v1/responses`, `/v1/models`, `/tokenize`, `/metrics`, and `/v1/messages`.

### Fixed
- **`tool-eval-bench --seed 42 --hardmode` (`95 / 100`, `168 / 176` pts, `14/16` categories at 100%)** (`e98ab27`):
  - Fixed multi-turn `tool_calls[].function.arguments` serialization so repaired JSON strings pass vLLM's OpenAI Pydantic validator on Turn 2+.
  - Fixed streaming `<think>...</think>` state machine (`inThink := reasoningCfg.EnableThinking`) so thinking tokens stream cleanly into `delta.reasoning_content` without leaking into `delta.content`.
  - Fixed `TC-11` (`tool: null` pseudo-tool rejection), `TC-64`/`TC-65` (structured output `<think></think>` closing with prefix-cache preservation), `TC-66`–`TC-69` (long-horizon 10–15 step autonomous planning loops), `TC-80` (adversarial sibling tool call sanitization), and `TC-85` (15-turn state tracking).
- **`~2.66 GiB` Eager Prefill UMA Spike in `Glm5NextLinearAttention.forward` (`MAX_BATCHED_TOKENS=4096`)** (`e98ab27`):
  - Gated `_projection_workspaces` and `_forward_gate_overlap` on `_gate_stream` in `worker/b12x_dflash_runner.py` to `hidden_states.size(0) <= 64` (decode & speculative verification), eliminating cross-stream `CUDACachingAllocator` block accumulation during 4,096-token eager prefill chunks (`+23.8%` prefill throughput, flat `0 MB` `cuda_mem` growth).
- **Low `enable_thinking=False` `DFlash2` (`K=7`) Draft Acceptance** (`e98ab27`):
  - Updated `worker/glm53_chat_template.jinja` and `pkg/gateway/main.go` to emit `<|assistant|><think></think>` when thinking is disabled, increasing `DFlash2` draft acceptance on structured code/JSON from `~38%` to **`85.0%–98.6%`** (`188.5–208.3 tok/s` across 8 distinct code streams).

---

## [2026-10-20] — Multi-Stream Crosstalk Fix & Persistent Weight Daemons

### Fixed
- **Multi-Stream Homogeneous-Batch Crosstalk Bug** (`8651697`):
  - Disabled length-only homogeneous-batch MoE deduplication (`SPARK_HOMOG_MOE_DEDUP=0`) by default so distinct concurrent prompts of equal token length never alias hidden states (`100%` `4/4` bitwise determinism between concurrent and serial streams).

### Added
- **Persistent CUDA IPC (`worker/cuda_ipc_daemon.py`) & POSIX `/dev/shm` (`pkg/shmdaemon/main.go`) Weight Daemons** (`131bc27`, `a007c06`):
  - Eliminated the GB10 UMA Linux page-cache `NV_ERR_NO_MEMORY` crash via `O_DIRECT` / `POSIX_FADV_DONTNEED` reads and 23 persistent `4.0 GiB` `cudaMalloc` IPC slabs (`85.28 GiB`/rank), reducing worker warm weight reload time from ~180s to **`0.87s – 1.59s`**.
  - Added custom `sm_121a` split-KV `DFlash2` attention kernel (`worker/kernels/dflash_attn.cu`, capped to GB10's 99 KB shared-memory limit) and 128/256 LCM KV page alignment (`--mamba-cache-mode align --block-size 256`).
