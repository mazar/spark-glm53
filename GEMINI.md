# GEMINI.md — Autonomous Agent Specification: `Spark-GLM53` Engine

## 1. Mission & Performance Targets
You are running directly on the **Head Node** of a **2x NVIDIA DGX Spark (`sm_121a`, GB10 Grace Blackwell)** cluster connected via dual-port ConnectX-7 RoCE v2. Your objective is to autonomously design, implement, test, and benchmark **`Spark-GLM53`**—a high-throughput, low-latency distributed inference engine for **`GLM-5.3-Flash`** (`Glm5NextForConditionalGeneration`, 320B total / 18B active parameters, NoPE-MLA + KDA linear attention).

You must outperform both existing community baselines:
1. **MiaAI-Lab (`GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold`):** Reaches 60.4 tok/s (1-stream prose) / 114.7 tok/s (structured) and 2.9M FP8 KV cache via EXL3 4bpw + `incoai/GLM-5.3-Flash-DFlash2` + copy-drafts, **but** suffers from slow **~1,950 tok/s prefill** (software EXL3 dequantization), lossy `DENSE=q4` weights, 2–6 minute cold restarts, and 96 brittle Python monkey-patches.
2. **Community `b12x` vLLM (`eugr/spark-vllm-b12x` + `nvidia/GLM-5.3-Flash-FP4-Local`):** Uses NVIDIA's Oct 16, 2026 QAD-quantized NVFP4 checkpoint (`modelopt_mixed` with compressed scales) for **3,000+ tok/s prefill**, **but** lacks fused CUDA-graph `DFlash2` + Copy-Draft speculation, suffers from GB10 UMA page-cache `NV_ERR_NO_MEMORY` crashes, and reloads 166 GiB from disk on every restart.

### Hard Acceptance Targets (2x DGX Spark, TP=2)
| Metric | MiaAI-Lab Baseline | Stock `b12x` Baseline | `Spark-GLM53` Target (Must Hit) |
| :--- | :--- | :--- | :--- |
| **Prefill (8K–64K prompt)** | ~1,950 tok/s | ~2,900 tok/s | **>= 3,600 tok/s** (NVFP4 `sm_121a` `B12X` + 4-rail RoCE) |
| **Decode (1-stream Prose)** | 60.4 tok/s | ~28–35 tok/s | **>= 62.0 tok/s** (`FULL_AND_PIECEWISE` CUDA Graphs + DFlash2/MTP) |
| **Decode (1-stream Code/JSON)**| 114.7 tok/s | ~45 tok/s | **>= 120.0 tok/s** (DFlash2 K=7 + AST/Suffix Copy-Drafts) |
| **Decode (8-stream Code)** | 167.0 tok/s | ~95 tok/s | **>= 185.0 tok/s** |
| **Quality (GSM8K / HumanEval)**| `DENSE=q4` (degraded) | QAD `modelopt_mixed` | **Zero degradation vs `nvidia/GLM-5.3-Flash-FP4-Local` (BF16 dense/MLA/routers)** |
| **KV Cache Pool (FP8)** | 2.9M tokens | ~2.0M tokens | **>= 2.2M tokens** (1,048,576 max context window) |
| **Worker Warm Restart Time** | 120–360 sec | 180 sec | **< 3.0 seconds** (Persistent `/dev/shm` `MAP_LOCKED` weight daemon) |

---

## 2. Hardware & Cluster Environment Discovery (Step 0)
Before writing code, inspect the live cluster environment autonomously:
1. **Discover Local & Peer Nodes:**
   - Inspect local network interfaces (`ip -br addr`, `ibdev2netdev`, `rdma link`).
   - Note that each physical ConnectX-7 QSFP port on GB10 bridges two PCIe Gen5 x4 links and exposes **twin netdevs / RoCE HCAs**: `enp1s0f0np0` + `enP2p1s0f0np0` (`rocep1s0f0,roceP2p1s0f0`) and `enp1s0f1np1` + `enP2p1s0f1np1` (`rocep1s0f1,roceP2p1s0f1`).
   - Locate the peer Spark worker node via `~/.ssh/config`, `/etc/hosts`, ARP table, or existing `scripts/local.sh` / env vars. Verify passwordless SSH (`ssh -o BatchMode=yes <worker> true`).
2. **Verify Checkpoints & Containers:**
   - Check `~/.cache/huggingface/hub` on both nodes for `nvidia/GLM-5.3-Flash-FP4-Local` (~166 GiB), `incoai/GLM-5.3-Flash-DFlash2` (~2.6 GiB), or `Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold`.
   - Check local Docker images (`docker images`) for `eugr/spark-vllm-b12x:latest` or TensorFold containers.
3. **Toolchains:**
   - Use **`uv`** for all Python virtual environments and package management.
   - Use **Go (1.23+)** (with **Bazel** or native `go build` depending on repo layout) for the Control Plane and Shared Memory Daemon.
   - Target **`sm_121a`** (`CUTE_DSL_ARCH="sm_121a"`) with CUDA 13 / FlashInfer (`local-inference-lab/flashinfer@main` with integrated `b12x`).

---

## 3. Critical GB10 (`sm_121a`) Hardware Bugs You Must Guard Against
Do **not** repeat known GB10 failure modes. Bake these invariants into your code:
1. **The GB10 UMA Page-Cache Trap (`NVRM: NV_ERR_NO_MEMORY`):**
   - Reading an 83 GiB weight shard per node fills the Linux page cache. When the NVIDIA driver allocates the contiguous KV slab via `_memdescAllocInternal`, it **refuses to reclaim Linux page cache** and crashes with `NV_ERR_NO_MEMORY` even when `MemAvailable` shows 35+ GiB free.
   - **Mandatory Fix:** Never read weights through buffered OS page cache during engine boot. Load weights once via `O_DIRECT` into a persistent `/dev/shm` POSIX shared-memory daemon (`mmap` with `MAP_SHARED | MAP_LOCKED`), or run an active cache-flusher (`posix_fadvise(..., POSIX_FADV_DONTNEED)` + `/proc/sys/vm/drop_caches`) before CUDA KV allocation.
2. **Headless Display Carve-Out (`dispram`):**
   - Reclaim ~2 GiB of unified memory per node by disabling the framebuffer console / display reservation when running headless (`nvidia_drm modeset=1 fbdev=0` or unloading desktop display reservations).
3. **NoPE-MLA `sm_121a` & FlashInfer Traps:**
   - `GLM-5.3-Flash` uses **NoPE MLA (`qk_rope_head_dim=0`)** interleaved with **KDA linear attention** and **Manifold-Constrained Hyper-Connections**.
   - Ensure the sparse-MLA kpool indexer initializes its top-$k$ buffer to `-1` and clamps indices (uninitialized `torch.empty` entries gather unwritten KV pages and produce intermittent `NaN`s).
   - Cap FA2 FP8 CTA shared memory tiles to `<= 99 KB` (Hopper's 228 KB default fails `cudaFuncSetAttribute` on GB10's ~101 KB opt-in max).
   - Enforce `VLLM_SSM_CONV_STATE_LAYOUT="DS"`, `--mamba-cache-mode align`, and `--block-size 256` (for `B12X`) so hybrid Mamba/KDA state pages align cleanly with MLA KV pages.
   - Audit transitive Python dependencies: never allow `nvidia-nccl-cu13` to downgrade to `2.29.7` (causes fabric internal errors).

---

## 4. Repository Architecture & Phased Build Plan

Create the following modular directory structure and complete each phase sequentially. **Do not advance to the next phase until the verification gate passes.**

```text
spark-glm53/
├── GEMINI.md
├── scripts/
│   ├── tune_sparks.sh          # RoCE v2 multirail, sysctl, drop_caches, headless UMA reclaim
│   └── benchmark_suite.py      # TTFT, 1..8 stream prose/code tok/s, needle-in-haystack, GSM8K/HumanEval
├── pkg/
│   ├── shmdaemon/              # Go + Cgo POSIX /dev/shm locked weight server & CUDA IPC / mmap bridge
│   └── gateway/                # Go Control Plane: Radix prefix router, Copy-Draft engine, glm47 tool parser
└── worker/
    ├── b12x_dflash_runner.py   # Patched sm_121a B12X + DFlash2 + Copy-Draft speculative worker
    ├── shm_safetensors.py      # Zero-copy safetensors tensor loader backed by /dev/shm pointers
    └── kernels/                # Custom Triton / CuTe DSL sm_121a kernels (fused verify + tree mask)
```

### Phase 1: Dual-Rail RoCE v2 & GB10 UMA Hygiene (`scripts/tune_sparks.sh`)
Write an idempotent cluster preparation script that runs across Head and Worker via SSH:
- Configures all available ConnectX-7 twin interfaces (`enp1s0f0np0`, `enP2p1s0f0np0`, and port 1 twins if cabled) on isolated `/24` subnets with MTU 4200 / DSCP 26 / PFC enabled and IPv6 link-local disabled so **RoCE v2 GID index 3** is deterministic.
- Validates line-rate RDMA bandwidth using `ib_write_bw -d rocep1s0f0` and `roceP2p1s0f0` (must hit **>= 215 Gbps aggregate** across dual x4 lanes, or **>= 420 Gbps** if dual-cabled).
- Sets required environment variables for `sm_121a`:
  ```bash
  export CUTE_DSL_ARCH="sm_121a"
  export SAFETENSORS_FAST_GPU="1"
  export VLLM_ENABLE_ROCE_ALLREDUCE="1"
  export VLLM_ROCE_ALLREDUCE_MAX_SIZE="2MB"
  export NCCL_IB_HCA="rocep1s0f0,roceP2p1s0f0"
  export NCCL_CROSS_NIC="1"
  export NCCL_BUFFSIZE=1048576
  export NCCL_MAX_CTAS=8
  export VLLM_SSM_CONV_STATE_LAYOUT="DS"
  export VLLM_USE_AOT_COMPILE="1"
  ```
- **Gate 1:** Execute `scripts/tune_sparks.sh --verify` across both nodes and log measured RDMA bandwidth and free contiguous UMA (`>112 GiB` per node).

### Phase 2: Persistent `/dev/shm` Zero-Copy Weight Daemon (`pkg/shmdaemon` + `worker/shm_safetensors.py`)
Eliminate both the 3-minute cold reload and the GB10 page-cache `NV_ERR_NO_MEMORY` bug:
- Build a lightweight Go/C daemon (`spark-shm-weights`) that runs persistently on both Sparks.
- On startup, it reads the rank-local tensor shards of `nvidia/GLM-5.3-Flash-FP4-Local` (~83 GiB/node) and `incoai/GLM-5.3-Flash-DFlash2` (~1.3 GiB/node) using `O_DIRECT` (bypassing Linux page cache) directly into a named POSIX shared memory segment (`/dev/shm/glm53_rank_{0,1}`) pinned with `mlock()`.
- Write `worker/shm_safetensors.py` to monkey-patch `safetensors.torch.load_file` / vLLM's `ModelOptFp4` weight loader: map the `/dev/shm` segment via `cudaHostRegister(..., cudaHostRegisterIoMemory | cudaHostRegisterMapped)` so the GPU reads/copies weights into device-mapped UMA in **< 2.0 seconds** with zero disk I/O and zero page-cache bloat.
- **Gate 2:** Kill and restart the Python worker process twice; verify weight load time drops from ~180s to **< 3.0s** and `/proc/meminfo` `Cached:` does not grow.

### Phase 3: `sm_121a` `B12X` + `DFlash2` + Copy-Draft Speculative Engine (`worker/`)
Fuse the best of `b12x` (3,500+ tok/s NVFP4 prefill) with MiaAI-Lab's speculative decode speed:
- Base the worker on `nvidia/GLM-5.3-Flash-FP4-Local` (`quantization="modelopt_mixed"`, `attention_backend="B12X"`, `moe_backend="b12x"`, `kv_cache_dtype="fp8"`, `block_size=256`).
- Integrate **`incoai/GLM-5.3-Flash-DFlash2` (K=7)** alongside a **Zero-Overhead Suffix/N-Gram Copy-Drafter** (which checks prompt + recent output tokens for exact code/JSON structural matches):
  - Previous community DFlash2 overlays forced `--enforce-eager` and lost KV pool capacity because auxiliary hidden-state capture broke piecewise CUDA graphs and KV page LCM alignment.
  - **Your Fix:** Align the DFlash2 drafter KV block size to the `256`-token LCM of the `B12X` target runner and pre-allocate a static auxiliary hidden-state capture buffer so **`--compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'`** stays enabled during speculative verification!
  - Implement **Dynamic Draft Switching**: when the suffix copy-drafter finds a high-confidence match (`>= 4` matching prefix tokens in code/JSON), verify a 12-token copy-draft tree; otherwise fall back to the 7-token `DFlash2` draft batch.
- **Gate 3:** Run `scripts/benchmark_suite.py`:
  - Verify prefill throughput on an 8K and 32K prompt is **>= 3,500 tok/s**.
  - Verify 1-stream decode is **>= 60 tok/s** on prose and **>= 115 tok/s** on code/structured outputs.
  - Verify multi-stream (`PARALLEL=4` and `PARALLEL=8`) output tokens are bitwise/semantically deterministic (`drafted == serial`).

### Phase 4: High-Throughput Go Control Plane (`pkg/gateway`)
Replace fragile Python HTTP/state monkey-patches with a compiled Go gateway on port `8888` fronting the internal engine on port `8000`:
- **256-Token Block-Aligned Radix Prefix Router & Spill Tier:**
  - Track active KV cache blocks in Go. Keep system prompts pinned across multi-turn agent loops (`kept-state` protection so long conversations never evict the shared agent system prompt).
  - Implement an asynchronous **NVMe KV Spill Tier** (`O_DIRECT` via `io_uring`/pre-allocated files) so evicted long-context prompt states (e.g., 64K–256K repo contexts) can be swapped back into UMA at 6+ GB/s instead of re-prefilling at 3,600 tok/s.
- **Deterministic `reasoning_effort` & `clear_thinking` State Machine:**
  - Support `reasoning_effort` (`low`, `medium`, `high`, `max`) by injecting the exact GLM-5.3 tail directive and enforcing token-budget soft/hard caps on `<think>...</think>` blocks.
  - Automatically strip historical `reasoning_content` (`clear_thinking=true`) on multi-turn requests so prior thinking traces never pollute the KV cache or break prefix-cache hits.
- **Streaming `glm47_repair` Tool Parser & Loop Guard:**
  - Parse `<|assistant|>`, `<think>`, and tool calls in Go on the streaming byte buffer, auto-repairing malformed JSON arguments and immediately aborting runaway repetition loops or disconnected client streams (freeing GPU slots within 1 decode step).
- **Dual API Compatibility:** Expose both OpenAI (`/v1/chat/completions`, `/v1/responses`, `/v1/models`, `/tokenize`, `/metrics`) and Anthropic (`/v1/messages`) endpoints.
- **Gate 4:** Run the full Hermes Agent / 20-tool gauntlet and multi-turn 64K prompt reuse test (verify second turn TTFT **< 100 ms**).

---

## 5. Autonomous Operating Rules for `agy`
1. **Work Iteratively & Measure Everything:** After every change, run a quick smoke test and log real numbers (`TTFT`, `prefill tok/s`, `decode tok/s`, `free GiB`, `KV pool tokens`) into `BENCHMARKS.md`.
2. **Never Hang the Cluster:** Always start the Worker rank before the Head rank, set a 45-second timeout on distributed `Gloo`/`NCCL` init, and ensure `./stop.sh` cleanly kills stale Python/vLLM/TensorFold processes on both nodes without wiping the persistent `/dev/shm/glm53_rank_*` weight segments unless `--purge-shm` is passed.
3. **First Action:** Start immediately by inspecting the local machine (`nvidia-smi`, `ip -br a`, `ibdev2netdev`, `df -h`, `~/.cache/huggingface/hub`, `docker images`), discovering the second DGX Spark node, and creating `BENCHMARKS.md` with the current baseline environment state.

