# Spark-GLM53: High-Throughput NVFP4 + DFlash2 Distributed Inference Engine for 2x NVIDIA DGX Spark (`sm_121a`)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Hardware](https://img.shields.io/badge/Hardware-2x_NVIDIA_DGX_Spark_(GB10_sm__121a)-76B900.svg)](#hardware--cluster-requirements)
[![Model](https://img.shields.io/badge/Model-GLM--5.3--Flash_(320B_/_18B_Active)-purple.svg)](#overview)

**`Spark-GLM53`** is a distributed Tensor-Parallel (`TP=2`) inference engine and Go control plane for **`GLM-5.3-Flash`** (`Glm5NextForConditionalGeneration`, 320B total / 18B active parameters, NoPE-MLA + KDA linear attention + Manifold-Constrained Hyper-Connections) running on **2x NVIDIA DGX Spark (`sm_121a`, GB10 Grace Blackwell)** nodes connected via dual-rail ConnectX-7 RoCE v2.

It combines:
1. **Native Blackwell `sm_121a` `B12X` NVFP4 + MXFP8 Kernels** (`modelopt_mixed`, preserving full BF16/MXFP8 precision on dense MLA, shared experts, and routers)
2. **CUDA-Graph-Captured `DFlash2` (`K=7`) Speculative Decoding** (`incoai/GLM-5.3-Flash-DFlash2` with custom split-KV `dflash_attn.cu` on `sm_121a` and 128/256 LCM block alignment)
3. **Zero-Copy Persistent CUDA IPC / `/dev/shm` Weight Daemons** (`spark-cuda-ipc-weights` + `spark-shm-weights`), eliminating the 3-minute cold reload (**warm restart in `< 1.6s`**) and preventing the GB10 UMA Linux page-cache `NV_ERR_NO_MEMORY` crash
4. **Compiled Go Control Plane (`spark-gateway`)** fronting the engine on port `8888` with a 256-token block-aligned Radix Prefix Router, NVMe `O_DIRECT` KV spill tier, deterministic `reasoning_effort` (`low`/`medium`/`high`/`max`) & `clear_thinking` state machine, streaming `glm47_repair` XML/JSON tool parser, `xgrammar` structured-output `<think></think>` framing, and dual OpenAI (`/v1/chat/completions`, `/v1/responses`) + Anthropic (`/v1/messages`) API compatibility.

---

## Benchmark Comparison: `Spark-GLM53` (NVFP4) vs. MiaAI-Lab `GLM-5.3-Flash-EXL3`

Evaluated head-to-head on the exact same 2x NVIDIA DGX Spark cluster using [`tool-eval-bench`](https://github.com/prettygood-software/llama-benchy) (`tool-eval-bench run --seed 42 --hardmode --base-url http://127.0.0.1:8888`, 88 scenarios across Categories A–P, `Reasoning Effort: Max`, `temperature: 0.0`):

| Metric (88 Scenarios, Hardmode, Seed 42) | MiaAI-Lab `GLM-5.3-Flash-EXL3` | `Spark-GLM53` (NVFP4 + `B12X` + `DFlash2`) | Delta |
| :--- | :--- | :--- | :--- |
| **Overall Score** | `94 / 100` (`165 / 176` pts) | **`95 / 100` (`168 / 176` pts)** | **+1% (+3 pts)** |
| **Structured Output (Category O)** | `83%` (`10 / 12` pts) | **`100%` (`12 / 12` pts)** | **+17% (+2 pts)** |
| **Autonomous Planning (Category M)** | `67%` (`4 / 6` pts) | **`100%` (`6 / 6` pts)** | **+33% (+2 pts)** |
| **Deployability Score** | `76` | **`77`** | **+1** |
| **Total Suite Wall-Clock Time** | `1,942.0 s` (`32.4 min`) | **`1,754.5 s` (`29.2 min`)** | **-187.5 s (~3.1 min faster)** |
| **Average First-Turn TTFT** | `1,429.4 ms` | **`921.2 ms`** | **-35.6% faster prefill** |
| **Median Turn Latency** | `4,811.0 ms` | **`4,343.5 ms`** | **-9.7% faster per turn** |
| **Total Completion Tokens Generated** | `58,704` | `56,977` | `-2.9%` (both `Reasoning Effort: Max`) |
| **8-Stream Concurrent Code Throughput** | `167.0 tok/s` | **`188.5 – 208.3 tok/s`** (`238.6` steady) | **+12.9% to +24.7%** |
| **Multi-Turn 64K Context Reuse TTFT** | ~450 ms | **`57.8 – 61.3 ms`** | **~7.5x faster** |
| **Worker Warm Restart Time (Weight Load)**| `120 – 360 s` | **`1.50 – 1.60 s`** (CUDA IPC) / **`0.87 s`** (`shm`) | **>100x faster** |

*(Dual-node power draw in Watts/Wh and thermal telemetry under matched DVFS clock policies will be added in an upcoming update).*

### Category-by-Category Breakdown (`tool-eval-bench --hardmode --seed 42`)
- **14 of 16 Categories at 100% (`PASS`):**
  - `A` Tool Selection (`6/6` — **100%**)
  - `B` Parameter Precision (`6/6` — **100%**)
  - `C` Multi-Step Chains (`8/8` — **100%**)
  - `D` Restraint & Refusal (`6/6` — **100%**)
  - `E` Error Recovery (`6/6` — **100%**)
  - `F` Localization (`6/6` — **100%**)
  - `G` Structured Reasoning (`6/6` — **100%**)
  - `H` Instruction Following (`10/10` — **100%**)
  - `I` Context & State (`20/20` — **100%**)
  - `J` Code Patterns (`6/6` — **100%**)
  - `L` Toolset Scale (`8/8` — **100%**)
  - `M` Autonomous Planning (`6/6` — **100%**)
  - `N` Creative Composition (`6/6` — **100%**)
  - `O` Structured Output (`12/12` — **100%**)
  - `K` Safety & Boundaries (`22/26` — **85%**)
  - `P` Hard Mode (`34/38` — **89%**)

---

## Hardware & Cluster Requirements

- **Nodes:** 2x NVIDIA DGX Spark (`sm_121a`, GB10 Grace Blackwell, 128 GB Unified LPDDR5X each)
- **Interconnect:** ConnectX-7 QSFP direct-attached cable (or switch) exposing twin PCIe Gen5 x4 RoCE v2 interfaces (`enp1s0f1np1` + `enP2p1s0f1np1` / `rocep1s0f1,roceP2p1s0f1`)
- **SSH:** Passwordless SSH configured from the Head node to the Worker node
- **Container Image:** `eugr/spark-vllm-b12x:latest` (tagged locally as `vllm-node-b12x:latest`) on both nodes
- **Model Checkpoints (in `~/.cache/huggingface/hub` on both nodes):**
  - Target Model: `local-inference-lab/GLM-5.3-Flash-NVFP4-Spark` (or `nvidia/GLM-5.3-Flash-FP4-Local`)
  - Speculative Drafter (`K=7`): `incoai/GLM-5.3-Flash-DFlash2`

---

## Quick Start (Any 2x DGX Spark Cluster)

All scripts auto-discover your `$HOME` directory, HuggingFace snapshot hashes, and peer DGX Spark worker node (`WORKER_HOST`). You can also override any setting via environment variables.

### 1. Clone the Repository & Build the Go Binaries
Run on the **Head Node**:
```bash
git clone https://github.com/mazar/spark-glm53.git
cd spark-glm53

# Build the Go Control Plane (spark-gateway) and POSIX /dev/shm Weight Daemon (spark-shm-weights)
mkdir -p bin
go build -o bin/spark-gateway ./pkg/gateway
go build -o bin/spark-shm-weights ./pkg/shmdaemon

# Copy the repository to the same path on the Worker node
export WORKER_HOST="${WORKER_HOST:-<your-worker-hostname-or-ip>}"
rsync -az --exclude='.git' --exclude='logs' --exclude='runs' ./ "$WORKER_HOST:$(pwd)/"
```

### 2. Download Checkpoints (If Not Already Cached)
Run on **both** Head and Worker nodes (uses a background `POSIX_FADV_DONTNEED` loop so downloading 175 GiB never fills the GB10 Linux page cache):
```bash
./scripts/fetch_nvidia_ckpt.sh
uvx --from 'huggingface_hub[hf_transfer]' hf download incoai/GLM-5.3-Flash-DFlash2
```

### 3. Tune Dual-Rail ConnectX-7 RoCE v2 & GB10 Unified Memory (Phase 1)
Configures MTU 4200, DSCP 26, deterministic RoCE v2 `GID[3]`, `108 GiB` hugepage-backed `/dev/shm`, `earlyoom` protection, and verifies `>= 215 Gbps` RDMA bandwidth + `> 112 GiB` available UMA per node:
```bash
WORKER_HOST="$WORKER_HOST" ./scripts/tune_sparks.sh --verify
```

### 4. Start the Distributed Engine + Go Control Plane
Starts Worker Rank 1 before Head Rank 0, initializes the persistent `85.28 GiB`/rank CUDA IPC weight slabs (first boot exports the rank shards once; all subsequent restarts load weights in **`~1.5 seconds`**), and launches `spark-gateway` on port **`8888`**:
```bash
WORKER_HOST="$WORKER_HOST" ./scripts/start_engine.sh
```

 Optional environment overrides for `start_engine.sh`:
- `WORKER_HOST`: Hostname or IP of the second DGX Spark (default: auto-detected via SSH on `10.100.240.2` / `spark-8a27`)
- `HEAD_ROCE_IP` / `WORKER_ROCE_IP`: RoCE v2 IP addresses (default: `10.100.240.1` / `10.100.240.2`)
- `GATEWAY_PORT`: Public OpenAI/Anthropic gateway port (default: `8888`)
- `ENGINE_PORT`: Internal vLLM engine port (default: `8000`)
- `KV_CACHE_MEMORY_BYTES`: Per-rank FP8 KV cache allocation in bytes (default: `16400000000` $\rightarrow$ **`2.21M` tokens**, `1,048,576` max context window)

### 5. Verify Health & Run Benchmarks
```bash
# Check Gateway & Radix Prefix Router status
curl -s http://127.0.0.1:8888/health | jq .

# Run a quick OpenAI chat completion request (with custom reasoning_effort)
curl -s http://127.0.0.1:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "GLM-5.3-Flash",
    "reasoning_effort": "low",
    "messages": [{"role": "user", "content": "Write a fast binary search in Rust."}]
  }' | jq .

# Run the Gate 3 + Gate 4 verification suite
python3 scripts/verify_gate4.py
python3 scripts/benchmark_suite.py

# Run the full 88-scenario tool-eval-bench hardmode suite
tool-eval-bench run --seed 42 --hardmode --base-url http://127.0.0.1:8888
```

### 6. Stopping the Engine
To stop the inference containers while keeping the `85.28 GiB`/rank weights pinned in memory for `< 1.6s` warm restarts:
```bash
./stop.sh
```
To also tear down the persistent weight daemons and free `/dev/shm`:
```bash
./stop.sh --purge-shm
```

---

## Thinking & `reasoning_effort` Controls

`spark-gateway` (`:8888`) natively respects per-request thinking and effort controls across all client formats while preserving multi-turn KV prefix-cache hits:
- **Default (`Max`):** When omitted, defaults to `<|system|>Reasoning Effort: Max` (matching the native GLM-5.3 Jinja template).
- **Per-Request Effort (`low`, `medium`, `high`, `max`):** Pass top-level `"reasoning_effort": "low"`, `"reasoning": {"effort": "low"}`, or `"chat_template_kwargs": {"reasoning_effort": "low"}`.
- **Disable Thinking (`none` / `enable_thinking: false`):** Pass `"reasoning_effort": "none"`, `"enable_thinking": false`, or `tool-eval-bench --no-think`. The template emits `<|assistant|><think></think>` so the model skips `<think>` and achieves **85%–98.6% `DFlash2` (`K=7`) draft acceptance** on code and structured JSON.
- **Guided Decoding (`response_format`):** When `response_format` (`json_schema` / `json_object`) is active on a turn, `spark-gateway` automatically closes `<think></think>` at the end of the prompt while preserving `<|system|>Reasoning Effort: Max` at token 0 so multi-turn tool $\rightarrow$ JSON workflows get a 100% KV prefix-cache hit on prior turns and emit schema-valid JSON directly into `message.content`.

---

## Repository Structure

```text
spark-glm53/
├── LICENSE                         # Apache 2.0 License
├── README.md                       # Quick Start, Architecture & Benchmark Comparison
├── BENCHMARKS.md                   # Detailed Gate 1–4 & tool-eval-bench Engineering Logs
├── stop.sh                         # Clean cluster teardown (preserves /dev/shm weights by default)
├── scripts/
│   ├── tune_sparks.sh              # Dual-Rail ConnectX-7 RoCE v2 & GB10 UMA hygiene + Gate 1 verify
│   ├── start_engine.sh             # Universal 2x DGX Spark TP=2 launcher + Go Control Plane startup
│   ├── fetch_nvidia_ckpt.sh        # O_DIRECT / POSIX_FADV_DONTNEED page-cache-safe checkpoint downloader
│   ├── benchmark_suite.py          # TTFT, 1..8 stream prose/code tok/s, determinism, GSM8K/HumanEval
│   └── verify_gate4.py             # Multi-turn 64K prefix reuse, Hermes 20-tool gauntlet & stream abort tests
├── pkg/
│   ├── shmdaemon/main.go           # Persistent POSIX /dev/shm O_DIRECT weight server (spark-shm-weights)
│   └── gateway/main.go             # Go Control Plane: Radix router, NVMe spill, glm47_repair, OpenAI/Anthropic API
└── worker/
    ├── b12x_dflash_runner.py       # Patched sm_121a B12X + DFlash2 CUDA-graph speculative worker
    ├── cuda_ipc_daemon.py          # Persistent 85.28 GiB/rank CUDA IPC weight slab server
    ├── shm_safetensors.py          # Zero-copy safetensors loader backed by CUDA IPC / /dev/shm pointers
    ├── glm53_chat_template.jinja   # GLM-5.3 chat template with <think></think> & prefix-cache header preservation
    └── kernels/                    # Custom Triton / CuTe DSL sm_121a speculative helper kernels
```

---

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
