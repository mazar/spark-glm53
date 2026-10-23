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

### Multi-Stream Prefill & Decode Throughput (`llama-benchy`, Concurrency `1..8`, Prose Corpus)
Measured head-to-head on the exact same 2x DGX Spark cluster via `llama-benchy --base-url http://127.0.0.1:8888/v1 --pp 512 --tg 256 --exact-tg --no-cache --concurrency 1 2 4 8`:

| Concurrency | `Spark-GLM53` Prefill (`pp512` / `pp2048`) | TensorFold EXL3 Prefill (`pp512`) | `Spark-GLM53` Decode Avg (`tg256` Total / Req) | `Spark-GLM53` Decode Peak (`tg256` Total / Req) | TensorFold EXL3 Decode Avg (`tg256` Total / Req) | TensorFold EXL3 Decode Peak (`tg256` Total / Req) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`c = 1`** | `898.67` / `1,768.63 tok/s` | — *(buffered by 400ms playout)* | `31.21` / `31.21 tok/s` | `49.67` / `49.67 tok/s` | **`63.06` / `63.06 tok/s`** | `63.33` / `63.33 tok/s` |
| **`c = 2`** | `1,327.75` / **`2,206.44 tok/s`** | `1,147.65 ± 268.46 tok/s` | `41.69` / `21.64 tok/s` | `76.00` / `40.00 tok/s` | **`82.10` / `43.48 tok/s`** | `88.00` / `44.00 tok/s` |
| **`c = 4`** | `1,778.45` / `2,033.66 tok/s` | `1,786.32 ± 324.40 tok/s` | `57.29` / `15.20 tok/s` | **`116.00`** / **`31.50 tok/s`** | **`105.14` / `28.06 tok/s`** | `113.67` / `28.42 tok/s` |
| **`c = 8`** | **`2,237.61`** / `2,041.40 tok/s` | `1,686.10 ± 3.84 tok/s` | `91.32` / `12.94 tok/s` | **`165.00`** / **`27.67 tok/s`** | **`130.79` / `18.62 tok/s`** | `152.33` / `19.12 tok/s` |

> **Architectural Trade-off (`NVFP4 + BF16/MXFP8 Dense` vs. `EXL3 4bpw + q4 Dense`):**
> - **TensorFold (`GLM-5.3-Flash-EXL3`)** quantizes not only the routed experts to 4-bit EXL3 trellis (`4bpw`), but also quantizes the dense MLA projections, shared experts, and dense layers to lossy 4-bit (`DENSE=q4`) and uses non-exact chunked KDA (`KDA_CHUNKED=1`). Halving the dense layer bytes reduces 1-stream decode memory traffic (`63.06 tok/s` avg prose at `c=1`), **but** software EXL3 trellis + `q4` dequantization on the GPU SMs burns **`+73%` higher cluster GPU power (`122.05 W` vs `70.53 W`)**, runs **`10°C–14°C` hotter on both GPUs (`79.0°C` vs `64.0–69.0°C`)**, and drops accuracy on `tool-eval-bench --hardmode` (`94/100` vs `95/100`).
> - **`Spark-GLM53` (`NVFP4` + `B12X` + `DFlash2`)** preserves full `BF16`/`MXFP8` precision on dense MLA, shared experts, and routers, uses hardware-native Blackwell `sm_121a` `B12X` Tensor Core instructions, delivers **`+32.7%` higher `pp512` prefill at `c=8` (`2,237.61` vs `1,686.10 tok/s`)**, higher peak `c=8` decode (`165.00` vs `152.33 tok/s`, and **`188.5–208.3 tok/s`** on 8-stream structured code), **`1.86x` larger FP8 KV pool at `PARALLEL=8` (`2.26M` vs `1.21M` tokens)**, and **`42%` lower cluster GPU power (`70.53 W` vs `122.05 W`)**.

### Dual-Node Power (`W` / `Wh`) & Thermal (`°C`) Telemetry (`scripts/run_power_temp_bench.py`)
Sampled at **1 Hz** across both DGX Spark nodes (`spark-5a4e` Head + `spark-8a27` Worker) via `nvidia-smi` and `/sys/class/thermal/thermal_zone0/temp` during the exact same `llama-benchy --pp 512 --tg 256 --exact-tg --no-cache --concurrency 1 2 4 8` benchmark:

| Metric (`spark-5a4e` Head + `spark-8a27` Worker) | `Spark-GLM53` (`NVFP4` + `B12X` + `DFlash2`) | TensorFold (`GLM-5.3-Flash-EXL3-4bpw` + `DENSE=q4`) | `Spark-GLM53` Advantage |
| :--- | :--- | :--- | :--- |
| **Cluster Idle GPU Power (`TP=2`)** | **`33.16 W`** (`16.82 W` + `16.34 W`) | `34.39 W` (`17.57 W` + `16.82 W`) | **-1.23 W** |
| **Head Node (`spark-5a4e`) Active Avg / Peak Power** | **`35.21 – 36.22 W` avg** / **`62.36 – 63.90 W` peak** | `60.57 W` avg / `84.75 W` peak | **-41.9% lower avg power** |
| **Worker Node (`spark-8a27`) Active Avg / Peak Power**| **`35.37 W` avg** / **`62.72 W` peak** | `61.48 W` avg / `83.57 W` peak | **-42.5% lower avg power** |
| **Cluster Total Active Avg GPU Power (`TP=2`)** | **`70.53 – 72.44 W`** | `122.05 W` | **-42.2% lower power (-51.5 W)** |
| **Cluster Total Peak Simultaneous GPU Power** | **`102.67 – 126.62 W`** | `147.92 W` (`168.32 W` node peak sum) | **-30.6% lower peak power** |
| **Head GPU Temperature (Start $\rightarrow$ Avg / Max)** | `45.0°C` $\rightarrow$ **`57.3–60.8°C` avg / `63.0–69.0°C` max** | `53.0°C` $\rightarrow$ `70.5°C` avg / `79.0°C` max | **-10.0°C to -16.0°C cooler GPU** |
| **Worker GPU Temperature (Start $\rightarrow$ Avg / Max)**| `48.7°C` $\rightarrow$ **`58.5°C` avg / `64.0°C` max** | `52.0°C` $\rightarrow$ `69.3°C` avg / `78.0°C` max | **-14.0°C cooler GPU** |
| **Head Grace SoC Temp (Avg / Max)** | **`76.9–81.7°C` avg / `84.9–94.3°C` max** | `82.2°C` avg / `95.7°C` max | **Cooler SoC** |
| **Worker Grace SoC Temp (Avg / Max)** | **`68.9°C` avg / `75.7°C` max** | `78.6°C` avg / `85.1°C` max | **-9.4°C cooler SoC** |
| **Average SM Clock (Head / Worker)** | **`2,535 MHz` / `2,559 MHz`** | `2,498 MHz` / `2,529 MHz` (thermal throttle) | **+30–37 MHz higher sustained boost**|
| **API Control-Plane Latency (`llama-benchy` probe)** | **`2.95 – 3.26 ms`** (Compiled Go Gateway) | `42.42 ms` (Python HTTP server) | **13.5x lower API overhead** |
| **FP8 KV Cache Pool at `PARALLEL=8`** | **`2,263,389` tokens** | `1,214,464` tokens | **1.86x larger multi-stream KV pool**|
| **Restart / Weight Load Time** | **`1.50 – 1.60 s`** (CUDA IPC) / **`0.87 s`** (`shm`) | `120.7 – 126.0 s` (`2m 15.7s` total) | **~80x – 140x faster restart** |

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
Run on the **Head Node** (`./scripts/start_engine.sh` will automatically sync the repository and binaries to the Worker node):
```bash
git clone https://github.com/mazar/spark-glm53.git
cd spark-glm53

# Build the Go Control Plane (spark-gateway) and POSIX /dev/shm Weight Daemon (spark-shm-weights)
mkdir -p bin
go build -o bin/spark-gateway ./pkg/gateway
go build -o bin/spark-shm-weights ./pkg/shmdaemon
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
│   ├── run_power_temp_bench.py     # 1 Hz dual-node power (Watts/Wh) & thermal (°C) telemetry wrapper
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
