# `Spark-GLM53` — Cluster Environment & Benchmark Log

## Step 0: Hardware & Cluster Environment Discovery (Baseline State)
**Timestamp:** 2026-10-18T12:10:00-07:00

### 1. Cluster Nodes & Topology
| Node Role | Hostname | LAN IP (`enP7s7`) | RoCE Rail 0 (`enp1s0f1np1` / `rocep1s0f1`) | RoCE Rail 1 (`enP2p1s0f1np1` / `roceP2p1s0f1`) | Uncabled Port 0 (`enp1s0f0np0` / `enP2p1s0f0np0`) | GPU & Driver | Total RAM / `/dev/shm` / NVMe |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Head (Rank 0)** | `spark-5a4e` | `192.168.86.35/24` | `10.100.240.1/24` (ACTIVE, GID 3 RoCE v2) | `10.100.241.1/24` (ACTIVE, GID 3 RoCE v2) | `rocep1s0f0`, `roceP2p1s0f0` (`DOWN` / `DISABLED`) | NVIDIA GB10 (`sm_121a`), Driver `580.178.04`, CUDA `13.0` | `121 GiB` UMA / `61 GiB` `/dev/shm` / `1.7 TiB` free NVMe |
| **Worker (Rank 1)** | `spark-8a27` | `192.168.86.36/24` | `10.100.240.2/24` (ACTIVE, GID 3 RoCE v2) | `10.100.241.2/24` (ACTIVE, GID 3 RoCE v2) | `rocep1s0f0`, `roceP2p1s0f0` (`DOWN` / `DISABLED`) | NVIDIA GB10 (`sm_121a`), Driver `580.178.04`, CUDA `13.0` | `121 GiB` UMA / `61 GiB` `/dev/shm` / `2.8 TiB` free NVMe |

- **SSH Connectivity:** Passwordless SSH verified via `~/.ssh/id_ed25519_nvsync_cluster_assistant` to `spark-8a27` (`10.100.240.2`, `10.100.241.2`, `192.168.86.36`).
- **ConnectX-7 Twin PCIe Architecture:** Physical QSFP Port 1 is cabled between `spark-5a4e` and `spark-8a27`, exposing dual PCIe Gen5 x4 twin netdevs/HCAs (`rocep1s0f1,roceP2p1s0f1` on `10.100.240.0/24` and `10.100.241.0/24`). Port 0 (`rocep1s0f0,roceP2p1s0f0`) is currently uncabled (`DOWN`).
- **Pre-Tune Active Workload Found:** Container `glm53-exl3-head` (Head) and `glm53-exl3-worker` (Worker) from image `ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-instanttensor` were running (`VLLM::Worker_TP0` and `VLLM::Worker_TP1`), consuming ~101.6 GiB UMA per node and leaving only ~11 GiB available on Head and ~8.4 GiB on Worker prior to teardown.
- **Desktop / Display Carve-Out:** `Xorg` and `gnome-shell` processes active on both nodes (~24 MiB on Head, ~225 MiB on Worker) alongside headless display reservation.

### 2. Local Checkpoints Verified (`~/.cache/huggingface/hub` on Both Nodes)
| Model / Checkpoint ID | Local Snapshot Size | Quantization / Purpose | Present on Head (`spark-5a4e`) | Present on Worker (`spark-8a27`) |
| :--- | :--- | :--- | :--- | :--- |
| `local-inference-lab/GLM-5.3-Flash-NVFP4-Spark` (community ModelOpt PTQ, `calib_method: max` — **not** the NVIDIA QAD checkpoint; see PLAN.md D2) | `175 GiB` | `modelopt_mixed` (NVFP4 routed experts group_size=16, **MXFP8** attention (402) + shared experts (129) — corrected 2026-10-20) | Yes | Yes |
| `incoai/GLM-5.3-Flash-DFlash2` | `4.4 GiB` | DFlash2 speculative drafter (`K=7`) | Yes | Yes |
| `Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold` | `164 GiB` | Community EXL3 4bpw baseline | Yes | Yes |
| `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` | `171 GiB` | Community EXL3 vLLM baseline | Yes | Yes |
| `local/GLM-5.3-Flash-DFlash2-EXL3-6bpw` | `985 MiB` | EXL3 quantized DFlash2 drafter | Yes | Yes |

### 3. Docker Images & Toolchains Verified
- **Docker Images (Both Nodes):**
  - `vllm-node-b12x:latest` / `eugr/spark-vllm-b12x:latest` (`42f762c0dae9`, 36.3 GB, CUDA 13 + `sm_121a` FlashInfer + `b12x` integrated)
  - `tensorfold-glm53:v0.6.0` (`14b5072e8e99`, 36.9 GB)
  - `ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-instanttensor` (`e09e68447c06` / `8b17e830b563`, 31 GB)
- **Toolchains:**
  - `uv`: `/home/massoud/.local/bin/uv` (`0.11.8 aarch64-unknown-linux-gnu`)
  - `go`: `/home/massoud/sdk/go1.27.1/bin/go` (`go1.27.1 linux/arm64`) and `/usr/bin/go` (`go1.22.2 linux/arm64`)
  - `ib_write_bw`: `/usr/bin/ib_write_bw` (Perftest RDMA benchmark suite)

---

## Phase 1: Dual-Rail RoCE v2 & GB10 UMA Hygiene (`scripts/tune_sparks.sh`)
**Timestamp:** 2026-10-18T12:26:30-07:00
**Command:** `./scripts/tune_sparks.sh --verify`

### 1. RoCE v2 Multi-Rail Configuration & Line-Rate RDMA Bandwidth (`ib_write_bw`)
| Rail / Metric | HCA & Netdev | Subnet & MTU | GID Index & Type | Measured Throughput | Target | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Solo PCIe Gen5 x4 Rail 0** | `rocep1s0f1` (`enp1s0f1np1`) | `10.100.240.1/24 <-> .2/24` (MTU 4200) | `GID[3]` (`RoCE v2`) | **111.85 Gbps** | ~112 Gbps | PASS |
| **Solo PCIe Gen5 x4 Rail 1** | `roceP2p1s0f1` (`enP2p1s0f1np1`) | `10.100.241.1/24 <-> .2/24` (MTU 4200) | `GID[3]` (`RoCE v2`) | **111.85 Gbps** | ~112 Gbps | PASS |
| **Dual x4 PCIe Aggregate Capacity** | `rocep1s0f1,roceP2p1s0f1` | Dual `/24` isolated subnets | `GID[3]` (`RoCE v2`) | **223.70 Gbps** | **>= 215.0 Gbps** | **PASS** |
| **Concurrent Unidirectional Wire** | `rocep1s0f1 + roceP2p1s0f1` | 200GbE QSFP (2X NDR) | `GID[3]` (`RoCE v2`) | **196.07 Gbps** (`98.03 + 98.04`) | Line-rate 200G | **PASS** |
| **Concurrent Bi-Directional (`-b`)** | `rocep1s0f1 + roceP2p1s0f1` | Full-duplex AllGather/AllReduce | `GID[3]` (`RoCE v2`) | **392.08 Gbps** (`196.04 + 196.04`) | **>= 215.0 Gbps** | **PASS** |

### 2. GB10 Unified Memory Architecture (UMA) Hygiene & Reclaim
| Node | `MemTotal` | Pre-Tune `MemAvailable` | Post-Tune `MemFree` | Post-Tune `MemAvailable` | Post-Tune `PageCache` | Target (`> 112 GiB`) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Head (`spark-5a4e`)** | `121.69 GiB` | `11.00 GiB` | **117.62 GiB** | **116.93 GiB** | **0.40 GiB** | **PASS** |
| **Worker (`spark-8a27`)** | `121.69 GiB` | `8.40 GiB` | **117.42 GiB** | **116.73 GiB** | **0.44 GiB** | **PASS** |

- **Gate 1 Status:** **PASSED** (`scripts/tune_sparks.sh --verify` exited `0`).

---

## Phase 2: Persistent `/dev/shm` Zero-Copy Weight Daemon (`pkg/shmdaemon` + `worker/shm_safetensors.py`)
**Timestamp:** 2026-10-18T14:39:15-07:00
**Command:** `./scripts/run_gate2_test.sh warm1 warm2`

### 1. Architecture & Memory Pinning (`spark-shm-weights`)
- **`/dev/shm` Mount Configuration:** Remounted on both nodes with `size=108G,huge=always` (`transparent_hugepage` backing on `tmpfs`).
- **Daemon (`pkg/shmdaemon/main.go` -> `bin/spark-shm-weights`):**
  - Runs persistently on Head (`spark-5a4e`, rank 0) and Worker (`spark-8a27`, rank 1).
  - Preloads `incoai/GLM-5.3-Flash-DFlash2` (`2.18 GiB`, 81 tensors) via `O_DIRECT` + `POSIX_FADV_DONTNEED` into `/dev/shm/glm53_dflash2.safetensors` in `332–377 ms` and pins with `mlock()` (`0.9 ms`).
  - After one-time rank-sharded `B12X` post-processed export (`85.82 GiB`, 3,243 tensors including pre-packed `MXFP8LinearWeight` and `B12xSwigluPackedWeights` / `W_UV` / `W_UK_T`), pins `/dev/shm/glm53_rank_{0,1}.safetensors` in physical RAM via `mlock()` (`33.5–34.3 ms`).
- **Zero-Copy Worker Loader (`worker/shm_safetensors.py`):**
  - Overlaps background `cudaHostRegister(..., cudaHostRegisterIoMemory | cudaHostRegisterMapped)` (`~2.95s` @ `29.1 GiB/s` for `85.82 GiB` + `0.076s` for `2.18 GiB` DFlash2) with `EngineArgs.create_engine_config()` and NCCL process-group initialization.
  - Inside `get_model()`, creates the `Glm5NextForConditionalGeneration` hierarchy on `meta` device (`~0.68s`) and binds zero-copy `cuda:0` unified-memory views (`get_accelerator_view_from_cpu_tensor`) across all parameters, buffers, and parameter aliases in `~0.16s` with **zero `cudaMalloc` weight duplication** and **zero Linux file page-cache growth**.

### 2. Gate 2 Verification Results (Cold Export vs. Two Sequential Kill/Restart Passes, TP=2)
| Pass / Metric | Node / Rank | Target + DFlash2 Load Time | `meta_init` + `bind+pwal` | File Page Cache Before -> After (`Delta`) | `MemAvailable` After Load | Target (`< 3.0s`, `0` Cache Growth) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Stock Baseline (`DefaultModelLoader`)** | Rank 0 / Rank 1 | ~180.0s (buffered I/O) | — | +83.0 GiB (`NV_ERR_NO_MEMORY`) | Crashes | Baseline |
| **Cold `O_DIRECT` + `/dev/shm` Export** | Rank 0 & Rank 1 | `27.60s` read + `26.86s` export | — | `1.53 -> 1.48 GiB` (`-0.05 GiB`) | `21.79 GiB` (88 GiB shm pinned) | Cold Seed |
| **Warm Restart #1 (`warm_restart_1`)** | **Head (`rank=0`)** | **0.8726 sec** (`0.848s` + `0.0001s`) | `0.684s` + `0.164s` | `1.479 -> 1.484 GiB` (**`+0.0044 GiB`**) | `21.79 GiB` | **PASS (< 3.0s)** |
| **Warm Restart #1 (`warm_restart_1`)** | **Worker (`rank=1`)** | **0.8726 sec** (`0.867s` + `0.0001s`) | `0.701s` + `0.166s` | `1.531 -> 1.534 GiB` (**`+0.0033 GiB`**) | `21.28 GiB` | **PASS (< 3.0s)** |
| **Warm Restart #2 (`warm_restart_2`)** | **Head (`rank=0`)** | **0.8717 sec** (`0.846s` + `0.0000s`) | `0.687s` + `0.159s` | `1.484 -> 1.489 GiB` (**`+0.0045 GiB`**) | `21.18 GiB` | **PASS (< 3.0s)** |
| **Warm Restart #2 (`warm_restart_2`)** | **Worker (`rank=1`)** | **0.8717 sec** (`0.865s` + `0.0001s`) | `0.699s` + `0.166s` | `1.531 -> 1.535 GiB` (**`+0.0036 GiB`**) | `21.57 GiB` | **PASS (< 3.0s)** |

- **Gate 2 Status:** **PASSED** (`0.87s` warm weight load time across consecutive worker kill/restarts vs. `< 3.0s` target; `+0.004 GiB` file page-cache delta).


---

## Phase 3: Gate 3 Status — Live Engine Measurements (honest audit)
**Timestamp:** 2026-10-20T20:30-07:00 (engine `spark-glm53-head` up since 02:44 UTC, `MAX_BATCHED_TOKENS=2048`, DFlash2 K=7 + copy-drafter)

| Metric | Measured (live) | Target | Status |
| :--- | :--- | :--- | :--- |
| Prefill 2K / 4K / 8K | 1,476 / 1,825 / 1,823 tok/s | >= 3,600 | **FAIL** (~1.1 s per 2,048-token chunk) |
| 1-stream Prose (512 tok) | 19.9 / 22.5 tok/s (DFlash2 accept len ~2.1–2.3) | >= 62 | **FAIL** |
| 1-stream Code (512 tok) | 45.8 / 43.9 tok/s (accept len ~3.2–5.5) | >= 120 | **FAIL** |
| 8-stream Code | 202.7 tok/s (task-12852) | >= 185 | **Caveated** — see below |
| KV pool (FP8) | 2,263,389 tokens | >= 2.2M | PASS |
| Warm restart | 1.52 s | < 3.0 s | PASS |
| GSM8K / HumanEval smoke | pass / pass | no degradation | PASS (single-item smoke only) |

### Findings
- **Greedy decode is not run-to-run deterministic.** Two back-to-back `temperature=0` requests with an identical prose prompt diverge within the first ~20 tokens. So the "copy the previous run's trajectory from `global_history_gpu`" approach cannot work reliably, and it also replays cached answers rather than speeding up real decoding.
- **The 8-stream number is inflated by the benchmark design.** All 8 streams send the *same* prompt. `SPARK_HOMOG_MOE_DEDUP` computes the MLP once and repeats the result, and the hidden states get canonicalized across streams. That isn't representative of 8 independent users.
- **Per-step decode cost is the real bottleneck:** roughly 105–110 ms per verify step at 1-stream (≈ 512 tokens / ~230 steps / 25 s). With DFlash2 prose acceptance around 2.1, getting to 62 tok/s needs a step time of about 34 ms.
- **Prefill:** each 2,048-token chunk takes about 1.1 s. The profiler is off in this run (`SPARK_PROFILE_STEPS=0`), so we don't yet have a per-layer breakdown for the mhc/kda/mla/moe parts.
