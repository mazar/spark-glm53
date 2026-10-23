#!/usr/bin/env bash
# scripts/start_engine.sh — Launch the 2x DGX Spark (TP=2) Spark-GLM53 Speculative Engine
# Always starts Worker (Rank 1) before Head (Rank 0) per GEMINI.md Rule 2.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST_HOME="${HOST_HOME:-$HOME}"
HOST_CACHE_DIR="${HOST_CACHE_DIR:-$HOST_HOME/.cache}"
DOCKER_IMAGE="${DOCKER_IMAGE:-vllm-node-b12x:latest}"
if ! docker image inspect "$DOCKER_IMAGE" >/dev/null 2>&1; then
  if docker image inspect "eugr/spark-vllm-b12x:latest" >/dev/null 2>&1; then
    DOCKER_IMAGE="eugr/spark-vllm-b12x:latest"
  fi
fi

# Auto-discover peer DGX Spark worker if WORKER_HOST is not explicitly set
discover_worker_host() {
  if [[ -n "${WORKER_HOST:-}" ]]; then
    echo "$WORKER_HOST"
    return
  fi
  for cand in spark-8a27 10.100.240.2 10.100.241.2; do
    if [[ "$cand" != "$(hostname)" ]] && ssh -o BatchMode=yes -o ConnectTimeout=2 "$cand" true 2>/dev/null; then
      echo "$cand"
      return
    fi
  done
  echo "10.100.240.2"
}
WORKER_HOST="$(discover_worker_host)"
HEAD_ROCE_IP="${HEAD_ROCE_IP:-10.100.240.1}"
WORKER_ROCE_IP="${WORKER_ROCE_IP:-10.100.240.2}"
MASTER_PORT="${MASTER_PORT:-29500}"
ENGINE_PORT="${ENGINE_PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1048576}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.850}"
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-16400000000}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-4096}"
SPEC_ENABLED="${SPEC_ENABLED:-1}"

# Dynamically resolve HuggingFace snapshot paths on host and inside container (/root/.cache/huggingface)
resolve_snapshot() {
  local repo_dir="$1"
  local base="$HOST_CACHE_DIR/huggingface/hub/$repo_dir/snapshots"
  if [[ -d "$base" ]]; then
    ls -1dt "$base"/* 2>/dev/null | head -n 1 || true
  fi
}

HOST_TARGET_SNAPSHOT="${HOST_TARGET_SNAPSHOT:-$(resolve_snapshot 'models--local-inference-lab--GLM-5.3-Flash-NVFP4-Spark')}"
if [[ -z "$HOST_TARGET_SNAPSHOT" || ! -d "$HOST_TARGET_SNAPSHOT" ]]; then
  HOST_TARGET_SNAPSHOT="${HOST_TARGET_SNAPSHOT:-$(resolve_snapshot 'models--nvidia--GLM-5.3-Flash-FP4-Local')}"
fi
HOST_DFLASH_SNAPSHOT="${HOST_DFLASH_SNAPSHOT:-$(resolve_snapshot 'models--incoai--GLM-5.3-Flash-DFlash2')}"

TARGET_SNAPSHOT="${TARGET_SNAPSHOT:-${HOST_TARGET_SNAPSHOT/#$HOST_CACHE_DIR/\/root\/.cache}}"
DFLASH_SNAPSHOT="${DFLASH_SNAPSHOT:-${HOST_DFLASH_SNAPSHOT/#$HOST_CACHE_DIR/\/root\/.cache}}"

log() { printf '[start_engine] %s\n' "$*"; }

# 0. Auto-build Go binaries (spark-gateway and spark-shm-weights) if missing
if [[ ! -x "$ROOT_DIR/bin/spark-gateway" || ! -x "$ROOT_DIR/bin/spark-shm-weights" ]]; then
  if command -v go >/dev/null 2>&1; then
    log "Building Go binaries (bin/spark-gateway, bin/spark-shm-weights)..."
    mkdir -p "$ROOT_DIR/bin"
    go build -o "$ROOT_DIR/bin/spark-gateway" "$ROOT_DIR/pkg/gateway"
    go build -o "$ROOT_DIR/bin/spark-shm-weights" "$ROOT_DIR/pkg/shmdaemon"
  fi
fi

# 1. Clean any prior engine containers (preserving spark-shm-weights)
"$ROOT_DIR/stop.sh"

# 1b. Reclaim headless display / desktop GPU contexts, disable host swap via nsenter, and flush OS file page cache on both nodes
log "Reclaiming headless GPU contexts, locking GPU clocks (3003 MHz), disabling host swap, and flushing file page cache..."
docker run --rm --privileged --gpus all --pid=host --ipc=host -v /proc:/host_proc -v "$HOST_CACHE_DIR:/host_cache" "$DOCKER_IMAGE" sh -c '
  nvidia-smi -pm 1 >/dev/null 2>&1 || true
  nvidia-smi -lgc 3003,3003 >/dev/null 2>&1 || true
  chown -R 1000:1000 /host_cache/b12x /host_cache/vllm /host_cache/vllm-glm53-flash 2>/dev/null || true
  rm -f /dev/shm/psm_* /dev/shm/sem.mp-* 2>/dev/null || true
  nsenter -t 1 -m -u -i -n -p systemctl stop gdm gdm3 lightdm display-manager apt-daily.timer apt-daily-upgrade.timer unattended-upgrades 2>/dev/null || true
  if [ -f /etc/default/earlyoom ] || nsenter -t 1 -m -- test -f /etc/default/earlyoom 2>/dev/null; then
    nsenter -t 1 -m -- sed -i "s|^EARLYOOM_ARGS=.*|EARLYOOM_ARGS=\"-r 3600 -p -M 98304,65536 -s 100,100 --prefer ^(VLLM\\|pt_main_thread\\|tensorfold) --avoid ^(systemd\\|sshd\\|dockerd\\|containerd\\|dbus-daemon\\|NetworkManager\\|gdm\\|Xorg\\|nvidia\\|agy\\|tmux\\|spark-shm-weigh)\"|" /etc/default/earlyoom 2>/dev/null || true
    nsenter -t 1 -m -u -i -n -p systemctl restart earlyoom 2>/dev/null || true
  fi
  pkill -9 -x chrome 2>/dev/null || true
  pkill -9 -x gnome-shell 2>/dev/null || true
  pkill -9 -x Xorg 2>/dev/null || true
  pkill -9 -f "_offline_compiler_spawn_environment" 2>/dev/null || true
  pkill -9 -f "multiprocessing.spawn import spawn_main" 2>/dev/null || true
  nsenter -t 1 -m -u -i -n -p swapoff -a 2>/dev/null || true
  echo 0 > /host_proc/sys/vm/watermark_boost_factor 2>/dev/null || true
  echo 65536 > /host_proc/sys/vm/min_free_kbytes 2>/dev/null || true
  sync && echo 3 > /host_proc/sys/vm/drop_caches 2>/dev/null || true
' || true
ssh -o BatchMode=yes "$WORKER_HOST" "docker run --rm --privileged --gpus all --pid=host --ipc=host -v /proc:/host_proc -v $HOST_CACHE_DIR:/host_cache $DOCKER_IMAGE sh -c '
  nvidia-smi -pm 1 >/dev/null 2>&1 || true
  nvidia-smi -lgc 3003,3003 >/dev/null 2>&1 || true
  chown -R 1000:1000 /host_cache/b12x /host_cache/vllm /host_cache/vllm-glm53-flash 2>/dev/null || true
  rm -f /dev/shm/psm_* /dev/shm/sem.mp-* 2>/dev/null || true
  nsenter -t 1 -m -u -i -n -p systemctl stop gdm gdm3 lightdm display-manager apt-daily.timer apt-daily-upgrade.timer unattended-upgrades 2>/dev/null || true
  if [ -f /etc/default/earlyoom ] || nsenter -t 1 -m -- test -f /etc/default/earlyoom 2>/dev/null; then
    nsenter -t 1 -m -- sed -i \"s|^EARLYOOM_ARGS=.*|EARLYOOM_ARGS=\\\"-r 3600 -p -M 98304,65536 -s 100,100 --prefer ^(VLLM\\\\|pt_main_thread\\\\|tensorfold) --avoid ^(systemd\\\\|sshd\\\\|dockerd\\\\|containerd\\\\|dbus-daemon\\\\|NetworkManager\\\\|gdm\\\\|Xorg\\\\|nvidia\\\\|agy\\\\|tmux\\\\|spark-shm-weigh)\\\"|\" /etc/default/earlyoom 2>/dev/null || true
    nsenter -t 1 -m -u -i -n -p systemctl restart earlyoom 2>/dev/null || true
  fi
  pkill -9 -x chrome 2>/dev/null || true
  pkill -9 -x gnome-shell 2>/dev/null || true
  pkill -9 -x Xorg 2>/dev/null || true
  pkill -9 -f \"_offline_compiler_spawn_environment\" 2>/dev/null || true
  pkill -9 -f \"multiprocessing.spawn import spawn_main\" 2>/dev/null || true
  nsenter -t 1 -m -u -i -n -p swapoff -a 2>/dev/null || true
  echo 0 > /host_proc/sys/vm/watermark_boost_factor 2>/dev/null || true
  echo 65536 > /host_proc/sys/vm/min_free_kbytes 2>/dev/null || true
  sync && echo 3 > /host_proc/sys/vm/drop_caches 2>/dev/null || true
'" || true

# 2. Sync worker code, chat template, and plugin metadata to Worker node
log "Syncing worker codebase, chat template, and b12x compile cache to $WORKER_HOST..."
cp -f "$ROOT_DIR/worker/glm53_chat_template.jinja" "$HOST_TARGET_SNAPSHOT/chat_template.jinja" || true
ssh -o BatchMode=yes "$WORKER_HOST" "mkdir -p $ROOT_DIR/worker/kernels $ROOT_DIR/spark_glm53-1.0.0.dist-info $ROOT_DIR/scripts $ROOT_DIR/bin $HOST_CACHE_DIR/b12x/compile/preparation"
rsync -az --omit-dir-times --exclude='__pycache__' --exclude='*.pyc' "$ROOT_DIR/worker/" "$WORKER_HOST:$ROOT_DIR/worker/"
rsync -az --omit-dir-times --exclude='__pycache__' --exclude='*.pyc' "$ROOT_DIR/spark_glm53-1.0.0.dist-info/" "$WORKER_HOST:$ROOT_DIR/spark_glm53-1.0.0.dist-info/"
if [[ -d "$ROOT_DIR/bin" ]]; then
  rsync -az --omit-dir-times "$ROOT_DIR/bin/" "$WORKER_HOST:$ROOT_DIR/bin/" || true
fi
scp -q "$ROOT_DIR/usercustomize.py" "$ROOT_DIR/scripts/benchmark_suite.py" "$WORKER_HOST:$ROOT_DIR/" || true
scp -q "$ROOT_DIR/worker/glm53_chat_template.jinja" "$WORKER_HOST:$HOST_TARGET_SNAPSHOT/chat_template.jinja" || true
if [[ -d "$HOST_CACHE_DIR/b12x" ]]; then
  rsync -az --omit-dir-times "$HOST_CACHE_DIR/b12x/" "$WORKER_HOST:$HOST_CACHE_DIR/b12x/" || true
fi

# 3. Verify persistent /dev/shm and CUDA IPC weight daemons on both nodes
SPARK_USE_CUDA_IPC="${SPARK_USE_CUDA_IPC:-1}"
log "Verifying persistent weight daemons on Head and Worker (SPARK_USE_CUDA_IPC=${SPARK_USE_CUDA_IPC})..."
if [[ -x "$ROOT_DIR/bin/spark-shm-weights" ]]; then
  if ! "$ROOT_DIR/bin/spark-shm-weights" --rank 0 --status >/dev/null 2>&1; then
    docker rm -f spark-shm-weights >/dev/null 2>&1 || true
    docker run -d --name spark-shm-weights --restart unless-stopped \
      --ipc=host --privileged --ulimit memlock=-1:-1 \
      -v /tmp:/tmp -v "$HOST_HOME:$HOST_HOME" \
      -v "$HOST_CACHE_DIR/huggingface:/root/.cache/huggingface" \
      --entrypoint "$ROOT_DIR/bin/spark-shm-weights" \
      "$DOCKER_IMAGE" --rank 0 >/dev/null 2>&1 || true
  fi
  ssh -o BatchMode=yes "$WORKER_HOST" "
    if ! $ROOT_DIR/bin/spark-shm-weights --rank 1 --status >/dev/null 2>&1; then
      docker rm -f spark-shm-weights >/dev/null 2>&1 || true
      docker run -d --name spark-shm-weights --restart unless-stopped \
        --ipc=host --privileged --ulimit memlock=-1:-1 \
        -v /tmp:/tmp -v $HOST_HOME:$HOST_HOME \
        -v $HOST_CACHE_DIR/huggingface:/root/.cache/huggingface \
        --entrypoint $ROOT_DIR/bin/spark-shm-weights \
        $DOCKER_IMAGE --rank 1 >/dev/null 2>&1 || true
    fi
  " || true
fi
python3 -c '
import os, socket
s = "/tmp/spark_shm_weights_rank_0.sock"
use_ipc = os.environ.get("SPARK_USE_CUDA_IPC", "1") == "1"
if os.path.exists(s):
    cmds = ["UNLOCK /dev/shm/glm53_dflash2.safetensors", "UNLOCK /dev/shm/glm53_rank_0.safetensors"]
    if not use_ipc and not os.path.exists("/dev/shm/glm53_rank_0.safetensors.cudaipc.json"):
        cmds.append("LOCK /dev/shm/glm53_rank_0.safetensors")
    for cmd in cmds:
        if cmd.startswith("LOCK") and not os.path.exists("/dev/shm/glm53_rank_0.safetensors"):
            continue
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(60.0)
            sock.connect(s)
            sock.sendall((cmd + "\n").encode())
            print("[start_engine] Head", cmd, "->", sock.recv(4096).decode().strip())
' || true
ssh -o BatchMode=yes "$WORKER_HOST" "SPARK_USE_CUDA_IPC=${SPARK_USE_CUDA_IPC} python3 -c '
import os, socket
s = \"/tmp/spark_shm_weights_rank_1.sock\"
use_ipc = os.environ.get(\"SPARK_USE_CUDA_IPC\", \"1\") == \"1\"
if os.path.exists(s):
    cmds = [\"UNLOCK /dev/shm/glm53_dflash2.safetensors\", \"UNLOCK /dev/shm/glm53_rank_1.safetensors\"]
    if not use_ipc and not os.path.exists(\"/dev/shm/glm53_rank_1.safetensors.cudaipc.json\"):
        cmds.append(\"LOCK /dev/shm/glm53_rank_1.safetensors\")
    for cmd in cmds:
        if cmd.startswith(\"LOCK\") and not os.path.exists(\"/dev/shm/glm53_rank_1.safetensors\"):
            continue
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(60.0)
            sock.connect(s)
            sock.sendall((cmd + \"\\n\").encode())
            print(\"[start_engine] Worker\", cmd, \"->\", sock.recv(4096).decode().strip())
'" || true
"$ROOT_DIR/bin/spark-shm-weights" --rank 0 --status || true
ssh -o BatchMode=yes "$WORKER_HOST" "$ROOT_DIR/bin/spark-shm-weights --rank 1 --status" || true
if [[ "$SPARK_USE_CUDA_IPC" == "1" ]]; then
  docker exec spark-shm-weights rm -f /dev/shm/glm53_dflash2.safetensors 2>/dev/null || true
  ssh -o BatchMode=yes "$WORKER_HOST" "docker exec spark-shm-weights rm -f /dev/shm/glm53_dflash2.safetensors 2>/dev/null || true" || true
fi

if [[ "$SPARK_USE_CUDA_IPC" == "1" ]]; then
  if ! python3 "$ROOT_DIR/worker/cuda_ipc_daemon.py" --rank 0 --status >/dev/null 2>&1; then
    docker rm -f spark-cuda-ipc-weights >/dev/null 2>&1 || true
    if python3 -c 'import os; p="/dev/shm/glm53_rank_0.safetensors"; st=os.stat(p); raise SystemExit(0 if st.st_blocks*512 >= st.st_size*0.9 else 1)' 2>/dev/null; then
      log "Starting persistent CUDA IPC weight server (spark-cuda-ipc-weights) on Head (Rank 0)..."
      docker run -d --name spark-cuda-ipc-weights \
        --gpus all --ipc=host --network=host --privileged \
        -v /tmp:/tmp -v "$ROOT_DIR:$ROOT_DIR" \
        -e PYTHONPATH="$ROOT_DIR" \
        --entrypoint python3 "$DOCKER_IMAGE" \
        "$ROOT_DIR/worker/cuda_ipc_daemon.py" --rank 0 >/dev/null
    else
      rm -f /dev/shm/glm53_rank_0.safetensors /dev/shm/glm53_rank_0.safetensors.cudaipc.json /dev/shm/glm53_rank_0.safetensors.w13_normalized 2>/dev/null || true
    fi
  fi
  if ! ssh -o BatchMode=yes "$WORKER_HOST" "python3 $ROOT_DIR/worker/cuda_ipc_daemon.py --rank 1 --status" >/dev/null 2>&1; then
    ssh -o BatchMode=yes "$WORKER_HOST" "docker rm -f spark-cuda-ipc-weights >/dev/null 2>&1 || true"
    if ssh -o BatchMode=yes "$WORKER_HOST" "python3 -c 'import os; p=\"/dev/shm/glm53_rank_1.safetensors\"; st=os.stat(p); raise SystemExit(0 if st.st_blocks*512 >= st.st_size*0.9 else 1)'" 2>/dev/null; then
      log "Starting persistent CUDA IPC weight server (spark-cuda-ipc-weights) on Worker (Rank 1)..."
      ssh -o BatchMode=yes "$WORKER_HOST" "docker run -d --name spark-cuda-ipc-weights \
          --gpus all --ipc=host --network=host --privileged \
          -v /tmp:/tmp -v $ROOT_DIR:$ROOT_DIR \
          -e PYTHONPATH=$ROOT_DIR \
          --entrypoint python3 $DOCKER_IMAGE \
          $ROOT_DIR/worker/cuda_ipc_daemon.py --rank 1 >/dev/null"
    else
      ssh -o BatchMode=yes "$WORKER_HOST" "rm -f /dev/shm/glm53_rank_1.safetensors /dev/shm/glm53_rank_1.safetensors.cudaipc.json /dev/shm/glm53_rank_1.safetensors.w13_normalized 2>/dev/null || true"
    fi
  fi
  if docker ps --format '{{.Names}}' | grep -qx "spark-cuda-ipc-weights"; then
    log "Waiting for Head (Rank 0) spark-cuda-ipc-weights readiness..."
    for _ in $(seq 1 90); do
      if python3 "$ROOT_DIR/worker/cuda_ipc_daemon.py" --rank 0 --status 2>/dev/null; then
        break
      fi
      sleep 1
    done
  fi
  if ssh -o BatchMode=yes "$WORKER_HOST" "docker ps --format '{{.Names}}' | grep -qx spark-cuda-ipc-weights"; then
    log "Waiting for Worker (Rank 1) spark-cuda-ipc-weights readiness..."
    for _ in $(seq 1 90); do
      if ssh -o BatchMode=yes "$WORKER_HOST" "python3 $ROOT_DIR/worker/cuda_ipc_daemon.py --rank 1 --status" 2>/dev/null; then
        break
      fi
      sleep 1
    done
  fi
fi

COMMON_DOCKER_ENV=(
  -e "PYTHONDONTWRITEBYTECODE=1"
  -e "PYTHONPATH=$ROOT_DIR"
  -e "SPARK_GLM53_ACTIVE=1"
  -e "HF_HUB_OFFLINE=1"
  -e "TRANSFORMERS_OFFLINE=1"
  -e "GLOO_SOCKET_IFNAME=enp1s0f1np1"
  -e "NCCL_SOCKET_IFNAME=enp1s0f1np1"
  -e "NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1"
  -e "NCCL_IB_GID_INDEX=3"
  -e "NCCL_CROSS_NIC=1"
  -e "NCCL_BUFFSIZE=1048576"
  -e "NCCL_MAX_CTAS=8"
  -e "NCCL_IGNORE_CPU_AFFINITY=1"
  -e "CUTE_DSL_ARCH=sm_121a"
  -e "SAFETENSORS_FAST_GPU=1"
  -e "VLLM_ENABLE_ROCE_ALLREDUCE=1"
  -e "VLLM_ROCE_ALLREDUCE_MAX_SIZE=${VLLM_ROCE_ALLREDUCE_MAX_SIZE:-16MB}"
  -e "VLLM_ENABLE_PCIE_ALLREDUCE=0"
  -e "VLLM_SSM_CONV_STATE_LAYOUT=DS"
  -e "VLLM_USE_AOT_COMPILE=1"
  -e "VLLM_USE_MEGA_AOT_ARTIFACT=1"
  -e "VLLM_MXFP8_LM_HEAD=${VLLM_MXFP8_LM_HEAD:-0}"
  -e "B12X_POLICY_MODE=auto"
  -e "B12X_MICRO_DYNAMIC_CUTOVER_PAIRS=${B12X_MICRO_DYNAMIC_CUTOVER_PAIRS:-0}"
  -e "SPARK_B12X_DIRECT_ROUTING_MAX_ROWS=${SPARK_B12X_DIRECT_ROUTING_MAX_ROWS:-}"
  -e "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0"
  -e "VLLM_GLM53_DFLASH_ATTN=1"
  -e "VLLM_GLM53_DFLASH_ATTN_MAX_REQS=16"
  -e "VLLM_GLM53_DFLASH_ATTN_WS_REQS=16"
  -e "VLLM_GLM53_L2_PREFETCH=${VLLM_GLM53_L2_PREFETCH:-0}"
  -e "GLM53_INDEXER_WORKSPACE=rightsize"
  -e "SPARK_SHM_ENABLE=1"
  -e "SPARK_SHM_MATERIALIZE_DENSE=${SPARK_SHM_MATERIALIZE_DENSE:-0}"
  -e "SPARK_ENABLE_COPY_DRAFT=${SPARK_ENABLE_COPY_DRAFT:-0}"
  -e "SPARK_DFLASH_STEPS=${SPARK_DFLASH_STEPS:-7}"
  -e "SPARK_COPY_DRAFT_SPAN=${SPARK_COPY_DRAFT_SPAN:-7}"
  -e "SPARK_PROFILE_STEPS=${SPARK_PROFILE_STEPS:-0}"
  -e "SPARK_LAYER_PROF=${SPARK_LAYER_PROF:-0}"
  -e "PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.145,max_split_size_mb:64"
)

SPEC_ARGS=()
if [[ "$SPEC_ENABLED" == "1" ]]; then
  NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-7}"
  SPEC_JSON="{\"method\":\"dflash\",\"model\":\"${DFLASH_SNAPSHOT}\",\"num_speculative_tokens\":${NUM_SPEC_TOKENS},\"attention_backend\":\"FLASH_ATTN\",\"kv_cache_dtype\":\"auto\"}"
  SPEC_ARGS=(--speculative-config "$SPEC_JSON")
fi

SPARK_KDA_PREFILL_BACKEND="${SPARK_KDA_PREFILL_BACKEND:-b12x}"
EXTRA_VLLM_ARGS=()
if [[ -n "${SPARK_KDA_PREFILL_BACKEND:-}" ]]; then
  EXTRA_VLLM_ARGS+=(--additional-config "{\"kda_prefill_backend\":\"${SPARK_KDA_PREFILL_BACKEND}\"}")
fi

COMPILATION_JSON='{"cudagraph_mode":"FULL_AND_PIECEWISE","max_cudagraph_capture_size":64,"cudagraph_capture_sizes":[1,2,4,8,16,32,64]}'

COMMON_VLLM_ARGS=(
  "$TARGET_SNAPSHOT"
  --served-model-name GLM-5.3-Flash
  --chat-template "$ROOT_DIR/worker/glm53_chat_template.jinja"
  --trust-remote-code
  --language-model-only
  --tensor-parallel-size 2
  --nnodes 2
  --master-addr "$HEAD_ROCE_IP"
  --master-port "$MASTER_PORT"
  --load-format b12x
  --quantization modelopt_mixed
  --attention-backend B12X
  --kernel-config '{"enable_flashinfer_autotune":false,"moe_backend":"b12x","linear_backend":"b12x"}'
  --moe-backend b12x
  --linear-backend b12x
  --kv-cache-dtype fp8
  --block-size 256
  --mamba-cache-mode align
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs 16
  --max-num-batched-tokens "$MAX_BATCHED_TOKENS"
  --gpu-memory-utilization "$GPU_MEM_UTIL"
  --kv-cache-memory-bytes "$KV_CACHE_MEMORY_BYTES"
  --enable-prefix-caching
  --no-async-scheduling
  --compilation-config "$COMPILATION_JSON"
  "${SPEC_ARGS[@]}"
  "${EXTRA_VLLM_ARGS[@]}"
)

# 4. Start Worker Rank 1 FIRST (GEMINI.md Rule 2)
log "Starting Worker Rank 1 container (spark-glm53-worker) on $WORKER_HOST..."
ssh -o BatchMode=yes "$WORKER_HOST" bash -s <<EOF
set -euo pipefail
mkdir -p $HOST_CACHE_DIR/b12x $HOST_CACHE_DIR/flashinfer $HOST_CACHE_DIR/vllm $HOST_CACHE_DIR/vllm-glm53-flash/triton
docker run -d --name spark-glm53-worker \
  --gpus all --ipc=host --network=host --privileged \
  --ulimit memlock=-1:-1 \
  -v $HOST_CACHE_DIR/huggingface:/root/.cache/huggingface \
  -v $HOST_CACHE_DIR/b12x:/root/.cache/b12x \
  -v $HOST_CACHE_DIR/flashinfer:/root/.cache/flashinfer \
  -v $HOST_CACHE_DIR/vllm:/root/.cache/vllm \
  -v $HOST_CACHE_DIR/vllm-glm53-flash/triton:/root/.triton/cache \
  -v /tmp:/tmp \
  -v $ROOT_DIR:$ROOT_DIR \
  -v $ROOT_DIR/usercustomize.py:/usr/lib/python3.12/sitecustomize.py:ro \
  -v $ROOT_DIR/usercustomize.py:/usr/local/lib/python3.12/dist-packages/usercustomize.py:ro \
  -e VLLM_HOST_IP=$WORKER_ROCE_IP \
  ${COMMON_DOCKER_ENV[*]} \
  --entrypoint python3 \
  $DOCKER_IMAGE \
  $ROOT_DIR/worker/b12x_dflash_runner.py serve \
  $(printf '%q ' "${COMMON_VLLM_ARGS[@]}") \
  --node-rank 1 --headless >/dev/null
EOF

sleep 2

# 5. Start Head Rank 0 container (spark-glm53-head) on port 8000
log "Starting Head Rank 0 container (spark-glm53-head) on $(hostname) port $ENGINE_PORT..."
mkdir -p "$HOST_CACHE_DIR/b12x" "$HOST_CACHE_DIR/flashinfer" "$HOST_CACHE_DIR/vllm" "$HOST_CACHE_DIR/vllm-glm53-flash/triton"
CLAUDE_VOL_ARGS=()
if [[ -d "$HOST_HOME/.claude" ]]; then
  CLAUDE_VOL_ARGS=(-v "$HOST_HOME/.claude:/root/.claude:ro" -v "$HOST_HOME/.claude:$HOST_HOME/.claude:ro")
fi
docker run -d --name spark-glm53-head \
  --gpus all --ipc=host --network=host --privileged \
  --ulimit memlock=-1:-1 \
  -v "$HOST_CACHE_DIR/huggingface:/root/.cache/huggingface" \
  -v "$HOST_CACHE_DIR/b12x:/root/.cache/b12x" \
  -v "$HOST_CACHE_DIR/flashinfer:/root/.cache/flashinfer" \
  -v "$HOST_CACHE_DIR/vllm:/root/.cache/vllm" \
  -v "$HOST_CACHE_DIR/vllm-glm53-flash/triton:/root/.triton/cache" \
  -v /tmp:/tmp \
  -v "$ROOT_DIR:$ROOT_DIR" \
  "${CLAUDE_VOL_ARGS[@]}" \
  -v "$ROOT_DIR/usercustomize.py:/usr/lib/python3.12/sitecustomize.py:ro" \
  -v "$ROOT_DIR/usercustomize.py:/usr/local/lib/python3.12/dist-packages/usercustomize.py:ro" \
  -e "VLLM_HOST_IP=$HEAD_ROCE_IP" \
  "${COMMON_DOCKER_ENV[@]}" \
  --entrypoint python3 \
  "$DOCKER_IMAGE" \
  "$ROOT_DIR/worker/b12x_dflash_runner.py" serve \
  "${COMMON_VLLM_ARGS[@]}" \
  --node-rank 0 --host 0.0.0.0 --port "$ENGINE_PORT" >/dev/null

log "Waiting for Spark-GLM53 engine on http://127.0.0.1:${ENGINE_PORT}/health ..."
for i in $(seq 1 360); do
  if curl -sf "http://127.0.0.1:${ENGINE_PORT}/health" >/dev/null 2>&1; then
    log "Spark-GLM53 engine is READY on port ${ENGINE_PORT}! (elapsed: $((i * 2))s)"
    docker logs spark-glm53-head > /tmp/spark_head.log 2>&1 || true
    # Reclaim the ~2.1 GiB of Linux file page cache accumulated during container boot
    docker exec spark-glm53-head sh -c 'sync && echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true' || true
    ssh -o BatchMode=yes "$WORKER_HOST" "docker exec spark-glm53-worker sh -c 'sync && echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true'" || true
    if [[ -x "$ROOT_DIR/bin/spark-gateway" ]]; then
      GATEWAY_PORT="${GATEWAY_PORT:-8888}"
      log "Starting Go Control Plane (spark-gateway) on port ${GATEWAY_PORT} -> http://127.0.0.1:${ENGINE_PORT} ..."
      docker exec spark-glm53-head sh -c 'for p in /proc/[0-9]*/comm; do [ "$(cat $p 2>/dev/null)" = "spark-gateway" ] && kill -9 $(echo $p | cut -d/ -f3); done' 2>/dev/null || true
      pkill -9 -x spark-gateway 2>/dev/null || true
      docker exec -d spark-glm53-head "$ROOT_DIR/bin/spark-gateway" \
        --listen ":${GATEWAY_PORT}" \
        --upstream "http://127.0.0.1:${ENGINE_PORT}" \
        --spill-dir "/tmp/spark_kv_spill"
      sleep 1
      if curl -sf "http://127.0.0.1:${GATEWAY_PORT}/health" >/dev/null 2>&1; then
        log "Go Control Plane (spark-gateway) is READY on port ${GATEWAY_PORT}!"
      fi
    fi
    exit 0
  fi
  if ! docker ps --format '{{.Names}}' | grep -qx "spark-glm53-head" || docker logs --tail 50 spark-glm53-head 2>&1 | grep -q "EngineCore failed to start"; then
    log "ERROR: spark-glm53-head exited prematurely. Logs:"
    docker logs --tail 200 spark-glm53-head || true
    log "Worker logs:"
    ssh -o BatchMode=yes "$WORKER_HOST" "docker logs --tail 120 spark-glm53-worker" || true
    "$ROOT_DIR/stop.sh" || true
    exit 1
  fi
  sleep 2
done

log "ERROR: Timed out waiting for engine health check."
docker logs --tail 200 spark-glm53-head || true
"$ROOT_DIR/stop.sh" || true
exit 1
