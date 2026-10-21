#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKER_HOST="spark-8a27"
MASTER_ADDR="10.100.240.1"
MASTER_PORT="29550"

# 1. Sync files to Worker node
ssh -o BatchMode=yes "$WORKER_HOST" "mkdir -p $ROOT_DIR/worker $ROOT_DIR/scripts $ROOT_DIR/bin"
scp -q "$ROOT_DIR/worker/shm_safetensors.py" "$WORKER_HOST:$ROOT_DIR/worker/shm_safetensors.py"
scp -q "$ROOT_DIR/scripts/verify_gate2.py" "$WORKER_HOST:$ROOT_DIR/scripts/verify_gate2.py"
scp -q "$ROOT_DIR/scripts/spark_env.sh" "$WORKER_HOST:$ROOT_DIR/scripts/spark_env.sh"

# 2. Ensure spark-shm-weights is running persistently on both nodes
if ! "$ROOT_DIR/bin/spark-shm-weights" --rank 0 --status >/dev/null 2>&1; then
  docker rm -f spark-shm-weights >/dev/null 2>&1 || true
  pkill -x spark-shm-weights >/dev/null 2>&1 || true
  docker run -d --name spark-shm-weights --restart unless-stopped \
    --ipc=host --privileged --ulimit memlock=-1:-1 \
    -v /tmp:/tmp -v /home/massoud:/home/massoud \
    -v /home/massoud/.cache/huggingface:/root/.cache/huggingface \
    --entrypoint "$ROOT_DIR/bin/spark-shm-weights" \
    vllm-node-b12x:latest --rank 0 >/dev/null
  sleep 2
fi
ssh -o BatchMode=yes "$WORKER_HOST" "
  if ! $ROOT_DIR/bin/spark-shm-weights --rank 1 --status >/dev/null 2>&1; then
    docker rm -f spark-shm-weights >/dev/null 2>&1 || true
    pkill -x spark-shm-weights >/dev/null 2>&1 || true
    docker run -d --name spark-shm-weights --restart unless-stopped \
      --ipc=host --privileged --ulimit memlock=-1:-1 \
      -v /tmp:/tmp -v /home/massoud:/home/massoud \
      -v /home/massoud/.cache/huggingface:/root/.cache/huggingface \
      --entrypoint $ROOT_DIR/bin/spark-shm-weights \
      vllm-node-b12x:latest --rank 1 >/dev/null
    sleep 2
  fi
"

run_tp2_pass() {
  local label="$1"
  local port="$2"
  echo ""
  echo "=================================================================="
  echo "  Running Gate 2 Pass: $label (TP=2 across spark-5a4e & spark-8a27)"
  echo "=================================================================="

  # Clean any prior gate2 container
  docker rm -f glm53-gate2-head >/dev/null 2>&1 || true
  ssh -o BatchMode=yes "$WORKER_HOST" "docker rm -f glm53-gate2-worker >/dev/null 2>&1 || true"

  # Start Worker (rank 1) FIRST per GEMINI.md Rule 2
  ssh -o BatchMode=yes "$WORKER_HOST" "
    docker run -d --name glm53-gate2-worker \
      --gpus all --ipc=host --network=host --privileged \
      --ulimit memlock=-1:-1 \
      -v /home/massoud/.cache/huggingface:/root/.cache/huggingface \
      -v /tmp:/tmp \
      -v $ROOT_DIR:/workspace \
      -e GLOO_SOCKET_IFNAME=enp1s0f1np1 \
      -e NCCL_SOCKET_IFNAME=enp1s0f1np1 \
      -e NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1 \
      -e NCCL_IB_GID_INDEX=3 \
      -e NCCL_CROSS_NIC=1 \
      -e CUTE_DSL_ARCH=sm_121a \
      -e SAFETENSORS_FAST_GPU=1 \
      -e VLLM_SSM_CONV_STATE_LAYOUT=DS \
      vllm-node-b12x:latest \
      python3 /workspace/scripts/verify_gate2.py \
        --rank 1 --world-size 2 --master-addr $MASTER_ADDR --master-port $port --run-label $label >/dev/null
  "

  # Start Head (rank 0) and stream output
  docker run --rm --name glm53-gate2-head \
    --gpus all --ipc=host --network=host --privileged \
    --ulimit memlock=-1:-1 \
    -v /home/massoud/.cache/huggingface:/root/.cache/huggingface \
    -v /tmp:/tmp \
    -v "$ROOT_DIR:/workspace" \
    -e GLOO_SOCKET_IFNAME=enp1s0f1np1 \
    -e NCCL_SOCKET_IFNAME=enp1s0f1np1 \
    -e NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1 \
    -e NCCL_IB_GID_INDEX=3 \
    -e NCCL_CROSS_NIC=1 \
    -e CUTE_DSL_ARCH=sm_121a \
    -e SAFETENSORS_FAST_GPU=1 \
    -e VLLM_SSM_CONV_STATE_LAYOUT=DS \
    vllm-node-b12x:latest \
    python3 /workspace/scripts/verify_gate2.py \
      --rank 0 --world-size 2 --master-addr "$MASTER_ADDR" --master-port "$port" --run-label "$label"

  # Wait for Worker container to exit and print its logs
  ssh -o BatchMode=yes "$WORKER_HOST" "
    docker wait glm53-gate2-worker >/dev/null 2>&1 || true
    docker logs glm53-gate2-worker 2>&1 | grep -E 'shm_safetensors|GATE2_RESULT_JSON' || docker logs glm53-gate2-worker | tail -n 30
    docker rm -f glm53-gate2-worker >/dev/null 2>&1 || true
  "

  echo "--- Daemon Status After $label ---"
  "$ROOT_DIR/bin/spark-shm-weights" --rank 0 --status
  ssh -o BatchMode=yes "$WORKER_HOST" "$ROOT_DIR/bin/spark-shm-weights --rank 1 --status"
}

for pass_arg in "$@"; do
  case "$pass_arg" in
    cold)
      run_tp2_pass "cold_boot_populate" 29550
      ;;
    warm1)
      run_tp2_pass "warm_restart_1" 29551
      ;;
    warm2)
      run_tp2_pass "warm_restart_2" 29552
      ;;
    all)
      run_tp2_pass "cold_boot_populate" 29550
      run_tp2_pass "warm_restart_1" 29551
      run_tp2_pass "warm_restart_2" 29552
      ;;
  esac
done
