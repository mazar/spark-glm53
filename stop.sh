#!/usr/bin/env bash
# stop.sh — Cleanly stop stale vLLM / TensorFold / Python inference processes on both DGX Sparks
# Preserves persistent /dev/shm/glm53_rank_* segments unless --purge-shm is passed.
set -euo pipefail

WORKER_HOST="${WORKER_HOST:-spark-8a27}"
PURGE_SHM=0
for arg in "$@"; do
  case "$arg" in
    --purge-shm) PURGE_SHM=1 ;;
  esac
done

log() { printf '[stop.sh] %s\n' "$*"; }

clean_node() {
  local role="$1"
  local purge="$2"
  echo "[stop.sh:${role}] Stopping inference containers and stale worker processes..."

  # Stop known inference containers if running
  for cname in $(docker ps -a --format '{{.Names}}' 2>/dev/null | grep -E '^(glm53|spark-glm53|vllm|tensorfold)' || true); do
    echo "[stop.sh:${role}] Stopping and removing container: $cname"
    docker rm -f "$cname" >/dev/null 2>&1 || true
  done

  # Kill any host-level stale vLLM / b12x / tensorfold python workers (excluding spark-shm-weights daemon)
  pkill -9 -f "[V]LLM::" 2>/dev/null || true
  pkill -9 -f "[v]llm\.entrypoints" 2>/dev/null || true
  pkill -9 -f "[b]12x_dflash_runner\.py" 2>/dev/null || true
  pkill -9 -f "[t]ensorfold serve" 2>/dev/null || true
  pkill -9 -x ib_write_bw 2>/dev/null || true
  pkill -9 -x spark-gateway 2>/dev/null || true

  if [[ "$purge" == "1" ]]; then
    echo "[stop.sh:${role}] --purge-shm specified: stopping spark-cuda-ipc-weights, spark-shm-weights and removing /dev/shm/glm53_*"
    docker rm -f spark-cuda-ipc-weights spark-shm-weights >/dev/null 2>&1 || true
    pkill -9 -f "[c]uda_ipc_daemon\.py" 2>/dev/null || true
    pkill -9 -x spark-shm-weights 2>/dev/null || true
    docker run --rm --ipc=host vllm-node-b12x:latest bash -c "rm -f /dev/shm/glm53_* /dev/shm/dflash2_* /tmp/spark_cuda_ipc_rank_*.sock" 2>/dev/null || rm -f /dev/shm/glm53_* /dev/shm/dflash2_* /tmp/spark_cuda_ipc_rank_*.sock 2>/dev/null || true
  else
    ls -lh /dev/shm/glm53_* 2>/dev/null || true
  fi

  # Reset GB10 driver utilization telemetry counter only when spark-cuda-ipc-weights is not holding an active context
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "spark-cuda-ipc-weights"; then
    docker run --rm --gpus all vllm-node-b12x:latest python3 -c "import torch; torch.cuda.init(); torch.cuda.synchronize()" >/dev/null 2>&1 || true
  fi
}

log "Cleaning Worker node ($WORKER_HOST)..."
ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_HOST" bash -s -- worker "$PURGE_SHM" <<EOF
$(declare -f clean_node)
clean_node "\$1" "\$2"
EOF

log "Cleaning Head node ($(hostname))..."
clean_node head "$PURGE_SHM"

log "Cluster stop complete."
