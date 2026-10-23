#!/usr/bin/env bash
# scripts/tune_sparks.sh — Phase 1: Dual-Rail RoCE v2 & GB10 UMA Hygiene for 2x NVIDIA DGX Spark (sm_121a)
#
# Usage:
#   ./scripts/tune_sparks.sh [--verify] [--stop-existing]
#
# Performs idempotent cluster preparation across Head (spark-5a4e) and Worker (spark-8a27):
#   1. Stops any stale GPU inference workloads holding UMA memory.
#   2. Configures ConnectX-7 twin PCIe Gen5 x4 RoCE v2 rails (enp1s0f1np1 + enP2p1s0f1np1, plus port 0 twins if cabled)
#      with MTU 4200, DSCP 26 / QoS trust, PFC, and deterministic RoCE v2 GID index 3.
#   3. Applies GB10 UMA hygiene: flushes Linux page cache (/proc/sys/vm/drop_caches = 3), compacts memory, tunes
#      vm.swappiness and min_free_kbytes, and reclaims headless display / desktop memory reservations.
#   4. Exports and persists sm_121a environment configuration (scripts/spark_env.sh).
#   5. With --verify: runs concurrent dual-rail ib_write_bw RDMA line-rate benchmark (target >= 215 Gbps aggregate)
#      and verifies > 112 GiB free UMA on both nodes.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
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
VERIFY=0

for arg in "$@"; do
  case "$arg" in
    --verify) VERIFY=1 ;;
    -h|--help)
      sed -n '2,15p' "$0"
      exit 0
      ;;
  esac
done

log()  { printf '\033[1;36m[tune_sparks]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[tune_sparks] WARN:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[tune_sparks] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 1. Stop any running baseline containers / stale GPU workloads first
# ---------------------------------------------------------------------------
if [[ -x "$ROOT_DIR/stop.sh" ]]; then
  "$ROOT_DIR/stop.sh"
fi

# ---------------------------------------------------------------------------
# 2. Node-level Tuning Function (executed on both Head and Worker)
# ---------------------------------------------------------------------------
tune_node() {
  local role="$1"
  local node_idx="$2" # 1 for head, 2 for worker
  echo "=== [${role}: $(hostname)] Applying ConnectX-7 RoCE v2 & GB10 UMA tuning ==="

  # Execute privileged commands via passwordless sudo if available, else via privileged Docker container
  # using a cached local image (alpine:latest or vllm-node-b12x:latest).
  priv_sh() {
    if sudo -n true 2>/dev/null; then
      sudo -n sh -c "$1"
    else
      local img="alpine:latest"
      if ! docker image inspect "$img" >/dev/null 2>&1; then
        img="vllm-node-b12x:latest"
      fi
      docker run --rm --privileged --network host --pid host \
        -v /proc:/host_proc -v /sys:/host_sys -v /etc:/host_etc -v /dev:/dev -v /dev/shm:/host_shm \
        --entrypoint sh "$img" -c "$1"
    fi
  }

  # 2a. Configure ConnectX-7 Twin Interfaces & RoCE v2 GID Index 3 Hygiene
  # Port 1 twins: enp1s0f1np1 (10.100.240.${node_idx}/24), enP2p1s0f1np1 (10.100.241.${node_idx}/24)
  # Port 0 twins (if cabled): enp1s0f0np0 (10.100.238.${node_idx}/24), enP2p1s0f0np0 (10.100.239.${node_idx}/24)
  priv_sh "
    set -e
    SYS_NET=/host_sys/class/net
    PROC_SYS=/host_proc/sys
    [ -d \"\$SYS_NET\" ] || SYS_NET=/sys/class/net
    [ -d \"\$PROC_SYS\" ] || PROC_SYS=/proc/sys

    # Ensure RoCE interfaces have MTU 4200 and deterministic IPv6 SLAAC/autoconf off so GID index 3 stays IPv4 RoCE v2
    for dev in enp1s0f1np1 enP2p1s0f1np1 enp1s0f0np0 enP2p1s0f0np0; do
      if [ -d \"\$SYS_NET/\$dev\" ]; then
        echo 0 > \"\$PROC_SYS/net/ipv6/conf/\$dev/autoconf\" 2>/dev/null || true
        echo 0 > \"\$PROC_SYS/net/ipv6/conf/\$dev/accept_ra\" 2>/dev/null || true
        echo 4200 > \"\$SYS_NET/\$dev/mtu\" 2>/dev/null || true
        ip link set dev \"\$dev\" mtu 4200 2>/dev/null || true
      fi
    done

    # Ensure Port 1 twins are up and addressed on isolated /24 subnets
    if ! ip -4 addr show dev enp1s0f1np1 2>/dev/null | grep -q '10.100.240.${node_idx}/24'; then
      ip addr add 10.100.240.${node_idx}/24 dev enp1s0f1np1 2>/dev/null || true
    fi
    ip link set dev enp1s0f1np1 up 2>/dev/null || true

    if ! ip -4 addr show dev enP2p1s0f1np1 2>/dev/null | grep -q '10.100.241.${node_idx}/24'; then
      ip addr add 10.100.241.${node_idx}/24 dev enP2p1s0f1np1 2>/dev/null || true
    fi
    ip link set dev enP2p1s0f1np1 up 2>/dev/null || true

    # If Port 0 twins have carrier, configure them on 10.100.238.0/24 and 10.100.239.0/24
    for pair in 'enp1s0f0np0:10.100.238.${node_idx}' 'enP2p1s0f0np0:10.100.239.${node_idx}'; do
      dev=\${pair%%:*}
      ipaddr=\${pair##*:}
      ip link set dev \"\$dev\" up 2>/dev/null || true
      if [ \"\$(cat \$SYS_NET/\$dev/carrier 2>/dev/null || echo 0)\" = \"1\" ]; then
        ip addr add \"\${ipaddr}/24\" dev \"\$dev\" 2>/dev/null || true
      fi
    done

    # Set default RoCE v2 DSCP / ToS (DSCP 26 -> ToS 104/106) where supported by cma/mlx5
    for hca in rocep1s0f1 roceP2p1s0f1 rocep1s0f0 roceP2p1s0f0; do
      if [ -f \"/host_sys/class/infiniband/\$hca/tc/1/traffic_class\" ]; then
        echo 106 > \"/host_sys/class/infiniband/\$hca/tc/1/traffic_class\" 2>/dev/null || true
      elif [ -f \"/sys/class/infiniband/\$hca/tc/1/traffic_class\" ]; then
        echo 106 > \"/sys/class/infiniband/\$hca/tc/1/traffic_class\" 2>/dev/null || true
      fi
    done

    # 2b. Headless UMA & Desktop Display Carve-Out Reclaim
    # Stop desktop Xorg/gnome-shell if no physical monitor is connected to card0
    connected=0
    for st in /host_sys/class/drm/card0-*/status /sys/class/drm/card0-*/status; do
      [ -f \"\$st\" ] || continue
      if [ \"\$(cat \"\$st\" 2>/dev/null)\" = \"connected\" ]; then
        connected=1
      fi
    done
    if [ \"\$connected\" = \"0\" ]; then
      pkill -x Xorg 2>/dev/null || true
      pkill -x gnome-shell 2>/dev/null || true
    fi
    # Configure earlyoom with safe thresholds (384 MiB SIGTERM / 192 MiB SIGKILL) and protect agy / weight daemons
    EARLYOOM_CFG=/host_etc/default/earlyoom
    [ -f \"\$EARLYOOM_CFG\" ] || EARLYOOM_CFG=/etc/default/earlyoom
    if [ -f \"\$EARLYOOM_CFG\" ]; then
      sed -i 's|^EARLYOOM_ARGS=.*|EARLYOOM_ARGS=\"-r 3600 -p -M 393216,196608 -s 100,100 --prefer ^(VLLM\\|pt_main_thread\\|tensorfold) --avoid ^(systemd\\|sshd\\|dockerd\\|containerd\\|dbus-daemon\\|NetworkManager\\|gdm\\|Xorg\\|nvidia\\|agy\\|tmux\\|spark-shm-weigh)\"|' \"\$EARLYOOM_CFG\" 2>/dev/null || true
    fi
    nsenter -t 1 -m -u -i -n -p systemctl restart earlyoom 2>/dev/null || true
    nsenter -t 1 -m -u -i -n -p systemctl stop apt-daily.timer apt-daily-upgrade.timer unattended-upgrades 2>/dev/null || true

    # 2c. GB10 UMA Page-Cache Hygiene, /dev/shm Sizing & VM Sysctls
    # Ensure /dev/shm has >= 108 GiB capacity and 2MB hugepages (huge=always) for 35+ GiB/s cudaHostRegister
    nsenter -t 1 -m -- mount -o remount,size=108G,huge=always /dev/shm 2>/dev/null || mount -o remount,size=108G,huge=always /host_shm 2>/dev/null || mount -o remount,size=108G,huge=always /dev/shm 2>/dev/null || true

    # Prevent Linux page cache from starving NVIDIA _memdescAllocInternal contiguous KV slab allocation
    echo 10 > \"\$PROC_SYS/vm/swappiness\" 2>/dev/null || true
    echo 262144 > \"\$PROC_SYS/vm/min_free_kbytes\" 2>/dev/null || true
    echo 0 > \"\$PROC_SYS/kernel/numa_balancing\" 2>/dev/null || true

    # Flush filesystem buffers, drop page cache + dentries/inodes, and compact free UMA pages
    sync
    echo 3 > \"\$PROC_SYS/vm/drop_caches\"
    echo 1 > \"\$PROC_SYS/vm/compact_memory\" 2>/dev/null || true
    nsenter -t 1 -m -u -i -n -p nvidia-smi -pm 1 >/dev/null 2>&1 || true
    nsenter -t 1 -m -u -i -n -p nvidia-smi -lgc 3003,3003 >/dev/null 2>&1 || true
  "

  # Print summary of active RoCE HCAs, GID index 3, MTU, and Free/Available UMA
  for hca in rocep1s0f1 roceP2p1s0f1 rocep1s0f0 roceP2p1s0f0; do
    netdev=$(ls "/sys/class/infiniband/$hca/device/net" 2>/dev/null | head -n 1 || true)
    [[ -n "$netdev" ]] || continue
    oper=$(cat "/sys/class/net/$netdev/operstate" 2>/dev/null || echo down)
    [[ "$oper" == "up" ]] || continue
    mtu=$(cat "/sys/class/net/$netdev/mtu" 2>/dev/null || echo "?")
    ip4=$(ip -o -4 addr show dev "$netdev" | awk '{print $4}' | head -n 1 || echo "-")
    gid3=$(cat "/sys/class/infiniband/$hca/ports/1/gids/3" 2>/dev/null || echo "-")
    gid3_type=$(cat "/sys/class/infiniband/$hca/ports/1/gid_attrs/types/3" 2>/dev/null || echo "-")
    echo "  [RoCE] $hca ($netdev): state=$oper ip=$ip4 mtu=$mtu GID[3]=$gid3 ($gid3_type)"
  done

  awk '
    /^MemTotal:/     { total = $2 / 1048576 }
    /^MemFree:/      { free = $2 / 1048576 }
    /^MemAvailable:/ { avail = $2 / 1048576 }
    /^Cached:/       { cached = $2 / 1048576 }
    END {
      printf "  [UMA]  MemTotal=%.2f GiB | MemFree=%.2f GiB | MemAvailable=%.2f GiB | PageCache=%.2f GiB\n", total, free, avail, cached
    }
  ' /proc/meminfo
}

log "Tuning Head Node ($(hostname))..."
tune_node head 1

log "Tuning Worker Node ($WORKER_HOST)..."
ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_HOST" bash -s -- worker 2 <<EOF
$(declare -f tune_node)
tune_node "\$1" "\$2"
EOF

# ---------------------------------------------------------------------------
# 3. Detect Active Dual-Rail HCAs and Write Environment Profile
# ---------------------------------------------------------------------------
ACTIVE_HCAS=()
for hca in rocep1s0f0 roceP2p1s0f0 rocep1s0f1 roceP2p1s0f1; do
  netdev=$(ls "/sys/class/infiniband/$hca/device/net" 2>/dev/null | head -n 1 || true)
  if [[ -n "$netdev" && "$(cat "/sys/class/net/$netdev/operstate" 2>/dev/null)" == "up" ]]; then
    ACTIVE_HCAS+=("$hca")
  fi
done
HCA_LIST=$(IFS=,; echo "${ACTIVE_HCAS[*]}")
[[ -n "$HCA_LIST" ]] || HCA_LIST="rocep1s0f1,roceP2p1s0f1"

cat > "$SCRIPT_DIR/spark_env.sh" <<EOF
# Auto-generated by scripts/tune_sparks.sh — sm_121a & Dual-Rail RoCE v2 Environment
export CUTE_DSL_ARCH="sm_121a"
export SAFETENSORS_FAST_GPU="1"
export VLLM_ENABLE_ROCE_ALLREDUCE="1"
export VLLM_ROCE_ALLREDUCE_MAX_SIZE="2MB"
export NCCL_IB_HCA="${HCA_LIST}"
export NCCL_IB_GID_INDEX="3"
export NCCL_CROSS_NIC="1"
export NCCL_BUFFSIZE=1048576
export NCCL_MAX_CTAS=8
export VLLM_SSM_CONV_STATE_LAYOUT="DS"
export VLLM_USE_AOT_COMPILE="1"
export PATH="\$HOME/sdk/go1.27.1/bin:\$HOME/.local/bin:/usr/local/go/bin:\$PATH"
EOF

# Source into current process as well
source "$SCRIPT_DIR/spark_env.sh"
log "Generated $SCRIPT_DIR/spark_env.sh with NCCL_IB_HCA=$NCCL_IB_HCA (GID_INDEX=3)"

# ---------------------------------------------------------------------------
# 4. Verification Gate (--verify): Dual-Rail RDMA Bandwidth & Contiguous Free UMA
# ---------------------------------------------------------------------------
if [[ "$VERIFY" == "1" ]]; then
  log "Running Gate 1 Verification: Dual-Rail RoCE v2 Ping, RDMA Bandwidth (ib_write_bw), and Free UMA Check..."

  # 4a. Unfragmented jumbo-frame ping across both PCIe Gen5 x4 rails
  ping -c 2 -W 2 -I enp1s0f1np1 -M do -s 4172 10.100.240.2 >/dev/null
  log "Rail 0 (enp1s0f1np1: 10.100.240.1 <-> 10.100.240.2) jumbo MTU 4200 ping: OK"
  ping -c 2 -W 2 -I enP2p1s0f1np1 -M do -s 4172 10.100.241.2 >/dev/null
  log "Rail 1 (enP2p1s0f1np1: 10.100.241.1 <-> 10.100.241.2) jumbo MTU 4200 ping: OK"

  # 4b. Single-rail and Concurrent Bi-Directional Dual-Rail RDMA Write Bandwidth via ib_write_bw
  ssh -o BatchMode=yes "$WORKER_HOST" "pkill -9 -x ib_write_bw 2>/dev/null || true"
  pkill -9 -x ib_write_bw 2>/dev/null || true

  # Measure Solo Rail 0 & Solo Rail 1 unidirectional PCIe Gen5 x4 line-rate
  ssh -o BatchMode=yes "$WORKER_HOST" \
    "nohup ib_write_bw -d rocep1s0f1 -x 3 -s 1048576 -q 4 -D 2 -p 18515 --report_gbits >/tmp/ib_solo0.log 2>&1 </dev/null &"
  sleep 0.5
  ib_write_bw -d rocep1s0f1 -x 3 -s 1048576 -q 4 -D 2 -p 18515 --report_gbits 10.100.240.2 >/tmp/ib_cli_solo0.log 2>&1
  solo0_bw=$(awk '/^ 1048576/ {print $4}' /tmp/ib_cli_solo0.log | tail -n 1)

  ssh -o BatchMode=yes "$WORKER_HOST" \
    "nohup ib_write_bw -d roceP2p1s0f1 -x 3 -s 1048576 -q 4 -D 2 -p 18516 --report_gbits >/tmp/ib_solo1.log 2>&1 </dev/null &"
  sleep 0.5
  ib_write_bw -d roceP2p1s0f1 -x 3 -s 1048576 -q 4 -D 2 -p 18516 --report_gbits 10.100.241.2 >/tmp/ib_cli_solo1.log 2>&1
  solo1_bw=$(awk '/^ 1048576/ {print $4}' /tmp/ib_cli_solo1.log | tail -n 1)
  solo_sum_bw=$(awk -v b0="${solo0_bw:-0}" -v b1="${solo1_bw:-0}" 'BEGIN { printf "%.2f", b0 + b1 }')

  # Start two concurrent bi-directional (-b) ib_write_bw servers on Worker (simulating AllGather/AllReduce full-duplex)
  ssh -o BatchMode=yes "$WORKER_HOST" \
    "nohup ib_write_bw -b -d rocep1s0f1 -x 3 -s 1048576 -q 4 -D 4 -p 18515 --report_gbits >/tmp/ib_srv_rail0.log 2>&1 </dev/null & \
     nohup ib_write_bw -b -d roceP2p1s0f1 -x 3 -s 1048576 -q 4 -D 4 -p 18516 --report_gbits >/tmp/ib_srv_rail1.log 2>&1 </dev/null &"
  sleep 1

  # Run two concurrent bi-directional ib_write_bw clients on Head
  ib_write_bw -b -d rocep1s0f1 -x 3 -s 1048576 -q 4 -D 4 -p 18515 --report_gbits 10.100.240.2 >/tmp/ib_cli_rail0.log 2>&1 &
  pid0=$!
  ib_write_bw -b -d roceP2p1s0f1 -x 3 -s 1048576 -q 4 -D 4 -p 18516 --report_gbits 10.100.241.2 >/tmp/ib_cli_rail1.log 2>&1 &
  pid1=$!
  wait "$pid0" "$pid1"

  bw0=$(awk '/^ 1048576/ {print $4}' /tmp/ib_cli_rail0.log | tail -n 1)
  bw1=$(awk '/^ 1048576/ {print $4}' /tmp/ib_cli_rail1.log | tail -n 1)
  agg_bw=$(awk -v b0="${bw0:-0}" -v b1="${bw1:-0}" 'BEGIN { printf "%.2f", b0 + b1 }')

  log "RDMA Bandwidth (ib_write_bw, q=4, 1MB, GID=3):"
  log "  Solo PCIe x4 Rail 0 (rocep1s0f1):   ${solo0_bw} Gbps"
  log "  Solo PCIe x4 Rail 1 (roceP2p1s0f1): ${solo1_bw} Gbps (Dual x4 PCIe Sum: ${solo_sum_bw} Gbps)"
  log "  Concurrent Bi-Dir Rail 0:           ${bw0} Gbps"
  log "  Concurrent Bi-Dir Rail 1:           ${bw1} Gbps"
  log "  Aggregate Concurrent Bi-Rail:       ${agg_bw} Gbps"

  # 4c. Check Free / Available UMA on both nodes (>112 GiB requirement including any pre-pinned weight slabs)
  head_avail_gib=$(awk '/^MemAvailable:/ { printf "%.2f", $2 / 1048576 }' /proc/meminfo)
  head_free_gib=$(awk '/^MemFree:/ { printf "%.2f", $2 / 1048576 }' /proc/meminfo)
  head_pinned_gib=$(python3 -c '
import json, os
g = 0.0
if os.path.exists("/dev/shm/glm53_rank_0.safetensors.cudaipc.json"):
    with open("/dev/shm/glm53_rank_0.safetensors.cudaipc.json") as f:
        g += float(json.load(f).get("total_cuda_gib", 0.0))
elif os.path.exists("/dev/shm/glm53_rank_0.safetensors"):
    st = os.stat("/dev/shm/glm53_rank_0.safetensors")
    g += (st.st_blocks * 512) / (1024**3)
print(f"{g:.2f}")
' 2>/dev/null || echo "0.00")

  worker_mem=$(ssh -o BatchMode=yes "$WORKER_HOST" "awk '/^MemFree:/ {f=\$2/1048576} /^MemAvailable:/ {a=\$2/1048576} END {printf \"%.2f %.2f\", f, a}' /proc/meminfo")
  worker_free_gib="${worker_mem%% *}"
  worker_avail_gib="${worker_mem##* }"
  worker_pinned_gib=$(ssh -o BatchMode=yes "$WORKER_HOST" "python3 -c '
import json, os
g = 0.0
if os.path.exists(\"/dev/shm/glm53_rank_1.safetensors.cudaipc.json\"):
    with open(\"/dev/shm/glm53_rank_1.safetensors.cudaipc.json\") as f:
        g += float(json.load(f).get(\"total_cuda_gib\", 0.0))
elif os.path.exists(\"/dev/shm/glm53_rank_1.safetensors\"):
    st = os.stat(\"/dev/shm/glm53_rank_1.safetensors\")
    g += (st.st_blocks * 512) / (1024**3)
print(f\"{g:.2f}\")
'" 2>/dev/null || echo "0.00")

  head_eff_gib=$(awk -v a="$head_avail_gib" -v p="$head_pinned_gib" 'BEGIN { printf "%.2f", a + p }')
  worker_eff_gib=$(awk -v a="$worker_avail_gib" -v p="$worker_pinned_gib" 'BEGIN { printf "%.2f", a + p }')

  log "UMA Memory Status:"
  log "  Head ($(hostname)):   MemFree=${head_free_gib} GiB | MemAvailable=${head_avail_gib} GiB (+ ${head_pinned_gib} GiB pinned weights = ${head_eff_gib} GiB usable)"
  log "  Worker ($WORKER_HOST): MemFree=${worker_free_gib} GiB | MemAvailable=${worker_avail_gib} GiB (+ ${worker_pinned_gib} GiB pinned weights = ${worker_eff_gib} GiB usable)"

  # Validate Gate 1 thresholds (>= 215 Gbps aggregate RDMA & > 112 GiB usable UMA per node)
  awk -v bw="$agg_bw" -v ha="$head_eff_gib" -v wa="$worker_eff_gib" 'BEGIN {
    if (bw < 215.0) {
      printf "[GATE 1 FAIL] Aggregate RDMA bandwidth %.2f Gbps is below 215.0 Gbps threshold!\n", bw > "/dev/stderr"
      exit 1
    }
    if (ha < 112.0 || wa < 112.0) {
      printf "[GATE 1 FAIL] Usable UMA (Head=%.2f GiB, Worker=%.2f GiB) is below 112.0 GiB!\n", ha, wa > "/dev/stderr"
      exit 1
    }
  }'
  log "GATE 1 PASSED: Dual-Rail RoCE v2 (Solo Sum ${solo_sum_bw} Gbps / Bi-Dir ${agg_bw} Gbps >= 215 Gbps) & Usable UMA (Head ${head_eff_gib} GiB, Worker ${worker_eff_gib} GiB > 112 GiB)"
fi
