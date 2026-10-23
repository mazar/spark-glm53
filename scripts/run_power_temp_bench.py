#!/usr/bin/env python3
"""
scripts/run_power_temp_bench.py — 1 Hz Dual-Node Power (Watts / Wh) & Temperature (°C)
Telemetry Wrapper for Spark-GLM53 vs. Baseline Engines on 2x NVIDIA DGX Spark (sm_121a).

Samples `nvidia-smi --query-gpu=power.draw,temperature.gpu,clocks.current.sm,utilization.gpu`
plus Linux `/sys/class/thermal/thermal_zone0/temp` at 1 Hz across BOTH Head and Worker nodes
while executing a benchmark command (e.g. `llama-benchy` or `tool-eval-bench`).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from typing import Any


def sample_head() -> dict[str, float] | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=power.draw,temperature.gpu,clocks.current.sm,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=2.0,
        ).strip()
        parts = [x.strip() for x in out.split(",")]
        cpu_temp = 0.0
        try:
            with open("/sys/class/thermal/thermal_zone0/temp") as f:
                cpu_temp = float(f.read().strip()) / 1000.0
        except Exception:
            pass
        return {
            "ts": time.time(),
            "power_w": float(parts[0]),
            "gpu_temp_c": float(parts[1]),
            "sm_clock_mhz": float(parts[2]),
            "gpu_util_pct": float(parts[3]),
            "soc_temp_c": cpu_temp,
        }
    except Exception:
        return None


def summarize_samples(samples: list[dict[str, float]], duration_s: float) -> dict[str, Any]:
    if not samples:
        return {}
    powers = [s["power_w"] for s in samples]
    gtemps = [s["gpu_temp_c"] for s in samples]
    stemps = [s["soc_temp_c"] for s in samples if s.get("soc_temp_c", 0) > 0]
    clocks = [s["sm_clock_mhz"] for s in samples]
    utils = [s["gpu_util_pct"] for s in samples]
    avg_p = sum(powers) / len(powers)
    energy_wh = avg_p * (duration_s / 3600.0)
    energy_j = avg_p * duration_s
    return {
        "sample_count": len(samples),
        "avg_power_w": round(avg_p, 2),
        "max_power_w": round(max(powers), 2),
        "min_power_w": round(min(powers), 2),
        "energy_wh": round(energy_wh, 4),
        "energy_joules": round(energy_j, 1),
        "avg_gpu_temp_c": round(sum(gtemps) / len(gtemps), 1),
        "max_gpu_temp_c": round(max(gtemps), 1),
        "start_gpu_temp_c": round(gtemps[0], 1),
        "end_gpu_temp_c": round(gtemps[-1], 1),
        "avg_soc_temp_c": round(sum(stemps) / len(stemps), 1) if stemps else 0.0,
        "max_soc_temp_c": round(max(stemps), 1) if stemps else 0.0,
        "avg_sm_clock_mhz": round(sum(clocks) / len(clocks), 0),
        "max_sm_clock_mhz": round(max(clocks), 0),
        "avg_gpu_util_pct": round(sum(utils) / len(utils), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="1 Hz Dual-Node Power & Thermal Benchmark Wrapper")
    parser.add_argument("--worker-host", default=os.environ.get("WORKER_HOST", "spark-8a27"))
    parser.add_argument("--label", default="Spark-GLM53")
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--output-json", default="/tmp/power_temp_report.json")
    parser.add_argument("cmd", nargs=argparse.REMAINDER, help="Command to run after '--'")
    args = parser.parse_args()

    cmd = args.cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        print("Error: specify a benchmark command after '--'", file=sys.stderr)
        sys.exit(1)

    head_samples: list[dict[str, float]] = []
    worker_samples: list[dict[str, float]] = []
    stop_event = threading.Event()

    # Start persistent SSH streaming sampler on Worker node to avoid per-second SSH handshake overhead
    worker_script = (
        "import subprocess, time, json\n"
        "while True:\n"
        "    try:\n"
        "        out = subprocess.check_output(['nvidia-smi','--query-gpu=power.draw,temperature.gpu,clocks.current.sm,utilization.gpu','--format=csv,noheader,nounits'], text=True).strip()\n"
        "        p = [x.strip() for x in out.split(',')]\n"
        "        st = 0.0\n"
        "        try:\n"
        "            st = float(open('/sys/class/thermal/thermal_zone0/temp').read().strip())/1000.0\n"
        "        except Exception: pass\n"
        "        print(json.dumps({'ts':time.time(),'power_w':float(p[0]),'gpu_temp_c':float(p[1]),'sm_clock_mhz':float(p[2]),'gpu_util_pct':float(p[3]),'soc_temp_c':st}), flush=True)\n"
        "    except Exception: pass\n"
        f"    time.sleep({args.interval})\n"
    )
    worker_proc = subprocess.Popen(
        ["ssh", "-o", "BatchMode=yes", args.worker_host, "python3", "-u", "-"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    assert worker_proc.stdin is not None
    worker_proc.stdin.write(worker_script)
    worker_proc.stdin.close()

    def read_worker_loop() -> None:
        assert worker_proc.stdout is not None
        for line in worker_proc.stdout:
            if stop_event.is_set():
                break
            line = line.strip()
            if not line:
                continue
            try:
                worker_samples.append(json.loads(line))
            except Exception:
                pass

    def read_head_loop() -> None:
        while not stop_event.is_set():
            t0 = time.perf_counter()
            s = sample_head()
            if s is not None:
                head_samples.append(s)
            elapsed = time.perf_counter() - t0
            stop_event.wait(max(0.05, args.interval - elapsed))

    t_head = threading.Thread(target=read_head_loop, daemon=True)
    t_worker = threading.Thread(target=read_worker_loop, daemon=True)
    t_head.start()
    t_worker.start()

    # Record 3 seconds of idle baseline before starting workload
    time.sleep(3.0)
    idle_head = summarize_samples(head_samples[:], 3.0)
    idle_worker = summarize_samples(worker_samples[:], 3.0)
    head_samples.clear()
    worker_samples.clear()

    print(f"[power_temp_bench] Starting workload ({args.label}): {' '.join(cmd)}", flush=True)
    t_start = time.perf_counter()
    proc = subprocess.run(cmd)
    duration_s = time.perf_counter() - t_start

    stop_event.set()
    try:
        worker_proc.terminate()
    except Exception:
        pass
    t_head.join(timeout=2.0)

    head_sum = summarize_samples(head_samples, duration_s)
    worker_sum = summarize_samples(worker_samples, duration_s)

    n = min(len(head_samples), len(worker_samples))
    cluster_powers = [head_samples[i]["power_w"] + worker_samples[i]["power_w"] for i in range(n)]
    cluster_avg_w = round(sum(cluster_powers) / len(cluster_powers), 2) if cluster_powers else round(
        head_sum.get("avg_power_w", 0.0) + worker_sum.get("avg_power_w", 0.0), 2
    )
    cluster_max_w = round(max(cluster_powers), 2) if cluster_powers else round(
        head_sum.get("max_power_w", 0.0) + worker_sum.get("max_power_w", 0.0), 2
    )
    cluster_energy_wh = round(cluster_avg_w * (duration_s / 3600.0), 4)
    cluster_energy_j = round(cluster_avg_w * duration_s, 1)

    report = {
        "label": args.label,
        "command": cmd,
        "exit_code": proc.returncode,
        "duration_seconds": round(duration_s, 2),
        "idle_baseline": {
            "head_avg_power_w": idle_head.get("avg_power_w"),
            "worker_avg_power_w": idle_worker.get("avg_power_w"),
            "cluster_idle_power_w": round(
                (idle_head.get("avg_power_w") or 0.0) + (idle_worker.get("avg_power_w") or 0.0), 2
            ),
            "head_idle_gpu_temp_c": idle_head.get("avg_gpu_temp_c"),
            "worker_idle_gpu_temp_c": idle_worker.get("avg_gpu_temp_c"),
        },
        "head_node": head_sum,
        "worker_node": worker_sum,
        "cluster_total": {
            "avg_power_w": cluster_avg_w,
            "max_power_w": cluster_max_w,
            "total_energy_wh": cluster_energy_wh,
            "total_energy_joules": cluster_energy_j,
            "max_gpu_temp_c": max(head_sum.get("max_gpu_temp_c", 0.0), worker_sum.get("max_gpu_temp_c", 0.0)),
            "max_soc_temp_c": max(head_sum.get("max_soc_temp_c", 0.0), worker_sum.get("max_soc_temp_c", 0.0)),
        },
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n=== Dual-Node Power & Thermal Telemetry Summary ===")
    print(json.dumps(report, indent=2), flush=True)
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
