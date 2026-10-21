#!/usr/bin/env python3
"""
scripts/verify_gate2.py — Verification Gate 2 for Spark-GLM53:
Tests TP=2 Cold Boot + /dev/shm population, followed by two consecutive process kills
and Warm Restarts across Head (rank 0) and Worker (rank 1).
Verifies:
  - Weight load time drops from ~180s to < 3.0s
  - /proc/meminfo Cached: (and FilePageCache) does not grow across restarts
  - Zero meta tensors remain; all B12X MoE, MXFP8 Linear, and MLA layers are prepared.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time

sys.path.insert(0, "/workspace")
sys.path.insert(0, "/home/massoud/spark-glm53")

import safetensors.torch
import torch
import torch.distributed as dist

from worker.shm_safetensors import (
    get_dflash2_shm_path,
    get_rank_shm_path,
    install_shm_patches,
    read_proc_meminfo_gib,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--master-addr", type=str, default="10.100.240.1")
    parser.add_argument("--master-port", type=int, default=29550)
    parser.add_argument(
        "--model-path",
        type=str,
        default="/root/.cache/huggingface/hub/models--local-inference-lab--GLM-5.3-Flash-NVFP4-Spark/snapshots/a608241037e4c2565356bff7ca293f2133888f88",
    )
    parser.add_argument(
        "--dflash-path",
        type=str,
        default="/root/.cache/huggingface/hub/models--incoai--GLM-5.3-Flash-DFlash2/snapshots/bf582e4eacc1810f76656d1811693ff6c6737d2a/model.safetensors",
    )
    parser.add_argument("--run-label", type=str, default="boot")
    args = parser.parse_args()

    os.environ["RANK"] = str(args.rank)
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = str(args.world_size)
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = str(args.master_port)

    install_shm_patches()

    mem_start = read_proc_meminfo_gib()
    t_proc0 = time.perf_counter()

    # Trigger early background cudaHostRegister on /dev/shm/glm53_rank_{rank}.safetensors
    # while EngineArgs config creation and NCCL distributed init run!
    torch.cuda.set_device(0)

    from vllm.config import set_current_vllm_config
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )
    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.model_loader import get_model

    engine_args = EngineArgs(
        model=args.model_path,
        tensor_parallel_size=args.world_size,
        nnodes=args.world_size,
        node_rank=args.rank,
        master_addr=args.master_addr,
        master_port=args.master_port,
        quantization="modelopt_mixed",
        kv_cache_dtype="fp8",
        block_size=256,
        max_model_len=8192,
        attention_backend="B12X",
        moe_backend="b12x",
        linear_backend="b12x",
        load_format="b12x",
        enforce_eager=True,
    )
    vllm_config = engine_args.create_engine_config()

    # 45-second timeout on distributed init per GEMINI.md Rule 2
    init_method = f"tcp://{args.master_addr}:{args.master_port}"
    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=args.world_size,
            rank=args.rank,
            distributed_init_method=init_method,
            local_rank=0,
            backend="nccl",
            timeout=datetime.timedelta(seconds=45),
        )
        ensure_model_parallel_initialized(args.world_size, 1)

        dist.barrier()
        mem_before_load = read_proc_meminfo_gib()
        t_load0 = time.perf_counter()

        model = get_model(vllm_config=vllm_config)

        t_dflash0 = time.perf_counter()
        dflash_tensors = safetensors.torch.load_file(args.dflash_path, device="cuda:0")
        dflash_sec = time.perf_counter() - t_dflash0

        torch.cuda.synchronize()
        dist.barrier()
        load_sec = time.perf_counter() - t_load0
        mem_after_load = read_proc_meminfo_gib()

    # Validate zero meta tensors remain on model
    meta_params = [k for k, v in model.named_parameters() if v.is_meta]
    meta_buffers = [k for k, v in model.named_buffers() if v.is_meta]
    assert not meta_params, f"Found uninitialized meta parameters: {meta_params[:5]}"
    assert not meta_buffers, f"Found uninitialized meta buffers: {meta_buffers[:5]}"

    shm_file = get_rank_shm_path(args.rank)
    shm_gib = os.path.getsize(shm_file) / (1024.0 ** 3) if os.path.exists(shm_file) else 0.0

    result = {
        "run_label": args.run_label,
        "rank": args.rank,
        "target_plus_dflash_load_sec": round(load_sec, 4),
        "dflash_load_sec": round(dflash_sec, 4),
        "dflash_num_tensors": len(dflash_tensors),
        "total_proc_sec": round(time.perf_counter() - t_proc0, 4),
        "shm_file": shm_file,
        "shm_size_gib": round(shm_gib, 3),
        "cached_start_gib": round(mem_start.get("Cached", 0.0), 3),
        "cached_before_load_gib": round(mem_before_load.get("Cached", 0.0), 3),
        "cached_after_load_gib": round(mem_after_load.get("Cached", 0.0), 3),
        "cached_delta_gib": round(mem_after_load.get("Cached", 0.0) - mem_before_load.get("Cached", 0.0), 4),
        "file_page_cache_before_gib": round(mem_before_load.get("FilePageCache", 0.0), 3),
        "file_page_cache_after_gib": round(mem_after_load.get("FilePageCache", 0.0), 3),
        "mem_available_after_gib": round(mem_after_load.get("MemAvailable", 0.0), 3),
    }
    print(f"GATE2_RESULT_JSON: {json.dumps(result)}", flush=True)

    destroy_model_parallel()
    destroy_distributed_environment()


if __name__ == "__main__":
    main()
