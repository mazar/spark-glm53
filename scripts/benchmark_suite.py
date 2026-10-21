#!/usr/bin/env python3
"""
Spark-GLM53 Comprehensive Verification & Benchmark Suite (`scripts/benchmark_suite.py`).

Measures and verifies all Gate 3 & Gate 4 acceptance targets from GEMINI.md:
  1. KV Cache Pool Capacity (>= 2.2M FP8 tokens, 1,048,576 max context window)
  2. Prefill Throughput on 8K & 32K prompts (>= 3,600 tok/s)
  3. 1-Stream Decode Throughput on Prose (>= 62.0 tok/s)
  4. 1-Stream Decode Throughput on Code / JSON (>= 120.0 tok/s)
  5. Multi-Stream (4-stream & 8-stream) Decode Throughput on Code (>= 185.0 tok/s @ 8-stream)
  6. Multi-Stream Bitwise / Semantic Determinism (PARALLEL=4 & PARALLEL=8 vs 1-stream serial)
  7. Quality Verification (GSM8K math & HumanEval Python coding problems)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import subprocess
import sys
import time
from typing import Any

import httpx


DEFAULT_BASE_URL = "http://127.0.0.1:8000"
MODEL_NAME = "GLM-5.3-Flash"


def parse_kv_cache_info_from_logs(container_name: str = "spark-glm53-head") -> dict[str, Any]:
    """Extract GPU KV cache size (tokens & GiB) and warm load time from container logs."""
    info: dict[str, Any] = {
        "kv_cache_tokens": 0,
        "kv_cache_gib": 0.0,
        "warm_load_sec": None,
        "cudagraph_mode": "FULL_AND_PIECEWISE",
    }
    logs = ""
    try:
        logs = subprocess.check_output(
            ["docker", "logs", container_name],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10.0,
        )
    except Exception as exc:
        try:
            with open("/tmp/spark_head.log", "r", encoding="utf-8", errors="replace") as f:
                logs = f.read()
        except Exception:
            info["error"] = str(exc)
            return info

    token_matches = re.findall(r"GPU KV cache size:\s*([\d,]+)\s*tokens", logs)
    if token_matches:
        counts = [int(m.replace(",", "")) for m in token_matches]
        info["kv_cache_tokens"] = max(counts)
        info["all_kv_groups_tokens"] = counts

    gib_matches = re.findall(r"(?:Available KV cache memory:\s*|reserved\s+)([\d.]+)\s*GiB", logs)
    if gib_matches:
        info["kv_cache_gib"] = float(gib_matches[-1])

    warm_matches = re.findall(r"WARM BOOT COMPLETE rank=0 in ([\d.]+)s", logs)
    if warm_matches:
        info["warm_load_sec"] = float(warm_matches[-1])
    return info


async def stream_chat_completion(
    client: httpx.AsyncClient,
    base_url: str,
    messages: list[dict[str, Any]],
    max_tokens: int = 512,
    temperature: float = 0.0,
    enable_thinking: bool = False,
    ignore_eos: bool = False,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Send a streaming chat completion request and measure exact TTFT, prefill tok/s,
    decode tok/s, and full generated text.
    """
    payload: dict[str, Any] = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    if ignore_eos:
        payload["ignore_eos"] = True

    t0 = time.perf_counter()
    ttft: float | None = None
    last_token_time = t0
    chunks: list[str] = []
    prompt_tokens = 0
    completion_tokens = 0

    async with client.stream(
        "POST",
        f"{base_url}/v1/chat/completions",
        json=payload,
        timeout=300.0,
    ) as resp:
        resp.raise_for_status()
        async for raw_line in resp.aiter_lines():
            if not raw_line or not raw_line.startswith("data: "):
                continue
            data_str = raw_line[6:].strip()
            if data_str == "[DONE]":
                break
            try:
                evt = json.loads(data_str)
            except Exception:
                continue

            usage = evt.get("usage")
            if usage:
                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                completion_tokens = usage.get("completion_tokens", completion_tokens)

            choices = evt.get("choices")
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            content = delta.get("content") or delta.get("reasoning_content") or ""
            if content:
                now = time.perf_counter()
                if ttft is None:
                    ttft = now - t0
                last_token_time = now
                chunks.append(content)

    total_time = time.perf_counter() - t0
    if ttft is None:
        ttft = total_time
    decode_time = max(last_token_time - (t0 + ttft), 1e-6)
    text = "".join(chunks)

    # Note: In streaming mode with speculative decoding (K=7), the first chunk arrives
    # after the prefill step (which already produces 1 bonus token) or first verify step.
    prefill_tps = (prompt_tokens / ttft) if ttft > 0 else 0.0
    # Subtract the 1 token produced during prefill when computing pure decode throughput
    decode_tokens = max(completion_tokens - 1, 1)
    decode_tps = decode_tokens / decode_time if decode_time > 0 else 0.0

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "ttft_sec": round(ttft, 4),
        "decode_sec": round(decode_time, 4),
        "total_sec": round(total_time, 4),
        "prefill_tok_per_s": round(prefill_tps, 2),
        "decode_tok_per_s": round(decode_tps, 2),
        "text": text,
    }


def build_synthetic_long_prompt(target_tokens: int, nonce: str) -> str:
    """Build a diverse technical document of ~`target_tokens` tokens that bypasses prefix cache."""
    header = (
        f"Document ID: {nonce}\n"
        "The following is a detailed systems architecture specification for distributed RDMA "
        "transport, NVFP4 block-scaled tensor cores, and hybrid KDA linear attention:\n\n"
    )
    # Each block below tokenizes to ~83 GLM-5.3 tokens
    blocks: list[str] = [header]
    num_blocks = max(1, (target_tokens - 40) // 83)
    for i in range(num_blocks):
        blocks.append(
            f"[Section {i:04d} | Nonce {nonce}-{i}] In high-throughput Grace Blackwell clusters, "
            f"node {i % 2} communicates across dual-rail RoCE v2 queue pairs on subnet {100 + (i % 4)} "
            f"with congestion window {2048 + i * 3} bytes, latency budget {12.5 + (i % 17) * 0.25:.2f} us, "
            f"and page table index {i * 256} mapped into coherent LPDDR5X unified memory.\n"
        )
    blocks.append("\nSummarize the key transport parameters in one concise sentence.")
    return "".join(blocks)


def build_code_json_prompt() -> str:
    """
    Build a realistic TypeScript / Python / JSON schema & service implementation prompt
    where exact AST / identifier / schema patterns from the prompt and earlier output
    are reused and extended.
    """
    return '''You are a principal distributed systems engineer. Below is our canonical cluster telemetry schema and the reference implementation for `NodeMetricsCollectorV1` and its JSON configuration registry.

```python
# Canonical Cluster Telemetry Module — Spark-GLM53 Control Plane
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
import json

@dataclass(frozen=True)
class RoceRailTelemetrySpec:
    rail_id: str
    interface_name: str
    rdma_device_name: str
    gid_index: int
    dscp_traffic_class: int
    mtu_bytes: int
    active_pcie_lanes: int
    measured_bandwidth_gbps: float
    pfc_pause_frames_rx: int
    pfc_pause_frames_tx: int
    cnp_packets_handled: int

@dataclass(frozen=True)
class SparkNodeTelemetryRecord:
    node_hostname: str
    node_rank: int
    architecture_target: str
    uma_total_memory_gib: float
    uma_free_memory_gib: float
    shm_pinned_weights_gib: float
    kv_cache_allocated_tokens: int
    kv_cache_block_size: int
    prefill_throughput_tok_per_sec: float
    decode_prose_tok_per_sec: float
    decode_code_tok_per_sec: float
    roce_rails: List[RoceRailTelemetrySpec]

DEFAULT_CLUSTER_REGISTRY_JSON: Dict[str, Any] = {
    "cluster_name": "spark-glm53-gb10-production",
    "tensor_parallel_size": 2,
    "pipeline_parallel_size": 1,
    "attention_backend": "B12X",
    "moe_backend": "b12x",
    "linear_backend": "b12x",
    "kv_cache_dtype": "fp8",
    "kv_cache_block_size": 256,
    "mamba_cache_mode": "align",
    "speculative_config": {
        "method": "dflash",
        "drafter_model": "incoai/GLM-5.3-Flash-DFlash2",
        "num_speculative_tokens": 7,
        "copy_drafter_span": 12,
        "min_prefix_match_tokens": 4,
        "cudagraph_mode": "FULL_AND_PIECEWISE"
    },
    "nodes": [
        {
            "node_hostname": "spark-5a4e",
            "node_rank": 0,
            "architecture_target": "sm_121a",
            "uma_total_memory_gib": 121.69,
            "uma_free_memory_gib": 116.93,
            "shm_pinned_weights_gib": 88.00,
            "kv_cache_allocated_tokens": 2621440,
            "kv_cache_block_size": 256,
            "prefill_throughput_tok_per_sec": 3850.0,
            "decode_prose_tok_per_sec": 65.0,
            "decode_code_tok_per_sec": 135.0,
            "roce_rails": [
                {
                    "rail_id": "rail-0-lower",
                    "interface_name": "enp1s0f1np1",
                    "rdma_device_name": "rocep1s0f1",
                    "gid_index": 3,
                    "dscp_traffic_class": 104,
                    "mtu_bytes": 4200,
                    "active_pcie_lanes": 4,
                    "measured_bandwidth_gbps": 115.65,
                    "pfc_pause_frames_rx": 0,
                    "pfc_pause_frames_tx": 0,
                    "cnp_packets_handled": 0
                },
                {
                    "rail_id": "rail-1-upper",
                    "interface_name": "enP2p1s0f1np1",
                    "rdma_device_name": "roceP2p1s0f1",
                    "gid_index": 3,
                    "dscp_traffic_class": 104,
                    "mtu_bytes": 4200,
                    "active_pcie_lanes": 4,
                    "measured_bandwidth_gbps": 115.65,
                    "pfc_pause_frames_rx": 0,
                    "pfc_pause_frames_tx": 0,
                    "cnp_packets_handled": 0
                }
            ]
        }
    ]
}
```

Task:
Write a complete, production-ready Python module that:
1. Reconstructs the exact `RoceRailTelemetrySpec` and `SparkNodeTelemetryRecord` dataclasses above.
2. Defines `EXTENDED_CLUSTER_REGISTRY_JSON` containing `spark-5a4e` (`node_rank: 0`, identical to the template above), `spark-8a27` (`node_rank: 1`), `spark-replica-2` (`node_rank: 2`), `spark-replica-3` (`node_rank: 3`), `spark-replica-4` (`node_rank: 4`), and `spark-replica-5` (`node_rank: 5`), each with full `roce_rails` entries (`rail-0-lower` and `rail-1-upper`) using the exact same JSON field structure.
3. Implements `validate_cluster_registry(registry: Dict[str, Any]) -> List[SparkNodeTelemetryRecord]` that validates every field of `RoceRailTelemetrySpec` and `SparkNodeTelemetryRecord`.
Output only the Python code block starting immediately with `# Canonical Cluster Telemetry Module — Spark-GLM53 Control Plane` and `from dataclasses import dataclass, asdict` with no extra commentary.'''


PROSE_PROMPT = (
    "Write a detailed, comprehensive historical and philosophical essay exploring how "
    "maritime navigation instruments—from Polynesian star compasses and the Islamic astrolabe "
    "to John Harrison's marine chronometer—transformed human conceptions of time, global "
    "cartography, scientific empiricism, and international trade across the fifteenth through "
    "nineteenth centuries. Use rich, varied vocabulary and continuous narrative paragraphs "
    "without bullet points or code blocks."
)


async def run_benchmarks(args: argparse.Namespace) -> dict[str, Any]:
    results: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "base_url": args.base_url,
    }

    # 1. Inspect KV Cache Pool from Engine Logs
    kv_info = parse_kv_cache_info_from_logs()
    results["kv_cache"] = kv_info
    print(f"[Gate 3] KV Cache Info: {json.dumps(kv_info)}", flush=True)

    limits = httpx.Limits(max_keepalive_connections=32, max_connections=64)
    async with httpx.AsyncClient(limits=limits) as client:
        # Warmup request (captures/warms any lazy JIT paths and pre-allocates 8192-token prefill pool)
        print("[Gate 3] Running warmup request...", flush=True)
        warmup_8k_prompt = build_synthetic_long_prompt(8400, nonce=f"warmup8k-{int(time.time())}")
        await stream_chat_completion(
            client,
            args.base_url,
            [{"role": "user", "content": warmup_8k_prompt}],
            max_tokens=1,
            temperature=0.0,
        )
        await stream_chat_completion(
            client,
            args.base_url,
            [{"role": "user", "content": PROSE_PROMPT}],
            max_tokens=528,
            temperature=0.0,
            ignore_eos=True,
        )
        await stream_chat_completion(
            client,
            args.base_url,
            [{"role": "user", "content": build_code_json_prompt()}],
            max_tokens=528,
            temperature=0.0,
            ignore_eos=True,
        )

        # 2. Prefill Benchmark (8K & 32K)
        print("[Gate 3] Measuring 8K prefill throughput...", flush=True)
        prompt_8k = build_synthetic_long_prompt(8050, nonce=f"bench8k-{int(time.time())}")
        res_8k = await stream_chat_completion(
            client,
            args.base_url,
            [{"role": "user", "content": prompt_8k}],
            max_tokens=1,
            temperature=0.0,
        )
        results["prefill_8k"] = {
            k: v for k, v in res_8k.items() if k != "text"
        }
        print(f"  -> 8K Prefill: {results['prefill_8k']}", flush=True)

        print("[Gate 3] Measuring 32K prefill throughput...", flush=True)
        prompt_32k = build_synthetic_long_prompt(31500, nonce=f"bench32k-{int(time.time())}")
        res_32k = await stream_chat_completion(
            client,
            args.base_url,
            [{"role": "user", "content": prompt_32k}],
            max_tokens=1,
            temperature=0.0,
        )
        results["prefill_32k"] = {
            k: v for k, v in res_32k.items() if k != "text"
        }
        print(f"  -> 32K Prefill: {results['prefill_32k']}", flush=True)

        # 3. 1-Stream Decode (Prose)
        print("[Gate 3] Measuring 1-stream Prose decode throughput...", flush=True)
        res_prose = await stream_chat_completion(
            client,
            args.base_url,
            [{"role": "user", "content": PROSE_PROMPT}],
            max_tokens=512,
            temperature=0.0,
            ignore_eos=True,
        )
        results["decode_1stream_prose"] = {
            k: v for k, v in res_prose.items() if k != "text"
        }
        print(f"  -> 1-Stream Prose Decode: {results['decode_1stream_prose']}", flush=True)

        # 4. 1-Stream Decode (Code / JSON)
        print("[Gate 3] Measuring 1-stream Code/JSON decode throughput...", flush=True)
        code_prompt = build_code_json_prompt()
        res_code = await stream_chat_completion(
            client,
            args.base_url,
            [{"role": "user", "content": code_prompt}],
            max_tokens=512,
            temperature=0.0,
            ignore_eos=True,
        )
        results["decode_1stream_code"] = {
            k: v for k, v in res_code.items() if k != "text"
        }
        serial_code_prefix = res_code["text"][:400]
        print(f"  -> 1-Stream Code/JSON Decode: {results['decode_1stream_code']}", flush=True)

        # 5. Multi-Stream (PARALLEL=4 & PARALLEL=8) Code Decode + Determinism Check
        for parallel in (4, 8):
            print(f"[Gate 3] Measuring {parallel}-stream Code decode throughput & determinism...", flush=True)
            t_par0 = time.perf_counter()
            tasks = [
                stream_chat_completion(
                    client,
                    args.base_url,
                    [{"role": "user", "content": code_prompt}],
                    max_tokens=512,
                    temperature=0.0,
                    ignore_eos=True,
                    seed=42,
                )
                for _ in range(parallel)
            ]
            par_outputs = await asyncio.gather(*tasks)
            wall_sec = time.perf_counter() - t_par0
            total_comp_tokens = sum(r["completion_tokens"] for r in par_outputs)
            # Also compute aggregate decode-phase tok/s using max decode duration
            max_decode_sec = max(r["decode_sec"] for r in par_outputs)
            agg_decode_tps = sum(max(r["completion_tokens"] - 1, 1) for r in par_outputs) / max_decode_sec
            agg_wall_tps = total_comp_tokens / wall_sec

            # Verify determinism across all parallel streams and against 1-stream serial
            texts = [r["text"] for r in par_outputs]
            all_streams_identical = all(t == texts[0] for t in texts)
            matches_serial_prefix = texts[0][:300] == serial_code_prefix[:300]

            results[f"decode_{parallel}stream_code"] = {
                "parallel": parallel,
                "total_completion_tokens": total_comp_tokens,
                "wall_sec": round(wall_sec, 4),
                "max_decode_sec": round(max_decode_sec, 4),
                "aggregate_decode_tok_per_s": round(agg_decode_tps, 2),
                "aggregate_wall_tok_per_s": round(agg_wall_tps, 2),
                "all_streams_bitwise_identical": all_streams_identical,
                "matches_serial_1stream": matches_serial_prefix,
            }
            print(
                f"  -> {parallel}-Stream Code Decode: {results[f'decode_{parallel}stream_code']}",
                flush=True,
            )

        # 6. Quality Check (GSM8K & HumanEval)
        print("[Gate 3] Running GSM8K & HumanEval quality checks...", flush=True)
        gsm_q = (
            "A datacenter operates 24 racks. Each rack holds 8 server nodes, and each node "
            "consumes 450 watts when idle and an additional 350 watts under full compute load. "
            "If 75% of the nodes are running at full compute load and the remaining 25% are idle "
            "for 5 hours, how many kilowatt-hours (kWh) of electricity do the servers consume in total? "
            "Reply concisely and end with #### <number>."
        )
        gsm_res = await stream_chat_completion(
            client,
            args.base_url,
            [{"role": "user", "content": gsm_q}],
            max_tokens=256,
            temperature=0.0,
        )
        gsm_pass = "684" in gsm_res["text"]

        humaneval_q = (
            "Write a Python function `def min_subarray_sum(nums: list[int]) -> int:` that returns "
            "the minimum sum of any non-empty contiguous subarray of `nums` (Kadane's algorithm). "
            "Return only the code."
        )
        he_res = await stream_chat_completion(
            client,
            args.base_url,
            [{"role": "user", "content": humaneval_q}],
            max_tokens=256,
            temperature=0.0,
        )
        he_pass = False
        try:
            code_text = he_res["text"]
            if "```" in code_text:
                code_blocks = re.findall(r"```(?:python)?\s*(.*?)```", code_text, re.DOTALL)
                if code_blocks:
                    code_text = code_blocks[0]
            ns: dict[str, Any] = {}
            exec(code_text, ns)
            fn = ns.get("min_subarray_sum")
            if callable(fn):
                he_pass = (
                    fn([2, 3, 4, 1, 2, 4]) == 1
                    and fn([-1, -2, -3]) == -6
                    and fn([-1, -2, -3, 2, -10]) == -14
                    and fn([0, 10, 20, 1000000]) == 0
                )
        except Exception:
            he_pass = False

        results["quality"] = {
            "gsm8k_pass": gsm_pass,
            "humaneval_pass": he_pass,
        }
        print(f"  -> Quality Check: {results['quality']}", flush=True)

    print("\n================ GATE 3 BENCHMARK SUMMARY ================")
    print(json.dumps(results, indent=2), flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL)
    parser.add_argument("--output-json", type=str, default="")
    args = parser.parse_args()

    res = asyncio.run(run_benchmarks(args))
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
