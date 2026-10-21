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

Methodology "honest-v1" (PLAN.md D1/D3, 2026-10-20): every timed prompt is unique to
the run, multi-stream uses distinct prompts, identical-prompt results are reported only
as `best_case_*`, determinism is semantic, and quality uses first-N GSM8K/HumanEval slices.
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


def _base_code_json_prompt() -> str:
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



# ---------------------------------------------------------------------------
# Honest-methodology additions (PLAN.md decision D1/D3, 2026-10-20)
#
# Rules:
#   * Every timed prompt is unique to this run (run nonce + content variation), so
#     the engine's cross-request copy-draft history can never replay a previously
#     generated answer into a timed measurement.
#   * Multi-stream throughput uses DISTINCT prompts per stream. The identical-
#     prompt case is still measured, but reported separately as `best_case_*`
#     and never used for pass/fail.
#   * Determinism is semantic (D3): exact-match rate + first divergence position
#     of each parallel stream vs. a serial run of the same prompt.
#   * Quality uses fixed, comparable slices of GSM8K test + HumanEval.
# ---------------------------------------------------------------------------

import os
import random
import statistics
import tempfile

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

TARGETS = {
    "prefill_tok_per_s": 3600.0,
    "decode_1stream_prose_tok_per_s": 62.0,
    "decode_1stream_code_tok_per_s": 120.0,
    "decode_8stream_code_tok_per_s": 185.0,
    "kv_cache_tokens": 2_200_000,
    "warm_load_sec": 3.0,
}

_VARIANT_WORDS = [
    "atlas", "borealis", "cobalt", "dynamo", "ember", "fjord", "granite", "helix",
    "ion", "juniper", "krypton", "lumen", "meridian", "nimbus", "onyx", "polaris",
    "quasar", "rhodium", "sierra", "tundra", "umbra", "vortex", "willow", "xenon",
]

PROSE_TOPICS = [
    "how maritime navigation instruments—from Polynesian star compasses and the Islamic "
    "astrolabe to John Harrison's marine chronometer—transformed human conceptions of time, "
    "cartography, scientific empiricism, and international trade",
    "how the development of the printing press, paper mills, and postal networks reshaped "
    "religious authority, scientific correspondence, and vernacular literature in early "
    "modern Europe",
    "how the construction of aqueducts, sewers, and public baths shaped urban life, public "
    "health, civic identity, and imperial power across the Roman world",
    "how the Silk Road caravan networks transmitted not only goods but also mathematics, "
    "medicine, religious ideas, and artistic motifs between China, Persia, and the "
    "Mediterranean",
    "how railways and standardized time zones transformed nineteenth-century industry, "
    "warfare, leisure travel, and the experience of distance",
    "how the domestication of horses changed warfare, trade, language spread, and social "
    "hierarchy across the Eurasian steppe",
    "how the invention of the telescope and microscope altered theology, philosophy, and "
    "the self-image of seventeenth-century scholars",
    "how monsoon wind knowledge organized Indian Ocean commerce, port cities, and cultural "
    "exchange between East Africa, Arabia, India, and Southeast Asia",
]


def build_prose_prompt(topic_idx: int, nonce: str) -> str:
    topic = PROSE_TOPICS[topic_idx % len(PROSE_TOPICS)]
    return (
        f"(Essay request {nonce}) Write a detailed, comprehensive historical and philosophical "
        f"essay exploring {topic}. Use rich, varied vocabulary and continuous narrative "
        "paragraphs without bullet points or code blocks."
    )


def build_code_json_prompt(variant: str | None = None) -> str:
    """Code/JSON prompt. `variant` renames clusters/hosts so the expected output
    differs from every other variant (defeats cross-request history replay) while
    keeping the same intra-request structure (legitimate copy-draft target)."""
    base = _base_code_json_prompt()
    if variant is None:
        return base
    rng = random.Random(variant)
    word = rng.choice(_VARIANT_WORDS)
    n_extra = rng.randint(3, 6)  # extra replicas beyond the 2 template nodes
    out = (
        base.replace("spark-glm53-gb10-production", f"{word}-gb10-{variant}")
        .replace("spark-replica-", f"{word}-replica-")
        .replace("spark-5a4e", f"{word}-head-{rng.randint(100, 999)}")
        .replace("spark-8a27", f"{word}-peer-{rng.randint(100, 999)}")
        .replace("115.65", f"{rng.uniform(100, 120):.2f}")
        .replace("3850.0", f"{rng.uniform(3000, 4200):.1f}")
    )
    # Change the number of requested replicas (affects output length/content).
    out = out.replace(
        f"`{word}-replica-5` (`node_rank: 5`)",
        ", ".join(
            f"`{word}-replica-{i}` (`node_rank: {i}`)" for i in range(5, 2 + n_extra + 1)
        ),
    )
    return out


def _first_divergence(a: str, b: str) -> int:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return -1 if len(a) == len(b) else min(len(a), len(b))


async def _gather_limited(coros: list, limit: int) -> list:
    sem = asyncio.Semaphore(limit)

    async def _run(c):
        async with sem:
            return await c

    return await asyncio.gather(*[_run(c) for c in coros])


def _strip_text(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if k != "text"}


async def run_parallel(
    client: httpx.AsyncClient, base_url: str, prompts: list[str], max_tokens: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    t0 = time.perf_counter()
    outs = await asyncio.gather(
        *[
            stream_chat_completion(
                client,
                base_url,
                [{"role": "user", "content": p}],
                max_tokens=max_tokens,
                temperature=0.0,
                ignore_eos=True,
            )
            for p in prompts
        ]
    )
    wall = time.perf_counter() - t0
    total = sum(r["completion_tokens"] for r in outs)
    max_dec = max(r["decode_sec"] for r in outs)
    agg_dec = sum(max(r["completion_tokens"] - 1, 1) for r in outs) / max_dec
    return (
        {
            "parallel": len(prompts),
            "distinct_prompts": len(set(prompts)),
            "total_completion_tokens": total,
            "wall_sec": round(wall, 3),
            "aggregate_decode_tok_per_s": round(agg_dec, 2),
            "aggregate_wall_tok_per_s": round(total / wall, 2),
        },
        outs,
    )


# ------------------------------- quality -----------------------------------

def _load_jsonl(name: str) -> list[dict[str, Any]]:
    with open(os.path.join(DATA_DIR, name), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _gsm8k_extract(text: str) -> str | None:
    m = re.findall(r"####\s*\$?\s*(-?[\d,]*\.?\d+)", text)
    if not m:
        m = re.findall(r"(-?[\d,]*\.?\d+)", text)
    if not m:
        return None
    return m[-1].replace(",", "").rstrip(".")


def _num_eq(a: str | None, b: str) -> bool:
    if a is None:
        return False
    try:
        return abs(float(a) - float(b.replace(",", ""))) < 1e-6
    except ValueError:
        return a.strip() == b.strip()


def _humaneval_check(prompt: str, completion: str, test: str, entry: str) -> bool:
    code = completion
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", completion, re.DOTALL)
    if blocks:
        code = max(blocks, key=len)
    if f"def {entry}" not in code:
        code = prompt + code
    program = code + "\n\n" + test + f"\n\ncheck({entry})\n"
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "prog.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(program)
        try:
            r = subprocess.run(
                [sys.executable, path], cwd=td, capture_output=True, timeout=15
            )
            return r.returncode == 0
        except subprocess.TimeoutExpired:
            return False


async def run_quality(
    client: httpx.AsyncClient, base_url: str, n_gsm: int, n_he: int, concurrency: int
) -> dict[str, Any]:
    gsm = _load_jsonl("gsm8k_test.jsonl")[:n_gsm]
    he = _load_jsonl("humaneval.jsonl")[:n_he]

    gsm_coros = [
        stream_chat_completion(
            client,
            base_url,
            [
                {
                    "role": "user",
                    "content": ex["question"]
                    + "\n\nSolve step by step, then give the final answer on the last line as `#### <number>`.",
                }
            ],
            max_tokens=768,
            temperature=0.0,
        )
        for ex in gsm
    ]
    he_coros = [
        stream_chat_completion(
            client,
            base_url,
            [
                {
                    "role": "user",
                    "content": "Complete the following Python function. Return the complete "
                    "function (with any needed imports) in a single ```python code block.\n\n"
                    + ex["prompt"],
                }
            ],
            max_tokens=768,
            temperature=0.0,
        )
        for ex in he
    ]
    t0 = time.perf_counter()
    gsm_res = await _gather_limited(gsm_coros, concurrency)
    he_res = await _gather_limited(he_coros, concurrency)
    elapsed = time.perf_counter() - t0

    gsm_ok = [
        _num_eq(_gsm8k_extract(r["text"]), ex["answer"].split("####")[-1].strip())
        for r, ex in zip(gsm_res, gsm)
    ]
    he_ok = [
        _humaneval_check(ex["prompt"], r["text"], ex["test"], ex["entry_point"])
        for r, ex in zip(he_res, he)
    ]
    return {
        "gsm8k_n": len(gsm_ok),
        "gsm8k_accuracy": round(sum(gsm_ok) / max(len(gsm_ok), 1), 4),
        "humaneval_n": len(he_ok),
        "humaneval_pass_at_1": round(sum(he_ok) / max(len(he_ok), 1), 4),
        "gsm8k_failed_idx": [i for i, ok in enumerate(gsm_ok) if not ok],
        "humaneval_failed_ids": [ex["task_id"] for ex, ok in zip(he, he_ok) if not ok],
        "elapsed_sec": round(elapsed, 1),
        "settings": "temperature=0, enable_thinking=False, max_tokens=768, first-N slices",
    }


# ------------------------------- main run ----------------------------------

async def run_benchmarks(args: argparse.Namespace) -> dict[str, Any]:
    run_id = args.run_id or f"r{int(time.time())}"
    sections = set(args.sections.split(","))
    results: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "base_url": args.base_url,
        "run_id": run_id,
        "methodology": "honest-v1 (PLAN.md D1/D3): unique prompts per run, distinct "
        "prompts per stream, identical-prompt numbers reported only as best_case_*",
    }

    kv_info = parse_kv_cache_info_from_logs()
    results["kv_cache"] = kv_info
    print(f"[bench] KV cache info: {json.dumps(kv_info)}", flush=True)

    limits = httpx.Limits(max_keepalive_connections=32, max_connections=64)
    async with httpx.AsyncClient(limits=limits) as client:
        # Warmup with prompts that are never timed or reused.
        print("[bench] warmup...", flush=True)
        await stream_chat_completion(
            client, args.base_url,
            [{"role": "user", "content": build_synthetic_long_prompt(8400, f"warm-{run_id}")}],
            max_tokens=1,
        )
        await stream_chat_completion(
            client, args.base_url,
            [{"role": "user", "content": "List twelve uses of copper, one per line."}],
            max_tokens=128, ignore_eos=True,
        )

        if "prefill" in sections:
            for label, n in (("prefill_8k", 8050), ("prefill_32k", 31500)):
                r = await stream_chat_completion(
                    client, args.base_url,
                    [{"role": "user", "content": build_synthetic_long_prompt(n, f"{label}-{run_id}")}],
                    max_tokens=1,
                )
                results[label] = _strip_text(r)
                print(f"[bench] {label}: {results[label]}", flush=True)

        if "decode1" in sections:
            prose_runs = []
            for i in range(args.n_prose):
                r = await stream_chat_completion(
                    client, args.base_url,
                    [{"role": "user", "content": build_prose_prompt(i, f"{run_id}-p{i}")}],
                    max_tokens=512, ignore_eos=True,
                )
                prose_runs.append(_strip_text(r))
                print(f"[bench] prose[{i}]: {prose_runs[-1]['decode_tok_per_s']} tok/s", flush=True)
            results["decode_1stream_prose"] = {
                "mean_decode_tok_per_s": round(statistics.mean(r["decode_tok_per_s"] for r in prose_runs), 2),
                "runs": prose_runs,
            }
            code_runs = []
            for i in range(args.n_code):
                r = await stream_chat_completion(
                    client, args.base_url,
                    [{"role": "user", "content": build_code_json_prompt(f"{run_id}-c{i}")}],
                    max_tokens=512, ignore_eos=True,
                )
                code_runs.append(_strip_text(r))
                print(f"[bench] code[{i}]: {code_runs[-1]['decode_tok_per_s']} tok/s", flush=True)
            results["decode_1stream_code"] = {
                "mean_decode_tok_per_s": round(statistics.mean(r["decode_tok_per_s"] for r in code_runs), 2),
                "runs": code_runs,
            }

        if "multi" in sections:
            for parallel in (4, 8):
                prompts = [build_code_json_prompt(f"{run_id}-m{parallel}-{i}") for i in range(parallel)]
                summary, outs = await run_parallel(client, args.base_url, prompts, 512)
                results[f"decode_{parallel}stream_code_distinct"] = summary
                print(f"[bench] {parallel}-stream distinct: {summary}", flush=True)
                if parallel == 4:
                    # Semantic determinism (D3): serial re-run of the same 4 prompts.
                    matches, divs = [], []
                    for p, par in zip(prompts, outs):
                        ser = await stream_chat_completion(
                            client, args.base_url, [{"role": "user", "content": p}],
                            max_tokens=512, ignore_eos=True,
                        )
                        matches.append(ser["text"] == par["text"])
                        divs.append(_first_divergence(ser["text"], par["text"]))
                    results["determinism_4stream_vs_serial"] = {
                        "exact_match_rate": round(sum(matches) / len(matches), 3),
                        "first_divergence_char": divs,
                        "avg_output_chars": round(statistics.mean(len(o["text"]) for o in outs)),
                    }
                    print(f"[bench] determinism: {results['determinism_4stream_vs_serial']}", flush=True)

            if not args.skip_best_case:
                same = build_code_json_prompt(f"{run_id}-same")
                summary, outs = await run_parallel(client, args.base_url, [same] * 8, 512)
                summary["all_streams_identical"] = all(o["text"] == outs[0]["text"] for o in outs)
                summary["note"] = "IDENTICAL prompt x8 — best case only, not used for pass/fail"
                results["best_case_8stream_identical_prompt"] = summary
                print(f"[bench] best-case identical x8: {summary}", flush=True)

        if "quality" in sections:
            print(f"[bench] quality: GSM8K n={args.quality_n}, HumanEval n={args.quality_n}...", flush=True)
            results["quality"] = await run_quality(
                client, args.base_url, args.quality_n, args.quality_n, args.quality_concurrency
            )
            print(f"[bench] quality: { {k: v for k, v in results['quality'].items() if not k.endswith(('_idx', '_ids'))} }", flush=True)

    # Pass/fail summary against GEMINI.md targets (only honest metrics).
    verdict: dict[str, Any] = {}
    def _v(name: str, value: float | None, target: float, higher_better: bool = True) -> None:
        if value is None:
            return
        ok = value >= target if higher_better else value < target
        verdict[name] = {"value": value, "target": target, "pass": ok}

    if "prefill_8k" in results:
        _v("prefill_8k_tok_per_s", results["prefill_8k"]["prefill_tok_per_s"], TARGETS["prefill_tok_per_s"])
        _v("prefill_32k_tok_per_s", results["prefill_32k"]["prefill_tok_per_s"], TARGETS["prefill_tok_per_s"])
    if "decode_1stream_prose" in results:
        _v("decode_1stream_prose", results["decode_1stream_prose"]["mean_decode_tok_per_s"], TARGETS["decode_1stream_prose_tok_per_s"])
        _v("decode_1stream_code", results["decode_1stream_code"]["mean_decode_tok_per_s"], TARGETS["decode_1stream_code_tok_per_s"])
    if "decode_8stream_code_distinct" in results:
        _v("decode_8stream_code_distinct", results["decode_8stream_code_distinct"]["aggregate_wall_tok_per_s"], TARGETS["decode_8stream_code_tok_per_s"])
    _v("kv_cache_tokens", kv_info.get("kv_cache_tokens"), TARGETS["kv_cache_tokens"])
    _v("warm_load_sec", kv_info.get("warm_load_sec"), TARGETS["warm_load_sec"], higher_better=False)
    results["verdict"] = verdict

    print("\n================ BENCHMARK SUMMARY (honest-v1) ================")
    for k, v in verdict.items():
        print(f"  {'PASS' if v['pass'] else 'FAIL'}  {k}: {v['value']}  (target {v['target']})")
    print(json.dumps(results, indent=2), flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL)
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument("--sections", type=str, default="prefill,decode1,multi,quality")
    parser.add_argument("--n-prose", type=int, default=3)
    parser.add_argument("--n-code", type=int, default=2)
    parser.add_argument("--quality-n", type=int, default=50)
    parser.add_argument("--quality-concurrency", type=int, default=8)
    parser.add_argument("--skip-best-case", action="store_true")
    args = parser.parse_args()

    res = asyncio.run(run_benchmarks(args))
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
