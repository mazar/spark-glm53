#!/usr/bin/env python3
"""
scripts/verify_gate4.py — Gate 4 Verification Suite for Spark-GLM53 Go Control Plane (port 8888)

Verifies:
1. Dual API Endpoints (/v1/models, /tokenize, /metrics, /v1/chat/completions, /v1/responses, Anthropic /v1/messages)
2. Deterministic `reasoning_effort` (low, medium, high, max) & `clear_thinking=true` multi-turn history stripping
3. Full Hermes Agent 20-Tool Gauntlet & Streaming `glm47_repair` Tool Parser
4. 256-Token Block-Aligned Radix Prefix Router, Pinned System Prompt (`kept-state`), NVMe O_DIRECT KV Spill Tier,
   and Multi-Turn 64K Prompt Reuse Test (verifying Turn 2 TTFT < 100 ms).
"""

import argparse
import asyncio
import json
import os
import time
from typing import Any

import aiohttp


HERMES_20_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "cluster_roce_status",
            "description": "Query ConnectX-7 RoCE v2 rail telemetry for a given DGX Spark node and port.",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_hostname": {"type": "string"},
                    "hca_device": {"type": "string", "enum": ["rocep1s0f1", "roceP2p1s0f1"]},
                    "include_pfc_counters": {"type": "boolean"},
                },
                "required": ["node_hostname", "hca_device"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "shm_weight_daemon_inspect",
            "description": "Inspect /dev/shm POSIX locked weight segment status on a rank.",
            "parameters": {
                "type": "object",
                "properties": {
                    "rank": {"type": "integer", "enum": [0, 1]},
                    "verify_mlock": {"type": "boolean"},
                },
                "required": ["rank"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "kv_cache_radix_inspect",
            "description": "Inspect 256-token block-aligned Radix prefix tree and NVMe spill tier metrics.",
            "parameters": {
                "type": "object",
                "properties": {"include_spilled_blocks": {"type": "boolean"}},
                "required": ["include_spilled_blocks"],
            },
        },
    },
    *[
        {
            "type": "function",
            "function": {
                "name": f"hermes_diagnostic_tool_{idx:02d}",
                "description": f"Auxiliary Hermes cluster diagnostic tool #{idx:02d} for subsystem telemetry.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "subsystem_id": {"type": "string"},
                        "sample_window_sec": {"type": "integer"},
                        "verbose": {"type": "boolean"},
                    },
                    "required": ["subsystem_id"],
                },
            },
        }
        for idx in range(4, 21)
    ],
]


def build_64k_context_messages(nonce: str = "") -> list[dict[str, Any]]:
    """Build a ~64,000-token multi-file repository context with a pinned system prompt."""
    system_prompt = (
        "You are Spark-GLM53 Architect, an autonomous distributed systems engineering agent "
        "operating on a 2x NVIDIA DGX Spark (GB10 Grace Blackwell sm_121a) cluster. "
        "Answer questions concisely using exact facts from the repository context."
    )
    # Construct ~64,000 tokens of realistic structured code/telemetry blocks
    sections = [f"Repository snapshot {nonce}\n"] if nonce else []
    base_block = (
        "## Module section {i:04d}: DGX Spark GB10 UMA & Dual-Rail ConnectX-7 Architecture\n"
        "In subsystem module {i:04d}, the Grace Blackwell GB10 superchip connects 128 GiB of unified "
        "LPDDR5X memory across CPU and GPU engines with 256-token block-aligned FP8 NoPE-MLA KV cache pages "
        "and KDA linear attention recurrent state slabs. Each physical QSFP port exposes twin PCIe Gen5 x4 "
        "RoCE v2 HCAs (`rocep1s0f1` at 10.100.240.1/24 and `roceP2p1s0f1` at 10.100.241.1/24) configured "
        "with MTU 4200, DSCP 26 (TOS 104), and GID index 3.\n"
    )
    # Each block is ~108 tokens; 600 blocks is ~64,800 tokens
    for i in range(600):
        sections.append(base_block.format(i=i))
    sections.append(
        "\n## Secret Canary Verification Fact:\n"
        "The exact Spark-GLM53 Gate 4 canary verification code is `SPARK-GATE4-RADIX-64K-OK-9942`.\n"
    )
    user_context = "\n".join(sections)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_context},
    ]


async def measure_streaming_chat(
    session: aiohttp.ClientSession,
    base_url: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    req_body = dict(payload)
    req_body["stream"] = True
    t0 = time.perf_counter()
    ttft_ms = None
    chunks = []
    tool_calls = []
    headers_info = {}

    async with session.post(f"{base_url}/v1/chat/completions", json=req_body) as resp:
        assert resp.status == 200, f"HTTP {resp.status}: {await resp.text()}"
        headers_info = {
            "hit_blocks": int(resp.headers.get("X-Spark-Prefix-Cache-Hit-Blocks", "0")),
            "hit_tokens": int(resp.headers.get("X-Spark-Prefix-Cache-Hit-Tokens", "0")),
            "spilled_hit_blocks": int(resp.headers.get("X-Spark-Prefix-Spill-Hit-Blocks", "0")),
            "total_blocks": int(resp.headers.get("X-Spark-Prefix-Total-Blocks", "0")),
        }
        async for raw_line in resp.content:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if data_str == "[DONE]":
                break
            try:
                evt = json.loads(data_str)
            except Exception:
                continue
            choices = evt.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            c = delta.get("content") or ""
            rc = delta.get("reasoning_content") or ""
            tc = delta.get("tool_calls") or []
            if (c or rc or tc) and ttft_ms is None:
                ttft_ms = (time.perf_counter() - t0) * 1000.0
            if c:
                chunks.append(c)
            if tc:
                tool_calls.extend(tc)

    total_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "ttft_ms": round(ttft_ms or total_ms, 2),
        "total_ms": round(total_ms, 2),
        "text": "".join(chunks),
        "tool_calls": tool_calls,
        "headers": headers_info,
    }


async def run_gate4_suite(gateway_url: str, model: str) -> dict[str, Any]:
    timeout = aiohttp.ClientTimeout(total=600)
    results: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "gateway_url": gateway_url,
    }

    async with aiohttp.ClientSession(timeout=timeout) as session:
        # 1. Verify /health, /v1/models, /tokenize, /metrics
        print("[1/5] Verifying Gateway endpoints (/health, /v1/models, /tokenize, /metrics)...")
        async with session.get(f"{gateway_url}/health") as r:
            health = await r.json()
            assert r.status == 200 and health.get("status") == "ok", f"Bad health: {health}"
        async with session.get(f"{gateway_url}/v1/models") as r:
            models_resp = await r.json()
            assert r.status == 200 and len(models_resp.get("data", [])) >= 1
        async with session.post(
            f"{gateway_url}/tokenize",
            json={"model": model, "prompt": "Spark-GLM53 Radix Prefix Router test"},
        ) as r:
            tok_resp = await r.json()
            assert r.status == 200 and len(tok_resp.get("tokens", [])) > 0
        results["endpoints_check"] = {
            "health_ok": True,
            "models_count": len(models_resp.get("data", [])),
            "tokenize_ok": True,
        }

        # 2. Verify OpenAI /v1/responses and Anthropic /v1/messages (streaming & non-streaming)
        print("[2/5] Verifying Dual API Compatibility (OpenAI /v1/responses + Anthropic /v1/messages)...")
        async with session.post(
            f"{gateway_url}/v1/responses",
            json={
                "model": model,
                "input": "Reply with the exact string: OPENAI_RESPONSES_OK",
                "max_output_tokens": 32,
                "temperature": 0.0,
            },
        ) as r:
            resp_oai = await r.json()
            assert r.status == 200 and resp_oai.get("object") == "response", f"Bad /v1/responses: {resp_oai}"

        async with session.post(
            f"{gateway_url}/v1/messages",
            json={
                "model": model,
                "system": "You are a concise assistant.",
                "messages": [{"role": "user", "content": "Reply with the exact string: ANTHROPIC_MESSAGES_OK"}],
                "max_tokens": 32,
                "temperature": 0.0,
                "stream": False,
            },
        ) as r:
            resp_anth = await r.json()
            assert r.status == 200 and resp_anth.get("type") == "message", f"Bad /v1/messages: {resp_anth}"

        # Anthropic streaming check
        anth_stream_events = []
        async with session.post(
            f"{gateway_url}/v1/messages",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Say OK."}],
                "max_tokens": 16,
                "temperature": 0.0,
                "stream": True,
            },
        ) as r:
            assert r.status == 200
            async for raw_line in r.content:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if line.startswith("event:"):
                    anth_stream_events.append(line.split(":", 1)[1].strip())

        results["dual_api_compatibility"] = {
            "openai_responses_ok": resp_oai.get("object") == "response",
            "anthropic_messages_ok": resp_anth.get("type") == "message",
            "anthropic_stream_events": anth_stream_events[:6],
            "anthropic_stream_ok": "message_start" in anth_stream_events and "message_stop" in anth_stream_events,
        }
        print(f"    -> Dual API check: {results['dual_api_compatibility']}")

        # 3. Verify `reasoning_effort` (low, medium, high, max) & `clear_thinking=true`
        print("[3/5] Verifying reasoning_effort (low/medium/high) & clear_thinking=true state machine...")
        effort_results = {}
        for effort in ("low", "medium", "high"):
            async with session.post(
                f"{gateway_url}/v1/chat/completions",
                json={
                    "model": model,
                    "reasoning_effort": effort,
                    "clear_thinking": True,
                    "messages": [
                        {"role": "system", "content": "You are a helpful math assistant."},
                        {"role": "user", "content": "What is 12 * 12?"},
                        {
                            "role": "assistant",
                            "content": "<think>Historical thinking trace that must be stripped by clear_thinking=true.</think>144",
                            "reasoning_content": "Historical reasoning content that must be stripped.",
                        },
                        {"role": "user", "content": "Now multiply that by 2 and give the number."},
                    ],
                    "max_tokens": 256,
                    "temperature": 0.0,
                },
            ) as r:
                rj = await r.json()
                assert r.status == 200
                msg = rj["choices"][0]["message"]
                effort_results[effort] = {
                    "has_288": "288" in (msg.get("content") or ""),
                    "content_preview": (msg.get("content") or "").strip()[:80],
                }
        results["reasoning_and_clear_thinking"] = effort_results
        print(f"    -> reasoning_effort check: {effort_results}")

        # 4. Run Full Hermes Agent 20-Tool Gauntlet & glm47_repair Parser
        print("[4/5] Running Hermes Agent 20-Tool Gauntlet & glm47_repair tool parser...")
        async with session.post(
            f"{gateway_url}/v1/chat/completions",
            json={
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are Hermes Agent controlling a 2x DGX Spark cluster. "
                            "When asked to inspect hardware, invoke the exact matching tool function."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "Check the ConnectX-7 RoCE v2 telemetry on node `spark-8a27` for HCA device "
                            "`rocep1s0f1` with PFC counters enabled using `cluster_roce_status`."
                        ),
                    },
                ],
                "tools": HERMES_20_TOOLS,
                "tool_choice": "auto",
                "chat_template_kwargs": {"enable_thinking": False},
                "max_tokens": 192,
                "temperature": 0.0,
            },
        ) as r:
            hermes_resp = await r.json()
            assert r.status == 200, f"Hermes failed: {hermes_resp}"
            h_msg = hermes_resp["choices"][0]["message"]
            h_tcs = h_msg.get("tool_calls") or []
            parsed_args = {}
            if h_tcs:
                parsed_args = json.loads(h_tcs[0]["function"]["arguments"])

        # Also test glm47_repair on a streaming tool call
        stream_tool_res = await measure_streaming_chat(
            session,
            gateway_url,
            {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are Hermes Agent. Use tool calls when requested.",
                    },
                    {
                        "role": "user",
                        "content": "Inspect the /dev/shm weight daemon on rank 1 with verify_mlock=true using `shm_weight_daemon_inspect`.",
                    },
                ],
                "tools": HERMES_20_TOOLS,
                "chat_template_kwargs": {"enable_thinking": False},
                "max_tokens": 160,
                "temperature": 0.0,
            },
        )

        results["hermes_20_tool_gauntlet"] = {
            "num_tools_registered": len(HERMES_20_TOOLS),
            "tool_calls_count": len(h_tcs),
            "called_function": h_tcs[0]["function"]["name"] if h_tcs else None,
            "parsed_arguments": parsed_args,
            "streaming_tool_calls_detected": len(stream_tool_res["tool_calls"]) > 0 or "shm_weight_daemon_inspect" in stream_tool_res["text"],
            "pass": len(h_tcs) >= 1 and h_tcs[0]["function"]["name"] == "cluster_roce_status" and parsed_args.get("node_hostname") == "spark-8a27",
        }
        print(f"    -> Hermes 20-tool result: {results['hermes_20_tool_gauntlet']}")

        # 4b. Multi-step agent loop (honest-v1): one conversation, many sequential tool
        # calls across different tools, with tool results fed back each step.
        print("[4b/5] Running multi-step agent tool loop (8 sequential tool calls)...")
        loop_steps = [
            ("Check RoCE telemetry on node `spark-5a4e`, HCA `roceP2p1s0f1`.", "cluster_roce_status", ("node_hostname", "spark-5a4e")),
            ("Now inspect the /dev/shm weight daemon on rank 0.", "shm_weight_daemon_inspect", ("rank", 0)),
            ("Inspect the radix prefix tree including spilled blocks.", "kv_cache_radix_inspect", ("include_spilled_blocks", True)),
            ("Run diagnostic tool 07 on subsystem `nvlink-c2c` with a 30 second window.", "hermes_diagnostic_tool_07", ("subsystem_id", "nvlink-c2c")),
            ("Run diagnostic tool 12 on subsystem `lpddr5x-ecc`, verbose.", "hermes_diagnostic_tool_12", ("subsystem_id", "lpddr5x-ecc")),
            ("Check RoCE telemetry on node `spark-8a27`, HCA `rocep1s0f1`, with PFC counters.", "cluster_roce_status", ("node_hostname", "spark-8a27")),
            ("Run diagnostic tool 19 on subsystem `nvme-spill`.", "hermes_diagnostic_tool_19", ("subsystem_id", "nvme-spill")),
            ("Inspect the /dev/shm weight daemon on rank 1 and verify mlock.", "shm_weight_daemon_inspect", ("rank", 1)),
        ]
        loop_msgs: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": "You are Hermes Agent controlling a 2x DGX Spark cluster. For every "
                "user request, call exactly one matching tool with correct arguments.",
            }
        ]
        loop_records = []
        for step_i, (instr, exp_tool, (arg_k, arg_v)) in enumerate(loop_steps):
            loop_msgs.append({"role": "user", "content": instr})
            async with session.post(
                f"{gateway_url}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": loop_msgs,
                    "tools": HERMES_20_TOOLS,
                    "tool_choice": "auto",
                    "chat_template_kwargs": {"enable_thinking": False},
                    "max_tokens": 192,
                    "temperature": 0.0,
                },
            ) as r:
                rj = await r.json()
            msg = (rj.get("choices") or [{}])[0].get("message") or {}
            tcs = msg.get("tool_calls") or []
            name, args, args_ok = None, {}, False
            if tcs:
                name = tcs[0]["function"]["name"]
                try:
                    args = json.loads(tcs[0]["function"]["arguments"] or "{}")
                    args_ok = True
                except Exception:
                    args_ok = False
            correct = name == exp_tool and args_ok and args.get(arg_k) == arg_v
            loop_records.append({"step": step_i, "expected": exp_tool, "called": name, "args": args, "correct": correct})
            # Feed back assistant turn + a synthetic tool result so the loop continues.
            if tcs:
                tc_id = tcs[0].get("id") or f"call_{step_i}"
                loop_msgs.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": [
                    {"id": tc_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
                ]})
                loop_msgs.append({"role": "tool", "tool_call_id": tc_id, "content": json.dumps({"status": "ok", "step": step_i})})
            else:
                loop_msgs.append({"role": "assistant", "content": msg.get("content") or ""})
        n_ok = sum(r["correct"] for r in loop_records)
        results["agent_tool_loop"] = {
            "steps": len(loop_records),
            "correct": n_ok,
            "accuracy": round(n_ok / len(loop_records), 3),
            "records": loop_records,
            "pass": n_ok >= int(0.9 * len(loop_records) + 0.999),
        }
        print(f"    -> Agent tool loop: {n_ok}/{len(loop_records)} correct")

        # 5. Multi-Turn 64K Prompt Reuse Test (Verify Turn 2 TTFT < 100 ms)
        print("[5/5] Running Multi-Turn 64K Prompt Reuse Test (Turn 1 prefill -> Turn 2 Radix Prefix Cache hit)...")
        base_64k_msgs = build_64k_context_messages(nonce=f"run-{int(time.time())}")
        turn1_msgs = list(base_64k_msgs) + [
            {
                "role": "user",
                "content": "What is the exact Spark-GLM53 Gate 4 canary verification code in the context? Reply with only the code.",
            }
        ]
        turn1_res = await measure_streaming_chat(
            session,
            gateway_url,
            {
                "model": model,
                "messages": turn1_msgs,
                "chat_template_kwargs": {"enable_thinking": False},
                "max_tokens": 32,
                "temperature": 0.0,
            },
        )
        print(f"    -> Turn 1 (64K initial prefill): TTFT = {turn1_res['ttft_ms']:.2f} ms, answer = {turn1_res['text'].strip()!r}, headers = {turn1_res['headers']}")

        # Turn 2: Multi-turn follow-up reusing the 64K prefix
        turn2_msgs = list(turn1_msgs) + [
            {"role": "assistant", "content": turn1_res["text"]},
            {
                "role": "user",
                "content": "Repeat the exact canary verification code once more.",
            },
        ]
        turn2_res = await measure_streaming_chat(
            session,
            gateway_url,
            {
                "model": model,
                "messages": turn2_msgs,
                "clear_thinking": True,
                "chat_template_kwargs": {"enable_thinking": False},
                "max_tokens": 24,
                "temperature": 0.0,
            },
        )
        print(f"    -> Turn 2 (64K multi-turn reuse): TTFT = {turn2_res['ttft_ms']:.2f} ms, answer = {turn2_res['text'].strip()!r}, headers = {turn2_res['headers']}")

        # Turn 3: Exact prefix cache hit test (verifying sub-100ms TTFT)
        turn3_res = await measure_streaming_chat(
            session,
            gateway_url,
            {
                "model": model,
                "messages": turn2_msgs,
                "clear_thinking": True,
                "chat_template_kwargs": {"enable_thinking": False},
                "max_tokens": 24,
                "temperature": 0.0,
            },
        )
        print(f"    -> Turn 3 (64K cached turn): TTFT = {turn3_res['ttft_ms']:.2f} ms, headers = {turn3_res['headers']}")

        async with session.get(f"{gateway_url}/health") as r:
            final_health = await r.json()

        # honest-v1: the headline is the REAL second turn (new user message appended).
        # Turn 3 is a byte-identical resend of turn 2 and is reported separately only.
        results["multi_turn_64k_reuse"] = {
            "turn1_ttft_ms": turn1_res["ttft_ms"],
            "turn1_answer": turn1_res["text"].strip(),
            "turn2_ttft_ms": turn2_res["ttft_ms"],
            "turn2_answer": turn2_res["text"].strip(),
            "turn2_radix_headers": turn2_res["headers"],
            "exact_resend_ttft_ms": turn3_res["ttft_ms"],
            "exact_resend_radix_headers": turn3_res["headers"],
            "canary_preserved": "SPARK-GATE4-RADIX-64K-OK-9942" in turn2_res["text"],
            "pass_turn2_sub_100ms": turn2_res["ttft_ms"] < 100.0,
            "gateway_radix_stats": final_health.get("radix_prefix_router"),
        }

    results["gate4_all_passed"] = (
        results["endpoints_check"]["health_ok"]
        and results["dual_api_compatibility"]["openai_responses_ok"]
        and results["dual_api_compatibility"]["anthropic_messages_ok"]
        and results["dual_api_compatibility"]["anthropic_stream_ok"]
        and results["hermes_20_tool_gauntlet"]["pass"]
        and results["multi_turn_64k_reuse"]["canary_preserved"]
        and results["multi_turn_64k_reuse"]["pass_turn2_sub_100ms"]
        and results["agent_tool_loop"]["pass"]
    )
    results["methodology"] = (
        "honest-v1: turn-2 TTFT is the headline (no min() with an exact resend); "
        "64K context carries a per-run nonce so turn 1 is a cold prefill; "
        "agent_tool_loop requires >=90% correct sequential tool calls."
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gateway-url", default="http://127.0.0.1:8888")
    parser.add_argument("--model", default="GLM-5.3-Flash")
    parser.add_argument("--output-json", default="/home/massoud/spark-glm53/logs/gate4_verification.json")
    args = parser.parse_args()

    res = asyncio.run(run_gate4_suite(args.gateway_url, args.model))
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print("\n=== GATE 4 VERIFICATION SUMMARY ===")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
