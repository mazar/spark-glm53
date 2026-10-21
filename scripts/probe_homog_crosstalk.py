"""Correctness probe: do two DIFFERENT prompts with IDENTICAL token length, sent
concurrently, each get their own answer? (Tests the length-only "homogeneous
batch" detection in b12x_dflash_runner.py.)"""
import asyncio, json, sys, httpx

BASE = "http://127.0.0.1:8000"
PAIRS = [
    ("What is the capital city of France? Answer with one word.",
     "What is the capital city of Japan? Answer with one word."),
    ("Write the numbers from one to twenty in English words, separated by commas.",
     "Write the numbers from one to twenty in French words, separated by commas."),
]


async def ntok(c, text):
    r = await c.post(f"{BASE}/tokenize", json={"model": "GLM-5.3-Flash", "messages": [{"role": "user", "content": text}]})
    return r.json()["count"]


async def chat(c, text, n=48):
    r = await c.post(f"{BASE}/v1/chat/completions", json={
        "model": "GLM-5.3-Flash", "messages": [{"role": "user", "content": text}],
        "max_tokens": n, "temperature": 0.0, "ignore_eos": True,
        "chat_template_kwargs": {"enable_thinking": False}}, timeout=120)
    return r.json()["choices"][0]["message"]["content"]


async def main():
    bad = 0
    async with httpx.AsyncClient() as c:
        for a, b in PAIRS:
            la, lb = await ntok(c, a), await ntok(c, b)
            sa, sb = await chat(c, a), await chat(c, b)
            pa, pb = await asyncio.gather(chat(c, a), chat(c, b))
            ok_a, ok_b = pa == sa, pb == sb
            crossed = pb == sa or pa == sb
            print(json.dumps({"len_a": la, "len_b": lb, "serial_a": sa[:80], "serial_b": sb[:80],
                              "par_a": pa[:80], "par_b": pb[:80], "a_matches_serial": ok_a,
                              "b_matches_serial": ok_b, "CROSSED_OUTPUTS": crossed}, ensure_ascii=False))
            bad += crossed
    sys.exit(1 if bad else 0)

asyncio.run(main())
