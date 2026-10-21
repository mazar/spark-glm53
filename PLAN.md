# Spark-GLM53 — Completion Plan (decided 2026-10-20, Claude)

Baseline commit: `131bc27` (tag `pre-audit-2026-10-20`), the state after 2 days of prior work, before any of these changes.

## Decisions

| # | Question | Decision | Why |
| :--- | :--- | :--- | :--- |
| D1 | Benchmark methodology | **Honest benchmarks.** Multi-stream uses distinct prompts. 1-stream uses fresh prompts, so cross-request history replay can't inflate them. Gate 4 headline is the *real* turn-2 TTFT. The tool gauntlet makes multiple tool calls. Quality uses ≥ 50-item GSM8K/HumanEval subsets. Old identical-prompt numbers are kept, reported separately as "best case". | Numbers must describe real usage. |
| D2 | Checkpoint | **Target `nvidia/GLM-5.3-Flash-FP4-Local` (as GEMINI.md specifies).** It is public on HF (165.3 GiB, sha `76348dd`). Download now in the background. Performance work continues on the community checkpoint in the meantime. Switch once the loader/shm export supports its formats. Fix the mislabeling right away. | Quality target is defined against it. |
| D3 | Multi-stream determinism | **Semantic, not bitwise.** Report exact-match rate + first-divergence token vs. serial, plus task correctness. Benchmark-only paths (`SPARK_HOMOG_MOE_DEDUP`, hidden-state canonicalization) go off by default. | Bitwise needs batch-invariant kernels, which are a large speed cost. |
| D4 | Targets that may exceed hardware | **Keep pushing.** If a profile-backed floor shows a target is unreachable, document best-achievable plus evidence. | — |

### Checkpoint facts (verified 2026-10-20)
- **In use:** `local-inference-lab/GLM-5.3-Flash-NVFP4-Spark`. Community ModelOpt PTQ (`calib_method: max`), downloaded Sep 27. NVFP4 routed experts, **MXFP8 attention (402) + shared experts (129)**. *Not* "nvidia layout / BF16 dense/MLA", despite how BENCHMARKS.md previously described it.
- **Target:** `nvidia/GLM-5.3-Flash-FP4-Local`. `quantized_layers`: MXFP8 ×205, **FP8_PB_WO ×170**, **W4A16_NVFP4 ×77**, NVFP4 ×42, plus `tensors/native-QB.safetensors`. This isn't pure BF16 for dense/MLA either, so GEMINI.md's "BF16 dense/MLA/routers" wording is inaccurate. The new formats (FP8_PB_WO, W4A16_NVFP4) are a loader/export risk.

## Steps (each has an exit criterion)

| Step | Work | Exit criterion | Restart? |
| :--- | :--- | :--- | :--- |
| 0 | Commit snapshot. Restart gateway. Start NVIDIA checkpoint download with page-cache hygiene (`POSIX_FADV_DONTNEED` loop). | Clean git; `:8888` up; download progressing | No |
| 1 | Honest benchmark suite (D1) + fix BENCHMARKS.md mislabel. Run against the live engine. | True baseline for every target | No |
| 2 | Profile: `SPARK_PROFILE_STEPS=1 SPARK_LAYER_PROF=1`, 8-token verify step + 2,048-token prefill chunk. Stock-b12x reference (`SPEC_ENABLED=0`). Drop the misleading `nvidia/...` served-model alias. | Per-ms time budget for decode step and prefill chunk | Yes |
| 3 | Decode: remove per-step host syncs (`.item()` in copy-drafter), make sure the spec path replays as CUDA graphs, attack the dominant profile items. | Step ≤ 45 ms at 1-stream; prose/code re-measured | Yes (iterative) |
| 4 | Prefill: start from stock b12x and re-add our components one at a time, keeping only non-regressing ones. Skip aux capture on non-final chunks. | ≥ 3,600 tok/s at 8K/32K, or documented ceiling | Yes |
| 5 | Turn benchmark-only paths off by default (D3). | Distinct-prompt 8-stream measured without them | Yes |
| 6 | NVIDIA checkpoint: loader/shm export support, load at TP=2, quality comparison, re-run perf. | Engine serves NVIDIA ckpt; quality report | Yes |
| 7 | Final gates: `benchmark_suite.py` + `verify_gate4.py`, update BENCHMARKS.md, commit + tag. | Every target pass/fail with evidence | — |

Every step ends with a commit. Every measurement goes into `BENCHMARKS.md`.

## Progress log
- 2026-10-20 21:31 — Step 0: snapshot committed (`131bc27`, tag `pre-audit-2026-10-20`).
