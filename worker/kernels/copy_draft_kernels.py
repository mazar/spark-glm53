"""
Zero-Overhead Suffix / N-Gram Copy-Drafter & Dynamic Draft Switching Kernels for GB10 (sm_121a).

Implements:
  1. `_suffix_copy_draft_kernel`: Triton GPU kernel that scans each active request's
     prompt + generated output token history (`all_token_ids[req_state_idx, :total_len]`)
     alongside `DFlash2`'s initial draft tokens (`d0, d1, d2`) for exact suffix/n-gram
     structural matches:
       - Pattern A (Full continuation agreement): `(t3, t2, t1, t0)` matches history and
         `history[c+1, c+2] == (d0, d1)` (with `+16` priority bonus when `history[c+3] == d2`),
         automatically disambiguating repeated blocks (e.g. `rail-0-lower` vs `rail-1-upper`).
       - Pattern E (1-token template variable substitution): `(t3, t2, t1, t0)` matches history,
         `history[c+1] != d0` (e.g. `"node_rank": 3`), and `history[c+2, c+3] == (d1, d2)`,
         preserving `draft_tokens[0] = d0` and filling `draft_tokens[1..K-1]` from the template!
       - Patterns B/C/D (Lookahead suffix shifts `s=1, 2, 3`): matches `(t2, t1, t0, d0, d1)`,
         `(t1, t0, d0, d1, d2)`, or `(t0, d0, d1, d2)` in history so the copy-drafter re-locks
         onto the template immediately after a variable token without waiting 3 steps.
  2. `_dynamic_draft_switch_kernel`: Fused Triton GPU kernel that dynamically merges the
     extracted 12-token copy-draft sequence into `draft_tokens` with zero CPU-GPU sync.
"""

from __future__ import annotations

import os
import torch
from vllm.triton_utils import tl, triton

MIN_PREFIX_MATCH_LEN = 4
MAX_PREFIX_MATCH_LEN = 64
COPY_DRAFT_SPAN = 7


@triton.jit
def _unconditional_copy_draft_kernel(
    all_token_ids_ptr,       # [max_num_reqs, max_model_len] int32
    all_token_ids_stride,    # int64
    total_len_ptr,           # [max_num_reqs] int32
    idx_mapping_ptr,         # [num_reqs] int32
    global_hist_ptr,         # [GLOBAL_HIST_CAP] int32
    global_hist_len,         # int32
    draft_tokens_ptr,        # [max_num_reqs, num_speculative_steps] int64 (out)
    draft_stride,            # int64
    out_match_count_ptr,     # [1] int32 (atomic counter of matched reqs)
    recent_window,           # int32
    prompt_window,           # int32
    UNCOND_MIN_MATCH: tl.constexpr,
    MAX_MATCH: tl.constexpr,
    NUM_SPEC_STEPS: tl.constexpr,
    SPAN_POW2: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_idx).to(tl.int64)
    total_len = tl.load(total_len_ptr + req_state_idx).to(tl.int32)

    if total_len < 8:
        return

    row_ptr = all_token_ids_ptr + req_state_idx * all_token_ids_stride
    t0 = tl.load(row_ptr + (total_len - 1))
    t1 = tl.load(row_ptr + (total_len - 2))
    t2 = tl.load(row_ptr + (total_len - 3))
    t3 = tl.load(row_ptr + (total_len - 4))
    t4 = tl.load(row_ptr + (total_len - 5))
    t5 = tl.load(row_ptr + (total_len - 6))
    t6 = tl.load(row_ptr + (total_len - 7))
    t7 = tl.load(row_ptr + (total_len - 8))

    is_1tok_loop = (t0 == t1) & (t1 == t2) & (t2 == t3)
    is_2tok_loop = (
        (t0 == t2) & (t2 == t4) & (t4 == t6)
        & (t1 == t3) & (t3 == t5) & (t5 == t7)
    )
    if is_1tok_loop | is_2tok_loop:
        return

    max_cand = total_len - NUM_SPEC_STEPS - 4
    min_cand_global = 7
    tid = tl.arange(0, BLOCK_SIZE)
    best_score = tl.zeros([BLOCK_SIZE], dtype=tl.int64)

    # Window 1: Recent intra-request window
    recent_start = tl.maximum(min_cand_global, max_cand - recent_window + 1)
    base = recent_start
    while base <= max_cand:
        c = base + tid
        valid = c <= max_cand
        c0 = tl.load(row_ptr + c, mask=valid, other=-1)
        c1 = tl.load(row_ptr + c - 1, mask=valid, other=-2)
        c2 = tl.load(row_ptr + c - 2, mask=valid, other=-3)
        c3 = tl.load(row_ptr + c - 3, mask=valid, other=-4)
        c4 = tl.load(row_ptr + c - 4, mask=valid, other=-5)
        c5 = tl.load(row_ptr + c - 5, mask=valid, other=-6)
        c6 = tl.load(row_ptr + c - 6, mask=valid, other=-7)
        c7 = tl.load(row_ptr + c - 7, mask=valid, other=-8)

        exact8 = (
            valid
            & (c0 == t0)
            & (c1 == t1)
            & (c2 == t2)
            & (c3 == t3)
            & (c4 == t4)
            & (c5 == t5)
            & (c6 == t6)
            & (c7 == t7)
        )
        m_len = tl.where(exact8, 8, 0)
        still_matching = exact8
        for ext in tl.static_range(8, MAX_MATCH):
            can_ext = still_matching & (c >= ext) & (total_len - 1 >= ext)
            ce = tl.load(row_ptr + c - ext, mask=can_ext, other=-1)
            te = tl.load(row_ptr + (total_len - 1 - ext), mask=(total_len - 1 >= ext), other=-2)
            still_matching = can_ext & (ce == te)
            m_len = tl.where(still_matching, ext + 1, m_len)

        copy_start = c + 1
        score = tl.where(
            m_len >= UNCOND_MIN_MATCH,
            (m_len.to(tl.int64) << 24) | (copy_start.to(tl.int64) & 0x3FFFFF),
            0,
        )
        best_score = tl.maximum(best_score, score)
        base += BLOCK_SIZE

    # Window 2: Prompt head window
    prompt_end = tl.minimum(recent_start - 1, prompt_window)
    base_p = min_cand_global
    while base_p <= prompt_end:
        c = base_p + tid
        valid = c <= prompt_end
        c0 = tl.load(row_ptr + c, mask=valid, other=-1)
        c1 = tl.load(row_ptr + c - 1, mask=valid, other=-2)
        c2 = tl.load(row_ptr + c - 2, mask=valid, other=-3)
        c3 = tl.load(row_ptr + c - 3, mask=valid, other=-4)
        c4 = tl.load(row_ptr + c - 4, mask=valid, other=-5)
        c5 = tl.load(row_ptr + c - 5, mask=valid, other=-6)
        c6 = tl.load(row_ptr + c - 6, mask=valid, other=-7)
        c7 = tl.load(row_ptr + c - 7, mask=valid, other=-8)

        exact8 = (
            valid
            & (c0 == t0)
            & (c1 == t1)
            & (c2 == t2)
            & (c3 == t3)
            & (c4 == t4)
            & (c5 == t5)
            & (c6 == t6)
            & (c7 == t7)
        )
        m_len = tl.where(exact8, 8, 0)
        still_matching = exact8
        for ext in tl.static_range(8, MAX_MATCH):
            can_ext = still_matching & (c >= ext) & (total_len - 1 >= ext)
            ce = tl.load(row_ptr + c - ext, mask=can_ext, other=-1)
            te = tl.load(row_ptr + (total_len - 1 - ext), mask=(total_len - 1 >= ext), other=-2)
            still_matching = can_ext & (ce == te)
            m_len = tl.where(still_matching, ext + 1, m_len)

        copy_start = c + 1
        score = tl.where(
            m_len >= UNCOND_MIN_MATCH,
            (m_len.to(tl.int64) << 24) | (copy_start.to(tl.int64) & 0x3FFFFF),
            0,
        )
        best_score = tl.maximum(best_score, score)
        base_p += BLOCK_SIZE

    # Window 3: Global cross-request token history window (completed prior requests)
    max_g_cand = global_hist_len - NUM_SPEC_STEPS - 4
    base_g = min_cand_global
    while base_g <= max_g_cand:
        c = base_g + tid
        valid = c <= max_g_cand
        c0 = tl.load(global_hist_ptr + c, mask=valid, other=-1)
        c1 = tl.load(global_hist_ptr + c - 1, mask=valid, other=-2)
        c2 = tl.load(global_hist_ptr + c - 2, mask=valid, other=-3)
        c3 = tl.load(global_hist_ptr + c - 3, mask=valid, other=-4)
        c4 = tl.load(global_hist_ptr + c - 4, mask=valid, other=-5)
        c5 = tl.load(global_hist_ptr + c - 5, mask=valid, other=-6)
        c6 = tl.load(global_hist_ptr + c - 6, mask=valid, other=-7)
        c7 = tl.load(global_hist_ptr + c - 7, mask=valid, other=-8)

        exact8 = (
            valid
            & (c0 == t0)
            & (c1 == t1)
            & (c2 == t2)
            & (c3 == t3)
            & (c4 == t4)
            & (c5 == t5)
            & (c6 == t6)
            & (c7 == t7)
        )
        m_len = tl.where(exact8, 8, 0)
        still_matching = exact8
        for ext in tl.static_range(8, MAX_MATCH):
            can_ext = still_matching & (c >= ext) & (total_len - 1 >= ext)
            ce = tl.load(global_hist_ptr + c - ext, mask=can_ext, other=-1)
            te = tl.load(row_ptr + (total_len - 1 - ext), mask=(total_len - 1 >= ext), other=-2)
            still_matching = can_ext & (ce == te)
            m_len = tl.where(still_matching, ext + 1, m_len)

        copy_start = c + 1
        # Give global history a +2 priority bonus on tie so completed full outputs win
        score = tl.where(
            m_len >= UNCOND_MIN_MATCH,
            ((m_len + 2).to(tl.int64) << 24)
            | (1 << 22)
            | (copy_start.to(tl.int64) & 0x3FFFFF),
            0,
        )
        best_score = tl.maximum(best_score, score)
        base_g += BLOCK_SIZE

    global_best = tl.max(best_score, axis=0)
    best_eff_len = (global_best >> 24).to(tl.int32)
    best_from_global = ((global_best >> 22) & 1).to(tl.int32)
    best_copy_start = (global_best & 0x3FFFFF).to(tl.int32)

    if best_eff_len >= UNCOND_MIN_MATCH:
        span_offs = tl.arange(0, SPAN_POW2)
        span_mask = span_offs < NUM_SPEC_STEPS
        src_ptr = tl.where(best_from_global != 0, global_hist_ptr, row_ptr)
        copied = tl.load(
            src_ptr + best_copy_start + span_offs,
            mask=span_mask,
            other=0,
        ).to(tl.int64)
        copied = tl.maximum(0, tl.minimum(copied, 151551))
        tl.store(
            draft_tokens_ptr + req_idx * draft_stride + span_offs,
            copied,
            mask=span_mask,
        )
        tl.atomic_add(out_match_count_ptr, 1)


@triton.jit
def _suffix_copy_draft_kernel(
    # Inputs
    all_token_ids_ptr,       # [max_num_reqs, max_model_len] int32
    all_token_ids_stride,    # int64
    total_len_ptr,           # [max_num_reqs] int32
    idx_mapping_ptr,         # [num_reqs] int32
    draft_tokens_ptr,        # [max_num_reqs, num_speculative_steps] int64
    draft_stride,            # int64
    global_hist_ptr,         # [GLOBAL_HIST_CAP] int32
    global_hist_len,         # int32
    # Outputs
    out_copy_tokens_ptr,     # [max_num_reqs, DRAFT_SPAN] int64
    out_copy_stride,         # int64
    out_match_len_ptr,       # [max_num_reqs] int32
    out_any_match_ptr,       # [1] int32
    # Config scalars
    recent_window,           # int32 (e.g. 16384)
    prompt_window,           # int32 (e.g. 8192)
    MIN_MATCH: tl.constexpr,
    MAX_MATCH: tl.constexpr,
    DRAFT_SPAN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_idx).to(tl.int64)
    total_len = tl.load(total_len_ptr + req_state_idx).to(tl.int32)

    # Need at least 8 tokens in current request to form an 8-token suffix
    if total_len < 8:
        tl.store(out_match_len_ptr + req_idx, 0)
        return

    row_ptr = all_token_ids_ptr + req_state_idx * all_token_ids_stride

    # Load the 8-token suffix ending at total_len - 1
    t0 = tl.load(row_ptr + (total_len - 1))
    t1 = tl.load(row_ptr + (total_len - 2))
    t2 = tl.load(row_ptr + (total_len - 3))
    t3 = tl.load(row_ptr + (total_len - 4))
    t4 = tl.load(row_ptr + (total_len - 5))
    t5 = tl.load(row_ptr + (total_len - 6))
    t6 = tl.load(row_ptr + (total_len - 7))
    t7 = tl.load(row_ptr + (total_len - 8))

    # Load DFlash2's initial draft tokens (d0, d1, d2) for branch disambiguation & lookahead
    d_row_ptr = draft_tokens_ptr + req_idx * draft_stride
    d0 = tl.load(d_row_ptr + 0).to(tl.int32)
    d1 = tl.load(d_row_ptr + 1).to(tl.int32)
    d2 = tl.load(d_row_ptr + 2).to(tl.int32)

    # Guard against trivial 1-token or 2-token repetition loops (e.g. "aaaa" or "abababab")
    is_1tok_loop = (t0 == t1) & (t1 == t2) & (t2 == t3)
    is_2tok_loop = (
        (t0 == t2) & (t2 == t4) & (t4 == t6)
        & (t1 == t3) & (t3 == t5) & (t5 == t7)
    )
    if is_1tok_loop | is_2tok_loop:
        tl.store(out_match_len_ptr + req_idx, 0)
        return

    # Candidate end index `c` must satisfy:
    #   7 <= c <= total_len - DRAFT_SPAN - 4
    max_cand = total_len - DRAFT_SPAN - 4
    min_cand_global = 7

    tid = tl.arange(0, BLOCK_SIZE)
    best_score = tl.zeros([BLOCK_SIZE], dtype=tl.int64)

    # Window 1: Recent window [max(min_cand_global, max_cand - recent_window + 1), max_cand]
    recent_start = tl.maximum(min_cand_global, max_cand - recent_window + 1)
    base = recent_start
    while base <= max_cand:
        c = base + tid
        valid = c <= max_cand
        c0 = tl.load(row_ptr + c, mask=valid, other=-1)
        c1 = tl.load(row_ptr + c - 1, mask=valid, other=-2)
        c2 = tl.load(row_ptr + c - 2, mask=valid, other=-3)
        c3 = tl.load(row_ptr + c - 3, mask=valid, other=-4)
        c4 = tl.load(row_ptr + c - 4, mask=valid, other=-5)
        c5 = tl.load(row_ptr + c - 5, mask=valid, other=-6)
        c6 = tl.load(row_ptr + c - 6, mask=valid, other=-7)
        c7 = tl.load(row_ptr + c - 7, mask=valid, other=-8)
        n0 = tl.load(row_ptr + c + 1, mask=valid, other=-9)
        n1 = tl.load(row_ptr + c + 2, mask=valid, other=-10)
        n2 = tl.load(row_ptr + c + 3, mask=valid, other=-11)

        m1 = valid & (c0 == t0)
        m2 = m1 & (c1 == t1)
        m3 = m2 & (c2 == t2)
        m4 = m3 & (c3 == t3)
        m5 = m4 & (c4 == t4)
        m6 = m5 & (c5 == t5)
        m7 = m6 & (c6 == t6)
        exact8 = m7 & (c7 == t7)

        # Contiguous suffix match length up to MAX_MATCH tokens
        m_len = tl.where(
            exact8,
            8,
            tl.where(
                m7,
                7,
                tl.where(
                    m6,
                    6,
                    tl.where(
                        m5,
                        5,
                        tl.where(
                            m4,
                            4,
                            tl.where(m3, 3, tl.where(m2, 2, tl.where(m1, 1, 0))),
                        ),
                    ),
                ),
            ),
        )
        still_matching = exact8
        for ext in tl.static_range(8, MAX_MATCH):
            can_ext = still_matching & (c >= ext) & (total_len - 1 >= ext)
            ce = tl.load(row_ptr + c - ext, mask=can_ext, other=-1)
            te = tl.load(row_ptr + (total_len - 1 - ext), mask=(total_len - 1 >= ext), other=-2)
            still_matching = can_ext & (ce == te)
            m_len = tl.where(still_matching, ext + 1, m_len)

        # High-precision copy-drafting across multi-token & bonus-token variable transitions:
        # 1. pat_a3: >=4 contiguous suffix tokens + (d0, d1, d2) == (n0, n1, n2)
        # 2. pat_a2: >=7 contiguous suffix tokens + (d0, d1) == (n0, n1)
        # 3. pat_a1: >=12 contiguous suffix tokens + d0 == n0
        # 4. pat_e:  >=3 contiguous suffix tokens + 1-token variable at d0 + (d1, d2) == (n1, n2)
        # 5. pat_var_t0: 1-token variable at t0 (bonus token) + c1..c4 == t1..t4 + (d0, d1) == (n0, n1)
        # 6. pat_relock: >=2 contiguous suffix tokens + (d0, d1, d2) == (n0, n1, n2) + pre-variable anchor
        c3_ok = (n0 == d0) & (n1 == d1) & (n2 == d2)
        pat_a3 = m4 & c3_ok
        pat_a2 = m7 & (n0 == d0) & (n1 == d1)
        pat_a1 = exact8 & (m_len >= 12) & (n0 == d0)
        pat_a = pat_a3 | pat_a2 | pat_a1

        pat_e = m3 & (~pat_a) & (n0 != d0) & (n1 == d1) & (n2 == d2)
        pat_var_t0 = (
            valid
            & (c0 != t0)
            & (c1 == t1)
            & (c2 == t2)
            & (c3 == t3)
            & (c4 == t4)
            & (n0 == d0)
            & (n1 == d1)
        )
        c8 = tl.load(row_ptr + c - 8, mask=valid & (c >= 8), other=-12)
        c9 = tl.load(row_ptr + c - 9, mask=valid & (c >= 9), other=-13)
        pre_var_ok = (
            ((c3 == t3) & (c4 == t4))
            | ((c4 == t4) & (c5 == t5))
            | ((c5 == t5) & (c6 == t6))
            | ((c4 == t3) & (c5 == t4))
            | ((c3 == t4) & (c4 == t5))
            | ((c5 == t3) & (c6 == t4))
            | ((c3 == t5) & (c4 == t6))
            | ((c6 == t4) & (c7 == t5))
            | ((c4 == t6) & (c5 == t7))
            | ((c7 == t5) & (c8 == t6))
            | ((c8 == t6) & (c9 == t7))
        )
        pat_relock = m2 & (~pat_a) & c3_ok & pre_var_ok
        pat_uncond = exact8 & (~pat_a) & (~pat_e) & (~pat_var_t0) & (~pat_relock)

        eff_len = tl.where(
            pat_a3,
            m_len + 16,
            tl.where(
                pat_a2 | pat_a1,
                m_len + 8,
                tl.where(
                    pat_var_t0 | pat_relock,
                    m_len + 10,
                    tl.where(
                        pat_e,
                        m_len + 12,
                        tl.where(pat_uncond, m_len, 0),
                    ),
                ),
            ),
        )
        keep_d0 = tl.where(pat_e & ~(pat_a | pat_var_t0 | pat_relock), 1, 0)
        copy_start = c + 1

        # Pack (eff_len << 24) | (keep_d0 << 23) | (0 << 22) | (copy_start & 0x3FFFFF)
        score = tl.where(
            eff_len > 0,
            (eff_len.to(tl.int64) << 24)
            | (keep_d0.to(tl.int64) << 23)
            | (copy_start.to(tl.int64) & 0x3FFFFF),
            0,
        )
        best_score = tl.maximum(best_score, score)
        base += BLOCK_SIZE

    # Window 2: Prompt head window [min_cand_global, min(recent_start - 1, prompt_window)]
    prompt_end = tl.minimum(recent_start - 1, prompt_window)
    base_p = min_cand_global
    while base_p <= prompt_end:
        c = base_p + tid
        valid = c <= prompt_end
        c0 = tl.load(row_ptr + c, mask=valid, other=-1)
        c1 = tl.load(row_ptr + c - 1, mask=valid, other=-2)
        c2 = tl.load(row_ptr + c - 2, mask=valid, other=-3)
        c3 = tl.load(row_ptr + c - 3, mask=valid, other=-4)
        c4 = tl.load(row_ptr + c - 4, mask=valid, other=-5)
        c5 = tl.load(row_ptr + c - 5, mask=valid, other=-6)
        c6 = tl.load(row_ptr + c - 6, mask=valid, other=-7)
        c7 = tl.load(row_ptr + c - 7, mask=valid, other=-8)
        n0 = tl.load(row_ptr + c + 1, mask=valid, other=-9)
        n1 = tl.load(row_ptr + c + 2, mask=valid, other=-10)
        n2 = tl.load(row_ptr + c + 3, mask=valid, other=-11)

        m1 = valid & (c0 == t0)
        m2 = m1 & (c1 == t1)
        m3 = m2 & (c2 == t2)
        m4 = m3 & (c3 == t3)
        m5 = m4 & (c4 == t4)
        m6 = m5 & (c5 == t5)
        m7 = m6 & (c6 == t6)
        exact8 = m7 & (c7 == t7)

        m_len = tl.where(
            exact8,
            8,
            tl.where(
                m7,
                7,
                tl.where(
                    m6,
                    6,
                    tl.where(
                        m5,
                        5,
                        tl.where(
                            m4,
                            4,
                            tl.where(m3, 3, tl.where(m2, 2, tl.where(m1, 1, 0))),
                        ),
                    ),
                ),
            ),
        )
        still_matching = exact8
        for ext in tl.static_range(8, MAX_MATCH):
            can_ext = still_matching & (c >= ext) & (total_len - 1 >= ext)
            ce = tl.load(row_ptr + c - ext, mask=can_ext, other=-1)
            te = tl.load(row_ptr + (total_len - 1 - ext), mask=(total_len - 1 >= ext), other=-2)
            still_matching = can_ext & (ce == te)
            m_len = tl.where(still_matching, ext + 1, m_len)

        c3_ok = (n0 == d0) & (n1 == d1) & (n2 == d2)
        pat_a3 = m4 & c3_ok
        pat_a2 = m7 & (n0 == d0) & (n1 == d1)
        pat_a1 = exact8 & (m_len >= 12) & (n0 == d0)
        pat_a = pat_a3 | pat_a2 | pat_a1

        pat_e = m3 & (~pat_a) & (n0 != d0) & (n1 == d1) & (n2 == d2)
        pat_var_t0 = (
            valid
            & (c0 != t0)
            & (c1 == t1)
            & (c2 == t2)
            & (c3 == t3)
            & (c4 == t4)
            & (n0 == d0)
            & (n1 == d1)
        )
        c8 = tl.load(row_ptr + c - 8, mask=valid & (c >= 8), other=-12)
        c9 = tl.load(row_ptr + c - 9, mask=valid & (c >= 9), other=-13)
        pre_var_ok = (
            ((c3 == t3) & (c4 == t4))
            | ((c4 == t4) & (c5 == t5))
            | ((c5 == t5) & (c6 == t6))
            | ((c4 == t3) & (c5 == t4))
            | ((c3 == t4) & (c4 == t5))
            | ((c5 == t3) & (c6 == t4))
            | ((c3 == t5) & (c4 == t6))
            | ((c6 == t4) & (c7 == t5))
            | ((c4 == t6) & (c5 == t7))
            | ((c7 == t5) & (c8 == t6))
            | ((c8 == t6) & (c9 == t7))
        )
        pat_relock = m2 & (~pat_a) & c3_ok & pre_var_ok
        pat_uncond = exact8 & (~pat_a) & (~pat_e) & (~pat_var_t0) & (~pat_relock)

        eff_len = tl.where(
            pat_a3,
            m_len + 16,
            tl.where(
                pat_a2 | pat_a1,
                m_len + 8,
                tl.where(
                    pat_var_t0 | pat_relock,
                    m_len + 10,
                    tl.where(
                        pat_e,
                        m_len + 12,
                        tl.where(pat_uncond, m_len, 0),
                    ),
                ),
            ),
        )
        keep_d0 = tl.where(pat_e & ~(pat_a | pat_var_t0 | pat_relock), 1, 0)
        copy_start = c + 1

        score = tl.where(
            eff_len > 0,
            (eff_len.to(tl.int64) << 24)
            | (keep_d0.to(tl.int64) << 23)
            | (copy_start.to(tl.int64) & 0x3FFFFF),
            0,
        )
        best_score = tl.maximum(best_score, score)
        base_p += BLOCK_SIZE

    # Window 3: Global cross-request token history window [min_cand_global, max_g_cand]
    max_g_cand = global_hist_len - DRAFT_SPAN - 4
    base_g = min_cand_global
    while base_g <= max_g_cand:
        c = base_g + tid
        valid = c <= max_g_cand
        c0 = tl.load(global_hist_ptr + c, mask=valid, other=-1)
        c1 = tl.load(global_hist_ptr + c - 1, mask=valid, other=-2)
        c2 = tl.load(global_hist_ptr + c - 2, mask=valid, other=-3)
        c3 = tl.load(global_hist_ptr + c - 3, mask=valid, other=-4)
        c4 = tl.load(global_hist_ptr + c - 4, mask=valid, other=-5)
        c5 = tl.load(global_hist_ptr + c - 5, mask=valid, other=-6)
        c6 = tl.load(global_hist_ptr + c - 6, mask=valid, other=-7)
        c7 = tl.load(global_hist_ptr + c - 7, mask=valid, other=-8)
        n0 = tl.load(global_hist_ptr + c + 1, mask=valid, other=-9)
        n1 = tl.load(global_hist_ptr + c + 2, mask=valid, other=-10)
        n2 = tl.load(global_hist_ptr + c + 3, mask=valid, other=-11)

        m1 = valid & (c0 == t0)
        m2 = m1 & (c1 == t1)
        m3 = m2 & (c2 == t2)
        m4 = m3 & (c3 == t3)
        m5 = m4 & (c4 == t4)
        m6 = m5 & (c5 == t5)
        m7 = m6 & (c6 == t6)
        exact8 = m7 & (c7 == t7)

        m_len = tl.where(
            exact8,
            8,
            tl.where(
                m7,
                7,
                tl.where(
                    m6,
                    6,
                    tl.where(
                        m5,
                        5,
                        tl.where(
                            m4,
                            4,
                            tl.where(m3, 3, tl.where(m2, 2, tl.where(m1, 1, 0))),
                        ),
                    ),
                ),
            ),
        )
        still_matching = exact8
        for ext in tl.static_range(8, MAX_MATCH):
            can_ext = still_matching & (c >= ext) & (total_len - 1 >= ext)
            ce = tl.load(global_hist_ptr + c - ext, mask=can_ext, other=-1)
            te = tl.load(row_ptr + (total_len - 1 - ext), mask=(total_len - 1 >= ext), other=-2)
            still_matching = can_ext & (ce == te)
            m_len = tl.where(still_matching, ext + 1, m_len)

        c3_ok = (n0 == d0) & (n1 == d1) & (n2 == d2)
        pat_a3 = m4 & c3_ok
        pat_a2 = m7 & (n0 == d0) & (n1 == d1)
        pat_a1 = exact8 & (m_len >= 12) & (n0 == d0)
        pat_a = pat_a3 | pat_a2 | pat_a1

        pat_e = m3 & (~pat_a) & (n0 != d0) & (n1 == d1) & (n2 == d2)
        pat_var_t0 = (
            valid
            & (c0 != t0)
            & (c1 == t1)
            & (c2 == t2)
            & (c3 == t3)
            & (c4 == t4)
            & (n0 == d0)
            & (n1 == d1)
        )
        c8 = tl.load(global_hist_ptr + c - 8, mask=valid & (c >= 8), other=-12)
        c9 = tl.load(global_hist_ptr + c - 9, mask=valid & (c >= 9), other=-13)
        pre_var_ok = (
            ((c3 == t3) & (c4 == t4))
            | ((c4 == t4) & (c5 == t5))
            | ((c5 == t5) & (c6 == t6))
            | ((c4 == t3) & (c5 == t4))
            | ((c3 == t4) & (c4 == t5))
            | ((c5 == t3) & (c6 == t4))
            | ((c3 == t5) & (c4 == t6))
            | ((c6 == t4) & (c7 == t5))
            | ((c4 == t6) & (c5 == t7))
            | ((c7 == t5) & (c8 == t6))
            | ((c8 == t6) & (c9 == t7))
        )
        pat_relock = m2 & (~pat_a) & c3_ok & pre_var_ok
        pat_uncond = exact8 & (~pat_a) & (~pat_e) & (~pat_var_t0) & (~pat_relock)

        eff_len = tl.where(
            pat_a3,
            m_len + 16,
            tl.where(
                pat_a2 | pat_a1,
                m_len + 8,
                tl.where(
                    pat_var_t0 | pat_relock,
                    m_len + 10,
                    tl.where(
                        pat_e,
                        m_len + 12,
                        tl.where(pat_uncond, m_len, 0),
                    ),
                ),
            ),
        )
        keep_d0 = tl.where(pat_e & ~(pat_a | pat_var_t0 | pat_relock), 1, 0)
        copy_start = c + 1

        # Set bit 22 = 1 to mark global history source
        score = tl.where(
            eff_len > 0,
            (eff_len.to(tl.int64) << 24)
            | (keep_d0.to(tl.int64) << 23)
            | (1 << 22)
            | (copy_start.to(tl.int64) & 0x3FFFFF),
            0,
        )
        best_score = tl.maximum(best_score, score)
        base_g += BLOCK_SIZE

    global_best = tl.max(best_score, axis=0)
    best_eff_len = (global_best >> 24).to(tl.int32)
    best_keep_d0 = ((global_best >> 23) & 1).to(tl.int32)
    best_from_global = ((global_best >> 22) & 1).to(tl.int32)
    best_copy_start = (global_best & 0x3FFFFF).to(tl.int32)

    # Encode (best_keep_d0 << 16) | best_eff_len into out_match_len_ptr
    tl.store(out_match_len_ptr + req_idx, (best_keep_d0 << 16) | best_eff_len)

    if best_eff_len >= MIN_MATCH:
        tl.store(out_any_match_ptr, 1)
        span_offs = tl.arange(0, DRAFT_SPAN)
        span_mask = span_offs < DRAFT_SPAN
        src_ptr = tl.where(best_from_global != 0, global_hist_ptr, row_ptr)
        copied = tl.load(
            src_ptr + best_copy_start + span_offs,
            mask=span_mask,
            other=0,
        ).to(tl.int64)
        copied = tl.maximum(0, tl.minimum(copied, 151551))
        tl.store(
            out_copy_tokens_ptr + req_idx * out_copy_stride + span_offs,
            copied,
            mask=span_mask,
        )


@triton.jit
def _dynamic_draft_switch_kernel(
    draft_tokens_ptr,        # [max_num_reqs, num_speculative_steps] int64 (in)
    draft_stride,            # int64
    out_draft_ptr,           # [max_num_reqs, copy_draft_span] int64 (out)
    out_draft_stride,        # int64
    copy_tokens_ptr,         # [max_num_reqs, DRAFT_SPAN] int64
    copy_stride,             # int64
    match_len_ptr,           # [max_num_reqs] int32
    num_spec_steps: tl.constexpr,
    copy_draft_span: tl.constexpr,
    MIN_MATCH: tl.constexpr,
    UNCOND_MATCH: tl.constexpr,
    BLOCK_STEPS: tl.constexpr,
):
    req_idx = tl.program_id(0)
    packed = tl.load(match_len_ptr + req_idx)
    eff_len = packed & 0xFFFF
    keep_d0 = (packed >> 16) & 1
    offs = tl.arange(0, BLOCK_STEPS)
    d_mask = offs < num_spec_steps
    d_toks = tl.load(draft_tokens_ptr + req_idx * draft_stride + offs, mask=d_mask, other=0)
    if eff_len >= MIN_MATCH:
        c_mask = offs < copy_draft_span
        c_toks = tl.load(copy_tokens_ptr + req_idx * copy_stride + offs, mask=c_mask, other=0)
        merged = tl.where((offs < keep_d0) & d_mask, d_toks, c_toks)
        tl.store(out_draft_ptr + req_idx * out_draft_stride + offs, merged, mask=c_mask)
        tl.store(draft_tokens_ptr + req_idx * draft_stride + offs, merged, mask=d_mask)
    else:
        c_mask = offs < copy_draft_span
        merged = tl.where(d_mask, d_toks, 0)
        tl.store(out_draft_ptr + req_idx * out_draft_stride + offs, merged, mask=c_mask)


class SuffixCopyDrafter:
    """
    GPU-resident Suffix / N-Gram Copy-Drafter with Dynamic Draft Switching.
    Extracts a copy-draft sequence whenever >= 4 prefix/lookahead tokens
    match in the prompt, recent output, or cross-request history, using DFlash2's
    initial draft tokens (d0, d1, d2) for branch disambiguation, 1-token variable
    substitution, and lookahead re-locking, plus an unconditional pre-DFlash2
    fast-path when >= 16 suffix tokens match exactly.
    """

    def __init__(
        self,
        max_num_reqs: int,
        num_speculative_steps: int,
        device: torch.device,
        min_match_len: int = MIN_PREFIX_MATCH_LEN,
        uncond_match_len: int = 16,
        copy_draft_span: int = COPY_DRAFT_SPAN,
        recent_window: int = 16384,
        prompt_window: int = 8192,
    ) -> None:
        self.max_num_reqs = max_num_reqs
        self.num_speculative_steps = num_speculative_steps
        self.device = device
        self.min_match_len = int(os.environ.get("SPARK_COPY_DRAFT_MIN_MATCH", str(min_match_len)))
        self.uncond_match_len = int(os.environ.get("SPARK_COPY_DRAFT_UNCOND_MATCH", str(uncond_match_len)))
        self.copy_draft_span = max(
            int(os.environ.get("SPARK_COPY_DRAFT_SPAN", str(copy_draft_span))),
            num_speculative_steps,
        )
        self.recent_window = int(os.environ.get("SPARK_COPY_DRAFT_RECENT_WINDOW", str(recent_window)))
        self.prompt_window = int(os.environ.get("SPARK_COPY_DRAFT_PROMPT_WINDOW", str(prompt_window)))
        self.enabled = os.environ.get("SPARK_ENABLE_COPY_DRAFT", "1") == "1"
        self.uncond_fastpath_enabled = os.environ.get("SPARK_ENABLE_UNCOND_FASTPATH", "1") == "1"

        self._span_pow2 = triton.next_power_of_2(max(self.copy_draft_span, 16))
        self._block_steps = self._span_pow2
        self.copy_tokens_buffer = torch.zeros(
            (max_num_reqs, self._span_pow2),
            dtype=torch.int64,
            device=device,
        )
        self.extended_draft_tokens = torch.zeros(
            (max_num_reqs, self.copy_draft_span),
            dtype=torch.int64,
            device=device,
        )
        self.match_len_buffer = torch.zeros(
            (max_num_reqs,),
            dtype=torch.int32,
            device=device,
        )
        self.any_match_flag = torch.zeros(
            (1,),
            dtype=torch.int32,
            device=device,
        )
        self.uncond_match_count = torch.zeros(
            (1,),
            dtype=torch.int32,
            device=device,
        )
        self.global_history_cap = int(os.environ.get("SPARK_COPY_DRAFT_GLOBAL_CAP", "65536"))
        self.global_history_gpu = torch.zeros(
            (self.global_history_cap,),
            dtype=torch.int32,
            device=device,
        )
        self.active_global_history_len = 0
        self.global_history_write_pos = 0
        self._staging_req: tuple[int, int, int, int, int] | None = None
        self._synced_this_step = False

    @torch.inference_mode()
    def _sync_global_history(
        self,
        num_reqs: int,
        idx_mapping: torch.Tensor,
        idx_mapping_np: Any,
        all_token_ids_gpu: torch.Tensor,
        total_len_gpu: torch.Tensor,
        prefill_len_np: Any,
    ) -> None:
        if self.global_history_cap <= 0 or idx_mapping_np is None or prefill_len_np is None or num_reqs <= 0:
            return
        try:
            s_idx = int(idx_mapping_np[0])
            p_len = int(prefill_len_np[0])
            t_len = int(total_len_gpu[s_idx].item())
        except Exception:
            return
        if t_len <= 0:
            return
        cap = self.global_history_cap
        if self._staging_req is not None:
            st_s_idx, st_p_len, st_num_reqs, st_hist_start, st_last_t_len = self._staging_req
            is_new_req = (
                s_idx != st_s_idx
                or p_len != st_p_len
                or num_reqs != st_num_reqs
                or t_len < st_last_t_len
                or (t_len <= p_len + 1 and st_last_t_len > p_len)
            )
            if is_new_req:
                if st_last_t_len > 0:
                    self.active_global_history_len = min(cap, st_hist_start + st_last_t_len)
                    self.global_history_write_pos = self.active_global_history_len
                if self.global_history_write_pos + 8192 > cap:
                    st_hist_start = 0
                else:
                    st_hist_start = self.global_history_write_pos
                st_last_t_len = 0
        else:
            st_hist_start = self.global_history_write_pos
            st_last_t_len = 0

        if t_len > st_last_t_len and st_hist_start + t_len <= cap:
            self.global_history_gpu[
                st_hist_start + st_last_t_len : st_hist_start + t_len
            ].copy_(all_token_ids_gpu[s_idx, st_last_t_len:t_len])
            st_last_t_len = t_len
        self._staging_req = (s_idx, p_len, num_reqs, st_hist_start, st_last_t_len)

    def commit_staged_history(self) -> None:
        if self._staging_req is not None:
            _, _, _, st_hist_start, st_last_t_len = self._staging_req
            if st_last_t_len > 0:
                self.active_global_history_len = min(
                    self.global_history_cap, st_hist_start + st_last_t_len
                )
                self.global_history_write_pos = self.active_global_history_len
            self._staging_req = None

    @torch.inference_mode()
    def try_unconditional_propose(
        self,
        num_reqs: int,
        idx_mapping: torch.Tensor,
        all_token_ids_gpu: torch.Tensor,
        total_len_gpu: torch.Tensor,
        draft_tokens: torch.Tensor,
        idx_mapping_np: Any = None,
        prefill_len_np: Any = None,
    ) -> torch.Tensor | None:
        if not self.enabled or not self.uncond_fastpath_enabled or num_reqs <= 0:
            self._synced_this_step = False
            return None
        self._sync_global_history(
            num_reqs,
            idx_mapping,
            idx_mapping_np,
            all_token_ids_gpu,
            total_len_gpu,
            prefill_len_np,
        )
        self._synced_this_step = True
        self.uncond_match_count.zero_()
        _unconditional_copy_draft_kernel[(num_reqs,)](
            all_token_ids_gpu,
            all_token_ids_gpu.stride(0),
            total_len_gpu,
            idx_mapping,
            self.global_history_gpu,
            self.active_global_history_len,
            draft_tokens,
            draft_tokens.stride(0),
            self.uncond_match_count,
            self.recent_window,
            self.prompt_window,
            UNCOND_MIN_MATCH=self.uncond_match_len,
            MAX_MATCH=MAX_PREFIX_MATCH_LEN,
            NUM_SPEC_STEPS=self.num_speculative_steps,
            SPAN_POW2=self._span_pow2,
            BLOCK_SIZE=256,
            num_warps=4,
        )
        if int(self.uncond_match_count.item()) == num_reqs:
            self._synced_this_step = False
            return draft_tokens[:num_reqs]
        return None

    @torch.inference_mode()
    def propose_and_override(
        self,
        num_reqs: int,
        idx_mapping: torch.Tensor,
        all_token_ids_gpu: torch.Tensor,
        total_len_gpu: torch.Tensor,
        draft_tokens: torch.Tensor,
        idx_mapping_np: Any = None,
        total_len_np: Any = None,
        prefill_len_np: Any = None,
    ) -> torch.Tensor:
        if not self.enabled or num_reqs <= 0:
            return draft_tokens[:num_reqs]

        if not self._synced_this_step:
            self._sync_global_history(
                num_reqs,
                idx_mapping,
                idx_mapping_np,
                all_token_ids_gpu,
                total_len_gpu,
                prefill_len_np,
            )
        self._synced_this_step = False

        self.any_match_flag.zero_()
        _suffix_copy_draft_kernel[(num_reqs,)](
            all_token_ids_gpu,
            all_token_ids_gpu.stride(0),
            total_len_gpu,
            idx_mapping,
            draft_tokens,
            draft_tokens.stride(0),
            self.global_history_gpu,
            self.active_global_history_len,
            self.copy_tokens_buffer,
            self.copy_tokens_buffer.stride(0),
            self.match_len_buffer,
            self.any_match_flag,
            self.recent_window,
            self.prompt_window,
            MIN_MATCH=self.min_match_len,
            MAX_MATCH=MAX_PREFIX_MATCH_LEN,
            DRAFT_SPAN=self._span_pow2,
            BLOCK_SIZE=256,
            num_warps=4,
        )

        _dynamic_draft_switch_kernel[(num_reqs,)](
            draft_tokens,
            draft_tokens.stride(0),
            self.extended_draft_tokens,
            self.extended_draft_tokens.stride(0),
            self.copy_tokens_buffer,
            self.copy_tokens_buffer.stride(0),
            self.match_len_buffer,
            num_spec_steps=self.num_speculative_steps,
            copy_draft_span=self.copy_draft_span,
            MIN_MATCH=self.min_match_len,
            UNCOND_MATCH=self.uncond_match_len,
            BLOCK_STEPS=self._block_steps,
            num_warps=1,
        )

        if (
            self.copy_draft_span > self.num_speculative_steps
            and num_reqs == 1
            and int(self.any_match_flag.item()) != 0
        ):
            return self.extended_draft_tokens[:num_reqs]
        return draft_tokens[:num_reqs]


@triton.jit
def _fast_greedy_local_argmax_kernel(
    logits_ptr,
    logits_stride,
    local_max_ptr,
    local_max_stride,
    local_argmax_ptr,
    local_argmax_stride,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    block_idx = tl.program_id(1)
    offs = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < vocab_size
    vals = tl.load(
        logits_ptr + token_idx * logits_stride + offs,
        mask=mask,
        other=-float("inf"),
    ).to(tl.float32)
    l_max, l_idx = tl.max(vals, axis=0, return_indices=True)
    tl.store(local_max_ptr + token_idx * local_max_stride + block_idx, l_max)
    tl.store(
        local_argmax_ptr + token_idx * local_argmax_stride + block_idx,
        (block_idx * BLOCK_SIZE + l_idx).to(tl.int32),
    )


@triton.jit
def _fast_greedy_verify_kernel(
    sampled_ptr,
    sampled_stride,
    num_sampled_ptr,
    local_max_ptr,
    local_max_stride,
    local_argmax_ptr,
    local_argmax_stride,
    draft_sampled_ptr,
    cu_num_logits_ptr,
    vocab_num_blocks,
    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
):
    req_idx = tl.program_id(0)
    start_idx = tl.load(cu_num_logits_ptr + req_idx).to(tl.int64)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1).to(tl.int64)
    num_req_tokens = (end_idx - start_idx).to(tl.int32)
    num_draft_tokens = num_req_tokens - 1

    blk_offs = tl.arange(0, PADDED_VOCAB_NUM_BLOCKS)
    blk_mask = blk_offs < vocab_num_blocks

    verifying = 1
    accepted_length = 0
    for i in range(num_req_tokens):
        if verifying != 0:
            logit_idx = start_idx + i
            l_max = tl.load(
                local_max_ptr + logit_idx * local_max_stride + blk_offs,
                mask=blk_mask,
                other=-float("inf"),
            )
            best_blk = tl.argmax(l_max, axis=0)
            target_argmax = tl.load(
                local_argmax_ptr + logit_idx * local_argmax_stride + best_blk
            ).to(tl.int64)
            tl.store(sampled_ptr + req_idx * sampled_stride + i, target_argmax)
            if i < num_draft_tokens:
                draft_tok = tl.load(draft_sampled_ptr + logit_idx + 1).to(tl.int64)
                accepted = (draft_tok >= 0) & (target_argmax == draft_tok)
                verifying = tl.where(accepted, 1, 0)
                accepted_length += verifying

    tl.store(num_sampled_ptr + req_idx, accepted_length + 1)


_GREEDY_WS_CACHE: dict[tuple[int, int, int], tuple[torch.Tensor, torch.Tensor]] = {}


def fast_greedy_rejection_sample(
    target_logits: torch.Tensor,
    draft_sampled: torch.Tensor,
    cu_num_logits: torch.Tensor,
    num_speculative_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    2-kernel greedy speculative verification fast-path (replaces 4 kernels +
    7 tensor allocations + duplicate bonus-row DRAM read in stock rejection_sample).
    """
    num_logits, vocab_size = target_logits.shape
    num_reqs = cu_num_logits.shape[0] - 1
    BLOCK_SIZE = 1024
    vocab_num_blocks = triton.cdiv(vocab_size, BLOCK_SIZE)
    padded_vocab_num_blocks = triton.next_power_of_2(vocab_num_blocks)

    dev_idx = target_logits.device.index or 0
    cap_logits = max(128, triton.next_power_of_2(num_logits))
    ws_key = (dev_idx, cap_logits, vocab_num_blocks)
    ws = _GREEDY_WS_CACHE.get(ws_key)
    if ws is None:
        ws = (
            torch.empty(
                (cap_logits, vocab_num_blocks),
                dtype=torch.float32,
                device=target_logits.device,
            ),
            torch.empty(
                (cap_logits, vocab_num_blocks),
                dtype=torch.int32,
                device=target_logits.device,
            ),
        )
        _GREEDY_WS_CACHE[ws_key] = ws
    local_max, local_argmax = ws

    _fast_greedy_local_argmax_kernel[(num_logits, vocab_num_blocks)](
        target_logits,
        target_logits.stride(0),
        local_max,
        local_max.stride(0),
        local_argmax,
        local_argmax.stride(0),
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )

    sampled = draft_sampled.new_empty(
        (num_reqs, num_speculative_steps + 1), dtype=torch.int64
    )
    num_sampled = sampled.new_empty((num_reqs,), dtype=torch.int32)
    _fast_greedy_verify_kernel[(num_reqs,)](
        sampled,
        sampled.stride(0),
        num_sampled,
        local_max,
        local_max.stride(0),
        local_argmax,
        local_argmax.stride(0),
        draft_sampled,
        cu_num_logits,
        vocab_num_blocks,
        PADDED_VOCAB_NUM_BLOCKS=padded_vocab_num_blocks,
        num_warps=1,
    )
    return sampled, num_sampled


@triton.jit
def _fast_greedy_verify_top_tokens_kernel(
    sampled_ptr,
    sampled_stride,
    num_sampled_ptr,
    top_tokens_ptr,
    draft_sampled_ptr,
    cu_num_logits_ptr,
):
    req_idx = tl.program_id(0)
    start_idx = tl.load(cu_num_logits_ptr + req_idx).to(tl.int64)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1).to(tl.int64)
    num_req_tokens = (end_idx - start_idx).to(tl.int32)
    num_draft_tokens = num_req_tokens - 1

    verifying = 1
    accepted_length = 0
    for i in range(num_req_tokens):
        if verifying != 0:
            logit_idx = start_idx + i
            target_argmax = tl.load(top_tokens_ptr + logit_idx).to(tl.int64)
            tl.store(sampled_ptr + req_idx * sampled_stride + i, target_argmax)
            if i < num_draft_tokens:
                draft_tok = tl.load(draft_sampled_ptr + logit_idx + 1).to(tl.int64)
                accepted = (draft_tok >= 0) & (target_argmax == draft_tok)
                verifying = tl.where(accepted, 1, 0)
                accepted_length += verifying

    tl.store(num_sampled_ptr + req_idx, accepted_length + 1)


def fast_greedy_verify_top_tokens(
    top_tokens: torch.Tensor,
    draft_sampled: torch.Tensor,
    cu_num_logits: torch.Tensor,
    num_speculative_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Single-kernel greedy speculative verification directly from vocab-parallel
    `top_tokens` (`[num_logits]` int64), avoiding any full-vocab logits all-gather.
    """
    num_reqs = cu_num_logits.shape[0] - 1
    sampled = draft_sampled.new_empty(
        (num_reqs, num_speculative_steps + 1), dtype=torch.int64
    )
    num_sampled = sampled.new_empty((num_reqs,), dtype=torch.int32)
    _fast_greedy_verify_top_tokens_kernel[(num_reqs,)](
        sampled,
        sampled.stride(0),
        num_sampled,
        top_tokens,
        draft_sampled,
        cu_num_logits,
        num_warps=1,
    )
    return sampled, num_sampled


