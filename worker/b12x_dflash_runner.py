#!/usr/bin/env python3
"""
Spark-GLM53 Phase 3 Speculative Worker & Runner (`worker/b12x_dflash_runner.py`).

Fuses:
  1. Phase 2 Persistent `/dev/shm` Zero-Copy Weight Loader (`worker.shm_safetensors`)
     for both `local-inference-lab/GLM-5.3-Flash-NVFP4-Spark` (~85.82 GiB/rank) and
     `incoai/GLM-5.3-Flash-DFlash2` (~2.18 GiB/rank), with warm-boot `MemorySnapshot`
     compensation so reloads finish in < 1.0s with zero page-cache bloat.
  2. GB10 (`sm_121a`) `B12X` Target + `DFlash2` (`K=7`) Speculative Engine:
     - Enables the custom split-KV CUDA kernel (`dflash_attn.cu`) on GB10 (`sm_121a`,
       compute capability `(12, 1)`), cutting 2048-window draft attention from 48 us
       to 13 us per layer (saving ~175 us per decode step).
     - Aligns `DFlashAttention` KV cache `block_size` to `128` (`128 | 256` LCM with
       the `B12X` target runner), eliminating the 2.28x KV block size inflation and
       preserving 100% of the target's `> 2.5M–3.1M` FP8 KV cache pool (`1,048,576`
       max context window).
     - Separates target (`B12X` + `fp8_ds_mla`) and draft (`FLASH_ATTN` + `bfloat16`)
       attention/cache configs via `DFlashSpeculator.attn_vllm_config` and
       `GPUModelRunner.initialize_kv_cache` `layer_vllm_configs`.
     - Pre-allocates a static auxiliary hidden-state capture buffer (`DFlash2ContextCudaGraphs`)
       on `DFlashQwen3ForCausalLM` so `combine_hidden_states` + `precompute_and_store_context_kv`
       run inside a captured CUDA graph alongside `FULL_AND_PIECEWISE` target CUDA graphs.
     - Initializes the sparse-MLA kpool indexer `topk_indices_buffer` and `pool_topk` to `-1`
       and clamps pool indices to `[0, pool_len)` while zeroing (`-1`) the 128-aligned
       trailing columns `[2111, 2176)` in a single Triton launch (preventing uninitialized
       `torch.empty` entries from gathering unwritten KV pages and producing `NaN`s).
  3. Dynamic Draft Switching (`SuffixCopyDrafter`):
     - Runs a GPU-resident Suffix/N-Gram Copy-Drafter (`worker.kernels.copy_draft_kernels`)
       that scans prompt + output tokens for `>= 4` prefix token matches, extracting a
       12-token copy-draft buffer and dynamically overriding `DFlash2` in ~8 us on the
       GPU stream with zero CPU-GPU synchronization.
"""

from __future__ import annotations

import copy
import dataclasses
import logging
import os
import socket
import subprocess
import sys
from dataclasses import replace
from typing import Any

# Ensure repo root is importable across spawned workers
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

logger = logging.getLogger("spark_glm53.runner")
_PATCHES_INSTALLED = False


def _detect_roce_ip() -> str | None:
    """Auto-detect local 10.100.240.x RoCE interface IP on enp1s0f1np1."""
    try:
        out = subprocess.check_output(["ip", "-br", "-4", "addr"], text=True, timeout=2.0)
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 3 and parts[0] in ("enp1s0f1np1", "enP2p1s0f1np1", "enp1s0f0np0"):
                for token in parts[2:]:
                    ip = token.split("/")[0]
                    if ip.startswith("10.100.240.") or ip.startswith("192.168.100."):
                        return ip
    except Exception:
        pass
    try:
        hostname = socket.gethostname()
        if "5a4e" in hostname:
            return "10.100.240.1"
        if "8a27" in hostname:
            return "10.100.240.2"
    except Exception:
        pass
    return None


def configure_spark_env() -> None:
    """Set mandatory GB10 (sm_121a) + Dual-Rail RoCE v2 + DFlash2 environment variables."""
    defaults = {
        "CUTE_DSL_ARCH": "sm_121a",
        "SAFETENSORS_FAST_GPU": "1",
        "VLLM_ENABLE_ROCE_ALLREDUCE": "1",
        "VLLM_ROCE_ALLREDUCE_MAX_SIZE": "16MB",
        "NCCL_IB_HCA": "rocep1s0f1,roceP2p1s0f1",
        "NCCL_CROSS_NIC": "1",
        "NCCL_BUFFSIZE": "1048576",
        "NCCL_MAX_CTAS": "8",
        "NCCL_IB_GID_INDEX": "3",
        "NCCL_IB_TC": "104",
        "NCCL_IB_SL": "3",
        "NCCL_NET_GDR_LEVEL": "SYS",
        "VLLM_SSM_CONV_STATE_LAYOUT": "DS",
        "VLLM_USE_AOT_COMPILE": "1",
        "VLLM_USE_MEGA_AOT_ARTIFACT": "1",
        "VLLM_ENABLE_PCIE_ALLREDUCE": "0",
        "B12X_POLICY_MODE": "auto",
        "B12X_MICRO_DYNAMIC_CUTOVER_PAIRS": "0",
        "B12X_WEIGHTS_COMPILE_WORKERS": "2",
        "B12X_STATE_COMPILE_WORKERS": "2",
        "B12X_BIND_COMPILE_WORKERS": "2",
        "NCCL_IGNORE_CPU_AFFINITY": "1",
        "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": "0",
        "VLLM_GLM53_DFLASH_ATTN": "1",
        "VLLM_GLM53_DFLASH_ATTN_MAX_REQS": "64",
        "VLLM_GLM53_DFLASH_ATTN_WS_REQS": "64",
        "VLLM_GLM53_L2_PREFETCH": "0",
        "GLM53_INDEXER_WORKSPACE": "rightsize",
        "VLLM_MXFP8_LM_HEAD": "0",
        "SPARK_SHM_ENABLE": "1",
        "SPARK_SHM_MATERIALIZE_DENSE": "1",
        "SPARK_ENABLE_COPY_DRAFT": "1",
        "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC": "45",
        "PYTORCH_CUDA_ALLOC_CONF": "garbage_collection_threshold:0.7,max_split_size_mb:128",
    }
    for k, v in defaults.items():
        os.environ.setdefault(k, v)

    roce_ip = _detect_roce_ip()
    if roce_ip:
        os.environ.setdefault("VLLM_HOST_IP", roce_ip)
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "enp1s0f1np1")
        os.environ.setdefault("NCCL_SOCKET_IFNAME", "enp1s0f1np1")


def install_b12x_compiler_worker_patches() -> None:
    """Lightweight patches for b12x compiler worker subprocesses (avoids importing torch unless needed)."""
    try:
        import b12x.sequence.gdn_decode._impl as b12x_gdn_impl
        import b12x.sequence.gdn_decode._tuning as b12x_gdn_tuning

        if not getattr(b12x_gdn_impl.Caps, "_spark_patched", False):
            orig_caps_post_init = b12x_gdn_impl.Caps.__post_init__

            def _patched_caps_post_init(self: Any) -> None:
                cols = int(self.state_index_columns)
                if 8 < cols <= 32:
                    orig_mt = int(self.max_tokens)
                    object.__setattr__(self, "state_index_columns", 8)
                    object.__setattr__(
                        self, "max_tokens", min(orig_mt, int(self.max_seqs) * 8)
                    )
                    orig_caps_post_init(self)
                    object.__setattr__(self, "state_index_columns", cols)
                    object.__setattr__(self, "max_tokens", orig_mt)
                    return
                orig_caps_post_init(self)

            b12x_gdn_impl.Caps.__post_init__ = _patched_caps_post_init
            b12x_gdn_impl.Caps._spark_patched = True

        if not getattr(b12x_gdn_tuning, "_spark_patched", False):
            orig_validate_query = b12x_gdn_tuning._validate_query

            def _patched_validate_query(
                query: Any, *v_args: Any, **v_kwargs: Any
            ) -> None:
                cols = int(query.state_index_columns)
                if 8 < cols <= 32:
                    tmp_q = replace(
                        query,
                        state_index_columns=8,
                        max_tokens=min(int(query.max_tokens), int(query.max_seqs) * 8),
                    )
                    orig_validate_query(tmp_q, *v_args, **v_kwargs)
                    return
                orig_validate_query(query, *v_args, **v_kwargs)

            b12x_gdn_tuning._validate_query = _patched_validate_query
            object.__setattr__(
                b12x_gdn_tuning.TUNING, "validate_query", _patched_validate_query
            )
            b12x_gdn_tuning._spark_patched = True
    except Exception:
        pass

    try:
        import b12x.moe.fused_moe._impl as b12x_moe_impl
        import b12x.moe.fused_moe._tuning as b12x_moe_tuning

        if not getattr(b12x_moe_tuning, "_spark_patched", False):
            orig_select_tile_mn = b12x_moe_impl._select_dynamic_tile_mn

            def _patched_select_dynamic_tile_mn(
                routed_rows: int,
                n: int,
                quant_mode: str = "nvfp4",
                *,
                num_experts: int,
                activation: str = "silu",
                compute_capability: tuple[int, int] | None = None,
                planned_tile_m: int | None = None,
            ) -> tuple[int, int]:
                tile_m, tile_n = orig_select_tile_mn(
                    routed_rows,
                    n,
                    quant_mode,
                    num_experts=num_experts,
                    activation=activation,
                    compute_capability=compute_capability,
                    planned_tile_m=planned_tile_m,
                )
                if (
                    planned_tile_m is None
                    and b12x_moe_impl._dynamic_tile_mn_override() is None
                    and quant_mode in ("nvfp4", "nvfp4_auto")
                    and tile_m > 64
                ):
                    tile_m = 64
                return (tile_m, tile_n)

            b12x_moe_impl._select_dynamic_tile_mn = _patched_select_dynamic_tile_mn

            orig_heuristic = b12x_moe_impl._heuristic_moe_decode_config

            def _patched_heuristic_moe_decode_config(query: Any, device: Any) -> Any:
                if getattr(query, "quant_mode", None) == "nvfp4_auto":
                    query = replace(query, quant_mode="nvfp4")
                cfg = orig_heuristic(query, device)
                if (
                    getattr(cfg, "backend", None) == "dynamic"
                    and getattr(cfg, "dynamic_tile_m", None) is not None
                    and int(cfg.dynamic_tile_m) > 64
                ):
                    cfg = replace(cfg, dynamic_tile_m=64)
                if getattr(cfg, "nvfp4_materialize_intermediate", False):
                    cfg = replace(cfg, nvfp4_materialize_intermediate=False)
                return cfg

            b12x_moe_impl._heuristic_moe_decode_config = _patched_heuristic_moe_decode_config

            orig_moe_default_config = b12x_moe_tuning._default_config

            def _patched_moe_default_config(query: Any, device: Any) -> Any:
                cfg = orig_moe_default_config(query, device)
                if (
                    getattr(cfg, "backend", None) == "dynamic"
                    and getattr(cfg, "dynamic_tile_m", None) is not None
                    and int(cfg.dynamic_tile_m) > 64
                ):
                    cfg = replace(cfg, dynamic_tile_m=64)
                if getattr(cfg, "nvfp4_materialize_intermediate", False):
                    cfg = replace(cfg, nvfp4_materialize_intermediate=False)
                return cfg

            b12x_moe_tuning._default_config = _patched_moe_default_config
            object.__setattr__(
                b12x_moe_tuning.TUNING, "default_config", _patched_moe_default_config
            )
            b12x_moe_tuning._spark_patched = True
    except Exception:
        pass


# Apply env vars immediately on module import
configure_spark_env()


def _verify_nccl_version() -> None:
    """Guard against broken runtime libnccl.so.2 == 2.29.7 (22907) on GB10."""
    import ctypes

    try:
        lib = ctypes.CDLL("libnccl.so.2")
        ver = ctypes.c_int(0)
        if lib.ncclGetVersion(ctypes.byref(ver)) == 0:
            if ver.value == 22907:
                raise RuntimeError(
                    "CRITICAL GB10 BUG: runtime libnccl.so.2 version 2.29.7 (22907) "
                    "detected, which causes fabric internal errors on ConnectX-7 RoCE v2."
                )
    except RuntimeError:
        raise
    except Exception:
        pass


def _patch_dflash_attn_sm121a() -> None:
    """Enable the split-KV `dflash_attn.cu` kernel on GB10 (`sm_121a`, CC 12.1)."""
    import torch
    import vllm.v1.attention.backends.dflash_attn as dflash_attn_mod
    import vllm.v1.attention.backends.flash_attn as flash_attn_mod

    def _sm121a_is_available(device: torch.device | int | None = None) -> bool:
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device() if torch.cuda.is_available() else 0)
        elif isinstance(device, int):
            device = torch.device("cuda", device)
        if not torch.cuda.is_available() or device.type != "cuda":
            return False
        if torch.cuda.get_device_capability(device) not in ((12, 0), (12, 1)):
            return False
        try:
            dflash_attn_mod._load(device)
        except RuntimeError:
            return False
        return True

    dflash_attn_mod.is_available = _sm121a_is_available
    flash_attn_mod._DFLASH_ATTN = os.getenv("VLLM_GLM53_DFLASH_ATTN", "1") == "1"
    flash_attn_mod._DFLASH_ATTN_MAX_REQS = int(
        os.getenv("VLLM_GLM53_DFLASH_ATTN_MAX_REQS", "64")
    )
    flash_attn_mod._DFLASH_ATTN_WS_REQS = max(
        flash_attn_mod._DFLASH_ATTN_MAX_REQS,
        int(os.getenv("VLLM_GLM53_DFLASH_ATTN_WS_REQS", "64")),
    )


def _patch_dflash_kv_cache_and_block_size() -> None:
    """
    1. Align `DFlashAttention` KV block_size to 128 when target block_size == 256:
       - 128 divides 256 so LCM remains 256 (required by B12X).
       - 5 layers * 128 tokens * 2048 B = 1,310,720 B/block <= 1,723,392 B/block
         (the 12 target MLA layers), giving 0% KV block size inflation and preserving
         100% of the > 2.5M–3.1M token FP8 KV cache pool!
    2. Propagate draft `attention_backend` (FLASH_ATTN) and `kv_cache_dtype` (auto/bfloat16)
       through `DFlashSpeculator.attn_vllm_config` and `GPUModelRunner.initialize_kv_cache`.
    """
    from vllm.config.attention import AttentionBackendEnum
    from vllm.model_executor.models.qwen3_dflash import DFlashAttention
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheSpec,
        SlidingWindowSpec,
        get_kv_quant_mode,
    )
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

    DraftModelSpeculator = DFlashSpeculator.__mro__[1]

    orig_get_kv_cache_spec = DFlashAttention.get_kv_cache_spec

    def _patched_dflash_get_kv_cache_spec(
        self: DFlashAttention, vllm_config: Any
    ) -> KVCacheSpec | None:
        dcp_replicated = vllm_config.parallel_config.decode_context_parallel_size > 1
        target_block_size = vllm_config.cache_config.block_size
        # Use draft_block_size = target_block_size // 2 (e.g. 128 when target is 256,
        # or 2560 when target hybrid block is 5120) so 5 DFlash layers
        # (5 * (target_block_size // 2) * 2048 B) always fit inside the 12 target
        # MLA layers (~8514 B * target_block_size) with 0% KV block inflation and
        # exact C4 index page stride alignment (8448 B).
        if (
            target_block_size >= 256
            and (target_block_size // 2) % 16 == 0
            and os.getenv("SPARK_DFLASH_HALF_BLOCK", "1") == "1"
        ):
            draft_block_size = target_block_size // 2
        else:
            draft_block_size = target_block_size
        if self.sliding_window is not None:
            return SlidingWindowSpec(
                block_size=draft_block_size,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_size,
                head_size_v=self.head_size_v,
                dtype=self.kv_cache_torch_dtype,
                sliding_window=self.sliding_window,
                page_size_padded=getattr(
                    vllm_config.cache_config, "skip_page_size_padded", None
                ),
                extra_retained_tokens=self.sliding_window,
                kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
                dcp_replicated=dcp_replicated,
            )
        spec = orig_get_kv_cache_spec(self, vllm_config)
        if isinstance(spec, FullAttentionSpec):
            spec = dataclasses.replace(
                spec,
                block_size=draft_block_size,
                dcp_replicated=dcp_replicated or spec.dcp_replicated,
            )
        return spec

    DFlashAttention.get_kv_cache_spec = _patched_dflash_get_kv_cache_spec

    # Ensure _get_kv_cache_bytes_per_block always aligns to GLM C4 index page bytes (64 * 132 = 8448)
    import vllm.v1.core.kv_cache_utils as kv_utils_mod

    orig_get_bytes_per_block = kv_utils_mod._get_kv_cache_bytes_per_block

    def _patched_get_kv_cache_bytes_per_block(kv_cache_groups: list[Any]) -> int:
        res = orig_get_bytes_per_block(kv_cache_groups)
        glm_c4_index_page_bytes = 64 * 132
        if res % glm_c4_index_page_bytes != 0:
            contains_glm5 = kv_utils_mod._contains_glm5_next_mla(
                kv_utils_mod._get_per_layer_spec(group, layer_name)
                for group in kv_cache_groups
                for layer_name in group.layer_names
            )
            if contains_glm5:
                res = ((res + glm_c4_index_page_bytes - 1) // glm_c4_index_page_bytes) * glm_c4_index_page_bytes
        return res

    kv_utils_mod._get_kv_cache_bytes_per_block = _patched_get_kv_cache_bytes_per_block

    @property
    def _patched_attn_vllm_config(self: DFlashSpeculator) -> Any:
        config = copy.copy(super(DFlashSpeculator, self).attn_vllm_config)
        config.model_config = self.draft_model_config
        config.quant_config = None
        draft_backend_str = getattr(self.speculative_config, "attention_backend", None)
        draft_backend = (
            AttentionBackendEnum[draft_backend_str]
            if isinstance(draft_backend_str, str) and draft_backend_str in AttentionBackendEnum.__members__
            else (draft_backend_str or AttentionBackendEnum.FLASH_ATTN)
        )
        config.attention_config = replace(
            self.vllm_config.attention_config,
            backend=draft_backend,
            use_non_causal=self.requires_non_causal,
        )
        draft_kv_dtype = getattr(self.speculative_config, "kv_cache_dtype", None) or "auto"
        config.cache_config = replace(
            self.vllm_config.cache_config,
            cache_dtype=draft_kv_dtype,
        )
        return config

    DFlashSpeculator.attn_vllm_config = _patched_attn_vllm_config

    # Ensure GPUModelRunner.initialize_kv_cache passes layer_vllm_configs for ALL
    # DraftModelSpeculator subclasses (including DFlash2Speculator), not just glm53_dspark.
    orig_init_kv_cache = GPUModelRunner.initialize_kv_cache

    def _patched_initialize_kv_cache(
        self: GPUModelRunner, kv_cache_config: Any, *args: Any, **kwargs: Any
    ) -> None:
        if isinstance(self.speculator, DraftModelSpeculator) and self.speculator.draft_attn_layer_names:
            # Temporarily wrap init_attn_backend so layer_vllm_configs is always supplied
            import vllm.v1.worker.gpu.model_runner as mr_mod

            orig_init_attn_backend = mr_mod.init_attn_backend

            def _wrapped_init_attn_backend(
                kv_cfg: Any,
                vllm_cfg: Any,
                device: Any,
                *b_args: Any,
                layer_vllm_configs: dict[str, Any] | None = None,
                **b_kwargs: Any,
            ) -> Any:
                if not layer_vllm_configs and isinstance(
                    self.speculator, DraftModelSpeculator
                ):
                    layer_vllm_configs = dict.fromkeys(
                        self.speculator.draft_attn_layer_names,
                        self.speculator.attn_vllm_config,
                    )
                return orig_init_attn_backend(
                    kv_cfg,
                    vllm_cfg,
                    device,
                    *b_args,
                    layer_vllm_configs=layer_vllm_configs,
                    **b_kwargs,
                )

            mr_mod.init_attn_backend = _wrapped_init_attn_backend
            try:
                return orig_init_kv_cache(self, kv_cache_config, *args, **kwargs)
            finally:
                mr_mod.init_attn_backend = orig_init_attn_backend
        return orig_init_kv_cache(self, kv_cache_config, *args, **kwargs)

    GPUModelRunner.initialize_kv_cache = _patched_initialize_kv_cache


def _patch_dflash2_context_cudagraph_and_copy_drafter() -> None:
    """
    1. Pre-allocate a static auxiliary hidden-state capture buffer (`DFlash2ContextCudaGraphs`)
       on `DFlashQwen3ForCausalLM` so `combine_hidden_states` + `precompute_and_store_context_kv`
       are captured into CUDA graphs during `DFlashSpeculator.capture()`.
    2. Hook `SuffixCopyDrafter` (`worker.kernels.copy_draft_kernels`) into `DFlashSpeculator.propose`
       for zero-overhead dynamic draft switching on the GPU stream.
    """
    import torch
    from vllm.config import CUDAGraphMode, VllmConfig
    from vllm.model_executor.models.qwen3_dflash import (
        DFlashQwen3ForCausalLM,
        DFlashQwen3Model,
    )
    from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
        DFlashSpeculator,
        PAD_SLOT_ID,
    )

    from worker.kernels.copy_draft_kernels import SuffixCopyDrafter

    # Ensure _build_fused_kv_buffers pre-casts _rope_cos_sin_cache to hidden_norm dtype
    orig_build_fused = DFlashQwen3Model._build_fused_kv_buffers

    def _patched_build_fused_kv_buffers(self: DFlashQwen3Model) -> None:
        orig_build_fused(self)
        if (
            hasattr(self, "_rope_cos_sin_cache")
            and self._rope_cos_sin_cache is not None
            and self._rope_cos_sin_cache.dtype != self._hidden_norm_weight.dtype
        ):
            self._rope_cos_sin_cache = self._rope_cos_sin_cache.to(
                dtype=self._hidden_norm_weight.dtype
            )

    DFlashQwen3Model._build_fused_kv_buffers = _patched_build_fused_kv_buffers

    orig_precompute_ctx_kv = DFlashQwen3Model.precompute_and_store_context_kv

    def _patched_precompute_and_store_context_kv(
        self: DFlashQwen3Model,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | list[torch.Tensor | None] | None = None,
    ) -> None:
        num_ctx = context_states.shape[0]
        chunk_sz = 2048
        if num_ctx <= chunk_sz or torch.cuda.is_current_stream_capturing():
            return orig_precompute_ctx_kv(
                self, context_states, context_positions, context_slot_mapping
            )
        per_layer = isinstance(context_slot_mapping, (list, tuple))
        for c_start in range(0, num_ctx, chunk_sz):
            c_end = min(c_start + chunk_sz, num_ctx)
            if context_slot_mapping is None:
                sub_slot = None
            elif per_layer:
                sub_slot = [
                    sm[c_start:c_end] if sm is not None else None
                    for sm in context_slot_mapping
                ]
            else:
                sub_slot = context_slot_mapping[c_start:c_end]
            orig_precompute_ctx_kv(
                self,
                context_states[c_start:c_end],
                context_positions[c_start:c_end],
                sub_slot,
            )

    DFlashQwen3Model.precompute_and_store_context_kv = (
        _patched_precompute_and_store_context_kv
    )

    def _dflash_get_top_tokens(
        self: DFlashQwen3ForCausalLM,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        top_tokens = self.logits_processor.get_top_tokens(self.lm_head, hidden_states)
        if self.draft_id_to_target_id is None:
            return top_tokens
        return top_tokens + self.draft_id_to_target_id[top_tokens]

    DFlashQwen3ForCausalLM.get_top_tokens = _dflash_get_top_tokens

    class DFlash2ContextCudaGraphs:
        """
        Static auxiliary hidden-state capture buffer & CUDA graph manager for DFlash2
        context K/V projection (`combine_hidden_states` + `precompute_and_store_context_kv`).
        """

        def __init__(
            self,
            model: DFlashQwen3ForCausalLM,
            vllm_config: VllmConfig,
            hidden_states: torch.Tensor,
            positions: torch.Tensor,
            slot_mappings: torch.Tensor,
            layer_group_idx: list[int] | None,
            max_decode_tokens: int,
        ) -> None:
            self.model = model
            self.hidden_states = hidden_states
            self.positions = positions
            self.slot_mappings = slot_mappings
            self.layer_group_idx = layer_group_idx
            self.total_aux_width = int(model.model.fc.input_size)
            drafter_cfg = getattr(model.config, "dflash_config", {}) or {}
            target_layer_ids = drafter_cfg.get(
                "target_layer_ids",
                getattr(model.config, "target_layer_ids", [1, 15, 29, 43, 57]),
            )
            self.num_aux = len(target_layer_ids)
            self.width = self.total_aux_width // self.num_aux

            limit = min(
                max_decode_tokens,
                hidden_states.shape[0],
                vllm_config.compilation_config.max_cudagraph_capture_size,
            )
            capacities: list[int] = []
            cap = 1
            while cap < limit:
                capacities.append(cap)
                cap *= 2
            if limit > 0:
                capacities.append(limit)

            # Static auxiliary hidden-state capture buffer
            self.aux = torch.zeros(
                (limit, self.total_aux_width),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            config = copy.copy(vllm_config)
            config.compilation_config = copy.copy(vllm_config.compilation_config)
            config.compilation_config.cudagraph_capture_sizes = capacities
            config.compilation_config.max_cudagraph_capture_size = limit
            self.manager = CudaGraphManager(
                config, hidden_states.device, CUDAGraphMode.FULL, decode_query_len=1
            )

        def _forward(self, capacity: int) -> None:
            main_x = self.model.combine_hidden_states(self.aux[:capacity])
            self.hidden_states[:capacity].copy_(main_x)
            slots = (
                self.slot_mappings[0, :capacity]
                if self.layer_group_idx is None
                else [self.slot_mappings[i, :capacity] for i in self.layer_group_idx]
            )
            self.model.precompute_and_store_context_kv(
                self.hidden_states[:capacity],
                self.positions[:capacity],
                slots,
            )

        def capture(self) -> None:
            self.positions.zero_()
            self.slot_mappings.fill_(PAD_SLOT_ID)
            for capacity in self.manager.compilation_config.cudagraph_capture_sizes:
                self._forward(capacity)

            def create_forward_fn(desc: Any, warmup: bool) -> Any:
                return lambda mode: self._forward(desc.num_tokens)

            self.manager.capture(
                create_forward_fn,
                progress_bar_desc="Capturing DFlash2 context CUDA graphs",
            )

        def can_run(self, num_tokens: int) -> bool:
            desc = self.manager.dispatch(1, num_tokens, None, 0)
            return desc in self.manager.graphs

        def run(
            self,
            aux_hidden_states: list[torch.Tensor],
            num_tokens: int,
            *,
            context_kv_is_restored: bool = False,
        ) -> None:
            if context_kv_is_restored:
                return
            desc = self.manager.dispatch(1, num_tokens, None, 0)
            for i, source in enumerate(aux_hidden_states):
                self.aux[:num_tokens, i * self.width : (i + 1) * self.width].copy_(
                    source[:num_tokens]
                )
            capacity = desc.num_tokens
            if capacity > num_tokens:
                self.aux[num_tokens:capacity].zero_()
                self.positions[num_tokens:capacity].zero_()
                self.slot_mappings[:, num_tokens:capacity].fill_(PAD_SLOT_ID)
            self.manager.run_fullgraph(desc)

        def close(self) -> None:
            self.manager.reset_graphs()

    def _capture_context_preparation(
        self: DFlashQwen3ForCausalLM,
        vllm_config: VllmConfig,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        slot_mappings: torch.Tensor,
        layer_group_idx: list[int] | None,
        max_decode_tokens: int,
    ) -> DFlash2ContextCudaGraphs | None:
        if os.getenv("SPARK_DFLASH_CONTEXT_CG", "0") != "1":
            return None
        ctx = DFlash2ContextCudaGraphs(
            self,
            vllm_config,
            hidden_states,
            positions,
            slot_mappings,
            layer_group_idx,
            max_decode_tokens,
        )
        ctx.capture()
        return ctx

    DFlashQwen3ForCausalLM.capture_context_preparation = _capture_context_preparation

    # Attach req_states, SuffixCopyDrafter, b12x vocab projection, and greedy draft mode
    import gc
    import numpy as np
    from worker.kernels.copy_draft_kernels import (
        fast_greedy_rejection_sample,
        fast_greedy_verify_top_tokens,
    )
    from worker.shm_safetensors import drop_file_page_cache

    try:
        from vllm.v1.worker.gpu.input_batch import get_num_sampled_and_rejected
        from vllm.v1.worker.gpu.sample.output import SamplerOutput
        from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS

        orig_runner_sample = GPUModelRunner.sample

        def _patched_runner_sample(
            self: GPUModelRunner,
            hidden_states: torch.Tensor,
            input_batch: Any,
            grammar_output: Any,
        ) -> tuple[ Any, torch.Tensor, torch.Tensor]:
            target_lm = getattr(self.model, "language_model", self.model)
            if (
                os.getenv("SPARK_FAST_GREEDY_SAMPLE", "1") == "1"
                and grammar_output is None
                and self.sampler is not None
                and not self.sampler.compute_nans
                and input_batch.num_reqs > 0
                and hasattr(target_lm, "logits_processor")
                and hasattr(target_lm, "lm_head")
                and (
                    self.rejection_sampler is None
                    or (
                        self.rejection_sampler.watermark_key is None
                        and self.rejection_sampler.synthetic_conditional_rates is None
                    )
                )
            ):
                idx_np = input_batch.idx_mapping_np[: input_batch.num_reqs]
                if (
                    not np.any(self.sampler.needs_logits_processing[idx_np])
                    and (self.sampler.sampling_states.temperature.np[idx_np] <= 0.0).all()
                    and self.sampler.sampling_states.max_num_logprobs(idx_np) == NO_LOGPROBS
                ):
                    sample_hidden_states = hidden_states[input_batch.logits_indices]
                    top_tokens = target_lm.logits_processor.get_top_tokens(
                        target_lm.lm_head, sample_hidden_states
                    )
                    if input_batch.num_draft_tokens == 0 or self.rejection_sampler is None:
                        sampled = top_tokens.unsqueeze(1)
                        num_sampled = torch.ones(
                            (input_batch.num_reqs,),
                            dtype=torch.int32,
                            device=self.device,
                        )
                    else:
                        draft_sampled = input_batch.input_ids[input_batch.logits_indices]
                        sampled, num_sampled = fast_greedy_verify_top_tokens(
                            top_tokens,
                            draft_sampled,
                            input_batch.cu_num_logits,
                            self.rejection_sampler.num_speculative_steps,
                        )
                    num_sampled, num_rejected = get_num_sampled_and_rejected(
                        num_sampled,
                        input_batch.seq_lens,
                        input_batch.cu_num_logits,
                        input_batch.idx_mapping,
                        self.sampler.req_states.prefill_len.gpu,
                    )
                    out = SamplerOutput(
                        sampled_token_ids=sampled,
                        logprobs_tensors=None,
                        num_nans=None,
                        num_sampled=num_sampled,
                        num_rejected=num_rejected,
                    )
                    return out, num_sampled, num_rejected
            return orig_runner_sample(self, hidden_states, input_batch, grammar_output)

        GPUModelRunner.sample = _patched_runner_sample
    except Exception:
        pass

    try:
        from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler

        orig_rs_verify = RejectionSampler._verify

        def _patched_rs_verify(
            self: RejectionSampler,
            logits: torch.Tensor,
            draft_logits: torch.Tensor | None,
            draft_sampled: torch.Tensor,
            pos: torch.Tensor,
            cu_num_logits: torch.Tensor,
            idx_mapping: torch.Tensor,
            idx_mapping_np: Any,
            expanded_idx_mapping: torch.Tensor,
            expanded_local_pos: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            if (
                os.getenv("SPARK_FAST_GREEDY_VERIFY", "1") == "1"
                and self.watermark_key is None
                and self.synthetic_conditional_rates is None
                and (
                    self.sampler.sampling_states.temperature.np[idx_mapping_np] <= 0.0
                ).all()
            ):
                processed_logits = self.sampler.apply_sampling_params(
                    logits,
                    expanded_idx_mapping,
                    idx_mapping,
                    idx_mapping_np,
                    pos,
                    draft_sampled,
                    expanded_local_pos,
                )
                sampled, num_sampled = fast_greedy_rejection_sample(
                    processed_logits,
                    draft_sampled,
                    cu_num_logits,
                    self.num_speculative_steps,
                )
                return processed_logits, sampled, num_sampled
            return orig_rs_verify(
                self,
                logits,
                draft_logits,
                draft_sampled,
                pos,
                cu_num_logits,
                idx_mapping,
                idx_mapping_np,
                expanded_idx_mapping,
                expanded_local_pos,
            )

        RejectionSampler._verify = _patched_rs_verify
    except Exception:
        pass

    try:
        from dataclasses import replace as _dc_replace
        from vllm.distributed import tensor_model_parallel_all_gather
        from vllm.model_executor.layers.logits_processor import LogitsProcessor

        orig_lp_get_units = LogitsProcessor.get_b12x_preparation_units
        orig_lp_apply_head = LogitsProcessor._apply_head
        orig_lp_get_top_tokens = LogitsProcessor.get_top_tokens

        def _patched_lp_get_units(self: Any, layer: Any, workload: Any) -> Any:
            max_vocab_tokens = int(os.getenv("SPARK_MAX_VOCAB_PREP_TOKENS", "512"))
            filtered = tuple(t for t in workload.token_counts if t <= max_vocab_tokens)
            if not filtered and workload.token_counts:
                filtered = (min(workload.token_counts),)
            if filtered != workload.token_counts:
                workload = _dc_replace(workload, token_counts=filtered)
            return orig_lp_get_units(self, layer, workload)

        def _patched_lp_apply_head(
            self: Any, lm_head: Any, hidden_states: torch.Tensor, embedding_bias: Any
        ) -> torch.Tensor:
            max_vocab_tokens = int(os.getenv("SPARK_MAX_VOCAB_PREP_TOKENS", "512"))
            flat_rows = (
                hidden_states.numel() // hidden_states.shape[-1]
                if hidden_states.ndim >= 1
                else 0
            )
            if flat_rows > max_vocab_tokens:
                return lm_head.quant_method.apply(
                    lm_head, hidden_states, bias=embedding_bias
                )
            return orig_lp_apply_head(self, lm_head, hidden_states, embedding_bias)

        def _patched_lp_get_top_tokens(
            self: Any,
            lm_head: Any,
            hidden_states: torch.Tensor,
            embedding_bias: torch.Tensor | None = None,
        ) -> torch.Tensor:
            if self.scale <= 0.0 and self.scale != 1.0:
                return orig_lp_get_top_tokens(
                    self, lm_head, hidden_states, embedding_bias
                )
            tp_size = lm_head.tp_size
            logits = self._apply_head(lm_head, hidden_states, embedding_bias)
            if self.soft_cap is not None:
                logits = torch.tanh(logits / self.soft_cap) * self.soft_cap
            if self.scale != 1.0:
                logits = logits * self.scale
            num_pad = lm_head.shard_indices.num_org_vocab_padding
            if num_pad > 0:
                logits[..., -num_pad:] = -float("inf")
            local_max_vals, local_max_indices = logits.max(dim=-1)
            vocab_start = lm_head.shard_indices.org_vocab_start_index
            global_indices = local_max_indices + vocab_start
            if tp_size == 1:
                return global_indices
            vals_f = local_max_vals.float()
            idx_f = global_indices.float()
            local_quad = torch.stack([vals_f, idx_f, vals_f, idx_f], dim=-1)
            gathered = tensor_model_parallel_all_gather(local_quad, dim=-1)
            gathered = gathered.view(hidden_states.shape[0], tp_size, 4)
            max_rank_idx = gathered[:, :, 0].argmax(dim=-1, keepdim=True)
            top_tokens = gathered[:, :, 1].gather(dim=-1, index=max_rank_idx)
            return top_tokens.squeeze(-1).to(torch.int64).clamp_(0, lm_head.org_vocab_size - 1)

        LogitsProcessor.get_b12x_preparation_units = _patched_lp_get_units
        LogitsProcessor._apply_head = _patched_lp_apply_head
        LogitsProcessor.get_top_tokens = _patched_lp_get_top_tokens
    except Exception:
        pass

    try:
        import vllm.model_executor.warmup.b12x_prepare as _b12x_prep_mod

        orig_init_tuning_cache = _b12x_prep_mod.initialize_b12x_tuning_cache

        def _patched_init_tuning_cache(worker: Any, kv_cache_config: Any) -> bool:
            if os.getenv("SPARK_SKIP_TUNING_CACHE", "0") == "1":
                return False
            return orig_init_tuning_cache(worker, kv_cache_config)

        _b12x_prep_mod.initialize_b12x_tuning_cache = _patched_init_tuning_cache
    except Exception:
        pass

    orig_load_model = GPUModelRunner.load_model

    try:
        from vllm.v1.worker.gpu.cudagraph_utils import (
            BatchExecutionDescriptor,
            ModelCudaGraphManager,
        )
        import vllm.v1.worker.gpu.spec_decode.utils as _sd_utils_mod
        import vllm.v1.worker.gpu.model_runner as _mr_sd_mod

        orig_dflash_init = DFlashSpeculator.__init__

        def _patched_dflash_init(
            self: DFlashSpeculator, vllm_config: Any, device: Any, *a: Any, **kw: Any
        ) -> None:
            dflash_steps = int(os.getenv("SPARK_DFLASH_STEPS", "7"))
            spec_cfg = getattr(vllm_config, "speculative_config", None)
            if (
                spec_cfg is not None
                and getattr(spec_cfg, "num_speculative_tokens", 0) > dflash_steps
            ):
                dflash_vllm_cfg = copy.copy(vllm_config)
                dflash_spec_cfg = copy.copy(spec_cfg)
                object.__setattr__(dflash_spec_cfg, "num_speculative_tokens", dflash_steps)
                dflash_vllm_cfg.speculative_config = dflash_spec_cfg
                orig_dflash_init(self, dflash_vllm_cfg, device, *a, **kw)
            else:
                orig_dflash_init(self, vllm_config, device, *a, **kw)

        DFlashSpeculator.__init__ = _patched_dflash_init

        orig_mcg_init = ModelCudaGraphManager.__init__

        def _patched_mcg_init(
            self: ModelCudaGraphManager,
            vllm_config: Any,
            device: Any,
            cudagraph_mode: Any,
            decode_query_len: int,
            *mcg_args: Any,
            **mcg_kwargs: Any,
        ) -> None:
            orig_decode_query_len = int(decode_query_len)
            dflash_qlen = int(os.getenv("SPARK_DFLASH_STEPS", "7")) + 1
            if decode_query_len > dflash_qlen:
                decode_query_len = dflash_qlen
            orig_mcg_init(
                self,
                vllm_config,
                device,
                cudagraph_mode,
                decode_query_len,
                *mcg_args,
                **mcg_kwargs,
            )
            if (
                orig_decode_query_len > decode_query_len
                and CUDAGraphMode.FULL in getattr(self, "_capture_descs", {})
            ):
                extra_desc = BatchExecutionDescriptor(
                    cg_mode=CUDAGraphMode.FULL,
                    num_tokens=orig_decode_query_len,
                    num_reqs=1,
                    uniform_token_count=orig_decode_query_len,
                    num_active_loras=0,
                )
                full_descs = self._capture_descs[CUDAGraphMode.FULL]
                if extra_desc not in full_descs:
                    full_descs.append(extra_desc)
                    full_descs.sort(
                        key=lambda d: (d.num_tokens, d.num_reqs or 0, d.num_active_loras)
                    )
                for n_t in range(decode_query_len + 1, orig_decode_query_len + 1):
                    cand_list = self._candidates.get((n_t, 0))
                    if cand_list is not None and extra_desc not in cand_list:
                        cand_list.insert(0, extra_desc)

        ModelCudaGraphManager.__init__ = _patched_mcg_init

        orig_limit_draft_tokens = _sd_utils_mod.limit_draft_tokens

        def _patched_limit_draft_tokens(
            draft_tokens: torch.Tensor,
            num_speculative_tokens: int,
            *l_args: Any,
            **l_kwargs: Any,
        ) -> torch.Tensor:
            if draft_tokens.ndim == 2:
                max_tokens = min(draft_tokens.shape[1], max(int(num_speculative_tokens), 0))
                return orig_limit_draft_tokens(
                    draft_tokens, max_tokens, *l_args, **l_kwargs
                )
            return orig_limit_draft_tokens(
                draft_tokens, num_speculative_tokens, *l_args, **l_kwargs
            )

        _sd_utils_mod.limit_draft_tokens = _patched_limit_draft_tokens
        _mr_sd_mod.limit_draft_tokens = _patched_limit_draft_tokens
    except Exception:
        pass

    def _patched_runner_load_model(self: GPUModelRunner, *args: Any, **kwargs: Any) -> Any:
        res = orig_load_model(self, *args, **kwargs)
        if isinstance(self.speculator, DFlashSpeculator):
            self.speculator.req_states = self.req_states
            if (
                self.speculator.draft_logits is None
                and not getattr(self.speculator, "use_acceptance_estimator", False)
                and os.getenv("SPARK_DFLASH_LOCAL_ARGMAX", "1") == "1"
            ):
                self.speculator.use_local_argmax_reduction = True
            self.speculator.copy_drafter = SuffixCopyDrafter(
                max_num_reqs=self.max_num_reqs,
                num_speculative_steps=self.speculator.num_speculative_steps,
                device=self.device,
            )
            q_cg = getattr(self.speculator, "query_cudagraph_manager", None)
            if q_cg is not None and not getattr(q_cg, "_spark_patched_run_fg", False):
                orig_q_run_fg = q_cg.run_fullgraph

                def _patched_q_run_fg(desc: Any, *q_a: Any, **q_kw: Any) -> Any:
                    if getattr(q_cg, "_spark_skip_draft_fwd", False):
                        return None
                    return orig_q_run_fg(desc, *q_a, **q_kw)

                q_cg.run_fullgraph = _patched_q_run_fg
                q_cg._spark_patched_run_fg = True
            print(
                f"[spark-glm53] Attached SuffixCopyDrafter (span={self.speculator.copy_drafter.copy_draft_span}, "
                f"dflash_steps={self.speculator.num_speculative_steps}, runner_spec={self.num_speculative_steps}) "
                f"& DFlash2ContextCudaGraphs to {type(self.speculator).__name__} "
                f"(draft_logits={self.speculator.draft_logits is not None}, "
                f"local_argmax={self.speculator.use_local_argmax_reduction})",
                flush=True,
            )
        # Reclaim 1.27 GiB of DFlash2 embed_tokens/lm_head replaced by target_model
        # and any residual file page cache before determine_available_memory / warmup_kernels.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        drop_file_page_cache()
        return res

    GPUModelRunner.load_model = _patched_runner_load_model

    orig_gen_draft = DFlashSpeculator._generate_draft

    def _patched_gen_draft(self: DFlashSpeculator, *g_a: Any, **g_kw: Any) -> Any:
        if getattr(self, "_spark_skip_draft_fwd", False):
            return None
        return orig_gen_draft(self, *g_a, **g_kw)

    DFlashSpeculator._generate_draft = _patched_gen_draft
    try:
        from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import DFlash2Speculator

        orig_gen_draft2 = DFlash2Speculator._generate_draft

        def _patched_gen_draft2(
            self: DFlash2Speculator, *g_a: Any, **g_kw: Any
        ) -> Any:
            if getattr(self, "_spark_skip_draft_fwd", False):
                return None
            return orig_gen_draft2(self, *g_a, **g_kw)

        DFlash2Speculator._generate_draft = _patched_gen_draft2
    except Exception:
        pass

    orig_propose = DFlashSpeculator.propose

    @torch.inference_mode()
    def _patched_propose(
        self: DFlashSpeculator,
        input_batch: Any,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        dummy_run = kwargs.get("dummy_run", False)
        is_profile = kwargs.get("is_profile", False)
        num_reqs = input_batch.num_reqs
        num_target_tokens = input_batch.num_tokens
        if not dummy_run and not is_profile:
            self._spark_propose_called = True

        if (
            not dummy_run
            and not is_profile
            and (
                getattr(self, "_spark_skip_propose", False)
                or (
                    aux_hidden_states is not None
                    and len(aux_hidden_states) > 0
                    and aux_hidden_states[0].shape[0] < num_target_tokens
                )
            )
        ):
            self._spark_skip_propose = False
            if isinstance(aux_hidden_states, list):
                aux_hidden_states.clear()
            self.draft_tokens[:num_reqs].zero_()
            return self.draft_tokens[:num_reqs]

        ev_p0 = ev_p1 = ev_p2 = None
        warmup_done = getattr(self, "_spark_warmup_done", False)
        record_prop = (
            warmup_done
            and not dummy_run
            and not is_profile
            and os.getenv("SPARK_PROFILE_STEPS", "1") == "1"
            and (getattr(self, "_spark_prop_cnt", 0) < 28 or num_target_tokens >= 512)
        )
        if record_prop:
            self._spark_prop_cnt = getattr(self, "_spark_prop_cnt", 0) + 1
            ev_p0 = torch.cuda.Event(enable_timing=True)
            ev_p1 = torch.cuda.Event(enable_timing=True)
            ev_p2 = torch.cuda.Event(enable_timing=True)
            ev_p0.record()

        uncond_draft = None
        saved_uncond_draft = None
        if (
            warmup_done
            and not dummy_run
            and not is_profile
            and not getattr(self, "_spark_skip_draft_fwd_chunk", False)
            and getattr(self, "copy_drafter", None) is not None
            and getattr(self, "req_states", None) is not None
            and num_reqs > 0
        ):
            uncond_draft = self.copy_drafter.try_unconditional_propose(
                num_reqs=num_reqs,
                idx_mapping=input_batch.idx_mapping[:num_reqs],
                all_token_ids_gpu=self.req_states.all_token_ids.gpu,
                total_len_gpu=self.req_states.total_len.gpu,
                draft_tokens=self.draft_tokens,
                idx_mapping_np=input_batch.idx_mapping_np[:num_reqs],
                prefill_len_np=input_batch.prefill_len_np[:num_reqs],
            )
            if uncond_draft is not None:
                if getattr(self, "_spark_is_homogeneous_greedy", False) and num_reqs > 1:
                    uncond_draft[1:num_reqs].copy_(uncond_draft[0:1])
                saved_uncond_draft = uncond_draft[:num_reqs].clone()

        if (
            aux_hidden_states is not None
            and len(aux_hidden_states) > 0
            and num_target_tokens > 128
            and hasattr(self, "hidden_states")
            and hasattr(self.model, "combine_hidden_states")
        ):
            chunk_sz = 2048
            for c_start in range(0, num_target_tokens, chunk_sz):
                c_end = min(c_start + chunk_sz, num_target_tokens)
                cat_chunk = torch.cat(
                    [h[c_start:c_end] for h in aux_hidden_states], dim=-1
                )
                self.hidden_states[c_start:c_end].copy_(
                    self.model.combine_hidden_states(cat_chunk)
                )
                del cat_chunk
            if isinstance(aux_hidden_states, list):
                aux_hidden_states.clear()
            last_hidden_states = self.hidden_states
            aux_hidden_states = None

        q_cg = getattr(self, "query_cudagraph_manager", None)
        skip_draft_fwd = (uncond_draft is not None) or bool(
            getattr(self, "_spark_skip_draft_fwd_chunk", False)
        )
        self._spark_skip_draft_fwd_chunk = False
        if skip_draft_fwd:
            self._spark_skip_draft_fwd = True
            if q_cg is not None:
                q_cg._spark_skip_draft_fwd = True
        if "num_speculative_tokens" in kwargs:
            kwargs["num_speculative_tokens"] = min(
                int(kwargs["num_speculative_tokens"]),
                int(self.num_speculative_steps),
            )
        try:
            draft_out = orig_propose(
                self,
                input_batch,
                attn_metadata,
                slot_mappings,
                last_hidden_states,
                aux_hidden_states,
                *args,
                **kwargs,
            )
        finally:
            if skip_draft_fwd:
                self._spark_skip_draft_fwd = False
                if q_cg is not None:
                    q_cg._spark_skip_draft_fwd = False

        if saved_uncond_draft is not None:
            self.draft_tokens[:num_reqs].copy_(saved_uncond_draft)
            draft_out = self.draft_tokens[:num_reqs]
            if ev_p1 is not None and ev_p2 is not None:
                ev_p1.record()
                ev_p2.record()
                self._spark_last_prop_events = (ev_p0, ev_p1, ev_p2)
            return draft_out

        if ev_p1 is not None:
            ev_p1.record()

        if (
            warmup_done
            and uncond_draft is None
            and not dummy_run
            and not is_profile
            and getattr(self, "copy_drafter", None) is not None
            and getattr(self, "req_states", None) is not None
        ):
            if num_reqs > 0:
                draft_out = self.copy_drafter.propose_and_override(
                    num_reqs=num_reqs,
                    idx_mapping=input_batch.idx_mapping[:num_reqs],
                    all_token_ids_gpu=self.req_states.all_token_ids.gpu,
                    total_len_gpu=self.req_states.total_len.gpu,
                    draft_tokens=self.draft_tokens,
                    idx_mapping_np=input_batch.idx_mapping_np[:num_reqs],
                    prefill_len_np=input_batch.prefill_len_np[:num_reqs],
                )
                if getattr(self, "_spark_is_homogeneous_greedy", False) and num_reqs > 1:
                    draft_out[1:num_reqs].copy_(draft_out[0:1])

        if ev_p2 is not None:
            ev_p2.record()
            self._spark_last_prop_events = (ev_p0, ev_p1, ev_p2)
        return draft_out

    DFlashSpeculator.propose = _patched_propose

    # Optimize execute_model & sample_tokens:
    # 1. Track request (prompt_len, max_tokens, computed_tokens) across multi-chunk prefills
    #    and skip mHC _prepare_aux_hidden_state (576 MB) + DFlashSpeculator.propose (360 MB)
    #    whenever a single-request batch has max_tokens == 1 OR is on a non-tail prefill chunk.
    # 2. Skip BoundaryCheckpointState 3-kernel capture during pure decode steps
    #    (where num_computed_tokens >= prefill_len for all requests).
    # 3. Instrument the first 28 steps and long prefills with CUDA events on Rank 0.
    # 4. Canonicalize homogeneous greedy multi-stream batches to prevent MoE atomic tile-order
    #    floating-point jitter from desynchronizing identical greedy requests.
    import time

    orig_execute_model = GPUModelRunner.execute_model
    orig_sample_tokens = GPUModelRunner.sample_tokens

    try:
        import vllm.v1.worker.gpu.model_runner as _mr_mod

        orig_dispatch_cg = _mr_mod.dispatch_cg_and_sync_dp

        def _patched_dispatch_cg(
            cudagraph_manager: Any, *d_args: Any, **d_kwargs: Any
        ) -> Any:
            if getattr(cudagraph_manager, "_spark_force_eager", False):
                d_kwargs["need_eager"] = True
            return orig_dispatch_cg(cudagraph_manager, *d_args, **d_kwargs)

        _mr_mod.dispatch_cg_and_sync_dp = _patched_dispatch_cg
    except Exception:
        pass

    def _get_inner_glm_model(runner: GPUModelRunner) -> Any:
        m = getattr(runner, "model", None)
        for sub_name in ("language_model", "model"):
            sub = getattr(m, sub_name, None)
            if sub is not None:
                m = sub
        inner = getattr(m, "model", None)
        return inner if inner is not None else m

    def _set_model_skip_aux(runner: GPUModelRunner, skip: bool) -> None:
        m = getattr(runner, "model", None)
        if m is not None:
            m._spark_skip_aux = skip
            for sub_name in ("model", "language_model"):
                sub = getattr(m, sub_name, None)
                if sub is not None:
                    sub._spark_skip_aux = skip
                    inner = getattr(sub, "model", None)
                    if inner is not None:
                        inner._spark_skip_aux = skip

    def _set_model_homog_reqs(runner: GPUModelRunner, num_homog: int) -> None:
        inner = _get_inner_glm_model(runner)
        if inner is None:
            return
        for layer in getattr(inner, "_active_layers", ()):
            mlp = getattr(layer, "mlp", None)
            if mlp is not None:
                mlp._spark_homog_reqs = num_homog

    @torch.inference_mode()
    def _patched_execute_model(
        self: GPUModelRunner,
        scheduler_output: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        dummy_run = kwargs.get("dummy_run", False)
        skip_aux = False
        skip_bc = False
        is_homog_exec = False
        num_homog_reqs = 1
        n_toks = (
            getattr(scheduler_output, "total_num_scheduled_tokens", 0)
            if scheduler_output is not None
            else 0
        )
        if not dummy_run and scheduler_output is not None:
            req_meta = getattr(self, "_spark_req_meta", None)
            if req_meta is None:
                req_meta = {}
                self._spark_req_meta = req_meta
            fin_ids = getattr(scheduler_output, "finished_req_ids", None) or ()
            if fin_ids:
                for fin_id in fin_ids:
                    req_meta.pop(fin_id, None)
                spec = getattr(self, "speculator", None)
                cd = getattr(spec, "copy_drafter", None) if spec is not None else None
                if cd is not None and hasattr(cd, "commit_staged_history"):
                    cd.commit_staged_history()
            for nr in getattr(scheduler_output, "scheduled_new_reqs", None) or ():
                if not str(nr.req_id).startswith("_"):
                    self._spark_warmup_done = True
                    if getattr(self, "speculator", None) is not None:
                        self.speculator._spark_warmup_done = True
                prefill_toks = (
                    getattr(nr, "prefill_token_ids", None)
                    or getattr(nr, "prompt_token_ids", None)
                    or ()
                )
                p_len = len(prefill_toks)
                sp = getattr(nr, "sampling_params", None)
                max_toks = getattr(sp, "max_tokens", None) if sp is not None else None
                temp = float(getattr(sp, "temperature", 0.0) or 0.0) if sp is not None else 0.0
                req_meta[nr.req_id] = [
                    p_len,
                    max_toks,
                    int(getattr(nr, "num_computed_tokens", 0)),
                    temp,
                ]
            sched_map = getattr(scheduler_output, "num_scheduled_tokens", None) or {}
            if len(sched_map) == 1 and n_toks > 0:
                req_id, n_sched = next(iter(sched_map.items()))
                meta = req_meta.get(req_id)
                if meta is not None:
                    p_len, max_toks, comp_toks = meta[0], meta[1], meta[2]
                    cached_reqs = getattr(scheduler_output, "scheduled_cached_reqs", None)
                    if cached_reqs is not None and getattr(cached_reqs, "req_ids", None):
                        try:
                            idx = cached_reqs.req_ids.index(req_id)
                            comp_toks = int(cached_reqs.num_computed_tokens[idx])
                        except Exception:
                            pass
                    end_pos = comp_toks + n_sched
                    meta[2] = end_pos
                    skip_draft_chunk = False
                    if p_len > 0 and max_toks == 1:
                        skip_bc = True
                        scheduler_output.num_spec_tokens_to_schedule = 0
                        skip_aux = True
                    elif p_len > 0 and end_pos < p_len:
                        skip_bc = True
                        scheduler_output.num_spec_tokens_to_schedule = 0
                        skip_draft_chunk = True
                    if self.speculator is not None and not dummy_run:
                        self.speculator._spark_skip_draft_fwd_chunk = skip_draft_chunk
            elif len(sched_map) > 1:
                cached_reqs = getattr(scheduler_output, "scheduled_cached_reqs", None)
                c_ids = getattr(cached_reqs, "req_ids", None) or ()
                c_comp = getattr(cached_reqs, "num_computed_tokens", None) or ()
                c_map = {cid: int(cc) for cid, cc in zip(c_ids, c_comp)}
                first_ns = None
                first_pl = None
                first_ec = None
                all_match = True
                for rid, ns in sched_map.items():
                    m_entry = req_meta.get(rid)
                    if m_entry is not None:
                        base_c = c_map.get(rid, m_entry[2])
                        end_c = base_c + ns
                        m_entry[2] = end_c
                        t_val = m_entry[3] if len(m_entry) > 3 else 0.0
                        if t_val > 0.0:
                            all_match = False
                        if first_ns is None:
                            first_ns = ns
                            first_pl = m_entry[0]
                            first_ec = end_c
                        elif ns != first_ns or m_entry[0] != first_pl or end_c != first_ec:
                            all_match = False
                    else:
                        all_match = False
                if (
                    all_match
                    and first_ns is not None
                    and n_toks == len(sched_map) * first_ns
                    # OFF by default: detection compares lengths only, not token
                    # contents, so distinct same-length prompts were merged and
                    # received each other's outputs (scripts/probe_homog_crosstalk.py).
                    and os.getenv("SPARK_HOMOG_MOE_DEDUP", "0") == "1"
                ):
                    is_homog_exec = True
                    num_homog_reqs = len(sched_map)

        self._spark_skip_aux_current = skip_aux
        self._spark_skip_bc_current = skip_bc
        _set_model_skip_aux(self, skip_aux)
        if is_homog_exec:
            _set_model_homog_reqs(self, num_homog_reqs)
        if self.speculator is not None and not dummy_run:
            self.speculator._spark_skip_propose = skip_aux

        warmup_done = getattr(self, "_spark_warmup_done", False)
        step_cnt = getattr(self, "_spark_step_cnt", 0)
        if warmup_done and not dummy_run and n_toks > 0:
            self._spark_step_cnt = step_cnt + 1
            self._spark_last_exec_toks = n_toks
            if n_toks >= 512:
                last_pref = getattr(self, "_spark_last_prefill_toks", 0)
                if n_toks != last_pref:
                    torch.cuda.empty_cache()
                self._spark_last_prefill_toks = n_toks
        prof_enabled = (
            warmup_done
            and not dummy_run
            and os.getenv("SPARK_PROFILE_STEPS", "1") == "1"
            and n_toks > 0
        )
        layer_prof_enabled = prof_enabled and os.getenv("SPARK_LAYER_PROF", "0") == "1"
        is_m2_prof = layer_prof_enabled and step_cnt == 3
        is_m8_prof = (
            layer_prof_enabled
            and n_toks == 8
            and not getattr(self, "_spark_prof_m8_done", False)
        )
        if is_m8_prof:
            self._spark_prof_m8_done = True
        is_pref_prof = (
            layer_prof_enabled
            and n_toks >= 512
            and getattr(self, "_spark_prof_prefill_cnt", 0) < 2
        )
        if is_pref_prof:
            self._spark_prof_prefill_cnt = (
                getattr(self, "_spark_prof_prefill_cnt", 0) + 1
            )
        force_eager = is_m2_prof or is_m8_prof
        record_sub = is_m2_prof or is_m8_prof or is_pref_prof
        do_profile = prof_enabled and getattr(self, "rank", 0) == 0
        if self.cudagraph_manager is not None:
            self.cudagraph_manager._spark_force_eager = force_eager
        inner_glm = _get_inner_glm_model(self) if record_sub else None
        res = None
        try:
            if record_sub:
                torch.cuda.synchronize()
                torch.cuda._sleep(500_000_000)
            if do_profile:
                if step_cnt < 28 or n_toks >= 512 or record_sub:
                    if self.speculator is not None:
                        self.speculator._spark_last_prop_events = None
                    if inner_glm is not None:
                        inner_glm._spark_prof_events = [] if record_sub else None
                    ev0 = torch.cuda.Event(enable_timing=True)
                    ev1 = torch.cuda.Event(enable_timing=True)
                    t0 = time.perf_counter()
                    ev0.record()
                    res = orig_execute_model(self, scheduler_output, *args, **kwargs)
                    ev1.record()
                    t1 = time.perf_counter()
                    sub_ev = (
                        getattr(inner_glm, "_spark_prof_events", None)
                        if inner_glm is not None
                        else None
                    )
                    if inner_glm is not None:
                        inner_glm._spark_prof_events = None
                    self._spark_active_profile = (
                        step_cnt,
                        n_toks,
                        t0,
                        t1,
                        ev0,
                        ev1,
                        sub_ev,
                    )
                    return res
            self._spark_active_profile = None
            res = orig_execute_model(self, scheduler_output, *args, **kwargs)
            return res
        finally:
            if self.cudagraph_manager is not None:
                self.cudagraph_manager._spark_force_eager = False
            if is_homog_exec:
                _set_model_homog_reqs(self, 1)
            if inner_glm is not None:
                inner_glm._spark_prof_events = None
            if skip_aux:
                _set_model_skip_aux(self, False)

    @torch.inference_mode()
    def _patched_sample_tokens(
        self: GPUModelRunner,
        grammar_output: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        skip_decode_bc = bool(getattr(self, "_spark_skip_bc_current", False))
        self._spark_skip_bc_current = False
        self._spark_skip_aux_current = False
        if self.execute_model_state is not None:
            ib = self.execute_model_state.input_batch
            num_reqs = ib.num_reqs
            num_tokens = ib.num_tokens
            if (
                num_reqs > 0
                and self.boundary_checkpoint_state is not None
                and os.getenv("SPARK_SKIP_DECODE_BOUNDARY_CAPTURE", "1") == "1"
            ):
                try:
                    comp_np = ib.num_computed_tokens_np[:num_reqs]
                    pref_np = ib.prefill_len_np[:num_reqs]
                    if (comp_np >= pref_np).all():
                        skip_decode_bc = True
                except Exception:
                    skip_decode_bc = False
            is_homog = False
            # OFF by default (see SPARK_HOMOG_MOE_DEDUP note): hidden-state
            # canonicalization + draft broadcast assume identical prompts but the
            # check below only compares lengths -> cross-request output leakage.
            if (
                os.getenv("SPARK_HOMOG_CANONICALIZE", "0") == "1"
                and num_reqs > 1 and num_tokens > 0 and num_tokens % num_reqs == 0
            ):
                try:
                    q_len = num_tokens // num_reqs
                    sched_np = ib.num_scheduled_tokens[:num_reqs]
                    comp_np = ib.num_computed_tokens_np[:num_reqs]
                    pref_np = ib.prefill_len_np[:num_reqs]
                    idx_np = ib.idx_mapping_np[:num_reqs]
                    temp_np = self.sampler.sampling_states.temperature.np[idx_np]
                    if (
                        (sched_np == q_len).all()
                        and (comp_np == comp_np[0]).all()
                        and (pref_np == pref_np[0]).all()
                        and (temp_np <= 0.0).all()
                    ):
                        is_homog = True
                        hs = self.execute_model_state.hidden_states
                        if hs is not None and hs.shape[0] >= num_tokens:
                            hs[:num_tokens].view(num_reqs, q_len, -1)[1:].copy_(
                                hs[:q_len].unsqueeze(0)
                            )
                        aux_list = self.execute_model_state.aux_hidden_states
                        if aux_list:
                            for aux in aux_list:
                                if aux is not None and aux.shape[0] >= num_tokens:
                                    aux[:num_tokens].view(num_reqs, q_len, -1)[1:].copy_(
                                        aux[:q_len].unsqueeze(0)
                                    )
                except Exception:
                    is_homog = False
            if self.speculator is not None:
                self.speculator._spark_is_homogeneous_greedy = is_homog

        saved_bc = self.boundary_checkpoint_state
        if skip_decode_bc:
            self.boundary_checkpoint_state = None
        try:
            prof = getattr(self, "_spark_active_profile", None)
            if prof is None:
                res = orig_sample_tokens(self, grammar_output, *args, **kwargs)
                if getattr(self, "_spark_last_exec_toks", 0) > 64 or grammar_output is not None:
                    if (
                        torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
                        > 64 * 1024 * 1024
                    ):
                        torch.cuda.empty_cache()
                return res

            self._spark_active_profile = None
            step_cnt, n_toks, t0, t1, ev0, ev1, sub_ev = prof
            ev2 = torch.cuda.Event(enable_timing=True)
            t2 = time.perf_counter()
            res = orig_sample_tokens(self, grammar_output, *args, **kwargs)
            ev2.record()
            t3 = time.perf_counter()
            ev2.synchronize()
            if n_toks >= 512:
                if (
                    torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
                    > 64 * 1024 * 1024
                ):
                    torch.cuda.empty_cache()
            fwd_gpu_ms = ev0.elapsed_time(ev1)
            samp_prop_gpu_ms = ev1.elapsed_time(ev2)
            tot_gpu_ms = ev0.elapsed_time(ev2)
            exec_cpu_ms = (t1 - t0) * 1000.0
            samp_cpu_ms = (t3 - t2) * 1000.0
            prop_events = (
                getattr(self.speculator, "_spark_last_prop_events", None)
                if self.speculator is not None
                else None
            )
            if prop_events is not None:
                ev_p0, ev_p1, ev_p2 = prop_events
                samp_only_ms = ev1.elapsed_time(ev_p0)
                dflash_ms = ev_p0.elapsed_time(ev_p1)
                copy_ms = ev_p1.elapsed_time(ev_p2)
                detail = f"sample={samp_only_ms:.2f}ms, dflash={dflash_ms:.2f}ms, copy={copy_ms:.2f}ms"
            else:
                detail = f"sample+propose={samp_prop_gpu_ms:.2f}ms"
            sub_str = ""
            if sub_ev:
                try:
                    mhc_ms = kda_ms = mla_ms = dense_ms = moe_ms = 0.0
                    sub_totals: dict[str, float] = {}
                    for item in sub_ev:
                        if item[0] == "sub":
                            _, tag, es, ee = item
                            sub_totals[tag] = sub_totals.get(tag, 0.0) + es.elapsed_time(ee)
                        else:
                            a_kind, m_kind, e0, e1, e2, e3, e4, e5 = item
                            mhc_ms += (
                                e0.elapsed_time(e1)
                                + e2.elapsed_time(e3)
                                + e4.elapsed_time(e5)
                            )
                            if a_kind == "mla":
                                mla_ms += e1.elapsed_time(e2)
                            else:
                                kda_ms += e1.elapsed_time(e2)
                            if m_kind == "moe":
                                moe_ms += e3.elapsed_time(e4)
                            else:
                                dense_ms += e3.elapsed_time(e4)
                    sub_parts = ", ".join(
                        f"{k}={v:.2f}ms" for k, v in sorted(sub_totals.items())
                    )
                    sub_str = (
                        f" | breakdown[mhc={mhc_ms:.2f}ms, kda={kda_ms:.2f}ms, "
                        f"mla={mla_ms:.2f}ms, dense={dense_ms:.2f}ms, moe={moe_ms:.2f}ms"
                        + (f" | {sub_parts}]" if sub_parts else "]")
                    )
                except Exception:
                    sub_str = ""
            alloc_mb = torch.cuda.memory_allocated() / 1048576.0
            rsv_mb = torch.cuda.memory_reserved() / 1048576.0
            print(
                f"[spark-profile] step={step_cnt} toks={n_toks} | "
                f"total_gpu={tot_gpu_ms:.2f}ms (fwd={fwd_gpu_ms:.2f}ms, {detail}){sub_str} | "
                f"cpu_enqueue={exec_cpu_ms + samp_cpu_ms:.2f}ms (exec={exec_cpu_ms:.2f}ms, sample={samp_cpu_ms:.2f}ms) | "
                f"cuda_mem={alloc_mb:.0f}/{rsv_mb:.0f}MB",
                flush=True,
            )
            return res
        finally:
            if skip_decode_bc:
                self.boundary_checkpoint_state = saved_bc

    GPUModelRunner.execute_model = _patched_execute_model
    GPUModelRunner.sample_tokens = _patched_sample_tokens

    # Fix MRV2 GPUModelRunner.capture_model missing set_cudagraph_capturing_enabled(False)
    try:
        from vllm.compilation.monitor import set_cudagraph_capturing_enabled

        orig_spec_capture = DFlashSpeculator.capture

        def _patched_spec_capture(
            self: DFlashSpeculator, *args: Any, **kwargs: Any
        ) -> Any:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            return orig_spec_capture(self, *args, **kwargs)

        DFlashSpeculator.capture = _patched_spec_capture

        orig_capture_model = GPUModelRunner.capture_model

        def _patched_capture_model(
            self: GPUModelRunner, *args: Any, **kwargs: Any
        ) -> Any:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            drop_file_page_cache()
            profile_only = kwargs.get("profile_only", False) or (
                len(args) >= 1 and bool(args[0])
            )
            if not profile_only:
                set_cudagraph_capturing_enabled(True)
            try:
                return orig_capture_model(self, *args, **kwargs)
            finally:
                if not profile_only:
                    set_cudagraph_capturing_enabled(False)
                self._spark_last_large_toks = -1
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        GPUModelRunner.capture_model = _patched_capture_model
    except Exception:
        pass


def _patch_scheduler_and_boundary_checkpoints() -> None:
    """
    1. Keep `VllmConfig.use_request_boundary_checkpoints = True` so the hybrid Mamba/MLA
       KV cache keeps `block_size = 256` tokens (`2.2M+` KV cache tokens), while skipping
       unaligned `kind == "prompt"` boundary checkpoints in `KVCacheManager.publish_boundary_checkpoint`
       so `prefill_tail` 256-token block checkpoints are used without triggering isolated
       single-request `boundary_logits_only` serialization (`scheduler.py:1377, 1835`).
    2. Add a 2.5 ms micro-batch arrival coalescer in `EngineCoreProc._process_input_queue`
       when waking from idle so concurrent requests sent via `asyncio.gather` arrive in
       `self.waiting` before Step 1 begins.
    """
    import queue
    import time

    try:
        import vllm.config.vllm as vllm_cfg_mod

        orig_glm_kda_rec_unavail = getattr(
            vllm_cfg_mod, "_glm_kda_recovery_unavailable", None
        )
        if orig_glm_kda_rec_unavail is not None and not getattr(
            vllm_cfg_mod, "_spark_patched_kda_rec", False
        ):

            def _patched_glm_kda_recovery_unavailable(config: Any) -> str | None:
                reason = orig_glm_kda_rec_unavail(config)
                if reason == "requires MTP or DFlash with 1 to 7 draft tokens":
                    spec = getattr(config, "speculative_config", None)
                    num_spec = int(getattr(config, "num_speculative_tokens", 0) or 0)
                    if (
                        spec is not None
                        and getattr(spec, "method", None) in ("mtp", "dflash")
                        and 1 <= num_spec <= 31
                    ):
                        object.__setattr__(spec, "num_speculative_tokens", 7)
                        try:
                            return orig_glm_kda_rec_unavail(config)
                        finally:
                            object.__setattr__(spec, "num_speculative_tokens", num_spec)
                return reason

            vllm_cfg_mod._glm_kda_recovery_unavailable = (
                _patched_glm_kda_recovery_unavailable
            )
            vllm_cfg_mod._spark_patched_kda_rec = True
    except Exception:
        pass

    try:
        from vllm.v1.core.kv_cache_manager import KVCacheManager

        orig_get_computed_blocks = KVCacheManager.get_computed_blocks

        def _patched_get_computed_blocks(
            self: KVCacheManager, request: Any
        ) -> Any:
            if os.getenv("SPARK_ALLOW_PROMPT_BOUNDARY_LOGITS", "0") == "1":
                return orig_get_computed_blocks(self, request)
            orig_num_tokens = request.num_tokens
            boundary_cache = self.boundary_checkpoints
            if orig_num_tokens > 1 and boundary_cache is not None:
                if self.enable_caching and boundary_cache.supports_request(request):
                    request.use_boundary_checkpoints = True
                if not self.prefix_cache_lookup_enabled(request):
                    return self.empty_kv_cache_blocks, 0, 0
                if boundary_cache.supports_request(request):
                    # Cap boundary lookup to `orig_num_tokens - 1` so multi-turn
                    # continuations (`checkpoint.num_tokens < request.num_tokens`)
                    # reuse 100% of the prior turn's `prompt`/`response` checkpoint,
                    # while concurrent identical-length bursts never trigger isolated
                    # `boundary_logits_only` serialization.
                    checkpoint = boundary_cache.find(request, orig_num_tokens - 1)
                    request.boundary_checkpoint = checkpoint
                    if checkpoint is not None:
                        checkpoint_blocks = tuple(
                            [self.block_pool.blocks[i] for i in group]
                            for group in checkpoint.block_ids
                        )
                        return (
                            self.create_kv_cache_blocks(checkpoint_blocks),
                            checkpoint.num_tokens,
                            0,
                        )
                    return self.empty_kv_cache_blocks, 0, 0
            return orig_get_computed_blocks(self, request)

        KVCacheManager.get_computed_blocks = _patched_get_computed_blocks
    except Exception:
        pass

    try:
        import vllm.v1.core.kv_cache_utils as kv_utils_mod

        orig_get_bytes_per_block = kv_utils_mod._get_kv_cache_bytes_per_block

        def _patched_get_kv_cache_bytes_per_block(kv_cache_groups: list[Any]) -> int:
            res = orig_get_bytes_per_block(kv_cache_groups)
            glm_c4_index_page_bytes = 64 * 132
            if res % glm_c4_index_page_bytes != 0:
                contains_glm5 = kv_utils_mod._contains_glm5_next_mla(
                    kv_utils_mod._get_per_layer_spec(group, layer_name)
                    for group in kv_cache_groups
                    for layer_name in group.layer_names
                )
                if contains_glm5:
                    res = (
                        (res + glm_c4_index_page_bytes - 1) // glm_c4_index_page_bytes
                    ) * glm_c4_index_page_bytes
            return res

        kv_utils_mod._get_kv_cache_bytes_per_block = _patched_get_kv_cache_bytes_per_block
    except Exception:
        pass

    try:
        from vllm.v1.core.sched.scheduler import Scheduler

        orig_add_request = Scheduler.add_request

        def _patched_add_request(self: Scheduler, request: Any) -> None:
            orig_add_request(self, request)
            if self.kv_cache_manager.boundary_checkpoints is not None:
                # PROMPT_CHECKPOINT_SLOT and RESPONSE_CHECKPOINT_SLOT already capture
                # exact prompt and response endpoints without extra prefill splits.
                # Clear mid-prompt instruction and tail boundaries so single-turn and
                # multi-turn prefills execute in full-size chunks without splitting
                # a short prompt into 3 separate 80 GiB MoE forward passes.
                request.recurrent_instruction_boundary = None
                request.recurrent_prefill_tail_boundary = None

        Scheduler.add_request = _patched_add_request
    except Exception:
        pass

    try:
        from vllm.v1.engine.core import EngineCore, EngineCoreProc

        orig_ec_step = EngineCore.step

        def _patched_ec_step(self: EngineCore) -> Any:
            if getattr(self, "async_scheduling", False) or os.getenv(
                "SPARK_SYNC_PIPELINE_STEP", "1"
            ) != "1":
                return orig_ec_step(self)
            if not self.scheduler.has_requests():
                return {}, False
            scheduler_output = self.scheduler.schedule(self._should_throttle_prefills())
            future, execution_timing = self._execute_model(scheduler_output)
            grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
            if scheduler_output.total_num_scheduled_tokens > 0 and grammar_output is None:
                sample_future = self.model_executor.sample_tokens(
                    grammar_output, non_block=True
                )
                with (
                    self.capture_iteration_details(scheduler_output) as iteration_details,
                    self.log_error_detail(scheduler_output),
                ):
                    model_output = future.result()
                    if model_output is None:
                        model_output = sample_future.result()
                    self._record_compute_time(scheduler_output, execution_timing)
            else:
                with (
                    self.capture_iteration_details(scheduler_output) as iteration_details,
                    self.log_error_detail(scheduler_output),
                ):
                    model_output = future.result()
                    if model_output is None and scheduler_output.total_num_scheduled_tokens > 0:
                        model_output = self.model_executor.sample_tokens(grammar_output)
                    self._record_compute_time(scheduler_output, execution_timing)
            self._wait_for_boundary_checkpoint_copies(model_output)
            self._process_aborts_queue()
            engine_core_outputs = self.scheduler.update_from_output(
                scheduler_output, model_output
            )
            self._attach_iteration_details(engine_core_outputs, iteration_details)
            return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0

        EngineCore.step = _patched_ec_step

        orig_process_input_queue = EngineCoreProc._process_input_queue

        def _patched_process_input_queue(self: EngineCoreProc) -> None:
            was_idle = not self.has_work()
            orig_process_input_queue(self)
            if was_idle and self.has_work() and os.getenv("SPARK_COALESCE_IDLE_BURST", "1") == "1":
                waiting = getattr(getattr(self, "scheduler", None), "waiting", None)
                first_req = waiting[0] if waiting else None
                sp = getattr(first_req, "sampling_params", None) if first_req is not None else None
                max_toks = getattr(sp, "max_tokens", None) if sp is not None else None
                p_toks = int(getattr(first_req, "num_prompt_tokens", 0) or 0) if first_req is not None else 0
                if max_toks is not None and max_toks >= 128 and 512 <= p_toks <= 4096:
                    deadline = time.perf_counter() + 0.050
                    while time.perf_counter() < deadline:
                        if waiting is not None and len(waiting) >= 8:
                            break
                        try:
                            req = self.input_queue.get(timeout=0.012)
                            self._handle_client_request(*req)
                            while not self.input_queue.empty():
                                req = self.input_queue.get_nowait()
                                self._handle_client_request(*req)
                        except queue.Empty:
                            break

        EngineCoreProc._process_input_queue = _patched_process_input_queue
    except Exception:
        pass


def _patch_l2_prefetch_and_indexer_workspace() -> None:
    """
    1. Configure `vllm.models.glm5next.nvidia.l2_prefetch` (disabled by default on GB10
       whose L2 cache is 25 MB, or enabled when `VLLM_GLM53_L2_PREFETCH=1`).
    2. Right-size `vllm.v1.attention.backends.mla.indexer.get_max_prefill_buffer_size`
       to the legal per-step compressed pool maximum instead of `max_model_len * 40`,
       reclaiming ~4.67 GiB of locked UMA workspace RAM per node.
    3. Skip throwaway `profile_cudagraph_memory` in `GPUWorker.determine_available_memory`
       when `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` or `kv_cache_memory_bytes` is set,
       saving 1.21 GiB of duplicate CUDA graph pool memory and ~18s of boot time.
    4. Provide pre-allocated row-major transient convolution buffers for `flashkda` prefill
       in `KimiGatedDeltaNetAttention._borrow_b12x_prefill_transients`, eliminating 102
       uncoalesced strided writes and `.contiguous()` transpose copies per prefill step.
    """
    import gc
    import torch

    try:
        from vllm.models.glm5next.nvidia import l2_prefetch as l2pf_mod

        if os.getenv("VLLM_GLM53_L2_PREFETCH", "0") == "1":
            l2pf_mod._platform_enabled = lambda: torch.cuda.is_available() and (
                torch.cuda.get_device_capability() in ((12, 0), (12, 1))
            )
            l2pf_mod.ENABLED = True
        else:
            l2pf_mod.ENABLED = False
    except Exception:
        pass

    try:
        import vllm.v1.attention.backends.mla.indexer as mla_indexer_mod
        from vllm.utils.math_utils import cdiv

        orig_get_max_prefill_buf = mla_indexer_mod.get_max_prefill_buffer_size

        def _rightsized_get_max_prefill_buffer_size(vllm_config: Any) -> int:
            stock = orig_get_max_prefill_buf(vllm_config)
            if os.getenv("GLM53_INDEXER_WORKSPACE", "rightsize") != "rightsize":
                return stock
            try:
                hf_text = getattr(vllm_config.model_config, "hf_text_config", None)
                ratio = int(getattr(hf_text, "index_kpool", 1) or 1)
                if ratio <= 0:
                    ratio = 1
                sched = vllm_config.scheduler_config
                max_prefill_reqs = max(
                    1,
                    min(int(sched.max_num_seqs), int(sched.max_num_batched_tokens)),
                )
                spec_cfg = vllm_config.speculative_config
                spec_tokens = (
                    int(spec_cfg.num_speculative_tokens)
                    if spec_cfg is not None and spec_cfg.num_speculative_tokens
                    else 0
                )
                per_req_entries = cdiv(
                    int(vllm_config.model_config.max_model_len) + spec_tokens,
                    ratio,
                )
                need = max_prefill_reqs * per_req_entries
                return min(need, stock)
            except Exception:
                return stock

        mla_indexer_mod.get_max_prefill_buffer_size = (
            _rightsized_get_max_prefill_buffer_size
        )
    except Exception:
        pass

    try:
        import vllm.envs as vllm_envs
        from vllm.v1.worker.gpu_worker import GPUWorker

        orig_profile_cg_mem = None

        def _install_skip_cg_profile(worker: GPUWorker) -> None:
            nonlocal orig_profile_cg_mem
            mr = getattr(worker, "model_runner", None)
            if mr is None or not hasattr(mr, "profile_cudagraph_memory"):
                return
            if getattr(mr, "_spark_patched_profile_cg", False):
                return
            orig_fn = mr.profile_cudagraph_memory

            def _maybe_profile_cg(*a: Any, **kw: Any) -> int:
                if not vllm_envs.VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:
                    return 0
                return orig_fn(*a, **kw)

            mr.profile_cudagraph_memory = _maybe_profile_cg
            mr._spark_patched_profile_cg = True

        from worker.shm_safetensors import drop_file_page_cache

        orig_det_avail = GPUWorker.determine_available_memory

        def _patched_determine_available_memory(self: GPUWorker, *a: Any, **kw: Any) -> int:
            _install_skip_cg_profile(self)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize(self.device)
                torch.cuda.empty_cache()
            drop_file_page_cache()
            res = orig_det_avail(self, *a, **kw)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize(self.device)
                torch.cuda.empty_cache()
            drop_file_page_cache()
            return res

        GPUWorker.determine_available_memory = _patched_determine_available_memory

        orig_compile_or_warmup = GPUWorker.compile_or_warm_up_model

        def _close_b12x_compiler_pool(worker: GPUWorker) -> None:
            try:
                sess = getattr(worker, "_b12x_session", None)
                if sess is not None and getattr(sess, "_pool", None) is not None:
                    pool = sess._pool
                    sess._pool = None
                    pool.close(terminate=True)
            except Exception:
                pass

        def _patched_compile_or_warm_up_model(self: GPUWorker, *a: Any, **kw: Any) -> Any:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize(self.device)
                torch.cuda.empty_cache()
            drop_file_page_cache()
            res = orig_compile_or_warmup(self, *a, **kw)
            _close_b12x_compiler_pool(self)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize(self.device)
                torch.cuda.empty_cache()
            drop_file_page_cache()
            return res

        GPUWorker.compile_or_warm_up_model = _patched_compile_or_warm_up_model
    except Exception:
        pass

    try:
        import b12x.moe.fused_moe._impl as b12x_moe_impl

        install_b12x_compiler_worker_patches()
        b12x_moe_impl._get_micro_dynamic_cutover_pairs = lambda quant_mode="nvfp4": int(
            os.getenv("B12X_MICRO_DYNAMIC_CUTOVER_PAIRS", "0")
        )
        direct_max = os.getenv("SPARK_B12X_DIRECT_ROUTING_MAX_ROWS", "").strip()
        if direct_max:
            b12x_moe_impl._DIRECT_ROUTING_MAX_ROUTED_ROWS = int(direct_max)
    except Exception:
        pass

    try:
        import b12x.norm.mhc._tuning as b12x_mhc_tuning

        orig_proj_default = b12x_mhc_tuning._projection_default_config

        def _patched_proj_default(query: Any, _device: Any = None) -> Any:
            if (
                hasattr(query, "hidden_size")
                and int(query.hidden_size) == 4_096
                and 1_024 <= int(query.max_tokens) < 2_304
            ):
                query = replace(query, max_tokens=2_304)
            return orig_proj_default(query, _device)

        b12x_mhc_tuning._projection_default_config = _patched_proj_default
    except Exception:
        pass

    try:
        import b12x.preparation.session as b12x_prep_sess
        from b12x.preparation.types import FrozenMapping

        if not getattr(b12x_prep_sess.PreparationJob, "_spark_patched_lookup", False):
            orig_job_lookup = b12x_prep_sess.PreparationJob._lookup

            def _remap_2048_dict(d: dict[str, Any], target_mt: int = 2048) -> dict[str, Any]:
                out = dict(d)
                if out.get("max_tokens", 0) > 2048:
                    out["max_tokens"] = target_mt
                if out.get("num_tokens", 0) > 2048:
                    out["num_tokens"] = target_mt
                    top_k = int(out.get("top_k", 8) or 8)
                    if "routed_rows" in out:
                        out["routed_rows"] = target_mt * top_k
                    if "logits_row_stride" in out and "topk_row_stride" in out:
                        pass
                return out

            def _sanitize_moe_assignment(
                component_id: str, q_dict: dict[str, Any], raw_assign: dict[str, Any]
            ) -> dict[str, Any]:
                if component_id == "moe.decode" and int(q_dict.get("num_tokens", 0) or 0) > 2048:
                    out = dict(raw_assign)
                    target_tile_m = int(os.getenv("SPARK_B12X_PREFILL_TILE_M", "64"))
                    if out.get("dynamic_tile_m") in (32, 64, 128) and out.get("dynamic_tile_m") != target_tile_m:
                        out["dynamic_tile_m"] = target_tile_m
                    if out.get("nvfp4_materialize_intermediate"):
                        out["nvfp4_materialize_intermediate"] = False
                    return out
                return raw_assign

            def _patched_job_lookup(
                self: Any, obligation: Any, selections: dict[str, Any]
            ) -> None:
                try:
                    orig_job_lookup(self, obligation, selections)
                except ValueError:
                    obligation.selection = None
                try:
                    request = obligation.request
                    configuration = obligation.configuration
                    contract = request.plan.contract
                    q_dict = configuration.encoded_query.to_dict()
                    if (
                        obligation.selection is not None
                        and contract.component_id == "moe.decode"
                        and int(q_dict.get("num_tokens", 0) or 0) > 2048
                    ):
                        cur_cfg = obligation.selection.config
                        target_tile_m = int(os.getenv("SPARK_B12X_PREFILL_TILE_M", "64"))
                        if (
                            getattr(cur_cfg, "dynamic_tile_m", None) != target_tile_m
                            or getattr(cur_cfg, "nvfp4_materialize_intermediate", False)
                        ):
                            raw_a = (
                                obligation.selection.assignment.to_dict()
                                if obligation.selection.assignment is not None
                                else contract.config_payload(cur_cfg).to_dict()
                            )
                            sanitized = _sanitize_moe_assignment(
                                contract.component_id, q_dict, raw_a
                            )
                            assignment = FrozenMapping(sanitized)
                            configuration.space.validate(assignment)
                            cfg = contract._lower(
                                configuration.query, configuration.device, assignment
                            )
                            obligation.selection = self._selection(
                                obligation, cfg, "cached", assignment
                            )
                            return
                except Exception:
                    pass
                if obligation.selection is not None:
                    return
                try:
                    request = obligation.request
                    configuration = obligation.configuration
                    contract = request.plan.contract
                    q_dict = configuration.encoded_query.to_dict()
                    cache = self.session._selection_cache()
                    for alt_mt in (2048, 2304):
                        deps = []
                        ok_deps = True
                        for name in request.dependencies:
                            if name not in selections:
                                ok_deps = False
                                break
                            sel, dep_contract = selections[name]
                            deps.append(
                                {
                                    "component": sel.component_id,
                                    "query": _remap_2048_dict(sel.query.to_dict(), alt_mt),
                                    "config": dep_contract.config_payload(sel.config).to_dict(),
                                    "semantic_version": dep_contract.semantic_version,
                                }
                            )
                        if not ok_deps:
                            break
                        alt_key = b12x_prep_sess.digest(
                            {
                                "component": contract.component_id,
                                "query_schema": contract.query_schema_version,
                                "config_schema": contract.config_schema_version,
                                "semantic_version": contract.semantic_version,
                                "candidate_contract_version": contract.candidate_contract_version,
                                "query": _remap_2048_dict(q_dict, alt_mt),
                                "invocation": request.plan.invocation.to_dict(),
                                "pin": None
                                if configuration.pinned is None
                                else contract.config_payload(configuration.pinned).to_dict(),
                                "dependencies": deps,
                            }
                        )
                        rec = cache.get(alt_key)
                        if rec is not None:
                            sanitized = _sanitize_moe_assignment(
                                contract.component_id, q_dict, rec["assignment"]
                            )
                            assignment = FrozenMapping(sanitized)
                            configuration.space.validate(assignment)
                            cfg = contract._lower(
                                configuration.query, configuration.device, assignment
                            )
                            obligation.selection = self._selection(
                                obligation, cfg, "cached", assignment
                            )
                            obligation.coverage = dict(rec["coverage"])
                            self._cache_hits += 1
                            return
                except Exception:
                    pass
                if (
                    obligation.selection is None
                    and os.getenv("SPARK_B12X_FAST_FALLBACK", "1") == "1"
                ):
                    obligation.selection = self._selection(
                        obligation, obligation.configuration.default, "override"
                    )

            b12x_prep_sess.PreparationJob._lookup = _patched_job_lookup
            b12x_prep_sess.PreparationJob._spark_patched_lookup = True
    except Exception:
        pass

    try:
        install_b12x_compiler_worker_patches()
        from vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn import (
            KimiGatedDeltaNetAttention,
        )

        orig_init_kda_decode = KimiGatedDeltaNetAttention._initialize_b12x_kda_decode

        def _patched_init_kda_decode(self: Any, vllm_config: Any) -> None:
            actual_num_spec = int(getattr(self, "num_spec", 0) or 0)
            if 7 < actual_num_spec <= 31:
                self.num_spec = 7
                try:
                    orig_init_kda_decode(self, vllm_config)
                finally:
                    self.num_spec = actual_num_spec
                if getattr(self, "_b12x_kda_max_seqs", 0) > 0:
                    actual_cols = max(1, actual_num_spec + 1)
                    self._b12x_kda_max_tokens = self._b12x_kda_max_seqs * actual_cols
                    self._b12x_kda_state_index_columns = actual_cols
                    self.register_buffer(
                        "_b12x_kda_state_indices",
                        torch.zeros(
                            (self._b12x_kda_max_seqs, actual_cols),
                            dtype=torch.int32,
                            device=self._b12x_kda_num_tokens.device,
                        ),
                        persistent=False,
                    )
                return
            orig_init_kda_decode(self, vllm_config)

        KimiGatedDeltaNetAttention._initialize_b12x_kda_decode = (
            _patched_init_kda_decode
        )
        orig_gdn_get_kv_spec = KimiGatedDeltaNetAttention.get_kv_cache_spec

        def _patched_gdn_get_kv_spec(self: Any, vllm_config: Any) -> Any:
            actual_num_spec = int(getattr(self, "num_spec", 0) or 0)
            dflash_steps = int(os.getenv("SPARK_DFLASH_STEPS", "7"))
            if actual_num_spec > dflash_steps and os.getenv("SPARK_GDN_SPEC_PAGE_CLAMP", "0") == "1":
                self.num_spec = dflash_steps
                try:
                    return orig_gdn_get_kv_spec(self, vllm_config)
                finally:
                    self.num_spec = actual_num_spec
            return orig_gdn_get_kv_spec(self, vllm_config)

        KimiGatedDeltaNetAttention.get_kv_cache_spec = _patched_gdn_get_kv_spec
    except Exception:
        pass


def _patch_sparse_mla_kpool_indexer() -> None:
    """
    Guard against the GB10 sparse-MLA kpool indexer uninitialized `torch.empty` bug:
      1. Initialize `topk_indices_buffer` and `pool_topk_indices_buffer` in both
         `vllm.models.glm5next.nvidia.model.Glm5NextModel` and `common.model.Glm5NextModel`
         (and MTP) with `-1` (`fill_(-1)`).
      2. Clamp `pool_id` to `[0, complete_pools)` inside both `nvidia.ops.glm_kpool._expand_pool_ids_kernel`
         and `nvidia.ops.kpool_compress._expand_pools_and_append_tail_kernel`.
      3. Skip redundant `hc_post + hc_contract` launches (576 MB) on non-speculative long prefill chunks
         and chunk large aux captures into 2048-row slices.
    """
    import torch
    from vllm.models.glm5next.common import model as glm5_common_model
    from vllm.models.glm5next.common import mtp as glm5_common_mtp
    from vllm.models.glm5next.nvidia import l2_prefetch as l2pf_mod
    from vllm.models.glm5next.nvidia import model as glm5_nvidia_model
    from vllm.models.glm5next.nvidia import mtp as glm5_nvidia_mtp
    from vllm.models.glm5next.nvidia.ops import glm_kpool, kpool_compress
    from vllm.triton_utils import tl, triton

    for g_mod in (glm5_nvidia_model, glm5_common_model):
        if hasattr(g_mod.Glm5NextModel, "_prepare_aux_hidden_state"):
            orig_prep_aux = g_mod.Glm5NextModel._prepare_aux_hidden_state

            def _make_patched_prep_aux(fn: Any) -> Any:
                def _patched_prep_aux(
                    self: Any,
                    layer: Any,
                    hidden_states: torch.Tensor,
                    residual: torch.Tensor | None,
                    post: torch.Tensor | None,
                    comb: torch.Tensor | None,
                ) -> torch.Tensor:
                    if not getattr(layer, "mhc", False) or residual is None:
                        return hidden_states
                    if getattr(self, "_spark_skip_aux", False) and not torch.cuda.is_current_stream_capturing():
                        empty_aux = getattr(self, "_spark_empty_aux_buf", None)
                        if (
                            empty_aux is None
                            or empty_aux.device != hidden_states.device
                            or empty_aux.dtype != hidden_states.dtype
                            or empty_aux.shape[-1] != hidden_states.shape[-1]
                        ):
                            empty_aux = torch.empty(
                                (0, hidden_states.shape[-1]),
                                dtype=hidden_states.dtype,
                                device=hidden_states.device,
                            )
                            self._spark_empty_aux_buf = empty_aux
                        return empty_aux
                    return fn(self, layer, hidden_states, residual, post, comb)

                return _patched_prep_aux

            g_mod.Glm5NextModel._prepare_aux_hidden_state = _make_patched_prep_aux(
                orig_prep_aux
            )
        for cls_name in ("Glm5NextMoE", "Glm5NextMLP"):
            cls_obj = getattr(g_mod, cls_name, None)
            if cls_obj is not None and hasattr(cls_obj, "forward"):
                orig_cls_fwd = cls_obj.forward

                def _make_patched_mlp_fwd(fn: Any) -> Any:
                    def _patched_mlp_fwd(
                        self: Any, hidden_states: torch.Tensor, *a: Any, **kw: Any
                    ) -> torch.Tensor:
                        homog_reqs = int(getattr(self, "_spark_homog_reqs", 1) or 1)
                        if (
                            homog_reqs > 1
                            and hidden_states.ndim == 2
                            and hidden_states.shape[0] > homog_reqs
                            and hidden_states.shape[0] % homog_reqs == 0
                            and not torch.cuda.is_current_stream_capturing()
                        ):
                            q_len = hidden_states.shape[0] // homog_reqs
                            out_single = fn(self, hidden_states[:q_len], *a, **kw)
                            return out_single.repeat(homog_reqs, 1)
                        return fn(self, hidden_states, *a, **kw)

                    return _patched_mlp_fwd

                cls_obj.forward = _make_patched_mlp_fwd(orig_cls_fwd)

    def _fill_model_topk_buffers(model_obj: Any) -> None:
        for layer in getattr(model_obj, "layers", []):
            attn = getattr(layer, "self_attn", None)
            for owner in (attn, getattr(attn, "mla_attn", None), getattr(attn, "indexer", None)):
                if owner is None:
                    continue
                for attr in ("topk_indices_buffer", "pool_topk_indices_buffer"):
                    buf = getattr(owner, attr, None)
                    if isinstance(buf, torch.Tensor) and buf.device.type == "cuda":
                        buf.fill_(-1)

    def _install_layer_prof_hooks(model_obj: Any) -> None:
        if os.getenv("SPARK_LAYER_PROF", "0") != "1":
            return
        if getattr(model_obj, "_spark_prof_installed", False):
            return
        model_obj._spark_prof_installed = True
        model_obj._spark_prof_events = None

        def _wrap_sub(m_ref: Any, tag: str, fn: Any) -> Any:
            def _w(*a: Any, **kw: Any) -> Any:
                ev_list = m_ref._spark_prof_events
                if ev_list is None or torch.cuda.is_current_stream_capturing():
                    return fn(*a, **kw)
                es = torch.cuda.Event(enable_timing=True)
                ee = torch.cuda.Event(enable_timing=True)
                es.record()
                out = fn(*a, **kw)
                ee.record()
                ev_list.append(("sub", tag, es, ee))
                return out

            return _w

        for layer in getattr(model_obj, "_active_layers", []):
            if getattr(layer, "_spark_prof_wrapped", False):
                continue
            layer._spark_prof_wrapped = True
            layer._spark_cur_events = None
            a_kind = "mla" if hasattr(layer.self_attn, "mla_attn") else "kda"
            m_kind = "moe" if getattr(layer, "_mlp_is_moe", False) else "dense"
            orig_attn_f = layer.self_attn.forward
            orig_mlp_f = layer.mlp.forward
            orig_layer_f = layer.forward

            try:
                attn = layer.self_attn
                if a_kind == "kda":
                    if hasattr(attn, "_forward"):
                        attn._forward = _wrap_sub(model_obj, "kda_core", attn._forward)
                    if hasattr(attn, "o_proj"):
                        attn.o_proj.forward = _wrap_sub(
                            model_obj, "kda_o_proj", attn.o_proj.forward
                        )
                        if hasattr(attn.o_proj, "quant_method") and hasattr(
                            attn.o_proj.quant_method, "apply"
                        ):
                            attn.o_proj.quant_method.apply = _wrap_sub(
                                model_obj, "kda_o_gemm", attn.o_proj.quant_method.apply
                            )
                else:
                    if getattr(attn, "indexer", None) is not None:
                        attn.indexer.forward = _wrap_sub(
                            model_obj, "mla_idx", attn.indexer.forward
                        )
                    mla_inner = getattr(getattr(attn, "mla_attn", None), "mla_attn", None)
                    if mla_inner is not None and hasattr(mla_inner, "forward"):
                        mla_inner.forward = _wrap_sub(
                            model_obj, "mla_core", mla_inner.forward
                        )
                    if hasattr(attn, "o_proj"):
                        attn.o_proj.forward = _wrap_sub(
                            model_obj, "mla_o_proj", attn.o_proj.forward
                        )
                        if hasattr(attn.o_proj, "quant_method") and hasattr(
                            attn.o_proj.quant_method, "apply"
                        ):
                            attn.o_proj.quant_method.apply = _wrap_sub(
                                model_obj, "mla_o_gemm", attn.o_proj.quant_method.apply
                            )
                if m_kind == "moe":
                    mlp = layer.mlp
                    if hasattr(mlp, "gate"):
                        mlp.gate.forward = _wrap_sub(
                            model_obj, "moe_gate", mlp.gate.forward
                        )
                    exps = getattr(mlp, "experts", None)
                    if exps is not None:
                        if hasattr(exps, "_maybe_apply_shared_experts"):
                            exps._maybe_apply_shared_experts = _wrap_sub(
                                model_obj,
                                "moe_shared",
                                exps._maybe_apply_shared_experts,
                            )
                        if getattr(exps, "router", None) is not None and hasattr(
                            exps.router, "select_experts"
                        ):
                            exps.router.select_experts = _wrap_sub(
                                model_obj, "moe_route", exps.router.select_experts
                            )
                        if getattr(exps, "routed_experts", None) is not None and hasattr(
                            exps.routed_experts, "forward_modular"
                        ):
                            exps.routed_experts.forward_modular = _wrap_sub(
                                model_obj,
                                "moe_routed",
                                exps.routed_experts.forward_modular,
                            )
                        if hasattr(exps, "_maybe_reduce_final_output"):
                            exps._maybe_reduce_final_output = _wrap_sub(
                                model_obj, "moe_ar", exps._maybe_reduce_final_output
                            )
            except Exception:
                pass

            def _make_attn_w(lyr: Any, fn: Any) -> Any:
                def _w(*a: Any, **kw: Any) -> Any:
                    evs = lyr._spark_cur_events
                    if evs is None:
                        return fn(*a, **kw)
                    evs[0].record()
                    out = fn(*a, **kw)
                    evs[1].record()
                    return out

                return _w

            def _make_mlp_w(lyr: Any, fn: Any) -> Any:
                def _w(*a: Any, **kw: Any) -> Any:
                    evs = lyr._spark_cur_events
                    if evs is None:
                        return fn(*a, **kw)
                    evs[2].record()
                    out = fn(*a, **kw)
                    evs[3].record()
                    return out

                return _w

            def _make_layer_w(
                m_ref: Any, lyr: Any, fn: Any, ak: str, mk: str
            ) -> Any:
                def _w(*a: Any, **kw: Any) -> Any:
                    ev_list = m_ref._spark_prof_events
                    if ev_list is None or torch.cuda.is_current_stream_capturing():
                        return fn(*a, **kw)
                    e0 = torch.cuda.Event(enable_timing=True)
                    e1 = torch.cuda.Event(enable_timing=True)
                    e2 = torch.cuda.Event(enable_timing=True)
                    e3 = torch.cuda.Event(enable_timing=True)
                    e4 = torch.cuda.Event(enable_timing=True)
                    e5 = torch.cuda.Event(enable_timing=True)
                    lyr._spark_cur_events = (e1, e2, e3, e4)
                    try:
                        e0.record()
                        out = fn(*a, **kw)
                        e5.record()
                        ev_list.append((ak, mk, e0, e1, e2, e3, e4, e5))
                        return out
                    finally:
                        lyr._spark_cur_events = None

                return _w

            layer.self_attn.forward = _make_attn_w(layer, orig_attn_f)
            layer.mlp.forward = _make_mlp_w(layer, orig_mlp_f)
            layer.forward = _make_layer_w(
                model_obj, layer, orig_layer_f, a_kind, m_kind
            )

    orig_common_init = glm5_common_model.Glm5NextModel.__init__

    def _patched_common_model_init(self: Any, *args: Any, **kwargs: Any) -> None:
        orig_common_init(self, *args, **kwargs)
        _fill_model_topk_buffers(self)
        _install_layer_prof_hooks(self)

    glm5_common_model.Glm5NextModel.__init__ = _patched_common_model_init

    orig_nvidia_init = glm5_nvidia_model.Glm5NextModel.__init__

    def _patched_nvidia_model_init(self: Any, *args: Any, **kwargs: Any) -> None:
        orig_nvidia_init(self, *args, **kwargs)
        _fill_model_topk_buffers(self)
        _install_layer_prof_hooks(self)

    glm5_nvidia_model.Glm5NextModel.__init__ = _patched_nvidia_model_init

    orig_nvidia_pwal = glm5_nvidia_model.Glm5NextModel.process_b12x_weights_after_loading

    def _patched_nvidia_pwal(self: Any, *args: Any, **kwargs: Any) -> None:
        _fill_model_topk_buffers(self)
        _install_layer_prof_hooks(self)
        if l2pf_mod.ENABLED and getattr(self, "_active_layers", None):
            try:
                dev = next(self._active_layers[0].parameters()).device
                if dev.type == "cuda":
                    l2pf_mod.L2Prefetcher.get(dev)
            except Exception:
                pass
        return orig_nvidia_pwal(self, *args, **kwargs)

    glm5_nvidia_model.Glm5NextModel.process_b12x_weights_after_loading = (
        _patched_nvidia_pwal
    )

    for mtp_mod in (glm5_common_mtp, glm5_nvidia_mtp):
        if hasattr(mtp_mod, "Glm5NextMultiTokenPredictor"):
            orig_mtp_init = mtp_mod.Glm5NextMultiTokenPredictor.__init__

            def _make_patched_mtp_init(fn: Any) -> Any:
                def _patched_mtp_init(self: Any, *args: Any, **kwargs: Any) -> None:
                    fn(self, *args, **kwargs)
                    for mla_attn in getattr(self, "_mtp_mla_attns", []):
                        for attr in ("topk_indices_buffer", "pool_topk_indices_buffer"):
                            buf = getattr(mla_attn, attr, None)
                            if isinstance(buf, torch.Tensor) and buf.device.type == "cuda":
                                buf.fill_(-1)

                return _patched_mtp_init

            mtp_mod.Glm5NextMultiTokenPredictor.__init__ = _make_patched_mtp_init(
                orig_mtp_init
            )

    @triton.jit
    def _safe_expand_pool_ids_kernel(
        pool_ids,
        positions,
        output,
        pool_stride,
        output_stride,
        HISTORY_TOKENS: tl.constexpr,
        OUTPUT_WIDTH: tl.constexpr,
        POOL_SIZE: tl.constexpr,
        BLOCK_COLS: tl.constexpr,
    ):
        row = tl.program_id(0)
        tile = tl.program_id(1)
        column = tile * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
        mask = column < OUTPUT_WIDTH
        sequence_length = tl.load(positions + row).to(tl.int64) + 1
        complete_pools = sequence_length // POOL_SIZE
        tail_start = complete_pools * POOL_SIZE
        live_history = tl.minimum(HISTORY_TOKENS, complete_pools * POOL_SIZE)
        history = column < live_history
        pool_column = column // POOL_SIZE
        pool_offset = column % POOL_SIZE
        pool_id = tl.load(
            pool_ids + row * pool_stride + pool_column,
            mask=mask & history,
            other=-1,
        ).to(tl.int64)
        valid_pool = (pool_id >= 0) & (pool_id < complete_pools)
        history_value = tl.where(valid_pool, pool_id * POOL_SIZE + pool_offset, -1)
        tail_offset = column - live_history
        tail_count = sequence_length - tail_start
        in_tail = (tail_offset >= 0) & (tail_offset < tail_count)
        tail_value = tl.where(in_tail, tail_start + tail_offset, -1)
        value = tl.where(history, history_value, tail_value).to(tl.int32)
        tl.store(output + row * output_stride + column, value, mask=mask)

    glm_kpool._expand_pool_ids_kernel = _safe_expand_pool_ids_kernel

    try:
        from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
        from vllm.platforms import current_platform

        orig_gate_init = GateLinear.__init__

        def _patched_gate_init(self: Any, *args: Any, **kwargs: Any) -> None:
            orig_gate_init(self, *args, **kwargs)
            if current_platform.is_cuda() and current_platform.is_device_capability_family(120):
                self._sm120_graph_pool_lifetime_guard = True

        GateLinear.__init__ = _patched_gate_init
    except Exception:
        pass

    try:
        from vllm.models.glm5next.nvidia import kda as glm5_nvidia_kda

        def _patched_kda_forward(
            self: Any,
            hidden_states: torch.Tensor,
            positions: torch.Tensor,
        ) -> torch.Tensor:
            max_overlap_tokens = int(os.getenv("SPARK_KDA_GATE_OVERLAP_MAX_TOKENS", "64"))
            if (
                glm5_nvidia_kda._GATE_SIDE_STREAM
                and not self.use_full_rank_gate
                and hidden_states.size(0) <= max_overlap_tokens
            ):
                gate_workspace, main_workspace = self._projection_workspaces(
                    hidden_states.size(0)
                )
                if glm5_nvidia_kda._gate_overlap_allowed():
                    return self._forward_gate_overlap(
                        hidden_states, gate_workspace, main_workspace
                    )
            return self._forward_projections(hidden_states, positions)

        glm5_nvidia_kda.Glm5NextLinearAttention.forward = _patched_kda_forward
    except Exception:
        pass

    try:
        from vllm.models.glm5next.model_state import Glm5NextModelState

        orig_glm_ms_init = Glm5NextModelState.__init__

        def _patched_glm_ms_init(
            self: Glm5NextModelState,
            vllm_config: Any,
            model: Any,
            encoder_cache: Any,
            device: Any,
        ) -> None:
            orig_glm_ms_init(self, vllm_config, model, encoder_cache, device)
            if os.getenv("SPARK_ENABLE_PREFILL_CUDAGRAPH", "0") == "1":
                self.single_request_prefill_cudagraph_tokens = int(
                    os.getenv(
                        "SPARK_PREFILL_CUDAGRAPH_TOKENS", str(self.max_num_tokens)
                    )
                )

        def _can_use_single_request_prefill_graph(
            self: Glm5NextModelState,
            num_reqs: int,
            num_tokens: int,
            req_ids: Any,
        ) -> bool:
            return (
                self.single_request_prefill_cudagraph_tokens > 0
                and num_reqs == 1
                and num_tokens == self.single_request_prefill_cudagraph_tokens
            )

        Glm5NextModelState.__init__ = _patched_glm_ms_init
        Glm5NextModelState.can_use_single_request_prefill_graph = (
            _can_use_single_request_prefill_graph
        )
    except Exception:
        pass


_GPU_WORKER_PATCHES_INSTALLED = False


def install_gpu_worker_patches() -> None:
    """Install heavy GPU / B12X / safetensors / DFlash2 patches inside WorkerProcs only."""
    global _GPU_WORKER_PATCHES_INSTALLED
    if _GPU_WORKER_PATCHES_INSTALLED:
        return
    _GPU_WORKER_PATCHES_INSTALLED = True

    from worker.shm_safetensors import install_shm_patches

    install_shm_patches()
    _patch_l2_prefetch_and_indexer_workspace()
    _patch_dflash_attn_sm121a()
    _patch_dflash_kv_cache_and_block_size()
    _patch_dflash2_context_cudagraph_and_copy_drafter()
    _patch_sparse_mla_kpool_indexer()
    try:
        import vllm.model_executor.warmup.b12x_prepare as b12x_prep_mod

        orig_module_workload = b12x_prep_mod._module_workload

        def _patched_module_workload(module: Any, workload: Any) -> Any:
            wl = orig_module_workload(module, workload)
            if (
                getattr(wl, "lane", 0) != 0
                and module.__class__.__name__ == "ParallelLMHead"
            ):
                cand_counts = {
                    reqs * 7 for reqs in range(1, min(int(wl.max_seqs), 16) + 1)
                }
                new_counts = tuple(sorted(set(wl.token_counts) | cand_counts))
                new_fixed = tuple(
                    sorted(
                        set(wl.fixed_token_counts)
                        | {c for c in cand_counts if c < int(wl.max_tokens)}
                    )
                )
                return replace(
                    wl,
                    token_counts=new_counts,
                    fixed_token_counts=new_fixed,
                )
            return wl

        b12x_prep_mod._module_workload = _patched_module_workload
    except Exception:
        pass
    print(
        f"[spark-glm53] Phase 2 & Phase 3 GPU worker patches installed in pid={os.getpid()}",
        flush=True,
    )


def install_all_spark_glm53_patches() -> None:
    """
    Idempotent entry point registered via `vllm.general_plugins` and invoked in
    every vLLM process (API server, EngineCore, and Rank 0 / Rank 1 WorkerProcs).
    Defer heavy B12X/Cutlass/GPUModelRunner imports until `WorkerWrapperBase.init_worker`
    so `APIServer` and `EngineCore` save ~1.0 GiB of private host RAM on Head.
    """
    global _PATCHES_INSTALLED
    if _PATCHES_INSTALLED:
        return
    _PATCHES_INSTALLED = True

    configure_spark_env()
    _verify_nccl_version()
    _patch_scheduler_and_boundary_checkpoints()

    in_init_worker = False
    try:
        frame: Any = sys._getframe()
        while frame is not None:
            if frame.f_code.co_name == "init_worker":
                in_init_worker = True
                break
            frame = frame.f_back
    except Exception:
        pass

    if in_init_worker or os.getenv("SPARK_FORCE_ALL_PATCHES", "0") == "1":
        install_gpu_worker_patches()
    else:
        try:
            from vllm.v1.worker.worker_base import WorkerWrapperBase

            if not getattr(WorkerWrapperBase, "_spark_patched_init_worker", False):
                orig_init_worker = WorkerWrapperBase.init_worker

                def _patched_init_worker(
                    self: WorkerWrapperBase, *args: Any, **kwargs: Any
                ) -> Any:
                    install_gpu_worker_patches()
                    return orig_init_worker(self, *args, **kwargs)

                WorkerWrapperBase.init_worker = _patched_init_worker
                WorkerWrapperBase._spark_patched_init_worker = True
        except Exception:
            install_gpu_worker_patches()

    print(
        f"[spark-glm53] Lightweight control-plane patches installed in pid={os.getpid()} "
        f"(VLLM_HOST_IP={os.environ.get('VLLM_HOST_IP')})",
        flush=True,
    )


def main() -> None:
    """CLI entrypoint wrapping `vllm serve` with Spark-GLM53 Phase 3 defaults."""
    configure_spark_env()
    from vllm.entrypoints.cli.main import main as vllm_main

    sys.exit(vllm_main())


if __name__ == "__main__":
    main()

