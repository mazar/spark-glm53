#!/usr/bin/env python3
"""
worker/shm_safetensors.py — Zero-copy /dev/shm POSIX shared-memory weight loader for GB10 (sm_121a).

Eliminates:
  1. The 180s cold safetensors reload across worker restarts (< 2.5s warm load).
  2. The GB10 UMA Page-Cache Trap (`NVRM: NV_ERR_NO_MEMORY` in `_memdescAllocInternal`)
     by keeping all disk reads out of the Linux page cache (`O_DIRECT` + `POSIX_FADV_DONTNEED`)
     and mapping `/dev/shm/glm53_rank_{0,1}.safetensors` directly into `cuda:0` unified
     address space via `cudaHostRegister(..., cudaHostRegisterIoMemory | cudaHostRegisterMapped)`
     + `get_accelerator_view_from_cpu_tensor` with zero extra `cudaMalloc` physical pages.
"""

from __future__ import annotations

import ctypes
import gc
import json
import logging
import mmap
import os
import socket
import struct
import threading
import time
import warnings
from typing import Any

import safetensors.torch
import torch
import torch.nn as nn

warnings.filterwarnings("ignore", message=".*The given buffer is not writable.*")

logger = logging.getLogger("spark_glm53.shm_safetensors")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[shm_safetensors] %(asctime)s %(levelname)s: %(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)

# POSIX & CUDA Runtime constants
POSIX_FADV_DONTNEED = 4
MADV_HUGEPAGE = 14
CUDA_HOST_REGISTER_MAPPED = 0x02
CUDA_HOST_REGISTER_IOMEMORY = 0x04

_libc = ctypes.CDLL("libc.so.6", use_errno=True)
_libc.posix_fadvise.argtypes = [ctypes.c_int, ctypes.c_long, ctypes.c_long, ctypes.c_int]
_libc.posix_fadvise.restype = ctypes.c_int
_libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
_libc.madvise.restype = ctypes.c_int

_cudart = ctypes.CDLL("libcudart.so")


class _CudaIpcMemHandleT(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_byte * 64)]


_cudart.cudaHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
_cudart.cudaHostRegister.restype = ctypes.c_int
_cudart.cudaHostUnregister.argtypes = [ctypes.c_void_p]
_cudart.cudaHostUnregister.restype = ctypes.c_int
_cudart.cudaIpcOpenMemHandle.argtypes = [
    ctypes.POINTER(ctypes.c_void_p),
    _CudaIpcMemHandleT,
    ctypes.c_uint,
]
_cudart.cudaIpcOpenMemHandle.restype = ctypes.c_int
_cudart.cudaGetLastError.argtypes = []
_cudart.cudaGetLastError.restype = ctypes.c_int
_cudart.cudaSetDevice.argtypes = [ctypes.c_int]
_cudart.cudaSetDevice.restype = ctypes.c_int

_SAFETENSORS_TO_TORCH_DTYPE: dict[str, torch.dtype] = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I16": torch.int16,
    "U16": torch.uint16,
    "I32": torch.int32,
    "I64": torch.int64,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F64": torch.float64,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
}
if hasattr(torch, "float8_e8m0fnu"):
    _SAFETENSORS_TO_TORCH_DTYPE["F8_E8M0"] = torch.float8_e8m0fnu
if hasattr(torch, "float4_e2m1fn_x2"):
    _SAFETENSORS_TO_TORCH_DTYPE["F4_E2M1"] = torch.float4_e2m1fn_x2

_TORCH_TO_SAFETENSORS_DTYPE: dict[torch.dtype, str] = {
    v: k for k, v in _SAFETENSORS_TO_TORCH_DTYPE.items()
}

# Keep global references to mmap and registered buffers so they are never GC'd
_REGISTERED_SHM_REGIONS: dict[str, dict[str, Any]] = {}
_REG_LOCK = threading.Lock()
_PRE_REGISTER_THREAD: threading.Thread | None = None

# UMA MemorySnapshot accounting state:
# Before load_model() claims an existing /dev/shm/glm53_rank_{rank}.safetensors file,
# psutil.virtual_memory().available is lower by that file's size (~83.5 GiB).
# We add the unclaimed shm file size back to MemorySnapshot.free_memory until load_model()
# starts, so vLLM's init_snapshot and memory_profiling compute the exact model consumption.
_SHM_CLAIMED_BY_WORKER = False


def read_proc_meminfo_gib() -> dict[str, float]:
    info: dict[str, float] = {}
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0].rstrip(":")
                    if key in ("MemTotal", "MemFree", "MemAvailable", "Cached", "Shmem", "ShmemHugePages"):
                        info[key] = float(parts[1]) / (1024.0 * 1024.0)
        info["FilePageCache"] = max(0.0, info.get("Cached", 0.0) - info.get("Shmem", 0.0))
    except Exception:
        pass
    return info


def drop_file_page_cache(fd: int | None = None) -> None:
    if fd is not None and fd >= 0:
        try:
            _libc.posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)
        except Exception:
            pass
    for drop_path in ("/proc/sys/vm/drop_caches", "/host_proc/sys/vm/drop_caches"):
        if os.path.exists(drop_path):
            try:
                with open(drop_path, "w", encoding="utf-8") as f:
                    f.write("1\n")
                break
            except Exception:
                pass


def get_rank_shm_path(rank: int | None = None) -> str:
    if rank is None:
        try:
            from vllm.distributed import get_tensor_model_parallel_rank
            rank = get_tensor_model_parallel_rank()
        except Exception:
            rank = int(os.environ.get("RANK", os.environ.get("NODE_RANK", os.environ.get("LOCAL_RANK", "0"))))
    custom = os.environ.get("SPARK_GLM53_RANK_SHM_PATH")
    if custom:
        return custom.format(rank=rank)
    return f"/dev/shm/glm53_rank_{rank}.safetensors"


def get_dflash2_shm_path() -> str:
    return os.environ.get("SPARK_GLM53_DFLASH_SHM_PATH", "/dev/shm/glm53_dflash2.safetensors")


def notify_shmdaemon_lock(shm_path: str, rank: int = 0) -> None:
    sock_path = f"/tmp/spark_shm_weights_rank_{rank}.sock"
    if not os.path.exists(sock_path):
        sock_path = "/tmp/spark_shm_weights_rank_0.sock"
    if not os.path.exists(sock_path):
        return
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(10.0)
            s.connect(sock_path)
            s.sendall(f"LOCK {shm_path}\n".encode("utf-8"))
            resp = s.recv(4096).decode("utf-8", errors="replace").strip()
            logger.info("spark-shm-weights LOCK %s -> %s", shm_path, resp)
    except Exception as exc:
        logger.warning("Could not notify spark-shm-weights at %s: %s", sock_path, exc)


def _cuda_host_register_buffer(cpu_buf: torch.Tensor, device_index: int = 0) -> float:
    t0 = time.perf_counter()
    _cudart.cudaSetDevice(device_index)
    base_ptr = cpu_buf.data_ptr()
    total_size = cpu_buf.numel()
    # Try cudaHostRegisterIoMemory | cudaHostRegisterMapped (6) first per GEMINI.md spec;
    # on tmpfs/regular pages where IoMemory is unsupported (err 712), clear error and use Mapped (2).
    rc = _cudart.cudaHostRegister(
        ctypes.c_void_p(base_ptr),
        ctypes.c_size_t(total_size),
        ctypes.c_uint(CUDA_HOST_REGISTER_IOMEMORY | CUDA_HOST_REGISTER_MAPPED),
    )
    if rc != 0:
        _cudart.cudaGetLastError()
        rc = _cudart.cudaHostRegister(
            ctypes.c_void_p(base_ptr),
            ctypes.c_size_t(total_size),
            ctypes.c_uint(CUDA_HOST_REGISTER_MAPPED),
        )
        if rc != 0:
            _cudart.cudaGetLastError()
            raise RuntimeError(f"cudaHostRegister({total_size} bytes) failed with CUDA error {rc}")
    return time.perf_counter() - t0


def _ensure_shm_moe_w13_normalized(shm_path: str, tensors: dict[str, torch.Tensor]) -> None:
    """
    Ensure that b12x gated MoE FC1 weights (`w13_weight` and `w13_weight_scale`) in `shm_path`
    are flipped from vLLM's `[gate; up]` (`w31`) order to b12x kernel's `[up; gate]` (`w13`) order
    EXACTLY ONCE across the lifetime of the `/dev/shm` segment, and register all `(w13_ptr, w13_scale_ptr)`
    pairs in `b12x.moe.fused_moe._impl._W13_NORMALIZED_STORAGES` on every warm load.
    """
    try:
        from b12x.moe.fused_moe import _impl as moe_impl
    except Exception:
        return

    w13_keys = [k for k in tensors.keys() if k.endswith(".routed_experts.w13_weight")]
    if not w13_keys:
        return

    marker_path = f"{shm_path}.w13_normalized"
    already_normalized = os.path.exists(marker_path)

    if not already_normalized:
        t0 = time.perf_counter()
        for w13_key in w13_keys:
            scale_key = w13_key + "_scale"
            w13 = tensors.get(w13_key)
            w13_s = tensors.get(scale_key)
            if w13 is None or w13_s is None or w13.numel() == 0:
                continue
            n = int(w13.shape[1]) // 2
            k = int(w13.shape[2]) * 2
            moe_impl._ensure_w13_kernel_order_inplace(
                w13, w13_s, n=n, k=k, quant_mode="nvfp4"
            )
        torch.cuda.synchronize()
        try:
            with open(marker_path, "w") as mf:
                mf.write("1\n")
            os.chmod(marker_path, 0o666)
        except Exception as exc:
            logger.warning("Could not write w13 normalization marker %s: %s", marker_path, exc)
        logger.info(
            "Normalized %d MoE w13 weight/scale tensors in-place in %s in %.3fs",
            len(w13_keys),
            shm_path,
            time.perf_counter() - t0,
        )
    else:
        for w13_key in w13_keys:
            scale_key = w13_key + "_scale"
            w13 = tensors.get(w13_key)
            w13_s = tensors.get(scale_key)
            if w13 is None or w13_s is None or w13.numel() == 0:
                continue
            moe_impl._W13_NORMALIZED_STORAGES[(w13.data_ptr(), w13_s.data_ptr())] = (w13, w13_s)


def _is_shm_file_nonsparse(shm_path: str, min_bytes: int = 1024 * 1024) -> bool:
    try:
        st = os.stat(shm_path)
        if st.st_size < min_bytes:
            return False
        return (st.st_blocks * 512) >= int(st.st_size * 0.90)
    except Exception:
        return False


def _is_cuda_ipc_daemon_alive(shm_path: str) -> bool:
    ipc_json_path = f"{shm_path}.cudaipc.json"
    if not os.path.exists(ipc_json_path):
        return False
    try:
        with open(ipc_json_path, "r", encoding="utf-8") as jf:
            ipc_meta = json.load(jf)
        rank = int(ipc_meta.get("rank", 0))
    except Exception:
        return False
    sock_path = f"/tmp/spark_cuda_ipc_rank_{rank}.sock"
    if not os.path.exists(sock_path):
        return False
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(2.0)
            s.connect(sock_path)
            s.sendall(b"STATUS\n")
            resp = s.recv(4096).decode("utf-8", errors="replace").strip()
            return '"ok": true' in resp or '"ok":true' in resp
    except Exception:
        return False


def register_and_map_shm_safetensors(shm_path: str, device_index: int = 0) -> dict[str, Any]:
    """
    Map a safetensors file in /dev/shm via persistent CUDA IPC (`cudaMalloc` slabs when
    `{shm_path}.cudaipc.json` is active) or `MAP_SHARED` + `cudaHostRegister` fallback.
    """
    with _REG_LOCK:
        if shm_path in _REGISTERED_SHM_REGIONS:
            return _REGISTERED_SHM_REGIONS[shm_path]

        ipc_json_path = f"{shm_path}.cudaipc.json"
        if os.path.exists(ipc_json_path):
            try:
                t_start = time.perf_counter()
                with open(ipc_json_path, "r", encoding="utf-8") as jf:
                    ipc_meta = json.load(jf)
                if _is_cuda_ipc_daemon_alive(shm_path):
                    _cudart.cudaSetDevice(device_index)
                    target_dev = torch.device(f"cuda:{device_index}")
                    t_ipc0 = time.perf_counter()
                    slab_storages: list[Any] = []
                    slab_tensors: list[torch.Tensor] = []
                    for slab in ipc_meta["slabs"]:
                        h = _CudaIpcMemHandleT.from_buffer_copy(bytes.fromhex(slab["handle_hex"]))
                        dev_ptr = ctypes.c_void_p()
                        rc = _cudart.cudaIpcOpenMemHandle(ctypes.byref(dev_ptr), h, 1)
                        if rc != 0:
                            _cudart.cudaGetLastError()
                            raise RuntimeError(
                                f"cudaIpcOpenMemHandle(slab={slab.get('slab_idx')}) failed with CUDA error {rc}"
                            )
                        sz = int(slab["size_bytes"])
                        st = torch._C._construct_storage_from_data_pointer(dev_ptr.value, target_dev, sz)
                        t_u8 = torch.tensor([], dtype=torch.uint8, device=target_dev).set_(st)
                        slab_storages.append(st)
                        slab_tensors.append(t_u8)
                    ipc_sec = time.perf_counter() - t_ipc0

                    t_views = time.perf_counter()
                    tensors: dict[str, torch.Tensor] = {}
                    for name, tinfo in ipc_meta["tensors"].items():
                        dtype = _SAFETENSORS_TO_TORCH_DTYPE[tinfo["dtype"]]
                        shape = tinfo["shape"]
                        s_idx = int(tinfo["slab_idx"])
                        s_off = int(tinfo["slab_offset"])
                        nb = int(tinfo["nbytes"])
                        tensors[name] = slab_tensors[s_idx][s_off : s_off + nb].view(dtype).reshape(shape)
                    for name, tinfo in ipc_meta.get("empty_tensors", {}).items():
                        dtype = _SAFETENSORS_TO_TORCH_DTYPE[tinfo["dtype"]]
                        shape = tinfo["shape"]
                        tensors[name] = torch.empty(shape, dtype=dtype, device=target_dev)
                    views_sec = time.perf_counter() - t_views
                    total_sec = time.perf_counter() - t_start
                    total_bytes = int(ipc_meta.get("total_cuda_bytes", 0))
                    gib = total_bytes / (1024.0**3)
                    logger.info(
                        "Mapped %s via CUDA IPC (%d slabs, %.2f GiB, %d tensors) -> cuda:%d in %.3fs "
                        "(cudaIpcOpenMemHandle=%.3fs, views=%.3fs)",
                        shm_path,
                        len(slab_tensors),
                        gib,
                        len(tensors),
                        device_index,
                        total_sec,
                        ipc_sec,
                        views_sec,
                    )
                    entry = {
                        "path": shm_path,
                        "size": total_bytes,
                        "is_cuda_ipc": True,
                        "slab_storages": slab_storages,
                        "slab_tensors": slab_tensors,
                        "tensors": tensors,
                        "metadata": ipc_meta.get("metadata", {}) or {},
                        "reg_sec": ipc_sec,
                        "views_sec": views_sec,
                    }
                    _REGISTERED_SHM_REGIONS[shm_path] = entry
                    return entry
            except Exception as exc:
                logger.warning("CUDA IPC attach for %s failed (%s); falling back to mmap+cudaHostRegister", shm_path, exc)

        if not os.path.exists(shm_path):
            raise FileNotFoundError(shm_path)
        if not _is_shm_file_nonsparse(shm_path, min_bytes=4096):
            st_bad = os.stat(shm_path)
            raise RuntimeError(
                f"{shm_path} is sparse/hole-punched (blocks={st_bad.st_blocks * 512 / (1024**3):.2f} GiB vs "
                f"size={st_bad.st_size / (1024**3):.2f} GiB) and CUDA IPC daemon is not active; "
                f"refusing to mmap/cudaHostRegister hole-punched file."
            )

        t_start = time.perf_counter()
        needs_w13_norm = ("glm53_rank_" in shm_path) and not os.path.exists(f"{shm_path}.w13_normalized")
        if needs_w13_norm:
            try:
                fd = os.open(shm_path, os.O_RDWR)
                prot = mmap.PROT_READ | mmap.PROT_WRITE
            except PermissionError:
                fd = os.open(shm_path, os.O_RDONLY)
                prot = mmap.PROT_READ
        else:
            fd = os.open(shm_path, os.O_RDONLY)
            prot = mmap.PROT_READ
        file_size = os.fstat(fd).st_size
        mm = mmap.mmap(fd, file_size, mmap.MAP_SHARED, prot)
        os.close(fd)

        cpu_buf = torch.frombuffer(mm, dtype=torch.uint8)

        header_len = struct.unpack("<Q", mm[:8])[0]
        header = json.loads(mm[8 : 8 + header_len].decode("utf-8"))
        data_start = 8 + header_len

        reg_sec = _cuda_host_register_buffer(cpu_buf, device_index=device_index)

        from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

        t_views = time.perf_counter()
        tensors: dict[str, torch.Tensor] = {}
        metadata: dict[str, Any] = header.get("__metadata__", {}) or {}
        target_dev = torch.device(f"cuda:{device_index}")

        for name, info in header.items():
            if name == "__metadata__":
                continue
            dtype = _SAFETENSORS_TO_TORCH_DTYPE[info["dtype"]]
            shape = info["shape"]
            start_off, end_off = info["data_offsets"]
            if end_off == start_off:
                tensors[name] = torch.empty(shape, dtype=dtype, device=target_dev)
            else:
                cpu_slice = cpu_buf[data_start + start_off : data_start + end_off]
                cuda_view = get_accelerator_view_from_cpu_tensor(cpu_slice)
                tensors[name] = cuda_view.view(dtype).reshape(shape)

        views_sec = time.perf_counter() - t_views
        total_sec = time.perf_counter() - t_start
        gib = file_size / (1024.0 ** 3)
        logger.info(
            "Mapped %s (%.2f GiB, %d tensors) -> cuda:%d in %.3fs "
            "(cudaHostRegister=%.3fs @ %.1f GiB/s, views=%.3fs)",
            shm_path,
            gib,
            len(tensors),
            device_index,
            total_sec,
            reg_sec,
            gib / max(reg_sec, 1e-6),
            views_sec,
        )

        entry = {
            "path": shm_path,
            "size": file_size,
            "mm": mm,
            "cpu_buf": cpu_buf,
            "tensors": tensors,
            "metadata": metadata,
            "reg_sec": reg_sec,
            "views_sec": views_sec,
        }
        _REGISTERED_SHM_REGIONS[shm_path] = entry
        return entry


def start_background_preregister(rank: int = 0, device_index: int = 0) -> None:
    """
    Launch a background thread to run cudaHostRegister on /dev/shm/glm53_rank_{rank}.safetensors
    while Gloo/NCCL distributed init and meta-model construction are running.
    When CUDA IPC slabs are active (`{shm_path}.cudaipc.json`), skip background thread
    because `cudaIpcOpenMemHandle` runs on the main CUDA thread in ~75ms.
    """
    global _PRE_REGISTER_THREAD
    shm_path = get_rank_shm_path(rank)
    if not os.path.exists(shm_path):
        return
    if _is_cuda_ipc_daemon_alive(shm_path) or not _is_shm_file_nonsparse(shm_path):
        return
    if _PRE_REGISTER_THREAD is not None:
        return

    def _worker() -> None:
        try:
            register_and_map_shm_safetensors(shm_path, device_index=device_index)
            dflash_shm = get_dflash2_shm_path()
            if os.path.exists(dflash_shm):
                register_and_map_shm_safetensors(dflash_shm, device_index=device_index)
        except Exception as exc:
            logger.warning("Background pre-registration failed: %s", exc)

    _PRE_REGISTER_THREAD = threading.Thread(target=_worker, name="shm-preregister", daemon=True)
    _PRE_REGISTER_THREAD.start()


def _is_runtime_scratch_buffer(name: str) -> bool:
    """
    Return True for mutable per-batch runtime scratch buffers that must stay in normal
    device memory rather than shared /dev/shm weight pages.
    """
    scratch_tokens = (
        ".indexer._tail",
        ".indexer._tail_snapshot",
        ".indexer.scratch.",
        ".indexer.indexer_op._active_width",
        "._b12x_kda_",
    )
    return any(tok in name for tok in scratch_tokens)


def _fix_glm53_indexer_buffers(model: nn.Module) -> None:
    """
    Enforce GEMINI.md Section 3 Rule 3:
    Initialize sparse-MLA kpool indexer top-k buffers to -1 so unwritten entries
    never gather garbage KV pages and produce intermittent NaNs.
    """
    for mod in model.modules():
        for attr in ("topk_indices_buffer", "pool_topk_indices_buffer"):
            buf = getattr(mod, attr, None)
            if isinstance(buf, torch.Tensor) and not buf.is_meta and buf.dtype in (torch.int32, torch.int64):
                buf.fill_(-1)


def export_and_rebind_model_to_shm(model: nn.Module, shm_path: str, rank: int = 0, device_index: int = 0) -> None:
    """
    Export all post-processed rank-local parameters and weight buffers of `model`
    into `shm_path` in Safetensors format in lockstep (`copy_` -> `resize_(0)` -> `empty_cache()`),
    then `cudaHostRegister` and rebind all model parameters/buffers to the zero-copy `/dev/shm` views.
    """
    t0 = time.perf_counter()
    mem_before = read_proc_meminfo_gib()
    logger.info(
        "Exporting post-processed rank %d weights to %s in lockstep (MemAvailable=%.2f GiB, FilePageCache=%.2f GiB)...",
        rank,
        shm_path,
        mem_before.get("MemAvailable", 0.0),
        mem_before.get("FilePageCache", 0.0),
    )

    # Collect all parameters and weight buffers
    items_to_export: list[tuple[str, torch.Tensor, bool]] = []  # (name, tensor, is_param)
    for name, param in model.named_parameters():
        items_to_export.append((name, param, True))
    for name, buf in model.named_buffers():
        if _is_runtime_scratch_buffer(name):
            continue
        items_to_export.append((name, buf, False))

    # Build Safetensors header with 256-byte alignment for every non-empty tensor
    header_dict: dict[str, Any] = {
        "__metadata__": {
            "format": "pt",
            "spark_glm53_preprocessed": "true",
            "rank": str(rank),
        }
    }
    curr_offset = 0
    for name, tensor, _ in items_to_export:
        st_dtype = _TORCH_TO_SAFETENSORS_DTYPE.get(tensor.dtype)
        if st_dtype is None:
            raise ValueError(f"Unsupported dtype {tensor.dtype} on tensor {name}")
        nbytes = tensor.numel() * tensor.element_size()
        if nbytes > 0:
            curr_offset = ((curr_offset + 255) // 256) * 256
        start_off = curr_offset
        end_off = curr_offset + nbytes
        header_dict[name] = {
            "dtype": st_dtype,
            "shape": list(tensor.shape),
            "data_offsets": [start_off, end_off],
        }
        curr_offset = end_off

    header_json = json.dumps(header_dict, separators=(",", ":")).encode("utf-8")
    # Pad header_json with spaces so 8 + len(header_json) is a multiple of 256 bytes
    pad_len = (256 - ((8 + len(header_json)) % 256)) % 256
    header_json += b" " * pad_len
    header_len = len(header_json)
    data_start = 8 + header_len
    total_file_size = data_start + curr_offset

    tmp_path = f"{shm_path}.tmp.{os.getpid()}"
    fd = os.open(tmp_path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o666)
    os.ftruncate(fd, total_file_size)
    mm = mmap.mmap(fd, total_file_size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
    os.close(fd)

    cpu_buf = torch.frombuffer(mm, dtype=torch.uint8)

    mm[:8] = struct.pack("<Q", header_len)
    mm[8:data_start] = header_json

    # Count references to each underlying storage_ptr in case any tensors alias storage
    storage_refs: dict[int, int] = {}
    for _, tensor, _ in items_to_export:
        if tensor.numel() > 0:
            sptr = tensor.untyped_storage().data_ptr()
            storage_refs[sptr] = storage_refs.get(sptr, 0) + 1

    try:
        # Lockstep copy from CUDA tensor -> CPU staging tensor -> /dev/shm mmap and immediate resize_(0) + empty_cache()
        freed_since_flush = 0
        flush_threshold = 2 * 1024 * 1024 * 1024  # 2 GiB
        for name, tensor, _ in items_to_export:
            start_off, end_off = header_dict[name]["data_offsets"]
            nbytes = end_off - start_off
            if nbytes > 0:
                sptr = tensor.untyped_storage().data_ptr()
                src_cpu = tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu()
                dst_u8 = cpu_buf[data_start + start_off : data_start + end_off]
                dst_u8.copy_(src_cpu, non_blocking=False)
                del src_cpu
                storage_refs[sptr] = storage_refs.get(sptr, 1) - 1
                if storage_refs[sptr] <= 0:
                    try:
                        tensor.untyped_storage().resize_(0)
                    except Exception:
                        pass
                freed_since_flush += nbytes
                if freed_since_flush >= flush_threshold:
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    freed_since_flush = 0

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        mm.close()
        try:
            os.unlink(f"{shm_path}.w13_normalized")
        except FileNotFoundError:
            pass
        os.replace(tmp_path, shm_path)
        os.chmod(shm_path, 0o666)
    except Exception:
        try:
            mm.close()
        except Exception:
            pass
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        raise

    export_sec = time.perf_counter() - t0
    logger.info(
        "Exported %.2f GiB (%d tensors) to %s in %.2fs. Pinning via spark-shm-weights...",
        total_file_size / (1024.0 ** 3),
        len(items_to_export),
        shm_path,
        export_sec,
    )

    if os.getenv("SPARK_USE_CUDA_IPC", "1") != "1":
        notify_shmdaemon_lock(shm_path, rank=rank)
    drop_file_page_cache()

    # Now map the newly written /dev/shm file with cudaHostRegister and rebind model tensors
    entry = register_and_map_shm_safetensors(shm_path, device_index=device_index)
    _ensure_shm_moe_w13_normalized(shm_path, entry["tensors"])
    _bind_shm_views_to_model(
        model,
        entry["tensors"],
        torch.device(f"cuda:{device_index}"),
        is_cold_rebind=True,
        is_cuda_ipc=bool(entry.get("is_cuda_ipc", False)),
    )


def _bind_shm_views_to_model(
    model: nn.Module,
    views_dict: dict[str, torch.Tensor],
    target_device: torch.device,
    is_cold_rebind: bool = False,
    is_cuda_ipc: bool = False,
) -> None:
    """
    Bind all zero-copy `cuda:0` views from `views_dict` onto `model` parameters and buffers,
    including any shared/aliased Parameter or buffer references across submodules.
    """
    modules_dict = dict(model.named_modules())

    param_aliases: dict[int, list[tuple[nn.Module, str]]] = {}
    buffer_aliases: dict[int, list[tuple[nn.Module, str]]] = {}
    obj_attr_aliases: dict[int, list[tuple[Any, str]]] = {}
    for m in modules_dict.values():
        for pname, pval in m._parameters.items():
            if pval is not None:
                param_aliases.setdefault(id(pval), []).append((m, pname))
        for bname, bval in m._buffers.items():
            if bval is not None:
                buffer_aliases.setdefault(id(bval), []).append((m, bname))
        for k, v in m.__dict__.items():
            if k in ("_parameters", "_buffers", "_modules"):
                continue
            if isinstance(v, torch.Tensor):
                obj_attr_aliases.setdefault(id(v), []).append((m, k))
            elif not isinstance(v, nn.Module) and hasattr(v, "__dict__"):
                for sk, sv in v.__dict__.items():
                    if isinstance(sv, torch.Tensor):
                        obj_attr_aliases.setdefault(id(sv), []).append((v, sk))

    materialize_dense = (
        (not is_cuda_ipc)
        and (not is_cold_rebind)
        and (os.getenv("SPARK_SHM_MATERIALIZE_DENSE", "0") == "1")
    )
    for full_name, view_tensor in views_dict.items():
        if "." in full_name:
            mod_name, attr_name = full_name.rsplit(".", 1)
        else:
            mod_name, attr_name = "", full_name
        mod = modules_dict.get(mod_name)
        if mod is None:
            continue

        # Always materialize the 0.10 GiB coalesced _shm_mxfp8_scale_mma permutation,
        # and when SPARK_SHM_MATERIALIZE_DENSE=1 (default), materialize the 3.32 GiB of
        # B12xBlockscaledLinear MXFP8 values + 1.43 GiB of dense/MLA/router/lm_head weights
        # into cudaMalloc device memory (6.37x faster M=8192 prefill & 1.89x faster M=64 decode),
        # while keeping the 79.73 GiB routed_experts + 0.59 GiB embed_tokens + unused scale_rows/visual
        # in zero-copy /dev/shm.
        if attr_name == "_shm_mxfp8_scale_mma" and view_tensor.ndim == 6:
            view_tensor = (
                view_tensor.permute(5, 2, 4, 0, 1, 3)
                .contiguous()
                .permute(3, 4, 1, 5, 2, 0)
            )
        elif (
            materialize_dense
            and view_tensor.numel() > 0
            and attr_name != "_shm_mxfp8_scale_rows"
            and "routed_experts." not in full_name
            and not full_name.endswith("embed_tokens.weight")
            and not full_name.endswith("lm_head.weight")
            and not full_name.startswith("visual.")
            and not (
                attr_name in ("weight", "weight_scale")
                and f"{mod_name}._shm_mxfp8_values" in views_dict
            )
        ):
            view_tensor = view_tensor.clone()

        if attr_name.startswith("_shm_mxfp8_"):
            mod.register_buffer(attr_name, view_tensor, persistent=True)
        elif attr_name in mod._parameters and mod._parameters[attr_name] is not None:
            old_param = mod._parameters[attr_name]
            old_id = id(old_param)
            if is_cold_rebind or old_param.is_meta or old_param.shape != view_tensor.shape:
                new_param = nn.Parameter(view_tensor, requires_grad=False)
                for k, v in old_param.__dict__.items():
                    if k not in new_param.__dict__:
                        setattr(new_param, k, v)
                for alias_mod, alias_attr in param_aliases.get(old_id, [(mod, attr_name)]):
                    alias_mod._parameters[alias_attr] = new_param
                for holder_obj, holder_attr in obj_attr_aliases.get(old_id, []):
                    setattr(holder_obj, holder_attr, new_param)
            else:
                old_param.data = view_tensor
        elif attr_name in mod._buffers:
            old_buf = mod._buffers[attr_name]
            old_id = id(old_buf) if old_buf is not None else -1
            for alias_mod, alias_attr in buffer_aliases.get(old_id, [(mod, attr_name)]):
                alias_mod._buffers[alias_attr] = view_tensor
            for holder_obj, holder_attr in obj_attr_aliases.get(old_id, []):
                setattr(holder_obj, holder_attr, view_tensor)
        else:
            # Post-processed parameters added during process_weights_after_loading (e.g. W_UV, W_UK_T)
            mod.register_parameter(attr_name, nn.Parameter(view_tensor, requires_grad=False))

    # Materialize any remaining meta buffers or plain tensor attributes (e.g. empty kv_cache or RotaryEmbedding)
    for mod_name, mod in modules_dict.items():
        for buf_name, buf in list(mod._buffers.items()):
            if isinstance(buf, torch.Tensor) and buf.is_meta:
                if buf_name == "cos_sin_cache" and hasattr(mod, "_compute_cos_sin_cache"):
                    mod._buffers[buf_name] = mod._compute_cos_sin_cache().to(target_device)
                elif buf_name in ("_k_scale", "_v_scale", "_q_scale", "_prob_scale"):
                    mod._buffers[buf_name] = torch.tensor(1.0, dtype=torch.float32, device=target_device)
                else:
                    mod._buffers[buf_name] = torch.zeros(buf.shape, dtype=buf.dtype, device=target_device)
        for attr_name, attr_val in list(mod.__dict__.items()):
            if isinstance(attr_val, torch.Tensor) and attr_val.is_meta:
                if attr_name.endswith("_scale_cpu"):
                    setattr(mod, attr_name, torch.tensor(1.0, dtype=attr_val.dtype, device="cpu"))
                else:
                    setattr(mod, attr_name, torch.empty(attr_val.shape, dtype=attr_val.dtype, device=target_device))
        router = getattr(mod, "router", None)
        rexp = getattr(mod, "routed_experts", None)
        if (
            router is not None
            and rexp is not None
            and getattr(router, "e_score_correction_bias", None) is not None
            and getattr(rexp, "e_score_correction_bias", None) is not None
        ):
            router.e_score_correction_bias = rexp.e_score_correction_bias

    if is_cold_rebind:
        # Refresh B12xMxfp8LinearKernel and B12xExperts holders so they point to the new /dev/shm views
        from b12x.gemm._shared.wo_mxfp8 import MXFP8Rows
        from b12x.gemm.blockscaled._linear import MXFP8LinearWeight
        from vllm.model_executor.kernels.linear.b12x_blockscaled import B12xBlockscaledLinear
        from vllm.utils.b12x import b12x_layer_prefix

        for mod in modules_dict.values():
            if hasattr(mod, "_shm_mxfp8_values") and getattr(mod, "b12x_mxfp8_packed_weight", None) is not None:
                in_f, pad_in_f, out_f = map(int, mod._shm_mxfp8_meta.tolist())
                s_mma = mod._shm_mxfp8_scale_mma
                if s_mma.ndim == 6 and not s_mma.permute(5, 2, 4, 0, 1, 3).is_contiguous():
                    s_mma = s_mma.permute(5, 2, 4, 0, 1, 3).contiguous().permute(3, 4, 1, 5, 2, 0)
                mod.b12x_mxfp8_packed_weight = MXFP8LinearWeight(
                    weight=MXFP8Rows(
                        values=mod._shm_mxfp8_values,
                        scale_rows=mod._shm_mxfp8_scale_rows,
                        scale_mma=s_mma,
                    ),
                    in_features=in_f,
                    padded_in_features=pad_in_f,
                    out_features=out_f,
                )
                name = b12x_layer_prefix(mod)
                mod.b12x_linear = B12xBlockscaledLinear(
                    mod.b12x_mxfp8_packed_weight,
                    recipe="mxfp8",
                    activation_mode=mod.b12x_activation_mode,
                    layer_name=name,
                )
            qm = getattr(mod, "quant_method", None)
            if qm is not None and hasattr(qm, "moe_kernel") and qm.moe_kernel is not None:
                if hasattr(mod, "_b12x_prepared_experts"):
                    mod._b12x_prepared_experts = None
                qm.moe_kernel.fused_experts.process_weights_after_loading(mod)
        if hasattr(model, "process_weights_after_loading"):
            model.process_weights_after_loading()


def install_shm_patches() -> None:
    """
    Install all monkey-patches for:
      1. safetensors.torch.load_file (zero-copy /dev/shm for DFlash2 + POSIX_FADV_DONTNEED for disk files)
      2. ModelOptNvFp4FusedMoE.process_weights_after_loading (support B12X with pre-processed weights)
      3. ModelOptLinearMethod.process_weights_after_loading & B12xMxfp8LinearKernel.process_weights_after_loading
      4. MLAAttention.process_weights_after_loading (skip W_UV/W_UK_T recomputation when pre-processed)
      5. MemorySnapshot.measure (accurate UMA free memory accounting when /dev/shm holds persistent weights)
      6. B12xModelLoader.load_model & DefaultModelLoader.load_model (< 2.5s warm load from /dev/shm)
    """
    if getattr(install_shm_patches, "_installed", False):
        return
    install_shm_patches._installed = True

    # 1. Patch torch.cuda.set_device and init_distributed_environment to kick off background
    #    cudaHostRegister during the 10s Gloo/NCCL rendezvous (well before load_model starts).
    _orig_set_device = torch.cuda.set_device

    def _infer_cli_rank() -> int:
        import sys
        for i, arg in enumerate(sys.argv):
            if arg == "--node-rank" and i + 1 < len(sys.argv):
                try:
                    return int(sys.argv[i + 1])
                except ValueError:
                    pass
            elif arg.startswith("--node-rank="):
                try:
                    return int(arg.split("=", 1)[1])
                except ValueError:
                    pass
        return int(os.environ.get("RANK", os.environ.get("NODE_RANK", os.environ.get("LOCAL_RANK", "0"))))

    def _patched_set_device(device: Any) -> None:
        _orig_set_device(device)
        try:
            dev_idx = torch.cuda.current_device()
            start_background_preregister(rank=_infer_cli_rank(), device_index=dev_idx)
        except Exception:
            pass

    torch.cuda.set_device = _patched_set_device

    try:
        import vllm.distributed.parallel_state as _ps_mod

        _orig_init_dist = _ps_mod.init_distributed_environment

        def _patched_init_dist(
            world_size: int = -1,
            rank: int = -1,
            distributed_init_method: str = "env://",
            local_rank: int = -1,
            backend: str = "nccl",
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            try:
                eff_rank = rank if rank >= 0 else _infer_cli_rank()
                dev_idx = local_rank if local_rank >= 0 else 0
                start_background_preregister(rank=eff_rank, device_index=dev_idx)
            except Exception:
                pass
            return _orig_init_dist(
                world_size,
                rank,
                distributed_init_method,
                local_rank,
                backend,
                *args,
                **kwargs,
            )

        _ps_mod.init_distributed_environment = _patched_init_dist
    except Exception:
        pass

    # 2. Patch safetensors.torch.load_file for DFlash2 /dev/shm zero-copy and disk page-cache eviction
    _orig_load_file = safetensors.torch.load_file

    def _patched_load_file(filename: str | os.PathLike, device: str | int = "cpu") -> dict[str, torch.Tensor]:
        fpath = os.fspath(filename)
        dflash_shm = get_dflash2_shm_path()
        if ("DFlash2" in fpath or "dflash" in fpath.lower()) and os.path.exists(dflash_shm):
            fpath = dflash_shm

        if fpath.startswith("/dev/shm/") and os.path.exists(fpath):
            dev_idx = torch.cuda.current_device() if torch.cuda.is_available() else 0
            entry = register_and_map_shm_safetensors(fpath, device_index=dev_idx)
            # Return shallow clones of views so caller mutations to parameter metadata don't alias
            return dict(entry["tensors"])

        res = _orig_load_file(fpath, device=device)
        try:
            fd = os.open(fpath, os.O_RDONLY)
            drop_file_page_cache(fd)
            os.close(fd)
        except Exception:
            pass
        return res

    safetensors.torch.load_file = _patched_load_file

    # Also patch safetensors.safe_open so any iterator-based disk reads drop page cache on exit
    _orig_safe_open = safetensors.safe_open

    class _ShmSafeOpenView:
        def __init__(self, shm_path: str):
            dev_idx = torch.cuda.current_device() if torch.cuda.is_available() else 0
            self._entry = register_and_map_shm_safetensors(shm_path, device_index=dev_idx)
            self._tensors: dict[str, torch.Tensor] = self._entry["tensors"]
            self._meta: dict[str, Any] = self._entry.get("metadata", {}) or {}

        def keys(self):
            return list(self._tensors.keys())

        def get_tensor(self, name: str) -> torch.Tensor:
            return self._tensors[name]

        def metadata(self) -> dict[str, Any]:
            return dict(self._meta)

    class _PageCacheCleanSafeOpen:
        def __init__(self, filename: Any, framework: str, device: Any = "cpu"):
            fpath = os.fspath(filename)
            dflash_shm = get_dflash2_shm_path()
            if ("DFlash2" in fpath or "dflash" in fpath.lower()) and os.path.exists(dflash_shm):
                fpath = dflash_shm
            self._filename = fpath
            if self._filename.startswith("/dev/shm/") and os.path.exists(self._filename):
                self._shm_view: _ShmSafeOpenView | None = _ShmSafeOpenView(self._filename)
                self._ctx = None
            else:
                self._shm_view = None
                self._ctx = _orig_safe_open(self._filename, framework=framework, device=device)

        def __enter__(self):
            if self._shm_view is not None:
                return self._shm_view
            return self._ctx.__enter__()

        def __exit__(self, exc_type, exc_val, exc_tb):
            if self._shm_view is not None:
                return False
            ret = self._ctx.__exit__(exc_type, exc_val, exc_tb)
            if not self._filename.startswith("/dev/shm/"):
                try:
                    fd = os.open(self._filename, os.O_RDONLY)
                    _libc.posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)
                    os.close(fd)
                except Exception:
                    pass
            return ret

    safetensors.safe_open = _PageCacheCleanSafeOpen  # type: ignore[assignment]
    from vllm.model_executor.model_loader import weight_utils as _weight_utils_mod
    _weight_utils_mod.safe_open = _PageCacheCleanSafeOpen  # type: ignore[assignment]

    # 3. Patch ModelOptNvFp4FusedMoE.process_weights_after_loading & ModelOptLinearMethod.process_weights_after_loading
    from vllm.model_executor.layers.quantization import modelopt as _modelopt_mod
    from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
    from vllm.model_executor.utils import (
        is_weights_pre_processed,
        weights_already_processed,
    )

    QuantizeMethodBase.supports_pre_processed_weights = True
    _modelopt_mod.ModelOptLinearMethod.supports_pre_processed_weights = True  # type: ignore[assignment]

    _orig_moe_pwal = _modelopt_mod.ModelOptNvFp4FusedMoE.process_weights_after_loading

    def _patched_moe_pwal(self, layer: Any) -> None:
        if is_weights_pre_processed():
            self._restore_padded_moe_dims(layer)
            self._build_moe_kernel(layer)
            return
        _orig_moe_pwal(self, layer)

    _modelopt_mod.ModelOptNvFp4FusedMoE.process_weights_after_loading = _patched_moe_pwal

    _orig_linear_pwal = _modelopt_mod.ModelOptLinearMethod.process_weights_after_loading

    def _patched_linear_pwal(self, layer: Any) -> None:
        if is_weights_pre_processed():
            layer.is_w4a16_nvfp4 = (
                self.spec.weight == _modelopt_mod.kNvfp4Static and self.spec.activation is None
            )
            if self.spec.weight == _modelopt_mod.kMxfp8Static and getattr(layer, "is_bmm", False):
                self.kernel = _modelopt_mod.init_mxfp8_linear_kernel(bmm_batch_size=layer.bmm_batch_size)
            self.kernel.process_weights_after_loading(layer)
            return
        _orig_linear_pwal(self, layer)

    _modelopt_mod.ModelOptLinearMethod.process_weights_after_loading = _patched_linear_pwal

    # 4. Patch B12xMxfp8LinearKernel.process_weights_after_loading
    from b12x.gemm._shared.wo_mxfp8 import MXFP8Rows
    from b12x.gemm.blockscaled._linear import MXFP8LinearWeight
    from vllm.model_executor.kernels.linear.b12x_blockscaled import B12xBlockscaledLinear
    from vllm.model_executor.kernels.linear.mxfp8.b12x import B12xMxfp8LinearKernel
    from vllm.utils.b12x import (
        b12x_layer_prefix,
        get_b12x_dense_activation_mode,
        register_b12x_layer,
        set_b12x_preparation_provider,
    )
    from vllm.utils.torch_utils import _encode_layer_name

    _orig_mxfp8_pwal = B12xMxfp8LinearKernel.process_weights_after_loading

    def _patched_mxfp8_pwal(self, layer: torch.nn.Module) -> None:
        if is_weights_pre_processed():
            if not hasattr(layer, "_shm_mxfp8_values"):
                weight = getattr(layer, "weight", None)
                scales = getattr(layer, "weight_scale", None)
                if (
                    weight is None
                    or weight.numel() == 0
                    or scales is None
                    or scales.numel() == 0
                ):
                    # e.g., kv_b_proj whose packed weight was intentionally released after absorption
                    return
                # Online-quantized layer (e.g., ParallelLMHead with VLLM_MXFP8_LM_HEAD=1)
                _orig_mxfp8_pwal(self, layer)
                return
            in_features, padded_in_features, out_features = map(int, layer._shm_mxfp8_meta.tolist())
            s_mma = layer._shm_mxfp8_scale_mma
            if s_mma.ndim == 6 and not s_mma.permute(5, 2, 4, 0, 1, 3).is_contiguous():
                s_mma = s_mma.permute(5, 2, 4, 0, 1, 3).contiguous().permute(3, 4, 1, 5, 2, 0)
            packed = MXFP8LinearWeight(
                weight=MXFP8Rows(
                    values=layer._shm_mxfp8_values,
                    scale_rows=layer._shm_mxfp8_scale_rows,
                    scale_mma=s_mma,
                ),
                in_features=in_features,
                padded_in_features=padded_in_features,
                out_features=out_features,
            )
            layer.b12x_mxfp8_packed_weight = packed
            layer.b12x_activation_mode = getattr(
                layer, "b12x_activation_mode", None
            ) or get_b12x_dense_activation_mode("mxfp8")
            layer.b12x_bf16_input_supported = (
                in_features % 128 == 0 and out_features % 8 == 0
            )
            name = b12x_layer_prefix(layer)
            layer.b12x_linear = B12xBlockscaledLinear(
                layer.b12x_mxfp8_packed_weight,
                recipe="mxfp8",
                activation_mode=layer.b12x_activation_mode,
                layer_name=name,
            )
            layer.b12x_layer_name = _encode_layer_name(name)
            register_b12x_layer(name, layer)
            if not getattr(layer, "b12x_preparation_suppressed", False):
                set_b12x_preparation_provider(layer, self)
            return

        _orig_mxfp8_pwal(self, layer)
        pw = getattr(layer, "b12x_mxfp8_packed_weight", None)
        if pw is not None:
            layer.register_buffer("_shm_mxfp8_values", pw.weight.values, persistent=True)
            layer.register_buffer("_shm_mxfp8_scale_rows", pw.weight.scale_rows, persistent=True)
            layer.register_buffer("_shm_mxfp8_scale_mma", pw.weight.scale_mma, persistent=True)
            layer.register_buffer(
                "_shm_mxfp8_meta",
                torch.tensor(
                    [pw.in_features, pw.padded_in_features, pw.out_features],
                    dtype=torch.int64,
                    device=pw.weight.values.device,
                ),
                persistent=True,
            )

    B12xMxfp8LinearKernel.process_weights_after_loading = _patched_mxfp8_pwal

    # 5. Patch MLAAttention.process_weights_after_loading
    from vllm.model_executor.layers.attention import mla_attention as _mla_mod

    _orig_mla_pwal = _mla_mod.MLAAttention.process_weights_after_loading

    def _patched_mla_pwal(self, act_dtype: torch.dtype) -> None:
        if is_weights_pre_processed():
            self.impl.process_weights_after_loading(act_dtype)
            weight = getattr(self, "W_UK_T", None)
            if (
                isinstance(weight, torch.Tensor)
                and weight.dtype == torch.bfloat16
                and _mla_mod.can_implement_bf16_mla_query(
                    num_heads=int(weight.shape[0]),
                    max_m=1,
                    nope_dim=int(weight.shape[1]),
                    latent_dim=int(weight.shape[2]),
                    output_dtype=torch.bfloat16,
                    device=weight.device,
                )
            ):
                self._b12x_query_plans = {}
                self._b12x_query_prefix = f"{self.layer_name}.mla_query"
                register_b12x_layer(self._b12x_query_prefix, self)
                self._b12x_query_layer_name = _encode_layer_name(self._b12x_query_prefix)
                set_b12x_preparation_provider(self, self)
            if self.impl.is_sparse and self.prefill_backend is None:
                self.kv_b_proj.b12x_mxfp8_packed_weight = None
                self.kv_b_proj.b12x_linear = None
                set_b12x_preparation_provider(self.kv_b_proj, None)
            return

        _orig_mla_pwal(self, act_dtype)
        # If kv_b_proj released its packed weight after W_UK_T/W_UV absorption, drop its _shm_mxfp8_* buffers too
        if getattr(self.kv_b_proj, "b12x_mxfp8_packed_weight", None) is None:
            for bname in ("_shm_mxfp8_values", "_shm_mxfp8_scale_rows", "_shm_mxfp8_scale_mma", "_shm_mxfp8_meta"):
                self.kv_b_proj._buffers.pop(bname, None)

    _mla_mod.MLAAttention.process_weights_after_loading = _patched_mla_pwal

    # 6. Patch MemorySnapshot.measure for accurate UMA accounting when /dev/shm already holds rank weights
    from vllm.utils import mem_utils as _mem_utils

    _orig_measure = _mem_utils.MemorySnapshot.measure

    def _patched_measure(self) -> None:
        _orig_measure(self)
        if not _SHM_CLAIMED_BY_WORKER:
            shm_sz = 0
            rank_p = get_rank_shm_path()
            if os.path.exists(rank_p):
                if _is_cuda_ipc_daemon_alive(rank_p):
                    try:
                        st = os.stat(rank_p)
                        shm_sz += int(st.st_size) + int(st.st_blocks * 512) + (640 * 1024 * 1024)
                    except Exception:
                        pass
                elif _is_shm_file_nonsparse(rank_p):
                    try:
                        shm_sz += os.path.getsize(rank_p)
                    except Exception:
                        pass
            dflash_p = get_dflash2_shm_path()
            if os.path.exists(dflash_p) and _is_shm_file_nonsparse(dflash_p):
                try:
                    shm_sz += os.path.getsize(dflash_p)
                except Exception:
                    pass
            if shm_sz > 0:
                self.free_memory = min(int(self.total_memory), int(self.free_memory) + int(shm_sz))
                self.cuda_memory = self.total_memory - self.free_memory
                self.non_torch_memory = self.cuda_memory - self.torch_memory

    _mem_utils.MemorySnapshot.measure = _patched_measure

    # 7. Patch B12xModelLoader.load_model and DefaultModelLoader.load_model
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
    from vllm.model_executor.model_loader.utils import (
        initialize_model,
        process_weights_after_loading,
    )
    from vllm.utils.torch_utils import set_default_torch_dtype

    _orig_default_load_model = DefaultModelLoader.load_model
    try:
        from b12x.integration.vllm.loader import B12xModelLoader
        _orig_b12x_load_model = B12xModelLoader.load_model
    except Exception:
        B12xModelLoader = DefaultModelLoader
        _orig_b12x_load_model = _orig_default_load_model

    def _call_orig_loader(loader_self: Any, vllm_config: Any, model_config: Any, prefix: str) -> nn.Module:
        loader_self._in_shm_cold_load = True
        try:
            if B12xModelLoader is not DefaultModelLoader and isinstance(loader_self, B12xModelLoader):
                return _orig_b12x_load_model(loader_self, vllm_config=vllm_config, model_config=model_config, prefix=prefix)
            return _orig_default_load_model(loader_self, vllm_config=vllm_config, model_config=model_config, prefix=prefix)
        finally:
            loader_self._in_shm_cold_load = False

    def _shm_load_model(
        self,
        vllm_config: Any,
        model_config: Any,
        prefix: str = "",
    ) -> nn.Module:
        global _SHM_CLAIMED_BY_WORKER
        if getattr(self, "_in_shm_cold_load", False):
            return _orig_default_load_model(self, vllm_config=vllm_config, model_config=model_config, prefix=prefix)
        arch = getattr(model_config, "architectures", []) or []
        is_glm53_target = any("Glm5Next" in a for a in arch) and not prefix
        if not is_glm53_target:
            t_draft0 = time.perf_counter()
            default_loader = DefaultModelLoader(self.load_config)
            default_loader._in_shm_cold_load = True
            model = _orig_default_load_model(
                default_loader, vllm_config=vllm_config, model_config=model_config, prefix=prefix
            )
            _fix_glm53_indexer_buffers(model)
            # DFlash2 copies its weights into device parameters during DefaultModelLoader.load_model,
            # so we can release the cudaHostRegister mapping on /dev/shm/glm53_dflash2.safetensors.
            dflash_shm = get_dflash2_shm_path()
            with _REG_LOCK:
                d_entry = _REGISTERED_SHM_REGIONS.pop(dflash_shm, None)
            if d_entry is not None:
                try:
                    torch.cuda.synchronize()
                    _cudart.cudaHostUnregister(ctypes.c_void_p(d_entry["cpu_buf"].data_ptr()))
                    d_entry["tensors"].clear()
                    del d_entry["cpu_buf"]
                    d_entry["mm"].close()
                except Exception:
                    pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            drop_file_page_cache()
            logger.info(
                "Loaded auxiliary/draft model (%s) via /dev/shm safe_open in %.3fs",
                ",".join(arch) or "unknown",
                time.perf_counter() - t_draft0,
            )
            return model

        _SHM_CLAIMED_BY_WORKER = True
        rank = 0
        try:
            from vllm.distributed import get_tensor_model_parallel_rank
            rank = get_tensor_model_parallel_rank()
        except Exception:
            rank = int(os.environ.get("RANK", os.environ.get("NODE_RANK", os.environ.get("LOCAL_RANK", "0"))))

        dev_idx = torch.cuda.current_device() if torch.cuda.is_available() else 0
        target_device = torch.device(f"cuda:{dev_idx}")
        shm_path = get_rank_shm_path(rank)

        if os.path.exists(shm_path) and not _is_cuda_ipc_daemon_alive(shm_path) and not _is_shm_file_nonsparse(shm_path):
            logger.warning(
                "Removing stale/sparse %s (and %s.cudaipc.json) because CUDA IPC daemon is not running.",
                shm_path,
                shm_path,
            )
            for stale_p in (shm_path, f"{shm_path}.cudaipc.json", f"{shm_path}.w13_normalized"):
                try:
                    if os.path.exists(stale_p):
                        os.unlink(stale_p)
                except Exception:
                    pass

        if os.path.exists(shm_path) and (
            _is_cuda_ipc_daemon_alive(shm_path) or _is_shm_file_nonsparse(shm_path, min_bytes=1024 * 1024 * 1024)
        ):
            # WARM BOOT PATH (< 2.5s zero-copy load from /dev/shm)
            t_warm0 = time.perf_counter()
            mem0 = read_proc_meminfo_gib()
            logger.info(
                "WARM BOOT rank=%d from %s (FilePageCache=%.2f GiB, Shmem=%.2f GiB)",
                rank,
                shm_path,
                mem0.get("FilePageCache", 0.0),
                mem0.get("Shmem", 0.0),
            )

            # Start/join registration thread while or before building meta model
            start_background_preregister(rank=rank, device_index=dev_idx)

            t_meta0 = time.perf_counter()
            with set_default_torch_dtype(model_config.dtype), torch.device("meta"):
                model = initialize_model(vllm_config=vllm_config, model_config=model_config, prefix=prefix)
            meta_sec = time.perf_counter() - t_meta0

            if _PRE_REGISTER_THREAD is not None:
                _PRE_REGISTER_THREAD.join()
            entry = register_and_map_shm_safetensors(shm_path, device_index=dev_idx)

            t_bind0 = time.perf_counter()
            _ensure_shm_moe_w13_normalized(shm_path, entry["tensors"])
            _bind_shm_views_to_model(
                model,
                entry["tensors"],
                target_device,
                is_cold_rebind=False,
                is_cuda_ipc=bool(entry.get("is_cuda_ipc", False)),
            )
            _fix_glm53_indexer_buffers(model)

            with weights_already_processed():
                process_weights_after_loading(model, model_config, target_device)
            bind_sec = time.perf_counter() - t_bind0

            total_warm_sec = time.perf_counter() - t_warm0
            mem1 = read_proc_meminfo_gib()
            logger.info(
                "WARM BOOT COMPLETE rank=%d in %.3fs (meta_init=%.3fs, bind+pwal=%.3fs) | "
                "FilePageCache: %.2f -> %.2f GiB (delta=%+.3f GiB) | MemAvailable=%.2f GiB",
                rank,
                total_warm_sec,
                meta_sec,
                bind_sec,
                mem0.get("FilePageCache", 0.0),
                mem1.get("FilePageCache", 0.0),
                mem1.get("FilePageCache", 0.0) - mem0.get("FilePageCache", 0.0),
                mem1.get("MemAvailable", 0.0),
            )
            return model.eval()

        # COLD BOOT PATH: load via original loader with per-shard page-cache eviction, then lockstep export to /dev/shm
        t_cold0 = time.perf_counter()
        logger.info("COLD BOOT rank=%d: %s not found; loading from disk and populating /dev/shm...", rank, shm_path)
        model = _call_orig_loader(self, vllm_config=vllm_config, model_config=model_config, prefix=prefix)
        _fix_glm53_indexer_buffers(model)
        drop_file_page_cache()
        export_and_rebind_model_to_shm(model, shm_path, rank=rank, device_index=dev_idx)
        logger.info("COLD BOOT + /dev/shm EXPORT COMPLETE rank=%d in %.2fs", rank, time.perf_counter() - t_cold0)
        return model.eval()

    B12xModelLoader.load_model = _shm_load_model
    if DefaultModelLoader is not B12xModelLoader:
        DefaultModelLoader.load_model = _shm_load_model

    logger.info("Installed /dev/shm zero-copy safetensors & B12X pre-processed weight patches.")
