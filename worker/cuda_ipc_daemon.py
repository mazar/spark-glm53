#!/usr/bin/env python3
"""
worker/cuda_ipc_daemon.py — Persistent CUDA IPC (`cudaMalloc` vidmem) weight server for GB10 (sm_121a).

Why this exists:
  On NVIDIA GB10 (`sm_121a`), `cudaHostRegister` (`/dev/shm` sysmem mapping) is mapped with
  host-coherent page attributes in the GPU GMMU, throttling `b12x` MoE (`routed_experts`, 79.73 GiB)
  and BF16 GEMMs (`lm_head`, 0.59 GiB) compared to native `cudaMalloc` (`vidmem`) allocations.
  Simultaneously, keeping 85.82 GiB in `/dev/shm` while cloning weights into `cudaMalloc`
  duplicates physical UMA pages.

  This daemon migrates the rank-local safetensors file (`/dev/shm/glm53_rank_{0,1}.safetensors`)
  in-place into ~22 contiguous 4.0 GiB `cudaMalloc` slabs:
    1. Copies each 4.0 GiB slab from `/dev/shm` into `cudaMalloc` (`cudaMemcpyHostToDevice`).
    2. Immediately punches a hole (`fallocate(FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE)`) in the
       corresponding `/dev/shm` file range so physical RAM is freed lockstep (peak transient overhead
       <= 4.0 GiB, steady-state duplicate memory = 0.00 GiB).
    3. Exports `cudaIpcMemHandle_t` handles to `/dev/shm/glm53_rank_{0,1}.cudaipc.json`.
    4. Any vLLM worker process attaches to all 85.2 GiB of native `cudaMalloc` weights via
       `cudaIpcOpenMemHandle` in < 0.15 seconds across warm restarts.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import mmap
import os
import signal
import socket
import struct
import sys
import time
from typing import Any

FALLOC_FL_KEEP_SIZE = 0x01
FALLOC_FL_PUNCH_HOLE = 0x02
CUDA_MEMCPY_HOST_TO_DEVICE = 1
CUDA_MEMCPY_DEVICE_TO_HOST = 2


class cudaIpcMemHandle_t(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_byte * 64)]


_libc: Any = None
_cudart: Any = None


def _init_native_libs() -> tuple[Any, Any]:
    global _libc, _cudart
    if _libc is None:
        _libc = ctypes.CDLL("libc.so.6", use_errno=True)
        _libc.fallocate.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int64, ctypes.c_int64]
        _libc.fallocate.restype = ctypes.c_int
    if _cudart is None:
        _cudart = ctypes.CDLL("libcudart.so")
        _cudart.cudaSetDevice.argtypes = [ctypes.c_int]
        _cudart.cudaSetDevice.restype = ctypes.c_int
        _cudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        _cudart.cudaMalloc.restype = ctypes.c_int
        _cudart.cudaFree.argtypes = [ctypes.c_void_p]
        _cudart.cudaFree.restype = ctypes.c_int
        _cudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        _cudart.cudaMemcpy.restype = ctypes.c_int
        _cudart.cudaDeviceSynchronize.argtypes = []
        _cudart.cudaDeviceSynchronize.restype = ctypes.c_int
        _cudart.cudaIpcGetMemHandle.argtypes = [ctypes.POINTER(cudaIpcMemHandle_t), ctypes.c_void_p]
        _cudart.cudaIpcGetMemHandle.restype = ctypes.c_int
    return _libc, _cudart


def _align_up(val: int, align: int = 256) -> int:
    return (val + align - 1) & ~(align - 1)


def _unlock_shmdaemon(shm_path: str, rank: int) -> None:
    for sock_path in (f"/tmp/spark_shm_weights_rank_{rank}.sock", "/tmp/spark_shm_weights_rank_0.sock"):
        if not os.path.exists(sock_path):
            continue
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                s.settimeout(10.0)
                s.connect(sock_path)
                s.sendall(f"UNLOCK {shm_path}\n".encode("utf-8"))
                resp = s.recv(4096).decode("utf-8", errors="replace").strip()
                print(f"[cuda_ipc_daemon] UNLOCK {shm_path} via {sock_path} -> {resp}", flush=True)
                return
        except Exception as exc:
            print(f"[cuda_ipc_daemon] Warning unlocking {sock_path}: {exc}", flush=True)


def _read_meminfo_gib() -> dict[str, float]:
    out: dict[str, float] = {}
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    k = parts[0].rstrip(":")
                    if k in ("MemTotal", "MemFree", "MemAvailable", "Cached", "Shmem"):
                        out[k] = float(parts[1]) / (1024.0 * 1024.0)
    except Exception:
        pass
    return out


def _punch_hole(fd: int, start_byte: int, end_byte: int) -> int:
    ps = ((start_byte + 4095) // 4096) * 4096
    pe = (end_byte // 4096) * 4096
    if pe <= ps:
        return 0
    rc = _libc.fallocate(fd, FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE, ps, pe - ps)
    if rc != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, f"fallocate(PUNCH_HOLE, {ps}, {pe - ps}) failed: {os.strerror(errno)}")
    return pe - ps


def run_daemon(rank: int, shm_path: str, ipc_json_path: str, sock_path: str, slab_gib: float = 4.0) -> None:
    import torch

    _init_native_libs()
    t0 = time.perf_counter()
    torch.cuda.init()
    _cudart.cudaSetDevice(0)

    if not os.path.exists(shm_path):
        raise FileNotFoundError(f"{shm_path} does not exist")

    # Ensure w13_normalized has already run before migrating bytes out of /dev/shm
    marker_path = f"{shm_path}.w13_normalized"
    if not os.path.exists(marker_path):
        print(f"[cuda_ipc_daemon] Normalizing w13 weights in {shm_path} before migration...", flush=True)
        from worker.shm_safetensors import register_and_map_shm_safetensors, _ensure_shm_moe_w13_normalized
        entry = register_and_map_shm_safetensors(shm_path, device_index=0)
        _ensure_shm_moe_w13_normalized(shm_path, entry["tensors"])

    _unlock_shmdaemon(shm_path, rank)
    mem_before = _read_meminfo_gib()

    fd = os.open(shm_path, os.O_RDWR)
    st = os.fstat(fd)
    file_size = st.st_size
    if st.st_blocks * 512 < file_size * 0.90:
        os.close(fd)
        if os.path.exists(ipc_json_path):
            try:
                os.unlink(ipc_json_path)
            except Exception:
                pass
        raise RuntimeError(
            f"[cuda_ipc_daemon] {shm_path} is sparse/hole-punched "
            f"(blocks={st.st_blocks * 512 / (1024**3):.2f} GiB vs size={file_size / (1024**3):.2f} GiB); "
            f"refusing to migrate incomplete /dev/shm file."
        )
    mm = mmap.mmap(fd, file_size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
    mm_buf = (ctypes.c_uint8 * file_size).from_buffer(mm)
    mm_base_ptr = ctypes.addressof(mm_buf)

    header_len = struct.unpack("<Q", mm[:8])[0]
    header = json.loads(mm[8 : 8 + header_len].decode("utf-8"))
    data_start = 8 + header_len
    metadata = header.get("__metadata__", {}) or {}

    active_tensors: list[tuple[int, int, str, str, list[int]]] = []
    empty_tensors: dict[str, dict[str, Any]] = {}
    punched_bytes = 0

    for name, info in header.items():
        if name == "__metadata__":
            continue
        s_off, e_off = int(info["data_offsets"][0]), int(info["data_offsets"][1])
        dtype_str = info["dtype"]
        shape = list(info["shape"])
        if e_off <= s_off:
            empty_tensors[name] = {"dtype": dtype_str, "shape": shape}
            continue
        if name.startswith("visual."):
            punched_bytes += _punch_hole(fd, data_start + s_off, data_start + e_off)
            continue
        active_tensors.append((s_off, e_off, name, dtype_str, shape))

    active_tensors.sort(key=lambda x: x[0])

    max_slab_bytes = int(slab_gib * (1024**3))
    slabs_plan: list[list[tuple[int, int, str, str, list[int], int]]] = []
    cur_slab: list[tuple[int, int, str, str, list[int], int]] = []
    cur_slab_bytes = 0

    for s_off, e_off, name, dtype_str, shape in active_tensors:
        nbytes = e_off - s_off
        aligned_off = _align_up(cur_slab_bytes, 256)
        if cur_slab and (aligned_off + nbytes > max_slab_bytes):
            slabs_plan.append(cur_slab)
            cur_slab = []
            cur_slab_bytes = 0
            aligned_off = 0
        cur_slab.append((s_off, e_off, name, dtype_str, shape, aligned_off))
        cur_slab_bytes = aligned_off + nbytes
    if cur_slab:
        slabs_plan.append(cur_slab)

    slab_ptrs: list[ctypes.c_void_p] = []
    slabs_meta: list[dict[str, Any]] = []
    tensors_meta: dict[str, dict[str, Any]] = {}
    total_cuda_bytes = 0

    print(
        f"[cuda_ipc_daemon] Migrating {len(active_tensors)} tensors from {shm_path} into {len(slabs_plan)} "
        f"cudaMalloc slabs (MemFree={mem_before.get('MemFree', 0.0):.2f} GiB, Shmem={mem_before.get('Shmem', 0.0):.2f} GiB)...",
        flush=True,
    )

    for slab_idx, slab_items in enumerate(slabs_plan):
        last_item = slab_items[-1]
        slab_size = _align_up(last_item[5] + (last_item[1] - last_item[0]), 4096)
        raw_ptr = ctypes.c_void_p()
        rc = _cudart.cudaMalloc(ctypes.byref(raw_ptr), ctypes.c_size_t(slab_size))
        if rc != 0:
            raise RuntimeError(f"cudaMalloc(slab={slab_idx}, size={slab_size}) failed with CUDA error {rc}")
        slab_ptrs.append(raw_ptr)
        total_cuda_bytes += slab_size

        for s_off, e_off, name, dtype_str, shape, slab_off in slab_items:
            nbytes = e_off - s_off
            rc = _cudart.cudaMemcpy(
                ctypes.c_void_p(raw_ptr.value + slab_off),
                ctypes.c_void_p(mm_base_ptr + data_start + s_off),
                ctypes.c_size_t(nbytes),
                CUDA_MEMCPY_HOST_TO_DEVICE,
            )
            if rc != 0:
                raise RuntimeError(f"cudaMemcpy({name}, {nbytes} bytes) failed with CUDA error {rc}")
            tensors_meta[name] = {
                "dtype": dtype_str,
                "shape": shape,
                "slab_idx": slab_idx,
                "slab_offset": slab_off,
                "nbytes": nbytes,
                "file_start_off": s_off,
                "file_end_off": e_off,
            }

        _cudart.cudaDeviceSynchronize()
        # Punch holes in /dev/shm immediately for all tensors in this slab
        first_s_off = slab_items[0][0]
        last_e_off = slab_items[-1][1]
        punched_bytes += _punch_hole(fd, data_start + first_s_off, data_start + last_e_off)

        h = cudaIpcMemHandle_t()
        rc = _cudart.cudaIpcGetMemHandle(ctypes.byref(h), raw_ptr)
        if rc != 0:
            raise RuntimeError(f"cudaIpcGetMemHandle(slab={slab_idx}) failed with CUDA error {rc}")
        slabs_meta.append(
            {
                "slab_idx": slab_idx,
                "size_bytes": slab_size,
                "handle_hex": bytes(h).hex(),
                "ptr": int(raw_ptr.value),
            }
        )

    del mm_buf
    mm.close()
    os.close(fd)

    payload = {
        "pid": os.getpid(),
        "rank": rank,
        "shm_path": shm_path,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_cuda_bytes": total_cuda_bytes,
        "punched_shm_bytes": punched_bytes,
        "slabs": slabs_meta,
        "tensors": tensors_meta,
        "empty_tensors": empty_tensors,
        "metadata": metadata,
    }
    tmp_json = f"{ipc_json_path}.tmp"
    with open(tmp_json, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    os.chmod(tmp_json, 0o666)
    os.replace(tmp_json, ipc_json_path)
    _unlock_shmdaemon(shm_path, rank)

    mem_after = _read_meminfo_gib()
    elapsed = time.perf_counter() - t0
    print(
        f"[cuda_ipc_daemon] MIGRATION COMPLETE rank={rank} in {elapsed:.2f}s | "
        f"cudaMalloc={total_cuda_bytes/(1024**3):.2f} GiB ({len(slabs_meta)} slabs) | "
        f"punched_shm={punched_bytes/(1024**3):.2f} GiB | "
        f"Shmem: {mem_before.get('Shmem', 0.0):.2f} -> {mem_after.get('Shmem', 0.0):.2f} GiB | "
        f"MemFree: {mem_before.get('MemFree', 0.0):.2f} -> {mem_after.get('MemFree', 0.0):.2f} GiB",
        flush=True,
    )

    if os.path.exists(sock_path):
        try:
            os.unlink(sock_path)
        except Exception:
            pass
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sock_path)
    os.chmod(sock_path, 0o666)
    srv.listen(8)

    stop_flag = False

    def _sig_handler(signum: int, _frame: Any) -> None:
        nonlocal stop_flag
        stop_flag = True

    signal.signal(signal.SIGTERM, _sig_handler)
    signal.signal(signal.SIGINT, _sig_handler)

    def _restore_to_shm() -> None:
        print(f"[cuda_ipc_daemon] Restoring {len(slabs_plan)} cudaMalloc slabs back to {shm_path}...", flush=True)
        r_fd = os.open(shm_path, os.O_RDWR)
        r_mm = mmap.mmap(r_fd, file_size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        r_buf = (ctypes.c_uint8 * file_size).from_buffer(r_mm)
        r_base = ctypes.addressof(r_buf)
        for s_idx, s_items in enumerate(slabs_plan):
            r_ptr = slab_ptrs[s_idx]
            for s_off, e_off, _name, _dt, _sh, slab_off in s_items:
                nb = e_off - s_off
                _cudart.cudaMemcpy(
                    ctypes.c_void_p(r_base + data_start + s_off),
                    ctypes.c_void_p(r_ptr.value + slab_off),
                    ctypes.c_size_t(nb),
                    CUDA_MEMCPY_DEVICE_TO_HOST,
                )
        _cudart.cudaDeviceSynchronize()
        del r_buf
        r_mm.close()
        os.close(r_fd)
        if os.path.exists(ipc_json_path):
            try:
                os.unlink(ipc_json_path)
            except Exception:
                pass
        print(f"[cuda_ipc_daemon] Restored {shm_path} from cudaMalloc slabs.", flush=True)

    srv.settimeout(1.0)
    while not stop_flag:
        try:
            conn, _ = srv.accept()
        except socket.timeout:
            continue
        except Exception:
            break
        with conn:
            try:
                cmd = conn.recv(1024).decode("utf-8", errors="replace").strip()
                if cmd.startswith("STATUS"):
                    status = {
                        "ok": True,
                        "pid": os.getpid(),
                        "rank": rank,
                        "slabs": len(slabs_meta),
                        "total_cuda_gib": round(total_cuda_bytes / (1024**3), 3),
                        "ipc_json_path": ipc_json_path,
                    }
                    conn.sendall((json.dumps(status) + "\n").encode("utf-8"))
                elif cmd.startswith("RESTORE"):
                    _restore_to_shm()
                    conn.sendall(b'{"ok":true,"restored":true}\n')
                    stop_flag = True
                elif cmd.startswith("STOP"):
                    conn.sendall(b'{"ok":true,"stopping":true}\n')
                    stop_flag = True
                else:
                    conn.sendall(b'{"ok":true}\n')
            except Exception:
                pass

    try:
        srv.close()
        if os.path.exists(sock_path):
            os.unlink(sock_path)
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Spark-GLM53 Persistent CUDA IPC Weight Server")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--shm-path", type=str, default="")
    parser.add_argument("--ipc-json", type=str, default="")
    parser.add_argument("--sock-path", type=str, default="")
    parser.add_argument("--slab-gib", type=float, default=4.0)
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()

    shm_path = args.shm_path or f"/dev/shm/glm53_rank_{args.rank}.safetensors"
    ipc_json = args.ipc_json or f"{shm_path}.cudaipc.json"
    sock_path = args.sock_path or f"/tmp/spark_cuda_ipc_rank_{args.rank}.sock"

    if args.status:
        if not os.path.exists(sock_path):
            print(json.dumps({"ok": False, "error": f"{sock_path} not found"}))
            sys.exit(1)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(5.0)
            s.connect(sock_path)
            s.sendall(b"STATUS\n")
            print(s.recv(4096).decode("utf-8", errors="replace").strip())
        return

    run_daemon(args.rank, shm_path, ipc_json, sock_path, slab_gib=args.slab_gib)


if __name__ == "__main__":
    main()
