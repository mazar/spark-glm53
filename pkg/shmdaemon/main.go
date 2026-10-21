package main

/*
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

static int raise_memlock_rlimit(void) {
    struct rlimit rl;
    rl.rlim_cur = RLIM_INFINITY;
    rl.rlim_max = RLIM_INFINITY;
    return setrlimit(RLIMIT_MEMLOCK, &rl);
}

// Copy src_path to dst_shm_path using O_DIRECT (4096-aligned 8 MiB chunks) +
// POSIX_FADV_DONTNEED so zero bytes remain in the Linux page cache.
static int odirect_copy_to_shm(const char* src_path, const char* dst_shm_path, size_t* out_size, char* err_buf, size_t err_cap) {
    struct stat st;
    if (stat(src_path, &st) != 0) {
        snprintf(err_buf, err_cap, "stat(%s) failed: %s", src_path, strerror(errno));
        return -1;
    }
    size_t file_size = (size_t)st.st_size;
    *out_size = file_size;

    // Check if dst_shm_path already exists with the exact size and is not sparse/hole-punched
    struct stat dst_st;
    if (stat(dst_shm_path, &dst_st) == 0 && (size_t)dst_st.st_size == file_size && file_size > 0 &&
        (size_t)dst_st.st_blocks * 512 >= (file_size * 9) / 10) {
        return 0;
    }

    int src_fd = open(src_path, O_RDONLY | O_DIRECT);
    int used_odirect = 1;
    if (src_fd < 0) {
        used_odirect = 0;
        src_fd = open(src_path, O_RDONLY);
        if (src_fd < 0) {
            snprintf(err_buf, err_cap, "open(%s) failed: %s", src_path, strerror(errno));
            return -1;
        }
    }

    char tmp_path[512];
    snprintf(tmp_path, sizeof(tmp_path), "%s.tmp.%d", dst_shm_path, (int)getpid());
    int dst_fd = open(tmp_path, O_RDWR | O_CREAT | O_TRUNC, 0666);
    if (dst_fd < 0) {
        snprintf(err_buf, err_cap, "open(%s) failed: %s", tmp_path, strerror(errno));
        close(src_fd);
        return -1;
    }

    if (ftruncate(dst_fd, (off_t)file_size) != 0) {
        snprintf(err_buf, err_cap, "ftruncate(%s, %zu) failed: %s", tmp_path, file_size, strerror(errno));
        close(dst_fd);
        unlink(tmp_path);
        close(src_fd);
        return -1;
    }

    void* dst_map = mmap(NULL, file_size, PROT_READ | PROT_WRITE, MAP_SHARED, dst_fd, 0);
    if (dst_map == MAP_FAILED) {
        snprintf(err_buf, err_cap, "mmap(%s, %zu) failed: %s", tmp_path, file_size, strerror(errno));
        close(dst_fd);
        unlink(tmp_path);
        close(src_fd);
        return -1;
    }
#ifdef MADV_HUGEPAGE
    madvise(dst_map, file_size, MADV_HUGEPAGE);
#endif

    const size_t align = 4096;
    const size_t chunk_size = 8 * 1024 * 1024; // 8 MiB
    void* bounce = NULL;
    if (posix_memalign(&bounce, align, chunk_size) != 0 || bounce == NULL) {
        snprintf(err_buf, err_cap, "posix_memalign failed");
        munmap(dst_map, file_size);
        close(dst_fd);
        unlink(tmp_path);
        close(src_fd);
        return -1;
    }

    size_t aligned_end = used_odirect ? ((file_size / align) * align) : file_size;
    size_t offset = 0;
    while (offset < aligned_end) {
        size_t to_read = chunk_size;
        if (offset + to_read > aligned_end) {
            to_read = aligned_end - offset;
        }
        ssize_t n = pread(src_fd, bounce, to_read, (off_t)offset);
        if (n <= 0) {
            snprintf(err_buf, err_cap, "pread(%s, off=%zu) failed: %s", src_path, offset, strerror(errno));
            free(bounce);
            munmap(dst_map, file_size);
            close(dst_fd);
            unlink(tmp_path);
            close(src_fd);
            return -1;
        }
        memcpy((uint8_t*)dst_map + offset, bounce, (size_t)n);
        posix_fadvise(src_fd, (off_t)offset, (off_t)n, POSIX_FADV_DONTNEED);
        offset += (size_t)n;
    }

    close(src_fd);

    if (offset < file_size) {
        int tail_fd = open(src_path, O_RDONLY);
        if (tail_fd < 0) {
            snprintf(err_buf, err_cap, "open tail(%s) failed: %s", src_path, strerror(errno));
            free(bounce);
            munmap(dst_map, file_size);
            close(dst_fd);
            unlink(tmp_path);
            return -1;
        }
        while (offset < file_size) {
            size_t to_read = file_size - offset;
            ssize_t n = pread(tail_fd, (uint8_t*)dst_map + offset, to_read, (off_t)offset);
            if (n <= 0) {
                snprintf(err_buf, err_cap, "pread tail(%s) failed: %s", src_path, strerror(errno));
                close(tail_fd);
                free(bounce);
                munmap(dst_map, file_size);
                close(dst_fd);
                unlink(tmp_path);
                return -1;
            }
            offset += (size_t)n;
        }
        posix_fadvise(tail_fd, 0, (off_t)file_size, POSIX_FADV_DONTNEED);
        close(tail_fd);
    }

    free(bounce);
    munmap(dst_map, file_size);
    close(dst_fd);

    if (rename(tmp_path, dst_shm_path) != 0) {
        snprintf(err_buf, err_cap, "rename(%s -> %s) failed: %s", tmp_path, dst_shm_path, strerror(errno));
        unlink(tmp_path);
        return -1;
    }
    chmod(dst_shm_path, 0666);
    return 0;
}

typedef struct {
    int fd;
    void* addr;
    size_t size;
    int locked;
} shm_pin_t;

static int pin_shm_file(const char* shm_path, shm_pin_t* out_pin, char* err_buf, size_t err_cap) {
    struct stat st;
    if (stat(shm_path, &st) != 0) {
        snprintf(err_buf, err_cap, "stat(%s) failed: %s", shm_path, strerror(errno));
        return -1;
    }
    size_t sz = (size_t)st.st_size;
    if (sz == 0) {
        snprintf(err_buf, err_cap, "shm file %s is empty", shm_path);
        return -1;
    }
    if ((size_t)st.st_blocks * 512 < (sz * 9) / 10) {
        snprintf(err_buf, err_cap, "shm file %s is sparse/hole-punched (blocks=%zu, size=%zu); refusing mlock",
                 shm_path, (size_t)st.st_blocks * 512, sz);
        return -1;
    }
    int fd = open(shm_path, O_RDONLY);
    if (fd < 0) {
        snprintf(err_buf, err_cap, "open(%s) failed: %s", shm_path, strerror(errno));
        return -1;
    }
    void* addr = mmap(NULL, sz, PROT_READ, MAP_SHARED, fd, 0);
    if (addr == MAP_FAILED) {
        snprintf(err_buf, err_cap, "mmap(%s, %zu) failed: %s", shm_path, sz, strerror(errno));
        close(fd);
        return -1;
    }
#ifdef MADV_HUGEPAGE
    madvise(addr, sz, MADV_HUGEPAGE);
#endif
    int locked = 0;
    if (mlock(addr, sz) == 0) {
        locked = 1;
    } else {
        // Still keep mmap open even if mlock hits transient limit
        locked = 0;
    }
    out_pin->fd = fd;
    out_pin->addr = addr;
    out_pin->size = sz;
    out_pin->locked = locked;
    return 0;
}

static void unpin_shm_file(shm_pin_t* pin) {
    if (pin->addr != NULL && pin->addr != MAP_FAILED && pin->size > 0) {
        if (pin->locked) {
            munlock(pin->addr, pin->size);
        }
        munmap(pin->addr, pin->size);
    }
    if (pin->fd >= 0) {
        close(pin->fd);
    }
    pin->addr = NULL;
    pin->size = 0;
    pin->fd = -1;
    pin->locked = 0;
}
*/
import "C"

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"time"
	"unsafe"
)

type PinnedSegment struct {
	Path      string    `json:"path"`
	SizeBytes uint64    `json:"size_bytes"`
	Locked    bool      `json:"locked"`
	PinnedAt  time.Time `json:"pinned_at"`
	cPin      C.shm_pin_t
}

type DaemonStatus struct {
	Rank        int                       `json:"rank"`
	PID         int                       `json:"pid"`
	SocketPath  string                    `json:"socket_path"`
	TotalBytes  uint64                    `json:"total_bytes"`
	TotalLocked uint64                    `json:"total_locked_bytes"`
	Segments    map[string]*PinnedSegment `json:"segments"`
}

type ShmDaemon struct {
	mu         sync.Mutex
	rank       int
	socketPath string
	segments   map[string]*PinnedSegment
}

func NewShmDaemon(rank int, socketPath string) *ShmDaemon {
	return &ShmDaemon{
		rank:       rank,
		socketPath: socketPath,
		segments:   make(map[string]*PinnedSegment),
	}
}

func (d *ShmDaemon) LockSegment(shmPath string) (*PinnedSegment, error) {
	d.mu.Lock()
	defer d.mu.Unlock()

	if existing, ok := d.segments[shmPath]; ok {
		if fi, err := os.Stat(shmPath); err == nil && uint64(fi.Size()) == existing.SizeBytes && existing.Locked {
			return existing, nil
		}
		C.unpin_shm_file(&existing.cPin)
		delete(d.segments, shmPath)
	}

	cPath := C.CString(shmPath)
	defer C.free(unsafe.Pointer(cPath))

	var pin C.shm_pin_t
	errBuf := (*C.char)(C.calloc(512, 1))
	defer C.free(unsafe.Pointer(errBuf))

	t0 := time.Now()
	rc := C.pin_shm_file(cPath, &pin, errBuf, 512)
	if rc != 0 {
		return nil, fmt.Errorf("%s", C.GoString(errBuf))
	}

	seg := &PinnedSegment{
		Path:      shmPath,
		SizeBytes: uint64(pin.size),
		Locked:    pin.locked != 0,
		PinnedAt:  time.Now(),
		cPin:      pin,
	}
	d.segments[shmPath] = seg
	log.Printf("[spark-shm-weights rank=%d] Pinned %s (%.2f GiB, mlock=%v) in %v",
		d.rank, shmPath, float64(seg.SizeBytes)/(1024*1024*1024), seg.Locked, time.Since(t0))
	return seg, nil
}

func (d *ShmDaemon) LoadAndLockFile(srcPath, dstShmPath string) (*PinnedSegment, error) {
	cSrc := C.CString(srcPath)
	defer C.free(unsafe.Pointer(cSrc))
	cDst := C.CString(dstShmPath)
	defer C.free(unsafe.Pointer(cDst))

	var outSize C.size_t
	errBuf := (*C.char)(C.calloc(512, 1))
	defer C.free(unsafe.Pointer(errBuf))

	t0 := time.Now()
	rc := C.odirect_copy_to_shm(cSrc, cDst, &outSize, errBuf, 512)
	if rc != 0 {
		return nil, fmt.Errorf("%s", C.GoString(errBuf))
	}
	log.Printf("[spark-shm-weights rank=%d] O_DIRECT loaded %s -> %s (%.2f GiB) in %v",
		d.rank, srcPath, dstShmPath, float64(outSize)/(1024*1024*1024), time.Since(t0))
	return d.LockSegment(dstShmPath)
}

func (d *ShmDaemon) UnlockSegment(shmPath string) {
	d.mu.Lock()
	defer d.mu.Unlock()
	if seg, ok := d.segments[shmPath]; ok {
		C.unpin_shm_file(&seg.cPin)
		delete(d.segments, shmPath)
		log.Printf("[spark-shm-weights rank=%d] Unpinned %s", d.rank, shmPath)
	}
}

func (d *ShmDaemon) Status() DaemonStatus {
	d.mu.Lock()
	defer d.mu.Unlock()

	st := DaemonStatus{
		Rank:       d.rank,
		PID:        os.Getpid(),
		SocketPath: d.socketPath,
		Segments:   make(map[string]*PinnedSegment, len(d.segments)),
	}
	for k, v := range d.segments {
		st.Segments[k] = v
		st.TotalBytes += v.SizeBytes
		if v.Locked {
			st.TotalLocked += v.SizeBytes
		}
	}
	return st
}

func (d *ShmDaemon) CloseAll() {
	d.mu.Lock()
	defer d.mu.Unlock()
	for k, seg := range d.segments {
		C.unpin_shm_file(&seg.cPin)
		delete(d.segments, k)
	}
}

func findDFlash2Safetensors(explicitPath string) string {
	if explicitPath != "" {
		if fi, err := os.Stat(explicitPath); err == nil && !fi.IsDir() {
			return explicitPath
		}
		cand := filepath.Join(explicitPath, "model.safetensors")
		if _, err := os.Stat(cand); err == nil {
			return cand
		}
	}
	home, _ := os.UserHomeDir()
	candidates := []string{
		filepath.Join(home, ".cache/huggingface/hub/models--incoai--GLM-5.3-Flash-DFlash2/snapshots/bf582e4eacc1810f76656d1811693ff6c6737d2a/model.safetensors"),
		"/root/.cache/huggingface/hub/models--incoai--GLM-5.3-Flash-DFlash2/snapshots/bf582e4eacc1810f76656d1811693ff6c6737d2a/model.safetensors",
		"/home/massoud/.cache/huggingface/hub/models--incoai--GLM-5.3-Flash-DFlash2/snapshots/bf582e4eacc1810f76656d1811693ff6c6737d2a/model.safetensors",
	}
	for _, c := range candidates {
		if _, err := os.Stat(c); err == nil {
			return c
		}
	}
	return ""
}

func (d *ShmDaemon) handleConn(conn net.Conn) {
	defer conn.Close()
	scanner := bufio.NewScanner(conn)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		parts := strings.Fields(line)
		cmd := strings.ToUpper(parts[0])
		switch cmd {
		case "STATUS":
			b, _ := json.Marshal(d.Status())
			fmt.Fprintf(conn, "%s\n", string(b))
		case "LOCK":
			if len(parts) < 2 {
				fmt.Fprintf(conn, `{"error":"usage: LOCK <shm_path>"}`+"\n")
				continue
			}
			seg, err := d.LockSegment(parts[1])
			if err != nil {
				fmt.Fprintf(conn, `{"error":%q}`+"\n", err.Error())
				continue
			}
			b, _ := json.Marshal(seg)
			fmt.Fprintf(conn, "%s\n", string(b))
		case "LOAD":
			if len(parts) < 3 {
				fmt.Fprintf(conn, `{"error":"usage: LOAD <src_path> <dst_shm_path>"}`+"\n")
				continue
			}
			seg, err := d.LoadAndLockFile(parts[1], parts[2])
			if err != nil {
				fmt.Fprintf(conn, `{"error":%q}`+"\n", err.Error())
				continue
			}
			b, _ := json.Marshal(seg)
			fmt.Fprintf(conn, "%s\n", string(b))
		case "UNLOCK":
			if len(parts) < 2 {
				fmt.Fprintf(conn, `{"error":"usage: UNLOCK <shm_path>"}`+"\n")
				continue
			}
			d.UnlockSegment(parts[1])
			fmt.Fprintf(conn, `{"ok":true}`+"\n")
		case "PING":
			fmt.Fprintf(conn, "PONG\n")
		default:
			fmt.Fprintf(conn, `{"error":"unknown command"}`+"\n")
		}
	}
}

func main() {
	rank := flag.Int("rank", 0, "Local TP rank on this node (0 on Head, 1 on Worker)")
	dflashPath := flag.String("dflash2-path", "", "Path to incoai/GLM-5.3-Flash-DFlash2 model.safetensors")
	rankShmPath := flag.String("rank-shm", "", "Path to rank-local /dev/shm weight file (defaults to /dev/shm/glm53_rank_<rank>.safetensors)")
	dflashShmPath := flag.String("dflash-shm", "/dev/shm/glm53_dflash2.safetensors", "Target /dev/shm path for DFlash2 weights")
	socketPath := flag.String("socket", "", "Unix control socket path (defaults to /tmp/spark_shm_weights_rank_<rank>.sock)")
	statusOnly := flag.Bool("status", false, "Query running daemon status and exit")
	lockCmd := flag.String("lock", "", "Tell running daemon to lock a /dev/shm file and exit")
	flag.Parse()

	if *rankShmPath == "" {
		*rankShmPath = fmt.Sprintf("/dev/shm/glm53_rank_%d.safetensors", *rank)
	}
	if *socketPath == "" {
		*socketPath = fmt.Sprintf("/tmp/spark_shm_weights_rank_%d.sock", *rank)
	}

	if *statusOnly || *lockCmd != "" {
		conn, err := net.DialTimeout("unix", *socketPath, 5*time.Second)
		if err != nil {
			log.Fatalf("Failed to connect to %s: %v", *socketPath, err)
		}
		defer conn.Close()
		if *lockCmd != "" {
			fmt.Fprintf(conn, "LOCK %s\n", *lockCmd)
		} else {
			fmt.Fprintf(conn, "STATUS\n")
		}
		resp, err := bufio.NewReader(conn).ReadString('\n')
		if err != nil {
			log.Fatalf("Failed to read response: %v", err)
		}
		fmt.Print(resp)
		return
	}

	if rc := C.raise_memlock_rlimit(); rc != 0 {
		log.Printf("[spark-shm-weights rank=%d] Warning: setrlimit(RLIMIT_MEMLOCK) returned %d (continuing)", *rank, int(rc))
	}

	daemon := NewShmDaemon(*rank, *socketPath)

	// 1. Load incoai/GLM-5.3-Flash-DFlash2 via O_DIRECT into /dev/shm/glm53_dflash2.safetensors and mlock() it
	if dfSrc := findDFlash2Safetensors(*dflashPath); dfSrc != "" {
		if _, err := daemon.LoadAndLockFile(dfSrc, *dflashShmPath); err != nil {
			log.Printf("[spark-shm-weights rank=%d] Warning: DFlash2 O_DIRECT load failed: %v", *rank, err)
		}
	}

	// 2. If /dev/shm/glm53_rank_<rank>.safetensors already exists (and is not migrated to CUDA IPC), pin it immediately with mlock()
	if _, ipcErr := os.Stat(*rankShmPath + ".cudaipc.json"); os.IsNotExist(ipcErr) {
		if fi, err := os.Stat(*rankShmPath); err == nil && fi.Size() > 0 {
			if _, err := daemon.LockSegment(*rankShmPath); err != nil {
				log.Printf("[spark-shm-weights rank=%d] Warning: locking %s failed: %v", *rank, *rankShmPath, err)
			}
		}
	}

	// 3. Background watcher: if /dev/shm/glm53_rank_<rank>.safetensors appears later (and is non-sparse and not CUDA IPC), automatically lock it
	go func() {
		ticker := time.NewTicker(2 * time.Second)
		defer ticker.Stop()
		for range ticker.C {
			if _, ipcErr := os.Stat(*rankShmPath + ".cudaipc.json"); ipcErr == nil {
				continue
			}
			if fi, err := os.Stat(*rankShmPath); err == nil && fi.Size() > 1024*1024*1024 {
				if sysSt, ok := fi.Sys().(*syscall.Stat_t); ok && uint64(sysSt.Blocks)*512 < (uint64(fi.Size())*9)/10 {
					continue
				}
				st := daemon.Status()
				if seg, ok := st.Segments[*rankShmPath]; !ok || seg.SizeBytes != uint64(fi.Size()) {
					_, _ = daemon.LockSegment(*rankShmPath)
				}
			}
		}
	}()

	_ = os.Remove(*socketPath)
	ln, err := net.Listen("unix", *socketPath)
	if err != nil {
		log.Fatalf("[spark-shm-weights rank=%d] Failed to listen on %s: %v", *rank, *socketPath, err)
	}
	_ = os.Chmod(*socketPath, 0666)
	log.Printf("[spark-shm-weights rank=%d] Listening on %s (watching %s)", *rank, *socketPath, *rankShmPath)

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sigCh
		log.Printf("[spark-shm-weights rank=%d] Shutting down daemon (keeping /dev/shm files intact)", *rank)
		_ = ln.Close()
		_ = os.Remove(*socketPath)
		daemon.CloseAll()
		os.Exit(0)
	}()

	for {
		conn, err := ln.Accept()
		if err != nil {
			return
		}
		go daemon.handleConn(conn)
	}
}
