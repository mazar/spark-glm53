#!/usr/bin/env bash
# Download nvidia/GLM-5.3-Flash-FP4-Local into the HF cache without bloating the
# Linux page cache (GB10 UMA trap, GEMINI.md §3.1): a side loop issues
# POSIX_FADV_DONTNEED on every blob in the repo cache dir every 15 s.
set -euo pipefail
REPO="nvidia/GLM-5.3-Flash-FP4-Local"
REV="${REV:-76348ddb2be2e35e39ecfc00e75dc07703f4f3d5}"
CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}/hub/models--nvidia--GLM-5.3-Flash-FP4-Local"
LOG_PREFIX="[fetch_nvidia_ckpt]"

fadvise_loop() {
  while true; do
    python3 - "$CACHE_DIR" <<'EOF' 2>/dev/null || true
import os, sys
root = sys.argv[1]
for dp, _, fns in os.walk(root):
    for fn in fns:
        p = os.path.join(dp, fn)
        if os.path.islink(p):
            continue
        try:
            fd = os.open(p, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
        except OSError:
            pass
EOF
    sleep 15
  done
}

mkdir -p "$CACHE_DIR"
fadvise_loop &
FADV_PID=$!
trap 'kill $FADV_PID 2>/dev/null || true' EXIT

echo "$LOG_PREFIX $(date -Is) start $REPO@$REV"
HF_HUB_DISABLE_XET=1 uvx --from 'huggingface_hub[hf_transfer]' hf download "$REPO" --revision "$REV" --max-workers 2
echo "$LOG_PREFIX $(date -Is) done"
du -sh "$CACHE_DIR"
