#!/bin/bash
# Extract landmarks from every video part as it finishes downloading.
# Idempotent: already-extracted clips are skipped, so re-running is free.
set -u
STATE="$(cd "$(dirname "$0")/." && pwd)/run"; mkdir -p "$STATE"
LOCK=$STATE/extract.lock.d
mkdir "$LOCK" 2>/dev/null || { echo "already running"; exit 0; }
trap 'rmdir "$LOCK"' EXIT
for i in $(seq 1 400); do
  n=$(ls data/video/*.done 2>/dev/null | wc -l | tr -d ' ')
  if [ "$n" -gt 0 ]; then
    echo "=== $(date +%H:%M:%S) extracting $n ready part(s) ==="
    .venv-mp/bin/python -u train/extract_video.py 2>&1 \
      | grep -vE "WARNING|absl|GL version|XNNPACK|inference_feedback|W0000|I0000|SymbolDatabase|warnings.warn"
    echo "clips so far: $(find data/video_landmarks -name '*.npz' | wc -l | tr -d ' ')"
  fi
  [ "$(ls data/video/*.extracted 2>/dev/null | wc -l | tr -d ' ')" -ge 46 ] && { echo "ALL_EXTRACTED"; break; }
  sleep 90
done
