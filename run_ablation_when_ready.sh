#!/bin/bash
# Re-run the FULL_FACE ablation as soon as the decisive categories are extracted.
#
# Why this waits rather than runs now
# -----------------------------------
# The first ablation (ARCHITECTURE.md §9) came back INCONCLUSIVE: mean delta
# 1.5 pp against a per-group spread of 5.5 pp. It ran on Adjectives, which carry
# almost no non-manual grammar — so it was structurally weak evidence about a
# change whose whole purpose is capturing eyebrows and mouth shape.
#
# Pronouns and Society are where ISL's question and negation marking actually
# lives. `fetch_video.sh` was reordered to pull them first for exactly this
# reason. This script waits for those five parts, then re-runs the same two-arm
# protocol over the widened set.
#
# Idempotent and safe to re-run: preprocess_face.py globs whatever is in
# data/video_landmarks, so it always uses everything extracted so far.
set -u
STATE="$(cd "$(dirname "$0")/." && pwd)/run"; mkdir -p "$STATE"
cd "$(dirname "$0")"

LOCK=$STATE/ablation.lock.d
mkdir "$LOCK" 2>/dev/null || { echo "already waiting/running"; exit 0; }
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

NEED=(Pronouns_1of2 Pronouns_2of2 Society_1of3 Society_2of3 Society_3of3)

ready() {
  for p in "${NEED[@]}"; do
    [ -f "data/video/${p}.zip.extracted" ] || return 1
  done
  return 0
}

echo "=== waiting for the decisive categories ==="
printf '  %s\n' "${NEED[@]}"
echo

# 8h ceiling at 60s polling: comfortably covers 6.3 GB plus extraction, and
# fails loudly rather than hanging forever if the fetcher dies.
for i in $(seq 1 480); do
  if ready; then
    echo "=== all five extracted after ~${i} min — running ablation ==="
    break
  fi
  if [ $((i % 15)) -eq 0 ]; then
    have=0
    for p in "${NEED[@]}"; do [ -f "data/video/${p}.zip.extracted" ] && have=$((have+1)); done
    echo "[$(date +%H:%M)] ${have}/5 decisive parts · $(ls data/video/*.extracted 2>/dev/null | wc -l | tr -d ' ')/46 total · $(find data/video_landmarks -name '*.npz' | wc -l | tr -d ' ') clips"
  fi
  sleep 60
done

if ! ready; then
  echo "TIMED OUT — decisive parts still missing. Fetcher may have stalled:"
  for p in "${NEED[@]}"; do
    [ -f "data/video/${p}.zip.extracted" ] && echo "  have $p" || echo "  MISSING $p"
  done
  exit 1
fi

echo
echo "=== preprocess_face.py ==="
.venv/bin/python train/preprocess_face.py 2>&1 | tail -20 || exit 1

echo
echo "=== train_ablation.py ==="
.venv-tf/bin/python train/train_ablation.py 2>&1 \
  | grep -vE "WARNING|absl|oneDNN|cpu_feature|W0000|I0000" | tail -40

echo
echo "=== ABLATION COMPLETE — see models/ablation_face.json ==="
