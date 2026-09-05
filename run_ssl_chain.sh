#!/usr/bin/env bash
# Wait for the ISL dictionary extraction to finish, then build the SSL corpus
# and run self-supervised pretraining. Unattended: the extraction is ~18 h.
#
#   ./run_ssl_chain.sh &
#
# A lock file, because this repo has lost work to duplicate supervisors before:
# two fetchers once ran at once and cost ~1 GB of re-downloading.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
LOCK="$ROOT/run/ssl_chain.pid"

if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK")" 2>/dev/null; then
  echo "already running as PID $(cat "$LOCK") — refusing to start a second"
  exit 1
fi
mkdir -p run
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "waiting for extraction to finish"
while pgrep -f "train/extract_islgov.py" >/dev/null; do
  n=$(find data/islgov_landmarks -name '*.npz' 2>/dev/null | wc -l | tr -d ' ')
  log "  extracting: $n/13665"
  sleep 600
done

n=$(find data/islgov_landmarks -name '*.npz' 2>/dev/null | wc -l | tr -d ' ')
log "extraction stopped with $n clips"
if [ "$n" -lt 500 ]; then
  # Better to stop than to report a number from a corpus too small to mean
  # anything. 191 clips already produced a "working" run that proved nothing.
  log "too few clips to draw any conclusion — stopping rather than reporting noise"
  exit 1
fi

log "building the SSL corpus"
.venv/bin/python train/preprocess_ssl.py || { log "preprocess_ssl failed"; exit 1; }

log "self-supervised pretraining + fine-tune evaluation"
PYTHONUNBUFFERED=1 .venv-tf/bin/python train/pretrain_ssl.py || { log "pretrain_ssl failed"; exit 1; }

log "done — results in run/ssl_eval.json"
