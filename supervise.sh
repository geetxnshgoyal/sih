#!/bin/bash
# Keep the fetch + extract pipeline alive.
#
# Why this exists
# ---------------
# On 31 Aug both jobs were found dead with /tmp wiped (reboot or tmp cleanup),
# and nothing noticed. The corpus is 56.8 GB over ~30 hours of wall clock, so a
# silent death costs however long it takes a human to look. The scripts are
# individually robust — resumable, idempotent, single-instance — but nothing was
# responsible for noticing they had stopped.
#
# This does exactly that and nothing more: every 2 minutes, if a job is not
# running and its work is not finished, start it. Both children hold their own
# mkdir locks, so a double start is harmless.
#
#   nohup ./supervise.sh > run/supervise.log 2>&1 &
#
# Stop with: pkill -f supervise.sh   (children keep running; kill them too if wanted)
set -u
STATE="$(cd "$(dirname "$0")/." && pwd)/run"; mkdir -p "$STATE"
cd "$(dirname "$0")"

LOCK=$STATE/supervise.lock.d
mkdir "$LOCK" 2>/dev/null || { echo "supervisor already running"; exit 0; }
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

alive() { pgrep -f "$1" >/dev/null 2>&1; }

# The fetcher runs PARALLEL workers as background subshells, and those inherit
# the parent's command line — so `pgrep -f fetch_video.sh` counts N+1 processes
# for one healthy fetcher. On 31 Aug that made this supervisor mistake its own
# workers for duplicates and kill them, then restart, then fight itself.
#
# So identify the fetcher by the PID it records, not by name matching.
fetcher_alive() {
  local pid
  pid=$(cat $STATE/fetch.pid 2>/dev/null) || return 1
  [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}
extracted() { ls data/video/*.extracted 2>/dev/null | wc -l | tr -d ' '; }

echo "supervisor up $(date '+%F %T') — $(extracted)/46 extracted"

while true; do
  n=$(extracted)

  if [ "$n" -ge 46 ]; then
    echo "[$(date +%H:%M)] all 46 parts extracted — supervisor exiting"
    exit 0
  fi

  # Never restart on a single missed check.
  #
  # A false negative here is far more expensive than a slow recovery: two
  # fetchers write the same .part file and clobber each other, and because
  # Zenodo does not honour range requests there is no resume — the part starts
  # again from zero. That is exactly how ~1 GB of People_1of5 was lost on
  # 31 Aug, and it is the same class of failure HANDOFF.md records ("two curls
  # writing the same path truncated each other, 1.1 GB collapsed to 14 MB").
  #
  # So confirm death twice, 10s apart, before touching the lock or restarting.
  confirm_down() {
    "$1" && return 1
    sleep 10
    "$1" && return 1
    return 0
  }
  extract_alive() { alive "run_extract_loop"; }

  if confirm_down fetcher_alive; then
    rmdir $STATE/fetch-video.lock.d 2>/dev/null
    rm -f $STATE/fetch.pid
    # Re-check after clearing: the lock going away can let a racing starter in.
    if ! fetcher_alive; then
      echo "[$(date +%H:%M)] fetcher down at ${n}/46 — restarting"
      ( cd data && PARALLEL=${PARALLEL:-3} nohup ./fetch_video.sh >> $STATE/fetch.log 2>&1 & )
    fi
  fi

  if confirm_down extract_alive; then
    rmdir $STATE/extract.lock.d 2>/dev/null
    if ! alive "run_extract_loop"; then
      echo "[$(date +%H:%M)] extract loop down at ${n}/46 — restarting"
      nohup ./run_extract_loop.sh >> $STATE/extract.log 2>&1 &
    fi
  fi

  # The duplicate-killer that used to live here counted parallel workers as
  # duplicate fetchers and killed them. The PID file above makes it redundant:
  # the lock plus a liveness check on one recorded PID is what enforces single
  # instance, and a wrong kill costs an entire ~1.3 GB part.

  sleep 120
done
