#!/bin/bash
# INCLUDE raw video, 46 parts (Zenodo 4010759, CC BY 4.0, 56.8 GB).
# Needed only to re-extract the 468-point face mesh the pose release lacks —
# ISL marks questions and negation on the face, and 11 coarse pose points
# cannot capture eyebrows or mouth shape.
#
# Single-instance enforced: an earlier version could be launched twice, and two
# curls writing the same path truncated each other (1.1 GB collapsed to 14 MB).
set -u
STATE="$(cd "$(dirname "$0")/.." && pwd)/run"; mkdir -p "$STATE"
LOCK=$STATE/fetch-video.lock
mkdir "$LOCK.d" 2>/dev/null || { echo "already running"; exit 0; }; echo $$ > $STATE/fetch.pid
trap 'rmdir "$LOCK.d" 2>/dev/null; rm -f $STATE/fetch.pid' EXIT


mkdir -p video
# Fetch order is by LINGUISTIC PRIORITY, not alphabetical.
#
# The point of re-extracting video is the 468-point face mesh, and the only
# question that matters is whether it earns its place (ARCHITECTURE.md §9). That
# question is decided by the categories carrying non-manual grammar — ISL marks
# questions with the eyebrows and negation with a head shake — which means
# Pronouns and Society. Adjectives, where the first ablation ran, carry almost
# none, which is exactly why that run came back INCONCLUSIVE.
#
# Alphabetically those categories are parts 39-45 of 46: the evidence needed to
# decide the change arrives only after ~50 GB, i.e. after the decision has
# effectively been made by default. Pulling them forward costs nothing — the same
# 46 parts are fetched either way — and buys a conclusive ablation days earlier.
#
# Everything downstream is order-independent: .done/.extracted markers make the
# fetcher and the extract loop idempotent, so reordering is safe mid-run.
python3 - <<'PY' > $STATE/include_files.txt
import json, urllib.request

# Earlier prefix = fetched sooner. Anything unlisted keeps alphabetical order
# after these, so the set fetched is unchanged.
PRIORITY = [
    "Pronouns",   # I, you, we, they — question and reference marking
    "Society",    # question-heavy: Election, Court, Religion, Law
    "Greetings",  # How are you / Good morning — the demo's opening phrases
    "People",     # relations, the other place non-manual marking shows up
]

def rank(key: str):
    for i, p in enumerate(PRIORITY):
        if key.startswith(p):
            return (0, i, key)
    return (1, 0, key)

d = json.load(urllib.request.urlopen("https://zenodo.org/api/records/4010759"))
# Emit "<key>\t<size>" so the shell can verify each download against the exact
# byte count the API reports, rather than trusting curl's exit status.
for f in sorted(d["files"], key=lambda x: rank(x["key"])):
    print(f"{f['key']}\t{f['size']}")
PY
total=$(wc -l < $STATE/include_files.txt | tr -d ' ')

# Zenodo does NOT support byte ranges. Verified 31 Aug:
#   curl -r 104857600-105906176 <file>  ->  HTTP 200, content-length 1208700458
# i.e. the FULL file, no 206, no content-range, no accept-ranges header.
#
# `curl -C -` against a server like that is worse than useless. curl sends a
# Range header, gets the whole file back, and in the bad case writes it at the
# partial's offset — producing an oversized archive that is silently corrupt and
# only fails later, in unzip, after the download cost has already been paid.
#
# So: no resume (there is nothing to resume to), download to a .part file, and
# verify the exact byte count before promoting it. A truncated transfer leaves
# .part behind and is simply redone; only a size-exact file is ever marked .done.
# That is slower per interruption but it is correct, which the previous version
# was not.
# Downloads run PARALLEL. Measured 31 Aug: a second concurrent connection served
# 1.78 MB/s while the first was already saturated, so a single stream was leaving
# throughput on the table. Zenodo's documented limit (x-ratelimit-limit: 133) is
# on REQUEST COUNT, not bandwidth — a whole-file download is one request, so a
# handful of concurrent transfers is nowhere near it.
#
# Each worker owns whole files, never a shared one. That matters: two writers on
# one path is the corruption HANDOFF.md records (1.1 GB collapsed to 14 MB) and
# the duplicate-fetcher bug that cost ~1 GB of People_1of5 on 31 Aug. Distinct
# .part paths make concurrency safe here where it was catastrophic there.
PARALLEL=${PARALLEL:-3}

fetch_one() {
  local key="$1" want="$2" pass="$3"
  [ -f "video/$key.done" ] && return 0
  [ -f "video/$key.extracted" ] && return 0
  # Another worker (or an earlier run) may already be on this file.
  [ -f "video/$key.part" ] && return 0

  echo "FETCH pass$pass $key ($((want/1048576)) MB)"
  # --speed-limit/--speed-time abort a transfer that drops below 1 KB/s for 60s,
  # which --retry alone does NOT cover: a half-open socket never errors, so curl
  # waits forever. That is exactly what happened when the machine changed
  # networks on 2 Sep — three workers sat at 0 MB/min with live PIDs and a
  # reachable server, and nothing timed out or retried.
  #
  # --connect-timeout bounds the handshake separately, since a dead route stalls
  # there rather than mid-body.
  curl -sL --retry 6 --retry-delay 5 --retry-all-errors \
       --connect-timeout 30 --speed-limit 1024 --speed-time 60 \
       -o "video/$key.part" "https://zenodo.org/records/4010759/files/$key?download=1"
  local got
  got=$(stat -f%z "video/$key.part" 2>/dev/null || echo 0)
  if [ "$got" = "$want" ]; then
    mv -f "video/$key.part" "video/$key"
    touch "video/$key.done"
    echo "OK $key $((got/1048576)) MB verified"
    return 0
  fi
  rm -f "video/$key.part"
  echo "FAIL $key got $((got/1048576))MB want $((want/1048576))MB — retrying next pass"
  return 1
}

for pass in $(seq 1 60); do
  missing=0
  while IFS=$'\t' read -r key want; do
    [ -z "$key" ] && continue
    [ -f "video/$key.done" ] && continue
    [ -f "video/$key.extracted" ] && continue
    [ -f "video/$key.part" ] && continue

    # Block until a worker slot frees up. `wait -n` returns on the first child
    # to finish, so slots are reused as soon as they open rather than in batches.
    while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do
      wait -n 2>/dev/null || sleep 2
    done
    fetch_one "$key" "$want" "$pass" &
  done < $STATE/include_files.txt

  wait   # let this pass drain before judging what is still missing

  while IFS=$'\t' read -r key want; do
    [ -z "$key" ] && continue
    [ -f "video/$key.done" ] || [ -f "video/$key.extracted" ] || missing=$((missing+1))
  done < $STATE/include_files.txt

  [ "$missing" -eq 0 ] && { echo "VIDEO_ALLDONE"; break; }
  echo "pass$pass done — $missing still missing"
  sleep 15
done
