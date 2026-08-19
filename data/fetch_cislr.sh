#!/bin/bash
# CISLR: Corpus for Indian Sign Language Recognition (Joshi et al., EMNLP 2022).
#
# Why this dataset, specifically
# -----------------------------
# The three-arm ablation (ARCHITECTURE.md §9) closed both open modelling
# questions and pointed at one thing: SIGNERS.
#
#   "What remains is statistical power: three signer groups cannot resolve a
#    difference this small. More GROUPS, not more categories, is what would
#    settle it."
#
# INCLUDE is 7 signers from one school in Chennai, one room, one camera
# distance: and the held-out-group spread on it is enormous (45% to 68%
# depending on which group is held out). That spread IS the problem: it dwarfs
# every architectural effect we measured, and it is why the honest number is
# 56.8% rather than the 90.3% a random split reports.
#
# CISLR is 71 signers over ~4,700 words. That is roughly 10x the signer
# diversity, and it is the only public ISL corpus that changes this particular
# constraint. At 1.59 GB it is also trivial next to INCLUDE's 56.8 GB.
#
# Access
# ------
# Gated (gated: auto) under AFL-3.0. A human must accept the conditions once,
# which includes agreeing to share contact information, that is the user's
# consent to give, so this script cannot and should not do it:
#
#   1. Sign in at https://huggingface.co/datasets/Exploration-Lab/CISLR
#   2. Accept the conditions (auto-approved)
#   3. Create a READ token at https://huggingface.co/settings/tokens
#   4. HF_TOKEN=hf_xxx ./data/fetch_cislr.sh
#
# The token is read from the environment and never written to disk here.
set -u
STATE="$(cd "$(dirname "$0")/.." && pwd)/run"; mkdir -p "$STATE"
DEST="$(cd "$(dirname "$0")" && pwd)/cislr"
REPO="Exploration-Lab/CISLR"

if [ -z "${HF_TOKEN:-}" ]; then
  cat <<'MSG'
HF_TOKEN is not set.

CISLR is gated: a human has to accept the licence once, which includes agreeing
to share contact information. That consent cannot be automated.

  1. https://huggingface.co/datasets/Exploration-Lab/CISLR  -> accept conditions
  2. https://huggingface.co/settings/tokens                 -> create a READ token
  3. HF_TOKEN=hf_xxx ./data/fetch_cislr.sh

MSG
  exit 1
fi

mkdir -p "$DEST"
echo "CISLR -> $DEST"

# Ask the API what is actually in the repo rather than assuming filenames; a
# gated repo returns 401 here, which is the cheapest possible check that the
# token works before pulling a gigabyte.
python3 - "$REPO" "$DEST" <<'PY'
import json, os, sys, urllib.request

repo, dest = sys.argv[1], sys.argv[2]
tok = os.environ["HF_TOKEN"]
req = urllib.request.Request(
    f"https://huggingface.co/api/datasets/{repo}?full=true",
    headers={"Authorization": f"Bearer {tok}"},
)
try:
    meta = json.load(urllib.request.urlopen(req))
except urllib.error.HTTPError as e:
    if e.code in (401, 403):
        sys.exit("token rejected or conditions not yet accepted, "
                 "open the dataset page, accept, then retry")
    raise

files = [f["rfilename"] for f in meta.get("siblings", [])]
print(f"  {len(files)} files in repo")
with open(os.path.join(dest, "_manifest.txt"), "w") as fh:
    fh.write("\n".join(files))

# Pull the small metadata + feature files first: the I3D features and the CSVs
# are enough to check signer coverage and label overlap with INCLUDE BEFORE
# committing to the video download.
small = [f for f in files if f.endswith((".csv", ".pkl", ".json", ".md"))]
print(f"  {len(small)} metadata/feature files to fetch first")
with open(os.path.join(dest, "_small.txt"), "w") as fh:
    fh.write("\n".join(small))
PY

# Fetch the metadata/feature files, then everything else. Same .part-then-verify
# discipline as fetch_video.sh, for the same reason.
fetch_list() {
  local list="$1"
  [ -s "$list" ] || return 0
  while read -r f; do
    [ -z "$f" ] && continue
    local out="$DEST/$f"
    [ -f "$out" ] && continue
    mkdir -p "$(dirname "$out")"
    echo "  fetch $f"
    curl -sL --retry 6 --retry-delay 5 --retry-all-errors \
         --connect-timeout 30 --speed-limit 1024 --speed-time 60 \
         -H "Authorization: Bearer $HF_TOKEN" \
         -o "$out.part" \
         "https://huggingface.co/datasets/$REPO/resolve/main/$f" \
      && mv -f "$out.part" "$out" \
      || { rm -f "$out.part"; echo "  FAIL $f"; }
  done < "$list"
}

fetch_list "$DEST/_small.txt"
echo
echo "metadata fetched. Inspect before pulling video:"
echo "  ls -la $DEST"
echo
echo "To fetch everything (including CISLR_v1.5-a_videos/, ~1.6 GB total):"
echo "  FETCH_ALL=1 HF_TOKEN=... $0"

if [ "${FETCH_ALL:-0}" = "1" ]; then
  echo
  echo "fetching all files..."
  fetch_list "$DEST/_manifest.txt"
  echo "done: $(find "$DEST" -type f | wc -l | tr -d ' ') files"
fi
