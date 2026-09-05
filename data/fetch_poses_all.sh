#!/bin/bash
# Fetch every OpenHands pose dataset (Zenodo 6674324, CC BY 4.0).
# Raw video is handled separately by stream_video.sh, which never keeps more
# than one category on disk (58 GiB free will not hold the 56.8 GB archive
# plus its extraction).
# Resumable: each completed file gets a .done marker, so re-running skips it.
set -u
mkdir -p poses video

fetch () {  # $1 = url, $2 = output path, $3 = label
  [ -f "$2.done" ] && { echo "SKIP $3"; return; }
  echo "FETCH $3"
  if curl -sL -C - --retry 8 --retry-delay 5 --retry-all-errors -o "$2" "$1"; then
    touch "$2.done"; echo "OK $3 $(du -h "$2" | cut -f1)"
  else
    echo "FAIL $3"
  fi
}

echo "== pose datasets =="
for f in AUTSL MS-ASL WLASL GSL LSA64_Cut LSA64_Full ASLLVD-Skeleton \
         RWTH-PHOENIX-Weather-Signer03-cutout; do
  fetch "https://zenodo.org/record/6674324/files/$f.zip?download=1" "poses/$f.zip" "$f"
done

echo "ALLDONE"
