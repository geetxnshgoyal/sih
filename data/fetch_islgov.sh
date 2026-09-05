#!/usr/bin/env bash
# Fetch only the ISL-dictionary clips whose word is already one of our 264
# classes. The full dataset is 75 GB of H.265; we need 292 files.
#
#   Indian Sign Language Dictionary, Government of India Open Data Portal,
#   re-encoded by silentone0725. MIT licensed -- redistributable, unlike CISLR.
#
#   ./data/fetch_islgov.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="silentone0725/Indian_Sign_Language_Data.gov_Rencoded"
OUT="$ROOT/data/islgov"
mkdir -p "$OUT"
"$ROOT/.venv/bin/python" - "$REPO" "$OUT" <<'PY'
import json, sys, os, urllib.parse, urllib.request, pathlib, time
repo, out = sys.argv[1], pathlib.Path(sys.argv[2])
todo = json.load(open(pathlib.Path(__file__).parent.parent / "data/meta/_islgov_todo.json"))
done = 0
for i, item in enumerate(todo, 1):
    dest = out / item["gloss"] / pathlib.Path(item["path"]).name
    if dest.exists() and dest.stat().st_size > 0:
        done += 1
        continue
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{urllib.parse.quote(item['path'])}"
    tmp = dest.with_suffix(".part")
    for attempt in range(4):
        try:
            urllib.request.urlretrieve(url, tmp)
            tmp.replace(dest)          # atomic: a kill leaves a .part, not a truncated mp4
            done += 1
            break
        except Exception as exc:
            if attempt == 3:
                print(f"  ! {item['path']}: {exc}", flush=True)
            time.sleep(2 * (attempt + 1))
    if i % 25 == 0:
        print(f"  {i}/{len(todo)}  ({done} have)", flush=True)
print(f"\n{done}/{len(todo)} clips -> {out}")
PY
