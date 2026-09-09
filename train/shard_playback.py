#!/usr/bin/env python3
"""Split the playback bank into lazy-loaded, stable shards."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "app/public/model/_signs.json"
OUT = ROOT / "app/public/signs"

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=32)
    args = ap.parse_args()
    if args.size < 1:
        ap.error("--size must be positive")
    library = json.loads(SOURCE.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    glosses = sorted(library, key=str.casefold)
    mapping, shards = {}, []
    for offset in range(0, len(glosses), args.size):
        names = glosses[offset:offset + args.size]
        payload = {name: library[name] for name in names}
        raw = json.dumps(payload, separators=(",", ":")).encode()
        digest = hashlib.sha256(raw).hexdigest()[:12]
        filename = f"signs-{offset // args.size:04d}-{digest}.json"
        (OUT / filename).write_bytes(raw)
        shards.append(filename)
        mapping.update({name: filename for name in names})
    index = {"version": 1, "count": len(glosses), "glosses": mapping, "shards": shards}
    (OUT / "index.json").write_text(json.dumps(index, separators=(",", ":")))
    print(f"{len(glosses)} signs -> {len(shards)} lazy shards in {OUT.relative_to(ROOT)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
