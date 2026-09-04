"""
Translate the phrase board into India's languages, locally and for free.

    .venv-tf/bin/python train/translate_phrasebook.py

Reads  app/src/lib/phrasebook.ts
Writes app/public/model/_phrasebook.json

Same split as train/translate_glosses.py and for the same reason: the English is
hand-written where judgement is needed, and a local open model does the
mechanical part. NLLB-200 distilled-600M, ungated, offline after first download.

This table is the product's reliable path. Recognition is 40.4% correct on an
unseen signer; a tapped phrase is 100%. So these translations matter more than
the model's, and every one of them should eventually be reviewed by a Deaf ISL
user rather than trusted because a machine produced it.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "app" / "src" / "lib" / "phrasebook.ts"
OUT = ROOT / "app" / "public" / "model" / "_phrasebook.json"

LANGS = {
    "hi-IN": "hin_Deva",
    "ta-IN": "tam_Taml",
    "te-IN": "tel_Telu",
    "bn-IN": "ben_Beng",
    "mr-IN": "mar_Deva",
}
MODEL = "facebook/nllb-200-distilled-600M"


def load_phrases() -> dict[str, str]:
    src = SRC.read_text()
    pairs = re.findall(r'\{\s*id:\s*"([^"]+)",\s*en:\s*"([^"]+)"', src)
    if not pairs:
        sys.exit(f"no phrases parsed from {SRC.relative_to(ROOT)}")
    return dict(pairs)


def dedupe(text: str) -> str:
    """Collapse a phrase repeated back to back.

    NLLB does this reliably on very short inputs, which is exactly the shape of
    the most important phrases here: "Yes." came back as "हाँ, हाँ।" and "No."
    as "না, না, না।". Note the separator is sometimes a COMMA rather than a
    sentence ender — splitting only on [.।?!] misses those, which is how the
    first pass let six through.
    """
    parts = [p.strip() for p in re.split(r"[.।?!,;]", text) if p.strip()]
    if len(parts) > 1 and len(set(parts)) == 1:
        tail = text[len(parts[0]):].lstrip(" ,;")
        return parts[0] + (tail[0] if tail and tail[0] in ".।?!" else "")
    return text


def main() -> int:
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError:
        print("run: .venv-tf/bin/pip install torch transformers sentencepiece sacremoses")
        return 1

    phrases = load_phrases()
    prev = json.loads(OUT.read_text()) if OUT.exists() else {}
    entries = prev.get("entries", {}) if isinstance(prev, dict) else {}
    print(f"{len(phrases)} phrases x {len(LANGS)} languages; {len(entries)} already done")

    todo = [k for k in phrases if k not in entries
            or not all(l in entries[k] for l in LANGS)]
    if not todo:
        print("nothing to do")
        return 0

    print(f"loading {MODEL} (cached after first run)...")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device).eval()
    print(f"device: {device}, {len(todo)} to translate")

    import datetime

    def write():
        OUT.write_text(json.dumps({
            "generated": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
            "model": MODEL,
            "entries": entries,
        }, ensure_ascii=False, indent=1))

    for i, key in enumerate(todo, 1):
        src = phrases[key]
        e = entries.get(key, {})
        e["en-IN"] = src
        for lang, flores in LANGS.items():
            if lang in e:
                continue
            enc = tok(src, return_tensors="pt").to(device)
            with torch.no_grad():
                gen = model.generate(**enc,
                                     forced_bos_token_id=tok.convert_tokens_to_ids(flores),
                                     max_new_tokens=64, num_beams=4)
            e[lang] = dedupe(tok.batch_decode(gen, skip_special_tokens=True)[0].strip())
        entries[key] = e
        print(f"  [{i}/{len(todo)}] {key:22s} {e['hi-IN'][:38]}")
        write()

    write()
    print(f"\nwrote {len(entries)} phrases to {OUT.relative_to(ROOT)}  ({OUT.stat().st_size//1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
