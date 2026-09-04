"""
Translate the gloss table into India's languages, locally and for free.

    .venv-tf/bin/python train/translate_glosses.py

Reads  app/tools/glossEnglish.ts    (hand-written English, reviewable)
Writes app/public/model/_utterances.json

Why this exists in this shape
-----------------------------
The first version of this step called a paid LLM API. That is the wrong
dependency for this project twice over: a hospital or government deployment
should not need a commercial API key, and the demo must not depend on a network
call it cannot control. So the job is split:

  1. gloss sequence -> English      hand-written in app/tools/glossEnglish.ts
  2. English -> hi/ta/te/bn/mr      this script, NLLB-200, on this machine

Step 1 needs judgement about what a signer MEANS ("I Doctor" is a request, not
a statement) and is short enough to review by eye. Step 2 is mechanical, and a
purpose-built open translation model does it better than a general one.

NLLB-200 distilled-600M is used because it is genuinely UNGATED — no Hugging
Face account, no token, no licence click. AI4Bharat's IndicTrans2 is the better
model for Indic specifically, and is also free and open, but it is gated behind
an account, so it is offered here as an opt-in rather than the default.

Everything runs offline after the first download. Nothing is sent anywhere.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENGLISH_TS = ROOT / "app" / "tools" / "glossEnglish.ts"
OUT = ROOT / "app" / "public" / "model" / "_utterances.json"

# NLLB uses FLORES-200 codes, which carry the script as well as the language —
# Devanagari vs Bengali vs Telugu are different writing systems, and the model
# needs to be told which one to produce.
LANGS = {
    "hi-IN": "hin_Deva",
    "ta-IN": "tam_Taml",
    "te-IN": "tel_Telu",
    "bn-IN": "ben_Beng",
    "mr-IN": "mar_Deva",
}

MODEL = "facebook/nllb-200-distilled-600M"


def load_english() -> dict[str, str]:
    """Parse the hand-written table out of the TS file.

    Deliberately a regex over one known-shape file rather than a build step:
    keeping English in TypeScript means the app and this script read the same
    source, and there is no generated intermediate to fall out of date.
    """
    src = ENGLISH_TS.read_text()
    body = src[src.index("GLOSS_ENGLISH"):]
    pairs = re.findall(r'^\s{2}"([^"]+)":\s*"([^"]+)",\s*$', body, re.M)
    if not pairs:
        sys.exit(f"no entries parsed from {ENGLISH_TS.relative_to(ROOT)}")
    return dict(pairs)


def load_meta() -> dict[str, dict]:
    """Per-sequence metadata the app needs alongside the translations.

    `reordered` says whether spoken order actually differs from signed order —
    the UI uses it to distinguish a real reordering from a passthrough, so it
    must be computed, not assumed. `register` drives nothing yet but is cheap to
    record and is the difference between "I need a doctor" and "Where is the
    doctor?" being treated the same.
    """
    corpus = (ROOT / "app" / "tools" / "glossCorpus.ts").read_text()
    body = corpus[corpus.index("GLOSS_CORPUS"):]
    out = {}
    for m in re.finditer(r'glosses:\s*\[([^\]]+)\][^}]*?domain:\s*"(\w+)"', body):
        gl = [g.strip().strip('"') for g in m.group(1).split(",")]
        out["|".join(gl)] = {"glosses": gl, "domain": m.group(2)}
    return out


def _dedupe(text: str) -> str:
    """Collapse "X. X." into "X."

    NLLB repeats itself on very short inputs — a one-word source like "Hello."
    reliably produced "హలో. హలో." in Telugu. Caught by scanning all 365
    translations, where it was the only defect; cheap to fix, and it would look
    like a broken app rather than a quirk of the model.
    """
    parts = [p.strip() for p in re.split(r"[.।?!]", text) if p.strip()]
    if len(parts) > 1 and len(set(parts)) == 1:
        tail = text[len(parts[0]):].lstrip()
        punct = tail[0] if tail and tail[0] in ".।?!" else ""
        return parts[0] + punct
    return text


def _write(entries: dict) -> None:
    """Emit the wrapper shape the app validates against."""
    import datetime
    OUT.write_text(json.dumps({
        "generated": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "model": MODEL,
        "entries": entries,
    }, ensure_ascii=False, indent=1))


def main() -> int:
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError:
        print("missing dependencies. Install them into the TF venv:\n"
              "  .venv-tf/bin/pip install torch transformers sentencepiece sacremoses")
        return 1

    english = load_english()
    print(f"{len(english)} sequences to translate into {len(LANGS)} languages")

    # The app expects {generated, model, entries:{...}} — NOT a flat map. It
    # validates that shape on load and silently degrades to the phrasebook if it
    # is wrong, so a flat map would look like the generator never ran. Contract
    # is in app/src/lib/glossTranslate.ts (interface Table).
    prev = json.loads(OUT.read_text()) if OUT.exists() else {}
    existing: dict = prev.get("entries", {}) if isinstance(prev, dict) else {}
    print(f"{len(existing)} already present in {OUT.relative_to(ROOT)}")

    meta = load_meta()

    # Show the download. Without this the script prints nothing for ~10 minutes
    # while it pulls 2.4 GB, which is indistinguishable from being hung — and a
    # silent long-running job gets killed by whoever is watching it.
    print(f"loading {MODEL}")
    print("  first run downloads ~2.4 GB and caches it in ~/.cache/huggingface;")
    print("  every run after this is offline and instant.")
    import os
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_info()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)
    hf_logging.set_verbosity_warning()
    # Apple Silicon: mps is roughly 3-4x cpu here and needs no extra setup.
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device).eval()
    print(f"device: {device}")

    out = dict(existing)
    keys = list(english)
    for i, key in enumerate(keys, 1):
        if key in out and all(l in out[key] for l in LANGS):
            continue
        src = english[key]
        entry = out.get(key, {"en-IN": src})
        entry["en-IN"] = src

        for lang, flores in LANGS.items():
            if lang in entry:
                continue
            enc = tok(src, return_tensors="pt").to(device)
            with torch.no_grad():
                gen = model.generate(
                    **enc,
                    forced_bos_token_id=tok.convert_tokens_to_ids(flores),
                    max_new_tokens=64,
                    num_beams=4,
                )
            text = tok.batch_decode(gen, skip_special_tokens=True)[0].strip()
            entry[lang] = _dedupe(text)

        m = meta.get(key, {"glosses": key.split("|"), "domain": "unknown"})
        entry["glosses"] = m["glosses"]
        entry["domain"] = m["domain"]
        # Spoken order differs from signed order whenever the English is not
        # simply the glosses concatenated.
        entry["reordered"] = src.rstrip(".?!").lower() != " ".join(m["glosses"]).lower()
        entry["register"] = ("question" if src.rstrip().endswith("?")
                             else "request" if " need " in src.lower()
                             else "statement")
        out[key] = entry
        print(f"  [{i}/{len(keys)}] {key:34s} {entry['hi-IN'][:34]}")
        # Write as we go: a long run interrupted halfway keeps its work.
        _write(out)

    _write(out)
    print(f"\nwrote {len(out)} entries to {OUT.relative_to(ROOT)}")
    print(f"size: {OUT.stat().st_size/1024:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
