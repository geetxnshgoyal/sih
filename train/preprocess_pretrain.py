"""
Build the pretraining set: MS-ASL + WLASL (+ AUTSL when its labels exist).

    .venv/bin/python train/preprocess_pretrain.py

Reads  data/poses/MS-ASL.zip, data/poses/WLASL.zip, data/meta/WLASL_v0.3.json
Writes data/pretrain.npz    X, y, labels, split, corpus

Why a different sign language is the right pretraining data
-----------------------------------------------------------
ARCHITECTURE.md 5.1: a model trained on INCLUDE scored 2.1% on CISLR against
0.38% chance. It had learned one corpus, seven students, one room, one camera , 
not the task. Two causes, and this script addresses the second:

  1. GEOMETRY: fixed in features.isotropic; already applied to the data here.
  2. VARIETY: INCLUDE cannot supply it and free ISL data of that kind does not
     exist. So borrow it from a language whose signs cannot leak answers.

What transfers is the encoder: what a handshape looks like on a webcam, how a
trajectory unfolds, which joints move together. The head is discarded and
retrained on ISL in train/pretrain.py. Measured: cross-corpus 4.9% -> 12.6%
from MS-ASL alone.

    corpus    clips    classes  signers  z?   labels
    MS-ASL    17,698     1,056    ~200   yes  directory name
    WLASL     21,083     2,000     119   yes  WLASL_v0.3.json (public, GitHub)
    AUTSL     36,305       226      43   NO   licence-gated, see below

MS-ASL and WLASL are both American Sign Language, so identical glosses are the
same sign and are merged into one class rather than kept separate, more clips
per class, which is what the encoder wants.

Both are already isotropic (nose-above-shoulders 0.559 and 0.572 in shoulder
widths, against ~0.578 true anatomy), so ASPECT is 1.0. It is passed explicitly
anyway: a silent default is what caused the original bug.

AUTSL is not used
-----------------
Two independent blockers, either one sufficient:

  - Its keypoints are (T, 75, 2). There is no z. Zero-filling it would teach the
    encoder that depth is always zero across most of its training data.
  - The pose release carries no labels, and the class mapping and split CSVs are
    behind the Ankara University licence agreement at cvml.ankara.edu.tr. They
    are not redistributable, so no fetch script can get them.

If you accept that licence and download train_labels.csv, drop it at
data/meta/autsl_train_labels.csv and this script will report what is still
missing. The z problem stands regardless, so treat AUTSL as a later experiment
rather than a quick win.
"""
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import features
from preprocess import SafeUnpickler

ROOT = Path(__file__).resolve().parent.parent
MSASL = ROOT / "data" / "poses" / "MS-ASL.zip"
WLASL = ROOT / "data" / "poses" / "WLASL.zip"
WLASL_META = ROOT / "data" / "meta" / "WLASL_v0.3.json"
AUTSL_LABELS = ROOT / "data" / "meta" / "autsl_train_labels.csv"
OUT = ROOT / "data" / "pretrain.npz"

ASPECT = 1.0        # both corpora verified isotropic; see the module docstring
MIN_FRAMES = 8      # fewer than this is a fragment, not a sign
MIN_PER_CLASS = 4   # a class the encoder cannot learn is noise in the head


def norm_gloss(s: str) -> str:
    return s.strip().lower().replace("_", " ")


def index_msasl() -> list[tuple]:
    """-> [(zip, member, gloss, split)]. Class is the directory name."""
    if not MSASL.exists():
        print(f"  ! missing {MSASL.relative_to(ROOT)}")
        return []
    z = zipfile.ZipFile(MSASL)
    out = []
    for n in z.namelist():
        if not n.endswith(".pkl"):
            continue
        p = n.split("/")
        if len(p) < 4:
            continue
        # MS-ASL/PKL_POSES/<split>/<class>/<clip>.pkl: the archive's own split
        # is signer-independent, which is what we want validation to measure.
        out.append((z, n, norm_gloss(p[-2]), "train" if p[-3] == "train" else "val"))
    print(f"  MS-ASL  {len(out)} clips")
    return out


def index_wlasl() -> list[tuple]:
    """-> [(zip, member, gloss, split)]. Class comes from the public metadata."""
    if not (WLASL.exists() and WLASL_META.exists()):
        print(f"  ! missing {WLASL.name} or {WLASL_META.name}, fetch the metadata:")
        print("      curl -sL -o data/meta/WLASL_v0.3.json \\")
        print("        https://raw.githubusercontent.com/dxli94/WLASL/master/"
              "start_kit/WLASL_v0.3.json")
        return []
    z = zipfile.ZipFile(WLASL)
    have = {Path(n).stem: n for n in z.namelist() if n.endswith(".pkl")}
    out = []
    for g in json.loads(WLASL_META.read_text()):
        for inst in g["instances"]:
            member = have.get(inst["video_id"])
            if member:
                out.append((z, member, norm_gloss(g["gloss"]),
                            "train" if inst["split"] == "train" else "val"))
    print(f"  WLASL   {len(out)} clips")
    return out


def main() -> int:
    print("indexing:")
    items = index_msasl() + index_wlasl()
    if not items:
        print("no pretraining data found")
        return 1
    if AUTSL_LABELS.exists():
        print(f"  note: {AUTSL_LABELS.name} found, but AUTSL is still unusable, "
              f"its keypoints have no z. See the module docstring.")

    counts: dict[str, int] = {}
    for _, _, g, _ in items:
        counts[g] = counts.get(g, 0) + 1
    keep = {g for g, k in counts.items() if k >= MIN_PER_CLASS}
    items = [it for it in items if it[2] in keep]
    labels = sorted(keep)
    label_id = {g: i for i, g in enumerate(labels)}
    print(f"\n{len(labels)} classes with >={MIN_PER_CLASS} clips  "
          f"({len(items)} clips total)")

    n = len(items)
    X = np.zeros((n, features.SEQ_LEN, features.N_POINTS, features.N_DIMS),
                 dtype=np.float32)
    y = np.zeros(n, dtype=np.int32)
    split = np.zeros(n, dtype="<U8")

    kept = short = bad = 0
    for i, (z, member, gloss, sp) in enumerate(items):
        if i % 4000 == 0 and i:
            print(f"  {i}/{n}", flush=True)
        try:
            obj = SafeUnpickler(io.BytesIO(z.read(member))).load()
            kp = obj["keypoints"]
            kp = kp[0] if kp.ndim == 4 else kp
            if kp.shape[0] < MIN_FRAMES or kp.shape[-1] < 3:
                short += 1
                continue
            seq = features.select_points(np.asarray(kp[..., :3], dtype=np.float64))
            seq = features.isotropic(seq, ASPECT)
            seq = features.resample(features.anchor(seq))
            if not np.all(np.isfinite(seq)):
                bad += 1
                continue
            X[kept], y[kept], split[kept] = seq, label_id[gloss], sp
            kept += 1
        except Exception:  # noqa: BLE001
            bad += 1

    X, y, split = X[:kept], y[:kept], split[:kept]
    print(f"\nkept {kept}  (dropped: {short} too short, {bad} unusable)")

    present = np.bincount(y, minlength=len(labels)) > 0
    if not present.all():
        remap = -np.ones(len(labels), dtype=np.int32)
        remap[present] = np.arange(present.sum(), dtype=np.int32)
        y = remap[y]
        labels = [labels[i] for i in np.flatnonzero(present)]
        print(f"dropped {int((~present).sum())} class(es) with no surviving clips")

    features.check_isotropy(X[:400], "MS-ASL + WLASL")

    c = np.bincount(y, minlength=len(labels))
    print(f"{len(labels)} classes | per class: min {c.min()} "
          f"median {int(np.median(c))} max {c.max()}")
    print(f"split: {int((split == 'train').sum())} train, "
          f"{int((split == 'val').sum())} val (signer-independent, corpus-defined)")
    print(f"X {X.shape}")

    np.savez_compressed(OUT, X=X, y=y, split=split, labels=np.array(labels))
    print(f"\nwrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
