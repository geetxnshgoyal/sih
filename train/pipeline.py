#!/usr/bin/env python3
"""Resumable, clinical-first ISL landmark pipeline.

Examples:
    python train/pipeline.py sync --priority clinical --batch-size 100
    python train/pipeline.py status
    python train/pipeline.py validate
    python train/pipeline.py retry-failed --batch-size 100
    python train/pipeline.py export-playback

Source extractors stream temporary video, publish landmark files atomically,
and remove the video. This coordinator limits each run, records progress, and
creates a provenance manifest that can be resumed by re-running the command.
It does not train or replace the deployed browser model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "train"
DATA = ROOT / "data"
META = DATA / "meta"
STATE = META / "pipeline-state.json"
MANIFEST = META / "landmark-manifest.jsonl"
EXTRACTION_VERSION = 2


@dataclass(frozen=True)
class Source:
    name: str
    script: str
    output: str
    license: str
    source_url: str
    supports_tier: bool = True
    supports_batch: bool = True
    pattern: str = "*.npz"


SOURCES = (
    Source("include", "download_include.py", "Pose_Signs", "CC BY 4.0",
           "https://zenodo.org/records/6674324", supports_tier=False,
           supports_batch=False, pattern="*.pkl"),
    Source("ncert", "fetch_ncert.py", "ncert_landmarks",
           "NCERT official educational material; retain source attribution and terms",
           "https://www.youtube.com/channel/UCT0s92hGjqLX6p7qY9BBrSA"),
    Source("islgov", "extract_islgov.py", "islgov_landmarks", "MIT",
           "https://huggingface.co/datasets/silentone0725/Indian_Sign_Language_Data.gov_Rencoded"),
    Source("shiksha", "fetch_shiksha.py", "shiksha_landmarks",
           "Publisher educational material; retain source attribution and terms",
           "https://www.youtube.com/@ISHShiksha"),
    Source("cislr", "extract_cislr.py", "cislr_landmarks",
           "Research dataset; follow accepted CISLR terms", "https://exploration-lab.github.io/CISLR/"),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_state() -> dict:
    if not STATE.exists():
        return {"version": 1, "runs": [], "failures": []}
    try:
        value = json.loads(STATE.read_text())
        return value if isinstance(value, dict) else {"version": 1, "runs": [], "failures": []}
    except (OSError, json.JSONDecodeError):
        return {"version": 1, "runs": [], "failures": []}


def save_state(state: dict) -> None:
    META.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
    tmp.replace(STATE)


def count_outputs(source: Source) -> int:
    return sum(1 for _ in (DATA / source.output).rglob(source.pattern)) if (DATA / source.output).exists() else 0


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_metadata(path: Path) -> dict:
    try:
        import numpy as np
        with np.load(path, allow_pickle=False) as item:
            def scalar(name, default=None):
                return item[name].item() if name in item.files and item[name].size == 1 else default
            return {
                "source_url": str(scalar("source_url", "")),
                "signer": str(scalar("signer", scalar("source", ""))),
                "aspect": float(scalar("aspect", 1.0)),
                "fps": float(scalar("fps", 15.0)),
                "face_available": bool("face" in item.files and np.any(item["face"] != 0)),
            }
    except Exception:  # noqa: BLE001
        return {"source_url": "", "signer": "", "aspect": None, "fps": None, "face_available": False}


def validate_file(path: Path) -> tuple[bool, str]:
    try:
        import numpy as np
        with np.load(path, allow_pickle=False) as item:
            required = {"pose", "lh", "rh"}
            if not required.issubset(item.files):
                return False, "missing pose/lh/rh arrays"
            pose, left, right = item["pose"], item["lh"], item["rh"]
            if pose.ndim != 3 or pose.shape[1] < 23 or pose.shape[2] < 3:
                return False, f"invalid pose shape {pose.shape}"
            if left.shape != (len(pose), 21, 3) or right.shape != (len(pose), 21, 3):
                return False, f"invalid hand shapes {left.shape}/{right.shape}"
            if len(pose) < 8 or not all(np.isfinite(x).all() for x in (pose, left, right)):
                return False, "too short or non-finite landmarks"
            hands = np.any(left[..., :2] != 0, axis=(1, 2)) | np.any(right[..., :2] != 0, axis=(1, 2))
            if hands.mean() < .6:
                return False, "hands visible in fewer than 60% of frames"
            if "face" in item.files:
                face = item["face"]
                if face.ndim != 3 or len(face) != len(pose) or face.shape[2] < 3 or not np.isfinite(face).all():
                    return False, f"invalid face shape {face.shape}"
            if "aspect" in item.files and not .2 < float(item["aspect"]) < 5:
                return False, "invalid aspect ratio"
        return True, "ok"
    except Exception as error:  # noqa: BLE001
        return False, str(error)


def safe_label(value: str) -> str:
    return "".join(char if char.isalnum() or char in " _-" else "_" for char in value)[:120]


def catalogue_rows(source: Source) -> list[tuple[str, str, str]]:
    """Return (relative output stem, label, original URL) without network I/O."""
    if source.name in {"ncert", "shiksha"}:
        path = META / f"{source.name}.json"
        if not path.exists() and source.name == "ncert" and (DATA / "ncert-catalogue.json").exists():
            path = DATA / "ncert-catalogue.json"
        if not path.exists():
            return []
        items = json.loads(path.read_text())
        if not isinstance(items, list):
            return []
        return [(f"{safe_label(item['word'])}/{item['id']}", item["word"],
                 f"https://www.youtube.com/watch?v={item['id']}") for item in items]
    if source.name == "islgov":
        path = META / "_islgov_files.json"
        if not path.exists():
            return []
        rows = []
        for remote in json.loads(path.read_text()):
            stem = Path(remote).stem
            label = safe_label(stem)
            rows.append((f"{label}/{stem}", label,
                         f"https://huggingface.co/datasets/silentone0725/Indian_Sign_Language_Data.gov_Rencoded/blob/main/{remote}"))
        return rows
    path = DATA / "cislr/_todo.json"
    if source.name == "cislr" and path.exists():
        return [(f"{item['gloss']}/{item['uid']}", item["gloss"], item.get("path", ""))
                for item in json.loads(path.read_text())]
    return []


def write_manifest(validate: bool = False) -> dict:
    META.mkdir(parents=True, exist_ok=True)
    rows, invalid = [], 0
    completed_ids = set()
    for source in SOURCES:
        base = DATA / source.output
        catalogue = {relative: (label, url) for relative, label, url in catalogue_rows(source)}
        if base.exists():
            for path in sorted(base.rglob(source.pattern)):
                ok, detail = ((True, "verified upstream pose archive") if path.suffix == ".pkl"
                              else validate_file(path)) if validate else (True, "not checked")
                invalid += int(not ok)
                relative = path.relative_to(base).with_suffix('').as_posix()
                item_id = f"{source.name}:{relative}"
                completed_ids.add(item_id)
                metadata = (file_metadata(path) if path.suffix == ".npz" else
                            {"source_url": "", "signer": "", "aspect": 16 / 9,
                             "fps": 15.0, "face_available": False})
                rows.append({
                    "id": item_id,
                    "label": path.parent.name,
                    "variant": path.stem,
                    "source": source.name,
                    "source_url": metadata["source_url"] or catalogue.get(relative, ("", source.source_url))[1],
                    "license": source.license,
                    "signer": metadata["signer"] or None,
                    "aspect": metadata["aspect"], "fps": metadata["fps"],
                    "face_available": metadata["face_available"],
                    "landmarks": path.relative_to(ROOT).as_posix(),
                    "sha256": sha256(path),
                    "status": "complete" if ok else "failed",
                    "detail": detail,
                    "extraction_version": EXTRACTION_VERSION,
                })
        for relative, (label, url) in catalogue.items():
            item_id = f"{source.name}:{relative}"
            if item_id in completed_ids:
                continue
            rows.append({
                "id": item_id, "label": label, "source": source.name,
                "source_url": url, "license": source.license,
                "status": "pending", "extraction_version": EXTRACTION_VERSION,
            })
    tmp = MANIFEST.with_suffix(".tmp")
    with tmp.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    tmp.replace(MANIFEST)
    statuses = {name: sum(row["status"] == name for row in rows)
                for name in ("complete", "pending", "skipped", "failed")}
    return {**statuses, "manifest": str(MANIFEST.relative_to(ROOT))}


def selected_sources(name: str) -> list[Source]:
    if name == "all":
        return list(SOURCES)
    return [source for source in SOURCES if source.name == name]


def sync(args, retry_only: bool = False) -> int:
    state = load_state()
    remaining = args.batch_size
    tiers = [0, 1, 2]
    failures = {entry.get("source") for entry in state.get("failures", [])} if retry_only else set()
    run = {"started_at": utc_now(), "batch_size": args.batch_size,
           "priority": args.priority, "source": args.source, "steps": []}

    for tier in tiers:
        for source in selected_sources(args.source):
            if remaining <= 0:
                break
            if not source.supports_tier and tier > 0:
                continue
            if source.name == "include" and count_outputs(source):
                continue
            if retry_only and source.name not in failures:
                continue
            before = count_outputs(source)
            command = [sys.executable, str(TRAIN / source.script)]
            if source.supports_batch:
                command += ["--batch-size", str(remaining)]
            if source.supports_tier:
                command += ["--tier", str(tier)]
            print(f"\n[{source.name}] tier {tier}, up to {remaining} clips", flush=True)
            result = subprocess.run(command, cwd=ROOT, check=False)
            after = count_outputs(source)
            added = max(0, after - before)
            remaining = max(0, remaining - added)
            step = {"source": source.name, "tier": tier, "requested": remaining + added,
                    "added": added, "exit_code": result.returncode}
            run["steps"].append(step)
            state["failures"] = [entry for entry in state.get("failures", []) if entry.get("source") != source.name]
            if result.returncode:
                state.setdefault("failures", []).append({"source": source.name, "tier": tier,
                                                         "at": utc_now(), "exit_code": result.returncode})
        if remaining <= 0:
            break

    run["finished_at"] = utc_now()
    run["added"] = args.batch_size - remaining
    state.setdefault("runs", []).append(run)
    state["runs"] = state["runs"][-50:]
    save_state(state)
    result = write_manifest(validate=False)
    print(f"\nAdded {run['added']} clips. {result['complete']} landmark files indexed.")
    print("Re-run the same command to continue.")
    return 0 if run["added"] or not state.get("failures") else 1


def status() -> int:
    state = load_state()
    output = {source.name: count_outputs(source) for source in SOURCES}
    manifest_status = {}
    if MANIFEST.exists():
        rows = [json.loads(line) for line in MANIFEST.read_text().splitlines() if line.strip()]
        manifest_status = {name: sum(row.get("status") == name for row in rows)
                           for name in ("complete", "pending", "skipped", "failed")}
    print(json.dumps({"outputs": output, "total": sum(output.values()),
                      "manifest": manifest_status,
                      "failed_sources": state.get("failures", []),
                      "last_run": state.get("runs", [])[-1:]}, indent=2))
    return 0


def export_playback() -> int:
    exported = subprocess.run([sys.executable, str(TRAIN / "export_signs.py"), "--all"], cwd=ROOT)
    if exported.returncode:
        return exported.returncode
    return subprocess.run([sys.executable, str(TRAIN / "shard_playback.py")], cwd=ROOT).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("sync", "retry-failed"):
        command = sub.add_parser(name)
        command.add_argument("--priority", choices=["clinical"], default="clinical")
        command.add_argument("--batch-size", type=int, default=100)
        command.add_argument("--source", choices=["all", *(source.name for source in SOURCES)], default="all")
    sub.add_parser("status")
    sub.add_parser("validate")
    sub.add_parser("export-playback")
    args = parser.parse_args()
    if getattr(args, "batch_size", 1) < 1:
        parser.error("--batch-size must be positive")
    if args.command == "sync":
        return sync(args)
    if args.command == "retry-failed":
        return sync(args, retry_only=True)
    if args.command == "status":
        return status()
    if args.command == "validate":
        print(json.dumps(write_manifest(validate=True), indent=2))
        return 0
    return export_playback()


if __name__ == "__main__":
    raise SystemExit(main())
