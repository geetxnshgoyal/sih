"""Fetch a small, CRC-verified raw-video sample from official INCLUDE ZIPs.

HTTP ranges avoid downloading the entire 57 GB video corpus. These clips are
for camera-extractor regression checks, not independent accuracy claims.
"""
import argparse
import io
import json
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class RemoteZip(io.RawIOBase):
    def __init__(self, url, size):
        self.url, self.size, self.pos = url, size, 0
        self.start, self.cache = 0, b""

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset + (self.pos if whence == 1 else self.size if whence == 2 else 0)
        if not 0 <= self.pos <= self.size: raise ValueError("Invalid archive offset")
        return self.pos

    def read(self, n=-1):
        n = self.size - self.pos if n < 0 else min(n, self.size - self.pos)
        if not n: return b""
        if not (self.start <= self.pos and self.pos + n <= self.start + len(self.cache)):
            # Small resumable ranges avoid losing an entire 15MB video on slow
            # connections. zipfile.read() will request the remaining bytes.
            count = min(max(n, 65536), 1024 * 1024, self.size - self.pos)
            result = subprocess.run([
                "curl", "--fail", "--silent", "--show-error", "--location",
                "--retry", "3", "--max-time", "180", "--max-filesize", str(count),
                "--range", f"{self.pos}-{self.pos+count-1}", self.url,
            ], check=True, capture_output=True)
            if len(result.stdout) != count: raise ValueError("Server ignored byte range")
            self.start, self.cache = self.pos, result.stdout
        value = self.cache[self.pos-self.start:self.pos-self.start+n]
        self.pos += len(value)
        return value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-word', type=int, default=2)
    args = ap.parse_args()
    meta_url = 'https://zenodo.org/api/records/4010759'
    meta = json.loads(subprocess.check_output(['curl', '-fsSL', meta_url]))
    out = ROOT / 'data/camera_validation'
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for spec in meta['files']:
        if not spec['key'].startswith('Greetings_'): continue
        print('Reading', spec['key'], flush=True)
        with zipfile.ZipFile(RemoteZip(spec['links']['self'], spec['size'])) as archive:
            counts = {}
            for entry in archive.infolist():
                parts = Path(entry.filename).parts
                if len(parts) < 2 or Path(entry.filename).suffix.lower() not in {'.mp4', '.mov', '.avi'}: continue
                label = parts[-2].split('.', 1)[-1].strip()
                if label.lower() not in {'hello', 'how are you', 'thank you', 'good morning'}: continue
                if counts.get(label, 0) >= args.per_word: continue
                counts[label] = counts.get(label, 0) + 1
                dest = out / label / Path(entry.filename).name
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists():
                    print('Downloading', label, dest.name, round(entry.compress_size/1e6, 1), 'MB', flush=True)
                    value = archive.read(entry)  # zipfile verifies the entry CRC
                    temporary = dest.with_suffix(dest.suffix + '.part')
                    temporary.write_bytes(value)
                    temporary.replace(dest)
                manifest.append({'label': label, 'file': str(dest.relative_to(ROOT)),
                                 'archive': spec['key'], 'entry': entry.filename,
                                 'crc32': entry.CRC, 'source': meta_url,
                                 'license': meta.get('metadata', {}).get('license')})
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print('Downloaded', len(manifest), 'source videos', flush=True)


if __name__ == '__main__': main()
