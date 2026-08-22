"""Resume and checksum-verify the original INCLUDE pose dataset from Zenodo."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import hashlib, json, subprocess, time, zipfile

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
URL = 'https://zenodo.org/api/records/6674324/files/INCLUDE.zip/content'
SIZE = 662887244
MD5 = 'f99f6f3ea50d5d94e0ffae44130a6672'
CHUNK = 8 * 1024 * 1024
PARTS = DATA / '.include-parts'
PARTS.mkdir(parents=True, exist_ok=True)
archive = DATA / 'INCLUDE.zip'

def checksum(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'md5').hexdigest()

if archive.exists() and archive.stat().st_size == SIZE and checksum(archive) == MD5:
    print('Archive already verified', flush=True)
else:
    # Retain complete chunks from the initial sequential download.
    if archive.exists() and archive.stat().st_size < SIZE:
        with archive.open('rb') as stream:
            for start in range(0, archive.stat().st_size // CHUNK * CHUNK, CHUNK):
                dest = PARTS / f'{start:012d}'
                block = stream.read(CHUNK)
                if not dest.exists(): dest.write_bytes(block)
    def fetch(start):
        end = min(start + CHUNK, SIZE) - 1
        dest = PARTS / f'{start:012d}'
        if dest.exists() and dest.stat().st_size == end-start+1: return dest
        partial = dest.with_suffix('.partial')
        headers = dest.with_suffix('.headers')
        for attempt in range(5):
            result = subprocess.run(['curl', '-sS', '-L', '--fail', '--connect-timeout', '20', '--max-time', '240',
                '--range', f'{start}-{end}', '-D', str(headers), '-o', str(partial), f'{URL}?part={start}'],capture_output=True,text=True)
            valid_range = headers.exists() and f'content-range: bytes {start}-{end}/{SIZE}' in headers.read_text().lower()
            if result.returncode == 0 and partial.exists() and partial.stat().st_size == end-start+1 and valid_range:
                partial.replace(dest); headers.unlink(missing_ok=True); return dest
            print(f'Retrying chunk {start // CHUNK}: {result.stderr.strip()[:120]}',flush=True)
            time.sleep(min(2 ** attempt,16))
        raise RuntimeError(f'Failed chunk {start}; completed chunks are retained for resume')
    starts = list(range(0,SIZE,CHUNK))
    completed = 0; began = time.monotonic()
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(fetch,start) for start in starts]
        for future in as_completed(futures):
            future.result(); completed += 1
            print(f'{completed}/{len(starts)} chunks ready ({time.monotonic()-began:.0f}s)',flush=True)
    combined = DATA / 'INCLUDE.verified-download'
    with combined.open('wb') as out:
        for start in starts:
            with (PARTS / f'{start:012d}').open('rb') as part:
                while block := part.read(1024*1024): out.write(block)
    if combined.stat().st_size != SIZE or checksum(combined) != MD5:
        raise RuntimeError('Checksum mismatch. Archive not accepted.')
    combined.replace(archive)
    print('Published MD5 checksum verified.',flush=True)

def safe_extract(path, destination):
    with zipfile.ZipFile(path) as z:
        for entry in z.infolist():
            target = (destination / entry.filename).resolve()
            if not target.is_relative_to(destination.resolve()) or (entry.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f'Unsafe archive entry: {entry.filename}')
        broken = z.testzip()
        if broken:
            raise ValueError(f'Corrupt member in {path.name}: {broken}')
        print(f'{path.name} CRC verified. Extracting...', flush=True)
        z.extractall(destination)

safe_extract(archive, DATA)
for nested in (DATA / 'INCLUDE' / 'Pose_Signs.zip', DATA / 'INCLUDE' / 'Train_Test_Split.zip'):
    safe_extract(nested, DATA)
(DATA/'INCLUDE-source.json').write_text(json.dumps({'source':'https://zenodo.org/records/6674324','download':URL,'bytes':SIZE,'md5':MD5,'verified':True},indent=2))
print('Extraction complete:',len(list(DATA.rglob('*.pkl'))),'pose clips',flush=True)
