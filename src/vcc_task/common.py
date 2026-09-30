"""Artifact identity and atomic, finite result serialization."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
NTC = 'non-targeting'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(16 << 20), b''):
            h.update(block)
    return h.hexdigest()


def ref(path):
    path = Path(path).resolve()
    return {'path': str(path.relative_to(ROOT)), 'sha256': sha256(path)}


def verified(record):
    path = ROOT / record['path']
    if ref(path) != record:
        raise ValueError(f'Frozen artifact changed: {path}')
    return path


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def stable_seed(seed, *parts):
    key = json.dumps([seed, *parts], ensure_ascii=False).encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], 'little')
