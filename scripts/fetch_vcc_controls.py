#!/usr/bin/env python
"""Verify or restore the registered, immutable VCC controls snapshot.

Run with the user's authenticated vcc installation, for example:
  micromamba run -n virtual-cell python scripts/fetch_vcc_controls.py --verify-only

Registration is intentionally separate: upstream replacements must receive a new
source/version instead of silently changing this source's locked hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check(path: Path, entry: dict) -> None:
    if path.stat().st_size != entry["bytes"]:
        raise ValueError(f"Size mismatch: {path}")
    expected = entry["checksum"]
    if not expected.startswith("sha256:"):
        raise ValueError("A fixed SHA-256 is required")
    with path.open("rb") as fh:
        actual = hashlib.file_digest(fh, "sha256").hexdigest()
    if actual != expected.removeprefix("sha256:"):
        raise ValueError(f"SHA-256 mismatch: {path}; refusing to replace it")


def restore(source: dict, verify_only: bool) -> None:
    dest = ROOT / source.get("root", "data/raw") / source["id"]
    entries = source["files"]
    provenance_path = dest / "SOURCE.json"
    if provenance_path.exists():
        if json.loads(provenance_path.read_text()) != source["provenance"]:
            raise ValueError(f"Provenance mismatch: {provenance_path}")
    elif verify_only:
        raise FileNotFoundError(f"Missing provenance: {provenance_path}")
    for entry in entries:
        if Path(entry["name"]).name != entry["name"]:
            raise ValueError("Only flat archive member names are supported")
    missing = []
    for entry in entries:
        path = dest / entry["name"]
        if path.exists():
            check(path, entry)
        else:
            missing.append(entry)
    if missing and verify_only:
        raise FileNotFoundError(f"Missing locked files: {[e['name'] for e in missing]}")
    if missing:
        logs = ROOT / "data/logs"
        logs.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="vcc-restore-", dir=logs) as temporary:
            staging = Path(temporary)
            archive_entry = next(e for e in entries if not e.get("archive_member"))
            archive = dest / archive_entry["name"]
            if not archive.exists():
                vcc = Path(sys.executable).parent / "vcc"
                subprocess.run(
                    [str(vcc), "datasets", "download", source["dataset_id"],
                     "--endpoint", source["endpoint"], "--dir", str(staging), "--json"],
                    check=True,
                )
                archive = staging / archive_entry["name"]
                # A newer authenticated bundle must never replace this snapshot.
                check(archive, archive_entry)
            with zipfile.ZipFile(archive) as bundle:
                for entry in missing:
                    if not entry.get("archive_member"):
                        continue
                    member = entry["archive_member"]
                    if member != entry["name"] or bundle.getinfo(member).is_dir():
                        raise ValueError(f"Invalid archive member: {member}")
                    with bundle.open(member) as src, (staging / member).open("xb") as out:
                        shutil.copyfileobj(src, out, length=1 << 20)
                    check(staging / member, entry)
            # All recovered files pass before any are published; never overwrite.
            dest.mkdir(parents=True, exist_ok=True)
            for entry in missing:
                path = staging / entry["name"]
                path.chmod(0o444)
                os.link(path, dest / entry["name"])
    if not provenance_path.exists():
        with provenance_path.open("x") as fh:
            json.dump(source["provenance"], fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        provenance_path.chmod(0o444)
    print(f"{source['id']}: {len(entries)} locked files verified; {source['snapshot_version']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-id", default="arc_vcc2026_controls")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    lock = json.loads((ROOT / "data/registry.lock.json").read_text())
    source = next(s for s in lock["sources"] if s["id"] == args.source_id)
    if source["kind"] != "vcc_cli":
        raise ValueError("Expected an authenticated vcc_cli source")
    restore(source, args.verify_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
