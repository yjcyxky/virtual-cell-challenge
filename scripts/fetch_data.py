#!/usr/bin/env python
"""Fetch the VCC 2026 training corpus from data/registry.lock.json.

Contract:
  * idempotent      -- a file already on disk at the locked size is skipped
  * resumable       -- interrupted transfers continue with curl -C -
  * verified        -- size always, upstream checksum whenever the host publishes one
  * provenance      -- each dataset dir gets SOURCE.json; every file lands in MANIFEST.tsv
  * immutable       -- --freeze strips write permission from data/raw once verified

Usage:
  python scripts/fetch_data.py --tier a,c            # start small
  python scripts/fetch_data.py --tier a,b,c,d        # everything in the lock
  python scripts/fetch_data.py --only depmap_24q4 --dry-run
  python scripts/fetch_data.py --verify-only         # re-check what is on disk
  python scripts/fetch_data.py --freeze              # make data/raw read-only
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "data" / "registry.lock.json"
RAW = ROOT / "data" / "raw"
MANIFEST = ROOT / "data" / "MANIFEST.tsv"
LOGDIR = ROOT / "data" / "logs"

# Politeness caps. Zenodo throttles hard per connection but scales with streams;
# GEO's FTP frontend does the opposite and will drop you for hammering it.
HOST_SLOTS = {
    "storage.googleapis.com": 4,
    "zenodo.org": 6,
    "ndownloader.figshare.com": 4,
    "ftp.ncbi.nlm.nih.gov": 2,
    "huggingface.co": 8,
}
DEFAULT_SLOTS = 2
DISK_HEADROOM = 50 * 2**30      # refuse to start if this much would not remain free
ATTEMPTS = 4                    # outer retries per file (figshare 202 / zenodo 504)
RETRY_WAIT = 20                 # seconds between outer retries

def dest_dir(src: dict) -> Path:
    """Where a source lands: <root>/<id>, root defaulting to data/raw.

    The directory name is the source id and nothing else. Tier is a judgement we
    make about a dataset, not a property of it, so it lives in sources.json as a
    field and never in a path -- re-tiering is then a one-line edit rather than a
    move of frozen, read-only files.
    """
    return ROOT / src.get("root", "data/raw") / src["id"]


_sems: dict[str, threading.Semaphore] = {}
_sems_lock = threading.Lock()
_print_lock = threading.Lock()
_manifest_lock = threading.Lock()


def slot(url: str) -> threading.Semaphore:
    host = urlparse(url).netloc
    with _sems_lock:
        if host not in _sems:
            _sems[host] = threading.Semaphore(HOST_SLOTS.get(host, DEFAULT_SLOTS))
        return _sems[host]


def say(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def digest(path: Path, algo: str, chunk: int = 1 << 24) -> str:
    h = hashlib.new(algo)
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def checksum_ok(path: Path, spec: str | None) -> bool | None:
    """True/False when the spec is checkable, None when there is nothing to check."""
    if not spec:
        return None
    kind, _, want = spec.partition(":")
    if kind == "md5":
        return digest(path, "md5") == want
    if kind == "md5-b64":
        return base64.b64encode(bytes.fromhex(digest(path, "md5"))).decode() == want
    if kind == "sha256":
        return digest(path, "sha256") == want
    return None


def fetch_one(entry: dict, dest: Path, verify_sums: bool) -> dict:
    """Download if needed, verify, return a manifest row."""
    url, want = entry["url"], entry.get("bytes")
    dest.parent.mkdir(parents=True, exist_ok=True)
    row = {"file": str(dest.relative_to(ROOT)), "url": url, "status": "", "bytes": 0}

    if dest.exists() and want and dest.stat().st_size == want:
        row["status"] = "cached"
    else:
        # Outer retry loop on top of curl's own. Two upstream behaviours need it and
        # neither shows up as a curl error: figshare answers 202 with an empty body
        # while it stages a large file, and Zenodo returns a 504 under load. Both
        # leave a short file behind, so the size check is what decides success.
        for attempt in range(1, ATTEMPTS + 1):
            with slot(url):
                t0 = time.time()
                tag = "" if attempt == 1 else f" (retry {attempt - 1})"
                say(f"  ↓ {dest.name[:54]:54}{tag:12s} {(want or 0) / 2**30:7.2f} GB")
                proc = subprocess.run(
                    ["curl", "-fsSL", "-C", "-", "--retry", "5", "--retry-delay", "5",
                     "--retry-all-errors", "--connect-timeout", "30", "-o", str(dest), url],
                    capture_output=True, text=True)
            got = dest.stat().st_size if dest.exists() else 0
            if want and got == want:
                mb, dt = got / 2**20, max(time.time() - t0, 1e-6)
                row["status"] = "downloaded"
                say(f"  ✓ {dest.name[:58]:58} {mb / 1024:7.2f} GB  {mb / dt:5.1f} MB/s")
                break
            if not want and proc.returncode == 0 and got:
                row["status"] = "downloaded"      # no locked size to check against
                break
            why = (f"curl {proc.returncode}: {proc.stderr.strip()[:90]}"
                   if proc.returncode else f"short read {got}/{want}")
            if attempt == ATTEMPTS:
                row["status"] = f"FAILED({why})"
                say(f"  ✗ {dest.name}: {why}")
                return row
            say(f"  … {dest.name[:48]:48} {why} — retrying in {RETRY_WAIT}s")
            if got == 0 and dest.exists():
                dest.unlink()                     # don't resume from an empty 202 body
            time.sleep(RETRY_WAIT)

    size = dest.stat().st_size
    row["bytes"] = size
    if want and size != want:
        row["status"] = f"SIZE-MISMATCH(got {size}, want {want})"
        return row

    if verify_sums:
        ok = checksum_ok(dest, entry.get("checksum"))
        if ok is False:
            row["status"] = "CHECKSUM-MISMATCH"
            return row
        row["checksum_verified"] = "yes" if ok else "no-upstream-sum"
    row["sha256"] = digest(dest, "sha256") if verify_sums else ""
    return row


def write_source_json(src: dict, dest_dir: Path, rows: list[dict]) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    locked_files = {f["name"]: f for f in src["files"]}
    files = []
    for row in rows:
        name = str(Path(row["file"]).relative_to(dest_dir.relative_to(ROOT)))
        entry = locked_files[name]
        files.append({
            "name": name, "bytes": row["bytes"],
            "sha256": row.get("sha256", ""), "url": row["url"],
            **{k: entry[k] for k in ("checksum", "origin_url", "checksum_source")
               if entry.get(k) is not None},
        })
    (dest_dir / "SOURCE.json").write_text(json.dumps({
        "id": src["id"],
        "tier": src["tier"],
        "title": src["title"],
        "citation": src["citation"],
        "license": src["license"],
        "why_this_project": src["why"],
        "caveat": src.get("caveat"),
        "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "retrieved_by": "scripts/fetch_data.py from data/registry.lock.json",
        "n_files": len(rows),
        "bytes": sum(r["bytes"] for r in rows),
        "files": files,
        **{key: src[key] for key in ("kind", "release", "organism", "count_feature", "snapshot_version",
                                    "selection_manifest", "selection_manifest_sha256", "selection") if key in src},
    }, indent=2) + "\n")


def append_manifest(rows: list[dict]) -> None:
    cols = ["file", "bytes", "sha256", "checksum_verified", "status", "url"]
    with _manifest_lock:
        new = not MANIFEST.exists()
        with MANIFEST.open("a") as fh:
            if new:
                fh.write("\t".join(cols) + "\n")
            for r in rows:
                fh.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")


def freeze(path: Path) -> int:
    n = 0
    for p in path.rglob("*"):
        if p.is_file():
            p.chmod(p.stat().st_mode & ~0o222)
            n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="a,b,c,d")
    ap.add_argument("--only", help="comma-separated source ids")
    ap.add_argument("--jobs", type=int, default=10, help="global worker cap")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify-only", action="store_true", help="no downloads; re-hash what exists")
    ap.add_argument("--no-verify-checksum", action="store_true",
                    help="skip hashing (much faster, weaker guarantee)")
    ap.add_argument("--freeze", action="store_true", help="chmod a-w data/raw and exit")
    args = ap.parse_args()

    LOGDIR.mkdir(parents=True, exist_ok=True)
    if args.freeze:
        print(f"froze {freeze(RAW)} files under {RAW}")
        return 0

    lock = json.loads(LOCK.read_text())
    tiers = set(args.tier.split(","))
    only = set(args.only.split(",")) if args.only else None
    todo = [s for s in lock["sources"]
            if s["tier"] in tiers and (only is None or s["id"] in only)]

    planned = sum(f["bytes"] or 0 for s in todo for f in s["files"])
    have = sum((dest_dir(s) / f["name"]).stat().st_size
               for s in todo for f in s["files"]
               if (dest_dir(s) / f["name"]).exists())
    free = shutil.disk_usage(ROOT).free
    print(f"plan: {len(todo)} sources, {sum(len(s['files']) for s in todo)} files, "
          f"{planned / 2**30:.1f} GB   on disk already {have / 2**30:.1f} GB")
    print(f"disk: {free / 2**30:.0f} GB free, need ~{(planned - have) / 2**30:.0f} GB")
    if not args.dry_run and free - (planned - have) < DISK_HEADROOM:
        print(f"REFUSING: would leave under {DISK_HEADROOM / 2**30:.0f} GB free", file=sys.stderr)
        return 2
    if args.dry_run:
        for s in todo:
            print(f"  [{s['tier']}] {s['id']:16s} {len(s['files']):4d} files  "
                  f"{sum(f['bytes'] or 0 for f in s['files']) / 2**30:8.2f} GB  "
                  f"-> {dest_dir(s).relative_to(ROOT)}")
        return 0

    verify = not args.no_verify_checksum
    failures = []
    for src in todo:
        if src["kind"] == "vcc_cli":
            cmd = [sys.executable, str(ROOT / "scripts/fetch_vcc_controls.py"),
                   "--source-id", src["id"]]
            if args.verify_only:
                cmd.append("--verify-only")
            # This path verifies read-only snapshots without rewriting SOURCE.json.
            # Authentication is resolved by vcc; signed URLs never enter the lock.
            if subprocess.run(cmd, check=False).returncode:
                failures.append({"status": "VCC-RESTORE-FAILED", "file": src["id"]})
            continue
        dest = dest_dir(src)
        say(f"\n=== [{src['tier']}] {src['id']} -> {dest.relative_to(ROOT)} "
            f"({len(src['files'])} files) ===")
        rows: list[dict] = []
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futs = {}
            for f in src["files"]:
                path = dest / f["name"]
                if args.verify_only and not path.exists():
                    continue
                futs[pool.submit(fetch_one, f, path, verify)] = f
            for fut in as_completed(futs):
                rows.append(fut.result())
        bad = [r for r in rows if r["status"] not in ("cached", "downloaded")]
        failures += bad
        write_source_json(src, dest, rows)
        append_manifest(rows)
        say(f"  {src['id']}: {len(rows) - len(bad)}/{len(rows)} ok, "
            f"{sum(r['bytes'] for r in rows) / 2**30:.2f} GB")

    print(f"\ndone. manifest -> {MANIFEST.relative_to(ROOT)}")
    if failures:
        print(f"{len(failures)} FAILED:", file=sys.stderr)
        for r in failures:
            print(f"  {r['status']:34s} {r['file']}", file=sys.stderr)
        return 1
    print("all files verified. run --freeze to make data/raw read-only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
