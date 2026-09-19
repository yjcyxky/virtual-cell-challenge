#!/usr/bin/env python
"""Read-only inventory and content audit against lock, SOURCE and MANIFEST.

Writes only the explicitly requested report. Does not invoke fetch_data.py,
because its legacy verify-only path rewrites provenance for public sources.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROGRESS_LOCK = threading.Lock()
BYTES_READ = 0


def audit_file(source: dict, entry: dict, provenance: list, latest: dict) -> dict:
    global BYTES_READ
    relative = Path(source.get("root", "data/raw")) / source["id"] / entry["name"]
    path = ROOT / relative
    result = {"source_id": source["id"], "file": str(relative), "issues": []}
    if not path.is_file():
        result["issues"].append("missing_file")
        return result
    before = path.stat()
    result.update(bytes=before.st_size, writable=bool(before.st_mode & 0o222))
    if entry.get("bytes") != before.st_size:
        result["issues"].append("locked_size_mismatch")
    matches = [f for f in provenance if f["name"] == entry["name"]]
    result["source_path_match"] = "relative_path"
    if not matches:
        # Historical fetcher stored only basenames, even for nested HF files.
        matches = [f for f in provenance if f["name"] == Path(entry["name"]).name
                   and f.get("url") == entry.get("url")]
        result["source_path_match"] = "basename_and_url"
    source_file = matches[0] if len(matches) == 1 else None
    if source_file is None:
        result["issues"].append("source_entry_missing_or_ambiguous")
    manifest_file = latest.get(str(relative))
    if manifest_file is None:
        result["issues"].append("manifest_entry_missing")
    spec = entry.get("checksum") or ""
    algorithm, _, expected = spec.partition(":")
    sha = hashlib.sha256()
    md5 = hashlib.md5() if algorithm in ("md5", "md5-b64") else None
    with path.open("rb") as fh:
        while chunk := fh.read(16 << 20):
            sha.update(chunk)
            if md5 is not None:
                md5.update(chunk)
            with PROGRESS_LOCK:
                BYTES_READ += len(chunk)
    actual = sha.hexdigest()
    result["sha256"] = actual
    for name, record in [("source", source_file), ("manifest", manifest_file)]:
        if record is not None and record.get("sha256") != actual:
            result["issues"].append(f"{name}_sha256_mismatch")
    upstream = None
    if algorithm == "sha256":
        upstream = actual == expected
    elif algorithm == "md5":
        upstream = md5.hexdigest() == expected
    elif algorithm == "md5-b64":
        upstream = base64.b64encode(md5.digest()).decode() == expected
    elif spec:
        result["issues"].append("unsupported_locked_checksum")
    result["locked_checksum_verified"] = upstream
    if upstream is False:
        result["issues"].append("locked_checksum_mismatch")
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (
        after.st_size, after.st_mtime_ns, after.st_ino
    ):
        result["issues"].append("changed_during_read")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--only", help="comma-separated source ids; default: all sources")
    args = parser.parse_args()
    started = datetime.now(timezone.utc).isoformat()
    inputs = {p: (ROOT / p).read_bytes() for p in [
        "docs/README.md", "data/sources.json", "data/registry.lock.json", "data/MANIFEST.tsv"
    ]}
    lock = json.loads(inputs["data/registry.lock.json"])
    if args.only:
        selected = set(args.only.split(","))
        unknown = selected - {s["id"] for s in lock["sources"]}
        if unknown:
            parser.error(f"Unknown source ids: {', '.join(sorted(unknown))}")
        lock["sources"] = [s for s in lock["sources"] if s["id"] in selected]
    latest = {}
    for row in csv.DictReader(inputs["data/MANIFEST.tsv"].decode().splitlines(), delimiter="\t"):
        latest[row["file"]] = row
    jobs = []
    sources = []
    for source in lock["sources"]:
        root = ROOT / source.get("root", "data/raw") / source["id"]
        source_path = root / "SOURCE.json"
        provenance = json.loads(source_path.read_text()) if source_path.is_file() else {}
        sources.append({"id": source["id"], "tier": source["tier"],
                        "directory": str(root.relative_to(ROOT)),
                        "source_json_present": source_path.is_file(),
                        "locked_files": len(source["files"]),
                        "locked_bytes": sum(e.get("bytes") or 0 for e in source["files"])})
        jobs.extend((source, entry, provenance.get("files", [])) for entry in source["files"])
    jobs.sort(key=lambda item: item[1].get("bytes") or 0, reverse=True)
    total = sum(s["locked_bytes"] for s in sources)
    results = []
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        pending = {executor.submit(audit_file, s, e, p, latest) for s, e, p in jobs}
        last_print = t0
        while pending:
            completed, pending = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
            results.extend(f.result() for f in completed)
            if time.monotonic() - last_print >= 15 or not pending:
                print(f"{len(results)}/{len(jobs)} files; {BYTES_READ / 1e9:.1f}/{total / 1e9:.1f} GB read", flush=True)
                last_print = time.monotonic()
    for source in sources:
        files = [r for r in results if r["source_id"] == source["id"]]
        source.update(verified_files=sum(not r["issues"] for r in files),
                      writable_files=sum(r.get("writable", False) for r in files),
                      basename_only_provenance=sum(r.get("source_path_match") == "basename_and_url" for r in files))
    unchanged = all((ROOT / p).read_bytes() == content for p, content in inputs.items())
    report = {"started_at": started, "completed_at": datetime.now(timezone.utc).isoformat(),
              "duration_seconds": round(time.monotonic() - t0, 2), "workers": args.workers,
              "input_sha256": {p: hashlib.sha256(content).hexdigest() for p, content in inputs.items()},
              "registry_inputs_unchanged": unchanged, "files": len(results), "bytes_read": BYTES_READ,
              "issue_files": sum(bool(r["issues"]) for r in results), "sources": sources,
              "file_results": sorted(results, key=lambda r: r["file"])}
    with args.output.open("x") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print(f"Report: {args.output}; issue files: {report['issue_files']}; inputs unchanged: {unchanged}")
    return 0 if unchanged and not report["issue_files"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
