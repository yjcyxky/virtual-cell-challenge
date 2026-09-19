#!/usr/bin/env python
"""Expand data/sources.json into data/registry.lock.json.

Queries each upstream API once, pinning every file's URL, byte size and (where the
host publishes one) an upstream checksum. Downstream, fetch_data.py reads only the
lock -- so a fetch is reproducible even if an upstream record gains or loses files,
and any drift shows up as a diff in the lock rather than as a silent change on disk.

Usage:  python scripts/build_lock.py [--only ID[,ID...]]
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "data" / "sources.json"
LOCK = ROOT / "data" / "registry.lock.json"
UA = {"User-Agent": "vcc2026-corpus-builder/1.0"}


def get_json(url: str, payload: dict | None = None) -> dict | list:
    data = json.dumps(payload).encode() if payload else None
    headers = dict(UA)
    if payload:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.load(r)


def head_size(url: str) -> int | None:
    req = urllib.request.Request(url, headers=UA, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            n = r.headers.get("Content-Length")
            return int(n) if n else None
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
        return None


# --- per-kind expanders: each returns a list of {name, url, bytes, checksum} ----

def expand_gcs_public(src: dict) -> list[dict]:
    out = []
    for obj in src["objects"]:
        meta = get_json(
            f"https://storage.googleapis.com/storage/v1/b/{src['bucket']}/o/"
            + urllib.parse.quote(obj, safe="")
        )
        out.append({
            "name": obj.rsplit("/", 1)[-1],
            "url": f"https://storage.googleapis.com/{src['bucket']}/{obj}",
            "bytes": int(meta["size"]),
            "checksum": f"md5-b64:{meta['md5Hash']}" if "md5Hash" in meta else None,
        })
    return out


def expand_zenodo(src: dict) -> list[dict]:
    rec = get_json(f"https://zenodo.org/api/records/{src['record']}")
    pats = src.get("include", ["*"])
    out = []
    for f in rec["files"]:
        key = f["key"]
        if not any(fnmatch.fnmatch(key, p) for p in pats):
            continue
        out.append({
            "name": key,
            "url": f"https://zenodo.org/api/records/{src['record']}/files/"
                   f"{urllib.parse.quote(key)}/content",
            "bytes": int(f["size"]),
            "checksum": f.get("checksum"),
        })
    return out


def expand_figshare(src: dict) -> list[dict]:
    # Always take the API's own download_url. Both figshare.com/ndownloader/... and
    # plus.figshare.com/ndownloader/... answer 202 with an empty body; the canonical
    # host is ndownloader.figshare.com for regular and "plus" articles alike.
    art = get_json(f"https://api.figshare.com/v2/articles/{src['article']}")
    wanted = set(src["include_names"])
    by_name = {f["name"]: f for f in art.get("files", [])}
    missing = wanted - by_name.keys()
    if missing:
        print(f"  ! {src['id']}: not in article: {sorted(missing)}", file=sys.stderr)
    out = []
    for name in src["include_names"]:
        f = by_name.get(name)
        if not f:
            continue
        out.append({
            "name": name,
            "url": f.get("download_url") or f"https://ndownloader.figshare.com/files/{f['id']}",
            "bytes": int(f["size"]),
            "checksum": f"md5:{f['supplied_md5']}" if f.get("supplied_md5") else None,
        })
    return out


def expand_http(src: dict) -> list[dict]:
    out = []
    for f in src["files"]:
        size = head_size(f["url"])
        pinned_size = f.get("bytes")
        if pinned_size is not None and size is not None and pinned_size != size:
            raise ValueError(f"{f['name']}: HTTP size {size} != pinned size {pinned_size}")
        out.append({
            "name": f["name"],
            "url": f["url"],
            "bytes": pinned_size if pinned_size is not None else size,
            "checksum": f.get("checksum"),
            **{k: f[k] for k in ("origin_url", "checksum_source") if k in f},
        })
    return out


def _hf_files(repo: str, kind: str) -> dict[str, dict]:
    from huggingface_hub import HfApi
    info = HfApi().repo_info(repo, repo_type=kind, files_metadata=True)
    return {s.rfilename: s for s in info.siblings}


def expand_hf(src: dict, kind: str) -> list[dict]:
    files = _hf_files(src["repo"], kind)
    names = []
    for pat in src.get("patterns", ["*"]):
        names += [n for n in files if fnmatch.fnmatch(n, pat)]
    if "shard_glob" in src:                       # strided subsample of a sharded set
        stride, want = src.get("shard_stride", 1), src["n_shards"]
        cand = [src["shard_glob"].format(i=i * stride) for i in range(want)]
        names += [n for n in cand if n in files]
    out, seen = [], set()
    prefix = ("datasets/" if kind == "dataset" else "") + src["repo"]
    for n in names:
        if n in seen:
            continue
        seen.add(n)
        s = files[n]
        sha = getattr(getattr(s, "lfs", None), "sha256", None)
        out.append({
            "name": n,
            "url": f"https://huggingface.co/{prefix}/resolve/main/{n}",
            "bytes": getattr(s, "size", None),
            "checksum": f"sha256:{sha}" if sha else None,
        })
    return out


EXPANDERS = {
    "gcs_public": expand_gcs_public,
    "zenodo": expand_zenodo,
    "figshare": expand_figshare,
    "http": expand_http,
    "hf_dataset": lambda s: expand_hf(s, "dataset"),
    "hf_model": lambda s: expand_hf(s, "model"),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-separated source ids")
    args = ap.parse_args()
    keep = set(args.only.split(",")) if args.only else None

    cfg = json.loads(SOURCES.read_text())
    # --only refreshes a subset; everything else is carried over from the existing
    # lock so a targeted rebuild never silently drops sources.
    previous = {}
    if LOCK.exists():
        previous = {s["id"]: s for s in json.loads(LOCK.read_text())["sources"]}

    entries, total = [], 0
    for src in cfg["sources"]:
        if src["kind"] in {"vcc_cli", "gcs_snapshot"}:
            # Signed download links expire; retain the registered content hashes.
            # Refreshing public sources must not silently replace a challenge panel.
            if src["id"] not in previous:
                raise SystemExit(f"Register the authenticated snapshot first: {src['id']}")
            pinned = previous[src["id"]]
            entries.append(pinned)
            total += sum(f["bytes"] or 0 for f in pinned["files"])
            print(f"[{src['tier']}] {src['id']:16s} kept fixed {src['kind']} snapshot")
            continue
        if keep and src["id"] not in keep:
            if src["id"] in previous:
                kept = previous[src["id"]]
                total += sum(f["bytes"] or 0 for f in kept["files"])
                entries.append(kept)
            continue
        print(f"[{src['tier']}] {src['id']:16s} ", end="", flush=True)
        try:
            files = EXPANDERS[src["kind"]](src)
        except Exception as e:                      # noqa: BLE001 - report and continue
            print(f"FAILED: {type(e).__name__}: {e}")
            continue
        size = sum(f["bytes"] or 0 for f in files)
        total += size
        print(f"{len(files):4d} files  {size / 2**30:8.2f} GB")
        entries.append({
            k: src[k] for k in ("id", "tier", "title", "citation", "license", "why")
        } | ({"root": src["root"]} if "root" in src else {})
          | {"caveat": src.get("caveat"), "kind": src["kind"], "files": files})

    LOCK.write_text(json.dumps({
        "schema_version": cfg["schema_version"],
        "locked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "total_bytes": total,
        "sources": entries,
    }, indent=2) + "\n")
    print(f"\nlocked {len(entries)} sources, {total / 2**30:.1f} GB -> {LOCK}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
