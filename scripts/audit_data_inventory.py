#!/usr/bin/env python
"""Read-only full-byte inventory audit. Emit JSON + standalone filterable HTML.

python scripts/audit_data_inventory.py --output data/assessments/inventory/report.json
Only the Python standard library is required. Existing results are never reused.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_URL = "https://github.com/yjcyxky/virtual-cell-challenge"
SOURCE_ROLES = {
    "arc_vcc2025_h1": ("CRISPRi response; all three public splits", [3, 4, 5]),
    "arc_vcc2026_controls": ("Anonymous A/B/C baseline context only", [6]),
    "replogle2022": ("CRISPRi; bulk and single-cell views may overlap", [7]),
    "nadig2025": ("CRISPRi response", [8]),
    "jiang2025": ("Conditional CRISPRi; conditions require verification", [9]),
    "mcfaline_figueroa2024": ("Genetic/drug interaction and drug response", [10, 11, 12]),
    "scperturb": ("Mixed modalities; RNA eligibility requires audit", [13, 14]),
    "scbasecount_2026_01_12_human": ("Selected context; perturbation provenance unresolved", [15, 16]),
    "tahoe100m": ("Selected drug-response shards; not complete Tahoe-100M", [17]),
    "depmap_24q4": ("Auxiliary dependency/expression prior", [18]),
    "networks": ("Auxiliary gene/network prior", [18]),
    "lincs_l1000": ("Auxiliary signatures; not raw single-cell counts", [18]),
    "vcc_gene_axis": ("Historical 2025 gene axis; not current challenge axis", [18]),
    "esm2_650m": ("Model identity only; not an expression dataset", [18]),
    "arc_se600m": ("Model identity only; not an expression dataset", [18]),
}
METHODS = {"inventory-v2": {
    "algorithm": "Read every registered file in 16 MiB chunks; SHA-256 and locked MD5 when available.",
    "comparisons": "Actual size versus lock; hash versus SOURCE, latest MANIFEST, and locked checksum.",
    "concurrency_guard": "Compare device/inode/size/mtime_ns/ctime_ns before/after read; recheck all registry and SOURCE inputs.",
    "limits": "Not a transactional filesystem snapshot. Hash identity does not certify scientific validity. Missing upstream checksum is unavailable, not failure. Write permissions do not imply mutation.",
    "scope": "Exactly registered files. Registration completeness does not establish upstream completeness.",
}}


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def signature(stat) -> tuple:
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def inside(path: Path, parent: Path) -> bool:
    return path.resolve().is_relative_to(parent.resolve())


def audit_file(root: Path, source: dict, entry: dict, provenance: list,
               latest: dict, progress) -> dict:
    relative = Path(source.get("root", "data/raw")) / source["id"] / entry["name"]
    path = root / relative
    result = {"source_id": source["id"], "file": str(relative), "issues": [],
              "url": entry.get("url"), "expected_bytes": entry.get("bytes"),
              "locked_checksum": entry.get("checksum"), "checksum_status": "unavailable"}
    if not inside(path, root / source.get("root", "data/raw") / source["id"]):
        result.update(status="failed", issues=["path_outside_source"])
        return result
    try:
        before = path.stat()
        if not path.is_file():
            raise FileNotFoundError(str(path))
        result.update(bytes=before.st_size, writable=bool(before.st_mode & 0o222))
        if entry.get("bytes") != before.st_size:
            result["issues"].append("locked_size_mismatch")
        matches = [f for f in provenance if f.get("name") == entry["name"]]
        result["source_path_match"] = "relative_path"
        if not matches:
            matches = [f for f in provenance if f.get("name") == Path(entry["name"]).name
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
        digest = hashlib.sha256()
        md5 = hashlib.md5() if algorithm in ("md5", "md5-b64") else None
        with path.open("rb") as fh:
            while chunk := fh.read(16 << 20):
                digest.update(chunk)
                if md5 is not None:
                    md5.update(chunk)
                progress(len(chunk))
            descriptor_after = os.fstat(fh.fileno())
        actual = digest.hexdigest()
        result["sha256"] = actual
        for name, record in [("source", source_file), ("manifest", manifest_file)]:
            if record is not None and record.get("sha256") != actual:
                result["issues"].append(f"{name}_sha256_mismatch")
        observed = {"sha256": actual}
        if md5 is not None:
            observed.update(md5=md5.hexdigest(), **{"md5-b64": base64.b64encode(md5.digest()).decode()})
        upstream = observed[algorithm] == expected if algorithm in observed else None
        result["locked_checksum_verified"] = upstream
        result["checksum_status"] = "verified" if upstream is True else "failed" if upstream is False else "unavailable"
        if spec and upstream is None:
            result["issues"].append("unsupported_locked_checksum")
            result["checksum_status"] = "unsupported"
        if upstream is False:
            result["issues"].append("locked_checksum_mismatch")
        if signature(before) != signature(path.stat()) or signature(before) != signature(descriptor_after):
            result["issues"].append("changed_during_read")
    except FileNotFoundError:
        result["issues"].append("missing_file")
    except OSError as exc:
        result["issues"].append("read_failed")
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["status"] = "failed" if result["issues"] else "completed"
    return result


def git_output(root, *args):
    command = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    return command.stdout.strip() if command.returncode == 0 else None


def coverage_summary(config, source):
    """State observed local counts separately from upstream metadata claims."""
    count = len(source["files"])
    if config.get("shard_glob"):
        shards = sum(bool(re.search(r"train-\d+-of-\d+", f["name"])) for f in source["files"])
        denominator = re.search(r"of-(\d+)", config["shard_glob"])
        total = int(denominator[1]) if denominator else "unknown"
        return f"{shards} expression shards / {total} declared upstream shards; {count - shards} metadata files"
    if config.get("selection"):
        n = sum(f["name"].endswith(".h5ad") for f in source["files"])
        total = config["selection"].get("full_human_files", "unknown")
        return f"{n} selected H5AD / {total} human files in selection metadata; {count - n} metadata files"
    return f"{count} registered files; selection follows registry filters/list; upstream completeness not certified"


def run_audit(root: Path, workers=2, only=None) -> dict:
    started, t0 = datetime.now(timezone.utc).isoformat(), time.monotonic()
    inputs = {p: (root / p).read_bytes() for p in (
        "docs/README.md", "data/sources.json", "data/registry.lock.json", "data/MANIFEST.tsv")}
    registered = json.loads(inputs["data/registry.lock.json"])["sources"]
    if only:
        unknown = set(only) - {s["id"] for s in registered}
        if unknown:
            raise ValueError(f"Unknown source ids: {sorted(unknown)}")
        registered = [s for s in registered if s["id"] in only]
    latest = {r["file"]: r for r in csv.DictReader(
        inputs["data/MANIFEST.tsv"].decode().splitlines(), delimiter="\t")}
    configs = {s["id"]: s for s in json.loads(inputs["data/sources.json"])["sources"]}
    jobs, sources = [], []
    for source in registered:
        directory = Path(source.get("root", "data/raw")) / source["id"]
        source_path = directory / "SOURCE.json"
        errors, provenance = [], {}
        try:
            inputs[str(source_path)] = (root / source_path).read_bytes()
            provenance = json.loads(inputs[str(source_path)])
        except (OSError, ValueError) as exc:
            errors.append(f"SOURCE unreadable: {type(exc).__name__}: {exc}")
        role, tickets = SOURCE_ROLES.get(source["id"], ("Unassessed", []))
        config = configs.get(source["id"], {})
        if config.get("selection_manifest"):
            selection_path = config["selection_manifest"]
            try:
                inputs[selection_path] = (root / selection_path).read_bytes()
                if sha256(inputs[selection_path]) != config.get("selection_manifest_sha256"):
                    errors.append("selection_manifest_sha256_mismatch")
            except OSError as exc:
                errors.append(f"selection manifest unreadable: {exc}")
        sources.append({
            "id": source["id"], "directory": str(directory), "provenance_errors": errors,
            "locked_files": len(source["files"]),
            "locked_bytes": sum(e.get("bytes") or 0 for e in source["files"]),
            "candidate_role": {"value": role, "evidence_kind": "assessment_scope_decision"},
            "coverage_summary": coverage_summary(config, source),
            "upstream_coverage": {"status": "not_established_by_inventory", "selection_evidence": config,
                                  "local_scope": "Exactly the registered files; no inferred upstream completeness"},
            "historical_claims": {k: source.get(k) for k in ("title", "citation", "license", "why", "caveat")},
            "specialist_assessment": {"status": "pending", "issues": [f"{REPO_URL}/issues/{n}" for n in tickets]},
        })
        jobs.extend((source, entry, provenance.get("files", [])) for entry in source["files"])
    jobs.sort(key=lambda item: item[1].get("bytes") or 0, reverse=True)
    total = sum(s["locked_bytes"] for s in sources)
    bytes_read, progress_lock = 0, threading.Lock()

    def progress(n):
        nonlocal bytes_read
        with progress_lock:
            bytes_read += n

    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = {executor.submit(audit_file, root, s, e, p, latest, progress) for s, e, p in jobs}
        last_print = t0
        while pending:
            completed, pending = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
            results.extend(f.result() for f in completed)
            if time.monotonic() - last_print >= 15 or not pending:
                print(f"{len(results)}/{len(jobs)} files; {bytes_read / 1e9:.1f}/{total / 1e9:.1f} GB read", flush=True)
                last_print = time.monotonic()
    for source in sources:
        files = [r for r in results if r["source_id"] == source["id"]]
        source.update(verified_files=sum(not r["issues"] for r in files),
                      issue_files=sum(bool(r["issues"]) for r in files),
                      checksum_unavailable=sum(r["checksum_status"] == "unavailable" for r in files),
                      writable_files=sum(r.get("writable", False) for r in files),
                      basename_only_provenance=sum(r.get("source_path_match") == "basename_and_url" for r in files))
        source["status"] = "failed" if source["issue_files"] or source["provenance_errors"] else "completed"
    changed = []
    for path, content in inputs.items():
        try:
            if (root / path).read_bytes() != content:
                changed.append(path)
        except OSError:
            changed.append(path)
    return {
        "schema_version": 2, "bundle_id": "inventory-" + uuid.uuid4().hex,
        "status": "failed" if changed or any(s["status"] == "failed" for s in sources) else "completed",
        "started_at": started, "completed_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": round(time.monotonic() - t0, 2), "workers": workers,
        "runtime": {"python": sys.version, "platform": platform.platform(), "dependencies": "Python standard library"},
        "code": {"commit": git_output(root, "rev-parse", "HEAD"),
                 "tracked_changes": git_output(root, "status", "--porcelain", "--untracked-files=no"),
                 "script_sha256": sha256(Path(__file__).read_bytes())},
        "input_sha256": {p: sha256(content) for p, content in inputs.items()},
        "registry_inputs_unchanged": not changed, "changed_inputs": changed,
        "exposure": {"content_integrity": "All selected bytes read", "response_statistics": "not_computed"},
        "status_definitions": {"completed": "Requested calculation finished", "pending": "Not attempted",
            "not_applicable": "Outside method domain", "not_estimable": "Attempted but scientific support insufficient",
            "blocked": "Required external input unavailable", "failed": "Integrity or execution failure"},
        "methods": METHODS, "files": len(results), "bytes_read": bytes_read,
        "issue_files": sum(bool(r["issues"]) for r in results), "sources": sources,
        "file_results": sorted(results, key=lambda r: r["file"]),
    }


def render(report: dict) -> str:
    # Untrusted filenames must not terminate the JSON script element.
    data = json.dumps(report, ensure_ascii=False).replace("<", "\\u003c")
    return '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>Dataset inventory assessment</title>
<style>body{font:16px system-ui;margin:2rem;max-width:1500px}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ccc;padding:.6rem;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere}input{padding:.6rem;width:50%}.failed{color:#b00020}details{margin:1rem 0}</style>
<h1>只读数据完整性评估</h1><p id="summary"></p>
<p>完整性通过不表示科学适用性通过。无上游 checksum ≠ 校验失败。登记集合 ≠ 上游全量。后续状态以关联 GitHub Issues 为准。</p>
<input id="query" aria-label="筛选来源或文件" placeholder="筛选来源、文件、异常"><label><input type="checkbox" id="errors" style="width:auto">仅异常文件</label>
<h2>来源与候选用途</h2><table><thead><tr><th>来源</th><th>完整性</th><th>范围</th><th>候选用途／专项评估</th></tr></thead><tbody id="sources"></tbody></table>
<h2>文件证据</h2><p id="count"></p><table><thead><tr><th>文件</th><th>字节</th><th>checksum</th><th>异常 / SHA-256</th></tr></thead><tbody id="files"></tbody></table>
<details><summary>固定方法、状态、输入与代码版本</summary><pre id="methods"></pre></details>
<details><summary>来源选择依据与历史声明（未经本检查认证）</summary><pre id="claims"></pre></details>
<script id="data" type="application/json">''' + data + '''</script><script>
const r=JSON.parse(document.getElementById('data').textContent);
const esc=x=>String(x??'unknown').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
document.getElementById('summary').textContent=`${r.bundle_id} | ${r.status} | ${r.files} files | ${(r.bytes_read/1e9).toFixed(3)} GB | ${r.issue_files} issue files | ${r.completed_at}`;
document.getElementById('methods').textContent=JSON.stringify({...r,sources:undefined,file_results:undefined},null,2);
document.getElementById('claims').textContent=JSON.stringify(r.sources.map(s=>({id:s.id,coverage:s.upstream_coverage,historical_claims:s.historical_claims})),null,2);
function draw(){const q=document.getElementById('query').value.toLowerCase(),bad=document.getElementById('errors').checked;
document.getElementById('sources').innerHTML=r.sources.filter(s=>JSON.stringify(s).toLowerCase().includes(q)).map(s=>`<tr><td>${esc(s.id)}</td><td class="${s.status}">${esc(s.status)}: ${s.verified_files}/${s.locked_files}<br>checksum unavailable: ${s.checksum_unavailable}</td><td>${esc(s.coverage_summary)}<br>${(s.locked_bytes/1e9).toFixed(3)} GB</td><td>${esc(s.candidate_role.value)}<br>${s.specialist_assessment.issues.map(u=>`<a href="${esc(u)}">#${u.split('/').pop()}</a>`).join(' ')} (pending at audit)</td></tr>`).join('');
const fs=r.file_results.filter(f=>(!bad||f.issues.length)&&JSON.stringify(f).toLowerCase().includes(q));
document.getElementById('count').textContent=`${fs.length} matching files; displaying first 250. Full evidence in JSON.`;
document.getElementById('files').innerHTML=fs.slice(0,250).map(f=>`<tr><td>${esc(f.file)}</td><td>${esc(f.bytes)}</td><td>${esc(f.checksum_status)}</td><td class="${f.issues.length?'failed':''}">${esc(f.issues.join(', '))}<br><small>${esc(f.sha256)}</small></td></tr>`).join('');}
document.getElementById('query').oninput=draw;document.getElementById('errors').onchange=draw;draw();</script></html>'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--only", help="comma-separated source ids; default all")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    root, output = args.root.resolve(), args.output.absolute()
    page = output.with_suffix(".html")
    if output.suffix != ".json":
        parser.error("output must end in .json")
    registered = json.loads((root / "data/registry.lock.json").read_text())["sources"]
    protected = [root / s.get("root", "data/raw") / s["id"] for s in registered]
    if any(inside(output, p) or inside(page, p) for p in protected):
        parser.error("outputs must be outside source directories")
    if output.exists() or page.exists() or output.is_symlink() or page.is_symlink():
        parser.error("output already exists; use a fresh result location")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as fh, page.open("x") as web:
        try:
            report = run_audit(root, args.workers, args.only.split(",") if args.only else None)
            report["artifacts"] = {"json": output.name, "html": page.name}
            report["reproduce"] = {"argv": sys.argv, "working_directory": str(Path.cwd())}
            json.dump(report, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
            web.write(render(report))
        except Exception as exc:
            json.dump({"status": "failed", "error": f"{type(exc).__name__}: {exc}"}, fh)
            raise
    print(f"Report: {output}; status: {report['status']}; issue files: {report['issue_files']}")
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
