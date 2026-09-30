#!/usr/bin/env python
"""Finish an Orion/CD4 acquisition: wait, verify, freeze this source, audit and index.

This worker does not declare partial downloads ready. A failed stage leaves a JSON
failure and exits. Re-run with the same output to resume completed audit/index work.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
CODE = ["scripts/finish_crispri_acquisition.py", "scripts/audit_data_inventory.py",
        "scripts/dossier/ingest_crispri.py", "scripts/dossier/challenge.py", "scripts/dossier/rna.py"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def process_identity(pid):
    try:
        # starttime prevents treating a reused PID as the original downloader.
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except FileNotFoundError:
        return None


def acquisition_progress(root, source):
    sizes, full = 0, 0
    for entry in source["files"]:
        path = root / source.get("root", "data/raw") / source["id"] / entry["name"]
        actual = path.stat().st_size if path.exists() else 0
        sizes += min(actual, entry["bytes"])
        full += actual == entry["bytes"]
    return {"bytes_present": sizes, "expected_bytes": sum(e["bytes"] for e in source["files"]),
            "files_at_expected_size": full, "expected_files": len(source["files"]),
            "size_is_not_content_verification": True}


def complete_provenance(root, source):
    base = root / source.get("root", "data/raw") / source["id"]
    provenance = json.loads((base / "SOURCE.json").read_text())
    records = {e["name"]: e for e in provenance["files"]}
    if len(records) != len(source["files"]) or len(records) != len(provenance["files"]):
        raise ValueError("incomplete_or_duplicate_source_provenance")
    for entry in source["files"]:
        row = records.get(entry["name"], {})
        if (row.get("bytes") != entry["bytes"] or row.get("url") != entry["url"]
                or len(row.get("sha256", "")) != 64
                or (base / entry["name"]).stat().st_size != entry["bytes"]):
            raise ValueError(f"incomplete_source_file:{entry['name']}")
    return base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-id", choices=["xaira_orion", "zhu2026_cd4"], required=True)
    parser.add_argument("--wait-pid", type=int, help="Existing fetch_data.py process; no second downloader is started")
    parser.add_argument("--analysis-python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    protected = (ROOT / "data/raw" / args.source_id).resolve()
    if output.is_relative_to(protected):
        parser.error("output must be outside the raw source")
    output.mkdir(parents=True, exist_ok=True)
    # One finishing worker per source, including invocations with different outputs.
    lock_path = ROOT / "data/logs" / f"finish-{args.source_id}.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source = next(s for s in json.loads((ROOT / "data/registry.lock.json").read_text())["sources"] if s["id"] == args.source_id)
        binding = {"source": source, "code_sha256": {p: sha(ROOT / p) for p in CODE},
                   "analysis_python": str(args.analysis_python.absolute())}
        state_path = output / "acquisition.json"
        previous = json.loads(state_path.read_text()) if state_path.exists() else {}
        if previous and previous["binding"] != binding:
            raise ValueError("worker_binding_changed_use_new_output")
        state = {**previous, "binding": binding, "pid": os.getpid(), "source_id": args.source_id,
                 "started_at": previous.get("started_at", datetime.now(timezone.utc).isoformat())}

        def update(stage, **extra):
            state.update(stage=stage, updated_at=datetime.now(timezone.utc).isoformat(), **extra)
            temporary = state_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
            temporary.replace(state_path)
            print(f"{state['updated_at']} {args.source_id}: {stage}", flush=True)

        def guard():
            if any(sha(ROOT / p) != h for p, h in binding["code_sha256"].items()):
                raise ValueError("finishing_code_changed_during_acquisition")
            current = next(s for s in json.loads((ROOT / "data/registry.lock.json").read_text())["sources"] if s["id"] == args.source_id)
            if current != source:
                raise ValueError("source_registration_changed_during_acquisition")

        try:
            if args.wait_pid and (identity := process_identity(args.wait_pid)):
                command = Path(f"/proc/{args.wait_pid}/cmdline").read_bytes().split(b"\0")
                if b"scripts/fetch_data.py" not in command or args.source_id.encode() not in command:
                    raise ValueError("wait_pid_is_not_requested_downloader")
                while process_identity(args.wait_pid) == identity:
                    update("waiting_for_download", progress=acquisition_progress(ROOT, source))
                    time.sleep(30)
            guard()
            base = complete_provenance(ROOT, source)
            update("freezing_completed_source", progress=acquisition_progress(ROOT, source))
            # Freeze exactly the registered source; never the global raw tree.
            for path in [base / e["name"] for e in source["files"]] + [base / "SOURCE.json"]:
                path.chmod(path.stat().st_mode & ~0o222)
            inventory = Path(state["inventory"]) if state.get("inventory") else None
            if inventory is None or not inventory.exists() or json.loads(inventory.read_text()).get("status") != "completed":
                inventory = output / f"inventory-{uuid.uuid4().hex[:12]}.json"
                update("auditing_all_bytes", inventory=str(inventory))
                subprocess.run([sys.executable, str(ROOT / "scripts/audit_data_inventory.py"), "--only", args.source_id,
                                "--workers", "2", "--output", str(inventory)], cwd=ROOT, check=True)
            if json.loads(inventory.read_text())["status"] != "completed":
                raise ValueError("source_inventory_failed")
            guard()
            update("indexing_all_cells_and_counts")
            subprocess.run([str(args.analysis_python.absolute()), str(ROOT / "scripts/dossier/ingest_crispri.py"),
                            "--source-id", args.source_id, "--inventory", str(inventory),
                            "--output", str(output / "index")], cwd=ROOT, check=True)
            guard()
            update("completed", report=str(output / "index/report.json"), error=None)
        except Exception as error:
            update("failed", failed_stage=state.get("stage"), error=f"{type(error).__name__}: {error}")
            raise


if __name__ == "__main__":
    main()
