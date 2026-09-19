#!/usr/bin/env bash
# 语料下载进度一览
cd "$(dirname "$0")/.." || exit 1
printf "%-34s %10s %10s  %s\n" "SOURCE" "ON DISK" "LOCKED" "PROGRESS"
.venv/bin/python - <<'PY'
import json, pathlib, sys
sys.path.insert(0, "scripts")
from fetch_data import dest_dir          # one resolver, shared with the fetcher
root = pathlib.Path(".")
lock = json.loads((root/"data/registry.lock.json").read_text())
tot_h = tot_w = 0
for s in lock["sources"]:
    d = dest_dir(s)
    want = sum(f["bytes"] or 0 for f in s["files"])
    have = sum((d/f["name"]).stat().st_size for f in s["files"] if (d/f["name"]).exists())
    tot_h += have; tot_w += want
    pct = 100*have/want if want else 0
    bar = "#" * int(pct/5) + "." * (20-int(pct/5))
    print(f"[{s['tier']}] {s['id']:29s} {have/2**30:7.2f}G {want/2**30:7.2f}G  {bar} {pct:5.1f}%")
print(f"\n{'TOTAL':34s} {tot_h/2**30:7.2f}G {tot_w/2**30:7.2f}G  {100*tot_h/tot_w:5.1f}%")
PY
echo; pids=$(pgrep -f "^[^ ]*python[^ ]* scripts/fetch_data\.py" | tr "\n" " ")
[ -n "$pids" ] && echo "下载进行中 (pid $pids)" || echo "无下载进程在跑"
df -h . | tail -1 | awk '{print "磁盘剩余: "$4"  已用 "$5}'
