#!/bin/bash
# Health check for the IMLE-WM revision pods (prefix nom): spend, watchdog, and per live pod the one-line
# queue status printed by /workspace/nom/code/status.sh, GPU use and disk.
# Usage: check_all.sh            (read-only; never changes a pod)
N="/Users/song-yonghwi/Documents/IMLE-WM/runpod"
al() { perl -e 'alarm shift; exec @ARGV' "$@"; }   # macOS has no `timeout`
echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) RunPod (nom)"
al 90 python3 "$N/budget.py" status 2>&1 | awk 'NR == 1 || $5 != "gone"'
al 90 python3 - "$N" << 'PY'
import json, sys, subprocess
sys.path.insert(0, sys.argv[1])
try:
    import budget
    _, alive, pods, total, rows = budget.snapshot()
    caps = json.load(open(f"{sys.argv[1]}/caps.json"))
    by = {}
    for tag, pid, gpu, rate, state, cost in rows:
        by[budget.pilot(tag)] = by.get(budget.pilot(tag), 0.0) + cost
    print("per group:", ", ".join(f"{k} ${v:.2f}/{caps.get(k, '-')}" for k, v in sorted(by.items())), f"| total ${total:.2f} / ${budget.LIMIT:.0f}")
except Exception as e:
    print("per-group summary failed:", e)
PY
for env in "$N"/pods/*.env; do
  [ -f "$env" ] || continue
  tag=$(basename "$env" .env)
  out=$(al 60 bash "$N/rsh.sh" "$tag" 'bash /workspace/nom/code/status.sh 2>&1 | tail -n 1; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | head -1; df -h /workspace 2>/dev/null | tail -1 | awk "{print \"disk \" \$5 \" used, \" \$4 \" free\"}"' 2>&1)
  echo "--- $tag | $(echo "$out" | tr '\n' '|' | cut -c1-400)"
done
pgrep -f "IMLE-WM/runpod/budget.py watch" > /dev/null && echo "watchdog: alive" || echo "watchdog: NOT RUNNING"
