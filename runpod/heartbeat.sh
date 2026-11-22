#!/bin/bash
# 30-minute check (run under Monitor): wait, then print one block: spend, per-pod status, errors in r1 logs,
# memory/GPU use, pull of small results, and whether the watchdog and caffeinate are alive.
cd /Users/song-yonghwi/Documents/IMLE-WM
al() { perl -e 'alarm shift; exec @ARGV' "$@"; }
sleep ${1:-1680}
echo "=== heartbeat $(date -u +%H:%MZ)"
al 120 python3 runpod/budget.py status 2>&1 | awk 'NR==1 || /RUNNING|EXITED|gone/' | tail -8
for env in runpod/pods/*.env; do
  t=$(basename $env .env)
  out=$(al 60 bash runpod/rsh.sh $t 'bash /workspace/nom/code/status.sh 2>&1 | tail -1; echo "errs: $(grep -lE "Traceback|out of memory|Killed|RuntimeError" /workspace/imle-wm/logs/r1_*.log /workspace/imle-wm/logs/eval_r1_*.log /workspace/nom/logs/*.log 2>/dev/null | xargs -n1 basename 2>/dev/null | head -6 | tr "\n" " ")"; mc=$(cat /sys/fs/cgroup/memory.current 2>/dev/null || cat /sys/fs/cgroup/memory/memory.usage_in_bytes 2>/dev/null || echo 0); echo "mem $((mc/1073741824))G gpu $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"' 2>&1 | tr '\n' '|' | cut -c1-420)
  echo "--- $t | $out"
done
al 900 bash runpod/pull_all.sh 2>&1 | tail -6
pgrep -f "IMLE-WM/runpod/budget.py watch" >/dev/null && echo "watchdog alive" || echo "WATCHDOG DOWN"
pgrep -f "caffeinate -i -s" >/dev/null && echo "caffeinate alive" || echo "CAFFEINATE DOWN"
