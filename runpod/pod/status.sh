#!/bin/bash
# One status line: queued jobs left in the queues this pod uses (those with a .lock file), r1 runs started/finished,
# failures, and live workers/evaluators/chains.
left=0; for q in /workspace/nom/code/queues/queue_*.txt; do [ -f "$q.lock" ] && left=$((left + $(grep -c -- '--name' "$q"))); done
started=$(ls -d /workspace/imle-wm/runs/r1_*/ 2>/dev/null | wc -l)
finished=0; for d in /workspace/imle-wm/runs/r1_*/; do [ -f "$d/ckpt.pt" ] && finished=$((finished+1)); done
fails=$(grep -c "rc=[1-9]" /workspace/imle-wm/logs/queue.log 2>/dev/null); fails=${fails:-0}
w=$(ps -eo args | grep -c "[r]unq.sh"); e=$(ps -eo args | grep -c "[e]valwatch.sh"); c=$(ps -eo args | grep -cE "[x]1chain|[e]3chain")
echo "tag=$(cat /workspace/nom/tag 2>/dev/null) queued=$left started=$started finished=$finished failed=$fails workers=$w evalwatch=$e chains=$c $(date -u +%H:%MZ)"
