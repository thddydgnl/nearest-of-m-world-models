#!/bin/bash
# One status line for check_all.sh: queued, running and finished r1 runs on this pod, plus the queue workers.
Q=$(ls /workspace/nom/code/queue_*.txt 2>/dev/null | head -1)
left=$( [ -n "$Q" ] && grep -c -- '--name' "$Q" || echo 0)
done_=$(ls -d /workspace/imle-wm/runs/r1_*/ 2>/dev/null | while read d; do [ -f "$d/ckpt.pt" ] && echo x; done | wc -l)
started=$(ls -d /workspace/imle-wm/runs/r1_*/ 2>/dev/null | wc -l)
workers=$(pgrep -fc "runq.sh" || true); evals=$(pgrep -fc "evalwatch.sh" || true)
fails=$(grep -c "rc=[1-9]" /workspace/imle-wm/logs/queue.log 2>/dev/null || echo 0)
echo "tag=$(cat /workspace/nom/tag 2>/dev/null) queued=$left started=$started finished=$done_ failed=$fails workers=$workers evalwatch=$evals updated=$(date -u +%H:%MZ)"
