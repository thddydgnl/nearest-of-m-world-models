#!/bin/bash
# Start <n> queue workers (runq.sh, all on GPU 0) on one queue file and, once per pod, the auto-evaluator for r1 runs.
# Usage (on the pod): bash launch_queue.sh /workspace/nom/code/queues/queue_x2a.txt 3
Q=$1; N=${2:-1}
for i in $(seq 1 $N); do
  setsid nohup bash /workspace/imle-wm/code/runq.sh 0 "$Q" > /workspace/nom/logs/worker_$(basename $Q .txt)_$i.log 2>&1 < /dev/null &
  sleep 20   # stagger the RAM-heavy token loads
done
if [ "${3:-}" != nowatch ] && ! ps -eo args | grep -q "[e]valwatch.sh"; then
  setsid nohup bash /workspace/imle-wm/code/evalwatch.sh 0 '^r1_' > /workspace/nom/logs/evalwatch.log 2>&1 < /dev/null &
fi
echo "launched $N workers on $Q"
