#!/bin/bash
# x1.1 follow-up: (1) when the control-queue workers have exited, add 2 arc workers (3 in total on the A4500);
# (2) when every training worker has exited, evaluate the finished r1 runs one at a time (the A4500's 20 GB could not
# hold an evaluation next to three training runs), then mark x1 done.
source /workspace/imle-wm/env.sh
cd /workspace/imle-wm/code
for i in $(seq 1 240); do ps -eo args | grep -q "[r]unq.sh 0 /workspace/nom/code/queues/queue_x1a.txt" || break; sleep 60; done
echo "$(date -u +%H:%M:%SZ) controls done; adding ${1:-1} arc worker(s)"
bash /workspace/nom/code/launch_queue.sh /workspace/nom/code/queues/queue_x1b.txt ${1:-1} nowatch
for i in $(seq 1 480); do ps -eo args | grep -q "[r]unq.sh" || break; sleep 60; done
echo "$(date -u +%H:%M:%SZ) training done; evaluating"
for d in /workspace/imle-wm/runs/r1_*/; do
  r=$(basename $d); [ -f $d/ckpt.pt ] && [ ! -f $d/eval.json ] || continue
  python -m imlewm.evaluate --run $r > /workspace/imle-wm/logs/eval_$r.log 2>&1 && echo "$(date -u +%H:%M:%SZ) evaluated $r"
done
touch /workspace/nom/out/x1_done; echo "$(date -u +%H:%M:%SZ) x1 done"
