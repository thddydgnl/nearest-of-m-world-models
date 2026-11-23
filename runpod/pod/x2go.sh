#!/bin/bash
# x2.3: wait for the Drive fetch and the venv, then run experiment 2b seed 1 (same arguments as the design's queue)
# with the auto-evaluator. Relaunch after pod x2.2 was deleted by the group cap (2026-11-23T03:23Z).
for i in $(seq 1 120); do
  grep -q "fetch done" /workspace/nom/logs/fetch.log 2>/dev/null && grep -q "env done" /workspace/nom/logs/env.log 2>/dev/null && break
  sleep 30
done
echo "$(date -u +%H:%M:%SZ) fetch and env ready"
mkdir -p /workspace/nom/code/queues /workspace/imle-wm/logs /workspace/imle-wm/runs
Q=/workspace/nom/code/queues/queue_x2b_s1.txt
echo "--data bounce_div_k4 --method imle --m 16 --stages single --wide_head 1024 --memo_every 2000 --z_from 5 --steps 48000 --save_every 12000 --seed 1 --name r1_imle16_48k_s1" > $Q
bash /workspace/nom/code/launch_queue.sh $Q 1
echo "$(date -u +%H:%M:%SZ) launched"
