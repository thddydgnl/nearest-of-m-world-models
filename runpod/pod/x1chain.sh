#!/bin/bash
# Experiment 1 on x1.1: generate the 90-degree arc dataset and its probe, then start the arc-run workers.
source /workspace/imle-wm/env.sh
cd /workspace/imle-wm/code
echo "$(date -u +%H:%M:%SZ) gen arc data"
python -m imlewm.gen_data --name bounce_div_k4_arc90 > /workspace/nom/logs/gen_arc.log 2>&1 || { echo "gen_data failed"; exit 1; }
echo "$(date -u +%H:%M:%SZ) probe"
python -m imlewm.probe --data bounce_div_k4_arc90 > /workspace/nom/logs/probe_arc.log 2>&1 || { echo "probe failed"; exit 1; }
echo "$(date -u +%H:%M:%SZ) arc data ready; starting arc workers"
bash /workspace/nom/code/launch_queue.sh /workspace/nom/code/queues/queue_x1b.txt ${1:-3}
