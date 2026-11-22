#!/bin/bash
# Experiment 1 recovery on x1.1: compute_stats of the arc dataset on the CPU (the GPU is shared with training and
# ran out of memory, as on the original server), then the probe, then <n> arc workers.
source /workspace/imle-wm/env.sh
cd /workspace/imle-wm/code
D=/workspace/imle-wm/data/bounce_div_k4_arc90
echo "$(date -u +%H:%M:%SZ) stats on CPU"
python -c "
import torch
torch.Tensor.cuda = lambda self, *a, **k: self
from imlewm.gen_data import compute_stats
compute_stats('$D')
" > /workspace/nom/logs/stats_arc.log 2>&1 || { echo "stats failed"; exit 1; }
python -c "
import json, sys; sys.argv = ['gen_data', '--name', 'bounce_div_k4_arc90']
json.dump({'name': 'bounce_div_k4_arc90', 'train_eps': 1000, 'test_eps': 250, 'T': 40, 'seed': 0,
           'note': 'stats computed on the CPU after a GPU OOM (revision r1)'}, open('$D/gen_args.json', 'w'), indent=1)
"
echo "$(date -u +%H:%M:%SZ) probe"
python -m imlewm.probe --data bounce_div_k4_arc90 > /workspace/nom/logs/probe_arc.log 2>&1 || { echo "probe failed"; exit 1; }
echo "$(date -u +%H:%M:%SZ) arc data ready; starting arc workers"
bash /workspace/nom/code/launch_queue.sh /workspace/nom/code/queues/queue_x1b.txt ${1:-1}
