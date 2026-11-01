#!/bin/bash
source ~/imle-wm/env.sh; cd $IMLE/code
CUDA_VISIBLE_DEVICES=0 python -c "from imlewm.gen_data import compute_stats; from imlewm.common import ds_dir; compute_stats(ds_dir(\"bounce_div_k4\"))" > $IMLE/logs/stats_div.log 2>&1
CUDA_VISIBLE_DEVICES=0 python -m imlewm.probe --data bounce_div_k4 > $IMLE/logs/probe_div.log 2>&1
./runq.sh 0 q0.txt & sleep 5; ./runq.sh 0 q0.txt & wait
