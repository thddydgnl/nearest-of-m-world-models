#!/bin/bash
source ~/imle-wm/env.sh; cd $IMLE/code
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1 python -m imlewm.train --data bounce_k4 --method mf --mf_p 1 --lr 1e-4 --mf_warmup 1000 --mf_ratio 0.1 --name mf_k4 > $IMLE/logs/mf_k4.log 2>&1
echo "$(date +%T) gpu1 end mf_k4" >> $IMLE/logs/queue.log
