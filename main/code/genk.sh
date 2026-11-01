#!/bin/bash
# usage: genk.sh <gpu> <K>; builds bounce_div_k<K> (A2-div with K bounce modes) like k4/k8:
# gen_data defaults (1000 train / 250 test episodes, T=40), stats, ball-position probe
source ~/imle-wm/env.sh; cd ~/imle-wm/code; export CUDA_VISIBLE_DEVICES=$1
D=~/imle-wm/data/bounce_div_k$2; L=~/imle-wm/logs/gen_div_k$2.log
[ -f $D/train_tokens.npy ] && [ -f $D/stoch_branch_tokens.npy ] || python -m imlewm.gen_data --name bounce_div_k$2 > $L 2>&1
[ -f $D/stats.pt ] || python -c "from imlewm.gen_data import compute_stats; from imlewm.common import ds_dir; compute_stats(ds_dir('bounce_div_k$2'))" >> $L 2>&1
[ -f $D/probe.pt ] || python -m imlewm.probe --data bounce_div_k$2 >> $L 2>&1
echo "$(date +%T) data bounce_div_k$2 ready" >> ~/imle-wm/logs/queue.log
