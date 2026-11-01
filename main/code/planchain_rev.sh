#!/bin/bash
# usage: planchain_rev.sh; second E13 planning worker running in parallel with planchain.sh (pc2).
# Walks the models in the reverse of pc2's order and skips any model that is finished or currently
# being planned by another worker, so the two workers do not duplicate work.
source ~/imle-wm/env.sh; cd ~/imle-wm/code; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=~/imle-wm/runs; L=~/imle-wm/logs/e13_plan.log
done_() { [ -f $R/plan/$1_edge_pos_std_s0.json ] && grep -q '"n_eps": 100' $R/plan/$1_edge_pos_std_s0.json; }
running_() { ps aux | grep -q "[-]-model $1 "; }
pl() {
  for try in 1 2 3 4; do
    done_ $1 && return
    running_ $1 && return
    G=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1)
    echo "$(date +%T) plan(rev) $1 try $try gpu $G" >> $L
    CUDA_VISIBLE_DEVICES=$G python -m imlewm.plan --model $1 --lams 0 --eps 100 --seed 0 --task edge --cost pos >> $L 2>&1
    done_ $1 || sleep 300
  done
}
for r in fm_s2 imle16_s2 reg_s2 reg_s1 fm_s1 imle16_s1 reg_s0 fm_s0 imle16_s0; do pl e13_push_$r; done
echo "$(date +%T) planchain_rev done" >> ~/imle-wm/logs/queue.log
