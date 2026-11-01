#!/bin/bash
# usage: planchain.sh; E13 hazard planning on the push env ("edge" task: the straight path runs over
# the ice along the pit; an upward ice deflection drops the ball into the absorbing pit).
# CEM-MPC, cost = mean probe distance to goal over m_p=8 imagined rollouts (lambda=0), 100 episodes,
# the same task seed for every model; oracle = true simulator. Waits for each checkpoint; each job runs
# on the GPU with the most free memory at that moment and is retried (up to 4 times) if it fails (OOM).
source ~/imle-wm/env.sh; cd ~/imle-wm/code; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=~/imle-wm/runs; L=~/imle-wm/logs/e13_plan.log
done_() { [ -f $R/plan/$1_edge_pos_std_s0.json ] && grep -q '"n_eps": 100' $R/plan/$1_edge_pos_std_s0.json; }
pl() {
  for try in 1 2 3 4; do
    done_ $1 && return
    G=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1)
    echo "$(date +%T) plan $1 try $try gpu $G" >> $L
    CUDA_VISIBLE_DEVICES=$G python -m imlewm.plan --model $1 --lams 0 --eps 100 --seed 0 --task edge --cost pos >> $L 2>&1
    done_ $1 || sleep 300
  done
}
pl oracle
for s in 0 1 2; do for m in imle16 fm reg; do
  r=e13_push_${m}_s$s; n=0; until [ -f $R/$r/ckpt.pt ] || [ $n -ge 200 ]; do sleep 120; n=$((n+1)); done
  pl $r
done; done
echo "$(date +%T) planchain done" >> ~/imle-wm/logs/queue.log
