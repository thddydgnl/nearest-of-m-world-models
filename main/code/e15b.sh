#!/bin/bash
# usage: e15b.sh <gpu>; E15 follow-ups (2026-10-31, results/E15_report.md): evaluates the qE15b.txt runs as
# their checkpoints appear, eval.json and e3.json as in e15chain.sh
#  A2-fix (bounce_k4, fixed appearance): ES m=8 with IMLE m=16 / m=4 references in the same recipe
#  A2-div K=4: per-token ES (--es_dist token), 2 seeds
source ~/imle-wm/env.sh; cd ~/imle-wm/code
export CUDA_VISIBLE_DEVICES=$1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=~/imle-wm/runs; L=~/imle-wm/logs/e15b.log
wait_ck() {  # at most 10h per run
  local n=0; until [ -f $R/$1/ckpt.pt ] || [ $n -ge 300 ]; do sleep 120; n=$((n+1)); done
  [ -f $R/$1/ckpt.pt ]
}
a2() {
  wait_ck $1 || return
  [ -f $R/$1/eval.json ] || python -m imlewm.evaluate --run $1 >> $L 2>&1
  [ -f $R/$1/e3.json ] || python -m imlewm.e3 --run $1 >> $L 2>&1
}
a2 e15_fix_es8_s0; a2 e15_a2_estok8_s0; a2 e15_fix_imle16_s0; a2 e15_a2_estok8_s1; a2 e15_fix_imle4_s0
echo "$(date +%T) e15b done" >> ~/imle-wm/logs/queue.log
