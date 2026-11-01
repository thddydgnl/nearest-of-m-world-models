#!/bin/bash
# usage: e16chain.sh <gpu>; E16 (2026-10-31, results/E16_report.md): evaluates the qE16_g0/g1.txt runs as
# their checkpoints appear, each step skipped if its output exists
#  E15 seeds (A2-div ES m=16, A2-fix ES m=8 / IMLE m=16 / IMLE m=4): eval.json and e3.json as in e15b.sh
#  N1b rollout-state ablation on A1 (--seq_feed random / gt): a1eval eval.json and e3 h=16 as in e11chain.sh
source ~/imle-wm/env.sh; cd ~/imle-wm/code
export CUDA_VISIBLE_DEVICES=$1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=~/imle-wm/runs; L=~/imle-wm/logs/e16.log
wait_ck() {  # at most 10h per run
  local n=0; until [ -f $R/$1/ckpt.pt ] || [ $n -ge 300 ]; do sleep 120; n=$((n+1)); done
  [ -f $R/$1/ckpt.pt ]
}
a2() {
  wait_ck $1 || return
  [ -f $R/$1/eval.json ] || python -m imlewm.evaluate --run $1 >> $L 2>&1
  [ -f $R/$1/e3.json ] || python -m imlewm.e3 --run $1 >> $L 2>&1
}
a1() {
  wait_ck $1 || return
  [ -f $R/$1/eval.json ] || python -m imlewm.a1eval --run $1 >> $L 2>&1
  [ -f $R/$1/e3.json ] || python -m imlewm.e3 --run $1 --hmax 16 --cs 8 >> $L 2>&1
}
a2 e15_fix_es8_s1; a2 e15_fix_imle16_s1; a2 e15_a2_es16_s1; a2 e15_fix_imle4_s1
a1 e16_a1_seqrand_s0; a1 e16_a1_seqrand_s1; a1 e16_a1_seqgt_s0; a1 e16_a1_seqgt_s1
echo "$(date +%T) e16 done" >> ~/imle-wm/logs/queue.log
