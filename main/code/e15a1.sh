#!/bin/bash
# usage: e15a1.sh <gpu>; energy score on A1 (contexts repeat), the control for the A2-div result (2026-10-31):
# evaluates e15_a1_es8_s{0,1} as their checkpoints appear, with the same readouts as the E1 / E12 A1 tables
#  eval.json (probe), eval_hm.json (heatmap), eval_hmn05.json (noise-0.5 heatmap), e3.json (rollout manifold)
source ~/imle-wm/env.sh; cd ~/imle-wm/code
export CUDA_VISIBLE_DEVICES=$1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=~/imle-wm/runs; L=~/imle-wm/logs/e15.log
wait_ck() {  # at most 10h per run
  local n=0; until [ -f $R/$1/ckpt.pt ] || [ $n -ge 300 ]; do sleep 120; n=$((n+1)); done
  [ -f $R/$1/ckpt.pt ]
}
a1() {
  wait_ck $1 || return
  [ -f $R/$1/eval.json ] || python -m imlewm.a1eval --run $1 >> $L 2>&1
  [ -f $R/$1/eval_hm.json ] || A1_READOUT=heatmap A1_DATA=bshapes_v3 python -m imlewm.a1eval --run $1 >> $L 2>&1
  [ -f $R/$1/eval_hmn05.json ] || A1_READOUT=heatmap A1_READOUT_TAG=n05 A1_DATA=bshapes_v3 python -m imlewm.a1eval --run $1 >> $L 2>&1
  [ -f $R/$1/e3.json ] || python -m imlewm.e3 --run $1 >> $L 2>&1
}
a1 e15_a1_es8_s0; a1 e15_a1_es8_s1
echo "$(date +%T) e15a1 done" >> ~/imle-wm/logs/queue.log
