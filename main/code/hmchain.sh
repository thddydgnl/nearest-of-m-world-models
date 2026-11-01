#!/bin/bash
# usage: hmchain.sh <gpu>; re-evaluates A1 models with FlowWM's heatmap readout (eval_hm*.json):
# joint (16 -> 16) runs j_* and the autoregressive E1 runs e1_a1_*; skips existing outputs and waits
# for joint runs that are still training
source ~/imle-wm/env.sh; cd ~/imle-wm/code
export CUDA_VISIBLE_DEVICES=$1 A1_READOUT=heatmap A1_DATA=bshapes_v3
R=~/imle-wm/runs; L=~/imle-wm/logs/hm_eval.log
jev() {  # joint run, method
  local n=0; until [ -f $R/$1/ckpt.pt ] || [ $n -ge 120 ]; do sleep 120; n=$((n+1)); done
  if [ "$2" = fm ]; then for f in 50 2; do [ -f $R/$1/eval_hm_nfe$f.json ] || python -m imlewm.joint --eval $1 --nfe $f >> $L 2>&1; done
  else [ -f $R/$1/eval_hm.json ] || python -m imlewm.joint --eval $1 >> $L 2>&1; fi
}
aev() {  # autoregressive E1 run, method
  if [ "$2" = fm ]; then for f in 50 2; do [ -f $R/$1/eval_hm_nfe$f.json ] || python -m imlewm.a1eval --run $1 --nfe $f >> $L 2>&1; done
  else [ -f $R/$1/eval_hm.json ] || python -m imlewm.a1eval --run $1 >> $L 2>&1; fi
}
for m in reg imle4 imle16 fm; do jev j_${m}_s0 $m; aev e1_a1_${m}_s0 $m; done
for s in 1 2; do for m in reg imle4 imle16 fm; do aev e1_a1_${m}_s$s $m; done; done
for s in 1 2; do for m in reg fm imle4 imle16; do jev j_${m}_s$s $m; done; done
echo "$(date +%T) hmchain done" >> ~/imle-wm/logs/queue.log
