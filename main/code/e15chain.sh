#!/bin/bash
# usage: e15chain.sh <gpu>; energy-score 1-step baseline (2026-10-30, results/novelty_analysis.md sec. 6):
# evaluates the qE15.txt runs as their checkpoints appear
#  A2-div K=4: eval.json (coverage/precision/recall, det error, latency) and e3.json (rollout manifold)
#  E14 asymmetric modes: modeshare.json (400 contexts x 64 samples, same as e14chain.sh)
source ~/imle-wm/env.sh; cd ~/imle-wm/code
export CUDA_VISIBLE_DEVICES=$1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=~/imle-wm/runs; L=~/imle-wm/logs/e15.log
wait_ck() {  # at most 10h per run
  local n=0; until [ -f $R/$1/ckpt.pt ] || [ $n -ge 300 ]; do sleep 120; n=$((n+1)); done
  [ -f $R/$1/ckpt.pt ]
}
a2() {
  wait_ck $1 || return
  [ -f $R/$1/eval.json ] || python -m imlewm.evaluate --run $1 >> $L 2>&1
  [ -f $R/$1/e3.json ] || python -m imlewm.e3 --run $1 >> $L 2>&1
}
ms() {
  wait_ck $1 || return
  [ -f $R/$1/modeshare.json ] || python -c "
import json, torch
from imlewm.check import memo_model
from imlewm.wm import load_model
torch.manual_seed(123)
m, ck = load_model('$R/$1/ckpt.pt')
r = memo_model(m, ck['args']['data'], n_ctx=400, n=64)
json.dump(r, open('$R/$1/modeshare.json', 'w'), indent=1)
print('$1', {k: (v['share0'], v['data_freq0'], v['cov']) for k, v in r.items()}, flush=True)" >> $L 2>&1
}
a2 e15_a2_es8_s0; ms e15_asym_es8_s0; a2 e15_a2_es16_s0; a2 e15_a2_es8_s1; a2 e15_a2_es8_s2; ms e15_asym_es8_s1
echo "$(date +%T) e15chain done" >> ~/imle-wm/logs/queue.log
