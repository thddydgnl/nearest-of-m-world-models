#!/bin/bash
# usage: e14chain.sh <gpu>; follow-ups from the 2026-10-16 literature re-check:
#  (1) asymmetric-mode bounce data bounce_div_k2_p80 (mode 0 has p=0.8) -> IMLE m=2/4/16 and FM runs
#      are appended to qE12.txt; theory (theory_m_law.md sec. 6) predicts IMLE mode-0 share 0.80/0.61/0.52
#  (2) noise-augmented FlowWM heatmap readout (noise std 0.5, FlowWM's noise_std0p5 config) and
#      re-evaluation of the A1 models with it (eval_hmn05*.json)
#  (3) mode-share evaluation of the E14 runs (runs/<run>/modeshare*.json), waiting for checkpoints
source ~/imle-wm/env.sh; cd ~/imle-wm/code
export CUDA_VISIBLE_DEVICES=$1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=~/imle-wm/runs; L=~/imle-wm/logs/e14.log; D=~/imle-wm/data/bounce_div_k2_p80
# (1) data
[ -f $D/stoch_branch_tokens.npy ] || python -m imlewm.gen_data --name bounce_div_k2_p80 >> $L 2>&1
[ -f $D/stats.pt ] || python -c "from imlewm.gen_data import compute_stats; from imlewm.common import ds_dir; compute_stats(ds_dir('bounce_div_k2_p80'))" >> $L 2>&1
[ -f $D/probe.pt ] || python -m imlewm.probe --data bounce_div_k2_p80 >> $L 2>&1
if ! grep -q e14_asym ~/imle-wm/code/qE12.txt && [ ! -d $R/e14_asym_m16_s0 ]; then
  : > /tmp/q14.txt
  for s in 0 1; do
    for m in 16 4 2; do echo "--data bounce_div_k2_p80 --method imle --stages single --wide_head 1024 --z_from 5 --m $m --steps 8000 --seed $s --name e14_asym_m${m}_s$s" >> /tmp/q14.txt; done
    echo "--data bounce_div_k2_p80 --method fm --t_shift 13 --stages single --wide_head 1024 --steps 8000 --seed $s --name e14_asym_fm_s$s" >> /tmp/q14.txt
  done
  flock ~/imle-wm/code/qE12.txt.lock bash -c "cat /tmp/q14.txt >> ~/imle-wm/code/qE12.txt"
  cat /tmp/q14.txt >> ~/imle-wm/logs/watchdog_args.txt
  echo "$(date +%T) e14 runs queued" >> ~/imle-wm/logs/queue.log
fi
# (2) noise-augmented readout + A1 re-evaluation
[ -f ~/imle-wm/data/bshapes_v3/heatmap_readout_n05.pt ] || python -m imlewm.heatmap --train --noise 0.5 --tag n05 >> $L 2>&1
export A1_READOUT=heatmap A1_READOUT_TAG=n05 A1_DATA=bshapes_v3
for s in 0 1 2; do
  for m in reg imle4 imle16; do
    [ -f $R/j_${m}_s$s/eval_hmn05.json ] || python -m imlewm.joint --eval j_${m}_s$s >> $L 2>&1
    [ -f $R/e1_a1_${m}_s$s/eval_hmn05.json ] || python -m imlewm.a1eval --run e1_a1_${m}_s$s >> $L 2>&1
  done
  for f in 50 2; do
    [ -f $R/j_fm_s$s/eval_hmn05_nfe$f.json ] || python -m imlewm.joint --eval j_fm_s$s --nfe $f >> $L 2>&1
    [ -f $R/e1_a1_fm_s$s/eval_hmn05_nfe$f.json ] || python -m imlewm.a1eval --run e1_a1_fm_s$s --nfe $f >> $L 2>&1
  done
done
unset A1_READOUT A1_READOUT_TAG
echo "$(date +%T) e14 readout n05 done" >> ~/imle-wm/logs/queue.log
# (3) mode shares (test and train contexts, 400 contexts x 64 samples)
ms() {  # run nfe
  local n=0; until [ -f $R/$1/ckpt.pt ] || [ $n -ge 400 ]; do sleep 120; n=$((n+1)); done
  local tag=""; [ -n "$2" ] && tag="_nfe$2"
  [ -f $R/$1/modeshare$tag.json ] || python -c "
import json, torch
from imlewm.check import memo_model
from imlewm.wm import load_model
torch.manual_seed(123)
m, ck = load_model('$R/$1/ckpt.pt')
r = memo_model(m, ck['args']['data'], n_ctx=400, n=64, nfe=${2:-None})
json.dump(r, open('$R/$1/modeshare$tag.json', 'w'), indent=1)
print('$1', '$2', {k: (v['share0'], v['data_freq0'], v['cov']) for k, v in r.items()}, flush=True)" >> $L 2>&1
}
for s in 0 1; do ms e14_asym_m16_s$s; ms e14_asym_m4_s$s; ms e14_asym_m2_s$s; ms e14_asym_fm_s$s 50; ms e14_asym_fm_s$s 2; done
echo "$(date +%T) e14chain done" >> ~/imle-wm/logs/queue.log
