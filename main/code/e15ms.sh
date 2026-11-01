#!/bin/bash
# re-run of the e15chain.sh mode-share step for one run (the chain's first try hit OOM next to ES m=16 training)
source ~/imle-wm/env.sh; cd ~/imle-wm/code
export CUDA_VISIBLE_DEVICES=$2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=~/imle-wm/runs
[ -f $R/$1/modeshare.json ] || python -c "
import json, torch
from imlewm.check import memo_model
from imlewm.wm import load_model
torch.manual_seed(123)
m, ck = load_model('$R/$1/ckpt.pt')
r = memo_model(m, ck['args']['data'], n_ctx=400, n=64)
json.dump(r, open('$R/$1/modeshare.json', 'w'), indent=1)
print('$1', {k: (v['share0'], v['data_freq0'], v['cov']) for k, v in r.items()}, flush=True)"
echo "$(date +%T) e15ms $1 done rc=$?" >> ~/imle-wm/logs/queue.log
