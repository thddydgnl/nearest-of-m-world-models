#!/bin/bash
# usage: evalwatch.sh <gpu> <run-name regex>; evaluates each finished matching run once
# fm: NFE 50 and 2 (FlowWM recipe), mf/imf/sc: NFE 1 and 2, others: 1. A1 (bshapes) runs use a1eval, B (pusht_*) b_eval, BAIR bair_eval.
source ~/imle-wm/env.sh
cd $IMLE/code
PAT=${2:-.}
while true; do
  for d in $IMLE/runs/*/; do
    r=$(basename $d); [ -f $d/ckpt.pt ] || continue
    echo "$r" | grep -qE "$PAT" || continue
    read meth data <<< $(python -c "import torch;a=torch.load('$d/ckpt.pt',map_location='cpu')['args'];print(a['method'],a['data'])" 2>/dev/null)
    case $meth in
      fm) NFES="50 2";; mf|imf|sc) NFES="1 2";; *) NFES="";;
    esac
    MOD=imlewm.evaluate; [[ "$data" == bshapes* ]] && MOD=imlewm.a1eval; [[ "$data" == pusht_* ]] && MOD=imlewm.b_eval; [[ "$data" == bair ]] && MOD=imlewm.bair_eval
    if [ -n "$NFES" ]; then
      for n in $NFES; do [ -f $d/eval_nfe$n.json ] || PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$1 python -m $MOD --run $r --nfe $n > $IMLE/logs/eval_${r}_nfe$n.log 2>&1; done
    else
      [ -f $d/eval.json ] || PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$1 python -m $MOD --run $r > $IMLE/logs/eval_$r.log 2>&1
    fi
  done
  sleep 60
done
