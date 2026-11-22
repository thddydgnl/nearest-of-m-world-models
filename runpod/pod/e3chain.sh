#!/bin/bash
# Rollout metrics (e3.py, h up to 32) for finished r1 runs that match <regex>, at each NFE in <nfes>, once each.
# Waits for the runs' standard evaluation first. Usage: bash e3chain.sh '^r1_fmcn' "50 2" <n_runs_expected>
source /workspace/imle-wm/env.sh
cd /workspace/imle-wm/code
PAT=$1; NFES=$2; NEXP=${3:-1}
for it in $(seq 1 600); do    # at most ~10 h
  done_all=1; n=0
  for d in /workspace/imle-wm/runs/*/; do
    r=$(basename $d); echo "$r" | grep -qE "$PAT" || continue
    n=$((n+1))
    [ -f $d/ckpt.pt ] && [ -f $d/eval_nfe50.json -o -f $d/eval.json ] || { done_all=0; continue; }
    for k in $NFES; do
      f=$d/e3_nfe$k.json; [ "$k" = 1 ] && f=$d/e3.json
      [ -f $f ] || python -m imlewm.e3 --run $r --hmax 32 --nfe $k --cs 8 > /workspace/nom/logs/e3_${r}_nfe$k.log 2>&1
    done
  done
  [ $done_all = 1 ] && [ $n -ge $NEXP ] && { echo "$(date -u +%H:%M:%SZ) e3 done for $n runs"; exit 0; }
  sleep 60
done
echo "e3chain timeout"
