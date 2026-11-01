#!/bin/bash
# usage: curvechain.sh <gpu> <run>; waits for the run to finish, then evaluates every ckpt_<step>.pt (NFE 1)
source ~/imle-wm/env.sh
cd $IMLE/code
R=$IMLE/runs/$2
until [ -f $R/ckpt.pt ]; do sleep 300; done
for f in $(ls $R/ckpt_*.pt | sort -V); do
  s=$(basename $f .pt)
  [ -f $R/eval_${s}_nfe1.json ] || PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$1 python -m imlewm.evaluate --run $2 --ckpt $s --nfe 1 > $IMLE/logs/eval_$2_$s.log 2>&1
done
echo "$(date +%T) curve $2 done" >> $IMLE/logs/queue.log
