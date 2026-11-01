#!/bin/bash
# E3 rollout-stability evaluation for all E1 models (GPU0, alongside training)
source ~/imle-wm/env.sh; cd $IMLE/code
for s in 0 1 2; do
  for b in a2 a1 b; do
    case $b in a1) H=16;; a2) H=32;; b) H=16;; esac
    for m in reg fm imle4 imle16 imf; do
      r=e1_${b}_${m}_s$s; [ -f $IMLE/runs/$r/ckpt.pt ] || continue
      nfe=1; [ $m = fm ] && nfe=2
      tag=""; [ $m = fm ] || [ $m = imf ] && tag="_nfe$nfe"
      [ -f $IMLE/runs/$r/e3$tag.json ] && continue
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 python -m imlewm.e3 --run $r --hmax $H --nfe $nfe > $IMLE/logs/e3_$r.log 2>&1
    done
  done
done
echo "$(date +%T) e3 chain done" >> $IMLE/logs/queue.log
