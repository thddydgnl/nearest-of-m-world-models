#!/bin/bash
# E5: run B planning for each model once its E1 checkpoint exists (GPU1, alongside training)
source ~/imle-wm/env.sh; cd $IMLE/code
JOBS="e1_b_imle16_s0:1:100 e1_b_reg_s0:1:100 e1_b_fm_s0:2:100 e1_b_imle16_s1:1:100 e1_b_reg_s1:1:100 e1_b_fm_s1:2:100 e1_b_imle16_s2:1:100 e1_b_reg_s2:1:100 e1_b_fm_s2:2:100 e1_b_fm_s0:50:30"
for j in $JOBS; do
  IFS=: read run nfe eps <<< "$j"
  seed=${run##*_s}
  out=$IMLE/runs/e5/${run}_nfe${nfe}_s${seed}.json
  [ -f $out ] && continue
  until [ -f $IMLE/runs/$run/ckpt.pt ]; do sleep 120; done
  echo "$(date +%T) e5 start $run nfe$nfe" >> $IMLE/logs/queue.log
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1 python -m imlewm.bplan --run $run --nfe $nfe --eps $eps --seed $seed > $IMLE/logs/e5_${run}_nfe$nfe.log 2>&1
  echo "$(date +%T) e5 end $run nfe$nfe rc=$?" >> $IMLE/logs/queue.log
done
