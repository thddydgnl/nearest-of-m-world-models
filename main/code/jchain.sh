#!/bin/bash
# usage: jchain.sh <gpu> "<method>:<name>:<extra args, comma-separated>" ...
# trains each joint-protocol (16 -> 16) run, then evaluates it (fm: NFE 50 and 2)
source ~/imle-wm/env.sh; cd ~/imle-wm/code; export CUDA_VISIBLE_DEVICES=$1; shift
for spec in "$@"; do
  meth=${spec%%:*}; rest=${spec#*:}; name=${rest%%:*}; extra=${rest#*:}; extra=${extra//,/ }
  echo "$(date +%T) gpu$CUDA_VISIBLE_DEVICES start $name" >> ~/imle-wm/logs/queue.log
  [ -f ~/imle-wm/runs/$name/ckpt.pt ] || python -m imlewm.joint --method $meth --name $name $extra > ~/imle-wm/logs/$name.log 2>&1
  echo "$(date +%T) gpu$CUDA_VISIBLE_DEVICES end $name rc=$?" >> ~/imle-wm/logs/queue.log
  if [ "$meth" = fm ]; then for n in 50 2; do python -m imlewm.joint --eval $name --nfe $n >> ~/imle-wm/logs/eval_$name.log 2>&1; done
  else python -m imlewm.joint --eval $name >> ~/imle-wm/logs/eval_$name.log 2>&1; fi
done
echo "$(date +%T) jchain gpu$CUDA_VISIBLE_DEVICES done" >> ~/imle-wm/logs/queue.log
