#!/bin/bash
# usage: runq.sh <gpu> <queue file>; pops one line at a time (flock) and runs it
source ~/imle-wm/env.sh
cd $IMLE/code
Q=$2
while true; do
  LINE=$(flock $Q.lock bash -c "head -n1 $Q; sed -i '1d' $Q")
  [ -z "$LINE" ] && break
  NAME=$(echo "$LINE" | grep -oP '(?<=--name )\S+')
  echo "$(date +%T) gpu$1 start $NAME" >> $IMLE/logs/queue.log
  CUDA_VISIBLE_DEVICES=$1 python -m imlewm.train $LINE > $IMLE/logs/$NAME.log 2>&1
  echo "$(date +%T) gpu$1 end $NAME rc=$?" >> $IMLE/logs/queue.log
done
