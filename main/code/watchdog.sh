#!/bin/bash
# usage: watchdog.sh <queue file> <evalwatch pattern>
# Server-side supervisor that does not depend on the Claude session (runs every 5 min):
#  - restarts dead queue workers (qE1g0/qE1g1) while the queue still has lines
#  - restarts the eval watcher (evalw)
#  - re-queues a failed training run once (front of the queue) after an OOM; other failures are
#    only logged, so a code bug does not loop
#  - flags runs whose last log line is older than 30 min (hang) and NaN losses
# Everything it does goes to logs/watchdog.log.
source ~/imle-wm/env.sh
cd $IMLE/code
Q=$1; PAT=$2
LOG=$IMLE/logs/watchdog.log
DONE=$IMLE/logs/.watchdog_seen
touch $DONE
say() { echo "$(date -u +%m-%dT%H:%M) $*" >> $LOG; }
while true; do
  nq=$(grep -cv '^\s*$' $Q)
  for g in 0 1; do
    if ! tmux has -t qE1g$g 2>/dev/null && [ "$nq" -gt 0 ]; then
      tmux new -d -s qE1g$g "bash $IMLE/code/runq.sh $g $Q"; say "restarted worker qE1g$g (queue $nq)"
    fi
  done
  if ! tmux has -t evalw 2>/dev/null; then
    tmux new -d -s evalw "bash $IMLE/code/evalwatch.sh 1 \"$PAT\""; say "restarted evalw"
  fi
  # failed runs: 'end <name> rc=<nonzero>' lines not handled yet
  grep -E 'end .* rc=[1-9]' $IMLE/logs/queue.log | while read -r line; do
    grep -qxF "$line" $DONE && continue
    echo "$line" >> $DONE
    name=$(echo "$line" | awk '{print $4}')
    if grep -q "OutOfMemory" $IMLE/logs/$name.log && ! grep -q "requeued $name" $LOG; then
      args=$(grep -h -- "--name $name\$" $IMLE/logs/watchdog_args.txt 2>/dev/null | tail -1)
      if [ -n "$args" ]; then
        cp $IMLE/logs/$name.log $IMLE/logs/${name}_oom.log
        echo "$args" > $Q.front
        flock $Q.lock bash -c "cat $Q.front $Q > $Q.tmp && cat $Q.tmp > $Q && rm -f $Q.tmp $Q.front"
        say "requeued $name after OOM"
      else
        say "FAILED $name (OOM, args unknown)"
      fi
    else
      say "FAILED $name: $(grep -m1 -E 'Error|error' $IMLE/logs/$name.log | cut -c1-150)"
    fi
  done
  # running runs: hang and nan checks
  for s in $(tail -n 40 $IMLE/logs/queue.log | grep start | awk '{print $4}'); do
    grep -q "end $s " $IMLE/logs/queue.log && continue
    f=$IMLE/logs/$s.log; [ -f $f ] || continue
    age=$(( $(date +%s) - $(stat -c %Y $f) ))
    [ $age -gt 1800 ] && ! grep -q "stale $s" $LOG && say "stale $s: no log line for $((age/60)) min"
    tail -n 1 $f | grep -qi "nan" && ! grep -q "nan $s" $LOG && say "nan $s: $(tail -n 1 $f | cut -c1-150)"
  done
  # remember the args of every queued line, for OOM re-queues
  cat $Q >> $IMLE/logs/watchdog_args.txt; sort -u -o $IMLE/logs/watchdog_args.txt $IMLE/logs/watchdog_args.txt
  sleep 300
done
