#!/bin/bash
# Analysis e scheduler (replaces the three e_replay.sh workers after their IMLE-WM replays):
# wait for the running replay processes, then regression s0-s2 in parallel, then FM-2 s0, s1, s2 one after the other
# (same total GPU work as running them together, but complete seeds arrive one at a time).
# usage: e_sched.sh <pid> [<pid> ...]   (PIDs of the running IMLE-WM replays)
source ~/imle-wm/env.sh
cd ~/imle-wm/code
O=/workspace/nom/out/e_planrisk
log() { echo "$(date -u +%H:%M:%SZ) $*"; }
for p in "$@"; do while kill -0 $p 2>/dev/null; do sleep 20; done; done
log "imle replays finished"
for s in 0 1 2; do
  python -m imlewm.r1_planrisk replay --model e13_push_reg_s$s --out $O > /workspace/nom/logs/e_replay_e13_push_reg_s$s.log 2>&1 &
done
wait
log "reg replays finished"
for s in 0 1 2; do
  [ -f /workspace/nom/STOP_E ] && { log "stop file found, skip remaining FM replays"; break; }
  log "start fm s$s"
  python -m imlewm.r1_planrisk replay --model e13_push_fm_s$s --out $O > /workspace/nom/logs/e_replay_e13_push_fm_s$s.log 2>&1
  log "end fm s$s rc $?"
done
log "e_sched done"
