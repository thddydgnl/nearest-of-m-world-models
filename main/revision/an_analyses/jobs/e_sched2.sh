#!/bin/bash
# Analysis e scheduler, second version: regression replays now (alongside the IMLE-WM replays), then FM-2 s0, s1, s2
# one after the other once the given PIDs (IMLE-WM replays) and the regression replays have finished.
# usage: e_sched2.sh <pid> [<pid> ...]
source ~/imle-wm/env.sh
cd ~/imle-wm/code
O=/workspace/nom/out/e_planrisk
log() { echo "$(date -u +%H:%M:%SZ) $*"; }
for s in 0 1 2; do
  python -m imlewm.r1_planrisk replay --model e13_push_reg_s$s --out $O > /workspace/nom/logs/e_replay_e13_push_reg_s$s.log 2>&1 &
done
log "reg replays started"
for p in "$@"; do while kill -0 $p 2>/dev/null; do sleep 20; done; done
log "imle replays finished"
wait
log "reg replays finished"
for s in 0 1 2; do
  [ -f /workspace/nom/STOP_E ] && { log "stop file found, skip remaining FM replays"; break; }
  log "start fm s$s"
  python -m imlewm.r1_planrisk replay --model e13_push_fm_s$s --out $O > /workspace/nom/logs/e_replay_e13_push_fm_s$s.log 2>&1
  log "end fm s$s rc $?"
done
log "e_sched2 done"
