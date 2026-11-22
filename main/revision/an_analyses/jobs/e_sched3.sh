#!/bin/bash
# Analysis e scheduler, third version: after the running FM-2 s0 replay (PID argument), FM-2 s1 and s2 one after the
# other with the model weights in channels_last layout (about 1.3x faster on the A40; same computation up to bf16 rounding).
# usage: e_sched3.sh <pid of the FM s0 replay>
source ~/imle-wm/env.sh
cd ~/imle-wm/code
O=/workspace/nom/out/e_planrisk
log() { echo "$(date -u +%H:%M:%SZ) $*"; }
while kill -0 $1 2>/dev/null; do sleep 20; done
log "fm s0 finished"
for s in 1 2; do
  [ -f /workspace/nom/STOP_E ] && { log "stop file found, skip remaining FM replays"; break; }
  log "start fm s$s (channels_last)"
  python -m imlewm.r1_planrisk replay --model e13_push_fm_s$s --channels_last --out $O > /workspace/nom/logs/e_replay_e13_push_fm_s$s.log 2>&1
  log "end fm s$s rc $?"
done
log "e_sched3 done"
