#!/bin/bash
# Analysis e: CEM planner replays, one model after the other. usage: e_replay.sh <model> [<model> ...]
source ~/imle-wm/env.sh
cd ~/imle-wm/code
O=/workspace/nom/out/e_planrisk
mkdir -p $O
for m in "$@"; do
  echo "$(date -u +%H:%M:%SZ) start $m"
  python -m imlewm.r1_planrisk replay --model $m --out $O > /workspace/nom/logs/e_replay_$m.log 2>&1
  echo "$(date -u +%H:%M:%SZ) end $m rc $?"
done
