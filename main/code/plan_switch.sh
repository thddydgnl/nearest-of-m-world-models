#!/bin/bash
# waits for the running FM-2 planning job (pid 430711) to finish, then replaces the old planning loop
n=0; while ps -p 430711 > /dev/null && [ $n -lt 120 ]; do sleep 60; n=$((n+1)); done
tmux kill-session -t pc 2>/dev/null; sleep 5
tmux new -d -s pc2 "bash ~/imle-wm/code/planchain.sh"
echo "$(date +%T) planning switched to retrying chain (pc2)" >> ~/imle-wm/logs/queue.log
