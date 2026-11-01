#!/bin/bash
# Starts the E11 follow-up jobs once both GPUs are usable; idempotent (skips running sessions).
# exit 1 = GPUs not available yet
source ~/imle-wm/env.sh
cd $IMLE/code
python -c "import torch,sys; sys.exit(0 if torch.cuda.device_count()==2 else 1)" 2>/dev/null || exit 1
for g in 0 1; do
  tmux has -t qE1g$g 2>/dev/null || tmux new -d -s qE1g$g "bash $IMLE/code/runq.sh $g $IMLE/code/qE11.txt"
done
tmux has -t evalw 2>/dev/null || tmux new -d -s evalw "bash $IMLE/code/evalwatch.sh 1 '^e11_'"
tmux has -t e11chain 2>/dev/null || tmux new -d -s e11chain "bash $IMLE/code/e11chain.sh 1"
tmux has -t watchdog 2>/dev/null || tmux new -d -s watchdog "bash $IMLE/code/watchdog.sh $IMLE/code/qE11.txt '^e11_'"
echo "$(date +%T) e11 started" >> $IMLE/logs/queue.log
