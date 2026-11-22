# runpod: RunPod tools for revision r1

Copied on 2026-11-23 from `~/Documents/physical ai - Neural Networks/runpod_tmlr/` (the user's RunPod workflow) and
adapted for IMLE-WM. Design: `main/revision/DESIGN_r1.md`.

- Pod prefix `nom`. The RunPod account is shared: never touch `atm-*`, `arx-*`, `pmx-*`, `pai-*`, `nnc-*`, `wmc-*`,
  `bx7-*` or any pod that is not in `pods.ledger`.
- `budget.py`: START_BALANCE 26.62 (shared account balance at the start), watchdog LIMIT 24 (the user's cap is 30),
  HARD_DEADLINE 2026-11-27T00:00Z. Spend is estimated from the ledger, never from the balance.
  Watchdog: `python3 -u /Users/song-yonghwi/Documents/IMLE-WM/runpod/budget.py watch 300 >> runpod/watchdog.log`.
- `caps.json`: x1 $4 (experiment 1), x2 $8 (experiment 2), x3 $4 (experiment 3), an $6 (analyses).
- GPU: `NVIDIA RTX A5000` (Secure, about $0.27/h), the GPU type of the original runs.

## Order of operations for one pod

1. `TERM_H=10 bash runpod/pod_up.sh <tag> "NVIDIA RTX A5000"` (tags x1.1, x2.1, x2.2, x3.1, an.1).
2. `bash runpod/put_rclone.sh <tag>`: copies only the read-only `[gdrive-ro]` remote (user-approved).
3. `bash runpod/sync.sh <tag> push runpod/pod/ /workspace/nom/code/` (needs rsync on the pod; setup installs it,
   so for the very first push use `rsh.sh <tag> 'apt-get update -qq && apt-get install -y -qq rsync'`).
4. `bash runpod/rsh.sh <tag> 'bash /workspace/nom/code/setup_pod.sh fetch bounce_div_k4'` (add `ckpt` and other
   datasets on an.1).
5. `bash runpod/sync.sh <tag> push main/code/ /workspace/imle-wm/code/`, then
   `bash runpod/rsh.sh <tag> 'bash /workspace/nom/code/setup_pod.sh env'`.
6. Copy the pod's queue file from `main/revision/queues/` to `/workspace/nom/code/` and start workers with
   `setsid nohup bash ~/imle-wm/code/runq.sh 0 /workspace/nom/code/queue_<x>.txt > /workspace/nom/logs/w<i>.log 2>&1 < /dev/null &`
   (several workers may share one GPU and one queue), plus `evalwatch.sh 0 '^r1_'`.
7. `bash runpod/check_all.sh` to check, `bash runpod/pull_all.sh` to pull results into `main/revision/<group>/raw/`,
   `bash runpod/pod_down.sh <tag>` when done.
