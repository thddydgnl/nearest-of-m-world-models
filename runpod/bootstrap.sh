#!/bin/bash
# Bootstrap one pod: apt tools, the read-only Drive remote (user-approved), pod scripts, rclone/uv, then start the
# Drive fetch and (unless "nocode") the code push + venv build in the background on the pod.
# Usage: bootstrap.sh <tag> [nocode] <fetch items...>      e.g. bootstrap.sh x1.1 bounce_div_k4
set -euo pipefail
source "$(dirname "$0")/common.sh"
TAG=$1; shift
CODE=1; if [ "${1:-}" = nocode ]; then CODE=0; shift; fi
al() { perl -e 'alarm shift; exec @ARGV' "$@"; }
al 600 bash "$INFRA/rsh.sh" "$TAG" "apt-get update -qq >/dev/null && apt-get install -y -qq rsync unzip >/dev/null && mkdir -p /workspace/nom/code /workspace/nom/logs /workspace/nom/out /workspace/imle-wm/code && echo $TAG > /workspace/nom/tag"
al 120 bash "$INFRA/put_rclone.sh" "$TAG"
al 300 bash "$INFRA/sync.sh" "$TAG" push "$INFRA/pod/" /workspace/nom/code/
al 600 bash "$INFRA/rsh.sh" "$TAG" "bash /workspace/nom/code/setup_pod.sh tools > /workspace/nom/logs/tools.log 2>&1; tail -1 /workspace/nom/logs/tools.log"
al 60 bash "$INFRA/rsh.sh" "$TAG" "setsid nohup bash /workspace/nom/code/setup_pod.sh fetch $* > /workspace/nom/logs/fetch.log 2>&1 < /dev/null &"
if [ $CODE = 1 ]; then
  al 600 bash "$INFRA/sync.sh" "$TAG" push /Users/song-yonghwi/Documents/IMLE-WM/main/code/ /workspace/imle-wm/code/ \
    --exclude '__pycache__' --exclude '.DS_Store' --exclude '*.lock'
  al 60 bash "$INFRA/rsh.sh" "$TAG" "setsid nohup bash /workspace/nom/code/setup_pod.sh env > /workspace/nom/logs/env.log 2>&1 < /dev/null &"
fi
echo "bootstrapped $TAG (fetch: $*; code+env: $CODE)"
