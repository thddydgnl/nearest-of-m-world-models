#!/bin/bash
# rsync helper (the pod needs `apt-get install -y rsync` once).
# Usage: sync.sh <tag> push <local> <remote> [rsync args] | sync.sh <tag> pull <remote> <local> [rsync args]
source "$(dirname "$0")/common.sh"
TAG=$1
MODE=$2
A=$3
B=$4
shift 4
source "$INFRA/pods/$TAG.env"
SSH="ssh -i $SSH_KEY -p $POD_PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=25 -o LogLevel=ERROR"
if [ "$MODE" = push ]; then
  exec rsync -az --no-owner --no-group -e "$SSH" "$@" "$A" "root@$POD_IP:$B"
else
  exec rsync -az -e "$SSH" "$@" "root@$POD_IP:$A" "$B"
fi
