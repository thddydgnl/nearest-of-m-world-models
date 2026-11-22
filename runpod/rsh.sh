#!/bin/bash
# SSH into the pod of one idea. Usage: rsh.sh <tag> [remote command...]   (stdin is forwarded)
source "$(dirname "$0")/common.sh"
TAG=$1
shift
source "$INFRA/pods/$TAG.env"
exec ssh -i "$SSH_KEY" -p "$POD_PORT" -o StrictHostKeyChecking=no -o UserKnownHostsFile="$INFRA/pods/$TAG.known_hosts" \
  -o ConnectTimeout=25 -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -o LogLevel=ERROR root@"$POD_IP" "$@"
