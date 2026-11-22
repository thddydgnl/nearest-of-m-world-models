#!/bin/bash
# Create one pod of the TMLR extension (tags td.N, dw.N, mt.N, k.N; replacements e.g. td.3b). Tries GPU types in order until one is available.
# Usage: [IMAGE=<docker image>] [TERM_H=4] [MIN_CUDA=12.6] pod_up.sh <tag> "<gpu1>|<gpu2>|..." [container_gb=40] [volume_gb=60]
# Default is the runpod-torch-v240 template (Ubuntu 22.04, Python 3.11); jobs install torch 2.7.1 (cu126) in a uv venv,
# so the host driver must support CUDA >= 12.6 (MIN_CUDA). Blackwell GPUs (RTX 5090, RTX PRO *) only with a CUDA 12.8+
# image, e.g. IMAGE=runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404 MIN_CUDA=12.8.
# Writes pods/<tag>.env (POD_ID/POD_IP/POD_PORT/POD_GPU/POD_COST) and appends an "up" event to pods.ledger.
set -u
source "$(dirname "$0")/common.sh"
TAG=$1
IFS='|' read -ra GPUS <<< "$2"
CDISK=${3:-40}
VOL=${4:-60}
NAME="$PREFIX-$TAG"
if [ -n "${IMAGE:-}" ]; then SRC=(--image "$IMAGE" --ports "22/tcp"); else SRC=(--template-id runpod-torch-v240); fi  # a bare image exposes no ssh port unless asked
if [ -f "$INFRA/pods/$TAG.env" ]; then echo "pods/$TAG.env exists; run pod_down.sh $TAG first"; exit 1; fi
for G in "${GPUS[@]}"; do
  echo "trying: $G (terminate-after $DEADLINE)"
  if runpodctl pod create "${SRC[@]}" --gpu-id "$G" --cloud-type SECURE --min-cuda-version "${MIN_CUDA:-12.6}" \
       --name "$NAME" --container-disk-in-gb "$CDISK" --volume-in-gb "$VOL" --volume-mount-path /workspace \
       --terminate-after "$DEADLINE" --wait --wait-timeout 12m > "$INFRA/pods/$TAG.json" 2> "$INFRA/pods/$TAG.err"; then
    ID=$(jq -r .id "$INFRA/pods/$TAG.json")
    COST=$(jq -r .costPerHr "$INFRA/pods/$TAG.json")
    NOW=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    jq -nc --arg id "$ID" --arg tag "$TAG" --arg gpu "$G" --arg cost "$COST" --arg time "$NOW" \
      '{id:$id, tag:$tag, gpu:$gpu, cost:$cost, event:"up", time:$time}' >> "$LEDGER"
    runpodctl ssh info "$ID" > "$INFRA/pods/$TAG.ssh.json"
    {
      echo "POD_ID=$ID"
      echo "POD_IP=$(jq -r .ip "$INFRA/pods/$TAG.ssh.json")"
      echo "POD_PORT=$(jq -r .port "$INFRA/pods/$TAG.ssh.json")"
      echo "POD_GPU=\"$G\""
      echo "POD_COST=$COST"
      echo "POD_CREATED=$NOW"
      echo "POD_DEADLINE=$DEADLINE"
    } > "$INFRA/pods/$TAG.env"
    cat "$INFRA/pods/$TAG.env"
    exit 0
  fi
  echo "failed: $(head -c 300 "$INFRA/pods/$TAG.err")"
  # The pod may exist although the read-back failed (rest.runpod.io outage, 2026-11-21): look it up by name through
  # GraphQL. Keep it if its ssh port comes up within 12 minutes, otherwise terminate it (recorded as a stray).
  FOUND=""
  for try in 1 2 3 4 5 6; do
    FOUND=$(python3 "$INFRA/rp_gql.py" find "$NAME" 2>/dev/null)
    [ -n "$FOUND" ] && break
    sleep 10
  done
  if [ -n "$FOUND" ]; then
    ID=$(echo "$FOUND" | head -1 | jq -r .id)
    COST=$(echo "$FOUND" | head -1 | jq -r .costPerHr)
    NOW=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    grep -q "\"id\":\"$ID\"" "$LEDGER" || jq -nc --arg id "$ID" --arg tag "$TAG" --arg gpu "$G" --arg cost "$COST" --arg time "$NOW" \
      '{id:$id, tag:$tag, gpu:$gpu, cost:$cost, event:"up", time:$time}' >> "$LEDGER"
    for XID in $(echo "$FOUND" | tail -n +2 | jq -r .id); do   # duplicates with our name: record, then terminate
      grep -q "\"id\":\"$XID\"" "$LEDGER" || jq -nc --arg id "$XID" --arg tag "$TAG-stray" --arg gpu "$G" --arg cost "$COST" \
        --arg time "$NOW" '{id:$id, tag:$tag, gpu:$gpu, cost:$cost, event:"up", time:$time}' >> "$LEDGER"
      python3 "$INFRA/rp_gql.py" terminate "$XID" && jq -nc --arg id "$XID" --arg tag "$TAG-stray" \
        --arg time "$(date -u +%Y-%m-%dT%H:%M:%SZ)" '{id:$id, tag:$tag, event:"deleted", time:$time}' >> "$LEDGER"
    done
    S=""
    for i in $(seq 1 72); do
      S=$(python3 "$INFRA/rp_gql.py" sshinfo "$ID" 2>/dev/null) && [ -n "$S" ] && break
      sleep 10
    done
    if [ -n "$S" ]; then
      {
        echo "POD_ID=$ID"
        echo "POD_IP=${S% *}"
        echo "POD_PORT=${S#* }"
        echo "POD_GPU=\"$G\""
        echo "POD_COST=$COST"
        echo "POD_CREATED=$NOW"
        echo "POD_DEADLINE=$DEADLINE"
      } > "$INFRA/pods/$TAG.env"
      echo "recovered through GraphQL (REST read-back failed)"
      cat "$INFRA/pods/$TAG.env"
      exit 0
    fi
    echo "pod $ID has no ssh port after 12 min: terminating it"
    python3 "$INFRA/rp_gql.py" terminate "$ID" && jq -nc --arg id "$ID" --arg tag "$TAG" \
      --arg time "$(date -u +%Y-%m-%dT%H:%M:%SZ)" '{id:$id, tag:$tag, event:"deleted", time:$time}' >> "$LEDGER"
  fi
done
echo "no gpu available"
exit 1
