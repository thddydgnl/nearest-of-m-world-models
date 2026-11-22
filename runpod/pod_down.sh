#!/bin/bash
# Delete the pod of one pilot. Refuses pods that are not in our ledger.
# Pull results first (sync.sh <tag> pull ...): the pod's disk is gone after this.
# Usage: pod_down.sh <tag>
source "$(dirname "$0")/common.sh"
TAG=$1
F="$INFRA/pods/$TAG.env"
[ -f "$F" ] || { echo "no pods/$TAG.env"; exit 1; }
source "$F"
grep -q "\"id\":\"$POD_ID\"" "$LEDGER" || { echo "$POD_ID not in ledger; refusing"; exit 1; }
if ! runpodctl pod delete "$POD_ID" 2> "$INFRA/pods/$TAG.del.err"; then
  echo "runpodctl delete failed ($(head -c 200 "$INFRA/pods/$TAG.del.err")); trying GraphQL"
  python3 "$INFRA/rp_gql.py" terminate "$POD_ID" || exit 1
fi
jq -nc --arg id "$POD_ID" --arg tag "$TAG" --arg time "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  '{id:$id, tag:$tag, event:"deleted", time:$time}' >> "$LEDGER"
mv "$F" "$INFRA/pods/$TAG.env.deleted-$(date -u +%H%M%S)"
echo "deleted $TAG $POD_ID"
