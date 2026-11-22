#!/bin/bash
# Shared settings for the RunPod pods of the IMLE-WM revision round r1 (main/revision/DESIGN_r1.md; sourced by the other scripts).
# The RunPod account is shared with the user's other studies: only pods recorded in pods.ledger are ours.
INFRA="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TERM_H=${TERM_H:-10}  # every pod auto-terminates this many hours after it is created (DESIGN_r1.md section 0)
DEADLINE=${DEADLINE:-$(date -u -v+"${TERM_H}"H +%Y-%m-%dT%H:%M:%SZ)}
PREFIX=nom   # nearest-of-m revision r1; never touch "atm", "arx", "pmx", "pai", "nnc", "wmc", "bx7" (other studies on the shared account)
LEDGER="$INFRA/pods.ledger"
SSH_KEY="$HOME/.runpod/ssh/runpodctl-ssh-key"
mkdir -p "$INFRA/pods"
