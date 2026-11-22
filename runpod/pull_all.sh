#!/bin/bash
# Pull the small result files of every live pod into main/revision/<group>/raw/<tag>/ :
#   runs/  <- /workspace/imle-wm/runs/r1_*/ without checkpoints (log.jsonl, eval*.json, e3*.json, modeshare*.json, ...)
#   out/   <- /workspace/nom/out/ (analysis outputs, each file < 20 MB)
# Groups: x1 -> x1_arc, x2 -> x2_long, x3 -> x3_ctxnoise, an -> an_analyses. Never deletes anything locally or on the pod.
# Usage: pull_all.sh [tag ...]     (default: every pod with a pods/<tag>.env)
N="/Users/song-yonghwi/Documents/IMLE-WM/runpod"
A="/Users/song-yonghwi/Documents/IMLE-WM/main/revision"
al() { perl -e 'alarm shift; exec @ARGV' "$@"; }
tags=("$@")
if [ ${#tags[@]} -eq 0 ]; then
  for env in "$N"/pods/*.env; do [ -f "$env" ] && tags+=("$(basename "$env" .env)"); done
fi
for tag in "${tags[@]}"; do
  case "${tag%%.*}" in
    x1) g=x1_arc ;; x2) g=x2_long ;; x3) g=x3_ctxnoise ;; an) g=an_analyses ;; *) echo "skip $tag (unknown group)"; continue ;;
  esac
  dst="$A/$g/raw/$tag"
  mkdir -p "$dst/runs" "$dst/out"
  al 900 bash "$N/sync.sh" "$tag" pull /workspace/imle-wm/runs/ "$dst/runs/" --include='r1_*/' --exclude='*.pt' \
    --include='r1_*/**' --exclude='*' 2> "$N/pods/$tag.pull.err"; rc1=$?
  al 900 bash "$N/sync.sh" "$tag" pull /workspace/nom/out/ "$dst/out/" --exclude='*.tmp*' 2>> "$N/pods/$tag.pull.err"; rc2=$?
  for rc in $rc1 $rc2; do
    if [ $rc -ne 0 ] && [ $rc -ne 24 ]; then echo "PULL FAILED $tag (rc $rc): $(head -c 300 "$N/pods/$tag.pull.err")"; fi
  done
  echo "pulled $tag -> main/revision/$g/raw/$tag ($(find "$dst" -type f | wc -l | tr -d ' ') files, $(du -sh "$dst" | cut -f1))"
done
