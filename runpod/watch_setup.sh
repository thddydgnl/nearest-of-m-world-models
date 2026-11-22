#!/bin/bash
# Poll every pod's fetch/env state once a minute; print a line when it changes; exit when all are ready or one fails.
cd /Users/song-yonghwi/Documents/IMLE-WM
al() { perl -e 'alarm shift; exec @ARGV' "$@"; }
prev=""
for i in $(seq 1 28); do
  cur=""; alldone=1
  for t in x1.1 x2.1 x2.2 x3.1 an.1; do
    s=$(al 45 bash runpod/rsh.sh $t 'f=/workspace/nom/logs/fetch.log; e=/workspace/nom/logs/env.log; fs=run; grep -q "fetch done" $f 2>/dev/null && fs=done; grep -qiE "not usable|Exiting with failure|ERROR|failed to" $f 2>/dev/null && fs=FAIL; es=run; [ -f $e ] || es=none; grep -q "env done" $e 2>/dev/null && es=done; grep -qiE "error:|Traceback" $e 2>/dev/null && es=FAIL; echo "$fs/$es"' 2>/dev/null)
    [ -z "$s" ] && s="unreachable"
    cur="$cur ${t}=${s}"
    if [ "$t" = an.1 ]; then case "$s" in done/*) ;; *) alldone=0 ;; esac; else [ "$s" = "done/done" ] || alldone=0; fi
  done
  if [ "$cur" != "$prev" ]; then echo "$(date -u +%H:%MZ)$cur"; prev="$cur"; fi
  [ $alldone = 1 ] && { echo "ALL READY"; exit 0; }
  case "$cur" in *FAIL*) echo "FAILURE DETECTED"; exit 1 ;; esac
  sleep 60
done
echo "WATCH TIMEOUT (28 min); re-arm"
