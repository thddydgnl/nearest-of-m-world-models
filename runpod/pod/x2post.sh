#!/bin/bash
# x2.1 follow-up (revision r1, experiment 2a and analysis g): when the three FM 36k runs are trained and evaluated at
# NFE 50 and 2 and the seed-0 rollout chain has finished, stop the auto-evaluator, measure the memorization index of
# the final checkpoints at NFE 50 and 2 (section 4c procedure, plus the 12k and 24k checkpoints at NFE 50), evaluate
# the 12k and 24k checkpoints at NFE 50 (learning curve), then run the timing analysis g on the otherwise idle A5000.
source /workspace/imle-wm/env.sh
cd /workspace/imle-wm/code
R=/workspace/imle-wm/runs; O=/workspace/nom/out; L=/workspace/nom/logs
ok() { for s in 0 1 2; do for n in 50 2; do [ -f $R/r1_fm36k_s$s/eval_nfe$n.json ] || return 1; done; done; return 0; }
for i in $(seq 1 480); do
  ok && ! ps -eo args | grep -q "[r]unq.sh" && ! ps -eo args | grep -q "[e]3chain.sh" && break
  sleep 60
done
echo "$(date -u +%H:%M:%SZ) FM 36k trained and evaluated, e3 chain finished"
for p in $(ps -eo pid,args | grep "[e]valwatch.sh" | awk '{print $1}'); do kill $p; done
sleep 5
RUNS="r1_fm36k_s0 r1_fm36k_s1 r1_fm36k_s2"
for f in 50 2; do
  python -m imlewm.r1_memo --runs $RUNS --nfe $f --n_ctx 200 --n 16 --out $O/fm36k_memo > $L/fm36k_memo_nfe$f.log 2>&1
  echo "$(date -u +%H:%M:%SZ) fm36k memo nfe $f rc=$?"
done
for c in ckpt_12000 ckpt_24000; do
  python -m imlewm.r1_memo --runs $RUNS --ckpt $c --nfe 50 --n_ctx 200 --n 16 --out $O/fm36k_memo > $L/fm36k_memo_$c.log 2>&1
  echo "$(date -u +%H:%M:%SZ) fm36k memo $c rc=$?"
  for s in 0 1 2; do
    [ -f $R/r1_fm36k_s$s/eval_${c}_nfe50.json ] || python -m imlewm.evaluate --run r1_fm36k_s$s --ckpt $c --nfe 50 > $L/eval_r1_fm36k_s${s}_$c.log 2>&1
    echo "$(date -u +%H:%M:%SZ) eval r1_fm36k_s$s $c rc=$?"
  done
done
python -m imlewm.r1_timing run --out $O/g_timing > $L/g_timing.log 2>&1
echo "$(date -u +%H:%M:%SZ) timing rc=$?"
touch $O/x2_done; echo "$(date -u +%H:%M:%SZ) x2 post done"
