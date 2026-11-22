#!/bin/bash
# x1.1 follow-up (revision r1, experiment 1 checks): once x1more.sh has evaluated the 12 runs (x1_done), run C1
# (ceiling and floor of the monitor on both geometries), C2 (tolerance-free index at 200 x 16 and 400 x 64) and the
# arc part of analysis a (token-space distortion), one at a time. Commands from main/revision/an_analyses/COMMANDS.md.
source /workspace/imle-wm/env.sh
cd /workspace/imle-wm/code
O=/workspace/nom/out; L=/workspace/nom/logs
for i in $(seq 1 600); do [ -f $O/x1_done ] && break; sleep 60; done
echo "$(date -u +%H:%M:%SZ) x1_done seen"
RUNS="r1_arc90_m2_s0 r1_arc90_m2_s1 r1_arc90_m4_s0 r1_arc90_m4_s1 r1_arc90_m8_s0 r1_arc90_m8_s1 r1_k4_m2_s0 r1_k4_m2_s1 r1_k4_m4_s0 r1_k4_m4_s1 r1_k4_m8_s0 r1_k4_m8_s1"
python -m imlewm.r1_calib --data bounce_div_k4_arc90 bounce_div_k4 --out $O/c1 > $L/c1.log 2>&1
echo "$(date -u +%H:%M:%SZ) C1 rc=$?"
python -m imlewm.r1_memo --runs $RUNS --n_ctx 200 --n 16 --out $O/c2 > $L/c2_200x16.log 2>&1
echo "$(date -u +%H:%M:%SZ) C2 200x16 rc=$?"
python -m imlewm.r1_memo --runs $RUNS --n_ctx 400 --n 64 --out $O/c2 > $L/c2_400x64.log 2>&1
echo "$(date -u +%H:%M:%SZ) C2 400x64 rc=$?"
python -m imlewm.r1_itok tok --data bounce_div_k4_arc90 --scale_runs r1_arc90_m2_s0 r1_arc90_m2_s1 r1_arc90_m4_s0 r1_arc90_m4_s1 r1_arc90_m8_s0 r1_arc90_m8_s1 --out $O/a_itok > $L/itok_arc90.log 2>&1
echo "$(date -u +%H:%M:%SZ) itok arc90 rc=$?"
python -m imlewm.r1_itok tok --data bounce_div_k4 --scale_runs r1_k4_m2_s0 r1_k4_m2_s1 r1_k4_m4_s0 r1_k4_m4_s1 r1_k4_m8_s0 r1_k4_m8_s1 --out $O/a_itok > $L/itok_k4.log 2>&1
echo "$(date -u +%H:%M:%SZ) itok k4 rc=$?"
touch $O/x1_post_done; echo "$(date -u +%H:%M:%SZ) x1 post done"
