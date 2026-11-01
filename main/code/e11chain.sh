#!/bin/bash
# usage: e11chain.sh <gpu>; follow-up evaluations (2026-10-15), each skipped if its output exists:
#  1) B per-start re-evaluation (paired bootstrap): E1 IMLE16 / FM-50 / FM-2, E7 Shortcut, E6 N1b vs N0
#  2) E3 rollout stability with FM at NFE 50 (A1, A2, B)
#  3) E3 for the E11 sequential-matching runs and their N0 controls (waits for the checkpoints)
source ~/imle-wm/env.sh
cd $IMLE/code
R=$IMLE/runs; L=$IMLE/logs
export CUDA_VISIBLE_DEVICES=$1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ps() {  # run nfe
  local tag=""; [ -n "$2" ] && tag="_nfe$2"
  [ -f $R/$1/eval_ps$tag.json ] || python -m imlewm.b_eval --run $1 --perstart ${2:+--nfe $2} >> $L/e11_perstart.log 2>&1
}
for s in 0 1 2; do
  ps e1_b_imle16_s$s; ps e1_b_fm_s$s 2; ps e7_b_sc_s$s 1; ps e6_b_seq4_s$s
done
ps e6_b_n0_8k; ps e6_b_n0_8k_s1; ps e6_b_n0_8k_s2
for s in 0 1 2; do ps e1_b_fm_s$s 50; done
e3() {  # run hmax nfe
  local tag=""; [ "$3" != "1" ] && tag="_nfe$3"
  [ -f $R/$1/e3$tag.json ] || python -m imlewm.e3 --run $1 --hmax $2 --nfe $3 --cs 8 >> $L/e11_e3.log 2>&1
}
for s in 0 1 2; do e3 e1_a2_fm_s$s 32 50; e3 e1_a1_fm_s$s 16 50; e3 e1_b_fm_s$s 16 50; done
for pair in e2_k4_m16_s0:32 e2_k4_m16_s1:32 e11_a2_seq4_s0:32 e11_a2_seq4_s1:32 e11_a1_n0_s0:16 e11_a1_n0_s1:16 e11_a1_seq4_s0:16 e11_a1_seq4_s1:16; do
  r=${pair%:*}; h=${pair#*:}
  until [ -f $R/$r/ckpt.pt ]; do sleep 300; done
  e3 $r $h 1
done
echo "$(date +%T) e11chain done" >> $L/queue.log
