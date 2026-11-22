#!/bin/bash
# Continuation of an_main.sh after its grid step (b), with the cheap steps first and the heavy FM-50 items last.
# usage: an_main2.sh <pid of the running b process>
source ~/imle-wm/env.sh
cd ~/imle-wm/code
O=/workspace/nom/out
log() { echo "$(date -u +%H:%M:%SZ) $*"; }
while kill -0 $1 2>/dev/null; do sleep 20; done
log "b finished"
GRID=""
for K in 4 8; do for m in 4 8 16 32; do for s in 0 1; do GRID="$GRID e2_k${K}_m${m}_s$s"; done; done; done
for m in 2 4 8 16; do for s in 0 1; do GRID="$GRID e12_k2_m${m}_s$s"; done; done
for m in 4 8 16 32 64; do for s in 0 1; do GRID="$GRID e12_k16_m${m}_s$s"; done; done
# b again: completes any grid checkpoint the first pass missed (existing outputs are skipped)
python -m imlewm.r1_memo --runs $GRID --n_ctx 400 --n 64 --out $O/b_grid

log "a itok"
for K in 2 4 8 16; do
  R=$(echo $GRID | tr ' ' '\n' | grep "_k${K}_" | tr '\n' ' ')
  python -m imlewm.r1_itok tok --data bounce_div_k$K --scale_runs $R --out $O/a_itok
done
python -m imlewm.r1_itok fit --out $O/a_itok --b_dir $O/b_grid

log "d offset"
python -m imlewm.r1_offset --out $O/d_offset
python -m imlewm.r1_memo --runs e14_asym_m16_s0 e14_asym_m4_s0 e14_asym_m2_s0 --n_ctx 400 --n 64 --out $O/d_offset

log "h samples"
python -m imlewm.r1_samples --out $O/h_samples

log "c nfe (monitor settings)"
for s in 0 1 2; do
  for f in 2 50 20; do python -m imlewm.r1_memo --runs e1_a2_fm_s$s --nfe $f --n_ctx 200 --n 16 --out $O/c_nfe; done
  for f in 1 20; do python -m imlewm.r1_memo --runs e1_a2_imf_s$s --nfe $f --n_ctx 200 --n 16 --out $O/c_nfe; done
  python -m imlewm.r1_memo --runs e7_a2_sc_s$s --nfe 1 --n_ctx 200 --n 16 --out $O/c_nfe
  python -m imlewm.r1_memo --runs e1_a2_imle16_s$s e1_a2_imle4_s$s e1_a2_reg_s$s --n_ctx 200 --n 16 --out $O/c_nfe
done
for f in 1 20; do python -m imlewm.r1_memo --runs imf_e0_ts1 --nfe $f --n_ctx 300 --n 16 --out $O/c_nfe; done

log "c1 test"
python -m imlewm.r1_calib --data bounce_div_k4 --out $O/c1_test

log "f diag"
python -m imlewm.r1_diag gtspread --out $O/f_diag
python -m imlewm.r1_diag script --script /workspace/imle-wm/bstep.py --name contact_bstep \
  --args e1_b_reg_s0 e1diag_b_reg36k e1_b_fm_s0@2 e1_b_fm_s0@50 e1_b_imle4_s0 e1_b_imle16_s0 --out $O/f_diag
python -m imlewm.r1_diag script --script /workspace/imle-wm/bdiag.py --name gtspread_bdiag --args --out $O/f_diag
for i in $(seq 1 60); do grep -q "bounce_k4 done" /workspace/nom/logs/fetch_k4.log && break; sleep 30; done
python -m imlewm.r1_diag novelty --data bounce_div_k4 bounce_k4 --out $O/f_diag

log "e riskdiag"
M9="e13_push_imle16_s0 e13_push_imle16_s1 e13_push_imle16_s2 e13_push_fm_s0 e13_push_fm_s1 e13_push_fm_s2 e13_push_reg_s0 e13_push_reg_s1 e13_push_reg_s2"
python -m imlewm.r1_planrisk riskdiag --models $M9 --nfe 2 --out $O/e_planrisk

log "c 400x64 (FM-2, iMF, Shortcut)"
for s in 0 1 2; do
  python -m imlewm.r1_memo --runs e1_a2_fm_s$s --nfe 2 --n_ctx 400 --n 64 --out $O/c_nfe
  python -m imlewm.r1_memo --runs e1_a2_imf_s$s --nfe 1 --n_ctx 400 --n 64 --out $O/c_nfe
  python -m imlewm.r1_memo --runs e7_a2_sc_s$s --nfe 1 --n_ctx 400 --n 64 --out $O/c_nfe
done
log "an_main2 core done"
[ -f /workspace/nom/STOP_OPT ] && { log "stop file: skip optional"; exit 0; }
python -m imlewm.r1_planrisk riskdiag --models e13_push_fm_s0 e13_push_fm_s1 e13_push_fm_s2 --nfe 50 --out $O/e_planrisk
log "an_main2 optional done"
