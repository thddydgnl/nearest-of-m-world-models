#!/bin/bash
# Analyses b, a, c, d, C1 test, h, f, riskdiag on an.1, in order. Each step skips outputs that already exist.
source ~/imle-wm/env.sh
cd ~/imle-wm/code
O=/workspace/nom/out
log() { echo "$(date -u +%H:%M:%SZ) $*"; }
GRID=""
for K in 4 8; do for m in 4 8 16 32; do for s in 0 1; do GRID="$GRID e2_k${K}_m${m}_s$s"; done; done; done
for m in 2 4 8 16; do for s in 0 1; do GRID="$GRID e12_k2_m${m}_s$s"; done; done
for m in 4 8 16 32 64; do for s in 0 1; do GRID="$GRID e12_k16_m${m}_s$s"; done; done

# 1. validation of r1_memo against check.memo_model (monitor settings), gate for everything that uses r1_memo
log "validate"
python -m imlewm.r1_memo --runs e2_k4_m4_s0 --n_ctx 200 --n 16 --out $O/b_grid --validate
python - <<'EOF' || { log "VALIDATION FAILED: stop"; exit 1; }
import json, sys
r = json.load(open("/workspace/nom/out/b_grid/validate_e2_k4_m4_s0_nfeNone_200x16.json"))
print("identical", r["identical"], r["max_abs_diff"]); sys.exit(0 if r["identical"] else 1)
EOF

# 2. analysis b: 34 grid checkpoints, 400 + 400 contexts x 64 samples
log "b grid"
python -m imlewm.r1_memo --runs $GRID --n_ctx 400 --n 64 --out $O/b_grid

# 3. analysis a: token-space distortion with Geman-McClure scales from the grid checkpoints, then the fit
log "a itok"
for K in 2 4 8 16; do
  R=$(echo $GRID | tr ' ' '\n' | grep "_k${K}_" | tr '\n' ' ')
  python -m imlewm.r1_itok tok --data bounce_div_k$K --scale_runs $R --out $O/a_itok
done
python -m imlewm.r1_itok fit --out $O/a_itok --b_dir $O/b_grid

# 4. analysis c: memorization index at each model's own NFE (monitor settings and 400 x 64)
log "c nfe"
for s in 0 1 2; do
  for f in 50 2 20; do
    python -m imlewm.r1_memo --runs e1_a2_fm_s$s --nfe $f --n_ctx 200 --n 16 --out $O/c_nfe
  done
  python -m imlewm.r1_memo --runs e1_a2_imf_s$s --nfe 1 --n_ctx 200 --n 16 --out $O/c_nfe
  python -m imlewm.r1_memo --runs e1_a2_imf_s$s --nfe 20 --n_ctx 200 --n 16 --out $O/c_nfe
  python -m imlewm.r1_memo --runs e7_a2_sc_s$s --nfe 1 --n_ctx 200 --n 16 --out $O/c_nfe
done
python -m imlewm.r1_memo --runs imf_e0_ts1 --nfe 1 --n_ctx 300 --n 16 --out $O/c_nfe
python -m imlewm.r1_memo --runs imf_e0_ts1 --nfe 20 --n_ctx 300 --n 16 --out $O/c_nfe
for s in 0 1 2; do
  python -m imlewm.r1_memo --runs e1_a2_fm_s$s --nfe 2 --n_ctx 400 --n 64 --out $O/c_nfe
  python -m imlewm.r1_memo --runs e1_a2_imf_s$s --nfe 1 --n_ctx 400 --n 64 --out $O/c_nfe
  python -m imlewm.r1_memo --runs e7_a2_sc_s$s --nfe 1 --n_ctx 400 --n 64 --out $O/c_nfe
  python -m imlewm.r1_memo --runs e1_a2_imle16_s$s e1_a2_imle4_s$s e1_a2_reg_s$s --n_ctx 200 --n 16 --out $O/c_nfe
done

# 5. analysis d: readout offset of the mode-share measurement; cross-check of r1_memo with the stored mode shares
log "d offset"
python -m imlewm.r1_offset --out $O/d_offset
python -m imlewm.r1_memo --runs e14_asym_m16_s0 e14_asym_m4_s0 e14_asym_m2_s0 --n_ctx 400 --n 64 --out $O/d_offset
python -m imlewm.r1_memo --runs e14_asym_fm_s0 --nfe 50 --n_ctx 400 --n 64 --out $O/d_offset

# 6. C1 on the standard geometry (script test; the design runs C1 on x1.1 for both geometries)
log "c1 test"
python -m imlewm.r1_calib --data bounce_div_k4 --out $O/c1_test

# 7. analysis h: qualitative samples
log "h samples"
python -m imlewm.r1_samples --out $O/h_samples

# 8. analysis f: true-future spread, contact error (bstep.py), bdiag GT spread, novelty
log "f diag"
python -m imlewm.r1_diag gtspread --out $O/f_diag
python -m imlewm.r1_diag script --script /workspace/imle-wm/bstep.py --name contact_bstep \
  --args e1_b_reg_s0 e1diag_b_reg36k e1_b_fm_s0@2 e1_b_fm_s0@50 e1_b_imle4_s0 e1_b_imle16_s0 --out $O/f_diag
python -m imlewm.r1_diag script --script /workspace/imle-wm/bdiag.py --name gtspread_bdiag --args --out $O/f_diag
until grep -q "bounce_k4 done" /workspace/nom/logs/fetch_k4.log; do sleep 30; done
python -m imlewm.r1_diag novelty --data bounce_div_k4 bounce_k4 --out $O/f_diag

# 9. analysis e, part 2: straight-push risk diagnostic with both criteria
log "e riskdiag"
M9="e13_push_imle16_s0 e13_push_imle16_s1 e13_push_imle16_s2 e13_push_fm_s0 e13_push_fm_s1 e13_push_fm_s2 e13_push_reg_s0 e13_push_reg_s1 e13_push_reg_s2"
python -m imlewm.r1_planrisk riskdiag --models $M9 --nfe 2 --out $O/e_planrisk
log "an_main done"
# optional, heavier: FM-50 risk diagnostic and FM-50 index with 400 x 64
python -m imlewm.r1_planrisk riskdiag --models e13_push_fm_s0 e13_push_fm_s1 e13_push_fm_s2 --nfe 50 --out $O/e_planrisk
for s in 0 1 2; do python -m imlewm.r1_memo --runs e1_a2_fm_s$s --nfe 50 --n_ctx 400 --n 64 --out $O/c_nfe; done
log "an_main optional done"
