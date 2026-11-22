#!/bin/bash
# Side job (more GPU share for the analyses while the planner replays run): d, h, C1 test and f now; an_main2.sh
# skips whatever exists when it gets there. usage: an_side.sh
source ~/imle-wm/env.sh
cd ~/imle-wm/code
O=/workspace/nom/out
log() { echo "$(date -u +%H:%M:%SZ) $*"; }
log "d offset"
python -m imlewm.r1_offset --out $O/d_offset
log "h samples"
python -m imlewm.r1_samples --out $O/h_samples
log "c1 test"
python -m imlewm.r1_calib --data bounce_div_k4 --out $O/c1_test
log "f diag"
python -m imlewm.r1_diag gtspread --out $O/f_diag
python -m imlewm.r1_diag novelty --data bounce_div_k4 bounce_k4 --out $O/f_diag
python -m imlewm.r1_diag script --script /workspace/imle-wm/bdiag.py --name gtspread_bdiag --args --out $O/f_diag
python -m imlewm.r1_diag script --script /workspace/imle-wm/bstep.py --name contact_bstep \
  --args e1_b_reg_s0 e1diag_b_reg36k e1_b_fm_s0@2 e1_b_fm_s0@50 e1_b_imle4_s0 e1_b_imle16_s0 --out $O/f_diag
log "an_side done"
