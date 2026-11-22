#!/bin/bash
# Revision r1, post-hoc check A-26 (not in the frozen design): rollout manifold distance with a training-split
# reference bank, seed-0 A2-div models of Table 2 (fast models first, FM-50 last). Writes /workspace/nom/out/e3bank/.
source /workspace/imle-wm/env.sh
cd /workspace/imle-wm/code
O=/workspace/nom/out/e3bank; L=/workspace/nom/logs/e3bank.log
python -m imlewm.r1_e3bank --runs e1_a2_reg_s0 e1_a2_imle4_s0 e1_a2_imle16_s0 e15_a2_es8_s0 --cs 8 --out $O >> $L 2>&1
python -m imlewm.r1_e3bank --runs e1_a2_imf_s0 --nfe 1 --cs 8 --out $O >> $L 2>&1
python -m imlewm.r1_e3bank --runs e1_a2_fm_s0 --nfe 2 --cs 8 --out $O >> $L 2>&1
python -m imlewm.r1_e3bank --runs e1_a2_fm_s0 --nfe 50 --cs 8 --out $O >> $L 2>&1
echo "$(date -u +%H:%M:%SZ) e3bank done" >> $L; touch $O/done
