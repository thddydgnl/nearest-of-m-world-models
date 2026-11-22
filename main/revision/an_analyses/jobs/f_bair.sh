#!/bin/bash
# Analysis f, BAIR part (if time permits): spread-to-error on training vs evaluation contexts with the original
# logs/bair_diag.py (96 sequences per split, h = 1, 4, 16), after the BAIR tokens are fetched.
source ~/imle-wm/env.sh
cd ~/imle-wm/code
for i in $(seq 1 120); do grep -q "bair done" /workspace/nom/logs/fetch_bair.log && break; sleep 30; done
grep -q "bair done" /workspace/nom/logs/fetch_bair.log || { echo "bair fetch not done"; exit 1; }
[ -f /workspace/nom/STOP_OPT ] && { echo "stop file"; exit 0; }
python -m imlewm.r1_diag script --script /workspace/imle-wm/logs/bair_diag.py --name bair_diag \
  --args e8_bair_imle4_s0 e8_bair_imle16_s0 e10_bair_m64_s0 --out /workspace/nom/out/f_diag
echo "f_bair done"
