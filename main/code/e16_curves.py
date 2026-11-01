"""E16 exploratory: per-horizon e3 curves of the N1b ablation (2-seed means) and the ground-truth dispersion of
the A1 futures, on the same 128 evaluation clips e3 draws (sequences(..., seed=0)). Run on the server:
ssh kiis-mvf-gpu 'source ~/imle-wm/env.sh; python -' < e16_curves.py"""
import json
import os

import numpy as np

from imlewm.common import ds_dir

R = os.path.expanduser("~/imle-wm/runs")
groups = {"N0": "e11_a1_n0_s{}", "matched": "e11_a1_seq4_s{}", "random": "e16_a1_seqrand_s{}", "gt": "e16_a1_seqgt_s{}"}
H = [1, 2, 4, 8, 12, 16]
for k, pat in groups.items():
    rs = [json.load(open(f"{R}/{pat.format(s)}/e3.json")) for s in range(2)]
    print(k, "n_seq", rs[0]["n_seq"], "hmax", rs[0]["hmax"])
    for key in ["manifold", "best_err", "spread"]:
        v = np.mean([r[key] for r in rs], 0)
        print("  ", key, [round(float(v[h - 1]), 2) for h in H])
    v = np.mean([r["spread"] for r in rs], 0) * 16 / 15  # e3 averages over all n x n pairs, diagonal included
    print("   spread off-diagonal", [round(float(v[h - 1]), 2) for h in H])

c = np.load(f"{ds_dir('bshapes_v3')}/evaluation_clips.npz")
print({f: c[f].shape for f in c.files})
fr, ic = c["frames"], c["ic"]
cen = np.stack([c["sq_pos"], c["ci_pos"]], 2)[:, 16:].mean(2)  # (clips, 16, 2): centroid, as e3's spread
per_ic = {}
for i in np.unique(ic):
    x = cen[ic == i]
    F = len(x)
    per_ic[i] = np.linalg.norm(x[:, None] - x[None], axis=-1).sum((0, 1)) / (F * (F - 1))  # off-diagonal mean
Fs = [int((ic == i).sum()) for i in np.unique(ic)]
print("n_ic", len(per_ic), "futures per ic", min(Fs), max(Fs))
allv = np.mean(list(per_ic.values()), 0)
sel = np.random.default_rng(0).choice(len(fr), min(128, len(fr)), replace=False)
selv = np.mean([per_ic[i] for i in ic[sel]], 0)
print("GT spread (off-diagonal), all ICs ", [round(float(allv[h - 1]), 2) for h in H])
print("GT spread (off-diagonal), e3 clips", [round(float(selv[h - 1]), 2) for h in H])
