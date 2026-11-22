"""Revision r1, analysis f (DESIGN_r1.md section 4f): diagnostics whose outputs were never saved (review D-10).

novelty  : check.novelty for the given datasets (test-to-train over train-to-train nearest-neighbour distance,
           PCA-64 of the last context frame; paper: 1.9x for A2-div, 0.35x for A2-fix)
gtspread : true-future spread of Table 4 from pusht_b/mf_test_v2.npz, in the units of the FlowWM errors
           (mean over steps 1..h and the 8 T vertices of the L2 distance, env px), as the mean over pairs of
           distinct futures and as the mean over all n^2 pairs (paper: 30.8 at h=8, 48.3 at h=16)
script   : run one of the original diagnostic scripts (bstep.py, bdiag.py, logs/bair_diag.py; their
           /home/yonghwi paths fixed on the pod copy) and keep stdout plus the parsed numbers

usage: python -m imlewm.r1_diag novelty --data bounce_div_k4 bounce_k4 --out /workspace/nom/out/f_diag
       python -m imlewm.r1_diag gtspread --out /workspace/nom/out/f_diag
       python -m imlewm.r1_diag script --script /workspace/imle-wm/bstep.py --args e1_b_reg_s0 ... --name contact --out ...
"""
import argparse
import os
import re
import subprocess
import sys
import time

import numpy as np

from . import r1_util
from .common import ds_dir


def novelty_cmd(datasets, out):
    import torch
    from .check import novelty
    for dname in datasets:
        p = f"{out}/novelty_{dname}.json"
        if os.path.exists(p):
            print("skip", p); continue
        torch.manual_seed(0)
        r = novelty(dname)
        r.update(data=dname, n_train=20000, n_test=400, pca_dim=64, seed=0)
        r1_util.write_json(r, p)
        print("novelty", r, flush=True)


def gtspread_cmd(out, data="pusht_b", fname="mf_test_v2.npz"):
    p = f"{out}/gtspread_{data}.json"
    if os.path.exists(p):
        print("skip", p); return
    import torch
    from .pusht_sim import tee_keypoints
    f = np.load(f"{ds_dir(data)}/{fname}")
    res = {"data": data, "file": fname, "n_starts": int(len(f["fut_states"])), "h": {}}
    for h in (1, 4, 8, 16):
        off, full, cen = [], [], []
        for i in range(len(f["fut_states"])):
            kp = torch.from_numpy(tee_keypoints(f["fut_states"][i][:, :h])).double()  # (64,h,8,2)
            n = kp.shape[0]
            e = (kp[:, None] - kp[None]).norm(dim=-1).mean((2, 3))  # (64,64) FlowWM trajectory distance
            full.append(float(e.mean())); off.append(float(e.sum() / (n * (n - 1))))
            c = kp[:, -1].mean(-2)  # block keypoint centroid at step h
            cen.append(float((c - c.mean(0, keepdim=True)).norm(dim=-1).mean()))  # bdiag.py's GT spread
        res["h"][h] = {"pairwise_distinct": float(np.mean(off)), "pairwise_all_n2": float(np.mean(full)),
                       "centroid_to_mean_last_step": float(np.mean(cen))}
    r1_util.write_json(res, p)
    print(res, flush=True)


def script_cmd(script, args, name, out):
    p = f"{out}/{name}.json"
    if os.path.exists(p):
        print("skip", p); return
    t0 = time.time()
    r = subprocess.run([sys.executable, script] + list(args), capture_output=True, text=True,
                       env=dict(os.environ, PYTHONPATH=os.path.expanduser("~/imle-wm/code")))
    txt = r.stdout
    with open(f"{out}/{name}.stdout.txt", "w") as fh:
        fh.write(txt + ("\n[stderr]\n" + r.stderr[-4000:] if r.returncode else ""))
    nums = [[float(x) for x in re.findall(r"-?\d+\.\d+", line)] for line in txt.splitlines()]
    r1_util.write_json({"script": script, "args": list(args), "returncode": r.returncode, "stdout": txt.splitlines(),
                        "numbers_per_line": nums, "wall_s": round(time.time() - t0, 1)}, p)
    print(txt, r.stderr[-2000:] if r.returncode else "", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["novelty", "gtspread", "script"])
    ap.add_argument("--data", nargs="*", default=[])
    ap.add_argument("--script")
    ap.add_argument("--args", nargs="*", default=[])
    ap.add_argument("--name")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.cmd == "novelty":
        novelty_cmd(a.data, a.out)
    elif a.cmd == "gtspread":
        gtspread_cmd(a.out)
    else:
        script_cmd(a.script, a.args, a.name, a.out)
