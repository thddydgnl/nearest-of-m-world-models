"""Revision r1, post-hoc check of review finding A-26: rollout manifold distance with a reference bank drawn from
the TRAINING split instead of the test split.

e3.py measures the distance from each predicted token map to its nearest real map in a bank of up to 6,000 frames of
the test split, which contains the true future frames of the evaluated rollouts; at short horizons an accurate
prediction can then sit closer to its own ground-truth frame than real frames sit to each other (values below 1).
This module repeats e3.run with the bank (and the real-to-real normalization) taken from the training split, so the
true futures are never in the bank. Everything else (sequences, seeds, PCA-64 projection, readout, chunking) is the
same as e3.py, which is not modified.

usage: python -m imlewm.r1_e3bank --runs e1_a2_imle16_s0 e1_a2_fm_s0 --nfe 50 --out /workspace/nom/out/e3bank
       (--nfe applies to fm/imf/sc runs; one-step methods ignore it, as in e3.py)
"""
import argparse
import json
import os

import numpy as np
import torch

from .common import ds_dir
from .e3 import gt_positions, readout, sequences
from .wm import K_CTX, Stats, load_model, rollout

RUNS = os.path.expanduser("~/imle-wm/runs")


@torch.no_grad()
def run(model, data, hmax, n=16, n_seq=128, nfe=2, seed=123, cs=4, bank_split="train"):
    torch.manual_seed(seed)
    d = ds_dir(data); st = Stats(data)
    ctx_i, fut_i, sel, split = sequences(data, hmax, n_seq)
    hmax = fut_i.shape[1]
    tok = np.load(f"{d}/{split}_tokens.npy", mmap_mode="r")
    btok = np.load(f"{d}/{bank_split}_tokens.npy", mmap_mode="r")
    acts_all = None if sel is None else np.load(f"{d}/{split}_meta.npz")["act"]
    P = st.pca[:, :64]
    proj = lambda x: (x @ P).flatten(1)
    rng = np.random.default_rng(0)
    bank_i = np.sort(rng.choice(len(btok), min(6000, len(btok)), replace=False))
    bank = torch.cat([proj(st.norm(torch.from_numpy(np.asarray(btok[bank_i[i:i + 1000]])).cuda()))
                      for i in range(0, len(bank_i), 1000)])
    dd = torch.cdist(bank[:2000], bank); dd[torch.arange(2000), torch.arange(2000)] = 1e9
    real_nn = dd.min(1).values.median()
    read = readout(data)
    gt = gt_positions(data, fut_i, split)
    man, best, spread, truth = [], [], [], []
    for i0 in range(0, len(ctx_i), cs):
        ci = ctx_i[i0:i0 + cs]; S = len(ci)
        ctx = st.norm(torch.from_numpy(np.asarray(tok[ci.reshape(-1)])).cuda()).view(S, K_CTX, 256, 384)
        if acts_all is None:
            acts = torch.zeros(S, hmax, 2, device="cuda")
        else:
            acts = torch.from_numpy(acts_all[sel[i0:i0 + cs, None] + np.arange(hmax)[None]]).cuda()
        R = rollout(model, ctx, acts, n, nfe, st.sigma_r)  # (S,n,H,N,D)
        flat = R.flatten(0, 2)
        m = torch.cat([torch.cdist(proj(flat[j:j + 1024]), bank).min(1).values for j in range(0, len(flat), 1024)])
        man.append((m / real_nn).view(S, n, hmax).mean(1).cpu())
        # the true future frames themselves, measured against the same training bank (reference level)
        fi = fut_i[i0:i0 + cs].reshape(-1)
        tf = st.norm(torch.from_numpy(np.asarray(tok[fi])).cuda())
        truth.append((torch.cdist(proj(tf), bank).min(1).values / real_nn).view(S, hmax).cpu())
        pr = torch.cat([read(flat[j:j + 512]) for j in range(0, len(flat), 512)]).view(S, n, hmax, -1, 2).cpu()
        g = gt[i0:i0 + cs].float()[:, None]
        err = (pr - g).norm(dim=-1).mean(-1)
        best.append(err.min(1).values)
        c = pr.mean(-2)
        spread.append((c[:, :, None] - c[:, None]).norm(dim=-1).mean((1, 2)))
    f = lambda xs: torch.cat(xs).mean(0).tolist()
    return {"manifold": f(man), "manifold_true_futures": f(truth), "best_err": f(best), "spread": f(spread),
            "hmax": hmax, "n_seq": len(ctx_i), "real_nn": real_nn.item(), "bank_split": bank_split}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--nfe", type=int, default=2)
    ap.add_argument("--hmax", type=int, default=32)
    ap.add_argument("--cs", type=int, default=4)
    ap.add_argument("--bank_split", default="train")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for r in a.runs:
        model, ck = load_model(f"{RUNS}/{r}/ckpt.pt")
        nfe = a.nfe if model.method in ("fm", "mf", "imf", "sc") else 1
        path = f"{a.out}/{r}_nfe{nfe}_{a.bank_split}bank.json"
        if os.path.exists(path):
            print("skip", path); continue
        res = run(model, ck["args"]["data"], a.hmax, nfe=a.nfe, cs=a.cs, bank_split=a.bank_split)
        res.update(run=r, nfe=nfe)
        tmp = path + ".tmp"; json.dump(res, open(tmp, "w"), indent=1); os.replace(tmp, path)
        ks = [k for k in (1, 4, 8, 16, 32) if k <= res["hmax"]]
        print(r, "nfe", nfe, {k: [round(res[k][i - 1], 2) for i in ks] for k in ("manifold", "manifold_true_futures")}, "h=", ks,
              flush=True)
