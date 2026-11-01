"""Evaluation on BAIR robot pushing (real video, no mode labels).

For n_seq held-out trajectories: context = frames 0..2, n rollouts of H=16 steps under the recorded
actions. Per horizon h (token RMS in the normalized DINO space):
  best_rms : best-of-n distance to the real future (does the sample set cover what happened)
  mean_rms : mean over samples
  manifold : nearest-neighbour distance (PCA-64) to a bank of real test frames / real-real NN median
  spread   : mean pairwise distance between the n samples (0 for a deterministic model)
Distribution metrics at h in {4, 8, 16}, on mean-pooled tokens projected to the top-32 PCs of the
real pooled features: Frechet distance (fd) and k-NN precision / recall (k=5, Kynkaanniemi et al.)
between all generated frames and the real frames at that horizon from all test trajectories.

usage: python -m imlewm.bair_eval --run bair_imle16_s0 [--nfe 2]
"""
import argparse
import json
import os

import numpy as np
import torch

from .common import ds_dir
from .wm import K_CTX, Stats, load_model, rollout

RUNS = os.path.expanduser("~/imle-wm/runs")
HS = [4, 8, 16]


def frechet(a, b):
    from scipy import linalg
    ma, mb = a.mean(0), b.mean(0)
    ca, cb = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    cs, _ = linalg.sqrtm(ca @ cb, disp=False)
    return float(((ma - mb) ** 2).sum() + np.trace(ca + cb - 2 * cs.real))


def knn_pr(real, fake, k=5):
    """Improved precision / recall (Kynkaanniemi et al. 2019) on feature tensors."""
    def radii(x):
        d = torch.cdist(x, x)
        return d.kthvalue(k + 1, dim=1).values
    rr, rf = radii(real), radii(fake)
    d = torch.cdist(fake, real)
    prec = (d <= rr[None]).any(1).float().mean().item()
    rec = (d.T <= rf[None]).any(1).float().mean().item()
    return prec, rec


@torch.no_grad()
def run(model, nfe, n=16, n_seq=256, H=16, seed=123):
    torch.manual_seed(seed)
    d = ds_dir("bair"); st = Stats("bair")
    tok = np.load(f"{d}/test_tokens.npy", mmap_mode="r")
    meta = np.load(f"{d}/test_meta.npz")
    seg, act = meta["seg"], meta["act"]
    first = np.r_[0, np.where(np.diff(seg) != 0)[0] + 1]  # first frame of each trajectory
    rng = np.random.default_rng(0)
    sel = np.sort(rng.choice(first, min(n_seq, len(first)), replace=False)) + K_CTX - 1  # last context frame
    P = st.pca[:, :64]
    proj = lambda x: (x @ P).flatten(1)
    bank_i = np.sort(rng.choice(len(tok), min(6000, len(tok)), replace=False))
    bank = torch.cat([proj(st.norm(torch.from_numpy(np.asarray(tok[bank_i[i:i + 1000]])).cuda()))
                      for i in range(0, len(bank_i), 1000)])
    dd = torch.cdist(bank[:2000], bank); dd[torch.arange(2000), torch.arange(2000)] = 1e9
    real_nn = dd.min(1).values.median()
    # real pooled features at each evaluated horizon, from all test trajectories
    pool = lambda x: x.mean(1)  # (B,N,D) -> (B,D)
    real_pool = {h: pool(st.norm(torch.from_numpy(np.asarray(tok[first + K_CTX - 1 + h])).cuda())) for h in HS}
    allr = torch.cat(list(real_pool.values()))
    mu = allr.mean(0)
    _, _, V = torch.linalg.svd(allr - mu, full_matrices=False)
    pc = lambda x: (x - mu) @ V[:32].T
    best, mean, man, spread = [], [], [], []
    gen_pool = {h: [] for h in HS}
    cs = 4
    for i0 in range(0, len(sel), cs):
        s = sel[i0:i0 + cs]; S = len(s)
        ci = (s[:, None] + np.arange(-K_CTX + 1, 1)[None]).reshape(-1)
        ctx = st.norm(torch.from_numpy(np.asarray(tok[ci])).cuda()).view(S, K_CTX, 256, 384)
        acts = torch.from_numpy(act[s[:, None] + np.arange(H)[None]]).cuda()
        gt = st.norm(torch.from_numpy(np.asarray(tok[(s[:, None] + np.arange(1, H + 1)[None]).reshape(-1)])).cuda()).view(S, 1, H, 256, 384)
        R = rollout(model, ctx, acts, n, nfe, st.sigma_r)  # (S,n,H,N,D)
        err = (R - gt).pow(2).mean((-1, -2)).sqrt()  # (S,n,H)
        best.append(err.min(1).values.cpu()); mean.append(err.mean(1).cpu())
        flat = R.flatten(0, 2)
        m = torch.cat([torch.cdist(proj(flat[j:j + 1024]), bank).min(1).values for j in range(0, len(flat), 1024)])
        man.append((m / real_nn).view(S, n, H).mean(1).cpu())
        Rf = R.permute(0, 2, 1, 3, 4).reshape(S * H, n, -1)
        pw = torch.cdist(Rf, Rf) / Rf.shape[-1] ** 0.5  # (S*H,n,n)
        spread.append((pw.sum((1, 2)) / (n * (n - 1))).view(S, H).cpu())
        for h in HS:
            gen_pool[h].append(pool(R[:, :, h - 1].flatten(0, 1)))
    f = lambda xs: torch.cat(xs).mean(0).tolist()
    res = {"best_rms": f(best), "mean_rms": f(mean), "manifold": f(man), "spread": f(spread),
           "H": H, "n": n, "n_seq": len(sel), "n_real": len(first)}
    for h in HS:
        g, r = pc(torch.cat(gen_pool[h])), pc(real_pool[h])
        res[f"fd_h{h}"] = frechet(g.double().cpu().numpy(), r.double().cpu().numpy())
        gi = torch.randperm(len(g), generator=torch.Generator().manual_seed(0))[:len(r)].to(g.device)
        res[f"prec_h{h}"], res[f"rec_h{h}"] = knn_pr(r, g[gi])
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--nfe", type=int, default=2)
    ap.add_argument("--n_seq", type=int, default=256)
    a = ap.parse_args()
    model, ck = load_model(f"{RUNS}/{a.run}/ckpt.pt")
    res = run(model, a.nfe, n_seq=a.n_seq)
    gen = model.method in ("fm", "mf", "imf", "sc")
    res.update(run=a.run, nfe=a.nfe if gen else 1, params=ck["params"], train_time=ck["train_time"], args=ck["args"])
    tag = f"_nfe{a.nfe}" if gen else ""
    json.dump(res, open(f"{RUNS}/{a.run}/eval{tag}.json", "w"), indent=1)
    ks = [1, 4, 8, 16]
    print(a.run, {k: [round(res[k][i - 1], 3) for i in ks] for k in ["best_rms", "mean_rms", "manifold", "spread"]},
          {k: round(v, 3) for k, v in res.items() if k.startswith(("fd_", "prec_", "rec_"))})
