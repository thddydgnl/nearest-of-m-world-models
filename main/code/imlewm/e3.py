"""E3 (C3): rollout stability vs horizon.

For held-out sequences, draw n rollouts of up to Hmax steps from K_CTX real context frames under
the recorded actions and report per horizon h:
  manifold : distance of each predicted token map to its nearest real token map (PCA-64 bank of
             real test frames), divided by the median real-to-real nearest-neighbour distance.
             ~1 means on the data manifold; growth means the rollout drifts off it.
  best_err : best-of-n readout error to the realized ground truth (px; A2 ball, A1 mean of the
             two shapes, B T-vertex L2)
  spread   : mean pairwise distance of the n readouts (px), to see collapse vs dispersion.
This replaces the decoder round-trip error of the plan with a decoder-free manifold distance.

usage: python -m imlewm.e3 --run e1_a2_imle16_s0 [--nfe 2]
"""
import argparse
import json
import os

import numpy as np
import torch

from .common import ds_dir
from .wm import K_CTX, Stats, load_model, rollout

RUNS = os.path.expanduser("~/imle-wm/runs")


def sequences(data, hmax, n_seq, seed=0):
    """Returns (ctx frame indices (S,K), future frame indices (S,H), action rows or None)."""
    d = ds_dir(data)
    rng = np.random.default_rng(seed)
    if os.path.exists(f"{d}/evaluation_clips.npz"):  # A1: contexts 13..15, futures 16..31
        fr = np.load(f"{d}/evaluation_clips.npz")["frames"]
        sel = rng.choice(len(fr), min(n_seq, len(fr)), replace=False)
        return fr[sel, 16 - K_CTX:16], fr[sel, 16:16 + hmax], None, "evaluation"
    seg = np.load(f"{d}/test_meta.npz")["seg"]
    starts = [t for t in range(K_CTX - 1, len(seg) - hmax) if seg[t - K_CTX + 1] == seg[t + hmax]]
    sel = np.sort(rng.choice(starts, min(n_seq, len(starts)), replace=False))
    ctx = sel[:, None] + np.arange(-K_CTX + 1, 1)[None]
    fut = sel[:, None] + np.arange(1, hmax + 1)[None]
    return ctx, fut, sel, "test"


def readout(data):
    """Returns f(tokens (B,N,D) normalized) -> positions (B, P, 2) in px, and gt from meta."""
    d = ds_dir(data)
    if data.startswith("bshapes"):
        os.environ["A1_DATA"] = data
        from . import a1eval
        a1eval.DATA = data
        p = a1eval.load_probe2()
        return lambda x: p(x)  # (B,2 shapes,2)
    if data.startswith("pusht"):
        from .b_eval import load_probeB
        from .pusht_sim import tee_keypoints
        p = load_probeB(data)
        return lambda x: torch.from_numpy(tee_keypoints(p(x).cpu().numpy())).to(x.device)  # (B,8,2)
    from .probe import load_probe
    p = load_probe(data)
    return lambda x: p(x)[:, None]  # (B,1,2)


def gt_positions(data, fut, split):
    d = ds_dir(data)
    if data.startswith("bshapes"):
        from .a1eval import frame_positions
        return torch.from_numpy(frame_positions(split)[fut])  # (S,H,2,2)
    pos = np.load(f"{d}/{split}_meta.npz")["pos"]
    if data.startswith("pusht"):
        from .pusht_sim import tee_keypoints
        return torch.from_numpy(tee_keypoints(pos[fut]))  # (S,H,8,2)
    return torch.from_numpy(pos[fut]).float()[:, :, None]  # (S,H,1,2)


@torch.no_grad()
def run(model, data, hmax, n=16, n_seq=128, nfe=2, seed=123, cs=4):
    torch.manual_seed(seed)
    d = ds_dir(data); st = Stats(data)
    ctx_i, fut_i, sel, split = sequences(data, hmax, n_seq)
    hmax = fut_i.shape[1]
    tok = np.load(f"{d}/{split}_tokens.npy", mmap_mode="r")
    acts_all = None if sel is None else np.load(f"{d}/{split}_meta.npz")["act"]
    P = st.pca[:, :64]
    proj = lambda x: (x @ P).flatten(1)
    rng = np.random.default_rng(0)
    bank_i = np.sort(rng.choice(len(tok), min(6000, len(tok)), replace=False))
    bank = torch.cat([proj(st.norm(torch.from_numpy(np.asarray(tok[bank_i[i:i + 1000]])).cuda()))
                      for i in range(0, len(bank_i), 1000)])
    dd = torch.cdist(bank[:2000], bank); dd[torch.arange(2000), torch.arange(2000)] = 1e9
    real_nn = dd.min(1).values.median()
    read = readout(data)
    gt = gt_positions(data, fut_i, split)
    man, best, spread = [], [], []
    # cs: sequences per chunk; n x H x 256 x 384 floats per sequence (~200 MB at H=32)
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
        pr = torch.cat([read(flat[j:j + 512]) for j in range(0, len(flat), 512)]).view(S, n, hmax, -1, 2).cpu()
        g = gt[i0:i0 + cs].float()[:, None]
        err = (pr - g).norm(dim=-1).mean(-1)  # (S,n,H)
        best.append(err.min(1).values)
        c = pr.mean(-2)  # (S,n,H,2)
        spread.append((c[:, :, None] - c[:, None]).norm(dim=-1).mean((1, 2)))
    f = lambda xs: torch.cat(xs).mean(0).tolist()
    # real_nn is ~0 when test frames repeat exactly (bounce_k4: fixed appearance), so manifold x real_nn
    # (the raw PCA-64 distance) is what compares across datasets
    return {"manifold": f(man), "best_err": f(best), "spread": f(spread), "hmax": hmax, "n_seq": len(ctx_i),
            "real_nn": real_nn.item()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--nfe", type=int, default=2)
    ap.add_argument("--hmax", type=int, default=32)
    ap.add_argument("--cs", type=int, default=4, help="sequences per chunk (larger = faster, more memory)")
    a = ap.parse_args()
    model, ck = load_model(f"{RUNS}/{a.run}/ckpt.pt")
    data = ck["args"]["data"]
    res = run(model, data, a.hmax, nfe=a.nfe, cs=a.cs)
    res.update(run=a.run, nfe=a.nfe if model.method in ("fm", "mf", "imf", "sc") else 1)
    tag = f"_nfe{a.nfe}" if model.method in ("fm", "mf", "imf", "sc") else ""
    json.dump(res, open(f"{RUNS}/{a.run}/e3{tag}.json", "w"), indent=1)
    h = res["hmax"]; ks = [k for k in [1, 4, 8, 16, 32] if k <= h]
    print(a.run, {k: [round(res[k][i - 1], 2) for i in ks] for k in ["manifold", "best_err", "spread"]}, "h=", ks)
