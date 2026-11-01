"""E16 (novelty_analysis.md sec 6, item 2): an effective dimension for Prop 2 on BAIR that is not fitted.

Residuals of the regression model at h=1 (real next frame minus its prediction, normalized DINO tokens
over the whole 16x16x384 map, the space IMLE matches in). If the generator spreads like the residual,
q = N(mu, rho^2 S), the nearest-of-m objective is minimized at rho = e_m / sqrt(d) with
d = (tr S)^2 / tr S^2, the participation ratio (Prop 2 has S = sigma^2 I_d, where this is d itself).
Prop 2 then predicts the spread / error ratio r(m) = sqrt(2) rho / sqrt(1 + rho^2) with nothing fitted.
Measured r = spread / mean_rms at h=1 from each IMLE run's eval.json (bair_eval, same 256 contexts).

tr S^2 comes from pairs of distinct residuals (unbiased); the plug-in value is printed too. Also printed:
cross-validated variance fractions along the top principal directions (fit on one half, measured on
the other), for the evaluation contexts, all test transitions, and the train split.

--dims (designed after the pooled d_PR came out ~1,000, 35x the fitted 29): per evaluation context, the PR
of n IMLE samples, and the share of the real residual (real next frame minus the sample mean) inside the
top-k sample principal directions of that context (f_own) and of the previous context (f_other).

usage: python -m imlewm.bair_deff [--reg e8_bair_reg_s0 e8_bair_reg_s1]
       python -m imlewm.bair_deff --dims e8_bair_imle4_s0 e8_bair_imle16_s0 e10_bair_m64_s0
"""
import argparse
import json
import os

import numpy as np
import torch
from scipy.special import ndtr

from .common import ds_dir
from .wm import K_CTX, Stats, load_model, rollout

RUNS = os.path.expanduser("~/imle-wm/runs")
MEASURED = {4: ["e8_bair_imle4_s0", "e8_bair_imle4_s1"], 16: ["e8_bair_imle16_s0", "e8_bair_imle16_s1"],
            64: ["e10_bair_m64_s0"]}


def e_max(m):
    """E max of m iid N(0, 1), by quadrature."""
    x = np.linspace(-12, 12, 240001)
    phi = np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi)
    return float(np.sum(x * m * phi * ndtr(x) ** (m - 1)) * (x[1] - x[0]))


def predict_r(d, m):
    rho = e_max(m) / np.sqrt(d)
    return float(np.sqrt(2) * rho / np.sqrt(1 + rho ** 2))


def measured(runs):
    v = [json.load(open(f"{RUNS}/{r}/eval.json")) for r in runs]
    v = [e["spread"][0] / e["mean_rms"][0] for e in v]
    return {"mean": float(np.mean(v)), "seeds": v}


def contexts(split, which, n_max):
    """Last context frame indices: 'eval' = 256 trajectory starts drawn as in bair_eval (on test these
    are its evaluation contexts), 'all' = random transitions with a full context in one trajectory."""
    seg = np.load(f"{ds_dir('bair')}/{split}_meta.npz")["seg"]
    if which == "eval":
        first = np.r_[0, np.where(np.diff(seg) != 0)[0] + 1]
        rng = np.random.default_rng(0)
        return np.sort(rng.choice(first, min(256, len(first)), replace=False)) + K_CTX - 1
    t = np.arange(K_CTX - 1, len(seg) - 1)
    t = t[seg[t - K_CTX + 1] == seg[t + 1]]
    return np.sort(np.random.default_rng(1).choice(t, min(n_max, len(t)), replace=False))


@torch.no_grad()
def residuals(model, split, idx, st, cs=64):
    d = ds_dir("bair")
    tok = np.load(f"{d}/{split}_tokens.npy", mmap_mode="r")
    act = np.load(f"{d}/{split}_meta.npz")["act"]
    out = []
    for i0 in range(0, len(idx), cs):
        s = idx[i0:i0 + cs]; S = len(s)
        ci = (s[:, None] + np.arange(-K_CTX + 1, 1)[None]).reshape(-1)
        ctx = st.norm(torch.from_numpy(np.asarray(tok[ci])).cuda()).view(S, K_CTX, 256, 384)
        a = torch.from_numpy(act[s[:, None]]).cuda()  # (S,1,A): the action leading to frame s+1
        pred = rollout(model, ctx, a, 1, 1, st.sigma_r)[:, 0, 0]  # (S,N,D)
        gt = st.norm(torch.from_numpy(np.asarray(tok[s + 1])).cuda())
        out.append((gt - pred).flatten(1).float())
    return torch.cat(out)  # (n, N*D)


@torch.no_grad()
def spectrum(R):
    R = R - R.mean(0, keepdim=True)
    n, p = R.shape
    G = (R @ R.T).double()
    tr = G.diagonal().sum() / (n - 1)  # tr S
    tr2 = (G.pow(2).sum() - G.diagonal().pow(2).sum()) / (n * (n - 1))  # tr S^2 from distinct pairs
    tr2_plug = G.pow(2).sum() / (n - 1) ** 2
    out = {"n": n, "pr": (tr ** 2 / tr2).item(), "pr_plugin": (tr ** 2 / tr2_plug).item(),
           "rms": (tr / p).sqrt().item()}
    del G
    h, fr = n // 2, []
    for A, B in ((R[:h], R[h:2 * h]), (R[h:2 * h], R[:h])):
        w, U = torch.linalg.eigh((A @ A.T).double())
        k = min(256, h - 1)
        w, U = w.flip(0)[:k], U.flip(1)[:, :k]
        V = (A.T.double() @ U) / w.clamp_min(1e-12).sqrt()  # (p,k) unit principal directions of A
        fr.append((B.double() @ V).pow(2).mean(0) / B.double().pow(2).sum(1).mean())
        del V
    f = torch.stack(fr).mean(0).cumsum(0)
    out["cv_frac_top"] = {k: f[k - 1].item() for k in (1, 2, 4, 8, 16, 32, 64, 128, 256) if k <= len(f)}
    return out


def pr_unbiased(Dv):
    """PR of the covariance of the rows of Dv (already centered), tr S^2 from distinct pairs."""
    n = Dv.shape[0]
    G = (Dv @ Dv.T).double()
    tr = G.diagonal().sum() / (n - 1)
    tr2 = (G.pow(2).sum() - G.diagonal().pow(2).sum()) / (n * (n - 1))
    return (tr ** 2 / tr2).item(), G


@torch.no_grad()
def sample_dims(model, st, n=64, k=16, cs=4):
    d = ds_dir("bair")
    tok = np.load(f"{d}/test_tokens.npy", mmap_mode="r")
    act = np.load(f"{d}/test_meta.npz")["act"]
    idx = contexts("test", "eval", 0)
    prs, f_own, f_other, pooled, U_prev = [], [], [], [], None
    for i0 in range(0, len(idx), cs):
        s = idx[i0:i0 + cs]; S = len(s)
        ci = (s[:, None] + np.arange(-K_CTX + 1, 1)[None]).reshape(-1)
        ctx = st.norm(torch.from_numpy(np.asarray(tok[ci])).cuda()).view(S, K_CTX, 256, 384)
        a = torch.from_numpy(act[s[:, None]]).cuda()
        Y = rollout(model, ctx, a, n, 1, st.sigma_r)[:, :, 0].flatten(2).float()  # (S,n,p)
        gt = st.norm(torch.from_numpy(np.asarray(tok[s + 1])).cuda()).flatten(1)
        for c in range(S):
            mu = Y[c].mean(0, keepdim=True)
            Dv = Y[c] - mu
            pr, G = pr_unbiased(Dv)
            prs.append(pr)
            w, V = torch.linalg.eigh(G)
            U = (Dv.T.double() @ V.flip(1)[:, :k]) / w.flip(0)[:k].clamp_min(1e-12).sqrt()  # (p,k)
            e = (gt[c] - mu[0]).double()
            f_own.append(((U.T @ e).pow(2).sum() / e.pow(2).sum()).item())
            if U_prev is not None:
                f_other.append(((U_prev.T @ e).pow(2).sum() / e.pow(2).sum()).item())
            U_prev = U
            pooled.append(Dv[:8].clone())  # 8 deviations per context, pooled (a view would keep all n alive)
    q = lambda v: [float(np.percentile(v, p)) for p in (25, 50, 75)]
    return {"n": n, "k": k, "n_ctx": len(prs), "pr_ctx_q": q(prs), "f_own_q": q(f_own), "f_other_q": q(f_other),
            "pr_pooled_within": pr_unbiased(torch.cat(pooled))[0]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reg", nargs="+", default=["e8_bair_reg_s0", "e8_bair_reg_s1"])
    ap.add_argument("--n_all", type=int, default=4096)
    ap.add_argument("--dims", nargs="+", default=None, help="IMLE runs for the per-context sample dimension")
    a = ap.parse_args()
    st = Stats("bair")
    if a.dims:
        res = {}
        for run in a.dims:
            model, ck = load_model(f"{RUNS}/{run}/ckpt.pt")
            m = ck["args"]["m"]
            s = sample_dims(model, st)
            s["m"] = m
            s["pred_r_from_ctx_pr"] = predict_r(s["pr_ctx_q"][1], m)
            res[run] = s
            print(run, json.dumps(s), flush=True)
            del model; torch.cuda.empty_cache()
        json.dump(res, open(f"{RUNS}/e16_bair_dims.json", "w"), indent=1)
        raise SystemExit
    res = {"e_m": {m: e_max(m) for m in MEASURED}, "measured_r": {m: measured(r) for m, r in MEASURED.items()}}
    print("e_m", res["e_m"], "measured r", {m: round(v["mean"], 3) for m, v in res["measured_r"].items()}, flush=True)
    for run in a.reg:
        model, ck = load_model(f"{RUNS}/{run}/ckpt.pt")
        for split, which in (("test", "eval"), ("test", "all"), ("train", "eval"), ("train", "all")):
            R = residuals(model, split, contexts(split, which, a.n_all), st)
            s = spectrum(R)
            del R; torch.cuda.empty_cache()
            s["pred_r"] = {m: predict_r(s["pr"], m) for m in MEASURED}
            res[f"{run}/{split}_{which}"] = s
            print(run, split, which, json.dumps(s), flush=True)
    json.dump(res, open(f"{RUNS}/e16_bair_deff.json", "w"), indent=1)
