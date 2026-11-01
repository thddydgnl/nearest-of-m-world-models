"""Evaluate a trained run: mode coverage, precision, diversity, best-of-m, action fidelity, horizon.

usage: python -m imlewm.evaluate --run imle_k4 [--nfe 20]
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from .common import ds_dir
from .probe import load_probe
from .wm import Stats, load_model, predict, rms, rollout

RUNS = os.path.expanduser("~/imle-wm/runs")


def ctx_tokens(test, idx, st):
    fi = (idx[:, None] + np.arange(-2, 1)[None]).reshape(-1)
    return st.norm(torch.from_numpy(test[fi]).cuda()).view(len(idx), 3, 256, 384)


def unique_branches(bpos, valid, tol=1.0):
    keep = []
    for b in np.where(valid)[0]:
        if all(np.linalg.norm(bpos[b] - bpos[c]) >= tol for c in keep):
            keep.append(b)
    return keep


def coverage_stats(samples, btok, bpos, probe, keep):
    """samples (n,N,D), btok (K,N,D) normalized, bpos (K,2). Returns dict for one context."""
    bt, bp = btok[keep], torch.as_tensor(bpos[keep], device=samples.device, dtype=torch.float32)
    Ku, n = len(keep), len(samples)
    dt = rms(samples[:, None], bt[None])  # (n,Ku)
    ps = probe(samples)
    dp = (ps[:, None] - bp[None]).norm(dim=-1)
    out = {"n_modes": Ku}
    if Ku > 1:
        sep_t = rms(bt[:, None], bt[None]) + torch.eye(Ku, device=bt.device) * 1e9
        sep_t = sep_t.min(1).values
        sep_p = (bp[:, None] - bp[None]).norm(dim=-1) + torch.eye(Ku, device=bt.device) * 1e9
        sep_p = sep_p.min(1).values
    else:
        sep_t = torch.full((1,), float("inf"), device=bt.device)
        sep_p = torch.full((1,), float("inf"), device=bt.device)
    tau_p = torch.clamp(0.45 * sep_p, max=4.0)
    jt = dt.argmin(1); okt = dt[torch.arange(n), jt] < 0.4 * sep_t[jt]
    jp = dp.argmin(1); okp = dp[torch.arange(n), jp] < tau_p[jp]
    out["cov_tok"] = len(set(jt[okt].tolist())) / Ku
    out["cov_pos"] = len(set(jp[okp].tolist())) / Ku
    out["prec_tok"] = okt.float().mean().item()
    out["prec_pos"] = okp.float().mean().item()
    out["minmatch_tok"] = dt.min(0).values.mean().item()
    out["minmatch_pos"] = dp.min(0).values.mean().item()
    if Ku > 1:
        iu = torch.triu_indices(n, n, 1)
        ds = rms(samples[iu[0]], samples[iu[1]]).mean()
        ib = torch.triu_indices(Ku, Ku, 1)
        db = rms(bt[ib[0]], bt[ib[1]]).mean()
        out["div_ratio"] = (ds / db).item()
    return out, ps


def prec_recall(real, fake, k=3):
    def radii(x):
        d = torch.cdist(x, x)
        return d.kthvalue(k + 1, dim=1).values
    rr, rf = radii(real), radii(fake)
    d = torch.cdist(fake, real)
    prec = (d <= rr[None]).any(1).float().mean().item()
    rec = (d.T <= rf[None]).any(1).float().mean().item()
    return prec, rec


def agg(rows):
    keys = set().union(*[r.keys() for r in rows])
    return {k: float(np.nanmean([r[k] for r in rows if k in r])) for k in keys}


@torch.no_grad()
def eval_bounce(model, data, nfe, n=16):
    d = ds_dir(data); st = Stats(data); probe = load_probe(data)
    test = np.load(f"{d}/test_tokens.npy", mmap_mode="r")
    ev = np.load(f"{d}/eval_idx.npz")
    idx, bpos = ev["stoch"], ev["stoch_branch_pos"]
    btok_all = np.load(f"{d}/stoch_branch_tokens.npy", mmap_mode="r")
    K = bpos.shape[1]
    rows, real_f, fake_f = [], [], []
    P8 = st.pca[:, :8]
    for i0 in range(0, len(idx), 40):
        ii = idx[i0:i0 + 40]
        ctx = ctx_tokens(test, ii, st)
        a = torch.zeros(len(ii), 2, device="cuda")
        S = predict(model, ctx, a, n, nfe, st.sigma_r)
        gt = st.norm(torch.from_numpy(test[ii + 1]).cuda())
        bt = st.norm(torch.from_numpy(btok_all[i0:i0 + 40]).cuda())
        for c in range(len(ii)):
            keep = unique_branches(bpos[i0 + c], np.ones(K, bool))
            r, _ = coverage_stats(S[c], bt[c], bpos[i0 + c], probe, keep)
            r["best_of_n"] = rms(S[c], gt[c][None]).min().item()
            r["mean_pred_err"] = rms(S[c].mean(0), gt[c]).item()
            rows.append(r)
            real_f.append((bt[c][keep] @ P8).flatten(1))
            fake_f.append((S[c][:len(keep)] @ P8).flatten(1))
    res = {"stoch": agg(rows)}
    res["stoch"]["precision_knn"], res["stoch"]["recall_knn"] = prec_recall(torch.cat(real_f), torch.cat(fake_f))
    # deterministic transitions
    det = ev["det"]; errs, perr = [], []
    meta = np.load(f"{d}/test_meta.npz")
    for i0 in range(0, len(det), 100):
        ii = det[i0:i0 + 100]
        ctx = ctx_tokens(test, ii, st)
        S = predict(model, ctx, torch.zeros(len(ii), 2, device="cuda"), 4, nfe, st.sigma_r)
        gt = st.norm(torch.from_numpy(test[ii + 1]).cuda())
        errs.append(rms(S, gt[:, None]).mean(1))
        gp = torch.from_numpy(meta["pos"][ii + 1]).float().cuda()
        perr.append((probe(S.flatten(0, 1)).view(len(ii), 4, 2) - gp[:, None]).norm(dim=-1).mean(1))
    res["det"] = {"tok_err": torch.cat(errs).mean().item(), "pos_err": torch.cat(perr).mean().item()}
    return res


@torch.no_grad()
def eval_push(model, data, nfe, n=16, n_hz=16):
    d = ds_dir(data); st = Stats(data); probe = load_probe(data)
    test = np.load(f"{d}/test_tokens.npy", mmap_mode="r")
    meta = np.load(f"{d}/test_meta.npz")
    ev = np.load(f"{d}/eval_idx.npz")
    idx = ev["step"]
    bt_all = np.load(f"{d}/branch_tokens.npy", mmap_mode="r")
    btc_all = np.load(f"{d}/branch_tokens_cf.npy", mmap_mode="r")
    rows, rows_cf, sens = [], [], []
    for i0 in range(0, len(idx), 40):
        ii = idx[i0:i0 + 40]; B = len(ii)
        ctx = ctx_tokens(test, ii, st)
        a = torch.from_numpy(meta["act"][ii]).cuda()
        acf = torch.from_numpy(ev["act_cf"][i0:i0 + 40]).cuda()
        # shared noise across a / a' for sensitivity
        z = torch.randn(B * n, model.zdim, device="cuda") if model.method in ("imle", "es") else None
        x0 = torch.randn(B * n, 256, 384, device="cuda") if model.method in ("fm", "mf", "imf", "sc") else None
        S = predict(model, ctx, a, n, nfe, st.sigma_r, z=z, x0=x0)
        Sc = predict(model, ctx, acf, n, nfe, st.sigma_r, z=z, x0=x0)
        bt = st.norm(torch.from_numpy(bt_all[i0:i0 + 40]).cuda())
        btc = st.norm(torch.from_numpy(btc_all[i0:i0 + 40]).cuda())
        for c in range(B):
            g = i0 + c
            keep = unique_branches(ev["branch_pos"][g], ev["branch_valid"][g])
            r, ps = coverage_stats(S[c], bt[c], ev["branch_pos"][g], probe, keep)
            rows.append(r)
            keepc = unique_branches(ev["branch_pos_cf"][g], ev["branch_valid_cf"][g])
            rc, psc = coverage_stats(Sc[c], btc[c], ev["branch_pos_cf"][g], probe, keepc)
            rows_cf.append(rc)
            # sensitivity: model change / true change (aligned mode index when counts match)
            va, vc = np.where(ev["branch_valid"][g])[0], np.where(ev["branch_valid_cf"][g])[0]
            if len(va) == len(vc):
                pairs = [(k, k) for k in va]
            else:
                pairs = [(k, l) for k in va for l in vc]
            gt_tok = np.mean([rms(bt[c][k], btc[c][l]).item() for k, l in pairs])
            gt_pos = np.mean([np.linalg.norm(ev["branch_pos"][g][k] - ev["branch_pos_cf"][g][l]) for k, l in pairs])
            if gt_pos > 2.0:
                sens.append({"sens_tok": rms(S[c], Sc[c]).mean().item() / gt_tok,
                             "sens_pos": (ps - psc).norm(dim=-1).mean().item() / gt_pos})
    res = {"step": agg(rows), "step_cf": agg(rows_cf),
           "sensitivity": {k: float(np.median([s[k] for s in sens])) for k in sens[0]}}
    # horizon
    hz = ev["horizon"]; hmax = int(ev["hmax"])
    bo_tok, bo_pos, mean_pos = [], [], []
    for i0 in range(0, len(hz), 25):
        ii = hz[i0:i0 + 25]
        ctx = ctx_tokens(test, ii, st)
        acts = torch.from_numpy(meta["act"][(ii[:, None] + np.arange(hmax)[None])]).cuda()
        R = rollout(model, ctx, acts, n_hz, nfe, st.sigma_r)  # (B,n,H,N,D)
        gfi = (ii[:, None] + np.arange(1, hmax + 1)[None]).reshape(-1)
        gt = st.norm(torch.from_numpy(test[gfi]).cuda()).view(len(ii), hmax, 256, 384)
        gp = torch.from_numpy(meta["pos"][gfi]).float().cuda().view(len(ii), hmax, 2)
        e = rms(R, gt[:, None])  # (B,n,H)
        bo_tok.append(e.min(1).values)
        pp = probe(R.flatten(0, 2)).view(len(ii), n_hz, hmax, 2)
        pe = (pp - gp[:, None]).norm(dim=-1)
        bo_pos.append(pe.min(1).values); mean_pos.append(pe.mean(1))
    res["horizon"] = {"best_tok": torch.cat(bo_tok).mean(0).tolist(),
                      "best_pos": torch.cat(bo_pos).mean(0).tolist(),
                      "mean_pos": torch.cat(mean_pos).mean(0).tolist()}
    return res


@torch.no_grad()
def latency(model, nfe, st, reps=50):
    ctx = torch.randn(1, 3, 256, 384, device="cuda"); a = torch.zeros(1, 2, device="cuda")
    for _ in range(5):
        predict(model, ctx, a, 1, nfe, st.sigma_r)
    torch.cuda.synchronize(); t = time.time()
    for _ in range(reps):
        predict(model, ctx, a, 1, nfe, st.sigma_r)
    torch.cuda.synchronize()
    return (time.time() - t) / reps * 1000


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--nfe", type=int, default=20)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--ckpt", default="ckpt", help="checkpoint file stem (ckpt_<step> for learning curves)")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    model, ck = load_model(f"{RUNS}/{a.run}/{a.ckpt}.pt")
    data = ck["args"]["data"]
    res = {"run": a.run, "nfe": a.nfe if model.method in ("fm", "mf", "imf", "sc") else 1, "args": ck["args"],
           "params": ck["params"], "train_time": ck["train_time"]}
    res["latency_ms"] = latency(model, a.nfe, Stats(data))
    res.update(eval_bounce(model, data, a.nfe) if data.startswith("bounce") else eval_push(model, data, a.nfe))
    tag = f"_nfe{a.nfe}" if model.method in ("fm", "mf", "imf", "sc") else ""
    if a.ckpt != "ckpt":
        tag = f"_{a.ckpt}{tag}"; res["step"] = ck.get("step")
    with open(f"{RUNS}/{a.run}/eval{tag}.json", "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "args"}, indent=1))
